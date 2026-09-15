"""Download, check and update the SMART-G auxiliary data.

The auxiliary datasets (aerosol and cloud optical properties,
atmospheric profiles, absorption cross sections, k-distributions,
REPTRAN tables, validation and IPRT data, ...) are hosted as
Nextcloud shares on docs.hygeos.com, except REPTRAN whose reference
is the libRadtran archive with a HYGEOS mirror as fallback.

Every download is recorded in a manifest (``.smartg_auxdata.json``)
inside the auxdata directory: the source used, its version
fingerprint (the WebDAV ETag of the share or the HTTP ETag of the
archive) and the download date. ``check_update`` compares the
manifest with the remote versions without downloading anything and
``update`` refreshes only the missing or outdated datasets, replacing
each directory in one rename so that an interrupted run never leaves
a half-updated dataset.

Key Classes
-----------
AuxData
    The auxiliary data directory: download, status, update.
Dataset
    One dataset: key, extracted directory and ordered sources.
NextcloudSource
    A folder shared on a Nextcloud instance, downloaded as a ZIP.
HttpArchiveSource
    A ZIP or gzipped TAR archive at a plain HTTP(S) URL.
Manifest
    The JSON record of the installed datasets.
DatasetStatus
    The comparison of one installed dataset with its remote source.
AuxDataDownloadError
    Raised at the end of a run when some datasets could not be
    installed.

Key Functions
-------------
download
    Download the datasets that are not yet on disk.
check_update
    Print and return the status of the datasets against the remote.
update
    Download the missing, outdated or unrecorded datasets.
extract_zip, extract_tar
    Safely extract an archive, optionally keeping one subfolder.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from email.message import Message
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from tqdm.auto import tqdm

from smartg.typing import PathType

NEXTCLOUD_BASE = "https://docs.hygeos.com"

# REPTRAN reference: https://www.libradtran.org/doku.php?id=download
LIBRADTRAN_REPTRAN_URL = (
    "https://www.libradtran.org/lib/exe/fetch.php"
    "?media=download:reptran_2024_all.tar.gz"
)
# HYGEOS mirror of the REPTRAN tables, used when libRadtran is down.
REPTRAN_MIRROR_TOKEN = "jHKMcZZmkf6xy7D"

MANIFEST_NAME = ".smartg_auxdata.json"
MANIFEST_SCHEMA_VERSION = 1

DEFAULT_TIMEOUT = 30.0  # seconds, per socket operation
RETRY_DELAYS = (1.0, 3.0)  # seconds between the attempts
CHUNK_SIZE = 1 << 20  # bytes read per iteration when streaming

# Single entry point of every network access, replaced in the tests.
_urlopen = urllib.request.urlopen

Status = Literal[
    "missing",
    "up to date",
    "outdated",
    "unknown",
    "unversioned",
    "unreachable",
]
VersionKind = Literal["etag", "last-modified", "unknown"]
DataSelector = str | Sequence[str]

# HTTP status codes worth another attempt.
_RETRY_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

# Errors of a download or an extraction that are reported per dataset
# instead of aborting the run. URLError, ConnectionError and
# TimeoutError are subclasses of OSError.
_FETCH_ERRORS = (
    OSError,
    ValueError,
    http.client.HTTPException,
    tarfile.TarError,
    zipfile.BadZipFile,
)

_PROPFIND_BODY = (
    b'<?xml version="1.0" encoding="utf-8"?>'
    b'<d:propfind xmlns:d="DAV:"><d:prop>'
    b"<d:getetag/><d:getlastmodified/>"
    b"</d:prop></d:propfind>"
)


class AuxDataDownloadError(RuntimeError):
    """Some datasets could not be installed.

    Attributes
    ----------
    failures : dict[str, str]
        The reason of the failure, by dataset key. The datasets that
        succeeded in the same run are installed and recorded.
    """

    def __init__(self, failures: Mapping[str, str]) -> None:
        self.failures = dict(failures)
        lines = [f"  {key}: {reason}" for key, reason in failures.items()]
        super().__init__(
            f"could not install {', '.join(failures)}:\n" + "\n".join(lines)
        )


# ---------------------------------------------------------------------
# Versions and manifest
# ---------------------------------------------------------------------


def _utc(moment: datetime | None) -> datetime | None:
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _parse_http_date(text: str | None) -> datetime | None:
    """Parse an RFC 1123 date, None when absent or unreadable."""
    if not text:
        return None
    try:
        return _utc(parsedate_to_datetime(text))
    except (TypeError, ValueError):
        return None


def _iso(moment: datetime | None) -> str | None:
    utc = _utc(moment)
    return None if utc is None else utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_iso() -> str:
    return _iso(datetime.now(timezone.utc)) or ""


def _parse_iso(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return _utc(datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return None


def _clean_etag(etag: str) -> str:
    """Strip the weak marker and the quotes of an HTTP ETag."""
    etag = etag.strip()
    if etag.startswith(("W/", "w/")):
        etag = etag[2:]
    return etag.strip('"')


@dataclass(frozen=True)
class RemoteVersion:
    """The version of a dataset as exposed by its remote source.

    Attributes
    ----------
    fingerprint : str
        The ETag when the source gives one, else the Last-Modified
        string, else ``"unknown"``. Two fingerprints of the same
        source are equal when the dataset did not change.
    kind : {"etag", "last-modified", "unknown"}
        What the fingerprint is.
    modified : datetime or None
        The remote modification time in UTC, when known.
    """

    fingerprint: str
    kind: VersionKind
    modified: datetime | None = None

    @classmethod
    def from_headers(
        cls, etag: str | None, last_modified: str | None
    ) -> RemoteVersion:
        modified = _parse_http_date(last_modified)
        if etag and etag.strip():
            return cls(_clean_etag(etag), "etag", modified)
        if last_modified and last_modified.strip():
            return cls(last_modified.strip(), "last-modified", modified)
        return cls("unknown", "unknown", None)


def _version_from_headers(headers: Message) -> RemoteVersion:
    return RemoteVersion.from_headers(
        headers.get("ETag"), headers.get("Last-Modified")
    )


def _parse_propfind(xml: bytes) -> RemoteVersion:
    """Read the ETag and modification time of a PROPFIND reply."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as err:
        raise ValueError(f"unreadable PROPFIND reply: {err}") from err
    dav = "{DAV:}"
    response = root.find(f"{dav}response")
    if response is None:
        raise ValueError("PROPFIND reply without any response element")
    return RemoteVersion.from_headers(
        response.findtext(f".//{dav}getetag"),
        response.findtext(f".//{dav}getlastmodified"),
    )


@dataclass(frozen=True)
class ManifestEntry:
    """The record of one installed dataset.

    Attributes
    ----------
    dirname : str
        The directory of the dataset inside the auxdata directory.
    source : str
        The name of the source it was downloaded from.
    url : str
        The download URL of that source.
    version : str
        The remote fingerprint at download time.
    version_kind : {"etag", "last-modified", "unknown"}
        What the fingerprint is.
    remote_modified : str or None
        The remote modification time, ISO 8601 UTC.
    downloaded_at : str
        The download time, ISO 8601 UTC.
    """

    dirname: str
    source: str
    url: str
    version: str
    version_kind: VersionKind
    remote_modified: str | None
    downloaded_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManifestEntry:
        return cls(**{f.name: data[f.name] for f in fields(cls)})


class Manifest:
    """The JSON record of the installed datasets.

    Parameters
    ----------
    path : str or path-like
        The manifest file, normally ``MANIFEST_NAME`` inside the
        auxdata directory.
    entries : dict[str, ManifestEntry], optional
        The records by dataset key.
    """

    def __init__(
        self,
        path: PathType,
        entries: Mapping[str, ManifestEntry] | None = None,
    ) -> None:
        self.path = Path(path)
        self.entries: dict[str, ManifestEntry] = dict(entries or {})

    @classmethod
    def load(cls, path: PathType) -> Manifest:
        """Read a manifest; a missing or unreadable file gives an
        empty one (unreadable files are reported)."""
        path = Path(path)
        if not path.is_file():
            return cls(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            version = data.get("schema_version")
            if version != MANIFEST_SCHEMA_VERSION:
                raise ValueError(f"unsupported schema_version {version!r}")
            entries = {
                key: ManifestEntry.from_dict(value)
                for key, value in data["datasets"].items()
            }
        except (ValueError, KeyError, TypeError, AttributeError) as err:
            print(f"Warning: ignoring the unreadable manifest {path} ({err})")
            return cls(path)
        return cls(path, entries)

    def save(self) -> None:
        """Write the manifest, replacing the file atomically."""
        data = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "datasets": {
                key: entry.to_dict() for key, entry in self.entries.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, key: str) -> ManifestEntry | None:
        return self.entries.get(key)

    def set(self, key: str, entry: ManifestEntry) -> None:
        self.entries[key] = entry

    def remove(self, key: str) -> None:
        self.entries.pop(key, None)


# ---------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------


def _open(
    request: urllib.request.Request, timeout: float
) -> http.client.HTTPResponse:
    """Open a request, retrying the transient failures.

    Server errors 408/425/429/5xx and connection errors are retried
    after the delays of ``RETRY_DELAYS``; the other HTTP errors (401,
    403, 404, ...) are raised at once.
    """
    attempts = len(RETRY_DELAYS) + 1
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            return _urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as err:
            if err.code not in _RETRY_CODES or last:
                raise
            error: Exception = err
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
        ) as err:
            if last:
                raise ConnectionError(
                    f"{request.full_url}: failed after {attempts} attempts"
                    f" ({err})"
                ) from err
            error = err
        delay = RETRY_DELAYS[attempt]
        print(f"  {request.full_url}: {error}, retrying in {delay:g} s")
        time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def _content_length(headers: Message) -> int | None:
    try:
        return int(headers["Content-Length"])
    except (KeyError, TypeError, ValueError):
        return None


def _download_to_file(
    request: urllib.request.Request,
    path: Path,
    *,
    timeout: float,
    progress: bool,
    label: str,
) -> Message:
    """Stream a response into a file and return its headers.

    A transfer interrupted midway is restarted from scratch (the
    Nextcloud archives are generated on the fly, ranges are useless).
    """
    attempts = len(RETRY_DELAYS) + 1
    for attempt in range(attempts):
        response = _open(request, timeout)
        try:
            with response:
                headers = response.headers
                total = _content_length(headers)
                with (
                    open(path, "wb") as out,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=label,
                        leave=False,
                        disable=not progress,
                    ) as bar,
                ):
                    while chunk := response.read(CHUNK_SIZE):
                        out.write(chunk)
                        bar.update(len(chunk))
            return headers
        except (
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
        ) as err:
            if attempt == attempts - 1:
                raise ConnectionError(
                    f"{request.full_url}: transfer failed after"
                    f" {attempts} attempts ({err})"
                ) from err
            delay = RETRY_DELAYS[attempt]
            print(f"  {label}: transfer interrupted ({err}),"
                  f" restarting in {delay:g} s")
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------
# Archives
# ---------------------------------------------------------------------

_DRIVE = re.compile(r"^[A-Za-z]:")


def _relocate(name: str, target_folder: str | None) -> str | None:
    """The path a member is extracted to, None when it is skipped.

    With a ``target_folder``, only the members inside a folder of
    that name are kept, and their path starts at that folder.
    """
    parts = PurePosixPath(name).parts
    if target_folder is not None:
        if target_folder not in parts:
            return None
        parts = parts[parts.index(target_folder) :]
    return "/".join(parts)


def _check_member_name(name: str) -> None:
    """Refuse the member paths that could escape the destination."""
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or "\\" in name
        or ".." in path.parts
        or _DRIVE.match(name)
    ):
        raise ValueError(f"unsafe archive member name {name!r}")


def extract_zip(
    zfile: PathType,
    dest: PathType,
    target_folder: str | None = None,
    progress: bool = True,
) -> None:
    """Extract a ZIP archive.

    Parameters
    ----------
    zfile : str or path-like
        The archive.
    dest : str or path-like
        The directory to extract into.
    target_folder : str, optional
        Keep only the members inside a folder of this name, with
        their path rewritten to start at that folder.
    progress : bool, optional
        Show a progress bar over the members.

    Raises
    ------
    ValueError
        A member path is absolute or escapes ``dest``. Nothing is
        written in that case.
    """
    dest = Path(dest)
    with zipfile.ZipFile(zfile) as archive:
        members: list[tuple[zipfile.ZipInfo, str]] = []
        for info in archive.infolist():
            name = _relocate(info.filename, target_folder)
            if name is None:
                continue
            _check_member_name(name)
            members.append((info, name))
        print(f"Extracting {len(members)} entries of {Path(zfile).name}")
        bar = tqdm(members, unit="file", leave=False, disable=not progress)
        for info, name in bar:
            target = dest / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def extract_tar(
    tfile: PathType,
    dest: PathType,
    target_folder: str | None = None,
    progress: bool = True,
) -> None:
    """Extract a gzipped TAR archive.

    Parameters
    ----------
    tfile : str or path-like
        The archive.
    dest : str or path-like
        The directory to extract into.
    target_folder : str, optional
        Keep only the members inside a folder of this name, with
        their path rewritten to start at that folder. This is how the
        ``reptran`` folder is taken out of the libRadtran archive.
    progress : bool, optional
        Show a progress bar over the members.

    Raises
    ------
    ValueError
        A member is absolute, escapes ``dest``, or is refused by the
        ``tarfile`` data filter (links outside the archive, devices).
    """
    dest = Path(dest)
    with tarfile.open(tfile, "r:gz") as archive:
        members: list[tarfile.TarInfo] = []
        for member in archive.getmembers():
            name = _relocate(member.name, target_folder)
            if name is None:
                continue
            _check_member_name(name)
            members.append(member.replace(name=name))
        print(f"Extracting {len(members)} entries of {Path(tfile).name}")
        bar = tqdm(
            total=len(members), unit="file", leave=False, disable=not progress
        )

        def counted(
            member: tarfile.TarInfo, path: str
        ) -> tarfile.TarInfo | None:
            bar.update(1)
            return tarfile.data_filter(member, path)

        try:
            archive.extractall(dest, members=members, filter=counted)
        except tarfile.FilterError as err:
            raise ValueError(f"archive member refused: {err}") from err
        finally:
            bar.close()


def _extract_archive(
    archive: Path,
    dest: Path,
    target_folder: str | None = None,
    progress: bool = True,
) -> None:
    """Extract a ZIP or gzipped TAR, whatever its name says."""
    with open(archive, "rb") as f:
        magic = f.read(4)
    if magic.startswith(b"PK\x03\x04"):
        extract_zip(archive, dest, target_folder, progress)
    elif magic.startswith(b"\x1f\x8b"):
        extract_tar(archive, dest, target_folder, progress)
    else:
        raise ValueError(
            f"{archive.name} is neither a ZIP nor a gzip archive (starts"
            f" with {magic!r}); the server probably answered with an"
            " error page"
        )


# ---------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------


class Source(Protocol):
    """Where a dataset can be fetched from."""

    @property
    def name(self) -> str:
        """The name recorded in the manifest."""
        ...

    @property
    def url(self) -> str:
        """The download URL."""
        ...

    def remote_version(
        self, timeout: float = DEFAULT_TIMEOUT
    ) -> RemoteVersion:
        """Query the remote version without downloading the data."""
        ...

    def fetch(
        self,
        dest: Path,
        *,
        label: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        progress: bool = True,
    ) -> RemoteVersion:
        """Download and extract the dataset into ``dest``."""
        ...


@dataclass(frozen=True)
class NextcloudSource:
    """A folder shared on a Nextcloud instance.

    The share is downloaded as a ZIP generated on the fly whose top
    level is the shared folder. Its version is the ETag of the share
    root, read through the public WebDAV endpoint (Nextcloud
    propagates the ETag of any changed file up to the root).

    Parameters
    ----------
    token : str
        The share token, the last part of the share URL.
    base_url : str, optional
        The Nextcloud instance.
    name : str, optional
        The source name recorded in the manifest.
    """

    token: str
    base_url: str = NEXTCLOUD_BASE
    name: str = "hygeos"

    @property
    def url(self) -> str:
        return f"{self.base_url}/s/{self.token}/download"

    @property
    def webdav_url(self) -> str:
        return f"{self.base_url}/public.php/webdav/"

    def _propfind(self) -> urllib.request.Request:
        auth = base64.b64encode(f"{self.token}:".encode()).decode("ascii")
        return urllib.request.Request(
            self.webdav_url,
            data=_PROPFIND_BODY,
            method="PROPFIND",
            headers={
                "Depth": "0",
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/xml; charset=utf-8",
            },
        )

    def remote_version(
        self, timeout: float = DEFAULT_TIMEOUT
    ) -> RemoteVersion:
        with _open(self._propfind(), timeout) as response:
            return _parse_propfind(response.read())

    def fetch(
        self,
        dest: Path,
        *,
        label: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        progress: bool = True,
    ) -> RemoteVersion:
        # The version is read before the transfer: the archive itself
        # carries no useful header (it is generated on the fly).
        version = self.remote_version(timeout)
        archive = dest / f"{label or 'archive'}.part"
        _download_to_file(
            urllib.request.Request(self.url),
            archive,
            timeout=timeout,
            progress=progress,
            label=label,
        )
        _extract_archive(archive, dest, progress=progress)
        archive.unlink()
        return version


@dataclass(frozen=True)
class HttpArchiveSource:
    """A ZIP or gzipped TAR archive at an HTTP(S) URL.

    Its version comes from the ETag, else the Last-Modified header,
    of the archive.

    Parameters
    ----------
    url : str
        The archive URL.
    member_root : str, optional
        Keep only the members inside a folder of this name, with
        their path rewritten to start at that folder.
    name : str, optional
        The source name recorded in the manifest.
    """

    url: str
    member_root: str | None = None
    name: str = "http"

    def remote_version(
        self, timeout: float = DEFAULT_TIMEOUT
    ) -> RemoteVersion:
        try:
            request = urllib.request.Request(self.url, method="HEAD")
            with _open(request, timeout) as response:
                return _version_from_headers(response.headers)
        except urllib.error.HTTPError as err:
            if err.code not in (405, 501):
                raise
        # HEAD is not supported: open a GET and close it at once.
        with _open(urllib.request.Request(self.url), timeout) as response:
            return _version_from_headers(response.headers)

    def fetch(
        self,
        dest: Path,
        *,
        label: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        progress: bool = True,
    ) -> RemoteVersion:
        archive = dest / f"{label or 'archive'}.part"
        headers = _download_to_file(
            urllib.request.Request(self.url),
            archive,
            timeout=timeout,
            progress=progress,
            label=label,
        )
        _extract_archive(archive, dest, self.member_root, progress)
        archive.unlink()
        return _version_from_headers(headers)


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """One auxiliary dataset.

    Attributes
    ----------
    key : str
        The short name used to select it (``"aer"``, ``"kdis"``, ...).
    dirname : str
        The directory its archive extracts to, inside the auxdata
        directory.
    sources : tuple[Source, ...]
        Where to fetch it from, in order of preference.
    description : str
        What it contains.
    """

    key: str
    dirname: str
    sources: tuple[Source, ...]
    description: str

    @property
    def primary(self) -> Source:
        return self.sources[0]

    def source_named(self, name: str) -> Source | None:
        for source in self.sources:
            if source.name == name:
                return source
        return None


def _hygeos(key: str, dirname: str, token: str, description: str) -> Dataset:
    return Dataset(key, dirname, (NextcloudSource(token),), description)


DATASETS: dict[str, Dataset] = {
    dataset.key: dataset
    for dataset in (
        _hygeos(
            "aer",
            "aerosols",
            "8PnKXFXQbmYyTte",
            "aerosol optical properties (OPAC mixtures and components)",
        ),
        _hygeos(
            "acs",
            "acs",
            "HwotAHPstdCCKcJ",
            "molecular absorption cross sections",
        ),
        _hygeos(
            "atm", "atmospheres", "z6MRf9g66WmWeBA", "atmospheric profiles"
        ),
        _hygeos(
            "STP",
            "STPs",
            "NW42DNPtKw3NNW7",
            "solar power tower files with the heliostat positions",
        ),
        _hygeos(
            "valid", "validation", "6EPBqwebn94NYPq", "validation files"
        ),
        _hygeos(
            "water",
            "water",
            "3NKP5tMsHKnNRpt",
            "water files needed by some simulations, including the ocean",
        ),
        _hygeos("kdis", "kdis", "CHTFFgHe6to39CR", "k-distributions"),
        _hygeos(
            "cld", "clouds", "agDWDy998j64SHf", "cloud optical properties"
        ),
        _hygeos(
            "IPRT",
            "IPRT",
            "i4QaxtpjSfjwtNk",
            "IPRT (International working group on Polarized Radiative"
            " Transfer) reference results and optical properties",
        ),
        Dataset(
            "reptran",
            "reptran",
            (
                HttpArchiveSource(
                    LIBRADTRAN_REPTRAN_URL,
                    member_root="reptran",
                    name="libradtran",
                ),
                NextcloudSource(REPTRAN_MIRROR_TOKEN),
            ),
            "REPTRAN absorption parameterization tables (libRadtran)",
        ),
    )
}


@dataclass(frozen=True)
class DatasetStatus:
    """The state of one dataset on disk against its remote source.

    Attributes
    ----------
    key : str
        The dataset key.
    status : str
        ``"missing"`` (not on disk), ``"up to date"``, ``"outdated"``
        (the remote changed since the download), ``"unknown"`` (on
        disk but not recorded in the manifest), ``"unversioned"``
        (the source gives no version to compare) or
        ``"unreachable"`` (the recorded source did not answer).
    local : ManifestEntry or None
        The manifest record, when there is one.
    remote : RemoteVersion or None
        The remote version, when it could be read.
    source : str or None
        The source the status refers to.
    note : str
        Details worth showing to the user.
    """

    key: str
    status: Status
    local: ManifestEntry | None = None
    remote: RemoteVersion | None = None
    source: str | None = None
    note: str = ""

    @property
    def needs_update(self) -> bool:
        """Whether ``update()`` downloads this dataset."""
        return self.status in ("missing", "outdated", "unknown")


def _resolve_dir(dname: PathType | None) -> Path:
    if dname is not None:
        return Path(dname).expanduser()
    try:
        from smartg.config import DIR_AUXDATA
    except NameError as err:
        raise ValueError(
            "no auxdata directory: pass dname or set the SMARTG_DIR_AUXDATA"
            " environment variable"
        ) from err
    return Path(DIR_AUXDATA)


def _fmt_date(moment: datetime | None) -> str:
    utc = _utc(moment)
    return "-" if utc is None else utc.strftime("%Y-%m-%d %H:%M")


def _print_table(rows: Iterable[Sequence[str]]) -> None:
    rows = [list(row) for row in rows]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        cells = [cell.ljust(width) for cell, width in zip(row, widths)]
        print("  ".join(cells).rstrip())


class AuxData:
    """The SMART-G auxiliary data directory.

    Parameters
    ----------
    dname : str or path-like, optional
        The auxdata directory. Defaults to the ``SMARTG_DIR_AUXDATA``
        environment variable (through ``smartg.config``).
    datasets : dict[str, Dataset], optional
        The datasets to manage, ``DATASETS`` by default.
    timeout : float, optional
        Seconds allowed to each network operation.
    progress : bool, optional
        Show progress bars.

    Attributes
    ----------
    dir : Path
        The auxdata directory.
    manifest : Manifest
        The record of the installed datasets, read at construction.
    datasets : dict[str, Dataset]
        The managed datasets by key.

    Examples
    --------
    >>> from smartg.auxdata import AuxData
    >>> aux = AuxData("/dir/where/to/save/data")
    >>> aux.download()  # everything missing
    >>> aux.check_update()  # what changed remotely, no download
    >>> aux.update()  # refresh the missing and outdated datasets
    """

    def __init__(
        self,
        dname: PathType | None = None,
        *,
        datasets: Mapping[str, Dataset] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        progress: bool = True,
    ) -> None:
        self.dir = _resolve_dir(dname)
        self.datasets = dict(DATASETS if datasets is None else datasets)
        self.timeout = timeout
        self.progress = progress
        self.manifest = Manifest.load(self.dir / MANIFEST_NAME)

    # -- selection and state -----------------------------------------

    def select(self, data_type: DataSelector = "all") -> list[Dataset]:
        """The datasets named by ``data_type``.

        Parameters
        ----------
        data_type : str or sequence of str
            ``"all"``, one dataset key, or several keys.

        Raises
        ------
        ValueError
            An unknown key.
        """
        if isinstance(data_type, str):
            keys = list(self.datasets) if data_type == "all" else [data_type]
        else:
            keys = list(dict.fromkeys(data_type))
        unknown = [key for key in keys if key not in self.datasets]
        if unknown:
            raise ValueError(
                f"Unknown dataset(s) {', '.join(map(repr, unknown))}. Must be"
                f" 'all' or one of: {', '.join(self.datasets)}"
            )
        return [self.datasets[key] for key in keys]

    def path(self, dataset: Dataset) -> Path:
        """The directory of a dataset."""
        return self.dir / dataset.dirname

    def is_present(self, dataset: Dataset) -> bool:
        return self.path(dataset).is_dir()

    def status(
        self, data_type: DataSelector = "all"
    ) -> dict[str, DatasetStatus]:
        """Compare the datasets with their remote source.

        Queries the remote versions (a few small requests) and
        downloads nothing.

        Returns
        -------
        dict[str, DatasetStatus]
            By dataset key, in selection order.
        """
        return {
            dataset.key: self._status_of(dataset)
            for dataset in self.select(data_type)
        }

    def _probe(
        self, dataset: Dataset
    ) -> tuple[RemoteVersion | None, str, str]:
        """The first remote version that answers, with the source
        name and the errors of the sources tried before."""
        notes: list[str] = []
        name = dataset.primary.name
        remote: RemoteVersion | None = None
        for source in dataset.sources:
            try:
                remote, name = source.remote_version(self.timeout), source.name
                break
            except _FETCH_ERRORS as err:
                notes.append(f"{source.name} unreachable: {err}")
        return remote, name, "; ".join(notes)

    def _status_of(self, dataset: Dataset) -> DatasetStatus:
        key = dataset.key
        entry = self.manifest.get(key)
        if not self.is_present(dataset):
            remote, source, note = self._probe(dataset)
            return DatasetStatus(key, "missing", entry, remote, source, note)
        if entry is None:
            remote, source, note = self._probe(dataset)
            note = "; ".join(
                n for n in ("not recorded, update() downloads it again", note)
                if n
            )
            return DatasetStatus(key, "unknown", None, remote, source, note)
        source = dataset.source_named(entry.source)
        if source is None:
            return DatasetStatus(
                key,
                "unknown",
                entry,
                None,
                entry.source,
                f"recorded source {entry.source!r} no longer exists,"
                " update() downloads it again",
            )
        try:
            remote = source.remote_version(self.timeout)
        except _FETCH_ERRORS as err:
            return DatasetStatus(
                key, "unreachable", entry, None, source.name, f"{err}"
            )
        notes: list[str] = []
        status: Status
        if remote.kind == "unknown" or entry.version_kind == "unknown":
            status = "unversioned"
            notes.append(f"{source.name} gives no version, update(force=True)"
                         " to download it again")
        elif remote.fingerprint == entry.version:
            status = "up to date"
        else:
            status = "outdated"
        if source is not dataset.primary:
            notes.append(
                f"fallback source, update(force=True) switches to"
                f" {dataset.primary.name} when it is reachable"
            )
        return DatasetStatus(
            key, status, entry, remote, source.name, "; ".join(notes)
        )

    def check_update(
        self, data_type: DataSelector = "all"
    ) -> dict[str, DatasetStatus]:
        """Print the status of the datasets and return it.

        Downloads nothing. See ``DatasetStatus`` for the meaning of
        the statuses.

        Parameters
        ----------
        data_type : str or sequence of str, optional
            ``"all"``, one dataset key, or several keys.

        Returns
        -------
        dict[str, DatasetStatus]
            By dataset key.
        """
        statuses = self.status(data_type)
        print(f"Auxiliary data in {self.dir}")
        rows = [(
            "dataset", "status", "downloaded (UTC)", "remote (UTC)",
            "source / note",
        )]
        for key, st in statuses.items():
            downloaded = None
            if st.local is not None:
                downloaded = _parse_iso(st.local.downloaded_at)
            remote = st.remote.modified if st.remote else None
            source = st.source or "-"
            rows.append((
                key,
                st.status,
                _fmt_date(downloaded),
                _fmt_date(remote),
                f"{source}: {st.note}" if st.note else source,
            ))
        _print_table(rows)
        todo = [key for key, st in statuses.items() if st.needs_update]
        blocked = [
            key for key, st in statuses.items() if st.status == "unreachable"
        ]
        if todo:
            print(f"update() would download: {', '.join(todo)}")
        elif not blocked:
            print("Everything is up to date.")
        if blocked:
            print(f"Could not be checked: {', '.join(blocked)}")
        return statuses

    # -- installation -----------------------------------------------

    def download(
        self, data_type: DataSelector = "all", force: bool = False
    ) -> None:
        """Download the datasets that are not on disk.

        Parameters
        ----------
        data_type : str or sequence of str, optional
            ``"all"``, one dataset key, or several keys.
        force : bool, optional
            Download the selected datasets even when already on disk.

        Raises
        ------
        AuxDataDownloadError
            At the end of the run, when some datasets failed. The
            other ones are installed and recorded.
        """
        self._cleanup_leftovers()
        queue: list[Dataset] = []
        for dataset in self.select(data_type):
            if self.is_present(dataset) and not force:
                print(
                    f"{dataset.key}: already in {self.path(dataset)}, skipped"
                    " (force=True to download it again, update() to refresh"
                    " it when the remote changed)"
                )
            else:
                queue.append(dataset)
        self._run(queue)

    def update(
        self, data_type: DataSelector = "all", force: bool = False
    ) -> None:
        """Download the missing, outdated and unrecorded datasets.

        Each replaced directory is swapped in one rename once its
        new content is fully extracted, and the manifest is updated
        after each dataset.

        Parameters
        ----------
        data_type : str or sequence of str, optional
            ``"all"``, one dataset key, or several keys.
        force : bool, optional
            Download every selected dataset, whatever its status.

        Raises
        ------
        AuxDataDownloadError
            At the end of the run, when some datasets failed.
        """
        self._cleanup_leftovers()
        queue: list[Dataset] = []
        for key, st in self.status(data_type).items():
            if force or st.needs_update:
                reason = "forced" if force else st.status
                print(f"{key}: {reason}, will be downloaded")
                queue.append(self.datasets[key])
            else:
                note = f" ({st.note})" if st.note else ""
                print(f"{key}: {st.status}, skipped{note}")
        self._run(queue)

    def _run(self, queue: list[Dataset]) -> None:
        if not queue:
            print("Nothing to download.")
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        failures: dict[str, str] = {}
        for dataset in queue:
            print(f"\n{dataset.key}: {dataset.description}")
            try:
                entry = self._install(dataset)
            except _FETCH_ERRORS as err:
                failures[dataset.key] = str(err)
                print(f"{dataset.key}: FAILED ({err})")
                continue
            self.manifest.set(dataset.key, entry)
            self.manifest.save()
            print(
                f"{dataset.key}: installed in {self.path(dataset)}"
                f" ({entry.source}, version {entry.version})"
            )
        done = len(queue) - len(failures)
        print(f"\n{done}/{len(queue)} dataset(s) installed in {self.dir}")
        if failures:
            raise AuxDataDownloadError(failures)

    def _install(self, dataset: Dataset) -> ManifestEntry:
        """Install a dataset from the first source that works."""
        error: Exception | None = None
        for source in dataset.sources:
            if error is not None:
                print(f"{dataset.key}: trying the fallback {source.name}")
            try:
                return self._install_from(dataset, source)
            except _FETCH_ERRORS as err:
                print(f"{dataset.key}: source {source.name} failed ({err})")
                error = err
        assert error is not None
        raise error

    def _install_from(self, dataset: Dataset, source: Source) -> ManifestEntry:
        # The staging directory lives next to the target so that the
        # final rename stays on one filesystem.
        stage = Path(
            tempfile.mkdtemp(prefix=f".{dataset.dirname}.tmp-", dir=self.dir)
        )
        try:
            print(f"{dataset.key}: downloading {source.url}")
            version = source.fetch(
                stage,
                label=dataset.key,
                timeout=self.timeout,
                progress=self.progress,
            )
            produced = stage / dataset.dirname
            if not produced.is_dir():
                found = ", ".join(sorted(p.name for p in stage.iterdir()))
                raise ValueError(
                    f"the archive of {dataset.key} produced"
                    f" {found or 'nothing'}, expected a {dataset.dirname!r}"
                    " directory"
                )
            self._swap_in(produced, dataset)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return ManifestEntry(
            dirname=dataset.dirname,
            source=source.name,
            url=source.url,
            version=version.fingerprint,
            version_kind=version.kind,
            remote_modified=_iso(version.modified),
            downloaded_at=_now_iso(),
        )

    def _swap_in(self, produced: Path, dataset: Dataset) -> None:
        """Move the extracted directory into place, replacing the
        previous one with two renames."""
        target = self.path(dataset)
        old: Path | None = None
        if target.exists() or target.is_symlink():
            old = self.dir / f".{dataset.dirname}.old-{os.urandom(4).hex()}"
            target.rename(old)
        try:
            os.replace(produced, target)
        except OSError:
            if old is not None:
                old.rename(target)
            raise
        if old is not None:
            try:
                shutil.rmtree(old)
            except OSError as err:
                print(
                    f"Warning: could not remove the previous copy {old}"
                    f" ({err}), remove it manually"
                )

    def _cleanup_leftovers(self) -> None:
        """Remove the staging and backup directories of interrupted
        runs."""
        if not self.dir.is_dir():
            return
        for dataset in self.datasets.values():
            patterns = (
                f".{dataset.dirname}.tmp-*",
                f".{dataset.dirname}.old-*",
            )
            for pattern in patterns:
                for leftover in sorted(self.dir.glob(pattern)):
                    print(f"Removing {leftover} left by an interrupted run")
                    shutil.rmtree(leftover, ignore_errors=True)


# ---------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------


def download(
    dname: PathType | None = None,
    data_type: DataSelector = "all",
    force: bool = False,
) -> None:
    """Download the SMART-G auxiliary data.

    The datasets already on disk are skipped, see ``update`` to
    refresh them when the remote changed.

    Parameters
    ----------
    dname : str or path-like, optional
        Directory where the data is saved. Defaults to the
        ``SMARTG_DIR_AUXDATA`` environment variable.
    data_type : str or sequence of str, optional
        ``"all"`` (default), one key, or several keys among:

        * aer -> aerosol optical properties
        * acs -> molecular absorption cross sections
        * atm -> atmospheric profiles
        * STP -> STP (Solar Power Tower) files with the heliostat
          positions
        * valid -> validation files
        * water -> water files needed by some simulations, including
          the ocean
        * kdis -> k-distributions
        * cld -> cloud optical properties
        * IPRT -> data from IPRT (International working group on
          Polarized Radiative Transfer)
        * reptran -> REPTRAN tables from libRadtran (HYGEOS mirror as
          fallback)
    force : bool, optional
        Download the selected datasets even when already on disk.

    Raises
    ------
    AuxDataDownloadError
        At the end of the run, when some datasets failed. The other
        ones are installed.

    Examples
    --------
    >>> from pathlib import Path
    >>> from smartg.auxdata import download
    >>> download(Path("/dir/where/to/save/data"))
    >>> download(Path("/dir/where/to/save/data"), ["aer", "cld"])
    """
    AuxData(dname).download(data_type, force=force)


def check_update(
    dname: PathType | None = None, data_type: DataSelector = "all"
) -> dict[str, DatasetStatus]:
    """Print which auxiliary datasets are missing or outdated.

    Queries the remote versions and downloads nothing.

    Parameters
    ----------
    dname : str or path-like, optional
        The auxdata directory. Defaults to the ``SMARTG_DIR_AUXDATA``
        environment variable.
    data_type : str or sequence of str, optional
        ``"all"`` (default), one key, or several keys (see
        ``download``).

    Returns
    -------
    dict[str, DatasetStatus]
        The status by dataset key; ``needs_update`` tells whether
        ``update`` would download it.

    Examples
    --------
    >>> from smartg.auxdata import check_update
    >>> statuses = check_update()
    >>> [key for key, st in statuses.items() if st.needs_update]
    """
    return AuxData(dname).check_update(data_type)


def update(
    dname: PathType | None = None,
    data_type: DataSelector = "all",
    force: bool = False,
) -> None:
    """Download the missing, outdated and unrecorded datasets.

    Parameters
    ----------
    dname : str or path-like, optional
        The auxdata directory. Defaults to the ``SMARTG_DIR_AUXDATA``
        environment variable.
    data_type : str or sequence of str, optional
        ``"all"`` (default), one key, or several keys (see
        ``download``).
    force : bool, optional
        Download every selected dataset, whatever its status.

    Raises
    ------
    AuxDataDownloadError
        At the end of the run, when some datasets failed.

    Examples
    --------
    >>> from smartg.auxdata import update
    >>> update()  # only what changed
    >>> update(data_type="reptran", force=True)
    """
    AuxData(dname).update(data_type, force=force)
