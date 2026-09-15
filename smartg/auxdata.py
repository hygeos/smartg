"""Download, check and update the SMART-G auxiliary data.

The auxiliary datasets (aerosol and cloud optical properties,
atmospheric profiles, absorption cross sections, k-distributions,
REPTRAN tables, validation and IPRT data, ...) are hosted as
Nextcloud shares on docs.hygeos.com, except REPTRAN whose reference
is the libRadtran archive with a HYGEOS mirror as fallback.

Every download is recorded in a manifest (``.smartg_auxdata.json``)
inside the auxdata directory: the source used, its version
fingerprint (the WebDAV ETag of the share or the HTTP ETag of the
archive), the download date and the checksum of every file.
``check_update`` compares the manifest with the remote versions and
with the files on disk without downloading anything, ``update``
refreshes only the missing or outdated datasets, replacing each
directory in one rename so that an interrupted run never leaves a
half-updated dataset, and ``restore`` puts back the remote copy of
the files modified or deleted locally: the remote is the reference.

Key Classes
-----------
AuxData
    The auxiliary data directory: download, status, update, restore.
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
FileCheck
    The files of one installed dataset against the manifest records.
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
restore
    Replace the locally modified or missing files by the remote ones.
extract_zip, extract_tar
    Safely extract an archive, optionally keeping one subfolder.
"""

from __future__ import annotations

import base64
import hashlib
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
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from email.message import Message
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol
from urllib.parse import quote

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
    "modified",
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
class FileRecord:
    """The size, modification time and SHA-256 of an installed file."""

    size: int
    mtime_ns: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> FileRecord:
    stat = path.stat()
    return FileRecord(stat.st_size, stat.st_mtime_ns, _sha256(path))


def _walk_files(root: Path) -> list[str]:
    """The regular files under root, as sorted POSIX relative paths."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.is_file() and not path.is_symlink():
                found.append(path.relative_to(root).as_posix())
    return sorted(found)


def _scan_files(
    root: Path, progress: bool = True, label: str = ""
) -> dict[str, FileRecord]:
    """Record every file under root."""
    names = _walk_files(root)
    bar = tqdm(
        names,
        unit="file",
        desc=f"{label} checksums".strip(),
        leave=False,
        disable=not progress,
    )
    return {name: _record(root / name) for name in bar}


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
    files : dict[str, FileRecord]
        The installed files by path relative to the dataset
        directory, as extracted.
    """

    dirname: str
    source: str
    url: str
    version: str
    version_kind: VersionKind
    remote_modified: str | None
    downloaded_at: str
    files: dict[str, FileRecord] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManifestEntry:
        plain = {
            f.name: data[f.name] for f in fields(cls) if f.name != "files"
        }
        files = {
            name: FileRecord(**record)
            for name, record in data.get("files", {}).items()
        }
        return cls(**plain, files=files)


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

    def fetch_files(
        self,
        dest: Path,
        dirname: str,
        names: Sequence[str],
        *,
        label: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        progress: bool = True,
    ) -> None:
        """Download some files of the dataset, given by their path
        relative to its directory ``dirname``, under ``dest``."""
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

    @property
    def _auth(self) -> str:
        token = base64.b64encode(f"{self.token}:".encode()).decode("ascii")
        return f"Basic {token}"

    def _propfind(self) -> urllib.request.Request:
        return urllib.request.Request(
            self.webdav_url,
            data=_PROPFIND_BODY,
            method="PROPFIND",
            headers={
                "Depth": "0",
                "Authorization": self._auth,
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

    def fetch_files(
        self,
        dest: Path,
        dirname: str,
        names: Sequence[str],
        *,
        label: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        progress: bool = True,
    ) -> None:
        # The share root is the dataset directory, so a path relative
        # to the dataset is a WebDAV path of the share.
        bar = tqdm(names, unit="file", desc=label, leave=False,
                   disable=not progress)
        for name in bar:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(
                self.webdav_url + quote(name),
                headers={"Authorization": self._auth},
            )
            _download_to_file(
                request, target, timeout=timeout, progress=False, label=name
            )


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

    def fetch_files(
        self,
        dest: Path,
        dirname: str,
        names: Sequence[str],
        *,
        label: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        progress: bool = True,
    ) -> None:
        # No access to single files: the whole archive is fetched and
        # only the wanted members are kept.
        whole = Path(tempfile.mkdtemp(prefix=".whole-", dir=dest))
        try:
            self.fetch(whole, label=label, timeout=timeout, progress=progress)
            for name in names:
                fetched = whole / dirname / name
                if fetched.is_file():
                    target = dest / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(fetched, target)
        finally:
            shutil.rmtree(whole, ignore_errors=True)


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
class FileCheck:
    """The files of an installed dataset against the manifest records.

    Attributes
    ----------
    modified : tuple[str, ...]
        The recorded files whose content changed since the download.
    missing : tuple[str, ...]
        The recorded files no longer on disk.
    extra : tuple[str, ...]
        The files on disk that were not part of the download. They
        are reported and kept.
    recorded : bool
        Whether the manifest has file records to compare with.
    """

    modified: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    recorded: bool = True

    @property
    def clean(self) -> bool:
        """No recorded file was modified or removed."""
        return not (self.modified or self.missing)

    @property
    def restorable(self) -> tuple[str, ...]:
        """The files ``restore()`` replaces by the remote ones."""
        return self.modified + self.missing

    def summary(self) -> str:
        parts = [
            f"{len(names)} {what}"
            for what, names in (
                ("modified", self.modified),
                ("missing", self.missing),
                ("extra", self.extra),
            )
            if names
        ]
        return ", ".join(parts) + " file(s)" if parts else "files intact"


@dataclass(frozen=True)
class DatasetStatus:
    """The state of one dataset on disk against its remote source.

    Attributes
    ----------
    key : str
        The dataset key.
    status : str
        ``"missing"`` (not on disk), ``"up to date"``, ``"modified"``
        (the remote did not change but some local files did, see
        ``files``), ``"outdated"`` (the remote changed since the
        download), ``"unknown"`` (on disk but not recorded in the
        manifest), ``"unversioned"`` (the source gives no version to
        compare) or ``"unreachable"`` (the recorded source did not
        answer).
    local : ManifestEntry or None
        The manifest record, when there is one.
    remote : RemoteVersion or None
        The remote version, when it could be read.
    source : str or None
        The source the status refers to.
    note : str
        Details worth showing to the user.
    files : FileCheck or None
        The local files against the records, for a recorded dataset.
    """

    key: str
    status: Status
    local: ManifestEntry | None = None
    remote: RemoteVersion | None = None
    source: str | None = None
    note: str = ""
    files: FileCheck | None = None

    @property
    def needs_update(self) -> bool:
        """Whether ``update()`` downloads this dataset."""
        return self.status in ("missing", "outdated", "unknown")

    @property
    def needs_restore(self) -> bool:
        """Whether ``restore()`` has files to replace."""
        return self.status == "modified"


class StatusReport(dict[str, DatasetStatus]):
    """The statuses by dataset key.

    A plain dict whose representation stays short when echoed in a
    notebook: the details of a dataset are in its ``DatasetStatus``.
    """

    def __repr__(self) -> str:
        inner = ", ".join(f"{key}: {st.status}" for key, st in self.items())
        return f"<StatusReport {inner}>"


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


def _confirm(prompt: str) -> bool:
    try:
        answer = input(prompt)
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


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

    def status(self, data_type: DataSelector = "all") -> StatusReport:
        """Compare the datasets with their remote source.

        Queries the remote versions (a few small requests) and
        downloads nothing.

        Returns
        -------
        StatusReport
            The ``DatasetStatus`` by dataset key, in selection order.
        """
        return StatusReport(
            (dataset.key, self._status_of(dataset))
            for dataset in self.select(data_type)
        )

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

    def verify(self, data_type: DataSelector = "all") -> dict[str, FileCheck]:
        """Compare the files on disk with the manifest records.

        Works offline. A file is read again only when its size or
        modification time changed since the download, so a content
        change that keeps both is not seen.

        Returns
        -------
        dict[str, FileCheck]
            By dataset key; ``recorded`` is False for the datasets
            that are missing or not in the manifest.
        """
        checks: dict[str, FileCheck] = {}
        for dataset in self.select(data_type):
            entry = self.manifest.get(dataset.key)
            if entry is None or not self.is_present(dataset):
                checks[dataset.key] = FileCheck(recorded=False)
            else:
                checks[dataset.key] = self._verify(dataset, entry)
        return checks

    def _verify(self, dataset: Dataset, entry: ManifestEntry) -> FileCheck:
        if not entry.files:
            return FileCheck(recorded=False)
        root = self.path(dataset)
        modified: list[str] = []
        missing: list[str] = []
        for name, record in entry.files.items():
            path = root / name
            if not path.is_file():
                missing.append(name)
                continue
            stat = path.stat()
            if stat.st_size != record.size:
                modified.append(name)
            elif stat.st_mtime_ns != record.mtime_ns:
                if _sha256(path) != record.sha256:
                    modified.append(name)
        extra = [name for name in _walk_files(root) if name not in entry.files]
        return FileCheck(tuple(modified), tuple(missing), tuple(extra))

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
        files = self._verify(dataset, entry)
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
                files,
            )
        notes: list[str] = []
        status: Status
        try:
            remote = source.remote_version(self.timeout)
        except _FETCH_ERRORS as err:
            remote = None
            status = "unreachable"
            notes.append(str(err))
        else:
            if remote.kind == "unknown" or entry.version_kind == "unknown":
                status = "unversioned"
                notes.append(
                    f"{source.name} gives no version, update(force=True) to"
                    " download it again"
                )
            elif remote.fingerprint == entry.version:
                status = "modified" if not files.clean else "up to date"
            else:
                status = "outdated"
        if not files.clean or files.extra:
            summary = files.summary()
            if status == "modified":
                summary += ", restore() replaces them by the remote ones"
            elif status == "outdated":
                summary += ", update() replaces the whole dataset"
            notes.insert(0, summary)
        if source is not dataset.primary:
            notes.append(
                f"fallback source, update(force=True) switches to"
                f" {dataset.primary.name} when it is reachable"
            )
        return DatasetStatus(
            key, status, entry, remote, source.name, "; ".join(notes), files
        )

    def check_update(self, data_type: DataSelector = "all") -> StatusReport:
        """Print the status of the datasets and return it.

        Downloads nothing. See ``DatasetStatus`` for the meaning of
        the statuses.

        Parameters
        ----------
        data_type : str or sequence of str, optional
            ``"all"``, one dataset key, or several keys.

        Returns
        -------
        StatusReport
            The ``DatasetStatus`` by dataset key.
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
        self._print_files(statuses)
        todo = [key for key, st in statuses.items() if st.needs_update]
        blocked = [
            key for key, st in statuses.items() if st.status == "unreachable"
        ]
        to_restore = [key for key, st in statuses.items() if st.needs_restore]
        if todo:
            print(f"update() would download: {', '.join(todo)}")
        if to_restore:
            print(f"restore() would repair: {', '.join(to_restore)}")
        if not (todo or blocked or to_restore):
            print("Everything is up to date.")
        if blocked:
            print(f"Could not be checked: {', '.join(blocked)}")
        return statuses

    @staticmethod
    def _print_files(
        statuses: Mapping[str, DatasetStatus], limit: int = 20
    ) -> None:
        """List the files that differ from the download."""
        for key, st in statuses.items():
            files = st.files
            if files is None or (files.clean and not files.extra):
                continue
            print(f"{key}: files differing from the download")
            lines = [
                (what, name)
                for what, names in (
                    ("modified", files.modified),
                    ("missing", files.missing),
                    ("extra, kept", files.extra),
                )
                for name in names
            ]
            for what, name in lines[:limit]:
                print(f"  {what:12} {name}")
            if len(lines) > limit:
                print(f"  ... and {len(lines) - limit} more")

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
            note = f" ({st.note})" if st.note else ""
            if force or st.needs_update:
                reason = "forced" if force else st.status
                print(f"{key}: {reason}, will be downloaded{note}")
                queue.append(self.datasets[key])
            else:
                print(f"{key}: {st.status}, skipped{note}")
        self._run(queue)

    def restore(
        self, data_type: DataSelector = "all", yes: bool = False
    ) -> dict[str, list[str]]:
        """Replace the locally modified or missing files by the remote
        ones.

        The files are listed and a confirmation is asked, unless
        ``yes`` is True. Only the datasets whose remote did not
        change since the download are repaired (status
        ``"modified"``); an outdated dataset is left to ``update()``,
        which replaces it entirely. The extra files are kept.

        Parameters
        ----------
        data_type : str or sequence of str, optional
            ``"all"``, one dataset key, or several keys.
        yes : bool, optional
            Do not ask for confirmation.

        Returns
        -------
        dict[str, list[str]]
            The restored files by dataset key.

        Raises
        ------
        AuxDataDownloadError
            At the end of the run, when some files could not be
            fetched.
        """
        statuses = self.status(data_type)
        todo: dict[str, DatasetStatus] = {}
        for key, st in statuses.items():
            if st.files is None or st.files.clean:
                continue
            if st.needs_restore:
                todo[key] = st
            elif st.status == "outdated":
                print(
                    f"{key}: {st.files.summary()} but the remote changed"
                    " too, run update() to replace the whole dataset"
                )
            else:
                print(
                    f"{key}: {st.files.summary()} but the dataset is"
                    f" {st.status}, nothing restored"
                )
        if not todo:
            print("No file to restore.")
            return {}
        print("Files to replace by the remote ones:")
        self._print_files(todo, limit=1000)
        count = sum(
            len(st.files.restorable) for st in todo.values() if st.files
        )
        if not yes and not _confirm(f"Replace these {count} file(s)? [y/N] "):
            print("Nothing restored.")
            return {}
        restored: dict[str, list[str]] = {}
        failures: dict[str, str] = {}
        for key, st in todo.items():
            dataset = self.datasets[key]
            assert st.local is not None and st.files is not None
            source = dataset.source_named(st.local.source)
            assert source is not None
            names = list(st.files.restorable)
            try:
                self._restore_files(dataset, source, st.local, names)
            except _FETCH_ERRORS as err:
                failures[key] = str(err)
                print(f"{key}: FAILED ({err})")
                continue
            restored[key] = names
            print(f"{key}: {len(names)} file(s) restored from {source.name}")
        if failures:
            raise AuxDataDownloadError(failures)
        return restored

    def _restore_files(
        self,
        dataset: Dataset,
        source: Source,
        entry: ManifestEntry,
        names: Sequence[str],
    ) -> None:
        for name in names:
            _check_member_name(name)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{dataset.dirname}.tmp-", dir=self.dir)
        )
        try:
            source.fetch_files(
                stage,
                dataset.dirname,
                names,
                label=dataset.key,
                timeout=self.timeout,
                progress=self.progress,
            )
            absent = [name for name in names if not (stage / name).is_file()]
            if absent:
                raise ValueError(
                    f"not in the remote {dataset.key} dataset:"
                    f" {', '.join(absent)}"
                )
            root = self.path(dataset)
            files = dict(entry.files)
            for name in names:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / name, target)
                files[name] = _record(target)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        self.manifest.set(dataset.key, replace(entry, files=files))
        self.manifest.save()

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
            # Recorded before the swap: a rename keeps the mtimes.
            files = _scan_files(produced, self.progress, dataset.key)
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
            files=files,
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
) -> StatusReport:
    """Print which auxiliary datasets are missing, outdated or modified.

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
    StatusReport
        The ``DatasetStatus`` by dataset key; ``needs_update`` tells
        whether ``update`` would download it, ``needs_restore``
        whether ``restore`` has files to replace.

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


def restore(
    dname: PathType | None = None,
    data_type: DataSelector = "all",
    yes: bool = False,
) -> dict[str, list[str]]:
    """Replace the locally modified or missing files by the remote ones.

    The files are listed and a confirmation is asked, unless ``yes``
    is True. The remote is the reference: the datasets whose remote
    changed too are left to ``update``.

    Parameters
    ----------
    dname : str or path-like, optional
        The auxdata directory. Defaults to the ``SMARTG_DIR_AUXDATA``
        environment variable.
    data_type : str or sequence of str, optional
        ``"all"`` (default), one key, or several keys (see
        ``download``).
    yes : bool, optional
        Do not ask for confirmation.

    Returns
    -------
    dict[str, list[str]]
        The restored files by dataset key.

    Examples
    --------
    >>> from smartg.auxdata import check_update, restore
    >>> check_update()  # lists the modified files
    >>> restore()  # asks before replacing them
    """
    return AuxData(dname).restore(data_type, yes=yes)
