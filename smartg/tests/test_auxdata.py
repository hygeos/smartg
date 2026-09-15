"""Tests of smartg.auxdata.

Everything runs offline against fake sources and archives built in
tmp_path, except the last test (marker slow) which queries the real
remote versions without downloading any data.
"""

from __future__ import annotations

import io
import json
import re
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path

import pytest

import smartg.auxdata as auxdata
from smartg.auxdata import (
    DATASETS,
    AuxData,
    AuxDataDownloadError,
    Dataset,
    HttpArchiveSource,
    Manifest,
    ManifestEntry,
    NextcloudSource,
    RemoteVersion,
    check_update,
    extract_tar,
    extract_zip,
)

UTC = timezone.utc

PROPFIND_XML = (
    b'<?xml version="1.0"?>'
    b'<d:multistatus xmlns:d="DAV:" xmlns:s="http://sabredav.org/ns">'
    b"<d:response><d:href>/public.php/webdav/</d:href><d:propstat><d:prop>"
    b"<d:getlastmodified>Mon, 27 Apr 2026 14:37:21 GMT</d:getlastmodified>"
    b"<d:resourcetype><d:collection/></d:resourcetype>"
    b"<d:getetag>&quot;69ef74a14dadf&quot;</d:getetag>"
    b"</d:prop><d:status>HTTP/1.1 200 OK</d:status></d:propstat>"
    b"</d:response></d:multistatus>"
)


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(auxdata, "RETRY_DELAYS", (0, 0))


def version(
    fingerprint: str, kind: auxdata.VersionKind = "etag"
) -> RemoteVersion:
    return RemoteVersion(fingerprint, kind, datetime(2026, 1, 2, tzinfo=UTC))


class FakeSource:
    """A source writing <dirname>/file.txt, counting its calls."""

    def __init__(self, name: str, dirname: str) -> None:
        self.name = name
        self.dirname = dirname
        self.version = version("v1")
        self.error: Exception | None = None
        self.fetch_error: Exception | None = None
        self.payload = "v1"
        self.version_calls = 0
        self.fetch_calls = 0

    @property
    def url(self) -> str:
        return f"fake://{self.name}/{self.dirname}"

    def remote_version(self, timeout: float = 30.0) -> RemoteVersion:
        self.version_calls += 1
        if self.error is not None:
            raise self.error
        return self.version

    def fetch(
        self,
        dest: Path,
        *,
        label: str = "",
        timeout: float = 30.0,
        progress: bool = True,
    ) -> RemoteVersion:
        self.fetch_calls += 1
        if self.fetch_error is not None:
            raise self.fetch_error
        (dest / self.dirname).mkdir()
        (dest / self.dirname / "file.txt").write_text(self.payload)
        return self.version


class FakeResponse(io.BytesIO):
    """What the fake urlopen returns."""

    def __init__(self, body: bytes, headers: dict[str, str] | None = None):
        super().__init__(body)
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value


@pytest.fixture
def sources() -> dict[str, FakeSource]:
    return {
        "a": FakeSource("one", "alpha"),
        "b_primary": FakeSource("prim", "beta"),
        "b_fallback": FakeSource("fall", "beta"),
    }


@pytest.fixture
def registry(sources) -> dict[str, Dataset]:
    return {
        "a": Dataset("a", "alpha", (sources["a"],), "alpha data"),
        "b": Dataset(
            "b",
            "beta",
            (sources["b_primary"], sources["b_fallback"]),
            "beta data",
        ),
    }


@pytest.fixture
def aux(tmp_path, registry) -> AuxData:
    return AuxData(tmp_path, datasets=registry, progress=False)


def entry_for(source: FakeSource, fingerprint: str = "v1") -> ManifestEntry:
    return ManifestEntry(
        dirname=source.dirname,
        source=source.name,
        url=source.url,
        version=fingerprint,
        version_kind="etag",
        remote_modified="2026-01-02T00:00:00Z",
        downloaded_at="2026-01-03T10:00:00Z",
    )


def make_zip(path: Path, names: dict[str, bytes | None]) -> None:
    """Write a zip; a None content makes a directory entry."""
    with zipfile.ZipFile(path, "w") as z:
        for name, content in names.items():
            if content is None:
                z.writestr(name, b"")
            else:
                z.writestr(name, content)


def make_targz(path: Path, names: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in names.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


# -- registry, parsing, manifest ----------------------------------------


def test_registry_consistency():
    assert list(DATASETS) == [
        "aer", "acs", "atm", "STP", "valid", "water", "kdis", "cld",
        "IPRT", "reptran",
    ]
    dirnames = [d.dirname for d in DATASETS.values()]
    assert len(set(dirnames)) == len(dirnames)
    assert all(d.sources for d in DATASETS.values())
    assert [s.name for s in DATASETS["reptran"].sources] == [
        "libradtran", "hygeos",
    ]
    pattern = re.compile(
        r"https://docs\.hygeos\.com/s/[A-Za-z0-9]{15}/download"
    )
    for dataset in DATASETS.values():
        for source in dataset.sources:
            if isinstance(source, NextcloudSource):
                assert pattern.fullmatch(source.url), source.url
    assert DATASETS["reptran"].primary.url.endswith("reptran_2024_all.tar.gz")


def test_parse_propfind():
    remote = auxdata._parse_propfind(PROPFIND_XML)
    assert remote == RemoteVersion(
        "69ef74a14dadf", "etag", datetime(2026, 4, 27, 14, 37, 21, tzinfo=UTC)
    )
    no_etag = PROPFIND_XML.replace(
        b"<d:getetag>&quot;69ef74a14dadf&quot;</d:getetag>", b""
    )
    remote = auxdata._parse_propfind(no_etag)
    assert remote.kind == "last-modified"
    assert remote.fingerprint == "Mon, 27 Apr 2026 14:37:21 GMT"
    nothing = re.sub(rb"<d:prop>.*</d:prop>", b"<d:prop/>", no_etag)
    assert auxdata._parse_propfind(nothing) == RemoteVersion(
        "unknown", "unknown", None
    )
    with pytest.raises(ValueError, match="unreadable"):
        auxdata._parse_propfind(b"<html>not xml")
    with pytest.raises(ValueError, match="without any response"):
        auxdata._parse_propfind(b'<d:multistatus xmlns:d="DAV:"/>')


def test_version_from_headers():
    headers = Message()
    headers["ETag"] = 'W/"abc"'
    headers["Last-Modified"] = "Thu, 22 Jan 2026 23:14:55 GMT"
    remote = auxdata._version_from_headers(headers)
    assert remote.fingerprint == "abc"
    assert remote.kind == "etag"
    assert remote.modified == datetime(2026, 1, 22, 23, 14, 55, tzinfo=UTC)
    del headers["ETag"]
    remote = auxdata._version_from_headers(headers)
    assert remote.kind == "last-modified"
    assert remote.fingerprint == "Thu, 22 Jan 2026 23:14:55 GMT"
    assert auxdata._version_from_headers(Message()).kind == "unknown"


def test_manifest_roundtrip(tmp_path, capsys):
    path = tmp_path / ".smartg_auxdata.json"
    assert Manifest.load(path).entries == {}
    manifest = Manifest(path)
    source = FakeSource("one", "alpha")
    manifest.set("a", entry_for(source))
    manifest.save()
    assert not list(tmp_path.glob("*.tmp"))
    data = json.loads(path.read_text())
    assert data["schema_version"] == 1
    assert data["datasets"]["a"]["source"] == "one"
    assert Manifest.load(path).entries == manifest.entries
    manifest.remove("a")
    assert manifest.get("a") is None

    path.write_text("{not json")
    assert Manifest.load(path).entries == {}
    assert "unreadable manifest" in capsys.readouterr().out
    path.write_text(json.dumps({"schema_version": 99, "datasets": {}}))
    assert Manifest.load(path).entries == {}
    assert "schema_version" in capsys.readouterr().out


def test_resolve_dir(monkeypatch, tmp_path):
    assert AuxData(tmp_path).dir == tmp_path
    assert AuxData("~/x").dir == Path.home() / "x"
    monkeypatch.delenv("SMARTG_DIR_AUXDATA", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delitem(sys.modules, "smartg.config", raising=False)
    with pytest.raises(ValueError, match="SMARTG_DIR_AUXDATA"):
        AuxData(None)
    monkeypatch.setenv("SMARTG_DIR_AUXDATA", str(tmp_path))
    monkeypatch.delitem(sys.modules, "smartg.config", raising=False)
    assert AuxData(None).dir == tmp_path


def test_select(aux):
    assert [d.key for d in aux.select()] == ["a", "b"]
    assert [d.key for d in aux.select("b")] == ["b"]
    assert [d.key for d in aux.select(["b", "a", "b"])] == ["b", "a"]
    with pytest.raises(ValueError, match="'nope'.*'all' or one of: a, b"):
        aux.select("nope")


# -- status -------------------------------------------------------------


def test_status_logic(aux, sources, tmp_path):
    one, prim, fall = sources["a"], sources["b_primary"], sources["b_fallback"]

    statuses = aux.status()
    assert {k: s.status for k, s in statuses.items()} == {
        "a": "missing", "b": "missing",
    }
    assert statuses["b"].source == "prim"
    assert statuses["b"].remote == prim.version
    assert all(s.needs_update for s in statuses.values())

    # missing, primary down: the remote column comes from the fallback
    prim.error = ConnectionError("down")
    st = aux.status("b")["b"]
    assert (st.status, st.source) == ("missing", "fall")
    assert "prim unreachable: down" in st.note
    prim.error = None

    # on disk without a record
    (tmp_path / "alpha").mkdir()
    st = aux.status("a")["a"]
    assert st.status == "unknown"
    assert st.needs_update
    assert "not recorded" in st.note

    # recorded and unchanged
    aux.manifest.set("a", entry_for(one))
    st = aux.status("a")["a"]
    assert (st.status, st.needs_update, st.note) == ("up to date", False, "")

    # the remote changed
    one.version = version("v2")
    assert aux.status("a")["a"].status == "outdated"
    assert aux.status("a")["a"].needs_update

    # no version to compare
    one.version = version("unknown", "unknown")
    st = aux.status("a")["a"]
    assert (st.status, st.needs_update) == ("unversioned", False)

    # the recorded source does not answer
    one.error = urllib.error.URLError("no route")
    st = aux.status("a")["a"]
    assert (st.status, st.needs_update) == ("unreachable", False)
    assert "no route" in st.note

    # a record from a source that no longer exists
    aux.manifest.set("a", entry_for(FakeSource("gone", "alpha")))
    st = aux.status("a")["a"]
    assert (st.status, st.needs_update) == ("unknown", True)

    # installed from the fallback: compared against the fallback only
    (tmp_path / "beta").mkdir()
    aux.manifest.set("b", entry_for(fall))
    prim.error = ConnectionError("down")
    st = aux.status("b")["b"]
    assert (st.status, st.source) == ("up to date", "fall")
    assert "fallback source" in st.note
    fall.version = version("v3")
    assert aux.status("b")["b"].status == "outdated"

    assert one.fetch_calls == prim.fetch_calls == fall.fetch_calls == 0


def test_check_update_prints_table(aux, sources, tmp_path, capsys):
    (tmp_path / "alpha").mkdir()
    aux.manifest.set("a", entry_for(sources["a"]))
    statuses = aux.check_update()
    assert set(statuses) == {"a", "b"}
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[1].split() == [
        "dataset", "status", "downloaded", "(UTC)", "remote", "(UTC)",
        "source", "/", "note",
    ]
    assert re.search(
        r"^a +up to date +2026-01-03 10:00 +2026-01-02 00:00 +one", out, re.M
    )
    assert re.search(r"^b +missing +- +2026-01-02 00:00 +prim", out, re.M)
    assert "update() would download: b" in out

    (tmp_path / "beta").mkdir()
    aux.manifest.set("b", entry_for(sources["b_primary"]))
    aux.check_update()
    assert "Everything is up to date." in capsys.readouterr().out


# -- download and update ------------------------------------------------


def test_download_skips_existing(aux, sources, tmp_path, capsys):
    one = sources["a"]
    aux.download("a")
    assert (tmp_path / "alpha" / "file.txt").read_text() == "v1"
    assert aux.manifest.get("a") == ManifestEntry.from_dict(
        json.loads((tmp_path / auxdata.MANIFEST_NAME).read_text())
        ["datasets"]["a"]
    )
    assert aux.manifest.get("a").version == "v1"
    assert one.fetch_calls == 1

    aux.download("a")
    assert one.fetch_calls == 1
    assert "already in" in capsys.readouterr().out

    aux.download("a", force=True)
    assert one.fetch_calls == 2


def test_update_replaces_directory(aux, sources, tmp_path):
    one = sources["a"]
    aux.download("a")
    (tmp_path / "alpha" / "stale.txt").write_text("old")
    one.version = version("v2")
    one.payload = "v2"

    aux.update("a")
    assert (tmp_path / "alpha" / "file.txt").read_text() == "v2"
    assert not (tmp_path / "alpha" / "stale.txt").exists()
    assert not list(tmp_path.glob(".alpha.*"))
    assert aux.manifest.get("a").version == "v2"
    assert one.fetch_calls == 2

    aux.update("a")
    assert one.fetch_calls == 2
    aux.update("a", force=True)
    assert one.fetch_calls == 3


def test_leftovers_cleaned(aux, tmp_path, capsys):
    stage = tmp_path / ".alpha.tmp-abc"
    stage.mkdir()
    (stage / "junk").write_text("x")
    (tmp_path / ".alpha.old-xyz").mkdir()
    aux.download("a")
    assert not stage.exists()
    assert not (tmp_path / ".alpha.old-xyz").exists()
    assert "interrupted run" in capsys.readouterr().out


def test_failures_collected_and_raised(aux, sources, tmp_path):
    sources["a"].fetch_error = ConnectionError("boom")
    with pytest.raises(AuxDataDownloadError) as info:
        aux.download()
    assert info.value.failures == {"a": "boom"}
    assert "a: boom" in str(info.value)
    assert (tmp_path / "beta" / "file.txt").exists()
    assert not (tmp_path / "alpha").exists()
    assert aux.manifest.get("b") is not None
    assert aux.manifest.get("a") is None
    assert not list(tmp_path.glob(".alpha.*"))


def test_fallback_source_used(aux, sources, capsys):
    sources["b_primary"].fetch_error = urllib.error.HTTPError(
        "http://x", 500, "Internal Server Error", Message(), None
    )
    aux.download("b")
    assert aux.manifest.get("b").source == "fall"
    assert sources["b_fallback"].fetch_calls == 1
    assert "trying the fallback fall" in capsys.readouterr().out


def test_wrong_top_level_dir(aux, sources, tmp_path):
    one = sources["a"]
    aux.download("a")
    one.dirname = "wrong"
    one.version = version("v2")
    with pytest.raises(AuxDataDownloadError) as info:
        aux.update("a")
    assert "expected a 'alpha' directory" in info.value.failures["a"]
    assert (tmp_path / "alpha" / "file.txt").read_text() == "v1"
    assert aux.manifest.get("a").version == "v1"
    assert not list(tmp_path.glob(".alpha.*"))


# -- archives -----------------------------------------------------------


def test_http_archive_source_targz(tmp_path):
    archive = tmp_path / "reptran_all.tar.gz"
    make_targz(archive, {
        "data/correlated_k/reptran/a.cdf": b"aaa",
        "data/correlated_k/reptran/sub/b.cdf": b"bbb",
        "data/other.txt": b"ooo",
    })
    source = HttpArchiveSource(archive.as_uri(), member_root="reptran")
    remote = source.remote_version()
    assert remote.kind == "last-modified"

    dest = tmp_path / "dest"
    dest.mkdir()
    fetched = source.fetch(dest, progress=False)
    assert fetched == remote
    assert (dest / "reptran" / "a.cdf").read_bytes() == b"aaa"
    assert (dest / "reptran" / "sub" / "b.cdf").read_bytes() == b"bbb"
    assert sorted(p.name for p in dest.iterdir()) == ["reptran"]


def test_nextcloud_source_end_to_end(tmp_path, monkeypatch):
    zip_path = tmp_path / "share.zip"
    make_zip(zip_path, {
        "aerosols/": None,
        "aerosols/OPAC/": None,
        "aerosols/OPAC/x.nc": b"netcdf",
    })
    zip_bytes = zip_path.read_bytes()
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.get_method() == "PROPFIND":
            return FakeResponse(PROPFIND_XML)
        assert request.get_method() == "GET"
        return FakeResponse(zip_bytes)  # no Content-Length, as Nextcloud

    monkeypatch.setattr(auxdata, "_urlopen", fake_urlopen)
    source = NextcloudSource("tok")
    aux = AuxData(
        tmp_path / "aux",
        datasets={"aer": Dataset("aer", "aerosols", (source,), "aerosols")},
        progress=False,
    )
    aux.download("aer")

    assert (tmp_path / "aux" / "aerosols" / "OPAC" / "x.nc").exists()
    entry = aux.manifest.get("aer")
    assert entry is not None
    assert entry.source == "hygeos"
    assert entry.url == "https://docs.hygeos.com/s/tok/download"
    assert entry.version == "69ef74a14dadf"
    assert entry.version_kind == "etag"
    assert entry.remote_modified == "2026-04-27T14:37:21Z"
    assert not list((tmp_path / "aux").glob(".aerosols.*"))

    propfind, get = requests
    assert propfind.full_url == "https://docs.hygeos.com/public.php/webdav/"
    assert propfind.get_header("Depth") == "0"
    assert propfind.get_header("Authorization") == "Basic dG9rOg=="
    assert get.full_url == source.url

    assert aux.status("aer")["aer"].status == "up to date"


def test_extract_zip_target_folder(tmp_path):
    archive = tmp_path / "a.zip"
    make_zip(archive, {"x/reptran/f.cdf": b"1", "x/y.txt": b"2"})
    dest = tmp_path / "dest"
    extract_zip(archive, dest, target_folder="reptran", progress=False)
    assert (dest / "reptran" / "f.cdf").read_bytes() == b"1"
    assert not (dest / "x").exists()


@pytest.mark.parametrize("name", ["../evil.txt", "/abs.txt", "C:/x.txt"])
def test_extract_rejects_traversal(tmp_path, name):
    dest = tmp_path / "dest"
    dest.mkdir()
    archive = tmp_path / "a.zip"
    make_zip(archive, {"ok.txt": b"1", name: b"2"})
    with pytest.raises(ValueError, match="unsafe archive member"):
        extract_zip(archive, dest, progress=False)
    assert not list(dest.iterdir())
    archive = tmp_path / "a.tar.gz"
    make_targz(archive, {"ok.txt": b"1", name: b"2"})
    with pytest.raises(ValueError, match="unsafe archive member"):
        extract_tar(archive, dest, progress=False)
    assert not list(dest.iterdir())


def test_extract_archive_sniffs_error_page(tmp_path):
    page = tmp_path / "archive.part"
    page.write_bytes(b"<html>share expired</html>")
    with pytest.raises(ValueError, match="neither a ZIP nor a gzip"):
        auxdata._extract_archive(page, tmp_path)


# -- retries --------------------------------------------------------------


def test_retry_transient_then_success(monkeypatch):
    calls = []

    def flaky(request, timeout):
        calls.append(request.full_url)
        if len(calls) < 3:
            raise urllib.error.URLError("down")
        return FakeResponse(b"ok")

    monkeypatch.setattr(auxdata, "_urlopen", flaky)
    with auxdata._open(urllib.request.Request("http://x/"), 1.0) as r:
        assert r.read() == b"ok"
    assert len(calls) == 3

    def always_down(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.URLError("down")

    calls.clear()
    monkeypatch.setattr(auxdata, "_urlopen", always_down)
    with pytest.raises(ConnectionError, match="after 3 attempts"):
        auxdata._open(urllib.request.Request("http://x/"), 1.0)
    assert len(calls) == 3

    def not_found(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", Message(), None
        )

    calls.clear()
    monkeypatch.setattr(auxdata, "_urlopen", not_found)
    with pytest.raises(urllib.error.HTTPError):
        auxdata._open(urllib.request.Request("http://x/"), 1.0)
    assert len(calls) == 1


def test_head_fallback_to_get(monkeypatch):
    methods = []

    def no_head(request, timeout):
        methods.append(request.get_method())
        if request.get_method() == "HEAD":
            raise urllib.error.HTTPError(
                request.full_url, 405, "Method Not Allowed", Message(), None
            )
        return FakeResponse(b"", {"ETag": '"e1"'})

    monkeypatch.setattr(auxdata, "_urlopen", no_head)
    remote = HttpArchiveSource("http://x/a.tar.gz").remote_version()
    assert remote == RemoteVersion("e1", "etag", None)
    assert methods == ["HEAD", "GET"]


# -- live -----------------------------------------------------------------


@pytest.mark.slow
def test_check_update_live(tmp_path):
    """Query the real remote versions; nothing is downloaded."""
    statuses = check_update(tmp_path)
    assert all(st.status == "missing" for st in statuses.values())
    for key, st in statuses.items():
        assert st.remote is not None, (key, st.note)
        assert st.remote.modified is not None, key
        if key != "reptran":
            assert st.remote.kind == "etag", key
    assert not any(tmp_path.iterdir())
