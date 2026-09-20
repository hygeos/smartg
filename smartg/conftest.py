"""Put the figures of the tests into the pytest-html report.

Generate images with matplotlib and use conftest.savefig(request)
instead of matplotlib's savefig.
Example:

    import conftest
    from matplotlib import pyplot as plt

    def test_html_report(request):
        plt.figure()
        plt.plot()
        conftest.savefig(request)


How to use it
-------------

- To use this module, please create the file `tests/conftest.py` in
  your repository, with the following content:
    from core.conftest import *

- Run pytest with the following options (can be added to pytest.ini):
    --html=tests/test_report.html --self-contained-html

    Sample pytest.ini:
        [pytest]
        addopts= -n auto --html=test_report.html --self-contained-html

Note: this requires pytest-html


Other features
--------------

- If you generate an image as a ByteIO buffer, you can add it
  to the report like so:

    def test_with_image(request):
        plt.plot(...)
        fp = io.BytesIO()
        plt.savefig(fp)
        conftest.add_image_to_report(request, fp)

- If you generate a file in your test, you can link it in the report
  using:

    conftest.add_link_to_report(request, path)

- The images embedded in the html file can be dragged in a browser to
  view them in full size.

Parameters
----------
The ini options this conftest adds to pytest.ini:

img_collapsible
    Whether to use a collapsible section to embed the images
    (default true).
    Ex: true, false
    Otherwise, the standard image integration is used.
img_size
    Image size, when using img_collapsible mode
    Ex: 100%, 250px (default)
dark_mode
    Whether to enable dark mode theme in HTML reports
    (default false).
    Ex: true, false (default)
monitor_peak_memory
    Whether to monitor and report peak RSS memory usage
    (default false).
    Ex: true, false (default)
"""

import base64
import io
import resource
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# Module-level variable to store config
_pytest_config = None


def _get_peak_rss_mb() -> float:
    """Return the peak RSS of the process, in MiB."""
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak_rss / 1024


def _monitor_peak_memory_enabled(config: pytest.Config) -> bool:
    """Return whether peak memory monitoring/reporting is enabled."""
    return {"true": True, "false": False}[
        (config.getini("monitor_peak_memory") or "false").lower()
    ]


def add_image_to_report(request: pytest.FixtureRequest, fp: io.BytesIO) -> None:
    """Append image data to ``request.node.images``.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request fixture of the test.
    fp : io.BytesIO
        The image, already encoded.
    """
    if not hasattr(request.node, "images"):
        request.node.images = []
    fp.seek(0)
    data = fp.read()
    request.node.images.insert(0, data)


def add_extra_to_report(request: pytest.FixtureRequest, *args: Any) -> None:
    """Add extra content to the pytest-html report.

    It goes through ``request.node.extras``.

    Examples
    --------
    >>> add_extra_to_report(request, 'sample text', 'text')
    >>> add_extra_to_report(request, 'sample html', 'html')
    >>> add_extra_to_report(request, 'https://www.wikipedia.org/',
    ...                     'url', 'Link text')
    """
    if not hasattr(request.node, "extras"):
        request.node.extras = []
    request.node.extras.insert(0, args)


def add_link_to_report(request: pytest.FixtureRequest, path: str | Path, name: str = "Link") -> None:
    """Add a link to the local file ``path``.

    The link is made relative to the directory of the html output.
    """
    assert Path(path).exists()
    htmlo = request.config.getoption("--html")
    if htmlo is None:
        return
    html_output = Path(htmlo).resolve()
    try:
        url = Path(path).resolve().relative_to(html_output.parent)
    except ValueError as err:
        raise Exception(
            f"path ({path}) must be a subfolder of output html ({html_output})"
        ) from err
    add_extra_to_report(request, str(url), "url", name)


def savefig(request: pytest.FixtureRequest, **kwargs: Any) -> None:
    """Wrap savefig so that the figure lands in the report.

    The image is added to ``request.node.images``, and ``kwargs``
    are passed on to ``plt.savefig``.
    """
    from matplotlib import pyplot as plt

    fp = io.BytesIO()
    plt.savefig(fp, **kwargs)
    add_image_to_report(request, fp)
    plt.close("all")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Declare the ini options this plugin reads."""
    parser.addini(
        "img_collapsible",
        "Whether to use a collapsible section to embed the images (default "
        + "true).",
    )
    parser.addini(
        "img_size",
        "Image size, when using img_collapsible mode. Ex: 100%, 250px "
        + "(default)",
    )
    parser.addini(
        "dark_mode",
        "Whether to enable dark mode theme in HTML reports (default false).",
    )
    parser.addini(
        "monitor_peak_memory",
        "Whether to monitor and report peak RSS memory usage (default false).",
        default="false",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Store config for later use in hooks."""
    global _pytest_config
    _pytest_config = config


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Function) -> Iterator[None]:
    """Attach the images, the extras and the peak memory to a report."""
    pytest_html: Any = item.config.pluginmanager.getplugin("html")
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and _monitor_peak_memory_enabled(item.config):
        report.peak_rss_mb = _get_peak_rss_mb()

    extra = getattr(report, "extra", [])
    img_size = item.config.getini("img_size") or "250px"
    img_use_extra = {"true": True, "false": False}[
        (item.config.getini("img_collapsible") or "true").lower()
    ]
    if (report.when == "call") and (pytest_html is not None):
        # add docstring
        doc = item.function.__doc__
        if doc is not None:
            extra.append(pytest_html.extras.html(f"<pre>{doc}</pre>"))

        # add images
        img_content = ""
        for image in getattr(item, "images", []):
            b64data = base64.b64encode(image).decode("ascii")
            if img_use_extra:
                img_content += (
                    f'<img src="data:image/png;base64,{b64data}"'
                    + f' style="max-width:{img_size};">'
                )
            else:
                extra.append(pytest_html.extras.image(b64data))

        if img_content:
            extra.append(
                pytest_html.extras.extra(
                    f"""
                <details open>
                    <summary>Images</summary>
                    {img_content}
                </details>
                """,
                    "html",
                )
            )

        # add other extras
        for a in getattr(item, "extras", []):
            extra.append(pytest_html.extras.extra(*a))

        report.extras = extra


def pytest_html_results_table_header(cells: list[str]) -> None:
    """Insert per-test memory columns in pytest-html results table."""
    if _pytest_config is not None and _monitor_peak_memory_enabled(
        _pytest_config
    ):
        cells.insert(2, "<th>Mem Peak (MiB)</th>")


def pytest_html_results_table_row(report: pytest.TestReport, cells: list[str]) -> None:
    """Render per-test memory values in pytest-html results table."""
    if _pytest_config is None or not _monitor_peak_memory_enabled(
        _pytest_config
    ):
        return

    peak_rss_mb = getattr(report, "peak_rss_mb", None)
    if peak_rss_mb is None:
        cells.insert(2, "<td></td>")
        return

    cells.insert(2, f"<td>{peak_rss_mb:.1f}</td>")


# Dark theme CSS embedded directly in conftest.py
DARK_CSS = """
<style type="text/css">
/* Dark theme for pytest-html */
:root {
    --text-color: #c4c4c4;
    --background-color: #272822;
    --container-bg: #272822;
    --table-bg: #2d2d2d;
    --border-color: #555555;
    --collapsible-bg: #2d2d2d;
    --log-bg: #1a1a1a;
    --passed-bg: #0f5132;
    --failed-bg: #842029;
    --skipped-bg: #664d03;
    --error-bg: #842029;
    --xfailed-bg: #664d03;
    --xpassed-bg: #842029;
    --rerun-bg: #664d03;
    --link-color: #4dabf7;
    --link-hover-color: #74c0fc;
    --collapsed-bg: #3d3d3d;
    --empty-color: #cccccc;
}

body {
    background-color: var(--background-color);
    color: var(--text-color);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-color);
}

p {
    color: var(--text-color);
}

.container {
    background-color: var(--container-bg);
    color: var(--text-color);
}

table {
    background-color: var(--table-bg);
    color: var(--text-color);
}

th, td {
    background-color: var(--table-bg);
    color: var(--text-color);
    border-color: var(--border-color);
}

.collapsible {
    background-color: var(--collapsible-bg);
}

.log {
    background-color: var(--log-bg);
    color: var(--text-color);
}

.logwrapper .log {
    color: var(--text-color) !important;
}

.logwrapper .log * {
    color: var(--text-color) !important;
}

.passed {
    background-color: var(--passed-bg);
}

.failed {
    background-color: var(--failed-bg);
}

.skipped {
    background-color: var(--skipped-bg);
}

.error {
    background-color: var(--error-bg);
}

.xfailed {
    background-color: var(--xfailed-bg);
}

.xpassed {
    background-color: var(--xpassed-bg);
}

.rerun {
    background-color: var(--rerun-bg);
}

.summary {
    background-color: var(--container-bg);
    color: var(--text-color);
}

.environment th, .environment td {
    background-color: var(--container-bg);
    color: var(--text-color);
}

.filters {
    background-color: var(--container-bg);
    color: var(--text-color);
}

.filters input, .filters select {
    background-color: var(--background-color);
    color: var(--text-color);
    border-color: var(--border-color);
}

.filters label {
    color: var(--text-color);
}

/* Make filter span text white while keeping colored backgrounds */
.filters .failed, .filters .passed, .filters .skipped, 
.filters .error, .filters .xfailed, .filters .xpassed, 
.filters .rerun, .filters .retried {
    color: var(--text-color) !important;
}

a {
    color: var(--link-color);
}

a:hover {
    color: var(--link-hover-color);
}

.collapsed .collapsible {
    background-color: var(--collapsed-bg);
}

.empty {
    color: var(--empty-color);
}
</style>
"""


def pytest_html_results_summary(prefix: list[str], summary: list[str], postfix: list[str]) -> None:
    """Embed the dark theme CSS when dark_mode is enabled."""
    if _pytest_config is not None:
        dark_mode_enabled = {"true": True, "false": False}[
            (_pytest_config.getini("dark_mode") or "false").lower()
        ]
        if dark_mode_enabled:
            prefix.append(DARK_CSS)
