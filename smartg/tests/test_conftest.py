"""GPU-free test of the pytest plugin of smartg/conftest.py."""

import subprocess
import sys
from pathlib import Path


def test_tests_run_without_pytest_html() -> None:
    """The pytest-html hooks are optional: pytest runs without it."""
    tests = Path(__file__).parent
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-p", "no:html",
            "-p", "no:cacheprovider", "-o", "addopts=", "--collect-only",
            "-q", str(tests / "test_cdf.py"),
        ],
        capture_output=True,
        text=True,
        cwd=tests.parent.parent,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
