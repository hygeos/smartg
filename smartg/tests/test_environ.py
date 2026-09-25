"""GPU-free tests of the context manager of smartg.environ."""

import os
from typing import cast

import pytest

from smartg.environ import modified_environ


def test_modified_environ_sets_and_restores() -> None:
    """A new variable is set in the block and removed after it."""
    assert "SMARTG_TEST_ENV" not in os.environ
    with modified_environ(SMARTG_TEST_ENV="1"):
        assert os.environ["SMARTG_TEST_ENV"] == "1"
    assert "SMARTG_TEST_ENV" not in os.environ


def test_modified_environ_refuses_non_str() -> None:
    """A value that is not a str raises and changes nothing."""
    with (
        pytest.raises(TypeError, match="SMARTG_TEST_INT must be a str"),
        modified_environ(SMARTG_TEST_STR="a", SMARTG_TEST_INT=cast(str, 0)),
    ):
        pass
    assert "SMARTG_TEST_STR" not in os.environ
    assert "SMARTG_TEST_INT" not in os.environ
