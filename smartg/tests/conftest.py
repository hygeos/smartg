"""Configuration shared by the tests of this directory.

It declares the slow marker that separates the two tiers of the
IPRT phase B and phase 3 tests, and leaves the slow one out unless
it is asked for.
"""

import pytest


def pytest_configure(config):
    """Declare the slow marker so that pytest does not warn."""
    config.addinivalue_line(
        "markers",
        "slow: full photon count IPRT benchmark reproduction against MYSTIC",
    )


def pytest_collection_modifyitems(config, items):
    """Leave the slow tier out unless it is explicitly asked for.

    A '-m "not slow"' in the addopts of pytest.ini would be the natural
    place for this default, but that file is in .gitignore, so it would
    only ever apply to the machine it was written on. An explicit -m on
    the command line takes precedence, which is how 'pytest -m slow'
    reaches the full photon count runs.
    """
    if config.getoption("-m"):
        return

    selected = []
    deselected = []
    for item in items:
        if item.get_closest_marker("slow") is None:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
