"""GPU-free tests of the progress bars of smartg.progress.

Outside a notebook the terminal bar of progressbar2 must show the
messages of Smartg.run, including a '%', and a terminal IPython shell
must not be taken for a notebook, where the widgets never update.
"""

import importlib
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

import smartg.progress


@pytest.fixture
def progress_module() -> Iterator[None]:
    """Choose the progress mode again after the test."""
    yield
    importlib.reload(smartg.progress)


@pytest.mark.parametrize(
    "message", ["Launched 50 photons", "Done! | Received 42.0% of photons"]
)
def test_terminal_bar_shows_messages(message: str) -> None:
    """The bar renders the message, and a '%' in it is plain text."""
    assert smartg.progress.mode == "progressbar2"
    pbar = smartg.progress.progress(100)
    assert isinstance(pbar, smartg.progress.ProgressProgressbar2)
    for finish in (False, True):
        if finish:
            pbar.finish(message)
        else:
            pbar.update(50, message)
        # the first widget of the bar is the label it draws
        bar = pbar.pbar
        assert bar.widgets[0](bar, bar.data()) == message


def test_terminal_ipython_is_not_a_notebook(
    progress_module: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal IPython shell gets the terminal bar, a kernel not."""
    import IPython.core.getipython
    from IPython.terminal.interactiveshell import TerminalInteractiveShell

    # a shell that is not initialized leaves the session unchanged
    terminal = TerminalInteractiveShell.__new__(TerminalInteractiveShell)
    for shell, mode in [
        (terminal, "progressbar2"),
        (SimpleNamespace(kernel=object()), "notebook"),
    ]:
        monkeypatch.setattr(
            IPython.core.getipython, "get_ipython", lambda s=shell: s
        )
        importlib.reload(smartg.progress)
        assert smartg.progress.mode == mode
