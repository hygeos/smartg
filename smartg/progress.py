"""Progress bar utilities for SMART-G.

Provides a unified :func:`progress` factory that returns a progress
bar adapted to the current execution environment:

* ``notebook`` mode -- uses ``ipywidgets`` (``FloatProgress``) when
    running inside an IPython/Jupyter kernel.
* ``progressbar2`` mode -- uses the ``progressbar2`` library when
    available.
* ``progressbar`` mode -- falls back to the legacy ``progressbar``
    library.

The active mode is selected automatically at import time. Callers can
also request an invisible (no-op) progress bar via ``activate=False``.

Key Functions
-------------
progress
    Create a progress-bar adapter for the active runtime
    environment.
"""

from __future__ import annotations

from typing import Any

from smartg.typing import RealNumber

FloatProgress: Any | None = None
Label: Any | None = None
Box: Any | None = None
Layout: Any | None = None
display: Any | None = None
FormatLabel: Any | None = None
ProgressBar: Any | None = None
ETA: Any | None = None
Percentage: Any | None = None
Bar: Any | None = None
WidgetBase: Any | None = None
mode: str
try:
    from IPython.core.getipython import get_ipython

    cfg = get_ipython()
    if cfg is None:
        raise NameError("Not running inside an IPython kernel")
    from IPython.display import display
    from ipywidgets import Box, FloatProgress, Label, Layout

    mode = "notebook"
except (NameError, ImportError):
    try:
        from progressbar import ETA, Bar, FormatLabel, Percentage, ProgressBar

        mode = "progressbar2"
    except ImportError:
        from progressbar import ETA, Bar, Percentage, ProgressBar
        from progressbar.widgets import WidgetBase

        mode = "progressbar"


def progress(
    vmax: RealNumber,
    activate: bool = True,
) -> (
    ProgressInvisible
    | ProgressNotebook
    | ProgressProgressbar2
    | ProgressProgressbar
):
    """Create a progress-bar adapter for the active runtime environment.

    Parameters
    ----------
    vmax : int or float
        Maximum value shown by the progress bar.
    activate : bool, optional
        If ``False``, return a no-op progress object.

    Returns
    -------
    ProgressInvisible or ProgressNotebook or ProgressProgressbar2 or
    ProgressProgressbar
        A progress adapter exposing ``update(value, message='')`` and
        ``finish(message='')``.
    """
    if not activate:
        return ProgressInvisible()
    elif mode == "notebook":
        return ProgressNotebook(vmax)
    elif mode == "progressbar2":
        return ProgressProgressbar2(vmax)
    elif mode == "progressbar":
        return ProgressProgressbar(vmax)
    else:
        raise ValueError("Invalid mode " + mode)


class ProgressInvisible(object):
    """No-op progress adapter for disabled progress reporting."""

    def update(self, value: RealNumber, message: str = "") -> None:
        """Ignore progress updates.

        Parameters
        ----------
        value : int or float
            Current progress value.
        message : str, optional
            Optional status message.
        """
        pass

    def finish(self, message: str = "") -> None:
        """Ignore completion notifications.

        Parameters
        ----------
        message : str, optional
            Optional final status message.
        """
        pass


class ProgressNotebook(object):
    """Notebook progress adapter based on ``ipywidgets`` widgets."""

    def __init__(self, vmax: RealNumber) -> None:
        """Initialize a notebook progress bar and display it.

        Parameters
        ----------
        vmax : int or float
            Maximum value shown by the progress bar.
        """
        self.vmax = vmax
        self.pbar = FloatProgress(min=0, max=vmax)  # type: ignore
        self.label = Label()  # type: ignore
        self.layout = Layout(
            display="flex",  # type: ignore
            align_items="center",
        )
        self.box = Box(
            [self.pbar, self.label],  # type: ignore
            layout=self.layout,
        )
        display(self.box)  # type: ignore

    def update(self, value: RealNumber, message: str = "") -> None:
        """Update the notebook progress value and label text.

        Parameters
        ----------
        value : int or float
            New progress value. Values above ``vmax`` are clamped.
        message : str, optional
            Text displayed next to the widget.
        """

        value = min(value, self.vmax)  # don't exceed max
        self.pbar.value = value
        self.label.value = message

    def finish(self, message: str = "") -> None:
        """Mark the notebook progress bar as complete.

        Parameters
        ----------
        message : str, optional
            Final status message.
        """
        self.pbar.bar_style = "success"
        self.pbar.value = self.vmax
        self.label.value = message


class ProgressProgressbar2(object):
    """Terminal progress adapter using the ``progressbar2`` package."""

    def __init__(self, max: RealNumber) -> None:
        """Initialize a ``progressbar2`` progress bar.

        Parameters
        ----------
        max : int or float
            Maximum value shown by the progress bar.
        """
        self.max = max
        self.label = FormatLabel("")  # type: ignore
        self.pbar = ProgressBar(
            widgets=[
                self.label,
                " ",
                Percentage(),  # type: ignore
                Bar(),  # type: ignore
                ETA(),  # type: ignore
            ],
            max_value=max,
        ).start()

    def update(self, value: RealNumber, message: str = "") -> None:
        """Update progress value and message for ``progressbar2``.

        Parameters
        ----------
        value : int or float
            New progress value. Values above ``max`` are clamped.
        message : str, optional
            Text rendered by the label widget.
        """

        value = min(value, self.max)  # don't exceed max
        self.label.format = message
        self.pbar.update(value)

    def finish(self, message: str = "") -> None:
        """Finalize the ``progressbar2`` progress bar.

        Parameters
        ----------
        message : str, optional
            Final status message.
        """
        self.pbar.finish()
        self.label.format = message


class ProgressProgressbar(object):
    """Terminal progress adapter using legacy ``progressbar`` API."""

    def __init__(self, max: RealNumber) -> None:
        """Initialize a legacy ``progressbar`` progress bar.

        Parameters
        ----------
        max : int or float
            Maximum value shown by the progress bar.
        """

        class Custom(WidgetBase):  # type: ignore
            def update(self, bar: Any) -> str:
                try:
                    return self.__text
                except AttributeError:
                    return ""

            def set(self, text: str) -> None:
                self.__text = text

        self.max = max
        self.custom = Custom()
        self.pbar = ProgressBar(
            widgets=[
                self.custom,
                " ",
                Percentage(),  # type: ignore
                Bar(),  # type: ignore
                ETA(),  # type: ignore
            ],
            maxval=max,
        ).start()

    def update(self, value: RealNumber, message: str = "") -> None:
        """Update progress value and custom text.

        Parameters
        ----------
        value : int or float
            New progress value. Values above ``max`` are clamped.
        message : str, optional
            Text shown before the percentage indicator.
        """

        value = min(value, self.max)  # don't exceed max
        self.custom.set(message)
        self.pbar.update(value)

    def finish(self, message: str = "") -> None:
        """Finalize the legacy progress bar.

        Parameters
        ----------
        message : str, optional
            Final status message.
        """
        self.pbar.finish()
        self.custom.set(message)
