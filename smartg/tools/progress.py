#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Progress bar utilities for SMART-G.

Provides a unified :func:`progress` factory that returns a progress bar
adapted to the current execution environment:

* ``notebook`` mode -- uses ``ipywidgets`` (``FloatProgress``) when running
  inside an IPython/Jupyter kernel.
* ``progressbar2`` mode -- uses the ``progressbar2`` library when available.
* ``progressbar`` mode -- falls back to the legacy ``progressbar`` library.

The active mode is selected automatically at import time. Callers can also
request an invisible (no-op) progress bar via ``activate=False``.
"""


FloatProgress = Label = Box = Layout = display = FormatLabel = None
ProgressBar = ETA = Percentage = Bar = WidgetBase = None
try:
    from IPython.core.getipython import get_ipython

    cfg = get_ipython()
    if cfg is None:
        raise NameError("Not running inside an IPython kernel")
    from ipywidgets import FloatProgress, Label, Box, Layout
    from IPython.display import display

    mode = "notebook"
except (NameError, ImportError):
    try:
        from progressbar import ProgressBar, ETA, Percentage, Bar, FormatLabel

        mode = "progressbar2"
    except ImportError:
        from progressbar import ProgressBar, ETA, Percentage, Bar
        from progressbar.widgets import WidgetBase

        mode = "progressbar"


def progress(vmax, activate=True):
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
    """
    A progress bar that does nothing
    """

    def update(self, value, message=""):
        pass

    def finish(self, message=""):
        pass


class ProgressNotebook(object):
    def __init__(self, vmax):
        """
        Initialize the progress bar object in the notebook

        vmax: maximum value of the progress bar
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

    def update(self, value, message=""):

        value = min(value, self.vmax)  # don't exceed max
        self.pbar.value = value
        self.label.value = message

    def finish(self, message=""):
        self.pbar.bar_style = "success"
        self.pbar.value = self.vmax
        self.label.value = message


class ProgressProgressbar2(object):
    def __init__(self, max):
        """
        Initialize the progress bar objectusing library 'progressbar2'
        max: maximum value of the progress bar
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

    def update(self, value, message=""):

        value = min(value, self.max)  # don't exceed max
        self.label.format = message
        self.pbar.update(value)

    def finish(self, message=""):
        self.pbar.finish()
        self.label.format = message


class ProgressProgressbar(object):
    def __init__(self, max):
        """
        Initialize the progress bar object using library 'progressbar'
        max: maximum value of the progress bar
        """

        class Custom(WidgetBase):  # type: ignore
            def update(self, bar):
                try:
                    return self.__text
                except AttributeError:
                    return ""

            def set(self, text):
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

    def update(self, value, message=""):

        value = min(value, self.max)  # don't exceed max
        self.custom.set(message)
        self.pbar.update(value)

    def finish(self, message=""):
        self.pbar.finish()
        self.custom.set(message)
