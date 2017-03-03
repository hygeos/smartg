#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

try:
    cfg = get_ipython()
    from ipywidgets import FloatProgress, Label, Box, Layout
    from IPython.display import display
    mode = 'notebook'
except NameError:
    try:
        from progressbar import ProgressBar, ETA, Percentage, Bar, FormatLabel
        mode = 'progressbar2'
    except ImportError:
        from progressbar import ProgressBar, Widget, ETA, Percentage, Bar
        mode = 'progressbar'


def Progress(vmax, activate=True):
    if not activate:
        return Progress_invisible()
    elif mode == 'notebook':
        return Progress_notebook(vmax)
    elif mode == 'progressbar2':
        return Progress_progressbar2(vmax)
    elif mode == 'progressbar':
        return Progress_progressbar(vmax)
    else:
        raise('Invalid mode '+mode)


class Progress_invisible(object):
    '''
    A progress bar that does nothing
    '''
    def update(self, value, message=''):
        pass
    def finish(self, message=''):
        pass


class Progress_notebook(object):

    def __init__(self, vmax):
        '''
        Initialize the progress bar object in the notebook

        vmax: maximum value of the progress bar
        '''
        self.vmax = vmax
        self.pbar = FloatProgress(min=0, max=vmax)
        self.label = Label()
        self.layout = Layout(display='flex', align_items='center')
        self.box = Box([self.pbar, self.label], layout=self.layout)
        display(self.box)

    def update(self, value, message=''):

        value = min(value, self.vmax)  # don't exceed max
        self.pbar.value = value
        self.label.value = message

    def finish(self, message=''):
        self.pbar.bar_style = 'success'
        self.pbar.value = self.vmax
        self.label.value = message


class Progress_progressbar2(object):

    def __init__(self, max):
        '''
        Initialize the progress bar objectusing library 'progressbar2'
        max: maximum value of the progress bar
        '''
        self.max = max
        self.label = FormatLabel('')
        self.pbar = ProgressBar(widgets=[self.label, ' ', Percentage(), Bar(), ETA()], maxval=max).start()

    def update(self, value, message=''):

        value = min(value, self.max)  # don't exceed max
        self.label.format = message
        self.pbar.update(value)

    def finish(self, message=''):
        self.pbar.finish()
        self.label.format = message

class Progress_progressbar(object):

    def __init__(self, max):
        '''
        Initialize the progress bar object using library 'progressbar'
        max: maximum value of the progress bar
        '''
        class Custom(Widget):
            def update(self, bar):
                try: return self.__text
                except: return ''
            def set(self, text):
                self.__text = text

        self.max = max
        self.custom = Custom()
        self.pbar = ProgressBar(widgets=[self.custom, ' ', Percentage(), Bar(), ETA()], maxval=max).start()

    def update(self, value, message=''):

        value = min(value, self.max)  # don't exceed max
        self.custom.set(message)
        self.pbar.update(value)

    def finish(self, message=''):
        self.pbar.finish()
        self.custom.set(message)



if __name__ == '__main__':
    from time import sleep
    p = Progress(100, activate=True)
    for i in range(100):
        sleep(0.01)
        p.update(i, '[{}] running'.format(i))
    p.finish('done')
