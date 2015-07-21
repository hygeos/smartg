try:
    from IPython.html.widgets import FloatProgress
    from IPython.display import display
    ipython_available = True
except:
    ipython_available = False
from progressbar import ProgressBar, Widget, ETA, Percentage, Bar

'''
A progress bar working both in console and notebook
'''

class Progress():
    def __init__(self, max):
        if ipython_available:
            try:
                self.pbar = FloatProgress(min=0, max=max)
                self.notebook_mode = True
            except RuntimeError:
                self.notebook_mode = False
        else: self.notebook_mode = False

        if self.notebook_mode:
            display(self.pbar)
        else:
            self.custom = Custom()
            self.pbar = ProgressBar(widgets=[self.custom, ' ', Percentage(), Bar(), ETA()], maxval=max).start()

    def update(self, value, text=''):

        if self.notebook_mode:
            self.pbar.value = value
            self.pbar.description = text
        else:
            self.pbar.update(value)
            self.custom.set(text)

    def finish(self, message=''):
        if self.notebook_mode:
            self.pbar.description=message
        else:
            self.custom.set(message)
            self.pbar.finish()


class Custom(Widget):
    '''
    A custom (console) ProgressBar widget to display arbitrary text
    '''
    def update(self, bar):
        try: return self.__text
        except: return ''
    def set(self, text):
        self.__text = text



if __name__ == '__main__':
    from time import sleep
    p = Progress(100)
    for i in xrange(100):
        sleep(0.01)
        p.update(i, 'running')
    p.finish('done')
