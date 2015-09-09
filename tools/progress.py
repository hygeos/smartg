try:
    import warnings
    warnings.simplefilter(action = "ignore", category = FutureWarning)
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
    def __init__(self, max, opt):
        '''
        Initialize the progress bar object
        max: maximum value of the progress bar
        opt: whether to show a progress bar (True/False)
             or a Queue object to store the progress as (max_value), then (current_value, message)
        '''
        self.max = max

        # determine mode:
        #    - 'queue': store progress in the queue
        #    - 'off': no progress bar (opt == False)
        #    - 'notebook': ipython notebook mode (opt == True)
        #    - 'console': console mode (opt == True)

        if not isinstance(opt, bool):
            self.mode = 'queue'
            self.queue = opt
        elif not opt:
            self.mode = 'off'
        elif ipython_available:
            try:
                self.pbar = FloatProgress(min=0, max=max)
                self.mode = 'notebook'
            except RuntimeError:
                self.mode = 'console'
        else: self.mode = 'console'

        if self.mode == 'notebook':
            display(self.pbar)
        elif self.mode == 'console':
            self.custom = Custom()
            self.pbar = ProgressBar(widgets=[self.custom, ' ', Percentage(), Bar(), ETA()], maxval=max).start()
        elif self.mode == 'queue':
            self.queue.put(max)
        # else, no nothing (off)

    def update(self, value, message=''):

        value = min(value, self.max)  # don't exceed max

        if self.mode == 'notebook':
            self.pbar.value = value
            self.pbar.description = message
        elif self.mode == 'console':
            self.pbar.update(value)
            self.custom.set(message)
        elif self.mode == 'queue':
            self.queue.put((value, message))
        # else, no nothing (off)

    def finish(self, message=''):
        if self.mode == 'notebook':
            self.pbar.description=message
        elif self.mode == 'console':
            self.custom.set(message)
            self.pbar.finish()
        elif self.mode == 'queue':
            self.queue.put(message)
        # else, no nothing (off)




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
