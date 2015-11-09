#!/usr/bin/env python
# encoding: utf-8


import warnings
warnings.simplefilter("ignore",DeprecationWarning)
from pylab import figure
import numpy as np
np.seterr(invalid='ignore', divide='ignore') # ignore division by zero errors
from luts import plot_polar


def smartg_view(mlut, QU=False, field='up (TOA)'):
    '''
    visualization of a smartg MLUT

    Options:
        QU: show Q and U also
    '''

    I = mlut['I_' + field]
    Q = mlut['Q_' + field]
    U = mlut['U_' + field]

    # polarized reflectance
    IP = (Q*Q + U*U).apply(np.sqrt, 'Pol. ref.')

    # polarization ratio
    PR = 100*IP/I
    PR.desc = 'Pol. ratio. (%)'

    if QU:
        fig = figure(figsize=(9, 9))
        plot_polar(I,  0, rect='421', sub='423', fig=fig)
        plot_polar(Q,  0, rect='422', sub='424', fig=fig)
        plot_polar(U,  0, rect='425', sub='427', fig=fig)
        plot_polar(PR, 0, rect='426', sub='428', fig=fig, vmin=0, vmax=100)
    else:
        # show only I and PR
        fig = figure(figsize=(9, 4.5))
        plot_polar(I,  0, rect='221', sub='223', fig=fig)
        plot_polar(PR, 0, rect='222', sub='224', fig=fig, vmin=0, vmax=100)

