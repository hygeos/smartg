#!/usr/bin/env python
# vim:fileencoding=utf-8



from numpy import loadtxt, zeros, NaN, interp
from pathlib import Path


def a_w(lam):
    '''
    pure water absorption from Pope and Fry + Palmer and Williams
    '''

    this_dir = Path(__file__).parent
    file_pope97 = this_dir / 'data' / 'pope97.dat'
    file_palmer = this_dir / 'data' / 'palmer74.dat'

    # 1) Pope&Fry
    wl_popefry = loadtxt(file_pope97, skiprows=6, usecols=(0,))
    aw_popefry = loadtxt(file_pope97, skiprows=6, usecols=(1,))
    aw_popefry *= 100. # convert cm-1 => m-1

    # 2) Palmer&Williams
    wl_palmerw = loadtxt(file_palmer, skiprows=5, usecols=(0,))[::-1]
    aw_palmerw = loadtxt(file_palmer, skiprows=5, usecols=(1,))[::-1]
    aw_palmerw *= 100. # convert cm-1 => m-1

    if not isinstance(lam,float) :
        aw = zeros(lam.shape, dtype='float32') + NaN
        pf = lam < 715.  # use pope&fry
        aw[pf] = interp(lam[pf], wl_popefry, aw_popefry, right=NaN, left=NaN)
        aw[~pf] = interp(lam[~pf], wl_palmerw, aw_palmerw, right=NaN, left=NaN)
        return aw
    else :
        if lam<715 : return interp(lam, wl_popefry, aw_popefry, right=NaN, left=NaN)
        else: return interp(lam, wl_palmerw, aw_palmerw, right=NaN, left=NaN)


