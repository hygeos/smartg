#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import numpy as np
from luts.luts import LUT, Idx

class Albedo_cst(object):
    '''
    Constant albedo (white)
    '''
    def __init__(self, alb):
        self.alb = alb

    def get(self, wl):
        alb = np.zeros(np.array(wl).shape, dtype='float32')
        alb[...] = self.alb
        return alb

class Albedo_speclib(object):
    '''
    Albedo from speclib
    (http://speclib.jpl.nasa.gov/)
    '''
    def __init__(self, filename):
        data = np.genfromtxt(filename, skip_header=26)
        # convert X axis from micrometers to nm
        # convert Y axis from percent to dimensionless
        self.data = LUT(data[:,1]/100., axes=[data[:,0]*1000.], names=['wavelength'])

    def get(self, wl):
        return self.data[Idx(wl, fill_value='extrapolate')]

class Albedo_spectrum(object):
    '''
    Albedo R(lambda)

    R spectral albedo, lam in nm
    '''
    def __init__(self, R, lam):
        self.data = LUT(R, axes=[lam], names=['wavelength'])

    def get(self, wl):
        return self.data[Idx(wl)]


class Albedo_map(object):
    '''
    Albedo map of Albedo objects:

    Albedo index (2D), x in km, y in km, Alist: list of Albedo objects
    '''
    def __init__(self, Ai, x, y, Alist):
        self.map = LUT(Ai, axes=[x, y], names=['X', 'Y'])
        self.list = Alist
        self.NALB = len(Alist)

    def get(self, wl):
        return np.stack([ALB.get(wl) for ALB in self.list]).T

    def get_map(self, x0, y0):
        return self.map[Idx(x0, round=True, fill_value='extrema'),
                        Idx(y0, round=True, fill_value='extrema')].astype(int)


