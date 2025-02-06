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
        return self.data[Idx(wl, fill_value='extrapolate')]


class Albedo_map(object):
    '''
    Albedo map of Albedo objects:
    A 2D horizontal map of spectral albedos can be constructed
    Spectral albedos are limited to a MAX_NREF=10 different kind, could be extended
    They should be defined using Albedo_cst, Albedo_spectrum or Albedo_speclib classes

    The horizontal grid for the 2D map of albedo is rectangular
    and x and y boundaries on the surface (in km) are encoded in monotonic np.arrays whose values 
    are the upper limit of the rectangles:
    if x = [x0, x1, x2, ..., xn], the the limits are [-Inf, x0], [x0, x1], ..., [xn-1, xn], 
    with xn big enough to be considered as +Inf 

    Albedo index (2D), x in km, y in km, Alist: list of Albedo objects
    we assigned each rectangle an index in Alb_list, in a 2D array of shape (len(x),len(y))
    Negative indices are for the surf properties.
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


