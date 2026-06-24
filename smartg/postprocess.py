#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Post-processing utilities for SMART-G output.

Irradiance computation from radiance (reflectance) fields.
'''


from numpy import pi, cos, trapezoid
from luts.luts import LUT, MLUT


def Irr(L, azimuth='Azimuth angles', zenith='Zenith angles'):
    '''
    Compute plane irradiance over dimensions (theta, phi)
    L: reflectance LUT
    phi: name of the azimuth axis in degrees
    theta: name of the zenith axis in degrees
    returns the irradiance value or a LUT for the remainding dimensions
    '''
    mu = (L.axis(zenith, aslut=True)*pi/180.).apply(cos)
    phi = L.axis(azimuth, aslut=True)*pi/180.
    return 1./pi*(mu*L).reduce(trapezoid, zenith, x=-mu[:]).reduce(trapezoid, azimuth, x=phi[:])


def SpherIrr(L, azimuth='Azimuth angles', zenith='Zenith angles'):
    '''
    Compute spherical irradiance over dimensions (theta, phi)
    L: reflectance LUT
    phi: name of the azimuth axis in degrees
    theta: name of the zenith axis in degrees
    returns the irradiance value or a LUT for the remainding dimensions
    '''
    mu = (L.axis(zenith, aslut=True)*pi/180.).apply(cos)
    phi = L.axis(azimuth, aslut=True)*pi/180.
    return 1./pi*(L).reduce(trapezoid, zenith, x=-mu[:]).reduce(trapezoid, azimuth, x=phi[:])


def reduce_Irr(m):
    res = MLUT()
    for d in m.datasets():
        if d.startswith('I_'):
            l = Irr(m[d])
            res.add_lut(l, desc=d.replace('I_', 'Pflux_'))
            l = SpherIrr(m[d])
            res.add_lut(l, desc=d.replace('I_', 'Sflux_'))
        if d.startswith('direct'):
            res.add_lut(m[d])
    return res
