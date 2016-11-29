#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from numpy import sin, cos, pi
import numpy as np
from .luts import LUT

def fournierForand(ang, n, mu):
    '''
    Fournier-Forand phase function
    Parameters:
        ang: angle in radians
        n: index of refraction of the particles
        mu: slope parameter of the hyperbolic dustribution
    Normalized to 4pi
    See http://www.oceanopticsbook.info/view/scattering/the_fournierforand_phase_function
    '''
    v = (3-mu)/2
    delta = 4/( 3*(n-1)*(n-1) )*sin(ang/2)*sin(ang/2)
    delta180 = 4/( 3*(n-1)*(n-1) )*sin(pi/2)*sin(pi/2)

    res = 1/( 4*pi*(1-delta)*(1-delta)*(delta**v) )*( v*(1-delta) - (1-(delta**v)) + ( delta*(1-(delta**v)) - v*(1-delta) )*1/(sin(ang/2)*sin(ang/2)) ) + (1-(delta180**v))/(16*pi*(delta180-1)*(delta180**v)) * (3*cos(ang)*cos(ang) - 1)
    res *= 4*pi

    return res

def fournierForandB(n, mu):
    '''
    backscatter fraction of the Fournier-Forand phase function
    '''
    d90 = 4./(3.*(n-1.)**2)*(sin(pi/4.)**2)
    v = (3.-mu)/2.
    B   = 1 - (1 - d90**(v+1) - 0.5*(1-d90**v))/((1-d90)*d90**v)
    return B

def henyeyGreenstein(angle, g):
    '''
    Henyey-Greenstein phase function
    Parameters:
        angle: angle in radians
        g: asymmetry coefficient
           (0: isotropic ; 1: highly peaked)
    Normalized to 4pi
    See http://www.oceanopticsbook.info/view/scattering/the_henyeygreenstein_phase_function
    '''
    return (1 - g*g)/((1 + g*g - 2*g*cos(angle))**1.5)

def integ_phase(ang, pha):
    '''
    Integrate pha(ang)*sin(ang) along the last axis
    ang in radians
    pha: phase function, dim [...,ang]
    '''
    assert not np.isnan(pha).any()

    dtheta = np.diff(ang)
    pm1 = pha[...,:-1]
    pm2 = pha[...,1:]
    sin1 = np.sin(ang[:-1])
    sin2 = np.sin(ang[1:])

    return np.sum(dtheta*((sin1*pm1+sin2*pm2)/3. + (sin1*pm2+sin2*pm1)/6.), axis=-1)

def calc_iphase(phase, wav_full, z_full):
    '''
    calculate phase function indices
    phase is a LUT of shape [wav, z, stk, theta]

    returns (pha, ipha) where:
        * pha is an array reshaped from phase to [wav*z, stk, theta]
        * ipha is an array of phase function indices (starting from 0)
          in the full array [wav_full, z_full]
    '''
    wav = phase.axes[0]
    altitude = phase.axes[1]

    nwav, nz, nstk, ntheta = phase.shape
    pha = phase.data.reshape(nwav*nz, nstk, ntheta)

    ipha_w = np.array([np.abs(wav - x).argmin() for x in wav_full])
    ipha_a = np.array([np.abs(altitude - x).argmin() for x in z_full])
    ipha = ipha_a[None,:] + ipha_w[:,None]*len(altitude)

    return (pha, ipha)
