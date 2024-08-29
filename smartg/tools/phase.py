#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from numpy import sin, cos, pi
import numpy as np
#from luts.luts import LUT

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

    with np.errstate(divide='ignore', invalid='ignore'):
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

def calc_iphase(phase, wav_full, z_full, old_method=False):
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

    ipha_w = np.array([np.abs(wav - x).argmin() for x in wav_full], dtype='int32')
    if old_method:
        ipha_a = np.array([np.abs(altitude - x).argmin() for x in z_full], dtype='int32')
    else:
        ipha_a = get_ipha_a(z_full=z_full, z_pf=altitude, phase=phase)
    ipha = ipha_a[None,:] + ipha_w[:,None]*len(altitude)

    return (pha, ipha)


def get_ipha_a(z_full, z_pf, phase=None):
    grid_full = z_full
    grid_pf = z_pf
    size_layers_full = np.concatenate(( np.array([1e6]), np.abs(np.diff(grid_full))))
    size_layers_pf = np.concatenate(( np.array([1e6]), np.abs(np.diff(grid_pf))))

    nz_full = len(grid_full)
    nz_pf = len(grid_pf)

    zmin_print = [-1.]
    zmax_print = [-1.]

    ida = np.full(nz_full, -1, dtype=np.int32)
    for i_full in range (0, nz_full):
        idz_full = (nz_full-1)-i_full
        zmin_full = grid_full[idz_full]
        zmax_full = grid_full[idz_full] + size_layers_full[idz_full]

        # First find all the z_pf layers respecting the 2 conditions
        ida_tmp = []
        for i_pf in range(0, nz_pf):
            idz_pf = (nz_pf-1)-i_pf
            zmin_pf = grid_pf[idz_pf]
            zmax_pf = grid_pf[idz_pf] + size_layers_pf[idz_pf]
            cond_1 = zmin_pf < zmax_full
            cond_2 = zmax_pf > zmin_full
            if (cond_1 and cond_2):
                ida_tmp.append(idz_pf)

        n_ida_tmp = len(ida_tmp)
        # if only one pf layer respect the conditions take directly its index
        if n_ida_tmp == 1 :
            ida[idz_full] = ida_tmp[0]
        # if more than one, we have to look which pf layer fill best the layer of z_full 
        else:
            pfs_weight = np.zeros(n_ida_tmp)
            for k in range (0, n_ida_tmp):
                zmin_pf_k = grid_pf[ida_tmp[k]]
                zmax_pf_k = grid_pf[ida_tmp[k]] + size_layers_pf[ida_tmp[k]]
                pf_full_min = max(zmin_pf_k, zmin_full)
                pf_full_max = min(zmax_pf_k, zmax_full)
                if ( (phase == None) or (np.sum(phase[0,ida_tmp[k],0,:]) > 0.) ):
                    pfs_weight[k] = pf_full_max - pf_full_min
                else:
                    if (zmax_pf_k < 1e6) and ( (zmin_pf_k not in zmin_print) and (zmax_pf_k not in zmax_print)):
                        print("Warning: null phase matrix between ", zmin_pf_k, " and ", zmax_pf_k,
                              " detected! Please check pfgrid and/or grid (z_atm) values.")
                        zmin_print.append(zmin_pf_k)
                        zmax_print.append(zmax_pf_k)
                    pfs_weight[k] = (pf_full_max - pf_full_min)*1e-6
            ida[idz_full] = ida_tmp[np.argmax(pfs_weight)]
    return ida
