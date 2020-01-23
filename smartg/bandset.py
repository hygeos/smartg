#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import sys
sys.path.insert(0, '/home/did/RTC/SMART-G/')
from smartg.rrs import L2d_inv, is_odd
from luts.luts import LUT
import scipy.constants as cst
from scipy.interpolate import interp1d

class BandSet(object):
    def __init__(self, wav):
        '''
        Common objet for formatting input bands definition

        Valid inputs:
            * float
            * 1-d array
            * KDIS or REPTRAN IBANDS LIST

        Methods:
            __getitem__: returns wavelength
        '''
        try:
            self.use_reptran_kdis = hasattr(wav[0], 'calc_profile')
        except:
            self.use_reptran_kdis = False

        if self.use_reptran_kdis:
            self.wav = [x.w for x in wav]
            self.data = wav
        else:
            self.wav = wav
            self.data = None

        assert isinstance(self.wav, (float, list, np.ndarray))
        self.wav = np.array(self.wav, dtype='float32')
        self.scalar = (self.wav.ndim == 0)
        if self.scalar:
            self.wav = self.wav.reshape(1)
        self.size = self.wav.size

    def __getitem__(self, key):
        return self.wav[key]

    def __len__(self):
        return self.size

    def calc_profile(self, prof):
        '''
        calculate the absorption profile for each band
        '''
        tau_mol = np.zeros((self.size, len(prof.z)), dtype='float32')

        if self.use_reptran_kdis:
            for i, w in enumerate(self.data):
                tau_mol[i,:] = w.calc_profile(prof)

        return tau_mol


def spectral_grids(lmin, lmax, datas, dl=None, dls=None):
    '''
    inputs:
        lmin  : lambda min (nm)
        lmax  : lambda max (nm)
        datas : solar spectrum data
    
    keywords:
        dl : high spectral resolution for absorption features # nm (default, None, same as datas)
        dls: low  spectral resolution for scattering features # nm (default None), NWS=1
    '''
    ## Solar spectrum input data ##
    wl0 = datas[:,0]
    # from mW/m2/nm to photons/cm2/s/nm
    E0  = datas[:,1] * 1e-3 * 1e-4  / (cst.h *cst.c)  * (wl0*1e-9)
    
    ## High spectral resolution grid (for absorption features)
    if dl is None:
        # Solar grid
        ii = np.where(( wl0>=lmin) & (wl0 <=lmax))
        wl = wl0[ii]
        NW = wl.size
        # Solar resolution
        dl = (wl[-1]-wl[0])/wl.size
    else:
        NW  = int((lmax-lmin)/dl) + 1
        wl  = np.linspace(lmin, lmax, num=NW) # wavelength grid
        
    ## RRS excitation wavelength grid for 243°K
    wl_RRS, _ = L2d_inv(wl, 90., 243.)
    lmin_RRS  = wl_RRS.min()
    lmax_RRS  = wl_RRS.max()
    
    # Solar spectrum LUT building
    ii = np.where(( wl0>lmin_RRS) & (wl0 <lmax_RRS))
    Es_LUT = LUT(E0[ii], axes=[wl0[ii]], names=['wavelength'], desc='Es')
    
    # low spectral resolution for scattering computations # nm 
    if dls is not None:
        NWS = int((lmax_RRS - lmin_RRS)/dls) 
        NWS = NWS if is_odd(NWS) else NWS+1
    else : NWS=1
    wls = np.linspace(lmin_RRS, lmax_RRS, num=NWS)
    
    # parameters for 1D linear interpolation of wl in wls
    if NWS>1:
        f  =  interp1d(wls,np.linspace(0, NWS-1, num=NWS))
        iw =  f(wl)
        iwls_in = np.floor(iw).astype(np.int8)        # index of lower wls value in the wls array, 
        wwls_in = (iw-iwls_in).astype(np.float32)  # floating proportion between iwls and iwls+1
        # special case for NWS
        ii = np.where(iwls_in==(NWS-1))
        iwls_in[ii] = NWS-2
        wwls_in[ii] = 1.
    else:
        iwls_in = np.array([0], dtype=np.int8)
        wwls_in = np.array([0], dtype=np.float32)
    
    return wl, wls, wl_RRS, Es_LUT, iwls_in, wwls_in
