#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Common miscellaneous tools
'''

import sys
sys.path.append('..')
import numpy as np
from profile.profil import REPTRAN
from luts import LUT, merge, MLUT
import scipy.constants as cst
from pylab import *



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
    return 1./pi*(mu*L).reduce(trapz, zenith, x=-mu[:]).reduce(trapz, azimuth, x=phi[:])

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
    return 1./pi*(L).reduce(trapz, zenith, x=-mu[:]).reduce(trapz, azimuth, x=phi[:]) 


def Int(wi, wb, ex, we, dl, M=None, field=None, lim=[400.,700.]):
    '''
    all input vectors have same length, coming from REPTRAN
    wi : input wavelengths of internal bands (nm)
    wb : input wavelengths of bands (nm)
    ex : input extra-terrestrial irradiances at internal bands (W m-2 nm-1)
    we : input weights of internal bands
    dl : input bandwidths of bands (nm)
    M  : optional LUT or MLUT with 3D (lambda,phi,theta) or 1D (lambda) radiative field to spectrally integrate
    field : optional field of MLUT (ex: 'I_up (TOA'), if None, the extraterrestrial irradiance is integrated
    lim: spectral boundaries for integration
    
    returns
    spectrally integrated intensity and averaged intensity
    spectrally integrated daily quanta and average daily quanta
    '''
    ok=np.where((wb.data >=lim[0]) & (wb.data <lim[1]))[0]
    if (M != None) :
        if (field != None) :
            L = M[field]
            tab = L.data
        else : 
            tab = M.data
        
        if tab.ndim == 3 :
            R = np.rollaxis(tab,0,3)
            E = LUT(sum(R[:,:,ok] * ex.data[ok] * we.data[ok] * dl.data[ok], axis=2), \
                axes=[L.axes[1], L.axes[2]], desc='E_'+L.desc, \
                names=[L.names[1], L.names[2]], attrs=L.attrs)
            Q = LUT(sum(R[:,:,ok] * wi.data[ok] * ex.data[ok] * we.data[ok] * dl.data[ok], axis=2) , \
                axes=[L.axes[1], L.axes[2]], desc='Q_'+L.desc, \
                names=[L.names[1], L.names[2]], attrs=L.attrs)
        else:

            E = sum(tab[ok] * ex.data[ok] * we.data[ok] * dl.data[ok])
            Q = sum(tab[ok] * wi.data[ok] * ex.data[ok] * we.data[ok] * dl.data[ok])
    else:

        E = sum(ex.data[ok] * we.data[ok] * dl.data[ok])
        Q = sum(ex.data[ok] * wi.data[ok] * we.data[ok] * dl.data[ok])
    
    norm = sum(we.data[ok] * dl.data[ok])
    E *= 1e-3 # from mW/m2/nm to W/m2/nm 
    Eavg = E/norm
    Q *= 1e-12/(cst.h*cst.c*cst.Avogadro) * 3600*24 # from mW/m2/nm*nm to W/m2/m*m (1e-12) and then to
        # Einstein/m2/day

    Qavg = Q/norm
        
    return E, Eavg, Q, Qavg
    
def nInt(wi, wb, ex, we, dl, M=None, field=None, lim=[400.,700.]):
    '''
    all input vectors have same length, coming from REPTRAN
    wi : input wavelengths of internal bands (nm)
    wb : input wavelengths of bands (nm)
    ex : input extra-terrestrial irradiances at internal bands (W m-2 nm-1)
    we : input weights of internal bands
    dl : input bandwidths of bands (nm)
    M  : optional LUT or MLUT with 3D (lambda,phi,theta) or 1D (lambda) radiative field to spectrally integrate
    field : optional field of MLUT (ex: 'I_up (TOA'), if None, the extraterrestrial irradiance is integrated
    lim: spectral boundaries for integration
    
    returns
    spectrally integrated intensity and averaged intensity
    spectrally integrated daily quanta and average daily quanta
    '''
    ok=np.where((wb.data >=lim[0]) & (wb.data <lim[1]))[0]
    if (M != None) :
        if (field != None) :
            L = M[field]
            tab = L.data
        else : 
            tab = M.data
        
        if tab.ndim == 3 :
            R = np.rollaxis(tab,0,3)
            E = LUT(sum(R[:,:,ok] * ex.data[ok] * we.data[ok] * dl.data[ok], axis=2), \
                axes=[L.axes[1], L.axes[2]], desc='E_'+L.desc, \
                names=[L.names[1], L.names[2]], attrs=L.attrs)
            Q = LUT(sum(R[:,:,ok] * wi.data[ok] * ex.data[ok] * we.data[ok] * dl.data[ok], axis=2) , \
                axes=[L.axes[1], L.axes[2]], desc='Q_'+L.desc, \
                names=[L.names[1], L.names[2]], attrs=L.attrs)
        else:

            E = sum(tab[ok] * ex.data[ok] * we.data[ok] * dl.data[ok])
            Q = sum(tab[ok] * wi.data[ok] * ex.data[ok] * we.data[ok] * dl.data[ok])
    else:

        E = sum(ex.data[ok] * we.data[ok] * dl.data[ok])
        Q = sum(ex.data[ok] * wi.data[ok] * we.data[ok] * dl.data[ok])
    
    norm = sum(we.data[ok] * dl.data[ok])
    E *= 1e-3 # from mW/m2/nm to W/m2/nm 
    Eavg = E/norm
    Q *= 1e-12/(cst.h*cst.c*cst.Avogadro) * 3600*24 # from mW/m2/nm*nm to W/m2/m*m (1e-12) and then to
        # Einstein/m2/day

    Qavg = Q/norm
        
    return E, Eavg, Q, Qavg


def SpecInt(wi, wb, ex, we, dl, M=None, field=None, lim=[400.,700.]):
    '''
    all input vectors have same length, coming from REPTRAN
    wi : input wavelengths of internal bands (nm)
    wb : input wavelengths of bands (nm)
    ex : input extra-terrestrial irradiances at internal bands (W m-2 nm-1)
    we : input weights of internal bands
    dl : input bandwidths of bands (nm)
    M  : optional LUT or MLUT with 3D (lambda,phi,theta) or 1D (lambda) radiative field to spectrally integrate
    lim: spectral boundaries for integration
    
    returns
    spectrally integrated intensity
    '''
    ok=np.where((wb.data >=lim[0]) & (wb.data <lim[1]))[0]
    if (M != None) :
        if (field != None) :
            L = M[field]
            tab = L.data
        else :
            tab = M.data

        if tab.ndim == 3 :
            R = np.rollaxis(tab,0,3)
            E = LUT(sum(R[:,:,ok] * we.data[ok] * dl.data[ok], axis=2), \
                axes=[L.axes[1], L.axes[2]], desc=L.desc, \
                names=[L.names[1], L.names[2]], attrs={'LAMBDA':lim[0]})
        else:
            E = sum(tab[ok] * we.data[ok] * dl.data[ok])
    else:
        E = sum(we.data[ok] * dl.data[ok])

    norm = sum(we.data[ok] * dl.data[ok])
    Eavg = E/norm

    return E, Eavg


def SpecInt2(wi, wb, ex, we, dl, M=None, field=None, Irradiance=False, PlaneIrr=True, lim=None, DL=None):
    Eavg=[] 
    if DL == None:
        wu = np.unique(wb.data)
        DL = 1e-5
    else:
        if lim == None:
            lim = [wi.data.min(),wi.data.max()]
        wu = np.linspace(lim[0],lim[1]-DL,endpoint=True,num=(lim[1]-lim[0])/DL)
    for linf in wu:
        E1,E2 = SpecInt(wi, wb, ex, we, dl, M=M, field=field,lim=[linf,linf+DL])
        if not Irradiance :
            Eavg.append(E2)
        else:
            if PlaneIrr:
                Eavg.append(Irr(E2))
            else:
                Eavg.append(SpherIrr(E2))
    Mavg = merge(Eavg, ['LAMBDA'])
    return  Mavg

def Int2(wi, wb, ex, we, dl, M=None, field=None, lim=[400.,700.], DL=1., Irradiance=False, PlaneIrr=True):
    l=[]
    Qavg=[]
    Eavg=[]
    Qint=[]
    Eint=[]   
    for linf in np.linspace(lim[0],lim[1]-DL,endpoint=True,num=(lim[1]-lim[0])/DL):
        E1,E2,Q1,Q2 = Int(wi, wb, ex, we, dl, M=M, field=field,lim=[linf,linf+DL])
        l.append(linf+DL/2.)
        if not Irradiance :
            Eint.append(E1)
            Eavg.append(E2)
            Qint.append(Q1)
            Qavg.append(Q2)
        else:
            if PlaneIrr:
                Eint.append(Irr(E1))
                Eavg.append(Irr(E2))
                Qint.append(Irr(Q1))
                Qavg.append(Irr(Q2))
            else:
                Eint.append(SpherIrr(E1))
                Eavg.append(SpherIrr(E2))
                Qint.append(SpherIrr(Q1))
                Qavg.append(SpherIrr(Q2))
                
    return l, Eint, Eavg, Qint, Qavg
