#!/usr/bin/env python
# -*- coding: utf-8 -*-

import geoclide as gc

import matplotlib.pyplot as plt
import numpy as np

from luts.luts import LUT, MLUT

from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as mp3d
from matplotlib import colors as mcolors

import re, six
from itertools import dropwhile

from scipy import interpolate

#TODO import below will be moved in the next major version
from smartg.geometry import Point, Vector, Normalize
from smartg.transform import Transform
from warnings import warn


def receiver_view(SMLUT, CAT = int(0), LOG_I=False, NAME_FILE = None, MTOA = 1320,
                  VMIN=None, VMAX=None, INT='none', W_VIEW = 'W'):

    '''
    Definition of receiver_view

    Display the distribution of the radiant flux at a given receiver in the relative
    coordinates i.e. the coordinates specific to the object (which move with the
    object if there is a use of transformation)

        ^ x
        |     Print with the following cordinate system
    y <--    

    SMLUT     : SMART-G return MLUT
    CAT       : By default = 0 (sum of all cats), else from cat 1 to 8
    LOG_I     : Enable log interval
    NAME_FILE : By default None. If not None create a pdf file in auxdata directory
                of the current print with the specified name
    MTOA      : Radiant exitance at TOA (Unit depending on W_VIEW, by default W/m2)
    VMIN      : Minimal distribution value (Unit depending on W_VIEW), not for log print
    VMAX      : Maximal distribution value (Unit depending on W_VIEW), not for log print
    INT       : Interpolations for imshow/matshow, i.e. nearest, bilinear, bicubic, ...
    W_VIEW    : Choices between "W" for Watt, "kW" for kiloWatt or "MW" for MegaWatt

    '''

    m = SMLUT['C_Receiver'][CAT,:,:]
    # Size of a Cell where the Cell surface = S_Cell*S_Cell
    S_Cell = float(SMLUT.attrs['S_Cell']) * 1e3 # mult by 1e3 to convert km to m
    wx = (SMLUT.axes['X_Cell_Index'].size * S_Cell)/ 2.
    wy = (SMLUT.axes['Y_Cell_Index'].size * S_Cell)/ 2.
    C_Surf = S_Cell*S_Cell

    if( W_VIEW == "W"):
        k = 1.; STRUNIT = "W";
    elif ( W_VIEW == "kW"):
        k = 1e-3; STRUNIT = "kW";
    elif ( W_VIEW == "MW"):
        k = 1e-6; STRUNIT = "MW";
    else :
        raise NameError('Unkonwn argument for W_VIEW!')

    plt.figure()

    if LOG_I == False :
        cax = plt.imshow((k*m*MTOA)/C_Surf, cmap=plt.get_cmap('jet'), interpolation=INT, \
                         vmin=VMIN, vmax=VMAX, extent = [wy,-wy,-wx,wx])
    else:
        m2 = m
        if (np.amin(m2) < 0.00001):
            valmin = 0.00001
        else:
            valmin = np.amin(m2)
            
        cax = plt.imshow((k*m*MTOA)/C_Surf, cmap=plt.get_cmap('jet'), \
                         norm=mcolors.LogNorm(vmin=valmin*MTOA, vmax=np.amax(m*MTOA)), \
                         interpolation=INT, extent = [wy,-wy,-wx,wx])

    cbar = plt.colorbar()
    cbar.remove()
    cbar = plt.colorbar(cax)
    cbar.set_label(r'Irradiance ('+STRUNIT+'.m$^{-2}$)', fontsize = 12)
    plt.xlabel(r'Position (m) in relative y axis')
    plt.ylabel(r'Position (m) in relative x axis')
    plt.title('Receiver surface')
    if (NAME_FILE is not None):
        plt.savefig(NAME_FILE + '.pdf')  



def cat_view(SMLUT, MTOA = 1320, NCL = "68%", UNIT = "FLUX_DENSITY", W_VIEW = "W", M_VIEW = "m",
             PRINT=True, ACC = 6, kdis_rep_bands=None):
    '''
    Definition of cat_view: The function take the photon weight collected by a receiver available from
    the MLUT returned by a SMART-G simulation, and normalize it to get results in term of flux,
    flux density or radiance in an other MLUT.

    ===ARGS:
    SMLUT          : SMART-G return MLUT (Multi-Layer Unit Tabular)
    MTOA           : Solar flux at TOA. If there is wl dim give an np.array with the flux in function of wl
    NCL            : Nominal Confidence Limit (for the error)
    UNIT           : Choice between 'FLUX' (Watt), 'FLUX_DENSITY' (Watt/meter²) and RADIANCE (Watt/meter²/sr)
    W_VIEW         : Choices between "W" for Watt, "kW" for kiloWatt or "MW" for MegaWatt
    M_VIEW         : Choices between "cm" for centimeter, "m" for meter,  "km" for kilometer, ...
    PRINT          : If True print results, if there is a dimension wl then print the spectraly integrated results
    ACC            : Accuracy, number of decimal points to show (integer) if print == True
    kdis_rep_bands : KDIS_IBAND_LIST or REPTRAN_IBAND_LIST object

    ===RETURN:
    output : return an MLUT with the intensity (flux, flux density or radiance) with the errors
    '''
    
    m = SMLUT

    # Initialize the output MLUT
    output = MLUT()
    
    # Add the Categories dimension to the output MLUT (See Moulana et al. 2019 for the 8 Categories)
    output.add_axis('Categories', m.axes['Categories'])
    
    # Parameters not dependant on the wavelength
    ALDEG = (float(m.attrs['ALDEG']))

    # Parameters needed in case kdis or reptran is used
    if (kdis_rep_bands is not None):
        _,wb,_,_,norm,norm_dl = kdis_rep_bands.get_weights(); wl_kdis_rep = wb.data
    
    # Check if there is a dimension wavelength
    isWaveAxis = 'wavelength' in m['wPhCats'].names
    
    # Fill needed parameters considering the case with and without the wl dimension
    if (isWaveAxis): NPH = m["norm_npho"][:]; NPH_int = float(m.attrs['NPHOTONS'])
    else : NPH = float(m.attrs['NPHOTONS'])
    
    # LUT with sum of photon weight (and squared weight) in function of Categories and (if there is wl dim) wevelength
    MF = m['wPhCats']; MF2 = m['wPhCats2']
    
    # The disired unit of measurement between Watt, kiloWatt, MegaWatt...
    if(W_VIEW == "uW"):     k = 1e6 ; STRUNIT = "microWatt"
    elif( W_VIEW == "mW"):  k = 1e3 ; STRUNIT = "milliWatt"
    elif( W_VIEW == "W"):   k = 1.  ; STRUNIT = "Watt"
    elif( W_VIEW == "kW"):  k = 1e-3; STRUNIT = "kiloWatt"
    elif( W_VIEW == "MW"):  k = 1e-6; STRUNIT = "MegaWatt"
    else : raise NameError('Unkonwn argument for W_VIEW!')

    # The disired unit of measurement of length (centimeter, meter, ...)
    if(M_VIEW == "mm"):     kl = 1e-3*1e-3; STRUNITL = "millimeter"
    elif( M_VIEW == "cm"):  kl = 1e-2*1e-2; STRUNITL = "centimeter"
    elif( M_VIEW == "dm"):  kl = 1e-1*1e-1; STRUNITL = "decimeter"
    elif( M_VIEW == "m"):   kl = 1.       ; STRUNITL = "meter"
    elif ( M_VIEW == "km"): kl = 1e3*1e3  ; STRUNITL = "kilometer"
    else : raise NameError('Unkonwn argument for M_VIEW!')

    if (UNIT == "FLUX"):
        cst = 1.*k; STRPRINT = "Flux in " + STRUNIT + " for each categories"; STRTY = "flux"
    elif (UNIT == "FLUX_DENSITY"):
        cst = (1.*k*kl)/(float(m.attrs['S_Receiver'])*1e6)
        STRPRINT = "Irradiance in " + STRUNIT + "/meter² for each categories"; STRTY = "irradiance"
    elif (UNIT == "RADIANCE"):
        cst = (1.*k*kl)/(float(m.attrs['S_Receiver'])*1e6)
        cst *= 2./(np.pi*(1 - np.cos(np.radians(2*ALDEG))))
        STRPRINT = "Radiance in " + STRUNIT + "/meter²/sr for each categories"; STRTY = "radiance"
    else:
        raise NameError('Unkonwn argument for UNIT!')
        
    if (isWaveAxis):
        cst*=float(m.attrs['n_cte'])
        cst*=np.sum(NPH) / NPH[:]
    else:
        cst*= float(m.attrs['n_cte'])
    
    # Normlalized intensity
    if (isWaveAxis):
        if (kdis_rep_bands is not None) :
            MF_N = (MF*cst*MTOA).reduce(np.sum, 'wavelength', grouping=wl_kdis_rep)
            MF_N_int = MF_N/norm
            MF2_N_int = (MF2*(cst*MTOA)*(cst*MTOA)).reduce(np.sum, 'wavelength', grouping=wl_kdis_rep)
            MF2_N_int /= norm
            MF_N_int = LUT(MF_N_int[:,:], axes=[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8],
                                          dtype=np.float64), np.array(MF_N.axes[1])],
                                          names=["Categories", "wavelength"])
            MF_N /= norm_dl
        else :
            MF_N = MF[:,:]*cst*MTOA
            MF2_N = MF2[:,:]*(cst*MTOA)*(cst*MTOA)
        MF_N = LUT(MF_N[:,:], axes=[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64), np.array(MF_N.axes[1])],
                              names=["Categories", "wavelength"])
        # Add the wl dimension in the output MLUT
        output.add_axis('wavelength', np.array(MF_N.axes[1]))
    else:
        MF_N = MF*cst*MTOA

    # Nominal confidence limit factor needed for the error calculation
    if (NCL == "68%"):      ld = 1
    elif (NCL == "87%"):    ld = 1.5
    elif (NCL == "95%"):    ld = 2
    elif (NCL == "99%"):    ld = 3
    elif (NCL == "99.99%"): ld = 4
    
    # Absolute error calculation and normalization, then convert to LUT
    if (isWaveAxis):
        sWl = len(m.axes['wavelength'][:])
        abs_err = np.zeros((9, sWl), dtype="float64")
        sum2Z   = np.zeros((9, sWl), dtype="float64")
        sumZ2   = np.zeros((9, sWl), dtype="float64")
        
        nBis = NPH[:] / (NPH[:] - 1)
        
        sum2Z[:,:] = (MF[:,:] * MF[:,:])/NPH[:]
        sumZ2 = MF2[:,:]
        abs_err[:,:] = (nBis * abs(sumZ2 - sum2Z))**0.5
        abs_err_LUT = LUT(abs_err[:,:],
                          axes=[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64), m.axes['wavelength']],
                          names=["Categories", "wavelength"])
        if (kdis_rep_bands is not None) :
            abs_err_LUT_N = (abs_err_LUT*cst*MTOA*ld).reduce(np.sum, 'wavelength', grouping=wl_kdis_rep)
            abs_err_LUT_N /= norm_dl
        else:
            abs_err_LUT_N = abs_err_LUT[:,:]*cst*MTOA*ld
        abs_err_LUT_N = LUT(abs_err_LUT_N[:,:], axes=[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64), np.array(abs_err_LUT_N.axes[1])],
                            names=["Categories", "wavelength"])

        abs_err_int = np.zeros(9, dtype="float64")
        sum2Z_int   = np.zeros(9, dtype="float64")
        sumZ2_int   = np.zeros(9, dtype="float64")

        nBis_int    = NPH_int / (NPH_int - 1)

        if (kdis_rep_bands is not None):
            MF_int = np.sum(MF_N_int[:,:], axis=1)
            MF2_int = np.sum(MF2_N_int[:,:], axis=1)
        else:
            MF_int  = np.sum(MF_N[:,:], axis=1)
            MF2_int = np.sum(MF2_N[:,:], axis=1)

        sum2Z_int[:] = (MF_int[:] * MF_int[:])/NPH_int
        sumZ2_int = MF2_int[:]
        abs_err_int[:] = (nBis_int * abs(sumZ2_int - sum2Z_int))**0.5
        abs_err_LUT_int = LUT(abs_err_int[:], axes=[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)], names=["Categories"])
        abs_err_LUT_N_int = abs_err_LUT_int

    else:
        abs_err = np.zeros(9, dtype="float64")
        sum2Z = np.zeros(9, dtype="float64")
        sumZ2 = np.zeros(9, dtype="float64")
        
        nBis = NPH / (NPH - 1)
        
        sum2Z[:] = (MF[:] * MF[:])/NPH
        sumZ2 = MF2[:]
        abs_err[:] = (nBis * abs(sumZ2 - sum2Z))**0.5
        abs_err_LUT = LUT(abs_err[:], axes=[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)], names=["Categories"])
        abs_err_LUT_N = abs_err_LUT*cst*MTOA*ld
        
    # Relative error calculation and LUT creation
    rel_err_LUT_N = (abs_err_LUT_N/MF_N) * 100
    
    # Create LUT for the number of photons in function of the Categories
    nb_Ph_LUT = LUT(m['cat_PhNb'][:], axes=[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)], names=["Categories"])
    
    # Add descriptions, then add LUTs to output MLUT
    MF_N.attrs['description'] = STRPRINT
    nb_Ph_LUT.attrs['description'] = "Number of photons in function of Categories"
    abs_err_LUT_N.attrs['description'] = 'Absolute error of ' + UNIT
    rel_err_LUT_N.attrs['description'] = 'Relative error in percentage of ' + UNIT
    
    output.add_lut(MF_N, desc=UNIT)
    output.add_lut(nb_Ph_LUT, desc='NbPhotons')
    output.add_lut(abs_err_LUT_N, desc='AbsoluteErr')
    output.add_lut(rel_err_LUT_N, desc='RelativeErr')

    if (kdis_rep_bands is not None):
        output.add_lut(MF_N_int, desc=UNIT+"_int")
        MF_N_tot = LUT(np.sum(MF_N_int[:,:], axis=1),
                       axes=[np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)],
                       names=["Categories"])
        output.add_lut(MF_N_tot, desc=UNIT+"_tot")
        output.add_lut(abs_err_LUT_int, desc="AbsoluteErr_tot")

    # If print == True ->
    if (PRINT == True):
        lP = ["(  D  )", "(  H  )", "(  E  )", "(  A  )", "( H+A )", "( H+E )", "( E+A )", "(H+E+A)"]
        intAcc = int(ACC)
        strAcc = str(intAcc)
        strAcc = "%." + strAcc + "f"
        
        mat = np.zeros((9,4), dtype="float64")
        if (isWaveAxis):
            if (kdis_rep_bands is not None): mat[:, 0] = np.sum(MF_N_int[:,:], axis=1)
            else: mat[:, 0] = np.sum(MF_N[:,:], axis=1) 
            mat[:, 1] = m['cat_PhNb'][:]                   # number of photons
            mat[:, 2] = abs_err_LUT_N_int[:]
            mat[:, 3] = (mat[:, 2]/mat[:,0])*100           # relative error
        else:
            mat[:, 0] = MF_N[:]          # normalized intensity
            mat[:, 1] = m['cat_PhNb'][:] # number of photons
            mat[:, 2] = abs_err_LUT_N[:] # absolute error
            mat[:, 3] = rel_err_LUT_N[:] # relative error
            
        print("**********************************************************")
        print(STRPRINT)
        print("**********************************************************")
        print("SUM_CATS      " + ": " + STRTY + "=", strAcc % (mat[0,0]), " number_ph=", np.uint64(mat[0,1]),
              " errAbs=", strAcc % (mat[0,2]), " err(%)=", strAcc % (mat[0,3]*ld))
        for i in range (0, 8):
            print("CAT",i+1, lP[i], ": " + STRTY + "=", strAcc % (mat[i+1,0]), " number_ph=", np.uint64(mat[i+1,1]),
                  " errAbs=", strAcc % (mat[i+1,2]), " err(%)=", strAcc % (mat[i+1,3]*ld))
    return output


def nopt_view(SMLUT, BACK=False, ACC = 6, NCL="68%", fl_TOA=None, NAATM=False):
    '''
    Definition of nopt_view

    In progress...
    '''
    m = SMLUT
    # Number of photons launched
    NPH = float(m.attrs['NPHOTONS'])
    # n/(n-1)
    NBIS = NPH/(NPH-1)

    if(fl_TOA is None):
        powc_H = m['powc_H'].data
    else:
        powc_H = 0.
        for i in range (0, len(fl_TOA)):
            powc_H += m['powc_H'].data[i]*fl_TOA[i]
        powc_H /= np.sum(fl_TOA)

    k = float(m.attrs['n_cte'])/powc_H

    intAcc = int(ACC)
    strAcc = str(intAcc)
    strAcc = "%." + strAcc + "f"
    if (NCL == "68%"): ld = 1
    elif (NCL == "87%"): ld = 1.5
    elif (NCL == "95%"): ld = 2
    elif (NCL == "99%"): ld = 3
    elif (NCL == "99.99%"): ld = 4

    print("**********************************************")
    print(" Optical Efficiencies")
    print("**********************************************")

    if(BACK == False): # Forward mode ->
        # Sum of weights
        # w0=wI, w1=wrhoM, w2=wrhoP, w3=wBM, w4=wBP, w5=wSM, w6=wSP, w7=wREC
        w0 = m['wLoss'][0]; w1 = m['wLoss'][1]; w2 = m['wLoss'][2];
        w3 = m['wLoss'][3]; w4 = m['wLoss'][4]; w5 = m['wLoss'][5]; 
        w6 = m['wLoss'][6]; w7 = m['cat_w'][2];
        # Sum of (weights²)
        w0_2 = m['wLoss2'][0]; w1_2 = m['wLoss2'][1]; w2_2 = m['wLoss2'][2];
        w3_2 = m['wLoss2'][3]; w4_2 = m['wLoss2'][4]; w5_2 = m['wLoss2'][5]; 
        w6_2 = m['wLoss2'][6]; w7_2 = m['cat_w2'][2];
        # (Sum of weights)² divided by the number of photons
        sum2Z = [(w0*w0)/NPH, (w1*w1)/NPH, (w2*w2)/NPH, (w3*w3)/NPH, (w4*w4)/NPH, \
                 (w5*w5)/NPH, (w6*w6)/NPH, (w7*w7)/NPH]
        # Sum of (weights²)
        sumZ2 = [w0_2, w1_2, w2_2, w3_2, w4_2, w5_2, w6_2, w7_2]
        dw = []
        for i in range (0, len(sum2Z)):
            dw_temp = ld*NBIS*(sumZ2[i]-sum2Z[i])**0.5
            dw.append(dw_temp)
        
        nopt=gc.clamp(k*w7, 0, 1);k_s = k/float(m.attrs['n_cos']);
        ncos=float(m.attrs['n_cos']);nsha=gc.clamp(k_s*w0, 0, 1);nref=gc.clamp(1-(w1/w0), 0, 1);
        nblo=gc.clamp(1-(w3/w2), 0, 1);nspi=gc.clamp(1-(w5/w4), 0, 1);natm=gc.clamp(w7/w6, 0, 1)

        d_nopt=abs(k)*dw[7];d_ncos=0.;d_nsha=abs(k_s)*dw[0];
        d_nref=abs(-1./w0)*dw[1] + abs(w1/w0**2)*dw[0]
        d_nblo=abs(-1./w2)*dw[3] + abs(w3/w2**2)*dw[2]
        d_nspi=abs(-1./w4)*dw[5] + abs(w5/w4**2)*dw[4]
        d_natm=abs(1./w6)*dw[7] + abs(w7/w6**2)*dw[6]

        print("nopt =", strAcc % nopt, ", errAbs =", strAcc % d_nopt, ", err% =", strAcc % ((d_nopt/nopt)*100))
        print("ncos =", strAcc % ncos, ", errAbs =", strAcc % d_ncos, ", err% =", strAcc % ((d_ncos/ncos)*100))
        print("nsha =", strAcc % nsha, ", errAbs =", strAcc % d_nsha, ", err% =", strAcc % ((d_nsha/nsha)*100))
        print("nref =", strAcc % nref, ", errAbs =", strAcc % d_nref, ", err% =", strAcc % ((d_nref/nref)*100))
        print("nblo =", strAcc % nblo, ", errAbs =", strAcc % d_nblo, ", err% =", strAcc % ((d_nblo/nblo)*100))
        print("nspi =", strAcc % nspi, ", errAbs =", strAcc % d_nspi, ", err% =", strAcc % ((d_nspi/nspi)*100))
        print("natm =", strAcc % natm, ", errAbs =", strAcc % d_natm, ", err% =", strAcc % ((d_natm/natm)*100))
    else: # Backward mode ->
        # Sum of weights
        # w0=wI, w1=wrhoM, w2=wREC
        w0=m['wLoss'][0];w1=m['wLoss'][1];w2=m['cat_w'][2];
        # Sum of (weights²)
        w0_2=m['wLoss2'][0];w1_2=m['wLoss2'][1];w2_2=m['cat_w2'][2];
        # (Sum of weights)² divided by the number of photons
        sum2Z = [(w0*w0)/NPH, (w1*w1)/NPH, (w2*w2)/NPH]
        # Sum of (weights²)
        sumZ2 = [w0_2, w1_2, w2_2]
        dw = []
        for i in range (0, len(sum2Z)):
            dw_temp = ld*NBIS*(sumZ2[i]-sum2Z[i])**0.5
            dw.append(dw_temp)
        nopt = gc.clamp(k*w2, 0, 1);ncos=float(m.attrs['n_cos']);nref=gc.clamp(1-(w1/w0), 0, 1);
        nsbsa = gc.clamp((k*w2)/(ncos*nref), 0, 1);

        d_nopt=abs(k)*dw[2];d_ncos = 0.;
        d_nref=abs(-1./w0)*dw[1] + abs(w1/w0**2)*dw[0]

        d_nsbsa=abs(k/(ncos*(1-(w1/w0))))*dw[2] + \
                 abs((k*w2)/(ncos*w0*(1-(w1/w0))**2))*dw[1] + \
                 abs((-k*w2*w1)/(ncos*w0*w0*(1-(w1/w0))**2))

        print("nopt =", strAcc % nopt, ", errAbs =", strAcc % d_nopt, ", err% =", strAcc % ((d_nopt/nopt)*100))
        print("ncos =", strAcc % ncos, ", errAbs =", strAcc % d_ncos, ", err% =", strAcc % ((d_ncos/ncos)*100))
        print("nref =", strAcc % nref, ", errAbs =", strAcc % d_nref, ", err% =", strAcc % ((d_nref/nref)*100))
        print("nsbsa =", strAcc % nsbsa, ", errAbs =", strAcc % d_nsbsa, ", err% =", strAcc % ((d_nsbsa/nsbsa)*100))

        if (NAATM):
            if(fl_TOA is None):
                naatm = m['n_aatm'].data
            else:
                naatm=0.
                for i in range (0, len(fl_TOA)):
                    naatm += m['n_aatm'].data[i]*fl_TOA[i]
                naatm /= np.sum(fl_TOA)
            print("naatm =", strAcc % naatm, " -> analytic approx of natm")


class Mirror(object):
    '''
    Definition of Mirror

    Glossy/specular material as pure and highly polished aluminum, silver
    behind glass mirror, ...

    reflectivity : The albedo of the object
    roughness    : Equal to alpha parameter according to Walter et al. 2007
    shadow       : Shadowing-Masking effect, by default not considered
    nind         : Relative refractive index air/material, by default
                   is None -> case of perfect mirror (nind = infinity)
    distribution : Two choices --> "Beckmann" or "GGX"
    '''
    def __init__(self, reflectivity = 1., roughness = 0., shadow = False, nind = None,
                 distribution = "Beckmann"):
        self.reflectivity = reflectivity
        self.roughness    = roughness
        self.shadow       = shadow
        if nind is None:
            self.nind     = -1
        else:
            self.nind     = nind
        if distribution == "Beckmann":
            self.distribution = 1
        elif distribution == "GGX":
            self.distribution = 2
        else:
            NameError('Please choose a distribution between str(Beckmann) or str(GGX)')

    def __str__(self):
        return 'Material -> Mirror : ' \
            'reflectivity=' + str(self.reflectivity) + ', roughness=' + str(self.roughness) \
            + ', shadow=' + str(self.shadow) + ', nind=' + str(self.nind) \
            + ', distribution=' + str(self.distribution)

class LambMirror(object):
    '''
    Definition of LambMirror

    Lambertian material, same probability of reflection in all the direction
    inside the hemisphere of the normal of the object surface

    reflectivity : The albedo of the object
    '''
    def __init__(self, reflectivity = 0.5):
        self.reflectivity = reflectivity
        

    def __str__(self):
        return 'Material -> Lambertian Mirror : ' \
            'reflectivity=' + str(self.reflectivity)

class Matte(object):
    '''
    Definition of Matte

    Diffuse material as Concrete, plastic, dust, ...

    reflectivity : The albedo of the object
    roughness    : Not yet available
    '''
    def __init__(self, reflectivity = 0., roughness = 0.):
        self.reflectivity = reflectivity
        self.roughness = roughness
        
    def __str__(self):
        return 'Material -> Matte : ' \
            'reflectivity=' + str(self.reflectivity) + ', roughness=' + str(self.roughness)

class Plane(object):
    '''
    Definition of Plane

    Plane constructed with 4 points : p1, p2, p3, p4

    p1 : x --> negative and y --> negative
    p2 : x --> positive and y --> negative
    p3 : x --> negative and y --> positive
    p4 : x --> positive and y --> positive
    '''
    def __init__(self, p1 = gc.Point(-0.5, -0.5, 0.), p2 = gc.Point(0.5, -0.5, 0.), \
                 p3 = gc.Point(-0.5, 0.5, 0.), p4 = gc.Point(0.5, 0.5, 0.)):
        # Avoid crash from old notebooks/scripts
        if isinstance(p1, Point): p1 = gc.Point(p1.x, p1.y, p1.z)
        if isinstance(p2, Point): p2 = gc.Point(p2.x, p2.y, p2.z)
        if isinstance(p3, Point): p3 = gc.Point(p3.x, p3.y, p3.z)
        if isinstance(p4, Point): p4 = gc.Point(p4.x, p4.y, p4.z)
        if (isinstance(p1, gc.Point) and isinstance(p2, gc.Point) and \
            isinstance(p3, gc.Point) and isinstance(p4, gc.Point)):
            if (  ( (p1.x == p3.x) and (p1.x < 0) )  and \
                  ( (p2.x == p4.x) and (p2.x > 0) )  and \
                  ( (p1.y == p2.y) and (p1.y < 0) )  and \
                  ( (p3.y == p4.y) and (p3.y > 0) )   ):
                self.p1 = p1
                self.p2 = p2
                self.p3 = p3
                self.p4 = p4
            elif ( (p1.x >= 0) or (p2.x <= 0) or (p1.y >= 0) or (p3.y >= 0) ):
                raise NameError( 'Those conditions must be filled! : ' + \
                                'p1.x < 0 , p1.y < 0 ,' + \
                                'p2.x > 0 , p2.y < 0 ,' + \
                                'p3.x < 0 , p3.y > 0 ,' + \
                                'p4.x > 0 , p4.y > 0' )
            elif ( (p1.x != p3.x) or (p2.x != p4.x) or (p1.y != p2.y) or (p3.y != p4.y) ):
                raise NameError('Your plane geometry must be at leat a rectangle!')
            else:
                NameError('Unknown error in Plane class!')
        else:
            raise NameError('All arguments must be Point type!')

    def __str__(self):
        return 'Coordinates of the Plane :\n' \
            '-> p1=(' + str(self.p1.x) + ', ' + str(self.p1.y) + ', ' + str(self.p1.z) + ')\n' + \
            '-> p2=(' + str(self.p2.x) + ', ' + str(self.p2.y) + ', ' + str(self.p2.z) + ')\n' + \
            '-> p3=(' + str(self.p3.x) + ', ' + str(self.p3.y) + ', ' + str(self.p3.z) + ')\n' + \
            '-> p4=(' + str(self.p4.x) + ', ' + str(self.p4.y) + ', ' + str(self.p4.z) + ')'

class Spheric(object):
    '''
    Definition of Spheric

    Sphere constructed with --->

    radius   : The radius of th e sphere
    radiusZ0 : Take into account all the sphere -> radiusZ0 = -radius
    radiusZ1 : Take into account all the sphere -> radiusZ1 = +radius
    phi      : The value of phi, 360 degrees is the value of a full sphere
    '''
    def __init__(self, radius = 10., z0 = None, z1 = None, phi = 360.):
        self.radius = radius
        self.phi = phi
        if (z0 == None):
            self.z0 = -1.*radius
        else:
            self.z0 = z0
        if (z1 == None):
            self.z1 = 1.*radius
        else:
            self.z1 = z1

    def __str__(self):
        return 'Sphere with the following caracteristics :\n' + \
            '-> radius = ' + str(self.radius) + '\n' + \
            '-> z0 = ' + str(self.z0) + '\n' + \
            '-> z1 = ' + str(self.z1) + '\n' + \
            '-> phi = ' + str(self.phi)


class Transformation():
    '''
    Definition of Transformation

    Enable to move, rotate a given object

    rotation      : 1D np array, 3 values for rotation in x, y and z (degree)
    translation   : 1D np array, 3 values for translation in x, y and z (kilometer)
    rotationOrder : Order of rotation, 6 choices : XYZ, XZY, YXZ, YZX, ZXY, ZYX
    '''
    def __init__(self, rotation = np.zeros(3, dtype=float), translation=np.zeros(3, dtype=float), \
                 rotationOrder = "XYZ"):
        self.rotation = rotation
        self.rotx = rotation[0]
        self.roty = rotation[1]
        self.rotz = rotation[2]
        self.rotOrder = rotationOrder
        self.translation = translation
        self.transx = translation[0]
        self.transy = translation[1]
        self.transz = translation[2]

    def __str__(self):
        return 'Transformation : rotation=(' + str(self.rotx) + ', ' + str(self.roty) + ', ' + \
            str(self.rotz) + ') and translation =(' + str(self.transx) + ', ' + \
            str(self.transy) + ', ' + str(self.transz) + ')'
    
class Entity(object):
    '''
    Definition of Entity

    This class enables the creation a 3D object

    entity : By default None. But useful in case where we need a copy of a given
             object
    name   : 2 choices --> reflector or receiver.
             If receiver is chosen, smartg will count the distribution flux
    TC     : Taille Cellules --> size of cells for the flux distribution (kilometer)
    bboxGPmin/max : in development...
    '''
    def __init__(self, entity = None, name="reflector", TC = 0.01, materialAV=Matte(), \
                 materialAR=Matte(), geo=Plane(), transformation=Transformation(), \
                 bboxGPmin = gc.Point(-100000., -100000., 0.), bboxGPmax = gc.Point(100000., 100000., 120.),
                 color = 'grey', alpha_color = 0.5):
        # Avoid crash from old notebooks/scripts
        if isinstance(bboxGPmin, Point): bboxGPmin = gc.Point(bboxGPmin.x, bboxGPmin.y, bboxGPmin.z)
        if isinstance(bboxGPmax, Point): bboxGPmax = gc.Point(bboxGPmax.x, bboxGPmax.x, bboxGPmax.y)
        if isinstance(entity, Entity) :
            self.name = entity.name; self.TC = entity.TC; self.materialAV = entity.materialAV
            self.materialAR = entity.materialAR; self.geo = entity.geo 
            self.transformation = entity.transformation
            self.bboxGPmin = entity.bboxGPmin; self.bboxGPmax = entity.bboxGPmax
            self.color = entity.color; self.alpha_color = alpha_color
        else:
            self.name = name
            self.TC = TC
            self.materialAV = materialAV
            self.materialAR = materialAR
            self.geo = geo
            self.transformation = transformation
            self.bboxGPmin = bboxGPmin
            self.bboxGPmax = bboxGPmax
            self.color = color
            self.alpha_color = alpha_color
        self.check = "Entity"

    def __str__(self):
        return 'The entity is a ' + str(self.name) + ' with the following carac:\n' + \
            str(self.materialAV) + '\n' + \
            str(self.geo) + '\n' + \
            str(self.transformation)
    
    def get_transformation(self):
        Trans = gc.get_translate_tf(gc.Vector(self.transformation.transx, self.transformation.transy, \
                                              self.transformation.transz))
        Rotx = gc.get_rotateX_tf(self.transformation.rotx)
        Roty = gc.get_rotateY_tf(self.transformation.roty)
        Rotz = gc.get_rotateZ_tf(self.transformation.rotz)

        # total tt of all transform together
        tt = None
        if   (self.transformation.rotOrder == "XYZ"): tt = Trans*Rotx*Roty*Rotz
        elif (self.transformation.rotOrder == "XZY"): tt = Trans*Rotx*Rotz*Roty
        elif (self.transformation.rotOrder == "YXZ"): tt = Trans*Roty*Rotx*Rotz
        elif (self.transformation.rotOrder == "YZX"): tt = Trans*Roty*Rotz*Rotx
        elif (self.transformation.rotOrder == "ZXY"): tt = Trans*Rotz*Rotx*Roty
        elif (self.transformation.rotOrder == "ZYX"): tt = Trans*Rotz*Roty*Rotx
        else: raise NameError('Unknown rotation order')

        return tt

class Heliostat(object):
    '''
    Definition of Heliostat
    
    This class enables the creation of heliostats i.e. a group of facets:

    POS           : Heliostat position stored in a Point class
    SPX, SPY      : The heliostat is splited in facets : SPx -> the number of time
                    we split in the x direction, SPy -> the same in y direction
    HSX           : Heliostat size in x direction
    HSY           : Heliostat size in y direction
    CURVE_FL      : Focal length : A curved heliostat is possible if curveFL is given
    REF           : Reflectivity of the heliostat
    ROUGH         : Roughness of the heliostat
    '''
    def __init__(self, POS = gc.Point(0., 0., 0.), SPX=int(2), SPY=int(2), HSX=0.02,
                 HSY=0.02, CURVE_FL=None, REF=1., ROUGH=0):
        # Avoid crash from old notebooks/scripts
        if isinstance(POS, Point): POS = gc.Point(POS.x, POS.y, POS.z)
        # Be sure that we split a heliostat by at least 2
        if (SPX*SPY < 2):
            raise Exception("The number of facets must be >= 2!")
        # Be sure that SPX and SPY are integer values
        if not ( isinstance(SPX, int) and isinstance(SPY, int) ):
            raise Exception("SPx and SPy must be integers")
        self.pos = POS
        self.sPx = SPX
        self.sPy = SPY
        self.hSx = HSX
        self.hSy = HSY
        self.curveFL = CURVE_FL
        self.ref = REF
        self.rough = ROUGH

    def __str__(self):
        return "POS=" + str(self.pos) + '; ' + "SPX=" + str(self.sPx) + '; ' + \
                "SPY=" + str(self.sPy) + '; ' + "HSX=" + str(self.hSx) + '; ' + \
                "HSY=" + str(self.hSy)  + '; ' + "CURVE_FL=" + str(self.curveFL) + \
                '; ' + "REF=" + str(self.ref) + '; ' + "ROUGH=" + str(self.rough)

class GroupE(object):
    '''
    Definition of GroupE

    This class is a group of entity objects

    LE   : List of entity objects
    BBOX : List with Pmin and Pmax (Point classes) to construct a custum
           bounding box, if None -> take Pmin and Pmax of LE[0]
    '''
    def __init__(self, LE=[Entity()], BBOX=None):
        self.le  = LE
        self.nob = len(LE)
        if BBOX is None:
            self.bboxGPmin = LE[0].bboxGPmin
            self.bboxGPmax = LE[0].bboxGPmax
        else:
            self.bboxGPmin = BBOX[0]
            self.bboxGPmax = BBOX[1]
        self.check = "GroupE"

def findRots(UI=None, UO=None, vecNF=None):
    '''
    Description of the function findRots:

    ===ARGS:
    UI    : Direction of the incoming ray or sun direction
    UO    : Opposite direction of the outcoming ray / direction from receiver to facet
    vecNF : Normal of the reflection surface, if given UI and UO are not needed

    ===RETURN:
    Return a list with rotation information, to reflect UI to -UO:
    list[0] -> rotYD : Rotation in Y direction
    list[0] -> rotZD : Rotation in Z direction
    list[2] -> TTT   : Rotation transform object 
    '''
    # 1)Find the normal of the facet but filled in a vector class
    if vecNF is not None: vNF = gc.Vector(vecNF)
    else: vNF = (UI + UO)*(-0.5)
    vNF = gc.normalize(vNF)
    vNF.z = np.clip(vNF.z, -1, 1) # Avoid nan value in next operations

    # 2) Apply the inverse rotation operations to find the necessary angles
    # 2.a) Initialisation
    loop=int(0); rotY=0; rotZ=0; opeZ=0;
    # The initial value of the facet normal is (0, 0, 1) but forced to (0, 0, 0)
    # to be sure to activate the while loop below
    vNF_initial = gc.Vector(0., 0., 0.)

    # 2.b) Rotations are found in the loop bellow, at the end we check if after applying
    #      the transform to the initial normal of the facet 'vNF_initial' we have the same
    #      value as the known well oriented facet normal 'vecNF'. If no rotation has been
    #      found an error message will appear
    while (abs(vNF_initial.x - vNF.x) > 1e-4 or abs(vNF_initial.y - vNF.y) > 1e-4 or 
           abs(vNF_initial.z - vNF.z) > 1e-4):
        loop += int(1)
        if loop > 4:
            raise NameError('No rotation has been found!')

        if (loop == 1):
            rotY = np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0): opeZ = 0
            else: opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = np.arccos(opeZ)
        elif(loop == 2):
            rotY = np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0): opeZ = 0
            else: opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = -np.arccos(opeZ)
        elif(loop == 3):
            rotY = -np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0): opeZ = 0
            else: opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = np.arccos(opeZ)
        elif(loop == 4):
            rotY = -np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0): opeZ = 0
            else: opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = -np.arccos(opeZ)
 
        rotYD = np.degrees(rotY); rotZD = np.degrees(rotZ);
        TTZ = gc.get_rotateZ_tf(rotZD); TTY = gc.get_rotateY_tf(rotYD);
        TTT = TTZ*TTY
        vNF_initial = gc.normalize(TTT(gc.Vector(0., 0., 1.)))

    return [rotYD, rotZD, TTT]

def generateMTF(HELIO=Heliostat(), PR = gc.Point(0., 0., 0.)):
    '''
    Under development...

    Giving a heliostat 'HELIO' and the position of a receiver 'PR'
    -->
    This function enables the computation of the transforms of each
    facets to curve the heliostat allowing facets to reflect in the
    center of the receiver (For the moment only on-axis method)
    '''
    # Avoid crash from old notebooks/scripts
    if isinstance(PR, Point): PR = gc.Point(PR.x, PR.y, PR.z)
    # Heliostat is splited in facets in x and y directions
    SPX = HELIO.sPx; SPY = HELIO.sPy;
    # Size in x and y of a given facet
    SFX = HELIO.hSx/SPX; SFY = HELIO.hSy/SPY
    wMx = SFX/2; wMy = SFY/2 # Size of a facet divided by 2

    POSH = gc.Point(HELIO.pos.x, HELIO.pos.y, HELIO.pos.z)
    APOSR = gc.Point(0., 0., 0.+(POSH - PR).Length())

    # Find the positions of facets and store them in matrix MPF[i][j]
    MPF = np.zeros((SPX, SPY), dtype="object") # Matrix of Point object of each facets
    for i in range (0, SPX):
        for j in range (0, SPY):
            MPF[i][j] = gc.Point(-(HELIO.hSx/2.) + (i*SFX) + wMx, -(HELIO.hSy/2.) + (j*SFY) + wMy, 0.)

    # Find transform in function of focal length (for the curve)
    MTF = np.zeros((SPX, SPY), dtype="object") # Matrix of Transform object of each facets
    for i in range (0, SPX):
        for j in range (0, SPY):
            UI = gc.Point(0., 0., 0.) - APOSR
            UI = gc.normalize(UI)
            UO = MPF[i][j] - APOSR
            UO = gc.normalize(UO)
            RINF  = findRots(UI=UI, UO=UO)
            MTF[i][j] = gc.Transform(RINF[2])

    return MTF

def generateLEfH(HELIO = Heliostat(), PR = None, THEDEG = 0., PHIDEG = 0., MTF=None):
    '''
    Definition of the function generateLEfH
    This function enables the conversion of an object heliostat to a list of 
    well oriented plane entity / facets to reflect to a given receiver

    ===ARGS:
    HELIO          : A heliostat class object
    THEDEG, PHIDEG : The theta and phi angles in degrees for the sun direction
    PR             : A class Point with the position of the receiver receiver
    MTF            : Under development
    
    ===RETURN:
    List of all well oriented plane entity / facets

    Convention -> here an example of a heliostat splited in 4 in x and y
    directions, the matrices used below follow this:

         j0   j1   j2   j3
       ---------------------
    i0 |f00 |f01 |f02 |f03 |    i, j           : matrix indices
       ---------------------    f00, f10, ...  : facet 0, 1, ...
    i1 |f10 |f11 |f12 |f13 |  
       -----------------------> y
    i2 |f20 |f21 |f22 |f23 |
       ----------|----------
    i3 |f30 |f31 |f32 |f33 |
       ----------|----------
                 x
    '''
    # Avoid crash from old notebooks/scripts
    if isinstance(PR, Point): PR = gc.Point(PR.x, PR.y, PR.z)
    # Be sure that the correct agrs have been given
    if not isinstance(HELIO, Heliostat):
        raise Exception("HELIO must be a Heliostat class!")
    if not isinstance(PR, gc.Point):
        raise Exception("The receiver position 'PR' must be a Point class!")

    # Direction of the sun (from (x,y,z) to (0,0,0))
    vSun = gc.ang2vec(THEDEG, PHIDEG, vec_view="nadir")
    # Heliostat is splited in facets in x and y directions
    SPX = HELIO.sPx; SPY = HELIO.sPy;
    # Size in x and y of a given facet
    SFX = HELIO.hSx/SPX; SFY = HELIO.hSy/SPY
    # Focal length or distance between heliostat and receiver
    FL = HELIO.curveFL
    # Position of the heliostat
    POSH = gc.Point(HELIO.pos.x, HELIO.pos.y, HELIO.pos.z)
    # Receiver assumed position or the assumed focal length point.
    # Needed to curve the heliostat
    if (FL is not None):
        APOSR = gc.Point(0., 0., 0.+FL)
    else:
        PHTEMP = gc.Point(POSH)
        DTEMP = (PHTEMP - PR).length()
        APOSR = gc.Point(0., 0., 0.+DTEMP)
    # For the bounding box
    bboxDist = np.sqrt(HELIO.hSx*HELIO.hSx + HELIO.hSy*HELIO.hSy)/2
        
    # Initialisation
    LF = [] # List of facets
    wMx = SFX/2; wMy = SFY/2 # Size of a facet divided by 2
    # Create one facet to be ready to clone other facets
    F1 = Entity(name = "reflector", \
                materialAV = Mirror(reflectivity = HELIO.ref, roughness = HELIO.rough), \
                materialAR = Matte(), \
                geo = Plane( p1 = gc.Point(-wMx, -wMy, 0.),
                             p2 = gc.Point(wMx, -wMy, 0.),
                             p3 = gc.Point(-wMx, wMy, 0.),
                             p4 = gc.Point(wMx, wMy, 0.) ), \
                transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                 translation = np.array([0., 0., 0.]) ))
    
    # Find the positions of facets and store them in matrix MPF[i][j]
    MPF = np.zeros((SPX, SPY), dtype="object") # Matrix of Point object of each facets
    for i in range (0, SPX):
        for j in range (0, SPY):
            MPF[i][j] = gc.Point(-(HELIO.hSx/2.) + (i*SFX) + wMx, -(HELIO.hSy/2.) + (j*SFY) + wMy, 0.)

    # Find transform in function of focal length (for the curve)
    if MTF is None:
        MTF = np.zeros((SPX, SPY), dtype="object") # Matrix of Transform object of each facets
        for i in range (0, SPX):
            for j in range (0, SPY):
                UI = gc.Point(0., 0., 0.) - APOSR
                UI = gc.normalize(UI)
                UO = MPF[i][j] - APOSR
                UO = gc.normalize(UO)
                RINF  = findRots(UI=UI, UO=UO)
                MTF[i][j] = gc.Transform(RINF[2])


    # Find the general heliostat rotation transform (like helistat is a unique facet)
    UI = gc.Vector(vSun.x, vSun.y, vSun.z); UO = POSH - PR;
    UI = gc.normalize(UI); UO = gc.normalize(UO);
    RINF2  = findRots(UI=UI, UO=UO)
    TTZY = RINF2[2]

    # Apply the general rotation transform to each facet point and then apply translation.
    # This gives the final position of each facet after rotation and translation of
    # the heliostat, stored in the matrix MPFAT 
    MPFAT = np.zeros((SPX, SPY), dtype="object") # equals to MPF after application of transform
    for i in range (0, SPX):
        for j in range (0, SPY):
            tempP = gc.Point(MPF[i][j])
            tempP = TTZY(tempP)
            tempP.x += POSH.x; tempP.y += POSH.y; tempP.z += POSH.z;
            MPFAT[i][j] = gc.Point(tempP)

    # Write the initial coordinate system in term of vectors (x, y and z)
    vecX = gc.Vector(1., 0., 0.); vecY = gc.Vector(0., 1., 0.); vecZ = gc.Vector(0., 0., 1.);

    # Apply the general rotation transform to find the new coordinate system of the heliostat
    vecX = TTZY(vecX); vecY = TTZY(vecY); vecZ = TTZY(vecZ);
    vecX = gc.normalize(vecX); vecY = gc.normalize(vecY); vecZ = gc.normalize(vecZ);

    # Create the transformation matrix allowing to move between the 2 coordinate systems
    nn1 = vecX; nn2 = vecY;nn3 = vecZ; 
    mm2 = np.zeros((4,4), dtype=np.float64)
    # Fill the transformation matrix (nn3 is the new z axis)
    mm2[0,0] = nn1.x ; mm2[0,1] = nn2.x ; mm2[0,2] = nn3.x ; mm2[0,3] = 0. ;
    mm2[1,0] = nn1.y ; mm2[1,1] = nn2.y ; mm2[1,2] = nn3.y ; mm2[1,3] = 0. ;
    mm2[2,0] = nn1.z ; mm2[2,1] = nn2.z ; mm2[2,2] = nn3.z ; mm2[2,3] = 0. ;
    mm2[3,0] = 0.    ; mm2[3,1] = 0.    ; mm2[3,2] = 0.    ; mm2[3,3] = 1. ;
    # Now create the transform object with the transformation matrix and its inverse
    mm2Inv = np.transpose(mm2)
    wTo = gc.Transform(m = mm2, mInv = mm2Inv) # move from world/initial to object∕new basis
    oTw = gc.Transform(m = mm2Inv, mInv = mm2) # move from object∕new to world/initial basis

    # The normal of the heliostat vecNH = z axis of the new coordinate system
    vecNH = gc.Vector(vecZ) # stored as a vector for transformation purposes
    for i in range (0, SPX):
        for j in range (0, SPY):
            # come back to the initial coordinate system
            vecNF = oTw(vecNH)
            # apply the transform of the facet to consider the curve effect
            vecNF = MTF[i][j](vecNF)
            # Now we return to the new coordinate system, which gives
            # then the normal of the facet (not heliostat) stored in MTF[i][j]
            vecNF = wTo(vecNF)
            vecNF = gc.normalize(vecNF)
    
            # Find the rotation transform
            RINF3 = findRots(vecNF=vecNF)

            # Once the rotation angles have been found, create the facet as entity object
            tempF1 = Entity(F1)
            tempF1.transformation = Transformation( rotation = np.array([0., RINF3[0], RINF3[1]]), \
                                                    translation = np.array([MPFAT[i][j].x, MPFAT[i][j].y, MPFAT[i][j].z]), \
                                                    rotationOrder = "ZYX")
            tempPP = gc.Point(POSH)
            
            tempF1.bboxGPmin = gc.Point(tempPP.x-bboxDist, tempPP.y-bboxDist, tempPP.z-bboxDist)
            tempF1.bboxGPmax = gc.Point(tempPP.x+bboxDist, tempPP.y+bboxDist, tempPP.z+bboxDist)
            LF.append(tempF1)

    return LF

def generateBox(dimXYZ=[0.05, 0.05, 0.05], pos=gc.Point(0., 0., 0.), matAV = "LambMirror",
        ref=[1., 1., 1., 1., 1., 1.], rough=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2], rotZ = 0., gap=0.0001,
        obj_type="environment", colors=None, alpha_color=None):
    """
    Description of the function :
    This function creates a box/building. The faces composing the box are
    following the convention used by Didier for 3D atm in SMART-G:
    Face 0 : Right. In the face -> (top Y+ ; right Z-)
    Face 1 : Left.  In the face -> (top Y+ ; right Z+)
    Face 2 : Back.  In the face -> (top Z- ; right X+)
    Face 3 : Front. In the face -> (top Z+ ; right X+)
    Face 4 : Top.   In the face -> (top Y+ ; right X+)
    Face 5 : Bot.   In the face -> (top Y+ ; right X-)
    
    !! BE CAREFUL !! origin is at the centre of Face 5, not a the center of the box !!
    
    ===ARGS:
    dimXYZ   : List with the dimensions of the box in x, y and z
    pos      : Point class with the localisation of the box, where origin is the center of F5
    matAV    : We can choose between "LambMirror" or "Mirror" for constant material in each faces,
               or a list of the material classes (Matte(), LambMirror() and Mirror()) for the 6 faces
    ref      : If matAV is "LambMirror" or "Mirror", we can specify the reflectivity of the faces
    rough    : If matAV is "Mirror" we can specify the roughness of the faces
    rotZ     : Global rotation of the box in the Z axis, in degrees (only global rotation in Z is enabled)
    gap      : gap to add in the global bounding box, can be sometimes useful for very small objects
    obj_type : choice between: 'environment', 'reflector' and 'receiver'
    colors   : list of str with face colors
    
    ===RETURN:
    Return a group of objects (i.e. a GroupE class composed of plane objects)
    """
    # Avoid crash from old notebooks/scripts
    if isinstance(pos, Point): pos = gc.Point(pos.x, pos.y, pos.z)
    # Material AV = front part (i.e. part outside the box) of Face 0 to Face 5,
    # back part (i.e. part inside the box) will be definite as matte (totally absorbant)
    matAVL = []
    if (matAV == "Mirror") :
        for i in range (0, 6):
            matAVL.append(Mirror(reflectivity = ref[i], roughness=rough[i]))
    elif (matAV == "LambMirror") :
        for i in range (0, 6):
            matAVL.append(LambMirror(reflectivity = ref[i]))
    else :
        matAVL = matAV

    # colors
    if colors is None : colors = ['grey', 'grey', 'grey', 'grey', 'grey', 'grey']
    if alpha_color is None : alpha_color = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    # === Commun parameters ===
    # Compute the half dimensions in X, Y and Z
    wMx = dimXYZ[0]/2.; wMy = dimXYZ[1]/2.; wMz = dimXYZ[2]/2.
    
    # With the global Z rotation, 4 translations are needed in the direction after the rotation, for Face 0 to 3 
    TT = gc.get_rotateZ_tf(rotZ)
    TX = gc.Vector(1., 0., 0.); TX = TT(TX); TX = gc.normalize(TX)*wMx
    TY = gc.Vector(0., 1., 0.); TY = TT(TY); TY = gc.normalize(TY)*wMy
    
    # Initialize a numpy array list of Points (p1 to p4 to construct a face) for all faces (from face 0 to 5)
    p1_F = np.empty(6, dtype=object); p2_F = np.empty(6, dtype=object); p3_F = np.empty(6, dtype=object); p4_F = np.empty(6, dtype=object)
    
    # Initialisze rotation needed to orient correctly each face
    rotX_F = np.zeros(6, dtype='float64'); rotY_F = np.zeros(6, dtype='float64')
    rotZ_F = np.full(6, rotZ) # for Z rotation it is the same value for all faces
    
    # Initialize translation variables of all faces
    transX_F = np.zeros(6, dtype='float64'); transY_F = np.zeros(6, dtype='float64'); transZ_F = np.zeros(6, dtype='float64')
    # === End commun parameters ===
    
    
    # Face 0 unique parameters
    p1_F[0] = gc.Point(-wMz, -wMy, 0.); p2_F[0] = gc.Point(wMz, -wMy, 0.); p3_F[0] = gc.Point(-wMz, wMy, 0.); p4_F[0] = gc.Point(wMz, wMy, 0.)
    rotX_F[0] = 0.; rotY_F[0] = 90.
    transX_F[0] = pos.x+TX.x; transY_F[0] = pos.y+TX.y; transZ_F[0] = pos.z + wMz
    
    # Face 1 unique parameters
    p1_F[1] = gc.Point(-wMz, -wMy, 0.); p2_F[1] = gc.Point(wMz, -wMy, 0.); p3_F[1] = gc.Point(-wMz, wMy, 0.); p4_F[1] = gc.Point(wMz, wMy, 0.)
    rotX_F[1] = 0.; rotY_F[1] = -90.
    transX_F[1] = pos.x-TX.x; transY_F[1] = pos.y-TX.y; transZ_F[1] = pos.z + wMz
    
    # Face 2 unique parameters
    p1_F[2] = gc.Point(-wMx, -wMz, 0.); p2_F[2] = gc.Point(wMx, -wMz, 0.); p3_F[2] = gc.Point(-wMx, wMz, 0.); p4_F[2] = gc.Point(wMx, wMz, 0.)
    rotX_F[2] = -90.; rotY_F[2] = 0.
    transX_F[2] = pos.x+TY.x; transY_F[2] = pos.y+TY.y; transZ_F[2] = pos.z + wMz
    
    # Face 3 unique parameters
    p1_F[3] = gc.Point(-wMx, -wMz, 0.); p2_F[3] = gc.Point(wMx, -wMz, 0.); p3_F[3] = gc.Point(-wMx, wMz, 0.); p4_F[3] = gc.Point(wMx, wMz, 0.)
    rotX_F[3] = 90.; rotY_F[3] = 0.
    transX_F[3] = pos.x-TY.x; transY_F[3] = pos.y-TY.y; transZ_F[3] = pos.z + wMz
    
    # Face 4 unique parameters
    p1_F[4] = gc.Point(-wMx, -wMy, 0.); p2_F[4] = gc.Point(wMx, -wMy, 0.); p3_F[4] = gc.Point(-wMx, wMy, 0.); p4_F[4] = gc.Point(wMx, wMy, 0.)
    rotX_F[4] = 0.; rotY_F[4] = 0.
    transX_F[4] = pos.x; transY_F[4] = pos.y; transZ_F[4] = pos.z + 2*wMz
    
    # Face 5 unique parameters
    p1_F[5] = gc.Point(-wMx, -wMy, 0.); p2_F[5] = gc.Point(wMx, -wMy, 0.); p3_F[5] = gc.Point(-wMx, wMy, 0.); p4_F[5] = gc.Point(wMx, wMy, 0.)
    rotX_F[5] = 0.; rotY_F[5] = 180.
    transX_F[5] = pos.x; transY_F[5] = pos.y; transZ_F[5] = pos.z
    
    # Create the faces and incorporate them in a list
    LOBJ = []
    for i in range (0, 6):
        F = Entity(name = obj_type, \
                   color = colors[i], \
                   alpha_color = alpha_color[i], \
                   materialAV = matAVL[i], \
                   materialAR = Matte(), \
                   geo = Plane( p1 = p1_F[i], p2 = p2_F[i], p3 = p3_F[i], p4 = p4_F[i] ), \
                   transformation = Transformation( rotation = np.array([rotX_F[i], rotY_F[i], rotZ_F[i]]),
                                                    translation = np.array([transX_F[i], transY_F[i], transZ_F[i]]), rotationOrder="ZXY" ))
        LOBJ.append(F)
    
    # Create a group of object with a global bounding box (can improve significantly the computational time!)
    maxXY = max(pos.x, 2*max(wMx, wMy))
    p_min = gc.Point( pos.x - maxXY - gap, pos.y - maxXY - gap, pos.z - gap)
    p_max = gc.Point( pos.x + maxXY + gap, pos.y + maxXY + gap, pos.z + 2*wMz + gap )
    GOBJ = GroupE(LE = LOBJ, BBOX = [p_min, p_max])
    
    return GOBJ

def Ref_Fresnel(dirEnt, geoTrans):
    '''
    Definition of Ref_Fresnel

    Simple Fresnel reflection

    dirE     : Direction of the ray entering on the surface of reflection
    geoTrans : Transformation of the surface where there is reflection

    return a Vector class containing the direction of the reflected ray
    '''
    if isinstance(dirEnt, gc.Vector) :
        dirE = dirEnt
    else :
        raise Exception("the dirEnt argument must be a Vector class")
    if isinstance(geoTrans, gc.Transform) :
        geoT = geoTrans
    else :
        raise Exception("the geoTrans argument must be a Transform class")

    # Default value of the surface plane normal
    NN = gc.Vector(0., 0., 1)
    
    # Real value of the normal after considering transformation
    TT = geoT
    NN = TT(NN)

    # Information needed from the incoming ray
    V = dirE
    V = gc.Vector(-V.x, -V.y, -V.z)
    
    # Use the equation of Fresnel reflection (plenty explained in pbrtv3 book)
    V = dirE + NN*(2*gc.dot(NN, V))

    # Be sure V is normalized
    V = gc.normalize(V)
    
    return V


def Analyse_create_entity(ENTITY, THEDEG = 0., PHIDEG = 0., PLANEDM = 'SM', RAYCOLOR = 'r', SR_VIEW=1,
                          xyz_limit = None, show_rays=True, rs_fac = 1):
    '''
    Definition of Analyse_create_entity

    Enable a 3D visualization of the created objects

    ENTITY    : A list of objects (Entity classes)
    THEDEG    : The zenith angle of the sun
    PHIDEG    : The azimuth angle of the sun
    PlaneDM   : Plane Draw method, two choices 'FM' (First Method) or 'SM'(seconde
                Method). By default 'SM', 'FM' is useful for debug issues
    RAYCOLR   : Sun rays color i.g. 'r', 'b', ...
    SR_VIEW   : Split the number of sun rays that can be seen in the figure
    xyz_limit : By default None and automatically choose x,y,z view limits,
                or can be forced by giving a dictionnary with x,y,z values (in km),
                i.g. {'x_min': 0., 'x_max': 10., 'y_min': 0., 'y_max':10., 'z_min': 0., 'z_max':10.}

    return a matplotlib fig
    '''
    
    ENTITY = convertLGtoLE(ENTITY)

    if (isinstance(ENTITY, Entity)):
        E = []
        E = np.append(E, ENTITY)
        # Enable generic local visualization (part1)
        if isinstance(E[0].geo, Plane):
            GLXmin = min(E[0].geo.p1.x, E[0].geo.p2.x, E[0].geo.p3.x, E[0].geo.p4.x)
            GLYmin = min(E[0].geo.p1.y, E[0].geo.p2.y, E[0].geo.p3.y, E[0].geo.p4.y)
            GLZmin = min(E[0].geo.p1.z, E[0].geo.p2.z, E[0].geo.p3.z, E[0].geo.p4.z)
            GLXmax = max(E[0].geo.p1.x, E[0].geo.p2.x, E[0].geo.p3.x, E[0].geo.p4.x)
            GLYmax = max(E[0].geo.p1.y, E[0].geo.p2.y, E[0].geo.p3.y, E[0].geo.p4.y)
            GLZmax = max(E[0].geo.p1.z, E[0].geo.p2.z, E[0].geo.p3.z, E[0].geo.p4.z)
            GLEcaX = abs(GLXmin-GLXmax); GLEcaY = abs(GLYmin-GLYmax); GLEcaZ = abs(GLZmin-GLZmax);
            GLEcaM = max(GLEcaX, GLEcaY, GLEcaZ)
        # End (part1)
            
    elif (all(isinstance(x, Entity) for x in ENTITY)):
        E = ENTITY
        # Enable generic local visualization (part2)
        # Be carful, if the local is greater than 100km the below need to be modified!
        GLXmin = 100.; GLYmin = 100.; GLZmin = 100.; GLXmax = -100.; GLYmax = -100.; GLZmax = -100.;
        for i in range(0, len(E)):
            if E[i].transformation.transx < GLXmin :
                GLXmin = E[i].transformation.transx
            if E[i].transformation.transx > GLXmax :
                GLXmax = E[i].transformation.transx
            if E[i].transformation.transy < GLYmin :
                GLYmin = E[i].transformation.transy
            if E[i].transformation.transy > GLYmax :
                GLYmax = E[i].transformation.transy
            if E[i].transformation.transz < GLZmin :
                GLZmin = E[i].transformation.transz
            if E[i].transformation.transz > GLZmax :
                GLZmax = E[i].transformation.transz
        GLEcaX = abs(GLXmin-GLXmax); GLEcaY = abs(GLYmin-GLYmax); GLEcaZ = abs(GLZmin-GLZmax);
        GLEcaM = max(GLEcaX, GLEcaY, GLEcaZ)
        # End (part2)
    else:
        raise NameError('ENTITY argument needs to be an Entity object or a list' + \
                        ' of Entity Objects ')

    # calculate the sun direction vector
    vSun = gc.ang2vec(THEDEG, PHIDEG, vec_view='nadir')
    
    wsx = -vSun.x; wsy=-vSun.y; wsz=-vSun.z;
    xs = np.linspace(0, 0.1*wsx, 100)
    ys = np.linspace(0, 0.1*wsy, 100)
    zs = np.linspace(0, 0.1*wsz, 100)

    sunDirection = vSun
    LMir = 0; LMir2 = 0;
    TabPhoton = []; atLeastOneInt = []; xn = []; yn = []; zn = []; xr = []; yr = []; zr = [];
    TabPhoton2 = [];atLeastOneInt2 = []; xr2 = []; yr2 = []; zr2 = [];

    for i in range(0, len(E)):
        if (E[i].name == "reflector"):
            LMir += 1
            atLeastOneInt = np.append(atLeastOneInt, False)
            xn = np.append(xn, None); yn = np.append(yn, None); zn = np.append(zn, None);
            xr = np.append(xr, None); yr = np.append(yr, None); zr = np.append(zr, None);      
            TabPhoton = np.append(TabPhoton, gc.Ray(o = gc.Point(wsx+E[i].transformation.transx, wsy+E[i].transformation.transy, wsz+E[i].transformation.transz), \
                                                    d = gc.Vector( sunDirection.x, sunDirection.y, sunDirection.z ), maxt = 1200.))

    # create the matplotlib figure
    fig = plt.figure()#figsize=[128, 96])
    ax = fig.add_subplot(111, projection=Axes3D.name)
    ax.scatter([-1,1], [-1,1], [-1,1], alpha=0.0)
    
    for k in range (0, len(E)):
        # ===================================================================================
        # En commun (!!reinitialized for each loop!!)
        # ===================================================================================
        # all transform separetly       
        Trans = gc.get_translate_tf(gc.Vector(E[k].transformation.transx, E[k].transformation.transy, \
                                              E[k].transformation.transz))
        Rotx = gc.get_rotateX_tf(E[k].transformation.rotx)
        Roty = gc.get_rotateY_tf(E[k].transformation.roty)
        Rotz = gc.get_rotateZ_tf(E[k].transformation.rotz)

        # total tt of all transform together
        tt = None
        if (E[k].transformation.rotOrder == "XYZ"):
            tt = Trans*Rotx*Roty*Rotz
        elif (E[k].transformation.rotOrder == "XZY"):
            tt = Trans*Rotx*Rotz*Roty
        elif (E[k].transformation.rotOrder == "YXZ"):
            tt = Trans*Roty*Rotx*Rotz
        elif (E[k].transformation.rotOrder == "YZX"):
            tt = Trans*Roty*Rotz*Rotx
        elif (E[k].transformation.rotOrder == "ZXY"):
            tt = Trans*Rotz*Rotx*Roty
        elif (E[k].transformation.rotOrder == "ZYX"):
            tt = Trans*Rotz*Roty*Rotx
        else:
            raise NameError('Unknown rotation order')
        
        tt_inv = tt.inverse()
        
        # ===================================================================================
        
        if isinstance(E[k].geo, Plane):
            # Vertex triangle indices
            vi = np.array([np.array([0, 1, 2]),                   # indices or triangle 1
                           np.array([2, 3, 1])], dtype=np.int32)  # indices of triangle 2

            # List of points of the plane
            P = np.array([np.array([E[k].geo.p1.x, E[k].geo.p1.y, E[k].geo.p1.z]),
                          np.array([E[k].geo.p2.x, E[k].geo.p2.y, E[k].geo.p2.z]),
                          np.array([E[k].geo.p3.x, E[k].geo.p3.y, E[k].geo.p3.z]),
                          np.array([E[k].geo.p4.x, E[k].geo.p4.y, E[k].geo.p4.z])], dtype = np.float64)
            
            #PlaneMesh = TriangleMesh(tt, tt_inv, vi, P)
            PlaneMesh = gc.TriangleMesh(vertices=P, faces=vi)
            PlaneMesh.apply_tf(tt)

            if(E[k].name == "reflector" and THEDEG != None):
                for i in range(0, LMir):
                    t_hit = float('inf')
                    ds = gc.calc_intersection(PlaneMesh, TabPhoton[i])
                    if(ds['is_intersection'].values):
                        if (ds['thit'].values < t_hit):
                            atLeastOneInt[i] = True
                            LMir2 += 1
                            xr2 = np.append(xr2, None); yr2 = np.append(yr2, None); zr2 = np.append(zr2, None); 
                            atLeastOneInt2 = np.append(atLeastOneInt2, False)
                            p_hit = gc.Point(ds['phit'].values)
                            t_hit = ds['thit'].values
                            sunDistance = sunDirection*t_hit
                            tnn = np.linspace(0, 0.001, 20)
                            P1 = p_hit  ; N1 = gc.Normal(ds['nhit'].values);
                            N1 = gc.face_forward(N1, sunDirection * -1)
                            # For ploting the normal and the red ray
                            xn[i] = P1.x + tnn * N1.x
                            yn[i] = P1.y + tnn * N1.y
                            zn[i] = P1.z + tnn * N1.z
                            #tr = np.linspace(Photon.mint, t_hit, 100)
                            tr = np.linspace(t_hit*0.98*(1/rs_fac), t_hit, 100)
                            xr[i] = TabPhoton[i].o.x + tr*TabPhoton[i].d.x
                            yr[i] = TabPhoton[i].o.y + tr*TabPhoton[i].d.y
                            zr[i] = TabPhoton[i].o.z + tr*TabPhoton[i].d.z
                            vecTemp = Ref_Fresnel(dirEnt = TabPhoton[i].d, geoTrans = tt)
                            # print("TabPhoton.d = (", TabPhoton[i].d.x, ", ", TabPhoton[i].d.y, ", ", TabPhoton[i].d.z, ")")
                            # print("vecTemp = (", vecTemp.x, ", ", vecTemp.y, ", ", vecTemp.z, ")")
                            TabPhoton2 = np.append(TabPhoton2, gc.Ray(o=p_hit, d=vecTemp, maxt=120))
                                               
            if (E[k].name == "receiver" and THEDEG != None):
                for i in range(0, LMir2):
                    t_hit = float('inf')
                    ds = gc.calc_intersection(PlaneMesh, TabPhoton2[i])
                    if(ds['is_intersection'].values):
                        atLeastOneInt2[i] = True
                        if (ds['thit'].values < t_hit):
                            p_hit = gc.Point(ds['phit'].values)
                            t_hit = ds['thit'].values
                            tr = np.linspace(TabPhoton2[i].mint, t_hit, 100)
                            xr2[i] = TabPhoton2[i].o.x + tr*TabPhoton2[i].d.x
                            yr2[i] = TabPhoton2[i].o.y + tr*TabPhoton2[i].d.y
                            zr2[i] = TabPhoton2[i].o.z + tr*TabPhoton2[i].d.z
                        
            # Triangles mesh parameters for plot
            # First method (draw even if there is error with an object, useful for debug):
            # ----------------------------->
            if (PLANEDM == 'FM'):
                for itri in range(0, PlaneMesh.ntriangles):
                    p0 = gc.Point(PlaneMesh.vertices[PlaneMesh.faces[itri,0],:])
                    p1 = gc.Point(PlaneMesh.vertices[PlaneMesh.faces[itri,1],:])
                    p2 = gc.Point(PlaneMesh.vertices[PlaneMesh.faces[itri,2],:])
                    Mat = np.array([[p0.x, p0.y, p0.z], \
                                    [p1.x, p1.y, p1.z], \
                                    [p2.x, p2.y, p2.z]])
                    face1 = mp3d.art3d.Poly3DCollection([Mat], alpha = E[k].alpha_color, linewidths=0.2)
                    face1.set_facecolor(mcolors.to_rgba(E[k].color))
                    ax.add_collection3d(face1)

            # Second method (better visual, avoid some matplotlib bugs):
            # ----------------------------->
            if (PLANEDM == 'SM'):
                p0_t0 = gc.Point(PlaneMesh.vertices[PlaneMesh.faces[0,0],:])
                p1_t0 = gc.Point(PlaneMesh.vertices[PlaneMesh.faces[0,1],:])
                p2_t0 = gc.Point(PlaneMesh.vertices[PlaneMesh.faces[0,2],:])
                p0_t1 = gc.Point(PlaneMesh.vertices[PlaneMesh.faces[1,0],:])
                p1_t1 = gc.Point(PlaneMesh.vertices[PlaneMesh.faces[1,1],:])
                p2_t1 = gc.Point(PlaneMesh.vertices[PlaneMesh.faces[1,2],:])
                Mat = np.array([[p0_t0.x, p0_t0.y, p0_t0.z], \
                                [p1_t0.x, p1_t0.y, p1_t0.z], \
                                [p2_t0.x, p2_t0.y, p2_t0.z], \
                                [p0_t1.x, p0_t1.y, p0_t1.z], \
                                [p1_t1.x, p1_t1.y, p1_t1.z], \
                                [p2_t1.x, p2_t1.y, p2_t1.z]])
                
                if (np.array_equal(Mat[:,0], np.full((6), Mat[0,0]))):
                    yy, zz = np.meshgrid(Mat[:,0], Mat[:,2])
                    xx = np.full((6,6), Mat[0,0])
                    ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba(E[k].color), alpha = E[k].alpha_color, \
                                    linewidth=0.2, antialiased=True)
                elif (np.array_equal(Mat[:,1], np.full((6), Mat[0,1]))):
                    xx, zz = np.meshgrid(Mat[:,0], Mat[:,2])
                    yy = np.full((6,6), Mat[0,1])
                    ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba(E[k].color), alpha = E[k].alpha_color, \
                                    linewidth=0.2, antialiased=True)
                elif (np.array_equal(Mat[:,2], np.full((6), Mat[0,2]))): # need to be verified
                    xx, yy = np.meshgrid(Mat[:,0], Mat[:,1])
                    zz = np.full((6,6), Mat[0,2])
                    ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba(E[k].color), alpha = E[k].alpha_color, \
                                    linewidth=0.2, antialiased=True)
                else:
                    ax.plot_trisurf(Mat[:,0], Mat[:,1], Mat[:,2], color = mcolors.to_rgba(E[k].color), \
                                    alpha = 0.5, linewidth=0.2, antialiased=True)

        elif isinstance(E[k].geo, Spheric):

            S = gc.Sphere(E[k].geo.radius, E[k].geo.z0, E[k].geo.z1, E[k].geo.phi)
            S_mesh = S.to_trianglemesh()
            S_mesh.apply_tf(tt)
            if S_mesh.oTw.is_identity(): vertices_sm = S_mesh.vertices
            else: vertices_sm = S_mesh.oTw(gc.Point(S_mesh.vertices)).to_numpy()

            ax.plot_trisurf(vertices_sm[:,0], vertices_sm[:,1], vertices_sm[:,2],
                            triangles = S_mesh.faces, color=E[k].color, alpha=E[k].alpha_color)

            for i in range(0, LMir):
                ds = gc.calc_intersection(S, TabPhoton[i])
                if(ds['is_intersection'].values):
                    atLeastOneInt[i] = True
                    t_hit = float('inf')
                    if (ds['thit'].values < t_hit):
                        p_hit = gc.Point(ds['phit'].values)
                        t_hit = ds['thit'].values
                        sunDistance = sunDirection*t_hit
                        tnn = np.linspace(0, 0.001, 20)
                        P1 = p_hit ; N1 = ds['nhit'].values;
                        N1 = gc.face_forward(N1, sunDirection * -1)
                        # For ploting the normal and the red ray
                        xn[i] = P1.x + tnn * N1.x
                        yn[i] = P1.y + tnn * N1.y
                        zn[i] = P1.z + tnn * N1.z
                        #tr = np.linspace(Photon.mint, t_hit, 100)
                        tr = np.linspace(t_hit*0.98*(1/rs_fac), t_hit, 100)
                        xr[i] = TabPhoton[i].o.x + tr*TabPhoton[i].d.x
                        yr[i] = TabPhoton[i].o.y + tr*TabPhoton[i].d.y
                        zr[i] = TabPhoton[i].o.z + tr*TabPhoton[i].d.z
        else:
            raise NameError('This geometry is unknown!')


    # ==============================================
    # plot all the geometries
    if (show_rays):
        for i in range(0, LMir):
            if (atLeastOneInt[i] and i%SR_VIEW ==0):
                ax.plot(xr[i], yr[i], zr[i], color=RAYCOLOR, linewidth=1*rs_fac)

        for i in range(0, LMir2):
            if (atLeastOneInt2[i] and i%SR_VIEW ==0):
                ax.plot(xr2[i], yr2[i], zr2[i], color=RAYCOLOR, linewidth=1*rs_fac)

    if (xyz_limit is not None):
        ax.set_xlim3d(xyz_limit['x_min'], xyz_limit['x_max'])
        ax.set_ylim3d(xyz_limit['y_min'], xyz_limit['y_max'])
        ax.set_zlim3d(xyz_limit['z_min'], xyz_limit['z_max'])
    # Enable generic local visualization (part3)   
    elif (len(E) == 1):
        if (GLEcaZ == GLEcaM):
            ax.set_zlim3d(E[0].transformation.transz+GLZmin, E[0].transformation.transz+GLZmax)
        else:
            TempVal = GLEcaM - GLEcaZ
            ax.set_zlim3d(E[0].transformation.transz + GLZmin-(0.5*GLEcaM), E[0].transformation.transz + GLZmax+(0.5*GLEcaM))
        if (GLEcaX == GLEcaM):
            ax.set_xlim3d(E[0].transformation.transx+GLXmin, E[0].transformation.transx+GLXmax)
        else:
            TempVal = GLEcaM - GLEcaX
            ax.set_xlim3d(E[0].transformation.transx+GLXmin-(0.5*GLEcaM), E[0].transformation.transx+GLXmax+(0.5*GLEcaM))
        if (GLEcaY == GLEcaM):
            ax.set_ylim3d(E[0].transformation.transy+GLYmin, E[0].transformation.transy+GLYmax)
        else:
            TempVal = GLEcaM - GLEcaY
            ax.set_ylim3d(E[0].transformation.transy+GLYmin-(0.5*GLEcaM), E[0].transformation.transy+GLYmax+(0.5*GLEcaM)) 
    else:
        ax.set_zlim3d(0, GLEcaM)
        if (GLEcaX == GLEcaM):
            ax.set_xlim3d(GLXmin, GLXmax)
        else:
            TempVal = GLEcaM - GLEcaX
            ax.set_xlim3d(GLXmin-(0.5*GLEcaM), GLXmax+(0.5*GLEcaM))
        if (GLEcaY == GLEcaM):
            ax.set_ylim3d(GLYmin, GLYmax)
        else:
            TempVal = GLEcaM - GLEcaY
            ax.set_ylim3d(GLYmin-(0.5*GLEcaM), GLYmax+(0.5*GLEcaM)) 
    # End (part3)    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # Show the geometries
    fig = ax.get_figure()
    return fig


def visualize_entity(ENTITY, THEDEG = 0., PHIDEG = 0., PLANEDM = 'SM', RAYCOLOR = 'r', SR_VIEW=1,
                     xyz_limit = None, show_rays=True, rs_fac = 1):
    '''
    Definition of visualize_entity

    Enable a 3D visualization of the created objects

    ENTITY    : A list of objects (Entity classes)
    THEDEG    : The zenith angle of the sun
    PHIDEG    : The azimuth angle of the sun
    PlaneDM   : Plane Draw method, two choices 'FM' (First Method) or 'SM'(seconde
                Method). By default 'SM', 'FM' is useful for debug issues
    RAYCOLR   : Sun rays color i.g. 'r', 'b', ...
    SR_VIEW   : Split the number of sun rays that can be seen in the figure
    xyz_limit : By default None and automatically choose x,y,z view limits,
                or can be forced by giving a dictionnary with x,y,z values (in km),
                i.g. {'x_min': 0., 'x_max': 10., 'y_min': 0., 'y_max':10., 'z_min': 0., 'z_max':10.}

    return a matplotlib fig
    '''
    
    ENTITY = convertLGtoLE(ENTITY)

    if (isinstance(ENTITY, Entity)):
        E = []
        E = np.append(E, ENTITY)
        # Enable generic local visualization (part1)
        if isinstance(E[0].geo, Plane):
            GLXmin = min(E[0].geo.p1.x, E[0].geo.p2.x, E[0].geo.p3.x, E[0].geo.p4.x)
            GLYmin = min(E[0].geo.p1.y, E[0].geo.p2.y, E[0].geo.p3.y, E[0].geo.p4.y)
            GLZmin = min(E[0].geo.p1.z, E[0].geo.p2.z, E[0].geo.p3.z, E[0].geo.p4.z)
            GLXmax = max(E[0].geo.p1.x, E[0].geo.p2.x, E[0].geo.p3.x, E[0].geo.p4.x)
            GLYmax = max(E[0].geo.p1.y, E[0].geo.p2.y, E[0].geo.p3.y, E[0].geo.p4.y)
            GLZmax = max(E[0].geo.p1.z, E[0].geo.p2.z, E[0].geo.p3.z, E[0].geo.p4.z)
            GLEcaX = abs(GLXmin-GLXmax); GLEcaY = abs(GLYmin-GLYmax); GLEcaZ = abs(GLZmin-GLZmax);
            GLEcaM = max(GLEcaX, GLEcaY, GLEcaZ)
        # End (part1)
            
    elif (all(isinstance(x, Entity) for x in ENTITY)):
        E = ENTITY
        # Enable generic local visualization (part2)
        # Be carful, if the local is greater than 100km the code below need to be modified!
        GLXmin = 100.; GLYmin = 100.; GLZmin = 100.; GLXmax = -100.; GLYmax = -100.; GLZmax = -100.
        for i in range(0, len(E)):
            if E[i].transformation.transx < GLXmin : GLXmin = E[i].transformation.transx
            if E[i].transformation.transx > GLXmax : GLXmax = E[i].transformation.transx
            if E[i].transformation.transy < GLYmin : GLYmin = E[i].transformation.transy
            if E[i].transformation.transy > GLYmax : GLYmax = E[i].transformation.transy
            if E[i].transformation.transz < GLZmin : GLZmin = E[i].transformation.transz
            if E[i].transformation.transz > GLZmax : GLZmax = E[i].transformation.transz
        GLEcaX = abs(GLXmin-GLXmax); GLEcaY = abs(GLYmin-GLYmax); GLEcaZ = abs(GLZmin-GLZmax)
        GLEcaM = max(GLEcaX, GLEcaY, GLEcaZ)
        # End (part2)
    else:
        raise NameError('ENTITY argument needs to be an Entity object or a list' + \
                        ' of Entity Objects ')

    # calculate the sun direction vector
    vSun = gc.ang2vec(THEDEG, PHIDEG, vec_view='nadir')
    wsx = -vSun.x; wsy=-vSun.y; wsz=-vSun.z

    lplaneMesh = []
    lMir_int = int(0)
    E_rec = []; E_ref = []
    for i in range(0, len(E)):
        if (E[i].name == "reflector"): E_ref.append(E[i])
        if (E[i].name == "receiver") : E_rec.append(E[i])

    nbRef = len(E_ref)
    xr = [None]*nbRef; yr = [None]*nbRef; zr = [None]*nbRef
    atLeastOneInt = [False]*nbRef
    TabPhoton2 = []

    for k in range (0, len(E_ref)):
        # Get the transformation
        tt = E_ref[k].get_transformation()

        photon_pos = gc.Point(wsx+E_ref[k].transformation.transx, wsy+E_ref[k].transformation.transy, wsz+E_ref[k].transformation.transz)
        photon = gc.Ray(o = photon_pos, d = vSun, maxt = 1200.)
    
        if isinstance(E_ref[k].geo, Plane):
           # Vertex triangle indices
            vi = np.array([np.array([0, 1, 2]),                   # indices or triangle 1
                           np.array([2, 3, 1])], dtype=np.int32)  # indices of triangle 2

            # List of points of the plane
            P = np.array([np.array([E[k].geo.p1.x, E[k].geo.p1.y, E[k].geo.p1.z]),
                          np.array([E[k].geo.p2.x, E[k].geo.p2.y, E[k].geo.p2.z]),
                          np.array([E[k].geo.p3.x, E[k].geo.p3.y, E[k].geo.p3.z]),
                          np.array([E[k].geo.p4.x, E[k].geo.p4.y, E[k].geo.p4.z])], dtype = np.float64)
            
            PlaneMesh = gc.TriangleMesh(vertices=P, faces=vi)
            PlaneMesh.apply_tf(tt)
            lplaneMesh.append(PlaneMesh)

            ds = gc.calc_intersection(PlaneMesh, photon)
            if(ds['is_intersection'].values and ds['thit'].values < float('inf')):
                atLeastOneInt[k] = True
                lMir_int += int(1)
                p_hit = gc.Point(ds['phit'].values)
                t_hit = ds['thit'].values
                tr = np.linspace(t_hit*0.98*(1/rs_fac), t_hit, 100)
                xr[k] = photon.o.x + tr*photon.d.x
                yr[k] = photon.o.y + tr*photon.d.y
                zr[k] = photon.o.z + tr*photon.d.z
                vecTemp = Ref_Fresnel(dirEnt = photon.d, geoTrans = tt)
                TabPhoton2 = np.append(TabPhoton2, gc.Ray(o=p_hit, d=vecTemp, maxt=120))

        else: raise NameError('This geometry is unknown or not yet accepted!')


    xr2 = [None]*lMir_int; yr2 = [None]*lMir_int; zr2 = [None]*lMir_int
    atLeastOneInt2 = [False]*lMir_int

    for k in range (0, len(E_rec)):
        # Get the transformation
        tt = E_rec[k].get_transformation()

        if isinstance(E_rec[k].geo, Plane):
            # Vertex triangle indices
            vi = np.array([np.array([0, 1, 2]),                   # indices or triangle 1
                           np.array([2, 3, 1])], dtype=np.int32)  # indices of triangle 2

            # List of points of the plane
            P = np.array([np.array([E[k].geo.p1.x, E[k].geo.p1.y, E[k].geo.p1.z]),
                          np.array([E[k].geo.p2.x, E[k].geo.p2.y, E[k].geo.p2.z]),
                          np.array([E[k].geo.p3.x, E[k].geo.p3.y, E[k].geo.p3.z]),
                          np.array([E[k].geo.p4.x, E[k].geo.p4.y, E[k].geo.p4.z])], dtype = np.float64)
            
            PlaneMesh = gc.TriangleMesh(vertices=P, faces=vi)
            PlaneMesh.apply_tf(tt)
            lplaneMesh.append(PlaneMesh)

            for i in range(0, lMir_int):
                ds = gc.calc_intersection(PlaneMesh, TabPhoton2[i])
                if(ds['is_intersection'].values and ds['thit'].values < float('inf')):
                    atLeastOneInt2[i] = True
                    p_hit = gc.Point(ds['phit'].values)
                    t_hit = ds['thit'].values
                    tr = np.linspace(TabPhoton2[i].mint, t_hit, 100)
                    xr2[i] = TabPhoton2[i].o.x + tr*TabPhoton2[i].d.x
                    yr2[i] = TabPhoton2[i].o.y + tr*TabPhoton2[i].d.y
                    zr2[i] = TabPhoton2[i].o.z + tr*TabPhoton2[i].d.z

    # create the matplotlib figure
    fig = plt.figure()#figsize=[128, 96])
    ax = fig.add_subplot(111, projection=Axes3D.name)
    ax.scatter([-1,1], [-1,1], [-1,1], alpha=0.0)

    for pMesh in lplaneMesh:
        # Triangles mesh parameters for plot
        # First method (draw even if there is error with an object, useful for debug):
        # ----------------------------->
        if (PLANEDM == 'FM'):
            for itri in range(0, pMesh.ntriangles):
                p0 = gc.Point(pMesh.vertices[pMesh.faces[itri,0],:])
                p1 = gc.Point(pMesh.vertices[pMesh.faces[itri,1],:])
                p2 = gc.Point(pMesh.vertices[pMesh.faces[itri,2],:])
                Mat = np.array([[p0.x, p0.y, p0.z], \
                                [p1.x, p1.y, p1.z], \
                                [p2.x, p2.y, p2.z]])
                face1 = mp3d.art3d.Poly3DCollection([Mat], alpha = E[k].alpha_color, linewidths=0.2)
                face1.set_facecolor(mcolors.to_rgba(E[k].color))
                ax.add_collection3d(face1)

        # Second method (better visual, avoid some matplotlib bugs):
        # ----------------------------->
        if (PLANEDM == 'SM'):
            p0_t0 = gc.Point(pMesh.vertices[pMesh.faces[0,0],:])
            p1_t0 = gc.Point(pMesh.vertices[pMesh.faces[0,1],:])
            p2_t0 = gc.Point(pMesh.vertices[pMesh.faces[0,2],:])
            p0_t1 = gc.Point(pMesh.vertices[pMesh.faces[1,0],:])
            p1_t1 = gc.Point(pMesh.vertices[pMesh.faces[1,1],:])
            p2_t1 = gc.Point(pMesh.vertices[pMesh.faces[1,2],:])
            Mat = np.array([[p0_t0.x, p0_t0.y, p0_t0.z], \
                            [p1_t0.x, p1_t0.y, p1_t0.z], \
                            [p2_t0.x, p2_t0.y, p2_t0.z], \
                            [p0_t1.x, p0_t1.y, p0_t1.z], \
                            [p1_t1.x, p1_t1.y, p1_t1.z], \
                            [p2_t1.x, p2_t1.y, p2_t1.z]])
            
            if (np.array_equal(Mat[:,0], np.full((6), Mat[0,0]))):
                yy, zz = np.meshgrid(Mat[:,0], Mat[:,2])
                xx = np.full((6,6), Mat[0,0])
                ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba(E[k].color), alpha = E[k].alpha_color, \
                                linewidth=0.2, antialiased=True)
            elif (np.array_equal(Mat[:,1], np.full((6), Mat[0,1]))):
                xx, zz = np.meshgrid(Mat[:,0], Mat[:,2])
                yy = np.full((6,6), Mat[0,1])
                ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba(E[k].color), alpha = E[k].alpha_color, \
                                linewidth=0.2, antialiased=True)
            elif (np.array_equal(Mat[:,2], np.full((6), Mat[0,2]))): # need to be verified
                xx, yy = np.meshgrid(Mat[:,0], Mat[:,1])
                zz = np.full((6,6), Mat[0,2])
                ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba(E[k].color), alpha = E[k].alpha_color, \
                                linewidth=0.2, antialiased=True)
            else:
                ax.plot_trisurf(Mat[:,0], Mat[:,1], Mat[:,2], color = mcolors.to_rgba(E[k].color), \
                                alpha = 0.5, linewidth=0.2, antialiased=True)

    # ==============================================
    # plot all the geometries
    if (show_rays):
        for i in range(0, nbRef):
            if (atLeastOneInt[i] and i%SR_VIEW ==0): ax.plot(xr[i], yr[i], zr[i], color=RAYCOLOR, linewidth=1*rs_fac)

        for i in range(0, lMir_int):
            if (atLeastOneInt2[i] and i%SR_VIEW ==0): ax.plot(xr2[i], yr2[i], zr2[i], color=RAYCOLOR, linewidth=1*rs_fac)

    if (xyz_limit is not None):
        ax.set_xlim3d(xyz_limit['x_min'], xyz_limit['x_max'])
        ax.set_ylim3d(xyz_limit['y_min'], xyz_limit['y_max'])
        ax.set_zlim3d(xyz_limit['z_min'], xyz_limit['z_max'])
    # Enable generic local visualization (part3)   
    elif (len(E) == 1):
        if (GLEcaZ == GLEcaM): ax.set_zlim3d(E[0].transformation.transz+GLZmin, E[0].transformation.transz+GLZmax)
        else                 : ax.set_zlim3d(E[0].transformation.transz + GLZmin-(0.5*GLEcaM), E[0].transformation.transz + GLZmax+(0.5*GLEcaM))
        if (GLEcaX == GLEcaM): ax.set_xlim3d(E[0].transformation.transx+GLXmin, E[0].transformation.transx+GLXmax)
        else                 : ax.set_xlim3d(E[0].transformation.transx+GLXmin-(0.5*GLEcaM), E[0].transformation.transx+GLXmax+(0.5*GLEcaM))
        if (GLEcaY == GLEcaM): ax.set_ylim3d(E[0].transformation.transy+GLYmin, E[0].transformation.transy+GLYmax)
        else                 : ax.set_ylim3d(E[0].transformation.transy+GLYmin-(0.5*GLEcaM), E[0].transformation.transy+GLYmax+(0.5*GLEcaM)) 
    else:
        ax.set_zlim3d(0, GLEcaM)
        if (GLEcaX == GLEcaM): ax.set_xlim3d(GLXmin, GLXmax)
        else                 : ax.set_xlim3d(GLXmin-(0.5*GLEcaM), GLXmax+(0.5*GLEcaM))
        if (GLEcaY == GLEcaM): ax.set_ylim3d(GLYmin, GLYmax)
        else                 : ax.set_ylim3d(GLYmin-(0.5*GLEcaM), GLYmax+(0.5*GLEcaM)) 
    # End (part3)    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # Show the geometries
    fig = ax.get_figure()
    return fig


def generateHfP(THEDEG=0., PHIDEG = 0., PH = [gc.Point(0., 0., 0.)], PR = gc.Point(0., 0., 0.), \
                HSX = 0.001, HSY = 0.001, REF = 1, ROUGH=0, HTYPE = None, LMTF = None):
    '''
    Definition of generateHfP

    Enable to generate well oriented Heliostats from their positions

    THEDEG  : Sun zenith angle (degree)
    PHIDEG  : Sun azimuth angle (degree)
    PH      : Coordinates of the center of heliostats (list of point classes)
    PR      : Coordinate of the center of the receiver (point class)
    HSX     : Heliostat size in x axis (kilometer)
    HSY     : Heliostat size in y axis (kilometer)
    REF     : reflectivity of the heliostats
    HTYPE   : If specified must be a class heliostat
    LMTF    : Under development
    return a list with Entity/GroupE object
    '''
    # Avoid crash from old notebooks/scripts
    PH_ = PH.copy()
    for iph in range(len(PH)):
        if isinstance(PH[iph], Point): PH_[iph] = gc.Point(PH[iph].x, PH[iph].y, PH[iph].z)
    if isinstance(PR, Point): PR = gc.Point(PR.x, PR.y, PR.z)

    lObj = []

    # Case where the heliostat is totally plane
    if (HTYPE is None):
        # compute the sun direction vector
        vSun = gc.ang2vec(THEDEG, PHIDEG, vec_view='nadir')
        bboxDist = np.sqrt(HSX*HSX + HSY*HSY)/2

        Hxx = HSX/2; Hyy = HSY/2
        objM = Entity(name = "reflector", \
                      materialAV = Mirror(reflectivity = REF, roughness = ROUGH), \
                      materialAR = Matte(reflectivity = 0.), \
                      geo = Plane( p1 = gc.Point(-Hxx, -Hyy, 0.),
                                   p2 = gc.Point(Hxx, -Hyy, 0.),
                                   p3 = gc.Point(-Hxx, Hyy, 0.),
                                   p4 = gc.Point(Hxx, Hyy, 0.) ), \
                      transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                       translation = np.array([0., 0., 0.]) ))


        for i in range (0, len(PH)):
            # 1) Find the normalized vector colinear (and same dir) to the normal of heliostat surface
            vecHR = PH_[i]-PR
            vecHR = gc.normalize(vecHR)

            # 2) Find the necessary rotations to apply on the heliostat to reflect to the receiver
            rInfo = findRots(UI=vSun, UO=vecHR)
            rotYD = rInfo[0]; rotZD = rInfo[1];

            # 3) Once the rotation angles have been found, create heliostat objects
            objMi = Entity(objM);
            objMi.bboxGPmin = gc.Point(PH_[i].x-bboxDist, PH_[i].y-bboxDist, PH_[i].z-bboxDist)
            objMi.bboxGPmax = gc.Point(PH_[i].x+bboxDist, PH_[i].y+bboxDist, PH_[i].z+bboxDist)
            objMi.transformation = Transformation( rotation = np.array([0., rotYD, rotZD]), \
                                                   translation = np.array([PH_[i].x, PH_[i].y, PH_[i].z]), \
                                                   rotationOrder = "ZYX")
            lObj.append(objMi)
    # Case where the heliostat is composed by facets (i.g. to consider the curvature)
    else:
        # Take the commun parameters of all heliostats
        SPX = HTYPE.sPx; SPY = HTYPE.sPy; HSX = HTYPE.hSx; HSY = HTYPE.hSy; CURVE_FL = HTYPE.curveFL;
        
        # Generate all the facets and store them as entity object in a list 
        for i in range (0, len(PH)):
            H0 = Heliostat(SPX=SPX, SPY=SPY, HSX=HSX, HSY=HSY, CURVE_FL=CURVE_FL, POS=PH_[i], REF=REF, ROUGH=ROUGH)
            if LMTF is None: TLE = generateLEfH(HELIO=H0, PR=PR, THEDEG=THEDEG, PHIDEG=PHIDEG)
            else: TLE = generateLEfH(HELIO=H0, PR=PR, THEDEG=THEDEG, PHIDEG=PHIDEG, MTF = LMTF[i])
            GTEMP = GroupE(LE = TLE)
            lObj.append(GTEMP)

    return lObj


def generateHfA(THEDEG=0., PHIDEG = 0., PR = gc.Point(0., 0., 50.), MINANG=0., \
                MAXANG=360., GAPDEG = 5., FDRH = 0.1, NBH = 10, GAPDIST = 0.01, \
                HSX = 0.001, HSY = 0.001, PILLH = 0.006, REF = 1, ROUGH=0,
                HTYPE=None, LMTF = None, RLPH = False):
    '''
    Definition of generateHfA

    Enable to generate well oriented Heliostats from two angles [MINANG, MAXANG] 
 
          y
          ^ 
          |/) ANG
          ---> x

    THEDEG  : Sun zenith angle (degree)
    PHIDEG  : Sun azimuth angle (degree)
    PR      : Coordinate of the center of the receiver (point class)
            # Heliostats are filled between MINANG and MAXANG
    MINANG  : min value of ANG (degree)
    MAXANG  : max value of ANG (degree)
    GAPDEG  : Fill heliostats every GAPDEG inside [MINANG, MAXANG] (degree)
    FDRH    : First Distance Receiver-Heliostat (kilometer)
    NBH     : number of heliostats to put every GAPDEG
    GAPDIST : After FDRH, the gap between heliostats (kilometer)
    HSX     : Heliostat size in x axis (kilometer)
    HSY     : Heliostat size in y axis (kilometer)
    PILLH   : Pillar height, distance Ground-Heliostat (kilometer)
    REF     : reflectivity of the heliostats
    HTYPE   : If specified must be a class heliostat
    RLPH    : I true return also the list with heliostat positions

    return a list with Entity/GroupE objects
    '''
    # Avoid crash from old notebooks/scripts
    if isinstance(PR, Point): PR = gc.Point(PR.x, PR.y, PR.z)

    # I) Find the position of all heliostats
    lenpH = int(  ( (MAXANG-MINANG)/GAPDEG )*NBH  )
    
    # To avoid a given bug
    if (MAXANG-MINANG < 360.000000001 and MAXANG-MINANG > 359.999999999):
        nbI = int(lenpH/NBH)
    else:
        nbI = int(lenpH/NBH) + 1

    print("Total number of Heliostats = ", nbI*NBH)
    
    pH = []
    myRotZ = MINANG

    if (MINANG != MAXANG):
        for i in range (0, nbI):
            Dhr = FDRH
            for j in range (0, NBH):
                myP = gc.Point(Dhr, 0., 0.)
                RotZT = gc.get_rotateZ_tf(myRotZ)
                myP=RotZT(myP)
                pH.append( gc.Point(myP.x, myP.y, myP.z+PILLH) )
                Dhr += GAPDIST
            myRotZ += GAPDEG
    else:
        Dhr = FDRH
        RotZT = gc.get_rotateZ_tf(myRotZ)
        for j in range (0, NBH):
            myP = gc.Point(Dhr, 0., 0.)
            myP=RotZT(myP)
            pH.append( gc.Point(myP.x, myP.y, myP.z+PILLH) )
            Dhr += GAPDIST      


    # II) Creation of heliostats
    lObj = []

    # Case where the heliostat is totally plane
    if (HTYPE is None):
        # calculate the sun direction vector
        vSun = gc.ang2vec(THEDEG, PHIDEG, vec_view='nadir')
        bboxDist = np.sqrt(HSX*HSX + HSY*HSY)/2

        Hxx = HSX/2; Hyy = HSY/2
        objM = Entity(name = "reflector", \
                      materialAV = Mirror(reflectivity = REF, roughness = ROUGH), \
                      materialAR = Matte(), \
                      geo = Plane( p1 = gc.Point(-Hxx, -Hyy, 0.),
                                   p2 = gc.Point(Hxx, -Hyy, 0.),
                                   p3 = gc.Point(-Hxx, Hyy, 0.),
                                   p4 = gc.Point(Hxx, Hyy, 0.) ), \
                      transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                       translation = np.array([0., 0., 0.]) ))

        for i in range (0, len(pH)):
            # 1) The vector of the photon after a reflection (here the opposite direction) 
            vecHR = pH[i]-PR
            vecHR = gc.normalize(vecHR)

            # 2) The incoming (vSun) and outcoming (vecHR) directions are known then find
            #    the rotation angles
            rInfo = findRots(UI=vSun, UO=vecHR)
            rotYD = rInfo[0]; rotZD = rInfo[1];

            # 3) Once the rotation angles have been found, create heliostat objects 
            objMi = Entity(objM);
            objMi.bboxGPmin = gc.Point(pH[i].x-bboxDist, pH[i].y-bboxDist, pH[i].z-bboxDist)
            objMi.bboxGPmax = gc.Point(pH[i].x+bboxDist, pH[i].y+bboxDist, pH[i].z+bboxDist)
            objMi.transformation = Transformation( rotation = np.array([0., rotYD, rotZD]), \
                                                   translation = np.array([pH[i].x, pH[i].y, pH[i].z]), \
                                                   rotationOrder = "ZYX")
            lObj.append(objMi)
        
    # Case where the heliostat is composed by facets (i.g. to consider the curvature)
    else:
        # Take the commun parameters of all heliostats
        SPX = HTYPE.sPx; SPY = HTYPE.sPy; HSX = HTYPE.hSx; HSY = HTYPE.hSy; CURVE_FL = HTYPE.curveFL;
        
        # Generate all the facets and store them as entity object in a list 
        for i in range (0, len(pH)):
            H0 = Heliostat(SPX=SPX, SPY=SPY, HSX=HSX, HSY=HSY, CURVE_FL=CURVE_FL, POS=pH[i], REF=REF, ROUGH=ROUGH)
            if LMTF is None: TLE = generateLEfH(HELIO=H0, PR=PR, THEDEG=THEDEG, PHIDEG=PHIDEG)
            else: TLE = generateLEfH(HELIO=H0, PR=PR, THEDEG=THEDEG, PHIDEG=PHIDEG, MTF = LMTF[i])
            GTEMP = GroupE(LE = TLE)
            lObj.append(GTEMP)
    
    if (RLPH):
        return lObj, pH
    else:
        return lObj

def convertLGtoLE(LGOBJ):
    '''
    Definition of the function convertLGtoLE
    
    ==== ARGS:
    LGOBJ : List containing Entity and GroupE objects

    ==== RETURN:
    LOBJ  : List with only Entity object
    '''
    nGObj=len(LGOBJ); LOBJ=[];

    for i in range (0, nGObj):
        if isinstance(LGOBJ[i], GroupE):
            LOBJ.extend(LGOBJ[i].le)
        elif isinstance(LGOBJ[i], Entity):
            LOBJ.append(LGOBJ[i])
        else:
            raise NameError('In the list, only Entity and GroupE classes are autorised!')

    return LOBJ

def convertVtoAngles(v, TYPE="Sensor", verbose=False):
    """
    Definition of the function convertVtoAngles

    coordinate system convention:

      y
      ^   x : right; y : front; z : top
      |
    z X -- > x

    The description of a direction by a vector v is
    converted by the description by 2 angles: Theta and Phi

    Arg:
    v     : A direction described by Vector class object

    TYPE  : By default TYPE=str(Sensor) where we look from (0,0,0) to
            (x,y,z) but we can be in Sun case i.e. TYPE=str(Sun) and
            where we look at the opposite side : from (x,y,z) to (0,0,0)

    Return Theta and Phi in degrees:
    Theta : Zenith angle, start at Z+ in plane ZX going
            in the trigonometric direction arround y axis

    Phi   : Azimuth angle, start at X+ in plane XY going in
            the trigonométric direction arround z axis
    """
    warn_message = 'Using convertVtoAngles from smartg.visualizegeo is depracated as of smartg v1.1.0. ' + \
                   'Use geoclide package instead.\n For example: import geoclide as gc; gc.vec2ang().'
    warn(warn_message, DeprecationWarning)

    if isinstance(v, Vector):
        # First be sure that the vector v is normalized
        if (TYPE == "Sensor"):
            v = Normalize(v)
        elif (TYPE == "Sun"):
            v = Normalize(Vector(-v.x, -v.y, -v.z)) # Sun we look at the oposite side
        else:
            raise NameError('TYPE arg must be str(Sensor) or str(Sun)')

        # Find rotations in Y and Z direction (see the function findRots)
        rInfo = findRots(vecNF=gc.Vector(v.x,v.y,v.z))
        Theta = rInfo[0]; Phi = rInfo[1];

        if verbose : print("Theta=", Theta, "Phi=", Phi)
        
        return Theta, Phi
    else:
        raise NameError('v argument must be a Vector')

def convertAnglestoV(THETA=0., PHI=0., TYPE="Sensor"):
    """
    Definition of the function convertVtoAngles

    coordinate system convention:

      y
      ^   x : right; y : front; z : top
      |
    z X -- > x

    The description of a direction by 2 angles THETA and PHI is
    converted by the description a vector v

    Arg:
    Theta : Zenith angle in degree, start at Z+ in plane ZX going
            in the trigonometric direction arround y axis

    Phi   : Azimuth angle in degree, start at X+ in plane XY going in
            the trigonométric direction arround z axis

    TYPE  : By default TYPE=str(Sensor) where we look from (0,0,0) to
            (x,y,z) but we can be in Sun case i.e. TYPE=str(Sun) and
            where we look at the opposite side : from (x,y,z) to (0,0,0)

    Return a normalized vector v:
    v     : A direction described by Vector class object
    """
    warn_message = 'Using convertAnglestoV from smartg.visualizegeo is depracated as of smartg v1.1.0. ' + \
                   'Use geoclide package instead.\n For example: import geoclide as gc; gc.ang2vec().'
    warn(warn_message, DeprecationWarning)

    if (TYPE == "Sensor"):
        # By default the vector v = (0, 0, 1) for THETA=0 and PHI=0
        v = Vector(0, 0, 1)
    elif (TYPE == "Sun"):
        # By default the vector v = (0, 0, -1) for THETA=0 and PHI=0
        v = Vector(0, 0, -1)
    else:
        raise NameError('TYPE arg must be str(Sensor) or str(Sun)')
    
    # Creation of the transform object
    TT = Transform()

    # Take the zenith angle for the first rotation in Y axis
    v = TT.rotateY(THETA)[v]

    # Take the azimuth angle for the second rotation in Z axis
    v = TT.rotateZ(PHI)[v]

    # Be sure v is normalized
    v = Normalize(v)
    
    return v

def rotate_vector(vector, rot_x, rot_y, rot_z, rot_order="xyz"):
    """
    Definition of the function rotate_vector

    coordinate system convention:

      y
      ^   x : right; y : front; z : top
      |
    z X -- > x

    Given a vector and rotations to perform to this vector in degrees
    in the x,y,z axes, with in option the rotation order

    Arg:
    vector    : A direction described by Vector class object
    rotx,y,z  : Rotations in x,y and z in degrees
    rot_order : str with the order of rotations i.g. 'xyz', zxy', ...

    Return:
    rotated_vector : The rotated (normalized) direction (also a Vector class)
    """
    # Avoid crash from old notebooks/scripts
    if isinstance(v, Vector): v = gc.Vector(v.x, v.y, v.z)

    TT = gc.Transform()
    tr_x = gc.get_rotateX_tf(rot_x)
    tr_y = gc.get_rotateY_tf(rot_y)
    tr_z = gc.get_rotateZ_tf(rot_z) 
    if rot_order == "XYZ":
        TT = tr_x*tr_y*tr_z
    elif rot_order == "XZY":
        TT = tr_x*tr_z*tr_y
    elif rot_order == "YXZ":
        TT = tr_y*tr_x*tr_z
    elif rot_order == "YZX":
        TT = tr_y*tr_z*tr_x
    elif rot_order == "ZXY":
        TT = tr_z*tr_x*tr_y
    elif rot_order == "ZYX":
        TT = tr_z*tr_y*tr_x
    else:
        raise NameError("Unknown rot_order value!")
    rotated_vector = TT(vector)
    rotated_vector = gc.normalize(rotated_vector)

    return rotated_vector


def interpolate_refls_from_wls (wls, refls, wls_new, extrapolate=False):
    """
        Definition: Giving a set of wavelengths (wls) and reflectivities (refls),
                    get the interpolated reflectivities folowing the new set of wavelengths (wls_total)
    
    ==== ARGS:
    wls     : List/array of wavelengths
    refls   : List/array with reflectivities at each wavelength of wls 
    wls_new : List/array of the new wavelengths where we want to interpolate

    ==== RETURN:
    refls_new : numpy array with the interpolated reflectivities
    """

    if extrapolate: f = interpolate.interp1d(wls, refls, fill_value='extrapolate')
    else : f = interpolate.interp1d(wls, refls, fill_value=(refls[0],refls[-1]), bounds_error=False)

    refls_new = f(wls_new)

    # Ensure relfectivities are between 0 and 1
    refls_new[refls_new<0] = 0
    refls_new[refls_new>1] = 1

    return refls_new


def is_comment(s):
    """
    function to check if a line
    starts with some character.
    Here # for comment
    """
    # return true if a line starts with #
    return s.startswith('#')

def extractPoints(filename):
    """
    Extract the coordinates of heliostats from a file.

    The input file must contain at least:
    1. A first comment line beginning with '#'
    2. An empty line
    3. Lines with the x, y, and z coordinates of each heliostat, separated by commas

    Parameters
    ----------
    filename : str | pathlib.Path
        The path to the file containing the heliostat coordinates.
    
    Returns
    -------
    out : list
        A list of `gc.Point` objects, each containing the x, y, and z coordinates
        of a heliostat.
    """

    # First check if filename is an str type
    try:
        with open(filename, "r") as file:
            for curline in dropwhile(is_comment, file):
                insideFile = file.read()
    except FileNotFoundError:
        print(str(filename) + ' has been not found')
    except IOError:
        print("Enter/Exit error with " + str(filename))
            
    # Looking for a float and fill it in listVal
    listVal = re.findall(r"-?[0-9]+\.?[0-9]*", insideFile)
        
    # Number of dimension and number of heliostats
    nbDim = 3 # x, y and z --> 3 dim
    nbH = int(len(listVal)/nbDim)

    # # Fill the x, y and z coordinates into a list of Point classes
    lPH = []
    for i in range (0, nbH):
        lPH.append(  gc.Point( float(listVal[i*nbDim]), float(listVal[(i*nbDim)+1]),
                               float(listVal[(i*nbDim)+2]) )  )

    return lPH

def random_equal_area_geometries(theta_in_degrees, phi_in_degrees, fov_radius_in_degrees=0.265, N=1):	# central direction
    '''
    equal area geometries inside a circle of radius fov_radius centred 
    phis, thetas are the angular coordinates in degrees w.r.t. the center of the circle		
    '''
    mum = np.cos(np.radians(fov_radius_in_degrees))
    # random sampling of theta according to equal area sampling
    ct=np.sqrt(1. - np.random.rand(N) * (1-mum**2))
    t = np.degrees(np.arccos(ct))
    # uniform sampling for azimuth
    p=np.random.rand(N) * 360.
    # deal with the particular case theta = 0 deg
    if (theta_in_degrees == 0):
        return {'th_deg': t, 'phi_deg': p, 'zip':True}
    # unit vector around which to rotate all previous directions
    u=gc.normalize(gc.cross(gc.ang2vec(0.,0.), gc.ang2vec(theta_in_degrees, phi_in_degrees)))
    # rotation matrix calculation
    R = gc.get_rotate_tf(theta_in_degrees, u).m[0:3,0:3]
    # new directions
    t2 = np.zeros_like(t)
    p2 = np.zeros_like(p)
    for k in range(N):
        v = gc.ang2vec(theta=t[k], phi=p[k]).to_numpy()
        vv = gc.Vector(R.dot(v))
        t2[k],p2[k] = gc.vec2ang(vv)

    return {'th_deg': t2, 'phi_deg': p2, 'zip':True}


def packed_geometries(theta_in_degrees, phi_in_degrees, fov_radius_in_degrees=0.265):	# central direction
    '''
    optimal packing of 19 equal small circles in a unit circle
    xs, ys are the coordinates of the small circle centers
    phis, thetas are the angular coordinates in degrees w.r.t. the center of the unit circle		
    '''
    xs = np.array([-0.205604646759568224693193969093,
        0.205604646759568224693193969093,
        -0.561722341219392118229847722909,
        0.561722341219392118229847722909,
        -0.205604646759568224693193969093,
        0.205604646759568224693193969093,
        -0.767326987978960342923041692002,
        0.767326987978960342923041692002,
        -0.411209293519136449386387938185,
        0.000000000000000000000000000000,
        0.411209293519136449386387938185,
        -0.767326987978960342923041692002,
        0.767326987978960342923041692002,
        -0.205604646759568224693193969093,
        0.205604646759568224693193969093,
        -0.561722341219392118229847722909,
        0.561722341219392118229847722909,
        -0.205604646759568224693193969093,
        0.205604646759568224693193969093
        ])
    phis = phi_in_degrees + fov_radius_in_degrees*xs			    

    ys = np.array([-0.767326987978960342923041692002, 
        -0.767326987978960342923041692002,
        -0.561722341219392118229847722909,
        -0.561722341219392118229847722909,
        -0.356117694459823893536653753817,
        -0.356117694459823893536653753817,
        -0.205604646759568224693193969093,
        -0.205604646759568224693193969093,
        0.000000000000000000000000000000,
        0.000000000000000000000000000000,
        0.000000000000000000000000000000,
        0.205604646759568224693193969093,
        0.205604646759568224693193969093,
        0.356117694459823893536653753817,
        0.356117694459823893536653753817,
        0.561722341219392118229847722909,
        0.561722341219392118229847722909,
        0.767326987978960342923041692002,
        0.767326987978960342923041692002
        ])
    thetas = theta_in_degrees + fov_radius_in_degrees*ys
    le={'th_deg':thetas, 'phi_deg':phis, 'zip':True}

    return le


    
if __name__ == '__main__':

    Heliostat1 = Entity(name = "reflector", \
                       materialAV = Mirror(reflectivity = 1., roughness = 0.1), \
                       materialAR = Matte(), \
                       geo = Plane( p1 = gc.Point(-10., -10., 0.),
                                    p2 = gc.Point(10., -10., 0.),
                                    p3 = gc.Point(-10., 10., 0.),
                                    p4 = gc.Point(10., 10., 0.) ), \
                       transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                        translation = np.array([0., 0., 0.]) ))

    Recepteur1 = Entity(name = "receiver", \
                        materialAV = Mirror(reflectivity = 1., roughness = 0.1), \
                        materialAR = Matte(), \
                        geo = Plane( p1 = gc.Point(-10., -10., 0.),
                                     p2 = gc.Point(10., -10., 0.),
                                     p3 = gc.Point(-10., 10., 0.),
                                     p4 = gc.Point(10., 10., 0.) ), \
                        transformation = Transformation( rotation = np.array([45., 0., 0.]), \
                                                         translation = np.array([0., -10., 80.]) ))
    Heliostat2 = Entity(name = "reflector", \
                        materialAV = Mirror(reflectivity = 1., roughness = 0.1), \
                        materialAR = Matte(), \
                        geo = Spheric( radius = 20.,
                                       z0 = -0.,
                                       z1 = 20.,
                                       phi = 360. ), \
                        transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                         translation = np.array([0., 15., 30.]) ))


    print("Helio1 :", Heliostat1)
    print("Recept1 :", Recepteur1)
    print("Helio2 :", Heliostat2)
    
    fig = Analyse_create_entity([Heliostat1, Recepteur1, Heliostat2], THEDEG = 0.)

    fig.show()
