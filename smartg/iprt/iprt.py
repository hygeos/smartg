#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from luts.luts import Idx, LUT
import xarray as xr



def seclect_iprt_IQUV(model_val, z_alti, thetas=None, phis=None, inv_thetas=False, inv_phis=False, change_U_sign=False,
                      I_index=int(6), va_index=int(4), phi_index=int(5), z_index=int(1), stdev=False):
    """
    Description: Select U,Q,U and V results from IPRT matrix results

    === Parameters:
    model_val       : Matrix with the model values (read from IPRT phase A result files)
    z_alti          : Keep only results at this z_alti
    thetas          : Keep only results with these theta values
    phis            : Same as thetas but with phi values
    inv_thetas      : Inverse the vector with theta values
    inv_phis        : Same as inv_thetas but with phi values
    change_U_sign   : multiply by -1 the U results (can be useful since in backward and forward the convention change)
    I_index         : In case we don't follow exactly the IPRT output format convention we can specify the index where I begin,
                      but next the order must be the same: I, Q, U and then V.
    va_index        : Same as I_index but with va (VZA)
    phi_index       : Same as I_index but with phi (VAA)
    stdev           : If true return also IQUV stdev

    === Retrun
    I,...,V              : The selected values of I, Q, U and V
    I,...,V,Istd,...Vstd : If stdev = True retrun also Istd to Vstd
    """

    NBS = model_val.shape[0]
    if thetas is None:
        s_thetas = []
        for i in range (0, NBS):
            if (model_val[i, z_index] ==z_alti): s_thetas.append(model_val[i,va_index])
        thetas = np.sort(np.unique(np.array(s_thetas)))
    
    if phis is None:
        s_phis = []
        for i in range (0, NBS):
            if (model_val[i, z_index] ==z_alti): s_phis.append(model_val[i,phi_index])
        phis = np.sort(np.unique(np.array(s_phis)))
    
    NTH = len(thetas)
    NPH = len(phis)

    I = np.zeros((NTH, NPH))
    Q = np.zeros((NTH, NPH))
    U = np.zeros((NTH, NPH))
    V = np.zeros((NTH, NPH))

    if stdev:
        Istd = np.zeros((NTH, NPH))
        Qstd = np.zeros((NTH, NPH))
        Ustd = np.zeros((NTH, NPH))
        Vstd = np.zeros((NTH, NPH))
    
    if change_U_sign: U_sign = int(-1)
    else            : U_sign = int(1)

    for i in range (0, NBS):
        if (model_val[i, z_index] == z_alti 
                and True in (thetas == model_val[i,va_index])
                and True in (phis == model_val[i,phi_index])):
            if inv_thetas: indi = int(np.squeeze(np.argwhere(thetas == model_val[i,va_index])))
            else         : indi = NTH-1-int(np.squeeze(np.argwhere(thetas == model_val[i,va_index])))
            if inv_phis  : indj = NPH-1-int(np.squeeze(np.argwhere(phis == model_val[i,phi_index])))
            else         : indj = int(np.squeeze(np.argwhere(phis == model_val[i,phi_index])))
            I[indi,indj] =  model_val[i, I_index]
            Q[indi,indj] =  model_val[i, I_index+1]
            U[indi,indj] =  model_val[i, I_index+2]*U_sign
            V[indi,indj] =  model_val[i, I_index+3]
            if stdev:
                Istd[indi,indj] =  model_val[i, I_index+4]
                Qstd[indi,indj] =  model_val[i, I_index+5]
                Ustd[indi,indj] =  model_val[i, I_index+6]*U_sign
                Vstd[indi,indj] =  model_val[i, I_index+7]

    if not stdev:
        return I, Q, U, V
    else:
        return I, Q, U, V, Istd, Qstd, Ustd, Vstd


def select_and_plot_polar_iprt(model_val, z_alti, depol=None, thetas=None, phis=None, inv_thetas=False, inv_phis=False, change_Q_sign=False, change_U_sign=False,
                               change_V_sign=False, maxI=None, maxQ=None, maxU=None, maxV=None,  cmapI=None, cmapQ=None, cmapU=None, cmapV=None,
                               forceIQUV = None, title=None, save_fig=None, sym=False, I_index=int(6), va_index=int(4),
                               phi_index=int(5), z_index=int(1), depol_index=int(0), outputIQUV=False, outputIQUVstd=False, avoid_plot=False):
    """
    Description: Select U,Q,U and V results from IPRT matrix results, then plot the results

    === Parameters:
    model_val       : Matrix with the model values (read from IPRT phase A result files)
    z_alti          : Keep only results at this z_alti
    depol           : depolarisation factor
    thetas          : Keep only results with these theta values
    phis            : Same as thetas but with phi values
    inv_thetas      : Inverse the vector with theta values
    inv_phis        : Same as inv_thetas but with phi values
    change_U_sign   : multiply by -1 the U results (can be useful since in backward and forward the convention change)
    change_V_sign   : same as change_U_sign but with V
    maxI,...,maxV   : We can specify the max values in I, Q, U and V for the plots
    cmapI,...,cmapV : We can specify a specific color map for I, Q, U or/and V results
    forceIQUV       : Circumvent the result selection of model_val by giving direclty the I, Q, U and V values (list of matrices)
    title           : Title of the global plot
    save_fig        : If given, save the figure at the given format, e.g save_fig='myFigName.png'
    sym             : IPRT phi results are from 0 to 180 deg, if sym = True plot the symmetrical results from 180 to 360 deg
    I_index         : In case we don't follow exactly the IPRT output format convention we can specify the index where I begin,
                      but next the order must be the same: I, Q, U and then V.
    va_index        : Same as I_index but with va (VZA)
    phi_index       : Same as I_index but with phi (VAA)
    outputIQUV      : If True, retrun I, Q, U and V values
    outputIQUVstd   : If True, retrun I, Q, U and V stdev values
    avoid_plot      : Do not plot, can be useful if we only want to get the I, Q, U and V values

    === Retrun
    valI,...,valV : if outputIQUV is True return the selected values of I, Q, U and V, else return nothing
    """

    NBS = model_val.shape[0]
    if thetas is None:
        s_thetas = []
        for i in range (0, NBS):
            cond_z_depol = (depol is None and (model_val[i, z_index] == z_alti)) or (depol is not None and ((model_val[i, z_index] == z_alti) and (model_val[i, depol_index] == depol)))
            if (cond_z_depol): s_thetas.append(model_val[i,va_index])
        thetas = np.sort(np.unique(np.array(s_thetas)))
    
    if phis is None:
        s_phis = []
        for i in range (0, NBS):
            cond_z_depol = (depol is None and (model_val[i, z_index] == z_alti)) or (depol is not None and ((model_val[i, z_index] == z_alti) and (model_val[i, depol_index] == depol)))
            if (cond_z_depol): s_phis.append(model_val[i,phi_index])
        phis = np.sort(np.unique(np.array(s_phis)))
    
    if sym: phis = np.concatenate((phis, phis+180))
    NTH = len(thetas)
    NPH = len(phis)
    if sym: NPH_D = round(NPH/2)
    else: NPH_D = NPH

    valI = np.zeros((NTH, NPH))
    valQ = np.zeros((NTH, NPH))
    valU = np.zeros((NTH, NPH))
    valV = np.zeros((NTH, NPH))

    if outputIQUVstd:
        valIstd = np.zeros((NTH, NPH_D))
        valQstd = np.zeros((NTH, NPH_D))
        valUstd = np.zeros((NTH, NPH_D))
        valVstd = np.zeros((NTH, NPH_D))

    if change_Q_sign: Q_sign = int(-1)
    else            : Q_sign = int(1) 
    if change_U_sign: U_sign = int(-1)
    else            : U_sign = int(1)
    if change_V_sign: V_sign = int(-1)
    else            : V_sign = int(1)

    if forceIQUV is not None:
        valI[:,0:NPH_D] = forceIQUV[0]
        valQ[:,0:NPH_D] = forceIQUV[1]
        valU[:,0:NPH_D] = forceIQUV[2]*U_sign
        valV[:,0:NPH_D] = forceIQUV[3]
    else:
        for i in range (0, NBS):
            cond_z_depol = (depol is None and (model_val[i, z_index] == z_alti)) or (depol is not None and ((model_val[i, z_index] == z_alti) and (model_val[i, depol_index] == depol)))
            if (cond_z_depol
                and True in (thetas == model_val[i,va_index])
                and True in (phis == model_val[i,phi_index])  ):
                if inv_thetas: indi = int(np.squeeze(np.argwhere(thetas == model_val[i,va_index])))
                else         : indi = NTH-1-int(np.squeeze(np.argwhere(thetas == model_val[i,va_index])))
                if inv_phis  : indj = NPH_D-1-int(np.squeeze(np.argwhere(phis[0:NPH_D] == model_val[i,phi_index])))
                else         : indj = int(np.squeeze(np.argwhere(phis[0:NPH_D] == model_val[i,phi_index])))
                valI[indi,indj] =  model_val[i, I_index]
                valQ[indi,indj] =  model_val[i, I_index+1]*Q_sign
                valU[indi,indj] =  model_val[i, I_index+2]*U_sign
                valV[indi,indj] =  model_val[i, I_index+3]*V_sign
                if (outputIQUVstd): 
                    valIstd[indi,indj] =  model_val[i, I_index+4]
                    valQstd[indi,indj] =  model_val[i, I_index+5]
                    valUstd[indi,indj] =  model_val[i, I_index+6]
                    valVstd[indi,indj] =  model_val[i, I_index+7] 

    if sym:
        for i in range(NTH):
                for j in range(NPH_D):
                    valI[i,NPH_D+j] =  valI[i,NPH_D-j-1]
                    valQ[i,NPH_D+j] =  valQ[i,NPH_D-j-1]
                    valU[i,NPH_D+j] =  valU[i,NPH_D-j-1]
                    valV[i,NPH_D+j] =  valV[i,NPH_D-j-1]


    if not avoid_plot:
        plt.rcParams.update({'font.size':13})

        thetas_scaled = (thetas - np.min(thetas))/(np.max(thetas)- np.min(thetas))*90.
        if maxI is None:
            maxI = max(np.abs(np.min(valI)), np.abs(np.max(valI)))
            minI = 0.
        else:
            minI=-maxI
        if maxQ is None: maxQ = max(np.abs(np.min(valQ)), np.abs(np.max(valQ)))
        if maxU is None: maxU = max(np.abs(np.min(valU)), np.abs(np.max(valU)))
        if maxV is None: maxV = max(np.abs(np.min(valV)), np.abs(np.max(valV)))

        if cmapI is None: cmapI = "jet"
        if cmapQ is None: cmapQ = "RdBu_r"
        if cmapU is None: cmapU = "RdBu_r"
        if cmapV is None: cmapV = "RdBu_r"

        fig, ax = plt.subplots(1,4, figsize=(12,4),subplot_kw=dict(projection='polar'))
        if title is not None: fig.suptitle(title)
        #csI = ax[0].contourf(np.deg2rad(phis), thetas[::-1], valI, cmap='jet', levels=np.linspace(0., 9.5e-2, 100, endpoint=True))
        ax[0].grid(False)
        csI = ax[0].pcolormesh(np.deg2rad(phis), thetas_scaled[::-1], valI, cmap=cmapI, vmin=minI, vmax=maxI, shading='gouraud')
        cbarI = fig.colorbar(csI, ax=ax[0], shrink=0.8, orientation='horizontal', ticks=np.linspace(minI, maxI, 3, endpoint=True), format="%4.1e")
        cbarI.set_label(r'I')
        ax[0].set_yticklabels([])
        ax[0].grid(axis='both', linewidth=1.5, linestyle=':', color='black', alpha=0.5)

        #csQ = ax[1].contourf(np.deg2rad(phis), thetas[::-1], valQ, cmap='RdBu_r', levels=np.linspace(-1.4e-2, 1.4e-2, 100, endpoint=True))
        ax[1].grid(False)
        csQ = ax[1].pcolormesh(np.deg2rad(phis), thetas_scaled[::-1], valQ, cmap=cmapQ, vmin=-maxQ, vmax=maxQ, shading='gouraud')
        cbarQ = fig.colorbar(csQ, ax=ax[1], shrink=0.8, orientation='horizontal', ticks=np.linspace(-maxQ, maxQ, 3, endpoint=True), format="%4.1e")
        cbarQ.set_label(r'Q')
        ax[1].set_yticklabels([])
        ax[1].grid(axis='both', linewidth=1.5, linestyle=':', color='black', alpha=0.5)

        #csU = ax[2].contourf(np.deg2rad(phis), thetas[::-1], -valU, cmap='RdBu_r', levels=np.linspace(-2.6e-2, 2.6e-2, 100, endpoint=True))
        ax[2].grid(False)
        csU = ax[2].pcolormesh(np.deg2rad(phis), thetas_scaled[::-1], valU, cmap=cmapU, vmin=-maxU, vmax=maxU, shading='gouraud')
        cbarU = fig.colorbar(csU, ax=ax[2], shrink=0.8, orientation='horizontal', ticks=np.linspace(-maxU, maxU, 3, endpoint=True), format="%4.1e")
        cbarU.set_label(r'U')
        ax[2].set_yticklabels([])
        ax[2].grid(axis='both', linewidth=1.5, linestyle=':', color='black', alpha=0.5)

        #csV = ax[3].contourf(np.deg2rad(phis), thetas[::-1], valV, cmap='RdBu_r', levels=np.linspace(-1e-5, 1e-5, 100, endpoint=True))
        ax[3].grid(False)
        csV = ax[3].pcolormesh(np.deg2rad(phis), thetas_scaled[::-1], valV, cmap=cmapV, vmin=-maxV, vmax=maxV, shading='gouraud')
        cbarV = fig.colorbar(csV, ax=ax[3], shrink=0.8, orientation='horizontal', ticks=np.linspace(-maxV, maxV, 3, endpoint=True), format="%4.1e")
        cbarV.set_label(r'V')
        ax[3].set_yticklabels([])
        ax[3].grid(axis='both', linewidth=1.5, linestyle=':', color='black', alpha=0.5)
        
        fig.tight_layout()
        if save_fig is not None: plt.savefig(save_fig)

    if outputIQUV and outputIQUVstd:
        return valI[:,0:NPH_D], valQ[:,0:NPH_D], valU[:,0:NPH_D], valV[:,0:NPH_D], valIstd, valQstd, valUstd, valVstd
    elif (outputIQUV):
        return valI[:,0:NPH_D], valQ[:,0:NPH_D], valU[:,0:NPH_D], valV[:,0:NPH_D]
    elif (outputIQUVstd) :
        valIstd, valQstd, valUstd, valVstd

def convert_SGout_to_IPRTout(lm, lU_sign, case_name, ldepol, lalt, lSZA, lSAA, lVZA, lVAA, file_name, output_layer=None, interp=False):
    """
    Description: Convert SMART-G output into IPRT ascii output format

    === Parameters:
    lm           : List of SMART-G output (MLUT object)
    lU_sign      : List with multiplication to perform to U of each output
    case_name    : The IPRT case name
    ldepol       : List of Depol values
    lalt         : List with the viewing altitude of each output
    lSZA         : List of Sun Zenith Angles
    lSAA         : List of Sun Azimuth Angles
    lVZA         : List or numpy 1d array with VZA values
    lVAA         : Same as lVZA but with VAA values
    file_name    : The name of the ascci file to be created
    output_layer : If None the output layer is always '_up (TOA)', else a list with wanted ones ('_down (0+)', ...)

    """
    output =  "# IPRT case " + case_name + "\n"
    output += "# RT model: SMARTG\n"
    output += "# depol altitude sza saa va phi I Q U V Istd Qstd Ustd Vstd\n"

    for im, m in enumerate(lm):
        fac = np.cos(np.radians(lSZA[im]))/np.pi
        VZA = lVZA[im]
        VAA = lVAA[im]
        if output_layer is None:
            output_layeri = '_up (TOA)'
        else:
            output_layeri = output_layer[im]
        for iza, za, in enumerate(VZA):
            for iaa, aa, in enumerate(VAA):
                if not interp:
                    I = m['I'+output_layeri][iaa,iza]*fac
                    Q = m['Q'+output_layeri][iaa,iza]*fac
                    U = m['U'+output_layeri][iaa,iza]*fac*lU_sign[im]
                    V = m['V'+output_layeri][iaa,iza]*fac

                    I_std = m['I_stdev'+output_layeri][iaa,iza]*fac
                    Q_std = m['Q_stdev'+output_layeri][iaa,iza]*fac
                    U_std = m['U_stdev'+output_layeri][iaa,iza]*fac
                    V_std = m['V_stdev'+output_layeri][iaa,iza]*fac
                else:
                    I = m['I'+output_layeri][Idx(aa),Idx(za)]*fac
                    Q = m['Q'+output_layeri][Idx(aa),Idx(za)]*fac
                    U = m['U'+output_layeri][Idx(aa),Idx(za)]*fac*lU_sign[im]
                    V = m['V'+output_layeri][Idx(aa),Idx(za)]*fac

                    I_std = m['I_stdev'+output_layeri][Idx(aa),Idx(za)]*fac
                    Q_std = m['Q_stdev'+output_layeri][Idx(aa),Idx(za)]*fac
                    U_std = m['U_stdev'+output_layeri][Idx(aa),Idx(za)]*fac
                    V_std = m['V_stdev'+output_layeri][Idx(aa),Idx(za)]*fac
                output+= f"{ldepol[im]:.2f} {lalt[im]:.1f} {lSZA[im]:.1f} {lSAA[im]:.1f} {za:.1f} {aa:.1f} {I:.5e} " + \
                         f"{Q:.5e} {U:.5e} {V:.5e} {I_std:.5e} {Q_std:.5e} {U_std:.5e} {V_std:.5e}\n"

    with open(file_name, 'w') as f:
        f.write(output)

def plot_iprt_radiances(IQUV_obs, IQUV_mod, IQUVstd_obs, IQUVstd_mod, xaxis, xlabel, IQUVyMin=None, IQUVyMax=None, title=None, save_fig=None):
    """
    Description: Plot radiances and dif between observation (or reference model) and model radiances

    === Parameters:
    IQUV_obs, IQUV_mod       : Numpy matrices with observation and model IQUV signals (matrix of dim [NSTK, NXAXIS])
    IQUVstd_obs, IQUVstd_mod : Numpy matrices with observation and model IQUV signal stdev
    xaxis                    : IQUV signals are varying as function of xaxis (can be VZA or VAA)
    xlabel                   : Plot xaxis label
    IQUVyMin                 : Min IQUV values to set for radiance plot ylim
    IQUVyMax                 : Max IQUV values to set for radiance plot ylim
    title                    : Title of the global plot
    save_fig                 : If given, save the figure at the given format, e.g save_fig='myFigName.png'
    """

    fig, ax = plt.subplots(2,4, figsize=(13,8))
    if title: fig.suptitle(title, fontsize=15)

    for istk in range (0, 4):
        if (istk==0): ax[0,istk].set_ylabel("normalized radiance", fontsize=13)
        ax[0,istk].set_xlabel(xlabel, fontsize=13)
        ax[0,istk].yaxis.set_major_formatter(mtick.FormatStrFormatter('%5.1e'))
        ax[0,istk].plot(xaxis, IQUV_obs[istk], color='red')
        ax[0,istk].plot(xaxis, IQUV_mod[istk], color='blue')
        if IQUVyMin is not None and IQUVyMax is not None:
            ax[0,istk].set_yticks(np.linspace(IQUVyMin[istk], IQUVyMax[istk], 6))
            ax[0,istk].set_yticks(np.linspace(IQUVyMin[istk], IQUVyMax[istk], 6))
            ax[0,istk].set_ylim(ymin=IQUVyMin[istk], ymax=IQUVyMax[istk])
        else:
            yt = ax[0,istk].get_yticks()
            ax[0,istk].locator_params(axis='y', nbins=6)
            if IQUVyMin is not None:
                ax[0,istk].set_yticks(np.linspace(IQUVyMin[istk], np.max(yt), 6))
                ax[0,istk].set_yticks(np.linspace(IQUVyMin[istk], np.max(yt), 6))
                ax[0,istk].set_ylim(ymin=IQUVyMin[istk], ymax=np.max(yt))
            elif IQUVyMax is not None:
                ax[0,istk].set_yticks(np.linspace(np.min(yt), IQUVyMax[istk], 6))
                ax[0,istk].set_yticks(np.linspace(np.min(yt), IQUVyMax[istk], 6))
                ax[0,istk].set_ylim(ymin=np.min(yt), ymax=IQUVyMax[istk])
            else:
                ax[0,istk].set_yticks(np.linspace(np.min(yt), np.max(yt), 6))
                ax[0,istk].set_yticks(np.linspace(np.min(yt), np.max(yt), 6))
                ax[0,istk].set_ylim(ymin=np.min(yt), ymax=np.max(yt))
        ax[0,istk].set_xlim(xmin=np.min(xaxis), xmax=np.max(xaxis))
        ax[0,istk].locator_params(axis='x', nbins=3)

        if (istk==0): ax[1,istk].set_ylabel("abs. diff", fontsize=13)
        ax[1,istk].set_xlabel(xlabel, fontsize=13)
        ax[1,istk].yaxis.set_major_formatter(mtick.FormatStrFormatter('%5.1e'))
        markers, caps, bars = ax[1,istk].errorbar(xaxis, IQUV_obs[istk,:]-IQUV_mod[istk,:], yerr=IQUVstd_obs[istk]+IQUVstd_mod[istk], fmt='x', color='blue', ecolor='grey', capsize=2)
        [bar.set_alpha(0.25) for bar in bars]
        [cap.set_alpha(0.25) for cap in caps]
        ax[1,istk].axhline(0, color='black')
        ax[1,istk].locator_params(axis='x', nbins=3)
        ax[1,istk].locator_params(axis='y', nbins=6)
        yt = ax[1,istk].get_yticks()
        ax[1,istk].set_yticks(np.linspace(np.min(yt), np.max(yt), 6))
        ax[1,istk].set_ylim(ymin=np.min(yt), ymax=np.max(yt))
    fig.tight_layout()
    if save_fig is not None: plt.savefig(save_fig)
    
def compute_deltam_IPRTout(obs, mod, I_obs_id=6, I_mod_id=6, print_res=True):
    if (not isinstance(obs, np.ndarray) or not isinstance(mod, np.ndarray)): raise NameError("obs and mod must be np.ndarray!")
    id_obs = [I_obs_id, I_obs_id+1, I_obs_id+2, I_obs_id+3]
    id_mod = [I_mod_id, I_mod_id+1, I_mod_id+2, I_mod_id+3]
    stk = ['I', 'Q', 'U', 'V']
    delta_m = np.zeros(4, dtype=np.float32)
    for i in range(len(stk)):
        with np.errstate(divide='raise', invalid='raise'):
            try:
                delta_m[i] = 100*np.sqrt(np.sum(  (obs[:,id_obs[i]]-mod[:,id_mod[i]])**2   )) / np.sqrt( np.sum(obs[:,id_obs[i]]**2) )
            except FloatingPointError:
                delta_m[i] = 0.
        if print_res: print(stk[i], f"{delta_m[i]:.3f}")
    return delta_m

def compute_deltam(obs, mod, print_res=True):

    if (isinstance(obs, np.ndarray)):
        obs_tmp = obs.copy()
        obs = []
        for i in range (0, 4): obs.append(obs_tmp[i,:])

    if (isinstance(mod, np.ndarray)):
        mod_tmp = mod.copy()
        mod = []
        for i in range (0, 4): mod.append(mod_tmp[i,:])
    
    stk = ['I', 'Q', 'U', 'V']
    delta_m = np.zeros(4, dtype=np.float32)
    for i in range(len(stk)):
        with np.errstate(divide='raise', invalid='raise'):
            try:
                delta_m[i] = 100*np.sqrt(np.sum(  (obs[i]-mod[i])**2   )) / np.sqrt( np.sum(obs[i]**2) )
            except FloatingPointError:
                delta_m[i] = 0.
        if print_res: print(stk[i], f"{delta_m[i]:.3f}")
    return delta_m



def groupIQUV(lI, lQ, lU, lV):

    NBVAL = int(0)
    I_tot = lI[0].flatten()
    Q_tot = lQ[0].flatten()
    U_tot = lU[0].flatten()
    V_tot = lV[0].flatten()

    for i in range (len(lI)):
        NBVAL += round(lI[i].shape[0]*lI[i].shape[1])
        if (i > 0):
            I_tot = np.concatenate((I_tot, lI[i].flatten()))
            Q_tot = np.concatenate((Q_tot, lQ[i].flatten()))
            U_tot = np.concatenate((U_tot, lU[i].flatten()))
            V_tot = np.concatenate((V_tot, lV[i].flatten()))

    IQUV_smartg_ref_tot = np.zeros((4,NBVAL), dtype=np.float32)
    IQUV_smartg_ref_tot[0,:] = I_tot
    IQUV_smartg_ref_tot[1,:] = Q_tot
    IQUV_smartg_ref_tot[2,:] = U_tot
    IQUV_smartg_ref_tot[3,:] = V_tot

    return IQUV_smartg_ref_tot

def read_phase_nth_cte(filename, nb_theta=int(721), convert_IparIper=True, normalize=False):
    """
    Description: Read libRatran aerosol/cloud files (i.g. wc.sol.mie.cdf) or monochromatic IPRT netcdf aerosol/cloud files,
    and convert to LUT object with a constant theta discretisation i.e. nb_theta = cte.

    === Parameters:
    filename         : File name with path location of netcdf file.
    nb_theta         : Number of theta discretization between 0 and 180 degrees.
    convert_IparIper : Convert IQUV phase matrix into IparIperUV phase matrix
    normalize        : Normalize such that the integral of P0 is equal to 2
    === Return:
    LUT object with the cloud phase matrix but with a constant theta number = nb_theta
    """

    ds = xr.open_dataset(filename)

    if 'hum' in ds.variables: 
        rh_reff = ds["hum"].data
        rh_or_reff = 'rh'
    elif 'reff' in ds.variables:
        rh_reff = ds["reff"].data
        rh_or_reff = 'reff'
    else:
        raise Exception('Error')
    
    phase = ds["phase"][:, :, :, :].data

    NBSTK   = ds.nphamat.size
    NBTHETA = nb_theta
    NBRH_OR_REFF  = rh_reff.size
    NWAV    = ds["wavelen"].size
    theta = np.linspace(0., 180., num=NBTHETA)
    wavelength = ds["wavelen"].data*1e3

    P = LUT( np.full((NWAV, NBRH_OR_REFF, NBSTK, NBTHETA), np.nan, dtype=np.float32),
                axes=[wavelength, rh_reff, None, theta],
                names=['wav_phase', rh_or_reff, 'stk', 'theta_atm'],
                desc="phase_atm" )

    for iwav in range (0, NWAV):
        for irhreff in range(NBRH_OR_REFF):
            for istk in range (NBSTK):
                # ntheta (wl, reff, stk)
                nth = ds["ntheta"][iwav, irhreff, istk].data

                # theta (wl, reff, stk, ntheta)
                th = ds["theta"][iwav, irhreff, istk, :].data

                P.data[iwav, irhreff, istk, :] = np.interp(theta, th[:nth], phase[iwav,irhreff,istk,:nth],  period=np.inf)

    if (convert_IparIper):
        # convert I, Q into Ipar, Iper
        if (NBSTK == 4): # spherical particles
            P0 = P.data[:,:,0,:].copy()
            P1 = P.data[:,:,1,:].copy()
            P.data[:,:,0,:] = P0+P1
            P.data[:,:,1,:] = P0-P1
        elif (NBSTK) == 6: # non spherical particles
            # note: the sign of P43/P34 affects only the sign of V, since V=0 for rayleigh scattering it does not matter 
            P0 = P.data[:,:,0,:].copy()
            P1 = P.data[:,:,1,:].copy()
            P4 = P.data[:,:,4,:].copy()
            P.data[:,:,0,:] = 0.5*(P0+2*P1+P4) # P11
            P.data[:,:,1,:] = 0.5*(P0-P4)      # P12=P21
            P.data[:,:,4,:] = 0.5*(P0-2*P1+P4) # P22
        else:
            raise NameError("Number of unique phase components is different than 4 or 6!")
        
        if normalize:
            for iwav in range (0, NWAV):
                for irhreff in range (0, NBRH_OR_REFF):
                    # Note: from Ipar Iper phase, if NBSTK=4 -> P0=(P11+P12)/2, and if NBSTK=6 -> P0=(P11+P22+2*P12)/2
                    f = P0[iwav, irhreff,:]
                    mu= np.cos(np.radians(theta))
                    Norm = np.trapz(f,-mu)
                    P.data[iwav,irhreff,:,:] *= 2./abs(Norm)
    return P