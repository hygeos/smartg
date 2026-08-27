#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helpers for the IPRT model intercomparison cases.

This module provides tools to convert SMART-G outputs to the IPRT
(International Polarized Radiative Transfer) ASCII format, to read
the IPRT reference results, and to plot the comparisons.

Key Functions
-------------
convert_sgout_to_iprtout
    Convert SMART-G output into the IPRT ASCII output format.
select_and_plot_polar_iprt
    Select I, Q, U and V results from an IPRT matrix and plot
    them in polar coordinates.
plot_iprt_radiances
    Plot radiances and the differences between a reference model
    and the model radiances.
read_phase_nth_cte
    Read a libRadtran or IPRT aerosol/cloud file and convert it
    to a LUT object.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from luts.luts import LUT
import xarray as xr



def select_iprt_iquv(model_val, z_alti, thetas=None, phis=None, inv_thetas=False, inv_phis=False, change_u_sign=False,
                      i_index=int(6), va_index=int(4), phi_index=int(5), z_index=int(1), stdev=False):
    """
    Description: Select U,Q,U and V results from IPRT matrix results

    === Parameters:
    model_val       : Matrix with the model values (read from IPRT phase A result files)
    z_alti          : Keep only results at this z_alti
    thetas          : Keep only results with these theta values
    phis            : Same as thetas but with phi values
    inv_thetas      : Inverse the vector with theta values
    inv_phis        : Same as inv_thetas but with phi values
    change_u_sign   : multiply by -1 the U results (can be useful since in backward and forward the convention change)
    i_index         : In case we don't follow exactly the IPRT output format convention we can specify the index where I begin,
                      but next the order must be the same: I, Q, U and then V.
    va_index        : Same as i_index but with va (VZA)
    phi_index       : Same as i_index but with phi (VAA)
    stdev           : If true return also IQUV stdev

    === Retrun
    I,...,V              : The selected values of I, Q, U and V
    I,...,V,Istd,...Vstd : If stdev = True retrun also Istd to Vstd
    """

    n_records = model_val.shape[0]
    if thetas is None:
        s_thetas = []
        for i in range (0, n_records):
            if (model_val[i, z_index] ==z_alti): s_thetas.append(model_val[i,va_index])
        thetas = np.sort(np.unique(np.array(s_thetas)))
    
    if phis is None:
        s_phis = []
        for i in range (0, n_records):
            if (model_val[i, z_index] ==z_alti): s_phis.append(model_val[i,phi_index])
        phis = np.sort(np.unique(np.array(s_phis)))
    
    n_theta = len(thetas)
    n_phi = len(phis)

    stokes_i = np.zeros((n_theta, n_phi))
    stokes_q = np.zeros((n_theta, n_phi))
    stokes_u = np.zeros((n_theta, n_phi))
    stokes_v = np.zeros((n_theta, n_phi))

    if stdev:
        stokes_i_std = np.zeros((n_theta, n_phi))
        stokes_q_std = np.zeros((n_theta, n_phi))
        stokes_u_std = np.zeros((n_theta, n_phi))
        stokes_v_std = np.zeros((n_theta, n_phi))
    
    if change_u_sign: u_sign = int(-1)
    else            : u_sign = int(1)

    for i in range (0, n_records):
        if (model_val[i, z_index] == z_alti 
                and True in (thetas == model_val[i,va_index])
                and True in (phis == model_val[i,phi_index])):
            if inv_thetas: indi = int(np.squeeze(np.argwhere(thetas == model_val[i,va_index])))
            else         : indi = n_theta-1-int(np.squeeze(np.argwhere(thetas == model_val[i,va_index])))
            if inv_phis  : indj = n_phi-1-int(np.squeeze(np.argwhere(phis == model_val[i,phi_index])))
            else         : indj = int(np.squeeze(np.argwhere(phis == model_val[i,phi_index])))
            stokes_i[indi,indj] =  model_val[i, i_index]
            stokes_q[indi,indj] =  model_val[i, i_index+1]
            stokes_u[indi,indj] =  model_val[i, i_index+2]*u_sign
            stokes_v[indi,indj] =  model_val[i, i_index+3]
            if stdev:
                stokes_i_std[indi,indj] =  model_val[i, i_index+4]
                stokes_q_std[indi,indj] =  model_val[i, i_index+5]
                stokes_u_std[indi,indj] =  model_val[i, i_index+6]*u_sign
                stokes_v_std[indi,indj] =  model_val[i, i_index+7]

    if not stdev:
        return stokes_i, stokes_q, stokes_u, stokes_v
    else:
        return stokes_i, stokes_q, stokes_u, stokes_v, stokes_i_std, stokes_q_std, stokes_u_std, stokes_v_std


def select_and_plot_polar_iprt(model_val, z_alti, depol=None, thetas=None, phis=None, inv_thetas=False, inv_phis=False, change_q_sign=False, change_u_sign=False,
                               change_v_sign=False, max_i=None, max_q=None, max_u=None, max_v=None,  cmap_i=None, cmap_q=None, cmap_u=None, cmap_v=None,
                               force_iquv = None, title=None, save_fig=None, sym=False, i_index=int(6), va_index=int(4),
                               phi_index=int(5), z_index=int(1), depol_index=int(0), output_iquv=False, output_iquv_std=False, avoid_plot=False):
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
    change_u_sign   : multiply by -1 the U results (can be useful since in backward and forward the convention change)
    change_v_sign   : same as change_u_sign but with V
    max_i,...,max_v   : We can specify the max values in I, Q, U and V for the plots
    cmap_i,...,cmap_v : We can specify a specific color map for I, Q, U or/and V results
    force_iquv       : Circumvent the result selection of model_val by giving direclty the I, Q, U and V values (list of matrices)
    title           : Title of the global plot
    save_fig        : If given, save the figure at the given format, e.g save_fig='myFigName.png'
    sym             : IPRT phi results are from 0 to 180 deg, if sym = True plot the symmetrical results from 180 to 360 deg
    i_index         : In case we don't follow exactly the IPRT output format convention we can specify the index where I begin,
                      but next the order must be the same: I, Q, U and then V.
    va_index        : Same as i_index but with va (VZA)
    phi_index       : Same as i_index but with phi (VAA)
    output_iquv      : If True, retrun I, Q, U and V values
    output_iquv_std   : If True, retrun I, Q, U and V stdev values
    avoid_plot      : Do not plot, can be useful if we only want to get the I, Q, U and V values

    === Retrun
    valI,...,valV : if output_iquv is True return the selected values of I, Q, U and V, else return nothing
    """

    n_records = model_val.shape[0]
    if thetas is None:
        s_thetas = []
        for i in range (0, n_records):
            cond_z_depol = (depol is None and (model_val[i, z_index] == z_alti)) or (depol is not None and ((model_val[i, z_index] == z_alti) and (model_val[i, depol_index] == depol)))
            if (cond_z_depol): s_thetas.append(model_val[i,va_index])
        thetas = np.sort(np.unique(np.array(s_thetas)))
    
    if phis is None:
        s_phis = []
        for i in range (0, n_records):
            cond_z_depol = (depol is None and (model_val[i, z_index] == z_alti)) or (depol is not None and ((model_val[i, z_index] == z_alti) and (model_val[i, depol_index] == depol)))
            if (cond_z_depol): s_phis.append(model_val[i,phi_index])
        phis = np.sort(np.unique(np.array(s_phis)))
    
    if sym: phis = np.concatenate((phis, phis+180))
    n_theta = len(thetas)
    n_phi = len(phis)
    if sym: n_phi_data = round(n_phi/2)
    else: n_phi_data = n_phi

    val_i = np.zeros((n_theta, n_phi))
    val_q = np.zeros((n_theta, n_phi))
    val_u = np.zeros((n_theta, n_phi))
    val_v = np.zeros((n_theta, n_phi))

    if output_iquv_std:
        val_i_std = np.zeros((n_theta, n_phi_data))
        val_q_std = np.zeros((n_theta, n_phi_data))
        val_u_std = np.zeros((n_theta, n_phi_data))
        val_v_std = np.zeros((n_theta, n_phi_data))

    if change_q_sign: q_sign = int(-1)
    else            : q_sign = int(1) 
    if change_u_sign: u_sign = int(-1)
    else            : u_sign = int(1)
    if change_v_sign: v_sign = int(-1)
    else            : v_sign = int(1)

    if force_iquv is not None:
        val_i[:,0:n_phi_data] = force_iquv[0]
        val_q[:,0:n_phi_data] = force_iquv[1]
        val_u[:,0:n_phi_data] = force_iquv[2]*u_sign
        val_v[:,0:n_phi_data] = force_iquv[3]
    else:
        for i in range (0, n_records):
            cond_z_depol = (depol is None and (model_val[i, z_index] == z_alti)) or (depol is not None and ((model_val[i, z_index] == z_alti) and (model_val[i, depol_index] == depol)))
            if (cond_z_depol
                and True in (thetas == model_val[i,va_index])
                and True in (phis == model_val[i,phi_index])  ):
                if inv_thetas: indi = int(np.squeeze(np.argwhere(thetas == model_val[i,va_index])))
                else         : indi = n_theta-1-int(np.squeeze(np.argwhere(thetas == model_val[i,va_index])))
                if inv_phis  : indj = n_phi_data-1-int(np.squeeze(np.argwhere(phis[0:n_phi_data] == model_val[i,phi_index])))
                else         : indj = int(np.squeeze(np.argwhere(phis[0:n_phi_data] == model_val[i,phi_index])))
                val_i[indi,indj] =  model_val[i, i_index]
                val_q[indi,indj] =  model_val[i, i_index+1]*q_sign
                val_u[indi,indj] =  model_val[i, i_index+2]*u_sign
                val_v[indi,indj] =  model_val[i, i_index+3]*v_sign
                if (output_iquv_std): 
                    val_i_std[indi,indj] =  model_val[i, i_index+4]
                    val_q_std[indi,indj] =  model_val[i, i_index+5]
                    val_u_std[indi,indj] =  model_val[i, i_index+6]
                    val_v_std[indi,indj] =  model_val[i, i_index+7] 

    if sym:
        for i in range(n_theta):
                for j in range(n_phi_data):
                    val_i[i,n_phi_data+j] =  val_i[i,n_phi_data-j-1]
                    val_q[i,n_phi_data+j] =  val_q[i,n_phi_data-j-1]
                    val_u[i,n_phi_data+j] =  val_u[i,n_phi_data-j-1]
                    val_v[i,n_phi_data+j] =  val_v[i,n_phi_data-j-1]


    if not avoid_plot:
        plt.rcParams.update({'font.size':13})

        thetas_scaled = (thetas - np.min(thetas))/(np.max(thetas)- np.min(thetas))*90.
        if max_i is None:
            max_i = max(np.abs(np.min(val_i)), np.abs(np.max(val_i)))
            min_i = 0.
        else:
            min_i=-max_i
        if max_q is None: max_q = max(np.abs(np.min(val_q)), np.abs(np.max(val_q)))
        if max_u is None: max_u = max(np.abs(np.min(val_u)), np.abs(np.max(val_u)))
        if max_v is None: max_v = max(np.abs(np.min(val_v)), np.abs(np.max(val_v)))

        if cmap_i is None: cmap_i = "jet"
        if cmap_q is None: cmap_q = "RdBu_r"
        if cmap_u is None: cmap_u = "RdBu_r"
        if cmap_v is None: cmap_v = "RdBu_r"

        fig, ax = plt.subplots(1,4, figsize=(12,4),subplot_kw=dict(projection='polar'))
        if title is not None: fig.suptitle(title)
        #csI = ax[0].contourf(np.deg2rad(phis), thetas[::-1], valI, cmap='jet', levels=np.linspace(0., 9.5e-2, 100, endpoint=True))
        ax[0].grid(False)
        cs_i = ax[0].pcolormesh(np.deg2rad(phis), thetas_scaled[::-1], val_i, cmap=cmap_i, vmin=min_i, vmax=max_i, shading='gouraud')
        cbar_i = fig.colorbar(cs_i, ax=ax[0], shrink=0.8, orientation='horizontal', ticks=np.linspace(min_i, max_i, 3, endpoint=True), format="%4.1e")
        cbar_i.set_label(r'I')
        ax[0].set_yticklabels([])
        ax[0].grid(axis='both', linewidth=1.5, linestyle=':', color='black', alpha=0.5)

        #csQ = ax[1].contourf(np.deg2rad(phis), thetas[::-1], valQ, cmap='RdBu_r', levels=np.linspace(-1.4e-2, 1.4e-2, 100, endpoint=True))
        ax[1].grid(False)
        cs_q = ax[1].pcolormesh(np.deg2rad(phis), thetas_scaled[::-1], val_q, cmap=cmap_q, vmin=-max_q, vmax=max_q, shading='gouraud')
        cbar_q = fig.colorbar(cs_q, ax=ax[1], shrink=0.8, orientation='horizontal', ticks=np.linspace(-max_q, max_q, 3, endpoint=True), format="%4.1e")
        cbar_q.set_label(r'Q')
        ax[1].set_yticklabels([])
        ax[1].grid(axis='both', linewidth=1.5, linestyle=':', color='black', alpha=0.5)

        #csU = ax[2].contourf(np.deg2rad(phis), thetas[::-1], -valU, cmap='RdBu_r', levels=np.linspace(-2.6e-2, 2.6e-2, 100, endpoint=True))
        ax[2].grid(False)
        cs_u = ax[2].pcolormesh(np.deg2rad(phis), thetas_scaled[::-1], val_u, cmap=cmap_u, vmin=-max_u, vmax=max_u, shading='gouraud')
        cbar_u = fig.colorbar(cs_u, ax=ax[2], shrink=0.8, orientation='horizontal', ticks=np.linspace(-max_u, max_u, 3, endpoint=True), format="%4.1e")
        cbar_u.set_label(r'U')
        ax[2].set_yticklabels([])
        ax[2].grid(axis='both', linewidth=1.5, linestyle=':', color='black', alpha=0.5)

        #csV = ax[3].contourf(np.deg2rad(phis), thetas[::-1], valV, cmap='RdBu_r', levels=np.linspace(-1e-5, 1e-5, 100, endpoint=True))
        ax[3].grid(False)
        cs_v = ax[3].pcolormesh(np.deg2rad(phis), thetas_scaled[::-1], val_v, cmap=cmap_v, vmin=-max_v, vmax=max_v, shading='gouraud')
        cbar_v = fig.colorbar(cs_v, ax=ax[3], shrink=0.8, orientation='horizontal', ticks=np.linspace(-max_v, max_v, 3, endpoint=True), format="%4.1e")
        cbar_v.set_label(r'V')
        ax[3].set_yticklabels([])
        ax[3].grid(axis='both', linewidth=1.5, linestyle=':', color='black', alpha=0.5)
        
        fig.tight_layout()
        if save_fig is not None: plt.savefig(save_fig)

    if output_iquv and output_iquv_std:
        return val_i[:,0:n_phi_data], val_q[:,0:n_phi_data], val_u[:,0:n_phi_data], val_v[:,0:n_phi_data], val_i_std, val_q_std, val_u_std, val_v_std
    elif (output_iquv):
        return val_i[:,0:n_phi_data], val_q[:,0:n_phi_data], val_u[:,0:n_phi_data], val_v[:,0:n_phi_data]
    elif (output_iquv_std) :
        val_i_std, val_q_std, val_u_std, val_v_std

def convert_sgout_to_iprtout(datasets, u_signs, case_name, depols, altitudes, szas, saas, vzas, vaas, file_name, output_layer=None, interp=False):
    """
    Description: Convert SMART-G output into IPRT ascii output format

    === Parameters:
    datasets           : List of SMART-G output (xarray Dataset; MLUT input is deprecated)
    u_signs      : List with multiplication to perform to U of each output
    case_name    : The IPRT case name
    depols       : List of Depol values
    altitudes         : List with the viewing altitude of each output
    szas         : List of Sun Zenith Angles
    saas         : List of Sun Azimuth Angles
    vzas         : List or numpy 1d array with VZA values
    vaas         : Same as vzas but with VAA values
    file_name    : The name of the ascci file to be created
    output_layer : If None the output layer is always '_up (TOA)', else a list with wanted ones ('_down (0+)', ...)

    """
    output =  "# IPRT case " + case_name + "\n"
    output += "# RT model: SMARTG\n"
    output += "# depol altitude sza saa va phi I Q U V Istd Qstd Ustd Vstd\n"

    for im, m in enumerate(datasets):
        if hasattr(m, 'to_xarray'): m = m.to_xarray()  # legacy MLUT input
        fac = np.cos(np.radians(szas[im]))/np.pi
        vza = vzas[im]
        vaa = vaas[im]
        if output_layer is None:
            layer = '_up (TOA)'
        else:
            layer = output_layer[im]
        for iza, za, in enumerate(vza):
            for iaa, aa, in enumerate(vaa):
                if not interp:
                    stokes_i = float(m['I'+layer][iaa,iza])*fac
                    stokes_q = float(m['Q'+layer][iaa,iza])*fac
                    stokes_u = float(m['U'+layer][iaa,iza])*fac*u_signs[im]
                    stokes_v = float(m['V'+layer][iaa,iza])*fac

                    stokes_i_std = float(m['I_stdev'+layer][iaa,iza])*fac
                    stokes_q_std = float(m['Q_stdev'+layer][iaa,iza])*fac
                    stokes_u_std = float(m['U_stdev'+layer][iaa,iza])*fac
                    stokes_v_std = float(m['V_stdev'+layer][iaa,iza])*fac
                else:
                    pos = {'Azimuth angles': aa, 'Zenith angles': za}
                    stokes_i = float(m['I'+layer].interp(pos))*fac
                    stokes_q = float(m['Q'+layer].interp(pos))*fac
                    stokes_u = float(m['U'+layer].interp(pos))*fac*u_signs[im]
                    stokes_v = float(m['V'+layer].interp(pos))*fac

                    stokes_i_std = float(m['I_stdev'+layer].interp(pos))*fac
                    stokes_q_std = float(m['Q_stdev'+layer].interp(pos))*fac
                    stokes_u_std = float(m['U_stdev'+layer].interp(pos))*fac
                    stokes_v_std = float(m['V_stdev'+layer].interp(pos))*fac
                output+= f"{depols[im]:.2f} {altitudes[im]:.1f} {szas[im]:.1f} {saas[im]:.1f} {za:.1f} {aa:.1f} {stokes_i:.5e} " + \
                         f"{stokes_q:.5e} {stokes_u:.5e} {stokes_v:.5e} {stokes_i_std:.5e} {stokes_q_std:.5e} {stokes_u_std:.5e} {stokes_v_std:.5e}\n"

    with open(file_name, 'w') as f:
        f.write(output)

def plot_iprt_radiances(iquv_obs, iquv_mod, iquv_std_obs, iquv_std_mod, xaxis, xlabel, iquv_ymin=None, iquv_ymax=None, title=None, save_fig=None):
    """
    Description: Plot radiances and dif between observation (or reference model) and model radiances

    === Parameters:
    iquv_obs, iquv_mod       : Numpy matrices with observation and model IQUV signals (matrix of dim [NSTK, NXAXIS])
    iquv_std_obs, iquv_std_mod : Numpy matrices with observation and model IQUV signal stdev
    xaxis                    : IQUV signals are varying as function of xaxis (can be VZA or VAA)
    xlabel                   : Plot xaxis label
    iquv_ymin                 : Min IQUV values to set for radiance plot ylim
    iquv_ymax                 : Max IQUV values to set for radiance plot ylim
    title                    : Title of the global plot
    save_fig                 : If given, save the figure at the given format, e.g save_fig='myFigName.png'
    """

    fig, ax = plt.subplots(2,4, figsize=(13,8))
    if title: fig.suptitle(title, fontsize=15)

    for istk in range (0, 4):
        if (istk==0): ax[0,istk].set_ylabel("normalized radiance", fontsize=13)
        ax[0,istk].set_xlabel(xlabel, fontsize=13)
        ax[0,istk].yaxis.set_major_formatter(mtick.FormatStrFormatter('%5.1e'))
        ax[0,istk].plot(xaxis, iquv_obs[istk], color='red')
        ax[0,istk].plot(xaxis, iquv_mod[istk], color='blue')
        if iquv_ymin is not None and iquv_ymax is not None:
            ax[0,istk].set_yticks(np.linspace(iquv_ymin[istk], iquv_ymax[istk], 6))
            ax[0,istk].set_yticks(np.linspace(iquv_ymin[istk], iquv_ymax[istk], 6))
            ax[0,istk].set_ylim(ymin=iquv_ymin[istk], ymax=iquv_ymax[istk])
        else:
            yt = ax[0,istk].get_yticks()
            ax[0,istk].locator_params(axis='y', nbins=6)
            if iquv_ymin is not None:
                ax[0,istk].set_yticks(np.linspace(iquv_ymin[istk], np.max(yt), 6))
                ax[0,istk].set_yticks(np.linspace(iquv_ymin[istk], np.max(yt), 6))
                ax[0,istk].set_ylim(ymin=iquv_ymin[istk], ymax=np.max(yt))
            elif iquv_ymax is not None:
                ax[0,istk].set_yticks(np.linspace(np.min(yt), iquv_ymax[istk], 6))
                ax[0,istk].set_yticks(np.linspace(np.min(yt), iquv_ymax[istk], 6))
                ax[0,istk].set_ylim(ymin=np.min(yt), ymax=iquv_ymax[istk])
            else:
                ax[0,istk].set_yticks(np.linspace(np.min(yt), np.max(yt), 6))
                ax[0,istk].set_yticks(np.linspace(np.min(yt), np.max(yt), 6))
                ax[0,istk].set_ylim(ymin=np.min(yt), ymax=np.max(yt))
        ax[0,istk].set_xlim(xmin=np.min(xaxis), xmax=np.max(xaxis))
        ax[0,istk].locator_params(axis='x', nbins=3)

        if (istk==0): ax[1,istk].set_ylabel("abs. diff", fontsize=13)
        ax[1,istk].set_xlabel(xlabel, fontsize=13)
        ax[1,istk].yaxis.set_major_formatter(mtick.FormatStrFormatter('%5.1e'))
        markers, caps, bars = ax[1,istk].errorbar(xaxis, iquv_obs[istk,:]-iquv_mod[istk,:], yerr=iquv_std_obs[istk]+iquv_std_mod[istk], fmt='x', color='blue', ecolor='grey', capsize=2)
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
    
def compute_deltam_iprtout(obs, mod, i_obs_id=6, i_mod_id=6, print_res=True):
    if (not isinstance(obs, np.ndarray) or not isinstance(mod, np.ndarray)): raise NameError("obs and mod must be np.ndarray!")
    id_obs = [i_obs_id, i_obs_id+1, i_obs_id+2, i_obs_id+3]
    id_mod = [i_mod_id, i_mod_id+1, i_mod_id+2, i_mod_id+3]
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



def group_iquv(i_list, q_list, u_list, v_list):

    n_values = int(0)
    i_tot = i_list[0].flatten()
    q_tot = q_list[0].flatten()
    u_tot = u_list[0].flatten()
    v_tot = v_list[0].flatten()

    for i in range (len(i_list)):
        n_values += round(i_list[i].shape[0]*i_list[i].shape[1])
        if (i > 0):
            i_tot = np.concatenate((i_tot, i_list[i].flatten()))
            q_tot = np.concatenate((q_tot, q_list[i].flatten()))
            u_tot = np.concatenate((u_tot, u_list[i].flatten()))
            v_tot = np.concatenate((v_tot, v_list[i].flatten()))

    iquv_tot = np.zeros((4,n_values), dtype=np.float32)
    iquv_tot[0,:] = i_tot
    iquv_tot[1,:] = q_tot
    iquv_tot[2,:] = u_tot
    iquv_tot[3,:] = v_tot

    return iquv_tot

def read_phase_nth_cte(filename, nb_theta=int(721), convert_ipar_iper=True, normalize=False):
    """
    Read libRatran aerosol/cloud files (i.g. wc.sol.mie.cdf) or monochromatic IPRT netcdf aerosol/cloud files,
    and convert to LUT object with a constant theta discretisation i.e. nb_theta = cte.

    Parameters
    ----------
    filename : str 
        File name with path location of netcdf file.
    nb_theta : int
        Number of theta discretization between 0 and 180 degrees.
    convert_ipar_iper : bool
        Convert IQUV phase matrix into IparIperUV phase matrix
    normalize : bool
        Normalize such that the integral of P0 is equal to 2
        
    Returns
    -------
    out : LUT
        The cloud phase matrix with a constant theta number
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

    n_stk   = ds.nphamat.size
    n_theta = nb_theta
    n_rh_or_reff  = rh_reff.size
    n_wav    = ds["wavelen"].size
    theta = np.linspace(0., 180., num=n_theta)
    wavelength = ds["wavelen"].data*1e3

    phase_matrix = LUT( np.full((n_wav, n_rh_or_reff, 6, n_theta), np.nan, dtype=np.float32),
                axes=[wavelength, rh_reff, None, theta],
                names=['wav_phase', rh_or_reff, 'stk', 'theta_atm'],
                desc="phase_atm" )

    for iwav in range (0, n_wav):
        for irhreff in range(n_rh_or_reff):
            for istk in range (n_stk):
                # ntheta (wl, reff, stk)
                nth = ds["ntheta"][iwav, irhreff, istk].data

                # theta (wl, reff, stk, ntheta)
                th = ds["theta"][iwav, irhreff, istk, :].data

                phase_matrix.data[iwav, irhreff, istk, :] = np.interp(theta, th[:nth], phase[iwav,irhreff,istk,:nth],  period=np.inf)
    if n_stk == 4:
        phase_matrix.data[:,:,4,:] = phase_matrix.data[:,:,0,:].copy()
        phase_matrix.data[:,:,5,:] = phase_matrix.data[:,:,2,:].copy()

    if normalize:
        for iwav in range (0, n_wav):
            for irhreff in range (0, n_rh_or_reff):
                # Note: from Ipar Iper phase, if NBSTK=4 -> P0=(P11+P12)/2, and if NBSTK=6 -> P0=(P11+P22+2*P12)/2
                f = phase_matrix.data[iwav,irhreff,0,:]
                mu= np.cos(np.radians(theta))
                norm = np.trapezoid(f,-mu)
                phase_matrix.data[iwav,irhreff,:,:] *= 2./abs(norm)

    if (convert_ipar_iper):
        # convert I, Q into Ipar, Iper
        if (n_stk == 4): # only spherical particles
            p0 = phase_matrix.data[:,:,0,:].copy()
            p1 = phase_matrix.data[:,:,1,:].copy()
            p4 = phase_matrix.data[:,:,4,:].copy()
            phase_matrix.data[:,:,0,:] = 0.5*(p0+2*p1+p4) # P11
            phase_matrix.data[:,:,1,:] = 0.5*(p0-p4)      # P12=P21
            phase_matrix.data[:,:,4,:] = 0.5*(p0-2*p1+p4) # P22
        elif (n_stk) == 6: # spherical or non spherical particles
            # note: the sign of P43/P34 affects only the sign of V, since V=0 for rayleigh scattering it does not matter 
            p0 = phase_matrix.data[:,:,0,:].copy()
            p1 = phase_matrix.data[:,:,1,:].copy()
            p4 = phase_matrix.data[:,:,4,:].copy()
            phase_matrix.data[:,:,0,:] = 0.5*(p0+2*p1+p4) # P11
            phase_matrix.data[:,:,1,:] = 0.5*(p0-p4)      # P12=P21
            phase_matrix.data[:,:,4,:] = 0.5*(p0-2*p1+p4) # P22
        else:
            raise NameError("Number of unique phase components is different than 4 or 6!")
        
    return phase_matrix