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
select_iprt_iquv
    Select I, Q, U and V results from an IPRT matrix.
select_and_plot_polar_iprt
    Select I, Q, U and V results from an IPRT matrix and plot
    them in polar coordinates.
plot_iprt_radiances
    Plot radiances and the differences between a reference model
    and the model radiances.
group_iquv
    Gather several IQUV result matrices into a single array.
compute_deltam
    Compute the IPRT delta_m metric between two IQUV signal sets.
compute_deltam_iprtout
    Compute the IPRT delta_m metric between two IPRT matrices.
read_phase_nth_cte
    Read a libRadtran or IPRT aerosol/cloud file and convert it
    to a LUT object.
"""

from pathlib import Path
import warnings

from luts.luts import LUT, MLUT
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import xarray as xr



def select_iprt_iquv(
    model_val: np.ndarray,
    z_alti: float,
    thetas: np.ndarray | None = None,
    phis: np.ndarray | None = None,
    inv_thetas: bool = False,
    inv_phis: bool = False,
    change_u_sign: bool = False,
    i_index: int = 6,
    va_index: int = 4,
    phi_index: int = 5,
    z_index: int = 1,
    stdev: bool = False,
) -> tuple[np.ndarray, ...]:
    """Select the I, Q, U and V results from an IPRT result matrix.

    The records of the matrix are scattered over the (theta, phi)
    grid, one Stokes parameter per output array.

    Parameters
    ----------
    model_val : ndarray
        Model values, as read from an IPRT phase A result file: one
        row per record, the columns following the IPRT convention.
    z_alti : float
        Keep only the records at this altitude, in km.
    thetas : ndarray, optional
        Keep only the records with these viewing zenith angles, in
        degrees. By default all the angles found at z_alti are kept,
        sorted in increasing order.
    phis : ndarray, optional
        Same as thetas, for the viewing azimuth angles.
    inv_thetas : bool
        Store the results by increasing zenith angle. By default the
        zenith axis is reversed.
    inv_phis : bool
        Store the results by decreasing azimuth angle. By default the
        azimuth axis follows the increasing order.
    change_u_sign : bool
        Multiply U by -1, the convention for U being opposite in the
        backward and in the forward mode.
    i_index : int
        Column index where I is found. Q, U and V must follow it in
        that order, and their standard deviations right after them.
    va_index : int
        Column index of the viewing zenith angle.
    phi_index : int
        Column index of the viewing azimuth angle.
    z_index : int
        Column index of the altitude.
    stdev : bool
        Also return the standard deviations of I, Q, U and V.

    Returns
    -------
    tuple of ndarray
        The I, Q, U and V values, each of shape (ntheta, nphi),
        followed by their standard deviations in the same order when
        stdev is True.
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


def select_and_plot_polar_iprt(
    model_val: np.ndarray,
    z_alti: float,
    depol: float | None = None,
    thetas: np.ndarray | None = None,
    phis: np.ndarray | None = None,
    inv_thetas: bool = False,
    inv_phis: bool = False,
    change_q_sign: bool = False,
    change_u_sign: bool = False,
    change_v_sign: bool = False,
    max_i: float | None = None,
    max_q: float | None = None,
    max_u: float | None = None,
    max_v: float | None = None,
    cmap_i: str | Colormap | None = None,
    cmap_q: str | Colormap | None = None,
    cmap_u: str | Colormap | None = None,
    cmap_v: str | Colormap | None = None,
    force_iquv: list[np.ndarray] | None = None,
    title: str | None = None,
    save_fig: str | Path | None = None,
    sym: bool = False,
    i_index: int = 6,
    va_index: int = 4,
    phi_index: int = 5,
    z_index: int = 1,
    depol_index: int = 0,
    output_iquv: bool = False,
    output_iquv_std: bool = False,
    avoid_plot: bool = False,
) -> tuple[np.ndarray, ...]:
    """Select the I, Q, U and V results and plot them in polar view.

    The selection follows select_iprt_iquv, with the depolarisation
    factor as an extra filter, and the four Stokes parameters are
    drawn side by side on polar axes.

    Parameters
    ----------
    model_val : ndarray
        Model values, as read from an IPRT phase A result file: one
        row per record, the columns following the IPRT convention.
    z_alti : float
        Keep only the records at this altitude, in km.
    depol : float, optional
        Keep only the records with this depolarisation factor. By
        default the factor is not used to filter the records.
    thetas : ndarray, optional
        Keep only the records with these viewing zenith angles, in
        degrees. By default all the angles found at z_alti are kept,
        sorted in increasing order.
    phis : ndarray, optional
        Same as thetas, for the viewing azimuth angles.
    inv_thetas : bool
        Store the results by increasing zenith angle. By default the
        zenith axis is reversed.
    inv_phis : bool
        Store the results by decreasing azimuth angle. By default the
        azimuth axis follows the increasing order.
    change_q_sign : bool
        Multiply Q by -1.
    change_u_sign : bool
        Multiply U by -1, the convention for U being opposite in the
        backward and in the forward mode.
    change_v_sign : bool
        Multiply V by -1.
    max_i, max_q, max_u, max_v : float, optional
        Upper bound of the colour scale of each panel. By default the
        largest absolute value of the panel is used. I is drawn from
        0 to max_i, the other panels from -max to +max.
    cmap_i, cmap_q, cmap_u, cmap_v : str or Colormap, optional
        Colour map of each panel, 'jet' for I and 'RdBu_r' for the
        other panels by default.
    force_iquv : list of ndarray, optional
        Plot these I, Q, U and V matrices instead of selecting them
        from model_val.
    title : str, optional
        Title of the whole figure.
    save_fig : str or Path, optional
        Save the figure at this path, the extension giving the
        format, e.g. save_fig='myFigName.png'.
    sym : bool
        The IPRT azimuth angles cover 0 to 180 degrees; also plot the
        symmetrical results from 180 to 360 degrees.
    i_index : int
        Column index where I is found. Q, U and V must follow it in
        that order, and their standard deviations right after them.
    va_index : int
        Column index of the viewing zenith angle.
    phi_index : int
        Column index of the viewing azimuth angle.
    z_index : int
        Column index of the altitude.
    depol_index : int
        Column index of the depolarisation factor.
    output_iquv : bool
        Return the selected I, Q, U and V values.
    output_iquv_std : bool
        Return the selected standard deviations of I, Q, U and V.
    avoid_plot : bool
        Skip the plot, which is useful when only the selected values
        are needed.

    Returns
    -------
    tuple of ndarray
        The I, Q, U and V values when output_iquv is True, followed
        by their standard deviations when output_iquv_std is also
        True. The standard deviations alone when they are the only
        ones requested, and an empty tuple when neither is.
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
            max_i = float(max(np.abs(np.min(val_i)), np.abs(np.max(val_i))))
            min_i = 0.
        else:
            min_i=-max_i
        if max_q is None: max_q = float(max(np.abs(np.min(val_q)), np.abs(np.max(val_q))))
        if max_u is None: max_u = float(max(np.abs(np.min(val_u)), np.abs(np.max(val_u))))
        if max_v is None: max_v = float(max(np.abs(np.min(val_v)), np.abs(np.max(val_v))))

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
        return val_i_std, val_q_std, val_u_std, val_v_std
    return ()

def convert_sgout_to_iprtout(
    datasets: list[xr.Dataset | MLUT],
    u_signs: list[float],
    case_name: str,
    depols: list[float],
    altitudes: list[float],
    szas: list[float],
    saas: list[float],
    vzas: list[np.ndarray],
    vaas: list[np.ndarray],
    file_name: str | Path,
    output_layer: list[str] | None = None,
    interp: bool = False,
) -> None:
    """Convert SMART-G outputs into the IPRT ASCII output format.

    All the list arguments are parallel to datasets: they hold, for
    each output, the geometry it was computed with. The radiances are
    normalised by cos(sza)/pi before being written.

    Parameters
    ----------
    datasets : list of Dataset
        SMART-G outputs. Legacy MLUT outputs are still accepted but
        are deprecated.
    u_signs : list of float
        Factor applied to U of each output, to reconcile the backward
        and the forward convention.
    case_name : str
        IPRT case name, written in the file header.
    depols : list of float
        Depolarisation factor of each output.
    altitudes : list of float
        Viewing altitude of each output, in km.
    szas : list of float
        Sun zenith angle of each output, in degrees.
    saas : list of float
        Sun azimuth angle of each output, in degrees.
    vzas : list of ndarray
        Viewing zenith angles of each output, in degrees.
    vaas : list of ndarray
        Viewing azimuth angles of each output, in degrees.
    file_name : str or Path
        Path of the ASCII file to write.
    output_layer : list of str, optional
        Output layer of each output, e.g. '_down (0+)'. By default
        '_up (TOA)' is used for all of them.
    interp : bool
        Interpolate the outputs at the requested angles instead of
        reading them at the matching indices.
    """
    output =  "# IPRT case " + case_name + "\n"
    output += "# RT model: SMARTG\n"
    output += "# depol altitude sza saa va phi I Q U V Istd Qstd Ustd Vstd\n"

    for im, m in enumerate(datasets):
        if isinstance(m, MLUT):
            warn_message = (
                "\nUsing an MLUT in datasets is deprecated, use an "
                + "xarray.Dataset instead."
            )
            warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
            m = m.to_xarray()
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

def plot_iprt_radiances(
    iquv_obs: np.ndarray,
    iquv_mod: np.ndarray,
    iquv_std_obs: np.ndarray,
    iquv_std_mod: np.ndarray,
    xaxis: np.ndarray,
    xlabel: str,
    iquv_ymin: np.ndarray | list[float] | None = None,
    iquv_ymax: np.ndarray | list[float] | None = None,
    title: str | None = None,
    save_fig: str | Path | None = None,
) -> None:
    """Plot radiances and their difference with a reference.

    The figure holds two rows of four panels: the observed and the
    modelled I, Q, U and V on top, and their absolute difference,
    with error bars, below.

    Parameters
    ----------
    iquv_obs : ndarray
        Observed, or reference model, I, Q, U and V signals, of shape
        (4, nxaxis).
    iquv_mod : ndarray
        Modelled I, Q, U and V signals, of the same shape.
    iquv_std_obs : ndarray
        Standard deviations of the observed signals.
    iquv_std_mod : ndarray
        Standard deviations of the modelled signals.
    xaxis : ndarray
        Abscissa the signals vary along, usually the viewing zenith
        or the viewing azimuth angle.
    xlabel : str
        Label of the abscissa.
    iquv_ymin : ndarray or list of float, optional
        Lower bound of the radiance panels, one per Stokes parameter.
        By default it is taken from the drawn values.
    iquv_ymax : ndarray or list of float, optional
        Upper bound of the radiance panels, one per Stokes parameter.
    title : str, optional
        Title of the whole figure.
    save_fig : str or Path, optional
        Save the figure at this path, the extension giving the
        format, e.g. save_fig='myFigName.png'.
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
        _, caps, bars = ax[1,istk].errorbar(xaxis, iquv_obs[istk,:]-iquv_mod[istk,:], yerr=iquv_std_obs[istk]+iquv_std_mod[istk], fmt='x', color='blue', ecolor='grey', capsize=2)
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
    
def compute_deltam_iprtout(
    obs: np.ndarray,
    mod: np.ndarray,
    i_obs_id: int = 6,
    i_mod_id: int = 6,
    print_res: bool = True,
) -> np.ndarray:
    """Compute the IPRT delta_m metric from two IPRT ASCII matrices.

    delta_m is the root mean square of the difference between the
    model and the reference, relative to the root mean square of the
    reference, in percent. It is reported per Stokes parameter, and
    is set to 0 when the reference is uniformly zero.

    Parameters
    ----------
    obs : ndarray
        Observed, or reference model, values, as read from an IPRT
        result file: one row per record.
    mod : ndarray
        Modelled values, with the records in the same order.
    i_obs_id : int
        Column index where I is found in obs. Q, U and V must follow
        it in that order.
    i_mod_id : int
        Same as i_obs_id, for mod.
    print_res : bool
        Print each delta_m next to its Stokes parameter.

    Returns
    -------
    ndarray
        The delta_m of I, Q, U and V, in percent.
    """
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

def compute_deltam(
    obs: np.ndarray | list[np.ndarray],
    mod: np.ndarray | list[np.ndarray],
    print_res: bool = True,
) -> np.ndarray:
    """Compute the IPRT delta_m metric from two IQUV signal sets.

    delta_m is the root mean square of the difference between the
    model and the reference, relative to the root mean square of the
    reference, in percent. It is reported per Stokes parameter, and
    is set to 0 when the reference is uniformly zero.

    Parameters
    ----------
    obs : ndarray or list of ndarray
        Observed, or reference model, I, Q, U and V signals, either
        as an array of shape (4, nvalues) or as four arrays.
    mod : ndarray or list of ndarray
        Modelled I, Q, U and V signals, in the same layout.
    print_res : bool
        Print each delta_m next to its Stokes parameter.

    Returns
    -------
    ndarray
        The delta_m of I, Q, U and V, in percent.
    """
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



def group_iquv(
    i_list: list[np.ndarray],
    q_list: list[np.ndarray],
    u_list: list[np.ndarray],
    v_list: list[np.ndarray],
) -> np.ndarray:
    """Gather several IQUV result matrices into a single array.

    Each matrix is flattened and the matrices are concatenated in the
    given order, so that several viewing configurations can be
    compared with a single delta_m.

    Parameters
    ----------
    i_list : list of ndarray
        The I matrices to gather.
    q_list : list of ndarray
        The Q matrices, in the same order.
    u_list : list of ndarray
        The U matrices, in the same order.
    v_list : list of ndarray
        The V matrices, in the same order.

    Returns
    -------
    ndarray
        The gathered signals, of shape (4, nvalues).
    """
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

def read_phase_nth_cte(
    filename: str | Path,
    nb_theta: int = 721,
    convert_ipar_iper: bool = True,
    normalize: bool = False,
) -> LUT:
    """Read an aerosol or cloud file on a constant theta grid.

    Both the libRadtran files (e.g. wc.sol.mie.cdf) and the
    monochromatic IPRT netCDF files are accepted. Their phase matrix
    is given on a theta grid whose length varies with the wavelength
    and the component; it is interpolated here on a single grid of
    nb_theta angles, which is what the LUT layout requires.

    Parameters
    ----------
    filename : str or Path
        Path of the netCDF file to read.
    nb_theta : int
        Number of theta values between 0 and 180 degrees.
    convert_ipar_iper : bool
        Convert the IQUV phase matrix into an IparIperUV one.
    normalize : bool
        Normalise the phase matrix so that the integral of P0 is
        equal to 2.

    Returns
    -------
    LUT
        The phase matrix, of shape (nwav, nrh_or_reff, 6, nb_theta),
        with the axes 'wav_phase', 'rh' or 'reff', 'stk' and
        'theta_atm'.
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