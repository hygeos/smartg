"""IPRT phase A tools.

Phase A holds the 1D cases A1 to A6 and B1 to B4. Their results are
ASCII tables, one record per line with the columns

    depol zout sza saa va phi I Q U V Istd Qstd Ustd Vstd

(indices 0 to 13), as in the MYSTIC reference files and as
convert_sgout_to_iprtout writes them. The delta_m metric on I, Q, U
and V arrays is in smartg.iprt.common.

Key Functions
-------------
convert_sgout_to_iprtout
    Convert SMART-G output into the IPRT phase A ASCII format.
select_iprt_iquv
    Select I, Q, U and V results from a phase A matrix.
select_and_plot_polar_iprt
    Select I, Q, U and V results from a phase A matrix and plot
    them in polar coordinates.
compute_deltam_iprtout
    Compute the IPRT delta_m metric between two phase A matrices.
read_iprt_output
    Read a phase A ASCII result file.
merge_least_noisy
    Merge two runs of a case, keeping the least noisy values.
compare_polar_iprt
    Compare a model with a reference in polar view, and their delta_m.
compare_plane_iprt
    Compare a model with a reference along a plane, and their delta_m.
"""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from luts.luts import MLUT
from matplotlib.colors import Colormap

from smartg.iprt.common import compute_deltam, group_iquv
from smartg.view import plot_iquv_comparison, plot_polar_iquv

STOKES = ("I", "Q", "U", "V")


def _iprt_records(
    model_val: np.ndarray,
    z_alti: float,
    depol: float | None,
    thetas: np.ndarray | None,
    phis: np.ndarray | None,
    va_index: int,
    phi_index: int,
    z_index: int,
    depol_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flag the kept records of a phase A matrix and find their angles.

    Parameters
    ----------
    model_val : ndarray
        Model values, as read from an IPRT phase A result file: one
        row per record, the columns following the IPRT convention.
    z_alti : float
        Keep only the records at this altitude, in km.
    depol : float, optional
        Also keep only the records with this depolarisation factor.
    thetas : ndarray, optional
        Viewing zenith angles, in degrees. By default all the angles
        of the kept records, sorted in increasing order.
    phis : ndarray, optional
        Same as thetas, for the viewing azimuth angles.
    va_index, phi_index, z_index, depol_index : int
        Column indices of the viewing zenith angle, the viewing
        azimuth angle, the altitude and the depolarisation factor.

    Returns
    -------
    keep : ndarray of bool
        True for the records at z_alti, and at depol when it is given.
    thetas : ndarray
        The given, or found, viewing zenith angles.
    phis : ndarray
        The given, or found, viewing azimuth angles.
    """
    keep = model_val[:, z_index] == z_alti
    if depol is not None:
        keep &= model_val[:, depol_index] == depol
    if thetas is None:
        thetas = np.unique(model_val[keep, va_index])
    if phis is None:
        phis = np.unique(model_val[keep, phi_index])
    return keep, thetas, phis


def select_iprt_iquv(
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
    i_index: int = 6,
    va_index: int = 4,
    phi_index: int = 5,
    z_index: int = 1,
    depol_index: int = 0,
    stdev: bool = False,
) -> tuple[np.ndarray, ...]:
    """Select the I, Q, U and V results from an IPRT phase A matrix.

    The records of the matrix are scattered over the (theta, phi)
    grid, one Stokes parameter per output array.

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
    stdev : bool
        Also return the standard deviations of I, Q, U and V. The sign
        changes do not apply to them.

    Returns
    -------
    tuple of ndarray
        The I, Q, U and V values, each of shape (ntheta, nphi),
        followed by their standard deviations in the same order when
        stdev is True.
    """
    keep, thetas, phis = _iprt_records(
        model_val, z_alti, depol, thetas, phis,
        va_index, phi_index, z_index, depol_index,
    )
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

    q_sign = -1 if change_q_sign else 1
    u_sign = -1 if change_u_sign else 1
    v_sign = -1 if change_v_sign else 1

    for i in np.flatnonzero(keep):
        if (
            True in (thetas == model_val[i, va_index])
            and True in (phis == model_val[i, phi_index])
        ):
            ith = int(np.squeeze(np.argwhere(
                thetas == model_val[i, va_index]
            )))
            iphi = int(np.squeeze(np.argwhere(
                phis == model_val[i, phi_index]
            )))
            indi = ith if inv_thetas else n_theta - 1 - ith
            indj = n_phi - 1 - iphi if inv_phis else iphi
            stokes_i[indi, indj] = model_val[i, i_index]
            stokes_q[indi, indj] = model_val[i, i_index + 1] * q_sign
            stokes_u[indi, indj] = model_val[i, i_index + 2] * u_sign
            stokes_v[indi, indj] = model_val[i, i_index + 3] * v_sign
            if stdev:
                stokes_i_std[indi, indj] = model_val[i, i_index + 4]
                stokes_q_std[indi, indj] = model_val[i, i_index + 5]
                stokes_u_std[indi, indj] = model_val[i, i_index + 6]
                stokes_v_std[indi, indj] = model_val[i, i_index + 7]

    if not stdev:
        return stokes_i, stokes_q, stokes_u, stokes_v
    else:
        return (
            stokes_i,
            stokes_q,
            stokes_u,
            stokes_v,
            stokes_i_std,
            stokes_q_std,
            stokes_u_std,
            stokes_v_std,
        )


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
    """Select phase A I, Q, U and V results and plot them in polar view.

    The values are selected with select_iprt_iquv and the four Stokes
    parameters are drawn side by side with smartg.view.plot_polar_iquv.

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
        largest absolute value of the panel is used. The Q, U and V
        panels are drawn from -max to +max. The I panel is drawn from
        0 to its largest value by default, and from -max_i to +max_i
        when max_i is given, e.g. for a difference.
    cmap_i, cmap_q, cmap_u, cmap_v : str or Colormap, optional
        Colour map of each panel, 'jet' for I and 'RdBu_r' for the
        other panels by default.
    force_iquv : list of ndarray, optional
        Plot these I, Q, U and V matrices, as given, instead of
        selecting them from model_val, which then only provides the
        angles. The sign changes do not apply to them.
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
    _, thetas, phis = _iprt_records(
        model_val, z_alti, depol, thetas, phis,
        va_index, phi_index, z_index, depol_index,
    )

    if force_iquv is not None:
        iquv = tuple(force_iquv)
        iquv_std = tuple(
            np.zeros((len(thetas), len(phis))) for _ in range(4)
        )
    else:
        selected = select_iprt_iquv(
            model_val,
            z_alti,
            depol=depol,
            thetas=thetas,
            phis=phis,
            inv_thetas=inv_thetas,
            inv_phis=inv_phis,
            change_q_sign=change_q_sign,
            change_u_sign=change_u_sign,
            change_v_sign=change_v_sign,
            i_index=i_index,
            va_index=va_index,
            phi_index=phi_index,
            z_index=z_index,
            depol_index=depol_index,
            stdev=output_iquv_std,
        )
        iquv = selected[:4]
        iquv_std = selected[4:]

    if not avoid_plot:
        plot_polar_iquv(
            iquv,
            thetas,
            phis,
            max_i=max_i,
            max_q=max_q,
            max_u=max_u,
            max_v=max_v,
            cmap_i=cmap_i,
            cmap_q=cmap_q,
            cmap_u=cmap_u,
            cmap_v=cmap_v,
            title=title,
            save_fig=save_fig,
            sym=sym,
        )

    if output_iquv and output_iquv_std:
        return iquv + iquv_std
    elif output_iquv:
        return iquv
    elif output_iquv_std:
        return iquv_std
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
    fname: str | Path,
    output_layer: list[str] | None = None,
    interp: bool = False,
) -> None:
    """Convert SMART-G outputs into the IPRT phase A ASCII format.

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
    fname : str or Path
        Path of the ASCII file to write.
    output_layer : list of str, optional
        Output layer of each output, e.g. '_down (0+)'. By default
        '_up (TOA)' is used for all of them.
    interp : bool
        Interpolate the outputs at the requested angles instead of
        reading them at the matching indices.
    """
    output = "# IPRT case " + case_name + "\n"
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
        fac = np.cos(np.radians(szas[im])) / np.pi
        vza = vzas[im]
        vaa = vaas[im]
        if output_layer is None:
            layer = '_up (TOA)'
        else:
            layer = output_layer[im]
        for iza, za in enumerate(vza):
            for iaa, aa in enumerate(vaa):
                if not interp:
                    stokes_i = float(m['I' + layer][iaa, iza]) * fac
                    stokes_q = float(m['Q' + layer][iaa, iza]) * fac
                    stokes_u = (
                        float(m['U' + layer][iaa, iza])
                        * fac * u_signs[im]
                    )
                    stokes_v = float(m['V' + layer][iaa, iza]) * fac

                    stokes_i_std = (
                        float(m['I_stdev' + layer][iaa, iza]) * fac
                    )
                    stokes_q_std = (
                        float(m['Q_stdev' + layer][iaa, iza]) * fac
                    )
                    stokes_u_std = (
                        float(m['U_stdev' + layer][iaa, iza]) * fac
                    )
                    stokes_v_std = (
                        float(m['V_stdev' + layer][iaa, iza]) * fac
                    )
                else:
                    pos = {'Azimuth angles': aa, 'Zenith angles': za}
                    stokes_i = float(m['I' + layer].interp(pos)) * fac
                    stokes_q = float(m['Q' + layer].interp(pos)) * fac
                    stokes_u = (
                        float(m['U' + layer].interp(pos))
                        * fac * u_signs[im]
                    )
                    stokes_v = float(m['V' + layer].interp(pos)) * fac

                    stokes_i_std = (
                        float(m['I_stdev' + layer].interp(pos)) * fac
                    )
                    stokes_q_std = (
                        float(m['Q_stdev' + layer].interp(pos)) * fac
                    )
                    stokes_u_std = (
                        float(m['U_stdev' + layer].interp(pos)) * fac
                    )
                    stokes_v_std = (
                        float(m['V_stdev' + layer].interp(pos)) * fac
                    )

                output += (
                    f"{depols[im]:.2f} {altitudes[im]:.1f} "
                    f"{szas[im]:.1f} {saas[im]:.1f} "
                    f"{za:.1f} {aa:.1f} "
                    f"{stokes_i:.5e} {stokes_q:.5e} "
                    f"{stokes_u:.5e} {stokes_v:.5e} "
                    f"{stokes_i_std:.5e} {stokes_q_std:.5e} "
                    f"{stokes_u_std:.5e} {stokes_v_std:.5e}\n"
                )

    with open(fname, 'w') as f:
        f.write(output)


def compute_deltam_iprtout(
    obs: np.ndarray,
    mod: np.ndarray,
    i_obs_id: int = 6,
    i_mod_id: int = 6,
    print_res: bool = True,
) -> np.ndarray:
    """Compute the IPRT delta_m metric from two phase A matrices.

    delta_m is the root mean square of the difference between the
    model and the reference, relative to the root mean square of the
    reference, in percent. It is reported per Stokes parameter, and
    is set to 0 when the reference is uniformly zero.

    Parameters
    ----------
    obs : ndarray
        Observed, or reference model, values, as read from an IPRT
        phase A result file: one row per record.
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
    if not isinstance(obs, np.ndarray) or not isinstance(mod, np.ndarray):
        raise TypeError("obs and mod must be np.ndarray!")
    id_obs = [i_obs_id, i_obs_id + 1, i_obs_id + 2, i_obs_id + 3]
    id_mod = [i_mod_id, i_mod_id + 1, i_mod_id + 2, i_mod_id + 3]
    stk = ['I', 'Q', 'U', 'V']
    delta_m = np.zeros(4, dtype=np.float32)
    for i in range(len(stk)):
        with np.errstate(divide='raise', invalid='raise'):
            try:
                delta_m[i] = (
                    100
                    * np.sqrt(np.sum(
                        (obs[:, id_obs[i]] - mod[:, id_mod[i]]) ** 2
                    ))
                    / np.sqrt(np.sum(obs[:, id_obs[i]] ** 2))
                )
            except FloatingPointError:
                delta_m[i] = 0.
        if print_res:
            print(stk[i], f"{delta_m[i]:.3f}")
    return delta_m


def read_iprt_output(file_res: str | Path) -> np.ndarray:
    """Read a phase A ASCII result file.

    Parameters
    ----------
    file_res : str or Path
        The file, e.g. a MYSTIC reference or a file written by
        convert_sgout_to_iprtout. Its comment lines start with '#'.

    Returns
    -------
    ndarray
        One row per record, the columns following the IPRT convention.
    """
    return pd.read_csv(file_res, header=None, sep=r"\s+", dtype=float,
                       comment="#").values


def merge_least_noisy(
    model_val: np.ndarray,
    model_val2: np.ndarray,
    i_index: int = 6,
) -> np.ndarray:
    """Merge two runs of a case, keeping the least noisy values.

    For each record and each Stokes parameter, the value and the
    standard deviation of the run with the smaller standard deviation
    are kept, the second run winning ties.

    Parameters
    ----------
    model_val, model_val2 : ndarray
        The two runs, as read from phase A result files, with the same
        records in the same order.
    i_index : int
        Column index of I. Q, U and V must follow it in that order, and
        their standard deviations right after them.

    Returns
    -------
    ndarray
        The merged records.

    Raises
    ------
    ValueError
        If the two runs do not hold the same records.
    """
    if (model_val.shape != model_val2.shape
            or not np.array_equal(model_val[:, :i_index],
                                  model_val2[:, :i_index])):
        raise ValueError("The two runs must hold the same records!")
    merged = model_val.copy()
    for istk in range(4):
        value, std = i_index + istk, i_index + 4 + istk
        second = ~(model_val[:, std] < model_val2[:, std])
        merged[second, value] = model_val2[second, value]
        merged[second, std] = model_val2[second, std]
    return merged


class PolarView(NamedTuple):
    """One set of results of a phase A case, drawn in polar view.

    Attributes
    ----------
    altitude : float
        The altitude of the results, in km.
    depol : float
        The depolarisation factor.
    sza, saa : float
        The sun zenith and azimuth angles, in degrees, for the titles.
    inv_thetas : bool
        Store the results by increasing zenith angle, see
        select_iprt_iquv.
    inv_thetas_mod : bool, optional
        Same as inv_thetas, for the model when it differs from the
        reference.
    thetas, phis : ndarray, optional
        The viewing zenith and azimuth angles to keep, in degrees. By
        default all the angles of the records.
    """

    altitude: float
    depol: float
    sza: float
    saa: float
    inv_thetas: bool = False
    inv_thetas_mod: bool | None = None
    thetas: np.ndarray | None = None
    phis: np.ndarray | None = None


def compare_polar_iprt(
    ref_val: np.ndarray,
    mod_val: np.ndarray,
    case_name: str,
    views: Sequence[PolarView],
    change_u_sign: bool = False,
    change_v_sign: bool = False,
    change_v_sign_mod: bool | None = None,
    ref_depol: float | None = None,
    sym: bool = False,
    ref_name: str = "MYSTIC",
    mod_name: str = "SMARTG",
    plot_ref: bool = True,
    plot_mod: bool = False,
    plot_diff: bool = True,
    mod_scales_from_ref: bool = False,
    print_res: bool = True,
) -> np.ndarray:
    """Compare a model with a reference in polar views, with delta_m.

    For each view, the I, Q, U and V values of both matrices are
    selected, the reference, the model and their difference are
    plotted in polar view as asked, and the delta_m of all the views
    together is computed.

    Parameters
    ----------
    ref_val, mod_val : ndarray
        The reference and the model, as read with read_iprt_output.
    case_name : str
        The name of the case, e.g. 'A1', for the titles.
    views : sequence of PolarView
        The sets of results to compare.
    change_u_sign, change_v_sign : bool
        Multiply U or V by -1, see select_iprt_iquv.
    change_v_sign_mod : bool, optional
        Same as change_v_sign, for the model when it differs from the
        reference.
    ref_depol : float, optional
        The depolarisation factor of the reference records, when it
        differs from the one of the views: some MYSTIC files hold 0
        instead of 0.03.
    sym : bool
        Also plot the symmetrical results from 180 to 360 degrees.
    ref_name, mod_name : str
        The names of the reference and of the model, for the titles.
    plot_ref, plot_mod, plot_diff : bool
        Plot the reference, the model, and the reference minus the
        model.
    mod_scales_from_ref : bool
        Draw the Q, U and V panels of the model on the colour scales of
        the reference.
    print_res : bool
        Print each delta_m next to its Stokes parameter.

    Returns
    -------
    ndarray
        The delta_m of I, Q, U and V over all the views, in percent.
    """
    if change_v_sign_mod is None:
        change_v_sign_mod = change_v_sign
    ref_all: list[tuple[np.ndarray, ...]] = []
    mod_all: list[tuple[np.ndarray, ...]] = []
    for view in views:
        head = (f"IPRT case {case_name} - depol = {view.depol} - SZA = "
                f"{view.sza:.0f} - SAA = {view.saa:.0f} - "
                f"{view.altitude:.0f}km")
        depol = view.depol if ref_depol is None else ref_depol
        angles: dict[str, Any] = {"z_alti": view.altitude,
                                  "thetas": view.thetas, "phis": view.phis,
                                  "sym": sym}

        iquv_ref = select_and_plot_polar_iprt(
            ref_val, depol=depol, inv_thetas=view.inv_thetas,
            change_u_sign=change_u_sign, change_v_sign=change_v_sign,
            title=f"{head} - {ref_name}", output_iquv=True,
            avoid_plot=not plot_ref, **angles,
        )
        scales: dict[str, Any] = {}
        if mod_scales_from_ref:
            scales = {f"max_{stk}": np.max(np.abs(values))
                      for stk, values in zip("quv", iquv_ref[1:],
                                             strict=True)}
        inv_thetas_mod = (view.inv_thetas if view.inv_thetas_mod is None
                          else view.inv_thetas_mod)
        iquv_mod = select_and_plot_polar_iprt(
            mod_val, depol=view.depol, inv_thetas=inv_thetas_mod,
            change_u_sign=change_u_sign, change_v_sign=change_v_sign_mod,
            title=f"{head} - {mod_name}", output_iquv=True,
            avoid_plot=not plot_mod, **scales, **angles,
        )

        diff = [ref - mod for ref, mod in zip(iquv_ref, iquv_mod,
                                              strict=True)]
        maxima: dict[str, Any] = {f"max_{stk}": np.max(np.abs(values))
                  for stk, values in zip("iquv", diff, strict=True)}
        select_and_plot_polar_iprt(
            ref_val, depol=depol, force_iquv=diff, cmap_i="RdBu_r",
            title=f"{head} - dif ({ref_name}-{mod_name})",
            avoid_plot=not plot_diff, **maxima, **angles,
        )
        ref_all.append(iquv_ref)
        mod_all.append(iquv_mod)

    iquv_ref_tot = group_iquv(*([iquv[istk] for iquv in ref_all]
                                for istk in range(4)))
    iquv_mod_tot = group_iquv(*([iquv[istk] for iquv in mod_all]
                                for istk in range(4)))
    return compute_deltam(obs=iquv_ref_tot, mod=iquv_mod_tot,
                          print_res=print_res)


def compare_plane_iprt(
    ref_val: np.ndarray,
    mod_val: np.ndarray,
    case_name: str,
    altitudes: Sequence[float],
    plane: Literal["principal", "almucantar"],
    iquv_ymins: Sequence[Sequence[float]] | None = None,
    iquv_ymaxs: Sequence[Sequence[float]] | None = None,
    quantities: Sequence[str] = ("transmittance", "reflectance"),
    inv_thetas: bool = True,
    change_u_sign: bool = False,
    ref_columns: tuple[int, int, int, int] = (6, 4, 5, 1),
    ref_name: str = "MYSTIC",
    mod_name: str = "SMARTG",
    print_res: bool = True,
) -> np.ndarray:
    """Compare a model with a reference along a plane, with delta_m.

    For each altitude, the I, Q, U and V values of both matrices and
    their standard deviations are selected, cut along the principal
    plane or the almucantar, and plotted with
    smartg.view.plot_iquv_comparison, the reference in red and the
    model in blue. The delta_m of all the altitudes together is
    computed.

    Parameters
    ----------
    ref_val, mod_val : ndarray
        The reference and the model, as read with read_iprt_output,
        each with a single azimuth plane and its opposite.
    case_name : str
        The name of the case, e.g. 'A5', for the titles.
    altitudes : sequence of float
        The altitudes of the results, in km.
    plane : {'principal', 'almucantar'}
        'principal' joins the two azimuths of the principal plane
        along the signed viewing zenith angle, 'almucantar' keeps the
        first zenith angle along the viewing azimuth angle.
    iquv_ymins, iquv_ymaxs : sequence of sequence of float, optional
        The lower and upper bounds of the I, Q, U and V panels, one
        sequence per altitude. By default the bounds of matplotlib.
    quantities : sequence of str
        What each altitude holds, for the titles.
    inv_thetas, change_u_sign : bool
        See select_iprt_iquv.
    ref_columns : tuple of 4 int
        The i_index, va_index, phi_index and z_index arguments of
        select_iprt_iquv for the reference, whose columns may differ
        from the ones of the model.
    ref_name, mod_name : str
        The names of the reference and of the model, for the titles.
    print_res : bool
        Print each delta_m next to its Stokes parameter.

    Returns
    -------
    ndarray
        The delta_m of I, Q, U and V over all the altitudes, in
        percent.
    """
    if plane == "principal":
        vza = np.unique(mod_val[:, 4])
        xaxis = np.sort(np.concatenate((vza - 180, 180 - vza)))
        xlabel = "VZA [deg]"

        def cut(values: np.ndarray) -> np.ndarray:
            return np.concatenate((values[:, 1], values[::-1, 0]))
    elif plane == "almucantar":
        xaxis = np.unique(mod_val[:, 5])
        xlabel = "VAA [deg]"

        def cut(values: np.ndarray) -> np.ndarray:
            return values[0, :]
    else:
        raise ValueError(f"Unknown plane '{plane}'!")

    ref_all, mod_all = [], []
    for ialt, altitude in enumerate(altitudes):
        options: dict[str, Any] = {"change_u_sign": change_u_sign,
                                   "inv_thetas": inv_thetas, "stdev": True}
        i_index, va_index, phi_index, z_index = ref_columns
        ref = select_iprt_iquv(ref_val, altitude, i_index=i_index,
                               va_index=va_index, phi_index=phi_index,
                               z_index=z_index, **options)
        mod = select_iprt_iquv(mod_val, altitude, **options)
        ref_cut, mod_cut = (
            np.array([cut(values) for values in selected],
                     dtype=np.float32)
            for selected in (ref, mod)
        )
        plot_iquv_comparison(
            iquv_obs=ref_cut[:4], iquv_mod=mod_cut[:4],
            iquv_std_obs=ref_cut[4:], iquv_std_mod=mod_cut[4:],
            xaxis=xaxis, xlabel=xlabel,
            iquv_ymin=(None if iquv_ymins is None
                       else list(iquv_ymins[ialt])),
            iquv_ymax=(None if iquv_ymaxs is None
                       else list(iquv_ymaxs[ialt])),
            title=(f"IPRT case {case_name} - {quantities[ialt]} - "
                   f"{ref_name} red, {mod_name} blue"),
        )
        ref_all.append(ref_cut[:4])
        mod_all.append(mod_cut[:4])

    return compute_deltam(obs=np.concatenate(ref_all, axis=1),
                          mod=np.concatenate(mod_all, axis=1),
                          print_res=print_res)
