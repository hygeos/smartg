"""IPRT phase B tools.

Phase B holds the 3D cases. SMART-G runs two of them: C2, a cubic
cloud, and C3, a cumulus cloud field. Their results are ASCII
tables, three comment lines then one record per line with the columns

    case theta_0 z theta phi ix iy I Q U V Istd Qstd Ustd Vstd

(indices 0 to 14), one block of records per case and, within a block,
one record per sensor, iy varying fastest. theta_0 is the solar zenith
angle, theta and phi the viewing zenith and azimuth angles, in
degrees, and ix, iy the 1-based sensor indices.

Both cases share the 9 viewing geometries of CASES. The cases 1 to 4
look up at the cloud from the bottom of the domain, the cases 5 to 9
look down at it from the top. The reference files hold the 9 cases
twice: the second time, at the case number plus ATM_CASE_OFFSET, with
the molecular atmosphere (C2) or the aerosols (C3).

The run helpers take the compiled Smartg object as argument, so that
importing this module compiles no SMART-G kernel. The delta_m metric
is in smartg.iprt.common and the camera plots wrap
smartg.view.camera_view.

Key Functions
-------------
read_iprt_iquv
    Read the I, Q, U and V maps of a case from a phase B result file.
smartg_iquv
    Extract the I, Q, U and V maps of a SMART-G camera run.
plot_camera_iquv
    Plot the I, Q, U and V maps of a case.
plot_camera_difference
    Plot the difference between two sets of I, Q, U and V maps.
compare_case
    Compare a SMART-G run with a reference file: plots and delta_m.
build_atm_c2
    Build the atmosphere of the C2 cubic cloud case.
sensor_grid_c2
    Build the 70 x 70 sensor grid of the C2 case.
build_cloud_c3
    Read the cumulus cloud field of the C3 case.
build_atm_c3
    Build the atmosphere of the C3 cumulus cloud case.
sensor_grid_c3
    Build the sensor grid of the central part of the C3 field.
run_case_backward
    Run one case in backward mode.
run_group_forward
    Run a group of cases in forward mode, in a single run.
find_optimal_xb_xg
    Find the CUDA block and grid sizes giving the shortest run.
"""

import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from smartg.albedo import AlbedoCst
from smartg.atmosphere import (
    AerOPAC,
    Atm1D,
    Atm3D,
    Cloud3D,
    read_i3rc_cloud,
)
from smartg.config import DIR_AUXDATA
from smartg.diff import diff1
from smartg.grid3d import Grid3D
from smartg.iprt.common import compute_deltam
from smartg.phase import read_phase_cdf
from smartg.sensor import Sensor, get_sensors_grid
from smartg.smartg import LocalEstimate, Smartg
from smartg.surface import LambSurface
from smartg.truncation import DMTrunc, GTTrunc
from smartg.typing import ThetaLike
from smartg.view import camera_view

logger = logging.getLogger(__name__)

DIR_PHASE_B = DIR_AUXDATA / "IPRT" / "phaseB"
MYSTIC_RES_C2 = DIR_PHASE_B / "mystic_res" / "iprt_case_C2_mystic.dat"
MYSTIC_RES_C3 = DIR_PHASE_B / "mystic_res" / "iprt_case_C3_mystic.dat"

# Number of the first case with the molecular atmosphere (C2) or the
# aerosols (C3) in the reference files, minus 1
ATM_CASE_OFFSET = 9

WAVELENGTH_C2 = 800.0
WAVELENGTH_C3 = 670.0
# Single scattering albedo of the 1D aerosol of C3 at WAVELENGTH_C3
SSA_AER_C3 = 0.931184

# Offset below the top of the domain, in km, that keeps the sensors
# looking down inside it
TOP_OFFSET = 1e-6

STOKES = ("I", "Q", "U", "V")


class CaseGeometry(NamedTuple):
    """Viewing and sun geometry of a phase B case.

    Attributes
    ----------
    pos_z_key : str
        Where the sensors are: 'bottom' or 'top' of the domain.
    theta : float
        Viewing zenith angle, in degrees.
    phi : float
        Viewing azimuth angle, in degrees.
    theta_0 : float
        Solar zenith angle, in degrees. The solar azimuth angle is
        PHI_0 for every case.
    """

    pos_z_key: str
    theta: float
    phi: float
    theta_0: float


class ForwardGroup(NamedTuple):
    """Cases computed together by a single forward run.

    In forward mode the sensors emit the photons towards the sun, and
    the viewing directions of the cases are zipped in the local
    estimate. The cases of a group therefore share their sun position
    and their sensor altitude.

    Attributes
    ----------
    cases : tuple of int
        The case numbers, in the order of the local estimate
        directions.
    reverse_theta : bool
        Give the local estimate 180 - theta instead of theta.
    count_level : int
        The SMART-G count level of the local estimate.
    output_layers : int
        The output_layers argument of Smartg.run.
    level : str
        The level of the results in the output Dataset, e.g. 'up (TOA)'
        for the variable 'I_up (TOA)'.
    """

    cases: tuple[int, ...]
    reverse_theta: bool
    count_level: int
    output_layers: int
    level: str


class PhaseBAtmosphere(NamedTuple):
    """What a phase B run needs from its atmosphere.

    Attributes
    ----------
    profile : xr.Dataset
        The 3D atmosphere profile, as Atm3D.calc returns it.
    grid3 : Grid3D
        The grid of the atmosphere cells.
    surface : LambSurface
        The Lambertian surface.
    wavelengths : ndarray
        The wavelengths of the run, in nm.
    """

    profile: xr.Dataset
    grid3: Grid3D
    surface: LambSurface
    wavelengths: np.ndarray


# The 9 viewing geometries of the C2 and C3 cases
CASES = {
    1: CaseGeometry("bottom", 40.0, 0.0, 20.0),
    2: CaseGeometry("bottom", 40.0, 60.0, 20.0),
    3: CaseGeometry("bottom", 40.0, 120.0, 20.0),
    4: CaseGeometry("bottom", 40.0, 180.0, 20.0),
    5: CaseGeometry("top", 180.0, 0.0, 40.0),
    6: CaseGeometry("top", 140.0, 0.0, 40.0),
    7: CaseGeometry("top", 140.0, 60.0, 40.0),
    8: CaseGeometry("top", 140.0, 120.0, 40.0),
    9: CaseGeometry("top", 140.0, 180.0, 40.0),
}
PHI_0 = 180.0

# The cases grouped by sun position, one forward run per group. The
# first group looks at the downward radiance below the cloud, the
# second one at the upward radiance at the top of the domain, and only
# the second one reverses the zenith angles of the local estimate.
FORWARD_GROUPS = {
    1: ForwardGroup((1, 2, 3, 4), False, 1, 3, "down (0+)"),
    2: ForwardGroup((5, 6, 7, 8, 9), True, 0, 1, "up (TOA)"),
}


def read_iprt_iquv(
    file_res: str | Path,
    case: int,
    n_x: int,
    n_y: int | None = None,
    stdev: bool = False,
    n_header: int = 3,
) -> tuple[np.ndarray, ...]:
    """Read the I, Q, U and V maps of a case from a phase B result file.

    The records of the case must form one block of n_x * n_y lines,
    the blocks following each other in the order of the case numbers.

    Parameters
    ----------
    file_res : str or Path
        The result file, in the IPRT phase B ASCII format.
    case : int
        The number of the case, from 1, i.e. its block in the file.
    n_x : int
        Number of sensors along the x axis.
    n_y : int, optional
        Number of sensors along the y axis. By default n_x.
    stdev : bool
        Also return the standard deviation maps, of the columns Istd,
        Qstd, Ustd and Vstd.
    n_header : int
        Number of comment lines at the top of the file.

    Returns
    -------
    tuple of ndarray
        The I, Q, U and V maps, of shape (n_y, n_x) and indexed as
        [iy, ix], followed by the four standard deviation maps with
        stdev.
    """
    if n_y is None:
        n_y = n_x
    n_sensors = n_x * n_y
    records = pd.read_csv(
        file_res,
        skiprows=n_sensors * (case - 1) + n_header,
        nrows=n_sensors,
        header=None,
        sep=r"\s+",
        dtype=float,
    ).values
    n_maps = 8 if stdev else 4
    # iy varies fastest in the file
    return tuple(
        records[:, 7 + icol].reshape(n_x, n_y).T for icol in range(n_maps)
    )


def smartg_iquv(
    ds: xr.Dataset,
    norm: float,
    n_x: int,
    n_y: int | None = None,
    level: str = "up (TOA)",
    direction: int = 0,
    u_sign: float = 1.0,
    v_sign: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the I, Q, U and V maps of a SMART-G camera run.

    The sensors of the run must come from get_sensors_grid, x varying
    fastest, and the Stokes variables must have the sensor index as
    their first dimension.

    Parameters
    ----------
    ds : xr.Dataset
        The output of Smartg.run.
    norm : float
        The factor applied to the four maps, cos(theta_0) / pi to get
        the IPRT radiances.
    n_x : int
        Number of sensors along the x axis.
    n_y : int, optional
        Number of sensors along the y axis. By default n_x.
    level : str
        The level of the results, e.g. 'up (TOA)' for the variable
        'I_up (TOA)'.
    direction : int
        Index of the local estimate direction: 0 in backward mode, the
        index of the case in its group in forward mode.
    u_sign, v_sign : float
        Factors applied to U and V to follow the IPRT convention: 1 and
        -1 in backward mode, -1 and 1 in forward mode.

    Returns
    -------
    tuple of 4 ndarray
        The I, Q, U and V maps, of shape (n_y, n_x) and indexed as
        [iy, ix].
    """
    if n_y is None:
        n_y = n_x
    n_sensors = n_x * n_y
    signs = (1.0, 1.0, u_sign, v_sign)
    i, q, u, v = (
        ds[f"{stk}_{level}"].values.reshape(n_sensors, -1)[:, direction]
        .reshape(n_y, n_x) * norm * sign
        for stk, sign in zip(STOKES, signs, strict=True)
    )
    return i, q, u, v


def _camera_options(camera_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the camera_view options of the phase B plots."""
    options: dict[str, Any] = {
        "interpolation": "none",
        "figsize": (10.5, 7),
        "fontsize": 16,
        "scale": False,
        "stokes": list(STOKES),
        "cbar_shrink": 1,
        "cbar_sci_format": True,
    }
    options.update(camera_kwargs)
    return options


def plot_camera_iquv(
    iquv: tuple[np.ndarray, ...] | list[np.ndarray],
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    title: str | None = None,
    i_vmin: float | None = None,
    **camera_kwargs: Any,
) -> Figure:
    """Plot the I, Q, U and V maps of a case.

    I goes from i_vmin to its maximum with the jet colormap, Q, U and V
    are centred on 0 with the coolwarm colormap.

    Parameters
    ----------
    iquv : tuple or list of 4 ndarray
        The I, Q, U and V maps, of shape (n_y, n_x).
    xgrid, ygrid : ndarray
        The sensor cell edges along the x and y axes, in km.
    title : str, optional
        The figure title.
    i_vmin : float, optional
        The lower bound of the I colour scale. By default the minimum
        of abs(I).
    **camera_kwargs
        Other arguments of smartg.view.camera_view, overriding the
        defaults of this function.

    Returns
    -------
    Figure
        The created figure.
    """
    i, q, u, v = iquv
    max_q, max_u, max_v = (np.max(np.abs(stk)) for stk in (q, u, v))
    if i_vmin is None:
        i_vmin = np.min(np.abs(i))
    options = _camera_options({
        "cmap": ["jet", "coolwarm", "coolwarm", "coolwarm"],
        "vmin": [i_vmin, -max_q, -max_u, -max_v],
        "vmax": [np.max(i), max_q, max_u, max_v],
        **camera_kwargs,
    })
    return camera_view(None, xgrid, ygrid, matrices=[i, q, u, v],
                       title=title, **options)


def plot_camera_difference(
    iquv_mod: tuple[np.ndarray, ...] | list[np.ndarray],
    iquv_ref: tuple[np.ndarray, ...] | list[np.ndarray],
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    title: str | None = None,
    diff_frac: float = 0.05,
    v_diff_frac: float | None = None,
    **camera_kwargs: Any,
) -> Figure:
    """Plot the difference between two sets of I, Q, U and V maps.

    The colour scales are centred on 0 and bounded by a fraction of the
    model maxima: the maximum of I, and the maxima of abs(Q), abs(U)
    and abs(V).

    Parameters
    ----------
    iquv_mod : tuple or list of 4 ndarray
        The model I, Q, U and V maps, of shape (n_y, n_x).
    iquv_ref : tuple or list of 4 ndarray
        The reference maps, subtracted from the model ones.
    xgrid, ygrid : ndarray
        The sensor cell edges along the x and y axes, in km.
    title : str, optional
        The figure title.
    diff_frac : float
        Bound of the I, Q and U colour scales, as a fraction of the
        model maximum.
    v_diff_frac : float, optional
        Same as diff_frac, for V. By default diff_frac.
    **camera_kwargs
        Other arguments of smartg.view.camera_view, overriding the
        defaults of this function.

    Returns
    -------
    Figure
        The created figure.
    """
    if v_diff_frac is None:
        v_diff_frac = diff_frac
    i, q, u, v = iquv_mod
    fracs = (diff_frac, diff_frac, diff_frac, v_diff_frac)
    maxima = (np.max(i), *(np.max(np.abs(stk)) for stk in (q, u, v)))
    lim = [maximum * frac
           for maximum, frac in zip(maxima, fracs, strict=True)]
    options = _camera_options({
        "cmap": ["coolwarm", "coolwarm", "coolwarm", "coolwarm"],
        "vmin": [-val for val in lim],
        "vmax": lim,
        **camera_kwargs,
    })
    matrices = [mod - ref
                for mod, ref in zip(iquv_mod, iquv_ref, strict=True)]
    return camera_view(None, xgrid, ygrid, matrices=matrices, title=title,
                       **options)


def central_slice(n_cells: int, n_sensors: int, edges: bool = False
                  ) -> slice:
    """Return the slice of the central sensors of an axis.

    Parameters
    ----------
    n_cells : int
        Number of cells along the axis.
    n_sensors : int
        Number of central cells to keep.
    edges : bool
        Slice the n_sensors + 1 cell edges instead of the cells.

    Returns
    -------
    slice
        The slice of the central cells, or of their edges.
    """
    start = (n_cells - n_sensors) // 2
    return slice(start, start + n_sensors + int(edges))


def compare_case(
    ds: xr.Dataset,
    norm: float,
    case: int,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    ref_file: str | Path,
    ref_case: int | None = None,
    ref_n_x: int | None = None,
    ref_n_y: int | None = None,
    level: str = "up (TOA)",
    direction: int = 0,
    u_sign: float = 1.0,
    v_sign: float = -1.0,
    case_name: str = "C2",
    title_suffix: str | None = None,
    mod_name: str = "SMART-G",
    ref_name: str = "MYSTIC",
    i_vmin: float | None = None,
    diff_frac: float = 0.05,
    v_diff_frac: float | None = None,
    plot: bool = True,
    print_res: bool = True,
) -> np.ndarray:
    """Compare a SMART-G run with a reference file: plots and delta_m.

    The SMART-G maps are extracted with smartg_iquv and the reference
    maps read with read_iprt_iquv. The function then plots the SMART-G
    maps and their difference with the reference, and computes the
    delta_m of I, Q, U and V.

    Parameters
    ----------
    ds : xr.Dataset
        The output of Smartg.run.
    norm : float
        The factor applied to the SMART-G maps, see smartg_iquv.
    case : int
        The case number, for the titles and, by default, to find the
        reference block.
    xgrid, ygrid : ndarray
        The sensor cell edges along the x and y axes, in km.
    ref_file : str or Path
        The reference result file, in the IPRT phase B ASCII format.
    ref_case : int, optional
        The block of the case in ref_file. By default case; add
        ATM_CASE_OFFSET for the cases with atmosphere.
    ref_n_x, ref_n_y : int, optional
        Number of sensors of ref_file along the x and y axes, when
        SMART-G only ran the central part of the reference sensors. By
        default the SMART-G sensor numbers.
    level : str
        The level of the SMART-G results, see smartg_iquv.
    direction : int
        Index of the local estimate direction, see smartg_iquv.
    u_sign, v_sign : float
        Factors applied to the SMART-G U and V, see smartg_iquv.
    case_name : str
        The name of the IPRT case, e.g. 'C2', at the start of the
        titles.
    title_suffix : str, optional
        Text at the end of the titles, e.g. 'without atm'.
    mod_name, ref_name : str
        Names of the model and of the reference, for the titles and
        the printed delta_m.
    i_vmin : float, optional
        Lower bound of the I colour scale, see plot_camera_iquv.
    diff_frac, v_diff_frac : float, optional
        Bounds of the difference colour scales, see
        plot_camera_difference.
    plot : bool
        Plot the SMART-G maps and the difference maps.
    print_res : bool
        Print the delta_m values.

    Returns
    -------
    ndarray
        The delta_m of I, Q, U and V, in percent.
    """
    n_x = xgrid.size - 1
    n_y = ygrid.size - 1
    iquv_mod = smartg_iquv(ds, norm, n_x, n_y, level=level,
                           direction=direction, u_sign=u_sign,
                           v_sign=v_sign)

    ref_n_x = n_x if ref_n_x is None else ref_n_x
    ref_n_y = n_y if ref_n_y is None else ref_n_y
    iquv_ref = read_iprt_iquv(ref_file,
                              case if ref_case is None else ref_case,
                              ref_n_x, ref_n_y)
    crop_x = central_slice(ref_n_x, n_x)
    crop_y = central_slice(ref_n_y, n_y)
    iquv_ref = tuple(stk[crop_y, crop_x] for stk in iquv_ref)

    if plot:
        suffix = "" if title_suffix is None else f" - {title_suffix}"
        head = f"{case_name} - case {case}"
        plot_camera_iquv(iquv_mod, xgrid, ygrid,
                         title=f"{head} - {mod_name}{suffix}",
                         i_vmin=i_vmin)
        plot_camera_difference(
            iquv_mod, iquv_ref, xgrid, ygrid,
            title=f"{head} - dif({mod_name} - {ref_name}){suffix}",
            diff_frac=diff_frac, v_diff_frac=v_diff_frac,
        )

    if print_res:
        print(f"{mod_name} (delta_m):")
    return compute_deltam(obs=list(iquv_ref), mod=list(iquv_mod),
                          print_res=print_res)


def build_atm_c2(
    tau_ray: float | None = None,
    truncation: DMTrunc | GTTrunc | None = None,
    n_theta: ThetaLike = 18001,
    scale: float = 1.0,
) -> PhaseBAtmosphere:
    """Build the atmosphere of the C2 cubic cloud case.

    The domain is 7 x 7 x 5 km, periodic along x and y. A cubic water
    cloud of 1 km, with an extinction coefficient of 10 per km, an
    effective radius of 10 microns and a single scattering albedo
    forced to 1, fills the cell between 3 and 4 km along x and y and
    between 2 and 3 km along z, above a Lambertian surface of albedo
    0.2, at 800 nm. The grid is reduced to the cells needed to hold
    the cloud, which is faster than the 70 x 70 x 5 grid of the IPRT
    description and gives the same atmosphere.

    Parameters
    ----------
    tau_ray : float, optional
        Total optical depth of a homogeneous Rayleigh layer filling the
        domain, without absorption. By default there is no molecular
        atmosphere at all.
    truncation : DMTrunc or GTTrunc, optional
        Truncation of the cloud phase matrix, applied by Atm3D.calc.
    n_theta : int, str or array_like
        Scattering angles of the phase matrix: a number of equally
        spaced ones, the angles themselves, or 'native' for the grid
        the file carries.
    scale : float
        Factor applied to the grid, e.g. to test cells of a very small
        size. The extinction coefficient is divided by it.

    Returns
    -------
    PhaseBAtmosphere
        The profile, the atmosphere grid, the surface and the
        wavelengths.
    """
    cld_phase = read_phase_cdf(
        DIR_PHASE_B / "opt_prop" / "watercloud_800.mie.cdf",
        n_theta=n_theta, normalize=False, output_sg_ready=False,
    )

    # Reduced grid, the cloud is the (2, 2, 2) cell
    xgrid = np.array([0.0, 3.0, 4.0, 7.0]) * scale
    ygrid = np.array([0.0, 3.0, 4.0, 7.0]) * scale
    zgrid = np.array([0.0, 2.0, 3.0, 5.0]) * scale
    grid3 = Grid3D(xgrid, ygrid, zgrid, periodic=True)

    # IPRT cell indices, from 1, of x, y and z
    cloud_indices = np.array([[2, 2, 2]], dtype=np.int32)
    cld_ext_coeff = np.array([10.0 * (1 / scale)], dtype=np.float64)
    reff = np.array([10.0], dtype=np.float64)
    cloud3 = Cloud3D(
        "wc",
        w_ref=WAVELENGTH_C2,
        ext_ref=cld_ext_coeff,
        cell_indices=cloud_indices,
        reff=reff,
        phase=cld_phase,
        ssa_cst=1.0,
    )

    sca_ray = abs_ray = None
    if tau_ray is None:
        atm_1d = Atm1D("afglt", tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0)
    else:
        atm_1d = Atm1D("afglt")
        dz = diff1(grid3.zGRID)
        tau_ray_cs = np.cumsum((dz / grid3.zGRID[-1]) * tau_ray).reshape(
            1, len(dz)
        )
        sca_ray = abs(diff1(tau_ray_cs, axis=1) / dz)
        sca_ray[np.isnan(sca_ray)] = 0
        abs_ray = np.zeros_like(sca_ray)

    wavelengths = np.array([WAVELENGTH_C2])
    atm3 = Atm3D(
        atm_1d=atm_1d,
        grid_3d=grid3,
        comp_3d=[cloud3],
        wavelength_phase=[WAVELENGTH_C2],
        mol_sca_1d=sca_ray,
        mol_abs_1d=abs_ray,
    )
    profile = atm3.calc(wavelengths, n_theta=n_theta,
                        truncation=truncation)

    return PhaseBAtmosphere(profile, grid3,
                            LambSurface(alb=AlbedoCst(0.2)), wavelengths)


def sensor_grid_c2(scale: float = 1.0) -> Grid3D:
    """Build the 70 x 70 sensor grid of the C2 case.

    Parameters
    ----------
    scale : float
        Factor applied to the grid, as in build_atm_c2.

    Returns
    -------
    Grid3D
        One sensor per 100 m cell over the 7 x 7 km domain. The z axis
        only sets the bottom and the top altitudes of the sensors.
    """
    return Grid3D(
        np.linspace(0.0, 7.0, 71) * scale,
        np.linspace(0.0, 7.0, 71) * scale,
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) * scale,
        periodic=True,
    )


def _coef_from_layer_od(od_layers: np.ndarray, zgrid_desc: np.ndarray
                        ) -> np.ndarray:
    """Convert the layer optical depths of the C3 file to coefficients.

    The IPRT file gives one optical depth per layer, from the top of
    the atmosphere downwards, while SMART-G expects coefficients on
    its altitude grid, hence the cumulated sum and the derivative.

    Parameters
    ----------
    od_layers : ndarray
        The optical depth of each layer, from the top.
    zgrid_desc : ndarray
        The layer edges, in km, in decreasing order.

    Returns
    -------
    ndarray
        The coefficients, in 1/km, of shape (1, zgrid_desc.size).
    """
    dz = diff1(zgrid_desc)
    od_cumulated = np.concatenate(([0.0], np.cumsum(od_layers))).reshape(
        1, len(dz)
    )
    coef = abs(diff1(od_cumulated, axis=1) / dz)
    coef[np.isnan(coef)] = 0
    return coef


def build_cloud_c3(
    n_theta: ThetaLike = 1801,
    scale: float = 1.0,
) -> tuple[Cloud3D, Grid3D]:
    """Read the cumulus cloud field of the C3 case.

    The field has 100 x 100 x 53 cells. Reading its phase matrices is
    the most expensive part of building a C3 atmosphere, so several
    atmospheres can share the result.

    Parameters
    ----------
    n_theta : int
        Number of scattering angles of the phase matrices.
    scale : float
        Factor applied to the grid.

    Returns
    -------
    cloud3 : Cloud3D
        The cloud component.
    grid3 : Grid3D
        The grid of the field, periodic along x and y.
    """
    cld_phase = read_phase_cdf(
        DIR_PHASE_B / "opt_prop" / "watercloud_670.mie.cdf",
        n_theta=n_theta, normalize=False, output_sg_ready=False,
    )
    cloud3 = Cloud3D(
        "wc",
        w_ref=WAVELENGTH_C3,
        ds=read_i3rc_cloud(DIR_PHASE_B / "grids" / "cumulus.dat",
                           loc_xgrid=0, loc_ygrid=0),
        phase=cld_phase,
        reff_acc=1,
        reff_min=5,
    )

    xgrid, ygrid, zgrid = cloud3.get_xyz_grid()
    grid3 = Grid3D(xgrid * scale, ygrid * scale, zgrid * scale,
                   periodic=True)

    return cloud3, grid3


def build_atm_c3(
    cloud_c3: tuple[Cloud3D, Grid3D],
    truncation: DMTrunc | GTTrunc | None = None,
    with_aer: bool = True,
    n_theta: ThetaLike = 1801,
) -> PhaseBAtmosphere:
    """Build the atmosphere of the C3 cumulus cloud case.

    The 3D cumulus field is mixed with the 1D profile of the IPRT
    description: Rayleigh scattering, molecular absorption and aerosol
    extinction at 670 nm, above a Lambertian surface of albedo 0.2.

    Parameters
    ----------
    cloud_c3 : tuple of Cloud3D and Grid3D
        The cloud field and its grid, from build_cloud_c3.
    truncation : DMTrunc or GTTrunc, optional
        Truncation of the phase matrices, applied by Atm3D.calc.
    with_aer : bool
        Add the 1D aerosol, with its single scattering albedo
        SSA_AER_C3.
    n_theta : int
        Number of scattering angles of the phase matrices. Each cloudy
        cell gets its own mixed matrix, about 18 GB of host memory
        with the 18001 angles of the IPRT description.

    Returns
    -------
    PhaseBAtmosphere
        The profile, the atmosphere grid, the surface and the
        wavelengths.
    """
    cloud3, grid3 = cloud_c3

    # Columns of the IPRT file: bottom, top, temperature, then the
    # molecular absorption at 0.67, 2.13 and 11.0 um, the Rayleigh
    # scattering at 0.67 um and the aerosol extinction at 0.67 and
    # 2.13 um. Only the 0.67 um ones are used.
    od = pd.read_csv(
        DIR_PHASE_B / "opt_prop" / "atmos_tau_cu.dat",
        comment="!",
        header=None,
        usecols=[3, 6, 7],
        sep=r"\s+",
        dtype=float,
    ).values
    zgrid_desc = grid3.zGRID[::-1]
    mol_abs = _coef_from_layer_od(od[:, 0], zgrid_desc)
    mol_sca = _coef_from_layer_od(od[:, 1], zgrid_desc)

    comp = []
    ext_aer = ssa_aer = None
    if with_aer:
        ext_aer = _coef_from_layer_od(od[:, 2], zgrid_desc)
        phase_waso = read_phase_cdf(
            DIR_PHASE_B / "opt_prop" / "waso_670.mie.cdf",
            n_theta=n_theta, normalize=False, output_sg_ready=False,
        )
        comp = [
            AerOPAC(
                "continental_clean",
                0.5,
                w_ref=550.0,
                phase=phase_waso.isel(wavelength_phase=0, reff=0),
            )
        ]
        ssa_aer = np.full_like(ext_aer, SSA_AER_C3)

    wavelengths = np.array([WAVELENGTH_C3])
    atm3 = Atm3D(
        atm_1d=Atm1D("afglt", comp=comp),
        grid_3d=grid3,
        comp_3d=[cloud3],
        wavelength_phase=[WAVELENGTH_C3],
        mol_sca_1d=mol_sca,
        mol_abs_1d=mol_abs,
        aer_ext_1d=ext_aer,
        aer_ssa_1d=ssa_aer,
    )
    profile = atm3.calc(wavelengths, n_theta=n_theta,
                        truncation=truncation)

    return PhaseBAtmosphere(profile, grid3,
                            LambSurface(alb=AlbedoCst(0.2)), wavelengths)


def sensor_grid_c3(grid3: Grid3D, n_sensors: int = 50) -> Grid3D:
    """Build the sensor grid of the central part of the C3 field.

    Parameters
    ----------
    grid3 : Grid3D
        The grid of the cloud field, from build_cloud_c3.
    n_sensors : int
        Number of central cells kept along x and y, one sensor per
        cell. The IPRT description uses the 100 cells.

    Returns
    -------
    Grid3D
        The sensor grid. Its z axis is the one of the field and only
        sets the bottom and the top altitudes of the sensors.
    """
    edges_x = central_slice(grid3.xgrid.size - 1, n_sensors, edges=True)
    edges_y = central_slice(grid3.ygrid.size - 1, n_sensors, edges=True)
    return Grid3D(grid3.xgrid[edges_x], grid3.ygrid[edges_y],
                  grid3.zgrid, periodic=True)


def resolve_pos_z(sensor_grid: Grid3D, key: str) -> float:
    """Return the altitude of the sensors.

    Parameters
    ----------
    sensor_grid : Grid3D
        The sensor grid.
    key : str
        'bottom' for the bottom of the domain, 'top' for TOP_OFFSET
        below its top.

    Returns
    -------
    float
        The altitude, in km.
    """
    if key == "bottom":
        return sensor_grid.zGRID[0]
    if key == "top":
        return sensor_grid.zGRID[-1] - TOP_OFFSET
    raise ValueError(f"Unknown pos_z key '{key}'!")


def case_norm(theta_0: float) -> float:
    """Return the normalisation of the SMART-G maps, cos(theta_0) / pi.

    Parameters
    ----------
    theta_0 : float
        The solar zenith angle, in degrees.

    Returns
    -------
    float
        The factor giving the IPRT radiances.
    """
    # A numpy float64 and not a Python float: a float32 map times a
    # Python float stays float32
    return np.cos(np.radians(theta_0)) / np.pi


def _camera_sensors(
    atm: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    pos_z: float,
    th_deg: float,
    ph_deg: float,
) -> list[Sensor]:
    """Return one sensor per cell of sensor_grid, all aimed alike."""
    return get_sensors_grid(
        sensor_grid.xgrid,
        sensor_grid.ygrid,
        pos_z=pos_z,
        th_deg=th_deg,
        ph_deg=ph_deg,
        fov=0.0,
        loc="ATMOS",
        cell_size=sensor_grid.xgrid[1] - sensor_grid.xgrid[0],
        # The sensor grid is not the atmosphere grid
        grid_3d=atm.grid3,
    )


def backward_run_kwargs(
    atm: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    case: int,
) -> dict[str, Any]:
    """Return the Smartg.run arguments of a case in backward mode.

    Parameters
    ----------
    atm : PhaseBAtmosphere
        The atmosphere, from build_atm_c2 or build_atm_c3.
    sensor_grid : Grid3D
        The sensor grid.
    case : int
        The case number, a key of CASES.

    Returns
    -------
    dict
        The wavelength, atmosphere, surface, sensor, le and stdev
        arguments.
    """
    geometry = CASES[case]
    sensors = _camera_sensors(
        atm, sensor_grid, resolve_pos_z(sensor_grid, geometry.pos_z_key),
        geometry.theta, geometry.phi,
    )
    # count_level 0: only the radiances at the top of the domain
    le = LocalEstimate(
        th_deg=np.array([geometry.theta_0]),
        phi_deg=np.array([PHI_0]),
        count_level=np.array([0]),
    )
    return {
        "wavelength": atm.wavelengths,
        "atmosphere": atm.profile,
        "sensor": sensors,
        "le": le,
        "surface": atm.surface,
        "stdev": True,
    }


def forward_run_kwargs(
    atm: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    group: ForwardGroup,
) -> dict[str, Any]:
    """Return the Smartg.run arguments of a group in forward mode.

    Parameters
    ----------
    atm : PhaseBAtmosphere
        The atmosphere, from build_atm_c2 or build_atm_c3.
    sensor_grid : Grid3D
        The sensor grid.
    group : ForwardGroup
        The group of cases, a value of FORWARD_GROUPS.

    Returns
    -------
    dict
        The th_deg, wavelength, atmosphere, surface, sensor, le and
        output_layers arguments.
    """
    geometries = [CASES[case] for case in group.cases]
    # All the cases of a group share the same sun position
    theta_0 = geometries[0].theta_0
    # The sensors are the source: they point at the sun
    sensors = _camera_sensors(
        atm, sensor_grid, resolve_pos_z(sensor_grid, "top"),
        180.0 - theta_0, 180.0 - PHI_0,
    )
    theta = np.array([geometry.theta for geometry in geometries])
    phi = np.array([geometry.phi for geometry in geometries])
    le = LocalEstimate(
        th_deg=180.0 - theta if group.reverse_theta else theta,
        phi_deg=phi + 180.0,
        count_level=np.full(len(group.cases), group.count_level),
        zip=True,
    )
    return {
        "th_deg": theta_0,
        "wavelength": atm.wavelengths,
        "atmosphere": atm.profile,
        "sensor": sensors,
        "le": le,
        "surface": atm.surface,
        "output_layers": group.output_layers,
    }


def run_case_backward(
    sg: Smartg,
    atm: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    case: int,
    n_photons: float,
    **run_kwargs: Any,
) -> tuple[xr.Dataset, float]:
    """Run one case in backward mode.

    Parameters
    ----------
    sg : Smartg
        SMART-G compiled with back=True and opt3d=True.
    atm : PhaseBAtmosphere
        The atmosphere, from build_atm_c2 or build_atm_c3.
    sensor_grid : Grid3D
        The sensor grid.
    case : int
        The case number, a key of CASES.
    n_photons : float
        Number of photons of the run.
    **run_kwargs
        Other arguments of Smartg.run, e.g. n_loop, n_icdf, seed,
        xblock, xgrid or depo.

    Returns
    -------
    ds : xr.Dataset
        The output of the run.
    norm : float
        The normalisation of the maps, see smartg_iquv.
    """
    ds = sg.run(**backward_run_kwargs(atm, sensor_grid, case),
                n_photons=n_photons, **run_kwargs)
    return ds, case_norm(CASES[case].theta_0)


def run_group_forward(
    sg: Smartg,
    atm: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    group: ForwardGroup,
    n_photons: float,
    **run_kwargs: Any,
) -> tuple[xr.Dataset, float]:
    """Run a group of cases in forward mode, in a single run.

    The results of the case group.cases[i] are at the local estimate
    direction i, on the level group.level.

    Parameters
    ----------
    sg : Smartg
        SMART-G compiled with back=False and opt3d=True.
    atm : PhaseBAtmosphere
        The atmosphere, from build_atm_c2 or build_atm_c3.
    sensor_grid : Grid3D
        The sensor grid.
    group : ForwardGroup
        The group of cases, a value of FORWARD_GROUPS.
    n_photons : float
        Number of photons of the run.
    **run_kwargs
        Other arguments of Smartg.run, e.g. n_loop, n_icdf, seed,
        xblock, xgrid or depo.

    Returns
    -------
    ds : xr.Dataset
        The output of the run.
    norm : float
        The normalisation of the maps, see smartg_iquv.
    """
    ds = sg.run(**forward_run_kwargs(atm, sensor_grid, group),
                n_photons=n_photons, **run_kwargs)
    return ds, case_norm(CASES[group.cases[0]].theta_0)


def find_optimal_xb_xg(
    sg: Smartg,
    xblocks: list[int],
    xgrids: list[int],
    n_photons: float,
    n_loop: float,
    **run_kwargs: Any,
) -> tuple[int, int]:
    """Find the CUDA block and grid sizes giving the shortest run.

    Every pair is timed by a short run. The optimal pair depends on the
    GPU. Note that the random number generator is seeded per thread,
    so that changing the pair changes the noise realisation even with
    a fixed seed.

    Parameters
    ----------
    sg : Smartg
        The compiled SMART-G.
    xblocks, xgrids : list of int
        The candidate xblock and xgrid values.
    n_photons, n_loop : float
        Number of photons of the timing runs, and per kernel loop.
    **run_kwargs
        The other arguments of Smartg.run, e.g. from
        backward_run_kwargs.

    Returns
    -------
    xblock, xgrid : int
        The pair with the shortest kernel time.
    """
    best_time = np.inf
    best_xblock, best_xgrid = xblocks[0], xgrids[0]
    for xgrid in xgrids:
        for xblock in xblocks:
            ds_test = sg.run(**run_kwargs, n_photons=n_photons,
                             n_loop=n_loop, xblock=xblock, xgrid=xgrid,
                             progress=False)
            time_s = float(ds_test.attrs["kernel time (s)"])
            logger.info(f"time (s) = {time_s}; xblock = {xblock}; "
                        f"xgrid = {xgrid}")
            if time_s < best_time:
                best_time = time_s
                best_xblock, best_xgrid = xblock, xgrid
    logger.info(f"Best xblock = {best_xblock}; best xgrid = {best_xgrid}")
    return best_xblock, best_xgrid
