"""Scattering phase matrix utilities for SMART-G.

This module provides phase matrix readers for multiple file formats,
and helper routines used to build the phase function input required by
SMART-G radiative transfer simulations.

Phase Matrix I/O
-----------------
read_phase
    Dispatch function that selects the appropriate reader based on file
    extension (``.dat``, ``.nc``, ``.cdf``).
read_phase_dat
    Read a monochromatic, vertically uniform phase matrix from a
    space-separated ``.dat`` file.
read_phase_nc
    Read and process phase function data from SMART-G NetCDF
    aerosol/cloud files (``.nc`` suffix).
read_phase_cdf
    Read and process phase function data from libRadtran NetCDF
    aerosol/cloud files (``.cdf`` suffix).
read_phase_nth_cte
    Read a libRadtran or monochromatic IPRT NetCDF aerosol/cloud
    file and resample its phase matrix onto a constant number of
    scattering angles.

Phase Matrix Processing
------------------------
theta_grid
    Build the scattering angle grid of a phase matrix, either
    equally spaced or clustered towards the forward and backward
    directions.
integ_phase
    Numerically integrate a phase function weighted by
    ``sin(theta)`` along the scattering angle axis.
calc_iphase
    Map phase functions onto the full wavelength/altitude grid
    and return compact index arrays.
get_ipha_a
    Map the phase-function altitude grid onto the model
    altitude grid by maximum vertical overlap.
expand_phase_4_to_6
    Complete a 4-term phase matrix (spherical particles) into its
    6-term equivalent, by duplicating F22 = F11 and F44 = F33.
convert_phase_to_iparper
    Convert a phase matrix from the IQUV Stokes convention to the
    parallel/perpendicular intensity convention used by SMART-G.
get_prof_phases
    Build the ``prof_phases`` tuple expected by ``Atm1D`` from a phase
    function ``DataArray`` and the full wavelength/altitude grids.

Key Functions
-------------
read_phase
    Read phase function data from a file and dispatch to the
    proper format reader.
calc_iphase
    Map phase functions onto the full wavelength/altitude grid.
get_prof_phases
    Generate the prof_phases parameter for Atm1D from phase
    function data.
convert_phase_to_iparper
    Convert a phase matrix to the parallel/perpendicular
    intensity convention.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from smartg.typing import PathType, NumericArrayLike
import pandas as pd
import xarray as xr
from luts.luts import LUT
from pytrunc.utils import quadrature_lobatto


THETA_GRID_KINDS = ('uniform', 'chebyshev', 'lobatto')


def theta_grid(
    n: int,
    kind: str = 'uniform',
    unit: str = 'deg',
) -> NDArray[np.float64]:
    """Build the scattering angle grid of a phase matrix.

    The grid spans the whole scattering range and includes both end
    points exactly, so it can be used directly as the ``theta_atm`` or
    ``theta_oc`` axis of a phase matrix.

    Parameters
    ----------
    n : int
        Number of scattering angles. Must be >= 2.
    kind : str, optional
        Node distribution. Choices are:

        - ``'uniform'`` -> equally spaced angles (default)
        - ``'lobatto'`` -> Gauss-Lobatto-Legendre nodes in theta
        - ``'chebyshev'`` -> Chebyshev-Lobatto nodes in theta,
          ``theta_i = 180 sin^2(pi i / (2 (n-1)))``
    unit : str, optional
        Unit of the returned angles, ``'deg'`` (default) or ``'rad'``.

    Returns
    -------
    ndarray
        Strictly increasing angles of shape ``(n,)``, from 0 to 180
        degrees, or from 0 to pi radians.

    Notes
    -----
    Both non-uniform kinds cluster their nodes towards 0 and 180
    degrees, which is what resolves the forward diffraction peak of
    large particles such as desert aerosols and cloud droplets. At
    n = 1801 they place 86 nodes below 1 degree where a uniform grid
    places 10, and they are interchangeable in practice: their nodes
    differ by at most 0.014 degrees.

    Beware that Lobatto nodes in mu = cos(theta), such as the ones the
    delta-M truncation uses to integrate Legendre moments, are of no
    use here. At n = 1801 their first node lies at 0.12 degrees, which
    is coarser than the 0.1 degrees of a uniform theta grid: only
    clustering in theta resolves the peak.

    ``'lobatto'`` is the recommended kind when the phase matrix is
    truncated, because ``pytrunc.utils.integrate_lobatto`` interpolates
    onto those very nodes before applying its weights. On a Lobatto
    grid that interpolation is the identity and the truncation
    quadrature becomes exact.

    Examples
    --------
    >>> theta_grid(5)
    array([  0.,  45.,  90., 135., 180.])
    >>> theta_grid(5, kind='chebyshev').round(2)
    array([  0.  ,  26.36,  90.  , 153.64, 180.  ])
    """
    if n < 2:
        raise ValueError(f"The n parameter must be >= 2, got {n}.")
    if kind not in THETA_GRID_KINDS:
        raise ValueError(
            f"Choices for the kind parameter are: {THETA_GRID_KINDS}, "
            f"got {kind!r}."
        )
    if unit not in ('deg', 'rad'):
        raise ValueError(
            "Choices for the unit parameter are: ('deg', 'rad'), "
            f"got {unit!r}."
        )

    span = 180.0 if unit == 'deg' else np.pi

    if kind == 'uniform':
        theta = np.linspace(0.0, span, n)
    elif kind == 'chebyshev':
        i = np.arange(n, dtype=np.float64)
        theta = span * np.sin(0.5 * np.pi * i / (n - 1)) ** 2
    elif n == 2:
        # quadrature_lobatto needs a Legendre polynomial of order
        # n - 1 >= 2; with only the two end points every kind agrees
        theta = np.array([0.0, span])
    else:
        # quadrature_lobatto caches and returns read-only arrays
        theta = quadrature_lobatto(0.0, span, n)[0].copy()

    # the end points must be exact: they bound the interpolations of
    # the callers, and the kernel maps theta = 180 degrees onto the
    # last table entry
    theta[0] = 0.0
    theta[-1] = span

    return theta


def integ_phase(
    ang: NDArray[np.floating[Any]],
    pha: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Numerically integrate a phase function weighted by sin(theta).

    Compute the integral of ``pha(ang) * sin(ang)`` along the last
    axis of *pha* using a composite rule that blends trapezoidal
    and Simpson's rules at each sub-interval.

    Parameters
    ----------
    ang : ndarray
        Scattering angles in radians, shape ``(nt,)``, strictly
        increasing.
    pha : ndarray
        Phase function values at *ang* of shape
        ``(..., nt)`` where ``...`` is any number of leading dimensions.

    Returns
    -------
    ndarray
        Integration result of shape ``(...)`` (last axis reduced).

    Notes
    -----
    Each sub-interval ``[ang[i], ang[i+1]]`` is integrated using a
    composite scheme that is exact for linear integrands on each
    piece:

    .. math::

        w_i = \\Delta\\theta_i \\left[
            \\frac{\\sin\\theta_i\\,p_i
                  + \\sin\\theta_{i+1}\\,p_{i+1}}{3}
          + \\frac{\\sin\\theta_i\\,p_{i+1}
                  + \\sin\\theta_{i+1}\\,p_i}{6}
        \\right]
    """
    assert not np.isnan(pha).any()

    dtheta = np.diff(ang)
    pm1 = pha[..., :-1]
    pm2 = pha[..., 1:]
    sin1 = np.sin(ang[:-1])
    sin2 = np.sin(ang[1:])

    return np.sum(
        dtheta
        * ((sin1 * pm1 + sin2 * pm2) / 3.0 + (sin1 * pm2 + sin2 * pm1) / 6.0),
        axis=-1,
    )


def calc_iphase(
    phase: xr.DataArray | LUT,
    wavelength_full: NumericArrayLike,
    z_full: NumericArrayLike,
    old_method: bool = False,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.int32]]:
    """Map phase functions onto the full wavelength/altitude grid.

    Reshape the phase DataArray into a compact array and compute an
    index array that maps each model grid point to the nearest
    phase-function entry.

    Parameters
    ----------
    phase : DataArray or LUT
        Phase function data as an ``xr.DataArray`` with
        coordinates ``wavelength_phase``, ``z_phase`` and dimensions
        ``(wavelength_phase, z_phase, nphamat, theta)``, or a LUT object
        exposing a ``to_xarray()`` method.
    wavelength_full : array_like
        Full model wavelength grid, shape ``(n_wavelength,)``.
    z_full : array_like
        Full model altitude grid, shape ``(nz,)``.
    old_method : bool, optional
        If ``True``, use simple nearest-neighbour matching for both
        wavelength and altitude grids.  If ``False`` (default),
        altitude mapping is delegated to :func:`get_ipha_a`
        (layer-overlap-aware with null-phase penalties).

    Returns
    -------
    pha : ndarray
        Phase function values reshaped from *phase* of shape
        ``(n_wavelength_pf * nz_pf, nstk, ntheta)``.
    ipha : ndarray
        Index array of shape ``(n_wavelength_full, nz_full)``
        mapping each model grid point to an entry in *pha*.  Indices
        are zero-based.
    """
    # Deals with the case where the legacy LUT object is used for phase
    if isinstance(phase, LUT):
        phase = phase.to_xarray()

    z_full = np.atleast_1d(z_full).astype(np.float32)
    wavelength_full = np.atleast_1d(wavelength_full).astype(np.float32)

    # Extract wavelength and altitude coordinates from DataArray
    wavelength = phase.coords["wavelength_phase"].values
    altitude = phase.coords["z_phase"].values

    n_wavelength, nz, nstk, ntheta = phase.shape
    pha = phase.values.reshape(n_wavelength * nz, nstk, ntheta)

    ipha_w = np.array(
        [np.abs(wavelength - x).argmin() for x in wavelength_full],
        dtype="int32",
    )
    if old_method:
        ipha_a = np.array(
            [np.abs(altitude - x).argmin() for x in z_full], dtype="int32"
        )
    else:
        ipha_a = get_ipha_a(z_full=z_full, z_pf=altitude, phase=phase)
    ipha = ipha_a[None, :] + ipha_w[:, None] * len(altitude)

    return (pha, ipha)


def get_ipha_a(
    z_full: NumericArrayLike,
    z_pf: NumericArrayLike,
    phase: xr.DataArray | None = None,
) -> NDArray[np.int32]:
    """Map the phase-function altitude grid onto the model altitude
    grid.  For each level in *z_full* find the phase-function layer with
    the largest vertical overlap, optionally penalising layers whose
    phase function is identically zero.

    Parameters
    ----------
    z_full : array_like
        Model altitude grid, shape ``(nz_full,)``.
    z_pf : array_like
        Phase-function altitude grid, shape ``(nz_pf,)``.
    phase : DataArray or None, optional
        If provided, the phase function values (dimension order
        ``wavelength, z, nphamat, theta``) are used to detect and
        penalise
        layers with a zero phase function.  Layers whose first
        wavelength / Stokes component sums to zero have their weight
        scaled by ``1e-6`` and trigger a warning.

    Returns
    -------
    ndarray
        Array of shape ``(nz_full,)`` mapping each level to the index
        into *z_pf* of the best-matching phase-function layer.

    Notes
    -----
    * When ``len(z_pf) == 1`` a single phase function is assumed
      for the entire column and an all-zeros index array is returned.
    * When *z_full* extends below *z_pf* (ocean case —i.e.,
      ``sum(z_full) < 0``), the lowest phase-function layer is assigned
      to all deeper levels.
    """
    z_full = np.atleast_1d(z_full).astype(np.float32)
    z_pf = np.atleast_1d(z_pf).astype(np.float32)

    # Particular case with only 1 phase matrix for the whole z column
    if len(z_pf) == 1:
        ida = np.zeros_like(z_full, dtype=np.int32)
        return ida

    grid_full = z_full
    grid_pf = z_pf
    size_layers_full = np.concatenate(
        (np.array([1e6]), np.abs(np.diff(grid_full)))
    )
    size_layers_pf = np.concatenate(
        (np.array([1e6]), np.abs(np.diff(grid_pf)))
    )

    nz_full = len(grid_full)
    nz_pf = len(grid_pf)

    zmin_print = [-1e8]
    zmax_print = [-1e8]

    ida = np.full(nz_full, -1, dtype=np.int32)
    for i_full in range(0, nz_full):
        idz_full = (nz_full - 1) - i_full
        zmin_full = grid_full[idz_full]
        zmax_full = grid_full[idz_full] + size_layers_full[idz_full]

        # First find all the z_pf layers respecting the 2 conditions
        ida_tmp = []
        for i_pf in range(0, nz_pf):
            idz_pf = (nz_pf - 1) - i_pf
            zmin_pf = grid_pf[idz_pf]
            zmax_pf = grid_pf[idz_pf] + size_layers_pf[idz_pf]
            cond_1 = zmin_pf < zmax_full
            cond_2 = zmax_pf > zmin_full
            if cond_1 and cond_2:
                ida_tmp.append(idz_pf)

        n_ida_tmp = len(ida_tmp)
        # if only one pf layer respects the conditions,
        # take directly its index
        if n_ida_tmp == 1:
            ida[idz_full] = ida_tmp[0]
        # if more than one, find which pf layer best fills
        # the z_full layer
        elif n_ida_tmp > 1:
            pfs_weight = np.zeros(n_ida_tmp)
            for k in range(0, n_ida_tmp):
                zmin_pf_k = grid_pf[ida_tmp[k]]
                zmax_pf_k = grid_pf[ida_tmp[k]] + size_layers_pf[ida_tmp[k]]
                pf_full_min = max(zmin_pf_k, zmin_full)
                pf_full_max = min(zmax_pf_k, zmax_full)
                # Check if phase matrix is non-zero
                if phase is None:
                    pfs_weight[k] = pf_full_max - pf_full_min
                else:  # phase is an xr.DataArray
                    if np.sum(phase.values[0, ida_tmp[k], 0, :]) > 0.0:
                        pfs_weight[k] = pf_full_max - pf_full_min
                    else:
                        if (zmax_pf_k < 1e6) and (
                            (zmin_pf_k not in zmin_print)
                            and (zmax_pf_k not in zmax_print)
                        ):
                            print(
                                "Warning: null phase matrix between ",
                                zmin_pf_k,
                                " and ",
                                zmax_pf_k,
                                " detected! Please check pfgrid and/or"
                                + " grid (z_atm) values.",
                            )
                            zmin_print.append(float(zmin_pf_k))
                            zmax_print.append(float(zmax_pf_k))
                        pfs_weight[k] = (pf_full_max - pf_full_min) * 1e-6
            ida[idz_full] = ida_tmp[np.argmax(pfs_weight)]
        elif (
            n_ida_tmp == 0 and np.sum(z_full) < 0.0
        ):  # Particular case of min z_pf > min z_full in ocean
            ida[idz_full] = int(len(z_pf) - 1)
    return ida


def read_phase_nc(
    fname: PathType,
    kind: str = "atm",
    normalize: bool = True,
    wavelength_phase: NumericArrayLike | None = None,
    pfgrid: NumericArrayLike | None = None,
    z_rh_reff: NumericArrayLike | None = None,
) -> xr.DataArray:
    """
    Read and process phase function data from SMART-G NetCDF
    aerosol/cloud files.

    Loads phase matrix data from SMART-G aerosol and cloud files with
    .nc suffix. Supports wavelength and humidity/effective radius
    interpolation and normalization. Produces phase function data ready
    for SMART-G's AerOPAC/Cloud phase parameter.

    Parameters
    ----------
    fname : str or path-like
        Path to a SMART-G phase function NetCDF file (suffix: .nc).

        The file must include variables:
        - 'phase': phase matrix data [rh/reff, wavelength, stk, theta]
        - 'wav': wavelength values (in nm)
        - 'theta': scattering angle grid (uniform, in degrees)
        - 'hum' or 'reff': relative humidity (%) or effective radius
          values

    kind : str, optional
        Medium label used in the theta dimension name ('theta_' + kind).
        Accepted values are:
        - 'atm' for atmosphere
        - 'oc' for ocean
        Default: 'atm'

    normalize : bool, optional
        If True, normalize the phase matrix P11 term such that the
        integral over all angles equals 2.
        Default: True

    wavelength_phase : float or array_like, optional
        Wavelength(s) (in nm) to interpolate to. Required if the file
        contains multiple wavelengths (n_wavelength > 1). This
        parameter has the same meaning as ``wavelength_phase`` in
        the ``Atm1D`` constructor.
        Default: None

    pfgrid : array_like, optional
        Altitude grid [z_top, z_1, z_2, ..., z_bottom] (in km,
        descending order) for altitude-dependent phase functions. If
        provided with n_rh_reff > 1, the z_rh_reff values will be
        interpolated onto this grid. The first element (z_top) is
        skipped; remaining elements define the z_phase coordinate. This
        parameter has the same meaning as ``pfgrid`` in the ``Atm1D``
        constructor.
        Default: None

    z_rh_reff : float or array_like, optional
        Interpolation target for the second phase-function axis:
        - aerosol files: relative humidity (%)
        - cloud files: effective radius (reff)
        Required if the file contains multiple rh/reff values
        (n_rh_reff > 1). If array_like (1-D), pfgrid must also be
        provided to map these values to specific altitudes, and
        ``len(z_rh_reff)`` must equal ``len(pfgrid) - 1``.
        Default: None

    Returns
    -------
    da_pha : DataArray
        Phase matrix as xarray DataArray with dimensions:
        - 'wavelength_phase': wavelength (in nm)
        - 'z_phase': altitude (in km) from pfgrid or [0.]
        - 'nphamat': phase matrix unique terms (0 to nphamat-1)
          nphamat = 4 for spherical particles only nphamat = 6 for
          spherical or non-spherical particles (for spherical:
          P22=P11, P44=P33)
        - 'theta_'+kind: scattering angle (in degrees)

        Coordinates are replaced/renamed such that the rh/reff dimension
        becomes 'z_phase' with values from pfgrid[1:] or [0.] if pfgrid
        is None.

    Examples
    --------
    Read phase function for a single wavelength and rh:

    >>> pha = read_phase_nc(
    ...     'desert_sol.nc', wavelength_phase=550.0,
    ...     z_rh_reff=[70.0, 60., 58.],
    ...     pfgrid=[100., 50., 10., 0.],
    ...     normalize=True)
    >>> pha.shape
    (1, 3, 6, 721)  # (wavelength_phase, z_phase, nphamat, theta_atm)
    """
    wavelength_phase = (
        np.asarray(wavelength_phase, dtype=np.float32)
        if wavelength_phase is not None else None
    )
    pfgrid = (
        np.atleast_1d(pfgrid).astype(np.float32)
        if pfgrid is not None
        else None
    )
    z_rh_reff = (
        np.asarray(z_rh_reff, dtype=np.float32)
        if z_rh_reff is not None
        else None
    )

    ds = xr.open_dataset(fname)

    if "hum" in ds.variables:
        rh_reff = ds["hum"].data
        rh_or_reff = "rh"
    elif "reff" in ds.variables:
        rh_reff = ds["reff"].data
        rh_or_reff = "reff"
    else:
        raise Exception("Error")

    ntheta = ds.dims["theta"]
    n_rh_reff = rh_reff.size
    n_wavelength = ds.dims["wav"]
    theta = ds.theta.values
    wavelength = ds.wav.values

    # Get nphamat from phase data shape
    nphamat = ds["phase"].shape[2]

    da_pha = xr.DataArray(
        np.zeros((n_wavelength, n_rh_reff, nphamat, ntheta)),
        coords=[wavelength, rh_reff, np.arange(nphamat), theta],
        dims=["wavelength_phase", rh_or_reff, "nphamat", "theta_" + kind],
        name="phase_" + kind,
    )
    da_pha.data[:, :, :, :] = ds["phase"].values.swapaxes(0, 1)

    if normalize:
        mu = np.cos(np.deg2rad(theta))
        idmu = np.argsort(mu)
        for i_wavelength in range(0, n_wavelength):
            for irhreff in range(0, n_rh_reff):
                f = da_pha.data[i_wavelength, irhreff, 0, :]  # P11 term
                norm = np.trapezoid(f[idmu], mu[idmu])
                da_pha.data[i_wavelength, irhreff, :, :] *= 2.0 / abs(norm)

    if n_wavelength > 1 and wavelength_phase is not None:
        da_pha = da_pha.interp(wavelength_phase=wavelength_phase)
    elif n_wavelength > 1 and wavelength_phase is None:
        raise ValueError(
            "wavelength_phase must be provided when n_wavelength > 1"
        )

    if n_rh_reff > 1 and z_rh_reff is not None:
        da_pha = da_pha.interp(
            {rh_or_reff: z_rh_reff}, kwargs={"bounds_error": True}
        )
    elif n_rh_reff > 1 and z_rh_reff is None:
        raise ValueError("z_rh_reff must be provided when n_rh_reff > 1")

    if pfgrid is None:
        z_phase = np.array([0.0], dtype=float)
    else:
        z_phase = np.atleast_1d(pfgrid).astype(np.float32)[1:]

    if da_pha.sizes[rh_or_reff] != z_phase.size:
        raise ValueError(
            f"Cannot replace '{rh_or_reff}' with 'z_phase': size mismatch "
            f"({da_pha.sizes[rh_or_reff]} vs {z_phase.size})."
        )
    da_pha = da_pha.assign_coords({rh_or_reff: z_phase}).rename(
        {rh_or_reff: "z_phase"}
    )

    return da_pha


def read_phase_dat(
    fname: PathType,
    kind: str = "atm",
    normalize: bool = True,
) -> xr.DataArray:
    """
    Read a phase matrix from a space-separated `.dat` file.

    The file is expected to have no header. The first column contains
    the scattering angles (in degrees), and the remaining columns
    contain the phase matrix elements (one column per element). The
    phase matrix is assumed to be monochromatic and vertically uniform
    (no wavelength or altitude dependence).

    Parameters
    ----------
    fname : str or path-like
        Path to the `.dat` phase function file.
    kind : str, optional
        Medium label used in the theta dimension name
        ('theta_' + kind). Accepted values are:
        - 'atm' for atmosphere
        - 'oc' for ocean
        Default: 'atm'
    normalize : bool, optional
        If True, normalize the phase matrix P11 term such that the
        integral over all angles equals 2.
        Default: True.

    Returns
    -------
    da_pha : DataArray
        Phase matrix with dimensions:

        - ``'wavelength_phase'`` : wavelength (single value: 0.0)
        - ``'z_phase'`` : altitude (single value: 0.0 km)
        - ``'nphamat'`` : phase matrix element index (0 to nphamat-1)
          nphamat = 4 for spherical particles only
          nphamat = 6 for spherical or non-spherical particles
          (for spherical: P22=P11, P44=P33)
        - ``'theta_' + kind`` : scattering angle in degrees

    Examples
    --------
    >>> pha = read_phase_dat('phase.dat', kind='atm', normalize=True)
    >>> pha.dims
    ('wavelength_phase', 'z_phase', 'nphamat', 'theta_atm')
    """
    df = pd.read_csv(fname, sep=r"\s+", header=None)

    theta = np.asarray(df.iloc[:, 0].values)
    pha = np.asarray(df.iloc[:, 1:].values)
    pha = pha.swapaxes(0, 1)

    if normalize:
        mu = np.cos(np.deg2rad(theta))
        idmu = np.argsort(mu)
        f = pha[0, :]  # P11 term
        pha = (2.0 * pha) / np.trapezoid(f[idmu], mu[idmu])

    # Add wavelength_phase and z_phase dimensions for consistency
    # with other readers
    wavelength_phase = np.array([0.0], dtype=float)
    z_phase = np.array([0.0], dtype=float)
    pha_4d = pha[
        np.newaxis, np.newaxis, :, :
    ]  # Add two dimensions at the front

    da_pha = xr.DataArray(
        pha_4d,
        coords=[wavelength_phase, z_phase, np.arange(pha.shape[0]), theta],
        dims=["wavelength_phase", "z_phase", "nphamat", "theta_" + kind],
        name="phase_" + kind,
    )

    return da_pha


def _phase_cdf_rh_or_reff(
    ds: xr.Dataset,
) -> tuple[NDArray[np.floating[Any]], str]:
    """Return the hum or reff axis of a cdf phase function file.

    Parameters
    ----------
    ds : Dataset
        The open phase function file.

    Returns
    -------
    rh_reff : ndarray
        Values of the relative humidity or effective radius axis.
    rh_or_reff : str
        Name of the axis, 'hum' or 'reff'.
    """
    if "hum" in ds.variables:
        return ds["hum"].data, "hum"
    if "reff" in ds.variables:
        return ds["reff"].data, "reff"
    raise ValueError(
        "The phase function file must contain either a 'hum' or "
        "a 'reff' variable."
    )


def _resample_cdf_phase(
    ds: xr.Dataset,
    theta: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Resample the phase matrix of a cdf file onto a uniform grid.

    The theta grid of a libRadtran cdf file varies with the
    wavelength, the rh/reff value and the matrix term; each entry is
    linearly interpolated here onto the single grid *theta*.

    Parameters
    ----------
    ds : Dataset
        The open phase function file, with the variables 'phase',
        'theta' and 'ntheta'.
    theta : ndarray
        Scattering angles to resample on, in degrees.

    Returns
    -------
    ndarray
        The resampled phase matrices, of shape
        (n_wavelength, n_rh_reff, nphamat, ntheta).
    """
    phase = ds["phase"][:, :, :, :].data
    n_wavelength, n_rh_reff, n_stk = phase.shape[:3]

    data = np.zeros((n_wavelength, n_rh_reff, n_stk, theta.size))
    for i_wavelength in range(0, n_wavelength):
        for irhreff in range(n_rh_reff):
            for istk in range(n_stk):
                # ntheta (wavelength, rh/reff, nphamat)
                nth = ds["ntheta"][i_wavelength, irhreff, istk].data

                # theta (wavelength, rh/reff, nphamat, ntheta)
                th = ds["theta"][i_wavelength, irhreff, istk, :].data

                data[i_wavelength, irhreff, istk, :] = np.interp(
                    theta,
                    th[:nth],
                    phase[i_wavelength, irhreff, istk, :nth],
                    period=np.inf,
                )
    return data


def read_phase_cdf(
    fname: PathType,
    kind: str = "atm",
    normalize: bool = True,
    ntheta_max: int = 18001,
    wavelength_phase: NumericArrayLike | None = None,
    pfgrid: NumericArrayLike | None = None,
    z_rh_reff: NumericArrayLike | None = None,
) -> xr.DataArray:
    """
    Read and process phase function data from libRadtran NetCDF
    aerosol/cloud files.

    Loads phase matrix data from libRadtran aerosol and cloud phase
    function files with .cdf suffix (e.g., 'ssam.mie.cdf',
    'wc.sol.mie.cdf'). Handles non-uniform theta grids from libRadtran
    by resampling to a uniform scattering angle grid. Supports
    wavelength and humidity/effective radius interpolation and
    normalization. Produces phase function data ready for SMART-G's
    AerOPAC/Cloud phase parameter.

    Parameters
    ----------
    fname : str or path-like
        Path to a libRadtran phase function NetCDF file (suffix:
        .cdf).
        Examples: 'ssam.mie.cdf', 'wc.sol.mie.cdf', 'cloud.water.cdf'.

        The file must include variables:
        - 'phase': phase matrix data [wavelength, rh/reff, nphamat,
          theta]
        - 'wavelen': wavelength values (in micrometers)
        - 'theta': scattering angle grids (non-uniform, varies per
          entry)
        - 'ntheta': number of valid theta values per entry
        - 'nphamat': number of Stokes matrix elements (typically 6)
        - 'hum' or 'reff': relative humidity (%) or effective radius
          values

    kind : str, optional
        Medium label used in the theta dimension name ('theta_' +
        kind). Accepted values are:
        - 'atm' for atmosphere
        - 'oc' for ocean
        Default: 'atm'

    normalize : bool, optional
        If True, normalize the phase matrix P11 term such that the
        integral over all angles equals 2.

    ntheta_max : int, optional
        Maximum number of scattering angle points to use. If the
        file provides higher resolution, it will be reduced to this
        limit.
        Default: 18001

    wavelength_phase : float or array_like, optional
        Wavelength(s) (in micrometers) to interpolate to. Required
        if the file contains multiple wavelengths (n_wavelength > 1).
        This parameter has the same meaning as ``wavelength_phase``
        in the ``Atm1D`` constructor.
        Default: None

    pfgrid : array_like, optional
        Altitude grid [z_top, z_1, z_2, ..., z_bottom] (in km,
        descending order) for altitude-dependent phase functions. If
        provided with n_rh_reff > 1, the z_rh_reff values will be
        interpolated onto this grid. The first element (z_top) is
        skipped; remaining elements define the z_phase coordinate.
        This parameter has the same meaning as ``pfgrid`` in the
        ``Atm1D`` constructor.
        Default: None

    z_rh_reff : float or array_like, optional
        Interpolation target for the second phase-function axis:
        - aerosol files: relative humidity (%)
        - cloud files: effective radius (reff)
        Required if the file contains multiple rh/reff values
        (n_rh_reff > 1). If array-like (1-D), pfgrid must also be
        provided to map these values to specific altitudes, and
        ``len(z_rh_reff)`` must equal ``len(pfgrid) - 1``.
        Default: None

    Returns
    -------
    da_pha : DataArray
        Phase matrix as xarray DataArray with dimensions:
        - 'wavelength_phase': wavelength (in nm)
        - 'z_phase': altitude (in km) from pfgrid or [0.]
        - 'nphamat': phase matrix unique terms (0 to nphamat-1)
          nphamat = 4 for spherical particles only
          nphamat = 6 for spherical or non-spherical particles (for
          spherical: P22=P11, P44=P33)
        - 'theta_'+kind: scattering angle (in degrees)

        Coordinates are replaced/renamed such that the rh/reff
        dimension becomes 'z_phase' with values from pfgrid[1:] or
        [0.] if pfgrid is None.

    Examples
    --------
    Read phase function for a single wavelength and rh:

    >>> pha = read_phase_cdf(
    ...     'ssam.mie.cdf', wavelength_phase=550.0,
    ...     z_rh_reff=[70.0, 60., 58.],
    ...     pfgrid=[100., 50., 10., 0.],
    ...     normalize=True)
    >>> pha.shape
    (1, 3, 6, 18001)  # (wavelength_phase, z_phase, nphamat, theta_atm)
    """
    wavelength_phase = (
        np.asarray(wavelength_phase, dtype=np.float32)
        if wavelength_phase is not None else None
    )
    pfgrid = (
        np.atleast_1d(pfgrid).astype(np.float32)
        if pfgrid is not None
        else None
    )
    z_rh_reff = (
        np.asarray(z_rh_reff, dtype=np.float32)
        if z_rh_reff is not None
        else None
    )

    ds = xr.open_dataset(fname)

    rh_reff, rh_or_reff = _phase_cdf_rh_or_reff(ds)

    dtheta_min = np.nanmin(np.abs(np.diff(ds.theta.values, axis=3)))
    ntheta = np.ceil(180 / dtheta_min).astype(int) + 1
    ntheta = min(ntheta, ntheta_max)  # be sure to not exceed ntheta_max
    nphamat = ds.nphamat.size
    n_rh_reff = rh_reff.size
    n_wavelength = ds["wavelen"].size
    theta = np.linspace(0, 180, ntheta)
    wavelength = ds["wavelen"].data * 1e3

    # checks at the beginning to avoid unnecessary computations
    if n_wavelength > 1 and wavelength_phase is None:
        raise ValueError(
            "Phase function file contains more than 1 wavelength. "
            "Please provide the 'wavelength_phase' parameter "
            "(float or 1-D array) "
            "to select/interpolate the desired wavelength(s)."
        )
    if n_rh_reff > 1 and (
        z_rh_reff is None or (not z_rh_reff.size == 1 and pfgrid is None)
    ):
        raise ValueError(
            f"Phase function file contains more than 1 {rh_or_reff} value. "
            "Please provide the 'z_rh_reff' parameter (float or 1-D array). "
            "If 'z_rh_reff' is a 1-D array, also provide the 'pfgrid' "
            "parameter (float or 1-D array) "
            f"to select/interpolate the desired {rh_or_reff} value(s)."
        )
    if n_rh_reff > 1 and z_rh_reff is not None and pfgrid is not None:
        if z_rh_reff.size != pfgrid.size - 1:
            raise ValueError(
                "Invalid 'z_rh_reff' size: when 'z_rh_reff' is a 1-D array, "
                "its size must be len(pfgrid) - 1. "
                f"Got len(z_rh_reff)={z_rh_reff.size}"
                f" and len(pfgrid)={pfgrid.size}."
            )
    elif n_rh_reff > 1 and (z_rh_reff is None or pfgrid is None):
        raise ValueError(
            "When the phase function file contains more than 1 "
            f"{rh_or_reff} value, both 'z_rh_reff' and 'pfgrid'"
            "parameters must be provided."
        )

    da_pha = xr.DataArray(
        _resample_cdf_phase(ds, theta),
        coords=[wavelength, rh_reff, np.arange(nphamat), theta],
        dims=["wavelength_phase", rh_or_reff, "nphamat", "theta_" + kind],
        name="phase_" + kind,
    )

    if normalize:
        mu = np.cos(np.deg2rad(theta))
        idmu = np.argsort(mu)
        for i_wavelength in range(0, n_wavelength):
            for irhreff in range(0, n_rh_reff):
                f = da_pha.data[i_wavelength, irhreff, 0, :]  # P11 term
                norm = np.trapezoid(f[idmu], mu[idmu])
                da_pha.data[i_wavelength, irhreff, :, :] *= 2.0 / abs(norm)

    if n_wavelength > 1:
        da_pha = da_pha.interp(wavelength_phase=wavelength_phase)

    if n_rh_reff > 1:
        da_pha = da_pha.interp(
            {rh_or_reff: z_rh_reff}, kwargs={"bounds_error": True}
        )

    if pfgrid is None:
        z_phase = np.array([0.0], dtype=float)
    else:
        z_phase = np.atleast_1d(pfgrid).astype(np.float32)[1:]

    if da_pha.sizes[rh_or_reff] != z_phase.size:
        raise ValueError(
            f"Cannot replace '{rh_or_reff}' with 'z_phase': size mismatch "
            f"({da_pha.sizes[rh_or_reff]} vs {z_phase.size})."
        )
    da_pha = da_pha.assign_coords({rh_or_reff: z_phase}).rename(
        {rh_or_reff: "z_phase"}
    )

    return da_pha


def read_phase(
    fname: PathType,
    kind: str = "atm",
    normalize: bool = True,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Read phase function data from a file and dispatch to the proper
    reader.

    This convenience function selects the backend according to the
    file suffix:

    - ``.dat`` -> :func:`read_phase_dat`
    - ``.nc`` -> :func:`read_phase_nc`
    - ``.cdf`` -> :func:`read_phase_cdf`

    Parameters
    ----------
    fname : str or path-like
        Path to a phase function file. Supported formats are
        ``.dat``, ``.nc``, and ``.cdf``.

    kind : str, optional
        Medium label used in the theta dimension name ('theta_' + kind).
        Accepted values are:
        - 'atm' for atmosphere
        - 'oc' for ocean
        Default: 'atm'

    normalize : bool, optional
        If True, normalize the phase matrix P11 term such that the
        integral over all angles equals 2.
        Default: True

    **kwargs : dict, optional
        Additional keyword arguments forwarded to the selected backend
        reader:

        - for ``.nc``: forwarded to :func:`read_phase_nc`
        - for ``.cdf``: forwarded to :func:`read_phase_cdf`

        Typical arguments include ``wavelength_phase``, ``pfgrid``,
        ``z_rh_reff``,
        and ``ntheta_max`` (only for ``.cdf``).

    Returns
    -------
    DataArray
        Phase matrix data as returned by the selected backend reader.
        All backends return a 4-dimensional array with dimensions:

        - ``('wavelength_phase', 'z_phase', 'nphamat',
          'theta_' + kind)``

        where 'nphamat' has size nphamat:
        - nphamat = 4 for spherical particles only
        - nphamat = 6 for spherical or non-spherical particles
          (for spherical: P22=P11, P44=P33)

    Examples
    --------
    >>> pha = read_phase('phase.dat', kind='atm', normalize=True)
    >>> pha = read_phase('desert_sol.nc', kind='atm',
    ...                  wavelength_phase=550.0,
    ...                  z_rh_reff=[70.0, 60.0, 58.0],
    ...                  pfgrid=[100.0, 50.0, 10.0, 0.0])
    >>> pha = read_phase('ssam.mie.cdf', kind='atm',
    ...                  wavelength_phase=550.0,
    ...                  z_rh_reff=[70.0, 60.0, 58.0],
    ...                  pfgrid=[100.0, 50.0, 10.0, 0.0],
    ...                  ntheta_max=18001)
    """

    fname = Path(fname)

    if not fname.is_file():
        raise FileNotFoundError(f"Phase function file not found: {fname}")

    supported_formats = [".dat", ".nc", ".cdf"]

    if fname.suffix == ".dat":
        return read_phase_dat(fname, kind=kind, normalize=normalize)
    elif fname.suffix == ".nc":
        return read_phase_nc(
            fname, kind=kind, normalize=normalize, **kwargs
        )
    elif fname.suffix == ".cdf":
        return read_phase_cdf(
            fname, kind=kind, normalize=normalize, **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported phase function file format: "
            f"{fname.suffix}. Supported formats: {supported_formats}"
        )


def expand_phase_4_to_6(
    phase: xr.DataArray | LUT | None,
) -> xr.DataArray | None:
    """
    Convert a 4-term phase matrix into its 6-term equivalent.

    The 4 terms (F11, F21, F33, F34) of a spherical particle are
    completed into the 6 terms expected by SMART-G by duplicating
    F22 = F11 and F44 = F33. This is the companion of the ``read_phase``
    family, whose readers return 4 terms for files describing spherical
    particles only.

    Parameters
    ----------
    phase : DataArray or LUT or None
        Phase matrices with dimensions [n_wavelength, nz, nphamat,
        angle]. A LUT is converted to a DataArray first.

    Returns
    -------
    DataArray or None
        The 6-term phase matrices, with the dimensions and coordinates
        of the input, or the input unchanged if it already has 6 terms
        or is None.

    Raises
    ------
    TypeError
        If `phase` is neither a DataArray, a LUT nor None.

    See Also
    --------
    convert_phase_to_iparper : Same completion applied to a plain
        ndarray, followed by the conversion of the IQUV convention into
        the parallel/perpendicular one used by SMART-G.
    """
    if phase is None:
        return None
    if isinstance(phase, LUT):
        phase = phase.to_xarray()
    if not isinstance(phase, xr.DataArray):
        raise TypeError(
            "The phase matrices must be provided as a DataArray or as "
            f"a LUT, not as a {type(phase).__name__}."
        )
    if phase.shape[2] != 4:
        return phase

    pha_6 = np.zeros(
        (phase.shape[0], phase.shape[1], 6, phase.shape[3]),
        dtype=np.float64,
    )
    pha_6[:, :, 0:4, :] = phase[:, :, :, :].copy()  # F11, F21, F33, F34
    pha_6[:, :, 4, :] = phase[:, :, 0, :].copy()  # F22 = F11
    pha_6[:, :, 5, :] = phase[:, :, 2, :].copy()  # F44 = F33
    axes = list(phase.dims)
    coords = {}
    for i, dim in enumerate(axes):
        if i == 2:
            coords[dim] = np.arange(6)
        elif (
            dim in phase.coords
            and phase.coords[dim].size == pha_6.shape[i]
        ):
            coords[dim] = phase.coords[dim].values
        else:
            coords[dim] = np.arange(pha_6.shape[i])
    return xr.DataArray(pha_6, dims=axes, coords=coords)


def convert_phase_to_iparper(
    pha: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """
    Convert phase matrix to parallel/perpendicular intensity
    convention.

    Converts the phase matrix from the standard IQUV (Stokes vector)
    convention to the Ipar/Iper (parallel/perpendicular intensity)
    convention used throughout SMART-G. This conversion is necessary
    when using the alternative Stokes representation where polarized
    light is characterized by (Ipar, Iper, U, V) instead of
    (I, Q, U, V).

    Parameters
    ----------
    pha : ndarray
        The phase matrix in IQUV convention. Can be either:
        - 2-D array of shape (nphamat, nth): phase matrix with Stokes
          components in dimension 0
        - 4-D array of shape (n1, n2, nphamat, nth): batch of phase
          matrices with Stokes components in dimension 2
        where nphamat is 4 (only spherical particles) or 6 (spherical
        and non-spherical particles), and nth is the number of
        scattering angles.

        Input phase matrix components in order:
        - If nphamat=4: p11, p21, p33, p34
        - If nphamat=6: p11, p21, p33, p34, p22, p44

    Returns
    -------
    pha_converted : ndarray
        The phase matrix converted to Ipar/Iper convention. Always has
        6 components output (dimensions are preserved except Stokes
        dimension becomes 6):
        - 2-D input returns shape (6, nth)
        - 4-D input returns shape (n1, n2, 6, nth)

    References
    ----------
    .. [1] Chandrasekhar, S. (2013). Radiative transfer. Courier
           Corporation.
    """

    ndim = pha.ndim
    if ndim not in (2, 4):
        raise ValueError("The phase matrix dimension must be 2 or 4!")

    # Normalize to 4D: (n1, n2, nphamat, nth)
    if ndim == 2:
        pha = pha[np.newaxis, np.newaxis, :, :]

    nphamat = pha.shape[2]
    nth = pha.shape[3]
    if nphamat not in (4, 6):
        raise ValueError(
            "The number of phase matrix terms must be equal to 4 or 6!"
        )

    pha_converted = np.zeros(
        (pha.shape[0], pha.shape[1], 6, nth), dtype=np.float64
    )
    if nphamat == 4:  # spherical particles
        pha_converted[:, :, 0:4, :] = pha
        pha_converted[:, :, 4, :] = pha[:, :, 0, :]  # p22 = p11
        pha_converted[:, :, 5, :] = pha[:, :, 2, :]  # p44 = p33
    else:  # non spherical particles
        pha_converted[:, :, :, :] = pha

    p0 = pha_converted[:, :, 0, :].copy()
    p1 = pha_converted[:, :, 1, :].copy()
    p4 = pha_converted[:, :, 4, :].copy()
    pha_converted[:, :, 0, :] = 0.5 * (p0 + 2 * p1 + p4)  # P11
    pha_converted[:, :, 1, :] = 0.5 * (p0 - p4)  # P12=P21
    pha_converted[:, :, 4, :] = 0.5 * (p0 - 2 * p1 + p4)  # P22

    if ndim == 2:
        return pha_converted[0, 0, :, :]

    return pha_converted


def get_prof_phases(
    phase: xr.DataArray,
    wavelength: NumericArrayLike,
    z: NumericArrayLike,
) -> tuple[NDArray[np.int32], list[xr.DataArray]]:
    """
    Generate prof_phases parameter for Atm1D from phase function data.

    Constructs the prof_phases tuple required by Atm1D initialization.
    This function directly produces the format needed for the
    prof_phases parameter.

    Parameters
    ----------
    phase : DataArray
        Phase matrix data read from read_phase(). Expected dimensions:
        ('wavelength_phase', 'z_phase', 'nphamat', 'theta_atm')
    wavelength : array_like
        Full wavelength grid in nanometers. Must match the wavelengths
        used in Atm1D.calc() method. Equivalent to the 'wavelength'
        parameter
        passed to Atm1D.calc().
    z : array_like
        Full altitude grid in kilometers (descending order from TOA to
        BOA). Must match the 'grid' parameter used in Atm1D
        initialization.

    Returns
    -------
    ipha : ndarray
        Phase matrix indices for mapping the full wavelength/altitude
        grid.
    phases : list of DataArray
        List of phase matrix DataArrays with dimensions
        ``('nphamat', 'theta_atm')``.
    """
    wavelength = np.atleast_1d(wavelength).astype(np.float32)
    z = np.atleast_1d(z).astype(np.float32)

    pha_atm, ipha_atm = calc_iphase(phase, wavelength, z)
    lpha_da = []
    for i in range(pha_atm.shape[0]):
        lpha_da.append(
            xr.DataArray(
                pha_atm[i, :, :],
                dims=["nphamat", "theta_atm"],
                coords={"theta_atm": phase.theta_atm.values},
            )
        )

    prof_phases = (ipha_atm, lpha_da)

    return prof_phases


def read_phase_nth_cte(
    filename: PathType,
    nb_theta: int = 721,
    normalize: bool = False,
) -> xr.DataArray:
    """Read an aerosol or cloud file on a constant theta grid.

    Both the libRadtran files (e.g. wc.sol.mie.cdf) and the
    monochromatic IPRT netCDF files are accepted. Their phase matrix
    is given on a theta grid whose length varies with the wavelength
    and the component; it is interpolated here on a single grid of
    nb_theta angles.

    The matrix keeps the IQUV convention of the file, the conversion
    into the parallel/perpendicular convention of the kernels being
    done by the run method.

    Parameters
    ----------
    filename : str or path-like
        Path of the netCDF file to read.
    nb_theta : int, optional
        Number of theta values between 0 and 180 degrees.
        Default: 721
    normalize : bool, optional
        If True, normalize the phase matrix so that the integral of
        the F11 term over all angles equals 2.
        Default: False

    Returns
    -------
    DataArray
        The phase matrix, of shape (n_wavelength, nrh_or_reff, 6,
        nb_theta),
        with the dimensions 'wavelength_phase' (nm), 'hum' or 'reff'
        (kept from the file), 'nphamat' and 'theta_atm' (degrees).
    """
    ds = xr.open_dataset(filename)

    rh_reff, rh_or_reff = _phase_cdf_rh_or_reff(ds)

    n_stk = ds.nphamat.size
    if n_stk not in (4, 6):
        raise ValueError(
            "The number of phase matrix terms in the file must be "
            f"equal to 4 or 6, got {n_stk}."
        )

    n_theta = nb_theta
    n_rh_or_reff = rh_reff.size
    n_wavelength = ds["wavelen"].size
    theta = np.linspace(0., 180., num=n_theta)
    wavelength = ds["wavelen"].data * 1e3

    data = np.full((n_wavelength, n_rh_or_reff, 6, n_theta), np.nan,
                   dtype=np.float32)
    data[:, :, :n_stk, :] = _resample_cdf_phase(ds, theta)

    if n_stk == 4:  # only spherical particles
        data[:, :, 4, :] = data[:, :, 0, :].copy()  # F22 = F11
        data[:, :, 5, :] = data[:, :, 2, :].copy()  # F44 = F33

    if normalize:
        mu = np.cos(np.deg2rad(theta))
        idmu = np.argsort(mu)
        for i_wavelength in range(0, n_wavelength):
            for irhreff in range(0, n_rh_or_reff):
                f = data[i_wavelength, irhreff, 0, :]  # F11 term
                norm = np.trapezoid(f[idmu], mu[idmu])
                data[i_wavelength, irhreff, :, :] *= 2. / abs(norm)

    return xr.DataArray(
        data,
        coords=[wavelength, rh_reff, np.arange(6), theta],
        dims=["wavelength_phase", rh_or_reff, "nphamat", "theta_atm"],
        name="phase_atm",
    )
