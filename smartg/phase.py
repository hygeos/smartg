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
    aerosol/cloud files (``.cdf`` suffix), the monochromatic IPRT
    files included, resampled onto one scattering angle grid.

Each reader returns either the phase matrix laid on a 1D profile
(``output_sg_ready=True``, the default: dimensions ``wavelength_phase``,
``z_phase``, ``nphamat``, ``theta_<kind>``, for the ``phase`` argument
of ``AerOPAC`` / ``Cloud`` / ``Hydrosol`` and ``Atm1D.prof_phases``) or
the table on the axes of the file (``output_sg_ready=False``:
``wavelength_phase``, ``hum`` or ``reff``, ``nphamat``,
``theta_<kind>``, for the ``phase`` argument of ``Cloud3D`` /
``Aer3D``).

Phase Matrix Processing
------------------------
theta_grid
    Build the scattering angle grid of a phase matrix, either
    equally spaced or clustered towards the forward and backward
    directions.
as_theta_grid
    Read a scattering angle grid from an ``n_theta`` argument, which
    is either a number of equally spaced angles or the angles
    themselves.
union_theta_grid
    Merge several scattering angle grids into the union of their
    nodes, on which a mixture of phase matrices is exact.
is_native_theta
    Whether an ``n_theta`` argument asks for the native grid of the
    source tables, i.e. is the string ``'native'``.
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

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from luts.luts import LUT
from numpy.typing import NDArray
from pytrunc.utils import quadrature_lobatto

from smartg.typing import NumericArrayLike, PathType, ThetaLike

THETA_GRID_KINDS = ('uniform', 'chebyshev', 'lobatto', 'peak')

# The ``n_theta`` value asking for the scattering angles the source
# tables are tabulated on, resolved by the phase methods themselves
NATIVE_THETA = 'native'


def theta_grid(
    n: int,
    kind: str = 'uniform',
    unit: str = 'deg',
    theta_fwd: float = 5.0,
    theta_bwd: float = 5.0,
    frac_fwd: float = 0.20,
    frac_bwd: float = 0.10,
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
        - ``'peak'`` -> three zones, equally spaced within each: a
          refined one below ``theta_fwd``, a refined one above
          ``180 - theta_bwd``, and the rest of the range between them
    unit : str, optional
        Unit of the returned angles, ``'deg'`` (default) or ``'rad'``.
    theta_fwd, theta_bwd : float, optional
        ``'peak'`` only. Width in degrees of the refined forward and
        backward zones, whatever ``unit`` is. Default 5 degrees each.
    frac_fwd, frac_bwd : float, optional
        ``'peak'`` only. Fraction of the ``n`` nodes given to each
        refined zone. Default 0.20 forward and 0.10 backward, which
        leaves 0.70 of them for the rest of the range.

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

    Clustering is not free: it takes its nodes from the middle of the
    range. Measured on ``watercloud_670.mie.cdf`` at n = 1801, the
    largest relative error of the table over 10 to 175 degrees is
    7.9e-3 on a Lobatto grid against 2.7e-3 on a uniform one, so a
    geometry that scatters mostly at middle angles is served worse by
    a clustered grid than by an equally spaced one of the same length.
    ``'peak'`` exists for that trade-off: unlike the two fixed kinds
    it says how many nodes each end may take. Its default is a
    compromise, marginally better than Lobatto on the worst band
    (6.5e-3 against 7.9e-3) and adjustable in either direction.

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
    elif kind == 'peak':
        if theta_fwd <= 0.0 or theta_bwd <= 0.0:
            raise ValueError(
                "The theta_fwd and theta_bwd parameters must be > 0, "
                f"got {theta_fwd} and {theta_bwd}."
            )
        if theta_fwd + theta_bwd >= 180.0:
            raise ValueError(
                "The refined zones must leave room between them: "
                f"theta_fwd + theta_bwd = {theta_fwd + theta_bwd} "
                "degrees, which is not < 180."
            )
        if frac_fwd <= 0.0 or frac_bwd <= 0.0:
            raise ValueError(
                "The frac_fwd and frac_bwd parameters must be > 0, "
                f"got {frac_fwd} and {frac_bwd}."
            )
        if n < 4:
            # too few nodes to carry three zones; every kind is the
            # two end points and whatever sits between them
            theta = np.linspace(0.0, span, n)
        else:
            # the zones are equally spaced inside themselves, so the
            # whole grid is described by where they meet and how many
            # nodes each one gets. The bounds keep one node for the
            # forward zone, one for the middle, and the two the
            # backward zone needs to reach 180 degrees.
            m_fwd = min(max(round(frac_fwd * n), 1), n - 3)
            m_bwd = min(max(round(frac_bwd * n), 2), n - 1 - m_fwd)
            m_mid = n - m_fwd - m_bwd
            # theta_fwd and theta_bwd are in degrees whatever unit is
            edge_fwd = theta_fwd * span / 180.0
            edge_bwd = span - theta_bwd * span / 180.0
            theta = np.concatenate(
                [
                    np.linspace(0.0, edge_fwd, m_fwd, endpoint=False),
                    np.linspace(edge_fwd, edge_bwd, m_mid,
                                endpoint=False),
                    np.linspace(edge_bwd, span, m_bwd),
                ]
            )
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


def is_native_theta(n_theta: ThetaLike) -> bool:
    """Whether an ``n_theta`` argument asks for the native angle grid.

    The phase methods accept the string ``'native'`` to keep the
    scattering angles their source tables are tabulated on; this is
    the test they use, written so that an array is never compared
    with a string.

    Examples
    --------
    >>> is_native_theta('native')
    True
    >>> is_native_theta(721)
    False
    >>> is_native_theta([0., 90., 180.])
    False
    """
    return isinstance(n_theta, str) and n_theta == NATIVE_THETA


def as_theta_grid(n_theta: ThetaLike) -> NDArray[np.float64]:
    """Read a scattering angle grid from an ``n_theta`` argument.

    Everywhere a phase matrix is built, its angular grid is described
    by a single ``n_theta`` argument that is either a number of
    equally spaced angles, or the angles themselves. This resolves
    both into the angles, in degrees. The string ``'native'`` is not
    resolved here, since it stands for the angles of source tables
    this function does not see: the phase methods resolve it
    themselves, see :func:`union_theta_grid`.

    Parameters
    ----------
    n_theta : int or array_like
        Number of equally spaced scattering angles, or the scattering
        angles themselves in degrees, from 0 to 180. Build a clustered
        grid with :func:`theta_grid`.

    Returns
    -------
    ndarray
        Strictly increasing angles in degrees, from 0 to 180.

    Raises
    ------
    TypeError
        If ``n_theta`` is a string: ``'native'`` is resolved by the
        phase methods, not here.
    ValueError
        If the angles are not strictly increasing, or do not span the
        whole scattering range.

    Examples
    --------
    >>> as_theta_grid(5)
    array([  0.,  45.,  90., 135., 180.])
    >>> as_theta_grid([0., 10., 180.])
    array([  0.,  10., 180.])
    """
    if isinstance(n_theta, str):
        raise TypeError(
            f"The n_theta argument {n_theta!r} names a grid this "
            "function cannot build: 'native' is resolved by the "
            "phase methods from their source tables, see "
            "union_theta_grid."
        )
    if np.ndim(n_theta) == 0:
        # a Python or NumPy scalar, or a 0-d array: item() gives the
        # Python scalar in every case
        return theta_grid(int(np.asarray(n_theta).item()))

    theta = np.ascontiguousarray(n_theta, dtype=np.float64)
    if theta.ndim != 1 or theta.size < 2:
        raise ValueError(
            "The scattering angles must be a 1-D array of at least 2 "
            f"values, got shape {theta.shape}."
        )
    if np.any(np.diff(theta) <= 0.0):
        raise ValueError(
            "The scattering angles must be strictly increasing."
        )
    if theta[0] != 0.0 or theta[-1] != 180.0:
        raise ValueError(
            "The scattering angles must span 0 to 180 degrees, got "
            f"{theta[0]} to {theta[-1]}."
        )
    return theta


def union_theta_grid(
    grids: Sequence[NumericArrayLike], tol: float = 1e-6
) -> NDArray[np.float64]:
    """Merge scattering angle grids into the union of their nodes.

    A phase matrix tabulated on a grid is, for the kernels, the
    piecewise linear interpolant of its nodes: the random walk draws
    its deflection from exactly that interpolant, whatever the grid.
    A weighted sum of such matrices is piecewise linear on the union
    of their breakpoints, so resampling every component onto that
    union before mixing them reproduces each one exactly, whereas
    mixing on the grid of any single component resamples the others.
    This is how a mixture of components with different native grids,
    say an OPAC aerosol on 1801 equally spaced angles and a water
    cloud on 594 angles clustered in its diffraction peak, keeps every
    node of every table.

    Parameters
    ----------
    grids : sequence of array_like
        Scattering angle grids in degrees, each strictly increasing
        from 0 to 180, as :func:`as_theta_grid` accepts them.
    tol : float, optional
        Two nodes closer than *tol* degrees are one node, the smaller
        being kept. The default of 1e-6 merges the float32 angles of
        the OPAC files, where 0.1 is stored as 0.100000001, with the
        float64 angles of the cloud files, without merging any two
        distinct nodes: the finest step of the auxdata is 0.01
        degrees.

    Returns
    -------
    ndarray
        Strictly increasing angles in degrees, from 0 to 180, float64.

    Raises
    ------
    ValueError
        If *grids* is empty, or one of them is not a valid scattering
        angle grid.

    Examples
    --------
    >>> union_theta_grid([[0., 90., 180.], [0., 45., 90., 180.]])
    array([  0.,  45.,  90., 180.])
    >>> union_theta_grid([[0., np.float32(0.1), 180.], [0., 0.1, 180.]])
    array([0.0e+00, 1.0e-01, 1.8e+02])
    """
    if len(grids) == 0:
        raise ValueError(
            "At least one scattering angle grid is needed to build "
            "their union."
        )
    theta = np.sort(np.concatenate([as_theta_grid(g) for g in grids]))
    keep = np.concatenate([[True], np.diff(theta) > tol])
    return as_theta_grid(theta[keep])


def _grid_label(comp: object) -> str:
    """Name a component in a message about its scattering angle grid.

    By its class and the stem of the file it was read from, if any.
    """
    name = getattr(comp, "fname", None)
    cls = type(comp).__name__
    if name is None or str(name) == "none":
        return cls
    return f"{cls}({Path(name).stem})"


def _common_theta_grid(
    grids: Sequence[NDArray[np.floating]],
    labels: Sequence[str],
    warn: bool = True,
) -> tuple[NDArray[np.float64], bool]:
    """Return the angle grid a set of phase matrices is mixed on.

    When every matrix carries the same grid, that grid. Otherwise the
    union of the grids, on which mixing the matrices is exact (see
    :func:`union_theta_grid`), announced by a warning naming the
    components and the grids involved when `warn` is True, since the
    mixture then lives on a grid nobody asked for explicitly.

    Parameters
    ----------
    grids : sequence of ndarray
        The scattering angles of each phase matrix, in degrees.
    labels : sequence of str
        What to call each matrix in the warning, one per grid.
    warn : bool, optional
        Whether a union is announced by a warning. False where the
        union is what was asked for, as ``n_theta='native'`` asks.

    Returns
    -------
    theta : ndarray
        The common grid, in degrees.
    resampled : bool
        Whether the grids differ, i.e. whether *theta* is their union
        and the matrices have to be resampled onto it.
    """
    ref = np.asarray(grids[0], dtype=np.float64)
    if all(np.array_equal(g, ref) for g in grids[1:]):
        return ref, False

    theta = union_theta_grid(grids)
    if not warn:
        return theta, True
    # one entry per distinct grid, so that a long list of matrices
    # sharing a few grids stays readable
    seen: list[tuple[str, int]] = []
    for label, grid in zip(labels, grids, strict=True):
        entry = (label, len(grid))
        if entry not in seen:
            seen.append(entry)
    described = ", ".join(f"{label} ({n} angles)" for label, n in seen)
    warnings.warn(
        f"The components {described} carry different phase angle "
        "grids; their phase matrices are mixed on the union of those "
        f"grids ({len(theta)} angles).",
        stacklevel=3,
    )
    return theta, True


def integ_phase(
    ang: NDArray[np.floating[Any]],
    pha: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    r"""Numerically integrate a phase function weighted by sin(theta).

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
    """Map the phase function altitudes onto the model altitudes.

    For each level in *z_full*, find the phase function layer with the
    largest vertical overlap, optionally penalising layers whose phase
    function is identically zero.

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
    for i_full in range(nz_full):
        idz_full = (nz_full - 1) - i_full
        zmin_full = grid_full[idz_full]
        zmax_full = grid_full[idz_full] + size_layers_full[idz_full]

        # First find all the z_pf layers respecting the 2 conditions
        ida_tmp = []
        for i_pf in range(nz_pf):
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
            for k in range(n_ida_tmp):
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
    output_sg_ready: bool = True,
) -> xr.DataArray:
    """Read phase function data from a SMART-G netCDF file.

    Loads phase matrix data from SMART-G aerosol and cloud files with
    .nc suffix. Supports wavelength and humidity/effective radius
    interpolation and normalization, and returns either the phase
    matrix laid on a 1D profile, ready for the `phase` parameter of
    `AerOPAC` / `Cloud`, or the table on the axes of the file, ready
    for the 3D components (`output_sg_ready`).

    Parameters
    ----------
    fname : str or path-like
        Path to a SMART-G phase function NetCDF file (suffix: .nc).

        The file must include variables:
        - 'phase': phase matrix data [rh/reff, wavelength, stk, theta]
        - 'wav': wavelength values (in nm)
        - 'theta': scattering angle grid (in degrees)
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
        the ``Atm1D`` constructor. Only with `output_sg_ready` = True.
        Default: None

    pfgrid : array_like, optional
        Altitude grid [z_top, z_1, z_2, ..., z_bottom] (in km,
        descending order) for altitude-dependent phase functions. If
        provided with n_rh_reff > 1, the z_rh_reff values will be
        interpolated onto this grid. The first element (z_top) is
        skipped; remaining elements define the z_phase coordinate. This
        parameter has the same meaning as ``pfgrid`` in the ``Atm1D``
        constructor. Only with `output_sg_ready` = True.
        Default: None

    z_rh_reff : float or array_like, optional
        Interpolation target for the second phase-function axis:
        - aerosol files: relative humidity (%)
        - cloud files: effective radius (reff)
        Required if the file contains multiple rh/reff values
        (n_rh_reff > 1). If array_like (1-D), pfgrid must also be
        provided to map these values to specific altitudes, and
        ``len(z_rh_reff)`` must equal ``len(pfgrid) - 1``. Only with
        `output_sg_ready` = True.
        Default: None

    output_sg_ready : bool, optional
        Which of the two layouts to return.
        True (default): the phase matrix ready for the ``phase``
        argument of ``AerOPAC``, ``Cloud`` and ``Hydrosol`` and for
        ``Atm1D.prof_phases``: interpolated at ``wavelength_phase`` and
        ``z_rh_reff`` and laid out on the profile altitudes, with the
        dimensions ('wavelength_phase', 'z_phase', 'nphamat',
        'theta_' + kind).
        False: the table as the file carries it, with the dimensions
        ('wavelength_phase', 'hum' or 'reff', 'nphamat', 'theta_' +
        kind) on every wavelength and humidity/effective radius of the
        file. This is what the ``phase`` argument of the 3D components
        ``Cloud3D`` and ``Aer3D`` takes, which interpolate it per cell
        themselves; ``wavelength_phase``, ``pfgrid`` and ``z_rh_reff``
        must then be left to None.

    Returns
    -------
    da_pha : DataArray
        Phase matrix as xarray DataArray with dimensions:
        - 'wavelength_phase': wavelength (in nm)
        - 'z_phase': altitude (in km) from pfgrid or [0.], or with
          `output_sg_ready` = False 'hum' or 'reff' as in the file
        - 'nphamat': phase matrix unique terms (0 to nphamat-1), as
          many as the file carries: nphamat = 4 for spherical
          particles only, nphamat = 6 for spherical or non-spherical
          particles (for spherical: P22=P11, P44=P33). The components
          complete 4 terms into 6 themselves.
        - 'theta_'+kind: scattering angle (in degrees)

        With `output_sg_ready` = True the coordinates are replaced /
        renamed such that the rh/reff dimension becomes 'z_phase' with
        values from pfgrid[1:] or [0.] if pfgrid is None.

    Examples
    --------
    Read phase function for a single wavelength and rh:

    >>> pha = read_phase_nc(  # doctest: +SKIP
    ...     'desert_sol.nc', wavelength_phase=550.0,
    ...     z_rh_reff=[70.0, 60., 58.],
    ...     pfgrid=[100., 50., 10., 0.],
    ...     normalize=True)
    >>> pha.shape  # doctest: +SKIP
    (1, 3, 6, 1801)  # (wavelength_phase, z_phase, nphamat, theta_atm)

    Read the table of a water cloud for a 3D cloud:

    >>> pha = read_phase_nc('wc_sol.nc',  # doctest: +SKIP
    ...                     output_sg_ready=False)
    >>> pha.dims  # doctest: +SKIP
    ('wavelength_phase', 'reff', 'nphamat', 'theta_atm')
    """
    if not output_sg_ready:
        _reject_profile_targets(wavelength_phase, pfgrid, z_rh_reff)
    wavelength_phase, pfgrid, z_rh_reff = _profile_targets(
        wavelength_phase, pfgrid, z_rh_reff
    )

    ds = xr.open_dataset(fname)

    rh_reff, rh_or_reff = _phase_cdf_rh_or_reff(ds)

    n_rh_reff = rh_reff.size
    n_wavelength = ds.sizes["wav"]
    theta = ds.theta.values
    wavelength = ds.wav.values

    # Get nphamat from phase data shape
    nphamat = ds["phase"].shape[2]

    if output_sg_ready:
        _check_profile_targets(
            n_wavelength, n_rh_reff, rh_or_reff,
            wavelength_phase, pfgrid, z_rh_reff,
        )

    da_pha = xr.DataArray(
        ds["phase"].values.swapaxes(0, 1).astype(np.float64),
        coords=[wavelength, rh_reff, np.arange(nphamat), theta],
        dims=["wavelength_phase", rh_or_reff, "nphamat", "theta_" + kind],
        name="phase_" + kind,
    )

    if normalize:
        _normalize_p11(da_pha.data, theta)

    if not output_sg_ready:
        return da_pha

    return _to_profile_layout(
        da_pha, rh_or_reff, wavelength_phase, pfgrid, z_rh_reff
    )


def read_phase_dat(
    fname: PathType,
    kind: str = "atm",
    normalize: bool = True,
) -> xr.DataArray:
    """
    Read a phase matrix from a space-separated `.dat` file.

    The file is expected to have no header. The first column contains
    the scattering angles (in degrees), in either order, and the
    remaining columns contain the phase matrix elements (one column per
    element). The phase matrix is assumed to be monochromatic and
    vertically uniform (no wavelength or altitude dependence).

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
        - ``'theta_' + kind`` : scattering angle in degrees, increasing

    Examples
    --------
    >>> pha = read_phase_dat('phase.dat', kind='atm',  # doctest: +SKIP
    ...                      normalize=True)
    >>> pha.dims  # doctest: +SKIP
    ('wavelength_phase', 'z_phase', 'nphamat', 'theta_atm')
    """
    df = pd.read_csv(fname, sep=r"\s+", header=None)

    theta = np.asarray(df.iloc[:, 0].values)
    pha = np.asarray(df.iloc[:, 1:].values)
    pha = pha.swapaxes(0, 1)
    # the angles come back increasing, as from the other readers,
    # whatever the order of the file
    order = np.argsort(theta, kind="stable")
    theta, pha = theta[order], pha[:, order]

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
    # ntheta (wavelength, rh/reff, nphamat)
    ntheta = ds["ntheta"][:, :, :].data
    # theta (wavelength, rh/reff, nphamat, ntheta)
    theta_file = ds["theta"][:, :, :, :].data
    n_wavelength, n_rh_reff, n_stk = phase.shape[:3]

    data = np.zeros((n_wavelength, n_rh_reff, n_stk, theta.size))
    for i_wavelength in range(n_wavelength):
        for irhreff in range(n_rh_reff):
            for istk in range(n_stk):
                nth = ntheta[i_wavelength, irhreff, istk]
                th = theta_file[i_wavelength, irhreff, istk, :]

                data[i_wavelength, irhreff, istk, :] = np.interp(
                    theta,
                    th[:nth],
                    phase[i_wavelength, irhreff, istk, :nth],
                    period=np.inf,
                )
    return data


def _reject_profile_targets(
    wavelength_phase: NumericArrayLike | None,
    pfgrid: NumericArrayLike | None,
    z_rh_reff: NumericArrayLike | None,
) -> None:
    """Refuse the profile-ready targets with the table output.

    ``wavelength_phase``, ``pfgrid`` and ``z_rh_reff`` say where to
    interpolate the table and how to lay it on the profile altitudes,
    which only the profile-ready output does.
    """
    given = [
        name
        for name, value in (
            ("wavelength_phase", wavelength_phase),
            ("pfgrid", pfgrid),
            ("z_rh_reff", z_rh_reff),
        )
        if value is not None
    ]
    if given:
        raise ValueError(
            f"The {', '.join(given)} parameter(s) describe the "
            "profile-ready output (output_sg_ready=True); leave them to "
            "None to get the table on the axes of the file "
            "(output_sg_ready=False)."
        )


def _profile_targets(
    wavelength_phase: NumericArrayLike | None,
    pfgrid: NumericArrayLike | None,
    z_rh_reff: NumericArrayLike | None,
) -> tuple[
    NDArray[np.float32] | None,
    NDArray[np.float32] | None,
    NDArray[np.float32] | None,
]:
    """Return the interpolation targets as 1-D float32 arrays.

    A scalar is made 1-D so that the interpolation keeps its
    dimension and the output stays 4-D.
    """
    return (
        None if wavelength_phase is None
        else np.atleast_1d(np.asarray(wavelength_phase, dtype=np.float32)),
        None if pfgrid is None
        else np.atleast_1d(np.asarray(pfgrid, dtype=np.float32)),
        None if z_rh_reff is None
        else np.atleast_1d(np.asarray(z_rh_reff, dtype=np.float32)),
    )


def _cdf_theta_grid(
    ds: xr.Dataset, n_theta: ThetaLike | None, n_theta_max: int
) -> NDArray[np.float64]:
    """Return the scattering angles a cdf file is resampled on.

    Parameters
    ----------
    ds : Dataset
        The open file, with the variables 'theta' and 'ntheta'.
    n_theta : None, int, str or array_like
        ``None`` for an equally spaced grid fine enough for the finest
        step of the file, capped at *n_theta_max* angles; a number of
        equally spaced angles or the angles themselves in degrees; or
        ``'native'`` for the union of every grid the file carries.
    n_theta_max : int
        The cap of the automatic grid.

    Returns
    -------
    ndarray
        Strictly increasing angles in degrees, from 0 to 180.
    """
    if n_theta is None:
        dtheta_min = np.nanmin(np.abs(np.diff(ds.theta.values, axis=3)))
        ntheta = np.ceil(180 / dtheta_min).astype(int) + 1
        ntheta = min(ntheta, n_theta_max)  # be sure to not exceed n_theta_max
        return np.linspace(0, 180, ntheta)

    if is_native_theta(n_theta):
        # one grid per (wavelength, rh/reff, term), stored descending
        # and padded past ntheta; identical grids are merged first
        theta_all = ds["theta"].values
        ntheta_all = ds["ntheta"].values
        grids: list[NDArray[np.float64]] = []
        for idx in np.ndindex(ntheta_all.shape):
            grid = np.sort(
                theta_all[idx][: int(ntheta_all[idx])].astype(np.float64)
            )
            if not any(np.array_equal(grid, g) for g in grids):
                grids.append(grid)
        return union_theta_grid(grids)

    return as_theta_grid(n_theta)


def _normalize_p11(
    data: NDArray[np.floating[Any]], theta: NDArray[np.floating[Any]]
) -> None:
    """Scale the phase matrices in place to the F11 normalization.

    F11 is made to integrate to 2 over ``cos(theta)``, for every
    (wavelength, rh/reff) entry of a
    ``(wavelength, rh/reff, nphamat, theta)`` array.
    """
    mu = np.cos(np.deg2rad(theta))
    idmu = np.argsort(mu)
    for i_wavelength in range(data.shape[0]):
        for irhreff in range(data.shape[1]):
            f = data[i_wavelength, irhreff, 0, :]  # P11 term
            norm = np.trapezoid(f[idmu], mu[idmu])
            data[i_wavelength, irhreff, :, :] *= 2.0 / abs(norm)


def _in_file_range(
    values: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
    name: str,
    axis: str,
    unit: str = "",
) -> NDArray[np.float64]:
    """Return the targets of an axis, checked against the file range.

    Parameters
    ----------
    values : ndarray
        Values of the axis in the phase file.
    targets : ndarray
        Values to interpolate the phase matrix at.
    name : str
        Name of the argument giving the targets, for the message.
    axis : str
        Name of the axis, for the message.
    unit : str, optional
        Unit of the axis, for the message.

    Returns
    -------
    ndarray
        The targets in float64, those within a relative 1e-6 of the
        range of the file, a float32 rounding, moved onto it.

    Raises
    ------
    ValueError
        If a target is outside the range of the file, where the
        interpolation would give NaN or a bounds error.
    """
    lo = float(np.min(values))
    hi = float(np.max(values))
    tol = 1e-6 * max(abs(lo), abs(hi))
    targets = np.asarray(targets, dtype=np.float64)
    outside = targets[(targets < lo - tol) | (targets > hi + tol)]
    if outside.size:
        raise ValueError(
            f"{name} {outside.tolist()} is outside the {axis} range of "
            f"the phase file, {lo:g} to {hi:g}{unit}."
        )
    return np.clip(targets, lo, hi)


def _wavelength_in_file(
    wavelength_file: NDArray[np.floating[Any]],
    wavelength_phase: NDArray[np.float32],
) -> NDArray[np.float64]:
    """Return the wavelength targets, checked against the file range.

    See `_in_file_range`, whose `values` and `targets` are here
    `wavelength_file` and `wavelength_phase`, in nm.
    """
    return _in_file_range(
        wavelength_file, wavelength_phase, "wavelength_phase",
        "wavelength", " nm",
    )


def _bracketing_indices(
    values: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]] | None,
) -> NDArray[np.intp]:
    """Return the indices of an axis needed to interpolate at targets.

    They are those of the values from the largest one below the
    smallest target to the smallest one above the largest target, so
    that a linear interpolation picks the same two values as on the
    whole axis, even at a target equal to a value. All of them without
    targets.
    """
    if targets is None:
        return np.arange(values.size)
    below = values[values < np.min(targets)]
    above = values[values > np.max(targets)]
    lo = np.max(below) if below.size else np.min(values)
    hi = np.min(above) if above.size else np.max(values)
    return np.flatnonzero((values >= lo) & (values <= hi))


def _to_profile_layout(
    da_pha: xr.DataArray,
    rh_or_reff: str,
    wavelength_phase: NDArray[np.float32] | None,
    pfgrid: NDArray[np.float32] | None,
    z_rh_reff: NDArray[np.floating[Any]] | None,
) -> xr.DataArray:
    """Lay a phase table on the profile.

    The table is interpolated at the target wavelengths and rh/reff
    values, and its rh/reff axis is renamed into the ``z_phase``
    altitudes of *pfgrid*, or ``[0.]`` without one.
    """
    if da_pha.sizes["wavelength_phase"] > 1 and wavelength_phase is not None:
        targets = _wavelength_in_file(
            da_pha["wavelength_phase"].values, wavelength_phase
        )
        da_pha = da_pha.interp(wavelength_phase=targets).assign_coords(
            wavelength_phase=wavelength_phase
        )

    if da_pha.sizes[rh_or_reff] > 1:
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
    return da_pha.assign_coords({rh_or_reff: z_phase}).rename(
        {rh_or_reff: "z_phase"}
    )


def _check_profile_targets(
    n_wavelength: int,
    n_rh_reff: int,
    rh_or_reff: str,
    wavelength_phase: NDArray[np.float32] | None,
    pfgrid: NDArray[np.float32] | None,
    z_rh_reff: NDArray[np.float32] | None,
) -> None:
    """Check that the targets the file needs were given.

    The check runs before any computation, and concerns the files with
    several wavelengths or rh/reff values.
    """
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
    if (n_rh_reff > 1 and z_rh_reff is not None and pfgrid is not None
            and z_rh_reff.size != pfgrid.size - 1):
        raise ValueError(
            "Invalid 'z_rh_reff' size: when 'z_rh_reff' is a 1-D array, "
            "its size must be len(pfgrid) - 1. "
            f"Got len(z_rh_reff)={z_rh_reff.size}"
            f" and len(pfgrid)={pfgrid.size}."
        )


def read_phase_cdf(
    fname: PathType,
    kind: str = "atm",
    normalize: bool = True,
    n_theta: ThetaLike | None = None,
    n_theta_max: int = 18001,
    wavelength_phase: NumericArrayLike | None = None,
    pfgrid: NumericArrayLike | None = None,
    z_rh_reff: NumericArrayLike | None = None,
    output_sg_ready: bool = True,
) -> xr.DataArray:
    """Read phase function data from a libRadtran netCDF file.

    Loads phase matrix data from libRadtran aerosol and cloud phase
    function files with .cdf suffix (e.g., 'ssam.mie.cdf',
    'wc.sol.mie.cdf'), the monochromatic IPRT files included. Their
    theta grids vary with the wavelength, the rh/reff value and the
    matrix term; every entry is resampled onto one scattering angle
    grid, chosen with `n_theta`. Supports wavelength and humidity /
    effective radius interpolation and normalization, and returns
    either the phase matrix laid on a 1D profile, ready for the
    `phase` parameter of `AerOPAC` / `Cloud`, or the table on the
    axes of the file, ready for the 3D components (`output_sg_ready`).

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
        - 'nphamat': number of Stokes matrix elements (4 or 6)
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
        Default: True

    n_theta : None, int, str or array_like, optional
        The scattering angles the phase matrices are resampled on:
        - None -> equally spaced angles, as many as the finest step of
          the file needs, at most `n_theta_max` (default)
        - an int -> that many equally spaced angles
        - an array -> the angles themselves in degrees, which
          :func:`theta_grid` can build clustered towards the forward
          and backward directions
        - 'native' -> the union of every grid the file carries, on
          which the file is reproduced exactly, since the kernels
          sample the piecewise linear interpolant of the table (see
          :func:`union_theta_grid`). Its size grows with the number of
          distinct grids of the file (2818 angles for the 25 radii of
          the IPRT ``watercloud_670.mie.cdf``, 38 for
          ``waso_670.mie.cdf``), each angle costing 28 bytes per phase
          function on the device.

    n_theta_max : int, optional
        Cap of the automatic grid (`n_theta` = None): if the file
        provides higher resolution, it will be reduced to this limit.
        Default: 18001

    wavelength_phase : float or array_like, optional
        Wavelength(s) (in nm) to interpolate to. Required if the file
        contains multiple wavelengths (n_wavelength > 1). This
        parameter has the same meaning as ``wavelength_phase`` in the
        ``Atm1D`` constructor. Only with `output_sg_ready` = True.
        Default: None

    pfgrid : array_like, optional
        Altitude grid [z_top, z_1, z_2, ..., z_bottom] (in km,
        descending order) for altitude-dependent phase functions. If
        provided with n_rh_reff > 1, the z_rh_reff values will be
        interpolated onto this grid. The first element (z_top) is
        skipped; remaining elements define the z_phase coordinate.
        This parameter has the same meaning as ``pfgrid`` in the
        ``Atm1D`` constructor. Only with `output_sg_ready` = True.
        Default: None

    z_rh_reff : float or array_like, optional
        Interpolation target for the second phase-function axis:
        - aerosol files: relative humidity (%)
        - cloud files: effective radius (reff)
        Required if the file contains multiple rh/reff values
        (n_rh_reff > 1). If array-like (1-D), pfgrid must also be
        provided to map these values to specific altitudes, and
        ``len(z_rh_reff)`` must equal ``len(pfgrid) - 1``. Only with
        `output_sg_ready` = True.
        Default: None

    output_sg_ready : bool, optional
        Which of the two layouts to return.
        True (default): the phase matrix ready for the ``phase``
        argument of ``AerOPAC``, ``Cloud`` and ``Hydrosol`` and for
        ``Atm1D.prof_phases``: interpolated at ``wavelength_phase`` and
        ``z_rh_reff`` and laid out on the profile altitudes, with the
        dimensions ('wavelength_phase', 'z_phase', 'nphamat',
        'theta_' + kind).
        False: the table as the file carries it, resampled onto the
        requested angles only, with the dimensions ('wavelength_phase',
        'hum' or 'reff', 'nphamat', 'theta_' + kind) on every
        wavelength and humidity/effective radius of the file. This is
        what the ``phase`` argument of the 3D components ``Cloud3D`` and
        ``Aer3D`` takes, which interpolate it per cell themselves;
        ``wavelength_phase``, ``pfgrid`` and ``z_rh_reff`` must then be
        left to None. The second axis is the file's: an aerosol file
        tabulated against an effective radius comes back on ``reff``,
        which ``Aer3D`` (humidity) does not take.

    Returns
    -------
    da_pha : DataArray
        Phase matrix as xarray DataArray with dimensions:
        - 'wavelength_phase': wavelength (in nm)
        - 'z_phase': altitude (in km) from pfgrid or [0.], or with
          `output_sg_ready` = False 'hum' or 'reff' as in the file
        - 'nphamat': phase matrix unique terms (0 to nphamat-1), as
          many as the file carries: nphamat = 4 for spherical
          particles only, nphamat = 6 for spherical or non-spherical
          particles (for spherical: P22=P11, P44=P33). The components
          complete 4 terms into 6 themselves.
        - 'theta_'+kind: scattering angle (in degrees)

        With `output_sg_ready` = True the coordinates are replaced /
        renamed such that the rh/reff dimension becomes 'z_phase' with
        values from pfgrid[1:] or [0.] if pfgrid is None.

    Examples
    --------
    Read phase function for a single wavelength and rh:

    >>> pha = read_phase_cdf(  # doctest: +SKIP
    ...     'ssam.mie.cdf', wavelength_phase=550.0,
    ...     z_rh_reff=[70.0, 60., 58.],
    ...     pfgrid=[100., 50., 10., 0.],
    ...     normalize=True)
    >>> pha.shape  # doctest: +SKIP
    (1, 3, 4, 18001)  # (wavelength_phase, z_phase, nphamat, theta_atm)

    The 4 terms of these spherical particles are completed to 6 by the
    components, or by `expand_phase_4_to_6`.

    Read the table of an IPRT water cloud for a 3D cloud, on the
    angles of the file:

    >>> pha = read_phase_cdf(  # doctest: +SKIP
    ...     'watercloud_670.mie.cdf', n_theta='native',
    ...     normalize=False, output_sg_ready=False)
    >>> pha.dims  # doctest: +SKIP
    ('wavelength_phase', 'reff', 'nphamat', 'theta_atm')
    """
    if not output_sg_ready:
        _reject_profile_targets(wavelength_phase, pfgrid, z_rh_reff)
    wavelength_phase, pfgrid, z_rh_reff = _profile_targets(
        wavelength_phase, pfgrid, z_rh_reff
    )

    ds = xr.open_dataset(fname)

    rh_reff, rh_or_reff = _phase_cdf_rh_or_reff(ds)

    nphamat = ds.nphamat.size
    if nphamat not in (4, 6):
        raise ValueError(
            "The number of phase matrix terms in the file must be "
            f"equal to 4 or 6, got {nphamat}."
        )
    n_rh_reff = rh_reff.size
    n_wavelength = ds["wavelen"].size
    theta = _cdf_theta_grid(ds, n_theta, n_theta_max)
    wavelength = ds["wavelen"].data * 1e3

    # checks at the beginning to avoid unnecessary computations
    if output_sg_ready:
        _check_profile_targets(
            n_wavelength, n_rh_reff, rh_or_reff,
            wavelength_phase, pfgrid, z_rh_reff,
        )
        # only the entries around the targets are resampled; the
        # automatic and native angle grids are those of the whole file
        index_wavelength = np.arange(n_wavelength)
        if n_wavelength > 1 and wavelength_phase is not None:
            index_wavelength = _bracketing_indices(
                wavelength,
                _wavelength_in_file(wavelength, wavelength_phase),
            )
        index_rh_reff = np.arange(n_rh_reff)
        if n_rh_reff > 1 and z_rh_reff is not None:
            # checked here, as only the entries around the targets are
            # kept: beyond the axis, a single entry would be kept and
            # used without interpolation
            z_rh_reff = _in_file_range(
                rh_reff, z_rh_reff, "z_rh_reff", rh_or_reff
            )
            index_rh_reff = _bracketing_indices(rh_reff, z_rh_reff)
        ds = ds.isel({
            ds["wavelen"].dims[0]: index_wavelength,
            ds[rh_or_reff].dims[0]: index_rh_reff,
        })
        rh_reff = rh_reff[index_rh_reff]
        wavelength = wavelength[index_wavelength]

    da_pha = xr.DataArray(
        _resample_cdf_phase(ds, theta),
        coords=[wavelength, rh_reff, np.arange(nphamat), theta],
        dims=["wavelength_phase", rh_or_reff, "nphamat", "theta_" + kind],
        name="phase_" + kind,
    )

    if normalize:
        _normalize_p11(da_pha.data, theta)

    if not output_sg_ready:
        return da_pha

    return _to_profile_layout(
        da_pha, rh_or_reff, wavelength_phase, pfgrid, z_rh_reff
    )


def read_phase(
    fname: PathType,
    kind: str = "atm",
    normalize: bool = True,
    output_sg_ready: bool = True,
    **kwargs: Any,
) -> xr.DataArray:
    """Read phase function data, dispatching to the proper reader.

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

    output_sg_ready : bool, optional
        Which of the two layouts to return.
        True (default): the phase matrix ready for the ``phase``
        argument of ``AerOPAC``, ``Cloud`` and ``Hydrosol`` and for
        ``Atm1D.prof_phases``: interpolated at ``wavelength_phase`` and
        ``z_rh_reff`` and laid out on the profile altitudes, with the
        dimensions ('wavelength_phase', 'z_phase', 'nphamat',
        'theta_' + kind).
        False: the table as the file carries it, with the dimensions
        ('wavelength_phase', 'hum' or 'reff', 'nphamat', 'theta_' +
        kind) on every wavelength and humidity/effective radius of the
        file. This is what the ``phase`` argument of the 3D components
        ``Cloud3D`` and ``Aer3D`` takes; ``wavelength_phase``,
        ``pfgrid`` and ``z_rh_reff`` must then be left to None. Only
        for ``.nc`` and ``.cdf`` files: a ``.dat`` file carries a single
        matrix, with no wavelength or humidity axis.

    **kwargs : dict, optional
        Additional keyword arguments forwarded to the selected backend
        reader:

        - for ``.nc``: forwarded to :func:`read_phase_nc`
        - for ``.cdf``: forwarded to :func:`read_phase_cdf`

        Typical arguments include ``wavelength_phase``, ``pfgrid``,
        ``z_rh_reff``, and ``n_theta`` and ``n_theta_max`` (only for
        ``.cdf``).

    Returns
    -------
    DataArray
        Phase matrix data as returned by the selected backend reader.
        All backends return a 4-dimensional array with dimensions:

        - ``('wavelength_phase', 'z_phase', 'nphamat',
          'theta_' + kind)``, or ``('wavelength_phase', 'hum' or
          'reff', 'nphamat', 'theta_' + kind)`` with
          ``output_sg_ready=False``

        where 'nphamat' has size nphamat:
        - nphamat = 4 for spherical particles only
        - nphamat = 6 for spherical or non-spherical particles
          (for spherical: P22=P11, P44=P33)

    Raises
    ------
    ValueError
        If the file format is not supported, or if ``output_sg_ready``
        is False for a ``.dat`` file.

    Examples
    --------
    >>> pha = read_phase('phase.dat', kind='atm',  # doctest: +SKIP
    ...                  normalize=True)
    >>> pha = read_phase('desert_sol.nc', kind='atm',  # doctest: +SKIP
    ...                  wavelength_phase=550.0,
    ...                  z_rh_reff=[70.0, 60.0, 58.0],
    ...                  pfgrid=[100.0, 50.0, 10.0, 0.0])
    >>> pha = read_phase('ssam.mie.cdf', kind='atm',  # doctest: +SKIP
    ...                  wavelength_phase=550.0,
    ...                  z_rh_reff=[70.0, 60.0, 58.0],
    ...                  pfgrid=[100.0, 50.0, 10.0, 0.0],
    ...                  n_theta_max=18001)
    >>> pha = read_phase('wc_sol.nc',  # doctest: +SKIP
    ...                  output_sg_ready=False)
    """
    fname = Path(fname)

    if not fname.is_file():
        raise FileNotFoundError(f"Phase function file not found: {fname}")

    supported_formats = [".dat", ".nc", ".cdf"]

    if fname.suffix == ".dat":
        if not output_sg_ready:
            raise ValueError(
                "A .dat file carries a single phase matrix, with no "
                "wavelength or humidity/effective radius axis: only "
                "the profile-ready output (output_sg_ready=True) "
                "exists for it."
            )
        return read_phase_dat(fname, kind=kind, normalize=normalize)
    elif fname.suffix == ".nc":
        return read_phase_nc(
            fname, kind=kind, normalize=normalize,
            output_sg_ready=output_sg_ready, **kwargs
        )
    elif fname.suffix == ".cdf":
        return read_phase_cdf(
            fname, kind=kind, normalize=normalize,
            output_sg_ready=output_sg_ready, **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported phase function file format: "
            f"{fname.suffix}. Supported formats: {supported_formats}"
        )


def _check_finite_phase(
    phase: xr.DataArray | LUT | NDArray[np.floating[Any]],
    owner: str,
    theta: NDArray[np.floating[Any]] | None = None,
) -> None:
    """Raise if phase matrices hold NaN or infinite values.

    The kernel samples the scattering angles from the tables and weighs
    the local estimates with them: a NaN there does not stop a run, it
    biases it. A table aligned on the angles of another one, as
    ``xr.concat`` or ``xr.merge`` do with ``join='outer'``, gets NaN
    wherever its own angles miss one of the other's.

    Parameters
    ----------
    phase : DataArray or LUT or ndarray
        The phase matrices, of any shape.
    owner : str
        What the matrices belong to, for the message, e.g.
        ``"the Hydrosol"``.
    theta : ndarray, optional
        The scattering angles in degrees along the last axis of an
        ndarray `phase`, to name the angles concerned. A DataArray
        carries its own, on its ``theta*`` dimension.

    Raises
    ------
    ValueError
        If a value of `phase` is NaN or infinite.
    """
    if isinstance(phase, LUT):
        phase = phase.to_xarray()
    if isinstance(phase, xr.DataArray):
        values = np.asarray(phase.values, dtype=np.float64)
        dim = next(
            (d for d in phase.dims if str(d).startswith("theta")), None
        )
        if dim is not None:
            theta = phase[dim].values
            values = np.moveaxis(values, phase.dims.index(dim), -1)
        else:
            theta = None
    else:
        values = np.asarray(phase, dtype=np.float64)
    bad = ~np.isfinite(values)
    if not bad.any():
        return
    where = ""
    if theta is not None and np.shape(theta) == values.shape[-1:]:
        angles = np.asarray(theta)[bad.reshape(-1, bad.shape[-1]).any(0)]
        shown = ", ".join(f"{a:g}" for a in angles[:5])
        more = ", ..." if len(angles) > 5 else ""
        where = f", at the scattering angles {shown}{more} degrees"
    raise ValueError(
        f"The phase matrices of {owner} hold {int(bad.sum())} values "
        f"that are NaN or infinite{where}. A table aligned on the "
        "angles of another one, as xr.concat or xr.merge do with "
        "join='outer', gets NaN where its own angles miss one of the "
        "other's."
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
    """Convert a phase matrix to the Ipar/Iper convention.

    The phase matrix goes from the standard IQUV (Stokes vector)
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
