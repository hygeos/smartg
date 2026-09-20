"""Preprocessing of atmospheric optical properties for SMART-G
simulations.

This module provides tools to build and preprocess atmospheric profiles
for use
as input to SMART-G radiative transfer simulations. It implements
various
atmospheric models and aerosol/cloud properties that can be combined
into
complete atmospheric profiles ready for simulation.

Workflow
--------
Typical usage involves:
1. Create an atmospheric profile using model classes (e.g., Atm1D)
2. Add atmospheric components (aerosols, clouds, surface) as needed
3. (Optional) Call the profile's `calc()` method to compute optical
   properties
   with specific parameters (if using optional parameters not set by
   default)
4. Pass the resulting profile object as the `atmosphere` parameter to
   `smartg.run()`

Key Classes
-----------
Atm1D
    1D atmospheric profile model. Provides vertical temperature and
    pressure profiles read from an auxiliary data file (AFGL standard
    atmospheres are shipped by default, but any other or user-provided
    profile can be used). Aerosols, clouds, and ocean surface can be
    added to build a complete atmospheric model.

Atm3D
    3D atmospheric profile model (for Smartg(opt3d=True) simulations).
    Combines a 1D background atmosphere (Atm1D), a 3D grid
    (smartg.grid3d.Grid3D) and 3D components (Cloud3D, Aer3D) into the
    3D profile consumed by smartg.run().

Cloud3D
    3D cloud component of Atm3D. The 3D distribution of the cloud
    extinction and droplet effective radius is provided as a dense
    xarray dataset (or NetCDF file), as raw arrays, or converted from
    the legacy I3RC/IPRT ASCII cloud files with read_i3rc_cloud.

Aer3D
    3D aerosol component of Atm3D. Bulk optical properties from the
    OPAC aerosol mixtures or species as a function of the relative
    humidity; the 3D distribution of the aerosol extinction and
    relative humidity is provided as a dense xarray dataset (or NetCDF
    file), as raw arrays, or converted from I3RC/IPRT-style ASCII
    files with read_i3rc_aerosol.

AerOPAC
    Aerosol Optical Properties from OPAC (Optical Properties of Aerosols
    and Clouds) database. Computes aerosol optical depth, single
    scattering
    albedo, and phase matrices for aerosol mixtures.

Cloud
    Cloud optical properties model. Similar to AerOPAC, provides cloud
    optical
    depth, single scattering albedo, and phase matrices. Used for
    representing
    cloud layers in atmospheric profiles.

AerUser
    User-defined aerosol model. Like AerOPAC, but the aerosol optical
    depth, single scattering albedo, and phase matrices are supplied
    directly by the user instead of being interpolated from the OPAC
    pre-calculated aerosol netcdf files.
"""

from __future__ import annotations

import copy
import warnings
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence, TYPE_CHECKING
from smartg.phase import (
    as_theta_grid, calc_iphase, expand_phase_4_to_6, is_native_theta,
    union_theta_grid,
)
from scipy.interpolate import make_interp_spline
from scipy.integrate import simpson
from scipy import constants
from scipy.constants import speed_of_light, Planck, Boltzmann
from smartg.bandset import BandSet
from smartg.config import DIR_AUXDATA
from gatiab import vec_float_indexing
from smartg.truncation import DMTrunc, GTTrunc
import pandas as pd
import xarray as xr
import re
from pytrunc.truncation import delta_m_phase_approx, gt_phase_approx
from smartg.typing import (
    NumericArrayLike, PathType, RealNumber, ThetaLike
)
from smartg.diff import diff1
from numpy.typing import NDArray
from typing import Any, cast
from luts.luts import LUT
from smartg.grid3d import Grid3D, create_1d_grid

if TYPE_CHECKING:
    # Imported only for type checking to avoid a circular import
    # (kdis.py imports od2k/blackbody_radiance from atmosphere.py).
    from smartg.kdis import KdisIband


# constants
M_H2O = 18.015  # g/mol


def _grid_label(comp: object) -> str:
    """Name a component in a message about its scattering angle grid,
    by its class and the stem of the file it was read from, if any.
    """
    name = getattr(comp, "fname", None)
    cls = type(comp).__name__
    if name is None or str(name) == "none":
        return cls
    return f"{cls}({Path(name).stem})"


def _common_theta_grid(
    grids: Sequence[NDArray[np.floating]], labels: Sequence[str]
) -> tuple[NDArray[np.float64], bool]:
    """The scattering angle grid a set of phase matrices is mixed on.

    When every matrix carries the same grid, that grid. Otherwise the
    union of the grids, on which mixing the matrices is exact (see
    :func:`smartg.phase.union_theta_grid`), announced by a warning
    naming the components and the grids involved, since the mixture
    then lives on a grid nobody asked for explicitly.

    Parameters
    ----------
    grids : sequence of ndarray
        The scattering angles of each phase matrix, in degrees.
    labels : sequence of str
        What to call each matrix in the warning, one per grid.

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


def _on_theta_grid(
    pha: xr.DataArray, theta: NDArray[np.float64], dim: str = "theta_atm"
) -> xr.DataArray:
    """A phase matrix on the scattering angle grid *theta*, resampled
    linearly unless it is already there.
    """
    if np.array_equal(pha.coords[dim].values, theta):
        return pha
    return pha.interp({dim: theta})


class AerOPAC(object):
    """
    Initialize the Aerosol OPAC model

    Parameters
    ----------
    fname : str or path-like
        Complete path to the aerosol file or fname for aerosols
        located in "auxdata/aerosols/OPAC/mixtures/".
        Available auxdata aerosols: antarctic, antarctic_spheric,
        arctic, continental_average,
        continental_clean, continental_polluted, desert, desert_spheric,
        maritime_clean,
        maritime_polluted, mineral_transported, maritime_tropical and
        urban
    tau_ref : float or array_like or DataArray or LUT or None
        Optical thickness at reference wavelength w_ref
    w_ref : float
        Wavelength in nanometers at reference optical depth tau_ref
    h_min_mix : float, optional
        Force min altitude of the mixture
    h_mix_max : float, optional
        Force max altitude of the mixture
    h_free_min : float, optional
        Force min altitude of the free troposphere
    h_free_max : float, optional
        Force max altitude of the free troposphere
    h_stra_min : float, optional
        Force min altitude of the stratosphere
    h_stra_max : float, optional
        Force max altitude of the stratosphere
    z_mix : float, optional
        Force scale height (see notes) of the mixture
    z_free : float, optional
        Force scale height (see notes) of the free troposphere
    z_stra : float, optional
        Force scale height (see notes) of the stratosphere
    ssa : array_like or DataArray or None, optional
        Force particle single scattering albedo. Default None.

        - if float -> same value for all wavelengths and altitudes
        - if sequence of int or float -> it will be converted into a 1-D
          ndarray.
        - if 1-D ndarray -> only wavelength dependence is considered
        - if 2-D ndarray -> wavelength and altitude dependence is
          considered
        - if DataArray -> wavelength and altitude dependence is
          considered

        Note that DataArray is more flexible since it allows
        interpolation if wavelengths
        in calc method are different (but not the case for the altitude
        axis).
    phase : None or DataArray, optional
        Phase matrix F as function of wavelength, altitude, stoke
        components and scattering angle
        The variable names must be:
        If 4-D matrix -> wavelength_phase, z_phase, nphamat, theta
        If 2-D matrix (assumed monochromatic and constant vertically) ->
        nphamat, theta
        Where:
        - wavelength_phase is the wavelength. It must be equal to
          the `wavelength_phase` parameter of Atm1D if defined, else
          the `wavelength` parameter wavelengths of the Atm1D calc
          method.
        - z_phase is the phase altitude. It must be equal to the
          `pfgrid[1:]` parameter
          of Atm1D
        - nphamat the phase matrix unique terms.
        - theta the scattering angle.

        The phase matrix terms (IQUV convention) must be given in the
        folowing order:
        - F11, F21, F33 and F34 if only 4 terms are given (only for
          spherical particles)
        - F11, F21, F33, F34, F22 and F44 if 6 terms are given (for both
          spherical and non-spherical particles)
    rh_mix/free/stra : None or float, optional
        Force relative humidity of mixture/free tropo/strato. Default
        None.

    Notes
    -----
    The scale height (see [1]) is the variable z in the
    following equation:

    - :math:`N(h) = N(0)exp(-h/z)`

    with N the number density and h the altitude

    References
    ----------
    .. [1] M. Hess, P. Koepke, and I. Schult,
       "Optical Properties of Aerosols and Clouds: The Software
       Package OPAC," _Bulletin of the American Meteorological
       Society_, vol. 79, no. 5, pp. 831-844, 1998.
       doi:10.1175/1520-0477(1998)079<0831:OPOAAC>2.0.CO;2.

    Examples
    --------
    >>> from smartg.atmosphere import AerOPAC
    >>> aer_mc = AerOPAC('maritime_clean', 0.1, 550.)
    >>> print(aer_mc.ds_mix)
    <xarray.Dataset> Size: 6MB
    Dimensions:  (hum: 8, wav: 26, stk: 4, theta: 1801)
    Coordinates:
    * hum      (hum) float32 32B 0.0 50.0 70.0 80.0 90.0 95.0 98.0 99.0
    * wav      (wav) float32 104B 250.0 300.0 350.0 ... 3.75e+03 4e+03
                                  4.5e+03
    * theta    (theta) float32 7kB 0.0 0.1 0.2 0.3 0.4 ... 179.7 179.8
                                   179.9 180.0
    Dimensions without coordinates: stk
    Data variables:
        ext      (hum, wav) float32 832B ...
        ssa      (hum, wav) float32 832B ...
        phase    (hum, wav, stk, theta) float32 6MB ...
    Attributes:
        name:        maritime_clean
        H_min_mix:   0
        H_mix_max:   2
        H_free_min:  2
        H_free_max:  12
        H_stra_min:  12
        H_stra_max:  35
        Z_mix:       1
        Z_free:      8
        Z_stra:      99
        date:        2025-06-03
        source:      Created by HYGEOS using MOPSMAP v1.0.
    """

    def __init__(
        self,
        fname: str | Path,
        tau_ref: float | NumericArrayLike | xr.DataArray | LUT | None,
        w_ref: float,
        h_min_mix: float | None = None,
        h_mix_max: float | None = None,
        h_free_min: float | None = None,
        h_free_max: float | None = None,
        h_stra_min: float | None = None,
        h_stra_max: float | None = None,
        z_mix: float | None = None,
        z_free: float | None = None,
        z_stra: float | None = None,
        ssa: NumericArrayLike | xr.DataArray | None = None,
        phase: xr.DataArray | LUT | None = None,
        rh_mix: float | None = None,
        rh_free: float | None = None,
        rh_stra: float | None = None,
    ) -> None:

        self.tau_ref = (
            tau_ref.to_xarray() if isinstance(tau_ref, LUT) else tau_ref
        )
        if np.isscalar(w_ref) or (
            isinstance(w_ref, np.ndarray) and w_ref.ndim == 0
        ):
            self.w_ref = np.array([w_ref])
        else:
            self.w_ref = np.array(w_ref)

        if isinstance(phase, xr.DataArray):
            self._phase = phase
        elif isinstance(phase, LUT):
            self._phase = phase.to_xarray()
        elif phase is None:
            self._phase = phase
        else:
            raise ValueError(
                "The phase variable must be an xr.DataArray or be None."
            )
        if self._phase is not None and "stk" in self._phase.dims:
            # legacy phase inputs name the term dimension stk
            self._phase = self._phase.rename(stk="nphamat")

        if ssa is None:
            self.ssa = None
        else:
            if isinstance(ssa, Sequence):
                ssa = np.array(ssa)
            if np.isscalar(ssa) or (
                isinstance(ssa, np.ndarray) and (ssa.ndim <= 2)
            ):
                self.ssa = ssa
            elif isinstance(ssa, LUT):
                self.ssa = ssa.to_xarray()
            elif isinstance(ssa, xr.DataArray):
                self.ssa = ssa
            else:
                raise ValueError(
                    "The ssa variable must a scalar, a list, an ndarray of "
                    + "dim <= 2, or an xr.DataArray."
                )

        fname = Path(fname)
        if fname.parent == Path("."):  # no directory given
            fname = (
                DIR_AUXDATA / "aerosols" / "OPAC" / "mixtures" / fname.name
            )

        # Add extension if needed
        if "_sol" not in fname.name and not fname.suffix == ".nc":
            fname = fname.with_name(fname.name + "_sol.nc")
        elif fname.suffix != ".nc":
            fname = fname.with_name(fname.name + ".nc")

        if not fname.exists():
            raise FileNotFoundError(f"{fname} does not exist")

        self.fname = fname

        self.ds_mix = xr.open_dataset(self.fname)
        # check if hum dim size == 1 (to avoid interpolation/indexing
        # crash)
        if self.ds_mix.sizes["hum"] == 1:
            hum_v1 = float(self.ds_mix.coords["hum"].values[0])
            hum_v2 = hum_v1 + 1
            ds2 = self.ds_mix.assign_coords(hum=[hum_v2])
            self.ds_mix = xr.concat([self.ds_mix, ds2], dim="hum")

        if h_min_mix is None:
            h_min_mix = float(self.ds_mix.attrs["H_mix_min"])
        if h_mix_max is None:
            h_mix_max = float(self.ds_mix.attrs["H_mix_max"])
        if h_free_min is None:
            h_free_min = float(self.ds_mix.attrs["H_free_min"])
        if h_free_max is None:
            h_free_max = float(self.ds_mix.attrs["H_free_max"])
        if h_stra_min is None:
            h_stra_min = float(self.ds_mix.attrs["H_stra_min"])
        if h_stra_max is None:
            h_stra_max = float(self.ds_mix.attrs["H_stra_max"])

        if z_mix is None:
            z_mix = float(self.ds_mix.attrs["Z_mix"])
        if z_free is None:
            z_free = float(self.ds_mix.attrs["Z_free"])
        if z_stra is None:
            if self.ds_mix.attrs["Z_stra"] == "99":
                z_stra = 1e6  # -> OPAC Z=99 for constant vertical dist
            else:
                z_stra = float(self.ds_mix.attrs["Z_stra"])

        self.hum_or_reff = "hum"
        self.free_tropo = None
        self.strato = None

        self.force_rh = [rh_mix, rh_free, rh_stra]
        self.vert_content = []
        self.h_min = []
        self.h_max = []
        self.z_sh = []

        if h_mix_max - h_min_mix > 1e-6:
            self.vert_content.append(self.ds_mix)
            self.h_min.append(h_min_mix)
            self.h_max.append(h_mix_max)
            self.z_sh.append(z_mix)
        if h_free_max - h_free_min > 1e-6:
            filename_tmp = (
                DIR_AUXDATA
                / "aerosols"
                / "OPAC"
                / "free_troposphere"
                / "free_troposphere_sol.nc"
            )
            self.free_tropo = xr.open_dataset(filename_tmp)
            # check we have the same wavelength dim than previous aer
            # pro in vert_content
            if len(self.vert_content) > 0:
                aer_prev = self.vert_content[-1]
                w_cur = self.free_tropo.wav.values
                w_prev = aer_prev.wav.values
                nwcur = len(w_cur)
                nwprev = len(w_prev)
                if nwcur != nwprev or (
                    nwcur == nwprev and not np.array_equal(w_cur, w_prev)
                ):
                    wavelength_clip = w_prev.clip(
                        min=w_cur.min().item(), max=w_cur.max().item()
                    )
                    self.free_tropo = self.free_tropo.interp(
                        wav=wavelength_clip
                    )
            self.vert_content.append(self.free_tropo)
            self.h_min.append(h_free_min)
            self.h_max.append(h_free_max)
            self.z_sh.append(z_free)
        if h_stra_max - h_stra_min > 1e-6:
            filename_tmp = (
                DIR_AUXDATA
                / "aerosols"
                / "OPAC"
                / "stratosphere"
                / "stratosphere_sol.nc"
            )
            self.strato = xr.open_dataset(filename_tmp)
            # check we have the same wavelength dim than previous aer
            # pro in vert_content
            if len(self.vert_content) > 0:
                aer_prev = self.vert_content[-1]
                w_cur = self.strato.wav.values
                w_prev = aer_prev.wav.values
                nwcur = len(w_cur)
                nwprev = len(w_prev)
                if nwcur != nwprev or (
                    nwcur == nwprev and not np.array_equal(w_cur, w_prev)
                ):
                    wavelength_clip = w_prev.clip(
                        min=w_cur.min().item(), max=w_cur.max().item()
                    )
                    self.strato = self.strato.interp(wav=wavelength_clip)
            self.vert_content.append(self.strato)
            self.h_min.append(h_stra_min)
            self.h_max.append(h_stra_max)
            self.z_sh.append(z_stra)

    def dtau_ssa(
        self,
        wavelength: np.ndarray,
        z: np.ndarray,
        rh: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate optical depth and single scattering albedo.

        Computes the spectral optical depth (dtau) and single scattering
        albedo (ssa)
        for aerosol/cloud layers at specified wavelengths and altitudes.
        This method
        works with both AerOPAC (aerosol) and Cloud classes (which
        inherits from AerOPAC).
        Handles vertical profiles (mixtures, free troposphere,
        stratosphere) and optional
        scaling/forcing of optical properties.

        Parameters
        ----------
        wavelength : array-like
            Wavelengths (in nm) at which to calculate optical properties
        z : array-like
            Altitude profile (in km) for which to calculate optical
            properties
        rh : float or array-like, optional
            Relative humidity (0-100). Only used with AerOPAC class;
            ignored for Cloud.
            Also ignored for specific vertical layers if their
            corresponding layer-specific
            humidity values (rh_mix, rh_free, rh_stra) are set to
            non-None during initialization.
            For example, if only rh_mix is specified, rh is ignored only
            in the mixture layer.

        Returns
        -------
        dtau : ndarray
            Optical depth with shape (len(wavelength), len(z))
        ssa : ndarray
            Single scattering albedo with shape (len(wavelength),
            len(z))
        """
        dtau = np.zeros((len(wavelength), len(z)), dtype=np.float32)
        dtau_ref = np.zeros((1, len(z)), dtype=np.float32)
        ssa = np.zeros_like(dtau)

        if self.hum_or_reff == "hum":
            hum_or_reff_val = rh
        elif isinstance(self, Cloud):
            hum_or_reff_val = self.reff
        else:
            raise ValueError(
                "ext and ssa must varies as function of hum or reff."
            )

        if np.isscalar(hum_or_reff_val) or (
            isinstance(hum_or_reff_val, np.ndarray)
            and hum_or_reff_val.ndim == 0
        ):
            hum_or_reff_val = np.array([hum_or_reff_val])
        else:
            hum_or_reff_val = np.array(hum_or_reff_val)

        ext_ = np.zeros_like(dtau)
        ext_ref_ = np.zeros_like(dtau_ref)
        ssa_ = np.zeros_like(dtau)
        hor = self.hum_or_reff
        for icont, cont in enumerate(self.vert_content):
            cont_hor_vals = cont.coords[hor].values.astype(np.float64)
            cont_wavelength_vals = cont.coords["wav"].values.astype(np.float64)
            ext_data = cont["ext"].values.astype(np.float64)
            ssa_data = cont["ssa"].values.astype(np.float64)
            if (hor == "hum") and (self.force_rh[icont] is not None):
                rh_reff = np.full_like(hum_or_reff_val, self.force_rh[icont])
            else:
                rh_reff = hum_or_reff_val
            # Axes values
            hor_vals = cont_hor_vals
            wavelength_vals = cont_wavelength_vals
            # Float indices with extrema fill for humidity/reff, strict
            # bounds for wavelength
            nhor = len(hor_vals)
            n_wavelength_orig = len(wavelength_vals)
            idf_hor = np.interp(
                np.asarray(rh_reff, dtype=np.float64),
                hor_vals,
                np.arange(nhor),
                left=0,
                right=nhor - 1,
            )
            idf_wavelength = np.interp(
                np.asarray(wavelength, dtype=np.float64),
                wavelength_vals,
                np.arange(n_wavelength_orig),
            )
            idf_wavelength_ref = np.interp(
                np.atleast_1d(np.asarray(self.w_ref, dtype=np.float64)),
                wavelength_vals,
                np.arange(n_wavelength_orig),
            )
            if len(rh_reff) == 1:
                # Interpolate along hor (dim 0) -> (1, wavelength_orig)
                ext_at_hor = cast(
                    NDArray,
                    vec_float_indexing(ext_data, [idf_hor, slice(None)]),
                )  # (1, wavelength_orig)
                ssa_at_hor = cast(
                    NDArray,
                    vec_float_indexing(ssa_data, [idf_hor, slice(None)]),
                )  # (1, wavelength_orig)
                # Transpose to (wavelength_orig, 1), interpolate
                # along wavelength (dim 0) -> (n_wavelength, 1)
                ext_tmp = cast(
                    NDArray,
                    vec_float_indexing(
                        ext_at_hor.T, [idf_wavelength, slice(None)]
                    ),
                )  # (n_wavelength, 1)
                ext_ref_tmp = cast(
                    NDArray,
                    vec_float_indexing(
                        ext_at_hor.T, [idf_wavelength_ref, slice(None)]
                    ),
                )  # (n_wavelength_ref, 1)
                ssa_tmp = cast(
                    NDArray,
                    vec_float_indexing(
                        ssa_at_hor.T, [idf_wavelength, slice(None)]
                    ),
                )  # (n_wavelength, 1)
                for iz in range(0, len(z)):
                    ext_[:, iz] = ext_tmp[:, 0]
                    ext_ref_[:, iz] = ext_ref_tmp[:, 0]
                    ssa_[:, iz] = ssa_tmp[:, 0]
            else:
                # Interpolate along hor (dim 0) -> (nhor_query,
                # wavelength_orig)
                ext_at_hor = cast(
                    NDArray,
                    vec_float_indexing(ext_data, [idf_hor, slice(None)]),
                )  # (nhor, wavelength_orig)
                ssa_at_hor = cast(
                    NDArray,
                    vec_float_indexing(ssa_data, [idf_hor, slice(None)]),
                )  # (nhor, wavelength_orig)
                # Transpose to (wavelength_orig, nhor), interpolate
                # along wavelength (dim 0) -> (n_wavelength, nhor)
                ext_ = cast(
                    NDArray,
                    vec_float_indexing(
                        ext_at_hor.T, [idf_wavelength, slice(None)]
                    ),
                )  # (n_wavelength, nhor)
                ext_ref_ = cast(
                    NDArray,
                    vec_float_indexing(
                        ext_at_hor.T, [idf_wavelength_ref, slice(None)]
                    ),
                )  # (n_wavelength_ref, nhor)
                ssa_ = cast(
                    NDArray,
                    vec_float_indexing(
                        ssa_at_hor.T, [idf_wavelength, slice(None)]
                    ),
                )  # (n_wavelength, nhor)
            dtau_ = np.zeros_like(dtau)
            dtau_ref_ = np.zeros_like(dtau_ref)
            h1 = np.maximum(self.h_min[icont], z[1:])
            h2 = np.minimum(self.h_max[icont], z[:-1])
            cond = h2 > h1
            dtau_[:, 1:][:, cond] = ext_[:, 1:][
                :, cond
            ] * get_aer_dist_integral(self.z_sh[icont], h1[cond], h2[cond])
            dtau += dtau_
            ssa += dtau_ * ssa_
            dtau_ref_[:, 1:][:, cond] = ext_ref_[:, 1:][
                :, cond
            ] * get_aer_dist_integral(self.z_sh[icont], h1[cond], h2[cond])
            dtau_ref += dtau_ref_

        ssa[dtau != 0] /= dtau[dtau != 0]

        # apply scaling factor to get the required optical thickness at
        # the
        # specified wavelength or force tau for all wavelengths
        if self.tau_ref is not None:
            if (
                isinstance(self.tau_ref, np.ndarray) and self.tau_ref.ndim == 0
            ) or np.isscalar(self.tau_ref):
                dtau *= np.asarray(self.tau_ref, dtype=np.float64) / np.sum(
                    dtau_ref
                )
            elif isinstance(self.tau_ref, xr.DataArray):
                # xr.DataArray
                wavelength_axis = self.tau_ref.coords[
                    self.tau_ref.dims[0]
                ].values.astype(np.float64)
                tau_ref_interp = np.interp(
                    np.asarray(wavelength, dtype=np.float64),
                    wavelength_axis,
                    self.tau_ref.values,
                )
                dtau *= (tau_ref_interp / np.sum(dtau, axis=1))[:, None]

        # force ssa
        if self.ssa is not None:
            if np.isscalar(self.ssa):  # scalar
                ssa[:, :] = float(cast(float, self.ssa))
            # ndarray with dim <= 2
            elif isinstance(self.ssa, np.ndarray):
                if self.ssa.ndim == 0:
                    ssa[:, :] = self.ssa
                elif self.ssa.ndim == 1:
                    # if 1d array -> only wavelength variability
                    ssa[:, :] = self.ssa[:, None]
                elif self.ssa.ndim == 2:
                    ssa[:, :] = self.ssa[:, :]
            elif isinstance(self.ssa, xr.DataArray):  # xr.DataArray
                wavelength_axis = self.ssa.coords[
                    self.ssa.dims[0]
                ].values.astype(np.float64)
                ssa_interp = np.interp(
                    np.asarray(wavelength, dtype=np.float64),
                    wavelength_axis,
                    self.ssa.values,
                )
                ssa[:, :] = ssa_interp[:, None]
        return dtau, ssa

    def native_theta(self) -> NDArray[np.float64]:
        """The scattering angles the component's tables carry.

        The grid of a user-supplied phase matrix when there is one,
        which is then the only table in use, else the union of the
        grids of the vertical contents (mixture layer, free
        troposphere, stratosphere), in degrees. This is the grid
        ``n_theta='native'`` resolves to for this component alone.

        Returns
        -------
        ndarray
            Strictly increasing angles in degrees, from 0 to 180.
        """
        if self._phase is not None:
            return as_theta_grid(
                self._phase.coords["theta_atm"].values.astype(np.float64)
            )
        if not self.vert_content:
            raise ValueError(
                "The component holds no vertical layer (every layer "
                "has a zero or negative thickness); it carries no "
                "scattering angle grid."
            )
        return union_theta_grid(
            [
                cont.coords["theta"].values.astype(np.float64)
                for cont in self.vert_content
            ]
        )

    def phase(
        self,
        wavelength: np.ndarray,
        z: np.ndarray,
        rh: np.ndarray,
        n_theta: ThetaLike = 721,
    ) -> xr.DataArray:
        """
        Calculate phase matrix for aerosols and clouds.

        Computes the phase matrix at specified wavelengths and altitudes
        for aerosol/cloud
        layers. This method works with both AerOPAC (aerosol) and Cloud
        classes (which
        inherits from AerOPAC). Handles vertical profiles (mixtures,
        free troposphere,
        stratosphere) and performs angle resampling. Supports both
        spherical (4 Stokes
        components) and non-spherical (6 components) particles.

        Parameters
        ----------
        wavelength : array-like
            Wavelengths (in nm) at which to calculate phase matrix
        z : array-like
            Altitude profile (in km) for which to calculate phase matrix
        rh : array-like
            Relative humidity (%). Must have size similar to z (altitude
            profile).
            Only used with AerOPAC class; ignored for Cloud.
            Relative humidity can be greater than 100%
            (supersaturation).
            Also ignored for specific vertical layers if their
            corresponding layer-specific
            humidity values (rh_mix, rh_free, rh_stra) are set to
            non-None during initialization.
            For example, if only rh_mix is specified, rh is ignored only
            in the mixture layer.
        n_theta : int, str or array_like, optional
            The number of equally spaced scattering angles the phase
            matrix is resampled on, the angles themselves in degrees
            (which `smartg.phase.theta_grid` can build clustered
            towards the forward and backward directions), or
            ``'native'`` to keep the angles the component's tables
            carry, see `native_theta`. A user-supplied phase matrix
            (the `phase` argument of the constructor) is returned on
            its own grid whatever `n_theta`. Default is 721.

        Returns
        -------
        phase_matrix : DataArray
            DataArray containing the phase matrix with dimensions
            [wavelength_phase, z_phase, nphamat, theta_atm].
            Shape is (len(wavelength), len(z)-1, nphamat, n_theta)
            where:
            - nphamat = 4 for spherical particles only (phase matrix
              unique terms P11, P21, P33, P34)
            - nphamat = 6 for spherical and non-spherical particles
              (additional phase matrix unique terms P22, P44)
            - theta_atm: scattering angles from 0° to 180°
        """

        if self._phase is not None:
            if self._phase.ndim == 2:
                # convert to 4-dim by inserting empty dimensions
                # wavelength_phase and z_phase
                dims = list(self._phase.dims)
                assert dims == ["nphamat", "theta_atm"]
                pha_ = self._phase.values[:, :]
                if pha_.shape[0] == 4:
                    pha_6 = np.zeros((6, pha_.shape[1]), dtype=pha_.dtype)
                    pha_6[0:4, :] = pha_
                    pha_6[4, :] = pha_[0, :].copy()  # F22 = F11
                    pha_6[5, :] = pha_[2, :].copy()  # F44 = F33
                    pha_ = pha_6
                return xr.DataArray(
                    pha_[None, None, :, :],
                    dims=[
                        "wavelength_phase", "z_phase", "nphamat",
                        "theta_atm",
                    ],
                    coords={
                        "wavelength_phase": [wavelength[0]],
                        "z_phase": [0.0],
                        "nphamat": np.arange(6),
                        "theta_atm": self._phase.coords["theta_atm"].values,
                    },
                )
            else:
                dims = list(self._phase.dims)
                pha_ = self._phase.values
                if pha_.shape[2] == 4:
                    pha_6 = np.zeros(
                        (pha_.shape[0], pha_.shape[1], 6, pha_.shape[3]),
                        dtype=pha_.dtype,
                    )
                    pha_6[:, :, 0:4, :] = pha_
                    # F22 = F11 ; F44 = F33
                    pha_6[:, :, 4, :] = pha_[:, :, 0, :].copy()
                    pha_6[:, :, 5, :] = pha_[:, :, 2, :].copy()
                    return xr.DataArray(
                        pha_6,
                        dims=dims,
                        coords={
                            dims[0]: self._phase.coords[dims[0]].values,
                            dims[1]: self._phase.coords[dims[1]].values,
                            "nphamat": np.arange(6),
                            dims[3]: self._phase.coords[dims[3]].values,
                        },
                    )
                return xr.DataArray(
                    pha_,
                    dims=dims,
                    coords={d: self._phase.coords[d].values for d in dims},
                )

        if not self.vert_content:
            raise ValueError(
                "The component holds no vertical layer (every layer "
                "has a zero or negative thickness); cannot compute "
                "its phase matrix."
            )

        theta = (
            self.native_theta() if is_native_theta(n_theta)
            else as_theta_grid(n_theta)
        )
        n_theta = len(theta)
        wavelength_tabulated = self.ds_mix.coords["wav"].values
        n_wavelength = len(wavelength)

        P_tot = 0.0
        dssa = 0.0
        for icont, cont in enumerate(self.vert_content):
            hor = self.hum_or_reff

            phase_data = cont["phase"].values
            hor_vals = cont.coords[hor].values.astype(np.float64)
            wavelength_vals = cont.coords["wav"].values.astype(np.float64)
            theta_orig = cont.coords["theta"].values.astype(np.float64)
            ext_data = cont["ext"].values.astype(np.float64)
            ssa_data = cont["ssa"].values.astype(np.float64)

            nphamat = phase_data.shape[2]
            nhor = len(hor_vals)
            n_wavelength_orig = len(wavelength_vals)

            # Wavelength optimization: subset to bracketing wavelengths
            if (np.max(wavelength) > np.max(wavelength_tabulated)) or (
                np.min(wavelength) < np.min(wavelength_tabulated)
            ):
                # Out of range: use full axis
                wavelength_subset = wavelength_vals
                phase_subset = phase_data
            else:
                range_ind = np.array(
                    [
                        np.argwhere(
                            (wavelength_tabulated <= np.min(wavelength))
                        )[-1][0],
                        np.argwhere(
                            (wavelength_tabulated >= np.max(wavelength))
                        )[0][0],
                    ]
                )
                ilam_tabulated = np.arange(len(wavelength_tabulated), dtype=int)
                ilam_opti = np.concatenate(
                    np.argwhere(
                        (ilam_tabulated >= range_ind[0])
                        & (ilam_tabulated <= range_ind[1])
                    )
                )
                wavelength_subset = wavelength_vals[ilam_opti]
                phase_subset = phase_data[:, ilam_opti, :, :]

            n_wavelength_sub = len(wavelength_subset)

            # Interpolate along wavelength: transpose to
            # (wavelength, hor, stk, theta) for vec_float_indexing
            if n_wavelength_sub > 1:
                idf_wavelength = np.interp(
                    np.asarray(wavelength, dtype=np.float64),
                    wavelength_subset,
                    np.arange(n_wavelength_sub),
                )
                phase_at_wavelength = cast(
                    NDArray,
                    vec_float_indexing(
                        np.ascontiguousarray(
                            phase_subset.transpose(1, 0, 2, 3)
                        ),
                        [idf_wavelength, slice(None), slice(None),
                         slice(None)],
                    ),
                )
            else:
                phase_at_wavelength = np.broadcast_to(
                    phase_subset.transpose(1, 0, 2, 3),
                    (n_wavelength, nhor, nphamat, len(theta_orig)),
                ).copy()
            # Result: (n_wavelength, hor, stk, theta_orig)

            # Theta resampling if needed: transpose to
            # (theta, n_wavelength, hor, stk)
            if not np.array_equal(theta, theta_orig):
                idf_theta = np.interp(
                    theta, theta_orig, np.arange(len(theta_orig))
                )
                phase_at_wavelength = cast(
                    NDArray,
                    vec_float_indexing(
                        np.ascontiguousarray(
                            phase_at_wavelength.transpose(3, 0, 1, 2)
                        ),
                        [idf_theta, slice(None), slice(None), slice(None)],
                    ),
                )
                # Result: (n_theta, n_wavelength, hor, stk) ->
                # transpose to (n_wavelength, hor, stk, n_theta)
                phase_at_wavelength = phase_at_wavelength.transpose(1, 2, 3, 0)
            # phase_at_wavelength: (n_wavelength, hor, stk, n_theta)

            # Determine humidity/reff values
            nphamat_ = 6
            if isinstance(self, Cloud):
                hum_or_reff_val = self.reff
            elif self.hum_or_reff == "hum":
                if self.force_rh[icont] is not None:
                    hum_or_reff_val = np.full_like(rh, self.force_rh[icont])
                else:
                    hum_or_reff_val = rh
            else:
                raise ValueError(
                    "Phase matrix must varies as function of hum or reff."
                )

            if np.isscalar(hum_or_reff_val) or (
                isinstance(hum_or_reff_val, np.ndarray)
                and hum_or_reff_val.ndim == 0
            ):
                hum_or_reff_val = np.array([hum_or_reff_val])
            else:
                hum_or_reff_val = np.array(hum_or_reff_val)

            # Interpolate along hor: transpose to
            # (hor, n_wavelength, stk, n_theta)
            if len(hum_or_reff_val) == 1:
                hor_query = hum_or_reff_val
                nz_phase = len(z) - 1
            else:
                hor_query = hum_or_reff_val[1:]
                nz_phase = len(hum_or_reff_val) - 1

            idf_hor = np.interp(
                np.asarray(hor_query, dtype=np.float64),
                hor_vals,
                np.arange(nhor),
                left=0,
                right=nhor - 1,
            )
            P_data = cast(
                NDArray,
                vec_float_indexing(
                    np.ascontiguousarray(
                        phase_at_wavelength.transpose(1, 0, 2, 3)
                    ),
                    [idf_hor, slice(None), slice(None), slice(None)],
                ),
            )
            # Result: (nz, n_wavelength, stk, n_theta) -> transpose
            # to (n_wavelength, nz, stk, n_theta)
            P_data = np.ascontiguousarray(P_data.transpose(1, 0, 2, 3)).astype(
                np.float32
            )
            if len(hum_or_reff_val) == 1:
                P_data = np.broadcast_to(
                    P_data,
                    (n_wavelength, nz_phase, P_data.shape[2], P_data.shape[3]),
                ).copy()

            # Expand 4 stk to 6 if needed
            if nphamat == 4:
                P_data_6 = np.zeros(
                    (n_wavelength, nz_phase, nphamat_, n_theta),
                    dtype="float32",
                )
                P_data_6[:, :, 0:4, :] = P_data
                # F22 = F11 ; F44 = F33
                P_data_6[:, :, 4, :] = P_data[:, :, 0, :].copy()
                P_data_6[:, :, 5, :] = P_data[:, :, 2, :].copy()
                P_data = P_data_6
            elif nphamat == 6:
                pass
            else:
                P_data_6 = np.zeros(
                    (n_wavelength, nz_phase, nphamat_, n_theta),
                    dtype="float32",
                )
                P_data_6[:, :, 0:nphamat, :] = P_data
                P_data = P_data_6

            P = xr.DataArray(
                P_data,
                dims=["wavelength_phase", "z_phase", "nphamat", "theta_atm"],
                coords={
                    "wavelength_phase": wavelength,
                    "z_phase": np.arange(P_data.shape[1]),
                    "nphamat": np.arange(P_data.shape[2]),
                    "theta_atm": theta,
                },
            )

            # Compute dtau and ssa using vec_float_indexing (same as
            # dtau_ssa)
            idf_hor_ext = np.interp(
                np.asarray(hum_or_reff_val, dtype=np.float64),
                hor_vals,
                np.arange(nhor),
                left=0,
                right=nhor - 1,
            )
            idf_wavelength_ext = np.interp(
                np.asarray(wavelength, dtype=np.float64),
                wavelength_vals,
                np.arange(n_wavelength_orig),
            )
            ext_at_hor = cast(
                NDArray,
                vec_float_indexing(ext_data, [idf_hor_ext, slice(None)]),
            )
            ssa_at_hor = cast(
                NDArray,
                vec_float_indexing(ssa_data, [idf_hor_ext, slice(None)]),
            )
            ext_ = cast(
                NDArray,
                vec_float_indexing(
                    ext_at_hor.T, [idf_wavelength_ext, slice(None)]
                ),
            )  # (n_wavelength, nhor_q)
            ssa_ = cast(
                NDArray,
                vec_float_indexing(
                    ssa_at_hor.T, [idf_wavelength_ext, slice(None)]
                ),
            )  # (n_wavelength, nhor_q)
            if len(hum_or_reff_val) == 1:
                ext_ = np.broadcast_to(ext_, (n_wavelength, len(z))).copy()
                ssa_ = np.broadcast_to(ssa_, (n_wavelength, len(z))).copy()

            dtau_ = np.zeros((len(wavelength), len(z)), dtype=np.float32)
            h1 = np.maximum(self.h_min[icont], z[1:])
            h2 = np.minimum(self.h_max[icont], z[:-1])
            cond = h2 > h1
            dtau_[:, 1:][:, cond] = ext_[:, 1:][
                :, cond
            ] * get_aer_dist_integral(self.z_sh[icont], h1[cond], h2[cond])
            dssa_ = dtau_ * ssa_  # NLAM, ALTITUDE
            dssa_ = dssa_[:, 1:, None, None]
            dssa += dssa_
            P_tot += P * dssa_

        P_tot = cast(xr.DataArray, P_tot)
        with np.errstate(divide="ignore", invalid="ignore"):
            P_tot.data /= dssa
        P_tot.data[np.isnan(P_tot.data)] = 0.0
        P_tot = P_tot.assign_coords(z_phase=z[1:])
        return P_tot

    @staticmethod
    def list() -> list[str]:
        """List available standard OPAC aerosol mixture files.

        Returns
        -------
        list of str
            List of available OPAC aerosol mixture filenames (without
            suffix).

        Examples
        --------
        >>> from smartg.atmosphere import AerOPAC
        >>> AerOPAC.list()
        ['antarctic', 'antarctic_spheric', 'arctic',
        'continental_average', ...]
        """
        base_dir = DIR_AUXDATA / "aerosols" / "OPAC" / "mixtures"
        files = list(base_dir.glob("*.nc"))
        return sorted([f.stem.replace("_sol", "") for f in files])


class Cloud(AerOPAC):
    """
    Initialize the cloud model

    Parameters
    ----------
    fname : str,
        Complete path to the cloud file or fname for clouds located
        in "auxdata/clouds/"
        Available auxdata clouds: wc, ic_baum_ghm, ic_baum_asc and
        ic_baum_sc
    reff : float
        Effective radius in micrometers
    zmin : float,
        Minimum altitude of the cloud
    zmax : float,
        Maximum altitude of the cloud
    tau_ref : float,
        Optical thickness at reference wavelength w_ref
    w_ref : float
        Wavelength in nanometers at reference optical thickness tau_ref
        ssa : None or float or list or 1-D ndarray or 2-D ndarray or
            DataArray, optional
        Force particle single scattering albedo.

        - if float -> same value for all wavelengths and altitudes
        - if list -> it will be converted into a 1-D ndarray.
        - if 1-D ndarray -> only wavelength dependence is considered
        - if 2-D ndarray -> wavelength and altitude dependence is
          considered
        - if DataArray -> wavelength and altitude dependence is
          considered

        Note that DataArray is more flexible since it allows
        interpolation if wavelengths
        in calc method are different (but not the case for the altitude
        axis).
    phase : None or DataArray, optional
        Phase matrix F as function of wavelength, altitude, stoke
        components and scattering angle
        The variable names must be:
        If 4-D matrix -> wavelength_phase, z_phase, nphamat, theta
        If 2-D matrix (assumed monochromatic and contant vertically) ->
        nphamat, theta
        Where:
        - wavelength_phase is the wavelength. It must be equal to
          the `wavelength_phase` parameter of Atm1D if defined, else
          the `wavelength` parameter wavelengths of the Atm1D calc
          method.
        - z_phase is the phase altitude. It must be equal to the
          `pfgrid[1:]` parameter
          of Atm1D
        - nphamat the phase matrix unique terms.
        - theta the scattering angle.

        The phase matrix terms (IQUV convention) must be given in the
        folowing order:
        - F11, F21, F33 and F34 if only 4 terms are given (only for
          spherical particles)
        - F11, F21, F33, F34, F22 and F44 if 6 terms are given (for both
          spherical and non-spherical particles)

    Examples
    --------
    >>> from smartg.atmophere import Cloud
    >>> cld_wc = Cloud('wc', 12.68, 2, 3, 10., 550.)
    >>> print(cld_wc.ds_mix)
    <xarray.Dataset> Size: 52MB
    Dimensions:  (reff: 26, wav: 209, stk: 4, theta: 594)
    Coordinates:
    * reff     (reff) float32 104B 5.0 6.0 7.0 8.0 9.0 ... 27.0 28.0
                                   29.0 30.0
    * wav      (wav) float32 836B 253.1 256.6 260.2 ... 4.38e+03
                                  4.441e+03
    * stk      (stk) int16 8B 0 1 2 3
    * theta    (theta) float64 5kB 0.0 0.01 0.02 0.03 ... 179.2 179.5
                                   179.8 180.0
    Data variables:
        phase    (reff, wav, stk, theta) float32 52MB 8.765e+03
                                                      8.759e+03 ... 0.0
        ext      (reff, wav) float64 43kB 123.1 123.1 123.2 ...
                                          4.619e+03 4.622e+03
        ssa      (reff, wav) float64 43kB 1.0 1.0 1.0 1.0 ... 0.6522
                                          0.636 0.6203
    Attributes:
        veff:     0.1
    """

    def __init__(
        self,
        fname: str | Path,
        reff: float,
        zmin: float,
        zmax: float,
        tau_ref: float,
        w_ref: float,
        ssa: float
        | list[float]
        | np.ndarray
        | xr.DataArray
        | LUT
        | None = None,
        phase: xr.DataArray | LUT | None = None,
    ) -> None:
        if zmax - zmin <= 1e-6:
            raise ValueError(
                "The cloud layer must have zmax > zmin, got "
                f"zmin={zmin} and zmax={zmax}."
            )
        self.reff = reff
        self.tau_ref = tau_ref
        if np.isscalar(w_ref) or (
            isinstance(w_ref, np.ndarray) and w_ref.ndim == 0
        ):
            self.w_ref = np.array([w_ref])
        else:
            self.w_ref = np.array(w_ref)

        if ssa is None:
            self.ssa = None
        else:
            if isinstance(ssa, list):
                ssa = np.array(ssa)
            if np.isscalar(ssa) or (
                isinstance(ssa, np.ndarray) and (ssa.ndim <= 2)
            ):
                self.ssa = ssa
            elif isinstance(ssa, LUT):
                self.ssa = ssa.to_xarray()
            elif isinstance(ssa, xr.DataArray):
                self.ssa = ssa
            else:
                raise ValueError(
                    "The ssa variable must a scalar, a list, an ndarray"
                    + " of dim <= 2, or an xr.DataArray."
                )

        fname = Path(fname)
        if fname.parent == Path("."):  # no directory given
            base_dir = Path(DIR_AUXDATA) / "clouds"
            fname = base_dir / fname.name

        if "_sol" not in fname.name and not fname.suffix == ".nc":
            fname = fname.with_name(fname.name + "_sol.nc")
        elif fname.suffix != ".nc":
            fname = fname.with_name(fname.name + ".nc")

        if not fname.exists():
            raise FileNotFoundError(f"{fname} does not exist")

        self.fname = fname

        self.ds_mix = xr.open_dataset(self.fname)
        # check if reff dim size == 1 (to avoid interpolation/indexing
        # crash)
        if self.ds_mix.sizes["reff"] == 1:
            reff_v1 = float(self.ds_mix.coords["reff"].values[0])
            reff_v2 = reff_v1 + 1
            ds2 = self.ds_mix.assign_coords(reff=[reff_v2])
            self.ds_mix = xr.concat([self.ds_mix, ds2], dim="reff")

        self.hum_or_reff = "reff"
        self.free_tropo = None
        self.strato = None

        self.vert_content = []
        self.h_min = []
        self.h_max = []
        self.z_sh = []

        if zmax - zmin > 1e-6:
            self.vert_content.append(self.ds_mix)
            self.h_min.append(zmin)
            self.h_max.append(zmax)
            self.z_sh.append(1e6)  # constant dist

        if isinstance(phase, xr.DataArray):
            self._phase = phase
        elif isinstance(phase, LUT):
            self._phase = phase.to_xarray()
        elif phase is None:
            self._phase = phase
        else:
            raise ValueError(
                "The phase variable must be an xr.DataArray or be None."
            )
        if self._phase is not None and "stk" in self._phase.dims:
            # legacy phase inputs name the term dimension stk
            self._phase = self._phase.rename(stk="nphamat")

    @staticmethod
    def list() -> list[str]:
        """List available standard cloud model files.

        Returns
        -------
        list of str
            List of available cloud model filenames (without suffix).

        Examples
        --------
        >>> from smartg.atmosphere import Cloud
        >>> Cloud.list()
        ['ic_baum_asc', 'ic_baum_ghm', 'ic_baum_sc', 'wc']
        """
        base_dir = Path(DIR_AUXDATA) / "clouds"
        files = list(base_dir.glob("*.nc"))
        return sorted([f.stem.replace("_sol", "") for f in files])


class AerUser(AerOPAC):
    """
    Initialize the user-defined aerosol model

    Parameters
    ----------
    aod : 2-D ndarray
        aerosol optical depth values with shape (len(hum), len(wavelength))
    ssa : 2-D ndarray
        Single scattering albedo values with shape (len(hum),
        len(wavelength))
    phase : 4-D ndarray
        Phase function values with shape (len(hum),
        len(wavelength), len(stk), len(theta)).

        Where len(stk) is the number of unique phase terms.

        The phase matrix terms must be given in the folowing order:
        - F11, F21, F33 and F34 if only 4 terms are given (only for
          spherical particles)
        - F11, F21, F33, F34, F22 and F44 if 6 terms are given (for both
          spherical and non-spherical particles)
    hum : 1-D ndarray
        Relative humidity values in percentage
    wavelength : 1-D ndarray
        Wavelength values in nanometers
    theta : 1-D ndarray
        Scattering angle values in degrees
    h_mix_min : float, optional
        Force min altitude of the mixture
    h_mix_max : float, optional
        Force max altitude of the mixture
    z_mix : float, optional
        Force scale height (see notes) of the mixture

    Notes
    -----
    The scale height (see Hess et al. 2004) is the variable Z in the
    following equation:

    - :math:`N(h) = N(0)exp(-h/Z)`

    with N the number density and h the altitude

    """

    def __init__(
        self,
        aod: np.ndarray,
        ssa: np.ndarray,
        phase: np.ndarray,
        hum: np.ndarray,
        wavelength: np.ndarray,
        theta: np.ndarray,
        h_mix_min: float = 0.0,
        h_mix_max: float = 2.0,
        z_mix: float = 2,
    ) -> None:

        self.fname = "none"
        self.tau_ref = None
        ext = aod / (
            z_mix * (np.exp(-h_mix_min / z_mix) - np.exp(-h_mix_max / z_mix))
        )

        # Create an xarray Dataset to hold the mixture data
        ds = xr.Dataset(
            {
                "ext": (("hum", "wav"), ext),
                "ssa": (("hum", "wav"), ssa),
                "phase": (("hum", "wav", "stk", "theta"), phase),
            },
            coords={
                "hum": hum,
                "wav": wavelength,
                "theta": theta,
                "stk": np.arange(phase.shape[2]),
            },
        )

        ds.attrs["name"] = "none"
        ds.attrs["H_mix_min"] = str(h_mix_min)
        ds.attrs["H_mix_max"] = str(h_mix_max)
        ds.attrs["Z_mix"] = str(z_mix)

        self.ds_mix = ds
        # check if hum dim size == 1 (to avoid interpolation/indexing
        # crash)
        if self.ds_mix.sizes["hum"] == 1:
            hum_v1 = float(self.ds_mix.coords["hum"].values[0])
            hum_v2 = hum_v1 + 1
            ds2 = self.ds_mix.assign_coords(hum=[hum_v2])
            self.ds_mix = xr.concat([self.ds_mix, ds2], dim="hum")

        self.w_ref = np.array([float(self.ds_mix.coords["wav"].values[0])])
        self.ssa = None

        self.hum_or_reff = "hum"
        self.free_tropo = None
        self.strato = None

        self.force_rh = [None]
        self.vert_content = []
        self.h_min = []
        self.h_max = []
        self.z_sh = []

        if h_mix_max - h_mix_min > 1e-6:
            self.vert_content.append(self.ds_mix)
            self.h_min.append(h_mix_min)
            self.h_max.append(h_mix_max)
            self.z_sh.append(z_mix)

        self._phase = None

    @staticmethod
    def list() -> list[str]:
        """"""
        raise NotImplementedError(
            "The list() method is not available for user-defined aerosols. "
            "User-defined aerosols are custom configurations and do not have "
            "a predefined list of available files."
        )


class Comp3D(ABC):
    """Base class for 3D atmospheric components (e.g. Cloud3D, Aer3D).

    A 3D component describes particles occupying a set of cells of a
    :class:`smartg.grid3d.Grid3D`, with per-cell optical properties.
    Implementations must provide the per-cell extinction, single
    scattering albedo and phase matrices used by :class:`Atm3D` to
    merge the component into the 3D atmospheric profile.
    """

    @abstractmethod
    def get_cell_indices(self) -> NDArray[np.int32]:
        """Return the (N, 3) 0-based (ix, iy, iz) indices of the cells
        occupied by the component, on the inner 3D grid (without
        boundary cells).
        """

    @abstractmethod
    def get_ext(self, wavelength: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return the (n_wavelength, N) extinction coefficients in
        km-1 of the component cells at the given wavelengths in nm.
        """

    @abstractmethod
    def get_ssa(self, wavelength: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return the (n_wavelength, N) single scattering albedos of the
        component cells at the given wavelengths in nm.
        """

    @abstractmethod
    def native_theta(self) -> NDArray[np.float64]:
        """The scattering angles the component's tables carry, in
        degrees, which ``n_theta='native'`` resolves to.
        """

    @abstractmethod
    def get_phase_set(
        self,
        wavelength_phase: NDArray[np.floating],
        n_theta: ThetaLike = 721,
    ) -> tuple[list[xr.DataArray], NDArray[np.int32], int]:
        """Return the component phase matrices.

        Parameters
        ----------
        wavelength_phase : ndarray
            The wavelengths of the phase matrices, in nm.
        n_theta : int, str or array_like, optional
            The number of equally spaced scattering angles, the angles
            themselves in degrees, or ``'native'`` for the angles the
            component's tables carry.

        Returns
        -------
        phases : list of DataArray
            The unique phase matrices ``('nphamat', 'theta_atm')``,
            as ``n_wavelength_phase`` consecutive blocks of
            ``n_unique`` matrices.
        cell_phase_index : ndarray
            (N,) index of each cell's phase matrix within one block.
        n_unique : int
            The number of unique phase matrices per wavelength block.
        """


class _Comp3DFile(Comp3D):
    """Shared implementation of the file-based 3D components.

    The bulk optical properties (spectral extinction, single
    scattering albedo and phase matrices as a function of a per-cell
    parameter) are read from a SMART-G NetCDF file, and the 3D
    distribution (per-cell extinction at the reference wavelength
    `w_ref` and per-cell parameter) is provided as a dense
    ``xr.Dataset``, as raw arrays or converted from the legacy
    I3RC/IPRT ASCII files. See :class:`Cloud3D` (parameter: the
    droplet effective radius) and :class:`Aer3D` (parameter: the
    relative humidity). The bulk optical properties are available
    as the ``ds_mix`` dataset attribute, like in the 1D
    :class:`AerOPAC` and :class:`Cloud` classes, and the 3D field
    as the ``ds_dist`` dataset attribute.
    """

    # per-cell parameter name in the public API and the dense
    # dataset: "reff" | "rh"
    _param_name: str
    # parameter axis name of the bulk optical properties file and of
    # the `phase` override matrix: "reff" | "hum"
    _bulk_axis: str
    # subdirectory of DIR_AUXDATA holding the bulk files
    _auxdata_subdir: tuple[str, ...]
    # out-of-range policy of the parameter lookups: "raise" rejects,
    # "clamp" clamps to the axis bounds
    _param_oor: str
    # component label used in the messages: "cloud" | "aerosol"
    _label: str

    def __init__(
        self,
        fname: PathType,
        w_ref: float | None = None,
        ds: xr.Dataset | PathType | None = None,
        param: NumericArrayLike | None = None,
        ext_ref: NumericArrayLike | None = None,
        cell_indices: NDArray[np.integer] | None = None,
        param_acc: int | None = None,
        param_min: float | None = None,
        param_max: float | None = None,
        phase: xr.DataArray | LUT | None = None,
        ssa_cst: float | None = None,
    ) -> None:

        fname = Path(fname)
        if fname.parent == Path("."):
            fname = Path(DIR_AUXDATA).joinpath(
                *self._auxdata_subdir
            ) / fname

        if "_sol" not in fname.name and fname.suffix != ".nc":
            fname = fname.with_name(fname.stem + "_sol.nc")
        elif fname.suffix != ".nc":
            fname = fname.with_name(fname.name + ".nc")

        if not fname.exists():
            raise FileNotFoundError(f"{fname} does not exist")

        self.fname = fname
        self.ds_mix = xr.open_dataset(self.fname)
        self.ssa_cst = ssa_cst

        if ds is not None:
            if not isinstance(ds, xr.Dataset):
                ds = xr.open_dataset(ds)
            missing = [
                v
                for v in ("ext", self._param_name)
                if v not in ds.data_vars
            ] + [
                c
                for c in ("x_bounds", "y_bounds", "z_bounds")
                if c not in ds.coords
            ]
            if missing:
                raise ValueError(
                    f"The 3D {self._label} dataset must define the "
                    f"'ext' and '{self._param_name}' variables over "
                    "('z', 'y', 'x') and the 'x_bounds', 'y_bounds' "
                    f"and 'z_bounds' coordinates; missing: {missing}"
                )
            self.ds_dist = ds
            # extract the occupied cells in C order with x slowest,
            # which follows the row order of the I3RC/IPRT ASCII files
            ext_xyz = ds["ext"].transpose("x", "y", "z").to_numpy()
            param_xyz = ds[self._param_name].transpose(
                "x", "y", "z"
            ).to_numpy()
            indices = np.argwhere(ext_xyz > 0.0)
            self._cell_indices = indices.astype(np.int32)
            self._ext_ref = ext_xyz[
                indices[:, 0], indices[:, 1], indices[:, 2]
            ]
            param = param_xyz[
                indices[:, 0], indices[:, 1], indices[:, 2]
            ]
            if w_ref is None:
                w_ref = ds.attrs.get("w_ref")
        else:
            if param is None or ext_ref is None or cell_indices is None:
                raise ValueError(
                    f"If ds is not given, then {self._param_name}, "
                    "ext_ref and cell_indices must all be given!"
                )
            self.ds_dist = None
            # the IPRT convention cell indices start at 1 instead of 0
            self._cell_indices = (
                np.asarray(cell_indices, dtype=np.int32) - 1
            )
            self._ext_ref = np.atleast_1d(
                np.asarray(ext_ref, dtype=np.float64)
            )
            param = np.atleast_1d(np.asarray(param, dtype=np.float64))

        if w_ref is None:
            raise ValueError(
                "w_ref must be given (or set as an attribute of ds)!"
            )
        self.w_ref = float(w_ref)

        param = np.asarray(param, dtype=np.float64)
        if param_acc is not None:
            param = np.around(param, decimals=param_acc)
        if param_min is not None:
            param[param < param_min] = param_min
        if param_max is not None:
            param[param > param_max] = param_max
        self._param = self._normalize_param(param)

        # Convert a legacy phase LUT into the DataArray used
        # internally
        if isinstance(phase, LUT):
            phase = phase.to_xarray()
        if isinstance(phase, xr.DataArray) and "stk" in phase.dims:
            # legacy phase inputs name the term dimension stk
            phase = phase.rename(stk="nphamat")

        if phase is None:
            self.phase = phase
        elif not isinstance(phase, xr.DataArray):
            raise ValueError(
                "phase must be an xr.DataArray or a LUT object!"
            )
        elif not all(
            item in phase.dims
            for item in [
                "wavelength_phase", self._bulk_axis, "nphamat", "theta_atm"
            ]
        ):
            raise ValueError(
                "Phase matrix must have 4 dimensions: wavelength_phase, "
                f"{self._bulk_axis}, nphamat and theta_atm"
            )
        else:
            # the readers keep the terms of the file: complete the 4
            # of spherical particles into 6, as the bulk file path of
            # get_phase does
            self.phase = expand_phase_4_to_6(phase)

    def _normalize_param(
        self, param: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Hook adjusting the per-cell parameter values at init, after
        the accuracy/clipping options. The base implementation returns
        them unchanged.
        """
        return param

    def _interp_axis(
        self,
        da: xr.DataArray,
        dim: str,
        values: NDArray[np.floating] | float,
        clamp: bool = False,
    ) -> xr.DataArray:
        """Linearly interpolate `da` along `dim` at `values`.

        Out-of-range values are clamped to the axis bounds when
        `clamp` is True and rejected otherwise, and a single-node
        axis only accepts its own value. An array of values becomes
        a pointwise 'cell' dimension, while a scalar removes the
        dimension.
        """
        axis = da[dim].values.astype(np.float64)
        vals = np.asarray(values, dtype=np.float64)
        if axis.size == 1:
            if not np.allclose(vals, axis[0]):
                raise ValueError(
                    f"The {dim} values must be equal to the single "
                    f"node of the {dim} axis ({axis[0]:g}), got "
                    f"{values}."
                )
            res = da.isel({dim: 0}, drop=True)
            if vals.ndim > 0:
                res = res.expand_dims({"cell": vals.size})
            return res
        lo, hi = axis.min(), axis.max()
        if clamp:
            vals = np.clip(vals, lo, hi)
        elif (vals < lo).any() or (vals > hi).any():
            raise ValueError(
                f"The {dim} values must be within the [{lo:g}, "
                f"{hi:g}] range of the {dim} axis, got {values}."
            )
        if vals.ndim > 0:
            return da.interp({dim: xr.DataArray(vals, dims="cell")})
        return da.interp({dim: vals[()]}).drop_vars(dim)

    def get_xyz_grid(
        self,
    ) -> tuple[
        NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]
    ]:
        """Return the x, y and z cell-boundary arrays of the component
        field, from which the :class:`smartg.grid3d.Grid3D` can be
        built. Only available with the dataset input route.
        """
        if self.ds_dist is None:
            raise ValueError(
                f"The {self._label} grid is only known when the "
                f"{self._label} is provided as a dataset (ds parameter)"
            )
        return (
            self.ds_dist["x_bounds"].to_numpy(),
            self.ds_dist["y_bounds"].to_numpy(),
            self.ds_dist["z_bounds"].to_numpy(),
        )

    def get_cell_indices(self) -> NDArray[np.int32]:
        return self._cell_indices

    def get_ext_ref(self) -> NDArray[np.float64]:
        """Return the (N,) component extinction coefficients in km-1
        at the reference wavelength `w_ref`.
        """
        return self._ext_ref

    def _interp_bulk_cells(self, var: str) -> xr.DataArray:
        """The 'ext' or 'ssa' bulk variable interpolated at the
        per-cell parameter values, over ('cell', 'wav').
        """
        return self._interp_axis(
            self.ds_mix[var],
            self._bulk_axis,
            self._param,
            clamp=(self._param_oor == "clamp"),
        )

    def get_ext(self, wavelength: NDArray[np.floating]) -> NDArray[np.float64]:
        n_wavelength = len(wavelength)
        ext = np.zeros((n_wavelength, self._ext_ref.size), dtype=np.float64)
        ext_cells = self._interp_bulk_cells("ext")
        ext_ref0 = self._interp_axis(
            ext_cells, "wav", self.w_ref
        ).values
        for iw in range(0, n_wavelength):
            ext_factor = (
                self._interp_axis(ext_cells, "wav", wavelength[iw]).values
                / ext_ref0
            )
            ext[iw, :] = self._ext_ref * ext_factor
        return ext

    def get_ssa(self, wavelength: NDArray[np.floating]) -> NDArray[np.float64]:
        n_wavelength = len(wavelength)
        ssa = np.ones((n_wavelength, self._ext_ref.size), dtype=np.float64)
        if self.ssa_cst is not None:
            ssa[:, :] = self.ssa_cst
        else:
            ssa_cells = self._interp_bulk_cells("ssa")
            for iw in range(0, n_wavelength):
                ssa[iw, :] = self._interp_axis(
                    ssa_cells, "wav", wavelength[iw]
                ).values
        return ssa

    def native_theta(self) -> NDArray[np.float64]:
        """The scattering angles the component's tables carry.

        The grid of a user-supplied phase matrix when there is one,
        else the grid of the bulk file, in degrees.
        """
        if self.phase is not None:
            theta = self.phase.coords["theta_atm"].values
        else:
            theta = self.ds_mix.coords["theta"].values
        return union_theta_grid([theta.astype(np.float64)])

    def get_phase(self, n_theta: ThetaLike = 721) -> xr.DataArray:
        """Return the component phase matrix DataArray with the
        dimensions ``('wavelength_phase', <parameter>, 'nphamat',
        'theta_atm')``, the parameter axis being ``'reff'`` or
        ``'hum'``.

        The matrices keep the IQUV convention of the source file, the
        conversion into the parallel/perpendicular convention of the
        kernels being done by the run method.

        Parameters
        ----------
        n_theta : int, str or array_like, optional
            The number of equally spaced scattering angles, the angles
            themselves in degrees, or ``'native'`` for the angles the
            component's tables carry, see `native_theta`.
        """
        theta = (
            self.native_theta() if is_native_theta(n_theta)
            else as_theta_grid(n_theta)
        )
        n_theta = len(theta)

        # First check if we have already phase
        if self.phase is not None:
            if np.array_equal(
                self.phase.coords["theta_atm"].values, theta
            ):
                return self.phase
            else:
                return self.phase.interp(theta_atm=theta)

        pha = self.ds_mix["phase"].interp(theta=theta).transpose(
            "wav", self._bulk_axis, "stk", "theta"
        )
        n_wavelength, n_param, nstk = pha.shape[:3]

        pha_ = np.zeros((n_wavelength, n_param, 6, n_theta), dtype=np.float64)
        pha_[:, :, :nstk, :] = pha.values

        if nstk == 4:  # spherical particles
            pha_[:, :, 4, :] = pha_[:, :, 0, :]  # F22 = F11
            pha_[:, :, 5, :] = pha_[:, :, 2, :]  # F44 = F33

        return xr.DataArray(
            pha_,
            coords=[
                self.ds_mix["wav"].values,
                self.ds_mix[self._bulk_axis].values,
                np.arange(6),
                theta,
            ],
            dims=["wavelength_phase", self._bulk_axis, "nphamat", "theta_atm"],
            name="phase_atm",
        )

    def get_phase_set(
        self,
        wavelength_phase: NDArray[np.floating],
        n_theta: ThetaLike = 721,
    ) -> tuple[list[xr.DataArray], NDArray[np.int32], int]:
        param_unique = np.unique(self._param)
        n_unique = param_unique.size

        phase = self.get_phase(n_theta=n_theta)
        clamp = self._param_oor == "clamp"

        phases = []
        for i_wavelength in range(0, len(wavelength_phase)):
            phase_w = self._interp_axis(
                phase, "wavelength_phase", wavelength_phase[i_wavelength]
            )
            # Loop only on the unique parameter values
            for iparam in range(0, n_unique):
                phases.append(
                    self._interp_axis(
                        phase_w,
                        self._bulk_axis,
                        param_unique[iparam],
                        clamp=clamp,
                    )
                )

        # Obtain the correct indices from the unique-value phase bank
        cell_phase_index = np.full(
            self._param.size, np.nan, dtype=np.int32
        )
        for iparam in range(0, n_unique):
            cell_phase_index[
                np.squeeze(np.argwhere(self._param == param_unique[iparam]))
            ] = iparam

        return phases, cell_phase_index, n_unique


class Cloud3D(_Comp3DFile):
    """3D cloud component.

    The cloud bulk optical properties (spectral extinction, single
    scattering albedo and phase matrices as a function of the droplet
    effective radius) are read from a SMART-G cloud NetCDF file. The 3D
    distribution of the cloud (per-cell extinction at the reference
    wavelength `w_ref` and effective radius) is provided either as a
    dense ``xr.Dataset`` (or NetCDF file path) following the SMART-G 3D
    cloud schema, or as raw arrays.

    The dense dataset schema is:
        - ``ext(z, y, x)`` : extinction coefficient in km-1 at `w_ref`,
          0 in cloud-free cells,
        - ``reff(z, y, x)`` : droplet effective radius in um,
        - coordinates ``x_bounds(x_b)``, ``y_bounds(y_b)``,
          ``z_bounds(z_b)`` : the cell boundaries in km,
        - optionally the attribute ``w_ref`` (in nm).
    Legacy I3RC/IPRT ASCII cloud files can be converted to this schema
    with :func:`read_i3rc_cloud`.

    Parameters
    ----------
    fname : PathType
        Cloud smartg fname with the bulk optical properties, choice
        between: 'wc', 'ic_baum_asc', 'ic_baum_ghm' and 'ic_baum_sc'
        (or the path to a file with the same structure).
    w_ref : float or None, optional
        Reference wavelength (nm) at which the cloud extinction is
        given. If None, taken from the ``w_ref`` attribute of `ds`.
    ds : xr.Dataset or PathType or None, optional
        The 3D cloud field following the dense schema described above,
        or the path of a NetCDF file containing it.
    reff : array_like or None, optional
        Numpy 1D array with the cloud effective radii (um) of each
        cloudy cell. Ignored if `ds` is given.
    ext_ref : array_like or None, optional
        Numpy 1D array with the cloud extinction coefficients (km-1) at
        `w_ref` of each cloudy cell. Ignored if `ds` is given.
    cell_indices : ndarray or None, optional
        (N, 3) array with the (ix, iy, iz) indices of the cloudy cells
        on the inner 3D grid, following the 1-based IPRT convention.
        Ignored if `ds` is given.
    reff_acc : int or None, optional
        Decimal accuracy of reff; the reff values are rounded to this
        number of decimals. By default None, i.e. keep the values as
        provided.
    reff_min, reff_max : float or None, optional
        The reff values less than reff_min are replaced by reff_min.
        The same for values greater than reff_max.
    phase : DataArray or LUT or None, optional
        The cloud phase matrix depending on wavelength_phase, reff,
        nphamat and theta_atm (e.g. from
        :func:`smartg.phase.read_phase_cdf` or
        :func:`smartg.phase.read_phase_nc` with
        ``output_sg_ready=False``; 4 terms are completed into 6). If
        None, the phase matrices are computed from the bulk optical
        properties file.
    ssa_cst : float or None, optional
        Force the cloud single scattering albedo to this constant
        value. If None, the single scattering albedo is interpolated
        from the bulk optical properties file.
    """

    _param_name = "reff"
    _bulk_axis = "reff"
    _auxdata_subdir = ("clouds",)
    _param_oor = "raise"
    _label = "cloud"

    def __init__(
        self,
        fname: PathType,
        w_ref: float | None = None,
        ds: xr.Dataset | PathType | None = None,
        reff: NumericArrayLike | None = None,
        ext_ref: NumericArrayLike | None = None,
        cell_indices: NDArray[np.integer] | None = None,
        reff_acc: int | None = None,
        reff_min: float | None = None,
        reff_max: float | None = None,
        phase: xr.DataArray | LUT | None = None,
        ssa_cst: float | None = None,
    ) -> None:
        super().__init__(
            fname,
            w_ref=w_ref,
            ds=ds,
            param=reff,
            ext_ref=ext_ref,
            cell_indices=cell_indices,
            param_acc=reff_acc,
            param_min=reff_min,
            param_max=reff_max,
            phase=phase,
            ssa_cst=ssa_cst,
        )

    @property
    def reff(self) -> NDArray[np.float64]:
        """The (N,) per-cell droplet effective radii in um."""
        return self._param

    @reff.setter
    def reff(self, value: NDArray[np.float64]) -> None:
        self._param = value


class Aer3D(_Comp3DFile):
    """3D aerosol component.

    The aerosol bulk optical properties (spectral extinction, single
    scattering albedo and phase matrices as a function of the relative
    humidity) are read from a SMART-G OPAC aerosol NetCDF file. The 3D
    distribution of the aerosol (per-cell extinction at the reference
    wavelength `w_ref` and relative humidity) is provided either as a
    dense ``xr.Dataset`` (or NetCDF file path) following the SMART-G
    3D aerosol schema, or as raw arrays.

    The relative humidities are clamped to the humidity axis of the
    bulk file (0-99 % for the OPAC species), matching the 1D
    :class:`AerOPAC` behavior; hydrophobic species with a single
    humidity node (e.g. 'inso', 'soot') ignore `rh` entirely. The
    vertical-structure attributes of the OPAC files (``H_mix_min``,
    ..., ``Z_stra``) are ignored: the 3D field prescribes the per-cell
    extinction directly.

    The dense dataset schema is:
        - ``ext(z, y, x)`` : extinction coefficient in km-1 at `w_ref`,
          0 in aerosol-free cells,
        - ``rh(z, y, x)`` : relative humidity in percent,
        - coordinates ``x_bounds(x_b)``, ``y_bounds(y_b)``,
          ``z_bounds(z_b)`` : the cell boundaries in km,
        - optionally the attribute ``w_ref`` (in nm).
    Legacy I3RC/IPRT-style ASCII files can be converted to this schema
    with :func:`read_i3rc_aerosol`.

    Parameters
    ----------
    fname : PathType
        Aerosol smartg fname with the bulk optical properties: an
        OPAC mixture name as in :class:`AerOPAC` ('continental_clean',
        'continental_average', 'continental_polluted', 'urban',
        'desert', 'maritime_clean', 'maritime_polluted',
        'maritime_tropical', 'antarctic', 'arctic', ...), an OPAC
        single species ('waso', 'inso', 'soot', 'suso', ...), or the
        path to a file with the same structure.
    w_ref : float or None, optional
        Reference wavelength (nm) at which the aerosol extinction is
        given. If None, taken from the ``w_ref`` attribute of `ds`.
    ds : xr.Dataset or PathType or None, optional
        The 3D aerosol field following the dense schema described
        above, or the path of a NetCDF file containing it.
    rh : array_like or None, optional
        Numpy 1D array with the relative humidities (percent) of each
        aerosol cell. Ignored if `ds` is given.
    ext_ref : array_like or None, optional
        Numpy 1D array with the aerosol extinction coefficients (km-1)
        at `w_ref` of each aerosol cell. Ignored if `ds` is given.
    cell_indices : ndarray or None, optional
        (N, 3) array with the (ix, iy, iz) indices of the aerosol
        cells on the inner 3D grid, following the 1-based IPRT
        convention. Ignored if `ds` is given.
    rh_acc : int or None, optional
        Decimal accuracy of rh; the rh values are rounded to this
        number of decimals. By default None, i.e. keep the values as
        provided. Recommended for continuous rh fields, to keep the
        set of unique phase matrices small.
    rh_min, rh_max : float or None, optional
        The rh values less than rh_min are replaced by rh_min. The
        same for values greater than rh_max.
    phase : DataArray or LUT or None, optional
        The aerosol phase matrix depending on wavelength_phase, hum,
        nphamat and theta_atm (the humidity axis is named ``hum`` as
        in the OPAC files; e.g. from :func:`smartg.phase.read_phase_nc`
        or :func:`smartg.phase.read_phase_cdf` with
        ``output_sg_ready=False``; 4 terms are completed into 6). If
        None, the phase matrices are computed from the bulk optical
        properties file.
    ssa_cst : float or None, optional
        Force the aerosol single scattering albedo to this constant
        value. If None, the single scattering albedo is interpolated
        from the bulk optical properties file.
    """

    _param_name = "rh"
    _bulk_axis = "hum"
    _auxdata_subdir = ("aerosols", "OPAC", "mixtures")
    _param_oor = "clamp"
    _label = "aerosol"

    def __init__(
        self,
        fname: PathType,
        w_ref: float | None = None,
        ds: xr.Dataset | PathType | None = None,
        rh: NumericArrayLike | None = None,
        ext_ref: NumericArrayLike | None = None,
        cell_indices: NDArray[np.integer] | None = None,
        rh_acc: int | None = None,
        rh_min: float | None = None,
        rh_max: float | None = None,
        phase: xr.DataArray | LUT | None = None,
        ssa_cst: float | None = None,
    ) -> None:
        super().__init__(
            fname,
            w_ref=w_ref,
            ds=ds,
            param=rh,
            ext_ref=ext_ref,
            cell_indices=cell_indices,
            param_acc=rh_acc,
            param_min=rh_min,
            param_max=rh_max,
            phase=phase,
            ssa_cst=ssa_cst,
        )

    def _normalize_param(
        self, param: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # hydrophobic species (e.g. 'inso', 'soot') have a single
        # humidity node, and the lookups on a size-1 axis reject any
        # other value even with the clamping policy: clamp rh to it
        hum = self.ds_mix["hum"].values.astype(np.float64)
        if hum.size == 1:
            param = np.full_like(param, hum[0])
        return param

    @property
    def rh(self) -> NDArray[np.float64]:
        """The (N,) per-cell relative humidities in percent."""
        return self._param

    @rh.setter
    def rh(self, value: NDArray[np.float64]) -> None:
        self._param = value


def read_i3rc_cloud(
    fname: PathType,
    loc_xgrid: str | RealNumber = "centered",
    loc_ygrid: str | RealNumber = "centered",
) -> xr.Dataset:
    """Read an I3RC/IPRT ASCII 3D cloud file (e.g. cumulus.dat) and
    convert it to the dense SMART-G 3D cloud dataset expected by
    :class:`Cloud3D`.

    The ASCII format is: one comment row, a row with the number of
    cells ``Nx Ny Nz`` and a flag, a row with the cell sizes ``Dx Dy``
    followed by the ``Nz + 1`` z boundaries in km, then one row per
    cloudy cell with the 1-based ``ix iy iz`` indices, the extinction
    coefficient in km-1 and the effective radius in um.

    Parameters
    ----------
    fname : PathType
        File name with path location of the ASCII cloud file.
    loc_xgrid, loc_ygrid : str or scalar, optional
        Location of the x and y grids. By default a str: "centered"
        i.e. the grid center is at coordinate 0. Or give a scalar with
        the starting position of the grid.

    Returns
    -------
    xr.Dataset
        Dataset with the ``ext(z, y, x)`` and ``reff(z, y, x)``
        variables and the ``x_bounds``, ``y_bounds`` and ``z_bounds``
        cell-boundary coordinates.
    """
    return _read_i3rc_field(fname, "reff", loc_xgrid, loc_ygrid)


def read_i3rc_aerosol(
    fname: PathType,
    loc_xgrid: str | RealNumber = "centered",
    loc_ygrid: str | RealNumber = "centered",
) -> xr.Dataset:
    """Read an I3RC/IPRT-style ASCII 3D aerosol file and convert it to
    the dense SMART-G 3D aerosol dataset expected by :class:`Aer3D`.

    The ASCII format is the one of :func:`read_i3rc_cloud`, with the
    fifth column holding the relative humidity in percent instead of
    the effective radius: one comment row, a row with the number of
    cells ``Nx Ny Nz`` and a flag, a row with the cell sizes ``Dx Dy``
    followed by the ``Nz + 1`` z boundaries in km, then one row per
    aerosol cell with the 1-based ``ix iy iz`` indices, the extinction
    coefficient in km-1 and the relative humidity in percent.

    Parameters
    ----------
    fname : PathType
        File name with path location of the ASCII aerosol file.
    loc_xgrid, loc_ygrid : str or scalar, optional
        Location of the x and y grids. By default a str: "centered"
        i.e. the grid center is at coordinate 0. Or give a scalar with
        the starting position of the grid.

    Returns
    -------
    xr.Dataset
        Dataset with the ``ext(z, y, x)`` and ``rh(z, y, x)``
        variables and the ``x_bounds``, ``y_bounds`` and ``z_bounds``
        cell-boundary coordinates.
    """
    return _read_i3rc_field(fname, "rh", loc_xgrid, loc_ygrid)


def _read_i3rc_field(
    fname: PathType,
    param_name: str,
    loc_xgrid: str | RealNumber = "centered",
    loc_ygrid: str | RealNumber = "centered",
) -> xr.Dataset:
    """Read an I3RC/IPRT-style ASCII 3D field (rows of 1-based
    ``ix iy iz`` indices, extinction coefficient and per-cell
    parameter) into the dense SMART-G 3D dataset, with the fifth
    column stored as the `param_name` variable.
    """
    # Read only the needed information, the two first rows.
    # Be careful ! The second row have a greater dimension than the
    # first one. Then -> two steps of reading.
    contentA = pd.read_csv(
        fname, skiprows=1, nrows=1, header=None, sep=r"\s+", dtype=float
    ).values
    contentB = pd.read_csv(
        fname, skiprows=2, nrows=1, header=None, sep=r"\s+", dtype=float
    ).values

    # If there are empty dimensions remove them
    contentA = np.squeeze(contentA)
    contentB = np.squeeze(contentB)

    # Number of cells in x and y axes
    Nx = int(contentA[0])
    Ny = int(contentA[1])

    # Cell sizes in x and y axes
    Dx = contentB[0]
    Dy = contentB[1]

    # Create x and y grid
    xgrid = create_1d_grid(Nx, Dx, loc=loc_xgrid)
    ygrid = create_1d_grid(Ny, Dy, loc=loc_ygrid)

    # Grid in the z axis can be directly read from the file
    zgrid = contentB[2:]
    Nz = zgrid.size - 1

    content = pd.read_csv(
        fname, skiprows=3, header=None, sep=r"\s+", dtype=float
    ).values
    cell_indices = content[:, :3].astype(np.int32) - 1  # 1-based indices
    ix, iy, iz = cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2]

    ext = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    param = np.zeros_like(ext)
    ext[iz, iy, ix] = content[:, 3]
    param[iz, iy, ix] = content[:, 4]

    return xr.Dataset(
        {
            "ext": (("z", "y", "x"), ext),
            param_name: (("z", "y", "x"), param),
        },
        coords={
            "x_bounds": ("x_b", xgrid),
            "y_bounds": ("y_b", ygrid),
            "z_bounds": ("z_b", zgrid),
        },
        attrs={"source": str(fname)},
    )


class Atmosphere(object):
    """Base class for atmosphere."""

    def calc(self, wavelength, *args, **kwargs) -> xr.Dataset:
        """
        Compute the atmospheric profile as an xr.Dataset.

        Implemented by the subclasses.
        """
        raise NotImplementedError


class Atm1D(Atmosphere):
    """
    1D atmospheric profile definition

    The atmospheric profile is read from an auxiliary data file. The
    profiles currently shipped with the auxiliary data are the AFGL
    standard atmospheres, but this is not a limitation: any other
    profile can be used. Users can provide their own atmospheric
    profile, either by placing the file in the atmospheric auxdata
    directory (and referring to it by fname) or by passing the full
    path to the `fname` parameter.

    Parameters
    ----------

    fname : str
        The atmospheric profile to use. The AFGL standard
        atmospheres are provided in the auxiliary data:
            - 'afglms' for Mid-Latitude Summer (45N July)
            - 'afglmw' for Mid-Latitude Winter (45N Jan)
            - 'afglss' for Sub Arctic Summer (60N July)
            - 'afglsw' for Sub Arctic Winter (60N Jan)
            - 'afglt' for Tropic (15N Annual Average)
            - 'afglus' for U.S. Standard (1976)

        Any other profile may be used as well: provide your own file
        either by placing it in the atmospheric auxdata directory (and
        passing its fname) or by passing its full path.

        File format: If a full path is not provided (only fname), the
        atmospheric
        auxdata directory is automatically prepended to the path. The
        file extension
        defaults to '.nc' if not specified. Only '.nc' (NetCDF) and
        '.dat' file
        formats are accepted. For '.dat' files, the libratran atmosphere
        file
        convention is used.
    comp:  list, optional
        Components particles (aerosols or clouds ) to consider, i.e. a
        list of aerOPAC, Cloud or/and AerUser objects.
    grid : array_like or str or None, optional
      The vertical grid (from TOA to BOA). The optical properties of the
      atmosphere are recalculated following
      the new grid. If None, the grid of the input profile is kept.
      If a string is provided, it is interpreted as a compact grid
      specification and converted to a NumPy array via
      :func:`strgrid_to_numpy` (see that function for the supported
      format).
    lat : float, optional
        The latitude used for Rayleigh optical depth calculation.
        Default: 45.
    p0 : float or None, optional
        Sea surface (bottom layer) pressure in hPa. If None, uses the
        pressure from the input profile. Scales all pressure values
        proportionally.
        Default: None
    tco3 : float or None, optional
        Total column vertically-integrated ozone in Dobson units (DU).
        If None, uses the value from the input profile. The ozone
        profile is scaled to match this column amount.
        Note: 1 DU = 2.1415e-5 kg m⁻².
        Default: None
    tcwp : float or None, optional
        Total column vertically-integrated water vapour in g/cm². If
        None, uses the value from the input profile. The water vapour
        profile is scaled to match this column amount.
        Default: None
    no2 : bool, optional
        Activate NO2 absorption. If False, NO2 density is set to zero.
        If True, the NO2 profile from the atmospheric file is retained.
        Default: True
    o3_h2o_alt : float or None, optional
        Altitude (km) at which the specified tco3 and tcwp values apply.
        When specified, the ozone and water vapor profiles are scaled
        such that the column amount from TOA to this altitude matches
        the provided tco3 and tcwp values. The full gaseous distribution
        from TOA to ground is preserved; only the scaling factor is
        adjusted to match the constraint at this reference altitude.
        Default: None
    tau_r : float or array_like or None, optional
        Force the Rayleigh optical thickness. If None, computed from
        atmospheric profile and wavelength.
    wavelength_phase : array_like or None, optional
        The wavelengths over which the phase matrices are
        calculated. Then use the nearest wavelength
        during cuda simulation. Useful to reduce the memory. If None,
        compute the phase matrix at all wavelengths.
    pfgrid : array_like or None, optional
        The vertcial grid (from TOA to BOA) over which the phase
        matrices are calculated. This parameter can help
        reduce the memory but must be used with care. If misused, it may
        lead to innaccurate results especially when
        multiple aerosols are mixed. The default value is [100, 0],
        meaning a single phase matrix is calculated over
        the entire column from 0 to 100 km. This is effcient and
        accurate when only 1 type of aerosol is present.
        However, if multiple aerosols are mixed (with different vertical
        distributions), a single phase matrix may
        introduce an important bias.
    prof_abs : None or 2-D ndarray, optional
        Force the gaseous absorption optical thickness vertical profile
        (nwavelength,nz), it shortcuts any further gaseous absorption
        computation.
    prof_ray : None or 2-D ndarray, optional
        Force the Rayleigh scattering optical thickness vertical
        profile (nwavelength,nz), it shortcuts any further Rayleigh
        scattering computation.
    prof_aer : None or tuple, optional
        A tuple (ext,ssa) with the aerosol extinction optical thickness
        profile (ext) and single scattering albedo arrays (ssa), it
        shortcuts any further particles scattering computation.
    prof_phases : tuple or None, optional
        A tuple (iphase, phases ) where iphase is the phase matrix
        indices profile (nwavelength,nz), and phases is a list of phase
        matrices LUT (as outputs of the `read_phase` utility).
    rh_cst : float or None, optional
        Force relative humidity to be constant at this value. If None,
        relative humidity is recalculated from the temperature and water
        vapor profiles.
        Default: None
    o3_acs : str, optional
        Path to ozone netcdf4 file with absorption coefficient cross
        section (SIGMA = 1E-20 * [C0 + C1*T + C2*T^2],
        in cm^2, and where T is in degrees Celcius). If only fname is
        given automatically look at "auxdata/acs/".
        By default use Bogumil Version 3.0 data. Available files in
        auxdata:
            - 'O3_acs_BogumilV3.0_coeffs.nc'
            - 'O3_acs_Chehade(Bogumil_revised)V4.1_coeffs.nc'
            - 'O3_acs_SerdyuchenkoV2.0_coeffs.nc'
    no2_acs : str, optional
        Path to NO2 netcdf4 file with absorption coefficient cross
        section (SIGMA = 1E-20 * [C0 + C1*T + C2*T^2],
        in cm^2, and where T is in degrees Celcius). If only fname is
        given automatically look at "auxdata/acs/".
        By default use Bogumil Version 1.0 data. Available files in
        auxdata:
            - 'NO2_acs_BogumilV1.0_coeffs.nc'
            - 'NO2_acs_Bingen_coeffs.nc'
    """

    def __init__(
        self,
        fname: PathType,
        comp: list[AerOPAC] | None = None,
        grid: NumericArrayLike | str | None = None,
        lat: float = 45.0,
        p0: float | None = None,
        tco3: float | None = None,
        tcwp: float | None = None,
        no2: bool = True,
        o3_h2o_alt: float | None = None,
        tau_r: float | NumericArrayLike | None = None,
        wavelength_phase: NumericArrayLike | None = None,
        pfgrid: NumericArrayLike | None = None,
        prof_abs: NDArray[np.floating] | None = None,
        prof_ray: NDArray[np.floating] | None = None,
        prof_aer: (
            tuple[NDArray[np.floating], NDArray[np.floating]] | None
        ) = None,
        prof_phases: tuple[NDArray[np.integer], list[Any]] | None = None,
        rh_cst: float | None = None,
        o3_acs: PathType = "O3_acs_BogumilV3.0_coeffs",
        no2_acs: PathType = "NO2_acs_BogumilV1.0_coeffs",
    ) -> None:

        self.lat = lat
        self.comp = [] if comp is None else comp
        self.wavelength_phase = (
            None if wavelength_phase is None
            else np.asarray(wavelength_phase)
        )
        self.pfgrid = (
            np.array([100.0, 0.0]) if pfgrid is None else np.asarray(pfgrid)
        )
        assert (np.diff(self.pfgrid) < 0.0).all()
        self.prof_abs = prof_abs
        self.prof_ray = prof_ray
        self.prof_aer = prof_aer
        self.prof_phases = prof_phases
        # store attribute using lowercase name for consistency
        self.rh_cst = rh_cst
        # 3D mode: only enabled by the private _Atm3DBackend used by
        # Atm3D
        self.opt3d = False
        self.cells = None

        self.tau_r = np.asarray(tau_r) if tau_r is not None else None
        if isinstance(grid, str):
            grid = strgrid_to_numpy(grid)
        grid = np.asarray(grid) if grid is not None else None
        fname = Path(fname)

        #
        # init directories and read atm file
        #
        if fname.parent == Path("."):
            fname = DIR_AUXDATA / "atmospheres" / fname.name
        # By default if no suffix is given consider it as a netcdf
        # file
        if not fname.exists() and fname.suffix == "":
            fname = fname.with_name(fname.name + ".nc")

        if fname.suffix == ".nc" or fname.suffix == ".dat":
            prof = ProfileBase(
                fname,
                tco3=tco3,
                tcwp=tcwp,
                tcno2=no2,
                p0=p0,
                rh_cst=rh_cst,
                o3_h2o_alt=o3_h2o_alt,
            )
        else:
            raise ValueError(
                "This file format is not supported. Only '.nc' and"
                + " '.dat' are supported."
            )

        #
        # read gaseous acs
        #
        O3_acs_path = Path(o3_acs)
        if O3_acs_path.parent == Path("."):
            O3_acs_path = DIR_AUXDATA / "acs" / O3_acs_path.name
        if not O3_acs_path.exists() and O3_acs_path.suffix != ".nc":
            O3_acs_path = O3_acs_path.with_name(O3_acs_path.name + ".nc")
        self.acs_o3 = xr.open_dataset(O3_acs_path)
        self.acs_o3 = self.acs_o3.rename({"wav": "wavelength"})

        NO2_acs_path = Path(no2_acs)
        if NO2_acs_path.parent == Path("."):
            NO2_acs_path = DIR_AUXDATA / "acs" / NO2_acs_path.name
        if not NO2_acs_path.exists() and NO2_acs_path.suffix != ".nc":
            NO2_acs_path = NO2_acs_path.with_name(NO2_acs_path.name + ".nc")
        self.acs_no2 = xr.open_dataset(NO2_acs_path)
        self.acs_no2 = self.acs_no2.rename({"wav": "wavelength"})

        #
        # regrid profile if required
        #
        # keep the source profile so that Atm3D can regrid it onto the
        # 3D vertical grid
        self._prof_src = prof
        if grid is None:
            self.prof = prof
        else:
            self.prof = prof.regrid(grid)

        #
        # calculate reduced profile
        # (for phase function blending)
        #
        self.prof_red = prof.regrid(self.pfgrid)

    def calc(
        self,
        wavelength: NumericArrayLike | BandSet,
        phase: bool = True,
        n_theta: ThetaLike = 721,
        use_old_calc_iphase: bool = False,
        truncation: DMTrunc | GTTrunc | None = None,
    ) -> xr.Dataset:
        """
        Profile and phase matrix calculation at bands / wavelength

        Parameters
        ----------
        wavelength : array_like or BandSet
            Wavelengths at which to calculate the profile. It can be a
            list of ReptranIband or KdisIband.
        n_theta : int, str or array_like, optional
            The number of equally spaced angles to be considered for
            the phase matrix, the angles themselves in degrees,
            which `smartg.phase.theta_grid` can build clustered
            towards the forward and backward directions, or
            ``'native'`` for the union of the angles the components'
            tables carry, on which their mixture is exact; pass
            ``theta_grid='phase'`` to `Smartg.run` to keep that grid
            on the device.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (depracated).
        truncation : None or DMTrunc or GTTrunc, optional
            The scattering phase truncation to use.

        Returns
        -------
        out : Dataset
            An xarray Dataset object with the profile and (if phase =
            True) the phase matrices.
        """

        if not isinstance(wavelength, BandSet):
            wavelength = BandSet(wavelength)

        profile = self.profile(wavelength)

        if phase:
            if self.wavelength_phase is None:
                wavelength_pha = wavelength[:]
            else:
                wavelength_pha = self.wavelength_phase
            pha = self.phase(wavelength_pha, n_theta=n_theta)
            ipha = None

            pro_var = list(profile.data_vars)
            if pha is not None or (
                self.opt3d and ("phase_atm" in pro_var) and truncation
            ):
                if pha is not None:
                    pha_, ipha = calc_iphase(
                        pha,
                        profile.coords["wavelength"].values,
                        profile.coords["z_atm"].values,
                        use_old_calc_iphase,
                    )
                else:  # 3D ATM
                    pha_ = profile["phase_atm"].values

                nphase = pha_.shape[0]

                # If truncation parameter is given compute truncated
                # phase function
                f = None
                pha_tr: NDArray[np.float64] | None = None
                if truncation is not None:
                    if self.opt3d:
                        theta = profile.coords["theta_atm"].values
                    else:
                        assert pha is not None, (
                            "Truncation is only possible if phase "
                            + "matrix is provided in 1D atm mode"
                        )
                        theta = (
                            pha.coords["theta_atm"].values
                            if hasattr(pha, "coords")
                            else pha.axes[-1]
                        )
                    pha_tr = np.zeros(pha_.shape, dtype=np.float64)
                    nphac = pha_.shape[1]
                    # initialize truncation-related locals to avoid static
                    # analyzer warnings about possibly unbound variables
                    m_max = None
                    f_ = None
                    th_tol = None
                    l_opti = False
                    th_f = None

                    if isinstance(truncation, DMTrunc):
                        m_max = truncation.m_max
                    elif isinstance(truncation, GTTrunc):
                        f_ = truncation.trunc_frac
                        th_tol = truncation.theta_tol
                        l_opti = truncation.lobatto_optimization
                        th_f = truncation.theta_tr
                    else:
                        raise ValueError("truncation method not recognized")
                    method = truncation.integral_method
                    f_pha = np.zeros(nphase, dtype=np.float64)
                    for iph in range(nphase):
                        if isinstance(truncation, DMTrunc):
                            ds_pha = cast(
                                xr.Dataset,
                                delta_m_phase_approx(
                                    pha_[iph, 0, :],
                                    theta,
                                    m_max,
                                    method=method,
                                ),
                            )

                        elif isinstance(truncation, GTTrunc):
                            ds_pha = cast(
                                xr.Dataset,
                                gt_phase_approx(
                                    pha_[iph, 0, :],
                                    theta,
                                    f_,
                                    method=method,
                                    th_tol=th_tol,
                                    th_f=th_f,
                                    lobatto_optimization=l_opti,
                                ),
                            )
                        f11_tr = ds_pha["phase_tr"].values
                        f = ds_pha["f"].values
                        f_pha[iph] = f
                        # One truncation factor for the whole medium.
                        # Every phase matrix of the profile is truncated
                        # here, and the optical depth and the single
                        # scattering albedo are rescaled further down
                        # with this one scalar f over every cell at
                        # once, so a second factor would have nothing to
                        # rescale with.
                        #
                        # Two things are missing to truncate one layer
                        # or one voxel alone. Indexing f_pha by
                        # iphase_atm would give one f per cell and lift
                        # this guard, unchanged wherever a single factor
                        # is used today. That alone is not enough: a
                        # cell mixing a peaked component with a smooth
                        # one carries a single mixed phase matrix, so
                        # truncating it truncates both. Truncating only
                        # the peaked one asks for the truncation to be
                        # carried by the component, applied before the
                        # mixing, with the scattering coefficient of
                        # that component alone scaled by 1 - f.
                        if iph > 0 and not np.isclose(
                            f_pha[iph], f_pha[0], atol=1e-6
                        ):
                            raise ValueError(
                                f"Only one truncation factor is "
                                f"supported: the phase matrix {iph} "
                                f"truncates at f={float(f_pha[iph]):.4g} "
                                f"against {float(f_pha[0]):.4g} for the "
                                "first one. Truncate components that "
                                "truncate alike, or a single one."
                            )

                        pha_tr[iph, 0, :] = f11_tr
                        beta = pha_tr[iph, 0, :] / pha_[iph, 0, :]
                        for icomp in range(1, nphac):
                            pha_tr[iph, icomp, :] = pha_[iph, icomp, :] * beta
                        if truncation.pha_scale_method == 2:
                            beta2 = 1.0 / (1 - f)
                            pha_tr[iph, 1, :] = pha_[iph, 1, :] * beta2
                            pha_tr[iph, 3, :] = pha_[iph, 3, :] * beta2

                if not self.opt3d:
                    assert pha is not None
                    theta_atm = (
                        pha.coords["theta_atm"].values
                        if hasattr(pha, "coords")
                        else pha.axes[-1]
                    )
                    profile = profile.assign_coords(theta_atm=theta_atm)
                    profile["phase_atm"] = xr.DataArray(
                        pha_,
                        dims=["iphase", "nphamat", "theta_atm"],
                        coords={
                            "iphase": np.arange(pha_.shape[0]),
                            "nphamat": np.arange(pha_.shape[1]),
                            "theta_atm": theta_atm,
                        },
                    )
                    profile["iphase_atm"] = xr.DataArray(
                        ipha,
                        dims=["wavelength", "z_atm"],
                        coords={
                            "wavelength": profile.coords["wavelength"],
                            "z_atm": profile.coords["z_atm"],
                        },
                    )
                else:
                    # In 3D the phase matrices and their indices come
                    # from the profile itself, so pha and ipha are None
                    attrs_tmp = (
                        profile["phase_atm"].attrs.copy()
                        if "phase_atm" in profile.data_vars
                        else {}
                    )
                    if "phase_atm" in profile.data_vars:
                        profile = profile.drop_vars("phase_atm")
                    theta_atm = (
                        pha.coords["theta_atm"].values
                        if hasattr(pha, "coords")
                        else as_theta_grid(pha_.shape[-1])
                    )
                    profile["phase_atm"] = xr.DataArray(
                        pha_,
                        dims=["iphase", "nphamat", "theta_atm"],
                        coords={
                            "iphase": np.arange(pha_.shape[0]),
                            "nphamat": np.arange(pha_.shape[1]),
                            "theta_atm": theta_atm,
                        },
                        attrs=attrs_tmp,
                    )

                if truncation is not None:
                    assert pha_tr is not None
                    assert f is not None
                    # profile.add_dataset('phase_atm_tr', pha_tr,
                    # axnames=['iphase', 'stk', 'theta_atm'])
                    attrs_tmp = profile["phase_atm"].attrs
                    if "phase_atm" in profile.data_vars:
                        profile = profile.drop_vars("phase_atm")
                    # the truncated matrix comes back on the grid
                    # it was given, so keep that grid
                    assert len(theta) == pha_tr.shape[-1]
                    theta_atm = theta
                    profile["phase_atm"] = xr.DataArray(
                        pha_tr,
                        dims=["iphase", "nphamat", "theta_atm"],
                        coords={
                            "iphase": np.arange(pha_tr.shape[0]),
                            "nphamat": np.arange(pha_tr.shape[1]),
                            "theta_atm": theta_atm,
                        },
                        attrs=attrs_tmp,
                    )

                    # case tau instead of coeff (1D atm)
                    if not self.opt3d:
                        dtau_p = diff1(profile["OD_p"].values, axis=1)
                        dtau_p_tr = (
                            1 - f * profile["ssa_p_atm"].values
                        ) * dtau_p
                        tau_p_tr = np.cumsum(dtau_p_tr, axis=1)
                        ssa_p_atm_tr = profile["ssa_p_atm"].values * (
                            (1 - f) / (1 - f * profile["ssa_p_atm"].values)
                        )
                        tau_atm_tr = (
                            tau_p_tr
                            + profile["OD_r"].values
                            + profile["OD_g"].values
                        )
                        dtau_r = diff1(profile["OD_r"].values, axis=1)
                        tau_sca_tr = np.cumsum(
                            dtau_r + dtau_p_tr * ssa_p_atm_tr, axis=1
                        )
                        with np.errstate(invalid="ignore", divide="ignore"):
                            ssa_atm_tr = (
                                dtau_r + dtau_p_tr * ssa_p_atm_tr
                            ) / diff1(tau_atm_tr, axis=1)
                        ssa_atm_tr[np.isnan(ssa_atm_tr)] = 1.0
                        with np.errstate(invalid="ignore", divide="ignore"):
                            pmol_tr = dtau_r / (
                                dtau_r + dtau_p_tr * ssa_p_atm_tr
                            )
                        pmol_tr[np.isnan(pmol_tr)] = 1.0

                        profile["OD_p"] = xr.DataArray(
                            tau_p_tr,
                            dims=["wavelength", "z_atm"],
                            coords={
                                "wavelength": profile.coords["wavelength"],
                                "z_atm": profile.coords["z_atm"],
                            },
                            attrs=profile["OD_p"].attrs,
                        )
                        profile["ssa_p_atm"] = xr.DataArray(
                            ssa_p_atm_tr,
                            dims=["wavelength", "z_atm"],
                            coords={
                                "wavelength": profile.coords["wavelength"],
                                "z_atm": profile.coords["z_atm"],
                            },
                            attrs=profile["ssa_p_atm"].attrs,
                        )
                        profile["OD_atm"] = xr.DataArray(
                            tau_atm_tr,
                            dims=["wavelength", "z_atm"],
                            coords={
                                "wavelength": profile.coords["wavelength"],
                                "z_atm": profile.coords["z_atm"],
                            },
                            attrs=profile["OD_atm"].attrs,
                        )
                        profile["OD_sca_atm"] = xr.DataArray(
                            tau_sca_tr,
                            dims=["wavelength", "z_atm"],
                            coords={
                                "wavelength": profile.coords["wavelength"],
                                "z_atm": profile.coords["z_atm"],
                            },
                            attrs=profile["OD_sca_atm"].attrs,
                        )
                        profile["ssa_atm"] = xr.DataArray(
                            ssa_atm_tr,
                            dims=["wavelength", "z_atm"],
                            coords={
                                "wavelength": profile.coords["wavelength"],
                                "z_atm": profile.coords["z_atm"],
                            },
                            attrs=profile["ssa_atm"].attrs,
                        )
                        profile["pmol_atm"] = xr.DataArray(
                            pmol_tr,
                            dims=["wavelength", "z_atm"],
                            coords={
                                "wavelength": profile.coords["wavelength"],
                                "z_atm": profile.coords["z_atm"],
                            },
                            attrs=profile["pmol_atm"].attrs,
                        )
                    # case coeff instead of tau (3D atm)
                    # sig for coeficients
                    else:
                        sig_p = profile["OD_p"].values
                        sig_p_tr = (
                            1 - f * profile["ssa_p_atm"].values
                        ) * sig_p
                        ssa_p_atm_tr = profile["ssa_p_atm"].values * (
                            (1 - f) / (1 - f * profile["ssa_p_atm"].values)
                        )
                        sig_atm_tr = (
                            sig_p_tr
                            + profile["OD_r"].values
                            + profile["OD_g"].values
                        )
                        sig_sca_tr = (
                            profile["OD_r"].values + sig_p_tr * ssa_p_atm_tr
                        )
                        with np.errstate(invalid="ignore", divide="ignore"):
                            ssa_atm_tr = (
                                profile["OD_r"].values
                                + sig_p_tr * ssa_p_atm_tr
                            ) / sig_atm_tr
                        ssa_atm_tr[np.isnan(ssa_atm_tr)] = 1.0
                        sig_r = profile["OD_r"].values
                        with np.errstate(invalid="ignore", divide="ignore"):
                            pmol_tr = sig_r / (sig_r + sig_p_tr * ssa_p_atm_tr)
                        pmol_tr[np.isnan(pmol_tr)] = 1.0

                        profile["OD_p"] = xr.DataArray(
                            sig_p_tr,
                            dims=["wavelength", "iopt"],
                            coords={
                                "wavelength": profile.coords["wavelength"]
                            },
                            attrs=profile["OD_p"].attrs,
                        )
                        profile["ssa_p_atm"] = xr.DataArray(
                            ssa_p_atm_tr,
                            dims=["wavelength", "iopt"],
                            coords={
                                "wavelength": profile.coords["wavelength"]
                            },
                            attrs=profile["ssa_p_atm"].attrs,
                        )
                        profile["OD_atm"] = xr.DataArray(
                            sig_atm_tr,
                            dims=["wavelength", "iopt"],
                            coords={
                                "wavelength": profile.coords["wavelength"]
                            },
                            attrs=profile["OD_atm"].attrs,
                        )
                        profile["OD_sca_atm"] = xr.DataArray(
                            sig_sca_tr,
                            dims=["wavelength", "iopt"],
                            coords={
                                "wavelength": profile.coords["wavelength"]
                            },
                            attrs=profile["OD_sca_atm"].attrs,
                        )
                        profile["ssa_atm"] = xr.DataArray(
                            ssa_atm_tr,
                            dims=["wavelength", "iopt"],
                            coords={
                                "wavelength": profile.coords["wavelength"]
                            },
                            attrs=profile["ssa_atm"].attrs,
                        )
                        profile["pmol_atm"] = xr.DataArray(
                            pmol_tr,
                            dims=["wavelength", "iopt"],
                            coords={
                                "wavelength": profile.coords["wavelength"]
                            },
                            attrs=profile["pmol_atm"].attrs,
                        )

        return profile

    def profile(
        self,
        wavelength: NumericArrayLike | BandSet,
        prof: ProfileBase | None = None,
    ) -> xr.Dataset:
        """Calculate the profile of optical properties at given
        wavelengths.

        Computes atmospheric optical properties (extinction, scattering,
        absorption)
        as a function of wavelength and altitude, including
        contributions from
        Rayleigh scattering, aerosols, and gaseous absorbers (O3, NO2,
        and molecular gases).

        Parameters
        ----------
        wavelength : array_like or BandSet
            Wavelengths at which to calculate optical properties [nm].
            If not a BandSet, it will be converted to one.
        prof : ProfileBase, optional
            Atmospheric profile containing altitude grids, temperature,
            pressure,
            and density profiles. Default is None; uses self.prof if not
            provided.

        Returns
        -------
        profile : Dataset
            Xarray dataset containing atmospheric optical properties
            with dimensions as a function of wavelength and altitude (or
            iopt grid for 3D mode).

            Key datasets included:

            - **n_atm**: Atmospheric refractive index [wavelength,
              z_atm]
            - **T_atm**: Temperature profile [z_atm] (K)
            - **OD_r**: Rayleigh cumulated optical thickness
              [wavelength, z_atm]
              or scattering coefficient (km⁻¹) in 3D mode
            - **OD_p**: Particles cumulated optical thickness
              [wavelength, z_atm]
              or extinction coefficient (km⁻¹) in 3D mode
            - **ssa_p_atm**: Particle single scattering albedo
              [wavelength, z_atm] or [wavelength, iopt]
            - **OD_g**: Cumulated gaseous absorption optical thickness
              [wavelength, z_atm]
              or absorption coefficient (km⁻¹) in 3D mode
            - **OD_atm**: Total cumulated optical thickness [wavelength,
              z_atm]
              or total extinction coefficient (km⁻¹) in 3D mode
            - **OD_sca_atm**: Cumulated scattering optical thickness
              [wavelength, z_atm]
            - **OD_abs_atm**: Cumulated absorption optical thickness
              [wavelength, z_atm]
            - **ssa_atm**: Total single scattering albedo [wavelength,
              z_atm]

        Notes
        -----
        The method can operate in two modes:

                - **1D Mode (opt3d=False)**: Returns cumulated optical
          thicknesses with axes
          [wavelength, z_atm]
                - **3D Mode (opt3d=True)**: Returns extinction/absorption
          coefficients with axes
          [wavelength, iopt] for use in 3D radiative transfer
          calculations. The dataset carries no ``z_atm`` axis: the
          ``iopt`` axis is not a vertical dependence but the set of
          unique optical properties (the 1D levels plus one entry
          per 3D component cell).

        Optical properties include:

        - Rayleigh scattering (from self.prof_ray or computed using
          Rayleigh optical depth)
        - Aerosol scattering and absorption from aerosol components
        - Gaseous absorption from ozone, NO2, and molecular gases (H2O,
          O2, CO2)
          using cross-sections (acs_o3, acs_no2) or REPTRAN/KDIS
          spectral databases

        Single scattering albedo is calculated as the ratio of
        scattering to extinction
        optical thicknesses for each layer.
        """
        if not isinstance(wavelength, BandSet):
            wavelength = BandSet(wavelength)

        if prof is None:
            prof = self.prof

        dz = -diff1(prof.z)

        if self.opt3d:
            # in 3D mode the property axis is not a vertical
            # dependence but the set of unique optical properties
            # (the 1D levels plus one entry per 3D component cell)
            pro = xr.Dataset(
                coords={
                    "iopt": np.arange(len(prof.z)),
                    "wavelength": wavelength[:],
                }
            )
        else:
            pro = xr.Dataset(
                coords={"z_atm": prof.z, "wavelength": wavelength[:]}
            )

        if self.opt3d and self.prof_ray is not None:
            ray_coef = np.zeros_like(self.prof_ray)
        else:
            ray_coef = np.zeros(
                (len(wavelength), len(prof.z)), dtype="float32"
            )
        if self.opt3d and self.prof_aer is not None:
            aer_coef = np.zeros_like(self.prof_aer[0])
        else:
            aer_coef = np.zeros(
                (len(wavelength), len(prof.z)), dtype="float32"
            )
        if self.opt3d and self.prof_abs is not None:
            abs_coef = np.zeros_like(self.prof_abs)
        else:
            abs_coef = np.zeros(
                (len(wavelength), len(prof.z)), dtype="float32"
            )

        # refractive index
        n = refractivity(
            wavelength[:] * 1e-3,
            prof.p,
            prof.t,
            np.divide(
                prof.dens_co2 * 1e6,
                prof.dens_air,
                out=np.zeros_like(prof.dens_co2),
                where=prof.dens_air != 0,
            ),
        )

        zdim = "iopt" if self.opt3d else "z_atm"
        pro["n_atm"] = xr.DataArray(
            n,
            dims=["wavelength", zdim],
            coords={
                "wavelength": pro.coords["wavelength"],
                zdim: pro.coords[zdim],
            },
            attrs={"description": "atmospheric refractive index"},
        )

        pro["T_atm"] = xr.DataArray(
            prof.t,
            dims=[zdim],
            coords={zdim: pro.coords[zdim]},
            attrs={"description": "temperature (K)"},
        )

        #
        # Rayleigh optical thickness
        #
        # cumulated Rayleigh optical thickness (wavelength, z)
        if self.prof_ray is None:
            tauray = rayleigh_od(
                wavelength[:] * 1e-3,
                prof.dens_co2 / prof.dens_air * 1e6,
                self.lat,
                prof.z * 1e3,
                prof.p,
            )
            dtaur = diff1(tauray, axis=1)
        else:
            dtaur = self.prof_ray
            tauray = np.cumsum(dtaur, axis=1)

        if self.tau_r is not None:
            # scale Rayleigh optical thickness
            if self.tau_r.ndim == 1:
                # for each wavelength
                tauray *= self.tau_r[:, None] / tauray[:, -1:]
            else:
                # scalar
                tauray *= self.tau_r / tauray[:, -1:]

        assert tauray.ndim == 2

        # Rayleigh optical thickness
        dtaur = diff1(tauray, axis=1)
        if not self.opt3d:
            pro["OD_r"] = xr.DataArray(
                tauray,
                dims=["wavelength", "z_atm"],
                coords={
                    "wavelength": pro.coords["wavelength"],
                    "z_atm": pro.coords["z_atm"],
                },
                attrs={"description": "Cumulated rayleigh optical thickness"},
            )
        else:
            if self.prof_ray is None:
                ray_coef = abs(dtaur / dz)
                ray_coef[~np.isfinite(ray_coef)] = 0.0
            else:
                ray_coef = self.prof_ray
            pro["OD_r"] = xr.DataArray(
                ray_coef,
                dims=["wavelength", "iopt"],
                coords={"wavelength": pro.coords["wavelength"]},
                attrs={
                    "description": "rayleigh scattering coefficient (km-1)"
                },
            )

        #
        # Aerosol optical thickness and single scattering albedo
        #
        if self.prof_aer is None:
            dtaua = np.zeros((len(wavelength), len(prof.z)), dtype="float32")
            ssa_p = np.zeros((len(wavelength), len(prof.z)), dtype="float32")
            for comp in self.comp:
                dtau_, ssa_ = comp.dtau_ssa(
                    wavelength[:], prof.z, prof.relative_humidity()
                )
                dtaua += dtau_
                ssa_p += dtau_ * ssa_
            ssa_p[dtaua != 0] /= dtaua[dtaua != 0]
            ssa_p[dtaua == 0] = 1.0
            taua = np.cumsum(dtaua, axis=1)

        else:
            (dtaua, ssa_p) = self.prof_aer
            taua = np.cumsum(dtaua, axis=1)

        if not self.opt3d:
            pro["OD_p"] = xr.DataArray(
                taua,
                dims=["wavelength", "z_atm"],
                coords={
                    "wavelength": pro.coords["wavelength"],
                    "z_atm": pro.coords["z_atm"],
                },
                attrs={
                    "description": "Cumulated particles optical thickness at "
                    + "each wavelength"
                },
            )
        else:
            if self.prof_aer is None:
                aer_coef = abs(dtaua / dz)
                aer_coef[~np.isfinite(aer_coef)] = 0.0
            else:
                (aer_coef, ssa_p) = self.prof_aer
            pro["OD_p"] = xr.DataArray(
                aer_coef,
                dims=["wavelength", "iopt"],
                coords={"wavelength": pro.coords["wavelength"]},
                attrs={
                    "description": "particles extinction coefficient (km-1)"
                },
            )

        if not self.opt3d:
            pro["ssa_p_atm"] = xr.DataArray(
                ssa_p,
                dims=["wavelength", "z_atm"],
                coords={
                    "wavelength": pro.coords["wavelength"],
                    "z_atm": pro.coords["z_atm"],
                },
                attrs={
                    "description": "Particles single scattering albedo of the "
                    + "layer"
                },
            )
        else:
            pro["ssa_p_atm"] = xr.DataArray(
                ssa_p,
                dims=["wavelength", "iopt"],
                coords={"wavelength": pro.coords["wavelength"]},
                attrs={
                    "description": "Particles single scattering albedo of the "
                    + "layer"
                },
            )

        if self.prof_abs is None:
            #
            # Ozone optical thickness
            #

            # Consider gaseous from reptran/kdis
            use_o3_acs = True
            use_no2_acs = True
            tau_o3 = np.zeros((len(wavelength), len(prof.z)), dtype="float32")
            tau_no2 = np.zeros((len(wavelength), len(prof.z)), dtype="float32")
            if wavelength.use_reptran_kdis:
                tau_mol = wavelength.calc_profile(self.prof) * dz
                # If not reptran (i.e. Kdis case) we set 03 and NO2 to 0
                # (already calculated in Kdis)
                if not (
                    str(wavelength.type_wavelength)
                    == "<class 'smartg.reptran.ReptranIband'>"
                ):
                    assert wavelength.data is not None
                    kdis_iband = cast("KdisIband", wavelength.data[0])
                    all_kdis_gas = (
                        kdis_iband.band.kdis.species
                        + kdis_iband.band.kdis.species_c
                    )
                    if "no2" in all_kdis_gas:
                        use_no2_acs = False
                    if "o3" in all_kdis_gas:
                        use_o3_acs = False
            else:
                tau_mol = (
                    np.zeros(
                        (len(wavelength), len(prof.z)), dtype="float32"
                    ) * dz
                )

            # Compute o3 and no2 (if kdis only compute them if not
            # already computed)
            if use_no2_acs or use_o3_acs:
                # Commun part
                t0 = 273.15  # in K
                t = prof.t[None, :]  # temperature variability in z
                if use_o3_acs:
                    # O3 optical thickness
                    min_wavelength = float(
                        np.min(self.acs_o3["wavelength"].values)
                    )
                    max_wavelength = float(
                        np.max(self.acs_o3["wavelength"].values)
                    )
                    wavelength_query = xr.DataArray(
                        wavelength[:], dims=["wavelength"]
                    )
                    c0 = (
                        self.acs_o3["O3_C0"]
                        .sel(wavelength=wavelength_query, method="nearest")
                        .values[:, None]
                    )
                    c1 = (
                        self.acs_o3["O3_C1"]
                        .sel(wavelength=wavelength_query, method="nearest")
                        .values[:, None]
                    )
                    c2 = (
                        self.acs_o3["O3_C2"]
                        .sel(wavelength=wavelength_query, method="nearest")
                        .values[:, None]
                    )
                    tau_o3 = c0 + c1 * (t - t0) + c2 * (t - t0) * (t - t0)
                    tau_o3[
                        ~np.logical_and(
                            wavelength[:] > min_wavelength,
                            wavelength[:] < max_wavelength,
                        )
                    ] = 0.0
                    tau_o3 *= (
                        prof.dens_o3 * 1e-15
                    )  # ACS in 10^(-20) cm2, convert in km-1
                    tau_o3 *= dz
                    tau_o3[tau_o3 < 0] = 0
                if use_no2_acs:
                    # NO2 optical thickness
                    min_wavelength = float(
                        np.min(self.acs_no2["wavelength"].values)
                    )
                    max_wavelength = float(
                        np.max(self.acs_no2["wavelength"].values)
                    )
                    wavelength_query = xr.DataArray(
                        wavelength[:], dims=["wavelength"]
                    )
                    c0 = (
                        self.acs_no2["NO2_C0"]
                        .sel(wavelength=wavelength_query, method="nearest")
                        .values[:, None]
                    )
                    c1 = (
                        self.acs_no2["NO2_C1"]
                        .sel(wavelength=wavelength_query, method="nearest")
                        .values[:, None]
                    )
                    c2 = (
                        self.acs_no2["NO2_C2"]
                        .sel(wavelength=wavelength_query, method="nearest")
                        .values[:, None]
                    )
                    tau_no2 = c0 + c1 * (t - t0) + c2 * (t - t0) * (t - t0)
                    tau_no2[
                        ~np.logical_and(
                            wavelength[:] > min_wavelength,
                            wavelength[:] < max_wavelength,
                        )
                    ] = 0.0
                    tau_no2 *= (
                        prof.dens_no2 * 1e-15
                    )  # ACS in 10^(-20) cm2, convert in km-1
                    tau_no2 *= dz
                    tau_no2[tau_no2 < 0] = 0

            #
            # Total gaseous optical thickness
            #
            dtaug = tau_o3 + tau_no2 + tau_mol
            taug = np.cumsum(dtaug, axis=1)

            if not self.opt3d:
                pro["OD_g"] = xr.DataArray(
                    taug,
                    dims=["wavelength", "z_atm"],
                    coords={
                        "wavelength": pro.coords["wavelength"],
                        "z_atm": pro.coords["z_atm"],
                    },
                    attrs={
                        "description": "Cumulated gaseous absorption optical "
                        + "thickness"
                    },
                )
            else:
                abs_coef = abs(dtaug / dz)
                abs_coef[~np.isfinite(abs_coef)] = 0.0
                pro["OD_g"] = xr.DataArray(
                    abs_coef,
                    dims=["wavelength", "iopt"],
                    coords={"wavelength": pro.coords["wavelength"]},
                    attrs={
                        "description": "gaseous absorption coefficient (km-1)"
                    },
                )

        else:
            dtaug = self.prof_abs
            taug = np.cumsum(dtaug, axis=1)
            if not self.opt3d:
                pro["OD_g"] = xr.DataArray(
                    taug,
                    dims=["wavelength", "z_atm"],
                    coords={
                        "wavelength": pro.coords["wavelength"],
                        "z_atm": pro.coords["z_atm"],
                    },
                    attrs={
                        "description": "Cumulated gaseous absorption optical "
                        + "thickness"
                    },
                )

            else:
                abs_coef = self.prof_abs
                pro["OD_g"] = xr.DataArray(
                    abs_coef,
                    dims=["wavelength", "iopt"],
                    coords={"wavelength": pro.coords["wavelength"]},
                    attrs={
                        "description": "gaseous absorption coefficient (km-1)"
                    },
                )

        #
        # Total optical thickness and other parameters
        #
        if not self.opt3d:
            tau_tot = tauray + taua + taug[:, :]
            pro["OD_atm"] = xr.DataArray(
                tau_tot,
                dims=["wavelength", "z_atm"],
                coords={
                    "wavelength": pro.coords["wavelength"],
                    "z_atm": pro.coords["z_atm"],
                },
                attrs={
                    "description": "Cumulated extinction optical thickness"
                },
            )

            tau_sca = np.cumsum(dtaur + dtaua * ssa_p, axis=1)
            pro["OD_sca_atm"] = xr.DataArray(
                tau_sca,
                dims=["wavelength", "z_atm"],
                coords={
                    "wavelength": pro.coords["wavelength"],
                    "z_atm": pro.coords["z_atm"],
                },
                attrs={
                    "description": "Cumulated scattering optical thickness"
                },
            )

            tau_abs = np.cumsum(dtaug[:, :] + dtaua * (1 - ssa_p), axis=1)
            pro["OD_abs_atm"] = xr.DataArray(
                tau_abs,
                dims=["wavelength", "z_atm"],
                coords={
                    "wavelength": pro.coords["wavelength"],
                    "z_atm": pro.coords["z_atm"],
                },
                attrs={
                    "description": "Cumulated absorption optical "
                    + "thickness"
                },
            )

            with np.errstate(invalid="ignore", divide="ignore"):
                ssa = (dtaur + dtaua * ssa_p) / diff1(tau_tot, axis=1)
            ssa[np.isnan(ssa)] = 1.0
            pro["ssa_atm"] = xr.DataArray(
                ssa,
                dims=["wavelength", "z_atm"],
                coords={
                    "wavelength": pro.coords["wavelength"],
                    "z_atm": pro.coords["z_atm"],
                },
                attrs={"description": "Single scattering albedo of the layer"},
            )

        else:
            tot_coef = ray_coef + aer_coef + abs_coef[:, :]
            pro["OD_atm"] = xr.DataArray(
                tot_coef,
                dims=["wavelength", "iopt"],
                coords={"wavelength": pro.coords["wavelength"]},
                attrs={"description": "extinction coefficient (km-1)"},
            )

            sca_coef = ray_coef + aer_coef * ssa_p
            pro["OD_sca_atm"] = xr.DataArray(
                sca_coef,
                dims=["wavelength", "iopt"],
                coords={"wavelength": pro.coords["wavelength"]},
                attrs={"description": "scattering coefficient (km-1)"},
            )

            tabs_coef = abs_coef + aer_coef * (1.0 - ssa_p)
            pro["OD_abs_atm"] = xr.DataArray(
                tabs_coef,
                dims=["wavelength", "iopt"],
                coords={"wavelength": pro.coords["wavelength"]},
                attrs={"description": "total absorption coefficient (km-1)"},
            )

            with np.errstate(invalid="ignore", divide="ignore"):
                ssa = (ray_coef + aer_coef * ssa_p) / tot_coef
            ssa[np.isnan(ssa)] = 1.0
            pro["ssa_atm"] = xr.DataArray(
                ssa,
                dims=["wavelength", "iopt"],
                coords={"wavelength": pro.coords["wavelength"]},
                attrs={"description": "Single scattering albedo of the layer"},
            )

        with np.errstate(invalid="ignore", divide="ignore"):
            pmol = dtaur / (dtaur + dtaua * ssa_p)
        pmol[np.isnan(pmol)] = 1.0
        if not self.opt3d:
            pro["pmol_atm"] = xr.DataArray(
                pmol,
                dims=["wavelength", "z_atm"],
                coords={
                    "wavelength": pro.coords["wavelength"],
                    "z_atm": pro.coords["z_atm"],
                },
                attrs={
                    "description": "Ratio of molecular scattering to total "
                    + "scattering of the layer"
                },
            )
        else:
            pro["pmol_atm"] = xr.DataArray(
                pmol,
                dims=["wavelength", "iopt"],
                coords={"wavelength": pro.coords["wavelength"]},
                attrs={
                    "description": "Ratio of molecular scattering to total "
                    + "scattering of the layer"
                },
            )

        pine = np.zeros_like(ssa)
        fqy1 = np.zeros_like(ssa)
        if not self.opt3d:
            pro["pine_atm"] = xr.DataArray(
                pine,
                dims=["wavelength", "z_atm"],
                coords={
                    "wavelength": pro.coords["wavelength"],
                    "z_atm": pro.coords["z_atm"],
                },
                attrs={
                    "description": "fraction of inelastic scattering of the "
                    + "layer"
                },
            )
            pro["FQY1_atm"] = xr.DataArray(
                fqy1,
                dims=["wavelength", "z_atm"],
                coords={
                    "wavelength": pro.coords["wavelength"],
                    "z_atm": pro.coords["z_atm"],
                },
                attrs={
                    "description": "fluoresence quantum yield of the "
                    + "layer"
                },
            )
        else:
            pro["pine_atm"] = xr.DataArray(
                pine,
                dims=["wavelength", "iopt"],
                coords={"wavelength": pro.coords["wavelength"]},
                attrs={
                    "description": "fraction of inelastic scattering of the "
                    + "layer"
                },
            )
            pro["FQY1_atm"] = xr.DataArray(
                fqy1,
                dims=["wavelength", "iopt"],
                coords={"wavelength": pro.coords["wavelength"]},
                attrs={
                    "description": "fluoresence quantum yield of the "
                    + "layer"
                },
            )

        if self.prof_phases is not None:
            ipha, phases = self.prof_phases
            if not self.opt3d:
                pro["iphase_atm"] = xr.DataArray(
                    ipha,
                    dims=["wavelength", "z_atm"],
                    coords={
                        "wavelength": pro.coords["wavelength"],
                        "z_atm": pro.coords["z_atm"],
                    },
                    attrs={"description": "index of phase matrix"},
                )
            else:
                pro["iphase_atm"] = xr.DataArray(
                    ipha,
                    dims=["wavelength", "iopt"],
                    coords={"wavelength": pro.coords["wavelength"]},
                    attrs={"description": "index of phase matrix"},
                )

            # convert legacy LUT to DataArray objects
            phases = [
                x.to_xarray() if hasattr(x, "to_xarray") else x for x in phases
            ]
            # bring every matrix onto the union of the distinct grids
            # they carry, which loses no node of any of them
            distinct: list[NDArray[np.float64]] = []
            for p in phases:
                grid = p.coords["theta_atm"].values.astype(np.float64)
                if not any(np.array_equal(grid, g) for g in distinct):
                    distinct.append(grid)
            theta, resampled = _common_theta_grid(
                distinct, [f"phase matrices on {len(g)} angles"
                           for g in distinct],
            )
            if resampled:
                phases = [_on_theta_grid(p, theta) for p in phases]
            pha = np.stack([p.values for p in phases]).astype(np.float64)
            pro = pro.assign_coords(theta_atm=theta)
            pro["phase_atm"] = xr.DataArray(
                pha,
                dims=["iphase", "nphamat", "theta_atm"],
                coords={
                    "iphase": np.arange(pha.shape[0]),
                    "nphamat": np.arange(pha.shape[1]),
                    "theta_atm": pro.coords["theta_atm"],
                },
                attrs={"description": "phase matrices"},
            )
        # Pure 3D
        #
        if self.opt3d:
            assert self.cells is not None
            (iopt, iabs, pmin, pmax, neighbour) = self.cells
            pro["iopt_atm"] = xr.DataArray(iopt, dims=["icell"])
            pro["iabs_atm"] = xr.DataArray(iabs, dims=["icell"])
            pro["pmin_atm"] = xr.DataArray(pmin, dims=["xyz", "icell"])
            pro["pmax_atm"] = xr.DataArray(pmax, dims=["xyz", "icell"])
            pro["neighbour_atm"] = xr.DataArray(
                neighbour, dims=["faces", "icell"]
            )

        return pro

    def native_theta(self) -> NDArray[np.float64]:
        """The union of the scattering angles the components carry.

        The grid ``n_theta='native'`` resolves to: every component is
        resampled onto it, which is exact since it holds every node
        of every component table, see
        :func:`smartg.phase.union_theta_grid`.

        Returns
        -------
        ndarray
            Strictly increasing angles in degrees, from 0 to 180.

        Raises
        ------
        ValueError
            If the atmosphere has no component.
        """
        if len(self.comp) == 0:
            raise ValueError(
                "The atmosphere has no aerosol or cloud component, so "
                "no native scattering angle grid."
            )
        theta, _ = _common_theta_grid(
            [comp.native_theta() for comp in self.comp],
            [_grid_label(comp) for comp in self.comp],
        )
        return theta

    def phase(
        self, wavelength: NumericArrayLike, n_theta: ThetaLike = 721
    ) -> xr.DataArray | None:
        """
        Calculate phase matrix of aerosols and clouds at specified
        wavelengths.

        Computes weighted average phase functions for all aerosol
        components
        using the reduced atmospheric profile. Each component's
        contribution
        is weighted by its optical depth and single scattering albedo.

        Parameters
        ----------
        wavelength : array_like
            Wavelengths at which to calculate phase matrix [nm].
            If scalar, will be converted to 1-D array.
        n_theta : int, str or array_like, optional
            The number of equally spaced scattering angles every
            component is resampled on, the angles themselves in
            degrees, or ``'native'`` for the union of the angles the
            components' tables carry, on which the mixture is exact,
            see `native_theta`. Default is 721, corresponding to
            equally spaced angles from 0° to 180°.

        Returns
        -------
        phase_matrix : DataArray or None
            DataArray containing the weighted average phase matrix with
            axes
            [wavelength_phase, z_phase, nphamat, theta_atm] if
            aerosol components are present.
            Shape is (len(wavelength), nz, nphamat, n_theta) where:
            - nz: number of altitude levels in the reduced profile
              (self.pfgrid)
            - nphamat = 4 for spherical particles only (phase matrix
              unique terms P11, P21, P33, P34)
            - nphamat = 6 for spherical and non-spherical particles
              (additional terms P22, P44)
            - theta_atm: scattering angles from 0° to 180°

            Returns None if no aerosol components are defined (self.comp
            is empty).

        Notes
        -----
        **Weighted averaging:** The phase matrix is computed as a
        weighted average
        across all aerosol components defined in the comp attribute:

        pha_total = [∑_i (pha_i x Δτ_i x ssa_i)) / (∑_i (Δτ_i x ssa_i)]

        where:

        - pha_i is the phase matrix of component i
        - Δτ_i is the optical depth of component i
        - ssa_i is the single scattering albedo of component i

        The relative humidity used for calculations is obtained from
        the reduced profile (self.prof_red).

        **Scattering angle grids:** the components are mixed on one
        grid. When they come back on different grids, which a
        user-supplied phase matrix does since it keeps its own, they
        are resampled onto the union of those grids, which loses no
        node of any of them, and a warning says so. The
        `wavelength_phase` and `z_phase` axes cannot be merged that
        way and must agree.

        Raises
        ------
        ValueError
            If the phase matrices of the components are not tabulated
            on the same `wavelength_phase` and `z_phase` axes.
        """
        wavelength = np.atleast_1d(wavelength)
        if len(self.comp) == 0:
            return None
        rh = self.prof_red.relative_humidity()

        theta_req = (
            self.native_theta() if is_native_theta(n_theta) else n_theta
        )
        phases = []
        weights = []
        for comp in self.comp:
            dtau, ssa_p = comp.dtau_ssa(wavelength, self.pfgrid, rh=rh)
            comp_pha = comp.phase(
                wavelength, self.pfgrid, rh, n_theta=theta_req
            )
            if hasattr(comp_pha, "to_xarray"):
                comp_pha = comp_pha.to_xarray()

            # dtau/ssa grids are defined on pfgrid boundaries; skip TOA
            # bound to match z_phase layers.
            weights.append(
                xr.DataArray(
                    dtau[:, 1:] * ssa_p[:, 1:],
                    dims=["wavelength_phase", "z_phase"],
                    coords={
                        "wavelength_phase":
                            comp_pha.coords["wavelength_phase"].values,
                        "z_phase": comp_pha.coords["z_phase"].values,
                    },
                )
            )
            phases.append(comp_pha)

        # the sum below aligns the coordinates by their intersection,
        # which must never be where the angles go: bring every matrix
        # onto the common grid first, and refuse the axes that have no
        # common grid
        ref = phases[0]
        for comp_pha in phases[1:]:
            for dim in ["wavelength_phase", "z_phase"]:
                if not np.array_equal(
                    comp_pha.coords[dim].values, ref.coords[dim].values
                ):
                    raise ValueError(
                        "The phase matrices of the components must "
                        f"share the same {dim} axis to be averaged. "
                        "Use a common wavelength_phase, or provide "
                        "the phase matrices directly."
                    )
        theta, resampled = _common_theta_grid(
            [p.coords["theta_atm"].values for p in phases],
            [_grid_label(comp) for comp in self.comp],
        )
        if resampled:
            phases = [_on_theta_grid(p, theta) for p in phases]

        pha = None
        norm = None
        for comp_pha, weight_2d in zip(phases, weights, strict=True):
            weighted_pha = comp_pha * weight_2d
            pha = weighted_pha if pha is None else (pha + weighted_pha)
            norm = weight_2d if norm is None else (norm + weight_2d)

        assert pha is not None and norm is not None
        return (pha / norm).fillna(0.0)

    def calc_split(
        self,
        wavelength: NumericArrayLike | BandSet,
        phase: bool = True,
        n_theta: ThetaLike = 721,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, list[xr.DataArray]],
    ]:
        """
        Computes atmospheric optical properties at specified wavelengths
        and
        separates them into decomposed components (absorption, Rayleigh
        scattering,
        aerosols, and phase functions). These returned profiles can be
        used as
        alternative inputs to initialize a new Atm1D instance.

        Parameters
        ----------
        wavelength : array_like or BandSet
            Wavelengths at which to calculate optical properties [nm].
        phase : bool, optional
            If True (default), calculates phase functions. Set to False
            to skip
            phase function computations for faster execution.
        n_theta : int, str or array_like, optional
            The number of equally spaced scattering angles for phase
            function resampling, the angles themselves in degrees, or
            ``'native'`` for the union of the angles the components'
            tables carry. Default is 721, corresponding to angles from
            0° to 180°. Only used if ``phase=True``.

        Returns
        -------
        prof_abs : ndarray
            Gaseous absorption coefficient [wavelength, altitude]
            (km⁻¹).
            Differential optical thickness for absorption from cumulated
            profile.
        prof_ray : ndarray
            Rayleigh scattering coefficient [wavelength, altitude]
            (km⁻¹).
            Differential optical thickness for Rayleigh from cumulated
            profile.
        (prof_aer, ssa_aer) : tuple
            Aerosol profiles with:

            - prof_aer: Aerosol extinction coefficient [wavelength,
              altitude] (km⁻¹)
            - ssa_aer: Particle single scattering albedo [wavelength,
              altitude]

        (pro_iphase, pro_phases) : tuple
            Phase function profiles with:

            - pro_iphase: Phase matrix indices array [wavelength,
              altitude]
            - pro_phases: List of xarray.DataArray phase matrices (one
              per phase index). Each entry is an `xr.DataArray`
              representing the phase matrix for that phase index
              (dimensions typically ['stk', 'theta_atm']).

        Notes
        -----
        This method is useful for decomposing atmospheric optical
        properties into
        separate components. The returned profiles can be used to
        recreate the
        atmospheric model by passing them as alternative inputs:

        - prof_abs: passed as the prof_abs parameter
        - prof_ray: passed as the prof_ray parameter
        - (prof_aer, ssa_aer): passed as the prof_aer parameter
        - (pro_iphase, pro_phases): passed to phase parameter handling

        All returned arrays are cast to float32 for memory efficiency.

        Examples
        --------
        >>> atm = Atm1D('afglus')
        >>> (prof_abs, prof_ray, (prof_aer, ssa_aer)
        ...  (pro_iphase, pro_phases)) = atm.calc_split(wavelength=500.)
        """
        if not isinstance(wavelength, BandSet):
            wavelength = np.atleast_1d(wavelength)
        pro = self.calc(wavelength=wavelength, phase=phase, n_theta=n_theta)
        pro_aer = diff1(pro["OD_p"].values.astype(np.float32), axis=1)
        ssa_aer = pro["ssa_p_atm"].values
        pro_ray = diff1(pro["OD_r"].values.astype(np.float32), axis=1)
        pro_abs = diff1(pro["OD_g"].values.astype(np.float32), axis=1)
        pro_iphase = pro["iphase_atm"].values
        pro_phases = [
            pro["phase_atm"].sel(iphase=i)
            for i in range(int(pro_iphase.max()) + 1)
        ]

        return pro_abs, pro_ray, (pro_aer, ssa_aer), (pro_iphase, pro_phases)


class _Atm3DBackend(Atm1D):
    """Private Atm1D specialization computing the 3D profile dataset.

    It holds the merged (1D levels + 3D component cells) optical
    properties assembled by :class:`Atm3D` over the optical-property
    index axis `grid`, with a dummy zeroed physical profile, and
    enables the 3D branches of :meth:`Atm1D.calc`: the optical
    properties are returned as coefficients in km-1 instead of
    cumulated optical thicknesses, and the cells datasets are attached
    to the profile.
    """

    def __init__(
        self,
        grid: NDArray[np.integer],
        prof_ray: NDArray[np.floating],
        prof_abs: NDArray[np.floating],
        prof_aer: tuple[NDArray[np.floating], NDArray[np.floating]],
        prof_phases: tuple[NDArray[np.integer], list[Any]] | None,
        cells: tuple[
            NDArray[np.integer],
            NDArray[np.integer],
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.integer],
        ],
        o3_acs: PathType = "O3_acs_BogumilV3.0_coeffs",
        no2_acs: PathType = "NO2_acs_BogumilV1.0_coeffs",
    ) -> None:

        self.lat = 45.0
        self.comp = []
        self.wavelength_phase = None
        self.pfgrid = np.array([100.0, 0.0])
        self.prof_abs = prof_abs
        self.prof_ray = prof_ray
        self.prof_aer = prof_aer
        self.prof_phases = prof_phases
        self.rh_cst = None
        self.opt3d = True
        self.cells = cells
        self.tau_r = None

        #
        # dummy zeroed profile over the optical-property index axis
        #
        grid = np.asarray(grid)
        Nopt = grid.size
        prof = ProfileBase(None)
        prof.z = np.arange(Nopt, dtype=np.float32)[::-1]
        attr_names = [
            "p",
            "t",
            "dens_air",
            "dens_h2o",
            "dens_o3",
            "dens_n2o",
            "dens_co",
            "dens_ch4",
            "dens_co2",
            "dens_o2",
            "dens_n2",
            "dens_no2",
            "dens_so2",
        ]
        for attr_name in attr_names:
            setattr(prof, attr_name, np.zeros(Nopt, dtype=np.float32))
        prof.rh_cst = None

        #
        # read gaseous acs
        #
        O3_acs_path = Path(o3_acs)
        if O3_acs_path.parent == Path("."):
            O3_acs_path = DIR_AUXDATA / "acs" / O3_acs_path.name
        if not O3_acs_path.exists() and O3_acs_path.suffix != ".nc":
            O3_acs_path = O3_acs_path.with_name(O3_acs_path.name + ".nc")
        self.acs_o3 = xr.open_dataset(O3_acs_path)
        self.acs_o3 = self.acs_o3.rename({"wav": "wavelength"})

        NO2_acs_path = Path(no2_acs)
        if NO2_acs_path.parent == Path("."):
            NO2_acs_path = DIR_AUXDATA / "acs" / NO2_acs_path.name
        if not NO2_acs_path.exists() and NO2_acs_path.suffix != ".nc":
            NO2_acs_path = NO2_acs_path.with_name(NO2_acs_path.name + ".nc")
        self.acs_no2 = xr.open_dataset(NO2_acs_path)
        self.acs_no2 = self.acs_no2.rename({"wav": "wavelength"})

        self._prof_src = prof
        self.prof = prof.regrid(grid)
        self.prof_red = prof.regrid(self.pfgrid)


class Atm3D(Atmosphere):
    """3D atmospheric profile definition.

    The 3D atmosphere combines a 1D background atmosphere (molecular
    scattering and absorption, plus optional 1D aerosols), a 3D grid
    and a list of 3D components (e.g. :class:`Cloud3D`,
    :class:`Aer3D`). The 1D
    background is evaluated on the vertical discretization of the 3D
    grid and shared by all the cells at the same altitude; the cells
    occupied by a 3D component get their own optical properties, mixing
    the component with the co-located 1D background particles.

    Example
    -------
    >>> atm3d = Atm3D(
    ...     atm_1d=Atm1D("afglt"),
    ...     grid_3d=Grid3D(xgrid, ygrid, zgrid, periodic=True),
    ...     comp_3d=[Cloud3D("wc", w_ref=800., ds=cloud_field)],
    ... )
    >>> pro = atm3d.calc(wavelength)

    Parameters
    ----------
    atm_1d : Atm1D
        The 1D background atmosphere. It must be built without the
        `grid` parameter: in 3D mode the vertical discretization is set
        by `grid_3d`.
    grid_3d : Grid3D
        The 3D grid of the atmosphere.
    comp_3d : list of Comp3D or None, optional
        The 3D components to consider, i.e. a list of :class:`Cloud3D`
        / :class:`Aer3D` objects. The cell indices must be unique
        within each component. In the cells shared by several
        components (and by the 1D aerosols), the extinction
        coefficients are summed, the single scattering albedos are
        extinction-weighted and the phase matrices are weighted by
        the scattering coefficients.
        If None or empty, the 3D atmosphere is horizontally uniform.
    wavelength_phase : array_like or None, optional
        The wavelengths over which the phase matrices are calculated.
        Then use the nearest wavelength during cuda simulation. Useful
        to reduce the memory. If None, compute the phase matrices at
        all wavelengths.
    mol_sca_1d : 2-D ndarray or None, optional
        Force the 1D molecular scattering (Rayleigh) coefficients in
        km-1, with the shape (nwavelength, NZ + 1) where NZ is the
        number of vertical cells of `grid_3d` including the boundary.
        If None, computed from `atm_1d`.
    mol_abs_1d : 2-D ndarray or None, optional
        Force the 1D molecular absorption coefficients in km-1, same
        shape as `mol_sca_1d`. If None, computed from `atm_1d`.
    aer_ext_1d : 2-D ndarray or None, optional
        Force the 1D aerosol extinction coefficients in km-1, same
        shape as `mol_sca_1d`. If None, computed from `atm_1d`.
    aer_ssa_1d : 2-D ndarray or None, optional
        Force the 1D aerosol single scattering albedos, same shape as
        `mol_sca_1d`. If None, computed from `atm_1d`.
    aer_phase_1d : tuple or None, optional
        Force the 1D aerosol phase matrices, as a tuple
        (iphase, phases) where iphase is the (nwavelength, NZ + 1)
        phase matrix indices profile and phases a DataArray (or
        legacy LUT) of phase matrices over (iphase, stk, theta_atm).
        If None, computed from `atm_1d`.
    """

    def __init__(
        self,
        atm_1d: Atm1D,
        grid_3d: Grid3D,
        comp_3d: Sequence[Comp3D] | None = None,
        wavelength_phase: NumericArrayLike | None = None,
        mol_sca_1d: NDArray[np.floating] | None = None,
        mol_abs_1d: NDArray[np.floating] | None = None,
        aer_ext_1d: NDArray[np.floating] | None = None,
        aer_ssa_1d: NDArray[np.floating] | None = None,
        aer_phase_1d: tuple[NDArray[np.integer], Any] | None = None,
    ) -> None:

        if not isinstance(atm_1d, Atm1D):
            raise TypeError("atm_1d must be an Atm1D object!")
        if not isinstance(grid_3d, Grid3D):
            raise TypeError("grid_3d must be a Grid3D object!")
        if atm_1d.prof is not atm_1d._prof_src:
            raise ValueError(
                "atm_1d must be built without the grid parameter: in 3D "
                "mode the vertical discretization is set by grid_3d"
            )

        comp_3d = [] if comp_3d is None else list(comp_3d)
        for comp in comp_3d:
            if not isinstance(comp, Comp3D):
                raise TypeError(
                    "comp_3d must be a list of Comp3D objects (e.g. "
                    "Cloud3D, Aer3D)!"
                )
        self.atm_1d = atm_1d
        self.grid_3d = grid_3d
        self.comp_3d = comp_3d
        self.wavelength_phase = (
            None if wavelength_phase is None
            else np.asarray(wavelength_phase)
        )
        self.mol_sca_1d = mol_sca_1d
        self.mol_abs_1d = mol_abs_1d
        self.aer_ext_1d = aer_ext_1d
        self.aer_ssa_1d = aer_ssa_1d
        self.aer_phase_1d = aer_phase_1d

        if len(comp_3d) == 1:
            # cell indices on the boundary-extended grid
            cell_indices = comp_3d[0].get_cell_indices().copy()
            # shift the x and y indices if the grid has horizontal
            # boundary cells
            if grid_3d.Nx < grid_3d.NX:
                cell_indices[:, :2] += 1
            self._cell_indices = cell_indices
            # flat indices of the component cells in the 3D grid
            self._cell_flat_indices = np.ravel_multi_index(
                (
                    cell_indices[:, 0],
                    cell_indices[:, 1],
                    cell_indices[:, 2],
                ),
                dims=(grid_3d.NX, grid_3d.NY, grid_3d.NZ),
            )
            self._comp_cell_pos = [
                np.arange(self._cell_flat_indices.size)
            ]
        elif comp_3d:
            # several components: the global cell list is the sorted
            # union of the component cells, with the position of each
            # component cell within the union kept for the mixing
            comp_flat = []
            for icomp, comp in enumerate(comp_3d):
                cell_indices = comp.get_cell_indices().copy()
                # shift the x and y indices if the grid has
                # horizontal boundary cells
                if grid_3d.Nx < grid_3d.NX:
                    cell_indices[:, :2] += 1
                flat = np.ravel_multi_index(
                    (
                        cell_indices[:, 0],
                        cell_indices[:, 1],
                        cell_indices[:, 2],
                    ),
                    dims=(grid_3d.NX, grid_3d.NY, grid_3d.NZ),
                )
                if np.unique(flat).size != flat.size:
                    raise ValueError(
                        f"comp_3d[{icomp}] has duplicated cell "
                        "indices: the cells of a component must be "
                        "unique"
                    )
                comp_flat.append(flat)
            union = np.unique(np.concatenate(comp_flat))
            self._cell_flat_indices = union
            self._cell_indices = np.stack(
                np.unravel_index(
                    union, (grid_3d.NX, grid_3d.NY, grid_3d.NZ)
                ),
                axis=1,
            ).astype(np.int32)
            self._comp_cell_pos = [
                np.searchsorted(union, flat) for flat in comp_flat
            ]
        else:
            self._cell_indices = None
            self._cell_flat_indices = None
            self._comp_cell_pos = None

    def native_theta(self) -> NDArray[np.float64]:
        """The union of the scattering angles the 1D and the 3D
        components carry, in degrees, which ``n_theta='native'``
        resolves to, see :func:`smartg.phase.union_theta_grid`.

        Raises
        ------
        ValueError
            If neither the 1D atmosphere nor the 3D field has a
            component.
        """
        comps = list(self.atm_1d.comp) + list(self.comp_3d)
        if not comps:
            raise ValueError(
                "The atmosphere has no aerosol or cloud component, so "
                "no native scattering angle grid."
            )
        theta, _ = _common_theta_grid(
            [comp.native_theta() for comp in comps],
            [_grid_label(comp) for comp in comps],
        )
        return theta

    def calc(
        self,
        wavelength: NumericArrayLike | BandSet,
        phase: bool = True,
        n_theta: ThetaLike = 721,
        use_old_calc_iphase: bool = False,
        truncation: DMTrunc | GTTrunc | None = None,
    ) -> xr.Dataset:
        """Compute the 3D atmospheric profile at given wavelengths.

        Parameters
        ----------
        wavelength : array_like or BandSet
            Wavelengths in nm.
        phase : bool, optional
            If True (default), compute the phase matrices.
        n_theta : int, str or array_like, optional
            The number of equally spaced scattering angles of the
            phase matrices, the angles themselves in degrees, or
            ``'native'`` for the union of the angles the 1D and 3D
            components' tables carry, on which their mixture is
            exact, see `native_theta`.
        use_old_calc_iphase : bool, optional
            Use the old (slower) implementation of calc_iphase.
        truncation : DMTrunc, GTTrunc or None, optional
            Phase matrix truncation method. See :meth:`Atm1D.calc`.

        Returns
        -------
        xr.Dataset
            The atmospheric profile, with the optical properties given
            as coefficients in km-1 over the `iopt` axis of unique
            optical properties, and the 3D cell datasets (`iopt_atm`,
            `iabs_atm`, `pmin_atm`, `pmax_atm`, `neighbour_atm`)
            consumed by :meth:`smartg.smartg.Smartg.run`.
        """
        if isinstance(wavelength, BandSet):
            wavelengths = np.asarray(wavelength.wavelength)
        else:
            wavelengths = np.atleast_1d(np.asarray(wavelength))
        wavelength_pha = (
            self.wavelength_phase
            if self.wavelength_phase is not None else wavelengths
        )
        if is_native_theta(n_theta):
            # resolved once here, so that the 1D and the 3D components
            # are all asked for the same grid
            n_theta = self.native_theta()

        #
        # 1D background optical properties on the 3D vertical grid
        #
        pha_1d = len(self.atm_1d.comp) > 0
        ipha_aer_1d = None
        pha_aer_1d = None
        mol_sca_1d = self.mol_sca_1d
        mol_abs_1d = self.mol_abs_1d
        ext_aer_1d = self.aer_ext_1d
        ssa_aer_1d = self.aer_ssa_1d
        if (
            mol_sca_1d is None
            or mol_abs_1d is None
            or ext_aer_1d is None
            or ssa_aer_1d is None
            or self.aer_phase_1d is None
        ):
            atm_1d = copy.copy(self.atm_1d)
            atm_1d.prof = self.atm_1d._prof_src.regrid(
                np.asarray(self.grid_3d.zGRID[::-1])
            )
            ds_1d = atm_1d.calc(wavelength, phase=pha_1d, n_theta=n_theta)
            if mol_sca_1d is None:
                mol_sca_1d = od2k(ds_1d, "OD_r")
            if mol_abs_1d is None:
                mol_abs_1d = od2k(ds_1d, "OD_g")
            if ext_aer_1d is None:
                ext_aer_1d = od2k(ds_1d, "OD_p")
            if ssa_aer_1d is None:
                ssa_aer_1d = ds_1d["ssa_p_atm"]
            if pha_1d:
                ipha_aer_1d = ds_1d["iphase_atm"]
                pha_aer_1d = ds_1d["phase_atm"]
        assert mol_sca_1d is not None and mol_abs_1d is not None
        assert ext_aer_1d is not None and ssa_aer_1d is not None

        if self.aer_phase_1d is not None:
            ipha_aer_1d = self.aer_phase_1d[0]
            pha_aer_1d = self.aer_phase_1d[1]

        # The merge works on xarray objects and numpy arrays: convert
        # a legacy 1d aerosol phase LUT to a DataArray, with the dim
        # names the merge expects
        if isinstance(pha_aer_1d, LUT):
            pha_aer_1d = pha_aer_1d.to_xarray()
        if isinstance(pha_aer_1d, xr.DataArray):
            pha_aer_1d = pha_aer_1d.rename(
                dict(
                    zip(
                        pha_aer_1d.dims,
                        ("iphase", "nphamat", "theta_atm"),
                        strict=True,
                    )
                )
            )
        if isinstance(ipha_aer_1d, xr.DataArray):
            ipha_aer_1d = ipha_aer_1d.values
        if isinstance(ssa_aer_1d, xr.DataArray):
            ssa_aer_1d = ssa_aer_1d.values

        #
        # merge the 1D background and the 3D components
        #
        mol_sca_glob = self._glob_molecular(mol_sca_1d)
        mol_abs_glob = self._glob_molecular(mol_abs_1d)
        ext_glob, ssa_glob, prof_phases = self._glob_particles(
            wavelengths,
            wavelength_pha,
            n_theta,
            ext_aer_1d,
            ssa_aer_1d,
            ipha_aer_1d,
            pha_aer_1d,
        )

        #
        # assemble the profile dataset
        #
        backend = _Atm3DBackend(
            grid=self._grid(),
            prof_ray=mol_sca_glob,
            prof_abs=mol_abs_glob,
            prof_aer=(ext_glob, ssa_glob),
            prof_phases=prof_phases,
            cells=self._cells_info(),
        )
        return backend.calc(
            wavelength,
            phase=phase,
            n_theta=n_theta,
            use_old_calc_iphase=use_old_calc_iphase,
            truncation=truncation,
        )

    def _grid(self) -> NDArray[np.integer]:
        """The optical-property index axis of the merged profile: the
        1D levels first, then one entry per 3D component cell.
        """
        Nopt = self.grid_3d.NZ + 1
        if self._cell_indices is not None:
            Nopt += self._cell_indices.shape[0]
        return np.arange(Nopt)

    def _glob_molecular(
        self, mol_1d: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Merge the (n_wavelength, NZ + 1) 1D molecular
        coefficients into the global (n_wavelength, Nopt) array: the
        component cells replicate the 1D
        value at the same altitude.
        """
        if self._cell_flat_indices is None:
            return mol_1d
        return np.concatenate(
            [
                mol_1d,
                mol_1d[
                    :,
                    self.grid_3d.NZ
                    - self.grid_3d.idz[self._cell_flat_indices],
                ],
            ],
            axis=1,
        )

    def _glob_particles(
        self,
        wavelengths: NDArray[np.floating],
        wavelength_pha: NDArray[np.floating],
        n_theta: ThetaLike,
        ext_aer_1d: NDArray[np.floating],
        ssa_aer_1d: NDArray[np.floating],
        ipha_aer_1d: NDArray[np.integer] | None,
        pha_aer_1d: Any,
    ) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        tuple[NDArray[np.int32], list[Any]] | None,
    ]:
        """Merge the 1D aerosols and the 3D component into the global
        (n_wavelength, Nopt) particle extinction and single scattering albedo
        arrays and the global phase matrix set.
        """
        NZ = self.grid_3d.NZ
        nbz = NZ + 1

        if not self.comp_3d:
            if pha_aer_1d is None:
                return ext_aer_1d, ssa_aer_1d, None
            phases = []
            for i_wavelength in range(0, len(wavelength_pha)):
                for iz in range(0, nbz):
                    phases.append(
                        pha_aer_1d.isel(iphase=ipha_aer_1d[i_wavelength, iz])
                    )
            ipha3d = np.zeros((len(wavelength_pha), nbz), dtype=np.int32)
            for i_wavelength in range(0, len(wavelength_pha)):
                ipha3d[i_wavelength, :] = np.arange(nbz, dtype=np.int32) + (
                    i_wavelength * nbz
                )
            return ext_aer_1d, ssa_aer_1d, (ipha3d, phases)

        if len(self.comp_3d) > 1:
            return self._glob_particles_multi(
                wavelengths,
                wavelength_pha,
                n_theta,
                ext_aer_1d,
                ssa_aer_1d,
                ipha_aer_1d,
                pha_aer_1d,
            )

        comp = self.comp_3d[0]
        assert self._cell_indices is not None
        n_cell = self._cell_indices.shape[0]
        ext_3d = comp.get_ext(wavelengths)
        ssa_3d = comp.get_ssa(wavelengths)
        cld_phases, cell_pha_idx, n_unique = comp.get_phase_set(
            wavelength_pha, n_theta=n_theta
        )

        ext_mix_3d = np.zeros((len(wavelengths), n_cell), dtype=np.float64)
        ssa_mix_3d = np.ones((len(wavelengths), n_cell), dtype=np.float64)

        if pha_aer_1d is None:  # case no 1d aer given
            ext_mix_3d[:, :] = ext_3d
            ssa_mix_3d[:, :] = ssa_3d

            phases = cld_phases
            # Concatenate plan parallel + 3d optical prop (first
            # without considering wavelength)
            phase_glob_indices_w0 = np.concatenate(
                [np.zeros(nbz, dtype=np.int32), cell_pha_idx[:]]
            )
        else:  # case list of 1d aer is given
            # bring the 1d aerosol and the component phase matrices
            # onto the union of their scattering angle grids, so that
            # the mixing arithmetic below aligns exactly and loses no
            # node of either
            theta, resampled = _common_theta_grid(
                [
                    cld_phases[0].coords["theta_atm"].values,
                    pha_aer_1d.coords["theta_atm"].values,
                ],
                [_grid_label(comp), "the 1D aerosols"],
            )
            if resampled:
                cld_phases = [_on_theta_grid(p, theta) for p in cld_phases]
            phase_aer_1d = _on_theta_grid(pha_aer_1d, theta)

            # the altitude level of the 1D aerosols co-located with
            # each component cell (kept as-is from the historical
            # implementation; note the inconsistency with the
            # NZ - idz mapping used for the molecular properties)
            idz_atm = []
            for icell in range(0, n_cell):
                idz = self._cell_indices[icell, 2]
                idz_atm.append(NZ + 1 - idz)

            # First plan parallel phase
            phases = []
            for i_wavelength in range(0, len(wavelength_pha)):
                for iz in range(0, nbz):
                    phases.append(
                        phase_aer_1d.isel(
                            iphase=ipha_aer_1d[i_wavelength, iz]
                        )
                    )

            # Second 3d mix phase, weighted by the extinctions at the
            # phase wavelengths
            ext_3d_pha = comp.get_ext(wavelength_pha)
            ssa_3d_pha = comp.get_ssa(wavelength_pha)
            for i_wavelength in range(0, len(wavelength_pha)):
                ssa_aer_tmp = ssa_aer_1d[i_wavelength, idz_atm]
                ext_aer_tmp = ext_aer_1d[i_wavelength, idz_atm]
                ext_mix_tmp = ext_aer_tmp + ext_3d_pha[i_wavelength, :]

                for icell in range(0, n_cell):
                    pha_cld_tmp = cld_phases[
                        i_wavelength * n_unique + cell_pha_idx[icell]
                    ]
                    pha_aer_tmp = phase_aer_1d.isel(
                        iphase=ipha_aer_1d[i_wavelength, idz_atm[icell]]
                    )
                    pha_tot = (
                        (
                            pha_aer_tmp
                            * ext_aer_tmp[icell]
                            * ssa_aer_tmp[icell]
                        )
                        + (
                            pha_cld_tmp
                            * ext_3d_pha[i_wavelength, icell]
                            * ssa_3d_pha[i_wavelength, icell]
                        )
                    ) / ext_mix_tmp[icell]
                    phases.append(pha_tot)

            # the mixed extinctions and ssa of the profile, at the
            # profile wavelengths
            for i_wavelength in range(0, len(wavelengths)):
                ssa_aer_tmp = ssa_aer_1d[i_wavelength, idz_atm]
                ext_aer_tmp = ext_aer_1d[i_wavelength, idz_atm]
                ext_mix_tmp = ext_aer_tmp + ext_3d[i_wavelength, :]
                ext_mix_3d[i_wavelength, :] = ext_mix_tmp
                ssa_mix_3d[i_wavelength, :] = (
                    ext_aer_tmp * ssa_aer_tmp
                    + ext_3d[i_wavelength, :] * ssa_3d[i_wavelength, :]
                ) / ext_mix_tmp

            # Concatenate plan parallel + 3d optical prop (first
            # without considering wavelength)
            phase_glob_indices_w0 = np.arange(
                nbz + n_cell, dtype=np.int32
            )

        # Create a table with only the component properties but in
        # global shape i.e. for each cells not sharing the same opt
        # prop, and other commun cells in z, following the plan
        # parallel 1D atm philosophy
        ext_glob = np.concatenate([ext_aer_1d, ext_mix_3d], axis=1)
        ssa_glob = np.concatenate([ssa_aer_1d[:, :], ssa_mix_3d], axis=1)

        # Now consider the wavelength dimension
        # NB: with a 1D aerosol the per-wavelength stride in `phases`
        # is nbz + n_cell, not n_unique, so the offset below is only
        # correct when len(wavelength_pha) == 1 (the only exercised
        # case;
        # kept as-is for consistency with the saved references)
        ipha3d = np.zeros(
            (len(wavelength_pha), phase_glob_indices_w0.size), dtype=np.int32
        )
        for i_wavelength in range(0, len(wavelength_pha)):
            ipha3d[i_wavelength, :] = phase_glob_indices_w0[:] + (
                i_wavelength * n_unique
            )

        return ext_glob, ssa_glob, (ipha3d, phases)

    def _glob_particles_multi(
        self,
        wavelengths: NDArray[np.floating],
        wavelength_pha: NDArray[np.floating],
        n_theta: ThetaLike,
        ext_aer_1d: NDArray[np.floating],
        ssa_aer_1d: NDArray[np.floating],
        ipha_aer_1d: NDArray[np.integer] | None,
        pha_aer_1d: Any,
    ) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        tuple[NDArray[np.int32], list[Any]] | None,
    ]:
        """Merge the 1D aerosols and several 3D components into the
        global (n_wavelength, Nopt) particle extinction and single scattering
        albedo arrays and the global phase matrix set.

        In each cell the extinctions are summed, the single
        scattering albedos are extinction-weighted and the phase
        matrices are weighted by the scattering coefficients and
        normalized by the total extinction, following the 1D/3D
        mixing conventions of `_glob_particles`.
        """
        NZ = self.grid_3d.NZ
        nbz = NZ + 1
        assert self._cell_indices is not None
        assert self._comp_cell_pos is not None
        n_cell = self._cell_indices.shape[0]

        # per-component optical properties and phase matrix sets
        ext_3d = [comp.get_ext(wavelengths) for comp in self.comp_3d]
        ssa_3d = [comp.get_ssa(wavelengths) for comp in self.comp_3d]
        ext_3d_pha = [comp.get_ext(wavelength_pha) for comp in self.comp_3d]
        ssa_3d_pha = [comp.get_ssa(wavelength_pha) for comp in self.comp_3d]
        phase_sets = [
            comp.get_phase_set(wavelength_pha, n_theta=n_theta)
            for comp in self.comp_3d
        ]

        # align every phase matrix set (and the 1D aerosol one) on
        # the union of their scattering angle grids, which loses no
        # node of any of them
        grids = [
            cld_phases[0].coords["theta_atm"].values
            for cld_phases, _, _ in phase_sets
        ]
        labels = [_grid_label(comp) for comp in self.comp_3d]
        if pha_aer_1d is not None:
            grids.append(pha_aer_1d.coords["theta_atm"].values)
            labels.append("the 1D aerosols")
        theta_ref, resampled = _common_theta_grid(grids, labels)
        cld_phases_all = []
        for cld_phases, _, _ in phase_sets:
            if resampled:
                cld_phases = [
                    _on_theta_grid(pha, theta_ref) for pha in cld_phases
                ]
            cld_phases_all.append(cld_phases)
        phase_aer_1d = None
        if pha_aer_1d is not None:
            phase_aer_1d = _on_theta_grid(pha_aer_1d, theta_ref)

        # the altitude level of the 1D aerosols co-located with each
        # cell (same historical mapping as `_glob_particles`)
        idz_atm = nbz - self._cell_indices[:, 2]

        # position of each cell of each component within the global
        # cell list
        local_pos = np.full(
            (len(self.comp_3d), n_cell), -1, dtype=np.int64
        )
        for icomp, pos in enumerate(self._comp_cell_pos):
            local_pos[icomp, pos] = np.arange(pos.size)

        # the mixed extinctions and ssa, at the profile wavelengths
        ext_mix_3d = np.zeros((len(wavelengths), n_cell), dtype=np.float64)
        sca_mix_3d = np.zeros_like(ext_mix_3d)
        for icomp, pos in enumerate(self._comp_cell_pos):
            ext_mix_3d[:, pos] += ext_3d[icomp]
            sca_mix_3d[:, pos] += ext_3d[icomp] * ssa_3d[icomp]
        if pha_aer_1d is not None:
            ext_mix_3d += ext_aer_1d[:, idz_atm]
            sca_mix_3d += ext_aer_1d[:, idz_atm] * ssa_aer_1d[
                :, idz_atm
            ]
        ssa_mix_3d = np.divide(
            sca_mix_3d,
            ext_mix_3d,
            out=np.ones_like(sca_mix_3d),
            where=ext_mix_3d > 0.0,
        )

        # the mixed phase matrices, one per cell and per phase
        # wavelength, weighted by the extinctions at the phase
        # wavelengths (sharing the matrices of the cells occupied by
        # a single component is a possible future memory
        # optimization)
        phases = []
        for i_wavelength in range(0, len(wavelength_pha)):
            if phase_aer_1d is not None:
                assert ipha_aer_1d is not None
                for iz in range(0, nbz):
                    phases.append(
                        phase_aer_1d.isel(
                            iphase=ipha_aer_1d[i_wavelength, iz]
                        )
                    )
            for icell in range(0, n_cell):
                pha_tot = None
                pha_first = None
                ext_tot = 0.0
                if phase_aer_1d is not None:
                    assert ipha_aer_1d is not None
                    idz = idz_atm[icell]
                    pha_first = phase_aer_1d.isel(
                        iphase=ipha_aer_1d[i_wavelength, idz]
                    )
                    pha_tot = pha_first * (
                        ext_aer_1d[i_wavelength, idz]
                        * ssa_aer_1d[i_wavelength, idz]
                    )
                    ext_tot += ext_aer_1d[i_wavelength, idz]
                for icomp in range(0, len(self.comp_3d)):
                    iloc = local_pos[icomp, icell]
                    if iloc < 0:
                        continue
                    _, cell_pha_idx, n_unique = phase_sets[icomp]
                    pha_cld = cld_phases_all[icomp][
                        i_wavelength * n_unique + cell_pha_idx[iloc]
                    ]
                    if pha_first is None:
                        pha_first = pha_cld
                    pha_comp = pha_cld * (
                        ext_3d_pha[icomp][i_wavelength, iloc]
                        * ssa_3d_pha[icomp][i_wavelength, iloc]
                    )
                    pha_tot = (
                        pha_comp if pha_tot is None
                        else pha_tot + pha_comp
                    )
                    ext_tot += ext_3d_pha[icomp][i_wavelength, iloc]
                assert pha_tot is not None and pha_first is not None
                if ext_tot > 0.0:
                    pha_tot = pha_tot / ext_tot
                else:
                    # never sampled (zero extinction): keep a valid
                    # unweighted matrix rather than a null one
                    pha_tot = pha_first
                phases.append(pha_tot)

        # the phase matrix indices, with the per-wavelength stride of
        # the `phases` layout above
        if phase_aer_1d is not None:
            stride = nbz + n_cell
            phase_glob_indices_w0 = np.arange(
                nbz + n_cell, dtype=np.int32
            )
        else:
            stride = n_cell
            phase_glob_indices_w0 = np.concatenate(
                [
                    np.zeros(nbz, dtype=np.int32),
                    np.arange(n_cell, dtype=np.int32),
                ]
            )
        ipha3d = np.zeros((len(wavelength_pha), nbz + n_cell), dtype=np.int32)
        for i_wavelength in range(0, len(wavelength_pha)):
            ipha3d[i_wavelength, :] = phase_glob_indices_w0[:] + (
                i_wavelength * stride
            )

        ext_glob = np.concatenate([ext_aer_1d, ext_mix_3d], axis=1)
        ssa_glob = np.concatenate([ssa_aer_1d, ssa_mix_3d], axis=1)
        return ext_glob, ssa_glob, (ipha3d, phases)

    def _cells_info(
        self,
    ) -> tuple[
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.int32],
    ]:
        """The per-cell optical and absorption property indices, cell
        bounding boxes and neighbours, as expected by the `cells`
        parameter of the profile backend.
        """
        NZ = self.grid_3d.NZ
        Nopt = self._grid().size

        iopt = np.zeros(self.grid_3d.NCELL, dtype=np.int32)
        iabs = np.zeros_like(iopt)
        # Scattering depending on Z for clear atmosphere (Rayleigh)
        iopt[:] = np.arange(Nopt)[NZ - self.grid_3d.idz]
        # Absorption depending on Z only
        iabs[:] = np.arange(Nopt)[NZ - self.grid_3d.idz]

        if self._cell_flat_indices is not None:
            iopt[self._cell_flat_indices] = NZ + 1 + np.arange(
                self._cell_flat_indices.size
            )

        return (
            iopt,
            iabs,
            self.grid_3d.pmin,
            self.grid_3d.pmax,
            self.grid_3d.neigh,
        )


class ProfileBase(object):
    """
    Atmospheric profile with physical properties.

    Reads and processes atmospheric profiles from files (NetCDF or
    libratran format).
    Allows customization of ozone, water vapor, and pressure profiles.
    Automatically
    scales gaseous constituents to specified total column amounts.

    Parameters
    ----------
    fname : path-like or None, optional
        Path to atmospheric profile file. Accepts .nc (NetCDF) or .dat
        (libratran) formats.
        If only fname is provided (no path), the auxdata directory is
        automatically prepended.
        If no suffix is provided, .nc is assumed by default.
    tco3 : float or None, optional
        Total column vertically-integrated ozone in Dobson units (DU).
        If None, uses the
        value from the atmospheric profile. The ozone profile is scaled
        to match this column
        amount. Note: 1 DU = 2.1415e-5 kg m⁻².
        Default: None
    tcwp : float or None, optional
        Total column vertically-integrated water vapour in g/cm². If
        None, uses the value
        from the atmospheric profile. The water vapour profile is scaled
        to match this column
        amount.
        Default: None
    tcno2 : bool or None, optional
        Total column vertically-integrated NO2. If False, NO2 density is
        set to zero.
        If True, NO2 profile from the atmospheric file is retained.
        Default: True
    p0 : float or None, optional
        Sea surface (bottom layer) pressure in hPa. If None, uses the
        pressure
        from the atmospheric profile. Scales all pressure values
        proportionally.
        Default: None
    rh_cst : float or None, optional
        Force relative humidity to be constant at this value. If None,
        relative
        humidity is recalculated from the temperature and water vapor
        profiles.
        Default: None
    o3_h2o_alt : float or None, optional
        Altitude (km) at which the specified tco3 and tcwp values apply.
        When specified,
        the ozone and water vapor profiles are scaled such that the
        column amount from TOA to this
        altitude matches the provided tco3 and tcwp values. The full
        gaseous distribution
        from TOA to ground is preserved; only the scaling factor is
        adjusted to match
        the constraint at this reference altitude.
        Default: None

    Notes
    -----
    File format support:
    - .nc (NetCDF): Expects variables 'P', 'T', 'dens', 'H2O', 'O3',
      etc. with
      dimension 'z_atm' for altitude
    - .dat (libratran): Text format with header line containing variable
      names
      (e.g., 'z(km) p(mb) T(K) air(cm-3) o3(cm-3) ...')
    """

    def __init__(
        self,
        fname: PathType | None,
        tco3: float | None = None,
        tcwp: float | None = None,
        tcno2: bool | None = True,
        p0: float | None = None,
        rh_cst: float | None = None,
        o3_h2o_alt: float | None = None,
    ) -> None:

        if fname is None:
            return
        fname = Path(fname)
        self.fname = fname

        if not fname.is_file():
            raise FileNotFoundError(
                f"Atmospheric profile file not found: {fname}"
            )

        if fname.suffix == ".dat":
            with open(fname) as f:
                lines = f.readlines()

            desc = None
            desc = ""
            n = 0
            for line in lines:
                if (
                    ("z(km)" in line)
                    and ("p(mb)" in line)
                    and ("T(K)" in line)
                    and ("air(cm-3)" in line)
                ):
                    desc = line
                    break
                else:
                    n += 1
            if desc == "":
                n = 0

            if desc is not None:
                # data = np.loadtxt(fname, dtype=np.float32,
                # comments="#", skiprows=n)
                data = pd.read_csv(
                    fname,
                    comment="#",
                    header=None,
                    sep=r"\s+",
                    dtype=np.float32,
                    skiprows=n,
                ).values
                self.z = data[:, 0]  # Altitude in km
                self.p = data[:, 1]  # pressure in hPa
                self.t = data[:, 2]  # temperature in K
                self.dens_air = data[:, 3]  # Air density in cm-3
                data2 = np.zeros((data.shape[0], 5))
                for i, gas in enumerate(["o3", "o2", "h2o", "co2", "no2"]):
                    try:
                        ind = desc.split().index(gas + "(cm-3)")
                        data2[:, i] = data[:, ind - 1]
                    except ValueError:
                        data2[:, i] = 0.0
                self.dens_o3 = data2[:, 0]  # Ozone density in cm-3
                self.dens_o2 = data2[:, 1]  # O2 density in cm-3
                self.dens_h2o = data2[:, 2]  # H2O density in cm-3
                self.dens_co2 = data2[:, 3]  # CO2 density in cm-3
                self.dens_no2 = data2[:, 4]  # NO2 density in cm-3
                nz = data.shape[0]
                self.dens_ch4 = np.zeros(nz, dtype=np.float32)
                self.dens_co = np.zeros(nz, dtype=np.float32)
                self.dens_n2o = np.zeros(nz, dtype=np.float32)
                self.dens_n2 = np.zeros(nz, dtype=np.float32)
                self.dens_so2 = np.zeros(nz, dtype=np.float32)
            else:
                raise ValueError("Invalid atmospheric file format")
        elif fname.suffix == ".nc":
            with xr.open_dataset(fname) as data:
                self.z = data.coords["z_atm"].values  # Altitude in km
                self.p = data["P"].values  # pressure in hPa
                self.t = data["T"].values  # temperature in K
                self.dens_air = data["dens"].values  # Air density cm-3
                self.dens_h2o = data["H2O"].values  # H2O density cm-3
                self.dens_o3 = data["O3"].values  # O3 density cm-3
                self.dens_n2o = data["N2O"].values  # N2O density cm-3
                self.dens_co = data["CO"].values  # CO density cm-3
                self.dens_ch4 = data["CH4"].values  # CH4 density cm-3
                self.dens_co2 = data["CO2"].values  # CO2 density cm-3
                self.dens_o2 = data["O2"].values  # O2 density cm-3
                self.dens_n2 = data["N2"].values  # N2 density cm-3
                self.dens_no2 = data["NO2"].values  # NO2 density cm-3
                self.dens_so2 = data["SO2"].values  # SO2 density cm-3

        self.rh_cst = rh_cst

        # scale to specified total ozone content
        if tco3 is not None:
            if o3_h2o_alt is None:
                denom = simpson(y=self.dens_o3, x=-self.z) * 1e5
                self.dens_o3 *= 2.69e16 * tco3 / denom
            else:
                _s_z = np.argsort(self.z)
                # k=1: linear interpolation; BSpline extrapolates
                # linearly beyond the data range by default
                f_dens_o3 = make_interp_spline(
                    self.z[_s_z], self.dens_o3[_s_z], k=1
                )
                z_alt = np.append(self.z[self.z > o3_h2o_alt], o3_h2o_alt)
                dens_o3_alt = f_dens_o3(z_alt)
                o3_afgl = (simpson(dens_o3_alt, -z_alt) * 1e5) / 2.69e16
                self.dens_o3 *= tco3 / o3_afgl
            if tco3 == 0:
                self.dens_o3[:] = 0.0

        # scale to total water vapor content
        if tcwp is not None:
            avogadro = constants.value("Avogadro constant")
            if o3_h2o_alt is None:
                denom = simpson(y=self.dens_h2o, x=-self.z) * 1e5
                self.dens_h2o *= tcwp / M_H2O * avogadro / denom
            else:
                _s_z = np.argsort(self.z)
                # k=1: linear interpolation; BSpline extrapolates
                # linearly beyond the data range by default
                f_dens_h2o = make_interp_spline(
                    self.z[_s_z], self.dens_h2o[_s_z], k=1
                )
                z_alt = np.append(self.z[self.z > o3_h2o_alt], o3_h2o_alt)
                dens_h2o_alt = f_dens_h2o(z_alt)
                h2o_afgl = (
                    simpson(y=dens_h2o_alt, x=-z_alt) * 1e5 * M_H2O
                ) / avogadro
                self.dens_h2o *= tcwp / h2o_afgl
            if tcwp == 0:
                self.dens_h2o[:] = 0.0

        if p0 is not None:
            self.p *= p0 / self.p[-1]

        if not tcno2:
            self.dens_no2[:] = 0.0

    def regrid(self, znew: NDArray) -> "ProfileBase":
        """Regrid atmospheric profile to a new altitude grid.

        Interpolates all atmospheric properties (pressure, temperature,
        and gas densities)
        from the current altitude grid to a new altitude grid using
        linear interpolation.
        Special boundary conditions are applied for pressure (using
        bounds_error=False with
        specific fill values) and temperature (using extrapolation).

        Parameters
        ----------
        znew : ndarray
            New altitude grid in kilometers. Must be a 1-D array of
            altitude values.
            The new grid can be coarser, finer, or irregular compared to
            the original grid.

        Returns
        -------
        ProfileBase
            New ProfileBase object with all atmospheric properties
            interpolated to the
            new altitude grid `znew`. The following attributes are
            interpolated:
            - z: altitude (km)
            - p: pressure (hPa)
            - T: temperature (K)
            - dens_air: air density (molecule/cm³)
            - dens_o3: ozone density (molecule/cm³)
            - dens_o2: oxygen density (molecule/cm³)
            - dens_h2o: water vapor density (molecule/cm³)
            - dens_co2: carbon dioxide density (molecule/cm³)
            - dens_no2: nitrogen dioxide density (molecule/cm³)
            - dens_ch4: methane density (molecule/cm³)
            - dens_co: carbon monoxide density (molecule/cm³)
            - dens_n2o: nitrous oxide density (molecule/cm³)
            - dens_n2: nitrogen density (molecule/cm³)
            - dens_so2: sulfur dioxide density (molecule/cm³)
            - RH_cst: constant relative humidity (None or float)
        """
        znew = np.atleast_1d(znew)
        prof = ProfileBase(None)
        z = self.z
        prof.z = znew
        _s = np.argsort(z)
        try:
            prof.p = np.interp(
                znew, z[_s], self.p[_s], left=1012.0, right=1e-5
            )
        except ValueError:
            print(
                "Error interpolating ({}, {}) -> ({}, {})".format(
                    z[0], z[-1], znew[0], znew[-1]
                )
            )
            print("atm_filename = {}".format(self.fname))
            raise
        # k=1: linear interpolation; BSpline extrapolates linearly
        # beyond the data range by default (replaces fill_value="extrapolate")
        _tmpT = make_interp_spline(z[_s], self.t[_s], k=1)
        prof.t = _tmpT(znew)

        prof.dens_air = np.interp(
            znew, z[_s], self.dens_air[_s], left=0.0, right=0.0
        )
        prof.dens_o3 = np.interp(
            znew, z[_s], self.dens_o3[_s], left=0.0, right=0.0
        )
        prof.dens_o2 = np.interp(
            znew, z[_s], self.dens_o2[_s], left=0.0, right=0.0
        )
        prof.dens_h2o = np.interp(
            znew, z[_s], self.dens_h2o[_s], left=0.0, right=0.0
        )
        prof.dens_co2 = np.interp(
            znew, z[_s], self.dens_co2[_s], left=0.0, right=0.0
        )
        prof.dens_no2 = np.interp(
            znew, z[_s], self.dens_no2[_s], left=0.0, right=0.0
        )
        prof.dens_ch4 = np.interp(
            znew, z[_s], self.dens_ch4[_s], left=0.0, right=0.0
        )
        prof.dens_co = np.interp(
            znew, z[_s], self.dens_co[_s], left=0.0, right=0.0
        )
        prof.dens_n2o = np.interp(
            znew, z[_s], self.dens_n2o[_s], left=0.0, right=0.0
        )
        prof.dens_n2 = np.interp(
            znew, z[_s], self.dens_n2[_s], left=0.0, right=0.0
        )
        prof.dens_so2 = np.interp(
            znew, z[_s], self.dens_so2[_s], left=0.0, right=0.0
        )

        prof.rh_cst = self.rh_cst

        return prof

    def relative_humidity(self) -> NDArray:
        """
        Calculate relative humidity profile for each atmospheric layer.

        Computes the relative humidity at all altitude levels based on
        the atmospheric
        profile's water vapor density, air density, pressure, and
        temperature.

        Returns
        -------
        rh : ndarray
            Relative humidity profile [%] with shape matching altitude
            grid.
            Values can exceed 100% if atmospheric conditions are
            supersaturated.

        Notes
        -----
        The relative humidity is calculated as:

        rh = (p_H₂O / p_sat) x 100

        where:

        - p_H₂O is the partial pressure of water vapor (from density
          ratio)
        - p_sat is the saturation vapor pressure at the given
          temperature

        If RH_cst (constant relative humidity) was set during
        initialization,
        that constant value is returned for all layers instead of
        calculating
        from the density/temperature profile.

        The saturation pressure calculation accounts for both water and
        ice phases
        using temperature-dependent formulas.
        """
        if getattr(self, "rh_cst", None) is not None:
            rh = np.full_like(self.t, self.rh_cst, dtype=np.float64)
        else:
            p_h2o = np.divide(
                self.dens_h2o * self.p,
                self.dens_air,
                out=np.zeros_like(self.dens_h2o),
                where=self.dens_air != 0,
            )
            p_sat = saturation_pressure(self.t) * 1e-2
            rh = (
                np.divide(
                    p_h2o, p_sat, out=np.zeros_like(p_h2o), where=p_sat != 0
                )
                * 100
            )

        return rh


def saturation_pressure(t: NumericArrayLike) -> float | NDArray:
    """Calculate saturation vapor pressure for water and ice phases.

    Uses the Huang (2018) empirical formula, which provides accurate
    saturation vapor pressure calculations for both liquid water and ice
    phases.

    Parameters
    ----------
    t : array_like
        Temperature in Kelvin [K]

    Returns
    -------
    sat_press : float or ndarray
        Saturation vapor pressure [Pa]

    Notes
    -----
    The function automatically selects the appropriate formula based on
    temperature:
    - For T > 273.15 K (0°C): liquid water phase formula
    - For T ≤ 273.15 K (0°C): ice phase formula

    References
    ----------
    Huang, J. (2018). A Simple Accurate Formula for Calculating
    Saturation
    Vapor Pressure of Water and Ice. Journal of Applied Meteorology and
    Climatology, 57(6), 1265-1272.
    """
    tc = np.asarray(t, dtype=np.float64) - 273.15  # temperature in C°
    sat_press = np.zeros_like(tc)

    is_water = tc > 0
    is_ice = np.logical_not(is_water)

    sat_press[is_water] = (
        np.exp(34.494 - 4924.99 / (tc[is_water] + 237.1))
    ) / ((tc[is_water] + 105) ** 1.57)

    sat_press[is_ice] = (np.exp(43.494 - (6545.8 / (tc[is_ice] + 278)))) / (
        (tc[is_ice] + 868) ** 2
    )
    if sat_press.ndim == 0:
        sat_press = float(sat_press)
    return sat_press


def f_n2(wavelength: NumericArrayLike) -> float | NDArray:
    """Compute the depolarization factor of N2 as a function of
    wavelength.

    Parameters
    ----------
    wavelength : array_like
        Wavelength in micrometers (μm).

    Returns
    -------
    float or ndarray
        Depolarization factor of N2. Same shape as input `wavelength`.

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    if wavelength.ndim == 0:
        wavelength = float(wavelength)
    return 1.034 + 3.17 * 1e-4 * wavelength ** (-2)


def f_o2(wavelength: NumericArrayLike) -> float | NDArray:
    """Compute the depolarization factor of O2 as a function of
    wavelength.

    Parameters
    ----------
    wavelength : array_like
        Wavelength in micrometers (μm).

    Returns
    -------
    float or ndarray
        Depolarization factor of O2. Same shape as input `wavelength`.

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    if wavelength.ndim == 0:
        wavelength = float(wavelength)
    return 1.096 + 1.385 * 1e-3 * wavelength ** (-2) + 1.448 * 1e-4 * wavelength ** (-4)


def f_air_co2(wavelength: NumericArrayLike, co2: NumericArrayLike) -> NDArray:
    """Calculates the depolarization factor for air using a composite
    formula based on the depolarization factors of N2 and O2, and the
    CO2 concentration. Produces a 2-D array with one value per
    wavelength-layer combination.

    Parameters
    ----------
    wavelength : array_like
        Wavelength values in micrometers (μm). Shape: (N,)
    co2 : array_like
        CO2 concentration in parts per million (ppm). Shape: (M,) or scalar.

    Returns
    -------
    ndarray
        Depolarization factor of air. Shape: (N, M), where N is the
        number of
        wavelengths and M is the number of layers.

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    fn2_reshp = np.atleast_1d(f_n2(wavelength)).reshape((-1, 1))
    fo2_reshp = np.atleast_1d(f_o2(wavelength)).reshape((-1, 1))
    co2_reshp = np.atleast_1d(co2).reshape((1, -1))

    return (
        78.084 * fn2_reshp
        + 20.946 * fo2_reshp
        + 0.934
        + co2_reshp * 1e-4 * 1.15
    ) / (78.084 + 20.946 + 0.934 + co2_reshp * 1e-4)


def n_air_co2_300(wavelength: NumericArrayLike) -> float | NDArray:
    """Compute the refractive index of dry air at 300 ppm CO2 as a
    function of wavelength.

    Parameters
    ----------
    wavelength : array_like
        Wavelength in micrometers (μm).

    Returns
    -------
    float or ndarray
        Refractive index of dry air at 300 ppm CO2. Same shape as input
        `wavelength`.

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    # ensure scalar input returns scalar output
    if wavelength.ndim == 0:
        wavelength = float(wavelength)
    return (
        1e-8
        * (
            8060.51
            + 2480990 / (132.274 - wavelength ** (-2))
            + 17455.7 / (39.32957 - wavelength ** (-2))
        )
        + 1.0
    )


def n_air_co2(wavelength: NumericArrayLike, co2: NumericArrayLike) -> NDArray:
    """Calculates the refractive index as function of wavelength and CO2
    concentration.

    Parameters
    ----------
    wavelength : array_like
        Wavelength values in micrometers (μm). Shape: (N,)
    co2 : array_like
        CO2 concentration in parts per million (ppm). Shape: (M,) or scalar.

    Returns
    -------
    ndarray
        Refractive index of air. Shape: (N, M), where N is the number of
        wavelengths and M is the number of layers.

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    n300 = np.atleast_1d(n_air_co2_300(wavelength)).reshape((-1, 1))
    co2_reshp = np.atleast_1d(co2).reshape((1, -1))
    return (n300 - 1) * (1 + 0.54 * (co2_reshp * 1e-6 - 0.0003)) + 1.0


def m_dry_air(co2: NumericArrayLike) -> float | NDArray:
    """Compute the mean molecular weight of dry air as a function of CO2
    concentration.

    Parameters
    ----------
    co2 : array_like
        CO2 concentration in parts per million (ppm).

    Returns
    -------
    float or ndarray
        Mean molecular weight of dry air in g/mol. Same shape as input
        `co2`.

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    co2 = np.asarray(co2, dtype=np.float64)
    # ensure scalar input returns scalar output
    if co2.ndim == 0:
        co2 = float(co2)
    return 15.0556 * co2 * 1e-6 + 28.9595


def rayleigh_crs(wavelength: NumericArrayLike, co2: NumericArrayLike) -> NDArray:
    """Compute the Rayleigh cross section.

    Parameters
    ----------
    wavelength : array_like
        The wavelength(s) in um
    co2 : array_like
        CO2 concentration(s) in ppm

    Returns
    -------
    ndarray
        The Rayleigh cross section (N wavelengths x M layers)

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    wavelength = np.atleast_1d(wavelength)
    co2 = np.atleast_1d(co2)

    # Ensure float64 due to numpy 2
    wavelength = wavelength.astype(np.float64)
    co2 = co2.astype(np.float64)

    avogadro = constants.value("Avogadro constant")
    ns = avogadro / 22.4141 * 273.15 / 288.15 * 1e-3
    nn2 = n_air_co2(wavelength, co2) ** 2

    return (
        24
        * np.pi**3
        * (nn2 - 1) ** 2
        / (wavelength[:, None] * 1e-4) ** 4
        / ns**2
        / (nn2 + 2) ** 2
        * f_air_co2(wavelength, co2)
    )


def gravity_z0(lat: NumericArrayLike) -> float | NDArray:
    """Compute gravitational acceleration at Earth's surface as a
    function of latitude.

    Parameters
    ----------
    lat : array_like
        Latitude values in degrees. Positive for North, negative for South.

    Returns
    -------
    float or ndarray
        Gravitational acceleration at ground level in m/s².

    References
    ----------
    .. [1] List, R. J. (1968). *Smithsonian Meteorological Tables*
    (Sixth revised
           edition; fourth reprint issued 1968). Smithsonian Institution
           Press,
           City of Washington, 527 pp.
    """
    lat = np.asarray(lat)
    if lat.ndim == 0:
        lat = float(lat)
    return 980.6160 * (
        1.0
        - 0.0026372 * np.cos(2 * lat * np.pi / 180.0)
        + 0.0000059 * np.cos(2 * lat * np.pi / 180.0) ** 2
    )


def gravity_z(
    lat: RealNumber,
    z: NumericArrayLike,
) -> float | NDArray:
    """Compute gravitational acceleration at a given altitude and
    latitude.

    Parameters
    ----------
    lat : float
        Latitude in degrees as a scalar (Python float or numpy scalar).
        Positive for North, negative for South.
    z : array_like
        Altitude(s) above sea level in meters.

    Returns
    -------
    float or ndarray
        Gravitational acceleration at the given altitude(s) and latitude
        in m/s².

    References
    ----------
    .. [1] List, R. J. (1968). *Smithsonian Meteorological Tables*
    (Sixth revised
           edition; fourth reprint issued 1968). Smithsonian Institution
           Press,
           City of Washington, 527 pp.
    """
    if not isinstance(lat, (float, int, np.floating, np.integer)):
        raise ValueError("The parameter lat must be a scalar value.")

    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 0:
        z = float(z)

    return (
        gravity_z0(lat)
        - (3.085462 * 1.0e-4 + 2.27 * 1.0e-7 * np.cos(2 * lat * np.pi / 180.0))
        * z
        + (7.254 * 1e-11 + 1e-13 * np.cos(2 * lat * np.pi / 180.0)) * z**2
        - (1.517 * 1e-17 + 6 * 1e-20 * np.cos(2 * lat * np.pi / 180.0)) * z**3
    )


def rayleigh_od(
    wavelength: NumericArrayLike,
    co2: NumericArrayLike = 400.0,
    lat: float = 45.0,
    z: NumericArrayLike = 0.0,
    p: NumericArrayLike = 1013.25,
    pressure: str = "surface",
) -> NDArray:
    """Compute Rayleigh optical depth.

    Uses the formulation from Bodhaine et al. (1999) to compute the
    Rayleigh optical depth for given wavelengths and atmospheric
    layers.

    Parameters
    ----------
    wavelength : array_like
        Wavelength(s) in micrometers. Shape (N,).
    co2 : array_like, optional
        CO2 concentration in parts per million (ppm). May be a scalar or
        an array with one entry per layer. Default is 400.0.
    lat : float, optional
        Latitude in degrees used for gravity calculation.
        Default is 45.0.
    z : array_like, optional
        Altitude(s) above sea level in meters. May be a scalar or
        have one entry per layer.
        Default is 0.0.
    p : array_like, optional
        Pressure in hPa. Interpretation depends on the ``pressure`` arg.
        Default is 1013.25.
    pressure : {'surface', 'sea-level'}, optional
        How to interpret ``P``:
        - 'surface' : ``P`` is the pressure at altitude ``z`` (default).
        - 'sea-level' : ``P`` is sea-level pressure and will be reduced
          to the altitude ``z``.

    Returns
    -------
    ndarray
        Rayleigh optical depth. The returned array has shape (N, M),
        where N corresponds to the number of wavelengths and M to the
        number of layers (from inputs such as ``z`` or ``co2``).

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    avogadro = constants.value("Avogadro constant")
    z = np.atleast_1d(z)
    wavelength = np.atleast_1d(wavelength)
    co2 = np.atleast_1d(co2)
    p = np.atleast_1d(p)

    # check that input arrays have compatible shapes
    if co2.shape != z.shape or co2.shape != p.shape:
        raise ValueError(
            "Input arrays co2, z, and p must have the same shape."
        )

    zs = 0.73737 * z + 5517.56  # effective mass-weighted altitude
    g_z = gravity_z(lat, zs)
    # air pressure at the pixel (i.e. at altitude) in hPa
    if pressure == "sea-level":
        # air pressure at pixel location in dyn / cm2, i.e. hPa * 1000
        p_surf = (p * (1.0 - 0.0065 * z / 288.15) ** 5.255) * 1000.0
    elif pressure == "surface":
        p_surf = p * 1000.0  # convert to dyn/cm2
    else:
        raise ValueError(f"Invalid pressure type ({pressure})")

    return rayleigh_crs(wavelength, co2) * p_surf * avogadro / m_dry_air(co2) / g_z


def refractivity(
    wavelength: NumericArrayLike,
    p: NumericArrayLike,
    t: NumericArrayLike,
    co2: NumericArrayLike,
) -> NDArray:
    """Calculate the refractive index of air as a function of
    wavelength, pressure, temperature, and CO2 concentration.

    Parameters
    ----------
    wavelength : array_like
        Wavelength in micrometers (um), shape (N,)
    p : array_like
        Atmospheric pressure in hectopascals (hPa), shape (M,)
    t : array_like
        Temperature in Kelvin (K), shape (M,)
    co2 : array_like
        CO2 concentration in parts per million (ppm), shape (M,)

    Returns
    -------
    ndarray
        Refractive index of air at the given conditions, shape (N, M)

    References
    ----------
    .. [1] Edlén, B. (1966). The refractive index of air. Metrologia,
    2(2), 71-80.
    """
    wavelength = np.atleast_1d(wavelength)
    p = np.atleast_1d(p)  # input pressure in hPa
    t = np.atleast_1d(t)  # input temperature in Kelvin
    co2 = np.atleast_1d(co2)

    # check that input arrays have compatible shapes
    if p.shape != t.shape or p.shape != co2.shape:
        raise ValueError(
            "Input arrays p, t, and co2 must have the same shape."
        )

    p_pa = p * 100.0
    t_c = t - 273.15
    ntp = 1 + (n_air_co2(wavelength[:], co2) - 1) * p * (
        1.0 + p_pa * (60.1 - 0.972 * t_c) * 1e-10
    ) / (96095.43 * (1 + 0.003661 * t_c))
    return ntp


def od2k(
    prof: xr.Dataset,
    dataset: str,
    axis: int = 1,
    zreverse: bool = False,
) -> NDArray:
    """Convert cumulated optical depth to a vertical coefficient
    profile.

    Parameters
    ----------
    prof : Dataset
        Atmospheric profile containing the cumulated optical depth
        dataset
        and the ``z_atm`` vertical coordinate.
    dataset : str
        Name of the cumulated optical depth dataset to convert.
    axis : int, optional
        Axis corresponding to the vertical dimension in ``dataset``.
        Default is 1.
    zreverse : bool, optional
        If True, reverse the vertical axis in the returned array.
        Default is False.

    Returns
    -------
    ndarray
        Two-dimensional array of vertical coefficients in km^-1 with
        shape
        ``(NW, NZ)``.
    """
    if hasattr(prof, "to_xarray"):
        prof = prof.to_xarray()

    ot = diff1(
        prof[dataset].to_numpy().astype(np.float32, copy=False), axis=axis
    )
    dz = diff1(prof.coords["z_atm"].to_numpy().astype(np.float32, copy=False))

    with np.errstate(invalid="ignore", divide="ignore"):
        k = abs(ot / dz)
    k[np.isnan(k)] = 0
    sl = slice(None, None, -1 if zreverse else 1)

    return k[:, sl]


def blackbody_radiance(
    wavelength: NumericArrayLike, T: NumericArrayLike
) -> float | NDArray:
    """
    Calculate the spectral blackbody radiance.

    Computes the spectral radiance of a perfectly emitting blackbody
    at a given wavelength and temperature according to Planck's law
    of blackbody radiation.

    Parameters
    ----------
    wavelength : array_like
        Wavelength in meters.
    T : array_like
        Temperature in Kelvin.

    Returns
    -------
    L_b_wavelength : NDArray
        Spectral radiance in W·m⁻³·sr⁻¹.

    References
    ----------
    .. [1] Lenoble, J. (1993). Atmospheric radiative transfer.
           A. Deepak Publishing.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.constants import speed_of_light, Planck, Boltzmann
    >>> wavelength = 10e-6  # 10 micrometers (thermal infrared)
    >>> T = 288.0    # 288 K (room temperature)
    >>> L_b_wavelength = blackbody_radiance(wavelength, T)
    >>> print(f"Spectral radiance: {L_b_wavelength:.2e} W·m⁻³·sr⁻¹")

    >>> # Calculate for multiple wavelengths at a fixed temperature
    >>> wavelengths = np.array([0.5e-6, 1e-6, 10e-6]) # UV, NIR, TIR
    >>> T = 5778  # Sun's surface temperature
    >>> L_b_wavelength = blackbody_radiance(wavelengths, T)
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    scalar_input = wavelength.ndim == 0 and T.ndim == 0
    try:
        np.broadcast_shapes(wavelength.shape, T.shape)
    except ValueError as err:
        raise ValueError("wavelength and T must be broadcastable") from err

    c1 = 2.0 * Planck * speed_of_light**2
    c2 = Planck * speed_of_light / Boltzmann
    L_b_wavelength = c1 / (
        (wavelength**5) * (np.exp(c2 / (wavelength * T)) - 1.0)
    )
    if scalar_input:
        return float(L_b_wavelength)
    return L_b_wavelength


def get_aer_dist_integral(
    z: NumericArrayLike,
    h_min: NumericArrayLike,
    h_max: NumericArrayLike,
) -> NDArray:
    """
    Compute the integral of exponential vertical distribution between
    two altitudes.

    Calculates the integral of an exponential distribution function
    over a vertical layer, used for computing the optical depth
    contribution of aerosols or clouds with a scale height `z` between
    altitudes `h_min` and `h_max`.

    Parameters
    ----------
    z : array_like
        Scale height in km. Defines the vertical distribution as
        N(h) = N(0)*exp(-h/z).
    h_min : array_like
        Minimum altitude in km (bottom of the layer).
    h_max : array_like
        Maximum altitude in km (top of the layer).

    Returns
    -------
    ndarray
        Integral of the exponential distribution between `h_min` and
        `h_max`, normalized by `z`.
    """
    z = np.atleast_1d(z)
    h_min = np.atleast_1d(h_min)
    h_max = np.atleast_1d(h_max)
    return -(z) * np.exp(-h_max / z) + (z) * np.exp(-h_min / z)


def check_date(dates: Iterable[str] | NDArray[np.str_], year: int) -> None:
    """Validate that all dates are from a single year and match the
    provided year.

    Parameters
    ----------
    dates : array-like of str
        Dates in format "dd:mm:yyyy"
    year : int
        Expected year in format yyyy

    Raises
    ------
    ValueError
        If dates contain multiple years or if the year doesn't match the
        expected year

    Returns
    -------
    None
    """
    # Normalize input to a list of strings so we accept numpy/pandas arrays
    dates_list = [str(d) for d in dates]

    if len(dates_list) == 0:
        raise ValueError("dates cannot be empty")

    # Extract years from dates using list comprehension
    years = np.unique([int(date.split(":")[-1]) for date in dates_list])

    if years.size != 1:
        raise ValueError(
            f"Multiple years found in dates: {years}. "
            "Data spanning multiple years is not supported for 'day of year'"
            " dimension."
        )

    extracted_year = years[0]
    if extracted_year != year:
        raise ValueError(
            f"Date year ({extracted_year}) does not match expected year"
            + f" ({year})."
        )


def read_aeronet_aod(file: PathType, year: int) -> xr.DataArray:
    """Extract AOD data from Aeronet file.

    Parameters
    ----------
    file : path-like
        Extinction AOD aeronet file path
    year : int
        The year for 'Day_of_Year(Fraction)' dimension creation

    Returns
    -------
    aod_ext_da : DataArray
        Lookup table with extinction AOD as function of
        Day_of_Year(Fraction) and wavelength
    """
    aod = pd.read_csv(file, sep=",", skiprows=6)
    ntime_aod = aod.index.size

    check_date(dates=aod["Date(dd:mm:yyyy)"].values, year=year)

    wavelength_ext = []
    for key in aod.keys():
        if "AOD_Extinction-Total" in key:
            str_bis = key.split("[")
            wavelength_ext.append(float(str_bis[1][:-3]))
    wavelength_ext = np.unique(wavelength_ext)
    n_wavelength_ext = len(wavelength_ext)

    mat_ext = np.zeros((ntime_aod, n_wavelength_ext), dtype=np.float64)
    for itime in range(0, ntime_aod):
        for i_wavelength, wavelength in enumerate(wavelength_ext):
            key = "AOD_Extinction-Total[" + str(int(wavelength)) + "nm]"
            mat_ext[itime, i_wavelength] = aod.iloc[itime][key]

    aod_ext_da = xr.DataArray(
        mat_ext,
        coords={
            "Day_of_Year(Fraction)": aod["Day_of_Year(Fraction)"].values,
            "wavelength": wavelength_ext,
        },
        dims=["Day_of_Year(Fraction)", "wavelength"],
        name="aod",
    )

    return aod_ext_da


def read_aeronet_ssa(file: PathType, year: int) -> xr.DataArray:
    """Extract SSA data from Aeronet file.

    Parameters
    ----------
    file : path-like
        Single scattering albedo aeronet file path
    year : int
        The year for 'Day_of_Year(Fraction)' dimension creation

    Returns
    -------
    ssa_da : DataArray
        Lookup table with single scattering albedo as function of
        Day_of_Year(Fraction)
        and wavelength
    """
    ssa = pd.read_csv(file, sep=",", skiprows=6)
    ntime_ssa = ssa.index.size

    check_date(dates=ssa["Date(dd:mm:yyyy)"].values, year=year)

    wavelength_ssa = []
    for key in ssa.keys():
        if "Single_Scattering_Albedo" in key:
            str_bis = key.split("[")
            wavelength_ssa.append(float(str_bis[1][:-3]))
    wavelength_ssa = np.unique(wavelength_ssa)
    n_wavelength_ssa = len(wavelength_ssa)

    mat_ssa = np.zeros((ntime_ssa, n_wavelength_ssa), dtype=np.float64)
    for itime in range(0, ntime_ssa):
        for i_wavelength, wavelength in enumerate(wavelength_ssa):
            key = "Single_Scattering_Albedo[" + str(int(wavelength)) + "nm]"
            mat_ssa[itime, i_wavelength] = ssa.iloc[itime][key]

    ssa_da = xr.DataArray(
        mat_ssa,
        coords={
            "Day_of_Year(Fraction)": ssa["Day_of_Year(Fraction)"].values,
            "wavelength": wavelength_ssa,
        },
        dims=["Day_of_Year(Fraction)", "wavelength"],
        name="ssa",
    )

    return ssa_da


def read_aeronet_pfn(file: PathType, year: int) -> xr.DataArray:
    """Extract PFN data from Aeronet file.

    Parameters
    ----------
    file : path-like
        Phase matrix aeronet file path
    year : int
        The year for 'Day_of_Year(Fraction)' dimension creation

    Returns
    -------
    phase_da : DataArray
        Lookup table with phase function matrix as function of
        Day_of_Year(Fraction),
        wavelength and theta_atm
    """
    pfn = pd.read_csv(file, sep=",", skiprows=6)
    # take only total of fine + coarse
    pfn = cast(pd.DataFrame, pfn[pfn["Phase_Function_Mode"] == "Total"])
    ntime_pfn = pfn.index.size

    check_date(dates=pfn["Date(dd:mm:yyyy)"].values, year=year)

    ang = []
    wavelength_pfn = []
    for key in pfn.keys():
        if "0000" in key:
            str_bis = key.split("[")
            ang.append(float(str_bis[0]))
            wavelength_pfn.append(float(str_bis[1][:-3]))
    ang = np.unique(ang)[::-1]
    wavelength_pfn = np.unique(wavelength_pfn)
    n_ang = len(ang)
    n_wavelength_pfn = len(wavelength_pfn)

    mat_pfn = np.zeros((ntime_pfn, n_wavelength_pfn, n_ang), dtype=np.float64)
    for itime in range(0, ntime_pfn):
        for i_wavelength, wavelength in enumerate(wavelength_pfn):
            for iang, ag in enumerate(ang):
                ang_str = "%.6f" % float(ag)
                key = ang_str + "[" + str(int(wavelength)) + "nm]"
                mat_pfn[itime, i_wavelength, iang] = pfn.iloc[itime][key]

    phase_da = xr.DataArray(
        mat_pfn,
        coords={
            "Day_of_Year(Fraction)": pfn["Day_of_Year(Fraction)"].values,
            "wavelength": wavelength_pfn,
            "theta_atm": ang,
        },
        dims=["Day_of_Year(Fraction)", "wavelength", "theta_atm"],
        name="pfn",
    )

    return phase_da


def atm_pro_from_aeronet(
    date: str,
    time: str,
    aod_file: str | xr.DataArray,
    ssa_file: str | xr.DataArray,
    pfn_file: str | xr.DataArray,
    b_wavelength: NumericArrayLike | BandSet,
    wavelength_phase: NumericArrayLike | None = None,
    grid: NumericArrayLike | None = None,
    atm_name: str = "afglt",
    P0: float | None = None,
    O3: float | None = None,
    H2O: float | None = None,
    O3_H2O_alt: float | None = None,
    h_mix_min: float = 0.0,
    h_mix_max: float = 2.0,
    z_mix: float = 8,
) -> xr.Dataset:
    """
    Create an atmosphere profil from aeronet files

    Parameters
    ----------
    date : str
        Date in the following format -> "yyyy-mm-dd"
    time : str
        Time in the following format -> "hh:mm:ss"
    aod_file : str or xr.DataArray
        Extinction AOD aeronet file (finishing by .aod) or aod DataArray
    ssa_file : str or xr.DataArray
        Single scattering albedo aeronet file (finishing by .ssa) or ssa
        DataArray
    pfn_file : str or xr.DataArray
        Phase matrix aeronet file (finishing by .pfn) or pfn DataArray
    b_wavelength : array_like or BandSet, optional
        Kdis bands or list of wavelengths
    wavelength_phase : array_like or None, optional
        List of wavelengths where the phase functions are computed
    grid : array_like or None, optional
        Altitude grid profil
    atm_name : str, optional
        The atmAFGL atmosphere used
    P0 : float, optional
        Surface pressure
    O3 : float, optional
        Scale ozone vertical column (Dobson units)
    H2O : float, optional
        Scale Water vertical column
    O3_H2O_alt : float or None, optional
        Altitude of H2O and O3 values, by default None and scale from
        z=0km
    h_mix_min : float, optional
        Force min altitude of the mixture
    h_mix_max : float, optional
        Force max altitude of the mixture
    z_mix : float, optional
        Force scale height (see notes) of the mixture

    Returns
    -------
    pro : Dataset
        The atmophere profil. Similar to the output of the calc method
        of Atm1D.

    Notes
    -----
    The scale height (see Hess et al. 2004) is the variable Z in the
    following equation:

    - :math:`N(h) = N(0)exp(-h/Z)`

    with N the number density and h the altitude
    """

    pd_date = pd.Timestamp(date + " " + time)
    n_sec_day = 24 * 60 * 60  # number of seconds in one day
    day_frac = 1 - (
        (
            n_sec_day
            - (pd_date.hour * 60 * 60 + pd_date.minute * 60 + pd_date.second)
        )
        / n_sec_day
    )
    day_year_frac = pd_date.day_of_year + day_frac
    print("day_year_frac =", day_year_frac)
    year = int(pd_date.year)

    if isinstance(aod_file, xr.DataArray):
        aod_lut = aod_file
    else:
        aod_lut = read_aeronet_aod(aod_file, year=year)
    if isinstance(ssa_file, xr.DataArray):
        ssa_lut = ssa_file
    else:
        ssa_lut = read_aeronet_ssa(ssa_file, year=year)
    if isinstance(pfn_file, xr.DataArray):
        pfn_lut = pfn_file
    else:
        pfn_lut = read_aeronet_pfn(pfn_file, year=year)

    if not isinstance(b_wavelength, BandSet):
        b_wavelength_BS = BandSet(b_wavelength)
    else:
        b_wavelength_BS = b_wavelength
    b_wavelength_unique = np.unique(b_wavelength_BS.wavelength)

    if wavelength_phase is None:
        pf_wavelength = b_wavelength_unique
    else:
        pf_wavelength = wavelength_phase

    fv_time = "extrapolate"
    aod_lut = aod_lut.interp(
        {
            "Day_of_Year(Fraction)": day_year_frac,
            "wavelength": b_wavelength_unique,
        },
        method="linear",
        kwargs={"fill_value": fv_time},
    ).drop_vars("Day_of_Year(Fraction)")
    ssa_lut = ssa_lut.interp(
        {
            "Day_of_Year(Fraction)": day_year_frac,
            "wavelength": b_wavelength_unique,
        },
        method="linear",
        kwargs={"fill_value": fv_time},
    ).drop_vars("Day_of_Year(Fraction)")
    pfn_lut = pfn_lut.interp(
        {
            "Day_of_Year(Fraction)": day_year_frac,
            "wavelength": b_wavelength_unique,
        },
        method="linear",
        kwargs={"fill_value": fv_time},
    ).drop_vars("Day_of_Year(Fraction)")

    aod_lut = aod_lut.where(aod_lut >= 0, 0)
    ssa_lut = ssa_lut.where(
        (ssa_lut >= 0) & (ssa_lut <= 1), np.clip(ssa_lut, 0, 1)
    )
    pfn_lut = pfn_lut.where(pfn_lut >= 0, 0)

    pfn_val = pfn_lut.values
    pfn_val = np.stack([pfn_val[:, :]] * 4, axis=1)
    pfn_val[:, 2:3, :] = 0.0
    pfn_lut = xr.DataArray(
        pfn_val,
        dims=["wavelength", "nphamat", "theta_atm"],
        coords={
            "wavelength": pfn_lut.wavelength,
            "nphamat": np.arange(4),
            "theta_atm": pfn_lut.theta_atm,
        },
    )

    hum = np.array([0.0])
    wavelength = aod_lut.wavelength.values.copy()
    theta = pfn_lut.theta_atm.values.copy()
    aod = aod_lut.values[None, :]
    ssa = ssa_lut.values[None, :]
    phase = pfn_lut.values[None, :, :, :]

    aer = AerUser(
        aod,
        ssa,
        phase,
        hum,
        wavelength,
        theta,
        h_mix_min=h_mix_min,
        h_mix_max=h_mix_max,
        z_mix=z_mix,
    )
    pro = Atm1D(
        atm_name,
        comp=[aer],
        grid=grid,
        p0=P0,
        tco3=O3,
        tcwp=H2O,
        wavelength_phase=pf_wavelength,
        o3_h2o_alt=O3_H2O_alt,
    ).calc(b_wavelength_BS)

    return pro


def _open_lut_datatree_as_xarray(
    input_path: PathType,
    group: str | None = None,
    datasets: Iterable[str] | None = None,
) -> xr.Dataset:
    """Read a LUT-style HDF5 group into an xarray.Dataset using
    xarray.open_datatree."""
    data_vars = {}

    tree = xr.open_datatree(input_path)
    try:
        group_path = (
            "/" if group in (None, "") else f"/{str(group).strip('/')}"
        )
        axis_path = "/axis" if group_path == "/" else f"{group_path}/axis"
        data_path = "/data" if group_path == "/" else f"{group_path}/data"

        axis_ds = tree[axis_path].to_dataset(inherit=False).load()
        data_ds = tree[data_path].to_dataset(inherit=False).load()
    finally:
        tree.close()

    if datasets is None:
        dataset_names = list(data_ds.data_vars)
    else:
        dataset_names = list(datasets)

    coords = {name: axis_ds[name].to_numpy() for name in axis_ds.data_vars}

    for name in dataset_names:
        if name not in data_ds:
            raise KeyError(f'Dataset "{name}" not available in DataTree group')

        dimensions = data_ds[name].attrs.get("dimensions")
        if dimensions is None:
            raise ValueError(
                f'Missing dimensions attribute for dataset "{name}"'
            )
        if isinstance(dimensions, bytes):
            dimensions = dimensions.decode()
        dims = tuple(
            dim.strip() for dim in dimensions.split(",") if dim.strip()
        )

        if len(dims) != data_ds[name].ndim:
            raise ValueError(
                f'Dataset "{name}" declares {len(dims)} dimensions but has'
                + f" {data_ds[name].ndim} axes"
            )

        data_vars[name] = (dims, data_ds[name].to_numpy())

    return xr.Dataset(data_vars=data_vars, coords=coords)


def artdeco_to_smartg_cld(
    input_path: PathType,
    output_path: PathType | None = None,
    h5_group: str | None = None,
    normalize: bool = True,
    overwrite: bool = False,
    veff: float | None = None,
    wavelength_max: float = 4500,
) -> xr.Dataset:
    """Convert ARTDECO cloud HDF5 file to SMART-G NetCDF file format.

    Reads cloud optical properties from an ARTDECO HDF5 file and
    converts them
    to an xarray dataset compatible with SMART-G cloud inputs.

    Parameters
    ----------
    input_path : path-like
        Path to the ARTDECO cloud HDF5 file.
    output_path : path-like, optional
        Output path for saving the converted SMART-G cloud NetCDF file.
        If None, the converted data is not saved to disk. Default: None
    h5_group : str, optional
        Group name within the HDF5 file to open. If None and the file
        contains
        only one group, that group is automatically selected. If the
        file contains
        multiple groups, a group name must be specified. Default: None
    normalize : bool, optional
        If True (default), normalize the p11 phase matrix component
        integral to 2.
        Default: True
    overwrite : bool, optional
        If True and output_path is given, overwrite existing file.
        Default: False
    veff : float, optional
        Effective volume fraction. Required if cloud properties are
        dependent on veff.
        Default: None
    wavelength_max : float, optional
        Maximum wavelength in nanometers. Only wavelengths <=
        wavelength_max are included.
        Default: 4500

    Returns
    -------
    ds : Dataset
        Dataset containing cloud optical properties. Includes
        coordinates:

        - reff: effective radius
        - wavelength: wavelength (nm)
        - stk: phase matrix unique terms (4 or 6)
        - theta: scattering angle (degrees)

        And datasets:

        - phase: phase matrix (normalized to 2 if normalize=True)
        - ext: extinction coefficient (km⁻¹)
        - ssa: single scattering albedo
    """
    # Deals with the case where h5_group is not provided
    if h5_group is None:
        tree = xr.open_datatree(input_path)
        keys = None
        try:
            if {"axis", "data"}.issubset(tree.children):
                h5_group = None
            else:
                keys = [
                    name
                    for name, child in tree.children.items()
                    if {"axis", "data"}.issubset(child.children)
                ]
        finally:
            tree.close()

        if h5_group is None and keys is not None:
            if len(keys) == 1:
                h5_group = keys[0]
            elif len(keys) > 1:
                raise ValueError(
                    "The h5 file has more than one group. Please choose one "
                    + "group between: "
                    + ", ".join(keys)
                )
            else:
                raise ValueError(
                    "Could not identify an ARTDECO group containing axis and "
                    + "data nodes."
                )

    art_cld = _open_lut_datatree_as_xarray(input_path, group=h5_group)

    # If p22 doesn't exist --> convention with 4 stk components
    # Care, phase_comp elements are sorted in a specific way
    if "p22_phase_function" in art_cld:
        nstk = int(6)
        phase_comp = [
            "p11_phase_function",
            "p21_phase_function",
            "p33_phase_function",
            "p34_phase_function",
            "p22_phase_function",
            "p44_phase_function",
        ]
    else:
        nstk = int(4)
        # here p33 = p44
        phase_comp = [
            "p11_phase_function",
            "p21_phase_function",
            "p44_phase_function",
            "p34_phase_function",
        ]

    # check if the cloud properties are dependant of veff
    is_veff = "veff" in art_cld.coords

    if is_veff and veff is None:
        veff_min = str(float(art_cld.coords["veff"].min()))
        veff_max = str(float(art_cld.coords["veff"].max()))
        raise ValueError(
            "The cloud file is dependant of veff. Please give a veff value "
            + f"between: {veff_min} and {veff_max}"
        )
    elif is_veff:
        art_cld = art_cld.interp(veff=np.array([veff])).squeeze(
            "veff", drop=True
        )

    reff = art_cld.coords["reff"].to_numpy().astype(np.float32, copy=False)
    nreff = len(reff)

    wavelength_full = np.round(
        art_cld.coords["wavelengths"].to_numpy().astype(np.float64, copy=False)
        * 1e3,
        decimals=3,
    ).astype(np.float32, copy=False)
    wavelength_idx = np.flatnonzero(wavelength_full <= wavelength_max)
    wavelength = wavelength_full[wavelength_idx]
    n_wavelength = len(wavelength)

    stk = np.arange(nstk, dtype=np.int16)

    mu = art_cld.coords["mu"].to_numpy().astype(np.float64, copy=False)
    theta_unsorted = np.rad2deg(np.arccos(mu))
    theta_idx = np.argsort(theta_unsorted)
    theta = theta_unsorted[theta_idx]
    mu_sorted = mu[theta_idx]
    ntheta = len(theta)

    phase = np.zeros((nreff, n_wavelength, nstk, ntheta), dtype=np.float32)
    for ipc, pc in enumerate(phase_comp):
        phac = art_cld[pc].transpose("reff", "wavelengths", "mu")
        phac = phac.isel(wavelengths=wavelength_idx, mu=theta_idx)
        phase[:, :, ipc, :] = phac.to_numpy().astype(np.float32, copy=False)

    if nstk == 6:
        pha_desc = (
            "phase matrix integral normalized to 2. stk order: p11, "
            + "p21, p33, p34, p22 and p44"
        )
    else:  # nstk == 4
        pha_desc = (
            "phase matrix integral normalized to 2. stk order: p11, "
            + "p21, p33 and p34"
        )

    # integral of P11 must be equal to 2
    if normalize:
        for i_wavelength in range(0, n_wavelength):
            for ireff in range(0, nreff):
                f = phase[ireff, i_wavelength, 0, :]  # P11
                norm = np.trapezoid(f, -mu_sorted)
                phase[ireff, i_wavelength, :, :] *= 2.0 / abs(norm)

    ext = (
        art_cld["Cext"]
        .transpose("reff", "wavelengths")
        .isel(wavelengths=wavelength_idx)
    )
    ssa = (
        art_cld["single_scattering_albedo"]
        .transpose("reff", "wavelengths")
        .isel(wavelengths=wavelength_idx)
    )

    ds = xr.Dataset(
        data_vars={
            "phase": (
                ("reff", "wav", "stk", "theta"),
                phase,
                {"description": pha_desc},
            ),
            "ext": (
                ("reff", "wav"),
                ext.to_numpy().astype(np.float64, copy=False),
                {"description": "extinction coefficient in km^-1"},
            ),
            "ssa": (
                ("reff", "wav"),
                ssa.to_numpy().astype(np.float64, copy=False),
                {"description": "single scattering albedo"},
            ),
        },
        coords={
            "reff": reff,
            "wav": wavelength,
            "stk": stk,
            "theta": theta,
        },
    )

    if veff is not None:
        ds.attrs["veff"] = veff

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"{output_path} already exists")
        ds.to_netcdf(output_path)

    return ds


def extract_split(
    ds_sg: xr.Dataset,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, list[xr.DataArray]],
]:
    """
    Use SMART-G run results to compute atmospheric optical
    properties at specified wavelengths and separates them into
    decomposed
    components (absorption, Rayleigh scattering, aerosols, and phase
    functions).
    These returned profiles can be used as alternative inputs to
    initialize a
    new Atm1D instance.

    Parameters
    ----------
    ds_sg : Dataset
        SMART-G run results containing the atmospheric optical
        properties.
        Must include the following datasets:

        - OD_p: particulate optical depth
        - OD_r: Rayleigh optical depth
        - OD_g: gaseous optical depth
        - ssa_p_atm: single scattering albedo of particles
        - iphase_atm: phase function indices
        - phase_atm: phase matrix function

    Returns
    -------
    prof_abs : ndarray
        Gaseous absorption optical depth profile.
    prof_ray : ndarray
        Rayleigh optical depth profile.
    prof_aer : tuple of (ndarray, ndarray)
        Tuple containing:

        - prof_aer[0]: Aerosol optical depth profile
        - prof_aer[1]: Single scattering albedo profile of aerosols
    prof_phase : tuple of (ndarray, list)
        Tuple containing:

        - prof_phase[0]: Phase function indices (iphase_atm)
        - prof_phase[1]: List of phase matrix DataArray objects for each
          index

    Examples
    --------
    >>> from smartg.atmosphere import extract_split, Atm1D
    >>> prof_abs, prof_ray, prof_aer, \
    ...     prof_phases = extract_split(mlut_result)
    >>> new_atm = Atm1D('afglt', prof_abs=prof_abs, prof_ray=prof_ray,
    ...     prof_aer=prof_aer, prof_phases=prof_phases)
    """
    if hasattr(ds_sg, "to_xarray"):
        ds_sg = ds_sg.to_xarray()
    pro_aer = diff1(
        ds_sg["OD_p"].to_numpy().astype(np.float32, copy=False), axis=1
    )
    ssa_aer = ds_sg["ssa_p_atm"].to_numpy()
    pro_ray = diff1(
        ds_sg["OD_r"].to_numpy().astype(np.float32, copy=False), axis=1
    )
    pro_abs = diff1(
        ds_sg["OD_g"].to_numpy().astype(np.float32, copy=False), axis=1
    )
    pro_iphase = ds_sg["iphase_atm"].to_numpy()
    pro_phases = [
        ds_sg["phase_atm"].isel(iphase=i, drop=True)
        for i in range(int(pro_iphase.max()) + 1)
    ]

    return pro_abs, pro_ray, (pro_aer, ssa_aer), (pro_iphase, pro_phases)


def strgrid_to_numpy(str_grid: str) -> np.ndarray:
    """
    Convert altitude grid specification string to numpy array.

    This function adopts py4cats' compact grid specification format,
    providing
    py4cats users with familiar syntax for altitude grid definition in
    SMARTG.

    Parameters
    ----------
    str_grid : str
        Compact grid specification string describing a piecewise-linear
        altitude
        grid. Format: 'start[step1]stop1[step2]stop2[step3]stop3...'

        Each segment is defined by:
        - start: starting altitude value (float, int, or scientific
          notation)
        - [step]: step size enclosed in square brackets
        - stop: ending altitude value (float, int, or scientific
          notation)

        Supports both positive and negative steps. Results are always
        monotonic
        across all segments.

    Returns
    -------
    ndarray
        1D array of altitude values. The array is sorted and contains
        the
        generated grid points covering all specified segments.

    Notes
    -----
    - Each segment creates a uniformly spaced array using numpy.linspace
    - The final endpoint is always included in the output
    - Intermediate endpoints between segments are included with their
      exact value
    - Step sizes can be positive or negative
    - Supports scientific notation (e.g., 1e-3, 2.5E+2)


    Examples
    --------
    Simple grid from TOA to ground (100 km to 0 km with step 1 km):

    >>> grid = strgrid_to_numpy('100[1]0')
    >>> grid
    array([100.,  99.,  98., ...,   2.,   1.,   0.])
    >>> len(grid)
    101

    Multi-segment grid with varying resolution (TOA to ground):

    >>> grid = strgrid_to_numpy('500[10]100[1]0')
    >>> grid[:5]
    array([500., 490., 480., 470., 460.])
    >>> grid[40:43]
    array([100.,  99.,  98.])

    Grid with scientific notation:

    >>> grid = strgrid_to_numpy('1[1e-1]1e-1[1e-2]0')
    >>> grid
    array([1.  , 0.9 , 0.8 , 0.7 , 0.6 , 0.5 , 0.4 , 0.3 , 0.2 , 0.1 ,
           0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.  ])
    >>> len(grid)
    20
    """

    # Split by bracketed steps to extract numbers and steps separately
    # re.split with capturing group keeps the steps
    # Result: [start, step1, stop1, step2, stop2, ...]
    parts = re.split(r"\[([\d.eE+-]+)\]", str_grid)

    if len(parts) < 3 or len(parts) % 2 == 0:
        raise ValueError(
            f'Cannot parse grid specification: "{str_grid}"\n'
            "Expected format: start[step]stop[step]stop...\n"
            'Example: "0[1]100[10]500"'
        )

    # Extract numbers (at even indices) and steps (at odd indices)
    numbers_str = [parts[i] for i in range(0, len(parts), 2)]
    steps_str = [parts[i] for i in range(1, len(parts), 2)]

    # Validate we have sensible input
    if not numbers_str or not steps_str:
        raise ValueError(
            f'Cannot parse grid specification: "{str_grid}"\n'
            "Expected format: start[step]stop[step]stop...\n"
            'Example: "0[1]100[10]500"'
        )

    # Convert to floats
    try:
        numbers = [float(x) for x in numbers_str]
        steps = [float(x) for x in steps_str]
    except ValueError as e:
        raise ValueError(
            f"Invalid numeric value in grid specification: {e}"
        ) from e

    # Validate steps are non-zero
    if any(step == 0 for step in steps):
        raise ValueError("Step size cannot be zero")

    # Convert to numpy arrays for vectorized operations
    numbers = np.asarray(numbers)
    steps = np.asarray(steps)

    # Vectorized calculation of points per segment
    n_array = np.round(np.abs(np.diff(numbers) / steps)).astype(int)
    n_array[-1] += 1  # Ensure final endpoint is included

    # Build piecewise linear grid
    segments: list[np.ndarray] = [
        np.linspace(
            float(numbers[i]),
            float(numbers[i + 1]),
            int(n),
            endpoint=(i == len(steps) - 1),
        )
        for i, n in enumerate(n_array.tolist())
    ]
    return np.concatenate(segments)
