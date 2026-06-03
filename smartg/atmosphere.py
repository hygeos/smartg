#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
1. Create an atmospheric profile using model classes (e.g., AtmAFGL)
2. Add atmospheric components (aerosols, clouds, surface) as needed
3. (Optional) Call the profile's `calc()` method to compute optical
   properties
   with specific parameters (if using optional parameters not set by
   default)
4. Pass the resulting profile object as the `atm` parameter to
   `smartg.run()`

Key Classes
-----------
AtmAFGL
    AFGL Standard U.S. Atmosphere model. Provides vertical temperature
    and
    pressure profiles. Aerosols, clouds, and ocean surface can be added
    to
    build a complete atmospheric model.

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
"""

import numpy as np
from pathlib import Path
from typing import Iterable
from os import PathLike
from smartg.phase import calc_iphase
from scipy.interpolate import make_interp_spline
from scipy.integrate import simpson
from scipy import constants
from scipy.constants import speed_of_light, Planck, Boltzmann
from smartg.bandset import BandSet
from smartg.config import DIR_AUXDATA
from gatiab import vec_float_indexing
import pandas as pd
import xarray as xr
import re
from pytrunc.truncation import delta_m_phase_approx, gt_phase_approx


class AerOPAC(object):
    """
    Initialize the Aerosol OPAC model

    Parameters
    ----------
    filename : str
        Complete path to the aerosol file or filename for aerosols
        located in "auxdata/aerosols/OPAC/mixtures/".
        Available auxdata aerosols: antarctic, antarctic_spheric,
        arctic, continental_average,
        continental_clean, continental_polluted, desert, desert_spheric,
        maritime_clean,
        maritime_polluted, mineral_transported, maritime_tropical and
        urban
    tau_ref : float
        Optical thickness at reference wavelength w_ref
    w_ref : float
        Wavelength in nanometers at reference optical depth tau_ref
    H_mix_min : float, optional
        Force min altitude of the mixture
    H_mix_max : float, optional
        Force max altitude of the mixture
    H_free_min : float, optional
        Force min altitude of the free troposphere
    H_free_max : float, optional
        Force max altitude of the free troposphere
    H_stra_min : float, optional
        Force min altitude of the stratosphere
    H_stra_max : float, optional
        Force max altitude of the stratosphere
    Z_mix : float, optional
        Force scale height (see notes) of the mixture
    Z_free : float, optional
        Force scale height (see notes) of the free troposphere
    Z_stra : float, optional
        Force scale height (see notes) of the stratosphere
    ssa : None | float | list | 1-D ndarray | 2-D ndarray |
    xr.DataArray, optional
        Force particle single scattering albedo. Default None.

        - if float -> same value for all wavelengths and altitudes
        - if list -> it will be converted into a 1-D ndarray.
        - if 1-D ndarray -> only wavelength dependence is considered
        - if 2-D ndarray -> wavelength and altitude dependence is
          considered
        - if xr.DataArray -> wavelength and altitude dependence is
          considered

        Note that xr.DataArray is more flexible since it allows
        interpolation if wavelengths
        in calc method are different (but not the case for the altitude
        axis).
    phase : None | xr.DataArray, optional
        Phase matrix F as function of wavelength, altitude, stoke
        components and scattering angle
        The variable names must be:
        If 4-D matrix -> wav_phase, z_phase, stk, theta
        If 2-D matrix (assumed monochromatic and constant vertically) ->
        stk, theta
        Where:
        - wav_phase is the wavelength. It must be equal to the `pfwav`
          parameter of AtmAFGL
          if defined, else `wav` parameter vavelengths of the AtmAFGL
          calc method.
        - z_phase is the phase altitude. It must be equal to the
          `pfgrid[1:]` parameter
          of AtmAFGL
        - stk the phase matrix unique terms.
        - theta the scattering angle.

        The phase matrix terms (IQUV convention) must be given in the
        folowing order:
        - F11, F21, F33 and F34 if only 4 terms are given (only for
          spherical particles)
        - F11, F21, F33, F34, F22 and F44 if 6 terms are given (for both
          spherical and non-spherical particles)
    rh_mix/free/stra : None | float, optional
        Force relative humidity of mixture/free tropo/strato. Default
        None.

    Notes
    -----
    The scale height (see Hess et al. 2004) is the variable Z in the
    following equation:

    - :math:`N(h) = N(0)exp(-h/Z)`

    with N the number density and h the altitude

    Examples
    --------
    >>> from smartg.atmosphere import AerOPAC
    >>> aer_mc = AerOPAC('maritime_clean', 0.1, 550.)
    >>> print(aer_mc.mixture)
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
        H_mix_min:   0
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
        filename: str | Path,
        tau_ref: float,
        w_ref: float,
        H_mix_min: float | None = None,
        H_mix_max: float | None = None,
        H_free_min: float | None = None,
        H_free_max: float | None = None,
        H_stra_min: float | None = None,
        H_stra_max: float | None = None,
        Z_mix: float | None = None,
        Z_free: float | None = None,
        Z_stra: float | None = None,
        ssa: float | list[float] | np.ndarray | xr.DataArray | None = None,
        phase: xr.DataArray | None = None,
        rh_mix: float | None = None,
        rh_free: float | None = None,
        rh_stra: float | None = None,
    ) -> None:

        self.tau_ref = (
            tau_ref.to_xarray() if hasattr(tau_ref, "to_xarray") else tau_ref
        )
        if np.isscalar(w_ref) or (
            isinstance(w_ref, np.ndarray) and w_ref.ndim == 0
        ):
            self.w_ref = np.array([w_ref])
        else:
            self.w_ref = np.array(w_ref)

        if isinstance(phase, xr.DataArray):
            self._phase = phase
        elif hasattr(phase, "to_xarray"):
            self._phase = phase.to_xarray()
        elif phase is None:
            self._phase = phase
        else:
            raise ValueError(
                "The phase variable must be an xr.DataArray or be None."
            )

        if ssa is None:
            self.ssa = None
        else:
            if isinstance(ssa, list):
                ssa = np.array(ssa)
            if np.isscalar(ssa) or (
                isinstance(ssa, np.ndarray) and (ssa.ndim <= 2)
            ):
                self.ssa = ssa
            elif hasattr(ssa, "to_xarray"):
                self.ssa = ssa.to_xarray()
            elif isinstance(ssa, xr.DataArray):
                self.ssa = ssa
            else:
                raise ValueError(
                    "The ssa variable must a scalar, a list, an ndarray of "
                    + "dim <= 2, or an xr.DataArray."
                )

        filename = Path(filename)
        if filename.parent == Path("."):  # no directory given
            filename = (
                DIR_AUXDATA / "aerosols" / "OPAC" / "mixtures" / filename.name
            )

        # Add extension if needed
        if "_sol" not in filename.name and not filename.suffix == ".nc":
            filename = filename.with_name(filename.name + "_sol.nc")
        elif filename.suffix != ".nc":
            filename = filename.with_name(filename.name + ".nc")

        if not filename.exists():
            raise FileNotFoundError(f"{filename} does not exist")

        self.filename = filename

        self.mixture = xr.open_dataset(self.filename)
        # check if hum dim size == 1 (to avoid interpolation/indexing
        # crash)
        if self.mixture.sizes["hum"] == 1:
            hum_v1 = float(self.mixture.coords["hum"].values[0])
            hum_v2 = hum_v1 + 1
            ds2 = self.mixture.assign_coords(hum=[hum_v2])
            self.mixture = xr.concat([self.mixture, ds2], dim="hum")

        if H_mix_min is None:
            H_mix_min = float(self.mixture.attrs["H_mix_min"])
        if H_mix_max is None:
            H_mix_max = float(self.mixture.attrs["H_mix_max"])
        if H_free_min is None:
            H_free_min = float(self.mixture.attrs["H_free_min"])
        if H_free_max is None:
            H_free_max = float(self.mixture.attrs["H_free_max"])
        if H_stra_min is None:
            H_stra_min = float(self.mixture.attrs["H_stra_min"])
        if H_stra_max is None:
            H_stra_max = float(self.mixture.attrs["H_stra_max"])

        if Z_mix is None:
            Z_mix = float(self.mixture.attrs["Z_mix"])
        if Z_free is None:
            Z_free = float(self.mixture.attrs["Z_free"])
        if Z_stra is None:
            if self.mixture.attrs["Z_stra"] == "99":
                Z_stra = 1e6  # -> OPAC Z=99 for constant vertical dist
            else:
                Z_stra = float(self.mixture.attrs["Z_stra"])

        self.hum_or_reff = "hum"
        self.free_tropo = None
        self.strato = None

        self.force_rh = [rh_mix, rh_free, rh_stra]
        self.vert_content = []
        self.H_min = []
        self.H_max = []
        self.Z_sh = []

        if H_mix_max - H_mix_min > 1e-6:
            self.vert_content.append(self.mixture)
            self.H_min.append(H_mix_min)
            self.H_max.append(H_mix_max)
            self.Z_sh.append(Z_mix)
        if H_free_max - H_free_min > 1e-6:
            filename_tmp = (
                DIR_AUXDATA
                / "aerosols"
                / "OPAC"
                / "free_troposphere"
                / "free_troposphere_sol.nc"
            )
            self.free_tropo = xr.open_dataset(filename_tmp)
            # check we have the same wl dim than previous aer pro in
            # vert_content
            if len(self.vert_content) > 0:
                aer_prev = self.vert_content[-1]
                w_cur = self.free_tropo.wav.values
                w_prev = aer_prev.wav.values
                nwcur = len(w_cur)
                nwprev = len(w_prev)
                if nwcur != nwprev or (
                    nwcur == nwprev and not np.array_equal(w_cur, w_prev)
                ):
                    wav_clip = w_prev.clip(
                        min=w_cur.min().item(), max=w_cur.max().item()
                    )
                    self.free_tropo = self.free_tropo.interp(wav=wav_clip)
            self.vert_content.append(self.free_tropo)
            self.H_min.append(H_free_min)
            self.H_max.append(H_free_max)
            self.Z_sh.append(Z_free)
        if H_stra_max - H_stra_min > 1e-6:
            filename_tmp = (
                DIR_AUXDATA
                / "aerosols"
                / "OPAC"
                / "stratosphere"
                / "stratosphere_sol.nc"
            )
            self.strato = xr.open_dataset(filename_tmp)
            # check we have the same wl dim than previous aer pro in
            # vert_content
            if len(self.vert_content) > 0:
                aer_prev = self.vert_content[-1]
                w_cur = self.strato.wav.values
                w_prev = aer_prev.wav.values
                nwcur = len(w_cur)
                nwprev = len(w_prev)
                if nwcur != nwprev or (
                    nwcur == nwprev and not np.array_equal(w_cur, w_prev)
                ):
                    wav_clip = w_prev.clip(
                        min=w_cur.min().item(), max=w_cur.max().item()
                    )
                    self.strato = self.strato.interp(wav=wav_clip)
            self.vert_content.append(self.strato)
            self.H_min.append(H_stra_min)
            self.H_max.append(H_stra_max)
            self.Z_sh.append(Z_stra)

    def dtau_ssa(
        self,
        wav: np.ndarray,
        Z: np.ndarray,
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
        wav : array-like
            Wavelengths (in nm) at which to calculate optical properties
        Z : array-like
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
            Optical depth with shape (len(wav), len(Z))
        ssa : ndarray
            Single scattering albedo with shape (len(wav), len(Z))
        """
        dtau = np.zeros((len(wav), len(Z)), dtype=np.float32)
        dtau_ref = np.zeros((1, len(Z)), dtype=np.float32)
        ssa = np.zeros_like(dtau)

        if self.hum_or_reff == "hum":
            hum_or_reff_val = rh
        elif isinstance(self, Cloud):
            hum_or_reff_val = self.reff
        else:
            raise NameError(
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
            cont_wav_vals = cont.coords["wav"].values.astype(np.float64)
            ext_data = cont["ext"].values.astype(np.float64)
            ssa_data = cont["ssa"].values.astype(np.float64)
            if (hor == "hum") and (self.force_rh[icont] is not None):
                rh_reff = np.full_like(hum_or_reff_val, self.force_rh[icont])
            else:
                rh_reff = hum_or_reff_val
            # Axes values
            hor_vals = cont_hor_vals
            wav_vals = cont_wav_vals
            # Float indices with extrema fill for humidity/reff, strict
            # bounds for wavelength
            nhor = len(hor_vals)
            nwav_orig = len(wav_vals)
            idf_hor = np.interp(
                np.asarray(rh_reff, dtype=np.float64),
                hor_vals,
                np.arange(nhor),
                left=0,
                right=nhor - 1,
            )
            idf_wav = np.interp(
                np.asarray(wav, dtype=np.float64),
                wav_vals,
                np.arange(nwav_orig),
            )
            idf_wav_ref = np.interp(
                np.atleast_1d(np.asarray(self.w_ref, dtype=np.float64)),
                wav_vals,
                np.arange(nwav_orig),
            )
            if len(rh_reff) == 1:
                # Interpolate along hor (dim 0) -> (1, wav_orig)
                ext_at_hor = vec_float_indexing(
                    ext_data, [idf_hor, slice(None)]
                )  # (1, wav_orig)
                ssa_at_hor = vec_float_indexing(
                    ssa_data, [idf_hor, slice(None)]
                )  # (1, wav_orig)
                # Transpose to (wav_orig, 1), interpolate along wav (dim
                # 0) -> (nwav, 1)
                ext_tmp = vec_float_indexing(
                    ext_at_hor.T, [idf_wav, slice(None)]
                )  # (nwav, 1)
                ext_ref_tmp = vec_float_indexing(
                    ext_at_hor.T, [idf_wav_ref, slice(None)]
                )  # (nwav_ref, 1)
                ssa_tmp = vec_float_indexing(
                    ssa_at_hor.T, [idf_wav, slice(None)]
                )  # (nwav, 1)
                for iz in range(0, len(Z)):
                    ext_[:, iz] = ext_tmp[:, 0]
                    ext_ref_[:, iz] = ext_ref_tmp[:, 0]
                    ssa_[:, iz] = ssa_tmp[:, 0]
            else:
                # Interpolate along hor (dim 0) -> (nhor_query,
                # wav_orig)
                ext_at_hor = vec_float_indexing(
                    ext_data, [idf_hor, slice(None)]
                )  # (nhor, wav_orig)
                ssa_at_hor = vec_float_indexing(
                    ssa_data, [idf_hor, slice(None)]
                )  # (nhor, wav_orig)
                # Transpose to (wav_orig, nhor), interpolate along wav
                # (dim 0) -> (nwav, nhor)
                ext_ = vec_float_indexing(
                    ext_at_hor.T, [idf_wav, slice(None)]
                )  # (nwav, nhor)
                ext_ref_ = vec_float_indexing(
                    ext_at_hor.T, [idf_wav_ref, slice(None)]
                )  # (nwav_ref, nhor)
                ssa_ = vec_float_indexing(
                    ssa_at_hor.T, [idf_wav, slice(None)]
                )  # (nwav, nhor)
            dtau_ = np.zeros_like(dtau)
            dtau_ref_ = np.zeros_like(dtau_ref)
            h1 = np.maximum(self.H_min[icont], Z[1:])
            h2 = np.minimum(self.H_max[icont], Z[:-1])
            cond = h2 > h1
            dtau_[:, 1:][:, cond] = ext_[:, 1:][
                :, cond
            ] * get_aer_dist_integral(self.Z_sh[icont], h1[cond], h2[cond])
            dtau += dtau_
            ssa += dtau_ * ssa_
            dtau_ref_[:, 1:][:, cond] = ext_ref_[:, 1:][
                :, cond
            ] * get_aer_dist_integral(self.Z_sh[icont], h1[cond], h2[cond])
            dtau_ref += dtau_ref_

        ssa[dtau != 0] /= dtau[dtau != 0]

        # apply scaling factor to get the required optical thickness at
        # the
        # specified wavelength or force tau for all wavelengths
        if self.tau_ref is not None:
            if (
                isinstance(self.tau_ref, np.ndarray) and self.tau_ref.ndim == 0
            ) or np.isscalar(self.tau_ref):
                dtau *= self.tau_ref / np.sum(dtau_ref)
            else:
                # xr.DataArray
                wav_axis = self.tau_ref.coords[
                    self.tau_ref.dims[0]
                ].values.astype(np.float64)
                tau_ref_interp = np.interp(
                    np.asarray(wav, dtype=np.float64),
                    wav_axis,
                    self.tau_ref.values,
                )
                dtau *= (tau_ref_interp / np.sum(dtau, axis=1))[:, None]

        # force ssa
        if self.ssa is not None:
            if np.isscalar(self.ssa):  # scalar
                ssa[:, :] = float(self.ssa)
            # ndarray with dim <= 2
            elif isinstance(self.ssa, np.ndarray):
                if self.ssa.ndim == 0:
                    ssa[:, :] = self.ssa
                elif self.ssa.ndim == 1:
                    ssa[:, :] = self.ssa[
                        :, None
                    ]  # If 1d array -> consider only wl variability
                elif self.ssa.ndim == 2:
                    ssa[:, :] = self.ssa[:, :]
            else:  # xr.DataArray
                wav_axis = self.ssa.coords[self.ssa.dims[0]].values.astype(
                    np.float64
                )
                ssa_interp = np.interp(
                    np.asarray(wav, dtype=np.float64),
                    wav_axis,
                    self.ssa.values,
                )
                ssa[:, :] = ssa_interp[:, None]
        return dtau, ssa

    def phase(
        self,
        wav: np.ndarray,
        Z: np.ndarray,
        rh: np.ndarray,
        NBTHETA: int = 721,
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
        wav : array-like
            Wavelengths (in nm) at which to calculate phase matrix
        Z : array-like
            Altitude profile (in km) for which to calculate phase matrix
        rh : array-like
            Relative humidity (%). Must have size similar to Z (altitude
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
        NBTHETA : int, optional
            Number of scattering angles for angle resampling. Default is
            721.

        Returns
        -------
        phase_matrix : xr.DataArray
            DataArray containing the phase matrix with dimensions
            [wav_phase, z_phase, stk, theta_atm].
            Shape is (len(wav), len(Z)-1, nphamat, NBTHETA) where:
            - nphamat = 4 for spherical particles only (phase matrix
              unique terms P11, P21, P33, P34)
            - nphamat = 6 for spherical and non-spherical particles
              (additional phase matrix unique terms P22, P44)
            - theta_atm: scattering angles from 0° to 180°
        """

        if self._phase is not None:
            if self._phase.ndim == 2:
                # convert to 4-dim by inserting empty dimensions
                # wav_phase and z_phase
                dims = list(self._phase.dims)
                assert dims == ["stk", "theta_atm"]
                pha_ = self._phase.values[:, :]
                if pha_.shape[0] == 4:
                    pha_6 = np.zeros((6, pha_.shape[1]), dtype=pha_.dtype)
                    pha_6[0:4, :] = pha_
                    pha_6[4, :] = pha_[0, :].copy()  # F22 = F11
                    pha_6[5, :] = pha_[2, :].copy()  # F44 = F33
                    pha_ = pha_6
                return xr.DataArray(
                    pha_[None, None, :, :],
                    dims=["wav_phase", "z_phase", "stk", "theta_atm"],
                    coords={
                        "wav_phase": [wav[0]],
                        "z_phase": [0.0],
                        "stk": np.arange(6),
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
                            "stk": np.arange(6),
                            dims[3]: self._phase.coords[dims[3]].values,
                        },
                    )
                return xr.DataArray(
                    pha_,
                    dims=dims,
                    coords={d: self._phase.coords[d].values for d in dims},
                )

        theta = np.linspace(0.0, 180.0, num=NBTHETA)
        lam_tabulated = self.mixture.coords["wav"].values
        nwav = len(wav)

        P_tot = 0.0
        dssa = 0.0
        for icont, cont in enumerate(self.vert_content):
            hor = self.hum_or_reff

            phase_data = cont["phase"].values
            hor_vals = cont.coords[hor].values.astype(np.float64)
            wav_vals = cont.coords["wav"].values.astype(np.float64)
            theta_orig = cont.coords["theta"].values.astype(np.float64)
            ext_data = cont["ext"].values.astype(np.float64)
            ssa_data = cont["ssa"].values.astype(np.float64)

            nphamat = phase_data.shape[2]
            nhor = len(hor_vals)
            nwav_orig = len(wav_vals)

            # Wavelength optimization: subset to bracketing wavelengths
            if (np.max(wav) > np.max(lam_tabulated)) or (
                np.min(wav) < np.min(lam_tabulated)
            ):
                # Out of range: use full axis
                wav_subset = wav_vals
                phase_subset = phase_data
            else:
                range_ind = np.array(
                    [
                        np.argwhere((lam_tabulated <= np.min(wav)))[-1][0],
                        np.argwhere((lam_tabulated >= np.max(wav)))[0][0],
                    ]
                )
                ilam_tabulated = np.arange(len(lam_tabulated), dtype=int)
                ilam_opti = np.concatenate(
                    np.argwhere(
                        (ilam_tabulated >= range_ind[0])
                        & (ilam_tabulated <= range_ind[1])
                    )
                )
                wav_subset = wav_vals[ilam_opti]
                phase_subset = phase_data[:, ilam_opti, :, :]

            nwav_sub = len(wav_subset)

            # Interpolate along wav: transpose to (wav, hor, stk, theta)
            # for vec_float_indexing
            if nwav_sub > 1:
                idf_wav = np.interp(
                    np.asarray(wav, dtype=np.float64),
                    wav_subset,
                    np.arange(nwav_sub),
                )
                phase_at_wav = vec_float_indexing(
                    np.ascontiguousarray(phase_subset.transpose(1, 0, 2, 3)),
                    [idf_wav, slice(None), slice(None), slice(None)],
                )
            else:
                phase_at_wav = np.broadcast_to(
                    phase_subset.transpose(1, 0, 2, 3),
                    (nwav, nhor, nphamat, len(theta_orig)),
                ).copy()
            # Result: (nwav, hor, stk, theta_orig)

            # Theta resampling if needed: transpose to (theta, nwav,
            # hor, stk)
            if NBTHETA != len(theta_orig):
                idf_theta = np.interp(
                    theta, theta_orig, np.arange(len(theta_orig))
                )
                phase_at_wav = vec_float_indexing(
                    np.ascontiguousarray(phase_at_wav.transpose(3, 0, 1, 2)),
                    [idf_theta, slice(None), slice(None), slice(None)],
                )
                # Result: (NBTHETA, nwav, hor, stk) -> transpose to
                # (nwav, hor, stk, NBTHETA)
                phase_at_wav = phase_at_wav.transpose(1, 2, 3, 0)
            # phase_at_wav: (nwav, hor, stk, NBTHETA)

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
                raise NameError(
                    "Phase matrix must varies as function of hum or reff."
                )

            if np.isscalar(hum_or_reff_val) or (
                isinstance(hum_or_reff_val, np.ndarray)
                and hum_or_reff_val.ndim == 0
            ):
                hum_or_reff_val = np.array([hum_or_reff_val])
            else:
                hum_or_reff_val = np.array(hum_or_reff_val)

            # Interpolate along hor: transpose to (hor, nwav, stk,
            # NBTHETA)
            if len(hum_or_reff_val) == 1:
                hor_query = hum_or_reff_val
                nz_phase = len(Z) - 1
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
            P_data = vec_float_indexing(
                np.ascontiguousarray(phase_at_wav.transpose(1, 0, 2, 3)),
                [idf_hor, slice(None), slice(None), slice(None)],
            )
            # Result: (nz, nwav, stk, NBTHETA) -> transpose to (nwav,
            # nz, stk, NBTHETA)
            P_data = np.ascontiguousarray(P_data.transpose(1, 0, 2, 3)).astype(
                np.float32
            )
            if len(hum_or_reff_val) == 1:
                P_data = np.broadcast_to(
                    P_data,
                    (nwav, nz_phase, P_data.shape[2], P_data.shape[3]),
                ).copy()

            # Expand 4 stk to 6 if needed
            if nphamat == 4:
                P_data_6 = np.zeros(
                    (nwav, nz_phase, nphamat_, NBTHETA), dtype="float32"
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
                    (nwav, nz_phase, nphamat_, NBTHETA), dtype="float32"
                )
                P_data_6[:, :, 0:nphamat, :] = P_data
                P_data = P_data_6

            P = xr.DataArray(
                P_data,
                dims=["wav_phase", "z_phase", "stk", "theta_atm"],
                coords={
                    "wav_phase": wav,
                    "z_phase": np.arange(P_data.shape[1]),
                    "stk": np.arange(P_data.shape[2]),
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
            idf_wav_ext = np.interp(
                np.asarray(wav, dtype=np.float64),
                wav_vals,
                np.arange(nwav_orig),
            )
            ext_at_hor = vec_float_indexing(
                ext_data, [idf_hor_ext, slice(None)]
            )
            ssa_at_hor = vec_float_indexing(
                ssa_data, [idf_hor_ext, slice(None)]
            )
            ext_ = vec_float_indexing(
                ext_at_hor.T, [idf_wav_ext, slice(None)]
            )  # (nwav, nhor_q)
            ssa_ = vec_float_indexing(
                ssa_at_hor.T, [idf_wav_ext, slice(None)]
            )  # (nwav, nhor_q)
            if len(hum_or_reff_val) == 1:
                ext_ = np.broadcast_to(ext_, (nwav, len(Z))).copy()
                ssa_ = np.broadcast_to(ssa_, (nwav, len(Z))).copy()

            dtau_ = np.zeros((len(wav), len(Z)), dtype=np.float32)
            h1 = np.maximum(self.H_min[icont], Z[1:])
            h2 = np.minimum(self.H_max[icont], Z[:-1])
            cond = h2 > h1
            dtau_[:, 1:][:, cond] = ext_[:, 1:][
                :, cond
            ] * get_aer_dist_integral(self.Z_sh[icont], h1[cond], h2[cond])
            dssa_ = dtau_ * ssa_  # NLAM, ALTITUDE
            dssa_ = dssa_[:, 1:, None, None]
            dssa += dssa_
            P_tot += P * dssa_

        with np.errstate(divide="ignore", invalid="ignore"):
            P_tot.data /= dssa
        P_tot.data[np.isnan(P_tot.data)] = 0.0
        P_tot = P_tot.assign_coords(z_phase=Z[1:])
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
    filename : str,
        Complete path to the cloud file or filename for clouds located
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
    ssa : None | float | list | 1-D ndarray | 2-D ndarray |
    xr.DataArray, optional
        Force particle single scattering albedo.

        - if float -> same value for all wavelengths and altitudes
        - if list -> it will be converted into a 1-D ndarray.
        - if 1-D ndarray -> only wavelength dependence is considered
        - if 2-D ndarray -> wavelength and altitude dependence is
          considered
        - if xr.DataArray -> wavelength and altitude dependence is
          considered

        Note that xr.DataArray is more flexible since it allows
        interpolation if wavelengths
        in calc method are different (but not the case for the altitude
        axis).
    phase : None | xr.DataArray, optional
        Phase matrix F as function of wavelength, altitude, stoke
        components and scattering angle
        The variable names must be:
        If 4-D matrix -> wav_phase, z_phase, stk, theta
        If 2-D matrix (assumed monochromatic and contant vertically) ->
        stk, theta
        Where:
        - wav_phase is the wavelength. It must be equal to the `pfwav`
          parameter of AtmAFGL
          if defined, else `wav` parameter wavelengths of the AtmAFGL
          calc method.
        - z_phase is the phase altitude. It must be equal to the
          `pfgrid[1:]` parameter
          of AtmAFGL
        - stk the phase matrix unique terms.
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
    >>> print(cld_wc.mixture)
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
        filename: str | Path,
        reff: float,
        zmin: float,
        zmax: float,
        tau_ref: float,
        w_ref: float,
        ssa: float | list[float] | np.ndarray | xr.DataArray | None = None,
        phase: xr.DataArray | None = None,
    ) -> None:
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
            elif hasattr(ssa, "to_xarray"):
                self.ssa = ssa.to_xarray()
            elif isinstance(ssa, xr.DataArray):
                self.ssa = ssa
            else:
                raise ValueError(
                    "The ssa variable must a scalar, a list, an ndarray"
                    + " of dim <= 2, or an xr.DataArray."
                )

        filename = Path(filename)
        if filename.parent == Path("."):  # no directory given
            base_dir = Path(DIR_AUXDATA) / "clouds"
            filename = base_dir / filename.name

        if "_sol" not in filename.name and not filename.suffix == ".nc":
            filename = filename.with_name(filename.name + "_sol.nc")
        elif filename.suffix != ".nc":
            filename = filename.with_name(filename.name + ".nc")

        if not filename.exists():
            raise FileNotFoundError(f"{filename} does not exist")

        self.filename = filename

        self.mixture = xr.open_dataset(self.filename)
        # check if reff dim size == 1 (to avoid interpolation/indexing
        # crash)
        if self.mixture.sizes["reff"] == 1:
            reff_v1 = float(self.mixture.coords["reff"].values[0])
            reff_v2 = reff_v1 + 1
            ds2 = self.mixture.assign_coords(reff=[reff_v2])
            self.mixture = xr.concat([self.mixture, ds2], dim="reff")

        self.hum_or_reff = "reff"
        self.free_tropo = None
        self.strato = None

        self.vert_content = []
        self.H_min = []
        self.H_max = []
        self.Z_sh = []

        if zmax - zmin > 1e-6:
            self.vert_content.append(self.mixture)
            self.H_min.append(zmin)
            self.H_max.append(zmax)
            self.Z_sh.append(1e6)  # constant dist

        if isinstance(phase, xr.DataArray):
            self._phase = phase
        elif hasattr(phase, "to_xarray"):
            self._phase = phase.to_xarray()
        elif phase is None:
            self._phase = phase
        else:
            raise ValueError(
                "The phase variable must be an xr.DataArray or be None."
            )

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
        aerosol optical depth values with shape (len(hum), len(wav))
    ssa : 2-D ndarray
        Single scattering albedo values with shape (len(hum), len(wav))
    phase : 4-D ndarray
        Phase function values with shape (len(hum), len(wav), len(stk),
        len(theta)).

        Where len(stk) is the number of unique phase terms.

        The phase matrix terms must be given in the folowing order:
        - F11, F21, F33 and F34 if only 4 terms are given (only for
          spherical particles)
        - F11, F21, F33, F34, F22 and F44 if 6 terms are given (for both
          spherical and non-spherical particles)
    hum : 1-D ndarray
        Relative humidity values in percentage
    wav : 1-D ndarray
        Wavelength values in nanometers
    theta : 1-D ndarray
        Scattering angle values in degrees
    H_mix_min : float, optional
        Force min altitude of the mixture
    H_mix_max : float, optional
        Force max altitude of the mixture
    Z_mix : float, optional
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
        wav: np.ndarray,
        theta: np.ndarray,
        H_mix_min: float = 0.0,
        H_mix_max: float = 2.0,
        Z_mix: float = 2,
    ) -> None:

        self.filename = "none"
        self.tau_ref = None
        ext = aod / (
            Z_mix * (np.exp(-H_mix_min / Z_mix) - np.exp(-H_mix_max / Z_mix))
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
                "wav": wav,
                "theta": theta,
                "stk": np.arange(phase.shape[2]),
            },
        )

        ds.attrs["name"] = "none"
        ds.attrs["H_mix_min"] = str(H_mix_min)
        ds.attrs["H_mix_max"] = str(H_mix_max)
        ds.attrs["Z_mix"] = str(Z_mix)

        self.mixture = ds
        # check if hum dim size == 1 (to avoid interpolation/indexing
        # crash)
        if self.mixture.sizes["hum"] == 1:
            hum_v1 = float(self.mixture.coords["hum"].values[0])
            hum_v2 = hum_v1 + 1
            ds2 = self.mixture.assign_coords(hum=[hum_v2])
            self.mixture = xr.concat([self.mixture, ds2], dim="hum")

        self.w_ref = np.array([float(self.mixture.coords["wav"].values[0])])
        self.ssa = None

        self.hum_or_reff = "hum"
        self.free_tropo = None
        self.strato = None

        self.force_rh = [None]
        self.vert_content = []
        self.H_min = []
        self.H_max = []
        self.Z_sh = []

        if H_mix_max - H_mix_min > 1e-6:
            self.vert_content.append(self.mixture)
            self.H_min.append(H_mix_min)
            self.H_max.append(H_mix_max)
            self.Z_sh.append(Z_mix)

        self._phase = None

    @staticmethod
    def list() -> list[str]:
        """"""
        raise NotImplementedError(
            "The list() method is not available for user-defined aerosols. "
            "User-defined aerosols are custom configurations and do not have "
            "a predefined list of available files."
        )


class Atmosphere(object):
    """Base class for atmosphere."""

    pass


class AtmAFGL(Atmosphere):
    """
    Atmospheric profile definition using AFGL data

    Parameters
    ----------

    atm_filename : str
        The AFGL atmosphere profile to use. Choice are:
            - 'afglms' for Mid-Latitude Summer (45N July)
            - 'afglmw' for Mid-Latitude Winter (45N Jan)
            - 'afglss' for Sub Arctic Summer (60N July)
            - 'afglsw' for Sub Arctic Winter (60N Jan)
            - 'afglt' for Tropic (15N Annual Average)
            - 'afglus' for U.S. Standard (1976)

        File format: If a full path is not provided (only filename), the
        atmospheric
        auxdata directory is automatically prepended to the path. The
        file extension
        defaults to '.nc' if not specified. Only '.nc' (NetCDF) and
        '.dat' file
        formats are accepted. For '.dat' files, the libratran atmosphere
        file
        convention is used.
    comp:  list, optional
        Components particles (aerosols or clouds) to consider, i.e. a
        list of aerOPAC or/and Cloud objects.
    grid : None | 1-D array-like, optional
      The vertical grid (from TOA to BOA). The optical properties of the
      atmosphere are recalculated following
      the new grid. If None, the AFGL grid is kept.
    lat : float, optional
        The latitude used for Rayleigh optical depth calculation.
        Default=45.
    P0:  None | float, optional
        The sea surface pressure. If None take P0 from the AFGL profil.
    O3 : None | float, optional
        The total ozone column in Dobson units. If None keep the total
        ozone content of the chosen atmospheric
        profile.
    H2O : None | float, optional
        The total water vapor column in g.cm-2. If None keep the total
        water vapor content of the chosen
        atmospheric profile.
    NO2: bool, optional
        Activate NO2 absorption (default True)
    O3_H2O_alt : None | float, optional
        Altitude (km) at which the specified O3 and H2O values apply.
        When specified,
        the O3 and H2O profiles are scaled such that the column amount
        from TOA to this
        altitude matches the provided O3 and H2O values. The full
        gaseous distribution
        from TOA to ground is preserved; only the scaling factor is
        adjusted to match
        the constraint at this reference altitude.
        Default: None
    tauR : None | float, optional
        Force the Rayleigh optical thickness. If None, computed from
        atmospheric profile and wavelength.
    pfwav : None | list, optional
        The list of wavelengths over which the phase matrices are
        calculated. Then use the nearest wavelength
        during cuda simulation. Useful to reduce the memory. If None,
        compute the phase matrix at all wavelengths.
    pfgrid : list, optional
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
    prof_abs : None | 2-D ndarray, optional
        - In 1D atm mode -> force the gaseous absorption optical
          thickness vertical profile (NWavelength,NZ),
        it shortcuts any further gaseous absorption computation.
        - In 3D atm mode -> just an optical properties index, it must be
          completed by the cells grid
    prof_ray : None | 2-D ndarray, optional
        - In 1D atm mode -> force the Rayleigh scattering optical
          thickness vertical profile (NWavelength,NZ),
        it shortcuts any further Rayleigh scattering computation.
        - In 3D atm mode -> just an optical properties index, it must be
          completed by the cells grid
    prof_aer : None | tuple, optional
        - In 1D atm mode - > A tuple (ext,ssa) with the aerosol
          extinction optical thickness profile (ext) and
        single scattering albedo arrays (ssa), it shortcuts any further
        particles scattering computation.
        - In 3D atm mode -> just an optical properties index, it must be
          completed by the cells grid
    prof_phases : None | tuple, optional
        A tuple (iphase, phases ) where iphase is the phase matrix
        indices profile (NWavelength,NZ),
        and  phases is a list of phase matrices LUT (as outputs of the
        `read_phase` utility).
    RH_cst : None | float, optional
        Force relative humidity to be constant. If None calculated
        depending on H2O vertical profile.
    O3_acs : str, optional
        Path to ozone netcdf4 file with absorption coefficient cross
        section (SIGMA = 1E-20 * [C0 + C1*T + C2*T^2],
        in cm^2, and where T is in degrees Celcius). If only filename is
        given automatically look at "auxdata/acs/".
        By default use Bogumil Version 3.0 data. Available files in
        auxdata:
            - 'O3_acs_BogumilV3.0_coeffs.nc'
            - 'O3_acs_Chehade(Bogumil_revised)V4.1_coeffs.nc'
            - 'O3_acs_SerdyuchenkoV2.0_coeffs.nc'
    NO2_acs : str, optional
        Path to NO2 netcdf4 file with absorption coefficient cross
        section (SIGMA = 1E-20 * [C0 + C1*T + C2*T^2],
        in cm^2, and where T is in degrees Celcius). If only filename is
        given automatically look at "auxdata/acs/".
        By default use Bogumil Version 1.0 data. Available files in
        auxdata:
            - 'NO2_acs_BogumilV1.0_coeffs.nc'
            - 'NO2_acs_Bingen_coeffs.nc'
    cells : None | tuple, optional
        If cells is given, then we are in 3D mode. Definitions:
           - 'iopt' gives the number of the optical property
             corresponding to the cells. iopt(Ncell)
           - 'iabs' gives the number of the absorption property
             corresponding to the cells. iabs(Ncell)
           - Bounding Boxes(1 Point Bottom Left pmin, 1 Point Top Right
             pmax) of the cells. pmin(3,Ncell). pmax(3,Ncell)
           and 6 neighbours index (positive X, negative X, positive Y,
           negative Y, positive Z, negative Z). neighbour(6,Ncell)
           it returns coefficients in (km-1) instead of optical
           thicknesses
    """

    def __init__(
        self,
        atm_filename,
        comp=[],
        grid=None,
        lat=45.0,
        P0=None,
        O3=None,
        H2O=None,
        NO2=True,
        O3_H2O_alt=None,
        tauR=None,
        pfwav=None,
        pfgrid=[100.0, 0.0],
        prof_abs=None,
        prof_ray=None,
        prof_aer=None,
        prof_phases=None,
        RH_cst=None,
        US=True,
        cells=None,
        O3_acs="O3_acs_BogumilV3.0_coeffs",
        NO2_acs="NO2_acs_BogumilV1.0_coeffs",
    ):

        self.lat = lat
        self.comp = comp
        self.pfwav = pfwav
        self.pfgrid = np.array(pfgrid)
        self.prof_abs = prof_abs
        self.prof_ray = prof_ray
        self.prof_aer = prof_aer
        self.prof_phases = prof_phases
        # store attribute using lowercase name for consistency
        self.rh_cst = RH_cst
        self.OPT3D = cells is not None
        if self.OPT3D:
            self.cells = cells

        self.tauR = tauR
        if tauR is not None:
            self.tauR = np.array(tauR)

        assert (np.diff(pfgrid) < 0.0).all()

        atm_filename = Path(atm_filename)

        #
        # init directories and read atm file
        #
        if atm_filename.name == "ATM3D":
            Nopt = grid.size
            prof = ProfileBase(None)
            prof.z = np.arange(Nopt, dtype=np.float32)[::-1]
            attr_names = [
                "P",
                "T",
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
            prof.rh_cst = RH_cst
        else:
            if atm_filename.parent == Path("."):
                atm_filename = DIR_AUXDATA / "atmospheres" / atm_filename.name
            # By default if no suffix is given consider it as a netcdf
            # file
            if not atm_filename.exists() and atm_filename.suffix == "":
                atm_filename = atm_filename.with_name(
                    atm_filename.name + ".nc"
                )

            if atm_filename.suffix == ".nc" or atm_filename.suffix == ".dat":
                prof = ProfileBase(
                    atm_filename,
                    tco3=O3,
                    tcwp=H2O,
                    tcno2=NO2,
                    p0=P0,
                    rh_cst=RH_cst,
                    o3_h2o_alt=O3_H2O_alt,
                )
            else:
                raise NameError(
                    "This file format is not supported. Only '.nc' and"
                    + " '.dat' are supported."
                )

        #
        # read gaseous acs
        #
        O3_acs_path = Path(O3_acs)
        if O3_acs_path.parent == Path("."):
            O3_acs_path = DIR_AUXDATA / "acs" / O3_acs_path.name
        if not O3_acs_path.exists() and O3_acs_path.suffix != ".nc":
            O3_acs_path = O3_acs_path.with_name(O3_acs_path.name + ".nc")
        self.acs_o3 = xr.open_dataset(O3_acs_path)
        self.acs_o3 = self.acs_o3.rename({"wav": "wavelength"})

        NO2_acs_path = Path(NO2_acs)
        if NO2_acs_path.parent == Path("."):
            NO2_acs_path = DIR_AUXDATA / "acs" / NO2_acs_path.name
        if not NO2_acs_path.exists() and NO2_acs_path.suffix != ".nc":
            NO2_acs_path = NO2_acs_path.with_name(NO2_acs_path.name + ".nc")
        self.acs_no2 = xr.open_dataset(NO2_acs_path)
        self.acs_no2 = self.acs_no2.rename({"wav": "wavelength"})

        #
        # regrid profile if required
        #
        if grid is None:
            self.prof = prof
        else:
            if isinstance(grid, str):
                grid = strgrid_to_numpy(grid)
            self.prof = prof.regrid(np.array(grid))

        #
        # calculate reduced profile
        # (for phase function blending)
        #
        self.prof_red = prof.regrid(pfgrid)

    def calc(
        self,
        wav,
        phase=True,
        NBTHETA=721,
        use_old_calc_iphase=False,
        truncation=None,
    ):
        """
        Profile and phase matrix calculation at bands / wav

        Parameters
        ----------
        wav : float | 1-D ndarray | BandSet | list
            Wavelengths at which to calculate the profile. It can be a
            list of REPTRAN_IBAND or KDIS_IBAND.
        NBTHETA : int, optional
            The number of angles to be considered for the phase matrix.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (depracated).
        truncation : None | DM_trunc | GT_trunc, optional
            The scattering phase truncation to use.

        Returns
        -------
        out : xr.Dataset
            An xarray.Dataset object with the profile and (if phase =
            True) the phase matrices.
        """

        if not isinstance(wav, BandSet):
            wav = BandSet(wav)

        profile = self.profile(wav)

        if phase:
            if self.pfwav is None:
                wav_pha = wav[:]
            else:
                wav_pha = self.pfwav
            pha = self.phase(wav_pha, NBTHETA=NBTHETA)

            pro_var = list(profile.data_vars)
            if pha is not None or (
                self.OPT3D and ("phase_atm" in pro_var) and truncation
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
                if truncation is not None:
                    if self.OPT3D:
                        theta = profile.coords["theta_atm"].values
                    else:
                        theta = (
                            pha.coords["theta_atm"].values
                            if hasattr(pha, "coords")
                            else pha.axes[-1]
                        )
                    pha_tr = np.zeros(pha_.shape, dtype=np.float64)
                    nphac = pha_.shape[1]
                    if truncation.tr_method == "DM":
                        m_max = truncation.m_max
                    elif truncation.tr_method == "GT":
                        f_ = truncation.trunc_frac
                        th_tol = truncation.theta_tol
                        l_opti = truncation.lobatto_optimization
                        th_f = truncation.theta_tr
                    else:
                        raise ValueError("truncation method not recognized")
                    method = truncation.integral_method
                    f_pha = np.zeros(nphase, dtype=np.float64)
                    for iph in range(nphase):
                        if truncation.tr_method == "DM":
                            ds_pha = delta_m_phase_approx(
                                pha_[iph, 0, :], theta, m_max, method=method
                            )

                        elif truncation.tr_method == "GT":
                            ds_pha = gt_phase_approx(
                                pha_[iph, 0, :],
                                theta,
                                f_,
                                method=method,
                                th_tol=th_tol,
                                th_f=th_f,
                                lobatto_optimization=l_opti,
                            )
                        f11_tr = ds_pha["phase_tr"].values
                        f = ds_pha["f"].values
                        f_pha[iph] = f
                        # Ensure for the moment only 1 unique truncation
                        # factor
                        if iph > 0 and not np.isclose(
                            f_pha[iph], f_pha[0], atol=1e-6
                        ):
                            raise ValueError(
                                "Several truncation factors f is not yet "
                                + "authorized"
                            )

                        pha_tr[iph, 0, :] = f11_tr
                        beta = pha_tr[iph, 0, :] / pha_[iph, 0, :]
                        for icomp in range(1, nphac):
                            pha_tr[iph, icomp, :] = pha_[iph, icomp, :] * beta
                        if truncation.pha_scale_method == 2:
                            beta2 = 1.0 / (1 - f)
                            pha_tr[iph, 1, :] = pha_[iph, 1, :] * beta2
                            pha_tr[iph, 3, :] = pha_[iph, 3, :] * beta2

                if not self.OPT3D:
                    theta_atm = (
                        pha.coords["theta_atm"].values
                        if hasattr(pha, "coords")
                        else pha.axes[-1]
                    )
                    profile = profile.assign_coords(theta_atm=theta_atm)
                    profile["phase_atm"] = xr.DataArray(
                        pha_,
                        dims=["iphase", "stk", "theta_atm"],
                        coords={
                            "iphase": np.arange(pha_.shape[0]),
                            "stk": np.arange(pha_.shape[1]),
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
                        else np.linspace(0.0, 180.0, pha_.shape[-1])
                    )
                    profile["phase_atm"] = xr.DataArray(
                        pha_,
                        dims=["iphase", "stk", "theta_atm"],
                        coords={
                            "iphase": np.arange(pha_.shape[0]),
                            "stk": np.arange(pha_.shape[1]),
                            "theta_atm": theta_atm,
                        },
                        attrs=attrs_tmp,
                    )

                if truncation is not None:
                    # profile.add_dataset('phase_atm_tr', pha_tr,
                    # axnames=['iphase', 'stk', 'theta_atm'])
                    attrs_tmp = profile["phase_atm"].attrs
                    if "phase_atm" in profile.data_vars:
                        profile = profile.drop_vars("phase_atm")
                    theta_atm = np.linspace(0.0, 180.0, pha_tr.shape[-1])
                    profile["phase_atm"] = xr.DataArray(
                        pha_tr,
                        dims=["iphase", "stk", "theta_atm"],
                        coords={
                            "iphase": np.arange(pha_tr.shape[0]),
                            "stk": np.arange(pha_tr.shape[1]),
                            "theta_atm": theta_atm,
                        },
                        attrs=attrs_tmp,
                    )

                    # case tau instead of coeff (1D atm)
                    if not self.OPT3D:
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

    def profile(self, wav, prof=None):
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
        wav : array-like or BandSet
            Wavelengths at which to calculate optical properties [nm].
            If not a BandSet, it will be converted to one.
        prof : ProfileBase, optional
            Atmospheric profile containing altitude grids, temperature,
            pressure,
            and density profiles. Default is None; uses self.prof if not
            provided.

        Returns
        -------
        profile : xr.Dataset
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

        - **1D Mode (OPT3D=False)**: Returns cumulated optical
          thicknesses with axes
          [wavelength, z_atm]
        - **3D Mode (OPT3D=True)**: Returns extinction/absorption
          coefficients with axes
          [wavelength, iopt] for use in 3D radiative transfer
          calculations

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
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)

        if prof is None:
            prof = self.prof

        dz = -diff1(prof.z)

        pro = xr.Dataset(coords={"z_atm": prof.z, "wavelength": wav[:]})

        # refractive index
        n = refractivity(
            wav[:] * 1e-3, prof.P, prof.T, prof.dens_co2 / prof.dens_air * 1e6
        )
        pro["n_atm"] = xr.DataArray(
            n,
            dims=["wavelength", "z_atm"],
            coords={
                "wavelength": pro.coords["wavelength"],
                "z_atm": pro.coords["z_atm"],
            },
            attrs={"description": "atmospheric refractive index"},
        )

        pro["T_atm"] = xr.DataArray(
            prof.T,
            dims=["z_atm"],
            coords={"z_atm": pro.coords["z_atm"]},
            attrs={"description": "temperature (K)"},
        )

        #
        # Rayleigh optical thickness
        #
        # cumulated Rayleigh optical thickness (wav, z)
        if self.prof_ray is None:
            tauray = rayleigh_od(
                wav[:] * 1e-3,
                prof.dens_co2 / prof.dens_air * 1e6,
                self.lat,
                prof.z * 1e3,
                prof.P,
            )
            dtaur = diff1(tauray, axis=1)
        else:
            dtaur = self.prof_ray
            tauray = np.cumsum(dtaur, axis=1)

        if self.tauR is not None:
            # scale Rayleigh optical thickness
            if self.tauR.ndim == 1:
                # for each wavelength
                tauray *= self.tauR[:, None] / tauray[:, -1:]
            else:
                # scalar
                tauray *= self.tauR / tauray[:, -1:]

        assert tauray.ndim == 2

        # Rayleigh optical thickness
        dtaur = diff1(tauray, axis=1)
        if not self.OPT3D:
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
            dtaua = np.zeros((len(wav), len(prof.z)), dtype="float32")
            ssa_p = np.zeros((len(wav), len(prof.z)), dtype="float32")
            for comp in self.comp:
                dtau_, ssa_ = comp.dtau_ssa(
                    wav[:], prof.z, prof.relative_humidity()
                )
                dtaua += dtau_
                ssa_p += dtau_ * ssa_
            ssa_p[dtaua != 0] /= dtaua[dtaua != 0]
            ssa_p[dtaua == 0] = 1.0
            taua = np.cumsum(dtaua, axis=1)

        else:
            (dtaua, ssa_p) = self.prof_aer
            taua = np.cumsum(dtaua, axis=1)

        if not self.OPT3D:
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

        if not self.OPT3D:
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
            tau_o3 = np.zeros((len(wav), len(prof.z)), dtype="float32")
            tau_no2 = np.zeros((len(wav), len(prof.z)), dtype="float32")
            if wav.use_reptran_kdis:
                tau_mol = wav.calc_profile(self.prof) * dz
                # If not reptran (i.e. Kdis case) we set 03 and NO2 to 0
                # (already calculated in Kdis)
                if not (
                    str(wav.type_wav)
                    == "<class 'smartg.reptran.REPTRAN_IBAND'>"
                ):
                    all_kdis_gas = (
                        wav.data[0].band.kdis.species
                        + wav.data[0].band.kdis.species_c
                    )
                    if "no2" in all_kdis_gas:
                        use_no2_acs = False
                    if "o3" in all_kdis_gas:
                        use_o3_acs = False
            else:
                tau_mol = (
                    np.zeros((len(wav), len(prof.z)), dtype="float32") * dz
                )

            # Compute o3 and no2 (if kdis only compute them if not
            # already computed)
            if use_no2_acs or use_o3_acs:
                # Commun part
                T0 = 273.15  # in K
                T = prof.T[None, :]  # temperature variability in z
                if use_o3_acs:
                    # O3 optical thickness
                    min_wl = float(np.min(self.acs_o3["wavelength"].values))
                    max_wl = float(np.max(self.acs_o3["wavelength"].values))
                    wl_query = xr.DataArray(wav[:], dims=["wavelength"])
                    C0 = (
                        self.acs_o3["O3_C0"]
                        .sel(wavelength=wl_query, method="nearest")
                        .values[:, None]
                    )
                    C1 = (
                        self.acs_o3["O3_C1"]
                        .sel(wavelength=wl_query, method="nearest")
                        .values[:, None]
                    )
                    C2 = (
                        self.acs_o3["O3_C2"]
                        .sel(wavelength=wl_query, method="nearest")
                        .values[:, None]
                    )
                    tau_o3 = C0 + C1 * (T - T0) + C2 * (T - T0) * (T - T0)
                    tau_o3[
                        ~np.logical_and(wav[:] > min_wl, wav[:] < max_wl)
                    ] = 0.0
                    tau_o3 *= (
                        prof.dens_o3 * 1e-15
                    )  # ACS in 10^(-20) cm2, convert in km-1
                    tau_o3 *= dz
                    tau_o3[tau_o3 < 0] = 0
                if use_no2_acs:
                    # NO2 optical thickness
                    min_wl = float(np.min(self.acs_no2["wavelength"].values))
                    max_wl = float(np.max(self.acs_no2["wavelength"].values))
                    wl_query = xr.DataArray(wav[:], dims=["wavelength"])
                    C0 = (
                        self.acs_no2["NO2_C0"]
                        .sel(wavelength=wl_query, method="nearest")
                        .values[:, None]
                    )
                    C1 = (
                        self.acs_no2["NO2_C1"]
                        .sel(wavelength=wl_query, method="nearest")
                        .values[:, None]
                    )
                    C2 = (
                        self.acs_no2["NO2_C2"]
                        .sel(wavelength=wl_query, method="nearest")
                        .values[:, None]
                    )
                    tau_no2 = C0 + C1 * (T - T0) + C2 * (T - T0) * (T - T0)
                    tau_no2[
                        ~np.logical_and(wav[:] > min_wl, wav[:] < max_wl)
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

            if not self.OPT3D:
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
            if not self.OPT3D:
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
        if not self.OPT3D:
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
        if not self.OPT3D:
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
        FQY1 = np.zeros_like(ssa)
        if not self.OPT3D:
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
                FQY1,
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
                FQY1,
                dims=["wavelength", "iopt"],
                coords={"wavelength": pro.coords["wavelength"]},
                attrs={
                    "description": "fluoresence quantum yield of the "
                    + "layer"
                },
            )

        if self.prof_phases is not None:
            ipha, phases = self.prof_phases
            if not self.OPT3D:
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

            # set the number of scattering angles to the maximum
            # # convert legacy LUT to DataArray objects
            phases = [
                x.to_xarray() if hasattr(x, "to_xarray") else x for x in phases
            ]
            ip = np.array([p.sizes["theta_atm"] for p in phases]).argmax()
            theta = phases[ip].coords["theta_atm"].values
            # TODO: use gatiab vec_float_indexing function bellow
            pha = np.stack([p.interp(theta_atm=theta).values for p in phases])
            pro = pro.assign_coords(theta_atm=theta)
            pro["phase_atm"] = xr.DataArray(
                pha,
                dims=["iphase", "stk", "theta_atm"],
                coords={
                    "iphase": np.arange(pha.shape[0]),
                    "stk": np.arange(pha.shape[1]),
                    "theta_atm": pro.coords["theta_atm"],
                },
                attrs={"description": "phase matrices"},
            )
        # Pure 3D
        #
        if self.OPT3D:
            (iopt, iabs, pmin, pmax, neighbour) = self.cells
            pro["iopt_atm"] = xr.DataArray(iopt, dims=["icell"])
            pro["iabs_atm"] = xr.DataArray(iabs, dims=["icell"])
            pro["pmin_atm"] = xr.DataArray(pmin, dims=["xyz", "icell"])
            pro["pmax_atm"] = xr.DataArray(pmax, dims=["xyz", "icell"])
            pro["neighbour_atm"] = xr.DataArray(
                neighbour, dims=["faces", "icell"]
            )

        return pro

    def phase(self, wav, NBTHETA=721):
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
        wav : scalar or array-like
            Wavelengths at which to calculate phase matrix [nm].
            If scalar, will be converted to 1-D array.
        NBTHETA : int, optional
            Number of scattering angles for angle resampling. Default is
            721,
            corresponding to angles from 0° to 180°.

        Returns
        -------
        phase_matrix : xr.DataArray or None
            DataArray containing the weighted average phase matrix with
            axes
            [wav_phase, z_phase, stk, theta_atm] if aerosol components
            are present.
            Shape is (len(wav), nz, nphamat, NBTHETA) where:
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
        """
        wav = np.atleast_1d(wav)
        pha = None
        norm = None
        rh = self.prof_red.relative_humidity()

        for comp in self.comp:
            dtau, ssa_p = comp.dtau_ssa(wav, self.pfgrid, rh=rh)
            comp_pha = comp.phase(wav, self.pfgrid, rh, NBTHETA=NBTHETA)
            if hasattr(comp_pha, "to_xarray"):
                comp_pha = comp_pha.to_xarray()

            # dtau/ssa grids are defined on pfgrid boundaries; skip TOA
            # bound to match z_phase layers.
            weight_2d = xr.DataArray(
                dtau[:, 1:] * ssa_p[:, 1:],
                dims=["wav_phase", "z_phase"],
                coords={
                    "wav_phase": comp_pha.coords["wav_phase"].values,
                    "z_phase": comp_pha.coords["z_phase"].values,
                },
            )

            weighted_pha = comp_pha * weight_2d
            pha = weighted_pha if pha is None else (pha + weighted_pha)
            norm = weight_2d if norm is None else (norm + weight_2d)

        if len(self.comp) > 0:
            pha = (pha / norm).fillna(0.0)
            return pha
        else:
            return None

    def calc_split(self, wav, phase=True, NBTHETA=721):
        """
        Computes atmospheric optical properties at specified wavelengths
        and
        separates them into decomposed components (absorption, Rayleigh
        scattering,
        aerosols, and phase functions). These returned profiles can be
        used as
        alternative inputs to initialize a new AtmAFGL instance.

        Parameters
        ----------
        wav : scalar or array-like
            Wavelengths at which to calculate optical properties [nm].
        phase : bool, optional
            If True (default), calculates phase functions. Set to False
            to skip
            phase function computations for faster execution.
        NBTHETA : int, optional
            Number of scattering angles for phase function resampling.
            Default is 721,
            corresponding to angles from 0° to 180°. Only used if
            phase=True.

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
            - pro_phases: List of phase matrix LUT objects, one for each
              phase index

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
        >>> atm = AtmAFGL('afglus')
        >>> (prof_abs, prof_ray, (prof_aer, ssa_aer)
        ...  (pro_iphase, pro_phases)) = atm.calc_split(wav=500.)
        """
        pro = self.calc(wav=wav, phase=phase, NBTHETA=NBTHETA)
        pro_aer = diff1(pro["OD_p"].values.astype(np.float32), axis=1)
        ssa_aer = pro["ssa_p_atm"].values
        pro_ray = diff1(pro["OD_r"].values.astype(np.float32), axis=1)
        pro_abs = diff1(pro["OD_g"].values.astype(np.float32), axis=1)
        pro_iphase = pro["iphase_atm"].values
        pro_phases = [
            pro["phase_atm"].sel(iphase=i).values
            for i in range(int(pro_iphase.max()) + 1)
        ]

        return pro_abs, pro_ray, (pro_aer, ssa_aer), (pro_iphase, pro_phases)


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
    fname : str | Path
        Path to atmospheric profile file. Accepts .nc (NetCDF) or .dat
        (libratran) formats.
        If only filename is provided (no path), the auxdata directory is
        automatically prepended.
        If no suffix is provided, .nc is assumed by default.
    tco3 : float | None, optional
        Total column vertically-integrated ozone in Dobson units (DU).
        If None, uses the
        value from the atmospheric profile. The ozone profile is scaled
        to match this column
        amount. Note: 1 DU = 2.1415e-5 kg m⁻².
        Default: None
    tcwp : float | None, optional
        Total column vertically-integrated water vapour in g/cm². If
        None, uses the value
        from the atmospheric profile. The water vapour profile is scaled
        to match this column
        amount.
        Default: None
    tcno2 : bool | None, optional
        Total column vertically-integrated NO2. If False, NO2 density is
        set to zero.
        If True, NO2 profile from the atmospheric file is retained.
        Default: True
    p0 : float | None, optional
        Sea surface (bottom layer) pressure in hPa. If None, uses the
        pressure
        from the atmospheric profile. Scales all pressure values
        proportionally.
        Default: None
    rh_cst : float | None, optional
        Force relative humidity to be constant at this value. If None,
        relative
        humidity is recalculated from the temperature and water vapor
        profiles.
        Default: None
    o3_h2o_alt : float | None, optional
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
        fname,
        tco3=None,
        tcwp=None,
        tcno2=True,
        p0=None,
        rh_cst=None,
        o3_h2o_alt=None,
    ):

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
                self.P = data[:, 1]  # pressure in hPa
                self.T = data[:, 2]  # temperature in K
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
                raise NameError("Invalid atmospheric file format")
        elif fname.suffix == ".nc":
            with xr.open_dataset(fname) as data:
                self.z = data.coords["z_atm"].values  # Altitude in km
                self.P = data["P"].values  # pressure in hPa
                self.T = data["T"].values  # temperature in K
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
            M_H2O = 18.015  # g/mol
            Avogadro = constants.value("Avogadro constant")
            if o3_h2o_alt is None:
                denom = simpson(y=self.dens_h2o, x=-self.z) * 1e5
                self.dens_h2o *= tcwp / M_H2O * Avogadro / denom
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
                ) / Avogadro
                self.dens_h2o *= tcwp / h2o_afgl
            if tcwp == 0:
                self.dens_h2o[:] = 0.0

        if p0 is not None:
            self.P *= p0 / self.P[-1]

        if not tcno2:
            self.dens_no2[:] = 0.0

    def regrid(self, znew):
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
        znew : 1-D ndarray
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
            - P: pressure (hPa)
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
            - RH_cst: constant relative humidity (None | float)
        """
        prof = ProfileBase(None)
        z = self.z
        prof.z = znew
        _s = np.argsort(z)
        try:
            prof.P = np.interp(
                znew, z[_s], self.P[_s], left=1012.0, right=1e-5
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
        _tmpT = make_interp_spline(z[_s], self.T[_s], k=1)
        prof.T = _tmpT(znew)

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

    def relative_humidity(self):
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
            rh = np.full_like(self.T, self.rh_cst, dtype=float)
        else:
            p_h2o = (self.dens_h2o / self.dens_air) * self.P
            p_sat = saturation_pressure(self.T)
            rh = (p_h2o / p_sat) * 100

        return rh


def saturation_pressure(T):
    """Calculate saturation vapor pressure for water and ice phases.

    Uses the Huang (2018) empirical formula, which provides accurate
    saturation vapor pressure calculations for both liquid water and ice
    phases.

    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin [K]

    Returns
    -------
    sat_press : float or numpy.ndarray
        Saturation vapor pressure [hPa]

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
    tc = T - 273.15  # temperature in C°
    sat_press = np.zeros_like(tc)

    is_water = tc > 0
    is_ice = np.logical_not(is_water)

    sat_press[is_water] = (
        np.exp(34.494 - 4924.99 / (tc[is_water] + 237.1))
    ) / ((tc[is_water] + 105) ** 1.57)

    sat_press[is_ice] = (np.exp(43.494 - (6545.8 / (tc[is_ice] + 278)))) / (
        (tc[is_ice] + 868) ** 2
    )
    return sat_press * 1e-2


def f_n2(lam):
    """Compute the depolarization factor of N2 as a function of
    wavelength.

    Parameters
    ----------
    lam : float | 1-D ndarray
        Wavelength in micrometers (μm).

    Returns
    -------
    out : float | 1-D ndarray
        Depolarization factor of N2. Same shape as input `lam`.

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    return 1.034 + 3.17 * 1e-4 * lam ** (-2)


def f_o2(lam):
    """Compute the depolarization factor of O2 as a function of
    wavelength.

    Parameters
    ----------
    lam : float | 1-D ndarray
        Wavelength in micrometers (μm).

    Returns
    -------
    out : float | 1-D ndarray
        Depolarization factor of O2. Same shape as input `lam`.

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    return 1.096 + 1.385 * 1e-3 * lam ** (-2) + 1.448 * 1e-4 * lam ** (-4)


def f_air_co2(lam, co2):
    """Calculates the depolarization factor for air using a composite
    formula based on the depolarization factors of N2 and O2, and the
    CO2 concentration. Produces a 2-D array with one value per
    wavelength-layer combination.

    Parameters
    ----------
    lam : 1-D ndarray
        Wavelength values in micrometers (μm). Shape: (N,)
    co2 : 1-D ndarray
        CO2 concentration in parts per million (ppm). Shape: (M,)

    Returns
    -------
    out : 2-D ndarray
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
    _FN2 = f_n2(lam).reshape((-1, 1))
    _FO2 = f_o2(lam).reshape((-1, 1))
    _CO2 = co2.reshape((1, -1))

    return (78.084 * _FN2 + 20.946 * _FO2 + 0.934 + _CO2 * 1e-4 * 1.15) / (
        78.084 + 20.946 + 0.934 + _CO2 * 1e-4
    )


def n_air_co2_300(lam):
    """Compute the refractive index of dry air at 300 ppm CO2 as a
    function of wavelength.

    Parameters
    ----------
    lam : float | ndarray
        Wavelength in micrometers (μm).

    Returns
    -------
    out : float | 1-D ndarray
        Refractive index of dry air at 300 ppm CO2. Same shape as input
        `lam`.

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    return (
        1e-8
        * (
            8060.51
            + 2480990 / (132.274 - lam ** (-2))
            + 17455.7 / (39.32957 - lam ** (-2))
        )
        + 1.0
    )


def n_air_co2(lam, co2):
    """Calculates the refractive index as function of wavelength and CO2
    concentration.

    Parameters
    ----------
    lam : 1-D ndarray
        Wavelength values in micrometers (μm). Shape: (N,)
    co2 : 1-D ndarray
        CO2 concentration in parts per million (ppm). Shape: (M,)

    Returns
    -------
    out : 2-D ndarray
        Refractive index of air. Shape: (N, M), where N is the number of
        wavelengths
        and M is the number of layers.

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    N300 = n_air_co2_300(lam).reshape((-1, 1))
    CO2 = co2.reshape((1, -1))
    return (N300 - 1) * (1 + 0.54 * (CO2 * 1e-6 - 0.0003)) + 1.0


def m_dry_air(co2):
    """Compute the mean molecular weight of dry air as a function of CO2
    concentration.

    Parameters
    ----------
    co2 : float | ndarray
        CO2 concentration in parts per million (ppm).

    Returns
    -------
    float | ndarray
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
    return 15.0556 * co2 * 1e-6 + 28.9595


def rayleigh_crs(lam, co2):
    """Compute the Rayleigh cross section.

    Parameters:
    -----------
    lam : 1-D ndarray
        The wavelength(s) in um
    co2 : float | 1-D ndarray
        CO2 concentration(s) in ppm

    Returns:
    out : 2-D ndarray
        The Rayleigh cross section (N wavelengths x M layers)

    References
    ----------
    .. [1] Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R.
    (1999).
        On Rayleigh Optical Depth Calculations. *Journal of Atmospheric
        and Oceanic
        Technology*, 16, 1854-1861.
    """
    if not isinstance(lam, np.ndarray):
        raise ValueError("The parameter lam must be a 1-D np.ndarray.")
    if not np.isscalar(co2) and not isinstance(co2, np.ndarray):
        raise ValueError(
            "The parameter co2 must be a scalar or a 1-D ndarray."
        )

    # Ensure float64 due to numpy 2
    lam = lam.astype(np.float64)
    co2 = np.float64(co2)

    Avogadro = constants.value("Avogadro constant")
    Ns = Avogadro / 22.4141 * 273.15 / 288.15 * 1e-3
    nn2 = n_air_co2(lam, co2) ** 2

    return (
        24
        * np.pi**3
        * (nn2 - 1) ** 2
        / (lam[:, None] * 1e-4) ** 4
        / Ns**2
        / (nn2 + 2) ** 2
        * f_air_co2(lam, co2)
    )


def gravity_z0(lat):
    """Compute gravitational acceleration at Earth's surface as a
    function of latitude.

    Parameters
    ----------
    lat : float
        Latitude in degrees (positive for North, negative for South).

    Returns
    -------
    out : float
        Gravitational acceleration at ground level in m/s².

    References
    ----------
    .. [1] List, R. J. (1968). *Smithsonian Meteorological Tables*
    (Sixth revised
           edition; fourth reprint issued 1968). Smithsonian Institution
           Press,
           City of Washington, 527 pp.
    """
    if not np.isscalar(lat):
        raise ValueError("The parameter lat must be a scalar value.")

    return 980.6160 * (
        1.0
        - 0.0026372 * np.cos(2 * lat * np.pi / 180.0)
        + 0.0000059 * np.cos(2 * lat * np.pi / 180.0) ** 2
    )


def gravity_z(lat, z):
    """Compute gravitational acceleration at a given altitude and
    latitude.

    Parameters
    ----------
    lat : float
        Latitude in degrees (positive for North, negative for South).
    z : float | 1D-ndarray | list
        Altitude(s) above sea level in meters.

    Returns
    -------
    out : float | 1D-ndarray
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
    if not np.isscalar(lat):
        raise ValueError("The parameter lat must be a scalar value.")

    if isinstance(z, list):
        z = np.asarray(z)

    return (
        gravity_z0(lat)
        - (3.085462 * 1.0e-4 + 2.27 * 1.0e-7 * np.cos(2 * lat * np.pi / 180.0))
        * z
        + (7.254 * 1e-11 + 1e-13 * np.cos(2 * lat * np.pi / 180.0)) * z**2
        - (1.517 * 1e-17 + 6 * 1e-20 * np.cos(2 * lat * np.pi / 180.0)) * z**3
    )


def rayleigh_od(
    lam, co2=400.0, lat=45.0, z=0.0, P=1013.25, pressure="surface"
):
    """
    Rayleigh optical depth from Bodhaine et al, 99 (N wavelengths x M
    layers)
        lam : wavelength in um (N)
        co2 : ppm (M)
        lat : deg (scalar)
        z : altitude in m (M)
        P : pressure in hPa (M)
            (surface or sea-level)
        pressure: str
            - 'surface': P provided at altitude z
            - 'sea-level': P provided at altitude 0
    """
    Avogadro = constants.value("Avogadro constant")
    zs = 0.73737 * z + 5517.56  # effective mass-weighted altitude
    G = gravity_z(lat, zs)
    # air pressure at the pixel (i.e. at altitude) in hPa
    if pressure == "sea-level":
        # air pressure at pixel location in dyn / cm2, i.e. hPa * 1000
        Psurf = (P * (1.0 - 0.0065 * z / 288.15) ** 5.255) * 1000.0
    elif pressure == "surface":
        Psurf = P * 1000.0  # convert to dyn/cm2
    else:
        raise ValueError(f"Invalid pressure type ({pressure})")

    return rayleigh_crs(lam, co2) * Psurf * Avogadro / m_dry_air(co2) / G


def refractivity(lam, P, T, co2):
    """Calculate the refractive index of air as a function of
    wavelength, pressure, temperature, and CO2 concentration.

    Parameters
    ----------
    lam : array_like
        Wavelength in micrometers (um), shape (N,)
    P : array_like
        Atmospheric pressure in hectopascals (hPa), shape (M,)
    T : array_like
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
    p = P * 100.0
    t = T - 273.15
    Ntp = 1 + (n_air_co2(lam[:], co2) - 1) * p * (
        1.0 + p * (60.1 - 0.972 * t) * 1e-10
    ) / (96095.43 * (1 + 0.003661 * t))
    return Ntp


def diff1(
    a: np.ndarray, axis: int = 0, samesize: bool = True
) -> np.ndarray:
    """
    Calculate the first difference of an array along a specified axis.

    Computes the difference between consecutive elements of the array
    along
    the specified axis using `numpy.diff`. By default (samesize=True),
    preserves the original array shape by padding with zeros.

    Parameters
    ----------
    a : ndarray
        Input array for which to compute differences.
    axis : int, optional
        Axis along which differences are computed. Default is 0.
    samesize : bool, optional
        If True (default), the output has the same shape as the input
        array
        with the first slice along the specified axis set to zero. If
        False,
        the output has size reduced by 1 along the specified axis.

    Returns
    -------
    diff : ndarray
        Differences between consecutive elements along the specified
        axis.
        If `samesize=True`, the result has the same shape as `a`.
        If `samesize=False`, the result has shape ``a.shape[axis] - 1``
        along
        the specified axis.

    Examples
    --------
    >>> a = np.array([[1, 2, 4, 8], [10, 20, 40, 80]])
    >>> diff1(a, axis=0, samesize=True)
    array([[ 0,  0,  0,  0],
           [ 9, 18, 36, 72]])
    >>> diff1(a, axis=0, samesize=False) # equivalent to np.diff(a, axis=0)
    array([[ 9, 18, 36, 72]])
    """
    if samesize:
        b = np.zeros_like(a)
        key = [slice(None)] * a.ndim
        key[axis] = slice(1, None, None)
        b[tuple(key)] = np.diff(a, axis=axis)[:]
        return b
    else:
        return np.diff(a, axis=axis)


def od2k(
    prof: xr.Dataset,
    dataset: str,
    axis: int = 1,
    zreverse: bool = False,
) -> np.ndarray:
    """Convert cumulated optical depth to a vertical coefficient
    profile.

    Parameters
    ----------
    prof : xr.Dataset
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
    wav: float | np.ndarray, T: float | np.ndarray
) -> float | np.ndarray:
    """
    Calculate the spectral blackbody radiance.

    Computes the spectral radiance of a perfectly emitting blackbody
    at a given wavelength and temperature according to Planck's law
    of blackbody radiation.

    Parameters
    ----------
    wav : float or ndarray
        Wavelength in meters.
    T : float or ndarray
        Temperature in Kelvin.

    Returns
    -------
    L_b_wl : float or ndarray
        Spectral radiance in W·m⁻³·sr⁻¹.

    References
    ----------
    .. [1] Lenoble, J. (1993). Atmospheric radiative transfer.
           A. Deepak Publishing.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.constants import speed_of_light, Planck, Boltzmann
    >>> wav = 10e-6  # 10 micrometers (thermal infrared)
    >>> T = 288.0    # 288 K (room temperature)
    >>> L_b_wl = blackbody_radiance(wav, T)
    >>> print(f"Spectral radiance: {L_b_wl:.2e} W·m⁻³·sr⁻¹")

    >>> # Calculate for multiple wavelengths at a fixed temperature
    >>> wavelengths = np.array([0.5e-6, 1e-6, 10e-6]) # UV, NIR, TIR
    >>> T = 5778  # Sun's surface temperature
    >>> L_b_wl = blackbody_radiance(wavelengths, T)
    """
    c1 = 2.0 * Planck * speed_of_light**2
    c2 = Planck * speed_of_light / Boltzmann
    L_b_wl = c1 / ((wav**5) * (np.exp(c2 / (wav * T)) - 1.0))
    return L_b_wl


def get_aer_dist_integral(
    Z: float | np.ndarray,
    H_min: float | np.ndarray,
    H_max: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute the integral of exponential vertical distribution between
    two altitudes.

    Calculates the integral of an exponential distribution function over
    a vertical
    layer, used for computing the optical depth contribution of aerosols
    or clouds
    with a scale height Z between altitudes H_min and H_max.

    Parameters
    ----------
    Z : float or ndarray
        Scale height in km. Defines the vertical distribution as N(h) =
        N(0)*exp(-h/Z).
    H_min : float or ndarray
        Minimum altitude in km (bottom of the layer).
    H_max : float or ndarray
        Maximum altitude in km (top of the layer).

    Returns
    -------
    float or ndarray
        Integral of the exponential distribution between H_min and
        H_max,
        normalized by Z.
    """
    return -(Z) * np.exp(-H_max / Z) + (Z) * np.exp(-H_min / Z)


def check_date(
    dates: np.ndarray | list[str], year: int
) -> None:
    """Validate that all dates are from a single year and match the
    provided year.

    Parameters
    ----------
    dates : 1d-array | list
        Dates in format "dd:mm:yyyy" (numpy array or list)
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
    if len(dates) == 0:
        raise ValueError("dates cannot be empty")

    # Extract years from dates using list comprehension
    years = np.unique([int(date.split(":")[-1]) for date in dates])

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


def read_Aeronet_AOD(
    file: str | PathLike, year: int
) -> xr.DataArray:
    """Extract AOD data from Aeronet file.

    Parameters
    ----------
    file : str | Pathlike
        Extinction AOD aeronet file path
    year : int
        The year for 'Day_of_Year(Fraction)' dimension creation

    Returns
    -------
    out : xr.DataArray
        Lookup table with extinction AOD as function of
        Day_of_Year(Fraction) and wavelength
    """
    AOD = pd.read_csv(file, sep=",", skiprows=6)
    NTIME_AOD = AOD.index.size

    check_date(dates=AOD["Date(dd:mm:yyyy)"].values, year=year)

    wav_ext = []
    for key in AOD.keys():
        if "AOD_Extinction-Total" in key:
            str_bis = key.split("[")
            wav_ext.append(float(str_bis[1][:-3]))
    wav_ext = np.unique(wav_ext)
    NWAV_EXT = len(wav_ext)

    mat_ext = np.zeros((NTIME_AOD, NWAV_EXT), dtype=np.float64)
    for itime in range(0, NTIME_AOD):
        for iwav, wav in enumerate(wav_ext):
            key = "AOD_Extinction-Total[" + str(int(wav)) + "nm]"
            mat_ext[itime, iwav] = AOD.iloc[itime][key]

    AOD_ext_lut = xr.DataArray(
        mat_ext,
        coords={
            "Day_of_Year(Fraction)": AOD["Day_of_Year(Fraction)"].values,
            "wavelength": wav_ext,
        },
        dims=["Day_of_Year(Fraction)", "wavelength"],
        name="aod",
    )

    return AOD_ext_lut


def read_Aeronet_SSA(
    file: str | PathLike, year: int
) -> xr.DataArray:
    """Extract SSA data from Aeronet file.

    Parameters
    ----------
    file : str | Pathlike
        Single scattering albedo aeronet file path
    year : int
        The year for 'Day_of_Year(Fraction)' dimension creation

    Returns
    -------
    out : xr.DataArray
        Lookup table with single scattering albedo as function of
        Day_of_Year(Fraction)
        and wavelength
    """
    SSA = pd.read_csv(file, sep=",", skiprows=6)
    NTIME_SSA = SSA.index.size

    check_date(dates=SSA["Date(dd:mm:yyyy)"].values, year=year)

    wav_ssa = []
    for key in SSA.keys():
        if "Single_Scattering_Albedo" in key:
            str_bis = key.split("[")
            wav_ssa.append(float(str_bis[1][:-3]))
    wav_ssa = np.unique(wav_ssa)
    NWAV_SSA = len(wav_ssa)

    mat_ssa = np.zeros((NTIME_SSA, NWAV_SSA), dtype=np.float64)
    for itime in range(0, NTIME_SSA):
        for iwav, wav in enumerate(wav_ssa):
            key = "Single_Scattering_Albedo[" + str(int(wav)) + "nm]"
            mat_ssa[itime, iwav] = SSA.iloc[itime][key]

    SSA_lut = xr.DataArray(
        mat_ssa,
        coords={
            "Day_of_Year(Fraction)": SSA["Day_of_Year(Fraction)"].values,
            "wavelength": wav_ssa,
        },
        dims=["Day_of_Year(Fraction)", "wavelength"],
        name="ssa",
    )

    return SSA_lut


def read_Aeronet_PFN(file: str | PathLike, year: int) -> xr.DataArray:
    """Extract PFN data from Aeronet file.

    Parameters
    ----------
    file : str | Pathlike
        Phase matrix aeronet file path
    year : int
        The year for 'Day_of_Year(Fraction)' dimension creation

    Returns
    -------
    out : xr.DataArray
        Lookup table with phase function matrix as function of
        Day_of_Year(Fraction),
        wavelength and theta_atm
    """
    PFN = pd.read_csv(file, sep=",", skiprows=6)
    PFN = PFN[
        PFN["Phase_Function_Mode"] == "Total"
    ]  # take only total of fine + coarse
    NTIME_PFN = PFN.index.size

    check_date(dates=PFN["Date(dd:mm:yyyy)"].values, year=year)

    ang = []
    wav_pfn = []
    for key in PFN.keys():
        if "0000" in key:
            str_bis = key.split("[")
            ang.append(float(str_bis[0]))
            wav_pfn.append(float(str_bis[1][:-3]))
    ang = np.unique(ang)[::-1]
    wav_pfn = np.unique(wav_pfn)
    NANG = len(ang)
    NWAV_PFN = len(wav_pfn)

    mat_pfn = np.zeros((NTIME_PFN, NWAV_PFN, NANG), dtype=np.float64)
    for itime in range(0, NTIME_PFN):
        for iwav, wav in enumerate(wav_pfn):
            for iang, ag in enumerate(ang):
                ang_str = "%.6f" % float(ag)
                key = ang_str + "[" + str(int(wav)) + "nm]"
                mat_pfn[itime, iwav, iang] = PFN.iloc[itime][key]

    phase_lut = xr.DataArray(
        mat_pfn,
        coords={
            "Day_of_Year(Fraction)": PFN["Day_of_Year(Fraction)"].values,
            "wavelength": wav_pfn,
            "theta_atm": ang,
        },
        dims=["Day_of_Year(Fraction)", "wavelength", "theta_atm"],
        name="pfn",
    )

    return phase_lut


def atm_pro_from_aeronet(
    date: str,
    time: str,
    aod_file: str | xr.DataArray,
    ssa_file: str | xr.DataArray,
    pfn_file: str | xr.DataArray,
    b_wav: list[float] | BandSet,
    pfwav: list[float] | None = None,
    grid: np.ndarray | None = None,
    atm_name: str = "afglt",
    P0: float | None = None,
    O3: float | None = None,
    H2O: float | None = None,
    O3_H2O_alt: float | None = None,
    H_mix_min: float = 0.0,
    H_mix_max: float = 2.0,
    Z_mix: float = 8,
) -> xr.Dataset:
    """
    Create an atmosphere profil from aeronet files

    Parameters
    ----------
    date : str
        Date in the following format -> "yyyy-mm-dd"
    time : str
        Time in the following format -> "hh:mm:ss"
    aod_file : str | xr.DataArray
        Extinction AOD aeronet file (finishing by .aod) or aod DataArray
    ssa_file : str | xr.DataArray
        Single scattering albedo aeronet file (finishing by .ssa) or ssa
        DataArray
    pfn_file : str | xr.DataArray
        Phase matrix aeronet file (finishing by .pfn) or pfn DataArray
    b_wav : list | BandSet
        Kdis bands or list of wavelenghts
    pfwav : list
        List of wavelenghts where the phase functions are computed
    grid : array-like
        Altitude grid profil
    atm_name : str
        The atmAFGL atmosphere used
    P0 : float
        Surface pressure
    O3 : float
        Scale ozone vertical column (Dobson units)
    H2O : float
        Scale Water vertical column
    O3_H2O_alt : float
        Altitude of H2O and O3 values, by default None and scale from
        z=0km
    H_mix_min : float, optional
        Force min altitude of the mixture
    H_mix_max : float, optional
        Force max altitude of the mixture
    Z_mix : float, optional
        Force scale height (see notes) of the mixture

    Returns
    -------
    out : xarray.Dataset
        The atmophere profil. Similar to the output of the calc method
        of AtmAFGL.

    Notes
    -----
    The scale height (see Hess et al. 2004) is the variable Z in the
    following equation:

    - :math:`N(h) = N(0)exp(-h/Z)`

    with N the number density and h the altitude
    """

    pd_date = pd.Timestamp(date + " " + time)
    nb_sec_day = 24 * 60 * 60  # number of seconds in one day
    day_frac = 1 - (
        (
            nb_sec_day
            - (pd_date.hour * 60 * 60 + pd_date.minute * 60 + pd_date.second)
        )
        / nb_sec_day
    )
    day_year_frac = pd_date.day_of_year + day_frac
    print("day_year_frac =", day_year_frac)
    year = pd_date.year

    if isinstance(aod_file, xr.DataArray):
        aod_lut = aod_file
    else:
        aod_lut = read_Aeronet_AOD(aod_file, year=year)
    if isinstance(ssa_file, xr.DataArray):
        ssa_lut = ssa_file
    else:
        ssa_lut = read_Aeronet_SSA(ssa_file, year=year)
    if isinstance(pfn_file, xr.DataArray):
        pfn_lut = pfn_file
    else:
        pfn_lut = read_Aeronet_PFN(pfn_file, year=year)

    if not isinstance(b_wav, BandSet):
        b_wav_BS = BandSet(b_wav)
    else:
        b_wav_BS = b_wav
    b_wav_unique = np.unique(b_wav_BS)
    if pfwav is None:
        pf_wav = b_wav_unique
    else:
        pf_wav = pfwav

    fv_time = "extrapolate"
    aod_lut = aod_lut.interp(
        {"Day_of_Year(Fraction)": day_year_frac, "wavelength": b_wav_unique},
        method="linear",
        kwargs={"fill_value": fv_time},
    ).drop_vars("Day_of_Year(Fraction)")
    ssa_lut = ssa_lut.interp(
        {"Day_of_Year(Fraction)": day_year_frac, "wavelength": b_wav_unique},
        method="linear",
        kwargs={"fill_value": fv_time},
    ).drop_vars("Day_of_Year(Fraction)")
    pfn_lut = pfn_lut.interp(
        {"Day_of_Year(Fraction)": day_year_frac, "wavelength": b_wav_unique},
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
        dims=["wavelength", "stk", "theta_atm"],
        coords={
            "wavelength": pfn_lut.wavelength,
            "stk": np.arange(4),
            "theta_atm": pfn_lut.theta_atm,
        },
    )

    hum = np.array([0.0])
    wav = aod_lut.wavelength.values.copy()
    theta = pfn_lut.theta_atm.values.copy()
    aod = aod_lut.values[None, :]
    ssa = ssa_lut.values[None, :]
    phase = pfn_lut.values[None, :, :, :]

    aer = AerUser(
        aod,
        ssa,
        phase,
        hum,
        wav,
        theta,
        H_mix_min=H_mix_min,
        H_mix_max=H_mix_max,
        Z_mix=Z_mix,
    )
    pro = AtmAFGL(
        atm_name,
        comp=[aer],
        grid=grid,
        P0=P0,
        O3=O3,
        H2O=H2O,
        pfwav=pf_wav,
        O3_H2O_alt=O3_H2O_alt,
    ).calc(b_wav_BS)

    return pro


def _open_lut_datatree_as_xarray(
    input_path: str | Path,
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
    input_path: str | Path,
    output_path: str | Path | None = None,
    h5_group: str | None = None,
    normalize: bool = True,
    overwrite: bool = False,
    veff: float | None = None,
    wl_max: float = 4500,
) -> xr.Dataset:
    """Convert ARTDECO cloud HDF5 file to SMART-G NetCDF file format.

    Reads cloud optical properties from an ARTDECO HDF5 file and
    converts them
    to an xarray dataset compatible with SMART-G cloud inputs.

    Parameters
    ----------
    input_path : str | Path
        Path to the ARTDECO cloud HDF5 file.
    output_path : str | Path, optional
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
    wl_max : float, optional
        Maximum wavelength in nanometers. Only wavelengths <= wl_max are
        included.
        Default: 4500

    Returns
    -------
    xr.Dataset
        Dataset containing cloud optical properties. Includes
        coordinates:

        - reff: effective radius
        - wav: wavelength (nm)
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

        if h5_group is None and "keys" in locals():
            if len(keys) == 1:
                h5_group = keys[0]
            elif len(keys) > 1:
                raise NameError(
                    "The h5 file has more than one group. Please choose one "
                    + "group between: "
                    + ", ".join(keys)
                )
            else:
                raise NameError(
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
        raise NameError(
            "The cloud file is dependant of veff. Please give a veff value "
            + f"between: {veff_min} and {veff_max}"
        )
    elif is_veff:
        art_cld = art_cld.interp(veff=np.array([veff])).squeeze(
            "veff", drop=True
        )

    reff = art_cld.coords["reff"].to_numpy().astype(np.float32, copy=False)
    nreff = len(reff)

    wav_full = np.round(
        art_cld.coords["wavelengths"].to_numpy().astype(np.float64, copy=False)
        * 1e3,
        decimals=3,
    ).astype(np.float32, copy=False)
    wav_idx = np.flatnonzero(wav_full <= wl_max)
    wav = wav_full[wav_idx]
    nwav = len(wav)

    stk = np.arange(nstk, dtype=np.int16)

    mu = art_cld.coords["mu"].to_numpy().astype(np.float64, copy=False)
    theta_unsorted = np.rad2deg(np.arccos(mu))
    theta_idx = np.argsort(theta_unsorted)
    theta = theta_unsorted[theta_idx]
    mu_sorted = mu[theta_idx]
    ntheta = len(theta)

    phase = np.zeros((nreff, nwav, nstk, ntheta), dtype=np.float32)
    for ipc, pc in enumerate(phase_comp):
        phac = art_cld[pc].transpose("reff", "wavelengths", "mu")
        phac = phac.isel(wavelengths=wav_idx, mu=theta_idx)
        phase[:, :, ipc, :] = phac.to_numpy().astype(np.float32, copy=False)

    if nstk == 6:
        pha_desc = (
            "phase matrix integral normalized to 2. stk order: p11, "
            + "p21, p33, p34, p22 and p44"
        )
    if nstk == 4:
        pha_desc = (
            "phase matrix integral normalized to 2. stk order: p11, "
            + "p21, p33 and p34"
        )

    # integral of P11 must be equal to 2
    if normalize:
        for iwav in range(0, nwav):
            for ireff in range(0, nreff):
                f = phase[ireff, iwav, 0, :]  # P11
                Norm = np.trapezoid(f, -mu_sorted)
                phase[ireff, iwav, :, :] *= 2.0 / abs(Norm)

    ext = (
        art_cld["Cext"]
        .transpose("reff", "wavelengths")
        .isel(wavelengths=wav_idx)
    )
    ssa = (
        art_cld["single_scattering_albedo"]
        .transpose("reff", "wavelengths")
        .isel(wavelengths=wav_idx)
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
            "wav": wav,
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
    new AtmAFGL instance.

    Parameters
    ----------
    ds_sg : xr.Dataset
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
    >>> from smartg.atmosphere import extract_split, AtmAFGL
    >>> prof_abs, prof_ray, prof_aer, \
    ...     prof_phases = extract_split(mlut_result)
    >>> new_atm = AtmAFGL('afglt', prof_abs=prof_abs, prof_ray=prof_ray,
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
