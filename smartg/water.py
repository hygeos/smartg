#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Preprocessing of oceanic optical properties for SMART-G simulations.

This module provides tools to build and preprocess water column profiles
for use as input to SMART-G radiative transfer simulations. Pure water
absorption and scattering are always present and computed intrinsically
(the water-equivalent of Rayleigh scattering in the atmosphere);
hydrosols (particles, CDOM, phytoplankton...) are added on top of it to
build a complete water model.

Workflow
--------
Typical usage involves:
1. Create a water profile using model classes (e.g., Water1D)
2. Add hydrosol components (chlorophyll-driven models, user-supplied
   inherent optical properties) as needed
3. (Optional) Call the profile's `calc()` method to compute optical
   properties with specific parameters (if using optional parameters not
   set by default)
4. Pass the resulting profile object as the `water` parameter to
   `smartg.run()`

Key Classes
-----------
Water1D
    1D water column profile model. Provides the depth grid, the pure
    water absorption and scattering coefficients (read from auxiliary
    data and from the standard spectral law), and the seafloor albedo.
    Hydrosols can be added to build a complete water model.

Hydrosol
    Hydrosol optical properties supplied directly by the user, i.e. the
    particle and CDOM absorption coefficients, the particle scattering
    coefficient, and either the phase matrices or the backscattering
    ratio from which a Fournier-Forand phase matrix is derived.

HydrosolPR
    Chlorophyll-driven hydrosol model using the Park & Ruddick
    parameterization. Like Hydrosol, but the absorption, scattering and
    backscattering ratio are derived from a chlorophyll concentration
    instead of being supplied by the user.

HydrosolZhai
    Chlorophyll-driven hydrosol model described in Zhai et al. (2017).
    Like HydrosolPR, but the chlorophyll concentration varies with
    depth, following the stratified trophic profile of Uitz et al.
    (2006).

WaterRw
    Model of water reflectance (lambertian reflector under the surface),
    without any water column optics.
"""

from __future__ import annotations
import numpy as np
import xarray as xr
from smartg.diff import diff1
from smartg.albedo import AlbedoCst, AlbedoLike
from smartg.phase import (
    integ_phase,
    calc_iphase,
    expand_phase_4_to_6,
)
from smartg.truncation import DM_trunc, GT_trunc
from pytrunc.phase import fournier_forand
from pytrunc.truncation import delta_m_phase_approx, gt_phase_approx
from smartg.bandset import BandSet
from smartg.config import DIR_AUXDATA as dir_aux
from smartg.interp import interp_1d_coord
from smartg.typing import PathType, NumericArrayLike
from pathlib import Path
from typing import TypedDict, cast
from numpy.typing import NDArray
from luts.luts import LUT


#: Default truncation of the derived water phase functions: the forward
#: peak below 5 deg is replaced following Iwabuchi & Suzuki (2009), with
#: a truncation fraction of 0.3 (larger fractions make the truncated
#: phase function negative for the most forward-peaked Fournier-Forand
#: mixtures).
DEFAULT_WATER_TRUNC = GT_trunc(trunc_frac=0.3, theta_tr=5.0)


def _truncate_f11(
    f11: NDArray,
    theta_deg: NDArray,
    truncation: DM_trunc | GT_trunc,
) -> tuple[NDArray, float]:
    """
    Truncate a single scalar phase function with pytrunc.

    The `pha_scale_method` attribute of the truncation has no effect
    here: only the F11 (and F22 = F11) terms of the water phase matrices
    are non-null, so the rescaling of the other terms is the identity.

    Parameters
    ----------
    f11 : ndarray
        1-D phase function, normalized to 2 over `theta_deg`.
    theta_deg : ndarray
        Scattering angles in degrees.
    truncation : DM_trunc or GT_trunc
        Truncation configuration.

    Returns
    -------
    f11_tr : ndarray
        Truncated phase function, normalized to 2 over `theta_deg`.
    f : float
        Truncation fraction, i.e. the fraction of the scattered energy
        removed with the forward peak.

    Raises
    ------
    ValueError
        If the truncation configuration is not recognized.
    """
    if isinstance(truncation, DM_trunc):
        ds_pha = cast(
            xr.Dataset,
            delta_m_phase_approx(
                f11,
                theta_deg,
                truncation.m_max,
                method=truncation.integral_method,
            ),
        )
    elif isinstance(truncation, GT_trunc):
        ds_pha = cast(
            xr.Dataset,
            gt_phase_approx(
                f11,
                theta_deg,
                truncation.trunc_frac,
                method=truncation.integral_method,
                th_tol=truncation.theta_tol,
                th_f=truncation.theta_tr,
                lobatto_optimization=truncation.lobatto_optimization,
            ),
        )
    else:
        raise ValueError("truncation method not recognized")

    return ds_pha["phase_tr"].values, float(ds_pha["f"].values)


class IOPDict(TypedDict):
    """
    Inherent optical properties returned by the `iop` and `coeffs`
    methods of the hydrosols. All entries are coefficients in m-1 with
    dimensions [nwav, nz], except `bbp_ratio` which is dimensionless and
    may be None when no backscattering ratio is available.
    """

    ap: NDArray
    bp: NDArray
    acdom: NDArray
    bbp_ratio: NDArray | None
    aphy: NDArray
    fqyc: NDArray


def _read_aw(dir_aux: PathType) -> xr.DataArray:
    """
    Read pure water absorption coefficient.

    Combines data from [1]_ for wavelengths <= 725 nm and [2]_ for
    wavelengths > 725 nm. Values are converted from cm^-1 to m^-1.

    Parameters
    ----------
    dir_aux : path-like
        Path to the auxiliary data directory. Must contain
        ``water/pope97.dat`` and ``water/palmer74.dat``.

    Returns
    -------
    aw : DataArray
        Pure water absorption coefficient [m^-1] as a function of
        wavelength [nm], with dimension ``('wavelength',)``.

    References
    ----------
    .. [1] R. M. Pope and E. S. Fry, "Absorption spectrum (380-700 nm)
       of pure water. II. Integrating cavity measurements," Appl. Opt.
       36, 8710-8723 (1997).
       https://doi.org/10.1364/AO.36.008710
    .. [2] K. F. Palmer and D. Williams, "Optical properties of water
       in the near infrared," J. Opt. Soc. Am. 64, 1107-1110 (1974).
       https://doi.org/10.1364/JOSA.64.001107
    """

    # Pope&Fry
    with open(Path(dir_aux) / "water" / "pope97.dat", "rb") as fp:
        for _ in range(6):
            fp.readline()  # skip the first 6 lines
        data_pf = np.genfromtxt(fp)
    aw_pf = data_pf[:, 1] * 100  #  convert from cm-1 to m-1
    lam_pf = data_pf[:, 0]
    ok_pf = lam_pf <= 725

    # Palmer&Williams
    data_pw = np.genfromtxt(
        Path(dir_aux) / "water" / "palmer74.dat", skip_header=5
    )
    aw_pw = data_pw[::-1, 1] * 100  #  convert from cm-1 to m-1
    lam_pw = data_pw[::-1, 0]
    ok_pw = lam_pw > 725

    aw = xr.DataArray(
        np.array(list(aw_pf[ok_pf]) + list(aw_pw[ok_pw])),
        dims=["wavelength"],
        coords={
            "wavelength": np.array(list(lam_pf[ok_pf]) + list(lam_pw[ok_pw]))
        },
    )

    return aw


class Hydrosol(object):
    """
    User-defined hydrosol model.

    The inherent optical properties are supplied directly, either as
    scalars or as arrays over the wavelength and depth grids of the
    Water1D profile the hydrosol is added to.

    Parameters
    ----------
    phase : DataArray or LUT or None, optional
        Phase matrices with dimensions [nwav, nz, stk, angle]. If None,
        the phase matrices are derived from `bbp_ratio` (see notes).
    bp : array_like or None, optional
        Particle scattering coefficient in m-1, dimensions [nwav, nz].
        If None, it is taken as null. A scalar or a lower-dimensional
        array is broadcast over [nwav, nz].
    ap : array_like or None, optional
        Particle absorption coefficient in m-1, same shape rules as
        `bp`.
    acdom : array_like or None, optional
        CDOM absorption coefficient in m-1, same shape rules as `bp`.
    bbp_ratio : array_like or None, optional
        Backscattering ratio (dimensionless), same shape rules as `bp`.
        Only used if `phase` is not provided.
    n_theta : int, optional
        Number of angles of the derived phase matrices.
    truncation : DM_trunc or GT_trunc or None, optional
        Truncation of the forward peak of the derived phase matrices,
        performed with pytrunc (see `smartg.truncation`, and the
        truncation of the atmospheric phase matrices in `Atm1D.calc`).
        None disables the truncation. Defaults to
        `DEFAULT_WATER_TRUNC`.
    pfwav : array_like or None, optional
        Wavelengths in nm at which the phase matrices are calculated. If
        None, they are calculated at all wavelengths.

    Raises
    ------
    TypeError
        If `phase` is neither a DataArray, a LUT nor None.

    Notes
    -----
    When `phase` is not provided, the phase matrices are derived from
    the backscattering ratio `bbp_ratio` following Park & Ruddick
    (2005), as a mixture of two Fournier-Forand phase functions. Their
    forward peak is truncated as configured by `truncation`, and the
    scattering coefficient `bp` is scaled by `1 - f`, with `f` the
    truncated fraction of the scattered energy.

    The pure water absorption and scattering coefficients are not
    defined here but in the Water1D profile, since pure water is always
    present.

    References
    ----------
    .. [1] Y.-J. Park and K. Ruddick, "Model of remote-sensing
       reflectance including bidirectional effects for case 1 and case 2
       waters," Appl. Opt. 44, 1236-1249 (2005).
    """

    def __init__(
        self,
        phase: xr.DataArray | LUT | None = None,
        bp: NumericArrayLike | None = None,
        ap: NumericArrayLike | None = None,
        acdom: NumericArrayLike | None = None,
        bbp_ratio: NumericArrayLike | None = None,
        n_theta: int = 721,
        truncation: DM_trunc | GT_trunc | None = DEFAULT_WATER_TRUNC,
        pfwav: NumericArrayLike | None = None,
    ) -> None:
        self.bp = bp
        self.ap = ap
        self.acdom = acdom
        self.bbp_ratio = bbp_ratio
        self._phase = expand_phase_4_to_6(phase)
        self.n_theta = n_theta
        self.truncation = truncation
        self.pfwav = None if pfwav is None else np.array(pfwav)

        self._pha: xr.DataArray | None = None
        self._coef_trunc: xr.DataArray | None = None
        self._bsca: NDArray | None = None

    def iop(self, wav: NDArray, z: NDArray) -> IOPDict:
        """
        Inherent optical properties of the hydrosol at the given
        wavelengths and depths.

        The coefficients supplied at construction time are broadcast
        over the wavelength and depth grids; those left to None are
        taken as null.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.

        Returns
        -------
        IOPDict
            Inherent optical properties, each with dimensions [len(wav),
            len(z)]:

            - 'ap' : particle absorption coefficient in m-1
            - 'bp' : particle scattering coefficient in m-1, before the
              truncation correction
            - 'acdom' : CDOM absorption coefficient in m-1
            - 'bbp_ratio' : backscattering ratio (dimensionless), or
              None if none was provided
            - 'aphy' : phytoplankton absorption coefficient in m-1, of
              which the fraction 'fqyc' fluoresces. Here it is taken
              equal to 'ap'.
            - 'fqyc' : fluorescence quantum yield (dimensionless), null
              for a user-defined hydrosol

        Raises
        ------
        ValueError
            If a supplied coefficient cannot be broadcast over
            [len(wav), len(z)].
        """
        shp = (len(wav), len(z))
        zeros = np.zeros(shp, dtype="float")

        def as_2d(x: NumericArrayLike | None) -> NDArray:
            """Broadcast `x` over [len(wav), len(z)], None giving 0."""
            if x is None:
                return zeros.copy()
            x = np.asarray(x, dtype="float")
            try:
                return np.broadcast_to(x, shp).copy()
            except ValueError:
                raise ValueError(
                    "Cannot evaluate the hydrosol coefficients over "
                    + f"{len(wav)} wavelengths and {len(z)} depths: the "
                    + f"provided arrays have shape {x.shape}."
                ) from None

        ap = as_2d(self.ap)
        return {
            "ap": ap,
            "bp": as_2d(self.bp),
            "acdom": as_2d(self.acdom),
            "bbp_ratio": (
                None if self.bbp_ratio is None else as_2d(self.bbp_ratio)
            ),
            "aphy": ap,
            "fqyc": zeros.copy(),
        }

    def _trunc_scaling(self) -> float:
        """
        Factor applied to the scattering coefficient to account for the
        truncation of the phase matrix forward peak.

        It multiplies the truncation factor `coef_trunc` returned by
        `calc_phase`, and is meant to be overridden by the subclasses
        whose phase matrices follow a different normalization.

        Returns
        -------
        float
            Always 1., i.e. `coef_trunc` is applied unchanged.
        """
        return 1.0

    def calc_phase(
        self,
        wav: NDArray,
        z: NDArray,
        bbp_ratio: NDArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Calculate the phase matrices and the associated truncation
        factor, as a mixture of two Fournier-Forand phase functions
        weighted by the backscattering ratio.

        The forward peak is truncated with pytrunc as configured by
        `truncation` (nothing is truncated when it is None), and the
        phase matrices are normalized to 2 over the angular grid. Only
        the F11 (and F22 = F11) terms are non-null: the mixture is
        treated as a scalar phase function.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.
        bbp_ratio : 2-D ndarray
            Backscattering ratio (dimensionless), dimensions [len(wav),
            len(z)].

        Returns
        -------
        pha_da : DataArray
            Phase matrices with dimensions [wav_phase, z_phase, stk,
            theta_oc].
        coef_trunc : DataArray
            Truncation factor `1 - f` with dimensions [wav_phase,
            z_phase], by which the scattering coefficient must be
            scaled to compensate for the truncated peak (`f` is the
            truncated fraction of the scattered energy). All ones when
            `truncation` is None.

        Raises
        ------
        ValueError
            If the truncated phase function is negative, which happens
            when the imposed truncation fraction exceeds the energy of
            the truncated peak (e.g. a GT truncation with both
            `trunc_frac` and `theta_tr` imposed and a too large
            `trunc_frac`).

        References
        ----------
        .. [1] Y.-J. Park and K. Ruddick, "Model of remote-sensing
           reflectance including bidirectional effects for case 1 and
           case 2 waters," Appl. Opt. 44, 1236-1249 (2005).
        """
        nwav = len(wav)
        nz = len(z)

        # particles phase function
        # see Park & Ruddick, 05
        # https://odnature.naturalsciences.be/downloads/publications/park_appliedoptics_2005.pdf
        ang = np.linspace(
            0, np.pi, self.n_theta, dtype="float64"
        )  # angle in radians
        # pytrunc's raw Fournier-Forand integrates to 1/(4*pi) over the
        # sphere: scale by 4*pi to normalize to 4*pi as before
        with np.errstate(divide="ignore", invalid="ignore"):
            ff1 = 4.0 * np.pi * fournier_forand(
                ang, 1.117, 3.695, theta_unit="rad"
            )
            ff2 = 4.0 * np.pi * fournier_forand(
                ang, 1.05, 3.259, theta_unit="rad"
            )
        # the Fournier-Forand functions diverge at 0: extend the second
        # angular bin into the first
        ff1[0] = ff1[1]
        ff2[0] = ff2[1]

        # the mixture only depends on the backscattering ratio:
        # normalize and truncate each unique value once
        r1 = (bbp_ratio - 0.002) / 0.028
        r1_uniq, inv = np.unique(r1.ravel(), return_inverse=True)
        inv = inv.reshape(nwav, nz)

        f11 = r1_uniq[:, None] * ff1 + (1 - r1_uniq[:, None]) * ff2

        # normalize
        f11 *= 2.0 / integ_phase(ang, f11)[:, None]

        coef = np.ones(len(r1_uniq), dtype="float64")
        if self.truncation is not None:
            theta_deg = np.rad2deg(ang)
            for i in range(len(r1_uniq)):
                f11[i], f = _truncate_f11(
                    f11[i], theta_deg, self.truncation
                )
                coef[i] = 1.0 - f
            if (f11 < 0.0).any():
                raise ValueError(
                    "The truncated water phase function is negative: "
                    "the truncation is inconsistent with the "
                    "Fournier-Forand mixture (e.g. a truncation "
                    "fraction larger than the energy of the truncated "
                    "peak). Lower trunc_frac, or let GT_trunc search "
                    "the truncation angle (theta_tr=None)."
                )

        pha = np.zeros((nwav, nz, 6, self.n_theta), dtype="float64")
        pha[:, :, 0, :] = f11[inv]
        pha[:, :, 4, :] = pha[:, :, 0, :]  # P22 = P11

        pha_da = xr.DataArray(
            pha,
            dims=["wav_phase", "z_phase", "stk", "theta_oc"],
            coords={
                "wav_phase": wav,
                "z_phase": z,
                "theta_oc": np.rad2deg(ang),
            },
        )
        coef_trunc = xr.DataArray(
            coef[inv],
            dims=["wav_phase", "z_phase"],
            coords={"wav_phase": wav, "z_phase": z},
        )

        return pha_da, coef_trunc

    def phase(
        self,
        wav: NDArray,
        z: NDArray,
        use_old_calc_iphase: bool = False,
    ) -> xr.DataArray | None:
        """
        Phase matrices of the hydrosol.

        The phase matrices supplied at construction time are returned as
        such; otherwise they are derived from the backscattering ratio
        (see `calc_phase`) and memoized.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        DataArray or None
            Phase matrices with dimensions [wav_phase, z_phase, stk,
            theta_oc], or None if the hydrosol does not scatter.

        Raises
        ------
        Exception
            If the hydrosol scatters but neither the phase matrices nor
            the backscattering ratio have been provided.
        """
        if self._phase is not None:
            return self._phase

        iop = self.iop(wav, z)
        if not (np.asarray(iop["bp"]) > 0).any():
            return None
        if iop["bbp_ratio"] is None:
            raise Exception(
                "No phase function nor bbp_ratio has been provided, but bp>0"
            )

        self._resolve_truncation(wav, z, use_old_calc_iphase)
        return self._pha

    def _resolve_truncation(
        self,
        wav: NDArray,
        z: NDArray,
        use_old_calc_iphase: bool = False,
    ) -> None:
        """
        Compute the phase matrices at the tabulation wavelengths
        `pfwav`, along with the associated truncation factor.

        The result is memoized in `_pha`, `_coef_trunc` and `_bsca`, so
        that the scattering coefficient and the phase matrices stay
        consistent whichever is requested first. Returns immediately if
        the cache is already filled. A single depth is tabulated when
        neither the backscattering ratio nor the scattering coefficient
        varies vertically.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm. Only used if `pfwav` is None.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated). Currently
            unused.

        Raises
        ------
        Exception
            If the backscattering ratio is not defined at the tabulation
            wavelengths.
        """
        if self._coef_trunc is not None:
            return

        wav_pha = wav if self.pfwav is None else self.pfwav
        z = np.asarray(z, dtype="float")
        iop = self.iop(wav_pha, z)
        bbp_ratio, bp = iop["bbp_ratio"], iop["bp"]
        if bbp_ratio is None:
            raise Exception(
                "No phase function nor bbp_ratio has been provided, but bp>0"
            )

        # tabulate a single depth if neither the phase matrices nor the
        # scattering coefficient vary vertically, to avoid duplicating
        # the phase matrices
        if np.allclose(bbp_ratio, bbp_ratio[:, :1]) and np.allclose(
            bp, bp[:, :1]
        ):
            sl = slice(0, 1)
        else:
            sl = slice(None)

        self._pha, self._coef_trunc = self.calc_phase(
            wav_pha, z[sl], bbp_ratio[:, sl]
        )
        self._bsca = (
            bp[:, sl] * self._coef_trunc.values * self._trunc_scaling()
        )

    def _coef_trunc_on(
        self,
        wav: NDArray,
        z: NDArray,
        use_old_calc_iphase: bool = False,
    ) -> NDArray:
        """
        Truncation factor mapped from the tabulation grid of the phase
        matrices onto the given wavelength and depth grids.

        Must be called after `_resolve_truncation` has filled the cache.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        ndarray
            Truncation factor with dimensions [len(wav), len(z)].
        """
        # only called once _resolve_truncation has filled the cache
        assert (self._pha is not None) and (self._coef_trunc is not None)

        # index with ipha, so that each wavelength/depth gets the factor
        # of the phase matrix it is actually assigned to
        _, ipha = calc_iphase(
            self._pha, np.asarray(wav), np.asarray(z), use_old_calc_iphase
        )
        return self._coef_trunc.values.ravel()[ipha]

    def scattering(self, pha: xr.DataArray) -> NDArray | None:
        """
        Scattering coefficient in m-1 of the hydrosol, on the tabulation
        grid of the given phase matrices.

        Used to weight the hydrosols when averaging their phase matrices
        in `Water1D.phase`.

        Parameters
        ----------
        pha : DataArray
            Phase matrices of this hydrosol, whose `wav_phase` and
            `z_phase` coordinates define the grid of the output.

        Returns
        -------
        ndarray or None
            Scattering coefficient in m-1 with dimensions [wav_phase,
            z_phase], corrected for the phase matrix truncation when the
            phase matrices are derived rather than supplied. None if the
            truncation has not been resolved yet.
        """
        if self._phase is None:
            return self._bsca
        return self.iop(
            pha.coords["wav_phase"].values, pha.coords["z_phase"].values
        )["bp"]

    def coeffs(
        self,
        wav: NDArray,
        z: NDArray,
        phase: bool = True,
        use_old_calc_iphase: bool = False,
    ) -> IOPDict:
        """
        Inherent optical properties of the hydrosol, with the scattering
        coefficient corrected for the phase matrix truncation.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.
        phase : bool, optional
            Whether the phase matrices are calculated. If False, no
            truncation correction is applied.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        IOPDict
            Same as the `iop` method, with 'bp' scaled by the truncation
            factor.

        Raises
        ------
        Exception
            If the hydrosol scatters but neither the phase matrices nor
            the backscattering ratio have been provided.
        """
        iop = self.iop(wav, z)

        if (self._phase is None) and (np.asarray(iop["bp"]) > 0).any():
            if iop["bbp_ratio"] is None:
                raise Exception(
                    "No phase function nor bbp_ratio has been provided, but bp>0"
                )
            if phase:
                self._resolve_truncation(wav, z, use_old_calc_iphase)
                coef_trunc = self._coef_trunc_on(wav, z, use_old_calc_iphase)
                iop["bp"] = iop["bp"] * coef_trunc * self._trunc_scaling()

        return iop


class HydrosolPR(Hydrosol):
    """
    Chlorophyll-driven hydrosol model, using a similar IOP
    parameterization as Polymer's PR model.

    The absorption, scattering and backscattering ratio are all derived
    from a single chlorophyll concentration, which does not vary with
    depth.

    Parameters
    ----------
    chl : float
        Chlorophyll concentration in mg/m3.
    n_theta : int, optional
        Number of angles of the derived phase matrices.
    truncation : DM_trunc or GT_trunc or None, optional
        Truncation of the forward peak of the derived phase matrices
        (see `Hydrosol`). None disables the truncation. Defaults to
        `DEFAULT_WATER_TRUNC`.
    pfwav : array_like or None, optional
        Wavelengths in nm at which the phase matrices are calculated. If
        None, they are calculated at all wavelengths.
    fqyc : float, optional
        Chlorophyll a fluorescence quantum yield.

    Notes
    -----
    The phytoplankton absorption follows Bricaud et al. (1998), the CDM
    absorption Bricaud et al. (2012), and the phase matrices are derived
    from the backscattering ratio as in the base class.

    References
    ----------
    .. [1] A. Bricaud, A. Morel, M. Babin, K. Allali, and H. Claustre,
       "Variations of light absorption by suspended particles with
       chlorophyll a concentration in oceanic (case 1) waters," J.
       Geophys. Res. 103, 31033-31044 (1998).
    .. [2] A. Bricaud, A. M. Ciotti, and B. Gentili, "Spatial-temporal
       variations in phytoplankton size and colored detrital matter
       absorption at global and regional scales," Global Biogeochem.
       Cycles 26, GB1010 (2012).

    Examples
    --------
    >>> from smartg.water import Water1D, HydrosolPR
    >>> water = Water1D(grid=[0, -5.], comp=[HydrosolPR(chl=0.5)])
    """

    def __init__(
        self,
        chl: float,
        n_theta: int = 72001,
        truncation: DM_trunc | GT_trunc | None = DEFAULT_WATER_TRUNC,
        pfwav: NumericArrayLike | None = None,
        fqyc: float = 0.0,
    ) -> None:
        super().__init__(n_theta=n_theta, truncation=truncation, pfwav=pfwav)
        self.chl = chl
        self.fqyc = fqyc

        # Bricaud (98)
        ap_bricaud = np.genfromtxt(
            dir_aux / "water" / "aph_bricaud_1998.txt",
            delimiter=",",
            skip_header=12,
        )  # header is lambda,Ap,Ep,Aphi,Ephi
        self.bricaud = xr.Dataset()
        self.bricaud = self.bricaud.assign_coords(wav=ap_bricaud[:, 0])
        self.bricaud["A"] = xr.DataArray(ap_bricaud[:, 1], dims=["wav"])
        self.bricaud["E"] = xr.DataArray(1 - ap_bricaud[:, 2], dims=["wav"])

    def _trunc_scaling(self) -> float:
        """
        Factor applied to the scattering coefficient, on top of the
        truncation factor (see `Hydrosol._trunc_scaling`).

        Returns
        -------
        float
            Always 0.5, the normalization of the Park & Ruddick phase
            function mixture used by this model.
        """
        return 0.5

    def iop(self, wav: NDArray, z: NDArray) -> IOPDict:
        """
        Inherent optical properties derived from the chlorophyll
        concentration.

        The chlorophyll concentration does not vary with depth, so the
        coefficients are computed spectrally and then broadcast over the
        depth profile.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm.
        z : ndarray
            Vertical grid of the water column in m. Only its length is
            used, since the coefficients are depth-independent.

        Returns
        -------
        IOPDict
            Same entries as `Hydrosol.iop`. Here 'ap' and 'aphy' are
            both the phytoplankton absorption, 'acdom' is the CDM
            absorption, and 'bbp_ratio' is always defined.
        """
        wav = np.asarray(wav, dtype="float")
        chl = self.chl

        # phytoplankton absorption
        aphy = interp_1d_coord(self.bricaud["A"], "wav", wav, extrema=True) * (
            chl ** interp_1d_coord(self.bricaud["E"], "wav", wav, extrema=True)
        )

        # chlorophyll fluorescence (scattering coefficient)
        fqyc = np.full_like(
            aphy, self.fqyc
        )  # Fluorescence Quantum Yield for Chlorophyll
        fqyc[wav < 370.0] = 0.0
        fqyc[wav > 690.0] = 0.0

        # CDM absorption central value
        # from Bricaud et al GBC, 2012 (data from nov 2007)
        fa = 1.0
        acdm443 = fa * 0.069 * (chl**1.070)

        s_cdom = 0.00262 * (acdm443 ** (-0.448))
        if s_cdom > 0.025:
            s_cdom = 0.025
        if s_cdom < 0.011:
            s_cdom = 0.011

        acdm = acdm443 * np.exp(-s_cdom * (wav - 443))

        bp = 0.416 * (chl**0.766) * 550.0 / wav

        #
        # backscattering coefficient
        #
        if chl < 2:
            v = 0.5 * (np.log10(chl) - 0.3)
        else:
            v = 0
        bbp_ratio = 0.002 + 0.01 * (0.5 - 0.25 * np.log10(chl)) * (
            (wav / 550.0) ** v
        )

        shp = (len(wav), len(z))

        def as_2d(x: NDArray) -> NDArray:
            """Broadcast the spectrum `x` over the depth profile."""
            return np.broadcast_to(x[:, None], shp).copy()

        aphy_2d = as_2d(aphy)
        return {
            "ap": aphy_2d,
            "bp": as_2d(bp),
            "acdom": as_2d(acdm),
            "bbp_ratio": as_2d(bbp_ratio),
            "aphy": aphy_2d,
            "fqyc": as_2d(fqyc),
        }


class HydrosolZhai(Hydrosol):
    """
    Chlorophyll-driven hydrosol model described in Zhai et al. (2017),
    where the chlorophyll concentration varies with depth.

    Parameters
    ----------
    chl_surf : float
        Chlorophyll concentration in mg/m3 at the surface. The
        concentration at depth is derived from it (see notes), so this
        is the surface value only, unlike the depth-independent `chl` of
        HydrosolPR.
    n_theta : int, optional
        Number of angles of the derived phase matrices.
    truncation : DM_trunc or GT_trunc or None, optional
        Truncation of the forward peak of the derived phase matrices
        (see `Hydrosol`). None disables the truncation. Defaults to
        `DEFAULT_WATER_TRUNC`.
    pfwav : array_like or None, optional
        Wavelengths in nm at which the phase matrices are calculated. If
        None, they are calculated at all wavelengths.
    euphotic_depth : float or None, optional
        Euphotic depth in m, noted Z_eu in [2]_. This is a positive
        depth, measured downwards from the surface, and not a z
        coordinate: unlike the `grid` of Water1D, it does not follow the
        convention of being negative below the surface. If None, it is
        computed from the chlorophyll climatology of [2]_.
    mixed : bool, optional
        Mixed (True) or stratified (False) waters. Only used to derive
        `euphotic_depth` from the climatology when it is not provided.
    fqyc : float, optional
        Chlorophyll a fluorescence quantum yield.

    Notes
    -----
    The chlorophyll profile is a continuous function of depth, so the
    coefficients are evaluated at whichever depths the Water1D profile
    provides.

    The vertical shape of the chlorophyll profile is the stratified
    trophic case 1 parametrization of [2]_, a Gaussian on a linear
    background expressed in reduced depth (see `chi` and `chl`). The
    Bricaud (1998) phytoplankton absorption is extended down to 360 nm
    following Wei et al. (2016), the spectral slope being taken
    symmetrical with respect to 440 nm over the 360-520 nm range.

    References
    ----------
    .. [1] P.-W. Zhai, Y. Hu, D. M. Winker, B. A. Franz, J. Werdell, and
       Y. Chen, "Vector radiative transfer model for coupled atmosphere
       and ocean systems including inelastic sources in ocean waters,"
       Opt. Express 25, A223-A239 (2017).
    .. [2] J. Uitz, H. Claustre, A. Morel, and S. B. Hooker, "Vertical
       distribution of phytoplankton communities in open ocean: An
       assessment based on surface chlorophyll," J. Geophys. Res. 111,
       C08005 (2006).
    """

    def __init__(
        self,
        chl_surf: float,
        n_theta: int = 7201,
        truncation: DM_trunc | GT_trunc | None = DEFAULT_WATER_TRUNC,
        pfwav: NumericArrayLike | None = None,
        euphotic_depth: float | None = None,
        mixed: bool = False,
        fqyc: float = 0.0,
    ) -> None:
        super().__init__(n_theta=n_theta, truncation=truncation, pfwav=pfwav)
        self.chl_surf = chl_surf
        self.fqyc = fqyc

        # Bricaud (98)
        # Absorption of the phytoplankton
        ap_bricaud = np.genfromtxt(
            dir_aux / "water" / "aph_bricaud_1998.txt",
            delimiter=",",
            skip_header=12,
        )  # header is lambda,Ap,Ep,Aphi,Ephi
        # Add extension to 360 nm (Wei et al., 2016)
        # spectral slope of aph is symetrical wrt 440 nm in the 360-520
        # spectral range
        w_uv = np.linspace(360.0, 398.0, num=20)
        a_bricaud = ap_bricaud[:, 1]
        e_bricaud = 1.0 - ap_bricaud[:, 2]
        w = ap_bricaud[:, 0]
        ii = np.where((w <= 520.0) & (w > 480.0))
        a_uv = np.zeros_like(w_uv)
        e_uv = np.zeros_like(w_uv)
        a_uv[::-1] = a_bricaud[ii]
        e_uv[::-1] = e_bricaud[ii]
        self.bricaud = xr.Dataset()
        self.bricaud = self.bricaud.assign_coords(
            wav=np.concatenate((w_uv, ap_bricaud[:, 0]))
        )
        self.bricaud["A"] = xr.DataArray(
            np.concatenate((a_uv, a_bricaud)), dims=["wav"]
        )
        self.bricaud["E"] = xr.DataArray(
            np.concatenate((e_uv, e_bricaud)), dims=["wav"]
        )

        # Chlorophyll integrated over the euphotic column, from which
        # the euphotic depth is derived when it is not provided
        if euphotic_depth is None:
            if not mixed:
                if chl_surf > 1.0:
                    chl_euphotic = 37.7 * chl_surf**0.615
                else:
                    chl_euphotic = 36.1 * chl_surf**0.357
            else:
                chl_euphotic = 42.1 * chl_surf**0.538
            euphotic_depth = 568.2 * chl_euphotic ** (-0.746)
        self.euphotic_depth = euphotic_depth

        # Reduced concentration chi and reduced depth zeta,
        # stratified trophic case 1 parametrization (Uitz et al., 2006)
        self.chi_b = 0.471
        self.s = 0.135
        self.chi_max = 1.572
        self.zeta_max = 0.969
        self.dzeta = 0.393

    def chi(self, zeta: float | NDArray) -> float | NDArray:
        """
        Reduced chlorophyll concentration at the reduced depth zeta.

        This is the dimensionless vertical shape of the chlorophyll
        profile: a Gaussian deep maximum on a linearly decreasing
        background, as parametrized by Uitz et al. (2006).

        Parameters
        ----------
        zeta : float or ndarray
            Reduced depth, i.e. the depth divided by the euphotic depth
            (positive, 1 at the euphotic depth).

        Returns
        -------
        float or ndarray
            Reduced chlorophyll concentration (dimensionless), same
            shape as `zeta`.
        """
        return (
            self.chi_b
            - self.s * zeta
            + self.chi_max
            * np.exp(-(((zeta - self.zeta_max) / self.dzeta) ** 2))
        )

    def chl(self, z: NumericArrayLike) -> NDArray:
        """
        Chlorophyll concentration at the given z coordinates.

        The reduced profile `chi` is rescaled so that its value at the
        surface is `chl_surf`, and clipped to a small positive value
        where the parametrization would turn negative.

        Parameters
        ----------
        z : array_like
            Vertical grid of the water column in m. The sign is
            irrelevant, since only the distance to the surface matters.

        Returns
        -------
        ndarray
            Chlorophyll concentration in mg/m3, same shape as `z`.
        """
        zeta = np.abs(np.asarray(z, dtype="float") / self.euphotic_depth)
        chl = self.chl_surf * self.chi(zeta) / self.chi(0.0)
        return np.where(chl < 0.0, 1e-8, chl)

    def _trunc_scaling(self) -> float:
        """
        Factor applied to the scattering coefficient, on top of the
        truncation factor (see `Hydrosol._trunc_scaling`).

        Returns
        -------
        float
            Always 0.5, the normalization of the Park & Ruddick phase
            function mixture used by this model.
        """
        return 0.5

    def iop(
        self,
        wav: NDArray,
        z: NDArray,
        p1: float = 0.33,
        r1: float = 0.5,
        r2: float = 0.5,
    ) -> IOPDict:
        """
        Inherent optical properties derived from the chlorophyll
        profile.

        The chlorophyll concentration is evaluated at each z coordinate
        (see `chl`), and the absorption, scattering and backscattering
        are made covariant with it.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.
        p1, r1 : float, optional
            Parameters related to particles extinction, see Zhai et al.
            2017. Currently unused.
        r2 : float, optional
            Parameter related to CDOM absorption, see Zhai et al. 2017.
            Currently unused.

        Returns
        -------
        IOPDict
            Same entries as `Hydrosol.iop`, all varying with depth. Here
            'ap' and 'aphy' are both the phytoplankton absorption,
            'acdom' is the CDOM absorption covariant with it, and
            'bbp_ratio' is that of the non-algal particles.
        """
        wav = np.asarray(wav, dtype="float")
        chl2, wav2 = np.meshgrid(self.chl(z), wav)

        # specific phytoplankton absorption
        chl2star = np.full_like(chl2, 1.0)
        aphystar = interp_1d_coord(
            self.bricaud["A"], "wav", wav2, extrema=True
        ) * (
            chl2star
            ** interp_1d_coord(self.bricaud["E"], "wav", wav2, extrema=True)
        )
        aphy = aphystar * chl2
        aphystar440 = interp_1d_coord(
            self.bricaud["A"], "wav", 440.0, extrema=True
        ) * (
            chl2star
            ** interp_1d_coord(self.bricaud["E"], "wav", wav2, extrema=True)
        )
        aphy440 = aphystar440 * chl2

        # phytoplankton covariant particles extinction
        piz440 = 0.68
        bp440 = aphy440 * piz440 / (1 - piz440)
        bp = bp440 * (wav2 / 440.0) ** (-1.0)

        # chlorophyll fluorescence (scattering coefficient)
        fqyc = np.full_like(
            aphy, self.fqyc
        )  # Fluorescence Quantum Yield for Chlorophyll
        fqyc[wav2 < 370.0] = 0.0
        fqyc[wav2 > 690.0] = 0.0

        # CDOM covariant absorption
        acdm440 = 0.24 * aphy440**0.43
        s_cdom = 0.02
        acdom = acdm440 * np.exp(-s_cdom * (wav2 - 440))

        # non-algal particles backscattering
        spm = 0.0  # g/m3
        gamma = 0.5
        bbpnap650 = 10 ** (
            1.03 * np.log10(spm) - 2.06
        )  # Neukermans et al 2012
        bbpnap = bbpnap650 * (wav2 / 650.0) ** (-gamma)
        bbp_ratio_nap = np.zeros_like(aphy)
        bbp_ratio_nap[:] = 0.04
        bpnap = bbpnap / bbp_ratio_nap
        bp += bpnap

        return {
            "ap": aphy,
            "bp": bp,
            "acdom": acdom,
            "bbp_ratio": bbp_ratio_nap,
            "aphy": aphy,
            "fqyc": fqyc,
        }


class Water(object):
    """Base class for water."""

    pass


class Water1D(Water):
    """
    1D water column profile definition.

    Pure water absorption and scattering are always present and computed
    here; hydrosols are added through the `comp` parameter.

    Parameters
    ----------
    grid : array_like, optional
        Vertical grid of the water column, in m, from the surface down
        to the sea floor. These are z coordinates, not depths: z is 0 at
        the surface and becomes more negative downwards, so the grid
        must be decreasing (e.g. [0., -2.5, -5.]). This is the oceanic
        counterpart of the `grid` parameter of `Atm1D`, and it defines
        the `z_oc` coordinate of the profile returned by `calc()`. Note
        that the first item of the grid is not used.
    comp : list, optional
        Hydrosols to consider, i.e. a list of Hydrosol, HydrosolPR
        or/and HydrosolZhai objects.
    aw : None or 2-D ndarray, optional
        Force the pure water absorption coefficient in m-1, with
        dimensions [nwav, nz]. If None, it is read from the auxiliary
        data (see `_read_aw`).
    bw : None or 2-D ndarray, optional
        Force the pure water scattering coefficient in m-1, with
        dimensions [nwav, nz]. If None, it is computed as
        19.3e-4*(wav/550)**-4.3.
    alb : albedo object, optional
        Albedo of the sea floor, i.e. of the reflector placed at the
        bottom of the water column, at the deepest level of `grid`. This
        is the reflectance of the sea bottom seen from within the water,
        and it is not related to the albedo of the air-water interface:
        the latter is set by the `surf` parameter of `smartg.run()`
        (e.g. `smartg.surface.LambSurface(alb=...)` or
        `RoughSurface(...)`). It fills
        the `albedo_seafloor` variable of the profile returned by
        `calc()`. If None, a black (non-reflecting) sea floor is used,
        i.e. `AlbedoCst(0.)`.

    Examples
    --------
    >>> from smartg.water import Water1D, HydrosolPR
    >>> water = Water1D(comp=[HydrosolPR(chl=0.5)])
    """

    def __init__(
        self,
        grid: NumericArrayLike = (0, -10000),
        comp: list[Hydrosol] | None = None,
        aw: NDArray | None = None,
        bw: NDArray | None = None,
        alb: AlbedoLike | None = None,
    ) -> None:
        self.grid = np.array(grid, dtype="float")
        self.comp = [] if comp is None else comp
        self.aw = aw
        self.bw = bw
        self.alb = AlbedoCst(0.0) if alb is None else alb

        self.aw_table = _read_aw(dir_aux)

    def calc(
        self,
        wav: NumericArrayLike | BandSet,
        phase: bool = True,
        use_old_calc_iphase: bool = False,
    ) -> xr.Dataset:
        """
        Profile and phase matrix calculation at the given wavelengths.

        The pure water and hydrosol coefficients are summed over the
        water column and cumulated into optical thicknesses along
        `grid`. The fluorescing fraction of the phytoplankton absorption
        is counted as (inelastic) scattering rather than absorption.

        Parameters
        ----------
        wav : array_like or BandSet
            Wavelengths in nm at which to calculate the profile.
        phase : bool, optional
            Whether to calculate the phase matrices. If False, the
            scattering coefficients are not corrected for the phase
            matrix truncation either.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        out : Dataset
            The profile, with coordinates `wavelength` and `z_oc`. It
            holds the cumulated optical thicknesses ('OD_oc' and its
            decomposition into 'OD_w', 'OD_p_oc', 'OD_y', 'OD_sca_oc'
            and 'OD_abs_oc'), the single scattering albedos ('ssa_oc',
            'ssa_p_oc', 'ssa_w'), the scattering ratios ('pmol_oc',
            'pine_oc'), the fluorescence quantum yield ('FQY1_oc'), the
            temperature ('T_oc') and the sea floor albedo
            ('albedo_seafloor'). If `phase` is True and at least one
            hydrosol scatters, it also holds 'phase_oc' and 'iphase_oc'
            on the added `theta_oc` coordinate.
        """
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)
        wav = np.array(wav)

        z = self.grid
        shp = (len(wav), len(z))
        wav2 = np.stack([wav] * len(z), axis=1)

        #
        # pure water absorption and scattering
        #
        if self.aw is None:
            aw = interp_1d_coord(self.aw_table, "wavelength", wav2)
        else:
            aw = self.aw

        if self.bw is None:
            bw = 19.3e-4 * ((wav2 / 550.0) ** -4.3)
        else:
            bw = self.bw

        #
        # hydrosols absorption and scattering
        #
        ap = np.zeros(shp, dtype="float")
        bp = np.zeros(shp, dtype="float")
        acdom = np.zeros(shp, dtype="float")
        aphy_fluo = np.zeros(shp, dtype="float")
        aphy = np.zeros(shp, dtype="float")

        for comp in self.comp:
            iop = comp.coeffs(
                wav, z, phase=phase, use_old_calc_iphase=use_old_calc_iphase
            )
            ap += iop["ap"]
            bp += iop["bp"]
            acdom += iop["acdom"]
            aphy += iop["aphy"]
            aphy_fluo += iop["aphy"] * iop["fqyc"]

        # the fluorescing fraction of the phytoplankton absorption is
        # counted as (inelastic) scattering instead of absorption
        atot = aw + ap - aphy_fluo + acdom
        btot = bw + bp + aphy_fluo

        pro = xr.Dataset()
        pro = pro.assign_coords(wavelength=wav[:], z_oc=z)

        pro["T_oc"] = xr.DataArray(
            np.array([280.0] * len(z), dtype="float32"), dims=["z_oc"]
        )

        #
        # phase matrices
        #
        if phase:
            pha = self.phase(wav, use_old_calc_iphase=use_old_calc_iphase)

            if pha is not None:
                pha_, ipha = calc_iphase(
                    pha,
                    pro.coords["wavelength"].values,
                    pro.coords["z_oc"].values,
                    use_old_calc_iphase,
                )

                pro = pro.assign_coords(theta_oc=pha.coords["theta_oc"].values)
                pro["phase_oc"] = xr.DataArray(
                    pha_, dims=["iphase", "stk", "theta_oc"]
                )
                pro["iphase_oc"] = xr.DataArray(
                    ipha, dims=["wavelength", "z_oc"]
                )

        dz = -diff1(z)
        tau_w = -(aw + bw) * dz
        tau_p = -(ap + bp) * dz
        tau_y = -(acdom) * dz
        tau_tot = -(atot + btot) * dz
        tau_sca = -(btot) * dz
        tau_abs = -(atot) * dz
        tau_ine = -(aphy_fluo) * dz
        tau_phy = -(aphy) * dz

        with np.errstate(invalid="ignore"):
            ssa_w = bw / (aw + bw)
        ssa_w[np.isnan(ssa_w)] = 1.0

        with np.errstate(invalid="ignore"):
            ssa_p = bp / (ap + bp)
        ssa_p[np.isnan(ssa_p)] = 1.0

        with np.errstate(invalid="ignore"):
            pmol = bw / (bw + bp)
        pmol[np.isnan(pmol)] = 1.0
        pmol[~np.isfinite(pmol)] = 1.0

        with np.errstate(invalid="ignore", divide="ignore"):
            pine = tau_ine / tau_sca
        pine[np.isnan(pine)] = 0.0
        pine[~np.isfinite(pine)] = 0.0

        with np.errstate(invalid="ignore"):
            ssa = tau_sca / tau_tot
        ssa[np.isnan(ssa)] = 1.0

        # ratio of the fluorescing to the total phytoplankton
        # absorption, i.e. the fluorescence quantum yield weighted over
        # the hydrosols
        with np.errstate(invalid="ignore", divide="ignore"):
            fqy1 = tau_ine / tau_phy
        fqy1[np.isnan(fqy1)] = 0.0
        fqy1[~np.isfinite(fqy1)] = 0.0

        pro["OD_w"] = xr.DataArray(
            np.cumsum(tau_w, out=tau_w, axis=1),
            dims=["wavelength", "z_oc"],
            attrs={
                "description": "Cumulated water optical thickness at each wavelength"
            },
        )

        pro["OD_p_oc"] = xr.DataArray(
            np.cumsum(tau_p, out=tau_p, axis=1),
            dims=["wavelength", "z_oc"],
            attrs={
                "description": "Cumulated oceanic particles optical thickness at each wavelength"
            },
        )

        pro["OD_y"] = xr.DataArray(
            np.cumsum(tau_y, out=tau_y, axis=1),
            dims=["wavelength", "z_oc"],
            attrs={
                "description": "Cumulated CDOM optical thickness at each wavelength"
            },
        )

        pro["OD_oc"] = xr.DataArray(
            np.cumsum(tau_tot, out=tau_tot, axis=1),
            dims=["wavelength", "z_oc"],
        )

        pro["OD_sca_oc"] = xr.DataArray(
            np.cumsum(tau_sca, out=tau_sca, axis=1),
            dims=["wavelength", "z_oc"],
        )

        pro["OD_abs_oc"] = xr.DataArray(
            np.cumsum(tau_abs, out=tau_abs, axis=1),
            dims=["wavelength", "z_oc"],
        )

        pro["pine_oc"] = xr.DataArray(pine, dims=["wavelength", "z_oc"])

        pro["pmol_oc"] = xr.DataArray(pmol, dims=["wavelength", "z_oc"])

        pro["ssa_oc"] = xr.DataArray(ssa, dims=["wavelength", "z_oc"])

        pro["ssa_p_oc"] = xr.DataArray(ssa_p, dims=["wavelength", "z_oc"])
        pro["ssa_w"] = xr.DataArray(ssa_w, dims=["wavelength", "z_oc"])

        pro["FQY1_oc"] = xr.DataArray(fqy1, dims=["wavelength", "z_oc"])

        pro["albedo_seafloor"] = xr.DataArray(
            self.alb.get(wav), dims=["wavelength"]
        )

        return pro

    def phase(
        self,
        wav: NDArray,
        use_old_calc_iphase: bool = False,
    ) -> xr.DataArray | None:
        """
        Calculate the phase matrices of the hydrosols, averaged over the
        hydrosols and weighted by their scattering coefficient.

        The depths are those of `grid`. When a single hydrosol scatters,
        its phase matrices are returned unchanged.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        out : DataArray or None
            The phase matrices with dimensions [wav_phase, z_phase, stk,
            theta_oc], or None if no hydrosol scatters.

        Raises
        ------
        ValueError
            If several hydrosols scatter but their phase matrices are
            not tabulated on the same `wav_phase`, `z_phase` and
            `theta_oc` grids, so that they cannot be averaged.
        """
        z = self.grid

        phases = []
        for comp in self.comp:
            pha = comp.phase(wav, z, use_old_calc_iphase=use_old_calc_iphase)
            if pha is not None:
                phases.append((comp, pha))

        if len(phases) == 0:
            return None
        if len(phases) == 1:
            return phases[0][1]

        ref = phases[0][1]
        for _, pha in phases[1:]:
            for dim in ["wav_phase", "z_phase", "theta_oc"]:
                if not np.array_equal(
                    pha.coords[dim].values, ref.coords[dim].values
                ):
                    raise ValueError(
                        "The phase matrices of the hydrosols must share the "
                        + f"same {dim} grid to be averaged. Use a common "
                        + "pfwav, or provide the phase matrices directly."
                    )

        pha_tot: xr.DataArray | float = 0.0
        bsca: xr.DataArray | float = 0.0
        for comp, pha in phases:
            # weight each hydrosol by its scattering coefficient, on the
            # tabulation grid of its phase matrices
            bsca_ = xr.DataArray(
                comp.scattering(pha),
                dims=["wav_phase", "z_phase"],
                coords={
                    "wav_phase": pha.coords["wav_phase"].values,
                    "z_phase": pha.coords["z_phase"].values,
                },
            )
            bsca = bsca + bsca_
            pha_tot = pha_tot + pha * bsca_

        with np.errstate(divide="ignore", invalid="ignore"):
            pha_tot = pha_tot / bsca
        return cast(xr.DataArray, pha_tot).fillna(0.0)


class WaterRw(Water):
    """
    Water reflectance model.

    The water is defined as a lambertian reflector placed just below the
    air-water interface, without any water column: the water body has
    neither geometric nor optical thickness.

    Parameters
    ----------
    alb : albedo object
        Albedo of the lambertian reflector, i.e. the water reflectance
        just below the surface. Although it is passed as an albedo (it
        is implemented as a lambertian reflector), the quantity to
        supply here is the subsurface irradiance reflectance R(0-) =
        Eu(0-)/Ed(0-). Do not supply a water-leaving reflectance such as
        rho_w or Rrs: those are defined above the interface, at the 0+
        level, and the air-water transmission would then be counted
        twice (see notes). Use an AlbedoSpectrum object to supply a
        spectrally varying R(0-).

    Notes
    -----
    This gives the reflectance at the 0- level, just below the
    interface, which is not the same as a lambertian surface at the 0+
    level, just above it: here the photons still cross the air-water
    interface, so the Fresnel transmission and the total internal
    reflection of the upwelling light are still accounted for by the
    `surf` parameter of `smartg.run()`.

    Note that, unlike the `alb` of Water1D, this reflector is not a sea
    floor: it stands for the water body itself, and it is placed at the
    top of the water column rather than at its bottom.

    Being lambertian, the reflector is isotropic, whereas the upwelling
    field of real water is not (Q = Eu/Lu differs from pi). The angular
    shape of the reflected light is therefore an approximation, even
    when the magnitude of R(0-) is exact.

    The same model can be obtained with an empty Water1D profile of null
    thickness::

        Water1D(grid=[0., 0.], comp=[], alb=alb)

    Both give the same optical thicknesses, single scattering albedo and
    seafloor albedo (pure water drops out on its own, since the layer
    has no thickness), but WaterRw is faster: it neither reads the pure
    water absorption auxiliary data nor computes any phase matrix.

    Examples
    --------
    >>> from smartg.water import WaterRw
    >>> from smartg.albedo import AlbedoCst
    >>> water = WaterRw(alb=AlbedoCst(0.05))
    """

    def __init__(self, alb: AlbedoLike) -> None:
        self.alb = alb

    def calc(self, wav: NumericArrayLike | BandSet) -> xr.Dataset:
        """
        Profile calculation at the given wavelengths.

        The water body has no thickness, so all the optical thicknesses
        are null and no phase matrix is computed. The only meaningful
        variable is 'albedo_seafloor', which carries the reflectance of
        the lambertian reflector.

        Parameters
        ----------
        wav : array_like or BandSet
            Wavelengths in nm at which to calculate the profile.

        Returns
        -------
        out : Dataset
            The profile, with the same variables as the one returned by
            `Water1D.calc` except the phase matrices, 'ssa_p_oc' and
            'ssa_w'. The `z_oc` coordinate holds two null levels.
        """
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)
        wav = np.array(wav)

        pro = xr.Dataset()
        pro = pro.assign_coords(wavelength=wav[:], z_oc=np.zeros(2))
        shp = (len(wav), 2)

        pro["T_oc"] = xr.DataArray(
            np.array([280.0, 280.0], dtype="float32"), dims=["z_oc"]
        )
        pro["OD_oc"] = xr.DataArray(
            np.zeros(shp, dtype="float32"), dims=["wavelength", "z_oc"]
        )
        pro["OD_w"] = xr.DataArray(
            np.zeros(shp, dtype="float32"), dims=["wavelength", "z_oc"]
        )
        pro["OD_p_oc"] = xr.DataArray(
            np.zeros(shp, dtype="float32"), dims=["wavelength", "z_oc"]
        )
        pro["OD_sca_oc"] = xr.DataArray(
            np.zeros(shp, dtype="float32"), dims=["wavelength", "z_oc"]
        )
        pro["OD_abs_oc"] = xr.DataArray(
            np.zeros(shp, dtype="float32"), dims=["wavelength", "z_oc"]
        )
        pro["OD_y"] = xr.DataArray(
            np.zeros(shp, dtype="float32"), dims=["wavelength", "z_oc"]
        )
        pro["pmol_oc"] = xr.DataArray(
            np.ones(shp, dtype="float32"), dims=["wavelength", "z_oc"]
        )
        pro["pine_oc"] = xr.DataArray(
            np.ones(shp, dtype="float32"), dims=["wavelength", "z_oc"]
        )
        pro["FQY1_oc"] = xr.DataArray(
            np.ones(shp, dtype="float32"), dims=["wavelength", "z_oc"]
        )
        pro["ssa_oc"] = xr.DataArray(
            np.ones(shp, dtype="float32"), dims=["wavelength", "z_oc"]
        )
        pro["albedo_seafloor"] = xr.DataArray(
            self.alb.get(wav), dims=["wavelength"]
        )

        return pro
