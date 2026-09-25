
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

from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import xarray as xr
from luts.luts import LUT
from numpy.typing import NDArray
from pytrunc.phase import fournier_forand

from smartg.albedo import AlbedoCst, AlbedoMap, SpectralAlbedoLike
from smartg.bandset import BandSet
from smartg.config import DIR_AUXDATA
from smartg.diff import diff1
from smartg.interp import interp_1d_coord
from smartg.phase import (
    as_theta_grid,
    calc_iphase,
    expand_phase_4_to_6,
    integ_phase,
)
from smartg.truncation import (
    DMTrunc,
    GTTrunc,
    as_truncation,
    truncate_phase_set,
)
from smartg.typing import NumericArrayLike, PathType

#: Recommended truncation of the Fournier-Forand phase functions the
#: hydrosols derive, to be asked for explicitly (the hydrosols truncate
#: nothing by default): the forward peak below 5 deg is replaced
#: following Iwabuchi & Suzuki (2009), with a truncation fraction of 0.3
#: (larger fractions make the truncated phase function negative for the
#: most forward-peaked Fournier-Forand mixtures).
DEFAULT_WATER_TRUNC = GTTrunc(trunc_frac=0.3, theta_tr=5.0)


def _refuse_albedo_map(alb: object) -> None:
    """Refuse an AlbedoMap as the albedo of a water profile.

    Raises
    ------
    TypeError
        If `alb` is an AlbedoMap, which gives one spectral albedo per
        entry of its map where the profile needs a single one.
    """
    if isinstance(alb, AlbedoMap):
        raise TypeError(
            "The alb of a water profile must be a spectral albedo "
            "(AlbedoCst, AlbedoSpeclib or AlbedoSpectrum): an AlbedoMap "
            "is only accepted as the alb of an Environment."
        )


def _interp_theta(
    values: NDArray, theta: NDArray, theta_new: NDArray
) -> NDArray:
    """
    Interpolate phase matrices linearly in angle.

    The kernel reads a phase matrix as linear in angle between the
    nodes of its grid, so that on a grid holding all the nodes of
    `theta` the interpolated matrices are the same functions.

    Parameters
    ----------
    values : ndarray
        Phase matrices, the angle being the last axis.
    theta : ndarray
        Increasing angles in degrees of `values`, from 0 to 180.
    theta_new : ndarray
        Angles in degrees to interpolate at, from 0 to 180.

    Returns
    -------
    ndarray
        The matrices at `theta_new`, same leading shape as `values`.
    """
    i = np.clip(
        np.searchsorted(theta, theta_new, side="right") - 1,
        0, len(theta) - 2,
    )
    t = (theta_new - theta[i]) / (theta[i + 1] - theta[i])
    return values[..., i] * (1.0 - t) + values[..., i + 1] * t


class IOPDict(TypedDict):
    """Inherent optical properties of a hydrosol.

    Returned by the `iop` and `coeffs` methods. All entries are
    coefficients in m-1 with dimensions [n_wavelength, nz], except
    `bbp_ratio`, which is dimensionless and may be None when no
    backscattering ratio is available.
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
    aw_pf = data_pf[:, 1] * 100  # convert from cm-1 to m-1
    wavelength_pf = data_pf[:, 0]
    ok_pf = wavelength_pf <= 725

    # Palmer&Williams
    data_pw = np.genfromtxt(
        Path(dir_aux) / "water" / "palmer74.dat", skip_header=5
    )
    aw_pw = data_pw[::-1, 1] * 100  # convert from cm-1 to m-1
    wavelength_pw = data_pw[::-1, 0]
    ok_pw = wavelength_pw > 725

    aw = xr.DataArray(
        np.array(list(aw_pf[ok_pf]) + list(aw_pw[ok_pw])),
        dims=["wavelength"],
        coords={
            "wavelength": np.array(
                list(wavelength_pf[ok_pf]) + list(wavelength_pw[ok_pw])
            )
        },
    )

    return aw


class Hydrosol:
    """
    User-defined hydrosol model.

    The inherent optical properties are supplied directly, either as
    scalars or as arrays over the wavelength and depth grids of the
    Water1D profile the hydrosol is added to.

    Parameters
    ----------
    phase : DataArray or LUT or None, optional
        Phase matrices with dimensions [n_wavelength, nz, nphamat,
        angle].
        If None, the phase matrices are derived from `bbp_ratio`
        (see notes).
    bp : array_like or None, optional
        Particle scattering coefficient in m-1, over the n_wavelength
        wavelengths and the nz levels of the profile. If None, it is
        taken as null. It may be a scalar, a depth profile of shape
        (nz,) or (1, nz), a spectrum of shape (n_wavelength, 1), or an
        array of shape (n_wavelength, nz). A 1-D array is a depth
        profile, and is refused as ambiguous when n_wavelength equals
        nz.
    ap : array_like or None, optional
        Particle absorption coefficient in m-1, same shape rules as
        `bp`.
    acdom : array_like or None, optional
        CDOM absorption coefficient in m-1, same shape rules as `bp`.
    bbp_ratio : array_like or None, optional
        Backscattering ratio (dimensionless), same shape rules as `bp`.
        Only used if `phase` is not provided.
    n_theta : int or array_like, optional
        Number of equally spaced angles of the derived phase matrices,
        or the angles themselves in degrees, which
        `smartg.phase.theta_grid` can build clustered towards the
        forward and backward directions.
    truncation : DMTrunc or GTTrunc or None, optional
        Truncation of the forward peak of the phase matrices, the ones
        supplied through `phase` as well as the derived ones, performed
        with `smartg.truncation.truncate_phase_set` as the `truncation`
        of the atmospheric components. None, the default, disables the
        truncation; `DEFAULT_WATER_TRUNC` is the one recommended for
        the derived phase functions. A phase function without a marked
        forward peak cannot be truncated: the truncation would leave it
        negative, and is refused.
    wavelength_phase : array_like or None, optional
        Wavelengths in nm at which the phase matrices are calculated. If
        None, they are calculated at all wavelengths. The coefficients
        supplied as arrays over the wavelengths of the profile are
        interpolated linearly onto them, and taken constant beyond the
        end wavelengths.

    Raises
    ------
    TypeError
        If `phase` is neither a DataArray, a LUT nor None, or if
        `truncation` is neither a DMTrunc, a GTTrunc nor None.

    Notes
    -----
    When `phase` is not provided, the phase matrices are derived from
    the backscattering ratio `bbp_ratio` following Park & Ruddick
    (2005), as a mixture of two Fournier-Forand phase functions. The
    angular grid resolves only part of their forward peak: `bp` is
    scaled by the resolved fraction, so that the particle
    backscattering coefficient is `bbp_ratio * bp` whatever `n_theta`
    (see `calc_phase`).

    Whether supplied or derived, their forward peak is truncated as
    configured by `truncation`, and the scattering coefficient `bp` is
    scaled by `1 - f`, with `f` the truncated fraction of the scattered
    energy of the phase matrix each wavelength and depth is given.

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
        truncation: DMTrunc | GTTrunc | None = None,
        wavelength_phase: NumericArrayLike | None = None,
    ) -> None:
        self.bp = bp
        self.ap = ap
        self.acdom = acdom
        self.bbp_ratio = bbp_ratio
        self._phase = expand_phase_4_to_6(phase)
        self.n_theta = n_theta
        self.truncation = as_truncation(truncation)
        self.wavelength_phase = (
            None if wavelength_phase is None
            else np.array(wavelength_phase)
        )

        self._pha: xr.DataArray | None = None
        self._coef_trunc: xr.DataArray | None = None
        self._bsca: NDArray | None = None
        # the wavelengths and depths the cache above was tabulated for,
        # and the wavelengths the supplied arrays were given over
        self._tab_grid: tuple[NDArray, NDArray, NDArray] | None = None

    def iop(self, wavelength: NDArray, z: NDArray) -> IOPDict:
        """
        Return the inherent optical properties of the hydrosol.

        They are given at the requested wavelengths and depths.

        The coefficients supplied at construction time are broadcast
        over the wavelength and depth grids; those left to None are
        taken as null.

        Parameters
        ----------
        wavelength : ndarray
            Wavelengths in nm.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.

        Returns
        -------
        IOPDict
            Inherent optical properties, each with dimensions
            [len(wavelength), len(z)]:

            - 'ap' : particle absorption coefficient in m-1
            - 'bp' : particle scattering coefficient in m-1, before the
              scaling by the factor of the phase matrices
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
            [len(wavelength), len(z)], or if it is a 1-D array while
            there are as many wavelengths as depths, which makes it
            ambiguous (see the class parameters).
        """
        n_wavelength, nz = len(wavelength), len(z)
        shp = (n_wavelength, nz)
        zeros = np.zeros(shp, dtype="float")

        def as_2d(x: NumericArrayLike | None) -> NDArray:
            """Broadcast `x` over the wavelength and depth grids."""
            if x is None:
                return zeros.copy()
            x = np.asarray(x, dtype="float")
            if x.ndim == 1 and x.size > 1 and x.size == nz == n_wavelength:
                raise ValueError(
                    f"A 1-D hydrosol coefficient of {x.size} values is "
                    f"ambiguous over {n_wavelength} wavelengths and {nz} "
                    f"depths: give a depth profile the shape (1, {nz}), "
                    f"and a spectrum the shape ({n_wavelength}, 1)."
                )
            try:
                return np.broadcast_to(x, shp).copy()
            except ValueError:
                raise ValueError(
                    "Cannot evaluate the hydrosol coefficients over "
                    f"{n_wavelength} wavelengths and {nz} depths: an "
                    f"array of shape {x.shape} is neither a depth "
                    f"profile of shape ({nz},) or (1, {nz}), a spectrum "
                    f"of shape ({n_wavelength}, 1), nor an array of "
                    f"shape ({n_wavelength}, {nz})."
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

    def _supplies_arrays(self) -> bool:
        """Whether a coefficient has been supplied as an array."""
        return any(
            x is not None and np.size(x) > 1
            for x in (self.bp, self.ap, self.acdom, self.bbp_ratio)
        )

    def _iop_on(
        self,
        wavelength_tab: NDArray,
        wavelength: NDArray,
        z: NDArray,
    ) -> IOPDict:
        """
        Return the inherent optical properties at other wavelengths.

        The coefficients supplied as arrays are given over the
        wavelengths of the profile: they are evaluated there, and
        interpolated linearly onto `wavelength_tab`, constant beyond
        the end wavelengths. The others are evaluated at
        `wavelength_tab` directly.

        Parameters
        ----------
        wavelength_tab : ndarray
            Wavelengths in nm at which the properties are returned,
            e.g. those the phase matrices are tabulated at.
        wavelength : ndarray
            Wavelengths in nm of the profile.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.

        Returns
        -------
        IOPDict
            Same as the `iop` method, with dimensions
            [len(wavelength_tab), len(z)].
        """
        wavelength_tab = np.asarray(wavelength_tab, dtype="float")
        wavelength = np.asarray(wavelength, dtype="float")
        if not self._supplies_arrays() or np.array_equal(
            wavelength_tab, wavelength
        ):
            return self.iop(wavelength_tab, z)

        iop = self.iop(wavelength, z)
        order = np.argsort(wavelength)

        def interp(x: NDArray) -> NDArray:
            """Interpolate each depth of `x` onto `wavelength_tab`."""
            return np.stack(
                [
                    np.interp(wavelength_tab, wavelength[order], col)
                    for col in x[order].T
                ],
                axis=1,
            )

        bbp_ratio = iop["bbp_ratio"]
        return {
            "ap": interp(iop["ap"]),
            "bp": interp(iop["bp"]),
            "acdom": interp(iop["acdom"]),
            "bbp_ratio": None if bbp_ratio is None else interp(bbp_ratio),
            "aphy": interp(iop["aphy"]),
            "fqyc": interp(iop["fqyc"]),
        }

    def calc_phase(
        self,
        wavelength: NDArray,
        z: NDArray,
        bbp_ratio: NDArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Calculate the phase matrices and their scattering factor.

        They are a mixture of two Fournier-Forand phase functions
        weighted by the backscattering ratio. Only the F11 (and F22 =
        F11) terms are non-null: the mixture is treated as a scalar
        phase function.

        The two functions are normalized analytically, to 2 over the
        cosine of the scattering angle, and diverge in the forward
        direction: the angular grid resolves only part of their forward
        peak, whose first bin is flattened. The part it misses is
        counted as unscattered, as a truncated peak is: the phase
        matrices are normalized to 2 over the grid, and the scattering
        coefficient is scaled by the resolved fraction, the integral of
        the mixture over the grid divided by 2. The particle
        backscattering coefficient is thus the backscattering ratio
        times the scattering coefficient, whatever `n_theta`.

        The normalized phase matrices are then truncated with pytrunc
        as configured by `truncation` (nothing is truncated when it is
        None).

        Parameters
        ----------
        wavelength : ndarray
            Wavelengths in nm.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.
        bbp_ratio : 2-D ndarray
            Backscattering ratio (dimensionless), dimensions
            [len(wavelength), len(z)].

        Returns
        -------
        pha_da : DataArray
            Phase matrices with dimensions [wavelength_phase,
            z_phase, nphamat, theta_oc].
        coef_trunc : DataArray
            Scattering factor with dimensions [wavelength_phase,
            z_phase], by which the scattering coefficient must be
            scaled: the resolved fraction of the mixture, times `1 - f`
            when `truncation` is set (`f` is the truncated fraction of
            the scattered energy of the normalized phase matrix). The
            resolved fraction tends to 1 as the grid is refined; it is
            below 1 up to a backscattering ratio of 0.03, and above
            beyond, where the mixture weights the second function
            negatively and is negative in the forward direction.

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
        n_wavelength = len(wavelength)
        nz = len(z)

        # particles phase function
        # see Park & Ruddick, 05
        # https://odnature.naturalsciences.be/downloads/publications/park_appliedoptics_2005.pdf
        # angle in radians
        ang = np.deg2rad(as_theta_grid(self.n_theta))
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
        inv = inv.reshape(n_wavelength, nz)

        f11 = r1_uniq[:, None] * ff1 + (1 - r1_uniq[:, None]) * ff2

        # the mixture integrates to 2 analytically: what the grid misses
        # of its forward peak is counted as unscattered, by scaling the
        # scattering coefficient by the resolved fraction, and the table
        # is normalized to 2 over the grid, as pytrunc and the kernel
        # expect
        resolved = integ_phase(ang, f11) / 2.0
        f11 /= resolved[:, None]

        coef = resolved
        if self.truncation is not None:
            # only F11 (and F22 = F11) is non-null, so it is truncated
            # alone, as a one-term matrix; the helper refuses a
            # negative result
            f11_tr, f = truncate_phase_set(
                f11[:, None, :], np.rad2deg(ang), self.truncation
            )
            f11 = f11_tr[:, 0, :]
            coef = resolved * (1.0 - f)

        pha = np.zeros((n_wavelength, nz, 6, len(ang)), dtype="float64")
        pha[:, :, 0, :] = f11[inv]
        pha[:, :, 4, :] = pha[:, :, 0, :]  # P22 = P11

        pha_da = xr.DataArray(
            pha,
            dims=["wavelength_phase", "z_phase", "nphamat", "theta_oc"],
            coords={
                "wavelength_phase": wavelength,
                "z_phase": z,
                "theta_oc": np.rad2deg(ang),
            },
        )
        coef_trunc = xr.DataArray(
            coef[inv],
            dims=["wavelength_phase", "z_phase"],
            coords={"wavelength_phase": wavelength, "z_phase": z},
        )

        return pha_da, coef_trunc

    def phase(
        self,
        wavelength: NDArray,
        z: NDArray,
        use_old_calc_iphase: bool = False,
    ) -> xr.DataArray | None:
        """
        Phase matrices of the hydrosol.

        The phase matrices supplied at construction time are returned
        truncated as configured by `truncation` (as such when it is
        None); otherwise they are derived from the backscattering ratio
        (see `calc_phase`). Both are memoized.

        Parameters
        ----------
        wavelength : ndarray
            Wavelengths in nm.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        DataArray or None
            Phase matrices with dimensions [wavelength_phase,
            z_phase, nphamat, theta_oc], or None if the hydrosol
            does not scatter.

        Raises
        ------
        Exception
            If the hydrosol scatters but neither the phase matrices nor
            the backscattering ratio have been provided.
        """
        if self._phase is not None:
            if self.truncation is None:
                return self._phase
            self._resolve_user_truncation()
            return self._pha

        iop = self.iop(wavelength, z)
        if not (np.asarray(iop["bp"]) > 0).any():
            return None
        if iop["bbp_ratio"] is None:
            raise ValueError(
                "No phase function nor bbp_ratio has been provided, but bp>0"
            )

        self._resolve_truncation(wavelength, z, use_old_calc_iphase)
        return self._pha

    def _resolve_truncation(
        self,
        wavelength: NDArray,
        z: NDArray,
        use_old_calc_iphase: bool = False,
    ) -> None:
        """
        Compute the phase matrices at the tabulation wavelengths.

        They are computed on `wavelength_phase`, along with the
        associated scattering factor (see `calc_phase`).

        The result is memoized in `_pha`, `_coef_trunc` and `_bsca`, so
        that the scattering coefficient and the phase matrices stay
        consistent whichever is requested first. Returns immediately if
        the cache already holds the tabulation of these wavelengths and
        depths, and computes it again otherwise, e.g. for a hydrosol
        reused at other wavelengths or on another grid. A single depth
        is tabulated when neither the backscattering ratio nor the
        scattering coefficient varies vertically.

        Parameters
        ----------
        wavelength : ndarray
            Wavelengths in nm of the profile. The phase matrices are
            tabulated at these unless `wavelength_phase` is set, and
            the coefficients supplied as arrays are given over them.
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
        wavelength_pha = np.asarray(
            wavelength if self.wavelength_phase is None
            else self.wavelength_phase,
            dtype="float",
        )
        z = np.asarray(z, dtype="float")
        # the supplied arrays are interpolated from the wavelengths of
        # the profile, which the tabulation then depends on
        wavelength_src = np.asarray(
            wavelength if self._supplies_arrays() else wavelength_pha,
            dtype="float",
        )
        if (
            self._coef_trunc is not None
            and self._tab_grid is not None
            and np.array_equal(self._tab_grid[0], wavelength_pha)
            and np.array_equal(self._tab_grid[1], z)
            and np.array_equal(self._tab_grid[2], wavelength_src)
        ):
            return
        self._tab_grid = (
            wavelength_pha.copy(), z.copy(), wavelength_src.copy()
        )

        iop = self._iop_on(wavelength_pha, wavelength, z)
        bbp_ratio, bp = iop["bbp_ratio"], iop["bp"]
        if bbp_ratio is None:
            raise ValueError(
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
            wavelength_pha, z[sl], bbp_ratio[:, sl]
        )
        self._bsca = bp[:, sl] * self._coef_trunc.values

    def _resolve_user_truncation(self) -> None:
        """
        Truncate the phase matrices supplied at construction time.

        Each distinct matrix is truncated once, and the result memoized
        in `_pha` and `_coef_trunc`, the truncation factor `1 - f` over
        [wavelength_phase, z_phase]. The supplied matrices do not depend
        on the grids of the profile, so they are truncated once for all.
        """
        if self._coef_trunc is not None:
            return
        assert self._phase is not None and self.truncation is not None
        pha_tr, f = truncate_phase_set(
            self._phase.values,
            self._phase.coords[self._phase.dims[-1]].values,
            self.truncation,
        )
        self._pha = self._phase.copy(data=pha_tr)
        self._coef_trunc = xr.DataArray(
            1.0 - f,
            dims=["wavelength_phase", "z_phase"],
            coords={
                dim: self._phase.coords[dim].values
                for dim in ["wavelength_phase", "z_phase"]
            },
        )

    def _coef_trunc_on(
        self,
        wavelength: NDArray,
        z: NDArray,
        use_old_calc_iphase: bool = False,
    ) -> NDArray:
        """
        Map the scattering factor of the phase matrices onto the grids.

        It is mapped from the tabulation grid of the phase matrices
        onto the given wavelength and depth grids.

        Must be called after `_resolve_truncation` or
        `_resolve_user_truncation` has filled the cache.

        Parameters
        ----------
        wavelength : ndarray
            Wavelengths in nm.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        ndarray
            Scattering factor with dimensions [len(wavelength), len(z)].
        """
        # only called once _resolve_truncation has filled the cache
        assert (self._pha is not None) and (self._coef_trunc is not None)

        # index with ipha, so that each wavelength/depth gets the factor
        # of the phase matrix it is actually assigned to
        _, ipha = calc_iphase(
            self._pha, np.asarray(wavelength), np.asarray(z),
            use_old_calc_iphase
        )
        return self._coef_trunc.values.ravel()[ipha]

    def scattering(self, pha: xr.DataArray) -> NDArray | None:
        """
        Return the scattering coefficient of the hydrosol, in m-1.

        It is given on the tabulation grid of the phase matrices.

        Parameters
        ----------
        pha : DataArray
            Phase matrices of this hydrosol, whose
            `wavelength_phase` and `z_phase` coordinates define the
            grid of the output.

        Returns
        -------
        ndarray or None
            Scattering coefficient in m-1 with dimensions
            [wavelength_phase, z_phase], scaled by the factor of the
            phase matrices. None if the derived phase matrices have not
            been calculated yet.
        """
        if self._phase is None:
            return self._bsca
        bp = self.iop(
            pha.coords["wavelength_phase"].values, pha.coords["z_phase"].values
        )["bp"]
        if self.truncation is None:
            return bp
        self._resolve_user_truncation()
        assert self._coef_trunc is not None
        return bp * self._coef_trunc.values

    def _scattering_on(
        self,
        wavelength_tab: NDArray,
        wavelength: NDArray,
        z: NDArray,
        use_old_calc_iphase: bool = False,
    ) -> NDArray:
        """
        Return the scattering coefficient at other wavelengths, in m-1.

        Used to weight the hydrosols when averaging their phase matrices
        in `Water1D.phase`. Must be called after `phase`.

        Parameters
        ----------
        wavelength_tab : ndarray
            Wavelengths in nm at which the coefficient is returned.
        wavelength : ndarray
            Wavelengths in nm of the profile.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        ndarray
            Scattering coefficient in m-1 with dimensions
            [len(wavelength_tab), len(z)], scaled by the factor of the
            phase matrix each wavelength and depth is given.
        """
        bp = self._iop_on(wavelength_tab, wavelength, z)["bp"]
        if self._phase is not None and self.truncation is None:
            return bp
        return bp * self._coef_trunc_on(
            wavelength_tab, z, use_old_calc_iphase
        )

    def coeffs(
        self,
        wavelength: NDArray,
        z: NDArray,
        phase: bool = True,
        use_old_calc_iphase: bool = False,
    ) -> IOPDict:
        """
        Return the inherent optical properties of the hydrosol.

        The scattering coefficient is scaled by the factor of the phase
        matrices, see `calc_phase`: the truncation factor, and, for the
        derived phase matrices, the fraction of the forward peak their
        grid resolves.

        Parameters
        ----------
        wavelength : ndarray
            Wavelengths in nm.
        z : ndarray
            Vertical grid of the water column in m. These are z
            coordinates: 0 at the surface, negative downwards.
        phase : bool, optional
            Whether the phase matrices are calculated. If False, the
            scattering coefficient is not scaled.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        IOPDict
            Same as the `iop` method, with 'bp' scaled by the factor of
            the phase matrices.

        Raises
        ------
        Exception
            If the hydrosol scatters but neither the phase matrices nor
            the backscattering ratio have been provided.
        """
        iop = self.iop(wavelength, z)
        if not (np.asarray(iop["bp"]) > 0).any():
            return iop

        if self._phase is None:
            if iop["bbp_ratio"] is None:
                raise ValueError(
                    "No phase function nor bbp_ratio has been "
                    "provided, but bp>0"
                )
            if not phase:
                return iop
            self._resolve_truncation(wavelength, z, use_old_calc_iphase)
        else:
            if not phase or self.truncation is None:
                return iop
            self._resolve_user_truncation()

        coef_trunc = self._coef_trunc_on(wavelength, z, use_old_calc_iphase)
        iop["bp"] = iop["bp"] * coef_trunc
        return iop


class HydrosolPR(Hydrosol):
    """
    Chlorophyll-driven hydrosol model of the Polymer PR type.

    It uses an inherent optical property parameterization similar to
    the PR model of Polymer.

    The absorption, scattering and backscattering ratio are all derived
    from a single chlorophyll concentration, which does not vary with
    depth.

    Parameters
    ----------
    chl : float
        Chlorophyll concentration in mg/m3.
    n_theta : int or array_like, optional
        Number of equally spaced angles of the derived phase matrices,
        or the angles themselves in degrees, which
        `smartg.phase.theta_grid` can build clustered towards the
        forward and backward directions.
    truncation : DMTrunc or GTTrunc or None, optional
        Truncation of the forward peak of the derived phase matrices
        (see `Hydrosol`). None, the default, disables the truncation;
        `DEFAULT_WATER_TRUNC` is the recommended one.
    wavelength_phase : array_like or None, optional
        Wavelengths in nm at which the phase matrices are calculated. If
        None, they are calculated at all wavelengths.
    fqyc : float, optional
        Chlorophyll a fluorescence quantum yield.

    Notes
    -----
    The phytoplankton absorption follows Bricaud et al. (1998), the CDM
    absorption Bricaud et al. (2012), and the particle scattering
    coefficient Loisel & Morel (1998), 0.416 chl^0.766 (550 /
    wavelength) in m-1. As in the base class, the phase matrices are
    derived from the backscattering ratio, and the scattering
    coefficient is scaled by their factor only (see
    `Hydrosol.calc_phase`).

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
        truncation: DMTrunc | GTTrunc | None = None,
        wavelength_phase: NumericArrayLike | None = None,
        fqyc: float = 0.0,
    ) -> None:
        super().__init__(
            n_theta=n_theta, truncation=truncation,
            wavelength_phase=wavelength_phase,
        )
        self.chl = chl
        self.fqyc = fqyc

        # Bricaud (98)
        ap_bricaud = np.genfromtxt(
            DIR_AUXDATA / "water" / "aph_bricaud_1998.txt",
            delimiter=",",
            skip_header=12,
        )  # header is lambda,Ap,Ep,Aphi,Ephi
        self.bricaud = xr.Dataset()
        self.bricaud = self.bricaud.assign_coords(wavelength=ap_bricaud[:, 0])
        self.bricaud["A"] = xr.DataArray(ap_bricaud[:, 1], dims=["wavelength"])
        self.bricaud["E"] = xr.DataArray(
            1 - ap_bricaud[:, 2], dims=["wavelength"]
        )

    def iop(self, wavelength: NDArray, z: NDArray) -> IOPDict:
        """
        Return the optical properties of the chlorophyll.

        They are derived from the chlorophyll concentration.

        The chlorophyll concentration does not vary with depth, so the
        coefficients are computed spectrally and then broadcast over the
        depth profile.

        Parameters
        ----------
        wavelength : ndarray
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
        wavelength = np.asarray(wavelength, dtype="float")
        chl = self.chl

        # phytoplankton absorption
        aphy = interp_1d_coord(
            self.bricaud["A"], "wavelength", wavelength, extrema=True
        ) * chl ** interp_1d_coord(
            self.bricaud["E"], "wavelength", wavelength, extrema=True
        )

        # chlorophyll fluorescence (scattering coefficient)
        fqyc = np.full_like(
            aphy, self.fqyc
        )  # Fluorescence Quantum Yield for Chlorophyll
        fqyc[wavelength < 370.0] = 0.0
        fqyc[wavelength > 690.0] = 0.0

        # CDM absorption central value
        # from Bricaud et al GBC, 2012 (data from nov 2007)
        fa = 1.0
        acdm443 = fa * 0.069 * (chl**1.070)

        s_cdom = 0.00262 * (acdm443 ** (-0.448))
        s_cdom = min(s_cdom, 0.025)
        s_cdom = max(s_cdom, 0.011)

        acdm = acdm443 * np.exp(-s_cdom * (wavelength - 443))

        bp = 0.416 * (chl**0.766) * 550.0 / wavelength

        #
        # backscattering coefficient
        #
        if chl < 2:
            v = 0.5 * (np.log10(chl) - 0.3)
        else:
            v = 0
        bbp_ratio = 0.002 + 0.01 * (0.5 - 0.25 * np.log10(chl)) * (
            (wavelength / 550.0) ** v
        )

        shp = (len(wavelength), len(z))

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
    Chlorophyll-driven hydrosol model of Zhai et al. (2017).

    The chlorophyll concentration varies with depth.

    Parameters
    ----------
    chl_surf : float
        Chlorophyll concentration in mg/m3 at the surface. The
        concentration at depth is derived from it (see notes), so this
        is the surface value only, unlike the depth-independent `chl` of
        HydrosolPR.
    n_theta : int or array_like, optional
        Number of equally spaced angles of the derived phase matrices,
        or the angles themselves in degrees, which
        `smartg.phase.theta_grid` can build clustered towards the
        forward and backward directions.
    truncation : DMTrunc or GTTrunc or None, optional
        Truncation of the forward peak of the derived phase matrices
        (see `Hydrosol`). None, the default, disables the truncation;
        `DEFAULT_WATER_TRUNC` is the recommended one.
    wavelength_phase : array_like or None, optional
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

    The particle scattering is covariant with the phytoplankton
    absorption, for a particle single scattering albedo of 0.68 at
    440 nm. As in the base class, the phase matrices are derived from
    the backscattering ratio, and the scattering coefficient is scaled
    by their factor only (see `Hydrosol.calc_phase`).

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
        truncation: DMTrunc | GTTrunc | None = None,
        wavelength_phase: NumericArrayLike | None = None,
        euphotic_depth: float | None = None,
        mixed: bool = False,
        fqyc: float = 0.0,
    ) -> None:
        super().__init__(
            n_theta=n_theta, truncation=truncation,
            wavelength_phase=wavelength_phase,
        )
        self.chl_surf = chl_surf
        self.fqyc = fqyc

        # Bricaud (98)
        # Absorption of the phytoplankton
        ap_bricaud = np.genfromtxt(
            DIR_AUXDATA / "water" / "aph_bricaud_1998.txt",
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
            wavelength=np.concatenate((w_uv, ap_bricaud[:, 0]))
        )
        self.bricaud["A"] = xr.DataArray(
            np.concatenate((a_uv, a_bricaud)), dims=["wavelength"]
        )
        self.bricaud["E"] = xr.DataArray(
            np.concatenate((e_uv, e_bricaud)), dims=["wavelength"]
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

    def iop(
        self,
        wavelength: NDArray,
        z: NDArray,
        p1: float = 0.33,
        r1: float = 0.5,
        r2: float = 0.5,
    ) -> IOPDict:
        """
        Return the optical properties of the chlorophyll profile.

        They are derived from the chlorophyll profile.

        The chlorophyll concentration is evaluated at each z coordinate
        (see `chl`), and the absorption, scattering and backscattering
        are made covariant with it.

        Parameters
        ----------
        wavelength : ndarray
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
        wavelength = np.asarray(wavelength, dtype="float")
        chl2, wavelength_2 = np.meshgrid(self.chl(z), wavelength)

        # specific phytoplankton absorption
        chl2star = np.full_like(chl2, 1.0)
        aphystar = interp_1d_coord(
            self.bricaud["A"], "wavelength", wavelength_2, extrema=True
        ) * (
            chl2star
            ** interp_1d_coord(
                self.bricaud["E"], "wavelength", wavelength_2, extrema=True
            )
        )
        aphy = aphystar * chl2
        aphystar440 = interp_1d_coord(
            self.bricaud["A"], "wavelength", 440.0, extrema=True
        ) * (
            chl2star
            ** interp_1d_coord(
                self.bricaud["E"], "wavelength", wavelength_2, extrema=True
            )
        )
        aphy440 = aphystar440 * chl2

        # phytoplankton covariant particles extinction
        piz440 = 0.68
        bp440 = aphy440 * piz440 / (1 - piz440)
        bp = bp440 * (wavelength_2 / 440.0) ** (-1.0)

        # chlorophyll fluorescence (scattering coefficient)
        fqyc = np.full_like(
            aphy, self.fqyc
        )  # Fluorescence Quantum Yield for Chlorophyll
        fqyc[wavelength_2 < 370.0] = 0.0
        fqyc[wavelength_2 > 690.0] = 0.0

        # CDOM covariant absorption
        acdm440 = 0.24 * aphy440**0.43
        s_cdom = 0.02
        acdom = acdm440 * np.exp(-s_cdom * (wavelength_2 - 440))

        # non-algal particles backscattering, none here (spm = 0)
        spm = 0.0  # g/m3
        bbp_ratio_nap = np.full_like(aphy, 0.04)
        if spm > 0.0:
            gamma = 0.5
            bbpnap650 = 10 ** (
                1.03 * np.log10(spm) - 2.06
            )  # Neukermans et al 2012
            bbpnap = bbpnap650 * (wavelength_2 / 650.0) ** (-gamma)
            bp += bbpnap / bbp_ratio_nap

        return {
            "ap": aphy,
            "bp": bp,
            "acdom": acdom,
            "bbp_ratio": bbp_ratio_nap,
            "aphy": aphy,
            "fqyc": fqyc,
        }


class Water:
    """Base class for water."""

    def calc(self, wavelength: NumericArrayLike | BandSet, *args: Any,
             **kwargs: Any) -> xr.Dataset:
        """
        Compute the water column profile as an xr.Dataset.

        Implemented by the subclasses.
        """
        raise NotImplementedError


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
        or/and HydrosolZhai objects. The phase matrices of several
        scattering hydrosols are averaged, see `phase`.
    aw : None or 2-D ndarray, optional
        Force the pure water absorption coefficient in m-1, with
        dimensions [n_wavelength, nz]. If None, it is read from the
        auxiliary data (see `_read_aw`).
    bw : None or 2-D ndarray, optional
        Force the pure water scattering coefficient in m-1, with
        dimensions [n_wavelength, nz]. If None, it is computed as
        19.3e-4*(wavelength/550)**-4.3.
    alb : AlbedoCst or AlbedoSpeclib or AlbedoSpectrum, optional
        Albedo of the sea floor, i.e. of the reflector placed at the
        bottom of the water column, at the deepest level of `grid`. This
        is the reflectance of the sea bottom seen from within the water,
        and it is not related to the albedo of the air-water interface:
        the latter is set by the `surface` parameter of `smartg.run()`
        (e.g. `smartg.surface.LambSurface(alb=...)` or
        `RoughSurface(...)`). It fills
        the `albedo_seafloor` variable of the profile returned by
        `calc()`. If None, a black (non-reflecting) sea floor is used,
        i.e. `AlbedoCst(0.)`.

    Raises
    ------
    TypeError
        If `alb` is an AlbedoMap, which only an Environment accepts.

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
        alb: SpectralAlbedoLike | None = None,
    ) -> None:
        _refuse_albedo_map(alb)
        self.grid = np.array(grid, dtype="float")
        self.comp = [] if comp is None else comp
        self.aw = aw
        self.bw = bw
        self.alb = AlbedoCst(0.0) if alb is None else alb

        self.aw_table = _read_aw(DIR_AUXDATA)

    def calc(
        self,
        wavelength: NumericArrayLike | BandSet,
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
        wavelength : array_like or BandSet
            Wavelengths in nm at which to calculate the profile.
        phase : bool, optional
            Whether to calculate the phase matrices. If False, the
            scattering coefficients are not scaled by the factor of the
            phase matrices either (see `Hydrosol.coeffs`).
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
        if not isinstance(wavelength, BandSet):
            wavelength = BandSet(wavelength)
        wavelength = np.array(wavelength)

        z = self.grid
        shp = (len(wavelength), len(z))
        wavelength_2 = np.stack([wavelength] * len(z), axis=1)

        #
        # pure water absorption and scattering
        #
        if self.aw is None:
            aw = interp_1d_coord(self.aw_table, "wavelength", wavelength_2)
        else:
            aw = self.aw

        if self.bw is None:
            bw = 19.3e-4 * ((wavelength_2 / 550.0) ** -4.3)
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
                wavelength, z, phase=phase,
                use_old_calc_iphase=use_old_calc_iphase
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
        pro = pro.assign_coords(wavelength=wavelength[:], z_oc=z)

        pro["T_oc"] = xr.DataArray(
            np.array([280.0] * len(z), dtype="float32"), dims=["z_oc"]
        )

        #
        # phase matrices
        #
        if phase:
            pha = self.phase(
                wavelength, use_old_calc_iphase=use_old_calc_iphase
            )

            if pha is not None:
                pha_, ipha = calc_iphase(
                    pha,
                    pro.coords["wavelength"].values,
                    pro.coords["z_oc"].values,
                    use_old_calc_iphase,
                )

                pro = pro.assign_coords(theta_oc=pha.coords["theta_oc"].values)
                pro["phase_oc"] = xr.DataArray(
                    pha_, dims=["iphase", "nphamat", "theta_oc"]
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
                "description": "Cumulated water optical thickness "
                               "at each wavelength"
            },
        )

        pro["OD_p_oc"] = xr.DataArray(
            np.cumsum(tau_p, out=tau_p, axis=1),
            dims=["wavelength", "z_oc"],
            attrs={
                "description": "Cumulated oceanic particles optical "
                               "thickness at each wavelength"
            },
        )

        pro["OD_y"] = xr.DataArray(
            np.cumsum(tau_y, out=tau_y, axis=1),
            dims=["wavelength", "z_oc"],
            attrs={
                "description": "Cumulated CDOM optical thickness at "
                               "each wavelength"
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
            self.alb.get(wavelength), dims=["wavelength"]
        )

        return pro

    def phase(
        self,
        wavelength: NDArray,
        use_old_calc_iphase: bool = False,
    ) -> xr.DataArray | None:
        """
        Calculate the phase matrices averaged over the hydrosols.

        They are weighted by the scattering coefficient of each
        hydrosol, scaled by the factor of its phase matrices.

        When a single hydrosol scatters, its phase matrices are returned
        unchanged. Otherwise they are averaged on a common grid: the
        `wavelength_phase` of the hydrosols if they share it, the
        wavelengths of the profile otherwise; their `z_phase` if they
        share it and no weight varies with depth, the depths of `grid`
        otherwise; their angles if they share them, all of their angles
        otherwise, each phase matrix being interpolated linearly in
        angle, as the kernel reads it.

        Parameters
        ----------
        wavelength : ndarray
            Wavelengths in nm.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        out : DataArray or None
            The phase matrices with dimensions [wavelength_phase,
            z_phase, nphamat, theta_oc], or None if no hydrosol
            scatters.
        """
        z = self.grid

        phases = []
        for comp in self.comp:
            pha = comp.phase(
                wavelength, z, use_old_calc_iphase=use_old_calc_iphase
            )
            if pha is not None:
                phases.append((comp, pha))

        if len(phases) == 0:
            return None
        if len(phases) == 1:
            return phases[0][1]

        def shared(dim: str) -> NDArray | None:
            """Return the `dim` coordinate if all phases share it."""
            ref = phases[0][1].coords[dim].values
            for _, pha in phases[1:]:
                if not np.array_equal(pha.coords[dim].values, ref):
                    return None
            return ref

        # the common grid of the average, see the docstring
        wavelength = np.asarray(wavelength, dtype="float")
        wavelength_c = shared("wavelength_phase")
        if wavelength_c is None:
            wavelength_c = wavelength
        weights = [
            comp._scattering_on(
                wavelength_c, wavelength, z, use_old_calc_iphase
            )
            for comp, _ in phases
        ]
        z_c = shared("z_phase")
        if z_c is not None and not np.array_equal(z_c, z):
            if all(np.allclose(w, w[:, :1]) for w in weights):
                weights = [
                    np.repeat(w[:, :1], len(z_c), axis=1) for w in weights
                ]
            else:
                z_c = None
        if z_c is None:
            z_c = z
        theta_c = shared("theta_oc")
        if theta_c is None:
            theta_c = np.unique(
                np.concatenate(
                    [pha.coords["theta_oc"].values for _, pha in phases]
                )
            )

        pha_tot: NDArray | float = 0.0
        bsca: NDArray | float = 0.0
        for (_, pha), weight in zip(phases, weights, strict=True):
            if np.array_equal(
                pha.coords["wavelength_phase"].values, wavelength_c
            ) and np.array_equal(pha.coords["z_phase"].values, z_c):
                values = pha.values
            else:
                # the phase matrix each wavelength and depth of the
                # common grid is given, as calc_iphase gives the profile
                flat, ipha = calc_iphase(
                    pha, wavelength_c, z_c, use_old_calc_iphase
                )
                values = flat[ipha]
            theta = pha.coords["theta_oc"].values
            if not np.array_equal(theta, theta_c):
                values = _interp_theta(values, theta, theta_c)
            bsca = bsca + weight
            pha_tot = pha_tot + values * weight[:, :, None, None]

        with np.errstate(divide="ignore", invalid="ignore"):
            pha_tot = pha_tot / np.asarray(bsca)[:, :, None, None]
        return xr.DataArray(
            np.where(np.isnan(pha_tot), 0.0, pha_tot),
            dims=["wavelength_phase", "z_phase", "nphamat", "theta_oc"],
            coords={
                "wavelength_phase": wavelength_c,
                "z_phase": z_c,
                "theta_oc": theta_c,
            },
        )


class WaterRw(Water):
    """
    Water reflectance model.

    The water is defined as a lambertian reflector placed just below the
    air-water interface, without any water column: the water body has
    neither geometric nor optical thickness.

    Parameters
    ----------
    alb : AlbedoCst or AlbedoSpeclib or AlbedoSpectrum
        Albedo of the lambertian reflector, i.e. the water reflectance
        just below the surface. Although it is passed as an albedo (it
        is implemented as a lambertian reflector), the quantity to
        supply here is the subsurface irradiance reflectance R(0-) =
        Eu(0-)/Ed(0-). Do not supply a water-leaving reflectance such as
        rho_w or Rrs: those are defined above the interface, at the 0+
        level, and the air-water transmission would then be counted
        twice (see notes). Use an AlbedoSpectrum object to supply a
        spectrally varying R(0-).

    Raises
    ------
    TypeError
        If `alb` is an AlbedoMap, which only an Environment accepts.

    Notes
    -----
    This gives the reflectance at the 0- level, just below the
    interface, which is not the same as a lambertian surface at the 0+
    level, just above it: here the photons still cross the air-water
    interface, so the Fresnel transmission and the total internal
    reflection of the upwelling light are still accounted for by the
    `surface` parameter of `smartg.run()`.

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

    def __init__(self, alb: SpectralAlbedoLike) -> None:
        _refuse_albedo_map(alb)
        self.alb = alb

    def calc(self, wavelength: NumericArrayLike | BandSet) -> xr.Dataset:
        """
        Profile calculation at the given wavelengths.

        The water body has no thickness, so all the optical thicknesses
        are null and no phase matrix is computed. The only meaningful
        variable is 'albedo_seafloor', which carries the reflectance of
        the lambertian reflector.

        Parameters
        ----------
        wavelength : array_like or BandSet
            Wavelengths in nm at which to calculate the profile.

        Returns
        -------
        out : Dataset
            The profile, with the same variables as the one returned by
            `Water1D.calc` except the phase matrices, 'ssa_p_oc' and
            'ssa_w'. The `z_oc` coordinate holds two null levels.
        """
        if not isinstance(wavelength, BandSet):
            wavelength = BandSet(wavelength)
        wavelength = np.array(wavelength)

        pro = xr.Dataset()
        pro = pro.assign_coords(wavelength=wavelength[:], z_oc=np.zeros(2))
        shp = (len(wavelength), 2)

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
            self.alb.get(wavelength), dims=["wavelength"]
        )

        return pro
