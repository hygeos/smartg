"""Spectral band definition and spectral grid construction for SMART-G.

This module provides tools to define the wavelength bands used in
SMART-G radiative transfer simulations and to build the spectral grids
required for absorption and scattering computations.

Key components
--------------
BandSet
    Common object for formatting input band definitions. Accepts a
    scalar float, a 1-D array of wavelengths, or a list of KDIS/RepTran
    ``IBandS`` objects (detected automatically via the ``BandLike``
    protocol). When built from ``IBandS`` objects, molecular absorption
    profiles are computed by delegating to each band's
    ``calc_profile`` method.

spectral_grids
    Build the high-resolution wavelength grid for absorption features,
    the low-resolution grid for scattering features, the Raman
    excitation grid, and a solar irradiance look-up table. Precomputed
    interpolation parameters map the high-resolution grid onto the
    low-resolution grid by 1-D linear interpolation.

Key Classes
-----------
BandSet
    Container for the spectral bands of a simulation; accepted as
    the wavelength parameter of Smartg.run.

Key Functions
-------------
spectral_grids
    Build spectral grids for absorption and scattering
    computations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import numpy as np
import scipy.constants as cst
from luts.luts import LUT
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from smartg.rrs import l2d_inv
from smartg.typing import BandLike, NumericArrayLike, RealNumber
from smartg.vrs import raman_inverse

if TYPE_CHECKING:
    # Imported only for type checking to avoid circular imports
    # (atmosphere.py imports from bandset.py at runtime).
    from smartg.atmosphere import ProfileBase


class BandSet(object):
    def __init__(self, wavelength: NumericArrayLike | list[BandLike]) -> None:
        """Initialize a BandSet from wavelength band definitions.

        Common object for formatting input band definitions. Accepts a
        scalar float, a 1-D array of wavelengths, or a list of KDIS or
        RepTran ``IBandS`` objects (detected automatically via the
        ``calc_profile`` attribute of the first element).

        The wavelengths are stored internally as a ``float32`` NumPy
        array. When the input is a scalar, it is reshaped to a 1-element
        array and ``scalar`` is set to ``True``.

        Parameters
        ----------
        wavelength : float, list, ndarray, or list of IBandS
            Wavelength band definition. A float or 1-D array of
            wavelengths (in nm), or a list of KDIS/RepTran ``IBandS``
            objects whose ``w`` attribute gives the band wavelength.

        Notes
        -----
        When ``wavelength`` is a list of ``IBandS`` objects,
        ``use_reptran_kdis`` is set to ``True``, ``data`` holds the
        original objects, and ``calc_profile`` delegates to each band's
        own ``calc_profile`` method. Otherwise ``data`` is ``None`` and
        ``calc_profile`` returns zeros.
        """
        # Detect KDIS/RepTran band objects via the BandLike protocol.
        # Scalar inputs (int/float) have no __getitem__ and fall back
        # to the except branch.
        try:
            first = (
                wavelength[0]
                if isinstance(wavelength, (list, np.ndarray))
                else wavelength
            )
            self.use_reptran_kdis: bool = isinstance(first, BandLike)
        except Exception:
            self.use_reptran_kdis = False

        self.type_wavelength: type | None = None
        if self.use_reptran_kdis:
            bands = cast(list[BandLike], wavelength)
            wavelength_vals: list[float] | NumericArrayLike = [
                x.w for x in bands
            ]
            self.data: list[BandLike] | None = bands
            self.type_wavelength = type(bands[0])
        else:
            wavelength_vals = cast(NumericArrayLike, wavelength)
            self.data = None
            self.type_wavelength = None

        assert isinstance(wavelength_vals, (float, list, np.ndarray))
        self.wavelength: NDArray[np.float32] = np.array(
            wavelength_vals, dtype="float32"
        )
        self.scalar: bool = self.wavelength.ndim == 0
        if self.scalar:
            self.wavelength = self.wavelength.reshape(1)
        self.size: int = int(self.wavelength.size)

    @overload
    def __getitem__(self, key: int) -> np.floating: ...
    @overload
    def __getitem__(
        self, key: slice | NDArray[np.integer]
    ) -> NDArray[np.float32]: ...
    def __getitem__(
        self, key: int | slice | NDArray[np.integer]
    ) -> NDArray[np.float32] | np.floating:
        """Return the wavelength(s) at the given index or slice.

        Parameters
        ----------
        key : int, slice, or array-like
            Index, slice, or fancy index into the internal wavelength
            array.

        Returns
        -------
        ndarray or float
            The wavelength value(s) selected from the internal
            ``float32`` wavelength array.
        """
        return self.wavelength[key]

    def __len__(self) -> int:
        """Return the number of bands in the set.

        Returns
        -------
        int
            Number of wavelength bands (``self.size``).
        """
        return self.size

    def calc_profile(self, prof: ProfileBase) -> NDArray[np.float32]:
        """Compute the molecular absorption profile for each band.

        For each band, calculate the absorption optical depth profile
        over the altitude grid of ``prof``. When the BandSet was built
        from KDIS/RepTran ``IBandS`` objects
        (``use_reptran_kdis is True``), each band's own
        ``calc_profile`` method is invoked. Otherwise an array of zeros
        is returned (no molecular absorption handled at the band level).

        Parameters
        ----------
        prof : Profile
            Atmospheric profile providing the altitude grid ``prof.z``
            (in km) on which the optical depth is evaluated.

        Returns
        -------
        tau_mol : ndarray
            Molecular absorption optical depth with shape
            ``(size, len(prof.z))`` and ``float32`` dtype.
        """
        tau_mol = np.zeros((self.size, len(prof.z)), dtype="float32")

        if self.use_reptran_kdis:
            assert self.data is not None
            for i, w in enumerate(self.data):
                tau_mol[i, :] = w.calc_profile(prof)

        return tau_mol


def spectral_grids(
    lmin: RealNumber,
    lmax: RealNumber,
    datas: NDArray[np.floating],
    dl: RealNumber | None = None,
    dls: RealNumber | None = None,
    raman: str = "RRS",
    unit: str = "mW/m2/nm",
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    LUT,
    NDArray[np.int8],
    NDArray[np.float32],
]:
    """Build spectral grids for absorption and scattering computations.

    Construct the high-resolution wavelength grid used for absorption
    features (``wavelength``), the low-resolution grid used for
    scattering features (``wavelengths``), and the Raman excitation
    grid (``wavelength_rs``). A solar irradiance LUT (``es_lut``) is
    built over the union of the scattering and Raman-shifted ranges,
    and precomputed interpolation parameters (``i_wavelengths_in``,
    ``w_wavelengths_in``) map each high-resolution wavelength onto the
    low-resolution grid by 1-D linear interpolation.

    When ``dl`` is ``None``, the high-resolution grid is taken directly
    from the solar spectrum samples falling within ``[lmin, lmax]``;
    otherwise it is a uniform grid from ``lmin`` to ``lmax`` with step
    ``dl``. When ``dls`` is ``None``, the low-resolution grid is
    collapsed to a single point (``nws = 1``); otherwise it is a
    uniform grid with step ``dls`` whose size is forced to be odd.

    Parameters
    ----------
    lmin : float
        Minimum wavelength of the spectral range (nm).
    lmax : float
        Maximum wavelength of the spectral range (nm).
    datas : ndarray
        Solar spectrum data with shape ``(N, 2)``: column 0 holds the
        wavelengths (nm) and column 1 the irradiance values.
    dl : float, optional
        High spectral resolution step for absorption features (nm).
        If ``None`` (default), the solar spectrum sampling within
        ``[lmin, lmax]`` is used as the high-resolution grid.
    dls : float, optional
        Low spectral resolution step for scattering features (nm).
        If ``None`` (default), the low-resolution grid is reduced to a
        single point (``nws = 1``).
    raman : {'RRS', 'VRS'}, optional
        Raman scattering type: ``'RRS'`` (rotational Raman, default)
        uses a 90 deg scattering angle and 243 K temperature;
        ``'VRS'`` (vibrational Raman) uses ``raman_inverse``.
    unit : {'mW/m2/nm', 'photons/cm2/s/nm'}, optional
        Unit of the solar irradiance in ``datas``. If
        ``'photons/cm2/s/nm'``, the values are converted from
        ``mW/m2/nm`` to photon flux. Default is ``'mW/m2/nm'``.

    Returns
    -------
    wavelength : ndarray
        High-resolution wavelength grid (nm) for absorption features.
    wavelengths : ndarray
        Low-resolution wavelength grid (nm) for scattering features
        (single point if ``dls`` is ``None``).
    wavelength_rs : ndarray
        Raman excitation wavelength grid (nm) corresponding to
        ``wavelength``.
    es_lut : LUT
        Solar irradiance look-up table over the union of
        ``wavelengths`` and ``wavelength_rs`` ranges, indexed by
        wavelength.
    i_wavelengths_in : ndarray of int8
        Index of the lower ``wavelengths`` value used to linearly
        interpolate each ``wavelength`` onto the low-resolution grid.
    w_wavelengths_in : ndarray of float32
        Floating-point weight (in ``[0, 1]``) between
        ``i_wavelengths_in`` and ``i_wavelengths_in + 1`` for the
        linear interpolation of ``wavelength`` in ``wavelengths``.
    """
    # Solar spectrum input data
    wavelength_0 = datas[:, 0]
    e0 = datas[:, 1]
    # Convert from mW/m2/nm to photons/cm2/s/nm
    if unit == "photons/cm2/s/nm":
        e0 *= 1e-3 * 1e-4 / (cst.h * cst.c) * (wavelength_0 * 1e-9)

    # High spectral resolution grid (for absorption features)
    if dl is None:
        # Solar grid
        ii = np.where((wavelength_0 >= lmin) & (wavelength_0 <= lmax))
        wavelength = wavelength_0[ii]
        nw = wavelength.size
        # Solar resolution
        dl = (wavelength[-1] - wavelength[0]) / wavelength.size
    else:
        nw = int((lmax - lmin) / dl) + 1
        wavelength = np.linspace(lmin, lmax, num=nw)  # wavelength grid

    if raman == "RRS":
        # RRS excitation wavelength grid for a scattering angle of
        # 90 deg and a temperature of 243 K.
        wavelength_rs, _ = l2d_inv(wavelength, 90.0, 243.0)
    else:
        # VRS excitation wavelength grid
        wavelength_rs, _ = raman_inverse(wavelength)

    lmin_rs = min(wavelength_rs.min(), wavelength.min())
    lmax_rs = max(wavelength_rs.max(), wavelength.max())

    # Solar spectrum LUT building
    ii = np.where((wavelength_0 >= lmin_rs) & (wavelength_0 <= lmax_rs))
    es_lut = LUT(
        e0[ii], axes=[wavelength_0[ii]], names=["wavelength"], desc="Es"
    )
    # Low spectral resolution for scattering computations (step in nm)
    if dls is not None:
        nws = int((lmax_rs - lmin_rs) / dls)
        # Ensure nws is odd
        nws = nws if (nws & 1) else nws + 1
    else:
        nws = 1
    wavelengths = np.linspace(lmin_rs, lmax_rs, num=nws)

    # Parameters for the 1-D linear interpolation of wavelength in
    # wavelengths
    if nws > 1:
        f = interp1d(wavelengths, np.linspace(0, nws - 1, num=nws))
        iw = f(wavelength)
        # Index of the lower wavelengths value in the wavelengths array
        i_wavelengths_in = np.floor(iw).astype(np.int8)
        # Floating-point proportion between i_wavelengths_in and
        # i_wavelengths_in + 1
        w_wavelengths_in = (iw - i_wavelengths_in).astype(np.float32)
        # Special case for the upper boundary
        ii = np.where(i_wavelengths_in == (nws - 1))
        i_wavelengths_in[ii] = nws - 2
        w_wavelengths_in[ii] = 1.0
    else:
        i_wavelengths_in = np.array([0], dtype=np.int8)
        w_wavelengths_in = np.array([0], dtype=np.float32)

    return (
        wavelength, wavelengths, wavelength_rs, es_lut,
        i_wavelengths_in, w_wavelengths_in,
    )
