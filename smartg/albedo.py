"""Surface albedo models for SMART-G simulations.

This module provides classes defining the spectral albedo of the
surface bounding a SMART-G radiative transfer simulation. Several
albedo representations are available, ranging from idealised flat
spectra to user-supplied or library-read spectra, and including a 2D
horizontal map of spectral albedos for spatially heterogeneous
surfaces.

The albedo objects expose a common ``get(wl)`` interface returning the
spectral albedo evaluated at the requested wavelengths (nm). They are
passed as the ``alb`` parameter when building a surface in
``smartg.atmosphere``.

Key Classes
-----------
AlbedoCst
    Constant (wavelength-independent) albedo. A single scalar value is
    returned for every wavelength, producing a flat spectral albedo.
    Useful for idealised surfaces such as a white Lambertian ground.

AlbedoSpeclib
    Spectral albedo read from a JPL spectral library file (ASTER
    Spectral Library ASCII format). The wavelength axis is converted
    from micrometers to nanometres and the reflectance from percent to
    a dimensionless albedo in ``[0, 1]``.

AlbedoSpectrum
    Spectral albedo defined by an explicit user-supplied spectrum
    ``r(lambda)``, given as albedo values and corresponding
    wavelengths (nm).

AlbedoMap
    2D horizontal map of spectral albedos. A rectangular grid of
    cells, each referencing one entry from a list of albedo objects,
    allows spatially heterogeneous surfaces to be represented.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from luts.luts import LUT, Idx

from smartg.typing import NumericArrayLike, PathType


class AlbedoCst(object):
    """
    Constant (wavelength-independent) albedo.

    A single scalar albedo value is returned for every wavelength,
    producing a flat spectral albedo. Useful for idealised surfaces
    such as a white Lambertian ground.

    Parameters
    ----------
    alb : float
        Constant albedo value (dimensionless, in ``[0, 1]``).

    Attributes
    ----------
    alb : float
        The constant albedo value stored at construction time.
    """

    def __init__(self, alb: float) -> None:
        self.alb: float = alb

    def get(self, wl: NumericArrayLike) -> NDArray[np.float32]:
        """
        Return the spectral albedo at the requested wavelengths.

        Parameters
        ----------
        wl : array_like
            Wavelengths (nm) at which to evaluate the albedo. The
            shape is preserved in the output.

        Returns
        -------
        ndarray
            Albedo values (float32), same shape as ``wl``, filled
            with ``self.alb``.
        """
        alb = np.zeros(np.array(wl).shape, dtype=np.float32)
        alb[...] = self.alb
        return alb


class AlbedoSpeclib(object):
    """
    Spectral albedo read from a JPL spectral library file.

    The input file is expected to follow the ASCII format of the
    ASTER Spectral Library (http://speclib.jpl.nasa.gov/), with a
    26-line header followed by two columns: wavelength (micrometers)
    and reflectance (percent). The wavelength axis is converted from
    micrometers to nanometres and the reflectance from percent to a
    dimensionless albedo in ``[0, 1]``.

    Parameters
    ----------
    filename : str or path-like
        Path to the JPL speclib ASCII file.

    Attributes
    ----------
    data : LUT
        Look-up table of albedo values indexed by wavelength (nm).
    """

    def __init__(self, filename: PathType) -> None:
        data = np.genfromtxt(filename, skip_header=26)
        # convert X axis from micrometers to nm
        # convert Y axis from percent to dimensionless
        self.data: LUT = LUT(
            data[:, 1] / 100.0,
            axes=[data[:, 0] * 1000.0],
            names=["wavelength"],
        )

    def get(self, wl: NumericArrayLike) -> NDArray[np.floating]:
        """
        Return the spectral albedo at the requested wavelengths.

        Values are linearly interpolated from the library spectrum
        and extrapolated outside the covered range.

        Parameters
        ----------
        wl : array_like
            Wavelengths (nm) at which to evaluate the albedo.

        Returns
        -------
        ndarray
            Albedo values, same shape as ``wl``.
        """
        return cast(
            NDArray[np.floating], self.data[Idx(wl, fill_value="extrapolate")]
        )


class AlbedoSpectrum(object):
    """
    Spectral albedo defined by an explicit spectrum ``r(lambda)``.

    Parameters
    ----------
    r : array_like
        Spectral albedo values (dimensionless).
    lam : array_like
        Wavelengths (nm) at which ``r`` is sampled. Must be the same
        length as ``r``.

    Attributes
    ----------
    data : LUT
        Look-up table of albedo values indexed by wavelength (nm).
    """

    def __init__(self, r: NumericArrayLike, lam: NumericArrayLike) -> None:
        self.data: LUT = LUT(r, axes=[lam], names=["wavelength"])

    def get(self, wl: NumericArrayLike) -> NDArray[np.floating]:
        """
        Return the spectral albedo at the requested wavelengths.

        Values are linearly interpolated from the input spectrum and
        extrapolated outside the sampled range.

        Parameters
        ----------
        wl : array_like
            Wavelengths (nm) at which to evaluate the albedo.

        Returns
        -------
        ndarray
            Albedo values, same shape as ``wl``.
        """
        return cast(
            NDArray[np.floating], self.data[Idx(wl, fill_value="extrapolate")]
        )


class AlbedoMap(object):
    """
    2D horizontal map of spectral albedos.

    A rectangular 2D grid of spectral albedos can be constructed. Each
    cell of the grid references one entry from a list of ``Albedo``
    objects (``AlbedoCst``, ``AlbedoSpectrum`` or
    ``AlbedoSpeclib``). The number of distinct spectral albedos is
    limited to ``MAX_NREF = 10`` but could be extended.

    The horizontal grid is rectangular. The ``x`` and ``y`` boundaries
    on the surface (in km) are encoded in monotonic ``np.ndarray`` whose
    values are the upper limit of the rectangles: if ``x = [x0, x1, x2,
    ..., xn]`` then the limits are ``[-Inf, x0]``, ``[x0, x1]``, ...,
    ``[xn-1, xn]``, with ``xn`` large enough to be considered as
    ``+Inf`` (and similarly for ``y``).

    Each rectangle is assigned an index in ``alist`` through the 2D
    array ``ai`` of shape ``(len(x), len(y))``. Negative indices are
    reserved for surface properties.

    Parameters
    ----------
    ai : ndarray of int
        2D array of shape ``(len(x), len(y))`` giving, for each grid
        cell, the index of the corresponding albedo in ``alist``.
        Negative indices are reserved for surface properties.
    x : ndarray
        Monotonic array of upper ``x`` boundaries (km) of the grid
        cells.
    y : ndarray
        Monotonic array of upper ``y`` boundaries (km) of the grid
        cells.
    alist : list of Albedo objects
        List of ``AlbedoCst``, ``AlbedoSpectrum`` or
        ``AlbedoSpeclib`` instances, one per distinct spectral
        albedo.

    Attributes
    ----------
    map : LUT
        Look-up table of albedo indices indexed by ``X`` and ``Y``
        (km).
    list : list of Albedo objects
        The list of distinct spectral albedos.
    nalb : int
        Number of distinct spectral albedos (``len(alist)``).
    """

    def __init__(
        self,
        ai: NDArray[np.integer],
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        alist: list[AlbedoCst | AlbedoSpeclib | AlbedoSpectrum],
    ) -> None:
        self.map: LUT = LUT(ai, axes=[x, y], names=["X", "Y"])
        self.list: list[AlbedoCst | AlbedoSpeclib | AlbedoSpectrum] = alist
        self.nalb: int = len(alist)

    def get(self, wl: NumericArrayLike) -> NDArray[np.floating]:
        """
        Return the spectral albedo of every entry in the map.

        Parameters
        ----------
        wl : array_like
            Wavelengths (nm) at which to evaluate the albedos.

        Returns
        -------
        ndarray
            Array of shape ``(len(wl), nalb)`` holding the spectral
            albedo of each entry in ``self.list`` at the requested
            wavelengths.
        """
        return np.stack([alb.get(wl) for alb in self.list]).T

    def get_map(
        self, x0: NumericArrayLike, y0: NumericArrayLike
    ) -> NDArray[np.integer]:
        """
        Return the albedo index at the requested surface positions.

        Parameters
        ----------
        x0 : array_like
            ``x`` surface coordinates (km).
        y0 : array_like
            ``y`` surface coordinates (km).

        Returns
        -------
        ndarray
            Albedo index (int) from ``alist`` at each ``(x0, y0)``
            position, obtained by rounding to the nearest grid cell.
        """
        return np.asarray(
            self.map[
                Idx(x0, round=True, fill_value="extrema"),
                Idx(y0, round=True, fill_value="extrema"),
            ]
        ).astype(int)
