"""Ocean vibrational Raman scattering spectrum.

This module provides the spectral response functions used to
model inelastic (vibrational Raman) scattering by liquid water in
ocean-color radiative-transfer simulations.

The Raman shift of liquid water covers the O-H stretching band,
roughly 2950-3850 cm-1. The spectral shape is modeled as a sum of
four Gaussian peaks fitted to laboratory measurements. Two helpers
are exposed:

* :func:`V2d`     -- forward spectrum: from excitation
  wavelength(s) to the Raman-shifted wavelength grid and response.
* :func:`V2d_inv` -- inverse spectrum: from a detected
  wavelength back to the excitation wavelength grid and response.

The internal helpers :func:`Gauss` and :func:`fR` build the underlying
Gaussian peaks and their normalized sum.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from smartg.typing import NumericArrayLike


def Gauss(
    ks: NumericArrayLike, Aj: float, kj: float, Dkj: float,
) -> NDArray[np.floating]:
    """Evaluate a single Gaussian Raman peak.

    Parameters
    ----------
    ks : array_like
        Wavenumber(s) at which the Gaussian is evaluated (cm-1).
    Aj : float
        Peak amplitude of the Gaussian (dimensionless weight).
    kj : float
        Center wavenumber of the Gaussian (cm-1).
    Dkj : float
        Full width at half maximum of the Gaussian (cm-1).

    Returns
    -------
    ndarray
        Gaussian values evaluated at ``ks``, with peak value
        ``Aj / Dkj`` at ``ks == kj``.
    """
    ks = np.atleast_1d(np.asarray(ks, dtype=np.float64))
    return Aj * 1./Dkj * np.exp(-4*np.log(2)*(ks-kj)**2/Dkj**2)


def fR(ks: NumericArrayLike) -> NDArray[np.floating]:
    """Normalized Raman spectral response of liquid water.

    Builds the O-H stretching band as a sum of four Gaussian peaks
    centered at 3250, 3425, 3530 and 3625 cm-1, with relative
    amplitudes 0.41, 0.39, 0.10 and 0.10 and FWHM 210, 175, 140 and
    140 cm-1. The result is normalized so that its integral over
    wavenumber equals one.

    Parameters
    ----------
    ks : array_like
        Wavenumber(s) at which the response is evaluated (cm-1).

    Returns
    -------
    ndarray
        Normalized Raman spectral response evaluated at ``ks``.
    """
    A  = np.array([0.41, 0.39, 0.10, 0.10])
    k  = np.array([3250., 3425., 3530., 3625.])
    Dk = np.array([210., 175., 140., 140.])
    norm = np.sum(A) * np.sqrt(np.pi/4/np.log(2))
    norm = 1./norm
    Su=np.zeros_like(ks)
    for j in range(4):
        Su+= Gauss(ks, A[j], k[j], Dk[j])

    return Su*norm


def V2d(
    lam: NumericArrayLike, Nl: int = 16,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Ocean vibrational Raman spectrum (forward).

    Given one or more excitation wavelengths, returns the Raman-shifted
    wavelength grid covering the O-H stretching band (shifts between
    2950 and 3850 cm-1) together with the corresponding spectral
    response.

    Parameters
    ----------
    lam : array_like
        Excitation wavelength(s) in nm. Shape ``(N,)``.
    Nl : int, optional
        Number of points in the returned Raman-shifted wavelength grid.
        Default is 16.

    Returns
    -------
    wgrid : ndarray
        Raman-shifted wavelength grid in nm, shape ``(Nl, N)``.
    response : ndarray
        Spectral response evaluated on ``wgrid``, shape ``(Nl, N)``.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    k   = 1e7/lam   # cm-1
    k0  = k - 2950. # cm-1
    k1  = k - 3850. # cm-1
    w0  = 1e7/k0
    w1  = 1e7/k1
    wgrid = np.linspace(w0, w1, num=Nl, dtype=np.float64)
    ks    = 1e7*(1./lam[np.newaxis,:]-1./wgrid)
    response = 1e7/wgrid**2 * fR(ks)
    return wgrid, response


def V2d_inv(
    lam: NumericArrayLike, Nl: int = 16,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Ocean vibrational Raman spectrum (inverse).

    Given one or more detected wavelengths, returns the excitation
    wavelength grid that would Raman-shift to those wavelengths,
    covering the O-H stretching band (shifts between 2950 and 3850
    cm-1), together with the corresponding spectral response.

    Parameters
    ----------
    lam : array_like
        Detected (Raman-shifted) wavelength(s) in nm. Shape ``(N,)``.
    Nl : int, optional
        Number of points in the returned excitation wavelength grid.
        Default is 16.

    Returns
    -------
    wgrid : ndarray
        Excitation wavelength grid in nm, shape ``(N, Nl)``.
    response : ndarray
        Spectral response evaluated on ``wgrid``, shape ``(N, Nl)``.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    k   = 1e7/lam # cm-1
    k0  = k + 3850. # cm-1
    k1  = k + 2950. # cm-1
    w0  = 1e7/k0
    w1  = 1e7/k1
    wgrid = np.linspace(w0, w1, num=Nl, dtype=np.float64).T
    ks    = 1e7*(1./wgrid - 1./lam[:,np.newaxis])
    response = 1e7/wgrid**2 * fR(ks)

    return wgrid, response
