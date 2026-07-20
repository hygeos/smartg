"""Ocean vibrational Raman scattering spectrum.

This module provides the spectral response functions used to
model inelastic (vibrational Raman) scattering by liquid water in
ocean-color radiative-transfer simulations.

The Raman shift of liquid water covers the O-H stretching band,
roughly 2950-3850 cm-1. The spectral shape is modeled as a sum of
four Gaussian peaks fitted to laboratory measurements. Two helpers
are exposed:

* :func:`raman_forward`     -- forward spectrum: from excitation
  wavelength(s) to the Raman-shifted wavelength grid and response.
* :func:`raman_inverse` -- inverse spectrum: from a detected
  wavelength back to the excitation wavelength grid and response.

The internal helpers :func:`gaussian_peak` and :func:`raman_response`
build the underlying Gaussian peaks and their normalized sum.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from smartg.typing import NumericArrayLike


def gaussian_peak(
    ks: NumericArrayLike,
    aj: float,
    kj: float,
    dkj: float,
) -> NDArray[np.floating]:
    """Evaluate a single Gaussian Raman peak.

    Parameters
    ----------
    ks : array_like
        Wavenumber(s) at which the Gaussian is evaluated (cm-1).
    aj : float
        Peak amplitude of the Gaussian (dimensionless weight).
    kj : float
        Center wavenumber of the Gaussian (cm-1).
    dkj : float
        Full width at half maximum of the Gaussian (cm-1).

    Returns
    -------
    ndarray
        Gaussian values evaluated at ``ks``, with peak value
        ``aj / dkj`` at ``ks == kj``.
    """
    ks = np.atleast_1d(np.asarray(ks, dtype=np.float64))
    return aj * 1.0 / dkj * np.exp(-4 * np.log(2) * (ks - kj) ** 2 / dkj**2)


def raman_response(ks: NumericArrayLike) -> NDArray[np.floating]:
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
    a = np.array([0.41, 0.39, 0.10, 0.10])
    k = np.array([3250.0, 3425.0, 3530.0, 3625.0])
    dk = np.array([210.0, 175.0, 140.0, 140.0])
    norm = np.sum(a) * np.sqrt(np.pi / 4 / np.log(2))
    norm = 1.0 / norm
    su = np.zeros_like(ks)
    for j in range(4):
        su += gaussian_peak(ks, a[j], k[j], dk[j])

    return su * norm


def raman_forward(
    lam: NumericArrayLike,
    nl: int = 16,
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
    nl : int, optional
        Number of points in the returned Raman-shifted wavelength grid.
        Default is 16.

    Returns
    -------
    wgrid : ndarray
        Raman-shifted wavelength grid in nm, shape ``(nl, N)``.
    response : ndarray
        Spectral response evaluated on ``wgrid``, shape ``(nl, N)``.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    k = 1e7 / lam  # cm-1
    k0 = k - 2950.0  # cm-1
    k1 = k - 3850.0  # cm-1
    w0 = 1e7 / k0
    w1 = 1e7 / k1
    wgrid = np.linspace(w0, w1, num=nl, dtype=np.float64)
    ks = 1e7 * (1.0 / lam[np.newaxis, :] - 1.0 / wgrid)
    response = 1e7 / wgrid**2 * raman_response(ks)
    return wgrid, response


def raman_inverse(
    lam: NumericArrayLike,
    nl: int = 16,
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
    nl : int, optional
        Number of points in the returned excitation wavelength grid.
        Default is 16.

    Returns
    -------
    wgrid : ndarray
        Excitation wavelength grid in nm, shape ``(N, nl)``.
    response : ndarray
        Spectral response evaluated on ``wgrid``, shape ``(N, nl)``.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    k = 1e7 / lam  # cm-1
    k0 = k + 3850.0  # cm-1
    k1 = k + 2950.0  # cm-1
    w0 = 1e7 / k0
    w1 = 1e7 / k1
    wgrid = np.linspace(w0, w1, num=nl, dtype=np.float64).T
    ks = 1e7 * (1.0 / wgrid - 1.0 / lam[:, np.newaxis])
    response = 1e7 / wgrid**2 * raman_response(ks)

    return wgrid, response
