"""Rotational Raman scattering (Ring effect) spectrum utilities.

This module builds the rotational Raman line list of dry air (N2
+ O2) and assembles the inelastic Ring spectrum that must be
folded into a radiative-transfer calculation to reproduce the
Ring effect observed in satellite backscatter ultraviolet
measurements.

The two main building blocks are:

* Bates (1984) Rayleigh-scattering depolarization for N2 and O2,
  used to compute Kattawar et al. (1981) Cabannes fractions
  :func:`f0_air`, :func:`f0_n2`, :func:`f0_o2`.
* Joiner et al. (1995) rotational Raman line strengths built from
  Boltzmann-weighted rigid-rotor populations and Placzek-Teller
  coefficients, exposed through :func:`l_air`, :func:`l2d` and
  :func:`l2d_inv`.

References
----------
.. [1] Bates, D. R. (1984). Rayleigh scattering by air.
   *Planetary and Space Science*, 32(6), 785-790.
   https://doi.org/10.1016/0032-0633(84)90102-8
.. [2] Kattawar, G. W., Young, A. T., & Humphreys, T. J. (1981).
   Inelastic scattering in planetary atmospheres. I. The Ring
   effect, without aerosols. *Astrophysical Journal, Part 1*,
   243, 1049-1057.
.. [3] Joiner, J., Bhartia, P. K., Cebula, R. P., Hilsenrath,
   E., McPeters, R. D., & Park, H. (1995). Rotational Raman
   scattering (Ring effect) in satellite backscatter ultraviolet
   measurements. *Applied Optics*, 34(21), 4513.
   https://doi.org/10.1364/AO.34.004513
"""

import scipy.constants as cst
import numpy as np

# Atmosphere model: dry-air molar mixing ratios (mol/mol).
X_N2 = 0.788
X_O2 = 0.212


# Bates, Planel. Space Sa., Vol.32, No.6, pp. 785-790. 1984
def fk_n2(lam):
    """King correction factor of N2 as a function of wavelength.

    Parameters
    ----------
    lam : float or array_like
        Wavelength in nanometers.

    Returns
    -------
    fk : float or ndarray
        Dimensionless King correction factor of N2. Same shape as
        ``lam``.

    References
    ----------
    .. [1] Bates, D. R. (1984). Rayleigh scattering by air.
       *Planetary and Space Science*, 32(6), 785-790.
       https://doi.org/10.1016/0032-0633(84)90102-8
    """
    return 1.034 + 3.17 * 1e-4 / ((lam * 1e-3) ** 2)


def epsilon_n2(lam):
    r"""Depolarization ratio of N2 as a function of wavelength.

    Computed from the King correction factor as
    :math:`\varepsilon_{N_2} = (F_K - 1) \times 4.5`, where the
    constant 4.5 follows Bates (1984).

    Parameters
    ----------
    lam : float or array_like
        Wavelength in nanometers.

    Returns
    -------
    eps : float or ndarray
        Dimensionless depolarization ratio of N2. Same shape as
        ``lam``.

    References
    ----------
    .. [1] Bates, D. R. (1984). Rayleigh scattering by air.
       *Planetary and Space Science*, 32(6), 785-790.
       https://doi.org/10.1016/0032-0633(84)90102-8
    """
    return (fk_n2(lam) - 1) * 4.5


def fk_o2(lam):
    """King correction factor of O2 as a function of wavelength.

    Parameters
    ----------
    lam : float or array_like
        Wavelength in nanometers.

    Returns
    -------
    fk : float or ndarray
        Dimensionless King correction factor of O2. Same shape as
        ``lam``.

    References
    ----------
    .. [1] Bates, D. R. (1984). Rayleigh scattering by air.
       *Planetary and Space Science*, 32(6), 785-790.
       https://doi.org/10.1016/0032-0633(84)90102-8
    """
    return (
        1.096
        + 1.385 * 1e-3 / ((lam * 1e-3) ** 2)
        + 1.448 * 1e-4 / ((lam * 1e-3) ** 4)
    )


def epsilon_o2(lam):
    r"""Depolarization ratio of O2 as a function of wavelength.

    Computed from the King correction factor as
    :math:`\varepsilon_{O_2} = (F_K - 1) \times 4.5`, where the
    constant 4.5 follows Bates (1984).

    Parameters
    ----------
    lam : float or array_like
        Wavelength in nanometers.

    Returns
    -------
    eps : float or ndarray
        Dimensionless depolarization ratio of O2. Same shape as
        ``lam``.

    References
    ----------
    .. [1] Bates, D. R. (1984). Rayleigh scattering by air.
       *Planetary and Space Science*, 32(6), 785-790.
       https://doi.org/10.1016/0032-0633(84)90102-8
    """
    return (fk_o2(lam) - 1) * 4.5


def epsilon_air(lam):
    r"""Effective depolarization ratio of dry air.

    Weighted sum of the N2 and O2 depolarization ratios using the
    standard dry-air mixing ratios :attr:`X_N2` and :attr:`X_O2`:
    :math:`\varepsilon_{\text{air}} = \varepsilon_{N_2} X_{N_2}
    + \varepsilon_{O_2} X_{O_2}`.

    Parameters
    ----------
    lam : float or array_like
        Wavelength in nanometers.

    Returns
    -------
    eps : float or ndarray
        Dimensionless effective depolarization ratio of dry air.
        Same shape as ``lam``.

    References
    ----------
    .. [1] Bates, D. R. (1984). Rayleigh scattering by air.
       *Planetary and Space Science*, 32(6), 785-790.
       https://doi.org/10.1016/0032-0633(84)90102-8
    """
    return epsilon_n2(lam) * X_N2 + epsilon_o2(lam) * X_O2


# Kattawar, Astrophysical Journal, Part 1, vol. 243, Feb. 1, 1981, p. 1049-1057.
def f0_air(lam, theta):
    r"""Cabannes fraction of dry air (Kattawar's ``f0``).

    Fraction of Rayleigh-scattered photons that are depolarized,
    i.e. the probability that the scattered photon retains the
    polarization memory. Computed from the dry-air depolarization
    ratio and the scattering angle using the analytical expression
    given by Kattawar et al. (1981).

    Parameters
    ----------
    lam : float or array_like
        Wavelength in nanometers.
    theta : float or array_like
        Scattering angle in degrees (0 = forward, 180 = backward).

    Returns
    -------
    f0 : float or ndarray
        Dimensionless Cabannes fraction. Same shape as ``lam`` and
        ``theta`` (broadcast).

    References
    ----------
    .. [1] Kattawar, G. W., Young, A. T., & Humphreys, T. J. (1981).
       Inelastic scattering in planetary atmospheres. I. The Ring
       effect, without aerosols. *Astrophysical Journal, Part 1*,
       243, 1049-1057.
    """
    eps = epsilon_air(lam)
    c2 = np.cos(np.radians(theta)) ** 2
    num = (180.0 + 13.0 * eps) + (180.0 + eps) * c2
    den = (180.0 + 52.0 * eps) + (180.0 + 4.0 * eps) * c2
    return num / den


def f0_n2(lam, theta):
    r"""Cabannes fraction of N2 (Kattawar's ``f0`` for pure N2).

    Parameters
    ----------
    lam : float or array_like
        Wavelength in nanometers.
    theta : float or array_like
        Scattering angle in degrees (0 = forward, 180 = backward).

    Returns
    -------
    f0 : float or ndarray
        Dimensionless Cabannes fraction of N2. Same shape as ``lam``
        and ``theta`` (broadcast).

    References
    ----------
    .. [1] Kattawar, G. W., Young, A. T., & Humphreys, T. J. (1981).
       Inelastic scattering in planetary atmospheres. I. The Ring
       effect, without aerosols. *Astrophysical Journal, Part 1*,
       243, 1049-1057.
    """
    eps = epsilon_n2(lam)
    c2 = np.cos(np.radians(theta)) ** 2
    num = (180.0 + 13.0 * eps) + (180.0 + eps) * c2
    den = (180.0 + 52.0 * eps) + (180.0 + 4.0 * eps) * c2
    return num / den


def f0_o2(lam, theta):
    r"""Cabannes fraction of O2 (Kattawar's ``f0`` for pure O2).

    Parameters
    ----------
    lam : float or array_like
        Wavelength in nanometers.
    theta : float or array_like
        Scattering angle in degrees (0 = forward, 180 = backward).

    Returns
    -------
    f0 : float or ndarray
        Dimensionless Cabannes fraction of O2. Same shape as ``lam``
        and ``theta`` (broadcast).

    References
    ----------
    .. [1] Kattawar, G. W., Young, A. T., & Humphreys, T. J. (1981).
       Inelastic scattering in planetary atmospheres. I. The Ring
       effect, without aerosols. *Astrophysical Journal, Part 1*,
       243, 1049-1057.
    """
    eps = epsilon_o2(lam)
    c2 = np.cos(np.radians(theta)) ** 2
    num = (180.0 + 13.0 * eps) + (180.0 + eps) * c2
    den = (180.0 + 52.0 * eps) + (180.0 + 4.0 * eps) * c2
    return num / den


# Joiner, J., Bhartia, P. K., Cebula, R. P., Hilsenrath, E., McPeters, R. D., & Park, H. (1995). Rotational Raman scattering (Ring effect)
# in satellite backscatter ultraviolet measurements. Applied Optics, 34(21), 4513. doi:10.1364/ao.34.004513
## !!!! Erreur dans le papier original sur les coeffs de Placzek-Teller Anti Stokes !!!


def k_ratio(lam, theta):
    r"""Joiner's O2-to-N2 Cabannes ratio.

    Ratio :math:`K(\lambda, \theta) = (1 - f_0^{O_2}) /
    (1 - f_0^{N_2})` used to weight the O2 rotational Raman
    contribution relative to N2 in the Ring-effect spectrum.

    Parameters
    ----------
    lam : float or array_like
        Wavelength in nanometers.
    theta : float or array_like
        Scattering angle in degrees (0 = forward, 180 = backward).

    Returns
    -------
    k : float or ndarray
        Dimensionless ratio. Same shape as ``lam`` and ``theta``
        (broadcast).

    References
    ----------
    .. [1] Joiner, J., Bhartia, P. K., Cebula, R. P., Hilsenrath,
       E., McPeters, R. D., & Park, H. (1995). Rotational Raman
       scattering (Ring effect) in satellite backscatter ultraviolet
       measurements. *Applied Optics*, 34(21), 4513.
       https://doi.org/10.1364/AO.34.004513
    """
    return (1.0 - f0_o2(lam, theta)) / (1.0 - f0_n2(lam, theta))


def bjm_plus(J):
    r"""Placzek-Teller coefficient for the Stokes branch (:math:`\Delta J = +2`).

    :math:`b_{J}^{+} = \frac{3 (J+1)(J+2)}{2 (2J+1)(2J+3)}`.

    Parameters
    ----------
    J : int or array_like of int
        Rotational quantum number(s).

    Returns
    -------
    b : float or ndarray
        Dimensionless coefficient. Same shape as ``J``.

    References
    ----------
    .. [1] Joiner, J., Bhartia, P. K., Cebula, R. P., Hilsenrath,
       E., McPeters, R. D., & Park, H. (1995). Rotational Raman
       scattering (Ring effect) in satellite backscatter ultraviolet
       measurements. *Applied Optics*, 34(21), 4513.
       https://doi.org/10.1364/AO.34.004513
    """
    return 3.0 * (J + 1) * (J + 2) / 2.0 / (2 * J + 1) / (2 * J + 3)


def bjm_minus(J):
    r"""Placzek-Teller coefficient for the anti-Stokes branch (:math:`\Delta J = -2`).

    :math:`b_{J}^{-} = \frac{3 J (J-1)}{2 (2J+1)(2J-1)}`, set to 0
    for :math:`J \le 1` (no physical transition).

    Parameters
    ----------
    J : int or array_like of int
        Rotational quantum number(s).

    Returns
    -------
    b : float or ndarray
        Dimensionless coefficient. Same shape as ``J``.

    References
    ----------
    .. [1] Joiner, J., Bhartia, P. K., Cebula, R. P., Hilsenrath,
       E., McPeters, R. D., & Park, H. (1995). Rotational Raman
       scattering (Ring effect) in satellite backscatter ultraviolet
       measurements. *Applied Optics*, 34(21), 4513.
       https://doi.org/10.1364/AO.34.004513
    """
    b = 3.0 * J * (J - 1) / 2.0 / (2 * J + 1) / (2 * J - 1)
    b[J <= 1] = 0.0
    return b


def l_o2(T):
    r"""O2 rotational Raman line list.

    Builds the rotational Raman spectrum of O2 from the rigid-rotor
    energy levels with rotational quantum number :math:`J \in
    [0, 36]`. For each transition, the Boltzmann weight is combined
    with the nuclear-spin degeneracy (``gj = 1`` for odd J, ``0``
    for even J, since :sup:`16`O has zero nuclear spin) and the
    Placzek-Teller coefficient. Line shifts :math:`\Delta\nu` are
    returned in cm:sup:`-1` and line strengths are normalised so
    that the total Stokes + anti-Stokes weight sums to one.

    Parameters
    ----------
    T : float
        Gas temperature in Kelvin.

    Returns
    -------
    dnu_stk : ndarray of float
        Frequency shift of each Stokes transition in cm-1 (negative).
    lj_stk : ndarray of float
        Normalised Stokes line strengths.
    dnu_astk : ndarray of float
        Frequency shift of each anti-Stokes transition in cm-1
        (positive).
    lj_astk : ndarray of float
        Normalised anti-Stokes line strengths.

    References
    ----------
    .. [1] Joiner, J., Bhartia, P. K., Cebula, R. P., Hilsenrath,
       E., McPeters, R. D., & Park, H. (1995). Rotational Raman
       scattering (Ring effect) in satellite backscatter ultraviolet
       measurements. *Applied Optics*, 34(21), 4513.
       https://doi.org/10.1364/AO.34.004513
    """
    B0 = 1.4378  # cm-1
    J = np.linspace(0, 36, num=37, dtype=np.int32)
    # 1 if J is odd, 0 if even. Faster than j % 2 != 0 (but only int!)
    gj = J & 1
    # B0 translated in m-1!!!
    Ej = J * (J + 1) * cst.h * cst.c * B0 * 100
    Fj = gj * (2 * J + 1) * np.exp(-Ej / (cst.k * T))

    lj_stk = Fj * bjm_plus(J)
    dnu_stk = -(4 * J + 6) * B0
    is_nonzero = lj_stk != 0.0
    lj_stk = lj_stk[is_nonzero]
    dnu_stk = dnu_stk[is_nonzero]
    lj_astk = Fj * bjm_minus(J)
    is_nonzero = lj_astk != 0.0
    dnu_astk = (4 * J - 2) * B0
    lj_astk = lj_astk[is_nonzero]
    dnu_astk = dnu_astk[is_nonzero]

    norm = lj_stk.sum() + lj_astk.sum()
    return dnu_stk, lj_stk / norm, dnu_astk, lj_astk / norm


def l_n2(T):
    r"""N2 rotational Raman line list.

    Builds the rotational Raman spectrum of N2 from the rigid-rotor
    energy levels with rotational quantum number :math:`J \in
    [0, 36]`. For each transition, the Boltzmann weight is combined
    with the nuclear-spin degeneracy (``gj = 6`` for even J, ``3``
    for odd J, the standard homonuclear diatomic convention) and
    the Placzek-Teller coefficient. Line shifts :math:`\Delta\nu`
    are returned in cm:sup:`-1` and line strengths are normalised so
    that the total Stokes + anti-Stokes weight sums to one.

    Parameters
    ----------
    T : float
        Gas temperature in Kelvin.

    Returns
    -------
    dnu_stk : ndarray of float
        Frequency shift of each Stokes transition in cm-1 (negative).
    lj_stk : ndarray of float
        Normalised Stokes line strengths.
    dnu_astk : ndarray of float
        Frequency shift of each anti-Stokes transition in cm-1
        (positive).
    lj_astk : ndarray of float
        Normalised anti-Stokes line strengths.

    References
    ----------
    .. [1] Joiner, J., Bhartia, P. K., Cebula, R. P., Hilsenrath,
       E., McPeters, R. D., & Park, H. (1995). Rotational Raman
       scattering (Ring effect) in satellite backscatter ultraviolet
       measurements. *Applied Optics*, 34(21), 4513.
       https://doi.org/10.1364/AO.34.004513
    """
    B0 = 1.9897  # cm-1
    J = np.linspace(0, 36, num=37, dtype=np.int32)
    # Bitwise AND with 1 selects odd J -> gj takes the odd branch value
    gj = np.where(J & 1, 3, 6)
    # B0 translated in m-1 !!!
    Ej = J * (J + 1) * cst.h * cst.c * B0 * 100
    Fj = gj * (2 * J + 1) * np.exp(-Ej / (cst.k * T))

    lj_stk = Fj * bjm_plus(J)
    dnu_stk = -(4 * J + 6) * B0
    is_nonzero = lj_stk != 0.0
    lj_stk = lj_stk[is_nonzero]
    dnu_stk = dnu_stk[is_nonzero]
    lj_astk = Fj * bjm_minus(J)
    is_nonzero = lj_astk != 0.0
    dnu_astk = (4 * J - 2) * B0
    lj_astk = lj_astk[is_nonzero]
    dnu_astk = dnu_astk[is_nonzero]

    norm = lj_stk.sum() + lj_astk.sum()
    return dnu_stk, lj_stk / norm, dnu_astk, lj_astk / norm


def l_air(lam, theta, T):
    r"""Air rotational Raman spectrum at a single excitation wavelength.

    Combines the N2 and O2 rotational Raman line lists, weights them
    by the dry-air mixing ratios (``X_N2``, ``X_O2``) and Joiner's
    O2-to-N2 Cabannes ratio :func:`k_ratio`, converts the
    frequency shifts :math:`\Delta\nu` (cm:sup:`-1`) into output
    wavelengths (nm), concatenates the four branches (Stokes/anti-
    Stokes for N2/O2), normalises the resulting spectrum to unit
    area and sorts it by increasing wavelength.

    Parameters
    ----------
    lam : float
        Excitation wavelength in nanometers (scalar).
    theta : float
        Scattering angle in degrees (0 = forward, 180 = backward).
    T : float
        Gas temperature in Kelvin.

    Returns
    -------
    lam_out : ndarray of float
        Output wavelengths in nanometers, sorted in increasing
        order.
    l_out : ndarray of float
        Normalised line intensities at each ``lam_out`` (spectrum
        integrates to 1).

    References
    ----------
    .. [1] Joiner, J., Bhartia, P. K., Cebula, R. P., Hilsenrath,
       E., McPeters, R. D., & Park, H. (1995). Rotational Raman
       scattering (Ring effect) in satellite backscatter ultraviolet
       measurements. *Applied Optics*, 34(21), 4513.
       https://doi.org/10.1364/AO.34.004513
    """
    dnu_stk_n2, lj_stk_n2, dnu_astk_n2, lj_astk_n2 = l_n2(T)
    dnu_stk_o2, lj_stk_o2, dnu_astk_o2, lj_astk_o2 = l_o2(T)
    lj_stk_n2 *= X_N2
    lj_astk_n2 *= X_N2
    lj_stk_o2 *= X_O2 * k_ratio(lam, theta)
    lj_astk_o2 *= X_O2 * k_ratio(lam, theta)

    nu0 = 1e7 / (lam)  # nu0 in cm-1
    # compute output lamnda in nm
    lam_stk_n2 = 1e7 / (nu0 + dnu_stk_n2)
    lam_astk_n2 = 1e7 / (nu0 + dnu_astk_n2)
    lam_stk_o2 = 1e7 / (nu0 + dnu_stk_o2)
    lam_astk_o2 = 1e7 / (nu0 + dnu_astk_o2)

    norm = (
        lj_stk_n2.sum() + lj_astk_n2.sum() + lj_stk_o2.sum() + lj_astk_o2.sum()
    )

    lam_out = np.concatenate(
        [lam_astk_n2, lam_astk_o2, lam_stk_n2, lam_stk_o2]
    )
    l_out = (
        np.concatenate([lj_astk_n2, lj_astk_o2, lj_stk_n2, lj_stk_o2]) / norm
    )
    ii = np.argsort(lam_out)

    # return spectrum with increasing wavelengths
    return lam_out[ii], l_out[ii]


def l2d(lam, theta, T):
    r"""Vectorised air rotational Raman spectrum over many wavelengths.

    Same physics as :func:`l_air`, but evaluated simultaneously for
    every excitation wavelength in ``lam``. The N2 and O2 line
    lists depend only on temperature and are reused across all
    ``lam``; the wavelength-dependent Cabannes ratio
    :func:`k_ratio` is broadcast to weight each line. The output
    is sorted by increasing wavelength along the last axis for
    each input excitation wavelength.

    Parameters
    ----------
    lam : ndarray of float, shape (nlam,)
        Excitation wavelengths in nanometers.
    theta : float
        Scattering angle in degrees (0 = forward, 180 = backward).
    T : float
        Gas temperature in Kelvin.

    Returns
    -------
    lam_out : ndarray of float, shape (nlam, nlines)
        Output wavelengths in nanometers, sorted in increasing
        order along the last axis.
    l_out : ndarray of float, shape (nlam, nlines)
        Normalised line intensities; each row integrates to 1.

    References
    ----------
    .. [1] Joiner, J., Bhartia, P. K., Cebula, R. P., Hilsenrath,
       E., McPeters, R. D., & Park, H. (1995). Rotational Raman
       scattering (Ring effect) in satellite backscatter ultraviolet
       measurements. *Applied Optics*, 34(21), 4513.
       https://doi.org/10.1364/AO.34.004513
    """
    kk = k_ratio(lam, theta)
    nlam = lam.size
    dnu_stk_n2, lj_stk_n2, dnu_astk_n2, lj_astk_n2 = l_n2(T)
    dnu_stk_o2, lj_stk_o2, dnu_astk_o2, lj_astk_o2 = l_o2(T)
    norm_n2 = lj_stk_n2.sum() + lj_astk_n2.sum()
    norm_o2 = lj_stk_o2.sum() + lj_astk_o2.sum()
    lj_stk_n2 /= norm_n2
    lj_astk_n2 /= norm_n2
    lj_stk_o2 /= norm_o2
    lj_astk_o2 /= norm_o2
    lj_stk_n2 = np.stack([lj_stk_n2] * nlam) * X_N2
    lj_astk_n2 = np.stack([lj_astk_n2] * nlam) * X_N2
    lj_stk_o2 = lj_stk_o2[np.newaxis, :] * X_O2 * kk[:, np.newaxis]
    lj_astk_o2 = lj_astk_o2[np.newaxis, :] * X_O2 * kk[:, np.newaxis]
    norm = (
        np.sum(lj_stk_n2, axis=1)
        + np.sum(lj_astk_n2, axis=1)
        + np.sum(lj_stk_o2, axis=1)
        + np.sum(lj_astk_o2, axis=1)
    )
    lj_stk_n2 /= norm[:, np.newaxis]
    lj_astk_n2 /= norm[:, np.newaxis]
    lj_stk_o2 /= norm[:, np.newaxis]
    lj_astk_o2 /= norm[:, np.newaxis]
    # we add also negative unity impulse at zero for removal of elastic
    # l_out  = np.concatenate([lj_astk_n2, lj_astk_o2, lj_stk_n2, lj_stk_o2, np.stack([np.array([0])]*nlam)], axis=1)
    # dnu_out= np.concatenate([dnu_astk_n2, dnu_astk_o2, dnu_stk_n2, dnu_stk_o2, np.array([0.])])
    l_out = np.concatenate(
        [lj_astk_n2, lj_astk_o2, lj_stk_n2, lj_stk_o2], axis=1
    )
    dnu_out = np.concatenate(
        [dnu_astk_n2, dnu_astk_o2, dnu_stk_n2, dnu_stk_o2]
    )

    # reorganization with lambda instead od Dnu and increasing order
    nu0 = 1e7 / lam
    lam_out = 1e7 / (nu0[:, np.newaxis] + dnu_out[np.newaxis, :])
    ii = np.argsort(lam_out, axis=1)
    lam_out = np.take_along_axis(lam_out, ii, axis=1)
    l_out = np.take_along_axis(l_out, ii, axis=1)

    return lam_out, l_out


def l2d_inv(lam, theta, T):
    r"""Inverse air rotational Raman spectrum (vectorised).

    Given a set of scattered wavelengths ``lam`` observed at angle
    ``theta``, returns the corresponding excitation wavelengths and
    the Ring-spectrum weights that map to each scattered
    wavelength. This is the inverse of :func:`l2d`: instead of
    computing ``output = excitation + Delta nu``, it computes
    ``excitation = output - Delta nu``. Used by callers (e.g.
    :func:`smartg.bandset.BandSet`) that need to fold the Ring
    spectrum into an output wavelength grid.

    Parameters
    ----------
    lam : ndarray of float, shape (nlam,)
        Scattered (output) wavelengths in nanometers.
    theta : float
        Scattering angle in degrees (0 = forward, 180 = backward).
    T : float
        Gas temperature in Kelvin.

    Returns
    -------
    lam_in : ndarray of float, shape (nlam, nlines)
        Excitation wavelengths in nanometers that contribute to
        each scattered wavelength, sorted in increasing order
        along the last axis.
    l_in : ndarray of float, shape (nlam, nlines)
        Normalised weights of each contribution; each row sums
        to 1.

    References
    ----------
    .. [1] Joiner, J., Bhartia, P. K., Cebula, R. P., Hilsenrath,
       E., McPeters, R. D., & Park, H. (1995). Rotational Raman
       scattering (Ring effect) in satellite backscatter ultraviolet
       measurements. *Applied Optics*, 34(21), 4513.
       https://doi.org/10.1364/AO.34.004513
    """

    dnu_stk_n2, lj_stk_n2, dnu_astk_n2, lj_astk_n2 = l_n2(T)
    dnu_stk_o2, lj_stk_o2, dnu_astk_o2, lj_astk_o2 = l_o2(T)
    # reorganization with lambda instead od Dnu
    dnu_in = np.concatenate([dnu_astk_n2, dnu_astk_o2, dnu_stk_n2, dnu_stk_o2])
    nu0 = 1e7 / lam
    lam_in1 = 1e7 / (nu0[:, np.newaxis] - dnu_stk_o2[np.newaxis, :])
    lam_in2 = 1e7 / (nu0[:, np.newaxis] - dnu_astk_o2[np.newaxis, :])
    nlam = lam.shape[0]
    kk1 = k_ratio(lam_in1, theta)
    kk2 = k_ratio(lam_in2, theta)
    norm_n2 = lj_stk_n2.sum() + lj_astk_n2.sum()
    norm_o2 = lj_stk_o2.sum() + lj_astk_o2.sum()
    lj_stk_n2 /= norm_n2
    lj_astk_n2 /= norm_n2
    lj_stk_o2 /= norm_o2
    lj_astk_o2 /= norm_o2
    lj_stk_n2 = np.stack([lj_stk_n2] * nlam) * X_N2
    lj_astk_n2 = np.stack([lj_astk_n2] * nlam) * X_N2
    lj_stk_o2 = lj_stk_o2[np.newaxis, :] * X_O2 * kk1
    lj_astk_o2 = lj_astk_o2[np.newaxis, :] * X_O2 * kk2
    norm = (
        np.sum(lj_stk_n2, axis=1)
        + np.sum(lj_astk_n2, axis=1)
        + np.sum(lj_stk_o2, axis=1)
        + np.sum(lj_astk_o2, axis=1)
    )
    lj_stk_n2 /= norm[:, np.newaxis]
    lj_astk_n2 /= norm[:, np.newaxis]
    lj_stk_o2 /= norm[:, np.newaxis]
    lj_astk_o2 /= norm[:, np.newaxis]
    l_in = np.concatenate(
        [lj_astk_n2, lj_astk_o2, lj_stk_n2, lj_stk_o2], axis=1
    )

    lam_in = 1e7 / (nu0[:, np.newaxis] - dnu_in[np.newaxis, :])
    ii = np.argsort(lam_in, axis=1)
    lam_in = np.take_along_axis(lam_in, ii, axis=1)
    l_in = np.take_along_axis(l_in, ii, axis=1)

    return lam_in, l_in
