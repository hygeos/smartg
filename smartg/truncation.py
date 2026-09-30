"""
Truncation methods for phase matrix simplification.

This module provides classes for phase matrix truncation techniques used
in Monte Carlo radiative transfer calculations. Truncation improves
Monte Carlo convergence by removing the sharp forward peak in the
phase function, which causes convergence problems requiring excessive
sampling.
By truncating this forward peak, fewer photon rays are needed to achieve
the same statistical accuracy (standard deviation).

Truncation benefits vary with aerosol type:

- **Large aerosols** (e.g., desert, clouds, sea salt): Significant
  convergence improvement and computational speedup.
- **Small aerosols** (e.g., continental): Limited or no improvement.

**Important:** Truncation introduces a bias in the results and should be
used with care. The trade-off between reduced computational cost and
introduced bias needs to be carefully considered for each application.

Available truncation methods:

- **Delta-M (DM)**: Classical Delta-M truncation that scales the first
  backscatter peak and removes high-order terms.
- **GT (Generalized Truncation)**: GT truncation as in Iwabuchi and
  Suzuki (2009), which provides an alternative approach to phase matrix
  simplification. With `trunc_frac=None` and an imposed angle, it cuts
  the phase matrix flat at its value at that angle, the truncation of
  the ocean of SMART-G 1.x.

These methods support different integration techniques (Lobatto
quadrature, trapezoid, Simpson) for computing phase matrix moments and
offer flexible scaling approaches.

A truncated phase matrix goes with a truncated fraction `f` of the
scattered energy, removed with the forward peak: the scattering
coefficient of the particles it describes must be scaled by `1 - f`,
their absorption being left unchanged (see `truncated_ext_ssa`).

A truncation is carried by the component whose phase matrix is
forward-peaked: the `truncation` parameter of the atmospheric
components `AerOPAC`, `Cloud`, `AerUser`, `Cloud3D` and `Aer3D`, and
of the hydrosols of `smartg.water`. Only that component is truncated:
its phase matrices are truncated before being mixed with those of the
other components of the layer or cell, and its own scattering is
scaled by `1 - f`. Several components may carry different
truncations.

Examples
--------
>>> from smartg.atmosphere import AerOPAC, Atm1D, Cloud
>>> from smartg.truncation import GTTrunc
>>> trunc = GTTrunc(trunc_frac=0.435, theta_tr=8.0)
>>> cloud = Cloud('wc', 10., 2., 3., 5., 550., truncation=trunc)
>>> aer = AerOPAC('continental_clean', 0.1, 550.)  # not truncated
>>> pro = Atm1D('afglt', comp=[aer, cloud]).calc(550.)

Key Classes
-----------
DMTrunc
    Delta-M truncation.
GTTrunc
    GT truncation, as in Iwabuchi and Suzuki (2009).

Key Functions
-------------
as_truncation
    Check the `truncation` parameter of a component, None meaning no
    truncation.
truncate_phase
    Truncate one phase matrix, and return its truncated fraction.
truncate_phase_set
    Truncate a set of phase matrices, each distinct one once.
truncated_ext_ssa
    Scale an extinction and a single scattering albedo for the
    truncation.
"""

from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from pytrunc.truncation import delta_m_phase_approx, gt_phase_approx

from smartg.phase import integ_phase, theta_grid


class DMTrunc:
    """Delta-M truncation.

    Parameters
    ----------
    n_streams : int
        Number of streams for the truncated phase function.
    integral_method : str, optional
        Integration method to use for computing the moments.
        Choices are:

        - 'lobatto' -> use Lobatto quadrature
        - 'trapezoid' -> use scipy.integrate.trapezoid method (default)
        - 'simpson' -> use scipy.integrate.simpson method
    pha_scale_method : int, optional
        Scaling method to use for the truncated phase matrix.
        Choices are:

        - 1 -> use Eq. 5 in Waquet et al. 2019 (Default)
        - 2 -> use ARTDECO way (same as 1, but with different rescalling
          for F21 and F34)
    n_theta_integral : int, optional
        Number of equally spaced scattering angles added to those of
        the phase matrix for the integrals of the truncation, 721 by
        default (0.25 degree steps). pytrunc integrates on as many
        nodes as it is given angles, and its Legendre moments need
        dense ones whatever the table: a matrix tabulated on few angles
        through the middle of the range, as a native grid can be, is
        truncated on the union of both grids, see `truncate_phase`. A
        grid that already holds these angles is truncated as it is.
    """

    def __init__(
        self,
        n_streams: int,
        integral_method: str = "trapezoid",
        pha_scale_method: int = 1,
        n_theta_integral: int = 721,
    ) -> None:
        # check parameter values
        if (
            isinstance(n_streams, bool)
            or not isinstance(n_streams, (int, np.integer))
            or n_streams < 1
        ):
            raise ValueError(
                "The n_streams parameter must be an integer >= 1."
            )
        integral_methods_ok = ["lobatto", "trapezoid", "simpson"]
        if integral_method not in integral_methods_ok:
            raise ValueError(
                "Choices for integral_method parameter are: "
                + f"{integral_methods_ok}."
            )
        if pha_scale_method not in [1, 2]:
            raise ValueError(
                "Choices for pha_scale_method parameter are: 1 or 2."
            )
        if (
            isinstance(n_theta_integral, bool)
            or not isinstance(n_theta_integral, (int, np.integer))
            or n_theta_integral < 2
        ):
            raise ValueError(
                "The n_theta_integral parameter must be an integer >= 2."
            )

        self.tr_method = "DM"
        self.m_max = n_streams
        self.integral_method = integral_method
        self.pha_scale_method = pha_scale_method
        self.n_theta_integral = int(n_theta_integral)


class GTTrunc:
    """GT truncation, as in Iwabuchi and Suzuki (2009).

    Parameters
    ----------
    trunc_frac : float or None
        The truncation fraction f, in ]0; 1[: the part of the scattering
        removed with the forward peak. None requires theta_tr: the
        phase matrix is then cut flat at its value at theta_tr, and f is
        the fraction that makes this plateau continuous with it (see
        theta_tr). The truncation of the ocean of SMART-G 1.x
        (`IOP_1`) is `GTTrunc(trunc_frac=None, theta_tr=5.0)`, but for
        its scattering coefficient, halved there by mistake.
    integral_method : str, optional
        Integration method to use for computing the moments.
        Choices are:

        - 'lobatto' ->  use Lobatto quadrature
        - 'trapezoid' ->  use scipy.integrate.trapezoid method (default)
        - 'simpson' ->  use scipy.integrate.simpson method
    theta_tol : None or float, optional
        Search the truncated angle between 0 and theta_tol
        (in degrees).
    theta_tr : None or float, optional
        Directly provide the truncated angle (in degrees), in ]0; 180[
        and not below half the first angle step of the phase matrix.
        If provided, theta_tol is ignored; trunc_frac is still used as
        the truncation fraction, only the search of the truncation
        angle is skipped. The truncated phase matrix is flat below
        theta_tr: with both imposed, the level of that plateau is set by
        the normalization and in general does not meet the phase matrix
        at theta_tr (six times above it with 0.3 and 5 degrees for the
        `HydrosolPR` of 0.5 mg/m3 at 500 nm); with trunc_frac None, it
        is the value of the phase matrix at theta_tr.
    lobatto_optimization : bool, optional
        If True, use the optimized Lobatto quadrature for the integral.
        Reduces significantly the computational time in case theta_tr
        is not provided.
    pha_scale_method : int, optional
        Scaling method to use for the truncated phase matrix.
        Choices are:

        - 1 -> use Eq. 5 in Waquet et al. 2019 (Default)
        - 2 -> use ARTDECO way (same as 1, but with different rescalling
          for F21 and F34)
    n_theta_integral : int, optional
        Number of equally spaced scattering angles added to those of
        the phase matrix for the integrals of the truncation, 721 by
        default (0.25 degree steps). pytrunc integrates on as many
        nodes as it is given angles, and its Legendre moments need
        dense ones whatever the table: a matrix tabulated on few angles
        through the middle of the range, as a native grid can be, is
        truncated on the union of both grids, see `truncate_phase`. A
        grid that already holds these angles is truncated as it is.
    """

    def __init__(
        self,
        trunc_frac: float | None,
        integral_method: str = "trapezoid",
        theta_tol: float | None = None,
        theta_tr: float | None = None,
        lobatto_optimization: bool = False,
        pha_scale_method: int = 1,
        n_theta_integral: int = 721,
    ) -> None:
        # check parameter values
        if trunc_frac is None:
            if theta_tr is None:
                raise ValueError(
                    "trunc_frac=None needs theta_tr: the truncation "
                    "fraction is then the one that makes the plateau "
                    "continuous with the phase matrix at theta_tr."
                )
        elif (
            isinstance(trunc_frac, bool)
            or not isinstance(
                trunc_frac, (int, float, np.integer, np.floating)
            )
            or not (0.0 < trunc_frac < 1.0)
        ):
            raise ValueError(
                "The trunc_frac parameter must be a scalar in the "
                + "interval ]0; 1[, or None with theta_tr."
            )
        integral_methods_ok = ["lobatto", "trapezoid", "simpson"]
        if integral_method not in integral_methods_ok:
            raise ValueError(
                "Choices for integral_method parameter are: "
                + f"{integral_methods_ok}."
            )
        if theta_tol is not None and (
            isinstance(theta_tol, bool)
            or not isinstance(
                theta_tol, (int, float, np.integer, np.floating)
            )
            or not (0.0 < theta_tol < 180.0)
        ):
            raise ValueError(
                "The theta_tol parameter must be a scalar in the "
                + "interval ]0; 180[."
            )
        if theta_tr is not None and (
            isinstance(theta_tr, bool)
            or not isinstance(
                theta_tr, (int, float, np.integer, np.floating)
            )
            or not (0.0 < theta_tr < 180.0)
        ):
            raise ValueError(
                "The theta_tr parameter must be a scalar in the "
                + "interval ]0; 180[."
            )
        if not isinstance(lobatto_optimization, bool):
            raise TypeError(
                "The lobatto_optimization parameter must be a boolean."
            )
        if pha_scale_method not in [1, 2]:
            raise ValueError(
                "Choices for pha_scale_method parameter are: 1 or 2."
            )
        if (
            isinstance(n_theta_integral, bool)
            or not isinstance(n_theta_integral, (int, np.integer))
            or n_theta_integral < 2
        ):
            raise ValueError(
                "The n_theta_integral parameter must be an integer >= 2."
            )

        self.tr_method = "GT"
        self.trunc_frac = trunc_frac
        self.integral_method = integral_method
        self.theta_tol = theta_tol
        self.theta_tr = theta_tr
        self.lobatto_optimization = lobatto_optimization
        self.pha_scale_method = pha_scale_method
        self.n_theta_integral = int(n_theta_integral)


def as_truncation(
    truncation: DMTrunc | GTTrunc | None,
) -> DMTrunc | GTTrunc | None:
    """Return the truncation a component carries, once checked.

    A truncation is only applied when asked for: the `truncation`
    parameter of every component defaults to None, no truncation.

    Parameters
    ----------
    truncation : DMTrunc or GTTrunc or None
        The `truncation` parameter of a component.

    Returns
    -------
    DMTrunc or GTTrunc or None
        `truncation` itself.

    Raises
    ------
    TypeError
        If `truncation` is neither a DMTrunc, a GTTrunc nor None, e.g.
        a boolean, which names no truncation method.
    """
    if truncation is None or isinstance(truncation, (DMTrunc, GTTrunc)):
        return truncation
    raise TypeError(
        "truncation must be a DMTrunc or a GTTrunc, or None for no "
        f"truncation, not {truncation!r}."
    )


def _integration_grid(
    theta_deg: NDArray[np.float64], n_theta_integral: int
) -> tuple[NDArray[np.float64], NDArray[np.intp] | None]:
    """Return the angles a truncation integrates a phase matrix on.

    The angles of the matrix, every one of them kept as it is, and the
    `n_theta_integral` equally spaced ones that lie within their range
    and are not within 1e-6 degree of one of them.

    Parameters
    ----------
    theta_deg : ndarray
        Strictly increasing angles of the matrix, in degrees.
    n_theta_integral : int
        Number of equally spaced angles from 0 to 180 degrees.

    Returns
    -------
    theta_int : ndarray
        The angles to integrate on, in degrees, strictly increasing.
    nodes : ndarray or None
        The index in `theta_int` of each angle of `theta_deg`, or None
        when no angle is added and `theta_int` is `theta_deg`.
    """
    extra = theta_grid(n_theta_integral)
    extra = extra[(extra > theta_deg[0]) & (extra < theta_deg[-1])]
    j = np.searchsorted(theta_deg, extra)
    gap = np.minimum(extra - theta_deg[j - 1], theta_deg[j] - extra)
    extra = extra[gap > 1e-6]
    if extra.size == 0:
        return theta_deg, None
    theta_int = np.concatenate([theta_deg, extra])
    order = np.argsort(theta_int, kind="stable")
    return theta_int[order], np.flatnonzero(order < len(theta_deg))


def truncate_phase(
    pha: ArrayLike,
    theta_deg: ArrayLike,
    truncation: DMTrunc | GTTrunc,
) -> tuple[NDArray[np.float64], float]:
    """Truncate the forward peak of one phase matrix.

    The F11 term is truncated with pytrunc, following `truncation`.
    The other terms are scaled by the ratio of the truncated F11 to
    the exact one, angle by angle (Eq. 5 in Waquet et al. 2019), and
    `pha_scale_method` 2 instead scales F21 and F34 by `1 / (1 - f)`.

    A null matrix, the one of a component absent from a layer, is
    returned unchanged with a null truncated fraction: it carries no
    peak to remove and no scattering to rescale.

    A truncation giving a negative F11 is refused: a negative phase
    function is no probability distribution, and the Monte Carlo
    sampling of the scattering angle breaks on it.

    pytrunc builds the truncated phase function of an F11 normalized
    to 2: an F11 normalized otherwise, by more than 1 % (to 4 pi, or a
    volume scattering function), is normalized before the truncation,
    and the truncated matrix comes back in the normalization of `pha`.

    pytrunc integrates on the angles it is given, one quadrature node
    per angle, and the Legendre moments of the truncation need dense
    nodes whatever the table. F11 is therefore truncated on its angles
    and the `n_theta_integral` equally spaced ones of `truncation`
    (721 by default), interpolated linearly in angle, as the kernel
    reads it, and the truncated F11 is returned on its own angles. A
    table that already holds those angles, such as 721, 7201, 18001 or
    72001 equally spaced ones, is truncated on its angles alone.

    Parameters
    ----------
    pha : array_like
        Phase matrix of shape (nphamat, ntheta), whose terms follow the
        SMART-G order F11, F21, F33, F34, F22, F44 (only the first ones
        when nphamat < 6), in any normalization.
    theta_deg : array_like
        Scattering angles in degrees, from 0 to 180.
    truncation : DMTrunc or GTTrunc
        Truncation configuration.

    Returns
    -------
    pha_tr : ndarray
        Truncated phase matrix, same shape as `pha`, in float64.
    f : float
        Truncated fraction of the scattered energy, i.e. the fraction
        removed with the forward peak.

    Raises
    ------
    TypeError
        If the truncation configuration is not recognized.
    ValueError
        If the GT truncation angle ``theta_tr`` is nearer to the first
        angle of the grid than to the second one, if F11 does not
        integrate to a positive value, or if the truncated F11 is
        negative, which happens when the truncation
        removes more energy than the forward peak holds: a GT
        truncation fraction larger than that energy (with the
        truncation angle imposed, or on a phase function without a
        peak), or the Legendre ringing of a Delta-M truncation with
        too few streams.
    """
    pha = np.asarray(pha, dtype=np.float64)
    theta_deg = np.asarray(theta_deg, dtype=np.float64)
    if not isinstance(truncation, (DMTrunc, GTTrunc)):
        raise TypeError("truncation method not recognized")
    f11 = pha[0]
    if not f11.any():
        return pha.copy(), 0.0

    theta_int, nodes = _integration_grid(
        theta_deg, truncation.n_theta_integral
    )
    f11_int = f11 if nodes is None else np.interp(theta_int, theta_deg, f11)
    # the normalization is the table's, on its own angles
    norm = float(integ_phase(np.deg2rad(theta_deg), f11))
    if not norm > 0.0:
        raise ValueError(
            f"F11 integrates to {norm:.3g}: it cannot be truncated."
        )
    # pytrunc truncates an F11 normalized to 2. A table normalized to
    # within 1 % is truncated as it is: the angle search of GT is so
    # sensitive that renormalizing to the accuracy of another
    # integration rule (1e-4) moves its truncation angle by tenths of a
    # degree, a change that tells nothing about the table
    scale = 2.0 / norm if abs(norm - 2.0) > 0.02 else 1.0
    if isinstance(truncation, DMTrunc):
        ds_pha = cast(
            xr.Dataset,
            delta_m_phase_approx(
                f11_int * scale,
                theta_int,
                truncation.m_max,
                method=truncation.integral_method,
            ),
        )
    else:
        # pytrunc takes the angle of the grid nearest to theta_tr; at
        # the first one it truncates nothing but still reports f
        if truncation.theta_tr is not None and np.argmin(
            np.abs(theta_int - truncation.theta_tr)
        ) == 0:
            raise ValueError(
                f"theta_tr = {truncation.theta_tr:g} degree is below the "
                "resolution of the phase matrix, whose first angles are "
                f"{theta_int[0]:g} and {theta_int[1]:g} degree: nothing "
                "would be truncated."
            )
        try:
            ds_pha = cast(
                xr.Dataset,
                gt_phase_approx(
                    f11_int * scale,
                    theta_int,
                    truncation.trunc_frac,
                    method=truncation.integral_method,
                    th_tol=truncation.theta_tol,
                    th_f=truncation.theta_tr,
                    lobatto_optimization=truncation.lobatto_optimization,
                ),
            )
        except ValueError as exc:
            if truncation.trunc_frac is not None:
                raise
            raise ValueError(
                f"{exc} A component whose phase function has no forward "
                "peak takes truncation=None (the hydrosols of "
                "smartg.water truncate by default)."
            ) from exc
    f11_tr = np.asarray(ds_pha["phase_tr"].values, dtype=np.float64) / scale
    f = float(ds_pha["f"].values)
    if (f11_tr < 0.0).any():
        raise ValueError(
            "The truncated phase function is negative (down to "
            f"{float(f11_tr.min()):.3g}, f = {f:.3g}): the truncation "
            "removes more energy than the forward peak holds. Lower "
            "trunc_frac or pass trunc_frac=None for a plateau "
            "continuous with the phase function, let GTTrunc search the "
            "truncation angle (theta_tr=None), raise n_streams, or do "
            "not truncate a phase function without a marked forward "
            "peak."
        )
    if nodes is not None:
        f11_tr = f11_tr[nodes]

    pha_tr = np.empty_like(pha)
    pha_tr[0] = f11_tr
    # the other terms keep their ratio to F11; where F11 vanishes, so
    # do they
    beta = np.divide(
        f11_tr, f11, out=np.zeros_like(f11), where=f11 != 0.0
    )
    pha_tr[1:] = pha[1:] * beta
    if truncation.pha_scale_method == 2 and pha.shape[0] > 3:
        pha_tr[1] = pha[1] / (1.0 - f)
        pha_tr[3] = pha[3] / (1.0 - f)
    return pha_tr, f


def truncate_phase_set(
    pha: ArrayLike,
    theta_deg: ArrayLike,
    truncation: DMTrunc | GTTrunc,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Truncate a set of phase matrices, each distinct one once.

    Identical matrices, which the tabulations share widely (a
    wavelength or a layer repeating the same matrix, the null matrix
    of the layers a component is absent from), are truncated once and
    share the result.

    Parameters
    ----------
    pha : array_like
        Phase matrices of shape (..., nphamat, ntheta), see
        `truncate_phase`.
    theta_deg : array_like
        Scattering angles in degrees, from 0 to 180.
    truncation : DMTrunc or GTTrunc
        Truncation configuration.

    Returns
    -------
    pha_tr : ndarray
        Truncated phase matrices, same shape as `pha`, in float64.
    f : ndarray
        Truncated fraction of each matrix, of shape `pha.shape[:-2]`.
    """
    pha = np.asarray(pha, dtype=np.float64)
    lead = pha.shape[:-2]
    flat = pha.reshape((-1,) + pha.shape[-2:])
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    pha_tr_uniq = np.empty_like(uniq)
    f_uniq = np.zeros(len(uniq), dtype=np.float64)
    for i in range(len(uniq)):
        pha_tr_uniq[i], f_uniq[i] = truncate_phase(
            uniq[i], theta_deg, truncation
        )
    return pha_tr_uniq[inv].reshape(pha.shape), f_uniq[inv].reshape(lead)


def truncated_ext_ssa(
    ext: ArrayLike,
    ssa: ArrayLike,
    f: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Rescale an extinction and its albedo for a truncation.

    The scattering `ext * ssa` is scaled by `1 - f` while the
    absorption `ext * (1 - ssa)` is kept:

    - ``ext_tr = ext * (1 - f * ssa)``
    - ``ssa_tr = ssa * (1 - f) / (1 - f * ssa)``

    The inputs are broadcast together. Where `f * ssa` is 1, i.e. the
    whole extinction is truncated scattering, `ext_tr` is null and
    `ssa` is kept unchanged.

    Parameters
    ----------
    ext : array_like
        Extinction coefficient or optical thickness, in any unit.
    ssa : array_like
        Single scattering albedo.
    f : array_like
        Truncated fraction of the phase matrix, see `truncate_phase`.

    Returns
    -------
    ext_tr : ndarray
        Truncated extinction, in the unit of `ext`.
    ssa_tr : ndarray
        Truncated single scattering albedo.
    """
    ext = np.asarray(ext)
    ssa = np.asarray(ssa)
    f = np.asarray(f)
    scale = 1.0 - f * ssa
    ext_tr = ext * scale
    ssa_tr = np.divide(
        ssa * (1.0 - f),
        scale,
        out=np.array(np.broadcast_to(ssa, scale.shape), dtype=scale.dtype),
        where=scale != 0.0,
    )
    return ext_tr, ssa_tr
