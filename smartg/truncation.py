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
  simplification.

These methods support different integration techniques (Lobatto
quadrature, trapezoid, Simpson) for computing phase matrix moments and
offer flexible scaling approaches.

Examples
--------
>>> from smartg.truncation import DM_trunc
>>> trunc = DM_trunc(nb_streams=16, integral_method='lobatto')

Key Classes
-----------
DM_trunc
    Delta-M truncation.
GT_trunc
    GT truncation, as in Iwabuchi and Suzuki (2009).
"""

import numpy as np
from typing import Optional


class DM_trunc(object):
    """
    Delta-M truncation

    Parameters
    ----------
    nb_streams : int
        Number of streams for the truncated phase function.
    integral_method : str, optional
        Integration method to use for computing the moments.
        Choices are:

        - 'lobatto' -> use Lobatto quadrature (default)
        - 'trapezoid' -> use scypi.integrate.trapezoid method
        - 'simpson' -> use scipy.integrate.simpson method
    pha_scale_method : int, optional
        Scaling method to use for the truncated phase matrix.
        Choices are:

        - 1 -> use Eq. 5 in Waquet et al. 2019 (Default)
        - 2 -> use ARTDECO way (same as 1, but with different rescalling
          for F21 and F34)
    """

    def __init__(
        self,
        nb_streams: int,
        integral_method: str = "lobatto",
        pha_scale_method: int = 1,
    ) -> None:
        # check parameter values
        if (
            isinstance(nb_streams, bool)
            or not isinstance(nb_streams, (int, np.integer))
            or nb_streams < 1
        ):
            raise ValueError(
                "The nb_streams parameter must be an integer >= 1."
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

        self.tr_method = "DM"
        self.m_max = nb_streams
        self.integral_method = integral_method
        self.pha_scale_method = pha_scale_method


class GT_trunc(object):
    """
    GT truncation, as in Iwabuchi and Suzuki (2009)

    Parameters
    ----------
    trunc_frac : float
        The truncature fraction
    integral_method : str, optional
        Integration method to use for computing the moments.
        Choices are:

        - 'lobatto' ->  use Lobatto quadrature (default)
        - 'trapezoid' ->  use scypi.integrate.trapezoid method
        - 'simpson' ->  use scipy.integrate.simpson method
    theta_tol : None or float, optional
        Search the truncated angle between 0 and theta_tol
        (in degrees).
    theta_tr : None or float, optional
        Directly provide the truncated angle (in degrees). If provided,
        theta_tol is ignored; trunc_frac is still used as the truncation
        fraction, only the search of the truncation angle is skipped.
    lobatto_optimization : bool, optional
        If True, use the optimized Lobatto quadrature for the integral.
        Reduces significantly the computational time in case theta_tr
        is not provided.
    pha_scale_method : int, optional
        Scaling method to use for the truncated phase matrix.
        Choices are:

        - 1 -> use Eq. 5 in Waquet et al. 2019 (Default)
        - 2 -> use ARTDECO way (same as 1, but with different rescalling
          for F12 and F34)
    """

    def __init__(
        self,
        trunc_frac: float,
        integral_method: str = "lobatto",
        theta_tol: Optional[float] = None,
        theta_tr: Optional[float] = None,
        lobatto_optimization: bool = False,
        pha_scale_method: int = 1,
    ) -> None:
        # check parameter values
        if (
            isinstance(trunc_frac, bool)
            or not isinstance(
                trunc_frac, (int, float, np.integer, np.floating)
            )
            or not (0.0 < trunc_frac < 1.0)
        ):
            raise ValueError(
                "The trunc_frac parameter must be a scalar in the "
                + "interval ]0; 1[."
            )
        integral_methods_ok = ["lobatto", "trapezoid", "simpson"]
        if integral_method not in integral_methods_ok:
            raise ValueError(
                "Choices for integral_method parameter are: "
                + f"{integral_methods_ok}."
            )
        if theta_tol is not None:
            if (
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
        if not isinstance(lobatto_optimization, bool):
            raise ValueError(
                "The lobatto_optimization parameter must be a boolean."
            )
        if pha_scale_method not in [1, 2]:
            raise ValueError(
                "Choices for pha_scale_method parameter are: 1 or 2."
            )

        self.tr_method = "GT"
        self.trunc_frac = trunc_frac
        self.integral_method = integral_method
        self.theta_tol = theta_tol
        self.theta_tr = theta_tr
        self.lobatto_optimization = lobatto_optimization
        self.pha_scale_method = pha_scale_method
