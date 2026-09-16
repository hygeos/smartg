"""Checks of IPRT runs against saved reference values, for the tests.

The IPRT phase B tests assert two observables of the I, Q, U and V
maps of a run: their delta_m against the reference model, within a
fractional band around a saved value, and their spatial means, within
an absolute band scaled by the mean of I. A Stokes component whose
signal is too small compared with the one of I is only logged. The
checks return the failure messages instead of asserting, so that a
test reports every case it runs.

Key Functions
-------------
ReferenceChecks
    Delta_m and mean checks sharing a logger and their tolerances.
"""

import logging
from dataclasses import dataclass

import numpy as np

from smartg.iprt.common import compute_deltam

STOKES = ("I", "Q", "U", "V")


@dataclass(frozen=True)
class ReferenceChecks:
    """Delta_m and mean checks sharing a logger and their tolerances.

    Attributes
    ----------
    logger : logging.Logger
        The logger of the calculated and reference values.
    signal_floor : float
        A component whose mean absolute value is below signal_floor
        times the one of I is logged but not asserted: it is Monte
        Carlo noise.
    mean_tol : float
        The tolerance on the mean of every component, as a fraction of
        the mean of I.
    """

    logger: logging.Logger
    signal_floor: float
    mean_tol: float

    def is_significant(self, signal_ref: tuple[float, ...] | None,
                       istk: int) -> bool:
        """Tell whether a Stokes component is worth asserting on.

        Parameters
        ----------
        signal_ref : tuple of float, optional
            The mean absolute values of I, Q, U and V. None, i.e. not
            yet measured, keeps every component so that a new reference
            gets fully logged.
        istk : int
            The index of the component, 0 for I.

        Returns
        -------
        bool
            True if the component carries enough signal.
        """
        if signal_ref is None:
            return True

        return signal_ref[istk] > self.signal_floor * signal_ref[0]

    def skipped(self, signal_ref: tuple[float, ...] | None) -> list[str]:
        """Return the names of the components left unasserted.

        Parameters
        ----------
        signal_ref : tuple of float, optional
            The mean absolute values of I, Q, U and V.

        Returns
        -------
        list of str
            The names of the components below the signal floor.
        """
        return [
            stk
            for istk, stk in enumerate(STOKES)
            if not self.is_significant(signal_ref, istk)
        ]

    def check_deltam(
        self,
        delta_m_ref: tuple[float, ...] | None,
        signal_ref: tuple[float, ...] | None,
        iquv_ref: tuple[np.ndarray, ...],
        iquv_mod: tuple[np.ndarray, ...],
        label: str,
        tol: float,
    ) -> list[str]:
        """Compare the delta_m values with the saved validated ones.

        Parameters
        ----------
        delta_m_ref : tuple of float, optional
            The saved delta_m of I, Q, U and V. With None the
            calculated values are logged and the case is reported as a
            failure, which is how a new reference is measured before
            being written in the tables of the test.
        signal_ref : tuple of float, optional
            The mean absolute values of I, Q, U and V, see
            signal_floor.
        iquv_ref, iquv_mod : tuple of ndarray
            The reference model (e.g. MYSTIC) and the SMART-G I, Q, U
            and V maps.
        label : str
            The case label, for the log and the messages.
        tol : float
            The two sided fractional band around the saved delta_m.

        Returns
        -------
        list of str
            The failure messages, empty if the case is ok.
        """
        delta_m = compute_deltam(
            obs=list(iquv_ref), mod=list(iquv_mod), print_res=False
        )

        if delta_m_ref is not None:
            self.logger.info(
                f"{label} - I={delta_m_ref[0]:.3f}; "
                f"Q={delta_m_ref[1]:.3f}; U={delta_m_ref[2]:.3f}; "
                f"V={delta_m_ref[3]:.3f} - ref delta_m:"
            )
        self.logger.info(
            f"{label} - I={delta_m[0]:.3f}; Q={delta_m[1]:.3f}; "
            f"U={delta_m[2]:.3f}; V={delta_m[3]:.3f} - calculated delta_m"
        )

        if delta_m_ref is None:
            return [
                f"{label}: no reference delta_m, see the log for the values"
            ]

        skipped = self.skipped(signal_ref)
        if skipped:
            self.logger.info(
                f"{label} - {', '.join(skipped)} below SIGNAL_FLOOR, "
                "not asserted"
            )

        errors = []
        for istk, stk in enumerate(STOKES):
            if not self.is_significant(signal_ref, istk):
                continue
            ref = delta_m_ref[istk]
            if abs(delta_m[istk] - ref) > tol * ref:
                errors.append(
                    f"{label}: problem with {stk} values, get "
                    f"{delta_m[istk]:.5f}. {stk} must be within "
                    f"[{(1 - tol) * ref:.5f}, {(1 + tol) * ref:.5f}]"
                )

        return errors

    def check_means(
        self,
        mean_ref: tuple[float, ...] | None,
        signal_ref: tuple[float, ...] | None,
        iquv_mod: tuple[np.ndarray, ...],
        label: str,
    ) -> list[str]:
        """Compare the spatial means with the saved validated ones.

        Unlike delta_m, the mean averages the Monte Carlo noise out, so
        it is the observable that stays sensitive to a systematic bias
        at low photon counts. Same contract as check_deltam.

        Parameters
        ----------
        mean_ref : tuple of float, optional
            The saved means of I, Q, U and V. With None the calculated
            values are logged and the case is reported as a failure.
        signal_ref : tuple of float, optional
            The mean absolute values of I, Q, U and V, see
            signal_floor.
        iquv_mod : tuple of ndarray
            The SMART-G I, Q, U and V maps.
        label : str
            The case label, for the log and the messages.

        Returns
        -------
        list of str
            The failure messages, empty if the case is ok.
        """
        means = tuple(float(np.mean(stk)) for stk in iquv_mod)

        if mean_ref is not None:
            self.logger.info(
                f"{label} - I={mean_ref[0]:.6e}; Q={mean_ref[1]:.6e}; "
                f"U={mean_ref[2]:.6e}; V={mean_ref[3]:.6e} - ref mean:"
            )
        self.logger.info(
            f"{label} - I={means[0]:.6e}; Q={means[1]:.6e}; "
            f"U={means[2]:.6e}; V={means[3]:.6e} - calculated mean"
        )

        if mean_ref is None:
            return [
                f"{label}: no reference mean, see the log for the values"
            ]

        # The mean of I sets the scale of the four tolerances, the
        # means of Q, U and V being much smaller than it, or free to
        # pass through zero
        tol = self.mean_tol * abs(mean_ref[0])

        errors = []
        for istk, stk in enumerate(STOKES):
            if not self.is_significant(signal_ref, istk):
                continue
            ref = mean_ref[istk]
            if abs(means[istk] - ref) > tol:
                errors.append(
                    f"{label}: problem with the mean of {stk}, get "
                    f"{means[istk]:.6e}. It must be within "
                    f"[{ref - tol:.6e}, {ref + tol:.6e}]"
                )

        return errors
