"""GPU-free tests of the reference checks of the IPRT phase B tests."""

import logging

import numpy as np

from smartg.iprt.common import compute_deltam
from smartg.tests.iprt_checks import ReferenceChecks

CHECKS = ReferenceChecks(logging.getLogger("test_iprt_checks"), 1e-3, 0.01)


def _maps() -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Return reference and model maps, Q being mostly noise."""
    rng = np.random.default_rng(1)
    i = np.full(400, 1.0)
    q_ref = np.full(400, 0.01)
    iquv_ref = (i, q_ref, 0.5 * q_ref, 0.1 * q_ref)
    iquv_mod = tuple(
        x + 0.01 * rng.standard_normal(400) for x in iquv_ref
    )
    return iquv_ref, iquv_mod


def test_check_deltam_refuses_a_map_of_zeros() -> None:
    """A zero map is refused though its delta_m is inside the band."""
    iquv_ref, iquv_mod = _maps()
    delta_m_ref = tuple(
        float(x)
        for x in compute_deltam(list(iquv_ref), list(iquv_mod), False)
    )
    signal_ref = tuple(float(np.mean(np.abs(x))) for x in iquv_mod)
    # the noisy Q has a delta_m of about 100 %, like a map of zeros
    assert 60.0 < delta_m_ref[1] < 140.0
    assert not CHECKS.check_deltam(
        delta_m_ref, signal_ref, iquv_ref, iquv_mod, "ok", 0.4
    )

    zero_q = (iquv_mod[0], np.zeros(400), iquv_mod[2], iquv_mod[3])
    errors = CHECKS.check_deltam(
        delta_m_ref, signal_ref, iquv_ref, zero_q, "zero Q", 0.4
    )
    assert len(errors) == 1
    assert "Q has lost its signal" in errors[0]
