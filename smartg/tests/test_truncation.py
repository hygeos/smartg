"""GPU-free tests of the phase matrix truncation helpers.

The helpers of `smartg.truncation` truncate one phase matrix, or a
set of them, with pytrunc, and rescale the extinction and the single
scattering albedo of the truncated particles. They are checked here
against direct pytrunc calls on a forward-peaked Henyey-Greenstein
matrix, which needs no auxiliary data.
"""

from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray
from pytrunc.truncation import delta_m_phase_approx, gt_phase_approx

import smartg.truncation as trunc_mod
from smartg.phase import integ_phase, theta_grid
from smartg.truncation import (
    DMTrunc,
    GTTrunc,
    truncate_phase,
    truncate_phase_set,
    truncated_ext_ssa,
)

THETA = theta_grid(1801)
GT = GTTrunc(trunc_frac=0.3, theta_tr=10.0)
DM = DMTrunc(n_streams=16)


def _hg_matrix(g: float) -> NDArray[np.float64]:
    """Build a 6-term phase matrix with a Henyey-Greenstein F11.

    F11 is normalized to 2 over THETA, and the other terms are smooth
    fractions of it, F22 = F11 and F44 = F33 as for spheres.
    """
    mu = np.cos(np.deg2rad(THETA))
    f11 = (1.0 - g**2) / (1.0 + g**2 - 2.0 * g * mu) ** 1.5
    f11 *= 2.0 / integ_phase(np.deg2rad(THETA), f11)
    sin2 = 1.0 - mu**2
    pha = np.empty((6, len(THETA)), dtype=np.float64)
    pha[0] = f11
    pha[1] = -0.3 * sin2 * f11
    pha[2] = 0.8 * f11
    pha[3] = 0.1 * sin2 * f11
    pha[4] = f11
    pha[5] = pha[2]
    return pha


def _count_calls(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> list[int]:
    """Count the calls to the pytrunc function `name` of the module."""
    calls = [0]
    func = getattr(trunc_mod, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return func(*args, **kwargs)

    monkeypatch.setattr(trunc_mod, name, wrapper)
    return calls


@pytest.mark.parametrize("truncation", [GT, DM], ids=["GT", "DM"])
def test_truncate_phase_matches_pytrunc(
    truncation: DMTrunc | GTTrunc,
) -> None:
    """F11 and f are pytrunc's; the other terms keep their F11 ratio."""
    pha = _hg_matrix(0.85)
    pha_tr, f = truncate_phase(pha, THETA, truncation)

    if isinstance(truncation, GTTrunc):
        ds = cast(xr.Dataset, gt_phase_approx(
            pha[0], THETA, truncation.trunc_frac,
            method=truncation.integral_method,
            th_tol=truncation.theta_tol, th_f=truncation.theta_tr,
            lobatto_optimization=truncation.lobatto_optimization,
        ))
    else:
        ds = cast(xr.Dataset, delta_m_phase_approx(
            pha[0], THETA, truncation.m_max,
            method=truncation.integral_method,
        ))
    assert f == pytest.approx(float(ds["f"].values), rel=0, abs=0)
    np.testing.assert_array_equal(pha_tr[0], ds["phase_tr"].values)
    beta = pha_tr[0] / pha[0]
    np.testing.assert_allclose(pha_tr[1:], pha[1:] * beta, rtol=1e-14)
    assert 0.0 < f < 1.0
    assert pha_tr[0, 0] < pha[0, 0]  # the forward peak is removed
    assert pha_tr.dtype == np.float64


def test_truncate_phase_scale_method_2() -> None:
    """ARTDECO scaling: F21 and F34 are divided by 1 - f instead."""
    pha = _hg_matrix(0.85)
    trunc2 = GTTrunc(trunc_frac=0.3, theta_tr=10.0, pha_scale_method=2)
    pha_tr1, f1 = truncate_phase(pha, THETA, GT)
    pha_tr2, f2 = truncate_phase(pha, THETA, trunc2)
    assert f1 == f2
    np.testing.assert_allclose(pha_tr2[[1, 3]], pha[[1, 3]] / (1 - f2))
    np.testing.assert_array_equal(pha_tr2[[0, 2, 4, 5]],
                                  pha_tr1[[0, 2, 4, 5]])


def test_truncate_phase_null_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A null matrix comes back unchanged, f = 0, without pytrunc."""
    calls = _count_calls(monkeypatch, "gt_phase_approx")
    pha = np.zeros((6, len(THETA)), dtype=np.float32)
    pha_tr, f = truncate_phase(pha, THETA, GT)
    assert f == 0.0
    np.testing.assert_array_equal(pha_tr, pha)
    assert calls[0] == 0


def test_truncate_phase_rejects_unknown_config() -> None:
    """Anything else than DMTrunc or GTTrunc is refused."""
    with pytest.raises(TypeError, match="not recognized"):
        truncate_phase(_hg_matrix(0.85), THETA, "GT")  # type: ignore


def test_truncate_phase_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each distinct matrix is truncated once, the result shared."""
    a, b = _hg_matrix(0.85), _hg_matrix(0.7)
    zero = np.zeros_like(a)
    pha = np.stack([np.stack([a, b]), np.stack([a, zero])])
    calls = _count_calls(monkeypatch, "gt_phase_approx")
    pha_tr, f = truncate_phase_set(pha, THETA, GT)

    assert calls[0] == 2  # a and b; the null matrix is skipped
    assert pha_tr.shape == pha.shape
    assert f.shape == (2, 2)
    pha_a, f_a = truncate_phase(a, THETA, GT)
    pha_b, f_b = truncate_phase(b, THETA, GT)
    np.testing.assert_array_equal(f, [[f_a, f_b], [f_a, 0.0]])
    np.testing.assert_array_equal(pha_tr[0, 0], pha_a)
    np.testing.assert_array_equal(pha_tr[1, 0], pha_a)
    np.testing.assert_array_equal(pha_tr[0, 1], pha_b)
    np.testing.assert_array_equal(pha_tr[1, 1], zero)


def test_truncated_ext_ssa() -> None:
    """Scattering scaled by 1 - f, absorption kept."""
    rng = np.random.default_rng(0)
    ext = rng.uniform(0.0, 5.0, (3, 7))
    ssa = rng.uniform(0.0, 1.0, (3, 7))
    f = rng.uniform(0.0, 0.9, (3, 7))
    ext_tr, ssa_tr = truncated_ext_ssa(ext, ssa, f)
    np.testing.assert_allclose(ext_tr * ssa_tr, (1 - f) * ext * ssa)
    np.testing.assert_allclose(ext_tr * (1 - ssa_tr), ext * (1 - ssa))

    # no truncation, no change
    ext_0, ssa_0 = truncated_ext_ssa(ext, ssa, 0.0)
    np.testing.assert_allclose(ext_0, ext)
    np.testing.assert_allclose(ssa_0, ssa)

    # everything truncated: no extinction left, the albedo is kept
    ext_1, ssa_1 = truncated_ext_ssa(2.0, 1.0, 1.0)
    assert float(ext_1) == 0.0
    assert float(ssa_1) == 1.0
