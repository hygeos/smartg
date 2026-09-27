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
    as_truncation,
    truncate_phase,
    truncate_phase_set,
    truncated_ext_ssa,
)

THETA = theta_grid(1801)
GT = GTTrunc(trunc_frac=0.3, theta_tr=10.0)
DM = DMTrunc(n_streams=64)


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


def _pytrunc(
    f11: NDArray[np.float64],
    theta: NDArray[np.float64],
    truncation: DMTrunc | GTTrunc,
) -> xr.Dataset:
    """Truncate F11 with pytrunc directly, as `truncation` asks."""
    if isinstance(truncation, GTTrunc):
        return cast(xr.Dataset, gt_phase_approx(
            f11, theta, truncation.trunc_frac,
            method=truncation.integral_method,
            th_tol=truncation.theta_tol, th_f=truncation.theta_tr,
            lobatto_optimization=truncation.lobatto_optimization,
        ))
    return cast(xr.Dataset, delta_m_phase_approx(
        f11, theta, truncation.m_max,
        method=truncation.integral_method,
    ))


@pytest.mark.parametrize("truncation", [GT, DM], ids=["GT", "DM"])
def test_truncate_phase_matches_pytrunc(
    truncation: DMTrunc | GTTrunc,
) -> None:
    """F11 and f are pytrunc's; the other terms keep their F11 ratio.

    pytrunc is given F11 on the integration grid, the 1801 angles of
    the matrix and the 721 of `n_theta_integral` that it lacks, and
    its result is read back on the 1801 angles.
    """
    pha = _hg_matrix(0.85)
    pha_tr, f = truncate_phase(pha, THETA, truncation)

    theta_int, nodes = trunc_mod._integration_grid(THETA, 721)
    assert nodes is not None
    ds = _pytrunc(np.interp(theta_int, THETA, pha[0]), theta_int, truncation)
    assert f == pytest.approx(float(ds["f"].values), rel=0, abs=0)
    np.testing.assert_array_equal(pha_tr[0], ds["phase_tr"].values[nodes])
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


def test_truncate_phase_refuses_negative_result() -> None:
    """A truncation removing more than the peak holds is refused.

    Imposing both the angle and a truncation fraction larger than the
    energy within that angle leaves a negative phase function.
    """
    with pytest.raises(ValueError, match="negative"):
        truncate_phase(_hg_matrix(0.85), THETA,
                       GTTrunc(trunc_frac=0.9, theta_tr=10.0))


@pytest.mark.parametrize(
    "theta_tr", [0.0, -5.0, float("nan"), 180.0, 200.0, True]
)
def test_gttrunc_refuses_theta_tr_outside_the_angles(
    theta_tr: float,
) -> None:
    """A truncation angle outside ]0; 180[ is refused."""
    with pytest.raises(ValueError, match="theta_tr"):
        GTTrunc(trunc_frac=0.3, theta_tr=theta_tr)


def test_truncate_phase_refuses_theta_tr_below_the_resolution() -> None:
    """An angle on the first node of the grid would truncate nothing.

    pytrunc then gives back F11 unchanged but still reports f, so the
    component would only lose a fraction f of its scattering.
    """
    with pytest.raises(ValueError, match="resolution"):
        truncate_phase(_hg_matrix(0.85), THETA,
                       GTTrunc(trunc_frac=0.3, theta_tr=0.05))


def test_truncate_phase_rejects_unknown_config() -> None:
    """Anything else than DMTrunc or GTTrunc is refused."""
    with pytest.raises(TypeError, match="not recognized"):
        truncate_phase(_hg_matrix(0.85), THETA, "GT")  # type: ignore


@pytest.mark.parametrize("truncation", [GT, DM], ids=["GT", "DM"])
def test_truncate_phase_any_normalization(
    truncation: DMTrunc | GTTrunc,
) -> None:
    """The truncation does not depend on the normalization of F11.

    pytrunc expects an F11 normalized to 2: a matrix normalized
    otherwise (here to 4 pi) is truncated as its normalized copy, and
    comes back in its own normalization.
    """
    pha = _hg_matrix(0.85)
    pha_tr, f = truncate_phase(pha, THETA, truncation)
    pha_tr_4pi, f_4pi = truncate_phase(2.0 * np.pi * pha, THETA, truncation)
    assert f_4pi == pytest.approx(f, rel=1e-12)
    np.testing.assert_allclose(pha_tr_4pi, 2.0 * np.pi * pha_tr, rtol=1e-10)


def test_as_truncation() -> None:
    """None is no truncation; a boolean names no truncation method."""
    assert as_truncation(None) is None
    assert as_truncation(GT) is GT
    for value in (False, True):
        with pytest.raises(TypeError, match="DMTrunc or a GTTrunc"):
            as_truncation(value)  # type: ignore
    with pytest.raises(TypeError, match="DMTrunc or a GTTrunc"):
        as_truncation("GT")  # type: ignore


def test_truncate_phase_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each distinct matrix is truncated once, the result shared."""
    a, b = _hg_matrix(0.85), _hg_matrix(0.9)
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


#: native-like grid: 0.01 degree steps through the forward peak, 3
#: degree ones beyond, as the OPAC and cloud tables carry
COARSE = np.concatenate(
    [np.linspace(0.0, 5.0, 501), np.arange(8.0, 180.0, 3.0), [180.0]]
)


@pytest.mark.parametrize("n", [721, 7201, 18001, 72001])
def test_integration_grid_of_a_grid_holding_its_angles(n: int) -> None:
    """Equally spaced grids holding the 721 angles are kept as such."""
    theta = theta_grid(n)
    theta_int, nodes = trunc_mod._integration_grid(theta, 721)
    assert nodes is None
    assert theta_int is theta


@pytest.mark.parametrize(
    "theta",
    [THETA, COARSE, np.linspace(0.0, 179.5, 360)],
    ids=["1801", "coarse", "short"],
)
def test_integration_grid_keeps_the_angles_and_adds_equal_ones(
    theta: NDArray[np.float64],
) -> None:
    """The angles of the matrix are kept, those of the grid added.

    Within their range only: a table stopping short of 180 degrees is
    not extrapolated.
    """
    theta_int, nodes = trunc_mod._integration_grid(theta, 721)
    assert nodes is not None
    np.testing.assert_array_equal(theta_int[nodes], theta)
    assert np.all(np.diff(theta_int) > 0.0)
    assert theta_int[0] == theta[0] and theta_int[-1] == theta[-1]
    uniform = theta_grid(721)
    inside = uniform[(uniform >= theta[0]) & (uniform <= theta[-1])]
    gap = np.abs(theta_int[:, None] - inside[None, :]).min(axis=0)
    assert gap.max() <= 1e-6


@pytest.mark.parametrize("truncation", [GT, DM], ids=["GT", "DM"])
def test_truncation_on_a_grid_holding_the_angles_is_pytrunc_s(
    truncation: DMTrunc | GTTrunc,
) -> None:
    """A table holding the 721 angles is truncated on its own ones."""
    theta = theta_grid(7201)
    mu = np.cos(np.deg2rad(theta))
    f11 = (1.0 - 0.85**2) / (1.0 + 0.85**2 - 2.0 * 0.85 * mu) ** 1.5
    f11 *= 2.0 / integ_phase(np.deg2rad(theta), f11)
    pha_tr, f = truncate_phase(f11[None, :], theta, truncation)
    ds = _pytrunc(f11, theta, truncation)
    assert f == float(ds["f"].values)
    np.testing.assert_array_equal(pha_tr[0], ds["phase_tr"].values)


def test_coarse_grid_is_truncated_on_the_integration_grid() -> None:
    """The integration grid brings a coarse table near a dense result.

    On 3 degree steps, the 64 stream moments of a Delta-M truncation
    are poorly integrated: with the 721 equally spaced angles added,
    `f` and the truncated F11 get closer to their values on 18001
    added angles than without any (n_theta_integral=2 adds none, the
    former behaviour).
    """
    mu = np.cos(np.deg2rad(COARSE))
    f11 = (1.0 - 0.85**2) / (1.0 + 0.85**2 - 2.0 * 0.85 * mu) ** 1.5
    f11 = (f11 * 2.0 / integ_phase(np.deg2rad(COARSE), f11))[None, :]
    res = {
        n: truncate_phase(f11, COARSE, DMTrunc(64, n_theta_integral=n))
        for n in (2, 721, 18001)
    }
    (pha_2, f_2), (pha_721, f_721), (pha_ref, f_ref) = (
        res[2], res[721], res[18001]
    )
    assert abs(f_721 - f_ref) < abs(f_2 - f_ref)
    err_721 = np.abs(pha_721[0] - pha_ref[0]).max()
    err_2 = np.abs(pha_2[0] - pha_ref[0]).max()
    assert err_721 < err_2
    assert pha_721.shape == f11.shape


@pytest.mark.parametrize("value", [1, 0, -5, True, 721.0, "721"])
def test_n_theta_integral_is_checked(value: Any) -> None:
    """Anything but an integer of at least 2 is refused."""
    with pytest.raises(ValueError, match="n_theta_integral"):
        GTTrunc(0.3, n_theta_integral=value)
    with pytest.raises(ValueError, match="n_theta_integral"):
        DMTrunc(64, n_theta_integral=value)


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
