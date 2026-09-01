#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for the angular grid of the phase tables.

The Monte Carlo kernel is not reproducible from one run to the next,
so the angular grid is checked where it *is* deterministic: the host
tables built by ``_calc_phase_host``, and the device lookup ``aIndex``
driven by a small probe kernel.
"""

import numpy as np
import pytest

from smartg.phase import THETA_GRID_KINDS, theta_grid

N_TEST = (2, 3, 9, 721, 1801)


# --------------------------------------------------------------------
# the generators
# --------------------------------------------------------------------


@pytest.mark.parametrize("kind", THETA_GRID_KINDS)
@pytest.mark.parametrize("n", N_TEST)
def test_theta_grid_shape_and_bounds(kind, n):
    """Every kind spans [0, 180] exactly and is strictly increasing."""
    theta = theta_grid(n, kind)

    assert theta.shape == (n,)
    assert theta[0] == 0.0
    assert theta[-1] == 180.0
    assert np.all(np.diff(theta) > 0.0)


@pytest.mark.parametrize("kind", THETA_GRID_KINDS)
@pytest.mark.parametrize("n", N_TEST)
def test_theta_grid_radians(kind, n):
    """The radian grid is the degree grid, converted."""
    theta = theta_grid(n, kind, unit="rad")

    assert theta[0] == 0.0
    assert theta[-1] == np.pi
    np.testing.assert_allclose(theta, np.deg2rad(theta_grid(n, kind)))


@pytest.mark.parametrize("n", N_TEST)
def test_theta_grid_uniform_is_linspace(n):
    """The default kind must stay the historical grid, exactly."""
    assert np.array_equal(theta_grid(n), np.linspace(0.0, 180.0, n))


def test_theta_grid_clusters_towards_the_peak():
    """Both non-uniform kinds resolve the forward peak far better.

    They are also interchangeable in practice, which is what lets the
    kernel invert the Chebyshev one analytically and treat the Lobatto
    one as a tabulated grid.
    """
    n = 1801
    uniform = theta_grid(n)
    cheby = theta_grid(n, "chebyshev")
    lobatto = theta_grid(n, "lobatto")

    assert np.sum(uniform < 1.0) == 10
    assert np.sum(cheby < 1.0) == 86
    assert np.sum(lobatto < 1.0) == 86
    assert np.abs(cheby - lobatto).max() < 0.02


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 1},
        {"n": 10, "kind": "gauss"},
        {"n": 10, "unit": "grad"},
    ],
)
def test_theta_grid_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        theta_grid(**kwargs)


# --------------------------------------------------------------------
# the device lookup
# --------------------------------------------------------------------


def _probe(theta, n, mode=0, ang=None):
    """Run the device ``aIndex`` over *theta*, return (iang, weight)."""
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    from pycuda.gpuarray import empty as gpuempty
    from pycuda.gpuarray import to_gpu

    from smartg.smartg import DIR_SRC, TYPE_AGRID

    mod = SourceModule(
        """
        #include "phase_grid.h"

        __device__ __constant__ struct AGrid Gd;

        extern "C" __global__ void probe(
            float *theta, int *iang, float *w, int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;
            int k;
            w[i] = aIndex(theta[i], Gd, &k);
            iang[i] = k;
        }
        """,
        include_dirs=[str(DIR_SRC)],
        no_extern_c=True,
    )

    theta = np.ascontiguousarray(theta, dtype=np.float32)
    ang_gpu = None if ang is None else to_gpu(
        np.ascontiguousarray(ang, dtype=np.float32)
    )

    rec = np.zeros(1, dtype=TYPE_AGRID)
    rec["n"] = n
    rec["mode"] = mode
    rec["log2n"] = int(np.floor(np.log2(n - 2))) if n > 2 else 0
    rec["ang"] = 0 if ang_gpu is None else int(ang_gpu.gpudata)
    cuda.memcpy_htod(mod.get_global("Gd")[0], rec)

    size = theta.size
    iang = gpuempty(size, np.int32)
    weight = gpuempty(size, np.float32)
    mod.get_function("probe")(
        to_gpu(theta), iang, weight, np.int32(size),
        block=(256, 1, 1), grid=((size + 255) // 256, 1),
    )
    return iang.get(), weight.get()


def _sample_angles(n, seed=0):
    """Node values, random angles and both end points."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            np.float32(np.pi) * np.arange(n, dtype=np.float32)
            / np.float32(n - 1),
            rng.uniform(0.0, np.pi, 200000),
            [0.0, np.pi],
        ]
    ).astype(np.float32)


def test_aindex_uniform_matches_the_historical_expression():
    """Mode 0 must reproduce ``theta*(NF-1)/PI`` bit for bit.

    This is what guarantees that a run left on the default grid is
    unaffected by the introduction of ``aIndex``; the kernel itself
    cannot be compared, as it is not reproducible run to run. The last
    interval is excluded, as that is the one the clamp fixes.
    """
    n = 10001
    theta = _sample_angles(n)
    x = theta * np.float32(n - 1) / np.float32(np.pi)
    inside = x < n - 2
    theta = theta[inside]
    x = x[inside]

    iang, weight = _probe(theta, n, mode=0)
    expected_iang = np.floor(x).astype(np.int32)
    expected_w = (x - expected_iang).astype(np.float32)

    assert np.array_equal(iang, expected_iang)
    assert weight.tobytes() == expected_w.tobytes()


@pytest.mark.parametrize("n", (3, 721, 10001))
def test_aindex_never_leaves_the_table(n):
    """The index must address an existing interval, for any angle.

    Both ``iang`` and ``iang + 1`` are read, so an index of ``n - 1``
    reaches into the next phase function, or past the allocation for
    the last one. Exact backscattering makes that reachable.
    """
    theta = np.concatenate(
        [_sample_angles(n), [np.pi, np.nextafter(np.pi, 4.0), np.nan]]
    ).astype(np.float32)
    iang, weight = _probe(theta, n, mode=0)

    assert iang.min() >= 0
    assert iang.max() <= n - 2
    assert np.all(weight >= 0.0)
    assert np.all(weight <= 1.0)


def test_aindex_backscattering_hits_the_last_node():
    """theta = 180 degrees must interpolate onto the last entry."""
    n = 10001
    iang, weight = _probe(np.array([np.pi]), n, mode=0)

    assert iang[0] == n - 2
    assert weight[0] == 1.0
