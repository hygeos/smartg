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
import xarray as xr

from smartg.phase import THETA_GRID_KINDS, as_theta_grid, theta_grid
from smartg.smartg import _calc_phase_host

N_TEST = (2, 3, 9, 721, 1801)


def _peaked_phase(theta_deg):
    """A phase function with a 0.2 degree wide forward peak.

    Stands in for a cloud droplet or a coarse desert aerosol: a
    diffraction peak far narrower than the grid step of any reasonable
    equally spaced grid, on a smooth background.
    """
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    g = 0.85
    background = (1.0 - g**2) / (1.0 + g**2 - 2.0 * g * np.cos(theta)) ** 1.5
    return 1e4 * np.exp(-(theta / np.deg2rad(0.2)) ** 2) + background


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


@pytest.mark.parametrize("n", (721, 1801, 10001))
@pytest.mark.parametrize("kind", ("chebyshev", "lobatto"))
def test_aindex_brackets_the_angle(n, kind):
    """The interval returned must contain the angle asked for.

    This is the property the interpolation relies on, and the only one
    that matters: it is what makes the weight a weight.
    """
    nodes = theta_grid(n, kind, unit="rad")
    theta = _sample_angles(n)
    theta = theta[(theta > 0.0) & (theta < np.float32(np.pi))]

    iang, weight = _probe(theta, n, mode=1, ang=nodes)

    lo = nodes[iang].astype(np.float32)
    hi = nodes[iang + 1].astype(np.float32)
    assert np.all(lo <= theta)
    assert np.all(theta <= hi)
    assert np.all((weight >= 0.0) & (weight <= 1.0))


@pytest.mark.parametrize("n", (721, 1801, 10001))
@pytest.mark.parametrize("kind", ("chebyshev", "lobatto"))
def test_aindex_reconstructs_the_angle(n, kind):
    """The index and weight together must give the angle back.

    Interpolating a phase matrix is only as good as this, so it is
    checked directly rather than through the index.
    """
    nodes = theta_grid(n, kind, unit="rad")
    theta = _sample_angles(n)
    theta = theta[(theta > 0.0) & (theta < np.float32(np.pi))]

    iang, weight = _probe(theta, n, mode=1, ang=nodes)

    lo = nodes[iang].astype(np.float32)
    hi = nodes[iang + 1].astype(np.float32)
    err = np.abs(lo + weight * (hi - lo) - theta) / (hi - lo)
    assert err.max() < 1e-5


@pytest.mark.parametrize("kind", ("chebyshev", "lobatto"))
def test_aindex_recovers_the_nodes(kind):
    """Interpolating at a node must land on that node."""
    n = 1801
    nodes = theta_grid(n, kind, unit="rad")

    iang, weight = _probe(nodes.astype(np.float32), n, mode=1, ang=nodes)

    # a node is either the lower end with weight 0, or the upper end
    # with weight 1
    recovered = np.where(weight > 0.5, iang + 1, iang)
    assert np.array_equal(recovered, np.arange(n))
    assert np.all(np.minimum(weight, 1.0 - weight) < 1e-5)


# --------------------------------------------------------------------
# the equal-angle table
# --------------------------------------------------------------------


def _profile(theta_deg):
    """A one-entry atmospheric profile carrying a peaked phase matrix."""
    theta_deg = np.asarray(theta_deg, dtype=np.float64)
    f11 = _peaked_phase(theta_deg)
    pha = np.zeros((1, 6, len(theta_deg)))
    pha[0, 0] = f11          # F11
    pha[0, 4] = f11          # F22 = F11, spherical particles
    return xr.Dataset(
        {"phase_atm": (("iphase", "nphamat", "theta_atm"), pha)},
        coords={"theta_atm": theta_deg},
    )


def _intensity(table):
    """The intensity the kernel reconstructs from an equal-angle row."""
    return table["a_P11"] + table["a_P22"] + 2.0 * table["a_P12"]


def _built_on(grid_deg, profile):
    """The particle row of both tables built on *grid_deg*."""
    phase, cdf = _calc_phase_host(
        profile, len(grid_deg), 0.0279, "atm",
        ang_a=np.deg2rad(grid_deg),
    )
    return phase[2], cdf[2]


def _table_on(grid_deg, profile):
    """The particle row of the phase matrix table."""
    return _built_on(grid_deg, profile)[0]


@pytest.mark.parametrize("n", (901, 1801))
def test_clustered_table_resolves_the_peak_better(n):
    """At equal size, a clustered table is the more accurate one.

    Same number of entries, same memory: the only difference is where
    the angles sit. If this stops holding, clustering has stopped
    paying for itself.
    """
    source = theta_grid(72001)
    profile = _profile(source)
    probe = np.sort(
        np.concatenate([source, 0.5 * (source[1:] + source[:-1])])
    )
    reference = _peaked_phase(probe)

    error = {}
    for kind in ("uniform", "lobatto"):
        grid = theta_grid(n, kind)
        got = np.interp(probe, grid, _intensity(_table_on(grid, profile)))
        # the table is normalised, so compare shapes not magnitudes
        got = got * (reference[-1] / got[-1])
        error[kind] = np.abs(got - reference).max() / reference.max()

    assert error["lobatto"] < error["uniform"] / 5.0


def test_adopting_the_matrix_grid_loses_nothing():
    """On its own grid, the table is the phase matrix, not a resample.

    This is what makes ``theta_grid='phase'`` worth having: the
    equal-angle half is carried across rather than interpolated, so a
    phase matrix given on a well chosen grid reaches the kernel intact.
    """
    grid = theta_grid(1801, "lobatto")

    got = _intensity(_table_on(grid, _profile(grid)))
    expected = _peaked_phase(grid)

    np.testing.assert_allclose(
        got / got[-1], expected / expected[-1], rtol=1e-5
    )


def test_as_theta_grid_accepts_a_grid_or_a_count():
    assert np.array_equal(as_theta_grid(5), theta_grid(5))
    lobatto = theta_grid(101, "lobatto")
    assert np.array_equal(as_theta_grid(lobatto), lobatto)


# --------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------

# Scattering angles of the check below. The sun is at the zenith and
# the cloud is thin, so the scattering angle is the viewing angle and
# the radiance follows P11 almost directly. The first two sit in the
# forward peak, where a 451 point equally spaced grid is up to 46%
# wrong; the last two are the control, where every grid agrees.
_PEAK_ANGLES = np.array([0.2, 1.0, 20.0, 60.0])


def test_clustered_grid_fixes_the_forward_peak_radiance():
    """The grid must change the radiance where the peak is, only there.

    This is the property the whole feature exists for, checked on a
    radiance rather than on a table: at equal table size the clustered
    grid must track the finely resolved reference through the forward
    peak, where the equally spaced grid of the same size cannot.

    The tolerances are wide because the kernel is not reproducible from
    one run to the next; the measured seed to seed spread here is 0.2%,
    against the 11% error this asserts.
    """
    from smartg.atmosphere import Atm1D, Cloud
    from smartg.smartg import Smartg

    wavelength = 670.0

    def radiance(grid):
        profile = Atm1D(
            "afglms",
            comp=[Cloud("wc", 12.68, 2.0, 3.0, 0.05, wavelength)],
            grid=[100.0, 50.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.0],
            pfgrid=[100.0, 0.0],
        ).calc(np.array([wavelength]), n_theta=grid)
        m = Smartg(pp=True, double=True).run(
            wavelength, atmosphere=profile, th_deg=0.0,
            le={"th_deg": _PEAK_ANGLES,
                "phi_deg": np.array([0.0]),
                "count_level": np.full(len(_PEAK_ANGLES), 1)},
            output_layers=3, theta_grid="phase", n_photons=2e7,
            seed=1234, xblock=128, xgrid=1024,
        )
        key = [str(k) for k in m if str(k).startswith("I_down")][0]
        return np.atleast_1d(np.squeeze(m[key].values)).ravel()[
            :len(_PEAK_ANGLES)]

    reference = radiance(theta_grid(12601))
    uniform = radiance(theta_grid(451))
    lobatto = radiance(theta_grid(451, "lobatto"))

    peak = slice(0, 2)
    control = slice(2, None)

    err_uniform = np.abs(uniform - reference) / reference
    err_lobatto = np.abs(lobatto - reference) / reference

    # in the peak the equally spaced grid is far off and the clustered
    # one is not, for the very same number of table entries
    assert err_uniform[peak].max() > 0.03
    assert err_lobatto[peak].max() < 0.01
    # away from the peak they are indistinguishable, which is what says
    # the difference above is the discretisation and not an offset
    assert err_uniform[control].max() < 0.01
    assert err_lobatto[control].max() < 0.01


# --------------------------------------------------------------------
# the cumulative distribution and the sampler
# --------------------------------------------------------------------


def _psample(u, table, cdf, grid_rad, ipha=0):
    """Run the device ``pSample`` over *u* on one uploaded table.

    Returns the drawn angles, the bin index and weight the sampler
    hands back, and the layout of ``struct PGrid`` on the device.
    """
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    from pycuda.gpuarray import empty as gpuempty
    from pycuda.gpuarray import to_gpu

    from smartg.smartg import DIR_SRC, TYPE_AGRID, TYPE_PGRID

    mod = SourceModule(
        """
        #include "phase_grid.h"

        __device__ __constant__ struct AGrid Gd;
        __device__ __constant__ struct PGrid Pd;

        extern "C" __global__ void probe(
            float *u, struct Phase *func, int ipha,
            float *theta, int *iang, float *zang, int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;
            int k; float w;
            theta[i] = pSample(u[i], ipha, Pd, Gd, func, &k, &w);
            iang[i] = k;
            zang[i] = w;
        }

        extern "C" __global__ void layout(int *out)
        {
            out[0] = (int)sizeof(struct PGrid);
            out[1] = (int)offsetof(struct PGrid, cdf);
            out[2] = (int)sizeof(struct Phase);
        }
        """,
        include_dirs=[str(DIR_SRC)],
        no_extern_c=True,
    )

    n = table.shape[-1]
    table_gpu = to_gpu(np.ascontiguousarray(table))
    cdf_gpu = to_gpu(np.ascontiguousarray(cdf, dtype=np.float32).ravel())
    ang_gpu = to_gpu(np.ascontiguousarray(grid_rad, dtype=np.float32))

    log2n = int(np.floor(np.log2(n - 2))) if n > 2 else 0
    g = np.zeros(1, dtype=TYPE_AGRID)
    g["n"] = n
    g["mode"] = 1
    g["log2n"] = log2n
    g["ang"] = int(ang_gpu.gpudata)
    cuda.memcpy_htod(mod.get_global("Gd")[0], g)
    p = np.zeros(1, dtype=TYPE_PGRID)
    p["n"] = n
    p["log2n"] = log2n
    p["cdf"] = int(cdf_gpu.gpudata)
    cuda.memcpy_htod(mod.get_global("Pd")[0], p)

    out = gpuempty(3, np.int32)
    mod.get_function("layout")(out, block=(1, 1, 1), grid=(1, 1))
    layout = [int(v) for v in out.get()]

    u = np.ascontiguousarray(u, dtype=np.float32).ravel()
    theta = gpuempty(u.size, np.float32)
    iang = gpuempty(u.size, np.int32)
    zang = gpuempty(u.size, np.float32)
    mod.get_function("probe")(
        to_gpu(u), table_gpu, np.int32(ipha), theta, iang, zang,
        np.int32(u.size),
        block=(256, 1, 1), grid=((u.size + 255) // 256, 1),
    )
    return theta.get(), iang.get(), zang.get(), layout


def test_the_phase_entry_is_24_bytes():
    """The struct carries the matrix only, not a copy per CDF node."""
    from smartg.smartg import TYPE_PHASE

    assert np.dtype(TYPE_PHASE).itemsize == 24


def test_the_device_structs_match_their_numpy_mirrors():
    """A mismatched pointer offset would read garbage, silently."""
    from smartg.smartg import TYPE_PGRID, TYPE_PHASE

    grid = theta_grid(9)
    table, cdf = _built_on(grid, _profile(grid))
    _, _, _, layout = _psample([0.5], table[None], cdf[None],
                               np.deg2rad(grid))
    assert layout[0] == TYPE_PGRID.itemsize
    assert layout[1] == TYPE_PGRID.fields["cdf"][1]
    assert layout[2] == np.dtype(TYPE_PHASE).itemsize


@pytest.mark.parametrize("kind", THETA_GRID_KINDS)
def test_cdf_spans_the_closed_probability_range(kind):
    """Every row runs from 0 to 1 on its own grid and never goes back.

    The sampler bisects it for the bin a uniform deviate falls in, so
    a row that does not reach 0 or 1 leaves part of the deviates with
    no bin, and one that goes backwards breaks the bisection.
    """
    grid = theta_grid(901, kind)
    cdf = _calc_phase_host(
        _profile(grid), len(grid), 0.0279, "atm",
        ang_a=np.deg2rad(grid),
    )[1]

    assert cdf.shape[-1] == len(grid)
    for row in range(cdf.shape[0]):
        assert cdf[row, 0] == 0.0
        assert cdf[row, -1] == pytest.approx(1.0, abs=2e-7)
        assert np.all(np.diff(cdf[row]) >= 0.0)


def test_the_cdf_is_the_exact_integral_of_the_table():
    """For a constant F11 the mass below theta is (1 - cos theta)/2.

    The bins are integrated with the true sin(theta), not a quadrature
    of it, so an isotropic table pins the distribution in closed form
    to float32 rounding, on any grid.
    """
    from smartg.smartg import TYPE_PHASE, _cdf_of_table

    for kind in ("uniform", "lobatto"):
        ang = np.deg2rad(theta_grid(1801, kind))
        table = np.zeros((1, len(ang)), dtype=TYPE_PHASE)
        table["a_P11"] = 0.5
        table["a_P22"] = 0.5
        cdf = _cdf_of_table(table, ang)
        assert cdf[0] == pytest.approx(0.5 * (1.0 - np.cos(ang)),
                                       abs=1e-6)


def test_sampling_reproduces_the_tabulated_distribution():
    """Drawing from the table must follow the table, bin by bin.

    This is what the sampler exists for: push uniform deviates through
    ``pSample`` and the empirical distribution of the drawn angles has
    to match the cumulative distribution of the very table it read,
    inside the bins as well as at their edges. The earlier sampler,
    linear between equal-probability nodes, was flat inside a bin and
    fails this at the 1e-3 level on a peaked phase function.
    """
    grid = theta_grid(1801, "lobatto")
    table, cdf = _built_on(grid, _profile(grid))
    ang = np.deg2rad(grid)

    u = (np.arange(1, 400001) - 0.5) / 400000.0
    drawn, iang, zang, _ = _psample(u, table[None], cdf[None], ang)

    # the mass the table puts below each drawn angle, from the same
    # closed form the host used, with the drawn angle inside its bin
    f11 = 0.5 * (table["a_P11"].astype(np.float64)
                 + table["a_P22"].astype(np.float64)
                 + 2.0 * table["a_P12"].astype(np.float64))
    th0 = ang[iang]
    dth = ang[iang + 1] - th0
    f0 = f11[iang]
    df = f11[iang + 1] - f0
    th = drawn.astype(np.float64)
    part = (
        f0 * (np.cos(th0) - np.cos(th))
        + df * ((np.sin(th) - np.sin(th0)) / dth
                - (th - th0) / dth * np.cos(th))
    )
    full = (
        f0 * (np.cos(th0) - np.cos(ang[iang + 1]))
        + df * ((np.sin(ang[iang + 1]) - np.sin(th0)) / dth
                - np.cos(ang[iang + 1]))
    )
    below = cdf[iang].astype(np.float64) + (
        cdf[iang + 1].astype(np.float64) - cdf[iang]) * part / full

    assert np.abs(below - u).max() < 2e-5
    # the index and weight it hands back are the drawn angle's
    assert np.allclose(th0 + zang * dth, drawn, atol=1e-6)
    assert np.all(np.diff(drawn) >= 0.0)
    assert drawn[0] >= 0.0 and drawn[-1] <= np.pi + 1e-6
