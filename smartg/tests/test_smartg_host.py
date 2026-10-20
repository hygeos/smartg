"""GPU-free tests of the host-side helpers of smartg.smartg.

They check the arithmetic that Smartg.run does on the host around
the kernel, without building a Smartg object.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from smartg import smartg as sg_module
from smartg.albedo import AlbedoCst, AlbedoMap
from smartg.atmosphere import AerOPAC, Atm1D
from smartg.bandset import BandSet
from smartg.sensor import Sensor, get_sensors_grid
from smartg.smartg import (
    LocalEstimate,
    Smartg,
    StdevLim,
    _calc_phase_host,
    _calc_solid_angles,
    _check_albedo_map_codes,
    _check_flat_surface_le,
    _check_forward_raster,
    _check_phase_memory,
    _check_sensor_directions,
    _device_memory,
    _distinct_phase_profile,
    _distinct_phase_rows,
    _finalize,
    _host_memory,
    _impact_init,
    _init_profile,
    _isotropic,
    _RatioStdev,
    _resolve_n_loop,
    _stdev_lim_reached,
    multi_profiles,
)
from smartg.surface import FlatSurface, RoughSurface
from smartg.water import HydrosolPR, Water1D


@pytest.mark.parametrize("sza_max", [30.0, 60.0, 90.0, 120.0])
def test_solid_angles_cover_the_zenith_range(sza_max: float) -> None:
    """Check that the bins cover 1 - cos(sza_max) of the hemisphere."""
    n_phi = 4
    tab_th, _, tab_omega = _calc_solid_angles(30, n_phi, sza_max)
    np.testing.assert_allclose(
        tab_omega.sum() * n_phi, 1.0 - np.cos(np.radians(sza_max))
    )
    assert tab_th[-1] < np.radians(sza_max)


def test_solid_angles_do_not_depend_on_sza_max() -> None:
    """Check that a bin keeps its solid angle when sza_max changes."""
    th_full, _, omega_full = _calc_solid_angles(45, 4, 90.0)
    th_part, _, omega_part = _calc_solid_angles(30, 4, 60.0)
    np.testing.assert_allclose(th_part, th_full[:30])
    np.testing.assert_allclose(omega_part, omega_full[:30], rtol=1e-3)


@pytest.mark.parametrize("sun_disc", [0.266, 2.0, 90.0])
def test_sun_disc_solid_angle_in_the_units_of_the_bins(
    sun_disc: float,
) -> None:
    """Check that the disc counts its solid angle as the bins do.

    A bin counts its solid angle as a fraction of the hemisphere, so
    must the disc, or the radiances of the disc and of the bins differ
    by 2 pi: the disc of 90 degrees is the whole hemisphere, as all the
    bins of 0 to 90 degrees are.
    """
    n_phi = 4
    _, _, omega_bins = _calc_solid_angles(30, n_phi, 90.0)
    _, _, omega_disc = _calc_solid_angles(30, n_phi, 90.0, sun_disc)
    np.testing.assert_allclose(
        omega_disc, 1.0 - np.cos(np.radians(sun_disc))
    )
    if sun_disc == 90.0:
        np.testing.assert_allclose(omega_disc, omega_bins.sum() * n_phi)


def _coast_map(codes: list[list[int]]) -> AlbedoMap:
    """Return a map of two cells along x, with two albedos."""
    return AlbedoMap(
        np.array(codes), np.array([0.0, 1e8]), np.array([1e8]),
        [AlbedoCst(0.1), AlbedoCst(0.2)],
    )


def test_albedo_map_codes_accepted() -> None:
    """Check the codes that address the albedo list of the map."""
    _check_albedo_map_codes(_coast_map([[0], [-1]]), water=True)
    _check_albedo_map_codes(_coast_map([[1], [-1]]), water=True)
    # without water, a negative code only selects the surface
    _check_albedo_map_codes(_coast_map([[0], [-5]]), water=False)


@pytest.mark.parametrize(
    ("codes", "water"),
    [([[2], [-1]], False), ([[0], [-2]], True)],
    ids=["land", "seafloor"],
)
def test_albedo_map_codes_refused(codes: list[list[int]], water: bool) -> None:
    """Check that a code past the albedo list raises a ValueError."""
    with pytest.raises(ValueError, match="list holds 2 albedos"):
        _check_albedo_map_codes(_coast_map(codes), water=water)


def _raster(xgrid: list[float], ygrid: list[float]) -> list[Sensor]:
    """Return the sensors of a raster of 1 km cells, x varying first."""
    return get_sensors_grid(
        np.array(xgrid), np.array(ygrid), loc="ATMOS", cell_size=1.0
    )


def test_forward_raster_accepted() -> None:
    """Check that the raster of get_sensors_grid is accepted."""
    _check_forward_raster(_raster([0, 1, 2, 3], [5, 6, 7]), 0.0, 5.0, 3, 2)


@pytest.mark.parametrize(
    "case", ["no cell", "missing", "y first", "other size"]
)
def test_forward_raster_refused(case: str) -> None:
    """Check the sensor lists that are not a complete raster."""
    sensors = _raster([0, 1, 2, 3], [5, 6, 7])
    if case == "no cell":
        sensors = [Sensor(loc="ATMOS")]
    elif case == "missing":
        del sensors[4]
    elif case == "y first":
        sensors = sorted(sensors, key=lambda s: s.dict["pos_x"])
    else:
        sensors[2].cell_size = 0.5
    with pytest.raises(ValueError, match="forward run in a 3D atmosphere"):
        _check_forward_raster(sensors, 0.0, 5.0, 3, 2)


# (loc, th_deg, sensor_type, fov) of the sensors that look at the
# interface they stand on, or that stand in a medium
LOOKING_AT_IT = [
    ("SURF0P", 150.0, 0, 0.0),
    ("SURF0P", 180.0, 1, 90.0),
    ("SURF0M", 40.0, 0, 0.0),
    ("SURF0M", 0.0, 2, 90.0),
    ("SEAFLOOR", 180.0, 0, 0.0),
    ("ATMOS", 40.0, 0, 0.0),
    ("OCEAN", 150.0, 0, 0.0),
]
LOOKING_AWAY = [
    ("SURF0P", 0.0, 0, 0.0),
    ("SURF0P", 90.0, 0, 0.0),
    ("SURF0P", 150.0, 1, 90.0),
    ("SURF0M", 150.0, 0, 0.0),
    ("SURF0M", 30.0, 1, 90.0),
    ("SEAFLOOR", 30.0, 0, 0.0),
]


def _sensor(loc: str, th_deg: float, sensor_type: int, fov: float) -> Sensor:
    """Return a sensor of the given localization and direction."""
    return Sensor(loc=loc, th_deg=th_deg, sensor_type=sensor_type, fov=fov)


@pytest.mark.parametrize("case", LOOKING_AT_IT)
def test_sensor_directions_accepted(
    case: tuple[str, float, int, float],
) -> None:
    """Check the sensors that look at the interface they stand on."""
    _check_sensor_directions([_sensor(*case)])


@pytest.mark.parametrize("case", LOOKING_AWAY)
def test_sensor_directions_refused(
    case: tuple[str, float, int, float],
) -> None:
    """Check the sensors on an interface that look away from it.

    Their photons meet the interface all the same, and are lost: a
    sensor at 'SURF0P' looking up measured 0, and one at 'SURF0M'
    looking down hung the kernel.
    """
    sensors = [_sensor("ATMOS", 40.0, 0, 0.0), _sensor(*case)]
    with pytest.raises(ValueError, match=f"sensor 1 at '{case[0]}'"):
        _check_sensor_directions(sensors)


def test_flat_surface_refused_with_le() -> None:
    """Check that a flat surface is refused with a local estimate.

    It reflects and refracts in a single direction, which the local
    estimate cannot aim at: the backward TOA radiance over water lost
    two thirds of its value, the one leaving the water.
    """
    le = LocalEstimate(th_deg=[30.0], phi_deg=[90.0])
    with pytest.raises(ValueError, match="FlatSurface"):
        _check_flat_surface_le(FlatSurface(), le)
    _check_flat_surface_le(FlatSurface(), None)
    _check_flat_surface_le(RoughSurface(wind=0.0), le)


def _profile_1d() -> xr.Dataset:
    """Return a two layer 1D profile of optical depth 0.3."""
    return xr.Dataset(
        {"OD_atm": (("wavelength", "z_atm"), [[0.0, 0.1, 0.3]])},
        coords={"wavelength": [550.0], "z_atm": [120.0, 10.0, 0.0]},
    )


@pytest.mark.parametrize("ph_deg", [0.0, 90.0, 225.0])
@pytest.mark.parametrize("pp", [True, False], ids=["pp", "spherical"])
def test_impact_point_aims_at_the_origin(
    monkeypatch: pytest.MonkeyPatch, pp: bool, ph_deg: float
) -> None:
    """Check that the default sensor aims at the origin, any azimuth.

    The default sensor starts at the impact point with the zenith
    angle 180 - th_deg and the azimuth ph_deg + 180, that is along
    -(sin th cos ph, sin th sin ph, cos th): its ray must go through
    the origin on the ground, and its direct transmission must not
    depend on the azimuth.
    """
    monkeypatch.setattr(sg_module, "to_gpu", np.asarray)
    th_deg = 60.0
    x0, trans = _impact_init(_profile_1d(), 1, th_deg, ph_deg, 6371.0, pp)
    _, trans_0 = _impact_init(_profile_1d(), 1, th_deg, 0.0, 6371.0, pp)
    th, ph = np.radians(th_deg), np.radians(ph_deg)
    v = -np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                   np.cos(th)])
    to_origin = np.array([0.0, 0.0, 0.0 if pp else 6371.0]) - x0
    distance = float(np.linalg.norm(to_origin))
    assert np.dot(to_origin, v) > 0
    np.testing.assert_allclose(
        np.cross(to_origin, v), 0.0, atol=1e-5 * distance
    )
    np.testing.assert_allclose(trans, trans_0)


def test_isotropic_matrix_does_not_polarize() -> None:
    """Check the phase matrix of an isotropic scattering.

    In the Iparallel/Iperpendicular convention of the kernel, it turns
    any Stokes vector into (I/2, I/2, 0, 0): the intensity is kept and
    the polarization lost.
    """
    table = _isotropic(5)
    for name, value in [("a_P11", 0.5), ("a_P12", 0.5), ("a_P22", 0.5),
                        ("a_P33", 0.0), ("a_P43", 0.0), ("a_P44", 0.0)]:
        np.testing.assert_array_equal(table[name], value)


@pytest.mark.parametrize(
    ("n_photons", "n_obj", "expected"),
    [(1e9, 0, 1e6), (3e5, 0, 1e4), (3e5, 2, 3e4), (20, 0, 1), (5, 2, 1)],
)
def test_default_n_loop(n_photons: float, n_obj: int, expected: float) -> None:
    """Check the default n_loop, which must launch at least a photon."""
    assert _resolve_n_loop(n_photons, None, n_obj) == expected


@pytest.mark.parametrize(
    ("n_photons", "n_loop", "match"),
    [(0.5, None, "n_photons"), (100, 0.5, "n_loop"), (100, 0, "n_loop")],
)
def test_n_loop_refused(
    n_photons: float, n_loop: float | None, match: str
) -> None:
    """Check that a run of no photon per loop raises a ValueError."""
    with pytest.raises(ValueError, match=f"{match} must be at least 1"):
        _resolve_n_loop(n_photons, n_loop, 0)


def _profile_with_phases(marker: float) -> xr.Dataset:
    """Return a profile of 3 phase matrices, of which it uses the first.

    It is what a profile with a wavelength_phase grid wider than its
    wavelengths looks like. Every matrix is filled with the marker.
    """
    return xr.Dataset(
        {
            "OD_atm": (("wavelength", "z_atm"), np.zeros((2, 3))),
            "iphase_atm": (
                ("wavelength", "z_atm"), np.zeros((2, 3), dtype=np.int32)
            ),
            "phase_atm": (
                ("iphase", "nphamat", "theta_atm"), np.full((3, 1, 2), marker)
            ),
        },
        coords={"wavelength": [500.0, 510.0], "z_atm": [2.0, 1.0, 0.0]},
    )


def test_multi_profiles_keeps_each_profile_phases() -> None:
    """Check that each profile of multi_profiles scatters with its own.

    The second profile must index the first of its own matrices, which
    follow the 3 of the first profile, and not the second matrix of the
    first profile.
    """
    pro = multi_profiles([_profile_with_phases(1.0),
                          _profile_with_phases(2.0)])
    iphase = pro["iphase_atm"].values
    phase = pro["phase_atm"].values
    np.testing.assert_array_equal(phase[iphase[:2]], 1.0)
    np.testing.assert_array_equal(phase[iphase[2:]], 2.0)


def test_no_direct_transmission_of_a_3d_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that a 3D profile has no analytic direct transmission.

    Its OD_atm holds the extinction coefficients of its unique optical
    properties, in km-1, and not an optical depth column.
    """
    monkeypatch.setattr(sg_module, "to_gpu", np.asarray)
    profile = xr.Dataset(
        {"OD_atm": (("wavelength", "iopt"), [[0.0, 0.0, 5.0, 20.0]])},
        coords={"wavelength": [550.0], "iopt": np.arange(4)},
    )
    _, trans = _impact_init(profile, 1, 30.0, 0.0, 6371.0, True)
    assert trans is None


def _loops(launched: np.ndarray, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Return the weights of loops launching the given photons.

    Each photon weighs 1 with the probability 0.3, in the bins of
    shape (level, stokes, sensor, wavelength, theta, phi) = (1, 1, 2,
    1, 1, 1).
    """
    rng = np.random.default_rng(seed)
    weights = np.array(
        [rng.binomial(n, 0.3) for n in launched.ravel()], dtype=np.float64
    ).reshape(len(launched), 1, 1, 2, 1, 1, 1)
    return tuple(weights)


def test_ratio_stdev_of_equal_loops() -> None:
    """Check the standard deviation of loops of equal photon counts.

    It is then the one of the per loop results over the square root of
    the number of loops, as the former estimate of Smartg.run.
    """
    launched = np.full((40, 2), 1000)
    stats = _RatioStdev((1, 1, 2, 1, 1, 1))
    x = []
    for weights, n in zip(_loops(launched), launched, strict=True):
        stats.add(weights, n.reshape(2, 1))
        x.append(weights / n.reshape(1, 1, 2, 1, 1, 1))
    x = np.array(x)
    expected = np.sqrt((x**2).mean(axis=0) - x.mean(axis=0) ** 2) / np.sqrt(
        len(x)
    )
    np.testing.assert_allclose(stats.mean(), x.mean(axis=0))
    np.testing.assert_allclose(stats.sigma(), expected, rtol=1e-9)


def test_ratio_stdev_of_a_loop_without_photon() -> None:
    """Check the standard deviation of a bin missed by some loops."""
    launched = np.tile([[3, 1000]], (40, 1))
    launched[::4, 0] = 0
    stats = _RatioStdev((1, 1, 2, 1, 1, 1))
    for weights, n in zip(_loops(launched), launched, strict=True):
        stats.add(weights, n.reshape(2, 1))
    sigma = stats.sigma().ravel()
    assert np.all(np.isfinite(sigma))
    # about sqrt(p (1 - p) / N) for N photons weighing 1 with p = 0.3
    n_total = launched.sum(axis=0)
    np.testing.assert_allclose(
        sigma, np.sqrt(0.21 / n_total), rtol=0.5
    )


def _stats(mean: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a result and its stdev with level 0 set, level 1 empty."""
    shape = (2, 4, 1, 1, 3, 2)
    means = np.zeros(shape)
    sigmas = np.zeros(shape)
    means[0] = mean
    sigmas[0] = sigma
    return means, sigmas


def test_stdev_lim_reached() -> None:
    """Check the stop of a run that reached its relative error."""
    means, sigmas = _stats(1.0, 0.005)
    lim = StdevLim(err_rel_min=1.0, n_loop_min=10)
    assert not _stdev_lim_reached(lim, 9, means, sigmas)[0]
    assert _stdev_lim_reached(lim, 10, means, sigmas) == (True, 0.005, 0.5)


def test_stdev_lim_level_without_signal() -> None:
    """Check that a level without any signal never stops the run."""
    means, sigmas = _stats(1.0, 0.005)
    lim = StdevLim(err_rel_min=1.0, n_loop_min=10, level=1)
    assert not _stdev_lim_reached(lim, 100, means, sigmas)[0]


def test_thermal_forward_needs_alt_pp() -> None:
    """Check that the forward thermal mode refuses the fast move.

    Its kernel did not compile, and with the duplicate declaration
    removed it gave an upward flux at the TOA 23 times too high in the
    SEVIRI window. The refusal comes before any CUDA call.
    """
    with pytest.raises(ValueError, match="needs alt_pp=True"):
        Smartg(thermal=True)


def _hg_rows(g: list[float], theta: np.ndarray) -> np.ndarray:
    """Return Henyey-Greenstein matrices of 6 terms, one per g."""
    mu = np.cos(np.radians(theta))
    rows = []
    for gi in g:
        f11 = (1.0 - gi**2) / (1.0 + gi**2 - 2.0 * gi * mu) ** 1.5
        rows.append(np.stack([f11, 0 * f11, f11, 0 * f11, f11, f11]))
    return np.array(rows)


def _repeated_profile(kind: str) -> xr.Dataset:
    """Return a profile whose matrices repeat, one being unused.

    Matrices A, B, A, null and C: the third repeats the first, the
    fourth is null (a layer without particles), the last is not used.
    """
    theta = np.linspace(0.0, 180.0, 181)
    phase = _hg_rows([0.7, 0.5, 0.7, 0.0, 0.3], theta)
    phase[3] = 0.0
    iphase = np.array([[0, 1, 3], [2, 1, 3]], dtype=np.int32)
    z = [f"z_{kind}", np.array([2.0, 1.0, 0.0])]
    return xr.Dataset(
        {
            f"OD_{kind}": (("wavelength", z[0]), np.ones((2, 3))),
            f"iphase_{kind}": (("wavelength", z[0]), iphase),
            f"phase_{kind}": (("iphase", "nphamat", f"theta_{kind}"), phase),
        },
        coords={"wavelength": [500.0, 500.0], z[0]: z[1],
                f"theta_{kind}": theta},
    )


def _assert_same_upload(pro: xr.Dataset, kind: str) -> xr.Dataset:
    """Check that the deduplicated upload gives every layer its rows.

    Every wavelength and layer reads the same table row and the same
    cumulative distribution as from the whole table, bitwise, and the
    profile itself is left as it was. Return the uploaded profile.
    """
    before = pro.copy(deep=True)
    dev = _distinct_phase_profile(pro, kind)
    assert dev is not None
    xr.testing.assert_identical(pro, before)
    old_tab, old_cdf = _calc_phase_host(pro, 1000, 0.03, kind)
    new_tab, new_cdf = _calc_phase_host(dev, 1000, 0.03, kind)
    old_i = pro[f"iphase_{kind}"].values + 2
    new_i = dev[f"iphase_{kind}"].values + 2
    assert new_tab[new_i].tobytes() == old_tab[old_i].tobytes()
    assert new_cdf[new_i].tobytes() == old_cdf[old_i].tobytes()
    assert new_tab[:2].tobytes() == old_tab[:2].tobytes()
    return dev


@pytest.mark.parametrize("kind", ["atm", "oc"])
def test_distinct_phase_profile_uploads_the_same_rows(kind: str) -> None:
    """A repeated matrix is uploaded once, an unused one not at all."""
    dev = _assert_same_upload(_repeated_profile(kind), kind)
    assert dev[f"phase_{kind}"].shape[0] == 3


@pytest.mark.parametrize("kind", ["atm", "oc"])
def test_distinct_phase_profile_of_a_built_profile(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The profile arrays uploaded beside the table are unchanged.

    The profiles are built at 500 and 600 nm on a wavelength_phase grid
    that repeats 500 nm, as the internal bands of a KDIS channel did.
    """
    monkeypatch.setattr(sg_module, "to_gpu", np.asarray)
    wavelength_phase = [500.0, 500.0, 600.0]
    if kind == "atm":
        pro = Atm1D(
            "afglt", comp=[AerOPAC("desert", 0.1, 550.0)],
            wavelength_phase=wavelength_phase,
        ).calc([500.0, 600.0])
    else:
        pro = Water1D(comp=[HydrosolPR(
            0.5, n_theta=721, wavelength_phase=wavelength_phase
        )]).calc([500.0, 600.0])
    dev = _assert_same_upload(pro, kind)
    assert dev[f"phase_{kind}"].shape[0] < pro[f"phase_{kind}"].shape[0]
    wavelength = BandSet(pro.coords["wavelength"].values)
    old_prof, _ = _init_profile(wavelength, pro, kind)
    new_prof, _ = _init_profile(wavelength, dev, kind)
    for field in old_prof.dtype.names:
        if field != "iphase":
            assert new_prof[field].tobytes() == old_prof[field].tobytes()


def test_distinct_phase_profile_left_alone() -> None:
    """A profile with nothing to remove, or not to touch, is kept."""
    pro = _repeated_profile("atm")
    distinct = pro.isel(iphase=[0, 1, 3])
    distinct["iphase_atm"] = (
        ("wavelength", "z_atm"), np.array([[0, 1, 2], [0, 1, 2]], np.int32)
    )
    assert _distinct_phase_profile(distinct, "atm") is distinct
    no_index = pro.drop_vars("iphase_atm")
    assert _distinct_phase_profile(no_index, "atm") is no_index
    nan = pro.copy(deep=True)
    nan["phase_atm"][4, 0, 0] = np.nan
    assert _distinct_phase_profile(nan, "atm") is nan
    phase = pro["phase_atm"].values
    assert _distinct_phase_rows(phase, np.array([[0, 5]])) is None
    assert _distinct_phase_rows(phase, np.array([[0.0, 1.0]])) is None


def test_multi_profiles_of_one_profile_upload_its_matrices_once() -> None:
    """Blocks of the same matrices are uploaded once, others apart."""
    same = multi_profiles([_profile_with_phases(1.0),
                           _profile_with_phases(1.0)])
    dev = _distinct_phase_profile(same, "atm")
    assert dev is not None and dev["phase_atm"].shape[0] == 1
    other = multi_profiles([_profile_with_phases(1.0),
                            _profile_with_phases(2.0)])
    dev = _distinct_phase_profile(other, "atm")
    assert dev is not None and dev["phase_atm"].shape[0] == 2
    phase, iphase = dev["phase_atm"].values, dev["iphase_atm"].values
    np.testing.assert_array_equal(phase[iphase[:2]], 1.0)
    np.testing.assert_array_equal(phase[iphase[2:]], 2.0)


def _fake_host(
    tmp_path: Path, mem_max: str = "max", current: int = 0,
    swap_max: str = "max",
) -> dict[str, Path]:
    """Write a /proc/meminfo, a /proc/self/cgroup and a cgroup tree.

    The host has 16 GiB of RAM, 2 GiB of swap and 12 GiB available;
    the process runs in a scope of a slice, the scope limited to
    `mem_max` with `current` bytes used and 1 GiB of page cache.
    """
    gib = 2**30
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        f"MemTotal: {16 * gib // 1024} kB\n"
        f"MemAvailable: {12 * gib // 1024} kB\n"
        f"SwapTotal: {2 * gib // 1024} kB\n"
    )
    root = tmp_path / "cgroup"
    scope = root / "user.slice" / "run.scope"
    scope.mkdir(parents=True)
    (root / "user.slice" / "memory.max").write_text("max\n")
    (scope / "memory.max").write_text(f"{mem_max}\n")
    (scope / "memory.swap.max").write_text(f"{swap_max}\n")
    (scope / "memory.current").write_text(f"{current}\n")
    (scope / "memory.stat").write_text(f"anon 0\nfile {gib}\n")
    proc = tmp_path / "cgroup_of_self"
    proc.write_text("0::/user.slice/run.scope\n")
    return {"meminfo": meminfo, "proc_cgroup": proc, "cgroup_root": root}


def test_host_memory_of_the_host(tmp_path: Path) -> None:
    """Without a cgroup limit: MemAvailable and RAM plus swap."""
    gib = 2**30
    available, limit = _host_memory(**_fake_host(tmp_path))
    assert available == 12 * gib
    assert limit == 18 * gib


def test_host_memory_of_a_limited_scope(tmp_path: Path) -> None:
    """A cgroup limit lowers both, its page cache counted as free."""
    gib = 2**30
    files = _fake_host(tmp_path, mem_max=str(8 * gib), current=6 * gib,
                       swap_max="0")
    available, limit = _host_memory(**files)
    assert available == 3 * gib
    assert limit == 8 * gib


def test_host_memory_unknown(tmp_path: Path) -> None:
    """Without /proc/meminfo, nothing is known and nothing checked."""
    assert _host_memory(meminfo=tmp_path / "none") == (None, None)


GIB = 2**30


@pytest.mark.parametrize(
    ("kind", "device", "host", "where"),
    [
        ("atm", (15 * GIB, 16 * GIB), (None, None), "of the GPU"),
        ("oc", None, (40 * GIB, 18 * GIB), "the host can hold"),
    ],
)
def test_phase_tables_refused(
    kind: str, device: Any, host: Any, where: str
) -> None:
    """Tables larger than the GPU or than the host are refused."""
    medium = "atmosphere" if kind == "atm" else "ocean"
    # 1602 matrices of 1e6 angles, 28 bytes each: 41.8 GiB
    with pytest.raises(ValueError, match=f"{medium}.*{where}.*n_icdf"):
        _check_phase_memory(kind, 1602, 1_000_000, device, host)


def test_phase_tables_warned() -> None:
    """Tables that fit, but not in the free memory, are warned of."""
    with pytest.warns(UserWarning, match="may fail for lack of memory"):
        _check_phase_memory(
            "atm", 300, 1_000_000, (5 * GIB, 16 * GIB), (20 * GIB, 64 * GIB)
        )


def test_phase_tables_accepted() -> None:
    """Tables that fit go through silently, as do unknown memories."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_phase_memory(
            "atm", 97, 1_000_000, (15 * GIB, 16 * GIB), (30 * GIB, 64 * GIB)
        )
        _check_phase_memory("oc", 10**6, 10**6, None, (None, None))


def test_device_memory_without_a_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a CUDA context, the GPU is not checked."""

    def no_context() -> None:
        raise RuntimeError("no context")

    monkeypatch.setattr(sg_module.cuda, "mem_get_info", no_context,
                        raising=False)
    assert _device_memory() is None
    monkeypatch.setattr(sg_module.cuda, "mem_get_info",
                        lambda: (1, 2), raising=False)
    assert _device_memory() == (1, 2)


class _NoError:
    """Stand for the GPU error counters of a run."""

    def get(self) -> np.ndarray:
        return np.zeros(32, dtype=np.uint64)


LEVELS = {
    "up (TOA)": sg_module.UPTOA, "down (0+)": sg_module.DOWN0P,
    "up (0-)": sg_module.UP0M, "down (0-)": sg_module.DOWN0M,
    "up (0+)": sg_module.UP0P, "down (B)": sg_module.DOWNB,
}


@pytest.mark.parametrize("output_layers", [0, 3, 5])
@pytest.mark.parametrize("mode", ["cone", "le", "zip"])
@pytest.mark.parametrize("no_aer", [False, True])
def test_finalize_normalizes_the_written_levels(
    output_layers: int, mode: str, no_aer: bool
) -> None:
    """Each output is the slice of the whole normalized table, bitwise.

    Values, dtype and strides are those of a view of the six levels
    divided as a whole, as they were before only the written levels
    were normalized.
    """
    rng = np.random.default_rng(output_layers)
    n_sensor, n_lam = 2, 3
    le = None
    if mode == "cone":
        n_th, n_phi = 5, 6
    else:
        zip_le = mode == "zip"
        le = LocalEstimate(th_deg=[0.0, 30.0, 60.0],
                           phi_deg=[0.0, 90.0, 180.0], zip=zip_le)
        n_th, n_phi = 3, 1 if zip_le else 3
    shape = (6, 4, n_sensor, n_lam, n_th, n_phi)
    photons = rng.random(shape).astype(np.float32)
    photons_no_aer = rng.random(shape).astype(np.float32)
    n_in = rng.integers(10, 20, (n_sensor, n_lam)).astype(np.uint64)
    n_out = rng.integers(0, 9, (6, n_sensor, n_lam, n_th, n_phi)).astype(
        np.uint64)
    ds = _finalize(
        tab_photons_tot=photons,
        tab_photons_tot_no_aer=photons_no_aer if no_aer else None,
        tab_dist_tot=np.zeros(1),
        tab_hist_tot=None,
        wavelength=np.array([500.0, 600.0, 700.0]),
        n_photons_in_tot=n_in,
        errorcount=_NoError(),  # pyright: ignore[reportArgumentType]
        n_photons_out_tot=n_out,
        n_photons_out_tot_no_aer=n_out if no_aer else None,
        output_layers=output_layers,
        tab_trans_dir=np.zeros((n_sensor, n_lam)),
        tab_trans_dir_analytic=None,
        attrs={},
        prof_atm=None,
        prof_oc=None,
        sigma=None,
        horiz=0,
        le=le,
        no_aer_output=no_aer,
    )
    norm_geo: Any = 1.0
    if mode == "cone":
        _, _, omega = _calc_solid_angles(n_th, n_phi)
        norm_geo = 2.0 * omega.reshape((1, 1, -1, 1))
    norm = norm_geo * n_in.reshape((1, 1, n_sensor, n_lam, 1, 1))
    iphi: Any = 0 if mode == "zip" else slice(None)
    written = 0
    for direction, lvl in LEVELS.items():
        if f"I_{direction}" not in ds:
            continue
        written += 1
        for tab, suffix in [(photons, "")] + (
                [(photons_no_aer, ", no_aer")] if no_aer else []):
            ref = (tab.astype("float64") / norm).swapaxes(4, 5)
            for i, stk in enumerate("IQUV"):
                out = ds[f"{stk}_{direction}{suffix}"].values
                exp = ref[lvl, i, :, :, iphi, :]
                assert out.dtype == exp.dtype and out.shape == exp.shape
                assert out.tobytes() == np.ascontiguousarray(exp).tobytes()
                assert out.strides == exp.strides
    assert written == {0: 1, 3: 6, 5: 3}[output_layers]
