"""GPU-free tests of the host-side helpers of smartg.smartg.

They check the arithmetic that Smartg.run does on the host around
the kernel, without building a Smartg object.
"""

import numpy as np
import pytest
import xarray as xr

from smartg import smartg as sg_module
from smartg.albedo import AlbedoCst, AlbedoMap
from smartg.sensor import Sensor, get_sensors_grid
from smartg.smartg import (
    LocalEstimate,
    StdevLim,
    _calc_solid_angles,
    _check_albedo_map_codes,
    _check_flat_surface_le,
    _check_forward_raster,
    _check_sensor_directions,
    _impact_init,
    _isotropic,
    _RatioStdev,
    _resolve_n_loop,
    _stdev_lim_reached,
    multi_profiles,
)
from smartg.surface import FlatSurface, RoughSurface


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
