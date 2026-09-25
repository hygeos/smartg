"""GPU-free tests of the host-side helpers of smartg.smartg.

They check the arithmetic that Smartg.run does on the host around
the kernel, without building a Smartg object.
"""

import numpy as np
import pytest

from smartg.albedo import AlbedoCst, AlbedoMap
from smartg.sensor import Sensor, get_sensors_grid
from smartg.smartg import (
    _calc_solid_angles,
    _check_albedo_map_codes,
    _check_forward_raster,
)


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
