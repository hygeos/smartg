"""GPU-free tests of the surface and albedo definitions."""

import numpy as np
import pytest

from smartg.albedo import AlbedoCst, AlbedoMap
from smartg.surface import Environment, LambSurface, RPVSurface, RTLSSurface
from smartg.water import Water1D, WaterRw


def _albedo_map() -> AlbedoMap:
    """Return a map of two albedos, one on each side of x = 0."""
    return AlbedoMap(
        np.array([[0], [1]]),
        np.array([0.0, 1e6]),
        np.array([1e6]),
        [AlbedoCst(0.1), AlbedoCst(0.5)],
    )


def test_albedo_map_refused_as_a_spectral_albedo() -> None:
    """An AlbedoMap is refused where a single spectral albedo is needed.

    It gives one spectral albedo per entry of its map, which the
    surfaces and the sea floor of a water profile cannot hold: they
    used to accept it, and Smartg.run or Water1D.calc to fail on it.
    """
    amap = _albedo_map()
    with pytest.raises(TypeError, match="Environment"):
        LambSurface(alb=amap)  # type: ignore
    with pytest.raises(TypeError, match="AlbedoMap"):
        RTLSSurface(k0=amap)  # type: ignore
    with pytest.raises(TypeError, match="AlbedoMap"):
        RPVSurface(k=amap)  # type: ignore
    with pytest.raises(TypeError, match="AlbedoMap"):
        Water1D(alb=amap)  # type: ignore
    with pytest.raises(TypeError, match="AlbedoMap"):
        WaterRw(alb=amap)  # type: ignore

    # it remains the albedo of an environment map, and a spectral
    # albedo remains accepted everywhere
    Environment(env=5, alb=amap)
    alb = AlbedoCst(0.2)
    LambSurface(alb=alb)
    RTLSSurface(k0=alb)
    RPVSurface(k=alb)
    WaterRw(alb=alb)
