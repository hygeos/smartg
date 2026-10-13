"""Tests of the surface and albedo definitions, and of the kernel BRDFs.

The definitions are checked without a GPU. The RTLS and RPV BRDFs of
the kernel are checked against their formulas (Lucht et al. 2000;
Rahman et al. 1993) on backward runs through a transparent
atmosphere: every photon of a sensor reaches the same ground point and
makes there one local estimate towards the sun, so that the run
returns the BRF of that ground point exactly, to be compared with the
formula at its angles.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from smartg.albedo import AlbedoCst, AlbedoMap
from smartg.atmosphere import Atm1D
from smartg.sensor import Sensor
from smartg.smartg import LocalEstimate, Smartg
from smartg.surface import Environment, LambSurface, RPVSurface, RTLSSurface
from smartg.water import Water1D, WaterRw

EARTH_RADIUS = 6371.0
Z_TOA = 120.0
# Parameters whose BRF stays positive at every angle below: k0, k1p
# and k2p of RTLS, R = k0 (1 + k1p Kgeo + k2p Kvol), and r0, k, Theta
# and rho_c of RPV
RTLS = (0.3, 0.2, 0.6)
RPV = (0.1, 0.7, -0.3, 0.1)
N_PER_SENSOR = 10_000
# Relative tolerance on the BRF; at an exact hot spot the float square
# roots of the kernel take arguments of rounding size
RTOL = 1e-4
RTOL_HOT_SPOT = 1e-3
# The sensors, (view zenith angle, azimuth of the view) in degrees, and
# the sun directions, (zenith, azimuth) in degrees. In plane parallel
# geometry they meet at three exact hot spots (20, 30 and 45 degrees),
# in the principal plane, at a nadir view and with the sun at zenith.
PP_SENSORS = [(0.0, 180.0), (20.0, 180.0), (30.0, 180.0), (45.0, 225.0),
              (60.0, 180.0)]
PP_SUNS = [(20.0, 0.0), (30.0, 0.0), (45.0, 45.0), (30.0, 180.0),
           (60.0, 90.0), (0.0, 0.0), (50.0, 45.0), (50.0, 225.0)]
# Half of their sun directions lie in the principal plane of each view,
# on either side, at azimuths whose cosines and sines are not round
PLANE_SENSORS = [(vza, psi) for vza in (10.0, 30.0, 50.0, 70.0)
                 for psi in (105.0, 165.0)]
PLANE_SUNS = [(ths, phs)
              for ths in (5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0)
              for phs in (105.0, 285.0, 165.0, 345.0)]


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


# --------------------------------------------------------------------
# the BRDFs of the kernel
# --------------------------------------------------------------------


@pytest.fixture(scope="module")
def kernels() -> dict[bool, Smartg]:
    """Hold the backward kernels, by pp, compiled on first use."""
    return {}


def _surface(kind: str) -> RTLSSurface | RPVSurface:
    """Return the RTLS or the RPV surface of the tests."""
    if kind == "rtls":
        k0, k1p, k2p = RTLS
        return RTLSSurface(k0=AlbedoCst(k0), k1p=AlbedoCst(k1p),
                           k2p=AlbedoCst(k2p))
    r0, k, bt, rc = RPV
    return RPVSurface(r0=AlbedoCst(r0), k=AlbedoCst(k), bt=AlbedoCst(bt),
                      rc=AlbedoCst(rc))


def _brf(kind: str, ths: NDArray[np.float64], thv: NDArray[np.float64],
         phi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the BRF of the RTLS or the RPV surface of the tests.

    Parameters
    ----------
    kind : str
        'rtls' or 'rpv'.
    ths, thv, phi : ndarray
        The sun and view zenith angles and their relative azimuth,
        0 when the sun is behind the sensor (the hot spot), in
        radians.

    Returns
    -------
    ndarray
        For RTLS, k0 (1 + k1p Kgeo + k2p Kvol), with the RossThick and
        the LiSparse-R (h/b = 2, b/r = 1) kernels of Lucht et al.
        (2000). For RPV, the BRF of Rahman et al. (1993) with the hot
        spot parameter rho_c.
    """
    mus, muv = np.cos(ths), np.cos(thv)
    tans, tanv = np.tan(ths), np.tan(thv)
    cos_xi = mus * muv + np.sin(ths) * np.sin(thv) * np.cos(phi)
    d2 = np.maximum(tans**2 + tanv**2 - 2 * tans * tanv * np.cos(phi), 0)
    if kind == "rtls":
        k0, k1p, k2p = RTLS
        xi = np.arccos(np.clip(cos_xi, -1, 1))
        kvol = ((np.pi / 2 - xi) * cos_xi + np.sin(xi)) / (mus + muv)
        kvol -= np.pi / 4
        sec = 1 / mus + 1 / muv
        cos_t = np.clip(
            2 / sec * np.sqrt(d2 + (tans * tanv * np.sin(phi))**2), -1, 1)
        t = np.arccos(cos_t)
        big_o = (t - np.sin(t) * cos_t) * sec / np.pi
        kgeo = big_o - sec + (1 + cos_xi) / (2 * mus * muv)
        return k0 * (1 + k1p * kgeo + k2p * kvol)
    r0, k, bt, rc = RPV
    minnaert = (mus * muv * (mus + muv))**(k - 1)
    hg = (1 - bt**2) / (1 + bt**2 + 2 * bt * cos_xi)**1.5
    return r0 * minnaert * hg * (1 + (1 - rc) / (1 + np.sqrt(d2)))


def _expected(kind: str, sensors: list[tuple[float, float]],
              suns: list[tuple[float, float]], pp: bool
              ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return the expected I_up (TOA) of each sensor and sun.

    Parameters
    ----------
    kind : str
        'rtls' or 'rpv'.
    sensors, suns : list of tuple
        The (zenith, azimuth) angles, in degrees, of the views and of
        the sun directions.
    pp : bool
        Plane parallel geometry, the ground vertical being then the Z
        axis. In spherical geometry it is the one of the point where
        the view from TOA meets the sphere of radius EARTH_RADIUS.

    Returns
    -------
    expected : ndarray
        On (sensor, sun): the BRF at the angles of the ground point,
        times the cosine of the sun there in spherical geometry
        (reflectance=False, radiance for a solar irradiance of pi).
    hot : ndarray
        The exact hot spots, where the sun is behind the sensor.
    """
    vza, psi = np.radians(np.array(sensors)).T
    th = np.pi - vza
    # the photon leaves the sensor along v, the local estimate towards
    # the sun along s, as the kernel builds them
    v = np.stack([np.sin(th) * np.cos(psi), np.sin(th) * np.sin(psi),
                  np.cos(th)], axis=-1)
    tsun, psun = np.radians(np.array(suns)).T
    s = np.stack([np.cos(psun) * np.sin(tsun), np.sin(psun) * np.sin(tsun),
                  np.cos(tsun)], axis=-1)
    if pp:
        n = np.tile([0.0, 0.0, 1.0], (len(v), 1))
    else:
        pos = np.array([0.0, 0.0, EARTH_RADIUS + Z_TOA])
        b = v @ pos
        t = -b - np.sqrt(b**2 - pos @ pos + EARTH_RADIUS**2)
        ground = pos + t[:, None] * v
        n = ground / np.linalg.norm(ground, axis=1)[:, None]
    a = -v
    muv = np.sum(a * n, axis=1)[:, None]
    mus = n @ s.T
    # relative azimuth from the projections on the ground plane
    a_h = (a - muv * n)[:, None, :]
    s_h = s[None, :, :] - mus[:, :, None] * n[:, None, :]
    norm = np.linalg.norm(a_h, axis=-1) * np.linalg.norm(s_h, axis=-1)
    cos_phi = np.where(norm > 1e-12, np.sum(a_h * s_h, axis=-1)
                       / np.where(norm > 1e-12, norm, 1.0), 1.0)
    phi = np.arccos(np.clip(cos_phi, -1, 1))
    brf = _brf(kind, np.arccos(np.clip(mus, -1, 1)),
               np.arccos(np.clip(np.broadcast_to(muv, mus.shape), -1, 1)),
               phi)
    hot = (a @ s.T) > 1 - 1e-9
    return (brf if pp else brf * mus), hot


def _run_brdf(kernels: dict[bool, Smartg], kind: str, pp: bool,
              seed: int, sensors: list[tuple[float, float]],
              suns: list[tuple[float, float]]
              ) -> tuple[NDArray[Any], NDArray[Any], Any]:
    """Run the sensors and the suns, return I, N and the attributes.

    Parameters
    ----------
    kernels : dict
        The fixture of the compiled kernels.
    kind : str
        'rtls' or 'rpv'.
    pp : bool
        Plane parallel geometry, else spherical.
    seed : int
        The seed of the run.
    sensors, suns : list of tuple
        The (zenith, azimuth) angles, in degrees, of the views and of
        the sun directions.

    Returns
    -------
    i_up, n_up : ndarray
        I_up (TOA) and N_up (TOA) on (sensor, sun).
    attrs : dict
        The attributes of the output.
    """
    if pp not in kernels:
        kernels[pp] = Smartg(pp=pp, back=True)
    z0 = 0.0 if pp else EARTH_RADIUS
    m = kernels[pp].run(
        550.0,
        atmosphere=Atm1D("afglt", tau_r=0.0, no2=False, tco3=0.0,
                         tcwp=0.0),
        surface=_surface(kind),
        sensor=[Sensor(pos_z=z0 + Z_TOA, th_deg=180.0 - vza, ph_deg=psi,
                       loc="ATMOS") for vza, psi in sensors],
        le=LocalEstimate(th_deg=[t for t, _ in suns],
                         phi_deg=[p for _, p in suns], zip=True,
                         count_level=[0] * len(suns)),
        n_photons=N_PER_SENSOR * len(sensors),
        reflectance=pp,
        seed=seed,
        progress=False,
    )

    def values(name: str) -> NDArray[Any]:
        da = m[name].transpose("sensor index", ...)
        return da.values.reshape(len(sensors), len(suns))

    return values("I_up (TOA)"), values("N_up (TOA)"), m.attrs


def _check_brf(kernels: dict[bool, Smartg], kind: str, pp: bool,
               seed: int, sensors: list[tuple[float, float]],
               suns: list[tuple[float, float]]) -> None:
    """Assert that the runs return the BRF, every photon counted."""
    i_up, n_up, attrs = _run_brdf(kernels, kind, pp, seed, sensors, suns)
    expected, hot = _expected(kind, sensors, suns, pp)
    # a local estimate whose weight turned NaN is not counted
    assert np.array_equal(n_up.sum(axis=0),
                          np.full(len(suns), attrs["NPhotonIn_sum"]))
    rtol = np.where(hot, RTOL_HOT_SPOT, RTOL)
    ratio = i_up / expected
    assert np.all(np.abs(ratio - 1) <= rtol), (
        f"run / formula, by sensor (rows) and sun:\n{np.round(ratio, 5)}")


@pytest.mark.parametrize("kind", ["rtls", "rpv"])
def test_brdf_plane_parallel(kernels: dict[bool, Smartg], kind: str) -> None:
    """The plane parallel BRF is the formula, at the hot spot too.

    Up to v2.0.3 rounding turned an angle NaN at an exact hot spot: the
    RPV local estimate of 20 degrees was lost, and the LiSparse kernel
    of RTLS lost its overlap term, the BRF of 45 degrees 21.6 % low.
    """
    _check_brf(kernels, kind, pp=True, seed=151, sensors=PP_SENSORS,
               suns=PP_SUNS)


@pytest.mark.parametrize("kind", ["rtls", "rpv"])
def test_brdf_principal_plane(kernels: dict[bool, Smartg],
                              kind: str) -> None:
    """The BRF of the principal plane is counted, as the others.

    Up to v2.0.3 the cosine of the relative azimuth of a sun and a view
    in the same vertical plane could round beyond 1, and its arccosine
    turn NaN: the reflection was not counted, and 377 of the 3375 such
    directions of a grid of 5 degrees returned 0.
    """
    _check_brf(kernels, kind, pp=True, seed=181, sensors=PLANE_SENSORS,
               suns=PLANE_SUNS)
