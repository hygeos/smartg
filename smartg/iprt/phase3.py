"""IPRT phase 3 cases.

Phase 3 holds the fully spherical cases D1 to D6, with one layer, and
E1 to E6, with a vertically inhomogeneous atmosphere. Each case_*
function runs its case with SMART-G, saves the runs at the bottom
(BOA) and at the top (TOA) of the atmosphere in the intermediate files
iprt_phase3_<case>_boa.nc and iprt_phase3_<case>_toa.nc, and converts
them to the IPRT phase 3 netCDF format, iprt_phase3_<case>.nc.
smartg/tests/test_iprt_phase3.py compares the D1 to E5 results with
saved ones.

The two backward SMART-G kernels of the runs, spherical and plane
parallel, are compiled by the first run that needs them, once per
process, so that importing the module needs no GPU. They remain
reachable as the module attributes S1DB and S1DB_PP.

Key Functions
-------------
case_d1, case_d2, case_d3, case_d4, case_d4_bis, case_d5, case_d6
    Run a one layer case, D6_pp being the plane parallel D6.
case_e1, case_e2, case_e3, case_e4, case_e5
    Run a vertically inhomogeneous case.
case_e6_v1, case_e6_v2, case_e6_v3
    Run the camera case E6, in three versions.
to_iprt_output
    Convert the BOA and TOA runs of a case to the IPRT format.
aer2smartg
    Convert an IPRT aerosol or cloud file to the SMART-G format.
plot_polar_iprt
    Plot phase 3 I, Q, U and V matrices in polar view.
plot_camera_iprt
    Plot phase 3 I, Q, U and V matrices of the camera case.
"""

from collections.abc import Callable
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import geoclide as gc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from smartg.albedo import AlbedoCst
from smartg.atmosphere import AerOPAC, Atm1D, Cloud
from smartg.config import DIR_AUXDATA
from smartg.phase import (
    calc_iphase,
    is_native_theta,
    read_phase_cdf,
    union_theta_grid,
)
from smartg.sensor import Sensor
from smartg.smartg import LocalEstimate, Smartg
from smartg.surface import LambSurface, RoughSurface
from smartg.view import plot_polar_iquv
from smartg.xarray import drop_axes


@cache
def _kernel(pp: bool) -> Smartg:
    """Return the backward kernel of the runs, compiled on first use.

    Parameters
    ----------
    pp : bool
        The plane parallel kernel (alt_pp) rather than the spherical
        one.

    Returns
    -------
    Smartg
        The kernel, the same object for every later call.
    """
    if pp:
        return Smartg(back=True, double=True, bias=True, pp=True,
                      alt_pp=True)
    return Smartg(back=True, double=True, bias=True, pp=False)


def __getattr__(name: str) -> Smartg:
    """Give S1DB and S1DB_PP, the kernels of former versions, lazily."""
    if name == "S1DB":
        return _kernel(False)
    if name == "S1DB_PP":
        return _kernel(True)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


OPT_PROP_PATH_PHASE3 = DIR_AUXDATA / "IPRT" / "phase3" / "opt_prop"

STOKES = ("I", "Q", "U", "V")
# Depolarisation factor of every case
DEPOL = 0.03
EARTH_RADIUS = 6371.0

# Sun and viewing angles of the cases D1 to E5, in degrees. The IPRT
# azimuth angles cover 0 to 180 degrees.
SZA = np.array([30.0, 60.0, 80.0, 87.0, 90.0, 93.0, 96.0, 99.0])
SAA = np.array([0.0])
VZA = np.array([0.0, 9.0, 18.0, 26.0, 34.0, 41.0, 48.0, 54.0, 60.0, 65.0,
                70.0, 74.0, 78.0, 81.0, 84.0, 86.0, 88.0, 89.0, 90.0])
VAA = np.linspace(0.0, 180.0, 19)

# The one layer of the D cases, from its top to the ground, in km
Z_ONE_LAYER = np.array([120.0, 0.0])

# Sun and viewing angles of the camera case E6, in degrees: the sun
# positions are stored on the longitude axis of the output
SZA_E6 = np.array([50.0, 90.0, 110.0, 130.0])
VZA_E6 = np.arange(0.0, 1.2 + 0.04, 0.04)
# Distance of the E6 camera from the center of the Earth, in km
CAMERA_E6 = 3e5
# Number of pixels along each side of the E6 camera
N_PIXELS_E6 = 61


def reshape_sza_vaa_vza(
    m: xr.Dataset,
    sza: np.ndarray,
    vaa: np.ndarray,
    vza: np.ndarray,
) -> xr.Dataset:
    """Reorganize a D1 to E5 run output on the (sza, vaa, vza) grid.

    The run stacks the viewing directions in the sensor dimension and
    the sun positions in the Zenith angles dimension; each variable is
    rebuilt on the sza, vaa and vza dimensions instead.

    Parameters
    ----------
    m : xr.Dataset
        The output of Smartg.run. A legacy MLUT is converted.
    sza, vaa, vza : ndarray
        The sun zenith, viewing azimuth and viewing zenith angles of
        the run, in degrees.

    Returns
    -------
    xr.Dataset
        The output on the sza, vaa and vza dimensions.
    """
    if hasattr(m, "to_xarray"):
        m = m.to_xarray()  # legacy MLUT
    m = drop_axes(m, "Azimuth angles")
    for name in list(m.data_vars):
        dims = m[name].dims
        attrs = m[name].attrs
        if "Zenith angles" in dims and "sensor index" in dims:
            data = np.swapaxes(
                m[name].data.reshape(len(vza), len(vaa), len(sza)), 0, 2
            )
            new_dims: tuple[str, ...] = ("sza", "vaa", "vza")
        elif "Zenith angles" in dims:
            data = m[name].data
            new_dims = ("sza",)
        elif "sensor index" in dims:
            data = np.swapaxes(m[name].data.reshape(len(vza), len(vaa)),
                               0, 1)
            new_dims = ("vaa", "vza")
        else:
            continue
        m = m.drop_vars([name])
        m[name] = xr.Variable(new_dims, data, attrs=attrs)
    m = drop_axes(m, "Zenith angles", "sensor index")
    return m.assign_coords(sza=sza, vaa=vaa, vza=vza)


def _sensor(
    pos: tuple[float, float, float],
    th_deg: float,
    ph_deg: float,
    **kwargs: Any,
) -> Sensor:
    """Return an ATMOS sensor, of type 0 by default."""
    options: dict[str, Any] = {"loc": "ATMOS", "sensor_type": 0}
    options.update(kwargs)
    return Sensor(pos_x=pos[0], pos_y=pos[1], pos_z=pos[2], th_deg=th_deg,
                  ph_deg=ph_deg, **options)


def _coords(point: gc.Point) -> tuple[float, float, float]:
    """Return the x, y and z coordinates of a single point, in km."""
    return float(point.x), float(point.y), float(point.z)


def get_d1_to_e5_boa_sensors(
    vza: np.ndarray,
    phi: np.ndarray,
    earth_radius: float,
) -> list[Sensor]:
    """Return the ground sensors of the cases D1 to E5.

    Parameters
    ----------
    vza : ndarray
        The viewing zenith angles, in degrees.
    phi : ndarray
        The viewing azimuth angles, in degrees, in the SMART-G
        convention.
    earth_radius : float
        The altitude of the ground from the center of the Earth, in
        km: the Earth radius, or 0 in plane parallel geometry.

    Returns
    -------
    list of Sensor
        One sensor per (vza, phi) pair, vza varying slowest.
    """
    sensors = []
    for vza_boa in vza:
        # 89.9999 instead of 90, a sensor at z=0 looking at the horizon
        # being a problem
        if vza_boa == 90.0:
            vza_boa = 89.9999
        for phi_boa in phi:
            sensors.append(_sensor((0.0, 0.0, earth_radius), vza_boa, phi_boa))
    return sensors


def get_d1_to_e5_toa_sensors_old(
    vza: np.ndarray,
    phi: np.ndarray,
    earth_radius: float,
    z: np.ndarray,
) -> list[Sensor]:
    """Return TOA sensors of the cases D1 to E5, an older version.

    The sensors are placed where the viewing direction from the ground
    point meets the top of the atmosphere, with theta and not the
    local zenith angle theta' there. get_d1_to_e5_toa_sensors is the
    version in use.

    Parameters
    ----------
    vza, phi : ndarray
        The viewing zenith and azimuth angles, in degrees.
    earth_radius : float
        The Earth radius, in km.
    z : ndarray
        The altitudes of the atmosphere, in km.

    Returns
    -------
    list of Sensor
        One sensor per (vza, phi) pair, vza varying slowest.
    """
    sensors = []
    zeros = np.zeros(len(phi), dtype=np.float64)
    origin = gc.Point(zeros, zeros,
                      np.full(len(phi), earth_radius, dtype=np.float64))
    toa_layer = gc.Sphere(earth_radius + np.max(z))
    vza_bis = vza.copy()
    vza_bis[vza_bis == 90.0] = 89.9999
    vza_toa = 180.0 - vza_bis
    phi_toa = phi + 180.0
    for ivza in range(len(vza)):
        dirs = gc.ang2vec(theta=vza_bis[ivza], phi=phi)
        ds_geo = gc.calc_intersection(toa_layer, gc.Ray(o=origin, d=dirs))
        for ivaa in range(len(phi)):
            if not ds_geo["is_intersection"].values[ivaa]:
                raise ValueError(
                    "No intersection has been found for "
                    f"VZA={vza_bis[ivza]} and VAA={phi[ivaa]}. Please "
                    "check the input parameters."
                )
            phit = gc.Point(ds_geo["phit"].values[ivaa, :])
            if ivaa == 9:
                print("vza=", vza[ivza], "; phi=", phi[ivaa], " ; point=",
                      phit)
            sensors.append(_sensor(_coords(phit), vza_toa[ivza],
                                   phi_toa[ivaa]))
    return sensors


def get_d1_to_e5_toa_sensors(
    vza: np.ndarray,
    phi: np.ndarray,
    earth_radius: float,
    z: np.ndarray,
) -> list[Sensor]:
    """Return the TOA sensors of the cases D1 to E5.

    The sensors are above the ground sensors, at the top of the
    atmosphere, and look down with the local zenith angle theta'.

    Parameters
    ----------
    vza, phi : ndarray
        The viewing zenith and azimuth angles, in degrees.
    earth_radius : float
        The altitude of the ground from the center of the Earth, in
        km.
    z : ndarray
        The altitudes of the atmosphere, in km.

    Returns
    -------
    list of Sensor
        One sensor per (vza, phi) pair, vza varying slowest.
    """
    return [
        _sensor((0.0, 0.0, earth_radius + np.max(z)), 180.0 - vza_i,
                phi_i + 180.0)
        for vza_i in vza
        for phi_i in phi
    ]


def get_e6_toa_sensors(
    vza: np.ndarray,
    phi: np.ndarray,
    earth_radius: float,
    z: np.ndarray,
) -> list[Sensor]:
    """Return the TOA sensors of the first version of the E6 case.

    The sensors are where the directions seen by the camera meet the
    top of the atmosphere.

    Parameters
    ----------
    vza, phi : ndarray
        The viewing zenith and azimuth angles of the camera, in
        degrees.
    earth_radius : float
        The Earth radius, in km.
    z : ndarray
        The altitudes of the atmosphere, in km.

    Returns
    -------
    list of Sensor
        One sensor per (vza, phi) pair, vza varying slowest.
    """
    sensors = []
    zeros = np.zeros(len(phi), dtype=np.float64)
    origin = gc.Point(zeros, zeros,
                      np.full(len(phi), CAMERA_E6, dtype=np.float64))
    toa_layer = gc.Sphere(earth_radius + np.max(z))
    vza_toa = 180.0 - vza
    phi_toa = phi + 180.0
    for ivza in range(len(vza)):
        dirs = -gc.ang2vec(theta=vza[ivza], phi=phi)
        ds_geo = gc.calc_intersection(toa_layer, gc.Ray(o=origin, d=dirs))
        for ivaa in range(len(phi)):
            if not ds_geo["is_intersection"].values[ivaa]:
                raise ValueError(
                    "No intersection has been found for "
                    f"VZA={vza[ivza]} and VAA={phi[ivaa]}. Please check "
                    "the input parameters."
                )
            phit = gc.Point(ds_geo["phit"].values[ivaa, :])
            sensors.append(_sensor(_coords(phit), vza_toa[ivza],
                                   phi_toa[ivaa]))
    return sensors


def _iprt_output_path(case_name: str, output_dir: str | Path) -> Path:
    """Create output_dir and return the IPRT output path of a case."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"iprt_phase3_{case_name}.nc"


def to_iprt_output(
    case_name: str,
    sza: np.ndarray,
    saa: np.ndarray,
    vza: np.ndarray,
    vaa: np.ndarray,
    z: np.ndarray,
    overwrite: bool = True,
    output_dir: str | Path = "./",
) -> None:
    """Convert the BOA and TOA runs of a case to the IPRT format.

    The intermediate files iprt_phase3_<case_name>_boa.nc and
    iprt_phase3_<case_name>_toa.nc are gathered in
    iprt_phase3_<case_name>.nc, with the radiance and its standard
    deviation on the (zout, sza, saa, vza, vaa, stokes) dimensions.
    The case 'e6' only has a TOA run, at the camera altitude.

    Parameters
    ----------
    case_name : str
        Name of the case, e.g. 'd1' or 'd6_pp'.
    sza, saa, vza, vaa : ndarray
        The sun zenith and azimuth, and the viewing zenith and azimuth
        angles, in degrees.
    z : ndarray
        The altitudes of the atmosphere, in km.
    overwrite : bool
        Write the output file even when it exists.
    output_dir : str or Path
        Folder of the intermediate and output files.
    """
    output_dir = Path(output_dir)
    f_path = _iprt_output_path(case_name, output_dir)
    if not overwrite and f_path.exists():
        print(f"File {f_path} already exists. Skipping.")
        return

    is_e6 = case_name == "e6"
    runs = []
    if not is_e6:
        runs.append(xr.open_dataset(
            output_dir / f"iprt_phase3_{case_name}_boa.nc"
        ))
    runs.append(xr.open_dataset(
        output_dir / f"iprt_phase3_{case_name}_toa.nc"
    ))
    zout = np.array([CAMERA_E6]) if is_e6 else np.array([0.0, np.max(z)])
    ds = xr.Dataset(coords={f"{case_name}_sza": sza,
                            f"{case_name}_saa": saa,
                            f"{case_name}_vza": vza,
                            f"{case_name}_vaa": vaa,
                            f"{case_name}_zout": zout,
                            "stokes": np.arange(4, dtype=np.int32)})

    shape = (len(runs), len(sza), len(saa), len(vza), len(vaa), 4)
    radiance = np.zeros(shape, dtype=np.float32)
    std = np.zeros_like(radiance)
    for izout, run in enumerate(runs):
        for istk, stk in enumerate(STOKES):
            radiance[izout, :, 0, :, :, istk] = np.swapaxes(
                run[f"{stk}_up (TOA)"].values, 1, 2
            )
            std[izout, :, 0, :, :, istk] = np.swapaxes(
                run[f"{stk}_stdev_up (TOA)"].values, 1, 2
            )

    dims = [f"{case_name}_zout", f"{case_name}_sza", f"{case_name}_saa",
            f"{case_name}_vza", f"{case_name}_vaa", "stokes"]
    ds[f"radiance_{case_name}"] = xr.DataArray(radiance, dims=dims)
    ds[f"std_{case_name}"] = xr.DataArray(std, dims=dims)
    ds.to_netcdf(f_path)


def _to_iprt_output_e6(
    case_name: str,
    sza: np.ndarray,
    saa: np.ndarray,
    nx: int,
    ny: int,
    vecs: gc.Vector,
    is_sens: np.ndarray | None,
    overwrite: bool,
    output_dir: str | Path,
) -> None:
    """Convert the camera run of a case E6 to the IPRT format.

    See to_iprt_output_e6_v1 and to_iprt_output_e6_v2.
    """
    output_dir = Path(output_dir)
    f_path = _iprt_output_path(case_name, output_dir)
    if not overwrite and f_path.exists():
        print(f"File {f_path} already exists. Skipping.")
        return

    run = xr.open_dataset(output_dir / f"iprt_phase3_{case_name}_toa.nc")
    ds = xr.Dataset(coords={
        f"{case_name}_lat": saa,
        f"{case_name}_lon": sza,
        f"{case_name}_prow": np.arange(ny, dtype=np.int32),
        f"{case_name}_pcol": np.arange(nx, dtype=np.int32),
        f"{case_name}_zout": np.array([CAMERA_E6]),
        "stokes": np.arange(4, dtype=np.int32),
    })
    n_lat, n_lon = len(saa), len(sza)
    radiance = np.zeros((1, n_lat, n_lon, ny, nx, 4), dtype=np.float32)
    std = np.zeros_like(radiance)
    vza = np.zeros((ny, nx), dtype=np.float64)
    vaa = np.zeros_like(vza)

    values = [run[f"{stk}_up (TOA)"].values for stk in STOKES]
    stdevs = [run[f"{stk}_stdev_up (TOA)"].values for stk in STOKES]
    vecs_x, vecs_y, vecs_z = (np.asarray(vecs.x), np.asarray(vecs.y),
                              np.asarray(vecs.z))
    for ilon in range(n_lon):
        # ic counts the pixels, isens the sensors: a pixel whose
        # direction misses the Earth has no sensor in v1
        ic = 0
        isens = 0
        for ix in range(nx):
            for iy in range(ny):
                th, ph = gc.vec2ang(
                    gc.Vector(vecs_x[ic], vecs_y[ic], vecs_z[ic]),
                    vec_view="nadir",
                )
                if th == 0.0 or th == 180.0:
                    ph = 0.0
                vza[iy, ix] = th
                vaa[iy, ix] = ph
                if is_sens is None or is_sens[ic]:
                    index = ic if is_sens is None else isens
                    for istk in range(4):
                        radiance[0, 0, ilon, iy, ix, istk] = (
                            values[istk][index, 0, ilon]
                        )
                        std[0, 0, ilon, iy, ix, istk] = (
                            stdevs[istk][index, 0, ilon]
                        )
                    isens += 1
                ic += 1

    dims = [f"{case_name}_zout", f"{case_name}_lat", f"{case_name}_lon",
            f"{case_name}_prow", f"{case_name}_pcol", "stokes"]
    ds[f"radiance_{case_name}"] = xr.DataArray(radiance, dims=dims)
    ds[f"std_{case_name}"] = xr.DataArray(std, dims=dims)
    pixel_dims = [f"{case_name}_prow", f"{case_name}_pcol"]
    ds[f"vza_{case_name}"] = xr.DataArray(vza, dims=pixel_dims)
    ds[f"vaa_{case_name}"] = xr.DataArray(vaa, dims=pixel_dims)
    ds.to_netcdf(f_path)


def to_iprt_output_e6_v1(
    case_name: str,
    sza: np.ndarray,
    saa: np.ndarray,
    nx: int,
    ny: int,
    is_sens: np.ndarray,
    vecs: gc.Vector,
    overwrite: bool = True,
    output_dir: str | Path = "./",
) -> None:
    """Convert the camera run of case_e6_v1 to the IPRT format.

    Parameters
    ----------
    case_name : str
        Name of the case, 'e6_v1'.
    sza, saa : ndarray
        The sun zenith and azimuth angles, in degrees, stored on the
        longitude and latitude axes.
    nx, ny : int
        Number of pixel columns and rows of the camera.
    is_sens : ndarray of bool
        Whether each pixel, column by column, has a sensor: its
        direction meets the Earth.
    vecs : gc.Vector
        The viewing direction of each pixel, column by column.
    overwrite : bool
        Write the output file even when it exists.
    output_dir : str or Path
        Folder of the intermediate and output files.
    """
    _to_iprt_output_e6(case_name, sza, saa, nx, ny, vecs, is_sens,
                       overwrite, output_dir)


def to_iprt_output_e6_v2(
    case_name: str,
    sza: np.ndarray,
    saa: np.ndarray,
    nx: int,
    ny: int,
    vecs: gc.Vector,
    overwrite: bool = True,
    output_dir: str | Path = "./",
) -> None:
    """Convert the camera run of case_e6_v2 or v3 to the IPRT format.

    Parameters
    ----------
    case_name : str
        Name of the case, 'e6_v2' or 'e6_v3'.
    sza, saa : ndarray
        The sun zenith and azimuth angles, in degrees, stored on the
        longitude and latitude axes.
    nx, ny : int
        Number of pixel columns and rows of the camera.
    vecs : gc.Vector
        The viewing direction of each pixel, column by column, every
        pixel having a sensor.
    overwrite : bool
        Write the output file even when it exists.
    output_dir : str or Path
        Folder of the intermediate and output files.
    """
    _to_iprt_output_e6(case_name, sza, saa, nx, ny, vecs, None, overwrite,
                       output_dir)


def _run_and_save(
    sg: Smartg,
    path: Path,
    sensors: list[Sensor],
    n_directions: int,
    n_photons: float,
    wavelength: float | np.ndarray,
    le: LocalEstimate,
    surface: LambSurface | RoughSurface | None,
    pro: xr.Dataset,
    depol: float,
    earth_radius: float,
    n_icdf: int,
    theta_grid: str | None,
    seed: int,
    angles: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    """Run SMART-G with the settings of every phase 3 case and save it.

    Parameters
    ----------
    sg : Smartg
        The compiled SMART-G.
    path : Path
        The file of the output.
    sensors : list of Sensor
        The sensors.
    n_directions : int
        Number of viewing directions, n_photons being per direction.
    n_photons : float
        Number of photons per viewing direction.
    wavelength : float or ndarray
        The wavelength, in nm.
    le : LocalEstimate
        The sun directions.
    surface : LambSurface or RoughSurface, optional
        The surface.
    pro : xr.Dataset
        The atmosphere profile.
    depol : float
        The depolarisation factor.
    earth_radius : float
        The Earth radius, in km.
    n_icdf : int
        The n_icdf argument of Smartg.run.
    theta_grid : str, optional
        The theta_grid argument of Smartg.run.
    seed : int
        Seed of the random numbers, -1 for one taken from the clock.
    angles : tuple of 3 ndarray, optional
        The (sza, vaa, vza) angles to reshape the output with
        reshape_sza_vaa_vza. By default the output is saved as is.
    """
    m = sg.run(wavelength=wavelength, n_photons=n_directions * n_photons,
               n_loop=n_photons, atmosphere=pro, sensor=sensors,
               output_layers=1, le=le, surface=surface, xblock=64,
               xgrid=1024, beer=1, depol=depol, reflectance=False,
               earth_radius=earth_radius, stdev=True, progress=True,
               n_icdf=n_icdf, theta_grid=theta_grid, seed=seed)
    if angles is not None:
        m = reshape_sza_vaa_vza(m, *angles)
    m.to_netcdf(str(path))


def run_sim(
    boa_path: Path | None,
    toa_path: Path | None,
    sza: np.ndarray,
    vza: np.ndarray,
    vaa: np.ndarray,
    z: np.ndarray,
    wavelength: float | np.ndarray,
    le: LocalEstimate,
    surface: LambSurface | RoughSurface | None,
    pro: xr.Dataset,
    n_photons: float,
    depol: float = DEPOL,
    earth_radius: float = EARTH_RADIUS,
    n_icdf: int = 18001,
    pp: bool = False,
    is_e6: bool = False,
    theta_grid: str | None = None,
    seed: int = -1,
) -> None:
    """Run the BOA and TOA simulations of a case and save them.

    Parameters
    ----------
    boa_path, toa_path : Path, optional
        The files of the BOA and TOA runs. A run whose path is None is
        skipped, and the BOA run always is for E6.
    sza, vza, vaa : ndarray
        The sun zenith, viewing zenith and viewing azimuth angles, in
        degrees.
    z : ndarray
        The altitudes of the atmosphere, in km.
    wavelength : float or ndarray
        The wavelength, in nm.
    le : LocalEstimate
        The sun directions.
    surface : LambSurface or RoughSurface, optional
        The surface.
    pro : xr.Dataset
        The atmosphere profile.
    n_photons : float
        Number of photons per viewing direction.
    depol : float
        The depolarisation factor.
    earth_radius : float
        The altitude of the ground from the center of the Earth, in
        km: the Earth radius, or 0 in plane parallel geometry.
    n_icdf : int
        The n_icdf argument of Smartg.run.
    pp : bool
        Run in plane parallel geometry, with the alt_pp kernel.
    is_e6 : bool
        Run the TOA sensors of the first version of E6.
    theta_grid : str, optional
        The theta_grid argument of Smartg.run.
    seed : int
        Seed of the random numbers, -1 for one taken from the clock.
    """
    sg = _kernel(pp)
    run_radius = EARTH_RADIUS if pp else earth_radius
    # The IPRT azimuth angles are anti-clockwise
    phi = -vaa
    common = {"n_directions": len(vza) * len(vaa), "n_photons": n_photons,
              "wavelength": wavelength, "le": le, "surface": surface,
              "pro": pro, "depol": depol, "earth_radius": run_radius,
              "n_icdf": n_icdf, "theta_grid": theta_grid, "seed": seed,
              "angles": (sza, vaa, vza)}

    if boa_path is not None and not is_e6:
        sensors = get_d1_to_e5_boa_sensors(vza, phi, earth_radius)
        _run_and_save(sg, boa_path, sensors, **common)

    if toa_path is not None:
        if is_e6:
            sensors = get_e6_toa_sensors(vza, phi, earth_radius, z)
        else:
            sensors = get_d1_to_e5_toa_sensors(vza, phi, earth_radius, z)
        _run_and_save(sg, toa_path, sensors, **common)


def aer2smartg(
    fname: str | Path | xr.Dataset,
    n_theta: int | str = 1801,
    rh_or_reff: str | None = None,
    rh_reff: np.ndarray | None = None,
) -> xr.Dataset:
    """Convert an IPRT aerosol or cloud file to the SMART-G format.

    The optical properties of the file, given per wavelength and
    relative humidity or effective radius, are written in the format
    AerOPAC and Cloud read. A file with a single wavelength gets a
    second one, 0.1 nm above, with the same properties.

    Parameters
    ----------
    fname : str or Path or xr.Dataset
        The file to convert, or its content.
    n_theta : int or str
        The number of equally spaced angles of the phase matrix, or
        'native' for the union of the angle grids the file carries, on
        which the phase matrix is kept without resampling.
    rh_or_reff : str, optional
        The name of the humidity or effective radius axis: 'hum' or
        'reff'. By default the one of the file.
    rh_reff : ndarray, optional
        The values of that axis. By default those of the file.

    Returns
    -------
    xr.Dataset
        The converted dataset.
    """
    if isinstance(fname, xr.Dataset):
        ds = fname
    else:
        fname = Path(fname)
        ds = xr.open_dataset(fname)

    if "hum" in ds.variables:
        axis_name = "hum"
    elif "reff" in ds.variables:
        axis_name = "reff"
    else:
        raise ValueError("The file has neither a 'hum' nor a 'reff' axis!")
    if rh_reff is None:
        rh_reff = ds[axis_name].values
    if rh_or_reff is None:
        rh_or_reff = axis_name
    assert rh_reff is not None

    phase = ds["phase"][:, :, :, :].values

    n_stk = ds.nphamat.size
    n_rh_reff = rh_reff.size
    if is_native_theta(n_theta):
        # one grid per (wavelength, rh/reff, term), stored descending
        # and padded past ntheta; identical grids are merged first
        theta_all = ds["theta"].values
        ntheta_all = ds["ntheta"].values
        grids: list[np.ndarray] = []
        for idx in np.ndindex(ntheta_all.shape):
            nth = int(ntheta_all[idx])
            grid = np.sort(theta_all[idx][:nth].astype(np.float64))
            if not any(np.array_equal(grid, g) for g in grids):
                grids.append(grid)
        theta = union_theta_grid(grids)
    else:
        theta = np.linspace(0.0, 180.0, num=int(n_theta))
    n_wav_file = ds["wavelen"].size
    n_wav = max(n_wav_file, 2)
    if n_wav > n_wav_file:
        wavelength = np.concatenate((ds["wavelen"].values * 1e3,
                                     ds["wavelen"].values * 1e3 + 0.1))
    else:
        wavelength = ds["wavelen"].values * 1e3

    ext_out = np.zeros((n_rh_reff, n_wav), dtype=np.float64)
    ssa_out = np.zeros_like(ext_out)
    pha_out = np.zeros((n_rh_reff, n_wav, n_stk, len(theta)),
                       dtype=np.float64)

    for i_wavelength in range(n_wav_file):
        for irhreff in range(n_rh_reff):
            ext_out[irhreff, i_wavelength] = ds["ext"][i_wavelength, irhreff]
            ssa_out[irhreff, i_wavelength] = ds["ssa"][i_wavelength, irhreff]
            for istk in range(n_stk):
                # ntheta (wavelength, reff, stk)
                nth = ds["ntheta"][i_wavelength, irhreff, istk].data
                # theta (wavelength, reff, stk, ntheta)
                th = ds["theta"][i_wavelength, irhreff, istk, :].data
                pha_out[irhreff, i_wavelength, istk, :] = np.interp(
                    theta, th[:nth],
                    phase[i_wavelength, irhreff, istk, :nth],
                    period=np.inf,
                )

    if n_wav > n_wav_file:
        ext_out[:, -1] = ext_out[:, 0]
        ssa_out[:, -1] = ssa_out[:, 0]
        pha_out[:, -1, :, :] = pha_out[:, 0, :, :]

    ds_out = xr.Dataset(coords={rh_or_reff: rh_reff, "wav": wavelength,
                                "theta": theta})
    ds_out["ext"] = xr.DataArray(ext_out, dims=[rh_or_reff, "wav"])
    ds_out["ext"].attrs = {"description": "Extinction coefficient"}
    ds_out["ssa"] = xr.DataArray(ssa_out, dims=[rh_or_reff, "wav"])
    ds_out["ssa"].attrs = {"description": "Single scattering albedo"}
    ds_out["phase"] = xr.DataArray(pha_out,
                                   dims=[rh_or_reff, "wav", "stk", "theta"])
    ds_out["phase"].attrs = {"description": "scattering phase matrix"}
    name = fname.name if isinstance(fname, Path) else "none"
    if rh_or_reff == "hum":
        ds_out.attrs = {"name": name,
                        "H_mix_min": "0.",
                        "H_mix_max": "2",
                        "H_free_min": "2",
                        "H_free_max": "12",
                        "H_stra_max": "12",
                        "Z_mix": "8",
                        "Z_free": "8",
                        "Z_stra": "99"}
    else:
        ds_out.attrs = {"name": name}

    return ds_out


def plot_polar_iprt(
    i: np.ndarray,
    q: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    thetas: np.ndarray,
    phis: np.ndarray,
    change_q_sign: bool = False,
    change_u_sign: bool = False,
    change_v_sign: bool = False,
    max_i: float | None = None,
    max_q: float | None = None,
    max_u: float | None = None,
    max_v: float | None = None,
    cmap_i: str | Colormap | None = None,
    cmap_q: str | Colormap | None = None,
    cmap_u: str | Colormap | None = None,
    cmap_v: str | Colormap | None = None,
    title: str | None = None,
    save_fig: str | Path | None = None,
    sym: bool = False,
    min_i: float | None = None,
) -> None:
    """Plot phase 3 I, Q, U and V matrices in polar view.

    Wrapper of smartg.view.plot_polar_iquv for the phase 3 layout,
    where row j holds the viewing zenith angle thetas[j] and is drawn
    at its own rescaled radius; plot_polar_iquv takes the rows in the
    reverse order.

    Parameters
    ----------
    i, q, u, v : ndarray
        Stokes parameters, each of shape (ntheta, nphi).
    thetas : ndarray
        Viewing zenith angles of the rows, in degrees. They are
        rescaled to span the radius from 0 to 90.
    phis : ndarray
        Viewing azimuth angles of the columns, in degrees.
    change_q_sign, change_u_sign, change_v_sign : bool
        Multiply Q, U or V by -1.
    max_i, max_q, max_u, max_v : float, optional
        Upper bound of the colour scale of each panel. By default the
        largest absolute value of the panel is used. The Q, U and V
        panels are drawn from -max to +max, the I panel from min_i.
    cmap_i, cmap_q, cmap_u, cmap_v : str or Colormap, optional
        Colour map of each panel, 'jet' for I and 'RdBu_r' for the
        other panels by default.
    title : str, optional
        Title of the whole figure.
    save_fig : str or Path, optional
        Save the figure at this path, the extension giving the
        format, e.g. save_fig='myFigName.png'.
    sym : bool
        The IPRT azimuth angles cover 0 to 180 degrees; also plot, from
        180 to 360 degrees, their mirror image about the principal
        plane, with the sign of U and V changed.
    min_i : float, optional
        Lower bound of the colour scale of the I panel. By default 0,
        or -max_i when max_i is given.
    """
    q_sign = -1 if change_q_sign else 1
    u_sign = -1 if change_u_sign else 1
    v_sign = -1 if change_v_sign else 1
    iquv = (np.asarray(i)[::-1],
            np.asarray(q)[::-1] * q_sign,
            np.asarray(u)[::-1] * u_sign,
            np.asarray(v)[::-1] * v_sign)
    plot_polar_iquv(iquv, np.asarray(thetas), np.asarray(phis),
                    max_i=max_i, max_q=max_q, max_u=max_u, max_v=max_v,
                    min_i=min_i, cmap_i=cmap_i, cmap_q=cmap_q,
                    cmap_u=cmap_u, cmap_v=cmap_v, title=title,
                    save_fig=save_fig, sym=sym)


def plot_camera_iprt(
    i: np.ndarray,
    q: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    i_min: float = 0.0,
    i_max: float | None = None,
    i_cmap: str = "viridis",
    title: str | None = None,
    save_fig: str | Path | None = None,
) -> Figure:
    """Plot phase 3 I, Q, U and V matrices of the camera case.

    The four panels are in one row, Q, U and V centred on 0 with the
    coolwarm colormap. The matplotlib font size is set to 16.

    Parameters
    ----------
    i, q, u, v : ndarray
        Stokes parameters, each of shape (nrow, ncol), row 0 at the
        top.
    i_min, i_max : float, optional
        Bounds of the I colour scale. By default from 0 to the maximum
        of I.
    i_cmap : str
        Colour map of the I panel.
    title : str, optional
        Title of the whole figure.
    save_fig : str or Path, optional
        Save the figure at this path, the extension giving the format.

    Returns
    -------
    Figure
        The created figure.
    """
    matplotlib.rcParams.update({"font.size": 16})
    fig, axs = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True,
                            sharex=True, sharey=True)
    if title is not None:
        fig.suptitle(title, fontsize=20)

    panels = (
        (i, i_min, i_max, i_cmap),
        *((stk, -np.max(np.abs(stk)), np.max(np.abs(stk)), "coolwarm")
          for stk in (q, u, v)),
    )
    for ax, name, (stk, vmin, vmax, cmap) in zip(axs, STOKES, panels,
                                                strict=True):
        img = ax.imshow(stk, vmin=vmin, vmax=vmax, origin="upper",
                        cmap=plt.get_cmap(cmap))
        cbar = plt.colorbar(img)
        cbar.set_label(name, fontsize=20)

    if save_fig is not None:
        plt.savefig(save_fig)
    return fig


def _run_spherical_case(
    case_name: str,
    build: Callable[[], tuple[xr.Dataset, Any, float | np.ndarray]],
    z: np.ndarray,
    n_photons: float,
    overwrite: bool,
    output_dir: str | Path,
    seed: int,
    sza: np.ndarray = SZA,
    vza: np.ndarray = VZA,
    earth_radius: float = EARTH_RADIUS,
    pp: bool = False,
    theta_grid: str | None = None,
) -> None:
    """Run one of the cases D1 to E5 and write its IPRT output file.

    Parameters
    ----------
    case_name : str
        Name of the case, e.g. 'd1'.
    build : callable
        Return the atmosphere profile, the surface and the wavelength
        of the case. It is only called when a run is needed.
    z : ndarray
        The altitudes of the atmosphere, in km.
    n_photons : float
        Number of photons per viewing direction.
    overwrite : bool
        Run the simulations even when their intermediate files exist.
    output_dir : str or Path
        Folder of the intermediate files and of the IPRT output file.
    seed : int
        Seed of the random numbers, -1 for one taken from the clock.
    sza, vza : ndarray
        The sun and viewing zenith angles, in degrees.
    earth_radius : float
        The altitude of the ground from the center of the Earth, in
        km.
    pp : bool
        Run in plane parallel geometry.
    theta_grid : str, optional
        The theta_grid argument of Smartg.run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    boa_path = output_dir / f"iprt_phase3_{case_name}_boa.nc"
    toa_path = output_dir / f"iprt_phase3_{case_name}_toa.nc"
    run_boa = overwrite or not boa_path.exists()
    run_toa = overwrite or not toa_path.exists()

    if run_boa or run_toa:
        pro, surface, wavelength = build()
        # The IPRT azimuth angles are anti-clockwise
        le = LocalEstimate(th_deg=sza, phi_deg=-SAA,
                           count_level=np.zeros_like(sza, dtype=np.int32))
        run_sim(boa_path if run_boa else None,
                toa_path if run_toa else None, sza, vza, VAA, z,
                wavelength, le, surface, pro, n_photons,
                earth_radius=earth_radius, pp=pp, theta_grid=theta_grid,
                seed=seed)

    to_iprt_output(case_name, sza, SAA, vza, VAA, z, overwrite=overwrite,
                   output_dir=output_dir)


def _rayleigh_layer(tau_ray: float, wavelength: float) -> xr.Dataset:
    """Return the profile of a D case with a single Rayleigh layer."""
    mol_sca = np.array([0.0, tau_ray])[None, :]
    mol_abs = np.array([0.0, 0.0])[None, :]
    return Atm1D("afglt", grid=Z_ONE_LAYER.copy(), prof_ray=mol_sca,
                 prof_abs=mol_abs).calc(wavelength)


def _particle_layer(
    fname: str,
    tau: float,
    ssa: float,
    wavelength: float,
) -> xr.Dataset:
    """Return the profile of a D case with a layer of particles.

    The phase matrix is kept on the scattering angles of the file.

    Parameters
    ----------
    fname : str
        The optical properties file, in OPT_PROP_PATH_PHASE3.
    tau : float
        The optical depth of the layer.
    ssa : float
        The single scattering albedo of the particles.
    wavelength : float
        The wavelength, in nm.

    Returns
    -------
    xr.Dataset
        The profile.
    """
    mol_sca = np.array([0.0, 0.0])[None, :]
    mol_abs = np.array([0.0, 0.0])[None, :]
    z = Z_ONE_LAYER.copy()
    nz = len(z[1:])

    tau_ext = np.full_like(mol_sca, tau, dtype=np.float32)
    tau_ext[:, 0] = 0.0  # dtau at TOA equal to 0
    ssa_ext = np.full_like(mol_sca, ssa, dtype=np.float32)

    wavelengths = np.array([wavelength])
    file_phase = read_phase_cdf(OPT_PROP_PATH_PHASE3 / fname,
                                n_theta="native", normalize=True,
                                output_sg_ready=False)
    theta = file_phase["theta_atm"].values
    pha = np.zeros((len(wavelengths), nz, file_phase.shape[2], len(theta)),
                   dtype=np.float32)
    # Same phase matrix at every altitude (here only one)
    for iz in range(nz):
        pha[:, iz, :, :] = (
            file_phase.isel({file_phase.dims[1]: 0})
            # a single wavelength in the file: interp would give NaN
            .sel(wavelength_phase=wavelengths, method="nearest").data
        )
    phase = xr.DataArray(
        pha,
        dims=["wavelength_phase", "z_phase", "nphamat", "theta_atm"],
        coords={"wavelength_phase": wavelengths, "z_phase": z[1:],
                "theta_atm": theta},
    )
    pha_atm, ipha_atm = calc_iphase(phase, np.array([wavelengths]), z)
    phases = [
        xr.DataArray(pha_atm[ipha, :, :], dims=["nphamat", "theta_atm"],
                     coords={"theta_atm": theta})
        for ipha in range(pha_atm.shape[0])
    ]

    return Atm1D(
        "afglt", grid=z, prof_ray=mol_sca, prof_abs=mol_abs,
        prof_aer=(tau_ext, ssa_ext), prof_phases=(ipha_atm, phases),
    ).calc(wavelengths, phase=False)


def _opac_from_iprt_file(
    fname: str,
    tmp_name: str,
    *aer_args: Any,
    **aer_kwargs: Any,
) -> AerOPAC:
    """Return an AerOPAC component from an IPRT aerosol file.

    The file is converted with aer2smartg on its native angles, for a
    relative humidity of 0, and written in a temporary file for
    AerOPAC to read.

    Parameters
    ----------
    fname : str
        The IPRT aerosol file, in OPT_PROP_PATH_PHASE3.
    tmp_name : str
        The name of the temporary converted file.
    *aer_args, **aer_kwargs
        The other arguments of AerOPAC.

    Returns
    -------
    AerOPAC
        The aerosol component.
    """
    ds = aer2smartg(OPT_PROP_PATH_PHASE3 / fname, n_theta="native",
                    rh_or_reff="hum", rh_reff=np.array([0.0]))
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / tmp_name
        ds.to_netcdf(file_path)
        return AerOPAC(str(file_path), *aer_args, **aer_kwargs)


def _read_usstd_column(fname: str, column: int) -> np.ndarray:
    """Return a column of an IPRT US standard profile file."""
    return pd.read_csv(OPT_PROP_PATH_PHASE3 / fname, header=None,
                       usecols=[column], dtype=float, skiprows=1,
                       sep=r"\s+", comment="#").values


def _usstd_profile(
    wavelength: int,
    absorption: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the US standard molecular profile of the E cases.

    Parameters
    ----------
    wavelength : int
        The wavelength of the files, in nm: 320 or 450.
    absorption : bool
        Read the absorption optical depths. Without, they are zero.

    Returns
    -------
    z : ndarray
        The altitudes, in km, from the top.
    sca, abs_ : ndarray
        The Rayleigh scattering and absorption optical depths, of
        shape (1, z.size).
    """
    sca_file = f"tau_rayleigh_{wavelength}nm_usstd.dat"
    z = np.squeeze(_read_usstd_column(sca_file, 0))
    sca = _read_usstd_column(sca_file, 1).reshape(1, len(z))
    if absorption:
        abs_file = f"tau_absorption_{wavelength}nm_usstd.dat"
        abs_ = _read_usstd_column(abs_file, 1).reshape(1, len(z))
    else:
        abs_ = np.zeros_like(sca)
    return z, sca, abs_


def _opac_free_and_stratosphere() -> dict[str, float]:
    """Return the AerOPAC free troposphere and stratosphere heights."""
    return {"h_free_min": 2.0, "h_free_max": 2.0, "h_stra_min": 12.0,
            "h_stra_max": 12.0, "z_mix": 1e6, "rh_mix": 0.0}


def case_d1(n_photons: float = 1e8, overwrite: bool = True,
            output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case D1: a Rayleigh layer without surface.

    The layer has an optical depth of 0.5, at 550 nm.

    Parameters
    ----------
    n_photons : float
        Number of photons per viewing direction.
    overwrite : bool
        Run the simulations even when their intermediate files exist.
    output_dir : str or Path
        Folder of the intermediate files and of the IPRT output file
        iprt_phase3_d1.nc.
    seed : int
        Seed of the random numbers, -1 for one taken from the clock.
    """
    def build() -> tuple[xr.Dataset, None, float]:
        return _rayleigh_layer(0.5, 550.0), None, 550.0

    _run_spherical_case("d1", build, Z_ONE_LAYER, n_photons, overwrite,
                        output_dir, seed)


def case_d2(n_photons: float = 1e8, overwrite: bool = True,
            output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case D2: a Rayleigh layer on a Lambertian surface.

    The layer has an optical depth of 0.1, at 550 nm, and the surface
    an albedo of 0.3.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    def build() -> tuple[xr.Dataset, LambSurface, float]:
        return (_rayleigh_layer(0.1, 550.0),
                LambSurface(alb=AlbedoCst(0.3)), 550.0)

    _run_spherical_case("d2", build, Z_ONE_LAYER, n_photons, overwrite,
                        output_dir, seed)


def case_d3(n_photons: float = 1e8, overwrite: bool = True,
            output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case D3: a layer of spherical aerosols.

    Water soluble aerosols, of optical depth 0.2 and single scattering
    albedo 0.975683 at 350 nm, without surface.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    def build() -> tuple[xr.Dataset, None, np.ndarray]:
        return (_particle_layer("waso.mie.cdf", 0.2, 0.975683, 350.0),
                None, np.array([350.0]))

    _run_spherical_case("d3", build, Z_ONE_LAYER, n_photons, overwrite,
                        output_dir, seed, theta_grid="phase")


def case_d4(n_photons: float = 1e8, overwrite: bool = True,
            output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case D4: a layer of spheroidal aerosols.

    Aerosols of optical depth 0.2 and single scattering albedo
    0.787581 at 350 nm, without surface. case_d4_bis builds the same
    layer in another way.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    def build() -> tuple[xr.Dataset, None, np.ndarray]:
        return (_particle_layer("sizedistr_spheroid.cdf", 0.2, 0.787581,
                                350.0),
                None, np.array([350.0]))

    _run_spherical_case("d4", build, Z_ONE_LAYER, n_photons, overwrite,
                        output_dir, seed, theta_grid="phase")


def case_d4_bis(n_photons: float = 1e8, overwrite: bool = True,
                output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case D4, the aerosols being an AerOPAC component.

    The spheroid file is converted with aer2smartg, and the aerosols
    fill the whole layer with an optical depth of 0.2 at 350 nm.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    def build() -> tuple[xr.Dataset, None, np.ndarray]:
        wavelength = np.array([350.0])
        aer = _opac_from_iprt_file(
            "sizedistr_spheroid.cdf", "spheroid_d4.nc", 0.2, 350.0,
            h_mix_min=0.0, h_mix_max=120.0, h_free_min=120.0,
            h_free_max=120.0, h_stra_min=120.0, h_stra_max=120.0,
            z_mix=1e6, rh_mix=0.0,
        )
        pro = Atm1D(
            "afglt", comp=[aer], grid=Z_ONE_LAYER.copy(),
            prof_ray=np.array([0.0, 0.0])[None, :],
            prof_abs=np.array([0.0, 0.0])[None, :],
        ).calc(wavelength, phase=True, n_theta="native")
        return pro, None, wavelength

    _run_spherical_case("d4_bis", build, Z_ONE_LAYER, n_photons, overwrite,
                        output_dir, seed, theta_grid="phase")


def case_d5(n_photons: float = 1e8, overwrite: bool = True,
            output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case D5: a water cloud layer.

    The cloud has an optical depth of 5 and a single scattering albedo
    of 0.999979 at 800 nm, without surface.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    def build() -> tuple[xr.Dataset, None, np.ndarray]:
        return (_particle_layer("watercloud.mie.cdf", 5.0, 0.999979,
                                800.0),
                None, np.array([800.0]))

    _run_spherical_case("d5", build, Z_ONE_LAYER, n_photons, overwrite,
                        output_dir, seed, theta_grid="phase")


def _ocean_d6() -> RoughSurface:
    """Return the rough ocean surface of the D6 cases."""
    return RoughSurface(wind=2.0, brdf=True, wave_shadow=True, nh2o=1.33)


def case_d6(n_photons: float = 1e8, overwrite: bool = True,
            output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case D6: a Rayleigh layer above a rough ocean.

    The layer has an optical depth of 0.1, at 550 nm, and the wind
    speed is 2 m/s.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    def build() -> tuple[xr.Dataset, RoughSurface, float]:
        return _rayleigh_layer(0.1, 550.0), _ocean_d6(), 550.0

    _run_spherical_case("d6", build, Z_ONE_LAYER, n_photons, overwrite,
                        output_dir, seed)


def case_d6_pp(n_photons: float = 1e8, overwrite: bool = True,
               output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case D6 in plane parallel geometry.

    The sun zenith angles stop at 87 degrees and the viewing zenith
    angles at 89 degrees.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    def build() -> tuple[xr.Dataset, RoughSurface, float]:
        return _rayleigh_layer(0.1, 550.0), _ocean_d6(), 550.0

    # The ground altitude in plane parallel geometry is 0
    _run_spherical_case("d6_pp", build, Z_ONE_LAYER, n_photons, overwrite,
                        output_dir, seed, sza=SZA[:4].copy(),
                        vza=VZA[:-1].copy(), earth_radius=0.0, pp=True)


def case_e1(n_photons: float = 1e8, overwrite: bool = True,
            output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case E1: a US standard Rayleigh atmosphere.

    At 450 nm, without absorption nor surface.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    z, sca, abs_ = _usstd_profile(450, absorption=False)

    def build() -> tuple[xr.Dataset, None, float]:
        pro = Atm1D("afglt", grid=z, prof_ray=sca,
                    prof_abs=abs_).calc(450.0)
        return pro, None, 450.0

    _run_spherical_case("e1", build, z, n_photons, overwrite, output_dir,
                        seed)


def case_e2(n_photons: float = 1e8, overwrite: bool = True,
            output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case E2: a US standard atmosphere with absorption.

    At 320 nm, without surface.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    z, sca, abs_ = _usstd_profile(320)

    def build() -> tuple[xr.Dataset, None, float]:
        pro = Atm1D("afglt", grid=z, prof_ray=sca,
                    prof_abs=abs_).calc(320.0)
        return pro, None, 320.0

    _run_spherical_case("e2", build, z, n_photons, overwrite, output_dir,
                        seed)


def case_e3(n_photons: float = 1e8, overwrite: bool = True,
            output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case E3: desert aerosols in the boundary layer.

    Aerosols of optical depth 0.5 below 3 km, in a US standard
    atmosphere at 450 nm, without surface.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    z, sca, abs_ = _usstd_profile(450)

    def build() -> tuple[xr.Dataset, None, np.ndarray]:
        wavelength = np.array([450.0])
        desert = _opac_from_iprt_file(
            "desert.cdf", "desert_e3.nc", 0.5, wavelength[0],
            h_mix_min=0.0, h_mix_max=3.0, **_opac_free_and_stratosphere(),
        )
        pro = Atm1D(
            "afglt", comp=[desert], grid=z, prof_ray=sca, prof_abs=abs_
        ).calc(wavelength, phase=True, n_theta="native")
        return pro, None, wavelength

    _run_spherical_case("e3", build, z, n_photons, overwrite, output_dir,
                        seed, theta_grid="phase")


def case_e4(n_photons: float = 1e8, overwrite: bool = True,
            output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case E4: desert and sulfate aerosols.

    The desert aerosols of E3, and sulfate aerosols of optical depth
    0.05 between 20 and 21 km, at 450 nm, without surface.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    z, sca, abs_ = _usstd_profile(450)

    def build() -> tuple[xr.Dataset, None, np.ndarray]:
        wavelength = np.array([450.0])
        desert = _opac_from_iprt_file(
            "desert.cdf", "desert_e4.nc", 0.5, wavelength[0],
            h_mix_min=0.0, h_mix_max=3.0, **_opac_free_and_stratosphere(),
        )
        sulfate = _opac_from_iprt_file(
            "sulfate.cdf", "sulfate_e4.nc", 0.05, wavelength[0],
            h_mix_min=20.0, h_mix_max=21.0,
            **_opac_free_and_stratosphere(),
        )
        pro = Atm1D(
            "afglt", comp=[desert, sulfate], grid=z, prof_ray=sca,
            prof_abs=abs_, pfgrid=[120.0, 21.0, 20.0, 3.0, 0.0],
        ).calc(wavelength, phase=True, n_theta="native")
        return pro, None, wavelength

    _run_spherical_case("e4", build, z, n_photons, overwrite, output_dir,
                        seed, theta_grid="phase")


def case_e5(n_photons: float = 1e8, overwrite: bool = True,
            output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT case E5: an ice cloud in a Rayleigh atmosphere.

    The phase matrix of the cloud is kept on the angle grid of the
    ic.ghm.baum.cdf file (498 angles, 0.01 degree apart in the forward
    peak), from the conversion of the file to the GPU tables.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    z, sca, abs_ = _usstd_profile(450)

    def build() -> tuple[xr.Dataset, None, np.ndarray]:
        wavelength = np.array([450.0])
        ds_ice = xr.open_dataset(OPT_PROP_PATH_PHASE3 / "ic.ghm.baum.cdf")
        # The wavelengths of the file are in micrometers: only those
        # between 400 and 500 nm are kept
        wavelengths_file = ds_ice.wavelen.values
        nlam = np.squeeze(np.argwhere(
            np.logical_and(wavelengths_file >= 0.4, wavelengths_file <= 0.5)
        ))
        # keep the angles of the file instead of resampling them
        ds_converted = aer2smartg(ds_ice.sel(nlam=nlam), n_theta="native")
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "ic_ghm_baum_e5.nc"
            ds_converted.to_netcdf(file_path)
            cloud = Cloud(str(file_path), reff=50.0, zmin=10.0, zmax=11.0,
                          tau_ref=1.0, w_ref=wavelength[0])
        pro = Atm1D(
            "afglt", comp=[cloud], grid=z, prof_ray=sca, prof_abs=abs_
        ).calc(wavelength, phase=True, n_theta="native")
        return pro, None, wavelength

    # The GPU tables adopt the angles of the phase matrices
    _run_spherical_case("e5", build, z, n_photons, overwrite, output_dir,
                        seed, theta_grid="phase")


def _profile_e6(camera_level: bool = False
                ) -> tuple[xr.Dataset, RoughSurface, np.ndarray,
                           np.ndarray]:
    """Return the atmosphere and the surface of the E6 cases.

    The US standard atmosphere at 450 nm, above a rough ocean with a
    wind speed of 5 m/s.

    Parameters
    ----------
    camera_level : bool
        Extend the atmosphere, without optical depth, up to the camera.

    Returns
    -------
    pro : xr.Dataset
        The profile.
    surface : RoughSurface
        The ocean.
    wavelength : ndarray
        The wavelength, in nm.
    z : ndarray
        The altitudes of the atmosphere, in km.
    """
    z, sca, abs_ = _usstd_profile(450)
    if camera_level:
        z = np.concatenate((np.array([CAMERA_E6]), z))
        sca = np.concatenate((np.array([[0.0]]), sca), axis=1)
        abs_ = np.concatenate((np.array([[0.0]]), abs_), axis=1)
    wavelength = np.array([450.0])
    pro = Atm1D("afglt", grid=z, prof_ray=sca, prof_abs=abs_).calc(wavelength)
    surface = RoughSurface(wind=5.0, brdf=True, wave_shadow=True, nh2o=1.33)
    return pro, surface, wavelength, z


def case_e6_old(n_photons: float = 1e8, overwrite: bool = True,
                output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the first version of the IPRT camera case E6.

    The camera at 300 000 km is described by the directions (vza, vaa)
    it sees, 0 to 1.2 degrees in zenith and 0 to 360 degrees in
    azimuth, and the sensors are where they meet the top of the
    atmosphere. The case_e6_v1 to v3 versions use a pixel grid.

    Parameters
    ----------
    n_photons, overwrite, output_dir, seed
        See case_d1.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    toa_path = output_dir / "iprt_phase3_e6_toa.nc"
    vaa = np.arange(0.0, 360.0 + 10, 10)
    z = _usstd_profile(450)[0]

    if overwrite or not toa_path.exists():
        pro, surface, wavelength, z = _profile_e6()
        le = LocalEstimate(th_deg=SZA_E6, phi_deg=-SAA,
                           count_level=np.zeros_like(SZA_E6,
                                                     dtype=np.int32))
        run_sim(None, toa_path, SZA_E6, VZA_E6, vaa, z, wavelength, le,
                surface, pro, n_photons, is_e6=True, seed=seed)

    to_iprt_output("e6", SZA_E6, SAA, VZA_E6, vaa, z, overwrite=overwrite,
                   output_dir=output_dir)


def _pixel_directions_e6(vza: np.ndarray, nx: int, ny: int) -> gc.Vector:
    """Return the viewing directions of the pixels of the E6 camera.

    The pixel centers are where the directions vza, in the plane of
    the sun, meet a ground plane 1 km below the camera, mirrored on
    both sides.

    Parameters
    ----------
    vza : ndarray
        The zenith angles of the centers, from the camera axis, in
        degrees.
    nx, ny : int
        Number of pixel columns and rows, 2 * vza.size - 1.

    Returns
    -------
    gc.Vector
        The unit direction of each pixel, column by column, the rows
        from the top left pixel.
    """
    nvza = len(vza)
    zeros = np.zeros(nvza, dtype=np.float64)
    camera = gc.Point(zeros, zeros, np.full(nvza, 1, dtype=np.float64))
    dirs = -gc.ang2vec(theta=vza, phi=180.0)
    ground = gc.BBox(p1=gc.Point(-np.inf, -np.inf, 0.0),
                     p2=gc.Point(np.inf, np.inf, 0.0))
    ds = gc.calc_intersection(ground, gc.Ray(o=camera, d=dirs))
    coord = np.concatenate((-(ds["phit"][1:, 0].values)[::-1],
                            ds["phit"][:, 0].values))

    n_sensors = nx * ny
    pts = np.zeros((n_sensors, 3), dtype=np.float64)
    pts[:, 0] = np.repeat(coord[:nx], ny)
    # to begin from the top left pixel
    pts[:, 1] = np.tile(coord[::-1][:ny], nx)
    zeros_ini = np.zeros(n_sensors, dtype=np.float64)
    points_ini = gc.Point(zeros_ini, zeros_ini,
                          np.full(n_sensors, 1, dtype=np.float64))
    return gc.normalize(gc.Point(pts) - points_ini)


def _direction_angles(vecs: gc.Vector, index: int) -> tuple[float, float]:
    """Return the zenith and azimuth angles of a direction, in degrees.

    The azimuth angle is 0 along the vertical.
    """
    th, ph = gc.vec2ang(gc.Vector(np.asarray(vecs.x)[index],
                                  np.asarray(vecs.y)[index],
                                  np.asarray(vecs.z)[index]))
    if th == 0.0 or th == 180.0:
        ph = 0.0
    return float(th), float(ph)


def _run_e6(
    sg: Smartg,
    sensors: list[Sensor],
    toa_path: Path,
    n_photons: float,
    pro: xr.Dataset,
    surface: RoughSurface,
    wavelength: np.ndarray,
    seed: int,
) -> None:
    """Run and save the camera run of an E6 version.

    Parameters
    ----------
    sg : Smartg
        The compiled SMART-G.
    sensors : list of Sensor
        The camera sensors.
    toa_path : Path
        The file of the run.
    n_photons : float
        Number of photons per pixel. The run launches
        N_PIXELS_E6 * N_PIXELS_E6 times it, whatever the number of
        sensors.
    pro : xr.Dataset
        The atmosphere profile.
    surface : RoughSurface
        The ocean.
    wavelength : ndarray
        The wavelength, in nm.
    seed : int
        Seed of the random numbers.
    """
    le = LocalEstimate(th_deg=SZA_E6, phi_deg=-SAA,
                       count_level=np.zeros_like(SZA_E6, dtype=np.int32))
    _run_and_save(sg, toa_path, sensors, N_PIXELS_E6 * N_PIXELS_E6,
                  n_photons, wavelength, le, surface, pro, DEPOL,
                  EARTH_RADIUS, 18001, None, seed)


def case_e6_v1(n_photons: float = 1e8, overwrite: bool = True,
               output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT camera case E6, one direction per pixel center.

    The sensors are where the pixel directions meet the top of the
    atmosphere; the pixels whose direction misses the Earth have none.

    Parameters
    ----------
    n_photons : float
        Number of photons per pixel of the camera. The photons of the
        pixels without a sensor go to the others: the 2997 sensors of
        the 3721 pixels receive about 1.24 times n_photons each.
    overwrite, output_dir, seed
        See case_d1.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    toa_path = output_dir / "iprt_phase3_e6_v1_toa.nc"
    nx = ny = N_PIXELS_E6
    n_sensors = nx * ny
    pro, surface, wavelength, z = _profile_e6()
    vecs = _pixel_directions_e6(VZA_E6, nx, ny)

    is_sens = np.full(n_sensors, False, dtype=np.bool)
    zeros = np.zeros(n_sensors, dtype=np.float64)
    origin = gc.Point(zeros, zeros,
                      np.full(n_sensors, CAMERA_E6, dtype=np.float64))
    toa_layer = gc.Sphere(EARTH_RADIUS + np.max(z))
    ds_toa = gc.calc_intersection(toa_layer, gc.Ray(o=origin, d=vecs))
    sensors = []
    for isens in range(n_sensors):
        if ds_toa["is_intersection"].values[isens]:
            is_sens[isens] = True
            phit = gc.Point(ds_toa["phit"].values[isens, :])
            th, ph = _direction_angles(vecs, isens)
            sensors.append(_sensor(_coords(phit), th, ph))

    if overwrite or not toa_path.exists():
        _run_e6(_kernel(False), sensors, toa_path, n_photons, pro, surface,
                wavelength, seed)

    to_iprt_output_e6_v1("e6_v1", SZA_E6, SAA, nx, ny, is_sens, vecs,
                         overwrite=overwrite, output_dir=output_dir)


def case_e6_v2(n_photons: float = 1e8, overwrite: bool = True,
               output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT camera case E6 with a field of view per pixel.

    Every pixel is a sensor of type 1 at the camera, with a field of
    view of 0.04 degree, run with the 3D objects kernel.

    Parameters
    ----------
    n_photons : float
        Number of photons per pixel.
    overwrite, output_dir, seed
        See case_d1.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    toa_path = output_dir / "iprt_phase3_e6_v2_toa.nc"
    nx = ny = N_PIXELS_E6
    pro, surface, wavelength, _ = _profile_e6()
    vecs = _pixel_directions_e6(VZA_E6, nx, ny)
    sensors = [
        _sensor((0.0, 0.0, CAMERA_E6), *_direction_angles(vecs, isens),
                sensor_type=1, fov=0.04, cell_size=-2)
        for isens in range(nx * ny)
    ]

    if overwrite or not toa_path.exists():
        sg = Smartg(back=True, double=True, bias=True, pp=False, obj3d=True)
        _run_e6(sg, sensors, toa_path, n_photons, pro, surface,
                wavelength, seed)

    to_iprt_output_e6_v2("e6_v2", SZA_E6, SAA, nx, ny, vecs,
                         overwrite=overwrite, output_dir=output_dir)


def case_e6_v3(n_photons: float = 1e8, overwrite: bool = True,
               output_dir: str | Path = "./", seed: int = -1) -> None:
    """Run the IPRT camera case E6, the atmosphere up to the camera.

    As case_e6_v2, with the spherical backward kernel, the atmosphere
    being extended up to the camera.

    Parameters
    ----------
    n_photons : float
        Number of photons per pixel.
    overwrite, output_dir, seed
        See case_d1.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    toa_path = output_dir / "iprt_phase3_e6_v3_toa.nc"
    nx = ny = N_PIXELS_E6
    pro, surface, wavelength, _ = _profile_e6(camera_level=True)
    vecs = _pixel_directions_e6(VZA_E6, nx, ny)
    sensors = [
        _sensor((0.0, 0.0, CAMERA_E6), *_direction_angles(vecs, isens),
                sensor_type=1, fov=0.04)
        for isens in range(nx * ny)
    ]

    if overwrite or not toa_path.exists():
        _run_e6(_kernel(False), sensors, toa_path, n_photons, pro, surface,
                wavelength, seed)

    # The v2 conversion applies to v3
    to_iprt_output_e6_v2("e6_v3", SZA_E6, SAA, nx, ny, vecs,
                         overwrite=overwrite, output_dir=output_dir)


if __name__ == "__main__":
    OUTPUT_DIR = "./res_iprt_phase3_1e8photons_v5/"
    case_e6_v1(n_photons=1e8, overwrite=False, output_dir=OUTPUT_DIR)
    case_e6_v2(n_photons=1e8, overwrite=False, output_dir=OUTPUT_DIR)
    case_e6_v3(n_photons=1e6, overwrite=False, output_dir=OUTPUT_DIR)
