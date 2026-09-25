"""Sensor definition and creation helpers.

Key Classes
-----------
Sensor
    Definition of a sensor (position, viewing direction, type, ...).

Key Functions
-------------
get_sensor
    Build a sensor on the atmospheric boundary from view angles.
get_sensors_grid
    Create one sensor per cell of a regular (x, y) raster.
"""

from __future__ import annotations

import warnings
from typing import cast

import geoclide as gc
import numpy as np
from numpy.typing import NDArray

from smartg.grid3d import Grid3D, locate_voxel_index

# Localization names, at the index of their device code, see
# communs.h. The empty names are the codes a sensor cannot take.
LOC_CODE: list[str] = ['', 'ATMOS', 'SURF0P', 'SURF0M', '', '',
                       'OCEAN', 'SEAFLOOR', 'OBJSURF']


class Sensor:
    """Definition of a sensor.

    A sensor is the point of the scene the photons are launched from,
    with a direction, a localization and the radiative quantity it
    estimates. Its parameters are gathered in the ``dict`` attribute,
    keyed and ordered like the fields of the sensor structure of the
    CUDA kernel.

    Parameters
    ----------
    pos_x : float, optional
        The sensor position along the x axis (km). Default 0.
    pos_y : float, optional
        The sensor position along the y axis (km). Default 0.
    pos_z : float, optional
        The sensor position along the z axis (km). It is the altitude
        in plane-parallel geometry and the distance from the Earth
        center in spherical-shell geometry. Default 0.
    th_deg : float, optional
        The source (forward mode) or viewing (backward mode) zenith
        angle, in degrees. Greater than 90 for a downward looking
        direction, smaller than 90 for an upward one. Default 180
        (nadir), the direction the default loc 'SURF0P' needs.
    ph_deg : float, optional
        The source (forward mode) or viewing (backward mode) azimuth
        angle, in degrees. Default 180.
    loc : str, optional
        Localization of the sensor. Possibilities are:

        * 'SURF0P' -> Just above the surface (air side), looking down
          at it (th_deg > 90). Default value.
        * 'SURF0M' -> Just below the surface (water side), looking up
          at it (th_deg < 90).
        * 'ATMOS' -> In the atmosphere, looking in any direction. At
          pos_z=0 it looks up from just above the surface.
        * 'OCEAN' -> In the ocean, looking in any direction. At
          pos_z=0 it looks down from just below the surface.
        * 'SEAFLOOR' -> On the sea floor, looking down at it
          (th_deg > 90).
        * 'OBJSURF' -> Start from a 3d object surface.

        The photons of a sensor on an interface ('SURF0P', 'SURF0M'
        or 'SEAFLOOR') meet it at once, so the sensor must look at
        it, with the whole cone of a flux sensor: `Smartg.run`
        refuses it otherwise.
    fov : float, optional
        The field of view in degrees. Only for a flux sensor, it is
        forced to 0 for a radiance one. Default 0.
    sensor_type : int, optional
        The radiative quantity type. Three possibilities:

        * 0 -> Radiance (default).
        * 1 -> Planar flux.
        * 2 -> Spherical flux.
    icell : int, optional
        The box index where the sensor is located. Only for
        simulations with a 3D atmosphere. Default 0.
    ilam_0 : int, optional
        The index of the first wavelength seen by the sensor. Default
        -1, where the sensor sees all the wavelengths and ilam_1 is
        ignored.
    ilam_1 : int, optional
        The index that stops, excluded, the wavelengths seen by the
        sensor. Default -1.
    direction : geoclide.Vector, optional
        The sensor direction given as a vector. When provided, th_deg
        and ph_deg are deduced from it. Default None.
    cell_size : float, optional
        The side (km) of the square cell, centered on (pos_x, pos_y),
        where the photon start positions are drawn. Default -1, where
        every photon starts at the sensor position. The special value
        -2 moves the start position to the intersection with the top
        of atmosphere sphere, in spherical geometry with 3D objects.
        In a forward run in a 3D atmosphere, the sensors are also the
        raster the photons leaving the domain are counted on: they
        must form a complete raster of equal cells, x varying first,
        as `get_sensors_grid` builds it, and the photons leaving
        outside it are not counted.

    Attributes
    ----------
    dict : dict
        The sensor parameters, keyed and ordered like the fields of
        the sensor structure of the CUDA kernel. The localization is
        stored as its device code and not as its name.
    cell_size : float
        The side (km) of the square cell where the photon start
        positions are drawn.

    Raises
    ------
    ValueError
        If direction is given but is not a geoclide Vector, or if loc
        is not one of the known localizations.

    Warns
    -----
    UserWarning
        If fov is greater than 0 for a radiance sensor, where it is
        not yet allowed: fov is then forced to 0.
    """

    def __init__(
        self,
        pos_x: float = 0.,
        pos_y: float = 0.,
        pos_z: float = 0.,
        th_deg: float = 180.,
        ph_deg: float = 180.,
        loc: str = 'SURF0P',
        fov: float = 0.,
        sensor_type: int = 0,
        icell: int = 0,
        ilam_0: int = -1,
        ilam_1: int = -1,
        direction: gc.Vector | None = None,
        cell_size: float = -1.,
    ) -> None:

        if isinstance(direction, gc.Vector):
            th_deg, ph_deg = cast(tuple[float, float],
                                  gc.vec2ang(direction))
        elif direction is not None:
            raise ValueError('direction argument must be a Vector')

        if fov > 0. and sensor_type == 0:
            warnings.warn(
                'fov > 0 is not yet allowed for radiance sensor '
                '(sensor_type=0). It will be forced to 0.', stacklevel=2)
            fov = 0.  # also already forced to 0 in the CUDA code

        self.dict: dict[str, float | int] = {
            'pos_x': pos_x,
            'pos_y': pos_y,
            'pos_z': pos_z,
            'th_deg': th_deg,
            'ph_deg': ph_deg,
            'loc': LOC_CODE.index(loc),
            'fov': fov,
            'sensor_type': sensor_type,
            'icell': icell,
            'ilam_0': ilam_0,
            'ilam_1': ilam_1,
        }
        self.cell_size: float = cell_size

    def __str__(self) -> str:
        """Return the position and direction of the sensor."""
        return ('SENSOR=-pos_x{pos_x}-pos_y{pos_y}-pos_z{pos_z}'
                '-theta={th_deg:.3f}-phi={ph_deg:.3f}'
                .format(**self.dict))


def get_sensor(
    vza_level: float,
    level: float = 0.,
    vaa: float = 0.,
    earth_radius: float = 6371.,
    height_toa: float = 120.,
    fov: float = 0.,
    sensor_type: int = 0,
    pp: bool = True,
    verbose: bool = False,
) -> Sensor:
    """Build a sensor on the atmospheric boundary from view angles.

    This helper is used in backward simulations. The viewing zenith
    angle (``vza_level``) is defined at altitude ``level`` and
    transformed into a sensor position on the top-of-atmosphere
    boundary.

    Parameters
    ----------
    vza_level : float
        Viewing zenith angle (degrees) defined at altitude ``level``.
    level : float, optional
        Altitude (km) at which ``vza_level`` is defined. Default is
        0.0 (ground).
    vaa : float, optional
        Viewing azimuth angle (degrees). Default is 0.0.
    earth_radius : float, optional
        Earth radius (km), used in spherical-shell geometry. Default
        is 6371.0.
    height_toa : float, optional
        Altitude (km) of the top of atmosphere. Default is 120.0.
    fov : float, optional
        Sensor field of view (degrees). Default is 0.0.
    sensor_type : int, optional
        Sensor measurement type:

        - 0: radiance (default)
        - 1: planar irradiance
        - 2: spherical irradiance
    pp : bool, optional
        If `True`, use plane-parallel geometry; if `False`, use
        spherical-shell geometry. Default is `True`.
    verbose : bool, optional
        If `True`, print the computed sensor position. Default is
        `False`.

    Returns
    -------
    Sensor
        Sensor instance positioned on the atmospheric boundary with
        orientation derived from the input angles.

    Raises
    ------
    ValueError
        If the ray from the given position and direction misses the
        atmospheric boundary.
    """
    large_dist = float("inf")  # large distance (km)
    # Compute the direction vector object from the zenith and
    # azimuth angles
    direction = cast(gc.Vector, gc.ang2vec(vza_level, vaa))
    # Compute the intersection of a ray from origin in the given
    # direction with the atmosphere boundary
    if pp:
        origin = gc.Point(0., 0., level)
        ray = gc.Ray(o=origin, d=direction)
        # Rectangle for atmosphere for PP
        boundary = gc.BBox(
            gc.Point(-large_dist, -large_dist, 0.),
            gc.Point(large_dist, large_dist, height_toa))
        _, t1, hit = boundary.intersect(ray, ds_output=False)
    else:
        origin = gc.Point(0., 0., earth_radius + level)
        ray = gc.Ray(o=origin, d=direction)
        # Create the Earth + atmosphere sphere for SS
        boundary = gc.Sphere(height_toa + earth_radius)
        t1, hit = boundary.is_intersection_t(ray)
    if not hit:
        raise ValueError(
            "The intersection test failed!! Check input parameters.")
    # Computation of the sensor position
    pos = origin + direction * cast(float, t1)
    if verbose:
        print("VZA =", vza_level, "--> pos =", pos)

    th, ph = cast(tuple[float, float],
                  gc.vec2ang(direction, vec_view='nadir'))
    if th in (0., 180.):
        # no impact on the I value, but possible impact on Q, U and V
        ph = vaa - 180.
    return Sensor(pos_x=float(pos.x), pos_y=float(pos.y),
                  pos_z=float(pos.z), th_deg=th, ph_deg=ph,
                  loc='ATMOS', fov=fov, sensor_type=sensor_type)


def _sensor_positions(
    grid_3d: Grid3D,
    pos_z: float,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.integer],
]:
    """Compute the sensor positions and cell indices of a 3D grid.

    One sensor is placed at the center of every (x, y) cell of the
    grid, at the altitude ``pos_z``.

    Parameters
    ----------
    grid_3d : Grid3D
        The 3D grid where the sensors are placed.
    pos_z : float
        The altitude of the sensors (km).

    Returns
    -------
    x_centers : NDArray
        The x coordinates of the cell centers, shape (Nx,).
    y_centers : NDArray
        The y coordinates of the cell centers, shape (Ny,).
    xx : NDArray
        The x coordinate of every sensor, shape (Ny, Nx).
    yy : NDArray
        The y coordinate of every sensor, shape (Ny, Nx).
    icells : NDArray
        The flat index of the grid cell containing every sensor,
        shape (Nx*Ny,).
    """
    x_centers = cast(NDArray[np.floating],
                     grid_3d.xgrid[:-1] + np.diff(grid_3d.xgrid) / 2.)
    y_centers = cast(NDArray[np.floating],
                     grid_3d.ygrid[:-1] + np.diff(grid_3d.ygrid) / 2.)

    xx, yy = cast(
        tuple[NDArray[np.floating], NDArray[np.floating]],
        np.meshgrid(x_centers, y_centers),
    )
    zz = np.zeros_like(xx) + pos_z
    icells = cast(
        NDArray[np.integer],
        locate_voxel_index(
            grid_3d.xGRID, grid_3d.yGRID, grid_3d.zGRID,
            xx.ravel(), yy.ravel(), zz.ravel(),
        ),
    )

    return x_centers, y_centers, xx, yy, icells


def _find_cell_index(value: float, grid: NDArray) -> int:
    """Find the index i of the cell holding a value.

    The cell i is the one such that ``grid[i] < value <=
    grid[i+1]``. Note the upper-closed convention: a value lying
    exactly on a boundary belongs to the cell below it.

    Parameters
    ----------
    value : float
        The coordinate to locate.
    grid : NDArray
        Numpy 1D array with the cell boundaries, in ascending order.

    Returns
    -------
    int
        The index of the cell containing the value.

    Raises
    ------
    ValueError
        If the value lies outside the grid.
    """
    for i in range(len(grid) - 1):
        if grid[i] < value <= grid[i + 1]:
            return i
    raise ValueError(
        f"the value {value} lies outside the grid [{grid[0]}, {grid[-1]}]!"
    )


def get_sensors_grid(
    xgrid: NDArray,
    ygrid: NDArray,
    pos_z: float = 120.,
    th_deg: float = 180.,
    ph_deg: float = 180.,
    fov: float = 0.,
    sensor_type: int = 0,
    loc: str = 'ATMOS',
    cell_size: float = -1.,
    grid_3d: Grid3D | None = None,
) -> list[Sensor]:
    """Create one sensor per cell of a regular (x, y) raster.

    The sensors are placed at the center of the cells defined by the
    ``xgrid`` and ``ygrid`` boundaries, at the altitude ``pos_z``,
    and all share the same viewing direction. The sensors are ordered
    row by row: x varies first, then y.

    Parameters
    ----------
    xgrid : NDArray
        Numpy 1D array with the cell boundaries in the x axis.
    ygrid : NDArray
        Numpy 1D array with the cell boundaries in the y axis.
    pos_z : float, optional
        The altitude of the sensors (km). Default: 120.
    th_deg : float, optional
        The viewing zenith angle (deg). Default: 180.
    ph_deg : float, optional
        The viewing azimuth angle (deg). Default: 180.
    fov : float, optional
        The field of view (deg). Default: 0.
    sensor_type : int, optional
        Radiance (0), planar flux (1) or spherical flux (2).
        Default: 0.
    loc : str, optional
        The localization of the sensors. Default: 'ATMOS'.
    cell_size : float, optional
        The size of the sensor cells (km). Default: -1 (not used).
    grid_3d : Grid3D, optional
        The 3D atmosphere grid. When given, the ``icell`` index of
        every sensor is set to the index of the atmosphere cell it
        belongs to. Not needed with a 1D atmosphere, where ``icell``
        is ignored.

    Returns
    -------
    list of Sensor
        The list of the (xgrid.size-1)*(ygrid.size-1) created
        sensors.
    """
    # TODO consider a possible variability between sensors
    # (positions, viewing angles, etc.)
    x_centers = xgrid[:-1] + np.diff(xgrid) / 2.
    y_centers = ygrid[:-1] + np.diff(ygrid) / 2.
    xx, yy = np.meshgrid(x_centers, y_centers)

    sensors = []
    for pos_x, pos_y in zip(xx.ravel(), yy.ravel(), strict=True):
        sensors.append(
            Sensor(pos_x=float(pos_x), pos_y=float(pos_y),
                   pos_z=pos_z, fov=fov, sensor_type=sensor_type,
                   th_deg=th_deg, ph_deg=ph_deg,
                   loc=loc, cell_size=cell_size)
        )

    if grid_3d is not None:
        # Set the cell index of every sensor to the index of the
        # atmosphere cell it belongs to
        *_, icells = _sensor_positions(grid_3d, pos_z)
        for sensor in sensors:
            idx = _find_cell_index(sensor.dict['pos_x'], grid_3d.xgrid)
            idy = _find_cell_index(sensor.dict['pos_y'], grid_3d.ygrid)
            sensor.dict['icell'] = int(icells[idx + grid_3d.Nx * idy])

    return sensors
