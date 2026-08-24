#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sensor definition and creation helpers.

Key Classes
-----------
Sensor
    Definition of a sensor (position, viewing direction, type, ...).

Key Functions
-------------
get_sensor
    Build a sensor located on the atmospheric boundary from view
    angles.
get_sensors_grid
    Create one sensor per cell of a regular (x, y) raster.
"""

from __future__ import annotations

import warnings

import geoclide as gc
import numpy as np
from numpy.typing import NDArray
from typing import cast

from smartg.grid3d import Grid3D, locate_voxel_index


# Localization codes of the device, see communs.h
LOC_CODE = ['', 'ATMOS', 'SURF0P', 'SURF0M', '', '', 'OCEAN',
            'SEAFLOOR', 'OBJSURF']


class Sensor(object):
    """
    Definition of the sensor

    Parameters
    ----------
    POSX : float, optional
       The sensor position along the x axis. Default 0.
    POSY : float, optional
        The sensor position along the y axis. Default 0.
    POSZ : float, optional
        The sensor position along the z axis. Default 0.
    THDEG : float, optional
        The source/viewing zenith angle in forward/backward mode.
        Zenith > 90 for downward looking, < 90 for upward.
        Default Zenith.
    PHDEG : float, optional
        The source/viewing azimuth angle in forward/backward mode.
        Zenith > 90 for downward looking, < 90 for upward.
        Default Zenith.
    LOC : str, optional
        Localization of the sensor. Possibilities are:

        * 'SURF0P' -> Start from the surface looking upward, at TOA
          (air side). Default value.
        * 'SURF0M' -> Start from the surface looking downward, at
          ocean surface (water side).
        * 'ATMOS' -> Start from the atmosphere.
        * 'OCEAN' -> Start from the ocean.
        * 'SEAFLOOR' -> Start from the sea floor.
        * 'OBJSURF' -> Start from a 3d object surface.
    FOV : float, optional
        The field of view in degrees. Default 0.
    TYPE : int, optional
        The radiative quantity type. Three possibilities:

        * 0 -> Radiance (default).
        * 1 -> Planar flux.
        * 2 -> Spherical flux.
    ICELL : int, optional
        The box index where the sensor is located. Only for
        simulations with a 3D atmosphere.
    """
    def __init__(self, POSX=0., POSY=0., POSZ=0., THDEG=0.,
                 PHDEG=180., LOC='SURF0P', FOV=0., TYPE=0, ICELL=0,
                 ILAM_0=-1, ILAM_1=-1, V=None, CELL_SIZE=-1.):

        if isinstance(V, gc.Vector):
            THDEG, PHDEG = gc.vec2ang(V)
        elif V is not None:
            raise ValueError('V argument must be a Vector')

        if FOV > 0. and TYPE == 0:
            warnings.warn(
                'FOV > 0 is not yet allowed for radiance sensor '
                '(TYPE=0). It will be forced to 0.')
            FOV = 0.  # also already forced to 0 in the CUDA code

        self.dict = {
            'POSX':  POSX,
            'POSY':  POSY,
            'POSZ':  POSZ,
            'THDEG': THDEG,
            'PHDEG': PHDEG,
            'LOC':   LOC_CODE.index(LOC),
            'FOV':   FOV,
            'TYPE':  TYPE,
            'ICELL': ICELL,
            'ILAM_0': ILAM_0,
            'ILAM_1': ILAM_1
        }
        self.cell_size = CELL_SIZE

    def __str__(self):
        return ('SENSOR=-POSX{POSX}-POSY{POSY}-POSZ{POSZ}'
                '-THETA={THDEG:.3f}-PHI={PHDEG:.3f}'
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
    """Build a sensor located on the atmospheric boundary from view
    angles.

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
    pos = origin + direction*cast(float, t1)
    if verbose:
        print("VZA =", vza_level, "--> pos =", pos)

    th, ph = cast(tuple[float, float],
                  gc.vec2ang(direction, vec_view='nadir'))
    if th == 0. or th == 180.:
        # no impact on the I value, but possible impact on Q, U and V
        ph = vaa - 180.
    return Sensor(POSX=float(pos.x), POSY=float(pos.y),
                  POSZ=float(pos.z), THDEG=th, PHDEG=ph,
                  LOC='ATMOS', FOV=fov, TYPE=sensor_type)


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
                     grid_3d.xgrid[:-1] + np.diff(grid_3d.xgrid)/2.)
    y_centers = cast(NDArray[np.floating],
                     grid_3d.ygrid[:-1] + np.diff(grid_3d.ygrid)/2.)

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
    """Find the index i of the cell such that grid[i] < value <= grid[i+1].

    Note the upper-closed convention: a value lying exactly on a
    boundary belongs to the cell below it.

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
    for i in range(0, len(grid)-1):
        if grid[i] < value <= grid[i+1]:
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
        The 3D atmosphere grid. When given, the ``ICELL`` index of
        every sensor is set to the index of the atmosphere cell it
        belongs to. Not needed with a 1D atmosphere, where ``ICELL``
        is ignored.

    Returns
    -------
    list of Sensor
        The list of the (xgrid.size-1)*(ygrid.size-1) created
        sensors.
    """
    # TODO consider a possible variability between sensors
    # (positions, viewing angles, etc.)
    x_centers = xgrid[:-1] + np.diff(xgrid)/2.
    y_centers = ygrid[:-1] + np.diff(ygrid)/2.
    xx, yy = np.meshgrid(x_centers, y_centers)

    sensors = []
    for pos_x, pos_y in zip(xx.ravel(), yy.ravel()):
        sensors.append(
            Sensor(POSX=float(pos_x), POSY=float(pos_y),
                   POSZ=pos_z, FOV=fov, TYPE=sensor_type,
                   THDEG=th_deg, PHDEG=ph_deg,
                   LOC=loc, CELL_SIZE=cell_size)
        )

    if grid_3d is not None:
        # Set the cell index of every sensor to the index of the
        # atmosphere cell it belongs to
        *_, icells = _sensor_positions(grid_3d, pos_z)
        for sensor in sensors:
            idx = _find_cell_index(sensor.dict['POSX'], grid_3d.xgrid)
            idy = _find_cell_index(sensor.dict['POSY'], grid_3d.ygrid)
            sensor.dict['ICELL'] = icells[idx + grid_3d.Nx*idy]

    return sensors
