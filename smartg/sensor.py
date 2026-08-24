#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sensor creation helpers for the SMART-G 3D atmosphere mode.

The 3D atmosphere itself is built with the Atm3D and Cloud3D
(module smartg.atmosphere) and Grid3D (module smartg.grid3d)
classes.

Key Functions
-------------
create_sensors
    Create one sensor per (x, y) column of a 3D grid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import cast

from smartg.grid3d import Grid3D, locate_voxel_index
from smartg.smartg import Sensor


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


def create_sensors(
    grid_3d: Grid3D,
    pos_z: float = 120.,
    th_deg: float = 180.,
    ph_deg: float = 180.,
    fov: float = 0.,
    sensor_type: int = 0,
    loc: str = 'ATMOS',
    cell_size: float = -1.,
    grid_3d_atm: Grid3D | None = None,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    list[Sensor],
    NDArray[np.integer],
]:
    """Create one sensor per (x, y) column of a 3D grid.

    The sensors are placed at the center of the (x, y) cells of the
    grid, at the altitude ``pos_z``, and all share the same viewing
    direction.

    Parameters
    ----------
    grid_3d : Grid3D
        The 3D grid where the sensors are placed.
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
    grid_3d_atm : Grid3D, optional
        The atmosphere grid, to be given when it differs from
        ``grid_3d``. The sensor cell indices then refer to the
        atmosphere grid instead of the sensor grid.

    Returns
    -------
    x_centers : NDArray
        The x coordinates of the cell centers, shape (Nx,).
    y_centers : NDArray
        The y coordinates of the cell centers, shape (Ny,).
    sensors : list of Sensor
        The list of the Nx*Ny created sensors.
    icells : NDArray
        The flat index of the ``grid_3d`` cell containing every
        sensor, shape (Nx*Ny,).
    """
    # TODO consider a possible variability between sensors
    # (positions, viewing angles, etc.)
    x_centers, y_centers, xx, yy, icells = _sensor_positions(
        grid_3d, pos_z)

    sensors = []
    for pos_x, pos_y, icell in zip(xx.ravel(), yy.ravel(), icells):
        sensors.append(
            Sensor(POSX=float(pos_x), POSY=float(pos_y),
                   POSZ=pos_z, FOV=fov,
                   TYPE=sensor_type, THDEG=th_deg, PHDEG=ph_deg,
                   LOC=loc, ICELL=icell, CELL_SIZE=cell_size)
        )

    if grid_3d_atm is not None:
        # Keep the sensor positions, but replace their cell index by
        # the index of the atmosphere cell they belong to
        *_, icells_atm = _sensor_positions(grid_3d_atm, pos_z)
        for sensor in sensors:
            idx = _find_cell_index(
                sensor.dict['POSX'], grid_3d_atm.xgrid)
            idy = _find_cell_index(
                sensor.dict['POSY'], grid_3d_atm.ygrid)
            sensor.dict['ICELL'] = icells_atm[idx + grid_3d_atm.Nx*idy]

    return x_centers, y_centers, sensors, icells
