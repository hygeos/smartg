#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from scipy.ndimage import map_coordinates
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray


def interp3(x, y, z, v, xi, yi, zi, **kwargs):
    """Sample a 3D array "v" with pixel corner locations at "x","y","z" at the
    points in "xi", "yi", "zi" using linear interpolation. Additional kwargs
    are passed on to ``scipy.ndimage.map_coordinates``."""
    assert v.ndim == 3

    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
    xi, yi, zi = np.atleast_1d(xi, yi, zi)
    for arr in [xi, yi, zi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y, z], [xi, yi, zi])]

    map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)


def interp2(x, y, v, xi, yi, **kwargs):
    """Sample a 2D array "v" with pixel corner locations at "x","y", at the
    points in "xi", "yi",  using linear interpolation. Additional kwargs
    are passed on to ``scipy.ndimage.map_coordinates``."""
    assert v.ndim == 2

    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
    xi, yi = np.atleast_1d(xi, yi)
    for arr in [xi, yi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y], [xi, yi])]

    map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)


def interp_1d_coord(
    da: xr.DataArray, coord_name: str, x: ArrayLike, extrema: bool = False
) -> NDArray[np.float64]:
    """Interpolate a 1-D coordinate with optional extrema clipping.

    Parameters
    ----------
    da : xarray.DataArray
        Input 1-D data array containing the values to interpolate.
    coord_name : str
        Name of the coordinate used as interpolation axis.
    x : array-like
        Query points where interpolated values are requested. Any shape is
        accepted and preserved in the output.
    extrema : bool, optional
        Boundary behavior:

            - ``False``: strict mode. Values outside coordinate bounds raise an
              exception.
            - ``True``: clip to boundary values (legacy extrema behavior).

    Returns
    -------
    numpy.ndarray
        Interpolated values with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``extrema`` is ``False`` and at least one query point lies
        outside the coordinate bounds.
    """
    coord = np.asarray(da.coords[coord_name].values, dtype="float64")
    values = np.asarray(da.values, dtype="float64")
    x_arr = np.asarray(x, dtype="float64")
    flat_x = x_arr.ravel()

    if extrema:
        y = np.interp(flat_x, coord, values, left=values[0], right=values[-1])
    else:
        xmin = coord.min()
        xmax = coord.max()
        if np.any((flat_x < xmin) | (flat_x > xmax)):
            raise ValueError(
                f"Out-of-range interpolation requested on '{coord_name}' "
                f"with extrema=False: valid range is [{xmin}, {xmax}]"
            )
        y = np.interp(flat_x, coord, values)

    return y.reshape(x_arr.shape)
