"""Linear interpolation helpers for regular and irregular grids.

This module provides thin wrappers around
``scipy.ndimage.map_coordinates`` (for N-D arrays sampled at arbitrary
query points) and ``numpy.interp`` (for 1-D coordinate arrays), with
support for increasing or decreasing coordinate axes and optional
boundary clipping.

Key Functions
-------------
interp3
    Sample a 3-D array at arbitrary query points via linear
    interpolation.
interp2
    Sample a 2-D array at arbitrary query points via linear
    interpolation.
interp_1d_coord
    Interpolate a 1-D coordinate with optional extrema clipping.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import map_coordinates


def _index_coords(
    corner_locs: ArrayLike, interp_locs: ArrayLike
) -> NDArray[np.float64]:
    """Return the fractional index of each point along an axis.

    Parameters
    ----------
    corner_locs : array_like
        Increasing or decreasing coordinates of the nodes.
    interp_locs : array_like
        Coordinates of the points.

    Returns
    -------
    ndarray
        Fractional index of each point among the nodes, clipped to
        the first and the last node.
    """
    corner_locs_arr = np.asarray(corner_locs, dtype=np.float64)
    interp_locs_arr = np.asarray(interp_locs, dtype=np.float64)
    index = np.arange(len(corner_locs_arr))
    if np.all(np.diff(corner_locs_arr) < 0):
        corner_locs_arr = corner_locs_arr[::-1]
        index = index[::-1]
    return np.interp(interp_locs_arr, corner_locs_arr, index)


def interp3(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    v: NDArray[np.number],
    xi: ArrayLike,
    yi: ArrayLike,
    zi: ArrayLike,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Sample a 3-D array at arbitrary points, linearly.

    The array ``v`` has pixel corner locations at coordinates ``x``,
    ``y``, ``z``. Values are interpolated at the points ``(xi, yi, zi)``
    using ``scipy.ndimage.map_coordinates`` (order=1, i.e. linear).

    Parameters
    ----------
    x, y, z : array_like
        1-D coordinate vectors giving the pixel corner locations of
        ``v`` along each axis. They may be increasing or decreasing.
    v : ndarray
        3-D array of values to interpolate (``v.ndim == 3``).
    xi, yi, zi : array_like
        Query point coordinates. Any shape is accepted and is
        preserved in the output.
    **kwargs
        Additional keyword arguments forwarded to
        ``scipy.ndimage.map_coordinates``.

    Returns
    -------
    ndarray
        Interpolated values with the same shape as ``xi`` (and ``yi``,
        ``zi``).
    """
    assert v.ndim == 3

    orig_shape = np.asarray(xi).shape
    xi, yi, zi = (a.reshape(-1) for a in np.atleast_1d(xi, yi, zi))

    output = np.empty(xi.shape, dtype=float)
    coords = [
        _index_coords(*item)
        for item in zip([x, y, z], [xi, yi, zi], strict=True)
    ]

    map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)


def interp2(
    x: ArrayLike,
    y: ArrayLike,
    v: NDArray[np.number],
    xi: ArrayLike,
    yi: ArrayLike,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Sample a 2-D array at arbitrary points, linearly.

    The array ``v`` has pixel corner locations at coordinates ``x``,
    ``y``. Values are interpolated at the points ``(xi, yi)`` using
    ``scipy.ndimage.map_coordinates`` (order=1, i.e. linear).

    Parameters
    ----------
    x, y : array_like
        1-D coordinate vectors giving the pixel corner locations of
        ``v`` along each axis. They may be increasing or decreasing.
    v : ndarray
        2-D array of values to interpolate (``v.ndim == 2``).
    xi, yi : array_like
        Query point coordinates. Any shape is accepted and is
        preserved in the output.
    **kwargs
        Additional keyword arguments forwarded to
        ``scipy.ndimage.map_coordinates``.

    Returns
    -------
    ndarray
        Interpolated values with the same shape as ``xi`` (and ``yi``).
    """
    assert v.ndim == 2

    orig_shape = np.asarray(xi).shape
    xi, yi = (a.reshape(-1) for a in np.atleast_1d(xi, yi))

    output = np.empty(xi.shape, dtype=float)
    coords = [
        _index_coords(*item)
        for item in zip([x, y], [xi, yi], strict=True)
    ]

    map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)


def _interp2_stack(
    x: ArrayLike,
    y: ArrayLike,
    v: Sequence[NDArray[np.number]],
    xi: ArrayLike,
    yi: ArrayLike,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Sample several 2-D arrays at the same points, linearly.

    The same as ``np.stack([interp2(x, y, a, xi, yi) for a in v])``,
    the index coordinates of the points computed once: each array goes
    through the very same ``map_coordinates`` call as in `interp2`.

    Parameters
    ----------
    x, y : array_like
        1-D coordinates of the nodes of each array, see `interp2`.
    v : sequence of ndarray
        2-D arrays to interpolate.
    xi, yi : array_like
        Query point coordinates, of any shape.
    **kwargs
        Additional keyword arguments forwarded to
        ``scipy.ndimage.map_coordinates``.

    Returns
    -------
    ndarray
        Interpolated values, of shape ``(len(v),) + xi.shape``.
    """
    orig_shape = np.asarray(xi).shape
    xi, yi = (a.reshape(-1) for a in np.atleast_1d(xi, yi))
    coords = [
        _index_coords(*item)
        for item in zip([x, y], [xi, yi], strict=True)
    ]
    output = np.empty((len(v),) + xi.shape, dtype=float)
    for values, out in zip(v, output, strict=True):
        assert values.ndim == 2
        map_coordinates(values, coords, order=1, output=out, **kwargs)
    return output.reshape((len(v),) + orig_shape)


def interp_1d_coord(
    da: xr.DataArray, coord_name: str, x: ArrayLike, extrema: bool = False
) -> NDArray[np.float64]:
    """Interpolate a 1-D coordinate with optional extrema clipping.

    Parameters
    ----------
    da : DataArray
        Input 1-D data array containing the values to interpolate.
    coord_name : str
        Name of the coordinate used as interpolation axis.
    x : array_like
        Query points where interpolated values are requested. Any shape
        is accepted and preserved in the output.
    extrema : bool, optional
        Boundary behavior:

            - ``False``: strict mode. Values outside coordinate bounds
              raise an exception.
            - ``True``: clip to boundary values (legacy extrema
              behavior).

    Returns
    -------
    ndarray
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
