"""Discrete difference helpers for vertical profiles.

This module provides the two first-difference operators used to turn a
profile of level values into per-layer increments, e.g. a grid of
altitudes or depths into layer thicknesses, or a cumulated optical
thickness into the optical thickness of each layer.

Both return an array of the same length as their input, so the result
stays aligned with the profile it was computed from. They differ only in
which of the two levels bounding a layer the increment is attributed to,
and therefore in which end is padded with a zero:

diff1
    Backward difference, ``a[i] - a[i-1]``, padded at the start. Used
    for the atmosphere, whose grids run downwards from the top of the
    atmosphere.
diff1_end
    Forward difference, ``a[i+1] - a[i]``, padded at the end. Used for
    the ocean, whose grids run downwards from the surface.

For the same input the two return the same differences, shifted by one
position::

    >>> import numpy as np
    >>> a = np.array([0., 10., 25., 45.])
    >>> diff1(a)
    array([ 0., 10., 15., 20.])
    >>> diff1_end(a)
    array([10., 15., 20.,  0.])
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from smartg.typing import NumericArrayLike


def diff1(a: np.ndarray, axis: int = 0, samesize: bool = True) -> NDArray:
    """
    Calculate the first difference of an array along a specified axis.

    Computes the difference between consecutive elements of the array
    along
    the specified axis using `numpy.diff`. By default (samesize=True),
    preserves the original array shape by padding with zeros.

    Parameters
    ----------
    a : ndarray
        Input array for which to compute differences.
    axis : int, optional
        Axis along which differences are computed. Default is 0.
    samesize : bool, optional
        If True (default), the output has the same shape as the input
        array
        with the first slice along the specified axis set to zero. If
        False,
        the output has size reduced by 1 along the specified axis.

    Returns
    -------
    ndarray
        Differences between consecutive elements along the specified
        axis.
        If `samesize=True`, the result has the same shape as `a`.
        If `samesize=False`, the result has shape ``a.shape[axis] - 1``
        along
        the specified axis.

    See Also
    --------
    diff1_end : Same differences, padded at the end instead of the
        start.

    Examples
    --------
    >>> a = np.array([[1, 2, 4, 8], [10, 20, 40, 80]])
    >>> diff1(a, axis=0, samesize=True)
    array([[ 0,  0,  0,  0],
           [ 9, 18, 36, 72]])
    >>> diff1(a, axis=0, samesize=False) # equivalent to np.diff(a, axis=0)
    array([[ 9, 18, 36, 72]])
    """
    if samesize:
        b = np.zeros_like(a)
        key = [slice(None)] * a.ndim
        key[axis] = slice(1, None, None)
        b[tuple(key)] = np.diff(a, axis=axis)[:]
        return b
    else:
        return np.diff(a, axis=axis)


def diff1_end(x: NumericArrayLike) -> NDArray:
    """
    Calculate the first difference of an array, padded at the end.

    This is the counterpart of `diff1`: it computes the same first
    differences, but attributes each of them to the lower of the two
    indices instead of the upper one, so that the padding zero falls at
    the end of the array rather than at its start. In other words it is
    the forward difference ``x[i+1] - x[i]``, whereas `diff1` is the
    backward difference ``x[i] - x[i-1]``.

    This is what a profile running downwards from its reference level
    calls for, such as the depth grid of an oceanic profile, where the
    thickness of a layer belongs to the level bounding it from above.

    Equivalent to ``numpy.ediff1d(x, to_end=[0.])``. Unlike `diff1`, it
    flattens its input and therefore only applies to 1-D profiles.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    ndarray
        The discrete difference array, same length as `x`, with a zero
        appended at the end.

    See Also
    --------
    diff1 : Same differences, padded at the start instead of the end.

    Examples
    --------
    >>> diff1_end(np.array([0., 10., 25., 45.]))
    array([10., 15., 20.,  0.])
    """
    return np.ediff1d(x, to_end=[0.])
