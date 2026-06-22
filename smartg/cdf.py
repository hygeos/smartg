#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inverse cumulative distribution function (icdf) utilities.

This module provides helpers to sample indices according to a
probability distribution function (PDF) by inverting its cumulative
distribution function (CDF). The returned indices can be used to draw
random samples that follow the input PDF, which is required by the
Monte Carlo radiative transfer solver in SMART-G to sample scattering
events from discrete phase functions.

Two entry points are exposed:

- :func:`icdf` for a 1-D PDF.
- :func:`icdf_2d` for a 2-D PDF, processed row-wise over its first
  axis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from smartg.typing import NumericArrayLike


def icdf(
    pdf: NumericArrayLike, n: int | None = None
) -> NDArray[np.integer]:
    """Invert the CDF of a 1-D PDF and return sampling indices.

    The cumulative distribution function (CDF) of the input
    probability distribution function (PDF) ``pdf`` is computed and
    normalised. Its inverse is then evaluated at ``n`` mid-points
    evenly spaced over ``[0, 1]``, yielding the indices of ``pdf``
    that should be sampled to follow the distribution. When ``n`` is
    not provided, it is automatically estimated so that the smallest
    CDF bin is sampled over at least ``n_min = 10`` values, bounding
    the maximum relative sampling error to ``1 / n_min``.

    Parameters
    ----------
    pdf : array_like
        1-D probability distribution function values. They need not
        be normalised; the CDF is normalised internally.
    n : int, optional
        Number of discretisation points for the inverse cumulative
        distribution function. If ``None`` (default), it is
        automatically estimated from the smallest CDF step so that
        the smallest bin is sampled over at least ``n_min = 10``
        values.

    Returns
    -------
    ndarray of int
        1-D array of length ``n`` holding the indices of ``pdf`` to
        sample in order to follow the input distribution.

    Notes
    -----
    The mid-points of the ``[0, 1]`` interval divided in ``n`` bins
    are used so that ``numpy.searchsorted`` finds the nearest
    neighbour of each mid-point in the CDF.
    """
    pdf = np.array(pdf)

    # calculate the cumulative distribution function
    cdf: NDArray[np.floating] = np.cumsum(pdf).astype("float32")
    cdf /= cdf[-1]  # normalization

    if n is None:
        # m is the size of smallest CDF value (relative to 1)
        m = np.amin(np.diff(cdf))
        # calculate the number of bins n in the icdf
        # such that the smallest bin be sampled over at least n_min
        # values to avoid sampling inaccuracies
        # (maximum relative error is then 1/n_min)
        n_min = 10.0
        n = int(np.round(n_min / m))

    #
    # inverse the CDF
    #
    # mid points of the [0,1] internal divided in n
    # (we use the mid points so find the nearest neighbour with
    # searchsorted)
    bins = np.linspace(0, 1, num=n, endpoint=False) + 1.0 / (2 * n)
    icdf: NDArray[np.integer] = np.searchsorted(cdf, bins)

    return icdf


def icdf_2d(
    pdf: NDArray[np.floating], n: int = 500
) -> NDArray[np.integer]:
    """Invert the CDF of a 2-D PDF row-wise and return sampling indices.

    :func:`icdf` is applied to each row of ``pdf`` (i.e. looping over
    the first axis), producing one set of sampling indices per row.

    Parameters
    ----------
    pdf : array_like
        2-D probability distribution function values of shape
        ``(n_rows, n_bins)``. Each row is treated as an independent
        1-D PDF and need not be normalised.
    n : int, optional
        Number of discretisation points for the inverse cumulative
        distribution function of each row. Default is ``500``.

    Returns
    -------
    ndarray of int
        2-D array of shape ``(n_rows, n)`` holding, for each row of
        ``pdf``, the indices to sample in order to follow the
        corresponding distribution.
    """
    # assert pdf.ndims==2
    ll: list[NDArray[np.integer]] = []
    for k in range(pdf.shape[0]):
        ll.append(icdf(pdf[k, :], n=n))

    return np.stack(ll)
