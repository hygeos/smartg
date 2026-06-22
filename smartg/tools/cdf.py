#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inverse cumulative distribution function (ICDF) utilities.

This module provides helpers to sample indices according to a
probability distribution function (PDF) by inverting its cumulative
distribution function (CDF). The returned indices can be used to draw
random samples that follow the input PDF, which is required by the
Monte Carlo radiative transfer solver in SMART-G to sample scattering
events from discrete phase functions.

Two entry points are exposed:

- :func:`ICDF` for a 1-D PDF.
- :func:`ICDF2D` for a 2-D PDF, processed row-wise over its first
  axis.
"""

import numpy as np


def ICDF(P, N=None):
    """Invert the CDF of a 1-D PDF and return sampling indices.

    The cumulative distribution function (CDF) of the input
    probability distribution function (PDF) ``P`` is computed and
    normalised. Its inverse is then evaluated at ``N`` mid-points
    evenly spaced over ``[0, 1]``, yielding the indices of ``P`` that
    should be sampled to follow the distribution. When ``N`` is not
    provided, it is automatically estimated so that the smallest CDF
    bin is sampled over at least ``Nmin = 10`` values, bounding the
    maximum relative sampling error to ``1 / Nmin``.

    Parameters
    ----------
    P : array_like
        1-D probability distribution function values. They need not
        be normalised; the CDF is normalised internally.
    N : int, optional
        Number of discretisation points for the inverse cumulative
        distribution function. If ``None`` (default), it is
        automatically estimated from the smallest CDF step so that
        the smallest bin is sampled over at least ``Nmin = 10``
        values.

    Returns
    -------
    ndarray of int
        1-D array of length ``N`` holding the indices of ``P`` to
        sample in order to follow the input distribution.

    Notes
    -----
    The mid-points of the ``[0, 1]`` interval divided in ``N`` bins
    are used so that ``numpy.searchsorted`` finds the nearest
    neighbour of each mid-point in the CDF.
    """
    P = np.array(P)

    # calculate the cumulative distribution function
    CDF = np.cumsum(P).astype("float32")
    CDF /= CDF[-1]  # normalization

    if N is None:
        # m is the size of smallest CDF value (relative to 1)
        m = np.amin(np.diff(CDF))
        # calculate the number of bins N in the ICDF
        # such that the smallest bin be sampled over at least Nmin
        # values to avoid sampling inaccuracies
        # (maximum relative error is then 1/Nmin)
        Nmin = 10.0
        N = int(np.round(Nmin / m))

    #
    # inverse the CDF
    #
    # mid points of the [0,1] internal divided in N
    # (we use the mid points so find the nearest neighbour with
    # searchsorted)
    bins = np.linspace(0, 1, num=N, endpoint=False) + 1.0 / (2 * N)
    ICDF = np.searchsorted(CDF, bins)

    return ICDF


def ICDF2D(P, N=500):
    """Invert the CDF of a 2-D PDF row-wise and return sampling indices.

    :func:`ICDF` is applied to each row of ``P`` (i.e. looping over
    the first axis), producing one set of sampling indices per row.

    Parameters
    ----------
    P : array_like
        2-D probability distribution function values of shape
        ``(n_rows, n_bins)``. Each row is treated as an independent
        1-D PDF and need not be normalised.
    N : int, optional
        Number of discretisation points for the inverse cumulative
        distribution function of each row. Default is ``500``.

    Returns
    -------
    ndarray of int
        2-D array of shape ``(n_rows, N)`` holding, for each row of
        ``P``, the indices to sample in order to follow the
        corresponding distribution.
    """
    # assert P.ndims==2
    ll = []
    for k in range(P.shape[0]):
        ll.append(ICDF(P[k, :], N=N))

    return np.stack(ll)
