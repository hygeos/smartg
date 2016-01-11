#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np



def ICDF(P, N=None):
    '''
    Calculate the inversed cumulative distribution function over N values
    returning the index following the probability distribution function (PDF) P

    Arguments:
        * P (1-d array)
            probability distribution function values
        * N: number of discretization for the inversed cumulative distribution function
            (automatically estimated if not provided)
    '''
    P = np.array(P)

    # calculate the cumulative distribution function
    CDF = np.cumsum(P).astype('float32')
    CDF /= CDF[-1]    # normalization

    if N is None:
        # m is the size of smallest CDF value (relative to 1)
        m = np.amin(np.diff(CDF))
        # calculate the number of bins N in the ICDF
        # such that the smallest bin be sampled over at least Nmin values
        # to avoid sampling inaccuracies
        # (maximum relative error is then 1/Nmin)
        Nmin = 10.
        N = int(np.round(Nmin/m))

    #
    # inverse the CDF
    #
    # mid points of the [0,1] internal divided in N
    # (we use the mid points so find the nearest neighbour with searchsorted)
    bins = np.linspace(0,1,num=N,endpoint=False)+1./(2*N)
    ICDF = np.searchsorted(CDF, bins)

    return ICDF


