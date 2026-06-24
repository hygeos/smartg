#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Post-processing utilities for SMART-G output.

This module provides helpers to integrate angular reflectance
fields over the upper hemisphere and produce irradiance-like
diagnostics.
"""

import numpy as np
from luts.luts import MLUT


def Irr(L, azimuth="Azimuth angles", zenith="Zenith angles"):
    """
    Compute plane irradiance from a reflectance LUT.

    The quantity is obtained by integrating reflectance over
    azimuth and zenith with a cos(theta) weighting, which gives
    the flux crossing a horizontal plane. The output is
    normalized by pi and keeps all non-angular dimensions.

    Parameters
    ----------
    L : LUT
        Reflectance LUT containing azimuth and zenith dimensions.
    azimuth : str, optional
        Name of the azimuth axis in degrees.
    zenith : str, optional
        Name of the zenith axis in degrees.

    Returns
    -------
    LUT or scalar
        Plane irradiance value. A scalar is returned when no
        dimensions remain after angular reduction, otherwise a
        LUT is returned.
    """
    mu = (L.axis(zenith, aslut=True) * np.pi / 180.0).apply(np.cos)
    phi = L.axis(azimuth, aslut=True) * np.pi / 180.0
    return (
        1.0
        / np.pi
        * (mu * L)
        .reduce(np.trapezoid, zenith, x=-mu[:])
        .reduce(np.trapezoid, azimuth, x=phi[:])
    )


def SpherIrr(L, azimuth="Azimuth angles", zenith="Zenith angles"):
    """
    Compute spherical irradiance from a reflectance LUT.

    The quantity is obtained by integrating reflectance over
    azimuth and zenith without a cos(theta) weighting, yielding
    scalar (actinic) flux. The output is normalized by pi and
    keeps all non-angular dimensions.

    Parameters
    ----------
    L : LUT
        Reflectance LUT containing azimuth and zenith dimensions.
    azimuth : str, optional
        Name of the azimuth axis in degrees.
    zenith : str, optional
        Name of the zenith axis in degrees.

    Returns
    -------
    LUT or scalar
        Spherical irradiance value. A scalar is returned when no
        dimensions remain after angular reduction, otherwise a
        LUT is returned.
    """
    mu = (L.axis(zenith, aslut=True) * np.pi / 180.0).apply(np.cos)
    phi = L.axis(azimuth, aslut=True) * np.pi / 180.0
    return (
        1.0
        / np.pi
        * (L)
        .reduce(np.trapezoid, zenith, x=-mu[:])
        .reduce(np.trapezoid, azimuth, x=phi[:])
    )


def reduce_Irr(m):
    """
    Create a new irradiance MLUT from radiance datasets.

    For each dataset whose name starts with ``I_``, this function
    computes plane irradiance (``Pflux_``) with ``Irr`` and
    spherical irradiance (``Sflux_``) with ``SpherIrr``. Datasets
    whose name starts with ``direct`` are copied unchanged.

    Parameters
    ----------
    m : MLUT
        Multi-LUT containing SMART-G radiance datasets.

    Returns
    -------
    MLUT
        New multi-LUT containing generated ``Pflux_`` and
        ``Sflux_`` datasets for each ``I_`` input and copied
        ``direct*`` datasets.
    """
    res = MLUT()
    for d in m.datasets():
        if d.startswith("I_"):
            l_tmp = Irr(m[d])
            res.add_lut(l_tmp, desc=d.replace("I_", "Pflux_"))
            l_tmp = SpherIrr(m[d])
            res.add_lut(l_tmp, desc=d.replace("I_", "Sflux_"))
        if d.startswith("direct"):
            res.add_lut(m[d])
    return res
