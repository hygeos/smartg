#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Post-processing utilities for SMART-G output.

This module provides helpers to integrate angular reflectance
fields over the upper hemisphere and produce irradiance-like
diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from luts.luts import LUT, MLUT

if TYPE_CHECKING:
    import xarray as xr


def Irr(
    L: LUT,
    azimuth: str = "Azimuth angles",
    zenith: str = "Zenith angles",
) -> LUT | float:
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
    zenith_axis = cast(LUT, L.axis(zenith, aslut=True))
    azimuth_axis = cast(LUT, L.axis(azimuth, aslut=True))
    mu = (zenith_axis * np.pi / 180.0).apply(np.cos)
    phi = azimuth_axis * np.pi / 180.0
    return (
        1.0
        / np.pi
        * (mu * L)
        .reduce(np.trapezoid, zenith, x=-mu[:])
        .reduce(np.trapezoid, azimuth, x=phi[:])
    )


def SpherIrr(
    L: LUT,
    azimuth: str = "Azimuth angles",
    zenith: str = "Zenith angles",
) -> LUT | float:
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
    zenith_axis = cast(LUT, L.axis(zenith, aslut=True))
    azimuth_axis = cast(LUT, L.axis(azimuth, aslut=True))
    mu = (zenith_axis * np.pi / 180.0).apply(np.cos)
    phi = azimuth_axis * np.pi / 180.0
    return (
        1.0
        / np.pi
        * (L)
        .reduce(np.trapezoid, zenith, x=-mu[:])
        .reduce(np.trapezoid, azimuth, x=phi[:])
    )


def reduce_Irr(m: MLUT) -> xr.Dataset:
    """
    Create an irradiance Dataset from radiance datasets.

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
    xr.Dataset
        xarray Dataset containing generated ``Pflux_`` and
        ``Sflux_`` variables for each ``I_`` input and copied
        ``direct`` variables.
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
    return res.to_xarray()
