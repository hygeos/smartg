#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Post-processing utilities for SMART-G output.

This module provides helpers to integrate angular reflectance fields
over the upper hemisphere and produce irradiance-like diagnostics.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from luts.luts import LUT, MLUT


def plane_irr(
    L: LUT | xr.DataArray,
    azimuth: str = "Azimuth angles",
    zenith: str = "Zenith angles",
) -> xr.DataArray:
    """
    Compute plane irradiance from a reflectance DataArray.

    The quantity is obtained by integrating reflectance over azimuth
    and zenith with a cos(theta) weighting, which gives the flux
    crossing a horizontal plane. The output is normalized by pi and
    keeps all non-angular dimensions.

    Parameters
    ----------
    L : LUT or xr.DataArray
        Reflectance field containing azimuth and zenith dimensions.
        LUT inputs are converted with ``to_xarray()``.
    azimuth : str, optional
        Name of the azimuth axis in degrees.
    zenith : str, optional
        Name of the zenith axis in degrees.

    Returns
    -------
    xr.DataArray
        Plane irradiance after angular reduction.
    """
    if isinstance(L, LUT):
        da = L.to_xarray()
    elif isinstance(L, xr.DataArray):
        da = L
    else:
        raise TypeError(
            f"L must be a LUT or xarray.DataArray, got {type(L).__name__}."
        )

    zenith_rad = np.deg2rad(da[zenith])
    azimuth_rad = np.deg2rad(da[azimuth])
    mu = np.cos(zenith_rad)
    integrand = (
        (da * mu)
        .assign_coords(
            __mu_int=(zenith, (-mu).data),
            __phi_int=(azimuth, azimuth_rad.data),
        )
        .swap_dims({zenith: "__mu_int", azimuth: "__phi_int"})
    )

    return (integrand / np.pi).integrate("__mu_int").integrate("__phi_int")


def spherical_irr(
    L: LUT | xr.DataArray,
    azimuth: str = "Azimuth angles",
    zenith: str = "Zenith angles",
) -> xr.DataArray:
    """
    Compute spherical irradiance from a reflectance DataArray.

    The quantity is obtained by integrating reflectance over azimuth
    and zenith without a cos(theta) weighting, yielding scalar
    (actinic) flux. The output is normalized by pi and keeps all
    non-angular dimensions.

    Parameters
    ----------
    L : LUT or xr.DataArray
        Reflectance field containing azimuth and zenith dimensions.
        LUT inputs are converted with ``to_xarray()``.
    azimuth : str, optional
        Name of the azimuth axis in degrees.
    zenith : str, optional
        Name of the zenith axis in degrees.

    Returns
    -------
    xr.DataArray
        Spherical irradiance after angular reduction.
    """
    if isinstance(L, LUT):
        da = L.to_xarray()
    elif isinstance(L, xr.DataArray):
        da = L
    else:
        raise TypeError(
            f"L must be a LUT or xarray.DataArray, got {type(L).__name__}."
        )

    zenith_rad = np.deg2rad(da[zenith])
    azimuth_rad = np.deg2rad(da[azimuth])
    mu = np.cos(zenith_rad)
    integrand = da.assign_coords(
        __mu_int=(zenith, (-mu).data),
        __phi_int=(azimuth, azimuth_rad.data),
    ).swap_dims({zenith: "__mu_int", azimuth: "__phi_int"})

    return (integrand / np.pi).integrate("__mu_int").integrate("__phi_int")


def irradiance_ds(m: MLUT | xr.Dataset) -> xr.Dataset:
    """
    Create an irradiance Dataset from radiance datasets.

    For each dataset whose name starts with ``I_``, this function
    computes plane irradiance (``Pflux_``) with ``plane_irr`` and
    spherical irradiance (``Sflux_``) with ``spherical_irr``.
    Datasets whose name starts with ``direct`` are copied unchanged.

    Parameters
    ----------
    m : MLUT or xr.Dataset
        SMART-G radiance container. MLUT inputs are converted with
        ``to_xarray()``.

    Returns
    -------
    xr.Dataset
        xarray Dataset containing generated ``Pflux_`` and ``Sflux_``
        variables for each ``I_`` input and copied ``direct``
        variables.
    """
    if isinstance(m, MLUT):
        ds_in = m.to_xarray()
    elif isinstance(m, xr.Dataset):
        ds_in = m
    else:
        raise TypeError(
            f"m must be an MLUT or xarray.Dataset, got {type(m).__name__}."
        )

    out_vars: dict[str, xr.DataArray] = {}
    for name, da in ds_in.data_vars.items():
        name_str = str(name)
        if name_str.startswith("I_"):
            out_vars[name_str.replace("I_", "Pflux_")] = (
                plane_irr(da)
            )
            out_vars[name_str.replace("I_", "Sflux_")] = (
                spherical_irr(da)
            )
        if name_str.startswith("direct"):
            out_vars[name_str] = da

    return xr.Dataset(data_vars=out_vars, attrs=ds_in.attrs)
