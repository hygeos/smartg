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
    da_refl: LUT | xr.DataArray,
    azimuth_name: str = "Azimuth angles",
    zenith_name: str = "Zenith angles",
) -> xr.DataArray:
    """
    Compute plane irradiance from a reflectance DataArray.

    The quantity is obtained by integrating reflectance over azimuth
    and zenith with a cos(theta) weighting, which gives the flux
    crossing a horizontal plane. The output is a normalized
    irradiance (dimensionless), normalized by the incident irradiance
    and by pi, and keeps all non-angular dimensions.

    To recover physical plane irradiance, multiply this normalized
    quantity by the incident solar irradiance at the considered
    wavelength.

    Parameters
    ----------
    da_refl : LUT or DataArray
        Reflectance field containing azimuth and zenith dimensions.
        LUT inputs are converted with ``to_xarray()``.
    azimuth_name : str, optional
        Name of the azimuth axis in degrees.
    zenith_name : str, optional
        Name of the zenith axis in degrees.

    Returns
    -------
    DataArray
        Normalized plane irradiance after angular reduction.
    """
    if isinstance(da_refl, LUT):
        da = da_refl.to_xarray()
    elif isinstance(da_refl, xr.DataArray):
        da = da_refl
    else:
        raise TypeError(
            "da_refl must be a LUT or xarray.DataArray, got "
            f"{type(da_refl).__name__}."
        )

    zenith_rad = np.deg2rad(da[zenith_name])
    azimuth_rad = np.deg2rad(da[azimuth_name])
    mu = np.cos(zenith_rad)
    integrand = (
        (da * mu)
        .assign_coords(
            __mu_int=(zenith_name, (-mu).data),
            __phi_int=(azimuth_name, azimuth_rad.data),
        )
        .swap_dims({zenith_name: "__mu_int", azimuth_name: "__phi_int"})
    )

    return (integrand / np.pi).integrate("__mu_int").integrate("__phi_int")


def spherical_irr(
    da_refl: LUT | xr.DataArray,
    azimuth_name: str = "Azimuth angles",
    zenith_name: str = "Zenith angles",
) -> xr.DataArray:
    """
    Compute spherical irradiance from a reflectance DataArray.

    The quantity is obtained by integrating reflectance over azimuth
    and zenith without a cos(theta) weighting, yielding scalar
    (actinic) flux. The output is a normalized irradiance
    (dimensionless), normalized by the incident irradiance and by pi,
    and keeps all non-angular dimensions.

    To recover physical spherical irradiance, multiply this normalized
    quantity by the incident solar irradiance at the considered
    wavelength.

    Parameters
    ----------
    da_refl : LUT or DataArray
        Reflectance field containing azimuth and zenith dimensions.
        LUT inputs are converted with ``to_xarray()``.
    azimuth_name : str, optional
        Name of the azimuth axis in degrees.
    zenith_name : str, optional
        Name of the zenith axis in degrees.

    Returns
    -------
    DataArray
        Normalized spherical irradiance after angular reduction.
    """
    if isinstance(da_refl, LUT):
        da = da_refl.to_xarray()
    elif isinstance(da_refl, xr.DataArray):
        da = da_refl
    else:
        raise TypeError(
            "da_refl must be a LUT or xarray.DataArray, got "
            f"{type(da_refl).__name__}."
        )

    zenith_rad = np.deg2rad(da[zenith_name])
    azimuth_rad = np.deg2rad(da[azimuth_name])
    mu = np.cos(zenith_rad)
    integrand = da.assign_coords(
        __mu_int=(zenith_name, (-mu).data),
        __phi_int=(azimuth_name, azimuth_rad.data),
    ).swap_dims({zenith_name: "__mu_int", azimuth_name: "__phi_int"})

    return (integrand / np.pi).integrate("__mu_int").integrate("__phi_int")


def irradiance_ds(ds_rad: MLUT | xr.Dataset) -> xr.Dataset:
    """
    Create a normalized irradiance Dataset from reflectance datasets.

    For each dataset whose name starts with ``I_``, this function
    computes plane irradiance (``Pflux_``) with ``plane_irr`` and
    spherical irradiance (``Sflux_``) with ``spherical_irr``.
    Datasets whose name starts with ``direct`` are copied unchanged.

    Parameters
    ----------
    ds_rad : MLUT or Dataset
        SMART-G reflectance container (dimensionless). MLUT inputs
        are converted with ``to_xarray()``.

    Returns
    -------
    Dataset
        xarray Dataset containing normalized ``Pflux_`` and
        ``Sflux_`` variables for each ``I_`` input and copied
        ``direct`` variables. Multiply the normalized fluxes by the
        incident solar irradiance to obtain physical irradiances.
    """
    if isinstance(ds_rad, MLUT):
        ds_in = ds_rad.to_xarray()
    elif isinstance(ds_rad, xr.Dataset):
        ds_in = ds_rad
    else:
        raise TypeError(
            "ds_rad must be an MLUT or xarray.Dataset, got "
            f"{type(ds_rad).__name__}."
        )

    out_vars: dict[str, xr.DataArray] = {}
    for name, da in ds_in.data_vars.items():
        name_str = str(name)
        if name_str.startswith("I_"):
            out_vars[name_str.replace("I_", "Pflux_")] = plane_irr(da)
            out_vars[name_str.replace("I_", "Sflux_")] = spherical_irr(da)
        if name_str.startswith("direct"):
            out_vars[name_str] = da

    return xr.Dataset(data_vars=out_vars, attrs=ds_in.attrs)
