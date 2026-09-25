"""
Post-processing utilities for SMART-G output.

This module provides helpers to integrate angular reflectance fields
over the upper hemisphere and produce irradiance-like diagnostics.

Key Functions
-------------
plane_irr
    Compute plane irradiance from a reflectance DataArray.
spherical_irr
    Compute spherical irradiance from a reflectance DataArray.
irradiance_ds
    Create a normalized irradiance Dataset from reflectance
    datasets.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from luts.luts import LUT, MLUT
from numpy.typing import NDArray


def _as_data_array(da_refl: LUT | xr.DataArray) -> xr.DataArray:
    """Return the reflectance field as a DataArray."""
    if isinstance(da_refl, LUT):
        return da_refl.to_xarray()
    if isinstance(da_refl, xr.DataArray):
        return da_refl
    raise TypeError(
        "da_refl must be a LUT or xarray.DataArray, got "
        f"{type(da_refl).__name__}."
    )


def _bin_solid_angles(
    zenith: NDArray[np.floating],
    azimuth: NDArray[np.floating],
) -> NDArray[np.float64] | None:
    """
    Return the solid angles of the cone-sampling bins of a grid.

    Without ``le``, ``Smartg.run`` counts the photons in ``n_theta``
    zenith bins of width ``dth`` and ``n_phi`` azimuth bins of width
    ``360 / n_phi`` covering the whole circle, and writes the bin
    centres, ``dth / 2, 3 dth / 2, ...`` and ``0, 360 / n_phi, ...``.

    Parameters
    ----------
    zenith : ndarray
        Zenith angles in degrees.
    azimuth : ndarray
        Azimuth angles in degrees.

    Returns
    -------
    ndarray or None
        Solid angle in sr of one bin of each zenith row, of shape
        ``(n_theta,)``, or None when the angles are not the bin
        centres of such a grid.
    """
    n_theta = zenith.size
    n_phi = azimuth.size
    if n_theta == 0 or n_phi == 0 or not zenith[0] > 0:
        return None
    dth = 2.0 * float(zenith[0])
    dphi = 360.0 / n_phi
    if not (
        np.allclose(zenith, dth * (np.arange(n_theta) + 0.5), rtol=1e-5)
        and np.allclose(azimuth, dphi * np.arange(n_phi), atol=1e-5 * dphi)
    ):
        return None
    theta = np.deg2rad(zenith.astype(np.float64))
    half = np.deg2rad(dth) / 2.0
    return np.deg2rad(dphi) * (np.cos(theta - half) - np.cos(theta + half))


def _integrate_hemisphere(
    da: xr.DataArray,
    azimuth_name: str,
    zenith_name: str,
    planar: bool,
) -> xr.DataArray:
    """
    Integrate a radiance field over the directions of its grid.

    Parameters
    ----------
    da : DataArray
        Normalized radiance with azimuth and zenith dimensions.
    azimuth_name : str
        Name of the azimuth axis in degrees.
    zenith_name : str
        Name of the zenith axis in degrees.
    planar : bool
        If True, weight the directions by cos(theta).

    Returns
    -------
    DataArray
        Integral divided by pi, over the non-angular dimensions.
    """
    zenith = np.asarray(da[zenith_name].values)
    azimuth = np.asarray(da[azimuth_name].values)
    mu = np.cos(np.deg2rad(zenith))
    omega = _bin_solid_angles(zenith, azimuth)
    if omega is not None:
        weight = omega * mu if planar else omega
        da_weight = xr.DataArray(weight / np.pi, dims=[zenith_name])
        return (da * da_weight).sum(
            dim=[zenith_name, azimuth_name], skipna=False
        )

    integrand = da * xr.DataArray(mu, dims=[zenith_name]) if planar else da
    integrand = integrand.assign_coords(
        __mu_int=(zenith_name, -mu),
        __phi_int=(azimuth_name, np.deg2rad(azimuth)),
    ).swap_dims({zenith_name: "__mu_int", azimuth_name: "__phi_int"})
    return (integrand / np.pi).integrate("__mu_int").integrate("__phi_int")


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

    Notes
    -----
    On the grid of a run without ``le``, whose angles are the centres
    of ``n_theta`` zenith and ``n_phi`` azimuth bins, each bin is
    weighted by its exact solid angle times the cosine of its centre.
    This is the inverse of the normalization of ``Smartg.run``, so the
    result is the flux of the photons counted in the bins. Any other
    grid, such as the directions of a local estimate, is integrated
    with the trapezoid rule between its first and last angles.
    """
    da = _as_data_array(da_refl)
    return _integrate_hemisphere(da, azimuth_name, zenith_name, True)


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

    Notes
    -----
    On the grid of a run without ``le``, whose angles are the centres
    of ``n_theta`` zenith and ``n_phi`` azimuth bins, each bin is
    weighted by its exact solid angle. Any other grid, such as the
    directions of a local estimate, is integrated with the trapezoid
    rule between its first and last angles.
    """
    da = _as_data_array(da_refl)
    return _integrate_hemisphere(da, azimuth_name, zenith_name, False)


def irradiance_ds(ds_rad: MLUT | xr.Dataset) -> xr.Dataset:
    """
    Create a normalized irradiance Dataset from reflectance datasets.

    For each dataset whose name starts with ``I_``, this function
    computes plane irradiance (``Pflux_``) with ``plane_irr`` and
    spherical irradiance (``Sflux_``) with ``spherical_irr``.
    Datasets whose name starts with ``direct`` are copied unchanged.

    The standard deviations written by ``Smartg.run(stdev=True)``
    (``I_stdev_*``) are left out: the error of an integral cannot be
    derived from the standard deviations of the bins alone.

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
        if name_str.startswith("I_") and "_stdev_" not in name_str:
            out_vars[name_str.replace("I_", "Pflux_")] = plane_irr(da)
            out_vars[name_str.replace("I_", "Sflux_")] = spherical_irr(da)
        if name_str.startswith("direct"):
            out_vars[name_str] = da

    return xr.Dataset(data_vars=out_vars, attrs=ds_in.attrs)
