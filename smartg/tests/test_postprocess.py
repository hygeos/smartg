"""GPU-free tests of the irradiance helpers of smartg.postprocess.

An isotropic radiance field is binned and normalized as Smartg.run does
without a local estimate: its plane irradiance is pi times the radiance
and its spherical irradiance twice that, so the normalized plane and
spherical irradiances must be the radiance and twice the radiance.
"""

import numpy as np
import pytest
import xarray as xr

from smartg.postprocess import irradiance_ds, plane_irr, spherical_irr

RADIANCE = 0.3


def _isotropic_run(n_theta: int, n_phi: int) -> xr.DataArray:
    """Return Smartg.run's radiance for an isotropic field.

    The photons of each bin are the exact flux of the field across the
    bin, and they are normalized as Smartg.run does for a run without
    a local estimate.
    """
    from smartg.smartg import _calc_solid_angles

    th, phi, omega = _calc_solid_angles(n_theta, n_phi)
    dth = th[1] - th[0]
    dphi = 2 * np.pi / n_phi
    # exact projected solid angle of each bin
    proj = dphi * (np.sin(th + dth / 2) ** 2 - np.sin(th - dth / 2) ** 2) / 2
    photons = RADIANCE * proj / np.pi
    radiance = photons / (2.0 * omega * np.cos(th))
    return xr.DataArray(
        np.broadcast_to(radiance, (n_phi, n_theta)),
        coords={
            "Azimuth angles": np.rad2deg(phi),
            "Zenith angles": np.rad2deg(th),
        },
        dims=["Azimuth angles", "Zenith angles"],
    )


@pytest.mark.parametrize(("n_theta", "n_phi"), [(45, 90), (10, 12)])
def test_irradiance_of_isotropic_radiance(n_theta: int, n_phi: int) -> None:
    """The bins give pi L and 2 pi L, not their trapezoid rule."""
    da = _isotropic_run(n_theta, n_phi)
    np.testing.assert_allclose(float(plane_irr(da)), RADIANCE, rtol=1e-12)
    # the kernel radiance is the flux over the solid angle times the
    # cosine of the bin centre, 1 - cos(dth / 2) below the true one
    rtol = 1 - np.cos(np.deg2rad(45.0 / n_theta)) + 1e-12
    np.testing.assert_allclose(
        float(spherical_irr(da)), 2 * RADIANCE, rtol=rtol
    )


def test_irradiance_keeps_other_dimensions() -> None:
    """A wavelength axis is kept and each wavelength is integrated."""
    da = _isotropic_run(45, 90)
    da = xr.concat([da, 2 * da], dim="wavelength")
    np.testing.assert_allclose(
        plane_irr(da).values, [RADIANCE, 2 * RADIANCE], rtol=1e-12
    )


def test_irradiance_on_other_grid_uses_trapezoid() -> None:
    """A grid that is not made of bin centres keeps the trapezoid."""
    da = xr.DataArray(
        np.ones((37, 91)),
        coords={
            "Azimuth angles": np.linspace(0.0, 360.0, 37),
            "Zenith angles": np.linspace(0.0, 90.0, 91),
        },
        dims=["Azimuth angles", "Zenith angles"],
    )
    np.testing.assert_allclose(float(plane_irr(da)), 1.0, rtol=1e-3)
    np.testing.assert_allclose(float(spherical_irr(da)), 2.0, rtol=1e-3)


def test_irradiance_ds_leaves_out_stdev() -> None:
    """The standard deviations are not integrated as radiances."""
    da = _isotropic_run(45, 90)
    ds = xr.Dataset({"I_up (TOA)": da, "I_stdev_up (TOA)": 0.1 * da})
    ds_irr = irradiance_ds(ds)
    assert sorted(map(str, ds_irr.data_vars)) == [
        "Pflux_up (TOA)",
        "Sflux_up (TOA)",
    ]
    np.testing.assert_allclose(
        float(ds_irr["Pflux_up (TOA)"]), RADIANCE, rtol=1e-12
    )
