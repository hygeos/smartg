"""Tests of the xarray helper module (no GPU required)."""

import numpy as np
import pytest
import xarray as xr

from smartg.xarray import dataset_to_mlut, drop_axes


def sample_dataset() -> xr.Dataset:
    """Small Dataset mimicking the run output structure."""
    ds = xr.Dataset()
    ds.coords['wavelength'] = np.array([400.0, 500.0, 600.0])
    ds.coords['Zenith angles'] = np.array([0.0, 30.0])
    ds['I_up (TOA)'] = xr.Variable(
        ('wavelength', 'Zenith angles'),
        np.arange(6.0).reshape(3, 2),
        attrs={'desc': 'radiance'},
    )
    # coordless dimension, as used by e.g. wLoss or cdist variables
    ds['counts'] = xr.Variable(('index',), np.array([1, 2, 3, 4]))
    ds.attrs['NPhotonIn_sum'] = np.int64(1000)
    ds.attrs['MODE'] = 'PPA'
    return ds


def test_dataset_to_mlut_roundtrip() -> None:
    """Check that a Dataset survives a round trip through MLUT."""
    ds = sample_dataset()
    mlut = dataset_to_mlut(ds)

    assert mlut.datasets() == list(ds.data_vars)
    for name in ds.data_vars:
        lut = mlut[name]
        assert lut.names == list(ds[name].dims)
        np.testing.assert_array_equal(lut.data, ds[name].values)

    # coordful dims keep their values, coordless dims get arange axes
    np.testing.assert_array_equal(
        mlut['I_up (TOA)'].axes[0], ds.coords['wavelength'].values
    )
    np.testing.assert_array_equal(mlut['counts'].axes[0], np.arange(4))

    # variable and global attributes survive
    assert mlut['I_up (TOA)'].attrs['desc'] == 'radiance'
    assert dict(mlut.attrs) == dict(ds.attrs)


def test_drop_axes_dim_and_coord() -> None:
    """Check that drop_axes removes a dimension and its coordinate."""
    ds = xr.Dataset()
    ds.coords['wavelength'] = np.array([500.0])
    ds['I'] = xr.Variable(('wavelength', 'z'), np.ones((1, 3)))
    out = drop_axes(ds, 'wavelength')
    assert 'wavelength' not in out.dims
    assert 'wavelength' not in out.coords
    assert out['I'].dims == ('z',)


def test_drop_axes_coord_only() -> None:
    # a coordinate used by no data variable (LE-zip 'Azimuth angles'
    # case) is removed regardless of its size
    """Check that drop_axes removes a coordinate on its own."""
    ds = xr.Dataset()
    ds.coords['Azimuth angles'] = np.array([0.0, 90.0, 180.0])
    ds['I'] = xr.Variable(('Zenith angles',), np.ones(4))
    out = drop_axes(ds, 'Azimuth angles')
    assert 'Azimuth angles' not in out.coords
    assert 'I' in out


def test_drop_axes_absent_name_is_ignored() -> None:
    """Check that an absent name leaves the Dataset untouched."""
    ds = xr.Dataset({'I': (('z',), np.ones(3))})
    out = drop_axes(ds, 'wavelength')
    assert out.identical(ds)


def test_drop_axes_multiple_names() -> None:
    """Check that drop_axes takes several names at once."""
    ds = xr.Dataset()
    ds.coords['a'] = np.array([1.0])
    ds['I'] = xr.Variable(('a', 'b', 'z'), np.ones((1, 1, 3)))
    out = drop_axes(ds, 'a', 'b')
    assert out['I'].dims == ('z',)


def test_drop_axes_size_error() -> None:
    """Check that an axis longer than one point is refused."""
    ds = xr.Dataset({'I': (('z',), np.ones(3))})
    with pytest.raises(ValueError):
        drop_axes(ds, 'z')
