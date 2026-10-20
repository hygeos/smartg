"""Tests of the xarray helper module (no GPU required)."""

import numpy as np
import pytest
import xarray as xr

from smartg.xarray import (
    _group_index,
    _group_sum,
    dataset_to_mlut,
    drop_axes,
)


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


KEYS = {
    "unique": [400.0, 500.0, 600.0, 700.0, 800.0],
    "decreasing": [800.0, 700.0, 600.0, 500.0, 400.0],
    "repeated": [400.0, 400.0, 500.0, 600.0, 600.0],
    "unsorted": [600.0, 400.0, 600.0, 500.0, 400.0],
    "nan": [600.0, np.nan, 600.0, 500.0, 400.0],
}
LAYOUTS = [
    ("wavelength",),
    ("wavelength", "n"),
    ("n", "wavelength"),
    ("sensor", "wavelength", "a", "z"),
    ("Categories", "wavelength"),
]


@pytest.mark.parametrize("dtype", ["f8", "f4", "i8"])
@pytest.mark.parametrize("dims", LAYOUTS, ids=["-".join(d) for d in LAYOUTS])
@pytest.mark.parametrize("key", KEYS)
def test_group_sum_as_groupby(key: str, dims: tuple[str, ...],
                              dtype: str) -> None:
    """The group sums are groupby's: values, layout, coords, attrs."""
    sizes = {"wavelength": 5, "n": 9, "sensor": 2, "a": 3, "z": 4,
             "Categories": 9}
    rng = np.random.default_rng(len(dims))
    values = (rng.random(tuple(sizes[d] for d in dims)) * 10).astype(dtype)
    if dtype != "i8":
        values.flat[3] = np.nan
    coords: dict = {"wavelength": np.arange(5.0) + 0.5}
    coords.update({d: np.arange(sizes[d]) for d in dims if d != "wavelength"})
    if "sensor" in dims:
        coords["sensor index"] = ("sensor", np.arange(2))
    data = xr.DataArray(values, dims=dims, coords=coords, name="x",
                        attrs={"desc": "x"})
    grouping = xr.DataArray(
        np.array(KEYS[key]), dims=("wavelength",),
        coords={"wavelength": coords["wavelength"]}, name="wavelength",
    )
    ref = data.groupby(grouping).sum(dim="wavelength")
    new = _group_sum(data, _group_index(grouping))
    xr.testing.assert_identical(new, ref)
    assert new.values.tobytes() == ref.values.tobytes()
    assert new.values.strides == ref.values.strides


def test_group_sum_refuses_a_misaligned_array() -> None:
    """An array over other wavelengths raises groupby's error."""
    grouping = xr.DataArray([1.0, 1.0], dims=("wavelength",),
                            coords={"wavelength": [1.0, 2.0]},
                            name="wavelength")
    data = xr.DataArray([1.0, 2.0], dims=("wavelength",),
                        coords={"wavelength": [1.0, 3.0]})
    with pytest.raises(ValueError, match="does not match the length"):
        _group_sum(data, _group_index(grouping))
    with pytest.raises(ValueError, match="all NaN"):
        _group_index(grouping * np.nan)
