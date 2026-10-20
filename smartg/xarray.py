"""Xarray helpers and conversions into legacy luts objects.

This module provides conversion functions from xarray ``DataArray`` and
``Dataset`` objects to the corresponding legacy ``LUT`` and ``MLUT``
objects, together with small xarray manipulation helpers.

Key Functions
-------------
dataarray_to_lut
    Convert a DataArray into a LUT.
dataset_to_mlut
    Convert a Dataset into an MLUT.
drop_axes
    Remove size-1 dimensions and their coordinates from a Dataset.

The private `_group_index` and `_group_sum` sum the internal bands of
each channel as ``groupby(...).sum`` does, bit for bit, the groups
built once for several arrays.
"""

from collections import OrderedDict
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from luts.luts import LUT, MLUT
from numpy.typing import NDArray


def dataarray_to_lut(data_array: xr.DataArray) -> LUT:
    """Convert a DataArray into a LUT.

    Parameters
    ----------
    data_array : DataArray
        DataArray to convert. Its dimensions are used as LUT axis names,
        and its coordinates are used as axis values when available.

    Returns
    -------
    LUT
        LUT containing the DataArray values, dimensions, and
        coordinates. DataArray attributes are also preserved.
    """
    name = data_array.name or "data"
    axes = [
        data_array.coords[dimension].to_numpy()
        if dimension in data_array.coords
        else np.arange(data_array.sizes[dimension])
        for dimension in data_array.dims
    ]
    description = data_array.attrs.get("desc", name)

    return LUT(
        data_array.to_numpy(),
        axes=axes,
        names=list(data_array.dims),
        desc=description,
        attrs=dict(data_array.attrs),
    )


def dataset_to_mlut(dataset: xr.Dataset) -> MLUT:
    """Convert a Dataset into an MLUT.

    Parameters
    ----------
    dataset : Dataset
        Dataset to convert. Each data variable is converted into a LUT.
        Dataset attributes are copied to the resulting MLUT.

    Returns
    -------
    MLUT
        MLUT containing one LUT for each data variable in the Dataset.
    """
    mlut = MLUT()

    for name, data_array in dataset.data_vars.items():
        mlut.add_lut(dataarray_to_lut(data_array), desc=name)

    mlut.attrs = OrderedDict(dataset.attrs)

    return mlut


def drop_axes(dataset: xr.Dataset, *names: str) -> xr.Dataset:
    """Remove size-1 dimensions and their coordinates from a Dataset.

    Equivalent of the legacy MLUT.dropaxis method: each named
    dimension is squeezed away from every data variable, and the
    matching coordinate is removed. A name used by no data variable
    only loses its coordinate; an absent name is ignored.

    Parameters
    ----------
    dataset : Dataset
        Dataset to squeeze.
    *names : str
        Names of the dimensions to remove. A dimension used by a
        data variable must have size 1.

    Returns
    -------
    Dataset
        New Dataset without the named dimensions and coordinates.
    """
    for name in names:
        used = any(
            name in variable.dims
            for variable in dataset.data_vars.values()
        )
        if used:
            if dataset.sizes[name] != 1:
                raise ValueError(
                    f'cannot drop dimension {name!r} of size '
                    f'{dataset.sizes[name]}'
                )
            dataset = dataset.isel({name: 0}, drop=True)
        elif name in dataset.coords:
            dataset = dataset.drop_vars(name)
    return dataset


class _GroupIndex(NamedTuple):
    """The groups of equal values of a 1-D key, see `_group_index`."""

    key: xr.DataArray
    # positions of each group, None for one group per position
    positions: tuple[NDArray[np.intp], ...] | None
    coord: xr.Variable


def _group_index(key: xr.DataArray) -> _GroupIndex:
    """Return the groups of the equal values of a key, as groupby does.

    Parameters
    ----------
    key : DataArray
        1-D key named after its dimension, such as the channel
        wavelength of each internal band.

    Returns
    -------
    _GroupIndex
        The key, the positions of each group in increasing order (the
        groups in the order of their sorted values, NaN keys in none),
        or None when the key values are unique and monotonic, and the
        coordinate of the grouped result, as xarray's groupby builds
        them without flox.

    Raises
    ------
    ValueError
        If the key is not 1-D and named after its dimension, or holds
        only NaN.
    """
    if key.ndim != 1 or key.name != key.dims[0]:
        raise ValueError(
            "the key must be one-dimensional and named after its dimension"
        )
    index = key.to_index()
    if index.is_unique and (
        index.is_monotonic_increasing or index.is_monotonic_decreasing
    ):
        return _GroupIndex(key, None, key.variable.to_base_variable())
    codes, uniques = pd.factorize(index, sort=True)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    # NaN keys, coded -1, belong to no group
    start = int(np.searchsorted(sorted_codes, 0))
    if start == len(codes):
        raise ValueError(
            "Failed to group data. Are you grouping by a variable that is "
            "all NaN?"
        )
    positions = tuple(
        np.split(
            order[start:],
            np.flatnonzero(np.diff(sorted_codes[start:])) + 1,
        )
    )
    coord = xr.Variable(key.name, np.asarray(uniques), attrs=key.attrs)
    return _GroupIndex(key, positions, coord)


def _nansum(values: NDArray, axis: int) -> NDArray:
    """Sum along an axis, NaN as zero, as xarray's nansum does."""
    if values.dtype.kind in "fc":
        return np.sum(
            np.where(np.isnan(values), np.zeros_like(values), values),
            axis=axis,
        )
    return np.sum(values, axis=axis)


def _group_sum(data_array: xr.DataArray, groups: _GroupIndex) -> xr.DataArray:
    """Return ``data_array.groupby(groups.key).sum(dim)``, bit for bit.

    The groups of `_group_index` can be reused for several arrays. Each
    group goes through the copy and the sum groupby gives it, its
    result keeps the memory layout, the coordinates, the name and the
    attributes groupby gives it. Chunked, non-numeric arrays and arrays
    with a multi-dimensional coordinate go through groupby itself.

    Parameters
    ----------
    data_array : DataArray
        Array to sum, along the dimension of the key.
    groups : _GroupIndex
        Groups of the key, see `_group_index`.

    Returns
    -------
    DataArray
        The sums of each group.

    Raises
    ------
    ValueError
        If the array is not aligned with the key, as groupby raises it.
    """
    key = groups.key
    dim = str(key.dims[0])
    if (
        data_array.chunks is not None
        or data_array.dtype.kind not in "biufc"
        or any(c.ndim > 1 for c in data_array.coords.values())
    ):
        return data_array.groupby(key).sum(dim=dim)
    try:
        xr.align(data_array, key, join="exact", copy=False)
    except ValueError as err:
        raise ValueError(
            "the group variable's length does not match the length of "
            "this variable along its dimensions"
        ) from err
    values = data_array.data
    axis = data_array.get_axis_num(dim)
    assert isinstance(axis, int)
    head = (slice(None),) * axis
    if groups.positions is None:
        parts = [
            _nansum(values[head + (slice(i, i + 1),)], axis)
            for i in range(values.shape[axis])
        ]
    else:
        parts = [_nansum(values[head + (p,)], axis) for p in groups.positions]
    # stacked on a first axis, then put back in place, as groupby does
    sums = np.moveaxis(np.stack(parts, axis=0), 0, axis)
    n_groups = len(parts)
    stacked = (n_groups,) + tuple(
        s for d, s in data_array.sizes.items() if d != dim
    )
    if n_groups == data_array.sizes[dim]:
        coords = {k: c for k, c in data_array.coords.items() if k != dim}
    else:
        # the coordinates groupby keeps: _replace_maybe_drop_dims of
        # DataArray pairs the sizes of the stacked sums with the dims in
        # order
        sizes = dict(zip(data_array.dims, stacked, strict=True))
        coords = {
            k: c for k, c in data_array.coords.items()
            if k != dim and c.shape == tuple(sizes[d] for d in c.dims)
        }
    keep = xr.get_options()["keep_attrs"]
    attrs = dict(data_array.attrs) if keep in ("default", True) else {}
    return xr.DataArray(
        sums, dims=data_array.dims, coords=coords, name=data_array.name,
        attrs=attrs,
    ).assign_coords({dim: groups.coord})
