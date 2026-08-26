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
"""

from collections import OrderedDict

from luts.luts import LUT, MLUT
import numpy as np
import xarray as xr


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
