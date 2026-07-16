"""Convert xarray objects into legacy luts objects.

This module provides conversion functions from xarray ``DataArray`` and
``Dataset`` objects to the corresponding legacy ``LUT`` and ``MLUT``
objects.
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
