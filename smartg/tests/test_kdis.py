"""Focused unit tests for KDIS channel utilities."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr

from smartg.atmosphere import od2k
from smartg.kdis import (
    Kdis,
    KdisIband,
    KdisIbandList,
    kdis_emission,
    reduce_kdis,
)


def _iband(**fields: Any) -> KdisIband:
    """Stand in for a KdisIband with only the fields the tests read."""
    return cast(KdisIband, SimpleNamespace(**fields))


def test_reduce_kdis_uses_channel_weights() -> None:
    """Reduce internal KDIS bands to one value per channel."""
    ibands = KdisIbandList(
        [
            _iband(w=450.0, weight=1.0, ex=1.0, dl=100.0),
            _iband(w=450.0, weight=0.5, ex=1.0, dl=100.0),
            _iband(w=650.0, weight=2.0, ex=1.0, dl=200.0),
        ]
    )
    dataset = xr.Dataset(
        {
            "I_test": (
                ("wavelength",),
                np.array([10.0, 20.0, 30.0]),
                {"desc": "I_test radiance"},
            ),
            "N_test": (
                ("wavelength",),
                np.array([1.0, 2.0, 3.0]),
                {"desc": "N_test photons"},
            ),
            "temperature": (
                ("wavelength",),
                np.array([100.0, 200.0, 300.0]),
                {"desc": "temperature"},
            ),
        },
        coords={"wavelength": np.array([450.0, 450.0, 650.0])},
        attrs={"source": "synthetic"},
    )

    reduced = reduce_kdis(dataset, ibands)

    assert list(reduced.data_vars) == ["I_test", "N_test"]
    assert reduced.attrs == {"source": "synthetic"}
    assert reduced["I_test"].attrs == {"desc": "I_test radiance"}
    np.testing.assert_allclose(
        reduced["I_test"].to_numpy(), [40.0 / 3.0, 30.0]
    )
    np.testing.assert_allclose(reduced["N_test"].to_numpy(), [4.0 / 3.0, 3.0])
    np.testing.assert_array_equal(
        reduced.wavelength.to_numpy(), [450.0, 650.0]
    )


def test_kdis_emission_returns_xarray_data_array() -> None:
    """Calculate the KDIS emission from xarray coordinates and dims."""
    ibands = KdisIbandList(
        [
            _iband(
                w=450.0,
                weight=1.0,
                ex=1.0,
                dl=100.0,
                band=SimpleNamespace(wmin=400.0, wmax=500.0, band=0),
            ),
            _iband(
                w=450.0,
                weight=0.5,
                ex=1.0,
                dl=100.0,
                band=SimpleNamespace(wmin=400.0, wmax=500.0, band=0),
            ),
            _iband(
                w=650.0,
                weight=2.0,
                ex=1.0,
                dl=200.0,
                band=SimpleNamespace(wmin=600.0, wmax=700.0, band=1),
            ),
        ]
    )
    dataset = xr.Dataset(
        {
            "OD_abs_atm": (
                ("wavelength", "z_atm"),
                np.array(
                    [
                        [0.0, 1.0],
                        [0.0, 2.0],
                        [0.0, 3.0],
                    ]
                ),
            ),
            "T_atm": (("z_atm",), np.array([280.0, 290.0])),
        },
        coords={
            "wavelength": np.array([450.0, 450.0, 650.0]),
            "z_atm": np.array([0.0, -1.0]),
        },
    )

    emission = kdis_emission(dataset, ibands)

    assert isinstance(emission, xr.DataArray)
    assert emission.name == "emission"
    assert emission.dims == ("wavelength", "z_atm")
    np.testing.assert_array_equal(
        emission.wavelength.to_numpy(), [450.0, 450.0, 650.0]
    )
    np.testing.assert_array_equal(emission.z_atm.to_numpy(), [0.0, 1000.0])
    assert emission.shape == (3, 2)


def test_missing_model_raises(tmp_path: Path) -> None:
    """A missing KDIS file raises instead of exiting the interpreter."""
    with pytest.raises(FileNotFoundError, match="kdis_nomodel_def.dat"):
        Kdis("nomodel", dir_data=tmp_path)


def _planck_band_mean(
    wmin: float, wmax: float, temperature: np.ndarray
) -> np.ndarray:
    """Return the Planck radiance averaged over [wmin, wmax] nm.

    In W m-2 sr-1 nm-1, by the midpoint rule on 2000 intervals.
    """
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    edges = np.linspace(wmin, wmax, 2001) * 1e-9  # m
    wavelength = 0.5 * (edges[1:] + edges[:-1])[:, None]
    radiance = (
        2.0 * h * c**2 / wavelength**5
        / np.expm1(h * c / (wavelength * k * temperature[None, :]))
    )
    return radiance.mean(axis=0) * 1e-9


def test_emission_takes_the_planck_average_of_each_channel() -> None:
    """Each internal band gets the Planck average of its own channel.

    The channels are neither contiguous nor in wavelength order in the
    file, as with several intervals or several sensors.
    """
    channels = {  # file index: limits in nm
        7: (3500.0, 4000.0),
        2: (10000.0, 11000.0),
        4: (8000.0, 9000.0),
    }
    selection = [(7, 3700.0), (7, 3900.0), (4, 8500.0), (2, 10500.0)]
    ibands = KdisIbandList(
        [
            cast(
                Any,
                SimpleNamespace(
                    w=w,
                    band=SimpleNamespace(
                        band=index,
                        wmin=channels[index][0],
                        wmax=channels[index][1],
                    ),
                ),
            )
            for index, w in selection
        ]
    )
    t_atm = np.array([220.0, 250.0, 290.0])
    dataset = xr.Dataset(
        {
            "OD_abs_atm": (
                ("wavelength", "z_atm"),
                np.outer([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 3.0]),
            ),
            "T_atm": (("z_atm",), t_atm),
        },
        coords={
            "wavelength": [w for _, w in selection],
            "z_atm": [2.0, 1.0, 0.0],
        },
    )

    emission = kdis_emission(dataset, ibands)

    kabs = od2k(dataset, "OD_abs_atm") * 1e-3
    for i, (index, _) in enumerate(selection):
        expected = kabs[i] * _planck_band_mean(*channels[index], t_atm)
        np.testing.assert_allclose(emission[i], expected, rtol=1e-6)
