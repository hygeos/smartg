"""Focused unit tests for KDIS channel utilities."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import xarray as xr

from smartg.kdis import KdisIband, KdisIbandList, kdis_emission, reduce_kdis


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
