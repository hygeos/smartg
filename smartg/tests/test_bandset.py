"""GPU-free tests of the spectral band set and grids."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr

from smartg.bandset import BandSet, spectral_grids


def _flat_solar_spectrum() -> np.ndarray:
    """Return 1000 mW/m2/nm from 250 to 800 nm, every 0.1 nm."""
    wavelength = np.arange(250.0, 800.0, 0.1)
    return np.column_stack([wavelength, np.full(wavelength.size, 1000.0)])


def test_spectral_grids_beyond_127_scattering_points() -> None:
    """The interpolation indices go beyond 127 scattering points."""
    wavelength, wavelengths, _, _, index, weight = spectral_grids(
        350.0, 500.0, _flat_solar_spectrum(), dl=0.5, dls=1.0
    )

    assert wavelengths.size > 128
    assert index.max() > 127
    lower, upper = wavelengths[index], wavelengths[index + 1]
    np.testing.assert_allclose(
        lower + weight * (upper - lower), wavelength, rtol=1e-6
    )


def test_spectral_grids_leaves_the_solar_spectrum_alone() -> None:
    """The photon flux conversion does not modify the caller's array."""
    datas = _flat_solar_spectrum()
    reference = datas.copy()

    first = spectral_grids(400.0, 500.0, datas, unit="photons/cm2/s/nm")[3]
    second = spectral_grids(400.0, 500.0, datas, unit="photons/cm2/s/nm")[3]

    np.testing.assert_array_equal(datas, reference)
    np.testing.assert_array_equal(first.data, second.data)
    # 1 mW/m2/nm at 450 nm is 2.265e11 photons/cm2/s/nm
    expected = 1000.0 * 1e-7 * 450e-9 / (6.62607015e-34 * 299792458.0)
    index = np.searchsorted(np.asarray(first.axes[0]), 450.0)
    assert first.data[index] == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    "wavelength",
    [500, np.int64(500), 500.0, np.float32(500.0), np.array(500.0)],
    ids=["int", "int64", "float", "float32", "0-d array"],
)
def test_bandset_real_scalars(wavelength: float) -> None:
    """Every real scalar gives a scalar band set."""
    bands = BandSet(wavelength)

    assert bands.scalar
    np.testing.assert_array_equal(bands[:], np.array([500.0], np.float32))


@pytest.mark.parametrize(
    "wavelength",
    [
        [400, 500],
        (400.0, 500.0),
        np.array([400.0, 500.0]),
        xr.DataArray([400.0, 500.0], dims="wavelength"),
    ],
    ids=["list", "tuple", "array", "DataArray"],
)
def test_bandset_real_sequences(wavelength: list[float]) -> None:
    """Every sequence of real numbers gives a band set."""
    bands = BandSet(wavelength)

    assert not bands.scalar and not bands.use_reptran_kdis
    np.testing.assert_array_equal(bands[:], [400.0, 500.0])


def test_bandset_tuple_of_internal_bands() -> None:
    """A tuple of internal bands is recognised as a list is."""
    ibands = tuple(
        SimpleNamespace(w=w, calc_profile=lambda prof: None)
        for w in (400.0, 500.0)
    )

    bands = BandSet(cast(Any, ibands))

    assert bands.use_reptran_kdis
    np.testing.assert_array_equal(bands[:], [400.0, 500.0])


@pytest.mark.parametrize(
    "wavelength", ["500", None, ["a", "b"], 500 + 1j], ids=repr
)
def test_bandset_refuses_non_real_input(wavelength: object) -> None:
    """A wavelength that is not real raises a TypeError saying so."""
    with pytest.raises(TypeError, match="wavelength must be a real"):
        BandSet(cast(Any, wavelength))
