"""GPU-free tests of the spectral band set and grids."""

import numpy as np
import pytest

from smartg.bandset import spectral_grids


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
    assert first[first.axis("wavelength").searchsorted(450.0)] == (
        pytest.approx(expected, rel=1e-3)
    )
