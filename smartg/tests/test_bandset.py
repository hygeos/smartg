"""GPU-free tests of the spectral band set and grids."""

import numpy as np

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
