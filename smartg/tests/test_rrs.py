"""GPU-free tests of the rotational Raman scattering helpers."""

import numpy as np
import pytest

from smartg.rrs import bjm_minus, bjm_plus, l2d, l2d_inv


def test_l2d_spectra_are_normalised() -> None:
    """l2d gives one sorted, normalised line spectrum per wavelength."""
    wavelength = np.array([400.0, 500.0])

    wavelength_out, l_out = l2d(wavelength, 90.0, 243.0)
    wavelength_in, l_in = l2d_inv(wavelength, 90.0, 243.0)

    assert wavelength_out.shape == l_out.shape == l_in.shape
    assert wavelength_out.shape[0] == 2
    np.testing.assert_allclose(l_out.sum(axis=1), 1.0)
    assert np.all(np.diff(wavelength_out, axis=1) > 0)
    # Stokes and anti-Stokes lines within 300 cm-1 of the excitation
    shift = 1e7 / wavelength[:, None] - 1e7 / wavelength_out
    assert np.all(np.abs(shift) < 300.0)
    assert np.all(shift.min(axis=1) < 0.0) and np.all(shift.max(axis=1) > 0)
    np.testing.assert_allclose(l_in.sum(axis=1), 1.0)
    assert np.all(np.diff(wavelength_in, axis=1) > 0)


def test_l2d_scalar_is_the_first_row() -> None:
    """A scalar wavelength gives the row of that wavelength."""
    wavelength_out, l_out = l2d(np.array([400.0, 500.0]), 90.0, 243.0)

    scalar_out, scalar_l = l2d(400.0, 90.0, 243.0)

    assert scalar_out.shape == (1, wavelength_out.shape[1])
    np.testing.assert_allclose(scalar_out[0], wavelength_out[0])
    np.testing.assert_allclose(scalar_l[0], l_out[0])


@pytest.mark.parametrize("j", [5, np.int64(5), np.array(5)])
def test_bjm_minus_scalar(j: int) -> None:
    """A scalar rotational number gives a float, as bjm_plus does."""
    value = bjm_minus(j)

    assert isinstance(value, float)
    assert value == pytest.approx(3.0 * 5 * 4 / 2.0 / 11 / 9)
    assert isinstance(bjm_plus(j), float)


def test_bjm_minus_vanishes_below_two() -> None:
    """There is no anti-Stokes transition from j = 0 or j = 1."""
    assert bjm_minus(0) == 0.0
    assert bjm_minus(1) == 0.0
    np.testing.assert_allclose(
        bjm_minus(np.arange(4)), [0.0, 0.0, 0.2, 3.0 * 3 * 2 / 2 / 7 / 5]
    )
