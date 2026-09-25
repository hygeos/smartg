"""GPU-free tests of the vibrational Raman scattering helpers."""

import numpy as np
import pytest

from smartg.vrs import raman_response


def test_raman_response_scalar_and_integer_input() -> None:
    """Scalar and integer wavenumbers give the float array result."""
    reference = raman_response(np.array([3300.0, 3400.0]))

    np.testing.assert_allclose(raman_response(3300.0), reference[:1])
    np.testing.assert_allclose(raman_response(3400), reference[1:])
    np.testing.assert_allclose(raman_response([3300, 3400]), reference)


def test_raman_response_is_normalised() -> None:
    """The O-H stretching band integrates to one over wavenumber."""
    ks = np.arange(2000.0, 5000.0, 0.5)

    assert np.trapezoid(raman_response(ks), ks) == pytest.approx(1.0)
