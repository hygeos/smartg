"""GPU-free tests of the thermal emission of the forward mode.

The layer emission probabilities that ``Smartg.run(cell_proba='auto')``
samples are checked against the Planck law written out here.
"""

import numpy as np
import pytest
import xarray as xr

from smartg.smartg import _emission_proba

H = 6.62607015e-34  # J s
C = 299792458.0  # m s-1
K_B = 1.380649e-23  # J K-1


def _planck(wavelength: float, temperature: np.ndarray) -> np.ndarray:
    """Return the Planck radiance at lambda in nm, up to a factor."""
    wavelength = wavelength * 1e-9  # m
    return 1.0 / wavelength**5 / np.expm1(
        H * C / (wavelength * K_B * temperature)
    )


def test_emission_proba_follows_the_planck_law() -> None:
    """Levels of equal absorption emit in proportion to B(lambda, T)."""
    wavelength = np.array([3900.0, 10800.0], dtype=np.float32)
    t_atm = np.array([217.0, 250.0, 300.0])
    prof_atm = xr.Dataset(
        {
            # 1 km-1 in the two layers under the top level
            "OD_abs_atm": (
                ("wavelength", "z_atm"),
                np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
            ),
            "T_atm": (("z_atm",), t_atm),
        },
        coords={"wavelength": wavelength, "z_atm": [2.0, 1.0, 0.0]},
    )

    proba = _emission_proba(prof_atm, wavelength)

    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(proba[:, 0], 0.0)
    for i, value in enumerate(wavelength):
        radiance = _planck(float(value), t_atm[1:])
        assert proba[i, 1] / proba[i, 2] == pytest.approx(
            radiance[0] / radiance[1], rel=1e-6
        )
