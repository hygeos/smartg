"""GPU-free tests of the thermal emission of the forward mode.

The layer emission probabilities that ``Smartg.run(cell_proba='auto')``
samples are checked against the Planck law written out here, and the
layer sampling table of ``cell_proba`` against the kernel indexing.
"""

import numpy as np
import pytest
import xarray as xr

from smartg.cdf import icdf_2d
from smartg.smartg import _cell_proba_table, _emission_proba

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


def _kernel_layers(table: np.ndarray) -> np.ndarray:
    """Return the layers the kernel draws from an uploaded table.

    The upload keeps the memory order of the host array, and the kernel
    reads ``cell_proba_icdf[k + ilam * NCELLPROBA]`` with ``NCELLPROBA``
    the number of columns of the table.
    """
    flat = table.ravel(order="K")
    n_lam, n = table.shape
    return np.array(
        [[flat[k + ilam * n] for k in range(n)] for ilam in range(n_lam)]
    )


@pytest.mark.parametrize("order", ["C", "F"])
def test_cell_proba_table_follows_the_kernel_indexing(order: str) -> None:
    """Each wavelength draws its layers from its own row of icdf_2d."""
    # wavelength 0 emits from layer 1, wavelength 1 from layer 3
    proba = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    icdf = np.asarray(icdf_2d(proba, 5), order=order)

    table = _cell_proba_table(icdf, 2)

    assert table.shape == (2, 5)
    np.testing.assert_array_equal(
        _kernel_layers(table), [[1, 1, 1, 1, 1], [3, 3, 3, 3, 3]]
    )


def test_cell_proba_table_keeps_each_wavelength_distribution() -> None:
    """A table of mixed rows is read row by row, not interleaved."""
    proba = np.array([[0.7, 0.2, 0.1], [0.05, 0.15, 0.8]])
    icdf = icdf_2d(proba, 4)

    np.testing.assert_array_equal(
        _kernel_layers(_cell_proba_table(icdf, 2)), icdf
    )


@pytest.mark.parametrize(
    ("cell_proba", "error", "match"),
    [
        (np.zeros((2, 5)), TypeError, "int64"),
        (np.zeros((2, 5), dtype=np.int32), TypeError, "int64"),
        (np.zeros((5, 2), dtype=np.int64), ValueError, "one row"),
        (np.zeros((3, 5), dtype=np.int64), ValueError, "one row"),
        (np.zeros(5, dtype=np.int64), ValueError, "one row"),
        (np.zeros((2, 0), dtype=np.int64), ValueError, "one row"),
    ],
    ids=["float", "int32", "transposed", "n_lam", "1-D", "empty"],
)
def test_cell_proba_table_refused(
    cell_proba: np.ndarray, error: type[Exception], match: str
) -> None:
    """Check that a table the kernel would misread is refused."""
    with pytest.raises(error, match=match):
        _cell_proba_table(cell_proba, 2)
