"""Tests of the photon histories post-processing, on the CPU.

They build Smartg outputs by hand, with the record layout the kernel
writes, and run the `smartg.histories` functions on them, without a
GPU. The replay of real histories is in test_smartg_jax.py.
"""

import os

# Must be set before importing JAX so only the CPU backend is used
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest
import xarray as xr

pytest.importorskip(
    "jax", reason="cannot test this since the jax package is not installed."
)

from smartg.histories import get_histories

N_ATM = 3
N_LOW = 2
HIST_DIMS = (
    "hist_level",
    "hist_photon",
    "hist_record",
    "hist_theta",
    "hist_sensor",
    "hist_phi",
)


def _record(
    d: list[float], s: list[float], w: list[float], nref: float = 0.0
) -> np.ndarray:
    """Return one history record, laid out as the kernel writes it."""
    flags = [0.0, nref, 0.0, 0.0, 0.0, 1.0, 0.0]
    return np.array([*d, *s, *w, *flags], dtype=np.float32)


def _output(
    records: list[tuple[int, np.ndarray]],
    max_hist: int,
    attrs: dict[str, str] | None = None,
) -> xr.Dataset:
    """
    Return a Smartg output holding the given histories.

    The records take their slots in turn from one counter shared by
    the two levels, as in the kernel, which drops them past max_hist.
    """
    n_cols = len(records[0][1])
    hist = np.zeros((2, max_hist, n_cols, 1, 1, 1), dtype=np.float32)
    for slot, (level, record) in enumerate(records[:max_hist]):
        hist[level, slot, :, 0, 0, 0] = record
    ds = xr.Dataset()
    ds["histories"] = (HIST_DIMS, hist)
    ds["Nphotons_in"] = (
        ("sensor_in", "wavelength_in"),
        np.full((1, N_LOW), 1000, dtype=np.uint64),
    )
    ds.coords["z_atm"] = np.linspace(3.0, 0.0, N_ATM + 1)
    ds.attrs.update(attrs or {})
    return ds


def _alternating(n_records: int) -> list[tuple[int, np.ndarray]]:
    """Return records going to the TOA and 0+ levels in turn."""
    return [
        (i % 2, _record([1.0] * N_ATM, [1.0, 0.0, 0.0, 0.0], [1.0] * N_LOW))
        for i in range(n_records)
    ]


@pytest.mark.parametrize(
    "attrs", [{"hist records": "30"}, None], ids=["counter", "old-output"]
)
def test_get_histories_saturated_two_levels(
    attrs: dict[str, str] | None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check the saturation warning when both levels are recorded.

    Each level holds only its share of the slots, so counting the
    records of one level never reached max_hist.
    """
    get_histories(_output(_alternating(30), 10, attrs), level=0)
    assert "History buffer saturated" in capsys.readouterr().out


@pytest.mark.parametrize("n_records", [4, 10])
def test_get_histories_not_saturated(
    n_records: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check that a buffer holding every history does not warn."""
    ds = _output(
        _alternating(n_records), 10, {"hist records": str(n_records)}
    )
    get_histories(ds, level=0)
    assert "saturated" not in capsys.readouterr().out
