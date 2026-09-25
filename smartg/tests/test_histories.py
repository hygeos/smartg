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
    directions: list[tuple[int, int, int]] | None = None,
    shape: tuple[int, int, int] = (1, 1, 1),
) -> xr.Dataset:
    """
    Return a Smartg output holding the given histories.

    The records take their slots in turn from one counter shared by
    the two levels and the directions, as in the kernel, which drops
    them past max_hist. Each fills the fields of its (theta, sensor,
    phi) direction only, in the shape of these three axes.
    """
    n_cols = len(records[0][1])
    if directions is None:
        directions = [(0, 0, 0)] * len(records)
    hist = np.zeros((2, max_hist, n_cols, *shape), dtype=np.float32)
    for slot, ((level, record), (ith, isen, iph)) in enumerate(
        zip(records[:max_hist], directions, strict=False)
    ):
        hist[level, slot, :, ith, isen, iph] = record
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


@pytest.mark.parametrize(
    ("shape", "idir", "isensor"),
    [((2, 1, 1), 1, 0), ((1, 2, 1), 0, 1), ((2, 1, 3), 5, 0)],
    ids=["zenith", "sensor", "zenith-azimuth"],
)
def test_get_histories_direction(
    shape: tuple[int, int, int], idir: int, isensor: int
) -> None:
    """Check that the records of one direction and sensor are read."""
    n_theta, n_sensor, n_phi = shape
    wanted = (idir // n_phi, isensor, idir % n_phi)
    records, directions = [], []
    for i in range(12):
        ith = i % n_theta
        isen = (i // n_theta) % n_sensor
        iph = (i // (n_theta * n_sensor)) % n_phi
        record = _record(
            [float(i)] * N_ATM, [i + 0.5, 0.0, 0.0, 0.0], [1.0] * N_LOW
        )
        records.append((0, record))
        directions.append((ith, isen, iph))
    ds = _output(records, 20, {"hist records": "12"}, directions, shape)
    _, s, d, *_ = get_histories(ds, idir=idir, isensor=isensor)
    expected = [i for i, dr in enumerate(directions) if dr == wanted]
    assert len(expected) > 0
    np.testing.assert_array_equal(d[:, 0], expected)
    np.testing.assert_array_equal(s[:, 0], np.add(expected, 0.5))


def test_get_histories_direction_invalid() -> None:
    """Check that a direction beyond the histories is refused."""
    ds = _output(_alternating(4), 10, {"hist records": "4"})
    with pytest.raises(IndexError, match="idir=1"):
        get_histories(ds, idir=1)
