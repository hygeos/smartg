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

from smartg.histories import amf_from_cdist, compute_amf, get_histories

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
    d: list[float],
    s: list[float],
    w: list[float],
    nref: float = 0.0,
    d_oc: tuple[float, ...] = (),
) -> np.ndarray:
    """Return one history record, laid out as the kernel writes it."""
    flags = [0.0, nref, 0.0, 0.0, 0.0, 1.0, 0.0]
    return np.array([*d_oc, *d, *s, *w, *flags], dtype=np.float32)


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


@pytest.mark.parametrize(
    "attrs",
    [{"ALIS n_oce_abs": "2", "ALIS n_atm_abs": "3"}, None],
    ids=["attributes", "old-output"],
)
def test_get_histories_ocean_columns(attrs: dict[str, str] | None) -> None:
    """Check that the ocean path lengths, first in a record, are left.

    The distances used to be read from the start of the record, so the
    Stokes vector and the weights were read shifted.
    """
    record = _record(
        [1.0, 2.0, 3.0],
        [0.5, 0.1, 0.0, 0.0],
        [0.9, 0.8],
        nref=1.0,
        d_oc=(7.0, 8.0),
    )
    ds = _output([(0, record)], 4, attrs)
    ds.coords["z_oc"] = [0.0, -10.0, -20.0]
    _, s, d, w, _, nref, *_ = get_histories(ds)
    np.testing.assert_allclose(d, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(s, [[0.5, 0.1, 0.0, 0.0]], rtol=1e-6)
    np.testing.assert_allclose(w, [[0.9, 0.8]], rtol=1e-6)
    np.testing.assert_allclose(nref, [1.0])


def _cdist_output(
    amf: list[float], n_oce_abs: int = 0, zip_le: bool = False
) -> xr.Dataset:
    """
    Return a hist=False ALIS output, one AMF per sensor.

    The layers are 1 km thick, and the ocean rows, first on the
    'cdist_layer' axis, hold path lengths of 100 km.
    """
    n_sensor = len(amf)
    cdist = np.ones((n_oce_abs + N_ATM, n_sensor, 1, 2))
    cdist[:n_oce_abs, :, :, 1] = 100.0
    cdist[n_oce_abs:, :, :, 1] = np.asarray(amf)[None, :, None]
    ds = xr.Dataset()
    if zip_le:
        dims = ["cdist_layer", "sensor index", "Zenith angles", "iAMF"]
    else:
        cdist = cdist[:, :, None]
        dims = [
            "cdist_layer",
            "sensor index",
            "Azimuth angles",
            "Zenith angles",
            "iAMF",
        ]
    ds["cdist_up (TOA)"] = (dims, cdist)
    ds.coords["z_atm"] = np.linspace(3.0, 0.0, N_ATM + 1)
    ds.attrs.update(
        {"ALIS n_oce_abs": str(n_oce_abs), "ALIS n_atm_abs": str(N_ATM)}
    )
    return ds


@pytest.mark.parametrize("zip_le", [False, True])
@pytest.mark.parametrize("isensor", [0, 1])
def test_compute_amf_sensor(isensor: int, zip_le: bool) -> None:
    """Check that the sensor axis is not read as scatter classes.

    With two sensors of AMF 1 and 3, the AMF used to be their mean,
    with the sensors presented as scatter classes.
    """
    amf, _, cdist = compute_amf(
        _cdist_output([1.0, 3.0], zip_le=zip_le), isensor=isensor
    )
    assert cdist.shape == (N_ATM, 2)
    np.testing.assert_allclose(amf["AMF"], [1.0 + 2.0 * isensor] * N_ATM)
    assert "AMF_cls" not in amf


def test_compute_amf_ocean_rows() -> None:
    """Check that the ocean layers of the cdist output are left out."""
    amf, _, cdist = compute_amf(_cdist_output([2.0], n_oce_abs=2))
    assert cdist.shape == (N_ATM, 2)
    np.testing.assert_allclose(amf["AMF"], [2.0] * N_ATM)


def _hist_amf_output(n_lam: int, n_low: int, step: int) -> xr.Dataset:
    """Return a hist=True output: half the photons reflected once.

    The reflected photons travel 3 km in each layer, the others 1 km,
    and the scattering corrections grow with the wavelength.
    """
    w = [float(k) for k in range(1, n_low + 1)]
    records = [
        (0, _record([1.0 + 2.0 * (i % 2)] * N_ATM, [1.0, 0, 0, 0], w,
                    nref=float(i % 2)))
        for i in range(10)
    ]
    ds = _output(records, 20, {"hist records": "10"})
    ds.coords["wavelength"] = np.linspace(500.0, 520.0, n_lam)
    ds.attrs["ALIS wavelength step"] = str(step)
    return ds


def test_compute_amf_default_wavelength_lr() -> None:
    """Check the default low resolution grid, n_low < n_lam.

    The default was every wavelength of the run, which jnp.interp
    refused against the n_low corrections of each history.
    """
    ds = _hist_amf_output(21, 5, 5)
    amf, *_ = compute_amf(ds, alb_ref=0.5)
    wavelength_lr = ds["wavelength"].values[::5]
    expected, *_ = compute_amf(
        ds, alb_ref=0.5, wavelength_lr_r=wavelength_lr
    )
    np.testing.assert_allclose(amf["AMF"], expected["AMF"])
    # median of the LR grid, whose correction is 3 for every photon
    np.testing.assert_allclose(amf["W"], [5 * 3 * (1 + 0.5)] * N_ATM)


@pytest.mark.parametrize(
    ("alb_ref", "amf_expected"), [(0.0, 1.0), (1.0, 2.0), (1e-6, 1.0)]
)
def test_compute_amf_black_surface(
    alb_ref: float, amf_expected: float
) -> None:
    """Check that alb_ref=0 drops the reflected photons.

    alb_ref=0 was taken for a white surface, alb_ref=1.
    """
    amf, *_ = compute_amf(_hist_amf_output(3, 3, 1), alb_ref=alb_ref)
    np.testing.assert_allclose(amf["AMF"], [amf_expected] * N_ATM, rtol=1e-5)


def test_amf_from_cdist_keys() -> None:
    """Check the keys of scatter classes with and without variance."""
    keys_cls = {"AMF", "W", "mean_dist", "std_AMF", "AMF_cls", "W_cls",
                "std_AMF_cls"}
    thick = np.ones(N_ATM)
    two = amf_from_cdist(np.ones((N_ATM, 4, 2)), thick)
    assert set(two) == keys_cls
    three = amf_from_cdist(np.ones((N_ATM, 4, 3)), thick)
    assert set(three) == keys_cls | {"var_within", "var_between"}
