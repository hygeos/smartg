"""Focused unit tests for KDIS channel utilities."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr

from smartg.atmosphere import od2k
from smartg.kdis import (
    Kdis,
    KdisIband,
    KdisIbandList,
    kdis_emission,
    reduce_kdis,
)

H = 6.62607015e-34  # J s
C = 299792458.0  # m s-1
K_B = 1.380649e-23  # J K-1


def _iband(**fields: Any) -> KdisIband:
    """Stand in for a KdisIband with only the fields the tests read."""
    return cast(KdisIband, SimpleNamespace(**fields))


def test_reduce_kdis_uses_channel_weights() -> None:
    """Reduce internal KDIS bands to one value per channel."""
    ibands = KdisIbandList(
        [
            _iband(w=450.0, weight=1.0, ex=1.0, dl=100.0),
            _iband(w=450.0, weight=0.5, ex=1.0, dl=100.0),
            _iband(w=650.0, weight=2.0, ex=1.0, dl=200.0),
        ]
    )
    dataset = xr.Dataset(
        {
            "I_test": (
                ("wavelength",),
                np.array([10.0, 20.0, 30.0]),
                {"desc": "I_test radiance"},
            ),
            "N_test": (
                ("wavelength",),
                np.array([1.0, 2.0, 3.0]),
                {"desc": "N_test photons"},
            ),
            "temperature": (
                ("wavelength",),
                np.array([100.0, 200.0, 300.0]),
                {"desc": "temperature"},
            ),
        },
        coords={"wavelength": np.array([450.0, 450.0, 650.0])},
        attrs={"source": "synthetic"},
    )

    reduced = reduce_kdis(dataset, ibands)

    assert list(reduced.data_vars) == ["I_test", "N_test"]
    assert reduced.attrs == {"source": "synthetic"}
    assert reduced["I_test"].attrs == {"desc": "I_test radiance"}
    np.testing.assert_allclose(
        reduced["I_test"].to_numpy(), [40.0 / 3.0, 30.0]
    )
    np.testing.assert_allclose(reduced["N_test"].to_numpy(), [4.0 / 3.0, 3.0])
    np.testing.assert_array_equal(
        reduced.wavelength.to_numpy(), [450.0, 650.0]
    )


def test_kdis_emission_returns_xarray_data_array() -> None:
    """Calculate the KDIS emission from xarray coordinates and dims."""
    ibands = KdisIbandList(
        [
            _iband(
                w=450.0,
                weight=1.0,
                ex=1.0,
                dl=100.0,
                band=SimpleNamespace(wmin=400.0, wmax=500.0, band=0),
            ),
            _iband(
                w=450.0,
                weight=0.5,
                ex=1.0,
                dl=100.0,
                band=SimpleNamespace(wmin=400.0, wmax=500.0, band=0),
            ),
            _iband(
                w=650.0,
                weight=2.0,
                ex=1.0,
                dl=200.0,
                band=SimpleNamespace(wmin=600.0, wmax=700.0, band=1),
            ),
        ]
    )
    dataset = xr.Dataset(
        {
            "OD_abs_atm": (
                ("wavelength", "z_atm"),
                np.array(
                    [
                        [0.0, 1.0],
                        [0.0, 2.0],
                        [0.0, 3.0],
                    ]
                ),
            ),
            "T_atm": (("z_atm",), np.array([280.0, 290.0])),
        },
        coords={
            "wavelength": np.array([450.0, 450.0, 650.0]),
            "z_atm": np.array([0.0, -1.0]),
        },
    )

    emission = kdis_emission(dataset, ibands)

    assert isinstance(emission, xr.DataArray)
    assert emission.name == "emission"
    assert emission.dims == ("wavelength", "z_atm")
    np.testing.assert_array_equal(
        emission.wavelength.to_numpy(), [450.0, 450.0, 650.0]
    )
    np.testing.assert_array_equal(emission.z_atm.to_numpy(), [0.0, 1000.0])
    assert emission.shape == (3, 2)


def _planck_channel(
    wmin: float, wmax: float, temperature: float
) -> float:
    """Return the Planck radiance averaged over [wmin, wmax] in nm.

    In W m-2 sr-1 nm-1, integrated here with the trapezoid rule on a
    fine grid.
    """
    wavelength = np.linspace(wmin, wmax, 4001) * 1e-9  # m
    radiance = (
        2 * H * C**2 / wavelength**5
        / np.expm1(H * C / (wavelength * K_B * temperature))
    )  # W m-3 sr-1
    return float(np.trapezoid(radiance, wavelength)) / (wmax - wmin)


def test_kdis_emission_values() -> None:
    """Check the emission against kabs times the channel Planck mean.

    Thermal infrared channels, listed out of order, on the descending
    altitudes in km of an Atm1D profile.
    """
    channels = [(12000.0, 13000.0), (10000.0, 11000.0), (12000.0, 13000.0)]
    ibands = KdisIbandList(
        [
            _iband(
                w=0.5 * (wmin + wmax), weight=1.0, ex=1.0, dl=wmax - wmin,
                band=SimpleNamespace(wmin=wmin, wmax=wmax, band=0),
            )
            for wmin, wmax in channels
        ]
    )
    t_atm = np.array([220.0, 250.0, 290.0])
    # cumulated absorption optical depths, from the top at 2 km
    od_abs = np.array(
        [[0.0, 0.5, 1.5], [0.0, 0.1, 0.4], [0.0, 2.0, 2.5]]
    )
    dataset = xr.Dataset(
        {
            "OD_abs_atm": (("wavelength", "z_atm"), od_abs),
            "T_atm": (("z_atm",), t_atm),
        },
        coords={
            "wavelength": [12500.0, 10500.0, 12500.0],
            "z_atm": [2.0, 1.0, 0.0],
        },
    )

    emission = kdis_emission(dataset, ibands)

    # km-1 over 1 km layers, in m-1; nothing above the top level
    kabs = np.diff(od_abs, axis=1, prepend=0.0) * 1e-3
    expected = np.array(
        [
            [_planck_channel(wmin, wmax, t) for t in t_atm]
            for wmin, wmax in channels
        ]
    ) * kabs
    np.testing.assert_allclose(emission.to_numpy(), expected, rtol=1e-6)


def test_missing_model_raises(tmp_path: Path) -> None:
    """A missing KDIS file raises instead of exiting the interpreter."""
    with pytest.raises(FileNotFoundError, match="kdis_nomodel_def.dat"):
        Kdis("nomodel", dir_data=tmp_path)


def _planck_band_mean(
    wmin: float, wmax: float, temperature: np.ndarray
) -> np.ndarray:
    """Return the Planck radiance averaged over [wmin, wmax] nm.

    In W m-2 sr-1 nm-1, by the midpoint rule on 2000 intervals.
    """
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    edges = np.linspace(wmin, wmax, 2001) * 1e-9  # m
    wavelength = 0.5 * (edges[1:] + edges[:-1])[:, None]
    radiance = (
        2.0 * h * c**2 / wavelength**5
        / np.expm1(h * c / (wavelength * k * temperature[None, :]))
    )
    return radiance.mean(axis=0) * 1e-9


def test_emission_takes_the_planck_average_of_each_channel() -> None:
    """Each internal band gets the Planck average of its own channel.

    The channels are neither contiguous nor in wavelength order in the
    file, as with several intervals or several sensors.
    """
    channels = {  # file index: limits in nm
        7: (3500.0, 4000.0),
        2: (10000.0, 11000.0),
        4: (8000.0, 9000.0),
    }
    selection = [(7, 3700.0), (7, 3900.0), (4, 8500.0), (2, 10500.0)]
    ibands = KdisIbandList(
        [
            cast(
                Any,
                SimpleNamespace(
                    w=w,
                    band=SimpleNamespace(
                        band=index,
                        wmin=channels[index][0],
                        wmax=channels[index][1],
                    ),
                ),
            )
            for index, w in selection
        ]
    )
    t_atm = np.array([220.0, 250.0, 290.0])
    dataset = xr.Dataset(
        {
            "OD_abs_atm": (
                ("wavelength", "z_atm"),
                np.outer([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 3.0]),
            ),
            "T_atm": (("z_atm",), t_atm),
        },
        coords={
            "wavelength": [w for _, w in selection],
            "z_atm": [2.0, 1.0, 0.0],
        },
    )

    emission = kdis_emission(dataset, ibands)

    kabs = od2k(dataset, "OD_abs_atm") * 1e-3
    for i, (index, _) in enumerate(selection):
        expected = kabs[i] * _planck_band_mean(*channels[index], t_atm)
        np.testing.assert_allclose(emission[i], expected, rtol=1e-6)


def _single_band_run(scalar_coordinate: bool) -> xr.Dataset:
    """Return the output of a run on one wavelength, as Smartg gives it.

    It keeps the wavelength as an attribute, or as a scalar coordinate
    once a wavelength is selected from a spectral output.
    """
    dataset = xr.Dataset(
        {
            "I_up (TOA)": (
                ("Azimuth angles", "Zenith angles"),
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                {"desc": "I_up (TOA)"},
            )
        },
        attrs={"wavelength": "[650.]"},
    )
    if scalar_coordinate:
        dataset = dataset.assign_coords(wavelength=650.0)
    return dataset


@pytest.mark.parametrize(
    "scalar_coordinate", [False, True], ids=["attribute", "coordinate"]
)
def test_reduce_a_single_internal_band(scalar_coordinate: bool) -> None:
    """A run on one internal band reduces to its own values."""
    ibands = KdisIbandList([_iband(w=650.0, weight=2.0, ex=1.0, dl=200.0)])
    dataset = _single_band_run(scalar_coordinate)

    reduced = reduce_kdis(dataset, ibands)

    np.testing.assert_array_equal(reduced.wavelength, [650.0])
    np.testing.assert_allclose(
        reduced["I_up (TOA)"].sel(wavelength=650.0),
        dataset["I_up (TOA)"],
    )
    assert reduced["I_up (TOA)"].attrs == {"desc": "I_up (TOA)"}


def test_reduce_without_wavelength_needs_one_internal_band() -> None:
    """Several internal bands need a wavelength dimension."""
    iband = _iband(w=650.0, weight=2.0, ex=1.0, dl=200.0)
    ibands = KdisIbandList([iband, iband])

    with pytest.raises(ValueError, match="holds 2 internal bands"):
        reduce_kdis(_single_band_run(False), ibands)


@pytest.mark.parametrize(
    ("c_desc", "c"),
    [("density", [1e15, 1e17]), ("molar_fraction", [1e-7, 1e-3])],
)
def test_calc_profile_clips_only_the_interpolation_point(
    c_desc: str, c: list[float]
) -> None:
    """Densities out of the concentration axis still scale the k.

    With k = 1 over the whole table, a layer without H2O absorbs
    nothing and a dense layer absorbs in proportion to its density.
    """
    kdis = SimpleNamespace(
        nsp_c=1,
        species_c=["h2o"],
        iki_eff_c=np.zeros((1, 1, 1), dtype=int),
        ki_c=np.ones((1, 1, 1, 2, 2, 2)),
        p=np.array([1.0, 1000.0]),
        t=np.array([200.0, 300.0]),
        c=np.array(c),
        c_desc=c_desc,
        nsp=0,
    )
    band = SimpleNamespace(
        kdis=kdis,
        band=0,
        awvl=[500.0],
        awvl_weight=np.array([1.0]),
        solarflux=1.0,
        dl=10.0,
    )
    zeros = np.zeros(2)
    prof = SimpleNamespace(
        t=np.array([250.0, 250.0]),
        p=np.array([500.0, 500.0]),
        dens_air=np.full(2, 1e19),
        dens_h2o=np.array([0.0, 5e17]),
        **{
            f"dens_{name}": zeros
            for name in ["co2", "o3", "no2", "co", "ch4", "o2", "n2", "n2o",
                         "so2"]
        },
    )

    kabs = KdisIband(cast(Any, band), 0).calc_profile(cast(Any, prof))

    np.testing.assert_allclose(kabs, [0.0, 5e17 * 1e5])
