"""Focused unit tests for KDIS channel utilities."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import interpn

from smartg.atmosphere import Atm1D, od2k
from smartg.bandset import BandSet
from smartg.interp import interp2
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


def _v205_calc_profile(iband: Any, prof: Any) -> np.ndarray:
    """Return the absorption of a band as v2.0.5 computed it."""
    species = [
        "h2o", "co2", "o3", "no2", "co", "ch4", "o2", "n2", "n2o", "so2",
    ]
    kdis = iband.band.kdis
    temperature = prof.t.copy()
    pressure = prof.p.copy()
    profile_length = len(temperature)
    data_molecules = np.zeros(profile_length, np.float64)
    density_molecules = np.zeros((profile_length, 10), np.float64)
    for j, name in enumerate(species):
        density_molecules[:, j] = getattr(prof, f"dens_{name}")[:]
    for s in range(kdis.nsp_c):
        molecular_index = species.index(kdis.species_c[s])
        coefficient_index = kdis.iki_eff_c[s, iband.band.band, iband.index]
        table = kdis.ki_c[s, iband.band.band, coefficient_index, :, :, :]
        if kdis.c_desc == "density":
            concentration = density_molecules[:, molecular_index].copy()
        else:
            concentration = density_molecules[:, molecular_index] / (
                prof.dens_air.copy()
            )
        concentration[concentration > np.max(kdis.c)] = np.max(kdis.c) * 0.99
        concentration[concentration < np.min(kdis.c)] = np.min(kdis.c) * 1.01
        pressure[pressure > np.max(kdis.p)] = np.max(kdis.p) * 0.99
        pressure[pressure < np.min(kdis.p)] = np.min(kdis.p) * 1.01
        temperature[temperature > np.max(kdis.t)] = np.max(kdis.t) * 0.99
        temperature[temperature < np.min(kdis.t)] = np.min(kdis.t) * 1.01
        points = np.concatenate(
            (np.array([pressure]), np.array([temperature]),
             np.array([concentration])), axis=0,
        ).T
        data_molecules += (
            interpn((kdis.p, kdis.t, kdis.c), table, points)
            * density_molecules[:, molecular_index]
        )
    for s in range(kdis.nsp):
        molecular_index = species.index(kdis.species[s])
        coefficient_index = kdis.iki_eff[s, iband.band.band, iband.index]
        table = kdis.ki[s, iband.band.band, coefficient_index, :, :]
        data_molecules += (
            interp2(kdis.p, kdis.t, np.squeeze(table), pressure, temperature)
            * density_molecules[:, molecular_index]
        )
    return data_molecules * 1e5


def _profiles() -> dict[str, Any]:
    atm = Atm1D("afglt")
    return {
        "afglt": atm.prof,
        "afglus": Atm1D("afglus").prof,
        "regridded": atm.prof.regrid(np.array([100.0, 30.0, 10.0, 3.0,
                                               1.0, 0.0])),
    }


KDIS_MODELS = {
    "kato2": ("kato2", {"lmax": 4500}),
    "S2": ("SENTINEL2_1_MSI", {}),
    "SLSTR": ("SENTINEL3_1_SLSTR", {"lmax": 4500}),
}


@pytest.mark.parametrize("model", KDIS_MODELS)
def test_calc_profiles_as_band_by_band(model: str) -> None:
    """The batch gives each band its v2.0.5 absorption, bitwise."""
    name, kwargs = KDIS_MODELS[model]
    bands = Kdis(name).to_smartg(**kwargs).l
    for prof in _profiles().values():
        batch = KdisIband._calc_profiles(bands, prof)
        ref = np.stack([_v205_calc_profile(b, prof) for b in bands])
        assert batch.dtype == ref.dtype and batch.shape == ref.shape
        assert batch.tobytes() == ref.tobytes()
        one = bands[len(bands) // 2]
        assert (one.calc_profile(prof).tobytes()
                == _v205_calc_profile(one, prof).tobytes())


def _synthetic_kdis(nsp_c: int, nsp: int, c_desc: str = "density") -> Any:
    rng = np.random.default_rng(0)
    n_k = 4
    return SimpleNamespace(
        nsp_c=nsp_c,
        species_c=["h2o", "co2"][:nsp_c],
        iki_eff_c=rng.integers(0, n_k, (nsp_c, 2, 6)),
        ki_c=rng.random((nsp_c, 2, n_k, 3, 4, 2)),
        c=np.array([1e15, 1e18]) if c_desc == "density"
        else np.array([1e-7, 1e-2]),
        c_desc=c_desc,
        nsp=nsp,
        species=["o3", "o2"][:nsp],
        iki_eff=rng.integers(0, n_k, (nsp, 2, 6)),
        ki=rng.random((nsp, 2, n_k, 3, 4)),
        p=np.array([1.0, 100.0, 1000.0]),
        t=np.array([180.0, 220.0, 260.0, 300.0]),
    )


def _synthetic_bands(kdis: Any) -> list[KdisIband]:
    bands = []
    for b in range(2):
        band = SimpleNamespace(kdis=kdis, band=b, awvl=[500.0 + b] * 6,
                               awvl_weight=np.ones(6), solarflux=1.0,
                               dl=10.0)
        bands += [KdisIband(cast(Any, band), i) for i in (0, 3, 1, 3, 5)]
    return bands


def _synthetic_profile() -> Any:
    # pressures and temperatures beyond the axes, which are clipped
    # only when a species has a concentration axis
    return SimpleNamespace(
        t=np.array([170.0, 250.0, 310.0]),
        p=np.array([0.5, 300.0, 1200.0]),
        dens_air=np.full(3, 1e19),
        **{f"dens_{name}": np.array([1e14, 1e16, 2e18]) for name in
           ["h2o", "co2", "o3", "no2", "co", "ch4", "o2", "n2", "n2o",
            "so2"]},
    )


@pytest.mark.parametrize(
    ("nsp_c", "nsp", "c_desc"),
    [(0, 2, "density"), (2, 1, "density"), (2, 2, "molar_fraction")],
)
def test_calc_profiles_synthetic(nsp_c: int, nsp: int, c_desc: str) -> None:
    """Repeated k indices, points off the axes, both concentrations."""
    kdis = _synthetic_kdis(nsp_c, nsp, c_desc)
    bands = _synthetic_bands(kdis)
    prof = _synthetic_profile()
    batch = KdisIband._calc_profiles(cast(Any, bands), prof)
    ref = np.stack([_v205_calc_profile(b, prof) for b in bands])
    assert batch.tobytes() == ref.tobytes()


class _OwnBand(KdisIband):
    """A subclass with its own absorption."""

    def calc_profile(self, prof: Any) -> np.ndarray:
        return np.full(len(prof.t), 7.0)


class _OtherBand:
    """A band of another type."""

    w = 600.0

    def calc_profile(self, prof: Any) -> np.ndarray:
        return np.full(len(prof.t), 3.0)


def test_calc_profiles_of_mixed_bands() -> None:
    """Bands of two models, a subclass and another type, interleaved."""
    kato2 = Kdis("kato2").to_smartg(lmin=500, lmax=700).l
    s2 = Kdis("SENTINEL2_1_MSI").to_smartg(lmax=700).l
    own = _OwnBand(kato2[0].band, 0)
    bands: list[Any] = [kato2[0], s2[0], own, kato2[1], _OtherBand(), s2[1]]
    prof = _profiles()["afglt"]
    batch = KdisIband._calc_profiles(bands, prof)
    for row, band in zip(batch, bands, strict=True):
        ref = (band.calc_profile(prof) if isinstance(band, (_OwnBand,
                                                            _OtherBand))
               else _v205_calc_profile(band, prof))
        assert row.tobytes() == np.asarray(ref, np.float64).tobytes()


def test_bandset_computes_kdis_bands_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BandSet hands its KDIS bands to one batch, as float32 rows."""
    bands = Kdis("kato2").to_smartg(lmin=500, lmax=700).l
    prof = _profiles()["afglt"]
    calls = []
    batch = KdisIband._calc_profiles

    def counted(ibands: Any, p: Any) -> np.ndarray:
        calls.append(len(ibands))
        return batch(ibands, p)

    monkeypatch.setattr(KdisIband, "_calc_profiles", staticmethod(counted))
    tau = BandSet(bands).calc_profile(prof)
    assert calls == [len(bands)]
    ref = np.stack([_v205_calc_profile(b, prof) for b in bands])
    assert tau.dtype == np.float32
    assert tau.tobytes() == ref.astype(np.float32).tobytes()
