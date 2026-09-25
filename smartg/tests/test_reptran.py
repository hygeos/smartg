"""Focused unit tests for REPTRAN channel utilities."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pytest
import xarray as xr

from smartg.atmosphere import ProfileBase, od2k
from smartg.reptran import (
    ReadCrs,
    Reptran,
    ReptranBand,
    ReptranIbandList,
    dir_reptran,
    reduce_reptran,
    reptran_emission,
)


@pytest.fixture
def synthetic_reptran() -> Reptran:
    """Build a small in-memory REPTRAN object for unit tests."""
    reptran = object.__new__(Reptran)
    reptran.fname = Path("synthetic.cdf")
    reptran.thermal = False
    reptran.wvl = np.array([400.0, 500.0, 600.0, 700.0])
    reptran.extra = np.array([1.0, 2.0, 3.0, 4.0])
    reptran.wvl_integral = np.array([200.0, 200.0])
    reptran.nwvl_in_band = np.array([2, 2])
    reptran.iwvl = np.array([[1, 3], [2, 4]])
    reptran.iwvl_weight = np.array([[1.0, 0.25], [0.5, 0.75]])
    reptran.cross_section_source = np.zeros((4, 8), dtype=int)
    reptran.band_names = [
        "bandfrom 390 to 510 nm",
        "synthetic channel",
    ]
    return reptran


def test_reptran_band_parses_limits_and_falls_back(
    synthetic_reptran: Reptran,
) -> None:
    """Parse channel limits and derive them for an unstructured name."""
    parsed = ReptranBand(synthetic_reptran, 0)
    fallback = ReptranBand(synthetic_reptran, 1)

    assert parsed.wmin == pytest.approx(390.0)
    assert parsed.wmax == pytest.approx(510.0)
    assert fallback.w == pytest.approx(650.0)
    assert fallback.wmin == pytest.approx(550.0)
    assert fallback.wmax == pytest.approx(750.0)
    assert parsed.fname == Path("synthetic.cdf")


def test_reptran_band_iterates_internal_bands(
    synthetic_reptran: Reptran,
) -> None:
    """Expose internal-band indices, wavelengths, and weights."""
    band = ReptranBand(synthetic_reptran, 0)
    internal_bands = list(band.ibands())

    assert len(internal_bands) == 2
    assert [internal.index for internal in internal_bands] == [0, 1]
    assert [internal.w for internal in internal_bands] == [400.0, 500.0]
    assert [internal.weight for internal in internal_bands] == [1.0, 0.5]


def test_reptran_selection_and_metadata(
    synthetic_reptran: Reptran,
) -> None:
    """Select channels and calculate groups, names, and weights."""
    reptran = synthetic_reptran
    selected = reptran.to_smartg(
        include="synthetic",
        lmin=550.0,
        lmax=750.0,
    )
    all_bands = ReptranIbandList(
        [internal for band in reptran.bands() for internal in band.ibands()]
    )
    weights = all_bands.get_weights()

    assert len(selected.l) == 2
    assert [internal.w for internal in selected.l] == [600.0, 700.0]
    np.testing.assert_array_equal(all_bands.get_groups(), [0, 0, 1, 1])
    assert set(all_bands.get_names()) == {
        "bandfrom 390 to 510 nm",
        "synthetic channel",
    }
    np.testing.assert_allclose(weights[0].to_numpy(), [1.0, 0.5, 0.25, 0.75])
    np.testing.assert_allclose(
        weights[3].to_numpy(), [200.0, 200.0, 200.0, 200.0]
    )


def test_reduce_reptran_uses_channel_weights(
    synthetic_reptran: Reptran,
) -> None:
    """Reduce a spectral variable to one value per channel."""
    ibands = ReptranIbandList(
        [
            internal
            for band in synthetic_reptran.bands()
            for internal in band.ibands()
        ]
    )
    dataset = xr.Dataset(
        {
            "I_test": (
                ("wavelength",),
                np.array([10.0, 20.0, 30.0, 40.0]),
                {"desc": "I_test radiance"},
            ),
            "temperature": (
                ("wavelength",),
                np.array([1.0, 2.0, 3.0, 4.0]),
                {"desc": "temperature"},
            ),
        },
        coords={"wavelength": synthetic_reptran.wvl},
    )

    reduced = reduce_reptran(dataset, ibands)

    assert list(reduced.data_vars) == ["I_test"]
    np.testing.assert_allclose(
        reduced["I_test"].to_numpy(), [40.0 / 3.0, 37.5]
    )
    np.testing.assert_array_equal(
        reduced.wavelength.to_numpy(), [450.0, 650.0]
    )


SPECIES = ["H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2", "N2"]


@pytest.mark.parametrize(
    "fname", ["reptran_solar_coarse", "reptran_thermal_coarse"]
)
def test_species_follow_the_file(fname: str) -> None:
    """The absorber slots follow the species_name of the file."""
    with xr.open_dataset(dir_reptran / f"{fname}.cdf") as dataset:
        names = [
            name.decode().strip()
            for name in dataset["species_name"].to_numpy()
        ]
    reptran = Reptran(fname)
    iband = next(reptran.band(0).ibands())

    assert names == SPECIES
    assert iband.species == names


class _UnitCrossSection:
    """Stand in for ReadCrs with a cross section of one everywhere.

    As in the REPTRAN files, only H2O has a mixing ratio axis.
    """

    opened: ClassVar[list[Path]] = []

    def __init__(self, fname: Path, iband: int) -> None:
        self.opened.append(Path(fname))
        self.pressure = np.array([1e3, 1e5])
        self.t_ref = np.array([250.0, 250.0])
        self.t_pert = np.array([-50.0, 0.0, 50.0])
        h2o = Path(fname).name.endswith("H2O")
        self.vmrs = np.array([0.0, 1.0]) if h2o else np.array([0.0])
        self.xsec = np.ones((3, self.vmrs.size, 2))


@pytest.mark.parametrize("slot", range(len(SPECIES)))
def test_calc_profile_scales_each_species_by_its_density(
    synthetic_reptran: Reptran,
    monkeypatch: pytest.MonkeyPatch,
    slot: int,
) -> None:
    """Each absorber scales its own lookup table by its own density."""
    monkeypatch.setattr("smartg.reptran.ReadCrs", _UnitCrossSection)
    _UnitCrossSection.opened = []
    densities = {
        name: 1e10 * (index + 1) for index, name in enumerate(SPECIES)
    }
    prof = SimpleNamespace(
        t=np.array([250.0, 250.0]),
        p=np.array([100.0, 500.0]),
        dens_air=np.full(2, 1e19),
        dens_no2=np.full(2, 1e14),
        **{
            f"dens_{name.lower()}": np.full(2, value)
            for name, value in densities.items()
        },
    )
    iband = next(ReptranBand(synthetic_reptran, 0).ibands())
    iband.crs_source = np.zeros(len(SPECIES), dtype=int)
    iband.crs_source[slot] = 1

    kabs = iband.calc_profile(cast(ProfileBase, prof))

    species = SPECIES[slot]
    assert [path.name for path in _UnitCrossSection.opened] == [
        f"synthetic.lookup.{species}"
    ]
    np.testing.assert_allclose(kabs, densities[species] * 1e-11)


def test_channel_names_are_text() -> None:
    """Channel names read from a file are text, found by band()."""
    reptran = Reptran("reptran_solar_msg")
    band = reptran.band("msg1_seviri_ch006")

    assert reptran.band_names[0] == "msg1_seviri_ch006"
    assert band.band == 0
    assert band.name == "msg1_seviri_ch006"


def test_channel_name_spaces_are_ignored() -> None:
    """A libRadtran band name is found with or without its spaces."""
    reptran = Reptran("reptran_solar_coarse")
    band = reptran.band(0)

    assert band.name == "bandfrom119.9976to120.0192nm"
    assert (band.wmin, band.wmax) == (119.9976, 120.0192)
    assert reptran.band("band from  119.9976 to  120.0192 nm").band == 0
    assert reptran.band(band.name).band == 0


def test_get_names_follow_the_reduced_axis() -> None:
    """Name i labels channel i of the reduce_reptran output."""
    reptran = Reptran("reptran_solar_msg")
    ibands = reptran.to_smartg(include="msg1")
    wavelength = np.array([iband.w for iband in ibands.l], dtype=np.float32)
    dataset = xr.Dataset(
        {"I_test": (("wavelength",), np.ones(wavelength.size))},
        coords={"wavelength": wavelength},
    )
    reduced = reduce_reptran(dataset, ibands)
    names = ibands.get_names()
    centres = [np.mean(reptran.band(name).awvl) for name in names]

    assert len(names) == reduced.sizes["wavelength"] > 2
    np.testing.assert_allclose(centres, reduced.wavelength, rtol=1e-6)
    assert ReptranIbandList(ibands.l[::-1]).get_names() == names


def _write_lookup(fname: Path) -> None:
    """Write an O2 lookup table of two internal bands, 2 and 3 m2."""
    xsec = np.empty((3, 1, 2, 2), dtype=np.float32)
    xsec[:, :, 0, :] = 2.0
    xsec[:, :, 1, :] = 3.0
    xr.Dataset(
        {
            "wvl_index": (("nwvl",), np.array([1, 2], dtype=np.int32)),
            "pressure": (("n_pressure",), np.array([1e3, 1e5])),
            "t_ref": (("n_pressure",), np.array([250.0, 250.0])),
            "t_pert": (("n_t_pert",), np.array([-50.0, 0.0, 50.0])),
            "vmrs": (("n_vmrs",), np.array([0.0])),
            "xsec": (("n_t_pert", "n_vmrs", "nwvl", "n_pressure"), xsec),
        }
    ).to_netcdf(fname)


def test_read_crs_keeps_the_directory(tmp_path: Path) -> None:
    """A lookup table path is read from its own directory."""
    _write_lookup(tmp_path / "custom.lookup.O2.cdf")

    crs = ReadCrs(tmp_path / "custom.lookup.O2", 2)

    assert crs.xsec.shape == (3, 1, 2)
    np.testing.assert_array_equal(crs.xsec, 3.0)
    np.testing.assert_array_equal(crs.pressure, [1e3, 1e5])


def test_read_crs_bare_name_uses_auxdata() -> None:
    """A bare lookup table name is read from the REPTRAN auxdata."""
    fname = "reptran_solar_msg.lookup.O2"
    with xr.open_dataset(dir_reptran / f"{fname}.cdf") as dataset:
        iband = int(dataset["wvl_index"][-1])
        xsec = dataset["xsec"][:, :, -1, :].to_numpy()

    crs = ReadCrs(fname, iband)

    assert crs.fname == dir_reptran / fname
    np.testing.assert_array_equal(crs.xsec, xsec)


def test_calc_profile_reads_the_tables_of_its_file(
    synthetic_reptran: Reptran, tmp_path: Path
) -> None:
    """A REPTRAN file out of the auxdata reads its own lookup tables."""
    _write_lookup(tmp_path / "custom.lookup.O2.cdf")
    synthetic_reptran.fname = tmp_path / "custom.cdf"
    iband = next(ReptranBand(synthetic_reptran, 0).ibands())
    iband.crs_source = np.zeros(len(SPECIES), dtype=int)
    iband.crs_source[SPECIES.index("O2")] = 1
    prof = SimpleNamespace(
        t=np.array([250.0, 250.0]),
        p=np.array([100.0, 500.0]),
        dens_air=np.full(2, 1e19),
        **{f"dens_{name.lower()}": np.full(2, 1e10) for name in SPECIES},
    )

    kabs = iband.calc_profile(cast(ProfileBase, prof))

    np.testing.assert_allclose(kabs, 2.0 * 1e10 * 1e-11)


def test_thermal_bandwidth_is_in_nanometres(
    synthetic_reptran: Reptran,
) -> None:
    """A thermal wavenumber integral becomes a bandwidth in nm."""
    synthetic_reptran.thermal = True
    parsed = ReptranBand(synthetic_reptran, 0)
    fallback = ReptranBand(synthetic_reptran, 1)
    # 200 cm-1 times the weighted mean of the squared wavelengths
    dl = 200.0 * (0.25 * 600.0**2 + 0.75 * 700.0**2) * 1e-7
    ibands = ReptranIbandList([*parsed.ibands(), *fallback.ibands()])

    assert parsed.dl == pytest.approx(120.0)
    assert fallback.dl == pytest.approx(dl)
    assert fallback.wmax - fallback.wmin == pytest.approx(dl)
    np.testing.assert_allclose(
        ibands.get_weights()[3].to_numpy(), [120.0, 120.0, dl, dl]
    )


def test_thermal_bandwidth_matches_the_solar_file() -> None:
    """A channel in both files gets about the same bandwidth in nm."""
    solar = Reptran("reptran_solar_msg").band("msg1_seviri_ch039")
    thermal = Reptran("reptran_thermal_msg").band("msg1_seviri_ch039")
    coarse = Reptran("reptran_thermal_coarse").band(0)

    assert not Reptran("reptran_solar_msg").thermal
    assert solar.dl == solar.r_int
    assert thermal.r_int == pytest.approx(365.6, abs=0.1)  # cm-1
    assert thermal.dl == pytest.approx(solar.r_int, rel=0.01)
    assert thermal.wmax - thermal.wmin == pytest.approx(thermal.dl)
    assert coarse.dl == pytest.approx(2509.4103 - 2500.0)


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
    ibands = ReptranIbandList(
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

    emission = reptran_emission(dataset, ibands)

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
        attrs={"wavelength": "[400.]"},
    )
    if scalar_coordinate:
        dataset = dataset.assign_coords(wavelength=400.0)
    return dataset


@pytest.mark.parametrize(
    "scalar_coordinate", [False, True], ids=["attribute", "coordinate"]
)
def test_reduce_a_single_internal_band(
    synthetic_reptran: Reptran, scalar_coordinate: bool
) -> None:
    """A run on one internal band reduces to its own values."""
    ibands = ReptranIbandList([ReptranBand(synthetic_reptran, 0).iband(0)])
    dataset = _single_band_run(scalar_coordinate)

    reduced = reduce_reptran(dataset, ibands)

    np.testing.assert_array_equal(reduced.wavelength, [450.0])
    np.testing.assert_allclose(
        reduced["I_up (TOA)"].sel(wavelength=450.0),
        dataset["I_up (TOA)"],
    )
    assert reduced["I_up (TOA)"].attrs == {"desc": "I_up (TOA)"}


def test_reduce_without_wavelength_needs_one_internal_band(
    synthetic_reptran: Reptran,
) -> None:
    """Several internal bands need a wavelength dimension."""
    ibands = ReptranIbandList(list(ReptranBand(synthetic_reptran, 0).ibands()))

    with pytest.raises(ValueError, match="holds 2 internal bands"):
        reduce_reptran(_single_band_run(False), ibands)
