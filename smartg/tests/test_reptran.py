"""Focused unit tests for REPTRAN channel utilities."""

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast

import numpy as np
import pytest
import xarray as xr

from smartg.atmosphere import ProfileBase
from smartg.reptran import (
    Reptran,
    ReptranBand,
    ReptranIbandList,
    dir_reptran,
    reduce_reptran,
)


@pytest.fixture
def synthetic_reptran() -> Reptran:
    """Build a small in-memory REPTRAN object for unit tests."""
    reptran = object.__new__(Reptran)
    reptran.fname = Path("synthetic.cdf")
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
