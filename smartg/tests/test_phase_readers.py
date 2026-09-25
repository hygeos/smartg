"""The two layouts of the phase matrix file readers.

Every reader returns either the phase matrix laid on a 1D profile
(``output_sg_ready=True``: interpolated at a wavelength and a humidity
or effective radius, on a ``z_phase`` axis, for ``AerOPAC`` / ``Cloud``)
or the table on the axes of the file (``output_sg_ready=False``: every
wavelength and ``hum`` / ``reff`` value, for ``Cloud3D`` / ``Aer3D``).
Checked on the IPRT phase B libRadtran files and on the OPAC auxdata,
without a GPU.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

import smartg.phase
from smartg.atmosphere import Aer3D, Cloud3D
from smartg.config import DIR_AUXDATA
from smartg.phase import (
    integ_phase,
    read_phase,
    read_phase_cdf,
    read_phase_nc,
    theta_grid,
    union_theta_grid,
)

OPT_PROP = DIR_AUXDATA / "IPRT" / "phaseB" / "opt_prop"
WC_CDF = OPT_PROP / "watercloud_670.mie.cdf"
WASO_CDF = OPT_PROP / "waso_670.mie.cdf"
# 400 to 700 nm by steps of 50 nm
DESERT_CDF = DIR_AUXDATA / "IPRT" / "phase3" / "opt_prop" / "desert.cdf"
DESERT_NC = DIR_AUXDATA / "aerosols" / "OPAC" / "mixtures" / "desert_sol.nc"
WC_NC = DIR_AUXDATA / "clouds" / "wc_sol.nc"

PROFILE_DIMS = ("wavelength_phase", "z_phase", "nphamat", "theta_atm")


def _file_grids(
    fname: Path,
) -> tuple[xr.Dataset, list[tuple[tuple[int, ...], NDArray[np.float64]]]]:
    """Return the distinct angle grids of a cdf file, sorted."""
    ds = xr.open_dataset(fname)
    theta, ntheta = ds["theta"].values, ds["ntheta"].values
    grids = []
    for idx in np.ndindex(ntheta.shape):
        grid = np.sort(theta[idx][: int(ntheta[idx])].astype(np.float64))
        if not any(np.array_equal(grid, g) for g in grids):
            grids.append((idx, grid))
    return ds, grids


# --------------------------------------------------------------------
# read_phase_cdf
# --------------------------------------------------------------------


def test_cdf_table_layout() -> None:
    """Check the axes, the shape and the type of a cdf table."""
    table = read_phase_cdf(WC_CDF, output_sg_ready=False, normalize=False)
    assert table.dims == ("wavelength_phase", "reff", "nphamat", "theta_atm")
    assert table.shape == (1, 25, 4, 18001)
    assert table.dtype == np.float64
    assert table.name == "phase_atm"
    ds = xr.open_dataset(WC_CDF)
    assert np.array_equal(table.coords["reff"].values, ds["reff"].values)
    np.testing.assert_allclose(table.coords["wavelength_phase"], [670.0])


def test_cdf_kind_names_the_angle_axis() -> None:
    """Check that kind='oc' names the axis theta_oc."""
    table = read_phase_cdf(WASO_CDF, kind="oc", output_sg_ready=False)
    assert table.dims[-1] == "theta_oc"
    assert table.name == "phase_oc"


def test_cdf_profile_layout_is_the_table_interpolated() -> None:
    """Check that the profile is the table read at the targets."""
    table = read_phase_cdf(WC_CDF, output_sg_ready=False, normalize=False)
    profile = read_phase_cdf(
        WC_CDF, z_rh_reff=[10.0, 12.0], pfgrid=[5.0, 2.0, 0.0],
        normalize=False,
    )
    assert profile.dims == PROFILE_DIMS
    np.testing.assert_allclose(profile.coords["z_phase"], [2.0, 0.0])
    expected = table.interp(reff=[10.0, 12.0])
    np.testing.assert_allclose(profile.values, expected.values)


def test_cdf_scalar_targets_keep_four_dimensions() -> None:
    # a scalar wavelength or humidity used to drop its dimension, and
    # a single humidity without pfgrid used to be refused
    """Check that a scalar target still gives four axes."""
    profile = read_phase_cdf(WC_CDF, z_rh_reff=10.0, normalize=False)
    assert profile.dims == PROFILE_DIMS
    assert profile.shape == (1, 1, 4, 18001)
    np.testing.assert_allclose(profile.coords["z_phase"], [0.0])
    single = read_phase_cdf(WASO_CDF, wavelength_phase=670.0)
    assert single.dims == PROFILE_DIMS
    assert single.shape[:2] == (1, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wavelength_phase": 670.0},
        {"pfgrid": [5.0, 0.0]},
        {"z_rh_reff": 10.0},
    ],
)
def test_cdf_table_layout_refuses_the_profile_targets(
    kwargs: dict[str, Any],
) -> None:
    """Check that the table layout refuses a profile target."""
    with pytest.raises(ValueError, match="output_sg_ready"):
        read_phase_cdf(WC_CDF, output_sg_ready=False, **kwargs)


def test_cdf_multi_reff_file_needs_a_target() -> None:
    """Check that a file of several radii demands a target."""
    with pytest.raises(ValueError, match="z_rh_reff"):
        read_phase_cdf(WC_CDF)


def test_cdf_resamples_only_the_entries_of_the_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that the entries around the targets alone are resampled."""
    resample = smartg.phase._resample_cdf_phase
    sizes = []

    def spy(ds: xr.Dataset, theta: NDArray[np.float64]) -> Any:
        sizes.append(ds["phase"].shape[:2])
        return resample(ds, theta)

    table = read_phase_cdf(DESERT_CDF, n_theta=721, output_sg_ready=False)
    monkeypatch.setattr(smartg.phase, "_resample_cdf_phase", spy)
    profile = read_phase_cdf(
        DESERT_CDF, n_theta=721, wavelength_phase=[450.0, 520.0]
    )
    # 400 is below the first target, 550 above the second
    assert sizes == [(4, 1)]
    expected = table.interp(wavelength_phase=[450.0, 520.0])
    np.testing.assert_array_equal(profile.values, expected.values)


@pytest.mark.parametrize("wavelength_phase", [350.0, [500.0, 710.0]])
def test_cdf_refuses_a_wavelength_outside_the_file(
    wavelength_phase: float | list[float],
) -> None:
    """Check that a wavelength outside the file raises, not NaN."""
    with pytest.raises(ValueError, match="400 to 700 nm"):
        read_phase_cdf(DESERT_CDF, wavelength_phase=wavelength_phase)


def test_cdf_takes_the_wavelengths_at_the_ends_of_the_file() -> None:
    """Check that the first and last wavelengths of the file work."""
    profile = read_phase_cdf(
        DESERT_CDF, n_theta=721, wavelength_phase=[400.0, 700.0]
    )
    assert not np.isnan(profile.values).any()
    np.testing.assert_allclose(profile.coords["wavelength_phase"],
                               [400.0, 700.0])


def test_cdf_automatic_grid_is_the_finest_step_of_the_file() -> None:
    # 0.01 degree steps in the water cloud peak give the 18001 cap;
    # the float32 step of waso, 0.19999695, gives 902 rather than 901
    """Check the automatic grid against two files of its own."""
    assert read_phase_cdf(WC_CDF, output_sg_ready=False).shape[-1] == 18001
    assert read_phase_cdf(WASO_CDF, output_sg_ready=False).shape[-1] == 902


def test_cdf_n_theta_count_or_angles() -> None:
    """Check that n_theta takes a count or the angles themselves."""
    count = read_phase_cdf(WASO_CDF, n_theta=901, output_sg_ready=False)
    np.testing.assert_allclose(
        count.coords["theta_atm"], np.linspace(0.0, 180.0, 901)
    )
    grid = theta_grid(901, "peak")
    peak = read_phase_cdf(WASO_CDF, n_theta=grid, output_sg_ready=False)
    assert np.array_equal(peak.coords["theta_atm"].values, grid)
    assert peak.shape[-1] == 901


@pytest.mark.parametrize(
    "fname, n_union", [(WC_CDF, 2818), (WASO_CDF, 38)]
)
def test_cdf_native_grid_reads_the_file_back_exactly(
    fname: Path,
    n_union: int,
) -> None:
    """Check that the native grid gives the file values back."""
    ds, grids = _file_grids(fname)
    table = read_phase_cdf(
        fname, n_theta="native", output_sg_ready=False, normalize=False
    )
    theta = table.coords["theta_atm"].values
    assert len(theta) == n_union
    assert np.array_equal(theta, union_theta_grid([g for _, g in grids]))

    # every entry of the file, read back at the nodes of its own grid
    ntheta = ds["ntheta"].values
    for idx, grid in grids:
        n = int(ntheta[idx])
        order = np.argsort(ds["theta"].values[idx][:n])
        file_values = ds["phase"].values[idx][:n][order]
        got = np.interp(grid, theta, table.values[idx])
        np.testing.assert_allclose(got, file_values, rtol=1e-6, atol=1e-12)


def test_cdf_normalize() -> None:
    """Check that normalising brings the integral of f11 to 2."""
    raw = read_phase_cdf(WC_CDF, output_sg_ready=False, normalize=False)
    normed = read_phase_cdf(WC_CDF, output_sg_ready=False, normalize=True)
    theta = normed.coords["theta_atm"].values
    # the reader normalises with the trapezoid rule in cos(theta):
    # exact under that rule, and within 1e-4 of integ_phase, which
    # integrates in theta
    mu = np.cos(np.deg2rad(theta))
    order = np.argsort(mu)
    for ireff in range(normed.sizes["reff"]):
        f11 = normed.values[0, ireff, 0]
        assert abs(np.trapezoid(f11[order], mu[order]) - 2.0) < 1e-12
        assert abs(integ_phase(np.deg2rad(theta), f11) - 2.0) < 1e-4
    # the file values are kept when not normalising: the file grid is
    # stored descending, its first value is the one at 180 degrees
    ds = xr.open_dataset(WC_CDF)
    assert np.isclose(raw.values[0, 0, 0, -1], ds["phase"].values[0, 0, 0, 0])


def test_cdf_keeps_the_terms_of_the_file() -> None:
    """Check that the four terms of the file are all kept."""
    for output_sg_ready in (True, False):
        table = read_phase_cdf(
            WASO_CDF, output_sg_ready=output_sg_ready
        )
        assert table.sizes["nphamat"] == 4


def test_cloud3d_completes_a_four_term_table() -> None:
    """Check that Cloud3D fills a four term table up to six."""
    table = read_phase_cdf(
        WC_CDF, n_theta=901, output_sg_ready=False, normalize=False
    )
    cloud = Cloud3D(
        "wc", w_ref=670.0, ext_ref=np.array([10.0]),
        cell_indices=np.array([[1, 1, 1]]), reff=np.array([10.0]),
        phase=table,
    )
    assert cloud.phase is not None
    assert cloud.phase.sizes["nphamat"] == 6
    full = cloud.get_phase(901)
    assert full.sizes["nphamat"] == 6
    np.testing.assert_allclose(full.values[:, :, 4], full.values[:, :, 0])
    np.testing.assert_allclose(full.values[:, :, 2], full.values[:, :, 5])
    phases, idx, n_unique = cloud.get_phase_set(np.array([670.0]), 901)
    assert n_unique == 1
    np.testing.assert_allclose(
        phases[idx[0]].values[:4], table.sel(reff=10.0).values[0]
    )


# --------------------------------------------------------------------
# read_phase_nc
# --------------------------------------------------------------------


def test_nc_table_layout() -> None:
    """Check the axes and the shape of a netCDF table."""
    table = read_phase_nc(DESERT_NC, output_sg_ready=False)
    assert table.dims == ("wavelength_phase", "hum", "nphamat", "theta_atm")
    assert table.shape == (26, 8, 6, 1801)
    cloud = read_phase_nc(WC_NC, output_sg_ready=False)
    assert cloud.dims == ("wavelength_phase", "reff", "nphamat", "theta_atm")
    assert cloud.sizes["nphamat"] == 4
    assert cloud.sizes["theta_atm"] == 594


def test_nc_profile_layout_is_the_table_interpolated() -> None:
    """Check that the netCDF profile is the table at the targets."""
    table = read_phase_nc(DESERT_NC, output_sg_ready=False)
    profile = read_phase_nc(DESERT_NC, wavelength_phase=550.0, z_rh_reff=70.0)
    assert profile.dims == PROFILE_DIMS
    assert profile.shape == (1, 1, 6, 1801)
    expected = table.interp(wavelength_phase=[550.0], hum=[70.0])
    np.testing.assert_allclose(profile.values, expected.values)


def test_nc_refuses_a_wavelength_outside_the_file() -> None:
    """Check that a netCDF wavelength outside the file raises."""
    with pytest.raises(ValueError, match="outside the wavelength range"):
        read_phase_nc(DESERT_NC, wavelength_phase=5000.0, z_rh_reff=50.0)


def test_nc_table_layout_refuses_the_profile_targets() -> None:
    """Check that the netCDF table layout refuses a target."""
    with pytest.raises(ValueError, match="output_sg_ready"):
        read_phase_nc(DESERT_NC, output_sg_ready=False, wavelength_phase=550.0)


def test_aer3d_takes_the_nc_table() -> None:
    """Check that Aer3D keeps the axes of the netCDF table."""
    table = read_phase_nc(DESERT_NC, output_sg_ready=False)
    aer = Aer3D(
        "desert", w_ref=550.0, ext_ref=np.array([0.1]),
        rh=np.array([70.0]), cell_indices=np.array([[1, 1, 1]]),
        phase=table,
    )
    assert aer.phase is not None
    assert aer.phase.dims == table.dims


# --------------------------------------------------------------------
# read_phase
# --------------------------------------------------------------------


def test_dispatcher_forwards_the_layout() -> None:
    """Check that read_phase forwards the layout to each reader."""
    cdf = read_phase(WASO_CDF, output_sg_ready=False)
    assert cdf.dims[1] == "reff"
    nc = read_phase(WC_NC, output_sg_ready=False)
    assert nc.dims[1] == "reff"
    profile = read_phase(WC_NC, wavelength_phase=550.0, z_rh_reff=10.0)
    assert profile.dims == PROFILE_DIMS


def test_dispatcher_refuses_the_table_layout_of_a_dat_file(
    tmp_path: Path,
) -> None:
    """Check that a .dat file has no table layout to give."""
    theta = np.linspace(0.0, 180.0, 19)
    table = np.column_stack([theta, np.ones(19), np.zeros(19)])
    fname = tmp_path / "phase.dat"
    np.savetxt(fname, table)
    assert read_phase(fname, kind="oc").dims[-1] == "theta_oc"
    with pytest.raises(ValueError, match="dat"):
        read_phase(fname, output_sg_ready=False)


def test_the_constant_theta_reader_is_gone() -> None:
    """Check that read_phase_nth_cte is no longer exported."""
    assert not hasattr(smartg.phase, "read_phase_nth_cte")
