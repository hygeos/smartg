"""GPU-free tests of the analysis helpers of smartg.view.

The optical efficiencies of nopt_view are checked on synthetic solar
tower power outputs: a run split into two identical wavelengths must
give the efficiencies of the same run at one wavelength. The polar
plots of the IPRT cases are checked on analytic Stokes parameters
with the symmetry of a plane-parallel atmosphere, and the channel sums
of cat_view on synthetic REPTRAN internal bands.
"""

import importlib
from types import SimpleNamespace
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from smartg.reptran import ReptranIbandList
from smartg.view import (
    _mirror_azimuths,
    camera_view,
    cat_view,
    compare,
    nopt_view,
    plot_polar_iquv,
    profile_view,
)

N_PHOTONS = 1000
# the loss weights W_I, W_rhoM, W_rhoP, W_BM, W_BP, W_SM, W_SP of a
# consistent run, their squares, and the receiver weight (category 2)
W_LOSS = np.array([100.0, 12.0, 88.0, 5.0, 83.0, 3.0, 80.0])
W_LOSS2 = 1.5 * W_LOSS**2 / N_PHOTONS
W_REC, W_REC2 = 70.0, 1.5 * 70.0**2 / N_PHOTONS
POWC_H = 150.0


def _stp_dataset(n_band: int) -> xr.Dataset:
    """Return the STP output of a run split into n_band wavelengths.

    Each wavelength receives the same share of the photons and of
    their weights.
    """
    cat_w = np.zeros(9)
    cat_w[2] = W_REC
    cat_w2 = np.zeros(9)
    cat_w2[2] = W_REC2
    band = np.full(n_band, 1.0 / n_band)
    ds = xr.Dataset(
        {
            "cat_w": ("Categories", cat_w),
            "cat_w2": ("Categories", cat_w2),
            "powc_H": ("wavelength", np.full(n_band, POWC_H)),
            "n_aatm": ("wavelength", np.full(n_band, 0.95)),
            "norm_npho": ("wavelength", N_PHOTONS * band),
        },
        attrs={"NPHOTONS": f"{N_PHOTONS:g}", "n_cte": "1.5", "n_cos": "0.9"},
    )
    if n_band == 1:
        ds["wLoss"] = ("index", W_LOSS)
        ds["wLoss2"] = ("index", W_LOSS2)
        ds["wPhCats"] = ("Categories", cat_w)
        ds["wPhCats2"] = ("Categories", cat_w2)
    else:
        dims = ("index", "wavelength")
        ds["wLoss"] = (dims, np.outer(W_LOSS, band))
        ds["wLoss2"] = (dims, np.outer(W_LOSS2, band))
        dims = ("Categories", "wavelength")
        ds["wPhCats"] = (dims, np.outer(cat_w, band))
        ds["wPhCats2"] = (dims, np.outer(cat_w2, band))
    return ds


@pytest.mark.parametrize("back", [False, True], ids=["forward", "backward"])
def test_nopt_view_bands_sum_to_one_band(
    back: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two halves of a run give the efficiencies of the whole run."""
    nopt_view(_stp_dataset(1), back=back, natm_approx=True)
    one = capsys.readouterr().out
    nopt_view(_stp_dataset(2), back=back, natm_approx=True)
    two = capsys.readouterr().out
    assert "nopt =" in one
    assert two == one
    nopt_view(_stp_dataset(2), back=back, natm_approx=True,
              mtoa=np.array([2.0, 2.0]))
    assert capsys.readouterr().out == one


def test_nopt_view_mtoa_size() -> None:
    """Refuse an mtoa without one value per wavelength."""
    with pytest.raises(ValueError, match="mtoa has 3 values"):
        nopt_view(_stp_dataset(2), mtoa=np.array([1.0, 2.0, 3.0]))


def _analytic_iquv(
    thetas: np.ndarray, phis: np.ndarray
) -> list[np.ndarray]:
    """Return Stokes parameters symmetric about the principal plane.

    I and Q are even functions of the azimuth angle, U and V odd.
    """
    theta, phi = np.meshgrid(thetas, np.radians(phis), indexing="ij")
    return [
        (1.0 + np.cos(np.radians(theta))) * (2.0 + np.cos(phi)),
        np.cos(2.0 * phi),
        np.sin(phi),
        np.sin(2.0 * phi),
    ]


def test_mirror_azimuths_signs() -> None:
    """The mirrored half keeps I and Q and flips the sign of U and V."""
    thetas = np.array([0.0, 30.0, 60.0])
    phis = np.arange(0.0, 181.0, 30.0)
    full_phis, mirrored = _mirror_azimuths(
        _analytic_iquv(thetas, phis), phis
    )
    np.testing.assert_allclose(full_phis, np.concatenate((phis, phis + 180)))
    expected = _analytic_iquv(thetas, full_phis)
    for values, ref in zip(mirrored, expected, strict=True):
        np.testing.assert_allclose(values, ref, atol=1e-12)


def test_mirror_azimuths_needs_symmetric_phis() -> None:
    """Refuse azimuth angles that do not mirror onto phis + 180."""
    phis = np.array([0.0, 45.0, 90.0, 180.0])
    with pytest.raises(ValueError, match="symmetric about 90"):
        _mirror_azimuths(_analytic_iquv(np.array([0.0, 10.0]), phis), phis)


def test_plot_polar_iquv_sym() -> None:
    """Draw the four panels over the full azimuth circle."""
    thetas = np.array([0.0, 30.0, 60.0])
    phis = np.arange(0.0, 181.0, 30.0)
    plot_polar_iquv(_analytic_iquv(thetas, phis), thetas, phis, sym=True)
    assert len(plt.gcf().axes) >= 4
    plt.close("all")


def _channel_ibands() -> ReptranIbandList:
    """Return two REPTRAN channels sharing an internal wavelength.

    Channel A (centre 450 nm, 100 nm wide) has internal bands at 400
    and 500 nm, channel B (centre 600 nm, 200 nm wide) at 500 and
    700 nm.
    """
    channel_a = SimpleNamespace(
        awvl=np.array([400.0, 500.0]),
        awvl_weight=np.array([0.6, 0.4]),
        aextra=np.ones(2),
        dl=100.0,
    )
    channel_b = SimpleNamespace(
        awvl=np.array([500.0, 700.0]),
        awvl_weight=np.array([0.3, 0.7]),
        aextra=np.ones(2),
        dl=200.0,
    )
    return ReptranIbandList(
        [
            cast(Any, SimpleNamespace(band=band, index=index))
            for band, index in [
                (channel_a, 0), (channel_a, 1), (channel_b, 0), (channel_b, 1)
            ]
        ]
    )


def test_cat_view_sums_the_internal_bands_of_each_channel() -> None:
    """The REPTRAN outputs of cat_view have one value per channel."""
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    cat_w = np.outer(np.ones(9), weights)
    ds = xr.Dataset(
        {
            "wPhCats": (("Categories", "wavelength"), cat_w),
            "wPhCats2": (("Categories", "wavelength"), cat_w**2 / 100.0),
            "norm_npho": ("wavelength", np.full(4, 250.0)),
            "cat_PhNb": ("Categories", np.full(9, 100.0)),
        },
        coords={
            "Categories": np.arange(9.0),
            "wavelength": np.array([400.0, 500.0, 500.0, 700.0], np.float32),
        },
        attrs={"ALDEG": "0", "NPHOTONS": "1000", "n_cte": "1.0"},
    )

    out = cat_view(
        ds, mtoa=1.0, output_unit="FLUX", print_results=False,
        kdis_rep_bands=_channel_ibands(),
    )

    # four bands of 250 photons each: cst = 4 in every band
    np.testing.assert_array_equal(out.wavelength, [450.0, 600.0])
    np.testing.assert_allclose(out["FLUX_int"], [[12.0, 28.0]] * 9)
    np.testing.assert_allclose(out["FLUX"], [[0.12, 0.14]] * 9)
    np.testing.assert_allclose(out["FLUX_tot"], np.full(9, 40.0))


def test_view_import_keeps_numpy_error_state() -> None:
    """Importing smartg.view leaves the NumPy error state alone."""
    import smartg.view

    with np.errstate(all="raise"):
        importlib.reload(smartg.view)
        assert set(np.geterr().values()) == {"raise"}


def _camera_matrices() -> list[np.ndarray]:
    """Return I, Q and an all-zero and an all-NaN matrix on 3 x 4."""
    i = np.linspace(1.0, 2.0, 12).reshape(3, 4)
    return [i, -0.1 * i, np.zeros((3, 4)), np.full((3, 4), np.nan)]


def test_camera_view_zero_and_nan_panels() -> None:
    """A panel with no non-zero value gets default colorbar ticks."""
    fig = camera_view(
        None, np.arange(5.0), np.arange(4.0), matrices=_camera_matrices(),
        stokes=["I", "Q", "U", "V"],
    )
    # 4 panels and their colorbars
    assert len(fig.axes) == 8
    plt.close(fig)


def test_camera_view_needs_one_stokes_per_matrix() -> None:
    """Two matrices with the default stokes raise a clear error."""
    with pytest.raises(ValueError, match="one label per matrix"):
        camera_view(
            None, np.arange(5.0), np.arange(4.0),
            matrices=_camera_matrices()[:2],
        )
    plt.close("all")


def test_profile_view_phase_index_of_the_wavelength() -> None:
    """The phase index axis shows the profile of wavelength iw."""
    z = np.array([100.0, 50.0, 20.0, 10.0, 0.0])
    od = np.array([[0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.2, 0.4, 0.6, 0.8]])
    iphase = np.array([[0, 1, 1, 2, 2], [0, 3, 3, 4, 4]])
    dims = ("wavelength", "z_atm")
    ds = xr.Dataset(
        {
            name: (dims, od)
            for name in ("OD_atm", "OD_sca_atm", "OD_r", "OD_p")
        }
        | {
            "OD_abs_atm": (dims, 0.1 * od),
            "OD_g": (dims, 0.1 * od),
            "ssa_p_atm": (dims, np.full(od.shape, 0.9)),
            "iphase_atm": (dims, iphase),
        },
        coords={"wavelength": [500.0, 600.0], "z_atm": z},
    )

    # the extinction per km is infinite at the top level
    with np.errstate(all="raise"):
        fig, ax = profile_view(ds, iw=1)

    ax2 = fig.axes[-1]
    assert ax2 is not ax
    (line,) = ax2.get_lines()
    np.testing.assert_array_equal(line.get_xdata(), iphase[1, 1:])
    np.testing.assert_array_equal(line.get_ydata(), z[1:])
    plt.close(fig)


def test_compare_panel_titles() -> None:
    """The panels are titled after the compared output variables."""
    phi = np.arange(0.0, 360.0, 30.0)
    th = np.linspace(1.0, 89.0, 45)
    dims = ("Azimuth angles", "Zenith angles")
    stokes = np.ones((phi.size, th.size))
    ds = xr.Dataset(
        {
            f"{stk}_up (TOA)": (dims, factor * stokes)
            for stk, factor in zip("IQUV", (1.0, 0.1, 0.05, 0.01))
        },
        coords={"Azimuth angles": phi, "Zenith angles": th},
    )

    fig = compare(ds, ds)

    titles = [ax.get_title() for ax in fig.axes[:4]]
    assert titles == [
        r"$I^{\uparrow}_{TOA}$",
        r"$Q^{\uparrow}_{TOA}$",
        r"$U^{\uparrow}_{TOA}$",
        r"$DoLP^{\uparrow}_{TOA}$",
    ]
    plt.close(fig)
