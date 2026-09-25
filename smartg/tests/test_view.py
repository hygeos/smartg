"""GPU-free tests of the analysis helpers of smartg.view.

The optical efficiencies of nopt_view are checked on synthetic solar
tower power outputs: a run split into two identical wavelengths must
give the efficiencies of the same run at one wavelength. The polar
plots of the IPRT cases are checked on analytic Stokes parameters
with the symmetry of a plane-parallel atmosphere.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from smartg.view import _mirror_azimuths, nopt_view, plot_polar_iquv

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
