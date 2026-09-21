"""GPU-free tests of the truncation carried by the components.

A component truncated through its `truncation` parameter is truncated
alone, before being mixed with the other components of its layer, and
only its own scattering is scaled by `1 - f`. The profiles returned by
`Atm1D.calc` are checked against the untruncated ones transformed by
hand, and against the profiles of each component alone. The last test
checks the memoized truncation of a hydrosol of `smartg.water`.
"""

import warnings
from typing import Any

import numpy as np
import pytest
import xarray as xr

from smartg.atmosphere import AerOPAC, Atm1D, Cloud
from smartg.diff import diff1
from smartg.phase import theta_grid
from smartg.truncation import (
    DMTrunc,
    GTTrunc,
    truncate_phase,
    truncated_ext_ssa,
)
from smartg.water import (
    DEFAULT_WATER_TRUNC,
    Hydrosol,
    HydrosolPR,
    Water1D,
)

WAV = np.array([550.0])
GRID = [100.0, 50.0, 20.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
N_THETA = theta_grid(1801)
GT = GTTrunc(trunc_frac=0.3, theta_tr=8.0)
DM = DMTrunc(n_streams=16)
# layer holding the cloud (2-3 km) on GRID, as a diff1 index
ICLD = 7


def _aerosol(**kwargs: Any) -> AerOPAC:
    """Build the OPAC aerosol mixed with the cloud."""
    return AerOPAC("continental_clean", 0.2, 550.0, **kwargs)


def _cloud(**kwargs: Any) -> Cloud:
    """Build the water cloud of the 2-3 km layer."""
    return Cloud("wc", 12.68, 2.0, 3.0, 5.0, 550.0, **kwargs)


def _calc(comps: list[AerOPAC], **kwargs: Any) -> xr.Dataset:
    """Compute the profile of a 1D atmosphere holding `comps`."""
    atm = Atm1D("afglms", comp=comps, grid=GRID, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return atm.calc(WAV, n_theta=N_THETA)


def _layer_sca(pro: xr.Dataset) -> np.ndarray:
    """Return the particle scattering optical depth of each layer."""
    return diff1(pro["OD_p"].values, axis=1) * pro["ssa_p_atm"].values


@pytest.mark.parametrize("truncation", [GT, DM], ids=["GT", "DM"])
def test_single_component(truncation: DMTrunc | GTTrunc) -> None:
    """A truncated cloud alone: the untruncated profile, transformed.

    The phase matrix is the untruncated one truncated, the particle
    extinction and single scattering albedo follow `truncated_ext_ssa`
    with its truncated fraction, and the absorption is unchanged.
    """
    full = _calc([_cloud()])
    trunc = _calc([_cloud(truncation=truncation)])

    theta = full.coords["theta_atm"].values
    pha_expected, f = truncate_phase(
        full["phase_atm"].values[0], theta, truncation
    )
    assert 0.0 < f < 1.0
    np.testing.assert_allclose(
        trunc["phase_atm"].values[0], pha_expected, rtol=1e-5, atol=1e-8
    )

    dtau_expected, ssa_expected = truncated_ext_ssa(
        diff1(full["OD_p"].values, axis=1), full["ssa_p_atm"].values, f
    )
    np.testing.assert_allclose(
        diff1(trunc["OD_p"].values, axis=1), dtau_expected,
        rtol=1e-5, atol=1e-7,
    )
    np.testing.assert_allclose(
        trunc["ssa_p_atm"].values, ssa_expected, rtol=1e-5
    )
    np.testing.assert_allclose(
        trunc["OD_abs_atm"].values, full["OD_abs_atm"].values, rtol=1e-5
    )
    np.testing.assert_allclose(
        trunc["OD_r"].values, full["OD_r"].values
    )
    # less particle scattering, hence more molecular scattering in
    # proportion, in the cloud layer
    assert trunc["pmol_atm"].values[0, ICLD] > full["pmol_atm"][0, ICLD]


def test_mixture_truncates_the_cloud_alone() -> None:
    """In a layer of aerosol and truncated cloud, only the cloud is cut.

    The layer holds the sum of the aerosol alone and of the truncated
    cloud alone. Its phase matrix is a mixture of the aerosol matrix
    and of the truncated cloud matrix, in which the weight of the
    cloud against the aerosol is the untruncated one times `1 - f`.
    """
    aer_alone = _calc([_aerosol()])
    cld_alone = _calc([_cloud(truncation=GT)])
    cld_full = _calc([_cloud()])
    mix = _calc([_aerosol(), _cloud(truncation=GT)])
    mix_full = _calc([_aerosol(), _cloud()])

    np.testing.assert_allclose(
        diff1(mix["OD_p"].values, axis=1),
        diff1(aer_alone["OD_p"].values, axis=1)
        + diff1(cld_alone["OD_p"].values, axis=1),
        rtol=1e-5, atol=1e-7,
    )
    np.testing.assert_allclose(
        _layer_sca(mix), _layer_sca(aer_alone) + _layer_sca(cld_alone),
        rtol=1e-5, atol=1e-7,
    )

    # the mixture is a convex combination of the two matrices; alpha
    # is the aerosol share of the scattering
    def aerosol_share(pro: xr.Dataset, cld: xr.Dataset) -> float:
        p_mix = pro["phase_atm"].values[0, 0]
        p_aer = aer_alone["phase_atm"].values[0, 0]
        p_cld = cld["phase_atm"].values[0, 0]
        alpha = np.dot(p_mix - p_cld, p_aer - p_cld) / np.dot(
            p_aer - p_cld, p_aer - p_cld
        )
        np.testing.assert_allclose(
            p_mix, p_cld + alpha * (p_aer - p_cld), rtol=1e-4, atol=1e-6
        )
        return float(alpha)

    alpha = aerosol_share(mix, cld_alone)
    alpha_full = aerosol_share(mix_full, cld_full)
    _, f = truncate_phase(
        cld_full["phase_atm"].values[0], N_THETA, GT
    )
    odds = alpha / (1.0 - alpha)
    odds_full = alpha_full / (1.0 - alpha_full)
    assert odds / odds_full == pytest.approx(1.0 / (1.0 - f), rel=1e-4)


def test_two_components_truncated_differently() -> None:
    """Each truncated component keeps its own truncated fraction."""
    aer_t = _aerosol(truncation=DM)
    cld_t = _cloud(truncation=GT)
    mix = _calc([aer_t, cld_t])
    np.testing.assert_allclose(
        diff1(mix["OD_p"].values, axis=1),
        diff1(_calc([aer_t])["OD_p"].values, axis=1)
        + diff1(_calc([cld_t])["OD_p"].values, axis=1),
        rtol=1e-5, atol=1e-7,
    )
    assert not np.isnan(mix["phase_atm"].values).any()


def test_straddling_layer_is_truncated_with_its_cloud() -> None:
    """A layer straddling pfgrid layers is truncated with its cloud.

    With the phase matrices tabulated over two layers, 100-2.3 and
    2.3-0 km, the 2-3 km profile layer straddles them. It takes the
    matrix of the lower one, where its thin 2-2.2 km cloud is, so its
    cloud scattering is scaled by the truncated fraction of the cloud
    matrix, as with a single pfgrid layer.
    """
    def thin_cloud(**kwargs: Any) -> Cloud:
        """Build a water cloud between 2 and 2.2 km."""
        return Cloud("wc", 12.68, 2.0, 2.2, 5.0, 550.0, **kwargs)

    two = _calc([_aerosol(), thin_cloud(truncation=GT)],
                pfgrid=[100.0, 2.3, 0.0])
    one = _calc([_aerosol(), thin_cloud(truncation=GT)])
    full = _calc([_aerosol(), thin_cloud()])
    assert two["iphase_atm"].values[0, ICLD] == 1
    np.testing.assert_allclose(
        two["OD_p"].values, one["OD_p"].values, rtol=1e-6
    )
    assert two["OD_p"].values[0, -1] < full["OD_p"].values[0, -1]


def test_calc_split_is_truncated() -> None:
    """calc_split returns the truncated particle profile."""
    atm = Atm1D("afglms", comp=[_aerosol(), _cloud(truncation=GT)],
                grid=GRID)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, (ext, ssa), _ = atm.calc_split(WAV, n_theta=N_THETA)
        pro = atm.calc(WAV, n_theta=N_THETA)
    np.testing.assert_allclose(
        ext, diff1(pro["OD_p"].values, axis=1), rtol=1e-6
    )
    np.testing.assert_allclose(ssa, pro["ssa_p_atm"].values)


def test_no_phase_no_truncation() -> None:
    """Without phase matrices, nothing is truncated."""
    full = Atm1D("afglms", comp=[_cloud()], grid=GRID).calc(
        WAV, phase=False
    )
    trunc = Atm1D("afglms", comp=[_cloud(truncation=GT)], grid=GRID).calc(
        WAV, phase=False
    )
    np.testing.assert_array_equal(trunc["OD_p"].values, full["OD_p"].values)


def test_negative_truncated_phase_is_refused() -> None:
    """A truncation leaving a negative phase function is refused.

    Delta-M with 8 streams rings below zero on the cloud phase
    function, which used to reach the profile unnoticed.
    """
    atm = Atm1D("afglms", comp=[_cloud(truncation=DMTrunc(n_streams=8))],
                grid=GRID)
    with pytest.raises(ValueError, match="negative"):
        atm.calc(WAV, n_theta=N_THETA)


def test_forced_particle_profile_is_refused() -> None:
    """A truncated component cannot ride on a forced prof_aer."""
    nz = len(GRID)
    prof_aer = (np.zeros((1, nz)), np.ones((1, nz)))
    atm = Atm1D("afglms", comp=[_cloud(truncation=GT)], grid=GRID,
                prof_aer=prof_aer)
    with pytest.raises(ValueError, match="prof_aer"):
        atm.calc(WAV, n_theta=N_THETA)


def test_hydrosol_reused_at_another_wavelength() -> None:
    """A reused hydrosol gives what a new one would.

    Its truncated phase matrices and truncation factor are memoized;
    used at other wavelengths (or on another grid), it must tabulate
    them again rather than serve the first ones. The backscattering
    ratio of HydrosolPR varies with the wavelength, so its phase
    matrix at 650 nm differs from the one at 450 nm.
    """
    grid = [0.0, -10.0]

    def hydrosol() -> HydrosolPR:
        """Build the truncated chlorophyll hydrosol."""
        return HydrosolPR(chl=0.5, n_theta=721,
                          truncation=DEFAULT_WATER_TRUNC)

    h = hydrosol()
    Water1D(grid=grid, comp=[h]).calc([450.0])
    again = Water1D(grid=grid, comp=[h]).calc([650.0])
    fresh = Water1D(grid=grid, comp=[hydrosol()]).calc([650.0])
    for var in ["OD_p_oc", "phase_oc", "iphase_oc"]:
        np.testing.assert_array_equal(
            again[var].values, fresh[var].values, err_msg=var
        )


WATER_GRID = [0.0, -10.0]
WATER_WAV = np.array([550.0])


def _ff_phase() -> xr.DataArray:
    """Return the untruncated Fournier-Forand mixture of a hydrosol.

    Derived from a backscattering ratio of 0.01, at 550 nm, on a single
    depth, in the shape the `phase` parameter of Hydrosol accepts.
    """
    h = Hydrosol(bp=0.1, bbp_ratio=0.01, n_theta=7201)
    pha, _ = h.calc_phase(WATER_WAV, np.array([0.0]), np.full((1, 1), 0.01))
    return pha


def test_hydrosol_supplied_phase_truncated_when_asked() -> None:
    """A supplied phase is truncated when asked, as a derived one is.

    Given the untruncated Fournier-Forand mixture a backscattering
    ratio derives, a hydrosol given the same truncation gives the
    profile of the hydrosol that derives it: the same truncated matrix,
    the same scattering scaled by 1 - f.
    """
    trunc = DEFAULT_WATER_TRUNC
    supplied = Water1D(
        grid=WATER_GRID,
        comp=[Hydrosol(phase=_ff_phase(), bp=0.1, truncation=trunc)],
    ).calc(WATER_WAV)
    derived = Water1D(
        grid=WATER_GRID,
        comp=[Hydrosol(bp=0.1, bbp_ratio=0.01, n_theta=7201,
                       truncation=trunc)],
    ).calc(WATER_WAV)
    np.testing.assert_allclose(
        supplied["phase_oc"].values, derived["phase_oc"].values,
        rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        supplied["OD_p_oc"].values, derived["OD_p_oc"].values, rtol=1e-12
    )
    # GT with a fraction of 0.3: 70 % of bp is left
    np.testing.assert_allclose(
        supplied["OD_p_oc"].values[0, -1], -0.7 * 0.1 * 10.0, rtol=1e-12
    )


def test_hydrosol_supplied_phase_untruncated_by_default() -> None:
    """Without a truncation, a supplied phase is kept as it is."""
    pha = _ff_phase()
    pro = Water1D(
        grid=WATER_GRID, comp=[Hydrosol(phase=pha, bp=0.1)]
    ).calc(WATER_WAV)
    np.testing.assert_array_equal(pro["phase_oc"].values[0], pha.values[0, 0])
    np.testing.assert_allclose(
        pro["OD_p_oc"].values[0, -1], -0.1 * 10.0, rtol=1e-12
    )


def test_hydrosol_supplied_flat_phase_refused() -> None:
    """A supplied phase without a forward peak cannot be truncated.

    The truncation leaves it negative, and is refused; untruncated, the
    default, it is accepted.
    """
    theta = theta_grid(721)
    mu = np.cos(np.deg2rad(theta))
    pha = np.zeros((1, 1, 6, len(theta)))
    pha[0, 0, 0] = pha[0, 0, 4] = 0.75 * (1.0 + 0.835 * mu**2)
    phase = xr.DataArray(
        pha,
        dims=["wavelength_phase", "z_phase", "nphamat", "theta_oc"],
        coords={"wavelength_phase": WATER_WAV, "z_phase": [0.0],
                "theta_oc": theta},
    )
    with pytest.raises(ValueError, match="negative"):
        Water1D(
            grid=WATER_GRID,
            comp=[Hydrosol(phase=phase, bp=0.1,
                           truncation=DEFAULT_WATER_TRUNC)],
        ).calc(WATER_WAV)
    Water1D(
        grid=WATER_GRID, comp=[Hydrosol(phase=phase, bp=0.1)]
    ).calc(WATER_WAV)


def test_a_boolean_is_no_truncation() -> None:
    """A component refuses a boolean truncation when it is built.

    The truncation is asked for with a DMTrunc or a GTTrunc; None, the
    default, is no truncation.
    """
    with pytest.raises(TypeError, match="DMTrunc or a GTTrunc"):
        _cloud(truncation=False)
    with pytest.raises(TypeError, match="DMTrunc or a GTTrunc"):
        Hydrosol(bp=0.1, bbp_ratio=0.01, truncation=True)  # type: ignore
