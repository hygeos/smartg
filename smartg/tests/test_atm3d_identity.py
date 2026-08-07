"""Bit-identity of the new Atm3D API against the legacy 3D flow.

Each test builds the profile dataset of an IPRT phase B atmosphere
(C2 cubic cloud, C3 cumulus) through both the legacy construction
(libATM3D.Atm3D getters re-injected into Atm1D("ATM3D", ...)) and the
new Atm3D user API, and asserts that the two datasets are strictly
identical (values, dtypes, coordinates and attributes). No GPU is
needed.

This test is transitional: it guards the migration of the IPRT tests
to the new API and is removed together with the legacy path.

The number of scattering angles is reduced compared to the IPRT tests:
both flows share it, so the comparison holds for any value, and the
C3 per-cell phase mixing is expensive at high angular resolution.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from smartg.atmosphere import (
    AerOPAC,
    Atm1D,
    Atm3D,
    Cloud3D,
    read_i3rc_cloud,
)
from smartg.config import DIR_AUXDATA
from smartg.diff import diff1
from smartg.grid3d import Grid3D
from smartg.libATM3D import Atm3D as LegacyAtm3D
from smartg.libATM3D import Cloud3D as LegacyCloud3D
from smartg.phase import read_cld_nth_cte
from smartg.truncation import GT_trunc

NTH = 361
SCALE = 1
TAU_RAYLEIGH = 0.5
W_REF_C3 = 670.0
SSA_AER_1D = 0.931184

GT_TRUNC_C2 = GT_trunc(
    trunc_frac=0.435,
    theta_tol=20,
    theta_tr=None,
    integral_method="lobatto",
    lobatto_optimization=True,
)


def _assert_identical(old: xr.Dataset, new: xr.Dataset):
    assert set(old.data_vars) == set(new.data_vars)
    assert set(old.coords) == set(new.coords)
    for name in list(old.data_vars) + list(old.coords):
        assert old[name].dtype == new[name].dtype, name
        assert np.array_equal(
            old[name].to_numpy(), new[name].to_numpy(), equal_nan=True
        ), name
    xr.testing.assert_identical(old, new)


# ========================= IPRT phase B C2 ============================


def _c2_grid():
    xgrid = np.array([0.0, 3.0, 4.0, 7.0]) * SCALE
    ygrid = np.array([0.0, 3.0, 4.0, 7.0]) * SCALE
    zgrid = np.array([0.0, 2.0, 3.0, 5.0]) * SCALE

    return Grid3D(xgrid, ygrid, zgrid, periodic=True)


def _c2_cld_phase():
    return read_cld_nth_cte(
        filename=DIR_AUXDATA
        / "IPRT"
        / "phaseB"
        / "opt_prop"
        / "watercloud_800.mie.cdf",
        nb_theta=NTH,
    )


def _c2_cloud_arrays():
    cloud_indices = np.zeros((1, 3), dtype=np.int32)
    cloud_indices[0, :] = np.array([2, 2, 2])
    cld_ext_coeff = np.zeros(1, dtype=np.float64)
    cld_ext_coeff[0] = 10.0
    reff = np.zeros_like(cld_ext_coeff, dtype=np.float64)
    reff[0] = 10.0

    return cloud_indices, cld_ext_coeff, reff


def _c2_mol_overrides(grid3):
    dz = diff1(grid3.zGRID)
    tau_ray_cs = np.cumsum((dz / grid3.zGRID[-1]) * TAU_RAYLEIGH).reshape(
        1, len(dz)
    )
    sca_ray = abs(diff1(tau_ray_cs, axis=1) / dz)
    sca_ray[np.isnan(sca_ray)] = 0

    return {"mol_sca_1d": sca_ray, "mol_abs_1d": np.zeros_like(sca_ray)}


def _build_c2_legacy(truncation=None, tau_ray=False, **atm3_kwargs):
    cld_phase = _c2_cld_phase()
    grid3 = _c2_grid()
    cloud_indices, cld_ext_coeff, reff = _c2_cloud_arrays()
    cloud3 = LegacyCloud3D(
        "wc",
        w_ref=800.0,
        ext_ref=cld_ext_coeff,
        xyz_grids=[grid3.xGRID, grid3.yGRID, grid3.zGRID],
        cell_indices=cloud_indices,
        reff=reff,
        phase=cld_phase,
    )
    if tau_ray:
        atm3_kwargs.update(_c2_mol_overrides(grid3))

    atm3 = LegacyAtm3D(
        "afglt",
        grid3,
        wls=np.array([800.0]),
        wl_ref=800.0,
        cloud_3d=cloud3,
        **atm3_kwargs,
    )

    grid = atm3.get_grid()
    prof_ray = atm3.get_glob_molecular_sca()
    prof_abs = atm3.get_glob_molecular_abs()
    ext_aer = atm3.get_glob_aer_ext() * (1 / SCALE)
    ssa_aer = np.ones_like(ext_aer)
    prof_phases = atm3.get_glob_aer_phase(wl_phase=[800.0], n_theta=NTH)
    cells = atm3.get_cells_info()

    atm3d = Atm1D(
        "ATM3D",
        grid=grid,
        prof_ray=prof_ray,
        prof_abs=prof_abs,
        prof_aer=(ext_aer, ssa_aer),
        prof_phases=prof_phases,
        cells=cells,
    )

    return atm3d.calc(atm3.wls, n_theta=NTH, truncation=truncation)


def _build_c2_new(truncation=None, tau_ray=False, **atm1d_kwargs):
    cld_phase = _c2_cld_phase()
    grid3 = _c2_grid()
    cloud_indices, cld_ext_coeff, reff = _c2_cloud_arrays()
    cloud3 = Cloud3D(
        "wc",
        w_ref=800.0,
        ext_ref=cld_ext_coeff,
        cell_indices=cloud_indices,
        reff=reff,
        phase=cld_phase,
        ssa_cst=1.0,
    )
    atm3_kwargs = _c2_mol_overrides(grid3) if tau_ray else {}

    atm3 = Atm3D(
        atm_1d=Atm1D("afglt", **atm1d_kwargs),
        grid_3d=grid3,
        comp_3d=[cloud3],
        pfwav=[800.0],
        **atm3_kwargs,
    )

    return atm3.calc(np.array([800.0]), n_theta=NTH, truncation=truncation)


@pytest.mark.parametrize("truncation", [None, GT_TRUNC_C2], ids=["", "gt"])
def test_identity_c2_noatm(truncation):
    old = _build_c2_legacy(
        truncation=truncation, tauR=0.0, NO2=False, O3=0.0, H2O=0.0
    )
    new = _build_c2_new(
        truncation=truncation, tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0
    )
    _assert_identical(old, new)


@pytest.mark.parametrize("truncation", [None, GT_TRUNC_C2], ids=["", "gt"])
def test_identity_c2_atm(truncation):
    old = _build_c2_legacy(truncation=truncation, tau_ray=True)
    new = _build_c2_new(truncation=truncation, tau_ray=True)
    _assert_identical(old, new)


# ========================= IPRT phase B C3 ============================


def _k_from_cumulated_od(od_layers, zgrid_desc):
    dz = diff1(zgrid_desc)
    od_cumulated = np.concatenate(([0.0], np.cumsum(od_layers))).reshape(
        1, len(dz)
    )
    k = abs(diff1(od_cumulated, axis=1) / dz)
    k[np.isnan(k)] = 0

    return k


def _c3_mol(grid3):
    od = pd.read_csv(
        DIR_AUXDATA / "IPRT" / "phaseB" / "opt_prop" / "atmos_tau_cu.dat",
        comment="!",
        header=None,
        usecols=[3, 6, 7],
        sep=r"\s+",
        dtype=float,
    ).values
    zgrid_desc = grid3.zGRID[::-1]

    return (
        _k_from_cumulated_od(od[:, 0], zgrid_desc),
        _k_from_cumulated_od(od[:, 1], zgrid_desc),
        _k_from_cumulated_od(od[:, 2], zgrid_desc),
    )


def _c3_aer(ext_aer):
    phase_waso = read_cld_nth_cte(
        DIR_AUXDATA / "IPRT" / "phaseB" / "opt_prop" / "waso_670.mie.cdf",
        nb_theta=NTH,
    )
    aer = AerOPAC(
        "continental_clean",
        0.5,
        w_ref=550.0,
        phase=phase_waso.sub()[0, 0, :, :],
    )

    return aer, np.full_like(ext_aer, SSA_AER_1D)


def _build_c3_legacy(with_aer):
    dir_phase_b = DIR_AUXDATA / "IPRT" / "phaseB"
    cld_phase = read_cld_nth_cte(
        filename=dir_phase_b / "opt_prop" / "watercloud_670.mie.cdf",
        nb_theta=NTH,
    )
    cloud3 = LegacyCloud3D(
        "wc",
        w_ref=W_REF_C3,
        ext_reff_filename=dir_phase_b / "grids" / "cumulus.dat",
        phase=cld_phase,
        reff_acc=1,
        reff_min=5,
    )
    xgrid, ygrid, zgrid = cloud3.get_xyz_grid(loc_xgrid=0, loc_ygrid=0)
    grid3 = Grid3D(xgrid * SCALE, ygrid * SCALE, zgrid * SCALE, periodic=True)

    mol_abs, mol_sca, ext_aer = _c3_mol(grid3)
    atm3_kwargs = {}
    if with_aer:
        aer, ssa_aer = _c3_aer(ext_aer)
        atm3_kwargs = {
            "comp": [aer],
            "aer_ext_1d": ext_aer,
            "aer_ssa_1d": ssa_aer,
            "nth_aer_1d": NTH,
        }

    atm3 = LegacyAtm3D(
        "afglt",
        grid3,
        wls=np.array([W_REF_C3]),
        wl_ref=W_REF_C3,
        cloud_3d=cloud3,
        mol_sca_1d=mol_sca,
        mol_abs_1d=mol_abs,
        **atm3_kwargs,
    )

    grid = atm3.get_grid()
    prof_ray = atm3.get_glob_molecular_sca() * (1 / SCALE)
    prof_abs = atm3.get_glob_molecular_abs() * (1 / SCALE)
    prof_phases, ext_aer_3d, ssa_aer_3d = atm3.get_glob_aer_phase_ext_ssa(
        wl_phase=[W_REF_C3], n_theta=NTH
    )
    cells = atm3.get_cells_info()

    atm3d = Atm1D(
        "ATM3D",
        grid=grid,
        prof_ray=prof_ray,
        prof_abs=prof_abs,
        prof_aer=(ext_aer_3d * (1 / SCALE), ssa_aer_3d),
        prof_phases=prof_phases,
        cells=cells,
    )

    return atm3d.calc(atm3.wls, n_theta=NTH)


def _build_c3_new(with_aer):
    dir_phase_b = DIR_AUXDATA / "IPRT" / "phaseB"
    cld_phase = read_cld_nth_cte(
        filename=dir_phase_b / "opt_prop" / "watercloud_670.mie.cdf",
        nb_theta=NTH,
    )
    cloud3 = Cloud3D(
        "wc",
        w_ref=W_REF_C3,
        ds=read_i3rc_cloud(
            dir_phase_b / "grids" / "cumulus.dat", loc_xgrid=0, loc_ygrid=0
        ),
        phase=cld_phase,
        reff_acc=1,
        reff_min=5,
    )
    xgrid, ygrid, zgrid = cloud3.get_xyz_grid()
    grid3 = Grid3D(xgrid * SCALE, ygrid * SCALE, zgrid * SCALE, periodic=True)

    mol_abs, mol_sca, ext_aer = _c3_mol(grid3)
    comp = []
    atm3_kwargs = {}
    if with_aer:
        aer, ssa_aer = _c3_aer(ext_aer)
        comp = [aer]
        atm3_kwargs = {"aer_ext_1d": ext_aer, "aer_ssa_1d": ssa_aer}

    atm3 = Atm3D(
        atm_1d=Atm1D("afglt", comp=comp),
        grid_3d=grid3,
        comp_3d=[cloud3],
        pfwav=[W_REF_C3],
        mol_sca_1d=mol_sca,
        mol_abs_1d=mol_abs,
        **atm3_kwargs,
    )

    return atm3.calc(np.array([W_REF_C3]), n_theta=NTH)


@pytest.mark.parametrize("with_aer", [True, False], ids=["aer", "noaer"])
def test_identity_c3(with_aer):
    old = _build_c3_legacy(with_aer)
    new = _build_c3_new(with_aer)
    _assert_identical(old, new)
