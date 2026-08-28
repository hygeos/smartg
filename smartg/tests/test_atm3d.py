"""GPU-free tests of the Atm3D 3D components and their mixing.

A small 4 x 1 x 3 voxel scene combines 3D components in the 1-3 km
layer (iz=1), without molecular atmosphere and without 1D aerosols:
a water cloud over 4 cells with an ice cloud in 1 shared cell, then
aerosols (Aer3D) alone and mixed with the water cloud. The tests
verify the Aer3D input routes and optical properties against the
OPAC auxdata, and the per-voxel mixing rules on the profile dataset
returned by Atm3D.calc: the extinctions are summed, the single
scattering albedos are extinction-weighted and the phase matrices
are weighted by the scattering coefficients.
"""

import numpy as np
import pytest
import xarray as xr

from smartg.atmosphere import (
    Aer3D,
    Atm1D,
    Atm3D,
    Cloud3D,
    read_i3rc_aerosol,
)
from smartg.config import DIR_AUXDATA
from smartg.grid3d import Grid3D

WAV = np.array([550.0])
NTH = 181

# 0-based (ix, iy, iz) voxels of the two clouds: the water cloud
# occupies the four x cells of the 1-3 km layer (iz=1), the ice cloud
# only the second one
WC_CELLS = [(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)]
IC_CELL = (1, 0, 1)


def _build_grid():
    return Grid3D(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0, 3.0, 4.0]),
        periodic=True,
    )


def _build_clouds():
    # 1-based IPRT convention for the cell indices; two distinct
    # effective radii for the water cloud so that its phase matrix
    # set has more than one unique matrix
    cld1 = Cloud3D(
        "wc",
        w_ref=550.0,
        ext_ref=np.array([5.0, 10.0, 15.0, 20.0]),
        reff=np.array([10.0, 10.0, 12.0, 12.0]),
        cell_indices=np.array(
            [[1, 1, 2], [2, 1, 2], [3, 1, 2], [4, 1, 2]]
        ),
    )
    cld2 = Cloud3D(
        "ic_baum_ghm",
        w_ref=550.0,
        ext_ref=np.array([2.0]),
        reff=np.array([30.0]),
        cell_indices=np.array([[2, 1, 2]]),
    )
    return cld1, cld2


def _build_profile(grid3, comp_3d):
    # the IPRT C2 "without atmosphere" configuration: no Rayleigh
    # scattering and no gaseous absorption
    atm_1d = Atm1D("afglt", tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0)
    atm3 = Atm3D(
        atm_1d=atm_1d,
        grid_3d=grid3,
        comp_3d=comp_3d,
        pfwav=[550.0],
    )
    return atm3.calc(WAV, n_theta=NTH)


@pytest.fixture(scope="module")
def scene():
    """The grid, the two clouds and the mixed profile"""
    grid3 = _build_grid()
    cld1, cld2 = _build_clouds()
    pro = _build_profile(grid3, [cld1, cld2])
    return grid3, cld1, cld2, pro


def _voxel_props(pro, grid3, cell):
    """The (ext, ssa, phase matrix) of a voxel of the profile"""
    icell = np.ravel_multi_index(
        cell, (grid3.NX, grid3.NY, grid3.NZ)
    )
    k = int(pro["iopt_atm"].values[icell])
    ext = float(pro["OD_p"].values[0, k])
    ssa = float(pro["ssa_p_atm"].values[0, k])
    ipha = int(pro["iphase_atm"].values[0, k])
    pha = pro["phase_atm"].values[ipha]
    return ext, ssa, pha


def _expected(cld):
    """The per-cell (ext, ssa, phase matrices) of a component"""
    ext = cld.get_ext(WAV)[0]
    ssa = cld.get_ssa(WAV)[0]
    luts, idx, _ = cld.get_phase_set(WAV, n_theta=NTH)
    pha = [luts[idx[j]].data for j in range(len(idx))]
    return ext, ssa, pha


def test_two_components_accepted(scene):
    # the profile combines the NZ + 1 background levels and the four
    # cloudy voxels of the union (the ice cloud cell is shared with
    # the water cloud), each with its own optical properties
    grid3, _, _, pro = scene
    nbz = grid3.NZ + 1
    assert pro["OD_p"].shape[1] == nbz + 4
    cells = sorted(set(WC_CELLS) | {IC_CELL})
    iopts = sorted(
        int(
            pro["iopt_atm"].values[
                np.ravel_multi_index(c, (grid3.NX, grid3.NY, grid3.NZ))
            ]
        )
        for c in cells
    )
    assert iopts == list(range(nbz, nbz + 4))


def test_extinction_summation(scene):
    grid3, cld1, cld2, pro = scene
    e1, _, _ = _expected(cld1)
    e2, _, _ = _expected(cld2)
    for j, cell in enumerate(WC_CELLS):
        ext, _, _ = _voxel_props(pro, grid3, cell)
        expected = e1[j] + (e2[0] if cell == IC_CELL else 0.0)
        assert np.isclose(ext, expected, rtol=1e-12), cell


def test_ssa_weighting(scene):
    grid3, cld1, cld2, pro = scene
    e1, s1, _ = _expected(cld1)
    e2, s2, _ = _expected(cld2)
    for j, cell in enumerate(WC_CELLS):
        _, ssa, _ = _voxel_props(pro, grid3, cell)
        if cell == IC_CELL:
            expected = (e1[j] * s1[j] + e2[0] * s2[0]) / (
                e1[j] + e2[0]
            )
        else:
            expected = s1[j]
        assert np.isclose(ssa, expected, rtol=1e-12), cell


def test_phase_mixing_shared_voxel(scene):
    grid3, cld1, cld2, pro = scene
    e1, s1, p1 = _expected(cld1)
    e2, s2, p2 = _expected(cld2)
    _, _, pha = _voxel_props(pro, grid3, IC_CELL)
    j = WC_CELLS.index(IC_CELL)
    expected = (e1[j] * s1[j] * p1[j] + e2[0] * s2[0] * p2[0]) / (
        e1[j] + e2[0]
    )
    assert np.allclose(pha, expected, rtol=1e-5, atol=1e-9)
    # the ice crystals are non-spherical: F22 differs from F11 in the
    # mixture
    assert np.max(np.abs(pha[4] - pha[0])) > 0.0


def test_phase_single_component_voxels(scene):
    # in the water-only voxels the stored matrix is the water cloud
    # one, scaled by its single scattering albedo (the mixed matrices
    # are normalized by the total extinction, not by the total
    # scattering; the scale is harmless as the phase matrices are
    # normalized when sampling the scattering angle)
    grid3, cld1, _, pro = scene
    _, s1, p1 = _expected(cld1)
    for j, cell in enumerate(WC_CELLS):
        if cell == IC_CELL:
            continue
        _, _, pha = _voxel_props(pro, grid3, cell)
        assert np.allclose(
            pha, s1[j] * p1[j], rtol=1e-5, atol=1e-9
        ), cell
        # water droplets are spherical: F22 == F11
        assert np.allclose(pha[4], pha[0], rtol=1e-12), cell


def test_phase_kept_in_iquv_convention(scene):
    # the profile stores the phase matrices in the IQUV convention of
    # the source files, the conversion into the parallel/perpendicular
    # convention of the kernels being done by the run method: F12 of a
    # spherical water cloud is nonzero, whereas its parallel/
    # perpendicular counterpart 0.5 * (F11 - F22) would be zero
    grid3, _, _, pro = scene
    cell = next(c for c in WC_CELLS if c != IC_CELL)
    _, _, pha = _voxel_props(pro, grid3, cell)
    assert np.max(np.abs(pha[1])) > 0.0


def test_empty_voxels_no_atmosphere(scene):
    grid3, _, _, pro = scene
    # the empty voxels fall back to the 1D background levels, without
    # particles nor molecular contribution here
    for cell in [(0, 0, 0), (0, 0, 2), (3, 0, 2)]:
        icell = np.ravel_multi_index(
            cell, (grid3.NX, grid3.NY, grid3.NZ)
        )
        k = int(pro["iopt_atm"].values[icell])
        assert k == grid3.NZ - grid3.idz[icell]
        assert np.isclose(pro["OD_p"].values[0, k], 0.0)
    assert np.allclose(pro["OD_r"].values, 0.0)
    assert np.allclose(pro["OD_g"].values, 0.0, atol=1e-10)


def test_component_order_invariance(scene):
    # the per-voxel properties do not depend on the component order
    grid3, cld1, cld2, pro = scene
    pro_swap = _build_profile(grid3, [cld2, cld1])
    for cell in sorted(set(WC_CELLS) | {IC_CELL}):
        ext_a, ssa_a, pha_a = _voxel_props(pro, grid3, cell)
        ext_b, ssa_b, pha_b = _voxel_props(pro_swap, grid3, cell)
        assert np.isclose(ext_a, ext_b, rtol=1e-12), cell
        assert np.isclose(ssa_a, ssa_b, rtol=1e-12), cell
        assert np.allclose(pha_a, pha_b, rtol=1e-9, atol=1e-12), cell


# ===================================================================
# Aer3D
# ===================================================================

# 1-based IPRT indices of the aerosol cells (the same four iz=1 cells
# as the water cloud) and their extinctions and relative humidities;
# the rh values sit on the OPAC humidity grid and two of them are
# shared, so the phase matrix set has 3 unique matrices
AER_CELL_INDICES = np.array(
    [[1, 1, 2], [2, 1, 2], [3, 1, 2], [4, 1, 2]]
)
AER_EXT = np.array([0.1, 0.2, 0.3, 0.4])
AER_RH = np.array([50.0, 70.0, 80.0, 80.0])


def _build_aerosol(name="desert", **kwargs):
    kwargs.setdefault("w_ref", 550.0)
    kwargs.setdefault("ext_ref", AER_EXT)
    kwargs.setdefault("rh", AER_RH)
    kwargs.setdefault("cell_indices", AER_CELL_INDICES)
    return Aer3D(name, **kwargs)


@pytest.fixture(scope="module")
def aer_scene():
    """The grid, the water cloud, the desert aerosol and the mixed
    profile (the aerosol shares all four water cloud voxels)
    """
    grid3 = _build_grid()
    cld1, _ = _build_clouds()
    aer = _build_aerosol()
    pro = _build_profile(grid3, [cld1, aer])
    return grid3, cld1, aer, pro


@pytest.fixture(scope="module")
def desert_ds():
    """The desert OPAC bulk optical properties, for expected values"""
    fname = (
        DIR_AUXDATA / "aerosols" / "OPAC" / "mixtures" / "desert_sol.nc"
    )
    return xr.open_dataset(fname)


def _dense_aerosol_dataset():
    """The dense-schema dataset equivalent to the raw-route inputs"""
    ext = np.zeros((3, 1, 4), dtype=np.float64)
    rh = np.zeros_like(ext)
    ext[1, 0, :] = AER_EXT
    rh[1, 0, :] = AER_RH
    return xr.Dataset(
        {
            "ext": (("z", "y", "x"), ext),
            "rh": (("z", "y", "x"), rh),
        },
        coords={
            "x_bounds": ("x_b", np.array([0.0, 1.0, 2.0, 3.0, 4.0])),
            "y_bounds": ("y_b", np.array([0.0, 1.0])),
            "z_bounds": ("z_b", np.array([0.0, 1.0, 3.0, 4.0])),
        },
        attrs={"w_ref": 550.0},
    )


def test_aer3d_raw_route():
    aer = _build_aerosol()
    assert np.array_equal(
        aer.get_cell_indices(), AER_CELL_INDICES - 1
    )
    assert np.array_equal(aer.get_ext_ref(), AER_EXT)
    assert np.array_equal(aer.rh, AER_RH)
    assert aer.w_ref == 550.0


def test_aer3d_dense_dataset_route():
    aer_raw = _build_aerosol()
    aer_ds = Aer3D("desert", ds=_dense_aerosol_dataset())
    assert np.array_equal(
        aer_ds.get_cell_indices(), aer_raw.get_cell_indices()
    )
    assert np.array_equal(aer_ds.get_ext_ref(), AER_EXT)
    assert np.array_equal(aer_ds.rh, AER_RH)
    assert aer_ds.w_ref == 550.0
    xgrid, ygrid, zgrid = aer_ds.get_xyz_grid()
    assert np.array_equal(xgrid, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(ygrid, [0.0, 1.0])
    assert np.array_equal(zgrid, [0.0, 1.0, 3.0, 4.0])
    assert np.array_equal(aer_ds.get_ext(WAV), aer_raw.get_ext(WAV))
    assert np.array_equal(aer_ds.get_ssa(WAV), aer_raw.get_ssa(WAV))


def test_aer3d_dataset_missing_var():
    ds = _dense_aerosol_dataset().drop_vars("rh")
    with pytest.raises(ValueError, match="'rh'"):
        Aer3D("desert", ds=ds)


def test_aer3d_raw_route_missing_args():
    with pytest.raises(ValueError, match="rh, ext_ref and"):
        Aer3D(
            "desert",
            w_ref=550.0,
            ext_ref=AER_EXT,
            cell_indices=AER_CELL_INDICES,
        )


def test_aer3d_unknown_species():
    with pytest.raises(FileNotFoundError):
        _build_aerosol("no_such_mixture")


def test_read_i3rc_aerosol(tmp_path):
    # synthetic I3RC/IPRT-style ASCII field matching the raw route
    lines = ["# synthetic 3D aerosol field\n", "4 1 3 2\n"]
    lines.append("1.0 1.0 0.0 1.0 3.0 4.0\n")
    for (ix, iy, iz), ext, rh in zip(
        AER_CELL_INDICES, AER_EXT, AER_RH, strict=True
    ):
        lines.append(f"{ix} {iy} {iz} {ext} {rh}\n")
    fname = tmp_path / "aerosol_field.dat"
    fname.write_text("".join(lines))

    ds = read_i3rc_aerosol(fname, loc_xgrid=0.0, loc_ygrid=0.0)
    assert np.array_equal(ds["ext"].values[1, 0, :], AER_EXT)
    assert np.array_equal(ds["rh"].values[1, 0, :], AER_RH)
    assert np.array_equal(ds["z_bounds"].values, [0.0, 1.0, 3.0, 4.0])

    aer_raw = _build_aerosol()
    aer_dat = Aer3D("desert", ds=ds, w_ref=550.0)
    assert np.array_equal(
        aer_dat.get_cell_indices(), aer_raw.get_cell_indices()
    )
    assert np.array_equal(aer_dat.get_ext(WAV), aer_raw.get_ext(WAV))


def test_aer3d_ext_spectral_scaling(desert_ds):
    # on-grid rh and wavelengths, so the expected values are direct
    # file lookups: ext(wav) = ext_ref * k(rh, wav) / k(rh, w_ref)
    wav_axis = desert_ds["wav"].values
    iw_ref = int(np.abs(wav_axis - 550.0).argmin())
    w_ref = float(wav_axis[iw_ref])
    wav = np.array([w_ref, float(wav_axis[iw_ref + 2])])
    aer = _build_aerosol(w_ref=w_ref)
    ext = aer.get_ext(wav)
    for j, rh in enumerate(AER_RH):
        k = desert_ds["ext"].sel(hum=rh)
        for iw, w in enumerate(wav):
            expected = AER_EXT[j] * float(
                k.sel(wav=w) / k.sel(wav=w_ref)
            )
            assert np.isclose(ext[iw, j], expected, rtol=1e-9), (iw, j)
    assert np.allclose(ext[0, :], AER_EXT, rtol=1e-9)


def test_aer3d_ssa_values(desert_ds):
    wav_axis = desert_ds["wav"].values
    w = float(wav_axis[np.abs(wav_axis - 550.0).argmin()])
    aer = _build_aerosol(w_ref=w)
    ssa = aer.get_ssa(np.array([w]))
    for j, rh in enumerate(AER_RH):
        expected = float(desert_ds["ssa"].sel(hum=rh, wav=w))
        assert np.isclose(ssa[0, j], expected, rtol=1e-9), j
    aer_cst = _build_aerosol(w_ref=w, ssa_cst=0.9)
    assert np.all(aer_cst.get_ssa(np.array([w])) == 0.9)


def test_aer3d_rh_clamping():
    # rh values outside the OPAC humidity axis (0-99 %) are clamped
    # to its extrema, as in the 1D AerOPAC
    aer_hi = _build_aerosol(
        ext_ref=np.array([0.1, 0.1]),
        rh=np.array([100.0, 99.0]),
        cell_indices=AER_CELL_INDICES[:2],
    )
    assert np.array_equal(
        aer_hi.get_ext(WAV)[:, 0], aer_hi.get_ext(WAV)[:, 1]
    )
    assert np.array_equal(
        aer_hi.get_ssa(WAV)[:, 0], aer_hi.get_ssa(WAV)[:, 1]
    )
    luts, idx, n_unique = aer_hi.get_phase_set(WAV, n_theta=NTH)
    assert n_unique == 2
    assert np.array_equal(luts[idx[0]].data, luts[idx[1]].data)

    aer_lo = _build_aerosol(
        ext_ref=np.array([0.1, 0.1]),
        rh=np.array([-10.0, 0.0]),
        cell_indices=AER_CELL_INDICES[:2],
    )
    assert np.array_equal(
        aer_lo.get_ssa(WAV)[:, 0], aer_lo.get_ssa(WAV)[:, 1]
    )


def test_aer3d_rh_acc_min_max():
    aer = _build_aerosol(
        ext_ref=AER_EXT[:2],
        rh=np.array([79.996, 80.004]),
        cell_indices=AER_CELL_INDICES[:2],
        rh_acc=1,
    )
    assert np.array_equal(aer.rh, [80.0, 80.0])
    *_, n_unique = aer.get_phase_set(WAV, n_theta=NTH)
    assert n_unique == 1

    aer = _build_aerosol(
        ext_ref=AER_EXT[:2],
        rh=np.array([30.0, 99.0]),
        cell_indices=AER_CELL_INDICES[:2],
        rh_min=50.0,
        rh_max=95.0,
    )
    assert np.array_equal(aer.rh, [50.0, 95.0])


def test_aer3d_hydrophobic_species():
    # 'inso' has a single humidity node: rh is ignored (clamped to it)
    aer = _build_aerosol(
        "inso",
        ext_ref=AER_EXT[:2],
        rh=np.array([30.0, 70.0]),
        cell_indices=AER_CELL_INDICES[:2],
    )
    hum0 = float(aer.ds_mix["hum"].values[0])
    assert np.array_equal(aer.rh, [hum0, hum0])
    ssa = aer.get_ssa(WAV)
    assert ssa[0, 0] == ssa[0, 1]
    *_, n_unique = aer.get_phase_set(WAV, n_theta=NTH)
    assert n_unique == 1


def test_aer3d_phase_stk_signature():
    # desert aerosols are non-spherical (6-term phase matrices): F22
    # differs from F11; continental_clean is spherical (4-term): its
    # F22 (row 4) is a copy of F11, populated by the 4 -> 6 expansion
    one_cell = dict(
        ext_ref=AER_EXT[:1],
        rh=AER_RH[:1],
        cell_indices=AER_CELL_INDICES[:1],
    )
    luts, idx, _ = _build_aerosol("desert", **one_cell).get_phase_set(
        WAV, n_theta=NTH
    )
    pha = luts[idx[0]].data
    assert np.max(np.abs(pha[4] - pha[0])) > 0.0

    luts, idx, _ = _build_aerosol(
        "continental_clean", **one_cell
    ).get_phase_set(WAV, n_theta=NTH)
    pha = luts[idx[0]].data
    assert np.allclose(pha[4], pha[0], rtol=1e-12)
    assert np.max(np.abs(pha[4])) > 0.0


def test_cloud_aerosol_mixing(aer_scene):
    # water cloud and desert aerosol in the same voxels: extinctions
    # summed, ssa extinction-weighted, phase matrices weighted by the
    # scattering coefficients
    grid3, cld1, aer, pro = aer_scene
    e_c, s_c, p_c = _expected(cld1)
    e_a, s_a, p_a = _expected(aer)
    for j, cell in enumerate(WC_CELLS):
        ext, ssa, pha = _voxel_props(pro, grid3, cell)
        ext_tot = e_c[j] + e_a[j]
        assert np.isclose(ext, ext_tot, rtol=1e-12), cell
        assert np.isclose(
            ssa,
            (e_c[j] * s_c[j] + e_a[j] * s_a[j]) / ext_tot,
            rtol=1e-12,
        ), cell
        expected = (
            e_c[j] * s_c[j] * p_c[j] + e_a[j] * s_a[j] * p_a[j]
        ) / ext_tot
        assert np.allclose(pha, expected, rtol=1e-5, atol=1e-9), cell
        # the non-spherical desert makes F22 differ from F11 even
        # though the water droplets are spherical
        assert np.max(np.abs(pha[4] - pha[0])) > 0.0, cell


def test_two_aerosols_mixing():
    # desert and continental_clean sharing one voxel
    grid3 = _build_grid()
    desert = Aer3D(
        "desert",
        w_ref=550.0,
        ext_ref=np.array([0.1, 0.2]),
        rh=np.array([70.0, 80.0]),
        cell_indices=np.array([[1, 1, 2], [2, 1, 2]]),
    )
    conti = Aer3D(
        "continental_clean",
        w_ref=550.0,
        ext_ref=np.array([0.3, 0.4]),
        rh=np.array([80.0, 90.0]),
        cell_indices=np.array([[2, 1, 2], [3, 1, 2]]),
    )
    pro = _build_profile(grid3, [desert, conti])
    e_d, s_d, p_d = _expected(desert)
    e_c, s_c, p_c = _expected(conti)

    # shared voxel (0-based (1, 0, 1))
    ext, ssa, pha = _voxel_props(pro, grid3, (1, 0, 1))
    ext_tot = e_d[1] + e_c[0]
    assert np.isclose(ext, ext_tot, rtol=1e-12)
    assert np.isclose(
        ssa, (e_d[1] * s_d[1] + e_c[0] * s_c[0]) / ext_tot, rtol=1e-12
    )
    expected = (
        e_d[1] * s_d[1] * p_d[1] + e_c[0] * s_c[0] * p_c[0]
    ) / ext_tot
    assert np.allclose(pha, expected, rtol=1e-5, atol=1e-9)

    # desert-only voxel: non-spherical signature
    _, _, pha = _voxel_props(pro, grid3, (0, 0, 1))
    assert np.max(np.abs(pha[4] - pha[0])) > 0.0
    # continental-only voxel: spherical, F22 == F11
    _, _, pha = _voxel_props(pro, grid3, (2, 0, 1))
    assert np.allclose(pha[4], pha[0], rtol=1e-12)
