"""GPU-free tests of the Atm3D multi-component mixing.

A small 4 x 1 x 3 voxel scene combines a water cloud (4 cells in the
1-3 km layer) and an ice cloud (1 cell in the same layer, shared with
the water cloud), without molecular atmosphere and without 1D
aerosols. The tests verify the per-voxel mixing rules on the profile
dataset returned by Atm3D.calc: the extinctions are summed, the
single scattering albedos are extinction-weighted and the phase
matrices are weighted by the scattering coefficients.
"""

import numpy as np
import pytest

from smartg.atmosphere import Atm1D, Atm3D, Cloud3D
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
    # the ice crystals are non-spherical: the 0.5 * (P11 - P22)
    # component of the mixture is nonzero
    assert np.max(np.abs(pha[1])) > 0.0


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
        # water droplets are spherical: P11 == P22
        assert np.allclose(pha[1], 0.0, atol=1e-12), cell


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
