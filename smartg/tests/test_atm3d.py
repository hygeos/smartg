"""GPU-free tests of the Atm3D 3D components and their mixing.

A small 4 x 1 x 3 voxel scene combines 3D components in the 1-3 km
layer (iz=1), without molecular atmosphere and without 1D aerosols:
a water cloud over 4 cells with an ice cloud in 1 shared cell, then
aerosols (Aer3D) alone and mixed with the water cloud. The tests
verify the Aer3D input routes and optical properties against the
OPAC auxdata, and the per-voxel mixing rules on the profile dataset
returned by Atm3D.calc: the extinctions are summed, the single
scattering albedos are extinction-weighted and the phase matrices
are weighted by the scattering coefficients, and normalized by their
sum, so that a mixed matrix keeps the normalization of the component
ones: the local estimate reads it as it is. The last tests truncate
one component, alone, before the mixing.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

import smartg.truncation as trunc_mod
from smartg.atmosphere import (
    Aer3D,
    AerOPAC,
    Atm1D,
    Atm3D,
    Cloud3D,
    read_i3rc_aerosol,
)
from smartg.config import DIR_AUXDATA
from smartg.grid3d import Grid3D, create_1d_grid, extend_1d_grid
from smartg.truncation import (
    DMTrunc,
    GTTrunc,
    truncate_phase,
    truncated_ext_ssa,
)

Scene = tuple[Grid3D, Cloud3D, Cloud3D, xr.Dataset]
AerScene = tuple[Grid3D, Cloud3D, Aer3D, xr.Dataset]
# per cell ext, ssa and phase matrices, from _expected
Expected = tuple[
    NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]
]

WAV = np.array([550.0])
NTH = 181

# 0-based (ix, iy, iz) voxels of the two clouds: the water cloud
# occupies the four x cells of the 1-3 km layer (iz=1), the ice cloud
# only the second one
WC_CELLS = [(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)]
IC_CELL = (1, 0, 1)


def _build_grid() -> Grid3D:
    """Build the 4 x 1 x 3 periodic grid the tests share."""
    return Grid3D(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0, 3.0, 4.0]),
        periodic=True,
    )


def _build_clouds(
    truncation: DMTrunc | GTTrunc | None = None,
) -> tuple[Cloud3D, Cloud3D]:
    # 1-based IPRT convention for the cell indices; two distinct
    # effective radii for the water cloud so that its phase matrix
    # set has more than one unique matrix
    """Build two clouds, one of them with two effective radii.

    The water cloud, the first one, carries `truncation`.
    """
    cld1 = Cloud3D(
        "wc",
        w_ref=550.0,
        ext_ref=np.array([5.0, 10.0, 15.0, 20.0]),
        reff=np.array([10.0, 10.0, 12.0, 12.0]),
        cell_indices=np.array(
            [[1, 1, 2], [2, 1, 2], [3, 1, 2], [4, 1, 2]]
        ),
        truncation=truncation,
    )
    cld2 = Cloud3D(
        "ic_baum_ghm",
        w_ref=550.0,
        ext_ref=np.array([2.0]),
        reff=np.array([30.0]),
        cell_indices=np.array([[2, 1, 2]]),
    )
    return cld1, cld2


def _build_profile(
    grid3: Grid3D,
    comp_3d: list[Cloud3D | Aer3D],
) -> xr.Dataset:
    # the IPRT C2 "without atmosphere" configuration: no Rayleigh
    # scattering and no gaseous absorption
    """Build the 3D profile of the given components."""
    atm_1d = Atm1D("afglt", tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0)
    atm3 = Atm3D(
        atm_1d=atm_1d,
        grid_3d=grid3,
        comp_3d=comp_3d,
        wavelength_phase=[550.0],
    )
    return atm3.calc(WAV, n_theta=NTH)


@pytest.fixture(scope="module")
def scene() -> Scene:
    """Build the grid, the two clouds and the mixed profile."""
    grid3 = _build_grid()
    cld1, cld2 = _build_clouds()
    pro = _build_profile(grid3, [cld1, cld2])
    return grid3, cld1, cld2, pro


def _voxel_props(
    pro: xr.Dataset,
    grid3: Grid3D,
    cell: tuple[int, int, int],
) -> tuple[float, float, NDArray[np.float64]]:
    """Return the ext, ssa and phase matrix of one voxel."""
    icell = np.ravel_multi_index(
        cell, (grid3.NX, grid3.NY, grid3.NZ)
    )
    k = int(pro["iopt_atm"].values[icell])
    ext = float(pro["OD_p"].values[0, k])
    ssa = float(pro["ssa_p_atm"].values[0, k])
    ipha = int(pro["iphase_atm"].values[0, k])
    pha = pro["phase_atm"].values[ipha]
    return ext, ssa, pha


def _expected(
    cld: Cloud3D | Aer3D,
) -> Expected:
    """Return the per cell ext, ssa and phase matrices."""
    ext = cld.get_ext(WAV)[0]
    ssa = cld.get_ssa(WAV)[0]
    luts, idx, _ = cld.get_phase_set(WAV, n_theta=NTH)
    pha = [luts[idx[j]].data for j in range(len(idx))]
    return ext, ssa, pha


def test_two_components_accepted(scene: Scene) -> None:
    # the profile combines the NZ + 1 background levels and the four
    # cloudy voxels of the union (the ice cloud cell is shared with
    # the water cloud), each with its own optical properties
    """Check that two components give one optical set per voxel."""
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


def test_extinction_summation(scene: Scene) -> None:
    """Check that the extinctions of a voxel add up."""
    grid3, cld1, cld2, pro = scene
    e1, _, _ = _expected(cld1)
    e2, _, _ = _expected(cld2)
    for j, cell in enumerate(WC_CELLS):
        ext, _, _ = _voxel_props(pro, grid3, cell)
        expected = e1[j] + (e2[0] if cell == IC_CELL else 0.0)
        assert np.isclose(ext, expected, rtol=1e-12), cell


def test_ssa_weighting(scene: Scene) -> None:
    """Check that the albedos are weighted by the extinctions."""
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


def test_phase_mixing_shared_voxel(scene: Scene) -> None:
    """Check the phase matrix a voxel holding two components gets."""
    grid3, cld1, cld2, pro = scene
    e1, s1, p1 = _expected(cld1)
    e2, s2, p2 = _expected(cld2)
    _, _, pha = _voxel_props(pro, grid3, IC_CELL)
    j = WC_CELLS.index(IC_CELL)
    expected = (e1[j] * s1[j] * p1[j] + e2[0] * s2[0] * p2[0]) / (
        e1[j] * s1[j] + e2[0] * s2[0]
    )
    assert np.allclose(pha, expected, rtol=1e-5, atol=1e-9)
    # the ice crystals are non-spherical: F22 differs from F11 in the
    # mixture
    assert np.max(np.abs(pha[4] - pha[0])) > 0.0


def test_phase_single_component_voxels(scene: Scene) -> None:
    # in the water-only voxels the stored matrix is the water cloud
    # one, as it is: normalized by the extinction instead of the
    # scattering, it was scaled by the single scattering albedo, which
    # the local estimate, reading the matrix as it is, carried into
    # the radiance
    """Check that a lone component keeps its own phase matrix."""
    grid3, cld1, _, pro = scene
    _, _, p1 = _expected(cld1)
    for j, cell in enumerate(WC_CELLS):
        if cell == IC_CELL:
            continue
        _, _, pha = _voxel_props(pro, grid3, cell)
        assert np.allclose(pha, p1[j], rtol=1e-5, atol=1e-9), cell
        # water droplets are spherical: F22 == F11
        assert np.allclose(pha[4], pha[0], rtol=1e-12), cell


def test_phase_kept_in_iquv_convention(scene: Scene) -> None:
    # the profile stores the phase matrices in the IQUV convention of
    # the source files, the conversion into the parallel/perpendicular
    # convention of the kernels being done by the run method: F12 of a
    # spherical water cloud is nonzero, whereas its parallel/
    # perpendicular counterpart 0.5 * (F11 - F22) would be zero
    """Check that the mixed matrix keeps its second element non null."""
    grid3, _, _, pro = scene
    cell = next(c for c in WC_CELLS if c != IC_CELL)
    _, _, pha = _voxel_props(pro, grid3, cell)
    assert np.max(np.abs(pha[1])) > 0.0


def test_empty_voxels_no_atmosphere(scene: Scene) -> None:
    """Check that a grid without atmosphere carries no optical depth."""
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


def test_component_order_invariance(scene: Scene) -> None:
    # the per-voxel properties do not depend on the component order
    """Check that the order of the components does not matter."""
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


def _build_aerosol(name: str = "desert", **kwargs: Any) -> Aer3D:
    """Build a 3D aerosol field of the given mixture."""
    kwargs.setdefault("w_ref", 550.0)
    kwargs.setdefault("ext_ref", AER_EXT)
    kwargs.setdefault("rh", AER_RH)
    kwargs.setdefault("cell_indices", AER_CELL_INDICES)
    return Aer3D(name, **kwargs)


@pytest.fixture(scope="module")
def aer_scene() -> AerScene:
    """Build the grid, the cloud, the aerosol and their profile.

    The aerosol shares all four voxels of the water cloud.
    """
    grid3 = _build_grid()
    cld1, _ = _build_clouds()
    aer = _build_aerosol()
    pro = _build_profile(grid3, [cld1, aer])
    return grid3, cld1, aer, pro


@pytest.fixture(scope="module")
def desert_ds() -> xr.Dataset:
    """Return the desert OPAC bulk properties, to expect from."""
    fname = (
        DIR_AUXDATA / "aerosols" / "OPAC" / "mixtures" / "desert_sol.nc"
    )
    return xr.open_dataset(fname)


def _dense_aerosol_dataset() -> xr.Dataset:
    """Build the dense dataset matching the raw route inputs."""
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


def test_aer3d_raw_route() -> None:
    """Check the aerosol field built from arrays."""
    aer = _build_aerosol()
    assert np.array_equal(
        aer.get_cell_indices(), AER_CELL_INDICES - 1
    )
    assert np.array_equal(aer.get_ext_ref(), AER_EXT)
    assert np.array_equal(aer.rh, AER_RH)
    assert aer.w_ref == 550.0


def test_aer3d_dense_dataset_route() -> None:
    """Check that the dense dataset route gives the same field."""
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


def test_aer3d_dataset_missing_var() -> None:
    """Check that a dataset missing 'rh' is refused."""
    ds = _dense_aerosol_dataset().drop_vars("rh")
    with pytest.raises(ValueError, match="'rh'"):
        Aer3D("desert", ds=ds)


def test_aer3d_raw_route_missing_args() -> None:
    """Check that the raw route refuses an incomplete set of arrays."""
    with pytest.raises(ValueError, match="rh, ext_ref and"):
        Aer3D(
            "desert",
            w_ref=550.0,
            ext_ref=AER_EXT,
            cell_indices=AER_CELL_INDICES,
        )


def test_aer3d_unknown_species() -> None:
    """Check that an unknown mixture raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _build_aerosol("no_such_mixture")


def test_read_i3rc_aerosol(tmp_path: Path) -> None:
    # synthetic I3RC/IPRT-style ASCII field matching the raw route
    """Check the reading of an I3RC style 3D aerosol file."""
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


def test_aer3d_ext_spectral_scaling(desert_ds: xr.Dataset) -> None:
    # on-grid rh and wavelengths, so the expected values are direct
    # file lookups:
    # ext(wavelength) = ext_ref * k(rh, wavelength) / k(rh, w_ref)
    """Check that the extinction is scaled to the other wavelengths."""
    wavelength_axis = desert_ds["wav"].values
    iw_ref = int(np.abs(wavelength_axis - 550.0).argmin())
    w_ref = float(wavelength_axis[iw_ref])
    wavelength = np.array([w_ref, float(wavelength_axis[iw_ref + 2])])
    aer = _build_aerosol(w_ref=w_ref)
    ext = aer.get_ext(wavelength)
    for j, rh in enumerate(AER_RH):
        k = desert_ds["ext"].sel(hum=rh)
        for iw, w in enumerate(wavelength):
            expected = AER_EXT[j] * float(
                k.sel(wav=w) / k.sel(wav=w_ref)
            )
            assert np.isclose(ext[iw, j], expected, rtol=1e-9), (iw, j)
    assert np.allclose(ext[0, :], AER_EXT, rtol=1e-9)


def test_aer3d_ssa_values(desert_ds: xr.Dataset) -> None:
    """Check the albedo, constant or read from the mixture."""
    wavelength_axis = desert_ds["wav"].values
    w = float(wavelength_axis[np.abs(wavelength_axis - 550.0).argmin()])
    aer = _build_aerosol(w_ref=w)
    ssa = aer.get_ssa(np.array([w]))
    for j, rh in enumerate(AER_RH):
        expected = float(desert_ds["ssa"].sel(hum=rh, wav=w))
        assert np.isclose(ssa[0, j], expected, rtol=1e-9), j
    aer_cst = _build_aerosol(w_ref=w, ssa_cst=0.9)
    assert np.all(aer_cst.get_ssa(np.array([w])) == 0.9)


def test_aer3d_rh_clamping() -> None:
    # rh values outside the OPAC humidity axis (0-99 %) are clamped
    # to its extrema, as in the 1D AerOPAC
    """Check that a humidity outside the table is clamped to it."""
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


def test_aer3d_rh_acc_min_max() -> None:
    """Check the humidity accepted between its bounds."""
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


def test_aer3d_hydrophobic_species() -> None:
    # 'inso' has a single humidity node: rh is ignored (clamped to it)
    """Check that a hydrophobic mixture ignores the humidity."""
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


def test_aer3d_phase_stk_signature() -> None:
    # desert aerosols are non-spherical (6-term phase matrices): F22
    # differs from F11; continental_clean is spherical (4-term): its
    # F22 (row 4) is a copy of F11, populated by the 4 -> 6 expansion
    """Check that the aerosol phase matrix has its polarised terms."""
    one_cell = {
        "ext_ref": AER_EXT[:1],
        "rh": AER_RH[:1],
        "cell_indices": AER_CELL_INDICES[:1],
    }
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


def test_cloud_aerosol_mixing(aer_scene: AerScene) -> None:
    # water cloud and desert aerosol in the same voxels: extinctions
    # summed, ssa extinction-weighted, phase matrices weighted by the
    # scattering coefficients
    """Check a voxel mixing a cloud and an aerosol."""
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
        ) / (e_c[j] * s_c[j] + e_a[j] * s_a[j])
        assert np.allclose(pha, expected, rtol=1e-5, atol=1e-9), cell
        # the non-spherical desert makes F22 differ from F11 even
        # though the water droplets are spherical
        assert np.max(np.abs(pha[4] - pha[0])) > 0.0, cell


def test_two_aerosols_mixing() -> None:
    # desert and continental_clean sharing one voxel
    """Check a voxel mixing two aerosol mixtures."""
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
    ) / (e_d[1] * s_d[1] + e_c[0] * s_c[0])
    assert np.allclose(pha, expected, rtol=1e-5, atol=1e-9)

    # desert-only voxel: its own matrix, not scaled by its single
    # scattering albedo (about 0.9), with the non-spherical signature
    _, _, pha = _voxel_props(pro, grid3, (0, 0, 1))
    assert np.allclose(pha, p_d[0], rtol=1e-5, atol=1e-9)
    assert np.max(np.abs(pha[4] - pha[0])) > 0.0
    # continental-only voxel: spherical, F22 == F11
    _, _, pha = _voxel_props(pro, grid3, (2, 0, 1))
    assert np.allclose(pha[4], pha[0], rtol=1e-12)


# ===================================================================
# 1D aerosols under the 3D components
# ===================================================================


def _profile_over_1d_aerosol(
    comp_3d: list[Cloud3D | Aer3D],
    wavelength: NDArray[np.float64] = WAV,
) -> xr.Dataset:
    """Build the profile of 3D components over a 1D aerosol."""
    atm_1d = Atm1D(
        "afglt", comp=[AerOPAC("continental_clean", 0.2, 550.0)],
        tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0,
    )
    atm3 = Atm3D(atm_1d=atm_1d, grid_3d=_build_grid(), comp_3d=comp_3d)
    return atm3.calc(wavelength, n_theta=NTH)


# 0-based voxels of the cloud of the tests below: one in the bottom
# layer (iz=0), one in the 1-3 km layer (iz=1)
LAYER_CELLS = [(0, 0, 0), (1, 0, 1)]


def _layer_cloud() -> Cloud3D:
    """Build a water cloud in the bottom and in the middle layer."""
    return Cloud3D(
        "wc",
        w_ref=550.0,
        ext_ref=np.array([5.0, 10.0]),
        reff=np.array([10.0, 12.0]),
        cell_indices=np.array([[1, 1, 1], [2, 1, 2]]),
    )


@pytest.mark.parametrize("n_comp", [1, 2], ids=["one", "two"])
def test_1d_aerosol_of_the_cell_layer(n_comp: int) -> None:
    """A 3D cell is mixed with the 1D aerosol of its own layer.

    The clear cell (3, 0, iz) of each layer carries the 1D aerosol of
    that layer alone, so the mixed cloud cells must add the cloud to
    it. A second component, an aerosol in the top layer away from the
    cloud, takes the several component path of the mixing.
    """
    grid3 = _build_grid()
    cld = _layer_cloud()
    comp_3d: list[Cloud3D | Aer3D] = [cld]
    if n_comp == 2:
        comp_3d.append(_build_aerosol(
            ext_ref=np.array([0.1]), rh=np.array([70.0]),
            cell_indices=np.array([[1, 1, 3]]),
        ))
    pro = _profile_over_1d_aerosol(comp_3d)
    e_c, s_c, p_c = _expected(cld)

    backgrounds = []
    for j, cell in enumerate(LAYER_CELLS):
        e_b, s_b, p_b = _voxel_props(pro, grid3, (3, 0, cell[2]))
        backgrounds.append(e_b)
        ext, ssa, pha = _voxel_props(pro, grid3, cell)
        ext_tot = e_c[j] + e_b
        assert np.isclose(ext, ext_tot, rtol=1e-6), cell
        assert np.isclose(
            ssa, (e_c[j] * s_c[j] + e_b * s_b) / ext_tot, rtol=1e-6
        ), cell
        expected = (e_c[j] * s_c[j] * p_c[j] + e_b * s_b * p_b) / (
            e_c[j] * s_c[j] + e_b * s_b
        )
        assert np.allclose(pha, expected, rtol=1e-5, atol=1e-9), cell
    # the two layers hold different amounts of aerosol, so that the
    # test tells the layer of a cell from its neighbours
    assert not np.isclose(backgrounds[0], backgrounds[1], rtol=1e-3)


@pytest.mark.parametrize("n_comp", [1, 2], ids=["one", "two"])
def test_1d_aerosol_phase_indices_per_wavelength(n_comp: int) -> None:
    """Each wavelength points at the phase matrices of its own.

    Over a 1D aerosol, a profile computed at two wavelengths must give
    every cell, at each wavelength, the phase matrix of the profile
    computed at that wavelength alone.
    """
    grid3 = _build_grid()

    def components() -> list[Cloud3D | Aer3D]:
        comp_3d: list[Cloud3D | Aer3D] = [_layer_cloud()]
        if n_comp == 2:
            comp_3d.append(_build_aerosol(
                ext_ref=np.array([0.1]), rh=np.array([70.0]),
                cell_indices=np.array([[1, 1, 3]]),
            ))
        return comp_3d

    wavelengths = np.array([550.0, 670.0])
    both = _profile_over_1d_aerosol(components(), wavelengths)
    iopt = both["iopt_atm"].values
    for i, wavelength in enumerate(wavelengths):
        alone = _profile_over_1d_aerosol(
            components(), np.array([wavelength])
        )
        np.testing.assert_array_equal(alone["iopt_atm"].values, iopt)
        for icell in range(grid3.NCELL):
            k = iopt[icell]
            pha = both["phase_atm"].values[both["iphase_atm"].values[i, k]]
            ref = alone["phase_atm"].values[
                alone["iphase_atm"].values[0, k]
            ]
            np.testing.assert_allclose(pha, ref, rtol=1e-6, atol=1e-10)
    # the two wavelengths differ, so that a wavelength pointing at the
    # matrices of the other one is caught
    k = iopt[np.ravel_multi_index(
        LAYER_CELLS[0], (grid3.NX, grid3.NY, grid3.NZ)
    )]
    iphase = both["iphase_atm"].values[:, k]
    assert not np.allclose(
        both["phase_atm"].values[iphase[0]],
        both["phase_atm"].values[iphase[1]],
    )


# ===================================================================
# Truncation carried by a component
# ===================================================================

GT = GTTrunc(trunc_frac=0.3, theta_tr=8.0)
DM = DMTrunc(n_streams=16)


def _expected_truncated(
    comp: Cloud3D | Aer3D,
    truncation: DMTrunc | GTTrunc | None,
) -> Expected:
    """Return the per cell ext, ssa and phase matrices, truncated.

    They are the untruncated ones of `_expected`, the matrix of each
    cell truncated alone and its extinction and albedo scaled with
    the truncated fraction.
    """
    ext, ssa, pha = _expected(comp)
    if truncation is None:
        return ext, ssa, pha
    luts, _, _ = comp.get_phase_set(WAV, n_theta=NTH)
    theta = luts[0].coords["theta_atm"].values
    pha_tr: list[NDArray[np.float64]] = []
    f: list[float] = []
    for p in pha:
        p_tr, f_j = truncate_phase(p, theta, truncation)
        pha_tr.append(p_tr)
        f.append(f_j)
    ext_tr, ssa_tr = truncated_ext_ssa(ext, ssa, np.array(f))
    assert all(0.0 < f_j < 1.0 for f_j in f)
    return (
        np.asarray(ext_tr, dtype=np.float64),
        np.asarray(ssa_tr, dtype=np.float64),
        pha_tr,
    )


def test_truncated_cloud_alone() -> None:
    """A truncated cloud alone gets its own truncated properties."""
    grid3 = _build_grid()
    cld1, _ = _build_clouds(truncation=GT)
    pro = _build_profile(grid3, [cld1])
    e, s, p = _expected_truncated(cld1, GT)
    e_full, _, _ = _expected(cld1)
    for j, cell in enumerate(WC_CELLS):
        ext, ssa, pha = _voxel_props(pro, grid3, cell)
        assert np.isclose(ext, e[j], rtol=1e-12), cell
        assert np.isclose(ssa, s[j], rtol=1e-12), cell
        assert np.allclose(pha, p[j], rtol=1e-9, atol=1e-12), cell
        assert ext < e_full[j]


@pytest.mark.parametrize(
    "trunc_cloud, trunc_aer",
    [(GT, None), (None, DM), (GT, DM)],
    ids=["cloud", "aerosol", "both"],
)
def test_truncated_component_in_shared_voxels(
    trunc_cloud: GTTrunc | None, trunc_aer: DMTrunc | None
) -> None:
    """Each component is truncated alone, then the voxel is mixed.

    The share of an untruncated component is left as it is.
    """
    grid3 = _build_grid()
    cld1, _ = _build_clouds(truncation=trunc_cloud)
    aer = _build_aerosol(truncation=trunc_aer)
    pro = _build_profile(grid3, [cld1, aer])
    e_c, s_c, p_c = _expected_truncated(cld1, trunc_cloud)
    e_a, s_a, p_a = _expected_truncated(aer, trunc_aer)
    for j, cell in enumerate(WC_CELLS):
        ext, ssa, pha = _voxel_props(pro, grid3, cell)
        ext_tot = e_c[j] + e_a[j]
        assert np.isclose(ext, ext_tot, rtol=1e-12), cell
        assert np.isclose(
            ssa, (e_c[j] * s_c[j] + e_a[j] * s_a[j]) / ext_tot,
            rtol=1e-12,
        ), cell
        expected = (
            e_c[j] * s_c[j] * p_c[j] + e_a[j] * s_a[j] * p_a[j]
        ) / (e_c[j] * s_c[j] + e_a[j] * s_a[j])
        assert np.allclose(pha, expected, rtol=1e-9, atol=1e-12), cell


def test_truncated_cloud_over_1d_aerosol() -> None:
    """A truncated cloud leaves the 1D aerosol background untouched.

    This is the pattern of the IPRT C3 case: the cells are mixed with
    the untruncated 1D aerosol, and only the cloud share of each cell
    changes.
    """
    grid3 = _build_grid()

    def profile(truncation: GTTrunc | None) -> xr.Dataset:
        cld1, _ = _build_clouds(truncation=truncation)
        atm_1d = Atm1D(
            "afglt", comp=[AerOPAC("continental_clean", 0.2, 550.0)],
            tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0,
        )
        atm3 = Atm3D(atm_1d=atm_1d, grid_3d=grid3, comp_3d=[cld1],
                     wavelength_phase=[550.0])
        return atm3.calc(WAV, n_theta=NTH)

    full = profile(None)
    trunc = profile(GT)
    cld1, _ = _build_clouds(truncation=GT)
    e_tr, s_tr, _ = _expected_truncated(cld1, GT)
    e_full, s_full, _ = _expected(cld1)

    nbz = grid3.NZ + 1
    np.testing.assert_array_equal(
        trunc["OD_p"].values[:, :nbz], full["OD_p"].values[:, :nbz]
    )
    ipha_bg = trunc["iphase_atm"].values[0, :nbz]
    np.testing.assert_array_equal(
        trunc["phase_atm"].values[ipha_bg],
        full["phase_atm"].values[full["iphase_atm"].values[0, :nbz]],
    )
    for j, cell in enumerate(WC_CELLS):
        ext_t, ssa_t, _ = _voxel_props(trunc, grid3, cell)
        ext_f, ssa_f, _ = _voxel_props(full, grid3, cell)
        assert np.isclose(ext_t - ext_f, e_tr[j] - e_full[j],
                          rtol=1e-9), cell
        assert np.isclose(
            ext_t * ssa_t - ext_f * ssa_f,
            e_tr[j] * s_tr[j] - e_full[j] * s_full[j],
            rtol=1e-9,
        ), cell


def test_truncation_once_per_unique_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cloud is truncated once per distinct matrix, not per cell."""
    calls = [0]
    func = trunc_mod.gt_phase_approx

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return func(*args, **kwargs)

    monkeypatch.setattr(trunc_mod, "gt_phase_approx", counting)
    cld1, _ = _build_clouds(truncation=GT)
    _build_profile(_build_grid(), [cld1])
    # four cells, two effective radii, one phase wavelength
    assert calls[0] == 2


def test_truncated_1d_component_with_forced_arrays() -> None:
    """Forced 1D aerosol arrays cannot go with a truncated 1D one."""
    atm_1d = Atm1D(
        "afglt",
        comp=[AerOPAC("continental_clean", 0.2, 550.0, truncation=GT)],
    )
    grid3 = _build_grid()
    ext = np.zeros((1, grid3.NZ + 1))
    with pytest.raises(ValueError, match="aer_ext_1d"):
        Atm3D(atm_1d=atm_1d, grid_3d=grid3, aer_ext_1d=ext)


# ===================================================================
# Grid3D neighbours
# ===================================================================


def test_periodic_single_cell_axis_wraps_onto_itself() -> None:
    """A periodic axis of one cell is its own neighbour, not a wall.

    The +Y and -Y faces of the 4 x 1 x 3 periodic grid were absorbing
    boundaries (-5), which kill every photon crossing them. The other
    faces are unchanged, and so is a non-periodic grid.
    """
    grid3 = _build_grid()
    cells = np.arange(grid3.NCELL)
    np.testing.assert_array_equal(grid3.neigh[2], cells)
    np.testing.assert_array_equal(grid3.neigh[3], cells)
    # +X of the last x cell wraps onto the first one
    last = np.ravel_multi_index((3, 0, 1), (grid3.NX, grid3.NY, grid3.NZ))
    first = np.ravel_multi_index((0, 0, 1), (grid3.NX, grid3.NY, grid3.NZ))
    assert grid3.neigh[0, last] == first
    # the top and the bottom stay the TOA (-1) and the BOA (-2)
    assert set(grid3.neigh[4, grid3.idz == grid3.NZ - 1]) == {-1}
    assert set(grid3.neigh[5, grid3.idz == 0]) == {-2}

    column = Grid3D(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]),
        np.array([0.0, 1.0, 2.0, 3.0]), periodic=True,
    )
    for face in range(4):
        np.testing.assert_array_equal(column.neigh[face], np.arange(3))

    closed = Grid3D(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0]),
        np.array([0.0, 1.0, 3.0, 4.0]),
    )
    assert set(closed.neigh[2]) == {-5} and set(closed.neigh[3]) == {-5}
    assert closed.neigh[0, last] == -5


def test_grid3d_invalid_arguments_raise_value_error() -> None:
    """Invalid grid arguments raise a ValueError, not a NameError."""
    x = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="1D"):
        Grid3D(x.reshape(1, 3), x, z)
    with pytest.raises(ValueError, match="ascending"):
        Grid3D(x[::-1], x, z)
    with pytest.raises(ValueError, match="horiz_extend_length"):
        Grid3D(x, x, z, periodic=True, horiz_extend_length=1.0)
    with pytest.raises(ValueError, match="vertical extend limit"):
        Grid3D(x, x, z, vert_extend_limit=0.5)
    with pytest.raises(ValueError, match="loc"):
        create_1d_grid(3, 1.0, loc=[1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="type"):
        extend_1d_grid(x, 1.0, type="foo")


# ===================================================================
# Phase wavelengths
# ===================================================================


@pytest.mark.parametrize(
    "n_comp, aer_1d",
    [(0, True), (1, False), (1, True), (2, True)],
    ids=["1d-only", "one", "one-over-1d", "two-over-1d"],
)
def test_fewer_phase_wavelengths_than_profile_ones(
    n_comp: int, aer_1d: bool
) -> None:
    """Every profile wavelength takes the nearest phase wavelength.

    With one phase wavelength for three profile ones, calc raised a
    CoordinateValidationError: the phase indices had one row per phase
    wavelength. Each wavelength must get the matrices of the profile
    computed at the phase wavelength alone.
    """
    grid3 = _build_grid()

    def profile(wavelength: list[float]) -> xr.Dataset:
        comp = [AerOPAC("continental_clean", 0.2, 550.0)] if aer_1d else []
        atm_1d = Atm1D(
            "afglt", comp=comp, tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0
        )
        comp_3d: list[Cloud3D | Aer3D] = [_layer_cloud()][:n_comp]
        if n_comp == 2:
            comp_3d.append(_build_aerosol(
                ext_ref=np.array([0.1]), rh=np.array([70.0]),
                cell_indices=np.array([[2, 1, 2]]),
            ))
        atm3 = Atm3D(atm_1d=atm_1d, grid_3d=grid3, comp_3d=comp_3d,
                     wavelength_phase=[550.0])
        return atm3.calc(np.array(wavelength), n_theta=NTH)

    three = profile([500.0, 550.0, 600.0])
    alone = profile([550.0])
    assert three["iphase_atm"].shape[0] == 3
    for iopt in range(alone.sizes["iopt"]):
        ref = alone["phase_atm"].values[alone["iphase_atm"].values[0, iopt]]
        for iw in range(3):
            pha = three["phase_atm"].values[
                three["iphase_atm"].values[iw, iopt]
            ]
            np.testing.assert_allclose(pha, ref, rtol=1e-6, atol=1e-10)


def test_forced_4_term_1d_aerosol_phase() -> None:
    """A forced 4-term 1D aerosol phase is completed to 6 terms.

    With a term coordinate, the mixing kept the 4 common terms of
    every cell and the non-spherical ice cloud lost its F22 and F44;
    without one, xarray raised an AlignmentError. The 1D aerosol is
    spherical: its 4 terms give the profile of its 6.
    """
    grid3 = _build_grid()
    cld = Cloud3D(
        "ic_baum_ghm", w_ref=550.0, ext_ref=np.array([2.0]),
        reff=np.array([30.0]), cell_indices=np.array([[2, 1, 2]]),
    )
    comp = [AerOPAC("continental_clean", 0.2, 550.0)]
    kwargs = {"tau_r": 0.0, "no2": False, "tco3": 0.0, "tcwp": 0.0}
    ref = Atm3D(Atm1D("afglt", comp=comp, **kwargs), grid3, [cld]).calc(
        WAV, n_theta=NTH
    )
    ds_1d = Atm1D(
        "afglt", comp=comp, grid=grid3.zGRID[::-1], **kwargs
    ).calc(WAV, n_theta=NTH)
    pha_4 = ds_1d["phase_atm"][:, :4, :].rename(nphamat="stk")
    for phases in (pha_4, pha_4.drop_vars("stk")):
        pro = Atm3D(
            Atm1D("afglt", comp=comp, **kwargs), grid3, [cld],
            aer_phase_1d=(ds_1d["iphase_atm"].values, phases),
        ).calc(WAV, n_theta=NTH)
        np.testing.assert_allclose(
            pro["phase_atm"].values, ref["phase_atm"].values, rtol=1e-6
        )
    _, _, pha = _voxel_props(pro, grid3, (1, 0, 1))
    assert np.max(np.abs(pha[4] - pha[0])) > 0.0
