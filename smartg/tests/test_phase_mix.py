"""Mixing phase matrices tabulated on different scattering angle grids.

An OPAC aerosol carries 1801 equally spaced angles, a cloud file 594
angles clustered in its diffraction peak. The device holds one grid
per medium, so a mixture has to be brought onto one grid: the union
of the components' grids, on which a sum of piecewise linear tables
is exact. ``n_theta='native'`` asks for that union, and a mixture
that comes back on different grids (a user phase matrix keeps its
own) is resampled onto it with a warning rather than silently
reduced to the common angles, which is what xarray's arithmetic did.

Everything here runs on the CPU except the last test, which needs a
GPU and checks the radiance in the forward peak of the mixture.
"""

import warnings
from typing import Any

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from smartg.atmosphere import Aer3D, AerOPAC, Atm1D, Atm3D, Cloud, Cloud3D
from smartg.grid3d import Grid3D
from smartg.phase import (
    as_theta_grid,
    is_native_theta,
    theta_grid,
    union_theta_grid,
)
from smartg.smartg import _calc_phase_host
from smartg.truncation import DMTrunc, GTTrunc
from smartg.typing import ThetaLike

WAVELENGTH = 550.0
WAV = np.array([WAVELENGTH])
GRID = [100.0, 50.0, 20.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
PFGRID = [100.0, 0.0]
MISMATCH = "different phase angle grids"


def _aerosol(**kwargs: Any) -> AerOPAC:
    """Build the OPAC aerosol the 1D tests mix."""
    return AerOPAC("continental_clean", 0.2, WAVELENGTH, **kwargs)


def _cloud(**kwargs: Any) -> Cloud:
    """Build the water cloud the 1D tests mix."""
    return Cloud("wc", 12.68, 2.0, 3.0, 5.0, WAVELENGTH, **kwargs)


def _atm(comps: list[AerOPAC | Cloud]) -> Atm1D:
    """Build the 1D atmosphere holding the given components."""
    return Atm1D("afglms", comp=comps, grid=GRID, pfgrid=PFGRID)


def _file_matrix(cloud: Cloud) -> xr.DataArray:
    """Return the cloud's own phase matrix as a user matrix.

    It is taken at the reff of the cloud and at 550 nm, on the grid
    of the file, in the 2-D shape the ``phase`` argument accepts.
    """
    ds = cloud.ds_mix
    da = ds["phase"].sel(reff=cloud.reff, method="nearest").sel(
        wav=WAVELENGTH * 1e-3, method="nearest"
    )
    return xr.DataArray(
        da.values,
        dims=["nphamat", "theta_atm"],
        coords={"theta_atm": ds.coords["theta"].values.astype(float)},
    )


def _contains(
    grid: NDArray[np.floating],
    nodes: NDArray[np.floating],
    tol: float = 1e-6,
) -> np.bool_:
    """Whether every node lies within *tol* degrees of a grid node."""
    i = np.clip(np.searchsorted(grid, nodes), 1, len(grid) - 1)
    return np.all(
        np.minimum(np.abs(grid[i] - nodes), np.abs(grid[i - 1] - nodes))
        <= tol
    )


# --------------------------------------------------------------------
# the union of grids
# --------------------------------------------------------------------


def test_union_keeps_every_node_once() -> None:
    """Check that the union holds each angle once, as float64."""
    union = union_theta_grid([[0.0, 90.0, 180.0], [0.0, 45.0, 90.0, 180.0]])
    assert np.array_equal(union, [0.0, 45.0, 90.0, 180.0])
    assert union.dtype == np.float64


def test_union_merges_float32_with_float64_angles() -> None:
    # the OPAC files store 0.1 as a float32, the cloud files as a
    # float64: one node, the smaller of the two kept
    """Check that a float32 grid merges cleanly with a float64 one."""
    coarse = np.array([0.0, 0.1, 0.2, 180.0], dtype=np.float32)
    fine = np.array([0.0, 0.1, 0.15, 0.2, 180.0])
    union = union_theta_grid([coarse, fine])
    assert len(union) == 5
    assert np.all(np.diff(union) > 0.0)
    assert _contains(union, coarse.astype(np.float64))
    assert _contains(union, fine)


def test_union_keeps_distinct_nodes_a_hundredth_of_a_degree_apart() -> None:
    """Check that angles 0.01 degree apart stay distinct."""
    union = union_theta_grid([[0.0, 0.01, 180.0], [0.0, 0.02, 180.0]])
    assert np.array_equal(union, [0.0, 0.01, 0.02, 180.0])


@pytest.mark.parametrize(
    "grids", ([], [[0.0, 90.0]], [[0.0, 90.0, 180.0], [10.0, 180.0]])
)
def test_union_rejects_bad_input(grids: list[list[float]]) -> None:
    """Check that a grid outside 0 to 180 degrees is refused."""
    with pytest.raises(ValueError):
        union_theta_grid(grids)


def test_native_is_resolved_by_the_phase_methods_only() -> None:
    """Check that 'native' is recognised but not built here."""
    assert is_native_theta("native")
    assert not is_native_theta(721)
    assert not is_native_theta(theta_grid(5))
    with pytest.raises(TypeError, match="native"):
        as_theta_grid("native")


# --------------------------------------------------------------------
# a 1D aerosol + cloud mixture
# --------------------------------------------------------------------


def test_component_native_grids_are_the_file_grids() -> None:
    """Check that each component reports the grid of its file."""
    aer, cld = _aerosol(), _cloud()
    assert np.array_equal(
        aer.native_theta(),
        aer.ds_mix.coords["theta"].values.astype(np.float64),
    )
    assert np.array_equal(
        cld.native_theta(),
        cld.ds_mix.coords["theta"].values.astype(np.float64),
    )
    # the two grids the whole module is about: they differ, and
    # neither contains the other
    assert not _contains(aer.native_theta(), cld.native_theta())
    assert not _contains(cld.native_theta(), aer.native_theta())


def test_native_mixture_lives_on_the_union_and_says_so() -> None:
    """Check that a native mixture takes the union and warns."""
    aer, cld = _aerosol(), _cloud()
    atm = _atm([aer, cld])
    with pytest.warns(UserWarning, match=MISMATCH):
        pha = atm.phase(WAV, n_theta="native")
    assert pha is not None
    theta = pha.coords["theta_atm"].values

    union = union_theta_grid([aer.native_theta(), cld.native_theta()])
    assert np.array_equal(theta, union)
    assert _contains(theta, aer.native_theta())
    assert _contains(theta, cld.native_theta())
    assert not np.isnan(pha.values).any()


def test_native_mixture_reproduces_each_component_exactly() -> None:
    """Resampled onto the union, a component is still its own table.

    Read back on its native nodes, the union table of each component
    is the file table: no node is lost, which is what makes the
    union the right grid to mix on.
    """
    aer, cld = _aerosol(), _cloud()
    atm = _atm([aer, cld])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        union = atm.native_theta()
    rh = atm.prof_red.relative_humidity()
    z = np.asarray(PFGRID, dtype=np.float64)

    for comp in (aer, cld):
        native = comp.phase(WAV, z, rh, n_theta="native")
        on_union = comp.phase(WAV, z, rh, n_theta=union)
        nodes = native.coords["theta_atm"].values
        for iterm in range(6):
            got = np.interp(
                nodes, union, on_union.values[0, 0, iterm].astype(float)
            )
            np.testing.assert_allclose(
                got, native.values[0, 0, iterm], rtol=1e-5, atol=1e-6
            )


def test_native_mixture_is_the_scattering_weighted_sum() -> None:
    """Check the mixed matrix against the weighted sum by hand."""
    aer, cld = _aerosol(), _cloud()
    atm = _atm([aer, cld])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixed = atm.phase(WAV, n_theta="native")
        assert mixed is not None
        union = mixed.coords["theta_atm"].values
    rh = atm.prof_red.relative_humidity()
    z = np.asarray(PFGRID, dtype=np.float64)

    num = 0.0
    den = 0.0
    for comp in (aer, cld):
        dtau, ssa = comp.dtau_ssa(WAV, z, rh=rh)
        w = dtau[0, 1] * ssa[0, 1]
        num = num + w * comp.phase(WAV, z, rh, n_theta=union).values[0, 0]
        den = den + w
    np.testing.assert_allclose(mixed.values[0, 0], num / den, rtol=1e-5)


def test_user_matrix_on_its_own_grid_is_mixed_on_the_union() -> None:
    """The regression this module guards against.

    A cloud built with its file matrix keeps the file grid whatever
    ``n_theta``; the aerosol comes back on the 721 default. The sum
    used to keep only the angles common to both: 175 of them, with
    nothing said. It is now the union of the two grids.
    """
    aer = _aerosol()
    cld = _cloud(phase=_file_matrix(_cloud()))
    with pytest.warns(UserWarning, match=MISMATCH):
        pha = _atm([aer, cld]).phase(WAV)
    assert pha is not None
    theta = pha.coords["theta_atm"].values

    expected = union_theta_grid([theta_grid(721), cld.native_theta()])
    assert np.array_equal(theta, expected)
    assert len(theta) > 721
    assert not np.isnan(pha.values).any()


def test_same_grid_mixture_neither_warns_nor_resamples() -> None:
    """Check that a shared grid is kept as it is, silently."""
    aer, cld = _aerosol(), _cloud()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pha = _atm([aer, cld]).phase(WAV, n_theta=721)
        assert pha is not None
    assert np.array_equal(pha.coords["theta_atm"].values, theta_grid(721))


def test_wavelength_axes_cannot_be_merged() -> None:
    # a 2-D user matrix is monochromatic, the OPAC aerosol is not:
    # there is no union of wavelengths, so this must not go through
    """Check that differing wavelength axes are refused."""
    aer = _aerosol()
    cld = _cloud(phase=_file_matrix(_cloud()))
    with pytest.raises(ValueError, match="wavelength_phase"):
        _atm([aer, cld]).phase(np.array([500.0, 600.0]))


def test_single_user_matrix_keeps_its_grid() -> None:
    # one component, no mixing: the default n_theta does not touch a
    # user matrix, as before
    """Check that a lone user matrix keeps its own grid."""
    cld = _cloud(phase=_file_matrix(_cloud()))
    pha = _atm([cld]).phase(WAV)
    assert pha is not None
    assert np.array_equal(
        pha.coords["theta_atm"].values, cld.native_theta()
    )


def test_calc_carries_the_union_to_the_profile() -> None:
    """Check that the profile takes the union as its angle axis."""
    aer, cld = _aerosol(), _cloud()
    atm = _atm([aer, cld])
    with pytest.warns(UserWarning, match=MISMATCH):
        pro = atm.calc(WAV, n_theta="native")
    union = union_theta_grid([aer.native_theta(), cld.native_theta()])
    assert np.array_equal(pro.coords["theta_atm"].values, union)
    assert pro["phase_atm"].shape[-1] == len(union)


def test_the_device_table_adopts_the_union_intact() -> None:
    """Built on its own grid, the table is the profile matrix."""
    aer, cld = _aerosol(), _cloud()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pro = _atm([aer, cld]).calc(WAV, n_theta="native")
    union = pro.coords["theta_atm"].values
    phase, cdf = _calc_phase_host(
        pro, len(union), 0.0279, "atm", ang_a=np.deg2rad(union)
    )
    assert phase.shape == (3, len(union)) == cdf.shape
    # the kernel reconstructs F11 from the parallel/perpendicular
    # terms; compare shapes, the table being normalised
    got = phase["a_P11"][2] + phase["a_P22"][2] + 2.0 * phase["a_P12"][2]
    expected = pro["phase_atm"].values[0, 0]
    np.testing.assert_allclose(
        got / got[-1], expected / expected[-1], rtol=1e-5
    )


@pytest.mark.parametrize(
    "truncation",
    [
        DMTrunc(n_streams=16, integral_method="lobatto"),
        DMTrunc(n_streams=16, integral_method="trapezoid"),
        DMTrunc(n_streams=16, integral_method="simpson"),
        GTTrunc(trunc_frac=0.3, integral_method="lobatto"),
    ],
    ids=["DM-lobatto", "DM-trapezoid", "DM-simpson", "GT-lobatto"],
)
def test_truncation_accepts_the_union_grid(
    truncation: DMTrunc | GTTrunc,
) -> None:
    """The truncation comes back on the irregular union, unchanged.

    The union has 0.01 degree bins in the peak and 1 degree bins in
    the body; the truncation of each component must integrate that
    grid as it is and give the truncation factor it gives on an
    equally spaced grid of the same order. Measured against 3601
    equally spaced angles, the factor moves by 4e-4 at most, less than
    the 6e-4 the integration methods differ by among themselves on
    that grid. The comparison here is against 1801 angles, to keep the
    test short, and the trapezoid factor on that grid is itself 1.6e-3
    from its converged value, which the union already reaches: hence
    the tolerance. The factor is the one of the whole particle column.
    Delta-M truncates both components; GT truncates the cloud alone,
    since on the aerosol, whose phase function has no marked forward
    peak, it leaves a negative phase function on the union.
    """
    trunc_aer = truncation if isinstance(truncation, DMTrunc) else None

    def factor(n_theta: ThetaLike) -> tuple[float, xr.Dataset]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = _atm([_aerosol(), _cloud()]).calc(WAV, n_theta=n_theta)
            trunc = _atm(
                [
                    _aerosol(truncation=trunc_aer),
                    _cloud(truncation=truncation),
                ]
            ).calc(WAV, n_theta=n_theta)
        f = 1.0 - trunc["OD_p"].values[0, -1] / full["OD_p"].values[0, -1]
        return f, trunc

    f_union, pro = factor("native")
    union = union_theta_grid(
        [_aerosol().native_theta(), _cloud().native_theta()]
    )
    assert np.array_equal(pro.coords["theta_atm"].values, union)
    assert pro["phase_atm"].shape[-1] == len(union)
    assert not np.isnan(pro["phase_atm"].values).any()

    f_uniform, _ = factor(theta_grid(1801))
    assert abs(f_union - f_uniform) < 2e-3


# --------------------------------------------------------------------
# the 3D merge
# --------------------------------------------------------------------


def _grid3() -> Grid3D:
    """Build the small periodic grid of the 3D tests."""
    return Grid3D(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0, 3.0, 4.0]),
        periodic=True,
    )


def _cloud3d() -> Cloud3D:
    """Build the 3D cloud the 3D tests mix."""
    return Cloud3D(
        "wc", w_ref=WAVELENGTH,
        ext_ref=np.array([5.0, 10.0]), reff=np.array([10.0, 12.0]),
        cell_indices=np.array([[1, 1, 2], [2, 1, 2]]),
    )


def _aer3d() -> Aer3D:
    """Build the 3D aerosol the 3D tests mix."""
    return Aer3D(
        "desert", w_ref=WAVELENGTH,
        ext_ref=np.array([0.1, 0.2]), rh=np.array([70.0, 80.0]),
        cell_indices=np.array([[2, 1, 2], [3, 1, 2]]),
    )


def _voxel(
    pro: xr.Dataset,
    grid3: Grid3D,
    cell: tuple[int, int, int],
) -> NDArray[np.float64]:
    """Return the phase matrix of one voxel of the profile."""
    icell = np.ravel_multi_index(cell, (grid3.NX, grid3.NY, grid3.NZ))
    k = int(pro["iopt_atm"].values[icell])
    ipha = int(pro["iphase_atm"].values[0, k])
    return pro["phase_atm"].values[ipha]


def test_3d_native_mixture_lives_on_the_union() -> None:
    """Check the 3D mixture, on the union, voxel by voxel."""
    grid3 = _grid3()
    cld, aer = _cloud3d(), _aer3d()
    atm3 = Atm3D(
        atm_1d=Atm1D("afglt", tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0),
        grid_3d=grid3, comp_3d=[cld, aer], wavelength_phase=[WAVELENGTH],
    )
    with pytest.warns(UserWarning, match=MISMATCH):
        pro = atm3.calc(WAV, n_theta="native")
    union = union_theta_grid([cld.native_theta(), aer.native_theta()])
    assert np.array_equal(pro.coords["theta_atm"].values, union)

    # the shared voxel, 0-based (1, 0, 1): the scattering weighted sum
    # of the two components on the union
    e_c, s_c = cld.get_ext(WAV)[0], cld.get_ssa(WAV)[0]
    e_a, s_a = aer.get_ext(WAV)[0], aer.get_ssa(WAV)[0]
    p_c, i_c, _ = cld.get_phase_set(WAV, n_theta=union)
    p_a, i_a, _ = aer.get_phase_set(WAV, n_theta=union)
    expected = (
        e_c[1] * s_c[1] * p_c[i_c[1]].values
        + e_a[0] * s_a[0] * p_a[i_a[0]].values
    ) / (e_c[1] + e_a[0])
    np.testing.assert_allclose(
        _voxel(pro, grid3, (1, 0, 1)), expected, rtol=1e-5, atol=1e-9
    )


def test_3d_merge_compares_grids_not_lengths() -> None:
    """A 1D aerosol grid of the component grid's length is not it.

    The multi-component merge used to relabel the 1D aerosol matrix
    with the component grid whenever the two had the same number of
    angles. Here they have 181 each, on different nodes: the merged
    profile has to carry their union.
    """
    n = 181
    user = _file_matrix(_cloud()).interp(theta_atm=theta_grid(n, "lobatto"))
    aer_1d = AerOPAC("continental_clean", 0.1, WAVELENGTH, phase=user)
    grid3 = _grid3()
    atm3 = Atm3D(
        atm_1d=Atm1D(
            "afglt", comp=[aer_1d],
            tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0,
        ),
        grid_3d=grid3, comp_3d=[_cloud3d(), _aer3d()],
        wavelength_phase=[WAVELENGTH],
    )
    with pytest.warns(UserWarning, match=MISMATCH):
        pro = atm3.calc(WAV, n_theta=n)
    theta = pro.coords["theta_atm"].values
    assert np.array_equal(
        theta, union_theta_grid([theta_grid(n), theta_grid(n, "lobatto")])
    )
    assert not np.isnan(pro["phase_atm"].values).any()


# --------------------------------------------------------------------
# on the GPU
# --------------------------------------------------------------------

_PEAK_ANGLES = np.array([0.2, 1.0, 20.0, 60.0])


def test_native_mixture_radiance_matches_a_fine_uniform_grid() -> None:
    """The union grid gives the radiance of a very fine uniform one.

    Transmitted radiance under a thin aerosol + cloud layer with the
    sun at the zenith, so the scattering angle is the viewing angle
    and the forward peak of the cloud is what is being measured. The
    native union (about 2000 angles) has to agree with 18001 equally
    spaced angles everywhere, peak included, whereas the 721 angle
    default cannot: measured 2.2% off at 0.2 degrees, the aerosol
    diluting the cloud peak that a cloud alone shows at 11%. Both
    tolerances sit well above the seed to seed spread of a kernel
    that does not reproduce from one run to the next (0.2% measured
    on the same setup).
    """
    pytest.importorskip("pycuda")
    from smartg.smartg import LocalEstimate, Smartg

    def radiance(n_theta: ThetaLike) -> NDArray[np.float64]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            profile = Atm1D(
                "afglms",
                comp=[
                    AerOPAC("continental_clean", 0.1, WAVELENGTH),
                    Cloud("wc", 12.68, 2.0, 3.0, 0.05, WAVELENGTH),
                ],
                grid=GRID, pfgrid=PFGRID,
            ).calc(WAV, n_theta=n_theta)
        m = Smartg(pp=True, double=True).run(
            WAVELENGTH, atmosphere=profile, th_deg=0.0,
            le=LocalEstimate(
                th_deg=_PEAK_ANGLES,
                phi_deg=np.array([0.0]),
                count_level=np.full(len(_PEAK_ANGLES), 1),
            ),
            output_layers=3, theta_grid="phase", n_photons=2e7,
            seed=1234, xblock=128, xgrid=1024,
        )
        key = next(str(k) for k in m if str(k).startswith("I_down"))
        return np.atleast_1d(np.squeeze(m[key].values)).ravel()[
            :len(_PEAK_ANGLES)]

    reference = radiance(theta_grid(18001))
    native = radiance("native")
    default = radiance(721)

    err_native = np.abs(native - reference) / reference
    err_default = np.abs(default - reference) / reference
    assert err_native.max() < 0.01
    assert err_default[0] > 0.015
