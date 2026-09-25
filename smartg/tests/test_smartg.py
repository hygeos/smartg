"""Tests of the SMART-G runs themselves, from the kernel up.

They compile each variant of the kernel, run an atmosphere, a
surface and water in the combinations the code allows, and check
the outputs and the arguments the run accepts.
"""

from typing import Any

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from smartg import conftest
from smartg.albedo import AlbedoCst, AlbedoMap
from smartg.atmosphere import AerOPAC, Atm1D, Atm3D, Cloud
from smartg.grid3d import Grid3D
from smartg.reptran import Reptran, reduce_reptran
from smartg.sensor import Sensor, get_sensors_grid
from smartg.smartg import Alis, LocalEstimate, Smartg
from smartg.surface import Environment, LambSurface, RoughSurface
from smartg.view import smartg_view
from smartg.water import HydrosolPR, Water1D
from smartg.xarray import dataset_to_mlut

Wavelength = float | list[float] | NDArray[np.float64]

N_PHOTONS = 1e4

wavelength_list = [500.0, [500.0], np.array([400.0, 600.0])]


@pytest.fixture(params=[True, False])
def sg(request: pytest.FixtureRequest) -> Smartg:
    """Build a Smartg, plane parallel and spherical in turn."""
    return Smartg(pp=request.param)


@pytest.mark.parametrize("pp", [True, False])
@pytest.mark.parametrize("back", [True, False])
def test_compile(pp: bool, back: bool) -> None:
    """Check that each compilation of the kernel goes through."""
    Smartg(pp=pp, back=back)


def test_basic(request: pytest.FixtureRequest) -> None:
    """Run the simplest case there is."""
    m = Smartg(autoinit=True).run(
        500.0, atmosphere=Atm1D("afglms"), n_photons=N_PHOTONS
    )
    smartg_view(m)
    conftest.savefig(request)


@pytest.mark.parametrize("wavelength", wavelength_list)
def test_atm(sg: Smartg, wavelength: Wavelength) -> None:
    """Check that a run keeps a wavelength axis only for a list."""
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])
    m = sg.run(wavelength, atmosphere=atmosphere, n_photons=N_PHOTONS)
    assert ("wavelength" in m.coords) == ("__getitem__" in dir(wavelength))


@pytest.mark.parametrize("wavelength", wavelength_list)
def test_cloud(sg: Smartg, wavelength: Wavelength) -> None:
    """Check the same for a run carrying a water cloud."""
    atmosphere = Atm1D(
        "afglt",
        comp=[
            AerOPAC("desert", 0.1, 550.0),
            Cloud("wc", 12.68, 2, 3, 10.0, 550.0),
        ],
        grid=[100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.0],
        pfgrid=[100.0, 10.0, 0.0],
    )
    m = sg.run(wavelength, atmosphere=atmosphere, n_photons=N_PHOTONS)
    assert ("wavelength" in m.coords) == ("__getitem__" in dir(wavelength))


@pytest.mark.parametrize("wavelength", wavelength_list)
@pytest.mark.parametrize("thv", [0.0, 40.0])
@pytest.mark.parametrize(
    "surface", [RoughSurface(wind=2.0), LambSurface(alb=AlbedoCst(0.2))]
)
def test_atm_surf(
    sg: Smartg,
    wavelength: Wavelength,
    surface: LambSurface | RoughSurface,
    thv: float,
) -> None:
    """Run an aerosol atmosphere over each kind of surface."""
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])

    sg.run(wavelength, atmosphere=atmosphere, surface=surface,
           th_deg=thv, n_photons=N_PHOTONS)


def test_surf_iop1_1() -> None:
    """Run a rough surface over water, without atmosphere."""
    surface = RoughSurface(wind=10.0)
    water = Water1D(comp=[HydrosolPR(chl=1.0)])
    Smartg().run([400.0, 500.0], surface=surface, water=water,
                 n_photons=N_PHOTONS)


def test_atm_surf_iop1() -> None:
    """Run an atmosphere, a rough surface and water together."""
    atmosphere = Atm1D(
        "afglt",
        comp=[AerOPAC("desert", 0.1, 550.0)],
        wavelength_phase=[500.0, 600.0],
        pfgrid=[100.0, 5.0, 0.0],
    )
    surface = RoughSurface(wind=10.0)
    water = Water1D(
        comp=[HydrosolPR(
            chl=1.0, wavelength_phase=np.array([450, 550, 650, 750])
        )]
    )
    wavelength = np.linspace(400, 800, 12)
    Smartg().run(wavelength, atmosphere=atmosphere, surface=surface,
                 water=water, n_photons=N_PHOTONS)


def test_reptran(sg: Smartg) -> None:
    """Run the REPTRAN bands of an MSG channel."""
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])
    surface = RoughSurface(wind=2.0)

    ibands = Reptran("reptran_solar_msg").to_smartg("msg1")

    res = sg.run(ibands.l, atmosphere=atmosphere, surface=surface,
                 water=None, n_photons=N_PHOTONS)
    reduce_reptran(res, ibands)


def test_locale_estimate(sg: Smartg) -> None:
    """Check that the local estimate gives a positive radiance."""
    atmosphere = Atm1D("afglt")
    surface = RoughSurface()
    wavelength = 400.0
    res = sg.run(
        wavelength,
        atmosphere=atmosphere,
        surface=surface,
        th_deg=10.0,
        le=LocalEstimate(
            th_deg=np.array([40.0], dtype="float32"),
            phi_deg=np.array([30.0], dtype="float32"),
        ),
        n_photons=N_PHOTONS,
    )
    assert (res["I_up (TOA)"].values > 0).all()


def test_local_estimate_angles() -> None:
    """Check that degrees and radians build the same estimate."""
    deg = LocalEstimate(th_deg=[0.0, 30.0], phi_deg=[0.0, 90.0])
    rad = LocalEstimate(
        th=np.array([0.0, 30.0]) * np.pi / 180.0,
        phi=np.array([0.0, 90.0]) * np.pi / 180.0,
    )
    np.testing.assert_allclose(deg.th, rad.th, rtol=1e-6)
    np.testing.assert_allclose(deg.phi, rad.phi, rtol=1e-6)
    assert deg.th.dtype == np.float32
    assert deg.zip is False
    assert deg.count_level is None


def test_local_estimate_count_level() -> None:
    """Check that the angles and levels are ravelled and typed."""
    le = LocalEstimate(
        th_deg=np.array([[0.0, 30.0]]),
        phi_deg=np.array([[0.0, 90.0]]),
        zip=True,
        count_level=[0, 4],
    )
    assert le.zip is True
    assert le.th.shape == (2,)
    assert le.count_level is not None
    assert le.count_level.dtype == np.int32
    np.testing.assert_array_equal(le.count_level, [0, 4])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"th_deg": [0.0]},
        {"th": [0.0], "th_deg": [0.0], "phi_deg": [0.0]},
        {"th_deg": [0.0, 30.0], "phi_deg": [0.0], "zip": True},
        {"th_deg": [0.0, 30.0], "phi_deg": [0.0, 90.0],
         "count_level": [0]},
    ],
)
def test_local_estimate_invalid(kwargs: dict[str, Any]) -> None:
    """Check that an inconsistent local estimate is rejected."""
    with pytest.raises(ValueError):
        LocalEstimate(**kwargs)


def test_alis_defaults() -> None:
    """Check that the ALIS options keep their documented values."""
    alis = Alis(n_low=10)
    assert alis.n_low == 10
    assert alis.hist is False
    assert alis.max_hist == 8000000
    assert alis.n_jac == 0
    assert alis.n_jac_abs is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_low": 0},
        {"n_low": 1},
        {"n_low": 10, "n_jac_abs": True},
        {"n_low": 10, "n_jac": 0, "n_jac_abs": True},
    ],
)
def test_alis_invalid(kwargs: dict[str, Any]) -> None:
    """Check that the kernel refuses what it cannot honour."""
    with pytest.raises(ValueError):
        Alis(**kwargs)


def test_le_dict_deprecated(sg: Smartg) -> None:
    """Check that the legacy le dict runs, warns and is kept."""
    le = {
        "th_deg": np.array([40.0], dtype="float32"),
        "phi_deg": np.array([30.0], dtype="float32"),
    }
    with pytest.warns(DeprecationWarning, match="LocalEstimate"):
        res = sg.run(
            400.0,
            atmosphere=Atm1D("afglt"),
            surface=RoughSurface(),
            le=le,
            n_photons=N_PHOTONS,
        )
    assert (res["I_up (TOA)"].values > 0).all()
    assert sorted(le) == ["phi_deg", "th_deg"]


def test_alis_options_dict_deprecated() -> None:
    """Check that the legacy alis_options map onto Alis."""
    with pytest.warns(DeprecationWarning, match="Alis"):
        Smartg(alis=True).run(
            np.array([400.0, 600.0]),
            atmosphere=Atm1D("afglt"),
            surface=RoughSurface(),
            alis_options={"nlow": -1, "njac": 0},
            n_photons=N_PHOTONS,
        )


@pytest.fixture(scope="module")
def sg_forward() -> Smartg:
    """Build the default Smartg, forward and without ALIS."""
    return Smartg()


INVALID_RUNS = [
    ({"flux": "planer"}, ValueError, "unknown flux"),
    (
        {"flux": "planar", "le": LocalEstimate(th_deg=[0.0], phi_deg=[0.0])},
        ValueError,
        "flux and le",
    ),
    ({"alis_options": Alis(n_low=-1)}, ValueError, "alis=True"),
    ({"environment": Environment(env=1)}, ValueError, "needs a surface"),
    (
        {"wavelength_proba": np.zeros(4, dtype=np.int32)},
        TypeError,
        "int64",
    ),
    ({"sensor_proba": np.zeros(4, dtype=np.int32)}, TypeError, "int64"),
    ({"cell_proba": "automatic"}, ValueError, "unknown cell_proba"),
    ({"cell_proba": "auto"}, ValueError, "forward thermal"),
    ({"cell_proba": np.zeros((3, 5))}, ValueError, "one column"),
]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    INVALID_RUNS,
    ids=[
        "flux", "flux-le", "alis", "environment", "wavelength_proba",
        "sensor_proba", "cell_proba-name", "cell_proba-auto",
        "cell_proba-shape",
    ],
)
def test_run_invalid(
    sg_forward: Smartg,
    kwargs: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    """Check that the run refuses the arguments it would ignore."""
    with pytest.raises(error, match=match):
        sg_forward.run(
            500.0, atmosphere=Atm1D("afglt"), n_photons=N_PHOTONS, **kwargs
        )


def test_dataset_to_mlut_roundtrip() -> None:
    """Check that the output converts to an MLUT losslessly."""
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])
    res = Smartg(autoinit=True).run(
        np.array([400.0, 600.0]),
        atmosphere=atmosphere,
        surface=LambSurface(alb=AlbedoCst(0.1)),
        n_photons=N_PHOTONS,
    )
    assert isinstance(res, xr.Dataset)

    mlut = dataset_to_mlut(res)
    assert mlut.datasets() == list(res.data_vars)
    for name in res.data_vars:
        lut = mlut[name]
        assert lut.names == list(res[name].dims)
        np.testing.assert_array_equal(lut.data, res[name].values)
        assert dict(lut.attrs) == dict(res[name].attrs)
    assert dict(mlut.attrs) == dict(res.attrs)


@pytest.mark.parametrize("rng", ["PHILOX", "CURAND_PHILOX"])
def test_rng(rng: str) -> None:
    """Run each of the random number generators."""
    atmosphere = Atm1D("afglt")
    surface = RoughSurface()
    wavelength = np.linspace(400, 800, 5)
    Smartg(rng=rng).run(wavelength, atmosphere=atmosphere,
                        surface=surface, n_photons=N_PHOTONS)


def test_adjacency() -> None:
    """Skipped, the adjacency effect is still in progress."""
    pytest.skip("Cannot test this, it is still in progress.")


def test_no_aer_output() -> None:
    """Check that the no_aer outputs match the ones with aerosols."""
    atm1 = Atm1D("afglt")
    water = Water1D(grid=[0, -5.0], comp=[HydrosolPR(chl=0.5)])
    surface = RoughSurface(wind=5.0, nh2o=1.34)
    sg = Smartg()
    le = LocalEstimate(
        th_deg=np.array([0.0, 45.0, 89.9]),
        phi_deg=np.array([0.0, 15.6, 289.0]),
    )
    m1 = sg.run(
        550.0,
        atmosphere=atm1,
        surface=surface,
        water=water,
        output_layers=3,
        le=le,
        n_photons=1e6,
        n_loop=1e6,
        no_aer_output=True,
    )
    assert np.allclose(
        m1["I_up (TOA)"].values,
        m1["I_up (TOA), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["Q_up (TOA)"].values,
        m1["Q_up (TOA), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["U_up (TOA)"].values,
        m1["U_up (TOA), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["V_up (TOA)"].values,
        m1["V_up (TOA), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["I_down (0+)"].values,
        m1["I_down (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["Q_down (0+)"].values,
        m1["Q_down (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["U_down (0+)"].values,
        m1["U_down (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["V_down (0+)"].values,
        m1["V_down (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["I_down (0-)"].values,
        m1["I_down (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["Q_down (0-)"].values,
        m1["Q_down (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["U_down (0-)"].values,
        m1["U_down (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["V_down (0-)"].values,
        m1["V_down (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["I_up (0+)"].values,
        m1["I_up (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["Q_up (0+)"].values,
        m1["Q_up (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["U_up (0+)"].values,
        m1["U_up (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["V_up (0+)"].values,
        m1["V_up (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["I_up (0-)"].values,
        m1["I_up (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["Q_up (0-)"].values,
        m1["Q_up (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["U_up (0-)"].values,
        m1["U_up (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["V_up (0-)"].values,
        m1["V_up (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["I_down (B)"].values,
        m1["I_down (B), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["Q_down (B)"].values,
        m1["Q_down (B), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["U_down (B)"].values,
        m1["U_down (B), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["V_down (B)"].values,
        m1["V_down (B), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )


def _z_scores(a: xr.Dataset, b: xr.Dataset, level: str) -> NDArray:
    """Return the differences of two runs over their joint stdev."""
    return (a[f"I_{level}"].values - b[f"I_{level}"].values) / np.sqrt(
        a[f"I_stdev_{level}"].values ** 2 + b[f"I_stdev_{level}"].values ** 2
    )


def _adjacency_run(alt_pp: bool, back: bool, seed: int) -> xr.Dataset:
    """Run a black disc of 2 km in a white environment, under dust."""
    # a single thick aerosol layer, where the height of a scattering
    # inside the layer sets how far from it the photon reaches the
    # ground
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 1.0, 550.0)],
                       grid=[100.0, 10.0, 0.0])
    kwargs: dict[str, Any] = {}
    if back:
        kwargs["sensor"] = Sensor(pos_z=100.0, th_deg=180.0, loc="ATMOS")
        th_deg = 30.0
    else:
        kwargs["th_deg"] = 30.0
        th_deg = 0.0
    le = LocalEstimate(th_deg=[th_deg], phi_deg=[0.0], count_level=[0])
    return Smartg(alt_pp=alt_pp, back=back).run(
        550.0,
        atmosphere=atmosphere,
        surface=LambSurface(alb=AlbedoCst(0.0)),
        environment=Environment(env=1, env_size=2.0, alb=AlbedoCst(1.0)),
        le=le,
        n_photons=1e7,
        stdev=True,
        seed=seed,
        progress=False,
        **kwargs,
    )


@pytest.mark.parametrize("back", [True, False])
def test_adjacency_fast_and_alt_pp_moves_agree(back: bool) -> None:
    """Check that the fast PP move places the photons at their height.

    The adjacency effect depends on where the photons reach the
    ground, so on the altitude of their scatterings: the fast move,
    which follows the optical depth, must agree with the alternative
    one, which follows the geometry.
    """
    fast = _adjacency_run(alt_pp=False, back=back, seed=11)
    alt = _adjacency_run(alt_pp=True, back=back, seed=12)
    assert np.all(np.abs(_z_scores(fast, alt, "up (TOA)")) < 5)


def _empty_atmosphere() -> Atm1D:
    """Return an atmosphere without scattering nor absorption."""
    return Atm1D("afglt", tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0)


def test_brdf_surface_in_3d_atmosphere() -> None:
    """Check that a 3D atmosphere reflects on a BRDF surface as in 1D.

    Without anything in the atmosphere the radiance is the one of the
    Cox and Munk BRDF alone, the same for every sensor of the 3D
    domain and for the 1D run.
    """
    surface = RoughSurface(wind=5.0, brdf=True)
    le = LocalEstimate(th_deg=[40.0], phi_deg=[0.0], count_level=[0])
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    grid3 = Grid3D(edges, edges, np.array([0.0, 1.0, 2.0]), periodic=True)
    sensors = get_sensors_grid(
        edges, edges, pos_z=2.0, th_deg=150.0, ph_deg=0.0, loc="ATMOS",
        cell_size=1.0, grid_3d=grid3,
    )
    m3 = Smartg(opt3d=True, alt_pp=True, back=True).run(
        550.0,
        atmosphere=Atm3D(atm_1d=_empty_atmosphere(), grid_3d=grid3),
        surface=surface,
        sensor=sensors,
        le=le,
        n_photons=1e6,
        seed=41,
        progress=False,
    )
    m1 = Smartg(alt_pp=True, back=True).run(
        550.0,
        atmosphere=_empty_atmosphere(),
        surface=surface,
        sensor=Sensor(pos_z=120.0, th_deg=150.0, ph_deg=0.0, loc="ATMOS"),
        le=le,
        n_photons=1e6,
        seed=42,
        progress=False,
    )
    np.testing.assert_allclose(
        m3["I_up (TOA)"].values.ravel(),
        float(m1["I_up (TOA)"].values.ravel()[0]),
        rtol=1e-3,
    )


def _sza_max_run(
    sg: Smartg, n_theta: int, sza_max: float, seed: int
) -> xr.Dataset:
    """Run the cone sampling of a dusty atmosphere on a zenith grid."""
    return sg.run(
        550.0,
        atmosphere=Atm1D("afglt", comp=[AerOPAC("desert", 0.3, 550.0)]),
        surface=LambSurface(alb=AlbedoCst(0.2)),
        th_deg=30.0,
        n_theta=n_theta,
        n_phi=2,
        sza_max=sza_max,
        output_layers=1,
        n_photons=1e7,
        stdev=True,
        seed=seed,
        progress=False,
    )


def test_sza_max_keeps_the_radiances(sg: Smartg) -> None:
    """Check that sza_max only cuts the zenith grid of the output.

    The 2 degree boxes up to 60 degrees are the same with 45 boxes up
    to 90 degrees and with 30 boxes up to 60 degrees, and so must be
    their radiances, at the top of the atmosphere as at the surface.
    """
    full = _sza_max_run(sg, 45, 90.0, 21)
    part = _sza_max_run(sg, 30, 60.0, 22)
    common = {"Zenith angles": slice(0, 30)}
    np.testing.assert_allclose(
        part["Zenith angles"].values, full["Zenith angles"][common].values
    )
    for level in ("up (TOA)", "down (0+)"):
        assert np.all(np.abs(_z_scores(full[common], part, level)) < 6)


def test_sun_disc_boxes_of_every_level() -> None:
    """Check that sun_disc counts the downward photons too.

    With sun_disc, a box counts only the photons within sun_disc of
    its centre, and its radiance is normalised by the solid angle of
    the disc instead of the one of the box, which divides it by 2 pi.
    Where the disc lies inside the box, 2 pi times the radiance must
    match the one of the whole box, at the top of the atmosphere as at
    the surface.
    """
    sg = Smartg()
    kwargs: dict[str, Any] = {
        "atmosphere": Atm1D("afglt", comp=[AerOPAC("desert", 0.3, 550.0)]),
        "surface": LambSurface(alb=AlbedoCst(0.2)),
        "th_deg": 30.0,
        "n_theta": 9,
        "n_phi": 36,
        "output_layers": 1,
        "n_photons": 2e7,
        "progress": False,
    }
    boxes = sg.run(550.0, seed=31, **kwargs)
    disc = sg.run(550.0, seed=32, sun_disc=2.0, **kwargs)
    for level in ("up (TOA)", "down (0+)"):
        # mean over the azimuth, where the 2 degree disc lies inside
        # the 10 x 10 degree boxes: 30 to 80 degrees of zenith angle
        ratio = (
            2 * np.pi * disc[f"I_{level}"].values.mean(axis=0)
            / boxes[f"I_{level}"].values.mean(axis=0)
        )
        np.testing.assert_allclose(ratio[3:8], 1.0, atol=0.05)


def test_albedo_map_near_the_origin_in_spherical_mode() -> None:
    """Check that a spherical run reads the albedo map at 1 km.

    Two nadir sensors look at the ground 1 km on each side of the
    origin, through an empty atmosphere, one onto a white strip of the
    map and the other onto black ground.
    """
    albedo_map = AlbedoMap(
        np.array([[0], [1], [0]]),
        np.array([0.5, 1.5, 1e8]),
        np.array([1e8]),
        [AlbedoCst(0.0), AlbedoCst(1.0)],
    )
    sensors = [
        Sensor(pos_x=x, pos_z=6371.0 + 120.0, th_deg=180.0, loc="ATMOS")
        for x in (1.0, -1.0)
    ]
    m = Smartg(pp=False, back=True).run(
        550.0,
        atmosphere=_empty_atmosphere(),
        surface=LambSurface(alb=AlbedoCst(0.0)),
        environment=Environment(env=5, alb=albedo_map),
        sensor=sensors,
        le=LocalEstimate(th_deg=[30.0], phi_deg=[0.0], count_level=[0]),
        n_photons=1e5,
        seed=51,
        progress=False,
    )
    white, black = m["I_up (TOA)"].values.ravel()
    assert white > 0.5
    assert black < 1e-6


def test_albedo_map_seafloor_below_the_land() -> None:
    """Check the seafloor below the land cells of an albedo map.

    The sensor looks at the sea near the coast, obliquely, so that the
    photons refracted into the water reach the seafloor below the land
    cell. There the seafloor keeps the albedo of the water profile,
    the same as the one the map gives below the sea cells here, so the
    radiance must be the one of a map of sea only.
    """
    sg = Smartg(back=True)
    water = Water1D(
        grid=[0.0, -5.0], comp=[HydrosolPR(chl=0.01)], alb=AlbedoCst(0.5)
    )
    alist = [AlbedoCst(0.2), AlbedoCst(0.5)]
    runs = []
    for seed, codes in ((91, [[1], [-1]]), (92, [[-1], [-1]])):
        albedo_map = AlbedoMap(
            np.array(codes), np.array([0.0, 1e8]), np.array([1e8]), alist
        )
        runs.append(sg.run(
            550.0,
            surface=RoughSurface(),
            water=water,
            environment=Environment(env=5, alb=albedo_map),
            sensor=Sensor(
                pos_x=0.5, th_deg=120.0, ph_deg=180.0, loc="SURF0P"
            ),
            le=LocalEstimate(th_deg=[30.0], phi_deg=[0.0], count_level=[0]),
            n_photons=1e6,
            stdev=True,
            seed=seed,
            progress=False,
        ))
    assert np.all(np.abs(_z_scores(runs[0], runs[1], "up (TOA)")) < 5)


def test_toa_sphere_start_without_objects() -> None:
    """Check the cell_size=-2 sensors of a run without 3D objects.

    They start on the top of atmosphere sphere: a far sensor whose line
    of sight grazes the atmosphere 30 km above the ground, missing the
    Earth, sees the light the atmosphere scatters, as does one looking
    at the ground.
    """
    distance = 6371.0 + 1e4
    alpha = np.degrees(np.arcsin((6371.0 + 30.0) / distance))
    sensors = [
        Sensor(pos_z=distance, th_deg=180.0 - a, ph_deg=0.0, loc="ATMOS",
               cell_size=-2)
        for a in (alpha, 10.0)
    ]
    m = Smartg(pp=False, obj3d=True, back=True).run(
        550.0,
        atmosphere=Atm1D("afglt"),
        sensor=sensors,
        n_photons=1e5,
        seed=61,
        progress=False,
    )
    assert np.all(m["N_up (TOA)"].values.reshape(2, -1).sum(axis=1) > 0)
