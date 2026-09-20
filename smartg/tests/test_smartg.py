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
from smartg.albedo import AlbedoCst
from smartg.atmosphere import AerOPAC, Atm1D, Cloud
from smartg.reptran import Reptran, reduce_reptran
from smartg.smartg import Alis, LocalEstimate, Smartg
from smartg.surface import LambSurface, RoughSurface
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
def test_atm_surf(sg: Smartg, wavelength: Wavelength, surface: LambSurface | RoughSurface, thv: float) -> None:
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


def test_alis_options_dict_deprecated(sg: Smartg) -> None:
    """Check that the legacy alis_options map onto Alis."""
    with pytest.warns(DeprecationWarning, match="Alis"):
        sg.run(
            np.array([400.0, 600.0]),
            atmosphere=Atm1D("afglt"),
            surface=RoughSurface(),
            alis_options={"nlow": -1, "njac": 0},
            n_photons=N_PHOTONS,
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
