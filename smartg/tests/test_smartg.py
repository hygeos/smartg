#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMART-G test suite using pytest
"""

import pytest
import numpy as np
import xarray as xr
from smartg.smartg import Alis, LocalEstimate, Smartg
from smartg.surface import RoughSurface, LambSurface
from smartg.albedo import AlbedoCst
from smartg.atmosphere import Atm1D, AerOPAC, Cloud
from smartg.water import HydrosolPR, Water1D
from smartg.reptran import Reptran, reduce_reptran
from smartg.view import smartg_view
from smartg.xarray import dataset_to_mlut
from smartg import conftest

NBPHOTONS = 1e4

wavelength_list = [500.0, [500.0], np.array([400.0, 600.0])]


@pytest.fixture(params=[True, False])
def sg(request):
    """
    A fixture to create multiple Smartg instances (pp or spherical)
    """
    return Smartg(pp=request.param)


@pytest.mark.parametrize("pp", [True, False])
@pytest.mark.parametrize("back", [True, False])
def test_compile(pp, back):
    Smartg(pp=pp, back=back)


def test_basic(request):
    """Most basic test"""
    m = Smartg(autoinit=True).run(
        500.0, atmosphere=Atm1D("afglms"), nb_photons=NBPHOTONS
    )
    smartg_view(m)
    conftest.savefig(request)


@pytest.mark.parametrize("wavelength", wavelength_list)
def test_atm(sg, wavelength):
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])
    m = sg.run(wavelength, atmosphere=atmosphere, nb_photons=NBPHOTONS)
    assert ("wavelength" in m.coords) == ("__getitem__" in dir(wavelength))


@pytest.mark.parametrize("wavelength", wavelength_list)
def test_cloud(sg, wavelength):
    atmosphere = Atm1D(
        "afglt",
        comp=[
            AerOPAC("desert", 0.1, 550.0),
            Cloud("wc", 12.68, 2, 3, 10.0, 550.0),
        ],
        grid=[100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.0],
        pfgrid=[100.0, 10.0, 0.0],
    )
    m = sg.run(wavelength, atmosphere=atmosphere, nb_photons=NBPHOTONS)
    assert ("wavelength" in m.coords) == ("__getitem__" in dir(wavelength))


@pytest.mark.parametrize("wavelength", wavelength_list)
@pytest.mark.parametrize("thv", [0.0, 40.0])
@pytest.mark.parametrize(
    "surface", [RoughSurface(wind=2.0), LambSurface(alb=AlbedoCst(0.2))]
)
def test_atm_surf(sg, wavelength, surface, thv):
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])

    sg.run(wavelength, atmosphere=atmosphere, surface=surface,
           th_deg=thv, nb_photons=NBPHOTONS)


def test_surf_iop1_1():
    surface = RoughSurface(wind=10.0)
    water = Water1D(comp=[HydrosolPR(chl=1.0)])
    Smartg().run([400.0, 500.0], surface=surface, water=water,
                 nb_photons=NBPHOTONS)


def test_atm_surf_iop1():
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
                 water=water, nb_photons=NBPHOTONS)


def test_reptran(sg):
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])
    surface = RoughSurface(wind=2.0)

    ibands = Reptran("reptran_solar_msg").to_smartg("msg1")

    res = sg.run(ibands.l, atmosphere=atmosphere, surface=surface,
                 water=None, nb_photons=NBPHOTONS)
    reduce_reptran(res, ibands)


def test_locale_estimate(sg):
    atmosphere = Atm1D("afglt")
    surface = RoughSurface()
    wavelength = 400.0
    res = sg.run(
        wavelength,
        atmosphere=atmosphere,
        surface=surface,
        th_deg=10.0,
        le={
            "th_deg": np.array([40.0], dtype="float32"),
            "phi_deg": np.array([30.0], dtype="float32"),
        },
        nb_photons=NBPHOTONS,
    )
    assert (res["I_up (TOA)"].values > 0).all()


def test_local_estimate_angles():
    """Degrees and radians build the same local estimate"""
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


def test_local_estimate_count_level():
    """The zenith angles and the levels are ravelled and typed"""
    le = LocalEstimate(
        th_deg=[[0.0, 30.0]],
        phi_deg=[[0.0, 90.0]],
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
def test_local_estimate_invalid(kwargs):
    """An incomplete or inconsistent local estimate is rejected"""
    with pytest.raises(ValueError):
        LocalEstimate(**kwargs)


def test_alis_defaults():
    """The optional ALIS options keep the documented defaults"""
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
def test_alis_invalid(kwargs):
    """The combinations the kernel cannot honour are rejected"""
    with pytest.raises(ValueError):
        Alis(**kwargs)


def test_le_dict_deprecated(sg):
    """The legacy le dictionary runs, warns, and is left untouched"""
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
            nb_photons=NBPHOTONS,
        )
    assert (res["I_up (TOA)"].values > 0).all()
    assert sorted(le) == ["phi_deg", "th_deg"]


def test_alis_options_dict_deprecated(sg):
    """The legacy alis_options keys map to the Alis parameters"""
    with pytest.warns(DeprecationWarning, match="Alis"):
        sg.run(
            np.array([400.0, 600.0]),
            atmosphere=Atm1D("afglt"),
            surface=RoughSurface(),
            alis_options={"nlow": -1, "njac": 0},
            nb_photons=NBPHOTONS,
        )


def test_dataset_to_mlut_roundtrip():
    """The run output converts losslessly to the legacy MLUT"""
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])
    res = Smartg(autoinit=True).run(
        np.array([400.0, 600.0]),
        atmosphere=atmosphere,
        surface=LambSurface(alb=AlbedoCst(0.1)),
        nb_photons=NBPHOTONS,
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
def test_rng(rng):
    atmosphere = Atm1D("afglt")
    surface = RoughSurface()
    wavelength = np.linspace(400, 800, 5)
    Smartg(rng=rng).run(wavelength, atmosphere=atmosphere,
                        surface=surface, nb_photons=NBPHOTONS)


def test_adjacency():
    pytest.skip("Cannot test this, it is still in progress.")


def test_no_aer_output():
    atm1 = Atm1D("afglt")
    water = Water1D(grid=[0, -5.0], comp=[HydrosolPR(chl=0.5)])
    surface = RoughSurface(wind=5.0, nh2o=1.34)
    sg = Smartg()
    le = {
        "th_deg": np.array([0.0, 45.0, 89.9]),
        "phi_deg": np.array([0.0, 15.6, 289.0]),
    }
    m1 = sg.run(
        550.0,
        atmosphere=atm1,
        surface=surface,
        water=water,
        output_layers=3,
        le=le,
        nb_photons=1e6,
        nb_loop=1e6,
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
