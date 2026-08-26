#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMART-G test suite using pytest
"""

import pytest
import numpy as np
from smartg.smartg import Smartg
from smartg.surface import RoughSurface, LambSurface
from smartg.albedo import AlbedoCst
from smartg.atmosphere import Atm1D, AerOPAC, Cloud
from smartg.water import HydrosolPR, Water1D
from smartg.reptran import Reptran, reduce_reptran
from smartg.view import smartg_view
from smartg import conftest

NBPHOTONS = 1e4

wav_list = [500.0, [500.0], np.array([400.0, 600.0])]


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
        500.0, atm=Atm1D("afglms"), nb_photons=NBPHOTONS
    )
    smartg_view(m)
    conftest.savefig(request)


@pytest.mark.parametrize("wav", wav_list)
def test_atm(sg, wav):
    atm = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])
    m = sg.run(wav, atm=atm, nb_photons=NBPHOTONS)
    m = m.to_xarray() if hasattr(m, "to_xarray") else m
    assert ("wavelength" in m.coords) == ("__getitem__" in dir(wav))


@pytest.mark.parametrize("wav", wav_list)
def test_cloud(sg, wav):
    atm = Atm1D(
        "afglt",
        comp=[
            AerOPAC("desert", 0.1, 550.0),
            Cloud("wc", 12.68, 2, 3, 10.0, 550.0),
        ],
        grid=[100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.0],
        pfgrid=[100.0, 10.0, 0.0],
    )
    m = sg.run(wav, atm=atm, nb_photons=NBPHOTONS)
    m = m.to_xarray() if hasattr(m, "to_xarray") else m
    assert ("wavelength" in m.coords) == ("__getitem__" in dir(wav))


@pytest.mark.parametrize("wav", wav_list)
@pytest.mark.parametrize("thv", [0.0, 40.0])
@pytest.mark.parametrize(
    "surf", [RoughSurface(wind=2.0), LambSurface(alb=AlbedoCst(0.2))]
)
def test_atm_surf(sg, wav, surf, thv):
    atm = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])

    sg.run(wav, atm=atm, surf=surf, th_v_deg=thv, nb_photons=NBPHOTONS)


def test_surf_iop1_1():
    surf = RoughSurface(wind=10.0)
    water = Water1D(comp=[HydrosolPR(chl=1.0)])
    Smartg().run([400.0, 500.0], surf=surf, water=water, nb_photons=NBPHOTONS)


def test_atm_surf_iop1():
    atm = Atm1D(
        "afglt",
        comp=[AerOPAC("desert", 0.1, 550.0)],
        pfwav=[500.0, 600.0],
        pfgrid=[100.0, 5.0, 0.0],
    )
    surf = RoughSurface(wind=10.0)
    water = Water1D(
        comp=[HydrosolPR(chl=1.0, pfwav=np.array([450, 550, 650, 750]))]
    )
    wav = np.linspace(400, 800, 12)
    Smartg().run(wav, atm=atm, surf=surf, water=water, nb_photons=NBPHOTONS)


def test_reptran(sg):
    atm = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])
    surf = RoughSurface(wind=2.0)

    ibands = Reptran("reptran_solar_msg").to_smartg("msg1")

    res = sg.run(ibands.l, atm=atm, surf=surf, water=None, nb_photons=NBPHOTONS)
    reduce_reptran(res, ibands)


def test_locale_estimate(sg):
    atm = Atm1D("afglt")
    surf = RoughSurface()
    wav = 400.0
    res = sg.run(
        wav,
        atm=atm,
        surf=surf,
        th_v_deg=10.0,
        le={
            "th_deg": np.array([40.0], dtype="float32"),
            "phi_deg": np.array([30.0], dtype="float32"),
        },
        nb_photons=NBPHOTONS,
    )
    res = res.to_xarray() if hasattr(res, "to_xarray") else res
    assert (res["I_up (TOA)"].values > 0).all()


@pytest.mark.parametrize("rng", ["PHILOX", "CURAND_PHILOX"])
def test_rng(rng):
    atm = Atm1D("afglt")
    surf = RoughSurface()
    wav = np.linspace(400, 800, 5)
    Smartg(rng=rng).run(wav, atm=atm, surf=surf, nb_photons=NBPHOTONS)


def test_adjacency():
    pytest.skip("Cannot test this, it is still in progress.")


def test_no_aer_output():
    atm1 = Atm1D("afglt")
    water = Water1D(grid=[0, -5.0], comp=[HydrosolPR(chl=0.5)])
    surf = RoughSurface(wind=5.0, nh2o=1.34)
    sg = Smartg()
    le = {
        "th_deg": np.array([0.0, 45.0, 89.9]),
        "phi_deg": np.array([0.0, 15.6, 289.0]),
    }
    m1 = sg.run(
        550.0,
        atm=atm1,
        surf=surf,
        water=water,
        output_layers=3,
        le=le,
        nb_photons=1e6,
        nb_loop=1e6,
        no_aer_output=True,
    )
    m1 = m1.to_xarray() if hasattr(m1, "to_xarray") else m1
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
