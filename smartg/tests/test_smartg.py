#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
SMART-G test suite using pytest
'''

import pytest
import numpy as np
from smartg.smartg import Smartg, RoughSurface, LambSurface, Albedo_cst
from smartg.atmosphere import AtmAFGL, AerOPAC, Cloud
from smartg.water import IOP_1
from smartg.reptran import REPTRAN, reduce_reptran
from smartg.tools.smartg_view import smartg_view
import os
from smartg import conftest

NBPHOTONS = 1e4

wav_list = [500.,
            [500.],
            np.array([400., 600.])
            ]

@pytest.fixture(params=[True, False])
def sg(request):
    '''
    A fixture to create multiple Smartg instances (pp or spherical)
    '''
    return Smartg(pp=request.param)


@pytest.mark.parametrize('pp', [True, False])
@pytest.mark.parametrize('back', [True, False])
def test_compile(pp, back):
    Smartg(pp=pp, back=back)

def test_basic(request):
    ''' Most basic test '''
    m = Smartg(autoinit=True).run(500., atm=AtmAFGL('afglms'), NBPHOTONS=NBPHOTONS)
    smartg_view(m)
    conftest.savefig(request)

@pytest.mark.parametrize('wav', wav_list)
def test_atm(sg, wav):
    atm = AtmAFGL('afglt', comp=[AerOPAC('desert', 0.1, 550.)])
    m = sg.run(wav, atm=atm, NBPHOTONS=NBPHOTONS)
    assert ('wavelength' in m.axes) == ('__getitem__' in dir(wav))

@pytest.mark.parametrize('wav', wav_list)
def test_cloud(sg, wav):
    atm = AtmAFGL('afglt',
                  comp=[AerOPAC('desert', 0.1, 550.),
                        Cloud('wc', 12.68, 2, 3, 10., 550.),
                       ],
                  grid=[100., 50., 20., 10., 5., 2., 1., 0.],
                  pfgrid=[100., 10., 0.])
    m = sg.run(wav, atm=atm, NBPHOTONS=NBPHOTONS)
    assert ('wavelength' in m.axes) == ('__getitem__' in dir(wav))

@pytest.mark.parametrize('wav', wav_list)
@pytest.mark.parametrize('thv', [0., 40.])
@pytest.mark.parametrize('surf', [RoughSurface(WIND=2.), LambSurface(ALB=Albedo_cst(0.2))])
def test_atm_surf(sg, wav, surf, thv):
    atm = AtmAFGL('afglt', comp=[AerOPAC('desert', 0.1, 550.)])

    sg.run(wav,
           atm=atm,
           surf=surf,
           THVDEG=thv,
           NBPHOTONS=NBPHOTONS)


def test_surf_iop1_1():
    surf = RoughSurface(WIND=10.)
    water = IOP_1(chl=1.)
    Smartg().run([400., 500.], surf=surf, water=water, NBPHOTONS=NBPHOTONS)


def test_atm_surf_iop1():
    atm = AtmAFGL('afglt', comp=[AerOPAC('desert', 0.1, 550.)],
                  pfwav=[500., 600.], pfgrid=[100., 5., 0.])
    surf = RoughSurface(WIND=10.)
    water = IOP_1(chl=1., pfwav=np.array([450, 550, 650, 750]))
    wav = np.linspace(400, 800, 12)
    Smartg().run(wav, atm=atm, surf=surf, water=water, NBPHOTONS=NBPHOTONS)


def test_reptran(sg):
    atm = AtmAFGL('afglt', comp=[AerOPAC('desert', 0.1, 550.)])
    surf = RoughSurface(WIND=2.)

    ibands = REPTRAN('reptran_solar_msg').to_smartg('msg1')

    res = sg.run(ibands.l, atm=atm, surf=surf, water=None, NBPHOTONS=NBPHOTONS)
    reduce_reptran(res, ibands)


def test_locale_estimate(sg):
    atm = AtmAFGL('afglt')
    surf = RoughSurface()
    wav = 400.
    res = sg.run(wav, atm=atm,
                 surf=surf,
                 THVDEG=10.,
                 le={
                     'th_deg':  np.array([40.], dtype='float32'),
                     'phi_deg': np.array([30.], dtype='float32'),
                     },
                 NBPHOTONS=NBPHOTONS)
    assert res['I_up (TOA)'][:,:] > 0


@pytest.mark.parametrize('rng', ['PHILOX', 'CURAND_PHILOX'])
def test_rng(rng):
    atm = AtmAFGL('afglt')
    surf = RoughSurface()
    wav = np.linspace(400, 800, 5)
    Smartg(rng=rng).run(
        wav,
        atm=atm,
        surf=surf,
        NBPHOTONS=NBPHOTONS)


def test_adjacency():
    pytest.skip("Cannot test this, it is still in progress.")


def test_no_aer_output():
    atm1 = AtmAFGL('afglt')
    water = IOP_1(chl=0.5, DEPTH=5.)
    surf = RoughSurface(WIND=5., NH2O=1.34)
    sg = Smartg()
    le = {'th_deg': np.array([0., 45., 89.9]), 'phi_deg': np.array([0., 15.6, 289.])}
    m1 = sg.run(550., atm=atm1, surf=surf, water=water, OUTPUT_LAYERS=3, le=le,
                NBPHOTONS=1e6, NBLOOP=1e6, no_aer_output=True)
    m1 = m1.to_xarray() # to prepare transition luts to xarray
    assert (np.allclose(m1['I_up (TOA)'].values, m1['I_up (TOA), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['Q_up (TOA)'].values, m1['Q_up (TOA), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['U_up (TOA)'].values, m1['U_up (TOA), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['V_up (TOA)'].values, m1['V_up (TOA), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['I_down (0+)'].values, m1['I_down (0+), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['Q_down (0+)'].values, m1['Q_down (0+), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['U_down (0+)'].values, m1['U_down (0+), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['V_down (0+)'].values, m1['V_down (0+), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['I_down (0-)'].values, m1['I_down (0-), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['Q_down (0-)'].values, m1['Q_down (0-), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['U_down (0-)'].values, m1['U_down (0-), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['V_down (0-)'].values, m1['V_down (0-), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['I_up (0+)'].values, m1['I_up (0+), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['Q_up (0+)'].values, m1['Q_up (0+), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['U_up (0+)'].values, m1['U_up (0+), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['V_up (0+)'].values, m1['V_up (0+), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['I_up (0-)'].values, m1['I_up (0-), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['Q_up (0-)'].values, m1['Q_up (0-), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['U_up (0-)'].values, m1['U_up (0-), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['V_up (0-)'].values, m1['V_up (0-), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['I_down (B)'].values, m1['I_down (B), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['Q_down (B)'].values, m1['Q_down (B), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['U_down (B)'].values, m1['U_down (B), no_aer'].values, 0, 1e-13, equal_nan=True))
    assert (np.allclose(m1['V_down (B)'].values, m1['V_down (B), no_aer'].values, 0, 1e-13, equal_nan=True))
    