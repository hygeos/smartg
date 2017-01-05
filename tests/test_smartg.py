#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Suite de tests SMARTG
'''

from __future__ import print_function, division
from smartg import Smartg, RoughSurface, LambSurface
from smartg import Environment, Albedo_cst
from atmosphere import AtmAFGL, AeroOPAC, CloudOPAC, read_phase
from water import IOP_Rw, IOP_1, IOP
from tools.luts import read_mlut
import numpy as np
from itertools import product
from unittest import skip
from reptran import REPTRAN, reduce_reptran
from tempfile import NamedTemporaryFile
from os import system

wav_list = [
            500.,
            np.array(500.),
            [400.],
            [500., 600.],
            np.linspace(400, 800, 4),
           ]

progress=False


class Runner(object):
    def __init__(self):
        self.Spp = Smartg(pp=True)
        self.Ssp = Smartg(pp=False)

    def run_pp_sp(self, wav, atm=None, surf=None, water=None):
        self.run_pp(wav, atm, surf, water)
        self.run_sp(wav, atm, surf, water)

    def print_time(self, attrs):
        print('total time={:.2f}s, kernel time={:.2f}s'.format(
                    float(attrs['processing time (s)']),
                    float(attrs['kernel time (s)']),
                    ))

    def check_save_read(self, res):
        for fmt in ['hdf4', 'netcdf4']:
            with NamedTemporaryFile() as f:
                res.save(f.name, overwrite=True, fmt=fmt, verbose=True)
                res1 = read_mlut(f.name, fmt=fmt)
                assert res.equal(res1, show_diff=True, attributes=False)
                print('Checked {} output: OK'.format(fmt))


    def run_pp(self, wav, atm=None, surf=None, water=None):
        res = self.Spp.run(wav, THVDEG=30, atm=atm, surf=surf, water=water,
                            progress=progress, NBPHOTONS=1e5)
        self.print_time(res.attrs)
        res.describe()

        self.check_save_read(res)

        return res

    def run_sp(self, wav, atm=None, surf=None, water=None):
        res = self.Ssp.run(wav, THVDEG=30, atm=atm, surf=surf, water=water,
                            progress=progress, NBPHOTONS=1e4)
        self.print_time(res.attrs)
        return res

runner = Runner()


def test_profile1():
    atm = AtmAFGL('afglt')
    for wav in wav_list:
        yield atm.calc, wav

def test_profile2():
    atm = AtmAFGL('afglms',
                  grid=[100., 50., 20., 10., 5., 2., 1., 0.],
                  pfgrid=[100., 10., 0.])
    for wav in wav_list:
        yield atm.calc, wav

def test_profile3():
    atm = AtmAFGL('afglms',
                  comp=[AeroOPAC('desert', 0.1, 550.)],
                  grid='100[20]10[1]0',
                  pfgrid=[100., 10., 0.])
    for wav in wav_list:
        yield atm.calc, wav

def test_profile4():
    atm = AtmAFGL('afglms',
                  comp=[AeroOPAC('desert', 0.1, 550.),
                        CloudOPAC('wc.sol', 12.68, 2, 3, 10., 550.),
                       ],
                  grid=[100., 50., 20., 10., 5., 2., 1., 0.],
                  pfgrid=[100., 10., 0.])
    for wav in wav_list:
        yield atm.calc, wav

def test_profile5():
    # set tauray
    pro = AtmAFGL('afglms', grid=[100, 20, 0.], tauR=0.14).calc(500.)
    pro.describe()
    assert np.isclose(pro['OD_r'][0,-1], 0.14)

    AtmAFGL('afglms', grid=[100, 20, 0.], tauR=0.14).calc([490.,500.])
    AtmAFGL('afglms', grid=[100, 20, 0.], tauR=[0.15, 0.14]).calc([490.,500.])

def test_profile6():
    # set ssa
    AtmAFGL('afglms', grid=[100, 20, 0.],
            comp=[AeroOPAC('urban', 0.1, 550., ssa=0.8)]
            ).calc(400.)

    AtmAFGL('afglms', grid=[100, 20, 0.],
            comp=[AeroOPAC('urban', 0.1, 550., ssa=0.8)]
            ).calc([400., 500., 600.])

    AtmAFGL('afglms', grid=[100, 20, 0.],
            comp=[AeroOPAC('urban', 0.1, 550., ssa=[0.76, 0.77, 0.78])]
            ).calc([400., 500., 600.])

def test_wav():

    res = runner.run_pp(500., atm=AtmAFGL('afglt'))
    assert 'wavelength' not in res.axes

    res = runner.run_pp([500.], atm=AtmAFGL('afglt'))
    assert 'wavelength' in res.axes

    res = runner.run_pp([400., 500.], atm=AtmAFGL('afglt'))
    assert 'wavelength' in res.axes

def test_aerosols1():
    atm = AtmAFGL('afglt', comp=[AeroOPAC('desert', 0.1, 550.) ])

    runner.run_pp_sp([400., 500.], atm=atm)
    runner.run_pp_sp(400., atm=atm)

def test_aerosols2():
    atm = AtmAFGL('afglt',
                  comp=[AeroOPAC('desert', 0.1, 550.),
                        CloudOPAC('wc.sol', 12.68, 2, 3, 10., 550.),
                       ],
                  grid=[100., 50., 20., 10., 5., 2., 1., 0.],
                  pfgrid=[100., 10., 0.])

    runner.run_pp_sp([400., 500., 600.], atm=atm)
    runner.run_pp_sp(400., atm=atm)

def test_atm_surf():
    atm = AtmAFGL('afglt', comp=[AeroOPAC('desert', 0.1, 550.)])

    runner.run_pp_sp([400., 500., 600.], atm=atm, surf=RoughSurface(WIND=2.))
    runner.run_pp_sp(400., atm=atm, surf=LambSurface(ALB=0.2))


def test_surf_iop1_1():
    surf = RoughSurface(WIND=10.)
    water = IOP_1(chl=1.)
    runner.run_pp([400., 500., 600.], surf=surf, water=water)
    runner.run_pp(400., surf=surf, water=water)

def test_surf_iop1_2():
    atm = AtmAFGL('afglt', comp=[AeroOPAC('desert', 0.1, 550.)], pfwav=[500., 600.], pfgrid=[100., 5., 0.])
    surf = RoughSurface(WIND=10.)
    water = IOP_1(chl=1., pfwav=np.array([450, 550, 650, 750]))
    wav = np.linspace(400, 800, 12)
    runner.run_pp_sp(wav, atm=atm, surf=surf, water=water)
    runner.run_pp_sp(400., atm=atm, surf=surf, water=water)

def test_surf_iop_1():
    #
    # test IOP in absence of a phase function (pure Rayleigh)
    #
    wl=[390.,450.,550.,650.]
    grid=[0.,15] # in m for water, here sea bottom at 15m depth

    # Seafloor albedo, here grey lambertian reflection with albedo of 0.1
    ALB=Albedo_cst(0.1)

    # particle absorption and scatering coefficient profiles (in m-1) and grid
    aw=np.array([[0.    , 0.     , 0.     , 0.]                                     , 
                 [0.0204, 0.0092 , 0.0565 , 0.3400]]).T # with shape (N wavelengths , N levels)
    bw=np.array([[0.,     0.,     0.,     0.],
                 [0.0134, 0.0045, 0.0019, 0.0010]]).T
    water_custom = IOP(phase=None, aw=aw, bw=bw, Z=grid, ALB=ALB)

    runner.run_pp_sp(wl, water=water_custom)


def test_iop():
    wav = np.array([400., 500., 600.])
    nwav = len(wav)
    iop1 = IOP_1(1.).calc_iop(wav)
    P = IOP_1(1.).phase(wav, np.array([0.05, 0.05, 0.05]))

    bp = np.zeros((nwav, 2), dtype='float32')
    bp[:,1] = iop1['bp']
    atot = np.zeros((nwav, 2), dtype='float32')
    atot[:,1] = iop1['atot']
    iop = IOP(phase=P['phase'], bp=bp, atot=atot)

    runner.run_pp(wav, water=iop)
    runner.run_pp(wav, surf=RoughSurface(), water=iop)

def test_atm_surf_ioprw():
    atm = AtmAFGL('afglt', comp=[AeroOPAC('desert', 0.1, 550.)])
    surf = RoughSurface(WIND=2.)
    water = IOP_Rw(ALB=Albedo_cst(0.05))

    runner.run_pp_sp([400., 500., 600.], atm=atm, surf=surf, water=water)
    runner.run_pp_sp(400., atm=atm, surf=surf, water=water)

def test_reuse1():
    atm = AtmAFGL('afglt', comp=[AeroOPAC('desert', 0.1, 550.)])
    surf = RoughSurface(WIND=2.)
    water = IOP_Rw(ALB=Albedo_cst(0.05))

    wl = [400., 500., 600.]

    res = runner.run_pp(wl, atm=atm, surf=surf, water=water)

    # reuse atm
    runner.run_pp(wl, atm=res, surf=surf, water=water)

    # reuse water
    runner.run_pp(wl, atm=atm, surf=surf, water=res)

    # reuse both
    runner.run_pp(wl, atm=res, surf=surf, water=res)


def test_reuse2():
    atm = AtmAFGL('afglt', comp=[AeroOPAC('desert', 0.1, 550.)])
    surf = RoughSurface(WIND=2.)
    water = IOP_Rw(ALB=Albedo_cst(0.05))

    wl = 500.

    res = runner.run_pp(wl, atm=atm, surf=surf, water=water)

    # reuse atm
    runner.run_pp(wl, atm=res, surf=surf, water=water)

    # reuse water
    runner.run_pp(wl, atm=atm, surf=surf, water=res)

    # reuse both
    runner.run_pp(wl, atm=res, surf=surf, water=res)


def test_reptran():
    atm = AtmAFGL('afglt', comp=[AeroOPAC('desert', 0.1, 550.)])
    surf = RoughSurface(WIND=2.)
    # water = IOP_Rw(ALB=Albedo_cst(0.05))

    ibands = REPTRAN('reptran_solar_msg').to_smartg('msg1')

    res = runner.run_pp(ibands.l, atm=atm, surf=surf, water=None)
    reduce_reptran(res, ibands)

    res = runner.run_sp(ibands.l, atm=atm, surf=surf, water=None)
    reduce_reptran(res, ibands)


def test_rng():
    atm = AtmAFGL('afglt')
    surf = RoughSurface()
    water = IOP_1(chl=1.)
    wav = np.linspace(400, 800, 12)
    for rng in [
                'CURAND_PHILOX',
                'PHILOX',
                ]:
        Smartg(rng=rng).run(wav, atm=atm,
                            surf=surf, water=water,
                            NBPHOTONS=1e6, progress=False)



