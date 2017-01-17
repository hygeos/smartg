#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
SMART-G examples
'''

from __future__ import print_function, division, absolute_import
from smartg.smartg import Smartg, LambSurface, RoughSurface
from smartg.atmosphere import AtmAFGL, AeroOPAC
from smartg.tools.luts import merge
from smartg.reptran import REPTRAN, reduce_reptran
from smartg.water import IOP_1
import numpy as np


def test_rayleigh():
    '''
    Basic Rayleigh example
    '''
    res = Smartg().run(wl=400., NBPHOTONS=1e6, atm=AtmAFGL('afglt'), progress=False)

    # NOTE:
    # the result is an object of class MLUT, that has many features, including
    # writing and reading in netcdf or hdf formats, interpolation, plotting,
    # indexing, etc.
    # example: to save in netcdf4 format, use:
    #          res.save('result.nc')

def test_sp():
    '''
    Basic test in spherical
    '''
    Smartg(pp=False).run(wl=400., NBPHOTONS=1e6, atm=AtmAFGL('afglt'), progress=False)

def test_rayleigh_grid():
    '''
    Use a custom atmosphere grid
    '''
    Smartg().run(wl=500., NBPHOTONS=1e7,
           atm=AtmAFGL('afglt', grid='100[75]25[5]10[1]0'), progress=False)


def test_aerosols():
    '''
    test with aerosols
    '''
    aer = AeroOPAC('maritime_clean', 0.4, 550.)
    pro = AtmAFGL('afglms', comp=[aer])
    Smartg().run(wl=490., atm=pro, NBPHOTONS=1e6, progress=False)


def test_atm_surf():
    '''
    atmosphere + lambertian surface of albedo 10%
    '''
    S = Smartg()
    return S.run(490., NBPHOTONS=1e6,
                 atm=AtmAFGL('afglms'),
                 surf=LambSurface(ALB=0.1), progress=False)


def test_atm_surf_ocean():
    S = Smartg()
    return S.run(490., NBPHOTONS=1e6, THVDEG=30.,
                 atm=AtmAFGL('afglms',
                             comp=[AeroOPAC('maritime_clean', 0.2, 550)]),
                 surf=RoughSurface(),
                 water=IOP_1(chl=1., NANG=1000), progress=False)


def test_surf_ocean():
    S = Smartg()
    return S.run(490., THVDEG=30., NBPHOTONS=1e6,
                 surf=RoughSurface(),
                 water=IOP_1(1., pfwav=[400.]), progress=False)


def test_ocean():
    S = Smartg()
    return S.run(wl=560., THVDEG=30.,
                 water=IOP_1(100.), NBPHOTONS=1e6, progress=False)


def test_reptran():
    aer = AeroOPAC('maritime_polluted', 0.4, 550.)
    pro = AtmAFGL('afglms.dat', comp=[aer], grid='100[75]25[5]10[1]0')
    ibands = REPTRAN('reptran_solar_msg').to_smartg('msg1_seviri_ch008')

    res = Smartg().run(ibands.l,
                       NBPHOTONS=1e7,
                       atm=pro, progress=False)

    return reduce_reptran(res, ibands)

def test_le():
    atm = AtmAFGL('afglt')
    surf = RoughSurface()
    water = IOP_1(chl=1.)
    wav = 450.
    res = Smartg().run(wav,
                 atm=atm,
                 surf=surf,
                 water=water,
                 THVDEG=10.,
                 le={
                     'th_deg':  30.,
                     'phi_deg': 0.,
                     },
                 NBPHOTONS=1e6, progress=False)
    assert res['I_up (TOA)'][:,:] > 0

def test_ozone_lut():
    '''
    Ozone Gaseous transmission for MERIS
    '''
    from itertools import product

    list_TCO = [350., 400., 450.]   # ozone column in DU
    list_AOT = [0.05, 0.3]     # aerosol optical thickness

    luts = []
    for TCO, AOT in product(list_TCO, list_AOT):

        aer = AeroOPAC('maritime_clean', AOT, 550.)
        pro = AtmAFGL('afglms', comp=[aer], O3=TCO)

        m = Smartg().run(wl=490., atm=pro, NBTHETA=50, NBPHOTONS=1e6, progress=False)

        m.set_attrs({'TCO':TCO, 'AOT': AOT})
        luts.append(m)

    merged = merge(luts, ['TCO', 'AOT'])
    merged.print_info()


def test_multispectral():

    atm = AtmAFGL('afglt',
        comp=[AeroOPAC('maritime_clean', 0.3, 550., )],
        grid='100[75]25[5]10[1]0', # optional, otherwise use default grid
        pfgrid=[100, 20, 0.],   # optional, otherwise use a single band 100-0
        pfwav=[400, 500, 600],  # optional, otherwise phase functions are calculated at all bands
        )

    return Smartg().run(np.linspace(400, 600, 10),
                        THVDEG=60.,
                        atm=atm,
                        surf=RoughSurface(),
                        water=IOP_1(1.),
                        progress=False)

