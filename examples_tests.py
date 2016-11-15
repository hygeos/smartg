#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
SMART-G examples
'''

from smartg import Smartg, LambSurface, RoughSurface
from atmosphere import AtmAFGL, AeroOPAC
import numpy as np
from nose.tools import nottest


def test_rayleigh():
    '''
    Basic Rayleigh example
    '''
    Smartg(debug=True).run(wl=400., NBPHOTONS=1e6, atm=AtmAFGL('afglt'), progress=False)

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
    return Smartg().run(490., NBPHOTONS=1e6,
                  atm=AtmAFGL('afglms'),
                  surf=LambSurface(ALB=0.1), progress=False)


@nottest
def test_atm_surf_ocean():
    return Smartg().run(490., NBPHOTONS=1e6,
                  atm=AtmAFGL('afglms',
                              aer=AeroOPAC('maritime_clean', 0.2, 550)),
                  surf=RoughSurface(),
                  NBTHETA=30,
                  water=IOP_MM(chl=1., NANG=1000), progress=False)


@nottest
def test_surf_ocean():
    return Smartg().run(490., THVDEG=30., NBPHOTONS=1e6,
                  surf=RoughSurface(),
                  water=IOP_MM(1., pfwav=[400.]), progress=False)


@nottest
def test_ocean():
    return Smartg().run(wl=560., THVDEG=30.,
                  water=IOP_SPM(100.), NBPHOTONS=1e6, progress=False)


# def test_reptran():
#     '''
#     using reptran
#     '''
#     aer = AeroOPAC('maritime_polluted', 0.4, 550.)
#     pro = AtmAFGL('afglms.dat', aer=aer, grid='100[75]25[5]10[1]0')
#     files, ibands = [], []
#     for iband in REPTRAN('reptran_solar_msg').band('msg1_seviri_ch008').ibands():
#         job = smartg('SMART-G-PP', wl=np.mean(iband.band.awvl),
#                 NBPHOTONS=5e8,
#                 iband=iband, atm=pro, progress=False)
#         files.append(job.output)
#         ibands.append(iband)
# 
#     reptran_merge(files, ibands)


@nottest
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
        pro = AtmAFGL('afglms', aer=aer, O3=TCO)

        m = Smartg().run(wl=490., atm=pro, NBTHETA=50, NBPHOTONS=1e6, progress=False)

        m.set_attrs({'TCO':TCO, 'AOT': AOT})
        luts.append(m)

    merged = merge(luts, ['TCO', 'AOT'])
    merged.print_info()

@nottest
def test_multispectral():
    '''
    multispectral processing
    '''

    pro = AtmAFGL('afglt',
        grid='100[75]25[5]10[1]0', # [100, 75, 50, 30, 20, 10, 5, 1, 0.],  # optional, otherwise use default grid
        # pfgrid=[100, 20, 0.],   # optional, otherwise use a single band 100-0
        # pfwav=[400, 500, 600], # optional, otherwise phase functions are calculated at all bands
        aer=AeroOPAC('maritime_clean', 0.3, 550., ),
        verbose=True)

    return Smartg().run(wl = np.linspace(400, 600, 2),
             THVDEG=60.,
             atm=pro,
             surf=RoughSurface(),
             water=IOP_SPM(1.), progress=False)

