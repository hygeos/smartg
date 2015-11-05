#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
SMART-G examples
'''

from smartg import smartg, Profile, AeroOPAC, LambSurface, RoughSurface
from smartg import IOP_MM, IOP_SPM, REPTRAN, reptran_merge, merge
import numpy as np


def test_rayleigh():
    '''
    Basic Rayleigh example
    '''
    smartg(wl=400., NBPHOTONS=1e7, atm=Profile('afglt'), progress=False)


def test_rayleigh_grid():
    '''
    Use a custom atmosphere grid
    '''
    smartg(wl=500., NBPHOTONS=1e7,
           atm=Profile('afglt', grid='100[75]25[5]10[1]0'), progress=False)


def test_aerosols():
    '''
    test with aerosols
    '''
    aer = AeroOPAC('maritime_clean', 0.4, 550.)
    pro = Profile('afglms', aer=aer)
    smartg(wl=490., atm=pro, NBPHOTONS=1e9, progress=False)


def test_atm_surf():
    '''
    atmosphere + lambertian surface of albedo 10%
    '''
    return smartg(490., NBPHOTONS=1e7,
                  atm=Profile('afglms'),
                  surf=LambSurface(ALB=0.1), progress=False)


def test_atm_surf_ocean():
    return smartg(490., NBPHOTONS=1e7,
                  atm=Profile('afglms',
                              aer=AeroOPAC('maritime_clean', 0.2, 550)),
                  surf=RoughSurface(),
                  NBTHETA=30,
                  water=IOP_MM(chl=1., NANG=1000), progress=False)


def test_surf_ocean():
    return smartg(490., THVDEG=30., NBPHOTONS=2e6,
                  surf=RoughSurface(),
                  water=IOP_MM(1., pfwav=[400.]), progress=False)


def test_ocean():
    return smartg(wl=560., THVDEG=30.,
                  water=IOP_SPM(100.), NBPHOTONS=5e6, progress=False)


def test_reptran():
    '''
    using reptran
    '''
    aer = AeroOPAC('maritime_polluted', 0.4, 550.)
    pro = Profile('afglms.dat', aer=aer, grid='100[75]25[5]10[1]0')
    files, ibands = [], []
    for iband in REPTRAN('reptran_solar_msg').band('msg1_seviri_ch008').ibands():
        job = smartg('SMART-G-PP', wl=np.mean(iband.band.awvl),
                NBPHOTONS=5e8,
                iband=iband, atm=pro, progress=False)
        files.append(job.output)
        ibands.append(iband)

    reptran_merge(files, ibands)


def test_ozone_lut():
    '''
    Ozone Gaseous transmission for MERIS
    '''
    from itertools import product

    list_TCO = [350., 400., 450.]   # ozone column in DU
    list_AOT = [0.05, 0.1, 0.4]     # aerosol optical thickness

    luts = []
    for TCO, AOT in product(list_TCO, list_AOT):

        aer = AeroOPAC('maritime_clean', AOT, 550.)
        pro = Profile('afglms', aer=aer, O3=TCO)

        m = smartg(wl=490., atm=pro, NBTHETA=50, NBPHOTONS=5e6, progress=False)

        m.set_attrs({'TCO':TCO, 'AOT': AOT})
        luts.append(m)

    merged = merge(luts, ['TCO', 'AOT'])
    merged.print_info()

def test_multispectral():
    '''
    process multiple bands at once
    '''

    pro = Profile('afglt',
        grid='100[75]25[5]10[1]0', # [100, 75, 50, 30, 20, 10, 5, 1, 0.],  # optional, otherwise use default grid
        # pfgrid=[100, 20, 0.],   # optional, otherwise use a single band 100-0
        # pfwav=[400, 500, 600], # optional, otherwise phase functions are calculated at all bands
        aer=AeroOPAC('maritime_clean', 0.3, 550., ),
        verbose=True)

    return smartg(wl = np.linspace(400, 600, 2),
             THVDEG=60.,
             atm=pro,
             surf=RoughSurface(),
             water=IOP_SPM(1.), progress=False)

def test_sp():
    '''
    Basic test in spherical
    '''
    smartg(pp=False, wl=400., NBPHOTONS=1e8, atm=Profile('afglt'), progress=False)

def test_pp_sp():
    '''
    Check consistency between PP and SP for large RTER
    '''
    M = []
    for pp in [True, False]:
        M.append(smartg(pp=pp, wl=400., NBPHOTONS=1e8, atm=Profile('afglt'), progress=False))
    mpp = M[0]
    msp = M[1]

    rdiff = ((mpp - msp)/msp)['I_up (TOA)']
    print np.mean(rdiff[:,:])
    assert False

def test_pp_sp2():
    '''
    check consistency of PP and SP without atmosphere
    '''
    # TODO: with or without ocean
    assert False

