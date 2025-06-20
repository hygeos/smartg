#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from smartg.atmosphere import AtmAFGL, AerOPAC, Cloud
import numpy as np

try:
    from smartg.tools.third_party_utils import change_altitude_grid
except ModuleNotFoundError:
    exist_third_party_utils = False
    pass
else:
    exist_third_party_utils = True
'''
Test profile calculation
'''

@pytest.fixture(params=[
            500.,
            np.array(500.),
            [400.],
            [500., 600.],
            np.linspace(400, 800, 4),
            ])
def wav(request):
    return request.param

def test_profile1(wav):
    atm = AtmAFGL('afglt')
    atm.calc(wav)

def test_profile2(wav):
    atm = AtmAFGL('afglms',
                  grid=[100., 50., 20., 10., 5., 2., 1., 0.],
                  pfgrid=[100., 10., 0.])
    atm.calc(wav)

@pytest.mark.skipif(exist_third_party_utils == False, reason="require third_party_utils external file")
def test_profile3(wav):
    atm = AtmAFGL('afglms',
                comp=[AerOPAC('desert', 0.1, 550.)],
                grid='100[20]10[1]0',
                pfgrid=[100., 10., 0.])
    atm.calc(wav)

def test_profile4(wav):
    atm = AtmAFGL('afglms',
                  comp=[AerOPAC('desert', 0.1, 550.),
                        Cloud('wc', 12.68, 2, 3, 10., 550.),
                       ],
                  grid=[100., 50., 20., 10., 5., 2., 1., 0.],
                  pfgrid=[100., 10., 0.])
    atm.calc(wav)

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
            comp=[AerOPAC('urban', 0.1, 550., ssa=0.8)]
            ).calc(400.)

    AtmAFGL('afglms', grid=[100, 20, 0.],
            comp=[AerOPAC('urban', 0.1, 550., ssa=0.8)]
            ).calc([400., 500., 600.])

    AtmAFGL('afglms', grid=[100, 20, 0.],
            comp=[AerOPAC('urban', 0.1, 550., ssa=[0.76, 0.77, 0.78])]
            ).calc([400., 500., 600.])

