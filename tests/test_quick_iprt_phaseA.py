#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from smartg.smartg import Smartg, Sensor, LambSurface, Albedo_cst
from smartg.atmosphere import AtmAFGL
import pandas as pd
import numpy as np
import glob

from smartg.iprt import convert_SGout_to_IPRTout, select_and_plot_polar_iprt, compute_deltam

import os
from . import conftest
os.environ['PATH'] += ':/usr/local/cuda/bin'



# Global variable(s)
SEED = 1e8
import subprocess
RGP = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8') # root path from smartg git project


@pytest.fixture(scope='module')
def S1DF():
    '''
    Backward compilation in 1D
    '''
    return Smartg(alt_pp=True, back=False, double=True, bias=True)


def test_A2(request, S1DF):
    mol_sca = np.array([0., 0.1])[None,:]
    mol_abs = np.array([0., 0.])[None,:]
    z       = np.array([1., 0.])
    atm     = AtmAFGL('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs).calc(550.)
    surf    = LambSurface(ALB=Albedo_cst(0.3))

    VZAMIN = 100.
    VZAMAX = 180.
    VZAINC = 5.
    VZA = np.arange(VZAMIN, VZAMAX+VZAINC, VZAINC)

    VAAMIN = 0.
    VAAMAX = 180.
    VAAINC = 5.
    VAA = np.arange(VAAMIN, VAAMAX+VAAINC, VAAINC)

    # SMART-G Forward TH and PHI using local estimate (anticlockwise) conversion with VZA and VAA MYSTIC (clockwise)
    TH  = 180.-VZA
    PHI = -VAA
    TH[TH==0] = 1e-6  # avoid problem due to special case of 0
    le     = {'th_deg':TH, 'phi_deg':PHI}

    SZA = 50.
    SAA = 0.
    PHI_0 = 180.-SAA # SMART-G anticlockwise converted to be consistent with MYSTIC

    mA2F = S1DF.run(THVDEG=SZA, PHVDEG=PHI_0, wl=550., NBPHOTONS=1e7, NBLOOP=1e6, atm=atm, OUTPUT_LAYERS=int(1),
                    le=le, surf=surf, XBLOCK = 64, XGRID = 1024, BEER=1, DEPO=0.03, stdev=True, SEED=SEED)

    # convert (Forward, U must be multiplied by -1)
    tmp_file_a2 = RGP + "/tests/tmp_phaseA_a2.dat"
    convert_SGout_to_IPRTout(lm=[mA2F, mA2F], lU_sign=[-1, -1], case_name="A2", depol=0.03, lalt=[0., 1.], SZA=50., SAA=0., lVZA=[180.-VZA, VZA],
                             lVAA=[VAA, VAA], file_name= tmp_file_a2, output_layer=['_down (0+)', '_up (TOA)'])
    
    smartg_a2 = pd.read_csv(RGP + "/tests/tmp_phaseA_a2.dat", header=None, sep=r'\s+', dtype=float, comment="#").values
    mystic_a2 = pd.read_csv(RGP + "/tests/IPRT_data/phaseA/iprt_case_a2_mystic.dat", header=None, sep=r'\s+', dtype=float, comment="#").values
    avoidP=False
    # ============ 0km of altitude
    title = "IPRT case A2 - depol = 0.03 - 0km - SMARTG"
    I_smartg_0km, Q_smartg_0km, U_smartg_0km, V_smartg_0km = select_and_plot_polar_iprt(smartg_a2, 0., title=title, change_U_sign=True, sym=True, outputIQUV=True, avoid_plot=avoidP)
    conftest.savefig(request, bbox_inches='tight')

    title = "IPRT case A2 - depol = 0.03 - 0km - MYSTIC"
    I_mystic_0km, Q_mystic_0km, U_mystic_0km, V_mystic_0km = select_and_plot_polar_iprt(mystic_a2, 0., title=title, change_U_sign=True, sym=True, outputIQUV=True, avoid_plot=avoidP)
    conftest.savefig(request, bbox_inches='tight')

    I_val = I_mystic_0km-I_smartg_0km
    Q_val = Q_mystic_0km-Q_smartg_0km
    U_val = U_mystic_0km-U_smartg_0km
    V_val = V_mystic_0km-V_smartg_0km
    maxI = max(np.abs(np.min(I_val)), np.abs(np.max(I_val)))
    maxQ = max(np.abs(np.min(Q_val)), np.abs(np.max(Q_val)))
    maxU = max(np.abs(np.min(U_val)), np.abs(np.max(U_val)))
    maxV = max(np.abs(np.min(V_val)), np.abs(np.max(V_val)))
    title = "IPRT case A2 - depol = 0.03 - 0km - dif (MYSTIC-SMARTG)"
    select_and_plot_polar_iprt(mystic_a2, 0., title=title, forceIQUV=[I_val, Q_val, U_val, V_val], maxI=maxI, maxQ=maxQ, maxU=maxU, maxV=maxV, cmapI='RdBu_r', avoid_plot=avoidP)
    conftest.savefig(request, bbox_inches='tight')

    # ============ 1km of altitude (TOA)
    title = "IPRT case A2 - depol = 0.03 - 1km - SMARTG"
    I_smartg_1km, Q_smartg_1km, U_smartg_1km, V_smartg_1km = select_and_plot_polar_iprt(smartg_a2, 1., title=title, change_U_sign=True, inv_thetas=True,sym=True, outputIQUV=True, avoid_plot=avoidP)
    conftest.savefig(request, bbox_inches='tight')

    title = "IPRT case A2 - depol = 0.03 - 1km - MYSTIC"
    I_mystic_1km, Q_mystic_1km, U_mystic_1km, V_mystic_1km = select_and_plot_polar_iprt(mystic_a2, 1., title=title, change_U_sign=True, inv_thetas=True, sym=True, outputIQUV=True, avoid_plot=avoidP)
    conftest.savefig(request, bbox_inches='tight')

    I_val = I_mystic_1km-I_smartg_1km
    Q_val = Q_mystic_1km-Q_smartg_1km
    U_val = U_mystic_1km-U_smartg_1km
    V_val = V_mystic_1km-V_smartg_1km
    maxI = max(np.abs(np.min(I_val)), np.abs(np.max(I_val)))
    maxQ = max(np.abs(np.min(Q_val)), np.abs(np.max(Q_val)))
    maxU = max(np.abs(np.min(U_val)), np.abs(np.max(U_val)))
    maxV = max(np.abs(np.min(V_val)), np.abs(np.max(V_val)))
    title = "IPRT case A2 - depol = 0.03 - 1km - dif (MYSTIC-SMARTG)"
    select_and_plot_polar_iprt(mystic_a2, 1., title=title, forceIQUV=[I_val, Q_val, U_val, V_val], maxI=maxI, maxQ=maxQ, maxU=maxU, maxV=maxV, cmapI='RdBu_r', avoid_plot=avoidP)
    conftest.savefig(request, bbox_inches='tight')

    # MYSTIC total IQUV
    I_mystic = np.concatenate((I_mystic_0km, I_mystic_1km), axis=0)
    Q_mystic = np.concatenate((Q_mystic_0km, Q_mystic_1km), axis=0)
    U_mystic = np.concatenate((U_mystic_0km, U_mystic_1km), axis=0)
    V_mystic = np.concatenate((V_mystic_0km, V_mystic_1km), axis=0)

    # SMARTG ref results
    smartg_a2_ref = pd.read_csv(RGP + "/tests/IPRT_data/phaseA/smartg_ref_res/iprt_case_a2_smartg_ref.dat", header=None, sep=r'\s+', dtype=float, comment="#").values
    I_smartg_0km_ref, Q_smartg_0km_ref, U_smartg_0km_ref, V_smartg_0km_ref = select_and_plot_polar_iprt(smartg_a2_ref, 0., change_U_sign=True, outputIQUV=True, avoid_plot=True)
    I_smartg_1km_ref, Q_smartg_1km_ref, U_smartg_1km_ref, V_smartg_1km_ref = select_and_plot_polar_iprt(smartg_a2_ref, 1., change_U_sign=True, inv_thetas=True, outputIQUV=True, avoid_plot=True)
    I_smartg_ref = np.concatenate((I_smartg_0km_ref, I_smartg_1km_ref), axis=0)
    Q_smartg_ref = np.concatenate((Q_smartg_0km_ref, Q_smartg_1km_ref), axis=0)
    U_smartg_ref = np.concatenate((U_smartg_0km_ref, U_smartg_1km_ref), axis=0)
    V_smartg_ref = np.concatenate((V_smartg_0km_ref, V_smartg_1km_ref), axis=0)

    # SMARTG results calculated during the test
    I_smartg = np.concatenate((I_smartg_0km, I_smartg_1km), axis=0)
    Q_smartg = np.concatenate((Q_smartg_0km, Q_smartg_1km), axis=0)
    U_smartg = np.concatenate((U_smartg_0km, U_smartg_1km), axis=0)
    V_smartg = np.concatenate((V_smartg_0km, V_smartg_1km), axis=0)

    # Compute the delta_m values from the ref smartg results
    delta_m_ref = compute_deltam(obs=[I_mystic, Q_mystic, U_mystic, V_mystic], mod=[I_smartg_ref, Q_smartg_ref, U_smartg_ref, V_smartg_ref], print_res=False)

    # Compute the delta_m values from the smartg test results
    delta_m = compute_deltam(obs=[I_mystic, Q_mystic, U_mystic, V_mystic], mod=[I_smartg, Q_smartg, U_smartg, V_smartg], print_res=False)

    # Max diff tolerated between ref values and test values (note: even if same SEED is used there is still a sligh difference due to atomicAdd functions in CUDA)
    errPercent=5
    maxDiffIQUV = np.abs(delta_m_ref*errPercent/100)

    # Check if the the test is ok by comparing smartg ref and smartg test
    assert not ( (delta_m[0] > delta_m_ref[0]+maxDiffIQUV[0]) or (delta_m[0] < delta_m_ref[0]-maxDiffIQUV[0]) ), f'Problem with I values, get {delta_m[0]:.5f} instead of {delta_m_ref[0]:.5f} +- {errPercent:.0f} per cent'
    assert not ( (delta_m[1] > delta_m_ref[1]+maxDiffIQUV[1]) or (delta_m[1] < delta_m_ref[1]-maxDiffIQUV[1]) ), f'Problem with Q values, get {delta_m[1]:.5f} instead of {delta_m_ref[1]:.5f} +- {errPercent:.0f} per cent'
    assert not ( (delta_m[2] > delta_m_ref[2]+maxDiffIQUV[2]) or (delta_m[2] < delta_m_ref[2]-maxDiffIQUV[2]) ), f'Problem with U values, get {delta_m[2]:.5f} instead of {delta_m_ref[2]:.5f} +- {errPercent:.0f} per cent'
    assert not ( (delta_m[3] > delta_m_ref[3]+maxDiffIQUV[3]) or (delta_m[3] < delta_m_ref[3]-maxDiffIQUV[3]) ), f'Problem with V values, get {delta_m[3]:.5f} instead of {delta_m_ref[3]:.5f} +- {errPercent:.0f} per cent'


# Remove tmp files
tmpFiles = glob.glob(RGP + "/tests/tmp_phaseA*")
for f in tmpFiles: os.remove(f)