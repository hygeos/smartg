#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Tested with the following GPUs: 3090
import pytest

from smartg.smartg import Smartg, Sensor, LambSurface, Albedo_cst
from smartg.atmosphere import AtmAFGL
import pandas as pd
import numpy as np
import glob

from smartg.iprt import convert_SGout_to_IPRTout, select_and_plot_polar_iprt, compute_deltam, seclect_iprt_IQUV, plot_iprt_radiances
from smartg.libATM3D import read_cld_nth_cte
from smartg.tools.phase import calc_iphase
from luts.luts import LUT

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
    return Smartg(alt_pp=True, back=False, double=True, bias=True, device=1)


def test_A2(request, S1DF):
    # === Atmosphere profil
    mol_sca = np.array([0., 0.1])[None,:]
    mol_abs = np.array([0., 0.])[None,:]
    z       = np.array([1., 0.])
    atm     = AtmAFGL('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs).calc(550.)
    surf    = LambSurface(ALB=Albedo_cst(0.3))

    # === Illumination conditions
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

    # === Simulation
    mA2F = S1DF.run(THVDEG=SZA, PHVDEG=PHI_0, wl=550., NBPHOTONS=1e7, NBLOOP=1e6, atm=atm, OUTPUT_LAYERS=int(1),
                    le=le, surf=surf, XBLOCK = 64, XGRID = 1024, BEER=1, DEPO=0.03, stdev=True, SEED=SEED)

    # === Convert smartg output to iprt ascii output format (Forward, U must be multiplied by -1)
    tmp_file_a2 = RGP + "/tests/tmp_phaseA_a2.dat"
    convert_SGout_to_IPRTout(lm=[mA2F, mA2F], lU_sign=[-1, -1], case_name="A2", depol=0.03, lalt=[0., 1.], SZA=50., SAA=0., lVZA=[180.-VZA, VZA],
                             lVAA=[VAA, VAA], file_name= tmp_file_a2, output_layer=['_down (0+)', '_up (TOA)'])
    
    # === Plot and comparison with MYSTIC (to save in the report)
    smartg_a2 = pd.read_csv(tmp_file_a2, header=None, sep=r'\s+', dtype=float, comment="#").values
    mystic_a2 = pd.read_csv(RGP + "/tests/IPRT_data/phaseA/iprt_case_a2_mystic.dat", header=None, sep=r'\s+', dtype=float, comment="#").values
    avoidP=False
    # 0km of altitude
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

    # 1km of altitude (TOA)
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

    # === Compute the delta_m values and analyse them with the previous saved validated ones
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
    errPercent=10
    maxDiffIQUV = np.abs(delta_m_ref*errPercent/100)

    # Check if the the test is ok by comparing smartg ref and smartg test
    assert not ( (delta_m[0] > delta_m_ref[0]+maxDiffIQUV[0]) or (delta_m[0] < delta_m_ref[0]-maxDiffIQUV[0]) ), f'Problem with I values, get {delta_m[0]:.5f} instead of {delta_m_ref[0]:.5f} +- {errPercent:.0f} per cent'
    assert not ( (delta_m[1] > delta_m_ref[1]+maxDiffIQUV[1]) or (delta_m[1] < delta_m_ref[1]-maxDiffIQUV[1]) ), f'Problem with Q values, get {delta_m[1]:.5f} instead of {delta_m_ref[1]:.5f} +- {errPercent:.0f} per cent'
    assert not ( (delta_m[2] > delta_m_ref[2]+maxDiffIQUV[2]) or (delta_m[2] < delta_m_ref[2]-maxDiffIQUV[2]) ), f'Problem with U values, get {delta_m[2]:.5f} instead of {delta_m_ref[2]:.5f} +- {errPercent:.0f} per cent'
    assert not ( (delta_m[3] > delta_m_ref[3]+maxDiffIQUV[3]) or (delta_m[3] < delta_m_ref[3]-maxDiffIQUV[3]) ), f'Problem with V values, get {delta_m[3]:.5f} instead of {delta_m_ref[3]:.5f} +- {errPercent:.0f} per cent'

def test_A5_pp(request, S1DF):
    # === Atmosphere profil
    z = np.array([1., 0.])
    mol_sca = np.array([0., 0.])[None,:]
    mol_abs= np.array([0., 0.])[None,:]
    cld_tau_ext = np.full_like(mol_sca, 5., dtype=np.float32)
    cld_tau_ext[:,0] = 0. # dtau TOA equal to 0
    cld_ssa = np.full_like(mol_sca, 0.999979, dtype=np.float32)
    prof_aer = (cld_tau_ext, cld_ssa)
    NTH = 18001  # The water cloud has a phase function with a non-negligible peak, then a sufficiently fine resolution is required.
    file_cld_phase = RGP + "/tests/IPRT_data/phaseA/watercloud.mie.cdf"
    cld_phase = read_cld_nth_cte(filename=file_cld_phase, nb_theta=NTH)
    pha_atm, ipha_atm = calc_iphase(cld_phase, np.array([800.]), z)
    lpha_lut = []
    for i in range (0, pha_atm.shape[0]): lpha_lut.append(LUT(pha_atm[i,:,:], axes=[None, np.linspace(0, 180, NTH)], names=['stk', 'theta_atm'])) 
    atm = AtmAFGL('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs, prof_aer=prof_aer, prof_phases=(ipha_atm, lpha_lut))
    pro = atm.calc(800., phase=False)
    surf  = None

    # === Illumination conditions
    SZA = 50.
    SAA = 0.
    PHI_0 = 180.-SAA # SMART-G anticlockwise converted to be consistent with MYSTIC

    VZAMIN = 100.
    VZAMAX = 180.
    VZAINC = 1.
    VZA = np.arange(VZAMIN, VZAMAX+VZAINC, VZAINC)

    VAA = np.array([0., 180.])

    # SMART-G Forward TH and PHI using local estimate (anticlockwise) conversion with VZA and VAA MYSTIC (clockwise)
    TH  = 180.-VZA
    PHI = -VAA
    TH[TH==0] = 1e-6 # avoid problem due to special case of 0
    le     = {'th_deg':TH, 'phi_deg':PHI}#, 'zip':True}

    # === Simulation
    mA5F_pp = S1DF.run(THVDEG=SZA, PHVDEG=PHI_0, wl=800., NBPHOTONS=1e7, NBLOOP=1e6, NF=NTH,atm=pro, OUTPUT_LAYERS=int(1),
                       le=le, surf=surf, XBLOCK = 64, XGRID = 1024, BEER=1, DEPO=0.03, stdev=True, SEED=SEED)
    
    # === Convert smartg output to iprt ascii output format 
    tmp_file_a5_pp = RGP + "/tests/tmp_phaseA_a5_pp.dat"
    convert_SGout_to_IPRTout(lm=[mA5F_pp, mA5F_pp], lU_sign=[-1, -1], case_name="A5", depol=0.03, lalt=[0., 1.], SZA=50., SAA=0., lVZA=[VZA, VZA],
                             lVAA=[VAA, VAA], file_name=tmp_file_a5_pp, output_layer=['_down (0+)', '_up (TOA)'])
    
    # === Plot and comparison with MYSTIC (to save in the report)
    smartg_a5_pp = pd.read_csv(tmp_file_a5_pp, header=None, sep=r'\s+', dtype=float, comment="#").values
    mystic_a5_pp = pd.read_csv(RGP + "/tests/IPRT_data/phaseA/iprt_case_a5_pp_mystic.dat", header=None, sep=r'\s+', dtype=float, comment="#").values
    VZAn = np.sort(np.concatenate((VZA-180, 180-VZA)))
    NVZA = len(VZAn)

    # Reflectance
    IQUVS_with_std = seclect_iprt_IQUV(smartg_a5_pp, 1., change_U_sign=False, inv_thetas=True, stdev=True)
    IQUVM_with_std = seclect_iprt_IQUV(mystic_a5_pp, 1., change_U_sign=False, inv_thetas=True, I_index=5, va_index=3, phi_index=4, z_index=0, stdev=True)

    # MYSTIC IQUV and stdev IQUV
    IQUVM_pp = np.zeros((4,NVZA), dtype=np.float32)
    IQUVstdM_pp = np.zeros((4,NVZA), dtype=np.float32)
    IQUVS_pp = np.zeros((4,NVZA), dtype=np.float32)
    IQUVstdS_pp = np.zeros((4,NVZA), dtype=np.float32)
    for i in range (0, 4):
        IQUVM_pp[i,:]=np.concatenate((IQUVM_with_std[i][:,1], IQUVM_with_std[i][::-1,0]))
        IQUVstdM_pp[i,:]=np.concatenate((IQUVM_with_std[i+4][:,1], IQUVM_with_std[i+4][::-1,0]))
        IQUVS_pp[i,:]=np.concatenate((IQUVS_with_std[i][:,1], IQUVS_with_std[i][::-1,0]))
        IQUVstdS_pp[i,:]=np.concatenate((IQUVS_with_std[i+4][:,1], IQUVS_with_std[i+4][::-1,0]))

    IQUVS_pp_tot = IQUVS_pp.copy()
    IQUVM_pp_tot = IQUVM_pp.copy()

    IQUVyMin = [0., -2e-2, -1.2e-4, -1e-5]
    IQUVyMax = [2.5e-1, 1.5e-2, 6e-5, 1e-5]
    plot_iprt_radiances(IQUV_obs=IQUVM_pp, IQUV_mod=IQUVS_pp, IQUVstd_obs=IQUVstdM_pp, IQUVstd_mod=IQUVstdS_pp, xaxis=VZAn,
                        xlabel= 'VZA [deg]', IQUVyMin=IQUVyMin, IQUVyMax=IQUVyMax, title='reflectance  MYSTIC-red SMARTG-blue')
    conftest.savefig(request, bbox_inches='tight')

    # Transmittance
    IQUVS_with_std = seclect_iprt_IQUV(smartg_a5_pp, 0., change_U_sign=False, inv_thetas=True, stdev=True)
    IQUVM_with_std = seclect_iprt_IQUV(mystic_a5_pp, 0., change_U_sign=False, inv_thetas=True, I_index=5, va_index=3, phi_index=4, z_index=0, stdev=True)

    # MYSTIC IQUV and stdev IQUV
    for i in range (0, 4):
        IQUVM_pp[i,:]=np.concatenate((IQUVM_with_std[i][:,1], IQUVM_with_std[i][::-1,0]))
        IQUVstdM_pp[i,:]=np.concatenate((IQUVM_with_std[i+4][:,1], IQUVM_with_std[i+4][::-1,0]))
        IQUVS_pp[i,:]=np.concatenate((IQUVS_with_std[i][:,1], IQUVS_with_std[i][::-1,0]))
        IQUVstdS_pp[i,:]=np.concatenate((IQUVS_with_std[i+4][:,1], IQUVS_with_std[i+4][::-1,0]))

    IQUVyMin = [0., -3e-3, -1.5e-4, -2e-5]
    IQUVyMax = [3.5, 4e-3, 2e-4, 3e-5]

    plot_iprt_radiances(IQUV_obs=IQUVM_pp, IQUV_mod=IQUVS_pp, IQUVstd_obs=IQUVstdM_pp, IQUVstd_mod=IQUVstdS_pp, xaxis=VZAn,
                        xlabel= 'VZA [deg]', IQUVyMin=IQUVyMin, IQUVyMax=IQUVyMax, title='transmittance  MYSTIC-red SMARTG-blue')
    conftest.savefig(request, bbox_inches='tight')

    IQUVS_pp_tot = np.concatenate((IQUVS_pp_tot, IQUVS_pp))
    IQUVM_pp_tot = np.concatenate((IQUVM_pp_tot, IQUVM_pp))

    # === Compute the delta_m values and analyse them with the previous saved validated ones
    # SMARTG ref results
    smartg_a5_pp_ref = pd.read_csv(RGP + "/tests/IPRT_data/phaseA/smartg_ref_res/iprt_case_a5_smartg_pp_ref.dat", header=None, sep=r'\s+', dtype=float, comment="#").values
    IQUVS_with_std_ref = seclect_iprt_IQUV(smartg_a5_pp_ref, 1., change_U_sign=False, inv_thetas=True, stdev=True)
    IQUVS_pp_ref = np.zeros((4,NVZA), dtype=np.float32)
    for i in range (0, 4): IQUVS_pp_ref[i,:]=np.concatenate((IQUVS_with_std_ref[i][:,1], IQUVS_with_std_ref[i][::-1,0]))
    IQUVS_pp_ref_tot = IQUVS_pp_ref.copy()
    IQUVS_with_std_ref = seclect_iprt_IQUV(smartg_a5_pp_ref, 0., change_U_sign=False, inv_thetas=True, stdev=True)
    for i in range (0, 4): IQUVS_pp_ref[i,:]=np.concatenate((IQUVS_with_std_ref[i][:,1], IQUVS_with_std_ref[i][::-1,0]))
    IQUVS_pp_ref_tot = np.concatenate((IQUVS_pp_ref_tot, IQUVS_pp_ref))

    # Compute the delta_m values from the ref smartg results
    delta_m_ref = compute_deltam(obs=IQUVM_pp_tot, mod=IQUVS_pp_ref_tot, print_res=False)

    # Compute the delta_m values from the smartg test results
    delta_m = compute_deltam(obs=IQUVM_pp_tot, mod=IQUVS_pp_tot, print_res=False)

    # Max diff tolerated between ref values and test values (note: even if same SEED is used there is still a sligh difference due to atomicAdd functions in CUDA)
    errPercent=10
    maxDiffIQUV = np.abs(delta_m_ref*errPercent/100)

    # Check if the the test is ok by comparing smartg ref and smartg test
    assert not ( (delta_m[0] > delta_m_ref[0]+maxDiffIQUV[0]) or (delta_m[0] < delta_m_ref[0]-maxDiffIQUV[0]) ), f'Problem with I values, get {delta_m[0]:.5f} instead of {delta_m_ref[0]:.5f} +- {errPercent:.0f} per cent'
    assert not ( (delta_m[1] > delta_m_ref[1]+maxDiffIQUV[1]) or (delta_m[1] < delta_m_ref[1]-maxDiffIQUV[1]) ), f'Problem with Q values, get {delta_m[1]:.5f} instead of {delta_m_ref[1]:.5f} +- {errPercent:.0f} per cent'
    assert not ( (delta_m[2] > delta_m_ref[2]+maxDiffIQUV[2]) or (delta_m[2] < delta_m_ref[2]-maxDiffIQUV[2]) ), f'Problem with U values, get {delta_m[2]:.5f} instead of {delta_m_ref[2]:.5f} +- {errPercent:.0f} per cent'
    assert not ( (delta_m[3] > delta_m_ref[3]+maxDiffIQUV[3]) or (delta_m[3] < delta_m_ref[3]-maxDiffIQUV[3]) ), f'Problem with V values, get {delta_m[3]:.5f} instead of {delta_m_ref[3]:.5f} +- {errPercent:.0f} per cent'


def test_A5_al(request, S1DF):
    # === Atmosphere profil
    z = np.array([1., 0.])
    mol_sca = np.array([0., 0.])[None,:]
    mol_abs= np.array([0., 0.])[None,:]
    cld_tau_ext = np.full_like(mol_sca, 5., dtype=np.float32)
    cld_tau_ext[:,0] = 0. # dtau TOA equal to 0
    cld_ssa = np.full_like(mol_sca, 0.999979, dtype=np.float32)
    prof_aer = (cld_tau_ext, cld_ssa)
    NTH = 18001  # The water cloud has a phase function with a non-negligible peak, then a sufficiently fine resolution is required.
    file_cld_phase = RGP + "/tests/IPRT_data/phaseA/watercloud.mie.cdf"
    cld_phase = read_cld_nth_cte(filename=file_cld_phase, nb_theta=NTH)
    pha_atm, ipha_atm = calc_iphase(cld_phase, np.array([800.]), z)
    lpha_lut = []
    for i in range (0, pha_atm.shape[0]): lpha_lut.append(LUT(pha_atm[i,:,:], axes=[None, np.linspace(0, 180, NTH)], names=['stk', 'theta_atm'])) 
    atm = AtmAFGL('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs, prof_aer=prof_aer, prof_phases=(ipha_atm, lpha_lut))
    pro = atm.calc(800., phase=False)
    surf  = None

    # === Illumination conditions
    SZA = 50.
    SAA = 0.
    PHI_0 = 180.-SAA # SMART-G anticlockwise converted to be consistent with MYSTIC

    VZA = np.array([130.])

    VAAMIN = 0.
    VAAMAX = 180.
    VAAINC = 1.
    VAA = np.arange(VAAMIN, VAAMAX+VAAINC, VAAINC)

    # SMART-G Forward TH and PHI using local estimate (anticlockwise) conversion with VZA and VAA MYSTIC (clockwise)
    TH  = 180.-VZA
    PHI = -VAA
    TH[TH==0] = 1e-6 # avoid problem due to special case of 0
    le     = {'th_deg':TH, 'phi_deg':PHI}

    # === Simulation
    mA5F_al = S1DF.run(THVDEG=SZA, PHVDEG=PHI_0, wl=800., NBPHOTONS=1e7, NBLOOP=1e6, NF=NTH,atm=pro, OUTPUT_LAYERS=int(1),
                       le=le, surf=surf, XBLOCK = 64, XGRID = 1024, BEER=1, DEPO=0.03, stdev=True, SEED=SEED)
    
    # === Convert smartg output to iprt ascii output format 
    tmp_file_a5_al = RGP + "/tests/tmp_phaseA_a5_al.dat"
    convert_SGout_to_IPRTout(lm=[mA5F_al, mA5F_al], lU_sign=[-1, -1], case_name="A5", depol=0.03, lalt=[0., 1.], SZA=50., SAA=0., lVZA=[VZA, VZA],
                             lVAA=[VAA, VAA], file_name=tmp_file_a5_al, output_layer=['_down (0+)', '_up (TOA)'])
    
    # === Plot and comparison with MYSTIC (to save in the report)
    smartg_a5_al = pd.read_csv(tmp_file_a5_al, header=None, sep=r'\s+', dtype=float, comment="#").values
    mystic_a5_al = pd.read_csv(RGP + "/tests/IPRT_data/phaseA/iprt_case_a5_al_mystic.dat", header=None, sep=r'\s+', dtype=float, comment="#").values
    VAAn = VAA
    NVAA = len(VAAn)

    # Reflectance
    IQUVS_with_std = seclect_iprt_IQUV(smartg_a5_al, 1., change_U_sign=False, inv_thetas=True, stdev=True)
    IQUVM_with_std = seclect_iprt_IQUV(mystic_a5_al, 1., change_U_sign=False, inv_thetas=True, I_index=5, va_index=3, phi_index=4, z_index=0, stdev=True)

    # MYSTIC IQUV and stdev IQUV
    IQUVM_al = np.zeros((4,NVAA), dtype=np.float32)
    IQUVstdM_al = np.zeros((4,NVAA), dtype=np.float32)
    IQUVS_al = np.zeros((4,NVAA), dtype=np.float32)
    IQUVstdS_al = np.zeros((4,NVAA), dtype=np.float32)
    for i in range (0, 4):
        IQUVM_al[i,:]=IQUVM_with_std[i][0,:]
        IQUVstdM_al[i,:]=IQUVM_with_std[i+4][0,:]
        IQUVS_al[i,:]=IQUVS_with_std[i][0,:]
        IQUVstdS_al[i,:]=IQUVS_with_std[i+4][0,:]

    IQUVS_al_tot = IQUVS_al.copy()
    IQUVM_al_tot = IQUVM_al.copy()

    IQUVyMin = [6e-2, -1e-2, -2e-3, -5e-5]
    IQUVyMax = [1.2e-1, 2e-2, 1.2e-2, 2e-5]
    plot_iprt_radiances(IQUV_obs=IQUVM_al, IQUV_mod=IQUVS_al, IQUVstd_obs=IQUVstdM_al, IQUVstd_mod=IQUVstdS_al, xaxis=VAAn,
                        xlabel= 'VZA [deg]', IQUVyMin=IQUVyMin, IQUVyMax=IQUVyMax, title='reflectance  MYSTIC-red SMARTG-blue')
    conftest.savefig(request, bbox_inches='tight')

    # Transmittance
    IQUVS_with_std = seclect_iprt_IQUV(smartg_a5_al, 0., change_U_sign=False, inv_thetas=True, stdev=True)
    IQUVM_with_std = seclect_iprt_IQUV(mystic_a5_al, 0., change_U_sign=False, inv_thetas=True, I_index=5, va_index=3, phi_index=4, z_index=0, stdev=True)

    # MYSTIC IQUV and stdev IQUV
    for i in range (0, 4):
        IQUVM_al[i,:]=IQUVM_with_std[i][0,:]
        IQUVstdM_al[i,:]=IQUVM_with_std[i+4][0,:]
        IQUVS_al[i,:]=IQUVS_with_std[i][0,:]
        IQUVstdS_al[i,:]=IQUVS_with_std[i+4][0,:]

    IQUVyMin = [0., -3.5e-3, -3e-3, -1.5e-5]
    IQUVyMax = [3.5, 5e-4, 5e-4, 2.5e-5]

    plot_iprt_radiances(IQUV_obs=IQUVM_al, IQUV_mod=IQUVS_al, IQUVstd_obs=IQUVstdM_al, IQUVstd_mod=IQUVstdS_al, xaxis=VAAn,
                        xlabel= 'VZA [deg]', IQUVyMin=IQUVyMin, IQUVyMax=IQUVyMax, title='transmittance  MYSTIC-red SMARTG-blue')
    conftest.savefig(request, bbox_inches='tight')

    IQUVS_al_tot = np.concatenate((IQUVS_al_tot, IQUVS_al))
    IQUVM_al_tot = np.concatenate((IQUVM_al_tot, IQUVM_al))

    # === Compute the delta_m values and analyse them with the previous saved validated ones
    # SMARTG ref results
    smartg_a5_al_ref = pd.read_csv(RGP + "/tests/IPRT_data/phaseA/smartg_ref_res/iprt_case_a5_smartg_al_ref.dat", header=None, sep=r'\s+', dtype=float, comment="#").values
    IQUVS_with_std_ref = seclect_iprt_IQUV(smartg_a5_al_ref, 1., change_U_sign=False, inv_thetas=True, stdev=True)
    IQUVS_al_ref = np.zeros((4,NVAA), dtype=np.float32)
    for i in range (0, 4): IQUVS_al_ref[i,:]=IQUVS_with_std_ref[i][0,:]
    IQUVS_al_ref_tot = IQUVS_al_ref.copy()
    IQUVS_with_std_ref = seclect_iprt_IQUV(smartg_a5_al_ref, 0., change_U_sign=False, inv_thetas=True, stdev=True)
    for i in range (0, 4): IQUVS_al_ref[i,:]=IQUVS_with_std_ref[i][0,:]
    IQUVS_al_ref_tot = np.concatenate((IQUVS_al_ref_tot, IQUVS_al_ref))

    # Compute the delta_m values from the ref smartg results
    delta_m_ref = compute_deltam(obs=IQUVM_al_tot, mod=IQUVS_al_ref_tot, print_res=False)

    # Compute the delta_m values from the smartg test results
    delta_m = compute_deltam(obs=IQUVM_al_tot, mod=IQUVS_al_tot, print_res=False)

    # Max diff tolerated between ref values and test values (note: even if same SEED is used there is still a sligh difference due to atomicAdd functions in CUDA)
    errPercent=10
    maxDiffIQUV = np.abs(delta_m_ref*errPercent/100)

    # Check if the the test is ok by comparing smartg ref and smartg test
    assert not ( (delta_m[0] > delta_m_ref[0]+maxDiffIQUV[0]) or (delta_m[0] < delta_m_ref[0]-maxDiffIQUV[0]) ), f'Problem with I values, get {delta_m[0]:.5f} instead of {delta_m_ref[0]:.5f} +- {errPercent:.0f} per cent'
    assert not ( (delta_m[1] > delta_m_ref[1]+maxDiffIQUV[1]) or (delta_m[1] < delta_m_ref[1]-maxDiffIQUV[1]) ), f'Problem with Q values, get {delta_m[1]:.5f} instead of {delta_m_ref[1]:.5f} +- {errPercent:.0f} per cent'
    assert not ( (delta_m[2] > delta_m_ref[2]+maxDiffIQUV[2]) or (delta_m[2] < delta_m_ref[2]-maxDiffIQUV[2]) ), f'Problem with U values, get {delta_m[2]:.5f} instead of {delta_m_ref[2]:.5f} +- {errPercent:.0f} per cent'
    assert not ( (delta_m[3] > delta_m_ref[3]+maxDiffIQUV[3]) or (delta_m[3] < delta_m_ref[3]-maxDiffIQUV[3]) ), f'Problem with V values, get {delta_m[3]:.5f} instead of {delta_m_ref[3]:.5f} +- {errPercent:.0f} per cent'

# Remove tmp files
tmpFiles = glob.glob(RGP + "/tests/tmp_phaseA*")
for f in tmpFiles: os.remove(f)