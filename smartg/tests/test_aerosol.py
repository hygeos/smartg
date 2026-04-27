#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from smartg.atmosphere import AtmAFGL, AerOPAC
from pathlib import Path
import logging
import numpy as np
import xarray as xr
from smartg import conftest
import matplotlib.pyplot as plt
from smartg.config import DIR_AUXDATA, DIR_ROOT


# ***************************** Global variable(s) ******************************
MIXTURES = ['continental_clean','continental_average','continental_polluted',
            'urban','desert_spheric', 'desert', 'maritime_clean','maritime_polluted',
            'maritime_tropical','arctic','antarctic_spheric', 'antarctic']

SPECIES = ['miam','micm','minm',
           'mian','micn','minn', 
           'sscm','ssam',
           'inso',
           'soot',
           'suso',
           'waso'
           ]
# *******************************************************************************

# *********************************** logging ***********************************
# Create log file
Path(DIR_ROOT / 'smartg' / 'tests' / 'logs').mkdir(parents=True, exist_ok=True)

# Create a named logger
logger = logging.getLogger('test_aerosol')
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

# Set the formatter for the console handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
datefmt='%m/%d/%Y %I:%M:%S%p')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

# Create a file handler
file_handler = logging.FileHandler(DIR_ROOT / 'smartg' / 'tests' / 'logs' / 'aerosol.log', mode='w')
file_handler.setLevel(logging.INFO)

# Set the formatter for the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
# *******************************************************************************


@pytest.mark.parametrize('mix', MIXTURES)
def test_aer_mixtures(request, mix):
    wls = np.array([400., 700.])
    aer = AerOPAC(mix, 1., 550., H_free_min=0., H_stra_max=0, H_stra_min=0., H_free_max=0.)
    pro = AtmAFGL('afglt', comp=[aer]).calc(wls).to_xarray()

    ref_fname = DIR_AUXDATA / 'aerosols' / 'test_ref' / 'atm_afglt_{}.nc'.format(mix)
    pro_ref = xr.open_dataset(ref_fname)

    tau_aer_400 = pro['OD_p'][0,-1].values
    tau_aer_ref_400 = pro_ref['OD_p'][0,-1].values
    tau_aer_700 = pro['OD_p'][1,-1].values
    tau_aer_ref_700 = pro_ref['OD_p'][1,-1].values
    ssa_aer_400 = pro['ssa_p_atm'][0,-1].values
    ssa_aer_ref_400 = pro_ref['ssa_p_atm'][0,-1].values
    ssa_aer_700 = pro['ssa_p_atm'][1,-1].values
    ssa_aer_ref_700 = pro_ref['ssa_p_atm'][1,-1].values

    logger.info(f"{mix} - 400nm - tau_ref={tau_aer_ref_400 :.3f} - tau_calc={tau_aer_400 :.3f}")
    logger.info(f"{mix} - 700nm - tau_ref={tau_aer_ref_700 :.3f} - tau_calc={tau_aer_700 :.3f}")
    logger.info(f"{mix} - 400nm - ssa_ref={ssa_aer_ref_400 :.3f} - ssa_calc={ssa_aer_400 :.3f}")
    logger.info(f"{mix} - 700nm - ssa_ref={ssa_aer_ref_700 :.3f} - ssa_calc={ssa_aer_700 :.3f}")

    assert np.isclose(tau_aer_400, tau_aer_ref_400, atol=2e-3), \
        f"Problem with {mix} tau value at 400nm, get {tau_aer_400:.5f} instead of {tau_aer_ref_400:.5f}"
    assert np.isclose(tau_aer_700, tau_aer_ref_700, atol=2e-3), \
        f"Problem with {mix} tau value at 700nm, get {tau_aer_700:.5f} instead of {tau_aer_ref_700:.5f}"
    
    assert np.isclose(ssa_aer_400, ssa_aer_ref_400, atol=2e-3), \
        f"Problem with {mix} ssa value at 400nm, get {ssa_aer_400:.5f} instead of {ssa_aer_ref_400:.5f}"
    assert np.isclose(ssa_aer_700, ssa_aer_ref_700, atol=2e-3), \
        f"Problem with {mix} ssa value at 700nm, get {ssa_aer_700:.5f} instead of {ssa_aer_ref_700:.5f}"

    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle(f"Phase function at 400nm and z=0km")
    stk_labels  = ['F11', 'F12', 'F33', 'F34', 'F22', 'F44']
    stk_indices = [   0,     1,     2,     3,     4,     5  ]
    for idx, (label, istk) in enumerate(zip(stk_labels, stk_indices)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(pro_ref['theta_atm'].values, pro_ref['phase_atm'].values[0, istk, :], '-k', label='reference')
        ax.plot(pro['theta_atm'].values,     pro['phase_atm'].values[0, istk, :],     '--r', label='calculated')
        ax.set_title(label)
        ax.set_xlabel(r'$\theta$ (°)')
        ax.grid()
        ax.set_xlim([0, 180])
        if idx == 0:
            ax.set_yscale('log')
            ax.legend()
    fig.tight_layout()
    conftest.savefig(request, bbox_inches='tight')

    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle(f"Diff phase function (ref - calc) at 400nm and z=0km")
    for idx, (label, istk) in enumerate(zip(stk_labels, stk_indices)):
        ax = axes[idx // 2, idx % 2]
        diff = pro_ref['phase_atm'].values[0, istk, :] - pro['phase_atm'].values[0, istk, :]
        ax.plot(pro_ref['theta_atm'].values, diff, '-b')
        ax.set_title(label)
        ax.set_xlabel(r'$\theta$ (°)')
        ax.grid()
        ax.set_xlim([0, 180])
    fig.tight_layout()
    conftest.savefig(request, bbox_inches='tight')

    assert np.all(np.isclose(pro['phase_atm'].values[:,:4,:], pro_ref['phase_atm'].values[:,:4,:], atol=1e-5, rtol=1e-3)), \
        f"Problem with {mix} phase function"


@pytest.mark.parametrize('spe', SPECIES)
def test_aer_species(request, spe):
    wls = np.array([400., 700.])
    aer = AerOPAC(spe, 1., 550., H_free_min=0., H_stra_max=0, H_stra_min=0., H_free_max=0.)
    pro = AtmAFGL('afglt', comp=[aer]).calc(wls).to_xarray()

    ref_fname = DIR_AUXDATA / 'aerosols' / 'test_ref' / 'atm_afglt_{}.nc'.format(spe)
    pro_ref = xr.open_dataset(ref_fname)

    tau_aer_400 = pro['OD_p'][0,-1].values
    tau_aer_ref_400 = pro_ref['OD_p'][0,-1].values
    tau_aer_700 = pro['OD_p'][1,-1].values
    tau_aer_ref_700 = pro_ref['OD_p'][1,-1].values
    ssa_aer_400 = pro['ssa_p_atm'][0,-1].values
    ssa_aer_ref_400 = pro_ref['ssa_p_atm'][0,-1].values
    ssa_aer_700 = pro['ssa_p_atm'][1,-1].values
    ssa_aer_ref_700 = pro_ref['ssa_p_atm'][1,-1].values

    logger.info(f"{spe} - 400nm - tau_ref={tau_aer_ref_400 :.3f} - tau_calc={tau_aer_400 :.3f}")
    logger.info(f"{spe} - 700nm - tau_ref={tau_aer_ref_700 :.3f} - tau_calc={tau_aer_700 :.3f}")
    logger.info(f"{spe} - 400nm - ssa_ref={ssa_aer_ref_400 :.3f} - ssa_calc={ssa_aer_400 :.3f}")
    logger.info(f"{spe} - 700nm - ssa_ref={ssa_aer_ref_700 :.3f} - ssa_calc={ssa_aer_700 :.3f}")

    assert np.isclose(tau_aer_400, tau_aer_ref_400, atol=2e-3), \
        f"Problem with {spe} tau value at 400nm, get {tau_aer_400:.5f} instead of {tau_aer_ref_400:.5f}"
    assert np.isclose(tau_aer_700, tau_aer_ref_700, atol=2e-3), \
        f"Problem with {spe} tau value at 700nm, get {tau_aer_700:.5f} instead of {tau_aer_ref_700:.5f}"
    
    assert np.isclose(ssa_aer_400, ssa_aer_ref_400, atol=2e-3), \
        f"Problem with {spe} ssa value at 400nm, get {ssa_aer_400:.5f} instead of {ssa_aer_ref_400:.5f}"
    assert np.isclose(ssa_aer_700, ssa_aer_ref_700, atol=2e-3), \
        f"Problem with {spe} ssa value at 700nm, get {ssa_aer_700:.5f} instead of {ssa_aer_ref_700:.5f}"
    
    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle(f"Phase function at 400nm and z=0km")
    stk_labels  = ['F11', 'F12', 'F33', 'F34', 'F22', 'F44']
    stk_indices = [   0,     1,     2,     3,     4,     5  ]
    for idx, (label, istk) in enumerate(zip(stk_labels, stk_indices)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(pro_ref['theta_atm'].values, pro_ref['phase_atm'].values[0, istk, :], '-k', label='reference')
        ax.plot(pro['theta_atm'].values,     pro['phase_atm'].values[0, istk, :],     '--r', label='calculated')
        ax.set_title(label)
        ax.set_xlabel(r'$\theta$ (°)')
        ax.grid()
        ax.set_xlim([0, 180])
        if idx == 0:
            ax.set_yscale('log')
            ax.legend()
    fig.tight_layout()
    conftest.savefig(request, bbox_inches='tight')

    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle(f"Diff phase function (ref - calc) at 400nm and z=0km")
    for idx, (label, istk) in enumerate(zip(stk_labels, stk_indices)):
        ax = axes[idx // 2, idx % 2]
        diff = pro_ref['phase_atm'].values[0, istk, :] - pro['phase_atm'].values[0, istk, :]
        ax.plot(pro_ref['theta_atm'].values, diff, '-b')
        ax.set_title(label)
        ax.set_xlabel(r'$\theta$ (°)')
        ax.grid()
        ax.set_xlim([0, 180])
    fig.tight_layout()
    conftest.savefig(request, bbox_inches='tight')

    assert np.all(np.isclose(pro['phase_atm'].values[:,0:4,:], pro_ref['phase_atm'].values[:,0:4,:], atol=1e-5, rtol=1e-3)), \
        f"Problem with {spe} phase function"


def test_desert_free_stra(request):
    wls = np.array([400., 700.])
    aer = AerOPAC('desert', 1., 550.)
    pro = AtmAFGL('afglt', comp=[aer], pfgrid=[100., 12., 6., 0.]).calc(wls).to_xarray()
    
    ref_fname = DIR_AUXDATA / 'aerosols' / 'test_ref' / 'atm_afglt_desert_free_stra.nc'
    pro_ref = xr.open_dataset(ref_fname)

    tau_aer_400 = pro['OD_p'][0,-1].values
    tau_aer_ref_400 = pro_ref['OD_p'][0,-1].values
    tau_aer_700 = pro['OD_p'][1,-1].values
    tau_aer_ref_700 = pro_ref['OD_p'][1,-1].values
    ssa_aer_400 = pro['ssa_p_atm'][0,-1].values
    ssa_aer_ref_400 = pro_ref['ssa_p_atm'][0,-1].values
    ssa_aer_700 = pro['ssa_p_atm'][1,-1].values
    ssa_aer_ref_700 = pro_ref['ssa_p_atm'][1,-1].values

    logger.info(f"desert free stra - 400nm - tau_ref={tau_aer_ref_400 :.3f} - tau_calc={tau_aer_400 :.3f}")
    logger.info(f"desert free stra - 700nm - tau_ref={tau_aer_ref_700 :.3f} - tau_calc={tau_aer_700 :.3f}")
    logger.info(f"desert free stra - 400nm - ssa_ref={ssa_aer_ref_400 :.3f} - ssa_calc={ssa_aer_400 :.3f}")
    logger.info(f"desert free stra - 700nm - ssa_ref={ssa_aer_ref_700 :.3f} - ssa_calc={ssa_aer_700 :.3f}")

    assert np.isclose(tau_aer_400, tau_aer_ref_400, atol=2e-3), \
        f"Problem with desert free stra tau value at 400nm, get {tau_aer_400:.5f} instead of {tau_aer_ref_400:.5f}"
    assert np.isclose(tau_aer_700, tau_aer_ref_700, atol=2e-3), \
        f"Problem with desert free stra tau value at 700nm, get {tau_aer_700:.5f} instead of {tau_aer_ref_700:.5f}"
    
    assert np.isclose(ssa_aer_400, ssa_aer_ref_400, atol=2e-3), \
        f"Problem with desert free stra ssa value at 400nm, get {ssa_aer_400:.5f} instead of {ssa_aer_ref_400:.5f}"
    assert np.isclose(ssa_aer_700, ssa_aer_ref_700, atol=2e-3), \
        f"Problem with desert free stra ssa value at 700nm, get {ssa_aer_700:.5f} instead of {ssa_aer_ref_700:.5f}"
    
    iph = pro['iphase_atm'][0,-1].values
    iph_ref = pro_ref['iphase_atm'][0,-1].values
    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle(f"Phase function at 400nm and z=0km")
    stk_labels  = ['F11', 'F12', 'F33', 'F34', 'F22', 'F44']
    stk_indices = [   0,     1,     2,     3,     4,     5  ]
    for idx, (label, istk) in enumerate(zip(stk_labels, stk_indices)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(pro_ref['theta_atm'].values, pro_ref['phase_atm'].values[iph_ref, istk, :], '-k', label='reference')
        ax.plot(pro['theta_atm'].values,     pro['phase_atm'].values[iph, istk, :],     '--r', label='calculated')
        ax.set_title(label)
        ax.set_xlabel(r'$\theta$ (°)')
        ax.grid()
        ax.set_xlim([0, 180])
        if idx == 0:
            ax.set_yscale('log')
            ax.legend()
    fig.tight_layout()
    conftest.savefig(request, bbox_inches='tight')

    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle(f"Diff phase function (ref - calc) at 400nm and z=0km")
    for idx, (label, istk) in enumerate(zip(stk_labels, stk_indices)):
        ax = axes[idx // 2, idx % 2]
        diff = pro_ref['phase_atm'].values[iph_ref, istk, :] - pro['phase_atm'].values[iph, istk, :]
        ax.plot(pro_ref['theta_atm'].values, diff, '-b')
        ax.set_title(label)
        ax.set_xlabel(r'$\theta$ (°)')
        ax.grid()
        ax.set_xlim([0, 180])
    fig.tight_layout()
    conftest.savefig(request, bbox_inches='tight')

    assert np.all(np.isclose(pro['phase_atm'].values[:,0:4,:], pro_ref['phase_atm'].values[:,0:4,:], atol=1e-5, rtol=1e-3)), \
        f"Problem with desert free stra phase function"


def test_dd_cc_mixture(request):
    wls = np.array([400., 700.])
    pfgrid = [100., 6., 5., 4., 3., 2., 1., 0.]
    aer1 = AerOPAC('desert', 1., 550.,
                    H_free_min=0., H_stra_max=0, 
                    H_stra_min=0., H_free_max=0.)
    aer2 = AerOPAC('continental_clean', 1., 550.,
                    H_free_min=0., H_stra_max=0, 
                    H_stra_min=0., H_free_max=0.)
    pro = AtmAFGL('afglt', comp=[aer1, aer2], pfgrid=pfgrid).calc(wls).to_xarray()
    
    ref_fname = DIR_AUXDATA / 'aerosols' / 'test_ref' / 'atm_afglt_desert_cont_clean_mix.nc'
    pro_ref = xr.open_dataset(ref_fname)

    tau_aer_400 = pro['OD_p'][0,-1].values
    tau_aer_ref_400 = pro_ref['OD_p'][0,-1].values
    tau_aer_700 = pro['OD_p'][1,-1].values
    tau_aer_ref_700 = pro_ref['OD_p'][1,-1].values
    ssa_aer_400 = pro['ssa_p_atm'][0,-1].values
    ssa_aer_ref_400 = pro_ref['ssa_p_atm'][0,-1].values
    ssa_aer_700 = pro['ssa_p_atm'][1,-1].values
    ssa_aer_ref_700 = pro_ref['ssa_p_atm'][1,-1].values

    logger.info(f"dd + cc - 400nm - tau_ref={tau_aer_ref_400 :.3f} - tau_calc={tau_aer_400 :.3f}")
    logger.info(f"dd + cc - 700nm - tau_ref={tau_aer_ref_700 :.3f} - tau_calc={tau_aer_700 :.3f}")
    logger.info(f"dd + cc - 400nm - ssa_ref={ssa_aer_ref_400 :.3f} - ssa_calc={ssa_aer_400 :.3f}")
    logger.info(f"dd + cc - 700nm - ssa_ref={ssa_aer_ref_700 :.3f} - ssa_calc={ssa_aer_700 :.3f}")

    assert np.isclose(tau_aer_400, tau_aer_ref_400, atol=2e-3), \
        f"Problem with dd + cc tau value at 400nm, get {tau_aer_400:.5f} instead of {tau_aer_ref_400:.5f}"
    assert np.isclose(tau_aer_700, tau_aer_ref_700, atol=2e-3), \
        f"Problem with dd + cc tau value at 700nm, get {tau_aer_700:.5f} instead of {tau_aer_ref_700:.5f}"
    
    assert np.isclose(ssa_aer_400, ssa_aer_ref_400, atol=2e-3), \
        f"Problem with dd + cc ssa value at 400nm, get {ssa_aer_400:.5f} instead of {ssa_aer_ref_400:.5f}"
    assert np.isclose(ssa_aer_700, ssa_aer_ref_700, atol=2e-3), \
        f"Problem with dd + cc ssa value at 700nm, get {ssa_aer_700:.5f} instead of {ssa_aer_ref_700:.5f}"
    
    iph = pro['iphase_atm'][0,-1].values
    iph_ref = pro_ref['iphase_atm'][0,-1].values
    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle(f"Phase function at 400nm and z=0km")
    stk_labels  = ['F11', 'F12', 'F33', 'F34', 'F22', 'F44']
    stk_indices = [   0,     1,     2,     3,     4,     5  ]
    for idx, (label, istk) in enumerate(zip(stk_labels, stk_indices)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(pro_ref['theta_atm'].values, pro_ref['phase_atm'].values[iph_ref, istk, :], '-k', label='reference')
        ax.plot(pro['theta_atm'].values,     pro['phase_atm'].values[iph, istk, :],     '--r', label='calculated')
        ax.set_title(label)
        ax.set_xlabel(r'$\theta$ (°)')
        ax.grid()
        ax.set_xlim([0, 180])
        if idx == 0:
            ax.set_yscale('log')
            ax.legend()
    fig.tight_layout()
    conftest.savefig(request, bbox_inches='tight')

    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle(f"Diff phase function (ref - calc) at 400nm and z=0km")
    for idx, (label, istk) in enumerate(zip(stk_labels, stk_indices)):
        ax = axes[idx // 2, idx % 2]
        diff = pro_ref['phase_atm'].values[iph_ref, istk, :] - pro['phase_atm'].values[iph, istk, :]
        ax.plot(pro_ref['theta_atm'].values, diff, '-b')
        ax.set_title(label)
        ax.set_xlabel(r'$\theta$ (°)')
        ax.grid()
        ax.set_xlim([0, 180])
    fig.tight_layout()
    conftest.savefig(request, bbox_inches='tight')
    
    assert np.all(np.isclose(pro['phase_atm'].values[:,0:4,:], pro_ref['phase_atm'].values[:,0:4,:], atol=1e-5, rtol=1e-3)), \
        f"Problem with dd + cc phase function"


def test_desert_one_wl(request):
    wl = 400.
    aer = AerOPAC('desert', 1., 550.,
                    H_free_min=0., H_stra_max=0, 
                    H_stra_min=0., H_free_max=0.)
    pro = AtmAFGL('afglt', comp=[aer]).calc(wl).to_xarray()
    
    ref_fname = DIR_AUXDATA / 'aerosols' / 'test_ref' / 'atm_afglt_desert.nc'
    pro_ref = xr.open_dataset(ref_fname)

    tau_aer_400 = pro['OD_p'][0,-1].values
    tau_aer_ref_400 = pro_ref['OD_p'][0,-1].values
    ssa_aer_400 = pro['ssa_p_atm'][0,-1].values
    ssa_aer_ref_400 = pro_ref['ssa_p_atm'][0,-1].values

    logger.info(f"desert one wl - 400nm - tau_ref={tau_aer_ref_400 :.3f} - tau_calc={tau_aer_400 :.3f}")
    logger.info(f"desert one wl - 400nm - ssa_ref={ssa_aer_ref_400 :.3f} - ssa_calc={ssa_aer_400 :.3f}")
    
    assert np.isclose(tau_aer_400, tau_aer_ref_400, atol=2e-3), \
        f"Problem with desert one wl tau value at 400nm, get {tau_aer_400:.5f} instead of {tau_aer_ref_400:.5f}"
    
    assert np.isclose(ssa_aer_400, ssa_aer_ref_400, atol=2e-3), \
        f"Problem with desert one wl ssa value at 400nm, get {ssa_aer_400:.5f} instead of {ssa_aer_ref_400:.5f}"
    
    iph = pro['iphase_atm'][0,-1].values
    iph_ref = pro_ref['iphase_atm'][0,-1].values
    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle(f"Phase function at 400nm and z=0km")
    stk_labels  = ['F11', 'F12', 'F33', 'F34', 'F22', 'F44']
    stk_indices = [   0,     1,     2,     3,     4,     5  ]
    for idx, (label, istk) in enumerate(zip(stk_labels, stk_indices)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(pro_ref['theta_atm'].values, pro_ref['phase_atm'].values[iph_ref, istk, :], '-k', label='reference')
        ax.plot(pro['theta_atm'].values,     pro['phase_atm'].values[iph, istk, :],     '--r', label='calculated')
        ax.set_title(label)
        ax.set_xlabel(r'$\theta$ (°)')
        ax.grid()
        ax.set_xlim([0, 180])
        if idx == 0:
            ax.set_yscale('log')
            ax.legend()
    fig.tight_layout()
    conftest.savefig(request, bbox_inches='tight')

    plt.close('all')
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle(f"Diff phase function (ref - calc) at 400nm and z=0km")
    for idx, (label, istk) in enumerate(zip(stk_labels, stk_indices)):
        ax = axes[idx // 2, idx % 2]
        diff = pro_ref['phase_atm'].values[iph_ref, istk, :] - pro['phase_atm'].values[iph, istk, :]
        ax.plot(pro_ref['theta_atm'].values, diff, '-b')
        ax.set_title(label)
        ax.set_xlabel(r'$\theta$ (°)')
        ax.grid()
        ax.set_xlim([0, 180])
    fig.tight_layout()
    conftest.savefig(request, bbox_inches='tight')

    ipha = pro['iphase_atm'][0,:].values
    ipha_ref = pro_ref['iphase_atm'][0,:].values
    assert np.all(np.isclose(pro['phase_atm'].values[ipha,0:4,:], pro_ref['phase_atm'].values[ipha_ref,0:4,:], atol=1e-5, rtol=1e-3)), \
        f"Problem with desert one wl phase function"