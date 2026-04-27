#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from smartg.atmosphere import AtmAFGL, AerOPAC
from pathlib import Path
import logging
import numpy as np
import xarray as xr


# ***************************** Global variable(s) ******************************
ROOTPATH = Path(__file__).resolve().parent.parent.parent

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
Path(ROOTPATH / 'smartg' / 'tests' / 'logs').mkdir(parents=True, exist_ok=True)

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
file_handler = logging.FileHandler(ROOTPATH / 'smartg' / 'tests' / 'logs' / 'aerosol.log', mode='w')
file_handler.setLevel(logging.INFO)

# Set the formatter for the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
# *******************************************************************************


@pytest.mark.parametrize('mix', MIXTURES)
def test_aer_mixtures(mix):
    wls = np.array([400., 700.])
    aer = AerOPAC(mix, 1., 550., H_free_min=0., H_stra_max=0, H_stra_min=0., H_free_max=0.)
    pro = AtmAFGL('afglt', comp=[aer]).calc(wls).to_xarray()

    ref_fname = ROOTPATH / 'smartg' / 'tests' / 'aer_auxdata_ref' / 'atm_afglt_{}.nc'.format(mix)
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


@pytest.mark.parametrize('spe', SPECIES)
def test_aer_species(spe):
    wls = np.array([400., 700.])
    aer = AerOPAC(spe, 1., 550., H_free_min=0., H_stra_max=0, H_stra_min=0., H_free_max=0.)
    pro = AtmAFGL('afglt', comp=[aer]).calc(wls).to_xarray()

    ref_fname = ROOTPATH / 'smartg' / 'tests' / 'aer_auxdata_ref' / 'atm_afglt_{}.nc'.format(spe)
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