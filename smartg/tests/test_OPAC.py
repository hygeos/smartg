#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from smartg.atmosphere import AtmAFGL, AerOPAC
import os
from pathlib import Path
import logging

# ***************************** Global variable(s) ******************************
ROOTPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

AER_SPHERIC = ['continental_clean','continental_average','continental_polluted','urban','desert_spheric',
               'maritime_clean','maritime_polluted','maritime_tropical','arctic','antarctic_spheric']

# OPAC 1998 paper table 3
AER_SPHERIC_OPAC = {}
AER_SPHERIC_OPAC.update({'continental_clean':{'tau':0.064, 'ssa':0.972}})
AER_SPHERIC_OPAC.update({'continental_average':{'tau':0.151, 'ssa':0.925}})
AER_SPHERIC_OPAC.update({'continental_polluted':{'tau':0.327, 'ssa':0.892}})
AER_SPHERIC_OPAC.update({'urban':{'tau':0.643, 'ssa':0.817}})
AER_SPHERIC_OPAC.update({'desert_spheric':{'tau':0.286, 'ssa':0.888}})
AER_SPHERIC_OPAC.update({'maritime_clean':{'tau':0.096, 'ssa':0.997}})
AER_SPHERIC_OPAC.update({'maritime_polluted':{'tau':0.117, 'ssa':0.975}})
AER_SPHERIC_OPAC.update({'maritime_tropical':{'tau':0.056, 'ssa':0.998}})
AER_SPHERIC_OPAC.update({'arctic':{'tau':0.063, 'ssa':0.887}})
AER_SPHERIC_OPAC.update({'antarctic_spheric':{'tau':0.072, 'ssa':1.000}})

DELTA_R_TAU = 0.6 # relative error in %
DELTA_R_SSA = 0.2 # relative error in %
# *******************************************************************************

# *********************************** logging ***********************************
# Create log file
Path(os.path.join(ROOTPATH, "tests/logs/")).mkdir(parents=True, exist_ok=True)

# Create a named logger
logger = logging.getLogger('test_OPAC')
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
file_handler = logging.FileHandler(ROOTPATH + "/tests/logs/OPAC.log", mode='w')
file_handler.setLevel(logging.INFO)

# Set the formatter for the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
# *******************************************************************************


@pytest.mark.parametrize('mod', AER_SPHERIC)
def test_aer_spheric(mod):
    wl_ref = 550.
    aer_comp = AerOPAC(mod, tau_ref=None, w_ref=wl_ref, rh_mix=80., rh_free=0., rh_stra=0.)
    pro = AtmAFGL('afglt', comp=[aer_comp]).calc(wl_ref)

    opac_tau = AER_SPHERIC_OPAC[mod]['tau']
    opac_ssa = AER_SPHERIC_OPAC[mod]['ssa']

    cond_tau1 = pro['OD_p'][0,-1] > (opac_tau - (opac_tau*DELTA_R_TAU*1e-2 + pro['OD_p'][0,-1]*DELTA_R_TAU*1e-2))
    cond_tau2 = pro['OD_p'][0,-1] < (opac_tau + (opac_tau*DELTA_R_TAU*1e-2 + pro['OD_p'][0,-1]*DELTA_R_TAU*1e-2))
    cond_ssa1 = pro['ssa_p_atm'][0,-1] > (opac_ssa - (opac_ssa*DELTA_R_SSA*1e-2 + pro['ssa_p_atm'][0,-1]*DELTA_R_SSA*1e-2))
    cond_ssa2 = pro['ssa_p_atm'][0,-1] < (opac_ssa + (opac_ssa*DELTA_R_SSA*1e-2 + pro['ssa_p_atm'][0,-1]*DELTA_R_SSA*1e-2))

    logger.info(f"{mod} - tau_ref={AER_SPHERIC_OPAC[mod]['tau'] :.3f} - tau_calc={pro['OD_p'][0,-1] :.3f}")
    logger.info(f"{mod} - ssa_ref={AER_SPHERIC_OPAC[mod]['ssa'] :.3f} - ssa_calc={pro['ssa_p_atm'][0,-1] :.3f}")

    assert ( cond_tau1 and cond_tau2 ), f"Problem with {mod} tau value, get {pro['OD_p'][0,-1]:.5f} " + \
        f"instead of {opac_tau:.5f} +- {(opac_tau*DELTA_R_TAU*1e-2 + pro['OD_p'][0,-1]*DELTA_R_TAU*1e-2):.5f}"
    
    assert ( cond_ssa1 and cond_ssa2 ), f"Problem with {mod} tau value, get {pro['ssa_p_atm'][0,-1]:.5f} " + \
        f"instead of {opac_ssa:.5f} +- {(opac_ssa*DELTA_R_SSA*1e-2 + pro['ssa_p_atm'][0,-1]*DELTA_R_SSA*1e-2):.5f}"
