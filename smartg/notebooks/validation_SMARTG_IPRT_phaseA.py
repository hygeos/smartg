# %% [markdown]
# # SMART-G validation IPRT phase A 
# - https://www.meteo.physik.uni-muenchen.de/~iprt/doku.php?id=start

# %% [markdown]
# ## Symbols used in this notebook
#
# | symbol | meaning |
# |---|---|
# | `i_mod`, `q_mod`, `u_mod`, `v_mod` | Stokes components simulated by SMART-G |
# | `i_rmod`, ... | the same for the reference model, MYSTIC |
# | `i_val`, ... | the values read back from the IPRT output file |
# | `i_std_mod`, ... | their Monte Carlo standard deviations |
# | `iquv_*` | the four components stacked, as `smartg.iprt.group_iquv` returns them |
# | `sza`, `saa` | solar zenith and azimuth angle, in degrees |
# | `vza`, `vaa` | viewing zenith and azimuth angle |
# | `phi_0` | `180 - saa`, the MYSTIC azimuth convention |
# | `m_a1b`, `m_b4f` | the run output of a case, backward or forward |
# | `pro_a1`, `surf_a1` | the profile and surface that case is built with |
#
# `_pp` and `_al` mark the principal plane and the almucantar of case A5,
# `_dep003` a depolarization of 0.03, and `_0km` / `_30km` the altitude
# of the sensor.

# %%
# %matplotlib inline
# next 2 lines allow to automatically reload modules that have been
# changed externally
# %reload_ext autoreload
# %autoreload 2

import sys
from pathlib import Path

try:
    import subprocess
    check = subprocess.check_call(['git', 'rev-parse', '--show-toplevel'],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.STDOUT)
    # Root Git Path
    ROOTPATH = subprocess.Popen(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
    ROOTPATH = Path(ROOTPATH)
except subprocess.CalledProcessError:
    ROOTPATH = Path.cwd()
sys.path.insert(0, str(ROOTPATH))

from smartg.config import DIR_AUXDATA
from smartg.xarray import drop_axes
from smartg.smartg import LocalEstimate, Smartg
from smartg.surface import RoughSurface, LambSurface
from smartg.albedo import AlbedoCst
from smartg.sensor import Sensor
from smartg.atmosphere import Atm1D

import pandas as pd
import numpy as np
import xarray as xr

from smartg.phase import read_phase, get_prof_phases

from smartg.iprt.iprt import (convert_sgout_to_iprtout,
                             select_and_plot_polar_iprt,
                             select_iprt_iquv, compute_deltam,
                             plot_iprt_radiances, group_iquv)


output_folder_path = ROOTPATH / 'smartg' / 'notebooks' / 'SMARTG_RES_IPRT_REF'
Path(output_folder_path).mkdir(parents=True, exist_ok=True)

OPT_PROP_PATH = DIR_AUXDATA / 'IPRT' / 'phaseA' / 'opt_prop'
MYSTIC_RES_PATH = DIR_AUXDATA / 'IPRT' / 'phaseA' / 'mystic_res'

# Compilation in forward and backward
s_1df = Smartg(alt_pp=False, back=False, double=True, bias=True)
s_1db = Smartg(alt_pp=False, back=True, double=True, bias=True)

SEED = 1e8
NB_PH = 1e7

# %% [markdown]
# ## Test cases including a single layer

# %% [markdown]
# ### Case A1

# %% [markdown]
# #### Atmosphere profil

# %%
mol_sca = np.array([0., 0.5])[None, :]
mol_abs= np.array([0., 0.])[None, :]
z = np.array([1., 0.])
wavelength = 550.

pro_a1 = Atm1D(
    'afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs).calc(wavelength)
surf_a1  = None

# %% [markdown]
# #### Compute radiances depol=0, sza=0, saa=65
# - Note: In forward there is a bug for sza near to 0 (see Q and U values), then simulation is performed in backward (more comsuming) 

# %%
sza = 0.
saa = 65.
phi_0 = 180.-saa # To follow MYSTIC convention
# count only uptoa
le     = LocalEstimate(
    th_deg=np.array([sza]), phi_deg=np.array([phi_0]),
    count_level=np.full((len(np.atleast_1d(sza))), 0, dtype=np.int32))

# BOA radiances
vza_min = 0.
vza_max = 80.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

# vaa from 0. to 180.
vaa_min = 0.
vaa_max = 360.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

lsensors = []
n_vza = len(vza)
n_vaa = len(vaa)
n_dir = round(n_vaa*n_vza)
for iza, za in enumerate(vza):
    for iaa, aa in enumerate(vaa):
        phi = -aa+180
        lsensors.append(Sensor(pos_z=np.min(z), th_deg=za, ph_deg=phi,
                               loc='ATMOS'))

m_a1b = s_1db.run(wavelength=550., n_photons=NB_PH*n_dir, n_loop=1e8,
                  atmosphere=pro_a1, sensor=lsensors, le=le, surface=surf_a1,
                  xblock=64, xgrid=1024, beer=1, depo=0.0, stdev=True,
                  progress=True, seed=SEED, output_layers=int(0))

m_a1b = drop_axes(m_a1b, 'Azimuth angles', 'Zenith angles')

for name in list(m_a1b.data_vars):
    if 'sensor index' in m_a1b[name].dims:
        mat_tmp = np.swapaxes(m_a1b[name].values.reshape(len(vza), len(vaa)),
                              0, 1)
        attrs_tmp = m_a1b[name].attrs
        m_a1b = m_a1b.drop_vars([name])
        m_a1b[name] = xr.Variable(('Azimuth angles', 'Zenith angles'), mat_tmp,
                                  attrs=attrs_tmp)
m_a1b = m_a1b.assign_coords({
    'Azimuth angles': -vaa+180., 'Zenith angles': vza})

m_a1b_boa_dep0 = drop_axes(m_a1b, 'sensor index')

# TOA radiances
vza_min = 100.
vza_max = 180.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

lsensors = []
n_vza = len(vza)
n_vaa = len(vaa)
n_dir = round(n_vaa*n_vza)
for iza, za in enumerate(vza):
    for iaa, aa in enumerate(vaa):
        phi = -aa+180
        lsensors.append(Sensor(pos_z=np.max(z), th_deg=za, ph_deg=phi,
                               loc='ATMOS'))

m_a1b = s_1db.run(wavelength=wavelength, n_photons=NB_PH*n_dir, n_loop=1e8,
                  atmosphere=pro_a1, sensor=lsensors, le=le, surface=surf_a1,
                  xblock=64, xgrid=1024, beer=1, depo=0.0, stdev=True,
                  progress=True, seed=SEED, output_layers=int(0))

m_a1b = drop_axes(m_a1b, 'Azimuth angles', 'Zenith angles')

for name in list(m_a1b.data_vars):
    if 'sensor index' in m_a1b[name].dims:
        mat_tmp = np.swapaxes(m_a1b[name].values.reshape(len(vza), len(vaa)),
                              0, 1)
        attrs_tmp = m_a1b[name].attrs
        m_a1b = m_a1b.drop_vars([name])
        m_a1b[name] = xr.Variable(('Azimuth angles', 'Zenith angles'), mat_tmp,
                                  attrs=attrs_tmp)
m_a1b = m_a1b.assign_coords({
    'Azimuth angles': -vaa+180., 'Zenith angles': vza})

m_a1b_toa_dep0 = drop_axes(m_a1b, 'sensor index')

# %%
m_a1b_boa_dep0.to_netcdf(output_folder_path / "iprt_a1_smartg_dep0_boa_ref.nc")
m_a1b_toa_dep0.to_netcdf(output_folder_path / "iprt_a1_smartg_dep0_toa_ref.nc")

# %% [markdown]
# #### Compute radiances depol=0.03, sza=30, saa=0

# %%
vza_min = 100.
vza_max = 180.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

vaa_min = 0.
vaa_max = 360.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

# SMART-G Forward th and phi using local estimate (anticlockwise)
# conversion with vza and vaa MYSTIC (clockwise)
th  = 180.-vza
phi = -vaa
th[th==0] = 1e-6  # avoid problem due to special case of 0
le     = LocalEstimate(th_deg=th, phi_deg=phi)  # , zip=True

sza = 30.0
saa = 0.
phi_0 = 180.-saa # SMART-G anticlockwise converted to be consistent with MYSTIC

m_a1f_dep003 = s_1df.run(th_deg=sza, ph_deg=phi_0, wavelength=wavelength,
                         n_photons=NB_PH*10,
                         n_loop=min(1e6, round(NB_PH/10.)), atmosphere=pro_a1,
                         output_layers=int(7), le=le, surface=surf_a1,
                         xblock=64, xgrid=1024, beer=1, depo=0.03, stdev=True,
                         seed=SEED)

# %%
m_a1f_dep003.to_netcdf(output_folder_path / "iprt_a1_smartg_dep003_ref.nc")

# %% [markdown]
# #### Compute radiances depol=0.1, sza=30, saa=65

# %%
vza_min = 100.
vza_max = 180.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

vaa_min = 0.
vaa_max = 360.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

# SMART-G Forward th and phi using local estimate (anticlockwise)
# conversion with vza and vaa MYSTIC (clockwise)
th  = 180.-vza
phi = -vaa
th[th==0] = 1e-6  # avoid problem due to special case of 0
le     = LocalEstimate(th_deg=th, phi_deg=phi)  # , zip=True

sza = 30.0
saa = 65.0
phi_0 = 180.-saa # SMART-G anticlockwise converted to be consistent with MYSTIC

# stdev_lim = StdevLim(err_rel_min=1e-2, format=".2e", verbose=True)
m_a1f_dep01 = s_1df.run(th_deg=sza, ph_deg=phi_0, wavelength=wavelength,
                        n_photons=NB_PH, n_loop=min(1e6, round(NB_PH/10.)),
                        atmosphere=pro_a1, output_layers=int(7), le=le,
                        surface=surf_a1, xblock=64, xgrid=1024, beer=1,
                        depo=0.1, stdev=True, seed=SEED)

# %%
m_a1f_dep01.to_netcdf(output_folder_path / "iprt_a1_smartg_dep01_ref.nc")

# %% [markdown]
# #### Convert into iprt output format

# %%
# Radiances
vza_boa_dep0 = m_a1b_boa_dep0.coords['Zenith angles'].values
vaa_boa_dep0 = 180.-m_a1b_boa_dep0.coords['Azimuth angles'].values
vza_toa_dep0 = m_a1b_toa_dep0.coords['Zenith angles'].values
vaa_toa_dep0 = 180.-m_a1b_toa_dep0.coords['Azimuth angles'].values

vza_dep003 = 180.-m_a1f_dep003.coords['Zenith angles'].values
vaa_dep003 = -m_a1f_dep003.coords['Azimuth angles'].values
vza_dep01 = 180.-m_a1f_dep01.coords['Zenith angles'].values
vaa_dep01 = -m_a1f_dep01.coords['Azimuth angles'].values

filename  = output_folder_path / "iprt_case_a1_smartg_ref.dat"
case_name = "A1"
depols    = [0., 0., 0.03, 0.03, 0.1, 0.1]
altitudes      = [0., 1., 0., 1., 0., 1.]
szas      = [0., 0., 30., 30., 30., 30.]
saas      = [65., 65., 0., 0., 65., 65.]

# convert
convert_sgout_to_iprtout(
                         datasets=[m_a1b_boa_dep0, m_a1b_toa_dep0,
                                   m_a1f_dep003, m_a1f_dep003,
                                   m_a1f_dep01, m_a1f_dep01],
                         u_signs=[1., 1, -1, -1, -1, -1], case_name=case_name,
                         depols=depols, altitudes=altitudes, szas=szas,
                         saas=saas,
                         vzas=[vza_boa_dep0, vza_toa_dep0,
                               180.-vza_dep003, vza_dep003,
                               180.-vza_dep01, vza_dep01],
                         vaas=[vaa_boa_dep0, vaa_toa_dep0,
                               vaa_dep003, vaa_dep003,
                               vaa_dep01, vaa_dep01],
                         file_name=filename,
                         output_layer=['_up (TOA)', '_up (TOA)',
                                       '_down (0+)', '_up (TOA)',
                                       '_down (0+)', '_up (TOA)'])

# %% [markdown]
# #### Comparison with MYSTIC

# %%
filename  = output_folder_path / "iprt_case_a1_smartg_ref.dat"
case_name = "A1"
depols    = [0., 0., 0.03, 0.03, 0.1, 0.1]
altitudes      = [0., 1., 0., 1., 0., 1.]
szas      = [0., 0., 30., 30., 30., 30.]
saas      = [65., 65., 0., 0., 65., 65.]
l_invth    = [False, True, False, True, False, True]

smartg_a1 = pd.read_csv(filename, header=None, sep=r'\s+', dtype=float,
                        comment="#").values
mystic_a1 = pd.read_csv(
    MYSTIC_RES_PATH / "iprt_case_a1_mystic.dat", header=None, sep=r'\s+',
    dtype=float, comment="#").values
model     = smartg_a1
ref_model = mystic_a1
mod_name  = "SMARTG"
rmod_name = "MYSTIC"
l_invth_rmod  = [False, True, False, True, False, True]
l_invth_mod   = [False, True, False, True, False, True]
l_u_sign_rmod  =[True, True, True, True, True, True]
l_u_sign_mod  = [True, True, True, True, True, True]
l_v_sign_rmod  =[False, False, False, False, False, False]
l_v_sign_mod  = [False, False, False, False, False, False]
avoid_p_rmod = False
avoid_p_mod  = True
avoid_p_dif  = False
sym = False

n_res = len(depols)
for ires in range (0, n_res):
    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f}  - "
             f"{altitudes[ires]:.0f}km - {rmod_name}")
    i_rmod, q_rmod, u_rmod, v_rmod = select_and_plot_polar_iprt(
        ref_model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_rmod[ires], change_v_sign=l_v_sign_rmod[ires],
        inv_thetas=l_invth[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_rmod)

    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{altitudes[ires]:.0f}km - {mod_name}")
    i_mod, q_mod, u_mod, v_mod = select_and_plot_polar_iprt(
        model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_mod[ires], change_v_sign=l_v_sign_mod[ires],
        inv_thetas=l_invth[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_mod)

    if (ires == 0):
        iquv_mod_tot = group_iquv(i_list=[i_mod], q_list=[q_mod],
                                  u_list=[u_mod], v_list=[v_mod])
        iquv_rmod_tot = group_iquv(i_list=[i_rmod], q_list=[q_rmod],
                                   u_list=[u_rmod], v_list=[v_rmod])
    else:
        iquv_mod_tot = np.concatenate((iquv_mod_tot, group_iquv(
            i_list=[i_mod], q_list=[q_mod],
            u_list=[u_mod], v_list=[v_mod])), axis=1)
        iquv_rmod_tot = np.concatenate((iquv_rmod_tot, group_iquv(
            i_list=[i_rmod], q_list=[q_rmod],
            u_list=[u_rmod], v_list=[v_rmod])), axis=1)

    i_val = i_rmod-i_mod
    q_val = q_rmod-q_mod
    u_val = u_rmod-u_mod
    v_val = v_rmod-v_mod
    max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
    max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
    max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
    max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
    title = (f"IPRT case {case_name} - depol = {depols[ires]}  - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{depols[ires]:.0f}km - dif ({rmod_name}-{mod_name})")
    select_and_plot_polar_iprt(ref_model, z_alti=altitudes[ires],
                               depol=depols[ires], title=title,
                               force_iquv=[i_val, q_val, u_val, v_val],
                               max_i=max_i, max_q=max_q, max_u=max_u,
                               max_v=max_v, cmap_i='RdBu_r',
                               avoid_plot=avoid_p_dif)

# %% [markdown]
# #### Compute delta_m

# %%
delta_m_a1 = compute_deltam(
    obs=iquv_rmod_tot, mod=iquv_mod_tot, print_res=True)

# %% [markdown]
# ### Case A2

# %% [markdown]
# #### Atmosphere profil

# %%
mol_sca = np.array([0., 0.1])[None, :]
mol_abs= np.array([0., 0.])[None, :]
z = np.array([1., 0.])
wavelength = 550.
pro_a2 = Atm1D(
    'afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs).calc(wavelength)
surf_a2  = LambSurface(alb=AlbedoCst(0.3))

# %% [markdown]
# #### Compute radiances

# %%
vza_min = 100.
vza_max = 180.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

vaa_min = 0.
vaa_max = 180.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

# SMART-G Forward th and phi using local estimate (anticlockwise)
# conversion with vza and vaa MYSTIC (clockwise)
th  = 180.-vza
phi = -vaa
th[th==0] = 1e-6  # avoid problem due to special case of 0
le     = LocalEstimate(th_deg=th, phi_deg=phi)  # , zip=True

sza = 50.
saa = 0.
phi_0 = 180.-saa # SMART-G anticlockwise converted to be consistent with MYSTIC

m_a2f = s_1df.run(th_deg=sza, ph_deg=phi_0, wavelength=550., n_photons=NB_PH,
                  n_loop=min(1e6, round(NB_PH/10.)), atmosphere=pro_a2,
                  output_layers=int(7), le=le, surface=surf_a2, xblock=64,
                  xgrid=1024, beer=1, depo=0.03, stdev=True, seed=SEED)

# %%
m_a2f.to_netcdf(output_folder_path / "iprt_a2_smartg_ref.nc")

# %% [markdown]
# #### Convert into iprt output format

# %%
# Radiances (Forward, U must be multiplied by -1)
m = m_a2f
vza = 180.-m.coords['Zenith angles'].values
vaa = -m.coords['Azimuth angles'].values

filename  = output_folder_path / "iprt_case_a2_smartg_ref.dat"
case_name = "A2"
depols    = [0.03, 0.03]
altitudes      = [0., 1.]
szas      = [50., 50.]
saas      = [0., 0.]

# convert
convert_sgout_to_iprtout(datasets=[m, m], u_signs=[-1, -1],
                         case_name=case_name, depols=depols,
                         altitudes=altitudes, szas=szas, saas=saas,
                         vzas=[180.-vza, vza], vaas=[vaa, vaa],
                         file_name=filename,
                         output_layer=['_down (0+)', '_up (TOA)'])

# %% [markdown]
# #### Comparison with MYSTIC

# %%
filename  = output_folder_path / "iprt_case_a2_smartg_ref.dat"
case_name = "A2"
depols    = [0.03, 0.03]
altitudes      = [0., 1.]
szas      = [50., 50.]
saas      = [0., 0.]

smartg_a2 = pd.read_csv(filename, header=None, sep=r'\s+', dtype=float,
                        comment="#").values
mystic_a2 = pd.read_csv(
    MYSTIC_RES_PATH / "iprt_case_a2_mystic.dat", header=None, sep=r'\s+',
    dtype=float, comment="#").values
model     = smartg_a2
ref_model = mystic_a2
mod_name  = "SMARTG"
rmod_name = "MYSTIC"
l_invth_rmod  = [False, True]
l_invth_mod   = [False, True]
l_u_sign_rmod  =[True, True]
l_u_sign_mod  = [True, True]
l_v_sign_rmod  =[False, False]
l_v_sign_mod  = [False, False]
avoid_p_rmod = False
avoid_p_mod  = True
avoid_p_dif  = False
sym = True

n_res = len(depols)
for ires in range (0, n_res):
    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f}  - "
             f"{altitudes[ires]:.0f}km - {rmod_name}")
    i_rmod, q_rmod, u_rmod, v_rmod = select_and_plot_polar_iprt(
        ref_model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_rmod[ires], change_v_sign=l_v_sign_rmod[ires],
        inv_thetas=l_invth[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_rmod)

    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{altitudes[ires]:.0f}km - {mod_name}")
    i_mod, q_mod, u_mod, v_mod = select_and_plot_polar_iprt(
        model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_mod[ires], change_v_sign=l_v_sign_mod[ires],
        inv_thetas=l_invth[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_mod)

    if (ires == 0):
        iquv_mod_tot = group_iquv(i_list=[i_mod], q_list=[q_mod],
                                  u_list=[u_mod], v_list=[v_mod])
        iquv_rmod_tot = group_iquv(i_list=[i_rmod], q_list=[q_rmod],
                                   u_list=[u_rmod], v_list=[v_rmod])
    else:
        iquv_mod_tot = np.concatenate((iquv_mod_tot, group_iquv(
            i_list=[i_mod], q_list=[q_mod],
            u_list=[u_mod], v_list=[v_mod])), axis=1)
        iquv_rmod_tot = np.concatenate((iquv_rmod_tot, group_iquv(
            i_list=[i_rmod], q_list=[q_rmod],
            u_list=[u_rmod], v_list=[v_rmod])), axis=1)

    i_val = i_rmod-i_mod
    q_val = q_rmod-q_mod
    u_val = u_rmod-u_mod
    v_val = v_rmod-v_mod
    max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
    max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
    max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
    max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
    title = (f"IPRT case {case_name} - depol = {depols[ires]}  - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{depols[ires]:.0f}km - dif ({rmod_name}-{mod_name})")
    select_and_plot_polar_iprt(ref_model, z_alti=altitudes[ires],
                               depol=depols[ires], title=title,
                               force_iquv=[i_val, q_val, u_val, v_val],
                               max_i=max_i, max_q=max_q, max_u=max_u,
                               max_v=max_v, cmap_i='RdBu_r',
                               avoid_plot=avoid_p_dif)

# %% [markdown]
# #### Compute delta_m

# %%
delta_m_a2 = compute_deltam(
    obs=iquv_rmod_tot, mod=iquv_mod_tot, print_res=True)

# %% [markdown]
# ### Case A3

# %% [markdown]
# #### Atmosphere profil

# %%
# z profil
z = np.array([1., 0.])
nz = len(z[1:])

# molecular scattering and absorption
mol_sca = np.array([0., 0.])[None, :]
mol_abs= np.array([0., 0.])[None, :]

# aerosol extinction and single scattering albedo
aer_tau_ext = np.full_like(mol_sca, 0.2, dtype=np.float32)
aer_tau_ext[:, 0] = 0. # dtau TOA equal to 0
aer_ssa = np.full_like(mol_sca, 0.975683, dtype=np.float32)
prof_aer = (aer_tau_ext, aer_ssa)

# aerosol phase matrix
wavelength = np.array([350.])
n_wavelength = len(wavelength)
file_aer_phase = OPT_PROP_PATH / 'waso.mie.cdf'
aer_phase = read_phase(fname=file_aer_phase, wavelength_phase=wavelength,
                       pfgrid=z)
n_theta = aer_phase.shape[-1]
nstk = aer_phase.shape[2]
prof_phases = get_prof_phases(aer_phase, wavelength, z)
# atmosphere profil
pro_a3 = Atm1D(
    'afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs, prof_aer=prof_aer,
    prof_phases=prof_phases).calc(wavelength, phase=False)

# ground surface
surf_a3  = None

# %% [markdown]
# #### Compute radiances

# %%
sza = 40.
saa = 0.
phi_0 = 180.-saa # SMART-G anticlockwise converted to be consistent with MYSTIC

vza_min = 100.
vza_max = 180.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

vaa_min = 0.
vaa_max = 180.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

# SMART-G Forward th and phi using local estimate (anticlockwise)
# conversion with vza and vaa MYSTIC (clockwise)
th  = 180.-vza
phi = -vaa
th[th==0] = 1e-6 # avoid problem due to special case of 0
le     = LocalEstimate(th_deg=th, phi_deg=phi)  # , zip=True

m_a3f = s_1df.run(th_deg=sza, ph_deg=phi_0, wavelength=wavelength,
                  n_photons=NB_PH, n_loop=min(1e6, round(NB_PH/10.)),
                  n_icdf=n_theta, atmosphere=pro_a3, output_layers=int(1),
                  le=le, surface=surf_a3, xblock=64, xgrid=1024, beer=1,
                  depo=0.0, stdev=True, seed=SEED*2)

# %%
m_a3f.to_netcdf(output_folder_path / "iprt_a3_smartg_ref.nc")

# %% [markdown]
# #### Convert into iprt output format

# %%
# Radiances (Forward, U must be multiplied by -1)
m = m_a3f
vza = 180.-m.coords['Zenith angles'].values
vaa = -m.coords['Azimuth angles'].values

filename  = output_folder_path / "iprt_case_a3_smartg_ref.dat"
case_name = "A3"
depols    = [0.0, 0.0]
altitudes      = [0., 1.]
szas      = [40., 40.]
saas      = [0., 0.]

# convert
convert_sgout_to_iprtout(datasets=[m, m], u_signs=[-1, -1],
                         case_name=case_name, depols=depols,
                         altitudes=altitudes, szas=szas, saas=saas,
                         vzas=[vza, vza], vaas=[vaa, vaa], file_name=filename,
                         output_layer=['_down (0+)', '_up (TOA)'])

# %% [markdown]
# #### Comparison with MYSTIC

# %%
filename     = output_folder_path / "iprt_case_a3_smartg_ref.dat"
case_name    = "A3"
depols       = [0.0, 0.0]
altitudes         = [0., 1.]
szas         = [40., 40.]
saas         = [0., 0.]

smartg_a3 = pd.read_csv(filename, header=None, sep=r'\s+', dtype=float,
                        comment="#").values
mystic_a3 = pd.read_csv(
    MYSTIC_RES_PATH / "iprt_case_a3_mystic.dat", header=None, sep=r'\s+',
    dtype=float, comment="#").values
model     = smartg_a3
ref_model = mystic_a3
mod_name  = "SMARTG"
rmod_name = "MYSTIC"
l_invth_rmod  = [False, True]
l_invth_mod   = [True, True]
l_u_sign_rmod  =[True, True]
l_u_sign_mod  = [True, True]
l_v_sign_rmod  =[True, True]
l_v_sign_mod  = [True, True]
avoid_p_rmod = False
avoid_p_mod  = True
avoid_p_dif  = False
sym = True

n_res = len(depols)
for ires in range (0, n_res):
    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f}  - "
             f"{altitudes[ires]:.0f}km - {rmod_name}")
    i_rmod, q_rmod, u_rmod, v_rmod = select_and_plot_polar_iprt(
        ref_model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_rmod[ires], change_v_sign=l_v_sign_rmod[ires],
        inv_thetas=l_invth_rmod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_rmod)

    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{altitudes[ires]:.0f}km - {mod_name}")
    i_mod, q_mod, u_mod, v_mod = select_and_plot_polar_iprt(
        model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_mod[ires], change_v_sign=l_v_sign_mod[ires],
        inv_thetas=l_invth_mod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_mod)

    if (ires == 0):
        iquv_mod_tot = group_iquv(i_list=[i_mod], q_list=[q_mod],
                                  u_list=[u_mod], v_list=[v_mod])
        iquv_rmod_tot = group_iquv(i_list=[i_rmod], q_list=[q_rmod],
                                   u_list=[u_rmod], v_list=[v_rmod])
    else:
        iquv_mod_tot = np.concatenate((iquv_mod_tot, group_iquv(
            i_list=[i_mod], q_list=[q_mod],
            u_list=[u_mod], v_list=[v_mod])), axis=1)
        iquv_rmod_tot = np.concatenate((iquv_rmod_tot, group_iquv(
            i_list=[i_rmod], q_list=[q_rmod],
            u_list=[u_rmod], v_list=[v_rmod])), axis=1)

    i_val = i_rmod-i_mod
    q_val = q_rmod-q_mod
    u_val = u_rmod-u_mod
    v_val = v_rmod-v_mod
    max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
    max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
    max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
    max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
    title = (f"IPRT case {case_name} - depol = {depols[ires]}  - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{depols[ires]:.0f}km - dif ({rmod_name}-{mod_name})")
    select_and_plot_polar_iprt(ref_model, z_alti=altitudes[ires],
                               depol=depols[ires], title=title,
                               force_iquv=[i_val, q_val, u_val, v_val],
                               max_i=max_i, max_q=max_q, max_u=max_u,
                               max_v=max_v, cmap_i='RdBu_r',
                               avoid_plot=avoid_p_dif)

# %% [markdown]
# #### Compute delta_m

# %%
delta_m_a3 = compute_deltam(
    obs=iquv_rmod_tot, mod=iquv_mod_tot, print_res=True)

# %% [markdown]
# ### Case A4

# %% [markdown]
# #### Atmosphere profil

# %%
# z profil
z = np.array([1., 0.])
nz = len(z[1:])

# molecular scattering and absorption
mol_sca = np.array([0., 0.])[None, :]
mol_abs= np.array([0., 0.])[None, :]

# aerosol extinction and single scattering albedo
aer_tau_ext = np.full_like(mol_sca, 0.2, dtype=np.float32)
aer_tau_ext[:, 0] = 0. # dtau TOA equal to 0
aer_ssa = np.full_like(mol_sca, 0.787581, dtype=np.float32)
prof_aer = (aer_tau_ext, aer_ssa)

# aerosol phase matrix
wavelength = np.array([350.])
n_wavelength = len(wavelength)

file_aer_phase = OPT_PROP_PATH / 'sizedistr_spheroid.cdf'
aer_phase = read_phase(fname=file_aer_phase, wavelength_phase=wavelength,
                       pfgrid=z)
n_theta = 18001#aer_phase.shape[-1]
aer_phase = aer_phase.interp(**{'theta_atm': np.linspace(
    0, 180, n_theta)}, method='linear')
nstk = aer_phase.shape[2]
prof_phases = get_prof_phases(aer_phase, wavelength, z)
# atmosphere profil
pro_a4 = Atm1D(
    'afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs, prof_aer=prof_aer,
    prof_phases=prof_phases).calc(wavelength, phase=False)

# ground surface
surf_a4  = None

# %% [markdown]
# #### Compute radiances

# %%
sza = 40.
saa = 0.
phi_0 = 180.-saa # SMART-G anticlockwise converted to be consistent with MYSTIC

vza_min = 100.
vza_max = 180.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

vaa_min = 0.
vaa_max = 180.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

# SMART-G Forward th and phi using local estimate (anticlockwise)
# conversion with vza and vaa MYSTIC (clockwise)
th  = 180.-vza
phi = -vaa
th[th==0] = 1e-6 # avoid problem due to special case of 0
le     = LocalEstimate(th_deg=th, phi_deg=phi)  # , zip=True

m_a4f = s_1df.run(th_deg=sza, ph_deg=phi_0, wavelength=wavelength,
                  n_photons=NB_PH, n_loop=min(1e6, round(NB_PH/100.)),
                  n_icdf=n_theta, atmosphere=pro_a4, output_layers=int(1),
                  le=le, surface=surf_a4, xblock=64, xgrid=1024, beer=1,
                  depo=0.0, stdev=True, seed=SEED)

# %%
m_a4f.to_netcdf(output_folder_path / "iprt_a4_smartg_ref.nc")

# %% [markdown]
# #### Convert into iprt output format

# %%
# Radiances (Forward, U must be multiplied by -1)
m = m_a4f
vza = 180.-m.coords['Zenith angles'].values
vaa = -m.coords['Azimuth angles'].values

filename  = output_folder_path / "iprt_case_a4_smartg_ref.dat"
case_name = "A4"
depols    = [0.0, 0.0]
altitudes      = [0., 1.]
szas      = [40., 40.]
saas      = [0., 0.]

# convert
convert_sgout_to_iprtout(datasets=[m, m], u_signs=[-1, -1],
                         case_name=case_name, depols=depols,
                         altitudes=altitudes, szas=szas, saas=saas,
                         vzas=[vza, vza], vaas=[vaa, vaa], file_name=filename,
                         output_layer=['_down (0+)', '_up (TOA)'])

# %% [markdown]
# #### Comparison with MYSTIC

# %%
filename     = output_folder_path / "iprt_case_a4_smartg_ref.dat"
case_name    = "A4"
depols       = [0.0, 0.0]
altitudes         = [0., 1.]
szas         = [40., 40.]
saas         = [0., 0.]

smartg_a4 = pd.read_csv(filename, header=None, sep=r'\s+', dtype=float,
                        comment="#").values
mystic_a4 = pd.read_csv(
    MYSTIC_RES_PATH / "iprt_case_a4_mystic.dat", header=None, sep=r'\s+',
    dtype=float, comment="#").values
model     = smartg_a4
ref_model = mystic_a4
mod_name  = "SMARTG"
rmod_name = "MYSTIC"
l_invth_rmod  = [False, True]
l_invth_mod   = [True, True]
l_u_sign_rmod  =[True, True]
l_u_sign_mod  = [True, True]
l_v_sign_rmod  =[True, True]
l_v_sign_mod  = [True, True]
avoid_p_rmod = False
avoid_p_mod  = True
avoid_p_dif  = False
sym = True

n_res = len(depols)
for ires in range (0, n_res):
    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f}  - "
             f"{altitudes[ires]:.0f}km - {rmod_name}")
    i_rmod, q_rmod, u_rmod, v_rmod = select_and_plot_polar_iprt(
        ref_model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_rmod[ires], change_v_sign=l_v_sign_rmod[ires],
        inv_thetas=l_invth_rmod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_rmod)

    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{altitudes[ires]:.0f}km - {mod_name}")
    i_mod, q_mod, u_mod, v_mod = select_and_plot_polar_iprt(
        model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_mod[ires], change_v_sign=l_v_sign_mod[ires],
        inv_thetas=l_invth_mod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_mod)

    if (ires == 0):
        iquv_mod_tot = group_iquv(i_list=[i_mod], q_list=[q_mod],
                                  u_list=[u_mod], v_list=[v_mod])
        iquv_rmod_tot = group_iquv(i_list=[i_rmod], q_list=[q_rmod],
                                   u_list=[u_rmod], v_list=[v_rmod])
    else:
        iquv_mod_tot = np.concatenate((iquv_mod_tot, group_iquv(
            i_list=[i_mod], q_list=[q_mod],
            u_list=[u_mod], v_list=[v_mod])), axis=1)
        iquv_rmod_tot = np.concatenate((iquv_rmod_tot, group_iquv(
            i_list=[i_rmod], q_list=[q_rmod],
            u_list=[u_rmod], v_list=[v_rmod])), axis=1)

    i_val = i_rmod-i_mod
    q_val = q_rmod-q_mod
    u_val = u_rmod-u_mod
    v_val = v_rmod-v_mod
    max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
    max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
    max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
    max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
    title = (f"IPRT case {case_name} - depol = {depols[ires]}  - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{depols[ires]:.0f}km - dif ({rmod_name}-{mod_name})")
    select_and_plot_polar_iprt(ref_model, z_alti=altitudes[ires],
                               depol=depols[ires], title=title,
                               force_iquv=[i_val, q_val, u_val, v_val],
                               max_i=max_i, max_q=max_q, max_u=max_u,
                               max_v=max_v, cmap_i='RdBu_r',
                               avoid_plot=avoid_p_dif)

# %% [markdown]
# #### Compute delta_m

# %%
delta_m_a4 = compute_deltam(
    obs=iquv_rmod_tot, mod=iquv_mod_tot, print_res=True)

# %% [markdown]
# ### Case A5

# %% [markdown]
# #### Atmosphere profil

# %%
from smartg.atmosphere import Cloud

# %%
# z profil
z = np.array([1., 0.])
nz = len(z[1:])

# molecular scattering and absorption
mol_sca = np.array([0., 0.])[None, :]
mol_abs= np.array([0., 0.])[None, :]

# cloud extinction and single scattering albedo
cld_tau_ext = np.full_like(mol_sca, 5., dtype=np.float32)
cld_tau_ext[:, 0] = 0. # dtau TOA equal to 0
cld_ssa = np.full_like(mol_sca, 0.999979, dtype=np.float32)
prof_aer = (cld_tau_ext, cld_ssa)

# cloud phase matrix
wavelength = np.array([800.])
n_wavelength = len(wavelength)
file_cld_phase = OPT_PROP_PATH / 'watercloud.mie.cdf'
cld_phase = read_phase(fname=file_cld_phase)
n_theta = cld_phase.shape[-1]
nstk = cld_phase.shape[2]
prof_phases = get_prof_phases(cld_phase, wavelength, z)
# atmosphere profil

pro_a5 = Atm1D('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs,
               prof_aer=prof_aer, prof_phases=prof_phases).calc(wavelength)

# ground surface
surf_a5  = None

# %% [markdown]
# #### Compute radiances (principal plane)

# %%
sza = 50.
saa = 0.
phi_0 = 180.-saa # SMART-G anticlockwise converted to be consistent with MYSTIC

vza_min = 100.
vza_max = 180.
vza_inc = 1.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

vaa = np.array([0., 180.])

# SMART-G Forward th and phi using local estimate (anticlockwise)
# conversion with vza and vaa MYSTIC (clockwise)
th  = 180.-vza
phi = -vaa
th[th==0] = 1e-6 # avoid problem due to special case of 0
le     = LocalEstimate(th_deg=th, phi_deg=phi)  # , zip=True

m_a5f_pp = s_1df.run(th_deg=sza, ph_deg=phi_0, wavelength=wavelength,
                     n_photons=NB_PH, n_loop=min(1e6, round(NB_PH/10.)),
                     n_icdf=n_theta, atmosphere=pro_a5, output_layers=int(7),
                     le=le, surface=surf_a5, xblock=64, xgrid=1024, beer=1,
                     depo=0.03, stdev=True, seed=SEED)

# %%
m_a5f_pp.to_netcdf(output_folder_path / "iprt_a5_smartg_pp_ref.nc")

# %% [markdown]
# #### Convert into iprt output format (principal plane)

# %%
# Radiances (Forward, U must be multiplied by -1)
m = m_a5f_pp
vza = 180.-m.coords['Zenith angles'].values
vaa = -m.coords['Azimuth angles'].values

filename  = output_folder_path / "iprt_case_a5_smartg_pp_ref.dat"
case_name = "A5_pp"
depols    = [0.03, 0.03]
altitudes      = [0., 1.]
szas      = [50., 50.]
saas      = [0., 0.]

# convert
convert_sgout_to_iprtout(datasets=[m, m], u_signs=[-1, -1],
                         case_name=case_name, depols=depols,
                         altitudes=altitudes, szas=szas, saas=saas,
                         vzas=[vza, vza], vaas=[vaa, vaa], file_name=filename,
                         output_layer=['_down (0+)', '_up (TOA)'])

# %% [markdown]
# #### Comparison with MYSTIC (principal plane)

# %%
filename  = output_folder_path / "iprt_case_a5_smartg_pp_ref.dat"
case_name = "A5_pp"
depols    = [0.03, 0.03]
altitudes      = [0., 1.]

smartg_a5_pp = pd.read_csv(filename, header=None, sep=r'\s+', dtype=float,
                           comment="#").values
mystic_a5_pp = pd.read_csv(
    MYSTIC_RES_PATH / "iprt_case_a5_pp_mystic.dat", header=None, sep=r'\s+',
    dtype=float, comment="#").values
model        = smartg_a5_pp
ref_model    = mystic_a5_pp
mod_name     = "SMARTG"
rmod_name    = "MYSTIC"

l_invth_rmod  = [True, True]
l_invth_mod   = [True, True]
l_u_sign_rmod  =[False, False]
l_u_sign_mod  = [False, False]

l_iquv_ymin = [[0., -3e-3, -1.5e-4, -2e-5], [0., -2e-2, -1.2e-4, -1e-5]]
l_iquv_ymax = [[3.5, 4e-3, 2e-4, 3e-5], [2.5e-1, 1.5e-2, 6e-5, 1e-5]]

quantity = ['transmittance', 'reflectance']

n_res = len(depols)
for ires in range (0, n_res):
    (i_rmod, q_rmod, u_rmod, v_rmod,
     i_std_rmod, q_std_rmod, u_std_rmod,
     v_std_rmod) = select_iprt_iquv(
        ref_model, altitudes[ires], change_u_sign=l_u_sign_rmod[ires],
        inv_thetas=l_invth_rmod[ires], i_index=5, va_index=3, phi_index=4,
        z_index=0, stdev=True)
    (i_mod, q_mod, u_mod, v_mod,
     i_std_mod, q_std_mod, u_std_mod,
     v_std_mod) = select_iprt_iquv(
        model, altitudes[ires], change_u_sign=l_u_sign_mod[ires],
        inv_thetas=l_invth_mod[ires], stdev=True)

    vza = np.unique(model[:, 4])
    vza_n = np.sort(np.concatenate((vza-180, 180-vza)))
    N_VZA = round(i_mod.shape[0]*2)

    # ref_model IQUV and stdev IQUV
    iquv_pp_rmod = np.zeros((4, N_VZA), dtype=np.float32)
    iquv_pp_rmod[0, :]=np.concatenate((i_rmod[:, 1], i_rmod[::-1, 0]))
    iquv_pp_rmod[1, :]=np.concatenate((q_rmod[:, 1], q_rmod[::-1, 0]))
    iquv_pp_rmod[2, :]=np.concatenate((u_rmod[:, 1], u_rmod[::-1, 0]))
    iquv_pp_rmod[3, :]=np.concatenate((v_rmod[:, 1], v_rmod[::-1, 0]))
    iquv_std_pp_rmod = np.zeros((4, N_VZA), dtype=np.float32)
    iquv_std_pp_rmod[0, :]=np.concatenate((i_std_rmod[:, 1],
                                           i_std_rmod[::-1, 0]))
    iquv_std_pp_rmod[1, :]=np.concatenate((q_std_rmod[:, 1],
                                           q_std_rmod[::-1, 0]))
    iquv_std_pp_rmod[2, :]=np.concatenate((u_std_rmod[:, 1],
                                           u_std_rmod[::-1, 0]))
    iquv_std_pp_rmod[3, :]=np.concatenate((v_std_rmod[:, 1],
                                           v_std_rmod[::-1, 0]))

    # model IQUV and stdev IQUV
    iquv_pp_mod = np.zeros((4, N_VZA), dtype=np.float32)
    iquv_pp_mod[0, :]=np.concatenate((i_mod[:, 1], i_mod[::-1, 0]))
    iquv_pp_mod[1, :]=np.concatenate((q_mod[:, 1], q_mod[::-1, 0]))
    iquv_pp_mod[2, :]=np.concatenate((u_mod[:, 1], u_mod[::-1, 0]))
    iquv_pp_mod[3, :]=np.concatenate((v_mod[:, 1], v_mod[::-1, 0]))
    iquv_std_pp_mod = np.zeros((4, N_VZA), dtype=np.float32)
    iquv_std_pp_mod[0, :]=np.concatenate((i_std_mod[:, 1], i_std_mod[::-1, 0]))
    iquv_std_pp_mod[1, :]=np.concatenate((q_std_mod[:, 1], q_std_mod[::-1, 0]))
    iquv_std_pp_mod[2, :]=np.concatenate((u_std_mod[:, 1], u_std_mod[::-1, 0]))
    iquv_std_pp_mod[3, :]=np.concatenate((v_std_mod[:, 1], v_std_mod[::-1, 0]))

    plot_iprt_radiances(
        iquv_obs=iquv_pp_rmod, iquv_mod=iquv_pp_mod,
        iquv_std_obs=iquv_std_pp_rmod, iquv_std_mod=iquv_std_pp_mod,
        xaxis=vza_n, xlabel='VZA [deg]', iquv_ymin=l_iquv_ymin[ires],
        iquv_ymax=l_iquv_ymax[ires],
        title=f'{quantity[ires]}  {rmod_name}-red {mod_name}-blue')

    if (ires == 0):
        iquv_pp_mod_tot = iquv_pp_mod.copy()
        iquv_pp_rmod_tot = iquv_pp_rmod.copy()
    else:
        iquv_pp_mod_tot = np.concatenate(
            (iquv_pp_mod_tot, iquv_pp_mod), axis=1)
        iquv_pp_rmod_tot = np.concatenate((iquv_pp_rmod_tot, iquv_pp_rmod),
                                          axis=1)

# %% [markdown]
# #### Compute delta_m (principal plane)

# %%
delta_m_a5_pp = compute_deltam(iquv_pp_rmod_tot, iquv_pp_mod_tot)

# %% [markdown]
# #### Compute radiances (almucantar)

# %%
sza = 50.
saa = 0.
phi_0 = 180.-saa # SMART-G anticlockwise converted to be consistent with MYSTIC

vza = np.array([130.])

vaa_min = 0.
vaa_max = 180.
vaa_inc = 1.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

# SMART-G Forward th and phi using local estimate (anticlockwise)
# conversion with vza and vaa MYSTIC (clockwise)
th  = 180.-vza
phi = -vaa
th[th==0] = 1e-6 # avoid problem due to special case of 0
le     = LocalEstimate(th_deg=th, phi_deg=phi)

m_a5f_al = s_1df.run(th_deg=sza, ph_deg=phi_0, wavelength=wavelength,
                     n_photons=NB_PH, n_loop=min(1e6, round(NB_PH/10.)),
                     n_icdf=n_theta, atmosphere=pro_a5, output_layers=int(7),
                     le=le, surface=surf_a5, xblock=64, xgrid=1024, beer=1,
                     depo=0.03, stdev=True, seed=SEED)

# %%
m_a5f_al.to_netcdf(output_folder_path / "iprt_a5_smartg_al_ref.nc")

# %% [markdown]
# #### Convert into iprt output format (almucantar)

# %%
# Radiances (Forward, U must be multiplied by -1)
m = m_a5f_al
vza = 180.-m.coords['Zenith angles'].values
vaa = -m.coords['Azimuth angles'].values

filename  = output_folder_path / "iprt_case_a5_smartg_al_ref.dat"
case_name = "A5_al"
depols    = [0.03, 0.03]
altitudes      = [0., 1.]
szas      = [50., 50.]
saas      = [0., 0.]

# convert
convert_sgout_to_iprtout(datasets=[m, m], u_signs=[-1, -1],
                         case_name=case_name, depols=depols,
                         altitudes=altitudes, szas=szas, saas=saas,
                         vzas=[vza, vza], vaas=[vaa, vaa], file_name=filename,
                         output_layer=['_down (0+)', '_up (TOA)'])

# %% [markdown]
# #### Comparison with MYSTIC (almucantar)

# %%
filename  = output_folder_path / "iprt_case_a5_smartg_al_ref.dat"
case_name = "A5_al"
depols    = [0.03, 0.03]
altitudes      = [0., 1.]

smartg_a5_al = pd.read_csv(filename, header=None, sep=r'\s+', dtype=float,
                           comment="#").values
mystic_a5_al = pd.read_csv(
    MYSTIC_RES_PATH / "iprt_case_a5_al_mystic.dat", header=None, sep=r'\s+',
    dtype=float, comment="#").values
model        = smartg_a5_al
ref_model    = mystic_a5_al
mod_name     = "SMARTG"
rmod_name    = "MYSTIC"

l_invth_rmod  = [True, True]
l_invth_mod   = [True, True]
l_u_sign_rmod  =[False, False]
l_u_sign_mod  = [False, False]

l_iquv_ymin = [[0., -3.5e-3, -3e-3, -1.5e-5], [6e-2, -1e-2, -2e-3, -5e-5]]
l_iquv_ymax = [[3.5, 5e-4, 5e-4, 2.5e-5], [1.2e-1, 2e-2, 1.2e-2, 2e-5]]

quantity = ['transmittance', 'reflectance']

n_res = len(depols)
for ires in range (0, n_res):
    (i_rmod, q_rmod, u_rmod, v_rmod,
     i_std_rmod, q_std_rmod, u_std_rmod,
     v_std_rmod) = select_iprt_iquv(
        ref_model, altitudes[ires], change_u_sign=l_u_sign_rmod[ires],
        inv_thetas=l_invth_rmod[ires], i_index=5, va_index=3, phi_index=4,
        z_index=0, stdev=True)
    (i_mod, q_mod, u_mod, v_mod,
     i_std_mod, q_std_mod, u_std_mod,
     v_std_mod) = select_iprt_iquv(
        model, altitudes[ires], change_u_sign=l_u_sign_mod[ires],
        inv_thetas=l_invth_mod[ires], stdev=True)

    vaa = np.unique(smartg_a5_al[:, 5])
    vaa_n = vaa
    N_VAA = round(i_mod.shape[1])

    # ref_model IQUV and stdev IQUV
    iquv_al_rmod = np.zeros((4, N_VAA), dtype=np.float32)
    iquv_al_rmod[0, :]=i_rmod[0, :]
    iquv_al_rmod[1, :]=q_rmod[0, :]
    iquv_al_rmod[2, :]=u_rmod[0, :]
    iquv_al_rmod[3, :]=v_rmod[0, :]
    iquv_std_al_rmod = np.zeros((4, N_VAA), dtype=np.float32)
    iquv_std_al_rmod[0, :]=i_std_rmod[0, :]
    iquv_std_al_rmod[1, :]=q_std_rmod[0, :]
    iquv_std_al_rmod[2, :]=u_std_rmod[0, :]
    iquv_std_al_rmod[3, :]=v_std_rmod[0, :]

    # model IQUV and stdev IQUV
    iquv_al_mod = np.zeros((4, N_VAA), dtype=np.float32)
    iquv_al_mod[0, :]=i_mod[0, :]
    iquv_al_mod[1, :]=q_mod[0, :]
    iquv_al_mod[2, :]=u_mod[0, :]
    iquv_al_mod[3, :]=v_mod[0, :]
    iquv_std_al_mod = np.zeros((4, N_VAA), dtype=np.float32)
    iquv_std_al_mod[0, :]=i_std_mod[0, :]
    iquv_std_al_mod[1, :]=q_std_mod[0, :]
    iquv_std_al_mod[2, :]=u_std_mod[0, :]
    iquv_std_al_mod[3, :]=v_std_mod[0, :]

    plot_iprt_radiances(
        iquv_obs=iquv_al_rmod, iquv_mod=iquv_al_mod,
        iquv_std_obs=iquv_std_al_rmod, iquv_std_mod=iquv_std_al_mod,
        xaxis=vaa_n, xlabel='VZA [deg]', iquv_ymin=l_iquv_ymin[ires],
        iquv_ymax=l_iquv_ymax[ires],
        title=f'{quantity[ires]}  {rmod_name}-red {mod_name}-blue')

    if (ires == 0):
        iquv_al_mod_tot = iquv_al_mod.copy()
        iquv_al_rmod_tot = iquv_al_rmod.copy()
    else:
        iquv_al_mod_tot = np.concatenate(
            (iquv_al_mod_tot, iquv_al_mod), axis=1)
        iquv_al_rmod_tot = np.concatenate((iquv_al_rmod_tot, iquv_al_rmod),
                                          axis=1)

# %% [markdown]
# #### Compute delta_m (almucantar)

# %%
delta_m_a5_al = compute_deltam(iquv_al_rmod_tot, iquv_al_mod_tot)

# %% [markdown]
# ### Case A6

# %% [markdown]
# #### Atmosphere profil

# %%
mol_sca = np.array([0., 0.1])[None, :]
mol_abs= np.array([0., 0.])[None, :]
z = np.array([1., 0.])
wavelength = 550.
pro_a6 = Atm1D('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs).calc(550.)
surf_a6  = RoughSurface(wind=2., brdf=True, wave_shadow=True, nh2o=1.33)

# %% [markdown]
# #### Compute radiances

# %%
vza_min = 100.
vza_max = 180.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

vaa_min = 0.
vaa_max = 180.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

# SMART-G Forward th and phi using local estimate (anticlockwise)
# conversion with vza and vaa MYSTIC (clockwise)
th  = 180.-vza
phi = -vaa
th[th==0] = 1e-6  # avoid problem due to special case of 0
le     = LocalEstimate(th_deg=th, phi_deg=phi)  # , zip=True

sza = 45.0
saa = 0.
phi_0 = 180.-saa # SMART-G anticlockwise converted to be consistent with MYSTIC

m_a6f = s_1df.run(th_deg=sza, ph_deg=phi_0, wavelength=wavelength,
                  n_photons=NB_PH, n_loop=min(1e6, round(NB_PH/10.)),
                  atmosphere=pro_a6, output_layers=int(7), le=le,
                  surface=surf_a6, xblock=64, xgrid=1024, beer=1, depo=0.03,
                  stdev=True, seed=SEED)

# %%
m_a6f.to_netcdf(output_folder_path / "iprt_a6_smartg_ref.nc")

# %% [markdown]
# #### Convert into iprt output format

# %%
# Radiances
vza = 180.-m_a6f.coords['Zenith angles'].values
vaa = -m_a6f.coords['Azimuth angles'].values

filename  = output_folder_path / "iprt_case_a6_smartg_ref.dat"
case_name = "A6"
depols    = [0.03, 0.03]
altitudes      = [0., 1.]
szas      = [45., 45.]
saas      = [0., 0.]

# convert
convert_sgout_to_iprtout(datasets=[m_a6f, m_a6f], u_signs=[-1, -1],
                         case_name="A6", depols=depols, altitudes=altitudes,
                         szas=szas, saas=saas, vzas=[180.-vza, vza],
                         vaas=[vaa, vaa], file_name=filename,
                         output_layer=['_down (0+)', '_up (TOA)'])

# %% [markdown]
# #### Comparison with MYSTIC

# %%
filename     = output_folder_path / "iprt_case_a6_smartg_ref.dat"
case_name    = "A6"
depols       = [0.03, 0.03]
altitudes         = [0., 1.]
szas         = [45., 45.]
saas         = [0., 0.]

smartg_a6 = pd.read_csv(filename, header=None, sep=r'\s+', dtype=float,
                        comment="#").values
mystic_a6 = pd.read_csv(
    MYSTIC_RES_PATH / "iprt_case_a6_mystic.dat", header=None, sep=r'\s+',
    dtype=float, comment="#").values
model     = smartg_a6
ref_model = mystic_a6
mod_name  = "SMARTG"
rmod_name = "MYSTIC"
l_invth_rmod  = [False, True]
l_invth_mod   = [False, True]
l_u_sign_rmod  =[True, True]
l_u_sign_mod  = [True, True]
l_v_sign_rmod  =[True, True]
l_v_sign_mod  = [True, True]
avoid_p_rmod = False
avoid_p_mod  = True
avoid_p_dif  = False
sym = True

n_res = len(depols)
for ires in range (0, n_res):
    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f}  - "
             f"{altitudes[ires]:.0f}km - {rmod_name}")
    i_rmod, q_rmod, u_rmod, v_rmod = select_and_plot_polar_iprt(
        ref_model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_rmod[ires], change_v_sign=l_v_sign_rmod[ires],
        inv_thetas=l_invth_rmod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_rmod)

    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{altitudes[ires]:.0f}km - {mod_name}")
    i_mod, q_mod, u_mod, v_mod = select_and_plot_polar_iprt(
        model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_mod[ires], change_v_sign=l_v_sign_mod[ires],
        inv_thetas=l_invth_mod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_mod)

    if (ires == 0):
        iquv_mod_tot = group_iquv(i_list=[i_mod], q_list=[q_mod],
                                  u_list=[u_mod], v_list=[v_mod])
        iquv_rmod_tot = group_iquv(i_list=[i_rmod], q_list=[q_rmod],
                                   u_list=[u_rmod], v_list=[v_rmod])
    else:
        iquv_mod_tot = np.concatenate((iquv_mod_tot, group_iquv(
            i_list=[i_mod], q_list=[q_mod],
            u_list=[u_mod], v_list=[v_mod])), axis=1)
        iquv_rmod_tot = np.concatenate((iquv_rmod_tot, group_iquv(
            i_list=[i_rmod], q_list=[q_rmod],
            u_list=[u_rmod], v_list=[v_rmod])), axis=1)

    i_val = i_rmod-i_mod
    q_val = q_rmod-q_mod
    u_val = u_rmod-u_mod
    v_val = v_rmod-v_mod
    max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
    max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
    max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
    max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
    title = (f"IPRT case {case_name} - depol = {depols[ires]}  - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{depols[ires]:.0f}km - dif ({rmod_name}-{mod_name})")
    select_and_plot_polar_iprt(ref_model, z_alti=altitudes[ires],
                               depol=depols[ires], title=title,
                               force_iquv=[i_val, q_val, u_val, v_val],
                               max_i=max_i, max_q=max_q, max_u=max_u,
                               max_v=max_v, cmap_i='RdBu_r',
                               avoid_plot=avoid_p_dif)

# %% [markdown]
# #### compute delta_m

# %%
delta_m_a6 = compute_deltam(
    obs=iquv_rmod_tot, mod=iquv_mod_tot, print_res=True)

# %% [markdown]
# ## Test cases with realistic atmospheric profiles

# %% [markdown]
# ### Case B1
# - In the paper and plots are not considered vza = 95 deg (30km) and 85 deg (0km),
# mistake in the description at (https://www.meteo.physik.uni-muenchen.de/~iprt/doku.php?id=intercomparisons:b1_rayleigh).

# %% [markdown]
# #### Atmosphere profil

# %%
mol_sca_filename  = OPT_PROP_PATH / 'tau_rayleigh_450.dat'
wavelength = 450.
z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0],
                           dtype=float, skiprows=1, sep=r'\s+').values)
zs = len(z)
sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float,
                  skiprows=1, sep=r'\s+').values.reshape(1, zs)
pro_b1 = Atm1D('afglt', grid=z, prof_ray=sca,
               prof_abs=np.zeros_like(sca)).calc(wavelength)
surf_b1 = None

# %% [markdown]
# #### Compute radiances at 0 and 30 km (BOA and TOA)

# %%
vza_min = 100.
vza_max = 180.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

vaa_min = 0.
vaa_max = 180.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

# SMART-G Forward th and phi using local estimate (anticlockwise)
# conversion with vza and vaa MYSTIC (clockwise)
th  = 180.-vza
phi = -vaa
th[th==0] = 1e-6  # avoid problem due to special case of 0
le     = LocalEstimate(th_deg=th, phi_deg=phi)  # , zip=True

sza = 60.0
saa = 0.
phi_0 = 180.-saa # SMART-G anticlockwise converted to be consistent with MYSTIC

m_b1f = s_1df.run(th_deg=sza, ph_deg=phi_0, wavelength=wavelength,
                  n_photons=NB_PH, n_loop=min(1e6, round(NB_PH/10.)),
                  atmosphere=pro_b1, output_layers=int(7), le=le,
                  surface=surf_b1, xblock=64, xgrid=1024, beer=1, depo=0.03,
                  stdev=True, seed=SEED)

# %%
m_b1f.to_netcdf(output_folder_path / "iprt_b1_smartg_ref.nc")

# %% [markdown]
# #### Convert into iprt output form

# %%
# Radiances
m = m_b1f
vza = 180.-m.coords['Zenith angles'].values
vaa = -m.coords['Azimuth angles'].values

filename  = output_folder_path / "iprt_case_b1_smartg_ref.dat"
case_name = "B1"
depols    = [0.03, 0.03]
altitudes      = [0., 30.]
szas      = [60., 60.]
saas      = [0., 0.]

# convert
convert_sgout_to_iprtout(datasets=[m, m], u_signs=[-1, -1],
                         case_name=case_name, depols=depols,
                         altitudes=altitudes, szas=szas, saas=saas,
                         vzas=[180.-vza, vza], vaas=[vaa, vaa],
                         file_name=filename,
                         output_layer=['_down (0+)', '_up (TOA)'])

# %% [markdown]
# #### Comparison with MYSTIC

# %%
filename     = output_folder_path / "iprt_case_b1_smartg_ref.dat"
case_name    = "B1"
depols       = [0.03, 0.03]
altitudes         = [0., 30.]
szas         = [60., 60.]
saas         = [0., 0.]
vzas         = [np.round(180.-vza), np.round(vza)]

smartg_b1 = pd.read_csv(filename, header=None, sep=r'\s+', dtype=float,
                        comment="#").values
mystic_b1 = pd.read_csv(
    MYSTIC_RES_PATH / "iprt_case_b1_mystic.dat", header=None, sep=r'\s+',
    dtype=float, comment="#").values
model     = smartg_b1
ref_model = mystic_b1
mod_name  = "SMARTG"
rmod_name = "MYSTIC"
l_invth_rmod  = [False, True]
l_invth_mod   = [False, True]
l_u_sign_rmod  =[True, True]
l_u_sign_mod  = [True, True]
l_v_sign_rmod  =[True, True]
l_v_sign_mod  = [True, True]
avoid_p_rmod = False
avoid_p_mod  = True
avoid_p_dif  = False
sym = True

n_res = len(depols)
for ires in range (0, n_res):
    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f}  - "
             f"{altitudes[ires]:.0f}km - {rmod_name}")
    i_rmod, q_rmod, u_rmod, v_rmod = select_and_plot_polar_iprt(
        ref_model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_rmod[ires], change_v_sign=l_v_sign_rmod[ires],
        inv_thetas=l_invth_rmod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_rmod, thetas=vzas[ires])

    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{altitudes[ires]:.0f}km - {mod_name}")
    i_mod, q_mod, u_mod, v_mod = select_and_plot_polar_iprt(
        model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_mod[ires], change_v_sign=l_v_sign_mod[ires],
        inv_thetas=l_invth_mod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_mod, thetas=vzas[ires])

    if (ires == 0):
        iquv_mod_tot = group_iquv(i_list=[i_mod], q_list=[q_mod],
                                  u_list=[u_mod], v_list=[v_mod])
        iquv_rmod_tot = group_iquv(i_list=[i_rmod], q_list=[q_rmod],
                                   u_list=[u_rmod], v_list=[v_rmod])
    else:
        iquv_mod_tot = np.concatenate((iquv_mod_tot, group_iquv(
            i_list=[i_mod], q_list=[q_mod],
            u_list=[u_mod], v_list=[v_mod])), axis=1)
        iquv_rmod_tot = np.concatenate((iquv_rmod_tot, group_iquv(
            i_list=[i_rmod], q_list=[q_rmod],
            u_list=[u_rmod], v_list=[v_rmod])), axis=1)

    i_val = i_rmod-i_mod
    q_val = q_rmod-q_mod
    u_val = u_rmod-u_mod
    v_val = v_rmod-v_mod
    max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
    max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
    max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
    max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
    title = (f"IPRT case {case_name} - depol = {depols[ires]}  - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{depols[ires]:.0f}km - dif ({rmod_name}-{mod_name})")
    select_and_plot_polar_iprt(ref_model, z_alti=altitudes[ires],
                               depol=depols[ires], title=title,
                               force_iquv=[i_val, q_val, u_val, v_val],
                               max_i=max_i, max_q=max_q, max_u=max_u,
                               max_v=max_v, cmap_i='RdBu_r',
                               avoid_plot=avoid_p_dif, thetas=vzas[ires])

# %% [markdown]
# #### Compute delta_m

# %%
delta_m_b1 = compute_deltam(
    obs=iquv_rmod_tot, mod=iquv_mod_tot, print_res=True)

# %% [markdown]
# ### Case B2

# %% [markdown]
# #### Atmosphere profil

# %%
mol_sca_filename  = OPT_PROP_PATH / 'tau_rayleigh_325.dat'
mol_abs_filename  = OPT_PROP_PATH / 'tau_molabs_325.dat'
wavelength = 325.
z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0],
                           dtype=float, skiprows=1, sep=r'\s+').values)
zs = len(z)
sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float,
                  skiprows=1, sep=r'\s+').values.reshape(1, zs)
abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float,
                  skiprows=1, sep=r'\s+').values.reshape(1, zs)
pro_b2 = Atm1D('afglt', grid=z, prof_ray=sca, prof_abs=abs).calc(wavelength)
surf_b2 = None

# %% [markdown]
# #### Compute radiances at 0 and 30 km

# %%
vza_min = 100.
vza_max = 180.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

vaa_min = 0.
vaa_max = 180.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

# SMART-G Forward th and phi using local estimate (anticlockwise)
# conversion with vza and vaa MYSTIC (clockwise)
th  = 180.-vza
phi = -vaa
th[th==0] = 1e-6  # avoid problem due to special case of 0
le     = LocalEstimate(th_deg=th, phi_deg=phi)

sza = 60.0
saa = 0.
phi_0 = 180.-saa # SMART-G anticlockwise converted to be consistent with MYSTIC

m_b2f = s_1df.run(th_deg=sza, ph_deg=phi_0, wavelength=wavelength,
                  n_photons=NB_PH, n_loop=min(1e6, round(NB_PH/10.)),
                  atmosphere=pro_b2, output_layers=int(7), le=le,
                  surface=surf_b2, xblock=64, xgrid=1024, beer=1, depo=0.03,
                  stdev=True, seed=SEED)

# %%
m_b2f.to_netcdf(output_folder_path / "iprt_b2_smartg_ref.nc")

# %% [markdown]
# #### Convert into iprt output format

# %%
m = m_b2f
# Radiances
vza = 180.-m.coords['Zenith angles'].values
vaa = -m.coords['Azimuth angles'].values

filename  = output_folder_path / "iprt_case_b2_smartg_ref.dat"
case_name = "B2"
depols    = [0.03, 0.03]
altitudes      = [0., 30.]
szas      = [60., 60.]
saas      = [0., 0.]

# convert
convert_sgout_to_iprtout(datasets=[m, m], u_signs=[-1, -1],
                         case_name=case_name, depols=depols,
                         altitudes=altitudes, szas=szas, saas=saas,
                         vzas=[180.-vza, vza], vaas=[vaa, vaa],
                         file_name=filename,
                         output_layer=['_down (0+)', '_up (TOA)'])

# %% [markdown]
# #### Comparison with MYSTIC

# %%
filename     = output_folder_path / "iprt_case_b2_smartg_ref.dat"
case_name    = "B2"
depols       = [0.03, 0.03]
altitudes         = [0., 30.]
szas         = [60., 60.]
saas         = [0., 0.]
vzas         = [np.round(180.-vza), np.round(vza)]

smartg_b2 = pd.read_csv(filename, header=None, sep=r'\s+', dtype=float,
                        comment="#").values
mystic_b2 = pd.read_csv(
    MYSTIC_RES_PATH / "iprt_case_b2_mystic.dat", header=None, sep=r'\s+',
    dtype=float, comment="#").values
model     = smartg_b2
ref_model = mystic_b2
mod_name  = "SMARTG"
rmod_name = "MYSTIC"
l_invth_rmod  = [False, True]
l_invth_mod   = [False, True]
l_u_sign_rmod  =[True, True]
l_u_sign_mod  = [True, True]
l_v_sign_rmod  =[True, True]
l_v_sign_mod  = [True, True]
avoid_p_rmod = False
avoid_p_mod  = True
avoid_p_dif  = False
sym = True

n_res = len(depols)
for ires in range (0, n_res):
    # !!!! mistake in MYSTIC depol, depol is set to 0 instead of 0.03
    # !!!!
    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f}  - "
             f"{altitudes[ires]:.0f}km - {rmod_name}")
    i_rmod, q_rmod, u_rmod, v_rmod = select_and_plot_polar_iprt(
        ref_model, z_alti=altitudes[ires], depol=0., title=title,
        change_u_sign=l_u_sign_rmod[ires], change_v_sign=l_v_sign_rmod[ires],
        inv_thetas=l_invth_rmod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_rmod, thetas=vzas[ires])

    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{altitudes[ires]:.0f}km - {mod_name}")
    i_mod, q_mod, u_mod, v_mod = select_and_plot_polar_iprt(
        model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_mod[ires], change_v_sign=l_v_sign_mod[ires],
        inv_thetas=l_invth_mod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_mod, thetas=vzas[ires])

    if (ires == 0):
        iquv_mod_tot = group_iquv(i_list=[i_mod], q_list=[q_mod],
                                  u_list=[u_mod], v_list=[v_mod])
        iquv_rmod_tot = group_iquv(i_list=[i_rmod], q_list=[q_rmod],
                                   u_list=[u_rmod], v_list=[v_rmod])
    else:
        iquv_mod_tot = np.concatenate((iquv_mod_tot, group_iquv(
            i_list=[i_mod], q_list=[q_mod],
            u_list=[u_mod], v_list=[v_mod])), axis=1)
        iquv_rmod_tot = np.concatenate((iquv_rmod_tot, group_iquv(
            i_list=[i_rmod], q_list=[q_rmod],
            u_list=[u_rmod], v_list=[v_rmod])), axis=1)

    i_val = i_rmod-i_mod
    q_val = q_rmod-q_mod
    u_val = u_rmod-u_mod
    v_val = v_rmod-v_mod
    max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
    max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
    max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
    max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
    title = (f"IPRT case {case_name} - depol = {depols[ires]}  - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{depols[ires]:.0f}km - dif ({rmod_name}-{mod_name})")
    select_and_plot_polar_iprt(ref_model, z_alti=altitudes[ires], depol=0.,
                               title=title,
                               force_iquv=[i_val, q_val, u_val, v_val],
                               max_i=max_i, max_q=max_q, max_u=max_u,
                               max_v=max_v, cmap_i='RdBu_r',
                               avoid_plot=avoid_p_dif, thetas=vzas[ires])

# %% [markdown]
# #### Compute delta_m

# %%
delta_m_b2 = compute_deltam(
    obs=iquv_rmod_tot, mod=iquv_mod_tot, print_res=True)

# %% [markdown]
# ### Case B3

# %% [markdown]
# #### Atmosphere profil

# %%
# molecular scattering and absorption
mol_sca_filename  = OPT_PROP_PATH / 'tau_rayleigh_350.dat'
mol_abs_filename  = OPT_PROP_PATH / 'tau_molabs_350.dat'
wavelength = np.array([350.])
n_wavelength= len(wavelength)
z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0],
                           dtype=float, skiprows=1, sep=r'\s+').values)
zs = len(z)
nz = len(z[1:])
mol_sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float,
                      skiprows=1, sep=r'\s+').values.reshape(1, zs)
mol_abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float,
                      skiprows=1, sep=r'\s+').values.reshape(1, zs)

# aerosol extinction and single scattering albedo
aer_ext_filename  = OPT_PROP_PATH / 'tau_aerosol.dat'
aer_tau_ext  = pd.read_csv(
    aer_ext_filename, header=None, usecols=[1], dtype=float, skiprows=1,
    sep=r'\s+').values.reshape(1, zs)
aer_ssa  = np.full_like(aer_tau_ext, 0.787581)
prof_aer = (aer_tau_ext, aer_ssa)

# aerosol phase matrix
file_aer_phase = OPT_PROP_PATH / 'sizedistr_spheroid.cdf'
aer_phase = read_phase(fname=file_aer_phase)
n_theta = 18001
aer_phase = aer_phase.interp(**{'theta_atm': np.linspace(
    0, 180, n_theta)}, method='linear')
nstk = aer_phase.shape[2]
prof_phases = get_prof_phases(aer_phase, wavelength=wavelength, z=z)

# atmosphere profil
pro_b3 = Atm1D(
    'afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs, prof_aer=prof_aer,
    prof_phases=prof_phases).calc(wavelength, phase=False)
surf_b3 = None

# %% [markdown]
# #### Compute radiances

# %%
vza_min = 100
vza_max = 180.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

vaa_min = 0.
vaa_max = 180.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

# SMART-G Forward th and phi using local estimate (anticlockwise)
# conversion with vza and vaa MYSTIC (clockwise)
th  = 180.-vza
phi = -vaa
th[th==0] = 1e-6  # avoid problem due to special case of 0
le     = LocalEstimate(th_deg=th, phi_deg=phi)

sza = 30.0
saa = 0.
phi_0 = 180.-saa # SMART-G anticlockwise converted to be consistent with MYSTIC

m_b3f = s_1df.run(th_deg=sza, ph_deg=phi_0, wavelength=wavelength,
                  n_photons=NB_PH, n_loop=min(1e6, round(NB_PH/10.)),
                  atmosphere=pro_b3, output_layers=int(7), le=le,
                  surface=surf_b3, xblock=64, xgrid=1024, beer=1, depo=0.03,
                  stdev=True, seed=SEED)

# %%
m_b3f.to_netcdf(output_folder_path / "iprt_b3_smartg_ref.nc")

# %% [markdown]
# #### Convert into iprt output format

# %%
m = m_b3f
# Radiances
vza = 180.-m.coords['Zenith angles'].values
vaa = -m.coords['Azimuth angles'].values

filename  = output_folder_path / "iprt_case_b3_smartg_ref.dat"
case_name = "B3"
depols    = [0.03, 0.03]
altitudes      = [0., 30.]
szas      = [30., 30.]
saas      = [0., 0.]

# convert
convert_sgout_to_iprtout(datasets=[m, m], u_signs=[-1, -1],
                         case_name=case_name, depols=depols,
                         altitudes=altitudes, szas=szas, saas=saas,
                         vzas=[180.-vza, vza], vaas=[vaa, vaa],
                         file_name=filename,
                         output_layer=['_down (0+)', '_up (TOA)'])

# %% [markdown]
# #### Comparison with MYSTIC

# %%
filename     = output_folder_path / "iprt_case_b3_smartg_ref.dat"
case_name    = "B3"
depols       = [0.03, 0.03]
altitudes         = [0., 30.]
szas         = [30., 30.]
saas         = [0., 0.]
vzas         = [np.round(180.-vza), np.round(vza)]

smartg_b3 = pd.read_csv(filename, header=None, sep=r'\s+', dtype=float,
                        comment="#").values
mystic_b3 = pd.read_csv(
    MYSTIC_RES_PATH / "iprt_case_b3_mystic.dat", header=None, sep=r'\s+',
    dtype=float, comment="#").values
model     = smartg_b3
ref_model = mystic_b3
mod_name  = "SMARTG"
rmod_name = "MYSTIC"
l_invth_rmod  = [False, True]
l_invth_mod   = [False, True]
l_u_sign_rmod  =[True, True]
l_u_sign_mod  = [True, True]
l_v_sign_rmod  =[True, True]
l_v_sign_mod  = [True, True]
avoid_p_rmod = False
avoid_p_mod  = True
avoid_p_dif  = False
sym = True

n_res = len(depols)
for ires in range (0, n_res):
    # !!!! mistake in MYSTIC depol, depol is set to 0 instead of 0.03
    # !!!!
    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f}  - "
             f"{altitudes[ires]:.0f}km - {rmod_name}")
    i_rmod, q_rmod, u_rmod, v_rmod = select_and_plot_polar_iprt(
        ref_model, z_alti=altitudes[ires], depol=0., title=title,
        change_u_sign=l_u_sign_rmod[ires], change_v_sign=l_v_sign_rmod[ires],
        inv_thetas=l_invth_rmod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_rmod, thetas=vzas[ires])

    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{altitudes[ires]:.0f}km - {mod_name}")
    i_mod, q_mod, u_mod, v_mod = select_and_plot_polar_iprt(
        model, z_alti=altitudes[ires], depol=depols[ires], title=title,
        change_u_sign=l_u_sign_mod[ires], change_v_sign=l_v_sign_mod[ires],
        inv_thetas=l_invth_mod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_mod, thetas=vzas[ires])

    if (ires == 0):
        iquv_mod_tot = group_iquv(i_list=[i_mod], q_list=[q_mod],
                                  u_list=[u_mod], v_list=[v_mod])
        iquv_rmod_tot = group_iquv(i_list=[i_rmod], q_list=[q_rmod],
                                   u_list=[u_rmod], v_list=[v_rmod])
    else:
        iquv_mod_tot = np.concatenate((iquv_mod_tot, group_iquv(
            i_list=[i_mod], q_list=[q_mod],
            u_list=[u_mod], v_list=[v_mod])), axis=1)
        iquv_rmod_tot = np.concatenate((iquv_rmod_tot, group_iquv(
            i_list=[i_rmod], q_list=[q_rmod],
            u_list=[u_rmod], v_list=[v_rmod])), axis=1)

    i_val = i_rmod-i_mod
    q_val = q_rmod-q_mod
    u_val = u_rmod-u_mod
    v_val = v_rmod-v_mod
    max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
    max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
    max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
    max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
    title = (f"IPRT case {case_name} - depol = {depols[ires]}  - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{depols[ires]:.0f}km - dif ({rmod_name}-{mod_name})")
    select_and_plot_polar_iprt(ref_model, z_alti=altitudes[ires], depol=0.,
                               title=title,
                               force_iquv=[i_val, q_val, u_val, v_val],
                               max_i=max_i, max_q=max_q, max_u=max_u,
                               max_v=max_v, cmap_i='RdBu_r',
                               avoid_plot=avoid_p_dif, thetas=vzas[ires])

# %% [markdown]
# #### Compute delta_m

# %%
delta_m_b3 = compute_deltam(
    obs=iquv_rmod_tot, mod=iquv_mod_tot, print_res=True)

# %% [markdown]
# ### Case B4

# %% [markdown]
# #### Atmosphere profil

# %%
# molecular scattering, no absorption
mol_sca_filename  = OPT_PROP_PATH / 'tau_rayleigh_800.dat'
wavelength = np.array([800.])
n_wavelength = len(wavelength)
z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0],
                           dtype=float, skiprows=1, sep=r'\s+').values)
zs = len(z)
nz = len(z[1:])
mol_sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float,
                      skiprows=1, sep=r'\s+').values.reshape(1, zs)

# cloud phase matrix
file_cld_phase = OPT_PROP_PATH / 'watercloud.mie.cdf'
cld_phase = read_phase(fname=file_cld_phase)
n_theta = cld_phase.shape[-1]

# cloud extinction and single scattering albedo
# below since we provide phase parameter the reff value and 'wc' file
# are ignored,
# but zmin, zmax, tau_ref and ssa are used to define the cloud vertical
# profile
cld = Cloud('wc', reff=10., zmin=2., zmax=3., tau_ref=5., w_ref=wavelength,
            ssa=0.999979, phase=cld_phase)

# atmosphere profil
pro_b4 = Atm1D(
    'afglt', comp=[cld], grid=z, prof_ray=mol_sca,
    prof_abs=np.zeros_like(mol_sca),
    #prof_aer=prof_aer, prof_phases=(ipha_atm, lpha_lut)
    ).calc(wavelength, phase=True, n_theta=n_theta)
surf_b4 = RoughSurface(wind=2., brdf=True, wave_shadow=True, nh2o=1.33)

# %% [markdown]
# #### Compute radiances (0 and 30 km)

# %%
vza_min = 100
vza_max = 180.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

vaa_min = 0.
vaa_max = 180.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

# SMART-G Forward th and phi using local estimate (anticlockwise)
# conversion with vza and vaa MYSTIC (clockwise)
th  = 180.-vza
phi = -vaa
th[th==0] = 1e-6  # avoid problem due to special case of 0
le     = LocalEstimate(th_deg=th, phi_deg=phi)

sza = 60.0
saa = 0.
phi_0 = 180.-saa # SMART-G anticlockwise converted to be consistent with MYSTIC

m_b4f = s_1df.run(
    th_deg=sza, ph_deg=phi_0, wavelength=wavelength,
    n_photons=NB_PH, n_loop=min(1e6, round(NB_PH/10.)),
    atmosphere=pro_b4, output_layers=int(7), le=le,
    surface=surf_b4, xblock=64, xgrid=1024, beer=1, depo=0.03,
    stdev=True, seed=SEED,
    n_icdf=n_theta)#, russian_roulette=1, russian_roulette_weight=0.1)
m_b4f2 = s_1df.run(th_deg=sza, ph_deg=phi_0, wavelength=wavelength,
                   n_photons=NB_PH, n_loop=min(1e6, round(NB_PH/10.)),
                   atmosphere=pro_b4, output_layers=int(7), le=le,
                   surface=surf_b4, xblock=64, xgrid=1024, beer=1, depo=0.03,
                   stdev=True, seed=SEED*2, n_icdf=n_theta)

# %%
m_b4f.to_netcdf(output_folder_path / "iprt_b4_smartg_ref.nc")
m_b4f2.to_netcdf(output_folder_path / "iprt_b4_smartg_ref2.nc")

# %% [markdown]
# #### Convert into iprt output format

# %%
m = m_b4f
# Radiances
vza = 180.-m.coords['Zenith angles'].values
vaa = -m.coords['Azimuth angles'].values

filename  = output_folder_path / "iprt_case_b4_smartg_ref.dat"
filename2  = output_folder_path / "iprt_case_b4_smartg_ref2.dat"
case_name = "B4"
depols    = [0.03, 0.03]
altitudes      = [0., 30.]
szas      = [60., 60.]
saas      = [0., 0.]

# convert
convert_sgout_to_iprtout(datasets=[m, m], u_signs=[-1, -1],
                         case_name=case_name, depols=depols,
                         altitudes=altitudes, szas=szas, saas=saas,
                         vzas=[180.-vza, vza], vaas=[vaa, vaa],
                         file_name=filename,
                         output_layer=['_down (0+)', '_up (TOA)'])
m = m_b4f2
convert_sgout_to_iprtout(datasets=[m, m], u_signs=[-1, -1],
                         case_name=case_name, depols=depols,
                         altitudes=altitudes, szas=szas, saas=saas,
                         vzas=[180.-vza, vza], vaas=[vaa, vaa],
                         file_name=filename2,
                         output_layer=['_down (0+)', '_up (TOA)'])

# %% [markdown]
# #### Comparison with MYSTIC

# %%
filename     = output_folder_path / "iprt_case_b4_smartg_ref.dat"
case_name    = "B4"
depols       = [0.03, 0.03]
altitudes         = [0., 30.]
szas         = [60., 60.]
saas         = [0., 0.]
vzas         = [np.round(180.-vza), np.round(vza)]
vaas         = [None, None]#[vaa, vaa]

smartg_b4 = pd.read_csv(filename, header=None, sep=r'\s+', dtype=float,
                        comment="#").values
mystic_b4 = pd.read_csv(
    MYSTIC_RES_PATH / "iprt_case_b4_mystic.dat", header=None, sep=r'\s+',
    dtype=float, comment="#").values
model     = smartg_b4
ref_model = mystic_b4
mod_name  = "SMARTG"
rmod_name = "MYSTIC"
l_invth_rmod  = [False, True]
l_invth_mod   = [False, True]
l_u_sign_rmod  =[True, True]
l_u_sign_mod  = [True, True]
l_v_sign_rmod  =[True, True]
l_v_sign_mod  = [True, True]
avoid_p_rmod = False
avoid_p_mod  = False
avoid_p_dif  = False
sym = True
use_2sim = True

if use_2sim:
    filename2 = output_folder_path / "iprt_case_b4_smartg_ref2.dat"
    model2    = pd.read_csv(filename2, header=None, sep=r'\s+', dtype=float,
                            comment="#").values

n_res = len(depols)
for ires in range (0, n_res):
    # !!!! mistake in MYSTIC depol, depol is set to 0 instead of 0.03
    # !!!!
    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f}  - "
             f"{altitudes[ires]:.0f}km - {rmod_name}")
    i_rmod, q_rmod, u_rmod, v_rmod = select_and_plot_polar_iprt(
        ref_model, z_alti=altitudes[ires], depol=0., title=title,
        change_u_sign=l_u_sign_rmod[ires], change_v_sign=l_v_sign_rmod[ires],
        inv_thetas=l_invth_rmod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_rmod, thetas=vzas[ires], phis=vaas[ires])
    if not use_2sim:
        title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
                 f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
                 f"{altitudes[ires]:.0f}km - {mod_name}")
        i_mod, q_mod, u_mod, v_mod = select_and_plot_polar_iprt(
            model, z_alti=altitudes[ires], depol=depols[ires], title=title,
            change_u_sign=l_u_sign_mod[ires], change_v_sign=l_v_sign_mod[ires],
            inv_thetas=l_invth_mod[ires], sym=sym, output_iquv=True,
            avoid_plot=avoid_p_mod, thetas=vzas[ires], phis=vaas[ires])
    else:
        title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
                 f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
                 f"{altitudes[ires]:.0f}km - {mod_name}")
        (i_mod, q_mod, u_mod, v_mod,
         i_std_mod, q_std_mod, u_std_mod,
         v_std_mod) = select_and_plot_polar_iprt(
            model, z_alti=altitudes[ires], depol=depols[ires],
            title=title, change_u_sign=l_u_sign_mod[ires],
            change_v_sign=l_v_sign_mod[ires],
            inv_thetas=l_invth_mod[ires], sym=sym,
            output_iquv=True, avoid_plot=True, thetas=vzas[ires],
            phis=vaas[ires], output_iquv_std=True)
        (i_mod2, q_mod2, u_mod2, v_mod2,
         i_std_mod2, q_std_mod2, u_std_mod2,
         v_std_mod2) = select_and_plot_polar_iprt(
            model2, z_alti=altitudes[ires], depol=depols[ires],
            title=title, change_u_sign=l_u_sign_mod[ires],
            change_v_sign=l_v_sign_mod[ires],
            inv_thetas=l_invth_mod[ires], sym=sym,
            output_iquv=True, avoid_plot=True, thetas=vzas[ires],
            phis=vaas[ires], output_iquv_std=True)
        i_mod3 = np.zeros_like(i_mod)
        q_mod3 = np.zeros_like(q_mod)
        u_mod3 = np.zeros_like(u_mod)
        v_mod3 = np.zeros_like(v_mod)
        for i in range (0, i_mod.shape[0]):
            for j in range (0, i_mod.shape[1]):
                if (i_std_mod[i, j] < i_std_mod2[i, j]):
                    i_mod3[i, j] = i_mod[i, j]
                else:
                    i_mod3[i, j]  = i_mod2[i, j]
                if (q_std_mod[
                    i, j] < q_std_mod2[i, j]): q_mod3[i, j]  = q_mod[i, j]
                else:
                    q_mod3[i, j]  = q_mod2[i, j]
                if (u_std_mod[
                    i, j] < u_std_mod2[i, j]): u_mod3[i, j]  = u_mod[i, j]
                else:
                    u_mod3[i, j]  = u_mod2[i, j]
                if (v_std_mod[
                    i, j] < v_std_mod2[i, j]): v_mod3[i, j]  = v_mod[i, j]
                else:
                    v_mod3[i, j]  = v_mod2[i, j]
        i_mod = i_mod3.copy()
        q_mod = q_mod3.copy()
        u_mod = u_mod3.copy()
        v_mod = v_mod3.copy()
        select_and_plot_polar_iprt(
            ref_model, z_alti=altitudes[0], depol=0., title=title,
            force_iquv=[i_mod, q_mod, u_mod, v_mod], avoid_plot=avoid_p_mod,
            sym=sym, thetas=vzas[ires], phis=vaas[ires],
            max_q=max(np.abs(np.min(q_rmod)), np.abs(np.max(q_rmod))),
            max_u=max(np.abs(np.min(u_rmod)), np.abs(np.max(u_rmod))),
            max_v=max(np.abs(np.min(v_rmod)), np.abs(np.max(v_rmod))))

    if (ires == 0):
        iquv_mod_tot = group_iquv(i_list=[i_mod], q_list=[q_mod],
                                  u_list=[u_mod], v_list=[v_mod])
        iquv_rmod_tot = group_iquv(i_list=[i_rmod], q_list=[q_rmod],
                                   u_list=[u_rmod], v_list=[v_rmod])
    else:
        iquv_mod_tot = np.concatenate((iquv_mod_tot, group_iquv(
            i_list=[i_mod], q_list=[q_mod],
            u_list=[u_mod], v_list=[v_mod])), axis=1)
        iquv_rmod_tot = np.concatenate((iquv_rmod_tot, group_iquv(
            i_list=[i_rmod], q_list=[q_rmod],
            u_list=[u_rmod], v_list=[v_rmod])), axis=1)

    i_val = i_rmod-i_mod
    q_val = q_rmod-q_mod
    u_val = u_rmod-u_mod
    v_val = v_rmod-v_mod
    max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
    max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
    max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
    max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
    title = (f"IPRT case {case_name} - depol = {depols[ires]}  - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{depols[ires]:.0f}km - dif ({rmod_name}-{mod_name})")
    select_and_plot_polar_iprt(ref_model, z_alti=altitudes[ires], depol=0.,
                               title=title,
                               force_iquv=[i_val, q_val, u_val, v_val],
                               max_i=max_i, max_q=max_q, max_u=max_u,
                               max_v=max_v, cmap_i='RdBu_r',
                               avoid_plot=avoid_p_dif, thetas=vzas[ires],
                               phis=vaas[ires])

# %% [markdown]
# #### Compute delta_m

# %%
delta_m_b4 = compute_deltam(
    obs=iquv_rmod_tot, mod=iquv_mod_tot, print_res=True)

# %% [markdown]
# #### Compute radiances in backward mode (0 km)

# %%
sza = 60.
saa = 0.
phi_0 = 180.-saa # To follow MYSTIC convention
# count_level=0 allows to count only uptoa and then the
# computional time
le     = LocalEstimate(
    th_deg=np.array([sza]), phi_deg=np.array([phi_0]),
    count_level=np.full((len(np.atleast_1d(sza))), 0, dtype=np.int32))
# vza
vza_min = 0.
vza_max = 80.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

# vaa from 0. to 180.
vaa_min = 0.
vaa_max = 180.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

lsensors = []
n_vza = len(vza)
n_vaa = len(vaa)
n_dir = round(n_vaa*n_vza)
for iza, za in enumerate(vza):
    for iaa, aa in enumerate(vaa):
        phi = -aa+180
        # ## Sensor at 1km altitude
        lsensors.append(Sensor(pos_z=0.0, th_deg=za, ph_deg=phi, loc='ATMOS'))

m_b4b_0km = s_1db.run(wavelength=wavelength, n_photons=NB_PH*n_dir,
                      n_loop=NB_PH, atmosphere=pro_b4, sensor=lsensors,
                      n_icdf=n_theta, le=le, surface=surf_b4, xblock=64,
                      xgrid=1024, beer=1, depo=0.03, stdev=True, progress=True,
                      seed=SEED, output_layers=int(0))

m_b4b_0km = drop_axes(m_b4b_0km, 'Azimuth angles', 'Zenith angles')

for name in list(m_b4b_0km.data_vars):
    if 'sensor index' in m_b4b_0km[name].dims:
        mat_tmp = np.swapaxes(
            m_b4b_0km[name].values.reshape(len(vza), len(vaa)), 0, 1)
        attrs_tmp = m_b4b_0km[name].attrs
        m_b4b_0km = m_b4b_0km.drop_vars([name])
        m_b4b_0km[name] = xr.Variable(('Azimuth angles', 'Zenith angles'),
                                      mat_tmp, attrs=attrs_tmp)
m_b4b_0km = m_b4b_0km.assign_coords({'Azimuth angles': -vaa+180.,
                                     'Zenith angles': vza})

m_b4b_0km = drop_axes(m_b4b_0km, 'sensor index')

m_b4b_0km2 = s_1db.run(wavelength=wavelength, n_photons=NB_PH*n_dir,
                       n_loop=NB_PH, atmosphere=pro_b4, sensor=lsensors,
                       n_icdf=n_theta, le=le, surface=surf_b4, xblock=64,
                       xgrid=1024, beer=1, depo=0.03, stdev=True,
                       progress=True, seed=SEED*2, output_layers=int(0))

m_b4b_0km2 = drop_axes(m_b4b_0km2, 'Azimuth angles', 'Zenith angles')

for name in list(m_b4b_0km2.data_vars):
    if 'sensor index' in m_b4b_0km2[name].dims:
        mat_tmp = np.swapaxes(
            m_b4b_0km2[name].values.reshape(len(vza), len(vaa)), 0, 1)
        attrs_tmp = m_b4b_0km2[name].attrs
        m_b4b_0km2 = m_b4b_0km2.drop_vars([name])
        m_b4b_0km2[name] = xr.Variable(('Azimuth angles', 'Zenith angles'),
                                       mat_tmp, attrs=attrs_tmp)
m_b4b_0km2 = m_b4b_0km2.assign_coords({'Azimuth angles': -vaa+180.,
                                       'Zenith angles': vza})

m_b4b_0km2 = drop_axes(m_b4b_0km2, 'sensor index')

# %%
m_b4b_0km.to_netcdf(output_folder_path / "iprt_b4_smartg_ref_BAK_0km.nc")
m_b4b_0km2.to_netcdf(output_folder_path / "iprt_b4_smartg_ref_BAK_0km2.nc")

# %% [markdown]
# #### Compute radiances in backward mode (30km)

# %%
sza = 60.
saa = 0.
phi_0 = 180.-saa # To follow MYSTIC convention
# count_level=0 allows to count only uptoa and then the
# computional time
le     = LocalEstimate(
    th_deg=np.array([sza]), phi_deg=np.array([phi_0]),
    count_level=np.full((len(np.atleast_1d(sza))), 0, dtype=np.int32))

# vza
vza_min = 100.
vza_max = 180.
vza_inc = 5.
vza = np.arange(vza_min, vza_max+vza_inc, vza_inc)

# vaa from 0. to 180.
vaa_min = 0.
vaa_max = 180.
vaa_inc = 5.
vaa = np.arange(vaa_min, vaa_max+vaa_inc, vaa_inc)

lsensors = []
n_vza = len(vza)
n_vaa = len(vaa)
n_dir = round(n_vaa*n_vza)
for iza, za in enumerate(vza):
    for iaa, aa in enumerate(vaa):
        phi = -aa+180
        # ## Sensor at 1km altitude
        lsensors.append(Sensor(pos_z=30.0, th_deg=za, ph_deg=phi, loc='ATMOS'))

m_b4b_30km = s_1db.run(wavelength=wavelength, n_photons=NB_PH*n_dir,
                       n_loop=NB_PH, atmosphere=pro_b4, sensor=lsensors,
                       n_icdf=n_theta, le=le, surface=surf_b4, xblock=64,
                       xgrid=1024, beer=1, depo=0.03, stdev=True,
                       progress=True, seed=SEED, output_layers=int(0))

m_b4b_30km = drop_axes(m_b4b_30km, 'Azimuth angles', 'Zenith angles')

for name in list(m_b4b_30km.data_vars):
    if 'sensor index' in m_b4b_30km[name].dims:
        mat_tmp = np.swapaxes(
            m_b4b_30km[name].values.reshape(len(vza), len(vaa)), 0, 1)
        attrs_tmp = m_b4b_30km[name].attrs
        m_b4b_30km = m_b4b_30km.drop_vars([name])
        m_b4b_30km[name] = xr.Variable(('Azimuth angles', 'Zenith angles'),
                                       mat_tmp, attrs=attrs_tmp)
m_b4b_30km = m_b4b_30km.assign_coords({'Azimuth angles': -vaa+180.,
                                       'Zenith angles': vza})

m_b4b_30km = drop_axes(m_b4b_30km, 'sensor index')

m_b4b_30km2 = s_1db.run(wavelength=wavelength, n_photons=NB_PH*n_dir,
                        n_loop=NB_PH, atmosphere=pro_b4, sensor=lsensors,
                        n_icdf=n_theta, le=le, surface=surf_b4, xblock=64,
                        xgrid=1024, beer=1, depo=0.03, stdev=True,
                        progress=True, seed=SEED*2, output_layers=int(0))

m_b4b_30km2 = drop_axes(m_b4b_30km2, 'Azimuth angles', 'Zenith angles')

for name in list(m_b4b_30km2.data_vars):
    if 'sensor index' in m_b4b_30km2[name].dims:
        mat_tmp = np.swapaxes(
            m_b4b_30km2[name].values.reshape(len(vza), len(vaa)), 0, 1)
        attrs_tmp = m_b4b_30km2[name].attrs
        m_b4b_30km2 = m_b4b_30km2.drop_vars([name])
        m_b4b_30km2[name] = xr.Variable(('Azimuth angles', 'Zenith angles'),
                                        mat_tmp, attrs=attrs_tmp)
m_b4b_30km2 = m_b4b_30km2.assign_coords({'Azimuth angles': -vaa+180.,
                                         'Zenith angles': vza})

m_b4b_30km2 = drop_axes(m_b4b_30km2, 'sensor index')

# %%
m_b4b_30km.to_netcdf(output_folder_path / "iprt_b4_smartg_ref_BAK_30km.nc")
m_b4b_30km2.to_netcdf(output_folder_path / "iprt_b4_smartg_ref_BAK_30km2.nc")

# %% [markdown]
# #### Convert into iprt output format

# %%
# Radiances
vza_0km = m_b4b_0km.coords['Zenith angles'].values
vaa_0km = 180-m_b4b_0km.coords['Azimuth angles'].values

vza_30km = m_b4b_30km.coords['Zenith angles'].values
vaa_30km = 180.-m_b4b_30km.coords['Azimuth angles'].values

filename  = output_folder_path / "iprt_case_b4_smartg_BAK_ref.dat"
filename2  = output_folder_path / "iprt_case_b4_smartg_BAK_ref2.dat"
case_name = "B4"
depols    = [0.03, 0.03]
altitudes      = [0., 30.]
szas      = [60., 60.]
saas      = [0., 0.]

# convert
convert_sgout_to_iprtout(datasets=[m_b4b_0km, m_b4b_30km], u_signs=[1, 1],
                         case_name=case_name, depols=depols,
                         altitudes=altitudes, szas=szas, saas=saas,
                         vzas=[vza_0km, vza_30km], vaas=[vaa_0km, vaa_30km],
                         file_name=filename,
                         output_layer=['_up (TOA)', '_up (TOA)'])

convert_sgout_to_iprtout(datasets=[m_b4b_0km2, m_b4b_30km2], u_signs=[1, 1],
                         case_name=case_name, depols=depols,
                         altitudes=altitudes, szas=szas, saas=saas,
                         vzas=[vza_0km, vza_30km], vaas=[vaa_0km, vaa_30km],
                         file_name=filename2,
                         output_layer=['_up (TOA)', '_up (TOA)'])

# %% [markdown]
# #### Comparison with MYSTIC

# %%
filename     = output_folder_path / "iprt_case_b4_smartg_BAK_ref.dat"
case_name    = "B4"
depols       = [0.03, 0.03]
altitudes         = [0., 30.]
szas         = [60., 60.]
saas         = [0., 0.]
vzas         = [np.round(180.-vza_30km), np.round(vza_30km)]
vaas         = [None, None]#[vaa, vaa]

smartg_b4 = pd.read_csv(filename, header=None, sep=r'\s+', dtype=float,
                        comment="#").values
mystic_b4 = pd.read_csv(
    MYSTIC_RES_PATH / "iprt_case_b4_mystic.dat", header=None, sep=r'\s+',
    dtype=float, comment="#").values
model     = smartg_b4
ref_model = mystic_b4
mod_name  = "SMARTG"
rmod_name = "MYSTIC"
l_invth_rmod  = [False, True]
l_invth_mod   = [False, True]
l_u_sign_rmod  =[True, True]
l_u_sign_mod  = [True, True]
l_v_sign_rmod  =[True, True]
l_v_sign_mod  = [False, False]
avoid_p_rmod = False
avoid_p_mod  = False
avoid_p_dif  = False
sym = True
use_2sim = True

if use_2sim:
    filename2 = output_folder_path / "iprt_case_b4_smartg_BAK_ref2.dat"
    model2    = pd.read_csv(filename2, header=None, sep=r'\s+', dtype=float,
                            comment="#").values

n_res = len(depols)
for ires in range (0, n_res):
    # !!!! mistake in MYSTIC depol, depol is set to 0 instead of 0.03
    # !!!!
    title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f}  - "
             f"{altitudes[ires]:.0f}km - {rmod_name}")
    i_rmod, q_rmod, u_rmod, v_rmod = select_and_plot_polar_iprt(
        ref_model, z_alti=altitudes[ires], depol=0., title=title,
        change_u_sign=l_u_sign_rmod[ires], change_v_sign=l_v_sign_rmod[ires],
        inv_thetas=l_invth_rmod[ires], sym=sym, output_iquv=True,
        avoid_plot=avoid_p_rmod, thetas=vzas[ires], phis=vaas[ires])


    if not use_2sim:
        title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
                 f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
                 f"{altitudes[ires]:.0f}km - {mod_name}")
        i_mod, q_mod, u_mod, v_mod = select_and_plot_polar_iprt(
            model, z_alti=altitudes[ires], depol=depols[ires], title=title,
            change_u_sign=l_u_sign_mod[ires], change_v_sign=l_v_sign_mod[ires],
            inv_thetas=l_invth_mod[ires], sym=sym, output_iquv=True,
            avoid_plot=avoid_p_mod, thetas=vzas[ires], phis=vaas[ires],
            max_q=max(np.abs(np.min(q_rmod)), np.abs(np.max(q_rmod))),
            max_u=max(np.abs(np.min(u_rmod)), np.abs(np.max(u_rmod))),
            max_v=max(np.abs(np.min(v_rmod)), np.abs(np.max(v_rmod))))
    else:
        title = (f"IPRT case {case_name} - depol = {depols[ires]} - SZA = "
                 f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
                 f"{altitudes[ires]:.0f}km - {mod_name}")
        (i_mod, q_mod, u_mod, v_mod,
         i_std_mod, q_std_mod, u_std_mod,
         v_std_mod) = select_and_plot_polar_iprt(
            model, z_alti=altitudes[ires], depol=depols[ires],
            title=title, change_u_sign=l_u_sign_mod[ires],
            change_v_sign=l_v_sign_mod[ires],
            inv_thetas=l_invth_mod[ires], sym=sym,
            output_iquv=True, avoid_plot=True, thetas=vzas[ires],
            phis=vaas[ires], output_iquv_std=True)
        (i_mod2, q_mod2, u_mod2, v_mod2,
         i_std_mod2, q_std_mod2, u_std_mod2,
         v_std_mod2) = select_and_plot_polar_iprt(
            model2, z_alti=altitudes[ires], depol=depols[ires],
            title=title, change_u_sign=l_u_sign_mod[ires],
            change_v_sign=l_v_sign_mod[ires],
            inv_thetas=l_invth_mod[ires], sym=sym,
            output_iquv=True, avoid_plot=True, thetas=vzas[ires],
            phis=vaas[ires], output_iquv_std=True)
        i_mod3 = np.zeros_like(i_mod)
        q_mod3 = np.zeros_like(q_mod)
        u_mod3 = np.zeros_like(u_mod)
        v_mod3 = np.zeros_like(v_mod)
        for i in range (0, i_mod.shape[0]):
            for j in range (0, i_mod.shape[1]):
                if (i_std_mod[i, j] < i_std_mod2[i, j]):
                    i_mod3[i, j] = i_mod[i, j]
                else:
                    i_mod3[i, j]  = i_mod2[i, j]
                if (q_std_mod[
                    i, j] < q_std_mod2[i, j]): q_mod3[i, j]  = q_mod[i, j]
                else:
                    q_mod3[i, j]  = q_mod2[i, j]
                if (u_std_mod[
                    i, j] < u_std_mod2[i, j]): u_mod3[i, j]  = u_mod[i, j]
                else:
                    u_mod3[i, j]  = u_mod2[i, j]
                if (v_std_mod[
                    i, j] < v_std_mod2[i, j]): v_mod3[i, j]  = v_mod[i, j]
                else:
                    v_mod3[i, j]  = v_mod2[i, j]
        i_mod = i_mod3.copy()
        q_mod = q_mod3.copy()
        u_mod = u_mod3.copy()
        v_mod = v_mod3.copy()
        select_and_plot_polar_iprt(
            ref_model, z_alti=altitudes[0], depol=0., title=title,
            force_iquv=[i_mod, q_mod, u_mod, v_mod], avoid_plot=avoid_p_mod,
            sym=sym, thetas=vzas[ires], phis=vaas[ires],
            max_q=max(np.abs(np.min(q_rmod)), np.abs(np.max(q_rmod))),
            max_u=max(np.abs(np.min(u_rmod)), np.abs(np.max(u_rmod))),
            max_v=max(np.abs(np.min(v_rmod)), np.abs(np.max(v_rmod))))

    if (ires == 0):
        iquv_mod_tot = group_iquv(i_list=[i_mod], q_list=[q_mod],
                                  u_list=[u_mod], v_list=[v_mod])
        iquv_rmod_tot = group_iquv(i_list=[i_rmod], q_list=[q_rmod],
                                   u_list=[u_rmod], v_list=[v_rmod])
    else:
        iquv_mod_tot = np.concatenate((iquv_mod_tot, group_iquv(
            i_list=[i_mod], q_list=[q_mod],
            u_list=[u_mod], v_list=[v_mod])), axis=1)
        iquv_rmod_tot = np.concatenate((iquv_rmod_tot, group_iquv(
            i_list=[i_rmod], q_list=[q_rmod],
            u_list=[u_rmod], v_list=[v_rmod])), axis=1)

    i_val = i_rmod-i_mod
    q_val = q_rmod-q_mod
    u_val = u_rmod-u_mod
    v_val = v_rmod-v_mod
    max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
    max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
    max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
    max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
    title = (f"IPRT case {case_name} - depol = {depols[ires]}  - SZA = "
             f"{szas[ires]:.0f} - SAA = {saas[ires]:.0f} - "
             f"{depols[ires]:.0f}km - dif ({rmod_name}-{mod_name})")
    select_and_plot_polar_iprt(ref_model, z_alti=altitudes[ires], depol=0.,
                               title=title,
                               force_iquv=[i_val, q_val, u_val, v_val],
                               max_i=max_i, max_q=max_q, max_u=max_u,
                               max_v=max_v, cmap_i='RdBu_r',
                               avoid_plot=avoid_p_dif, thetas=vzas[ires],
                               phis=vaas[ires])

# %% [markdown]
# #### Compute delta_m

# %%
delta_m_b4_bak = compute_deltam(obs=iquv_rmod_tot, mod=iquv_mod_tot,
                                print_res=True)
