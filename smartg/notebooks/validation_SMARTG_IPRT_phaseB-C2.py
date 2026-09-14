# %% [markdown]
# # SMART-G validation IPRT phase B - Cubic cloud (C2)
# - https://www.meteo.physik.uni-muenchen.de/~iprt/doku.php?id=start

# %% [markdown]
# ## Symbols used in this notebook
#
# | symbol | meaning |
# |---|---|
# | `i_smartg`, `q_smartg`, `u_smartg`, `v_smartg` | Stokes matrices simulated by SMART-G |
# | `i_mystic`, ... | the same, read from the MYSTIC reference |
# | `iquv_*` | the four components stacked, as `smartg.iprt.group_iquv` returns them |
# | `theta`, `phi` | sensor viewing zenith and azimuth, in degrees |
# | `theta_0`, `phi_0` | solar zenith and azimuth |
# | `tau_r` | Rayleigh optical depth of the column |
# | `pos_z` | sensor altitude, in km |
# | `m_3d_c2_<n>_*` | the run output of case `<n>`, with or without atmosphere |
# | `norm_c2_<n>_*` | the normalisation applied to that case |
#
# `X_BLOCK` and `X_GRID` are the CUDA launch geometry; `xblock_run` and
# `xgrid_run` are the values a run is actually given, chosen either from
# those constants or by the tuning loop.

# %%
# %matplotlib inline
# next 2 lines allow to automatically reload modules that have been
# changed externally
# %load_ext autoreload
# %autoreload 2

import sys
from pathlib import Path

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
sys.path.insert(0, ROOTPATH)

from smartg.config import DIR_AUXDATA
from smartg.sensor import get_sensors_grid
from smartg.view import satellite_view
from smartg.grid3d import Grid3D
from smartg.smartg import LocalEstimate, Smartg
from smartg.surface import LambSurface
from smartg.albedo import AlbedoCst
from smartg.atmosphere import Atm1D, Atm3D, Cloud3D, read_i3rc_cloud
from smartg.phase import read_phase_cdf
from smartg.diff import diff1
import pandas as pd
import numpy as np
from smartg.iprt.iprt import compute_deltam, group_iquv

s_3db = Smartg(opt3d=True, alt_pp=True, alis=False, back=True, double=True,
               bias=True)
s_3df = Smartg(opt3d=True, alt_pp=True, alis=False, back=False, double=True,
               bias=True)
NB_PHOTONS = 49e9
NB_LOOP    = 1e8

# In case we want to find the optimal number of block and grid (CUDA)
# for each sim
# The values bellow can (have may be to) be modified (depending on the
# GPU)
FIND_OPTIMAL_XB_XG = False
X_BLOCKS = [32, 64, 128]
X_GRIDS = [512, 1024]
CHECK_NB_LOOP = 1e8
CHECK_NB_PHOTONS = 1e8

# Else take the following number of block and grid (values accepted by
# most of GPUs after 10xx series)
X_BLOCK = 128
X_GRID = 1024

# %% [markdown]
# ## Without atmosphere

# %% [markdown]
# ### Commun to all cases

# %% [markdown]
# #### Atmosphere profil

# %%
scale = 1 # can be useful for grid with very small cells

# ========= phase matrix
N_THETA = 18001 # 1801 is not enough for case 6, then we take 18001
file_cld_phase = (DIR_AUXDATA / 'IPRT' / 'phaseB' / 'opt_prop'
                  / 'watercloud_800.mie.cdf')
cld_phase = read_phase_cdf(
    file_cld_phase, n_theta=N_THETA, normalize=False, output_sg_ready=False
)

# ========= grid
# ********* method reduced grid = faster *********
xgrid = np.array([0., 3., 4., 7.])*scale
ygrid = np.array([0., 3., 4., 7.])*scale
zgrid = np.array([0., 2., 3., 5.])*scale
grid3 = Grid3D(xgrid, ygrid, zgrid, periodic=True)
# First column x, second y and third z
# We follow the IPRT convention for indices (start at 1 instead of 0)
# the cubic cloud is between 3 and 4 km in x and y, and between 2 and 3
# km in z.
# Its single scattering albedo is forced to 1 (non absorbing).
cloud_indices1 = np.zeros((1, 3), dtype=np.int32)
cloud_indices1[0, :] = np.array([
    2, 2, 2]) #, x, y and z indices according to x, y and z grids
cld_ext_coeff1 = np.zeros(1, dtype=np.float64)
cld_ext_coeff1[0] = 10.*(1/scale)
reff = np.zeros_like(cld_ext_coeff1, dtype=np.float64)
reff[0] = 10.
cloud3 = Cloud3D('wc', w_ref=800., ext_ref=cld_ext_coeff1,
                 cell_indices=cloud_indices1, reff=reff, phase=cld_phase,
                 ssa_cst=1.)
# **********************************************************************

# ********* method complete grid = slower but same grid as in IPRT paper
# *********
# cubic_cloud_f= DIR_AUXDATA / 'IPRT' / 'phaseB' / 'grids' /
# 'C2_cloud_70x70x5.dat'
# cloud3 = Cloud3D('wc', w_ref=800., ds=read_i3rc_cloud(cubic_cloud_f,
# loc_xgrid=0, loc_ygrid=0),
#                  phase=cld_phase, ssa_cst=1.)
# xgrid, ygrid, zgrid = cloud3.get_xyz_grid()
# grid3 = Grid3D(xgrid*scale, ygrid*scale, zgrid*scale, periodic=True)
# **********************************************************************

### profiles computations
wavelengths = np.array([800.])
atm3 = Atm3D(atm_1d=Atm1D('afglt', tau_r=0., no2=False, tco3=0., tcwp=0.),
             grid_3d=grid3, comp_3d=[cloud3], wavelength_phase=[800.])
pro_3d3_c2_noatm = atm3.calc(wavelengths, n_theta=N_THETA)

surf_c2 = LambSurface(alb=AlbedoCst(0.2))


# %% [markdown]
# #### Function to print results

# %%
def print_c2_res_noatm(m, norm, tcase, grid3_sensors, u_sign=1, v_sign=-1,
                       m_i=None, m_q=None, m_u=None, m_v=None):

    if m_i is None:
        i_smartg = m["I_up (TOA)"].values[
        :, 0, 0].reshape(70, 70)*norm
    else:
        i_smartg = np.asarray(m_i).reshape(70, 70)*norm
    # I_SMARTG_stdev = m["I_stdev_up
    # (TOA)"].values[:,0,0].reshape(70,70)*norm
    if m_q is None:
        q_smartg = m["Q_up (TOA)"].values[
        :, 0, 0].reshape(70, 70)*norm
    else:
        q_smartg = np.asarray(m_q).reshape(70, 70)*norm
    # Q_SMARTG_stdev = m["Q_stdev_up
    # (TOA)"].values[:,0,0].reshape(70,70)*norm
    if m_u is None:
        u_smartg = m["U_up (TOA)"].values[
        :, 0, 0].reshape(70, 70)*norm*u_sign
    else:
        u_smartg = np.asarray(m_u).reshape(70, 70)*norm*u_sign
    # U_SMARTG_stdev = m["U_stdev_up
    # (TOA)"].values[:,0,0].reshape(70,70)*norm
    if m_v is None:
        v_smartg = m["V_up (TOA)"].values[
        :, 0, 0].reshape(70, 70)*norm*v_sign
    else:
        v_smartg = np.asarray(m_v).reshape(70, 70)*norm*v_sign
    # V_SMARTG_stdev = m["V_stdev_up
    # (TOA)"].values[:,0,0].reshape(70,70)*norm



    file_res = (DIR_AUXDATA / 'IPRT' / 'phaseB' / 'mystic_res'
                / 'iprt_case_C2_mystic.dat')
    read_res = pd.read_csv(
        file_res, skiprows=(4900*(tcase-1)) + 3, nrows=4900, header=None,
        sep='\\s+', dtype=float).values
    i_mystic = read_res[:, 7].reshape(70, 70).T
    # I_MYSTIC_stdev = read_res[:,11].reshape(70,70).T
    q_mystic = read_res[:, 8].reshape(70, 70).T
    # Q_MYSTIC_stdev = read_res[:,12].reshape(70,70).T
    u_mystic = read_res[:, 9].reshape(70, 70).T
    # U_MYSTIC_stdev = read_res[:,13].reshape(70,70).T
    v_mystic = read_res[:, 10].reshape(70, 70).T
    # V_MYSTIC_stdev = read_res[:,14].reshape(70,70).T


    stk = ['I', 'Q', 'U', 'V']
    xgrid = grid3_sensors.xgrid
    ygrid = grid3_sensors.ygrid
    interp_name = 'none'
    fig_size = (10.5, 7)
    font_size=int(16)
    cb_shrink = 1
    color_bar = ['jet', 'coolwarm', 'coolwarm', 'coolwarm']
    vmin = [np.min(np.abs(i_smartg)), -np.max(np.abs(q_smartg)),
            -np.max(np.abs(u_smartg)), -np.max(np.abs(v_smartg))]
    vmax = [np.max(i_smartg), np.max(np.abs(q_smartg)),
            np.max(np.abs(u_smartg)), np.max(np.abs(v_smartg))]
    color_bar_std = ['coolwarm', 'coolwarm', 'coolwarm', 'coolwarm']
    vmin_std = [
        -np.max(i_smartg)*0.05, -np.max(np.abs(q_smartg))*0.05,
        -np.max(np.abs(u_smartg))*0.05, -np.max(np.abs(v_smartg))*0.015]
    vmax_std = [np.max(i_smartg)*0.05, np.max(np.abs(q_smartg))*0.05,
                np.max(np.abs(u_smartg))*0.05, np.max(np.abs(v_smartg))*0.015]


    mat_force = [i_smartg, q_smartg, u_smartg, v_smartg]
    satellite_view(m, xgrid, ygrid, interpolation=interp_name, cmap=color_bar,
                   figsize=fig_size, fontsize=font_size, vmin=vmin,
                   vmax=vmax, scale=False, stokes=stk, matrices=mat_force,
                   cbar_shrink=cb_shrink, cbar_sci_format=True,
                   title=f"C2 - case {tcase} - SMART-G - without atm")

    mat_force = [i_smartg-i_mystic, q_smartg-q_mystic, u_smartg-u_mystic,
                 v_smartg-v_mystic]
    satellite_view(
        m, xgrid, ygrid, interpolation=interp_name, cmap=color_bar_std,
        figsize=fig_size, fontsize=font_size, vmin=vmin_std, vmax=vmax_std,
        scale=False, stokes=stk, matrices=mat_force, cbar_shrink=cb_shrink,
        cbar_sci_format=True,
        title=f"C2 - case {tcase} - dif(SMART-G - MYSTIC) - without atm")



    # print deltam
    iquv_smartg = group_iquv(i_list=[i_smartg], q_list=[q_smartg],
                             u_list=[u_smartg], v_list=[v_smartg])
    iquv_mystic = group_iquv(i_list=[i_mystic], q_list=[q_mystic],
                             u_list=[u_mystic], v_list=[v_mystic])

    print("SMART-G (delta_m):")
    delta_m = compute_deltam(obs=iquv_mystic, mod=iquv_smartg, print_res=True)

# %% [markdown]
# ### Backward simulations

# %% [markdown]
# #### Case 1

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[0]

print("pos_z = ", pos_z)
theta = 40.
phi = 0.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 20.
phi_0    = 180.
# count_level = 0 -> only COUNT TOA (default value = -2, i.e., count
# everything)
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_1_noatm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                            n_loop=NB_LOOP, atmosphere=pro_3d3_c2_noatm,
                            sensor=sensors, le=le, surface=surf_c2,
                            n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                            stdev=True)
norm_c2_1_noatm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_1_noatm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_1_noatm
norm = norm_c2_1_noatm
tcase = int(1)

print_c2_res_noatm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 2

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[0]

print("pos_z = ", pos_z)
theta = 40.
phi = 60.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 20.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_2_noatm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                            n_loop=NB_LOOP, atmosphere=pro_3d3_c2_noatm,
                            sensor=sensors, le=le, surface=surf_c2,
                            n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                            stdev=True)
norm_c2_2_noatm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_2_noatm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_2_noatm
norm = norm_c2_2_noatm
tcase = int(2)

print_c2_res_noatm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 3

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[0]

print("pos_z = ", pos_z)
theta = 40.
phi = 120.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 20.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_3_noatm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                            n_loop=NB_LOOP, atmosphere=pro_3d3_c2_noatm,
                            sensor=sensors, le=le, surface=surf_c2,
                            n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                            stdev=True)
norm_c2_3_noatm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_3_noatm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_3_noatm
norm = norm_c2_3_noatm
tcase = int(3)

print_c2_res_noatm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 4

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[0]

print("pos_z = ", pos_z)
theta = 40.
phi = 180.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 20.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_4_noatm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                            n_loop=NB_LOOP, atmosphere=pro_3d3_c2_noatm,
                            sensor=sensors, le=le, surface=surf_c2,
                            n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                            stdev=True)
norm_c2_4_noatm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_4_noatm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_4_noatm
norm = norm_c2_4_noatm
tcase = int(4)

print_c2_res_noatm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 5

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6*scale

print("pos_z = ", pos_z)
theta = 180.
phi = 0.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 40.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_5_noatm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                            n_loop=NB_LOOP, atmosphere=pro_3d3_c2_noatm,
                            sensor=sensors, le=le, surface=surf_c2,
                            n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                            stdev=True)
norm_c2_5_noatm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_5_noatm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_5_noatm
norm = norm_c2_5_noatm
tcase = int(5)

print_c2_res_noatm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 6

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6*scale

print("pos_z = ", pos_z)
theta = 140.
phi = 0.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 40.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_6_noatm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                            n_loop=NB_LOOP, atmosphere=pro_3d3_c2_noatm,
                            sensor=sensors, le=le, surface=surf_c2,
                            n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                            stdev=True)
norm_c2_6_noatm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_6_noatm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_6_noatm
norm = norm_c2_6_noatm
tcase = int(6)

print_c2_res_noatm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 7

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6*scale

print("pos_z = ", pos_z)
theta = 140.
phi = 60.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 40.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_7_noatm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                            n_loop=NB_LOOP, atmosphere=pro_3d3_c2_noatm,
                            sensor=sensors, le=le, surface=surf_c2,
                            n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                            stdev=True)
norm_c2_7_noatm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_7_noatm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_7_noatm
norm = norm_c2_7_noatm
tcase = int(7)

print_c2_res_noatm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 8

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6*scale

print("pos_z = ", pos_z)
theta = 140.
phi = 120.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 40.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_8_noatm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                            n_loop=NB_LOOP, atmosphere=pro_3d3_c2_noatm,
                            sensor=sensors, le=le, surface=surf_c2,
                            n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                            stdev=True)
norm_c2_8_noatm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_8_noatm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_8_noatm
norm = norm_c2_8_noatm
tcase = int(8)

print_c2_res_noatm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 9

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6*scale

print("pos_z = ", pos_z)
theta = 140.
phi = 180.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 40.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_9_noatm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                            n_loop=NB_LOOP, atmosphere=pro_3d3_c2_noatm,
                            sensor=sensors, le=le, surface=surf_c2,
                            n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                            stdev=True)
norm_c2_9_noatm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_9_noatm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_9_noatm
norm = norm_c2_9_noatm
tcase = int(9)

print_c2_res_noatm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# ### Forward simulations

# %% [markdown]
# #### Case 1 - 4

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6

print("pos_z = ", pos_z)

# ======= Case 1 - 4
    # Sun position
theta_0    = 20.
phi_0    = 180.

theta_0_bis = 180.- theta_0
phi_0_bis = 180.- phi_0

# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta_0_bis,
    ph_deg=phi_0_bis, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

theta = np.array([40., 40., 40., 40.])
phi = np.array([0., 60., 120., 180.])

le     = LocalEstimate(th_deg=np.array(theta),
                       phi_deg=np.array(phi+180.),
                       count_level=np.array([1, 1, 1, 1]),
                       zip=True)

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3df.run(th_deg=theta_0, wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, output_layers=3)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_1to4_f_noatm = s_3df.run(th_deg=theta_0, wavelength=wavelengths,
                                 n_photons=NB_PHOTONS, n_loop=NB_LOOP,
                                 atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                 le=le, surface=surf_c2, n_icdf=N_THETA,
                                 xblock=xblock_run, xgrid=xgrid_run,
                                 output_layers=3)
norm_c2_1to4_f_noatm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_1to4_f_noatm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_1to4_f_noatm
norm = norm_c2_1to4_f_noatm
for i in range (0, 4):
    tcase = int(i+1)
    ind_za = round(tcase - 1)
    print("TESTCASE:", tcase)
    print_c2_res_noatm(m, norm, tcase, grid3_sensors, u_sign=-1, v_sign=1,
                       m_i=m['I_down (0+)'][:, ind_za],
                       m_q=m['Q_down (0+)'][:, ind_za],
                       m_u=m['U_down (0+)'][:, ind_za],
                       m_v=m['V_down (0+)'][:, ind_za])

# %% [markdown]
# #### Case 5 -9

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6

print("pos_z = ", pos_z)

# Sun position
theta_0    = 40.
phi_0    = 180.

theta_0_bis = 180.- theta_0
phi_0_bis = 180.- phi_0

# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta_0_bis,
    ph_deg=phi_0_bis, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

theta = np.array([180., 140., 140., 140., 140.])
phi = np.array([0., 0., 60., 120., 180.])

le     = LocalEstimate(th_deg=np.array(180.-theta),
                       phi_deg=np.array(phi+180.),
                       count_level=np.array([0, 0, 0, 0, 0]),
                       zip=True)

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3df.run(th_deg=theta_0, wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, output_layers=1)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_5to9_f_noatm = s_3df.run(th_deg=theta_0, wavelength=wavelengths,
                                 n_photons=NB_PHOTONS, n_loop=NB_LOOP,
                                 atmosphere=pro_3d3_c2_noatm, sensor=sensors,
                                 le=le, surface=surf_c2, n_icdf=N_THETA,
                                 xblock=xblock_run, xgrid=xgrid_run,
                                 output_layers=1)
norm_c2_5to9_f_noatm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_5to9_f_noatm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_5to9_f_noatm
norm = norm_c2_5to9_f_noatm
for i in range (0, 5):
    tcase = int(i+5)
    ind_za = round(tcase - 5)
    print("TESTCASE:", tcase)
    print_c2_res_noatm(m, norm, tcase, grid3_sensors, u_sign=-1, v_sign=1,
                       m_i=m['I_up (TOA)'][:, ind_za],
                       m_q=m['Q_up (TOA)'][:, ind_za],
                       m_u=m['U_up (TOA)'][:, ind_za],
                       m_v=m['V_up (TOA)'][:, ind_za])

# %% [markdown]
# ## With atmosphere

# %% [markdown]
# ### commun to all cases

# %% [markdown]
# #### Atmosphere profil

# %%
scale = 1 # can be useful for grid with very small cells

# ========= phase matrix
N_THETA = 18001 # 1801 is not enough for case 6, then we take 18001
file_cld_phase = (DIR_AUXDATA / 'IPRT' / 'phaseB' / 'opt_prop'
                  / 'watercloud_800.mie.cdf')
cld_phase = read_phase_cdf(
    file_cld_phase, n_theta=N_THETA, normalize=False, output_sg_ready=False
)

# ========= grid
# ********* method reduced grid = faster *********
xgrid = np.array([0., 3., 4., 7.])*scale
ygrid = np.array([0., 3., 4., 7.])*scale
zgrid = np.array([0., 2., 3., 5.])*scale
grid3 = Grid3D(xgrid, ygrid, zgrid, periodic=True)
# First column x, second y and third z
# We follow the IPRT convention for indices (start at 1 instead of 0)
# the cubic cloud is between 3 and 4 km in x and y, and between 2 and 3
# km in z.
# Its single scattering albedo is forced to 1 (non absorbing).
cloud_indices1 = np.zeros((1, 3), dtype=np.int32)
cloud_indices1[0, :] = np.array([
    2, 2, 2]) #, x, y and z indices according to x, y and z grids
cld_ext_coeff1 = np.zeros(1, dtype=np.float64)
cld_ext_coeff1[0] = 10.*(1/scale)
reff = np.zeros_like(cld_ext_coeff1, dtype=np.float64)
reff[0] = 10.
cloud3 = Cloud3D('wc', w_ref=800., ext_ref=cld_ext_coeff1,
                 cell_indices=cloud_indices1, reff=reff, phase=cld_phase,
                 ssa_cst=1.)
# **********************************************************************

# ========= homogeneous Rayleigh layer
tau_r = 0.5 # total optical rayleigh depth

dz = diff1(grid3.zGRID)
# homogeneous distri
tau_r_cs = np.cumsum((dz/grid3.zGRID[-1])*tau_r).reshape(1, len(dz))
ot = diff1(tau_r_cs, axis=1)
k  = abs(ot/dz)
k[np.isnan(k)] = 0

sca_ray = k # rayleigh sca coefficient
abs_gas = np.zeros_like(sca_ray)

### profiles computations
wavelengths = np.array([800.])
atm3 = Atm3D(atm_1d=Atm1D('afglt'), grid_3d=grid3, comp_3d=[cloud3],
             wavelength_phase=[800.], mol_sca_1d=sca_ray, mol_abs_1d=abs_gas)
pro_3d3_c2_atm = atm3.calc(wavelengths, n_theta=N_THETA)

surf_c2 = LambSurface(alb=AlbedoCst(0.2))


# %% [markdown]
# #### Function to print results

# %%
def print_c2_res_atm(m, norm, tcase, grid3_sensors, u_sign=1, v_sign=-1,
                     m_i=None, m_q=None, m_u=None, m_v=None):

    if m_i is None:
        i_smartg = m["I_up (TOA)"].values[
        :, 0, 0].reshape(70, 70)*norm
    else:
        i_smartg = np.asarray(m_i).reshape(70, 70)*norm
    # I_SMARTG_stdev = m["I_stdev_up
    # (TOA)"].values[:,0,0].reshape(70,70)*norm
    if m_q is None:
        q_smartg = m["Q_up (TOA)"].values[
        :, 0, 0].reshape(70, 70)*norm
    else:
        q_smartg = np.asarray(m_q).reshape(70, 70)*norm
    # Q_SMARTG_stdev = m["Q_stdev_up
    # (TOA)"].values[:,0,0].reshape(70,70)*norm
    if m_u is None:
        u_smartg = m["U_up (TOA)"].values[
        :, 0, 0].reshape(70, 70)*norm*u_sign
    else:
        u_smartg = np.asarray(m_u).reshape(70, 70)*norm*u_sign
    # U_SMARTG_stdev = m["U_stdev_up
    # (TOA)"].values[:,0,0].reshape(70,70)*norm
    if m_v is None:
        v_smartg = m["V_up (TOA)"].values[
        :, 0, 0].reshape(70, 70)*norm*v_sign
    else:
        v_smartg = np.asarray(m_v).reshape(70, 70)*norm*v_sign
    # V_SMARTG_stdev = m["V_stdev_up
    # (TOA)"].values[:,0,0].reshape(70,70)*norm

    tcase_bis = tcase + int(9)
    file_res = (DIR_AUXDATA / 'IPRT' / 'phaseB' / 'mystic_res'
                / 'iprt_case_C2_mystic.dat')
    read_res = pd.read_csv(
        file_res, skiprows=(4900*(tcase_bis-1)) + 3, nrows=4900,
        header=None, sep='\\s+', dtype=float).values
    i_mystic = read_res[:, 7].reshape(70, 70).T
    # I_MYSTIC_stdev = read_res[:,11].reshape(70,70).T
    q_mystic = read_res[:, 8].reshape(70, 70).T
    # Q_MYSTIC_stdev = read_res[:,12].reshape(70,70).T
    u_mystic = read_res[:, 9].reshape(70, 70).T
    # U_MYSTIC_stdev = read_res[:,13].reshape(70,70).T
    v_mystic = read_res[:, 10].reshape(70, 70).T
    # V_MYSTIC_stdev = read_res[:,14].reshape(70,70).T


    stk = ['I', 'Q', 'U', 'V']
    xgrid = grid3_sensors.xgrid
    ygrid = grid3_sensors.ygrid
    interp_name = 'none'
    fig_size = (10.5, 7)
    font_size=int(16)
    cb_shrink = 1
    color_bar = ['jet', 'coolwarm', 'coolwarm', 'coolwarm']
    vmin = [0., -np.max(np.abs(q_smartg)), -np.max(np.abs(u_smartg)),
            -np.max(np.abs(v_smartg))]
    vmax = [np.max(i_smartg), np.max(np.abs(q_smartg)),
            np.max(np.abs(u_smartg)), np.max(np.abs(v_smartg))]
    color_bar_std = ['coolwarm', 'coolwarm', 'coolwarm', 'coolwarm']
    vmin_std = [-np.max(i_smartg)*0.05, -np.max(np.abs(q_smartg))*0.05,
                -np.max(np.abs(u_smartg))*0.05, -np.max(np.abs(v_smartg))*0.05]
    vmax_std = [np.max(i_smartg)*0.05, np.max(np.abs(q_smartg))*0.05,
                np.max(np.abs(u_smartg))*0.05, np.max(np.abs(v_smartg))*0.05]


    mat_force = [i_smartg, q_smartg, u_smartg, v_smartg]
    satellite_view(m, xgrid, ygrid, interpolation=interp_name, cmap=color_bar,
                   figsize=fig_size, fontsize=font_size, vmin=vmin,
                   vmax=vmax, scale=False, stokes=stk, matrices=mat_force,
                   cbar_shrink=cb_shrink, cbar_sci_format=True,
                   title=f"C2 - case {tcase} - SMART-G - with atm")

    mat_force = [i_smartg-i_mystic, q_smartg-q_mystic, u_smartg-u_mystic,
                 v_smartg-v_mystic]
    satellite_view(
        m, xgrid, ygrid, interpolation=interp_name, cmap=color_bar_std,
        figsize=fig_size, fontsize=font_size, vmin=vmin_std, vmax=vmax_std,
        scale=False, stokes=stk, matrices=mat_force, cbar_shrink=cb_shrink,
        cbar_sci_format=True,
        title=f"C2 - case {tcase} - dif(SMART-G - MYSTIC) - with atm")


    # print deltam
    iquv_smartg = group_iquv(i_list=[i_smartg], q_list=[q_smartg],
                             u_list=[u_smartg], v_list=[v_smartg])
    iquv_mystic = group_iquv(i_list=[i_mystic], q_list=[q_mystic],
                             u_list=[u_mystic], v_list=[v_mystic])

    print("SMART-G (delta_m):")
    delta_m = compute_deltam(obs=iquv_mystic, mod=iquv_smartg, print_res=True)

# %% [markdown]
# ### Backward simulations

# %% [markdown]
# #### Case 1

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[0]

print("pos_z = ", pos_z)
theta = 40.
phi = 0.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 20.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_atm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True, depo=0)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_1_atm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                          n_loop=NB_LOOP, atmosphere=pro_3d3_c2_atm,
                          sensor=sensors, le=le, surface=surf_c2,
                          n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                          stdev=True, depo=0)
norm_c2_1_atm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_1_atm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_1_atm
norm = norm_c2_1_atm
tcase = int(1)

print_c2_res_atm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 2

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[0]

print("pos_z = ", pos_z)
theta = 40.
phi = 60.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 20.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_atm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True, depo=0)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_2_atm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                          n_loop=NB_LOOP, atmosphere=pro_3d3_c2_atm,
                          sensor=sensors, le=le, surface=surf_c2,
                          n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                          stdev=True, depo=0)
norm_c2_2_atm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_2_atm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_2_atm
norm = norm_c2_2_atm
tcase = int(2)

print_c2_res_atm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 3

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[0]

print("pos_z = ", pos_z)
theta = 40.
phi = 120.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 20.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_atm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True, depo=0)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_3_atm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                          n_loop=NB_LOOP, atmosphere=pro_3d3_c2_atm,
                          sensor=sensors, le=le, surface=surf_c2,
                          n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                          stdev=True, depo=0)
norm_c2_3_atm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_3_atm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_3_atm
norm = norm_c2_3_atm
tcase = int(3)

print_c2_res_atm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 4

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[0]

print("pos_z = ", pos_z)
theta = 40.
phi = 180.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 20.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_atm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True, depo=0)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_4_atm = s_3db.run(wavelength=wavelengths, n_photons=NB_PHOTONS,
                          n_loop=NB_LOOP, atmosphere=pro_3d3_c2_atm,
                          sensor=sensors, le=le, surface=surf_c2,
                          n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                          stdev=True, depo=0)
norm_c2_4_atm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_4_atm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_4_atm
norm = norm_c2_4_atm
tcase = int(4)

print_c2_res_atm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 5

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6*scale

print("pos_z = ", pos_z)
theta = 180.
phi = 0.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 40.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_atm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True, depo=0)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_5_atm = s_3db.run(wavelength=wavelengths, n_photons=1e9,
                          n_loop=NB_LOOP, atmosphere=pro_3d3_c2_atm,
                          sensor=sensors, le=le, surface=surf_c2,
                          n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                          stdev=True, depo=0)
norm_c2_5_atm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_5_atm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_5_atm
norm = norm_c2_5_atm
tcase = int(5)

print_c2_res_atm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 6

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6*scale

print("pos_z = ", pos_z)
theta = 140.
phi = 0.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 40.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_atm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True, depo=0)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_6_atm = s_3db.run(wavelength=wavelengths, n_photons=1e9,
                          n_loop=NB_LOOP, atmosphere=pro_3d3_c2_atm,
                          sensor=sensors, le=le, surface=surf_c2,
                          n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                          stdev=True, depo=0)
norm_c2_6_atm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_6_atm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_6_atm
norm = norm_c2_6_atm
tcase = int(6)

print_c2_res_atm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 7

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6*scale

print("pos_z = ", pos_z)
theta = 140.
phi = 60.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 40.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_atm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True, depo=0)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_7_atm = s_3db.run(wavelength=wavelengths, n_photons=1e9,
                          n_loop=NB_LOOP, atmosphere=pro_3d3_c2_atm,
                          sensor=sensors, le=le, surface=surf_c2,
                          n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                          stdev=True, depo=0)
norm_c2_7_atm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_7_atm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_7_atm
norm = norm_c2_7_atm
tcase = int(7)

print_c2_res_atm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 8

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6*scale

print("pos_z = ", pos_z)
theta = 140.
phi = 120.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 40.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_atm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True, depo=0)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_8_atm = s_3db.run(wavelength=wavelengths, n_photons=1e9,
                          n_loop=NB_LOOP, atmosphere=pro_3d3_c2_atm,
                          sensor=sensors, le=le, surface=surf_c2,
                          n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                          stdev=True, depo=0)
norm_c2_8_atm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_8_atm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_8_atm
norm = norm_c2_8_atm
tcase = int(8)

print_c2_res_atm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# #### Case 9

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6*scale

print("pos_z = ", pos_z)
theta = 140.
phi = 180.
# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta,
    ph_deg=phi, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

# Sun position
theta_0    = 40.
phi_0    = 180.
le     = LocalEstimate(th_deg=np.array([theta_0]),
                       phi_deg=np.array([phi_0]),
                       count_level=np.array([0]))  # , zip=True

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3db.run(wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_atm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, stdev=True, depo=0)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_9_atm = s_3db.run(wavelength=wavelengths, n_photons=1e9,
                          n_loop=NB_LOOP, atmosphere=pro_3d3_c2_atm,
                          sensor=sensors, le=le, surface=surf_c2,
                          n_icdf=N_THETA, xblock=xblock_run, xgrid=xgrid_run,
                          stdev=True, depo=0)
norm_c2_9_atm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_9_atm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_9_atm
norm = norm_c2_9_atm
tcase = int(9)

print_c2_res_atm(m, norm, tcase, grid3_sensors)

# %% [markdown]
# ### Forward simulations

# %% [markdown]
# #### Case 1 - 4

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6

print("pos_z = ", pos_z)

# ======= Case 1 - 4
    # Sun position
theta_0    = 20.
phi_0    = 180.

theta_0_bis = 180.- theta_0
phi_0_bis = 180.- phi_0

# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta_0_bis,
    ph_deg=phi_0_bis, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

theta = np.array([40., 40., 40., 40.])
phi = np.array([0., 60., 120., 180.])

le     = LocalEstimate(th_deg=np.array(theta),
                       phi_deg=np.array(phi+180.),
                       count_level=np.array([1, 1, 1, 1]),
                       zip=True)

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3df.run(th_deg=theta_0, ph_deg=phi_0,
                                   wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_atm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, output_layers=3, depo=0)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_1to4_f_atm = s_3df.run(th_deg=theta_0, wavelength=wavelengths,
                               n_photons=NB_PHOTONS, n_loop=NB_LOOP,
                               atmosphere=pro_3d3_c2_atm, sensor=sensors,
                               le=le, surface=surf_c2, n_icdf=N_THETA,
                               xblock=xblock_run, xgrid=xgrid_run,
                               output_layers=3, depo=0)
norm_c2_1to4_f_atm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_1to4_f_atm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_1to4_f_atm
norm = norm_c2_1to4_f_atm
for i in range (0, 4):
    tcase = int(i+1)
    ind_za = round(tcase - 1)
    print("TESTCASE:", tcase)
    print_c2_res_atm(m, norm, tcase, grid3_sensors, u_sign=-1, v_sign=1,
                     m_i=m['I_down (0+)'][:, ind_za],
                     m_q=m['Q_down (0+)'][:, ind_za],
                     m_u=m['U_down (0+)'][:, ind_za],
                     m_v=m['V_down (0+)'][:, ind_za])

# %% [markdown]
# #### Case 5 - 9

# %% [markdown]
# ##### Run

# %%
xgrid_sensors = np.linspace(0., 7., 71)*scale
ygrid_sensors = np.linspace(0., 7., 71)*scale
zgrid_sensors = np.array([0., 1., 2., 3., 4., 5.])*scale
grid3_sensors = Grid3D(xgrid_sensors, ygrid_sensors, zgrid_sensors,
                       periodic=True)

# Placement of sensors
pos_z = grid3_sensors.zGRID[-1]-1e-6

print("pos_z = ", pos_z)

# Sun position
theta_0    = 40.
phi_0    = 180.

theta_0_bis = 180.- theta_0
phi_0_bis = 180.- phi_0

# !!!! grid3 is different than sensors grid !!!
sensors = get_sensors_grid(
    grid3_sensors.xgrid, grid3_sensors.ygrid, pos_z=pos_z, th_deg=theta_0_bis,
    ph_deg=phi_0_bis, fov=0., loc='ATMOS',
    cell_size=grid3_sensors.xgrid[1]-grid3_sensors.xgrid[0], grid_3d=grid3)

theta = np.array([180., 140., 140., 140., 140.])
phi = np.array([0., 0., 60., 120., 180.])

le     = LocalEstimate(th_deg=np.array(180.-theta),
                       phi_deg=np.array(phi+180.),
                       count_level=np.array([0, 0, 0, 0, 0]),
                       zip=True)

if FIND_OPTIMAL_XB_XG:
    k_time = np.inf
    for xgrid_try in X_GRIDS:
        for xblock_try in X_BLOCKS:
                m_test = s_3df.run(th_deg=theta_0, wavelength=wavelengths,
                                   n_photons=CHECK_NB_PHOTONS,
                                   n_loop=CHECK_NB_LOOP,
                                   atmosphere=pro_3d3_c2_atm, sensor=sensors,
                                   le=le, surface=surf_c2, n_icdf=N_THETA,
                                   xblock=xblock_try, xgrid=xgrid_try,
                                   progress=False, output_layers=1, depo=0)
                if float(m_test.attrs['kernel time (s)']) < k_time:
                    k_time = float(m_test.attrs['kernel time (s)'])
                    best_xb = xblock_try
                    best_xg = xgrid_try
                print("time (s) =", m_test.attrs['kernel time (s)'],
                      "; xblock =", xblock_try, "; xgrid =", xgrid_try)
    print("\n\nBest xblock =", best_xb, " : best xgrid =", best_xg)
    xblock_run = best_xb
    xgrid_run = best_xg
else:
    xblock_run = X_BLOCK
    xgrid_run = X_GRID

m_3d_c2_5to9_f_atm = s_3df.run(th_deg=theta_0, wavelength=wavelengths,
                               n_photons=NB_PHOTONS, n_loop=NB_LOOP,
                               atmosphere=pro_3d3_c2_atm, sensor=sensors,
                               le=le, surface=surf_c2, n_icdf=N_THETA,
                               xblock=xblock_run, xgrid=xgrid_run,
                               output_layers=1, depo=0)
norm_c2_5to9_f_atm = np.cos(np.radians(theta_0))/np.pi
print("kernel time (s) =",
      "{:.2f}".format(float(m_3d_c2_5to9_f_atm.attrs['kernel time (s)'])))

# %% [markdown]
# ##### Results

# %%
m = m_3d_c2_5to9_f_atm
norm = norm_c2_5to9_f_atm
for i in range (0, 5):
    tcase = int(i+5)
    ind_za = round(tcase - 5)
    print("TESTCASE:", tcase)
    print_c2_res_atm(m, norm, tcase, grid3_sensors, u_sign=-1, v_sign=1,
                     m_i=m['I_up (TOA)'][:, ind_za],
                     m_q=m['Q_up (TOA)'][:, ind_za],
                     m_u=m['U_up (TOA)'][:, ind_za],
                     m_v=m['V_up (TOA)'][:, ind_za])
