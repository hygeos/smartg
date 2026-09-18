# %% [markdown]
# # Smart-G demo notebook
#
# This is an interactive document allowing to run Smart-G with python and visualize the results. <br>
# *Tips*: cells can be executed with shift-enter. Tooltips can be obtained with shift-tab. More information [here](http://ipython.org/notebook.html) or in the help menu. [A table of content can also be added](https://github.com/minrk/ipython_extensions#table-of-contents).

# %% [markdown]
# ## Symbols used in this notebook
#
# | symbol | meaning |
# |---|---|
# | `stk_i`, `stk_q`, `stk_u`, `stk_v` | Stokes components of the radiance |
# | `lp`, `dolp` | linearly polarized radiance, and its degree in percent |
# | `sza`, `saa` | solar zenith and azimuth angle, in degrees |
# | `vza`, `vaa` | viewing zenith and azimuth angle |
# | `raa` | relative azimuth, `vaa - saa`; `raa = 0` is the principal plane |
# | `th0`, `th_deg`, `ph_deg` | the angles a run is given, in degrees |
# | `h_toa`, `r_ter` | top-of-atmosphere height and Earth radius, in km |
# | `zts` | tangent heights of the limb sensors, in km |
# | `t_up`, `t_down`, `t_dif` | direct, and diffuse, transmissions |
# | `K_VIS`, `KP_VIS` | Ross-Thick Li-Sparse kernel weights, absolute and relative to `k0` |
# | `aot` | aerosol optical thickness |
#
# The Stokes naming follows `smartg.view`, which calls the four
# components `stk_i` to `stk_v`.

# %%
# %matplotlib inline
# the next 2 lines allow to automatically reload modules that have
# been changed externally
# %reload_ext autoreload
# %autoreload 2
import sys, os
from pathlib import Path

# ── Make sure nvcc is on PATH (required by pycuda/smartg kernel
# compilation) ──
_cuda_bin = Path('/usr/local/cuda/bin')
if _cuda_bin.exists() and str(_cuda_bin) not in os.environ.get('PATH', ''):
    os.environ['PATH'] = str(_cuda_bin) + ':' + os.environ.get('PATH', '')

try:
    import subprocess
    check = subprocess.check_call(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    # Root Git Path
    ROOTPATH = subprocess.Popen(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
    ROOTPATH = Path(ROOTPATH)
except subprocess.CalledProcessError:
    ROOTPATH = Path.cwd()
sys.path.insert(0, str(ROOTPATH))

from smartg.config import DIR_AUXDATA
from smartg.smartg import Alis, LocalEstimate, Smartg
from smartg.sensor import Sensor
from smartg.surface import RoughSurface, LambSurface, Environment, RTLSSurface
from smartg.atmosphere import Atm1D, Cloud, AerOPAC
from smartg.phase import read_phase
from smartg.water import Water1D, Hydrosol, HydrosolPR
from smartg.albedo import AlbedoSpeclib, AlbedoCst, AlbedoSpectrum, AlbedoMap
from smartg.reptran import Reptran, reduce_reptran
from smartg.postprocess import irradiance_ds
from smartg.xarray import drop_axes
from smartg.view import spectrum_view, transect_view, smartg_view, input_view
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('always', DeprecationWarning)

# %% [markdown]
# # Quick Start

# %% [markdown]
# ## Run your first simulation

# %%
m = Smartg().run(wavelength=500., th_deg=30., n_photons=1e9,
                 atmosphere=Atm1D('afglt'))

# %% [markdown]
# ## How to view the results

# %%
# basic visualization
fig = smartg_view(m)

# %%
# access a variable (DataArray) within the Dataset
stk_i = m['I_up (TOA)']

# %%
print(stk_i)

# %%
# access values in this DataArray
print(stk_i[5, ::5].values)  # by index
# or by coordinate value with interpolation
print(stk_i.interp({'Azimuth angles': 90., 'Zenith angles': 45.}).values)

# %%
# calculate polarized light
# use operations between DataArrays, and apply sqrt
stk_q, stk_u = m['Q_up (TOA)'], m['U_up (TOA)']
lp = np.sqrt(stk_q*stk_q + stk_u*stk_u).rename('Lp_up (TOA)')

# %%
# 2D visualization (azimuth vs zenith map)
lp.plot()

# %%
# transect at a given azimuth angle (nearest neighbour selection)
stk_i.sel({'Azimuth angles': 45.}, method='nearest').plot()
plt.ylim(0, 0.3)

# %%
# interpolate at an azimuth angle (here 2D -> 1D)
stk_i.interp({'Azimuth angles': 90.}).plot()

# %% [markdown]
# ## A few examples

# %%
# Basic Rayleigh example, no surface, full view
fig = smartg_view(
    Smartg(rng='CURAND_PHILOX').run(wavelength=500., th_deg=30.,
                                    n_photons=1e9,
                                    atmosphere=Atm1D('afglt')),
    full=True)

# %%
# only Rayleigh with a custom grid, a custom depolarization ratio
# and a custom surface pressure
fig = smartg_view(Smartg().run(
    wavelength=500., th_deg=30., depo=0., n_photons=1e8,
    atmosphere=Atm1D('afglt', grid=np.linspace(100., 0., num=16),
                     p0=990.)))

# %%
# Rayleigh + aerosols (UV), downward radiance at the ground
aer = AerOPAC('urban', 0.4, 550.)
pro = Atm1D('afglms', comp=[aer])
fig = smartg_view(Smartg().run(wavelength=322., th_deg=60.,
                               atmosphere=pro, n_photons=1e8,
                               output_layers=1),
                  field='down (0+)')

# %%
# atmosphere + surface
wavelength= 490.
azimuth_transect = 10.
m_as = Smartg().run(wavelength, n_photons=1e8, th_deg=45.,
        atmosphere=pro,
        surface=LambSurface(alb=AlbedoCst(0.1)))
iaz = int(np.abs(m_as['Azimuth angles'].values - azimuth_transect).argmin())
fig= smartg_view(m_as, ind=[iaz], qu=True)

# %%
print(m)

# %%
# atmosphere + surface + océan
atmosphere = Atm1D('afglms', tco3=0., no2=False)
surface = RoughSurface(wind=5., nh2o=1.34)
# Case I water Inherent Optical Properties depending on Chlorophyll
# concentration only
water = Water1D(grid=[0., -5.], comp=[HydrosolPR(chl=0.5)])
# compute outputs at the surface and bottom of ocean also,
# view results for upwelling and downwelling radiances
wavelength = 500.
th0 = 60.
m = Smartg().run(wavelength, n_photons=1e9, th_deg=th0,
                 output_layers=3, beer=1, atmosphere=atmosphere,
                 surface=surface, water=water)
fields = ['up (TOA)', 'up (0+)', 'up (0-)', 'down (0+)', 'down (0-)',
          'down (B)']
log_is = [True, False, True, True, True, False]
for log_i, field in zip(log_is, fields):
    fig = smartg_view(m, field=field, log_i=log_i)

# %%
# surface + ocean
atmosphere = None
surface = RoughSurface(wind=5., nh2o=1.34)
water = Water1D(grid=[0., -5.], comp=[HydrosolPR(chl=0.5)])
# compute outputs at the surface also, view results for downwelling
# at bottom of ocean
fig = smartg_view(Smartg().run(wavelength, n_photons=1e9, th_deg=th0,
                               output_layers=3, beer=1,
                               atmosphere=None, surface=surface,
                               water=water),
                  field='down (B)', log_i=True)

# %%
# ocean only
atmosphere = None
surface = None
water = Water1D(grid=[0., -5.], comp=[HydrosolPR(chl=0.5)])
# compute outputs at the surface also, view results for downwelling
# at bottom of ocean
fig = smartg_view(Smartg().run(wavelength, n_photons=1e9, th_deg=th0,
                               output_layers=3, beer=1,
                               atmosphere=None, surface=surface,
                               water=water),
                  field='down (B)', log_i=True)

# %% [markdown]
# # Advanced use

# %% [markdown]
# ## Local Estimate

# %%
# %%time
# In order to compute accurately radiances in particular directions, set
# the LE mode
# 'le' keyword is a dictionnary containig directions vectors (in radians
# or deg) and coded as float32
# by default the result is calculated for each Ntheta x Nphi combination
le = LocalEstimate(th_deg=np.linspace(0, 89., num=8),
                   phi_deg=np.linspace(360, 0., num=12))
# revisit atmosphere + surface + ocean
# the number of photons should be dramatically reduces since
# each photon participates to the computation of all directions
wavelength = 500.
th0= 60.
m = Smartg().run(wavelength=wavelength, n_photons=1e6, th_deg=th0,
                 output_layers=3, le=le,
                 atmosphere=Atm1D('afglms', tco3=0., no2=False),
                 surface=RoughSurface(wind=5., nh2o=1.34),
                 water=Water1D(grid=[0., -5.], comp=[HydrosolPR(chl=0.5)]))

fig= smartg_view(m, field='up (0-)', log_i=True, i_min=-2.5, i_max=-1.)

# %%
# you can also speficy only couples of zenith and azimuth angles,
# resulting in only 1-dimensional Ncouple output
# using the keyword 'zip':True in the le dictionary. in that case
# Nphi=Ntheta=Ncouple.
# You can also specify angles in degree
n_dir  = 12
le = LocalEstimate(th_deg=np.linspace(0, 89., num=n_dir),
                   phi_deg=np.linspace(360., 0., num=n_dir),
                   zip=True)
mz = Smartg().run(wavelength=wavelength, n_photons=1e6, th_deg=th0,
                  output_layers=3, le=le,
                  atmosphere=Atm1D('afglms', tco3=0., no2=False),
                  surface=RoughSurface(wind=5., nh2o=1.34),
                  water=Water1D(grid=[0., -5.], comp=[HydrosolPR(chl=0.5)]))
# the result has an "Azimuth angles" coordinate, but no variable depends
# on it
print(mz)

# %%
# Comparison of outputs
# polar scatter plot, with color proportional to I and size to DoLP
fig  = plt.figure()
ax = fig.add_subplot(111, projection='polar')
stk_i, stk_q, stk_u = m['I_up (0-)'], m['Q_up (0-)'], m['U_up (0-)']
dolp = np.sqrt((stk_q*stk_q + stk_u*stk_u)/(stk_i*stk_i)*100)
rad, theta = np.meshgrid(m['Zenith angles'].values,
                         m['Azimuth angles'].values/180*np.pi)
ax.scatter(theta.ravel(), rad.ravel(), c=np.log10(stk_i.data.ravel()),
           s=dolp.data.ravel()*50, cmap='jet', alpha=0.5, vmin=-2.5, vmax=-1.)

stk_i, stk_q, stk_u = mz['I_up (0-)'], mz['Q_up (0-)'], mz['U_up (0-)']
dolp = np.sqrt((stk_q*stk_q + stk_u*stk_u)/(stk_i*stk_i)*100)
ax.set_ylim([0, 90])
ax.scatter(mz['Azimuth angles'].values/180*np.pi, mz['Zenith angles'].values,
           c=np.log10(stk_i.data), s=dolp.data*50, cmap='jet', alpha=0.5,
           vmin=-2.5, vmax=-1., marker='s', edgecolors='k')

# %% [markdown]
# ## Atmospheric profile
# Aerosols and Cloud optical properties are those of OPAC. The files containing these properties have been
# generated by www.libradtran.org. <br>
# For the moment Only spherical particles are handled

# %%
# monochromatic computation for custom aerosols and cloud
wlref = 550.
# Aerosols and cloud optical properties using OPAC database as processed
# by the the libradtran (www.libradtran.org)
# set AOT at the reference wavelength wavelength to 0.3
aer = AerOPAC( 'desert', 0.3, wlref)
                                # and set aerosol type to 'desert'
# set cloud to water cloud with reff=11 mic.
cld = Cloud('wc', 11., 2, 3., 1., wlref)
                                # the cloud is located between 2. and 3
                                # km, with
                                # Optical thickness at the reference
                                # wavelength wavelength set to 1.

pro = Atm1D('afglt',    # tropical atmosphere
              # particles in atmosphere are a mix of aerosols and cloud
              comp=[aer, cld],
              # scale ozone vertical column to 0 Dobson units (here no
              # absoprtion by ozone)
              tco3=0.,
              no2=False, # disable absorption by NO2
              # scale water vapour column to 2 g/cm-2, but no H2O
              # absoprtion, just hygroscopic computation for aerosols
              tcwp=2.,
              p0=980., # set sea level pressure to 980 hPa
              tau_r=0.1,   # force Rayleigh optical thickness
              # set vertical grid, surface altitude at 1.15 km
              grid=[100., 75., 50., 25., 15., 10., 5., 3., 2., 1.15],
              # vertical grid for the computation of particles phase
              # functions
              pfgrid=[100., 25., 15., 10., 5., 3., 2., 1.15]
             )

azimuth_transect = 30.
m = Smartg().run(wavelength=wlref, th_deg=60., atmosphere=pro, n_photons=1e9)
iaz = int(np.abs(m['Azimuth angles'].values - azimuth_transect).argmin())
_ = smartg_view(m, ind=[iaz], log_i=True)

# %%
# View inputs
input_view(m, kind='atm', zmax=50)

# %%
# Full aerosol customization
# 1) import aerosols scattering matrix from external text file with 5
# columns:
# angle, P11, P12, P33, P43 --> pha=read_phase(DIR_AUXDATA /
# '/validation/opt_kokha_aer_standard.dat')
# OR angle, F11 = (P11+P12)/2, F22=(P11-P12)/2, F33=P33, F43=P43 (Smartg
# Iparper convention)
pha=read_phase(DIR_AUXDATA / 'validation' / 'opt_kokha_aer_nostandard.dat')
# the conversion into the smartg Iparper convention is done by run
pha=read_phase(DIR_AUXDATA / 'validation' / 'opt_kokha_aer_standard.dat')

# 2) Set single scattering albedo of aerosols to 0.80 for each layer and
# set the aerosol phase function
aer=AerOPAC('maritime_clean', 0.3262, wlref, ssa=0.80, phase=pha)

#3) build profile
atm_custom = Atm1D('afglmw', comp=[aer]
                    # 4) Could also import aerosol profiles (extinction
                    # and ssa) from external files
                    #,prof_aer= (aer_ext_valid,aer_ssa_valid)
                    )

azimuth_transect = 5.
m = Smartg().run(wavelength=wlref, th_deg=60., atmosphere=atm_custom,
                 n_photons=1e9)
iaz = int(np.abs(m['Azimuth angles'].values - azimuth_transect).argmin())
_ = smartg_view(m, ind=[iaz])

# %% [markdown]
# ## Truncation
#
# An example using a water cloud and 2 differents truncation methods, from Iwabuchi et al. 2009

# %%
from smartg.truncation import GTTrunc

sza = np.array([60.])
saa = np.array([180.])

vza = np.linspace(-89., 89., 179)
vaa = np.array([180.])

le = LocalEstimate(th_deg=vza, phi_deg=vaa)
stg = Smartg(pp=True, double=True, device=0)

zgrid = np.array([1., 0.])
cld1 = Cloud('wc', 8., 0., 1., 1., 500., ssa=1.)
cld5 = Cloud('wc', 8., 0., 1., 5., 500., ssa=1.)
cld20 = Cloud('wc', 8., 0., 1., 20., 500., ssa=1.)

nph = 1e6
nbloop = 1e5

# %% [markdown]
# ### Reference radiances (without truncation) 

# %%
# tau = 1
atmosphere = Atm1D('afglt', comp=[cld1], grid=zgrid, tco3=0., no2=False,
                   tcwp=0., tau_r=0.).calc(500., n_theta=18001)

m1 = stg.run(wavelength=500., atmosphere=atmosphere, ph_deg=saa[0],
             th_deg=sza[0], le=le, n_loop=nbloop, n_photons=nph,
             output_layers=7, reflectance=False)

# tau = 5
atmosphere = Atm1D('afglt', comp=[cld5], grid=zgrid, tco3=0., no2=False,
                   tcwp=0., tau_r=0.).calc(500., n_theta=18001)

m5 = stg.run(wavelength=500., atmosphere=atmosphere, ph_deg=saa[0],
             th_deg=sza[0], le=le, n_loop=nbloop, n_photons=nph,
             output_layers=7, reflectance=False)

# tau = 20
atmosphere = Atm1D('afglt', comp=[cld20], grid=zgrid, tco3=0., no2=False,
                   tcwp=0., tau_r=0.).calc(500., n_theta=18001)

m20 = stg.run(wavelength=500., atmosphere=atmosphere, ph_deg=saa[0],
              th_deg=sza[0], le=le, n_loop=nbloop, n_photons=nph,
              output_layers=7, reflectance=False)

# plot iwabuchi Fig. 3a anb 3c
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs = axs.ravel()
axs[0].plot(vza, m1['I_up (TOA)'][0, :][::-1] , 'b-', label=r'$\tau$ = 1')
axs[0].plot(vza, m5['I_up (TOA)'][0, :][::-1] , 'g-', label=r'$\tau$ = 5')
axs[0].plot(vza, m20['I_up (TOA)'][0, :][::-1] , 'r-', label=r'$\tau$ = 20')
axs[0].set_yscale('log')
axs[0].legend()
axs[0].set_ylim(1e-2, 1e1)
axs[0].set_xlim(-89, 89)
axs[0].set_title('I (reflection)')
axs[1].plot(vza, m1['I_down (0+)'][0, :] , 'b-', label=r'$\tau$ = 1')
axs[1].plot(vza, m5['I_down (0+)'][0, :] , 'g-', label=r'$\tau$ = 5')
axs[1].plot(vza, m20['I_down (0+)'][0, :] , 'r-', label=r'$\tau$ = 20')
axs[1].set_yscale('log')
axs[1].legend()
axs[1].set_ylim(1e-2, 1e3)
axs[1].set_xlim(-89, 89)
axs[1].set_title('I (transmission)')
plt.tight_layout()

# %% [markdown]
# ### radiances with GT truncation

# %%
# simple GT trunction (without correction) -> scheme S in Iwabuchi et
# al. 2009
trunc = GTTrunc(trunc_frac=0.435, theta_tol=20, theta_tr=None,
                 integral_method='lobatto', lobatto_optimization=True)

# tau = 1
atmosphere = Atm1D(
    'afglt', comp=[cld1], grid=zgrid, tco3=0., no2=False,
    tcwp=0., tau_r=0.).calc(500., n_theta=18001,
                           truncation=trunc)

m1_gt = stg.run(wavelength=500., atmosphere=atmosphere, ph_deg=saa[0],
                th_deg=sza[0], le=le, n_loop=nbloop, n_photons=nph,
                output_layers=7, reflectance=False)

# tau = 5
atmosphere = Atm1D(
    'afglt', comp=[cld5], grid=zgrid, tco3=0., no2=False,
    tcwp=0., tau_r=0.).calc(500., n_theta=18001,
                           truncation=trunc)

m5_gt = stg.run(wavelength=500., atmosphere=atmosphere, ph_deg=saa[0],
                th_deg=sza[0], le=le, n_loop=nbloop, n_photons=nph,
                output_layers=7, reflectance=False)

# tau = 20
atmosphere = Atm1D(
    'afglt', comp=[cld20], grid=zgrid, tco3=0., no2=False,
    tcwp=0., tau_r=0.).calc(500., n_theta=18001,
                           truncation=trunc)

m20_gt = stg.run(wavelength=500., atmosphere=atmosphere, ph_deg=saa[0],
                 th_deg=sza[0], le=le, n_loop=nbloop, n_photons=nph,
                 output_layers=7, reflectance=False)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs = axs.ravel()
axs[0].plot(vza, m1_gt['I_up (TOA)'][0, :][::-1] , 'b-', label=r'$\tau$ = 1')
axs[0].plot(vza, m5_gt['I_up (TOA)'][0, :][::-1] , 'g-', label=r'$\tau$ = 5')
axs[0].plot(vza, m20_gt['I_up (TOA)'][0, :][::-1] , 'r-', label=r'$\tau$ = 20')
axs[0].set_yscale('log')
axs[0].legend()
axs[0].set_ylim(1e-2, 1e1)
axs[0].set_xlim(-89, 89)
axs[0].set_title('I (reflection)')
axs[1].plot(vza, m1_gt['I_down (0+)'][0, :] , 'b-', label=r'$\tau$ = 1')
axs[1].plot(vza, m5_gt['I_down (0+)'][0, :] , 'g-', label=r'$\tau$ = 5')
axs[1].plot(vza, m20_gt['I_down (0+)'][0, :] , 'r-', label=r'$\tau$ = 20')
axs[1].set_yscale('log')
axs[1].legend()
axs[1].set_ylim(1e-2, 1e3)
axs[1].set_xlim(-89, 89)
axs[1].set_title('I (transmission)')
plt.tight_layout()

# %% [markdown]
# ## Multispectral

# %% [markdown]
# ### Independant spectral computations

# %% tags=["Definition"]
# %%time
# multispectral simulation
# computation done independently for each wavelegnth, N_PHOTONS is
# shared equally
# between all wavelegths (thus Monte Carlo NOISE in the spectrum)
# wavelengths is a list or numpy array
n_wl= 301
wavelength = np.linspace(400., 430., num=n_wl)
n_photons = 1e5 # photons per wavelength
# monochromatic computation for custom aerosols and cloud
wavelength_0 = 415.
# Aerosols and cloud optical properties using OPAC database as processed
# by the the libradtran (www.libradtran.org)
# set AOT at the reference wavelength wavelength_0 to 0.5
aer1 = AerOPAC( 'urban',  0.5, wavelength_0)
                                # and set aerosol type to 'urban'
pro = Atm1D('afglt',    # tropical atmosphere
              # particles in atmosphere are a mix of aerosols 1 and 2
              comp=[aer1],
              # set vertical grid, surface altitude at 1.15 km
              grid=[100, 75, 50, 30, 20, 10, 5, 2., 1.15],
              pfgrid=[100, 10, 5, 2., 1.15],
              # wavelengths for which the phase function is computed
              wavelength_phase=[400., 430.],
                                     # optional, otherwise phase
                                     # functions are calculated at all
                                     # bands
                                     # nearest neighbour is then used
                                     # during the RT computation
              no2=True, # NO2 included
              tco3=0. # no ozone
             )

GREY_ALB = AlbedoCst(0.1)
surface = LambSurface(alb=GREY_ALB)
le = LocalEstimate(th=np.array([30., 30.]) * np.pi/180,
                   phi=np.array([45., 60.]) * np.pi/180,
                   zip=True)
m = Smartg(alt_pp=True, double=True).run(wavelength=wavelength, le=le,
           th_deg=60., n_photons=n_photons*n_wl,
           atmosphere=pro, output_layers=1,
           surface=surface)
m = drop_axes(m, 'Azimuth angles')
plt.plot(m['wavelength'], m['I_up (TOA)'][:, 0], '+r',
         label=r'$\Delta\Phi=${:.0f}°'.format(le.phi[0]*180/np.pi))
plt.plot(m['wavelength'], m['I_up (TOA)'][:, 1], '.r',
         label=r'$\Delta\Phi=${:.0f}°'.format(le.phi[1]*180/np.pi))
plt.ylim(0.21, 0.27)
plt.legend()
print(' GPU time: ', m.attrs['kernel time (s)'], 's')

# %% [markdown]
# ### Profiles reuse

# %%
# %%time
# once computed, profiles may be re-used
# it could save a lot of time for multi spectral computation
m2 = Smartg(alt_pp=True, double=True).run(wavelength=wavelength, le=le,
           th_deg=60., n_photons=n_photons*n_wl,
           atmosphere=m, output_layers=1,
           surface=surface)
m2 = drop_axes(m2, 'Azimuth angles')
plt.plot(m2['wavelength'], m2['I_up (TOA)'][:, 0], '+c',
         label=r'$\Delta\Phi=${:.0f}°'.format(le.phi[0]*180/np.pi))
plt.plot(m2['wavelength'], m2['I_up (TOA)'][:, 1], '.c',
         label=r'$\Delta\Phi=${:.0f}°'.format(le.phi[1]*180/np.pi))
plt.ylim(0.21, 0.27)
plt.legend()
print(' GPU time: ', m2.attrs['kernel time (s)'], 's')

# %% [markdown]
# ### Correlated spectral computations: ALIS method 
# The ALIS method is described in <br>
# Emde, C., Buras, R., and Mayer, B.: ALIS: An efficient method to
# compute high spectral resolution polarized solar radiances using the Monte Carlo approach, J. Quant. Spectrosc. Ra., 112, 1622–1631, 2011.

# %%
# %%time
# Compile with the alis options
# with ALIS options, alt_pp is mandatory
s_alis = Smartg(alt_pp=True, double=True, alis=True, amf_variance=True,
                nscl=14, norders=2,
                scatter_classes='scattering_order_per_layer', cdist_wabs=False)
# then run specifying the alis_options
# main keyword is n_low, the number of low spectral resolution
# computation
N_LOW = 31
wavelength_lr = np.linspace(wavelength.min(), wavelength.max(), num=N_LOW)
# of the scattering correction terms: specify -1 for all wavelengths
# if the alt_pp option is chosen for compilation: slowest procedure for
# photon propagation but
# then the cumulative distance traveled by photons in the atmospheric
# layer is recorded

m3 = s_alis.run(wavelength=wavelength, le=le, alis_options=Alis(n_low=N_LOW),
           th_deg=60., n_photons=n_photons, stdev=True,
           atmosphere=m, output_layers=0,
           surface=surface)
m3 = drop_axes(m3, 'Azimuth angles')

# for the same number of photons, the spectrum is much less noisy
plt.plot(m['wavelength'], m['I_up (TOA)'][:, 0], '+r',
         label='no alis 1 : {:.0e} phot.; {:.5f} (s)'.format(
             n_photons*n_wl,
             float(m.attrs['kernel time (s)'])))
plt.plot(m['wavelength'],  m['I_up (TOA)'][:, 1], '.r')
plt.plot(m2['wavelength'], m2['I_up (TOA)'][:, 0], '+c',
         label='no alis 2 : {:.0e} phot.; {:.5f} (s)'.format(
             n_photons*n_wl,
             float(m2.attrs['kernel time (s)'])))
plt.plot(m2['wavelength'], m2['I_up (TOA)'][:, 1], '.c')
plt.plot(m3['wavelength'], m3['I_up (TOA)'][:, 0], '-k',
         label='alis          :{:.0e} phot.; {:.5f} (s)'.format(
             n_photons,
             float(m3.attrs['kernel time (s)'])))
plt.plot(m3['wavelength'], m3['I_up (TOA)'][:, 1], '-k')
plt.legend()
plt.ylim(0.21, 0.27)
print(' GPU time: ', m3.attrs['kernel time (s)'], 's')

# %%
# As remarked in Emde et al., 2011, alis method is very efficient for
# DOAS.
# example : differences in spectra with and without NO2

pro0 = Atm1D('afglt', no2=False, tco3=0.,
              comp=[aer1],
              grid=[100, 75, 50, 30, 20, 10, 5, 2., 1.15],
              pfgrid=[100, 10, 5, 2., 1.15],
              wavelength_phase=[400., 430.])

m0 = s_alis.run(wavelength=wavelength, le=le, alis_options=Alis(n_low=N_LOW),
           th_deg=60., n_photons=n_photons, stdev=True,
           atmosphere=pro0, output_layers=1,
           surface=surface)
m0 = drop_axes(m0, 'Azimuth angles')

print(' GPU time: ', m0.attrs['kernel time (s)'], 's')

# %%
# Differential absorption
diff_abs = (m3['I_up (TOA)']/m0['I_up (TOA)'])[:, 0]
plt.plot(diff_abs['wavelength'], diff_abs, '+b-')
diff_abs = (m3['I_up (TOA)']/m0['I_up (TOA)'])[:, 1]
plt.plot(diff_abs['wavelength'], diff_abs, '.b-')
# Overplot NO2 direct transmission
t_down = np.exp(m3['OD_g'][:, -1]*(-1)/np.cos(np.radians(60.)))
t_up   = np.exp(m3['OD_g'][:, -1]*(-1)/np.cos(np.radians(30.)))
t_direct = t_up*t_down
plt.plot(t_direct['wavelength'], t_direct, 'r.-')
plt.ylim(0.97, 1.)
doas = r'DOAS at the TOA; $\Delta\Phi=${:.0f}°'
plt.legend([doas.format(le.phi[0]*180/np.pi),
            doas.format(le.phi[1]*180/np.pi),
            'Direct gaseous transmission Sun-ground-sensor'])

# %% [markdown]
# ## Water profile
# Full profile customization

# %%
wavelength  = np.array([450., 550.]) # N=2
grid= [0., -2.5, -5., -7.5, -10.]
# Seafloor albedo, here grey lambertian reflection with albedo of 0.05
alb=AlbedoCst(0.05)

#1) Read one phase matrix for particles
pha=read_phase(DIR_AUXDATA / 'validation' / 'opt_hydrosols.dat', kind='oc')
# 2) import vertical profiles of pure water absorption and scattering
# coefficient,
# particle absorption and scattering coefficient profiles (in m-1)
# with shape (N wavelengths, M levels)
# !! level Number 0 is not used, the first layer (L=0) is homogeneous
# with coefficient
# index 1, layer number L-1 is homogeneous with coefficient index M-1

atot = np.array([[0., 0.21841, 0.05465, 0.00924, 0.00924],
              [0., 0.09881, 0.06415, 0.05650, 0.05650]])
ap   = np.array([[0., 1.25744e-01, 1.34828e-02, 4.79320e-10, 4.79320e-10],
              [0., 3.10215e-02, 3.32626e-03, 1.18250e-10, 1.18250e-10]])
acdom= np.array([[0., 8.34474e-02, 3.19477e-02, 2.00125e-05, 2.00125e-05],
              [0., 1.12934e-02, 4.32364e-03, 2.70840e-06, 2.70840e-06]])
aw   = np.array([[0., 0.00922, 0.00922, 0.00922, 0.00922],
              [0., 0.05650, 0.05650, 0.05650, 0.05650]])
bw   = np.array([[0., 0.00459, 0.00459, 0.00459, 0.00459],
              [0., 0.00193, 0.00193, 0.00193, 0.00193]])
bp   = np.array([[0., 2.83546e-01, 3.04030e-02, 1.08084e-09, 1.08084e-09],
              [0., 2.31992e-01, 2.48752e-02, 8.84323e-10, 8.84323e-10]])

water_custom = Water1D(grid=grid, aw=aw, bw=bw, alb=alb,
                       comp=[Hydrosol(phase=pha, bp=bp, acdom=acdom, ap=ap)])

azimuth_transect = 30.
m1 = Smartg().run(wavelength=wavelength, th_deg=60., water=water_custom,
                  n_photons=1e9, n_phi=24, n_theta=24, n_loop=1e8)
iaz = int(np.abs(m1['Azimuth angles'].values - azimuth_transect).argmin())
_ = smartg_view(m1, ind=[iaz], interp_dict={'wavelength': 450.}, i_min=0,
                i_max=0.015)
_ = smartg_view(m1, ind=[iaz], interp_dict={'wavelength': 550.}, i_min=0,
                i_max=0.015)

# %%
fig = input_view(m1,  kind='oc', iw=0, zmax=-10)

# %% [markdown]
# ## Absorption

# %% [markdown]
# ### Monochromatic absorption

# %%
# standard gaseous absorption with ozone and NO2
# we use the single scattering albedo (ssa) method for computing
# absorption (beer=0)
# this is the default method
atmosphere = Atm1D('afglms', tco3=300., no2=True
             # optionally import gaseous absorption vertical profile
             #,prof_abs= gas_profile
              )
_ = smartg_view(Smartg().run(th_deg=60., wavelength=550., n_photons=1e9,
                             atmosphere=atmosphere, beer=0))
# we can also use the equivalent theorem (Beer Lambert law) method for
# computing absorption (beer=1)
_ = smartg_view(Smartg().run(th_deg=60., wavelength=550., n_photons=1e9,
                             atmosphere=atmosphere, beer=1))

# %% [markdown]
# ### Band gaseous absorption using REPTRAN
# <br> REPTRAN is described in Gasteiger et al., 2014 and the data is available at www.libradtran.org <br>
# J. Gasteiger, C. Emde, B. Mayer, R. Buras, S.A. Buehler, O. Lemke, Representative wavelengths absorption parameterization applied to satellite channels and spectral bands, Journal of Quantitative Spectroscopy and Radiative Transfer, Volume 148, November 2014, Pages 99-115, ISSN 0022-4073, http://dx.doi.org/10.1016/j.jqsrt.2014.06.024.

# %% [markdown]
# #### Example 1: Computation of reflectance in MSG-SEVIRI VIS08 channel

# %%
# REPTRAN k distribution file here MSG/SEVIRI solar channels
SEVIRI_SOLAR = Reptran('reptran_solar_msg')

# several ways of selecting bands
# 1) selecting all bands
ibands = SEVIRI_SOLAR.to_smartg()
# 2) selecting one specific band
ibands = SEVIRI_SOLAR.to_smartg(include='msg1_seviri_ch008')
# 3) selecting all bands that contains "msg1" in the band name
ibands = SEVIRI_SOLAR.to_smartg(include='msg1')
# 4) selecting all bands whose wavelengths of internal bands satisfy the
# min and max conditions
ibands = SEVIRI_SOLAR.to_smartg(lmax=700.)
# 5) a mix of 3) and 4)
ibands = SEVIRI_SOLAR.to_smartg(include='msg1', lmin=600., lmax=1000.)
# 6) note that lmin and lmax can be lists (they should have the same
# length)
ibands = SEVIRI_SOLAR.to_smartg(include='msg1', lmin=[600., 1400.],
                                lmax=[700., 2000.])

surface= RoughSurface(sur=1, wind=5., nh2o=1.34)
atmosphere = Atm1D('afglms', tcwp=4.)

# Run Smart-g for Reptran list of ibands
m1  = Smartg(double=True).run(th_deg=30, wavelength=ibands.l, n_photons=1e9,
                              atmosphere=atmosphere, surface=surface, beer=1,
                              progress=False)

# Postprocessing: regrouping internal bands information into real band
# (spectral integration)
m1r = reduce_reptran(m1, ibands)

# Plotting bands
iaz = int(np.abs(m1r['Azimuth angles'].values - 0.).argmin())
for i, w in enumerate(m1r['wavelength'].values):
    # we use for that a subset of the m1r Dataset
    _ = smartg_view(m1r.isel(wavelength=i), ind=[iaz])
    print(ibands.get_names()[i])

# %% [markdown]
# #### Example 2: Polarized reflectance in O2A band in sun glint view with Local Estimate

# %%
# %%time
# k distribution file, here full solar channels at coarse resolution
SOLAR_COARSE = Reptran('reptran_solar_coarse')
ibands = SOLAR_COARSE.to_smartg(lmin=757., lmax=770.) # within O2A bands
surface=RoughSurface(sur=1, wind=5., nh2o=1.34)
atmosphere=Atm1D('afglms', p0=900.,
                 comp=[AerOPAC('continental_average', 1., 764.)],
                 wavelength_phase=[764.])
# Evaluate reflectance in specific direction Ths=[60.] and raa=[180.]
# using LE
# the directions, in radians and coded as float32
le = LocalEstimate(
    th=np.array([60.], dtype=np.float32)*np.pi/180,
    phi=np.array([180.], dtype=np.float32)*np.pi/180)
spp_mult=Smartg(double=True)

# %%
# %%time
# Compute first atmosphere profiles at all wavelengths and phase
# functions for re-use
atmbase=atmosphere.calc(ibands.l)

# %%
# %%time
# Several runs: it goes faster as long as atmosphere does not change
for col, th0 in zip(['r', 'g', 'b', 'k'], [0., 30., 60., 75.]):
    # we use the single scattering albedo (ssa) method for computing
    # absorption (beer=0)
    # this is the default method
    m2_ssa = spp_mult.run(th_deg=th0, wavelength=ibands.l, n_photons=1e7,
                          beer=0, atmosphere=atmbase, surface=surface, le=le,
                          progress=False)
    # the local estimate grid holds a single direction, squeeze it
    m2r_ssa = drop_axes(reduce_reptran(m2_ssa, ibands), 'Azimuth angles',
                        'Zenith angles')
    stk_q=m2r_ssa['Q_up (TOA)']
    stk_u=m2r_ssa['U_up (TOA)']

    lp = np.sqrt(stk_q*stk_q + stk_u*stk_u).rename('LP')
    lp.plot.line('.-'+col, label=str(th0))
plt.ylim(0, 0.15)
plt.legend(loc='best')

# %% [markdown]
# #### Example 3: Computation of Sentinel3/OLCI spectrum

# %%
spp=Smartg()

# REPTRAN k distribution file here Sentinel 3 solar channels
SENTINEL_SOLAR = Reptran('reptran_solar_sentinel')
ibands = SENTINEL_SOLAR.to_smartg(include='olci') # select OLCI bands

th0=50.
le = LocalEstimate(
    th=np.array([th0], dtype=np.float32)*np.pi/180,
    phi=np.array([150.], dtype=np.float32)*np.pi/180)
atmosphere=Atm1D('afglms', wavelength_phase=[400., 700., 1000., 1300., 1600,
                                             1900., 2200.])

water=Water1D(grid=[0., -50.],
              comp=[HydrosolPR(1., wavelength_phase=[400., 700.])])

surface=RoughSurface()

# full simulation
m_wsa=reduce_reptran(spp.run(th_deg=th0, wavelength=ibands.l, le=le,
                             atmosphere=atmosphere, surface=surface,
                             water=water, beer=1, progress=True,
                             output_layers=3, n_photons=1e8), ibands)

# just water
m_w = reduce_reptran(
    spp.run(th_deg=th0, wavelength=ibands.l, le=le, atmosphere=None,
            surface=None, water=water, beer=1, progress=True,
            output_layers=3, n_photons=1e8),
    ibands)

# just surface
m_s= reduce_reptran(spp.run(th_deg=th0, wavelength=ibands.l, le=le,
                            atmosphere=None, surface=surface, water=None,
                            beer=1, progress=True, output_layers=3,
                            n_photons=1e8), ibands)

# no atmosphere
m_ws=reduce_reptran(spp.run(th_deg=th0, wavelength=ibands.l, le=le,
                            atmosphere=None, surface=surface, water=water,
                            beer=1, progress=True, output_layers=3,
                            n_photons=1e8), ibands)

# %% [markdown]
# * plots in the perspective of atmospheric corrections

# %%
sel = {'Azimuth angles': 0, 'Zenith angles': 0}
m_ws['I_up (0-)'].isel(sel).plot.line('c^-', label=r'$0^-  \uparrow$, no atm.')
m_ws['I_up (0+)'].isel(sel).plot.line('c.-', label=r'$0^+  \uparrow$, no atm.')

m_w['I_up (0-)'].isel(sel).plot.line('g^-',
                                     label=r'$0^-  \uparrow$, just water')

m_s['I_up (0+)'].isel(sel).plot.line(
    'r.-', label=r'$0^+  \uparrow$, just surf')

m_wsa['I_up (0-)'].isel(sel).plot.line('k^-', label=r'$0^-  \uparrow$')
m_wsa['I_up (0+)'].isel(sel).plot.line('k.-', label=r'$0^+  \uparrow$')
m_wsa['I_up (TOA)'].isel(sel).plot.line('kx-', label=r'$TOA \uparrow$')
plt.ylim(0, 0.2)
plt.legend(loc='best')

# %%
fig=spectrum_view(m_w.isel({'Zenith angles': 0, 'Azimuth angles': 0}),
                  field='up (0-)', color='r', fmt='.-', log_i=True)
_=spectrum_view(m_ws.isel({'Zenith angles': 0, 'Azimuth angles': 0}),
                field='up (0-)', color='b', fmt='.-', log_i=True, fig=fig)
_=spectrum_view(m_wsa.isel({'Zenith angles': 0, 'Azimuth angles': 0}),
                field='up (0-)', color='k', fmt='.-', log_i=True, fig=fig,
                vmax=-1, vmin=-5)

# %% [markdown]
# #### S2-MSI spectral response functions

# %%
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
srf_file = (DIR_AUXDATA / 'validation'
            / 'Sentinel-2A MSI Spectral Responses.xlsx')
SRF = pd.read_excel(srf_file, sheet_name='Spectral Responses')
w_srf=np.array(SRF['SR_WL'])
winf=[]
wsup=[]
wmedian=[]
w_l=np.array([])
srf_l=np.array([])

for b, srf in enumerate(SRF):
    srf_band=np.array(SRF[srf])
    if b!=0 :
        # edge detection for Filter wavelength boundaries
        # Get a function that evaluates the linear spline at any x
        spline = InterpolatedUnivariateSpline(w_srf, srf_band, k=1)
        # Get a function that evaluates the derivative of the linear
        # spline at any x
        dfdw =  spline.derivative()
        dydw =  dfdw(w_srf)

        # detect edges for SRFs with a threshold on first derivative
        ok=np.where(abs(dydw)>0.0001)
        plt.plot(w_srf[ok], srf_band[ok])
        # store boundaries of filter for further reptran, add a 2 nm
        # margin
        w0=w_srf[ok[0][0]]-2
        w1=w_srf[ok[0][-1]]+2
        winf.append(w0)
        wsup.append(w1)
        wmedian.append((w0+w1)/2.)
print(SRF.columns.tolist()[1:])
print(winf, wsup)

# %%
#1) use reptran with solar regular wavelength grid
SOLAR_COARSE = Reptran('reptran_solar_coarse')
ibands = SOLAR_COARSE.to_smartg(lmin=winf, lmax=wsup)

#2) use reprtran parametrization for S2/MSI channels directly
SENTINEL_SOLAR = Reptran('reptran_solar_sentinel')
ibands2 = SENTINEL_SOLAR.to_smartg(include='sentinel2a') # select S2/MSI bands
atmosphere=Atm1D('afglms', wavelength_phase=[400., 800., 1200., 1600., 2200.])
surface=RoughSurface()
th = np.linspace(0., 60.,  num=6, dtype=np.float32)
phi= np.linspace(0., 180., num=9, dtype=np.float32)
le = LocalEstimate(th=th *np.pi/180, phi=phi*np.pi/180)
spp_mult=Smartg()

# %%
# %%time
# Compute first atmospheric profiles at all wavelengths and phase
# functions for re-use
atmbase=atmosphere.calc(ibands.l)

# %%
# %%time
# Run Smart-g for Reptran list of ibands and reduce
m  = reduce_reptran(spp_mult.run(th_deg=60, wavelength=ibands.l,
                                 n_photons=1e6, le=le, surface=surface,
                                 beer=1, atmosphere=atmbase), ibands)

# %%
# %%time
# Run directly smart-g without atmospheric pre computations
# Run Smart-g for Reptran list of ibands2 and reduce
m2  = reduce_reptran(spp_mult.run(th_deg=60, wavelength=ibands2.l,
                                  n_photons=1e6, le=le, surface=surface,
                                  beer=1, atmosphere=atmosphere), ibands2)

# %%
# final multiplication with SRFs
print('--------------------------------------')
print('REPTRAN SPECTRUM OUTPUT')
print('--------------------------------------')
print(m)
wr=m['wavelength'].values # reptran wavelength grid
datasets=[]
for b, srf in enumerate(SRF.columns.tolist()[1:]):
    srf_band=np.array(SRF[srf])
    # interpolate SRFs to reptran wavelengths
    spline = InterpolatedUnivariateSpline(w_srf, srf_band, k=1)
    fr = xr.DataArray(spline(wr), coords={'wavelength': wr},
                      dims=['wavelength'])
    # SRF weighted average over the band of every reduced variable
    with xr.set_options(keep_attrs=True):
        ds = (m * fr).mean('wavelength') / fr.mean('wavelength')
    ds.attrs = dict(m.attrs)
    ds.attrs['median wavelength'] = wmedian[b]
    ds.attrs['band'] = b+1
    datasets.append(ds)
# stack the bands along a new 'median wavelength' dimension
band_dim = xr.DataArray(wmedian, dims='median wavelength',
                        name='median wavelength')
ms2 = xr.concat(datasets, dim=band_dim,
                combine_attrs='drop_conflicts')
ms2 = ms2.assign_coords(band=('median wavelength', np.arange(len(wmedian))+1))
print('')
print('--------------------------------------')
print('S2 CHANNELS OUTPUT')
print('--------------------------------------')
print(ms2)
print('')
print('--------------------------------------')
print('REPTRAN S2 CHANNELS OUTPUT')
print('--------------------------------------')
print(m2)

pos = {'Azimuth angles': 160., 'Zenith angles': 50.}
m['I_up (TOA)'].interp(pos).plot.line('g.',
                                      label=r'$TOA^\uparrow$ : Reptran coarse')
ms2['I_up (TOA)'].interp(pos).plot.line(
    'r^-', label=r'$TOA^\uparrow$ : Reptran coarse + S2 SRFs')
m2['I_up (TOA)'].interp(pos).plot.line('bv:',
                                       label=r'$TOA^\uparrow$ : Reptran S2')
plt.ylim(0, 0.3)
plt.legend()

# %%
iaz = int(np.abs(ms2['Azimuth angles'].values - 180.).argmin())
for i, (b, col) in enumerate(zip(wmedian, ['r', 'b', 'g', 'c']*2)):
    if i==0:
        fig=transect_view(ms2.isel({'median wavelength': i}), ind=[iaz],
                          log_i=True, color='k', fmt='-')
    _=transect_view(ms2.isel({'median wavelength': i}), ind=[iaz], log_i=True,
                    fig=fig, color=col, fmt='-')
    _=transect_view(m2.isel({'wavelength': i}), ind=[iaz], log_i=True, fig=fig,
                    color=col, fmt=':')

# %% [markdown]
# ## BRDF

# %% [markdown]
# ### Ross-Thick Li-Sparse kernel

# %%
##### Spectral RTLS BRDF ######
#Definition of a Ross-Thick Li-Sparse reflector
#K0, K1, K2 are the 3 coefficients of:
#K0 : Spectral Albedo of the isotropic (lambertian) kernel
#K1 : Spectral weight the F1 (geometric) kernel
#K2 : Spectral weight the F2 (volumetric) kernel
#--------------
# in SMART-G the RTLSSurface() objects is initialized with:
#kp = (k0 , k1p, k2p): a tuple of
#k0 : Spectral Albedo of the isotropic (lambertian) kernel
#k1p: Spectral relative weight the F1 (geometric) kernel (=K1/K0)
#k2p: Spectral relative weight the F2 (volumetric) kernel(=K2/K0)
######################
K_VIS = (0.06, 0.05, 0.3) # vegetation in VIS
KP_VIS= (K_VIS[0], K_VIS[1]/K_VIS[0], K_VIS[2]/K_VIS[0])
K_NIR = (0.36, 0.05, 0.3) # vegetation in NIR
KP_NIR= (K_NIR[0], K_NIR[1]/K_NIR[0], K_NIR[2]/K_NIR[0])
wavelength   = np.array([440., 760.])
kp   = (AlbedoSpectrum(np.array([KP_VIS[0], KP_NIR[0]]), wavelength),
        AlbedoSpectrum(np.array([KP_VIS[1], KP_NIR[1]]), wavelength),
        AlbedoSpectrum(np.array([KP_VIS[2], KP_NIR[2]]), wavelength))
surface    = RTLSSurface(kp=kp)
atmosphere = Atm1D('afglt')
# atmosphere + surface
azimuth_transect = (10., 90)
n_dir  = 24
le = LocalEstimate(th_deg=np.linspace(0, 80., num=n_dir),
                   phi_deg=np.linspace(360., 0., num=n_dir),
                   zip=False)
m  = Smartg().run(wavelength, n_photons=1e6, th_deg=25., le=le,
                  atmosphere=atmosphere, surface=surface)
ind_az = [int(np.abs(m['Azimuth angles'].values - a).argmin())
          for a in azimuth_transect]
fig= smartg_view(m, ind=ind_az, qu=False,
                 interp_dict={'wavelength': m['wavelength'].values[0]})
fig= smartg_view(m, ind=ind_az, qu=False,
                 interp_dict={'wavelength': m['wavelength'].values[1]})

# %% [markdown]
# ## Irradiances

# %%
spp        = Smartg()
surface    = RoughSurface(sur=3, wind=10., nh2o=1.34)
atmosphere = Atm1D('afglms')
water      = Water1D(grid=[0., -100.],
                     comp=[HydrosolPR(chl=1.1, wavelength_phase=[550.])])
wavelength = np.linspace(400., 700., num=11)
th0        = 75.

# %%
# 1) planar and spherical fluxes using the "flux" keyword
# Flux are computed FAST in Cone sampling mode
n_photons=1e7
m_planar = spp.run(th_deg=th0, wavelength=wavelength, n_photons=n_photons,
                   atmosphere=atmosphere, surface=surface, water=water,
                   flux='planar', output_layers=3)
m_spherical = spp.run(th_deg=th0, wavelength=wavelength, n_photons=n_photons,
                      atmosphere=atmosphere, surface=surface, water=water,
                      flux='spherical', output_layers=3)

# 2) fluxes could be recomputed from radiances, with care about the
# directional sampling and statistics
# in general, longer and less accurate
m_rad  = spp.run(th_deg=th0, wavelength=wavelength, n_photons=n_photons,
                 atmosphere=atmosphere, n_theta=360, n_phi=360,
                 surface=surface, water=water, output_layers=3)
# directional integration for irradiances
m_irr = irradiance_ds(m_rad)

# %%
# plot downward fluxes spectra just underwater
plt.plot(m_planar['wavelength'], m_planar['flux_down (0-)'], 'r.-',
         label='down 0- planar')
plt.plot(m_spherical['wavelength'], m_spherical['flux_down (0-)'], 'b.-',
         label='down 0- spherical')
plt.ylim(0, 1.5)
m_irr['Pflux_down (0-)'].plot.line(color='r', marker='x', linestyle='-',
                                   label='down 0- planar from radiance')
m_irr['Sflux_down (0-)'].plot.line(color='b', marker='x', linestyle='-',
                                   label='down 0- spherical from radiance')
plt.legend(loc='best')
# plot upward fluxes spectra just underwater
plt.figure()
plt.plot(m_planar['wavelength'], m_planar['flux_up (0-)'], 'r.-',
         label='up 0- planar')
plt.plot(m_spherical['wavelength'], m_spherical['flux_up (0-)'], 'b.-',
         label='up 0- spherical')
plt.ylim(0, 0.05)
m_irr['Pflux_up (0-)'].plot.line(color='r', marker='x', linestyle='-',
                                 label='up 0- planar from radiance')
m_irr['Sflux_up (0-)'].plot.line(color='b', marker='x', linestyle='-',
                                 label='up 0- spherical from radiance')
plt.legend(loc='best')

# 3) For the particular "Down (0+)" Irradiance, it contains only Diffuse
# component, the direct component
# is obtained trough the 'direct transmission' field
t_direct = m_planar['direct transmission']
plt.figure()
plt.plot(m_planar['wavelength'], m_planar['flux_down (0+)'], 'r.--',
         label='down 0+ diffuse planar')
plt.plot(m_planar['wavelength'], m_planar['flux_down (0+)']+t_direct, 'r.-',
         label='down 0+ total planar')
plt.plot(m_spherical['wavelength'], m_spherical['flux_down (0+)'], 'b.--',
         label='down 0+ diffuse spherical')
plt.plot(m_spherical['wavelength'], m_spherical['flux_down (0+)']+t_direct,
         'b.-', label='down 0+ total spherical')
plt.plot(m_planar['wavelength'], m_planar['flux_down (0-)'], 'rx-',
         label='down 0- planar')
plt.plot(m_spherical['wavelength'], m_spherical['flux_down (0-)'], 'bx-',
         label='down 0- spherical')
plt.ylim(0, 1.5)
plt.legend(loc='best')

# %% [markdown]
# ## Thermal source (dev)

# %%
#REPTRAN k distribution file here MSG/SEVIRI1 thermal channels
from smartg.reptran import reptran_avg_emission
SEVIRI_THERMAL = Reptran('reptran_thermal_msg')

# %%
# Compile with thermal option
# Only in forward mode for the moment, no ground
s_thermal = Smartg(thermal=True, alt_pp=True, back=False)
for platform in ['1', '2', '3', '4']:
    ibands         = SEVIRI_THERMAL.to_smartg(include='msg'+platform)
    atmosphere     = Atm1D('afglt', grid=np.linspace(50, 0, num=51))
    prof_atm       = atmosphere.calc(ibands.l)
    # DIRECT option activated i.e direct transmitted radiance form
    # source to receiver counted
    # (as opposite to solar computation where the direct light is not
    # counted)
    # cell_proba has to be set to 'auto' : Distribution of thermal
    # source within the atmosphere set automatically
    fl = s_thermal.run(direct=True, wavelength=ibands.l, flux='planar',
                       n_photons=1e8, atmosphere=prof_atm, cell_proba='auto')
    # The Column integrated thermal Emission is computed using the
    # function Avg_Emission
    # is it used in the reduce final computation
    fl_int = reduce_reptran(
        fl, ibands, integrated=True,
        extern_weights=reptran_avg_emission(prof_atm, ibands))
    plt.errorbar(fl_int['wavelength'].values,
              fl_int['flux_up (TOA)'].values,
              label=platform, marker='^')
plt.ylim(0, 2)
plt.ylabel('TOA irradiance (W.m-2)')
plt.legend()

# %% [markdown]
# ## Looping over parameters

# %%
# loop over Aerosol Optical Thickness
ds_aot = []
aots = np.linspace(0, 1.5, 5)
spp=Smartg()
for aot in aots:
    m_aot = spp.run(th_deg=30.,
               wavelength=443., n_photons=1e7,
               atmosphere=Atm1D('afglt', comp=[AerOPAC('desert', aot, 443.)]),
               surface=RoughSurface(wind=5.))
    ds_aot.append(m_aot)

# %%
# Concatenate all previous looped Datasets along a new 'AOT' dimension
m_aot = xr.concat(ds_aot, dim=xr.DataArray(aots, dims='AOT', name='AOT'))
print(m_aot)

# %%
m_aot['I_up (TOA)'].interp(
    {'Azimuth angles': 90., 'Zenith angles': 45.}).plot(marker='o')

# %% [markdown]
# ## Difference between two simulations

# %%
# %%time
# Here we compare the TOA radiance simulated with
# plane parallel and spherical atmospheres, at 400 nm.
atmosphere=Atm1D('afglt', tco3=0., no2=False)
ths = np.concatenate((
            np.linspace(0. , 75., num=12, dtype=np.float32),
            np.linspace(76., 89., num=12, dtype=np.float32)))
le = LocalEstimate(
    th_deg=ths,
    phi_deg=np.array([0., 180.], dtype=np.float32))

spp=Smartg(pp=True,   double=True, back=True, alt_pp=True)
s_sp=Smartg(pp=False,  double=True, back=True)

# show relative difference
plt.figure(figsize=(6, 6))
results = []
for sg in [spp, s_sp]:
    m_res = sg.run(earth_radius=6370., th_deg=30., wavelength=400.,
                   n_photons=5e7, le=le, atmosphere=atmosphere, stdev=True)
    results.append(m_res)

pp = results[0]['I_up (TOA)'].interp({'Azimuth angles': 0.})
sp = results[1]['I_up (TOA)'].interp({'Azimuth angles': 0.})
pp_std = results[0]['I_stdev_up (TOA)'].interp({'Azimuth angles': 0.})
sp_std = results[1]['I_stdev_up (TOA)'].interp({'Azimuth angles': 0.})
diff = 100.*(pp-sp)/sp
plt.plot(diff['Zenith angles'], diff, '.:')
plt.ylim(-3., 3.)
plt.title('Plane Parallel - Spherical (%)')
unc = np.sqrt((sp_std*sp_std/sp/sp) + (pp_std*pp_std/pp/pp))*100
plt.errorbar(diff['Zenith angles'], diff.values, yerr=unc.values, fmt='none')

# %% [markdown]
# ## Interactive simulation

# %%
from ipywidgets import interact_manual

sg=Smartg()
le = LocalEstimate(phi=np.array([0.]),
                   th=np.array([60.])*np.pi/180)

def simulate(thvdeg, surface, aerosol_model, aot550):
    surface = {True: RoughSurface(), False: None}[surface]
    aer = AerOPAC(aerosol_model, aot550, 550.)
    out = sg.run(th_deg=thvdeg, wavelength=[443.], n_photons=1e5,
                 le=le, atmosphere=Atm1D('afglt', comp=[aer]),
                 surface=surface, progress=False)
    print('TOA Intensity : %.5f' % out['I_up (TOA)'].data)

interact_manual(simulate, thvdeg=(0, 90), surface=True,
                aerosol_model=AerOPAC('desert', 0.1, 550.).list(),
                aot550=(0.0, 5., 0.05))

# %% [markdown]
# # Backward mode and Sensor class

# %% [markdown]
# ## Limb geometry

# %%
# %%time
# Backward mode is needed when the scene is not invariant by horizontal
# translation anymore
# e.g. spherical shell atmosphere or horizontally varying surface
# reflectance
# In that case, the keyword back=True should be used for compilation
# (the bias sampling scheme for scattering (default) is mandatory,
# bias=True)
# Here the example deals with spherical shell atmosphere (pp=False)
sg = Smartg(back=True, pp=False, double=True)
# in backward mode, photons are injected FROM the sensor, and the
# outputs correspond to solar geometries
# So we define where the sensor is located with the Sensor class
# in this example, the sensor is placed at the TOA and is looking at the
# limb
h_toa = 120.
r_ter = 6371. # Earth's radius
wavelength   = [430., 660., 840.]
#
#
grid    = np.linspace(h_toa, 0., num=51)
atm1  = Atm1D('afglsw', grid=grid)

# lambertian surface
alb  = AlbedoCst(0.1)
surface = LambSurface(alb=alb)
def zt2thv(zt, r_ter=6371., h_toa=120.):
    return np.arcsin((r_ter+zt)/(r_ter+h_toa))*180/np.pi
# For a vertical profile in backward mode, several sensors are needed
# we make a loop on trigonometric tangent heights and also viewing
# azimuths
# and make a LIST of sensors

from itertools import product
sensors= []
n_zt = 16 # number of tangent heights
n_phi = 4 # relative azimuths to the sun
zts   = np.linspace(2., 50,  num=n_zt)
dphis = np.linspace(0., 91, num=n_phi)
for zt, dphi in product(zts, dphis):
    sensors.append(
                # sensor object creation
                # Sensor coordinates (in km) (default:origin(0.,0.,0.))
                Sensor(pos_x=0.,
                pos_y=0.,
                # in spherical mode set Z to the distance from Earth
                # center,
                pos_z=h_toa+r_ter,
                                    # in pp, Z is the altitude
                th_deg=180.-zt2thv(zt, h_toa=h_toa),
                                    # Sensor 'Emitting' zenith angle,
                                    # from 0: Zenith,
                                           # to 180.: Nadir (default:0.)
                # Sensor 'Emitting' azimuth angle (default:0.)
                ph_deg=dphi,
                # location of sensor (default: (SURF0P, just above
                # surface)
                loc='ATMOS',
                fov=0.,             # Sensor FOV (default 0.)
                # Sensor type :Radiance (0), Planar flux (1),
                sensor_type=0
                                           # Spherical Flux (2),
                                           # (default 0)
                    )
               )
n_photons = 2e5 * n_zt * n_phi # so 2e5 photons per sensor

ths    = np.array([60., 80., 91., 94.]) # different SZA
phis   = np.array([0.])
# Local estimate directions correspond to solar positions
le = LocalEstimate(th_deg=ths, phi_deg=phis)

# the sensor keyword accepts a list of sensor obsjects
m1  = sg.run(wavelength=wavelength, surface=surface, le=le, atmosphere=atm1,
             n_photons=n_photons, earth_radius=r_ter, n_icdf=1e5,
             refraction=True, sensor=sensors, reflectance=False)
# look at the result: a new dimension  'sensor index' is present in the
# output

# %%
# You can get back a 2D output of intensities by reshaping to (n_zt,
# n_phi) what is inside the 'sensor index'
# dimension
col = ['r', 'g', 'b']
lin = ['-', ':', '--', '-.']
fig, ax =plt.subplots(1, 4)
fig.set_size_inches(12, 4)

for i, (ts, ls) in enumerate(zip(ths, lin)):
    for  j, dp in enumerate(dphis):
        for k, (w, c) in enumerate(zip(wavelength[::-1], col)):
            i_2d = m1['I_up (TOA)'].isel({'Azimuth angles': 0}).interp(
                     wavelength=w, kwargs={'fill_value': 'extrapolate'}
                 ).interp({'Zenith angles': ts}).values.reshape(n_zt, n_phi)
            if( i==0 and j==0):
                ax[j].semilogx(i_2d[:, j], zts, ls, color=c,
                                                label='%.0f nm'%(w))
            if( k==0 and j==1):
                ax[j].semilogx(i_2d[:, j], zts, ls, color=c,
                                                label='%.0f°'%(ts))
            else:
                ax[j].semilogx(i_2d[:, j], zts, ls, color=c)
        ax[j].grid()
        ax[j].set_xlim([0.0002, 2])
        ax[j].set_xlabel(r'$\pi I$')
        ax[j].set_title(r'$\Delta\Phi:%.0f$°'%dp)
    ax[0].set_ylabel(r'$z_t (km)$')
    ax[0].legend(title=r'$\lambda$')
    ax[1].legend(title=r'$SZA$')

# %% [markdown]
# ## Horizontal inhomogeneity

# %%
# Goal: to simulate an observation from a satellite sensor
# for 1 thv and several azimuths (Almucantar)
# with a straight coastline (limit ocean land at x=0, ocean for x <0)
# the sensor is looking to a water pixel located at a varying distance
# from the coastline
# we simulate a coastline as being the zone near a big circle, whose
# radius is 1e6 km
# centred on a point located far from the sensor (-1e6 km from the
# origin )
# The interior of the circle is the ocean
# Principal plane reflectance and DoLP for 380, 500 and 800 nm
wavelength      = [ 380., 500., 800.]
col     = ['b', 'g', 'r']

# Solar geometries in backward mode
nsaa    = 4
nsza    = 24
SZA_MAX = 70.
saas    = np.linspace(0., 270.,    num=nsaa, dtype=np.float32, endpoint=True)
szas    = np.linspace(0., SZA_MAX, num=nsza, dtype=np.float32, endpoint=True)
le      = LocalEstimate(th_deg=szas, phi_deg=saas)

aer     = AerOPAC('maritime_polluted', 0.3, 500.)
atmosphere  = Atm1D('afglms', comp=[aer])
# Cox & Munk BRDF, time symetrical
surface     = RoughSurface(wind=5., brdf=True)

vza     = 30.
env_radius       = 1.0e6 #(km)
# The Environement object creates a disk of ocean surface with radius
# ENV_SIZE
# centred on X0,Y0 , surrounded by lambertian reflector of albedo alb
environment = Environment(env=1,
                  # radius of the circle with ocean surface condition
                  env_size=env_radius ,
                  # X coordinate of the center of the circle
                  x0=-env_radius,
                  y0=0,
                  # Lambertian grey albedo of the land zone (snow)
                  alb=AlbedoCst(0.5)
                 )
s_back      = Smartg(back=True,  double=True) ## Plane Parallel

# %%
# %%time
h_toa  = 120.
# 4 View Azimuth Angles; vaa=0 the sensor is 'above water'
vaas  = [0., 90., 180., 270.]
dists = [0.250, 5., env_radius]   # 3 distances to the coast (km)

sensors= []
cases  = list(product(vaas, dists))
for vaa, dist in cases:
    # sensor is placed at the TOA and is looking down
    # to the point (-dist, 0., 0.) from several relative azimuths
    # to the coastline (vaa);
    delta_h = h_toa   * np.tan(np.radians(vza))
    delta_x = delta_h * np.cos(np.radians(180-vaa))
    delta_y = delta_h * np.sin(np.radians(180-vaa))
    sensors.append(
            Sensor(
            # Sensor coordinates (in km) (default:origin(0.,0.,0.))
            pos_x=-dist + delta_x,
            pos_y=delta_y,
            pos_z=h_toa,
            # Sensor 'Emitting' zenith angle, from 0: Zenith to 180.:
            # Nadir (default:0.)
            th_deg=180-vza,
            ph_deg=vaa,     # Sensor 'Emitting' azimuth angle (default:0.)
            # location of sensor (default: (SURF0P, just above surface)
            loc='ATMOS',
            fov=0.,         # Sensor FOV (default 0.)
            # Sensor type :Radiance (0), Planar flux (1), Spherical Flux
            # (2), (default 0)
            sensor_type=0
            )
    )
n_photons = 1e6 * len(vaas) * len(dists)
m  = s_back.run(wavelength=wavelength, atmosphere=atmosphere, surface=surface,
                environment=environment, n_photons=n_photons, le=le,
                sensor=sensors, progress=True)

# %%
# the transects are plotted for raa=0, (principal plane)
raa = 0.

def azimuth_index(saa):
    # azimuth plane index nearest to the wanted azimuth angle
    return [int(np.abs(m['Azimuth angles'].values - saa).argmin())]

for i, w in enumerate(wavelength):
    # for vaa=0, the sensor is 'above water'
    vaa = vaas[0]
    saa = vaa - raa
    ind = cases.index((vaa, dists[0])) # retrieving sensor index from the cases
    fig=transect_view(m.isel({'sensor index': ind}),
                      interp_dict={'wavelength': w}, color='m', fmt=':',
                      ind=azimuth_index(saa))
    ind = cases.index((vaa, dists[1]))
    _=transect_view(m.isel({'sensor index': ind}),
                    interp_dict={'wavelength': w}, color='m', fmt='--',
                    ind=azimuth_index(saa), fig=fig)

    # for vaa=90, the sensor is 'above the coastline'
    vaa = vaas[1]
    saa = vaa - raa
    ind = cases.index((vaa, dists[0]))
    _=transect_view(m.isel({'sensor index': ind}),
                    interp_dict={'wavelength': w}, color='C1', fmt=':',
                    ind=azimuth_index(saa), fig=fig)
    ind = cases.index((vaa, dists[1]))
    _=transect_view(m.isel({'sensor index': ind}),
                    interp_dict={'wavelength': w}, color='C1', fmt='--',
                    ind=azimuth_index(saa), fig=fig)

    # for vaa=180, the sensor is 'above land'
    vaa = vaas[2]
    saa = vaa - raa
    ind = cases.index((vaa, dists[0]))
    _=transect_view(m.isel({'sensor index': ind}),
                    interp_dict={'wavelength': w}, color='g', fmt=':',
                    ind=azimuth_index(saa), fig=fig)
    ind = cases.index((vaa, dists[1]))
    _=transect_view(m.isel({'sensor index': ind}),
                    interp_dict={'wavelength': w}, color='g', fmt='--',
                    ind=azimuth_index(saa), fig=fig)

    # whatever vaa, the last distance is 1e6 km similar to the infinite
    # homogeneous ocean case
    ind = cases.index((vaa, dists[-1]))
    _=transect_view(m.isel({'sensor index': ind}),
                    interp_dict={'wavelength': w}, color='k', fmt='-',
                    ind=azimuth_index(saa), fig=fig, vmin=0, vmax=.4)
    plt.text(-50., 95, r'$\lambda$=%.0f nm'%w)
    if i==2:
        plt.plot([-50, -25], [90, 90], 'k-')
        plt.text(-23, 90, 'homogeneous ocean')
        plt.plot([-50, -25], [85, 85], 'k:')
        plt.text(-23, 85, 'd=250 m')
        plt.plot([-50, -25], [80, 80], 'k--')
        plt.text(-23, 80, 'd=5000 m')
        plt.plot([-50, -25], [75, 75], 'm-')
        plt.text(-23, 75, r'VAA=0$\degree$  (above water)', color='m')
        plt.plot([-50, -25], [70, 70], 'C1-')
        plt.text(-23, 70, r'VAA=90$\degree$ (along the coast)', color='C1')
        plt.plot([-50, -25], [65, 65], 'g-')
        plt.text(-23, 65, r'VAA=180$\degree$ (above land)', color='g')

# %% [markdown]
# ## Albedo map

# %%
# A 2D horizontal map of spectral albedos can be constructed
# Spectral albedos are limited to a MAX_NREF=10 different kind, could be
# extended
# They should be defined using AlbedoCst, AlbedoSpectrum or
# AlbedoSpeclib classes
speclib = DIR_AUXDATA / 'validation'
snow_file = 'jhu.becknic.water.snow.granular.82um.medium.spectrum.txt'
soil_file = 'jhu.becknic.soil.alfisol.paleustalf.coarse.87P473.spectrum.txt'
alb_list = [AlbedoSpeclib(speclib / snow_file),
            AlbedoSpeclib(speclib / soil_file)]
# The horizontal grid for the 2D map of albedo (env=5) is rectangular,
# centred on x0,y0 keywords of the Environement object
# and x and y boundaries are encoded in monotonic np.arrays whose values
# are the upper limit of the rectangles:
# if x = [x0, x1, x2, ..., xn], the the limits are [-Inf, x0], [x0, x1],
# ..., [xn-1, xn], with xn should be big to be considered as +Inf
# Example :  a land square of 20 km edge with two albedos (soil and
# snow) and large ocean zone from -Inf to -10km and +10km to +Inf
x_bins = np.array([-10, 0, 10, 1e8])
y_bins = np.array([-10, 10, 1e8])
# Then we assigned each rectangle an index in alb_list, in a 2D array of
# shape (len(x),len(y))
# Example :  the map points to the ocean surface with seafloor soil
# albedo (surface : negative indices) for all rectangles
# except the two central ones with land soil and snow albedos
# (environment : positive indices)
# Negative indices are for the surface properties instead of the 2D env
# albedos
surface = RoughSurface(wind=5., wave_shadow=True)
water= Water1D(grid=[0., -10.],
               comp=[HydrosolPR(chl=0.1, wavelength_phase=[600.])])
ai  = np.array([[-1, -1, -1], [-1, 0, -1], [-1, 1, -1], [-1, -1, -1]])
#
# we build the Albedo 2D object
alb = AlbedoMap(ai, x_bins, y_bins, alb_list)
# Finally define the Environment object with this 2D albedo
environment = Environment(env=5, x0=0, y0=0, alb=alb)

# %%
# Let us simulate an image seen from Top and looking at nadir (in
# backward mode)
n_sensors  = 8
x0 = np.linspace(-10.1, 10.1, num=n_sensors, endpoint=True)
y0 = np.linspace(-10.1, 10.1, num=n_sensors, endpoint=True)
# building sensors list
sensors=[]
for a in x0:
        for b in y0:
                 sensors.append(Sensor(pos_z=120., pos_x=a, pos_y=b,
                                       loc='ATMOS', th_deg=180.))

# defining Sun's output direction
le = LocalEstimate(th_deg=np.array([60.]),
                   phi_deg=np.array([0.]), zip=True)
atmosphere = Atm1D('afglt', comp=[AerOPAC('continental_clean', 0.3, 550.)],
                   wavelength_phase=[600.])
wavelength   = np.linspace(400., 900., num=9)
#
#RUN
res=Smartg().run(wavelength=wavelength, le=le, sensor=sensors,
                 atmosphere=atmosphere, surface=surface,
                 environment=environment, water=water, n_photons=1e8, beer=1,
                 russian_roulette=0)
res = drop_axes(res, 'Zenith angles')

# %%
fig  = plt.figure(figsize=(8, 8))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 3),
                 axes_pad=0.3,
                 label_mode="L",
                 )
# images : loop on wavelength
for (ax, w, im) in zip(grid, wavelength, res['I_up (TOA)'].data.T):
    img = ax.imshow(im.reshape(n_sensors, n_sensors).T, origin='lower', vmin=0,
                    vmax=1, cmap=plt.cm.jet)
    ax.set_title('%.0f nm'%w)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
#fig.colorbar()

# %% [markdown]
# # Counters use
#             SMIN : Minimum Interaction (scattering/reflection) order: Default 0
#             SMAX : Maximum Interaction (scattering/reflection) order: Default 1e6          
#             RMIN : Minimum Reflection (by surface only, not environement) order: Default 0
#             RMAX : Maximum Reflection (by surface only, not environement) order: Default 1e6
#             DIRECT : Include directly transmitted photons: Default False

# %%
#
atmosphere = Atm1D('afglus', tco3=300.)
wavelength   = 590.
sza  = 30.
# downward surface planar diffuse irradiance (Diffuse transmission if
# black surface)
t_dif  = Smartg().run(wavelength=wavelength, atmosphere=atmosphere,
                      surface=None, flux='planar', n_photons=1e8, th_deg=sza,
                      output_layers=1)['flux_down (0+)'].data
# downward surface planar total irradiance (Total transmission if balck
# surface), we set the DIRECT
# keyword to True for including direct transmission
t_tot = Smartg().run(
    wavelength=wavelength, atmosphere=atmosphere, surface=None,
    flux='planar', n_photons=1e8, direct=True, th_deg=sza,
    output_layers=1)['flux_down (0+)'].data
##
## We add surface
##
surface = LambSurface(alb=AlbedoCst(1.0))
# downward surface planar diffuse irradiance (Diffuse transmission if
# only photons having not been reflected are considered)
t_dif_0 = Smartg().run(
    wavelength=wavelength, atmosphere=atmosphere, surface=surface,
    environment=None, flux='planar', n_photons=1e8, r_max=0,
    th_deg=sza, output_layers=1)['flux_down (0+)'].data
# Multiple scattering contribution to the Diffuse transmission
t_dif_0_ms  = Smartg().run(wavelength=wavelength, atmosphere=atmosphere,
                           surface=surface, environment=None, flux='planar',
                           n_photons=1e8, r_max=0, s_min=2, th_deg=sza,
                           output_layers=1)['flux_down (0+)'].data
# Single scattering contribution to the Diffuse transmission
t_dif_0_ss  = Smartg().run(wavelength=wavelength, atmosphere=atmosphere,
                           surface=surface, environment=None, flux='planar',
                           n_photons=1e8, r_max=0, s_max=1, th_deg=sza,
                           output_layers=1)['flux_down (0+)'].data
print(f"T (Total transmission) = {t_tot:.5f}")
print(f"Tdif (Diffuse transmission, black surface) = {t_dif:.5f}")
print(f"Tdif_0 (Diffuse transmission, no surface reflection) = {t_dif_0:.5f}")
print(f"Tdif_0_MS (Multiple scattering contribution) = {t_dif_0_ms:.5f}")
print(f"Tdif_0_SS (Single scattering contribution) = {t_dif_0_ss:.5f}")
