# %% [markdown]
# # Smart-G — Photon Histories & JAX post-processing
#
# This notebook demonstrates:
# - outputting per-photon histories from the ALIS correlated-spectral method,
# - post-hoc gas absorption via Beer-Lambert law applied to per-photon path lengths,
# - Jacobians (∂ρ/∂kabs, ∂ρ/∂T, ∂ρ/∂P) computed with `jax.grad` at zero MC cost.
#
# > **Environment**: requires the `extra` environment (includes `jax[cuda12]` and `radis`).
# > Run: `pixi run -e extra jupyter notebook`

# %% [markdown]
# ## Symbols used in this notebook
#
# | symbol | meaning |
# |---|---|
# | `rho_i`, `rho_q`, `rho_u` | Stokes reflectances at high spectral resolution |
# | `dolp` | degree of linear polarization, `sqrt(rho_q² + rho_u²) / rho_i` |
# | `sza`, `vza` | solar and viewing zenith angle, in degrees |
# | `kabs` | gas absorption coefficient, per km |
# | `amf` | air mass factor |
# | `n_h`, `s_h`, `d_h`, `w_h` | per-photon history records: count, scattering, path lengths, weights |
# | `dij`, `ki` | the ALIS path-length and weight arrays, as in `smartg.histories` |
# | `di_dkabs`, `drho_dp` | Jacobians, read as ∂numerator/∂denominator |
# | `_j` suffix | the value at one high-resolution wavelength |
#
# The Stokes reflectances follow the `rho_*` notation of the markdown
# above; the package itself names the four components `stk_i` to
# `stk_v`.

# %%
# %matplotlib inline
# next 2 lines allow to automatically reload modules that have been
# changed externally
# %reload_ext autoreload
# %autoreload 2
import os
import sys
from pathlib import Path

# ── Make sure nvcc is on PATH (required by pycuda/smartg kernel
# compilation) ──
_cuda_bin = Path('/usr/local/cuda/bin')
if _cuda_bin.exists() and str(_cuda_bin) not in os.environ.get('PATH', ''):
    os.environ['PATH'] = str(_cuda_bin) + ':' + os.environ.get('PATH', '')

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

import contextlib
import io
import logging
import warnings

import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from jax import jit, vmap

from smartg.albedo import AlbedoCst
from smartg.atmosphere import AerOPAC, Atm1D, od2k
from smartg.config import DIR_AUXDATA
from smartg.diff import diff1
from smartg.histories import get_histories, si
from smartg.smartg import Alis, LocalEstimate, Smartg
from smartg.surface import LambSurface
from smartg.xarray import drop_axes

warnings.filterwarnings("ignore")
warnings.simplefilter('always', DeprecationWarning)

# %% [markdown]
# ## Absorption spectra
#
# - Here the extreral RADIS module is used
# - It results in an vertical profile of gaseous absorption coefficient or or gaeous optical thickness at high spectral resolution

# %%
# ── RADIS/pandas HDF5 compatibility patches
# ───────────────────────────────
#
# Two problems with this version of RADIS + pandas:
#
#  (A) READ: corrupted/fixed-format cache files cause a TypeError inside
# DataFileManager.read_metadata (via pandas HDFStore._create_storer).
# RADIS already handles AttributeError("Attribute 'metadata' does not
# exist") by raising DeprecatedFileWarning → auto-deleting the bad file
# and re-downloading.  We just convert the TypeError to that
# AttributeError.
#
# (B) WRITE: for some molecules (CO2, N2O, …) the `branch` column is
# parsed
# as integers stored in an object array; pandas' table-format HDF5
# writer
# rejects non-string object columns.  We coerce branch to str before
# write.
#
from radis.api.hdf5 import DataFileManager as _DFM

# ── Patch A: read_metadata
# ────────────────────────────────────────────────
_orig_read_meta = _DFM.read_metadata

def _patched_read_meta(self, fname, key='df'):
    try:
        return _orig_read_meta(self, fname, key)
    except TypeError:
        # Converts pandas _TABLE_MAP KeyError (corrupt/fixed-format
        # file)
        # into the AttributeError that RADIS uses to trigger
        # auto-removal.
        raise AttributeError("Attribute 'metadata' does not exist")

_DFM.read_metadata = _patched_read_meta

# ── Patch B: write
# ────────────────────────────────────────────────────────
_orig_write = _DFM.write

def _patched_write(self, file, df, append=False, **kw):
    import pandas as _pd
    if isinstance(df, _pd.DataFrame) and 'branch' in df.columns:
        if df['branch'].dtype == object:
            import pandas.api.types as _pat
            sample = df['branch'].dropna()
            if len(sample) and not _pat.is_string_dtype(sample):
                df = df.copy()
                df['branch'] = df['branch'].astype(str)
    return _orig_write(self, file, df, append=append, **kw)

_DFM.write = _patched_write

print("RADIS patches applied (read_metadata + write).")

SEED = 1234

# %%
import radis
from radis import SpectrumFactory

# ──────────────── USER PARAMETERS
# ─────────────────────────────────────────
# absorbing molecule: 'O2', 'H2O', 'CO2', 'CH4', 'N2O', 'CO', …
MOLECULE  = 'O2'
WL_MIN_NM = 765.    # nm  — start of high-resolution spectral grid
WL_MAX_NM = 768.     # nm  — end
N_LOW_R    = 3        # number of low-resolution ALIS wavelength points

# Default VMR (volume mixing ratio) per molecule — override here if
# needed
_VMR_DEFAULTS = {
    'O2': 0.2095,
    'H2O': 0.025,
    'CO2': 420e-6,
    'CH4': 1.9e-6,
    'N2O': 320e-9,
    'CO': 0.1e-6,
}
vmr_scalar = _VMR_DEFAULTS.get(MOLECULE, 1e-6)
# ──────────────────────────────────────────────────────────────────────

_ISOTOPES = {
    'O2': '1',       'H2O': '1,2,3',  'CO2': '1,2,3,4',
    'CH4': '1,2',     'N2O': '1,2,3,4,5',  'CO': '1,2',
}
ISOTOPE = _ISOTOPES.get(MOLECULE, '1')

# ── vertical grid: interface altitudes (NZ=9)
# ─────────────────────────────
_afglt_grid = np.array([100., 75., 50., 30., 20., 10., 5., 2., 1., 0.])  # km
_prof       = Atm1D('afglt', grid=_afglt_grid.tolist()).prof
z_iface     = _afglt_grid
z_mid       = 0.5 * (z_iface[:-1] + z_iface[1:])   # (NL=8,) layer midpoints
t_mid       = np.interp(z_mid, _prof.z[::-1], _prof.t[::-1])
p_mid       = np.interp(z_mid, _prof.z[::-1], _prof.p[::-1])
n_mid       = np.interp(z_mid, _prof.z[::-1], _prof.dens_air[::-1])
thick       = np.abs(np.diff(z_iface))  # (NL=8,) layer thicknesses in km
n_layers   = len(z_mid)

# ── VMR vertical profile (NL,)
# ────────────────────────────────────────────
# O2, CO2, CH4, … are well-mixed; H2O scales approximately with
# pressure.
_vmr_profiles = {
    'O2': vmr_scalar * np.ones(n_layers),
    'H2O': vmr_scalar * (p_mid / p_mid[-1]),     # surface-pressure scaling
}
vmr_profile = _vmr_profiles.get(MOLECULE, vmr_scalar * np.ones(n_layers))

# ── Wavenumber range (from user wavelength bounds)
# ─────────────────────────
wn_min = float(1e7 / WL_MAX_NM) - 1.
wn_max = float(1e7 / WL_MIN_NM) + 1.

# ── O₂-A band HR grid & ALIS low-res grid
# ─────────────────────────────────
# placeholder; updated from RADIS output
wavelength_hr   = np.array([WL_MIN_NM, WL_MAX_NM])
wavelength_lr_r = np.linspace(WL_MIN_NM, WL_MAX_NM, N_LOW_R)

# ── SpectrumFactory + HITRAN download (once, then cached)
# ─────────────────
_sink = io.StringIO()
radis.config['SPARSE_WAVERANGE'] = False
with (warnings.catch_warnings(),
      contextlib.redirect_stdout(_sink),
      contextlib.redirect_stderr(_sink)):
    warnings.simplefilter('ignore')
    logging.disable(logging.CRITICAL)
    # auto-selects grid step fine enough for narrowest line
    sfac = SpectrumFactory(wavenum_min=wn_min, wavenum_max=wn_max,
                           molecule=MOLECULE, isotope=ISOTOPE,
                           mole_fraction=1.0, wstep='auto', verbose=False,
                           warnings={})
    # downloads once; uses ~/.radis/ cache afterwards
    sfac.fetch_databank('hitran')
    logging.disable(logging.NOTSET)
print(f"SpectrumFactory ready — {MOLECULE} (isotope {ISOTOPE}, "
      f"VMR_scalar={vmr_scalar:.4g})")

# ── Absorption coefficient per layer
# ──────────────────────────────────────
# wstep='auto' may choose a slightly different grid size per T/P
# condition,
# so we establish a reference wavelength grid from the first layer and
# interpolate all subsequent layers onto it.
sigma_layers = []
_wl_ref = None
with (warnings.catch_warnings(),
      contextlib.redirect_stdout(_sink),
      contextlib.redirect_stderr(_sink)):
    warnings.simplefilter('ignore')
    logging.disable(logging.CRITICAL)
    for k in range(n_layers):
        vmr_k = float(vmr_profile[k])
        s = sfac.eq_spectrum(
            Tgas=float(t_mid[k]),
            pressure=float(p_mid[k]) * 1e-3,   # hPa → bar
            mole_fraction=vmr_k,
        )
        wn_out, sigma_k = s.get('xsection')
        wavelength_out_k = 1e7 / wn_out[::-1]          # ascending wavelength
        sigma_k  = sigma_k[::-1]                # align with wavelength_out_k
        if _wl_ref is None:
            # fix reference grid from layer 0
            _wl_ref = wavelength_out_k
        else:
            # resample onto ref
            sigma_k = np.interp(_wl_ref, wavelength_out_k, sigma_k)
        n_k = vmr_k * n_mid[k]                  # [molec/cm³]
        sigma_layers.append(sigma_k * n_k * 1e5)  # cm⁻¹ → km⁻¹
    logging.disable(logging.NOTSET)

# (NWL_HR,)
wavelength_hr   = _wl_ref
# (NWL_HR, NL=8)
kabs_hr = jnp.array(np.stack(sigma_layers, axis=1), dtype=jnp.float32)
# update to match wavelength_hr
wavelength_lr_r = np.linspace(wavelength_hr.min(),
                              wavelength_hr.max(), N_LOW_R)

# ── prof_abs: shape (NWL_HR, NZ=9) matching prof.z
# ────────────────────────
# diff1 pads dz[0]=0 at TOA → prepend a zero column before the NL layer
# ODs.
# (NWL_HR, 8)
_layer_od = np.array(kabs_hr) * thick[np.newaxis, :]
# (NWL_HR, 9)
prof_abs = np.concatenate(
    [np.zeros((len(wavelength_hr), 1), dtype='float32'),
     _layer_od], axis=1)

print(f"wavelength_hr     : [{wavelength_hr.min():.3f}, "
      f"{wavelength_hr.max():.3f}] nm,  {len(wavelength_hr)} points")
print(f"kabs_hr   : {kabs_hr.shape}  max = {float(kabs_hr.max()):.3e} km⁻¹")
print(f"prof_abs  : {prof_abs.shape}  max = {prof_abs.max():.3e}  (layer OD, "
      f"col 0 = TOA zero)")

# %%
# %%time
# multispectral simulation
# independent computation per wavelength, n_photons shared equally
# between all wavelegths (thus Monte Carlo NOISE in the spectrum)
# wavelengths is a list or numpy array
n_wl= wavelength_hr.size
n_photons = 1e5 # photons per wavelength
# monochromatic computation for custom aerosols and cloud
wavelength_0 = 765.
# Aerosols and cloud optical properties using OPAC database as processed
# by the the libradtran (www.libradtran.org)
# set AOT at the reference wavelength wavelength_0 to 0.5
aer1 = AerOPAC( 'desert',  0.25, wavelength_0)
                                # and set aerosol type to
                                # 'maritime_clean'
# tropical atmosphere with O2 absorption in the O2-A band
pro = Atm1D('afglt',
              # particles in atmosphere are a mix of aerosols 1 and 2
              comp=[aer1],
              # set vertical grid, surface altitude at 1.15 km
              grid=_afglt_grid,
              pfgrid=[100, 10, 5, 2., 0.],
              # wavelengths for which the phase function is computed
              wavelength_phase=[wavelength_0],
                                     # optional, otherwise phase
                                     # functions are calculated at all
                                     # bands
                                     # nearest neighbour is then used
                                     # during the RT computation
              no2=False, # NO2 included
              tco3=0., # no ozone
              prof_abs=prof_abs   # (NWL_HR, NL) layer OD = kabs_hr * thick
             )
pro0 = Atm1D('afglt',    # tropical atmosphere, no gaseous absorption
              # particles in atmosphere are a mix of aerosols 1 and 2
              comp=[aer1],
              # set vertical grid, surface altitude at 1.15 km
              grid=_afglt_grid,
              pfgrid=[100, 10, 5, 2., 0.],
              # wavelengths for which the phase function is computed
              wavelength_phase=[wavelength_0],
                                     # optional, otherwise phase
                                     # functions are calculated at all
                                     # bands
                                     # nearest neighbour is then used
                                     # during the RT computation
              no2=False, # NO2 included
              tco3=0., # no ozone
             )

ALBEDO = 0.3
#ALBEDO = 0.
GREY_ALB = AlbedoCst(ALBEDO)
surface = LambSurface(alb=GREY_ALB)
sza, vza = 30., 20.
le = LocalEstimate(th=np.array([vza]) *np.pi/180,
                   phi=np.array([180.]) *np.pi/180, zip=False)
mc = Smartg(alt_pp=True, double=True).run(wavelength=wavelength_hr, le=le,
           th_deg=sza, n_photons=n_photons*n_wl,
           atmosphere=pro, output_layers=1,
           surface=surface)
mc = drop_axes(mc, 'Azimuth angles')
plt.plot(mc['wavelength'], mc['I_up (TOA)'][:, 0], '-r',
         label=r'$\Delta\Phi=${:.0f}°'.format(le.phi[0]*180/np.pi))
plt.legend()
print(' GPU time: ', mc.attrs['kernel time (s)'], 's')

# %% [markdown]
# ### Correlated spectral computations: ALIS method 
# The ALIS method is described in <br>
# Emde, C., Buras, R., and Mayer, B.: ALIS: An efficient method to
# compute high spectral resolution polarized solar radiances using the Monte Carlo approach, J. Quant. Spectrosc. Ra., 112, 1622–1631, 2011.

# %%
# %%time
# Compile with the alis options
# with ALIS options, alt_pp is mandatory
s_alis = Smartg(alt_pp=True, double=True, alis=True)
# then run specifying the alis_options
# main keyword is n_low, the number of low spectral resolution
# computation
# of the scattering correction terms: specify -1 for all wavelengths
# if the alt_pp option is chosen for compilation: slowest procedure for
# photon propagation but
# then the cumulative distance traveled by photons in the atmospheric
# layer is recorded
# specify the number of low spectral resolution computations for the
# scattering correction terms
alis_options = Alis(n_low=N_LOW_R, hist=False)
# and does not export photon histories
m = s_alis.run(seed=SEED, wavelength=wavelength_hr, le=le,
               alis_options=alis_options, th_deg=sza, n_photons=n_photons,
               stdev=True, atmosphere=mc, output_layers=0, surface=surface)
m = drop_axes(m, 'Azimuth angles')

# for the same number of photons, the spectrum is much less noisy
plt.plot(mc['wavelength'], mc['I_up (TOA)'][:, 0], '-r',
         label='no alis: {:.0e} phot.; {:.5f} (s)'.format(
             n_photons*n_wl, float(mc.attrs['kernel time (s)'])))
plt.plot(m['wavelength'], m['I_up (TOA)'][:, 0], '-k',
         label='alis     :{:.0e} phot.; {:.5f} (s)'.format(
             n_photons, float(m.attrs['kernel time (s)'])))
plt.legend()
#plt.ylim(0.21,0.27)
print(' GPU time: ', m.attrs['kernel time (s)'], 's')

# %% [markdown]
# ### Histories

# %%
# We give an example of outputing photon's histories in the alis method.
# This allows to compute various quantities jacobians relative to
# absorption or reflection
# see Multispectral example (output m)
# We extract the absorption coefficient vertical profiles at high
# spectral resolution
kabs = od2k(m, 'OD_abs_atm')[:, 1:]
s_alis = Smartg(alt_pp=True, double=True, alis=True)
# Computations of histories are done on the low resolution grid (only
# scattering !)
# !! n_loop and n_photons should be equal (one UNIQUE pass)
# the surface albedo as to be set to 1. Reflection is computed
# afterwards using the photon histories
# and the high resolution albedos
surf_hist = LambSurface(alb=AlbedoCst(1.))
# specify the number of
alis_options = Alis(n_low=wavelength_lr_r.size, hist=True,
                    max_hist=np.int64(n_photons*20))
# low spectral resolution
# computations for the scattering correction terms, phtons histories are
# recorded and the maximum
# number of histories is set to n_photons*10
# (to avoid memory overflow)
# we use the low resolution grid for the atmospheric properties
# (absorption and scattering)
atmosphere = pro0.calc(wavelength_lr_r)

# %%
# %%time
s_alis = Smartg(alt_pp=True, double=True, alis=True)
m_hist=s_alis.run(seed=SEED, wavelength=wavelength_lr_r, le=le,
                  alis_options=alis_options, th_deg=sza, n_photons=n_photons,
                  n_loop=n_photons, atmosphere=atmosphere, output_layers=1,
                  surface=surf_hist)
print("m_hist {:.2g} photons : {:.5f} (ms)".format(
    n_photons, float(m_hist.attrs['kernel time (s)'])*1000))

# %%
# %%time
# amf simulation — must run HERE, before s_alis.clear_context() hands
# the GPU to JAX
s_amf = Smartg(alt_pp=True, double=True, alis=True,
    amf_variance=True, nscl=_afglt_grid.size, norders=2,
    scatter_classes='scattering_order_per_layer', cdist_wabs=False)
alis_options_amf = Alis(n_low=-1, hist=False)
m_amf = s_amf.run(wavelength=wavelength_lr_r, le=le,
                  alis_options=alis_options_amf, th_deg=sza,
                  n_photons=n_photons, n_loop=n_photons,
                  atmosphere=atmosphere, output_layers=1, surface=surface)
print("m_amf {:.2g} photons : {:.5f} (ms)".format(
    n_photons, float(m_amf.attrs['kernel time (s)'])*1000))

# %%
# %%time
# Since always autoinit = True used until now, only 1 current context
# We can clear manually the context (!!! all previous Smartg object
# cannot be reused, they must be reinitialized !!!)
s_alis.clear_context() # now we can use jax
jax.default_backend()
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
print(jax.devices())

# LEVEL = 0 for TOA, 1 for downward at 0+
(n_h, s_h, d_h, w_h, nrrs_h, nref_h, nsif_h, nvrs_h, nenv_h,
 nint_h, _) = get_histories(m_hist, level=0, verbose=False)

# Upload to JAX device (GPU) once
s_h    = jnp.array(s_h)     # (NLE, NStokes)
d_h    = jnp.array(d_h)     # (NLE, NL)
w_h    = jnp.array(w_h)     # (NLE, NLR)
nref_h = jnp.array(nref_h)  # (NLE,)

# ── Shared per-photon kernel (reused by all subsequent Jacobian cells)
# ────
kabs_j   = jnp.array(kabs,               dtype=jnp.float32)  # (NWL_HR, NL)
# (NWL_HR,)
wavelength_hr_j  = jnp.array(wavelength_hr,              dtype=jnp.float32)
# (NWL_HR,)
alb_hr_j = jnp.array(GREY_ALB.get(wavelength_hr), dtype=jnp.float32)

def _si_one(sik, wi_lr, dij, ki, wavelength_i, kabs_i, alb_i):
    """Beer-Lambert weight: one Stokes component, one photon,
    one wavelength."""
    wi = jnp.interp(wavelength_i, wavelength_lr_r, wi_lr)
    return sik * wi * jnp.exp(-jnp.sum(dij * kabs_i)) * alb_i**ki

# over NLE photons
_si_photons = vmap(_si_one, in_axes=(0, 0, 0, 0, None, None, None))
# over the Stokes components
_si_stokes = vmap(_si_photons,
                  in_axes=(1, None, None, None, None, None, None))
# _si_stokes(s_h, w_h, d_h, nref_h, wavelength_scalar, kabs_1d,
# alb_scalar) → (NStokes, NLE)

_n_h    = float(n_h)
n_wl     = int(kabs_j.shape[0])
n_l     = int(kabs_j.shape[1])
n_stokes = int(s_h.shape[1])

# ── Forward stokes + variance via fori_loop
# ───────────────────────────────
# fori_loop → XLA while_loop; XLA does NOT pre-allocate backward
# checkpoints
# → no rematerialization warning
def _body_fw(i, carry):
    s_sum, s2_sum = carry
    si = _si_stokes(s_h, w_h, d_h, nref_h, wavelength_hr_j[i], kabs_j[i],
                    alb_hr_j[i])  # (NStokes, NLE)
    return (s_sum.at[i].set(si.sum(axis=1)),
            s2_sum.at[i].set((si**2).sum(axis=1)))

_zeros = (jnp.zeros((n_wl, n_stokes)),
          jnp.zeros((n_wl, n_stokes)))
stokes_j, stokes2_j = jit(
    lambda: lax.fori_loop(0, n_wl, _body_fw, _zeros))()
stokes  = np.array(stokes_j).T / _n_h   # (NStokes, NWL_HR)
stokes2 = np.array(stokes2_j).T / _n_h  # (NStokes, NWL_HR)

std   = np.sqrt((stokes2 - stokes**2) / _n_h)
upper = stokes + 1.95*std
lower = stokes - 1.95*std

# Extract data arrays
wavelength_mc   = mc['wavelength'].values
i_mc    = mc['I_up (TOA)'].values[:, 0]
wavelength_alis = m['wavelength'].values
# same wavelength grid as wavelength_hr
i_alis  = m['I_up (TOA)'].values[:, 0]
i_hist  = stokes[0, :]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

# ── Top: spectra
# ───────────────────────────────────────────────────────────
ax1.plot(wavelength_mc, i_mc, '-r',
         label='no alis:   {:.0e} phot.; {:.5f} s'.format(
             n_photons*n_wl, float(mc.attrs['kernel time (s)'])))
ax1.plot(wavelength_alis, i_alis, '-k',
         label='alis:      {:.0e} phot.; {:.5f} s'.format(
             n_photons, float(m.attrs['kernel time (s)'])))
ax1.plot(wavelength_hr, i_hist, '-c',
         label='alis hist: {:.0e} phot.; {:.5f} s'.format(
             n_photons,
             float(m_hist.attrs['kernel time (s)'])))
ax1.fill_between(wavelength_hr, lower[0, :], upper[0, :], facecolor='c',
                 edgecolor='c', alpha=0.4, label='95 % CI (hist)')
ax1.set_ylabel('Reflectance')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── Middle: hist − alis
# ────────────────────────────────────────────────────
alis_ref = np.interp(wavelength_hr, wavelength_alis, i_alis)
diff_alis       = i_hist  - alis_ref
diff_alis_upper = upper[0, :] - alis_ref
diff_alis_lower = lower[0, :] - alis_ref
rel_alis        = 100. * diff_alis       / alis_ref
rel_alis_upper  = 100. * diff_alis_upper / alis_ref
rel_alis_lower  = 100. * diff_alis_lower / alis_ref

ax2.plot(wavelength_hr, diff_alis, '-c', lw=1.5, label='hist − alis')
ax2.fill_between(wavelength_hr, diff_alis_lower, diff_alis_upper,
                 facecolor='c', edgecolor='c', alpha=0.3, label='95 % CI')
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.set_ylabel('Δ Reflectance (absolute)')
ax2.set_title('hist − ALIS (reference)')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2r = ax2.twinx()
ax2r.set_ylabel('Relative difference (%)', color='grey')
ax2r.tick_params(axis='y', labelcolor='grey')
ax2r.set_ylim(100. * np.array(ax2.get_ylim()) / np.mean(alis_ref))

# ── Bottom: hist − no alis
# ─────────────────────────────────────────────────
mc_ref          = np.interp(wavelength_hr, wavelength_mc, i_mc)
diff_mc         = i_hist  - mc_ref
diff_mc_upper   = upper[0, :] - mc_ref
diff_mc_lower   = lower[0, :] - mc_ref
rel_mc          = 100. * diff_mc       / mc_ref
rel_mc_upper    = 100. * diff_mc_upper / mc_ref
rel_mc_lower    = 100. * diff_mc_lower / mc_ref

ax3.plot(wavelength_hr, diff_mc, '-r', lw=1.5, label='hist − no alis')
ax3.fill_between(wavelength_hr, diff_mc_lower, diff_mc_upper, facecolor='r',
                 edgecolor='r', alpha=0.2, label='95 % CI')
ax3.axhline(0, color='k', lw=0.8, ls='--')
ax3.set_xlabel('Wavelength (nm)')
ax3.set_ylabel('Δ Reflectance (absolute)')
ax3.set_title('hist − no-ALIS (reference)')
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3r = ax3.twinx()
ax3r.set_ylabel('Relative difference (%)', color='grey')
ax3r.tick_params(axis='y', labelcolor='grey')
ax3r.set_ylim(100. * np.array(ax3.get_ylim()) / np.mean(mc_ref))

plt.tight_layout()


# %% [markdown]
# ### Pure-scattering run + post-hoc gas absorption (RADIS / Beer-Lambert via histories)
#
# Run SmartG **once** in pure-scattering mode (no gas absorption).  
# Per-photon path lengths `U[Nph, NL]` and weights `w[Nph]` are stored.  
# Any cross-section `σ(ν̃, T_k, P_k)` from **RADIS** (or Atm1D fallback) is then applied post-hoc:
#
# $$\tau_i(\tilde\nu) = \mathbf{U}_i \cdot \boldsymbol{\sigma}(\tilde\nu) \qquad W_i = e^{-\tau_i} \qquad \rho(\tilde\nu) = \frac{\mathbf{w}^\top \mathbf{W}}{\sum w_i}$$
#
# Jacobians $\partial\rho/\partial c$ and SRF convolution come **for free** via `jax.grad`.

# %%
# %%time
# stokes-I spectrum + ∂ρ_I/∂kabs via analytic gradient + fori_loop
#
# kabs_j, wavelength_hr_j, alb_hr_j, _si_stokes, _n_h, n_wl, n_l defined
# in cell 9.
#
# Analytic: ∂Si/∂kabs[l] = -D_i[l]·Si  →  ∂ρ_I/∂kabs[l] = -Σᵢ
# d_h[i,l]·si_I[i] / N

def _body_i(i, carry):
    rho_arr, jac_arr = carry
    si = _si_stokes(s_h, w_h, d_h, nref_h, wavelength_hr_j[i], kabs_j[i],
                    alb_hr_j[i])  # (NStokes, NLE)
    rho_i_j = jnp.sum(si[0]) / _n_h                  # scalar
    jac_i = -jnp.dot(d_h.T, si[0]) / _n_h          # (NL,)
    return rho_arr.at[i].set(rho_i_j), jac_arr.at[i].set(jac_i)

rho_jax_j, di_dkabs_j = jit(
    lambda: lax.fori_loop(0, n_wl, _body_i,
                          (jnp.zeros(n_wl), jnp.zeros((n_wl, n_l))))
)()
rho_jax  = np.array(rho_jax_j)    # (NWL_HR,)
di_dkabs = np.array(di_dkabs_j)   # (NWL_HR, NL)

# ── SRF convolution
# ───────────────────────────────────────────────────────
wavelength_c, fwhm = 766.5, 1.25
sig   = fwhm / (2 * np.sqrt(2 * np.log(2)))
srf   = np.exp(-0.5 * ((wavelength_hr - wavelength_c) / sig)**2)
srf  /= np.trapezoid(srf, wavelength_hr)
rho_conv = float(np.trapezoid(srf * rho_jax, wavelength_hr))
print(f"SRF-convolved reflectance at {wavelength_c} nm (FWHM={fwhm} nm): "
      f"{rho_conv:.5f}")

# ── Plots
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(wavelength_hr, rho_jax, 'b-', lw=1.5, label='HR spectrum')
axes[0].axhline(rho_conv, color='r', ls='--', lw=1.2,
                label=f'SRF-convolved = {rho_conv:.4f}')
ax2 = axes[0].twinx()
ax2.fill_between(wavelength_hr, srf, alpha=0.20, color='orange')
ax2.set_ylabel('SRF', color='darkorange')
axes[0].set_xlabel('wavelength (nm)')
axes[0].set_ylabel('ρ')
axes[0].set_title('HR spectrum + Gaussian SRF convolution')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

im = axes[1].imshow(di_dkabs.T, aspect='auto', origin='upper',
                    extent=[wavelength_hr[0], wavelength_hr[-1],
                            di_dkabs.shape[1], 0],
                    cmap='RdBu_r')
plt.colorbar(im, ax=axes[1],
             label=r'$\partial\rho/\partial k_\mathrm{abs}$  (km)')
axes[1].set_xlabel('wavelength (nm)')
axes[1].set_ylabel('layer (0 = TOA)')
axes[1].set_title(r'Jacobian $\partial\rho/\partial k_\mathrm{abs}$  '
                  r'(analytic)')

plt.tight_layout()

# %%
# %%time
# ─── Gas Absorber Jacobians  ∂ρ/∂T  and  ∂ρ/∂VMR
# ─────────────────────────
#
# Inherits from cell 4:  MOLECULE, wn_min, wn_max, sfac, vmr_profile,
# n_layers, t_mid, p_mid, n_mid, z_mid, wavelength_hr
#
# Computes:
# ∂ρ/∂T_k(ν)     — temperature Jacobian   (forward-difference ΔT = 1 K)
#   ∂ρ/∂VMR_k(ν)   — concentration Jacobian (analytical, linear in VMR)

import contextlib
import io
import logging
import warnings

import jax.numpy as jnp

delta_t = 1.0    # K  (forward-difference step for T-Jacobian)
_sink = io.StringIO()

# ── 1. kabs per layer at reference T and T+ΔT
# ─────────────────────────────
kabs_t   = np.zeros((len(wavelength_hr), n_layers), dtype='float32')
kabs_t_dt = np.zeros_like(kabs_t)

with (warnings.catch_warnings(),
      contextlib.redirect_stdout(_sink),
      contextlib.redirect_stderr(_sink)):
    warnings.simplefilter('ignore')
    logging.disable(logging.CRITICAL)
    for k_idx in range(n_layers):
        vmr_k = float(vmr_profile[k_idx])
        p_bar = float(p_mid[k_idx]) * 1e-3          # hPa → bar
        n_k   = vmr_k * n_mid[k_idx]               # number density [molec/cm³]

        s0 = sfac.eq_spectrum(Tgas=float(t_mid[k_idx]), pressure=p_bar,
                              mole_fraction=vmr_k)
        wn0, x0 = s0.get('xsection')
        wavelength_s0 = 1e7 / wn0[::-1]
        kabs_t[:, k_idx] = np.interp(
            wavelength_hr, wavelength_s0, x0[::-1]) * n_k * 1e5

        s1 = sfac.eq_spectrum(Tgas=float(t_mid[k_idx]) + delta_t,
                              pressure=p_bar, mole_fraction=vmr_k)
        wn1, x1 = s1.get('xsection')
        wavelength_s1 = 1e7 / wn1[::-1]
        kabs_t_dt[:, k_idx] = np.interp(
            wavelength_hr, wavelength_s1, x1[::-1]) * n_k * 1e5

    logging.disable(logging.NOTSET)

print(f"kabs_T: {kabs_t.shape},  max = {kabs_t.max():.3e} km⁻¹")

# ── 2. T-Jacobian via chain rule
# ───────────────────────────────────────────
# (n_wl, NL) km⁻¹/K
dkabs_dt   = (kabs_t_dt - kabs_t) / delta_t
dkabs_dt_j = jnp.array(dkabs_dt, dtype=jnp.float32)
di_dkabs_j = jnp.array(di_dkabs, dtype=jnp.float32)
drho_dt_wl = di_dkabs_j * dkabs_dt_j
dwl        = float(wavelength_hr[1] - wavelength_hr[0])
drho_dt    = np.array(jnp.sum(drho_dt_wl, axis=0) * dwl)         # (NL,)
print(f"max |∂ρ/∂T_{MOLECULE}|   = {np.abs(drho_dt).max():.3e} K⁻¹")

# ── 3. VMR-Jacobian (analytical: kabs ∝ VMR)
# ──────────────────────────────
with np.errstate(divide='ignore', invalid='ignore'):
    dkabs_dvmr = np.where(vmr_profile[None, :] > 0,
                          kabs_t / vmr_profile[None, :], 0.)       # (n_wl, NL)
drho_dvmr_wl = di_dkabs * dkabs_dvmr
drho_dvmr    = drho_dvmr_wl.sum(axis=0) * dwl                     # (NL,)
print(f"max |∂ρ/∂VMR_{MOLECULE}| = {np.abs(drho_dvmr).max():.3e}")

# ── 4. Plots
# ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f'{MOLECULE} gas Jacobians   '
             f'(band {wn_min:.0f}–{wn_max:.0f} cm⁻¹)',
             fontsize=13)

# [0,0] kabs spectrum per layer
for k_idx in range(n_layers):
    axes[0, 0].plot(wavelength_hr, kabs_t[:, k_idx], lw=0.8,
                    label=f'{z_mid[k_idx]:.1f} km' if k_idx % 2 == 0 else None)
axes[0, 0].set_xlabel('wavelength (nm)')
axes[0, 0].set_ylabel(r'$k_\mathrm{abs}$  (km⁻¹)')
axes[0, 0].set_title(f'{MOLECULE} absorption coefficient')
axes[0, 0].legend(fontsize=7, ncol=2)
axes[0, 0].grid(True, alpha=0.3)

# [0,1] ∂ρ/∂T profile
axes[0, 1].plot(drho_dt, z_mid, color='C3', lw=1.5)
axes[0, 1].axvline(0, color='k', lw=0.5, ls='--')
axes[0, 1].set_xlabel(r'$\partial\rho/\partial T_k$  [K$^{-1}$]')
axes[0, 1].set_ylabel('Altitude (km)')
axes[0, 1].set_title(f'{MOLECULE} Temperature Jacobian\n(band-integrated)')
axes[0, 1].grid(True, alpha=0.3)

# [1,0] ∂ρ/∂VMR spectral heatmap
vmax_v = float(np.abs(drho_dvmr_wl).max()) or 1e-30
im0 = axes[1, 0].imshow(drho_dvmr_wl.T, aspect='auto', origin='upper',
                        extent=[wavelength_hr[0], wavelength_hr[-1],
                                n_layers, 0],
                        cmap='RdBu_r', vmin=-vmax_v, vmax=vmax_v)
plt.colorbar(im0, ax=axes[1, 0], label=r'$\partial\rho/\partial\mathrm{VMR}$')
axes[1, 0].set_xlabel('wavelength (nm)')
axes[1, 0].set_ylabel('layer (0 = TOA)')
axes[1, 0].set_title(f'{MOLECULE} VMR Jacobian (spectral)')

# [1,1] ∂ρ/∂VMR profile
axes[1, 1].plot(drho_dvmr, z_mid, color='C2', lw=1.5)
axes[1, 1].axvline(0, color='k', lw=0.5, ls='--')
axes[1, 1].set_xlabel(r'$\partial\rho/\partial\mathrm{VMR}$')
axes[1, 1].set_ylabel('Altitude (km)')
axes[1, 1].set_title(f'{MOLECULE} VMR Jacobian (band-integrated)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# %%time
# Pressure Jacobian  ∂ρ/∂P_k  via finite-difference + chain rule
# Inherits: sfac, vmr_profile, n_layers, t_mid, p_mid, n_mid,
# wavelength_hr (cell 4)
# Reuses  : di_dkabs (cell 11)
#
# ΔP = 1 hPa forward difference on RADIS pressure broadening
# n_mid kept fixed (pure spectroscopic sensitivity, not hydrostatic)

import contextlib
import io
import logging
import warnings

import jax.numpy as jnp

delta_p = 1.0   # hPa
_sink2 = io.StringIO()

kabs_p   = np.zeros((len(wavelength_hr), n_layers), dtype='float32')
kabs_p_dp = np.zeros_like(kabs_p)

with (warnings.catch_warnings(),
      contextlib.redirect_stdout(_sink2),
      contextlib.redirect_stderr(_sink2)):
    warnings.simplefilter('ignore')
    logging.disable(logging.CRITICAL)
    for k_idx in range(n_layers):
        vmr_k = float(vmr_profile[k_idx])
        n_k   = vmr_k * n_mid[k_idx]

        s0 = sfac.eq_spectrum(Tgas=float(t_mid[k_idx]),
                              pressure=float(p_mid[k_idx]) * 1e-3,
                              mole_fraction=vmr_k)
        wn_s0, sig0 = s0.get('xsection')
        wavelength_s0 = 1e7 / wn_s0[::-1]
        kabs_p[:, k_idx] = np.interp(
            wavelength_hr, wavelength_s0, sig0[::-1]) * n_k * 1e5

        s1 = sfac.eq_spectrum(Tgas=float(t_mid[k_idx]),
                              pressure=(float(p_mid[k_idx]) + delta_p) * 1e-3,
                              mole_fraction=vmr_k)
        wn_s1, sig1 = s1.get('xsection')
        wavelength_s1 = 1e7 / wn_s1[::-1]
        kabs_p_dp[:, k_idx] = np.interp(wavelength_hr, wavelength_s1,
                                        sig1[::-1]) * n_k * 1e5

    logging.disable(logging.NOTSET)

# forward-difference ∂kabs/∂P  [km⁻¹ hPa⁻¹],  shape (NWL_HR, NL)
dkabs_dp   = (kabs_p_dp - kabs_p) / delta_p
dkabs_dp_j = jnp.array(dkabs_dp, dtype=jnp.float32)
print(f"dkabs_dP: {dkabs_dp.shape},  "
      f"max |∂kabs/∂P| = {float(np.abs(dkabs_dp).max()):.3e} km⁻¹/hPa")

# --- chain rule  ∂ρ/∂P_k(ν) = di_dkabs[ν,k] * dkabs_dp[ν,k]
# -------------
di_dkabs_j = jnp.array(di_dkabs, dtype=jnp.float32)   # (NWL_HR, NL)
drho_dp_wl = di_dkabs_j * dkabs_dp_j                   # (NWL_HR, NL)

dwl = float(wavelength_hr[1] - wavelength_hr[0])
drho_dp = jnp.sum(drho_dp_wl, axis=0) * dwl             # (NL,)
drho_dp = np.array(drho_dp)
print(f"∂ρ/∂P  profile: shape={drho_dp.shape}, "
      f"max |∂ρ/∂P| = {np.abs(drho_dp).max():.3e} hPa⁻¹")

# --- plot
# ------------------------------------------------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax_p = axes[0]
ax_p.plot(drho_dp, z_mid, color='C0', lw=1.5)
ax_p.axvline(0, color='k', lw=0.5, ls='--')
ax_p.set_xlabel(r'$\partial\rho/\partial P_k$  [hPa$^{-1}$]')
ax_p.set_ylabel('Altitude (km)')
ax_p.set_title(f'{MOLECULE} Pressure Jacobian\n(integrated over band)')
ax_p.grid(True, alpha=0.3)

ax_2d = axes[1]
ext = [wavelength_hr[0], wavelength_hr[-1], z_mid[0], z_mid[-1]]
vmax = float(np.abs(drho_dp_wl).max())
im = ax_2d.imshow(drho_dp_wl.T, aspect='auto', origin='lower',
                  extent=ext, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax_2d,
             label=r'$\partial\rho/\partial P_k$ [nm$^{-1}$·hPa$^{-1}$]')
ax_2d.set_xlabel('Wavelength (nm)')
ax_2d.set_ylabel('Altitude (km)')
ax_2d.set_title(rf'$\partial\rho/\partial P_k(\lambda)$  — {MOLECULE}')

plt.tight_layout()
plt.show()


# %%
# %%time
# Polarized reflectance + Jacobians via analytic gradients
#
# Reuses _si_stokes, _n_h, n_wl, n_l defined in the previous cell.
#
# Analytic gradients of Si = sik · w_i · exp(-D_i · kabs) · alb^ki :
#   ∂Si/∂kabs[l] = -D_i[l] · Si   →  ∂ρ/∂kabs[l] = -Σᵢ d_h[i,l]·sᵢ / N
# ∂Si/∂alb     = (ki/alb) · Si  →  ∂ρ/∂alb    =  Σᵢ nref_h[i]·sᵢ /
# (alb·N)
#
# fori_loop → XLA while_loop; no backward-pass checkpoint budget.
#   dolp(ν) = √(ρ_Q² + ρ_U²) / ρ_I

def _body_pol(i, carry):
    r_q, r_i, r_u, dq_k, dq_a, di_a = carry
    kabs_i = kabs_j[i]
    wavelength_i   = wavelength_hr_j[i]
    alb_i  = alb_hr_j[i]

    # (NStokes, NLE)
    si = _si_stokes(s_h, w_h, d_h, nref_h, wavelength_i, kabs_i, alb_i)

    rho_i_j = jnp.sum(si[0]) / _n_h
    rho_q_j = jnp.sum(si[1]) / _n_h
    rho_u_j = jnp.sum(si[2]) / _n_h

    dq_dkabs_i = -jnp.dot(d_h.T, si[1]) / _n_h          # (NL,)
    safe_alb   = jnp.where(alb_i > 0., alb_i, 1.)
    dq_dalb_i  = jnp.dot(nref_h, si[1]) / (safe_alb * _n_h)
    di_dalb_i  = jnp.dot(nref_h, si[0]) / (safe_alb * _n_h)

    return (
        r_q.at[i].set(rho_q_j),
        r_i.at[i].set(rho_i_j),
        r_u.at[i].set(rho_u_j),
        dq_k.at[i].set(dq_dkabs_i),
        dq_a.at[i].set(dq_dalb_i),
        di_a.at[i].set(di_dalb_i),
    )

init_pol = (
    jnp.zeros(n_wl), jnp.zeros(n_wl), jnp.zeros(n_wl),
    jnp.zeros((n_wl, n_l)), jnp.zeros(n_wl), jnp.zeros(n_wl),
)
rho_q, rho_i, rho_u, dq_dkabs, dq_dalb, di_dalb = jit(
    lambda: lax.fori_loop(0, n_wl, _body_pol, init_pol)
)()

rho_q    = np.array(rho_q)
rho_u    = np.array(rho_u)
rho_i    = np.array(rho_i)
dq_dkabs = np.array(dq_dkabs)
dq_dalb  = np.array(dq_dalb)
di_dalb  = np.array(di_dalb)

rho_pol = np.sqrt(rho_q**2 + rho_u**2)
dolp    = rho_pol / np.maximum(rho_i, 1e-12)
print(f"peak  ρ_Q   = {rho_q.min():.4f} … {rho_q.max():.4f}")
print(f"peak  ρ_U   = {rho_u.min():.4f} … {rho_u.max():.4f}")
print(f"peak  ρ_pol = {rho_pol.max():.4f}")
print(f"peak  DoLP  = {dolp.max():.4f}")
print(f"∂ρ_I/∂alb range: {di_dalb.min():.4f} … {di_dalb.max():.4f}")
print(f"∂ρ_Q/∂alb range: {dq_dalb.min():.4f} … {dq_dalb.max():.4f}")

# --- plots (2×2 layout)
# ---------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# ── [0,0] I / Q / U reflectance
# ───────────────────────────────────────────
ax0  = axes[0, 0]
ax0r = ax0.twinx()
l1, = ax0.plot( wavelength_hr, rho_i, 'b',  lw=1.2, label=r'$\rho_I$ (left)')
l2, = ax0r.plot(wavelength_hr, rho_q, 'r',  lw=1.2, label=r'$\rho_Q$ (right)')
l3, = ax0r.plot(wavelength_hr, rho_u, 'g',  lw=1.2, label=r'$\rho_U$ (right)')
ax0r.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
_qu_lo = min(rho_q.min(), rho_u.min())
_qu_hi = max(rho_q.max(), rho_u.max())
_pad   = (_qu_hi - _qu_lo) * 0.15 if _qu_hi != _qu_lo else max(
    abs(_qu_hi), 1e-12) * 0.15
ax0r.set_ylim(_qu_lo - _pad, _qu_hi + _pad)
ax0.set_xlabel('wavelength (nm)')
ax0.set_ylabel(r'$\rho_I$', color='b')
ax0.tick_params(axis='y', labelcolor='b')
ax0r.set_ylabel(r'$\rho_Q$,  $\rho_U$', color='dimgrey')
ax0r.tick_params(axis='y', labelcolor='dimgrey')
ax0.set_title('I / Q / U reflectance')
ax0.legend(handles=[l1, l2, l3], fontsize=9, loc='best')
ax0.grid(True, alpha=0.3)

# ── [0,1] dolp (left) + sqrt(Q²+U²) (right)
# ──────────────────────────────
ax1  = axes[0, 1]
ax1r = ax1.twinx()
l4, = ax1.plot( wavelength_hr, dolp,    'b',   lw=1.2, label=r'DoLP (left)')
l5, = ax1r.plot(wavelength_hr, rho_pol, 'r--', lw=1.2,
                label=r'$\sqrt{\rho_Q^2+\rho_U^2}$ (right)')
ax1.set_xlabel('wavelength (nm)')
ax1.set_ylabel('DoLP', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1r.set_ylabel(r'$\sqrt{\rho_Q^2+\rho_U^2}$', color='r')
ax1r.tick_params(axis='y', labelcolor='r')
ax1.set_title('DoLP vs polarized reflectance')
ax1.legend(handles=[l4, l5], fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

# ── [1,0] Jacobian ∂ρ_Q/∂kabs (heatmap)
# ──────────────────────────────────
ax2 = axes[1, 0]
vmax = float(np.abs(dq_dkabs).max())
im = ax2.imshow(dq_dkabs.T, aspect='auto', origin='upper',
                extent=[wavelength_hr[0], wavelength_hr[-1],
                        dq_dkabs.shape[1], 0],
                cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax2,
             label=r'$\partial\rho_Q/\partial k_\mathrm{abs}$  (km)')
ax2.set_xlabel('wavelength (nm)')
ax2.set_ylabel('layer (0 = TOA)')
ax2.set_title(r'Jacobian $\partial\rho_Q/\partial k_\mathrm{abs}$')

# ── [1,1] Albedo Jacobians
# ────────────────────────────────────────────────
ax3  = axes[1, 1]
ax3r = ax3.twinx()
l6, = ax3.plot(wavelength_hr, di_dalb, 'b', lw=1.2,
               label=r'$\partial\rho_I/\partial a$ (left)')
l7, = ax3r.plot(wavelength_hr, dq_dalb, 'r', lw=1.2,
                label=r'$\partial\rho_Q/\partial a$ (right)')
ax3r.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
ax3.set_xlabel('wavelength (nm)')
ax3.set_ylabel(r'$\partial\rho_I/\partial a$', color='b')
ax3.tick_params(axis='y', labelcolor='b')
ax3r.set_ylabel(r'$\partial\rho_Q/\partial a$', color='r')
ax3r.tick_params(axis='y', labelcolor='r')
ax3.set_title('Albedo Jacobians')
ax3.legend(handles=[l6, l7], fontsize=9, loc='best')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# # AMFs

# %%
# amf analysis — uses m_amf computed in the previous cell (before JAX
# init)
# In the cdist output, the iAMF axis stores:
#   iAMF=0: Σ(w), iAMF=1: Σ(d·w), iAMF=2: Σ(d²·w)
# With nscl>1, an extra iSCL axis decomposes by last-scattering-layer
# class.
#
# Variance decomposition (law of total variance):
#   Var(D) = Var_within + Var_between
#   Var_within  = Σ_c (W_c/W) · Var_c(D)       (intra-class variability)
#   Var_between = Σ_c (W_c/W) · (μ_c - μ)²     (inter-class variability)

cdist = m_amf['cdist_up (TOA)']
ax_names = list(cdist.dims)
print(f"cdist shape = {cdist.shape}, axes = {ax_names}")

# ── Collapse azimuth + theta to index 0, keep nlayer / nscl / niamf
# ───────
cd = cdist.data   # raw numpy array
idx = [slice(None)] * cd.ndim
for i, name in enumerate(ax_names):
    if name in ('Azimuth angles', 'Zenith angles'):
        idx[i] = 0
cd = cd[tuple(idx)]   # (nlayer, [nscl,] niamf)

has_scl = 'iSCL' in ax_names
nscl    = cdist.shape[ax_names.index('iSCL')] if has_scl else 1

if has_scl:
    # cd shape: (nlayer, nscl, niamf)
    w_cls          = cd[:, :, 0]     # (nlayer, nscl)
    mean_dist_cls  = cd[:, :, 1] / np.where(w_cls > 0, w_cls, 1.)
    mean_dist2_cls = cd[:, :, 2] / np.where(w_cls > 0, w_cls, 1.)
    var_dist_cls   = mean_dist2_cls - mean_dist_cls**2

    W         = w_cls.sum(axis=1)
    wd        = cd[:, :, 1].sum(axis=1)
    wd2       = cd[:, :, 2].sum(axis=1)
    mean_dist = wd  / np.where(W > 0, W, 1.)

    frac_cls    = w_cls / np.where(W > 0, W, 1.)[:, None]
    var_within  = (frac_cls * var_dist_cls).sum(axis=1)
    var_between = (frac_cls
                   * (mean_dist_cls - mean_dist[:, None])**2).sum(axis=1)
    var_dist    = var_within + var_between
else:
    # cd shape: (nlayer, niamf)
    W          = cd[:, 0]
    mean_dist  = cd[:, 1] / np.where(W > 0, W, 1.)
    mean_dist2 = cd[:, 2] / np.where(W > 0, W, 1.)
    var_dist   = mean_dist2 - mean_dist**2

# Layer thicknesses and amf = <D> / Δz
thick_amf = abs(np.diff(m_amf['z_atm'].values))
amf       = mean_dist / thick_amf

# Analytical single-scatter amf for comparison
sza_rad = sza * np.pi / 180
vza_rad = le.th[0]
amf_ss  = 1./np.cos(sza_rad) + 1./np.cos(vza_rad)

std_amf = np.sqrt(np.maximum(var_dist, 0.)) / thick_amf

# --- Diagnostic: total ---
print(f"Expected single-scatter AMF = sec({sza:.0f}°) + "
      f"sec({vza_rad*180/np.pi:.0f}°) = {amf_ss:.4f}")
print(f"nscl = {nscl}")
print(f"\nTotal moments per layer:")
if has_scl:
    print(f"{'lay':>3s}  {'Δz(km)':>8s}  {'<D>':>10s}  {'Var(D)':>12s}  "
          f"{'Var_within':>12s}  {'Var_between':>12s}  {'%between':>9s}  "
          f"{'AMF':>8s}  {'σ(AMF)':>8s}  {'CV%':>6s}")
    for i in range(len(amf)):
        cv     = 100*std_amf[i]/amf[i]     if amf[i]     > 0 else 0
        pct_bw = 100*var_between[i]/var_dist[i] if var_dist[i] > 0 else 0
        print(f"{i:3d}  {thick_amf[i]:8.3f}  {mean_dist[i]:10.4f}  "
              f"{var_dist[i]:12.6f}  {var_within[i]:12.6f}  "
              f"{var_between[i]:12.6f}  {pct_bw:8.1f}%  {amf[i]:8.4f}  "
              f"{std_amf[i]:8.4f}  {cv:6.1f}")
else:
    print(f"{'lay':>3s}  {'Δz(km)':>8s}  {'<D>':>10s}  {'Var(D)':>12s}  "
          f"{'AMF':>8s}  {'σ(AMF)':>8s}  {'CV%':>6s}")
    for i in range(len(amf)):
        cv = 100*std_amf[i]/amf[i] if amf[i] > 0 else 0
        print(f"{i:3d}  {thick_amf[i]:8.3f}  {mean_dist[i]:10.4f}  "
              f"{var_dist[i]:12.6f}  {amf[i]:8.4f}  {std_amf[i]:8.4f}  "
              f"{cv:6.1f}")

# --- Per scatter-class breakdown (if present) ---
if has_scl:
    for label, arr in [
        ("Per scatter-class weights (fraction of total W per layer):", None),
        ("Per scatter-class mean distance:", mean_dist_cls),
        ("Per scatter-class variance:", var_dist_cls),
    ]:
        print(f"\n{label}")
        hdr = f"{'lay':>3s}  " + "  ".join(
            [f"{'cls'+str(j):>8s}" for j in range(nscl)])
        print(hdr)
        for i in range(len(amf)):
            if arr is None:
                fracs = w_cls[i] / W[i] if W[i] > 0 else w_cls[i]*0
                vals  = [f"{fracs[j]:8.4f}" for j in range(nscl)]
            else:
                vals = [f"{arr[i,j]:8.4f}" if w_cls[i, j] > 0
                        else f"{'---':>8s}"
                        for j in range(nscl)]
            print(f"{i:3d}  " + "  ".join(vals))

# --- Plot: amf with error bars ---
fig, axes = plt.subplots(1, 2 if has_scl else 1,
                         figsize=(14 if has_scl else 8, 4))
ax = axes[0] if has_scl else axes
ax_var = axes[1] if has_scl else None

x = np.arange(len(amf))
width = 0.4

ax.bar(x - width/2, mean_dist, color='b', width=width,
       label='mean distance (km)')
ax.set_ylabel('mean distance traveled (km)', color='b')
ax.set_xlabel('layer #')
ax2 = ax.twinx()
ax2.bar(x + width/2, amf, color='r', width=width, alpha=0.7, label='AMF (MC)')
ax2.errorbar(x + width/2, amf, yerr=std_amf, fmt='none', ecolor='k', capsize=3,
             label=r'$\pm\sigma$(AMF)')
ax2.axhline(amf_ss, color='green', ls='--', lw=1.5,
            label=f'single-scatter AMF = {amf_ss:.3f}')
ax2.set_ylabel('AMF', color='r')
ax2.set_ylim(bottom=0)
ax.set_title(f'AMF per layer (SZA={sza:.0f}°, VZA={vza_rad*180/np.pi:.0f}°, '
             f'nscl={nscl})')
fig.legend(loc='upper right', bbox_to_anchor=(0.55 if has_scl else 0.95, 0.95),
           fontsize=8)

if has_scl and ax_var is not None:
    ax_var.bar(x, var_within / thick_amf**2, color='steelblue',
               label=r'$\mathrm{Var_{within}}$ (intra-class)')
    ax_var.bar(x, var_between / thick_amf**2, bottom=var_within / thick_amf**2,
               color='coral', label=r'$\mathrm{Var_{between}}$ (inter-class)')
    ax_var.set_xlabel('layer #')
    ax_var.set_ylabel(r'Var(AMF)')
    ax_var.set_title('Variance decomposition by scattering class')
    ax_var.legend(fontsize=8)

plt.tight_layout()

# %%
if has_scl:
    nlayer = len(amf)
    amf_cls = mean_dist_cls / thick[:, None]   # (nlayer, nscl)
    std_cls = np.sqrt(np.maximum(var_dist_cls, 0.)) / thick[:, None]
    cmap = plt.get_cmap('tab20', nscl)

    # --- Figure: amf per scatter class (grouped bars) ---
    fig2, (ax_amf, ax_std) = plt.subplots(1, 2, figsize=(14, 4))
    active_cls = [j for j in range(nscl) if np.any(w_cls[:, j] > 0)]
    nc = len(active_cls)
    bw = 0.8 / (nc + 1)

    for k, j in enumerate(active_cls):
        offset = (k - nc/2) * bw
        mask = w_cls[:, j] > 0
        ax_amf.bar(x[mask] + offset, amf_cls[mask, j], width=bw, color=cmap(j),
                   alpha=0.8, label=f'cls {j}')
        ax_std.bar(x[mask] + offset, std_cls[mask, j], width=bw, color=cmap(j),
                   alpha=0.8, label=f'cls {j}')
    # total
    ax_amf.bar(x + (nc/2)*bw, amf, width=bw, color='k', alpha=0.5,
               label='total')
    ax_amf.axhline(amf_ss, color='green', ls='--', lw=1.5,
                   label=f'single-scatter = {amf_ss:.3f}')
    ax_amf.set_xlabel('layer #')
    ax_amf.set_ylabel('AMF')
    ax_amf.set_title('Mean AMF per scatter class')
    ax_amf.legend(fontsize=7, ncol=min(nc+2, 5))
    ax_amf.set_ylim(bottom=0)

    ax_std.bar(x + (nc/2)*bw, std_amf, width=bw, color='k', alpha=0.5,
               label='total')
    ax_std.set_xlabel('layer #')
    ax_std.set_ylabel(r'$\sigma$(AMF)')
    ax_std.set_title(r'$\sigma$(AMF) per scatter class')
    ax_std.legend(fontsize=7, ncol=min(nc+2, 5))
    plt.tight_layout()

else:
    print("No scatter-class decomposition (nscl=1). Re-run with nscl>1 to see "
          "iSCL plots.")

# %% [markdown]
# # Validation vs DA

# %%
# Keep the validation run within a practical GPU memory limit while
# retaining
# enough Monte Carlo photons for a useful precision comparison.
# n_photons is the number of Monte Carlo photons simulated by each run.
# max_hist is the number of photon-history rows reserved on the GPU.
_VALIDATION_MAX_PHOTONS = 1_000_000
_VALIDATION_MAX_HIST = 2_000_000
_original_validation_run = Smartg.run


def _memory_safe_validation_run(self, *args, **kwargs):
    global n_photons
    n_photons = min(int(n_photons), _VALIDATION_MAX_PHOTONS)
    kwargs["n_photons"] = min(
        int(kwargs.get("n_photons", n_photons)),
        _VALIDATION_MAX_PHOTONS,
    )
    kwargs["n_loop"] = min(
        int(kwargs.get("n_loop", n_photons)),
        _VALIDATION_MAX_PHOTONS,
    )
    alis_options = kwargs.get("alis_options")
    if alis_options is not None and alis_options.hist:
        kwargs["alis_options"] = Alis(
            n_low=alis_options.n_low,
            hist=True,
            max_hist=min(int(alis_options.max_hist),
                         _VALIDATION_MAX_HIST),
            n_jac=alis_options.n_jac,
            n_jac_abs=alis_options.n_jac_abs,
        )
    return _original_validation_run(self, *args, **kwargs)


Smartg.run = _memory_safe_validation_run

# %%
typ='desert' # tau=0.25
n_photons=2e6
####################""""""
fgas = Path(DIR_AUXDATA) / 'validation' / f"cTauGas_ray_{typ}_O2.dat"
gas_valid   = diff1(np.loadtxt(fgas, skiprows=7)[:, 1:].T, axis=1)
z_valid   = np.loadtxt(fgas, skiprows=7)[:, 0]
w_valid   = np.array(open(fgas).readlines()[5].split()).astype(float)
fray = Path(DIR_AUXDATA) / 'validation' / f"cTauRay_ray_{typ}_O2.dat"
ray_valid = diff1(np.loadtxt(fray, skiprows=7)[:, 1:].T, axis=1)
faer_abs = Path(DIR_AUXDATA) / 'validation' / f"cTauAbs_ptcle_ray_{typ}_O2.dat"
aer_abs_valid = diff1(np.loadtxt(faer_abs, skiprows=7)[:, 1:].T, axis=1)
faer_sca = Path(DIR_AUXDATA) / 'validation' / f"cTauSca_ptcle_ray_{typ}_O2.dat"
aer_sca_valid = diff1(np.loadtxt(faer_sca, skiprows=7)[:, 1:].T, axis=1)
# aerosols phase matrix import
faer_phase= Path(DIR_AUXDATA) / 'validation' / f"phasemat_ray_{typ}_O2.dat"
f=open(faer_phase, 'r')
N=np.genfromtxt(faer_phase, usecols=range(1), max_rows=1, dtype=int)
wavelength_phase=[]
n_pf=3
data=np.zeros((n_pf, 1, N, 5), dtype=np.float32)
for k in range(n_pf):
    wavelength_phase.append(np.genfromtxt(faer_phase, usecols=range(1),
                                          skip_header=(1+(2+N)*k), max_rows=1))
    data[k, 0, :, :] = np.genfromtxt(faer_phase, usecols=range(5),
                                  skip_header=(1+(2+N)*k+2), max_rows=N)
data=data.swapaxes(2, 3)

# From iparper to standard phase convention
pha_data = data[:, :, 1:, :].copy()
pha_data[:, :, 0, :] = (data[:, :, 1, :] + data[:, :, 2, :])*0.5
pha_data[:, :, 1, :] = (data[:, :, 1, :] - data[:, :, 2, :])*0.5

phase_valid = xr.DataArray(pha_data,
        dims=['wavelength_phase', 'z_phase', 'nphamat', 'theta_atm'],
        coords={'wavelength_phase': wavelength_phase, 'z_phase': [0],
                  'theta_atm': data[0, 0, 0, :]})
data_valid = np.loadtxt(
    Path(DIR_AUXDATA) / 'validation'
    / f"artdeco_lbl_nstr_32_ray_{typ}_O2.dat")
aer_ext_valid  = aer_sca_valid + aer_abs_valid
aer_ssa_valid  = aer_sca_valid / aer_ext_valid
aer_ssa_valid[aer_ext_valid==0]=1.
comp=[AerOPAC('desert', 0.5, 550., phase=phase_valid)]
atm_valid = Atm1D('afglmw', grid=z_valid, tco3=0., no2=False,
                  wavelength_phase=wavelength_phase, comp=comp,
                  prof_ray=ray_valid, prof_aer=(aer_ext_valid, aer_ssa_valid),
                  prof_abs=gas_valid)
sigma_valid = od2k(atm_valid.calc(w_valid), 'OD_abs_atm')[:, 1:]
###############

le = LocalEstimate(th_deg=np.array([20.]),
                   phi_deg=np.array([180.]), zip=False)
N_LOW = 3
wavelength_lr= np.linspace(w_valid.min(), w_valid.max(), num=N_LOW)

sg = Smartg(alis=True, alt_pp=True)
m1 = sg.run(seed=0, th_deg=30., wavelength=w_valid, surface=None, le=le,
            beer=0, atmosphere=atm_valid.calc(w_valid), depo=0.,
            alis_options=Alis(n_low=N_LOW, hist=False),
            n_photons=n_photons,
            n_loop=n_photons, n_icdf=1e3)
m1 = drop_axes(m1, 'Zenith angles', 'Azimuth angles')
m2 = sg.run(seed=0, th_deg=30., wavelength=w_valid, surface=None, le=le,
            beer=0, atmosphere=atm_valid.calc(w_valid), depo=0.,
            alis_options=Alis(n_low=N_LOW, hist=True,
                              max_hist=np.int64(n_photons*5)),
            n_photons=n_photons, n_loop=n_photons, n_icdf=1e3)
m2 = drop_axes(m2, 'Zenith angles', 'Azimuth angles')
sg.clear_context()
print ('GPU time no hist: %.4f'%float(m1.attrs['kernel time (s)']), 's')
print ('GPU time hist: %.4f'%float(m2.attrs['kernel time (s)']), 's')

# run on CPU to avoid slow GPU XLA compilation
with jax.default_device(jax.devices("cpu")[0]):
    n_h, s_h, d_h, w_h, _, nref_h, _, _, _, _, _ = get_histories(
        m2, level=0, verbose=True)

    # Upload to JAX device once
    s_h    = jnp.array(s_h,    dtype=jnp.float32)  # (NLE, NStokes)
    d_h    = jnp.array(d_h,    dtype=jnp.float32)  # (NLE, NL)
    w_h    = jnp.array(w_h,    dtype=jnp.float32)  # (NLE, NLR)
    nref_h = jnp.array(nref_h, dtype=jnp.float32)  # (NLE,)

    # Shared arrays
    kabs_j   = jnp.array(sigma_valid,          dtype=jnp.float32)  # (n_wl, NL)
    # (n_wl,)
    wavelength_hr_j  = jnp.array(w_valid,              dtype=jnp.float32)
    # (n_wl,) — no surface
    alb_hr_j = jnp.zeros(len(w_valid),         dtype=jnp.float32)
    wavelength_lr_j  = jnp.array(wavelength_lr, dtype=jnp.float32)  # (NLR,)

    def _si_one(sik, wi_lr, dij, ki, wavelength_i, kabs_i, alb_i):
        wi = jnp.interp(wavelength_i, wavelength_lr_j, wi_lr)
        return sik * wi * jnp.exp(-jnp.sum(dij * kabs_i)) * alb_i**ki

    _si_photons = vmap(_si_one, in_axes=(0, 0, 0, 0, None, None, None))
    _si_stokes  = vmap(_si_photons, in_axes=(1, None, None, None, None, None,
                                             None))

    _n_h     = float(n_h)
    n_wl      = int(kabs_j.shape[0])
    n_stokes = int(s_h.shape[1])

    def _body_fw(i, carry):
        s_sum, s2_sum = carry
        si = _si_stokes(s_h, w_h, d_h, nref_h, wavelength_hr_j[i], kabs_j[i],
                        alb_hr_j[i])
        return (s_sum.at[i].set(si.sum(axis=1)),
                s2_sum.at[i].set((si**2).sum(axis=1)))

    _zeros = (jnp.zeros((n_wl, n_stokes)),
              jnp.zeros((n_wl, n_stokes)))
    stokes_j, stokes2_j = jit(
        lambda: lax.fori_loop(0, n_wl, _body_fw, _zeros))()
    stokes  = np.array(stokes_j).T / _n_h   # (NStokes, n_wl)
    stokes2 = np.array(stokes2_j).T / _n_h  # (NStokes, n_wl)
    std     = np.sqrt((stokes2 - stokes**2) / _n_h)
    upper   = stokes + 1.96 * std
    lower   = stokes - 1.96 * std

    I  = stokes[0]
    Q  = stokes[1]

#####################
# Comparison plots: DA reference vs ALIS (no hist) vs ALIS (hist + JAX)
i_valid = data_valid[:, 1]
q_valid = data_valid[:, 2]
i_alis  = m1['I_up (TOA)'].data.flatten()
q_alis  = m1['Q_up (TOA)'].data.flatten()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# ── Top: absolute reflectance
# ──────────────────────────────────────────────
ax1.plot(w_valid, i_valid, 'r-', lw=1.2, label='Doubling Adding (32 streams)')
ax1.plot(w_valid, i_alis,     'c-',  lw=1.2, label='SMART-G ALIS (no hist)')
ax1.plot(w_valid, I,          'b-',  lw=1.2, label='SMART-G ALIS (hist + JAX)')
ax1.fill_between(w_valid, lower[0], upper[0], facecolor='b', alpha=0.2,
                 label='95 % CI (hist)')
ax1.set_ylabel('I_up (TOA)')
ax1.set_title('Validation: DA vs ALIS no-hist vs ALIS hist')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 0.02)

# ── Bottom: relative difference vs DA reference
# ───────────────────────────
diff_alis = (i_alis - i_valid) / i_valid * 100
diff_hist = (I      - i_valid) / i_valid * 100
diff_hist_upper = (upper[0] - i_valid) / i_valid * 100
diff_hist_lower = (lower[0] - i_valid) / i_valid * 100

ax2.plot(w_valid, diff_hist, 'b-', lw=1.2, label='ALIS hist − DA (%)')
ax2.fill_between(w_valid, diff_hist_lower, diff_hist_upper, facecolor='b',
                 alpha=0.2, label='95 % CI')
ax2.plot(w_valid, diff_alis, 'c-', lw=1.2, label='ALIS no-hist − DA (%)')
ax2.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Relative difference (%)')
ax2.set_title('Relative difference with respect to DA reference')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-1, 1)

plt.tight_layout()

# ── Polarized reflectance: Q comparison
# ────────────────────────────────────
# dolp = |Q|/I (U ≈ 0 for this geometry)
dolp_valid = np.abs(q_valid) / i_valid
dolp_alis  = np.abs(q_alis) / i_alis
dolp_hist  = np.abs(Q) / I

fig2, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# ── Panel 1: Q absolute
# ────────────────────────────────────────────────────
axes[0].plot(w_valid, q_valid, 'r-', lw=1.2,
             label='Doubling Adding (32 streams)')
axes[0].plot(w_valid, q_alis,  'c-',  lw=1.2, label='SMART-G ALIS (no hist)')
axes[0].plot(w_valid, Q, 'b-', lw=1.2, label='SMART-G ALIS (hist + JAX)')
axes[0].fill_between(w_valid, lower[1], upper[1], facecolor='b', alpha=0.2,
                     label='95 % CI (hist)')
axes[0].set_ylabel('Q_up (TOA)')
axes[0].set_title('Polarized reflectance: Stokes Q comparison')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# ── Panel 2: Q relative difference
# ─────────────────────────────────────────
diff_q_alis = (q_alis - q_valid) / np.abs(q_valid) * 100
diff_q_hist = (Q      - q_valid) / np.abs(q_valid) * 100
diff_q_hist_upper = (upper[1] - q_valid) / np.abs(q_valid) * 100
diff_q_hist_lower = (lower[1] - q_valid) / np.abs(q_valid) * 100

axes[1].plot(w_valid, diff_q_hist, 'b-', lw=1.2, label='ALIS hist − DA (%)')
axes[1].fill_between(w_valid, diff_q_hist_lower, diff_q_hist_upper,
                     facecolor='b', alpha=0.2, label='95 % CI')
axes[1].plot(w_valid, diff_q_alis, 'c-', lw=1.2, label='ALIS no-hist − DA (%)')
axes[1].axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
axes[1].set_ylabel('Relative difference (%)')
axes[1].set_title('Q relative difference with respect to DA reference')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-1, 1)

# ── Panel 3: dolp comparison
# ───────────────────────────────────────────────
axes[2].plot(w_valid, dolp_valid, 'r-', lw=1.2,
             label='Doubling Adding (32 streams)')
axes[2].plot(w_valid, dolp_alis,  'c-', lw=1.2, label='SMART-G ALIS (no hist)')
axes[2].plot(w_valid, dolp_hist, 'b-', lw=1.2,
             label='SMART-G ALIS (hist + JAX)')
axes[2].set_xlabel('Wavelength (nm)')
axes[2].set_ylabel('DoLP = |Q| / I')
axes[2].set_title('Degree of Linear Polarization comparison')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
