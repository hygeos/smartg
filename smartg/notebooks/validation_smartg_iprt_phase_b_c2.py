# %% [markdown]
# # SMART-G validation: IPRT phase B, cubic cloud (C2)
#
# This notebook runs the cubic cloud case C2 of the phase B of IPRT (the
# International Polarized Radiative Transfer model intercomparison) with
# the 3D mode of SMART-G, and compares the results with MYSTIC.
#
# - IPRT: https://www.meteo.physik.uni-muenchen.de/~iprt/doku.php?id=start
# - Phase B paper: Emde et al. (2018), *IPRT polarized radiative transfer
#   model intercomparison project - Three-dimensional test cases (phase
#   B)*, JQSRT.
#
# ## The case
#
# | | |
# |---|---|
# | domain | 7 x 7 x 5 km, periodic along x and y |
# | cloud | a 1 km cube between 3 and 4 km along x and y, 2 and 3 km along z |
# | cloud optics | extinction 10 /km, effective radius 10 µm, single scattering albedo 1, Mie phase matrix at 800 nm |
# | surface | Lambertian, albedo 0.2 |
# | sensors | 70 x 70, one per 100 m cell |
# | atmosphere | none, then a homogeneous Rayleigh layer of optical depth 0.5, without depolarization |
#
# The 9 viewing geometries, in degrees, the solar azimuth angle `phi_0`
# being 180 everywhere:
#
# | case | sensors at the | theta | phi | theta_0 |
# |---|---|---|---|---|
# | 1 | bottom (0 km) | 40 | 0 | 20 |
# | 2 | bottom (0 km) | 40 | 60 | 20 |
# | 3 | bottom (0 km) | 40 | 120 | 20 |
# | 4 | bottom (0 km) | 40 | 180 | 20 |
# | 5 | top (5 km) | 180 | 0 | 40 |
# | 6 | top (5 km) | 140 | 0 | 40 |
# | 7 | top (5 km) | 140 | 60 | 40 |
# | 8 | top (5 km) | 140 | 120 | 40 |
# | 9 | top (5 km) | 140 | 180 | 40 |
#
# The cases 1 to 4 give the radiance transmitted below the cloud, the
# cases 5 to 9 the radiance reflected at the top of the domain. The
# reference file `smartg.iprt.phase_b.MYSTIC_RES_C2` holds the 9 cases
# without atmosphere, then the 9 cases with atmosphere
# (`ATM_CASE_OFFSET` later).
#
# Each case is run twice:
#
# - in **backward** mode, one run per case: the photons leave the
#   sensors in their viewing direction, and the local estimate is taken
#   towards the sun;
# - in **forward** mode, one run per group of cases sharing the sun
#   position (`FORWARD_GROUPS`): the photons leave the sensors towards
#   the sun, and the viewing directions of the cases are zipped in the
#   local estimate.
#
# To follow the IPRT convention, SMART-G's V is multiplied by -1 in
# backward mode and U by -1 in forward mode.
#
# ## The comparison
#
# Each case gives two figures, the SMART-G I, Q, U and V maps and their
# differences with MYSTIC, and the delta_m of each Stokes parameter:
#
# delta_m = 100 * sqrt(sum((SMART-G - MYSTIC)^2)) / sqrt(sum(MYSTIC^2))
#
# in percent, over the 4900 sensors. Monte Carlo noise dominates the
# delta_m of the small components: IPRT reports values of several
# hundred percent on V for every model. The last section gathers the
# delta_m of all the cases.
#
# ## Running it
#
# A CUDA GPU and the IPRT data of the auxdata (`IPRT/phaseB`) are
# needed. With the benchmark photon counts of the settings below, the
# whole notebook is long to run: lower `N_PHOTONS` for a quick look. The
# same cases, with fewer photons, are the non-regression tests
# `smartg/tests/test_iprt_phase_b_c2.py`.
#
# Names used in the notebook:
#
# | name | meaning |
# |---|---|
# | `atm_noatm`, `atm_rayleigh` | the atmospheres without and with the Rayleigh layer |
# | `sensor_grid` | the 70 x 70 sensor grid |
# | `ds`, `norm` | the output of the last run, and the factor cos(theta_0) / pi applied to its maps |
# | `runs` | every `(ds, norm)`, by (atmosphere, mode, case or cases) |
# | `delta_m_all` | the delta_m of I, Q, U and V, by (atmosphere, mode, case) |

# %%
# %matplotlib inline
# Reload the modules changed externally
# %load_ext autoreload
# %autoreload 2

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from smartg.iprt.phase_b import (
    ATM_CASE_OFFSET,
    FORWARD_GROUPS,
    MYSTIC_RES_C2,
    ForwardGroup,
    PhaseBAtmosphere,
    backward_run_kwargs,
    build_atm_c2,
    compare_case,
    find_optimal_xb_xg,
    forward_run_kwargs,
    run_case_backward,
    run_group_forward,
    sensor_grid_c2,
)
from smartg.smartg import Smartg

# %% [markdown]
# ## Settings
#
# The photon counts are those of the IPRT benchmark. The seed is fixed,
# so that a rerun gives the same figures; -1 draws a new noise
# realisation at every run. The CUDA block and grid sizes change the
# noise realisation too: `FIND_OPTIMAL_XB_XG` times the candidates
# before every run and keeps the fastest pair, instead of `X_BLOCK` and
# `X_GRID`.

# %% tags=["parameters"]
N_PHOTONS = 49e9
# The backward cases 5 to 9 with atmosphere are the slowest ones
N_PHOTONS_ATM_TOP = 1e9
N_LOOP = 1e8
N_THETA = 18001  # 1801 scattering angles are not enough for the case 6
SEED = 1234
TAU_RAYLEIGH = 0.5
DEPO_ATM = 0.0

# Accepted by most GPUs after the 10xx series
X_BLOCK = 128
X_GRID = 1024

FIND_OPTIMAL_XB_XG = False
X_BLOCKS = [32, 64, 128]
X_GRIDS = [512, 1024]
CHECK_N_PHOTONS = 1e8
CHECK_N_LOOP = 1e8

# %%
s_3db = Smartg(opt3d=True, alt_pp=True, alis=False, back=True,
               double=True, bias=True)
s_3df = Smartg(opt3d=True, alt_pp=True, alis=False, back=False,
               double=True, bias=True)

sensor_grid = sensor_grid_c2()
runs: dict[tuple, tuple[xr.Dataset, float]] = {}
delta_m_all: dict[tuple[str, str, int], np.ndarray] = {}


# %% [markdown]
# ## Helper functions
#
# `run_backward` and `run_forward` run a case, or a group of cases, with
# the settings above, and `compare` and `compare_group` draw the figures
# and the delta_m with `smartg.iprt.phase_b.compare_case`.

# %%
def atm_name(with_atm: bool) -> str:
    """Return the label of an atmosphere, for titles and keys."""
    return "with atm" if with_atm else "without atm"


def run_options(sg: Smartg, geometry: dict[str, Any], with_atm: bool
                ) -> dict[str, Any]:
    """Return the Smartg.run options shared by every run.

    Parameters
    ----------
    sg : Smartg
        The compiled SMART-G.
    geometry : dict
        The run arguments of the case, from backward_run_kwargs or
        forward_run_kwargs, to time the CUDA sizes.
    with_atm : bool
        Whether the atmosphere has the Rayleigh layer, which is then
        run without depolarization.

    Returns
    -------
    dict
        The n_icdf, n_loop, seed, xblock, xgrid and depo arguments.
    """
    options: dict[str, Any] = {"n_icdf": N_THETA}
    if with_atm:
        options["depo"] = DEPO_ATM
    if FIND_OPTIMAL_XB_XG:
        xblock, xgrid = find_optimal_xb_xg(
            sg, X_BLOCKS, X_GRIDS, CHECK_N_PHOTONS, CHECK_N_LOOP,
            **geometry, **options,
        )
    else:
        xblock, xgrid = X_BLOCK, X_GRID
    return {**options, "n_loop": N_LOOP, "seed": SEED, "xblock": xblock,
            "xgrid": xgrid}


def run_backward(atm: PhaseBAtmosphere, case: int, with_atm: bool,
                 n_photons: float = N_PHOTONS) -> tuple[xr.Dataset, float]:
    """Run one case in backward mode and store it in runs.

    Parameters
    ----------
    atm : PhaseBAtmosphere
        The atmosphere.
    case : int
        The case number.
    with_atm : bool
        Whether atm has the Rayleigh layer.
    n_photons : float
        Number of photons.

    Returns
    -------
    ds : xr.Dataset
        The output of the run.
    norm : float
        The normalisation of the maps.
    """
    options = run_options(
        s_3db, backward_run_kwargs(atm, sensor_grid, case), with_atm
    )
    ds, norm = run_case_backward(s_3db, atm, sensor_grid, case, n_photons,
                                 **options)
    print(f"kernel time (s) = {float(ds.attrs['kernel time (s)']):.2f}")
    runs[atm_name(with_atm), "backward", case] = ds, norm
    return ds, norm


def run_forward(atm: PhaseBAtmosphere, group: ForwardGroup, with_atm: bool,
                n_photons: float = N_PHOTONS) -> tuple[xr.Dataset, float]:
    """Run a group of cases in forward mode and store it in runs.

    Parameters
    ----------
    atm : PhaseBAtmosphere
        The atmosphere.
    group : ForwardGroup
        The group of cases, a value of FORWARD_GROUPS.
    with_atm : bool
        Whether atm has the Rayleigh layer.
    n_photons : float
        Number of photons.

    Returns
    -------
    ds : xr.Dataset
        The output of the run, for all the cases of the group.
    norm : float
        The normalisation of the maps.
    """
    options = run_options(
        s_3df, forward_run_kwargs(atm, sensor_grid, group), with_atm
    )
    ds, norm = run_group_forward(s_3df, atm, sensor_grid, group, n_photons,
                                 **options)
    print(f"kernel time (s) = {float(ds.attrs['kernel time (s)']):.2f}")
    runs[atm_name(with_atm), "forward", group.cases] = ds, norm
    return ds, norm


def compare(ds: xr.Dataset, norm: float, case: int, with_atm: bool,
            group: ForwardGroup | None = None) -> None:
    """Compare a case with MYSTIC, and store its delta_m.

    Parameters
    ----------
    ds : xr.Dataset
        The output of the run.
    norm : float
        The normalisation of the maps.
    case : int
        The case number.
    with_atm : bool
        Whether the run has the Rayleigh layer.
    group : ForwardGroup, optional
        The group of the forward run holding the case. By default the
        run is a backward one.
    """
    mode = "backward" if group is None else "forward"
    print(f"Case {case} - {mode} - {atm_name(with_atm)}")
    forward: dict[str, Any] = {}
    if group is not None:
        forward = {"level": group.level,
                   "direction": group.cases.index(case),
                   "u_sign": -1.0, "v_sign": 1.0}
    delta_m_all[atm_name(with_atm), mode, case] = compare_case(
        ds, norm, case, sensor_grid.xgrid, sensor_grid.ygrid,
        MYSTIC_RES_C2,
        ref_case=case + ATM_CASE_OFFSET if with_atm else case,
        title_suffix=f"{atm_name(with_atm)} - {mode}",
        i_vmin=0.0 if with_atm else None,
        v_diff_frac=0.05 if with_atm else 0.015,
        **forward,
    )


def compare_group(ds: xr.Dataset, norm: float, group: ForwardGroup,
                  with_atm: bool) -> None:
    """Compare every case of a forward run with MYSTIC.

    Parameters
    ----------
    ds : xr.Dataset
        The output of the forward run.
    norm : float
        The normalisation of the maps.
    group : ForwardGroup
        The group of cases of the run.
    with_atm : bool
        Whether the run has the Rayleigh layer.
    """
    for case in group.cases:
        compare(ds, norm, case, with_atm, group)


# %% [markdown]
# ## Without atmosphere
#
# The cloud alone, above the surface.

# %%
atm_noatm = build_atm_c2(n_theta=N_THETA)

# %% [markdown]
# ### Backward simulations

# %% [markdown]
# #### Case 1
#
# Sensors at the bottom, theta = 40, phi = 0,
# theta_0 = 20.

# %%
ds, norm = run_backward(atm_noatm, 1, with_atm=False)
compare(ds, norm, 1, with_atm=False)

# %% [markdown]
# #### Case 2
#
# Sensors at the bottom, theta = 40, phi = 60,
# theta_0 = 20.

# %%
ds, norm = run_backward(atm_noatm, 2, with_atm=False)
compare(ds, norm, 2, with_atm=False)

# %% [markdown]
# #### Case 3
#
# Sensors at the bottom, theta = 40, phi = 120,
# theta_0 = 20.

# %%
ds, norm = run_backward(atm_noatm, 3, with_atm=False)
compare(ds, norm, 3, with_atm=False)

# %% [markdown]
# #### Case 4
#
# Sensors at the bottom, theta = 40, phi = 180,
# theta_0 = 20.

# %%
ds, norm = run_backward(atm_noatm, 4, with_atm=False)
compare(ds, norm, 4, with_atm=False)

# %% [markdown]
# #### Case 5
#
# Sensors at the top, theta = 180, phi = 0,
# theta_0 = 40.

# %%
ds, norm = run_backward(atm_noatm, 5, with_atm=False)
compare(ds, norm, 5, with_atm=False)

# %% [markdown]
# #### Case 6
#
# Sensors at the top, theta = 140, phi = 0,
# theta_0 = 40.

# %%
ds, norm = run_backward(atm_noatm, 6, with_atm=False)
compare(ds, norm, 6, with_atm=False)

# %% [markdown]
# #### Case 7
#
# Sensors at the top, theta = 140, phi = 60,
# theta_0 = 40.

# %%
ds, norm = run_backward(atm_noatm, 7, with_atm=False)
compare(ds, norm, 7, with_atm=False)

# %% [markdown]
# #### Case 8
#
# Sensors at the top, theta = 140, phi = 120,
# theta_0 = 40.

# %%
ds, norm = run_backward(atm_noatm, 8, with_atm=False)
compare(ds, norm, 8, with_atm=False)

# %% [markdown]
# #### Case 9
#
# Sensors at the top, theta = 140, phi = 180,
# theta_0 = 40.

# %%
ds, norm = run_backward(atm_noatm, 9, with_atm=False)
compare(ds, norm, 9, with_atm=False)

# %% [markdown]
# ### Forward simulations

# %% [markdown]
# #### Cases 1 to 4

# %%
group = FORWARD_GROUPS[1]
ds, norm = run_forward(atm_noatm, group, with_atm=False)
compare_group(ds, norm, group, with_atm=False)

# %% [markdown]
# #### Cases 5 to 9

# %%
group = FORWARD_GROUPS[2]
ds, norm = run_forward(atm_noatm, group, with_atm=False)
compare_group(ds, norm, group, with_atm=False)

# %% [markdown]
# ## With atmosphere
#
# A homogeneous Rayleigh layer of optical depth `TAU_RAYLEIGH`, without
# absorption, fills the domain.

# %%
atm_rayleigh = build_atm_c2(tau_ray=TAU_RAYLEIGH, n_theta=N_THETA)

# %% [markdown]
# ### Backward simulations

# %% [markdown]
# #### Case 1
#
# Sensors at the bottom, theta = 40, phi = 0,
# theta_0 = 20.

# %%
ds, norm = run_backward(atm_rayleigh, 1, with_atm=True)
compare(ds, norm, 1, with_atm=True)

# %% [markdown]
# #### Case 2
#
# Sensors at the bottom, theta = 40, phi = 60,
# theta_0 = 20.

# %%
ds, norm = run_backward(atm_rayleigh, 2, with_atm=True)
compare(ds, norm, 2, with_atm=True)

# %% [markdown]
# #### Case 3
#
# Sensors at the bottom, theta = 40, phi = 120,
# theta_0 = 20.

# %%
ds, norm = run_backward(atm_rayleigh, 3, with_atm=True)
compare(ds, norm, 3, with_atm=True)

# %% [markdown]
# #### Case 4
#
# Sensors at the bottom, theta = 40, phi = 180,
# theta_0 = 20.

# %%
ds, norm = run_backward(atm_rayleigh, 4, with_atm=True)
compare(ds, norm, 4, with_atm=True)

# %% [markdown]
# #### Case 5
#
# Sensors at the top, theta = 180, phi = 0,
# theta_0 = 40.

# %%
ds, norm = run_backward(atm_rayleigh, 5, with_atm=True,
                        n_photons=N_PHOTONS_ATM_TOP)
compare(ds, norm, 5, with_atm=True)

# %% [markdown]
# #### Case 6
#
# Sensors at the top, theta = 140, phi = 0,
# theta_0 = 40.

# %%
ds, norm = run_backward(atm_rayleigh, 6, with_atm=True,
                        n_photons=N_PHOTONS_ATM_TOP)
compare(ds, norm, 6, with_atm=True)

# %% [markdown]
# #### Case 7
#
# Sensors at the top, theta = 140, phi = 60,
# theta_0 = 40.

# %%
ds, norm = run_backward(atm_rayleigh, 7, with_atm=True,
                        n_photons=N_PHOTONS_ATM_TOP)
compare(ds, norm, 7, with_atm=True)

# %% [markdown]
# #### Case 8
#
# Sensors at the top, theta = 140, phi = 120,
# theta_0 = 40.

# %%
ds, norm = run_backward(atm_rayleigh, 8, with_atm=True,
                        n_photons=N_PHOTONS_ATM_TOP)
compare(ds, norm, 8, with_atm=True)

# %% [markdown]
# #### Case 9
#
# Sensors at the top, theta = 140, phi = 180,
# theta_0 = 40.

# %%
ds, norm = run_backward(atm_rayleigh, 9, with_atm=True,
                        n_photons=N_PHOTONS_ATM_TOP)
compare(ds, norm, 9, with_atm=True)

# %% [markdown]
# ### Forward simulations

# %% [markdown]
# #### Cases 1 to 4

# %%
group = FORWARD_GROUPS[1]
ds, norm = run_forward(atm_rayleigh, group, with_atm=True)
compare_group(ds, norm, group, with_atm=True)

# %% [markdown]
# #### Cases 5 to 9

# %%
group = FORWARD_GROUPS[2]
ds, norm = run_forward(atm_rayleigh, group, with_atm=True)
compare_group(ds, norm, group, with_atm=True)

# %% [markdown]
# ## Summary
#
# The delta_m, in percent, of every case run above.

# %%
summary = pd.DataFrame(delta_m_all, index=["I", "Q", "U", "V"]).T
summary.index.names = ["atmosphere", "mode", "case"]
summary.round(3)
