# %% [markdown]
# # SMART-G validation IPRT phase 3
# - https://www.meteo.physik.uni-muenchen.de/~iprt/doku.php?id=start
#
# The cases are run by `smartg/iprt/phase3.py`. Each `case_*` function
# computes the BOA and TOA radiances, stores them in intermediate files
# (`*_boa.nc`, `*_toa.nc`) and converts them to the IPRT phase 3 output
# format, `iprt_phase3_<case>.nc`, in `OUTPUT_DIR`. This notebook runs
# the cases and draws the Stokes parameters of those files.
#
# `N_PHOTONS` is the number of photons per viewing direction: 1e5 is
# enough to check that everything runs, the reference results use 1e8.
# `N_LOOP` is the number of photons per kernel launch, `N_PHOTONS` when
# it is `None`: one launch per viewing direction, the spread of the
# launches giving the standard deviation. `XBLOCK` and `XGRID`, the
# threads per block and the blocks of a launch, set the speed of the
# runs, not their expected values.
# With `OVERWRITE = False` the files already in `OUTPUT_DIR` are reused
# and no simulation is run again.

# %%
# %matplotlib inline
# Reload the modules changed externally
# %reload_ext autoreload
# %autoreload 2

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

# importing the module compiles the SMART-G kernels of the runs
from smartg.iprt import phase3 as p3
from smartg.iprt.phase3 import plot_camera_iprt, plot_polar_iprt

# %%
N_PHOTONS = 1e5  # photons per viewing direction, 1e8 for the reference results
N_LOOP = None  # photons per kernel launch, None for N_PHOTONS
XBLOCK = 64  # threads per block of a kernel launch
XGRID = 1024  # blocks of a kernel launch
OVERWRITE = True  # False: reuse the files already in OUTPUT_DIR
OUTPUT_DIR = Path("./res_iprt_phase3")
MOD_NAME = "SMART-G"
DEPOL = 0.03  # depolarisation factor of every case


# %%
def open_case(case_name: str,
              output_dir: str | Path | None = None) -> xr.Dataset:
    """Open the IPRT phase 3 output file of a case.

    Parameters
    ----------
    case_name : str
        Name of the case, e.g. 'd1' or 'e6_v2', as it appears in the
        file name iprt_phase3_<case_name>.nc.
    output_dir : str or Path, optional
        Folder of the file. By default OUTPUT_DIR.

    Returns
    -------
    Dataset
        The content of the file, loaded in memory.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    path = Path(output_dir) / f"iprt_phase3_{case_name}.nc"
    return xr.open_dataset(path).load()


def open_comparison(case_name: str, output_dir: str | Path | None = None,
                    case_ref: str | None = None,
                    output_dir_ref: str | Path | None = None
                    ) -> tuple[xr.Dataset, xr.Dataset | None, str]:
    """Open a case and the reference case it may be compared to.

    Parameters
    ----------
    case_name : str
        Name of the case, e.g. 'd1' or 'e6_v2'.
    output_dir : str or Path, optional
        Folder of the case_name results. By default OUTPUT_DIR.
    case_ref : str, optional
        Name of the reference case. By default there is none.
    output_dir_ref : str or Path, optional
        Folder of the case_ref results. By default output_dir.

    Returns
    -------
    ds : Dataset
        The results of case_name.
    ds_ref : Dataset or None
        The results of case_ref, or None without case_ref.
    label : str
        What is drawn, for the figure titles: case_name, or
        case_name-case_ref followed by the two folder names when they
        differ.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if output_dir_ref is None:
        output_dir_ref = output_dir
    ds = open_case(case_name, output_dir)
    if case_ref is None:
        return ds, None, case_name
    ds_ref = open_case(case_ref, output_dir_ref)
    label = f"{case_name}-{case_ref}"
    if Path(output_dir).resolve() != Path(output_dir_ref).resolve():
        label += (f" ({Path(output_dir).name}"
                  f" - {Path(output_dir_ref).name})")
    return ds, ds_ref, label


def plot_case_polar(case_name: str, iz: int,
                    output_dir: str | Path | None = None,
                    case_ref: str | None = None,
                    output_dir_ref: str | Path | None = None,
                    norm: float = 1. / np.pi) -> None:
    """Draw the Stokes parameters of a D or E case in polar view.

    One figure is drawn per sun zenith angle with plot_polar_iprt, the
    viewing azimuth columns at 360 - vaa. With case_ref, the difference
    case_name - case_ref is drawn instead, on a symmetric I colour
    scale.

    Parameters
    ----------
    case_name : str
        Name of the case, e.g. 'd1' or 'd6_pp'.
    iz : int
        Index of the output level: 0 for BOA, 1 for TOA.
    output_dir : str or Path, optional
        Folder of the case_name results. By default OUTPUT_DIR.
    case_ref : str, optional
        Name of the reference case, with the same angles as case_name.
        Give case_ref=case_name to compare one case between two
        folders. By default no difference is drawn.
    output_dir_ref : str or Path, optional
        Folder of the case_ref results. By default output_dir.
    norm : float
        Factor applied to the radiances.
    """
    ds, ds_ref, label = open_comparison(case_name, output_dir, case_ref,
                                        output_dir_ref)
    vza = ds[f"{case_name}_vza"].values
    vaa = ds[f"{case_name}_vaa"].values
    sza = ds[f"{case_name}_sza"].values
    zout = ds[f"{case_name}_zout"].values[iz]
    rad = ds[f"radiance_{case_name}"]
    for isza in range(len(sza)):
        stokes = [rad[iz, isza, 0, :, :, k].values * norm for k in range(4)]
        scales: dict[str, Any] = {}
        if ds_ref is not None:
            rad_ref = ds_ref[f"radiance_{case_ref}"]
            stokes = [s - rad_ref[iz, isza, 0, :, :, k].values * norm
                      for k, s in enumerate(stokes)]
            max_i = np.max(np.abs(stokes[0]))
            scales = {"min_i": -max_i, "max_i": max_i, "cmap_i": "RdBu_r"}
        title = (f"IPRT case {label} - depol = {DEPOL} - "
                 f"SZA = {sza[isza]:.0f} - SAA = 0 - {zout:.0f}km - "
                 f"{MOD_NAME}")
        plot_polar_iprt(*stokes, thetas=vza, phis=vaa[::-1] + 180.,
                        title=title, **scales)


def plot_case_camera(case_name: str,
                     output_dir: str | Path | None = None,
                     case_ref: str | None = None,
                     output_dir_ref: str | Path | None = None,
                     norm: float = 1. / np.pi) -> None:
    """Draw the Stokes parameters of an E6 case in camera view.

    One figure is drawn per sun position, stored on the lon axis of the
    file, with plot_camera_iprt. With case_ref, the difference
    case_name - case_ref is drawn instead, on a symmetric I colour
    scale.

    Parameters
    ----------
    case_name : str
        Name of the case, e.g. 'e6_v1'.
    output_dir : str or Path, optional
        Folder of the case_name results. By default OUTPUT_DIR.
    case_ref : str, optional
        Name of the reference case, with the same pixels as case_name.
        Give case_ref=case_name to compare one case between two
        folders. By default no difference is drawn.
    output_dir_ref : str or Path, optional
        Folder of the case_ref results. By default output_dir.
    norm : float
        Factor applied to the radiances.
    """
    ds, ds_ref, label = open_comparison(case_name, output_dir, case_ref,
                                        output_dir_ref)
    lon = ds[f"{case_name}_lon"].values
    zout = ds[f"{case_name}_zout"].values[0]
    rad = ds[f"radiance_{case_name}"]
    for ilon in range(len(lon)):
        stokes = [rad[0, 0, ilon, :, :, k].values * norm for k in range(4)]
        scales: dict[str, Any] = {}
        if ds_ref is not None:
            rad_ref = ds_ref[f"radiance_{case_ref}"]
            stokes = [s - rad_ref[0, 0, ilon, :, :, k].values * norm
                      for k, s in enumerate(stokes)]
            max_i = np.max(np.abs(stokes[0]))
            scales = {"i_min": -max_i, "i_max": max_i,
                      "i_cmap": "coolwarm"}
        title = (f"IPRT case {label} - depol = {DEPOL} - lat = 0 - "
                 f"lon = {lon[ilon]:.0f} - z = {zout:.0f}km - {MOD_NAME}")
        plot_camera_iprt(stokes[0], stokes[1], stokes[2], stokes[3],
                         title=title, **scales)


# %% [markdown]
# ## D - Test cases for fully spherical geometry with one layer

# %% [markdown]
# ### Case D1
# Rayleigh layer, optical thickness 0.5 at 550 nm, black surface.

# %%
p3.case_d1(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
           n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("d1", iz=0)  # BOA

# %%
plot_case_polar("d1", iz=1)  # TOA

# %% [markdown]
# ### Case D2
# Rayleigh layer, optical thickness 0.1 at 550 nm, over a Lambertian
# surface of albedo 0.3.

# %%
p3.case_d2(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
           n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("d2", iz=0)  # BOA

# %%
plot_case_polar("d2", iz=1)  # TOA

# %% [markdown]
# ### Case D3
# Aerosol layer (`waso.mie.cdf`), optical thickness 0.2 at 350 nm,
# single scattering albedo 0.975683, phase matrix given through
# `prof_phases`. The phase matrix is kept on the 68 angles of the file.

# %%
p3.case_d3(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
           n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("d3", iz=0)  # BOA

# %%
plot_case_polar("d3", iz=1)  # TOA

# %% [markdown]
# ### Case D4
# Spheroidal aerosol layer (`sizedistr_spheroid.cdf`), optical thickness
# 0.2 at 350 nm, single scattering albedo 0.787581, phase matrix given
# through `prof_phases`. The phase matrix is kept on the 1801 angles of
# the file.

# %%
p3.case_d4(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
           n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("d4", iz=0)  # BOA

# %%
plot_case_polar("d4", iz=1)  # TOA

# %% [markdown]
# #### Variant D4_bis
# The same aerosol, built with `AerOPAC` from the file converted by
# `aer2smartg`.

# %%
p3.case_d4_bis(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
               n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("d4_bis", iz=0)  # BOA

# %%
plot_case_polar("d4_bis", iz=1)  # TOA

# %% [markdown]
# ### Case D5
# Water cloud layer (`watercloud.mie.cdf`), optical thickness 5 at 800
# nm, single scattering albedo 0.999979. The phase matrix is kept on the
# 450 angles of the file.

# %%
p3.case_d5(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
           n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("d5", iz=0)  # BOA

# %%
plot_case_polar("d5", iz=1)  # TOA

# %% [markdown]
# ### Case D6
# Rayleigh layer, optical thickness 0.1 at 550 nm, over a rough ocean
# surface (wind speed 2 m/s).

# %%
p3.case_d6(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
           n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("d6", iz=0)  # BOA

# %%
plot_case_polar("d6", iz=1)  # TOA

# %% [markdown]
# #### Variant D6_pp
# The same case in plane-parallel geometry, for sun zenith angles up to
# 87° and viewing zenith angles up to 89°.

# %%
p3.case_d6_pp(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
              n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("d6_pp", iz=0)  # BOA

# %%
plot_case_polar("d6_pp", iz=1)  # TOA

# %% [markdown]
# ## E - Test cases for fully spherical geometry for a vertically inhomogeneous atmosphere

# %% [markdown]
# ### Case E1
# US standard Rayleigh profile at 450 nm.

# %%
p3.case_e1(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
           n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("e1", iz=0)  # BOA

# %%
plot_case_polar("e1", iz=1)  # TOA

# %% [markdown]
# ### Case E2
# US standard Rayleigh and absorption profiles at 320 nm.

# %%
p3.case_e2(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
           n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("e2", iz=0)  # BOA

# %%
plot_case_polar("e2", iz=1)  # TOA

# %% [markdown]
# ### Case E3
# US standard profiles at 450 nm with a desert aerosol (`desert.cdf`) of
# optical thickness 0.5 between 0 and 3 km. The phase matrix is kept on
# the 361 angles of the file.

# %%
p3.case_e3(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
           n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("e3", iz=0)  # BOA

# %%
plot_case_polar("e3", iz=1)  # TOA

# %% [markdown]
# ### Case E4
# Case E3 with, in addition, a sulfate aerosol (`sulfate.cdf`) of
# optical thickness 0.05 between 20 and 21 km. Both phase matrices are
# kept on the 361 angles of their files.

# %%
p3.case_e4(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
           n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("e4", iz=0)  # BOA

# %%
plot_case_polar("e4", iz=1)  # TOA

# %% [markdown]
# ### Case E5
# US standard profiles at 450 nm with an ice cloud (`ic.ghm.baum.cdf`),
# effective radius 50 µm, optical thickness 1, between 10 and 11 km. The
# phase matrix is kept on the 498 angles of the file, 0.01° apart in the
# forward peak.

# %%
p3.case_e5(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
           n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_polar("e5", iz=0)  # BOA

# %%
plot_case_polar("e5", iz=1)  # TOA

# %% [markdown]
# ### Case E6
# Camera at 300 000 km with 61 x 61 pixels covering ±1.2°, looking at
# the Earth: US standard profiles at 450 nm over a rough ocean surface
# (wind speed 5 m/s). The four sun positions are stored on the lon axis.

# %% [markdown]
# #### Variant E6_v1
# Version 1: one direction per pixel centre, the sensors being placed
# where it enters the atmosphere; the pixels missing the atmosphere stay
# at 0.

# %%
p3.case_e6_v1(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
              n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_camera("e6_v1")

# %% [markdown]
# #### Variant E6_v2
# Version 2: sensors at 300 000 km with a field of view of 0.04°
# (`obj3d` kernel).

# %%
p3.case_e6_v2(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
              n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_camera("e6_v2")

# %% [markdown]
# #### Variant E6_v3
# Version 3: the same sensors, with the atmosphere grid extended up to
# 300 000 km.

# %%
p3.case_e6_v3(n_photons=N_PHOTONS, overwrite=OVERWRITE, output_dir=OUTPUT_DIR,
              n_loop=N_LOOP, xblock=XBLOCK, xgrid=XGRID)

# %%
plot_case_camera("e6_v3")

# %% [markdown]
# ## Comparing two results
# `plot_case_polar` and `plot_case_camera` draw the difference
# `case_name - case_ref` when given `case_ref`. `case_name` is read in
# `output_dir` and `case_ref` in `output_dir_ref`, so the two results
# can come from two different folders, e.g. two runs with different
# numbers of photons. Give the same name to both to compare one case
# between them.

# %%
COMPARE_DIR = OUTPUT_DIR  # folder of the compared results
# folder of the reference results, e.g. Path("./res_iprt_phase3_1e8photons")
COMPARE_DIR_REF = OUTPUT_DIR

# %% [markdown]
# D4 and D4_bis describe the same atmosphere, built in two ways: their
# difference must stay within the Monte Carlo noise.

# %%
plot_case_polar("d4", iz=0, output_dir=COMPARE_DIR,
                case_ref="d4_bis", output_dir_ref=COMPARE_DIR_REF)  # BOA

# %% [markdown]
# Camera difference between the E6 variants v2 and v3.

# %%
plot_case_camera("e6_v2", output_dir=COMPARE_DIR,
                 case_ref="e6_v3", output_dir_ref=COMPARE_DIR_REF)

# %% [markdown]
# One case between the two folders, here D6 at TOA. The difference is
# zero as long as `COMPARE_DIR_REF` is `COMPARE_DIR`.

# %%
plot_case_polar("d6", iz=1, output_dir=COMPARE_DIR,
                case_ref="d6", output_dir_ref=COMPARE_DIR_REF)  # TOA
