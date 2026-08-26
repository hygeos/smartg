# SMART-G CHANGELOG


## v2.0.0
Release date: xxx

Note: this changelog entry has been started during the `v2.0.0dev1` stage,
updated for `v2.0.0dev2`, and will be completed and corrected before the
final `v2.0.0` release.

* Several breaking changes
  - The `AtmAFGL` class has been renamed to `Atm1D`, with PEP 8 constructor
    parameters:
    - `atm_filename` → `fname`
    - `P0`           → `p0`
    - `O3`           → `tco3`
    - `H2O`          → `tcwp`
    - `NO2`          → `no2`
    - `O3_H2O_alt`   → `o3_h2o_alt`
    - `tauR`         → `tau_r`
    - `RH_cst`       → `rh_cst`
    - `O3_acs` / `NO2_acs` → `o3_acs` / `no2_acs`
    - the unused `US` parameter has been removed
    - the `NBTHETA` parameter of the profile/phase methods is now `n_theta`
  - The `smartg/tools/` folder has been dissolved: the `interp`, `progress`
    and `cdf` modules have been moved to `smartg/`, `modified_environ` has
    been moved and renamed to `smartg.environ`, and the remaining legacy
    content now lives in `smartg/obselete_files/`
    - New `smartg.postprocess` module regrouping the irradiance
      post-processing, with PEP 8 names: `Irr` → `plane_irr`,
      `SpherIrr` → `spherical_irr`, `reduce_Irr` → `irradiance_ds` (which
      now returns an `xr.Dataset`)
    - `diff1` and `diff1_end` have been moved into the new `smartg.diff`
      module, `expand_phase_4_to_6` into `smartg.phase` and the
      `AlbedoLike` alias into `smartg.albedo`
  - The `visualizegeo` module has been renamed to `smartg.objects3d`, with
    PEP 8 function names (`findRots` → `find_rots`,
    `generateMTF` → `generate_mtf`, `generateLEfH` → `generate_le_h`,
    `generateBox` → `generate_box`, `generateHfP` → `generate_h_p`,
    `generateHfA` → `generate_h_a`, `Ref_Fresnel` → `ref_fresnel`,
    `convertLGtoLE` → `convert_lg_to_le`, ...); `visualize_entity` has been
    moved to the view module
  - PEP 8 renames of the spectral and utility modules:
    - kdis: `KDIS` → `Kdis`, `KDIS_BAND` → `KdisBand`,
      `KDIS_IBAND` → `KdisIband`, `KDIS_IBAND_LIST` → `KdisIbandList`,
      `Kdis_Emission` / `Kdis_Avg_Emission` → `kdis_emission` /
      `kdis_avg_emission`; `reduce_kdis` completely rewritten
    - reptran: `REPTRAN` → `Reptran`, `REPTRAN_BAND` → `ReptranBand`,
      `REPTRAN_IBAND` → `ReptranIband`,
      `REPTRAN_IBAND_LIST` → `ReptranIbandList`, `Reptran_Emission` /
      `Reptran_Avg_Emission` → `reptran_emission` / `reptran_avg_emission`,
      `filename` → `fname`; the `output_type` parameter of `get_weights`
      has been removed and `reduce_reptran` / `reptran_emission` now return
      xarray objects
    - rrs: `Fk_N2` / `Fk_O2` → `fk_n2` / `fk_o2`, `Epsilon_N2` /
      `Epsilon_O2` / `Epsilon_air` → `epsilon_n2` / `epsilon_o2` /
      `epsilon_air`, `f0_N2` / `f0_O2` → `f0_n2` / `f0_o2`, `K` → `k_ratio`
    - cdf: `ICDF` → `icdf`, `ICDF2D` → `icdf_2d`
    - progress: `Progress` → `progress`, `Progress_notebook` →
      `ProgressNotebook`, `Progress_invisible` → `ProgressInvisible`, ...
    - albedo: `Albedo_cst` → `AlbedoCst`, `Albedo_speclib` →
      `AlbedoSpeclib`, `Albedo_spectrum` → `AlbedoSpectrum`,
      `Albedo_map` → `AlbedoMap`
    - bandset: the `Raman` parameter is now `raman`
  - The water module has been restructured for consistency with the
    atmosphere module: the `IOP*` classes (`IOP_base`, `IOP`, `IOP_1`,
    `IOP_Rw`, `IOP_profile`) have been replaced by the new `Water` /
    `Water1D` / `WaterRw` and `Hydrosol` / `HydrosolPR` / `HydrosolZhai`
    class hierarchy, with PEP 8 parameter names; the legacy water folder
    has been moved to `smartg/obselete_files/`
  - `saturation_pressure` now returns Pa instead of hPa
  - The internal data structures have been migrated from the legacy
    LUT/MLUT objects to xarray across the package (atmosphere, smartg,
    water, reptran, postprocess, views); the new `smartg.xarray` module
    provides `dataarray_to_lut` / `dataset_to_mlut` converters for
    backward compatibility
  - The functions of `smartg.atmosphere` now raise `ValueError` instead of
    `NameError` on invalid inputs
  - The `phase` module has been moved from `smartg/tools/` to `smartg/`
    -> import from `smartg.phase` instead of `smartg.tools.phase`
  - The `read_phase`, `read_phase_dat`, `read_phase_nc`, `read_phase_cdf` and
    `convert_phase_to_iparper` functions have been moved from `smartg.atmosphere`
    to `smartg.phase`
  - `read_phase_dat`, `read_phase_nc` and `read_phase_cdf` now always return a
    4-D `xr.DataArray` (dims: `wav_phase`, `z_phase`, `stk`, `theta_atm/oc`).
    The `nphamat` dimension is no longer squeezed when its size is 1.
  - The `filename` parameter of `read_phase`, `read_phase_dat`, `read_phase_nc`
    and `read_phase_cdf` has been renamed to `fname` (positional usage is
    unaffected, keyword usage must be updated).
  - The `conv_Iparper` parameter has been removed from `AerOPAC.phase()`,
    `Cloud.phase()` and `Atm1D.calc()`. The IQ → Ipar/Iper conversion is now
    performed automatically inside the `run()` method (only for atmospheric phases).
  - Several functions in `smartg.atmosphere` have been renamed for consistency:
    - `pha2Iparperconv`  → `convert_phase_to_iparper`
    - `BPlanck`          → `blackbody_radiance`
    - `rod`              → `rayleigh_od`
    - `raycrs`           → `rayleigh_crs`
    - `g` / `g0`         → `gravity_z` / `gravity_z0`
    - `FN2` / `FO2` / `Fair` → `f_n2` / `f_o2` / `f_air_co2`
    - `ma`               → `m_dry_air`
    - `n300` / `n_air`   → `n_air_co2_300` / `n_air_co2`
    - `RH` method        → `relative_humidity`
  - Several functions in `smartg.tools.smartg_view` have been renamed:
    - `plot_polar_xr`    → `plot_polar`
    - `transect2D_xr`    → `transect_2d` (via `transect2D`)
    - `ds_out` parameter → `ds_sg` in `smartg_view`
  - The `smartg.tools.smartg_view` module (then `smartg.smartg_view`) has
    been renamed to `smartg.view`
  - `smartg_view`, `transect_view`, `spectrum_view`, `profile_view` and `phase_view`
    now expect an `xr.Dataset` instead of an MLUT (MLUT still accepted with a
    deprecation warning)
  - The `new_atm` parameter of `Atm1D` (formerly `AtmAFGL`) has been removed
  - The 3D atmosphere construction API has been reworked (see New features):
    - The `Atm3D` and `Cloud3D` classes of `smartg.libATM3D` and their
      getter-based construction have been removed
    - The `smartg.libATM3D` module has been removed entirely:
      - `create_sensors` has been reworked into
        `smartg.sensor.get_sensors_grid(xgrid, ygrid, pos_z, th_deg,
        ph_deg, fov, sensor_type, loc, cell_size, grid_3d)`: the
        sensor raster is now given by the two boundary arrays, the
        atmosphere `Grid3D` is only needed in 3D (`ICELL` is ignored
        by the kernel in 1D) and the function returns the sensor
        list only
      - `satellite_view` has been moved to `smartg.view`, with PEP 8
        parameter names (`interp_name` → `interpolation`,
        `color_bar` → `cmap`, `color_reverse` → `cmap_reverse`,
        `fig_size` → `figsize`, `font_size` → `fontsize`,
        `save_file` → `save_path`, `stk` → `stokes`,
        `mat_force` → `matrices`, `cb_shrink` → `cbar_shrink`,
        `cb_sform` → `cbar_sci_format`, `fig_title` → `title`); it now
        returns the created `Figure`
      - The helpers `OOMFormatter`, `find_order`, `find_order_or_none`,
        `get_tv`, `find_id` and `get_sensors_pos_icells_from_3Dgrid` are now
        private
  - The `Sensor` class, the `get_sensor` function (formerly `Get_Sensor`)
    and the `LOC_CODE` constant have been moved from `smartg.smartg` to the
    new `smartg.sensor` module; they are still re-exported by
    `smartg.smartg`, so existing imports keep working. The `type`
    parameter of `get_sensor` has been renamed to `sensor_type`
  - The surface classes `FlatSurface`, `RoughSurface`, `LambSurface`,
    `RTLSSurface`, `RPVSurface` and `Environment` have been moved from
    `smartg.smartg` to the new `smartg.surface` module; they are NOT
    re-exported by `smartg.smartg`, so imports must be updated. The albedo
    classes (`AlbedoCst`, ...) are no longer re-exported by `smartg.smartg`
    either: import them from `smartg.albedo`. The constructor parameters of
    the surface classes and `Environment` have been renamed to snake case
    (`WIND` → `wind`, `ALB` → `alb`, `ENV_SIZE` → `env_size`, ...), and the
    `Environment` attributes `NENV`/`NXENVMAP`/`NYENVMAP` are now
    `nenv`/`nxenvmap`/`nyenvmap`
    - `Grid3D` and the voxel geometry helpers (`Get_3Dcells`,
      `locate_voxel_index`, ...) have been moved from `smartg.libATM3D` to the
      new `smartg.grid3d` module
    - `read_cld_nth_cte` has been moved from `smartg.libATM3D` to `smartg.phase`
    - The `cells` parameter and the `"ATM3D"` sentinel filename of `Atm1D`
      have been removed: a 3D atmosphere is now built with
      `smartg.atmosphere.Atm3D`
  - The `CusForward` and `CusBackward` launching-mode classes have been
    moved from `smartg.smartg` to `smartg.objects3d`; they are NOT
    re-exported by `smartg.smartg`, so imports must be updated. Their
    constructor parameters have been renamed to snake case (`CFX` → `cfx`,
    `LMODE` → `lmode`, `POS` → `pos`, `REC` → `rec`, ...), with
    `TYPE` → `sampling` (`type` would shadow the builtin)
  - The internal helpers of the smartg module are now private
    (`finalize` → `_finalize`, `calc_solid_angles` → `_calc_solid_angles`,
    `isotropic` → `_isotropic`, `rayleigh` → `_rayleigh`,
    `loop_kernel` → `_loop_kernel`), and its module-level constants follow
    PEP 8 (`type_Phase` → `TYPE_PHASE`, `type_IObjets` → `TYPE_IOBJECTS`,
    `dir_src` → `DIR_SRC`, ...); the unused `LOC_CODE`, `get_sensor` and
    `LUT` re-exports and the unused `src_kernel2` path have been removed
  - The `change_altitude_grid` external function has been removed (use `str2grid_arr`)
  - The deprecated `lib3D` module and legacy geometric modules have been removed
  - Several obsolete utility functions removed: `average`, `isiterable`, `isnumeric`,
    `vapor_pressure`, `lut_to_xr`, `compare_spectrum`, `convertVtoAngles`,
    `convertAnglestoV`, `Analyse_create_entity`, `trapzinterp`
  - The `fournier_forand` function has been removed from `smartg.phase`
    -> use `pytrunc.phase.fournier_forand` (pytrunc >= 2)
  - The `theta_trunc` parameter of `Hydrosol`, `HydrosolPR` and `HydrosolZhai`
    has been replaced by `truncation` (`DM_trunc | GT_trunc | None`, default
    `GT_trunc(trunc_frac=0.3, theta_tr=5.0)`): the water phase functions are
    now truncated with pytrunc like the atmospheric ones, and the scattering
    coefficient is scaled by `1 - f`. `None` disables the truncation.

* New features
  - New 3D atmosphere user API in `smartg.atmosphere`: a 3D atmosphere is now
    built directly as
    `Atm3D(atm_1d=Atm1D(...), grid_3d=Grid3D(...), comp_3d=[Cloud3D(...)])`
    and its `calc()` method returns the profile dataset consumed by
    `Smartg.run`, replacing the manual assembly of the `libATM3D.Atm3D`
    getter outputs into `Atm1D("ATM3D", ...)`
    - New `Comp3D` base class of the 3D components (`Cloud3D` and the new
      `Aer3D` implement it)
    - The new `Cloud3D` accepts the 3D cloud field as a dense `xr.Dataset` (or
      NetCDF file) with the `ext(z, y, x)` and `reff(z, y, x)` variables and
      the cell-boundary coordinates, as raw arrays (IPRT 1-based indices), or
      converted from the legacy I3RC/IPRT ASCII files with the new
      `read_i3rc_cloud` function
    - New `ssa_cst` parameter of `Cloud3D` to force the cloud single
      scattering albedo
    - `Atm3D` supports several 3D components in `comp_3d`: in the cells
      shared by several components (and by the 1D aerosols), the extinctions
      are summed, the single scattering albedos are extinction-weighted and
      the phase matrices are weighted by the scattering coefficients
    - New `Aer3D` 3D aerosol component: bulk optical properties from the
      OPAC aerosol mixtures or species ('desert', 'continental_clean',
      'waso', ...) as a function of the relative humidity; the 3D
      distribution (extinction at `w_ref` and per-cell rh, clamped to the
      file's humidity range as in the 1D `AerOPAC`) follows the same three
      routes as `Cloud3D` (dense dataset with `rh(z, y, x)`, raw arrays,
      ASCII files via the new `read_i3rc_aerosol` function), with the
      `rh_acc`/`rh_min`/`rh_max`, `phase` and `ssa_cst` options
  - New `AerUser` class in `smartg.atmosphere` to define custom aerosol / cloud
    optical properties (extinction, SSA, phase matrix) from user-supplied data
  - New `get_prof_phases` utility function to easily extract phase matrices from
    an existing simulation profile
  - `prof_phases` parameter of `Atm1D` now also accepts `xr.DataArray` objects
    in addition to LUT objects
  - New `read_phase` dispatcher function accepting `.dat`, `.nc` and `.cdf` files
  - New `read_phase_nc` function to read phase matrices from NetCDF files
  - New `read_phase_cdf` function to read phase matrices from libRadtran-style
    CDF files, with optional wavelength / altitude sub-selection
  - ALIS mode: new options for AMF computation (`amf_variance`, `cdist_wabs`,
    `nscl`, `scatter_classes`, `norders` parameters of `Smartg`)
    - `amf_variance=True`: enables storage of ⟨D²⟩ (second moment of photon
      path lengths) for Jensen bias correction
    - `cdist_wabs=True`: includes absorption weight in `cdist` accumulation
    - `nscl` / `scatter_classes` / `norders`: scatter-class decomposition for
      AMF (by last-scattering layer, scattering order, or both combined)
    - `njac_abs`: Jacobians for absorption only
  - Photon histories mode consolidated: jax post-processing for the AMF
    computation from the photon histories, `last_scatter_layer` index added
    to the per-photon history record, warning when the history buffer
    saturates, and buffer size capped to fit 16 GB GPUs
  - New pytest validation suites:
    - IPRT phase B C2 and C3 3D test cases, with a fast and a slow tier and
      pinned GPU references
    - Hydrolight validation of the water module (`test_water.py`)
    - kdis and reptran tests
    - GPU-free tests of the 3D profile construction (`test_atm3d.py`:
      Atm3D multi-component mixing, Cloud3D, Aer3D)
  - README overhauled, with the new SMART-G logo
  - Updated dependency requirements: geoclide >= 4 (the 3D object code has
    been adapted to the geoclide 4 API), pytrunc >= 2, gatiab >= 1.1.2
  - `MAX_NREF` increased from 10 to 100
  - Aeronet read functions (`read_Aeronet_PFN`, etc.) now return `xr.DataArray`
    instead of LUT objects
  - Push to PyPI workflow added

* Corrections
  - Important corrections in the water (ocean) module:
    - Phase matrix always extended to 6 Stokes components (P22=P11, P44=P33 for
      spherical particles)
    - Dimension names harmonised (`wav_phase_oc` / `z_phase_oc` → `wav_phase` /
      `z_phase`) for consistency with the atmospheric phase pipeline
  - Fix out-of-bounds index error in `AerOPAC.dtau_ssa` when relative humidity
    or wavelength is exactly at the upper axis boundary
  - Fix the 1D aerosol phase mixing with a 3D cloud
  - Fix the phase truncation with a 3D atmosphere
  - `FlatSurface` now sets `alb` and `kp` to None like the other surfaces,
    fixing an `AttributeError` in `Smartg.run` with `surf=FlatSurface()`
  - Passing the individual coefficients `k0`/`k1p`/`k2p` to `RTLSSurface` or
    `r0`/`k`/`bt`/`rc` to `RPVSurface` no longer raises a `TypeError`
    (the overrides were assigned into a tuple)
  - Printing a `LambSurface`, `RTLSSurface` or `RPVSurface` no longer raises a
    `KeyError` (`__str__` referenced a non-existent `SURFALB` key), and
    `RPVSurface` no longer labels itself `RTLS`
  - The remaining `NameError` exceptions raised by `CusForward`, `CusBackward`
    and the `cusL` guard of `Smartg.run` are now `ValueError`, and the `V`
    validity check of `CusBackward` no longer compares a `Vector` with `!=`
  - All the `NameError` and generic `Exception` exceptions of the smartg
    module are now `ValueError` (`RuntimeError` for the impact-point solver),
    and two `UnboundLocalError` hazards are fixed: the error format of a
    receiver run without `stdev`, and the base normal of a spherical
    reflector in the RF launching mode
  - Fix a bug in `read_cld_nth_cte`
  - Fix the numpy 2.5 shape-setter deprecation in the interp module, and the
    strictly-increasing coordinate requirement of `make_interp_spline`
  - The `ipha` parameter of `phase_view` in `smartg_view` is now flexible:
    accepts an `int`, an `xr.DataArray` scalar, or a 1-D ndarray of indices;
    validation against the correct wavelength slice of `iphase_atm/oc` is performed

* Deprecation removal
  - All functions and classes deprecated before v1.2.0 have been removed
    (see breaking changes above)
  - Removed deprecated modules:
    - `smartg.geometry`
    - `smartg.transform`
    - deprecated legacy geometric modules replaced by `geoclide`
    - old deprecated `lib3D` module


## v1.2.0
Release date: 2026-03-13

* The use of a scattering phase truncation is now possible. Two truncation methods added.
  - The Delta-m trunctation (DM) 
  - The Geometric Truncation (GT)

* Update of the demo notebook by adding an example using the GT truncation method


## v1.1.5
Release date: 2026-03-05

* Correct crash occuring when creating a 3D atmospheric profil

* Update IPRT phase B C2 notebook

* Add warning message in case FOV > 0 and TYPE=0 in Sensor class
  - FOV is forced to 0 is that case


## v1.1.4
Release date: 2026-02-10

* Corrections in demo_notebook and atmosphere.py
  - Add patched transect2D function in smartg_view.py that properly reuses
    existing subplot axes when overlaying multiple transects on the same figure
  - Add early netCDF4 import in atmosphere.py to prevent HDF5 library conflicts

* Correct crash occuring with newer numpy version
  - Replace np.trapz by np.trapezoid

  
## v1.1.3
Release date: 2025-12-01

* Correction in device.cu to avoid an NVCC crash on Windows

* Correction and update in README

* Update of the default cache_dir value to improve cross-platform compatibility

* Correction in the extraction for the reptran auxdata .tar archive


## v1.1.2
Release date: 2025-11-27

* Remove hitran-api

* Correction in MANIFEST.in file

* Require geoclide >= 3 and remove python maximum version constraint

* Correction in test_smartg_jax

* Cleaning before preparing a release for conda-forge


## v1.1.1
Release date: 2025-11-25

* Corrections in AerOPAC and Cloud classes
  - Proper handling of the `phase` parameter
  - Docstring updated and corrected
  
* Correction of bugs while using RoughSurface for simulations in spherical shell geometry
  - Fix bug when using reflectance=false (for both BRDF=true and BRDF=False)
  - Fig bug with showdowing-masking function. Completely fixed for BRDF=true. 
    Partially for BRDF=false, still very small bias for very high VZA values with
    sensor at TOA. 
    

* Use of template for all geometric classes in the source CUDA code

* Remove an unused notebook

* Only use of pathlib to improve cross-platform compatibility


## v1.1.0
Release date: 2025-08-19

* Smartg can be used with Jax (see demo_notebook)

* New way to download and consider auxiliary data

* Manual pycuda context init now possible

* Adaptation to numpy 2

* A pyproject.toml is now available i.e., installation with pip now possible

* The polarization can be desactivated by setting pol_off=True in smartg run method.

* The intern python geometric modules are now depracated, the geoclide package is used instead
  - See https://github.com/hygeos/geoclide
  - Several bugs are corrected
  - Best performance thanks to numpy
  - More functionalities are available

* Update of several notebooks

* Results with only photons not scattered by aerosols can be added to the output by setting 
  no_aer_output=True in smartg run method

* The OUTPUT_LAYERS parameter of the smartg run method improved
  - more options
  - not only remove useless outputs but avoid also useless photon counting

* New key for local estimate intput dictionary
  - the key 'count_level' to choose if we want to count only a particular level (in plan parallel), 
    see the doctring of the smartg run method

* The smartg 3D atm mode is now validated, but still in dev!!
  - A huge change (modules, classes, ...) is planned for this mode, so it is not recommended to use it for now.
  - The IPRT phase B C2 and C3 test cases have been validated
  - The notebook validation_SMARTG_IPRT_phaseB-C2.ipynb is available with examples but will be 
    obselete in future release!
  - Only clouds in 3D for the moment
  - Only 1 type of cloud per simulation for the moment

* Several improvements
  - Rewrite the python docstring of several functions and classes
  - Some corrections
  - ...


## v1.0.8
Release date: 2025-08-18

* Correct an important bug with the photon orthogonal direction initialization
  - Wrong values were visible only for the Q and U stokes components in case
    we are not in radiance. Otherwise no impact.


## v1.0.7
Release data: 2025-06-03

* Correct the env file by removing default conda channel

* Update the README

* Correct several bugs occuring when a user incorporate its own mixture file
  - For mixture with wl axis dimension different to the provided mixture wl dimension
  - For mixture with a humidity axis size equal to 1

* Complete the python doc of AerOPAC
  - Missing Z_mix, Z_free and Z_stra definitions


## v1.0.6
Release data: 2025-05-25

* Avoid bug due to scipy function renaming since version 1.14
  - Force scipy version<1.14 while installing python dependencies


## v1.0.5
Release date: 2024-12-03

* Correction of a bug in transform matrix inversion when 2 transform objects
  are multiplied (Python part not CUDA)

* Cleaning and some corrections
  - hygeos url link corrected
  - some cleaning in transform.py
  - add SMART-G logo and DOI in README

* Rewriting AerOPAC and Cloud python documentation


## v1.0.4
Release date: 2024-09-11

* Correction of a crash occuring while using the new calc_iphase function
  (introduced in v1.0.3) inside ocean.


## v1.0.3
Release date: 2024-08-29

* Correction of a bug in function calc_iphase.
  - The bug may appear while giving a pfgrid array with a size > 3,
    and with pfgrid != grid (z_atm)


## v1.0.2
Release date: 2024-08-22

* Correction of a bug that occurs with high values of water vapor (H2O)


## v1.0.1
Release date: 2024-06-20

* Missing auxiliary data in Makefile added (Clouds and IPRT).

* The previous CHANGELOG corrected.


## v1.0.0
Release date: 2024-05-17

* Downloading libRadtran is not needed anymore. Auxiliary data has been completely
  rebuilt and can be downloaded using the Makefile.

* New way to compute the OPAC aerosol models (use AerOPAC instead of AeroOPAC).
    - Important correction concerning the mixing of species!
    - The OPAC mixtures are pre-calculated with MOPSMAP (https://mopsmap.net/).
    - The OPAC models vertical distribution can be ajusted (composed of aerosols
      from mixture and/or free troposphere and/or stratophere).
    - OPAC models updated! see [Koepke et al. 2015]. The new desert and antartic models
      with spheroid particles are considered. The old spherical versions are also
      available, see -> 'desert_spheric' and 'antartic_spheric'. 

* New cloud models (use Cloud instead of CloudOPAC), taken from ARTDECO
    - small correction on the phase matrix computation
    - water cloud can have now an effective radius up 30 um (instead of 14 previouly).
    - Ice clouds are considered! 3 available: baum_ghm, baum_asc and baum_sc

* New way to consider the gaseous density vertical distribution (still with AFGL)

* New way to consider the gaseous absorption cross section (NO2 and O3)
    - Add new O3 Bogumil data (Chehade et al. 2013)
    - Add O3 acs from Serdyuchenko et al. 2014
    - Add NO2 data from Bingen et al. 2019

* K-distribution now available !
    - 3 kdis -> kato, kato2 and SENTINEL_2_1_MSI
    - The kdis format (ascii or h5) is now automatically recognized

* In general: The repository has been cleaned. And several fixes and improvements
  have been made.


## v0.9.4
Release date: 2024-02-27

* Bug corrections
    - bug in generatePro_multi visible when using pfwav corrected
    - bug when using pfgrid in the calculation of the phase matrix corrected


## v0.9.3
Release date: 2024-02-19

* General Updates (codes and notebooks) to work with last python packages, and corrections

* 3D objects improvements
    - 3D objects can be used in LE (Local Estimate) mode
    - Cuboid objects can be created using the function generateBox where each face have
     its own albedo (see visualizegeo module)
    - A 3D object face albedo can now vary spectrally

* Validation with all IPRT phase A test cases
  - The complete notebook with all test cases is available
  - Some tests can be tested with pytest -> use pytest tests/test_quick_iprt_phaseA.py

* Spheriod aerosols (see IPRT phase A notebook case A4) and ice clouds are now considered
  if the phase matrix is given manually.

* Several improvements in 3D atmosphere mode. But still in development and not documented


## v0.9.2
Release date: 2019-05-23

* Source code release !
  This release now includes the source code for the kernel instead of binaries.
  This allows for more flexilibity with respect to compilation options.

* New! 3D objects (in development)
    - 3D objects require smartg option Smartg(opt3D=True) and work in either forward or backward mode
    - Designed to simulate a solar tower power plant
    - Please check notebooks/demo_notebook_objects.ipynb
    - The following features are implemented:
        -> Reflectors (mirrors with optional roughness)
        -> Receivers with flux distribution map (direct, diffuse...)
        -> Custom photon launching options (target an area or specific objects:
           see option Smartg().run(cusL=...))
        -> Utilities to quickly generate a 3D scene with heliostats and a tower
           (see smartg/visualizegeo.py)

* New! Refraction and limb geometry
    - Please check notebooks/demo_notebook.ipynb

* ALIS method is now validated and extended to water
    - Please check notebooks/Validation_smartg_compilation.ipynb
      which compiles all validation exercises for Smart-G and an example of
      perturbative Jacobians

* Locale estimate: add 'zip' option to allow for non-cartesian product of
  output angles

* Notebooks have been extended

* New! 3D atmosphere (in development - undocumented)


## v0.9.1
Release date: 2018-11-01

* Add compilation support for additional architectures, including GeForce 20xx
  series (Turing).
* Fix OPAC phase function interpolation
* Add aerosol altitude scaling options


## v0.9
Release date: 2018-09-21

First public release.

