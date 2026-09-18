# SMART-G CHANGELOG


## v2.0.0
Release date: xxx

Note: this changelog entry has been started during the `v2.0.0dev1` stage,
updated for `v2.0.0dev2`, `v2.0.0dev3` and `v2.0.0dev4`, and will be
completed and corrected before the final `v2.0.0` release.

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
    - iprt: `seclect_iprt_IQUV` → `select_iprt_iquv` (the typo
      included), `convert_SGout_to_IPRTout` →
      `convert_sgout_to_iprtout`, `compute_deltam_IPRTout` →
      `compute_deltam_iprtout`, `groupIQUV` → `group_iquv`,
      `plot_iprt_radiances` → `smartg.view.plot_iquv_comparison`, and
      their keyword arguments (`lSZA` → `szas`, `lI` → `i_list`, ...);
      in `smartg.iprt.phase3`, `case_D1` ... `case_E6_v3` →
      `case_d1` ... `case_e6_v3`, the parameters of `plot_polar_iprt`
      (`I` → `i`, `change_Q_sign` → `change_q_sign`, `maxI` → `max_i`,
      `cmapI` → `cmap_i`, `minI` → `min_i`, ...) and of
      `plot_camera_iprt` (`I` → `i`, `I_min` → `i_min`, `I_max` →
      `i_max`, `I_cmap` → `i_cmap`). `run_sim` now takes the paths of
      the BOA and TOA runs, None skipping a run, instead of the
      overwrite flag and the existence of the files, and it and the
      `get_*_sensors` helpers no longer take `nvza` and `nvaa`
    - histories: `Si` → `si`, `Si2` → `si2`, `BigSum` → `big_sum`, and
      their parameters (`Dij` → `dij`, `Ki` → `ki`, `S` → `s`,
      `only_I` → `only_i`); the `LEVEL` and `IDIR` parameters of
      `get_histories` are now `level` and `idir`
  - The IPRT tools are split into one module per phase of the
    `smartg.iprt` package: `smartg.iprt.iprt` becomes
    `smartg.iprt.common` (`group_iquv`, `compute_deltam`) and
    `smartg.iprt.phase_a` (`convert_sgout_to_iprtout`,
    `select_iprt_iquv`, `select_and_plot_polar_iprt`,
    `compute_deltam_iprtout`), and `smartg.iprt.iprt_phase3_runs`
    becomes `smartg.iprt.phase3`. The new `smartg.iprt.phase_b`
    gathers the phase B helpers that the C2 and C3 tests and the C2
    notebook each defined: the C2 and C3 atmospheres and sensor grids,
    the backward and forward runs of the 9 cases (`CASES`,
    `FORWARD_GROUPS`), the reading of the phase B ASCII tables
    (`read_iprt_iquv`, standard deviations included), the extraction
    of the SMART-G maps (`smartg_iquv`), the camera plots
    (`plot_camera_iquv`, `plot_camera_difference`) and `compare_case`,
    which replaces the `print_c2_res_noatm` and `print_c2_res_atm`
    functions of the notebook for any case, grid, level, direction and
    reference file, and returns the delta_m values
  - `smartg.iprt.phase3`: the `nphotons` parameter of every `case_*`
    function is now `n_photons`, the spelling `Smartg.run` and the rest
    of the package use
  - The `filename` parameter is now `fname`, as in the spectral modules,
    in `AerOPAC`, `Cloud`, `read_i3rc_aerosol`, `read_i3rc_cloud`,
    `AlbedoSpeclib`, `Hydrosol`, `HydrosolPR`, `HydrosolZhai` and
    `extract_points`; the classes store it as `self.fname`
  - The `lam` parameter and attribute of `AlbedoSpectrum`, `smartg.rrs`,
    `smartg.vrs` and `smartg.histories` is now `wavelength`
  - `smartg.truncation`: the `DM_trunc` and `GT_trunc` classes follow the
    CapWords convention as `DMTrunc` and `GTTrunc`
  - A wrong argument type now raises `TypeError` instead of
    `ValueError` in `as_theta_grid`, `LambSurface`, `GTTrunc` and the
    water phase truncation, and the water hydrosols raise `ValueError`
    instead of a bare `Exception` when neither a phase function nor a
    backscattering ratio is given
  - The IPRT validation notebooks follow PEP 8 module names:
    `validation_SMARTG_IPRT_phaseA.py` → `validation_smartg_iprt_phase_a.py`,
    `validation_SMARTG_IPRT_phaseB-C2.py` →
    `validation_smartg_iprt_phase_b_c2.py` and
    `validation_SMARTG_IPRT_phase3.py` → `validation_smartg_iprt_phase3.py`,
    the last one newly tracked
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
    backward compatibility, and `drop_axes`, the equivalent of the
    `MLUT.dropaxis` method
  - The tracked notebooks and tests no longer use LUT/MLUT either: the
    demo notebook selects and plots straight from the run Dataset
    instead of converting it back with `dataset_to_mlut`, and the phase
    matrices they build by hand are `xr.DataArray` objects
  - `Smartg.run` returns an `xr.Dataset` instead of an MLUT. The
    variable names, their order, the coordinates and the attributes
    are unchanged, and the dimensions which were anonymous in the
    MLUT are named (`sensor_in` / `wavelength_in`, `cdist_layer`,
    `hist_*`);
    `smartg.xarray.dataset_to_mlut` converts the output back to an MLUT
  - The `mixture` attribute of `AerOPAC`, `Cloud` and `AerUser` is now
    `ds_mix`, and holds the bulk optical properties as an `xr.Dataset`
    instead of an MLUT
  - The functions of `smartg.atmosphere` now raise `ValueError` instead of
    `NameError` on invalid inputs
  - The `phase` module has been moved from `smartg/tools/` to `smartg/`
    -> import from `smartg.phase` instead of `smartg.tools.phase`
  - The `read_phase`, `read_phase_dat`, `read_phase_nc`, `read_phase_cdf` and
    `convert_phase_to_iparper` functions have been moved from `smartg.atmosphere`
    to `smartg.phase`
  - `read_phase_dat`, `read_phase_nc` and `read_phase_cdf` now always return a
    4-D `xr.DataArray` (dims: `wavelength_phase`, `z_phase`, `nphamat`,
    `theta_atm/oc`). The `nphamat` dimension is no longer squeezed when its
    size is 1.
  - The `standard` parameter of the phase readers has been removed:
    files are expected in the standard IQUV convention, `run` doing the
    conversion into the parallel/perpendicular convention of the kernels
  - The phase-matrix term dimension is now named `nphamat` internally
    (was `stk`; the auxdata files keep `stk`, which is renamed on load),
    and the run output dimensions `stk_atm` / `stk_oc` are now
    `nphamat_atm` / `nphamat_oc`
  - The phase-matrix wavelength dimension is now named `wavelength_phase`
    (was `wav_phase`); it remains distinct from the `wavelength` axis of
    the profiles and run outputs
  - The wavelength parameters are now consistently named `wavelength`:
    `Smartg.run(wl=...)` is now `run(wavelength=...)`, the `get(wl)`
    method of the albedo objects is now `get(wavelength)`, the `wav`
    parameter of `Atm1D.calc` / `calc_split`, `BandSet` and the water
    classes is now `wavelength`, and the `pfwav` parameter of the
    atmosphere and water components is now `wavelength_phase` (matching
    the dimension it defines). The auxdata files keep their `wav`
    dimension, mirrored by the in-memory `ds_mix` datasets
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
      - `satellite_view` has been moved to `smartg.view` and renamed
        `camera_view`, with PEP 8 parameter names (`interp_name` →
        `interpolation`, `color_bar` → `cmap`, `color_reverse` →
        `cmap_reverse`, `fig_size` → `figsize`, `font_size` →
        `fontsize`, `save_file` → `save_path`, `stk` → `stokes`,
        `mat_force` → `matrices`, `cb_shrink` → `cbar_shrink`,
        `cb_sform` → `cbar_sci_format`, `fig_title` → `title`). It now
        returns the created `Figure` and applies `xlim` and `ylim` in
        every layout. It also takes the parameters of the
        `satellite_view_3MI` variant written for the 3MI simulator:
        `log_scale` (logarithmic color scale, for all the panels or
        per panel), `xlabel` and `ylabel` (the axis labels), and
        `layout` (`"row"` puts all the panels in one row)
      - The helpers `OOMFormatter`, `find_order`, `find_order_or_none`,
        `get_tv`, `find_id` and `get_sensors_pos_icells_from_3Dgrid` are now
        private
    - `Grid3D` and the voxel geometry helpers (`Get_3Dcells`,
      `locate_voxel_index`, ...) have been moved from `smartg.libATM3D` to the
      new `smartg.grid3d` module
    - The constant-theta readers `read_cld_nth_cte` of `smartg.libATM3D`
      and `read_phase_nth_cte` of `smartg.iprt` have been merged into
      `read_phase_cdf`: the two were the same, only the iprt one
      handling the `hum` axis as well as `reff`, and they read the same
      libRadtran / IPRT files as `read_phase_cdf`, differing only in
      what they handed back. `read_phase_cdf(fname, n_theta=...,
      normalize=False, output_sg_ready=False)` is the former call
      `read_phase_nth_cte(filename=..., nb_theta=...)`: the table on
      the wavelength and `hum` / `reff` axes of the file, which the
      `phase` argument of `Cloud3D` / `Aer3D` takes, as an
      `xr.DataArray`. The `convert_IparIper` parameter is gone, the
      conversion into the parallel/perpendicular convention being done
      by `run`. Two further differences: the table keeps the number of
      terms of the file (4 for spherical particles) instead of
      completing them into 6, which the components now do themselves,
      and it is float64 rather than float32, so the device tables of
      the IPRT C2 and C3 cases move by one float32 rounding on a tenth
      of their entries
    - The `cells` parameter and the `"ATM3D"` sentinel filename of `Atm1D`
      have been removed: a 3D atmosphere is now built with
      `smartg.atmosphere.Atm3D`
    - The 3D profile dataset carries its optical properties on a
      coordinate-bearing `iopt` axis; the `z_atm` axis it also had,
      holding plain indices labelled as altitudes, is gone. The 1D-only
      paths (the STP optical efficiencies, `cell_proba='auto'` and
      `_find_extinction`) raise an informative error in 3D instead of
      computing garbage from that index axis
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
  - The `CusForward` and `CusBackward` launching-mode classes have been
    moved from `smartg.smartg` to `smartg.objects3d`; they are NOT
    re-exported by `smartg.smartg`, so imports must be updated. Their
    constructor parameters have been renamed to snake case (`CFX` → `cfx`,
    `LMODE` → `lmode`, `POS` → `pos`, `REC` → `rec`, ...), with
    `TYPE` → `sampling` (`type` would shadow the builtin)
    - The `CusBackward` parameters have been renamed further, to spell out
      what they carry: `pos` → `position`, `thdeg` → `th_deg`,
      `phdeg` → `ph_deg`, `v` → `normal`, `aldeg` → `receiver_fov`,
      `rec` → `receiver` and `lmode` → `mode`, which `CusForward` follows
      (`lmode` → `mode` there too). `normal` also accepts a `Normal` now,
      converted to a `Vector`
    - The keys of the `dict` attribute of both classes are snake case now,
      and named after the constructor parameters they carry: `POS` →
      `position`, `THDEG` → `th_deg`, `PHDEG` → `ph_deg`,
      `VSUN` → `v_sun`, `SFOV` → `sun_fov`, `ALDEG` → `receiver_fov`,
      `REC` → `receiver`,
      `LMODE` → `mode`, `CFX` → `cfx`, `FOV` → `fov`, ..., with
      `TYPE` → `sampling_code` (it holds the code, not the `sampling`
      string). The `ALDEG` attribute of the output dataset keeps its name
  - The parameters of `Smartg` and `Smartg.run` follow PEP 8. Constructor:
    `obj3D` → `obj3d` and `opt3D` → `opt3d`. `run`:
    - `NBPHOTONS` → `n_photons`, `NBLOOP` → `n_loop`,
      `NBTHETA` → `n_theta`, `NBPHI` → `n_phi`, `NF` → `n_icdf`
    - `THVDEG` → `th_deg`, `PHVDEG` → `ph_deg`, `SEED` → `seed`,
      `RTER` → `earth_radius`, `DEPO` → `depo`, `DEPO_WATER` → `depo_water`
    - `OUTPUT_LAYERS` → `output_layers`, `XBLOCK` → `xblock`,
      `XGRID` → `xgrid`, `BEER` → `beer`, `RR` → `russian_roulette`,
      `WEIGHTRR` → `russian_roulette_weight`, `SZA_MAX` → `sza_max`,
      `SUN_DISC` → `sun_disc`
    - `SMIN`/`SMAX`/`RMIN`/`RMAX` → `s_min`/`s_max`/`r_min`/`r_max`,
      `FFS` → `ffs`, `DIRECT` → `direct`,
      `OCEAN_INTERACTION` → `ocean_interaction`, `myObjects` → `my_objects`,
      `cusL` → `cus_l`, `IsAtm` → `is_atm`
    The output variable names are unchanged; the `le` and `alis_options`
    dictionaries have since become the `LocalEstimate` and `Alis` classes
    (see New features)
  - Every count is spelled `n_`: the `nb_` prefix of `StdevLim(nb_loop_min)`,
    `DM_trunc(nb_streams)` and `aer2smartg(nb_theta)` is gone, they are now
    `n_loop_min`, `n_streams` and `n_theta`
  - The abbreviated parameters of `Smartg.run` have been given their full
    name: `atm` → `atmosphere`, `surf` → `surface` and `env` → `environment`
  - The `th_v_deg` and `ph_v_deg` angles of `Smartg.run` are now `th_deg` and
    `ph_deg`: they are the sun angles in forward mode and the viewing angles
    in backward mode, so the `v` of the viewing direction did not belong in
    their name
  - The `n_f` parameter of `Smartg.run` is now `n_icdf`, after the `icdf` and
    `icdf_2d` helpers: it is the number of points of the inverted functions
    it sizes (the phase functions and the wavelength probability)
  - The `r_r` and `weight_r_r` parameters of `Smartg.run` are now spelled out
    as `russian_roulette` and `russian_roulette_weight`
  - The `pol_off` parameter of `Smartg.run` has become `polarization`, with
    the opposite meaning and a `True` default: polarized light is considered
    unless `polarization=False` is passed. The `pol_off` parameter of the
    internal `_rayleigh` and `_calc_phase_gpu` helpers follows
  - The `Sensor` constructor parameters are now PEP 8: `POSX`/`POSY`/`POSZ`
    → `pos_x`/`pos_y`/`pos_z`, `THDEG` → `th_deg`, `PHDEG` → `ph_deg`,
    `LOC` → `loc`, `FOV` → `fov`, `TYPE` → `sensor_type`,
    `ICELL` → `icell`, `ILAM_0`/`ILAM_1` → `ilam_0`/`ilam_1`,
    `CELL_SIZE` → `cell_size` and `V` → `direction`. The keys of the
    `Sensor.dict` record and the fields of the `TYPE_SENSOR` numpy dtype
    they fill follow the same naming. The `Sensor` class is now type
    hinted, and its docstring documents every parameter
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
  - The declared dependencies have been trimmed and bounded. `pyarrow`,
    `pyhdf` and `statsmodels` are no longer declared, as no module nor
    notebook imports them (`pyhdf` still comes in as a dependency of
    `luts`), and the `ephem` and `docformatter` pixi dependencies have
    been dropped. The libraries whose API SMART-G calls directly are
    now capped at their next major version (`numpy>=2,<3`,
    `jupytext>=1.16,<2`, `geoclide>=4.0.0,<5`, `pytrunc>=2.0.0,<3`,
    `gatiab>=1.1.2,<2`), and the supported Python versions are
    `>=3.11,<3.15`, which is what the classifiers already announced

* New features
  - The scattering angles of a phase matrix no longer have to be equally
    spaced. Clustering them towards the forward and backward directions
    resolves the diffraction peak of large particles (desert aerosols,
    cloud droplets) with far fewer angles, which is what sizes the phase
    tables on the GPU: each angle costs 24 bytes per phase function.
    - New `smartg.phase.theta_grid(n, kind)` builds such a grid, `kind`
      being `'uniform'`, `'lobatto'` (Gauss-Lobatto-Legendre, the same
      nodes the truncation integrates on), `'chebyshev'` or `'peak'`
    - Clustering is not free: it takes its nodes from the middle of the
      range, where 1801 Lobatto angles are 3 times less accurate than
      1801 equally spaced ones (7.9e-3 against 2.7e-3 over 10 to 175
      degrees, on `watercloud_670.mie.cdf`). `'peak'` says how many
      nodes each end may take instead of fixing the shape, and its
      default is marginally better than Lobatto on the worst band
      (6.5e-3 against 7.9e-3)
    - The `n_theta` argument of the phase methods of `Atm1D`, `AerOPAC`,
      `Cloud`, `Hydrosol` and of `read_phase_cdf` now accepts those
      angles directly, in addition to a number of equally spaced ones
    - New `theta_grid` parameter of `Smartg.run` choosing the grid of
      the GPU tables: `'phase'` adopts the grid the phase matrices
      already carry, so no resampling takes place at all; a kind name or
      an explicit array are also accepted. The default is unchanged
    - Measured on the transmitted radiance under a thin water cloud,
      where the scattering angle is the viewing angle: at 451 angles,
      an equally spaced grid is 11% off in the forward peak while a
      Lobatto grid of that same size is within 0.24% of a 12601 angle
      reference, i.e. 28 times its size. Away from the peak all the
      grids agree to 0.1%, so the gain is in the peak alone
    - Note that this only pays where the phase matrix is sampled near
      a sharp feature. A geometry that sees the smooth 20 to 180
      degree body of the phase function, which is what the IPRT C3
      cases do, is unaffected by the grid: their delta_m against
      MYSTIC is uncorrelated with the discretisation error
  - A mixture of components tabulated on different scattering angle
    grids can keep every node of every table: `n_theta='native'` of
    the `calc` and `phase` methods of `Atm1D` and `Atm3D` (and of the
    component phase methods) resolves to the union of the grids the
    components' source tables carry, which is where a sum of the
    piecewise linear tables the kernel samples is exact. New
    `smartg.phase.union_theta_grid` builds that union, merging the
    float32 angles of the OPAC files with the float64 ones of the
    cloud files, and every component and atmosphere exposes it as
    `native_theta()`. Pass `theta_grid='phase'` to `Smartg.run` to keep
    it on the device
    - Measured on `wc` at 550 nm mixed with `continental_clean`:
      resampling the cloud (594 angles, 0.01 degree steps in the
      forward peak) onto the 721 angle default loses 99% of its peak
      below 2 degrees and 6% of its normalization, onto the 1801
      angle grid of the aerosol 14% and 1.3%; the 2019 angle union
      reproduces both tables to 1e-13
    - A warning names the components whose grids differ and the union
      they are mixed on, whether the union was asked for with
      `'native'` or forced by a user phase matrix, which keeps its own
      grid whatever `n_theta`
  - The phase matrix file readers return either of two layouts, chosen
    by the new `output_sg_ready` parameter of `read_phase`,
    `read_phase_nc` and `read_phase_cdf`: True (the default, and the
    former output) lays the matrix on a 1D profile, interpolated at
    `wavelength_phase` and `z_rh_reff` on a `z_phase` axis, for the
    `phase` argument of `AerOPAC` / `Cloud` / `Hydrosol` and
    `Atm1D.prof_phases`; False returns the table on the wavelength and
    `hum` / `reff` axes of the file, which the `phase` argument of
    `Cloud3D` / `Aer3D` takes, and which `read_phase_nth_cte` alone
    used to produce. A `.dat` file, a single matrix, has only the
    first layout
    - `read_phase_cdf` gains `n_theta`: `None` keeps the automatic
      equally spaced grid capped by `ntheta_max`, an int or the angles
      themselves choose the grid, and `'native'` resamples onto the
      union of every grid the file carries (2818 angles for the 25
      radii of the IPRT `watercloud_670.mie.cdf`, 38 for
      `waso_670.mie.cdf`), on which the file is reproduced exactly
  - A random walk now draws its deflection from exactly the phase
    matrix it then reads. `struct Phase` used to interleave a second
    copy of the matrix at equal-probability nodes, and the kernel drew
    the deflection by interpolating the inverse cumulative distribution
    linearly between those nodes, which samples a staircase density,
    constant inside each bin, and never corrects the mismatch with the
    smooth matrix. That mismatch is ~1e-4 per event; a cloud multiplies
    it by its ~1e3 scattering orders. Measured on IPRT C3 case 6 with
    1e6 photons per sensor and 3 seeds per grid, it was a 2.6% bias of
    the mean reflected intensity (delta_m of 3.0 against MYSTIC) that
    only fell to 1.1 with 12601 nodes, whatever the angle grid: an exact
    2818 angle grid from the file scored the same 3.0 as an equally
    spaced one of that length, and the same 1.1 once it drew from a
    12601 node distribution.
    - The cumulative distribution is now tabulated at the nodes of the
      angle grid itself, one float per entry, integrated exactly for
      the tabulated matrix (F11 linear in theta between nodes times the
      true sin(theta)), and `pSample` inverts one bin exactly: four
      Newton steps on the bin's mass, itself a 3 point Gauss-Legendre
      sum, which is what stays accurate in float32 inside the 0.01
      degree bins of a forward peak where the closed form cancels. The
      drawn density is the table's own interpolant, at any grid size,
      with no knob
    - The 6 copied matrix terms and the separate angle table are gone;
      an entry costs 24 bytes plus the 4 of its cumulative probability,
      instead of 52. This is what makes the 18001 angle IPRT C3 case
      with aerosols fit on a 16 GB card: 20491 phase matrices cost
      19.2 GB interleaved, 10.3 GB now
    - Reading the matrix at the drawn angle rather than at the
      equal-probability node is a numerical change for polarized runs:
      the intensity is untouched, since the weight update divides by
      the phase function it just multiplied by, but the polarization
      ratios now come from the angle grid
    - New `smartg/tests/test_phase_grid_ice.py` runs the kernel on
      the grids: the Iwabuchi and Suzuki (2009) figure 3 setup of the
      demo notebook, truncation off, on the `ic_baum_asc` ice table,
      reflected and transmitted radiance at two viewing angles for
      every grid kind at 451 and 1801 nodes, the file grid and a
      uniform 18001 grid, against saved 1e8 photon values at a fixed
      seed, within 4 sigma of the Monte Carlo noise the run estimates
      for itself, so that the check holds on any GPU model; the 1e10
      photon reference of the study is logged
  - The `le` and `alis_options` parameters of `Smartg.run` now take the new
    `LocalEstimate` and `Alis` objects, like every other structured
    parameter of the method. A dictionary is still accepted, with a
    deprecation warning, and its keys stay documented
    - `LocalEstimate(th=, phi=, th_deg=, phi_deg=, zip=, count_level=)`
      keeps the names of the former keys. It validates the angles once, so
      `run` no longer writes the radians back into the caller's dictionary,
      where they shadowed any later change to `th_deg`
    - `Alis(n_low=, hist=, max_hist=, n_jac=, n_jac_abs=)` spells out the
      former `nlow`, `njac` and `njac_abs` keys, and `n_low` is now
      required rather than raising a `KeyError` from inside `run`
    - Both classes raise a `ValueError` for the inconsistent inputs that
      used to pass silently: mismatched zipped angles, a `count_level` of
      the wrong length, `n_jac_abs` without a positive `n_jac`, and an
      `n_low` of 1, which the kernel divides by
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
    - Tests of the scattering angle grids: the host phase tables and
      the device lookup (`test_phase_grid.py`), the mixing of
      components on the union of their grids (`test_phase_mix.py`)
      and the two layouts of the phase file readers
      (`test_phase_readers.py`), mostly GPU-free
  - New `v_sun` parameter of `CusBackward`: the sun direction of a backward
    object simulation can be given as a vector (for example
    `gc.ang2vec(sza, phi, vec_view='nadir')`) on the launching mode itself,
    where it replaces the direction computed from the `th_deg` and
    `ph_deg` angles of `Smartg.run`
  - New `sun_fov` parameter of `CusBackward`: the angular size of the sun in
    a backward object simulation is carried by the launching mode itself
    (0.266 degree by default, the solar disc) and uploaded as the
    `CBACK_SFOVd` device constant, so the B and BR modes no longer depend on
    the `sun_disc` parameter of `Smartg.run`. It applies only without the
    `le` parameter, where `le_fov` gives the source its angular size
    instead; a `CusBackward` run which uses `le` without `le_fov` now warns
    that the local estimate looks at a point source
  - New `le_fov` parameter of `Smartg.run`: the local estimate directions of
    an object simulation are sampled inside a cone of that half-angle, which
    gives its angular size to the source they look at (the Sun disc for
    example). Uploaded as the `LE_FOVd` device constant, it replaces
    `sun_disc` in that role. It works in every build, not only with
    `obj3d=True`: the directions sampled inside the cone live in a device
    buffer allocated only when `le_fov` is set, and sized to the directions
    actually requested
  - README overhauled, with the new SMART-G logo
  - Updated dependency requirements: geoclide >= 4 (the 3D object code has
    been adapted to the geoclide 4 API), pytrunc >= 2, gatiab >= 1.1.2
  - `MAX_NREF` increased from 10 to 100
  - Aeronet read functions (`read_Aeronet_PFN`, etc.) now return `xr.DataArray`
    instead of LUT objects
  - Push to PyPI workflow added
  - The notebooks are tracked, and shipped in the source distribution, as
    jupytext percent scripts (`.py`) instead of `.ipynb` files. jupytext
    is a new dependency: it opens the scripts as notebooks in Jupyter and
    pairs each one with a local `.ipynb` notebook that keeps the outputs,
    and the new `sync-notebooks` pixi task synchronizes the pairs (see
    the README)
  - `smartg.auxdata` rewritten around `AuxData`, `Dataset` and source
    classes (`NextcloudSource`, `HttpArchiveSource`):
    - `download` skips the datasets already on disk (`force=True` to
      download them again), accepts a list of keys, and `dname` defaults
      to `SMARTG_DIR_AUXDATA`
    - new `check_update`: compares the data on disk with the remote
      versions (WebDAV ETag of the HYGEOS shares, HTTP ETag of the
      libRadtran archive) without downloading anything, prints a table
      and returns the statuses
    - new `update`: downloads only the missing, outdated or unrecorded
      datasets; each directory is replaced in one rename once the new
      content is fully extracted
    - a `.smartg_auxdata.json` manifest in the auxdata directory records
      the source, version and date of every downloaded dataset, and the
      checksum of every file: `check_update` also lists the files
      modified or deleted locally, and the new `restore` replaces them by
      the remote copies after confirmation (file by file for the HYGEOS
      shares); `AuxData.verify` does the local check offline
    - the failures are collected and raised at the end of a run
      (`AuxDataDownloadError`) instead of being printed and ignored;
      the downloads are streamed with a `tqdm` progress bar (new
      dependency) and retried on transient errors; the archive members
      are checked against path traversal
    - the reptran reference moves to the libRadtran 2024 archive
      (`reptran_2024_all.tar.gz`, the 2017 link is dead), the HYGEOS
      mirror stays the fallback
    - `AUXDATA_DICT`, the `*_URL` constants and `safe_download` are
      removed
  - A ruff configuration in `pyproject.toml`: a line length of 79 and the
    PEP 8, naming, numpy docstring and annotation rules on top of the
    default ones, for the whole package. `smartg/obselete_files`, whose
    unused Python 2 modules no longer parse, is excluded from ruff and
    from pyright
  - New `smartg.view.plot_polar_iquv` drawing I, Q, U and V matrices in
    polar view, split from `select_and_plot_polar_iprt`, which now selects
    with `select_iprt_iquv` and plots with it. `select_iprt_iquv` gains
    the `depol`, `change_q_sign`, `change_v_sign` and `depol_index`
    parameters of the wrapper, in the same order (`depol` is now its
    third positional parameter). `plot_polar_iprt` of
    `smartg.iprt.phase3` draws with it too, passing its `min_i` on.
    `plot_camera_iprt` now returns the created `Figure`
  - New `read_iprt_output`, `merge_least_noisy`, `PolarView`,
    `compare_polar_iprt` and `compare_plane_iprt` in
    `smartg.iprt.phase_a`: read a phase A result file, merge two runs
    of a case keeping the least noisy values, and compare a model with
    a reference in polar views or along the principal plane or the
    almucantar, with the plots and the delta_m of the whole case. The
    phase A notebook compares its 12 cases with them instead of 12
    copies of the same cells. Its difference plots now draw the
    symmetrical azimuths like the others, and its almucantar plots
    label their axis VAA instead of VZA
  - The aerosol and cloud cases of `smartg.iprt.phase3` (D3, D4,
    D4_bis, D5, E3, E4, E5) run on the native scattering angles of
    their files instead of 18001 resampled ones, through
    `read_phase_cdf(n_theta='native')`, the new
    `aer2smartg(n_theta='native')`, `calc(n_theta='native')` and
    `theta_grid='phase'`, which `run_sim` now passes to `Smartg.run`.
    The tables reproduce the files at every node, and the radiances
    agree with the resampled ones within the Monte Carlo noise. Every
    case takes a `seed`, -1 (the clock) by default
  - New `smartg/tests/test_iprt_phase3.py` runs the IPRT phase 3 cases
    D1 to D6 and E1 to E5 and compares them with saved 1e8 photons per
    direction results (`IPRT/phase3/smartg_ref_res/` of the auxdata)
    within the Monte Carlo noise of both, at 1e6 photons per direction
    by default (about 5 min for the file) and at 1e8 under the `slow`
    marker

* Corrections
  - Fix the nodes of the cumulative distribution a scattering deflection
    is drawn from. `_calc_phase_host` and `_isotropic` placed them at
    probabilities `(i+1)/n`, but the kernel indexes them with
    `RAND*(n-1)`, i.e. at `i/(n-1)`: the sampled probability spanned
    `]1/n, 1]` instead of `]0, 1]` and the first `1/n` of the scattering
    probability, the sharpest part of a forward peak, was unreachable.
    `_rayleigh` inverts its CDF analytically and already used `i/(n-1)`,
    so the molecular rows and the particle rows of the same table were
    sampling distributions offset by one node from each other. Every
    Monte Carlo result moves slightly; the fractions scattered below 1
    degree now match the tabulated phase function at every table size
    (0.2391 against 0.2386 at 1801 nodes, on a water cloud at 670 nm)
  - Fix two out-of-bounds reads of the phase tables in `device.cu`: both
    halves read entry `iang` and `iang+1`, so an index of `NF-1` reached
    into the next phase function, or past the end of the allocation for
    the last one. The equal-angle half reached it at a scattering angle
    of exactly 180 degrees, which a backward local estimate does hit,
    and the equal-probability half when `RAND` returned 1
  - The scattering angle axis of a truncated phase matrix is no longer
    overwritten with an equally spaced one in `Atm1D.calc`, which
    discarded the grid the matrix was built on
  - `Component.phase` and `_Comp3DFile.get_phase` no longer treat two
    angle grids of the same length as the same grid
  - `Atm1D.phase` no longer mixes components tabulated on different
    scattering angle grids through xarray's inner join, which silently
    kept only the angles common to both: 175 of them for an OPAC
    aerosol on the 721 angle default and a cloud carrying its file
    matrix on 594 angles. The matrices are now resampled onto the
    union of their grids, with a warning, and the `wavelength_phase`
    and `z_phase` axes, which have no union, must agree or raise. The
    3D merge does the same instead of adopting the grid of the first
    component or the longest one, and `_glob_particles_multi` no
    longer relabels a 1D aerosol matrix with the component grid when
    the two merely have the same length
  - `Cloud3D` / `Aer3D` complete a 4-term user phase matrix into 6
    terms, as they do for their bulk file; a 4-term matrix used to
    reach the 3D mixing as it was
  - `read_phase_nc` failed with a `KeyError` on any file with several
    humidities or radii given a scalar `z_rh_reff`, and a scalar
    `wavelength_phase` collapsed the output of `read_phase_nc` and
    `read_phase_cdf` to 3 dimensions: a scalar target now keeps its
    dimension, and a single `z_rh_reff` needs no `pfgrid`, as
    documented
  - The `Path + str` concatenations of `smartg.iprt.phase3`
    raised a `TypeError` before any run
  - A phase 3 case run with `overwrite=False`, when its intermediate
    files existed but not its IPRT output file, wrote a NaN top
    altitude in the `zout` axis of that file: the altitudes were only
    read before a run
  - `aer2smartg` gave the duplicated wavelength of a single wavelength
    file the properties of the last humidity or radius for every
    one. The phase 3 cases were not affected, their converted files
    having a single humidity
  - Seven of the 17 IPRT phase 3 cases of `smartg.iprt.phase3` did not
    run. `aer2smartg` still used the removed `NBTHETA` (D4_bis, E3, E4,
    E5), and D3, D4 and D5 gave `calc_iphase` a LUT named `wavelength`
    and `z` instead of `wavelength_phase` and `z_phase`. Past that
    crash, those three returned zero radiances: their files carry a
    single wavelength, on which `interp` gives NaN phase matrices, so
    the wavelength is now selected. D4 and D4_bis, the same aerosol
    built through `prof_phases` and through `AerOPAC`, now agree
    within the Monte Carlo noise
  - Important corrections in the water (ocean) module:
    - Phase matrix always extended to 6 Stokes components (P22=P11, P44=P33 for
      spherical particles)
    - Dimension names harmonised (`wav_phase_oc` / `z_phase_oc` →
      `wavelength_phase` / `z_phase`) for consistency with the atmospheric
      phase pipeline
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
  - The custom launching modes are now checked against the compilation
    options of `Smartg`: a `CusBackward` requires `back=True` and a
    `CusForward` requires `back=False`. Only the deprecated `B` mode was
    checked, so the launching code of the three other modes (which the
    kernel compiles only for the matching mode) was silently ignored.
    Similarly `my_objects` now requires `obj3d=True`, the `lmode` value is
    validated by the `CusForward` and `CusBackward` constructors, and a
    `cus_l` which is neither of these two classes is refused
  - The cone of the local estimate (`le_fov`) is now sampled around every
    requested direction. It was applied only to a single direction
    (`NBPHId == 1 && NBTHETAd == 1`) and silently skipped otherwise, so a
    multi-direction run ignored the angular size of the source: the
    radiance of the solar direction of a 5 degrees cone came out 6.8 times
    too high compared with the same direction run alone
  - Each local estimate direction of a receiver run is now weighted by its
    own zenith. `countPhotonObj3D` used `cos(tabthv[0])`, the zenith of the
    first direction, for all of them, and applied it to the virtual photon
    itself, so that projection also reached the radiance counted just after
    in the scattering direction loop. On a six-direction run the radiance
    was off by up to 37 %, and the receiver signal by 0.5 to 2 %
  - All the `NameError` and generic `Exception` exceptions of the smartg
    module are now `ValueError` (`RuntimeError` for the impact-point solver),
    and two `UnboundLocalError` hazards are fixed: the error format of a
    receiver run without `stdev`, and the base normal of a spherical
    reflector in the RF launching mode
  - Fix a bug in the constant-theta phase reader (`read_cld_nth_cte`,
    since merged into `read_phase_cdf`)
  - The phase matrices of a 3D component reached the kernels in the
    IQUV convention: `run` converts them into the parallel/perpendicular
    convention of the kernels and the conversion is an involution, so
    the second conversion done in `Comp3D.get_phase` cancelled it. The
    4 → 6 term expansion, which is not part of the conversion, is kept.
    The reader of the IPRT phase 3 runs is fixed the same way
  - A `Cloud` built with `zmax <= zmin` is refused at construction: it
    ended up with no vertical layer, and crashed far away with an
    `AttributeError` on `P_tot`, or divided by zero and filled `OD_p`
    with NaN when `phase=False`. `AerOPAC.phase` also raises when no
    layer is left
  - Fix the numpy 2.5 shape-setter deprecation in the interp module, and the
    strictly-increasing coordinate requirement of `make_interp_spline`
  - The `ipha` parameter of `phase_view` in `smartg_view` is now flexible:
    accepts an `int`, an `xr.DataArray` scalar, or a 1-D ndarray of indices;
    validation against the correct wavelength slice of `iphase_atm/oc` is performed
  - `compute_deltam_iprtout` raises a `TypeError` instead of a
    `NameError` when its inputs are not arrays
  - `select_iprt_iquv` with `change_u_sign=True` returned the standard
    deviation of U with a negative sign; only U itself changes sign now,
    as in `select_and_plot_polar_iprt`

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

