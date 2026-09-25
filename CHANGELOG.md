# SMART-G CHANGELOG


## v2.0.0b1
Release date: 2026-09-22

Note: this changelog entry has been started during the `v2.0.0dev1` stage,
updated for `v2.0.0dev2`, `v2.0.0dev3`, `v2.0.0dev4`, `v2.0.0dev5` and
the `v2.0.0b1` beta, and will be completed and corrected before the
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
    - `diff1` has been moved from `smartg.atmosphere` into the new
      `smartg.diff` module, next to the new `diff1_end`; the new
      `expand_phase_4_to_6` is in `smartg.phase` and the new `AlbedoLike`
      alias in `smartg.albedo`
  - The `visualizegeo` module has been renamed to `smartg.objects3d`, with
    PEP 8 names:
    - functions: `findRots` → `find_rots`, `generateMTF` →
      `generate_mtf`, `generateLEfH` → `generate_le_h`, `generateBox` →
      `generate_box`, `generateHfP` → `generate_h_p`, `generateHfA` →
      `generate_h_a`, `Ref_Fresnel` → `ref_fresnel`, `convertLGtoLE` →
      `convert_lg_to_le` and `extractPoints` → `extract_points`;
      `visualize_entity`, `receiver_view`, `cat_view` and `nopt_view`
      have been moved to the view module
    - the classes keep their names, not their parameters and
      attributes: `Transformation`: `rotationOrder` → `rotation_order`
      (attribute `rotOrder` → `rot_order`); `Entity`: `TC` → `tc`,
      `materialAV` → `material_front`, `materialAR` → `material_back`,
      `bboxGPmin` / `bboxGPmax` → `bbox_pmin` / `bbox_pmax`, parameters
      and attributes alike; `GroupE`: `LE` → `entities`, `BBOX` →
      `bbox`, and the `bboxGPmin` / `bboxGPmax` attributes →
      `bbox_pmin` / `bbox_pmax`; `Heliostat`: `POS` → `pos`, `SPX` /
      `SPY` → `n_facets_x` / `n_facets_y`, `HSX` / `HSY` →
      `helio_size_x` / `helio_size_y`, `CURVE_FL` →
      `curve_focal_length`, `REF` → `reflectivity`, `ROUGH` →
      `roughness`, and the `sPx` / `sPy` / `hSx` / `hSy` / `curveFL`
      attributes → `n_facets_x` / `n_facets_y` / `helio_size_x` /
      `helio_size_y` / `curve_focal_length`
    - the parameters of the functions, in the same order: `find_rots`:
      `UI` / `UO` / `vecNF` → `dir_in` / `dir_out` / `normal`;
      `generate_mtf` and `generate_le_h`: `HELIO` → `heliostat`, `PR` →
      `receiver_pos`, `THEDEG` / `PHIDEG` → `theta_deg` / `phi_deg`,
      `MTF` → `facet_transforms`; `generate_box`: `dimXYZ` →
      `dim_xyz`, `matAV` → `material_front`, `ref` → `reflectivity`,
      `rough` → `roughness`, `rotZ` → `rot_z`; `ref_fresnel`:
      `dirEnt` / `geoTrans` → `dir_in` / `geo_transform`;
      `generate_h_p` and `generate_h_a`: `THEDEG` / `PHIDEG` →
      `theta_deg` / `phi_deg`, `PH` → `heliostat_pos_list`, `PR` →
      `receiver_pos`, `MINANG` / `MAXANG` / `GAPDEG` → `min_ang_deg` /
      `max_ang_deg` / `gap_ang_deg`, `FDRH` → `first_dist`, `NBH` →
      `n_heliostats`, `GAPDIST` → `gap_dist`, `HSX` / `HSY` →
      `helio_size_x` / `helio_size_y`, `PILLH` → `pillar_height`,
      `REF` → `reflectivity`, `ROUGH` → `roughness`, `HTYPE` →
      `heliostat_type`, `LMTF` → `facet_transforms_list`, `RLPH` →
      `return_positions`; `convert_lg_to_le`: `LGOBJ` → `obj_list`;
      `rotate_vector`: `rot_order` → `rotation_order`;
      `interpolate_refls_from_wls`: `wls` / `refls` / `wls_new` →
      `wavelengths` / `reflectivities` / `new_wavelengths`;
      `is_comment`: `s` → `line`
    - the parameters of the functions moved to the view module, in the
      same order: `visualize_entity`: `ENTITY` → `entities`, `THEDEG` /
      `PHIDEG` → `theta_deg` / `phi_deg`, `PLANEDM` → `draw_method`,
      `RAYCOLOR` → `ray_color`, `SR_VIEW` → `sr_view`; `receiver_view`:
      `SMLUT` → `ds_sg_out`, `CAT` → `cat`, `LOG_I` →
      `log_color_scale`, `NAME_FILE` → `save_path`, `MTOA` → `mtoa`,
      `VMIN` / `VMAX` → `vmin` / `vmax`, `INT` → `interpolation`,
      `W_VIEW` → `flux_unit`; `cat_view`: `SMLUT` → `ds`, `MTOA` →
      `mtoa`, `NCL` → `ncl`, `UNIT` → `output_unit`, `W_VIEW` →
      `flux_unit`, `M_VIEW` → `length_unit`, `PRINT` →
      `print_results`, `ACC` → `accuracy`; `nopt_view`: `SMLUT` →
      `ds`, `BACK` → `back`, `ACC` → `acc`, `NCL` → `ncl`, `fl_TOA` →
      `mtoa`, `NAATM` → `natm_approx`
  - PEP 8 renames of the spectral and utility modules:
    - kdis: `KDIS` → `Kdis`, `KDIS_BAND` → `KdisBand`,
      `KDIS_IBAND` → `KdisIband`, `KDIS_IBAND_LIST` → `KdisIbandList`,
      `Kdis_Emission` / `Kdis_Avg_Emission` → `kdis_emission` /
      `kdis_avg_emission`; `reduce_kdis` completely rewritten;
      `Kdis.get_weight`, which returned five LUTs, is now
      `Kdis.get_weights` and returns the six values of
      `KdisIbandList.get_weights` (the bandwidth-weighted norm added),
      and both return `xr.DataArray` objects instead of LUTs
    - reptran: `REPTRAN` → `Reptran`, `REPTRAN_BAND` → `ReptranBand`,
      `REPTRAN_IBAND` → `ReptranIband`,
      `REPTRAN_IBAND_LIST` → `ReptranIbandList`, `Reptran_Emission` /
      `Reptran_Avg_Emission` → `reptran_emission` / `reptran_avg_emission`,
      `filename` → `fname`; `ReptranIbandList.get_weights` returns
      `xr.DataArray` objects instead of LUTs, and `reduce_reptran` /
      `reptran_emission` now return xarray objects
    - rrs: `Fk_N2` / `Fk_O2` → `fk_n2` / `fk_o2`, `Epsilon_N2` /
      `Epsilon_O2` / `Epsilon_air` → `epsilon_n2` / `epsilon_o2` /
      `epsilon_air`, `f0_N2` / `f0_O2` → `f0_n2` / `f0_o2`, `K` →
      `k_ratio`, `bjp` / `bjm` → `bjm_plus` / `bjm_minus`, `L_O2` /
      `L_N2` / `L` → `l_o2` / `l_n2` / `l_air`, `L2d` / `L2d_inv` →
      `l2d` / `l2d_inv`; `is_odd` has been removed
    - vrs: `Gauss` → `gaussian_peak`, `fR` → `raman_response`, `V2d` /
      `V2d_inv` → `raman_forward` / `raman_inverse`
    - cdf: `ICDF` → `icdf`, `ICDF2D` → `icdf_2d`
    - progress: `Progress` → `progress`, `Progress_notebook` →
      `ProgressNotebook`, `Progress_invisible` → `ProgressInvisible`,
      `Progress_progressbar` → `ProgressProgressbar` and
      `Progress_progressbar2` → `ProgressProgressbar2`
    - albedo: `Albedo_cst` → `AlbedoCst`, `Albedo_speclib` →
      `AlbedoSpeclib`, `Albedo_spectrum` → `AlbedoSpectrum`,
      `Albedo_map` → `AlbedoMap`
    - bandset: the `Raman` parameter is now `raman`
    - iprt: `seclect_iprt_IQUV` → `select_iprt_iquv` (the typo
      included), `convert_SGout_to_IPRTout` →
      `convert_sgout_to_iprtout`, `compute_deltam_IPRTout` →
      `compute_deltam_iprtout`, `groupIQUV` → `group_iquv`,
      `plot_iprt_radiances` → `smartg.view.plot_iquv_comparison`, and
      their keyword arguments:
      - `select_iprt_iquv`: `change_U_sign` → `change_u_sign`,
        `I_index` → `i_index`
      - `select_and_plot_polar_iprt`: `change_Q_sign` / `change_U_sign`
        / `change_V_sign` → `change_q_sign` / `change_u_sign` /
        `change_v_sign`, `maxI` / `maxQ` / `maxU` / `maxV` → `max_i` /
        `max_q` / `max_u` / `max_v`, `cmapI` / `cmapQ` / `cmapU` /
        `cmapV` → `cmap_i` / `cmap_q` / `cmap_u` / `cmap_v`,
        `forceIQUV` → `force_iquv`, `I_index` → `i_index`,
        `outputIQUV` → `output_iquv`, `outputIQUVstd` →
        `output_iquv_std`
      - `convert_sgout_to_iprtout`: `lm` → `datasets`, `lU_sign` →
        `u_signs`, `ldepol` → `depols`, `lalt` → `altitudes`, `lSZA` /
        `lSAA` / `lVZA` / `lVAA` → `szas` / `saas` / `vzas` / `vaas`,
        `file_name` → `fname`
      - `compute_deltam_iprtout`: `I_obs_id` / `I_mod_id` →
        `i_obs_id` / `i_mod_id`
      - `group_iquv`: `lI` / `lQ` / `lU` / `lV` → `i_list` / `q_list` /
        `u_list` / `v_list`
      - `plot_iquv_comparison`: `IQUV_obs` / `IQUV_mod` → `iquv_obs` /
        `iquv_mod`, `IQUVstd_obs` / `IQUVstd_mod` → `iquv_std_obs` /
        `iquv_std_mod`, `IQUVyMin` / `IQUVyMax` → `iquv_ymin` /
        `iquv_ymax`
      - in `smartg.iprt.phase3`, `case_D1` ... `case_E6_v3` →
        `case_d1` ... `case_e6_v3`; `plot_polar_iprt`: `I` / `Q` / `U`
        / `V` → `i` / `q` / `u` / `v`, `minI` → `min_i` and the
        `change_*_sign`, `max*` and `cmap*` renames of
        `select_and_plot_polar_iprt`; `plot_camera_iprt`: `I` / `Q` /
        `U` / `V` → `i` / `q` / `u` / `v`, `I_min` / `I_max` /
        `I_cmap` → `i_min` / `i_max` / `i_cmap`; `aer2smartg`:
        `filename` → `fname`. `run_sim` now takes the paths of the BOA
        and TOA runs, None skipping a run, instead of the overwrite flag
        and the existence of the files; it no longer takes `phi`, it and
        the `get_*_sensors` helpers no longer take `nvza` and `nvaa`,
        and its `nphotons`, `wl`, `surf`, `dep` and `ntheta` parameters
        are now `n_photons`, `wavelength`, `surface`, `depol` and
        `n_icdf`
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
  - `smartg.iprt.phase3`: the `earth_r` parameter of `run_sim` and of
    the sensor helpers (`get_d1_to_e5_boa_sensors`,
    `get_d1_to_e5_toa_sensors`, `get_d1_to_e5_toa_sensors_old`,
    `get_e6_toa_sensors`) is now `earth_radius`, as in `Smartg.run`
  - The `filename` parameter is now `fname`, as in the spectral modules,
    in `AerOPAC`, `Cloud`, `AlbedoSpeclib` and `extract_points`;
    `AerOPAC` and `Cloud` store it as `self.fname`
  - The vertical structure parameters of `AerOPAC` are lower case, as in
    `AerUser`: `H_mix_min` / `H_mix_max` → `h_mix_min` / `h_mix_max`
    (`h_min_mix` in 2.0.0b1), `H_free_min` / `H_free_max` →
    `h_free_min` / `h_free_max`, `H_stra_min` / `H_stra_max` →
    `h_stra_min` / `h_stra_max`, and `Z_mix` / `Z_free` / `Z_stra` →
    `z_mix` / `z_free` / `z_stra`; the attributes of the OPAC files keep
    their names
  - The `lam` parameter and attribute of `AlbedoSpectrum`,
    `smartg.atmosphere`, `smartg.rrs`, `smartg.vrs` and
    `smartg.histories` is now `wavelength`
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
  - The relative humidity of the `Atm1D` profiles (`relative_humidity`,
    formerly the `RH` method), which sets the hygroscopic growth of the
    OPAC aerosols, is computed with the new `saturation_pressure` (Huang
    2018), over ice below 0 °C, where v1.2.0 took the saturation over
    liquid water at every temperature. The humidity is unchanged above
    0 °C and higher below: in `afglsw`, 80.7 → 94.4 % at the surface and
    23.7 → 39.0 % at 8 km, and the column single scattering albedo of
    `continental_average` at 550 nm goes from 0.919 to 0.946; in
    `afglus`, 50.6 → 72.5 % at 8 km
  - The internal data structures have been migrated from the legacy
    LUT/MLUT objects to xarray in most of the package (atmosphere,
    smartg, water, reptran, postprocess, views); the albedo classes
    (their `data` and `map` attributes) and the solar spectrum returned
    by `smartg.bandset.spectral_grids` are still LUTs. The new
    `smartg.xarray` module provides `dataarray_to_lut` /
    `dataset_to_mlut` converters for backward compatibility, and
    `drop_axes`, the equivalent of the `MLUT.dropaxis` method
  - The tracked notebooks and tests no longer use LUT/MLUT either: the
    demo notebook selects and plots straight from the run Dataset
    instead of converting it back with `dataset_to_mlut`, and the phase
    matrices they build by hand are `xr.DataArray` objects
  - `Smartg.run` returns an `xr.Dataset` instead of an MLUT. The
    variable names, their order, the coordinates and the attributes
    are unchanged, the dimensions which were anonymous in the
    MLUT are named (`sensor_in` / `wavelength_in`, `cdist_layer`,
    `hist_*`), and those of the phase matrices are renamed (see
    below);
    `smartg.xarray.dataset_to_mlut` converts the output back to an MLUT
  - The `mixture` attribute of `AerOPAC`, `Cloud` and `AerUser` is now
    `ds_mix`, and holds the bulk optical properties as an `xr.Dataset`
    instead of an MLUT
  - The functions of `smartg.atmosphere` now raise `ValueError` instead of
    `NameError` on invalid inputs
  - `Grid3D`, `create_1d_grid` and `extend_1d_grid` of `smartg.grid3d`
    raise `ValueError` instead of `NameError` on invalid values (a grid
    that is not 1-D or not sorted, `periodic` with `horiz_extend_length`,
    a `vert_extend_limit` within the grid, an unknown `loc` or `type`)
  - The `phase` module has been moved from `smartg/tools/` to `smartg/`
    -> import from `smartg.phase` instead of `smartg.tools.phase`
  - `read_phase` and `convert_phase_to_iparper` (formerly
    `pha2Iparperconv`) have been moved from `smartg.atmosphere` to
    `smartg.phase`, where the new `read_phase_dat`, `read_phase_nc` and
    `read_phase_cdf` readers live too
  - `read_phase_dat`, `read_phase_nc` and `read_phase_cdf` now always return a
    4-D `xr.DataArray` (dims: `wavelength_phase`, `z_phase`, `nphamat`,
    `theta_atm/oc`). The `nphamat` dimension is no longer squeezed when its
    size is 1.
  - The `standard` parameter of the phase readers has been removed:
    files are expected in the standard IQUV convention, `run` doing the
    conversion into the parallel/perpendicular convention of the kernels
  - The phase-matrix term dimension is now named `nphamat` internally
    (was `stk`; the auxdata files keep `stk`, which is renamed on load),
    and the `iphase` / `stk` dimensions of `phase_atm` and `phase_oc` in
    the run output are now `phase_index_atm` / `nphamat_atm` and
    `phase_index_oc` / `nphamat_oc`
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
    `Cloud.phase()` and `Atm1D.calc()`. The IQ → Ipar/Iper conversion is
    now performed automatically inside the `run()` method (only for
    atmospheric phases).
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
    - `Profile_base`     → `ProfileBase`, with PEP 8 parameters:
      `atm_filename` → `fname`, `O3` → `tco3`, `H2O` → `tcwp`, `NO2` →
      `tcno2`, `P0` → `p0`, `RH_cst` → `rh_cst`, `O3_H2O_alt` →
      `o3_h2o_alt`, and the unused `US` removed
  - The Aeronet readers of `smartg.atmosphere` follow PEP 8:
    `read_Aeronet_AOD` → `read_aeronet_aod`, `read_Aeronet_SSA` →
    `read_aeronet_ssa` and `read_Aeronet_PFN` → `read_aeronet_pfn`
    (they now return an `xr.DataArray`, see New features)
  - `atm_pro_from_aeronet` has been rewritten: `b_wav` →
    `b_wavelength`, `pfwav` → `wavelength_phase`, `z_profil` → `grid`,
    `P0` / `O3` / `H2O` / `O3_H2O_alt` → `p0` / `o3` / `h2o` /
    `o3_h2o_alt`; the `dens` aerosol profile is replaced by the new
    `h_mix_min`, `h_mix_max` and `z_mix` of the `AerUser` component it
    builds, and `fill_value_time` is gone. The
    `atm_pro_from_aeronet_opti` and `atm_pro_from_aeronet_opti2`
    variants have been removed
  - `blackbody_radiance` takes `temperature` (was `T`)
  - Several functions of `smartg.tools.smartg_view` have been renamed or
    replaced:
    - `transect2D` → `transect_2d`, which takes an `xr.DataArray`
    - `plot_polar`, which was the LUT function of `luts` imported there,
      is now an own function taking an `xr.DataArray`
    - their parameters follow PEP 8: `mlut` → `ds_sg` (`mref` →
      `ds_ref` in `compare`, `lut` → `da` in `transect_2d` and
      `spectrum`), `logI` → `log_i` (in `mdesc` too), `QU` → `qu`,
      `Circ` → `circ`, `Imin` / `Imax` / `Pmin` / `Pmax` → `i_min` /
      `i_max` / `p_min` / `p_max` (which the new `interp_dict`
      parameter of `smartg_view` now precedes), and in `compare`
      `U_sign` → `u_sign`, `same_U_convention` → `same_u_conv`,
      `U_symetry` → `u_symetry`, `Nparam` → `nparam`,
      `same_azimuth_convention` → `same_azi_conv` and `SZA_MAX` →
      `sza_max`
  - The `smartg.tools.smartg_view` module (then `smartg.smartg_view`) has
    been renamed to `smartg.view`
  - `smartg_view`, `transect_view`, `spectrum_view`, `profile_view` and
    `phase_view` now expect an `xr.Dataset` instead of an MLUT (MLUT
    still accepted with a deprecation warning)
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
    - `Grid3D` and the grid helpers of `smartg.libATM3D` (`is_sorted`,
      `is_same_cell_size`, `create_1d_grid`, `extend_1d_grid`,
      `Get_3Dcells`, `Get_3Dcells_indices`, `Get_3Dcells_neighbours` and
      `locate_3Dregular_cells`) have been moved to the new
      `smartg.grid3d` module, which also holds the new
      `locate_voxel_index`. The voxel geometry helpers are now
      `get_3d_cells`, `get_3d_cells_indices`, `get_3d_cells_neighbours`
      and `locate_3d_regular_cells`, with lowercase parameters (`nx`,
      `ny`, `nz`, `dx`, `dy`, `dz`, `boundary_abs`, `boundary_boa`,
      `boundary_toa`, `horiz_extent_length`, `sat_altitude`); the
      `Grid3D` attributes keep their names
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
    new `smartg.sensor` module. Only `Sensor` can still be imported from
    `smartg.smartg`: the imports of `get_sensor` and `LOC_CODE` must be
    updated. The parameters of `get_sensor` follow PEP 8: `VZA_lev` →
    `vza_level`, `LEVEL` → `level`, `VAA` → `vaa`, `RTER` →
    `earth_radius`, `H` → `height_toa`, `FOV` → `fov`, `TYPE` →
    `sensor_type` and `PP` → `pp`
  - The surface classes `FlatSurface`, `RoughSurface`, `LambSurface`,
    `RTLSSurface`, `RPVSurface` and `Environment` have been moved from
    `smartg.smartg` to the new `smartg.surface` module; they are NOT
    re-exported by `smartg.smartg` (which imports `Environment` for its
    own use only), so imports must be updated. The albedo classes
    (`AlbedoCst`, `AlbedoSpeclib`, `AlbedoSpectrum` and `AlbedoMap`) are
    no longer re-exported by `smartg.smartg` either (which imports
    `AlbedoMap` for its own use only): import them from
    `smartg.albedo`. The constructor parameters of the surface classes
    and `Environment` have been renamed to snake case (`SUR` → `sur`,
    `NH2O` → `nh2o`, `WIND` → `wind`, `WAVE_SHADOW` → `wave_shadow`,
    `BRDF` → `brdf`, `SINGLE` → `single`, `ALB` → `alb`, `ENV` → `env`,
    `ENV_SIZE` → `env_size`, `X0` → `x0`, `Y0` → `y0`, `NENV` → `nenv`,
    `NXENVMAP` → `nxenvmap` and `NYENVMAP` → `nyenvmap`; those of
    `RTLSSurface` and `RPVSurface` were already lower case), and the
    `Environment` attributes `NENV`/`NXENVMAP`/`NYENVMAP` are now
    `nenv`/`nxenvmap`/`nyenvmap`
  - The `CusForward` and `CusBackward` launching-mode classes have been
    moved from `smartg.smartg` to `smartg.objects3d`; they are NOT
    re-exported by `smartg.smartg`, so imports must be updated. Their
    constructor parameters have been renamed to snake case, with `TYPE`
    → `sampling` (`type` would shadow the builtin), and those of
    `CusBackward` spell out what they carry:
    - `CusForward`: `CFX` → `cfx`, `CFY` → `cfy`, `CFTX` → `cftx`,
      `CFTY` → `cfty`, `CFTZ` → `cftz`, `FOV` → `fov`, `TYPE` →
      `sampling`, `LMODE` → `mode`, `LPH` → `lph` and `LPR` → `lpr`
    - `CusBackward`: `POS` → `position`, `THDEG` → `th_deg`, `PHDEG` →
      `ph_deg`, `V` → `normal`, `ALDEG` → `receiver_fov`, `REC` →
      `receiver`, `TYPE` → `sampling`, `LMODE` → `mode`, `LPH` → `lph`
      and `LPR` → `lpr`. `normal` also accepts a `Normal` now, converted
      to a `Vector`
    - The keys of the `dict` attribute of both classes are snake case now,
      and named after the constructor parameters they carry: `CFX` →
      `cfx`, `CFY` → `cfy`, `CFTX` → `cftx`, `CFTY` → `cfty`, `CFTZ` →
      `cftz`, `FOV` → `fov`, `POS` → `position`, `THDEG` → `th_deg`,
      `PHDEG` → `ph_deg`, `ALDEG` → `receiver_fov`, `REC` →
      `receiver`, `LMODE` → `mode`, `LPH` → `lph` and `LPR` → `lpr`,
      with `TYPE` → `sampling_code` (it holds the code, not the
      `sampling` string); `CusBackward` also has the `v_sun` and
      `sun_fov` keys of its new parameters. The `ALDEG` attribute of the
      output dataset keeps its name
  - The parameters of `Smartg` and `Smartg.run` follow PEP 8. Constructor:
    `obj3D` → `obj3d` and `opt3D` → `opt3d`. `run`:
    - `NBPHOTONS` → `n_photons`, `NBLOOP` → `n_loop`,
      `NBTHETA` → `n_theta`, `NBPHI` → `n_phi`, `NF` → `n_icdf`,
      `wl_proba` → `wavelength_proba`
    - `THVDEG` → `th_deg`, `PHVDEG` → `ph_deg`, `SEED` → `seed`,
      `RTER` → `earth_radius`, `DEPO` → `depol`, `DEPO_WATER` → `depol_water`
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
  - `StdevLim(stk)` → `StdevLim(stokes)`, the spelling of the rest of the
    package, and its `dict` key with it
  - The abbreviated parameters of `Smartg.run` have been given their full
    name: `atm` → `atmosphere`, `surf` → `surface` and `env` → `environment`
  - `THVDEG` and `PHVDEG` of `Smartg.run` became `th_deg` and `ph_deg`,
    without the `v` of a viewing direction: they are the sun angles in
    forward mode and the viewing angles in backward mode
  - `NF` of `Smartg.run` became `n_icdf`, after the `icdf` and `icdf_2d`
    helpers: it is the number of points of the inverted functions it
    sizes (the phase functions and the wavelength probability)
  - The depolarization is spelled `depol` throughout, the physics term and
    the spelling the `smartg.iprt` modules already used: `DEPO` and
    `DEPO_WATER` of `Smartg.run` are now `depol` and `depol_water`, as are
    the depolarization parameters of the internal `_rayleigh`,
    `_calc_phase_host` and `_calc_phase_gpu` helpers
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
  - The internal helpers of the smartg module are now private:
    `finalize` → `_finalize`, `calcOmega` → `_calc_solid_angles`,
    `isotropic` → `_isotropic`, `rayleigh` → `_rayleigh`, `calculF` →
    `_calc_phase_gpu`, `InitConst` → `_init_const`, `init_profile` →
    `_init_profile`, `loop_kernel` → `_loop_kernel`, `get_git_attrs` →
    `_get_git_attrs`, `impactInit` → `_impact_init`, `init_rng` →
    `_init_rng`, `RNG_PHILOX` → `_RngPhilox`, `RNG_CURAND_PHILOX` →
    `_RngCurandPhilox`, `initObj` → `_init_obj`, `normalizeRecIrr` →
    `_normalize_rec` and `findExtinction` → `_find_extinction`. Its
    module-level constants follow PEP 8: `dir_src` → `DIR_SRC`,
    `src_device` → `SRC_DEVICE`, `type_Phase` → `TYPE_PHASE`,
    `type_Spectrum` → `TYPE_SPECTRUM`, `type_EnvMap` → `TYPE_ENV_MAP`,
    `type_Profile` → `TYPE_PROFILE`, `type_Cell` → `TYPE_CELL`,
    `type_Sensor` → `TYPE_SENSOR`, `type_Spectrum_obj` →
    `TYPE_SPECTRUM_OBJ`, `type_IObjets` → `TYPE_IOBJECTS` and
    `type_GObj` → `TYPE_GOBJ`. The `LUT` re-export and the unused
    `src_kernel2` path have been removed
  - A `grid` given to `Atm1D` as a string is now converted by the new
    `smartg.atmosphere.strgrid_to_numpy`, whose docstring gives the
    format, instead of the external `change_altitude_grid` function,
    which only an unshipped `smartg.tools.third_party_utils` module
    provided
  - The unused `lib3D` module, a copy of the 3D grid helpers of
    `smartg.libATM3D`, has been removed
  - Several obsolete utility functions removed: `average`, `isiterable`,
    `isnumeric`, `vapor_pressure`, `trapzinterp`, `generatePro_multi`,
    `conv_pha3D_to_pha4D`, `compute_AB_coeff`, `get_AB_coeff`,
    `get_AB_coeff2` and the `Profile_base2` class of
    `smartg.atmosphere`, `compare_spectrum` of `smartg.tools.smartg_view`,
    and `convertVtoAngles`, `convertAnglestoV`, `Analyse_create_entity`,
    `random_equal_area_geometries` and `packed_geometries` of
    `visualizegeo`
  - The `fournierForand`, `fournierForandB` and `henyeyGreenstein`
    functions of `smartg.tools.phase` have been removed -> use
    `pytrunc.phase.fournier_forand` and `pytrunc.phase.henyey_greenstein`
    (pytrunc >= 2)
  - The `ang_trunc` parameter of the `IOP`, `IOP_1` and `IOP_profile`
    classes has been replaced, in the `Hydrosol`, `HydrosolPR` and
    `HydrosolZhai` classes that took over, by `truncation`
    (`DMTrunc | GTTrunc | None`): the water
    phase functions are truncated with pytrunc like the atmospheric ones, and
    the scattering coefficient is scaled by `1 - f`. The truncation is only
    applied when asked for: the default is `None`, no truncation, where
    v1.2.0 always cut the forward peak below 5 degrees. Pass
    `truncation=DEFAULT_WATER_TRUNC` (`GTTrunc(trunc_frac=0.3, theta_tr=5.0)`,
    the recommended one) to truncate the derived phase functions
  - The phase matrix truncation is carried by the components: the
    `truncation` argument of `Atm1D.calc` and `Atm3D.calc` has been
    removed, and `AerOPAC`, `Cloud`, `AerUser`, `Cloud3D` and `Aer3D` take
    a `truncation` parameter (`DMTrunc | GTTrunc | None`, default `None`),
    like the hydrosols
    - only the components given a truncation are truncated; the others,
      such as a smooth aerosol mixed with a cloud, keep their phase matrix
    - each component is truncated alone, before the components of a layer
      or of a cell are mixed, and its own scattering is scaled by `1 - f`
      while its absorption is kept; the phase matrices are mixed with the
      truncated scattering coefficients as weights
    - several components may truncate differently, which the "Only one
      truncation factor is supported" error used to refuse
    - a component alone gives the profile the `truncation` argument gave,
      to float32 rounding (the IPRT C2 profiles are identical); a mixture
      differs, e.g. the 1D aerosol of the IPRT C3 case is no longer
      truncated with its cloud
    - `calc_split` returns the truncated profile, and `calc(phase=False)`
      truncates nothing, as for the hydrosols
    - `Atm1D.profile` takes the truncated fraction of each component as a
      keyword-only `comp_trunc_frac`, which `calc` fills; called without
      it, `profile` truncates nothing, as before
    - a truncated 1D component is refused with a forced particle profile
      (`prof_aer` of `Atm1D`, `aer_ext_1d`, `aer_ssa_1d` or `aer_phase_1d`
      of `Atm3D`), which would stay untruncated
    - a 3D component is truncated once per distinct phase matrix instead of
      once per cell: 95 pytrunc calls instead of 20489 for the C3 cloud
    - a truncation leaving a negative phase function raises a `ValueError`,
      as it already did for the hydrosols, instead of reaching the profile
      unnoticed: GT on a phase function without a marked forward peak (a
      continental aerosol), or Delta-M with too few streams
    - no component is truncated unless asked for: `truncation=None`, the
      default of the atmospheric components as of the hydrosols, is no
      truncation, and anything else than a `DMTrunc`, a `GTTrunc` or `None`
      (a boolean included) raises a `TypeError` when the component is built
  - A `Hydrosol` given its own phase matrices (`phase=`) now truncates them
    too when given a `truncation`, as the derived ones, and scales its
    scattering coefficient by `1 - f`. A phase function without a marked
    forward peak cannot be truncated: the truncation would leave it
    negative, and is refused
  - The `show_trunc` option of `smartg.view.phase_view` has been removed: it
    read a `phase_atm_tr` / `phase_oc_tr` variable that no profile carries
    any more, the profile holding only the (truncated) matrices the
    simulation uses. Plot the profile computed without truncation and the
    truncated one on the same axes instead (`fig` and `axarr` arguments)
  - The `force_4stk` option of `smartg.view.phase_view` is now
    `force_4stokes`, following the `stk` → `stokes` renames
  - `Smartg.run` refuses, with a `ValueError` or a `TypeError` saying
    why, the arguments it used to ignore or misread: an unknown `flux`
    (it then counted radiances), `flux` together with `le` (the local
    estimate was dropped), `alis_options` on a `Smartg` built without
    `alis=True` (they had no effect), an `environment` without a
    `surface`, `wavelength_proba` or `sensor_proba` arrays that are not
    `int64`, an unknown `cell_proba` string, `cell_proba='auto'` outside
    the forward thermal mode and a `cell_proba` array without one column
    per wavelength. Some were assertions, the others went unnoticed
  - `smartg.kdis` raises a `FileNotFoundError` for a missing file and a
    `ValueError` for an unsorted or incompatible table, where it printed
    the problem and called `sys.exit()`, which also shut down the Jupyter
    kernel. An h5 file with unsorted wavelengths claimed that the h5
    format was not implemented for concentration dependent species
  - The declared dependencies have been trimmed and bounded. `pyarrow`,
    `pyhdf` and `statsmodels` are no longer declared, as no module nor
    notebook imports them (`pyhdf` still comes in as a dependency of
    `luts`). `gatiab`, which `smartg.atmosphere` now imports, has moved
    from the `extra` group to the required dependencies, and `extra`
    holds `jax[cuda12]`, `radis` and `hitran-api`, the last two for the
    photon histories notebook. The libraries whose API SMART-G calls
    directly are
    now capped at their next major version (`numpy>=2,<3`,
    `jupytext>=1.16,<2`, `geoclide>=4.0.0,<5`, `pytrunc>=2.0.0,<3`,
    `gatiab>=1.1.2,<2`), and the supported Python versions are
    `>=3.11,<3.15`: Python 3.10, which v1.2.0 supported
    (`requires-python >= 3.10`), is no longer supported

* New features
  - New `smartg.truncation.truncate_phase`, `truncate_phase_set` and
    `truncated_ext_ssa` truncate a phase matrix, or each distinct matrix
    of a set once, and rescale the extinction and the single scattering
    albedo of the truncated particles; a null matrix passes through with
    `f = 0`
    - `truncate_phase` normalizes an F11 normalized otherwise than to 2 by
      more than 1 % (to 4 pi, or a volume scattering function) before the
      truncation, as pytrunc expects, and gives the truncated matrix back
      in the normalization of its input
    - `as_truncation` returns the `truncation` given to a component once
      checked, raising a `TypeError` for anything else than a `DMTrunc`, a
      `GTTrunc` or `None`
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
    `phase` argument of `AerOPAC` / `Cloud` / `Hydrosol` and, through
    `get_prof_phases`, `Atm1D.prof_phases`; False returns the table on the
    wavelength and `hum` / `reff` axes of the file, which the `phase`
    argument of `Cloud3D` / `Aer3D` takes, and which `read_phase_nth_cte`
    alone used to produce. A `.dat` file, a single matrix, has only the
    first layout
    - `read_phase_cdf` gains `n_theta`: `None` keeps the automatic
      equally spaced grid capped by `n_theta_max` (`ntheta_max` in
      2.0.0b1), an int or the angles themselves choose the grid, and
      `'native'` resamples onto the union of every grid the file
      carries (2818 angles for the 25 radii of the IPRT
      `watercloud_670.mie.cdf`, 38 for `waso_670.mie.cdf`), on which
      the file is reproduced exactly
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
      OPAC aerosol mixtures or species (such as 'desert',
      'continental_clean' or 'waso') as a function of the relative
      humidity; the 3D distribution (extinction at `w_ref` and per-cell
      rh, clamped to the file's humidity range as in the 1D `AerOPAC`)
      follows the same three routes as `Cloud3D` (dense dataset with
      `rh(z, y, x)`, raw arrays, ASCII files via the new
      `read_i3rc_aerosol` function), with the `rh_acc`/`rh_min`/`rh_max`,
      `phase` and `ssa_cst` options
  - New `AerUser` class in `smartg.atmosphere` to define custom aerosol / cloud
    optical properties (extinction, SSA, phase matrix) from user-supplied data
  - New `get_prof_phases` utility function to easily extract phase
    matrices from an existing simulation profile
  - `prof_phases` parameter of `Atm1D` now also accepts `xr.DataArray` objects
    in addition to LUT objects
  - `read_phase` is now a dispatcher accepting `.dat`, `.nc` and `.cdf`
    files
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
  - The Aeronet readers `read_aeronet_aod`, `read_aeronet_ssa` and
    `read_aeronet_pfn` now return an `xr.DataArray` instead of a LUT
  - New `smartg.atmosphere.saturation_pressure`, the saturation vapour
    pressure in Pa of Huang (2018), over liquid water above 0 °C and
    over ice below, used by `relative_humidity`
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
      download them again), accepts a list of keys, and its `savepath`
      parameter is now `dname`, which defaults to `SMARTG_DIR_AUXDATA`
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
    - `AUXDATA_DICT`, `safe_download` and the per-dataset `*_URL`
      constants (`AER_URL`, `ACS_URL`, `ATM_URL`, `STP_URL`,
      `VALID_URL`, `WATER_URL`, `KDIS_URL`, `CLOUD_URL`, `IPRT_URL`,
      `REPTRAN_URL` and `REPTRAN_URL_HYG`) are removed: the datasets are
      described by `DATASETS`, and the new `LIBRADTRAN_REPTRAN_URL`
      holds the address of the libRadtran reptran archive
  - A ruff configuration in `pyproject.toml`: a line length of 79 and the
    PEP 8, naming, numpy docstring and annotation rules on top of the
    default ones, for the whole package, which passes them along with
    pyright: every module, test and demo notebook is documented in
    numpydoc style, type-hinted (`Smartg.run` and the profile builders
    state what they accept, `smartg.surface.SurfaceLike` names the
    surface classes) and reflowed to 79 columns of code and 72 of
    prose. Along the way the type checks of `smartg.objects3d`,
    `smartg.smartg`, `smartg.view` and `smartg.grid3d` raise
    `TypeError` where they raised `NameError` or a bare `Exception`,
    and `smartg.view.mdesc` accepts a name without the space before
    the level. `smartg/obselete_files`, whose unused Python 2 modules
    no longer parse, is excluded from ruff and from pyright
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
  - Fix the pytest plugin `smartg/conftest.py`, which stopped pytest with
    an `INTERNALERROR` (unknown hook `pytest_html_results_summary`) when
    pytest-html, an optional test dependency, is not installed: its
    pytest-html hooks are now optional
  - Fix `GTTrunc`, which accepted any `theta_tr`: 0, a negative or NaN
    angle, or one below half the first angle step of the phase matrix,
    left the phase matrix unchanged but still removed `trunc_frac` of the
    scattering, and 180 or more gave an unrelated pytrunc error. `GTTrunc`
    now requires `theta_tr` in ]0; 180[ and `truncate_phase` refuses an
    angle nearest to the first angle of the grid, with a `ValueError`
  - Fix `smartg.cdf.icdf` with `n=None`, which raised an `OverflowError`
    when a probability after the first one was zero, and sized `n`
    without the first probability, which could then get no sample.
    `n` now gives the smallest non-zero probability, the first one
    included, 10 samples, and a `pdf` that cannot size it raises a
    `ValueError`
  - Fix `smartg.phase.read_phase_cdf`, `read_phase_nc` and `read_phase`
    with a `wavelength_phase` outside the wavelengths of a file that has
    several: they returned an all-NaN matrix, which `Atm1D` turned into a
    null phase matrix while keeping the optical depth of the component.
    They now raise a `ValueError` giving the range of the file
  - Fix the memory use of `smartg.phase.read_phase_cdf` with the profile
    layout: it resampled every entry of the file before interpolating at
    `wavelength_phase` and `z_rh_reff`, 8.8 GB for the IPRT ice cloud file
    at the default angles. Only the entries around the targets are
    resampled now, with the same result
  - Fix the panel titles of `smartg.view.compare`, garbled since
    v2.0.0dev1 (`$^{\downarrow}_{}I$` for `I_up (TOA)`): they are built
    from the compared variable again, for instance `$I^{\uparrow}_{TOA}$`
  - Fix `smartg.view.profile_view` and `input_view`, which since
    v2.0.0dev1 sliced the wavelength axis of the phase matrix index: the
    index axis was left empty, or showed one line per wavelength when
    there were as many wavelengths as levels. It shows again the index
    profile of wavelength `iw`
  - Fix `smartg.view.camera_view`, and so the IPRT phase B camera plots,
    which raised a `ValueError` for a panel whose values are all zero
    (for instance V of a Rayleigh scene) or all NaN. Given `matrices`,
    `camera_view` now requires one `stokes` label per matrix and says so,
    instead of raising an `IndexError` with the default `stokes='I'`
  - Fix the import of `smartg.view`, and so of the `smartg.iprt` phase
    modules, which called `np.seterr(invalid='ignore', divide='ignore')`
    and silenced these NumPy warnings for the whole process. Only the
    plotting functions that divide by values that may be zero ignore
    them now, and they restore the caller's NumPy error state
  - Fix `smartg.environ.modified_environ` with a value that is not a
    string, which raised a misleading `KeyError` on exit: it now raises a
    `TypeError` naming the variable before changing the environment. Its
    docstring no longer says that the values are converted to strings
  - Fix the terminal progress bar of `Smartg.run`, which never showed its
    messages (photons launched, error estimate, received fraction): they
    went to a copy of the label that progressbar2 does not draw. A `%` in
    a message is now shown as is. A terminal IPython shell is no longer
    taken for a notebook, where the widget was printed once and never
    updated
  - Fix `smartg.postprocess.plane_irr` and `spherical_irr`, and so
    `irradiance_ds`, which applied the trapezoid rule to the bin centres
    of a run without `le` and dropped half a zenith bin at each end and
    one azimuth bin: the irradiances were 1.2 % and 2.9 % low with the
    default 45 x 90 bins. Each bin is now weighted by its exact solid
    angle, which gives back the flux of the counted photons; other grids,
    such as a local estimate's, keep the trapezoid rule. `irradiance_ds`
    no longer integrates the `I_stdev_*` standard deviations as radiances
  - Fix `Smartg.run(stdev=True)` for a sensor or a wavelength that gets no
    photon in some kernel loops, as with many sensors or a
    `wavelength_proba`: each loop was normalised by its own photon count,
    so that its `_stdev_` outputs were NaN. The standard deviation is now
    the one of the ratio of the weights to the photons over all the loops,
    the same as before when every loop launches the same photons. Fix
    also `stdev_lim` on a level or a Stokes component without any signal,
    for instance the default top of the atmosphere with `output_layers=4`:
    its zero error stopped the run after `n_loop_min` loops. Such a slice
    no longer stops the run. The errors were already in v1.2.0
  - Fix the `direct transmission (dev)` output with `Smartg(double=False)`:
    the kernel wrote single precision values into a double precision
    buffer, read as about 1 whatever the optical depth. The buffer now
    has the precision of the kernel, and always one value per sensor and
    wavelength: without atmosphere it held one, which the kernel overran
    with several sensors or wavelengths. The analytic `direct
    transmission` is no longer written for a 3D atmosphere, where it was
    `exp(-k/cos(th_deg))` of the extinction coefficient `k` of its last
    optical property, and `Smartg(opt3d=True, pp=False)`, which the kernel
    does not support and which stopped at a bare `AssertionError`, raises
    a `ValueError`. The errors were already in v1.2.0
  - Fix `multi_profiles`, which shifted the phase matrix indices of each
    profile by the largest index it used plus one instead of by its
    number of phase matrices: when a profile did not use all of its
    matrices, as with a `wavelength_phase` grid wider than the run's
    wavelengths, the next profiles scattered with the matrices of the
    previous ones. Its docstring no longer claims to convert DataArray
    inputs, which raise a `TypeError`. The error was already in v1.2.0
  - Fix `Smartg.run` with fewer than 30 photons (10 with 3D objects),
    which never returned: the default `n_loop`, `n_photons/30`, reached
    the kernel as 0, which launches no photon. It is now at least 1, and
    `n_photons` or `n_loop` below 1 raises a `ValueError`. The error was
    already in v1.2.0
  - Fix `Smartg.run` with a negative `depol` or `depol_water`, an
    undocumented switch to an unvalidated "isotropic" molecular phase
    matrix, which kept Q and multiplied U and V by sqrt(2) at every
    scattering: both now raise a `ValueError`. The isotropic matrix of the
    internal phase tables now keeps the intensity and does not polarize.
    The error was already in v1.2.0
  - Fix the `device` parameter of `Smartg`, ignored with `autoinit=False`,
    whose context went to the default device, and, silently, by every
    `autoinit` Smartg after the first of the process, which all run on
    the device of the first. The context of `autoinit=False` is now
    created on `device`, and an `autoinit` Smartg asking for another
    device than the one of `pycuda.autoinit` raises a `ValueError`, as
    `device` together with the environment variable `CUDA_DEVICE` now
    does instead of an `AssertionError`
  - Fix the default sensor of `Smartg.run` (without `sensor`) for a
    non-zero `ph_deg`: it started from the entry point of azimuth 0, so
    that its ray missed the origin. In spherical mode the sun reached the
    ground at another zenith angle (61.9 instead of 60 degrees for
    `ph_deg=90`) or missed the Earth, and with an `Environment` the direct
    beam landed away from the target (98 km away for `th_deg=30`,
    `ph_deg=90`). The kernel of the 3D objects already turned it; the
    plane-parallel runs without environment do not change. The error was
    already in v1.2.0
  - Fix the `Sensor(cell_size=-2)` of a spherical run with
    `Smartg(obj3d=True)` but without 3D objects: the altitude of the top
    of the atmosphere was only sent to the kernel with objects, so that
    these sensors started on the ground sphere instead. The lines of sight
    crossing the atmosphere above the Earth's limb were lost, and the
    other ones started on the ground, missing the path down to it through
    the atmosphere. The error was already in v1.2.0
  - Fix the forward thermal emission without `cell_proba`
    (`Smartg(thermal=True)`, still in development): the emitting layer
    was drawn among the layers 0 to NATM-1 instead of 1 to NATM, so that
    the bottom layer never emitted and about one photon in NATM read the
    profile before its first level. The error was already in v1.2.0
  - Fix `Smartg(rng='CURAND_PHILOX')`: `curand_uniform` returns exactly 1
    for about 3e-8 of its draws, which made the sensor, wavelength or
    icdf index one past the end of its array (a misplaced count, a NaN or
    an illegal memory access), and the optical depth to the next event
    infinite. Its draws now stay below 1, as the ones of the default
    `PHILOX`. The error was already in v1.2.0
  - Fix the forward runs in a 3D atmosphere (`opt3d=True`, `back=False`)
    whose sensors are not the complete raster, x varying first, that the
    kernel counts the photons leaving the domain on: the photons leaving
    outside it were counted on the edge sensors or in the next row, and
    the kernel printed a line for each of them. `Smartg.run` now raises a
    `ValueError` for such sensors (`get_sensors_grid` builds the raster),
    and the photons leaving outside the raster are not counted. The
    error was already in v1.2.0
  - Fix the seafloor of `Environment(env=5)` with water below a land cell
    of the `AlbedoMap`, reached by photons travelling under the coast:
    the kernel read the albedo before the start of the albedo list, 0 or
    another field of the spectrum, and overwrote the photon's counter of
    environment reflections. The seafloor there now keeps the albedo of
    the water profile. Below a water cell coded -k it is the albedo k of
    the list, as before; this convention is now documented in `AlbedoMap`
    and `Environment`, and `Smartg.run` raises a `ValueError` for a code
    outside the list. The bathymetry section of `demo_notebook_objects`
    gave its ocean cells a black seafloor instead of the sand it
    describes
  - Fix the albedo map of `Environment(env=5)` in spherical mode
    (`pp=False`): the distance to the origin of the map was the arc cosine
    of a single precision cosine, which rounds to 1 near the origin, so
    that the points within about 2 km of it were read at the origin, the
    ones a little further at about 3.1 km, or in the last cell of the map.
    The error decreased with the distance, to tens of metres beyond 50 km.
    It was already in v1.2.0
  - Fix `Smartg.run(sun_disc=...)` with the cone sampling at the downward
    levels (`down (0+)`, `down (0-)`, `down (B)`): each box was centred on
    an upward direction, so that no downward photon but the ones near the
    horizon was counted and those radiances were zero. `sun_disc` is also
    set by a planar flux `Sensor` with a `fov`. The error was already in
    v1.2.0
  - Fix `Smartg.run(sza_max=...)` with the cone sampling (without `le`)
    when `sza_max` is not 90: the radiances were multiplied by
    `1 - cos(sza_max)`, halved at 60 degrees, and the kernel binned every
    level in plane-parallel mode, and every level but the top of the
    atmosphere in spherical mode, over 0 to 90 degrees while the output
    labelled the boxes 0 to `sza_max`. The boxes now span 0 to `sza_max`
    at every level, the photons leaving beyond are not counted, and the
    radiances no longer depend on `sza_max`. The error was already in
    v1.2.0
  - Fix the reflection on `RoughSurface(brdf=True)` in a 3D atmosphere
    (`opt3d=True`): the photon left the surface from the cell whose index
    is the number of optical properties instead of the cell it reached the
    surface in, so that the reflected photons were lost or attenuated
    along a wrong path. The error was already in v1.2.0
  - Fix the altitude of the photons in the default plane-parallel move
    (`pp=True` without `alt_pp`): after each move in the atmosphere a
    photon was placed at the altitude mirrored inside its layer, while its
    optical depth was right. The 1D radiances only depend on the optical
    depth and do not change; the horizontal positions of the surface hits
    do, and with them the adjacency effect of an `Environment` and the
    runs with 3D objects and an atmosphere (`obj3d=True`). The error was
    already in v1.2.0
  - Fix the opaque `AttributeError: 'Spheric' object has no attribute 'p1'`
    that `Smartg.run` raised for a `Spheric` receiver, or a `Spheric`
    reflector in the RF mode, and `CusBackward` for a `Spheric` receiver
    in the BR mode. Both raise a `ValueError` saying that these objects
    must have a `Plane` geometry
  - Fix the STP extinction from TOA to the mean heliostat altitude, which
    gives the `n_tr` and `powc_H` outputs and so every efficiency of
    `nopt_view`: it interpolated the optical depth as if the level below
    the heliostats were at z = 0, and raised an `IndexError` after the run
    when the profile ended above them. It interpolates between the two
    levels around the heliostats, and a `ValueError` is raised before the
    run when they are outside the profile. Profiles whose level below the
    heliostats is at z = 0, such as the AFGL ones, are not affected
  - Fix `generate_h_p` and `generate_h_a` with a `heliostat_type`, which
    took its facets and sizes but silently dropped its reflectivity and
    roughness: the heliostats were perfect mirrors. Their `reflectivity`
    and `roughness` now default to None, which takes the values of the
    template when it is given, and 1 and 0 otherwise; values given to the
    generator still win
  - Fix the automatic bounding box of a rotated `Spheric` entity, built from
    two opposite corners of its local box only: a rotation that is not a
    multiple of 90 degrees gave a box too small, or flat at 45 degrees
    about z, and the kernel skipped the rays outside it, so that parts of
    the sphere were invisible. The box holds the eight transformed corners
  - Fix the validation of `Plane`, which blamed the signs of the corners of
    any invalid plane with a positive `p3.y`, and reported an "Unknown
    error" for a rectangle below the x axis. It raises a `ValueError`
    naming the violated condition, where it raised a `NameError`
  - Fix the copy of an `Entity` (`Entity(entity)`), which reset its
    `alpha_color` to 0.5 instead of copying it
  - Fix `smartg.objects3d.rotate_vector`, which raised a `NameError` with
    its default `rotation_order='xyz'` and with every lower case order its
    docstring shows. The order is read in either case, the default is
    `'XYZ'`, and an unknown order raises a `ValueError`
  - Fix `BandSet`, and so the `wavelength` of `Smartg.run`, `Atm1D.calc`
    and `Water1D.calc`, which refused an integer, a NumPy integer or
    `float32` scalar, a tuple or a DataArray with a bare `AssertionError`.
    Any real number, sequence or array of real numbers is accepted, a
    tuple of internal bands as a list is, and other input raises a
    `TypeError` saying what is expected
  - Fix `smartg.bandset.spectral_grids(unit='photons/cm2/s/nm')`, which
    converted the irradiance column of the caller's `datas` in place, so
    that a second call with the same array converted it again, about 1e11
    times too large. `unit` is the unit of the returned `es_lut`; `datas`
    is always in mW/m2/nm
  - Fix `smartg.bandset.spectral_grids` with 131 or more scattering
    wavelengths (for instance 150 nm at `dls=1`): the interpolation
    indices were `int8`, so NumPy 2 raised an `OverflowError`, and NumPy 1
    silently wrapped the indices above 127. They are now `int32`
  - Fix `smartg.vrs.raman_response` for a scalar wavenumber, which raised
    a `ValueError` since v2.0.0dev2, and for integer wavenumbers, which
    raised a `UFuncTypeError`; it returns an array of at least one
    dimension. The internal callers pass float arrays and were not
    affected
  - Fix `smartg.rrs.l2d`, which raised a `TypeError` on every call since
    v2.0.0dev2 (`np.atleast_1d` was given a `dtype`), and
    `smartg.rrs.bjm_minus`, which raised one for a scalar `j`, as its
    docstring allows. The Raman path of `Smartg.run` uses `l2d_inv` and
    was not affected
  - Fix `Smartg.run(cell_proba='auto')` (forward thermal mode): the
    probability of each level to emit took the Planck function at the
    wavelength in nm read as metres, where it is proportional to the
    temperature, so the levels emitted in proportion to k_abs T instead of
    k_abs B(lambda, T): the warm low levels were under-sampled and the
    cold high ones over-sampled (already so in v1.2.0)
  - Fix the KDIS models whose concentration axis is a density (the ascii
    models with a concentration dependent species, and the h5 ones whose
    `rho` is described as `density`): clipping the density to the axis
    also clipped the density that scales the absorption, so a layer
    without the gas absorbed as if it held 1.01 times the smallest
    tabulated density, and a denser layer than the largest was capped.
    The shipped models (kato, kato2, SENTINEL2_1_MSI) use molar fractions
    and are unchanged
  - Fix `cat_view(kdis_rep_bands=...)` with REPTRAN bands, and the weight
    sums returned by `ReptranIbandList.get_weights`: since v2.0.0dev2 they
    grouped by internal-band wavelength instead of by channel, so
    `cat_view` gave one value per internal band divided by its own weight,
    and merged the internal bands of two channels at the same wavelength.
    They sum the internal bands of each channel again, on the channel
    central wavelengths, as v1.2.0 did. KDIS bands are unchanged
  - Fix `reduce_reptran` and `reduce_kdis` on a run over a single internal
    band (24 of the 40 `reptran_solar_sentinel` channels, kato bands 5 to
    10...), whose output has no `wavelength` dimension: they raised a
    `KeyError` since v2.0.0dev2, and now return that band as its channel.
    Several internal bands without a `wavelength` dimension raise a
    `ValueError`
  - Fix `reptran_emission`, `kdis_emission` and their `*_avg_emission`:
    each internal band took the Planck average of the i-th channel in
    wavelength order, where i was its channel index in the file minus that
    of the first band, so the averages were right only for channels
    contiguous in the file and in wavelength order. Several sensors
    (`include='msg'`), a whole MODIS sensor or out of order channels got
    another channel's average, and several `lmin`/`lmax` intervals or
    `band_indices` with a gap raised an `IndexError`. Each internal band
    now takes the average over its own channel limits
  - Fix the bandwidth of the REPTRAN thermal channels: the `wvl_integral`
    of a thermal file is a wavenumber integral in cm-1, which
    `ReptranIbandList.get_weights` returned as a bandwidth in nm, so
    `reduce_reptran(integrated=True)` of a thermal run was 1.5 (MSG
    ch039) to 18 (ch134) times too small, and the limits derived for the
    channels whose name holds none (all the sensor channels) were as many
    times too narrow. The new `ReptranBand.dl` is the bandwidth in nm:
    `r_int` for a solar file, and for a thermal one, told by its lack of
    `extra`, the width of a named band or `r_int` converted to nm with the
    squared internal wavelengths. Solar files are unchanged
  - Fix a REPTRAN file given with its directory, `Reptran('/data/x.cdf')`:
    `ReadCrs` dropped the directory and read the lookup tables
    `x.lookup.<species>.cdf` from the auxdata REPTRAN folder, so
    `Atm1D.calc` raised a `FileNotFoundError` or silently used the tables
    of an auxdata file of the same name. A bare name is still looked up in
    the auxdata
  - Fix the order of `ReptranIbandList.get_names`, which changed from one
    Python process to the next (it came from a `set`), so the demo
    notebook labelled the reduced channels with the wrong names. The names
    are now sorted by channel central wavelength, the order of the
    `wavelength` axis of `reduce_reptran`
  - Fix the REPTRAN channel names, which kept the repr of the bytes read
    from the file (`"b'msg1_seviri_ch006'"`), so that
    `Reptran.band('msg1_seviri_ch006')` raised a `ValueError`. The names
    of `Reptran.band_names`, `ReptranBand.name` and
    `ReptranIbandList.get_names` are now plain text, still without spaces,
    and `Reptran.band` ignores the spaces of the name it is given
  - Fix the N2O absorption of REPTRAN: `ReptranIband.calc_profile` scaled
    the N2O cross sections by the NO2 density, about 1e4 times smaller in
    the AFGL profiles (and zero with `no2=False`), so N2O absorption was
    practically missing from the SWIR and thermal bands (already so in
    v1.2.0)
  - `Atm3D.calc(phase=False)` no longer computes the phase matrices and
    no longer truncates the components, as `Atm1D.calc(phase=False)`,
    which its docstring refers to: the flag had no effect, so the call
    cost the full phase construction and returned truncated extinctions
  - Fix the molecular share of the scattering (`pmol_atm`) of the `Atm3D`
    cells, which came from a float32 cumulated sum of the Rayleigh
    coefficients over every cell, differentiated back: its error grew
    with the number of cells (4e-4 relative with 10 000 cells, 1e-2
    absolute with millions). It is now computed from the coefficients
  - Fix a 4-term `aer_phase_1d` forced into `Atm3D`: the mixing kept the
    4 terms it shares with the 6-term 3D components, so that a
    non-spherical component (ice cloud, desert aerosol) lost its F22 and
    F44 in every mixed cell, or raised an xarray `AlignmentError` when
    the term axis had no coordinate. It is now completed to 6 terms
    (F22 = F11, F44 = F33), and the other term counts are refused
  - Fix the phase matrices `Atm3D` mixes in the cells shared by several
    3D components, or by a component and the 1D aerosols: normalized by
    the total extinction instead of the total scattering, they were
    scaled by the single scattering albedo of the cell particles, which
    the local estimate, reading the matrix as it is, carried into the
    radiance (11 % low for a desert aerosol cell next to another
    component), and which biased the ALIS mixture ratios. They are now
    normalized as in `Atm1D`. The spatial means of the IPRT C3 case 4
    change by 2e-4 in I and 1.6 % in Q, far within the test tolerance
  - Fix `Atm3D(wavelength_phase=...)` with a number of phase wavelengths
    other than the number of profile wavelengths, which raised a
    `CoordinateValidationError` whenever the profile carried phase
    matrices: each profile wavelength now takes the matrices of the
    nearest phase wavelength, as documented and as in `Atm1D`, where it
    took those of the phase wavelength of the same rank
  - Fix the periodic `Grid3D` with a single cell along x or y (a 2-D
    (x, z) field, or a horizontally uniform column): the faces of that
    axis were absorbing boundaries, which killed every photon crossing
    them, instead of wrapping the cell onto itself. The other grids are
    unchanged
  - Fix `extract_split`, which raised a `ValueError` on every `Smartg.run`
    output holding phase matrices: it indexed them along `iphase`, which
    the run output names `phase_index_atm`. It takes the first dimension
    of `phase_atm` whatever its name, and returns no phase profile (None)
    when there is no `phase_atm`
  - Fix `Atm1D.profile(wavelength, prof=...)` with REPTRAN or KDIS bands:
    the gas absorption was computed on the profile of the atmosphere
    instead of `prof`, which raised a broadcasting error, or took the
    absorption of other altitudes
  - Fix the `.dat` profiles without the libRadtran header line (or with
    one spelled differently): every gas column was read as zero, with no
    warning, removing the O3 and H2O absorption and drying the aerosols.
    The columns are now read in the libRadtran order, z(km) p(mb) T(K)
    air(cm-3) o3 o2 h2o co2 no2 (cm-3), with a warning saying so and
    naming the gases a shorter file lacks
  - `Atm1D.calc` refuses a `pfgrid` whose last level lies above the bottom
    of the profile grid with a `ValueError`: the layers below it got the
    phase index -1, which the kernel reads as the VRS phase function, and
    at the other wavelengths a matrix of the previous wavelength
  - `Atm1D` refuses a `grid` reaching above the top (or below the bottom)
    of its profile file with a `ValueError` naming them, unless `prof_ray`
    is given: the Rayleigh optical thickness of the levels beyond came out
    NaN from a 0 / 0 CO2 ratio, and spread through `OD_sca_atm`. That
    ratio is now guarded, as the one of the refractive index was
  - `Atm1D` refuses a `grid` that does not decrease strictly, from TOA to
    BOA, with a `ValueError`: an increasing one went through and gave NaN
    particle optical thicknesses. Such a `pfgrid` raises a `ValueError`
    instead of a bare `AssertionError`, and the parse errors of
    `strgrid_to_numpy` give a TOA to BOA example instead of an
    increasing one
  - `Cloud` refuses an effective radius outside the range of its file,
    and `AerOPAC` and `Cloud` refuse a wavelength, or the reference
    wavelength of a scalar `tau_ref`, outside the wavelengths of their
    tables, with a `ValueError`, as `Cloud3D` does and as v1.2.0 did for
    the wavelengths: they silently took the optical properties of the
    end of the tables (`Cloud('wc', 50., ...)` those of 30 um, a
    thermal infrared run those of 4.4 um). The relative humidity is
    still clamped to the tables, as in `Aer3D`. `atm_pro_from_aeronet`
    tabulates the aerosol at its `wavelength_phase` as well
  - Fix `Atm1D.calc_split(phase=False)`, and `calc_split` on an atmosphere
    without any component, which raised a `KeyError` on `iphase_atm`: the
    phase profile is then returned as None, which `Atm1D(prof_phases=...)`
    accepts. Its docstring now calls the profiles it returns the optical
    thicknesses of the layers, not coefficients
  - Fix the 2-D user `phase` of `AerOPAC` and `Cloud`, documented as
    constant vertically, which raised an xarray error in `Atm1D.calc` at
    several wavelengths or with a `pfgrid` of several layers: it is now
    the same matrix at every wavelength and in every layer. A 4-D user
    phase not over the `wavelength_phase` and `pfgrid` axes of the
    mixing, and a 2-D one not over `('nphamat', 'theta_atm')`, raise a
    `ValueError` saying so
  - Fix `AerOPAC('mineral_transported')`, documented as available, which
    failed on `float('None')`: its file gives no default layer heights.
    They must now be passed (`h_mix_min`, `h_mix_max` and `z_mix`), which
    the error says; a layer whose heights the file does not give is
    absent unless they are all passed. `AerOPAC.list` no longer returns
    the single OPAC species and the free troposphere and stratosphere
    layers, which lie in the same folder but are not mixtures
  - Fix the `ssa` of `AerOPAC` and `Cloud` forced by a DataArray over the
    wavelength and the altitude, as documented, which always raised: it is
    now interpolated onto every grid the component is evaluated on. A 1-D
    or 2-D array that does not match that grid (the `wavelength_phase` or
    the `pfgrid` of `Atm1D`) raises a `ValueError` saying so instead of a
    broadcasting error, and the docstrings say which form works where
  - Fix the aerosol phase matrix of `atm_pro_from_aeronet`, which was
    flipped in angle and not physical. `read_aeronet_pfn` returned the
    angles from 180 to 0 degrees, as the files list them, and `AerUser`
    resampled them as increasing, moving the forward peak to 180
    degrees; `read_aeronet_pfn` now returns them increasing, and
    `AerUser` sorts its angles. The scalar AERONET phase function was
    also turned into F21 = F34 = F11 and F33 = 0, fully polarizing every
    scattering; it is now the non-polarizing F11 = F22 = F33 = F44 and
    F21 = F34 = 0
  - Fix the `tau_ref` of `AerOPAC` and `Cloud` given as a list, a tuple or
    a 1-D array, documented as accepted: it was silently ignored, and the
    component kept the optical thickness of the OPAC number densities.
    One value is now taken as that value, and several raise a
    `TypeError`. A DataArray still forces the optical thickness at every
    wavelength, `w_ref` being ignored, which the docstring now says
  - Fix `refractivity`, which took the pressure in hPa instead of Pa in
    the leading factor of the Edlén equation: `n_atm - 1` was 100 times
    too small, so that the runs with `refraction=True` hardly refracted.
    Their results change
  - `HydrosolZhai`, and a `Hydrosol` whose scattering coefficient alone
    varies with depth, tabulate their phase matrices on a single depth when
    they are the only scattering hydrosol, instead of one identical matrix
    per wavelength and depth (510 matrices of 7201 angles, 176 MB, for 10
    wavelengths and 51 levels)
  - Fix a `Hydrosol` given a 1-D coefficient with as many values as there
    are wavelengths and depths, which was read silently as a depth profile
    where a spectrum was likely meant: it now raises a `ValueError`. The
    docstring states the rule: a 1-D array is a depth profile, and a
    spectrum takes the shape `(n_wavelength, 1)`
  - Fix `Water1D` given several scattering hydrosols tabulated on different
    grids, which it refused with an error suggesting a remedy that did not
    work: a hydrosol constant with depth with one that varies (`HydrosolPR`
    with `HydrosolZhai`), hydrosols of different `n_theta` (their defaults
    differ) or `wavelength_phase`, or a `Hydrosol` given its phase and a `bp`
    array. Their phase matrices are now averaged on a common grid, holding
    the angles of all of them, with unchanged results where they already
    shared one
  - Fix a `Hydrosol` given its coefficients as arrays over the wavelengths
    of the profile and a `wavelength_phase`: the arrays were refused, or
    paired by position with the tabulation wavelengths when these were as
    many; they are now interpolated linearly onto `wavelength_phase`
  - Fix `LambSurface`, `RTLSSurface`, `RPVSurface`, `Water1D` and `WaterRw`,
    which accepted an `AlbedoMap` as albedo or BRDF coefficient, as the
    `LambSurface` error message and the `AlbedoLike` alias advertised, and
    then failed in `Smartg.run` or `Water1D.calc`: they now raise a
    `TypeError`, an `AlbedoMap` being only the `alb` of an `Environment`.
    The new `smartg.albedo.SpectralAlbedoLike` alias annotates the
    parameters that take a single spectral albedo
  - `HydrosolZhai` no longer warns of a division by zero in `log10` on
    every evaluation: its null concentration of non-algal particles is
    skipped instead of taken to the logarithm
  - Fix the particle backscattering of the phase matrices `Hydrosol`,
    `HydrosolPR` and `HydrosolZhai` derive from the backscattering ratio:
    the forward peak of the Fournier-Forand mixture their angular grid
    does not resolve was spread over all the angles, so that the
    backscattering coefficient missed `bbp_ratio * bp` by +10 % to +51 %
    (-6 % for `HydrosolZhai`). As in v1.2.0, that peak is now counted as
    unscattered: the scattering coefficient is scaled by the resolved
    fraction of the mixture (0.72 for a ratio of 0.01 on the 721 angles of
    `Hydrosol`, 0.91 for `HydrosolPR(chl=0.5)` on its 72001), times
    `1 - f` with a truncation. Without truncation, the backscattering is
    now right to 0.2 %. The mixture is also clipped at zero: with a ratio of
    0.04, beyond the 0.03 of the Park & Ruddick weights, that of
    `HydrosolZhai` was negative below 0.1 degree on its 7201 angles, a
    density the kernel cannot sample; `HydrosolZhai` now scatters 1.5 %
    more (a resolved fraction of 1.080 instead of 1.064)
  - Fix `HydrosolPR` and `HydrosolZhai`, which halved their particle
    scattering coefficient whenever the phase matrices were calculated, as
    in every `Smartg.run`, but not with `Water1D.calc(phase=False)`: they
    now scale `bp` as `Hydrosol` does, by the factor of their phase
    matrices only. Their results change: the particle scattering is twice
    that of 2.0.0b1 and v1.2.0, where the factor 0.5 dated from 2017, times
    the resolved fraction above
  - Fix `smartg.objects3d.extract_points`: it dropped the first line of
    coordinates when no blank line followed the comments, misread the
    numbers written with an exponent, and printed a message and returned
    no point for a missing file, which now raises a `FileNotFoundError`
  - `atm_pro_from_aeronet`, the ALIS histories allocation and
    `generate_h_a` no longer print unconditionally
  - Fix `Smartg.run(cell_proba=...)` given a 2-D array, its documented
    type, which always raised a `ValueError`: the array was compared with
    the string `'auto'`
  - Fix the 1D aerosol mixed into the cells of a 3D component of `Atm3D`:
    each cell took the aerosol of the layer below its own, and a cell in
    the bottom layer raised an `IndexError`. It now takes the aerosol of
    its own layer, as the molecular properties already did
  - Fix the phase matrices of a single 3D component of `Atm3D` over a 1D
    aerosol at several phase wavelengths: every wavelength but the first
    pointed at the matrices of another wavelength, or of other cells
  - Fix the receiver tallies of the 3D objects in double precision on the
    GPUs older than compute capability 6.0: the category weights lost
    their wavelength, and category 7 never added its weight
  - Fix the optical losses at the heliostats of a solar tower power run
    over several wavelengths: `wLoss` and `wLoss2` summed the weights of
    all the wavelengths, and now carry a `wavelength` axis, as `wPhCats`
    does. `nopt_view` weights each band by its share of `mtoa` (equally
    when `None`) in the numerators and denominators alike; it took the
    `powc_H` of the first band and the unweighted loss weights before.
    Single wavelength runs are unchanged. The kernel also wrote the 7
    loss weights of a forward run with heliostats and no receiver into a
    one element array
  - Fix the sign of U and V on the 180-360 degree half of the polar plots
    of `plot_polar_iquv(sym=True)` (the IPRT phase A figures): the half is
    the mirror image of the computed one about the principal plane, where
    U and V are odd. Azimuth angles that are not symmetric about 90
    degrees, which could not be mirrored, now raise a `ValueError`
  - Fix the phase matrix of a profile layer straddling two `pfgrid` layers:
    it took the `pfgrid` layer it overlapped most, possibly one where its
    particles are absent (a thin cloud low in a layer whose upper part lies
    in another `pfgrid` layer scattered with the aerosol matrix). It now
    takes, at each wavelength, the `pfgrid` layer holding the largest part
    of its particle scattering. The profiles whose `pfgrid` levels are
    profile levels, as all those of the tests and notebooks, are unchanged
  - Fix the phase matrices of a hydrosol reused at other wavelengths or on
    another grid: its memoized (truncated) phase matrices and truncation
    factor were those of its first use, e.g. the 450 nm matrix served
    again at 650 nm by a loop over the wavelengths
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
    Similarly `my_objects` now requires `obj3d=True`, the `mode` value is
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
    validation against the correct wavelength slice of `iphase_atm/oc` is
    performed
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
    - `smartg.diffgeom` and `smartg.shape`, the other legacy geometric
      modules, all replaced by `geoclide`


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

