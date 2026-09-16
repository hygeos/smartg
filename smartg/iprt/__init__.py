"""Tools for the IPRT model intercomparison cases.

IPRT (International Polarized Radiative Transfer) compares polarized
radiative transfer models in three phases, each with its own cases and
result format:

- phase A, the 1D cases A1 to A6 and B1 to B4: ASCII tables, one record
  per line with the columns depol zout sza saa va phi I Q U V Istd Qstd
  Ustd Vstd;
- phase B, the 3D cases (C2 and C3 here): ASCII tables with the columns
  case theta_0 z theta phi ix iy I Q U V Istd Qstd Ustd Vstd;
- phase 3, the cases D1 to D6 and E1 to E6: netCDF files.

Modules
-------
common
    Tools valid for every phase: the delta_m metric and the grouping
    of its I, Q, U and V inputs.
phase_a
    Conversion to, selection from and comparison of the phase A ASCII
    tables.
phase_b
    Atmospheres, sensors and runs of the C2 and C3 cases, reading of
    the phase B ASCII tables, camera plots and comparison with a
    reference. Importing it compiles no SMART-G kernel.
phase3
    Runs of the phase 3 cases, written in the IPRT netCDF format.
    Importing it compiles the SMART-G kernels.

The polar and radiance plots do not depend on the phase and are in
smartg.view: plot_polar_iquv, plot_iquv_comparison and camera_view.

The package imports none of its modules, so that the phase shows in
every import, e.g. ``from smartg.iprt.phase_a import select_iprt_iquv``.
"""
