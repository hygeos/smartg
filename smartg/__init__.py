"""SMART-G, a GPU Monte Carlo radiative transfer code.

Speed-up Monte carlo Advanced Radiative Transfer code using GPU. A
simulation is assembled from the scene modules, run by the CUDA kernels
of :mod:`smartg.smartg`, and returned as an :class:`xarray.Dataset`.

Running a Simulation
--------------------
smartg
    ``Smartg`` compiles the kernels, ``Smartg.run`` launches the photons
    and returns the results; ``LocalEstimate`` and ``Alis`` select the
    estimator and the spectral mode.
sensor
    ``Sensor``, the position, direction and field of view a result is
    computed for.
bandset
    The spectral grid a simulation is run on.

Describing the Scene
--------------------
atmosphere
    Atmospheric profiles, ``Atm1D`` and ``Atm3D``, their molecular,
    aerosol (``AerOPAC``) and cloud (``Cloud``) components.
water
    Water column profiles, ``Water1D`` and ``WaterRw``, and their
    hydrosols.
surface, albedo
    The air-water interface, the ground and their reflectances.
objects3d, grid3d
    3D objects and the voxel grid of the 3D atmosphere mode.

Optical Properties
------------------
phase, truncation
    Scattering phase matrices, and the truncation of their forward peak.
kdis, reptran
    Gaseous absorption parameterizations.
rrs, vrs
    Rotational and vibrational Raman scattering.

Results
-------
postprocess, histories, diff
    Irradiances, photon histories of the ALIS mode, and profile
    differences.
view
    Plots of the outputs.

Support
-------
auxdata
    Download and update of the auxiliary data.
config, typing, environ, progress, interp, cdf, xarray
    Constants, type aliases and numerical helpers.

The IPRT model intercomparison tools are in the :mod:`smartg.iprt`
subpackage. Nothing is imported here, so every module shows in its
import, e.g. ``from smartg.atmosphere import Atm1D``.
"""
