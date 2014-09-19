#!/bin/bash
# Example 1 Basic
# The examples are coming from fast computations of a black, wind roughened sea surface with only molecular scattering and absorption by ozone in a spherical shell or plane parallel atmosphere
#
# One input file, polar plot (theta_s,phi) of I,Q,U and IP, default settings, output in a PNG file
python smartg_analyze.py  -s gallery/example1.png data/out_2_560.000000_thv-40.000000-sp.hdf
#
# Example 2 Customized
# One input file , polar plot (theta_s,phi) of I,Q,U and Polarization Ratio with scale set to 100%, reflectance scale set to 0.3, 
#     , with a transect for the plane of azimuth=120 deg, output in a PNG file
python smartg_analyze.py  -s gallery/example2.png  -p 100 -r 0.3 -t 120 data/out_2_560.000000_thv-40.000000-sp.hdf
#
# Example 3 Differences
# Absolute Difference of two files (here spherical - plane parallel), polar plot (theta_s,phi) of I,Q,U and Polarization Ratio (diff) with scale set to 2%, reflectance (diff) scale set to 0.05, 
#     , with a transect for the plane of azimuth=120 deg, output in a PNG file
python smartg_analyze.py  -s gallery/example3.png  -p 2 -r 0.05 -t 120 data/out_2_560.000000_thv-40.000000-sp.hdf data/out_2_560.000000_thv-40.000000-pp.hdf
#
# Example 4 3D dataset 
# List of input files with ONE parameter varying (here for example LAMBDA plotted from 400 to 700), polar plot (lambda,phi) of I,Q,U, Polarization Ratio with scale set to 100, 
#    ,  with theta_s fixed at a value of 60 deg,
#    , reflectance scale set to 0.2, with a transect along the lambda axis for the plane of azimuth=90 deg, output in a PNG file
python smartg_analyze.py  -s gallery/example4.png -r 0.2 -p 100 -o LAMBDA,400,700 -l 0,60 -t 120 list.txt
#
