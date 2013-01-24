#!/bin/bash

mkdir resultat
rm resultat/*temoin

echo
echo "--- Compilation en mode plan parallèle ---"
echo
make clean
make SPH=no EXEC=mccuda-pp

echo
echo "--- Compilation en mode sphérique ---"
echo
make clean
make SPH=yes EXEC=mccuda-sp


echo
echo "--- Lancement des simulations tests ---"
echo
echo "--- 1/6 --- Plan Parallele-Rayleigh-Sol_noir ---"
echo
./mccuda-pp input/PP-Rayleigh-Sol_noir-1e9_photons.inp | tee resultat/PP-Rayleigh-Sol_noir-1e9_photons.out
python scripts/analyse_2D.py -r 0.4 -p 100 -s resultat/PP-Rayleigh-Sol_noir-1e9_photons.png resultat/PP-Rayleigh-Sol_noir-1e9_photons.hdf
echo
echo "--- 2/6 --- Plan Parallele-Rayleigh-Aerosol-Sol_noir ---"
echo
./mccuda-pp input/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.inp | tee resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.out
python scripts/analyse_2D.py -r 0.4 -p 100 -s resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.png resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.hdf
echo
echo "--- 3/6 --- Plan Parallele-Rayleigh-Aerosol-Glitter ---"
echo
./mccuda-pp input/PP-Rayleigh-Aerosol-Glitter-1e9_photons.inp | tee resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.out
python scripts/analyse_2D.py -r 0.4 -p 100 -s resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.png resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.hdf
echo
echo "--- 4/6 --- Spherique-Rayleigh-Sol_noir ---"
echo
./mccuda-sp input/SP-Rayleigh-Sol_noir-1e9_photons.inp | tee resultat/SP-Rayleigh-Sol_noir-1e9_photons.out
python scripts/analyse_2D.py -r 0.4 -p 100 -s resultat/SP-Rayleigh-Sol_noir-1e9_photons.png resultat/SP-Rayleigh-Sol_noir-1e9_photons.hdf
echo
echo "--- 5/6 --- Spherique-Rayleigh-Aerosol-Sol_noir ---"
echo
./mccuda-sp input/SP-Rayleigh-Aerosol-Sol_noir-1e10_photons.inp | tee resultat/SP-Rayleigh-Aerosol-Sol_noir-1e10_photons.out
python scripts/analyse_2D.py -r 0.4 -p 100 -s resultat/SP-Rayleigh-Aerosol-Sol_noir-1e10_photons.png resultat/SP-Rayleigh-Aerosol-Sol_noir-1e10_photons.hdf
echo
echo "--- 6/6 --- Spherique-Rayleigh-Aerosol-Glitter ---"
echo
./mccuda-sp input/SP-Rayleigh-Aerosol-Glitter-1e9_photons.inp | tee resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.out
python scripts/analyse_2D.py -r 0.4 -p 100 -s resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.png resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.hdf
echo "--- Simulations tests effectuees, sorties dans : resultat --- "


#mv  resultat/PP-Rayleigh-Sol_noir-1e9_photons.hdf resultat/PP-Rayleigh-Sol_noir-1e9_photons.reference.hdf
#mv  resultat/PP-Rayleigh-Sol_noir-1e9_photons.out resultat/PP-Rayleigh-Sol_noir-1e9_photons.reference.out
#mv  resultat/PP-Rayleigh-Sol_noir-1e9_photons.png resultat/PP-Rayleigh-Sol_noir-1e9_photons.reference.png
#mv  resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.hdf resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.reference.hdf
#mv  resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.out resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.reference.out
#mv  resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.png resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.reference.png
#mv  resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.hdf resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.reference.hdf
#mv  resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.out resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.reference.out
#mv  resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.png resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.reference.png
#mv  resultat/SP-Rayleigh-Sol_noir-1e9_photons.hdf resultat/SP-Rayleigh-Sol_noir-1e9_photons.reference.hdf
#mv  resultat/SP-Rayleigh-Sol_noir-1e9_photons.out resultat/SP-Rayleigh-Sol_noir-1e9_photons.reference.out
#mv  resultat/SP-Rayleigh-Sol_noir-1e9_photons.png resultat/SP-Rayleigh-Sol_noir-1e9_photons.reference.png
#mv  resultat/SP-Rayleigh-Aerosol-Sol_noir-1e9_photons.hdf resultat/SP-Rayleigh-Aerosol-Sol_noir-1e9_photons.reference.hdf
#mv  resultat/SP-Rayleigh-Aerosol-Sol_noir-1e9_photons.out resultat/SP-Rayleigh-Aerosol-Sol_noir-1e9_photons.reference.out
#mv  resultat/SP-Rayleigh-Aerosol-Sol_noir-1e9_photons.png resultat/SP-Rayleigh-Aerosol-Sol_noir-1e9_photons.reference.png
#mv  resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.hdf resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.reference.hdf
#mv  resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.out resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.reference.out
#mv  resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.png resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.reference.png

rm  resultat/*temoin

