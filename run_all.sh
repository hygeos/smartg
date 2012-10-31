#!/bin/bash

echo
echo "--- Compilation en mode plan parallèle ---"
echo
make clean
make SPH=no EXEC=mccuda-pp

echo
echo "--- Compilation en mode sphérique ---"
echo
make clean
make SPH=yes EXEC=mccuda-sph




