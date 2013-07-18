#!/usr/bin/env python
# vim:fileencoding=utf-8



from os import system
from os.path import exists, basename
from glob import glob


exec_sp = 'SMART-G-sp'
exec_pp = 'SMART-G-pp'
dir_out = 'resultat'


if not exists(dir_out):
    system('mkdir {}'.format(dir_out))
system('rm {}/*temoin'.format(dir_out))


print
print "--- Compilation en mode plan parallèle ---"
print
system('make clean')
system('make SPH=no EXEC={}'.format(exec_pp))

print
print "--- Compilation en mode sphérique ---"
print
system('make clean')
system('make SPH=yes EXEC={}'.format(exec_sp))


print
print "--- Lancement des simulations tests ---"
print

# input files loop
for inp in glob('input/*.inp'):
    binp = basename(inp)

    # executable name
    if binp.startswith('PP'):
        exe = exec_pp
    else:
        exe = exec_sp

    # execution
    print '--- {} ---'.format(binp)
    cmd = './{exe} {inp} | tee {out}'.format(exe=exe,
        inp=inp, out=dir_out+'/'+binp.replace('.inp', '.out'))
    ret = system(cmd)
    if ret:
        print 'Return value is {}, stopping now.'.format(ret)
        exit(1)

    # generate graph
    hdf = dir_out+'/'+binp.replace('.inp', '.hdf')
    png = dir_out+'/'+binp.replace('.inp', '.png')
    cmd = 'python tools/analyse_2D.py -r 0.4 -p 100 -s {png} {hdf}'.format(png=png, hdf=hdf)
    ret = system(cmd)
    if ret:
        print 'Return value is {}, stopping now.'.format(ret)
        exit(1)
# 
# 
# #mv  resultat/PP-Rayleigh-Sol_noir-1e9_photons.hdf resultat/PP-Rayleigh-Sol_noir-1e9_photons.reference.hdf
# #mv  resultat/PP-Rayleigh-Sol_noir-1e9_photons.out resultat/PP-Rayleigh-Sol_noir-1e9_photons.reference.out
# #mv  resultat/PP-Rayleigh-Sol_noir-1e9_photons.png resultat/PP-Rayleigh-Sol_noir-1e9_photons.reference.png
# #mv  resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.hdf resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.reference.hdf
# #mv  resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.out resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.reference.out
# #mv  resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.png resultat/PP-Rayleigh-Aerosol-Sol_noir-1e9_photons.reference.png
# #mv  resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.hdf resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.reference.hdf
# #mv  resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.out resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.reference.out
# #mv  resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.png resultat/PP-Rayleigh-Aerosol-Glitter-1e9_photons.reference.png
# #mv  resultat/SP-Rayleigh-Sol_noir-1e9_photons.hdf resultat/SP-Rayleigh-Sol_noir-1e9_photons.reference.hdf
# #mv  resultat/SP-Rayleigh-Sol_noir-1e9_photons.out resultat/SP-Rayleigh-Sol_noir-1e9_photons.reference.out
# #mv  resultat/SP-Rayleigh-Sol_noir-1e9_photons.png resultat/SP-Rayleigh-Sol_noir-1e9_photons.reference.png
# #mv  resultat/SP-Rayleigh-Aerosol-Sol_noir-1e9_photons.hdf resultat/SP-Rayleigh-Aerosol-Sol_noir-1e9_photons.reference.hdf
# #mv  resultat/SP-Rayleigh-Aerosol-Sol_noir-1e9_photons.out resultat/SP-Rayleigh-Aerosol-Sol_noir-1e9_photons.reference.out
# #mv  resultat/SP-Rayleigh-Aerosol-Sol_noir-1e9_photons.png resultat/SP-Rayleigh-Aerosol-Sol_noir-1e9_photons.reference.png
# #mv  resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.hdf resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.reference.hdf
# #mv  resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.out resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.reference.out
# #mv  resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.png resultat/SP-Rayleigh-Aerosol-Glitter-1e9_photons.reference.png
# 
# rm  resultat/*temoin
# 
