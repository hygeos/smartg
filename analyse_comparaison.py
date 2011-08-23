#!/usr/bin/env python
import os
import sys
import pyhdf.SD
from pylab import *
import gzip
import struct


	          #####################
	         # RESULTATS FORTRAN #
	        #####################

file_fortran_zip = "/home/tristan/Desktop/Progs_existants/Fortran/bin/out.ran=0001.wav=635.ths=70.000.tr=0.0533.ta=0.0000.pi0=0.967.H=002.000.bin.gz"

(NSTK,NTHV,NBPHI_fortran,NTYP) = (4, 180, 180, 8)
dt = dtype([
('version', float32),
('nphotons', int64),
('nthv', int32),
('nphi', int32),
('thetas', float32),
('iprofil', int32),
('isur', int32),
('isim', int32),
('initgerme', int32),
('idioptre', int32),
('real_toRay', float32),
('real_toaer', float32),
('real_windspeed', float32),
('real_wl', float32),
('real_nh2o', float32),
('real_refl', float32, (NSTK,NTHV,NBPHI_fortran,NTYP)),
('real_znad', float32, (8*NTYP,)),
('real_upun', float32),
('real_upab', float32),
('real_dnun', float32),
('real_dnab', float32),
('real_dnabdirect', float32),
('real_dnabplus', float32),
('biais', int64),
('duree', float32, (3,)),
('real_thv_bornes', float32, (NTHV,)),
('pi', float32),
('real_phi_bornes', float32, (NBPHI_fortran+1,)),
])

# lecture du fichier fortran (bin)
file_fortran_bin = gzip.open(file_fortran_zip)
file_fortran_bin.read(8)
st = file_fortran_bin.read()
contenu_fortran = fromstring(st, dtype=dt, count=1)
# creation du tableau fortran
tab_fortran = {}
for i in dt.names:
	if prod(shape(contenu_fortran[i])) == 1:
		tab_fortran[i] = contenu_fortran[i][0]
	else:
		tab_fortran[i] = ravel(contenu_fortran[i]).reshape(dt[i].shape, order='F')
file_fortran_bin.close()


	          ##################
	         # RESULTATS CUDA #
	        ##################

# verification de l'existence du fichier hdf
if os.path.exists("out_prog/Resultats.hdf"):
	# on vide le dossier de sortie du script
	os.system("rm -rf out_scripts/analyse_comparaison")
	os.mkdir("out_scripts/analyse_comparaison")
	# lecture du fichier hdf
	file_cuda = 'out_prog/Resultats.hdf'
	sd_cuda = pyhdf.SD.SD(file_cuda)
	# lecture du nombre de valeurs de phi
	NBPHI_cuda = getattr(sd_cuda,'NBPHI')
else:
	sys.stdout.write("Pas de fichier Resultats.hdf\n")
	sys.exit()


	          #######################
	         # CREATION GRAPHIQUES #
	        #######################
	        
# Pour comparer les 2 resultats il faut que phi parcourt un meme intervalle et qu'il y ait le meme nombre de boites selon phi
# Fortran :  intervalle=[0,PI]   nombre_de_boites=NBPHI_fortran
# Cuda :     intervalle=[0,2PI]  nombre_de_boites=NBPHI_cuda
# On va projeter les resultats du cuda sur [0,PI]
if (NBPHI_cuda/2) == NBPHI_fortran:
	for iphi in xrange(NBPHI_cuda/2):
		# initialisation
		listePlots = []
		listeLegends = []
		
		# fortran
		listePlots.append(plot(tab_fortran['real_thv_bornes'], tab_fortran['real_refl'][0, :, iphi, 0]))
		listeLegends.append('Fortran')
		
		# cuda
		name_1 = 'Resultats (iphi = ' + str(iphi) + ')'
		sds_cuda_1 = sd_cuda.select(name_1)
		tab_cuda_1 = sds_cuda_1.get()
		phi = getattr(sds_cuda_1,'phi')
		name_2 = "Resultats (iphi = " + str(NBPHI_cuda-iphi-1) + ")"
		sds_hdf_2 = sd_cuda.select(name_2)
		tab_cuda_2 = sds_hdf_2.get()
		listePlots.append(plot(tab_cuda_1[:,1],(tab_cuda_1[:,0]+tab_cuda_2[:,0])/2))
		listeLegends.append('Cuda')
		
		# commun
		legend(listePlots, listeLegends, loc='best', numpoints=1)
		title('Comparaison avec le resultat fortran pour phi='+str(phi))
		xlabel('Theta (rad)')
		ylabel('Eclairement')
		grid(True)
		savefig('out_scripts/analyse_comparaison/comparaison_fortran_phi='+str(phi)+'.png', dpi=(140))
		figure()
else:
	sys.stdout.write("Les tableaux ne font pas la meme taille\n")
		
