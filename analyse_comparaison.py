#!/usr/bin/env python
import os
import sys
import pyhdf.SD
from pylab import *
import gzip
import struct


	          ##############
	         # PARAMETRES #
	        ##############

path_fortran_zip = "/home/tristan/Desktop/Progs_existants/Fortran/bin/out.ran=0001.wav=635.ths=70.000.tr=0.0533.ta=0.0000.pi0=0.967.H=002.000.bin.gz"
# Si le fichier suivant n'existe pas le prog s'arrete
path_cuda = "out_prog/Resultats.hdf"
# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = "out_scripts/analyse_comparaison"


	          #####################
	         # RESULTATS FORTRAN #
	        #####################


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
file_fortran_bin = gzip.open(path_fortran_zip)
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
if os.path.exists(path_cuda):
	# on vide le dossier de sortie du script
	os.system("rm -rf "+path_dossier_sortie)
	os.mkdir(path_dossier_sortie)
	# lecture du fichier hdf
	sd_cuda = pyhdf.SD.SD(path_cuda)
	# lecture du nombre de valeurs de phi
	NBPHI_cuda = getattr(sd_cuda,'NBPHI')
else:
	sys.stdout.write("Pas de fichier "+path_cuda+"\n")
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
		savefig(path_dossier_sortie+"/comparaison_fortran_phi="+str(phi)+".png", dpi=(140))
		figure()
else:
	sys.stdout.write("Les tableaux ne font pas la meme taille\n")
	sys.exit()


	          ###############################
	         # CREATION FICHIER PARAMETRES #
	        ###############################

NBPHOTONS = getattr(sd_cuda,'NBPHOTONS')
NBLOOP = getattr(sd_cuda,'NBLOOP')
SEED = getattr(sd_cuda,'SEED')
XBLOCK = getattr(sd_cuda,'XBLOCK')
YBLOCK = getattr(sd_cuda,'YBLOCK')
XGRID = getattr(sd_cuda,'XGRID')
YGRID = getattr(sd_cuda,'YGRID')
NBTHETA = getattr(sd_cuda,'NBTHETA')
NBPHI = getattr(sd_cuda,'NBPHI')
NBSTOKES = getattr(sd_cuda,'NBSTOKES')
THSDEG = getattr(sd_cuda,'THSDEG')
LAMBDA = getattr(sd_cuda,'LAMBDA')
TAURAY = getattr(sd_cuda,'TAURAY')
TAUAER = getattr(sd_cuda,'TAUAER')
W0AER = getattr(sd_cuda,'W0AER')
PROFIL = getattr(sd_cuda,'PROFIL')
HA = getattr(sd_cuda,'HA')
HR = getattr(sd_cuda,'HR')
ZMIN = getattr(sd_cuda,'ZMIN')
ZMAX = getattr(sd_cuda,'ZMAX')
WINDSPEED = getattr(sd_cuda,'WINDSPEED')
NH2O = getattr(sd_cuda,'NH2O')
SIM = getattr(sd_cuda,'SIM')
SUR = getattr(sd_cuda,'SUR')
DIOPTRE = getattr(sd_cuda,'DIOPTRE')
CONPHY = getattr(sd_cuda,'CONPHY')
DIFFF = getattr(sd_cuda,'DIFFF')
PATHRESULTATSHDF = getattr(sd_cuda,'PATHRESULTATSHDF')
PATHTEMOINHDF = getattr(sd_cuda,'PATHTEMOINHDF')
# creation du fichier contenant les parametres de la simulation
fichierParametres = open(path_dossier_sortie+"/Parametres.txt", "w")
fichierParametres.write("NBPHOTONS = " + str(NBPHOTONS) + "\n")
fichierParametres.write("NBLOOP = " + str(NBLOOP) + "\n")
fichierParametres.write("SEED = " + str(SEED) + "\n")	
fichierParametres.write("XBLOCK = " + str(XBLOCK) + "\n")
fichierParametres.write("YBLOCK = " + str(YBLOCK) + "\n")
fichierParametres.write("XGRID = " + str(XGRID) + "\n")
fichierParametres.write("YGRID = " + str(YGRID) + "\n")
fichierParametres.write("NBTHETA = " + str(NBTHETA) + "\n")
fichierParametres.write("NBPHI = " + str(NBPHI) + "\n")
fichierParametres.write("NBSTOKES = " + str(NBSTOKES) + "\n")
fichierParametres.write("THSDEG = " + str(THSDEG) + "\n")
fichierParametres.write("LAMBDA = " + str(LAMBDA) + "\n")
fichierParametres.write("TAURAY = " + str(TAURAY) + "\n")
fichierParametres.write("TAUAER = " + str(TAUAER) + "\n")
fichierParametres.write("W0AER = " + str(W0AER) + "\n")
fichierParametres.write("PROFIL = " + str(PROFIL) + "\n")
fichierParametres.write("HA = " + str(HA) + "\n")
fichierParametres.write("HR = " + str(HR) + "\n")
fichierParametres.write("ZMIN = " + str(ZMIN) + "\n")
fichierParametres.write("ZMAX = " + str(ZMAX) + "\n")
fichierParametres.write("WINDSPEED = " + str(WINDSPEED) + "\n")
fichierParametres.write("NH2O = " + str(NH2O) + "\n")
fichierParametres.write("SIM = " + str(SIM) + "\n")
fichierParametres.write("SUR = " + str(NBPHOTONS) + "\n")
fichierParametres.write("DIOPTRE = " + str(DIOPTRE) + "\n")
fichierParametres.write("CONPHY = " + str(CONPHY) + "\n")
fichierParametres.write("DIFFF = " + str(DIFFF) + "\n")
fichierParametres.write("PATHRESULTATSHDF = " + str(PATHRESULTATSHDF) + "\n")
fichierParametres.write("PATHTEMOINHDF = " + str(PATHTEMOINHDF) + "\n")
fichierParametres.close()
		
