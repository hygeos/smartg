#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import warnings
warnings.simplefilter("ignore",DeprecationWarning)
import pyhdf.SD
from pylab import *
import gzip
import struct


	          ##############
	         # PARAMETRES #
	        ##############
# Résultats Fortran
path_fortran_zip = \
"/home/florent/MC/bin/res_corrects/out_atmos_seule.ran=0050.wav=443.ths=30.000.tr=0.0000.ta=0.1000.difff=0001.pi0=0.967.H=002.000.\
mod=valid_T70.443.bin.gz"

# Nom du fichier hdf à analyser SANS l'extension hdf
nom_hdf = "hdf_atmos_seule_tauRay=0.000000_tauAer=0.100000_difff=1_ths=30.000000"
# Chemin complet du hdf cuda
path_cuda = "../out_prog/Resultats_" + nom_hdf + ".hdf"

# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = "../out_scripts/comparaison/comparaison_" + nom_hdf


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

file_fortran_bin.read(4)	#read(8) pour les anciens fichiers de sortie, read(4) pour les nouveaux

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
	
	# Récupération des valeurs de theta
	name = "Valeurs de theta echantillonnees"
	hdf_theta = sd_cuda.select(name)
	theta = hdf_theta.get()
	
	# Récupération des valeurs de phi
	name = "Valeurs de phi echantillonnees"
	hdf_phi = sd_cuda.select(name)
	phi = hdf_phi.get()

	name = "Valeur de la reflectance pour un phi et theta donnes"
	sds_cuda = sd_cuda.select(name)
	data = sds_cuda.get()		
	
else:
	sys.stdout.write("Pas de fichier "+path_cuda+"\n")
	sys.exit()


			  ###########################################
			 # Informations sur le traitement en cours #
			###########################################
			
sys.stdout.write("\n -------------------------------------------------------------------------------\n")
sys.stdout.write("| Le fichier cuda est " + path_cuda + "\t|\n")
sys.stdout.write("| Le fichier fortran est " + path_fortran_zip + "\t|\n")
sys.stdout.write("| Les résultats sont stockés dans " + path_dossier_sortie + "\t|\n")
sys.stdout.write(" -------------------------------------------------------------------------------\n")
	

	          #######################
	         # CREATION GRAPHIQUES #
	        #######################

dep = 6	# Indice de départ pour le tracé
fin = 177	# Indice de fin pour le tracé

#NOTE: Les indices ci-dessus ont été mis en place pour 2 raisons.
#	- Ils permettent de rogner la simulation si nécessaire. Par exemple, il est inutile de tracer le dernier angle (theta=90°) car il ne représente rien physiquement
#	- Il a également été remarqué que pour un indice de theta donné, la boite n'est pas la même en Fortran ou en Cuda. C'est pourquoi pour comparer les mêmes boites, il faut prendre l'indice i en Cuda et l'indice i-1 en Fortran. La ième boite Cuda correspond à la ième-1 boite en Fortran
	
# Pour comparer les 2 resultats il faut que phi parcourt un meme intervalle et qu'il y ait le meme nombre de boites selon phi
# Fortran :  intervalle=[0,PI]   nombre_de_boites=NBPHI_fortran
# Cuda :     intervalle=[0,2PI]  nombre_de_boites=NBPHI_cuda
# On va projeter les resultats du cuda sur [0,PI]

#ith=0
#print 'theta=' + str(theta[ith])
#print '-------Fortran-------'
#for iphi in xrange(0,25):
	
	#print 'phi(bornes)='+str(tab_fortran['real_phi_bornes'][iphi]*180/3.1415) + '\tr=' + str(tab_fortran['real_refl'][0, ith, iphi,
#0])

##cuda
#print ('--------CUDA--------')
#for iphi in xrange(0,25):
	
	#print 'phi='+str(phi[iphi]) + '\tr=' + str(data[iphi,ith+1])
	
#for i in xrange(180):
	#print 'theta[{0}] = {1:10.8}'.format(i,theta[i])
	
#for i in xrange(NBPHI_cuda):
	#print 'phi[{0}] = {1:10.8}'.format(i,phi[i])
	
#########################

if (NBPHI_cuda/2) == NBPHI_fortran:
	for iphi in xrange(0,NBPHI_cuda/2,5):
			
		# initialisation
		listePlots = []
		listeLegends = []
		figure()
		# fortran
		#listePlots.append(plot(tab_fortran['real_thv_bornes'][dep-1:fin-1], tab_fortran['real_refl'][0, dep-1:fin-1, iphi, 0]))
		listePlots.append(plot(theta[dep:fin], tab_fortran['real_refl'][0, dep-1:fin-1, iphi, 0]))
		listeLegends.append('Fortran')
		#cuda
		listePlots.append(plot(theta[dep:fin],(data[iphi,dep:fin]+data[NBPHI_cuda-iphi-1,dep:fin])/2))
		listeLegends.append('Cuda')
		
		# commun
		legend(listePlots, listeLegends, loc='best', numpoints=1)
		title('Comparaison avec le resultat fortran pour phi='+str(phi[iphi])+" deg")
		xlabel('Theta (deg)')
		ylabel('Reflectance')
		grid(True)
		savefig(path_dossier_sortie+"/comparaison_fortran_phi="+str(phi[iphi])+".png", dpi=(140))
		
		##########################################
		#	Figures d'analyse plus spécifiques	#
		########################################
		
		# Figure d'évaluation du taux d'erreur - RAPPORT
		figure()
		listePlots = []
		listeLegends = []
		listePlots.append( plot(theta[dep:fin], (tab_fortran['real_refl'][0, dep-1:fin-1,iphi,0])/
							((data[iphi,dep:fin]+data[NBPHI_cuda-iphi-1,dep:fin])/2) ) )
		listeLegends.append('Rapport Fortran/Cuda')
		
		#Régression linéaire
		(ar,br)=polyfit(theta[dep:fin],(tab_fortran['real_refl'][0, dep-1:fin-1, iphi,0])/
						((data[iphi,dep:fin]+data[NBPHI_cuda-iphi-1,dep:fin])/2) ,1)
		regLin=polyval([ar,br],theta[dep:fin])
		
		listePlots.append( plot(theta[dep:fin], regLin) )
		listeLegends.append('Regression lineaire y='+str(ar)+'x+'+str(br))
		legend(listePlots, listeLegends, loc='best', numpoints=1)
		
		title("Rapport des resultats Fortran et Cuda pour phi="+str(phi[iphi])+" deg")
		xlabel("Theta (deg)")
		ylabel("Rapport des reflactances")
		grid(True)
		savefig(path_dossier_sortie+"/rapport_reflectance_phi="+str(phi[iphi])+".png", dpi=(140))
		figure()
		
		# Figure d'évaluation du taux d'erreur - DIFFERENCE
		plot(theta[dep:fin],tab_fortran['real_refl'][0, dep-1:fin-1, iphi, 0]-(data[iphi,dep:fin]+data[NBPHI_cuda-iphi-1,dep:fin])/2)
		title("Difference des resultats Fortran - Cuda pour phi="+str(phi[iphi])+" deg")
		xlabel("Theta (deg)")
		ylabel("Difference des reflactances")
		grid(True)
		savefig(path_dossier_sortie+"/difference_reflectance_phi="+str(phi[iphi])+".png", dpi=(140))

	
	
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
		
