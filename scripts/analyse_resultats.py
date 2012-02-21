#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import warnings
warnings.simplefilter("ignore",DeprecationWarning)
import pyhdf.SD
from pylab import *


	          ##############
	         # PARAMETRES #
	        ##############

# Nom du fichier hdf à analyser SANS l'extension hdf (A MODIFIER)
nom_hdf = "Resultats_dioptre_agite_seul_ths=70.000000_ws=5.000000"

# Si le fichier suivant n'existe pas le prog s'arrete
path_hdf = "../out_prog/" + nom_hdf + ".hdf"
# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = "../out_scripts/analyse_" + nom_hdf
	
			  #######################
	         # CREATION GRAPHIQUES #
	        #######################

# verification de l'existence du fichier hdf
if os.path.exists(path_hdf):
	# on vide le dossier de sortie du script
	os.system("rm -rf "+path_dossier_sortie)
	os.mkdir(path_dossier_sortie)
	# lecture du fichier hdf
	sd_hdf = pyhdf.SD.SD(path_hdf)
	# lecture du nombre de valeurs de phi
	NBPHI = getattr(sd_hdf,'NBPHI')

	print '\n -------------------------------------------------------------------------------'
	print 'Le fichier traité est ' + path_hdf
	print 'Les résultats sont stockés dans ' + path_dossier_sortie
	print '-------------------------------------------------------------------------------'

	# Récupération des valeurs de theta
	name = "Valeurs de theta echantillonnees"
	hdf_theta = sd_hdf.select(name)
	theta = hdf_theta.get()
	
	# Récupération des valeurs de phi
	name = "Valeurs de phi echantillonnees"
	hdf_phi = sd_hdf.select(name)
	phi = hdf_phi.get()	
	
	# pour chaque iphi on cree un graphique
	sys.stdout.write("Realisation des graphiques en cours\n")
	# print 'Realisation des graphiques en cours'
	for iphi in xrange(NBPHI):
		
		# lecture du dataset
		name = "Valeur de la reflectance pour un phi et theta donnes"
		sds_hdf = sd_hdf.select(name)
		# recuperation du tableau et de la valeur de phi
		data = sds_hdf.get()
		# creation et sauvegarde du graphique
		plot(theta[:],data[iphi,:])
		title("Reflectance en fonction de theta pour phi="+str(phi[iphi]))
		xlabel("Theta (rad)")
		ylabel("Reflectance")
		grid(True)
		savefig(path_dossier_sortie+"/analyse_resultats_iphi="+str(phi[iphi])+".png", dpi=(140))
		figure()
	
		
else:
	sys.stdout.write("Pas de fichier "+path_hdf+"\n")
	sys.exit()
	

	          ###############################
	         # CREATION FICHIER PARAMETRES #
	        ###############################

# Recuperation des parametres
NBPHOTONS = getattr(sd_hdf,'NBPHOTONS')
NBLOOP = getattr(sd_hdf,'NBLOOP')
SEED = getattr(sd_hdf,'SEED')
XBLOCK = getattr(sd_hdf,'XBLOCK')
YBLOCK = getattr(sd_hdf,'YBLOCK')
XGRID = getattr(sd_hdf,'XGRID')
YGRID = getattr(sd_hdf,'YGRID')
NBTHETA = getattr(sd_hdf,'NBTHETA')
NBPHI = getattr(sd_hdf,'NBPHI')
THSDEG = getattr(sd_hdf,'THSDEG')
LAMBDA = getattr(sd_hdf,'LAMBDA')
TAURAY = getattr(sd_hdf,'TAURAY')
TAUAER = getattr(sd_hdf,'TAUAER')
W0AER = getattr(sd_hdf,'W0AER')
PROFIL = getattr(sd_hdf,'PROFIL')
HA = getattr(sd_hdf,'HA')
HR = getattr(sd_hdf,'HR')
ZMIN = getattr(sd_hdf,'ZMIN')
ZMAX = getattr(sd_hdf,'ZMAX')
WINDSPEED = getattr(sd_hdf,'WINDSPEED')
NH2O = getattr(sd_hdf,'NH2O')
SIM = getattr(sd_hdf,'SIM')
SUR = getattr(sd_hdf,'SUR')
DIOPTRE = getattr(sd_hdf,'DIOPTRE')
CONPHY = getattr(sd_hdf,'CONPHY')
DIFFF = getattr(sd_hdf,'DIFFF')
PATHRESULTATSHDF = getattr(sd_hdf,'PATHRESULTATSHDF')
PATHTEMOINHDF = getattr(sd_hdf,'PATHTEMOINHDF')

# Creation du fichier contenant les parametres de la simulation
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

