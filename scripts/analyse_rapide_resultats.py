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
nom_hdf = "Resultats_test_tauRay=0.053300_tauAer=0.100000_difff=0_ths=30.000000_sim=-1"
# a commenter si pas besoin
#nom_hdf = nom_hdf + sys.argv[1] + ".000000"

# Si le fichier suivant n'existe pas le prog s'arrete
path_hdf = "../out_prog/" + nom_hdf + ".hdf"
# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = "../out_scripts/analyse_rapide/"
	
			  #######################
	         # CREATION GRAPHIQUES #
	        #######################

# verification de l'existence du fichier hdf
if os.path.exists(path_hdf):
	# on vide le dossier de sortie du script
	#os.system("rm -rf "+path_dossier_sortie)
	#os.mkdir(path_dossier_sortie)
	# lecture du fichier hdf
	sd_hdf = pyhdf.SD.SD(path_hdf)
	# lecture du nombre de valeurs de phi
	NBPHI = getattr(sd_hdf,'NBPHI')

	sys.stdout.write(" ----------------------------------------------------------------\n")
	sys.stdout.write("| Le fichier traité est " + path_hdf + "\t|\n")
	sys.stdout.write("| Les résultats sont stockés dans " + path_dossier_sortie + "\t|\n")
	sys.stdout.write(" ----------------------------------------------------------------\n")
	
	# Récupération des valeurs de theta
	name = "Valeurs de theta echantillonnees"
	hdf_theta = sd_hdf.select(name)
	theta = hdf_theta.get()
	
	# Récupération des valeurs de phi
	name = "Valeurs de phi echantillonnees"
	hdf_phi = sd_hdf.select(name)
	phi = hdf_phi.get()	

	iphi=1
		
	# lecture du dataset
	name = "Valeur de la reflectance pour un phi et theta donnes"
	sds_hdf = sd_hdf.select(name)
	# recuperation du tableau et de la valeur de phi
	data = sds_hdf.get()
	# creation et sauvegarde du graphique
	plot(theta[0:179],data[iphi,0:179])
	title("Reflectance en fonction de theta pour phi="+str(phi[iphi])+" deg")
	xlabel("Theta (deg)")
	ylabel("Reflectance")
	grid(True)
	#savefig(path_dossier_sortie+"/analyse_ths=" + sys.argv[1] +".png", dpi=(140))
	savefig(path_dossier_sortie + nom_hdf +"_analyse.png", dpi=(140))
	figure()
	

		
else:
	sys.stdout.write("Pas de fichier "+path_hdf+"\n")
	sys.exit()
