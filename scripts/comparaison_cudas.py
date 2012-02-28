#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pyhdf.SD
from pylab import *
import numpy as np


# Fichier qui supperpose les résultats de deux résultats Cuda. 
# A été utilisé pour vérifier que la base pour un résultat normal ou buggé est la même

	          ##############
	         # PARAMETRES #
	        ##############

# Nom du fichier hdf à analyser SANS l'extension hdf (A MODIFIER)
nom_hdf1 = "Resultats_ok"
# Si le fichier suivant n'existe pas le prog s'arrete
path_hdf1 = "out_prog/test/" + nom_hdf1 + ".hdf"

# Nom du fichier hdf à analyser SANS l'extension hdf (A MODIFIER)
nom_hdf2 = "Resultats_bug"
# Si le fichier suivant n'existe pas le prog s'arrete
path_hdf2 = "out_prog/test/" + nom_hdf2 + ".hdf"

# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = "out_scripts/analyse_comparaison_cudas/"
#path_dossier_sortie = "out_prog/bug/"
	
			  #######################
	         # CREATION GRAPHIQUES #
	        #######################
dep = 4
fin = 178
# verification de l'existence du fichier hdf
if os.path.exists(path_hdf1) and os.path.exists(path_hdf2):
	# on vide le dossier de sortie du script
	#os.system("rm -rf "+path_dossier_sortie)
	#os.mkdir(path_dossier_sortie)
	# lecture du fichier hdf
	sd_hdf1 = pyhdf.SD.SD(path_hdf1)
	sd_hdf2 = pyhdf.SD.SD(path_hdf2)

	# lecture du nombre de valeurs de phi
	NBPHI = getattr(sd_hdf1,'NBPHI')
	
	# Récupération des valeurs de theta
	name = "Valeurs de theta echantillonnees"
	hdf_theta = sd_hdf1.select(name)
	theta = hdf_theta.get()
	
	# Récupération des valeurs de phi
	name = "Valeurs de phi echantillonnees"
	hdf_phi = sd_hdf1.select(name)
	phi = hdf_phi.get()	

	iphi=1
		
	# lecture du dataset
	listePlots = []
	listeLegends = []
	
	name = "Valeur de la reflectance pour un phi et theta donnes"
	sds_hdf1 = sd_hdf1.select(name)
	# recuperation du tableau
	data1 = sds_hdf1.get()
	
	sds_hdf2 = sd_hdf2.select(name)
	# recuperation du tableau
	data2 = sds_hdf2.get()
	
	# creation et sauvegarde du graphique
	data2_masked = np.zeros(180)
	data2_masked = np.ma.masked_where(data2[iphi,:] > 0.5 , data2[iphi,:])

	listePlots.append(plot(theta[dep:fin], data1[iphi,dep:fin]))
	listePlots.append(plot(theta[dep:fin], data2_masked[dep:fin]))

	title("Reflectance en fonction de theta pour phi="+str(phi[iphi])+" deg pour un resultat correct et un bug")
	xlabel("Theta (deg)")
	ylabel("Reflectance")
	grid(True)
	#savefig(path_dossier_sortie+"/analyse_ths=" + sys.argv[1] +".png", dpi=(140))
	savefig("out_prog/test/analyse_bug.png", dpi=(140))
	figure()
	

		
else:
	sys.stdout.write("Pas de fichier "+path_hdf1 + "ou" +path_hdf2 +"\n")
	sys.exit()
