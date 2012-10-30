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

#
# Paramètres à modifier
#
#-----------------------------------------------------------------------------------------------------------------------
type_simu = "SIM_0"
date_simu = "11052012"
angle = "30"
geometrie = "PARALLELE"		#Géométrie de l'atmosphère

# Nom du fichier Cuda sans extension .hdf
nom_cuda = "out_CUDA_ths=30.00_tRay=0.0533_tAer=0.0000_ws=5.00_sim=0"

# Indices ci-dessus ont été mis en place car ils permettent de rogner la simulation si nécessaire.
# Les bords peuvent fausser les graphiques.
dep = 3			# Indice de départ pour le tracé
fin = 177		# Indice de fin pour le tracé
pas_figure = 20	# Pas en phi pour le tracé des graphiques
#-----------------------------------------------------------------------------------------------------------------------


######################################################
##				CHEMIN DES FICHIERS					##
######################################################

# Nom complet du fichier Cuda
path_cuda = "/home/florent/MCCuda/validation/"+geometrie+"/"+type_simu+"/simulation_"+date_simu+"/" + nom_cuda + ".hdf"

# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = \
"/home/florent/MCCuda/validation/"+geometrie+"/"+type_simu+"/graph_"+date_simu+"/analyse_rapide/"+"CUDA_rapide"+nom_cuda


##########################################################
##				DONNEES FICHIER CUDA					##
##########################################################

# verification de l'existence du fichier hdf
if os.path.exists(path_cuda):

	# lecture du fichier hdf
	sd_cuda = pyhdf.SD.SD(path_cuda)
	# lecture du nombre de valeurs de phi
	NBPHI_cuda = getattr(sd_cuda,'NBPHI')
	NBTHETA_cuda = getattr(sd_cuda,'NBTHETA')

	# Récupération des valeurs de theta
	name = "Valeurs de theta echantillonnees"
	hdf_theta = sd_cuda.select(name)
	theta = hdf_theta.get()

	# Récupération des valeurs de phi
	name = "Valeurs de phi echantillonnees"
	hdf_phi = sd_cuda.select(name)
	phi = hdf_phi.get()

	sds_cuda = sd_cuda.select("Valeurs de la reflectance (I)")
	data = sds_cuda.get()		

else:
	sys.stdout.write("Pas de fichier "+path_cuda+"\n")
	sys.exit()


##############################################################
##				INFORMATION A L'UTILISATEUR					##
##############################################################

sys.stdout.write("\n#-------------------------------------------------------------------------------#\n")
sys.stdout.write("# Le fichier cuda est " + path_cuda + "\n")
sys.stdout.write("# Les résultats sont stockés dans " + path_dossier_sortie + "\n")
sys.stdout.write("#-------------------------------------------------------------------------------#\n")


os.system("rm -rf "+ path_dossier_sortie)
os.system("mkdir -p "+ path_dossier_sortie)


##################################################################################
##				CREATION/CHOIX/MODIFICATION DE CERTAINES DONNES					##
##################################################################################

# Sauvegarde de la grandeur désirée
data_cuda = zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)

data_cuda = data[0:NBPHI_cuda,:]

# Infos en commentaire sur le graph
commentaire = type_simu + ' - ' + angle


##########################################################
##				CREATION DES GRAPHIQUES					##
##########################################################

#---------------------------------------------------------

# Calcul pour l'ergonomie des graphiques
max_data = data_cuda[0:NBPHI_cuda-pas_figure+1,dep:fin].max()
min_data = data_cuda[0:NBPHI_cuda-pas_figure+1,dep:fin].min()

#---------------------------------------------------------

for iphi in xrange(0,NBPHI_cuda,pas_figure):
	
	# initialisation
	figure()
	#cuda
	plot(theta[dep:fin],data_cuda[iphi][dep:fin])
	title( 'Analyse rapide pour Cuda pour phi='+str(phi[iphi])+' deg' )
	xlabel( 'Theta (deg)' )
	ylabel( 'I', rotation='horizontal' )
	#axis([0,theta[fin],0.99*min_data, 1.01*max_data])
	figtext(0.25, 0.7, commentaire+" deg", fontdict=None)
	figtext(0, 0, "Date: "+date_simu+"\nFichier cuda: "+nom_cuda, fontdict=None,size='xx-small')
	grid(True)
	savefig( path_dossier_sortie+'/analyse_rapide_Cuda_phi='+str(phi[iphi])+'.png', dpi=(140) )

