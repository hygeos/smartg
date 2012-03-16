#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import warnings
warnings.simplefilter("ignore",DeprecationWarning)
import pyhdf.SD
from pylab import *


##################################################
##				FICHIERS A LIRE					##
##################################################

#
# Paramètres à modifier
#
#-----------------------------------------------------------------------------------------------------------------------
type_simu = "molecules_seules"
date_simu = "16032012"
angle = '30'
# Nom du fichier Cuda sans extension .hdf
nom_cuda = "out_CUDA_atmos_ths=30.00_tRay=0.0533_tAer=0.0000"

# Indices ci-dessus ont été mis en place car ils permettent de rogner la simulation si nécessaire.
# Les bords peuvent fausser les graphiques.
dep = 3			# Indice de départ pour le tracé
fin = 177		# Indice de fin pour le tracé
pas_figure = 5	# Pas en phi pour le tracé des graphiques
#-----------------------------------------------------------------------------------------------------------------------


######################################################
##				SELECTION DES DONNEES				##
######################################################

flag=True
while flag:
	print '\n\nQuelles donnees voulez-vous tracer?'
	#print '1:Reflectance\n2:Q\n3:U\n4:Lumiere polarisee'
	choix = raw_input('i >pour la reflectance\nq >pour Q\nu >pour U\nl >pour la lumiere polarisee\n')
	
	if choix == 'i':
		nom_data_cuda = "Valeurs de la reflectance (I)"
		colonne_donnee_sos = 2
		type_donnees = "I"
		flag=False
		
	elif choix == 'q':
		nom_data_cuda = "Valeurs de Q"
		colonne_donnee_sos = 3
		type_donnees = "Q"
		flag=False
		
	elif choix == 'u':
		nom_data_cuda = "Valeurs de U"
		colonne_donnee_sos = 4
		type_donnees = "U"
		flag=False
		
	elif choix == 'l':
		nom_data_cuda = "Valeurs de la lumiere polarisee (LP)"
		colonne_donnee_sos = 5
		type_donnees = "LP"
		flag=False
		
	else:
			print 'Choix incorrect, recommencez'
			
print 'C\'est parti pour la simulation de {0}'.format(type_donnees)


######################################################
##				CHEMIN DES FICHIERS					##
######################################################

# Nom complet du fichier Cuda
path_cuda = "/home/florent/MCCuda/validation/SPHERIQUE/"+type_simu+"/simulation_"+date_simu+"/" + nom_cuda + ".hdf"

# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = \
"/home/florent/MCCuda/validation/SPHERIQUE/"+type_simu+"/graph_"+date_simu+"/"+type_donnees+"/"+type_donnees+"_CUDA_" + nom_cuda


##########################################################
##				DONNEES FICHIER CUDA					##
##########################################################

# verification de l'existence du fichier CUDA
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

	sds_cuda = sd_cuda.select(nom_data_cuda)
	data = sds_cuda.get()		
	
else:
	sys.stdout.write("Pas de fichier "+path_cuda+"\n")
	sys.exit()


##############################################################
##				INFORMATION A L'UTILISATEUR					##
##############################################################

sys.stdout.write("\n#-------------------------------------------------------------------------------#\n")
sys.stdout.write("# Le fichier Cuda est " + path_cuda + "\n")
sys.stdout.write("# Les résultats sont stockés dans " + path_dossier_sortie + "\n")
sys.stdout.write("#-------------------------------------------------------------------------------#\n")


os.system("rm -rf "+ path_dossier_sortie)
os.system("mkdir -p "+ path_dossier_sortie)


##################################################################################
##				CREATION/CHOIX/MODIFICATION DE CERTAINES DONNES					##
##################################################################################

# Pour comparer les 2 resultats il faut que phi parcourt un meme intervalle et qu'il y ait le meme nombre de boites selon phi
# SOS : intervalle=[0,PI]
# Cuda : intervalle=[0,2PI]  nombre_de_boites=NBPHI_cuda
# On va projeter les resultats du cuda sur [0,PI]

data_cuda = zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)

data_cuda = data[0:NBPHI_cuda,]

# Infos en commentaire sur le graph
commentaire = type_simu + ' - ' + angle


##########################################################
##				CREATION DES GRAPHIQUES					##
##########################################################

for iphi in xrange(0,NBPHI_cuda,pas_figure):
	
	figure()
	plot(theta[dep:fin], data_cuda[iphi][dep:fin])
	
	title( type_donnees + ' Cuda pour phi='+str(phi[iphi])+' deg' )
	xlabel( 'Theta (deg)' )
	ylabel( type_donnees, rotation='horizontal' )
	figtext(0.25, 0.7, commentaire+" deg", fontdict=None)
	figtext(0, 0, "Date: "+date_simu+"\nFichier cuda: "+nom_cuda, fontdict=None, size='xx-small')
	grid(True)
	savefig( path_dossier_sortie+'/'+type_donnees+'_Cuda_phi='+str(phi[iphi])+'.png', dpi=(140) )
	

##########################################################
##				CREATION FICHIER PARAMETRE				##
##########################################################

# Création du fichier texte de sortie
fichierSortie = open(path_dossier_sortie+"/Parametres_CUDA_"+nom_cuda+".txt", "w")

# Récupération des données
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
PATHDIFFAER = getattr(sd_cuda,'PATHDIFFAER')
PATHPROFILATM = getattr(sd_cuda,'PATHPROFILATM')

# Ecriture dans le fichier
fichierSortie.write(" ##### Paramètres CUDA #####\n") 
fichierSortie.write('NBPHOTONS = {0:.2e}\n'.format(NBPHOTONS))
fichierSortie.write("NBLOOP = " + str(NBLOOP) + "\n")
fichierSortie.write("SEED = " + str(SEED) + "\n")	
fichierSortie.write("XBLOCK = " + str(XBLOCK) + "\n")
fichierSortie.write("YBLOCK = " + str(YBLOCK) + "\n")
fichierSortie.write("XGRID = " + str(XGRID) + "\n")
fichierSortie.write("YGRID = " + str(YGRID) + "\n")
fichierSortie.write("NBTHETA = " + str(NBTHETA) + "\n")
fichierSortie.write("NBPHI = " + str(NBPHI) + "\n")
fichierSortie.write("THSDEG = " + str(THSDEG) + "\n")
fichierSortie.write("LAMBDA = " + str(LAMBDA) + "\n")
fichierSortie.write("TAURAY = " + str(TAURAY) + "\n")
fichierSortie.write("TAUAER = " + str(TAUAER) + "\n")
fichierSortie.write("W0AER = " + str(W0AER) + "\n")
fichierSortie.write("PROFIL = " + str(PROFIL) + "\n")
fichierSortie.write("HA = " + str(HA) + "\n")
fichierSortie.write("HR = " + str(HR) + "\n")
fichierSortie.write("ZMIN = " + str(ZMIN) + "\n")
fichierSortie.write("ZMAX = " + str(ZMAX) + "\n")
fichierSortie.write("WINDSPEED = " + str(WINDSPEED) + "\n")
fichierSortie.write("NH2O = " + str(NH2O) + "\n")
fichierSortie.write("SIM = " + str(SIM) + "\n")
fichierSortie.write("SUR = " + str(NBPHOTONS) + "\n")
fichierSortie.write("DIOPTRE = " + str(DIOPTRE) + "\n")
fichierSortie.write("CONPHY = " + str(CONPHY) + "\n")
fichierSortie.write("DIFFF = " + str(DIFFF) + "\n")
fichierSortie.write("PATHRESULTATSHDF = " + str(PATHRESULTATSHDF) + "\n")
fichierSortie.write("PATHTEMOINHDF = " + str(PATHTEMOINHDF) + "\n")
fichierSortie.write("PATHDIFFAER = " + str(PATHDIFFAER) + "\n")
fichierSortie.write("PATHPROFILATM = " + str(PATHPROFILATM) + "\n")


