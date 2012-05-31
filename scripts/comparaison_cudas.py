#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import warnings
warnings.simplefilter("ignore",DeprecationWarning)
import pyhdf.SD
from pylab import *
import numpy as np


# Fichier qui supperpose les résultats de deux résultats Cuda. 

##################################################
##				FICHIERS A LIRE					##
##################################################

#
# Paramètres à modifier
#
#-----------------------------------------------------------------------------------------------------------------------
type_simu = "molecules_dioptre_agite"
date_simu = "16052012"
angle = "30"
geometrie = "SPHERIQUE"		#Géométrie de l'atmosphère

# Nom du fichier Cuda sans extension .hdf
nom_cuda1 = "out_CUDA_atmos_dioptre_agite_ths=30.00_tRay=0.0533_tAer=0.0000_ws=5.00_v1.2"
# Nom du fichier Fortran sans l'extension .bin.gz
nom_cuda2 = "out_CUDA_atmos_dioptre_agite_ths=30.00_tRay=0.0533_tAer=0.0000_ws=5.00"

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
		type_donnees = "I"
		flag=False
	elif choix == 'q':
		nom_data_cuda = "Valeurs de Q"
		type_donnees = "Q"
		flag=False
	elif choix == 'u':
		nom_data_cuda = "Valeurs de U"
		type_donnees = "U"
		flag=False
	elif choix == 'l':
		nom_data_cuda = "Valeurs de la lumiere polarisee (LP)"
		type_donnees = "LP"
		flag=False
	else:
			print 'Choix incorrect, recommencez'
			
print 'C\'est parti pour la simulation de {0}'.format(type_donnees)


######################################################
##				CHEMIN DES FICHIERS					##
######################################################

# Nom complet du fichier Fortran
path_cuda1 = "/home/florent/MCCuda/validation/"+geometrie+"/"+type_simu+"/simulation_"+date_simu+"/"+ nom_cuda1+".hdf"

# Nom complet du fichier Cuda
path_cuda2 = "/home/florent/MCCuda/validation/"+geometrie+"/"+type_simu+"/simulation_"+date_simu+"/" + nom_cuda2 + ".hdf"

# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = \
"/home/florent/MCCuda/validation/"+geometrie+"/"+type_simu+"/graph_"+date_simu+"/"+type_donnees+"/"+type_donnees+"_CUDA_CUDA_"+\
nom_cuda1


##########################################################
##				DONNEES FICHIER CUDA 1					##
##########################################################

# verification de l'existence du fichier hdf
if os.path.exists(path_cuda1):
	
	# lecture du fichier hdf
	sd_cuda1 = pyhdf.SD.SD(path_cuda1)
	# lecture du nombre de valeurs de phi
	NBPHI_cuda = getattr(sd_cuda1,'NBPHI')
	NBTHETA_cuda = getattr(sd_cuda1,'NBTHETA')
	
	# Récupération des valeurs de theta
	name = "Valeurs de theta echantillonnees"
	hdf_theta = sd_cuda1.select(name)
	theta = hdf_theta.get()
	
	# Récupération des valeurs de phi
	name = "Valeurs de phi echantillonnees"
	hdf_phi = sd_cuda1.select(name)
	phi = hdf_phi.get()
	
	sds_cuda1 = sd_cuda1.select(nom_data_cuda)
	data1 = sds_cuda1.get()		
	
else:
	sys.stdout.write("Pas de fichier "+path_cuda1+"\n")
	sys.exit()


##########################################################
##				DONNEES FICHIER CUDA 2					##
##########################################################

# verification de l'existence du fichier hdf
if os.path.exists(path_cuda2):

	# lecture du fichier hdf
	sd_cuda2 = pyhdf.SD.SD(path_cuda2)

	sds_cuda2 = sd_cuda2.select(nom_data_cuda)
	data2 = sds_cuda2.get()		
	
else:
	sys.stdout.write("Pas de fichier "+path_cuda2+"\n")
	sys.exit()


##############################################################
##				INFORMATION A L'UTILISATEUR					##
##############################################################

sys.stdout.write("\n#-------------------------------------------------------------------------------#\n")
sys.stdout.write("# Fichier cuda1 -> " + path_cuda1 + "\n")
sys.stdout.write("# Fichier cuda2 -> " + path_cuda2 + "\n")
sys.stdout.write("# Résultats stockés dans " + path_dossier_sortie + "\n")
sys.stdout.write("#-------------------------------------------------------------------------------#\n")


os.system("rm -rf "+ path_dossier_sortie)
os.system("mkdir -p "+ path_dossier_sortie)


##################################################################################
##				CREATION/CHOIX/MODIFICATION DE CERTAINES DONNES					##
##################################################################################

# Sauvegarde de la grandeur désirée
data_cuda1 = zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)
data_cuda2 = zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)

data_cuda1 = data1[0:NBPHI_cuda,:]
data_cuda2 = data2[0:NBPHI_cuda,:]

# Infos en commentaire sur le graph
commentaire = type_simu + ' - ' + angle


##########################################################
##				CREATION DES GRAPHIQUES					##
##########################################################

for iphi in xrange(0,NBPHI_cuda,pas_figure):
		
	# initialisation
	listePlots = []
	listeLegends = []
	figure()
	# cuda1
	listePlots.append( plot(theta[dep:fin],data_cuda1[iphi][dep:fin]) )
	listeLegends.append('Cuda1')
	#cuda2
	listePlots.append( plot(theta[dep:fin],data_cuda2[iphi][dep:fin]) )
	listeLegends.append('Cuda2')
	
	# commun
	legend(listePlots, listeLegends, loc='best', numpoints=1)
	title( type_donnees + ' pour comparaison Cuda pour phi='+str(phi[iphi])+' deg' )
	xlabel( 'Theta (deg)' )
	ylabel( type_donnees, rotation='horizontal' )
	figtext(0.25, 0.7, commentaire+" deg", fontdict=None)
	figtext(0, 0, "Date: "+date_simu+"\nFichier cuda1: "+nom_cuda1+"\nFichier cuda2: "+nom_cuda2, fontdict=None,
			size='xx-small')
	grid(True)
	savefig( path_dossier_sortie+'/c_'+type_donnees+'_Cuda_Cuda_phi='+str(phi[iphi])+'.png', dpi=(140) )
	
	##########################################
	##				RAPPORT					##
	##########################################
	
	figure()
	listePlots = []
	listeLegends = []
	listePlots.append( plot( theta[dep:fin], data_cuda1[iphi][dep:fin]/data_cuda2[iphi][dep:fin] ) )
	listeLegends.append('Rapport de '+type_donnees+' Cuda1/Cuda2')
	
	#Régression linéaire
	(ar,br) = polyfit( theta[dep:fin], data_cuda1[iphi][dep:fin]/data_cuda2[iphi][dep:fin] ,1 )
	regLin = polyval( [ar,br],theta[dep:fin] )
	
	listePlots.append( plot(theta[dep:fin], regLin) )
	listeLegends.append( 'Regression lineaire y='+str(ar)+'x+'+str(br) )
	legend( listePlots, listeLegends, loc='best', numpoints=1 )
	
	title( 'Rapport des '+type_donnees+' Cuda_Cuda pour phi='+str(phi[iphi])+' deg' )
	xlabel( 'Theta (deg)' )
	ylabel( 'Rapport des '+type_donnees )
	figtext(0.4, 0.25, commentaire+" deg", fontdict=None)
	figtext(0, 0, "Date: "+date_simu+"\nFichier cuda1: "+nom_cuda1+"\nFichier cuda2: "+nom_cuda2, fontdict=None,
			size='xx-small')
	grid(True)
	savefig( path_dossier_sortie+'/rapport_'+type_donnees+'_Cuda_Cuda_phi=' +str(phi[iphi])+'.png', dpi=(140) )
	
	##########################################
	##				DIFFERENCE				##
	##########################################
	figure()
	plot( theta[dep:fin], data_cuda1[iphi][dep:fin]-data_cuda2[iphi][dep:fin] )
	title( 'Difference des '+type_donnees+ ' Cuda1 - Cuda2 pour phi='+str(phi[iphi])+' deg' )
	xlabel( 'Theta (deg)' )
	ylabel( 'Difference des '+type_donnees )
	figtext(0.4, 0.25, commentaire+" deg", fontdict=None)
	figtext(0, 0, "Date: "+date_simu+"\nFichier cuda1: "+nom_cuda1+"\nFichier cuda2: "+nom_cuda2, fontdict=None,
			size='xx-small')
	grid(True)
	savefig( path_dossier_sortie+'/difference_'+type_donnees+'_Cuda_Cuda_phi='+str(phi[iphi])+'.png', dpi=(140) )



##########################################################
##				CREATION FICHIER PARAMETRE				##
##########################################################

# creation du fichier contenant les parametres de la simulation
fichierSortie = open(path_dossier_sortie+'/Parametres.txt', 'w')

# Récupération des données de Cuda1
NBPHOTONS = getattr(sd_cuda1,'NBPHOTONS')
tempsEcoule = getattr(sd_cuda1,'tempsEcoule')
NBLOOP = getattr(sd_cuda1,'NBLOOP')
SEED = getattr(sd_cuda1,'SEED')
XBLOCK = getattr(sd_cuda1,'XBLOCK')
YBLOCK = getattr(sd_cuda1,'YBLOCK')
XGRID = getattr(sd_cuda1,'XGRID')
YGRID = getattr(sd_cuda1,'YGRID')
NBTHETA = getattr(sd_cuda1,'NBTHETA')
NBPHI = getattr(sd_cuda1,'NBPHI')
THSDEG = getattr(sd_cuda1,'THSDEG')
LAMBDA = getattr(sd_cuda1,'LAMBDA')
TAURAY = getattr(sd_cuda1,'TAURAY')
TAUAER = getattr(sd_cuda1,'TAUAER')
W0AER = getattr(sd_cuda1,'W0AER')
PROFIL = getattr(sd_cuda1,'PROFIL')
HA = getattr(sd_cuda1,'HA')
HR = getattr(sd_cuda1,'HR')
ZMIN = getattr(sd_cuda1,'ZMIN')
ZMAX = getattr(sd_cuda1,'ZMAX')
WINDSPEED = getattr(sd_cuda1,'WINDSPEED')
NH2O = getattr(sd_cuda1,'NH2O')
SIM = getattr(sd_cuda1,'SIM')
SUR = getattr(sd_cuda1,'SUR')
DIOPTRE = getattr(sd_cuda1,'DIOPTRE')
CONPHY = getattr(sd_cuda1,'CONPHY')
DIFFF = getattr(sd_cuda1,'DIFFF')
PATHRESULTATSHDF = getattr(sd_cuda1,'PATHRESULTATSHDF')
PATHTEMOINHDF = getattr(sd_cuda1,'PATHTEMOINHDF')

fichierSortie.write('\n\n***Paramètres de Cuda1***\n\n')

# Ecriture dans le fichier
fichierSortie.write('NBPHOTONS = {0:.2e}\n'.format(NBPHOTONS))
fichierSortie.write('Temps écoulé = {0:.2e}\n'.format(tempsEcoule))
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



# Récupération des données de Cuda2
NBPHOTONS = getattr(sd_cuda2,'NBPHOTONS')
tempsEcoule = getattr(sd_cuda2,'tempsEcoule')
NBLOOP = getattr(sd_cuda2,'NBLOOP')
SEED = getattr(sd_cuda2,'SEED')
XBLOCK = getattr(sd_cuda2,'XBLOCK')
YBLOCK = getattr(sd_cuda2,'YBLOCK')
XGRID = getattr(sd_cuda2,'XGRID')
YGRID = getattr(sd_cuda2,'YGRID')
NBTHETA = getattr(sd_cuda2,'NBTHETA')
NBPHI = getattr(sd_cuda2,'NBPHI')
THSDEG = getattr(sd_cuda2,'THSDEG')
LAMBDA = getattr(sd_cuda2,'LAMBDA')
TAURAY = getattr(sd_cuda2,'TAURAY')
TAUAER = getattr(sd_cuda2,'TAUAER')
W0AER = getattr(sd_cuda2,'W0AER')
PROFIL = getattr(sd_cuda2,'PROFIL')
HA = getattr(sd_cuda2,'HA')
HR = getattr(sd_cuda2,'HR')
ZMIN = getattr(sd_cuda2,'ZMIN')
ZMAX = getattr(sd_cuda2,'ZMAX')
WINDSPEED = getattr(sd_cuda2,'WINDSPEED')
NH2O = getattr(sd_cuda2,'NH2O')
SIM = getattr(sd_cuda2,'SIM')
SUR = getattr(sd_cuda2,'SUR')
DIOPTRE = getattr(sd_cuda2,'DIOPTRE')
CONPHY = getattr(sd_cuda2,'CONPHY')
DIFFF = getattr(sd_cuda2,'DIFFF')
PATHRESULTATSHDF = getattr(sd_cuda2,'PATHRESULTATSHDF')
PATHTEMOINHDF = getattr(sd_cuda2,'PATHTEMOINHDF')

fichierSortie.write('\n\n***Paramètres de Cuda2***\n\n')

# Ecriture dans le fichier
fichierSortie.write('NBPHOTONS = {0:.2e}\n'.format(NBPHOTONS))
fichierSortie.write('Temps écoulé = {0:.2e}\n'.format(tempsEcoule))
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

fichierSortie.close()

print '################################################'
print 'Simulation de {0} terminee pour Cuda-Fortran'.format(type_donnees)
print '################################################'
