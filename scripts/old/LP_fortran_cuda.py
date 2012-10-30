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

##################################################
##				FICHIERS A LIRE					##
##################################################

#
# Paramètres à modifier
#
#-----------------------------------------------------------------------------------------------------------------------
type_simu = "molecules_seules"
date_simu = "23022012"
# Nom du fichier Cuda sans extension .hdf
nom_cuda = "out_CUDA_atmos_tRay=0.0533_tAer=0.0000_diff=1_ths=70.00"
# Nom du fichier Fortran sans l'extension .bin.gz
nom_fortran = "out_FORTRAN.ran=0002.wav=443.ths=70.000.tr=0.0533.ta=0.0000.difff=0001.pi0=0.967.H=002.000"
#-----------------------------------------------------------------------------------------------------------------------

# Nom complet du fichier Fortran
path_fortran = "/home/florent/MCCuda/validation/"+type_simu+"/simulation_"+date_simu+"/"+ nom_fortran+".bin.gz"

# Nom complet du fichier Cuda
path_cuda = "/home/florent/MCCuda/validation/"+type_simu+"/simulation_"+date_simu+"/" + nom_cuda + ".hdf"

# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = "/home/florent/MCCuda/validation/"+type_simu+"/graph_"+date_simu+"/LP/LP_FORTRAN_CUDA_" + nom_cuda
os.system("rm -rf "+ path_dossier_sortie)
os.system("mkdir -p "+ path_dossier_sortie)


##########################################################
##				DONNEES FICHIER FORTRAN					##
##########################################################

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
if os.path.exists(path_fortran):
	file_fortran_bin = gzip.open(path_fortran)

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

# Données de sortie dans le tableau fortran : tab_fortran['real_refl'][grandeur,itheta,iphi,type d'evenement subi par le photon]
# avec grandeur : 	0 pour I
#					1 pour Q
#					2 pour U

else:
	sys.stdout.write("Pas de fichier "+path_fortran+"\n")
	sys.exit()


##########################################################
##				DONNEES FICHIER CUDA					##
##########################################################

# verification de l'existence du fichier hdf
if os.path.exists(path_cuda):
	# on vide le dossier de sortie du script
	os.system("rm -rf "+path_dossier_sortie)
	os.mkdir(path_dossier_sortie)
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

	name = "Valeurs de la lumiere polarisee (LP)"
	sds_cuda = sd_cuda.select(name)
	data = sds_cuda.get()		
	
else:
	sys.stdout.write("Pas de fichier "+path_cuda+"\n")
	sys.exit()


##############################################################
##				INFORMATION A L'UTILISATEUR					##
##############################################################
			
sys.stdout.write("\n#-------------------------------------------------------------------------------#\n")
sys.stdout.write("# Le fichier cuda est " + path_cuda + "\n")
sys.stdout.write("# Le fichier fortran est " + path_fortran + "\n")
sys.stdout.write("# Les résultats sont stockés dans " + path_dossier_sortie + "\n")
sys.stdout.write("#-------------------------------------------------------------------------------#\n")
	

##############################################################################
##				CREATION/MODIFICATION DE CERTAINES DONNES					##
##############################################################################

# Sauvegarde de la grandeur désirée, ici la luminance polarisée (LP)
data_fortran = zeros((NBPHI_fortran, NTHV), dtype=float)
data_cuda = zeros((NBPHI_cuda/2, NBTHETA_cuda), dtype=float)

# Il a été remarqué que pour un indice de theta donné, la boite n'est pas la même en Fortran ou en Cuda. C'est pourquoi
# pour comparer les mêmes boites, il faut prendre l'indice i en Cuda et l'indice i-1 en Fortran. La ième boite Cuda correspond à la
# ième-1 boite en Fortran

for iphi in xrange(0,NBPHI_fortran):
	for i in xrange(NTHV):
		data_fortran[iphi][i]=math.sqrt(pow(tab_fortran['real_refl'][1,i-1,iphi,0],2)+pow( tab_fortran['real_refl'][2,i-1,iphi,0],2))

# Pour comparer les 2 resultats il faut que phi parcourt un meme intervalle et qu'il y ait le meme nombre de boites selon phi
# Fortran :  intervalle=[0,PI]   nombre_de_boites=NBPHI_fortran
# Cuda :     intervalle=[0,2PI]  nombre_de_boites=NBPHI_cuda
# On va projeter les resultats du cuda sur [0,PI]

for iphi in xrange(0,NBPHI_cuda/2):
	for ith in xrange(NBTHETA_cuda):
		data_cuda[iphi][ith] = ( data[iphi,ith]+data[NBPHI_cuda-iphi-1,ith] )/2


##########################################################
##				CREATION DES GRAPHIQUES					##
##########################################################

dep = 6		# Indice de départ pour le tracé
fin = 177	# Indice de fin pour le tracé
type_donnees = 'LP'	# Nom à insérer dans la légende

# REMARQUE: Les indices ci-dessus ont été mis en place car ils permettent de rogner la simulation si nécessaire.
# Les bords peuvent fausser les graphiques.

if (NBPHI_cuda/2) == NBPHI_fortran:
	for iphi in xrange(0,NBPHI_cuda/2,5):
			
		# initialisation
		listePlots = []
		listeLegends = []
		figure()
		# fortran
		#listePlots.append(plot(tab_fortran['real_thv_bornes'][dep-1:fin-1], tab_fortran['real_refl'][0, dep-1:fin-1, iphi, 0]))
		listePlots.append( plot(theta[dep:fin], data_fortran[iphi][dep:fin]) )
		listeLegends.append('Fortran')
		#cuda
		listePlots.append( plot(theta[dep:fin],data_cuda[iphi][dep:fin]) )
		listeLegends.append('Cuda')
		
		# commun
		legend(listePlots, listeLegends, loc='best', numpoints=1)
		title( type_donnees + ' pour Cuda et Fortran a phi='+str(phi[iphi])+' deg' )
		xlabel( 'Theta (deg)' )
		ylabel( type_donnees )
		grid(True)
		savefig( path_dossier_sortie+'/c_'+type_donnees+'_Fortran_Cuda_phi='+str(phi[iphi])+'.png', dpi=(140) )
		
		##########################################
		##				RAPPORT					##
		##########################################
		
		figure()
		listePlots = []
		listeLegends = []
		listePlots.append( plot( theta[dep:fin], data_fortran[iphi][dep:fin]/data_cuda[iphi][dep:fin] ) )
		listeLegends.append('Rapport de '+type_donnees+' Fortran/Cuda')
		
		#Régression linéaire
		(ar,br) = polyfit( theta[dep:fin], data_fortran[iphi][dep:fin]/data_cuda[iphi][dep:fin] ,1 )
		regLin = polyval( [ar,br],theta[dep:fin] )
		
		listePlots.append( plot(theta[dep:fin], regLin) )
		listeLegends.append( 'Regression lineaire y='+str(ar)+'x+'+str(br) )
		legend( listePlots, listeLegends, loc='best', numpoints=1 )
		
		title( 'Rapport des '+type_donnees+' Fortran_Cuda pour phi='+str(phi[iphi])+' deg' )
		xlabel( 'Theta (deg)' )
		ylabel( 'Rapport des '+type_donnees )
		grid(True)
		savefig( path_dossier_sortie+'/rapport_'+type_donnees+'_Fortran_Cuda_phi='+str(phi[iphi])+'.png', dpi=(140) )
		figure()
		
		##########################################
		##				DIFFERENCE				##
		##########################################
		
		plot( theta[dep:fin], data_fortran[iphi][dep:fin]-data_cuda[iphi][dep:fin] )
		title( 'Difference des '+type_donnees+ ' Fortran - Cuda pour phi='+str(phi[iphi])+' deg' )
		xlabel( 'Theta (deg)' )
		ylabel( 'Difference des '+type_donnees )
		grid(True)
		savefig( path_dossier_sortie+'/difference_'+type_donnees+'_Fortran_Cuda_phi='+str(phi[iphi])+'.png', dpi=(140) )

	
else:
	sys.stdout.write('Les tableaux ne font pas la meme taille\n')
	sys.exit()


##########################################################
##				CREATION FICHIER PARAMETRE				##
##########################################################

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
fichierParametres.write('NBPHOTONS = {0:.2e}\n'.format(NBPHOTONS))
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

fichierParametres.write("\n\n##### Paramètres Fortran #####\n")
fichierParametres.write("Nombres de photons: {0:.2e}\n".format(tab_fortran['nphotons']))
fichierParametres.write("Germe: " + str(tab_fortran['initgerme']) + "\n")
fichierParametres.write("\n\n")

fichierParametres.close()
		
