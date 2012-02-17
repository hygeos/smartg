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
nom_hdf = "atmos_dioptre_agitee_tauRay=0.001000_tauAer=0.000000_ths=70.000000_ws=5.000000"
path_hdf = "../out_prog/Resultats_" + nom_hdf + ".hdf"

# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = "../out_scripts/quantification_bruit/quantification_bruit_" + nom_hdf

path_ref = "/home/florent/entree/new_out_SOS_toray_0.001_ths_70_vent_5_MU400.txt"

os.system("rm -rf "+ path_dossier_sortie)
os.system("mkdir -p "+ path_dossier_sortie)


# verification de l'existence du fichier hdf
if os.path.exists(path_hdf):
	# on vide le dossier de sortie du script
	#os.system("rm -rf "+path_dossier_sortie)
	#os.mkdir(path_dossier_sortie)
	# lecture du fichier hdf
	hdf_cuda = pyhdf.SD.SD(path_hdf)
	# lecture du nombre de valeurs de phi
	NBPHI_cuda = getattr(hdf_cuda,'NBPHI')
	
	# Récupération des valeurs de theta
	name = "Valeurs de theta echantillonnees"
	hdf_theta = hdf_cuda.select(name)
	theta = hdf_theta.get()
	NBTHETA = getattr(hdf_cuda,'NBTHETA')
	
	# Récupération des valeurs de phi
	name = "Valeurs de phi echantillonnees"
	hdf_phi = hdf_cuda.select(name)
	phi = hdf_phi.get()	

	# lecture du dataset
	name = "Valeur de la reflectance pour un phi et theta donnes"
	sds_hdf = hdf_cuda.select(name)
	# recuperation du tableau et de la valeur de phi
	data_hdf = sds_hdf.get()
		
else:
	sys.stdout.write("Pas de fichier "+path_hdf+"\n")
	sys.exit()
	
# Récupération des données de références
if os.path.exists(path_ref):
	
	# data_ref[iphi][ith] = reflectance
	# ith est le num de la ith-ème boite theta. Boites de pas de 0.5 centrées tous les 0.5
	# iphi est le num de la ith-ème boite phi
	data_ref = zeros((NBPHI_cuda/2,2*(NBTHETA-1)),dtype=float)
	fic_ref = open(path_ref, "r")
	
	for ligne in fic_ref:
		donnees = ligne.rstrip('\n\r').split("\t")
		#data_ref[0][0]= float(donnees[0])	
		#data_ref[0][1]= float(donnees[1])
		#data_ref[0][2]= float(donnees[2])
		if float(donnees[1]) < 89.6:
			data_ref[int(float(donnees[0]))][int(2*float(donnees[1]))] = float(donnees[2])
			# print 'data_ref[{0}][{1}] = {2}'.format(int(float(donnees[0])),int(2*float(donnees[1])),float(donnees[2]))
		
	fic_ref.close()
		
else:
	sys.stdout.write("Pas de fichier "+path_ref+"\n")
	sys.exit()
	
sys.stdout.write(" ----------------------------------------------------------------\n")
sys.stdout.write("| Le fichier traité est " + path_hdf + "\t|\n")
sys.stdout.write("| Le fichier de référence est " + path_ref + "\t|\n")
sys.stdout.write("| Les résultats sont stockés dans " + path_dossier_sortie + "\t|\n")
sys.stdout.write(" ----------------------------------------------------------------\n")
	
			 #############################
			# Création de graphiques	#
			############################

dep = 2			# Mettre 1 au minimum
fin = NBTHETA-1

# initialisation
listePlots = []
listeLegends = []

for iphi in xrange(0,NBPHI_cuda/2,5):
	
	listePlots = []
	listeLegends = []
	# Référence
	listePlots.append(plot(theta[dep:fin], data_ref[iphi][dep:fin]))
	listeLegends.append('Code SOS')

	# Cuda
	listePlots.append(plot(theta[dep:fin], data_hdf[iphi][dep:fin]))
	listeLegends.append('Cuda')

	# commun
	legend(listePlots, listeLegends, loc='best', numpoints=1)
	title('Comparaison references et cuda pour phi='+str(phi[iphi])+" deg")
	xlabel('Theta (deg)')
	ylabel('Reflectance')
	grid(True)
	savefig(path_dossier_sortie+"/comparaison_phi="+str(phi[iphi])+".png", dpi=(140))
	figure()

		##########################################
		#	Figures d'analyse plus spcifiques	#
		########################################
		
# Figure d'évaluation du taux d'erreur - RAPPORT
# phi = 0

	listePlots = []
	listeLegends = []
	listePlots.append( plot(theta[dep:fin],(data_ref[iphi][dep:fin])/
								((data_hdf[iphi,dep:fin]+data_hdf[NBPHI_cuda-iphi-1,dep:fin])/2)) )
	listeLegends.append('Rapport SOS/Cuda')
	#Régression linéaire

	(ar,br)=polyfit(theta[dep:fin],( data_ref[iphi][dep:fin])/((data_hdf[iphi,dep:fin]+data_hdf[NBPHI_cuda-iphi-1,dep:fin])/2) ,1)
	regLin=polyval([ar,br],theta[dep:fin])

	listePlots.append( plot(theta[dep:fin], regLin) )
	listeLegends.append('Regression lineaire y='+str(ar)+'x+'+str(br))
	legend(listePlots, listeLegends, loc='best', numpoints=1)

	title("Rapport des resultats SOS et Cuda pour phi="+str(phi[iphi])+" deg")
	xlabel("Theta (deg)")
	ylabel("Rapport des reflactances SOS/Cuda")
	grid(True)
	savefig(path_dossier_sortie+"/rapport_reflectance_phi="+str(phi[iphi])+".png", dpi=(140))
	figure()


# Figure d'évaluation du taux d'erreur - DIFFERENCE
	plot(theta[dep:fin],data_ref[iphi][dep:fin]-((data_hdf[iphi,dep:fin]+data_hdf[NBPHI_cuda-iphi-1,dep:fin])/2) )
	title("Difference des resultats SOS - Cuda pour phi="+str(phi[iphi])+" deg")
	xlabel("Theta (deg)")
	ylabel("Difference des reflactances")
	grid(True)
	savefig(path_dossier_sortie+"/difference_reflectance_phi="+str(phi[iphi])+".png", dpi=(140))
	figure()

	          ###############################
	         # CREATION FICHIER DE SORTIE  #
	        ###############################

NBPHOTONS = getattr(hdf_cuda,'NBPHOTONS')
NBLOOP = getattr(hdf_cuda,'NBLOOP')
SEED = getattr(hdf_cuda,'SEED')
XBLOCK = getattr(hdf_cuda,'XBLOCK')
YBLOCK = getattr(hdf_cuda,'YBLOCK')
XGRID = getattr(hdf_cuda,'XGRID')
YGRID = getattr(hdf_cuda,'YGRID')
NBTHETA = getattr(hdf_cuda,'NBTHETA')
NBPHI = getattr(hdf_cuda,'NBPHI')
THSDEG = getattr(hdf_cuda,'THSDEG')
LAMBDA = getattr(hdf_cuda,'LAMBDA')
TAURAY = getattr(hdf_cuda,'TAURAY')
TAUAER = getattr(hdf_cuda,'TAUAER')
W0AER = getattr(hdf_cuda,'W0AER')
PROFIL = getattr(hdf_cuda,'PROFIL')
HA = getattr(hdf_cuda,'HA')
HR = getattr(hdf_cuda,'HR')
ZMIN = getattr(hdf_cuda,'ZMIN')
ZMAX = getattr(hdf_cuda,'ZMAX')
WINDSPEED = getattr(hdf_cuda,'WINDSPEED')
NH2O = getattr(hdf_cuda,'NH2O')
SIM = getattr(hdf_cuda,'SIM')
SUR = getattr(hdf_cuda,'SUR')
DIOPTRE = getattr(hdf_cuda,'DIOPTRE')
CONPHY = getattr(hdf_cuda,'CONPHY')
DIFFF = getattr(hdf_cuda,'DIFFF')
PATHRESULTATSHDF = getattr(hdf_cuda,'PATHRESULTATSHDF')
PATHTEMOINHDF = getattr(hdf_cuda,'PATHTEMOINHDF')
PATHDIFFAER = getattr(hdf_cuda,'PATHDIFFAER')
PATHPROFILATM = getattr(hdf_cuda,'PATHPROFILATM')

# creation du fichier contenant les parametres de la simulation
sortie = open(path_dossier_sortie+"/Quantification_bruit_"+nom_hdf+".txt", "w")
sortie.write(" ##### Paramètres CUDA #####\n") 
sortie.write("NBPHOTONS = " + str(NBPHOTONS) + "\n")
sortie.write("NBLOOP = " + str(NBLOOP) + "\n")
sortie.write("SEED = " + str(SEED) + "\n")	
sortie.write("XBLOCK = " + str(XBLOCK) + "\n")
sortie.write("YBLOCK = " + str(YBLOCK) + "\n")
sortie.write("XGRID = " + str(XGRID) + "\n")
sortie.write("YGRID = " + str(YGRID) + "\n")
sortie.write("NBTHETA = " + str(NBTHETA) + "\n")
sortie.write("NBPHI = " + str(NBPHI) + "\n")
sortie.write("THSDEG = " + str(THSDEG) + "\n")
sortie.write("LAMBDA = " + str(LAMBDA) + "\n")
sortie.write("TAURAY = " + str(TAURAY) + "\n")
sortie.write("TAUAER = " + str(TAUAER) + "\n")
sortie.write("W0AER = " + str(W0AER) + "\n")
sortie.write("PROFIL = " + str(PROFIL) + "\n")
sortie.write("HA = " + str(HA) + "\n")
sortie.write("HR = " + str(HR) + "\n")
sortie.write("ZMIN = " + str(ZMIN) + "\n")
sortie.write("ZMAX = " + str(ZMAX) + "\n")
sortie.write("WINDSPEED = " + str(WINDSPEED) + "\n")
sortie.write("NH2O = " + str(NH2O) + "\n")
sortie.write("SIM = " + str(SIM) + "\n")
sortie.write("SUR = " + str(NBPHOTONS) + "\n")
sortie.write("DIOPTRE = " + str(DIOPTRE) + "\n")
sortie.write("CONPHY = " + str(CONPHY) + "\n")
sortie.write("DIFFF = " + str(DIFFF) + "\n")
sortie.write("PATHRESULTATSHDF = " + str(PATHRESULTATSHDF) + "\n")
sortie.write("PATHTEMOINHDF = " + str(PATHTEMOINHDF) + "\n")
sortie.write("PATHDIFFAER = " + str(PATHDIFFAER) + "\n")
sortie.write("PATHPROFILATM = " + str(PATHPROFILATM) + "\n")

###########################################################

# Calcul de statistique

moyDiff = 0		# Moyenne de Ref-Cuda
sigmaDiff = 0
moyDiffAbs = 0	# Moyenne de |Ref-Cuda|
sigmaDiffAbs = 0
moyRap = 0		# Moyenne de Ref/Cuda
sigmaRap = 0
moyRapAbs = 0	# Moyenne de |Ref/Cuda|
sigmaRapAbs = 0


# Calcul des moyennes
for iphi in xrange(NBPHI_cuda/2):
	for ith in xrange(dep,fin):	# Calcul sur tout l'espace
		moyDiff += data_ref[iphi][ith]-(data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2
		moyDiffAbs += abs(data_ref[iphi][ith]-(data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2 )
		moyRap += data_ref[iphi][ith]/((data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2)
		moyRapAbs += abs(1-data_ref[iphi][ith]/((data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2) )

moyDiff = moyDiff/((fin-dep+1)*NBPHI_cuda/2)
moyDiffAbs = moyDiffAbs/((fin-dep+1)*NBPHI_cuda/2)
moyRap = moyRap/((fin-dep+1)*NBPHI_cuda/2)
moyRapAbs = moyRapAbs/((fin-dep+1)*NBPHI_cuda/2)
	
	
for iphi in xrange(NBPHI_cuda/2):
	for ith in xrange(dep,fin):	# Calcul sur tout l'espace
		
		# Calcul des écarts type
		sigmaDiff += pow( moyDiff - (data_ref[iphi][ith]-(data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2 ) ,2.0 )
		sigmaDiffAbs += pow( moyDiffAbs - abs( data_ref[iphi][ith]-((data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2) ) ,2.0 )
		sigmaRap += pow( moyRap - ( data_ref[iphi][ith]/((data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2) ) ,2.0 )
		sigmaRapAbs += pow( moyRapAbs - abs(1-data_ref[iphi][ith]/((data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2) ) ,2.0 )

	
sigmaDiff = math.sqrt( sigmaDiff/((fin-dep)*NBPHI_cuda/2) )
sigmaDiffAbs = math.sqrt( sigmaDiffAbs/((fin-dep)*NBPHI_cuda/2) )
sigmaRap = math.sqrt( sigmaRap/((fin-dep)*NBPHI_cuda/2) )
sigmaRapAbs = math.sqrt( sigmaRapAbs/((fin-dep)*NBPHI_cuda/2) )


print "\n====================:Résultats:===================="
print 'Moyenne de la différence Ref-Cuda = {0:.4e}'.format(moyDiff)
print 'Ecart type de la différence Ref-Cuda = {0:.4e}\n'.format(sigmaDiff)

print 'Moyenne de la valeur absolue de la différence = {0:.4e}'.format(moyDiffAbs)
print 'Ecart type de la valeur absolue de la différence = {0:.4e}\n'.format(sigmaDiffAbs)

print 'Moyenne du rapport Ref/Cuda = {0:.4e}'.format(moyRap)
print 'Ecart type du rapport Ref/Cuda = {0:.4e}\n'.format(sigmaRap)

print 'Pourcentage erreur moyen Ref/Cuda = {0:.4e} %'.format(moyRapAbs*100)
print 'Ecart type de erreur = {0:.4e}\n'.format(sigmaRapAbs)
print "==================================================="

sortie.write( "\n====================:Résultats:====================\n")
sortie.write( 'Moyenne de la différence Ref-Cuda = {0:.4e}\n'.format(moyDiff))
sortie.write( 'Ecart type de la différence Ref-Cuda = {0:.4e}\n\n'.format(sigmaDiff))

sortie.write( 'Moyenne de la valeur absolue de la différence = {0:.4e}\n'.format(moyDiffAbs))
sortie.write( 'Ecart type de la valeur absolue de la différence = {0:.4e}\n\n'.format(sigmaDiffAbs))

sortie.write( 'Moyenne du rapport Ref/Cuda = {0:.4e}\n'.format(moyRap))
sortie.write( 'Ecart type du rapport Ref/Cuda = {0:.4e}\n\n'.format(sigmaRap))

sortie.write( 'Pourcentage erreur moyen Ref/Cuda = {0:.4e} %\n'.format(moyRapAbs*100))
sortie.write( 'Ecart type de erreur = {0:.4e}\n'.format(sigmaRapAbs))
sortie.write( "===================================================\n")

# calculs par couronne de theta
# Dth de 10 deg, il y aura donc NBTHETA/180*10 échantillons par couronnes car theta=[0;180°]
pas = NBTHETA/180*10
ith0 = 1
	
for icouronne in xrange( NBTHETA/(NBTHETA/180*10) ):	# Pour chaque couronne
	
	if icouronne == 17 or icouronne==0:	#on modifie les paramètres pour éviter le débodement de tableau sur la dernière couronne
		pas=NBTHETA/180*10 - 1
	
	else:
		pas = NBTHETA/180*10
		
	moyDiff = 0		# Moyenne de Fortran-Cuda
	sigmaDiff = 0
	moyDiffAbs = 0	# Moyenne de |Fortran-Cuda|
	sigmaDiffAbs = 0
	moyRap = 0		# Moyenne de Fortran/Cuda
	sigmaRap = 0
	moyRapAbs = 0	# Moyenne de |Fortran/Cuda|
	sigmaRapAbs = 0
	
	# Calcul des moyennes
	for ith in xrange(ith0,ith0+pas):
		for iphi in xrange(NBPHI_cuda/2):
			
			moyDiff += data_ref[iphi][ith]-((data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2)
			moyDiffAbs += abs(data_ref[iphi][ith]-(data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2 )
			moyRap += data_ref[iphi][ith]/((data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2)
			moyRapAbs += abs(1-data_ref[iphi][ith]/((data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2) )
			
	moyDiff = moyDiff/(pas*NBPHI_cuda/2)
	moyDiffAbs = moyDiffAbs/(pas*NBPHI_cuda/2)
	moyRap = moyRap/(pas*NBPHI_cuda/2)
	moyRapAbs = moyRapAbs/(pas*NBPHI_cuda/2)

	# Calcul des écarts type
	for ith in xrange(ith0,ith0+pas):
		for iphi in xrange(NBPHI_cuda/2):
			
			sigmaDiff += pow( moyDiff - (data_ref[iphi][ith]-(data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2 ) ,2.0 )
			sigmaDiffAbs += pow( moyDiffAbs - abs( data_ref[iphi][ith]-((data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2) ) ,2.0 )
			sigmaRap += pow( moyRap - ( data_ref[iphi][ith]/((data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2) ) ,2.0 )
			sigmaRapAbs += pow( moyRapAbs - abs(1-data_ref[iphi][ith]/((data_hdf[iphi,ith]+data_hdf[NBPHI_cuda-iphi-1,ith])/2) ) ,2.0 )

	sigmaDiff = math.sqrt( sigmaDiff/(pas*NBPHI_cuda/2) )
	sigmaDiffAbs = math.sqrt( sigmaDiffAbs/(pas*NBPHI_cuda/2) )
	sigmaRap = math.sqrt( sigmaRap/(pas*NBPHI_cuda/2) )
	sigmaRapAbs = math.sqrt( sigmaRapAbs/(pas*NBPHI_cuda/2) )
			

	print "\n====================:Résultats par couronne:===================="
	print '==================:Couronne #{0:2d} -{1:3d}->{2:3d} deg==================='.format(ith0/10,ith0*90/NBTHETA,
(ith0+pas)*90/NBTHETA)
	print 'Moyenne de la différence Ref-Cuda = {0:.4e}'.format(moyDiff)
	print 'Ecart type de la différence Ref-Cuda = {0:.4e}\n'.format(sigmaDiff)

	print 'Moyenne de la valeur absolue de la différence = {0:.4e}'.format(moyDiffAbs)
	print 'Ecart type de la valeur absolue de la différence = {0:.4e}\n'.format(sigmaDiffAbs)

	print 'Moyenne du rapport Ref/Cuda = {0:.4e}'.format(moyRap)
	print 'Ecart type du rapport Ref/Cuda = {0:.4e}\n'.format(sigmaRap)

	print 'Pourcentage erreur Ref/Cuda = {0:.4e} %'.format(moyRapAbs*100)
	print 'Ecart type de erreur = {0:.4e}\n'.format(sigmaRapAbs)
	print "================================================================"

	sortie.write( "\n====================:Résultats par couronne:====================\n")
	sortie.write( '==================:Couronne #{0:2d} -{1:3d}->{2:3d} deg===================\n'.format(ith0/10,ith0*90/NBTHETA,
(ith0+pas)*90/NBTHETA))
	sortie.write( 'Moyenne de la différence Ref-Cuda = {0:.4e}\n'.format(moyDiff))
	sortie.write( 'Ecart type de la différence Ref-Cuda = {0:.4e}\n\n'.format(sigmaDiff))

	sortie.write( 'Moyenne de la valeur absolue de la différence = {0:.4e}\n'.format(moyDiffAbs))
	sortie.write( 'Ecart type de la valeur absolue de la différence = {0:.4e}\n\n'.format(sigmaDiffAbs))

	sortie.write( 'Moyenne du rapport Ref/Cuda = {0:.4e}\n'.format(moyRap))
	sortie.write( 'Ecart type du rapport Ref/Cuda = {0:.4e}\n\n'.format(sigmaRap))

	sortie.write( 'Pourcentage erreur Ref/Cuda = {0:.4e} %\n'.format(moyRapAbs*100))
	sortie.write( 'Ecart type de erreur Ref/Cuda = {0:.4e}\n'.format(sigmaRapAbs))
	sortie.write( "================================================================\n")

	ith0 += pas


sortie.close()
