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
type_simu = "atmos_seule"
date_simu = "04052012"
angle = '30'
geometrie = "PARALLELE"		#Géométrie de l'atmosphère

# Nom du fichier sos sans extension .hdf
nom_sos = "out_SOS_toray_0.0533_toaer_0.1_T70"
# Nom du fichier Fortran sans l'extension .bin.gz
nom_fortran = "out.ran=7543.wav=443.ths=30.000.tr=0.0533.ta=0.1000.difff=0000.pi0=0.967.H=002.000.mod=pf.txt"


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
		colonne_donnee_sos = 2
		type_donnees = "I"
		flag=False
	elif choix == 'q':
		colonne_donnee_sos = 3
		type_donnees = "Q"
		flag=False
	elif choix == 'u':
		colonne_donnee_sos = 4
		type_donnees = "U"
		flag=False
	elif choix == 'l':
		colonne_donnee_sos = 5
		type_donnees = "LP"
		flag=False
	else:
		print 'Choix incorrect, recommencez'
	
print 'C\'est parti pour la simulation de {0}'.format(type_donnees)


######################################################
##				CHEMIN DES FICHIERS					##
######################################################

# Nom complet du fichier SOS
path_sos = "/home/florent/MCCuda/validation/fichier_ref_sos/" + nom_sos + ".txt"

# Nom complet du fichier Fortran
path_fortran = "/home/florent/MCCuda/validation/"+geometrie+"/"+type_simu+"/simulation_"+date_simu+"/"+ nom_fortran+".bin.gz"

# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = \
"/home/florent/MCCuda/validation/"+geometrie+"/"+type_simu+"/graph_"+date_simu+"/"+type_donnees+"/"+type_donnees+"_FORTRAN_SOS_" + \
nom_sos


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


##################################################
##				DONNEES FICHIER SOS				##
##################################################

# verification de l'existence du fichier SOS
if os.path.exists(path_sos):
	
	# data_sos[iphi][ith] = grandeur
	# ith est le num de la ith-ème boite theta. Boites de pas de 0.5 centrées tous les 0.5
	# iphi est le num de la ith-ème boite phi
	data_sos = zeros((NBPHI_fortran,2*(NTHV-1)),dtype=float)
	fichier_sos = open(path_sos, "r")
	
	for ligne in fichier_sos:
		donnees = ligne.rstrip('\n\r').split("\t")		# Lecture
		if donnees[0]=='':
			donnees = donnees[1:]						# Suppression des possibles tabulations en début de ligne
		
		if float(donnees[1]) < 89.6:
			data_sos[int(float(donnees[0]))][int(2*float(donnees[1]))] = float(donnees[colonne_donnee_sos])
		
	fichier_sos.close()
		
else:
	sys.stdout.write("Pas de fichier "+path_sos+"\n")
	sys.exit()


##############################################################
##				INFORMATION A L'UTILISATEUR					##
##############################################################

sys.stdout.write("\n#-------------------------------------------------------------------------------#\n")
sys.stdout.write("# Le fichier sos est " + path_sos + "\n")
sys.stdout.write("# Le fichier fortran est " + path_fortran + "\n")
sys.stdout.write("# Les résultats sont stockés dans " + path_dossier_sortie + "\n")
sys.stdout.write("#-------------------------------------------------------------------------------#\n")
	

os.system("rm -rf "+ path_dossier_sortie)
os.system("mkdir -p "+ path_dossier_sortie)


##################################################################################
##				CREATION/CHOIX/MODIFICATION DE CERTAINES DONNES					##
##################################################################################

# Sauvegarde de la grandeur désirée
data_fortran = zeros((NBPHI_fortran, NTHV), dtype=float)
#theta = zeros((1, NTHV), dtype=float)
#phi = zeros((1, NBPHI_fortran), dtype=float)

# Il a été remarqué que pour un indice de theta donné, la boite n'est pas la même en Fortran ou en Cuda. C'est pourquoi
# pour comparer les mêmes boites, il faut prendre l'indice i en Cuda et l'indice i-1 en Fortran. La ième boite Cuda correspond à la
# ième-1 boite en Fortran

if choix=='i':
	for iphi in xrange(0,NBPHI_fortran):
		for ith in xrange(NTHV):
			data_fortran[iphi][ith] = tab_fortran['real_refl'][0, ith-1, iphi, 0]
			
elif choix=='q':
	for iphi in xrange(0,NBPHI_fortran):
		for i in xrange(NTHV):
			data_fortran[iphi][i] = tab_fortran['real_refl'][1, i-1, iphi, 0]
			
elif choix=='u':
	for iphi in xrange(0,NBPHI_fortran):
		for i in xrange(NTHV):
			data_fortran[iphi][i] = tab_fortran['real_refl'][2, i-1, iphi, 0]

elif choix=='l':
	for iphi in xrange(0,NBPHI_fortran):
		for i in xrange(NTHV):
			data_fortran[iphi][i]=math.sqrt(pow(tab_fortran['real_refl'][1,i-1,iphi,0],2)+
										pow(tab_fortran['real_refl'][2,i-1,iphi,0],2))

#for i in xrange(NTHV):
theta = tab_fortran['real_thv_bornes']*180/3.1415 - 0.25
phi = tab_fortran['real_phi_bornes']*180/3.1415+0.5


# Infos en commentaire sur le graph
commentaire = type_simu + ' - ' + angle

##########################################################
##				CREATION DES GRAPHIQUES					##
##########################################################

for iphi in xrange(0,NBPHI_fortran,pas_figure):
		
	# initialisation
	listePlots = []
	listeLegends = []
	figure()
	# fortran
	#listePlots.append(plot(tab_fortran['real_thv_bornes'][dep-1:fin-1], tab_fortran['real_refl'][0, dep-1:fin-1, iphi,0]))
	listePlots.append( plot(theta[dep:fin], data_fortran[iphi][dep:fin]) )
	listeLegends.append('Fortran')
	#sos
	listePlots.append( plot(theta[dep:fin],data_sos[iphi][dep:fin]) )
	listeLegends.append('SOS')
	
	# commun
	legend(listePlots, listeLegends, loc='best', numpoints=1)
	title( type_donnees + ' pour SOS et Fortran pour phi='+str(phi[iphi])+' deg' )
	xlabel( 'Theta (deg)' )
	ylabel( type_donnees, rotation='horizontal' )
	figtext(0.25, 0.7, commentaire+" deg", fontdict=None)
	figtext(0, 0, "Date: "+date_simu+"\nFichier SOS: "+nom_sos+"\nFichier fortran: "+nom_fortran, fontdict=None, size='xx-small')
	grid(True)
	savefig( path_dossier_sortie+'/c_'+type_donnees+'_Fortran_SOS_phi='+str(phi[iphi])+'.png', dpi=(140) )
	
	##########################################
	##				RAPPORT					##
	##########################################
	
	figure()
	listePlots = []
	listeLegends = []
	listePlots.append( plot( theta[dep:fin], data_fortran[iphi][dep:fin]/data_sos[iphi][dep:fin] ) )
	listeLegends.append('Rapport de '+type_donnees+' Fortran/SOS')
	
	#Régression linéaire
	(ar,br) = polyfit( theta[dep:fin], data_fortran[iphi][dep:fin]/data_sos[iphi][dep:fin] ,1 )
	regLin = polyval( [ar,br],theta[dep:fin] )
	
	listePlots.append( plot(theta[dep:fin], regLin) )
	listeLegends.append( 'Regression lineaire y='+str(ar)+'x+'+str(br) )
	legend( listePlots, listeLegends, loc='best', numpoints=1 )
	
	title( 'Rapport des '+type_donnees+' Fortran_SOS pour phi='+str(phi[iphi])+' deg' )
	xlabel( 'Theta (deg)' )
	ylabel( 'Rapport des '+type_donnees )
	figtext(0.4, 0.25, commentaire+" deg", fontdict=None)
	figtext(0, 0, "Date: "+date_simu+"\nFichier SOS: "+nom_sos+"\nFichier fortran: "+nom_fortran, fontdict=None, size='xx-small')
	grid(True)
	savefig( path_dossier_sortie+'/rapport_'+type_donnees+'_Fortran_SOS_phi=' +str(phi[iphi])+'.png', dpi=(140) )
	
	##########################################
	##				DIFFERENCE				##
	##########################################
	figure()
	plot( theta[dep:fin], data_sos[iphi][dep:fin]-data_fortran[iphi][dep:fin] )
	title( 'Difference des '+type_donnees+ ' SOS - Fortran pour phi='+str(phi[iphi])+' deg' )
	xlabel( 'Theta (deg)' )
	ylabel( 'Difference des '+type_donnees )
	figtext(0.4, 0.25, commentaire+" deg", fontdict=None)
	figtext(0, 0, "Date: "+date_simu+"\nFichier SOS: "+nom_sos+"\nFichier fortran: "+nom_fortran, fontdict=None, size='xx-small')
	grid(True)
	savefig( path_dossier_sortie+'/difference_'+type_donnees+'_SOS_Fortran_phi='+str(phi[iphi])+'.png', dpi=(140) )


##########################################################
##				CREATION FICHIER PARAMETRE				##
##########################################################

# creation du fichier contenant les parametres de la simulation
fichierSortie = open(path_dossier_sortie+'/Parametres.txt', 'w')

# Ecriture des données en sortie
fichierSortie.write("\n\n##### Paramètres Fortran #####\n")
fichierSortie.write("Nombres de photons: {0:.2e}\n".format(tab_fortran['nphotons']))
fichierSortie.write("Germe: " + str(tab_fortran['initgerme']) + "\n")
fichierSortie.write("\n\n")


##################################################
##				CALCUL STATISTIQUES				##
##################################################

moyDiff = 0		# Moyenne de SOS-Fortran
sigmaDiff = 0
moyDiffAbs = 0	# Moyenne de |SOS-Fortran|
sigmaDiffAbs = 0
moyRap = 0		# Moyenne de SOS/Fortran
sigmaRap = 0
moyRapAbs = 0	# Moyenne de |SOS/Fortran|
sigmaRapAbs = 0

##-- Calcul des moyennes --##

for iphi in xrange(NBPHI_fortran):
	for ith in xrange(dep,fin):	# Calcul sur tout l'espace
		moyDiff += data_sos[iphi][ith]-data_fortran[iphi][ith]
		moyDiffAbs += abs(data_sos[iphi][ith]-data_fortran[iphi][ith])
		moyRap += data_sos[iphi][ith]/data_fortran[iphi][ith]
		moyRapAbs += abs(1-data_sos[iphi][ith]/data_fortran[iphi][ith] )

moyDiff = moyDiff/((fin-dep)*NBPHI_fortran)
moyDiffAbs = moyDiffAbs/((fin-dep)*NBPHI_fortran)
moyRap = moyRap/((fin-dep)*NBPHI_fortran)
moyRapAbs = moyRapAbs/((fin-dep)*NBPHI_fortran)
	
##-- Calcul des écart type --##

for iphi in xrange(NBPHI_fortran):
	for ith in xrange(dep,fin):	# Calcul sur tout l'espace
		
		# Calcul des écarts type
		sigmaDiff += pow( moyDiff - (data_sos[iphi][ith]-data_fortran[iphi][ith])  ,2.0 )
		sigmaDiffAbs += pow( moyDiffAbs - abs( data_sos[iphi][ith]-data_fortran[iphi][ith] ) ,2.0 )
		sigmaRap += pow( moyRap - ( data_sos[iphi][ith]/data_fortran[iphi][ith] ) ,2.0 )
		sigmaRapAbs += pow( moyRapAbs - abs(1-data_sos[iphi][ith]/data_fortran[iphi][ith]) ,2.0 )

	
sigmaDiff = math.sqrt( sigmaDiff/((fin-dep)*NBPHI_fortran) )
sigmaDiffAbs = math.sqrt( sigmaDiffAbs/((fin-dep)*NBPHI_fortran) )
sigmaRap = math.sqrt( sigmaRap/((fin-dep)*NBPHI_fortran) )
sigmaRapAbs = math.sqrt( sigmaRapAbs/((fin-dep)*NBPHI_fortran) )


print "\n====================:Résultats:===================="
print 'Moyenne de la différence SOS-Fortran = {0:.4e}'.format(moyDiff)
print 'Ecart type de la différence SOS-Fortran = {0:.4e}\n'.format(sigmaDiff)

print 'Moyenne de la valeur absolue de la différence = {0:.4e}'.format(moyDiffAbs)
print 'Ecart type de la valeur absolue de la différence = {0:.4e}\n'.format(sigmaDiffAbs)

print 'Moyenne du rapport SOS/Fortran = {0:.4e}'.format(moyRap)
print 'Ecart type du rapport SOS/Fortran = {0:.4e}\n'.format(sigmaRap)

print 'Pourcentage erreur moyen SOS/Fortran = {0:.4e} %'.format(moyRapAbs*100)
print 'Ecart type de erreur = {0:.4e}\n'.format(sigmaRapAbs)
print "==================================================="

fichierSortie.write( "\n====================:Résultats:====================\n")
fichierSortie.write( 'Moyenne de la différence SOS-Fortran = {0:.4e}\n'.format(moyDiff))
fichierSortie.write( 'Ecart type de la différence SOS-Fortran = {0:.4e}\n\n'.format(sigmaDiff))

fichierSortie.write( 'Moyenne de la valeur absolue de la différence = {0:.4e}\n'.format(moyDiffAbs))
fichierSortie.write( 'Ecart type de la valeur absolue de la différence = {0:.4e}\n\n'.format(sigmaDiffAbs))

fichierSortie.write( 'Moyenne du rapport SOS/Fortran = {0:.4e}\n'.format(moyRap))
fichierSortie.write( 'Ecart type du rapport SOS/Fortran = {0:.4e}\n\n'.format(sigmaRap))

fichierSortie.write( 'Pourcentage erreur moyen SOS/Fortran = {0:.4e} %\n'.format(moyRapAbs*100))
fichierSortie.write( 'Ecart type de erreur = {0:.4e}\n'.format(sigmaRapAbs))
fichierSortie.write( "===================================================\n")


##-- Calculs par couronne de theta --##
# Dth de 10 deg, il y aura donc NBTHETA/180*10 échantillons par couronnes car theta=[0;180°]
pas = NBTHETA/180*10
ith0 = 1

for icouronne in xrange( NBTHETA/(NBTHETA/180*10) ):	# Pour chaque couronne
	
	if icouronne == 17 or icouronne==0:	#on modifie les paramètres pour éviter le débodement de tableau sur la dernière couronne
		pas=NBTHETA/180*10 - 1
	else:
		pas = NBTHETA/180*10
		
	moyDiff = 0		# Moyenne de SOS-Fortran
	sigmaDiff = 0
	moyDiffAbs = 0	# Moyenne de |SOS-Fortran|
	sigmaDiffAbs = 0
	moyRap = 0		# Moyenne de SOS/Fortran
	sigmaRap = 0
	moyRapAbs = 0	# Moyenne de |SOS/Fortran|
	sigmaRapAbs = 0
	
	##-- Calcul des moyennes --##
	for ith in xrange(ith0,ith0+pas):
		for iphi in xrange(NBPHI_fortran):
			
			moyDiff += data_sos[iphi][ith]-data_fortran[iphi][ith]
			moyDiffAbs += abs(data_sos[iphi][ith]-data_fortran[iphi,ith] )
			moyRap += data_sos[iphi][ith]/data_fortran[iphi,ith]
			moyRapAbs += abs(1-data_sos[iphi][ith]/data_fortran[iphi,ith] )
			
	moyDiff = moyDiff/(pas*NBPHI_fortran)
	moyDiffAbs = moyDiffAbs/(pas*NBPHI_fortran)
	moyRap = moyRap/(pas*NBPHI_fortran)
	moyRapAbs = moyRapAbs/(pas*NBPHI_fortran)

	##-- Calcul des écart type --##
	for ith in xrange(ith0,ith0+pas):
		for iphi in xrange(NBPHI_fortran):
			
			sigmaDiff += pow( moyDiff - (data_sos[iphi][ith]-data_fortran[iphi,ith] ) ,2.0 )
			sigmaDiffAbs += pow( moyDiffAbs - abs( data_sos[iphi][ith]-data_fortran[iphi,ith] ),2.0 )
			sigmaRap += pow( moyRap - ( data_sos[iphi][ith]/data_fortran[iphi,ith]) ,2.0 )
			sigmaRapAbs += pow( moyRapAbs - abs(1-data_sos[iphi][ith]/data_fortran[iphi,ith] ),2.0 )

	sigmaDiff = math.sqrt( sigmaDiff/(pas*NBPHI_fortran) )
	sigmaDiffAbs = math.sqrt( sigmaDiffAbs/(pas*NBPHI_fortran) )
	sigmaRap = math.sqrt( sigmaRap/(pas*NBPHI_fortran) )
	sigmaRapAbs = math.sqrt( sigmaRapAbs/(pas*NBPHI_fortran) )
			

	print "\n====================:Résultats par couronne:===================="
	print '==================:Couronne #{0:2d} -{1:3d}->{2:3d}deg==================='.format(ith0/10,ith0*90/NBTHETA,
																								(ith0+pas)*90/NBTHETA)
	print 'Moyenne de la différence SOS-Fortran = {0:.4e}'.format(moyDiff)
	print 'Ecart type de la différence SOS-Fortran = {0:.4e}\n'.format(sigmaDiff)

	print 'Moyenne de la valeur absolue de la différence = {0:.4e}'.format(moyDiffAbs)
	print 'Ecart type de la valeur absolue de la différence = {0:.4e}\n'.format(sigmaDiffAbs)

	print 'Moyenne du rapport SOS/Fortran = {0:.4e}'.format(moyRap)
	print 'Ecart type du rapport SOS/Fortran = {0:.4e}\n'.format(sigmaRap)

	print 'Pourcentage erreur SOS/Fortran = {0:.4e} %'.format(moyRapAbs*100)
	print 'Ecart type de erreur = {0:.4e}\n'.format(sigmaRapAbs)
	print "================================================================"

	fichierSortie.write( "\n====================:Résultats par couronne:====================\n")
	fichierSortie.write( '==================:Couronne #{0:2d}-{1:3d}->{2:3d}deg===================\n'.format(ith0/10,
																						ith0*90/NBTHETA, (ith0+pas)*90/NBTHETA))
	fichierSortie.write( 'Moyenne de la différence SOS-Fortran = {0:.4e}\n'.format(moyDiff))
	fichierSortie.write( 'Ecart type de la différence SOS-Fortran = {0:.4e}\n\n'.format(sigmaDiff))

	fichierSortie.write( 'Moyenne de la valeur absolue de la différence = {0:.4e}\n'.format(moyDiffAbs))
	fichierSortie.write( 'Ecart type de la valeur absolue de la différence = {0:.4e}\n\n'.format(sigmaDiffAbs))

	fichierSortie.write( 'Moyenne du rapport SOS/Fortran = {0:.4e}\n'.format(moyRap))
	fichierSortie.write( 'Ecart type du rapport SOS/Fortran = {0:.4e}\n\n'.format(sigmaRap))

	fichierSortie.write( 'Pourcentage erreur SOS/Fortran = {0:.4e} %\n'.format(moyRapAbs*100))
	fichierSortie.write( 'Ecart type de erreur SOS/Fortran = {0:.4e}\n'.format(sigmaRapAbs))
	fichierSortie.write( "================================================================\n")

	ith0 += pas
	
fichierSortie.close()

print '################################################'
print 'Simulation de {0} terminee pour Fortran-SOS'.format(type_donnees)
print '################################################'
