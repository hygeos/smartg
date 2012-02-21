#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import warnings
warnings.simplefilter("ignore",DeprecationWarning)
import pyhdf.SD
from pylab import *
import gzip


	          ##############
	         # PARAMETRES #
	        ##############

# Résultats Fortran
resultats = "out_atmos_seule.ran=0050.wav=443.ths=30.000.tr=0.0000.ta=0.1000.difff=0001.pi0=0.967.H=002.000.mod=valid_T70.443"
path_fortran_zip = "/home/florent/MC/bin/res_corrects/" + resultats + ".bin.gz"

# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = "../out_scripts/quantification_bruit_fortran/quantification_fortran_" + resultats

path_ref = "/home/florent/entree/new_out_SOS_toaer_0.1_ths_30_T70_443_MU400.txt"

os.system("rm -rf "+ path_dossier_sortie)
os.system("mkdir -p "+ path_dossier_sortie)


	##############################
	# Récupération des données	#
	############################	

#Récupération des données Fortran
(NSTK,NBTHETA,NBPHI,NTYP) = (4, 180, 180, 8)
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
('real_refl', float32, (NSTK,NBTHETA,NBPHI,NTYP)),
('real_znad', float32, (8*NTYP,)),
('real_upun', float32),
('real_upab', float32),
('real_dnun', float32),
('real_dnab', float32),
('real_dnabdirect', float32),
('real_dnabplus', float32),
('biais', int64),
('duree', float32, (3,)),
('real_thv_bornes', float32, (NBTHETA,)),
('pi', float32),
('real_phi_bornes', float32, (NBPHI+1,)),
])

# lecture du fichier fortran (bin)
file_fortran_bin = gzip.open(path_fortran_zip)
file_fortran_bin.read(4)	#read(4) pour les nouveaux fichiers, read(8) pour les anciens
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
		
		
# Récupération des données de références

if os.path.exists(path_ref):
	
	# data_ref[iphi][ith] = reflectance
	# ith est le num de la ith-ème boite theta. Boites de pas de 0.5 centrées tous les 0.5
	# iphi est le num de la ith-ème boite phi
	data_ref = zeros((NBPHI,2*(NBTHETA-1)),dtype=float)
	fic_ref = open(path_ref, "r")
	
	for ligne in fic_ref:
		donnees = ligne.rstrip('\n\r').split("\t")
		# Lecture phi	th	reflectance
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
sys.stdout.write("| Le fichier traité est " + path_fortran_zip + "\t|\n")
sys.stdout.write("| Le fichier de référence est " + path_ref + "\t|\n")
sys.stdout.write("| Les résultats sont stockés dans " + path_dossier_sortie + "\t|\n")
sys.stdout.write(" ----------------------------------------------------------------\n")
	
			 #############################
			# Création de graphiques	#
			############################
dep = 2
fin = NBTHETA-1


for i in xrange(NBTHETA):
	#print "theta:{0} pour i={1}".format(tab_fortran['real_thv_bornes'][i],i)
	tab_fortran['real_thv_bornes'][i] = tab_fortran['real_thv_bornes'][i]*180/3.1415 - 0.25
	tab_fortran['real_phi_bornes'][i] = tab_fortran['real_phi_bornes'][i]*180/3.1415
	#print "theta:{0}".format(tab_fortran['real_thv_bornes'][i])
	
	
for iphi in xrange(0,NBPHI,5):
	# initialisation
	listePlots = []
	listeLegends = []

	# Référence
	listePlots.append(plot(tab_fortran['real_thv_bornes'][dep-1:fin-1], data_ref[iphi][dep:fin]))
	listeLegends.append('Code SOS')

	# Fortran
	listePlots.append(plot(tab_fortran['real_thv_bornes'][dep-1:fin-1], tab_fortran['real_refl'][0, dep-1:fin-1, iphi,
0]))
	listeLegends.append('Fortran')

	# commun
	legend(listePlots, listeLegends, loc='best', numpoints=1)
	title('Comparaison references et Fortran pour phi='+str(tab_fortran['real_phi_bornes'][iphi])+" deg")
	xlabel('Theta (deg)')
	ylabel('Reflectance')
	grid(True)
	savefig(path_dossier_sortie+"/comparaison_phi="+str(tab_fortran['real_phi_bornes'][iphi])+".png", dpi=(140))
	figure()

		##########################################
		#	Figures d'analyse plus spécifiques	#
		########################################
	
	## Figure d'évaluation du taux d'erreur - RAPPORT
	listePlots = []
	listeLegends = []
	listePlots.append( plot(tab_fortran['real_thv_bornes'][dep-1:fin-1], (tab_fortran['real_refl'][0,
dep-1:fin-1,iphi,0])/
	data_ref[iphi][dep:fin] ) )
	listeLegends.append('Rapport Fortran/SOS')
	
	#Régression linéaire
	(ar,br)=polyfit(tab_fortran['real_thv_bornes'][dep-1:fin-1],(tab_fortran['real_refl'][0, dep-1:fin-1, iphi,0])/
	data_ref[iphi][dep:fin] ,1)
	regLin=polyval([ar,br],tab_fortran['real_thv_bornes'][dep-1:fin-1])
	
	listePlots.append( plot(tab_fortran['real_thv_bornes'][dep-1:fin-1], regLin) )
	listeLegends.append('Regression lineaire y='+str(ar)+'x+'+str(br))
	legend(listePlots, listeLegends, loc='best', numpoints=1)
	
	title("Rapport des resultats Fortran et SOS pour phi="+str(tab_fortran['real_phi_bornes'][iphi])+" deg")
	xlabel("Theta (deg)")
	ylabel("Rapport des reflactances")
	grid(True)
	savefig(path_dossier_sortie+"/rapport_reflectance_phi="+str(tab_fortran['real_phi_bornes'][iphi])+".png", dpi=(140))
	figure()
	
	# Figure d'évaluation du taux d'erreur - DIFFERENCE
	listePlots = []
	listeLegends = []
	plot(tab_fortran['real_thv_bornes'][dep-1:fin-1],-tab_fortran['real_refl'][0, dep-1:fin-1, iphi,
0]+(data_ref[iphi,dep:fin]) )
	title("Difference des resultats SOS - Fortran pour phi="+str(tab_fortran['real_phi_bornes'][iphi])+" deg")
	xlabel("Theta (deg)")
	ylabel("Difference des reflactances")
	grid(True)
	savefig(path_dossier_sortie+"/difference_reflectance_phi="+str(tab_fortran['real_phi_bornes'][iphi])+".png",
dpi=(140))
	figure()



# creation du fichier contenant les parametres de la simulation
sortie = open(path_dossier_sortie+"/Quantification_bruit_"+resultats+".txt", "w")
#sortie.write("NBPHOTONS = " + str(NBPHOTONS) + "\n")
#sortie.write("NBLOOP = " + str(NBLOOP) + "\n")
#sortie.write("SEED = " + str(SEED) + "\n")	
#sortie.write("XBLOCK = " + str(XBLOCK) + "\n")
#sortie.write("YBLOCK = " + str(YBLOCK) + "\n")
#sortie.write("XGRID = " + str(XGRID) + "\n")
#sortie.write("YGRID = " + str(YGRID) + "\n")
#sortie.write("NBTHETA = " + str(NBTHETA) + "\n")
#sortie.write("NBPHI = " + str(NBPHI) + "\n")
#sortie.write("THSDEG = " + str(THSDEG) + "\n")
#sortie.write("LAMBDA = " + str(LAMBDA) + "\n")
#sortie.write("TAURAY = " + str(TAURAY) + "\n")
#sortie.write("TAUAER = " + str(TAUAER) + "\n")*180/3.1415
#sortie.write("W0AER = " + str(W0AER) + "\n")
#sortie.write("PROFIL = " + str(PROFIL) + "\n")
#sortie.write("HA = " + str(HA) + "\n")
#sortie.write("HR = " + str(HR) + "\n")
#sortie.write("ZMIN = " + str(ZMIN) + "\n")
#sortie.write("ZMAX = " + str(ZMAX) + "\n")
#sortie.write("WINDSPEED = " + str(WINDSPEED) + "\n")
#sortie.write("NH2O = " + str(NH2O) + "\n")
#sortie.write("SIM = " + str(SIM) + "\n")
#sortie.write("SUR = " + str(NBPHOTONS) + "\n")
#sortie.write("DIOPTRE = " + str(DIOPTRE) + "\n")
#sortie.write("CONPHY = " + str(CONPHY) + "\n")
#sortie.write("DIFFF = " + str(DIFFF) + "\n")
#sortie.write("PATHRESULTATSHDF = " + str(PATHRESULTATSHDF) + "\n")
#sortie.write("PATHTEMOINHDF = " + str(PATHTEMOINHDF) + "\n")
#sortie.write("PATHDIFFAER = " + str(PATHDIFFAER) + "\n")
#sortie.write("PATHPROFILATM = " + str(PATHPROFILATM) + "\n")

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
for iphi in xrange(NBPHI):
	for ith in xrange(dep,fin):	# Calcul sur tout l'espace
		moyDiff += data_ref[iphi][ith]-tab_fortran['real_refl'][0, ith-1, iphi,0]
		moyDiffAbs += abs(data_ref[iphi][ith]-tab_fortran['real_refl'][0, ith-1, iphi,0] )
		moyRap += data_ref[iphi][ith]/tab_fortran['real_refl'][0, ith-1, iphi,0]
		moyRapAbs += abs(1-data_ref[iphi][ith]/tab_fortran['real_refl'][0, ith-1, iphi,0] )

moyDiff = moyDiff/((fin-dep)*NBPHI)
moyDiffAbs = moyDiffAbs/((fin-dep)*NBPHI)
moyRap = moyRap/((fin-dep)*NBPHI)
moyRapAbs = moyRapAbs/((fin-dep)*NBPHI)
	
	
for iphi in xrange(NBPHI):
	for ith in xrange(dep,fin):	# Calcul sur tout l'espace
		
		# Calcul des écarts type
		sigmaDiff += pow( moyDiff - (data_ref[iphi][ith]-tab_fortran['real_refl'][0, ith-1, iphi,0]) ,2.0 )
		sigmaDiffAbs += pow( moyDiffAbs - abs( data_ref[iphi][ith]-tab_fortran['real_refl'][0, ith-1, iphi,0]) ,2.0 )
		sigmaRap += pow( moyRap - ( data_ref[iphi][ith]/tab_fortran['real_refl'][0, ith-1, iphi,0] ) ,2.0 )
		sigmaRapAbs += pow( moyRapAbs - abs(1-data_ref[iphi][ith]/tab_fortran['real_refl'][0, ith-1, iphi,0] ) ,2.0 )

	
sigmaDiff = math.sqrt( sigmaDiff/((fin-dep)*NBPHI) )
sigmaDiffAbs = math.sqrt( sigmaDiffAbs/((fin-dep)*NBPHI) )
sigmaRap = math.sqrt( sigmaRap/((fin-dep)*NBPHI) )
sigmaRapAbs = math.sqrt( sigmaRapAbs/((fin-dep)*NBPHI) )


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
	
for icouronne in xrange(NBTHETA/pas):	# Pour chaque couronne
	
	if icouronne == 17 or icouronne == 0:	#on modifie les paramètres pour éviter le débodement de tableau sur la dernière couronne
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
		for iphi in xrange(NBPHI):
			
			moyDiff += data_ref[iphi][ith]-tab_fortran['real_refl'][0, ith-1, iphi,0]
			moyDiffAbs += abs(data_ref[iphi][ith]-tab_fortran['real_refl'][0, ith-1, iphi,0] )
			moyRap += data_ref[iphi][ith]/tab_fortran['real_refl'][0, ith-1, iphi,0]
			moyRapAbs += abs(1-data_ref[iphi][ith]/tab_fortran['real_refl'][0, ith-1, iphi,0] )
			
	moyDiff = moyDiff/(pas*NBPHI)
	moyDiffAbs = moyDiffAbs/(pas*NBPHI)
	moyRap = moyRap/(pas*NBPHI)
	moyRapAbs = moyRapAbs/(pas*NBPHI)

	# Calcul des écarts type
	for ith in xrange(ith0,ith0+pas):
		for iphi in xrange(NBPHI):
			
			sigmaDiff += pow( moyDiff - tab_fortran['real_refl'][0, ith-1, iphi,0] ,2.0 )
			sigmaDiffAbs += pow( moyDiffAbs - abs( data_ref[iphi][ith]-tab_fortran['real_refl'][0, ith-1, iphi,0] ) ,2.0 )
			sigmaRap += pow( moyRap - ( data_ref[iphi][ith]/tab_fortran['real_refl'][0, ith-1, iphi,0] ) ,2.0 )
			sigmaRapAbs += pow( moyRapAbs - abs(1-data_ref[iphi][ith]/tab_fortran['real_refl'][0, ith-1, iphi,0] ) ,2.0 )

	sigmaDiff = math.sqrt( sigmaDiff/(pas*NBPHI) )
	sigmaDiffAbs = math.sqrt( sigmaDiffAbs/(pas*NBPHI) )
	sigmaRap = math.sqrt( sigmaRap/(pas*NBPHI) )
	sigmaRapAbs = math.sqrt( sigmaRapAbs/(pas*NBPHI) )
			

	print "\n====================:Résultats par couronne:===================="
	print '==================:Couronne #{0:2d} -{1:3d}->{2:3d} deg==================='.format(ith0/10,ith0*180/NBTHETA, (ith0+pas)*180/NBTHETA)
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
	sortie.write( '==================:Couronne #{0:2d} -{1:3d}->{2:3d} deg===================\n'.format(ith0/10,ith0*180/NBTHETA, (ith0+pas)*180/NBTHETA))
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
