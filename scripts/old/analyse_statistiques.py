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


	          ##############
	         # PARAMETRES #
	        ##############
# Résultats Fortran
path_fortran_zip = \
"/home/florent/MC/bin/out_atmos_dioptre.ran=0173.wav=443.ths=70.000.tr=0.0010.ta=0.0000.difff=0000.pi0=0.967.H=002.000.vent=05.000.\
bin.gz"

# Nom du fichier hdf à analyser SANS l'extension hdf
nom_hdf = "new_fortran_atmos_dioptre_agite_tauRay=0.001000_tauAer=0.000000_ths=70.000000_ws=5.000000"
# Chemin complet du hdf cuda
path_cuda = "../out_prog/Resultats_" + nom_hdf + ".hdf"

# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = "../out_scripts/statistiques/"
#os.system("rm -rf "+path_dossier_sortie)
os.system("mkdir -p "+ path_dossier_sortie)



	          #####################
	         # RESULTATS FORTRAN #
	        #####################


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
file_fortran_bin = gzip.open(path_fortran_zip)
file_fortran_bin.read(4)
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


	          ##################
	         # RESULTATS CUDA #
	        ##################

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
	
	name = "Valeur de la reflectance pour un phi et theta donnes"
	sds_cuda = sd_cuda.select(name)
	data = sds_cuda.get()	
	
else:
	sys.stdout.write("Pas de fichier "+path_cuda+"\n")
	sys.exit()

sys.stdout.write("\n -------------------------------------------------------------------------------\n")
sys.stdout.write("| Le fichier cuda est " + path_cuda + "\t|\n")
sys.stdout.write("| Le fichier fortran est " + path_fortran_zip + "\t|\n")
sys.stdout.write("| Les résultats sont stockés dans " + path_dossier_sortie + "\t|\n")
sys.stdout.write(" -------------------------------------------------------------------------------\n")

#######################################################

#for i in xrange(180):
	#print 'theta[{0}] = {1:10.8}'.format(i,theta[i])
	
#for i in xrange(NBPHI_cuda):
	#print 'phi[{0}] = {1:10.8}'.format(i,phi[i])

#######################################################

	          ###############################
	         # CREATION FICHIER DE SORTIE  #
	        ###############################

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
sortie = open(path_dossier_sortie+"Statistiques_"+nom_hdf+".txt", "w")
sortie.write('NBPHOTONS = {0:.2e}\n'.format(NBPHOTONS))
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

sortie.write("\n\n##### Paramètres Fortran #####\n")
sortie.write("Nombres de photons: " + str(tab_fortran['nphotons']) + "\n")
sortie.write("Germe: " + str(tab_fortran['initgerme']) + "\n")
sortie.write("\n\n")


	          #######################
	         # CREATION GRAPHIQUES #
	        #######################

moyDiff = 0		# Moyenne de Fortran-Cuda
sigmaDiff = 0
moyDiffAbs = 0	# Moyenne de |Fortran-Cuda|
sigmaDiffAbs = 0
moyRap = 0		# Moyenne de Fortran/Cuda
sigmaRap = 0
moyRapAbs = 0	# Moyenne de |Fortran/Cuda|
sigmaRapAbs = 0

dep = 4	# Indice de départ pour le calcul
fin = NBTHETA_cuda-2	# Indice de fin pour le calcul

# NOTE: Les indices ci-dessus ont été mis en place pour 2 raisons.
#	- Ils permettent de rogner la simulation si nécessaire. Par exemple, il est inutile de tracer le dernier angle (theta=90°) car il
# 		ne représente rien physiquement
#	- Il a également été remarqué que pour un indice de theta donné, la boite n'est pas la même en Fortran ou en Cuda. C'est pourquoi
#		pour comparer les mêmes boites, il faut prendre l'indice i en Cuda et l'indice i-1 en Fortran. La ième boite Cuda correspond à la
#		ième-1 boite en Fortran
	
# Pour comparer les 2 resultats il faut que phi parcourt un meme intervalle et qu'il y ait le meme nombre de boites selon phi
# Fortran :  intervalle=[0,PI]   nombre_de_boites=NBPHI_fortran
# Cuda :     intervalle=[0,2PI]  nombre_de_boites=NBPHI_cuda
# On va projeter les resultats du cuda sur [0,PI]


if (NBPHI_cuda/2) == NBPHI_fortran:
	for iphi in xrange(NBPHI_cuda/2):
		for ith in xrange(dep,fin):	# Calcul sur tout l'espace
			
			# Calcul des moyennes
			moyDiff += tab_fortran['real_refl'][0, ith, iphi, 0]-(data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2
			moyDiffAbs += abs(tab_fortran['real_refl'][0, ith, iphi, 0]-(data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2 )
			moyRap += tab_fortran['real_refl'][0, ith, iphi, 0]/((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2)
			moyRapAbs += abs(1-tab_fortran['real_refl'][0, ith, iphi, 0]/((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2) )

	moyDiff = moyDiff/((fin-dep)*NBPHI_cuda/2)
	moyDiffAbs = moyDiffAbs/((fin-dep)*NBPHI_cuda/2)
	moyRap = moyRap/((fin-dep)*NBPHI_cuda/2)
	moyRapAbs = moyRapAbs/((fin-dep)*NBPHI_cuda/2)
	
	for iphi in xrange(NBPHI_cuda/2):
		for ith in xrange(dep,fin):	# Calcul sur tout l'espace
			
			# Calcul des écarts type
			sigmaDiff += pow( moyDiff - (tab_fortran['real_refl'][0, ith, iphi, 0]-(data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2
) ,2.0 )
			sigmaDiffAbs += pow( moyDiffAbs - abs( tab_fortran['real_refl'][0, ith, iphi,
0]-((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2) ) ,2.0 )
			sigmaRap += pow( moyRap - ( tab_fortran['real_refl'][0, ith, iphi,
0]/((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2) ) ,2.0 )
			sigmaRapAbs += pow( moyRapAbs - abs(1-tab_fortran['real_refl'][0, ith, iphi,
0]/((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2) ) ,2.0 )

	
	sigmaDiff = math.sqrt( sigmaDiff/((fin-dep)*NBPHI_cuda/2) )
	sigmaDiffAbs = math.sqrt( sigmaDiffAbs/((fin-dep)*NBPHI_cuda/2) )
	sigmaRap = math.sqrt( sigmaRap/((fin-dep)*NBPHI_cuda/2) )
	sigmaRapAbs = math.sqrt( sigmaRapAbs/((fin-dep)*NBPHI_cuda/2) )
	
	
	print "\n====================:Résultats:===================="
	print 'Moyenne de la différence Fortran-Cuda = {0:.4e}'.format(moyDiff)
	print 'Ecart type de la différence Fortran-Cuda = {0:.4e}\n'.format(sigmaDiff)
	
	print 'Moyenne de la valeur absolue de la différence = {0:.4e}'.format(moyDiffAbs)
	print 'Ecart type de la valeur absolue de la différence = {0:.4e}\n'.format(sigmaDiffAbs)
	
	print 'Moyenne du rapport Fortran/Cuda = {0:.4e}'.format(moyRap)
	print 'Ecart type du rapport Fortran/Cuda = {0:.4e}\n'.format(sigmaRap)
	
	print 'Pourcentage erreur moyen Fortran/Cuda = {0:.4e} %'.format(moyRapAbs*100)
	print 'Ecart type du rapport absolue Fortran/Cuda = {0:.4e}\n'.format(sigmaRapAbs)
	print "==================================================="

	sortie.write( "\n====================:Résultats:====================\n")
	sortie.write( 'Moyenne de la différence Fortran-Cuda = {0:.4e}\n'.format(moyDiff))
	sortie.write( 'Ecart type de la différence Fortran-Cuda = {0:.4e}\n\n'.format(sigmaDiff))
	
	sortie.write( 'Moyenne de la valeur absolue de la différence = {0:.4e}\n'.format(moyDiffAbs))
	sortie.write( 'Ecart type de la valeur absolue de la différence = {0:.4e}\n\n'.format(sigmaDiffAbs))
	
	sortie.write( 'Moyenne du rapport Fortran/Cuda = {0:.4e}\n'.format(moyRap))
	sortie.write( 'Ecart type du rapport Fortran/Cuda = {0:.4e}\n\n'.format(sigmaRap))
	
	sortie.write( 'Pourcentage erreur moyen Fortran/Cuda = {0:.4e} %\n'.format(moyRapAbs*100))
	sortie.write( 'Ecart type du rapport absolue Fortran/Cuda = {0:.4e}\n'.format(sigmaRapAbs))
	sortie.write( "===================================================\n")
	
	##################################
	# calculs par couronne de theta #
	################################
	
	# Dth de 10 deg, il y aura donc NBTHETA_cuda/180*10 échantillons par couronnes car theta=[0;180°]
	pas = NBTHETA_cuda/180*10
	ith0 = 0
	
	for icouronne in xrange(NBTHETA_cuda/pas):	# Pour chaque couronne
		
		if icouronne == NBTHETA_cuda/pas-1:	#on modifie les paramètres pour éviter le débodement de tableau sur la dernière couronne
			pas=pas-1
			
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
				
				moyDiff += tab_fortran['real_refl'][0, ith, iphi, 0]-(data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2
				moyDiffAbs += abs(tab_fortran['real_refl'][0, ith, iphi, 0]-(data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2 )
				moyRap += tab_fortran['real_refl'][0, ith, iphi, 0]/((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2)
				moyRapAbs += abs(1-tab_fortran['real_refl'][0, ith, iphi, 0]/((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2) )
				
		moyDiff = moyDiff/(pas*NBPHI_cuda/2)
		moyDiffAbs = moyDiffAbs/(pas*NBPHI_cuda/2)
		moyRap = moyRap/(pas*NBPHI_cuda/2)
		moyRapAbs = moyRapAbs/(pas*NBPHI_cuda/2)
		
		# Calcul des écarts type
		for ith in xrange(ith0,ith0+pas):
			for iphi in xrange(NBPHI_cuda/2):
				
				sigmaDiff += pow( moyDiff - (tab_fortran['real_refl'][0, ith, iphi,
0]-(data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2 ) ,2.0 )
				sigmaDiffAbs += pow( moyDiffAbs - abs( tab_fortran['real_refl'][0, ith, iphi,
0]-((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2) ) ,2.0 )
				sigmaRap += pow( moyRap - ( tab_fortran['real_refl'][0, ith, iphi,
0]/((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2) ) ,2.0 )
				sigmaRapAbs += pow( moyRapAbs - abs(1-tab_fortran['real_refl'][0, ith, iphi,
0]/((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2) ) ,2.0 )

		sigmaDiff = math.sqrt( sigmaDiff/(pas*NBPHI_cuda/2) )
		sigmaDiffAbs = math.sqrt( sigmaDiffAbs/(pas*NBPHI_cuda/2) )
		sigmaRap = math.sqrt( sigmaRap/(pas*NBPHI_cuda/2) )
		sigmaRapAbs = math.sqrt( sigmaRapAbs/(pas*NBPHI_cuda/2) )
			
		print "\n====================:Résultats par couronne:===================="
		print '================:Couronne theta #{0:2d} -{1:3d}->{2:3d} deg================='.format(ith0/10,ith0*90/NBTHETA_cuda,
(ith0+pas)*90/NBTHETA_cuda)
		print 'Moyenne de la différence Fortran-Cuda = {0:.4e}'.format(moyDiff)
		print 'Ecart type de la différence Fortran-Cuda = {0:.4e}\n'.format(sigmaDiff)
		
		print 'Moyenne de la valeur absolue de la différence = {0:.4e}'.format(moyDiffAbs)
		print 'Ecart type de la valeur absolue de la différence = {0:.4e}\n'.format(sigmaDiffAbs)
		
		print 'Moyenne du rapport Fortran/Cuda = {0:.4e}'.format(moyRap)
		print 'Ecart type du rapport Fortran/Cuda = {0:.4e}\n'.format(sigmaRap)
		
		print 'Pourcentage erreur moyen Fortran/Cuda = {0:.4e} %'.format(moyRapAbs*100)
		print 'Ecart type du rapport absolue Fortran/Cuda = {0:.4e}\n'.format(sigmaRapAbs)
		print "================================================================"
		
		sortie.write( "\n====================:Résultats par couronne:====================\n")
		sortie.write( '================:Couronne theta #{0:2d}\
-{1:3d}->{2:3d} deg=================\n'.format(ith0/10,ith0*90/NBTHETA_cuda, (ith0+pas)*90/NBTHETA_cuda))
		sortie.write( 'Moyenne de la différence Fortran-Cuda = {0:.4e}\n'.format(moyDiff))
		sortie.write( 'Ecart type de la différence Fortran-Cuda = {0:.4e}\n\n'.format(sigmaDiff))
		
		sortie.write( 'Moyenne de la valeur absolue de la différence = {0:.4e}\n'.format(moyDiffAbs))
		sortie.write( 'Ecart type de la valeur absolue de la différence = {0:.4e}\n\n'.format(sigmaDiffAbs))
		
		sortie.write( 'Moyenne du rapport Fortran/Cuda = {0:.4e}\n'.format(moyRap))
		sortie.write( 'Ecart type du rapport Fortran/Cuda = {0:.4e}\n\n'.format(sigmaRap))
		
		sortie.write( 'Pourcentage erreur moyen Fortran/Cuda = {0:.4e} %\n'.format(moyRapAbs*100))
		sortie.write( 'Ecart type du rapport absolue Fortran/Cuda = {0:.4e}\n'.format(sigmaRapAbs))
		sortie.write( "================================================================\n")
		
		ith0 += pas
		
	################################
	# calculs par couronne de Phi #
	##############################
		
	# Dphi de 10 deg, il y aura donc NBPHI_cuda/2*180*10 échantillons par couronnes car phi=[0;180°] en sortie
	pas = NBPHI_cuda/360*10
	iphi0 = 0
		
	for icouronne in xrange(NBPHI_cuda/2/pas):	# Pour chaque couronne
		iphi0 = icouronne*pas
		
		if icouronne == NBPHI_cuda/2/pas-1:	#on modifie les paramètres pour éviter le débodement de tableau sur la dernière couronne
			pas=pas-1
		
		moyDiff = 0		# Moyenne de Fortran-Cuda
		sigmaDiff = 0
		moyDiffAbs = 0	# Moyenne de |Fortran-Cuda|
		sigmaDiffAbs = 0
		moyRap = 0		# Moyenne de Fortran/Cuda
		sigmaRap = 0
		moyRapAbs = 0	# Moyenne de |Fortran/Cuda|
		sigmaRapAbs = 0
		
		# Calcul des moyennes
		for ith in xrange(NBTHETA_cuda-1):
			for iphi in xrange(iphi0,iphi0+pas):
				
				moyDiff += tab_fortran['real_refl'][0, ith, iphi, 0]-(data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2
				moyDiffAbs += abs(tab_fortran['real_refl'][0, ith, iphi, 0]-(data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2 )
				moyRap += tab_fortran['real_refl'][0, ith, iphi, 0]/((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2)
				moyRapAbs += abs(1-tab_fortran['real_refl'][0, ith, iphi, 0]/((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2) )
				
		moyDiff = moyDiff/(pas*NBTHETA_cuda)
		moyDiffAbs = moyDiffAbs/(pas*NBTHETA_cuda)
		moyRap = moyRap/(pas*NBTHETA_cuda)
		moyRapAbs = moyRapAbs/(pas*NBTHETA_cuda)
				
		# Calcul des écarts type
		for ith in xrange(NBTHETA_cuda-1):
			for iphi in xrange(iphi0,iphi0+pas):
				
				sigmaDiff += pow( moyDiff - (tab_fortran['real_refl'][0, ith, iphi,0]
									-(data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2 ) ,2.0 )
				sigmaDiffAbs += pow( moyDiffAbs - abs( tab_fortran['real_refl'][0, ith, iphi,0]
									-((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2) ) ,2.0 )
				sigmaRap += pow( moyRap - ( tab_fortran['real_refl'][0, ith, iphi,0]
									/((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2) ) ,2.0 )
				sigmaRapAbs += pow( moyRapAbs - abs(1-tab_fortran['real_refl'][0, ith, iphi,0]
									/((data[iphi,ith+1]+data[NBPHI_cuda-iphi-1,ith+1])/2) ) ,2.0 )

		sigmaDiff = math.sqrt( sigmaDiff/(pas*NBTHETA_cuda) )
		sigmaDiffAbs = math.sqrt( sigmaDiffAbs/(pas*NBTHETA_cuda) )
		sigmaRap = math.sqrt( sigmaRap/(pas*NBTHETA_cuda) )
		sigmaRapAbs = math.sqrt( sigmaRapAbs/(pas*NBTHETA_cuda) )
				
				
		
		print "\n====================:Résultats par couronne:===================="
		print '==================:Couronne Psi #{0:2d} -{1:3d}->{2:3d} deg================='.format(iphi0/10,iphi0*360/NBPHI_cuda,
(iphi0+pas)*360/NBPHI_cuda)
		print 'Moyenne de la différence Fortran-Cuda = {0:.4e}'.format(moyDiff)
		print 'Ecart type de la différence Fortran-Cuda = {0:.4e}\n'.format(sigmaDiff)
		
		print 'Moyenne de la valeur absolue de la différence = {0:.4e}'.format(moyDiffAbs)
		print 'Ecart type de la valeur absolue de la différence = {0:.4e}\n'.format(sigmaDiffAbs)
		
		print 'Moyenne du rapport Fortran/Cuda = {0:.4e}'.format(moyRap)
		print 'Ecart type du rapport Fortran/Cuda = {0:.4e}\n'.format(sigmaRap)
		
		print 'Pourcentage erreur moyen Fortran/Cuda = {0:.4e} %'.format(moyRapAbs*100)
		print 'Ecart type du rapport absolue Fortran/Cuda = {0:.4e}\n'.format(sigmaRapAbs)
		print "================================================================"
		
		sortie.write( "\n====================:Résultats par couronne:====================\n")
		sortie.write( '==================:Couronne Psi #{0:2d}\
-{1:3d}->{2:3d} deg=================\n'.format(iphi0/10,iphi0*360/NBPHI_cuda, (iphi0+pas)*360/NBPHI_cuda))
		
		sortie.write( 'Moyenne de la différence Fortran-Cuda = {0:.4e}\n'.format(moyDiff))
		sortie.write( 'Ecart type de la différence Fortran-Cuda = {0:.4e}\n\n'.format(sigmaDiff))
		
		sortie.write( 'Moyenne de la valeur absolue de la différence = {0:.4e}\n'.format(moyDiffAbs))
		sortie.write( 'Ecart type de la valeur absolue de la différence = {0:.4e}\n\n'.format(sigmaDiffAbs))
		
		sortie.write( 'Moyenne du rapport Fortran/Cuda = {0:.4e}\n'.format(moyRap))
		sortie.write( 'Ecart type du rapport Fortran/Cuda = {0:.4e}\n\n'.format(sigmaRap))
		
		sortie.write( 'Pourcentage erreur moyen Fortran/Cuda = {0:.4e} %\n'.format(moyRapAbs*100))
		sortie.write( 'Ecart type du rapport absolue Fortran/Cuda = {0:.4e}\n'.format(sigmaRapAbs))
		sortie.write( "================================================================\n")
		
		iphi0 += pas
	
else:
	sys.stdout.write("Les tableaux ne font pas la meme taille\n")
	sys.exit()
	
sortie.close()

