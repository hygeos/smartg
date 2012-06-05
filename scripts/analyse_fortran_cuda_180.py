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

rc('font', family='serif')

##################################################
##				FICHIERS A LIRE					##
##################################################

#
# Paramètres à modifier
#
#-----------------------------------------------------------------------------------------------------------------------
type_simu = "molecules_dioptre_agite"
date_simu = "05062012"
angle = "30"
geometrie = "PARALLELE"		#Géométrie de l'atmosphère

# Nom du fichier Cuda sans extension .hdf
nom_cuda = "out_CUDA_atmos_dioptre_agite_ths=30.00_tRay=0.0533_tAer=0.0000_ws=5.00"

# Nom du fichier Fortran sans l'extension .bin.gz
nom_fortran = "out.ran=8880.wav=443.ths=30.000.tr=0.0533.ta=0.0000.pi0=0.967.H=002.000.vent=05.000_sauv"

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
path_fortran = "/home/florent/MCCuda/validation/"+geometrie+"/"+type_simu+"/"+ nom_fortran+".bin.gz"

# Nom complet du fichier Cuda
path_cuda = "/home/florent/MCCuda/validation/"+geometrie+"/"+type_simu+"/simulation_"+date_simu+"/" + nom_cuda + ".hdf"

# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = \
"/home/florent/MCCuda/validation/"+geometrie+"/"+type_simu+"/graph_"+date_simu+"/"+type_donnees+"/"+type_donnees+"_FORTRAN_CUDA_"+\
nom_cuda


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
sys.stdout.write("# Le fichier cuda est " + path_cuda + "\n")
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
data_cuda = zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)

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
			

# Pour comparer les 2 resultats il faut que phi parcourt un meme intervalle et qu'il y ait le meme nombre de boites selon phi
# Fortran :  intervalle=[0,PI]   nombre_de_boites=NBPHI_fortran
# Cuda :     intervalle=[0,2PI]  nombre_de_boites=NBPHI_cuda
# On va projeter les resultats du cuda sur [0,PI]

#if choix != 'u':
	#for iphi in xrange(0,NBPHI_cuda):
		#for ith in xrange(NBTHETA_cuda):
			#data_cuda[iphi][ith] = (data[iphi,ith]+data[NBPHI_cuda-iphi-1,ith] )/2
			
#else:
#for iphi in xrange(0,NBPHI_cuda):
	#for ith in xrange(NBTHETA_cuda):
data_cuda = data[0:NBPHI_cuda,:]

# Infos en commentaire sur le graph
commentaire = type_simu + ' - ' + angle



##########################################################
##				CREATION DES GRAPHIQUES					##
##########################################################

#---------------------------------------------------------

# Calcul pour l'ergonomie des graphiques
max_data = max(data_cuda[0:NBPHI_cuda-pas_figure+1,dep:fin].max(),data_fortran[0:NBPHI_cuda-pas_figure+1,dep:fin].max())
min_data = min(data_cuda[0:NBPHI_cuda-pas_figure+1,dep:fin].min(),data_fortran[0:NBPHI_cuda-pas_figure+1,dep:fin].min())

max_diff = (data_fortran[0:NBPHI_cuda-pas_figure+1,dep:fin]-data_cuda[0:NBPHI_cuda-pas_figure+1,dep:fin]).max()
#---------------------------------------------------------


# Création de la figure récapitulative
rc('text', usetex=True)

Ymax1 = max(data_cuda[0,:].max(),data_cuda[179,:].max(),data_cuda[89,:].max())
Ymax2 = max(data_fortran[0,2:].max(),data_fortran[179,2:].max(),data_fortran[89,2:].max())
Ymax = max(Ymax1, Ymax2)

Ymin1 = min(data_cuda[0,:].min(),data_cuda[179,:].min(),data_cuda[89,:].min())
Ymin2 = min(data_fortran[0,:].min(),data_fortran[179,:].min(),data_fortran[89,:].min())
Ymin = min(Ymin1, Ymin2)

subplot(1, 3, 1)
subplots_adjust(wspace=0)
theta_inv = theta[::-1]
data_fortran_phi0_inv = data_fortran[0][::-1]
data_cuda_phi0_inv = data_cuda[0][::-1]
plot(theta_inv[:178], data_fortran_phi0_inv[:178],label='Fortran')	# Plot pour phi=0
plot(theta_inv[:178],data_cuda_phi0_inv[:178],label='Cuda')
axis(xmin=90, xmax=0, ymin=1*Ymin, ymax=1*Ymax)
xlabel(r'$\theta_v$ ($\phi=%.1f$)' % (phi[0]))
grid()

subplot(1, 3, 2)
subplots_adjust(wspace=0)
plot(theta[2:], data_fortran[179][2:],label='Fortran')	# Plot pour phi=0
plot(theta[2:],data_cuda[179][2:],label='Cuda')
axis(xmin=0, xmax=90, ymin=1*Ymin, ymax=1*Ymax)
xlabel(r'$\theta_v$ ($\phi=%.1f$)' % (phi[179]))
grid()

subplot(1, 3, 3)
subplots_adjust(wspace=0)
plot(theta[2:], data_fortran[89][2:],label='Fortran')	# Plot pour phi=0
plot(theta[2:],data_cuda[89][2:],label='Cuda')
axis(xmin=0, xmax=90, ymin=1*Ymin, ymax=1*Ymax)
xlabel(r'$\theta_v$ ($\phi=%.1f$)' % (phi[89]))
grid()

legend()

savefig( path_dossier_sortie+'/global_'+type_donnees+'_Fortran_Cuda.png', dpi=(140) )

##########
rc('text', usetex=False)


if (NBPHI_cuda) == NBPHI_fortran:
	for iphi in xrange(0,NBPHI_cuda,pas_figure):
			
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
		title( type_donnees + ' pour Cuda et Fortran pour phi='+str(phi[iphi])+' deg' )
		xlabel( 'Theta (deg)' )
		ylabel( type_donnees, rotation='horizontal' )
		#axis([0,theta[fin],0.99*min_data, 1.01*max_data])
		figtext(0.25, 0.7, commentaire+" deg", fontdict=None)
		figtext(0, 0, "Date: "+date_simu+"\nFichier cuda: "+nom_cuda+"\nFichier fortran: "+nom_fortran, fontdict=None,
				size='xx-small')
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
		
		max_rapport = (abs(data_fortran[iphi,dep:fin]/data_cuda[iphi,dep:fin])).max()
		axis([0,theta[fin],2-1.01*max_rapport, 1.01*max_rapport])
		title( 'Rapport des '+type_donnees+' Fortran_Cuda pour phi='+str(phi[iphi])+' deg' )
		xlabel( 'Theta (deg)' )
		ylabel( 'Rapport des '+type_donnees )
		figtext(0.4, 0.25, commentaire+" deg", fontdict=None)
		figtext(0, 0, "Date: "+date_simu+"\nFichier cuda: "+nom_cuda+"\nFichier fortran: "+nom_fortran, fontdict=None,
				size='xx-small')
		grid(True)
		savefig( path_dossier_sortie+'/rapport_'+type_donnees+'_Fortran_Cuda_phi=' +str(phi[iphi])+'.png', dpi=(140) )
		
		##########################################
		##				DIFFERENCE				##
		##########################################
		figure()
		plot( theta[dep:fin], data_fortran[iphi][dep:fin]-data_cuda[iphi][dep:fin] )
		
		max_diff = (abs(data_fortran[iphi,dep:fin]-data_cuda[iphi,dep:fin])).max()
		axis([0,theta[fin],-1.01*max_diff, 1.01*max_diff])
		title( 'Difference des '+type_donnees+ ' Fortran - Cuda pour phi='+str(phi[iphi])+' deg' )
		xlabel( 'Theta (deg)' )
		ylabel( 'Difference des '+type_donnees )
		figtext(0.4, 0.25, commentaire+" deg", fontdict=None)
		figtext(0, 0, "Date: "+date_simu+"\nFichier cuda: "+nom_cuda+"\nFichier fortran: "+nom_fortran, fontdict=None,
				size='xx-small')
		grid(True)
		savefig( path_dossier_sortie+'/difference_'+type_donnees+'_Fortran_Cuda_phi='+str(phi[iphi])+'.png', dpi=(140) )

	
else:
	sys.stdout.write('Les tableaux ne font pas la meme taille\n')
	sys.exit()


##########################################################
##				CREATION FICHIER PARAMETRE				##
##########################################################

# creation du fichier contenant les parametres de la simulation
fichierSortie = open(path_dossier_sortie+'/Parametres.txt', 'w')

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

# Ecriture dans le fichier
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

fichierSortie.write("\n\n##### Paramètres Fortran #####\n")
fichierSortie.write("Nombres de photons: {0:.2e}\n".format(tab_fortran['nphotons']))
fichierSortie.write("Germe: " + str(tab_fortran['initgerme']) + "\n")
fichierSortie.write("\n\n")


##################################################
##				CALCUL STATISTIQUES				##
##################################################

moyDiff = 0		# Moyenne de Fortran-Cuda
sigmaDiff = 0
moyDiffAbs = 0	# Moyenne de |Fortran-Cuda|
sigmaDiffAbs = 0
moyRap = 0		# Moyenne de Fortran/Cuda
sigmaRap = 0
moyRapAbs = 0	# Moyenne de |Fortran/Cuda|
sigmaRapAbs = 0

##-- Calcul des moyennes --##

for iphi in xrange(NBPHI_cuda):
	for ith in xrange(dep,fin):	# Calcul sur tout l'espace
		moyDiff += data_fortran[iphi][ith]-data_cuda[iphi][ith]
		moyDiffAbs += abs(data_fortran[iphi][ith]-data_cuda[iphi][ith])
		moyRap += data_fortran[iphi][ith]/data_cuda[iphi][ith]
		moyRapAbs += abs(1-data_fortran[iphi][ith]/data_cuda[iphi][ith] )

moyDiff = moyDiff/((fin-dep)*NBPHI_cuda)
moyDiffAbs = moyDiffAbs/((fin-dep)*NBPHI_cuda)
moyRap = moyRap/((fin-dep)*NBPHI_cuda)
moyRapAbs = moyRapAbs/((fin-dep)*NBPHI_cuda)
	
##-- Calcul des écart type --##

for iphi in xrange(NBPHI_cuda):
	for ith in xrange(dep,fin):	# Calcul sur tout l'espace
		
		# Calcul des écarts type
		sigmaDiff += pow( moyDiff - (data_fortran[iphi][ith]-data_cuda[iphi][ith])  ,2.0 )
		sigmaDiffAbs += pow( moyDiffAbs - abs( data_fortran[iphi][ith]-data_cuda[iphi][ith] ) ,2.0 )
		sigmaRap += pow( moyRap - ( data_fortran[iphi][ith]/data_cuda[iphi][ith] ) ,2.0 )
		sigmaRapAbs += pow( moyRapAbs - abs(1-data_fortran[iphi][ith]/data_cuda[iphi][ith]) ,2.0 )

	
sigmaDiff = math.sqrt( sigmaDiff/((fin-dep)*NBPHI_cuda) )
sigmaDiffAbs = math.sqrt( sigmaDiffAbs/((fin-dep)*NBPHI_cuda) )
sigmaRap = math.sqrt( sigmaRap/((fin-dep)*NBPHI_cuda) )
sigmaRapAbs = math.sqrt( sigmaRapAbs/((fin-dep)*NBPHI_cuda) )


print "\n====================:Résultats:===================="
print 'Moyenne de la différence Fortran-Cuda = {0:.4e}'.format(moyDiff)
print 'Ecart type de la différence Fortran-Cuda = {0:.4e}\n'.format(sigmaDiff)

print 'Moyenne de la valeur absolue de la différence = {0:.4e}'.format(moyDiffAbs)
print 'Ecart type de la valeur absolue de la différence = {0:.4e}\n'.format(sigmaDiffAbs)

print 'Moyenne du rapport Fortran/Cuda = {0:.4e}'.format(moyRap)
print 'Ecart type du rapport Fortran/Cuda = {0:.4e}\n'.format(sigmaRap)

print 'Pourcentage erreur moyen Fortran/Cuda = {0:.4e} %'.format(moyRapAbs*100)
print 'Ecart type de erreur = {0:.4e}\n'.format(sigmaRapAbs)
print "==================================================="

fichierSortie.write( "\n====================:Résultats:====================\n")
fichierSortie.write( 'Moyenne de la différence Fortran-Cuda = {0:.4e}\n'.format(moyDiff))
fichierSortie.write( 'Ecart type de la différence Fortran-Cuda = {0:.4e}\n\n'.format(sigmaDiff))

fichierSortie.write( 'Moyenne de la valeur absolue de la différence = {0:.4e}\n'.format(moyDiffAbs))
fichierSortie.write( 'Ecart type de la valeur absolue de la différence = {0:.4e}\n\n'.format(sigmaDiffAbs))

fichierSortie.write( 'Moyenne du rapport Fortran/Cuda = {0:.4e}\n'.format(moyRap))
fichierSortie.write( 'Ecart type du rapport Fortran/Cuda = {0:.4e}\n\n'.format(sigmaRap))

fichierSortie.write( 'Pourcentage erreur moyen Fortran/Cuda = {0:.4e} %\n'.format(moyRapAbs*100))
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
		
	moyDiff = 0		# Moyenne de Fortran-Cuda
	sigmaDiff = 0
	moyDiffAbs = 0	# Moyenne de |Fortran-Cuda|
	sigmaDiffAbs = 0
	moyRap = 0		# Moyenne de Fortran/Cuda
	sigmaRap = 0
	moyRapAbs = 0	# Moyenne de |Fortran/Cuda|
	sigmaRapAbs = 0
	
	##-- Calcul des moyennes --##
	for ith in xrange(ith0,ith0+pas):
		for iphi in xrange(NBPHI_cuda):
			
			moyDiff += data_fortran[iphi][ith]-data_cuda[iphi][ith]
			moyDiffAbs += abs(data_fortran[iphi][ith]-data_cuda[iphi,ith] )
			moyRap += data_fortran[iphi][ith]/data_cuda[iphi,ith]
			moyRapAbs += abs(1-data_fortran[iphi][ith]/data_cuda[iphi,ith] )
			
	moyDiff = moyDiff/(pas*NBPHI_cuda)
	moyDiffAbs = moyDiffAbs/(pas*NBPHI_cuda)
	moyRap = moyRap/(pas*NBPHI_cuda)
	moyRapAbs = moyRapAbs/(pas*NBPHI_cuda)

	##-- Calcul des écart type --##
	for ith in xrange(ith0,ith0+pas):
		for iphi in xrange(NBPHI_cuda):
			
			sigmaDiff += pow( moyDiff - (data_fortran[iphi][ith]-data_cuda[iphi,ith] ) ,2.0 )
			sigmaDiffAbs += pow( moyDiffAbs - abs( data_fortran[iphi][ith]-data_cuda[iphi,ith] ),2.0 )
			sigmaRap += pow( moyRap - ( data_fortran[iphi][ith]/data_cuda[iphi,ith]) ,2.0 )
			sigmaRapAbs += pow( moyRapAbs - abs(1-data_fortran[iphi][ith]/data_cuda[iphi,ith] ),2.0 )

	sigmaDiff = math.sqrt( sigmaDiff/(pas*NBPHI_cuda) )
	sigmaDiffAbs = math.sqrt( sigmaDiffAbs/(pas*NBPHI_cuda) )
	sigmaRap = math.sqrt( sigmaRap/(pas*NBPHI_cuda) )
	sigmaRapAbs = math.sqrt( sigmaRapAbs/(pas*NBPHI_cuda) )
			

	print "\n====================:Résultats par couronne:===================="
	print '==================:Couronne #{0:2d} -{1:3d}->{2:3d}deg==================='.format(ith0/10,ith0*90/NBTHETA,
																								(ith0+pas)*90/NBTHETA)
	print 'Moyenne de la différence Fortran-Cuda = {0:.4e}'.format(moyDiff)
	print 'Ecart type de la différence Fortran-Cuda = {0:.4e}\n'.format(sigmaDiff)

	print 'Moyenne de la valeur absolue de la différence = {0:.4e}'.format(moyDiffAbs)
	print 'Ecart type de la valeur absolue de la différence = {0:.4e}\n'.format(sigmaDiffAbs)

	print 'Moyenne du rapport Fortran/Cuda = {0:.4e}'.format(moyRap)
	print 'Ecart type du rapport Fortran/Cuda = {0:.4e}\n'.format(sigmaRap)

	print 'Pourcentage erreur Fortran/Cuda = {0:.4e} %'.format(moyRapAbs*100)
	print 'Ecart type de erreur = {0:.4e}\n'.format(sigmaRapAbs)
	print "================================================================"

	fichierSortie.write( "\n====================:Résultats par couronne:====================\n")
	fichierSortie.write( '==================:Couronne #{0:2d}-{1:3d}->{2:3d}deg===================\n'.format(ith0/10,
																						ith0*90/NBTHETA, (ith0+pas)*90/NBTHETA))
	fichierSortie.write( 'Moyenne de la différence Fortran-Cuda = {0:.4e}\n'.format(moyDiff))
	fichierSortie.write( 'Ecart type de la différence Fortran-Cuda = {0:.4e}\n\n'.format(sigmaDiff))

	fichierSortie.write( 'Moyenne de la valeur absolue de la différence = {0:.4e}\n'.format(moyDiffAbs))
	fichierSortie.write( 'Ecart type de la valeur absolue de la différence = {0:.4e}\n\n'.format(sigmaDiffAbs))

	fichierSortie.write( 'Moyenne du rapport Fortran/Cuda = {0:.4e}\n'.format(moyRap))
	fichierSortie.write( 'Ecart type du rapport Fortran/Cuda = {0:.4e}\n\n'.format(sigmaRap))

	fichierSortie.write( 'Pourcentage erreur Fortran/Cuda = {0:.4e} %\n'.format(moyRapAbs*100))
	fichierSortie.write( 'Ecart type de erreur Fortran/Cuda = {0:.4e}\n'.format(sigmaRapAbs))
	fichierSortie.write( "================================================================\n")

	ith0 += pas
	
fichierSortie.close()

print '################################################'
print 'Simulation de {0} terminee pour Cuda-Fortran'.format(type_donnees)
print '################################################'


