#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import warnings
warnings.simplefilter("ignore",DeprecationWarning)
import pyhdf.SD
from pylab import *
from os.path import basename


##################################################
##                PARAMETRES                    ##
##################################################

#
# Paramètres à modifier
#
#-----------------------------------------------------------------------------------------------------------------------

angle = '30'
type_simu = "molecules_seules"

path_sos = '../validation/SOS-ths_30-tauray_0.0533-rho_0.0-PP-UP.txt'
path_cuda = "../resultat/PP-Rayleigh-Sol_noir-1e9_photons.hdf"


# Indices ci-dessus ont été mis en place car ils permettent de rogner la simulation si nécessaire.
# Les bords peuvent fausser les graphiques.
dep = 3            # Indice de départ pour le tracé
fin = 177         # Indice de fin pour le tracé
pas_figure = 15   # Pas en phi pour le tracé des graphiques

type_donnees = "I" # I, Q, U, LP

# Si le dossier suivant existe deja il est supprime puis recree
path_dossier_sortie = '../validation-cuda-sos-PP/' + type_donnees


#-----------------------------------------------------------------------------------------------------------------------


######################################################
##                INITIALISATIONS                   ##
######################################################


nom_sos = basename(path_sos).replace('.txt', '')
nom_cuda = basename(path_cuda).replace('.hdf', '')


if type_donnees == 'I':
    nom_data_cuda = "Valeurs de la reflectance (I)"
    colonne_donnee_sos = 2

elif type_donnees == 'Q':
    nom_data_cuda = "Valeurs de Q"
    colonne_donnee_sos = 3

elif type_donnees == 'U':
    nom_data_cuda = "Valeurs de U"
    colonne_donnee_sos = 4

elif type_donnees == 'LP':
    nom_data_cuda = "Valeurs de la lumiere polarisee (LP)"
    colonne_donnee_sos = 5

else:
    print 'Erreur type_donnees'



##########################################################
##                DONNEES FICHIER CUDA                    ##
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


##################################################
##                DONNEES FICHIER SOS                ##
##################################################

# verification de l'existence du fichier SOS
if os.path.exists(path_sos):

    # data_sos[iphi][ith] = grandeur
    # ith est le num de la ith-ème boite theta. Boites de pas de 0.5 centrées tous les 0.5
    # iphi est le num de la ith-ème boite phi
    data_sos = zeros((NBPHI_cuda,2*(NBTHETA_cuda-1)),dtype=float)
    fichier_sos = open(path_sos, "r")

    for ligne in fichier_sos:
        donnees = ligne.rstrip('\n\r').split("\t")        # Lecture
        if donnees[0]=='':
            donnees = donnees[1:]                        # Suppression des possibles tabulations en début de ligne

        if float(donnees[1]) < 89.6:
            data_sos[int(float(donnees[0]))][int(2*float(donnees[1]))] = float(donnees[colonne_donnee_sos])

    fichier_sos.close()

else:
    sys.stdout.write("Pas de fichier "+path_sos+"\n")
    sys.exit()

##############################################################
##                INFORMATION A L'UTILISATEUR                    ##
##############################################################

sys.stdout.write("\n#-------------------------------------------------------------------------------#\n")
sys.stdout.write("# Le fichier Cuda est " + path_cuda + "\n")
sys.stdout.write("# Le fichier SOS est " + path_sos + "\n")
sys.stdout.write("# Les résultats sont stockés dans " + path_dossier_sortie + "\n")
sys.stdout.write("#-------------------------------------------------------------------------------#\n")


os.system("rm -rf "+ path_dossier_sortie)
os.system("mkdir -p "+ path_dossier_sortie)


##################################################################################
##                CREATION/CHOIX/MODIFICATION DE CERTAINES DONNES                    ##
##################################################################################

# Pour comparer les 2 resultats il faut que phi parcourt un meme intervalle et qu'il y ait le meme nombre de boites selon phi
# SOS : intervalle=[0,PI]
# Cuda : intervalle=[0,2PI]  nombre_de_boites=NBPHI_cuda
# On va projeter les resultats du cuda sur [0,PI]

data_cuda = zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)

#if choix != 'u':
    #for iphi in xrange(0,NBPHI_cuda):
        #for ith in xrange(NBTHETA_cuda):
            #data_cuda[iphi][ith] = (data[iphi,ith]+data[NBPHI_cuda-iphi-1,ith] )/2


#else:    # Il ne faut pas moyenner U qui est antisymétrique
data_cuda = data[0:NBPHI_cuda,]

# Infos en commentaire sur le graph
commentaire = type_simu + ' - ' + angle

##########################################################
##                CREATION DES GRAPHIQUES                    ##
##########################################################

for iphi in xrange(0,NBPHI_cuda,pas_figure):
    print 'iphi = %d/%d' % (iphi, NBPHI_cuda)

    figure()
    # Référence
    plot(theta[dep:fin], data_sos[iphi][dep:fin],label='SOS')

    # Cuda
    plot(theta[dep:fin], data_cuda[iphi][dep:fin],label='CUDA')

    # commun
    legend(loc='best')
    title( type_donnees + ' pour Cuda et SOS pour phi='+str(phi[iphi])+' deg' )
    xlabel( 'Theta (deg)' )
    ylabel( type_donnees, rotation='horizontal' )
    figtext(0.25, 0.7, commentaire+" deg", fontdict=None)
    figtext(0, 0, "\nFichier SOS: "+nom_sos+"\nFichier cuda: "+nom_cuda, fontdict=None, size='xx-small')
    grid(True)
    savefig( path_dossier_sortie+'/c_'+type_donnees+'_SOS_Cuda_phi='+str(phi[iphi])+'.png', dpi=(140) )
    
    ##########################################
    ##                RAPPORT                    ##
    ##########################################
    figure()
    plot( theta[dep:fin], data_sos[iphi][dep:fin]/data_cuda[iphi][dep:fin],label='Rapport de '+type_donnees+' SOS/Cuda'  )
    
    #Régression linéaire
    (ar,br) = polyfit( theta[dep:fin], data_sos[iphi][dep:fin]/data_cuda[iphi][dep:fin] ,1 )
    regLin = polyval( [ar,br],theta[dep:fin] )
    
    plot(theta[dep:fin], regLin,label='Regression lineaire y='+str(ar)+'x+'+str(br)) 
    legend(loc='best')
    title( 'Rapport des '+type_donnees+' SOS_Cuda pour phi='+str(phi[iphi])+' deg' )
    xlabel( 'Theta (deg)' )
    ylabel( 'Rapport des '+type_donnees )
    figtext(0.4, 0.25, commentaire+" deg", fontdict=None)
    figtext(0, 0, "\nFichier SOS: "+nom_sos+"\nFichier cuda: "+nom_cuda, fontdict=None, size='xx-small')
    grid(True)
    savefig( path_dossier_sortie+'/rapport_'+type_donnees+'_SOS_Cuda_phi=' +str(phi[iphi])+'.png', dpi=(140) )

    ##########################################
    ##                DIFFERENCE                ##
    ##########################################
    figure()
    plot( theta[dep:fin], data_sos[iphi][dep:fin]-data_cuda[iphi][dep:fin] )
    title( 'Difference des '+type_donnees+ ' SOS - Cuda pour phi='+str(phi[iphi])+' deg' )
    xlabel( 'Theta (deg)' )
    ylabel( 'Difference des '+type_donnees )
    figtext(0.4, 0.25, commentaire+" deg", fontdict=None)
    figtext(0, 0, "\nFichier SOS: "+nom_sos+"\nFichier cuda: "+nom_cuda, fontdict=None, size='xx-small')
    grid(True)
    savefig( path_dossier_sortie+'/difference_'+type_donnees+'_SOS_Cuda_phi='+str(phi[iphi])+'.png', dpi=(140) )


######################################################################
##                CREATION FICHIER PARAMETRE/STATISTIQUES                ##
######################################################################

# Création du fichier texte de sortie
fichierSortie = open(path_dossier_sortie+"/Statistiques_SOS_CUDA_"+nom_cuda+".txt", "w")

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
if 'CONPHY' in sd_cuda.attributes():
    CONPHY = getattr(sd_cuda,'CONPHY')
else:
    CONPHY = None
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
fichierSortie.write("SUR = " + str(SUR) + "\n")
fichierSortie.write("DIOPTRE = " + str(DIOPTRE) + "\n")
fichierSortie.write("CONPHY = " + str(CONPHY) + "\n")
fichierSortie.write("PATHRESULTATSHDF = " + str(PATHRESULTATSHDF) + "\n")
fichierSortie.write("PATHTEMOINHDF = " + str(PATHTEMOINHDF) + "\n")
fichierSortie.write("PATHDIFFAER = " + str(PATHDIFFAER) + "\n")
fichierSortie.write("PATHPROFILATM = " + str(PATHPROFILATM) + "\n")


##################################################
##                CALCUL STATISTIQUES                ##
##################################################

moyDiff = 0        # Moyenne de SOS-Cuda
sigmaDiff = 0
moyDiffAbs = 0    # Moyenne de |SOS-Cuda|
sigmaDiffAbs = 0
moyRap = 0        # Moyenne de SOS/Cuda
sigmaRap = 0
moyRapAbs = 0    # Moyenne de |SOS/Cuda|
sigmaRapAbs = 0

##-- Calcul des moyennes --##

for iphi in xrange(NBPHI_cuda):
    for ith in xrange(dep,fin):    # Calcul sur tout l'espace
        moyDiff += data_sos[iphi][ith]-data_cuda[iphi][ith]
        moyDiffAbs += abs(data_sos[iphi][ith]-data_cuda[iphi][ith])
        moyRap += data_sos[iphi][ith]/data_cuda[iphi][ith]
        moyRapAbs += abs(1-data_sos[iphi][ith]/data_cuda[iphi][ith] )

moyDiff = moyDiff/((fin-dep)*NBPHI_cuda)
moyDiffAbs = moyDiffAbs/((fin-dep)*NBPHI_cuda)
moyRap = moyRap/((fin-dep)*NBPHI_cuda)
moyRapAbs = moyRapAbs/((fin-dep)*NBPHI_cuda)
    
##-- Calcul des écart type --##

for iphi in xrange(NBPHI_cuda):
    for ith in xrange(dep,fin):    # Calcul sur tout l'espace
        
        # Calcul des écarts type
        sigmaDiff += pow( moyDiff - (data_sos[iphi][ith]-data_cuda[iphi][ith])  ,2.0 )
        sigmaDiffAbs += pow( moyDiffAbs - abs( data_sos[iphi][ith]-data_cuda[iphi][ith] ) ,2.0 )
        sigmaRap += pow( moyRap - ( data_sos[iphi][ith]/data_cuda[iphi][ith] ) ,2.0 )
        sigmaRapAbs += pow( moyRapAbs - abs(1-data_sos[iphi][ith]/data_cuda[iphi][ith]) ,2.0 )

    
sigmaDiff = math.sqrt( sigmaDiff/((fin-dep)*NBPHI_cuda) )
sigmaDiffAbs = math.sqrt( sigmaDiffAbs/((fin-dep)*NBPHI_cuda) )
sigmaRap = math.sqrt( sigmaRap/((fin-dep)*NBPHI_cuda) )
sigmaRapAbs = math.sqrt( sigmaRapAbs/((fin-dep)*NBPHI_cuda) )


print "\n====================:Résultats:===================="
print 'Moyenne de la différence SOS-Cuda = {0:.4e}'.format(moyDiff)
print 'Ecart type de la différence SOS-Cuda = {0:.4e}\n'.format(sigmaDiff)

print 'Moyenne de la valeur absolue de la différence = {0:.4e}'.format(moyDiffAbs)
print 'Ecart type de la valeur absolue de la différence = {0:.4e}\n'.format(sigmaDiffAbs)

print 'Moyenne du rapport SOS/Cuda = {0:.4e}'.format(moyRap)
print 'Ecart type du rapport SOS/Cuda = {0:.4e}\n'.format(sigmaRap)

print 'Pourcentage erreur moyen SOS/Cuda = {0:.4e} %'.format(moyRapAbs*100)
print 'Ecart type de erreur = {0:.4e}\n'.format(sigmaRapAbs)
print "==================================================="

fichierSortie.write( "\n====================:Résultats:====================\n")
fichierSortie.write( 'Moyenne de la différence SOS-Cuda = {0:.4e}\n'.format(moyDiff))
fichierSortie.write( 'Ecart type de la différence SOS-Cuda = {0:.4e}\n\n'.format(sigmaDiff))

fichierSortie.write( 'Moyenne de la valeur absolue de la différence = {0:.4e}\n'.format(moyDiffAbs))
fichierSortie.write( 'Ecart type de la valeur absolue de la différence = {0:.4e}\n\n'.format(sigmaDiffAbs))

fichierSortie.write( 'Moyenne du rapport SOS/Cuda = {0:.4e}\n'.format(moyRap))
fichierSortie.write( 'Ecart type du rapport SOS/Cuda = {0:.4e}\n\n'.format(sigmaRap))

fichierSortie.write( 'Pourcentage erreur moyen SOS/Cuda = {0:.4e} %\n'.format(moyRapAbs*100))
fichierSortie.write( 'Ecart type de erreur = {0:.4e}\n'.format(sigmaRapAbs))
fichierSortie.write( "===================================================\n")


##-- Calculs par couronne de theta --##
# Dth de 10 deg, il y aura donc NBTHETA/180*10 échantillons par couronnes car theta=[0;180°]
pas = NBTHETA/180*10
ith0 = 1

for icouronne in xrange( NBTHETA/(NBTHETA/180*10) ):    # Pour chaque couronne
    
    if icouronne == 17 or icouronne==0:    #on modifie les paramètres pour éviter le débodement de tableau sur la dernière couronne
        pas=NBTHETA/180*10 - 1
    else:
        pas = NBTHETA/180*10
        
    moyDiff = 0        # Moyenne de SOS-Cuda
    sigmaDiff = 0
    moyDiffAbs = 0    # Moyenne de |SOS-Cuda|
    sigmaDiffAbs = 0
    moyRap = 0        # Moyenne de SOS/Cuda
    sigmaRap = 0
    moyRapAbs = 0    # Moyenne de |SOS/Cuda|
    sigmaRapAbs = 0
    
    ##-- Calcul des moyennes --##
    for ith in xrange(ith0,ith0+pas):
        for iphi in xrange(NBPHI_cuda):
            
            moyDiff += data_sos[iphi][ith]-data_cuda[iphi][ith]
            moyDiffAbs += abs(data_sos[iphi][ith]-data_cuda[iphi,ith] )
            moyRap += data_sos[iphi][ith]/data_cuda[iphi,ith]
            moyRapAbs += abs(1-data_sos[iphi][ith]/data_cuda[iphi,ith] )
            
    moyDiff = moyDiff/(pas*NBPHI_cuda)
    moyDiffAbs = moyDiffAbs/(pas*NBPHI_cuda)
    moyRap = moyRap/(pas*NBPHI_cuda)
    moyRapAbs = moyRapAbs/(pas*NBPHI_cuda)

    ##-- Calcul des écart type --##
    for ith in xrange(ith0,ith0+pas):
        for iphi in xrange(NBPHI_cuda):
            
            sigmaDiff += pow( moyDiff - (data_sos[iphi][ith]-data_cuda[iphi,ith] ) ,2.0 )
            sigmaDiffAbs += pow( moyDiffAbs - abs( data_sos[iphi][ith]-data_cuda[iphi,ith] ),2.0 )
            sigmaRap += pow( moyRap - ( data_sos[iphi][ith]/data_cuda[iphi,ith]) ,2.0 )
            sigmaRapAbs += pow( moyRapAbs - abs(1-data_sos[iphi][ith]/data_cuda[iphi,ith] ),2.0 )

    sigmaDiff = math.sqrt( sigmaDiff/(pas*NBPHI_cuda) )
    sigmaDiffAbs = math.sqrt( sigmaDiffAbs/(pas*NBPHI_cuda) )
    sigmaRap = math.sqrt( sigmaRap/(pas*NBPHI_cuda) )
    sigmaRapAbs = math.sqrt( sigmaRapAbs/(pas*NBPHI_cuda) )
            

    print "\n====================:Résultats par couronne:===================="
    print '==================:Couronne #{0:2d} -{1:3d}->{2:3d}deg==================='.format(ith0/10,ith0*90/NBTHETA,
                                                                                                (ith0+pas)*90/NBTHETA)
    print 'Moyenne de la différence SOS-Cuda = {0:.4e}'.format(moyDiff)
    print 'Ecart type de la différence SOS-Cuda = {0:.4e}\n'.format(sigmaDiff)

    print 'Moyenne de la valeur absolue de la différence = {0:.4e}'.format(moyDiffAbs)
    print 'Ecart type de la valeur absolue de la différence = {0:.4e}\n'.format(sigmaDiffAbs)

    print 'Moyenne du rapport SOS/Cuda = {0:.4e}'.format(moyRap)
    print 'Ecart type du rapport SOS/Cuda = {0:.4e}\n'.format(sigmaRap)

    print 'Pourcentage erreur SOS/Cuda = {0:.4e} %'.format(moyRapAbs*100)
    print 'Ecart type de erreur = {0:.4e}\n'.format(sigmaRapAbs)
    print "================================================================"

    fichierSortie.write( "\n====================:Résultats par couronne:====================\n")
    fichierSortie.write( '==================:Couronne #{0:2d}-{1:3d}->{2:3d}deg===================\n'.format(ith0/10,
                                                                                        ith0*90/NBTHETA, (ith0+pas)*90/NBTHETA))
    fichierSortie.write( 'Moyenne de la différence SOS-Cuda = {0:.4e}\n'.format(moyDiff))
    fichierSortie.write( 'Ecart type de la différence SOS-Cuda = {0:.4e}\n\n'.format(sigmaDiff))

    fichierSortie.write( 'Moyenne de la valeur absolue de la différence = {0:.4e}\n'.format(moyDiffAbs))
    fichierSortie.write( 'Ecart type de la valeur absolue de la différence = {0:.4e}\n\n'.format(sigmaDiffAbs))

    fichierSortie.write( 'Moyenne du rapport SOS/Cuda = {0:.4e}\n'.format(moyRap))
    fichierSortie.write( 'Ecart type du rapport SOS/Cuda = {0:.4e}\n\n'.format(sigmaRap))

    fichierSortie.write( 'Pourcentage erreur SOS/Cuda = {0:.4e} %\n'.format(moyRapAbs*100))
    fichierSortie.write( 'Ecart type de erreur SOS/Cuda = {0:.4e}\n'.format(sigmaRapAbs))
    fichierSortie.write( "================================================================\n")

    ith0 += pas
    
fichierSortie.close()

print '################################################'
print 'Simulation de {0} terminee pour Cuda-SOS'.format(type_donnees)
print '################################################'
