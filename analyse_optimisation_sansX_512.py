#!/usr/bin/env python
import sys
import os
import numpy
from time import time
from pylab import *
import marshal

# -------------------------------------------------------------------------------------------

def lancerSimulation(tabTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI) :

	temps = 0
	done = 0
	for sim in listeTemps:
		if (sim[0]==NBPHOTONS and sim[1]==NBLOOP and sim[2]==XBLOCK and sim[3]==YBLOCK and sim[4]==XGRID and sim[5]==YGRID and sim[6]==NBTHETA and sim[7]==NBPHI):
			temps = sim[8]
			done = 1
	if done == 0:
		fichierParametres = open("tmp/opt_sansX_512_param.txt", "w")
		fichierParametres.write("NBPHOTONS = " + str(NBPHOTONS) + "\n")
		fichierParametres.write("NBLOOP = " + str(NBLOOP) + "\n")
		fichierParametres.write("XBLOCK = " + str(XBLOCK) + "\n")
		fichierParametres.write("YBLOCK = " + str(YBLOCK) + "\n")
		fichierParametres.write("XGRID = " + str(XGRID) + "\n")
		fichierParametres.write("YGRID = " + str(YGRID) + "\n")
		fichierParametres.write("NBTHETA = " + str(NBTHETA) + "\n")
		fichierParametres.write("NBPHI = " + str(NBPHI) + "\n")
		fichierParametres.write("NBSTOKES = 2\n")
		fichierParametres.write("THSDEG = 70.\n")
		fichierParametres.write("LAMBDA = 635.\n")
		fichierParametres.write("TAURAY = 0.05330\n")
		fichierParametres.write("TAUAER = 0.0\n")
		fichierParametres.write("W0AER = 0.966883724738261\n")
		fichierParametres.write("PROFIL = 0\n")
		fichierParametres.write("HA = 2.0\n")
		fichierParametres.write("HR = 8.0\n")
		fichierParametres.write("ZMIN = 0.\n")
		fichierParametres.write("ZMAX = 1.\n")
		fichierParametres.write("WINDSPEED = 5.0\n")
		fichierParametres.write("NH2O = 1.33\n")
		fichierParametres.write("SIM = -2\n")
		fichierParametres.write("SUR = 1\n")
		fichierParametres.write("DIOPTRE = 1\n")
		fichierParametres.write("CONPHY = 0.1\n")
		fichierParametres.write("DIFFF = 0\n")
		fichierParametres.close()
		
		start = time()
		os.system("./Prog tmp/opt_sansX_512_param.txt")
		temps = time() - start
		os.system("rm -f tmp/opt_sansX_512_param.txt")
		
		# lecture fichier hdf
		file_hdf = 'out_prog/Resultats.hdf'
		hdf = pyhdf.SD.SD(file_hdf)
		# lecture du nombre de photons traites et du nombre de photons demandes
		NBPHOTONS = getattr(hdf,'NBPHOTONS')
		nbPhotonsTot = getattr(hdf,'nbPhotonsTot')
		
		temps = temps*NBPHOTONS/nbPhotonsTot

		listeTemps.append([NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI,temps])
		marshal.dump(listeTemps, open("tmp/opt_sansX_512_sauv.txt", 'wb'))
	return temps

# -------------------------------------------------------------------------------------------

if os.path.exists('tmp/opt_sansX_512_sauv.txt'):
	cont = 1
	while cont:
		sys.stdout.write('Continuer avec les simulations sauvegardees? [Y/n]\n')
		choice = raw_input().lower()
		if (choice == '' or choice == 'y' or choice == 'Y' or choice == 'yes' or choice == 'Yes'):
			listeTemps = marshal.load(open('tmp/opt_sansX_512_sauv.txt', "rb"))
			cont = 0
		elif (choice == 'n' or choice == 'N' or choice == 'no' or choice == 'No'):
			listeTemps = []
			os.system('rm -f tmp/opt_sansX_512_sauv.txt')
			cont = 0
else:
	listeTemps = []

os.system("make clean")
os.system("make")
os.system("rm -rf out_scripts/analyse_optimisation_sansX_512")
os.mkdir("out_scripts/analyse_optimisation_sansX_512")

# -----------------------------------------Variation NBLOOP normal------------------------------------------------

XBLOCK = 12
YBLOCK = 1
XGRID = 12
YGRID = 1
NBTHETA = 180
NBPHI = 360
listeNBPHOTONS = range(500000000, 1100000000, 100000000)
listeNBLOOP = range(1000, 50000, 4000)
listeNBLOOP.extend([60000, 80000, 100000])
listePlots = []
listeLegends = []
for NBPHOTONS in listeNBPHOTONS:
	listeGraphe = []
	for NBLOOP in listeNBLOOP:
		temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
		listeGraphe.append(temps)
	listePlots.append(plot(listeNBLOOP,listeGraphe, marker='.'))
	listeLegends.append('NBPHOTONS = ' + str(NBPHOTONS))
legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps en fonction de NBLOOP pour differents NBPHOTONS")
xlabel("NBLOOP")
ylabel("TEMPS")
grid(True)
savefig("out_scripts/analyse_optimisation_sansX_512/Variation NBLOOP.png", dpi=(140))
figure()

# ----------------------------------------Variation NBLOOP zoom------------------------------------------------

XBLOCK = 16
YBLOCK = 1
XGRID = 16
YGRID = 1
NBTHETA = 180
NBPHI = 360
listeNBPHOTONS = range(100000000, 510000000, 100000000)
listeNBLOOP = range(1000, 50000, 4000)
listePlots = []
listeLegends = []
for NBPHOTONS in listeNBPHOTONS:
	listeGraphe = []
	for NBLOOP in listeNBLOOP:
		temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
		listeGraphe.append(temps)
	listePlots.append(plot(listeNBLOOP,listeGraphe, marker='.'))
	listeLegends.append('NBPHOTONS = ' + str(NBPHOTONS))
legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps en fonction de NBLOOP pour differents NBPHOTONS")
xlabel("NBLOOP")
ylabel("TEMPS")
grid(True)
savefig("out_scripts/analyse_optimisation_sansX_512/Variation NBLOOP zoom.png", dpi=(140))
figure()

# ----------------------------------------Variation NBPHOTONS normal------------------------------------------------

NBLOOP = 10000
XBLOCK = 16
YBLOCK = 1
XGRID = 16
YGRID = 1
NBTHETA = 180
NBPHI = 360
listeNBPHOTONS = [10000000, 50000000, 100000000, 200000000, 400000000, 600000000, 800000000, 1000000000]
listePlots = []
listeLegends = []

listeGraphe = []
for NBPHOTONS in listeNBPHOTONS:
	temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
	listeGraphe.append(temps)
listePlots.append(plot(listeNBPHOTONS,listeGraphe, marker='.'))
listeLegends.append("NBTHREADS=4*4*4*4")

listeNBPHOTONS = [10000000, 100000000, 500000000, 1000000000, 5000000000]
YBLOCK = 8
YGRID = 8
listeGraphe = []
for NBPHOTONS in listeNBPHOTONS:
	temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
	listeGraphe.append(temps)
listePlots.append(plot(listeNBPHOTONS,listeGraphe, marker='.'))
listeLegends.append("NBTHREADS=4*8*4*8")

legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps en fonction de NBPHOTONS")
xmin, xmax = xlim()   # return the current xlim
xlim(0, xmax)  # set the xlim to xmin, xmax
ymin, ymax = ylim()   # return the current xlim
ylim(0, ymax)  # set the xlim to xmin, xmax
xlabel("NBPHOTONS")
ylabel("TEMPS")
grid(True)
savefig("out_scripts/analyse_optimisation_sansX_512/Variation NBPHOTONS.png", dpi=(140))
figure()

# ----------------------------------------Variation NBPHOTONS zoom-------------------------------------

NBLOOP = 1000
XBLOCK = 16
YBLOCK = 1
XGRID = 16
YGRID = 1
NBTHETA = 180
NBPHI = 360
listeNBPHOTONS = [100000, 500000, 1000000, 2000000, 4000000, 6000000, 8000000, 10000000]
listePlots = []
listeLegends = []

listeGraphe = []
for NBPHOTONS in listeNBPHOTONS:
	temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
	listeGraphe.append(temps)
listePlots.append(plot(listeNBPHOTONS,listeGraphe, marker='.'))
listeLegends.append("NBTHREADS=4*4*4*4")

YBLOCK = 8
YGRID = 8
listeGraphe = []
for NBPHOTONS in listeNBPHOTONS:
	temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
	listeGraphe.append(temps)
listePlots.append(plot(listeNBPHOTONS,listeGraphe, marker='.'))
listeLegends.append("NBTHREADS=4*8*4*8")

legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps en fonction de NBPHOTONS")
xmin, xmax = xlim()   # return the current xlim
xlim(0, xmax)  # set the xlim to xmin, xmax
ymin, ymax = ylim()   # return the current xlim
ylim(0, ymax)  # set the xlim to xmin, xmax
xlabel("NBPHOTONS")
ylabel("TEMPS")
grid(True)
savefig("out_scripts/analyse_optimisation_sansX_512/Variation NBPHOTONS zoom.png", dpi=(140))
figure()

# -------------------------------- Variation NBCASES ---------------------------------

listeNBPHOTONS = [10000000,100000000]
NBLOOP = 1000
XBLOCK = 64
YBLOCK = 1
XGRID = 42
YGRID = 1
listeNBTHETA = range(1,1000,5)
NBPHI = 100
listeLegend = []
listeAbscisses = []
for NBTHETA in listeNBTHETA:
	listeAbscisses.append(NBTHETA*NBPHI)
listePlot = []
for NBPHOTONS in listeNBPHOTONS:
	listeGraphe = []
	for NBTHETA in listeNBTHETA:
		temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
		listeGraphe.append(temps)
	listePlot.append(plot(listeAbscisses,listeGraphe, marker='.'))
	listeLegend.append("NBPHOTONS = " + str(NBPHOTONS))
legend(listePlot, listeLegend, loc='best', numpoints=1)
title("Temps en fonction du nombre de cases")
xlabel("Nombre de cases")
ylabel("TEMPS")
grid(True)
savefig("out_scripts/analyse_optimisation_sansX_512/Variations NBCASES.png", dpi=(140))
figure()

# -------------------------------- Variation NBCASES zoom---------------------------------

listeNBPHOTONS = [10000000,100000000]
NBLOOP = 1000
XBLOCK = 64
YBLOCK = 1
XGRID = 42
YGRID = 1
listeNBTHETA = range(500,800,2)
NBPHI = 100
listeLegend = []
listeAbscisses = []
for NBTHETA in listeNBTHETA:
	listeAbscisses.append(NBTHETA*NBPHI)
listePlot = []
for NBPHOTONS in listeNBPHOTONS:
	listeGraphe = []
	for NBTHETA in listeNBTHETA:
		temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
		listeGraphe.append(temps)
	listePlot.append(plot(listeAbscisses,listeGraphe, marker='.'))
	listeLegend.append("NBPHOTONS = " + str(NBPHOTONS))
legend(listePlot, listeLegend, loc='best', numpoints=1)
title("Temps en fonction du nombre de cases")
xlabel("Nombre de cases")
ylabel("TEMPS")
grid(True)
savefig("out_scripts/analyse_optimisation_sansX_512/Variations NBCASES zoom.png", dpi=(140))
figure()

# --------------------------------Variation NBTHREADS NBBLOCKS---------------------------------

NBPHOTONS = 100000000
NBLOOP = 2000
listeXBLOCK = range(1,513,5)
YBLOCK = 1
listeXGRID = [100,200,500, 1000, 2000, 5000, 10000, 50000]
YGRID = 1
NBTHETA = 180
NBPHI = 360
listePlots = []
listeLegends = []
for XGRID in listeXGRID:
	listeGraphe = []
	for XBLOCK in listeXBLOCK:
		temps = lancerSimulation(listeTemps,NBPHOTONS,int(110000000/(XBLOCK*XGRID)),XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
		listeGraphe.append(temps)
	listePlots.append(plot(listeXBLOCK,listeGraphe, marker='.'))
	listeLegends.append('NBBLOCKS = ' + str(XGRID))
legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps en fonction du NBTHREADS pour differents NBBLOCKS (NBPHOTONS=1milliard)")
xlabel("NBTHREADS")
ylabel("TEMPS")
grid(True)
savefig("out_scripts/analyse_optimisation_sansX_512/Variation NBTHREADS NBBLOCKS.png", dpi=(140))
figure()

# --------------------------------Variation NBBLOCKS---------------------------------

NBPHOTONS = 1000000000
NBLOOP = 2000
listeXBLOCK = [64,128, 256, 512]
YBLOCK = 1
listeXGRID = range(20,5000,50)
YGRID = 1
NBTHETA = 180
NBPHI = 360
listePlots = []
listeLegends = []
for XBLOCK in listeXBLOCK:
	listeGraphe = []
	for XGRID in listeXGRID:
		temps = lancerSimulation(listeTemps,NBPHOTONS,int(110000000/(XBLOCK*XGRID)),XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
		listeGraphe.append(temps)
	listePlots.append(plot(listeXGRID,listeGraphe, marker='.'))
	listeLegends.append('NBTHREADS/BLOCK = ' + str(XBLOCK))
legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps en fonction du NBBLOCKS pour differents NBTHREADS (NBPHOTONS=1milliard)")
xlabel("NBBLOCKS")
ylabel("TEMPS")
grid(True)
savefig("out_scripts/analyse_optimisation_sansX_512/Variation NBBLOCKS.png", dpi=(140))
figure()

# --------------------------------Variation NBBLOCKS zoom---------------------------------

NBPHOTONS = 1000000000
NBLOOP = 2000
listeXBLOCK = [64,128, 256, 512]
YBLOCK = 1
listeXGRID = range(20,700,2)
YGRID = 1
NBTHETA = 180
NBPHI = 360
listePlots = []
listeLegends = []
for XBLOCK in listeXBLOCK:
	listeGraphe = []
	for XGRID in listeXGRID:
		temps = lancerSimulation(listeTemps,NBPHOTONS,int(110000000/(XBLOCK*XGRID)),XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
		listeGraphe.append(temps)
	listePlots.append(plot(listeXGRID,listeGraphe, marker='.'))
	listeLegends.append('NBTHREADS/BLOCK = ' + str(XBLOCK))
legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps en fonction du NBBLOCKS pour differents NBTHREADS (NBPHOTONS=1milliard)")
xlabel("NBBLOCKS")
ylabel("TEMPS")
grid(True)
savefig("out_scripts/analyse_optimisation_sansX_512/Variation NBBLOCKS zoom.png", dpi=(140))
figure()

# --------------------------------------------------------------------------------------------

# os.system("shutdown -s -t 0")
