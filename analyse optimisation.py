#!/usr/bin/env python
import sys
import os
import numpy
from time import time
# import pylab
from pylab import *

# -------------------------------------------------------------------------------------------

def lancerSimulation(tabTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI) :

	temps = float(0)
	done = 0
	for tab in listeTemps:
		if (tab[0:8] == [NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI]).all():
			temps = tab[8]
			done = 1
	if done == 0:
		fichierParametres = open("Parametres.txt", "w") # argh j'ai tout ecrase !
		fichierParametres.write("NBPHOTONS = " + str(NBPHOTONS) + "\n")
		fichierParametres.write("NBLOOP = " + str(NBLOOP) + "\n")
		fichierParametres.write("XBLOCK = " + str(XBLOCK) + "\n")
		fichierParametres.write("YBLOCK = " + str(YBLOCK) + "\n")
		fichierParametres.write("XGRID = " + str(XGRID) + "\n")
		fichierParametres.write("YGRID = " + str(YGRID) + "\n")
		fichierParametres.write("NBTHETA = " + str(NBTHETA) + "\n")
		fichierParametres.write("NBPHI = " + str(NBPHI) + "\n")
		fichierParametres.write("NBSTOKES = 2\n")
		fichierParametres.write("THETASOL = 70.\n")
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
		os.system("./LancePhotons Parametres.txt")
		temps = time() - start
		
		listeTemps.append(array([NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI,temps]))
	return temps

# -------------------------------------------------------------------------------------------

os.system("cd /home/tristan/Desktop/TristanCuda")
os.system("make clean")
os.system("make")
listeTemps = []

# ----------------------------------------Variation NBLOOP zoom------------------------------------------------

XBLOCK = 4
YBLOCK = 4
XGRID = 4
YGRID = 4
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
savefig("out/Variation NBLOOP zoom.png", dpi=(140))
figure()

# -----------------------------------------Variation NBLOOP normal------------------------------------------------

XBLOCK = 3
YBLOCK = 4
XGRID = 3
YGRID = 4
NBTHETA = 180
NBPHI = 360
listeNBPHOTONS = range(500000000, 1100000000, 100000000)
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
savefig("out/Variation NBLOOP.png", dpi=(140))
figure()

# ----------------------------------------Variation NBPHOTONS normal------------------------------------------------

NBLOOP = 10000
XBLOCK = 4
YBLOCK = 4
XGRID = 4
YGRID = 4
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
savefig("out/Variation NBPHOTONS.png", dpi=(140))
figure()

# ----------------------------------------Variation NBPHOTONS zoom-------------------------------------

NBLOOP = 1000
XBLOCK = 4
YBLOCK = 4
XGRID = 4
YGRID = 4
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
savefig("out/Variation NBPHOTONS zoom.png", dpi=(140))
figure()

# ---------------------------------------NBTHREADS constant----------------------------------------------------

NBTHETA = 180
NBPHI = 360
NBPHOTONS = 100000000
NBLOOP = 1000
XGRID = 2
YGRID = 2
listeXBLOCK = [1,2,3,4,6,8,12,24]
listePlots = []
listeLegends = []

listeGraphe = []
for XBLOCK in listeXBLOCK:
	temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,24/XBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
	listeGraphe.append(temps)
listePlots.append(plot(range(len(listeXBLOCK)),listeGraphe, marker='.'))
listeLegends.append("NBLOOP=1000 XGRID=2")

NBLOOP = 10000
listeGraphe = []
for XBLOCK in listeXBLOCK:
	temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,24/XBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
	listeGraphe.append(temps)
listePlots.append(plot(range(len(listeXBLOCK)),listeGraphe, marker='.'))
listeLegends.append("NBLOOP=10000 XGRID=2")

XGRID = 50
listeGraphe = []
for XBLOCK in listeXBLOCK:
	temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,24/XBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
	listeGraphe.append(temps)
listePlots.append(plot(range(len(listeXBLOCK)),listeGraphe, marker='.'))
listeLegends.append("NBLOOP=10000 XGRID=50")
	
legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps pour (XBLOCK,YBLOCK)=(1,24),(2,12)..(24,1)")
ylabel("TEMPS")
grid(True)
savefig("out/NBTHREADS constant.png", dpi=(140))
figure()

# ---------------------------------------NBBLOCKS constant----------------------------------------------------

NBTHETA = 180
NBPHI = 360
NBPHOTONS = 100000000
NBLOOP = 1000
XBLOCK = 2
YBLOCK = 2
listeXGRID = [1,2,3,4,6,8,12,24]
listePlots = []
listeLegends = []

listeGraphe = []
for XGRID in listeXGRID:
	temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,24/XGRID,NBTHETA,NBPHI)
	listeGraphe.append(temps)
listePlots.append(plot(range(len(listeXGRID)),listeGraphe, marker='.'))
listeLegends.append("NBLOOP=1000 XBLOCK=2")

NBLOOP = 10000
listeGraphe = []
for XGRID in listeXGRID:
	temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,24/XGRID,NBTHETA,NBPHI)
	listeGraphe.append(temps)
listePlots.append(plot(range(len(listeXGRID)),listeGraphe, marker='.'))
listeLegends.append("NBLOOP=10000 XBLOCK=2")

XBLOCK = 50
listeGraphe = []
for XGRID in listeXGRID:
	temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,24/XGRID,NBTHETA,NBPHI)
	listeGraphe.append(temps)
listePlots.append(plot(range(len(listeXGRID)),listeGraphe, marker='.'))
listeLegends.append("NBLOOP=10000 XBLOCK=50")
	
legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps pour (XGRID,YGRID)=(1,24),(2,12)..(24,1)")
ylabel("TEMPS")
grid(True)
savefig("out/NBBLOCKS constant.png", dpi=(140))
figure()

# --------------------------------Variation NBTHREADS NBBLOCKS---------------------------------

NBPHOTONS = 1000000000
NBLOOP = 2000
listeXBLOCK = range(10,141,10)
YBLOCK = 1
listeXGRID = range(20,121,20)
YGRID = 1
NBTHETA = 180
NBPHI = 360
listePlots = []
listeLegends = []
for XGRID in listeXGRID:
	listeGraphe = []
	for XBLOCK in listeXBLOCK:
		temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
		listeGraphe.append(temps)
	listePlots.append(plot(listeXBLOCK,listeGraphe, marker='.'))
	listeLegends.append('NBBLOCKS = ' + str(XGRID))
legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps en fonction du NBTHREADS pour differents NBBLOCKS (NBPHOTONS=1milliard)")
xlabel("NBTHREADS")
ylabel("TEMPS")
grid(True)
savefig("out/Variation NBTHREADS NBBLOCKS.png", dpi=(140))
figure()

# --------------------------------Variation NBTHREADS NBBLOCKS zoom---------------------------------

NBPHOTONS = 1000000000
NBLOOP = 2000
listeXBLOCK = range(40,71,2)
YBLOCK = 1
listeXGRID = range(20,121,20)
YGRID = 1
NBTHETA = 180
NBPHI = 360
listePlots = []
listeLegends = []
for XGRID in listeXGRID:
	listeGraphe = []
	for XBLOCK in listeXBLOCK:
		temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
		listeGraphe.append(temps)
	listePlots.append(plot(listeXBLOCK,listeGraphe, marker='.'))
	listeLegends.append('NBBLOCKS = ' + str(XGRID))
legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps en fonction du NBTHREADS pour differents NBBLOCKS (NBPHOTONS=1milliard)")
xlabel("NBTHREADS")
ylabel("TEMPS")
grid(True)
savefig("out/Variation NBTHREADS NBBLOCKS zoom.png", dpi=(140))
figure()

NBPHOTONS = 10000000
NBLOOP = 50
listeXBLOCK = range(10,181,10)
YBLOCK = 1
listeXGRID = range(20,181,20)
YGRID = 1
NBTHETA = 180
NBPHI = 360
listePlots = []
listeLegends = []
for XGRID in listeXGRID:
	listeGraphe = []
	for XBLOCK in listeXBLOCK:
		temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
		listeGraphe.append(temps)
	listePlots.append(plot(listeXBLOCK,listeGraphe, marker='.'))
	listeLegends.append('NBBLOCKS = ' + str(XGRID))
legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps en fonction du NBTHREADS pour differents NBBLOCKS (NBPHOTONS=10millions)")
xlabel("NBTHREADS")
ylabel("TEMPS")
grid(True)
savefig("out/Variation NBTHREADS NBBLOCKS bis.png", dpi=(140))
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
savefig("out/Variations NBCASES.png", dpi=(140))
figure()

# -------------------------------- Variation NBCASES cible1 ---------------------------------

NBPHOTONS = 100000000
NBLOOP = 1000
XBLOCK = 64
YBLOCK = 1
XGRID = 42
YGRID = 1
listeNBTHETA = range(900,1000,1)
NBPHI = 100
listeAbscisses = []
for NBTHETA in listeNBTHETA:
	listeAbscisses.append(NBTHETA*NBPHI)
listeGraphe = []
for NBTHETA in listeNBTHETA:
	temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
	listeGraphe.append(temps)
plot(listeAbscisses,listeGraphe, marker='.')
title("Temps en fonction du nombre de cases (NBPHOTONS=100millions)")
xlabel("Nombre de cases")
ylabel("TEMPS")
grid(True)
savefig("out/Variations NBCASES cible1.png", dpi=(140))
figure()

# -------------------------------- Variation NBCASES cible2 ---------------------------------

NBPHOTONS = 100000000
NBLOOP = 1000
XBLOCK = 64
YBLOCK = 1
XGRID = 42
YGRID = 1
listeNBTHETA = range(4600,4700,1)
NBPHI = 20
listeAbscisses = []
for NBTHETA in listeNBTHETA:
	listeAbscisses.append(NBTHETA*NBPHI)
listeGraphe = []
for NBTHETA in listeNBTHETA:
	temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
	listeGraphe.append(temps)
plot(listeAbscisses,listeGraphe, marker='.')
title("Temps en fonction du nombre de cases (NBPHOTONS=100millions)")
xlabel("Nombre de cases")
ylabel("TEMPS")
grid(True)
savefig("out/Variations NBCASES cible2.png", dpi=(140))
figure()

# --------------------------------Variation NBBLOCKS---------------------------------

NBPHOTONS = 1000000000
NBLOOP = 2000
listeXBLOCK = [32,64]
YBLOCK = 1
listeXGRID = range(20,121,2)
YGRID = 1
NBTHETA = 180
NBPHI = 360
listePlots = []
listeLegends = []
for XBLOCK in listeXBLOCK:
	listeGraphe = []
	for XGRID in listeXGRID:
		temps = lancerSimulation(listeTemps,NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI)
		listeGraphe.append(temps)
	listePlots.append(plot(listeXGRID,listeGraphe, marker='.'))
	listeLegends.append('NBTHREADS/BLOCK = ' + str(XBLOCK))
legend(listePlots, listeLegends, loc='best', numpoints=1)
title("Temps en fonction du NBBLOCKS pour differents NBTHREADS (NBPHOTONS=1milliard)")
xlabel("NBBLOCKS")
ylabel("TEMPS")
grid(True)
savefig("out/Variation NBBLOCKS.png", dpi=(140))
figure()

# --------------------------------Variation NBTHREADS NBBLOCKS bis---------------------------------

# -------------------------------------------------------------------------------------------

# os.system("shutdown -s -t 0")
