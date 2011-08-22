#!/usr/bin/env python
import os
import sys
import pyhdf.SD
from pylab import *

# verification de l'existence du fichier hdf
if os.path.exists("out_prog/Resultats.hdf"):
	# on vide le dossier de sortie du script
	os.system("rm -rf out_scripts/analyse_symetrie")
	os.mkdir("out_scripts/analyse_symetrie")
	# lecture du fichier hdf
	file_hdf = 'out_prog/Resultats.hdf'
	sd_hdf = pyhdf.SD.SD(file_hdf)
	# lecture du nombre de valeurs de phi
	NBPHI = getattr(sd_hdf,'NBPHI')

	# pour chaque iphi de 0..NBPHI/2-1 on cree un graphique
	for iphi in xrange(NBPHI/2):
		# lecture du premier dataset (iphi)
		name_1 = "Resultats (iphi = " + str(iphi) + ")"
		sds_hdf_1 = sd_hdf.select(name_1)
		# recuperation du premier tableau et de la valeur de phi
		data_1 = sds_hdf_1.get()
		phi = getattr(sds_hdf_1,'phi')
		# lecture du deuxieme dataset (NBPHI-iphi-1)
		name_2 = "Resultats (iphi = " + str(NBPHI-iphi-1) + ")"
		sds_hdf_2 = sd_hdf.select(name_2)
		# recuperation du deuxieme tableau
		data_2 = sds_hdf_2.get()
		# creation et sauvegarde du graphique
		listePlots = []
		listeLegends = []
		listePlots.append(plot(data_1[:,1],data_1[:,0]))
		listeLegends.append('phi')
		listePlots.append(plot(data_2[:,1],data_2[:,0]))
		listeLegends.append('2PI-phi')
		legend(listePlots, listeLegends, loc='best', numpoints=1)
		title("Comparaison entre les graphes symetriques pour phi="+str(phi))
		xlabel("Theta (rad)")
		ylabel("Eclairement")
		grid(True)
		savefig("out_scripts/analyse_symetrie/analyse_symetrie_phi="+str(phi)+".png", dpi=(140))
		figure()
else:
	sys.stdout.write("Pas de fichier Resultats.hdf\n")

