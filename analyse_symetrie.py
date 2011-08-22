#!/usr/bin/env python
import os
import sys
import pyhdf.SD
from pylab import *

# verification de l'existence du fichier hdf
if os.path.exists("out_prog/Comparaison.hdf"):
	# on vide le dossier de sortie du script
	os.system("rm -rf out_scripts/analyse_symetrie")
	os.mkdir("out_scripts/analyse_symetrie")
	# lecture du fichier hdf
	file_hdf = 'out_prog/Comparaison.hdf'
	sd_hdf = pyhdf.SD.SD(file_hdf)
	# lecture du nombre de valeurs de phi
	NBPHI = getattr(sd_hdf,'NBPHI')
	NBPHI = NBPHI/2 # car on compare (phi) et (NBPHI-phi)
	
	# pour chaque phi on cree un graphique
	for iphi in xrange(NBPHI):
		# lecture du dataset
		name = "Comparaison (iphi = " + str(iphi) + ")"
		sds_hdf = sd_hdf.select(name)
		# recuperation du tableau et de la valeur de phi
		data = sds_hdf.get()
		phi = getattr(sds_hdf,'phi')
		# creation et sauvegarde du graphique
		listePlots = []
		listeLegends = []
		listePlots.append(plot(data[:,0],data[:,1]))
		listeLegends.append('phi')
		listePlots.append(plot(data[:,0],data[:,2]))
		listeLegends.append('2PI-phi')
		legend(listePlots, listeLegends, loc='best', numpoints=1)
		title("Comparaison entre les graphes symetriques")
		xlabel("Theta (rad)")
		ylabel("Eclairement")
		grid(True)
		savefig("out_scripts/analyse_symetrie/analyse_symetrie_phi="+str(phi)+".png", dpi=(140))
		figure()
else:
	sys.stdout.write("Pas de fichier Comparaison.hdf\n")

