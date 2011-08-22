#!/usr/bin/env python
import os
import sys
import pyhdf.SD
from pylab import *

# verification de l'existence du fichier hdf
if os.path.exists("out_prog/Resultats.hdf"):
	# on vide le dossier de sortie du script
	os.system("rm -rf out_scripts/analyse_resultats")
	os.mkdir("out_scripts/analyse_resultats")
	# lecture du fichier hdf
	file_hdf = 'out_prog/Resultats.hdf'
	sd_hdf = pyhdf.SD.SD(file_hdf)
	# lecture du nombre de valeurs de phi
	NBPHI = getattr(sd_hdf,'NBPHI')

	# pour chaque phi on cree un graphique
	for iphi in xrange(NBPHI):
		# lecture du dataset
		name = "Resultats (iphi = " + str(iphi) + ")"
		sds_hdf = sd_hdf.select(name)
		# recuperation du tableau et de la valeur de phi
		data = sds_hdf.get()
		phi = getattr(sds_hdf,'phi')
		# creation et sauvegarde du graphique
		plot(data[:,1],data[:,0])
		title("Eclairement en fonction de theta pour phi="+str(phi))
		xlabel("Theta (rad)")
		ylabel("Eclairement")
		grid(True)
		savefig("out_scripts/analyse_resultats/analyse_resultats_iphi="+str(phi)+".png", dpi=(140))
		figure()
else:
	sys.stdout.write("Pas de fichier Resultats.hdf\n")

