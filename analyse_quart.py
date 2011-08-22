#!/usr/bin/env python
import os
import sys
import pyhdf.SD
from pylab import *

# verification de l'existence du fichier hdf
if os.path.exists("out_prog/Quart.hdf"):
	# on vide le dossier de sortie du script
	os.system("rm -rf out_scripts/analyse_quart")
	os.mkdir("out_scripts/analyse_quart")
	# lecture du fichier hdf
	file_hdf = 'out_prog/Quart.hdf'
	sd_hdf = pyhdf.SD.SD(file_hdf)
	# lecture du nombre de valeurs de phi
	NBPHI = getattr(sd_hdf,'NBPHI')
	NBPHI = NBPHI/2 # car on a somme (phi) et (NBPHI-phi)

	# pour chaque phi on cree un graphique
	for iphi in xrange(NBPHI):
		# lecture du dataset
		name = "Quart (iphi = " + str(iphi) + ")"
		sds_hdf = sd_hdf.select(name)
		# recuperation du tableau et de la valeur de phi
		data = sds_hdf.get()
		phi = getattr(sds_hdf,'phi')
		# creation et sauvegarde du graphique
		plot(data[:,1],data[:,0])
		savefig("out_scripts/analyse_quart/Quart (iphi = "+str(phi)+").png", dpi=(140))
		figure()
else:
	sys.stdout.write("Pas de fichier Quart.hdf\n")


