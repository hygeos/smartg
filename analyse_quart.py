#!/usr/bin/env python

import os
import sys
import pyhdf.SD
from pylab import *

if os.path.exists("out_prog/Quart.hdf"):
	os.system("rm -rf out_scripts/analyse_quart")
	os.mkdir("out_scripts/analyse_quart")

	# lecture fichier hdf
	file_hdf = 'out_prog/Quart.hdf'

	sd_hdf = pyhdf.SD.SD(file_hdf)
	
	# lecture du nombre de valeurs de phi
	NBPHI = getattr(sd_hdf,'NBPHI')
	NBPHI = NBPHI/2

	# lecture dataset
	for iphi in xrange(NBPHI):
		name = "Quart (iphi = " + str(iphi) + ")"
		sds_hdf = sd_hdf.select(name)
		data = sds_hdf.get()
		phi = getattr(sds_hdf,'phi')
		plot(data[:,1],data[:,0])
		savefig("out_scripts/analyse_quart/Quart (iphi = "+str(phi)+").png", dpi=(140))
		figure()
else:
	sys.stdout.write("Pas de fichier Quart.hdf\n")


