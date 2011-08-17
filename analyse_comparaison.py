#!/usr/bin/env python

import os
import sys
import pyhdf.SD
from pylab import *

if os.path.exists("out_prog/Comparaison.hdf"):
	os.system("rm -rf out_scripts/analyse_comparaison")
	os.mkdir("out_scripts/analyse_comparaison")

	# lecture fichier hdf
	file_hdf = 'out_prog/Comparaison.hdf'

	hdf = pyhdf.SD.SD(file_hdf)

	# lecture dataset
	for iphi in xrange(5):
		name = "Comparaison (iphi = " + str(iphi) + ")"
		data = hdf.select(name).get()
		plot(data[:,0],data[:,1])
		plot(data[:,0],data[:,2])
		savefig("out_scripts/analyse_comparaison/"+name+".png", dpi=(140))
		figure()
else:
	sys.stdout.write("Pas de fichier Comparaison.hdf\n")

