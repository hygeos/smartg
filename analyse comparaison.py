#!/usr/bin/env python

import pyhdf.SD
from pylab import *

# lecture fichier hdf
file_hdf = '/home/tristan/Desktop/TristanCuda/out/Comparaison.hdf'

hdf = pyhdf.SD.SD(file_hdf)

# lecture dataset
for iphi in xrange(5):
	name = "Comparaison (iphi = " + str(iphi) + ")"
	data = hdf.select(name).get()
	plot(data[:,0],data[:,1])
	plot(data[:,0],data[:,2])
	figure()
show()
