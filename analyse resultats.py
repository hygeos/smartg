#!/usr/bin/env python

import pyhdf.SD
from pylab import *

# lecture fichier hdf
file_hdf = '/home/tristan/Desktop/TristanCuda/out/Resultats.hdf'

hdf = pyhdf.SD.SD(file_hdf)

# lecture dataset
for iphi in xrange(10):
	name = "Resultats (iphi = " + str(iphi) + ")"
	data = hdf.select(name).get()
	plot(data[:,1],data[:,0])
	figure()

show()
