#!/usr/bin/env python

import pyhdf.SD
from pylab import *

# lecture fichier hdf
file_hdf = '/home/tristan/Desktop/TristanCuda/out/Quart.hdf'

hdf = pyhdf.SD.SD(file_hdf)

# lecture dataset
for iphi in xrange(5):
	name = "Quart (iphi = " + str(iphi) + ")"
	data = hdf.select(name).get()
	plot(data[:,1],data[:,0])
	figure()
show()

