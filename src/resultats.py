#!/usr/bin/env python

import pyhdf.SD
from pylab import *

# lecture fichier hdf
file_hdf = '/home/tristan/Desktop/TristanCuda/Resultats.hdf'

hdf = pyhdf.SD.SD(file_hdf)

# lecture dataset
for iphi in xrange(10):
	name = "Resultats (iphi = " + str(iphi) + ")"
	data = hdf.select(name).get()
	plot(data[:,1],data[:,0])
	figure()


# affichage datasets
# print hdf.datasets()

#file_bin = '/home/tristan/Desktop/Fortran/bin/out.ran=0001.wav=635.ths=30.000.tr=0.0533.ta=0.0000.pi0=0.967.H=002.000.bin.gz'

# lecture fichier binaire
# data2 = fromfile(file_bin, dtype='float32').reshape((180, 360))
# plot(data2[:,1])
# figure()
# graphe
#plot(data[:,0])
# savefig('figure.png')

# figure()
#plot(data[:,1])

show()
