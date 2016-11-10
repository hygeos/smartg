#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tools.luts import MLUT
import numpy as np



class IOP_base(object):
    pass

class IOP_Rw(IOP_base):
    def __init__(self, ALB):
        '''
        Defines a model of water reflectance (lambertian under the surface)

        ALB: albedo object of the lambertian reflector
        '''
        self.ALB = ALB

    def calc(self, wav):
        '''
        Profile and phase function calculation at bands wav (nm)
        '''
        pro = MLUT()
        pro.add_axis('wavelength', wav)
        pro.add_axis('z', np.arange(2))
        shp = (len(wav), 2)

        pro.add_dataset('tau_tot', np.zeros(shp, dtype='float32'),
                        ['wavelength', 'z'])
        pro.add_dataset('tau_sca', np.zeros(shp, dtype='float32'),
                        ['wavelength', 'z'])
        pro.add_dataset('tau_abs', np.zeros(shp, dtype='float32'),
                        ['wavelength', 'z'])
        pro.add_dataset('pmol', np.zeros(shp, dtype='float32'),
                        ['wavelength', 'z'])
        pro.add_dataset('ssa', np.ones(shp, dtype='float32'),
                        ['wavelength', 'z'])
        pro.add_dataset('pabs', np.zeros(shp, dtype='float32'),
                        ['wavelength', 'z'])
        pro.add_dataset('albedo_seafloor',
                        self.ALB.get(wav), ['wavelength'])

        return pro


class IOP_MM(IOP_base):
    '''
    IOP model after Morel and Maritorena (2001)
    '''
    def __init__(self):
        raise NotImplementedError

    def calc(self, wav):
        pro = MLUT()

        return pro

