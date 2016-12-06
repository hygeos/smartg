#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, division
import numpy as np
from collections import Iterable

class BandSet(object):
    def __init__(self, wav):
        '''
        Common objet for formatting input bands definition

        Valid inputs:
            * float
            * 1-d array
            * KDIS or REPTRAN IBANDS LIST

        Methods:
            __getitem__: returns wavelength
        '''
        try:
            self.use_reptran_kdis = hasattr(wav[0], 'calc_profile')
        except:
            self.use_reptran_kdis = False

        if self.use_reptran_kdis:
            self.wav = [x.w for x in wav]
            self.data = wav
        else:
            self.wav = wav
            self.data = None

        assert isinstance(self.wav, (float, list, np.ndarray))
        self.wav = np.array(self.wav, dtype='float32')
        self.scalar = (self.wav.ndim == 0)
        if self.scalar:
            self.wav = self.wav.reshape(1)
        self.size = self.wav.size

    def __getitem__(self, key):
        return self.wav[key]

    def __len__(self):
        return self.size

    def calc_profile(self, prof):
        '''
        calculate the absorption profile for each band
        '''
        tau_mol = np.zeros((self.size, len(prof.z)), dtype='float32')

        if self.use_reptran_kdis:
            for i, w in enumerate(self.data):
                tau_mol[i,:] = w.calc_profile(prof)

        return tau_mol




