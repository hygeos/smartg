#!/usr/bin/env python
# vim:fileencoding=utf-8


'''
Calculate the ocean water absorption, scattering coefficients and phase
function using the wavelength and the chlorophyll concentration
'''

from phase_functions import PhaseFunction
from iop import IOP
from numpy import cos, pi, array
from numpy import arange, zeros
import numpy as np


wl_1 = array([350.0525, 450.0666, 550.084, 650.099])

ablk = array([0.0204, 0.0092, 0.3400, 0.6240])

wblk = array([0.3964, 0.3285, 0.0325, 0.0029])


class IOP_AOS_WATER(IOP):
    '''

    Parameters:
        NANG: number if angles for the phase function
        ang_trunc: truncation angle in degrees
        ALB: albedo of the sea floor
        pfwav: list of arrays at which the phase functions are calculated
    '''
    def __init__(self, NANG=72001, ALB=0., pfwav=None, verbose=False):
        self.NANG = NANG
        self.alb = ALB
        self.pfwav=pfwav
        self.verbose = verbose
        self.last = None   # last parameters (wl, profil, phases) , to avoid
                           # reprocessing wavelengths that have been calxculated already

    def calc(self, wl, skip_phase=False):
        '''
        Calculate atot, btot and phase function (monochromatic)

        Arguments:
            wl: wavelength in nm

        Returns (atot, [(b0, P0), (b1, P1), ...]) where:
            * atot is the total absorption coefficient
            * bi is the scattering coefficient (without truncation) of the ith component
            * Pi is the phase function of the ith component
        '''
        NANG = self.NANG

        #
        # wavelength index
        #
        if (wl < wl_1[0]) or (wl > wl_1[-1]):
            raise Exception('Error, wavelength {} is out of range ({}, {})'.format(wl, wl_1[0], wl_1[-1]))
        i1 = int((wl - wl_1[0])/(wl_1[1] - wl_1[0]))

        #
        # pure water coefficients
        #
        a0 = ablk[i1]
        b0 = a0*wblk[i1]/(1.-wblk[i1])
        #
        # phase function
        #
        if skip_phase:
            P0 = None
            P1 = None
        else:
            ang = pi * arange(NANG, dtype='float64')/(NANG-1)    # angle in radians
            # pure water
            pf0 = zeros((NANG, 4), dtype='float64') # pure water phase function with 0 depol factor
            pf1 = zeros((NANG, 4), dtype='float64')
            pf0[:,0] = 0.75
            pf0[:,1] = 0.75 * cos(ang)**2
            pf0[:,2] = 0.75 * cos(ang)
            pf0[:,3] = 0.
            P0 = PhaseFunction(ang, pf0, degrees=False)
            P1 = PhaseFunction(ang, pf1, degrees=False)


        return a0, [(b0, P0), (0., P1)]

    def __str__(self):
        return 'CHL={}'.format(self.chl)



if __name__ == '__main__':

    mod = IOP_AOS_WATER(pfwav=[350., 450., 550., 650.])
    # print mod.calc(500.)
    print mod.calc_bands(np.linspace(350, 650, 4))
