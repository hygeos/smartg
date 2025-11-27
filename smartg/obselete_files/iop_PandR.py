#!/usr/bin/env python
# vim:fileencoding=utf-8

'''
Calculate the ocean water absorption, scattering coefficients and phase
function using the wavelength and the SPM concentration
'''

from numpy import sin, cos, pi, exp
from numpy import arange, zeros, log10, array
from phase_functions import fournierForand, PhaseFunction, fournierForandB
from iop import IOP
import sys
sys.path.append('/home/did/RTC/SMART-G/tools/water/cython')
from waterPR import ParkRuddick

rep='/home/francois/MERIS/polymer_py/auxdata/common/'


class IOP_PandR(IOP):
    '''
    Initialize the IOP model (Chl dominated waters)

    Arguments:
        CHL: Chlorophyll mg/m3
        NANG: number of angles for Phase function
        ang_trunc : truncature angle for Fournier Forand Phase function
        ALB: albedo of the sea floor
        DEPTH: depth in m , default None = Semi infinite
        pfwav: list of arrays at which the phase functions are calculated
    '''
    def __init__(self, CHL, NANG=72001, ang_trunc=5., ALB=0., DEPTH=None, pfwav=None, verbose=False):
        self.__CHL = CHL
        self.__NANG = NANG
        self.__ang_trunc = ang_trunc
        self.alb = ALB
        self.depth = DEPTH
        self.pfwav=pfwav
        self.verbose = verbose
        self.last = None   # stores last result (managed by parent class)


    def calc(self, w, skip_phase=False, nbp=1.15):
        '''
        Calculate atot, btot and phase function (monochromatic)

        Arguments:
            w: wavelength in nm

        Returns (atot, btot, phase) where:
            * atot is the total absorption coefficient
            * btot is the total scattering coefficient
            * phase is the PhaseFunction object
        '''

        PR = ParkRuddick(rep, debug=True, alt_gamma_bb=True)
        CHL = self.__CHL
        NANG = self.__NANG
        ang_trunc = self.__ang_trunc
        PR.calc(w, log10(CHL))
        self.iops = PR.iops()

        # pure sea water scattering 
        bw = self.iops['bw']

        # Scattering coefficient of particles
        btot = self.iops['btot']
        bp = btot - bw

        #
        # total absorption
        atot = self.iops['atot']

        if skip_phase:
            P0, P1 = None, None
        else:
            #
            # phase function
            #
            ang = pi * arange(NANG, dtype='float64')/(NANG-1)    # angle in radians

            # pure water
            pf0 = zeros((NANG, 4), dtype='float64') # pure water phase function
            pf0[:,1] = 0.75
            pf0[:,0] = 0.75 * cos(ang)**2
            pf0[:,2] = 0.75 * cos(ang)
            pf0[:,3] = 0.
            P0 = PhaseFunction(ang, 2*pf0, degrees=False)

            # particles (troncature)
            # particle slope of the scattering spectral dependency
            gamma = self.iops['gamma']
            itronc = int(NANG * ang_trunc/180.)
            pf1 = zeros((NANG, 4), dtype='float64') # pure water phase function
            # assuming that the slope of Junge power law mu and slope of spectral dependence of scattering is mu=3+gamma
            pf1[itronc:,0] = 0.5*fournierForand(ang[itronc:],nbp,3.+gamma)
            pf1[:itronc,0] = 0.5*fournierForand(ang[itronc ],nbp,3.+gamma) 
            pf1[:,1] = pf1[:,0]
            pf1[:,2] = 0.
            pf1[:,3] = 0.

            # normalization after truncation
            integ_ff = 0.
            integ_ff_back = 0.
            for iang in xrange(1, NANG):
                dtheta = ang[iang] - ang[iang-1]
                pm1 = pf1[iang-1,0] + pf1[iang-1,1]
                pm2 = pf1[iang,0] + pf1[iang,1]
                sin1 = sin(ang[iang-1])
                sin2 = sin(ang[iang])
                integ_ff += dtheta*((sin1*pm1+sin2*pm2)/3. + (sin1*pm2+sin2*pm1)/6.)
                if ang[iang]>pi/2. :
                    integ_ff_back += dtheta*((sin1*pm1+sin2*pm2)/3. + (sin1*pm2+sin2*pm1)/6.)
            rat1 = integ_ff/2.
            pf1 *= 1/rat1
            P1 = PhaseFunction(ang, 2*pf1, degrees=False, coef_trunc=rat1)

        return atot, [(bw, P0), (bp, P1)]

    def __str__(self):
        return 'SPM={}'.format(self.__SPM)
