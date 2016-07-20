#!/usr/bin/env python
# vim:fileencoding=utf-8

'''
Calculate the ocean water absorption, scattering coefficients and phase
function using the wavelength and the SPM concentration
'''

from numpy import sin, cos, pi, exp
from numpy import arange, zeros, log10
from seawater_scattering_Zhang2009 import swscat
from water_absorption import a_w
from phase_functions import fournierForand, PhaseFunction, fournierForandB
from iop import IOP



class IOP_SPM(IOP):
    '''
    Initialize the IOP model (SPM dominated waters)

    Arguments:
        SPM: suspended particulate matter in g/m3
        NANG: number of angles for Phase function
        ang_trunc : truncature angle for Fournier Forand Phase function
        gamma: is the spectral dependency of the particulate backscattering
        alpha: parameter for CDOM absorption
        nbp: refractive index of particles (relative to water)
        ALB: albedo of the sea floor
        DEPTH: depth in m , default None = Semi infinite
        pfwav: list of arrays at which the phase functions are calculated
    '''
    def __init__(self, SPM, NANG=72001, ang_trunc=5., gamma=0.5,
            alpha=1., nbp=1.15, ALB=0., DEPTH=None, pfwav=None, verbose=False):
        self.__SPM = SPM
        self.__NANG = NANG
        self.__ang_trunc = ang_trunc
        self.__gamma = gamma
        self.__alpha = alpha
        self.__nbp = nbp
        self.alb = ALB
        self.depth = DEPTH
        self.pfwav=pfwav
        self.verbose = verbose
        self.last = None   # stores last result (managed by parent class)


    def calc(self, w, skip_phase=False):
        '''
        Calculate atot, btot and phase function (monochromatic)

        Arguments:
            w: wavelength in nm

        Returns (atot, btot, phase) where:
            * atot is the total absorption coefficient
            * btot is the total scattering coefficient
            * phase is the PhaseFunction object
        '''

        SPM = self.__SPM
        NANG = self.__NANG
        ang_trunc = self.__ang_trunc
        nbp = self.__nbp
        gamma = self.__gamma
        alpha = self.__alpha

        # pure sea water scattering 
        bw = swscat(w)

        # particulate backscattering
        bbp650 = 10**(1.03*log10(SPM) - 2.06) # Neukermans et al 2012
        bbp = bbp650*(w/650.)**(-gamma)

        # CDM absorption
        aCDM = alpha*0.031*SPM*exp(-0.0123*(w-443.))

        # pure sea water absorption
        aw = a_w(w)

        # Backscattering ratio of particles (non-troncated)
        Bp = fournierForandB(nbp, 3.+gamma)
        # Scattering coefficient of particles
        bp   = bbp/Bp


        #
        # total absorption
        atot = aCDM + aw

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


