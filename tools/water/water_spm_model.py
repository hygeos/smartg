#!/usr/bin/env python
# vim:fileencoding=utf-8

'''
Calculate the ocean water absorption, scattering coefficients and phase
function using the wavelength and the SPM concentration
'''
import sys
sys.path.append("profile water")
from numpy import sin, cos, pi, exp
from numpy import arange, zeros, log10
from seawater_scattering_Zhang2009 import swscat
from water_absorption import a_w
from optparse import OptionParser
from os.path import exists, join, dirname
from os import makedirs

install_dir='/home/did/RTC/SMART-G/'

def fournierForand(ang, n, mu):
    '''
    Fournier-Forand phase function
    n : refractive index
    mu : slope of the Junge size distribution
    '''
    v = (3-mu)/2
    delta = 4/( 3*(n-1)*(n-1) )*sin(ang/2)*sin(ang/2)
    delta180 = 4/( 3*(n-1)*(n-1) )*sin(pi/2)*sin(pi/2)

    res = 1/( 4*pi*(1-delta)*(1-delta)*(delta**v) )*( v*(1-delta) - (1-(delta**v)) + ( delta*(1-(delta**v)) - v*(1-delta) )*1/(sin(ang/2)*sin(ang/2)) ) + (1-(delta180**v))/(16*pi*(delta180-1)*(delta180**v)) * (3*cos(ang)*cos(ang) - 1)
    res *= 4*pi

    return res


class WaterModelSPM(object):
    '''
    Initialize the Model (SPM dominated waters)

    Arguments:
        SPM: suspended particulate matter in g/m3
        NSCOCE: number of angles for Phase function
        ang_trunc : truncature angle for Fournier Forand Phase function
        gamma: is the spectral dependency of the particulate backscattering
        alpha: parameter for CDOM absorption
        nbp: refractive index of particles (relative to water)
    '''
    def __init__(self, SPM, NSCOCE=72001, ang_trunc=5., gamma=0.5, alpha=1., nbp=1.15):
        self.__SPM = SPM
        self.__NSCOCE = NSCOCE
        self.__ang_trunc = ang_trunc
        self.__gamma = gamma
        self.__alpha = alpha
        self.__nbp = nbp

    def __name_phase(self, directory, wl):
        '''
        name the phase function file
        '''
        return join(directory, 'fic/pf_ocean_spm_%.2f_%.0fnm.txt'%(self.__SPM, wl))

    def calc(self, w, dir_phase):
        '''
        Calculate the total IOPs (atot, btot and phase function)

        Arguments:
            w: wavelength in nm
            dir_phase: directory for storing the phase function
        '''
        SPM = self.__SPM
        NSCOCE = self.__NSCOCE
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

        #
        # phase function
        #
        ang = pi * arange(NSCOCE, dtype='float64')/(NSCOCE-1)    # angle in radians

        # pure water
        pf0 = zeros((NSCOCE, 4), dtype='float64') # pure water phase function
        pf0[:,0] = 0.75
        pf0[:,1] = 0.75 * cos(ang)**2
        pf0[:,2] = 0.75 * cos(ang)
        pf0[:,3] = 0.

        # particles (troncature)
        itronc = int(NSCOCE * ang_trunc/180.)
        pf1 = zeros((NSCOCE, 4), dtype='float64') # pure water phase function
        # assuming that the slope of Junge power law mu and slope of spectral dependence of scattering is mu=3+gamma
        pf1[itronc:,0] = 0.5*fournierForand(ang[itronc:],nbp,3.+gamma)
        pf1[:itronc,0] = 0.5*fournierForand(ang[itronc ],nbp,3.+gamma) 
        pf1[:,1] = pf1[:,0]
        pf1[:,2] = 0.
        pf1[:,3] = 0.

        # normalization after truncation
        integ_ff = 0.
        integ_ff_back = 0.
        for iang in xrange(1, NSCOCE):
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
        bbp *= rat1

        # Backscattering ratio of particles
        Bp   = integ_ff_back/integ_ff 
        # Scattering coefficient of particles
        bp   = bbp/Bp

        #
        # total absorption
        atot = aCDM + aw

        # total scattering
        btot = bw + bp

        # total scattering function
        pf = (bw*pf0 + bp*pf1)/btot

        # write phase function
        file_phase = self.__name_phase(dir_phase, w)
        if not exists(dirname(file_phase)):
            makedirs(dirname(file_phase))
        if not exists(file_phase):
            fo = open(file_phase, 'w')
            fo.write('# SPM concentration: {}\n'.format(SPM))
            fo.write('# wavelength: {}\n'.format(w))
            fo.write('# total absorption coefficient: {}\n'.format(atot))
            fo.write('# total scattering coefficient: {}\n'.format(btot))
            fo.write('# truncating at {} deg\n'.format(ang_trunc))
            for i in xrange(NSCOCE):
                fo.write('{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        ang[i] * 180/pi,
                        pf[i,0],
                        pf[i,1],
                        pf[i,2],
                        pf[i,3],
                        ))
            fo.close()
        else:
            print '{} exists'.format(file_phase)

        return atot, btot, file_phase
    
    def __str__(self):
        return 'SPM={}'.format(self.__SPM)


if __name__ == '__main__':
    ####################################################################################################################################      # FIXME
    parser = OptionParser(usage='%prog [options] Type %prog -h for help\n')
    parser.add_option('-S','--SPM',
                dest='SPM',
                type='float',
                default=1.,
                help='SPM in mg/l'
                )
    parser.add_option('-w', '--wavel',
                dest='w',
                type='float',
                default=550.,
                help='wavelength (nm), default 550 nm' 
                )
    parser.add_option('-N', '--NSCOCE',
                dest='NSCOCE',
                type='int',
                default=72001,
                help='Number of output angles of the phase function' 
                )
    parser.add_option('-t', '--trunc',
                dest='ang_trunc',
                type='float',
                default=5.,
                help='Truncation angle of the phase function' 
                )
    parser.add_option('-g', '--gamma',
                dest='gamma',
                type='float',
                default=0.5,
                help='spectral slope of the particles scattering coefficient'
                )
    parser.add_option('-a', '--alpha',
                dest='alpha',
                type='float',
                default=1.0,
                help='parameter for CDOM absorption'
                )
    parser.add_option('-n', '--nbp',
                dest='nbp',
                type='float',
                default=1.15,
                help='refractive index of particles (relative to water)'
                )
    parser.add_option('-R', '--REPTRAN',
                dest='rep',
                type='string',
                help='REPTRAN molecular absorption file' 
                )
    parser.add_option('-C', '--CHANNEL',
                dest='channel',
                type='string',
                help='Sensor channel name (use with REPTRAN)' 
                )
                
    (options, args) = parser.parse_args()
    if len(args) != 2 :
        parser.print_usage()
        exit(1)
    iop(options)
