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
from profil import readREPTRAN

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


def iop(options):
    '''
    calculate the total IOPs of SPM dominated waters (atot,btot and phase function)

    SPM: suspended particulate matter in g/m3
    w: wavelength in nm
    NANG: number of angles for Phase function
    ang_trunc : truncature angle for Fournier Forand Phase function
    gamma: is the spectral dependency of the particulate backscattering
    alpha: parameter for CDOM absorption
    nbp: refractive index of particles (relative to water)
    if reptran option then read the worresponding file to get the wavelengths
    '''
    
    if (options.rep!=None and  options.channel!=None):
        reptran=readREPTRAN(options.rep)
        reptran.selectBand(reptran.Bandname2Band(options.channel)) # selection d'une bande en particulier dans le fichier reptran, lecture du nombre de bandes internes et de l'integrale en nm de la bande
        wl=reptran.awvl
        
    else:
        wl=[options.w]
    
    atotlist=[]
    btotlist=[]    
    iopnamelist=[]
    
    for w in wl:    
        SPM=options.SPM
        NANG=options.NANG
        ang_trunc=options.ang_trunc
        nbp=options.nbp
        gamma=options.gamma
        alpha=options.alpha
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
        ang = pi * arange(NANG, dtype='float64')/(NANG-1)    # angle in radians
    
        # pure water
        pf0 = zeros((NANG, 4), dtype='float64') # pure water phase function
        pf0[:,0] = 0.75
        pf0[:,1] = 0.75 * cos(ang)**2
        pf0[:,2] = 0.75 * cos(ang)
        pf0[:,3] = 0.
    
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
    
        #
        # display results
        #
        outname=install_dir+'fic/pf_ocean-spm_%.2f_%.0fnm.txt'%(SPM,w)
        fo=open(outname,'w')
        fo.write('# SPM concentration: {}\n'.format(SPM))
        fo.write('# wavelength: {}\n'.format(w))
        fo.write('# total absorption coefficient: {}\n'.format(atot))
        fo.write('# total scattering coefficient: {}\n'.format(btot))
        fo.write('# truncating at {} deg\n'.format(ang_trunc))
        for i in xrange(NANG):
            fo.write('{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    ang[i] * 180/pi,
                    pf[i,0],
                    pf[i,1],
                    pf[i,2],
                    pf[i,3],
                    ))
        iopnamelist.append(outname)
        atotlist.append(atot)
        btotlist.append(btot)
    return atotlist,btotlist,iopnamelist
      


if __name__ == '__main__':
    ####################################################################################################################################     
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
    parser.add_option('-N', '--NANG',
                dest='NANG',
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
