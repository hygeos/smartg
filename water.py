#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tools.luts import MLUT, LUT, Idx
import numpy as np
from warnings import warn
from albedo import Albedo_cst
from os.path import realpath, dirname, join
from atmosphere import NPSTK
from tools.phase import fournierForand, integ_phase, calc_iphase

this_dir = dirname(realpath(__file__))
dir_aux = join(this_dir, 'auxdata')



def read_aw(dir_aux):
    '''
    Read pure water absorption from pope&fry, 97 (<700nm)
    and palmer&williams, 74 (>700nm)
    '''

    # Pope&Fry
    data_pf = np.genfromtxt(join(dir_aux, 'water', 'pope97.dat'), skip_header=6)
    aw_pf = data_pf[:,1] * 100 #  convert from cm-1 to m-1
    lam_pf = data_pf[:,0]
    ok_pf = lam_pf <= 700

    # Palmer&Williams
    data_pw = np.genfromtxt(join(dir_aux, 'water', 'palmer74.dat'), skip_header=5)
    aw_pw = data_pw[::-1,1] * 100 #  convert from cm-1 to m-1
    lam_pw = data_pw[::-1,0]
    ok_pw = lam_pw > 700

    aw = LUT(
            np.array(list(aw_pf[ok_pf]) + list(aw_pw[ok_pw])),
            axes=[np.array(list(lam_pf[ok_pf]) + list(lam_pw[ok_pw]))]
            )

    return aw


class IOP_base(object):
    pass

class IOP(IOP_base):
    '''
    Profile of water properties defined by:
        * phase: phase functions
          LUT with dimensions [nwav, nz, stk, angle]
        * bp, bw: particle and water scattering coefficient in m-1
          numpy arrays, dimensions [nwav, nZ]
          by default, bw takes default values
        * atot, ap, aw: total, particle and water absorption coefficient in m-1
          numpy array, dimensions [nwav, nZ]
          either provide atot or ap
          if ap is provided, aw can be provided as well, otherwise it takes default values
        * Z: profile of depths in m
        * ALB: seafloor albedo (albedo object)

        NOTE: first item in dimension Z is not used
    '''
    def __init__(self, phase, bp=None, bw=None,
                 atot=None, ap=None, aw=None,
                 Z=[0, 10000], ALB=Albedo_cst(0.)):

        self.Z = np.array(Z, dtype='float')
        self.bp = bp
        self.bw = bw
        self.atot = atot
        self.ap = ap
        self.aw = aw
        self.phase = phase
        self.ALB = ALB

    def calc(self, wav):

        # absorption
        atot = self.atot
        if atot is None:
            if self.aw is None:
                aw = read_aw(dir_aux)
                self.aw = aw[Idx(wav)]
            atot = self.aw + self.ap


        # scattering
        bp = self.bp

        bw = self.bw
        if bw is None:
            bw = 19.3e-4*((wav[:]/550.)**-4.3)
            bw = bw[:,None]  # add a dimension Z for broadcasting

        btot = bw + bp

        pro = MLUT()

        pro.add_axis('wavelength', wav[:])
        pro.add_axis('z_oc', -self.Z)

        if self.phase is not None:

            pha = self.phase
            pha_, ipha = calc_iphase(pha, pro.axis('wavelength'), pro.axis('z_oc'))

            pro.add_axis('theta_oc', pha.axis('theta_oc'))
            pro.add_dataset('phase_oc', pha_, ['iphase', 'stk', 'theta_oc'])
            pro.add_dataset('iphase_oc', ipha, ['wavelength', 'z_oc'])

        dz = np.diff(self.Z)

        tau_tot = - (atot + btot)*dz
        pro.add_dataset('OD_oc', tau_tot,
                        ['wavelength', 'z_oc'])

        tau_sca = - btot*dz
        pro.add_dataset('OD_sca_oc', tau_sca,
                        ['wavelength', 'z_oc'])

        tau_abs = - (atot * dz)
        pro.add_dataset('OD_abs_oc', tau_abs,
                        ['wavelength', 'z_oc'])

        pmol = bp/btot
        pro.add_dataset('pmol_oc', pmol,
                        ['wavelength', 'z_oc'])

        with np.errstate(invalid='ignore'):
            ssa = tau_sca/tau_tot

        ssa[np.isnan(ssa)] = 1.

        pro.add_dataset('ssa_oc', ssa,
                        ['wavelength', 'z_oc'])

        pro.add_dataset('albedo_seafloor',
                        self.ALB.get(wav),
                        ['wavelength'])

        return pro





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
        pro.add_axis('wavelength', wav[:])
        pro.add_axis('z_oc', np.arange(2))
        shp = (len(wav), 2)

        pro.add_dataset('OD_oc', np.zeros(shp, dtype='float32'),
                        ['wavelength', 'z_oc'])
        pro.add_dataset('OD_sca_oc', np.zeros(shp, dtype='float32'),
                        ['wavelength', 'z_oc'])
        pro.add_dataset('OD_abs_oc', np.zeros(shp, dtype='float32'),
                        ['wavelength', 'z_oc'])
        pro.add_dataset('pmol_oc', np.zeros(shp, dtype='float32'),
                        ['wavelength', 'z_oc'])
        pro.add_dataset('ssa_oc', np.ones(shp, dtype='float32'),
                        ['wavelength', 'z_oc'])
        pro.add_dataset('albedo_seafloor',
                        self.ALB.get(wav), ['wavelength'])

        return pro


class IOP_1(IOP_base):
    '''
    IOP model
    using similar IOP parameterization as Polymer's PR model

    Parameters:
        chl: chlorophyll concentration in mg/m3
        NANG: number of angles for the phase function
        ang_trunc: truncation angle in degrees
        ALB: albedo of the sea floor (Albedo instance)
        DEPTH: depth in m
        pfwav: list of wavelengths at which the phase functions are calculated
    '''
    def __init__(self, chl, pfwav=None, ALB=Albedo_cst(0.),
                 DEPTH=10000, NANG=72001, ang_trunc=5.):
        self.chl = chl
        self.depth = float(DEPTH)
        self.NANG = NANG
        self.ang_trunc = ang_trunc
        self.ALB = ALB
        if pfwav is None:
            self.pfwav = None
        else:
            self.pfwav = np.array(pfwav)

        #
        # read pure water absorption coefficient
        #
        self.aw = read_aw(dir_aux)

        # Bricaud (98)
        ap_bricaud = np.genfromtxt(join(dir_aux, 'water', 'aph_bricaud_1998.txt'),
                                   delimiter=',', skip_header=12)  # header is lambda,Ap,Ep,Aphi,Ephi
        self.BRICAUD = MLUT()
        self.BRICAUD.add_axis('wav', ap_bricaud[:,0])
        self.BRICAUD.add_dataset('A', ap_bricaud[:,1], axnames=['wav'])
        self.BRICAUD.add_dataset('E', 1-ap_bricaud[:,2], axnames=['wav'])


    def calc_iop(self, wav, coef_trunc=2.):
        '''
        inherent optical properties calculation

        wav: wavelength in nm
        coef_trunc: integrate (normalized to 2) of the truncated phase function
        -> scattering coefficient has to be multiplied by coef_trunc/2.
        '''
        chl = self.chl

        # pure water absorption
        aw = self.aw[Idx(wav[:])]

        # phytoplankton absorption
        aphy = (self.BRICAUD['A'][Idx(wav, fill_value='extrema')]
                * (chl**self.BRICAUD['E'][Idx(wav, fill_value='extrema')]))

        # CDM absorption central value
        # from Bricaud et al GBC, 2012 (data from nov 2007)
        fa = 1.
        aCDM443 = fa * 0.069 * (chl**1.070)

        S = 0.00262*(aCDM443**(-0.448))
        if (S > 0.025): S=0.025
        if (S < 0.011): S=0.011

        aCDM = aCDM443 * np.exp(-S*(wav - 443))

        atot = aw + aphy + aCDM

        # Pure Sea water scattering coefficient
        bw = 19.3e-4*((wav/550.)**-4.3)

        bp = 0.416*(chl**0.766)*550./wav

        bp *= coef_trunc/2.

        btot = bw + bp

        #
        # backscattering coefficient
        #
        if chl < 2:
            v = 0.5*(np.log10(chl) - 0.3)
        else:
            v = 0
        Bp = 0.002 + 0.01*( 0.5-0.25*np.log10(chl))*((wav/550.)**v)

        return {'atot': atot,
                'aphy': aphy,
                'aw': aw,
                'bw': bw,
                'btot': btot,
                'bp': bp,
                'Bp': Bp,
                }


    def calc(self, wav, phase=True):
        '''
        Profile and phase function calculation

        wav: wavelength in nm
        '''
        wav = np.array(wav)
        pro = MLUT()
        pro.add_axis('wavelength', wav[:])
        pro.add_axis('z_oc', np.array([0., -self.depth]))

        if phase:
            if self.pfwav is None:
                wav_pha = wav
            else:
                wav_pha = self.pfwav
            Bp = self.calc_iop(wav_pha)['Bp']
            pha = self.phase(wav_pha, Bp)

            pha_, ipha = calc_iphase(pha['phase'], pro.axis('wavelength'), pro.axis('z_oc'))

            # index with ipha and reshape to broadcast to [wav, z]
            coef_trunc = pha['coef_trunc'].data.ravel()[ipha][:,0]  # discard dimension 'z_oc'

            pro.add_axis('theta_oc', pha.axis('theta_oc'))
            pro.add_dataset('phase_oc', pha_, ['iphase', 'stk', 'theta_oc'])
            pro.add_dataset('iphase_oc', ipha, ['wavelength', 'z_oc'])
        else:
            coef_trunc = 2.

        iop = self.calc_iop(wav, coef_trunc=coef_trunc)

        shp = (len(wav), 2)

        tau_tot = np.zeros(shp, dtype='float32')
        tau_tot[:,1] = - ((iop['atot'] + iop['btot']) * self.depth)
        pro.add_dataset('OD_oc', tau_tot,
                        ['wavelength', 'z_oc'])

        tau_sca = np.zeros(shp, dtype='float32')
        tau_sca[:,1] = - (iop['btot'] * self.depth)
        pro.add_dataset('OD_sca_oc', tau_sca,
                        ['wavelength', 'z_oc'])

        tau_abs = np.zeros(shp, dtype='float32')
        tau_abs[:,1] = - (iop['atot'] * self.depth)
        pro.add_dataset('OD_abs_oc', tau_abs,
                        ['wavelength', 'z_oc'])

        pmol = np.zeros(shp, dtype='float32')
        pmol[:,1] = iop['bw']/iop['btot']
        pro.add_dataset('pmol_oc', pmol,
                        ['wavelength', 'z_oc'])

        with np.errstate(invalid='ignore'):
            ssa = tau_sca/tau_tot

        ssa[np.isnan(ssa)] = 1.

        pro.add_dataset('ssa_oc', ssa,
                        ['wavelength', 'z_oc'])

        pro.add_dataset('albedo_seafloor',
                        self.ALB.get(wav),
                        ['wavelength'])


        return pro

    def phase(self, wav, Bp):
        '''
        Calculate the phase function and associated truncation factor
        as a MLUT
        Bp is the backscattering ratio
        '''
        nwav = len(wav)

        # particles phase function
        # see Park & Ruddick, 05
        # https://odnature.naturalsciences.be/downloads/publications/park_appliedoptics_2005.pdf
        ang = np.linspace(0, np.pi, self.NANG, dtype='float64')    # angle in radians
        ff1 = fournierForand(ang, 1.117,3.695)[None,:]
        ff2 = fournierForand(ang, 1.05, 3.259)[None,:]

        itronc = int(self.NANG * self.ang_trunc/180.)
        pha = np.zeros((nwav, 1, NPSTK, self.NANG), dtype='float64')
        r1 = ((Bp - 0.002)/0.028)[:,None]

        pha[:,0,0,:] = 0.5*(r1*ff1 + (1-r1)*ff2)

        # truncate
        pha[:,0,0,:itronc] = pha[:,0,0,itronc][:,None]

        pha[:,0,1,:] = pha[:,0,0,:]
        pha[:,0,2,:] = 0.
        pha[:,0,3,:] = 0.

        pha[:,:,:,0] = 0.

        # normalize
        integ_ff = integ_phase(ang, pha[:,0,0,:] + pha[:,0,1,:])
        pha *= 2./integ_ff[:,None,None,None]

        # create output MLUT
        result = MLUT()
        result.add_axis('wav_phase_oc', wav)
        result.add_axis('z_phase_oc', np.array([0.]))
        result.add_axis('theta_oc', ang*180./np.pi)
        result.add_dataset('phase', pha, ['wav_phase_oc', 'z_phase_oc', 'stk', 'theta_oc'])
        result.add_dataset('coef_trunc', integ_ff[:,None], ['wav_phase_oc', 'z_phase_oc'])

        return result

class IOP_AOS_WATER(IOP_base):
    '''
    IOP model
    using similar IOP parameterization as AOS model case II (Pure water) for validation
    Parameters:
        ALB: albedo of the sea floor (Albedo instance)
        DEPTH: depth in m
    '''
    def __init__(self, ALB=Albedo_cst(0.), DEPTH=10000):
        self.depth = float(DEPTH)
        self.ALB = ALB

    def calc_iop(self, wav):
        '''
        inherent optical properties calculation

        wav: wavelength in nm
        '''
        wl_1 = np.array([350.0525, 450.0666, 550.084, 650.099])

        aw   = np.array([0.0204, 0.0092, 0.0565, 0.3400])
        bw   = np.array([0.0134, 0.0045, 0.0019, 0.0010])

        #ablk = np.array([0.0204, 0.0092, 0.3400, 0.6240])
        #wblk = np.array([0.3964, 0.3285, 0.0325, 0.0029])

        #
        # wavelength index
        #
        #if (wav < wl_1[0]) or (wav > wl_1[-1]):
        #    raise Exception('Error, wavelength {} is out of range ({}, {})'.format(wav, wl_1[0], wl_1[-1]))
        #i1 = int((wav - wl_1[0])/(wl_1[1] - wl_1[0]))
        i1 = np.array([np.abs(wl_1 - x).argmin() for x in wav])

        atot = aw[i1]
        aphy = aw[i1]*0.
        btot = bw[i1]
        Bp   = bw[i1]*0.

        return {'atot': atot,
                'aphy': aphy,
                'aw': atot,
                'bw': btot,
                'btot': btot,
                'Bp': Bp,
                }


    def calc(self, wav):
        '''
        Profile and phase function calculation

        wav: wavelength in nm
        '''
        wav = np.array(wav)
        pro = MLUT()
        pro.add_axis('wavelength', wav)
        pro.add_axis('z_oc', np.array([0., -self.depth]))

        iop = self.calc_iop(wav)

        shp = (len(wav), 2)

        tau_tot = np.zeros(shp, dtype='float32')
        tau_tot[:,1] = - ((iop['atot'] + iop['btot']) * self.depth)
        pro.add_dataset('OD_oc', tau_tot,
                        ['wavelength', 'z_oc'])

        tau_sca = np.zeros(shp, dtype='float32')
        tau_sca[:,1] = - (iop['btot'] * self.depth)
        pro.add_dataset('OD_sca_oc', tau_sca,
                        ['wavelength', 'z_oc'])

        tau_abs = np.zeros(shp, dtype='float32')
        tau_abs[:,1] = - (iop['atot'] * self.depth)
        pro.add_dataset('OD_abs_oc', tau_abs,
                        ['wavelength', 'z_oc'])

        pmol = np.zeros(shp, dtype='float32')
        pmol[:,1] = iop['bw']/iop['btot']
        pro.add_dataset('pmol_oc', pmol,
                        ['wavelength', 'z_oc'])

        with np.errstate(invalid='ignore'):
            ssa = tau_sca/tau_tot

        ssa[np.isnan(ssa)] = 1.

        pro.add_dataset('ssa_oc', ssa,
                        ['wavelength', 'z_oc'])

        pro.add_dataset('albedo_seafloor',
                        self.ALB.get(wav),
                        ['wavelength'])

        return pro
