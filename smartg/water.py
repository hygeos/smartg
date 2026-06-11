#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import numpy as np
import xarray as xr
from warnings import warn
from smartg.atmosphere import diff1
from smartg.albedo import Albedo_cst
from smartg.config import NPSTK
from smartg.phase import fournierForand, integ_phase, calc_iphase
from smartg.bandset import BandSet
from smartg.config import DIR_AUXDATA as dir_aux
from smartg.interp import interp_1d_coord

def diff2(x):
    return np.ediff1d(x, to_end=[0.])


def _read_aw(dir_aux):
    '''
    Read pure water absorption from pope&fry, 97 (<700nm)
    and palmer&williams, 74 (>700nm)
    '''

    # Pope&Fry
    with open(dir_aux / 'water' / 'pope97.dat', 'rb') as fp:
        for i in range(6): fp.readline()  # skip the first 6 lines
        data_pf = np.genfromtxt(fp)
    aw_pf = data_pf[:,1] * 100 #  convert from cm-1 to m-1
    lam_pf = data_pf[:,0]
    ok_pf = lam_pf <= 725

    # Palmer&Williams
    data_pw = np.genfromtxt(dir_aux / 'water' / 'palmer74.dat', skip_header=5)
    aw_pw = data_pw[::-1,1] * 100 #  convert from cm-1 to m-1
    lam_pw = data_pw[::-1,0]
    ok_pw = lam_pw > 725

    aw = xr.DataArray(
            np.array(list(aw_pf[ok_pf]) + list(aw_pw[ok_pw])),
            dims=['wavelength'],
            coords={'wavelength': np.array(list(lam_pf[ok_pf]) + list(lam_pw[ok_pw]))}
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
    def __init__(self, phase=None, bp=None, bw=None,
                 atot=None, ap=None, aw=None, aCDOM=None, Bp=None,
                 Z=[0, -10000], NANG=721, ang_trunc=5., pfwav=None, ALB=Albedo_cst(0.)):

        self.Z = np.array(Z, dtype='float')
        self.bp = bp
        self.bw = bw
        self.atot = atot
        self.ap = ap
        self.aw = aw
        if phase is not None and hasattr(phase, 'to_xarray'):
            phase = phase.to_xarray()
        if phase is not None and phase.shape[2] == 4:
            pha_6 = np.zeros((phase.shape[0], phase.shape[1], 6, phase.shape[3]), dtype=np.float64)
            pha_6[:,:,0:4,:] = phase[:,:,:,:].copy() # F11, F12, F33, F34
            pha_6[:,:,4,:] = phase[:,:,0,:].copy() # F22 = F11
            pha_6[:,:,5,:] = phase[:,:,2,:].copy() # F44 = F33
            axes = list(phase.dims)
            coords = {}
            for i, dim in enumerate(axes):
                if i == 2:
                    coords[dim] = np.arange(6)
                elif dim in phase.coords and phase.coords[dim].size == pha_6.shape[i]:
                    coords[dim] = phase.coords[dim].values
                else:
                    coords[dim] = np.arange(pha_6.shape[i])
            phase = xr.DataArray(
                pha_6,
                dims=axes,
                coords=coords)
        self.phase = phase
        self.aCDOM = aCDOM
        self.Bp  = Bp
        self.ALB = ALB
        self.NANG= NANG
        self.ang_trunc= ang_trunc
        self.coef_trunc = None
        self.NLAYER = len(Z)
        self.pfwav = pfwav

        self.AW = _read_aw(dir_aux)

    def calc(self, wav, use_old_calc_iphase=False):
        shp = [x.shape for x in [self.bp, self.bw, self.atot, self.ap, self.aw, self.aCDOM, self.Bp] if x is not None][0]

        if not isinstance(wav, BandSet):
            wav = BandSet(wav)
        wav = np.array(wav)

        # absorption
        if self.ap is None:
            ap = np.zeros(shp, dtype='float')
        else:
            ap = self.ap
        
        # multilayer case
        NZ = ap.size//wav.size
        wav2 = np.stack([wav]*NZ,axis=1)

        if self.aCDOM is None:
            aCDOM = np.zeros(shp, dtype='float')
        else:
            aCDOM = self.aCDOM

        if self.aw is None:
            aw = interp_1d_coord(self.AW, 'wavelength', wav2)
        else:
            aw = self.aw

        if self.atot is None:
            atot = aw + ap + aCDOM
        else:
            atot = self.atot

        # scattering
        if self.bp is None:
            bp = np.zeros(shp, dtype='float')
        else:
            bp = self.bp

        if self.pfwav is None:
            self.pfwav = wav

        if (self.phase is None) and (self.Bp is None) and ((np.array(bp) > 0).any()):
            raise Exception('No phase function nor Bp has been provided, but bp>0')

        bw = self.bw
        if bw is None:
            bw = 19.3e-4*((wav2/550.)**-4.3)

        FQYC = np.zeros(shp, dtype='float')

        pro = xr.Dataset()
        pro = pro.assign_coords(wavelength=wav[:], z_oc=self.Z)

        pro['T_oc'] = xr.DataArray(np.array([280.]*len(self.Z), dtype='float32'), dims=['z_oc'])

        if (self.phase is None) and (self.Bp is not None) and ((np.array(bp) > 0).any()):
            self.phase, self.coef_trunc = self.calc_phase(self.pfwav[:])
            bp *= self.coef_trunc.values

        btot = bw + bp

        if self.phase is not None:

            pha = self.phase
            pha_, ipha = calc_iphase(pha, pro.coords['wavelength'].values, pro.coords['z_oc'].values, use_old_calc_iphase)

            pro = pro.assign_coords(theta_oc=pha.coords['theta_oc'].values)
            pro['phase_oc'] = xr.DataArray(pha_, dims=['iphase', 'stk', 'theta_oc'])
            pro['iphase_oc'] = xr.DataArray(ipha, dims=['wavelength', 'z_oc'])

        dz = - diff1(self.Z)
        #dz = - diff1(self.Z) * aw/aw
        tau_w   = - (aw   + bw  ) * dz
        tau_p   = - (ap + bp  ) * dz
        tau_y   = - (aCDOM             ) * dz
        tau_tot = - (atot + btot) * dz
        tau_sca = - (btot              ) * dz
        tau_abs = - (atot              ) * dz
        tau_ine = - (ap * FQYC) * dz

        with np.errstate(invalid='ignore'):
            ssa_w   = bw/(aw   + bw)
        ssa_w[np.isnan(ssa_w)] = 1.

        with np.errstate(invalid='ignore'):
            ssa_p   = bp/(ap + bp)
        ssa_p[np.isnan(ssa_p)] = 1.

        with np.errstate(invalid='ignore'):
            pmol    = bw/(bw   + bp)
        pmol[np.isnan(pmol)]   = 1.
        pmol[~np.isfinite(pmol)] = 1.

        with np.errstate(invalid='ignore'):
            pine = tau_ine/tau_sca
        pine[np.isnan(pine)] = 0.
        pine[~np.isfinite(pine)] = 0.

        with np.errstate(invalid='ignore'):
            ssa = tau_sca/tau_tot
        ssa[np.isnan(ssa)] = 1.

        pro['OD_w'] = xr.DataArray(np.cumsum(tau_w, out=tau_w, axis=1),
                        dims=['wavelength', 'z_oc'],
                        attrs={'description':
                               'Cumulated water optical thickness at each wavelength'})

        pro['OD_p_oc'] = xr.DataArray(np.cumsum(tau_p, out=tau_p, axis=1),
                        dims=['wavelength', 'z_oc'],
                        attrs={'description':
                               'Cumulated oceanic particles optical thickness at each wavelength'})

        pro['OD_y'] = xr.DataArray(np.cumsum(tau_y, out=tau_y, axis=1),
                        dims=['wavelength', 'z_oc'],
                        attrs={'description':
                               'Cumulated CDOM optical thickness at each wavelength'})

        pro['OD_oc'] = xr.DataArray(np.cumsum(tau_tot, out=tau_tot, axis=1),
                        dims=['wavelength', 'z_oc'])

        pro['OD_sca_oc'] = xr.DataArray(np.cumsum(tau_sca, out=tau_sca, axis=1),
                        dims=['wavelength', 'z_oc'])

        pro['OD_abs_oc'] = xr.DataArray(np.cumsum(tau_abs, out=tau_abs, axis=1),
                        dims=['wavelength', 'z_oc'])

        pro['pine_oc'] = xr.DataArray(pine,
                        dims=['wavelength', 'z_oc'])

        pro['pmol_oc'] = xr.DataArray(pmol,
                        dims=['wavelength', 'z_oc'])

        pro['ssa_oc'] = xr.DataArray(ssa,
                        dims=['wavelength', 'z_oc'])

        pro['ssa_p_oc'] = xr.DataArray(ssa_p,
                        dims=['wavelength', 'z_oc'])
        pro['ssa_w'] = xr.DataArray(ssa_w,
                        dims=['wavelength', 'z_oc'])
        # NEW !!!
        pro['FQY1_oc'] = xr.DataArray(FQYC,
                        dims=['wavelength', 'z_oc'])
        # NEW !!!

        pro['albedo_seafloor'] = xr.DataArray(self.ALB.get(wav),
                        dims=['wavelength'])
        return pro



    def calc_phase(self, wav):
        '''
        Calculate the phase function and associated truncation factor
        as a MLUT
        '''
        nwav = len(wav)
        nz   = self.NLAYER

        # particles phase function
        # see Park & Ruddick, 05
        # https://odnature.naturalsciences.be/downloads/publications/park_appliedoptics_2005.pdf
        ang = np.linspace(0, np.pi, self.NANG, dtype='float64')    # angle in radians
        ff1 = fournierForand(ang, 1.117,3.695)[None,None,:]
        ff2 = fournierForand(ang, 1.05, 3.259)[None,None,:]

        itronc = int(self.NANG * self.ang_trunc/180.)
        pha = np.zeros((nwav, nz, 6, self.NANG), dtype='float64')
        r1 = ((self.Bp - 0.002)/0.028)[:,:,None]

        pha[:,:,0,:] = r1*ff1 + (1-r1)*ff2

        # truncate
        pha[:,:,0,:itronc] = pha[:,:,0,itronc][:,:,None]

        pha[:,:,1,:] = 0.
        pha[:,:,2,:] = 0.
        pha[:,:,3,:] = 0.
        pha[:,:,4,:] = pha[:,:,0,:].copy() # P22 = P11
        pha[:,:,5,:] = pha[:,:,2,:].copy() # P44 = P33

        pha[:,:,:,0] = 0.

        # normalize
        integ_ff = integ_phase(ang, pha[:,:,0,:])
        pha *= 2./integ_ff[:,:,None,None]

        P = xr.DataArray(pha,  # stk, theta
            dims=['wav_phase', 'z_phase', 'stk', 'theta_oc'],
            coords={'wav_phase': wav, 'z_phase': self.Z, 'theta_oc': np.rad2deg(ang)},
           )
        coef_trunc = xr.DataArray(integ_ff[:,:]*0.5, dims=['wav_phase', 'z_phase'], 
                                  coords={'wav_phase': wav, 'z_phase': self.Z})

        return P, coef_trunc




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
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)
        wav = np.array(wav)

        pro = xr.Dataset()
        pro = pro.assign_coords(wavelength=wav[:], z_oc=np.zeros(2))
        shp = (len(wav), 2)

        pro['T_oc'] = xr.DataArray(np.array([280., 280.], dtype='float32'), dims=['z_oc'])
        pro['OD_oc'] = xr.DataArray(np.zeros(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['OD_w'] = xr.DataArray(np.zeros(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['OD_p_oc'] = xr.DataArray(np.zeros(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['OD_sca_oc'] = xr.DataArray(np.zeros(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['OD_abs_oc'] = xr.DataArray(np.zeros(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['OD_y'] = xr.DataArray(np.zeros(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['pmol_oc'] = xr.DataArray(np.ones(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['pine_oc'] = xr.DataArray(np.ones(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['FQY1_oc'] = xr.DataArray(np.ones(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['ssa_oc'] = xr.DataArray(np.ones(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['albedo_seafloor'] = xr.DataArray(self.ALB.get(wav), dims=['wavelength'])

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
        FQYC: Chorophyll a fluorescence Quantum Yield
    '''
    def __init__(self, chl, pfwav=None, ALB=Albedo_cst(0.),
                 DEPTH=10000, NANG=72001, ang_trunc=5., FQYC=0.0):
        self.chl = chl
        self.depth = float(DEPTH)
        self.NANG = NANG
        self.ang_trunc = ang_trunc
        self.ALB = ALB
        self.FQYC = FQYC
        if pfwav is None:
            self.pfwav = None
        else:
            self.pfwav = np.array(pfwav)

        #
        # read pure water absorption coefficient
        #
        self.aw = _read_aw(dir_aux)

        # Bricaud (98)
        ap_bricaud = np.genfromtxt(dir_aux / 'water' / 'aph_bricaud_1998.txt',
                                   delimiter=',', skip_header=12)  # header is lambda,Ap,Ep,Aphi,Ephi
        self.BRICAUD = xr.Dataset()
        self.BRICAUD = self.BRICAUD.assign_coords(wav=ap_bricaud[:,0])
        self.BRICAUD['A'] = xr.DataArray(ap_bricaud[:,1], dims=['wav'])
        self.BRICAUD['E'] = xr.DataArray(1-ap_bricaud[:,2], dims=['wav'])


    def calc_iop(self, wav, coef_trunc=2.):
        '''
        inherent optical properties calculation

        wav: wavelength in nm
        coef_trunc: integrate (normalized to 2) of the truncated phase function
        -> scattering coefficient has to be multiplied by coef_trunc/2.
        '''
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)
        wav = np.array(wav)


        chl = self.chl

        # pure water absorption
        aw = interp_1d_coord(self.aw, 'wavelength', wav[:], extrema=True)

        # phytoplankton absorption
        aphy = (interp_1d_coord(self.BRICAUD['A'], 'wav', wav, extrema=True)
            * (chl**interp_1d_coord(self.BRICAUD['E'], 'wav', wav, extrema=True)))


        # NEW !!!
        # chlorophyll fluorescence (scattering coefficient)
        FQYC = np.full_like(aphy, self.FQYC) # Fluorescence Quantum Yield for Chlorophyll
        FQYC[wav<370.]=0.
        FQYC[wav>690.]=0.
        # NEW !!!

        # CDM absorption central value
        # from Bricaud et al GBC, 2012 (data from nov 2007)
        fa = 1.
        aCDM443 = fa * 0.069 * (chl**1.070)

        S = 0.00262*(aCDM443**(-0.448))
        if (S > 0.025): S=0.025
        if (S < 0.011): S=0.011

        aCDM = aCDM443 * np.exp(-S*(wav - 443))

        # NEW !!!
        atot = aw + aphy * (1.-FQYC) + aCDM
        # NEW !!!

        # Pure Sea water scattering coefficient
        bw = 19.3e-4*((wav/550.)**-4.3)

        bp = 0.416*(chl**0.766)*550./wav

        bp *= coef_trunc/2.

        # NEW !!!
        btot = bw + bp + aphy * FQYC
        # NEW !!!

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
                'aCDM': aCDM,
                'aw': aw,
                'bw': bw,
                'btot': btot,
                'bp': bp,
                'Bp': Bp,
                'FQYC': FQYC
                }


    def calc(self, wav, phase=True, use_old_calc_iphase=False):
        '''
        Profile and phase function calculation

        wav: wavelength in nm
        '''
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)

        wav = np.array(wav)
        pro = xr.Dataset()
        pro = pro.assign_coords(wavelength=wav[:], z_oc=np.array([0., -self.depth]))
        pro['T_oc'] = xr.DataArray(np.array([280., 280.], dtype='float32'), dims=['z_oc'])

        if phase:
            if self.pfwav is None:
                wav_pha = wav
            else:
                wav_pha = self.pfwav
            Bp = self.calc_iop(wav_pha)['Bp']
            pha = self.phase(wav_pha, Bp)

            pha_, ipha = calc_iphase(pha['phase'], pro.coords['wavelength'].values, pro.coords['z_oc'].values, use_old_calc_iphase)

            # index with ipha and reshape to broadcast to [wav, z]
            coef_trunc = pha['coef_trunc'].values.ravel()[ipha][:,0]  # discard dimension 'z_oc'

            pro = pro.assign_coords(theta_oc=pha.coords['theta_oc'].values)
            pro['phase_oc'] = xr.DataArray(pha_, dims=['iphase', 'stk', 'theta_oc'])
            pro['iphase_oc'] = xr.DataArray(ipha, dims=['wavelength', 'z_oc'])
        else:
            coef_trunc = 2.

        iop = self.calc_iop(wav, coef_trunc=coef_trunc)

        shp = (len(wav), 2)

        tau_w = np.zeros(shp, dtype='float32')
        tau_y = np.zeros(shp, dtype='float32')
        tau_p = np.zeros(shp, dtype='float32')
        ssa_w = np.zeros(shp, dtype='float32')
        ssa_p = np.zeros(shp, dtype='float32')
        tau_y[:,1]   = - iop['aCDM'] * self.depth
        tau_w[:,1]   = -(iop['aw'] + iop['bw'])*self.depth
        tau_p[:,1]   = -(iop['aphy'] + iop['bp'])*self.depth
        ssa_w[:,1]   =  iop['bw']/(iop['aw'] + iop['bw'])
        with np.errstate(invalid='ignore'):
            ssa_p[:,1] = iop['bp']/(iop['aphy'] + iop['bp'])
        ssa_p[np.isnan(ssa_p)] = 0.

        pro['OD_w'] = xr.DataArray(tau_w,
                        dims=['wavelength', 'z_oc'],
                        attrs={'description':
                               'Cumulated water optical thickness at each wavelength'})

        pro['OD_p_oc'] = xr.DataArray(tau_p,
                        dims=['wavelength', 'z_oc'],
                        attrs={'description':
                               'Cumulated oceanic particles optical thickness at each wavelength'})

        pro['OD_y'] = xr.DataArray(tau_y,
                        dims=['wavelength', 'z_oc'],
                        attrs={'description':
                               'Cumulated CDOM optical thickness at each wavelength'})

        tau_tot = np.zeros(shp, dtype='float32')
        tau_tot[:,1] = - ((iop['atot'] + iop['btot']) * self.depth)
        pro['OD_oc'] = xr.DataArray(tau_tot,
                        dims=['wavelength', 'z_oc'])

        tau_sca = np.zeros(shp, dtype='float32')
        tau_sca[:,1] = - (iop['btot'] * self.depth)
        pro['OD_sca_oc'] = xr.DataArray(tau_sca,
                        dims=['wavelength', 'z_oc'])

        tau_abs = np.zeros(shp, dtype='float32')
        tau_abs[:,1] = - (iop['atot'] * self.depth)
        pro['OD_abs_oc'] = xr.DataArray(tau_abs,
                        dims=['wavelength', 'z_oc'])

        pmol = np.ones(shp, dtype='float32')
        pmol[:,1] = iop['bw']/(iop['bw']+iop['bp'])
        pro['pmol_oc'] = xr.DataArray(pmol,
                        dims=['wavelength', 'z_oc'])

        with np.errstate(invalid='ignore'):
            ssa = tau_sca/tau_tot
        ssa[np.isnan(ssa)] = 1.

        pro['ssa_oc'] = xr.DataArray(ssa,
                        dims=['wavelength', 'z_oc'])
        pro['ssa_p_oc'] = xr.DataArray(ssa_p,
                        dims=['wavelength', 'z_oc'])
        pro['ssa_w'] = xr.DataArray(ssa_w,
                        dims=['wavelength', 'z_oc'])

        # NEW !!!
        tau_ine = np.zeros(shp, dtype='float32')
        tau_ine[:,1] = -((iop['aphy'] * iop['FQYC']) * self.depth)
        with np.errstate(invalid='ignore', divide='ignore'):
            pine = tau_ine/tau_sca
        pine[~np.isfinite(pine)] = 0.
        pro['pine_oc'] = xr.DataArray(pine,
                        dims=['wavelength', 'z_oc'])

        FQY1 = np.zeros(shp, dtype='float32')
        FQY1[:,1] = iop['FQYC']
        pro['FQY1_oc'] = xr.DataArray(FQY1,
                        dims=['wavelength', 'z_oc'])

        # NEW !!!
        pro['albedo_seafloor'] = xr.DataArray(self.ALB.get(wav),
                        dims=['wavelength'])


        return pro

    def phase(self, wav, Bp):
        '''
        Calculate the phase function and associated truncation factor
        as a MLUT
        Bp is the backscattering ratio
        '''
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)
        wav = np.array(wav)

        nwav = len(wav)

        # particles phase function
        # see Park & Ruddick, 05
        # https://odnature.naturalsciences.be/downloads/publications/park_appliedoptics_2005.pdf
        ang = np.linspace(0, np.pi, self.NANG, dtype='float64')    # angle in radians
        ff1 = fournierForand(ang, 1.117,3.695)[None,:]
        ff2 = fournierForand(ang, 1.05, 3.259)[None,:]

        itronc = int(self.NANG * self.ang_trunc/180.)
        pha = np.zeros((nwav, 1, 6, self.NANG), dtype='float64')
        r1 = ((Bp - 0.002)/0.028)[:,None]

        pha[:,0,0,:] = r1*ff1 + (1-r1)*ff2

        # truncate
        pha[:,0,0,:itronc] = pha[:,0,0,itronc][:,None]

        pha[:,0,1,:] = 0.
        pha[:,0,2,:] = 0.
        pha[:,0,3,:] = 0.
        pha[:,0,4,:] = pha[:,0,0,:].copy() # P22 = P11
        pha[:,0,5,:] = pha[:,0,2,:].copy() # P44 = P33

        pha[:,:,:,0] = 0.

        # normalize
        integ_ff = integ_phase(ang, pha[:,0,0,:])
        pha *= 2./integ_ff[:,None,None,None]

        # create output Dataset
        result = xr.Dataset()
        result = result.assign_coords(wav_phase=wav, z_phase=np.array([0.]), theta_oc=ang*180./np.pi)
        result['phase'] = xr.DataArray(pha, dims=['wav_phase', 'z_phase', 'stk', 'theta_oc'])
        result['coef_trunc'] = xr.DataArray(integ_ff[:,None]*0.5, dims=['wav_phase', 'z_phase'])

        return result


class IOP_profile(IOP_base):
    '''
    IOP model described in Zhai et al, optics express, 2017

    Parameters:
        chl: chlorophyll concentration in mg/m3 at the surface
        NANG: number of angles for the phase function
        ang_trunc: truncation angle in degrees
        ALB: albedo of the sea floor (Albedo instance)
        DEPTH: depth in m
        pfwav: list of wavelengths at which the phase functions are calculated
        NLAYER: Number of vertical layers
        Zeu   : Euphotic Depth (m), default is computed from climatology
        MIXED: Mixed or Statified waters
        FQYC : Fluorescence Quantum Yield for Chlorophyll
    '''
    def __init__(self, chls, pfwav=None, ALB=Albedo_cst(0.),
                 DEPTH=300., NANG=7201, ang_trunc=5., NLAYER=20, Zeu=None, MIXED=False, FQYC=0.):
        self.chls = chls
        self.depth = float(DEPTH)
        self.NANG = NANG
        self.ang_trunc = ang_trunc
        self.ALB = ALB
        self.FQYC = FQYC
        self.NLAYER=NLAYER
        if pfwav is None:
            self.pfwav = None
        else:
            self.pfwav = np.array(pfwav)

        #
        # read pure water absorption coefficient
        #
        self.aw = _read_aw(dir_aux)

        # Bricaud (98)
        # Absorption of the phytoplankton
        ap_bricaud = np.genfromtxt(dir_aux / 'water' / 'aph_bricaud_1998.txt',
                                   delimiter=',', skip_header=12)  # header is lambda,Ap,Ep,Aphi,Ephi
        # Add extension to 360 nm (Wei et al., 2016)
        # spectral slope of aph is symetrical wrt 440 nm in the 360-520 spectral range
        wUV = np.linspace(360., 398., num=20)
        A   = ap_bricaud[:,1]
        B   = 1.-ap_bricaud[:,2]
        w   = ap_bricaud[:,0]
        ii  = np.where((w<=520.) & (w>480.))
        AUV = np.zeros_like(wUV)
        BUV = np.zeros_like(wUV)
        AUV[::-1] = A[ii]
        BUV[::-1] = B[ii]
        self.BRICAUD = xr.Dataset()
        self.BRICAUD = self.BRICAUD.assign_coords(wav=np.concatenate((wUV,ap_bricaud[:,0])))
        self.BRICAUD['A'] = xr.DataArray(np.concatenate((AUV,A)), dims=['wav'])
        self.BRICAUD['E'] = xr.DataArray(np.concatenate((BUV,B)), dims=['wav'])

        ## Determine Chl vertical profile
        #1. Determine chlorophyll interagted column until Euphotic Depth Zeu 
        if Zeu is None:
            if not MIXED:
                if (chls > 1.): chl_zeu = 37.7*chls**0.615 
                else: chl_zeu = 36.1*chls**0.357
            else:
                chl_zeu = 42.1*chls**0.538
            Zeu = 568.2*chl_zeu**(-0.746)
         
        #2. Derive Euphotic Depth and make vertical grid
        Zmax= min(DEPTH, 5*Zeu)
        self.z = - np.linspace(0, Zmax, num=NLAYER+1)
        #self.z = np.linspace(0, Zmax, num=NLAYER+1)

        #3. Introduce reduced concentration chi and reduced depth zeta
        # Stratified Trophic case 1 parametrization
        #   see J. Uitz, H. Claustre, A. Morel, and S. B. Hooker, 
        #   “Vertical distribution of phytoplankton communities in open ocean: 
        #   An assessment based on surface chlorophyll,” J. Geophys. Res. 111, C08005 (2006).
        self.chi_b    = 0.471
        self.s        = 0.135
        self.chi_max  = 1.572
        self.zeta_max = 0.969
        self.Dzeta    = 0.393
        zeta = abs(self.z/Zeu)
        chl  = self.chls*self.chi(zeta)/self.chi(0.)
        chl[chl<0.]=1e-8
        self.chl = chl
        self.chlmean = np.trapezoid(chl,-self.z)/DEPTH
        #self.chlmean = np.trapezoid(chl,self.z)/DEPTH

    def chi(self, zeta):
        return self.chi_b - self.s*zeta + self.chi_max*np.exp(-((zeta-self.zeta_max)/self.Dzeta)**2)


    def calc_iop(self, wav, coef_trunc=2., p1=0.33, R1=0.5, R2=0.5):
        '''
        inherent optical properties calculation

        wav: wavelength in nm
        coef_trunc: integrate (normalized to 2) of the truncated phase function
        -> scattering coefficient has to be multiplied by coef_trunc/2.
        p1 and R1 are parameters explained in Zhai et al. 2017, related to particles extinction
        R2 is related to CDOM absorption
        '''
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)
        wav = np.array(wav)

        chl = self.chl
        chl2, wav2 = np.meshgrid(chl,wav)

        # pure water absorption
        aw = interp_1d_coord(self.aw, 'wavelength', wav2)
        # Pure Sea water scattering coefficient
        bw = 19.3e-4*((wav2/550.)**-4.32)

        # phytoplankton absorption
        #aphy = (self.BRICAUD['A'][Idx(wav2, fill_value='extrema')]
        #        * (chl2**self.BRICAUD['E'][Idx(wav2, fill_value='extrema')]))
        #aphy440B = (self.BRICAUD['A'][Idx(440., fill_value='extrema')]
        #        * (chl2**self.BRICAUD['E'][Idx(440., fill_value='extrema')]))
        # specific phytoplankton absorption
        chl2star=np.full_like(chl2, 1.)
        aphystar = (interp_1d_coord(self.BRICAUD['A'], 'wav', wav2, extrema=True)
            * (chl2star**interp_1d_coord(self.BRICAUD['E'], 'wav', wav2, extrema=True)))
        aphy = aphystar * chl2
        aphystar440 = (interp_1d_coord(self.BRICAUD['A'], 'wav', 440., extrema=True)
            * (chl2star**interp_1d_coord(self.BRICAUD['E'], 'wav', wav2, extrema=True)))
        aphy440 = aphystar440 * chl2
        # phytoplankton covariant particles extinction
        ## Zhai et al., 2017
        # cp550 = p1*chl2**0.57
        # n1    = -0.4 + (1.6+1.2*R1)/(1+chl2**0.5)
        # cp    = cp550*(550./wav2)**n1
        # bp    = cp-aphy

        ## return to IOP_1 definition
        #bp = 0.416*(chl2**0.766)*550./wav2
        piz440=0.68
        bp440 = aphy440 * piz440/(1-piz440)
        bp = bp440 *(wav2/440.)**(-1.)
        #bp = 0.416*(chl2**0.766)*(wav2/550.)**(-1.)*6
        # correct for phase function truncation
        bp *= coef_trunc/2.
        # NEW !!!
        # chlorophyll fluorescence (scattering coefficient)
        FQYC = np.full_like(aphy, self.FQYC) # Fluorescence Quantum Yield for Chlorophyll
        FQYC[wav2<370.]=0.
        FQYC[wav2>690.]=0.

        # CDOM covariant absorption
        ## Zhai et al., 2017
        # p2       = 0.3 + (5.7*R2*aphy440)/(0.02+aphy440)
        # aCDOM440 = p2 * aphy440
        # aCDOM    = aCDOM440 *np.exp(-0.014*(wav2-440.))
        # CDM absorption central value

        ## return to IOP_1 definition
        # from Bricaud et al GBC, 2012 (data from nov 2007)
        #fa = 1.
        #aCDM443 = fa * 0.069 * (chl2**1.070)
        aCDM440 = 0.24*aphy440**0.43

        #S = 0.00262*(aCDM443**(-0.448))
        #S[S > 0.025] = 0.025
        #S[S < 0.011] = 0.011
        S=0.02

        aCDOM = aCDM440 * np.exp(-S*(wav2 - 440))
        #aCDOM = aCDM443 * np.exp(-S*(wav2 - 443))

        # NEW !!!
        atot = aw + aphy*(1.-FQYC) + aCDOM
        # NEW !!!

        # NEW !!!
        SPM = 0. # g/m3
        #SPM = 10. # g/m3
        gamma=0.5
        bbpnap650 = 10**(1.03*np.log10(SPM) - 2.06) # Neukermans et al 2012
        bbpnap = bbpnap650*(wav2/650.)**(-gamma)
        Bpnap = np.zeros_like(aphy)
        Bpnap[:] = 0.04
        bpnap  = bbpnap/Bpnap
        bp += bpnap
        btot = bw + bp + aphy* FQYC
        # NEW !!!

        #
        # backscattering coefficient
        #
        v = np.zeros_like(aphy)
        good=np.where(chl2<2.)
        v[good] = 0.5*(np.log10(chl2[good]) - 0.3)
        #Bp = 0.002 + 0.01*( 0.5-0.25*np.log10(chl2))*((wav2/550.)**v)
        #Bp[np.isinf(Bp)]=0.5
        #Bp[np.isnan(Bp)]=0.5


        Bp = Bpnap

        return {'atot': atot,
                'aphy': aphy,
                'aCDOM':aCDOM,
                'aw': aw,
                'bw': bw,
                'btot': btot,
                'bp': bp,
                'Bp': Bp,
                'FQYC': FQYC
                }

    def calc(self, wav, phase=True, use_old_calc_iphase=False):
        '''
        Profile and phase function calculation

        wav: wavelength in nm
        '''
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)

        wav = np.array(wav)
        pro = xr.Dataset()
        pro = pro.assign_coords(wavelength=wav[:], z_oc=self.z)
        #pro.add_axis('z_oc', -self.z)
        pro['T_oc'] = xr.DataArray(np.array([280.]*len(self.z), dtype='float32'), dims=['z_oc'])


        if phase:
            if self.pfwav is None:
                wav_pha = wav
            else:
                wav_pha = self.pfwav
            Bp = self.calc_iop(wav_pha)['Bp']
            pha = self.phase(wav_pha, Bp[:,1:])
            phase_lut = pha['phase']
            if phase_lut.shape[2] == 4:
                pha_6 = np.zeros((phase_lut.shape[0], phase_lut.shape[1], 6, phase_lut.shape[3]), dtype=np.float64)
                pha_6[:,:,0:4,:] = phase_lut[:,:,:,:].copy() # F11, F12, F33, F34
                pha_6[:,:,4,:] = phase_lut[:,:,0,:].copy() # F22 = F11
                pha_6[:,:,5,:] = phase_lut[:,:,2,:].copy() # F44 = F33
                axes = list(phase_lut.dims)
                coords = {}
                for i, dim in enumerate(axes):
                    if i == 2:
                        coords[dim] = np.arange(6)
                    elif dim in phase_lut.coords and phase_lut.coords[dim].size == pha_6.shape[i]:
                        coords[dim] = phase_lut.coords[dim].values
                    else:
                        coords[dim] = np.arange(pha_6.shape[i])
                phase_lut = xr.DataArray(
                    pha_6,
                    dims=axes,
                    coords=coords)

            pha_, ipha = calc_iphase(phase_lut, pro.coords['wavelength'].values, pro.coords['z_oc'].values, use_old_calc_iphase)

            # index with ipha and reshape to broadcast to [wav, z]
            coef_trunc = pha['coef_trunc'].values.ravel()[ipha][:,:]

            pro = pro.assign_coords(theta_oc=pha.coords['theta_oc'].values)
            pro['phase_oc'] = xr.DataArray(pha_, dims=['iphase', 'stk', 'theta_oc'])
            pro['iphase_oc'] = xr.DataArray(ipha, dims=['wavelength', 'z_oc'])
        else:
            coef_trunc = 2.

        iop = self.calc_iop(wav, coef_trunc=coef_trunc)

        dz   = -diff1(self.z)
        #dz   = diff1(self.z)
        zeros = np.zeros((len(wav),1))
        for key in iop.keys():
            iop[key] = np.append(zeros, iop[key][:,:-1], axis=1)

        tau_w   = - (iop['aw']   + iop['bw']  ) * dz
        tau_p   = - (iop['aphy'] + iop['bp']  ) * dz
        tau_y   = - (iop['aCDOM']             ) * dz
        tau_tot = - (iop['atot'] + iop['btot']) * dz
        tau_sca = - (iop['btot']              ) * dz
        tau_abs = - (iop['atot']              ) * dz
        tau_ine = - (iop['aphy'] * iop['FQYC']) * dz


        with np.errstate(invalid='ignore'):
            ssa_w   = iop['bw']/(iop['aw']   + iop['bw'])
        ssa_w[np.isnan(ssa_w)] = 1.

        with np.errstate(invalid='ignore'):
            ssa_p   = iop['bp']/(iop['aphy'] + iop['bp'])
        ssa_p[np.isnan(ssa_p)] = 1.

        with np.errstate(invalid='ignore'):
            pmol    = iop['bw']/(iop['bw']   + iop['bp'])
        pmol[np.isnan(pmol)]   = 1.

        with np.errstate(invalid='ignore'):
            pine = tau_ine/tau_sca
        pine[np.isnan(pine)] = 0.

        with np.errstate(invalid='ignore'):
            ssa = tau_sca/tau_tot
        ssa[np.isnan(ssa)] = 1.

        pro['OD_w'] = xr.DataArray(np.cumsum(tau_w, out=tau_w, axis=1),
                        dims=['wavelength', 'z_oc'],
                        attrs={'description':
                               'Cumulated water optical thickness at each wavelength'})

        pro['OD_p_oc'] = xr.DataArray(np.cumsum(tau_p, out=tau_p, axis=1),
                        dims=['wavelength', 'z_oc'],
                        attrs={'description':
                               'Cumulated oceanic particles optical thickness at each wavelength'})

        pro['OD_y'] = xr.DataArray(np.cumsum(tau_y, out=tau_y, axis=1),
                        dims=['wavelength', 'z_oc'],
                        attrs={'description':
                               'Cumulated CDOM optical thickness at each wavelength'})

        pro['OD_oc'] = xr.DataArray(np.cumsum(tau_tot, out=tau_tot, axis=1),
                        dims=['wavelength', 'z_oc'])

        pro['OD_sca_oc'] = xr.DataArray(np.cumsum(tau_sca, out=tau_sca, axis=1),
                        dims=['wavelength', 'z_oc'])

        pro['OD_abs_oc'] = xr.DataArray(np.cumsum(tau_abs, out=tau_abs, axis=1),
                        dims=['wavelength', 'z_oc'])

        pro['pine_oc'] = xr.DataArray(pine,
                        dims=['wavelength', 'z_oc'])

        pro['pmol_oc'] = xr.DataArray(pmol,
                        dims=['wavelength', 'z_oc'])

        pro['ssa_oc'] = xr.DataArray(ssa,
                        dims=['wavelength', 'z_oc'])

        pro['ssa_p_oc'] = xr.DataArray(ssa_p,
                        dims=['wavelength', 'z_oc'])
        pro['ssa_w'] = xr.DataArray(ssa_w,
                        dims=['wavelength', 'z_oc'])
        # NEW !!!
        pro['FQY1_oc'] = xr.DataArray(iop['FQYC'],
                        dims=['wavelength', 'z_oc'])
        # NEW !!!

        pro['albedo_seafloor'] = xr.DataArray(self.ALB.get(wav),
                        dims=['wavelength'])

        return pro


    def phase(self, wav, Bp):
        '''
        Calculate the phase function and associated truncation factor
        as a MLUT
        Bp is the backscattering ratio
        '''
        nwav = len(wav)
        nz   = self.NLAYER

        # particles phase function
        # see Park & Ruddick, 05
        # https://odnature.naturalsciences.be/downloads/publications/park_appliedoptics_2005.pdf
        ang = np.linspace(0, np.pi, self.NANG, dtype='float64')    # angle in radians
        ff1 = fournierForand(ang, 1.117,3.695)[None,None,:]
        ff2 = fournierForand(ang, 1.05, 3.259)[None,None,:]

        itronc = int(self.NANG * self.ang_trunc/180.)
        pha = np.zeros((nwav, nz, 6, self.NANG), dtype='float64')
        r1 = ((Bp - 0.002)/0.028)[:,:,None]

        pha[:,:,0,:] = r1*ff1 + (1-r1)*ff2

        # truncate
        pha[:,:,0,:itronc] = pha[:,:,0,itronc][:,:,None]

        pha[:,:,1,:] = 0.
        pha[:,:,2,:] = 0.
        pha[:,:,3,:] = 0.
        pha[:,:,4,:] = pha[:,:,0,:].copy() # P22 = P11
        pha[:,:,5,:] = pha[:,:,2,:].copy() # P44 = P

        pha[:,:,:,0] = 0.

        # normalize
        integ_ff = integ_phase(ang, pha[:,:,0,:])
        pha *= 2./integ_ff[:,:,None,None]

        # create output Dataset
        result = xr.Dataset()
        result = result.assign_coords(wav_phase=wav, z_phase=self.z[:-1], theta_oc=ang*180./np.pi)
        #result.add_axis('z_phase', -self.z[:-1])
        result['phase'] = xr.DataArray(pha, dims=['wav_phase', 'z_phase', 'stk', 'theta_oc'])
        result['coef_trunc'] = xr.DataArray(integ_ff[:,:]*0.5, dims=['wav_phase', 'z_phase'])

        return result
