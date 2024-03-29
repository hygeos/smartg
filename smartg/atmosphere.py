#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, division, absolute_import
import numpy as np
from os.path import join, dirname, exists, basename
from glob import glob
from luts.luts import MLUT, LUT, Idx
from smartg.tools.phase import calc_iphase
try:
    from smartg.tools.third_party_utils import change_altitude_grid
except ModuleNotFoundError:
    pass
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy import constants
from scipy.constants import speed_of_light, Planck, Boltzmann
from smartg.bandset import BandSet
import netCDF4
from smartg.config import NPSTK, dir_libradtran_opac
from smartg.config import dir_libradtran_atmmod
from smartg.config import dir_libradtran_crs
from warnings import warn
import sys
import pandas as pd
if sys.version_info[:2] >= (3, 0):
    xrange = range


class Species(object):
    '''
    Optical properties of one species

    List of species:
        inso.mie, miam.mie, miam_spheroids.tmatrix,
        micm.mie, micm_spheroids.tmatrix, minm.mie,
        minm_spheroids.tmatrix, mitr.mie,
        mitr_spheroids.tmatrix, soot.mie,
        ssam.mie, sscm.mie, suso.mie,
        waso.mie, wc.sol.mie
    '''
    def __init__(self, species, wav_clip=False):

        self.name = species
        self.wav_clip = wav_clip
        fname = join(dir_libradtran_opac, 'optprop', species+'.cdf')
        if not exists(fname):
            raise Exception('file {} does not exist'.format(fname))
        self.fname = fname

        nc = netCDF4.Dataset(fname)

        self._wav = nc.variables["wavelen"][:]*1e3  # wavelength (converted µm -> nm)

        if 'hum' in nc.variables: 
            self._rh_reff = nc.variables['hum'][:]
            self._rh_or_reff = 'rh'
        elif 'reff' in nc.variables:
            self._rh_reff = nc.variables['reff'][:]
            self._rh_or_reff = 'reff'
        else:
            raise Exception('Error')
        self._nrh_reff = len(self._rh_reff)

        # density in g/cm^3
        # note: bypass the dimension lambda
        # constant values for all wavelengths
        self._rho = LUT(
                nc.variables["rho"][0,:],
                axes = [self._rh_reff],
                names = [self._rh_or_reff],
            )

        # extinction coefficient (nlam, rh) in km^-1/(g/m^3)
        self._ext = LUT(
                    np.array(nc.variables['ext'][:]),
                    axes = [self._wav, self._rh_reff],
                    names = ['lambda', self._rh_or_reff],
                )

        # single scattering albedo (nlam, nhum)
        self._ssa = LUT(
                    np.array(nc.variables['ssa']),
                    axes = [self._wav, self._rh_reff],
                    names = ['lambda', self._rh_or_reff],
                )

        # scattering angle in degrees (nlam, nhum, nphamat, nthetamax)
        self._theta = LUT(
            nc.variables['theta'][:],
            axes=[self._wav, self._rh_reff, None, None],
            names=['lam', self._rh_or_reff, 'stk', 'nthetamax'])


        # phase matrix (nlam, nhum, nphamat, nthetamax)
        self._phase = LUT(
            nc.variables['phase'][:],
            axes=[self._wav, self._rh_reff, None, None],
            names=['lam', self._rh_or_reff, 'stk', 'nthetamax'])

        # number of scattering angles (nlam, nhum, nphamat)
        self._ntheta = LUT(
                nc.variables['ntheta'][:],
                axes=[self._wav, self._rh_reff, None],
                names=['lam', self._rh_or_reff, 'nthetamax'])

        nc.close()

    def ext_ssa(self, wav, rh=None, reff=None):
        '''
        returns the extinction coefficient and single scattering albedo of
        each layer
            (N x M) or (N x 1) if species does not depend on rh

        parameters:
            wav: array of wavelength in nm (N wavelengths)

            rh: relative humidity (M layers)
                *or*
            reff: effective radius
        '''
        assert isiterable(wav)

        if (self._nrh_reff == 1):
            # wavelength interpolation
            ext = self._ext[Idx(wav), 0]
            ssa = self._ssa[Idx(wav), 0]

            # create empty dimension for rh
            ext = ext[:,None]
            ssa = ssa[:,None]

        elif reff is not None:
            assert self._rh_or_reff == 'reff'
            reff2 = 0.*wav + reff   # so that reff2 has same size as wav

            # wavelength interpolation
            fv = 'extrema' if self.wav_clip else None
            ext = self._ext[Idx(wav, fill_value=fv), Idx(reff2)]
            ssa = self._ssa[Idx(wav, fill_value=fv), Idx(reff2)]

            # create empty dimension for rh
            ext = ext[:,None]
            ssa = ssa[:,None]

        elif rh is not None: # the component properties depend on RH (thus Z)
            assert self._rh_or_reff == 'rh'

            [wav2, rh2] = np.broadcast_arrays(wav[:,None], rh[None,:])
            ext = self._ext[Idx(wav2), Idx(rh2, fill_value='extrema,warn')]
            ssa = self._ssa[Idx(wav2), Idx(rh2, fill_value='extrema,warn')]

            ext *= self._rho[Idx(rh, fill_value='extrema,warn')]/self._rho[Idx(50.)]

        else:
            raise Exception('Error')

        return ext, ssa

    def phase(self, wav, rh, NBTHETA, reff=None):
        '''
        phase function of species at wavelengths wav
        resampled over NBTHETA angles
        '''

        theta = np.linspace(0., 180., num=NBTHETA)
        lam_tabulated = self._phase.axis('lam')
        nlam_tabulated = len(lam_tabulated)

        if (self._nrh_reff > 1) and (self._rh_or_reff == 'rh'):
            # drop first altitude element
            P = LUT(
                np.zeros((nlam_tabulated, len(rh)-1, NPSTK, NBTHETA), dtype='float32')+np.NaN,
                axes=[lam_tabulated, None, None, theta],
                names=['wav_phase', 'z_phase', 'stk', 'theta_atm'],
                )  # nlam_tabulated, nrh, stk, NBTHETA
            for irh_, rh_ in enumerate(rh[1:]):

                irh = Idx(rh_, round=True, fill_value='extrema')
                #P.sub()[Idx(wav),:,:]

                # interpolate each tabulated wavelength
                for ilam in range(nlam_tabulated):
                    for istk in range(NPSTK):
                        th = self._theta[ilam,irh,istk,:]
                        nth = self._ntheta[ilam,irh,istk]
                        #P.data[ilam,irh_,istk,:] = np.interp(theta, th[:nth], self._phase[ilam,irh,istk,:nth]) 
                        P.data[ilam,irh_,istk,:] = interp1d(
                                th[:nth],
                                self._phase[ilam,irh,istk,:nth])(theta)


        else: # phase function does not depend on rh
            P = LUT(
                np.zeros((nlam_tabulated, 1, NPSTK, NBTHETA), dtype='float32')+np.NaN,
                axes=[lam_tabulated, None, None, theta],
                names=['wav_phase', 'z_phase', 'stk', 'theta_atm'],
                )  # nlam_tabulated, nrh, stk, NBTHETA

            if (self._rh_or_reff == 'reff') and (reff is not None):
                irh = Idx(reff, round=True).index(self._phase.axes[1])
            else:
                irh = 0

            # interpolate each tabulated wavelength
            for ilam in range(nlam_tabulated):
                for istk in range(NPSTK):
                    th = self._theta[ilam,irh,istk,:]
                    nth = self._ntheta[ilam,irh,istk]
                    irh_ = 0
                    #P.data[ilam,irh_,istk,:] = np.interp(theta, th[:nth], self._phase[ilam,irh,istk,:nth])
                    P.data[ilam,irh_,istk,:] = interp1d(
                            th[:nth],
                            self._phase[ilam,irh,istk,:nth])(theta)


        # convert I, Q into Ipar, Iper
        P0 = P.data[:,:,0,:].copy()
        P1 = P.data[:,:,1,:].copy()
        P.data[:,:,0,:] = P0+P1
        P.data[:,:,1,:] = P0-P1

        return P.sub()[Idx(wav),:,:,:]
    
    
    def phaseOpti(self, wav, rh, NBTHETA, reff=None):
        '''
        phase function of species at wavelengths wav
        resampled over NBTHETA angles
        '''

        theta = np.linspace(0., 180., num=NBTHETA)
        lam_tabulated = np.array(self._phase.axis('lam'))

        # The optimisation consists to not interpolate at all wavelengths of lam_tabulated,
        # but only the wavelengths of lam_tabulated closely in the range of np.min(wav) and np.max(wav)
        range_ind = np.array([np.argwhere((lam_tabulated <= np.min(wav)))[-1][0],
                              np.argwhere((lam_tabulated >= np.max(wav)))[0][0]])
        ilam_tabulated = np.arange(len(lam_tabulated), dtype=int)
        ilam_opti = np.concatenate(np.argwhere((ilam_tabulated >= range_ind[0]) &
                                               (ilam_tabulated <= range_ind[1])))
        lam_opti = lam_tabulated[ilam_opti]
        nlam_opti = len(lam_opti)
        
        if (self._nrh_reff > 1) and (self._rh_or_reff == 'rh'):
            P = LUT( np.zeros((nlam_opti, len(rh)-1, NPSTK, NBTHETA), dtype='float32')+np.NaN,
                     axes=[lam_opti, None, None, theta],
                     names=['wav_phase', 'z_phase', 'stk', 'theta_atm'],
                    )  # nlam_tabulated, nrh, stk, NBTHETA
            lRh = rh[1:]
        else:
            P = LUT( np.zeros((nlam_opti, 1, NPSTK, NBTHETA), dtype='float32')+np.NaN,
                     axes=[lam_opti, None, None, theta],
                     names=['wav_phase', 'z_phase', 'stk', 'theta_atm'],
                    )  # nlam_tabulated, nrh, stk, NBTHETA
            lRh = [1]
            
        for irh_, rh_ in enumerate(lRh):
            if (self._nrh_reff > 1) and (self._rh_or_reff == 'rh'):
                irh = Idx(rh_, round=True, fill_value='extrema')
            elif (self._rh_or_reff == 'reff') and (reff is not None):
                irh = Idx(reff, round=True).index(self._phase.axes[1])
            else:
                irh = 0
            for idd, ilam in enumerate(ilam_opti):
                    for istk in range(NPSTK):
                        th = self._theta[ilam,irh,istk,:]
                        nth = self._ntheta[ilam,irh,istk]
                        P.data[idd,irh_,istk,:] = np.interp(theta, th[:nth], self._phase[ilam,irh,istk,:nth], period=np.inf)
                        # P.data[idd,irh_,istk,:] = interp1d(th[:nth], self._phase[ilam,irh,istk,:nth])(theta)

        # convert I, Q into Ipar, Iper
        P0 = P.data[:,:,0,:].copy()
        P1 = P.data[:,:,1,:].copy()
        P.data[:,:,0,:] = P0+P1
        P.data[:,:,1,:] = P0-P1

        if (nlam_opti == 1): return P
        else : return P.sub()[Idx(wav),:,:,:]


    @staticmethod
    def list():
        '''
        list standard species files in opac
        '''
        files = glob(join(dir_libradtran_opac, 'optprop', '*.cdf'))
        return map(lambda x: basename(x)[:-4], files)



class SpeciesUser(Species):
    def __init__(self, name, ext, ssa, phase, fill_value=None):
        #Species.__init__(self, species='inso.mie', wav_clip=wav_clip)
        lam   = ext.axis('wavelength')
        lamp  = phase.axis('wavelength')
        nlam  = lam.size
        nlamp = lamp.size
        theta = phase.axis('theta_atm')
        ntheta= theta.size
        self.name = name
        self.fill_value = fill_value
        self._rh_or_reff = 'rh' 
        self._nrh_reff = 1 # no RH dependence
        self._ext = LUT(
            ext.data[:, None],
            axes=[lam, None],
            names=['lam', self._rh_or_reff])
        self._ssa = LUT(
            ssa.data[:, None],
            axes=[lam, None],
            names=['lam', self._rh_or_reff])
        # scattering angle in degrees (nlam, nhum, nphamat, nthetamax)
        self._theta = LUT(
            np.tile(theta,(nlamp,1,4,1)),
            axes=[lamp, None, None, None],
            names=['lam', self._rh_or_reff, 'stk', 'nthetamax'])
        # phase matrix (nlam, nhum, nphamat, nthetamax)
        self._phase = LUT(
            phase.data[:,None,:,:],
            axes=[lamp, None, None, None],
            names=['lam', self._rh_or_reff, 'stk', 'nthetamax'])
        # number of scattering angles (nlam, nhum, nphamat)     
        self._ntheta = LUT(
            np.ones((nlamp, 1, 4), dtype=np.int32)*ntheta,
            axes=[lamp, None, None],
            names=['lam', self._rh_or_reff, 'stk'])
        

    def ext_ssa(self, wav):
        ''' 
        returns the extinction coefficient and single scattering albedo of
        each layer
            (N x 1) if species does not depend on rh

        parameters:
            wav: array of wavelength in nm (N wavelengths)
        '''
        assert isiterable(wav)

        # wavelength interpolation
        ext = self._ext[Idx(wav, fill_value=self.fill_value), 0]
        ssa = self._ssa[Idx(wav, fill_value=self.fill_value), 0]

        # create empty dimension for rh
        ext = ext[:,None]
        ssa = ssa[:,None]
        
        return ext, ssa
    
    def phase(self, wav, NBTHETA, phaseOpti=False):
        '''
        phase function of species at wavelengths wav
        resampled over NBTHETA angles
        '''
        theta = np.linspace(0., 180., num=NBTHETA)
        lam_tabulated = self._phase.axis('lam')
        nlam_tabulated = len(lam_tabulated)

        P = LUT(
            np.zeros((nlam_tabulated, 1, 4, NBTHETA), dtype='float32')+np.NaN,
            axes=[lam_tabulated, None, None, theta],
            names=['wav_phase', 'z_phase', 'stk', 'theta_atm'],
            )  # nlam_tabulated, nrh, stk, NBTHETA

        # interpolate each tabulated wavelength
        for ilam in range(nlam_tabulated):
            for istk in range(4):
                th = self._theta[ilam,0,istk,:]
                nth = self._ntheta[ilam,0,istk]
                P.data[ilam,0,istk,:] = interp1d(
                        th[:nth],
                        self._phase[ilam,0,istk,:nth])(theta)

        # convert I, Q into Ipar, Iper
        P0 = P.data[:,:,0,:].copy()
        P1 = P.data[:,:,1,:].copy()
        P.data[:,:,0,:] = P0+P1
        P.data[:,:,1,:] = P0-P1

        return P.sub()[Idx(wav),:,:,:]



class AeroOPAC(object):
    '''
    Initialize the Aerosol OPAC model

    Args:
        filename: name of the aerosol file.
                  If no directory is specified, assume directory
                  <libradtran>/data/aerosol/OPAC/standard_aerosol_files
                  aerosol files can be:
                      'antarctic', 'continental_average',
                      'continental_clean', 'continental_polluted',
                      'desert', 'desert_spheroids',
                      'maritime_clean', 'maritime_polluted',
                      'maritime_tropical', 'urban'

        tau_ref: optical thickness at wavelength wref
        w_ref: reference wavelength (nm) for aot
        ssa: force particle single scattering albedo
             (scalar or 1-d array-like for multichromatic)

        phase: LUT of phase function
               (can be read from file with read_phase)

        Example: AeroOPAC('maritime_clean', 0.1, 550.).calc(400.)
    '''
    def __init__(self, filename, tau_ref, w_ref, zmin=None, zmax=None, ssa=None, phase=None):
        self.tau_ref = tau_ref
        self.w_ref = w_ref
        self.reff = None
        self._phase = phase

        if ssa is None:
            self.ssa = None
        else:
            self.ssa = np.array(ssa)

        if dirname(filename) == '':
            self.filename = join(dir_libradtran_opac, 'standard_aerosol_files', filename)
        else:
            self.filename = filename
        if not filename.endswith('.dat'):
            self.filename += '.dat'

        self.basename = basename(self.filename)
        if self.basename.endswith('.dat'):
            self.basename = self.basename[:-4]

        assert exists(self.filename), '{} does not exist'.format(self.filename)

        #
        # read list of species
        #
        species = None
        for line in open(self.filename,'r').readlines():
            if 'z(km)' in line:
                species = line.split()[2:]
                break
        assert species is not None

        #
        # load species properties
        #
        self.species = []
        for s in species:
            self.species.append(Species(s+'.mie'))

        #
        # read profile of mass concentrations
        #
        data = np.loadtxt(self.filename)

        self.zopac = data[::-1,0]   # altitude in km (increasing)

        if zmin is None:
            zmin = self.zopac[0]
        if zmax is None:
            zmax = self.zopac[-1]

        # scale zopac between zmin and zmax
        self.zopac = zmin + (zmax-zmin)*(self.zopac - self.zopac[0])/(self.zopac[-1] - self.zopac[0])

        self.densities = data[::-1,1:]  # vertical profile of mass concentration (g/m3)
                                        # (zopac, species)

    def set_densities(self, Z, densities, species=None) :
        '''
        assign densities vertical profiles of each specie in the list
        Arguments:
            Z: altitude in km (M) (increasing order)
            densities : array of specie density profile (MxN) in g/m3
            species: list of  OPAC species name (N), if none self.species is used
        '''
        if species is not None:
            self.species = []
            for s in species:
                self.species.append(Species(s+'.mie'))
        assert len(self.species) == densities.shape[1]

        self.zopac = Z
        self.densities = densities


    def dtau_ssa(self, wav, Z, rh=None):
        '''
        returns a profile of optical thickness and single scattering albedo at
        each wavelength
        (N wavelengths x M layers)

        Arguments:
            wav: wavelength in nm (N)
                 (scalar or array)
            Z: altitude in km (M)
            rh: relative humidity (M)
        '''

        assert Z.shape == rh.shape
        dtau = 0.
        dtau_ref = 0.
        ssa = 0.
        dZ = -diff1(Z)
        w0 = np.array([self.w_ref], dtype='float32')
        for i, s in enumerate(self.species):
            # integrate density along altitude
            dens = trapzinterp(
                    self.densities[:,i],
                    self.zopac, Z
                    )
            ext, ssa_ = s.ext_ssa(wav, rh, reff=self.reff)
            dtau_ = ext * dens * dZ
            dtau += dtau_

            ssa += dtau_ * ssa_

            ext, ssa_ = s.ext_ssa(w0, rh, reff=self.reff)
            dtau_ref += ext * dens * dZ

        ssa[dtau!=0] /= dtau[dtau!=0]

        # apply scaling factor to get the required optical thickness at the
        # specified wavelength
        dtau *= self.tau_ref/np.sum(dtau_ref)

        # force ssa
        if self.ssa is not None:
            if self.ssa.ndim == 0: # scalar
                ssa[:,:] = self.ssa
            else:
                ssa[:,:] = self.ssa[:,None]

        return dtau, ssa

    def phase(self, wav, Z, rh=None, NBTHETA=721, phaseOpti=False):
        '''
        Phase function calculation at wavelength wav and altitudes Z
        relative humidity is rh
        angle resampling over NBTHETA angles
        '''

        if self._phase is not None:
            if self._phase.ndim == 2:
                # convert to 4-dim by inserting empty dimensions wav_phase
                # and z_phase
                assert self._phase.names == ['stk', 'theta_atm']
                pha = LUT(self._phase.data[None,None,:,:],
                          names = ['wav_phase', 'z_phase'] + self._phase.names,
                          axes = [np.array([wav[0]]), np.array([0.])] + self._phase.axes,
                         )

                return pha
            else:
                return self._phase


        P = 0.
        dssa = 0.
        dtau = 0.
        
        
        for i, s in enumerate(self.species):
            # integrate density along altitude
            dens = trapzinterp(
                    self.densities[:,i],
                    self.zopac, Z)

            # optical properties of the current species
            ext, ssa = s.ext_ssa(wav, rh, reff=self.reff)
            dtau_ = ext * dens * (-diff1(Z))
            dtau += dtau_

            dssa_ = dtau_*ssa  # NLAM, ALTITUDE
            dssa_ = dssa_[:,1:,None,None]
            dssa += dssa_
            if not phaseOpti:
                P += s.phase(wav, rh, NBTHETA, reff=self.reff)*dssa_  # (NLAM, ALTITUDE-1, NPSTK, NBTHETA)
            else:
                P += s.phaseOpti(wav, rh, NBTHETA, reff=self.reff)*dssa_

        with np.errstate(divide='ignore'):
            P.data /= dssa
        P.data[np.isnan(P.data)] = 0.

        P.axes[1] = Z[1:]
    
        return P

    @staticmethod
    def list():
        '''
        list standard aerosol files in opac
        '''
        files = glob(join(dir_libradtran_opac, 'standard_aerosol_files', '*.dat'))
        return map(lambda x: basename(x)[:-4], files)


class CloudOPAC(AeroOPAC):
    '''
    Single species, localized between zmin and zmax,
    with and effective radius reff

    wav_clip: if True, don't raise Error upon interpolation error in
    wavelength, use the extrema values

    Example: CloudOPAC('wc.sol', 12.68, 2, 3, 10., 550.)
             # water cloud mie, reff=12.68 between 2 and 3 km
             # total optical thickness of 10 at 550 nm
    '''
    def __init__(self, species, reff, zmin, zmax, tau_ref, w_ref,
                 phase=None, wav_clip=False):
        self.reff = reff
        self.tau_ref = tau_ref
        self.w_ref = w_ref
        self.species = [Species(species+'.mie', wav_clip=wav_clip)]
        self.zopac     = np.array([zmax, zmax, zmin, zmin, 0.], dtype='f')
        self.densities = np.array([  0.,   1.,   1.,   0., 0.], dtype='f')[:,None]
        self.ssa = None
        self._phase = phase


class CompOPAC(AeroOPAC):
    '''
    Single species, localized using a profile density, z,
    and a RH profile

    wav_clip: if True, don't raise Error upon interpolation error in
    wavelength, use the extrema values

    Example: CompOPAC('inso.mie', atm.prof.RH(), density, atm.prof.z, 10., 550.)
             # total optical thickness of 10 at 550 nm
    '''
    def __init__(self, species, rh, density, z, tau_ref, w_ref,
                 phase=None, wav_clip=False):
        self.reff = None
        self.rh = rh
        self.tau_ref = tau_ref
        self.w_ref = w_ref
        self.species = [Species(species+'.mie', wav_clip=wav_clip)]
        self.zopac     =  z
        self.densities = density[:,None]
        self.ssa = None
        self._phase = phase


class CompUser(object):
    '''
    Single species, localized using a profile density, z

    wav_clip: if True, don't raise Error upon interpolation error in
    wavelength, use the extrema values

    Example: CompUser(species, density, atm.prof.z, 10., 550.)
             # total optical thickness of 10 at 550 nm
    '''
    def __init__(self, species, density, z, tau_ref, w_ref, phase=None, ssa=None, wav_clip=False):
        self.tau_ref = tau_ref
        self.w_ref = w_ref
        self.z     =  z
        self.zopac     =  z
        self.densities = density[:,None]/np.trapz(density, x=-z)
        self._phase = phase
        self.species=[species]
        self.ssa = ssa


    def dtau_ssa(self, wav, Z, rh):
        '''
        returns a profile of optical thickness and single scattering albedo at
        each wavelength
        (N wavelengths x M layers)

        Arguments:
        wav: wavelength in nm (N)
             (scalar or array)
        Z: altitude in km (M)
        '''
        dtau = 0.
        dtau_ref = 0.
        ssa = 0.
        dZ = -diff1(Z)
        w0 = np.array([self.w_ref], dtype='float32')
        for i, s in enumerate(self.species):
            # integrate density along altitude
            dens = trapzinterp(
                    self.densities[:,i],
                    self.zopac, Z
                    )
            ext, ssa_ = s.ext_ssa(wav)
            dtau_ = ext * dens * dZ
            dtau += dtau_

            ssa += dtau_ * ssa_

            ext, ssa_ = s.ext_ssa(w0)
            dtau_ref += ext * dens * dZ

        ssa[dtau!=0] /= dtau[dtau!=0]

        # apply scaling factor to get the required optical thickness at the
        # specified wavelength
        dtau *= self.tau_ref/np.sum(dtau_ref)

        # force ssa
        if self.ssa is not None:
            if self.ssa.ndim == 0: # scalar
                ssa[:,:] = self.ssa
            else:
                ssa[:,:] = self.ssa[:,None]

        return dtau, ssa


    def phase(self, wav, Z, rh, NBTHETA=721, phaseOpti=False):
        '''
        Phase function calculation at wavelength wav and altitudes Z
        relative humidity is rh
        angle resampling over NBTHETA angles
        '''

        if self._phase is not None:
            if self._phase.ndim == 2:
                # convert to 4-dim by inserting empty dimensions wav_phase
                # and z_phase
                assert self._phase.names == ['stk', 'theta_atm']
                pha = LUT(self._phase.data[None,None,:,:],
                          names = ['wav_phase', 'z_phase'] + self._phase.names,
                          axes = [np.array([wav[0]]), np.array([0.])] + self._phase.axes,
                         )

                return pha
            else:
                return self._phase

        P = 0.
        dssa = 0.
        dtau = 0.

        for i, s in enumerate(self.species):
            # integrate density along altitude
            dens = trapzinterp(
                    self.densities[:,i],
                    self.zopac, Z)

            # optical properties of the current species
            ext, ssa = s.ext_ssa(wav)
            dtau_ = ext * dens * (-diff1(Z))
            dtau += dtau_

            dssa_ = dtau_*ssa  # NLAM, ALTITUDE
            dssa_ = dssa_[:,1:,None,None]
            dssa += dssa_

            P += s.phase(wav, NBTHETA)*dssa_  # (NLAM, ALTITUDE-1, NPSTK, NBTHETA)

        with np.errstate(divide='ignore'):
            P.data /= dssa
        P.data[np.isnan(P.data)] = 0.

        P.axes[1] = (Z[1:]+Z[:-1])/2.
        return P



class Atmosphere(object):
    ''' Base class for atmosphere '''
    pass


class AtmAFGL(Atmosphere):
    '''
    Atmospheric profile definition using AFGL data

    Arguments:
        - atm_filename AFGL atmosphere file
          if provided without a directory, use default directory dir_libradtran_atmmod
          atmosphere files should be:
            'afglms', 'afglmw', 'afglss', 'afglsw', 'afglt',
            'afglus', 'afglus_ch4_vmr', 'afglus_co_vmr', 'afglus_n2_vmr',
            'afglus_n2o_vmr', 'afglus_no2', 'mcclams', 'mcclamw'

   Keywords:
        - comp: list of components particles objects (aerosol, clouds)
        - grid: new grid altitudes (list of decreasing altitudes in km), if None, the default AFGL grid is kept
        - lat: latitude (for Rayleigh optical depth calculation, default=45.)
        - P0: Sea surface pressure (default: SSP from AFGL)
        Gaseous absorption:
        - O3: total ozone column (Dobson units),
          or None to use atmospheric profile value (default)
        - H2O: total water vapour column (g.cm-2), or None to use atmospheric
          profile value (default)
        - NO2: activate NO2 absorption (default True)
        - O3_H2O_alt : altitude of H2O and O3 values, by default None and scale from z=0km
        - tauR: Rayleigh optical thickness, default None: computed
          from atmospheric profile and wavelength

        User specified optical properties:
            One can specify directly the optical properties of the medium:
            in 1D it corresponds to vertical profile
            in 3D is is just an optical properties index, it must be completed by the cells grid
        - prof_abs: the gaseous absorption optical thickness profile  provided by user
                    if directly used, it shortcuts any further gaseous absorption computation
                    array of dimension (NWavelength,NZ)
        - prof_ray: the rayleigh scattering optical thickness profile  provided by user
                    if directly used, it shortcuts any further rayleigh scattering computation
                    array of dimension (NWavelength,NZ)
        - prof_aer: a tuple (ext,ssa) the aerosol extinction optical thickness profile and single scattering albedo arrays  
                    provided by user, each array has dimensions (NWavelength,NZ)
                    if directly used, it shortcuts any further particles scattering computation
        - prof_phases: a tuple (iphase, phases ) where iphase is the phase matrix indices profile (NWavelength,NZ), 
                       and  phases is a list of phase matrices LUT (as outputs of the 'read_phase' utility) 
        - RH_cst :  force relative humidity o be constant, default (None, recalculated)

        Phase functions definition:
        - pfwav: a list of wavelengths over which the phase functions are calculated
          default: None (all wavelengths)
        - pfgrid: altitude grid over which the phase function is calculated
          can be provided as an array of decreasing altitudes or a gridspec
          default value: [100, 0]

        3D:
        - if cells is present: then atmosphere is 3D
           Cells represent the 3D definition of  atmosphere
           1) 'iopt' gives the number of the optical property corresponding to the cells; iopt(Ncell)
           2) 'iabs' gives the number of the absorption property corresponding to the cells; iabs(Ncell)
           3) Bounding Boxes(1 Point Bottom Left pmin, 1 Point Top Right pmax) of the cells; pmin(3,Ncell); pmax(3,Ncell)
           and 6 neighbours index (positive X, negative X, positive Y, negative Y, positive Z, negative Z); neighbour(6,Ncell)
           it returns coefficients in (km-1) instead of optical thicknesses
             
   Outputs:
        By default return vertically integrated optical thicknesses from TOA
    '''
    def __init__(self, atm_filename, comp=[],
                 grid=None, lat=45.,
                 P0=None, O3=None, H2O=None, NO2=True,
                 O3_H2O_alt=None,
                 tauR=None,
                 pfwav=None, pfgrid=[100., 0.], prof_abs=None,
                 prof_ray=None, prof_aer=None, prof_phases=None,
                 RH_cst=None, US=True,
                 cells=None):

        self.lat = lat
        self.comp = comp
        self.pfwav = pfwav
        self.pfgrid = np.array(pfgrid)
        self.prof_abs = prof_abs
        self.prof_ray = prof_ray
        self.prof_aer = prof_aer
        self.prof_phases = prof_phases
        self.RH_cst = RH_cst
        self.US = US
        self.OPT3D = cells is not None
        if self.OPT3D : self.cells = cells

        self.tauR = tauR
        if tauR is not None:
            self.tauR = np.array(tauR)

        assert (np.diff(pfgrid) < 0.).all()

        #
        # init directories
        #
        # TODO Trick bellow to improve
        if atm_filename == "ATM3D":
            Nopt = grid.size
            atm_arr = np.zeros((Nopt,9))
            atm_arr[:,0] = np.arange(Nopt)[::-1]
            np.savetxt('./tmp.dat', atm_arr)
            atm_filename = "./tmp.dat"

        if dirname(atm_filename) == '':
            atm_filename = join(dir_libradtran_atmmod, atm_filename)
        if (not exists(atm_filename)) and (not atm_filename.endswith('.dat')):
            atm_filename += '.dat'

        crs_O3_filename  = join(dir_libradtran_crs, 'crs_O3_UBremen_cf.dat')
        crs_NO2_filename = join(dir_libradtran_crs, 'crs_NO2_UBremen_cf.dat')
        
        # read crs ozone file
        #_crs_chappuis = np.loadtxt(crs_O3_filename, comments="#")
        _crs_chappuis = pd.read_csv(crs_O3_filename, comment="#", header=None, sep=r'\s+', dtype=float).values
        #_crs_chappuis = np.genfromtxt(crs_O3_filename, comments="#")
        self.crs_chappuis = LUT(
                _crs_chappuis[:,1:],
                axes=[_crs_chappuis[:,0], None],
                names=['wavelength', None],
                )

        # read crs no2 file
        #_crs_no2 = np.loadtxt(crs_NO2_filename, comments="#")
        #_crs_no2 = np.genfromtxt(crs_NO2_filename, comments="#")
        _crs_no2 = pd.read_csv(crs_NO2_filename, comment="#", header=None, sep=r'\s+', dtype=float).values
        self.crs_no2 = LUT(
                _crs_no2[:,1:],
                axes=[_crs_no2[:,0], None],
                names=['wavelength', None],
                )
        self.crs_no2

        # read afgl file
        prof = Profile_base(atm_filename, O3=O3,
                            H2O=H2O, NO2=NO2, P0=P0, RH_cst=RH_cst, US=US, O3_H2O_alt=O3_H2O_alt
                            )

        #
        # regrid profile if required
        #
        if grid is None:
            self.prof = prof
        else:
            if isinstance(grid, str):
                grid = change_altitude_grid(prof.z, grid)
            self.prof = prof.regrid(np.array(grid))

        #
        # calculate reduced profile
        # (for phase function blending)
        #
        self.prof_red = prof.regrid(pfgrid)



    def calc(self, wav, phase=True, NBTHETA=721, phaseOpti=False):
        '''
        Profile and phase function calculation at bands wav

        phase: boolean (activate phase function calculation)

        Returns: profile + phase function MLUT
        '''
        
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)
            
        profile = self.profile(wav)
        
        if phase:
            if self.pfwav is None:
                wav_pha = wav[:]
            else:
                wav_pha = self.pfwav
            pha = self.phase(wav_pha, NBTHETA=NBTHETA, phaseOpti=phaseOpti)
            
            if pha is not None:
                pha_, ipha = calc_iphase(pha, profile.axis('wavelength'), profile.axis('z_atm'))
                profile.add_axis('theta_atm', pha.axes[-1])
                profile.add_dataset('phase_atm', pha_, ['iphase', 'stk', 'theta_atm'])
                if not self.OPT3D:
                    profile.add_dataset('iphase_atm', ipha, ['wavelength', 'z_atm'])
                else :
                    profile.add_dataset('iphase_atm', ipha, ['wavelength', 'iopt'])
        return profile


    def profile(self, wav, prof=None):
        '''
        Calculate the profile of optical properties at given wavelengths
        wav: array of wavelength in nm
        prof: profile of densities (default: self.prof)

        returns: the profile of optical properties
        '''
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)

        if prof is None:
            prof = self.prof

        dz = -diff1(prof.z)

        pro = MLUT()
        pro.add_axis('z_atm', prof.z)
        pro.add_axis('wavelength', wav[:])

        # refractive index
        n = refractivity(wav[:]*1e-3, prof.P, prof.T,prof.dens_co2/prof.dens_air*1e6)
        pro.add_dataset('n_atm', n, axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'atmospheric refractive index'})


        pro.add_dataset('T_atm', self.prof.T, axnames=['z_atm'],
                        attrs={'description':
                               'temperature (K)'})
        
        #
        # Rayleigh optical thickness
        #
        # cumulated Rayleigh optical thickness (wav, z)
        if self.prof_ray is None :
            tauray = rod(wav[:]*1e-3, prof.dens_co2/prof.dens_air*1e6, self.lat,
                     prof.z*1e3, prof.P)
            dtaur  = diff1(tauray, axis=1)
        else : 
            dtaur = self.prof_ray
            tauray= np.cumsum(dtaur,axis=1)

        if self.tauR is not None:
            # scale Rayleigh optical thickness
            if self.tauR.ndim == 1:
                # for each wavelength
                tauray *= self.tauR[:,None]/tauray[:,-1:]
            else:
                # scalar
                tauray *= self.tauR/tauray[:,-1:]

        assert tauray.ndim == 2

        # Rayleigh optical thickness
        dtaur = diff1(tauray, axis=1)
        if not self.OPT3D : 
            pro.add_dataset('OD_r', tauray, axnames=['wavelength', 'z_atm'],
            attrs={'description':
            'Cumulated rayleigh optical thickness'})
        else:
            if self.prof_ray is None:
                ray_coef = abs(dtaur/dz)
                ray_coef[~np.isfinite(ray_coef)] = 0.
            else:
                ray_coef = self.prof_ray
            pro.add_dataset('OD_r', ray_coef, axnames=['wavelength', 'iopt'],
            attrs={'description':
            'rayleigh scattering coefficient (km-1)'})

        #
        # Aerosol optical thickness and single scattering albedo
        #
        if self.prof_aer is None :
            dtaua = np.zeros((len(wav), len(prof.z)), dtype='float32')
            ssa_p = np.zeros((len(wav), len(prof.z)), dtype='float32')
            for comp in self.comp:
                dtau_, ssa_ = comp.dtau_ssa(wav[:], prof.z, prof.RH())
                dtaua += dtau_
                ssa_p+= dtau_ * ssa_
            ssa_p[dtaua!=0] /= dtaua[dtaua!=0]
            ssa_p[dtaua==0] = 1.
            taua = np.cumsum(dtaua, axis=1)

        else:
            (dtaua, ssa_p) = self.prof_aer
            taua= np.cumsum(dtaua,axis=1)

        if not self.OPT3D : 
            pro.add_dataset('OD_p', taua,
            axnames=['wavelength', 'z_atm'],
            attrs={'description':
            'Cumulated particles optical thickness at each wavelength'})
        else:
            if self.prof_aer is None:
                aer_coef = abs(dtaua/dz)
                aer_coef[~np.isfinite(aer_coef)] = 0.
            else : (aer_coef, ssa_p) = self.prof_aer
            pro.add_dataset('OD_p', aer_coef,
            axnames=['wavelength', 'iopt'],
            attrs={'description':
            'particles extinction coefficient (km-1)'})

        if not self.OPT3D:
            pro.add_dataset('ssa_p_atm', ssa_p, axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Particles single scattering albedo of the layer'})
        else :
            pro.add_dataset('ssa_p_atm', ssa_p, axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'Particles single scattering albedo of the layer'})


            
        if self.prof_abs is None:
            #
            # Ozone optical thickness
            #
            
            T0 = 273.15  # in K
            T = LUT(prof.T, names=['temperature'])
            tau_o3  = self.crs_chappuis.sub()[Idx(wav, fill_value='extrema'), 0]
            tau_o3 += self.crs_chappuis.sub()[Idx(wav, fill_value='extrema'), 1]*(T - T0)
            tau_o3 += self.crs_chappuis.sub()[Idx(wav, fill_value='extrema'), 2]*(T - T0)*(T - T0)

            # LUT in 10^(-20) cm2, convert in km-1
            tau_o3 *= prof.dens_o3 * 1e-15
            tau_o3 *= dz
            if not (tau_o3.data >= 0).all():
                warn('Negative values in tau_o3 ({}%, min value is {}, set to 0)'.format(
                    100.*np.sum(tau_o3.data<0)/float(tau_o3.data.size),
                    tau_o3.data[tau_o3.data == np.amin(tau_o3.data)]
                    #tau_o3.data[tau_o3.data == np.amin(tau_o3.data)][0]
                    ))
            tau_o3.data[tau_o3.data < 0] = 0

            #
            # NO2 optical thickness
            #
            tau_no2  = self.crs_no2.sub()[Idx(wav, fill_value='extrema'), 0]
            tau_no2 += self.crs_no2.sub()[Idx(wav, fill_value='extrema'), 1]*(T - T0)
            tau_no2 += self.crs_no2.sub()[Idx(wav, fill_value='extrema'), 2]*(T - T0)*(T - T0)

            tau_no2 *= prof.dens_no2 * 1e-15
            tau_no2 *= dz
            # if not (tau_no2.data >= 0).all():
            #     warn('Negative values in tau_no2 ({}%, min value is {}, set to 0)'.format(
            #         100.*np.sum(tau_no2.data<0)/float(tau_no2.data.size),
            #         tau_no2.data[tau_no2.data == np.amin(tau_no2.data)][0]
            #         ))
            tau_no2.data[tau_no2.data < 0] = 0

            #
            # other gases (reptran)
            #
            if wav.use_reptran_kdis:
                tau_mol = wav.calc_profile(self.prof) * dz
                # If not reptran (i.e. Kdis case) we set 03 and NO2 to 0 (already calculated in Kdis)
                if not (str(wav.type_wav) == "<class 'smartg.reptran.REPTRAN_IBAND'>"):
                    tau_o3.data[:] = 0.
                    tau_no2.data[:] = 0.
            else:
                tau_mol = np.zeros((len(wav), len(prof.z)), dtype='float32') * dz
                
            #
            # Total gaseous optical thickness
            #
            dtaug = tau_o3 + tau_no2 + tau_mol
            taug = dtaug.apply(lambda x: np.cumsum(x, axis=1))
            #taug.attrs['description'] = 'Cumulated gaseous absorption optical thickness'
            #pro.add_lut(taug, desc='OD_g')
            if not self.OPT3D:
                pro.add_dataset('OD_g', taug.data,
                axnames=['wavelength', 'z_atm'],
                attrs={'description': 'Cumulated gaseous absorption optical thickness'})
            else:
                abs_coef = abs(dtaug.data/dz)
                abs_coef[~np.isfinite(abs_coef)] = 0.
                pro.add_dataset('OD_g', abs_coef, axnames=['wavelength', 'iopt'],
                  attrs={'description':
                         'gaseous absorption coefficient (km-1)'})

        else:
            dtaug = self.prof_abs
            taug  = np.cumsum(dtaug,axis=1)
            if not self.OPT3D:
                pro.add_dataset('OD_g', taug, axnames=['wavelength', 'z_atm'],
                  attrs={'description':
                         'Cumulated gaseous absorption optical thickness'})

            else: 
                abs_coef = self.prof_abs
                pro.add_dataset('OD_g', abs_coef, axnames=['wavelength', 'iopt'],
                  attrs={'description':
                         'gaseous absorption coefficient (km-1)'})

                
        #
        # Total optical thickness and other parameters
        #
        if not self.OPT3D:
            tau_tot = tauray + taua + taug[:,:]
            pro.add_dataset('OD_atm', tau_tot,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Cumulated extinction optical thickness'})

            tau_sca = np.cumsum(dtaur + dtaua*ssa_p, axis=1)
            pro.add_dataset('OD_sca_atm', tau_sca,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Cumulated scattering optical thickness'})

            tau_abs = np.cumsum(dtaug[:,:] + dtaua*(1-ssa_p), axis=1)
            pro.add_dataset('OD_abs_atm', tau_abs,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Cumulated absorption optical thickness'})

            with np.errstate(invalid='ignore', divide='ignore'):
                ssa = (dtaur+ dtaua*ssa_p)/diff1(tau_tot, axis=1)
            ssa[np.isnan(ssa)] = 1.
            pro.add_dataset('ssa_atm', ssa,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Single scattering albedo of the layer'})


        else:
            tot_coef = ray_coef + aer_coef + abs_coef[:,:]
            pro.add_dataset('OD_atm', tot_coef,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'extinction coefficient (km-1)'})

            sca_coef = ray_coef + aer_coef*ssa_p
            pro.add_dataset('OD_sca_atm', sca_coef,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'scattering coefficient (km-1)'})

            tabs_coef = abs_coef + aer_coef*(1.-ssa_p)
            pro.add_dataset('OD_abs_atm', tabs_coef,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'total absorption coefficient (km-1)'})

            with np.errstate(invalid='ignore', divide='ignore'):
                ssa = (ray_coef+ aer_coef*ssa_p)/tot_coef
            ssa[np.isnan(ssa)] = 1.
            pro.add_dataset('ssa_atm', ssa,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'Single scattering albedo of the layer'})

        with np.errstate(invalid='ignore', divide='ignore'):
            pmol = dtaur/(dtaur + dtaua*ssa_p)
        pmol[np.isnan(pmol)] = 1.
        if not self.OPT3D:
            pro.add_dataset('pmol_atm', pmol,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Ratio of molecular scattering to total scattering of the layer'})
        else :
            pro.add_dataset('pmol_atm', pmol,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'Ratio of molecular scattering to total scattering of the layer'})

            
        pine = np.zeros_like(ssa)
        FQY1 = np.zeros_like(ssa)
        if not self.OPT3D:
            pro.add_dataset('pine_atm', pine,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'fraction of inelastic scattering of the layer'})
            pro.add_dataset('FQY1_atm', FQY1,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'fluoresence quantum yield of the layer'})
        else :
            pro.add_dataset('pine_atm', pine,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'fraction of inelastic scattering of the layer'})
            pro.add_dataset('FQY1_atm', FQY1,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'fluoresence quantum yield of the layer'})


        if self.prof_phases is not None:
            ipha, phases = self.prof_phases
            if not self.OPT3D:
                pro.add_dataset('iphase_atm', ipha, axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'index of phase matrix'})
            else :
                pro.add_dataset('iphase_atm', ipha, axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'index of phase matrix'})

            # set the number of scattering angles to the maximum
            ip  = np.array([p.axis('theta_atm').size for p in phases]).argmax()
            theta = phases[ip].axis('theta_atm')
            pha = np.stack([p[:,Idx(theta)] for p in phases])
            pro.add_axis('theta_atm', theta)
            pro.add_dataset('phase_atm', pha, axnames=['iphase', 'stk', 'theta_atm'],
                    attrs={'description':
                           'phase matrices'})
        # Pure 3D
        #
        if self.OPT3D:
            (iopt, iabs, pmin, pmax, neighbour) = self.cells
            pro.add_dataset('iopt_atm', iopt, axnames=['icell'])
            pro.add_dataset('iabs_atm', iabs, axnames=['icell'])
            pro.add_dataset('pmin_atm', pmin, axnames=['xyz', 'icell'])
            pro.add_dataset('pmax_atm', pmax, axnames=['xyz', 'icell'])
            pro.add_dataset('neighbour_atm', neighbour, axnames=['faces', 'icell'])

        return pro


    def phase(self, wav, NBTHETA=721, phaseOpti=False):
        '''
        Phase functions calculation at bands, using reduced profile
        '''
        wav = np.array(wav)
        if wav.ndim == 0:
            wav = wav.reshape(1)
        pha = 0.
        norm = 0.
        rh = self.prof_red.RH()

        for comp in self.comp:
            dtau, ssa_p = comp.dtau_ssa(wav, self.pfgrid, rh=rh)
            dtau = dtau[:,1:][:,:,None,None]
            ssa_p = ssa_p[:,1:][:,:,None,None]
            pha += comp.phase(wav, self.pfgrid, rh, NBTHETA=NBTHETA, phaseOpti=phaseOpti)*dtau*ssa_p
            norm += dtau*ssa_p
        if len(self.comp) > 0:
            pha /= norm
            pha.data[np.isnan(pha.data)] = 0.

            return pha
        else:
            return None

    def calc_split(self, wav, phase=True, NBTHETA=721):
        '''
        compute optical properties and return the vertical profiles
        prof_abs, prof_ray, prof_aer, and prof_phases the alternative inputs of the AtmAFGL
        '''
        pro = self.calc(wav=wav, phase=phase, NBTHETA=NBTHETA)
        pro_aer = diff1(pro['OD_p'].data.astype(np.float32), axis=1)
        ssa_aer = pro['ssa_p_atm'].data
        pro_ray = diff1(pro['OD_r'].data.astype(np.float32), axis=1)
        pro_abs = diff1(pro['OD_g'].data.astype(np.float32), axis=1)
        pro_iphase = pro['iphase_atm'].data
        pro_phases = [pro['phase_atm'].sub({'iphase':i}) for i in range(pro_iphase.max()+1)]

        return pro_abs, pro_ray, (pro_aer, ssa_aer), (pro_iphase, pro_phases)


def read_phase(filename, standard=False, kind='atm'):
    '''
    Read phase function from filename as a LUT

    standard: standard phase function definition, otherwise Smart-g definition
    '''
    data2 = pd.read_csv(filename, sep=r'\s+', header=None)

    theta = np.array(data2[0])
    pha   = np.array(data2[[1,2,3,4]])

    if standard:
        pha[:,0] = data2[1] + data2[2]
        pha[:,1] = data2[1] - data2[2]
        pha[:,2] = data2[3]
        pha[:,3] = data2[4]

    # Normalization to Sum_-1_+1 P(mu) dmu = 2.
    f = (pha[:,0] + pha[:,1])/2.
    mu= np.cos(np.radians(theta))
    Norm = np.trapz(f,-mu)
    pha *= (2./abs(Norm))

    P = LUT(pha.swapaxes(0, 1),  # stk, theta
            axes=[None, theta],
            names=['stk', 'theta_'+kind],
           )

    return P


def trapzinterp(y, x, xnew, samesize=True):
    '''
    integrate y(x) using the composite trapezoidal rule, interpolated on a new grid xnew
    if samesize: returns an array of same size as xnew, whose first element is y[xnew[0]]
    otherwise, returns an array of size len(xnew)-1
    '''
    # revert x and y such that x be increasing
    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]

    # y values in the new grid
    ynew = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0.)(xnew)
    #ynew = np.interp(xnew, x, y, left=0., right=0.)
                     
    # indices of xnew in x
    idx = np.searchsorted(x, xnew)

    # for every interval of the new grid
    nnew= len(xnew)
    integ = np.array([], dtype='f')
    if samesize:
        integ = np.append(integ, ynew[0])
    for i in xrange(nnew-1):

        i1, i2 = idx[i], idx[i+1]

        if i1 <= i2:
            xx = x[i1:i2]
            yy = y[i1:i2]
        else:
            xx = x[i2:i1][::-1]
            yy = y[i2:i1][::-1]

        xx = np.insert(xx, 0, xnew[i])
        xx = np.append(xx, xnew[i+1])

        yy = np.insert(yy, 0, ynew[i])
        yy = np.append(yy, ynew[i+1])

        integ = np.append(integ, np.trapz(yy, x=xx)/(xnew[i+1] - xnew[i]))

    return integ



class Profile_base(object):
    '''
    Profile of physical properties
    - atm_filename: AFGL filename
    - O3: total ozone column (Dobson units),
      or None to use atmospheric profile value (default)
    - H2O: total water vapour column (g.cm-2), or None to use atmospheric
      profile value (default)
    - P0: sea surface pressure (hPa)
    - RH_cst: force Relative humidity to be constant, (defualt recalculated)
    '''
    def __init__(self, atm_filename, O3=None, H2O=None, NO2=True, P0=None, RH_cst=None, US=True, O3_H2O_alt=None):

        if atm_filename is None:
            return
        self.atm_filename = atm_filename
        with open(atm_filename) as f:
            lines = f.readlines()

        desc = None
        desc = ''
        n=0
        for line in lines:
            if ('z(km)' in line) and ('p(mb)' in line) and ('T(K)' in line) and ('air(cm-3)' in line) :
                desc = line
                break
            else:
                n+=1
        if desc=='' : n = 0

        if desc is not None:
            #data = np.loadtxt(atm_filename, dtype=np.float32, comments="#", skiprows=n)
            data = pd.read_csv(atm_filename, comment="#", header=None, sep=r'\s+', dtype=np.float32, skiprows=n).values
            self.z        = data[:,0] # Altitude in km
            self.P        = data[:,1] # pressure in hPa
            self.T        = data[:,2] # temperature in K
            self.dens_air = data[:,3] # Air density in cm-3
            data2 = np.zeros((data.shape[0], 5))
            for i,gas in enumerate(['o3','o2','h2o','co2','no2']):
                try : 
                    ind = desc.split().index(gas+'(cm-3)')
                    data2[:,i] = data[:, ind-1]
                except ValueError:
                    data2[:,i] = 0.
            self.dens_o3  = data2[:,0] # Ozone density in cm-3
            self.dens_o2  = data2[:,1] # O2 density in cm-3
            self.dens_h2o = data2[:,2] # H2O density in cm-3
            self.dens_co2 = data2[:,3] # CO2 densiraise NameError('Invalid atmospheric file format')ty in cm-3
            self.dens_no2 = data2[:,4] # NO2 density in cm-3
        else:
            raise NameError('Invalid atmospheric file format')

        self.RH_cst   = RH_cst

        # scale to specified total O3 content
        if O3 is not None:
            if O3_H2O_alt is None:
                self.dens_o3 *= 2.69e16 * O3 / (simpson(y=self.dens_o3, x=-self.z) * 1e5)
            else:
                f_dens_o3 = interp1d(self.z, self.dens_o3, fill_value='extrapolate')
                z_alt = np.append(self.z[self.z>O3_H2O_alt], O3_H2O_alt)
                dens_o3_alt = f_dens_o3(z_alt)
                o3_afgl = (simpson(dens_o3_alt, -z_alt) * 1e5)/2.69e16
                self.dens_o3 *= O3/o3_afgl
            if O3==0 : self.dens_o3[:] = 0.

        # scale to total H2O content
        if H2O is not None:
            M_H2O = 18.015 # g/mol
            Avogadro = constants.value('Avogadro constant')
            if O3_H2O_alt is None:
                self.dens_h2o *= H2O/ M_H2O * Avogadro / (simpson(y=self.dens_h2o, x=-self.z) * 1e5)
            else:
                f_dens_h2o = interp1d(self.z, self.dens_h2o, fill_value='extrapolate')
                z_alt = np.append(self.z[self.z>O3_H2O_alt], O3_H2O_alt)
                dens_h2o_alt = f_dens_h2o(z_alt)
                h2o_afgl = (simpson(y=dens_h2o_alt, x=-z_alt) * 1e5 * M_H2O)/Avogadro
                self.dens_h2o *= H2O/h2o_afgl
            if H2O==0 : self.dens_h2o[:] = 0.

        if P0 is not None:
            self.P *= P0/self.P[-1]

        if not NO2:
            self.dens_no2[:] = 0.

        #
        # read standard US atmospheres for other gases
        #
        '''
        ch4_filename = join(dir_libradtran_atmmod, 'afglus_ch4_vmr.dat')
        co_filename = join(dir_libradtran_atmmod, 'afglus_co_vmr.dat')
        n2o_filename = join(dir_libradtran_atmmod, 'afglus_n2o_vmr.dat')
        n2_filename = join(dir_libradtran_atmmod, 'afglus_n2_vmr.dat')
        datach4 = np.loadtxt(ch4_filename, comments="#")
        self.dens_ch4 = interp1d(datach4[:,0] , datach4[:,1])(self.z) * self.dens_air # CH4 density en cm-3
        dataco = np.loadtxt(co_filename, comments="#")
        self.dens_co = interp1d(dataco[:,0] , dataco[:,1])(self.z) * self.dens_air # CH4 density en cm-3
        datan2o = np.loadtxt(n2o_filename, comments="#")
        self.dens_n2o = interp1d(datan2o[:,0] , datan2o[:,1])(self.z) * self.dens_air # CH4 density en cm-3
        datan2 = np.loadtxt(n2_filename, comments="#")
        self.dens_n2 = interp1d(datan2[:,0] , datan2[:,1])(self.z) * self.dens_air # CH4 density en cm-3
        '''

        if US:
            ch4_filename = join(dir_libradtran_atmmod, 'afglus_ch4_vmr.dat')
            co_filename = join(dir_libradtran_atmmod, 'afglus_co_vmr.dat')
            n2o_filename = join(dir_libradtran_atmmod, 'afglus_n2o_vmr.dat')
            n2_filename = join(dir_libradtran_atmmod, 'afglus_n2_vmr.dat')
            #datach4 = np.loadtxt(ch4_filename, comments="#")
            datach4 = pd.read_csv(ch4_filename, comment="#", header=None, sep=r'\s+', dtype=float).values
            self.dens_ch4 = interp1d(datach4[:,0] , datach4[:,1])(self.z) * self.dens_air # CH4 density en cm-3
            #self.dens_ch4 = np.interp(self.z, datach4[:,0] , datach4[:,1]) * self.dens_air # CH4 density en cm-3
            #dataco = np.loadtxt(co_filename, comments="#")
            dataco = pd.read_csv(co_filename, comment="#", header=None, sep=r'\s+', dtype=float).values
            self.dens_co = interp1d(dataco[:,0] , dataco[:,1])(self.z) * self.dens_air # CH4 density en cm-3
            #self.dens_co = np.interp(self.z, dataco[:,0] , dataco[:,1]) * self.dens_air # CH4 density en cm-3
            #datan2o = np.loadtxt(n2o_filename, comments="#")
            datan2o = pd.read_csv(n2o_filename, comment="#", header=None, sep=r'\s+', dtype=float).values
            self.dens_n2o = interp1d(datan2o[:,0] , datan2o[:,1])(self.z) * self.dens_air # CH4 density en cm-3
            #self.dens_n2o = np.interp(self.z, datan2o[:,0] , datan2o[:,1]) * self.dens_air # CH4 density en cm-3
            #datan2 = np.loadtxt(n2_filename, comments="#")
            datan2 = pd.read_csv(n2_filename, comment="#", header=None, sep=r'\s+', dtype=float).values
            self.dens_n2 = interp1d(datan2[:,0] , datan2[:,1])(self.z) * self.dens_air # CH4 density en cm-3
            #self.dens_n2 = np.interp(self.z, datan2[:,0] , datan2[:,1]) * self.dens_air # CH4 density en cm-3
            #
            self.dens_so2 = np.zeros_like(self.dens_air)
        else:
            nz = data.shape[0]
            self.dens_ch4 = [0] * nz
            self.dens_co = [0] * nz
            self.dens_n2o = [0] * nz
            self.dens_n2 = [0] * nz
            self.dens_so2 = [0] * nz



    def regrid(self, znew):
        '''
        regrid profile and returns a new profile
        '''

        prof = Profile_base(None)
        z = self.z
        prof.z = znew
        try:
            prof.P = interp1d(z, self.P, bounds_error=False, fill_value=(1012., 1e-5))(znew)
            #prof.P = np.interp(znew, z, self.P, right=1012., left=1e-5)
        except ValueError:
            print('Error interpolating ({}, {}) -> ({}, {})'.format(z[0], z[-1], znew[0], znew[-1]))
            print('atm_filename = {}'.format(self.atm_filename))
            raise
        prof.T = interp1d(z, self.T, fill_value='extrapolate')(znew) # No found np.interp with extrapolate

        prof.dens_air = interp1d(z, self.dens_air, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_o3  = interp1d(z, self.dens_o3, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_o2  = interp1d(z, self.dens_o2, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_h2o = interp1d(z, self.dens_h2o, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_co2 = interp1d(z, self.dens_co2, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_no2 = interp1d(z, self.dens_no2, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_ch4 = interp1d(z, self.dens_ch4, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_co  = interp1d(z, self.dens_co, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_n2o = interp1d(z, self.dens_n2o, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_n2  = interp1d(z, self.dens_n2, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_so2 = interp1d(z, self.dens_so2, bounds_error=False, fill_value=(0., 0.))  (znew)
        
        # prof.dens_air = np.interp(znew, z, self.dens_air, right=0., left=0.)
        # prof.dens_o3  = np.interp(znew, z, self.dens_o3, right=0., left=0.)
        # prof.dens_o2  = np.interp(znew, z, self.dens_o2, right=0., left=0.)
        # prof.dens_h2o = np.interp(znew, z, self.dens_h2o, right=0., left=0.)
        # prof.dens_co2 = np.interp(znew, z, self.dens_co2, right=0., left=0.)
        # prof.dens_no2 = np.interp(znew, z, self.dens_no2, right=0., left=0.)
        # prof.dens_ch4 = np.interp(znew, z, self.dens_ch4, right=0., left=0.)
        # prof.dens_co  = np.interp(znew, z, self.dens_co, right=0., left=0.)
        # prof.dens_n2o = np.interp(znew, z, self.dens_n2o, right=0., left=0.)
        # prof.dens_n2  = np.interp(znew, z, self.dens_n2, right=0., left=0.)
        prof.RH_cst   = self.RH_cst

        return prof

    def RH(self):
        '''
        returns profile of relative humidity for each layer
        '''
        rh = self.dens_h2o/vapor_pressure(self.T)*100.
        if self.RH_cst is not None : rh[:] = self.RH_cst
        return rh


def FN2(lam):
    ''' depolarisation factor of N2
        lam : um
    '''
    return 1.034 + 3.17 *1e-4 *lam**(-2)


def FO2(lam):
    ''' depolarisation factor of O2
        lam : um
    '''
    return 1.096 + 1.385 *1e-3 *lam**(-2) + 1.448 *1e-4 *lam**(-4)

def vapor_pressure(T):
    T0=273.15
    A=T0/T
    Avogadro = constants.value('Avogadro constant')
    M_H2O=18.015
    mh2o=M_H2O/Avogadro
    return A*np.exp(18.916758 - A * (14.845878 + A*2.4918766))/mh2o/1.e6


def Fair(lam, co2):
    ''' depolarisation factor of air for CO2 (N wavelengths x M layers)
        lam : um (N)
        co2 : ppm (M)
    '''
    _FN2 = FN2(lam).reshape((-1,1))
    _FO2 = FO2(lam).reshape((-1,1))
    _CO2 = co2.reshape((1,-1))

    return ((78.084 * _FN2 + 20.946 * _FO2 + 0.934 +
            _CO2*1e-4 *1.15)/(78.084+20.946+0.934+_CO2*1e-4))


def n300(lam):
    ''' index of refraction of dry air  (300 ppm CO2)
        lam : um
    '''
    return 1e-8 * ( 8060.51 + 2480990/(132.274 - lam**(-2)) + 17455.7/(39.32957 - lam**(-2))) + 1.


def n_air(lam, co2):
    ''' index of refraction of dry air (N wavelengths x M layers)
        lam : um (N)
        co2 : ppm (M)
    '''
    N300 = n300(lam).reshape((-1,1))
    CO2 = co2.reshape((1,-1))
    return ((N300 - 1) * (1 + 0.54*(CO2*1e-6 - 0.0003)) + 1.)

def ma(co2):
    ''' molecular volume
        co2 : ppm
    '''
    return 15.0556 * co2*1e-6 + 28.9595

def raycrs(lam, co2):
    ''' Rayleigh cross section (N wavelengths x M layers)
        lam : um (N)
        co2 : ppm ((M)
    '''
    LAM = lam.reshape((-1,1))
    Avogadro = constants.value('Avogadro constant')
    Ns = Avogadro/22.4141 * 273.15/288.15 * 1e-3
    nn2 = n_air(lam, co2)**2
    return (24*np.pi**3 * (nn2-1)**2/(LAM*1e-4)**4/Ns**2/(nn2+2)**2 * Fair(lam, co2))

def g0(lat):
    ''' gravity acceleration at the ground
        lat : deg
    '''
    assert isnumeric(lat)
    return (980.6160 * (1. - 0.0026372 * np.cos(2*lat*np.pi/180.)
            + 0.0000059 * np.cos(2*lat*np.pi/180.)**2))

def g(lat, z) :
    ''' gravity acceleration at altitude z
        lat : deg (scalar)
        z : m
    '''
    assert isnumeric(lat)
    return (g0(lat) - (3.085462 * 1.e-4 + 2.27 * 1.e-7 * np.cos(2*lat*np.pi/180.)) * z
            + (7.254 * 1e-11 + 1e-13 * np.cos(2*lat*np.pi/180.)) * z**2
            - (1.517 * 1e-17 + 6 * 1e-20 * np.cos(2*lat*np.pi/180.)) * z**3)

def rod(lam, co2=400., lat=45., z=0., P=1013.25, pressure='surface'):
    """
    Rayleigh optical depth from Bodhaine et al, 99 (N wavelengths x M layers)
        lam : wavelength in um (N)
        co2 : ppm (M)
        lat : deg (scalar)
        z : altitude in m (M)
        P : pressure in hPa (M)
            (surface or sea-level)
        pressure: str
            - 'surface': P provided at altitude z
            - 'sea-level': P provided at altitude 0
    """
    Avogadro = constants.value('Avogadro constant')
    zs = 0.73737 * z + 5517.56  # effective mass-weighted altitude
    G = g(lat, zs)
    # air pressure at the pixel (i.e. at altitude) in hPa
    if pressure == 'sea-level':
        Psurf = (P * (1. - 0.0065 * z / 288.15) ** 5.255) * 1000.  # air pressure at pixel location in dyn / cm2, which is hPa * 1000
    elif pressure == 'surface':
        Psurf = P * 1000.  # convert to dyn/cm2
    else:
        raise ValueError(f'Invalid pressure type ({pressure})')

    return raycrs(lam, co2) * Psurf * Avogadro/ma(co2)/G

def refractivity(lam,P,T,co2):
    ''' Refractivity of air
        lam : um (N)
        P   : hPa (M)
        T   : K (M)
        co2 : ppm (M)
    '''
    p= P*100.
    t = T-273.15
    Ntp = 1 + (n_air(lam[:],co2) - 1) * p * (1.+p*(60.1-0.972*t)*1e-10)\
        /(96095.43 * (1 + 0.003661 * t))
    return Ntp

def diff1(A, axis=0, samesize=True):
    if samesize:
        B = np.zeros_like(A)
        key = [slice(None)]*A.ndim
        key[axis] = slice(1, None, None)
        B[tuple(key)] = np.diff(A, axis=axis)[:]
        return B
    else:
        return np.diff(A, axis=axis)

def average(A):
    '''
    returns average value within each interval

    A: input array, size N
    returns averaged array of size N-1
    '''
    return 0.5*(A[1:] + A[:-1])


def isiterable(x):
    return hasattr(x, '__iter__')

def isnumeric(x):
    try:
        float(x)
        return True
    except TypeError:
        return False

def get_o2_abs(z, wl, afgl='afglus', DOWNLOAD=False, P0=None, verbose=False, zint=None):
    '''
    return vertical profile of O2 absorption coefficient
    
    Inputs:
        wl : 1D array of wavlength (nm)
        z  : 1D array of altitude in descending order (km)

    Keywords:
        afgl : string describing which AFGL standard atmosphere, default 'afglus'
        DOWNLOAD: Download HITRAN lines parameters, Default False
        P0 : surface pressure (hPa), default None: uses AFGL
        zint : vertical grid to interpolate into, default None, use z


    Outputs:
        2D array (NW, NZ) of absorption coefficient in km-1), (in grid zint if present, otherwise z)
    '''
    import hapi
    from Voigt_gpu import absorptionCoefficient_Voigt_gpu
    #Connect to HITRAN database
    hapi.db_begin('data')
    vmin = 1e7/wl.max()
    vmax = 1e7/wl.min()
    NW   = wl.size
    dv   = (vmax-vmin)/NW
    if DOWNLOAD:
        hapi.fetch('O2i1',7,1,vmin-100,vmax+100)
        hapi.fetch('O2i2',7,2,vmin-100,vmax+100)
        hapi.fetch('O2i3',7,3,vmin-100,vmax+100)  
    if P0 is None : atm = AtmAFGL(afgl, grid=z, O3=0., NO2=False)
    else :          atm = AtmAFGL(afgl, grid=z, O3=0., NO2=False, P0=P0)
    NLE = atm.prof.z.size
    # prepare array for o2 absorption coefficients
    ao2 = np.zeros((NW+5,NLE), dtype=np.float64)
    # compute O2 absorption coefficient with 'GPU' version of HAPI Voigt profile function
    j=0
    for p,t,o2,z in zip(atm.prof.P,atm.prof.T,atm.prof.dens_o2,atm.prof.z):
        nuo2,coefo2 = absorptionCoefficient_Voigt_gpu(SourceTables=['O2i1','O2i2','O2i3'],
            HITRAN_units=True,
            OmegaRange=[vmin,vmax],OmegaStep=dv,GammaL='gamma_self',
            Environment={'p':p/1013.,'T':t})
        ao2[:nuo2.size,j] = coefo2 * o2 * 1e5
        j=j+1
        if verbose : print('level %f completed'%z)
    wlabs=(1e7/nuo2)
    # back to increasing wavelengths
    wlabs = wlabs[::-1]
    ao2   = ao2[:nuo2.size,:]
    ao2   = ao2[::-1,:]
    #interpolation into the solar grid
    ab    = LUT(ao2, axes=[wlabs, atm.prof.z], names=['wavelength', 'z'] )
    
    if zint is None : return ab[Idx(wl, fill_value='extrema'),:]
    else : 
        zint2, wl2 = np.meshgrid(zint, wl)
        return ab[Idx(wl2  , fill_value='extrema'),
                  Idx(zint2, fill_value='extrema')]



def get_co2_abs(z, wl, afgl='afglus', DOWNLOAD=False, P0=None, verbose=False, zint=None):
    '''
    return vertical profile of CO2 absorption coefficient
    
    Inputs:
        wl : 1D array of wavlength (nm)
        z  : 1D array of altitude in descending order (km)

    Keywords:
        afgl : string describing which AFGL standard atmosphere, default 'afglus'
        DOWNLOAD: Download HITRAN lines parameters, Default False
        P0 : surface pressure (hPa), default None: uses AFGL
        zint : vertical grid to interpolate into, default None, use z

    Outputs:
        2D array (NW, NZ) of absorption coefficient in km-1), (in grid zint if present, otherwise z)
    '''
    import hapi
    from Voigt_gpu import absorptionCoefficient_Voigt_gpu
    #Connect to HITRAN database
    hapi.db_begin('data')
    vmin = 1e7/wl.max()
    vmax = 1e7/wl.min()
    NW   = wl.size
    dv   = (vmax-vmin)/NW
    if DOWNLOAD:
        hapi.fetch('CO2i1',2,1,vmin-100,vmax+100)
        hapi.fetch('CO2i2',2,2,vmin-100,vmax+100)
        hapi.fetch('CO2i3',2,3,vmin-100,vmax+100)  
    if P0 is None : atm = AtmAFGL(afgl, grid=z, O3=0., NO2=False)
    else :          atm = AtmAFGL(afgl, grid=z, O3=0., NO2=False, P0=P0)
    NLE = atm.prof.z.size
    # prepare array for co2 absorption coefficients
    aco2 = np.zeros((NW+5,NLE), dtype=np.float64)
    # compute CO2 absorption coefficient with 'GPU' version of HAPI Voigt profile function
    j=0
    for p,t,co2,z in zip(atm.prof.P,atm.prof.T,atm.prof.dens_co2,atm.prof.z):
        nuco2,coefco2 = absorptionCoefficient_Voigt_gpu(SourceTables=['CO2i1','CO2i2','CO2i3'],
            HITRAN_units=True,
            OmegaRange=[vmin,vmax],OmegaStep=dv,GammaL='gamma_self',
            Environment={'p':p/1013.,'T':t})
        aco2[:nuco2.size,j] = coefco2 * co2 * 1e5
        j=j+1
        if verbose : print('level %f completed'%z)
    wlabs=(1e7/nuco2)
    # back to increasing wavelengths
    wlabs = wlabs[::-1]
    aco2   = aco2[:nuco2.size,:]
    aco2   = aco2[::-1,:]
    #interpolation into the solar grid
    ab    = LUT(aco2, axes=[wlabs, atm.prof.z], names=['wavelength', 'z'] )
    
    if zint is None : return ab[Idx(wl, fill_value='extrema'),:]
    else : 
        zint2, wl2 = np.meshgrid(zint, wl)
        return ab[Idx(wl2  , fill_value='extrema'),
                  Idx(zint2, fill_value='extrema')]


def od2k(prof, dataset, axis=1, zreverse=False):
    '''
    From integrated Optical Depth to vertical coefficient in km-1)

    Inputs:
        prof : atmosphere profile (MLUT) as computed by calc method of AtmAFGL
        dataset : name of the dataset to be processed

    Keywords:
        axis : number of the vertical dimension, default 1
        zreverse : invert the vertical axis, default False

    Outputs:
        2D array (NW, NZ) of vertical coefficient (km-1)
    '''
    ot = diff1(prof[dataset].data.astype(np.float32), axis=axis)
    dz = diff1(prof.axis('z_atm')).astype(np.float32)
    k  = abs(ot/dz)
    k[np.isnan(k)] = 0
    sl = slice(None,None,-1 if zreverse else 1)
    
    return k[:,sl]


def BPlanck(wav, T):
    a = 2.0*Planck*speed_of_light**2
    b = Planck*speed_of_light/(wav*Boltzmann*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity


def generatePro_multi(pro_models, b_wav, aots, atm='afglt', factor=None,
                      pfwav=None, P0=None, O3=None, H2O=None, grid=None, O3_H2O_alt=None):
    """
    This function return an atmosphere profile giving a list of several atmosphere profiles. Different profiles can be
    merged together, and a possible mix can be considered by correctly ajusting the aots and factor parameters.

    ===ARGS:
    pro_models : List of atmosphere profiles (from the smartg atmAFGL function)
    b_wav      : Kdis bands or list of wavelenghts
    wl_ref     : Wavelenght used as reference for calculations
    aots       : List of aot of each profile in pro_models at the wavelenght wl_ref
    atm        : The atmAFGL atmosphere used
    factor     : List of factors in order to consider the mixing of models
    pfwav      : List of wavelenghts where the phase functions are computed
    P0         : Surface pressure
    O3         : Scale ozone vertical column (Dobson units)
    H2O        : Scale Water vertical column
    grid       : Shape of vertical atm layers

    ===RETURN:
    pro_atm_tot : A unique atmosphere profile

    ===Exemple:
    pro_modelA = atmAFGL('urban', ...)
    pro_modelB = atmAFGL('desert', ...)
    A,B = functionToGetAB(...)
    proFinal = ([pro_modelA, pro_modelB], b_wav, aots=[0.05, 0.5], atm='afgus', factor=[A, B],
                      pfwav=[550.], P0=1013.25, O3=300., H2O=2., wl_ref=550., grid=None)
    """
    
    if not isinstance(b_wav, BandSet): b_wav_BS = BandSet(b_wav)
    else : b_wav_BS = b_wav
    if (pfwav is None): wav = b_wav_BS
    else: wav = pfwav
    s_z = len(pro_models[0].axis('z_atm')); s_wav=len(wav);
    
    aot_pro = []; assa_pro=[]; Dz=[]; apf=[]; aec_pro=[]
    aec_pro_lut = []; assa_pro_lut=[];assa_pro_bis=[]; saec_pro=[]
    APF_top=[]; APF_bot=[]; assa_top=[]; sumAOT_models=0;
    n_models=len(pro_models)
    if factor is None:
        factor=np.full(n_models, 1, dtype=int)
    
    for j in range (0, n_models):
        sumAOT_models += aots[j]*factor[j]
        aot_pro.append(diff1(pro_models[j][ 'OD_p' ].data*factor[j], axis=1))
        assa_pro.append(pro_models[j][ 'ssa_p_atm' ].data)
        Dz.append(diff1(pro_models[j].axis( 'z_atm' )))
        apf.append(pro_models[j][ 'phase_atm' ].data[ :, None, :, :])
        aec_pro.append(aot_pro[j]/Dz[j])
        assa_pro[j][~np.isfinite( assa_pro[j] )] = 1.; aec_pro[j][~np.isfinite( aec_pro[j] )] = 0.;

        # If pfwav is given this enables to avoid some useless calculations
        if (pfwav is not None):
            aec_pro_lut.append(LUT(aec_pro[j], names=[ 'wav_phase', 'z_phase'], 
                                axes=[ b_wav_BS[:], pro_models[0].axis('z_atm')]))
            assa_pro_lut.append(LUT(assa_pro[j], names=[ 'wav_phase', 'z_phase'], 
                                axes=[ b_wav_BS[:], pro_models[0].axis('z_atm')]))
            aec_pro[j] = np.zeros((s_wav, s_z), dtype=np.float64);
            assa_pro_bis.append(np.zeros((s_wav, s_z), dtype=np.float64));
            for i in range (0, s_wav):
                aec_pro[j][i,:]  = aec_pro_lut[j][Idx(pfwav[i]),:]
                assa_pro_bis[j][i,:]  = assa_pro_lut[j][Idx(pfwav[i]),:]
        else:
            assa_pro_bis.append(assa_pro[j])

        saec_pro.append((aec_pro[j]*assa_pro_bis[j])[:, :, np.newaxis, np.newaxis]) 
        APF_top.append(apf[j]*saec_pro[j])
        APF_bot.append(saec_pro[j])
        assa_top.append(aot_pro[j]*assa_pro[j])
    
    sAPF_top=APF_top[0]; sAPF_bot = APF_bot[0]; sAOT=aot_pro[0]; s_assa_top=assa_top[0]
    if n_models > 1:
        for k in range (0, n_models-1):
            sAPF_top+=APF_top[k+1]
            sAPF_bot+= APF_bot[k+1]
            sAOT+=aot_pro[k+1]
            s_assa_top+=assa_top[k+1]
            
    apf_tot = sAPF_top/sAPF_bot
    apf_tot[np.isnan(apf_tot)] = 0.
    pha_tot = LUT( apf_tot, names=[ 'wav_phase', 'z_phase', 'stk', 'theta_atm'], 
              axes=[ wav[:], pro_models[0].axis('z_atm'), None, pro_models[0].axis('theta_atm') ] )
    # ============================================================
    
    # Compute the AOT and SSA of the mix model
    aot_pro_tot  = sAOT
    aot_pro_tot[~np.isfinite(aot_pro_tot)] = 0.
    assa_pro_tot = s_assa_top/aot_pro_tot
    assa_pro_tot[~np.isfinite(assa_pro_tot)] = 1.
    
    # Create the LUT profile of the mix model
    pro_atm_tot = AtmAFGL(atm, O3=O3, H2O=H2O, P0=P0, grid=grid, pfwav=wav[:], prof_aer=(aot_pro_tot,assa_pro_tot), O3_H2O_alt=O3_H2O_alt).calc(b_wav, phase=False)
    pha_atm, ipha_atm = calc_iphase(pha_tot, b_wav, grid)
    pro_atm_tot.add_axis('theta_atm', pha_tot.axes[-1])
    pro_atm_tot.add_dataset('phase_atm', pha_atm, ['iphase', 'stk', 'theta_atm'])
    pro_atm_tot.add_dataset('iphase_atm', ipha_atm, ['wavelength', 'z_atm'])

    return pro_atm_tot


def compute_AB_coeff(AOT_OBS_wl1, AOT_OBS_wl2, AOT_modelA_wl1, AOT_modelA_wl2, AOT_modelB_wl1, AOT_modelB_wl2):
    """
    This function give the A, B factors to multiply with the AOTs of reference models (desert, continental clean, ...),
    in order to get a mix profile with AOTs and Angstrom coefficient in good agreement with observation.

    ===ARGS:
    AOT_OBS_wli    : Observaton AOT (scalar) at wavelenght wli
    AOT_modelX_wli : AOT (scalar) at wavelenght wli of the model X

    ===RETURN:
    A, B : factors of model A and B
    """

    ratioA = AOT_modelA_wl2/AOT_modelA_wl1
    B = (AOT_OBS_wl2 - AOT_OBS_wl1*ratioA) / (AOT_modelB_wl2 - AOT_modelB_wl1*ratioA)
    A = (AOT_OBS_wl1 - B*AOT_modelB_wl1) / AOT_modelA_wl1
    
    return A, B



def get_AB_coeff(modelA, modelB, wl1, wl2, AOT_OBS_wl1, AOT_OBS_wl2,
                 aot_refA=0.05, aot_refB=0.5, atm='afglt', P0=None,
                 O3=None, H2O=None, wl_ref=550., grid=None, O3_H2O_alt=None):

    """
    This function give the A, B factors to multiply with the AOTs of reference models (desert, continental clean, ...),
    in order to get a mix profile with AOTs and Angstrom coefficient in good agreement with observation.

    ===ARGS:
    modelA      : First OPAC model to use for the mix (i.g. 'desert', 'urban', ...)
    modelB      : Second OPAC model to use for the mix
    wli         : Observation wavelenghts i used for mixing
    AOT_OBS_wli : Obsevaton AOT (scalar) at wavelenght wli at z=0 or if O3_H2O_alt not none at z=O3_H2O_alt
    aot_refA/B  : reference AOT for model A and B at z=0
    atm         : The atmAFGL atmosphere used
    P0          : Surface pressure
    O3          : Scale ozone vertical column (Dobson units)
    wl_ref      : Wavelenght used as reference for some calculations
    grid        : Shape of vertical atm layers
    O3_H2O_alt  : altitude of H2O and O3 values, by default None and scale from z=0km

    ===RETURN:
    A, B : factors of model A and B
    """

    prof_MA = AtmAFGL(atm, comp=[AeroOPAC(modelA, aot_refA, wl_ref)],
                      O3=O3, P0=P0, H2O=H2O, grid=grid, O3_H2O_alt=O3_H2O_alt).calc([wl1, wl2], phase=False, phaseOpti=True)

    prof_MB = AtmAFGL(atm, comp=[AeroOPAC(modelB, aot_refB, wl_ref)],
                      O3=O3, P0=P0, H2O=H2O, grid=grid, O3_H2O_alt=O3_H2O_alt).calc([wl1, wl2], phase=False, phaseOpti=True)

    # Compute AOT of the 2 models at the 2 wavelenghts (wl1 and wl2)
    if (O3_H2O_alt is None): alt = -1
    else: alt = Idx(O3_H2O_alt)
    AOT_modelA_wl1 = prof_MA['OD_p'][Idx(wl1), alt]; AOT_modelA_wl2 = prof_MA['OD_p'][Idx(wl2), alt];
    AOT_modelB_wl1 = prof_MB['OD_p'][Idx(wl1), alt]; AOT_modelB_wl2 = prof_MB['OD_p'][Idx(wl2), alt];

    # print(alt)
    # print("MA (w1\w2)", prof_MA['OD_p'][Idx(wl1), alt], prof_MA['OD_p'][Idx(wl2), alt])
    # print("MB (w1\w2)", prof_MB['OD_p'][Idx(wl1), alt], prof_MB['OD_p'][Idx(wl2), alt])

    # Compute now the AOT of the mixted model at the 2 wl and the A and B parameters to know the proportion of each model
    A, B = compute_AB_coeff(AOT_OBS_wl1, AOT_OBS_wl2, AOT_modelA_wl1, AOT_modelA_wl2, AOT_modelB_wl1, AOT_modelB_wl2)

    return A, B


def check_date (dates, year):
    """
    Description: Make sure the parameter "dates" does not have dates of different years and
                 make sure the unique year of all the dates is equal to the "year" parameter

    ===Parameters:
    dates : Numpy array or list of dates at format "yyyy:mm:dd"
    year  : Interger at format yyyy
    """

    date_list = []
    dates_unique = np.unique(dates)
    for date in dates_unique: date_list.append(date.split(':')[-1])
    date_list = np.unique(date_list)

    if date_list.size != 1: raise NameError('Since the return result is with a "day of year" dimension, " \
                                        + " a file on several years of data is not authorised!')
    if int(date_list) != year: raise NameError('The chosen year and the data year are not the same!')

    return

def read_Aeronet_AOD(file, year):
    """
    Description: Extraction of AOD data from Aeronet file
                 and incoporate them in a LUT object

    === Parameters:
    file : Extinction AOD aeronet file (finishing by .aod)
    year : The year for 'Day_of_Year(Fraction)' dimension creation

    === Return
    LUT object with extinction AOD in function of Day_of_Year(Fraction) and wavelength
    """

    AOD = pd.read_csv(file, sep=',', skiprows=6)
    NTIME_AOD = AOD.index.size

    check_date (dates=AOD["Date(dd:mm:yyyy)"].values, year=year)

    wav_ext = []
    for key in AOD.keys():
        if 'AOD_Extinction-Total' in key:
            str_bis = key.split('[')
            wav_ext.append(float(str_bis[1][:-3]))
    wav_ext = np.unique(wav_ext)        
    NWAV_EXT = len(wav_ext)

    mat_ext = np.zeros((NTIME_AOD, NWAV_EXT), dtype=np.float64)
    for itime in range (0, NTIME_AOD):
        ind = (AOD.index == itime)
        for iwav, wav in enumerate(wav_ext):
            key = 'AOD_Extinction-Total[' + str(int(wav)) + 'nm]'
            mat_ext[itime, iwav] = AOD[ind][key]

    AOD_ext_lut = LUT(mat_ext, axes=[AOD["Day_of_Year(Fraction)"].values, wav_ext],
                      names=['Day_of_Year(Fraction)', 'wavelength'])

    return AOD_ext_lut
    
def read_Aeronet_SSA(file, year):
    """
    Description: Extraction of SSA data from Aeronet file
                 and incoporate them in a LUT object

    === Parameters:
    file : Single scattering albedo aeronet file (finishing by .ssa)
    year : The year for 'Day_of_Year(Fraction)' dimension creation

    === Return
    LUT object with SSA in function of Day_of_Year(Fraction) and wavelength
    """
    SSA = pd.read_csv(file, sep=',', skiprows=6)
    NTIME_SSA = SSA.index.size

    check_date (dates=SSA["Date(dd:mm:yyyy)"].values, year=year)

    wav_ssa = []
    for key in SSA.keys():
        if 'Single_Scattering_Albedo' in key:
            str_bis = key.split('[')
            wav_ssa.append(float(str_bis[1][:-3]))
    wav_ssa = np.unique(wav_ssa)        
    NWAV_SSA = len(wav_ssa)

    mat_ssa = np.zeros((NTIME_SSA, NWAV_SSA), dtype=np.float64)
    for itime in range (0, NTIME_SSA):
        ind = (SSA.index == itime)
        for iwav, wav in enumerate(wav_ssa):
            key = 'Single_Scattering_Albedo[' + str(int(wav)) + 'nm]'
            mat_ssa[itime, iwav] = SSA[ind][key]

    SSA_lut = LUT(mat_ssa, axes=[SSA["Day_of_Year(Fraction)"].values, wav_ssa],
                  names=['Day_of_Year(Fraction)', 'wavelength'])

    return SSA_lut

def read_Aeronet_PFN(file, year):
    """
    Description: Extraction of PFN data from Aeronet file
                 and incoporate them in a LUT object

    === Parameters:
    file : Phase matrix aeronet file (finishing by .pfn)
    year : The year for 'Day_of_Year(Fraction)' dimension creation

    === Return
    LUT object with PFN in function of Day_of_Year(Fraction), wavelength and thetha_atm
    """
    PFN = pd.read_csv(file, sep=',', skiprows=6)
    PFN = PFN[PFN['Phase_Function_Mode']=='Total'] # take only total of fine + coarse
    NTIME_PFN = PFN.index.size

    check_date (dates=PFN["Date(dd:mm:yyyy)"].values, year=year)

    ang = []; wav_pfn = []
    for key in PFN.keys():
        if '0000' in key:
            str_bis = key.split('[')
            ang.append(float(str_bis[0]))
            wav_pfn.append(float(str_bis[1][:-3]))
    ang = np.unique(ang)[::-1]
    wav_pfn = np.unique(wav_pfn)
    NANG = len(ang)
    NWAV_PFN = len(wav_pfn)

    mat_pfn = np.zeros((NTIME_PFN, NWAV_PFN, NANG), dtype=np.float64)
    for itime in range (0, NTIME_PFN):
        ind = (PFN.index == itime)
        for iwav, wav in enumerate(wav_pfn):
            for iang, ag in enumerate(ang):
                ang_str = "%.6f" % float(ag)
                key = ang_str + "[" + str(int(wav)) + 'nm]'
                mat_pfn[itime, iwav, iang] = PFN[ind][key]
    
    phase_lut = LUT(mat_pfn, axes=[PFN["Day_of_Year(Fraction)"].values, wav_pfn, ang],
                    names=['Day_of_Year(Fraction)', 'wavelength', 'theta_atm'])

    return phase_lut


def atm_pro_from_aeronet(date, time, aod_file, ssa_file, pfn_file, b_wav, pfwav=None,
                         z_profil=np.linspace(100., 0., num=101), dens = None,
                         atm_name="afglt", P0=None, O3=None, H2O=None, O3_H2O_alt=None,
                         fill_value_time=None):
    """
    Description: Create an atmosphere MLUT object from aeronet files.

    === Parameters:
    date            : Date in the following format -> "yyyy-mm-dd"
    time            : Time in the following format -> "hh:mm:ss"
    AOD_file        : Extinction AOD aeronet file (finishing by .aod) or aod LUT
    ssa_file        : Single scattering albedo aeronet file (finishing by .ssa) or ssa LUT
    pfn_file        : Phase matrix aeronet file (finishing by .pfn) or pfn LUT
    bwav            : Kdis bands or list of wavelenghts
    pfwav           : List of wavelenghts where the phase functions are computed
    z_profil        : Altitude grid profil
    dens            : aerosol density in funtion of z_profil
    atm_name        : The atmAFGL atmosphere used
    P0              : Surface pressure
    O3              : Scale ozone vertical column (Dobson units)
    H2O             : Scale Water vertical column
    O3_H2O_alt      : Altitude of H2O and O3 values, by default None and scale from z=0km
    fill_value_time : Passed to interp1d for time interpolation e.g "fill_value='extrema'"

    === return
    SMART-G atmosphere profil MLUT
    """

    pd_date = pd.Timestamp(date + " " + time)
    nb_sec_day = 24*60*60 # number of seconds in one day
    day_frac = 1 - ( (nb_sec_day - (pd_date.hour*60*60 + pd_date.minute*60 + pd_date.second)) / nb_sec_day )
    day_year_frac = pd_date.day_of_year + day_frac; print('day_year_frac =', day_year_frac)
    year = pd_date.year

    if isinstance(aod_file, LUT): aod_lut = aod_file
    else: aod_lut = read_Aeronet_AOD(aod_file, year=year)
    if isinstance(ssa_file, LUT): ssa_lut = ssa_file
    else: ssa_lut = read_Aeronet_SSA(ssa_file, year=year)
    if isinstance(pfn_file, LUT): pfn_lut = pfn_file
    else: pfn_lut = read_Aeronet_PFN(pfn_file, year=year)
       
    aod_lut = aod_lut.sub({"Day_of_Year(Fraction)": Idx(day_year_frac, fill_value=fill_value_time)})
    ssa_lut = ssa_lut.sub({"Day_of_Year(Fraction)": Idx(day_year_frac, fill_value=fill_value_time)})
    pfn_lut = pfn_lut.sub({"Day_of_Year(Fraction)": Idx(day_year_frac, fill_value=fill_value_time)})

    if not isinstance(b_wav, BandSet): b_wav_BS = BandSet(b_wav)
    else : b_wav_BS = b_wav
    b_wav_unique = np.unique(b_wav_BS)
    if (pfwav is None): pf_wav = b_wav_unique
    else: pf_wav = pfwav

    f_ext      = interp1d(aod_lut.axes[0], aod_lut[:], fill_value='extrapolate')
    ext_interp = f_ext(b_wav_unique)
    if (np.any(ext_interp < 0)):
        print("Warning: AOD interpolation have values < 0, those values will be set to 0.")
        ext_interp[ext_interp < 0] = 0

    f_ssa      = interp1d(ssa_lut.axes[0], ssa_lut[:], fill_value='extrapolate')
    ssa_interp = f_ssa(b_wav_unique)
    if (np.any(ssa_interp < 0)):
        print("Warning: SSA interpolation have values < 0, those values will be set to 0.")
        ssa_interp[ssa_interp < 0] = 0
    if (np.any(ssa_interp > 1)):
        print("Warning: SSA interpolation have values > 1, those values will be set to 1.")
        ssa_interp[ssa_interp > 1] = 1
    
    wav_pfn = pfn_lut.axes[0]
    ang_pfn = pfn_lut.axes[1]
    pfn_interp = np.zeros((len(b_wav_unique), len(ang_pfn)), dtype=np.float64)  
    for iang in range (0, len(ang_pfn)):
        f_pfn = interp1d(wav_pfn, pfn_lut[:,iang], fill_value='extrapolate')
        pfn_interp[:,iang] = f_pfn(b_wav_unique)
    pfn_interp = np.stack([pfn_interp[:,:]]*4, axis=1)
    pfn_interp[:,2:3,:]=0.
    if(np.any(pfn_interp < 0)):
        print("Warning: PFN interpolation have values < 0, those values will be set to 0.")
        pfn_interp[pfn_interp < 0] = 0
    
    ext_interp_lut = LUT(ext_interp, axes=[b_wav_unique], names=['wavelength'])
    ssa_interp_lut = LUT(ssa_interp, axes=[b_wav_unique], names=['wavelength'])
    pfn_interp_lut = LUT(pfn_interp, axes=[b_wav_unique, None, ang_pfn], names=['wavelength', 'None', 'theta_atm'])
    aeronet_specie = SpeciesUser(name='aeronet', ext=ext_interp_lut, ssa=ssa_interp_lut, phase=pfn_interp_lut, fill_value='extrema')


    if dens is None: D=0.33; aero_dens = np.exp(-(z_profil-5)**2/D**2)
    else: aero_dens = dens

    comp  = CompUser(aeronet_specie, aero_dens, z_profil, aod_lut[0], aod_lut.axes[0][0])
    atm_pro = AtmAFGL(atm_name, grid=z_profil, P0=P0, O3=O3, H2O=H2O, comp=[comp], pfwav=pf_wav, O3_H2O_alt=O3_H2O_alt).calc(b_wav_BS, phaseOpti=True)

    return atm_pro

def atm_pro_from_aeronet_opti(date, time, aod_file, ssa_file, pfn_file, b_wav, pfwav=None,
                              z_profil=np.linspace(100., 0., num=101), dens = None,
                              atm_name="afglt", P0=None, O3=None, H2O=None, O3_H2O_alt=None,
                              fill_value_time=None):
    """
    Description: Optimized version of atm_pro_from_aeronet(), but still in development.

    === Parameters:
    date            : Date in the following format -> "yyyy-mm-dd"
    time            : Time in the following format -> "hh:mm:ss"
    AOD_file        : Extinction AOD aeronet file (finishing by .aod) or aod LUT
    ssa_file        : Single scattering albedo aeronet file (finishing by .ssa) or ssa LUT
    pfn_file        : Phase matrix aeronet file (finishing by .pfn) or pfn LUT
    bwav            : Kdis bands or list of wavelenghts
    pfwav           : List of wavelenghts where the phase functions are computed
    z_profil        : Altitude grid profil
    dens            : aerosol density in funtion of z_profil
    atm_name        : The atmAFGL atmosphere used
    P0              : Surface pressure
    O3              : Scale ozone vertical column (Dobson units)
    H2O             : Scale Water vertical column
    O3_H2O_alt      : Altitude of H2O and O3 values, by default None and scale from z=0km
    fill_value_time : Passed to interp1d for time interpolation e.g "fill_value='extrema'"

    === return
    SMART-G atmosphere profil MLUT
    """

    pd_date = pd.Timestamp(date + " " + time)
    nb_sec_day = 24*60*60 # number of seconds in one day
    day_frac = 1 - ( (nb_sec_day - (pd_date.hour*60*60 + pd_date.minute*60 + pd_date.second)) / nb_sec_day )
    day_year_frac = pd_date.day_of_year + day_frac; print('day_year_frac =', day_year_frac)
    year = pd_date.year

    if isinstance(aod_file, LUT): aod_lut = aod_file
    else: aod_lut = read_Aeronet_AOD(aod_file, year=year)
    if isinstance(ssa_file, LUT): ssa_lut = ssa_file
    else: ssa_lut = read_Aeronet_SSA(ssa_file, year=year)
    if isinstance(pfn_file, LUT): pfn_lut = pfn_file
    else: pfn_lut = read_Aeronet_PFN(pfn_file, year=year)
       
    aod_lut = aod_lut.sub({"Day_of_Year(Fraction)": Idx(day_year_frac, fill_value=fill_value_time)})
    ssa_lut = ssa_lut.sub({"Day_of_Year(Fraction)": Idx(day_year_frac, fill_value=fill_value_time)})
    pfn_lut = pfn_lut.sub({"Day_of_Year(Fraction)": Idx(day_year_frac, fill_value=fill_value_time)})

    if not isinstance(b_wav, BandSet): b_wav_BS = BandSet(b_wav)
    else : b_wav_BS = b_wav
    b_wav_unique = np.unique(b_wav_BS)
    if (pfwav is None): pf_wav = b_wav_unique
    else: pf_wav = pfwav

    wav_interp_ext = aod_lut.axes[0]
    if (b_wav_unique[0] < wav_interp_ext[0]): wav_interp_ext = np.concatenate([[b_wav_unique[0]], wav_interp_ext])
    if (b_wav_unique[-1] > wav_interp_ext[-1]): wav_interp_ext = np.concatenate([wav_interp_ext, [b_wav_unique[-1]]])
    f_ext      = interp1d(aod_lut.axes[0], aod_lut[:], fill_value='extrapolate')
    ext_interp = f_ext(wav_interp_ext)
    if (np.any(ext_interp < 0)):
        print("Warning: AOD interpolation have values < 0, those values will be set to 0.")
        ext_interp[ext_interp < 0] = 0

    wav_interp_ssa = ssa_lut.axes[0]
    if (b_wav_unique[0] < wav_interp_ssa[0]): wav_interp_ssa = np.concatenate([[b_wav_unique[0]], wav_interp_ssa])
    if (b_wav_unique[-1] > wav_interp_ssa[-1]): wav_interp_ssa = np.concatenate([wav_interp_ssa, [b_wav_unique[-1]]])
    f_ssa      = interp1d(ssa_lut.axes[0], ssa_lut[:], fill_value='extrapolate')
    ssa_interp = f_ssa(wav_interp_ssa)
    if (np.any(ssa_interp < 0)):
        print("Warning: SSA interpolation have values < 0, those values will be set to 0.")
        ssa_interp[ssa_interp < 0] = 0
    if (np.any(ssa_interp > 1)):
        print("Warning: SSA interpolation have values > 1, those values will be set to 1.")
        ssa_interp[ssa_interp > 1] = 1
    
    wav_interp_pfn = pfn_lut.axes[0]
    if (b_wav_unique[0] < wav_interp_pfn[0]): wav_interp_pfn = np.concatenate([[b_wav_unique[0]], wav_interp_pfn])
    if (b_wav_unique[-1] > wav_interp_pfn[-1]): wav_interp_pfn = np.concatenate([wav_interp_pfn, [b_wav_unique[-1]]])
    wav_pfn = pfn_lut.axes[0]
    ang_pfn = pfn_lut.axes[1]
    pfn_interp = np.zeros((len(wav_interp_pfn), len(ang_pfn)), dtype=np.float64)  
    for iang in range (0, len(ang_pfn)):
        f_pfn = interp1d(wav_pfn, pfn_lut[:,iang], fill_value='extrapolate')
        pfn_interp[:,iang] = f_pfn(wav_interp_pfn)
    pfn_interp = np.stack([pfn_interp[:,:]]*4, axis=1)
    pfn_interp[:,2:3,:]=0.
    if(np.any(pfn_interp < 0)):
        print("Warning: PFN interpolation have values < 0, those values will be set to 0.")
        pfn_interp[pfn_interp < 0] = 0
    
    ext_interp_lut = LUT(ext_interp, axes=[wav_interp_ext], names=['wavelength'])
    ssa_interp_lut = LUT(ssa_interp, axes=[wav_interp_ext], names=['wavelength'])
    pfn_interp_lut = LUT(pfn_interp, axes=[wav_interp_ext, None, ang_pfn], names=['wavelength', 'None', 'theta_atm'])
    aeronet_specie = SpeciesUser(name='aeronet', ext=ext_interp_lut, ssa=ssa_interp_lut, phase=pfn_interp_lut, fill_value='extrema')


    if dens is None: D=0.33; aero_dens = np.exp(-(z_profil-5)**2/D**2)
    else: aero_dens = dens

    comp  = CompUser(aeronet_specie, aero_dens, z_profil, aod_lut[0], aod_lut.axes[0][0])
    atm_pro = AtmAFGL(atm_name, grid=z_profil, P0=P0, O3=O3, H2O=H2O, comp=[comp], pfwav=pf_wav, O3_H2O_alt=O3_H2O_alt).calc(b_wav_BS, phaseOpti=True)

    # TODO finish the development below (for computational time optimisation).
    # ext_tot = np.zeros((len(ext_interp_lut.axes[0]), len(z_profil)), dtype=np.float64)
    # ssa_tot = np.zeros((len(ssa_interp_lut.axes[0]), len(z_profil)), dtype=np.float64)
    # pfn_tot = np.zeros((len(pfn_interp_lut.axes[0]), len(z_profil), len(pfn_interp_lut.axes[1]), len(pfn_interp_lut.axes[2])), dtype=np.float64)    

    # atm_pro2 = AtmAFGL(atm_name, comp=[AeroOPAC('desert', aod_lut[0], aod_lut.axes[0][0], phase=phase_tot_lut) ], grid=z_profil, pfwav=pf_wav,
    #                                  prof_aer=(prof_aer_ext_new, prof_aer_ssa_new)).calc(b_wav_BS, phaseOpti=True)

    return atm_pro

def atm_pro_from_aeronet_opti2(date, time, aod_file, ssa_file, pfn_file, b_wav, pfwav=None,
                               z_profil=np.linspace(100., 0., num=101), dens = None,
                               atm_name="afglt", P0=None, O3=None, H2O=None, O3_H2O_alt=None,
                               fill_value_time=None, rayleigh_atm=None, NBTHETA=721):
    """
    Description:  Optimized version (time computation) of atm_pro_from_aeronet()

    === Parameters:
    date            : Date in the following format -> "yyyy-mm-dd"
    time            : Time in the following format -> "hh:mm:ss"
    AOD_file        : Extinction AOD aeronet file (finishing by .aod) or aod LUT
    ssa_file        : Single scattering albedo aeronet file (finishing by .ssa) or ssa LUT
    pfn_file        : Phase matrix aeronet file (finishing by .pfn) or pfn LUT
    bwav            : Kdis bands or list of wavelenghts
    pfwav           : List of wavelenghts where the phase functions are computed
    z_profil        : Altitude grid profil
    dens            : aerosol density in funtion of z_profil
    atm_name        : The atmAFGL atmosphere used
    P0              : Surface pressure
    O3              : Scale ozone vertical column (Dobson units)
    H2O             : Scale Water vertical column
    O3_H2O_alt      : Altitude of H2O and O3 values, by default None and scale from z=0km
    fill_value_time : Passed to interp1d for time interpolation e.g "fill_value='extrema'"
    rayleigh_atm    : MLUT objet with the rayleigh atm profil, if None create it
    NBTHETA         : Phase matrix angle resampling over NBTHETA angles

    === return
    SMART-G atmosphere profil MLUT
    """

    if rayleigh_atm is None :
        rayleigh_atm = AtmAFGL(atm_name, grid=z_profil, P0=P0, O3=O3, H2O=H2O, O3_H2O_alt=O3_H2O_alt).calc(b_wav, phase=False)

    pd_date = pd.Timestamp(date + " " + time)
    nb_sec_day = 24*60*60 # number of seconds in one day
    day_frac = 1 - ( (nb_sec_day - (pd_date.hour*60*60 + pd_date.minute*60 + pd_date.second)) / nb_sec_day )
    day_year_frac = pd_date.day_of_year + day_frac; print('day_year_frac =', day_year_frac)
    year = pd_date.year

    if isinstance(aod_file, LUT): aod_lut = aod_file
    else: aod_lut = read_Aeronet_AOD(aod_file, year=year)
    if isinstance(ssa_file, LUT): ssa_lut = ssa_file
    else: ssa_lut = read_Aeronet_SSA(ssa_file, year=year)
    if isinstance(pfn_file, LUT): pfn_lut = pfn_file
    else: pfn_lut = read_Aeronet_PFN(pfn_file, year=year)
       
    aod_lut = aod_lut.sub({"Day_of_Year(Fraction)": Idx(day_year_frac, fill_value=fill_value_time)})
    ssa_lut = ssa_lut.sub({"Day_of_Year(Fraction)": Idx(day_year_frac, fill_value=fill_value_time)})
    pfn_lut = pfn_lut.sub({"Day_of_Year(Fraction)": Idx(day_year_frac, fill_value=fill_value_time)})

    if not isinstance(b_wav, BandSet): b_wav_BS = BandSet(b_wav)
    else : b_wav_BS = b_wav
    b_wav_unique = np.unique(b_wav_BS)
    if (pfwav is None): pf_wav = b_wav_unique
    else: pf_wav = pfwav

    f_ext      = interp1d(aod_lut.axes[0], aod_lut[:], fill_value='extrapolate')
    ext_interp = f_ext(b_wav_unique)
    if (np.any(ext_interp < 0)):
        print("Warning: AOD interpolation have values < 0, those values will be set to 0.")
        ext_interp[ext_interp < 0] = 0

    f_ssa      = interp1d(ssa_lut.axes[0], ssa_lut[:], fill_value='extrapolate')
    ssa_interp = f_ssa(b_wav_unique)
    if (np.any(ssa_interp < 0)):
        print("Warning: SSA interpolation have values < 0, those values will be set to 0.")
        ssa_interp[ssa_interp < 0] = 0
    if (np.any(ssa_interp > 1)):
        print("Warning: SSA interpolation have values > 1, those values will be set to 1.")
        ssa_interp[ssa_interp > 1] = 1
    
    wav_pfn = pfn_lut.axes[0]
    ang_pfn = pfn_lut.axes[1]
    pfn_interp = np.zeros((len(b_wav_unique), len(ang_pfn)), dtype=np.float64)  
    for iang in range (0, len(ang_pfn)):
        f_pfn = interp1d(wav_pfn, pfn_lut[:,iang], fill_value='extrapolate')
        pfn_interp[:,iang] = f_pfn(b_wav_unique)
    pfn_interp = np.stack([pfn_interp[:,:]]*4, axis=1)
    pfn_interp[:,2:3,:]=0.
    if(np.any(pfn_interp < 0)):
        print("Warning: PFN interpolation have values < 0, those values will be set to 0.")
        pfn_interp[pfn_interp < 0] = 0
    
    ext_interp_lut = LUT(ext_interp, axes=[b_wav_unique], names=['wavelength'])
    ssa_interp_lut = LUT(ssa_interp, axes=[b_wav_unique], names=['wavelength'])
    pfn_interp_lut = LUT(pfn_interp, axes=[b_wav_unique, None, ang_pfn], names=['wavelength', 'None', 'theta_atm'])
    aeronet_specie = SpeciesUser(name='aeronet', ext=ext_interp_lut, ssa=ssa_interp_lut, phase=pfn_interp_lut, fill_value='extrema')


    if dens is None: D=0.33; aero_dens = np.exp(-(z_profil-5)**2/D**2)
    else: aero_dens = dens

    comp  = CompUser(aeronet_specie, aero_dens, z_profil, aod_lut[0], aod_lut.axes[0][0])

    atm_pro = MLUT()
    atm_pro.add_axis('z_atm', rayleigh_atm.axes["z_atm"][:])
    atm_pro.add_axis('wavelength', rayleigh_atm.axes["wavelength"][:])
    atm_pro.add_lut(rayleigh_atm["n_atm"])
    atm_pro.add_lut(rayleigh_atm["T_atm"])
    atm_pro.add_lut(rayleigh_atm["OD_r"])
    atm_pro.add_lut(rayleigh_atm["OD_g"])
    atm_pro.add_lut(rayleigh_atm["pine_atm"])
    atm_pro.add_lut(rayleigh_atm["FQY1_atm"])

    tau_r = rayleigh_atm["OD_r"][:,:]
    tau_g = rayleigh_atm["OD_g"][:,:]
    dtau_r = diff1(tau_r, axis=1)
    dtau_g = diff1(tau_g, axis=1)

    # Praticle/aerosol optical depth and single scattering albedo
    dtau_p, ssa_tmp = comp.dtau_ssa(np.array(b_wav_BS), z_profil, None)
    dtau_p = np.float32(dtau_p); ssa_tmp = np.float32(ssa_tmp)
    tau_p = np.cumsum(dtau_p, axis=1)
    ssa_p = dtau_p * ssa_tmp; ssa_p[dtau_p!=0] /= dtau_p[dtau_p!=0]; ssa_p[dtau_p==0] = 1.
    atm_pro.add_dataset('OD_p', tau_p, axnames=['wavelength', 'z_atm'],
                        attrs={'description': 'Cumulated particles optical thickness at each wavelength'})
    atm_pro.add_dataset('ssa_p_atm', ssa_p, axnames=['wavelength', 'z_atm'],
                        attrs={'description': 'Particles single scattering albedo of the layer'})

    # Atmosphere (total) optical depth
    tau_atm = tau_r + tau_g + tau_p
    atm_pro.add_dataset('OD_atm', tau_atm, axnames=['wavelength', 'z_atm'],
                        attrs={'description': 'Cumulated extinction optical thickness'})
    
    # Scattering optical depth
    tau_sca = np.cumsum(dtau_r + dtau_p*ssa_p, axis=1)
    atm_pro.add_dataset('OD_sca_atm', tau_sca, axnames=['wavelength', 'z_atm'],
                        attrs={'description': 'Cumulated scattering optical thickness'})

    # Absorption optical depth
    tau_abs = np.cumsum(dtau_g + dtau_p*(1-ssa_p), axis=1)
    atm_pro.add_dataset('OD_abs_atm', tau_abs, axnames=['wavelength', 'z_atm'],
                        attrs={'description': 'Cumulated absorption optical thickness'})

    # Atmosphere (total) single scattering albedo
    dtau_atm = diff1(tau_atm, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ssa_atm = (dtau_r + dtau_p*ssa_p)/dtau_atm
    ssa_atm[np.isnan(ssa_atm)] = 1.
    atm_pro.add_dataset('ssa_atm', ssa_atm, axnames=['wavelength', 'z_atm'],
                        attrs={'description': 'Single scattering albedo of the layer'})

    # Ratio of molecular scattering by total sattering
    with np.errstate(invalid='ignore', divide='ignore'):
        pmol_atm = dtau_r/(dtau_r + dtau_p*ssa_p)
    pmol_atm[np.isnan(pmol_atm)] = 1.
    atm_pro.add_dataset('pmol_atm', pmol_atm, axnames=['wavelength', 'z_atm'],
                        attrs={'description': 'Ratio of molecular scattering to total scattering of the layer'})


    # Phase matrix and phase indices
    pha_ = comp.phase(np.array(pf_wav), np.array([100., 0.]), None, NBTHETA=NBTHETA)
    pha_atm, ipha_atm = calc_iphase(pha_, np.array(b_wav_BS), z_profil)
    print("iphase shape=", ipha_atm.shape)
    atm_pro.add_axis('theta_atm', pha_.axes[-1])
    atm_pro.add_dataset('phase_atm', pha_atm, ['iphase', 'stk', 'theta_atm'])
    atm_pro.add_dataset('iphase_atm', ipha_atm, ['wavelength', 'z_atm'])

    return atm_pro
