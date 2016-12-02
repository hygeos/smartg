#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, division, absolute_import
import numpy as np
from os.path import join, dirname, exists, basename
from glob import glob
from tools.luts import MLUT, LUT, read_mlut_netcdf4, Idx
from tools.phase import calc_iphase
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.constants import codata
import netCDF4
from warnings import warn, catch_warnings
import sys
if sys.version_info[:2] >= (3, 0):
    xrange = range


dir_libradtran = '/home/applis/libRadtran/libRadtran-2.0/'
dir_libradtran_reptran =  join(dir_libradtran, 'data/correlated_k/reptran/')
dir_libradtran_opac =  join(dir_libradtran, 'data/aerosol/OPAC/')
dir_libradtran_atmmod = join(dir_libradtran, 'data/atmmod/')
dir_libradtran_crs = join(dir_libradtran, 'data/crs/')

NPSTK = 4 # number of Stokes parameters of the radiation field



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
    def __init__(self, species):

        self.name = species
        fname = join(dir_libradtran_opac, 'optprop', species+'.cdf')
        if not exists(fname):
            raise Exception('file {} does not exist'.format(fname))

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
            ext = self._ext[Idx(wav), Idx(reff2)]
            ssa = self._ssa[Idx(wav), Idx(reff2)]

            # create empty dimension for rh
            ext = ext[:,None]
            ssa = ssa[:,None]

        elif rh is not None: # the component properties depend on RH (thus Z)
            assert self._rh_or_reff == 'rh'

            [wav2, rh2] = np.broadcast_arrays(wav[:,None], rh[None,:])
            ext = self._ext[Idx(wav2), Idx(rh2, fill_value='extrema,warn')]
            ssa = self._ssa[Idx(wav2), Idx(rh2, fill_value='extrema,warn')]

            ext *= self._rho[Idx(rh)]/self._rho[Idx(50.)]

        else:
            raise Exception('Error')

        return ext, ssa

    def phase(self, wav, rh, NBTHETA, reff=None):
        '''
        phase function of species at wavelengths wav
        resampled over NBTHETA angles
        '''

        theta = np.linspace(0., 180., num=NBTHETA)
        NLAM = len(wav)

        if (self._nrh_reff > 1) and (self._rh_or_reff == 'rh'):
            # drop first altitude element

            P = np.zeros((NLAM, len(rh)-1, NPSTK, NBTHETA), dtype='float32')

            for irh_, rh_ in enumerate(rh[1:]):

                irh = Idx(rh_, round=True)

                # interpolate on wav and rh
                pha = self._phase.sub()[Idx(wav,round=True),irh,:,:]  # (lam, stk, nthetamax)
                th = self._theta[Idx(wav,round=True),irh,:,:]   # (lam, stk, nthetamax)
                nth = self._ntheta[Idx(wav,round=True),irh,:]   # (lam, stk)

                if (NBTHETA < nth).any():
                    warn('Insufficient number of sampling angles for phase function')

                for ilam in xrange(NLAM):
                    nth_ = nth[ilam, 0]
                    assert (nth_ == nth[ilam, :]).all()

                    # convert I, Q => Ipar and Iper
                    P0 = interp1d(th[ilam,0,:nth_], pha[ilam,0,:nth_])(theta)
                    P1 = interp1d(th[ilam,1,:nth_], pha[ilam,1,:nth_])(theta)
                    P[ilam, irh_, 0, :] = P0 + P1
                    P[ilam, irh_, 1, :] = P0 - P1
                    P[ilam, irh_, 2, :] = interp1d(th[ilam,2,:nth_], pha[ilam,2,:nth_])(theta)
                    P[ilam, irh_, 3, :] = interp1d(th[ilam,3,:nth_], pha[ilam,3,:nth_])(theta)

        else: # phase function does not depend on rh
            P = np.zeros((NLAM, 1, NPSTK, NBTHETA), dtype='float32')
            if (self._rh_or_reff == 'reff') and (reff is not None):
                irh = Idx(reff).index(self._phase.axes[1])
            else:
                irh = 0

            # interpolate on wav and rh
            pha = self._phase.sub()[Idx(wav,round=True),irh,:,:]  # (lam, stk, nthetamax)
            th = self._theta[Idx(wav,round=True),irh,:,:]   # (lam, stk, nthetamax)
            nth = self._ntheta[Idx(wav,round=True),irh,:]   # (lam, stk)

            if (NBTHETA < nth).any():
                warn('Insufficient number of sampling angles for phase function')

            for ilam in xrange(NLAM):
                nth_ = nth[ilam, 0]
                assert (nth[ilam, :] == nth_).all()
                P0 = interp1d(th[ilam,0,:nth_], pha[ilam,0,:nth_])(theta)
                P1 = interp1d(th[ilam,1,:nth_], pha[ilam,1,:nth_])(theta)
                P[ilam, 0, 0, :] = P0 + P1
                P[ilam, 0, 1, :] = P0 - P1
                P[ilam, 0, 2, :] = interp1d(th[ilam,2,:nth_], pha[ilam,2,:nth_])(theta)
                P[ilam, 0, 3, :] = interp1d(th[ilam,3,:nth_], pha[ilam,3,:nth_])(theta)

        return LUT(P,
                   axes=[wav, None, None, theta],
                   names=['wav_phase', 'z_phase', 'stk', 'theta'],
                   )


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
    def __init__(self, filename, tau_ref, w_ref, ssa=None, phase=None):
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
        self.densities = data[::-1,1:]  # vertical profile of mass concentration (g/m3)
                                        # (zopac, species)

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

    def phase(self, wav, Z, rh=None, NBTHETA=7201):
        '''
        Phase function calculation at wavelength wav and altitudes Z
        relative humidity is rh
        angle resampling over NBTHETA angles
        '''
        if self._phase is not None:
            if self._phase.ndim == 2:
                # convert to 4-dim by inserting empty dimensions wav_phase
                # and z_phase
                assert self._phase.names == ['stk', 'theta']
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
            P += s.phase(wav, rh, NBTHETA, reff=self.reff)*dssa_  # (NLAM, ALTITUDE-1, NPSTK, NBTHETA)

        with np.errstate(divide='ignore'):
            P.data /= dssa
        P.data[np.isnan(P.data)] = 0.

        P.axes[1] = average(Z)

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

    TODO: phase

    Example: CloudOPAC('wc.sol', 12.68, 2, 3, 10., 550.)
             # water cloud mie, reff=12.68 between 2 and 3 km
             # total optical thickness of 10 at 550 nm
    '''
    def __init__(self, species, reff, zmin, zmax, tau_ref, w_ref, phase=None):
        self.reff = reff
        self.tau_ref = tau_ref
        self.w_ref = w_ref
        self.species = [Species(species+'.mie')]
        self.zopac     = np.array([zmax, zmax, zmin, zmin, 0.], dtype='f')
        self.densities = np.array([  0.,   1.,   1.,   0., 0.], dtype='f')[:,None]
        self.ssa = None
        self._phase = phase


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
        - comp: list of components objects (aerosol, clouds)
        - grid: new grid altitudes (list of decreasing altitudes in km)
        - lat: latitude (for Rayleigh optical depth calculation, default=45.)
        - P0: Sea surface pressure
              (default: SSP from AFGL)

        Gaseous absorption:
        - O3: total ozone column (Dobson units),
          or None to use atmospheric profile value (default)
        - H2O: total water vapour column (g.cm-2), or None to use atmospheric
          profile value (default)
        - NO2: activate NO2 absorption (default True)
        - tauR: Rayleigh optical thickness, default None computed
          from atmospheric profile and wavelength
        - prof_abs: the gaseous absorption optical thickness profile  provided by user
                    if directly used, it shortcuts any further gaseous absorption computation, array of dimension (NWavelength,NZ)
        - prof_ray: the rayleigh scattering optical thickness profile  provided by user
                    if directly used, it shortcuts any further rayleigh scattering computation , array of dimension (NWavelength,NZ)
        - prof_aer: a tuple (ext,ssa) the aerosol extinction optical thickness profile and single scattering albedo arrays  
                    provided by user, each array has dimensions (NWavelength,NZ)
                    if directly used, it shortcuts any further rayleigh scattering computation

        Phase functions definition:
        - pfwav: a list of wavelengths over which the phase functions are calculated
          default: None (all wavelengths)
        - pfgrid: altitude grid over which the phase function is calculated
          can be provided as an array of decreasing altitudes or a gridspec
          default value: [100, 0]
    '''
    def __init__(self, atm_filename, comp=[],
                 grid=None, lat=45.,
                 P0=None, O3=None, H2O=None, NO2=True,
                 tauR=None,
                 pfwav=None, pfgrid=[100., 0.], prof_abs=None,
                 prof_ray=None, prof_aer=None):

        self.lat = lat
        self.comp = comp
        self.pfwav = pfwav
        self.pfgrid = np.array(pfgrid)
        self.prof_abs = prof_abs
        self.prof_ray = prof_ray
        self.prof_aer = prof_aer

        self.tauR = tauR
        if tauR is not None:
            self.tauR = np.array(tauR)

        assert (np.diff(pfgrid) < 0.).all()

        #
        # init directories
        #
        if dirname(atm_filename) == '':
            atm_filename = join(dir_libradtran_atmmod, atm_filename)
        if (not exists(atm_filename)) and (not atm_filename.endswith('.dat')):
            atm_filename += '.dat'

        crs_O3_filename = join(dir_libradtran_crs, 'crs_O3_UBremen_cf.dat')
        crs_NO2_filename = join(dir_libradtran_crs, 'crs_NO2_UBremen_cf.dat')

        # read crs ozone file
        _crs_chappuis = np.loadtxt(crs_O3_filename, comments="#")
        self.crs_chappuis = LUT(
                _crs_chappuis[:,1:],
                axes=[_crs_chappuis[:,0], None],
                names=['wavelength', None],
                )

        # read crs no2 file
        _crs_no2 = np.loadtxt(crs_NO2_filename, comments="#")
        self.crs_no2 = LUT(
                _crs_no2[:,1:],
                axes=[_crs_no2[:,0], None],
                names=['wavelength', None],
                )
        self.crs_no2

        # read afgl file
        prof = Profile_base(atm_filename, O3=O3,
                            H2O=H2O, NO2=NO2, P0=P0)

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
        self.prof_red = prof.regrid(pfgrid)    # TODO: only if necessary


    def calc(self, wav, phase=True):
        '''
        Profile and phase function calculation at bands wav

        phase: boolean (activate phase function calculation)

        Returns: profile + phase function MLUT
        '''

        profile = self.profile(wav)

        if phase:
            if self.pfwav is None:
                wav_pha = wav
            else:
                wav_pha = self.pfwav
            pha = self.phase(wav_pha)

            if pha is not None:
                pha_, ipha = calc_iphase(pha, profile.axis('wavelength'), profile.axis('z_atm'))
                profile.add_axis('theta', pha.axes[-1])
                profile.add_dataset('phase_atm', pha_, ['iphase_atm', 'stk', 'theta'])
                profile.add_dataset('iphase_atm', ipha, ['wavelength', 'z_atm'])

        return profile


    def profile(self, wav, prof=None):
        '''
        Calculate the profile of optical properties at given wavelengths
        wav: array of wavelength in nm
        prof: profile of densities (default: self.prof)

        returns: the profile of optical properties
        '''
        wav = np.array(wav)
        if wav.ndim == 0:
            wav = wav.reshape(1)

        if prof is None:
            prof = self.prof

        dz = -diff1(prof.z)

        pro = MLUT()
        pro.add_axis('z_atm', prof.z)
        pro.add_axis('wavelength', wav)

        #
        # Rayleigh optical thickness
        #
        # cumulated Rayleigh optical thickness (wav, z)
        if self.prof_ray is None :
            tauray = rod(wav*1e-3, prof.dens_co2/prof.dens_air*1e6, self.lat,
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
        pro.add_dataset('OD_r', tauray, axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Cumulated rayleigh optical thickness'})

        #
        # Aerosol optical thickness and single scattering albedo
        #
        if self.prof_aer is None :
            dtaua = np.zeros((len(wav), len(prof.z)), dtype='float32')
            ssa_p = np.zeros((len(wav), len(prof.z)), dtype='float32')
            for comp in self.comp:
                dtau_, ssa_ = comp.dtau_ssa(wav, prof.z, prof.RH())
                dtaua += dtau_
                ssa_p+= dtau_ * ssa_
            ssa_p[dtaua!=0] /= dtaua[dtaua!=0]
            ssa_p[dtaua==0] = 1.
            taua = np.cumsum(dtaua, axis=1)

        else:
            (dtaua, ssa_p) = self.prof_aer
            taua= np.cumsum(dtaua,axis=1)

        pro.add_dataset('OD_p', taua,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Cumulated particles optical thickness at each wavelength'})
        pro.add_dataset('ssa_p_atm', ssa_p, axnames=['wavelength', 'z_atm'],
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
                    tau_o3.data[tau_o3.data == np.amin(tau_o3.data)][0]
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
            if not (tau_no2.data >= 0).all():
                warn('Negative values in tau_no2 ({}%, min value is {}, set to 0)'.format(
                    100.*np.sum(tau_no2.data<0)/float(tau_no2.data.size),
                    tau_no2.data[tau_no2.data == np.amin(tau_no2.data)][0]
                    ))
            tau_no2.data[tau_no2.data < 0] = 0

            #
            # Total gaseous optical thickness
            #
            dtaug = tau_o3 + tau_no2   # TODO: other gases
            taug = dtaug.apply(lambda x: np.cumsum(x, axis=1))
            taug.attrs['description'] = 'Cumulated gaseous absorption optical thickness'
            pro.add_lut(taug, desc='OD_g')
            
        else : 
            dtaug = self.prof_abs
            taug  = np.cumsum(dtaug,axis=1)
            pro.add_dataset('OD_g', taug, axnames=['wavelength', 'z_atm'],
                  attrs={'description':
                         'Cumulated gaseous absorption optical thickness'})
            

        #
        # Total optical thickness and other parameters
        #
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

        with np.errstate(invalid='ignore'):
            pmol = dtaur/(dtaur + dtaua)
        pmol[np.isnan(pmol)] = 1.
        pro.add_dataset('pmol_atm', pmol,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Ratio of molecular scattering to total scattering of the layer'})

        with np.errstate(invalid='ignore'):
            ssa = (dtaur+ dtaua*ssa_p)/diff1(tau_tot, axis=1)
        ssa[np.isnan(ssa)] = 1.
        pro.add_dataset('ssa_atm', ssa,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Single scattering albedo of the layer'})

        return pro


    def phase(self, wav):
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
            pha += comp.phase(wav, self.pfgrid, rh)*dtau*ssa_p
            norm += dtau*ssa_p

        if len(self.comp) > 0:
            pha /= norm
            pha.data[np.isnan(pha.data)] = 0.

            return pha
        else:
            return None


def read_phase(filename, standard=False):
    '''
    Read phase function from filename as a LUT

    standard: standard phase function definition, otherwise Smart-g definition
    '''
    data = np.loadtxt(filename)

    theta = data[:,0]
    pha=data[:,1:]
    if standard:
        pha[:,0] = data[:,1] + data[:,2]
        pha[:,1] = data[:,1] - data[:,2]
        pha[:,2] = data[:,3]
        pha[:,3] = data[:,4]

    P = LUT(pha.swapaxes(0, 1),  # stk, theta
            axes=[None, theta],
            names=['stk', 'theta'],
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
    '''
    def __init__(self, atm_filename, O3=None, H2O=None, NO2=True, P0=None):

        if atm_filename is None:
            return

        data = np.loadtxt(atm_filename, comments="#")

        self.z        = data[:,0] # Altitude in km
        self.P        = data[:,1] # pressure in hPa
        self.T        = data[:,2] # temperature in K
        self.dens_air = data[:,3] # Air density in cm-3
        self.dens_o3  = data[:,4] # Ozone density in cm-3
        self.dens_o2  = data[:,5] # O2 density in cm-3
        self.dens_h2o = data[:,6] # H2O density in cm-3
        self.dens_co2 = data[:,7] # CO2 density in cm-3
        self.dens_no2 = data[:,8] # NO2 density in cm-3

        # scale to specified total O3 content
        if O3 is not None:
            self.dens_o3 *= 2.69e16 * O3 / (simps(self.dens_o3, -self.z) * 1e5)

        # scale to total H2O content
        if H2O is not None:
            M_H2O = 18.015 # g/mol
            Avogadro = codata.value('Avogadro constant')
            self.dens_h2o *= H2O/ M_H2O * Avogadro / (simps(self.dens_h2o, -self.z) * 1e5)

        if P0 is not None:
            self.P *= P0/self.P[-1]

        if not NO2:
            self.dens_no2[:] = 0.


    def regrid(self, znew):
        '''
        regrid profile and returns a new profile
        '''

        prof = Profile_base(None)
        z = self.z
        prof.z = znew
        prof.P = interp1d(z, self.P)(znew)
        prof.T = interp1d(z, self.T)(znew)

        prof.dens_air = trapzinterp(self.dens_air, z, znew)
        prof.dens_o3  = trapzinterp(self.dens_o3, z, znew)
        prof.dens_o2  = trapzinterp(self.dens_o2, z, znew)
        prof.dens_h2o = trapzinterp(self.dens_h2o, z, znew)
        prof.dens_co2 = trapzinterp(self.dens_co2, z, znew)
        prof.dens_no2 = trapzinterp(self.dens_no2, z, znew)

        return prof

    def RH(self):
        '''
        returns profile of relative humidity for each layer
        '''
        rh = self.dens_h2o/vapor_pressure(self.T)*100.
        return rh



def change_altitude_grid(zOld, gridSpec):
    """ Setup a new altitude grid and interpolate profiles to new grid. """
    zFirst, zLast =  zOld[0], zOld[-1]
    #specs = re.split ('[-\s:,]+',gridSpec)
    if gridSpec.count('[')+gridSpec.count(']') == 0:
        if gridSpec.count(',')==0:
            try:
                deltaZ = float(gridSpec)
            except ValueError:
                raise Exception('z grid spacing not a number!')
            # set up new altitude grid
            zNew = np.arange(zFirst, zLast+deltaZ, deltaZ)
        elif gridSpec.count(',')==1:
            try:
                zLow,zHigh = map(float, gridSpec.split(','))
            except ValueError:
                raise Exception('z grid spacing not a pair of floats!')

            # for new grid simply extract old grid points within given bounds
            # (also include altitudes slightly outside)
            eps = min(zOld[1:]-zOld[:-1])/10.
            zNew = np.compress(np.logical_and(np.greater_equal(zOld,zLow-eps), np.less_equal(zOld,zHigh+eps)), zOld)
        elif gridSpec.count(',') == 2:
            try:
                zLow,zHigh,deltaZ = map(float, gridSpec.split(','))
            except ValueError:
                raise Exception('z grid spacing not a triple of floats (zLow.zHigh,deltaZ)!')

            # set up new altitude grid
            zNew = np.arange(max(zLow,zFirst), min(zHigh,zLast)+deltaZ, deltaZ)
        elif gridSpec.count(',')>2:
            try:
                zNew = np.array(map(float, gridSpec.split(',')))
            except ValueError:
                raise Exception('z grid not a set of floats separated by commas!')
    elif gridSpec.count('[') == gridSpec.count(']') > 0:
        zNew = parseGridSpec(gridSpec)
    if not zFirst <= zNew[0] < zNew[-1] <= zLast:
        pass
        #raise SystemExit, '%s  %f %f  %s  %f %f' % ('ERROR: new zGrid', zNew[0],zNew[-1], ' outside old grid', zFirst, zLast)
    else:
        raise Exception('New altitude not specified correctly\n'
                'either simply give altitude step size, a pair of lower,upper limits, or "start(step)stop"!')
    return zNew

def parseGridSpec (gridSpec):
    """ Set up (altitude) grid specified in format 'start[step1]stop1[step2]stop' or similar. """

    # get indices of left and right brackets
    lp = [];  rp = []
    for i in range(len(gridSpec)):
        if (gridSpec[i]=='['):
            lp.append(i)
        elif (gridSpec[i]==']'):
            rp.append(i)
        else:
            pass
    if len(lp) != len(rp):
        print('cannot parse grid specification\n'
                'number of opening and closing braces differs!\n'
                'Use format start[step]stop')
        raise SystemExit

    # parse
    gridStart = [];  gridStop = [];  gridStep = []
    for i in range(len(lp)):
        if i>0:  start=rp[i-1]+1
        else:    start=0
        if i<len(lp)-1: stop=lp[i+1]
        else:           stop=len(gridSpec)

        try:
            gridStart.append(float(gridSpec[start:lp[i]]))
        except ValueError:
            raise Exception('cannot parse grid start specification\nstring not a number!')
        try:
            gridStep.append(float(gridSpec[lp[i]+1:rp[i]]))
        except ValueError:
            raise Exception('cannot parse grid step specification\nstring not a number!')
        try:
            gridStop.append(float(gridSpec[rp[i]+1:stop]))
        except ValueError:
            raise Exception('cannot parse grid stop specification\nstring not a number!')

    # create the new grid (piecewise linspace)
    newGrid = []
    for i in range(len(lp)):
        n = int(round(abs((gridStop[i] - gridStart[i])/gridStep[i])))
        endpoint = (i == len(lp)-1)
        if endpoint: n += 1
        newGrid.extend(list(np.linspace(gridStart[i], gridStop[i], n, endpoint=endpoint)))

    return np.array(newGrid)


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
    Avogadro = codata.value('Avogadro constant')
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
    Avogadro = codata.value('Avogadro constant')
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

def rod(lam, co2, lat, z, P):
    ''' Rayleigh optical depth (N wavelengths x M layers)
        lam : um (N)
        co2 : ppm (M)
        lat : deg (scalar)
        z : m (M)
        P : hPa (M)
    '''
    Avogadro = codata.value('Avogadro constant')
    MA = ma(co2).reshape((1,-1))
    G = g(lat, z)
    return raycrs(lam, co2) * P*1e3 * Avogadro/MA/G



def diff1(A, axis=0, samesize=True):
    if samesize:
        B = np.zeros_like(A)
        key = [slice(None)]*A.ndim
        key[axis] = slice(1, None, None)
        B[key] = np.diff(A, axis=axis)[:]
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

