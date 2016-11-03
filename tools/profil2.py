#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO
# base class for atmospheric profile ?
#   -> CloudOPAC
#   -> AeroOPAC
#   -> other ?
# adaptation pour profile océanique ?
# controler les paramètres d'absorption
# controler l'épaisseur optique totale
# fournir des profiles de vapeur d'eau (et autres) "custom"
# fournir plusieurs composantes (aerosol, cloud, etc)



from __future__ import print_function, division, absolute_import
import numpy as np
from os.path import join, dirname, exists, basename
from glob import glob
from .luts import MLUT, LUT, read_mlut_netcdf4, Idx
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.constants import codata
import netCDF4
from warnings import warn
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
    '''
    def __init__(self, species):

        self.name = species
        fname = join(dir_libradtran_opac, 'optprop', species+'.cdf')
        if not exists(fname):
            raise Exception('file {} does not exist'.format(fname))

        nc = netCDF4.Dataset(fname)

        self._wav = nc.variables["wavelen"][:]*1e3  # wavelength (converted µm -> nm)

        if u'hum' in nc.variables.keys(): 
            self._rhgrid = nc.variables["hum"][:]
            self._nrh = len(self._rhgrid)
        else:
            self._rhgrid = None
            self._nrh = 1

        # density in g/cm^3
        # note: bypass the dimension lambda
        # constant values for all wavelengths
        self._rho = LUT(
                nc.variables["rho"][0,:],
                axes = [self._rhgrid],
                names = ['rh'],
            )

        # extinction coefficient (nlam, rh) in km^-1/(g/m^3)
        self._ext = LUT(
                    np.array(nc.variables["ext"][:]),
                    axes = [self._wav, self._rhgrid],
                    names = ['lambda', 'rh'],
                )

        # single scattering albedo (nlam, nhum)
        self._ssa = LUT(
                    np.array(nc.variables['ssa']),
                    axes = [self._wav, self._rhgrid],
                    names = ['lambda', 'rh'],
                )

        # scattering angle in degrees (nlam, nhum, nphamat, nthetamax)
        self._theta = LUT(
            nc.variables['theta'][:],
            axes=[self._wav, self._rhgrid, None, None],
            names=['lam', 'rh', 'stk', 'nthetamax'])


        # phase matrix (nlam, nhum, nphamat, nthetamax)
        self._phase = LUT(
            nc.variables['phase'][:],
            axes=[self._wav, self._rhgrid, None, None],
            names=['lam', 'rh', 'stk', 'nthetamax'])

        # number of scattering angles (nlam, nhum, nphamat)
        self._ntheta = LUT(
                nc.variables['ntheta'][:],
                axes=[self._wav, self._rhgrid, None],
                names=['lam', 'rh', 'nthetamax'])

        nc.close()

    def ext_ssa(self, wav, rh=None):
        '''
        returns the extinction coefficient and single scattering albedo of
        each layer
            (N x M) or (N x 1) if species does not depend on rh

        parameters:
            wav: array of wavelength in nm (N wavelengths)
            rh: relative humidity (M layers)
        '''
        assert isiterable(wav)

        if self._nrh > 1: # if the component properties depend on RH (thus Z)
            assert rh is not None

            [wav2, rh2] = np.broadcast_arrays(wav[:,None], rh[None,:])
            ext = self._ext[Idx(wav2), Idx(rh2, fill_value='extrema,warn')]
            ssa = self._ssa[Idx(wav2), Idx(rh2, fill_value='extrema,warn')]

            ext *= self._rho[Idx(rh)]/self._rho[Idx(50.)]


        else: # nothing depends on RH for this component
            assert self._ext.shape[1] == 1 # no dependency on reff

            # wavelength interpolation
            ext = self._ext[Idx(wav), 0]
            ssa = self._ssa[Idx(wav), 0]

            # create empty dimension for rh
            ext = ext[:,None]
            ssa = ssa[:,None]

        return ext, ssa

    def phase(self, wav, rh, NBTHETA):
        '''
        phase function of species at wavelengths wav
        resampled over NBTHETA angles
        '''

        theta = np.linspace(0., 180., num=NBTHETA)
        NLAM = len(wav)

        if self._nrh > 1:
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
            irh = 0

            # interpolate on wav and rh
            pha = self._phase.sub()[Idx(wav,round=True),irh,:,:]  # (lam, stk, nthetamax)
            th = self._theta[Idx(wav,round=True),irh,:,:]   # (lam, stk, nthetamax)
            nth = self._ntheta[Idx(wav,round=True),irh,:]   # (lam, stk)

            if (NBTHETA < nth).any():
                warn('Insufficient number of sampling angles for phase function')

            for ilam in xrange(NLAM):
                for istk in xrange(NPSTK):
                    nth_ = nth[ilam, istk]
                    P[ilam, 0, istk, :] = interp1d(th[ilam,istk,:nth_], pha[ilam,istk,:nth_])(theta)

        return LUT(P,
                   axes=[wav, None, None, theta],
                   names=['wav', 'altitude', 'stk', 'theta'],
                   attrs={'nrh': self._nrh})


class AeroOPAC(object):
    '''
    Initialize the Aerosol OPAC model

    Args:
        filename: name of the aerosol file. If no directory is specified,
                  assume directory <libradtran>/data/aerosol/OPAC/standard_aerosol_files
        tau_ref: optical thickness at wavelength wref
        w_ref: reference wavelength (nm) for aot
    '''
    def __init__(self, filename, tau_ref, w_ref):
        self.tau_ref = tau_ref
        self.w_ref = w_ref

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

    def dtau_ssa(self, wav, Z, rh):
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
            ext, ssa_ = s.ext_ssa(wav, rh)
            dtau_ = ext * dens * dZ
            dtau += dtau_

            ext, ssa_ = s.ext_ssa(w0,rh)
            dtau_ref += ext * dens * dZ

            ssa += dtau_ * ssa_

        ssa[dtau!=0] /= dtau[dtau!=0]

        # apply scaling factor to get the required optical thickness at the
        # specified wavelength
        dtau *= self.tau_ref/np.sum(dtau_ref)

        return dtau, ssa

    def phase(self, wav, Z, rh, NBTHETA=7201):
        '''
        Phase function calculation at wavelength wav and altitudes Z
        relative humidity is rh
        angle resampling over NBTHETA angles
        '''
        P = 0.
        dssa = 0.

        for i, s in enumerate(self.species):
            # integrate density along altitude
            dens = trapzinterp(
                    self.densities[:,i],
                    self.zopac, Z)

            # optical properties of the current species
            ext, ssa = s.ext_ssa(wav, rh)
            dtau = ext * dens * (-diff1(Z))

            dssa_ = dtau*ssa  # NLAM, ALTITUDE
            dssa += dssa_
            P += s.phase(wav, rh, NBTHETA)*dssa_[:,1:,None,None]  # (NLAM, ALTITUDE-1, NPSTK, NBTHETA)

        P.axes[1] = average(Z)    # FIXME: weight according to tau

        return P

    @staticmethod
    def list():
        '''
        list standard aerosol files in opac
        '''
        files = glob(join(dir_libradtran_opac, 'standard_aerosol_files', '*.dat'))
        return map(lambda x: basename(x)[:-4], files)


class AtmAFGL(object):
    '''
    Atmospheric profile definition using AFGL data

    Arguments:
        - atm_filename AFGL atmosphere file
          if provided without a directory, use default directory dir_libradtran_atmmod
        - comp: list of components objects (aerosol, clouds)
        - grid: new grid altitudes (list of decreasing altitudes in km)
        - lat: latitude (for Rayleigh optical depth calculation, default=45.)

        Gaseous absorption:
        - O3: total ozone column (Dobson units),
          or None to use atmospheric profile value (default)
        - H2O: total water vapour column (g.cm-2), or None to use atmospheric
          profile value (default)
        - NO2: activate NO2 absorption (default True)

        Phase functions definition:
        - pfwav: a list of wavelengths over which the phase functions are calculated
          default: None (all wavelengths)
        - pfgrid: altitude grid over which the phase function is calculated
          can be provided as an array of decreasing altitudes or a gridspec
          default value: [100, 0]
    '''
    def __init__(self, atm_filename, comp=[], grid=None, lat=45.,
                 O3=None, H2O=None, NO2=True,
                 pfwav=None, pfgrid=np.array([100., 0.])):

        self.lat = lat
        self.comp = comp
        self.pfwav = pfwav
        self.pfgrid = pfgrid
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
        prof = Profile_base(atm_filename, O3=O3, H2O=H2O, NO2=NO2)

        #
        # regrid profile if required
        #
        if grid is None:
            self.prof = prof
        else:
            if isinstance(grid, str):
                grid = change_altitude_grid(prof.z, grid)
            self.prof = prof.regrid(grid)

        #
        # calculate reduced profile
        # (for phase function blending)
        #
        self.prof_red = prof.regrid(pfgrid)    # TODO: only if necessary


    def calc(self, wav, phase=True):
        '''
        Profile and phase function calculation

        phase: boolean (activate phase function calculation)

        Returns: profile + phase function MLUT
        '''

        profile = self.calc_profile(wav)

        if phase:
            if self.pfwav is None:
                pha = self.phase(wav)
            else:
                pha = self.phase(self.pfwav)

            if pha is not None:
                profile.add_lut(pha, desc='phase')

            # fill profile with phase function indices (TODO)

        return profile


    def calc_profile(self, wav, prof=None):
        '''
        Calculate the profile of optical properties at given wavelengths
        wav: array of wavelength in nm
        prof: profile of densities (default: self.prof)

        returns: the profile of optical properties
        '''
        wav = to_array(wav)

        if prof is None:
            prof = self.prof

        dz = -diff1(prof.z)

        pro = MLUT()
        pro.add_axis('z', prof.z)
        pro.add_axis('wavelength', wav)

        #
        # Rayleigh optical thickness
        #
        # cumulated Rayleigh optical thickness (wav, z)
        tauray = rod(wav*1e-3, prof.dens_co2/prof.dens_air*1e6, self.lat,
                     prof.z*1e3, prof.P)
        assert tauray.ndim == 2

        # Rayleigh optical thickness
        dtaur = diff1(tauray, axis=1)
        pro.add_dataset('tau_r', tauray, axnames=['wavelength', 'z'],
                        attrs={'description':
                               'Cumulated rayleigh optical thickness'})

        #
        # Aerosol optical thickness and single scattering albedo
        #
        dtaua = np.zeros((len(wav), len(prof.z)), dtype='float32')
        ssa = np.zeros((len(wav), len(prof.z)), dtype='float32')
        for comp in self.comp:
            dtau_, ssa_ = comp.dtau_ssa(wav, prof.z, prof.RH())
            dtaua += dtau_
            ssa += dtau_ * ssa_
        ssa[dtaua!=0] /= dtaua[dtaua!=0]

        taua = np.cumsum(dtaua, axis=1)
        pro.add_dataset('tau_a', taua,
                        axnames=['wavelength', 'z'],
                        attrs={'description':
                               'Cumulated aerosol optical thickness at each wavelength'})
        pro.add_dataset('ssa', ssa, axnames=['wavelength', 'z'],
                        attrs={'description':
                               'Single scattering albedo'})


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
        taug.attrs['description'] = 'Cumulated gaseous optical thickness'
        pro.add_lut(taug, desc='taug')

        #
        # Total optical thickness and other parameters
        #
        tau_tot = tauray + taua + taug[:,:]
        pro.add_dataset('tau_tot', tau_tot,
                        axnames=['wavelength', 'z'],
                        attrs={'description':
                               'Cumulated total optical thickness'})

        tau_sca = np.cumsum(dtaur + dtaua*ssa, axis=1)
        pro.add_dataset('tau_sca', tau_sca,
                        axnames=['wavelength', 'z'],
                        attrs={'description':
                               'Cumulated scattering optical thickness'})

        tau_abs = np.cumsum(dtaug[:,:] + dtaua*(1-ssa), axis=1)
        pro.add_dataset('tau_abs', tau_abs,
                        axnames=['wavelength', 'z'],
                        attrs={'description':
                               'Cumulated absorption optical thickness'})

        pmol = dtaur/(dtaur + dtaua)
        pro.add_dataset('pmol', pmol,
                        axnames=['wavelength', 'z'],
                        attrs={'description':
                               'Ratio of molecular scattering to total scattering'})

        pabs = dtaug[:,:]/diff1(tau_tot, axis=1)
        pro.add_dataset('pabs', pabs,
                        axnames=['wavelength', 'z'],
                        attrs={'description':
                               'Ratio of gaseous absorption to total extinction'})

        return pro


    def phase(self, wav):
        '''
        Phase functions calculation at bands, using reduced profile
        '''
        wav = to_array(wav)
        prof_reduced = self.calc_profile(wav, self.prof_red)
        pha = 0.
        for comp in self.comp:
            warn('weigh the phase functions using prof_reduced')
            # print(comp)
            pha += comp.phase(wav, self.pfgrid, self.prof_red.RH())

        if len(self.comp) == 0:
            return None
        else:
            return pha

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
    '''
    def __init__(self, atm_filename, O3=None, H2O=None, NO2=True):

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


def to_array(x):
    '''
    Converts input to ndarray, from scalar, list or ndarray
    '''
    if isinstance(x, np.ndarray):
        r = x
    elif isinstance(x, list):
        r = np.array(x)
    else:
        assert isnumeric(x)
        r = np.array([x])

    try:
        r*2.
    except:
        raise Exception('Should be numeric ({})'.format(x))

    return r

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

if __name__ == "__main__":

    # aer = AeroOPAC('maritime_clean', 0.4, 550.)
    # aer._readStandardAerosolFile()

    # atm = AtmAFGL('afglms')
    # atm.profile([400., 500.]).describe()
    # atm = AtmAFGL('afglms', comp=[AeroOPAC('maritime_clean', 0.5, 550.)])
    # atm.profile(500.).describe()

    # Species('wc.sol.mie').dtau(wav=[400., 500.])
    # print '---'
    # read_opac_component('soot.mie').print_info()

    # TODO: test
    pass
