#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, division, absolute_import
import numpy as np
from luts.luts import LUT, MLUT
import xarray as xr
from smartg.atmosphere import od2k, blackbody_radiance
from pathlib import Path
from scipy.integrate import quad, simpson
from smartg.config import DIR_AUXDATA
from scipy.interpolate import interp1d
import netCDF4
import warnings
from smartg.interp import interp2, interp3

dir_reptran = DIR_AUXDATA / 'reptran'

def reduce_reptran(mlut, ibands, use_solar=False, integrated=False, extern_weights=None):
    '''
    Compute the final spectral signal from an xarray Dataset and
    ReptranIbandList weights.

    MLUT input is supported temporarily for backwards compatibility and is
    converted to an xarray Dataset.
    '''
    if isinstance(mlut, MLUT):
        warnings.warn(
            "Passing an MLUT to reduce_reptran is deprecated; pass an "
            "xarray Dataset instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        mlut = mlut.to_xarray()
    elif isinstance(mlut, xr.DataArray):
        mlut = mlut.to_dataset(name=mlut.name or 'data')
    elif not isinstance(mlut, xr.Dataset):
        raise TypeError("mlut must be an xarray Dataset or an MLUT")

    we, wb, ex, dl, _, _ = ibands.get_weights(output_type='DataArray')
    wavelength = mlut.coords['wavelength']
    grouping = xr.DataArray(
        wb.to_numpy(),
        dims=('wavelength',),
        coords={'wavelength': wavelength},
        name='wavelength',
    )

    if extern_weights is not None:
        if isinstance(extern_weights, LUT):
            warnings.warn(
                "Passing a LUT as extern_weights is deprecated; pass an "
                "xarray DataArray instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            extern_weights = extern_weights.to_xarray()
        elif not isinstance(extern_weights, xr.DataArray):
            raise TypeError("extern_weights must be an xarray DataArray or LUT")

    factor = we * ex * dl if use_solar else we * dl
    norm = we.groupby(grouping).sum(dim='wavelength')
    norm_dl = (we * dl).groupby(grouping).sum(dim='wavelength')

    result = xr.Dataset(attrs=mlut.attrs)
    prefixes = ('I_', 'Q_', 'U_', 'V_', 'transmission', 'flux')
    for name, data_array in mlut.data_vars.items():
        description = data_array.attrs.get('desc', name)
        if not any(prefix in description for prefix in prefixes):
            continue

        attrs = dict(data_array.attrs)
        weighted = data_array * factor
        if extern_weights is not None:
            weighted = weighted * extern_weights

        reduced = weighted.groupby(grouping).sum(dim='wavelength')
        reduced = reduced / (norm if integrated else norm_dl)
        reduced.attrs = attrs
        result[name] = reduced

    return result


def reptran_emission(mlut, ibands):
    '''
    Return Thermal emission
    '''
    if hasattr(mlut, 'to_xarray'):
        mlut = mlut.to_xarray()

    z_axis = mlut.coords['z_atm'].to_numpy()
    wavelength_axis = mlut.coords['wavelength'].to_numpy()
    t_atm = mlut['T_atm'].to_numpy()

    bsgroup = ibands.get_groups()
    kabs    = od2k(mlut, 'OD_abs_atm') * 1e-3 # m-1
    z       = -z_axis * 1e3 # m
    wmin = np.unique([ib.band.wmin for ib in ibands.l])
    wmax = np.unique([ib.band.wmax for ib in ibands.l])
    Avg_B  = np.zeros((len(wmin), len(z)))
    for i,(wmin,wmax) in enumerate(zip(wmin,wmax)):    
        for j,T in enumerate(t_atm):
            lmin, lmax = wmin*1e-9, wmax*1e-9 # m
            dl         = wmax-wmin # nm
            Avg_B[i,j] = quad(blackbody_radiance, lmin, lmax, args=T)[0]/(dl)
    Emission = LUT(kabs * Avg_B[bsgroup, :], 
               axes = [wavelength_axis, z], 
               names= ['wavelength','z_atm'])
    return Emission


def reptran_avg_emission(mlut, ibands):
    '''
    Return vertically integrated Thermal emission
    '''
    if hasattr(mlut, 'to_xarray'):
        mlut = mlut.to_xarray()

    z_axis = mlut.coords['z_atm'].to_numpy()

    return (4*np.pi)*reptran_emission(mlut, ibands).reduce(simpson, 'z_atm', x=-z_axis * 1e3)



class ReptranIband(object):
    '''
    REPTRAN internal band

    Arguments:
        band: ReptranBand object
        index: band index
        iband: internal band index
    '''
    def __init__(self, band, index):

        self.band = band     # parent ReptranBand
        self.index = index   # internal band index
        self.w = band.awvl[index]  # band wavelength
        self._iband=band._iband[index]
        self.weight =  band.awvl_weight[index]  # weight
        self.extra = band.aextra[index]  # solar irradiance
        self.crs_source = band.across_section_source[index,:]  # table of absorbing gases
        self.species=['H2O','CO2','O3','N2O','CO','CH4','O2','N2']
        self.filename = Path(band.filename)

    def calc_profile(self, prof):
        '''
        calculate a gaseous absorption profile for this internal band
        using temperature T and pressure P, and profile of molecular density of
        various gases stored in the profile prof
        '''
        Nmol = 8
        T = prof.t
        P = prof.p
        M = len(T)

        densmol = np.zeros((M, Nmol), np.float64)
        densmol[:,0] = prof.dens_h2o
        densmol[:,1] = prof.dens_co2
        densmol[:,2] = prof.dens_o3
        densmol[:,3] = prof.dens_no2
        densmol[:,4] = prof.dens_co
        densmol[:,5] = prof.dens_ch4
        densmol[:,6] = prof.dens_o2
        densmol[:,7] = prof.dens_n2

        xh2o = prof.dens_h2o/prof.dens_air

        datamol = np.zeros(M, np.float64)

        assert len(prof.t) == len(prof.p)

        # for each gas
        for ig in np.arange(Nmol):

            # si le gaz est absorbant a cette lambda
            if self.crs_source[ig]==1:

                # on recupere la LUT d'absorption
                crs_filename = self.filename.with_suffix('')  # supprime l'extension
                crs_filename = crs_filename.with_name(f"{crs_filename.name}.lookup.{self.species[ig]}")
                crs_mol = ReadCrs(crs_filename, self._iband)

                # interpolation du profil vertical de temperature de reference dans les LUT
                f = interp1d(crs_mol.pressure,crs_mol.t_ref, fill_value='extrapolate')
                #f = interp1d(crs_mol.pressure,crs_mol.t_ref)

                # ecart en temperature par rapport au profil de reference (ou P de reference est en Pa et P AFGL en hPa)
                dT = T - f(P*100)

                if ig == 0 :  # si h2o
                    # interpolation dans la LUT d'absorption en fonction de
                    # pression, ecart en temperature et vmr de h2o et mutiplication par la densite,
                    # calcul de reptran avec LUT en 10^(-20) m2, passage en km-1
                    datamol += interp3(crs_mol.t_pert,crs_mol.vmrs,crs_mol.pressure,crs_mol.xsec,dT,xh2o,P*100) * densmol[:,ig] * 1e-11
                else:
                    tab = crs_mol.xsec
                    # interpolation dans la LUT d'absorption en fonction de
                    # pression, ecart en temperature et mutiplication par la densite,
                    # calcul de reptran avec LUT en 10^(-20) m2, passage en km-1 
                    datamol += interp2(crs_mol.t_pert,crs_mol.pressure,np.squeeze(tab),dT,P*100) * densmol[:,ig] * 1e-11

        return datamol


class ReptranBand(object):
    def __init__(self, reptran, band):

        self.band = band
        self.nband = reptran.nwvl_in_band[self.band] # the number of internal bands (representative bands) in this channel
        self._iband = reptran.iwvl[:self.nband,self.band] # the indices of the internal bands within the wavelength grid for this channel
        self.awvl = reptran.wvl[self._iband-1] # the corresponsing wavelenghts of the internal bands
        self.awvl_weight = reptran.iwvl_weight[:self.nband,self.band] # the weights of the internal bands for this channel
        self.aextra = reptran.extra[self._iband-1] # the extra terrestrial solar irradiance of the internal bands for this channel
        self.across_section_source = reptran.cross_section_source[self._iband-1] # the source of absorption by species of the internal bands for this channel
        self.name = reptran.band_names[band]
        self.filename = Path(reptran.filename)    
        # the wavelength integral (width) of this channel
        self.Rint = reptran.wvl_integral[self.band]
        
        try:
            self.wmin = float(self.name.split('to')[0].rstrip().split('bandfrom')[1])
            self.wmax = float(self.name.split('to')[1].rstrip().split('nm')[0])
        except:
            self.w    = np.mean(self.awvl)
            self.wmin = self.w - self.Rint/2.
            self.wmax = self.w + self.Rint/2.


    def iband(self, index):
        '''
        returns internal band by its number (starting at zero)
        '''
        return ReptranIband(self, index)

    def ibands(self):
        '''
        iterate over each internal band
        '''
        for i in range(self.nband):
            yield self.iband(i)
            

class Reptran(object):
    '''
    REPTRAN correlated-k file
    if provided without a directory, look to auxdata/reptran directory
    '''

    def __init__(self,filename):
        filename = Path(filename)
        if filename.parent == Path('.'):
            self.filename = dir_reptran / filename
        else:
            self.filename = filename

        if not filename.suffix == '.cdf':
            self.filename = self.filename.with_name(self.filename.name + '.cdf')

        self._read_file_general()

    def _read_file_general(self):
        nc = netCDF4.Dataset(self.filename)
        self.wvl = nc.variables['wvl'][:] # the wavelength grid
        if 'extra' in nc.variables.keys():
            self.extra = nc.variables['extra'][:] # the extra terrestrial solar irradiance for the walength grid
        else:
            self.extra = np.ones_like(self.wvl)
        self.wvl_integral = nc.variables['wvl_integral'][:] # the wavelength integral (width) of each sensor channel
        self.nwvl_in_band = nc.variables['nwvl_in_band'][:] # the number of internal bands (representative bands) in each sensor channel
        self.iwvl = nc.variables['iwvl'][:] # the indices of the internal bands within the wavelength grid for each sensor channel
        self.iwvl_weight = nc.variables['iwvl_weight'][:] # the weight associated to each internal band
        self.cross_section_source = nc.variables['cross_section_source'][:] # for each internal band, the list of species that participated to the absorption computation 

        self.band_names = []
        for bname in nc.variables['band_name']:  # the names of the sensor channels
            self.band_names.append(str(bname.tobytes()).replace(' ', ''))

    def nbands(self):
        '''
        number of bands
        '''
        return len(self.wvl_integral)

    def band(self, band):
        '''
        returns a ReptranBand
        band can be defined either by an integer, or a string
        '''
        if isinstance(band, str):
            return self.band(self.band_names.index(band))
        else:
            return ReptranBand(self, band)

    def bands(self):
        '''
        iterates over all bands
        '''
        for i in range(self.nbands()):
            yield self.band(i)

    def to_smartg(self, include='', lmin=-np.inf, lmax=np.inf,band_indices=None ):
        '''
        return a ReptranIbandList for Smartg.run() method
        '''
        ik_l=[]
        if band_indices is None:
            bl = self.bands()
        else:
            bl = [self.band(i) for i in band_indices]
            
        if not isinstance(lmin,(list,np.ndarray)):
            lmin=[lmin]
            lmax=[lmax]
        for k in bl:
            if (include in k.name):
                for ii in range(len(lmin)):
                    if (k.wmin >= lmin[ii]) and (k.wmax <= lmax[ii]):
                        for ik in k.ibands():
                            ik_l.append(ik)

        assert len(ik_l) != 0

        return ReptranIbandList(sorted(ik_l, key=lambda x:x.w))

class ReptranIbandList(object):
    '''
    Reptran list of internal bands
    '''

    def __init__(self, l):
        self.l=l

    def get_weights(self, output_type='LUT'):
        '''
        return weights, wavelengths, solarflux, bandwidth, bandwidth weighted normalization in postprocessing
        as MLUT objects

        Outputs:
        weights, wavelengths, solarflux, bandwidth, norm_bandwidth , norm

        '''
        we_l=[]
        ex_l=[]
        dl_l=[]
        wb_l=[]
        wi_l=[]
        for iband in self.l:
            #for iband in band.ibands():
                wi = iband.band.awvl[iband.index] # wvl of internal band
                wi_l.append(wi)
                we = iband.band.awvl_weight[iband.index] # weight of internal band
                we_l.append(we)
                ex = iband.band.aextra[iband.index] # E0 of internal band
                ex_l.append(ex)
                dl = iband.band.Rint # bandwidth
                dl_l.append(dl)
                wb = np.mean(iband.band.awvl[:])
                wb_l.append(wb)
        
        if output_type == 'LUT':
            wi_arr = np.array(wi_l, dtype=np.float32)
            wb=LUT(np.array(wb_l, dtype=np.float32),axes=[wi_arr],names=['wavelength'],desc='wavelength central band')
            we=LUT(np.array(we_l),axes=[wi_arr],names=['wavelength'],desc='Weight')
            ex=LUT(np.array(ex_l),axes=[wi_arr],names=['wavelength'],desc='E0')
            dl=LUT(np.array(dl_l),axes=[wi_arr],names=['wavelength'],desc='Dlambda')
            norm_dl = (we*dl).reduce(np.sum,'wavelength',grouping=wb.data)
            norm = we.reduce(np.sum,'wavelength',grouping=wb.data)
        elif output_type == 'DataArray':
            wi_arr = np.array(wi_l, dtype=np.float32)
            wb=xr.DataArray(np.array(wb_l, dtype=np.float32),dims=['wavelength'],coords={'wavelength': wi_arr},
                            name='wavelength',attrs={'desc': 'wavelength central band'})
            we=xr.DataArray(np.array(we_l),dims=['wavelength'],coords={'wavelength': wi_arr},
                            name='weight',attrs={'desc': 'Weight'})
            ex=xr.DataArray(np.array(ex_l),dims=['wavelength'],coords={'wavelength': wi_arr},
                            name='solarflux',attrs={'desc': 'E0'})
            dl=xr.DataArray(np.array(dl_l),dims=['wavelength'],coords={'wavelength': wi_arr},
                            name='bandwidth',attrs={'desc': 'Dlambda'})
            norm_dl = (we*dl).groupby('wavelength').sum(dim='wavelength')
            norm = we.groupby('wavelength').sum(dim='wavelength')
        else:
            raise ValueError("output_type must be either 'LUT' or 'DataArray'")
        
        return we, wb, ex, dl, norm, norm_dl 


    def get_groups(self):
        '''
        '''
        bsgroup=[]
        for iband in self.l:
            bsgroup.append(iband.band.band)
        bsgroup = np.array(bsgroup)
        return bsgroup-bsgroup[0]

        
    def get_names(self):
        '''
        return band names
        '''
        names=[]

        for iband in self.l:
            names.append(iband.band.name)

        return list(set(names))


class ReadCrs(object):
    def __init__(self,filename,iband):
        self.filename=Path(filename)
        self._read_file_general(iband)

    def _read_file_general(self,iband):
        nc=netCDF4.Dataset(dir_reptran / f'{self.filename.name}.cdf')
        self.wvl_index=nc.variables['wvl_index'][:]
        ii=list(self.wvl_index).index(iband)
        dat=nc.variables['xsec'][:]
        self.xsec=dat[:,:,ii,:]
        self.pressure=nc.variables['pressure'][:]
        self.t_ref=nc.variables['t_ref'][:]
        self.t_pert=nc.variables['t_pert'][:]
        self.vmrs=nc.variables['vmrs'][:]
