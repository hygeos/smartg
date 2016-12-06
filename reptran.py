#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, division
import numpy as np
from tools.luts import LUT, MLUT
from os.path import dirname, join
from atmosphere import dir_libradtran_reptran
from scipy.interpolate import interp1d
import netCDF4
from tools.interp import interp2, interp3


def reduce_reptran(mlut, ibands):
    '''
    Compute the final spectral signal from mlut output of smart_g and
    REPTRAN_IBAND_LIST weights
    '''
    we, wb, ex, dl, norm = ibands.get_weights()
    res = MLUT()
    for l in mlut:
        for pref in ['I_','Q_','U_','V_','transmission','flux'] :
            if pref in l.desc:
                    lr = (l*we).reduce(np.sum,'Wavelength',grouping=wb.data)/norm
                    res.add_lut(lr, desc=l.desc)
    res.attrs = mlut.attrs
    return res



class REPTRAN_IBAND(object):
    '''
    REPTRAN internal band

    Arguments:
        band: REPTRAN_BAND object
        index: band index
        iband: internal band index
    '''
    def __init__(self, band, index):

        self.band = band     # parent REPTRAN_BAND
        self.index = index   # internal band index
        self.w = band.awvl[index]  # band wavelength
        self._iband=band._iband[index]
        self.weight =  band.awvl_weight[index]  # weight
        self.extra = band.aextra[index]  # solar irradiance
        self.crs_source = band.across_section_source[index,:]  # table of absorbing gases
        self.species=['H2O','CO2','O3','N2O','CO','CH4','O2','N2']
        self.filename = band.filename

    def calc_profile(self, prof):
        '''
        calculate a gaseous absorption profile for this internal band
        using temperature T and pressure P, and profile of molecular density of
        various gases stored in the profile prof
            densmol has shape (Nlayer_atmo, Ngas)
            densmol[:,0]=gh2o.dens*gh2o.scalingfact
            densmol[:,1]=co2
            densmol[:,2]=go3.dens*go3.scalingfact
            densmol[:,3]=n2o
            densmol[:,4]=co
            densmol[:,5]=ch4
            densmol[:,6]=o2
            densmol[:,7]=n2
        '''
        Nmol = 8
        T = prof.T
        P = prof.P
        M = len(T)

        densmol = np.zeros((M, Nmol), np.float)
        densmol[:,0] = prof.dens_h2o
        densmol[:,1] = prof.dens_co2
        densmol[:,2] = prof.dens_o3
        densmol[:,3] = prof.dens_no2
        densmol[:,4] = prof.dens_co
        densmol[:,5] = prof.dens_ch4
        densmol[:,6] = prof.dens_o2
        densmol[:,7] = prof.dens_n2

        xh2o = prof.dens_h2o/prof.dens_air

        datamol = np.zeros(M, np.float)

        assert len(prof.T) == len(prof.P)

        # for each gas
        for ig in np.arange(Nmol):

            # si le gaz est absorbant a cette lambda
            if self.crs_source[ig]==1:

                # on recupere la LUT d'absorption
                crs_filename = self.filename[:-4] + '.lookup.' + self.species[ig]
                crs_mol = readCRS(crs_filename, self._iband)

                # interpolation du profil vertical de temperature de reference dans les LUT
                f = interp1d(crs_mol.pressure,crs_mol.t_ref)

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


class REPTRAN_BAND(object):
    def __init__(self, reptran, band):

        self.band = band
        self.nband = reptran.nwvl_in_band[self.band] # the number of internal bands (representative bands) in this channel
        self._iband = reptran.iwvl[:self.nband,self.band] # the indices of the internal bands within the wavelength grid for this channel
        self.awvl = reptran.wvl[self._iband-1] # the corresponsing wavelenghts of the internal bands
        self.awvl_weight = reptran.iwvl_weight[:self.nband,self.band] # the weights of the internal bands for this channel
        self.aextra = reptran.extra[self._iband-1] # the extra terrestrial solar irradiance of the internal bands for this channel
        self.across_section_source = reptran.cross_section_source[self._iband-1] # the source of absorption by species of the internal bands for this channel
        self.name = reptran.band_names[band]
        self.filename = reptran.filename    
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
        return REPTRAN_IBAND(self, index)

    def ibands(self):
        '''
        iterate over each internal band
        '''
        for i in range(self.nband):
            yield self.iband(i)
            

class REPTRAN(object):
    '''
    REPTRAN correlated-k file
    if provided without a directory, assuming dir_libradtran_reptran
    '''

    def __init__(self,filename):
        if dirname(filename) == '':
            self.filename = join(dir_libradtran_reptran, filename)
        else:
            self.filename = filename

        if not self.filename.endswith('.cdf'):
            self.filename += '.cdf'

        self._readFileGeneral()

    def _readFileGeneral(self):
        nc = netCDF4.Dataset(self.filename)
        self.wvl = nc.variables['wvl'][:] # the wavelength grid
        self.extra = nc.variables['extra'][:] # the extra terrestrial solar irradiance for the walength grid
        self.wvl_integral = nc.variables['wvl_integral'][:] # the wavelength integral (width) of each sensor channel
        self.nwvl_in_band = nc.variables['nwvl_in_band'][:] # the number of internal bands (representative bands) in each sensor channel
        self.iwvl = nc.variables['iwvl'][:] # the indices of the internal bands within the wavelength grid for each sensor channel
        self.iwvl_weight = nc.variables['iwvl_weight'][:] # the weight associated to each internal band
        self.cross_section_source = nc.variables['cross_section_source'][:] # for each internal band, the list of species that participated to the absorption computation 

        self.band_names = []
        for bname in nc.variables['band_name']:  # the names of the sensor channels
            self.band_names.append(str(bname.tostring()).replace(' ', ''))

    def nbands(self):
        '''
        number of bands
        '''
        return len(self.wvl_integral)

    def band(self, band):
        '''
        returns a REPTRAN_BAND
        band can be defined either by an integer, or a string
        '''
        if isinstance(band, str):
            return self.band(self.band_names.index(band))
        else:
            return REPTRAN_BAND(self, band)

    def bands(self):
        '''
        iterates over all bands
        '''
        for i in range(self.nbands()):
            yield self.band(i)

    def to_smartg(self, include='', lmin=-np.inf, lmax=np.inf):
        '''
        return a REPTRAN_IBAND_LIST for Smartg.run() method
        '''
        ik_l=[]
        for k in self.bands():
            if (include in k.name) and (k.wmin >= lmin) and (k.wmax <= lmax):
                for ik in k.ibands():
                    ik_l.append(ik)

        assert len(ik_l) != 0

        return REPTRAN_IBAND_LIST(sorted(ik_l, key=lambda x:x.w))

class REPTRAN_IBAND_LIST(object):
    '''
    REPTRAN LIST OF IBANDS
    '''

    def __init__(self, l):
        self.l=l

    def get_weights(self):
        '''
        return weights, wavelengths, solarflux, band width and normalization in postprocessing
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
        wb=LUT(np.array(wb_l),axes=[np.array(wi_l)],names=['Wavelength'],desc='Wavelength central band')
        we=LUT(np.array(we_l),axes=[np.array(wi_l)],names=['Wavelength'],desc='Weight')
        ex=LUT(np.array(ex_l),axes=[np.array(wi_l)],names=['Wavelength'],desc='E0')
        dl=LUT(np.array(dl_l),axes=[np.array(wi_l)],names=['Wavelength'],desc='Dlambda')
        norm = we.reduce(np.sum,'Wavelength',grouping=wb.data)
        return we, wb, ex, dl, norm 
        
    def get_names(self):
        '''
        return band names
        '''
        names=[]

        for iband in self.l:
            names.append(iband.band.name)

        return list(set(names))


class readCRS(object):
    def __init__(self,filename,iband):
        self.filename=filename
        self._readFileGeneral(iband)

    def _readFileGeneral(self,iband):
        nc=netCDF4.Dataset(join(dir_libradtran_reptran, self.filename+'.cdf'))
        self.wvl_index=nc.variables['wvl_index'][:]
        ii=list(self.wvl_index).index(iband)
        dat=nc.variables['xsec'][:]
        self.xsec=dat[:,:,ii,:]
        self.pressure=nc.variables['pressure'][:]
        self.t_ref=nc.variables['t_ref'][:]
        self.t_pert=nc.variables['t_pert'][:]
        self.vmrs=nc.variables['vmrs'][:]