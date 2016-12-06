#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, division
import numpy as np
from tools.luts import LUT
from os.path import dirname, join, exists
from atmosphere import dir_libradtran_reptran
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
import os
import netCDF4
from itertools import product
from warnings import warn


def interp3(x, y, z, v, xi, yi, zi, **kwargs):
    """Sample a 3D array "v" with pixel corner locations at "x","y","z" at the
    points in "xi", "yi", "zi" using linear interpolation. Additional kwargs
    are passed on to ``scipy.ndimage.map_coordinates``."""
    assert v.ndim==3
    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
    xi, yi, zi = np.atleast_1d(xi, yi, zi)
    for arr in [xi, yi, zi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y, z], [xi, yi, zi])]

    map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)

def interp2(x, y, v, xi, yi, **kwargs):
    """Sample a 2D array "v" with pixel corner locations at "x","y", at the
    points in "xi", "yi",  using linear interpolation. Additional kwargs
    are passed on to ``scipy.ndimage.map_coordinates``."""
    assert v.ndim==2
    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
    xi, yi = np.atleast_1d(xi, yi)
    for arr in [xi, yi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y], [xi, yi])]

    map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)

def skipcomment(f):
    while(True):
        pos=f.tell()
        if not f.readline().strip().startswith('#'): break
    f.seek(pos,0)    


class KDIS(object):

    def __init__(self, model, dir_data, format='ascii'):

        # read the entire K-distribution definition from files
        #
        # Selection of the desired KDIS band or absorbing gases
        # must be done later while setting up the artdeco variables
   
        self.model = model
    
        if format == 'ascii':
            filename = dir_data+'kdis_'+model+'_def.dat'
            if not os.path.isfile(filename):
                print("(kdis_coef) ERROR")
                print("            Missing file:", filename)
                exit()
            fdef = open(filename,'r')
            skipcomment(fdef)
            tmp = fdef.readline()
            self.nmaxai = int(tmp.split()[0])
            skipcomment(fdef)
            tmp = fdef.readline()
            self.nsp     = int(tmp.split()[0])
            self.fcont   = np.zeros(self.nsp)
            self.species = []
            skipcomment(fdef)
            for i in range(self.nsp):
                tmp = fdef.readline()
                if int(tmp.split()[1]) == 1:
                    print(" kdis_coeff ERROR")
                    print("            read NOT implemented for concentration dependent species")
                    exit()
                self.species.append(tmp.split()[0])
                self.fcont[i] = float(tmp.split()[2])
            self.nsp_c = 0
            skipcomment(fdef)
            tmp = fdef.readline()
            self.nwvl = int(tmp.split()[0])
            self.wvlband = np.zeros((3, self.nwvl))
            skipcomment(fdef)
            for i in range(self.nwvl):
                tmp = fdef.readline()
                self.wvlband[0,i] = float(tmp.split()[1])*1e3 # from mic to nm
                self.wvlband[1,i] = float(tmp.split()[2])*1e3
                self.wvlband[2,i] = float(tmp.split()[3])*1e3
                if i>0:
                    if self.wvlband[0,i] < self.wvlband[0,i-1]:
                        print(" kdis_coeff ERROR")
                        print("            wavelengths must be sorted in increasing order")
                        exit()
            skipcomment(fdef)
            tmp = fdef.readline()
            skipcomment(fdef)
            self.np = int(tmp.split()[0])
            self.p = np.zeros(self.np)
            for i in range(self.np):
                tmp = fdef.readline()
                self.p[i] = float(tmp.split()[0])
                if i>0:
                    if self.p[i] < self.p[i-1]:
                        print(" kdis_coeff ERROR")
                        print("            pressure must be sorted in increasing order")
                        exit()
            skipcomment(fdef)
            tmp = fdef.readline()
            skipcomment(fdef)
            self.nt = int(tmp.split()[0])
            self.t = np.zeros(self.nt)
            for i in range(self.nt):
                tmp = fdef.readline()
                self.t[i] = float(tmp.split()[0])
                if i>0:
                    if self.t[i] < self.t[i-1]:
                        print(" kdis_coeff ERROR")
                        print("            temperature must be sorted in increasing order")
                        exit()
            fdef.close()
    
            self.nai   = np.zeros((self.nsp,self.nwvl), dtype='int')
            self.ki    = np.zeros((self.nsp,self.nwvl,self.nmaxai,self.np,self.nt))
            self.ai    = np.zeros((self.nsp,self.nwvl,self.nmaxai))
            self.xsect = np.zeros((self.nsp,self.nwvl))
    
            for isp in range(self.nsp):
                filename = dir_data+'kdis_'+model+'_'+self.species[isp]+'.dat'
                if not os.path.isfile(filename):
                    print("(kdis_coef) ERROR")
                    print("            Missing file:", filename)
                    exit()
                f = open(filename,'r')
                skipcomment(f)
                for iwvl in range(self.nwvl):
                    tmp = f.readline()
                    self.nai[isp,iwvl]   = int(tmp.split()[1])
                    self.xsect[isp,iwvl] = float(tmp.split()[2])
                for iwvl in range(self.nwvl):
                    if self.nai[isp,iwvl]>1:
                        skipcomment(f)
                        tmp = f.readline()
                        #print 'nai, nmaxai=',self.nai[isp,iwvl], self.nmaxai
                        for iai in range(self.nai[isp,iwvl]):
                            #print iai, float(tmp.split()[iai])
                            self.ai[isp,iwvl,iai] = float(tmp.split()[iai])
    
                        for it in range(self.nt):
                            for ip in range(self.np):
                                tmp = f.readline()
                                for iai in range(self.nai[isp,iwvl]):
                                    self.ki[isp,iwvl,iai,ip,it] = float(tmp.split()[iai])
    
                f.close()
            
            filename = dir_data+'kdis_'+model+'_'+'solarflux.dat'
            if not os.path.isfile(filename):
                    print("(kdis_coef) ERROR")
                    print("            Missing file:", filename)
                    exit()
            fsol = open(filename,'r')
            skipcomment(fsol)
            tmp = fsol.readline()
            skipcomment(fsol)
            tmp = fsol.readline()
            nn = float(tmp.split()[0])
            if nn != self.nwvl :
                print(" solar flux and kdis have uncompatible band number")
                exit()
            skipcomment(fsol)
            self.solarflux = np.zeros(self.nwvl)
            skipcomment(fsol)
            for i in range(self.nwvl):
                tmp = fsol.readline()
                self.solarflux[i] = float(tmp.split()[0])
                
            fsol.close()
            
        # support for multi species   

        self.nai_eff    = np.prod(self.nai, axis=0)
        self.nmaxai_eff = np.max(self.nai_eff)
        self.ai_eff  = np.zeros((self.nwvl, self.nmaxai_eff))
        self.iki_eff = np.zeros((self.nsp,self.nwvl,self.nmaxai_eff), dtype='int')

        for iwvl in range(self.nwvl):

            iai_eff = 0
            nested_list = []
            for isp in range(self.nsp):
                nested_list.append(range(self.nai[isp,iwvl]))
            for k in product(*nested_list):
                self.iki_eff[:,iwvl, iai_eff] = k
                self.ai_eff[iwvl, iai_eff]    = 1.0
                for isp in range(self.nsp):
                    if self.nai[isp,iwvl] > 1:
                        self.ai_eff[iwvl, iai_eff ] *= self.ai[isp,iwvl,self.iki_eff[isp,iwvl, iai_eff]]
                
                iai_eff += 1
            
                
    def nbands(self):
        '''
        number of bands
        '''
        return self.nwvl

    def band(self, band):
        '''
        returns a KDIS_BAND
        '''
        return KDIS_BAND(self, band)

    def bands(self):
        '''
        iterates over all bands
        '''
        for i in range(self.nbands()):
            yield self.band(i)    
            
    def to_smartg(self, lmin=None, lmax=None):
        '''
        return a list of KDIS_IBANDS for Smartg.run() method
        '''
        ik_l=[]
        for k in self.bands():
            if (lmin is None and lmax is None) or \
               (lmin is None and k.wmax <= lmax) or \
               (lmax is None and k.wmin >= lmin) or \
               (k.wmin >= lmin and k.wmax <= lmax) :
                for ik in k.ibands():
                    ik_l.append(ik)
        return KDIS_IBAND_LIST(ik_l)

    def get_weight(self):
        '''
        return weights, wavelengths, solarflux, band width and normalization in postprocessing
        '''
        wb_l=[]
        we_l=[]
        ex_l=[]
        dl_l=[]
        for k in self.bands():
            for ik in k.ibands():
                wb_l.append(ik.w)
                we_l.append(ik.weight)
                ex_l.append(ik.ex)
                dl_l.append(ik.dl)
        wb=LUT(np.array(wb_l),axes=[wb_l],names=['Wavelength'],desc='Wavelength')
        we=LUT(np.array(we_l),axes=[wb_l],names=['Wavelength'],desc='Weight')
        ex=LUT(np.array(ex_l),axes=[wb_l],names=['Wavelength'],desc='SolarFlux')
        dl=LUT(np.array(dl_l),axes=[wb_l],names=['Wavelength'],desc='BandWidth')
        norm = we.reduce(np.sum,'Wavelength',grouping=wb.data)
        return we, wb, ex, dl, norm
   
class KDIS_IBAND(object):
    '''
    KDIS internal band

    Arguments:
        band: KDIS_BAND object
        index: band index
        iband: internal band index
    '''
    def __init__(self, band, index):

        self.band = band     # parent KDIS_BAND
        self.index = index   # internal band index
        self.w = band.awvl[index]  # band wavelength
        self.ex= band.solarflux # solar irradiance
        self.dl= band.dl  #bandwidth
        self.xsect = band.xsect  # absorption or not
#        if self.xsect != 0 : self.weight =  band.awvl_weight[index]  # weight
#        else : self.weight = 1.
        self.weight =  band.awvl_weight[index]  # weight
        #self.species=['H2O','CO2','O3','N2O','CO','CH4','O2','N2']
           
                
    def ki_interp(P, T, ki, Pout, Tout):
        ki_t = np.zeros(len(T))
        for it in range(len(T)):
            fki        = interp1d(P, ki[:,it], kind='linear')
            ki_t[it] = fki(Pout)
        fki = interp1d(T, ki_t, kind='linear')
        return fki(Tout)
        
    
    def calc_profile(self, T, P, densmol):
        '''
        calculate a gaseous absorption profile for this internal band
        using temperature T and pressure P, and profile of molecular density of
        various gases stored in densmol
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
        raise Exception('Fix KDIS_IBAND.calc_profile following REPTRAN_IBAND.calc_profile')

        species = ['h2o', 'co2', 'o3', 'n2o', 'co', 'ch4', 'o2', 'n2']
        Nmol = 1
        M = len(T)
        datamol = np.zeros(M, np.float)

        assert densmol.shape[1] == 8
        assert len(T) == len(P)
        assert len(T) == densmol.shape[0]

        # for each gas
        for ig in range(self.band.kdis.nsp):
            specie = self.band.kdis.species[ig]
            ispecie= species.index(specie)
            if self.xsect[ig] == -1 :
                ikig = self.band.kdis.iki_eff[ig, self.band.band, self.index]
                tab = self.band.kdis.ki[ig, self.band.band, ikig, :, :]
                datamol += interp2(self.band.kdis.p, self.band.kdis.t, np.squeeze(tab), P, T) * densmol[:,ispecie]

        return datamol      
        
        
class KDIS_BAND(object):
    def __init__(self, kdis, band):

        self.kdis = kdis # parent kdis coeff
        self.band = band
        self.w = kdis.wvlband[0, self.band]
        self.wmin = kdis.wvlband[1, self.band]
        self.wmax = kdis.wvlband[2, self.band]
        #self.nband = kdis.nai[0, self.band] # the number of internal bands (representative bands) in this channel
        self.nband = kdis.nai_eff[self.band] # the number of internal bands (representative bands) in this channel
        #self.iband= np.arange(self.nband)
        self.awvl = [self.w]*self.nband # the corresponsing wavelenghts of the internal bands
        self.awvl_weight = kdis.ai_eff[self.band, :self.nband] # the weights of the internal bands for this channel
        #self.awvl_weight = kdis.ai[0, self.band, :self.nband] # the weights of the internal bands for this channel
        self.dl = (kdis.wvlband[2,self.band] - kdis.wvlband[1,self.band]) # bandwidth
        self.solarflux = kdis.solarflux[self.band]/self.dl # the extra terrestrial solar irradiance of the internal bands for this channel    
        self.xsect = kdis.xsect[:,self.band] # the source of absorption by species of the internal bands for this channel


    def iband(self, index):
        '''
        returns internal band by its number (starting at zero)
        '''
        return KDIS_IBAND(self, index)

    def ibands(self):
        '''
        iterate over each internal band
        '''
        for i in range(self.nband):
            yield self.iband(i)

class KDIS_IBAND_LIST(object):  
    '''
    KDIS list of IBANDS
    '''
    def __init__(self, l):
        self.l=l
        
    def get_weights(self):
        '''
        return weights, wavelengths, solarflux, band width and normalization in postprocessing
        '''
        wb_l=[]
        we_l=[]
        ex_l=[]
        dl_l=[]
        for ik in self.l:
                wb_l.append(ik.w)
                we_l.append(ik.weight)
                ex_l.append(ik.ex)
                dl_l.append(ik.dl)
        wb=LUT(np.array(wb_l),axes=[wb_l],names=['Wavelength'],desc='Wavelength')
        we=LUT(np.array(we_l),axes=[wb_l],names=['Wavelength'],desc='Weight')
        ex=LUT(np.array(ex_l),axes=[wb_l],names=['Wavelength'],desc='SolarFlux')
        dl=LUT(np.array(dl_l),axes=[wb_l],names=['Wavelength'],desc='BandWidth')
        norm = we.reduce(np.sum,'Wavelength',grouping=wb.data)
        return we, wb, ex, dl, norm    
        

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
        print(prof.dens_h2o)

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
        for bname in nc.variables['band_name'][:]:  # the names of the sensor channels
            self.band_names.append(bname.tostring().replace(' ', ''))

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
