#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import numpy as np
from numpy.core.fromnumeric import shape
from luts.luts import LUT, MLUT
from smartg.atmosphere import od2k, BPlanck
from scipy.integrate import quad, simps
import sys
from scipy.interpolate import interp1d
from scipy.interpolate import interpn
import os
from itertools import product
from smartg.tools.interp import interp2
import h5py

def reduce_kdis(mlut, ibands, use_solar=False, integrated=False, extern_weights=None):
    '''
    Compute the final spectral signal from mlut output of smart_g and
    KDIS_IBAND_LIST weights
    '''
    we, wb, ex, dl, norm, norm_dl = ibands.get_weights()
    res = MLUT()
    for l in mlut:
        for pref in ['I_','Q_','U_','V_','transmission','flux'] :
             if pref in l.desc:
                if extern_weights is not None:
                    tmp = l.desc
                    l = l*extern_weights 
                    l.desc = tmp
                if use_solar : lr = (l*we*ex*dl).reduce(np.sum,'wavelength',grouping=wb.data)
                else         : lr = (l*we*dl   ).reduce(np.sum,'wavelength',grouping=wb.data)
                if integrated: lr = lr/norm
                else         : lr = lr/norm_dl
                res.add_lut(lr, desc=l.desc)
    res.attrs = mlut.attrs
    return res



def Kdis_Emission(mlut, ibands):
    '''
    Return Thermal emission
    '''
    bsgroup = ibands.get_groups()
    kabs    = od2k(mlut, 'OD_abs_atm') * 1e-3 # m-1
    z       = -mlut.axis('z_atm') * 1e3 # m
    wmin = np.unique([ib.band.wmin for ib in ibands.l])
    wmax = np.unique([ib.band.wmax for ib in ibands.l])
    Avg_B  = np.zeros((len(wmin), len(z)))
    for i,(wmin,wmax) in enumerate(zip(wmin,wmax)):    
        for j,T in enumerate(mlut['T_atm'].data):
            lmin, lmax = wmin*1e-9, wmax*1e-9 # m
            dl         = wmax-wmin # nm
            Avg_B[i,j] = quad(BPlanck, lmin, lmax, args=T)[0]/(dl)
    Emission = LUT(kabs * Avg_B[bsgroup, :], 
               axes = [mlut.axis('wavelength'), z], 
               names= ['wavelength','z_atm'])

    return Emission


def Kdis_Avg_Emission(mlut, ibands):
    '''
    Return vertically integrated Thermal emission
    '''
    return (4*np.pi)*Kdis_Emission(mlut, ibands).reduce(simps, 'z_atm', x=-mlut.axis('z_atm') * 1e3)


class KDIS(object):

    def __init__(self, model, dir_data, format='ascii'):

        # read the entire K-distribution definition from files
        #
        # Selection of the desired KDIS band or absorbing gases
        # must be done later while setting up the artdeco variables
   
        self.model = model
    
        is_sorted = lambda a: np.all(a[:-1] <= a[1:])        
        
        if format == 'ascii':

            filename = dir_data+'kdis_'+model+'_def.dat'
            if not os.path.isfile(filename):
                print("(kdis_coef) ERROR")
                print("            Missing file:", filename)
                sys.exit()
            fdef = open(filename,'r')
            skipcomment(fdef)
            tmp = fdef.readline()
            self.nmaxai = int(tmp.split()[0])
            skipcomment(fdef)
            tmp = fdef.readline()
            self.nsp_tot =  int(tmp.split()[0])
            self.nsp     = 0
            self.fcont   = []
            self.species = []
            self.nsp_c     = 0
            self.fcont_c   = []
            self.species_c = []
            skipcomment(fdef)
            for i in range(self.nsp_tot):
                tmp = fdef.readline()
                if int(tmp.split()[1]) == 0:
                    self.nsp  = self.nsp  + 1
                    self.species.append(tmp.split()[0])
                    self.fcont.append( float(tmp.split()[2] ) )
                elif  int(tmp.split()[1]) == 1:
                    self.nsp_c  = self.nsp_c  + 1
                    self.species_c.append(tmp.split()[0])
                    self.fcont_c.append( float(tmp.split()[2] ) )
            self.fcont   = np.array(self.fcont)        
            self.fcont_c = np.array(self.fcont_c)        
            skipcomment(fdef)
            tmp = fdef.readline()
            self.nwvl = int(tmp.split()[0])
            self.wvlband = np.zeros((3, self.nwvl))
            skipcomment(fdef)
            for i in range(self.nwvl):
                tmp = fdef.readline()                
                self.wvlband[0,i] = float(tmp.split()[1])*1e3
                self.wvlband[1,i] = float(tmp.split()[2])*1e3
                self.wvlband[2,i] = float(tmp.split()[3])*1e3
                if i>0:
                    if self.wvlband[0,i] < self.wvlband[0,i-1]:
                        print(" kdis_coeff ERROR")
                        print("            wavelengths must be sorted in increasing order")
                        sys.exit()
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
                        sys.exit()
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
                        sys.exit()
            if self.nsp_c > 0:
                skipcomment(fdef)
                tmp = fdef.readline()
                skipcomment(fdef)
                self.nc = int(tmp.split()[0])
                self.c = np.zeros(self.nc)
                for i in range(self.nc):
                    tmp = fdef.readline()                
                    self.c[i] = float(tmp.split()[0])
                    if i>0:
                        if self.c[i] < self.c[i-1]:
                            print(" kdis_coeff ERROR")
                            print("            concentration must be sorted in increasing order")
                            sys.exit()
            fdef.close()
            if self.nsp > 0:
                self.nai   = np.zeros((self.nsp,self.nwvl), dtype='int')
                self.ki    = np.zeros((self.nsp,self.nwvl,self.nmaxai,self.np,self.nt))
                self.ai    = np.zeros((self.nsp,self.nwvl,self.nmaxai))
            if self.nsp_c > 0:
                self.nai_c   = np.zeros((self.nsp_c,self.nwvl), dtype='int')
                self.ki_c    = np.zeros((self.nsp_c,self.nwvl,self.nmaxai,self.np,self.nt,self.nc))
                self.ai_c    = np.zeros((self.nsp_c,self.nwvl,self.nmaxai))
            for isp in range(self.nsp):
                filename = dir_data+'kdis_'+model+'_'+self.species[isp]+'.dat'
                if not os.path.isfile(filename):
                    print("(kdis_coef) ERROR")
                    print("            Missing file:", filename)
                    sys.exit()                    
                f = open(filename,'r')
                skipcomment(f)
                for iwvl in range(self.nwvl):
                    tmp = f.readline()
                    self.nai[isp,iwvl]   = int(tmp.split()[1])
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
            if self.nsp_c > 0:
                self.c_desc = "density"
            else:
                self.c_desc = "none"
            for isp in range(self.nsp_c):
                filename = dir_data+'kdis_'+model+'_'+self.species_c[isp]+'.dat'
                if not os.path.isfile(filename):
                    print("(kdis_coef) ERROR")
                    print("            Missing file:", filename)
                    sys.exit()                    
                f = open(filename,'r')
                skipcomment(f)
                for iwvl in range(self.nwvl):
                    tmp = f.readline()
                    self.nai_c[isp,iwvl]   = int(tmp.split()[1])
                for iwvl in range(self.nwvl):
                    if self.nai_c[isp,iwvl]>1:
                        skipcomment(f)
                        tmp = f.readline()
                        for iai in range(self.nai_c[isp,iwvl]):
                            self.ai_c[isp,iwvl,iai] = float(tmp.split()[iai])  
                        for ic in range(self.nc):
                            for it in range(self.nt):
                                for ip in range(self.np):
                                    tmp = f.readline()
                                    for iai in range(self.nai_c[isp,iwvl]):
                                        self.ki_c[isp,iwvl,iai,ip,it,ic] = float(tmp.split()[iai])                                  
                f.close()
            
            filename = dir_data+'kdis_'+model+'_'+'solarflux.dat'
            if not os.path.isfile(filename):
                filename = dir_data+'solrad_'+'kdis_'+model+'_'+'thuillier2003.dat'
                if not os.path.isfile(filename):
                    print("(kdis_coef) ERROR")
                    print("            Missing file:", filename)
                    sys.exit()
            fsol = open(filename,'r')
            skipcomment(fsol)
            tmp = fsol.readline()
            skipcomment(fsol)
            tmp = fsol.readline()
            nn = float(tmp.split()[0])
            if nn != self.nwvl :
                print(" solar flux and kdis have uncompatible band number")
                sys.exit()
            skipcomment(fsol)
            self.solarflux = np.zeros(self.nwvl)
            skipcomment(fsol)
            for i in range(self.nwvl):
                tmp = fsol.readline()
                self.solarflux[i] = float(tmp.split()[0])
            fsol.close()
        
        elif format in ["h5","hdf5"]:

            filename = dir_data+'kdis_'+model+'.h5'
            f = h5py.File(filename,"r")
            self.nmaxai = np.copy(f["def"]["maxnai"])
            species_tot = list(f["coeff"].keys())
            self.nsp_tot =  len(species_tot)
            self.nsp     = 0
            self.fcont   = []
            self.species = []
            self.nsp_c     = 0
            self.fcont_c   = []
            self.species_c = []
            for isp, specie in enumerate(species_tot):
                if "rho_dep" in list(f["coeff"][specie].attrs.keys()):
                    if f["coeff"][specie].attrs['rho_dep']:
                        rho_dep = True
                    else:
                        rho_dep = False
                else:
                    rho_dep = False
                if not rho_dep:
                    self.nsp  = self.nsp  + 1
                    self.species.append(specie)
                    self.fcont.append( f["coeff"][specie].attrs['add_continuum'] )
                else:
                    self.nsp_c  = self.nsp_c  + 1
                    self.species_c.append(specie)
                    self.fcont_c.append(  f["coeff"][specie].attrs['add_continuum'] )
            self.fcont   = np.array(self.fcont)        
            self.fcont_c = np.array(self.fcont_c)        
            self.nwvl = len(f['def']['central_wvl'])
            self.wvlband = np.zeros((3, self.nwvl))
            self.wvlband[0,:] = np.copy(f['def']['central_wvl'])*1e3
            self.wvlband[1,:] = np.copy(f['def']['min_wvl'])*1e3
            self.wvlband[2,:] = np.copy(f['def']['max_wvl'])*1e3
            self.p = np.copy(f['def']['pressure'])
            self.t = np.copy(f['def']['temperature'])
            self.np = len(self.p)
            self.nt = len(self.t)            
            if self.nsp_c > 0:
                self.c = np.copy(f['def']['rho'])
                self.c_desc = f['def']['rho'].attrs["desc"].decode()
                self.nc = len(self.c)
                if not is_sorted(self.c):
                    print(" kdis_coeff ERROR")
                    print("            concentration must be sorted in increasing order")
                    sys.exit()
            else:
                self.c_desc = "none"
            if not is_sorted(self.wvlband[0,:]):
                print(" kdis_coeff ERROR")
                print("            (h5 format) read NOT implemented for concentration dependent species")
                sys.exit()
            if not is_sorted(self.p):
                print(" kdis_coeff ERROR")
                print("            pressure must be sorted in increasing order")
                sys.exit()
            if not is_sorted(self.t):
                print(" kdis_coeff ERROR")
                print("            temperature must be sorted in increasing order")
                sys.exit()
            if self.nsp>0:
                self.nai   = np.zeros((self.nsp,self.nwvl), dtype='int')
                self.ki    = np.zeros((self.nsp,self.nwvl,self.nmaxai,self.np,self.nt))
                self.ai    = np.zeros((self.nsp,self.nwvl,self.nmaxai))
                for isp, specie in enumerate(self.species):
                    self.nai[isp,:]      = np.copy(f["coeff"][specie]["nai"])
                    nai_tmp = np.nanmax(self.nai[isp,:])    
                    self.ki[isp,:,0:nai_tmp,:,:] = np.copy(f["coeff"][specie]["ki"][:,0:nai_tmp,:,:]) 
                    self.ai[isp,:,0:nai_tmp]     = np.copy(f["coeff"][specie]["ai"][:,0:nai_tmp]) 
            if self.nsp_c>0:
                self.nai_c   = np.zeros((self.nsp_c,self.nwvl), dtype='int')
                self.ki_c    = np.zeros((self.nsp_c,self.nwvl,self.nmaxai,self.np,self.nt,self.nc))
                self.ai_c    = np.zeros((self.nsp_c,self.nwvl,self.nmaxai))
                for isp, specie in enumerate(self.species_c):
                    self.nai_c[isp,:]      = np.copy(f["coeff"][specie]["nai"])
                    nai_tmp = np.nanmax(self.nai_c[isp,:])    
                    self.ki_c[isp,:,0:nai_tmp,:,:] = np.copy(f["coeff"][specie]["ki"][:,0:nai_tmp,:,:]) 
                    self.ai_c[isp,:,0:nai_tmp]     = np.copy(f["coeff"][specie]["ai"][:,0:nai_tmp]) 
            #solar flux
            grp  = f.require_group("solrad")
            self.solarflux = grp["solrad"][:]            
            f.close()

            for isp, specie in enumerate(self.species):
                self.species[isp] = self.species[isp].lower() 
            for isp, specie in enumerate(self.species_c):
                self.species_c[isp] = self.species_c[isp].lower()

        # support for multi species   
        if  (self.nsp>0) and (self.nsp_c>0) : 
            self.nai_eff    = np.prod( np.append(self.nai, self.nai_c, axis=0 ) , axis=0 )
        elif self.nsp>0 :
            self.nai_eff    = np.prod( self.nai , axis=0 )
        elif self.nsp_c>0 :
            self.nai_eff    = np.prod( self.nai_c, axis=0 )
        self.nmaxai_eff = np.max(self.nai_eff)        
        self.ai_eff     = np.zeros((self.nwvl,self.nmaxai_eff))
        if  self.nsp>0:
            self.iki_eff    = np.zeros((self.nsp,  self.nwvl,self.nmaxai_eff), dtype='int')
        if  self.nsp_c>0:    
            self.iki_eff_c  = np.zeros((self.nsp_c,self.nwvl,self.nmaxai_eff), dtype='int')
        for iwvl in range(self.nwvl):
            iai_eff = 0
            nested_list = []
            for isp in range(self.nsp):
                nested_list.append(range(self.nai[isp,iwvl]))
            for isp_c in range(self.nsp_c):
                nested_list.append(range(self.nai_c[isp_c,iwvl]))
            for k in product(*nested_list):
                if self.nsp>0:
                    self.iki_eff[:,iwvl, iai_eff]   = k[0:self.nsp]
                if self.nsp_c>0:
                    self.iki_eff_c[:,iwvl, iai_eff] = k[self.nsp:self.nsp+self.nsp_c]
                self.ai_eff[iwvl, iai_eff]    = 1.0                
                for isp in range(self.nsp):
                    if (self.nai[isp,iwvl] >= 1) and (self.ai[isp,iwvl,self.iki_eff[isp,iwvl,iai_eff]] != 0.0):
                        self.ai_eff[iwvl, iai_eff ] *= self.ai[isp,iwvl,self.iki_eff[isp,iwvl,iai_eff]]    
                for isp_c in range(self.nsp_c):
                    if (self.nai_c[isp_c,iwvl] >= 1) and (self.ai_c[isp_c,iwvl,self.iki_eff_c[isp_c,iwvl,iai_eff]]!= 0.0):
                        self.ai_eff[iwvl, iai_eff ] *= self.ai_c[isp_c,iwvl,self.iki_eff_c[isp_c,iwvl,iai_eff]]
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
            

    def to_smartg(self, include='', lmin=-np.inf, lmax=np.inf):
        '''
        return a list of KDIS_IBANDS for Smartg.run() method
        '''
        ik_l=[]

        if not isinstance(lmin,(list,np.ndarray)):
            lmin=[lmin]
            lmax=[lmax]
        for k in self.bands():
            for ii in range(len(lmin)):
                if (k.wmin >= lmin[ii]) and (k.wmax <= lmax[ii]):
                    for ik in k.ibands():
                        ik_l.append(ik)

        assert len(ik_l) != 0

        return KDIS_IBAND_LIST(sorted(ik_l, key=lambda x:x.w))


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
        wb=LUT(np.array(wb_l),axes=[wb_l],names=['wavelength'],desc='wavelength')
        we=LUT(np.array(we_l),axes=[wb_l],names=['wavelength'],desc='weight')
        ex=LUT(np.array(ex_l),axes=[wb_l],names=['wavelength'],desc='solarflux')
        dl=LUT(np.array(dl_l),axes=[wb_l],names=['wavelength'],desc='bandwidth')
        norm = we.reduce(np.sum,'wavelength',grouping=wb.data)
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
#        if self.xsect != 0 : self.weight =  band.awvl_weight[index]  # weight
#        else : self.weight = 1.
        self.weight =  band.awvl_weight[index]  # weight
        #self.species=['H2O','CO2','O3','N2O','CO','CH4','O2','N2']
           
                
    # def ki_interp(P, T, ki, Pout, Tout):
    #     ki_t = np.zeros(len(T))
    #     for it in range(len(T)):
    #         fki        = interp1d(P, ki[:,it], kind='linear')
    #         ki_t[it] = fki(Pout)
    #     fki = interp1d(T, ki_t, kind='linear')
    #     return fki(Tout)
        

    def calc_profile(self, prof):
        '''
        calculate a gaseous absorption profile for this internal band
        using temperature T and pressure P, and profile of molecular density of
        various gases stored in densmol
        '''

        species = ['h2o', 'co2', 'o3', 'n2o', 'co', 'ch4', 'o2', 'n2']
        T = prof.T
        P = prof.P
        Ngas = 8
        M = len(T)
        datamol = np.zeros(M, np.float)

        densmol = np.zeros((M, Ngas), np.float)
        densmol[:,0] = prof.dens_h2o
        densmol[:,1] = prof.dens_co2
        densmol[:,2] = prof.dens_o3
        densmol[:,3] = prof.dens_no2
        densmol[:,4] = prof.dens_co
        densmol[:,5] = prof.dens_ch4
        densmol[:,6] = prof.dens_o2
        densmol[:,7] = prof.dens_n2

    
        # for each gas
        for ig_c in range(self.band.kdis.nsp_c):
            specie_c = self.band.kdis.species_c[ig_c]
            ispecie_c= species.index(specie_c)
            ikig_c = self.band.kdis.iki_eff_c[ig_c, self.band.band, self.index]
            tab = self.band.kdis.ki_c[ig_c, self.band.band, ikig_c, :, :, :]
            points = ( self.band.kdis.p, self.band.kdis.t, self.band.kdis.c )
            if self.band.kdis.c_desc == "density":
                C = densmol[:,ispecie_c] 
            elif self.band.kdis.c_desc == "molar_fraction":
                C = densmol[:,ispecie_c] / prof.dens_air 
            C[C>np.max(self.band.kdis.c)]=np.max(self.band.kdis.c)*0.99
            C[C<np.min(self.band.kdis.c)]=np.min(self.band.kdis.c)*1.01            
            P[P>np.max(self.band.kdis.p)]=np.max(self.band.kdis.p)*0.99
            P[P<np.min(self.band.kdis.p)]=np.min(self.band.kdis.p)*1.01            
            T[T>np.max(self.band.kdis.t)]=np.max(self.band.kdis.t)*0.99
            T[T<np.min(self.band.kdis.t)]=np.min(self.band.kdis.t)*1.01            
            values = np.concatenate( (np.array([P]), np.array([T]), np.array([C])), axis=0).T
            #print(values)
            #print(interpn(points, tab, values))
            datamol +=  interpn(points, tab, values) * densmol[:,ispecie_c]

        for ig in range(self.band.kdis.nsp):
            specie = self.band.kdis.species[ig]
            ispecie= species.index(specie)
            ikig = self.band.kdis.iki_eff[ig, self.band.band, self.index]
            tab = self.band.kdis.ki[ig, self.band.band, ikig, :, :]
            datamol += interp2(self.band.kdis.p, self.band.kdis.t, np.squeeze(tab), P, T) * densmol[:,ispecie]

        return datamol*1e5


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
        wb=LUT(np.array(wb_l),axes=[wb_l],names=['wavelength'],desc='wavelength')
        we=LUT(np.array(we_l),axes=[wb_l],names=['wavelength'],desc='weight')
        ex=LUT(np.array(ex_l),axes=[wb_l],names=['wavelength'],desc='solarflux')
        dl=LUT(np.array(dl_l),axes=[wb_l],names=['wavelength'],desc='bandwidth')
        norm_dl = (we*dl).reduce(np.sum,'wavelength',grouping=wb.data)
        norm = we.reduce(np.sum,'wavelength',grouping=wb.data)
        return we, wb, ex, dl, norm, norm_dl    



    def get_groups(self):
        '''
        '''
        bsgroup=[]
        for iband in self.l:
            bsgroup.append(iband.band.band)
        bsgroup = np.array(bsgroup)
        return bsgroup-bsgroup[0]



def skipcomment(f):
    while(True):
        pos=f.tell()
        if not f.readline().strip().startswith('#'): break
    f.seek(pos,0)    
