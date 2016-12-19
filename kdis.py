#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
from tools.luts import LUT, MLUT
from os.path import dirname, join, exists
from scipy.interpolate import interp1d
import os
from itertools import product
from tools.interp import interp2, interp3


def reduce_kdis(mlut, ibands, use_solar=False, integrated=False):
    '''
    Compute the final spectral signal from mlut output of smart_g and
    KDIS_IBAND_LIST weights
    '''
    we, wb, ex, dl, norm, norm_dl = ibands.get_weights()
    res = MLUT()
    for l in mlut:
        for pref in ['I_','Q_','U_','V_','transmission','flux'] :
            if pref in l.desc:
                if use_solar : lr = (l*we*ex*dl).reduce(np.sum,'wavelength',grouping=wb.data)
                else         : lr = (l*we*dl   ).reduce(np.sum,'wavelength',grouping=wb.data)/norm
                if integrated: lr = lr/norm
                else         : lr = lr/norm_dl
                res.add_lut(lr, desc=l.desc)
    res.attrs = mlut.attrs
    return res

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
        for ig in range(self.band.kdis.nsp):
            specie = self.band.kdis.species[ig]
            ispecie= species.index(specie)
            if self.xsect[ig] == -1 :
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
        wb=LUT(np.array(wb_l),axes=[wb_l],names=['wavelength'],desc='wavelength')
        we=LUT(np.array(we_l),axes=[wb_l],names=['wavelength'],desc='weight')
        ex=LUT(np.array(ex_l),axes=[wb_l],names=['wavelength'],desc='solarflux')
        dl=LUT(np.array(dl_l),axes=[wb_l],names=['wavelength'],desc='bandwidth')
        norm_dl = (we*dl).reduce(np.sum,'wavelength',grouping=wb.data)
        norm = we.reduce(np.sum,'wavelength',grouping=wb.data)
        return we, wb, ex, dl, norm, norm_dl    

def skipcomment(f):
    while(True):
        pos=f.tell()
        if not f.readline().strip().startswith('#'): break
    f.seek(pos,0)    
