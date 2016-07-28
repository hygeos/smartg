#!/usr/bin/env python
# encoding: utf-8



import os
from os.path import join, dirname, exists, basename
from string  import count,split
import numpy as np
from optparse import OptionParser
from scipy.interpolate import interp1d
from scipy.integrate import romberg
from scipy.constants import codata
import netCDF4
from scipy.ndimage import map_coordinates
from scipy.integrate import simps
import tempfile
from phase_functions import PhaseFunction
from glob import glob
from smartg.tools.luts import LUT

dir_libradtran = '/home/applis/libRadtran/libRadtran-2.0/'
dir_libradtran_reptran =  join(dir_libradtran, 'data/correlated_k/reptran/')
dir_libradtran_opac =  join(dir_libradtran, 'data/aerosol/OPAC/')
dir_libradtran_atmmod = join(dir_libradtran, 'data/atmmod/')
dir_libradtran_crs = join(dir_libradtran, 'data/crs/')

NPSTK = 4 # number of Stokes parameters of the radiation field


            
            

class AeroOPAC(object):
    '''
    Initialize the Aerosol OPAC model

    Args:
        filename: name of the aerosol file. If no directory is specified,
                  assume directory <libradtran>/data/aerosol/OPAC/standard_aerosol_files
        tau: optical thickness at wavelength wref
        wref: reference wavelength (nm) for aot
        overwrite: recalculate and overwrite phase functions
    '''
    def __init__(self, filename, tau, wref, overwrite=True, RH=None, allow_regrid=True):

        self.__tau = tau
        self.__wref = wref
        self.overwrite = overwrite
        self.phases = None
        self.allow_regrid = allow_regrid

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

        self.scalingfact=1.
        self.scaleonly = False
        self._readStandardAerosolFile()

        # interpolated profiles, initialized by regrid()
        self.z = None
        self.dens = None

        
        if RH is not None : self.RH = RH
        else : self.RH = None

    @staticmethod
    def listStandardAerosolFiles():
        files = glob(join(dir_libradtran_opac, 'standard_aerosol_files', '*.dat'))
        return map(lambda x: basename(x)[:-4], files)

    def _readStandardAerosolFile(self):
        self.species=['inso','waso','soot','ssam','sscm','minm','miam',
                      'micm','mitr','suso','minm_spheroids',
                      'miam_spheroids','micm_spheroids','mitr_spheroids']
        self.scamatlist=[]
        data=np.loadtxt(self.filename)
        self.zopac=data[:,0] # altitudes du fichier de melange de composantes
        self.densities=data[:,1:] # profil vertical concentration massique (g/m3) des differentes composantes du modele
        for line in open(self.filename,'r').readlines():
            if line.startswith("z(km)",2,7):
                break
        self.aspecies=line.split()[2:] # lecture des nom des differentes composantes du modele
        self.ispecies=[]
        for species in self.aspecies:   # pour chaque composante on lit les proporites de diffusion de la LUT Scamat
            self.ispecies.append(self.species.index(species))
            self.scamatlist.append(ScaMat(species+'.mie'))   

    def init(self, z, T, h2o):
        '''
        Initialize the model using height profile, temperature and h2o conc.
        '''
        if self.allow_regrid : self.regrid(z)
        else : 
            self.dens = self.densities
            self.z = z
        self.__T = T
        self.__h2o = h2o
        if not self.scaleonly:
            self.setTauref(self.__tau, self.__wref)
            self.scalingfact = 1.

    def regrid(self,znew):
        '''
        reechantillonage vertical des concentrations massiques
        '''
        self.z = znew
        N=len(self.aspecies)
        M=len(znew)
        self.dens = np.zeros((M,N),np.float32)
        for k in range(N):
            self.dens[:,k] = trapzinterp(self.densities[:,k], self.zopac, znew)


    def calcTau(self,w): # calcul des propritees optiques du melange en fonction de l'alitude et aussi integrees sur la verticale


        M=len(self.z)
        if self.RH is None : 
            T = self.__T
            h2o = self.__h2o
            rh=h2o/vapor_pressure(T)*100 # calcul du profil vertical de RH
        else :
            rh = np.zeros(M,np.float32)
            rh[:] = self.RH
        self.dtau_tot=np.zeros(M,np.float32)
        k=0
        for scamat in self.scamatlist: 
            if scamat.nrh>1: # si les prorietes de la composante dependent de RH (donc de Z)
                rho=scamat.rho[0,:] # lecture du prfil de densite
                tabext=np.squeeze(scamat.ext) # tableau de la section efficace de diffusion (en km-1/(g/m3)) donnes pour RH=50%
                frho=interp1d(scamat.rhgrid,rho,bounds_error=False,fill_value=0.) # interpolation de la densite en fonction de RH
                for m in range(M):
                    if m==0:
                        dz=0.
                        dtau=0.
                    else:
                        ext0=interp2(scamat.wlgrid,scamat.rhgrid,tabext,w*1.e-3,rh[m]) # interpolation pour la longueur d'onde et la RH du niveau en cours
                        ext=ext0*frho(rh[m])/frho(50.)*self.dens[m,k] # calcul du coefficient de diffusion et ajustement pour RH du niveau
                        dz = self.z[m-1]-self.z[m]
                        dtau = dz * ext #* self.scalingfact # calcul de l'epaisseur optique du niveau, eventuellement mise a l'echelle
                    self.dtau_tot[m]+=dtau #somme sur les composantes

            else:  # idem mais rien de depend de RH pour cette composante
                tab=np.squeeze(scamat.ext)
                fext=interp1d(scamat.wlgrid,tab,bounds_error=False,fill_value=0.)
                ext0=fext(w*1e-3)
                for m in range(M):
                    if m==0:
                        dz=0.
                        dtau=0.
                    else:
                        ext=ext0*self.dens[m,k]
                        dz = self.z[m-1]-self.z[m]
                        dtau = dz * ext #* self.scalingfact
                    self.dtau_tot[m]+=dtau

            k=k+1

        self.tau_tot=np.sum(self.dtau_tot)

    def calc(self,w, NTHETA=7201):
        '''
        calcul des propritees optiques du melange en fonction de l'alitude et
        aussi integrees sur la verticale à la longueur d'onde w (nm)
        retourne le profil d'épaisseur optique intégrée et le profil de ssa
            - dataaer: le profile d'épaisseurs optiques totales
            - dtau_tot: le profile d'épaisseurs optiques de chaque couche
            - ssa_tot, l'albedo de diffusion simple de chaque couche
        '''
        M=len(self.z)
        if self.RH is None : 
            h2o = self.__h2o
            T = self.__T
            rh=h2o/vapor_pressure(T)*100 # calcul du profil vertical de RH
        else :
            rh = np.zeros(M,np.float32)
            rh[:] = self.RH
#        MMAX=5000 # Nb de polynome de Legendre au total maximum
        self.dtau_tot=np.zeros(M,np.float32)
        self.ssa_tot=np.zeros(M,np.float32)
#        self.pmom_tot=np.zeros((M,NPSTK,MMAX),np.float64)
        self.theta = np.linspace(0.,180.,num=NTHETA,endpoint=True,dtype=np.float64)
        self.phase_tot=np.zeros((M,NPSTK,NTHETA),np.float64)

        if not self.scaleonly: # if we have to compute all optical properties
            
            norm=np.zeros(M,np.float32)
            k=0
            for scamat in self.scamatlist:
                fiw=interp1d(scamat.wlgrid,np.arange(len(scamat.wlgrid))) # function to locate wavelength index in grid (float)
                iw=fiw(w*1e-3) # floating wavelength index 
                if scamat.nrh>1: # si les prorietes de la composante dependent de RH (donc de Z)
                    fir=interp1d(scamat.rhgrid,np.arange(len(scamat.rhgrid))) # function to locate RH index in grid 
                    rho=scamat.rho[0,:] # lecture du profil de densite
                    tabext=np.squeeze(scamat.ext) # tableau de la section efficace de diffusion (en km-1/(g/m3)) donnes pour RH=50%
                    frho=interp1d(scamat.rhgrid,rho,bounds_error=False,fill_value=0.) # interpolation de la densite en fonction de RH
                    for m in range(M):
                        ir=fir(rh[m]) # floating rh index
                        if m==0:
                            dz=0.
                            dtau=0.
                            dssa=1.
                            dp=[0.,0.,0.,0.]
                            nmax=[0,0,0,0]
                            ext=0.
                        else:
                            ext0=interp2(scamat.wlgrid,scamat.rhgrid,tabext,w*1.e-3,rh[m]) # interpolation pour la longueur d'onde et la RH du niveau en cours
                            tabssa=np.squeeze(scamat.ssa) # tableau des albedo de diffusion simple 
                            ssa=interp2(scamat.wlgrid,scamat.rhgrid,tabssa,w*1.e-3,rh[m]) # interpolation pour la longueur d'onde et la RH du niveau en cours
                            ext=ext0*frho(rh[m])/frho(50.)*self.dens[m,k] # calcul du coefficient de diffusion et ajustement pour RH du niveau
                            dz = self.z[m-1]-self.z[m]                       
                            dtau = dz * ext * self.scalingfact # calcul de l'epaisseur optique du niveau, eventuellement mise a l'echelle
                            dssa = dtau*ssa # ssa pondere par l'epsaissuer optique
                            norm[m]+=dssa
                            for n in range(NPSTK): # pour chaque element de la matrice de Stokes independant (NPSTK pour Mie) 
    #                            nmax=scamat.nmom[int(iw),int(ir),n]
    #                            dp[n]= scamat.pmom[int(iw),int(ir),n,:nmax]*dssa # plus proche voisin pour pmom pondere par ssa et tau de la composante
                                nmax=scamat.ntheta[int(iw),int(ir),n]
                                ftheta = interp1d(scamat.theta[int(iw),int(ir),n,:nmax],scamat.phase[int(iw),int(ir),n,:nmax]) # function to interpolate phase function
                                dp[n]= ftheta(self.theta)*dssa # plus proche voisin pour pmom pondere par ssa et tau de la composante
                        self.dtau_tot[m]+=dtau #somme sur les composantes
                        self.ssa_tot[m]+=dssa #moyenne pondere par l'epassieur optique pour ssa
                        for n in range(NPSTK):
    #                        nmax=scamat.nmom[int(iw),int(ir),n]
    #                        self.pmom_tot[m,n,:nmax]+=dp[n]
                            self.phase_tot[m,n,:]+=dp[n]
    
                else:  # idem mais rien de depend de RH pour cette composante
                    tabext=np.squeeze(scamat.ext)
                    fext=interp1d(scamat.wlgrid,tabext,bounds_error=False,fill_value=0.)                       
                    ext0=fext(w*1e-3)
                    tabssa=np.squeeze(scamat.ssa)
                    fssa=interp1d(scamat.wlgrid,tabssa,bounds_error=False,fill_value=0.)                       
                    ssa=fssa(w*1e-3)
    #                nmom=np.squeeze(scamat.nmom)
    #                pmom=np.squeeze(scamat.pmom)
                    ntheta=np.squeeze(scamat.ntheta)
                    theta=np.squeeze(scamat.theta)
                    phase=np.squeeze(scamat.phase)
    
                    for m in xrange(M):
                        if m==0:
                            dz=0.
                            dtau=0.
                            dssa=1.
                            dp=[0.,0.,0.,0.]
                            nmax=[0,0,0,0]
                            ext=0.
                        else:
                            ext=ext0*self.dens[m,k]
                            dz = self.z[m-1]-self.z[m]
                            dtau = dz * ext * self.scalingfact
                            dssa = dtau*ssa
                            norm[m]+=dssa
                            for n in range(NPSTK): # pour chaque element de la matrice de Stokes independant (NPSTK pour Mie) 
    #                            nmax=nmom[int(iw),n]
    #                            dp[n]= pmom[int(iw),n,:nmax]*dssa # plus proche voisin pour pmom pondere par ssa et tau de la composante
                                nmax=ntheta[int(iw),n]
                                ftheta = interp1d(theta[int(iw),n,:nmax],phase[int(iw),n,:nmax]) # function to interpolate phase function
                                dp[n]= ftheta(self.theta)*dssa # plus proche voisin pour phase pondere par ssa et tau de la composante
                                
                        self.dtau_tot[m]+=dtau
                        self.ssa_tot[m]+=dssa
                        for n in range(NPSTK):
    #                        nmax=nmom[int(iw),n]
    #                        self.pmom_tot[m,n,:nmax]+=dp[n]
                            self.phase_tot[m,n,:]+=dp[n]
    
                k=k+1 # each component
    
            for m in xrange(M):
                if m==0:
                    self.ssa_tot[m]=1.
                    for n in range(NPSTK):
    #                    self.pmom_tot[m,n,:]=0.
                        self.phase_tot[m,n,:]=0.
                else:
                    if (self.dtau_tot[m]>1e-8 and norm[m] > 1e-8): 
                        self.ssa_tot[m]/=self.dtau_tot[m]
                        for n in range(NPSTK):  
    #                        self.pmom_tot[m,n,:]/=norm[m]
                            self.phase_tot[m,n,:]/=norm[m]
                    else:
                        self.ssa_tot[m]=1.
                        for n in range(NPSTK):
    #                        self.pmom_tot[m,n,:]=0.
                            self.phase_tot[m,n,:]=0.

        else : # we just rescale the optical thicknesses
            for m in xrange(M):
                self.dtau_tot[m] *= self.scalingfact
            
        self.tau_tot=np.sum(self.dtau_tot)
#        self.MMAX=MMAX

        dataaer  = np.zeros(M, np.float)
        for m in xrange(M):
            dataaer[m] = np.sum(self.dtau_tot[:m+1])
       
        return (dataaer, self.dtau_tot, self.ssa_tot)


    def setTauref(self, tauref, wref): # On fixe l'AOT a une valeur pour une longueur d'onde de reference
        self.calcTau(wref) # calcul de l'AOT a la longueur d'onde de reference
        self.scalingfact=tauref/self.tau_tot # calcul du facteur d'echelle   
       
        

    def phase(self, wl, NTHETA=7201):
        '''
        returns the phase matrix for all layers

        wl is the wavelength in nm
        NTHETA: number of angles
        '''
        M=len(self.z)
        N=len(self.theta)
        pha=np.zeros((N, NPSTK),np.float64)
        if self.phases == None:
            self.phases = []
            for m in range(1, M):   # not the top boundary
                pha[:,0] = self.phase_tot[m,0,:] + self.phase_tot[m,1,:]
                pha[:,1] = self.phase_tot[m,0,:] - self.phase_tot[m,1,:]
                pha[:,2] = self.phase_tot[m,2,:]
                pha[:,3] = self.phase_tot[m,3,:]
                self.phases.append(PhaseFunction(self.theta, pha, degrees=True))

        return self.phases

    def __str__(self):
        return 'AER={base}-AOT={aot}'.format(base=self.basename, aot=self.__tau)
        
    def setphase(self, filename, standard=False):
        '''
        read the phase matrix for all layers

        filename where the phase matrix is stored (Smart-g format as defaut, otherwise standard)
        '''
        data = np.loadtxt(filename)
        M=len(self.z)
        theta = data[:,0]
        pha=data[:,1:]
        if standard:
            pha[:,0] = data[:,1] + data[:,2]
            pha[:,1] = data[:,1] - data[:,2]
            pha[:,2] = data[:,3]
            pha[:,3] = data[:,4]
        self.phases = []

        for m in range(1, M):   # not the top boundary
            self.phases.append(PhaseFunction(theta, pha, degrees=True))
        

class CloudOPAC(object):
    '''
    Initialize the Cloud OPAC model

    Args:
        basename: string for naming the "mixture"
        components: list of tuple of the cloud profile:
            tuple (species name, concentration, reff, zmin, zmax)
        tau: optical thickness at wavelength wref
        wref: reference wavelength (nm) for aot
        overwrite: recalculate and overwrite phase functions
    '''
    def __init__(self, basename, components, tau, wref, overwrite=False):

        self.__tau = tau
        self.__wref = wref
        self.basename = basename
        self.components = components
        self.overwrite = overwrite

        self.scalingfact=1.
        self._readStandardOPACComponent()

        # interpolated profiles, initialized by regrid()
        self.z = None
        self.dens = None

    @staticmethod
    def listStandardOPACComponents():
        files = glob(join(dir_libradtran_opac, 'optprop/', '*.cdf'))
        return map(lambda x: basename(x)[:-4], files)

    def _readStandardOPACComponent(self):
        self.scamatlist=[]
        self.aspecies = []
        self.reff = np.zeros(len(self.components),dtype=float)

        # FIXME
        # the following is only valid for 1 component
        # => need to properly deal with multiple components
        assert len(self.components) == 1, 'Not implemented for multiple components'

        for k in range(len(self.components)):
            component = self.components[k]
            (species, conc, reff, zmin, zmax) = component
            self.reff[k] = reff
            self.aspecies.append(species)
            self.scamatlist.append(ScaMat(species))

            self.zopac = np.array([zmax, zmax, zmin, zmin, 0.], dtype='f')
            self.densities = np.array([  0., conc, conc,   0., 0.], dtype='f').reshape((5,1))


    def init(self, z):
        '''
        Initialize the model using height profile
        '''
        self.scalingfact = 1.
        self.regrid(z)
        self.setTauref(self.__tau, self.__wref)

    def regrid(self,znew):
        '''
        reechantillonage vertical des concentrations massiques
        '''
        self.z = znew
        N=len(self.aspecies)
        M=len(znew)
        self.dens = np.zeros((M,N),np.float32)
        for k in range(N):
            self.dens[:,k] = trapzinterp(self.densities[:,k], self.zopac, znew)

    def calcTau(self,w): # calcul des propritees optiques du melange en fonction de l'alitude et aussi integrees sur la verticale
        M=len(self.z)
        self.dtau_tot=np.zeros(M,np.float32)
        k=0
        for scamat in self.scamatlist: 

            tab=np.squeeze(scamat.ext)
            if len(scamat.reffgrid)>1: # if the optical properties depnds on reff
                ext0=interp2(scamat.wlgrid,scamat.reffgrid,tab,w*1e-3,self.reff[k])
            else:
                fext=interp1d(scamat.wlgrid,tab,bounds_error=False,fill_value=0.)                       
                ext0=fext(w*1e-3)
            for m in range(M):
                if m==0:
                    dz=0.
                    dtau=0.
                else:
                    ext=ext0*self.dens[m,k]
                    dz = self.z[m-1]-self.z[m]
                    dtau = dz * ext * self.scalingfact
                self.dtau_tot[m]+=dtau

            k=k+1

        self.tau_tot=np.sum(self.dtau_tot)

    def calc(self,w,NTHETA=7201):
        '''
        calcul des propritees optiques du melange en fonction de l'alitude et
        aussi integrees sur la verticale à la longueur d'onde w (nm)
        retourne le profil d'épaisseur optique intégrée et le profil de ssa
        '''

        M=len(self.z)
        self.dtau_tot=np.zeros(M,np.float32)
        self.ssa_tot=np.zeros(M,np.float32)
#        MMAX=5000
#        self.pmom_tot=np.zeros((M,NPSTK,MMAX),np.float64)
        self.theta = np.linspace(0.,180.,num=NTHETA,endpoint=True,dtype=np.float64)
        self.phase_tot=np.zeros((M,NPSTK,NTHETA),np.float64)
        norm=np.zeros(M,np.float32)
        k=0
        for scamat in self.scamatlist:

#            MMAX2 = scamat.pmom.shape[-1]
            MMAX2 = scamat.phase.shape[-1]
            fiw=interp1d(scamat.wlgrid,np.arange(len(scamat.wlgrid))) # function to locate wavelength index in grid (float)
            iw=fiw(w*1e-3) # floating wavelength index 
            
            tabext=np.squeeze(scamat.ext)
            tabssa=np.squeeze(scamat.ssa)
            if len(scamat.reffgrid)>1: 
                ext0=interp2(scamat.wlgrid,scamat.reffgrid,tabext,w*1e-3,self.reff[k])
                ssa=interp2(scamat.wlgrid,scamat.reffgrid,tabssa,w*1e-3,self.reff[k])
                fir=interp1d(scamat.reffgrid,np.arange(len(scamat.reffgrid)))
                ir=int(fir(self.reff[k])) # nearest neighbour interpolation for reff on phase function
#                nmom=np.squeeze(scamat.nmom[:,ir])
#                pmom=np.squeeze(scamat.pmom[:,ir,:,:])
                ntheta=np.squeeze(scamat.ntheta[:,ir])
                theta=np.squeeze(scamat.theta[:,ir,:,:])
                phase=np.squeeze(scamat.phase[:,ir,:,:])
            else:
                fext=interp1d(scamat.wlgrid,tabext,bounds_error=False,fill_value=0.)
                ext0=fext(w*1e-3)
                fssa=interp1d(scamat.wlgrid,tabssa,bounds_error=False,fill_value=0.)
                ssa=fssa(w*1e-3)
#                nmom=np.squeeze(scamat.nmom)
#                pmom=np.squeeze(scamat.pmom)
                ntheta=np.squeeze(scamat.ntheta)
                theta=np.squeeze(scamat.theta)
                phase=np.squeeze(scamat.phase)

            for m in xrange(M):
                if m==0:
                    dz=0.
                    dtau=0.
                    dssa=1.
                    dp=[0.,0.,0.,0.]
                    nmax=[0,0,0,0]
                    ext=0.
                else:
                    ext=ext0*self.dens[m,k]
                    dz = self.z[m-1]-self.z[m]
                    dtau = dz * ext * self.scalingfact
                    dssa = dtau*ssa
                    norm[m]+=dssa
                    for n in range(NPSTK): # pour chaque element de la matrice de Stokes independant (NPSTK pour Mie) 
                        if ntheta.ndim==2:
#                        if nmom.ndim==2:
#                            nmax=nmom[int(iw),n]
                            nmax=ntheta[int(iw),n]
                            ftheta = interp1d(theta[int(iw),n,:nmax],phase[int(iw),n,:nmax]) # function to interpolate phase function
                            dp[n]= ftheta(self.theta)*dssa # plus proche voisin pour phase pondere par ssa et tau de la composante

                        else:
#                            nmax=nmom[int(iw)]
                            nmax=ntheta[int(iw)]
                            ftheta = interp1d(theta[int(iw),n,:nmax],phase[int(iw),n,:nmax]) # function to interpolate phase function
                            dp[n]= ftheta(self.theta)*dssa # plus proche voisin pour phase pondere par ssa et tau de la composante
                self.dtau_tot[m]+=dtau
                self.ssa_tot[m]+=dssa
                for n in range(NPSTK):
#                    if nmom.ndim==2:
#                        nmax=nmom[int(iw),n]
#                    else:
#                        nmax=nmom[int(iw)]

#                    nmax=min(MMAX2,nmax)
#                    self.pmom_tot[m,n,:nmax]+=dp[n]
                    self.phase_tot[m,n,:]+=dp[n]

            k=k+1 # each component

        for m in xrange(M):
            if m==0:
                self.ssa_tot[m]=1.
                for n in range(NPSTK):
#                    self.pmom_tot[m,n,:]=0.
                    self.phase_tot[m,n,:]=0.
            else:
                if (self.dtau_tot[m]>1e-8 and norm[m] > 1e-8): 
                    self.ssa_tot[m]/=self.dtau_tot[m]
                    for n in range(NPSTK):  
#                        self.pmom_tot[m,n,:]/=norm[m]
                        self.phase_tot[m,n,:]/=norm[m]
                else:
                    self.ssa_tot[m]=1.
                    for n in range(NPSTK):
#                        self.pmom_tot[m,n,:]=0.
                        self.phase_tot[m,n,:]=0.


        self.tau_tot=np.sum(self.dtau_tot)
        self.MMAX=MMAX2

        dataaer  = np.zeros(M, np.float)
        for m in xrange(M):
            dataaer[m] = np.sum(self.dtau_tot[:m+1])

        return (dataaer, self.dtau_tot, self.ssa_tot)


    def setTauref(self,tauref,wref): # On fixe l'AOT a une valeur pour une longueur d'onde de reference
        self.calcTau(wref) # calcul de l'AOT a la longueur d'onde de reference
        self.scalingfact=tauref/self.tau_tot # calcul du facteur d'echelle

        
    def phase(self, wl, NTHETA=7201):
        '''
        returns the phase matrix for all layers

        wl is the wavelength in nm
        NTHETA: number of angles
        '''
        M=len(self.z)
#        Leg=Legendres(self.MMAX,NTHETA)
        N=len(self.theta)
        pha=np.zeros((N, NPSTK),np.float64)
        phases = []

        for m in range(1, M):   # not the top boundary
#            theta, pha = Mom2Pha(self.pmom_tot[m,:,:],Leg)
#            phases.append(PhaseFunction(theta, pha, degrees=True))
            pha[:,0] = self.phase_tot[m,0,:] + self.phase_tot[m,1,:]
            pha[:,1] = self.phase_tot[m,0,:] - self.phase_tot[m,1,:]
            pha[:,2] = self.phase_tot[m,2,:]
            pha[:,3] = self.phase_tot[m,3,:]
            phases.append(PhaseFunction(self.theta, pha, degrees=True))

        return phases

    def __str__(self):
        return 'CLO={base}-AOT={aot}'.format(base=self.basename, aot=self.__tau)


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


def trapzinterp_alternate(y, x, xnew):
    '''
    alternate implementation of trapzinterp
    slower but simpler
    '''

    f = interp1d(x, y, bounds_error=False, fill_value=0.)

    integ = [f(xnew[0])]

    for i in xrange(len(xnew)-1):
        integ.append(romberg(f, a=xnew[i], b=xnew[i+1])/(xnew[i+1] - xnew[i]))

    return np.array(integ)



class Gas(object):
    def __init__(self,z,dens):
        self.z=z
        self.dens=dens
        self.scalingfact=1.
        self.initcol= self.calcColumn()
        self.Avogadro=codata.value('Avogadro constant')

    def calcColumn(self):
        return simps(self.dens,-self.z) * 1e5 

    def setColumn(self,DU=None, Dens=None):
        if DU != None : self.scalingfact = 2.69e16 * DU / self.calcColumn() 
        M_H2O=18.015
        if Dens !=None: self.scalingfact = Dens/ M_H2O * self.Avogadro/ self.calcColumn()

    def getDU(self):
        return self.calcColumn() / 2.69e16 * self.scalingfact

    def getDens(self):
        M_H2O=18.015
        return self.calcColumn() / self.Avogadro * M_H2O * self.scalingfact

    def regrid(self,znew):
        self.dens = trapzinterp(self.dens, self.z, znew)
        self.z = znew


class ScaMat(object):
    def __init__(self,species):
        self.species=species
        self._readScaMatFile()

    def _readScaMatFile(self):

        fname=join(dir_libradtran_opac, 'optprop', self.species+'.cdf')
        nc=netCDF4.Dataset(fname)
        self.wlgrid=nc.variables["wavelen"][:]

        self.theta=nc.variables["theta"][:]
        self.phase=nc.variables["phase"][:]
        self.pmom=nc.variables["pmom"][:]
        self.ntheta=nc.variables["ntheta"][:]
        self.nmom=nc.variables["nmom"][:]
        self.ext=nc.variables["ext"][:]
        self.ssa=nc.variables["ssa"][:]
        self.rho=nc.variables["rho"][:] 
        if u'reff' in nc.variables.keys(): 
            self.reffgrid=nc.variables["reff"][:]
        else :
            self.reffgrid=[1]
        if u'hum' in nc.variables.keys(): 
            self.rhgrid=nc.variables["hum"][:]
            self.nrh=len(self.rhgrid)
        else:
            self.rhgrid=1
            self.nrh=[1]


class Legendres(object):
    def __init__(self,nterm,ntheta):
        mu=np.linspace(0.,np.pi,ntheta,endpoint=True,dtype=np.float64)
        mu=np.cos(mu)
        un64=np.ones_like(mu)
        #zero64=np.ones_like(mu) !! FIXME DR
        zero64=np.zeros_like(mu)
        self.p1=np.zeros((nterm+1,ntheta),np.float64)
        self.p2=np.zeros((nterm+1,ntheta),np.float64)
        self.p1[0,:]=un64
        self.p1[1,:]=mu
        self.p2[0,:]=zero64
        self.p2[1,:]=zero64
        self.p2[2,:]=un64 * 3. * (1.-mu*mu)/(2. *np.sqrt(6.* un64))
        for k in range(nterm):
            dk=np.float64(k)
            if k>=1:
                self.p1[k+1,:]= ((2.*dk+1.) * mu * self.p1[k,:] - dk * self.p1[k-1,:] ) / (dk+1)
            if k>=2:
                PAR1=(2.*dk+1.)/np.sqrt((dk+3.*un64)*(dk-1.))
                PAR2=mu*self.p2[k,:]
                PAR3=(np.sqrt((dk+2.*un64)*(dk-2.)))/(2.*dk+1.)
                PAR4=self.p2[k-1,:]
                self.p2[k+1,:]=PAR1*(PAR2-PAR3*PAR4)
        self.mu=mu
        self.ntheta=ntheta
        self.nterm=nterm

def Mom2Pha(Mom,Leg):
    sumP=np.zeros_like(Leg.mu)
    sumQ=np.zeros_like(Leg.mu)
    sumU=np.zeros_like(Leg.mu)
    sumV=np.zeros_like(Leg.mu)
    pha=np.zeros((Leg.ntheta, NPSTK),np.float64)
    for k in range(Leg.nterm):
        sumP=sumP+Mom[0,k]*Leg.p1[k,:]
        sumQ=sumQ+Mom[1,k]*Leg.p2[k,:]
        sumU=sumU+Mom[2,k]*Leg.p1[k,:]
        sumV=sumV+Mom[3,k]*Leg.p1[k,:]
    pha[:,0]=sumP+sumQ
    pha[:,1]=sumP-sumQ
    pha[:,2]=sumU
    pha[:,3]=sumV
    return np.arccos(Leg.mu)/np.pi*180.,pha
    
    
def Pha2Pha(Pha):
    pha=np.zeros((Pha.shape()[-1], NPSTK),np.float64)
    pha[:,0] = Pha[0,:] + Pha[1,:]
    pha[:,0] = Pha[0,:] - Pha[1,:]
    pha[:,2] = Pha[2,:]
    pha[:,3] = Pha[3,:]
    return pha


def skipcomment(f):
    while(True):
        pos=f.tell()
        if not f.readline().strip().startswith('#'): break
    f.seek(pos,0)    

def getKey(custom):
    return custom.w
    
class KDIS(object):
    
    def __init__(self, model, dir_data, format='ascii'):

        # read the entire K-distribution definition from files
        #
        # Selection of the desired KDIS band or absorbing gases
        # must be done later while setting up the artdeco variables
   
        from itertools import product
        self.model = model
    
        if format == 'ascii':
            filename = dir_data+'kdis_'+model+'_def.dat'
            if not os.path.isfile(filename):
                print "(kdis_coef) ERROR"
                print "            Missing file:", filename
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
            for i in xrange(self.nsp):
                tmp = fdef.readline()
                if int(tmp.split()[1]) == 1:
                    print " kdis_coeff ERROR"
                    print "            read NOT implemented for concentration dependent species"
                    exit()
                self.species.append(tmp.split()[0])
                self.fcont[i] = float(tmp.split()[2])
            self.nsp_c = 0
            skipcomment(fdef)
            tmp = fdef.readline()
            self.nwvl = int(tmp.split()[0])
            self.wvlband = np.zeros((3, self.nwvl))
            skipcomment(fdef)
            for i in xrange(self.nwvl):
                tmp = fdef.readline()
                self.wvlband[0,i] = float(tmp.split()[1])*1e3 # from mic to nm
                self.wvlband[1,i] = float(tmp.split()[2])*1e3
                self.wvlband[2,i] = float(tmp.split()[3])*1e3
                if i>0:
                    if self.wvlband[0,i] < self.wvlband[0,i-1]:
                        print " kdis_coeff ERROR"
                        print "            wavelengths must be sorted in increasing order"
                        exit()
            skipcomment(fdef)
            tmp = fdef.readline()
            skipcomment(fdef)
            self.np = int(tmp.split()[0])
            self.p = np.zeros(self.np)
            for i in xrange(self.np):
                tmp = fdef.readline()
                self.p[i] = float(tmp.split()[0])
                if i>0:
                    if self.p[i] < self.p[i-1]:
                        print " kdis_coeff ERROR"
                        print "            pressure must be sorted in increasing order"
                        exit()
            skipcomment(fdef)
            tmp = fdef.readline()
            skipcomment(fdef)
            self.nt = int(tmp.split()[0])
            self.t = np.zeros(self.nt)
            for i in xrange(self.nt):
                tmp = fdef.readline()
                self.t[i] = float(tmp.split()[0])
                if i>0:
                    if self.t[i] < self.t[i-1]:
                        print " kdis_coeff ERROR"
                        print "            temperature must be sorted in increasing order"
                        exit()
            fdef.close()
    
            self.nai   = np.zeros((self.nsp,self.nwvl), dtype='int')
            self.ki    = np.zeros((self.nsp,self.nwvl,self.nmaxai,self.np,self.nt))
            self.ai    = np.zeros((self.nsp,self.nwvl,self.nmaxai))
            self.xsect = np.zeros((self.nsp,self.nwvl))
    
            for isp in xrange(self.nsp):
                filename = dir_data+'kdis_'+model+'_'+self.species[isp]+'.dat'
                if not os.path.isfile(filename):
                    print "(kdis_coef) ERROR"
                    print "            Missing file:", filename
                    exit()
                f = open(filename,'r')
                skipcomment(f)
                for iwvl in xrange(self.nwvl):
                    tmp = f.readline()
                    self.nai[isp,iwvl]   = int(tmp.split()[1])
                    self.xsect[isp,iwvl] = float(tmp.split()[2])
                for iwvl in xrange(self.nwvl):
                    if self.nai[isp,iwvl]>1:
                        skipcomment(f)
                        tmp = f.readline()
                        #print 'nai, nmaxai=',self.nai[isp,iwvl], self.nmaxai
                        for iai in xrange(self.nai[isp,iwvl]):
                            #print iai, float(tmp.split()[iai])
                            self.ai[isp,iwvl,iai] = float(tmp.split()[iai])
    
                        for it in xrange(self.nt):
                            for ip in xrange(self.np):
                                tmp = f.readline()
                                for iai in xrange(self.nai[isp,iwvl]):
                                    self.ki[isp,iwvl,iai,ip,it] = float(tmp.split()[iai])
    
                f.close()
            
            filename = dir_data+'kdis_'+model+'_'+'solarflux.dat'
            if not os.path.isfile(filename):
                    print "(kdis_coef) ERROR"
                    print "            Missing file:", filename
                    exit()
            fsol = open(filename,'r')
            skipcomment(fsol)
            tmp = fsol.readline()
            skipcomment(fsol)
            tmp = fsol.readline()
            nn = float(tmp.split()[0])
            if nn != self.nwvl :
                print " solar flux and kdis have uncompatible band number"
                exit()
            skipcomment(fsol)
            self.solarflux = np.zeros(self.nwvl)
            skipcomment(fsol)
            for i in xrange(self.nwvl):
                tmp = fsol.readline()
                self.solarflux[i] = float(tmp.split()[0])
                
            fsol.close()
            
        # support for multi species   

        self.nai_eff    = np.prod(self.nai, axis=0)
        self.nmaxai_eff = np.max(self.nai_eff)
        self.ai_eff  = np.zeros((self.nwvl, self.nmaxai_eff))
        self.iki_eff = np.zeros((self.nsp,self.nwvl,self.nmaxai_eff), dtype='int')

        for iwvl in xrange(self.nwvl):

            iai_eff = 0
            nested_list = []
            for isp in xrange(self.nsp):
                nested_list.append(range(self.nai[isp,iwvl]))
            for k in product(*nested_list):
                self.iki_eff[:,iwvl, iai_eff] = k
                self.ai_eff[iwvl, iai_eff]    = 1.0
                for isp in xrange(self.nsp):
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
        for i in xrange(self.nbands()):
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
        for it in xrange(len(T)):
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
        species = ['h2o', 'co2', 'o3', 'n2o', 'co', 'ch4', 'o2', 'n2']
        Nmol = 1
        M = len(T)
        datamol = np.zeros(M, np.float)

        assert densmol.shape[1] == 8
        assert len(T) == len(P)
        assert len(T) == densmol.shape[0]

        # for each gas
        for ig in xrange(self.band.kdis.nsp):
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
        for i in xrange(self.nband):
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

    def calc_profile(self, T, P, xh2o, densmol):
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
        Nmol = 8
        M = len(T)
        datamol = np.zeros(M, np.float)

        assert densmol.shape[1] == Nmol
        assert len(T) == len(P)
        assert len(T) == densmol.shape[0]

        # for each gas
        for ig in np.arange(Nmol):

            # si le gaz est absorbant a cette lambda
            if self.crs_source[ig]==1:

                # on recupere la LUT d'absorption
                crs_filename = self.filename[:-4] + '.lookup.' + self.species[ig]
                crs_mol = readCRS(crs_filename, self._iband)

                # interpolation du profil vertical de temperature de reference dans les LUT
                f=interp1d(crs_mol.pressure,crs_mol.t_ref)

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
        for i in xrange(self.nband):
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
        for i in xrange(self.nbands()):
            yield self.band(i)
            
    def to_smartg(self, lmin=None, lmax=None, include=''):
        '''
        return a REPTRAN_IBAND_LIST for Smartg.run() method
        '''
        ik_l=[]
        for k in self.bands():
            if (include in k.name) and ( \
               (lmin is None and lmax is None) or \
               (lmin is None and k.wmax <= lmax) or \
               (lmax is None and k.wmin >= lmin) or \
               (k.wmin >= lmin and k.wmax <= lmax)) :
                for ik in k.ibands():
                    ik_l.append(ik)
        
        return REPTRAN_IBAND_LIST(sorted(ik_l, key=lambda x:x.w))
            
class REPTRAN_IBAND_LIST(object):
    '''
    KDIS list of IBANDS
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
        wb=LUT(np.array(wb_l),axes=[wi_l],names=['Wavelength'],desc='Wavelength central band')
        we=LUT(np.array(we_l),axes=[wi_l],names=['Wavelength'],desc='Weight')
        ex=LUT(np.array(ex_l),axes=[wi_l],names=['Wavelength'],desc='E0')
        dl=LUT(np.array(dl_l),axes=[wi_l],names=['Wavelength'],desc='Dlambda')
        norm = we.reduce(np.sum,'Wavelength',grouping=wb.data)
        return we, wb, ex, dl, norm 
        

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

def vapor_pressure(T):
    T0=273.15
    A=T0/T
    Avogadro = codata.value('Avogadro constant')
    M_H2O=18.015
    mh2o=M_H2O/Avogadro
    return A*np.exp(18.916758 - A * (14.845878 + A*2.4918766))/mh2o/1.e6

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

# Rayleigh Optical thickness for Bodhaine et al. 1999
# Ozone Chappuis band cross section data from University of Bremen
# Atmospheric profile readers and Gridd parsing routine adapted from the Py4CATS soffware package

def from_this_dir(filename):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

def isnumeric(x) :
    try : 
        float(x)
        return True
    except ValueError :
        return False

def rho(F) :
    return (6.*F-6)/(3+7.*F)

def FF(rho):
    return (6+3*rho)/(6-7*rho)

def g0(lat) : 
    ''' gravity acceleration at the ground
        lat : deg
    '''
    return 980.6160 * (1. - 0.0026372 * np.cos(2*lat*np.pi/180.) + 0.0000059 * np.cos(2*lat*np.pi/180.)**2)

def g(lat,z) :
    ''' gravity acceleration at altitude z
        lat : deg
        z : m
    '''
    return g0(lat) - (3.085462 * 1.e-4 + 2.27 * 1.e-7 * np.cos(2*lat*np.pi/180.)) * z \
            + (7.254 * 1e-11 + 1e-13 * np.cos(2*lat*np.pi/180.)) * z**2  \
            - (1.517 * 1e-17 + 6 * 1e-20 * np.cos(2*lat*np.pi/180.)) * z**3

def zc(z) :
    ''' effective mass weighted altitude from US statndard
        z : m
    '''
    return 0.73737 * z + 5517.56

def FN2(lam) : 
    ''' depolarisation factor of N2
        lam : um
    '''
    return 1.034 + 3.17 *1e-4 *lam**(-2)

def FO2(lam) : 
    ''' depolarisation factor of O2
        lam : um
    '''
    return 1.096 + 1.385 *1e-3 *lam**(-2) + 1.448 *1e-4 *lam**(-4)

def Fair360(lam) : 
    ''' depolarisation factor of air for 360 ppm CO2
    '''
    return (78.084 * FN2(lam) + 20.946 * FO2(lam) +0.934 + 0.036 *1.15)/(78.084+20.946+0.934+0.036)

def PR(theta,rho):
    gam=rho/(2-rho)
    return 1/4./np.pi * 3/(4*(1+2*gam)) * ((1-gam)*np.cos(theta*np.pi/180.)**2 + (1+3*gam))

def Fair(lam,co2) : 
    ''' depolarisation factor of air for CO2
        lam : um
        co2 : ppm
    '''
    return (78.084 * FN2(lam) + 20.946 * FO2(lam) + 0.934 + co2*1e-4 *1.15)/(78.084+20.946+0.934+co2*1e-4)

def ma(co2):
    ''' molecular volume
        co2 : ppm
    '''
    return 15.0556 * co2*1e-6 + 28.9595

def n300(lam):
    ''' index of refraction of dry air  (300 ppm CO2)
        lam : um
    '''
    return 1e-8 * ( 8060.51 + 2480990/(132.274 - lam**(-2)) + 17455.7/(39.32957 - lam**(-2))) + 1.

def n(lam,co2):
    ''' index of refraction odf dry air
        lam : um
        co2 : ppm
    '''
    return (n300(lam)-1) * (1 + 0.54*(co2*1e-6 - 0.0003)) + 1.

def raycrs(lam,co2):
    ''' Rayleigh cross section
        lam : um
        co2 : ppm
    '''
    Avogadro = codata.value('Avogadro constant')
    Ns = Avogadro/22.4141 * 273.15/288.15 * 1e-3
    nn2 = n(lam,co2)**2
    return 24*np.pi**3 * (nn2-1)**2 /(lam*1e-4)**4/Ns**2/(nn2+2)**2 * Fair(lam,co2)

def rod(lam,co2,lat,z,P):
    ''' Rayleigh optical depth
        lam : um
        co2 : ppm
        lat : deg
        z : m
        P : hPa
    '''
    Avogadro = codata.value('Avogadro constant')
    return raycrs(lam,co2) * P*1e3 * Avogadro/ma(co2) /g(lat,z)

####################################################################################################################################

def change_altitude_grid(zOld, gridSpec):
        """ Setup a new altitude grid and interpolate profiles to new grid. """
        zFirst, zLast =  zOld[0], zOld[-1]
        #specs = re.split ('[-\s:,]+',gridSpec)
        if count(gridSpec,'[')+count(gridSpec,']')==0:
            if count(gridSpec,',')==0:
                try:                deltaZ = float(gridSpec)
                except ValueError:  raise SystemExit, 'z grid spacing not a number!'
                # set up new altitude grid
                zNew = np.arange(zFirst, zLast+deltaZ, deltaZ)
            elif count(gridSpec,',')==1:
                try:                zLow,zHigh = map(float,split(gridSpec,','))
                except ValueError:  raise SystemExit, 'z grid spacing not a pair of floats!'
                # for new grid simply extract old grid points within given bounds (also include altitudes slightly outside)
                eps  = min( zOld[1:]-zOld[:-1] ) / 10.
                zNew = np.compress(np.logical_and(np.greater_equal(zOld,zLow-eps), np.less_equal(zOld,zHigh+eps)), zOld)
            elif count(gridSpec,',')==2:
                try:                zLow,zHigh,deltaZ = map(float,split(gridSpec,','))
                except ValueError:  raise SystemExit, 'z grid spacing not a triple of floats (zLow.zHigh,deltaZ)!'
                # set up new altitude grid
                zNew = np.arange(max(zLow,zFirst), min(zHigh,zLast)+deltaZ, deltaZ)
            elif count(gridSpec,',')>2:
                try:                zNew = np.array(map(float, split(gridSpec,',')))
                except ValueError:  raise SystemExit, 'z grid not a set of floats separated by commas!'
        elif count(gridSpec,'[')==count(gridSpec,']') > 0:
              zNew = parseGridSpec (gridSpec)
        if not zFirst <= zNew[0] < zNew[-1] <= zLast:
            pass 
            #raise SystemExit, '%s  %f %f  %s  %f %f' % ('ERROR: new zGrid', zNew[0],zNew[-1], ' outside old grid', zFirst, zLast)
        else:
               raise SystemExit, 'New altitude not specified correctly\n' + \
                     'either simply give altitude step size, a pair of lower,upper limits,  or "start(step)stop"!'
        return zNew

####################################################################################################################################
def parseGridSpec (gridSpec):
    """ Set up (altitude) grid specified in format 'start[step1]stop1[step2]stop' or similar. """

    # get indices of left and right brackets
    lp = [];  rp = []
    for i in xrange(len(gridSpec)):
        if   (gridSpec[i]=='['):  lp.append(i)
        elif (gridSpec[i]==']'):  rp.append(i)
        else:                     pass
    if len(lp) != len(rp):
        print 'cannot parse grid specification\nnumber of opening and closing braces differs!\nUse format start[step]stop'
        raise SystemExit

    # parse
    gridStart = [];  gridStop = [];  gridStep = []
    for i in xrange(len(lp)):
        if i>0:  start=rp[i-1]+1
        else:    start=0
        if i<len(lp)-1: stop=lp[i+1]
        else:           stop=len(gridSpec)

        try:
            gridStart.append(float(gridSpec[start:lp[i]]))
        except ValueError:
            print 'cannot parse grid start specification\nstring not a number!'
            raise SystemExit
        try:
            gridStep.append(float(gridSpec[lp[i]+1:rp[i]]))
        except ValueError:
            print 'cannot parse grid step specification\nstring not a number!'
            raise SystemExit
        try:
            gridStop.append(float(gridSpec[rp[i]+1:stop]))
        except ValueError:
            print 'cannot parse grid stop specification\nstring not a number!'
            raise SystemExit

    # create the new grid (piecewise linspace)
    newGrid = []
    for i in xrange(len(lp)):
        n = int(round(abs((gridStop[i] - gridStart[i])/gridStep[i])))
        endpoint = (i == len(lp)-1)
        if endpoint: n += 1
        newGrid.extend(list(np.linspace(gridStart[i], gridStop[i], n, endpoint=endpoint)))

    return np.array(newGrid)



####################################################################################################################################

class Profile(object):
    '''
    Initialize profile generator
    Arguments:
        - atm_filename AFGL atmosphere file
          if provided without a directory, use default directory dir_libradtran_atmmod
        - aer: Aerosol object which provides the aerosol profile and phase
          function.
          if None, AOT=0.
        - grid: custom grid. Can be provided as an array of decreasing altitudes or a gridSpec (string)
          default value: None (use default grid)
        - pfgrid: phase function grid, the grid over which the phase function is calculated
          can be provided as an array of decreasing altitudes or a gridspec
          default value: [100, 0]
        - pfwav: a list of wavelengths over which the phase functions are calculated
          default: None (all wavelengths)
        - tauR: Rayleigh optical thickness, default None computed from atmospheric profile and wavelength
        - ssa : arbitrarily set the aerosol signle scatering albedo to a constant value
        - lat: latitude (for Rayleigh optical depth calculation, default=45.)
        - O3: total ozone column (Dobson units), or None to use atmospheric
          profile value (default)
        - H2O: total water vapour column (mol cm-2), or None to use atmospheric
          profile value (default)
        - NO2: activate NO2 absorption (default True)
        - P0: Pressure at the sea level in <TODO>
              default: surface pressure from AFGL file
    '''
    def __init__(self, atm_filename, aer=None, grid=None, cloud=None,
                pfgrid=[100., 0.], pfwav=None, tauR=None, ssa=None,
                lat=45., O3=None, H2O=None, NO2=True, verbose=False, 
                overwrite=False, P0=None):

        self.atm_filename = atm_filename
        self.pfwav = pfwav
        self.pfgrid = pfgrid
        self.cache_phase_keys = []
        self.cache_phase_values = []
        self.cache_prof_keys = []
        self.cache_prof_values = []


        crs_O3_filename = join(dir_libradtran_crs, 'crs_O3_UBremen_cf.dat')
        crs_NO2_filename = join(dir_libradtran_crs, 'crs_NO2_UBremen_cf.dat')
        ch4_filename = join(dir_libradtran_atmmod, 'afglus_ch4_vmr.dat')
        co_filename = join(dir_libradtran_atmmod, 'afglus_co_vmr.dat')
        n2o_filename = join(dir_libradtran_atmmod, 'afglus_n2o_vmr.dat')
        n2_filename = join(dir_libradtran_atmmod, 'afglus_n2_vmr.dat')

        if dirname(atm_filename) == '':
            atm_filename = join(dir_libradtran_atmmod, atm_filename)
        if (not exists(atm_filename)) and (not atm_filename.endswith('.dat')):
            atm_filename += '.dat'

        self.basename = basename(atm_filename)
        if self.basename.endswith('.dat'):
            self.basename = self.basename[:-4]

        # lecture du fichier atmosphere AFGL
        data = np.loadtxt(atm_filename, comments="#")
        z = data[:,0] # en km
        self.P = data[:,1] # en hPa
        self.T = data[:,2] # en K
        self.air = data[:,3] # Air density en cm-3
        o3 = data[:,4] # Ozone density en cm-3
        self.o2 = data[:,5] # O2 density en cm-3
        h2o = data[:,6] # H2O density en cm-3
        self.co2 = data[:,7] # CO2 density en cm-3
        self.no2 = data[:,8] # NO2 density en cm-3

        if (P0 != None):
            self.P *= P0/self.P[-1]


        # lecture des fichiers US Standard atmosphere pour les autres gaz
        datach4 = np.loadtxt(ch4_filename, comments="#")
        self.ch4=datach4[:,1] * self.air # CH4 density en cm-3
        dataco = np.loadtxt(co_filename, comments="#")
        self.co=dataco[:,1] * self.air # CO density en cm-3
        datan2o = np.loadtxt(n2o_filename, comments="#")
        self.n2o=datan2o[:,1] * self.air # N2O density en cm-3
        datan2 = np.loadtxt(n2_filename, comments="#")
        self.n2=datan2[:,1] * self.air # N2 density en cm-3

        # lecture du fichier crs de l'ozone dans les bandes de Chappuis
        self.crs_chappuis = np.loadtxt(crs_O3_filename, comments="#")
        self.crs_no2 = np.loadtxt(crs_NO2_filename, comments="#")

        self.go3 = Gas(z,o3)
        self.go3.setColumn(DU=O3)
        self.gh2o = Gas(z,h2o)
        self.gh2o.setColumn(Dens=H2O)

        #-------------------------------------------
        # optionnal regrid
        if grid != None:
            if isinstance(grid, str):
                znew = change_altitude_grid(z, grid)
            else:
                znew = grid

            self.P = interp1d(z, self.P)(znew)
            self.T = interp1d(z, self.T)(znew)
            airnew = trapzinterp(self.air, z, znew)
            o3 = trapzinterp(o3, z, znew)
            h2o = trapzinterp(h2o, z, znew)
            self.o2 = trapzinterp(self.o2, z, znew)
            #self.h2o = trapzinterp(self.h2o, z, znew)
            self.co2 = trapzinterp(self.co2, z, znew)
            self.no2 = trapzinterp(self.no2, z, znew)
            self.ch4 = trapzinterp(self.ch4/self.air, z, znew)*airnew
            self.co = trapzinterp(self.co/self.air, z, znew)*airnew
            self.n2o = trapzinterp(self.n2o/self.air, z, znew)*airnew
            self.n2 = trapzinterp(self.n2/self.air, z, znew)*airnew
            z = znew
            self.air = airnew

            self.go3.regrid(znew)
            self.gh2o.regrid(znew)

        self.z = z
        self.lat = lat
        self.O3 = O3
        self.H2O = H2O
        self.NO2 = NO2
        self.tauR = tauR
        self.ssa = ssa

        self.aer = aer
        if self.aer is not None:
            self.aer.init(z, self.T, self.gh2o.dens*self.gh2o.scalingfact)

        self.cloud = cloud
        if self.cloud is not None:
            self.cloud.init(z)

        self.verbose = verbose
        self.overwrite = overwrite

    def write(self, w, dir_profile, dir_phases, dir_list_phases):
        '''
        Write profiles and phase functions at bands w (list)

        returns a tuple (profiles, phases) where
        - profiles is the filename containing the concatenated profiles
        - phase is a file containing the list pf phase functions
        '''
        # convert to list if wl is a scalar
        if isinstance(w, (float, int, REPTRAN_IBAND, KDIS_IBAND)):
            w = [w]

        use_reptran = isinstance(w[0], REPTRAN_IBAND)
        use_kdis    = isinstance(w[0], KDIS_IBAND)
        if (use_reptran or use_kdis):
            wl = map(lambda x:x.w, w)
        else:
            wl = w

        if use_reptran : profiles, phases = self.calc_bands(w)
        if use_kdis    : profiles, phases = self.calc_bands(w)
        else : profiles, phases = self.calc_bands(wl)

        header = "# I ALT   hmol(I) haer(I)  H(I)  "
        header += "XDEL(I)  YDEL(I)  XSSA(I)  percent_abs  IPHA  LAM={} nm\n"

        # write the profiles
        file_profiles = tempfile.mktemp(dir=dir_profile, prefix='profil_aer_')
        fp = open(file_profiles, 'w')
        for i, pro in enumerate(profiles):
            fp.write(header.format(wl[i]))
            for m in xrange(len(pro)):
                line = "%d\t%7.2f\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%d\n" % tuple(pro[m])
                fp.write(line)
        fp.close()

        # write the phase functions
        # and list of phase functions
        file_list_phases = tempfile.mktemp(dir=dir_list_phases, prefix='list_phases_')
        fp = open(file_list_phases, 'w')
        for phase in phases:
            file_phase = tempfile.mktemp(dir=dir_phases, prefix='pf_')
            phase.write(file_phase)
            fp.write(file_phase+'\n')
        fp.close()

        return file_profiles, file_list_phases

    def calc_phase(self, w):
        '''
        calculate phase functions for bands self.pfwav and coarse profile self.pfgrid
        returns a list pf phase functions, and an array of indices (z, w) that give the
        index of the phase function associated with each layer and each band of the find grids

        -> create a new profile at a grid pfgrid (coarse grid) and for bands pfwav
        '''
        # convert to list if wl is a scalar
        if isinstance(w, (float, int, REPTRAN_IBAND, KDIS_IBAND)):
            w = [w]

        for i in xrange(len(self.cache_phase_keys)):
            if np.alltrue(self.cache_phase_keys[i] == w):
                return self.cache_phase_values[i]

        use_reptran = isinstance(w[0], REPTRAN_IBAND)
        use_kdis = isinstance(w[0], KDIS_IBAND)
        if (use_reptran or use_kdis):
            wl = map(lambda x:x.w, w)
        else:
            wl = w
        # indices of the phase matrices for each layer and each band of the main profile
        indices = np.zeros((len(self.z), len(wl)), dtype='i')

        phases = []  # list of the phase matrices

        if (self.aer is None) and (self.cloud is None):
            return phases, indices

        if self.pfwav is None:
            pfwav = wl
        else:
            pfwav = self.pfwav

        pro = Profile(self.atm_filename, aer=self.aer, cloud=self.cloud, grid=self.pfgrid,
                lat=self.lat, O3=self.O3, H2O=self.H2O, NO2=self.NO2, verbose=self.verbose, overwrite=self.overwrite)

        # calculate the indices of wl in pfwav
        ind_wl = []
        for wav in wl:
            ind_wl.append(np.abs(wav - np.array(pfwav)).argmin())

        # calculate the indices of z in pfgrid
        # we select the index of the pfgrid layer immediately lower than the value of z
        # (without the 1st element)
        ind_z = (np.searchsorted(-np.array(self.pfgrid[1:]), -np.array(self.z)))
        ind_z[ind_z<0] = 0
        ind_z[ind_z > len(self.pfgrid)-2] = len(self.pfgrid)-2

        # fill the 2D array indices
        max = 0
        band_index_current = 0
        for i in xrange(len(wl)):
            if ind_wl[i] > band_index_current:
                band_index_current = ind_wl[i]
                max += ind_z.max()+1
            indices[:,i] = ind_z[:] + max


        for ww in pfwav:
            if self.verbose:
                print 'Computing phase functions at {} nm'.format(ww)
            pro.calc(ww)

            # aerosol profile and phase functions
            if self.aer is not None:
                pfaer = pro.aer.phase(ww)
                _, dtauaer, ssaaer = pro.aer.calc(ww)
            else:
                pfaer = 0.
                dtauaer, ssaaer = 0., 0.

            # cloud profile and phase function
            if self.cloud is not None:
                pfcld = pro.cloud.phase(ww)
                _, dtaucld, ssacld = pro.cloud.calc(ww)
            else:
                pfcld = 0.
                dtaucld, ssacld = 0., 0.

            for i in xrange(len(self.pfgrid)-1):
                # the total phase function is the average of pfaer and pfcld
                # with a prorata of dtau*ssa
                pf = 0.
                norm = 0.

                if self.aer is not None:
                    pf += dtauaer[i+1]*ssaaer[i+1]*pfaer[i]
                    norm += dtauaer[i+1]*ssaaer[i+1]
                if self.cloud is not None:
                    pf += dtaucld[i+1]*ssacld[i+1]*pfcld[i]
                    norm += dtaucld[i+1]*ssacld[i+1]

                phases.append(pf/norm)

        if self.verbose:
            print 'Computed {} phase function'.format(len(phases))
            print '  {} layers: {} within {}'.format(len(self.pfgrid)-1, self.pfgrid, self.z)
            print '  {} bands: {} within {}'.format(len(pfwav), pfwav, wl)
            print '  indices:'
            print indices

        self.cache_phase_keys.append(w)
        self.cache_phase_values.append((phases, indices))

        return phases, indices

    def calc_bands(self, w):
        '''
        Profile calculation at bands w
        w is a list of bands (list of floats)
        returns a list of profiles and a list of corresponding phase functions

        a profile is a structured array with records:
        (I,ALT,hmol,haer,H,XDEL,YDEL,XSSA,percent_abs)
        '''
        # convert to list if wl is a scalar
        if isinstance(w, (float, int, REPTRAN_IBAND, KDIS_IBAND)):
            w = [w]
        use_reptran = isinstance(w[0], REPTRAN_IBAND)
        use_kdis = isinstance(w[0], KDIS_IBAND)
        if (use_reptran or use_kdis):
            wl = map(lambda x:x.w, w)
        else:
            wl = w

        # calculate the phase functions
        phases, indices = self.calc_phase(wl)

        # calculate the profiles
        profiles = []
        for i in xrange(len(wl)):
            if (use_reptran or use_kdis) : 
                pro = self.calc(w[i])
            else : pro  = self.calc(wl[i])

            # setup the phase functions indices in profile
            pro['IPHA'] = indices[:,i]

            profiles.append(pro)

        return profiles, phases

    def calc(self, w):
        '''
        Profile calculation at a monochromatic band w

        returns the profile at this band, a structured array with records:
        (I,ALT,hmol,haer,H,XDEL,YDEL,XSSA,percent_abs, IPHA)

        '''
        #for i in xrange(len(self.cache_prof_keys)):
         #   if np.alltrue(self.cache_prof_keys[i] == w):
         #       return self.cache_prof_values[i]
        if w in self.cache_prof_keys:
            i = self.cache_prof_keys.index(w)
            #return self.cache_prof_values[i]
            
        use_reptran = isinstance(w, REPTRAN_IBAND)
        use_kdis    = isinstance(w, KDIS_IBAND)
        if (use_reptran or use_kdis):
            wl = w.w
        else:
            wl = w

        if (self.aer is not None) and (list(self.z) != list(self.aer.z)):
            # re-initialize aer if necessary
            self.aer.init(self.z, self.T, self.gh2o.dens*self.gh2o.scalingfact)

        if (self.cloud is not None) and (list(self.z) != list(self.cloud.z)):
            # re-initialize cloud if necessary
            self.cloud.init(self.z)

        z = self.z
        M = len(z)  # Final number of layer

        if self.aer is not None:
            if w in self.cache_prof_keys:
                dataaer = self.cache_prof_values[i]['haer']
                ssaaer  = self.cache_prof_values[i]['XSSA']
            else:
                dataaer, _, ssaaer = self.aer.calc(wl)
        else :
            dataaer = np.zeros(M, np.float)
            ssaaer = np.zeros(M, np.float)

        if self.cloud is not None:
            dataclo, _, ssaclo = self.cloud.calc(wl)
        else:
            dataclo = np.zeros(M, np.float)
            ssaclo = np.zeros(M, np.float)

        if (use_reptran or use_kdis):
            Nmol = 8
            densmol = np.zeros((M, Nmol), np.float)
            densmol[:,0] = self.gh2o.dens*self.gh2o.scalingfact
            densmol[:,1] = self.co2
            densmol[:,2] = self.go3.dens*self.go3.scalingfact
            densmol[:,3] = self.n2o
            densmol[:,4] = self.co
            densmol[:,5] = self.ch4
            densmol[:,6] = self.o2
            densmol[:,7] = self.n2
            xh2o = self.gh2o.dens*self.gh2o.scalingfact/self.air   # h2o vmr
            if use_reptran : datamol = w.calc_profile(self.T, self.P, xh2o, densmol)
            else :           datamol = w.calc_profile(self.T, self.P, densmol)*1e5
        else:
            datamol = np.zeros(M, np.float)


        # profiles of o3 and Rayleigh
        datao3  = np.zeros(M, np.float)
        datano2  = np.zeros(M, np.float)
        dataray = np.zeros(M, np.float)
        
        scaleR = 1.
        if self.tauR != None :
            scaleR = self.tauR/rod(wl*1e-3, self.co2[-1]/self.air[-1]*1e6, self.lat, z[-1]*1e3, self.P[-1])
            
        for m in xrange(M):

            #### Chappuis bands
            # SIGMA = (C0 + C1*(T-T0) + C2*(T-T0)^2) * 1.E-20 cm^2
            T0 = 273.15  # in K

            datao3[m] =  np.interp(wl,self.crs_chappuis[:,0],self.crs_chappuis[:,1]) \
                       + np.interp(wl,self.crs_chappuis[:,0],self.crs_chappuis[:,2])*(self.T[m]-T0) \
                       + np.interp(wl,self.crs_chappuis[:,0],self.crs_chappuis[:,3])*(self.T[m]-T0)**2
            # calcul de Chapuis avec LUT en 10^(-20) cm2, passage en km-1 
            datao3[m] *= self.go3.dens[m] * self.go3.scalingfact * 1e-15

            if self.NO2:
                datano2[m] =  np.interp(wl,self.crs_no2[:,0],self.crs_no2[:,1]) \
                            + np.interp(wl,self.crs_no2[:,0],self.crs_no2[:,2])*(self.T[m]-T0) \
                            + np.interp(wl,self.crs_no2[:,0],self.crs_no2[:,3])*(self.T[m]-T0)**2
                datano2[m] *= self.no2[m] * 1e-15

            dataray[m] = rod(wl*1e-3, self.co2[m]/self.air[m]*1e6, self.lat, z[m]*1e3, self.P[m]) * scaleR

        datano2[datano2 < 0] = 0.


        # create the profile
        profile = np.zeros(M, dtype=[('I', int),
                                     ('ALT', float),
                                     ('hmol', float),
                                     ('haer', float), 
                                     ('H', float), 
                                     ('XDEL', float), 
                                     ('YDEL', float), 
                                     ('XSSA', float), 
                                     ('percent_abs', float), 
                                     ('IPHA', int), 
                                     ])

        for m in xrange(M):

            if m==0 : 
                dz=0.
                hg=0.
                taur_prec=0. #rayleigh
                taua_prec=0. #aerosols
                tauc_prec=0. #clouds
                profile[m] = (m, z[m], 0., 0., 0. , 0., 1., 1., 0., 0)
            else : 
                dz = z[m-1]-z[m]
                taur = dataray[m] - taur_prec
                taur_prec = dataray[m]
                taua = dataaer[m] - taua_prec
                taua_prec = dataaer[m]
                tauc = dataclo[m] - tauc_prec
                tauc_prec = dataclo[m]
                taug = (datao3[m] + datano2[m] + datamol[m])*dz
                # tau = taur+taua+taug
                tau = taur+taua+tauc+taug
                abs = taug/tau
                # xdel = taua/(tau*(1-abs))
                xdel = (taua+tauc)/(tau*(1-abs))
                ydel = taur/(tau*(1-abs))
                hg += taug
                # htot = dataray[m]+dataaer[m]+hg
                htot = dataray[m]+dataaer[m]+dataclo[m]+hg
                # xssa=ssaaer[m]
                if ((taua+tauc) <=0):
                    xssa=1.
                elif (self.ssa!= None) :
                    xssa=self.ssa
                else:
                    xssa=(ssaaer[m]*taua+ssaclo[m]*tauc)/(taua+tauc)
                profile[m] = (m, z[m], dataray[m], dataaer[m]+dataclo[m], htot , xdel, ydel, xssa, abs, 0)

        self.cache_prof_keys.append(w)
        self.cache_prof_values.append(profile)

        return profile

    def __str__(self):
        '''
        returns a string class descriptor
        (for building a default file name)
        '''
        S = 'ATM={atmfile}-O3={O3}-H2O={H2O}'.format(atmfile=self.basename, O3=self.O3, H2O=self.H2O)

        if self.aer is not None:
            S += '-{}'.format(str(self.aer))
            
        if self.cloud is not None:
            S += '-{}'.format(str(self.cloud))

        return S




def main():
    ####################################################################################################################################     
    parser = OptionParser(usage='%prog [options] file_in_atm\nfile_in_atm: see {}\nType %prog -h for help\n'.format(dir_libradtran_atmmod))
    parser.add_option('-n','--noabs',
                dest='noabs',
                action='store_true',
                default=False,
                help='no gaseous absorption'
                )
    parser.add_option('-o','--o3',
                dest='o3',
                type='float',
                default=None,
                help='Ozone vertical column in Dobson units ,default: atmospheric profile value'
                )
    parser.add_option('-w', '--wavel',
                dest='w',
                type='float',
                default=550.,
                help='wavelength (nm), default 550 nm' 
                )
    parser.add_option('-W', '--Wavel_ref',
                dest='wref',
                type='float',
                default=550.,
                help='wavelength (nm) reference for aerosol optical properties, default %default nm' 
                )
    parser.add_option('-l', '--lat',
                dest='lat',
                type='float',
                default=45.,
                help='latitude (deg), default %default' 
                )
    parser.add_option('-g', '--grid',
                dest='grid',
                type='string',
                help='vertical grid format : start[step]Z1[step1].....[stepN]stop (km) with start>stop' 
                )
    parser.add_option('-A', '--AOT',
                dest='aot',
                type='float',
                default=0.,
                help='Aerosol Optical Thickness at the current wavelength , default 0. (no aerosols)' 
                )
    parser.add_option('-O', '--OPAC',
                dest='opac',
                type='string',
                help='name of the aerosol model from OPAC\n' \
                      +'    antarctic,continental_average,continental_clean,continental_polluted\n'  \
                      +'    desert, desert_nonspherical,maritime_clean,maritime_polluted,maritime_tropical,urban\n' \
                      +'    and eventually a user defined aerosol model in the tools/profile/standard_aerosol_files/ directory\n'
                )
    parser.add_option('-N', '--NSCAER',
                dest='NSCAER',
                type='int',
                default=1800,
                help='number of scattering angles of the aerosol phase matrix, if options P is chosen, default:1800\n'
                )
    parser.add_option('-p','--phase',
                dest='phase',
                action='store_true',
                default=False,
                help='Phase matrix computation (could slow down the process depending on NSCAER'
                )
    parser.add_option('-R', '--REPTRAN',
                dest='rep',
                type='string',
                help='REPTRAN molecular absorption file: see {} for channel list'.format(join(dir_libradtran_reptran, 'channel_list.txt'))
                )
    parser.add_option('-C', '--CHANNEL',
                dest='channel',
                type='string',
                help='Sensor channel name (use with REPTRAN)' 
                )

    (options, args) = parser.parse_args()
    if len(args) != 1 :
        parser.print_usage()
        exit(1)

    atm_filename = args[0]
    if options.aot > 0.:
        aer = AeroOPAC(options.opac, options.aot, options.wref)
    else:
        aer = None

    Profile(atm_filename, grid=options.grid, aerosol=aer).calc(options.w)


def example1():
    '''
    basic example without aerosols
    no gaseous absorption
    '''
    Profile('afglt.dat', O3=0.).write(500., dir='tmp/')

def example2():
    '''
    using a custom grid , O3 absorption using default values, and aerosols
    '''
    Profile('afglt.dat',
            grid='100[75]25[5]10[1]0',
            aer=AeroOPAC('maritime_polluted', 0.4, 550.)
            ).write(500., dir='tmp/')

def example3():
    '''
    using reptran with aerosols
    also write phase functions
    '''
    aer = AeroOPAC('desert', 0.4, 550.)
    pro = Profile('afglt.dat', aer=aer)
    rep = REPTRAN('reptran_solar_sentinel.cdf')
    for band in rep.band_names:
        print band
    band = rep.band('sentinel3_slstr_b4')
    for iband in band.ibands():
        print '* Band', iband.w
        pro.write(iband, dir='tmp/')
        avg_wvl = np.mean(iband.band.awvl)  # average wavelength of the iband
        phase = aer.phase(avg_wvl, dir='tmp/')
        print 'phase function', phase
        
def example4():
    '''
    using reptran with aerosols and clouds for PAR
    also write phase functions
    '''
    wref=550.
    aer = AeroOPAC('maritime_clean', 0.1, wref)

    # Cumulus Maritime (OPAC)
    # 1 layer : water cloud mie, relative concentration = 1, reff=12.68 between 2 and 3 km
    # total OT of 50 at reference wavelength
    cloud = CloudOPAC('CUMA',[('wc.sol.mie',1.,12.68,2.,3.)], 5., wref)

    pro = Profile('afglss.dat', aer=aer, cloud=cloud, grid='100[25]25[5]5[1]0')
    rep = REPTRAN('reptran_solar_coarse.cdf')
    sampling = 100
    L = rep.band_names
    for i in range(0,len(L),sampling):
        band = rep.band(L[i])
        if band.awvl[-1] < 380. : continue
        if band.awvl[0] > 700. : break
        print '--- Band', L[i]
        for iband in band.ibands():
            wi = iband.band.awvl[iband.index] # wvl of internal band
            print '* Band', iband.index, iband.iband, wi
            pro.write(iband,'./','./','./')
            # aer.calc(wi)
            # aer.phase(wi, dir='tmp/',NTHETA=721)
#            print 'phase function', phase


if __name__ == '__main__':
    example4()
