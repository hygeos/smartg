#!/usr/bin/env python
# encoding: utf-8



import os
from os.path import join, dirname, exists, realpath, basename
from string  import count,split
import numpy as np
from optparse import OptionParser
from scipy.interpolate import interp1d
from scipy.constants import codata
import netCDF4
from scipy.ndimage import map_coordinates
from scipy.integrate import simps
import tempfile
from glob import glob

dir_libradtran = '/home/applis/libRadtran-2.0-beta/'
dir_libradtran_reptran =  join(dir_libradtran, 'data/correlated_k/reptran/')
dir_libradtran_opac =  join(dir_libradtran, 'data/aerosol/OPAC/')
dir_libradtran_atmmod = join(dir_libradtran, 'data/atmmod/')
dir_libradtran_crs = join(dir_libradtran, 'data/crs/')


class AeroOPAC(object):
    '''
    Initialize the Aerosol OPAC model

    Args:
        filename: name of the aerosol file. If no directory is specified,
                  assume directory <libradtran>/data/aerosol/OPAC/standard_aerosol_files
        tau: optical thickness at wavelength wref
        wref: reference wavelength (nm) for aot
        layer_phase: the layer index at which the phase function is chosen
        overwrite: recalculate and overwrite phase functions
    '''
    def __init__(self, filename, tau, wref, layer_phase=None, overwrite=False):

        self.__tau = tau
        self.__wref = wref
        self.__layer_phase = layer_phase
        self.overwrite = overwrite

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
        self._readStandardAerosolFile()

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
            self.scamatlist.append(ScaMat(species))   

    def init(self, z, T, h2o):
        '''
        Initialize the model using height profile, temperature and h2o conc.
        '''
        self.regrid(z)
        self.setTauref(T, h2o, self.__tau, self.__wref)
        self.__T = T
        self.__h2o = h2o

    def regrid(self,znew):
        '''
        reechantillonage vertical des concentrations massiques
        '''
        N=len(self.aspecies)
        M=len(znew)
        tmp=np.zeros((M,N),np.float32)
        for k in range(N):
            f=interp1d(self.zopac,self.densities[:,k],bounds_error=False,fill_value=0.)
            tmp[:,k]=f(znew)
        self.densities=tmp
        self.zopac=znew

    def calcTau(self,T,h2o,w): # calcul des propritees optiques du melange en fonction de l'alitude et aussi integrees sur la verticale
        rh=h2o/vapor_pressure(T)*100 # calcul du profil vertical de RH
        M=len(self.zopac)
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
                        ext=ext0*frho(rh[m])/frho(50.)*self.densities[m,k] # calcul du coefficient de diffusion et ajustement pour RH du niveau
                        dz = self.zopac[m-1]-self.zopac[m]
                        dtau = dz * ext * self.scalingfact # calcul de l'epaisseur optique du niveau, eventuellement mise a l'echelle
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
                        ext=ext0*self.densities[m,k]
                        dz = self.zopac[m-1]-self.zopac[m]                       
                        dtau = dz * ext * self.scalingfact
                    self.dtau_tot[m]+=dtau

            k=k+1

        self.tau_tot=np.sum(self.dtau_tot)

    def calc(self,w):
        '''
        calcul des propritees optiques du melange en fonction de l'alitude et
        aussi integrees sur la verticale à la longueur d'onde w (nm)
        retourne le profil d'épaisseur optique intégrée et le profil de ssa
        '''
        h2o = self.__h2o
        T = self.__T

        rh=h2o/vapor_pressure(T)*100 # calcul du profil vertical de RH
        M=len(self.zopac)
        MMAX=5000 # Nb de polynome de Legendre au total
        self.dtau_tot=np.zeros(M,np.float32)
        self.ssa_tot=np.zeros(M,np.float32)
        self.pmom_tot=np.zeros((M,4,MMAX),np.float64)
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
                        ext=ext0*frho(rh[m])/frho(50.)*self.densities[m,k] # calcul du coefficient de diffusion et ajustement pour RH du niveau
                        dz = self.zopac[m-1]-self.zopac[m]                       
                        dtau = dz * ext * self.scalingfact # calcul de l'epaisseur optique du niveau, eventuellement mise a l'echelle
                        dssa = dtau*ssa # ssa pondere par l'epsaissuer optique
                        norm[m]+=dssa                   
                        for n in range(4): # pour chaque element de la matrice de Stokes independant (4 pour Mie) 
                            nmax=scamat.nmom[int(iw),int(ir),n]
                            dp[n]= scamat.pmom[int(iw),int(ir),n,:nmax]*dssa # plus proche voisin pour pmom pondere par ssa et tau de la composante
                    self.dtau_tot[m]+=dtau #somme sur les composantes
                    self.ssa_tot[m]+=dssa #moyenne pondere par l'epassieur optique pour ssa
                    for n in range(4):
                        nmax=scamat.nmom[int(iw),int(ir),n]
                        self.pmom_tot[m,n,:nmax]+=dp[n]

            else:  # idem mais rien de depend de RH pour cette composante
                tabext=np.squeeze(scamat.ext)
                fext=interp1d(scamat.wlgrid,tabext,bounds_error=False,fill_value=0.)                       
                ext0=fext(w*1e-3)
                tabssa=np.squeeze(scamat.ssa)
                fssa=interp1d(scamat.wlgrid,tabssa,bounds_error=False,fill_value=0.)                       
                ssa=fssa(w*1e-3)
                nmom=np.squeeze(scamat.nmom)
                pmom=np.squeeze(scamat.pmom)

                for m in xrange(M):
                    if m==0:
                        dz=0.
                        dtau=0.
                        dssa=1.
                        dp=[0.,0.,0.,0.]
                        nmax=[0,0,0,0]
                        ext=0.
                    else:                                                       
                        ext=ext0*self.densities[m,k]
                        dz = self.zopac[m-1]-self.zopac[m]                       
                        dtau = dz * ext * self.scalingfact
                        dssa = dtau*ssa
                        norm[m]+=dssa
                        for n in range(4): # pour chaque element de la matrice de Stokes independant (4 pour Mie) 
                            nmax=nmom[int(iw),n]
                            dp[n]= pmom[int(iw),n,:nmax]*dssa # plus proche voisin pour pmom pondere par ssa et tau de la composante

                    self.dtau_tot[m]+=dtau
                    self.ssa_tot[m]+=dssa
                    for n in range(4):
                        nmax=nmom[int(iw),n]
                        self.pmom_tot[m,n,:nmax]+=dp[n]

            k=k+1 # each component

        for m in xrange(M):
            if m==0:
                self.ssa_tot[m]=1.
                for n in range(4):
                    self.pmom_tot[m,n,:]=0.
            else:
                if (self.dtau_tot[m]>1e-8 and norm[m] > 1e-8): 
                    self.ssa_tot[m]/=self.dtau_tot[m]
                    for n in range(4):  
                        self.pmom_tot[m,n,:]/=norm[m]
                else:
                    self.ssa_tot[m]=1.
                    for n in range(4):
                        self.pmom_tot[m,n,:]=0.


        self.tau_tot=np.sum(self.dtau_tot)
        self.MMAX=MMAX

        dataaer  = np.zeros(M, np.float)
        for m in xrange(M):
            dataaer[m] = np.sum(self.dtau_tot[:m+1])

        return (dataaer, self.ssa_tot)


    def setTauref(self,T,h2o,tauref,wref): # On fixe l'AOT a une valeur pour une longueur d'onde de reference
        self.calcTau(T,h2o,wref) # calcul de l'AOT a la longueur d'onde de reference
        self.scalingfact=tauref/self.tau_tot # calcul du facteur d'echelle        

    def calcPha(self, w, pattern, dir, NTHETA=7201):
        '''
        calculate and write phase matrices for each layer
        Arguments:
            w: wavelength in nm
            NTHETA: number of angles
            dir: location of the output file
            pattern: output file pattern, formatted by the aerosol specie,
                     the wavelength and the layer
        Returns: the list of files generated
        '''
        self.NTHETA=NTHETA
        M=len(self.zopac)
        Leg=Legendres(self.MMAX,NTHETA)
        ret = []
        list_skipped = []   # list of files skipped because existing
        list_overwritten = []   # list of files overwritten

        for m in range(M):

            output = join(dir, pattern%(basename(self.filename[:-4]),w,m))
            ret.append(output)
            if exists(output):
                if self.overwrite:
                    os.remove(output)
                    list_overwritten.append(output)
                else:
                    list_skipped.append(output)
                    continue
            theta,pha=Mom2Pha(self.pmom_tot[m,:,:],Leg)
            if not exists(dirname(output)):
                # create output directory if necessary
                os.makedirs(dirname(output))

            f=open(output,'w')
            for j in range(NTHETA):
                f.write("%18.8E"%theta[j] + "  %20.11E  %20.11E  %20.11E  %20.11E\n"%tuple(pha[:,j]))
            f.close()

        if len(list_skipped) > 0:
            print 'INFO: skipping {} and {} other files'.format(list_skipped[0], len(list_skipped)-1)
        if len(list_overwritten) > 0:
            print 'INFO: overwritten {} and {} other files'.format(list_overwritten[0], len(list_overwritten)-1)

        return ret

    def phase(self, wl, dir, pattern='pf_%s_%inm_layer-%i.txt',NTHETA=7201):
        '''
        creates the phase function corresponding to layer_phase
        and returns the corresponding file name

        wl is the wavelength in nm
        dir: directory for storing the phase function files
        pattern: output file pattern, formatted by the aerosol specie,
                 the wavelength and the layer
                 default: 'pf_%s_%inm_layer-%i.txt'
        '''
        if self.__layer_phase is None:
            return None

        ret = self.calcPha(wl, pattern, dir,NTHETA=NTHETA)

        return ret[self.__layer_phase]

    def __str__(self):
        return 'AER={base}-AOT={aot}'.format(base=self.basename, aot=self.__tau)


class Gas(object):
    def __init__(self,z,dens):
        self.z=z
        self.dens=dens
        self.scalingfact=1.
        self.initcol= self.calcColumn()

    def calcColumn(self):
        return simps(self.dens,-self.z) * 1e5 

    def setColumn(self,DU=None, Dens=None):
        if DU != None : self.scalingfact = 2.69e16 * DU / self.calcColumn() 
        if Dens !=None: self.scalingfact = Dens / self.calcColumn()

    def getDU(self):
#        return self.initcol * self.scalingfact / 2.69e16
        return self.calcColumn() / 2.69e16 * self.scalingfact

    def regrid(self,znew):
        f=interp1d(self.z,self.dens,kind='linear')
        self.dens=f(znew)
        self.z=znew


class ScaMat(object):
    def __init__(self,species):
        self.species=species
        self._readScaMatFile()

    def _readScaMatFile(self):
        if not 'spheroids' in self.species :
            fname=join(dir_libradtran_opac, 'optprop', self.species+'.mie.cdf')
        else:
            fname=join(dir_libradtran_opac, 'optprop', self.species+'.tmatrix.cdf')
        nc=netCDF4.Dataset(fname)
        self.wlgrid=nc.variables["wavelen"][:]
        self.rhgrid=nc.variables["hum"][:]
        self.nrh=len(self.rhgrid)
        self.thgrid=nc.variables["theta"][:]
        self.phase=nc.variables["phase"][:]
        self.pmom=nc.variables["pmom"][:]
        self.nth=nc.variables["ntheta"][:]
        self.nmom=nc.variables["nmom"][:]
        self.ext=nc.variables["ext"][:]
        self.ssa=nc.variables["ssa"][:]
        self.rho=nc.variables["rho"][:]

class Legendres(object):
    def __init__(self,nterm,ntheta):
        mu=np.linspace(0.,np.pi,ntheta,endpoint=True,dtype=np.float64)
        mu=np.cos(mu)
        un64=np.ones_like(mu)
        zero64=np.ones_like(mu)
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
    sumZ=np.zeros_like(Leg.mu)
    pha=np.zeros((4,Leg.ntheta),np.float64)
    for k in range(Leg.nterm):
        sumP=sumP+Mom[0,k]*Leg.p1[k,:]
        sumQ=sumQ+Mom[1,k]*Leg.p2[k,:]
        sumU=sumU+Mom[2,k]*Leg.p1[k,:]
    pha[0,:]=sumP+sumQ
    pha[1,:]=sumP-sumQ
    pha[2,:]=sumU
    pha[3,:]=sumZ
    return np.arccos(Leg.mu)/np.pi*180.,pha

class REPTRAN_IBAND(object):
    '''
    REPTRAN internal band

    Arguments:
        band: REPTRAN_BAND object
        index: band index
        iband: internal band index
    '''
    def __init__(self, band, index, iband):

        self.band = band     # parent REPTRAN_BAND
        self.index = index   # internal band index
        self.iband = iband   # internal band number
        self.w = band.awvl[index]  # band wavelength
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
            densmol[:,0]=h2o
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
                crs_mol = readCRS(crs_filename, self.iband)

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

    def iband(self, index):
        '''
        returns internal band by its number (starting at zero)
        '''
        return REPTRAN_IBAND(self, index, self._iband[index])

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

    def band(self, band_):
        '''
        returns a REPTRAN_BAND
        band can be defined either by an integer, or a string
        '''
        if isinstance(band_, str):
            return self.band(self.band_names.index(band_))
        else:
            return REPTRAN_BAND(self, band_)

    def bands(self):
        '''
        iterates over all bands
        '''
        for i in xrange(self.nbands()):
            yield self.band(i)


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
        - grid: custom grid. Can be provided as an array of altitudes or a gridSpec (string)
          default value: None (use default grid)
        - lat: latitude (for Rayleigh optical depth calculation, default=45.)
        - O3: total ozone column (Dobson units), or None to use atmospheric
          profile value (default)
        - NO2: activate ON2 absorption (default True)
    '''
    def __init__(self, atm_filename, aer=None, grid=None,
                lat=45., O3=None, NO2=True, verbose=False, overwrite=False):

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
        self.h2o = data[:,6] # H2O density en cm-3
        self.co2 = data[:,7] # CO2 density en cm-3
        self.no2 = data[:,8] # NO2 density en cm-3

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

        #-------------------------------------------
        # optionnal regrid
        if grid != None:
            if isinstance(grid, str):
                znew = change_altitude_grid(z, grid)
            else:
                znew = grid

            self.P = interp1d(z, self.P)(znew)
            self.T = interp1d(z, self.T)(znew)
            airnew = interp1d(z, self.air)(znew)
            o3 = interp1d(z, o3)(znew)
            self.o2 = interp1d(z, self.o2)(znew)
            self.h2o = interp1d(z, self.h2o)(znew)
            self.co2 = interp1d(z, self.co2)(znew)
            self.no2 = interp1d(z, self.no2)(znew)
            self.ch4 = interp1d(z, self.ch4/self.air)(znew)*airnew
            self.co = interp1d(z, self.co/self.air)(znew)*airnew
            self.n2o = interp1d(z, self.n2o/self.air)(znew)*airnew
            self.n2 = interp1d(z, self.n2/self.air)(znew)*airnew
            z = znew
            self.air = airnew

            self.go3.regrid(znew)

        self.z = z
        self.lat = lat
        self.O3 = O3
        self.NO2 = NO2

        self.aer = aer
        if self.aer is not None:
            self.aer.init(z, self.T, self.h2o)

        self.verbose = verbose
        self.overwrite = overwrite

    def write(self, w, output_file=None, dir=None):
        '''
        write a profile at a monochromatic wavelength w (nm)
        Produce file output_file.

        w can be either:
            - a wavelength in nm (float)
            - a REPTRAN_IBAND object
        if output_file is None, use a default file name in
        directory dir.

        returns: profile filename
        '''
        #
        # Initialization
        #
        use_reptran = isinstance(w, REPTRAN_IBAND)
        z = self.z
        M = len(z)  # Final number of layer
        if use_reptran:
            wl = w.w
        else:
            wl = w

        if output_file is None:
            assert dir is not None
            # generate a unique file name
            output_file = tempfile.mktemp(dir=dir,
                    prefix='profile_', suffix='.tmp')

        if exists(output_file):
            if self.overwrite:
                os.remove(output_file)
                if self.verbose: print 'Removed {}'.format(output_file)
            else:
                if self.verbose: print 'File {} exists!'.format(output_file)
                return output_file

        if not exists(dirname(output_file)):
            os.makedirs(dirname(output_file))

        # calculate the profile
        pro = self.calc(w)

        # write the header
        fp = open(output_file, 'w')
        outstr = "# I   ALT               hmol(I)         haer(I)         H(I)            "
        outstr += "XDEL(I)         YDEL(I)     XSSA(I)     percent_abs       LAM=  %7.2f nm" % (wl)
        if use_reptran:
            outstr += ', WEIGHT= %7.5f, E0=%9.3f, Rint=%8.3f' % (w.weight, w.extra, w.band.Rint)
        # print outstr
        fp.write(outstr)

        for m in xrange(M):
            outstr = "%d\t%7.2f\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t" % tuple(pro[m])

            # print outstr
            fp.write('\n' + outstr)
        
        fp.write('\n')
        if self.verbose: print 'write', output_file

        return output_file


    def calc(self, w):
        '''
        Profile calculation at a monochromatic wavelength w (nm)
        Returns a numpy structured array

        w can be either:
            - a wavelength in nm (float)
            - a REPTRAN_IBAND object

        returns: a structured array with records 
        (I,ALT,hmol,haer,H,XDEL,YDEL,XSSA,percent_abs)
        containing the profile, which can be accessed like
                profile['ALT'][0]  # altitude of top layer
                profile['hmol']    # the whole profile of Rayleigh optical thickness
        '''

        #
        # Initialization
        #
        use_reptran = isinstance(w, REPTRAN_IBAND)

        z = self.z
        M = len(z)  # Final number of layer
        if use_reptran:
            wl = w.w
        else:
            wl = w
        if self.aer is not None:
            dataaer, ssaaer = self.aer.calc(wl)
        else:
            dataaer = np.zeros(M, np.float)
            ssaaer = np.zeros(M, np.float)

        if use_reptran:
            Nmol = 8
            densmol = np.zeros((M, Nmol), np.float)
            densmol[:,0] = self.h2o
            densmol[:,1] = self.co2
            densmol[:,2] = self.go3.dens*self.go3.scalingfact
            densmol[:,3] = self.n2o
            densmol[:,4] = self.co
            densmol[:,5] = self.ch4
            densmol[:,6] = self.o2
            densmol[:,7] = self.n2
            xh2o = self.h2o/self.air   # h2o vmr
            datamol = w.calc_profile(self.T, self.P, xh2o, densmol)
        else:
            datamol = np.zeros(M, np.float)


        # profiles of o3 and Rayleigh
        datao3  = np.zeros(M, np.float)
        datano2  = np.zeros(M, np.float)
        dataray = np.zeros(M, np.float)
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

            dataray[m] = rod(wl*1e-3, self.co2[m]/self.air[m]*1e6, self.lat, z[m]*1e3, self.P[m])

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
                                     ])

        for m in xrange(M):

            if m==0 : 
                dz=0.
                hg=0.
                taur_prec=0.
                taua_prec=0.
                profile[m] = (m, z[m], 0., 0., 0. , 0., 1., 1., 0.)
            else : 
                dz = z[m-1]-z[m]
                taur = dataray[m] - taur_prec
                taur_prec = dataray[m]
                taua = dataaer[m] - taua_prec
                taua_prec = dataaer[m]
                taug = (datao3[m] + datano2[m] + datamol[m])*dz
                tau = taur+taua+taug
                abs = taug/tau
                xdel = taua/(tau*(1-abs))
                ydel = taur/(tau*(1-abs))
                hg += taug
                htot = dataray[m]+dataaer[m]+hg
                xssa=ssaaer[m]
                profile[m] = (m, z[m], dataray[m], dataaer[m], htot , xdel, ydel, xssa, abs)

        return profile

    def __str__(self):
        '''
        returns a string class descriptor
        (for building a default file name)
        '''
        S = 'ATM={atmfile}-O3={O3}'.format(atmfile=self.basename, O3=self.O3)

        if self.aer is not None:
            S += '-{}'.format(str(self.aer))

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
    aer = AeroOPAC('desert', 0.4, 550., layer_phase=0)
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
    using reptran for PAR
    also write phase functions
    '''
    aer = AeroOPAC('maritime_clean', 0.1, 550., layer_phase=-1)
    pro = Profile('afglss.dat', aer=aer, grid='100[25]25[5]5[1]0',)
    rep = REPTRAN('reptran_solar_coarse.cdf')
    sampling = 1
    L = rep.band_names
    for i in range(0,len(L),sampling):
        band = rep.band(L[i])
        if band.awvl[-1] < 380. : continue
        if band.awvl[0] > 700. : break
        print '--- Band', L[i]
        for iband in band.ibands():
            wi = iband.band.awvl[iband.index] # wvl of internal band
            print '* Band', iband.index, iband.iband, wi
            pro.write(iband, output_file ='tmp/profil_PAR_%s-%dof%d.txt'%(L[i],iband.index+1,iband.band.nband))
            # aer.calc(wi)
            # aer.phase(wi, dir='tmp/',NTHETA=721)
#            print 'phase function', phase


if __name__ == '__main__':
    example4()
