#!/usr/bin/env python
# encoding: utf-8

# ipython notebook --no-browser --ip='*'

#import des modules en cuda
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from sys import path
from gpustruct import GPUStruct
import time
from time import sleep
import subprocess
import numpy as np
from pyhdf.SD import SD, SDC
from profile.profil import AeroOPAC, Profile, REPTRAN, REPTRAN_IBAND, CloudOPAC
from water.iop_spm import IOP_SPM
from water.iop_mm import IOP_MM
from os.path import dirname, realpath, join, exists, basename, isdir
from os import makedirs, remove
import textwrap
import tempfile
from progress import Progress
from luts import merge, read_lut_hdf, read_mlut_hdf,LUT,MLUT

#
# set up default directories
#
dir_install = dirname(dirname(realpath(__file__)))    # base smartg directory is one directory above here
dir_tmp = join(dir_install, 'tmp/')
dir_list_pf_aer = join(dir_tmp, 'list_pf_aer/')
dir_list_pf_oce = join(dir_tmp, 'list_pf_oce/')
dir_phase_water = join(dir_tmp, 'phase_water/')
dir_phase_aero = join(dir_tmp, 'phase_aerosols/')
dir_albedo = join(dir_tmp, 'albedo/')
dir_cmdfiles = join(dir_tmp, 'command_files/')
dir_profil_aer = join(dir_tmp, 'profile_aer/')
dir_profil_oce = join(dir_tmp, 'profile_oce/')
dir_output = join(dir_tmp, 'results')



class Smartg(object):
    '''
    Run a SMART-G job

    Arguments:
        - exe: smart-g executable
        - wl: wavelength in nm (float)
              or a list of wavelengths, or an array
              used for phase functions calculation (always)
              and profile calculation (if iband is None)   # FIXME
        - iband: a REPTRAN_BAND object describing the internal band
            default None (no reptran mode)
        - output: the name of the file to create. If None (default),
          automatically choose a filename in directory dir
        - dir: directory for automatic filename if output=None
        - overwrite: if True, remove output file first if it exists
        - skip_existing: skip existing file if overwrite is False
                         otherwise (False, default), raise an Exception
        - cmdfile: the name of the command file to write. If None (default),
          automatically choose a filename.
        - atm: Profile object
            default None (no atmosphere)
        - surf: Surface object
            default None (no surface)
        - water: Iop object, providing options relative to the ocean surface
            default None (no ocean)
        - env: environment effect parameters (dictionary)
            default None (no environment effect)
        The other acuments (NBPHOTONS, THVDEG, etc) are passed directly to the
        command file.

    Attributes:
        - output: the name of the result file
        - cmdfile: name of the command file
    '''
    def __init__(self, exe, wl, output=None, dir=dir_output,
           overwrite=False, skip_existing=False,
           cmdfile=None, iband=None,
           atm=None, surf=None, water=None, env=None,
           NBPHOTONS=1e9, DEPO=0.0279, THVDEG=0., SEED=-1,
           NBTHETA=45, NBPHI=45,
           NFAER=10000, NFOCE=10000, WRITE_PERIOD=-1,
           OUTPUT_LAYERS=0, XBLOCK=256, XGRID=256,
           NBLOOP=5000):
        #
        # initialization
        #

        if cmdfile is None:
            cmdfile = tempfile.mktemp(suffix='.txt',
                    prefix='smartg_command_',
                    dir=dir_cmdfiles)
        self.cmdfile = cmdfile

        assert isinstance(wl, (float, list, np.ndarray))
        assert (iband is None) or isinstance(iband, REPTRAN_IBAND)

        if isinstance(wl, (list, np.ndarray)):
            nlam = len(wl)
        else:
            nlam = 1

        if output is None:
            #
            # default file name
            #
            assert dir is not None
            list_filename = [exe]   # executable

            if iband is None:
                pass
            else:
                list_filename.append('WL={:.2f}'.format(iband.w))

            list_filename.append('THV={:.1f}'.format(THVDEG))

            if atm is not None: list_filename.append(str(atm))
            if surf is not None: list_filename.append(str(surf))
            if water is not None: list_filename.append(str(water))
            if env is not None: list_filename.append(str(env))

            filename = '_'.join(list_filename)
            filename = join(dir, filename + '.hdf')
            output = filename

        self.output = output

        #if exists(output):
        #    if overwrite:
        #        remove(output)
        #    else:
        #        if skip_existing:
        #            return
        #        else:
        #            raise Exception('File {} exists'.format(output))

        ensure_dir_exists(output)
        ensure_dir_exists(cmdfile)

        #
        # make dictionary of parameters
        #
        D = {
                'NBPHOTONS': str(int(NBPHOTONS)),
                'THVDEG': np.array([THVDEG],dtype=np.float32),
                'DEPO': np.array([DEPO], dtype=np.float32),
                'SEED': SEED,
                'NBTHETA': np.array([NBTHETA], dtype=np.int32),
                'NBPHI': np.array([NBPHI], dtype=np.int32),
                'NFAER': np.array([NFAER], dtype=np.uint32),
                'NFOCE': np.array([NFOCE], dtype=np.uint32),
                'WRITE_PERIOD': WRITE_PERIOD,
                'OUTPUT_LAYERS': np.array([OUTPUT_LAYERS], dtype=np.uint32),
                'XBLOCK': np.array([XBLOCK], dtype=np.int32),
                'YBLOCK': np.array([1], dtype=np.int32),
                'XGRID': np.array([XGRID], dtype=np.int32),
                'YGRID': np.array([1], dtype=np.int32),
                'NBLOOP': np.array([NBLOOP],dtype=np.uint32),
                'NLAM': np.array([nlam], dtype=np.int32),
                }



        # we use a separate disctionary to store the default parameters
        # which should not override the specified ones
        Ddef = {}

        # determine SIM
        if (atm is not None) and (surf is None) and (water is None):
            SIM = -2  # atmosphere only
        elif (atm is None) and (surf is not None) and (water is None):
            SIM = -1  # surface only
        elif (atm is None) and (surf is not None) and (water is not None):
            SIM = 0  # ocean + dioptre
        elif (atm is not None) and (surf is not None) and (water is None):
            SIM = 1  # atmosphere + dioptre
        elif (atm is not None) and (surf is not None) and (water is not None):
            SIM = 2  # atmosphere + dioptre + ocean
        elif (atm is None) and (surf is None) and (water is not None):
            SIM = 3  # ocean only
        else:
            raise Exception('Error in SIM')

        D.update(SIM=np.array([SIM],dtype=np.int32))
        # output file
        D.update(PATHRESULTATSHDF=output)

        #
        # atmosphere
        #
        nprofilesAtm={}
        if atm is not None:
            # write the profile
            if isinstance(wl, (float, int, REPTRAN_IBAND)):
                wl = [wl]
                D.update(LAMBDA=wl)
                profilesAtm, phasesAtm = atm.calc_bands(wl)
                for key in profilesAtm[0].dtype.names:
                    nprofilesAtm[key]=[]
                    for profile in profilesAtm:
                        nprofilesAtm[key]=np.append(nprofilesAtm[key],profile[key])

            D.update(LAMBDA=wl)
        else:  
            # no atmosphere
            Ddef.update(PATHDIFFAER='None')
            Ddef.update(PATHPROFILATM='None')
            nprofilesAtm['H']=[0]
            nprofilesAtm['YDEL']=[0]
            nprofilesAtm['XSSA']=[0]
            nprofilesAtm['percent_abs']=[0]
            nprofilesAtm['IPHA']=[0]
            nprofilesAtm['ALT']=[0]


        #
        # surface
        #
        if surf is None:
            # default surface parameters
            surf = FlatSurface()
            Ddef.update(surf.dict)
        else:
            D.update(surf.dict)


        #
        # ocean profile
        #
        nprofilesOc={}
        nprofilesOc['HO'],nprofilesOc['SSO'],nprofilesOc['IPO']=np.zeros(nlam*2,dtype=np.float32),np.zeros(nlam*2,dtype=np.float32),np.zeros(nlam*2,dtype=np.float32)
        if water is None:
            # use default water values
            nprofilesOc['HO'],nprofilesOc['SSO'],nprofilesOc['IPO']=[0],[0],[0]
        else:
            if isinstance(wl, (float, int, REPTRAN_IBAND)):
                wl = [wl]
            D.update(LAMBDA=wl)
	   
            profilesOc, phasesOc = water.calc_bands(wl)
            for ilam in xrange(0,nlam):
                nprofilesOc['HO'][ilam*2]=0
                nprofilesOc['SSO'][ilam*2]=1
                nprofilesOc['IPO'][ilam*2]=0
                nprofilesOc['HO'][ilam*2+1]=-1.e10
                nprofilesOc['SSO'][ilam*2+1]=profilesOc[ilam][1]/(profilesOc[ilam][1]+profilesOc[ilam][0])
                nprofilesOc['IPO'][ilam*2+1]=profilesOc[ilam][2]


        #
        # environment effect
        #
        if env is None:
            # default values (no environment effect)
            env = Environment()
            Ddef.update(env.dict)
        else:
            D.update(env.dict)

        #
        # update the dictionary with the default parameters
        #
        for k, v in Ddef.items():
            if not k in D:
                D.update({k: v})

        #
        # write the albedo file
        #
        file_alb = tempfile.mktemp(dir=dir_albedo, prefix='albedo_')
        ensure_dir_exists(dir_albedo)
        if 'SURFALB' in D:
            surf_alb = D['SURFALB']
        else:
            surf_alb = -999.
        if water is None:
            seafloor_alb = -999.
        else:
            seafloor_alb = water.alb

        with open(file_alb, 'w') as f:
            f.write('# Surface_alb Seafloor_alb\n')

            albedo=np.zeros(2*nlam)
            for i in xrange(nlam):
                # FIXME: implement spectral albedo
                f.write('{} {}\n'.format(surf_alb, seafloor_alb))
                albedo[2*i]=surf_alb
                albedo[2*i+1]=seafloor_alb

        D.update(PATHALB=file_alb)

        #
        # write the command file
        #
        ################################################################

     
        exe=str(exe)

        #options=['-DRANDPHILOX4x32_7','-DPARAMETRES','-DPROGRESSION']
        if exe=='SMART-G-PP':
            options=['-DRANDPHILOX4x32_7']
        elif exe=='SMART-G-SP':
            options=['-DRANDPHILOX4x32_7','-DSPHERIQUE']

	    # load device.cu
        src_device = open("/home/Younes/SMARTG/src/device.cu").read()

	    # compilation du kernel
        
        mod = SourceModule(src_device,
		        nvcc='/usr/local/cuda/bin/nvcc',
		        options=options,
		        no_extern_c=True,
		         cache_dir='/tmp/',
		        include_dirs=[
                    '/home/Younes/SMARTG/src/',
                    '/home/Younes/SMARTG/src/incRNGs/Random123/'
		            
		            ])

        
        kern = mod.get_function('lancementKernelPy')


        if( SIM==0 or SIM==2 or SIM==3 ):
            NOCE=1
            if phasesOc!=[]:
                foce,phasesOcm,NPHAOCE,imax=calculF(phasesOc,NFOCE,SIM)
		
            else:
                foce,NPHAOCE,imax,phasesOcm=[0],0,0,[0]
        else:
            foce=[0]
            NOCE=0
	              


        if(SIM == -2 or SIM == 1 or SIM == 2):
            NATM=len(profilesAtm[0])-1
            HATM=nprofilesAtm['ALT'][0]
	    
            if phasesAtm!=[]:
                faer,phasesAtmm,NPHAAER,imax=calculF(phasesAtm,NFAER,SIM)
                
            else:
                faer,NPHAAER,imax,phasesAtmm=[0],0,0,[0]
                  
        else:
            faer=[0]
            NATM=0
            HATM=0
        
        
        x0,y0,z0,zph0,hph0=0,0,0,[],[]
        x0,y0,z0,zph0,hph0=impactInit(HATM,NATM,nlam,nprofilesAtm['ALT'],nprofilesAtm['H'],THVDEG,options)

        if '-DSPHERIQUE' in options:
            TAUATM = nprofilesAtm['H'][NATM];
        if '-DSPHERIQUE' in options:
            tabTransDir=np.zeros(nlam,dtype=np.float64)
            for ilam in xrange(0,nlam):
                tabTransDir[ilam] = np.exp(-hph0[NATM+ilam*(NATM+1)]);
        tmp=[]

        
    

        tmp = [(np.uint64,'*nbPhotonsInter', np.zeros(nlam,dtype=np.uint64)),
	           (np.float32,'*tabPhotonsEvents', np.zeros(5*4*NBTHETA * NBPHI * nlam,dtype=np.float32)),
	           (np.float32,'*faer', faer),
	           (np.float32,'*foce', foce),
	           (np.float32,'*ho', nprofilesOc['HO']),
               (np.float32,'*sso',nprofilesOc['SSO']),
               (np.int32,'*ipo',nprofilesOc['IPO']),
	           (np.float32,'*h', nprofilesAtm['H']),
	           (np.float32,'*pMol', nprofilesAtm['YDEL']),
	           (np.float32,'*ssa', nprofilesAtm['XSSA']),
	           (np.float32,'*abs', nprofilesAtm['percent_abs']),
	           (np.int32,'*ip', nprofilesAtm['IPHA']),
	           (np.float32,'*alb', albedo),
	           (np.float32,'*lambda', wl),
	           (np.float32,'*z', nprofilesAtm['ALT'])]

  

        if '-DSPHERIQUE' in options:
            tmp+=[(np.float32,'*hph0',hph0),(np.float32,'*zph0',zph0)]

        if '-DRANDPHILOX4x32_7' in options:
            tmp+=[(np.uint32,'*etat', np.zeros(XBLOCK*1*XGRID*1,dtype=np.uint32)),(np.uint32,'config',0)]

        Tableau = GPUStruct(tmp)

        tmp=[(np.uint32,'nbPhotons', 0),(np.int32,'erreurpoids', 0),(np.int32,'erreurtheta', 0)]
        if '-DPROGRESSION' in options:
            tmp2=[(np.uint64,'nbThreads', 0),(np.uint64,'nbPhotonsSor', 0),(np.uint32,'erreurvxy', 0),(np.int32,'erreurvy', 0),(np.int32,'erreurcase', 0)]
            tmp+=tmp2
        Var = GPUStruct(tmp)
        Init = GPUStruct([(np.float32,'x0', x0),(np.float32,'y0', y0),(np.float32,'z0', z0)])

        #initialisation des constantes
        D['NBPHOTONS']=np.array([D['NBPHOTONS']],dtype=np.int_)
        THV=D['THVDEG']*0.017453293
        STHV=np.array([np.sin(THV)],dtype=np.float32)
        CTHV=np.array([np.cos(THV)],dtype=np.float32)
        GAMAbis=D['DEPO'] / (2-D['DEPO'])
        DELTAbis = (1.0 - GAMAbis) / (1.0 + 2.0*GAMAbis)
        DELTA_PRIMbis = GAMAbis / (1.0 + 2.0*GAMAbis)
        BETAbis  = 3./2. * DELTA_PRIMbis
        ALPHAbis = 1./8. * DELTAbis
        Abis = 1. + BETAbis / (3.0 * ALPHAbis)
        ACUBEbis = Abis * Abis* Abis

        D.update(NATM=np.array([NATM],dtype=np.int32))
        D.update(NOCE=np.array([NOCE],dtype=np.int32))
        D.update(HATM=np.array([HATM],dtype=np.float32))
        D.update(THV=THV)
        D.update(STHV=STHV)
        D.update(CTHV=CTHV)
        D.update(GAMA=GAMAbis)
        D.update(DELTA=DELTAbis)
        D.update(DELTA_PRIM=DELTA_PRIMbis)
        D.update(BETA=BETAbis)
        D.update(ALPHA=ALPHAbis)
        D.update(A=Abis)
        D.update(ACUBE=ACUBEbis)
        #affichage des parametres
        if '-DPARAMETRES' in options:
            afficheParametres(D,options)

     
       #transfert des structures de donnees dans le device 
        Var.copy_to_gpu()
        Tableau.copy_to_gpu()
        Init.copy_to_gpu()



       #transfert des constantes dans le device
        Dbis={}
        for key in D.keys():
            if key in ('NBPHOTONS','NBLOOP','THVDEG','DEPO','WINDSPEED',
                       'THV','GAMA','XBLOCK','YBLOCK','XGRID','YGRID',
                       'STHV','CTHV','NLAM','NOCE','SIM','NATM','BETA',
                       'ALPHA','ACUBE','A','DELTA','NFAER',
                       'NBTHETA','NBPHI','OUTPUT_LAYERS',
                       'SUR','DIOPTRE','ENV','ENV_SIZE',
                       'NH2O','X0','Y0','DELTA_PRIM','NFOCE','NFAER'):
                exec("a,_=mod.get_global('%sd')"%key)
                cuda.memcpy_htod(a, D[key])

        
        
        #execution du kernel
        tempsPrec = 0
        nbPhotonsTot=0
        nbPhotonsTotInter=np.zeros(nlam,dtype=np.uint64)
        nbPhotonsSorTot=0
        tabPhotonsTot=np.zeros(5*4*NBTHETA * NBPHI * nlam,dtype=np.float32)
        p = Progress(NBPHOTONS)

	    
        #kern(Var.get_ptr(),Tableau.get_ptr(),Init.get_ptr(),block=(1,1,1), grid=(1,1,1))
        
         
        #Tableau.copy_from_gpu()
        #Var.copy_from_gpu()
        #nbPhotonsTot+=Var.nbPhotons
	
        
        #if '-DPROGRESSION' in options:
        #    nbPhotonsSorTot += Var.nbPhotonsSor;
        #    afficheProgress(nbPhotonsTot,NBPHOTONS,tempsPrec,Var,options,nbPhotonsSorTot)
        
        passageBoucle = False;
        if(nbPhotonsTot < NBPHOTONS):
            passageBoucle = True;
    
	########################
	#########BOUCLE#########
	########################
	
        while(nbPhotonsTot < NBPHOTONS):

            #remise à zero de certaines variables de certains tableaux
            Tableau.tabPhotonsEvents=np.zeros(5*4*NBTHETA * NBPHI * nlam,dtype=np.float32)
            Tableau.nbPhotonsInter=np.zeros(nlam,dtype=np.int32)
            Var.nbPhotons=np.uint32(0)
            if '-DPROGRESSION' in options:
                Var.nbPhotonsSor = np.uint32(0)
            

            Tableau.copy_to_gpu()
            Var.copy_to_gpu()          
            
            kern(Var.get_ptr(),Tableau.get_ptr(),Init.get_ptr(),block=(256,1,1), grid=(256,1,1))
            
            
	        #recuperation du resultat
            Tableau.copy_from_gpu()
            Var.copy_from_gpu()
            nbPhotonsTot+=Var.nbPhotons
	    
            #Creation d'un fichier témoin pour pouvoir reprendre la simulation en cas d'arrêt
            tabPhotonsTot=[x + y for x, y in zip(tabPhotonsTot, Tableau.tabPhotonsEvents)]
            for ilam in xrange(0,nlam):
                nbPhotonsTotInter[ilam]+=Tableau.nbPhotonsInter[ilam]
		
            if '-DPROGRESSION' in options:
                nbPhotonsSorTot += Var.nbPhotonsSor;
            #afficheProgress(nbPhotonsTot,NBPHOTONS,tempsPrec,Var,options,nbPhotonsSorTot)
	    
            if nbPhotonsTot>NBPHOTONS:
	        p.update(NBPHOTONS,afficheProgress(nbPhotonsTot,NBPHOTONS,tempsPrec,Var,options,nbPhotonsSorTot))
            else:
                p.update(nbPhotonsTot,afficheProgress(nbPhotonsTot,NBPHOTONS,tempsPrec,Var,options,nbPhotonsSorTot))
	    

        # Si on n'est pas passé dans la boucle on affiche quand-même l'avancement de la simulation
        #if(passageBoucle==False):
            #afficheProgress(nbPhotonsTot,NBPHOTONS,tempsPrec,Var,options,nbPhotonsSorTot)

        #finalisation
        tabFinalEvent=np.zeros(5*4*NBTHETA*NBPHI*nlam,dtype=np.float64)
        tabTh = np.zeros(NBTHETA,dtype=np.float64)
        tabPhi = np.zeros(NBPHI,dtype=np.float64)

	
        #calcul du tableau final
        for k in xrange(0,5):
            calculTabFinal(tabFinalEvent[k*4*NBTHETA*NBPHI*nlam:(k+1)*4*NBTHETA*NBPHI*nlam], tabTh, tabPhi,tabPhotonsTot[k*4*NBTHETA*NBPHI*nlam:(k+1)*4*NBTHETA*NBPHI*nlam],nbPhotonsTot,nbPhotonsTotInter,NBTHETA,NBPHI,nlam)

        #stockage dans un fichier HDF
        self.output=creerHDFResultats(tabFinalEvent,NBPHI,NBTHETA,tabTh,tabPhi,nlam,tabPhotonsTot)
	p.finish('traitement termine :'+afficheProgress(nbPhotonsTot,NBPHOTONS,tempsPrec,Var,options,nbPhotonsSorTot))
	

    def read(self, dataset=None):
        '''
        read SMARTG result as a LUT (if dataset is provided) or MLUT (default)
        '''
        if dataset is not None:
            return read_lut_hdf(self.output, dataset)
        else:
            return read_mlut_hdf(self.output)


    def view(self, QU=False, field='up (TOA)'):
        '''
        visualization of a smartg result

        Options:
            QU: show Q and U also (default, False)
        '''
        from smartg_view import smartg_view

        smartg_view(self.output, QU=QU, field=field)


def ensure_dir_exists(file_or_dir):
    if isdir(file_or_dir):
        dir_name = file_or_dir
    else:
        dir_name = dirname(file_or_dir)
    if not exists(dir_name):
        makedirs(dir_name)


def command_file_template(dict):
    '''
    returns the content of the command file based on dict
    '''
    return textwrap.dedent("""
        ################ SIMULATION #####################
        # Number of "Photons" to inject (unsigned long long)
        NBPHOTONS = {NBPHOTONS}

        # View Zenith Angle in degree (float)
        THVDEG = {THVDEG}

        # Number of output azimut angle boxes from 0 to PI
        NBPHI = {NBPHI}
        # Number of output zenith angle boxes from 0 to PI
        NBTHETA = {NBTHETA}

        # Simulation type
            # -2 Atmosphere only
            # -1 Dioptre only
            #  0 Ocean and dioptre
            #  1 Atmosphere and dioptre
            #  2 Atmosphere, dioptre and ocean
            #  3 Ocean only
        SIM = {SIM}

        # Dioptre type
            # 0 = plan
            # 1 = roughened sea surface with multiple reflections
            # 2 = roughened sea surface without multiple reflections
            # 3 = lambertian reflector (LAND)
        DIOPTRE = {DIOPTRE}

        # Processes at the surface dioptre
            # 1 Forced reflection
            # 2 Forced transmission
            # 3 Reflection and transmission
        SUR = {SUR}

        # Output layers as a binary flag 
            # 0  TOA always present
            # 1  Add BOA (0+) downward and BOA (0-) upward
            # 2  Add BOA (0+) upward and BOA (0-) downward
        OUTPUT_LAYERS = {OUTPUT_LAYERS}

        # Absolute name of output file 
        PATHRESULTATSHDF = {PATHRESULTATSHDF}

        # SEED for random number series
            # SEED > 0 Random series generated from this SEED (allow to redo the same simulation)
            # SEED =-1 A SEED is randomly generated
        SEED = {SEED}

        ################ ATMOSPHERE #####################
        # Depolarization coefficient of air
        DEPO = {DEPO}

        # Absolute name of file containing the vertical profile of atmosphere
            # Format 
	PATHPROFILATM={PATHPROFILATM}
        # Absolute name of file containing the atmospheric phase matrix 
            # Format 
        PATHDIFFAER = {PATHDIFFAER}

        ################ SURFACE #####################
        # Absolute name of file containing the Land and Seafloor lambertian albedo 
            # Format 
        PATHALB = {PATHALB}

        # Windspeed (m/s) (if DIOPTRE = 1,2 or 4)
        WINDSPEED = {WINDSPEED}

        # Relatibve refarctive index air/water
        NH2O = {NH2O}

        #_______________ Environement effects _____________________#

        # Environment effects (circular target surrrounded by environment)
          # 0  No effect (target horizontally homogeneous)
          # 1  Effect included
        ENV = {ENV}
        # Target radius (km)
        ENV_SIZE= {ENV_SIZE}
        # X0 horizontal shift (in km) in X direction between the center of the target and the point on Earth viewed
        X0= {X0}
        # Y0 horizontal shift (in km) in Y direction between the center of the target and the point on Earth viewed
        Y0= {Y0}

        ################ OCEAN   #####################
        # Absolute name of file containing the Vertical profile of Ocean optical depth and single scattering albedo
            # Format 
        PATHPROFILOCE = {PATHPROFILOCE}
        # Absolute name of file containing the Ocean phase function 
            # Format 
        PATHDIFFOCE = {PATHDIFFOCE}

        ################ PARAMETERS   #####################

        # number of samples for the computation of the Cumulative Distribution Function of the aerosol phase matrix
        NFAER = {NFAER}

        # number of samples for the computation of the Cumulative Distribution Function of the ocean phase matrix
        NFOCE = {NFOCE}

        # LOOP number in the kernel for each thread
        NBLOOP = {NBLOOP} 

        #_______________ GPU _____________________#

        # Threads organized as BLOCKS of size XBLOCK*YBLOCK, with XBLOCK a multiple of 32
        XBLOCK = {XBLOCK} 
        YBLOCK = {YBLOCK} 

        # et les blocks sont eux-même rangés dans un grid de taille XGRID*YGRID (limite par watchdog il faut tuer X + limite XGRID<65535 et YGRID<65535)
        # BLOCKS organized as GRID of size XGRID*YGRID with XGRID<65535 and YGRID<65535
        XGRID = {XGRID} 
        YGRID = {YGRID} 

        # Device selection (-1 to select the 1st available device)
        DEVICE  -1
    """).format(**dict)


class FlatSurface(object):
    '''
    Definition of a flat sea surface

    Arguments:
        SUR: Processes at the surface dioptre
            # 1 Forced reflection
            # 2 Forced transmission
            # 3 Reflection and transmission
        NH2O: Relative refarctive index air/water
    '''
    def __init__(self, SUR=3, NH2O=1.33):
        self.dict = {
                'SUR': np.array([SUR],dtype=np.int32),
                'DIOPTRE': np.array([0],dtype=np.int32),
                'WINDSPEED': np.array([-999.],dtype=np.float32),
                'NH2O': np.array([NH2O],dtype=np.float32),
                }
    def __str__(self):
        return 'FLATSURF-SUR={SUR}'.format(**self.dict)

class RoughSurface(object):
    '''
    Definition of a roughened sea surface

    Arguments:
        MULT: include multiple reflections at the surface
              (True => DIOPTRE=1 ; False => DIOPTRE=2)
        WIND: wind speed (m/s)
        SUR: Processes at the surface dioptre
            # 1 Forced reflection
            # 2 Forced transmission
            # 3 Reflection and transmission
        NH2O: Relative refarctive index air/water
    '''
    def __init__(self, MULT=False, WIND=5., SUR=3, NH2O=1.33):
        self.dict = {
                'SUR': np.array([SUR],dtype=np.int32),
                'DIOPTRE': {True:np.array([1],dtype=np.int32), False:np.array([2],dtype=np.int32)}[MULT],
                'WINDSPEED': np.array([WIND],dtype=np.float32),
                'NH2O': np.array([NH2O],dtype=np.float32),
                }
    def __str__(self):
        return 'ROUGHSUR={SUR}-WIND={WINDSPEED}-DI={DIOPTRE}'.format(**self.dict)


class LambSurface(object):
    '''
    Definition of a lambertian reflector

    ALB: Albedo of the reflector
    '''
    def __init__(self, ALB=0.5):
        self.dict = {
                'SUR':np.array([1],dtype=np.int32),
                'DIOPTRE': np.array([3],dtype=np.int32),
                'SURFALB': ALB,
                'WINDSPEED':np.array([-999.],dtype=np.float32),
                'NH2O': np.array([-999.],dtype=np.float32),
                }
    def __str__(self):
        return 'LAMBSUR-ALB={SURFALB}'.format(**self.dict)

class Environment(object):
    '''
    Stores the smartg parameters relative the the environment effect
    '''
    def __init__(self, ENV=0, ENV_SIZE=0., X0=0., Y0=0., ALB=0.5):
        self.dict = {
                'ENV': np.array([ENV],dtype=np.int),
                'ENV_SIZE': np.array([ENV_SIZE],dtype=np.float32),
                'X0': np.array([X0],dtype=np.float32),
                'Y0': np.array([Y0],dtype=np.float32),
                'SURFALB': ALB,
                }

    def __str__(self):
        return 'ENV={ENV_SIZE}-X={X0:.1f}-Y={Y0:.1f}'.format(**self.dict)


def reptran_merge(files, ibands, output=None):
    '''
    merge (average) results from several correlated-k bands

    Arguments:
        * files: a list of smartg files to merge
        * ibands: a list of corresponding REPTRAN_IBANDs
        * output: the hdf file to create
            if None (default), the output file is determined by extracting the
            common prefix and suffix of all input files, and insert the band
            name inbetween

    Returns the output file name
    '''

    if output is None:
        # determine the common prefix and suffix of all files
        # and insert the band name between those
        base = basename(files[0])
        i = base.find('_WL')
        i += 3
        j = base.find('_', i)
        output = join(dirname(files[0]),
                    (base[:i]               # prefix
                    + ibands[0].band.name  # iband name
                    + base[j:]))            # suffix

    print 'Merging {} files into {}'.format(len(files), output)

    hdf_out = SD(output, SDC.WRITE|SDC.CREATE)
    hdf_ref = SD(files[0])
    for dataset in hdf_ref.datasets():

        sdsref = hdf_ref.select(dataset)
        rank = sdsref.info()[1]
        shape = sdsref.info()[2]
        dtype  = sdsref.info()[3]
        if rank < 2:
            # axis: write the axis as-is
            S = sdsref.get()
        else:
            # average all files
            S, norm = 0., 0.
            for i in xrange(len(files)):

                file = files[i]
                iband = ibands[i]

                hdf = SD(file)
                data = hdf.select(dataset).get()
                hdf.end()

                S += data * iband.weight * iband.extra
                norm += iband.weight * iband.extra

            S /= norm
            S = S.astype(data.dtype)

        # write the dataset
        sds = hdf_out.create(dataset, dtype, shape)
        sds.setcompress(SDC.COMP_DEFLATE, 9)
        sds[:] = S[:]
        # copy sds attributes from first file
        for a in sdsref.attributes().keys():
            setattr(sds, a, sdsref.attributes()[a])
        sds.endaccess()

    # copy global attributes from first file
    for a in hdf_ref.attributes():
        setattr(hdf_out, a, hdf_ref.attributes()[a])

    hdf_out.end()

    return output

##################################################
##################################################
##################################################
def creerHDFResultats(tabFinal,NBPHI,NBTHETA,tabTh,tabPhi,NLAM,tabPhotonsTot):
    #creation de la lookup table specifique a tabFinal


    tabThBis = np.round(tabTh/0.017453293)
    tabPhiBis = np.round(tabPhi/0.017453293)
    wl=np.arange(NLAM)



    label=['I_up (TOA)','Q_up (TOA)','U_up (TOA)']
 

    luts=[]

    for i in xrange(0,3):
        if NLAM==1:
	    a=tabFinal[i*NBPHI*NBTHETA*NLAM:(i+1)*NBPHI*NBTHETA*NLAM]
            a.resize(NBPHI,NBTHETA)
            b= LUT(a, axes=[tabPhiBis,tabThBis],names=['Azimut angles','Zenith angles'],desc=label[i])
            luts.append(b)
	else:
            a=tabFinal[i*NBPHI*NBTHETA*NLAM:(i+1)*NBPHI*NBTHETA*NLAM]
            a.resize(NLAM,NBPHI,NBTHETA)
            b= LUT(a, axes=[wl,tabPhiBis,tabThBis],names=['Wavelet length','Azimut angles','Zenith angles'],desc=label[i])
            luts.append(b)



    HDF=MLUT(luts)

    #HDF.save('test2.hdf',overwrite=True)

    return HDF

    


def impactInit(HATM,NATM,NLAM,ALT,H,THVDEG,options):

    vx = -np.sin(THVDEG*0.017453293)
    vy = 0.
    vz = -np.cos(THVDEG*0.017453293)
    # Calcul du point d'impact
    thv = THVDEG*0.017453293
    rdelta = 4*6400*6400 + 4*(np.tan(thv)*np.tan(thv)+1)*(HATM*HATM+2*HATM*6400)
    localh = (-2.*6400+np.sqrt(rdelta) )/(2.*(np.tan(thv)*np.tan(thv)+1.))

    x0 = localh*np.tan(thv)
    y0 = 0
    z0 = localh
    zph0,hph0=[],[]

    if '-DSPHERIQUE' in options:
        z0+=6400
        zph0=np.zeros((NATM+1),dtype=np.float32)
        hph0=np.zeros((NATM+1)*NLAM,dtype=np.float32)

    xphbis = x0;
    yphbis = y0;
    zphbis = z0;

    for icouche in xrange(1,NATM+1):
        rdelta = 4.*(vx*xphbis + vy*yphbis + vz*zphbis)*(vx*xphbis + vy*yphbis + vz*zphbis)- 4.*(xphbis*xphbis + yphbis*yphbis + zphbis*zphbis-(ALT[icouche]+6400)*(ALT[icouche]+6400));
        rsol1 = 0.5*(-2*(vx*xphbis + vy*yphbis + vz*zphbis) + np.sqrt(rdelta))
        rsol2 = 0.5*(-2*(vx*xphbis + vy*yphbis + vz*zphbis) - np.sqrt(rdelta))

		    # solution : la plus petite distance positive
        if rsol1>0:
            if rsol2>0:
			    rsolfi = min(rsol1,rsol2)
            else:
			    rsolfi = rsol1;
        else:
            if rsol2>0:
                rsolfi=rsol2

        if '-DSPHERIQUE' in options:
            zph0[icouche] = zph0[icouche-1] + np.float32(rsolfi)
            for ilam in xrange(0,NLAM):
                hph0[icouche + ilam*(NATM+1)] = hph0[icouche-1+ ilam*(NATM+1)] +( abs( H[icouche+ ilam*(NATM+1)] - H[icouche-1+ ilam*(NATM+1)])*rsolfi )/( abs( ALT[icouche-1] - ALT[icouche]) );


	    xphbis+= vx*rsolfi;
	    yphbis+= vy*rsolfi;
	    zphbis+= vz*rsolfi;


    return x0,y0,z0,zph0,hph0




def calculOmega(tabTh,tabPhi,tabOmega,NBTHETA,NBPHI):

    tabds=np.zeros(NBTHETA * NBPHI,dtype=np.float64)
    # Zenith angles of the center of the output angular boxes
    dth = 1.5707963 / NBTHETA
    tabTh[0] = dth / 2.

    for ith in xrange(1,NBTHETA):
        tabTh[ith] = tabTh[ith-1] + dth


    # Azimut angles of the center of the output angular boxes
    dphi = 3.1415927 / NBPHI
    tabPhi[0] = dphi / 2.
    for iphi in xrange(1,NBPHI):
        tabPhi[iphi] = tabPhi[iphi-1] + dphi

	# Solid angles of the output angular boxes
    sumds = 0
    for ith in xrange(0,NBTHETA):
        dth = 1.5707963 / NBTHETA
        for iphi in xrange(0,NBPHI):
		    tabds[ith * NBPHI + iphi] = np.sin(tabTh[ith]) * dth * dphi;
		    sumds += tabds[ith * NBPHI + iphi]


	# Normalisation de l'aire de chaque morceau de sphère
    for ith in xrange(0,NBTHETA):
        for iphi in xrange(0,NBPHI):
            tabOmega[ith * NBPHI + iphi] = tabds[ith * NBPHI + iphi] / sumds






def calculTabFinal(tabFinal,tabTh,tabPhi,tabPhotonsTot,nbPhotonsTot,nbPhotonsTotInter,NBTHETA,NBPHI,NLAM):

    tabOmega = np.zeros(NBTHETA * NBPHI,dtype=np.float32)
    calculOmega(tabTh, tabPhi, tabOmega,NBTHETA,NBPHI)
    
	# Remplissage du tableau final
    for iphi in xrange(0,NBPHI):
        for ith in xrange(0,NBTHETA):
            norm = 2.0 * tabOmega[ith*NBPHI+iphi] * np.cos(tabTh[ith])
            for i in xrange(0,NLAM):
                normInter = norm * nbPhotonsTotInter[i]
                # Reflectance
                tabFinal[0*NBTHETA*NBPHI*NLAM+i*NBTHETA*NBPHI+iphi*NBTHETA+ith] = (tabPhotonsTot[0*NBPHI*NBTHETA*NLAM+i*NBTHETA*NBPHI+ith*NBPHI+iphi] + tabPhotonsTot[1*NBPHI*NBTHETA*NLAM+i*NBTHETA*NBPHI+ith*NBPHI+iphi])/ normInter
                # Q
                tabFinal[1*NBTHETA*NBPHI*NLAM + i*NBTHETA*NBPHI + iphi*NBTHETA + ith]  = (tabPhotonsTot[0*NBPHI*NBTHETA*NLAM + i*NBTHETA*NBPHI + ith*NBPHI + iphi]-tabPhotonsTot[1*NBPHI*NBTHETA*NLAM+i*NBTHETA*NBPHI + ith*NBPHI + iphi])/normInter
                # U
                tabFinal[2*NBTHETA*NBPHI*NLAM + i*NBTHETA*NBPHI + iphi*NBTHETA + ith] = (tabPhotonsTot[2*NBPHI*NBTHETA*NLAM + i*NBTHETA*NBPHI + ith*NBPHI + iphi]) / normInter
                # N
                tabFinal[3*NBTHETA*NBPHI*NLAM + i*NBTHETA*NBPHI + iphi*NBTHETA + ith] = (tabPhotonsTot[3*NBPHI*NBTHETA*NLAM + i*NBTHETA*NBPHI + ith*NBPHI + iphi])








def afficheParametres(D,options):
    print ("#--------- Paramètres de simulation --------#\n");
    print (" NBPHOTONS =\t"+str(D['NBPHOTONS'][0]));
    print (" NBTHETA =\t"+str(D['NBTHETA'][0]));
    print (" NBPHI\t=\t"+str(D['NBPHI'][0]));
    print (" THVDEG\t=\t"+str(D['THVDEG'][0])+"(degrés)");
    print (" LAMBDA\t=\t"+str(D['LAMBDA'][0]));
    print (" NLAM\t=\t"+str(D['NLAM'][0]));
    print (" SIM\t=\t"+str(D['SIM'][0]));
    if( D['SIM']==-2 ):
        print ("\t(Atmosphère seule)");
    if( D['SIM']==-1 ):
        print ("\t(Dioptre seul)");
    if( D['SIM']==0 ):
        print ("\t(Océan et Surface)");
    if( D['SIM']==1 ):
        print ("\t(Atmosphère et Surface)");
    if( D['SIM']==2 ):
        print ("\t(Atmosphère, Dioptre et Océan)");
    if( D['SIM']==3 ):
        print ("\t(Océan seul)");
    print (" SEED\t=\t"+str(D['SEED']));
    print ("#------- Paramètres de performances --------#\n");
    print (" NBLOOP\t=\t"+str(D['NBLOOP'][0]));
    print (" XBLOCK\t=\t"+str(D['XBLOCK'][0]));
    print (" YBLOCK\t=\t"+str(D['YBLOCK'][0]));
    print (" XGRID\t=\t"+str(D['XGRID'][0]));
    print (" YGRID\t=\t"+str(D['YGRID'][0]));


    print ("#--------------- Atmosphère ----------------#\n");
    if( D['SIM'][0]==-2 or D['SIM'][0]==1 or D['SIM'][0]==2 ):

        if '-DSPHERIQUE' in options:
            print (" Géométrie de l'atmosphère: \tSphérique");
        else:
            print (" Géométrie de l'atmosphère: \tParallèle");

        print (" NFAER\t=\t"+str(D['NFAER'][0]));
        print (" NATM\t=\t"+str(D['NATM'][0]));
        print (" HATM\t=\t"+str(D['HATM'][0]));

    else:
        print ("\tPas de contribution de l'atmosphère\n");



    print ("#--------- Contribution du dioptre ---------#\n");
    if( D['SIM'][0]==-1 or D['SIM'][0]==0 or D['SIM'][0]==1 or D['SIM'][0]==2 ):
        print (" SUR\t=\t"+str(D['SUR'][0]));
        print (" DIOPTRE =\t"+str(D['DIOPTRE'][0]));
        print (" WINDSPEED =\t"+str(D['WINDSPEED'][0]));
    else:
        print ("\tPas de dioptre\n");


    print ("#--------- Contribution de l'environnement -----#\n");
    if( D['ENV'][0]!=0):
        print (" ENV_SIZE\t=\t"+str(D['ENV_SIZE'][0])+"(km)");
        print (" X0 =\t"+str(D['X0'][0])+"(km)");
        print (" Y0 =\t"+str(D['Y0'][0])+"(km)");
    else:
        print ("\tPas d'effet d'environnement\n");


        print ("#----------------- Océan ------------------#\n");
        print (" NFOCE\t=\t"+str(D['NFOCE'][0]));
        print (" NH2O\t=\t"+str(D['NH2O'][0]));



    # Calcul la date et l'heure courante
    date = time.localtime();
    print ("Date de début  : %02u/%02u/%04u %02u:%02u:%02u\n"% (date.tm_mday, date.tm_mon, date.tm_year,
		   date.tm_hour, date.tm_min, date.tm_sec));

######################################################
######################################################
######################################################



#nbPhotonsTot,var,tempsPrec
def afficheProgress(nbPhotonsTot,NBPHOTONS,tempsPrec,var,options,nbPhotonsSorTot):
	#Calcul la date et l'heure courante
    date=time.localtime()


    # Calcul du temps ecoule et restant
    tempsProg = time.clock()
    tempsTot = tempsProg + tempsPrec
    tempsEcoule = tempsTot
    hEcoulees = tempsEcoule / 3600
    minEcoulees = (tempsEcoule%3600) / 60
    secEcoulees = tempsEcoule%60

    tempsRestant = (tempsTot * (NBPHOTONS / nbPhotonsTot - 1.))
    if tempsRestant < 0:
        tempsRestant = 0
    hRestantes = tempsRestant / 3600
    minRestantes = (tempsRestant%3600) / 60
    secRestantes = tempsRestant%60
    # Calcul du pourcentage de photons traités
    pourcent = (100 * nbPhotonsTot / NBPHOTONS);
    # Affichage
    chaine = ''
    #chaine += '--------------------------------------\n'
    chaine += 'Photons lances : %e (%3d%%)' % (nbPhotonsTot,pourcent)
    '''
    chaine += 'Temps ecoule   : %d h %2d min %2d sec' % (hEcoulees, minEcoulees, secEcoulees)
    chaine += 'Temps restant  : %d h %2d min %2d sec' % (hRestantes, minRestantes, secRestantes)
    chaine += 'Date actuelle  : %02u/%02u/%04u %02u:%02u:%02u' % (date.tm_mday, date.tm_mon, date.tm_year, date.tm_hour,date.tm_min, date.tm_sec)
    chaine += '--------------------------------------\n'
    '''
    if '-DPROGRESSION' in options:
        print ' - phot sortis: %e ' % (nbPhotonsSorTot);

    return chaine


def calculF(phases,N,SIM):
    nmax,n,imax=0,0,0
    phasesAtmm=[]
    for idx,phase in enumerate(phases):
        if phase.N>nmax:
            imax,nmax=idx,phase.N
        n+=1
    phase_H=np.zeros(5*n*N,dtype=np.float32)
    for idx,phase in enumerate(phases):
        if idx!=imax:
            phase.ang.resize(nmax)
            phase.phase.resize(nmax,4)
        tmp=np.append(phase.ang,phase.phase)
        phasesAtmm=np.append(phasesAtmm,tmp)
        scum = np.zeros(phase.N)
        #conversion en gradiant
        if(SIM == -2 or SIM == 1 or SIM == 2):
            phase.ang*=0.017453293
        for iang in xrange(1,phase.N):
            dtheta=phase.ang[iang]-phase.ang[iang-1]
            pm1= phase.phase[iang-1,1] + phase.phase[iang-1,0]
            pm2= phase.phase[iang,1] + phase.phase[iang,0]
            sin1= np.sin(phase.ang[iang-1])
            sin2= np.sin(phase.ang[iang])
            scum[iang] = scum[iang-1] + dtheta*( (sin1*pm1+sin2*pm2)/3 + (sin1*pm2+sin2*pm1)/6 )*6.2831853;

          #normalisation
        for iang in xrange(0,phase.N):
            scum[iang] /= scum[phase.N-1]
         #calcul des faer
        ipf=0
        for iang in xrange(0,N):
            z=np.float64(iang+1)/np.float64(N)
            while scum[ipf+1]<z:
                ipf +=1

            phase_H[idx*5*N+iang*5+4] = np.float32( ((scum[ipf+1]-z)*phase.ang[ipf] + (z-scum[ipf])*phase.ang[ipf+1])/(scum[ipf+1]-scum[ipf]) )
            phase_H[idx*5*N+iang*5+0] = np.float32( phase.phase[ipf,1])
            phase_H[idx*5*N+iang*5+1] = np.float32( phase.phase[ipf,0])
            phase_H[idx*5*N+iang*5+2] = np.float32( phase.phase[ipf,2])
            phase_H[idx*5*N+iang*5+3] = np.float32(0)


    return phase_H,phasesAtmm,n,imax


def test_rayleigh():
    '''
    Basic Rayleigh example
    '''
    return Smartg('SMART-G-PP', wl=400., NBPHOTONS=1e9, atm=Profile('afglt'), overwrite=True)

def test_kokhanovsky():
    '''
    Just Rayleigh : kokhanovsky test case
    '''
    return Smartg('SMART-G-PP', wl=500., DEPO=0., NBPHOTONS=1e9,
            atm=Profile('afglt', grid='100[75]25[5]10[1]0'),
            output=join(dir_output, 'example_kokhanovsky.hdf'))


def test_rayleigh_aerosols():
    '''
    with aerosols
    '''
    aer = AeroOPAC('maritime_clean', 0.4, 550.)
    pro = Profile('afglms', aer=aer)

    return Smartg('SMART-G-PP', wl=np.linspace(400, 600, 10.), atm=pro, NBPHOTONS=1e9)


def test_atm_surf():
    # lambertian surface of albedo 10%
    return Smartg('SMART-G-PP', 490., NBPHOTONS=1e9,
            output=join(dir_output, 'test_atm_surf.hdf'),
            atm=Profile('afglms'),
            surf=LambSurface(ALB=0.1),
            overwrite=True)


def test_atm_surf_ocean():
    return Smartg('SMART-G-PP', wl=np.linspace(400, 600, 10.), NBPHOTONS=1e7,
            atm=Profile('afglms', aer=AeroOPAC('maritime_clean', 0.2, 550)),
            surf=RoughSurface(),
            NBTHETA=30,
            water=IOP_MM(chl=1., NANG=1000),
            overwrite=True)


def test_surf_ocean():
    return Smartg('SMART-G-SP',490., THVDEG=30., NBPHOTONS=2e6,
            surf=RoughSurface(),
            water=IOP_MM(1., pfwav=[400.]))



def test_ocean():
    return Smartg('SMART-G-PP', wl=np.linspace(400, 600, 10.), THVDEG=30.,
            water=IOP_MM(chl=1.,pfwav=[400,500]), NBPHOTONS=5e6)


def test_reptran():
    '''
    using reptran
    '''
    aer = AeroOPAC('maritime_polluted', 0.4, 550.)
    pro = Profile('afglms.dat', aer=aer, grid='100[75]25[5]10[1]0')
    files, ibands = [], []
    for iband in REPTRAN('reptran_solar_msg').band('msg1_seviri_ch008').ibands():
        job = Smartg('SMART-G-PP', wl=np.mean(iband.band.awvl),
                NBPHOTONS=5e8,
                iband=iband, atm=pro)
        files.append(job.output)
        ibands.append(iband)

    reptran_merge(files, ibands)


def test_ozone_lut():
    '''
    Ozone Gaseous transmission for MERIS
    '''
    from itertools import product

    list_TCO = [350., 400., 450.]   # ozone column in DU
    list_AOT = [0.05, 0.1, 0.4]     # aerosol optical thickness

    luts = []
    for TCO, AOT in product(list_TCO, list_AOT):

        aer = AeroOPAC('maritime_clean', AOT, 550.)
        pro = Profile('afglms', aer=aer, O3=TCO)

        job = Smartg('SMART-G-PP', wl=490., atm=pro, NBTHETA=50, NBPHOTONS=5e6)

        lut = job.read('I_up (TOA)')
        lut.attrs.update({'TCO':TCO, 'AOT': AOT})
        luts.append(lut)
    merged = merge(luts, ['TCO', 'AOT'])
    merged.print_info()
    merged.savesave(join(dir_output, 'test_ozone.hdf'))

def test_multispectral():
    '''
    process multiple bands at once
    '''

    pro = Profile('afglt',
    grid=[100, 75, 50, 30, 20, 10, 5, 1, 0.],  # optional, otherwise use default grid
    pfgrid=[100, 20, 0.],   # optional, otherwise use a single band 100-0
    pfwav=[400, 500, 600], # optional, otherwise phase functions are calculated at all bands
    aer=AeroOPAC('maritime_clean', 0.3, 550.),
    verbose=True)

    return Smartg('SMART-G-PP', wl = np.linspace(400, 600, 10.),
             THVDEG=60.,
             atm=pro,
             surf=RoughSurface(),
             water=IOP_SPM(1.),
             overwrite=True)



def set_persistant(db,objet,cle):
    db[cle]=objet 

if __name__ == '__main__':
    test_rayleigh()
    #test_kokhanovsky()
    #test_rayleigh_aerosols()
    #test_atm_surf()
    #test_atm_surf_ocean()
    #test_surf_ocean()
    #test_ocean()
    #test_reptran()
    #test_ozone_lut()
    #test_multispectral()
