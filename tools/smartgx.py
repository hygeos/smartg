#!/usr/bin/env python
# encoding: utf-8


'''
SMART-G
Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU
'''


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


import numpy as np
from smartg import Smartg, RoughSurface, LambSurface, FlatSurface, Environment
from gpustruct import GPUStruct
import time
from numpy import pi
from pyhdf.SD import SD, SDC
from profile.profil import AeroOPAC, Profile, REPTRAN, REPTRAN_IBAND, CloudOPAC
from water.iop_spm import IOP_SPM
from water.iop_mm import IOP_MM
from water.phase_functions import PhaseFunction
from os.path import dirname, realpath, join, basename
import textwrap
from progress import Progress
from luts import merge, read_lut_hdf, read_mlut_hdf, LUT, MLUT


# set up default directories
#
# base smartg directory is one directory above here
dir_install = dirname(dirname(realpath(__file__)))
dir_kernel = join(dir_install, 'src/')

class Smartg(object):
    '''
    Run a SMART-G job

    Arguments:

        - wl: wavelength in nm (float)
              or a list of wavelengths, or an array
              used for phase functions calculation (always)
              and profile calculation (if iband is None)   # FIXME
        - pp:
            True: use plane parallel geometry (default)
            False: use spherical shell geometry
        - iband: a REPTRAN_BAND object describing the internal band
            default None (no reptran mode)
        - atm: Profile object
            default None (no atmosphere)
        - surf: Surface object
            default None (no surface)
        - water: Iop object, providing options relative to the ocean surface
            default None (no ocean)
        - env: environment effect parameters (dictionary)
            default None (no environment effect)

    Attributes:
        - output: the name of the result file
    '''
    def __init__(self, wl, pp=True,
           iband=None, atm=None, surf=None, water=None, env=None,
           NBPHOTONS=1e9, DEPO=0.0279, THVDEG=0., SEED=-1,
           NBTHETA=45, NBPHI=45,
           NFAER=10000, NFOCE=10000,
           OUTPUT_LAYERS=0, XBLOCK=256, XGRID=256,
           NBLOOP=None):

        #
        # initialization
        #

        if NBLOOP is None:
            NBLOOP = NBPHOTONS/30

        # number of output levels
        NLVL = 5

        # number of Stockes parameters
        NPSTK = 4

        assert isinstance(wl, (float, list, np.ndarray))
        assert (iband is None) or isinstance(iband, REPTRAN_IBAND)

        if isinstance(wl, (list, np.ndarray)):
            NLAM = len(wl)
        else:
            NLAM = 1

        D = {
                'NBPHOTONS': str(int(NBPHOTONS)),
                'THVDEG': np.array([THVDEG], dtype=np.float32),
                'DEPO': np.array([DEPO], dtype=np.float32),
                'SEED': SEED,
                'NBTHETA': np.array([NBTHETA], dtype=np.int32),
                'NBPHI': np.array([NBPHI], dtype=np.int32),
                'NFAER': np.array([NFAER], dtype=np.uint32),
                'NFOCE': np.array([NFOCE], dtype=np.uint32),
                'OUTPUT_LAYERS': np.array([OUTPUT_LAYERS], dtype=np.uint32),
                'XBLOCK': np.array([XBLOCK], dtype=np.int32),
                'YBLOCK': np.array([1], dtype=np.int32),
                'XGRID': np.array([XGRID], dtype=np.int32),
                'YGRID': np.array([1], dtype=np.int32),
                'NBLOOP': np.array([NBLOOP], dtype=np.uint32),
                'NLAM': np.array([NLAM], dtype=np.int32),
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

        D.update(SIM=np.array([SIM], dtype=np.int32))

        #
        # atmosphere
        # get the phase function and the atmospheric profile

        nprofilesAtm, phasesAtm, NATM, HATM = get_profAtm(wl,atm,D)
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
        # get the phase function and oceanic profile
        nprofilesOc, phasesOc, NOCE = get_profOc(wl, water, D, NLAM)
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
        if 'SURFALB' in D:
            surf_alb = D['SURFALB']
        else:
            surf_alb = -999.
        if water is None:
            seafloor_alb = -999.
        else:
            seafloor_alb = water.alb

        albedo = np.zeros(2*NLAM)
        for i in xrange(NLAM):
        # FIXME: implement spectral albedo
            albedo[2*i] = surf_alb
            albedo[2*i+1] = seafloor_alb

        """
        compilation option
        list of compilation flag :
            - DRANDPHILOX4x32_7 : Utilisation du random Philox-4x32-7
            - DPROGRESSION : Calcul et affichage de la progression de la simulation
            - DSPHERIQUE : Calcul en sphérique
            - DDEBUG : Ajout de tests intermédiaires utilisés lors du débugage
        """

        options = ['-DRANDPHILOX4x32_7','-DPROGRESSION']
        # options.extend(['-DPARAMETRES','-DPROGRESSION'])
        if not pp:
            options.append('-DSPHERIQUE')

        # load device.cu
        src_device = open(dir_kernel+'device.cu').read()

        # kernel compilation
        mod = SourceModule(src_device,
                           nvcc='/usr/local/cuda/bin/nvcc',
                           options=options,
                           no_extern_c=True,
                           cache_dir='/tmp/',
                           include_dirs=[dir_kernel, dir_kernel+'incRNGs/Random123/'])
        # get the kernel
        kern = mod.get_function('lancementKernelPy')

	# computation of the phase function
        if(SIM == 0 or SIM == 2 or SIM == 3):
            if phasesOc != []:
                foce, phasesOcm, NPHAOCE, imax = calculF(phasesOc, NFOCE)
            else:
                foce, NPHAOCE, imax, phasesOcm = [0], 0, 0, [0]
        else:
            foce, phasesOcm = [0], [0]

        if(SIM == -2 or SIM == 1 or SIM == 2):
            if phasesAtm != []:
                faer, phasesAtmm, NPHAAER, imax = calculF(phasesAtm, NFAER)
                
            else:
                faer, NPHAAER, imax, phasesAtmm = [0], 0, 0, [0]
                  
        else:
            faer, phasesAtmm = [0], [0]

        
        # computation of the impact point
        x0, y0, z0, zph0, hph0 = impactInit(HATM, NATM, NLAM, nprofilesAtm['ALT'], nprofilesAtm['H'], THVDEG, options)

        if '-DSPHERIQUE' in options:
            TAUATM = nprofilesAtm['H'][NATM];
            tabTransDir = np.zeros(NLAM,dtype=np.float64)
            for ilam in xrange(0, NLAM):
                tabTransDir[ilam] = np.exp(-hph0[NATM + ilam * (NATM + 1)])

            if '-DDEBUG' in options:
                print ("Paramètres initiaux du photon: taumax0=%lf - zintermax=%lf - (%lf,%lf,%lf)\n" %
		       hph0[NATM+1], zph0[NATM+1], x0, y0, z0)

        # write the input variables into data structures
        Tableau, Var, Init = InitSD(nprofilesAtm, nprofilesOc, NLAM,
                                NLVL, NPSTK, NBTHETA, NBPHI, faer, foce,
                                albedo, wl, hph0, zph0, x0, y0, z0,
                                XBLOCK, XGRID, options)

        # initialization of the constants
        InitConstantes(D, surf, env, NATM, NOCE, HATM, mod)

        # Loop and kernel call
        nbPhotonsTot, nbPhotonsTotInter , nbPhotonsTotInter, nbPhotonsSorTot, tabPhotonsTot, p = loop_kernel(NBPHOTONS, Tableau, Var, Init,
                                                                                                          NLVL, NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                                                                                                          NLAM, options, kern)
        # compute the final result
        tabFinalEvent = np.zeros(NLVL*NPSTK*NBTHETA*NBPHI*NLAM, dtype=np.float64)
        tabTh = np.zeros(NBTHETA, dtype=np.float64)
        tabPhi = np.zeros(NBPHI, dtype=np.float64)

	# Création et calcul du tableau final (regroupant le poids de tous les photons ressortis sur une demi-sphère,
	# par unité de surface)

        for k in xrange(0, 5):
            calculTabFinal(tabFinalEvent[k*NPSTK*NBTHETA*NBPHI*nlam:(k+1)*NPSTK*NBTHETA*NBPHI*nlam],
                           tabTh, tabPhi, tabPhotonsTot[k*NPSTK*NBTHETA*NBPHI*nlam:(k+1)*NPSTK*NBTHETA*NBPHI* nlam],
                           nbPhotonsTot, nbPhotonsTotInter, NBTHETA, NBPHI, nlam)

        # stockage des resultats dans une MLUT
        self.output = creerMLUTsResultats(tabFinalEvent, NBPHI, NBTHETA, tabTh, tabPhi, nlam, tabPhotonsTot,nbPhotonsTot,D,nprofilesAtm,nprofilesOc)
        p.finish('traitement termine :' + afficheProgress(nbPhotonsTot, NBPHOTONS, options, nbPhotonsSorTot))
  
    def view(self, QU=False, field='up (TOA)'):
        '''
        visualization of a smartg result

        Options:
            QU: show Q and U also (default, False)
        '''
        from smartg_view import smartg_view

        smartg_view(self.output, QU=QU, field=field)


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

def creerMLUTsResultats(tabFinal, NBPHI, NBTHETA, tabTh, tabPhi, NLAM,tabPhotonsTot,nbPhotonsTot,D,nprofilesAtm,nprofilesOc):
    """
    store the result in a MLUT
    Arguments :
        - tabTh : Zenith angles
        - tabPhi : Azimutal angles
        - tabFinal : R, Q, U of all outgoing photons
        - NBPHI : Number of intervals in azimuth angle
        - NBTHETA : Number of intervals in zenith
        - NLAM : Number of wavelet length
        - tabPhotonsTot : Total weight of all outgoing photons
        - nbPhotonsTot : Total number of photons processed
        - D : Dictionary containing all the parameters required to launch the simulation by the kernel
        - nprofilesAtm : List of atmospheric profiles set contiguously
        - nprofilesOc : List of oceanic profiles set contiguously
    Returns :
        - Res : MLUT corresponding to the final result

    """

    # theta in degrees
    tabThBis = np.round(tabTh/(np.pi / 180))
    # phi in degrees
    tabPhiBis = np.round(tabPhi/(np.pi / 180))

    nbwl = np.arange(NLAM)

    label = ['I_up (TOA)', 'Q_up (TOA)', 'U_up (TOA)','N_up (TOA)']

    luts = []

    # ecriture des données de sorties
    for i in xrange(0, 4):
        if NLAM == 1:
            a = tabFinal[i*NBPHI*NBTHETA*NLAM:(i+1)*NBPHI*NBTHETA*NLAM]
            a.resize(NBPHI, NBTHETA)
            b =  LUT(a, axes=[tabPhiBis, tabThBis], names=['Azimut angles', 'Zenith angles'], desc=label[i])
            luts.append(b)
        else:
            a = tabFinal[i*NBPHI*NBTHETA*NLAM:(i+1)*NBPHI*NBTHETA*NLAM]
            a.resize(NLAM, NBPHI, NBTHETA)
            b = LUT(a, axes=[nbwl, tabPhiBis, tabThBis], names=['Wavelet length', 'Azimut angles', 'Zenith angles'], desc=label[i])
            luts.append(b)
   
    if D['OUTPUT_LAYERS'][0] == 1:
        label = ['I_down (0+)', 'Q_down (0+)', 'U_down (0+)','N_down (0+)']
        for i in xrange(0, 4):
            if NLAM == 1:
                a = tabFinal[(4+i)*NBPHI*NBTHETA*NLAM:(4+i+1)*NBPHI*NBTHETA*NLAM]
                a.resize(NBPHI, NBTHETA)
                b =  LUT(a, axes=[tabPhiBis, tabThBis], names=['Azimut angles', 'Zenith angles'], desc=label[i])
                luts.append(b)
            else:
                a = tabFinal[(4+i)*NBPHI*NBTHETA*NLAM:((4+i+1))*NBPHI*NBTHETA*NLAM]
                a.resize(NLAM, NBPHI, NBTHETA)
                b = LUT(a, axes=[nbwl, tabPhiBis, tabThBis], names=['Wavelet length', 'Azimut angles', 'Zenith angles'], desc=label[i])
                luts.append(b)

        label = ['I_up (0-)', 'Q_up (0-)', 'U_up (0-)','N_up (0-)']
        for i in xrange(0, 4):
            if NLAM == 1:
                a = tabFinal[(4*4+i)*NBPHI*NBTHETA*NLAM:(4*4+i+1)*NBPHI*NBTHETA*NLAM]
                a.resize(NBPHI, NBTHETA)
                b =  LUT(a, axes=[tabPhiBis, tabThBis], names=['Azimut angles', 'Zenith angles'], desc=label[i])
                luts.append(b)
            else:
                a = tabFinal[(4*4+i)*NBPHI*NBTHETA*NLAM:(4*4+i+1)*NBPHI*NBTHETA*NLAM]
                a.resize(NLAM, NBPHI, NBTHETA)
                b = LUT(a, axes=[nbwl, tabPhiBis, tabThBis], names=['Wavelet length', 'Azimut angles', 'Zenith angles'], desc=label[i])
                luts.append(b)

    if D['OUTPUT_LAYERS'][0] == 2:
        label = ['I_down (0-)', 'Q_down (0-)', 'U_down (0-)','N_down (0-)']
        for i in xrange(0, 4):
            if NLAM == 1:
                a = tabFinal[(2*4+i)*NBPHI*NBTHETA*NLAM:(2*4+i+1)*NBPHI*NBTHETA*NLAM]
                a.resize(NBPHI, NBTHETA)
                b =  LUT(a, axes=[tabPhiBis, tabThBis], names=['Azimut angles', 'Zenith angles'], desc=label[i])
                luts.append(b)
            else:
                a = tabFinal[(2*4+i)*NBPHI*NBTHETA*NLAM:((2*4+i+1))*NBPHI*NBTHETA*NLAM]
                a.resize(NLAM, NBPHI, NBTHETA)
                b = LUT(a, axes=[nbwl, tabPhiBis, tabThBis], names=['Wavelet length', 'Azimut angles', 'Zenith angles'], desc=label[i])
                luts.append(b)

        label = ['I_up (0+)', 'Q_up (0+)', 'U_up (0+)','N_up (0+)']
        for i in xrange(0, 4):
            if NLAM == 1:
                a = tabFinal[(3*4+i)*NBPHI*NBTHETA*NLAM:(3*4+i+1)*NBPHI*NBTHETA*NLAM]
                a.resize(NBPHI, NBTHETA)
                b =  LUT(a, axes=[tabPhiBis, tabThBis], names=['Azimut angles', 'Zenith angles'], desc=label[i])
                luts.append(b)
            else:
                a = tabFinal[(3*4+i)*NBPHI*NBTHETA*NLAM:(3*4+i+1)*NBPHI*NBTHETA*NLAM]
                a.resize(NLAM, NBPHI, NBTHETA)
                b = LUT(a, axes=[nbwl, tabPhiBis, tabThBis], names=['Wavelet length', 'Azimut angles', 'Zenith angles'], desc=label[i])
                luts.append(b)

    
    Res = MLUT(luts)

    # ecriture des profiles Atmosphériques
    if D['SIM'][0]==-2 or D['SIM'][0]==1 or D['SIM'][0]==2:
        luts = []
        keys=nprofilesAtm.keys()
        for key in keys:
            a=nprofilesAtm[key]
            b=np.resize(a,(NLAM,D['NATM']+1))
            c=LUT(b, desc=key)
            luts.append(c)
        profAtm = MLUT(luts)

    # ecriture des profiles Océaniques
    if D['SIM'][0]==0 or D['SIM'][0]==2 or D['SIM'][0]==3:
        luts = []
        keys=nprofilesOc.keys()
        for key in keys:
            a=nprofilesOc[key]
            b=np.resize(a,(NLAM,D['NOCE']+1))
            c=LUT(b, desc=key)
            luts.append(c)
        profOc = MLUT(luts)

    return Res

def impactInit(HATM, NATM, NLAM, ALT, H, THVDEG, options):
    """
    Calcul du profil que le photon va rencontrer lors de son premier passage dans l'atmosphère
    Sauvegarde de ce profil dans tab et sauvegarde des coordonnées initiales du photon dans init

    Arguments :
        - HATM : Altitude of the Top of Atmosphere
        - NATM : Number of layers of the atmosphere
        - NLAM : Number of wavelet length
        - ALT : Altitude of the atmosphere
        - H : optical thickness of each layer in the atmosphere
        - THVDEG : View Zenith Angle in degree
        - options : compilation options

    Returns :
        - (x0, y0, z0) : carthesian coordinates
        - hph0 : Optical thickness seen in front of the photon
        - zph0 : Corresponding Altitude
    """
    vx = -np.sin(THVDEG * np.pi / 180)
    vy = 0.
    vz = -np.cos(THVDEG * np.pi /180)
    # Calcul du point d'impact
    
    thv = THVDEG * np.pi / 180
    
    rdelta = 4 * 6400 * 6400 + 4 * (np.tan(thv) * np.tan(thv) + 1) * (HATM * HATM + 2 * HATM * 6400)
    localh = (-2. * 6400 + np.sqrt(rdelta) )/(2. * (np.tan(thv) * np.tan(thv) + 1.))

    x0 = localh * np.tan(thv)
    y0 = 0
    z0 = localh
    zph0, hph0 = [], []

    if '-DSPHERIQUE' in options:
        z0 += 6400
        zph0 = np.zeros((NATM + 1), dtype=np.float32)
        hph0 = np.zeros((NATM + 1)*NLAM, dtype=np.float32)

    xphbis = x0;
    yphbis = y0;
    zphbis = z0;
    
    for icouche in xrange(1, NATM + 1):
        rdelta = 4. * (vx * xphbis + vy * yphbis + vz * zphbis) * (vx * xphbis + vy * yphbis
                    + vz * zphbis) - 4. * (xphbis * xphbis + yphbis * yphbis
                    + zphbis * zphbis - (ALT[icouche] + 6400) * (ALT[icouche] + 6400))
        rsol1 = 0.5 * (-2 * (vx * xphbis + vy * yphbis + vz * zphbis) + np.sqrt(rdelta))
        rsol2 = 0.5 * (-2 * (vx * xphbis + vy * yphbis + vz * zphbis) - np.sqrt(rdelta))

        # solution : la plus petite distance positive
        if rsol1 > 0:
            if rsol2 > 0:
                rsolfi = min(rsol1, rsol2)
            else:
                rsolfi = rsol1;
        else:
            if rsol2 > 0:
                rsolfi = rsol2

        if '-DSPHERIQUE' in options:
            zph0[icouche] = zph0[icouche-1] + np.float32(rsolfi)
            for ilam in xrange(0, NLAM):
                hph0[icouche + ilam * (NATM + 1)] = hph0[icouche - 1 + ilam * (NATM + 1)] + (abs(H[icouche + ilam * (NATM + 1)]
                                                    - H[icouche - 1+ ilam * (NATM + 1)]) * rsolfi )/(abs(ALT[icouche - 1] - ALT[icouche]));

                xphbis += vx*rsolfi;
                yphbis += vy*rsolfi;
                zphbis += vz*rsolfi;


    return x0, y0, z0, zph0, hph0

def calculOmega(tabTh, tabPhi, tabOmega, NBTHETA, NBPHI):

    """
    compute the normalized area of each box, its theta , its psi in the form of 3 arrays

    Arguments :
        - tabTh : Zenith angles
        - tabPhi : Azimutal angles
        - NBTHETA : Number of intervals in zenith
        - NBPHI : Number of intervals in azimuth angle
        - tabOmega : Solid angle

    """


    tabds = np.zeros(NBTHETA * NBPHI, dtype=np.float64)
    # Zenith angles of the center of the output angular boxes
    dth = np.pi / 2 / NBTHETA
    tabTh[0] = dth / 2.

    for ith in xrange(1, NBTHETA):
        tabTh[ith] = tabTh[ith - 1] + dth

    # Azimut angles of the center of the output angular boxes
    dphi = pi/NBPHI
    tabPhi[0] = dphi / 2.
    for iphi in xrange(1, NBPHI):
        tabPhi[iphi] = tabPhi[iphi - 1] + dphi

    # Solid angles of the output angular boxes
    sumds = 0
    for ith in xrange(0, NBTHETA):
        dth = pi/(2 * NBTHETA)
        for iphi in xrange(0, NBPHI):
            tabds[ith * NBPHI + iphi] = np.sin(tabTh[ith]) * dth * dphi;
            sumds += tabds[ith * NBPHI + iphi]

    # Normalisation de l'aire de chaque morceau de sphère
    for ith in xrange(0, NBTHETA):
        for iphi in xrange(0, NBPHI):
            tabOmega[ith * NBPHI + iphi] = tabds[ith * NBPHI + iphi] / sumds



def calculTabFinal(tabFinal, tabTh, tabPhi, tabPhotonsTot, nbPhotonsTot, nbPhotonsTotInter, NBTHETA, NBPHI, NLAM):
    """
    compute the tabFinal corresponding to R, Q, U of all outgoing photons

    Arguments:
        - tabFinal : R, Q, U of all outgoing photons
        - tabTh : Zenith angles
        - tabPhi : Azimutal angles
        - tabPhotonsTot : Total weight of all outgoing photons
        - nbPhotonsTot : Total number of photons processed
        - nbPhotonsTotInter : Total number of photons processed by interval
        - NBTHETA : Number of intervals in zenith
        - NBPHI : Number of intervals in azimuth angle
        - NLAM  : Number of wavelet length

    """
    # Fonction qui remplit le tabFinal correspondant à la reflectance (R), Q et U sur tous l'espace de sorti (dans chaque boite)
    tabOmega = np.zeros(NBTHETA * NBPHI, dtype=np.float64)
    calculOmega(tabTh, tabPhi, tabOmega, NBTHETA, NBPHI)
    
    # Remplissage du tableau final
    for iphi in xrange(0, NBPHI):
        for ith in xrange(0, NBTHETA):
            norm = 2.0 * tabOmega[ith * NBPHI + iphi] * np.cos(tabTh[ith])
            for i in xrange(0, NLAM):
                normInter = norm * nbPhotonsTotInter[i]
                # Reflectance
                tabFinal[0 * NBTHETA * NBPHI * NLAM + i * NBTHETA * NBPHI + iphi * NBTHETA + ith] = (tabPhotonsTot[0 * NBPHI * NBTHETA * NLAM + i * NBTHETA * NBPHI + ith * NBPHI + iphi]  + tabPhotonsTot[1 * NBPHI * NBTHETA * NLAM+i * NBTHETA * NBPHI+ith * NBPHI + iphi]) / normInter
                # Q
                tabFinal[1 * NBTHETA * NBPHI * NLAM + i * NBTHETA * NBPHI + iphi * NBTHETA + ith]  = (tabPhotonsTot[0 * NBPHI * NBTHETA * NLAM + i * NBTHETA * NBPHI + ith * NBPHI + iphi] - tabPhotonsTot[1 * NBPHI * NBTHETA * NLAM + i * NBTHETA * NBPHI + ith * NBPHI + iphi]) / normInter
                # U
                tabFinal[2 * NBTHETA * NBPHI * NLAM + i * NBTHETA * NBPHI + iphi * NBTHETA + ith] = (tabPhotonsTot[2 * NBPHI * NBTHETA * NLAM + i * NBTHETA * NBPHI + ith * NBPHI + iphi]) / normInter
                # N
                tabFinal[3 * NBTHETA * NBPHI * NLAM + i * NBTHETA * NBPHI + iphi * NBTHETA + ith] = (tabPhotonsTot[3 * NBPHI * NBTHETA * NLAM + i * NBTHETA * NBPHI + ith * NBPHI + iphi])


def afficheProgress(nbPhotonsTot, NBPHOTONS, options, nbPhotonsSorTot):
    """
    function showing the progression of the radiative transfert simulation

    Arguments :
        - nbPhotonsTot : Total number of photons processed
        - NBPHOTONS : Number of photons injected
        - options : compilation options
        - nbPhotonsSorTot : Total number of outgoing photons
    ----------------------------------------------------
    Returns :
        - chaine : string containing the information concerning the progression

    """

    # Calcul du pourcentage de photons traités
    pourcent = (100 * nbPhotonsTot / NBPHOTONS);
    # Affichage
    chaine = ''
    chaine += 'Photons lances : %e (%3d%%)' % (nbPhotonsTot, pourcent)

    if '-DPROGRESSION' in options:
        chaine += ' - phot sortis: %e ' % (nbPhotonsSorTot);

    return chaine


def calculF(phases, N):

    """
    Compute CDF of scattering phase matrices

    Arguments :
        - phases : list of phase functions
        - N : Number of discrete values of the phase function
    --------------------------------------------------
    Returns :
        - idx  : corresponding to a function phase
        - imax : index of the function phase containing the maximal number of angles describing the phase function
        - nmax : maximal number of angles describing the phase function (equivalent to MLSAAER and MLSAOCE in the old version)
        - phases_list : list of phase function set contiguously
        - phase_H : cumulative distribution of the phase functions

    NB : le programme smartg.py écrit les fonctions de phases dans des fichiers et convertit les angles en degrees si ces derniers sont en radians
    Ici la fonction calculF recupere les angles et verifie si ces derniers sont en degres et les convertit en radians

    """
    nmax, n, imax=0, 0, 0
    phases_list = []
    # define the number of phases functions and the maximal number of angles describing each of them
    for idx, phase in enumerate(phases):
        if phase.N>nmax:
            imax, nmax = idx, phase.N
        n += 1
    # Initialize the cumulative distribution function
    phase_H = np.zeros(5*n*N, dtype=np.float32)
    for idx, phase in enumerate(phases):
        if idx != imax:
            # resizing the attributes of the phase object following the nmax
            phase.ang.resize(nmax)
            phase.phase.resize(nmax, 4)
        tmp = np.append(phase.ang, phase.phase)
        phases_list = np.append(phases_list, tmp)
        scum = np.zeros(phase.N)
        # conversion en gradiant
        if phase.degrees == True:
           angles = phase.ang_in_rad()
           # un biais apparait lorsque la ligne ci-dessous est mise en commentaire dans le test rayleigh + aerosols ???
           # l'attribut ang de l'objet phase n'est pourtant utilisé que dans cette fonction
           phase.ang = phase.ang_in_rad()
        else:
           angles = phase.ang

        scum = [0]
        dtheta = np.diff(angles)
        pm = phase.phase[:, 1] + phase.phase[:, 0]
        sin = np.sin(angles)
        tmp = dtheta * ((sin[:-1] * pm[:-1] + sin[1:] * pm[1:]) / 3. +  (sin[:-1] * pm[1:] + sin[1:] * pm[:-1])/6. )* np.pi * 2.
        scum = np.append(scum,tmp)
        scum = np.cumsum(scum)
        scum /= scum[phase.N-1]

        ipf = 0
        converted_N = np.float64(N)
        for iang in xrange(0, N):
            base_index = idx*5*N+iang*5
            z = np.float64(iang+1)/converted_N
            while scum[ipf+1]<z:
                ipf += 1

            phase_H[base_index+4] = np.float32( ((scum[ipf+1]-z)*phase.ang[ipf] + (z-scum[ipf])*phase.ang[ipf+1])/(scum[ipf+1]-scum[ipf]) )
            phase_H[base_index+0] = np.float32( phase.phase[ipf, 1])
            phase_H[base_index+1] = np.float32( phase.phase[ipf, 0])
            phase_H[base_index+2] = np.float32( phase.phase[ipf, 2])
            phase_H[base_index+3] = np.float32(0)


    return phase_H, phases_list, n, imax

def test_rayleigh():
    '''
    Basic Rayleigh example
    '''
    return Smartg(wl=400., NBPHOTONS=1e9, atm=Profile('afglt'))

def test_kokhanovsky():
    '''
    Just Rayleigh : kokhanovsky test case
    '''
    return Smartg('SMART-G-PP', wl=500., DEPO=0., NBPHOTONS=1e9,
            atm=Profile('afglt', grid='100[75]25[5]10[1]0'))

def test_rayleigh_aerosols():
    '''
    with aerosols
    '''
    aer = AeroOPAC('maritime_clean', 0.4, 550.)
    pro = Profile('afglms', aer=aer)

    return Smartg(wl=490., atm=pro, NBPHOTONS=1e9)

def test_atm_surf():
    # lambertian surface of albedo 10%
    return Smartg('SMART-G-PP', 490., NBPHOTONS=1e9,
            atm = Profile('afglms'),
            surf = LambSurface(ALB=0.1))


def test_atm_surf_ocean():
    return Smartg('SMART-G-PP', 490., NBPHOTONS=1e7,
            atm=Profile('afglms', aer=AeroOPAC('maritime_clean', 0.2, 550)),
            surf=RoughSurface(),
            NBTHETA=30,
            water=IOP_MM(chl=1., NANG=1000))


def test_surf_ocean():
    return Smartg('SMART-G-PP',490., THVDEG=30., NBPHOTONS=2e6,
            surf=RoughSurface(),
            water=IOP_MM(1., pfwav=[400.]))



def test_ocean():
    return Smartg(wl=560., THVDEG=30.,
            water=IOP_SPM(100.), NBPHOTONS=5e6)


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
             water=IOP_SPM(1.))


def InitConstantes(D,surf,env,NATM,NOCE,HATM,mod):

    """
    Initialize the constants in python and send them to the device memory

    Arguments:

        - D: Dictionary containing all the parameters required to launch the simulation by the kernel
        - surf : surf: Surface object
        - env : environment effect parameters (dictionary)
        - NATM : Number of layers of the atmosphere
        - NOCE : Number of layers of the ocean
        - HATM : Altitude of the Top of Atmosphere
        - mod : PyCUDA module compiling the kernel

    """


    D['NBPHOTONS'] = np.array([D['NBPHOTONS']], dtype=np.int_)
    THV = D['THVDEG']*np.pi/180.
    STHV = np.array([np.sin(THV)], dtype=np.float32)
    CTHV = np.array([np.cos(THV)], dtype=np.float32)
    GAMAbis = D['DEPO'] / (2-D['DEPO'])
    DELTAbis = (1.0 - GAMAbis) / (1.0 + 2.0 *GAMAbis)
    DELTA_PRIMbis = GAMAbis / (1.0 + 2.0*GAMAbis)
    BETAbis  = 3./2. * DELTA_PRIMbis
    ALPHAbis = 1./8. * DELTAbis
    Abis = 1. + BETAbis / (3.0 * ALPHAbis)
    ACUBEbis = Abis * Abis* Abis

    if surf != None:
        D['SUR'] = np.array(D['SUR'], dtype=np.int32)
        D['DIOPTRE'] = np.array(D['DIOPTRE'], dtype=np.int32)
        D['WINDSPEED'] = np.array(D['WINDSPEED'], dtype=np.float32)
        D['NH2O'] = np.array(D['NH2O'], dtype=np.float32)
    if env != None:
        D['ENV'] = np.array(D['ENV'], dtype=np.int32)
        D['ENV_SIZE'] = np.array(D['ENV_SIZE'], dtype=np.float32)
        D['X0'] = np.array(D['X0'], dtype=np.float32)
        D['Y0'] = np.array(D['Y0'], dtype=np.float32)
    # dictionary update

    D.update(NATM=np.array([NATM], dtype=np.int32))
    D.update(NOCE=np.array([NOCE], dtype=np.int32))
    D.update(HATM=np.array([HATM], dtype=np.float32))
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

    # copy the constants into the device memory
    for key in ('NBPHOTONS', 'NBLOOP', 'THVDEG', 'DEPO', 'WINDSPEED',
                   'THV', 'GAMA', 'XBLOCK', 'YBLOCK', 'XGRID', 'YGRID',
                   'STHV', 'CTHV', 'NLAM', 'NOCE', 'SIM', 'NATM', 'BETA',
                   'ALPHA', 'ACUBE', 'A', 'DELTA', 'NFAER',
                   'NBTHETA', 'NBPHI', 'OUTPUT_LAYERS',
                   'SUR', 'DIOPTRE', 'ENV', 'ENV_SIZE',
                   'NH2O', 'X0', 'Y0', 'DELTA_PRIM', 'NFOCE', 'NFAER'):
        a,_ = mod.get_global('%sd'%key)
        cuda.memcpy_htod(a, D[key])

def InitSD(nprofilesAtm, nprofilesOc, nlam,
           NLVL, NPSTK, NBTHETA, NBPHI, faer,
           foce, albedo, wl, hph0, zph0, x0, y0,
           z0,XBLOCK, XGRID, options):

    """
    Initialize the principles data structures in python and send them the device memory

    Arguments:

        - nprofilesAtm: Atmospheric profile
        - nprofilesOc : Oceanic profile
        - nlam: Number of wavelet length
        - NLVL : Number of output levels
        - NPSTK : Number of stockes parameter
        - NBTHETA : Number of intervals in zenith
        - NBPHI : Number of intervals in azimuth angle
        - faer : CDF of scattering phase matrices (Atmosphere)
        - foce : CDF of scattering phase matrices (Ocean)
        - albedo : Spectral Albedo
        - wl: wavelet length
        - hph0 : Optical thickness seen in front of the photon
        - zph0 : Corresponding Altitude
        - (x0,y0,z0) : Initial coordinates of the photon
        - XBLOCK: Block Size
        - XGRID : Grid Size
        - options: compilation options

    -----------------------------------------------------------

    Returns the following GPUStruct Class:
        * Tableau : Class containing the arrays sent to the device
            Attributes :
                - nbPhotonsInter : number of photons injected by interval of nlam
                - tabPhotons :  stockes parameters of all photons
                - faer : cumulative distribution of the phase functions related to the aerosol
                - foce : cumulative distribution of the phase functions related to the ocean
                - ho : optical thickness of each layer of the ocean model
                - sso : albedo of simple diffusion in ocean
                - ipo : vertical profile of ocean phase function index
                - h : optical thickness of each layer of the atmospheric model
                - pMol : proportion of molecules in each layer of atmospheric model
                - ssa : albedo of simple diffusion of the aerosols in each layer of the atmospheric model
                - abs : proportion of absorbent in each layer of the atmospheric model
                - lambda : wavelet lenghts
                - z : altitudes level in the atmosphere
            optional:
            if SPHERIQUE FLAG
                - hph0 : optical thickness seen in front of the photon
                - zph0 : corresponding altitude
            if DRANDPHILOX4x32_7 FLAG
                - etat : related to the generation of random number
                - config : related to the generation of random number

        * Var : Class containing the variables sent to the device
            Attributes :
                - nbPhotons : Number of photons processed during a kernel call
                - nThreadsActive : Number of active threads
                - erreurpoids : Number of photons having a weight abnormally high
                - erreurtheta : Number of photons ignored
            if PROGRESSION FLAG
                - nbThreads : Total number of thread launched
                - nbPhotonsSor : number of photons reaching the space during a kernel call
                - erreurvxy : number of outgoing photons in the zenith
                - erreurvy : number of outgoing photons
                - erreurcase : number of photons stored in a non existing box

        * Init : Class containing the initial parameters of the photon
            Attributes :
                - x0,y0,z0 : carthesian coordinates of the photon at the initialization
        -----------------------------------------------------------------------------------
        NB : When calling the class GPUStruct, it is important to consider the following elements:
             the structure in python has to be exactly the same as the structure in C ie:
            - the number of attributes as to be the same
            - the types of attributes have to be the same
            - the order of the attributes declared has to be the same
            the kernel has to take pointers as arguments in CUDA and python

        The guideline of this class is defined in the program GPUStruct.py with an example


    """
    tmp = []
    tmp = [(np.uint64, '*nbPhotonsInter', np.zeros(nlam, dtype=np.uint64)),
           (np.float32, '*tabPhotons', np.zeros(NLVL * NPSTK * NBTHETA * NBPHI * nlam, dtype=np.float32)),
           (np.float32, '*faer', faer),
           (np.float32, '*foce', foce),
           (np.float32, '*ho', nprofilesOc['HO']),
           (np.float32, '*sso', nprofilesOc['SSO']),
           (np.int32, '*ipo', nprofilesOc['IPO']),
           (np.float32, '*h', nprofilesAtm['H']),
           (np.float32, '*pMol', nprofilesAtm['YDEL']),
           (np.float32, '*ssa', nprofilesAtm['XSSA']),
           (np.float32, '*abs', nprofilesAtm['percent_abs']),
           (np.int32, '*ip', nprofilesAtm['IPHA']),
           (np.float32, '*alb', albedo),
           (np.float32, '*lambda', wl),
           (np.float32, '*z', nprofilesAtm['ALT'])]

    if '-DSPHERIQUE' in options:
        tmp += [(np.float32, '*hph0', hph0), (np.float32, '*zph0', zph0)]
    if '-DRANDPHILOX4x32_7' in options:
        tmp += [(np.uint32, '*etat', np.zeros(XBLOCK*1*XGRID*1, dtype=np.uint32)), (np.uint32, 'config', 0)]


    Tableau = GPUStruct(tmp)

    tmp = [(np.uint64, 'nbPhotons', 0),(np.int32, 'nThreadsActive', 0), (np.int32, 'erreurpoids', 0), (np.int32, 'erreurtheta', 0)]
    if '-DPROGRESSION' in options:
        tmp2 = [(np.uint64, 'nbThreads', 0), (np.uint64, 'nbPhotonsSor', 0), (np.uint32, 'erreurvxy', 0), (np.int32, 'erreurvy', 0), (np.int32, 'erreurcase', 0)]
        tmp += tmp2
    Var = GPUStruct(tmp)
    Init = GPUStruct([(np.float32, 'x0', x0), (np.float32, 'y0', y0), (np.float32, 'z0', z0)])
    # copy the data to the GPU
    Var.copy_to_gpu(['nbPhotons'])
    Tableau.copy_to_gpu(['tabPhotons','nbPhotonsInter'])
    Init.copy_to_gpu()

    return Tableau,Var,Init


def get_profAtm(wl, atm, D):

    """
    get the atmospheric profile, the altitude of the top of Atmosphere, the number of layers of the atmosphere

    Arguments :
        - wl : wavelet length
        - atm : Profile object
         default None (no atmosphere)
        - D: Dictionary containing all the parameters required to launch the simulation by the kernel
    -----------------------------------------------------------------------------------------------------------
    Returns :
        - phasesAtm : Atmospheric phase functions
        - nprofilesAtm : List of atmospheric profiles set contiguously
        - NATM : Number of layers of the atmosphere
        - HATM : Altitude of the Top of Atmosphere

    """
    nprofilesAtm = {}
    if atm is not None:
        # write the profile
        if isinstance(wl, (float, int, REPTRAN_IBAND)):
            wl = [wl]
        profilesAtm, phasesAtm = atm.calc_bands(wl)
        # the altitude is get only by the first atmospheric profile
        nprofilesAtm['ALT'] = profilesAtm[0]['ALT']
        # remove the key Altitude from the list of keys
        keys = [x for x in profilesAtm[0].dtype.names if x != 'ALT']
        for key in keys:
            nprofilesAtm[key] = []
            for profile in profilesAtm:
                nprofilesAtm[key] = np.append(nprofilesAtm[key], profile[key])
        D.update(LAMBDA=wl)
        NATM = len(profilesAtm[0])-1
        HATM = nprofilesAtm['ALT'][0]
    else:
        # no atmosphere
        phasesAtm = []
        nprofilesAtm['H'] = [0]
        nprofilesAtm['YDEL'] = [0]
        nprofilesAtm['XSSA'] = [0]
        nprofilesAtm['percent_abs'] = [0]
        nprofilesAtm['IPHA'] = [0]
        nprofilesAtm['ALT'] = [0]
        NATM = 0
        HATM = 0

    return nprofilesAtm, phasesAtm, NATM, HATM

def get_profOc(wl, water, D, nlam):

    """
    get the oceanic profile, the altitude of the top of Atmosphere, the number of layers of the atmosphere
    Arguments :
        - wl : wavelet length
        - water : Profile object
            default None (no atmosphere)
        - D : Dictionary containing all the parameters required to launch the simulation by the kernel
        - nlam : Number of wavelet length
    -------------------------------------------------------------------------------------------------------
    Returns :
        - phasesOc : Oceanic phase functions
        - nprofilesOc : List of oceanic profiles set contiguously
        - NOCE : Number of layers of the ocean

    """

    nprofilesOc = {}
    nprofilesOc['HO'], nprofilesOc['SSO'], nprofilesOc['IPO'] = np.zeros(nlam*2, dtype=np.float32), np.zeros(nlam*2, dtype=np.float32), np.zeros(nlam*2, dtype=np.float32)
    if water is None:
            # use default water values
        nprofilesOc['HO'], nprofilesOc['SSO'], nprofilesOc['IPO'],phasesOc = [0], [0], [0], []
        NOCE = 0
    else:
        if isinstance(wl, (float, int, REPTRAN_IBAND)):
            wl = [wl]
        D.update(LAMBDA=wl)
        profilesOc, phasesOc = water.calc_bands(wl)
        for ilam in xrange(0, nlam):
            nprofilesOc['HO'][ilam*2] = 0
            nprofilesOc['SSO'][ilam*2] = 1
            nprofilesOc['IPO'][ilam*2] = 0
            nprofilesOc['HO'][ilam*2+1] = -1.e10
            nprofilesOc['SSO'][ilam*2+1] = profilesOc[ilam][1]/(profilesOc[ilam][1]+profilesOc[ilam][0])
            nprofilesOc['IPO'][ilam*2+1] = profilesOc[ilam][2]
            # parametrer les indices
        NOCE = 1

    return nprofilesOc, phasesOc, NOCE


def loop_kernel(NBPHOTONS, Tableau, Var, Init, NLVL,
                NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                nlam, options , kern):
    """
    launch the kernel several time until the targeted number of photons injected is reached

    Arguments:
        - NBPHOTONS : Number of photons injected
        - Tableau : Class containing the arrays sent to the device
        - Var : Class containing the variables sent to the device
        - Init : Class containing the initial parameters of the photon
        - NLVL : Number of output levels
        - NPSTK : Number of stockes parameter
        - BLOCK : Block dimension
        - XGRID : Grid dimension
        - NBTHETA : Number of intervals in zenith
        - nlam : Number of wavelet length
        - options : compilation options
        - kern : kernel launching the transfert radiative simulation
    --------------------------------------------------------------
    Returns :
        - nbPhotonsTot : Total number of photons processed
        - nbPhotonsTotInter : Total number of photons processed by interval
        - nbPhotonsSorTot : Total number of outgoing photons
        - tabPhotonsTot : Total weight of all outgoing photons
        - p : progression bar

    """

    # Initialize of the parameters
    nbPhotonsTot = 0
    nbPhotonsTotInter = np.zeros(nlam, dtype=np.uint64)
    nbPhotonsSorTot = 0
    tabPhotonsTot = np.zeros(NLVL*NPSTK*NBTHETA * NBPHI * nlam, dtype=np.float32)

    # Initialize the progress bar
    p = Progress(NBPHOTONS)

    # skip List used to avoid transfering arrays already sent into the device
    skipTableau = ['faer', 'foce', 'ho', 'sso', 'ipo', 'h', 'pMol', 'ssa', 'abs', 'ip', 'alb', 'lambda', 'z']
    skipVar = ['erreurtheta', 'erreurpoids', 'nThreadsActive', 'nbThreads', 'erreurvxy', 'erreurvy', 'erreurcase']

    while(nbPhotonsTot < NBPHOTONS):

        Tableau.tabPhotons = np.zeros(NLVL*NPSTK*NBTHETA * NBPHI * nlam, dtype=np.float32)
        Tableau.nbPhotonsInter = np.zeros(nlam, dtype=np.int32)
        Var.nbPhotons = np.uint32(0)
        if '-DPROGRESSION' in options:
            Var.nbPhotonsSor = np.uint32(0)

            # transfert the data from the host to the device
        Tableau.copy_to_gpu(skipTableau)
        Var.copy_to_gpu(skipVar)

        # kernel launch
        kern(Var.get_ptr(), Tableau.get_ptr(), Init.get_ptr(), block=(XBLOCK, 1, 1), grid=(XGRID, 1, 1))

        # transfert the result from the device to the host
        Tableau.copy_from_gpu(skipTableau)
        Var.copy_from_gpu(skipVar)

        # get the results
        nbPhotonsTot += Var.nbPhotons
        tabPhotonsTot+=Tableau.tabPhotons

        for ilam in xrange(0, nlam):
            nbPhotonsTotInter[ilam] += Tableau.nbPhotonsInter[ilam]

        if '-DPROGRESSION' in options:
            nbPhotonsSorTot += Var.nbPhotonsSor;

        # update of the progression Bar
        if nbPhotonsTot > NBPHOTONS:
            p.update(NBPHOTONS, afficheProgress(nbPhotonsTot, NBPHOTONS, options, nbPhotonsSorTot))
        else:
            p.update(nbPhotonsTot, afficheProgress(nbPhotonsTot, NBPHOTONS, options, nbPhotonsSorTot))

    return nbPhotonsTot, nbPhotonsTotInter , nbPhotonsTotInter, nbPhotonsSorTot, tabPhotonsTot, p


def impactInit(HATM, NATM, NLAM, ALT, H, THVDEG, options):
    """
    Calcul du profil que le photon va rencontrer lors de son premier passage dans l'atmosphère
    Sauvegarde de ce profil dans tab et sauvegarde des coordonnées initiales du photon dans init

    Arguments :
        - HATM : Altitude of the Top of Atmosphere
        - NATM : Number of layers of the atmosphere
        - NLAM : Number of wavelet length
        - ALT : Altitude of the atmosphere
        - H : optical thickness of each layer in the atmosphere
        - THVDEG : View Zenith Angle in degree
        - options : compilation options

    Returns :
        - (x0, y0, z0) : carthesian coordinates
        - hph0 : Optical thickness seen in front of the photon
        - zph0 : Corresponding Altitude
    """
    vx = -np.sin(THVDEG * np.pi / 180)
    vy = 0.
    vz = -np.cos(THVDEG * np.pi / 180)
    # Calcul du point d'impact

    thv = THVDEG * np.pi / 180

    rdelta = 4 * 6400 * 6400 + 4 * (np.tan(thv) * np.tan(thv) + 1) * (HATM * HATM + 2 * HATM * 6400)
    localh = (-2. * 6400 + np.sqrt(rdelta) )/(2. * (np.tan(thv) * np.tan(thv) + 1.))

    x0 = localh * np.tan(thv)
    y0 = 0
    z0 = localh
    zph0, hph0 = [], []

    if '-DSPHERIQUE' in options:
        z0 += 6400
        zph0 = np.zeros((NATM + 1), dtype=np.float32)
        hph0 = np.zeros((NATM + 1)*NLAM, dtype=np.float32)

    xphbis = x0;
    yphbis = y0;
    zphbis = z0;

    for icouche in xrange(1, NATM + 1):
        rdelta = 4. * (vx * xphbis + vy * yphbis + vz * zphbis) * (vx * xphbis + vy * yphbis
                    + vz * zphbis) - 4. * (xphbis * xphbis + yphbis * yphbis
                    + zphbis * zphbis - (ALT[icouche] + 6400) * (ALT[icouche] + 6400))
        rsol1 = 0.5 * (-2 * (vx * xphbis + vy * yphbis + vz * zphbis) + np.sqrt(rdelta))
        rsol2 = 0.5 * (-2 * (vx * xphbis + vy * yphbis + vz * zphbis) - np.sqrt(rdelta))

        # solution : la plus petite distance positive
        if rsol1 > 0:
            if rsol2 > 0:
                rsolfi = min(rsol1, rsol2)
            else:
                rsolfi = rsol1;
        else:
            if rsol2 > 0:
                rsolfi = rsol2

        if '-DSPHERIQUE' in options:
            zph0[icouche] = zph0[icouche-1] + np.float32(rsolfi)
            for ilam in xrange(0, NLAM):
                hph0[icouche + ilam * (NATM + 1)] = hph0[icouche - 1 + ilam * (NATM + 1)] + (abs(H[icouche + ilam * (NATM + 1)]
                                                    - H[icouche - 1 + ilam * (NATM + 1)]) * rsolfi )/(abs(ALT[icouche - 1] - ALT[icouche]));

        xphbis += vx*rsolfi;
        yphbis += vy*rsolfi;
        zphbis += vz*rsolfi;


    return x0, y0, z0, zph0, hph0



if __name__ == '__main__':

    #test_rayleigh()
    #test_kokhanovsky()
    test_rayleigh_aerosols()
    # test_atm_surf()
    # test_atm_surf_ocean()
    # test_surf_ocean()
    #test_ocean()
    # test_reptran()
    # test_ozone_lut()
    # test_multispectral()
