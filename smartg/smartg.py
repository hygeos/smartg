#!/usr/bin/env python
# encoding: utf-8


'''
SMART-G
Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU
'''


from __future__ import print_function, division, absolute_import
import numpy as np
from datetime import datetime
from numpy import pi
from smartg.atmosphere import Atmosphere
from smartg.water import IOP_base
from os.path import dirname, realpath, join, exists
from warnings import warn
from smartg.albedo import Albedo_cst
from smartg.tools.progress import Progress
from smartg.tools.luts import MLUT
from scipy.interpolate import interp1d
import subprocess
from collections import OrderedDict
from pycuda.gpuarray import to_gpu, zeros as gpuzeros
import pycuda.driver as cuda
from smartg.bandset import BandSet
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.driver import module_from_buffer


# set up directories
dir_root = dirname(realpath(__file__))
dir_src = join(dir_root, 'src/')
dir_bin = join(dir_root, 'bin/')
src_device = join(dir_src, 'device.cu')
binnames = {
            True: join(dir_bin, 'pp.cubin'),
            False: join(dir_bin, 'sp.cubin'),
        }


# constants definition
# (should match #defines in src/communs.h)
UPTOA = 0
DOWN0P = 1
DOWN0M = 2
UP0P = 3
UP0M = 4
DOWNB = 5

#
# type definitions (should match cuda struct definitions)
#
type_Phase = [
    ('p_ang', 'float32'),  # \
    ('p_P11', 'float32'),  #  |
    ('p_P12', 'float32'),  #  | equally spaced in
    ('p_P22', 'float32'),  #  | scattering probability
    ('p_P33', 'float32'),  #  | [0, 1]
    ('p_P43', 'float32'),  #  |
    ('p_P44', 'float32'),  # /

    ('a_P11', 'float32'),  # \
    ('a_P12', 'float32'),  #  |
    ('a_P22', 'float32'),  #  | equally spaced in scat.
    ('a_P33', 'float32'),  #  | angle [0, 180]
    ('a_P43', 'float32'),  #  |
    ('a_P44', 'float32'),  # /
    ]

type_Spectrum = [
    ('lambda'      , 'float32'),
    ('alb_surface' , 'float32'),
    ('alb_seafloor', 'float32'),
    ]

type_Profile = [
    ('z',      'float32'),    # // altitude
    ('OD',    'float32'),     # // cumulated extinction optical thickness (from top)
    ('OD_sca', 'float32'),    # // cumulated scattering optical thickness (from top)
    ('OD_abs', 'float32'),    # // cumulated absorption optical thickness (from top)
    ('pmol',   'float32'),    # // probability of pure Rayleigh scattering event
    ('ssa',    'float32'),    # // layer single scattering albedo
    ('iphase', 'int32'),      # // phase function index
    ]

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
                'SUR': SUR,
                'DIOPTRE': 0,
                'WINDSPEED': -999.,
                'NH2O': NH2O,
                }
    def __str__(self):
        return 'FLATSURF-SUR={SUR}'.format(**self.dict)

class RoughSurface(object):
    '''
    Definition of a roughened sea surface

    Arguments:
        WIND: wind speed (m/s)
        SUR: Processes at the surface dioptre
            # 1 Forced reflection
            # 2 Forced transmission
            # 3 Reflection and transmission
        NH2O: Relative refarctive index air/water
    '''
    def __init__(self, WIND=5., SUR=3, NH2O=1.33):
        self.dict = {
                'SUR': SUR,
                'DIOPTRE': 1,
                'WINDSPEED': WIND,
                'NH2O': NH2O,
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
                'SUR': 1,
                'DIOPTRE': 3,
                'SURFALB': ALB,
                'WINDSPEED': -999.,
                'NH2O': -999.,
                }
    def __str__(self):
        return 'LAMBSUR-ALB={SURFALB}'.format(**self.dict)


class Environment(object):
    '''
    Stores the smartg parameters relative the the environment effect

    ENV: activate environment effect (default 0, deactivated)
    ENV_SIZE, X0, Y0: radius and position of the adjacency effect circle
    ALB: albedo model

    NB: water is inside, lambertian is outside.
    '''
    def __init__(self, ENV=0, ENV_SIZE=0., X0=0., Y0=0., ALB=Albedo_cst(0.5)):
        self.dict = {
                'ENV': ENV,
                'ENV_SIZE': ENV_SIZE,
                'X0': X0,
                'Y0': Y0,
                }
        self.alb = ALB

    def __str__(self):
        return 'ENV={ENV_SIZE}-X={X0:.1f}-Y={Y0:.1f}'.format(**self.dict)


class Smartg(object):

    def __init__(self, pp=True, debug=False,
                 alt_move=False, debug_photon=False,
                 double=False, alis=None, rng='PHILOX'):
        '''
        Initialization of the Smartg object

        Performs the compilation and loading of the kernel.
        This class is designed so split compilation and kernel loading from the
        code execution: in case of successive smartg executions, the kernel
        loading time is not repeated.

        Arguments:
            - pp:
                True: use plane parallel geometry (default)
                False: use spherical shell geometry

            Compilation flags, not available if the kernel is provided as a binary:

            - debug: set to True to activate debug mode (optional stdout if problems are detected)

            - alt_move: set to true to activate the alternate move scheme in move_sp.

            - debug_photon: activate the display of photon path for the thread 0

            - double : accumulate photons table in double precision, default single

            - alis : dictionary, if present implement the ALIS method (Emde et al. 2010) for treating gaseous absorption, with field 'nlow'
                is the number of wavelength to fit the spectral dependence of scattering, 
                nlow-1 has to divide NW-1 where NW is the number of wavelengths, nlow has to be lesser than MAX_NLOW that is defined in communs.h

            - rng: choice of pseudo-random number generator:
                   * PHILOX
                   * CURAND_PHILOX
        '''

        self.pp = pp
        self.double = double
        self.alis = alis
        self.rng = init_rng(rng)

        #
        # compilation option
        #
        options = []
        # options = ['-g', '-G']
        if not pp:
            # spherical shell calculation
            options.append('-DSPHERIQUE')
        if debug:
            # additional tests for debugging
            options.append('-DDEBUG')
        if alt_move:
            options.append('-DALT_MOVE')
        if debug_photon:
            options.append('-DDEBUG_PHOTON')
        if double:
            # counting in double precision
            # ! slows down processing
            options.append('-DDOUBLE')
        if alis:
            options.append('-DALIS')
        options.append('-D'+rng)

        #
        # compile the kernel or load binary
        #
        time_before_compilation = datetime.now()
        if exists(src_device):

            # load device.cu
            src_device_content = open(src_device).read()

            # kernel compilation
            self.mod = SourceModule(src_device_content,
                               nvcc='nvcc',
                               options=options,
                               no_extern_c=True,
                               cache_dir='/tmp/',
                               include_dirs=[dir_src,
                                   join(dir_src, 'incRNGs/Random123/')])
        elif exists(binnames[pp]):
            # load existing binary
            print('Loading binary', binnames[pp])
            self.mod = module_from_buffer(open(binnames[pp], 'rb').read())

        else:
            raise IOError('Could not find {} or {}.'.format(src_device, binnames[pp]))

        # load the kernel
        self.kernel = self.mod.get_function('launchKernel')

        #
        # common attributes
        #
        self.common_attrs = OrderedDict()
        self.common_attrs['compilation_time'] = (datetime.now()
                        - time_before_compilation).total_seconds()
        self.common_attrs['device'] = pycuda.autoinit.device.name()
        self.common_attrs['pycuda_version'] = pycuda.VERSION_TEXT
        self.common_attrs['cuda_version'] = '.'.join(map(str, pycuda.driver.get_version()))
        self.common_attrs.update(get_git_attrs())


    def run(self, wl,
             atm=None, surf=None, water=None, env=None,
             NBPHOTONS=1e9, DEPO=0.0279, DEPO_WATER= 0.0906, THVDEG=0., SEED=-1,
             RTER=6371., wl_proba=None,
             NBTHETA=45, NBPHI=90, NF=1e6,
             OUTPUT_LAYERS=0, XBLOCK=256, XGRID=256,
             NBLOOP=None, progress=True,
             le=None, flux=None, stdev=False, BEER=0):
        '''
        Run a SMART-G simulation

        Arguments:

            - wl: a scalar or list/array of wavelengths (in nm)
                  or a list of REPTRAN or KDIS IBANDS

            - atm: Profile object
                default None (no atmosphere)
                Example:
                    # clear atmosphere, AFGL midlatitude summer
                    AtmAFGL('afglms')
                    # AFGL tropical with maritime clear aerosols AOT(550)=0.3
                    AtmAFGL('afglt', aer=[AeroOPAC('maritime_clean', 0.3, 550.)])

            - surf: Surface object
                default None (no surface)
                RoughSurface(WIND=5.)  # wind-roughened ocean surface
                FlatSurface()          # flat air-water interface
                LambSurface(ALB=0.1)   # Lambertian surface of albedo 0.1

            - water: water object, providing options relative to the ocean surface
                default None (no ocean)

            - env: environment effect object (a.k.a. adjacency effect)
                default None (no environment effect)

            - NBPHOTONS: number of photons launched

            - DEPO: (Air) Rayleigh depolarization ratio

            - DEPO_WATER: (Water) Rayleigh depolarization ratio

            - THVDEG: zenith angle of the observer in degrees
                the result corresponds to various positions of the sun
                NOTE: in plane parallel geometry, due to Fermat's principle, we
                can exchange the positions of the sun and observer.

            - SEED: integer used to initiate the series of random numbers
                default: based on clock time

            - RTER: earth radius in km

            - wl_proba: inversed cumulative distribution function for wavelength selection
                        (it is the result of function ICDF(proba, N))

            - NBTHETA: number of zenith angles in output

            - NBPHI: number of azimuth angles in output

            - NF: number of discretization of :
                    * the inversed aerosol phase functions
                    * the inversed ocean phase functions
                    * the inversed probability of each wavelength occurence

            - OUTPUT_LAYERS: control the output layers. Add the following values:
                0: top of atmosphere only (TOA)
                1: add output layers at (0+, down) and (0-, up)
                2: add output layers at (0-, down) and (0+, up)
                Example: OUTPUT_LAYERS=3 to use all output layers.

            - XBLOCK and XGRID: control the number of blocks and grid size for
              the GPU execution

            - NBLOOP: number of photons launched in one kernel run

            - progress: whether to show a progress bar (True/False)

            - le: Local Estimate method activation
                  Provide output geometries in radians like so:
                  le={'th': <array-like>, 'phi': <array-like>}
                  or:
                  le={'th_deg': <array-like>, 'phi_deg': <array-like>}    # to provide angles in degrees
                  angles can be provided as scalar, lists or 1-dim arrays
                  default None: cone sampling
                  NOTE: Overrides NBPHI and NBTHETA

            - flux: if specified output is 'planar' or 'spherical' flux instead of radiance

            - stdev: calculate the standard deviation between each kernel run

            - BEER: if BEER=1 compute absorption using Beer-Lambert law, otherwise compute it with the Single scattering albedo
                (BEER automatically set to 1 if ALIS is chosen)


        Return value:
        ------------

        Returns a MLUT object containing:
            - the polarized dimensionless reflectance (I,Q,U,V) at the
              different layers
            - the number of photons (N) received at each layer
            - the profiles and phase functions
            - attributes

        Example:
            M = Smartg().run(wl=400., NBPHOTONS=1e7, atm=Profile('afglt'))
            M['I_up (TOA)'][:,:] contains the top of atmosphere reflectance
        '''

        #
        # initialization
        #

        if NBPHI%2 == 1:
            warn('Odd number of azimuth')

        if NBLOOP is None:
            NBLOOP = min(NBPHOTONS/30, 1e8)
        NF = int(NF)

        # number of output levels
        # warning! values defined in communs.h should be < LVL
        NLVL = 6
        #NLVL = 5

        # number of Stokes parameters of the radiation field
        NPSTK = 4

        t0 = datetime.now()

        attrs = OrderedDict()
        attrs.update({'processing started at': t0})
        attrs.update({'VZA': THVDEG})
        attrs.update({'MODE': {True: 'PPA', False: 'SSA'}[self.pp]})
        attrs.update({'XBLOCK': XBLOCK})
        attrs.update({'XGRID': XGRID})
        attrs.update({'NPHOTONS': '{:g}'.format(NBPHOTONS)})


        if not isinstance(wl, BandSet):
            wl = BandSet(wl)
        NLAM = wl.size

        NLOW=-1
        if self.alis is not None :
            BEER=1
            if (self.alis['nlow'] ==-1) : NLOW=NLAM
            else: NLOW=self.alis['nlow']

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

        #
        # atmosphere
        #
        if isinstance(atm, Atmosphere):
            prof_atm = atm.calc(wl)
        else:
            prof_atm = atm

        if prof_atm is not None:
            faer = calculF(prof_atm, NF, DEPO, kind='atm')
            prof_atm_gpu = init_profile(wl, prof_atm, 'atm')
            NATM = len(prof_atm.axis('z_atm')) - 1
        else:
            faer = gpuzeros(1, dtype='float32')
            prof_atm_gpu = to_gpu(np.zeros(1, dtype=type_Profile))
            NATM = 0


        # computation of the impact point
        X0, tabTransDir = impactInit(prof_atm, NLAM, THVDEG, RTER, self.pp)

        #
        # surface
        #
        if surf is None:
            # default surface parameters
            surf = FlatSurface()

        #
        # ocean
        #
        if isinstance(water, IOP_base):
            prof_oc = water.calc(wl)
        else:
            prof_oc = water

        if prof_oc is not None:
            foce = calculF(prof_oc, NF, DEPO_WATER, kind='oc')
            prof_oc_gpu = init_profile(wl, prof_oc, 'oc')
            NOCE = len(prof_oc.axis('z_oc')) - 1
        else:
            foce = gpuzeros(1, dtype='float32')
            prof_oc_gpu = to_gpu(np.zeros(1, dtype=type_Profile))
            NOCE = 0

        #
        # albedo and adjacency effect
        #
        spectrum = np.zeros(NLAM, dtype=type_Spectrum)
        spectrum['lambda'] = wl[:]
        if env is None:
            # default values (no environment effect)
            env = Environment()
            if 'SURFALB' in surf.dict:
                spectrum['alb_surface'] = surf.dict['SURFALB']
            else:
                spectrum['alb_surface'] = -999.
        else:
            assert surf is not None
            spectrum['alb_surface'] = env.alb.get(wl[:])

        if water is None:
            spectrum['alb_seafloor'] = -999.
        else:
            spectrum['alb_seafloor'] = prof_oc['albedo_seafloor'].data[...]
        spectrum = to_gpu(spectrum)

        # Local Estimate option
        LE = 0
        if le is not None:
            LE = 1
            if not 'th' in le:
                le['th'] = np.array(le['th_deg'], dtype='float32').ravel() * np.pi/180.
            else:
                le['th'] = np.array(le['th'], dtype='float32').ravel()
            if not 'phi' in le:
                le['phi'] = np.array(le['phi_deg'], dtype='float32').ravel() * np.pi/180.
            else:
                le['phi'] = np.array(le['phi'], dtype='float32').ravel()
            NBTHETA =  le['th'].shape[0]
            NBPHI   =  le['phi'].shape[0]

        FLUX = 0
        if flux is not None:
            LE=0
            if flux== 'spherical' : 
                FLUX = 1

        if wl_proba is not None:
            assert wl_proba.dtype == 'int64'
            wl_proba_icdf = to_gpu(wl_proba)
            NWLPROBA = len(wl_proba_icdf)
        else:
            wl_proba_icdf = gpuzeros(1, dtype='int64')
            NWLPROBA = 0

        # initialization of the constants
        InitConst(surf, env, NATM, NOCE, self.mod,
                       NBPHOTONS, NBLOOP, THVDEG, DEPO,
                       XBLOCK, XGRID, NLAM, SIM, NF,
                       NBTHETA, NBPHI, OUTPUT_LAYERS,
                       RTER, LE, FLUX, NLVL, NPSTK,
                       NWLPROBA, BEER, NLOW)


        # Initialize the progress bar
        p = Progress(NBPHOTONS, progress)

        # Initialize the RNG
        SEED = self.rng.setup(SEED, XBLOCK, XGRID)

        # Loop and kernel call
        (NPhotonsInTot,
                tabPhotonsTot, errorcount, NPhotonsOutTot, sigma, Nkernel, secs_cuda_clock
                ) = loop_kernel(NBPHOTONS, faer, foce,
                                NLVL, NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                                NLAM, self.double, self.kernel, p, X0, le, spectrum,
                                prof_atm_gpu, prof_oc_gpu,
                                wl_proba_icdf, stdev, self.rng)
        attrs['kernel time (s)'] = secs_cuda_clock
        attrs['number of kernel iterations'] = Nkernel
        attrs['seed'] = SEED
        attrs.update(self.common_attrs)

        # finalization
        output = finalize(tabPhotonsTot, wl[:], NPhotonsInTot, errorcount, NPhotonsOutTot,
                           OUTPUT_LAYERS, tabTransDir, SIM, attrs, prof_atm, prof_oc,
                           sigma, le=le, flux=flux)
        output.set_attr('processing time (s)', (datetime.now() - t0).total_seconds())

        p.finish('Done! | Received {:.1%} of {:.3g} photons ({:.1%})'.format(
            np.sum(NPhotonsOutTot[0,...])/float(np.sum(NPhotonsInTot)),
            np.sum(NPhotonsInTot),
            np.sum(NPhotonsInTot)/float(NBPHOTONS),
            ))

        if wl.scalar:
            output = output.dropaxis('wavelength')
            output.attrs['wavelength'] = wl[:]

        return output


def calcOmega(NBTHETA, NBPHI):
    '''
    returns the zenith and azimuth angles, and the solid angles
    '''

    # zenith angles
    dth = (np.pi/2)/NBTHETA
    tabTh = np.linspace(dth/2, np.pi/2-dth/2, NBTHETA, dtype='float64')

    # azimuth angles
    #dphi = np.pi/NBPHI
    #tabPhi = np.linspace(dphi/2, np.pi-dphi/2, NBPHI, dtype='float64')
    dphi = 2*np.pi/NBPHI
    tabPhi = np.linspace(0., 2*np.pi-dphi, NBPHI, dtype='float64')


    # solid angles
    tabds = np.sin(tabTh) * dth * dphi

    # normalize to 1
    tabOmega = tabds/(sum(tabds)*NBPHI)

    return tabTh, tabPhi, tabOmega


def finalize(tabPhotonsTot, wl, NPhotonsInTot, errorcount, NPhotonsOutTot,
             OUTPUT_LAYERS, tabTransDir, SIM, attrs, prof_atm, prof_oc,
             sigma, le=None, flux=None):
    '''
    create and return the final output
    '''
    (NLVL,NPSTK,NLAM,NBTHETA,NBPHI) = tabPhotonsTot.shape

    # normalization in case of radiance
    # (broadcast everything to dimensions (LVL,NPSTK,LAM,THETA,PHI))
    norm_npho = NPhotonsInTot.reshape((1,1,-1,1,1))
    if flux is None:
        if le!=None : 
            tabTh = le['th']
            tabPhi = le['phi']
            norm_geo =  np.cos(tabTh).reshape((1,1,1,-1,1))
        else : 
            tabTh, tabPhi, tabOmega = calcOmega(NBTHETA, NBPHI )
            norm_geo = 2.0 * tabOmega.reshape((1,1,-1,1)) * np.cos(tabTh).reshape((1,1,-1,1))
    else:
        norm_geo = 1.
        tabTh, tabPhi, _ = calcOmega(NBTHETA, NBPHI)

    # normalization
    tabFinal = tabPhotonsTot.astype('float64')/(norm_geo*norm_npho)

    # swapaxes : (th, phi) -> (phi, theta)
    tabFinal = tabFinal.swapaxes(3,4)
    NPhotonsOutTot = NPhotonsOutTot.swapaxes(2,3)
    if sigma is not None:
        sigma /= norm_geo
        sigma = sigma.swapaxes(3,4)


    #
    # create the MLUT object
    #
    m = MLUT()

    # add the axes
    axnames = ['Azimuth angles', 'Zenith angles']
    m.add_axis('Zenith angles', tabTh*180./np.pi)
    m.add_axis('Azimuth angles', tabPhi*180./np.pi)
    if NLAM > 1:
        m.add_axis('wavelength', wl)
        ilam = slice(None)
        axnames.insert(0, 'wavelength')
    else:
        m.set_attr('wavelength', str(wl))
        ilam = 0

    m.add_dataset('I_up (TOA)', tabFinal[UPTOA,0,ilam,:,:], axnames)
    m.add_dataset('Q_up (TOA)', tabFinal[UPTOA,1,ilam,:,:], axnames)
    m.add_dataset('U_up (TOA)', tabFinal[UPTOA,2,ilam,:,:], axnames)
    m.add_dataset('V_up (TOA)', tabFinal[UPTOA,3,ilam,:,:], axnames)
    if sigma is not None:
        m.add_dataset('I_stdev_up (TOA)', sigma[UPTOA,0,ilam,:,:], axnames)
        m.add_dataset('Q_stdev_up (TOA)', sigma[UPTOA,1,ilam,:,:], axnames)
        m.add_dataset('U_stdev_up (TOA)', sigma[UPTOA,2,ilam,:,:], axnames)
        m.add_dataset('V_stdev_up (TOA)', sigma[UPTOA,3,ilam,:,:], axnames)
    m.add_dataset('N_up (TOA)', NPhotonsOutTot[UPTOA,ilam,:,:], axnames)

    if OUTPUT_LAYERS & 1:
        m.add_dataset('I_down (0+)', tabFinal[DOWN0P,0,ilam,:,:], axnames)
        m.add_dataset('Q_down (0+)', tabFinal[DOWN0P,1,ilam,:,:], axnames)
        m.add_dataset('U_down (0+)', tabFinal[DOWN0P,2,ilam,:,:], axnames)
        m.add_dataset('V_down (0+)', tabFinal[DOWN0P,3,ilam,:,:], axnames)
        if sigma is not None:
            m.add_dataset('I_stdev_down (0+)', sigma[DOWN0P,0,ilam,:,:], axnames)
            m.add_dataset('Q_stdev_down (0+)', sigma[DOWN0P,1,ilam,:,:], axnames)
            m.add_dataset('U_stdev_down (0+)', sigma[DOWN0P,2,ilam,:,:], axnames)
            m.add_dataset('V_stdev_down (0+)', sigma[DOWN0P,3,ilam,:,:], axnames)
        m.add_dataset('N_down (0+)', NPhotonsOutTot[DOWN0P,ilam,:,:], axnames)

        m.add_dataset('I_up (0-)', tabFinal[UP0M,0,ilam,:,:], axnames)
        m.add_dataset('Q_up (0-)', tabFinal[UP0M,1,ilam,:,:], axnames)
        m.add_dataset('U_up (0-)', tabFinal[UP0M,2,ilam,:,:], axnames)
        m.add_dataset('V_up (0-)', tabFinal[UP0M,3,ilam,:,:], axnames)
        if sigma is not None:
            m.add_dataset('I_stdev_up (0-)', sigma[UP0M,0,ilam,:,:], axnames)
            m.add_dataset('Q_stdev_up (0-)', sigma[UP0M,1,ilam,:,:], axnames)
            m.add_dataset('U_stdev_up (0-)', sigma[UP0M,2,ilam,:,:], axnames)
            m.add_dataset('V_stdev_up (0-)', sigma[UP0M,3,ilam,:,:], axnames)
        m.add_dataset('N_up (0-)', NPhotonsOutTot[UP0M,ilam,:,:], axnames)

    if OUTPUT_LAYERS & 2:
        m.add_dataset('I_down (0-)', tabFinal[DOWN0M,0,ilam,:,:], axnames)
        m.add_dataset('Q_down (0-)', tabFinal[DOWN0M,1,ilam,:,:], axnames)
        m.add_dataset('U_down (0-)', tabFinal[DOWN0M,2,ilam,:,:], axnames)
        m.add_dataset('V_down (0-)', tabFinal[DOWN0M,3,ilam,:,:], axnames)
        if sigma is not None:
            m.add_dataset('I_stdev_down (0-)', sigma[DOWN0M,0,ilam,:,:], axnames)
            m.add_dataset('Q_stdev_down (0-)', sigma[DOWN0M,1,ilam,:,:], axnames)
            m.add_dataset('U_stdev_down (0-)', sigma[DOWN0M,2,ilam,:,:], axnames)
            m.add_dataset('V_stdev_down (0-)', sigma[DOWN0M,3,ilam,:,:], axnames)
        m.add_dataset('N_down (0-)', NPhotonsOutTot[DOWN0M,ilam,:,:], axnames)

        m.add_dataset('I_up (0+)', tabFinal[UP0P,0,ilam,:,:], axnames)
        m.add_dataset('Q_up (0+)', tabFinal[UP0P,1,ilam,:,:], axnames)
        m.add_dataset('U_up (0+)', tabFinal[UP0P,2,ilam,:,:], axnames)
        m.add_dataset('V_up (0+)', tabFinal[UP0P,3,ilam,:,:], axnames)
        if sigma is not None:
            m.add_dataset('I_stdev_up (0+)', sigma[UP0P,0,ilam,:,:], axnames)
            m.add_dataset('Q_stdev_up (0+)', sigma[UP0P,1,ilam,:,:], axnames)
            m.add_dataset('U_stdev_up (0+)', sigma[UP0P,2,ilam,:,:], axnames)
            m.add_dataset('V_stdev_up (0+)', sigma[UP0P,3,ilam,:,:], axnames)
        m.add_dataset('N_up (0+)', NPhotonsOutTot[UP0P,ilam,:,:], axnames)

        m.add_dataset('I_down (B)', tabFinal[DOWNB,0,ilam,:,:], axnames)
        m.add_dataset('Q_down (B)', tabFinal[DOWNB,1,ilam,:,:], axnames)
        m.add_dataset('U_down (B)', tabFinal[DOWNB,2,ilam,:,:], axnames)
        m.add_dataset('V_down (B)', tabFinal[DOWNB,3,ilam,:,:], axnames)
        if sigma is not None:
            m.add_dataset('I_stdev_down (B)', sigma[DOWNB,0,ilam,:,:], axnames)
            m.add_dataset('Q_stdev_down (B)', sigma[DOWNB,1,ilam,:,:], axnames)
            m.add_dataset('U_stdev_down (B)', sigma[DOWNB,2,ilam,:,:], axnames)
            m.add_dataset('V_stdev_down (B)', sigma[DOWNB,3,ilam,:,:], axnames)
        m.add_dataset('N_down (B)', NPhotonsOutTot[DOWNB,ilam,:,:], axnames)


    # direct transmission
    m.add_dataset('direct transmission', tabTransDir,
                   axnames=['wavelength'])

    # write atmospheric profiles
    if prof_atm is not None:
        m.add_lut(prof_atm['OD_atm'])
        m.add_lut(prof_atm['OD_sca_atm'])
        m.add_lut(prof_atm['OD_abs_atm'])
        m.add_lut(prof_atm['pmol_atm'])
        m.add_lut(prof_atm['ssa_atm'])
        m.add_lut(prof_atm['ssa_p_atm'])
        if 'phase_atm' in prof_atm.datasets():
            m.add_lut(prof_atm['phase_atm'])
            m.add_lut(prof_atm['iphase_atm'])

    # write ocean profiles
    if prof_oc is not None:
        m.add_lut(prof_oc['OD_oc'])
        m.add_lut(prof_oc['OD_sca_oc'])
        m.add_lut(prof_oc['OD_abs_oc'])
        m.add_lut(prof_oc['pmol_oc'])
        m.add_lut(prof_oc['ssa_oc'])
        if 'ssa_p_oc' in prof_oc.datasets():
            m.add_lut(prof_oc['ssa_p_oc'])
        if 'phase_oc' in prof_oc.datasets():
            m.add_lut(prof_oc['phase_oc'])
            m.add_lut(prof_oc['iphase_oc'])
        m.add_lut(prof_oc['albedo_seafloor'])

    # write the error count
    err = errorcount.get()
    for i, d in enumerate([
            'ERROR_THETA',
            'ERROR_CASE',
            'ERROR_VXY',
            'ERROR_MAX_LOOP',
            ]):
        m.set_attr(d, err[i])

    # write attributes
    for k, v in attrs.items():
        m.set_attr(k, str(v))

    # fluxes post-processing
    if flux is not None:
        m.set_attr('flux', flux)
        for d in m.datasets():
            if (('_stdev_' in d)
                    or (d.startswith('Q_'))
                    or (d.startswith('U_'))
                    or (d.startswith('V_'))
                    ):
                m.rm_lut(d)
            elif d.startswith('I_') or d.startswith('N_'):
                l = m[d].reduce(np.sum, 'Azimuth angles').reduce(np.sum, 'Zenith angles', as_lut=True)
                m.rm_lut(d)
                m.add_lut(l, desc=d.replace('I_', 'flux_'))

    return m


def rayleigh(N, DEPO):
    '''
    Rayleigh phase function, incl. cumulative
    over N angles
    DEPO: depolarization coefficient
    '''
    pha = np.zeros(N, dtype=type_Phase, order='C')

    GAMA = DEPO / (2- DEPO)
    DELTA = np.float32((1.0 - GAMA) / (1.0 + 2.0 *GAMA));
    DELTA_PRIM = np.float32(GAMA / (1.0 + 2.0*GAMA));
    DELTA_SECO = np.float32((1.0 - 3.0*GAMA) / (1.0 - GAMA));
    BETA  = np.float32(3./2. * DELTA_PRIM);
    ALPHA = np.float32(1./8. * DELTA);
    A = np.float32(1. + BETA / (3.0 * ALPHA));

    i = np.arange(int(N), dtype=np.float32)
    thetaLE = np.linspace(0., pi, int(N), endpoint=True, dtype=np.float64)
    b = ((i/(N-1)) - 4.0*ALPHA - BETA) / (2.0*ALPHA)
    u = (-b + (A**3.0 + b**2.0)**(1.0/2.0))**(1.0/3.0)
    cTh = u - (A/u)
    cTh = np.clip(cTh, -1, 1)
    cTh2 = cTh*cTh
    theta = np.arccos(cTh)
    cThLE = np.cos(thetaLE)
    cTh2LE = cThLE*cThLE
    T_demi = (3.0/2.0)
    P11 = T_demi*(DELTA+DELTA_PRIM)
    P12 = T_demi*DELTA_PRIM
    P33bis = T_demi*DELTA
    P44bis = P33bis*DELTA_SECO

    # parameters equally spaced in scattering probabiliy [0, 1]
    pha['p_P11'][:] = P11
    pha['p_P12'][:] = P12
    pha['p_P22'][:] = T_demi*(DELTA*cTh2[:] + DELTA_PRIM)
    pha['p_P33'][:] = P33bis*cTh[:] # U
    pha['p_P44'][:] = P44bis*cTh[:] # V
    pha['p_ang'][:] = theta[:] # angle

    # parameters equally spaced in scattering angle [0, 180]
    pha['a_P11'][:] = P11
    pha['a_P12'][:] = P12
    pha['a_P22'][:] = T_demi*(DELTA*cTh2LE[:] + DELTA_PRIM) 
    pha['a_P33'][:] = P33bis*cThLE[:]  # U
    pha['a_P44'][:] = P44bis*cThLE[:]  # V

    return pha


def calculF(profile, N, DEPO, kind):
    '''
    Calculate cumulated phase functions from profile
    N: number of angles
    DEPO: depolarization factor 'atmospheric'
    kind: 'atm' or 'oc'
    ray (boolean): include rayleigh phase function
    '''

    name_phase = 'phase_{}'.format(kind)
    if name_phase in profile.datasets():
        nphases = profile[name_phase].shape[0]
    else:
        nphases = 0

    nphases += 1   # include Rayleigh phase function

    # Initialize the cumulative distribution function
    if nphases > 0:
        shp = (nphases, N)
    else:
        shp = (1, N)
    phase_H = np.zeros(shp, dtype=type_Phase, order='C')

    # Set Rayleigh phase function
    phase_H[0,:] = rayleigh(N, DEPO)
    if 'theta_'+kind in profile.axes:
        angles = profile.axis('theta_'+kind) * pi/180.
        assert angles[-1] < 3.15   # assert that angles are in radians
        dtheta = np.diff(angles)

    idx = 1
    for ipha in range(nphases-1):

        phase = profile[name_phase][ipha, :, :]  # ipha, stk, theta

        scum = [0]
        pm = phase[1, :] + phase[0, :]
        sin = np.sin(angles)
        tmp = dtheta * ((sin[:-1] * pm[:-1] + sin[1:] * pm[1:]) / 3.
                        + (sin[:-1] * pm[1:] + sin[1:] * pm[:-1])/6.) * np.pi * 2.
        scum = np.append(scum,tmp)
        scum = np.cumsum(scum)
        scum /= scum[-1]

        # probability between 0 and 1
        z = (np.arange(N, dtype='float64')+1)/N
        angN = (np.arange(N, dtype='float64'))/(N-1)*np.pi
        f1 = interp1d(angles, phase[1,:])
        f2 = interp1d(angles, phase[0,:])
        f3 = interp1d(angles, phase[2,:])
        f4 = interp1d(angles, phase[3,:])

        # parameters equally spaced in scattering probability
        phase_H['p_P11'][idx, :] = interp1d(scum, phase[1,:])(z)  # I par P11
        phase_H['p_P22'][idx, :] = interp1d(scum, phase[0,:])(z)  # I per P22
        phase_H['p_P33'][idx, :] = interp1d(scum, phase[2,:])(z)  # U P33
        phase_H['p_P43'][idx, :] = interp1d(scum, phase[3,:])(z)  # V P43
        phase_H['p_P44'][idx, :] = interp1d(scum, phase[2,:])(z)  # V P44= P33
        phase_H['p_ang'][idx, :] = interp1d(scum, angles)(z) # angle

        # parameters equally spaced in scattering angle [0, 180]
        phase_H['a_P11'][idx, :] = f1(angN)  # I par P11
        phase_H['a_P22'][idx, :] = f2(angN)  # I per P22
        phase_H['a_P33'][idx, :] = f3(angN)  # U P33
        phase_H['a_P43'][idx, :] = f4(angN)  # V P43
        phase_H['a_P44'][idx, :] = f3(angN)  # V P44=P33

        idx += 1

    return to_gpu(phase_H)


def InitConst(surf, env, NATM, NOCE, mod,
                   NBPHOTONS, NBLOOP, THVDEG, DEPO,
                   XBLOCK, XGRID,NLAM, SIM, NF,
                   NBTHETA, NBPHI, OUTPUT_LAYERS,
                   RTER, LE, FLUX, NLVL, NPSTK, NWLPROBA, BEER, NLOW) :

    """
    Initialize the constants in python and send them to the device memory

    Arguments:

        - D: Dictionary containing all the parameters required to launch the simulation by the kernel
        - surf : Surface object
        - env : environment effect parameters (dictionary)
        - NATM : Number of layers of the atmosphere
        - NOCE : Number of layers of the ocean
        - HATM : Altitude of the Top of Atmosphere
        - mod : PyCUDA module compiling the kernel
    """

    # compute some needed constants
    THV = THVDEG * np.pi/180.
    STHV = np.sin(THV)
    CTHV = np.cos(THV)

    def copy_to_device(name, scalar, dtype):
        cuda.memcpy_htod(mod.get_global(name)[0], np.array([scalar], dtype=dtype))

    # copy constants to device
    copy_to_device('NBLOOPd', NBLOOP, np.uint32)
    copy_to_device('NOCEd', NOCE, np.int32)
    copy_to_device('OUTPUT_LAYERSd', OUTPUT_LAYERS, np.uint32)
    copy_to_device('NF', NF, np.uint32)
    copy_to_device('NATMd', NATM, np.int32)
    copy_to_device('XBLOCKd', XBLOCK, np.int32)
    copy_to_device('YBLOCKd', 1, np.int32)
    copy_to_device('XGRIDd', XGRID, np.int32)
    copy_to_device('YGRIDd', 1, np.int32)
    copy_to_device('NBTHETAd', NBTHETA, np.int32)
    copy_to_device('NBPHId', NBPHI, np.int32)
    copy_to_device('NLAMd', NLAM, np.int32)
    copy_to_device('SIMd', SIM, np.int32)
    copy_to_device('LEd', LE, np.int32)
    copy_to_device('FLUXd', FLUX, np.int32)
    copy_to_device('NLVLd', NLVL, np.int32)
    copy_to_device('NPSTKd', NPSTK, np.int32)
    copy_to_device('BEERd', BEER, np.int32)
    copy_to_device('NLOWd', NLOW, np.int32)
    if surf != None:
        copy_to_device('SURd', surf.dict['SUR'], np.int32)
        copy_to_device('DIOPTREd', surf.dict['DIOPTRE'], np.int32)
        copy_to_device('WINDSPEEDd', surf.dict['WINDSPEED'], np.float32)
        copy_to_device('NH2Od', surf.dict['NH2O'], np.float32)
    if env != None:
        copy_to_device('ENVd', env.dict['ENV'], np.int32)
        copy_to_device('ENV_SIZEd', env.dict['ENV_SIZE'], np.float32)
        copy_to_device('X0d', env.dict['X0'], np.float32)
        copy_to_device('Y0d', env.dict['Y0'], np.float32)
    copy_to_device('STHVd', STHV, np.float32)
    copy_to_device('CTHVd', CTHV, np.float32)
    copy_to_device('RTER', RTER, np.float32)
    copy_to_device('NWLPROBA', NWLPROBA, np.int32)


def init_profile(wl, prof, kind):
    '''
    take the profile as a MLUT, and setup the gpu structure

    kind = 'atm' or 'oc' for atmosphere or ocean
    '''

    # reformat to smartg format

    NATM = len(prof.axis('z_'+kind)) - 1
    shp = (len(wl), NATM+1)
    prof_gpu = np.zeros(shp, dtype=type_Profile, order='C')
    prof_gpu['z'][0,:] = prof.axis('z_'+kind)
    prof_gpu['z'][1:,:] = -999.      # other wavelengths are NaN

    prof_gpu['OD'][:,:] = prof['OD_'+kind].data[...]
    prof_gpu['OD_sca'][:] = prof['OD_sca_'+kind].data[...]
    prof_gpu['OD_abs'][:] = prof['OD_abs_'+kind].data[...]
    prof_gpu['pmol'][:] = prof['pmol_'+kind].data[...]
    prof_gpu['ssa'][:] = prof['ssa_'+kind].data[...]
    if 'iphase_'+kind in prof.datasets():
        prof_gpu['iphase'][:] = prof['iphase_'+kind].data[...]

    return to_gpu(prof_gpu)


def get_git_attrs():
    R = {}

    # check current commit
    p = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    if p.wait():
        return {}
    else:
        shasum = p.communicate()[0].strip()
        R.update({'git_commit_ref': shasum})

    # check if repo is dirty
    p = subprocess.Popen(['git', 'status', '--porcelain',
                          '--untracked-files=no'],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if p.wait():
        return {}
    else:
        is_dirty = len(p.communicate()[0]) != 0
        R.update({'git_dirty_repo': int(is_dirty)})
    return R


def loop_kernel(NBPHOTONS, faer, foce, NLVL,
                NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                NLAM, double, kern, p, X0, le, spectrum,
                prof_atm, prof_oc, wl_proba_icdf, stdev, rng):
    """
    launch the kernel several time until the targeted number of photons injected is reached

    Arguments:
        - NBPHOTONS : Number of photons injected
        - Tableau : Class containing the arrays sent to the device
        - NLVL : Number of output levels
        - NPSTK : Number of Stokes parameters + 1 for number of photons
        - BLOCK : Block dimension
        - XGRID : Grid dimension
        - NBTHETA : Number of intervals in zenith
        - NLAM : Number of wavelengths
        - options : compilation options
        - kern : kernel launching the transfert radiative
        - p: progress bar object
        - X0: initial coordinates of the photon entering the atmosphere
    --------------------------------------------------------------
    Returns :
        - nbPhotonsTot : Total number of photons processed
        - NPhotonsInTot : Total number of photons processed by interval
        - nbPhotonsSorTot : Total number of outgoing photons
        - tabPhotonsTot : Total weight of all outgoing photons

    """
    # Initializations
    nThreadsActive = gpuzeros(1, dtype='int32')
    Counter = gpuzeros(1, dtype='uint64')

    # Initialize the array for error counting
    NERROR = 32
    errorcount = gpuzeros(NERROR, dtype='uint64')

    # Initialize of the parameters
    tabPhotonsTot = gpuzeros((NLVL,NPSTK,NLAM,NBTHETA,NBPHI), dtype=np.float64)
    N_simu = 0
    if stdev:
        # to calculate the standard deviation of the result, we accumulate the
        # parameters and their squares
        # finally we extrapolate in 1/sqrt(N_simu)
        sum_x = 0.
        sum_x2 = 0.

    # arrays for counting the input photons (per wavelength)
    NPhotonsIn = gpuzeros(NLAM, dtype=np.uint64)
    NPhotonsInTot = gpuzeros(NLAM, dtype=np.uint64)

    # arrays for counting the output photons
    NPhotonsOut = gpuzeros((NLVL,NLAM,NBTHETA,NBPHI), dtype=np.uint64)
    NPhotonsOutTot = gpuzeros((NLVL,NLAM,NBTHETA,NBPHI), dtype=np.uint64)

    if double:
        tabPhotons = gpuzeros((NLVL,NPSTK,NLAM,NBTHETA,NBPHI), dtype=np.float64)
    else:
        tabPhotons = gpuzeros((NLVL,NPSTK,NLAM,NBTHETA,NBPHI), dtype=np.float32)

    # local estimates angles
    if le != None:
        tabthv = to_gpu(le['th'].astype('float32'))
        tabphi = to_gpu(le['phi'].astype('float32'))
    else:
        tabthv = gpuzeros(1, dtype='float32')
        tabphi = gpuzeros(1, dtype='float32')

    secs_cuda_clock = 0.
    while(np.sum(NPhotonsInTot.get()) < NBPHOTONS):

        tabPhotons.fill(0.)
        NPhotonsOut.fill(0)
        NPhotonsIn.fill(0)
        Counter.fill(0)

        start_cuda_clock = cuda.Event()
        end_cuda_clock = cuda.Event()
        start_cuda_clock.record()

        # kernel launch
        kern(spectrum, X0, faer, foce,
                errorcount, nThreadsActive, tabPhotons,
                Counter, NPhotonsIn, NPhotonsOut, tabthv, tabphi,
                prof_atm, prof_oc, wl_proba_icdf, rng.state,
                block=(XBLOCK, 1, 1), grid=(XGRID, 1, 1))
        end_cuda_clock.record()
        end_cuda_clock.synchronize()
        secs_cuda_clock = secs_cuda_clock + start_cuda_clock.time_till(end_cuda_clock)

        cuda.Context.synchronize()

        L = NPhotonsIn   # number of photons launched by last kernel
        NPhotonsInTot += L

        NPhotonsOutTot += NPhotonsOut
        S = tabPhotons   # sum of weights for the last kernel
        tabPhotonsTot += S

        N_simu += 1
        if stdev:
            L = L.reshape((1,1,-1,1,1))   # broadcast to tabPhotonsTot
            warn('stdev is activated: it is known to slow down the code considerably.')
            SoverL = S.get()/L.get()
            sum_x += SoverL
            sum_x2 += (SoverL)**2

        # update of the progression Bar
        sphot = np.sum(NPhotonsInTot.get())
        p.update(sphot,
                'Launched {:.3g} photons'.format(sphot))
    secs_cuda_clock = secs_cuda_clock*1e-3

    if stdev:
        # finalize the calculation of the standard deviation
        sigma = np.sqrt(sum_x2/N_simu - (sum_x/N_simu)**2)

        # extrapolate in 1/sqrt(N_simu)
        sigma /= np.sqrt(N_simu)
    else:
        sigma = None

    return NPhotonsInTot.get(), tabPhotonsTot.get(), errorcount, NPhotonsOutTot.get(), sigma, N_simu, secs_cuda_clock


def impactInit(prof_atm, NLAM, THVDEG, Rter, pp):
    '''
    Calculate the coordinates of the entry point in the atmosphere
    and direct transmission of the atmosphere

    Returns :
        - [x0, y0, z0] : cartesian coordinates
    '''
    if prof_atm is None:
        Hatm = 0.
        NATM = 0
    else:
        Zatm = prof_atm.axis('z_atm')
        Hatm = Zatm[0]
        NATM = len(Zatm)-1

    vx = -np.sin(THVDEG * np.pi / 180)
    vy = 0.
    vz = -np.cos(THVDEG * np.pi / 180)
    Rter = np.double(Rter)

    tautot = np.zeros(NLAM, dtype=np.float64)

    if pp:
        z0 = Hatm
        x0 = Hatm*np.tan(THVDEG*np.pi/180.)
        y0 = 0.

        if NATM != 0:
            for ilam in range(NLAM):
                if prof_atm['OD_atm'].ndim == 2:
                    # lam, z
                    tautot[ilam] = prof_atm['OD_atm'][ilam, NATM]/np.cos(THVDEG*pi/180.)
                elif prof_atm['OD_atm'].ndim == 1:
                    # z
                    tautot[ilam] = prof_atm['OD_atm'][NATM]/np.cos(THVDEG*pi/180.)
                else:
                    raise Exception('invalid number of dimensions in prof_atm')
    else:
        tanthv = np.tan(THVDEG*np.pi/180.)

        # Pythagorean theorem in right triangle OMZ, where:
        # * O is the center of the earth
        # * M is the entry point in the atmosphere, has cartesian coordinates (x0, y0, Rter+z0)
        #     (origin is at the surface)
        # * Z is the projection of M on z axis
        # tan(thv) = x0/z0
        # Rter is the radius of the earth and Hatm the thickness of the atmosphere
        # solve the equation x0^2 + (Rter+z0)^2 = (Rter+Hatm)^2 for z0
        delta = 4*Rter**2 + 4*(tanthv**2 + 1) * (Hatm**2 + 2*Hatm*Rter)
        z0 = (-2.*Rter + np.sqrt(delta))/(2 *(tanthv**2 + 1.))
        x0 = z0*tanthv
        y0 = 0.
        z0 += Rter

        # loop over the NATM atmosphere layers to find the total optical thickness
        xph = x0
        yph = y0
        zph = z0
        for i in range(1, NATM+1):
            # V is the direction vector, X is the position vector, D is the
            # distance to the next layer and R is the position vector at the
            # next layer
            # we have: R = X + V.D
            # R² = X² + (V.D)² + 2XVD
            # where R is Rter+ALT[i]
            # solve for D:
            delta = 4.*(vx*xph + vy*yph + vz*zph)**2 - 4*((xph**2 + yph**2 + zph**2) - (Rter + Zatm[i])**2)

            # the 2 solutions are:
            D1 = 0.5 * (-2. * (vx*xph+vy*yph+vz*zph) + np.sqrt(delta))
            D2 = 0.5 * (-2. * (vx*xph+vy*yph+vz*zph) - np.sqrt(delta))

            # the solution is the smallest positive one
            if D1 > 0:
                if D2 > 0:
                    D = min(D1, D2)
                else:
                    D = D1
            else:
                if D2 > 0:
                    D = D2
                else:
                    raise Exception('No solution in impactInit')

            # photon moves forward
            xph += vx * D
            yph += vy * D
            zph += vz * D

            for ilam in range(NLAM):
                # optical thickness of the layer in vertical direction
                hlay0 = abs(prof_atm['OD_atm'][ilam, i] - prof_atm['OD_atm'][ilam, i - 1])

                # thickness of the layer
                D0 = abs(Zatm[i-1] - Zatm[i])

                # optical thickness of the layer at current wavelength
                hlay = hlay0*D/D0

                # cumulative optical thickness
                tautot[ilam] += hlay

    return to_gpu(np.array([x0, y0, z0], dtype='float32')), np.exp(-tautot)


def init_rng(rng):
    if rng == 'PHILOX':
        return RNG_PHILOX()
    elif rng == 'CURAND_PHILOX':
        return RNG_CURAND_PHILOX()
    else:
        raise Exception('Invalid RNG "{}"'.format(rng))


class RNG_PHILOX(object):
    def __init__(self):
        pass

    def setup(self, SEED, XBLOCK, XGRID):
        if SEED == -1:
            # SEED is based on clock
            SEED = np.uint32((datetime.now()
                - datetime.utcfromtimestamp(0)).total_seconds()*1000)

        state = np.zeros(XBLOCK*XGRID+1, dtype='uint32')
        state[0] = SEED
        self.state = to_gpu(state)

        return SEED


class RNG_CURAND_PHILOX(object):
    def __init__(self):
        # build module containing initilization functions
        source = r'''
        #include <curand.h>
        #include <curand_kernel.h>

        #define YBLOCKd 1
        #define YGRIDd 1

        __device__ __constant__ int XBLOCKd;
        __device__ __constant__ int XGRIDd;
        __device__ __constant__ int SEEDd;

        extern "C" {
        __global__ void get_state_size(int *s) {
            *s = sizeof(curandStatePhilox4_32_10_t);
        }

        __global__ void setup(curandStatePhilox4_32_10_t *state) {
            int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);
            curand_init(SEEDd, idx, 0, &state[idx]);
        }
        }
        '''
        self.mod = SourceModule(source, no_extern_c=True)

        # get state size
        s = gpuzeros(1, dtype='int32')
        self.mod.get_function('get_state_size')(s, block=(1, 1, 1), grid=(1, 1, 1))
        self.STATE_SIZE = int(s.get())  # size in bytes

    def setup(self, SEED, XBLOCK, XGRID):
        if SEED == -1:
            # SEED is based on clock
            SEED = np.uint32((datetime.now()
                - datetime.utcfromtimestamp(0)).total_seconds()*1000)

        cuda.memcpy_htod(self.mod.get_global('XBLOCKd')[0], np.array([XBLOCK], dtype=np.int32))
        cuda.memcpy_htod(self.mod.get_global('XGRIDd')[0], np.array([XGRID], dtype=np.int32))
        cuda.memcpy_htod(self.mod.get_global('SEEDd')[0], np.array([XGRID], dtype=np.int32))

        # setup RNG
        self.state = gpuzeros(self.STATE_SIZE*XBLOCK*XGRID, dtype='uint8')
        setup = self.mod.get_function('setup')
        setup(self.state, block=(XBLOCK,1,1), grid=(XGRID, 1, 1))

        return SEED
