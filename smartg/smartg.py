#!/usr/bin/env python
# encoding: utf-8


'''
SMART-G
Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU
'''


import os
import numpy as np
from datetime import datetime
from numpy import pi
from smartg.atmosphere import Atmosphere
from smartg.water import IOP_base
from os.path import dirname, realpath, join, exists
from warnings import warn
from smartg.albedo import Albedo_cst, Albedo_spectrum_map 
from smartg.tools.progress import Progress
from smartg.tools.modified_environ import modified_environ
from luts.luts import MLUT
from scipy.interpolate import interp1d
import subprocess
from collections import OrderedDict
from pycuda.gpuarray import to_gpu, zeros as gpuzeros
import pycuda.driver as cuda
from smartg.bandset import BandSet
from pycuda.compiler import SourceModule
from pycuda.driver import module_from_buffer
# bellow necessary for object incorporation
from smartg.geometry import Vector, Point, Normal, Ray, BBox, CoordinateSystem, \
    Normalize, Dot
from smartg.transform import Transform, Aff
from smartg.visualizegeo import Mirror, Plane, Spheric, Transformation, \
    Entity, Analyse_create_entity, LambMirror, Matte, convertVtoAngles, \
    convertAnglestoV, GroupE


# set up directories
dir_root = dirname(dirname(realpath(__file__)))
dir_src = join(dir_root, 'smartg/src/')
src_device = join(dir_src, 'device.cu')

# constants definition
# (should match #defines in src/communs.h)
SPACE    =  0
ATMOS    =  1
SURF0P   =  2   # surface (air side)
SURF0M   =  3   # surface (water side)
ABSORBED =  4
NONE     =  5
OCEAN    =  6
SEAFLOOR =  7
OBJSURF  =  8
LOC_CODE = ['','ATMOS','SURF0P','SURF0M','','','OCEAN','SEAFLOOR', 'OBJSURF']

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
    ('n',      'float32'),    # // refractive index
    ('OD',     'float32'),    # // cumulated extinction optical thickness (from top)
    ('OD_sca', 'float32'),    # // cumulated scattering optical thickness (from top)
    ('OD_abs', 'float32'),    # // cumulated absorption optical thickness (from top)
    ('pmol',   'float32'),    # // probability of pure Rayleigh scattering event
    ('ssa',    'float32'),    # // layer single scattering albedo
    ('pine',   'float32'),    # // layer fraction of inelastic scattering
    ('FQY1',   'float32'),    # // layer Fluorescence Quantum Yield of 1st specie
    ('iphase', 'int32'),      # // phase function index
    ]

type_Cell = [
    ('iopt',     'int32'),    # // Optical scattering properties index
    ('iabs',     'int32'),    # // Optical absorbing properties index
    ('pminx',  'float32'),    # // Box point pmin.x
    ('pminy',  'float32'),    # // Box point pmin.y
    ('pminz',  'float32'),    # // Box point pmin.z
    ('pmaxx',  'float32'),    # // Box point pmax.x
    ('pmaxy',  'float32'),    # // Box point pmax.y
    ('pmaxz',  'float32'),    # // Box point pmax.z
    ('neighbour1', 'int32'),   # // neighbour box index +X
    ('neighbour2', 'int32'),   # // neighbour box index -X
    ('neighbour3', 'int32'),   # // neighbour box index +Y
    ('neighbour4', 'int32'),   # // neighbour box index -Y
    ('neighbour5', 'int32'),   # // neighbour box index +Z
    ('neighbour6', 'int32'),   # // neighbour box index -Z
    ]

type_Sensor = [
    ('POSX',   'float32'),    # // X position of the sensor
    ('POSY',   'float32'),    # // Y position of the sensor
    ('POSZ',   'float32'),    # // Z position of the sensor (fromp Earth's center in spherical, from the ground in PP)
    ('THDEG',  'float32'),    # // zenith angle of viewing direction (Zenith> 90 for downward looking, <90 for upward, default Zenith)
    ('PHDEG',  'float32'),    # // azimut angle of viewing direction
    ('LOC',    'int32'),      # // localization (ATMOS=1, ...), see constant definitions in communs.h
    ('FOV',    'float32'),    # // sensor FOV (degree) 
    ('TYPE',   'int32'),      # // sensor type: Radiance (0), Planar flux (1), Spherical Flux (2), default 0
    ('ICELL',  'int32'),      # // Box in which the sensor is
    ]

type_IObjets = [
    ('geo', 'int32'),         # 1 = sphere, 2 = plane, ...
    ('materialAV', 'int32'),  # 1 = LambMirror, 2 = Matte,
    ('materialAR', 'int32'),  # 3 = Mirror, ... (AV = avant, AR = Arriere)
    ('type', 'int32'),        # 1 = reflector, 2 = receiver
    ('reflecAV', 'float32'),  # reflectivity of materialAV
    ('reflecAR', 'float32'),  # reflectivity of materialAR
    ('roughAV', 'float32'),   # roughness of materialAV
    ('roughAR', 'float32'),   # roughness of materialAR
    ('shdAV', 'int32'),       # shadow option of materialAV, 0=false, 1=true
    ('shdAR', 'int32'),       # shadow option of materialAR
    ('nindAV', 'float32'),    # refractive index of materialAV
    ('nindAR', 'float32'),    # refractive index of materialAR
    ('distAV', 'int32'),      # distribution used for materialAV, 1=Beck, 2=GGX
    ('distAR', 'int32'),      # distribution used for materialAR
    
    ('p0x', 'float32'),       # \            \
    ('p0y', 'float32'),       #  | point p0   \
    ('p0z', 'float32'),       # /              \ 
                              #                 |
    ('p1x', 'float32'),       # \               | 
    ('p1y', 'float32'),       #  | point p1     | 
    ('p1z', 'float32'),       # /               |
                              #                 | Plane Object  
    ('p2x', 'float32'),       # \               | 
    ('p2y', 'float32'),       #  | point p2     |
    ('p2z', 'float32'),       # /               | 
                              #                 |
    ('p3x', 'float32'),       # \              /
    ('p3y', 'float32'),       #  | point p3   /
    ('p3z', 'float32'),       # /            /

    ('myRad', 'float32'),     # \
    ('z0', 'float32'),        #  | Sperical Object
    ('z1', 'float32'),        #  |
    ('phi', 'float32'),       # /
    
    ('mvRx', 'float32'),      # \
    ('mvRy', 'float32'),      #  | Transformation type rotation
    ('mvRz', 'float32'),      # /
    ('rotOrder', 'int32'),    # rotation order: 1=XYZ; 2=XZY;...

    ('mvTx', 'float32'),      # \
    ('mvTy', 'float32'),      #  | tranformation type translation 
    ('mvTz', 'float32'),      # /

    ('nBx', 'float32'),       # \
    ('nBy', 'float32'),       #  | normalBase de l'obj apres trans 
    ('nBz', 'float32'),       # /
    ]

type_GObj = [
    ('nObj', 'int32'),        # Number of objects in this group
    ('index', 'int32'),       # Index at the table of IObjects where
                              # we start to fill the objects of the group

    ('bPminx', 'float32'),    #\
    ('bPminy', 'float32'),    # |
    ('bPminz', 'float32'),    # | Bounding box of the group        
    ('bPmaxx', 'float32'),    # |
    ('bPmaxy', 'float32'),    # |
    ('bPmaxz', 'float32'),    #/
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
                'WAVE_SHADOW': 0,
                'BRDF' : 0,
                'SINGLE' : 1,
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
        WAVE_SHADOW : include wave shadowing effect (default not)
        BRDF : replace slope sampling by Cox & Munk BRDF, no ocean, just reflection
        SINGLE: dont allow multiple reflections/refractions at the interface, default False
    '''
    def __init__(self, WIND=5., SUR=3, NH2O=1.33, WAVE_SHADOW=False, BRDF=False, SINGLE=False):

        self.dict = {
                'SUR': SUR if not BRDF else 1,
                'DIOPTRE': 1,
                'WINDSPEED': WIND,
                'NH2O': NH2O,
                'WAVE_SHADOW': 1 if WAVE_SHADOW else 0,
                'BRDF': 1 if BRDF else 0,
                'SINGLE': 1 if SINGLE else 0,
                }
    def __str__(self):
        return 'ROUGHSUR={SUR}-WIND={WINDSPEED}-DI={DIOPTRE}-WAVE_SHADOW={WAVE_SHADOW}-BRDF={BRDF}-SINGLE={SINGLE}'.format(**self.dict)


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
                'WAVE_SHADOW': 0,
                'BRDF': 1,
                'SINGLE': 1,
                }
    def __str__(self):
        return 'LAMBSUR-ALB={SURFALB}'.format(**self.dict)


class Environment(object):
    '''
    Stores the smartg parameters relative the the environment effect

    ENV: environment effect (default 0: deactivated, 1: horizontal cst albedo ALB outside an horizontal disk, 
                             water is inside, lambertian ALB is outside.
                             2: ALB map2D modulated by checkerboard spatial function)
    ENV_SIZE, X0, Y0: radius and position of the circle outside which ALB model is applied for case 1),
                             size of the spatial pattern (in km), and in the direction X and or Y applied (
                             X=1, applied to X, X=0 Not applied to X; idem for Y)
    ALB: albedo spectral model

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

class Sensor(object):
    '''
    Definition of the sensor

    POS: Position (X,Y,Z) in cartesian coordinates, default origin
    TH,PH: Direction (theta, phi) of zenith and azimuth angles of viewing direction
            (Zenith> 90 for downward looking, <90 for upward, default Zenith)
    LOC: Localization (default SURF0P)
    FOV: Field of View (deg, default 0.)
    TYPE: Radiance (0), Planar flux (1), Spherical Flux (2), default 0
    ICELL: Box index in which the sensor is (3D)
    '''
    def __init__(self, POSX=0., POSY=0., POSZ=0., THDEG=0., PHDEG=180.,
                 LOC='SURF0P', FOV=0., TYPE=0, ICELL=0, V = None):

        if (isinstance(V, Vector)):
            THDEG, PHDEG = convertVtoAngles(V)
        elif (V != None):
            raise NameError('V argument must be a Vector')
        
        self.dict = {
            'POSX':  POSX,
            'POSY':  POSY,
            'POSZ':  POSZ,
            'THDEG': THDEG,
            'PHDEG': PHDEG,
            'LOC'  : LOC_CODE.index(LOC),
            'FOV':   FOV,
            'TYPE':  TYPE,
            'ICELL': ICELL
        }

    def __str__(self):
        return 'SENSOR=-POSX{POSX}-POSY{POSY}-POSZ{POSZ}-THETA={THDEG:.3f}-PHI={PHDEG:.3f}'.format(**self.dict)
        
class CusForward(object):
    '''
    Definition of CusForward 

    Custum rectangular forward mode of surface X*Y

    CFX  : Size in X (only for FF LMODE)
    CFY  : Size in Y (only for FF LMODE)
    CFTX : Translation in x axis (only for FF LMODE)
    CFTY : Translation in y axis (only for FF LMODE)
    FOV  : Field of view or half-angle of the sun (only for FF LMODE)
    TYPE : Sampling type, 2 choices: 'lambertian' or 'isotropic' (only for FF LMODE)
    LMODE (Launching mode) : RF = Restricted Forward OR FF = Full Forward
                             RF --> Launch the photons such that the direct beams
                                    fill only reflector objects
                             FF --> Launch the photons in a rectangle from TOA
                                    whrere the beams at the center, with the solar
                                    dir, targets by default the origin point (0,0,0)
    '''
    def __init__(self, CFX=0., CFY=0., CFTX = 0., CFTY = 0., FOV = 0., TYPE = "lambertian",
                 LMODE = "RF", LPH=None, LPR=None):

        if (TYPE == "lambertian"): TYPE = 1
        elif (TYPE == "isotropic"): TYPE = 2
        elif (TYPE == "disk"): TYPE = 3
        else: raise NameError('You must choose lambertian or isotropic sampling')

        self.dict = {
            'CFX':   CFX,
            'CFY':   CFY,
            'CFTX':  CFTX,
            'CFTY':  CFTY,
            'FOV':   FOV,
            'TYPE':  TYPE,
            'LMODE': LMODE,
            # under developement->
            'LPH':     LPH,
            'LPR':     LPR
        }
        
    def __str__(self):
        return 'CusForward=-CFX{CFX}-CFY{CFY}-CFTX{CFTX}-CFTY{CFTY}'.format(**self.dict) + \
            '-FOV{FOV}-TYPE{TYPE}-LMODE{LMODE}'.format(**self.dict)

class CusBackward(object):
    '''
    Definition of CusBackward 

    POS     : Position (X,Y,Z) in cartesian coordinates, default origin (class Point)
    TH,PH   : Direction (theta, phi) of zenith and azimuth angles of viewing direction
              (Zenith> 90 for downward looking, <90 for upward, default Zenith)
    ALDEG   : Launch in a solid angle where alpha is the half-angle of the cone
    V       : Direction but represented by a Vector type, if not None replace TH, PH
    REC     : Receiver object must be specified in and only in BR mode
    TYPE    : Sampling type : 2 choices: 'lambertian' or 'isotropic' (only BR mode)
    LMODE (Launching mode) : B = basic Backward, BR = Backward with receiver
                             B -> Launch the photons from a given point in a given
                                  direction (eventually can choose a ramdom vector
                                  in a solid angle arround a given direction delimited
                                  by the half-angle ALDEG)
                             B -> Launch the photons from a given receiver (plane obj)
                                  in a given direction (also eventually can choose a
                                  ramdom vector in a solid angle delimited by ALDEG)
    '''
    def __init__(self, POS = Point(0., 0., 0.), THDEG = 0., PHDEG = 0., V = None,
                 ALDEG = 0., REC = None, TYPE = "lambertian", LMODE = "B", LPH = None, LPR = None):

        if (isinstance(V, Vector)): THDEG, PHDEG = convertVtoAngles(V)
        elif (V != None): raise NameError('V argument must be a Vector')
        if LMODE == "BR" and not isinstance(REC, Entity):
            raise NameError('In BR LMODE you have to specify a receiver!')
        if (TYPE == "lambertian"): TYPE = 1
        elif (TYPE == "isotropic"): TYPE = 2
        else: raise NameError('You must choose lambertian or isotropic sampling')

        self.dict = {
            'POS':    POS,
            'THDEG':  THDEG,
            'PHDEG':  PHDEG,
            'ALDEG':  ALDEG,
            'REC':    REC,
            'TYPE':   TYPE,
            'LMODE':  LMODE,
            # under developement->
            'LPH':     LPH,
            'LPR':     LPR
        }

    def __str__(self):
        return 'CusBackward:-POS={POS}-THDEG={THDEG}-PHDEG={PHDEG}'.format(**self.dict) + \
            '-ALDEG={ALDEG}-TYPE{TYPE}-LMODE={LMODE}'.format(**self.dict)
    
class Smartg(object):
    '''Initialization of the Smartg object

    Performs the compilation and loading of the kernel.
    This class is designed so split compilation and kernel loading from the
    code execution: in case of successive smartg executions, the kernel
    loading time is not repeated.

    Args:

        pp: plane parallel or spherical
            True: use plane parallel geometry (default)
            False: use spherical shell geometry

        debug: set to True to activate debug mode (optional stdout if problems are detected)

        verbose_photon: activate the display of photon path for the thread 0

        double: accumulate photons table in double precision, default double
            This operation is much faster on GPUs with ARCH >= 600
            (Pascal architecture, like GeForce 10xx or greater)

        alis: boolean, if present implement the ALIS method (Emde et al. 2010) for treating gaseous absorption and perturbed profile

        back: boolean, if True, run in backward mode, default forward mode

        bias: boolean, if True, use the bias sampling scheme, default True

        obj3D: Set to True to enable simulation with 3D objects

        opt3D: Set to True to enable simulation with 3D optical properties

        alt_pp: boolean, if True new PP progation scheme is used

        rng: choice of pseudo-random number generator:
            * PHILOX
            * CURAND_PHILOX

        - device: device number (str or int) to be set to CUDA_DEVICE environment variable for use by 'import pycuda.autoinit'
            see https://documen.tician.de/pycuda/util.html
            Please note that after the first pycuda.autoinit, the device used by pycuda will not change.
        - sif : boolean, if True Sun Induced Fluorescence included, default False

        - rng: choice of pseudo-random number generator:
                * PHILOX
                * CURAND_PHILOX

    '''
    def __init__(self, pp=True, debug=False,
                 verbose_photon=False,
                 double=True, alis=False, back=False, bias=True, alt_pp=False, obj3D=False, 
                 opt3D=False, device=None, sif=False, rng='PHILOX'):
        assert not ((device is not None) and ('CUDA_DEVICE' in os.environ)), "Can not use the 'device' option while the CUDA_DEVICE is set"

        if device is not None:
            env_modif = {'CUDA_DEVICE': str(device)}
        else:
            env_modif = {}
        with modified_environ(**env_modif):
            import pycuda.autoinit

        self.pp = pp
        self.double = double
        self.alis = alis
        self.rng = init_rng(rng)
        self.back= back
        self.obj3D= obj3D
        self.opt3D= opt3D

        #
        # compilation option
        #
        options = []
        #options = ['-G']
        #options = ['-g', '-G']
        if not pp:
            # spherical shell calculation
            # automatically with ALT_PP (for eventually ocean propagation)
            options.append('-DSPHERIQUE')
            options.append('-DALT_PP')
        if alt_pp:
            # new Plane Parallel propagation scheme
            options.append('-DALT_PP')
        if opt3D:
            # 3D optical properties enabled
            # automatically with ALT_PP
            # for the moment inconsistent with OBJ3D
            options.append('-DALT_PP')
            options.append('-DOPT3D')
        if debug:
            # additional tests for debugging
            options.append('-DDEBUG')
        if verbose_photon:
            options.append('-DVERBOSE_PHOTON')
        if double:
            # counting in double precision
            # ! slows down processing
            options.append('-DDOUBLE')
        if alis:
            options.append('-DALIS')
        if sif:
            options.append('-DSIF')
        if back:
            # backward mode
            options.append('-DBACK')
        if bias:
            # bias sampling scheme for scattering and reflection/transmission
            options.append('-DBIAS')
        if obj3D:
            # 3D Object mode
            options.append('-DOBJ3D')
        options.append('-D'+rng)


        #
        # compile the kernel or load binary
        #
        time_before_compilation = datetime.now()

        # load device.cu
        src_device_content = open(
            src_device, encoding='ascii', errors='ignore'
            ).read()

        # kernel compilation
        self.mod = SourceModule(src_device_content,
                           nvcc='nvcc',
                           options=options,
                           no_extern_c=True,
                           cache_dir='/tmp/',
                           include_dirs=[dir_src,
                               join(dir_src, 'incRNGs/Random123/')])

        # load the kernel
        self.kernel = self.mod.get_function('launchKernel')
        #self.kernel2 = self.mod.get_function('launchKernel2')

        #
        # common attributes
        #
        self.common_attrs = OrderedDict()
        self.common_attrs['compilation_time'] = (datetime.now()
                        - time_before_compilation).total_seconds()
        self.common_attrs['device'] = pycuda.autoinit.device.name()
        try:
            self.common_attrs['device_number'] = pycuda.autoinit.device.get_attributes()[pycuda._driver.device_attribute.MULTI_GPU_BOARD_GROUP_ID]
        except AttributeError:
            self.common_attrs['device_number'] = 'undefined'
        self.common_attrs['pycuda_version'] = pycuda.VERSION_TEXT
        self.common_attrs['cuda_version'] = '.'.join([str(x) for x in pycuda.driver.get_version()])
        self.common_attrs.update(get_git_attrs())


    def run(self, wl,
             atm=None, surf=None, water=None, env=None, map2D=None, alis_options=None,
             NBPHOTONS=1e9, DEPO=0.0279, DEPO_WATER= 0.0906, THVDEG=0., PHVDEG=0., SEED=-1,
             RTER=6371., wl_proba=None,
             NBTHETA=45, NBPHI=90, NF=1e6,
             OUTPUT_LAYERS=0, XBLOCK=256, XGRID=256,
             NBLOOP=None, progress=True, 
             le=None, flux=None, stdev=False, BEER=1, RR=1, WEIGHTRR=0.1, SZA_MAX=90., SUN_DISC=0,
             sensor=None, refraction=False, reflectance=True,
             myObjects=None, interval = None,
             IsAtm = 1, cusL = None, SMAX=1e6, FFS=False, DIRECT=False):
        '''
        Run a SMART-G simulation

        Arguments:

            wl: a scalar or list/array of wavelengths (in nm)
                  or a list of REPTRAN or KDIS IBANDS

            atm: Profile object
                default None (no atmosphere)
                Example:
                    # clear atmosphere, AFGL midlatitude summer
                    AtmAFGL('afglms')
                    # AFGL tropical with maritime clear aerosols AOT(550)=0.3
                    AtmAFGL('afglt', aer=[AeroOPAC('maritime_clean', 0.3, 550.)])

            surf: Surface object

                * default None (no surface)
                * RoughSurface(WIND=5.)  # wind-roughened ocean surface
                * FlatSurface()          # flat air-water interface
                * LambSurface(ALB=0.1)   # Lambertian surface of albedo 0.1

            water: water object, providing options relative to the ocean surface
                default None (no ocean)

            env: environment effect object (a.k.a. adjacency effect)
                default None (no environment effect)

            alis_options : required if compiled already with the alis option. Dictionary, field 'nlow'
                is the number of wavelength  where the spectral dependency of scattering is calculated, 
                nlow-1 has to divide NW-1 where NW is the number of wavelengths, nlow has to be lesser than MAX_NLOW that is defined in communs.h,
                optionnal field 'njac' is the number of perturbed profiles, default is zero (None): no Jacobian

            NBPHOTONS: number of photons launched

            DEPO: (Air) Rayleigh depolarization ratio

            DEPO_WATER: (Water) Rayleigh depolarization ratio

            THVDEG: zenith angle of the observer in degrees
                the result corresponds to various positions of the sun
                NOTE: in plane parallel geometry, due to Fermat's principle, we
                can exchange the positions of the sun and observer.

            PHVDEG: azimuth angle of the observer in degrees
                the result corresponds to various positions of the sun
                NOTE: It can be very useful to modify only this value instead
                      of all the positions of all the objects

            SEED: integer used to initiate the series of random numbers
                default: based on clock time

            RTER: earth radius in km

            wl_proba: inversed cumulative distribution function for wavelength selection
                        (it is the result of function ICDF(proba, N))

            NBTHETA: number of zenith angles in output

            NBPHI: number of azimuth angles in output

            NF: number of discretization of :
                    * the inversed aerosol phase functions
                    * the inversed ocean phase functions
                    * the inversed probability of each wavelength occurence

            OUTPUT_LAYERS: control the output layers. Add the following values:
                0: top of atmosphere only (TOA)
                1: add output layers at (0+, down) and (0-, up)
                2: add output layers at (0-, down) and (0+, up)
                Example: OUTPUT_LAYERS=3 to use all output layers.

            XBLOCK and XGRID: control the number of blocks and grid size for
              the GPU execution

            NBLOOP: number of photons launched in one kernel run

            progress: whether to show a progress bar (True/False)

            le: Local Estimate method activation
                  Provide output geometries in radians like so:
                  le={'th': <array-like>, 'phi': <array-like>}
                  or:
                  le={'th_deg': <array-like>, 'phi_deg': <array-like>}    # to provide angles in degrees
                  The defaut output is two dimensional NBPHI x NBTHETA
                  If 'zip' is present in the dictionary, then 'th' and 'phi' covary and the output is only
                  one-dimensional NBTHETA, but user should verify that NBPHI==NBTHETA
                  Angles can be provided as scalar, lists or 1-dim arrays
                  Default None: cone sampling
                  NOTE: Overrides NBPHI and NBTHETA

            flux: if specified output is 'planar' or 'spherical' flux instead of radiance

            stdev: calculate the standard deviation between each kernel run

            RR: Russian Roulette ON  = 1
                                   OFF = 0

            WEIGHTRR threshold weight to apply the Russian Roulette

            BEER: if BEER=1 compute absorption using Beer-Lambert law, otherwise compute it with the Single scattering albedo
                (BEER automatically set to 1 if ALIS is chosen)

            SZA_MAX : Maximum SZA for solar BOXES in case a Regulard grid and cone sampling

            SUN_DISC : Angular size of the Sun disc in degree, 0 (default means no angular size)

            sensor : sensor object or list, backward mode (from sensor to source), back should be set to True in the smartg constructor

            refraction : include atmospheric refraction

            reflectance : if flux is None, output is in reflectance units if True,(for plane parallel atmosphere). Otherwise
                is is in radiance units with Solar irradiance set to PI (default False)
            
            myObjects : liste d'objets (objets de classe entity)
        
            interval : liste composée de deux listes [[pxmin, pymin, pzmin], [[pxmin, pymin, pzmin]]
                         interval définit l'interval d'études des objets délimitée par deux points (pmin et pmax).

            IsAtm (effet uniquement si myObjects != None) : si égal à 0 , cela permet dans le cas sans atmosphère,
                      d'empêcher certaines fuites de photons.

            cusL : None is the default mode (sun is a ponctual source targeting the origin (0,0,0)), else it
                      enable to use the RF, FF or B launching mode (see the class CusForward) --> cusL=CusForward(...)

            SMAX : Maximum Scattering oorder: Default 1e6

            FFS : Forced First Scattering (for use in spherical limb geometry only): Default False

            DIRECT : Include directly transmitted photons: Default False

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
            M['I_up (TOA)'][:,:] contains the top of atmosphere radiance/reflectance
        '''


        # Compute the sun direction as vector 
        vSun = convertAnglestoV(THETA=THVDEG, PHI=PHVDEG, TYPE="Sun") 
        vSun = Normalize(vSun)

        # First check if back option is activated in case of the use of cusBackward launching mode
        surfLPH = 0;
        if (cusL is not None):
            if (cusL.dict['LMODE'] == "B" and self.back == False):
                raise NameError('CusBackward can be use only with the compilation option back=True')
            elif (sensor != None):
                raise NameError('The use of sensor(s) and a custum launching mode' + \
                                ' (cusForward or cusBackward) is prohibited!')
            elif (cusL.dict['LMODE'] == "B"):
                sensor = Sensor(POSX=cusL.dict['POS'].x, POSY=cusL.dict['POS'].y, POSZ=cusL.dict['POS'].z,
                                THDEG=cusL.dict['THDEG'], PHDEG=cusL.dict['PHDEG'], LOC='ATMOS',
                                FOV=0.0, TYPE=0)
                                #FOV=cusL.dict['ALDEG'], TYPE=cusL.dict['TYPE'])
            elif (cusL.dict['LMODE'] == "BR"):
                sensor = Sensor(POSX=cusL.dict['REC'].transformation.transx,
                                POSY=cusL.dict['REC'].transformation.transy,
                                POSZ=cusL.dict['REC'].transformation.transz,
                                THDEG=cusL.dict['THDEG'], PHDEG=cusL.dict['PHDEG'], LOC='ATMOS',
                                FOV=0.0, TYPE=0)
                                #FOV=cusL.dict['ALDEG'], TYPE=cusL.dict['TYPE'])
            elif (cusL.dict['LMODE'] == "FF"):
                # The projected surface at TOA where the photons are launched
                DotNN = Dot(vSun*-1, Vector(0., 0., 1.))
                surfLPH = float(cusL.dict['CFX'])*float(cusL.dict['CFY'])*DotNN

        #
        # initialization
        #              
        
        # Begin initialization with OBJ ============================
        if (myObjects is not None):
            # Main bounding box initialization
            if interval is not None:
                Pmin_x = interval[0][0];Pmin_y = interval[0][1];Pmin_z = interval[0][2];
                Pmax_x = interval[1][0];Pmax_y = interval[1][1];Pmax_z = interval[1][2];
            else:
                Pmin_x = -100000; Pmin_y = -100000; Pmin_z = 0;
                Pmax_x = 100000;  Pmax_y = 100000; Pmax_z = 120;

            # Initiliaze all the parameters linked with 3D objects
            (nGObj, nObj, nRObj, surfLPH_RF, nb_H, zAlt_H, totS_H, TC, nbCx, nbCy,
             myObjects0, myGObj0, myRObj0) = initObj(LGOBJ=myObjects, vSun=vSun, CUSL=cusL)

            # If we are in RF mode don't forget to update the value of surfLPH
            if (surfLPH_RF is not None): surfLPH = surfLPH_RF

        else:
            myObjects0 = gpuzeros(1, dtype='int32')
            myGObj0 = gpuzeros(1, dtype='int32')
            myRObj0 = gpuzeros(1, dtype='int32')
            nObj = 0; nGObj=0; nRObj=0; Pmin_x = None; Pmin_y = None; Pmin_z = None;
            Pmax_x = None; Pmax_y = None; Pmax_z = None;
            IsAtm = None; TC = None; nbCx = 10; nbCy = 10; nb_H = 0
        # END OBJ ===================================================

        if NBPHI%2 == 1:
            warn('Odd number of azimuth')

        if (NBLOOP is None) and (nObj <= 0):
            NBLOOP = min(NBPHOTONS/30, 1e6)
        elif (NBLOOP is None) and (nObj > 0):
            NBLOOP = min(NBPHOTONS/10, 1e6)

        NF = int(NF)

        # number of output levels
        # warning! values defined in communs.h should be < LVL
        NLVL = 6

        # warning! values defined in communs.h 
        MAX_HIST = 2048 * 2048
        #MAX_NLOW = 101
        MAX_NLOW = 401

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

        NLOW=0
        hist=False
        HIST=0
        NJAC=0
        if alis_options is not None :
            if 'hist' in alis_options.keys():
                if alis_options['hist']: hist=True
            if 'njac' in alis_options.keys():
                NJAC=alis_options['njac']
            if (alis_options['nlow'] ==-1) : NLOW=NLAM
            else: NLOW=alis_options['nlow']
            BEER=1
            assert (NLOW <= MAX_NLOW)
        
        if hist : HIST=1

        if surf is not None:
            if surf.dict['BRDF'] !=0 :
                water = None # special case BRDF, water is shortcut

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
            prof_atm_gpu, cell_atm_gpu = init_profile(wl, prof_atm, 'atm')
            NATM = len(prof_atm.axis('z_atm')) - 1
            if self.opt3D : 
                NATM_ABS = prof_atm['iabs_atm'].data.max().astype(np.int32)
            else : NATM_ABS = NATM
        else:
            faer = gpuzeros(1, dtype='float32')
            prof_atm_gpu = to_gpu(np.zeros(1, dtype=type_Profile))
            cell_atm_gpu = to_gpu(np.zeros(1, dtype=type_Cell))
            NATM = 0
            NATM_ABS = 0

        # computation of the impact point
        X0, tabTransDir = impactInit(prof_atm, NLAM, THVDEG, RTER, self.pp)

        # sensor definition
        if sensor is None:
            # by defaut sensor in forward mode, with ZA=180.-THVDEG, PHDEG=180., FOV=0.
            if (SIM == 3):
                sensor2 = [Sensor(THDEG=180.-THVDEG, PHDEG=PHVDEG+180., LOC='OCEAN')] 
            elif ((SIM == -1) or (SIM == 0)):  
                sensor2 = [Sensor(THDEG=180.-THVDEG, PHDEG=PHVDEG+180., LOC='SURF0P')] 
            else:
                if (cusL is not None): # for FF mode
                    sensor2 = [Sensor(POSX=X0.get()[0], POSY=X0.get()[1], POSZ=X0.get()[2],
                                      THDEG=180.-THVDEG, PHDEG=PHVDEG+180., LOC='ATMOS')]
                                      #FOV=0.0, TYPE=0)]
                                      #FOV=cusL.dict['FOV'], TYPE=cusL.dict['TYPE'])]
                else:
                    sensor2 = [Sensor(POSX=X0.get()[0], POSY=X0.get()[1], POSZ=X0.get()[2], THDEG=180.-THVDEG, PHDEG=PHVDEG+180., LOC='ATMOS')]
            
        if isinstance(sensor, Sensor):
            sensor2=[sensor]
        if isinstance(sensor, list):
            sensor2=sensor

        NSENSOR=len(sensor2)

        tab_sensor = np.zeros(NSENSOR, dtype=type_Sensor, order='C')
        for (i,s) in enumerate(sensor2) :
            for k in s.dict.keys():
                  tab_sensor[i][k] = s.dict[k]
        tab_sensor = to_gpu(tab_sensor)

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
            prof_oc_gpu, cell_oc_gpu = init_profile(wl, prof_oc, 'oc')
            NOCE = len(prof_oc.axis('z_oc')) - 1
            if self.opt3D : NOCE_ABS = prof_oc['iabs_oc'].data.max().astype(np.int32)
            else : NOCE_ABS = NOCE
        else:
            foce = gpuzeros(1, dtype='float32')
            prof_oc_gpu = to_gpu(np.zeros(1, dtype=type_Profile))
            cell_oc_gpu = to_gpu(np.zeros(1, dtype=type_Cell))
            NOCE = 0
            NOCE_ABS = 0

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
            mapalb = env.alb.get(wl[:])
            if mapalb.ndim==3:
                mapalb = to_gpu(mapalb)
            else:
                spectrum['alb_surface'] = env.alb.get(wl[:])

        if water is None:
            spectrum['alb_seafloor'] = -999.
        else:
            spectrum['alb_seafloor'] = prof_oc['albedo_seafloor'].data[...]
        spectrum = to_gpu(spectrum)


        # Local Estimate option
        LE = 0
        ZIP= 0
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
            NBPHI   = le['phi'].shape[0]

            if 'zip' in le:
                if le['zip']:
                    assert NBPHI==NBTHETA
                    ZIP = 1
                    NBPHI = 1 

        '''
        # Multiple Init Direction
        MI = 0
        if mi is not None:
            MI = 1
            if not 'th' in mi:
                mi['th'] = np.array(mi['th_deg'], dtype='float32').ravel() * np.pi/180.
            else:
                mi['th'] = np.array(mi['th'], dtype='float32').ravel()
            if not 'phi' in le:
                mi['phi'] = np.array(mi['phi_deg'], dtype='float32').ravel() * np.pi/180.
            else:
                mi['phi'] = np.array(mi['phi'], dtype='float32').ravel()
            NBTHETA0 =  mi['th'].shape[0]
            NBPHI0   =  mi['phi'].shape[0]
         '''


        FLUX = 0
        if flux is not None:
            LE=0
            if flux== 'planar' : 
                FLUX = 1
            if flux== 'spherical' : 
                FLUX = 2


        if wl_proba is not None:
            assert wl_proba.dtype == 'int64'
            wl_proba_icdf = to_gpu(wl_proba)
            NWLPROBA = len(wl_proba_icdf)
        else:
            wl_proba_icdf = gpuzeros(1, dtype='int64')
            NWLPROBA = 0

        REFRAC = 0
        if refraction: 
            REFRAC=1
            fname='/home/did/RTC/SMART-G/smartg/refraction_LUT.nc'

        HORIZ = 1
        if (not self.pp and not reflectance): HORIZ = 0


        # initialization of the constants
        InitConst(surf, env, NATM, NATM_ABS, NOCE, NOCE_ABS, self.mod,
                  NBPHOTONS, NBLOOP, THVDEG, DEPO,
                  XBLOCK, XGRID, NLAM, SIM, NF,
                  NBTHETA, NBPHI, OUTPUT_LAYERS,
                  RTER, LE, ZIP, FLUX, FFS, DIRECT, NLVL, NPSTK,
                  NWLPROBA, BEER, SMAX, RR, WEIGHTRR, NLOW, NJAC, 
                  NSENSOR, REFRAC, HORIZ, SZA_MAX, SUN_DISC, cusL, nObj, nGObj, nRObj,
                  Pmin_x, Pmin_y, Pmin_z, Pmax_x, Pmax_y, Pmax_z, IsAtm,
                  TC, nbCx, nbCy, vSun, HIST)

        # Initialize the progress bar
        p = Progress(NBPHOTONS, progress)

        # Initialize the RNG
        SEED = self.rng.setup(SEED, XBLOCK, XGRID)

        # Loop and kernel call
        (NPhotonsInTot, tabPhotonsTot, tabDistTot, tabHistTot, errorcount, 
         NPhotonsOutTot, sigma, Nkernel, secs_cuda_clock, cMatVisuRecep, vecLoss, matCats, matLoss
        ) = loop_kernel(NBPHOTONS, faer, foce,
                        NLVL, NATM, NATM_ABS, NOCE, NOCE_ABS, MAX_HIST, NLOW, NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                        NLAM, NSENSOR, self.double, self.kernel, None, p, X0, le, tab_sensor, spectrum,
                        prof_atm_gpu, prof_oc_gpu, cell_atm_gpu, cell_oc_gpu,
                        wl_proba_icdf, stdev, self.rng, self.alis, myObjects0, TC, nbCx, nbCy, myGObj0, myRObj0, hist=hist)

        attrs['kernel time (s)'] = secs_cuda_clock
        attrs['number of kernel iterations'] = Nkernel
        attrs['seed'] = SEED
        attrs.update(self.common_attrs)

        # If there is a receiver -> normalization of the signal collected
        if (TC is not None):
            cMatVisuRecep, matCats = normalizeRecIrr(cMatVisuRecep=cMatVisuRecep, matCats=matCats,
                nbCx=nbCx, nbCy=nbCy, NBPHOTONS=NBPHOTONS, surfLPH=surfLPH, TC=TC, cusL=cusL,
                SUN_DISC=SUN_DISC)

        if (nb_H > 0 and (cusL.dict['LMODE'] == "FF" or cusL.dict['LMODE'] == "BR") and TC is not None):
            MZAlt_H = zAlt_H/nb_H; weightR=matCats[2, 1]; SREC=TC*TC*nbCx*nbCy;
            # dicSTP : tuple incorporating parameters for Solar Tower Power applications
            dicSTP = {"nb_H":nb_H, "totS_H":totS_H, "surfTOA":surfLPH, "MZAlt_H":MZAlt_H, "vSun":vSun, "wRec":matCats[2, 1],
                      "SREC":SREC, "LPH":cusL.dict['LPH'], "LPR":cusL.dict['LPR']}
        else: # If there are no heliostats --> there is no STP and then no analyses of optical losses
            dicSTP = None; vecLoss = None; weightR=0
                
        # finalization
        output = finalize(tabPhotonsTot, tabDistTot, tabHistTot, wl[:], NPhotonsInTot, errorcount, NPhotonsOutTot,
                          OUTPUT_LAYERS, tabTransDir, SIM, attrs, prof_atm, prof_oc,
                          sigma, THVDEG, HORIZ, le=le, flux=flux, back=self.back, 
                          SZA_MAX=SZA_MAX, SUN_DISC=SUN_DISC, hist=hist, cMatVisuRecep=cMatVisuRecep,
                          losses=vecLoss, dicSTP=dicSTP, matCats=matCats, matLoss=matLoss)
        
        output.set_attr('processing time (s)', (datetime.now() - t0).total_seconds())

        if self.alis:
            p.finish('Done! | Received {:.1%} of {:.3g} photons ({:.1%})'.format(
            np.sum(NPhotonsOutTot[0,...])/float(np.sum(NPhotonsInTot)),
            np.sum(NPhotonsInTot)/float(NLAM),
            np.sum(NPhotonsInTot)/float(NBPHOTONS)/float(NLAM),
            ))
        else:
            p.finish('Done! | Received {:.1%} of {:.3g} photons ({:.1%})'.format(
            np.sum(NPhotonsOutTot[0,...])/float(np.sum(NPhotonsInTot)),
            np.sum(NPhotonsInTot),
            np.sum(NPhotonsInTot)/float(NBPHOTONS),
            ))

        if wl.scalar:
            output = output.dropaxis('wavelength')
            output.attrs['wavelength'] = wl[:]

        return output


def calcOmega(NBTHETA, NBPHI, SZA_MAX=90., SUN_DISC=0):
    '''
    returns the zenith and azimuth angles, and the solid angles
    '''

    # zenith angles
    #dth = (np.pi/2)/NBTHETA
    #tabTh = np.linspace(dth/2, np.pi/2-dth/2, NBTHETA, dtype='float64')

    # zenith angles PI
    #dth = (np.pi)/NBTHETA
    #tabTh = np.linspace(dth/2, np.pi-dth/2, NBTHETA, dtype='float64')

    # zenith angles SZA_MAX
    dth = (SZA_MAX/180.*np.pi)/NBTHETA
    tabTh = np.linspace(dth/2, SZA_MAX/180.*np.pi-dth/2, NBTHETA, dtype='float64')

    # azimuth angles
    #dphi = np.pi/NBPHI
    #tabPhi = np.linspace(dphi/2, np.pi-dphi/2, NBPHI, dtype='float64')
    dphi = 2*np.pi/NBPHI
    tabPhi = np.linspace(0., 2*np.pi-dphi, NBPHI, dtype='float64')


    # solid angles
    tabds = np.sin(tabTh) * dth * dphi

    # normalize to 1
    tabOmega = tabds/(sum(tabds)*NBPHI)
    if (SUN_DISC !=0) : tabOmega[:]= 2*np.pi * (1. - np.cos(SUN_DISC*np.pi/180))

    return tabTh, tabPhi, tabOmega


def finalize(tabPhotonsTot, tabDistTot, tabHistTot, wl, NPhotonsInTot, errorcount, NPhotonsOutTot,
             OUTPUT_LAYERS, tabTransDir, SIM, attrs, prof_atm, prof_oc,
             sigma, THVDEG, HORIZ, le=None, flux=None,
             back=False, SZA_MAX=90., SUN_DISC=0, hist=False, cMatVisuRecep = None,
             losses = None, dicSTP = None, matCats=None, matLoss=None):
    '''
    create and return the final output
    '''
    (NLVL,NPSTK,NSENSOR,NLAM,NBTHETA,NBPHI) = tabPhotonsTot.shape
    #(NLVL,NPSTK,NLAM,NBTHETA,NBPHI) = tabPhotonsTot.shape

    # normalization in case of radiance
    # (broadcast everything to dimensions (LVL,NPSTK,SENSOR,LAM,THETA,PHI))
    ## (broadcast everything to dimensions (LVL,NPSTK,LAM,THETA,PHI))
    norm_npho = NPhotonsInTot.reshape((1,1,NSENSOR,NLAM,1,1))
    #norm_npho = NPhotonsInTot.reshape((1,1,-1,1,1))
    if flux is None:
        if le!=None : 
            tabTh = le['th']
            tabPhi = le['phi']
            norm_geo =  1. 
        else : 
            tabTh, tabPhi, tabOmega = calcOmega(NBTHETA, NBPHI, SZA_MAX=SZA_MAX, SUN_DISC=SUN_DISC)
            if HORIZ==1 : norm_geo = 2.0 * tabOmega.reshape((1,1,-1,1)) * np.cos(tabTh).reshape((1,1,-1,1))
            else :  norm_geo = 2.0 * tabOmega.reshape((1,1,-1,1)) 
    else:
        norm_geo = 1.
        tabTh, tabPhi, _ = calcOmega(NBTHETA, NBPHI, SZA_MAX=SZA_MAX, SUN_DISC=SUN_DISC)

    # normalization
    tabFinal = tabPhotonsTot.astype('float64')/(norm_geo*norm_npho)
    tabDistFinal = tabDistTot.astype('float64')
    if hist : tabHistFinal = tabHistTot

    # swapaxes : (th, phi) -> (phi, theta)
    tabFinal = tabFinal.swapaxes(4,5)
    if len(tabDistFinal) >1 : tabDistFinal = tabDistFinal.swapaxes(3,4)
    if hist : tabHistFinal = tabHistFinal.swapaxes(3,4)
    NPhotonsOutTot = NPhotonsOutTot.swapaxes(3,4)
    if sigma is not None:
        sigma /= norm_geo
        sigma = sigma.swapaxes(4,5)


    #
    # create the MLUT object
    #
    m = MLUT()

    # add the axes
    axnames  = ['Zenith angles']
    axnames2 = ['None', 'Zenith angles']
    if hist : 
        axnames3 = ['N', 'Info', 'Zenith angles']
        m.add_dataset('Nphotons_in',  NPhotonsInTot)
        #m.add_dataset('Nphotons_out',  NPhotonsOutTot)
    iphi     = slice(None)
    #if hist : axnames3 = ['None', 'None', 'Azimuth angles', 'Zenith angles']
    m.set_attr('zip', 'False')

    if le is not None:
        if 'zip' in le:
            if le['zip'] : 
                m.set_attr('zip', 'True')
                iphi = 0
            else:
                axnames.insert(0, 'Azimuth angles')
                axnames2.insert(1,'Azimuth angles')
                if hist : axnames3.insert(2, 'Azimuth angles')
        else:
            axnames.insert(0, 'Azimuth angles')
            axnames2.insert(1,'Azimuth angles')
            if hist : axnames3.insert(2, 'Azimuth angles')
    else:
        axnames.insert(0, 'Azimuth angles')
        axnames2.insert(1,'Azimuth angles')
        if hist : axnames3.insert(2, 'Azimuth angles')

    m.add_axis('Zenith angles', tabTh*180./np.pi)
    m.add_axis('Azimuth angles', tabPhi*180./np.pi)
    
    if NLAM > 1:
        m.add_axis('wavelength', wl)
        ilam = slice(None)
        axnames.insert(0, 'wavelength')
    else:
        m.set_attr('wavelength', str(wl))
        ilam = 0

    if NSENSOR > 1:
        m.add_axis('sensor index', np.arange(NSENSOR))
        isen = slice(None)
        axnames.insert(0, 'sensor index')
        axnames2.insert(1, 'sensor index')
        if hist: axnames3.insert(2, 'sensor index')
    else:
        isen=0

    m.add_dataset('I_up (TOA)', tabFinal[UPTOA,0,isen,ilam,iphi,:], axnames)
    m.add_dataset('Q_up (TOA)', tabFinal[UPTOA,1,isen,ilam,iphi,:], axnames)
    m.add_dataset('U_up (TOA)', tabFinal[UPTOA,2,isen,ilam,iphi,:], axnames)
    m.add_dataset('V_up (TOA)', tabFinal[UPTOA,3,isen,ilam,iphi,:], axnames)
    if sigma is not None:
        m.add_dataset('I_stdev_up (TOA)', sigma[UPTOA,0,isen,ilam,iphi,:], axnames)
        m.add_dataset('Q_stdev_up (TOA)', sigma[UPTOA,1,isen,ilam,iphi,:], axnames)
        m.add_dataset('U_stdev_up (TOA)', sigma[UPTOA,2,isen,ilam,iphi,:], axnames)
        m.add_dataset('V_stdev_up (TOA)', sigma[UPTOA,3,isen,ilam,iphi,:], axnames)
    m.add_dataset('N_up (TOA)', NPhotonsOutTot[UPTOA,isen,ilam,iphi,:], axnames)
    if len(tabDistFinal) > 1: m.add_dataset('cdist_up (TOA)', tabDistFinal[UPTOA,:,isen,iphi,:], axnames2)
    if hist : m.add_dataset('disth_up (TOA)', tabHistFinal[:,:,isen,iphi,:],axnames3)
    #if hist : m.add_dataset('disth_down (0+)', tabHistFinal[UPTOA,:,:,isen,iphi,:],axnames3)

    if OUTPUT_LAYERS & 1:
        m.add_dataset('I_down (0+)', tabFinal[DOWN0P,0,isen,ilam,iphi,:], axnames)
        m.add_dataset('Q_down (0+)', tabFinal[DOWN0P,1,isen,ilam,iphi,:], axnames)
        m.add_dataset('U_down (0+)', tabFinal[DOWN0P,2,isen,ilam,iphi,:], axnames)
        m.add_dataset('V_down (0+)', tabFinal[DOWN0P,3,isen,ilam,iphi,:], axnames)
        if sigma is not None:
            m.add_dataset('I_stdev_down (0+)', sigma[DOWN0P,0,isen,ilam,iphi,:], axnames)
            m.add_dataset('Q_stdev_down (0+)', sigma[DOWN0P,1,isen,ilam,iphi,:], axnames)
            m.add_dataset('U_stdev_down (0+)', sigma[DOWN0P,2,isen,ilam,iphi,:], axnames)
            m.add_dataset('V_stdev_down (0+)', sigma[DOWN0P,3,isen,ilam,iphi,:], axnames)
        m.add_dataset('N_down (0+)', NPhotonsOutTot[DOWN0P,isen,ilam,iphi,:], axnames)
        if len(tabDistFinal) > 1: m.add_dataset('cdist_down (0+)', tabDistFinal[DOWN0P,:,isen,iphi,:],axnames2)
        #if hist : m.add_dataset('disth_down (0+)', tabHistFinal[DOWN0P,:,:,isen,iphi,:],axnames3)

        m.add_dataset('I_up (0-)', tabFinal[UP0M,0,isen,ilam,iphi,:], axnames)
        m.add_dataset('Q_up (0-)', tabFinal[UP0M,1,isen,ilam,iphi,:], axnames)
        m.add_dataset('U_up (0-)', tabFinal[UP0M,2,isen,ilam,iphi,:], axnames)
        m.add_dataset('V_up (0-)', tabFinal[UP0M,3,isen,ilam,iphi,:], axnames)
        if sigma is not None:
            m.add_dataset('I_stdev_up (0-)', sigma[UP0M,0,isen,ilam,iphi,:], axnames)
            m.add_dataset('Q_stdev_up (0-)', sigma[UP0M,1,isen,ilam,iphi,:], axnames)
            m.add_dataset('U_stdev_up (0-)', sigma[UP0M,2,isen,ilam,iphi,:], axnames)
            m.add_dataset('V_stdev_up (0-)', sigma[UP0M,3,isen,ilam,iphi,:], axnames)
        m.add_dataset('N_up (0-)', NPhotonsOutTot[UP0M,isen,ilam,iphi,:], axnames)
        if len(tabDistFinal) > 1: m.add_dataset('cdist_up (0-)', tabDistFinal[UP0M,:,isen,iphi,:],axnames2)
        #if hist : m.add_dataset('disth_up (0-)', tabHistFinal[UP0M,:,:,isen,iphi,:],axnames3)

    if OUTPUT_LAYERS & 2:
        m.add_dataset('I_down (0-)', tabFinal[DOWN0M,0,isen,ilam,iphi,:], axnames)
        m.add_dataset('Q_down (0-)', tabFinal[DOWN0M,1,isen,ilam,iphi,:], axnames)
        m.add_dataset('U_down (0-)', tabFinal[DOWN0M,2,isen,ilam,iphi,:], axnames)
        m.add_dataset('V_down (0-)', tabFinal[DOWN0M,3,isen,ilam,iphi,:], axnames)
        if sigma is not None:
            m.add_dataset('I_stdev_down (0-)', sigma[DOWN0M,0,isen,ilam,iphi,:], axnames)
            m.add_dataset('Q_stdev_down (0-)', sigma[DOWN0M,1,isen,ilam,iphi,:], axnames)
            m.add_dataset('U_stdev_down (0-)', sigma[DOWN0M,2,isen,ilam,iphi,:], axnames)
            m.add_dataset('V_stdev_down (0-)', sigma[DOWN0M,3,isen,ilam,iphi,:], axnames)
        m.add_dataset('N_down (0-)', NPhotonsOutTot[DOWN0M,isen,ilam,iphi,:], axnames)
        if len(tabDistFinal) > 1: m.add_dataset('cdist_down (0-)', tabDistFinal[DOWN0M,:,isen,iphi,:],axnames2)
        #if hist : m.add_dataset('disth_down (0-)', tabHistFinal[DOWN0M,:,:,isen,iphi,:],axnames3)

        m.add_dataset('I_up (0+)', tabFinal[UP0P,0,isen,ilam,iphi,:], axnames)
        m.add_dataset('Q_up (0+)', tabFinal[UP0P,1,isen,ilam,iphi,:], axnames)
        m.add_dataset('U_up (0+)', tabFinal[UP0P,2,isen,ilam,iphi,:], axnames)
        m.add_dataset('V_up (0+)', tabFinal[UP0P,3,isen,ilam,iphi,:], axnames)
        if sigma is not None:
            m.add_dataset('I_stdev_up (0+)', sigma[UP0P,0,isen,ilam,iphi,:], axnames)
            m.add_dataset('Q_stdev_up (0+)', sigma[UP0P,1,isen,ilam,iphi,:], axnames)
            m.add_dataset('U_stdev_up (0+)', sigma[UP0P,2,isen,ilam,iphi,:], axnames)
            m.add_dataset('V_stdev_up (0+)', sigma[UP0P,3,isen,ilam,iphi,:], axnames)
        m.add_dataset('N_up (0+)', NPhotonsOutTot[UP0P,isen,ilam,iphi,:], axnames)
        if len(tabDistFinal) > 1: m.add_dataset('cdist_up (0+)', tabDistFinal[UP0P,:,isen,iphi,:],axnames2)
        #if hist : m.add_dataset('disth_up (0+)', tabHistFinal[UP0P,:,:,isen,iphi,:],axnames3)

        m.add_dataset('I_down (B)', tabFinal[DOWNB,0,isen,ilam,iphi,:], axnames)
        m.add_dataset('Q_down (B)', tabFinal[DOWNB,1,isen,ilam,iphi,:], axnames)
        m.add_dataset('U_down (B)', tabFinal[DOWNB,2,isen,ilam,iphi,:], axnames)
        m.add_dataset('V_down (B)', tabFinal[DOWNB,3,isen,ilam,iphi,:], axnames)
        if sigma is not None:
            m.add_dataset('I_stdev_down (B)', sigma[DOWNB,0,isen,ilam,iphi,:], axnames)
            m.add_dataset('Q_stdev_down (B)', sigma[DOWNB,1,isen,ilam,iphi,:], axnames)
            m.add_dataset('U_stdev_down (B)', sigma[DOWNB,2,isen,ilam,iphi,:], axnames)
            m.add_dataset('V_stdev_down (B)', sigma[DOWNB,3,isen,ilam,iphi,:], axnames)
        m.add_dataset('N_down (B)', NPhotonsOutTot[DOWNB,isen,ilam,iphi,:], axnames)
        if len(tabDistFinal) > 1: m.add_dataset('cdist_down (B)', tabDistFinal[DOWNB,:,isen,iphi,:],axnames2)
        #if hist : m.add_dataset('disth_down (B)', tabHistFinal[DOWNB,:,:,isen,iphi,:],axnames3)


    # direct transmission
    m.add_dataset('direct transmission', tabTransDir,
                   axnames=['wavelength'])

    # write atmospheric profiles
    if prof_atm is not None:
        m.add_lut(prof_atm['n_atm'])
        m.add_lut(prof_atm['OD_r'])
        m.add_lut(prof_atm['OD_p'])
        m.add_lut(prof_atm['OD_g'])
        m.add_lut(prof_atm['OD_atm'])
        m.add_lut(prof_atm['OD_sca_atm'])
        m.add_lut(prof_atm['OD_abs_atm'])
        m.add_lut(prof_atm['pmol_atm'])
        m.add_lut(prof_atm['ssa_atm'])
        m.add_lut(prof_atm['ssa_p_atm'])
        if 'phase_atm' in prof_atm.datasets():
            m.add_lut(prof_atm['phase_atm'])
            m.add_lut(prof_atm['iphase_atm'])
        if 'pine_atm' in prof_atm.datasets():
            m.add_lut(prof_atm['pine_atm'])
            m.add_lut(prof_atm['FQY1_atm'])

        if 'neighbour_atm' in prof_atm.datasets():
            m.add_lut(prof_atm['iopt_atm'])
            m.add_lut(prof_atm['iabs_atm'])
            m.add_lut(prof_atm['pmin_atm'])
            m.add_lut(prof_atm['pmax_atm'])
            m.add_lut(prof_atm['neighbour_atm'])

    # write ocean profiles
    if prof_oc is not None:
        m.add_lut(prof_oc['OD_w'])
        m.add_lut(prof_oc['OD_p_oc'])
        m.add_lut(prof_oc['OD_y'])
        m.add_lut(prof_oc['OD_oc'])
        m.add_lut(prof_oc['OD_sca_oc'])
        m.add_lut(prof_oc['OD_abs_oc'])
        m.add_lut(prof_oc['pmol_oc'])
        m.add_lut(prof_oc['ssa_oc'])
        if 'ssa_w' in prof_oc.datasets():
            m.add_lut(prof_oc['ssa_w'])
        if 'ssa_p_oc' in prof_oc.datasets():
            m.add_lut(prof_oc['ssa_p_oc'])
        if 'phase_oc' in prof_oc.datasets():
            m.add_lut(prof_oc['phase_oc'])
            m.add_lut(prof_oc['iphase_oc'])
        if 'pine_oc' in prof_oc.datasets():
            m.add_lut(prof_oc['pine_oc'])
            m.add_lut(prof_oc['FQY1_oc'])
        m.add_lut(prof_oc['albedo_seafloor'])

        if 'neighbour_oc' in prof_oc.datasets():
            m.add_lut(prof_oc['iopt_oc'])
            m.add_lut(prof_oc['iabs_oc'])
            m.add_lut(prof_oc['pmin_oc'])
            m.add_lut(prof_oc['pmax_oc'])
            m.add_lut(prof_oc['neighbour_oc'])

    # write the error )count
    err = errorcount.get()
    for i, d in enumerate([
            'ERROR_THETA',
            'ERROR_CASE',
            'ERROR_VXY',
            'ERROR_MAX_LOOP',
            ]):
        m.set_attr(d, err[i])

    # write attributes
    for k, v in list(attrs.items()):
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

    if (cMatVisuRecep is not None):
        m.add_dataset('C_Receiver', cMatVisuRecep[0][:][:], ['Horizontal pixel', 'Vertical pixel'])
        m.add_dataset('C1_Receiver', cMatVisuRecep[1][:][:], ['Horizontal pixel', 'Vertical pixel'])
        m.add_dataset('C2_Receiver', cMatVisuRecep[2][:][:], ['Horizontal pixel', 'Vertical pixel'])
        m.add_dataset('C3_Receiver', cMatVisuRecep[3][:][:], ['Horizontal pixel', 'Vertical pixel'])
        m.add_dataset('C4_Receiver', cMatVisuRecep[4][:][:], ['Horizontal pixel', 'Vertical pixel'])
        m.add_dataset('C5_Receiver', cMatVisuRecep[5][:][:], ['Horizontal pixel', 'Vertical pixel'])
        m.add_dataset('C6_Receiver', cMatVisuRecep[6][:][:], ['Horizontal pixel', 'Vertical pixel'])
        m.add_dataset('C7_Receiver', cMatVisuRecep[7][:][:], ['Horizontal pixel', 'Vertical pixel'])
        m.add_dataset('C8_Receiver', cMatVisuRecep[8][:][:], ['Horizontal pixel', 'Vertical pixel'])        

    if (matCats is not None):
        m.add_dataset('cat_PhNb', matCats[:,0], ['CAT'], attrs={'description':
            'Photons number received (sum of all cats indice 0) +\n' + \
            '       for each cats (from indice 1 to 8)'})
        m.add_dataset('cat_w', matCats[:,1], ['CAT'], attrs={'description':
            'Sum of photons weight (sum of all cats indice 0) +\n' + \
            '       for each cats (from indice 1 to 8)'})
        m.add_dataset('cat_w2', matCats[:,2], ['CAT'], attrs={'description':
            'Sum of Photons weight^2 (sum of all cats indice 0) +\n' + \
            '       for each cats (from indice 1 to 8)'})
        m.add_dataset('cat_irr', matCats[:,3], ['CAT'], attrs={'description':
            'Irradiance at the receiver, must multiply by E_TOA, result is in W/m^2\n' + \
            '       (sum of all cats indice 0) for each cats (from indice 1 to 8)'})
        m.add_dataset('cat_errAbs', matCats[:,4], ['CAT'], attrs={'description':
            'Relative error of the intensity (sum of all cats indice 0) +\n' + \
            '       for each cats (from indice 1 to 8)'})
        m.add_dataset('cat_err%', matCats[:,5], ['CAT'], attrs={'description':
            'Absolute error of the irradiance,  must multiply by E_TOA,\n' + \
            'result is in W/m^2 (sum of all cats indice 0) + \nfor each cats (from indice 1 to 8)'})

    if (losses is not None):
        
        # Find the extinction between TOA and heliostats
        if ((dicSTP["LPH"] is None) or (dicSTP["LPR"] is None)):
            # find the atm layer where the mean heliostats z altitude is located
            Ci = 0
            while(prof_atm.axis('z_atm')[Ci] > dicSTP["MZAlt_H"]):
                Ci += 1
        
            # extinction optical thickness distance between start to end (here end is the mean z altitude of heliostats)
            tau_ext = (prof_atm['OD_atm'].data[0][Ci] -  prof_atm['OD_atm'].data[0][Ci-1]) * (dicSTP["MZAlt_H"]/prof_atm.axis('z_atm')[Ci-1])
            tau_ext = prof_atm['OD_atm'].data[0][Ci] - tau_ext

            # Beer-Lamber law to find the transmisttance
            Tr_tau = np.exp(-abs(tau_ext/-dicSTP["vSun"].z))
        else: # This is under developement ->
            SUM_Tr_tau = 0
            for i in range (0, len(dicSTP["LPH"])):
                time = (120. - dicSTP["LPH"][i].z)/dicSTP["vSun"].z
                PSUN_TOA = dicSTP["LPH"][i] + (dicSTP["vSun"]*time)
                SUM_Tr_tau += findExtinction(PSUN_TOA, dicSTP["LPH"][i], prof_atm)
            Tr_tau = SUM_Tr_tau/len(dicSTP["LPH"])
        
        # theoric computation of the total power collected by all the heliostats (here the sun flux density is equal to 1) 
        P_pyt = Tr_tau*dicSTP["totS_H"]    

        m.add_dataset('n_tr', np.array([Tr_tau], dtype=np.float64), ['transmission from TOA to H'])
        if (back):
            P_cud = matCats[2, 3]*dicSTP["SREC"] # power collected by the reciever
            ncos = max(0, min(losses[2]/losses[1], 1))
            nref = max(0, min(losses[2]/losses[0], 1))
            # === Under developement ->
            if ((dicSTP["LPH"] is not None) and (dicSTP["LPR"] is not None)):
                naatm=0
                for i in range (len(dicSTP["LPH"])):
                    naatm += findExtinction(dicSTP["LPH"][i], dicSTP["LPR"][0], prof_atm)
                naatm /= len(dicSTP["LPH"])
                m.add_dataset('n_aatm', np.array([naatm], dtype=np.float64), ['Atmospheric Approx Efficiency'])
            # ===
            nsbsa = P_cud/(P_pyt*ncos*nref)
            m.add_dataset('n_sbsa', np.array([nsbsa], dtype=np.float64), ['Shadow, blocking, spillage, atmos Efficiencies'])
            m.add_dataset('n_opt', np.array([ncos*nref*nsbsa], dtype=np.float64), ['Optical Efficiency'])
            k = P_cud/(matCats[2,1]*P_pyt)
        else:
            # collecte the weight from the photons reaching the heliostats if there is not cos effect (cuda), then convert to power
            P_cud = (losses[1]*dicSTP["surfTOA"])/float(NPhotonsInTot)
            # we can now compute the shadow efficiency
            nsha = max(0, min(P_cud/P_pyt, 1))
            ncos = max(0, min(losses[0]/losses[1], 1))
            nref = max(0, min(losses[2]/losses[0], 1))
            nspi = max(0, min(losses[3]/losses[2], 1))
            nblo = max(0, min((losses[3]-losses[4])/losses[3], 1))
            natm = max(0, min(dicSTP["wRec"]/(losses[3]-losses[4]), 1))
            m.add_dataset('n_sha', np.array([nsha], dtype=np.float64), ['Shadow Efficiency'])
            m.add_dataset('n_spi', np.array([nspi], dtype=np.float64), ['Spillage Efficiency'])
            m.add_dataset('n_blo', np.array([nblo], dtype=np.float64), ['blocking Efficiency'])
            m.add_dataset('n_opt', np.array([nsha*ncos*nref*nspi*natm*nblo], dtype=np.float64), ['Optical Efficiency'])
            m.add_dataset('n_atmo', np.array([natm], dtype=np.float64), ['Atmospheric Efficiency'])
            k = dicSTP["surfTOA"]/(float(NPhotonsInTot)*P_pyt)
        m.add_dataset('n_cte', np.array([k], dtype=np.float64), ['Constante used for losses error'])
        m.add_dataset('n_cos', np.array([ncos], dtype=np.float64), ['Cosine Efficiency'])
        m.add_dataset('n_ref', np.array([nref], dtype=np.float64), ['Reflection Efficiency'])
        m.add_dataset('wLoss', np.array(matLoss[:,0], dtype=np.float64), ['Vector with opt weights'])
        m.add_dataset('wLoss2', np.array(matLoss[:,1], dtype=np.float64), ['Vector with opt weights²'])

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
    P22 = T_demi*(DELTA+DELTA_PRIM)
    P12 = T_demi*DELTA_PRIM
    P33bis = T_demi*DELTA
    P44bis = P33bis*DELTA_SECO

    # parameters equally spaced in scattering probabiliy [0, 1]
    pha['p_P11'][:] = T_demi*(DELTA*cTh2[:] + DELTA_PRIM)
    pha['p_P12'][:] = P12
    pha['p_P22'][:] = P22
    pha['p_P33'][:] = P33bis*cTh[:] # U
    pha['p_P44'][:] = P44bis*cTh[:] # V
    pha['p_ang'][:] = theta[:] # angle

    # parameters equally spaced in scattering angle [0, 180]
    pha['a_P11'][:] = T_demi*(DELTA*cTh2LE[:] + DELTA_PRIM) 
    pha['a_P12'][:] = P12
    pha['a_P22'][:] = P22
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
        # f1 = interp1d(angles, phase[1,:])
        # f2 = interp1d(angles, phase[0,:])
        f1 = interp1d(angles, phase[0,:])
        f2 = interp1d(angles, phase[1,:])
        f3 = interp1d(angles, phase[2,:])
        f4 = interp1d(angles, phase[3,:])

        # parameters equally spaced in scattering probability
        # phase_H['p_P11'][idx, :] = interp1d(scum, phase[1,:])(z)  # I par P11
        # phase_H['p_P22'][idx, :] = interp1d(scum, phase[0,:])(z)  # I per P22
        phase_H['p_P11'][idx, :] = interp1d(scum, phase[0,:])(z)  # I par P11
        phase_H['p_P22'][idx, :] = interp1d(scum, phase[1,:])(z)  # I per P22
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


def InitConst(surf, env, NATM, NATM_ABS, NOCE, NOCE_ABS, mod,
              NBPHOTONS, NBLOOP, THVDEG, DEPO,
              XBLOCK, XGRID,NLAM, SIM, NF,
              NBTHETA, NBPHI, OUTPUT_LAYERS,
              RTER, LE, ZIP, FLUX, FFS, DIRECT, NLVL, NPSTK, NWLPROBA, BEER, SMAX, RR, 
              WEIGHTRR, NLOW, NJAC, NSENSOR, REFRAC, HORIZ, SZA_MAX, SUN_DISC, cusL, nObj, nGObj, nRObj,
              Pmin_x, Pmin_y, Pmin_z, Pmax_x, Pmax_y, Pmax_z, IsAtm, TC, nbCx, nbCy, vSun, HIST) :
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

    PZd = 120.
    tTemp = PZd/-vSun.z
    PXd = -vSun.x * tTemp
    PYd = -vSun.y * tTemp

    def copy_to_device(name, scalar, dtype):
        cuda.memcpy_htod(mod.get_global(name)[0], np.array([scalar], dtype=dtype))

    # copy constants to device
    copy_to_device('NBLOOPd', NBLOOP, np.uint32)
    copy_to_device('NOCEd', NOCE, np.int32)
    copy_to_device('NOCE_ABSd', NOCE_ABS, np.int32)
    copy_to_device('OUTPUT_LAYERSd', OUTPUT_LAYERS, np.uint32)
    copy_to_device('NF', NF, np.uint32)
    copy_to_device('NATMd', NATM, np.int32)
    copy_to_device('NATM_ABSd', NATM_ABS, np.int32)
    copy_to_device('XBLOCKd', XBLOCK, np.int32)
    copy_to_device('YBLOCKd', 1, np.int32)
    copy_to_device('XGRIDd', XGRID, np.int32)
    copy_to_device('YGRIDd', 1, np.int32)
    copy_to_device('NBTHETAd', NBTHETA, np.int32)
    copy_to_device('NBPHId', NBPHI, np.int32)
    copy_to_device('NLAMd', NLAM, np.int32)
    copy_to_device('SIMd', SIM, np.int32)
    copy_to_device('LEd', LE, np.int32)
    copy_to_device('ZIPd', ZIP, np.int32)
    copy_to_device('FLUXd', FLUX, np.int32)
    copy_to_device('FFSd', 1 if FFS else 0, np.int32)
    copy_to_device('DIRECTd', 1 if DIRECT else 0, np.int32)
    #copy_to_device('MId', MI, np.int32)
    copy_to_device('NLVLd', NLVL, np.int32)
    copy_to_device('NPSTKd', NPSTK, np.int32)
    copy_to_device('BEERd', BEER, np.int32)
    copy_to_device('SMAXd', SMAX, np.int32)
    copy_to_device('RRd', RR, np.int32)
    copy_to_device('WEIGHTRRd', WEIGHTRR, np.float32)
    copy_to_device('NLOWd', NLOW, np.int32)
    copy_to_device('NJACd', NJAC, np.int32)
    copy_to_device('HISTd', HIST, np.int32)
    copy_to_device('NSENSORd', NSENSOR, np.int32)
    if surf != None:
        copy_to_device('SURd', surf.dict['SUR'], np.int32)
        copy_to_device('BRDFd', surf.dict['BRDF'], np.int32)
        copy_to_device('DIOPTREd', surf.dict['DIOPTRE'], np.int32)
        copy_to_device('WINDSPEEDd', surf.dict['WINDSPEED'], np.float32)
        copy_to_device('NH2Od', surf.dict['NH2O'], np.float32)
        copy_to_device('WAVE_SHADOWd', surf.dict['WAVE_SHADOW'], np.int32)
        copy_to_device('SINGLEd', surf.dict['SINGLE'], np.int32)
    if env != None:
        copy_to_device('ENVd', env.dict['ENV'], np.int32)
        copy_to_device('ENV_SIZEd', env.dict['ENV_SIZE'], np.float32)
        copy_to_device('X0d', env.dict['X0'], np.float32)
        copy_to_device('Y0d', env.dict['Y0'], np.float32)
    copy_to_device('STHVd', STHV, np.float32)
    copy_to_device('CTHVd', CTHV, np.float32)
    copy_to_device('RTER', RTER, np.float32)
    copy_to_device('NWLPROBA', NWLPROBA, np.int32)
    copy_to_device('REFRACd', REFRAC, np.int32)
    copy_to_device('HORIZd', HORIZ, np.int32)
    copy_to_device('SZA_MAXd', SZA_MAX, np.float32)
    copy_to_device('SUN_DISCd', SUN_DISC, np.float32)
    # copy en rapport avec les objets :
    if nObj != 0:
        copy_to_device('nObj', nObj, np.int32)
        copy_to_device('nGObj', nGObj, np.int32)
        copy_to_device('nRObj', nRObj, np.int32)
        copy_to_device('Pmin_x', Pmin_x, np.float32)
        copy_to_device('Pmin_y', Pmin_y, np.float32)
        copy_to_device('Pmin_z', Pmin_z, np.float32)
        copy_to_device('Pmax_x', Pmax_x, np.float32)
        copy_to_device('Pmax_y', Pmax_y, np.float32)
        copy_to_device('Pmax_z', Pmax_z, np.float32)
        copy_to_device('IsAtm', IsAtm, np.int32)
        copy_to_device('DIRSXd', vSun.x, np.float32)
        copy_to_device('DIRSYd', vSun.y, np.float32)
        copy_to_device('DIRSZd', vSun.z, np.float32)
        copy_to_device('PXd', PXd, np.float32)
        copy_to_device('PYd', PYd, np.float32)
        copy_to_device('PZd', PZd, np.float32)
        if TC is not None:
            copy_to_device('TCd', TC, np.float32)
            copy_to_device('nbCx', nbCx, np.int32)
            copy_to_device('nbCy', nbCy, np.int32)
        if (  (cusL != None) and (cusL.dict['LMODE'] == "RF")  ):
            copy_to_device('LMODEd', 1, np.int32)
        if (  (cusL != None) and (cusL.dict['LMODE'] == "FF")  ):
            copy_to_device('CFXd', cusL.dict['CFX'], np.float32)
            copy_to_device('CFYd', cusL.dict['CFY'], np.float32)
            copy_to_device('CFTXd', cusL.dict['CFTX'], np.float32)
            copy_to_device('CFTYd', cusL.dict['CFTY'], np.float32)
            copy_to_device('ALDEGd', cusL.dict['FOV'], np.float32)
            copy_to_device('TYPEd', cusL.dict['TYPE'], np.int32)
            copy_to_device('LMODEd', 2, np.int32)
        if (  (cusL != None) and (cusL.dict['LMODE'] == "B" or cusL.dict['LMODE'] == "BR")  ):
            copy_to_device('THDEGd', cusL.dict['THDEG'], np.float32)
            copy_to_device('PHDEGd', cusL.dict['PHDEG'], np.float32)
            copy_to_device('ALDEGd', cusL.dict['ALDEG'], np.float32)
            copy_to_device('TYPEd', cusL.dict['TYPE'], np.int32)
        if (  (cusL != None) and (cusL.dict['LMODE'] == "B")  ):    
            copy_to_device('LMODEd', 3, np.int32)
        if (  (cusL != None) and (cusL.dict['LMODE'] == "BR")  ):
            copy_to_device('LMODEd', 4, np.int32)
        if (cusL == None):
            copy_to_device('LMODEd', 0, np.int32)
        
def init_profile(wl, prof, kind):
    '''
    take the profile as a MLUT, and setup the gpu structure
    kind = 'atm' or 'oc' for atmosphere or ocean
    '''

    NREF = len(prof.axis('z_'+kind))
    # reformat to smartg format
    if 'iopt_'+kind in prof.datasets(): 
        NLAY = len(prof['OD_'+kind].data[0,:])
    else:
        NLAY = len(prof.axis('z_'+kind))
    shp = (len(wl), NLAY)
    prof_gpu = np.zeros(shp, dtype=type_Profile, order='C')

    if kind == "oc":
        if 'iopt_oc' not in prof.datasets():
            prof_gpu['z'][0,:] = prof.axis('z_'+kind)  * 1e-3 # to Km
            cell_gpu = np.zeros(1, dtype=type_Cell)
        else: 
            cell_gpu = np.zeros(len(prof['iopt_oc'].data), dtype=type_Cell)
        prof_gpu['n'][0,:] = 1.34;
    else:
        if 'iopt_atm' not in prof.datasets():
            prof_gpu['z'][0,:] = prof.axis('z_'+kind)
            prof_gpu['n'][:,:] = prof['n_'+kind].data[...]
            cell_gpu = np.zeros(1, dtype=type_Cell)
        else:
            cell_gpu = np.zeros(len(prof['iopt_atm'].data), dtype=type_Cell)
    prof_gpu['z'][1:,:] = -999.      # other wavelengths are NaN

    prof_gpu['OD'][:,:] = prof['OD_'+kind].data[...]
    prof_gpu['OD_sca'][:] = prof['OD_sca_'+kind].data[...]
    prof_gpu['OD_abs'][:] = prof['OD_abs_'+kind].data[...]
    prof_gpu['pmol'][:] = prof['pmol_'+kind].data[...]
    prof_gpu['ssa'][:] = prof['ssa_'+kind].data[...]
    prof_gpu['pine'][:] = prof['pine_'+kind].data[...]
    prof_gpu['FQY1'][:] = prof['FQY1_'+kind].data[...]
    if 'iphase_'+kind in prof.datasets():
        prof_gpu['iphase'][:] = prof['iphase_'+kind].data[...]

    if len(cell_gpu)>1:
        cell_gpu['iopt'][:]  = prof['iopt_'+kind].data[...]
        cell_gpu['iabs'][:]  = prof['iabs_'+kind].data[...]
        cell_gpu['pminx'][:] = prof['pmin_'+kind].data[0,:]
        cell_gpu['pminy'][:] = prof['pmin_'+kind].data[1,:]
        cell_gpu['pminz'][:] = prof['pmin_'+kind].data[2,:]
        cell_gpu['pmaxx'][:] = prof['pmax_'+kind].data[0,:]
        cell_gpu['pmaxy'][:] = prof['pmax_'+kind].data[1,:]
        cell_gpu['pmaxz'][:] = prof['pmax_'+kind].data[2,:]
        cell_gpu['neighbour1'][:] = prof['neighbour_'+kind].data[0,:]
        cell_gpu['neighbour2'][:] = prof['neighbour_'+kind].data[1,:]
        cell_gpu['neighbour3'][:] = prof['neighbour_'+kind].data[2,:]
        cell_gpu['neighbour4'][:] = prof['neighbour_'+kind].data[3,:]
        cell_gpu['neighbour5'][:] = prof['neighbour_'+kind].data[4,:]
        cell_gpu['neighbour6'][:] = prof['neighbour_'+kind].data[5,:]
        
    return to_gpu(prof_gpu), to_gpu(cell_gpu)



def multi_profiles(profs, kind='atm'):
    '''
    Internal reorganization of list of profiles for Jacobian (with finite differences) or sensitivities

    Input: 
        profs : list of profiles (coming either from atm.calc() or water.calc())
        kind  : atmospheric 'atm' or oceanic 'oc'
    ''' 
    
    first=profs[0]
    pro=MLUT()
    for (axname, axis) in first.axes.items():
        if 'wavelength' in axname: 
            axis=list(axis)*len(profs)
        pro.add_axis(axname, axis)

    for d in first.datasets():
        if 'iphase' in d :
            imax=0
            k=0
            for M in profs:            
                im = np.unique(M[d].data).max() + 1
                if k==0 : data =  M[d].data[:] + imax
                else: data = np.concatenate((data, M[d].data[:] + imax), axis=0)
                imax=im 
                k=k+1
            pro.add_dataset(d, data, ['wavelength', 'z_'+kind])
        else:
            if d==('phase_'+kind) :
                imax=0
                k=0
                for M in profs:            
                    if k==0 : data =  M[d].data[:]
                    else: data = np.concatenate((data, M[d].data[:]), axis=0)
                    k=k+1
                pro.add_dataset(d, data, ['iphase', 'stk', 'theta_'+kind])
            else:
                imax=0
                k=0
                for M in profs:            
                    if k==0 : data =  M[d].data[:]
                    else: data = np.concatenate((data, M[d].data[:]), axis=0)
                    k=k+1
                if data.ndim==2 : pro.add_dataset(d, data, ['wavelength', 'z_'+kind])
                if data.ndim==1 : pro.add_dataset(d, data, ['wavelength'])
    return pro

def reduce_diff(m, varnames, delta=None):
    '''
    Reduce finite differences run in ALIS mode, to obtain Jacobians (with finite differences) or sensitivities

    Input: 
        m : MLUT ouput of SMART-G
        varnames  : list of variable names for which sensitivity is calculated
    Keyword:
        delta : eventually list of perturbation (float) for each variable, Jacobians are calculated instead of sensitivities 
    '''

    res=MLUT()
    NDIFF = len(varnames)
    NWL   = m.axis('wavelength').shape
    NW    = int(NWL[0]/(NDIFF+1))
    
    for l in m:
        for pref in ['I_','Q_','U_','V_','transmission','flux'] :
            if pref in l.desc:
                iw = l.names.index('wavelength')
                lr = l.sub(d={'wavelength':np.arange(NW)})
                res.add_lut(lr, desc=l.desc)
                for k,varname in enumerate(varnames):
                    lr = l.sub(d={'wavelength':np.arange(NW)+(k+1)*NW}) - l.sub(d={'wavelength':np.arange(NW)})
                    if delta is not None:
                        lr = lr/delta[k]
                        lr.desc = 'd'+l.desc+'/'+'d'+varname
                    else:
                        lr.desc = 'd'+l.desc+'->('+varname+')'
                    lr.names[iw]= 'wavelength'
                    lr.axes[iw] = m.axis('wavelength')[:NW]
                    res.add_lut(lr)
    res.attrs = m.attrs
    return res



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

 
def loop_kernel(NBPHOTONS, faer, foce, NLVL, NATM, NATM_ABS, NOCE, NOCE_ABS, MAX_HIST, NLOW,
                NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                NLAM, NSENSOR, double, kern, kern2, p, X0, le, tab_sensor, spectrum,
                prof_atm, prof_oc, cell_atm, cell_oc, wl_proba_icdf, stdev, rng, alis, myObjects0, TC, nbCx, nbCy, myGObj0, myRObj0, hist=False):
    """
    launch the kernel several time until the targeted number of photons injected is reached

    Arguments:
        - NBPHOTONS : Number of photons injected
        - Tableau : Class containing the arrays sent to the device
        - NLVL : Number of output levels
        - NATM : Number of atmospheric layers
        - NPSTK : Number of Stokes parameters + 1 for number of photons
        - BLOCK : Block dimension
        - XGRID : Grid dimension
        - NBTHETA : Number of intervals in zenith
        - NLAM : Number of wavelengths
        - options : compilation options
        - kern : kernel launching the transfert radiative
        - p: progress bar object
        - X0: initial coordinates of the photon entering the atmosphere
        - myObjects0 : gpu array containing the information of all the objects
        - TC : if there is a receiver object, this is the size of 1 cell (result visualisation)
        - nbCx, nbCy : number of cells in x and y directions
    --------------------------------------------------------------
    Returns :
        - nbPhotonsTot : Total number of photons processed
        - NPhotonsInTot : Total number of photons processed by interval
        - nbPhotonsSorTot : Total number of outgoing photons
        - tabPhotonsTot : Total weight of all outgoing photons
        - tabDistTot    : Total distance traveled by photons in atmospheric layers
        - tabMatRecep : Matrix containing the photon weight for each cell of the receiver
        - vecCats : vector containing the photon's number/weight of each category

    """
    # Initializations
    nThreadsActive = gpuzeros(1, dtype='int32')
    Counter = gpuzeros(1, dtype='uint64')
    
    if double : FDTYPE=np.float64
    else : FDTYPE=np.float32

    # If a receiver object is used then : initialize matrix and vectors for gains and losses
    if TC is not None:
        nbPhCat = gpuzeros(8, dtype=np.uint64) # vector to fill the number of photons for  each categories
        wPhCat = gpuzeros(8, dtype=FDTYPE)  # vector to fill the weight of photons for each categories
        wPhCat2 = gpuzeros(8, dtype=FDTYPE)  # sum of squared photons weight for each cats
        tabObjInfo = gpuzeros((9, nbCx, nbCy), dtype=FDTYPE)
        wPhLoss = gpuzeros(5, dtype=FDTYPE)
        wPhLoss2 = gpuzeros(5, dtype=FDTYPE)
        tabMatRecep = np.zeros((9, nbCx, nbCy), dtype=np.float64)
        # Matrix where lines : l0 = SumCats, l1=cat1, l2=cat2, ... l8=cat8
        # And columns : c0=nbPhotons , c1=weight, c2=weight2, c3=irradiance(in watt), c4=errAbs, c5=err%
        matCats = np.zeros((9, 6), dtype=np.float64)
        # vector where: v[0]=Wi, v[1]=Wi/(n.dirS), v[2]=Wo, v[3]=Wr
        vecLoss = np.zeros((5), dtype=np.float64)
        matLoss = np.zeros((5, 2), dtype=np.float64)
    else:
        nbPhCat = gpuzeros(1, dtype=np.uint64)
        wPhCat = gpuzeros(1, dtype=FDTYPE)
        wPhCat2 = gpuzeros(1, dtype=FDTYPE)
        wPhLoss = gpuzeros(1, dtype=FDTYPE)
        wPhLoss2 = gpuzeros(1, dtype=FDTYPE)
        tabObjInfo = gpuzeros((1, 1, 1), dtype=FDTYPE)
        
    # Initialize the array for error counting
    NERROR = 32
    errorcount = gpuzeros(NERROR, dtype='uint64')

    
    if ((NATM+NOCE >0) and (NATM_ABS+NOCE_ABS <500) and alis) : tabDistTot = gpuzeros((NLVL,NATM_ABS+NOCE_ABS,NSENSOR,NBTHETA,NBPHI), dtype=np.float64)
    else : tabDistTot = gpuzeros((1), dtype=np.float64)
    if hist : tabHistTot = gpuzeros((MAX_HIST,(NATM_ABS+NOCE_ABS+NPSTK+NLOW+3),NSENSOR,NBTHETA,NBPHI), dtype=np.float32)
    else : tabHistTot = gpuzeros((1), dtype=np.float32)

    # Initialize of the parameters
    tabPhotonsTot = gpuzeros((NLVL,NPSTK,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.float64)
    N_simu = 0
    if stdev:
        # to calculate the standard deviation of the result, we accumulate the
        # parameters and their squares
        # finally we extrapolate in 1/sqrt(N_simu)
        sum_x = 0.
        sum_x2 = 0.

    # arrays for counting the input photons (per wavelength)
    NPhotonsIn = gpuzeros((NSENSOR,NLAM), dtype=np.uint64)
    NPhotonsInTot = gpuzeros((NSENSOR,NLAM), dtype=np.uint64)
    
    # arrays for counting the output photons
    NPhotonsOut = gpuzeros((NLVL,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.uint64)
    NPhotonsOutTot = gpuzeros((NLVL,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.uint64)

    if double:
        tabPhotons = gpuzeros((NLVL,NPSTK,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.float64)
        if ((NATM+NOCE >0) and (NATM_ABS+NOCE_ABS <500) and alis) : tabDist = gpuzeros((NLVL,NATM_ABS+NOCE_ABS,NSENSOR,NBTHETA,NBPHI), dtype=np.float64)
        else : tabDist = gpuzeros((1), dtype=np.float64)
    else:
        tabPhotons = gpuzeros((NLVL,NPSTK,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.float32)
        if ((NATM+NOCE >0) and (NATM_ABS+NOCE_ABS <500) and alis) : tabDist = gpuzeros((NLVL,NATM_ABS+NOCE_ABS,NSENSOR,NBTHETA,NBPHI), dtype=np.float32)
        else : tabDist = gpuzeros((1), dtype=np.float32)

    if hist : tabHist = gpuzeros((MAX_HIST,(NATM_ABS+NOCE_ABS+NPSTK+NLOW+3),NSENSOR,NBTHETA,NBPHI), dtype=np.float32)
    else : tabHist = gpuzeros((1), dtype=np.float32)

    # local estimates angles
    if le != None:
        tabthv = to_gpu(le['th'].astype('float32'))
        tabphi = to_gpu(le['phi'].astype('float32'))
    else:
        tabthv = gpuzeros(1, dtype='float32')
        tabphi = gpuzeros(1, dtype='float32')

    secs_cuda_clock = 0.
    iopp = 1
    alis_norm = NLAM if NLOW!=0 else 1
    while((np.sum(NPhotonsInTot.get())/alis_norm) < NBPHOTONS):
        tabPhotons.fill(0.)
        NPhotonsOut.fill(0)
        NPhotonsIn.fill(0)
        Counter.fill(0)
        # en rapport avec les objets
        tabObjInfo.fill(0)
        wPhCat.fill(0)
        wPhCat2.fill(0)
        wPhLoss.fill(0)
        wPhLoss2.fill(0)
        nThreadsActive.fill(XBLOCK*XGRID)
        
        start_cuda_clock = cuda.Event()
        end_cuda_clock = cuda.Event()
        start_cuda_clock.record()

        # kernel launch
        kern(spectrum, X0, faer, foce,
             errorcount, nThreadsActive, tabPhotons, tabDist, tabHist,
             Counter, NPhotonsIn, NPhotonsOut, tabthv, tabphi, tab_sensor,
             prof_atm, prof_oc, cell_atm, cell_oc, wl_proba_icdf, rng.state, tabObjInfo,
             myObjects0, myGObj0, myRObj0, nbPhCat, wPhCat, wPhCat2,
             wPhLoss, wPhLoss2, block=(XBLOCK, 1, 1), grid=(XGRID, 1, 1))

        end_cuda_clock.record()
        end_cuda_clock.synchronize()
        secs_cuda_clock = secs_cuda_clock + start_cuda_clock.time_till(end_cuda_clock)

        cuda.Context.synchronize()
        np.set_printoptions(precision=5, linewidth=150)

        if TC is not None:
            # Tableau de la repartition des poids (photons) sur la surface du recepteur
            tabMatRecep += tabObjInfo[:, :, :].get()
            vecLoss += wPhLoss[:].get()
            matLoss[:,0] += wPhLoss[:].get()
            matLoss[:,1] += wPhLoss2[:].get()
            matCats[0,1] += np.sum(wPhCat[:].get())
            matCats[0,2] += np.sum(wPhCat2[:].get())
            for i in range (0, 8):
                # Comptage des poids pour chaque categories
                matCats[i+1,1] += wPhCat[i].get();    # sum of wi
                matCats[i+1,2] += wPhCat2[i].get();   # sum of wi²
        
        L = NPhotonsIn   # number of photons launched by last kernel
        NPhotonsInTot += L

        NPhotonsOutTot += NPhotonsOut
        S = tabPhotons   # sum of weights for the last kernel
        tabPhotonsTot += S
        
        T = tabDist
        tabDistTot += T
        if hist :
            tabHistTot = tabHist

        N_simu += 1
        if stdev:
            (NSENSOR,NLAM) = NPhotonsIn.shape
            L = L.reshape((1,1,NSENSOR,NLAM,1,1))   # broadcast to tabPhotonsTot
            #warn('stdev is activated: it is known to slow down the code considerably.')
            SoverL = S.get()/L.get()
            sum_x += SoverL
            sum_x2 += (SoverL)**2

        # update of the progression Bar
        sphot = np.sum(NPhotonsInTot.get())/alis_norm
        p.update(sphot,
                'Launched {:.3g} photons'.format(sphot))
    # END WHILE LOOP
    secs_cuda_clock = secs_cuda_clock*1e-3

    if TC is not None: # If there is a receiver obj
        nBis = NBPHOTONS/(NBPHOTONS-1)
        # Count the total number of photons received and also for each cats
        matCats[0,0] = np.sum(nbPhCat[:].get())
        for i in range (0, 8): # Here for each cats
            matCats[i+1,0] = nbPhCat[i].get();
        
        # Relative and absolute error for sum of cats and also for each cats
        for i in range (0, 9):
            if (matCats[i,0] != 0 and matCats[i,1] != 0):
                # Monte carlo err computation see the book of Dunn and Shultis
                sum2Z = (matCats[i,1]*matCats[i,1])/NBPHOTONS
                sumZ2 = matCats[i,2]
                matCats[i,4] = (nBis * (sumZ2 - sum2Z))**0.5 # errAbs not normalized
                matCats[i,5] = (matCats[i,4]/matCats[i,1])*100 # err%
    else:
        tabMatRecep = None; vecLoss = None; matCats = None; matLoss=None

    if stdev:
        # finalize the calculation of the standard deviation
        sigma = np.sqrt(sum_x2/N_simu - (sum_x/N_simu)**2)

        # extrapolate in 1/sqrt(N_simu)
        sigma /= np.sqrt(N_simu)
    else:
        sigma = None

    return NPhotonsInTot.get(), tabPhotonsTot.get(), tabDistTot.get(), tabHistTot.get(), errorcount, \
        NPhotonsOutTot.get(), sigma, N_simu, secs_cuda_clock, tabMatRecep, vecLoss, matCats, matLoss


def impactInit(prof_atm, NLAM, THVDEG, Rter, pp):
    '''
    Calculate the coordinates of the entry point in the atmosphere
    and direct transmission of the atmosphere

    Returns :
        - [x0, y0, z0] : cartesian coordinates
    '''
    if prof_atm is None:
        Hatm = 0.
        natm = 0
    else:
        Zatm = prof_atm.axis('z_atm')
        Hatm = Zatm[0]
        natm = len(Zatm)-1

    vx = -np.sin(THVDEG * np.pi / 180)
    vy = 0.
    vz = -np.cos(THVDEG * np.pi / 180)
    Rter = np.double(Rter)

    tautot = np.zeros(NLAM, dtype=np.float64)

    if pp:
        z0 = Hatm
        x0 = Hatm*np.tan(THVDEG*np.pi/180.)
        y0 = 0.

        if natm != 0:
            for ilam in range(NLAM):
                if prof_atm['OD_atm'].ndim == 2:
                    # lam, z
                    #tautot[ilam] = prof_atm['OD_atm'][ilam, natm]/np.cos(THVDEG*pi/180.)
                    tautot[ilam] = prof_atm['OD_atm'][ilam, -1]/np.cos(THVDEG*pi/180.)
                elif prof_atm['OD_atm'].ndim == 1:
                    # z
                    #tautot[ilam] = prof_atm['OD_atm'][natm]/np.cos(THVDEG*pi/180.)
                    tautot[ilam] = prof_atm['OD_atm'][-1]/np.cos(THVDEG*pi/180.)
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
        for i in range(1, natm+1):
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

def initObj(LGOBJ, vSun, CUSL=None):
    '''
    Definition of the function LOBJ

    ===ARGS:
    LGOBJ : List of object groups
    CUSL  : Custom lanching mode class (i.g. cusForward())
    vSun  : Vector with the sun direction (needed in RF mode)
    
    ===RETURN:
    nGObj     : The number of groups
    nObj      : The number of objects
    nRObj     : The number of receiver objects
    surfLPH   : Surface where photons are launched, in backward = None
    nb_H      : The number of heliostats (or heliostat facets)
    zAlt_H    : Sum of z altitude of all heliostats (or heliostat facets)
    totS_H    : The total surface of heliostats
    TC        : Receiver cell size
    nbCx      : The number of receiver cells in x direction 
    nbCy      : The number of receiver cells in y direction
    LOBJGPU   : GPU array of objects of type 'type_IObjets'
    LGOBJGPU  : GPU array of objects of type 'type_GObj'
    LROBJGPU  : GPU array of only receiver objects of type 'type_IObjets'
    '''

    ind = 0; LOBJ = []; nGObj = len(LGOBJ); nRObj = 0; INDROBJ = [];
    LGOBJGPU = np.zeros(nGObj, dtype=type_GObj, order='C')

    # Creation of a list with only Entity objects and creation
    # of a GPU table with object groups parameters

    for i in range (0, nGObj):
        LGOBJGPU['index'][i] = ind 
        LGOBJGPU['bPminx'][i] = LGOBJ[i].bboxGPmin.x; LGOBJGPU['bPminy'][i] = LGOBJ[i].bboxGPmin.y;
        LGOBJGPU['bPminz'][i] = LGOBJ[i].bboxGPmin.z; LGOBJGPU['bPmaxx'][i] = LGOBJ[i].bboxGPmax.x;
        LGOBJGPU['bPmaxy'][i] = LGOBJ[i].bboxGPmax.y; LGOBJGPU['bPmaxz'][i] = LGOBJ[i].bboxGPmax.z;
        if (LGOBJ[i].check == "GroupE"):
            LGOBJGPU['nObj'][i] = LGOBJ[i].nob
            ind += LGOBJ[i].nob
            LOBJ.extend(LGOBJ[i].le)
        elif (LGOBJ[i].check == "Entity"):
            LGOBJGPU['nObj'][i] = 1
            ind += 1
            LOBJ.append(LGOBJ[i])
        else:
            raise NameError('In myObjects list, only Entity and GroupE classes are autorised!')

    LGOBJGPU = to_gpu(LGOBJGPU)

    nObj = len(LOBJ)

    if CUSL != None and CUSL.dict['LMODE'] == "BR":
        LOBJGPU = np.zeros(nObj+1, dtype=type_IObjets, order='C')
        TC=CUSL.dict['REC'].TC
        sizeXmin = min(CUSL.dict['REC'].geo.p1.x, CUSL.dict['REC'].geo.p2.x,
                       CUSL.dict['REC'].geo.p3.x, CUSL.dict['REC'].geo.p4.x)
        sizeXmax = max(CUSL.dict['REC'].geo.p1.x, CUSL.dict['REC'].geo.p2.x,
                       CUSL.dict['REC'].geo.p3.x, CUSL.dict['REC'].geo.p4.x)
        sizeX = sizeXmax - sizeXmin
        sizeYmin = min(CUSL.dict['REC'].geo.p1.y, CUSL.dict['REC'].geo.p2.y,
                       CUSL.dict['REC'].geo.p3.y, CUSL.dict['REC'].geo.p4.y)
        sizeYmax = max(CUSL.dict['REC'].geo.p1.y, CUSL.dict['REC'].geo.p2.y,
                       CUSL.dict['REC'].geo.p3.y, CUSL.dict['REC'].geo.p4.y)
        sizeY = sizeYmax - sizeYmin
        nbCx = int(sizeX/TC)
        nbCy = int(sizeY/TC)
        LOBJGPU['mvRx'][nObj] = CUSL.dict['REC'].transformation.rotx
        LOBJGPU['mvRy'][nObj] = CUSL.dict['REC'].transformation.roty
        LOBJGPU['mvRz'][nObj] = CUSL.dict['REC'].transformation.rotz
        if (CUSL.dict['REC'].transformation.rotOrder == "XYZ"):
            LOBJGPU['rotOrder'][nObj] = 1
        elif(CUSL.dict['REC'].transformation.rotOrder == "XZY"):
            LOBJGPU['rotOrder'][nObj] = 2
        elif(CUSL.dict['REC'].transformation.rotOrder == "YXZ"):
            LOBJGPU['rotOrder'][nObj] = 3
        elif(CUSL.dict['REC'].transformation.rotOrder == "YZX"):
            LOBJGPU['rotOrder'][nObj] = 4
        elif(CUSL.dict['REC'].transformation.rotOrder == "ZXY"):
            LOBJGPU['rotOrder'][nObj] = 5
        elif(CUSL.dict['REC'].transformation.rotOrder == "ZYX"):
            LOBJGPU['rotOrder'][nObj] = 6
        else:
            raise NameError('Unknown rotation order')
        LOBJGPU['mvTx'][nObj] = CUSL.dict['REC'].transformation.transx
        LOBJGPU['mvTy'][nObj] = CUSL.dict['REC'].transformation.transy
        LOBJGPU['mvTz'][nObj] = CUSL.dict['REC'].transformation.transz

        INDROBJ.append(nObj) # For the creation of GPU table with only receivers
    else:
        LOBJGPU = np.zeros(nObj, dtype=type_IObjets, order='C')
        TC = None; nbCx = int(0); nbCy = int(0);

    # Initialization before the coming loop
    pp1 = 0.; pp2 = 0.; pp3 = 0.; pp4 = 0.;
    nb_H = 0; zAlt_H = 0.; totS_H = 0.;
    if (CUSL != None and CUSL.dict['LMODE'] == "RF"): surfLPH = 0;
    else: surfLPH = None;

    # ********************************************
    # Begining of the loop to consider all objects
    for i in range (0, nObj):
        # ==== At this moment only 2 choices -> spherical or plane surface
        # Here if this is a spherical object
        if isinstance(LOBJ[i].geo, Spheric):
            LOBJGPU['geo'][i] = 1
            LOBJGPU['myRad'][i] = LOBJ[i].geo.radius
            LOBJGPU['z0'][i] = LOBJ[i].geo.z0
            LOBJGPU['z1'][i] = LOBJ[i].geo.z1
            LOBJGPU['phi'][i] = LOBJ[i].geo.phi
        # Here if this is a plane object
        elif isinstance(LOBJ[i].geo, Plane):
            LOBJGPU['geo'][i] = 2
            LOBJGPU['p0x'][i] = LOBJ[i].geo.p1.x
            LOBJGPU['p0y'][i] = LOBJ[i].geo.p1.y
            LOBJGPU['p0z'][i] = LOBJ[i].geo.p1.z
            LOBJGPU['p1x'][i] = LOBJ[i].geo.p2.x
            LOBJGPU['p1y'][i] = LOBJ[i].geo.p2.y
            LOBJGPU['p1z'][i] = LOBJ[i].geo.p2.z
            LOBJGPU['p2x'][i] = LOBJ[i].geo.p3.x
            LOBJGPU['p2y'][i] = LOBJ[i].geo.p3.y
            LOBJGPU['p2z'][i] = LOBJ[i].geo.p3.z
            LOBJGPU['p3x'][i] = LOBJ[i].geo.p4.x
            LOBJGPU['p3y'][i] = LOBJ[i].geo.p4.y
            LOBJGPU['p3z'][i] = LOBJ[i].geo.p4.z

            # Get the normal of the plane Object after considering transform
            # 1) The intial normal is known ->
            normalBase = Vector(0, 0, 1);

            # 2) Consider the rotation transform in X, Y et Z
            TpT0 = Transform()
            TpRX0 = TpT0.rotateX(LOBJ[i].transformation.rotation[0])
            TpRY0 = TpT0.rotateY(LOBJ[i].transformation.rotation[1])
            TpRZ0 = TpT0.rotateZ(LOBJ[i].transformation.rotation[2])
            if (LOBJ[i].transformation.rotOrder == "XYZ"):
                TpT0 = TpRX0*TpRY0*TpRZ0
            elif(LOBJ[i].transformation.rotOrder == "XZY"):
                TpT0 = TpRX0*TpRZ0*TpRY0
            elif(LOBJ[i].transformation.rotOrder == "YXZ"):
                TpT0 = TpRY0*TpRX0*TpRZ0
            elif(LOBJ[i].transformation.rotOrder == "YZX"):
                TpT0 = TpRY0*TpRZ*TpRX0
            elif(LOBJ[i].transformation.rotOrder == "ZXY"):
                TpT0 = TpRZ0*TpRX0*TpRY0
            elif(LOBJ[i].transformation.rotOrder == "ZYX"):
                TpT0 = TpRZ0*TpRY0*TpRX0
            else:
                raise NameError('Unknown rotation order')

            # 3) Application of rotation transform
            normalBase = TpT0[normalBase]
            normalBase = Normalize(normalBase)
            LOBJGPU['nBx'][i] = normalBase.x
            LOBJGPU['nBy'][i] = normalBase.y
            LOBJGPU['nBz'][i] = normalBase.z

        else:    # si l'objet est autre chose (inconnu)
            raise NameError("Your geometry can be only spheric or plane, please" + \
                            " choose between Spheric or Plane classes!")
        # ====

        # ==== Affectation of transformations (rotations and translations)
        LOBJGPU['mvRx'][i] = LOBJ[i].transformation.rotx
        LOBJGPU['mvRy'][i] = LOBJ[i].transformation.roty
        LOBJGPU['mvRz'][i] = LOBJ[i].transformation.rotz
        if (LOBJ[i].transformation.rotOrder == "XYZ"):
            LOBJGPU['rotOrder'][i] = 1
        elif(LOBJ[i].transformation.rotOrder == "XZY"):
            LOBJGPU['rotOrder'][i] = 2
        elif(LOBJ[i].transformation.rotOrder == "YXZ"):
            LOBJGPU['rotOrder'][i] = 3
        elif(LOBJ[i].transformation.rotOrder == "YZX"):
            LOBJGPU['rotOrder'][i] = 4
        elif(LOBJ[i].transformation.rotOrder == "ZXY"):
            LOBJGPU['rotOrder'][i] = 5
        elif(LOBJ[i].transformation.rotOrder == "ZYX"):
            LOBJGPU['rotOrder'][i] = 6
        else:
            raise NameError('Unknown rotation order')
        LOBJGPU['mvTx'][i] = LOBJ[i].transformation.transx
        LOBJGPU['mvTy'][i] = LOBJ[i].transformation.transy
        LOBJGPU['mvTz'][i] = LOBJ[i].transformation.transz
        # ====

        # ==== Consider the material of the object
        # 1) Front part of the object (AV for the french word 'AVant')
        # Initialization
        LOBJGPU['materialAV'][i] = 0; LOBJGPU['shdAV'][i] = 0
        LOBJGPU['nindAV'][i] = 1; LOBJGPU['distAV'][i] = 0
        # Commun to all materials
        LOBJGPU['reflecAV'][i] = LOBJ[i].materialAV.reflectivity
        LOBJGPU['roughAV'][i] = LOBJ[i].materialAV.roughness
        # Particularity of each material
        if isinstance(LOBJ[i].materialAV, LambMirror):
            LOBJGPU['materialAV'][i] = 1
        elif isinstance(LOBJ[i].materialAV, Matte):
            LOBJGPU['materialAV'][i] = 2
        elif isinstance(LOBJ[i].materialAV, Mirror):
            LOBJGPU['materialAV'][i] = 3
            LOBJGPU['shdAV'][i] = int(LOBJ[i].materialAV.shadow)
            LOBJGPU['nindAV'][i] = LOBJ[i].materialAV.nind
            LOBJGPU['distAV'][i] = LOBJ[i].materialAV.distribution

        # 1) Back part of the object (AR for the french word 'ARriere')
        # Initialization
        LOBJGPU['materialAR'][i] = 0; LOBJGPU['shdAR'][i] = 0;
        LOBJGPU['nindAR'][i] = 1; LOBJGPU['distAR'][i] = 0
        # Commun to all materials
        LOBJGPU['reflecAR'][i] = LOBJ[i].materialAR.reflectivity
        LOBJGPU['roughAR'][i] = LOBJ[i].materialAR.roughness
        # Particularity of each material
        if isinstance(LOBJ[i].materialAR, LambMirror):
            LOBJGPU['materialAR'][i] = 1
        elif isinstance(LOBJ[i].materialAR, Matte):
            LOBJGPU['materialAR'][i] = 2
        elif isinstance(LOBJ[i].materialAR, Mirror):
            LOBJGPU['materialAR'][i] = 3
            LOBJGPU['shdAR'][i] = int(LOBJ[i].materialAR.shadow)
            LOBJGPU['nindAR'][i] = LOBJ[i].materialAR.nind
            LOBJGPU['distAR'][i] = LOBJ[i].materialAR.distribution
        # ====

        # ==== 2 possibilities : the object is a relfector or a receiver
        # Case of reflector object
        if (LOBJ[i].name == "reflector"):
            LOBJGPU['type'][i] = 1

            # Collect informations needed for STP applications
            if (  isinstance(LOBJ[i].geo, Plane) and \
                  ( isinstance(LOBJ[i].materialAR, Mirror) or \
                    isinstance(LOBJ[i].materialAV, Mirror) )  ):
                nb_H += 1
                zAlt_H += LOBJ[i].transformation.transz
                totS_H += abs(LOBJ[i].geo.p1.x)*abs(LOBJ[i].geo.p1.y)*4;

            # Crucial step for the result visualization in RF mode
            if (CUSL is not None and CUSL.dict['LMODE'] == "RF"):
                # Take the 4 initial points of the plane object 
                pp1 = LOBJ[i].geo.p1; pp2 = LOBJ[i].geo.p2
                pp3 = LOBJ[i].geo.p3; pp4 = LOBJ[i].geo.p4
                # Method to find the area of a convex rectangle
                DotP = Dot(vSun*-1, normalBase)
                TwoAAbis = abs((pp1.x - pp4.x)*(pp2.y - pp3.y)) + abs((pp2.x - pp3.x)*(pp1.y - pp4.y))
                surfLPHbis = (TwoAAbis/2.) * DotP
                surfLPH += surfLPHbis

        # Case of receiver object
        elif (LOBJ[i].name == "receiver"):
            LOBJGPU['type'][i] = 2
            TC=LOBJ[i].TC
            sizeXmin = min(LOBJ[i].geo.p1.x, LOBJ[i].geo.p2.x,
                           LOBJ[i].geo.p3.x, LOBJ[i].geo.p4.x)
            sizeXmax = max(LOBJ[i].geo.p1.x, LOBJ[i].geo.p2.x,
                           LOBJ[i].geo.p3.x, LOBJ[i].geo.p4.x)
            sizeX = sizeXmax - sizeXmin
            sizeYmin = min(LOBJ[i].geo.p1.y, LOBJ[i].geo.p2.y,
                           LOBJ[i].geo.p3.y, LOBJ[i].geo.p4.y)
            sizeYmax = max(LOBJ[i].geo.p1.y, LOBJ[i].geo.p2.y,
                           LOBJ[i].geo.p3.y, LOBJ[i].geo.p4.y)
            sizeY = sizeYmax - sizeYmin
            nbCx = int(sizeX/TC)
            nbCy = int(sizeY/TC)

            INDROBJ.append(i) # For the creation of GPU table with only receivers

        # This part is currently under development
        elif (LOBJ[i].name == "env"):
            LOBJGPU['type'][i] = 3 
        else:
            raise NameError('You have to specify if your object is a reflector or a receiver!')
        # ====
    # End of the loop
    # ********************************************
    
    # Creation of GPU table with only receivers
    nRObj = len(INDROBJ)
    if nRObj > 0:
        LROBJGPU = np.zeros(nRObj, dtype=type_IObjets, order='C')
        for i in range (0, nRObj):
            LROBJGPU[:][i] = LOBJGPU[:][INDROBJ[i]]
    else:
        LROBJGPU = np.zeros(1, dtype=type_IObjets, order='C')

    LOBJGPU = to_gpu(LOBJGPU)
    LROBJGPU = to_gpu(LROBJGPU)

    return nGObj, nObj, nRObj, surfLPH, nb_H, zAlt_H, totS_H, TC, nbCx, nbCy, LOBJGPU, LGOBJGPU, LROBJGPU

def normalizeRecIrr(cMatVisuRecep, matCats, nbCx, nbCy, NBPHOTONS, surfLPH, TC, cusL, SUN_DISC):
    '''
    Description of the function normalizeRecIrr

    This function enables the normalization of the signal collected
    by a given receiver, to get the irradiance in W/m² a multiplication by 
    the sun irradiance at TOA remain still needed.

    ==== ARGS:
    cMatVisuRecep : Matrix containing the signal weight collected by each
                    cell of a given receiver
    matCats       : Matrix containing the total signal collected (not splited
                    in cells) + the signal collected by each categories (see
                    moulana et al. 2019)
    nbCx          : Number of receiver cells in x direction
    nbCy          : Number of receiver cells in y direction
    NBPHOTONS     : The total number of launched photons
    TC            : Size of a square cell, (TC -> french word 'Taille Cellule') 
    cusL          : Custum launching mode class (see cusForward, cusBackward)
    SUN_DISC      : The half-angle of the sun solid angle

    === RETURN:
    cMatVisuRecep : With normalized values
    matCats       : With normalized values
    '''
    if (cusL is None):
        cMatVisuRecep[:][:][:] = (cMatVisuRecep[:][:][:]*nbCx*nbCy)/NBPHOTONS
        for i in range (0, 9):
            matCats[i,3] = matCats[i,1]/NBPHOTONS # intensity
            matCats[i,4] /= NBPHOTONS # Absolute err
    elif (cusL.dict['LMODE'] == "FF" or cusL.dict['LMODE'] == "RF"):
        for i in range (0, 9):
            cMatVisuRecep[i][:][:] = cMatVisuRecep[i][:][:] * ((surfLPH)/(TC*TC*NBPHOTONS))
            matCats[i,3] = matCats[i,1] * ((surfLPH)/(TC*TC*nbCx*nbCy*NBPHOTONS))
            matCats[i,4] *= ((surfLPH)/(TC*TC*nbCx*nbCy*NBPHOTONS))
    elif (cusL.dict['LMODE'] == "B" or cusL.dict['LMODE'] == "BR"):
        normBR = 2
        #lambertian sampling normalization
        if (cusL.dict['TYPE'] == 1): normBR = 1-np.cos(np.radians(cusL.dict['ALDEG']))
        #isotropic sampling normalization
        elif (cusL.dict['TYPE'] == 2): normBR = 2*(1-np.cos(np.radians(cusL.dict['ALDEG'])))
        cMatVisuRecep[:][:][:] = (cMatVisuRecep[:][:][:]*nbCx*nbCy*normBR)/(NBPHOTONS*2*(1-np.cos(np.radians(SUN_DISC))))
        for i in range (0, 9):
            matCats[i,3] = (matCats[i,1]*normBR)/(NBPHOTONS*2*(1-np.cos(np.radians(SUN_DISC))))
            matCats[i,4] /= NBPHOTONS*2*(1-np.cos(np.radians(SUN_DISC)))/normBR

    return cMatVisuRecep, matCats


def findExtinction(IP, FP, prof_atm):
    '''
    Description of the function findAtmLoss

    This function enables the calculation of the extinction
    between an initial point 'IP' to a final point 'FP'
    \!/ === ONLY VALID WITH A 1D PP ATM ===

    ==== ARGS:
    IP       : Initial position (Point class)
    FP       : Final position (Point class)
    prof_atm : Atmosphere profil class

    ==== RETURN:
    n_ext   : Extinction between IP and FP
    '''
    # Be sure IP and FP are Point classes
    if not all(isinstance(i, Point) for i in [IP, FP]):
        raise NameError('Both IP and FP must be Point classes!')

    # If there is no atm then there are no scattering and abs -> n_ext = 1
    if (prof_atm is None):
        n_ext = 1
        return n_ext

    # Vector/direction from IP to FP
    Vec = FP - IP

    # Find the atm layer of the initial location
    lay = int(0)
    while(prof_atm.axis('z_atm')[lay] > IP.z):
        lay += int(1)
        
    # Initialization
    tauHit = 0. # Optical depth distance (from IP to FP)
    ilayer2 = lay;

    # Case with only 1 layer: n = 1
    if (FP.z >= prof_atm.axis('z_atm')[ilayer2] and FP.z < prof_atm.axis('z_atm')[ilayer2-1]):
        # delta_i is: Delta(tau)1 = |tau(i-1) - tau(i)|
        delta_i = abs(prof_atm['OD_atm'].data[0][ilayer2-1] - prof_atm['OD_atm'].data[0][ilayer2])
        # tauHit = (Delat(D1)/Delat(Z1))*delta_i
        tauHit += ((IP - FP).Length()/abs(prof_atm.axis('z_atm')[ilayer2-1]-prof_atm.axis('z_atm')[ilayer2]))*delta_i
    else: # Case with several layers: n >= 2
        # Find the layer where there is intersection
        ilayer2 = int(1)
        while(prof_atm.axis('z_atm')[ilayer2] > FP.z and prof_atm.axis('z_atm')[ilayer2] > 0.):
            ilayer2+=int(1)

        higher = False
        ilayer = lay
        oldP = IP
        
        # Check if the photon come from higher or lower layer
        if(ilayer < ilayer2): # true if the photon come from higher layer
            higher =  True

        while(ilayer != ilayer2):
            if(higher):
                timeT = abs(prof_atm.axis('z_atm')[ilayer] - oldP.z)/abs(Vec.z)
            else:
                timeT = abs(prof_atm.axis('z_atm')[ilayer-1] - oldP.z)/abs(Vec.z)
            newP = oldP + (Vec*timeT);
            delta_i = abs(prof_atm['OD_atm'].data[0][ilayer]-prof_atm['OD_atm'].data[0][ilayer-1])
            tauHit += ((newP - oldP).Length()/abs(prof_atm.axis('z_atm')[ilayer-1]-prof_atm.axis('z_atm')[ilayer]))*delta_i
        
            if(higher): # the photon come from higher layer
                ilayer+= int(1)
            else: # the photon come from lower layer
                ilayer-= int(1)
            oldP = newP # Update the position of the photon
        
        # Calculate and add the last tau distance when ilayer is equal to ilayer2
        delta_i = abs(prof_atm['OD_atm'].data[0][ilayer2]-prof_atm['OD_atm'].data[0][ilayer2-1])
        tauHit += ((FP - oldP).Length()/abs(prof_atm.axis('z_atm')[ilayer2-1]-prof_atm.axis('z_atm')[ilayer2]))*delta_i

    n_ext = np.exp(-abs(tauHit))

    return n_ext
