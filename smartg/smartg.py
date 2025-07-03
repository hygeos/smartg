#!/usr/bin/env python
# encoding: utf-8


'''
SMART-G
Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU
'''


import os
import numpy as np
from datetime import datetime, timezone
from numpy import pi
from smartg.atmosphere import Atmosphere, od2k, BPlanck
from smartg.water import IOP_base
from smartg.shape import BBox, Sphere
from os.path import dirname, realpath, join, exists
from warnings import warn, simplefilter
from smartg.albedo import Albedo_cst, Albedo_speclib, Albedo_spectrum, Albedo_map
from smartg.tools.progress import Progress
from smartg.tools.cdf import ICDF2D
from smartg.tools.modified_environ import modified_environ
from luts.luts import LUT, MLUT
from scipy.interpolate import interp1d
#from scipy.integrate import simpson
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
    
import functools
from copy import deepcopy


simplefilter('always', DeprecationWarning)


# set up directories
dir_root = dirname(dirname(realpath(__file__)))
dir_src = join(dir_root, 'smartg/src/')
src_device = join(dir_src, 'device.cu')
src_kernel2 = join(dir_src, 'kernel2.cu')
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
MAX_NREF = 10

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

type_Spectrum = np.dtype([
    ('lambda'      , 'float32'),
    ('alb_surface' , 'float32'),
    ('alb_seafloor', 'float32'),
    ('alb_env',      'float32'),
    ('k1p_surface' , 'float32'),
    ('k2p_surface' , 'float32'),
    ('k3p_surface' , 'float32'),
    ('alb_envs' , 'float32', MAX_NREF),
    ])

type_EnvMap = np.dtype([
    ('x',      'float32'),    # // x coordinate on the ground
    ('y',      'float32'),    # // y coordinate on the ground
    ('env_index',   'int32'),   # // environment index map
])

type_Profile = [
    ('z',      'float32'),    # // altitude
    ('n',      'float32'),    # // refractive index
    ('T',      'float32'),    # // temperature
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
    ('ILAM_0', 'int32'),      # // Wavelength start index that the sensor 'sees' (default -1 : all) 
    ('ILAM_1', 'int32'),      # // Wavelength stop  index that the sensor 'sees' (default -1 : all) 
    ]

type_Spectrum_obj = [
    ('reflectAV', 'float32'),
    ('reflectAR', 'float32'),
]

type_IObjets = [
    ('geo', 'int32'),         # 1 = sphere, 2 = plane, ...
    ('materialAV', 'int32'),  # 1 = LambMirror, 2 = Matte,
    ('materialAR', 'int32'),  # 3 = Mirror, ... (AV = avant, AR = Arriere)
    ('type', 'int32'),        # 1 = reflector, 2 = receiver
    ('reflectAV', 'float32'),  # reflectivity of materialAV
    ('reflectAR', 'float32'),  # reflectivity of materialAR
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
    """
    Definition of a flat sea surface

    Parameters
    ----------
    SUR : int, optional
        The processes at the surface dioptre. 2 choices:
            - 1 -> Force reflection
            - 3 -> Reflection and transmission
    NH2O : float, optional
        The relative refarctive index air/water
    """
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
    """
    Definition of a roughened sea surface

    Parameters
    ----------
    WIND : float, optional
        The wind speed (m/s)
    SUR : int, optional
        The processes at the surface dioptre. 2 choices:
            - 1 -> Force reflection
            - 3 -> Reflection and transmission
    NH2O : float, optional
        The relative refarctive index air/water
    WAVE_SHADOW : bool, optional
        Include wave shadowing effect. Default False.
    BRDF : bool, optional
        Replace slope sampling by Cox & Munk BRDF, no ocean, just reflection
    SINGLE : bool, optional
        Deactivate multiple reflections/refractions at the interface. Default False.
    """
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
        self.alb=None
        self.kp=None
    def __str__(self):
        return 'ROUGHSUR={SUR}-WIND={WINDSPEED}-DI={DIOPTRE}-WAVE_SHADOW={WAVE_SHADOW}-BRDF={BRDF}-SINGLE={SINGLE}'.format(**self.dict)


class LambSurface(object):
    """
    Definition of a lambertian reflector

    Parameters
    ----------
    ALB : Albedo_cst, | Albedo_speclib | Albedo_spectrum | Albedo_map, optional
        The albedo spectral model.
    """
    def __init__(self, ALB=Albedo_cst(0.5)):

        if (not isinstance(ALB, (Albedo_cst, Albedo_speclib, Albedo_spectrum, Albedo_map))):
            raise ValueError('The parameter ALB must be one of the following objects: Albedo_cst, ' + 
                             'Albedo_speclib, Albedo_spectrum or Albedo_map.')
        self.dict = {
                'SUR': 1,
                'DIOPTRE': 3,
                'WINDSPEED': -999.,
                'NH2O': -999.,
                'WAVE_SHADOW': 0,
                'BRDF': 1,
                'SINGLE': 1,
                }
        self.alb = ALB
    def __str__(self):
        return 'LAMBSUR-ALB={SURFALB}'.format(**self.dict)


class RTLSSurface(object):
    """
    Definition of a Ross-Thick Li-Sparse reflector

    Parameters
    ----------
    kp: None | tuple, optional
        The Ross-Thick Li-Sparse coefficients (deprecated, see notes). Form of the tuple:
         
        * k0 : Albedo_cst | Albedo_speclib | Albedo_spectrum | Albedo_map
            -> The spectral albedo of the isotropic (lambertian) kernel
        * k1p: Albedo_cst | Albedo_speclib | Albedo_spectrum | Albedo_map
           -> The relative weight of the F1 (geometric) kernel (=K1/K0)
        * k2p: Albedo_cst | Albedo_speclib | Albedo_spectrum | Albedo_map
           -> The relative weight of the F2 (volumetric) kernel (=K2/K0)
    k0 : None | Albedo_cst | Albedo_speclib | Albedo_spectrum | Albedo_map, optional
        The spectral albedo of the isotropic (lambertian) kernel. Default Albedo_cst(0.5).
    k1p : None | Albedo_cst | Albedo_speclib | Albedo_spectrum | Albedo_map, optional
        The relative weight of the F1 (geometric) kernel (=K1/K0). Default Albedo_cst(0.5).
    k2p : None | Albedo_cst | Albedo_speclib | Albedo_spectrum | Albedo_map, optional
        The relative weight of the F2 (volumetric) kernel (=K2/K0). Default Albedo_cst(0.5).

    Notes
    -----
    The parameter kp is depracated. Use k0, k1p and k2p instead. Providing parameters k0, k1p 
    and k2p with kp will circumvent kp values.
    """
    def __init__(self, kp=None, k0=None, k1p=None, k2p=None):

        kp_bis = (Albedo_cst(0.5), Albedo_cst(0.0), Albedo_cst(0.0))
        if kp is not None:
            warn_message = "\nThe use of parameter `kp` is deprecated as of SMART-G 1.1.0 " + \
                           "and will be removed in one of the next release.\n" + \
                           "Please use k0, k1p and k2p instead."
            warn(warn_message, DeprecationWarning)
            kp_bis = deepcopy(kp)
        
        if k0 is not None: kp_bis[0] = k0
        if k1p is not None: kp_bis[1] = k1p
        if k2p is not None : kp_bis[2] = k2p

        self.dict = {
                'SUR': 1,
                'DIOPTRE': 4,
                'WINDSPEED': -999.,
                'NH2O': -999.,
                'WAVE_SHADOW': 0,
                'BRDF': 1,
                'SINGLE': 1,
                }
        self.kp = kp_bis+(Albedo_cst(0.0),)
        self.alb= None
    def __str__(self):
        return 'RTLS-ALB={SURFALB}'.format(**self.dict)


class RPVSurface(object):
    """
    Definition of a RPV reflector

    Parameters
    ----------
    kp : tuple, optional
        The RPV coefficients. Form of the tuple:

        * r0 : Albedo_cst
            -> Normalization.
        * k : Albedo_cst
            -> Minnaert exponent.
        * bt : Albedo_cst
            -> Henyey-Greenstein asymetry parameter.
        * rc : Albedo_cst
            -> Hotspot parameter.
    
    r0 : None | Albedo_cst | Albedo_speclib | Albedo_spectrum | Albedo_map, optional
       Normalization.
    k : None | Albedo_cst | Albedo_speclib | Albedo_spectrum | Albedo_map, optional
        Minnaert exponent.
    bt : None | Albedo_cst | Albedo_speclib | Albedo_spectrum | Albedo_map, optional
        Henyey-Greenstein asymetry parameter.
    rc : None | Albedo_cst | Albedo_speclib | Albedo_spectrum | Albedo_map, optional
        Hotspot parameter.
    
    Notes
    -----
    The parameter kp is depracated. Use r0, k, bt and rc instead. Providing parameters r0, 
    k, bt and rc with kp will circumvent kp values.

    References
    ----------
    Rahman, H., M. M. Verstraete, and B. Pinty, (1993) Coupled surface-atmosphere 
    reflectance (CSAR) model. 1. Model description and inversion on synthetic data, 
    JGR, 98, 20,779-20,789.

    """
    def __init__(self, kp=None, r0=None, k=None, bt=None, rc=None):

        kp_bis = (Albedo_cst(0.5), Albedo_cst(0.0), Albedo_cst(0.0), Albedo_cst(0.0))
        if kp is not None:
            warn_message = "\nThe use of parameter `kp` is deprecated as of SMART-G 1.1.0 " + \
                           "and will be removed in one of the next release.\n" + \
                           "Please use r0, k, bt and rc instead."
            warn(warn_message, DeprecationWarning)
            kp_bis = deepcopy(kp)
        
        if r0 is not None: kp_bis[0] = r0
        if k is not None: kp_bis[1] = k
        if bt is not None : kp_bis[2] = bt
        if rc is not None : kp_bis[3] = rc

        self.dict = {
                'SUR': 1,
                'DIOPTRE': 5,
                'WINDSPEED': -999.,
                'NH2O': -999.,
                'WAVE_SHADOW': 0,
                'BRDF': 1,
                'SINGLE': 1,
                }
        self.kp = kp_bis
        self.alb= None
    def __str__(self):
        return 'RTLS-ALB={SURFALB}'.format(**self.dict)



class Environment(object):
    """
    Stores the smartg parameters relative the the environment effect

    Parameters
    ----------
    ENV : int, optional
        The type of environment to consider. Possibilities are:

        * -1 -> The opposite of 1
        *  0 -> Deactivated. The default value.
        *  1 -> Horizontal cst albedo ALB outside an horizontal disk, water is inside, 
                lambertian ALB is outside.
        *  2 -> ALB is a gaussian centred on X0,Y0 with a maximum of ALB_SURF and an asymptotic value 
                of ALB of the environement. The square of the sigma is ENV_SIZE.
        *  3 -> ALB map2D modulated by checkerboard spatial function
        *  4 -> Same as 1 but for a band defined as Abs(X) <= ENV_SIZE, -4 for Abs(X)>= ENV_SIZE
        *  5 -> 2D horizontal map of albedos for the whole surface, need ALB to be ALbedo_map object 
                in that case the surface keyword of Smartg run method is not unused
    
    ENV_SIZE : float, optional
        Definitions:
        
            - The radius (in km) of the circle outside which ALB model is applied for ENV = -1 or 1.
            - The square (in km) of the sigma of the gaussian for ENV = 2
            - The size of the spatial pattern (in km) for ENV = 3
    X0 : float, optional
        The X origin position
    Y0 : float, optional
        The Y origin position
    ALB : Albedo_cst | Albedo_speclib | Albedo_spectrum | Albedo_map, optional
        The albedo spectral model
    NENV : int, optional
        In progress...
    NXENVMAP : int, optional
        In progress...
    NYENVMAP : int, optional
        In progress...

    """
    def __init__(self, ENV=0, ENV_SIZE=1.e6, X0=0., Y0=0., ALB=Albedo_cst(0.0), NENV=1,
                NXENVMAP=0, NYENVMAP=0):
        self.dict = {
                'ENV': ENV,
                'ENV_SIZE': ENV_SIZE,
                'X0': X0,
                'Y0': Y0,
                }
        self.alb = ALB
        self.NENV= NENV
        self.NXENVMAP= NXENVMAP
        self.NYENVMAP= NYENVMAP

    def __str__(self):
        return 'ENV={ENV_SIZE}-X={X0:.1f}-Y={Y0:.1f}'.format(**self.dict)

class Sensor(object):
    """
    Definition of the sensor

    Parameters
    ----------
    POSX : float, optional
       The sensor position along the x axis. Default 0.
    POSY : float, optional
        The sensor position along the y axis. Default 0.
    POSZ : float, optional
        The sensor position along the z axis. Default 0.
    THDEG : float, optional
        The source/viewing zenith angle in forward/backward mode. Zenith > 90 for downward looking, 
        < 90 for upward. Default Zenith.
    PHDEG : float, optional
        The source/viewing azimuth angle in forward/backward mode. Zenith > 90 for downward looking,
        <90 for upward. Default Zenith.
    LOC : str, optional
        Localization of the sensor. Possibilities are:

        * 'SURF0P' -> Start from the surface looking upward, at TOA (air side). Default value.
        * 'SURF0M' -> Start from the surface looking downward, at ocean surface (water side).
        * 'ATMOS' -> Start from the atmosphere.
        * 'OCEAN' -> Start from the ocean.
        * 'SEAFLOOM' -> Start from the ocean surface.
        * 'OBJSURF' -> Start from a 3d object surface.
    FOV : float, optional
        The field of view in degrees. Default 0.
    TYPE : int, optional
        The radiative quantity type. Three possibilities:

        * 0 -> Radiance (default).
        * 1 -> Planar flux.
        * 2 -> Spherical flux.
    ICELL : int, optional
        The box index where the sensor is located. Only for simulations with a 3D atmosphere.
    """
    def __init__(self, POSX=0., POSY=0., POSZ=0., THDEG=0., PHDEG=180.,
                 LOC='SURF0P', FOV=0., TYPE=0, ICELL=0, ILAM_0=-1, ILAM_1=-1, V = None, CELL_SIZE = -1):

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
            'ICELL': ICELL,
            'ILAM_0': ILAM_0,
            'ILAM_1': ILAM_1
        }
        self.cell_size = CELL_SIZE

    def __str__(self):
        return 'SENSOR=-POSX{POSX}-POSY{POSY}-POSZ{POSZ}-THETA={THDEG:.3f}-PHI={PHDEG:.3f}'.format(**self.dict)

class StdevLim(object):
    """
    Definition of the class StdevLim


    Parameters
    ----------
    err_abs_min : float, optional
        The minimum absolute error. Stop the simulation if max abs error <= err_abs_min.
    err_rel_min : float, optional
        Theminimum relative error in percentage.
    nb_loop_min : int, optional
        The minimum kernel loop number before allowing to stop the simulation.
    stk : int, optional
        The stoke component to consider. Choices are:
        
            * 0 -> I stoke component (Default)
            * 1 -> Q stoke component
            * 2 -> U stoke component
            * 3 -> V stoke component
    llevl : int, optional
        The level to use to analyse the standart deviations. Six choices:

            * 0 -> UPTOA (Default)
            * 1 -> DOWN0P
            * 2 -> DOWN0M
            * 3 -> UP0P
            * 4 -> UP0M
            * 5 -> DOWNB
    verbose : bool, optional
        Activate verbose mode to print the max absolute and relative errors at each kernel loop.
    format : str, optional
        The verbose print format for abs and rel max values.

    Notes
    -----
    For the moment, it does not work correctly with kdis and reptran.
    """

    def __init__(self, err_abs_min=float(0), err_rel_min=float(0), nb_loop_min=int(10),
     stk=int(0), level=int(0), verbose=False, format=".5e"):
      
        self.dict = {
            'err_abs_min':  err_abs_min,
            'err_rel_min':  err_rel_min,
            'nb_loop_min':  nb_loop_min,
            'stk'        :  stk,
            'level'      :  level,
            'verbose'    :  verbose,
            'format'     :  format
        }

    def __str__(self):
        return self.dict.__str__()
    
    def __repr__(self):
        return 'Stdevlim dict: %s' %  self.dict.__repr__()
        
class CusForward(object):
    """
    Definition of CusForward 

    - Custum rectangular forward mode of surface X*Y

    Parameters
    ----------
    CFX : float, optional
        The size along the x axis (only for FF LMODE)
    CFY : float, optional
        The size along the y axis (only for FF LMODE)
    CFTX : float, optional
        The translation to apply in x axis (only for FF LMODE)
    CFTY : float, optional
        The translation to apply in y axis (only for FF LMODE)
    FOV : float, optional
        The field of view or half-angle of the sun (only for FF LMODE)
    TYPE : str, optional
        The sampling type (only for FF LMODE). 2 choices:
         
            * 'lambertian'
            * 'isotropic'
    LMODE : str, optional
        The launching mode. Two choices:

            * 'RF' -> Restricted Forward. Launch the photons such that the direct beams
                      fill only reflector objects
            * 'FF' -> Full Forward. Launch the photons in a rectangle from TOA whrere the 
                      beams at the center targets the origin point (0,0,0).
    """
    def __init__(self, CFX=0., CFY=0., CFTX = 0., CFTY = 0., CFTZ= 0., FOV = 0., TYPE = "isotropic",
                 LMODE = "RF", LPH=None, LPR=None):

        if (TYPE == "lambertian"): TYPE = 1
        elif (TYPE == "isotropic"): TYPE = 2
        elif (TYPE == "disk"): TYPE = 3 # in development
        else: raise NameError('You must choose lambertian or isotropic sampling')

        self.dict = {
            'CFX':   CFX,
            'CFY':   CFY,
            'CFTX':  CFTX,
            'CFTY':  CFTY,
            'CFTZ':  CFTZ,
            'FOV':   FOV,
            'TYPE':  TYPE,
            'LMODE': LMODE,
            # under developement->
            'LPH':     LPH,
            'LPR':     LPR
        }
        
    def __str__(self):
        return 'CusForward=-CFX{CFX}-CFY{CFY}-CFTX{CFTX}-CFTY{CFTY}-CFTZ{CFTZ}'.format(**self.dict) + \
            '-FOV{FOV}-TYPE{TYPE}-LMODE{LMODE}'.format(**self.dict)

class CusBackward(object):
    """
    Definition of CusBackward

    - Use a point/plane receiver sensor in backward.

    Parameters
    ----------
    POS : Point, optional
        The position (X,Y,Z) in cartesian coordinates.
    THDEG : float, optional
        The zenith angle in degrees.
    PHDEG : float, optional
        The azimuth angle in degrees.
    V : Vector, optional
        The normal vector of the receiver. If provided, circumvent THDEG and PHDEG.
    ALDEG : float, optional
        Launch in a solid angle where alpha is the half-angle of the cone.
    REC : Entity, optional
        The receiver object to be used in 'BR' mode. It must be a plane Entity object of 
        type 'receiver'. The photon position is sampled at the receiver surface.
    TYPE : str, optional
        The sampling type (only for BR LMODE). 2 choices:
         
            * 'lambertian'
            * 'isotropic'
    LMODE : str, optional
        The launching mode. 2 choices:

            * 'B' -> Basic backward. Launch the photons from a given point in a given direction 
                     (eventually can choose a ramdom vector in a solid angle arround a given direction
                     delimited by the half-angle ALDEG).
            * 'BR' -> Backward with receiver. Launch the photons from a given receiver (plane object)
                      in a given direction (also eventually can choose a ramdom vector in a solid angle 
                      delimited by ALDEG).
    """
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
    """
    Initialization of the Smartg object

    Performs the compilation and loading of the kernel.
    This class is designed so split compilation and kernel loading from the
    code execution: in case of successive smartg executions, the kernel
    loading time is not repeated.

    Parameters
    ----------
    pp :  bool, optional
        Use a plane parallel atmosphere, else spherical atmosphere
    autoinit : bool, optional
        Use pycuda autoinit to initialize pycuda context.
    debug : bool, optional
        Activate debug mode (optional stdout if problems are detected)
    verbose_photon : bool, optional
        Activate the display of photon path for the thread 0
    double : bool, optional
        Accumulate photons table in double precision (default double).
    alis : bool, optional
        Use the ALIS method (Emde et al. 2010) for treating gaseous absorption and perturbed profile.
        The parameter alt_pp must be set to True.
    back : bool, optional
        Activate backward mode (else forward)
    bias : bool, optional
        Use the bias sampling scheme
    alt_pp : bool, optional
        Use a plane parallel propagation scheme following the photon at each layer.
        Increase the computational time, but allow the use of the ALIS method
    obj3D : bool, optional
        Allow 3D objects
    opt3D : bool, optional
        Activate the 3D atmosphere mode
    device : int | str, optional
        The device number / GPU to use. The GPU numbers can be obtained with the command `nvidia-smi`.
    sif : bool, optional
        Include the Sun Induced Fluorescence
    thermal : bool, optional
        Still in dev...
    rng : str, optional
        The pseudo-random number generator to use, only 2 choice:
        - PHILOX
        - CURAND_PHILOX
    cache_dir : str, optional
        Path to the directory where the cache files are stored.
    keep_context: None | bool, optional
        Only in case autoinit is set to False. This parameter allows to keep or not the context 
        after the use of the run method. By default (for the case autoinit=False) kill the context after the use of the run method.
    """
    def __init__(self, pp=True, debug=False, autoinit=True,
                 verbose_photon=False,
                 double=True, alis=False, back=False, bias=True, alt_pp=False, obj3D=False, 
                 opt3D=False, device=None, sif=False, thermal=False, rng='PHILOX', cache_dir='/tmp/',
                 keep_context=None):
        assert not ((device is not None) and ('CUDA_DEVICE' in os.environ)), "Can not use the 'device' option while the CUDA_DEVICE is set"

        if device is not None:
            env_modif = {'CUDA_DEVICE': str(device)}
        else:
            env_modif = {}

        if not autoinit:
            self.keep_context = keep_context if keep_context is not None else False
        else:
            if keep_context is not None:
                raise ValueError("The parameter keep_context can be defined only if 'autoinit' is False.")
            self.keep_context = True
            
        if (autoinit):
            with modified_environ(**env_modif):
                try:
                    import pycuda.autoinit
                    self.ctx = pycuda.autoinit.context
                except:
                    # In case cuda context has been manually popped
                    from importlib import reload, import_module
                    pycuda.autoinit = import_module('pycuda.autoinit')
                    reload(pycuda.autoinit)
                    self.ctx = pycuda.autoinit.context
        else:
            import pycuda
            import pycuda.driver as cuda
            cuda.init()
            from pycuda.tools import make_default_context
            self.ctx = make_default_context()
        
        self.autoinit = autoinit
        self.pp = pp
        self.double = double
        self.alis = alis
        self.rng = init_rng(rng)
        self.back= back
        self.thermal=thermal
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
        if thermal:
            # thermal source
            options.append('-DTHERMAL')
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
        #options.append('-lineinfo')


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
                           cache_dir=cache_dir,
                           include_dirs=[dir_src,
                               join(dir_src, 'incRNGs/Random123/')])

        # load the kernel
        self.kernel = self.mod.get_function('launchKernel')
        #self.kernel2 = self.mod.get_function('launchKernel2')
        self.kernel2 = self.mod.get_function('reduce_absorption_gpu')

        #
        # common attributes
        #
        self.common_attrs = OrderedDict()
        self.common_attrs['compilation_time'] = (datetime.now()
                        - time_before_compilation).total_seconds()
        if (autoinit):
            self.common_attrs['device'] = pycuda.autoinit.device.name()
            try:
                self.common_attrs['device_number'] = pycuda.autoinit.device.get_attributes()[pycuda._driver.device_attribute.MULTI_GPU_BOARD_GROUP_ID]
            except AttributeError:
                self.common_attrs['device_number'] = 'undefined'
        else:
            self.common_attrs['device'] =self.ctx.get_device().name()
            try:
                self.common_attrs['device_number'] = self.ctx.get_device().get_attributes()[pycuda._driver.device_attribute.MULTI_GPU_BOARD_GROUP_ID]
            except Exception:
                self.common_attrs['device_number'] = 'undefined'
        self.common_attrs['pycuda_version'] = pycuda.VERSION_TEXT
        self.common_attrs['cuda_version'] = '.'.join([str(x) for x in pycuda.driver.get_version()])
        self.common_attrs.update(get_git_attrs())


    def clear_context(self):
        """
        Manually kill the cuda context

        - Once you call this, you can no longer use the method run(), 
          for that the smartg object must be reinitialized.
        """
        try:
            self.ctx.pop()
            self.ctx.detach()
            self.ctx = None
            from pycuda.tools import clear_context_caches
            clear_context_caches()
            if self.autoinit:
                # In case of autoinit delete pycuda.autoinit
                import pycuda.autoinit
                del pycuda.autoinit
        except:
            print("There is no current context to clear.")


    def run(self, wl, atm=None, surf=None, water=None, env=None, alis_options=None,
            NBPHOTONS=1e9, DEPO=0.0279, DEPO_WATER= 0.0906, THVDEG=0., PHVDEG=0., SEED=-1,
            RTER=6371., wl_proba=None, sensor_proba=None, cell_proba=None,
            NBTHETA=45, NBPHI=90, NF=1e6,
            OUTPUT_LAYERS=0, XBLOCK=256, XGRID=256,
            NBLOOP=None, progress=True, 
            le=None, flux=None, stdev=False, stdev_lim=None,
            BEER=1, RR=0, WEIGHTRR=0.1, SZA_MAX=90., SUN_DISC=0,
            sensor=None, refraction=False, reflectance=True,
            myObjects=None, interval = None,
            IsAtm = 1, cusL = None, SMIN=0, SMAX=1e6, RMIN=0, RMAX=1e6, FFS=False, DIRECT=False,
            OCEAN_INTERACTION=None, pol_off=False, no_aer_output=False):
        """
        Run a SMART-G simulation

        Parameters
        ----------
        wl : float | list | 1-D ndarray
            Wavelength(s) in nm. It can be a list of REPTRAN_IBAND or KDIS_IBAND objects.
        atm : None | AtmAFGL | MLUT, optional
            The atmosphere profile. If None, there is no atmosphere.
        surf : None | RoughSurface | FlatSurface | LambSurface, optional
            The surface profile. If None, there is no surface.
        water : None | IOP | IOP_1, optional
            The water profile. If None, there is no water.
        env : None | Environemnt, optional
            The environment (adjacency effect) profile. If None, there is no environment.
        alis_options : None | dict, optional
            The alis options (the compilation option alis must be set to True).
            The dictionary keys:

            * 'nlow' : int
                -> The number of low spectral resolution computation. If nlow = -1 select all wavelengths.
            * 'hist' : bool, optional
                -> Activate history. If the key does not exist the history mode is not activated.
            * 'max_hist' : int, optional
                -> The max number of history (only if hist is True). Default 8e6.
            * 'njac' : int, optional
                -> The number of perturbed profiles. Default no Jacobian.

            Note: Optional for the dictionary keys indicate that the key is not required to be present.
        NBPHOTONS : int, optional
            The total number of photons used for the simulation. Default 1e9.
        DEPO : float, optional
            The Rayleigh depolarization factor (air). Default 0.0279.
        DEPO_WATER : float, optional
            The Rayleigh depolarization factor (water). Default 0.0906.
        THVDEG : float, optional
            The sun/viewing zenith angle in forward/backward mode, in degrees. This parameter is ignored 
            if the parameter `sensor` is used.
        PHVDEG : float, optional
            The sun/viewing azimuth angle in forward/backward mode, in degrees. This parameter is ignored 
            if the parameter `sensor` is used.
        SEED : int, optional
            The seed used to initiate the series of random numbers. Default based on clock time.
        RTER : float, optional 
            The earth radius in km
        wl_proba : None | 1-D ndarray, optional
            The inversed cumulative distribution function for wavelength selection. It is for example 
            the result of function ICDF(proba, N).
        sensor_proba : None | 1-D ndarray, optional
           The inversed cumulative distribution function for sensor selection. It is for example 
           the result of function ICDF(proba, N).
        cell_proba : None | 2-D ndarray, optional
            The inversed cumulative distribution function for cell selection. It is for example 
            the result of function ICDF2D(proba, N).
        NBTHETA : int, optional
            The number of viewing/sun zenith angles in forward/backward for the cone sampling.
            This parameter is ignored if the parameter `le` is used.
        NBPHI : int, optional
            The number of viewing/sun azimuth angles in forward/backward for the cone sampling.
            This parameter is ignored if the parameter `le` is used.
        NF : int, optional
            The number of discretization of:
                - the inversed aerosol phase functions
                - the inversed ocean phase functions
                - the inversed probability of each wavelength occurence
        OUTPUT_LAYERS : int, optional
            Which layers to consider. Possibilities are the following:
                - -1 -> consider no layer (for development purposes)
                -  0 -> up (TOA)
                -  1 -> up (TOA), down (0+) and up (0-)
                -  2 -> up (TOA), down (0-), up (0+) and down (B)
                -  3 -> consider all output layers.
                -  4 -> down (0+) and up (0-)
                -  5 -> down (0-), up (0+) and down (B)
                -  6 -> down (0-) and up (0+)
                -  7 -> up (TOA) and down (0+)

            Note: Consider only the needed layers may reduce significantly the computational time.
        XBLOCK : int, optional
            The number of cuda blocks.
        XGRID : int, optional
            The number of cuda grids.
        NBLOOP : None | float, optional
            The number of photons launched in one kernel run.
        progress : bool, optional
            Activate the progress bar. Default True.    
        le : None | dict, optional 
            Activate the Local Estimate method. The le dictionary keys:

            * 'th' : 1-D ndarray | list, optional
                -> The zenith angles in radians.
            * 'phi' : 1-D ndarray | list, optional
                -> The azimuth angles in radians.
            * 'th_deg' : 1-D ndarray | list, optional
                -> The zenith angles in degrees. Only if 'th' is not provided.
            * 'phi_deg' : 1-D ndarray | list, optional
                -> The azimuth angles in degrees. Only if 'phi' is not provided.
            * 'zip' : bool, optional
               -> If True, then 'th' and 'phi' covary and the output is only one-dimensional NBTHETA, 
               but user should verify that NBPHI==NBTHETA.
            * 'count_level' : 1-D ndarray | list, optional
                -> The level to consider. Possibilities: -2(all), -1(none), 0(UPTOA), 1(DOWN0P), 2(DOWN0M),
                   3(UP0P), 4(UP0M) or 5(DOWNB). The level to consider may change only with th/th_deg. The 
                   array must be of length NBTHETA. If the key is not present it will be the same as 
                   count_level = np.full_like(th/th_deg, -2, dtype=np.int32).
            
            Note: Optional for the dictionary keys indicate that the key is not required to be present. 
            If th/phi are not provided, th_deg/phi_deg must be given.     
        flux : None | str, optional
            Activate the flux mode (instead of radiance). Only 2 choices:
                - 'planar'
                - 'spherical'
        stdev : bool, optional
            Activate the calculation of the standard deviation (between each kernel run).
        stdev_lim : None | StdevLim, optional
            To stop the computation if the standard deviation is above a certain limit. Only if stdev is True.
        BEER : int, optional
            If BEER=1 compute absorption using Beer-Lambert law, otherwise compute it with the Single scattering albedo. 
            BEER automatically set to 1 if ALIS is True.
        RR: int, optional
            Activate the Russian Roulette. ON = 1 and OFF = 0.
        WEIGHTRR : float, optional
            The threshold weight to apply to the Russian Roulette.
        SZA_MAX : float, optional
            The maximum SZA value for solar BOXES in case a Regulard grid and cone sampling.
        SUN_DISC : float, optional
            The angular size of the Sun disc in degrees, 0 (default means no angular size)
        sensor : None | Sensor | list, optional
            The light source / sensor (Sensor object or list of Sensor objects) in forward / backward mode.
        refraction : bool, option
            If True include atmospheric refraction.
        reflectance : bool, optional
           Convert output to reflectance units, otherwise in radiance units with Solar irradiance set to PI. 
           Only of flux is None and for plane parallel atmosphere.
        myObjects : None | list, optional
            A list of 3d objects (Entity objects) that will be used in the simulation. Currently sphere and plane objects 
            are considered. The compilation option `obj3d` must be set to True.
        interval : None | list, optional
            A principal bounding box in case 3d objects are incorporated. It must be a list composed of 2 lists with the bbox 
            min and max values [[xmin, ymin, zmin], [xmax, ymax, zmax]].
        IsAtm : int, optional
            If IsAtm=0 provide more robust test with 3d objects in case the atmosphere we remove the atmosphere.
        cusL : None | CusForward | CusForward, optional
            Use the RF, FF (CusFroward) or B (CusBackward) launching modes. The compilation option `obj3d` must be set to True.
        SMIN : int, optional
            The minimum number of interactions (scattering/reflection). Default 0.
        SMAX : int, optional
            The maximum number of iteractions (scattering/reflection). Default 1e6.
        RMIN : int, optional
            The minimum number of reflections (by surface only, not environement). Default 0.
        RMAX : int, optional
            The maximum number of reflections (by surface only, not environement). Default 1e6
        FFS : bool, optional
            Forced First Scattering (for use in spherical limb geometry only). Default False.
        DIRECT : bool, optional
            Include directly transmitted photons. Default False.
        OCEAN_INTERACTION : None | int, optional
            If OCEAN_INTERACTION=1 select photons that interact with ocean. Default None, no selection.
        pol_off : bool, optional
            Deactivate (if True) the consideration of polarized light. Default False.
        no_aer_output : bool, optional
            Add output where only photons not scattered by aerosols are considered. Default False.
            For example, in output MLUT we have m['I_up (TOA)'], so now we will get also m['I_up (TOA), no_aer'].

        Returns
        -------
        out : MLUT
            A look-up table containing the simulation results and more, e.g.:
            - the polarized dimensionless reflectance (I,Q,U,V) at the different layers
            - the number of photons (N) received at each layer
            - the profiles and phase functions
            - attributes
            - ...

        Notes
        -----

        In cone sampling, the sun/sensor is targeting the origin (0,0,0) in forward/backward.
            
        Examples
        --------
        >>> from smartg.smartg import Smartg, RoughSurface
        >>> from smartg.atmosphere import AtmAFGL, AerOPAC
        >>> from smartg.water import IOP_1
        >>> atm = AtmAFGL('afglt', comp=[AerOPAC('maritime_clean', 0.5, 550.)])
        >>> water = IOP_1(chl=0.5, DEPTH=5.)
        >>> surf = RoughSurface(WIND=5., NH2O=1.34)
        >>> m = Smartg().run(wl=550., atm=atm, water=water, surf=surf)
        >>> # Look at the top of atmosphere radiance/reflectance (key: 'I_up (TOA)')
        >>> m['I_up (TOA)'].describe()
        LUT "I_up (TOA)" (float64 between 0.0704 and 0.161):
          Dim 0 (Azimuth angles): 90 values in [0.0, 356.0]
          Dim 1 (Zenith angles): 45 values in [1.0, 89.0]
        >>> m['I_up (TOA)'].data
        array([[0.15751, 0.14705, 0.14806, ..., 0.12496, 0.11015, 0.07649],
               [0.15424, 0.14913, 0.1453 , ..., 0.12659, 0.11015, 0.07356],
               [0.14755, 0.14612, 0.14571, ..., 0.12354, 0.11139, 0.07547],
               ...,
               [0.15698, 0.14824, 0.14262, ..., 0.12455, 0.10815, 0.07464],
               [0.15662, 0.15111, 0.14541, ..., 0.12609, 0.11155, 0.07262],
               [0.15648, 0.14717, 0.14103, ..., 0.12593, 0.10935, 0.07744]], shape=(90, 45))

        """

        if (not self.pp and water is not None): raise NameError("Ocean + spherical atm is not allowed! Still in progress...")

        if ( not (OUTPUT_LAYERS in (np.arange(9, dtype=np.int32)-1)) ):
            raise ValueError('The OUTPUT_LAYERS value must be an integer between -1 and 7.')

        # Compute the sun direction as vector 
        vSun = convertAnglestoV(THETA=THVDEG, PHI=PHVDEG, TYPE="Sun") 
        vSun = Normalize(vSun)

        # First check if back option is activated in case of the use of cusBackward launching mode
        surfLPH = 0
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
                if (cusL.dict['TYPE'] == 2 and cusL.dict['FOV'] > 1e-6): #isotropic
                    surfLPH = float(cusL.dict['CFX'])*float(cusL.dict['CFY'])
                else:
                    surfLPH = float(cusL.dict['CFX'])*float(cusL.dict['CFY'])*DotNN

        #
        # initialization
        #              
        
        # Begin initialization with OBJ ============================
        if (myObjects is not None):
            # Main bounding box initialization
            if interval is not None:
                Pmin_x = interval[0][0];Pmin_y = interval[0][1];Pmin_z = interval[0][2]
                Pmax_x = interval[1][0];Pmax_y = interval[1][1];Pmax_z = interval[1][2]
            else:
                Pmin_x = -100000; Pmin_y = -100000; Pmin_z = 0
                Pmax_x = 100000;  Pmax_y = 100000; Pmax_z = 120

            # Initiliaze all the parameters linked with 3D objects
            (nGObj, nObj, nRObj, surfLPH_RF, nb_H, zAlt_H, totS_H, TC, nbCx, nbCy,
             myObjects0, myGObj0, myRObj0, mySPECTObj0, n_cos) = initObj(LGOBJ=myObjects, vSun=vSun, wl=wl, CUSL=cusL)

            # If we are in RF mode don't forget to update the value of surfLPH
            if (surfLPH_RF is not None): surfLPH = surfLPH_RF

        else:
            myObjects0 = gpuzeros(1, dtype=np.uint32)
            #myObjects0 = gpuzeros(1, dtype='int32')
            myGObj0 = gpuzeros(1, dtype='int32')
            myRObj0 = gpuzeros(1, dtype='int32')
            mySPECTObj0 = gpuzeros(1, dtype='int32') # normally 2 dims: obj dim + wl dim
            nObj = 0; nGObj=0; nRObj=0; Pmin_x = None; Pmin_y = None; Pmin_z = None
            Pmax_x = None; Pmax_y = None; Pmax_z = None
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
        # Maximum number of photons histories (alis=True and alis_options['hist'] = True), otherwise 0 (no histories)
        MAX_HIST = np.int64(1)
        MAX_NLOW = 801

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
                if alis_options['hist']: 
                    hist=True
                    if 'max_hist' in alis_options.keys():
                        MAX_HIST=np.int64(alis_options['max_hist'])
                    else : MAX_HIST=np.int64(8000000)
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
        elif (isinstance(atm, MLUT) or (atm is None)):
            prof_atm = atm
        else:
            raise NameError('atm must be an Atmosphere class or an MLUT class or equal to None!')

        if (isinstance(atm, MLUT)):
            ZTOA = atm.axes['z_atm'][0]
        else:
            ZTOA = 120.
  
        if prof_atm is not None:
            faer = calculF(prof_atm, NF, DEPO, kind='atm', pol_off=pol_off)
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
        #X0, _ = impactInit(prof_atm, NLAM, THVDEG, RTER, self.pp)
        X0, tabTransDir_analytic = impactInit(prof_atm, NLAM, THVDEG, RTER, self.pp)

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
        elif isinstance(sensor, Sensor):
            sensor2=[sensor]
        elif isinstance(sensor, list):
            sensor2=sensor
        else:
            raise NameError('sensor must be a Sensor class, a list or Sensor classes or equal to None!')

        NSENSOR=len(sensor2)

        tab_sensor = np.zeros(NSENSOR, dtype=type_Sensor, order='C')
        for (i,s) in enumerate(sensor2) :
            for k in s.dict.keys():
                  tab_sensor[i][k] = s.dict[k]
        tab_sensor = to_gpu(tab_sensor)

        # The min and max posx and posy of sensors. Useful for forward mode in 3d atm
        sxmin = np.inf
        sxmax = -np.inf
        symin = np.inf
        symax = -np.inf
        for sens in sensor2:
            if (sens.cell_size > 0):
                half_csize = 0.5*sens.cell_size
                sxmin = min(sxmin, sens.dict['POSX']-half_csize)
                sxmax = max(sxmax, sens.dict['POSX']+half_csize)
                symin = min(symin, sens.dict['POSY']-half_csize)
                symax = max(symax, sens.dict['POSY']+half_csize)
            else:
                sxmin = min(sxmin, sens.dict['POSX'])
                sxmax = max(sxmax, sens.dict['POSX'])
                symin = min(symin, sens.dict['POSY'])
                symax = max(symax, sens.dict['POSY'])
        if sensor2[0].cell_size > 0:
            nbsx = round((sxmax - sxmin)/sensor2[0].cell_size)
            nbsy = round((symax - symin)/sensor2[0].cell_size)
        else:
            nbsx = 0
            nbsy = 0


        #
        # ocean
        #
        if isinstance(water, IOP_base):
            prof_oc = water.calc(wl)
        elif(isinstance(water, MLUT) or (water is None)):
            prof_oc = water
        else:
            raise NameError('water must be an IOP_base class or equal to None!')

        if prof_oc is not None:
            foce = calculF(prof_oc, NF, DEPO_WATER, kind='oc', pol_off=pol_off)
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
        envmap = np.zeros(1, dtype=type_EnvMap)
        spectrum['lambda'] = wl[:]
        if env is None:
            # default values (no environment effect)
            env = Environment()
            if surf is not None :
                if surf.alb is not None:
                    spectrum['alb_surface'] = surf.alb.get(wl[:])
                elif surf.kp is not None:
                    spectrum['alb_surface'] = surf.kp[0].get(wl[:])
                    spectrum['k1p_surface'] = surf.kp[1].get(wl[:])
                    spectrum['k2p_surface'] = surf.kp[2].get(wl[:])
                    spectrum['k3p_surface'] = surf.kp[3].get(wl[:])
                else:
                    spectrum['alb_surface'] = -999.
            else:
                spectrum['alb_surface'] = -999.
        else:
            assert surf is not None
            if surf.alb is not None:
               spectrum['alb_surface'] = surf.alb.get(wl[:])
            elif surf.kp is not None :
               spectrum['alb_surface'] = surf.kp[0].get(wl[:])
               spectrum['k1p_surface'] = surf.kp[1].get(wl[:])
               spectrum['k2p_surface'] = surf.kp[2].get(wl[:])
               spectrum['k3p_surface'] = surf.kp[3].get(wl[:])
            albenv = env.alb.get(wl[:])
            if albenv.ndim==2:
                env.NENV = albenv.shape[1]
                spectrum['alb_envs'][:,:env.NENV] = albenv
                shp = env.alb.map.data.shape
                env.NXENVMAP = shp[0]
                env.NYENVMAP = shp[1]
                size = shp[0]*shp[1]
                envmap = np.zeros(shp, dtype=type_EnvMap)
                X, Y = np.meshgrid(env.alb.map.axis('X'), env.alb.map.axis('Y'), indexing='ij')
                envmap['x'] = X
                envmap['y'] = Y
                envmap['env_index']=env.alb.get_map(X,Y)         
            else:
                spectrum['alb_env'] = albenv

        if water is None:
            spectrum['alb_seafloor'] = -999.
        else:
            spectrum['alb_seafloor'] = prof_oc['albedo_seafloor'].data[...]

        envmap = to_gpu(envmap)
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
            
            if 'count_level' in le:
                le['count_level'] = np.array(le['count_level'], dtype='int32').ravel()
                assert len(le['count_level']) == NBTHETA



        FLUX = 0
        if flux is not None:
            LE=0
            if flux== 'planar' : 
                FLUX = 1
            if flux== 'spherical' : 
                FLUX = 2
            if flux== 'tilted planar' : 
                FLUX = 3


        if wl_proba is not None:
            assert wl_proba.dtype == 'int64'
            wl_proba_icdf = to_gpu(wl_proba)
            NWLPROBA = len(wl_proba_icdf)
        else:
            wl_proba_icdf = gpuzeros(1, dtype='int64')
            NWLPROBA = 0

        if sensor_proba is not None:
            assert sensor_proba.dtype == 'int64'
            sensor_proba_icdf = to_gpu(sensor_proba)
            NSENSORPROBA = len(sensor_proba_icdf)
        else:
            sensor_proba_icdf = gpuzeros(1, dtype='int64')
            NSENSORPROBA = 0

        if cell_proba is not None:
            if (cell_proba == 'auto') and not self.back and self.thermal:
                kabs = od2k(prof_atm, 'OD_abs_atm')
                z = -prof_atm.axis('z_atm')
                B = BPlanck(wl[:, None], prof_atm['T_atm'].data[None, :])
                Emission     = LUT(kabs * B, axes = [wl, z], names= ['wavelength','z_atm'])
                Norm_Emission = (4*np.pi) * Emission.reduce(np.sum, 'z_atm')
                P_Emission   = Emission * (4*np.pi) / Norm_Emission
                cell_proba_icdf   = to_gpu(ICDF2D(P_Emission.data).T)
                NCELLPROBA = cell_proba_icdf.shape[0]
            else:
                assert cell_proba.shape[1] == NLAM
                cell_proba_icdf = to_gpu(cell_proba)
                NCELLPROBA = cell_proba.shape[0]
        else:
            cell_proba_icdf = gpuzeros(1, dtype='int64')
            NCELLPROBA = 0

        REFRAC = 0
        if refraction: 
            REFRAC=1

        HORIZ = 1
        if (not self.pp and not reflectance): HORIZ = 0

        # initialization of the constants
        InitConst(surf, env, NATM, NATM_ABS, NOCE, NOCE_ABS, self.mod,
                  NBPHOTONS, NBLOOP, THVDEG, DEPO,
                  XBLOCK, XGRID, NLAM, SIM, NF,
                  NBTHETA, NBPHI, OUTPUT_LAYERS,
                  RTER, LE, ZIP, FLUX, FFS, DIRECT, OCEAN_INTERACTION, NLVL, NPSTK,
                  NWLPROBA, NSENSORPROBA, NCELLPROBA, BEER, SMIN, SMAX, RMIN, RMAX, RR, WEIGHTRR, NLOW, NJAC, 
                  NSENSOR, REFRAC, HORIZ, SZA_MAX, SUN_DISC, cusL, nObj, nGObj, nRObj,
                  Pmin_x, Pmin_y, Pmin_z, Pmax_x, Pmax_y, Pmax_z, IsAtm,
                  TC, nbCx, nbCy, vSun, HIST, ZTOA, sensor2[0].cell_size,
                  sxmin, sxmax, symin, symax, nbsx, nbsy, no_aer_output)

        # Initialize the progress bar
        p = Progress(NBPHOTONS, progress)

        # Initialize the RNG
        SEED = self.rng.setup(SEED, XBLOCK, XGRID)

        # Loop and kernel call
        (NPhotonsInTot, tabPhotonsTot, tabPhotonsTotNoAer, tabDistTot, tabHistTot, tabTransDir, errorcount, 
         NPhotonsOutTot, NPhotonsOutTotNoAer, sigma, Nkernel, secs_cuda_clock, cMatVisuRecep, matCats, matLoss, wPhCats, wPhCats2
        ) = loop_kernel(NBPHOTONS, faer, foce,
                        NLVL, NATM, NATM_ABS, NOCE, NOCE_ABS, MAX_HIST, NLOW, NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                        NLAM, NSENSOR, self.double, self.kernel, self.kernel2, p, X0, le, tab_sensor, envmap, spectrum,
                        prof_atm_gpu, prof_oc_gpu, cell_atm_gpu, cell_oc_gpu,
                        wl_proba_icdf, sensor_proba_icdf, cell_proba_icdf, stdev, stdev_lim, self.rng, self.alis,
                        myObjects0, TC, nbCx, nbCy, myGObj0, myRObj0, mySPECTObj0, hist=hist)

        attrs['kernel time (s)'] = secs_cuda_clock
        attrs['number of kernel iterations'] = Nkernel
        attrs['seed'] = SEED
        attrs.update(self.common_attrs)

        # If there is a receiver -> normalization of the signal collected
        if (TC is not None):
            cMatVisuRecep, matCats, n_cte = normalizeRecIrr(cMatVisuRecep=cMatVisuRecep, matCats=matCats,
                nbCx=nbCx, nbCy=nbCy, NBPHOTONS=float(np.sum(NPhotonsInTot)), surfLPH=surfLPH, TC=TC, cusL=cusL,
                SUN_DISC=SUN_DISC, LE=LE)

        if (nb_H > 0 and TC is not None and cusL is not None):
            MZAlt_H = zAlt_H/nb_H; SREC=TC*TC*nbCx*nbCy #; weightR=matCats[2, 1]
            # dicSTP : tuple incorporating parameters for Solar Tower Power applications
            if(self.back) : ALDEG = cusL.dict['ALDEG']
            else : ALDEG = 0.
            dicSTP = {"nb_H":nb_H, "n_cos": n_cos, "totS_H":totS_H, "surfTOA":surfLPH, "MZAlt_H":MZAlt_H, "vSun":vSun, "wRec":matCats[2, 1],
                      "SREC":SREC, "TC":TC, "LPH":cusL.dict['LPH'], "LPR":cusL.dict['LPR'], "prog":progress, "n_cte":n_cte, "ALDEG":ALDEG}
        # If there are no heliostats --> no analyses of optical losses
        elif(TC is not None and cusL is not None):
            SREC=TC*TC*nbCx*nbCy; matLoss = None #;weightR=matCats[2, 1]
            if(self.back) : ALDEG = cusL.dict['ALDEG']
            else : ALDEG = 0.
            dicSTP = {"vSun":vSun, "wRec":matCats[2, 1], "SREC":SREC, "TC":TC, "LPH":cusL.dict['LPH'],
                      "LPR":cusL.dict['LPR'], "prog":progress, "n_cte":n_cte, "ALDEG":ALDEG}
        elif(TC is not None):
            SREC=TC*TC*nbCx*nbCy; matLoss = None
            dicSTP = {"vSun":vSun, "SREC":SREC, "TC":TC, "n_cte":n_cte}
        # If there are no heliostats and receiver --> there is no STP
        else: 
            dicSTP = None; matLoss = None #; weightR=0
                
        # finalization
        output = finalize(tabPhotonsTot, tabPhotonsTotNoAer, tabDistTot, tabHistTot, wl[:], NPhotonsInTot, errorcount,
                          NPhotonsOutTot, NPhotonsOutTotNoAer, OUTPUT_LAYERS, tabTransDir, tabTransDir_analytic, SIM,
                          attrs, prof_atm, prof_oc, sigma, THVDEG, HORIZ, le=le, flux=flux, back=self.back, 
                          SZA_MAX=SZA_MAX, SUN_DISC=SUN_DISC, hist=hist, cMatVisuRecep=cMatVisuRecep,
                          dicSTP=dicSTP, matCats=matCats, matLoss=matLoss, wPhCats=wPhCats, wPhCats2=wPhCats2,
                          no_aer_output=no_aer_output)
        
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
        
        if not self.autoinit and not self.keep_context:
            self.ctx.pop()
            self.ctx.detach()
            self.ctx = None
            from pycuda.tools import clear_context_caches
            clear_context_caches()

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


def finalize(tabPhotonsTot, tabPhotonsTotNoAer, tabDistTot, tabHistTot, wl, NPhotonsInTot, errorcount, NPhotonsOutTot,
             NPhotonsOutTotNoAer, OUTPUT_LAYERS, tabTransDir, tabTransDir_analytic, SIM, attrs, prof_atm, prof_oc,
             sigma, THVDEG, HORIZ, le=None, flux=None,
             back=False, SZA_MAX=90., SUN_DISC=0, hist=False, cMatVisuRecep = None,
             dicSTP = None, matCats=None, matLoss=None, wPhCats=None, wPhCats2=None, no_aer_output=False):
    '''
    create and return the final output
    '''
    (_,_,NSENSOR,NLAM,NBTHETA,NBPHI) = tabPhotonsTot.shape

    # normalization in case of radiance
    # (broadcast everything to dimensions (LVL,NPSTK,SENSOR,LAM,THETA,PHI))
    norm_npho = NPhotonsInTot.reshape((1,1,NSENSOR,NLAM,1,1))
    if flux is None:
        if le!=None : 
            tabTh = le['th']
            tabPhi = le['phi']
            if 'zip' not in le.keys():
                zip = False
            else : zip = le['zip']
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
    tabFinalNoAer = tabPhotonsTotNoAer.astype('float64')/(norm_geo*norm_npho)
    tabDistFinal = tabDistTot.astype('float64')
    #if hist : tabHistFinal = tabHistTot

    # swapaxes : (th, phi) -> (phi, theta)
    tabFinal = tabFinal.swapaxes(4,5)
    tabFinalNoAer = tabFinalNoAer.swapaxes(4,5)
    if len(tabDistFinal) >1 : tabDistFinal = tabDistFinal.swapaxes(3,4)
    if hist : tabHistTot = tabHistTot.swapaxes(3,4)
    NPhotonsOutTot = NPhotonsOutTot.swapaxes(3,4)
    NPhotonsOutTotNoAer = NPhotonsOutTotNoAer.swapaxes(3,4)
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
    if hist : m.add_dataset('Nphotons_in',  NPhotonsInTot)

    iphi     = slice(None)
    m.set_attr('zip', 'False')
    m.set_attr('NPhotonIn_sum', np.sum(NPhotonsInTot))

    if le is not None: m.set_attr('LE', int(1))
    else: m.set_attr('LE', int(0))

    if le is not None:
        if 'zip' in le:
            if le['zip'] : 
                m.set_attr('zip', 'True')
                iphi = 0
            else:
                axnames.insert(0, 'Azimuth angles')
                axnames2.insert(1,'Azimuth angles')
        else:
            axnames.insert(0, 'Azimuth angles')
            axnames2.insert(1,'Azimuth angles')
    else:
        axnames.insert(0, 'Azimuth angles')
        axnames2.insert(1,'Azimuth angles')

    m.add_axis('Zenith angles', tabTh*180./np.pi)
    m.add_axis('Azimuth angles', tabPhi*180./np.pi)
    
    axnames4=[]
    if NLAM > 1:
        m.add_axis('wavelength', wl)
        ilam = slice(None)
        axnames.insert(0, 'wavelength')
        axnames4.insert(0, 'wavelength')
    else:
        m.set_attr('wavelength', str(wl))
        ilam = 0

    if NSENSOR > 1:
        m.add_axis('sensor index', np.arange(NSENSOR))
        isen = slice(None)
        axnames.insert(0, 'sensor index')
        axnames2.insert(1, 'sensor index')
        axnames4.insert(0, 'sensor index')
    else:
        isen=0

    write_UPTOA = OUTPUT_LAYERS in (0, 1, 2, 3, 7)
    write_DOWN0P = OUTPUT_LAYERS in (1, 3, 4, 7)
    write_DOWN0M = OUTPUT_LAYERS in (2, 3, 5, 6)
    write_UP0P = OUTPUT_LAYERS in (2, 3, 5, 6)
    write_UP0M = OUTPUT_LAYERS in (1, 3, 4)
    write_DOWNB = OUTPUT_LAYERS in (2, 3, 5)

    if write_UPTOA:
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
        if no_aer_output:
            m.add_dataset('I_up (TOA), no_aer', tabFinalNoAer[UPTOA,0,isen,ilam,iphi,:], axnames)
            m.add_dataset('Q_up (TOA), no_aer', tabFinalNoAer[UPTOA,1,isen,ilam,iphi,:], axnames)
            m.add_dataset('U_up (TOA), no_aer', tabFinalNoAer[UPTOA,2,isen,ilam,iphi,:], axnames)
            m.add_dataset('V_up (TOA), no_aer', tabFinalNoAer[UPTOA,3,isen,ilam,iphi,:], axnames)
            m.add_dataset('N_up (TOA), no_aer', NPhotonsOutTotNoAer[UPTOA,isen,ilam,iphi,:], axnames)
        if len(tabDistFinal) > 1: 
            if zip : m.add_dataset('cdist_up (TOA)', np.squeeze(tabDistFinal[UPTOA,:,isen,:,:]),  ['None','Zenith angles','iAMF'])
            else   : m.add_dataset('cdist_up (TOA)', tabDistFinal[UPTOA,:,isen,:,:,:],['None','Azimuth angles','Zenith angles','iAMF'])
    
    if hist : m.add_dataset('histories', tabHistTot)
    
    if write_DOWN0P:
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
        if no_aer_output:
            m.add_dataset('I_down (0+), no_aer', tabFinalNoAer[DOWN0P,0,isen,ilam,iphi,:], axnames)
            m.add_dataset('Q_down (0+), no_aer', tabFinalNoAer[DOWN0P,1,isen,ilam,iphi,:], axnames)
            m.add_dataset('U_down (0+), no_aer', tabFinalNoAer[DOWN0P,2,isen,ilam,iphi,:], axnames)
            m.add_dataset('V_down (0+), no_aer', tabFinalNoAer[DOWN0P,3,isen,ilam,iphi,:], axnames)
            m.add_dataset('N_down (0+), no_aer', NPhotonsOutTotNoAer[DOWN0P,isen,ilam,iphi,:], axnames)
        if len(tabDistFinal) > 1: 
            if zip : m.add_dataset('cdist_down (0+)', np.squeeze(tabDistFinal[DOWN0P,:,isen,:,:]),  ['None','Zenith angles','iAMF'])
            else   : m.add_dataset('cdist_down (0+)', tabDistFinal[DOWN0P,:,isen,:,:,:],['None','Azimuth angles','Zenith angles','iAMF'])
    if write_UP0M:
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
        if no_aer_output:
            m.add_dataset('I_up (0-), no_aer', tabFinalNoAer[UP0M,0,isen,ilam,iphi,:], axnames)
            m.add_dataset('Q_up (0-), no_aer', tabFinalNoAer[UP0M,1,isen,ilam,iphi,:], axnames)
            m.add_dataset('U_up (0-), no_aer', tabFinalNoAer[UP0M,2,isen,ilam,iphi,:], axnames)
            m.add_dataset('V_up (0-), no_aer', tabFinalNoAer[UP0M,3,isen,ilam,iphi,:], axnames)
            m.add_dataset('N_up (0-), no_aer', NPhotonsOutTotNoAer[UP0M,isen,ilam,iphi,:], axnames)
        if len(tabDistFinal) > 1: 
            if zip : m.add_dataset('cdist_up (0-)', np.squeeze(tabDistFinal[UP0M,:,isen,:,:]),  ['None','Zenith angles','iAMF'])
            else   : m.add_dataset('cdist_up (0-)', tabDistFinal[UP0M,:,isen,:,:,:],['None','Azimuth angles','Zenith angles','iAMF'])

    if write_DOWN0M:
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
        if no_aer_output:
            m.add_dataset('I_down (0-), no_aer', tabFinalNoAer[DOWN0M,0,isen,ilam,iphi,:], axnames)
            m.add_dataset('Q_down (0-), no_aer', tabFinalNoAer[DOWN0M,1,isen,ilam,iphi,:], axnames)
            m.add_dataset('U_down (0-), no_aer', tabFinalNoAer[DOWN0M,2,isen,ilam,iphi,:], axnames)
            m.add_dataset('V_down (0-), no_aer', tabFinalNoAer[DOWN0M,3,isen,ilam,iphi,:], axnames)
            m.add_dataset('N_down (0-), no_aer', NPhotonsOutTotNoAer[DOWN0M,isen,ilam,iphi,:], axnames)
        if len(tabDistFinal) > 1: 
            if zip : m.add_dataset('cdist_down (0-)', np.squeeze(tabDistFinal[DOWN0M,:,isen,:,:]),  ['None','Zenith angles','iAMF'])
            else   : m.add_dataset('cdist_down (0-)', tabDistFinal[DOWN0M,:,isen,:,:,:],['None','Azimuth angles','Zenith angles','iAMF'])
    if write_UP0P:
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
        if no_aer_output:
            m.add_dataset('I_up (0+), no_aer', tabFinalNoAer[UP0P,0,isen,ilam,iphi,:], axnames)
            m.add_dataset('Q_up (0+), no_aer', tabFinalNoAer[UP0P,1,isen,ilam,iphi,:], axnames)
            m.add_dataset('U_up (0+), no_aer', tabFinalNoAer[UP0P,2,isen,ilam,iphi,:], axnames)
            m.add_dataset('V_up (0+), no_aer', tabFinalNoAer[UP0P,3,isen,ilam,iphi,:], axnames)
            m.add_dataset('N_up (0+), no_aer', NPhotonsOutTotNoAer[UP0P,isen,ilam,iphi,:], axnames)
        if len(tabDistFinal) > 1: 
            if zip : m.add_dataset('cdist_up (0+)', np.squeeze(tabDistFinal[UP0P,:,isen,:,:]),  ['None','Zenith angles','iAMF'])
            else   : m.add_dataset('cdist_up (0+)', tabDistFinal[UP0P,:,isen,:,:,:],['None','Azimuth angles','Zenith angles','iAMF'])
    if write_DOWNB:
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
        if no_aer_output:
            m.add_dataset('I_down (B), no_aer', tabFinalNoAer[DOWNB,0,isen,ilam,iphi,:], axnames)
            m.add_dataset('Q_down (B), no_aer', tabFinalNoAer[DOWNB,1,isen,ilam,iphi,:], axnames)
            m.add_dataset('U_down (B), no_aer', tabFinalNoAer[DOWNB,2,isen,ilam,iphi,:], axnames)
            m.add_dataset('V_down (B), no_aer', tabFinalNoAer[DOWNB,3,isen,ilam,iphi,:], axnames)
            m.add_dataset('N_down (B), no_aer', NPhotonsOutTotNoAer[DOWNB,isen,ilam,iphi,:], axnames)
        if len(tabDistFinal) > 1: 
            if zip : m.add_dataset('cdist_down (B', np.squeeze(tabDistFinal[DOWNB,:,isen,:,:]),  ['None','Zenith angles','iAMF'])
            else   : m.add_dataset('cdist_down (B)', tabDistFinal[DOWNB,:,isen,:,:,:],['None','Azimuth angles','Zenith angles','iAMF'])


    # write atmospheric profiles
    if prof_atm is not None:
        # direct transmission
        m.add_dataset('direct transmission', tabTransDir_analytic,
                   axnames=['wavelength'])
        m.add_dataset('direct transmission (dev)', np.exp(-tabTransDir[isen,ilam]), axnames4)

        m.add_lut(prof_atm['n_atm'])
        m.add_lut(prof_atm['T_atm'])
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
        m.add_lut(prof_oc['T_oc'])
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
        # Indice 0 = Sum of all Cats, then cat1 to cat8, def of cats -> see Moulana et al, 2019
        m.add_axis('Categories', np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32))
        var_x, var_y = np.shape(cMatVisuRecep[0][:][:])
        x_indices = np.arange(var_x); y_indices = np.arange(var_y)
        m.add_axis('X_Cell_Index', x_indices); m.add_axis('Y_Cell_Index', y_indices)
        m.add_dataset('C_Receiver', cMatVisuRecep[:][:][:], ['Categories', 'X_Cell_Index', 'Y_Cell_Index'])
        m.set_attr('S_Receiver', str(dicSTP["SREC"])) # Receiver surface in km²
        m.set_attr('S_Cell', str(dicSTP["TC"]))       # Cell surface in km²
        # half-angle of the receiver solid angle
        if (back == True) : m.set_attr('ALDEG', str(dicSTP["ALDEG"]))
        else : m.set_attr('ALDEG', str(90))

    if (matCats is not None):
        m.add_dataset('cat_PhNb', matCats[:,0], ['Categories'])
        m.add_dataset('cat_w', matCats[:,1], ['Categories'])
        m.add_dataset('cat_w2', matCats[:,2], ['Categories'])
        m.add_dataset('cat_irr', matCats[:,3], ['Categories'])
        m.add_dataset('cat_errAbs', matCats[:,4], ['Categories'])
        m.add_dataset('cat_err%', matCats[:,5], ['Categories'])
        
        arrWC = np.zeros((9, NLAM), dtype=np.float64)
        arrWC2 = np.zeros((9, NLAM), dtype=np.float64)
            
        arrWC[0, ilam] = np.sum(wPhCats[:, ilam], axis=0)
        arrWC[1:, ilam] = wPhCats[:, ilam]
        
        arrWC2[0, ilam] = np.sum(wPhCats2[:, ilam], axis=0)
        arrWC2[1:, ilam] = wPhCats2[:, ilam]
        
        axe_wPh = ['Categories']
        if (NLAM > 1): axe_wPh.append('wavelength')
            
        m.add_dataset('wPhCats', arrWC[:,ilam], axe_wPh)
        m.add_dataset('wPhCats2', arrWC2[:,ilam], axe_wPh)
        m.add_dataset('norm_npho', norm_npho[0,0,0,:,0,0], ['wavelength'])
        
        m.set_attr('n_cte', str(dicSTP["n_cte"]))

    if (matLoss is not None):
        m.add_dataset('wLoss', np.array(matLoss[:,0], dtype=np.float64), ['index'])
        m.add_dataset('wLoss2', np.array(matLoss[:,1], dtype=np.float64), ['index'])
        m.set_attr('n_cos', str(dicSTP["n_cos"]))
        
        # To consider also the multispectral case
        if (NLAM > 1) : lwl = len(wl)
        else : lwl = 1

        # ======== Find the extinction between TOA and heliostats
        tau_ext = np.zeros(lwl, dtype=np.float64)
        Tr_tau = np.zeros(lwl, dtype=np.float64)
        P_pyt = np.zeros(lwl, dtype=np.float64)

        # find the atm layer where the mean heliostats z altitude is located
        Ci = 0
        while(prof_atm.axis('z_atm')[Ci] > dicSTP["MZAlt_H"]):
            Ci += 1

        for i in range (0, lwl):
            tau_ext[i] = (prof_atm['OD_atm'].data[i,Ci] -  prof_atm['OD_atm'].data[i,Ci-1]) * \
                         (dicSTP["MZAlt_H"]/prof_atm.axis('z_atm')[Ci-1])
            tau_ext[i] = prof_atm['OD_atm'].data[i,Ci] - tau_ext[i]
            # Beer-Lamber law to find the transmisttance
            Tr_tau[i] = np.exp(-abs(tau_ext[i]/-dicSTP["vSun"].z))
            # theoric computation of the total power collected by all the heliostats
            P_pyt[i] = Tr_tau[i]*dicSTP["totS_H"]*1e6 # mult by 1e6 to convert km² to m²
        # Save results
        m.add_dataset('n_tr', Tr_tau, ['wavelength'])
        m.add_dataset('powc_H', P_pyt, ['wavelength'])
        # ========

        # === Here allows the calculation of the analytical approx of n_atm in backward ->
        if (back and (dicSTP["LPH"] is not None) and (dicSTP["LPR"] is not None)):
            naatm = np.zeros(lwl, dtype=np.float64)
            p = Progress(lwl-1, dicSTP["prog"])
            for j in range (0, lwl):
                SUM_naatm=0
                p.update(j+1, 'n_aatm computed : {:.3g} / {:.3g}'.format(j+1, lwl))
                for i in range (len(dicSTP["LPH"])):
                    SUM_naatm += findExtinction(dicSTP["LPH"][i], dicSTP["LPR"][0], prof_atm, j)
                naatm[j] = SUM_naatm/len(dicSTP["LPH"])
            p.finish('Done! | Analytic approx of n_atm computed for {:.3g} wavelengths'.format(lwl))
            m.add_dataset('n_aatm', naatm, ['wavelength'])
        # ===

    return m


def isotropic(N):
    '''
    isotropic phase function, incl. cumulative
    over N angles
    '''
    phase_H = np.zeros(N, dtype=type_Phase, order='C')
    angles = np.linspace(0., pi, int(N), endpoint=True, dtype=np.float64)
    scum = [0]
    norm = 0.5
    phase= np.zeros((4,N), dtype='float64') 
    phase[0,:] = 0.5/norm
    phase[1,:] = 0.5/norm
    phase[2,:] = 0.5/norm
    phase[3,:] = 0.5/norm
    pm = phase[1, :] + phase[0, :]
    sin = np.sin(angles)
    dtheta = np.diff(angles)
    tmp = dtheta * ((sin[:-1] * pm[:-1] + sin[1:] * pm[1:]) / 3.
                    + (sin[:-1] * pm[1:] + sin[1:] * pm[:-1])/6.) * np.pi * 2.
    scum = np.append(scum,tmp)
    scum = np.cumsum(scum)
    scum /= scum[-1]

    # probability between 0 and 1
    z = (np.arange(N, dtype='float64')+1)/N
    angN = (np.arange(N, dtype='float64'))/(N-1)*np.pi
    f1 = interp1d(angles, phase[0,:])
    f2 = interp1d(angles, phase[1,:])
    f3 = interp1d(angles, phase[2,:])
    f4 = interp1d(angles, phase[3,:])

    # parameters equally spaced in scattering probability
    phase_H['p_P11'][:] = interp1d(scum, phase[0,:])(z)  # I par P11
    phase_H['p_P22'][:] = interp1d(scum, phase[1,:])(z)  # I per P22
    phase_H['p_P33'][:] = interp1d(scum, phase[2,:])(z)  # U P33
    phase_H['p_P43'][:] = interp1d(scum, phase[3,:])(z)  # V P43
    phase_H['p_P44'][:] = interp1d(scum, phase[2,:])(z)  # V P44= P33
    phase_H['p_ang'][:] = interp1d(scum, angles)(z) # angle

    # parameters equally spaced in scattering angle [0, 180]
    phase_H['a_P11'][:] = f1(angN)  # I par P11
    phase_H['a_P22'][:] = f2(angN)  # I per P22
    phase_H['a_P33'][:] = f3(angN)  # U P33
    phase_H['a_P43'][:] = f4(angN)  # V P43
    phase_H['a_P44'][:] = f3(angN)  # V P44=P33


    return phase_H



def rayleigh(N, DEPO, pol_off=False):
    """
    Rayleigh phase function, incl. cumulative over N angles
    
    Parameters
    ----------
    N : int
        number of angles
    DEPO : float
        depolarization coefficient
    pol_off : bool, optional

    Returns
    -------
    out : type_Phase
        the rayleigh phase information
    """
    pha = np.zeros(N, dtype=type_Phase, order='C')

    GAMA = DEPO / (2- DEPO)
    DELTA = np.float32((1.0 - GAMA) / (1.0 + 2.0 *GAMA))
    DELTA_PRIM = np.float32(GAMA / (1.0 + 2.0*GAMA))
    BETA  = np.float32(3./2. * DELTA_PRIM)
    ALPHA = np.float32(1./8. * DELTA)
    A = np.float32(1. + BETA / (3.0 * ALPHA))

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

    DELTA_SECO = np.float32((1.0 - 3.0*GAMA) / (1.0 - GAMA))
    T_demi = (3.0/2.0)
    P22 = T_demi*(DELTA+DELTA_PRIM)
    P12 = T_demi*DELTA_PRIM
    P33bis = T_demi*DELTA
    P44bis = P33bis*DELTA_SECO

    if pol_off:
        # P(theta) -> phase matrix in Iperpar convention
        # F(theta) -> phase matrix in IQUV convention
        # from IQUV to IperIpar (in the case only IQUV F11 != 0 i.e. no polarisation)
        # P11 = ((3./8.)*DELTA*(cTh2[:]-1)) + 0.5
        # A_P11 = ((3./8.)*DELTA*(cTh2LE[:]-1)) + 0.5
        P11 = T_demi*(DELTA*cTh2[:] + DELTA_PRIM)
        A_P11 = T_demi*(DELTA*cTh2LE[:] + DELTA_PRIM)
        F11 = 0.5 * (P11 + 2*P12 + P22)
        A_F11 = 0.5 * (A_P11 + 2*P12 + P22)
        # pha['p_P11'][:] = P11
        # pha['p_P12'][:] = P11
        # pha['p_P22'][:] = P11

        # pha['p_ang'][:] = theta[:] # angle
        # pha['a_P11'][:] = A_P11
        # pha['a_P12'][:] = A_P11
        # pha['a_P22'][:] = A_P11
        pha['p_P11'][:] = 0.5*F11
        pha['p_P12'][:] = 0.5*F11
        pha['p_P22'][:] = 0.5*F11
        pha['p_ang'][:] = theta[:] # angle

        pha['a_P11'][:] = 0.5*A_F11
        pha['a_P12'][:] = 0.5*A_F11
        pha['a_P22'][:] = 0.5*A_F11
    else:
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


def calculF(profile, N, DEPO, kind, pol_off=False):
    """
    Calculate cumulated phase functions from profile

    Parameters
    ----------
    profile : MLUT
        The atmosphere/ocean profile.
    N : int
        The number of angles
    DEPO : float
        The depolarization factor 'atmospheric'.
    kind : str
        The kind of profile. Can be 'atm' or 'oc'.
    pol_off : bool, optional
        Deactivate polarization. Default False (meaning polarization is on).
    """

    name_phase = 'phase_{}'.format(kind)
    if name_phase in profile.datasets():
        nphases = profile[name_phase].shape[0]
    else:
        nphases = 0

    nphases += 2    # include Rayleigh and VRS phase function
    #nphases += 1   # include Rayleigh phase function

    # Initialize the cumulative distribution function
    if nphases > 0:
        shp = (nphases, N)
    else:
        shp = (1, N)
    phase_H = np.zeros(shp, dtype=type_Phase, order='C')

    # Set Rayleigh phase function or isotropic if DEPO <0
    if DEPO >=0 : phase_H[0,:] = rayleigh(N, DEPO, pol_off=pol_off)
    # no pol_off in isotropic because the function needs first to be corrected
    else : phase_H[0,:]        = isotropic(N) 
    if 'theta_'+kind in profile.axes:
        angles = profile.axis('theta_'+kind) * pi/180.
        assert angles[-1] < 3.15   # assert that angles are in radians
        dtheta = np.diff(angles)

    # Set VRS phase function
    phase_H[1,:] = rayleigh(N, 0.17)

    idx = 2
    #idx = 1
    for ipha in range(nphases-2):
    #for ipha in range(nphases-1):

        phase = profile[name_phase][ipha, :, :]  # ipha, stk, theta

        if pol_off:
            if (len(phase[:,0]) == 4):
                raise NameError("old profiles with only 4 stk available are not supported without polarization")
            # back to IQUV convention to obtain F11
            F11 = 0.5 * (phase[0,:] + 2*phase[1,:] + phase[4,:])
            # reset all values to 0
            phase[:,:] = 0.
            # reconvert to Iperpar but without considering polarization
            phase[0,:] = 0.5 * F11
            phase[1,:] = 0.5 * F11
            phase[4,:] = 0.5 * F11

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

        if (len(phase[:,0]) == 4): # spherical particle
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
        else: # non spherical particle
            f5 = interp1d(angles, phase[4,:])
            f6 = interp1d(angles, phase[5,:])

            scum = [0]
            pm = 0.5*(phase[0, :] + 2*phase[1, :] + phase[4, :])
            sin = np.sin(angles)
            tmp = dtheta * ((sin[:-1] * pm[:-1] + sin[1:] * pm[1:]) / 3.
                            + (sin[:-1] * pm[1:] + sin[1:] * pm[:-1])/6.) * np.pi * 2.
            scum = np.append(scum,tmp)
            scum = np.cumsum(scum)
            scum /= scum[-1]

            phase_H['p_P11'][idx, :] = interp1d(scum, phase[0,:])(z)  # I P11
            phase_H['p_P22'][idx, :] = interp1d(scum, phase[4,:])(z)  # I P22
            phase_H['p_P12'][idx, :] = interp1d(scum, phase[1,:])(z)  # P12=P21
            phase_H['p_P33'][idx, :] = interp1d(scum, phase[2,:])(z)  # U P33
            phase_H['p_P43'][idx, :] = interp1d(scum, phase[3,:])(z)  # V P43
            phase_H['p_P44'][idx, :] = interp1d(scum, phase[5,:])(z)  # V P44= P33
            phase_H['p_ang'][idx, :] = interp1d(scum, angles)(z) # angle

            phase_H['a_P11'][idx, :] = f1(angN)  # I par P11
            phase_H['a_P22'][idx, :] = f5(angN)  # I per P22
            phase_H['a_P12'][idx, :] = f2(angN)  # I per P22
            phase_H['a_P33'][idx, :] = f3(angN)  # U P33
            phase_H['a_P43'][idx, :] = f4(angN)  # V P43
            phase_H['a_P44'][idx, :] = f6(angN)  # V P44=P33
            #phase_H['a_P33'][idx, :] = f6(angN)  # V P44=P33

        idx += 1

    return to_gpu(phase_H)


def InitConst(surf, env, NATM, NATM_ABS, NOCE, NOCE_ABS, mod,
              NBPHOTONS, NBLOOP, THVDEG, DEPO,
              XBLOCK, XGRID,NLAM, SIM, NF,
              NBTHETA, NBPHI, OUTPUT_LAYERS,
              RTER, LE, ZIP, FLUX, FFS, DIRECT, OCEAN_INTERACTION, 
              NLVL, NPSTK, NWLPROBA, NSENSORPROBA, NCELLPROBA,  BEER, SMIN, SMAX, RMIN, RMAX, RR, 
              WEIGHTRR, NLOW, NJAC, NSENSOR, REFRAC, HORIZ, SZA_MAX, SUN_DISC, cusL, nObj, nGObj, nRObj,
              Pmin_x, Pmin_y, Pmin_z, Pmax_x, Pmax_y, Pmax_z, IsAtm, TC, nbCx, nbCy, vSun, HIST, ZTOA,
              cell_size, sxmin, sxmax, symin, symax, nbsx, nbsy, no_aer_output) :
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

    if (  (cusL is not None) and (cusL.dict['LMODE'] == "FF")  ):
        PZd = ZTOA+cusL.dict['CFTZ']
    else:
        PZd = ZTOA
    tTemp = PZd/-vSun.z
    PXd = -vSun.x * tTemp
    PYd = -vSun.y * tTemp

    def copy_to_device(name, scalar, dtype):
        cuda.memcpy_htod(mod.get_global(name)[0], np.array([scalar], dtype=dtype))

    # copy constants to device
    copy_to_device('NBLOOPd', NBLOOP, np.uint32)
    copy_to_device('NOCEd', NOCE, np.int32)
    copy_to_device('NOCE_ABSd', NOCE_ABS, np.int32)
    copy_to_device('OUTPUT_LAYERSd', OUTPUT_LAYERS, np.int32)
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
    copy_to_device('cell_sized', cell_size, np.float32)
    copy_to_device('sxmind', sxmin, np.float32)
    copy_to_device('sxmaxd', sxmax, np.float32)
    copy_to_device('symind', symin, np.float32)
    copy_to_device('symaxd', symax, np.float32)
    copy_to_device('nbsxd', nbsx, np.uint32)
    copy_to_device('nbsyd', nbsy, np.uint32)
    copy_to_device('no_aer_outd', int(no_aer_output), np.int32)
    if OCEAN_INTERACTION is None:
        copy_to_device('OCEAN_INTERACTIONd', -1, np.int32)
    else:
        copy_to_device('OCEAN_INTERACTIONd', 1 if OCEAN_INTERACTION else 0, np.int32)
    #copy_to_device('MId', MI, np.int32)
    copy_to_device('NLVLd', NLVL, np.int32)
    copy_to_device('NPSTKd', NPSTK, np.int32)
    copy_to_device('BEERd', BEER, np.int32)
    copy_to_device('SMINd', SMIN, np.int32)
    copy_to_device('SMAXd', SMAX, np.int32)
    copy_to_device('RMINd', RMIN, np.int32)
    copy_to_device('RMAXd', RMAX, np.int32)
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
        copy_to_device('NENVd', env.NENV, np.int32)
        copy_to_device('NXENVMAPd', env.NXENVMAP, np.int32)
        copy_to_device('NYENVMAPd', env.NYENVMAP, np.int32)
    copy_to_device('STHVd', STHV, np.float32)
    copy_to_device('CTHVd', CTHV, np.float32)
    copy_to_device('RTER', RTER, np.float32)
    copy_to_device('NWLPROBA', NWLPROBA, np.int32)
    copy_to_device('NSENSORPROBA', NSENSORPROBA, np.int32)
    copy_to_device('NCELLPROBA', NCELLPROBA, np.int32)
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
        copy_to_device('DIRSXd', vSun.x, np.float64)
        copy_to_device('DIRSYd', vSun.y, np.float64)
        copy_to_device('DIRSZd', vSun.z, np.float64)
        copy_to_device('PXd', PXd, np.float32)
        copy_to_device('PYd', PYd, np.float32)
        copy_to_device('PZd', PZd, np.float32)
        copy_to_device('ZTOAd', ZTOA, np.float32)
        if TC is not None:
            copy_to_device('TCd', TC, np.float32)
            copy_to_device('nbCx', nbCx, np.int32)
            copy_to_device('nbCy', nbCy, np.int32)
        if (  (cusL is not None) and (cusL.dict['LMODE'] == "RF")  ):
            copy_to_device('LMODEd', 1, np.int32)
        if (  (cusL is not None) and (cusL.dict['LMODE'] == "FF")  ):
            copy_to_device('CFXd', cusL.dict['CFX'], np.float32)
            copy_to_device('CFYd', cusL.dict['CFY'], np.float32)
            copy_to_device('CFTXd', cusL.dict['CFTX'], np.float32)
            copy_to_device('CFTYd', cusL.dict['CFTY'], np.float32)
            copy_to_device('ALDEGd', cusL.dict['FOV'], np.float32)
            copy_to_device('TYPEd', cusL.dict['TYPE'], np.int32)
            copy_to_device('LMODEd', 2, np.int32)
        if (  (cusL is not None) and (cusL.dict['LMODE'] == "B" or cusL.dict['LMODE'] == "BR")  ):
            copy_to_device('THDEGd', cusL.dict['THDEG'], np.float32)
            copy_to_device('PHDEGd', cusL.dict['PHDEG'], np.float32)
            copy_to_device('ALDEGd', cusL.dict['ALDEG'], np.float32)
            copy_to_device('TYPEd', cusL.dict['TYPE'], np.int32)
        if (  (cusL is not None) and (cusL.dict['LMODE'] == "B")  ):    
            copy_to_device('LMODEd', 3, np.int32)
        if (  (cusL is not None) and (cusL.dict['LMODE'] == "BR")  ):
            copy_to_device('LMODEd', 4, np.int32)
        if (cusL is None):
            copy_to_device('LMODEd', 0, np.int32)
        
def init_profile(wl, prof, kind):
    '''
    take the profile as a MLUT, and setup the gpu structure
    kind = 'atm' or 'oc' for atmosphere or ocean
    '''

    #NREF = len(prof.axis('z_'+kind))
    # reformat to smartg format
    if 'iopt_'+kind in prof.datasets(): 
        NLAY = len(prof['OD_'+kind].data[0,:])
    else:
        NLAY = len(prof.axis('z_'+kind))
    shp = (len(wl), NLAY)
    prof_gpu = np.zeros(shp, dtype=type_Profile, order='C')

    if kind == "oc":
        if 'iopt_oc' not in prof.datasets():
            prof_gpu['z'][0,:] = prof.axis('z_'+kind)
            #prof_gpu['z'][0,:] = prof.axis('z_'+kind)  * 1e-3 # to Km
            prof_gpu['T'][0,:] = prof['T_'+kind].data[:]
            cell_gpu = np.zeros(1, dtype=type_Cell)
        else: 
            cell_gpu = np.zeros(len(prof['iopt_oc'].data), dtype=type_Cell)
        prof_gpu['n'][0,:] = 1.34
    else:
        if 'iopt_atm' not in prof.datasets():
            prof_gpu['z'][0,:] = prof.axis('z_'+kind)
            prof_gpu['T'][0,:] = prof['T_'+kind].data[:]
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
                imax+=im 
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
            elif d==('T_'+kind) :
                pro.add_dataset(d, first[d].data[:], ['z_'+kind])
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

# for record but no longer the prefered method
'''
def reduce_histories(kernel2, tabHist, wl, sigma, NLOW, NBTHETA=1, alb_in=None, XBLOCK=512, XGRID=512, verbose=False):
   NL    = sigma.shape[1]
   w     = tabHist[:, NL+4:-6]
   #w     = tabHist[:, NL+4:-5]
   ngood = np.sum(w[:,0]!=0)

   S       = np.zeros((ngood,4),dtype=np.float32) 
   cd      = tabHist[:ngood,     :NL  ]
   S[:,:4] = tabHist[:ngood, NL  :NL+4]
   w       = tabHist[:ngood, NL+4:-6  ]
   nrrs    = tabHist[:ngood,      -6  ]
   nref    = tabHist[:ngood,      -5  ]
   nsif    = tabHist[:ngood,      -4  ]
   nvrs    = tabHist[:ngood,      -3  ]
   nenv    = tabHist[:ngood,      -2  ]
   ith     = tabHist[:ngood,      -1  ]

   NT      = XBLOCK*XGRID       # Maximum Number of threads
   NPHOTON = cd.shape[0]        # Number of photons

   NLAYER  = sigma.shape[1]     # Number of vertical layer
   NWVL    = sigma.shape[0]     # Number of wavelength for absorption and output
   NGROUP  = NT//NWVL           # Number of groups of photons
   NTHREAD = NGROUP*NWVL        # Number of threads used
   NBUNCH  = NPHOTON//NGROUP    # Number of photons per group
   NP_REST = NPHOTON%(NGROUP*NBUNCH) # Number of additional photons in the last group

   wls= np.linspace(wl[0], wl[-1], num=NLOW, dtype=np.float32)
   f  =  interp1d(wls,np.linspace(0, NLOW-1, num=NLOW))
   iw =  f(wl)
   iwls_in = np.floor(iw).astype(np.int8)        # index of lower wls value in the wls array, 
   wwls_in = (iw-iwls_in).astype(np.float32)  # floating proportion between iwls and iwls+1
   # special case for NLOW
   ii = np.where(iwls_in==(NLOW-1))
   iwls_in[ii] = NLOW-2
   wwls_in[ii] = 1.

   if verbose : 
       fmt = 'Max Number of Threads : {}\nNumber of Threads : {}\n'
       fmt+= 'Number of groups of photons: {}\nNumber of photons : {}\n'
       fmt+= 'Number of layer: {}\nNumber of wavelength for absorption and output : {}\n'    
       fmt+= 'Number of photons per group : {}\nNumber of additional photons in the last group : {}\n'
       fmt+= 'Number of wavelength for scattering correction : {}\n'
       print(fmt.format(NT, NTHREAD, NGROUP, NPHOTON, NLAYER, NWVL, NBUNCH, NP_REST, NLOW))

   if alb_in is None  : alb_in = np.zeros(2*NWVL, dtype=np.float32)

   sigma_ab_in  = np.zeros((NLAYER, NWVL), order='C', dtype=np.float32)
   sigma_ab_in[:,:]  = sigma.swapaxes(0,1)

   cd_in        = cd.reshape((NPHOTON, NLAYER), order='C').astype(np.float32)
   S_in         = S.reshape((NPHOTON, 4),       order='C').astype(np.float32)
   weight_in    = w.reshape((NPHOTON, NLOW),order='C').astype(np.float32)
   nrrs_in      = nrrs.reshape(NPHOTON,order='C').astype(np.int8)
   nsif_in      = nsif.reshape(NPHOTON,order='C').astype(np.int8)
   nref_in      = nref.reshape(NPHOTON,order='C').astype(np.int8)
   nvrs_in      = nvrs.reshape(NPHOTON,order='C').astype(np.int8)
   nenv_in      = nenv.reshape(NPHOTON,order='C').astype(np.int8)
   ith_in       = ith.reshape( NPHOTON,order='C').astype(np.int8)

   res_out      = gpuzeros((4, NWVL, NBTHETA),   dtype=np.float64)
   res_sca      = gpuzeros((4, NWVL, NBTHETA),   dtype=np.float64)
   res_rrs      = gpuzeros((4, NWVL, NBTHETA),   dtype=np.float64)
   res_sif      = gpuzeros((4, NWVL, NBTHETA),   dtype=np.float64)
   res_vrs      = gpuzeros((4, NWVL, NBTHETA),   dtype=np.float64)

   kernel2(np.int64(NPHOTON), np.int64(NLAYER), np.int64(NWVL), 
                     np.int64(NTHREAD), np.int64(NGROUP), np.int64(NBUNCH), 
                     np.int64(NP_REST), np.int64(NLOW), np.int64(NBTHETA),
                     res_out, res_sca, res_rrs, res_sif, res_vrs, 
                     to_gpu(sigma_ab_in), to_gpu(alb_in), to_gpu(cd_in),
                     to_gpu(S_in), to_gpu(weight_in), to_gpu(nrrs_in), 
                     to_gpu(nref_in), to_gpu(nsif_in), to_gpu(nvrs_in),
                     to_gpu(nenv_in), to_gpu(ith_in), to_gpu(iwls_in), to_gpu(wwls_in), 
                     block=(XBLOCK,1,1),grid=(XGRID,1,1))
   return res_out, res_sca, res_rrs, res_sif, res_vrs
'''


def loop_kernel(NBPHOTONS, faer, foce, NLVL, NATM, NATM_ABS, NOCE, NOCE_ABS, MAX_HIST, NLOW,
                NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                NLAM, NSENSOR, double, kern, kern2, p, X0, le, tab_sensor, envmap, spectrum,
                prof_atm, prof_oc, cell_atm, cell_oc, wl_proba_icdf, sensor_proba_icdf, cell_proba_icdf,
                stdev, stdev_lim, rng, alis, myObjects0, TC, nbCx, nbCy, myGObj0, myRObj0, mySPECTObj0, hist=False):
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
        - kern : kernel launching the radiative transfer
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
    nThreadsActive = gpuzeros(1, dtype=np.uint32)
    Counter = gpuzeros(1, dtype=np.uint64)

    
    if double : FDTYPE=np.float64
    else : FDTYPE=np.float32

    # If a receiver object is used then : initialize matrix and vectors for gains and losses
    if TC is not None:
        nbPhCat = gpuzeros(8, dtype=np.uint64) # vector to fill the number of photons for  each categories
        wPhCat = gpuzeros((8, NLAM), dtype=FDTYPE)  # vector to fill the weight of photons for each categories
        wPhCatTot = gpuzeros((8, NLAM), dtype=FDTYPE)
        wPhCat2 = gpuzeros((8, NLAM), dtype=FDTYPE)  # sum of squared photons weight for each cats
        wPhCat2Tot = gpuzeros((8, NLAM), dtype=FDTYPE)
        tabObjInfo = gpuzeros((9, nbCx, nbCy), dtype=FDTYPE)
        wPhLoss = gpuzeros(7, dtype=FDTYPE)
        wPhLoss2 = gpuzeros(7, dtype=FDTYPE)
        tabMatRecep = np.zeros((9, nbCx, nbCy), dtype=np.float64)
        
        # Matrix where lines : l0 = SumCats, l1=cat1, l2=cat2, ... l8=cat8
        # And columns : c0=nbPhotons , c1=weight, c2=weight2, c3=flux(in watt), c4=errAbs, c5=err%
        matCats = np.zeros((9, 6), dtype=np.float64)
        
        # Matrix where: M[0,0]=W_I, M[1,0]=W_rhoM, M[2,0]=W_rhoP, M[3,0]=W_BM, M[4,0]=W_BP, M[5,0]=W_SM, M[6,0]=W_SP
        # and : M[0,1]=W_I², M[1,1]=W_rhoM², M[2,1]=W_rhoP², M[3,1]=W_BM², M[4,1]=W_BP², M[5,1]=W_SM², M[6,1]=W_SP²
        matLoss = np.zeros((7, 2), dtype=np.float64)
    else:
        nbPhCat = gpuzeros((1, 1), dtype=np.uint64)
        wPhCat = gpuzeros((1, 1), dtype=FDTYPE)
        wPhCat2 = gpuzeros((1, 1), dtype=FDTYPE)
        wPhCatTot = gpuzeros((1, 1), dtype=FDTYPE)
        wPhCat2Tot = gpuzeros((1, 1), dtype=FDTYPE)
        wPhLoss = gpuzeros(1, dtype=FDTYPE)
        wPhLoss2 = gpuzeros(1, dtype=FDTYPE)
        tabObjInfo = gpuzeros((1, 1, 1), dtype=FDTYPE)
        
    # Initialize the array for error counting
    NERROR = 32
    errorcount = gpuzeros(NERROR, dtype='uint64')

    if (NATM >0):
        tabTransDir = gpuzeros((NSENSOR,NLAM), dtype=np.float64)
    else :
        tabTransDir = gpuzeros((1,1), dtype=np.float64)
    
    if ((NATM+NOCE >0) and (NATM_ABS+NOCE_ABS <500) and alis) : 
        tabDistTot = gpuzeros((NLVL,NATM_ABS+NOCE_ABS,NSENSOR,NBTHETA,NBPHI,2), dtype=np.float64)
    else : 
        tabDistTot = gpuzeros((1), dtype=np.float64)

    '''
    if hist : 
        tabHistTot = gpuzeros((2,MAX_HIST,(NATM_ABS+NOCE_ABS+NPSTK+NLOW+6),NSENSOR,NBTHETA,NBPHI), dtype=np.float32)
        dz    = abs(np.diff(prof_atm.get()['z'][0,:]))
        sigma = np.diff(prof_atm.get()['OD_abs'][:,:])/dz
        alb_in= np.concatenate([spectrum.get()['alb_surface'][:], spectrum.get()['alb_env'][:]])
        wl = spectrum.get()['lambda'][:]
    else :
        tabHistTot = gpuzeros((1), dtype=np.float32)
    '''

    # Initialize of the parameters
    tabPhotonsTot = gpuzeros((NLVL,NPSTK,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.float64)
    tabPhotonsTotNoAer = gpuzeros((NLVL,NPSTK,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.float64)
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
    NPhotonsOutNoAer = gpuzeros((NLVL,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.uint64)
    NPhotonsOutTot = gpuzeros((NLVL,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.uint64)
    NPhotonsOutTotNoAer = gpuzeros((NLVL,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.uint64)

    if double:
        tabPhotons = gpuzeros((NLVL,NPSTK,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.float64)
        tabPhotonsNoAer = gpuzeros((NLVL,NPSTK,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.float64)
        if ((NATM+NOCE >0) and (NATM_ABS+NOCE_ABS <500) and alis) : 
            tabDist = gpuzeros((NLVL,NATM_ABS+NOCE_ABS,NSENSOR,NBTHETA,NBPHI,2), dtype=np.float64)
        else :
            tabDist = gpuzeros((1), dtype=np.float64)
    else:
        tabPhotons = gpuzeros((NLVL,NPSTK,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.float32)
        tabPhotonsNoAer = gpuzeros((NLVL,NPSTK,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.float32)
        if ((NATM+NOCE >0) and (NATM_ABS+NOCE_ABS <500) and alis) : 
            tabDist = gpuzeros((NLVL,NATM_ABS+NOCE_ABS,NSENSOR,NBTHETA,NBPHI,2), dtype=np.float32)
        else : 
            tabDist = gpuzeros((1), dtype=np.float32)

    if hist : 
        #tabHist = gpuzeros((MAX_HIST,(NATM_ABS+NOCE_ABS+NPSTK+NLOW+6),1,NBTHETA,1), dtype=np.float32)
        #tabHist = gpuzeros((2,MAX_HIST,(NATM_ABS+NOCE_ABS+NPSTK+NLOW+6),1,1,1), dtype=np.float32)
        tabHistTot = gpuzeros((2,MAX_HIST,(NATM_ABS+NOCE_ABS+NPSTK+NLOW+6),NSENSOR,NBTHETA,NBPHI), dtype=np.float32)

    else : 
        tabHistTot = gpuzeros((1), dtype=np.float32)

    # local estimates angles
    if le != None:
        tabthv = to_gpu(le['th'].astype('float32'))
        tabphi = to_gpu(le['phi'].astype('float32'))
        if 'count_level' in le: tablevel = to_gpu(le['count_level'].astype('int32'))
        else: tablevel = to_gpu(np.full((NBTHETA), -2).astype('int32'))
    else:
        tabthv = gpuzeros(1, dtype='float32')
        tabphi = gpuzeros(1, dtype='float32')
        tablevel = to_gpu(np.array([-2]).astype('int32'))

    secs_cuda_clock = 0.
    alis_norm = NLAM if NLOW!=0 else 1
    while((np.sum(NPhotonsInTot.get())/alis_norm) < NBPHOTONS):
        tabPhotons.fill(0.)
        tabPhotonsNoAer.fill(0.)
        NPhotonsOut.fill(0)
        NPhotonsOutNoAer.fill(0)
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
        kern(envmap, spectrum, X0, faer, foce,
             errorcount, nThreadsActive, tabPhotons, tabDist, tabHistTot, MAX_HIST, tabPhotonsNoAer, tabTransDir,
             Counter, NPhotonsIn, NPhotonsOut, NPhotonsOutNoAer, tabthv, tabphi, tablevel, tab_sensor,
             prof_atm, prof_oc, cell_atm, cell_oc, wl_proba_icdf, sensor_proba_icdf, cell_proba_icdf, 
             rng.state, tabObjInfo,
             myObjects0, myGObj0, myRObj0, mySPECTObj0, nbPhCat, wPhCat, wPhCat2,
             wPhLoss, wPhLoss2, block=(XBLOCK, 1, 1), grid=(XGRID, 1, 1))

        end_cuda_clock.record()
        end_cuda_clock.synchronize()
        secs_cuda_clock = secs_cuda_clock + start_cuda_clock.time_till(end_cuda_clock)

        cuda.Context.synchronize()
        np.set_printoptions(precision=5, linewidth=150)

        if TC is not None:
            # Matrix with the photon weights distribution on the receiver surface
            tabMatRecep += tabObjInfo[:, :, :].get()
            # Fill the matrix malLoss with the photon weights for losses estimates
            matLoss[:,0] += wPhLoss[:].get()
            matLoss[:,1] += wPhLoss2[:].get()
            # Begin to fill the matrix matCats
            matCats[0,1] += np.sum(wPhCat[:, :].get())
            matCats[0,2] += np.sum(wPhCat2[:, :].get())
            wPhCatTot += wPhCat
            wPhCat2Tot += wPhCat2
            for i in range (0, 8):
                # Count the photon weights for each category
                matCats[i+1,1] += np.sum(wPhCat[i, :].get())    # sum of wi
                matCats[i+1,2] += np.sum(wPhCat2[i, :].get())   # sum of wi
        
        L = NPhotonsIn   # number of photons launched by last kernel
        NPhotonsInTot += L

        NPhotonsOutTot += NPhotonsOut
        S = tabPhotons   # sum of weights for the last kernel

        NPhotonsOutTotNoAer += NPhotonsOutNoAer
        SRayleigh = tabPhotonsNoAer   # sum of weights for the last kernel

        if(not hist) : 
            tabPhotonsTot += S
            tabPhotonsTotNoAer += SRayleigh
        
        T = tabDist
        tabDistTot += T
        
        ## for record but no longer the prefered method
        '''
        if hist :
            tabHistTot = tabHist
            res,res_sca,res_rrs,res_sif,res_vrs = reduce_histories(kern2, np.squeeze(tabHist.get()), wl, sigma, NLOW,
                                NBTHETA=NBTHETA, alb_in=alb_in)
            if NBTHETA>1:
                tabPhotonsTot[0,:,:,:,:,:] += res[:,None,:,:,None]
                tabPhotonsTot[1,:,:,:,:,:] += res_sca[:,None,:,:,None]
                tabPhotonsTot[2,:,:,:,:,:] += res_rrs[:,None,:,:,None]
                tabPhotonsTot[3,:,:,:,:,:] += res_sif[:,None,:,:,None]
                tabPhotonsTot[4,:,:,:,:,:] += res_vrs[:,None,:,:,None]
            else :
                tabPhotonsTot[0,:,:,:,:,:] += res[:,None,:,None]
                tabPhotonsTot[1,:,:,:,:,:] += res_sca[:,None,:,None]
                tabPhotonsTot[2,:,:,:,:,:] += res_rrs[:,None,:,None]
                tabPhotonsTot[3,:,:,:,:,:] += res_sif[:,None,:,None]
                tabPhotonsTot[4,:,:,:,:,:] += res_vrs[:,None,:,None]
        '''
        
        N_simu += 1

        sphot = np.sum(NPhotonsInTot.get())/alis_norm
        if stdev:
            (NSENSOR,NLAM) = NPhotonsIn.shape
            L = L.reshape((1,1,NSENSOR,NLAM,1,1))   # broadcast to tabPhotonsTot
            #warn('stdev is activated: it is known to slow down the code considerably.')
            SoverL = S.get()/L.get()
            sum_x += SoverL
            sum_x2 += (SoverL)**2

            if stdev_lim is not None:
                sigma_bis = np.sqrt(sum_x2/N_simu - (sum_x/N_simu)**2)
                sigma_bis /= np.sqrt(N_simu)
                sigma_bis[np.isnan(sigma_bis)] = 0

                abs_min = stdev_lim.dict['err_abs_min']
                rel_min = stdev_lim.dict['err_rel_min']
                min_loop = stdev_lim.dict['nb_loop_min']
                stk_stdev = stdev_lim.dict['stk']
                level_stdev = stdev_lim.dict['level']
                format_std = stdev_lim.dict['format']

                avg = sum_x/N_simu
                err_rel = (sigma_bis / avg)*100
                err_rel[np.isnan(err_rel)] = 0
                max_rerr = np.max(err_rel[level_stdev,stk_stdev,:,:,:,:])
                max_aerr = np.max(sigma_bis[level_stdev,stk_stdev,:,:,:,:])

                if (stdev_lim.dict['verbose']):
                    print(f"max rel_err = {max_rerr:{format_std}}; max abs_err = {max_aerr:{format_std}}")

                if ( (N_simu >= min_loop and max_aerr <= abs_min) or (N_simu >= min_loop and max_rerr <= rel_min) ):
                    # update of the progression Bar
                    p.update(sphot, f"Launched {sphot:.3g} photons; err[abs] = {max_aerr:{format_std}}; err[rel] = {max_rerr:{format_std}};")
                    break

        if TC is not None and stdev_lim is not None:
            NBPHOTONS_tmp = np.sum(NPhotonsInTot.get())
            nBis = NBPHOTONS_tmp/(NBPHOTONS_tmp-1)
            sum2Z = (matCats[0,1]*matCats[0,1])/NBPHOTONS_tmp
            sumZ2 = matCats[0,2]
            if (le is None): num = (nBis * (sumZ2 - sum2Z))**0.5
            else: num = (nBis * abs(sumZ2 - sum2Z))**0.5
            den = matCats[0,1]
            err_p_tmp = (num/den)*100
            min_loop = stdev_lim.dict['nb_loop_min']
            rel_min = stdev_lim.dict['err_rel_min']

            if (stdev_lim.dict['verbose']):
                print(f"relative_err = {err_p_tmp:{format_std}}")

            # update of the progression Bar
            p.update(sphot, f"Launched {sphot:.3g} photons; err[rel] = {err_p_tmp:{format_std}};")

            if (N_simu >= min_loop and err_p_tmp <= rel_min):
                NBPHOTONS = NBPHOTONS_tmp
                break  
        elif (stdev and stdev_lim is not None):
            # update of the progression Bar
            p.update(sphot, f"Launched {sphot:.3g} photons; err[abs] = {max_aerr:{format_std}}; err[rel] = {max_rerr:{format_std}};")
        else:
            # update of the progression Bar
            p.update(sphot, 'Launched {:.3g} photons'.format(sphot))
            
    # END WHILE LOOP
    secs_cuda_clock = secs_cuda_clock*1e-3

    if TC is not None: # If there is a receiver obj
        nBis = NBPHOTONS/(NBPHOTONS-1)
        # Count the total number of photons received and also for each cats
        matCats[0,0] = np.sum(nbPhCat[:].get())
        for i in range (0, 8): # Here for each cats
            matCats[i+1,0] = nbPhCat[i].get()
        
        # Relative and absolute error for sum of cats and also for each cats
        for i in range (0, 9):
            if (matCats[i,0] != 0 and matCats[i,1] != 0):
                # Monte carlo err computation see the book of Dunn and Shultis
                sum2Z = (matCats[i,1]*matCats[i,1])/NBPHOTONS
                sumZ2 = matCats[i,2]
                if (le is None):
                    matCats[i,4] = (nBis * (sumZ2 - sum2Z))**0.5 # errAbs not normalized
                else:
                    matCats[i,4] = (nBis * abs(sumZ2 - sum2Z))**0.5
                matCats[i,5] = (matCats[i,4]/matCats[i,1])*100 # err%
    else:
        tabMatRecep = None; matCats = None; matLoss=None

    if stdev:
        # finalize the calculation of the standard deviation
        sigma = np.sqrt(sum_x2/N_simu - (sum_x/N_simu)**2)

        # extrapolate in 1/sqrt(N_simu)
        sigma /= np.sqrt(N_simu)
    else:
        sigma = None


    return NPhotonsInTot.get(), tabPhotonsTot.get(), tabPhotonsTotNoAer.get(), tabDistTot.get(), tabHistTot.get(), tabTransDir.get(), errorcount, \
        NPhotonsOutTot.get(), NPhotonsOutTotNoAer.get(), sigma, N_simu, secs_cuda_clock, tabMatRecep, matCats, matLoss, wPhCatTot.get(), wPhCat2Tot.get()


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
            # A multiply by 1000 has been removed to avoid OverflowError due to uint32 limit
            SEED = np.uint32((datetime.now(tz=timezone.utc)
                              - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds())

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
        s = gpuzeros(1, dtype=np.uint32)
        self.mod.get_function('get_state_size')(s, block=(1, 1, 1), grid=(1, 1, 1))
        self.STATE_SIZE = int(np.squeeze(s.get()))  # size in bytes

    def setup(self, SEED, XBLOCK, XGRID):
        if SEED == -1:
            # SEED is based on clock
            SEED = np.uint32((datetime.now(tz=timezone.utc)
                             - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds())

        cuda.memcpy_htod(self.mod.get_global('XBLOCKd')[0], np.array([XBLOCK], dtype=np.int32))
        cuda.memcpy_htod(self.mod.get_global('XGRIDd')[0], np.array([XGRID], dtype=np.int32))
        cuda.memcpy_htod(self.mod.get_global('SEEDd')[0], np.array([XGRID], dtype=np.int32))

        # setup RNG
        self.state = gpuzeros(self.STATE_SIZE*XBLOCK*XGRID, dtype='uint8')
        setup = self.mod.get_function('setup')
        setup(self.state, block=(XBLOCK,1,1), grid=(XGRID, 1, 1))

        return SEED

def initObj(LGOBJ, vSun, wl, CUSL=None):
    '''
    Definition of the function LOBJ

    ===ARGS:
    LGOBJ : List of object groups
    CUSL  : Custom lanching mode class (i.g. cusForward())
    wl    : A scalar or list/array of wavelengths (in nm) or a list of REPTRAN or KDIS IBANDS
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
    LROBJSPECT: GPU array of objects of type 'type_Spectrum_obj'
    '''

    ind = 0; LOBJ = []; nGObj = len(LGOBJ); nRObj = 0; INDROBJ = []
    LGOBJGPU = np.zeros(nGObj, dtype=type_GObj, order='C')

    # Creation of a list with only Entity objects and creation
    # of a GPU table with object groups parameters

    for i in range (0, nGObj):
        LGOBJGPU['index'][i] = ind 
        LGOBJGPU['bPminx'][i] = LGOBJ[i].bboxGPmin.x; LGOBJGPU['bPminy'][i] = LGOBJ[i].bboxGPmin.y
        LGOBJGPU['bPminz'][i] = LGOBJ[i].bboxGPmin.z; LGOBJGPU['bPmaxx'][i] = LGOBJ[i].bboxGPmax.x
        LGOBJGPU['bPmaxy'][i] = LGOBJ[i].bboxGPmax.y; LGOBJGPU['bPmaxz'][i] = LGOBJ[i].bboxGPmax.z
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
        TC = None; nbCx = int(0); nbCy = int(0)
    
    # To consider the spectral variability of objs reflectivity
    nObjTotal = LOBJGPU.size
    if not isinstance(wl, BandSet): wl = BandSet(wl)
    NLAM = wl.size
    LOBJSPECT = np.zeros((nObjTotal*NLAM), dtype=type_Spectrum_obj, order='C')

    # Initialization before the coming loop
    pp1 = 0.; pp2 = 0.; pp3 = 0.; pp4 = 0.
    nb_H = 0; zAlt_H = 0.; totS_H = 0.; ncos=0.
    if (CUSL != None and CUSL.dict['LMODE'] == "RF"): surfLPH = 0
    else: surfLPH = None

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
            normalBase = Vector(0, 0, 1)

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
                TpT0 = TpRY0*TpRZ0*TpRX0
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
        # LOBJGPU['reflectAV'][i] = LOBJ[i].materialAV.reflectivity
        LOBJGPU['reflectAV'][i] = 0
        if (np.array(LOBJ[i].materialAV.reflectivity).size == 1):
            LOBJSPECT['reflectAV'][(i*NLAM):((i*NLAM)+NLAM)] = np.full((NLAM), LOBJ[i].materialAV.reflectivity)
        elif (LOBJ[i].materialAV.reflectivity.size != NLAM):
            raise NameError('The number of reflectivities must be equal to the number of wavelengths!')
        else:
            LOBJSPECT['reflectAV'][(i*NLAM):((i*NLAM)+NLAM)] = LOBJ[i].materialAV.reflectivity[:]
        # Particularity of each material
        if isinstance(LOBJ[i].materialAV, LambMirror):
            LOBJGPU['materialAV'][i] = 1
            LOBJGPU['roughAV'][i] = 0.
        elif isinstance(LOBJ[i].materialAV, Matte):
            LOBJGPU['materialAV'][i] = 2
            LOBJGPU['roughAV'][i] = LOBJ[i].materialAV.roughness
        elif isinstance(LOBJ[i].materialAV, Mirror):
            LOBJGPU['materialAV'][i] = 3
            LOBJGPU['shdAV'][i] = int(LOBJ[i].materialAV.shadow)
            LOBJGPU['nindAV'][i] = LOBJ[i].materialAV.nind
            LOBJGPU['distAV'][i] = LOBJ[i].materialAV.distribution
            LOBJGPU['roughAV'][i] = LOBJ[i].materialAV.roughness
        else:
            raise NameError('Unknown material AV')

        # 1) Back part of the object (AR for the french word 'ARriere')
        # Initialization
        LOBJGPU['materialAR'][i] = 0; LOBJGPU['shdAR'][i] = 0
        LOBJGPU['nindAR'][i] = 1; LOBJGPU['distAR'][i] = 0
        # Commun to all materials
        #LOBJGPU['reflectAR'][i] = LOBJ[i].materialAR.reflectivity
        LOBJGPU['reflectAR'][i] = 0
        if (np.array(LOBJ[i].materialAR.reflectivity).size == 1):
            LOBJSPECT['reflectAR'][(i*NLAM):((i*NLAM)+NLAM)] = np.full((NLAM), LOBJ[i].materialAR.reflectivity)
        elif (LOBJ[i].materialAR.reflectivity.size != NLAM):
            raise NameError('The number of reflectivities must be equal to the number of wavelengths!')
        else:
            LOBJSPECT['reflectAR'][(i*NLAM):((i*NLAM)+NLAM)] = LOBJ[i].materialAR.reflectivity[:]
        # Particularity of each material
        if isinstance(LOBJ[i].materialAR, LambMirror):
            LOBJGPU['materialAR'][i] = 1
            LOBJGPU['roughAR'][i] = 0.
        elif isinstance(LOBJ[i].materialAR, Matte):
            LOBJGPU['materialAR'][i] = 2
            LOBJGPU['roughAR'][i] = LOBJ[i].materialAR.roughness
        elif isinstance(LOBJ[i].materialAR, Mirror):
            LOBJGPU['materialAR'][i] = 3
            LOBJGPU['shdAR'][i] = int(LOBJ[i].materialAR.shadow)
            LOBJGPU['nindAR'][i] = LOBJ[i].materialAR.nind
            LOBJGPU['distAR'][i] = LOBJ[i].materialAR.distribution
            LOBJGPU['roughAR'][i] = LOBJ[i].materialAR.roughness
        else:
            raise NameError('Unknown material AR')
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
                totS_H += abs(LOBJ[i].geo.p1.x)*abs(LOBJ[i].geo.p1.y)*4
                ncos += Dot(normalBase, Vector(-vSun.x, -vSun.y, -vSun.z))

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
        elif (LOBJ[i].name == "environment"):
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
    LOBJSPECT = to_gpu(LOBJSPECT)
    # update the value of ncos
    if nb_H > 0:
        n_cos = ncos/nb_H
    else:
        n_cos = 1

    return nGObj, nObj, nRObj, surfLPH, nb_H, zAlt_H, totS_H, TC, nbCx, nbCy, LOBJGPU, \
        LGOBJGPU, LROBJGPU, LOBJSPECT, n_cos

def normalizeRecIrr(cMatVisuRecep, matCats, nbCx, nbCy, NBPHOTONS, surfLPH, TC, cusL, SUN_DISC, LE):
    '''
    Description of the function normalizeRecIrr

    This function enables the normalization of the signal collected
    by a given receiver, to get the collected Power in Watt a multiplication by 
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
    LE            : boolean indicating if we are in LE mode

    === RETURN:
    cMatVisuRecep : With normalized values
    matCats       : With normalized values
    normC         : Constant enabling normalization
    '''
    S_rec = TC*TC*nbCx*nbCy # receiver surface in km²
    S_rec_m = S_rec * 1e6   # receiver surface in m²

    # Normalize intensities such that only a mult by E_TOA is still needed to obtain power unit
    if (cusL is None):
        normC = 1.
        # normC = 1./NBPHOTONS
        # # Weights -> propor to w/m², mult by S_rec_m is needed to get something propor to watt unit
        # normC *= S_rec_m
        # cMatVisuRecep[:][:][:] = cMatVisuRecep[:][:][:]*normC
        # for i in range (0, 9):
        #     matCats[i,3] = matCats[i,1]*normC # intensity
        #     matCats[i,4] *= normC # Absolute err
    elif (cusL.dict['LMODE'] == "FF" or cusL.dict['LMODE'] == "RF"):
        # Here results are already propor to watt unit
        normC = (surfLPH*1e6)/NBPHOTONS  # Here multiply by 1e6 to convert km² to m²
        normFF = 1.
        #lambertian sampling normalization
        if (cusL.dict['LMODE'] == "FF" and cusL.dict['TYPE'] == 1 and cusL.dict['FOV'] > 1e-6):
            normFF = ( 1-np.cos(np.radians(2*cusL.dict['FOV'])) ) / (4*( 1-np.cos(np.radians(cusL.dict['FOV'])) ))
        #isotropic sampling normalization
        elif (cusL.dict['LMODE'] == "FF" and cusL.dict['TYPE'] == 2 and cusL.dict['FOV'] > 1e-6):
            normFF = 1.
        normC *= normFF
        for i in range (0, 9):
            cMatVisuRecep[i][:][:] = cMatVisuRecep[i][:][:]*normC
            matCats[i,3] = matCats[i,1]*normC
            matCats[i,4] *= normC
    elif (cusL.dict['LMODE'] == "B" or cusL.dict['LMODE'] == "BR"):
        normBR = 2
        #lambertian sampling normalization
        if (cusL.dict['TYPE'] == 1): normBR = (1-np.cos(np.radians(2*cusL.dict['ALDEG'])))/2.
        #isotropic sampling normalization
        elif (cusL.dict['TYPE'] == 2): normBR = 2*(1-np.cos(np.radians(cusL.dict['ALDEG'])))

        if not LE: normC = normBR/(NBPHOTONS*2*(1-np.cos(np.radians(SUN_DISC))))
        else: normC = normBR/NBPHOTONS

        # Weights -> propor to w/m², mult by S_rec_m is needed to get something propor to watt unit
        normC *= S_rec_m
        cMatVisuRecep[:][:][:] = cMatVisuRecep[:][:][:]*normC
        for i in range (0, 9):
            matCats[i,3] = matCats[i,1]*normC
            matCats[i,4] *= normC
    else:
        raise NameError('Unknown launching mode!')

    return cMatVisuRecep, matCats, normC


def findExtinction(IP, FP, prof_atm, W_IND = int(0)):
    '''
    Description of the function findAtmLoss

    This function enables the calculation of the extinction
    between an initial point 'IP' to a final point 'FP'
    === ONLY VALID WITH A 1D PP ATM ===

    ==== ARGS:
    IP       : Initial position (Point class)
    FP       : Final position (Point class)
    prof_atm : Atmosphere profil class
    W_IND    : wavelength index (default = 0)

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
    ilayer2 = lay

    # Case with only 1 layer: n = 1
    if (FP.z >= prof_atm.axis('z_atm')[ilayer2] and FP.z < prof_atm.axis('z_atm')[ilayer2-1]):
        # delta_i is: Delta(tau)1 = |tau(i-1) - tau(i)|
        delta_i = abs(prof_atm['OD_atm'].data[W_IND, ilayer2-1] - prof_atm['OD_atm'].data[W_IND, ilayer2])
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
            newP = oldP + (Vec*timeT)
            delta_i = abs(prof_atm['OD_atm'].data[W_IND, ilayer]-prof_atm['OD_atm'].data[W_IND, ilayer-1])
            tauHit += ((newP - oldP).Length()/abs(prof_atm.axis('z_atm')[ilayer-1]-prof_atm.axis('z_atm')[ilayer]))*delta_i
        
            if(higher): # the photon come from higher layer
                ilayer+= int(1)
            else: # the photon come from lower layer
                ilayer-= int(1)
            oldP = newP # Update the position of the photon
        
        # Calculate and add the last tau distance when ilayer is equal to ilayer2
        delta_i = abs(prof_atm['OD_atm'].data[W_IND, ilayer2]-prof_atm['OD_atm'].data[W_IND, ilayer2-1])
        tauHit += ((FP - oldP).Length()/abs(prof_atm.axis('z_atm')[ilayer2-1]-prof_atm.axis('z_atm')[ilayer2]))*delta_i

    n_ext = np.exp(-abs(tauHit))

    return n_ext

    
def Get_Sensor(VZA_lev, LEVEL=0., VAA=0., RTER=6370., H=120., FOV=0., TYPE=0., PP=True, verbose=False):
    '''
    Return the Sensor object for atmosphere height of H for backward simulations where
    VZA_lev is the View Zenith Angle defined at the level: origin (0, 0, RTER+LEVEL) for Spherical
    Shell (SS), (0, 0, LEVEL) for Plane Parallel (PP)

    Input:
        VZA_lev: View Zenith Angle defined at altitude LEVEL of Atmosphere
        
    Keywords:
        LEV : Altitude (km) where the VZA is defined, default 0. (ground)
        VAA : View Azimuth Angle
        FOV : Field of View (de), default 0.
        TYPE: Type od sensor, default 0 (radiance), 1 (planar irradiance), 2 (spherical irradiance)
        RTER: Earth radius (km)
        H   : Altitude of the Atmosphere (km)
        SS  : Plane Parallel (PP default or SS)
    '''
    nothing = Transform() # i.e. no rotation and no translation
    radius = (H + RTER)
    large_dist = float("inf") # large distance(km)
    origin = Point(0., 0., LEVEL) if PP else Point(0., 0., RTER+LEVEL)
    # Boundaries
    if PP: Boundary = BBox(Point(-large_dist, -large_dist, 0.), Point(large_dist, large_dist, H)) # Rectangle for atmosphere for PP
    else : Boundary = Sphere(nothing, nothing, radius, -radius, radius, 360) # Create the Earth + atmosphere sphere for SS
    # Compute the direction vector object from Zenith and Azimuth angles
    dir = convertAnglestoV(THETA=VZA_lev, TYPE='Sensor', PHI=180+VAA)
    # Make a ray from origin in direction dir
    ray = Ray(o=origin, d=dir)
    # Compute the intersection with the Boundary
    if PP: (_, t1, hit) = Boundary.IntersectP(ray)
    else : hit = Boundary.Intersect(ray) 
    if not hit: raise NameError("The intersection test failed!! Check input paramaters.")
    # Computations of sensor position
    if PP: pos = origin + dir*t1
    else : pos = origin + dir*Boundary.thit
    if verbose : print("VZA =", VZA_lev, "--> pos =", pos)

    return Sensor(POSX=pos.x, POSY=pos.y, POSZ=pos.z, THDEG=180.-VZA_lev, PHDEG=VAA, LOC='ATMOS', FOV=FOV, TYPE=TYPE)


