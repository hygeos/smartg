#!/usr/bin/env python
# encoding: utf-8


'''
SMART-G
Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU
'''



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
from pycuda.compiler import SourceModule
from pycuda.driver import module_from_buffer
# bellow necessary for object incorporation
import smartg.geometry
from smartg.geometry import Vector, Point, Normal, Ray, BBox, CoordinateSystem, Normalize
import smartg.transform
from smartg.transform import Transform, Aff
import smartg.visualizegeo
from smartg.visualizegeo import Mirror, Plane, Spheric, Transformation, \
    Entity, Analyse_create_entity, LambMirror, Matte


# set up directories
dir_root = dirname(dirname(realpath(__file__)))
dir_src = join(dir_root, 'smartg/src/')
dir_bin = join(dir_root, 'bin/')
src_device = join(dir_src, 'device.cu')
binnames = { # keys are (PP, ALIS)
        (True , False): join(dir_bin, 'pp.cubin'),
        (True , True ): join(dir_bin, 'pp.alis.cubin'),
        (False, False): join(dir_bin, 'sp.cubin'),
        (False, True ): join(dir_bin, 'sp.alis.cubin'),
        }

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

type_Sensor = [
    ('POSX',   'float32'),    # // X position of the sensor
    ('POSY',   'float32'),    # // Y position of the sensor
    ('POSZ',   'float32'),    # // Z position of the sensor (fromp Earth's center in spherical, from the ground in PP)
    ('THDEG',  'float32'),    # // zenith angle of viewing direction (Zenith> 90 for downward looking, <90 for upward, default Zenith)
    ('PHDEG',  'float32'),    # // azimut angle of viewing direction
    ('LOC',    'int32'),      # // localization (ATMOS=1, ...), see constant definitions in communs.h
    ('FOV',    'float32'),    # // sensor FOV (degree) 
    ('TYPE',   'int32'),      # // sensor type: Radiance (0), Planar flux (1), Spherical Flux (2), default 0
    ]

type_IObjets = [
    ('geo', 'int32'),         # 1 = sphere, 2 = plane, ...
    ('materialAV', 'int32'),  # 1 = LambMirror, 2 = Matte,
    ('materialAR', 'int32'),  # 3 = Mirror, ... (AV = avant, AR = Arriere)
    ('type', 'int32'),        # 1 = reflector, 2 = receiver
    ('reflecAV', 'float32'),  # reflectivity of materialAV
    ('reflecAR', 'float32'),  # reflectivity of materialAR
    
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

class Sensor(object):
    '''
    Definition of the sensor

    POS: Position (X,Y,Z) in cartesian coordinates, default origin
    TH,PH: Direction (theta, phi) of zenith and azimuth angles of viewing direction
            (Zenith> 90 for downward looking, <90 for upward, default Zenith)
    LOC: Localization (default SURF0P)
    FOV: Field of View (deg, default 0.)
    TYPE: Radiance (0), Planar flux (1), Spherical Flux (2), default 0
    '''
    def __init__(self, POSX=0., POSY=0., POSZ=0., THDEG=0., PHDEG=180., LOC='SURF0P', FOV=0., TYPE=0):
        self.dict = {
                'POSX':  POSX,
                'POSY':  POSY,
                'POSZ':  POSZ,
                'THDEG': THDEG,
                'PHDEG': PHDEG,
                'LOC'  : LOC_CODE.index(LOC),
                'FOV':   FOV,
                'TYPE':  TYPE
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
    LMODE (Launching mode) : RF = Restricted Forward OR FF = Full Forward
                             RF --> Launch the photons such that the direct beams
                                    fill only reflector objects
                             FF --> Launch the photons in a rectangle from TOA
                                    whrere the beams at the center, with the solar
                                    dir, targets by default the origin point (0,0,0)
    '''
    def __init__(self, CFX=0., CFY=0., CFTX = 0., CFTY = 0., LMODE = "RF"):
        self.dict = {
            'CFX':   CFX,
            'CFY':   CFY,
            'CFTX':   CFTX,
            'CFTY':   CFTY,
            'LMODE': LMODE
        }

    def __str__(self):
        return 'CusForward=-CFX{CFX}-CFY{CFY}'.format(**self.dict)


class Smartg(object):

    def __init__(self, pp=True, debug=False,
                 debug_photon=False,
                 double=False, alis=False, back=False, bias=True, alt_pp=False, obj3D = False, rng='PHILOX'):
        '''
        Initialization of the Smartg object

        Performs the compilation and loading of the kernel.
        This class is :esigned so split compilation and kernel loading from the
        code execution: in case of successive smartg executions, the kernel
        loading time is not repeated.

        Arguments:
            - pp:
                True: use plane parallel geometry (default)
                False: use spherical shell geometry

            Compilation flags, not available if the kernel is provided as a binary:

            - debug: set to True to activate debug mode (optional stdout if problems are detected)

            - debug_photon: activate the display of photon path for the thread 0

            - double : accumulate photons table in double precision, default single

            - alis : boolean, if present implement the ALIS method (Emde et al. 2010) for treating gaseous absorption and perturbed profile

            - back : boolean, if True, run in backward mode, default forward mode
            
            - bias : boolean, if True, use the bias sampling scheme, default True

            - obj3D : Set to True to enable simulation with 3D objects

            - alt_pp: boolean, if True new PP progation scheme is used
            
            - rng: choice of pseudo-random number generator:
                   * PHILOX
                   * CURAND_PHILOX
        '''
        import pycuda.autoinit

        self.pp = pp
        self.double = double
        self.alis = alis
        self.rng = init_rng(rng)
        self.back= back
        self.obj3D= obj3D

        #
        # compilation option
        #
        options = []
        # options = ['-g', '-G']
        if not pp:
            # spherical shell calculation
            options.append('-DSPHERIQUE')
        if alt_pp:
            # new Plane Parallel propagation scheme
            options.append('-DALT_PP')
        if debug:
            # additional tests for debugging
            options.append('-DDEBUG')
        if debug_photon:
            options.append('-DDEBUG_PHOTON')
        if double:
            # counting in double precision
            # ! slows down processing
            options.append('-DDOUBLE')
        if alis:
            options.append('-DALIS')
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
        else:
            binname = binnames[(pp, alis)]
            if exists(binname):
                # load existing binary
                print('Loading binary', binname)
                self.mod = module_from_buffer(open(binname, 'rb').read())

            else:
                raise IOError('Could not find {} or {}.'.format(src_device, binname))

        # load the kernel
        self.kernel = self.mod.get_function('launchKernel')
        self.kernel2 = self.mod.get_function('launchKernel2')

        #
        # common attributes
        #
        self.common_attrs = OrderedDict()
        self.common_attrs['compilation_time'] = (datetime.now()
                        - time_before_compilation).total_seconds()
        self.common_attrs['device'] = pycuda.autoinit.device.name()
        self.common_attrs['pycuda_version'] = pycuda.VERSION_TEXT
        #self.common_attrs['cuda_version'] = '.'.join(map(str, pycuda.driver.get_version()))        
        self.common_attrs['cuda_version'] = '.'.join(str(pycuda.driver.get_driver_version()))
        self.common_attrs.update(get_git_attrs())


    def run(self, wl,
             atm=None, surf=None, water=None, env=None, alis_options=None,
             NBPHOTONS=1e9, DEPO=0.0279, DEPO_WATER= 0.0906, THVDEG=0., PHVDEG=0., SEED=-1,
             RTER=6371., wl_proba=None,
             NBTHETA=45, NBPHI=90, NF=1e6,
             OUTPUT_LAYERS=0, XBLOCK=256, XGRID=256,
             NBLOOP=None, progress=True,
             le=None, flux=None, stdev=False, BEER=1, RR=1, WEIGHTRR=0.1, SZA_MAX=90., SUN_DISC=0,
             sensor=None, refraction=False, reflectance=True, myObjects=None, interval = None, IsAtm = 1, cusForward = None):
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

            - alis_options : required if compiled already with the alis option. Dictionary, field 'nlow'
                is the number of wavelength to fit the spectral dependence of scattering, 
                nlow-1 has to divide NW-1 where NW is the number of wavelengths, nlow has to be lesser than MAX_NLOW that is defined in communs.h,
                optionnal field 'njac' is the number of perturbed profiles, default is zero (None): no Jacobian

            - NBPHOTONS: number of photons launched

            - DEPO: (Air) Rayleigh depolarization ratio

            - DEPO_WATER: (Water) Rayleigh depolarization ratio

            - THVDEG: zenith angle of the observer in degrees
                the result corresponds to various positions of the sun
                NOTE: in plane parallel geometry, due to Fermat's principle, we
                can exchange the positions of the sun and observer.

            - PHVDEG: azimuth angle of the observer in degrees
                the result corresponds to various positions of the sun
                NOTE: It can be very useful to modify only this value instead
                      of all the positions of all the objects

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
                  The defaut output is two dimensional NBPHI x NBTHETA
                  If 'zip' is present in the dictionary, then 'th' and 'phi' covary and the output is only
                  one-dimensional NBTHETA, but user should verify that NBPHI==NBTHETA
                  Angles can be provided as scalar, lists or 1-dim arrays
                  Default None: cone sampling
                  NOTE: Overrides NBPHI and NBTHETA

            - alis_options : required if compiled already with the alis option. Dictionary, field 'nlow'
                is the number of wavelength to fit the spectral dependence of scattering, 
                nlow-1 has to divide NW-1 where NW is the number of wavelengths, nlow has to be lesser than MAX_NLOW that is defined in communs.h,

            - flux: if specified output is 'planar' or 'spherical' flux instead of radiance

            - stdev: calculate the standard deviation between each kernel run

            - RR: Russian Roulette ON  = 1
                                   OFF = 0

            - WEIGHTRR threshold weight to apply the Russian Roulette

            - BEER: if BEER=1 compute absorption using Beer-Lambert law, otherwise compute it with the Single scattering albedo
                (BEER automatically set to 1 if ALIS is chosen)

            - SZA_MAX : Maximum SZA for solar BOXES in case a Regulard grid and cone sampling

            - SUN_DISC : Angular size of the Sun disc in degree, 0 (default means no angular size)

            - sensor : sensor object or list, backward mode (from sensor to source), back should be set to True in the smartg constructor

            - refraction : include atmospheric refraction

            - reflectance : if flux is None, output is in reflectance units if True,(for plane parallel atmosphere). Otherwise
                is is in radiance units with Solar irradiance set to PI (default False)
            
            - myObjects : liste d'objets (objets de classe entity)
        
            - interval : liste composée de deux listes [[pxmin, pymin, pzmin], [[pxmin, pymin, pzmin]]
                         interval définit l'interval d'études des objets délimitée par deux points (pmin et pmax).

            - IsAtm (effet uniquement si myObjects != None) : si égal à 0 , cela permet dans le cas sans atmosphère,
                      d'empêcher certaines fuites de photons.

            - cusForward : None is the default mode (sun is a ponctual source targeting the origin (0,0,0)), else it
                           enable to use the RF or FF launching mode (see the class CusForward) --> cusForward=CusForward(...)

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
        
        #
        # Les Objets
        #
        if ((myObjects is not None) or (cusForward is not None)):
            # Prendre en compte la direction du soleil avec l'angle zenithal et azimuth
            vSun = Vector(0., 0., -1.)
            tSunTheta = Transform(); tSunPhi = Transform(); tSunThethaPhi = Transform();
            tSunTheta = tSunThethaPhi.rotateY(THVDEG) 
            tSunPhi = tSunThethaPhi.rotateZ(PHVDEG)   # pas vérifié car valeur gene = 0
            tSunThethaPhi = tSunTheta * tSunPhi
            vSun = tSunThethaPhi[vSun]
            vSun = Normalize(vSun)
            
        if (myObjects is not None):
            # Initialisations
            if interval is not None:
                Pmin_x = interval[0][0]
                Pmin_y = interval[0][1]
                Pmin_z = interval[0][2]
                Pmax_x = interval[1][0]
                Pmax_y = interval[1][1]
                Pmax_z = interval[1][2]
            else:
                Pmin_x = -300; Pmin_y = -300; Pmin_z = 0;
                Pmax_x = 300;  Pmax_y = 300; Pmax_z = 120;
            nObj = len(myObjects)
            myObjects0 = np.zeros(nObj, dtype=type_IObjets, order='C')
            TC = None; nbCx = int(0); nbCy = int(0);
            pp1 = 0.; pp2 = 0.; pp3 = 0.; pp4 = 0.;
            surfMir = 0.

            # Début de la boucle pour la prise en compte de tous les objets
            for i in range (0, nObj):

                # Pour l'instant 2 choix possibles : surface Sphérique ou Plane
                if isinstance(myObjects[i].geo, Spheric):    # si l'objet est une sphère
                    myObjects0['geo'][i] = 1 # reconnaitre la forme sphérique sur Cuda
                    
                    myObjects0['myRad'][i] = myObjects[i].geo.radius
                    myObjects0['z0'][i] = myObjects[i].geo.z0
                    myObjects0['z1'][i] = myObjects[i].geo.z1
                    myObjects0['phi'][i] = myObjects[i].geo.phi
                    
                elif isinstance(myObjects[i].geo, Plane):    # si l'objet est une surface plane
                    myObjects0['geo'][i] = 2 # reconnaitre la forme plane sur Cuda
                    
                    myObjects0['p0x'][i] = myObjects[i].geo.p1.x
                    myObjects0['p0y'][i] = myObjects[i].geo.p1.y
                    myObjects0['p0z'][i] = myObjects[i].geo.p1.z
                    
                    myObjects0['p1x'][i] = myObjects[i].geo.p2.x
                    myObjects0['p1y'][i] = myObjects[i].geo.p2.y
                    myObjects0['p1z'][i] = myObjects[i].geo.p2.z
                    
                    myObjects0['p2x'][i] = myObjects[i].geo.p3.x
                    myObjects0['p2y'][i] = myObjects[i].geo.p3.y
                    myObjects0['p2z'][i] = myObjects[i].geo.p3.z
                    
                    myObjects0['p3x'][i] = myObjects[i].geo.p4.x
                    myObjects0['p3y'][i] = myObjects[i].geo.p4.y
                    myObjects0['p3z'][i] = myObjects[i].geo.p4.z

                    normalBase = Normal(0, 0, 1);
                    # Prise en compte des transfos de rot en X, Y et Z
                    TpT0 = Transform()
                    TpRX0 = TpT0.rotateX(myObjects[i].transformation.rotx)
                    TpRY0 = TpT0.rotateY(myObjects[i].transformation.roty)
                    TpRZ0 = TpT0.rotateZ(myObjects[i].transformation.rotz)
                    TpT0 = TpRX0*TpRY0*TpRZ0
                    normalBase = TpT0[normalBase]
                    myObjects0['nBx'][i] = normalBase.x
                    myObjects0['nBy'][i] = normalBase.y
                    myObjects0['nBz'][i] = normalBase.z
                    
                else:    # si l'objet est autre chose (inconnu)
                    raise NameError("Your geometry can be only spheric or plane, please" + \
                                    " choose between Spheric or Plane classes!")

                # Affectation des transformations (rotations et translations)
                myObjects0['mvRx'][i] = myObjects[i].transformation.rotx
                myObjects0['mvRy'][i] = myObjects[i].transformation.roty
                myObjects0['mvRz'][i] = myObjects[i].transformation.rotz

                if (myObjects[i].transformation.rotOrder == "XYZ"):
                    myObjects0['rotOrder'][i] = 1
                elif(myObjects[i].transformation.rotOrder == "XZY"):
                    myObjects0['rotOrder'][i] = 2
                elif(myObjects[i].transformation.rotOrder == "YXZ"):
                    myObjects0['rotOrder'][i] = 3
                elif(myObjects[i].transformation.rotOrder == "YZX"):
                    myObjects0['rotOrder'][i] = 4
                elif(myObjects[i].transformation.rotOrder == "ZXY"):
                    myObjects0['rotOrder'][i] = 5
                elif(myObjects[i].transformation.rotOrder == "ZYX"):
                    myObjects0['rotOrder'][i] = 6
                else:
                    raise NameError('Unknown rotation order')

                myObjects0['mvTx'][i] = myObjects[i].transformation.transx
                myObjects0['mvTy'][i] = myObjects[i].transformation.transy
                myObjects0['mvTz'][i] = myObjects[i].transformation.transz

                # Prendre en compte le matériau de l'objet
                if isinstance(myObjects[i].materialAV, LambMirror):
                    myObjects0['materialAV'][i] = 1
                    myObjects0['reflecAV'][i] = myObjects[i].materialAV.reflectivity
                elif isinstance(myObjects[i].materialAV, Matte):
                    myObjects0['materialAV'][i] = 2
                    myObjects0['reflecAV'][i] = myObjects[i].materialAV.reflectivity
                elif isinstance(myObjects[i].materialAV, Mirror):
                    myObjects0['materialAV'][i] = 3
                    myObjects0['reflecAV'][i] = myObjects[i].materialAV.reflectivity
                else:
                    myObjects0['materialAV'][i] = 0
                    myObjects0['reflecAV'][i] = 0
                    
                if isinstance(myObjects[i].materialAR, LambMirror):
                    myObjects0['materialAR'][i] = 1
                    myObjects0['reflecAR'][i] = myObjects[i].materialAR.reflectivity
                elif isinstance(myObjects[i].materialAR, Matte):
                    myObjects0['materialAR'][i] = 2
                    myObjects0['reflecAR'][i] = myObjects[i].materialAR.reflectivity
                elif isinstance(myObjects[i].materialAR, Mirror):
                    myObjects0['materialAR'][i] = 3
                    myObjects0['reflecAR'][i] = myObjects[i].materialAR.reflectivity
                else:
                    myObjects0['materialAR'][i] = 0
                    myObjects0['reflecAR'][i] = 0
                    
                # Deux possibilités : l'objet est un reflecteur ou un recepteur   
                if (myObjects[i].name == "reflector"):
                    myObjects0['type'][i] = 1 # pour reconnaitre le reflect sur Cuda

                    # Etape cruciale pour la visualisation des résulats (mode forward restreint uniquement)
                    if (cusForward is not None):
                        if (cusForward.dict['LMODE'] == "RF"):
                            # Récupération des 4 points formant le rectangle représentant le reflecteur
                            pp1 = myObjects[i].geo.p1
                            pp2 = myObjects[i].geo.p2
                            pp3 = myObjects[i].geo.p3
                            pp4 = myObjects[i].geo.p4

                            # Prise en compte des transfos de rot en X et Y et Z
                            TpT = Transform()
                            TpRX = TpT.rotateX(myObjects[i].transformation.rotx)
                            TpRY = TpT.rotateY(myObjects[i].transformation.roty)
                            TpRZ = TpT.rotateZ(myObjects[i].transformation.rotz)
                            TpT = TpRX*TpRY*TpRZ
                            invTpT = TpT.inverse(TpT)

                            # Application des transfos sur les 4 points
                            pp1 = TpT[pp1]
                            pp2 = TpT[pp2]
                            pp3 = TpT[pp3]
                            pp4 = TpT[pp4]

                            # ====================
                            timeRefDir1b = (-1.* pp1.z)/vSun.z
                            timeRefDir2b = (-1.* pp2.z)/vSun.z
                            timeRefDir3b = (-1.* pp3.z)/vSun.z
                            timeRefDir4b = (-1.* pp4.z)/vSun.z
                            pp1b = pp1 + vSun*timeRefDir1b
                            pp2b = pp2 + vSun*timeRefDir2b
                            pp3b = pp3 + vSun*timeRefDir3b
                            pp4b = pp4 + vSun*timeRefDir4b
                            #print ("pointbis", pp1b, pp2b, pp3b, pp4b)
                            TwoAAbis = abs((pp1b.x - pp4b.x)*(pp2b.y - pp3b.y)) + abs((pp2b.x - pp3b.x)*(pp1b.y - pp4b.y))
                            surfMirbis = TwoAAbis/2.
                            #print ("surfMirbis", surfMirbis)
                            # ====================
                            surfMir += surfMirbis
                    
                elif (myObjects[i].name == "receiver"):
                    myObjects0['type'][i] = 2 # pour reconnaitre le recept sur Cuda
                    TC=myObjects[i].TC
                    sizeXmin = min(myObjects[i].geo.p1.x, myObjects[i].geo.p2.x,
                                   myObjects[i].geo.p3.x, myObjects[i].geo.p4.x)
                    sizeXmax = max(myObjects[i].geo.p1.x, myObjects[i].geo.p2.x,
                                   myObjects[i].geo.p3.x, myObjects[i].geo.p4.x)
                    sizeX = sizeXmax - sizeXmin
                    sizeYmin = min(myObjects[i].geo.p1.y, myObjects[i].geo.p2.y,
                                   myObjects[i].geo.p3.y, myObjects[i].geo.p4.y)
                    sizeYmax = max(myObjects[i].geo.p1.y, myObjects[i].geo.p2.y,
                                   myObjects[i].geo.p3.y, myObjects[i].geo.p4.y)
                    sizeY = sizeYmax - sizeYmin
                    nbCx = int(sizeX/TC)
                    nbCy = int(sizeY/TC)
                elif (myObjects[i].name == "surf"):
                    myObjects0['type'][i] = 3 # pour reconnaitre le recept sur Cuda
                else:
                    raise NameError('You have to specify if your object is a reflector or a receiver!')
                
            myObjects0 = to_gpu(myObjects0)          
        else:
            nObj = 0
            myObjects0 = None #np.zeros(1, dtype=type_IObjets, order='C')
            Pmin_x = None; Pmin_y = None; Pmin_z = None;
            Pmax_x = None; Pmax_y = None; Pmax_z = None;
            IsAtm = None; TC = None; nbCx = 10; nbCy = 10;
        # END OBJ ===================================================

        if cusForward is not None:
            if (cusForward.dict['LMODE'] == "FF"):
                nn3 = Normal(vSun)
                nn1, nn2 = CoordinateSystem(nn3)
                # Création d'une matrice appelé matrice de passage
                mm2 = np.zeros((4,4), dtype=np.float64)
                # Remplissage de la matrice de passage en fonction du repère (nn3 étant le nouvel axe z)
                mm2[0,0] = nn1.x ; mm2[0,1] = nn2.x ; mm2[0,2] = nn3.x ; mm2[0,3] = 0. ;
                mm2[1,0] = nn1.y ; mm2[1,1] = nn2.y ; mm2[1,2] = nn3.y ; mm2[1,3] = 0. ;
                mm2[2,0] = nn1.z ; mm2[2,1] = nn2.z ; mm2[2,2] = nn3.z ; mm2[2,3] = 0. ;
                mm2[3,0] = 0.    ; mm2[3,1] = 0.    ; mm2[3,2] = 0.    ; mm2[3,3] = 1. ;
                xnn = float(cusForward.dict['CFX'])/2.
                ynn = float(cusForward.dict['CFY'])/2.
                ppn1 = Point(-xnn, -ynn, 0.)
                ppn2 = Point(xnn, -ynn, 0.)
                ppn3 = Point(-xnn, ynn, 0.)
                ppn4 = Point(xnn, ynn, 0.)
                # Création de la transformation permettant le changement de base
                mm2Inv = np.transpose(mm2)
                ooTwn = Transform(m = mm2, mInv = mm2Inv)
                ppn1 = ooTwn[ppn1]
                ppn2 = ooTwn[ppn2]
                ppn3 = ooTwn[ppn3]
                ppn4 = ooTwn[ppn4]
                v_nn = ooTwn[vSun]
                timen1 = (-1.* ppn1.z)/v_nn.z
                timen2 = (-1.* ppn2.z)/v_nn.z
                timen3 = (-1.* ppn3.z)/v_nn.z
                timen4 = (-1.* ppn4.z)/v_nn.z
                ppn1 += v_nn*timen1
                ppn2 += v_nn*timen2
                ppn3 += v_nn*timen3
                ppn4 += v_nn*timen4
                TwoAn = abs((ppn1.x - ppn4.x)*(ppn2.y - ppn3.y)) + abs((ppn2.x - ppn3.x)*(ppn1.y - ppn4.y))
                surfMir = TwoAn/2.

        
        #
        # initialization
        #              
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
        MAX_HIST = 1024*1024

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

        NLOW=1
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
            prof_atm_gpu = init_profile(wl, prof_atm, 'atm')
            NATM = len(prof_atm.axis('z_atm')) - 1
        else:
            faer = gpuzeros(1, dtype='float32')
            prof_atm_gpu = to_gpu(np.zeros(1, dtype=type_Profile))
            NATM = 0


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

        # cusForward definition
        if cusForward is None:
            cusForward = CusForward(CFX=0., CFY=0., CFMODE = 2)
            #cusForward = CusForward(CFX=0., CFY=0., CFMODE = 0)
            
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
        if refraction: REFRAC=1

        HORIZ = 1
        if (not self.pp and not reflectance): HORIZ = 0

        # initialization of the constants
        InitConst(surf, env, NATM, NOCE, self.mod,
                  NBPHOTONS, NBLOOP, THVDEG, DEPO,
                  XBLOCK, XGRID, NLAM, SIM, NF,
                  NBTHETA, NBPHI, OUTPUT_LAYERS,
                  RTER, LE, ZIP, FLUX, NLVL, NPSTK,
                  NWLPROBA, BEER, RR, WEIGHTRR, NLOW, NJAC, 
                  NSENSOR, REFRAC, HORIZ, SZA_MAX, SUN_DISC, cusForward, nObj,
                  Pmin_x, Pmin_y, Pmin_z, Pmax_x, Pmax_y, Pmax_z, IsAtm, TC, nbCx, nbCy, HIST)

        # Initialize the progress bar
        p = Progress(NBPHOTONS, progress)

        # Initialize the RNG
        SEED = self.rng.setup(SEED, XBLOCK, XGRID)

        # Loop and kernel call
        (NPhotonsInTot, tabPhotonsTot, tabDistTot, tabHistTot, errorcount, 
         NPhotonsOutTot, sigma, Nkernel, secs_cuda_clock, cMatVisuRecep, categories
        ) = loop_kernel(NBPHOTONS, faer, foce,
                        NLVL, NATM, NOCE, MAX_HIST, NLOW, NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                        NLAM, NSENSOR, self.double, self.kernel, self.kernel2, p, X0, le, tab_sensor, spectrum,
                        prof_atm_gpu, prof_oc_gpu,
                        wl_proba_icdf, stdev, self.rng, myObjects0, TC, nbCx, nbCy, hist=hist)

        attrs['kernel time (s)'] = secs_cuda_clock
        attrs['number of kernel iterations'] = Nkernel
        attrs['seed'] = SEED
        attrs.update(self.common_attrs)

        # En rapport avec l'implémentation des objets (permet le visuel des res du recept)
        if (TC is not None):
            if (cusForward is not None):
                for i in range (0, 9):
                    cMatVisuRecep[i][:][:] = cMatVisuRecep[i][:][:] * ((surfMir)/(TC*TC*NBPHOTONS))
            else:
                cMatVisuRecep[:][:][:] = cMatVisuRecep[:][:][:]
                                                                   
        # finalization
        output = finalize(tabPhotonsTot, tabDistTot, tabHistTot, wl[:], NPhotonsInTot, errorcount, NPhotonsOutTot,
                          OUTPUT_LAYERS, tabTransDir, SIM, attrs, prof_atm, prof_oc,
                          sigma, THVDEG, reflectance, HORIZ, le=le, flux=flux, back=self.back, 
                          SZA_MAX=SZA_MAX, SUN_DISC=SUN_DISC, hist=hist, cMatVisuRecep=cMatVisuRecep, cats = categories)
        
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
             sigma, THVDEG, reflectance, HORIZ, le=None, flux=None,
             back=False, SZA_MAX=90., SUN_DISC=0, hist=False, cMatVisuRecep = None, cats = None):
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
    tabDistFinal = tabDistFinal.swapaxes(3,4)
    if hist : tabHistFinal = tabHistFinal.swapaxes(4,5)
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
    if hist : axnames3 = ['None', 'None', 'Zenith angles']
    iphi     = slice(None)
    if hist : axnames3 = ['None', 'None', 'Azimuth angles', 'Zenith angles']
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
    m.add_dataset('cdist_up (TOA)', tabDistFinal[UPTOA,:,isen,iphi,:], axnames2)
    if hist : m.add_dataset('disth_up (TOA)', tabHistFinal[UPTOA,:,:,isen,iphi,:],axnames3)

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
        m.add_dataset('cdist_down (0+)', tabDistFinal[DOWN0P,:,isen,iphi,:],axnames2)
        if hist : m.add_dataset('disth_down (0+)', tabHistFinal[DOWN0P,:,:,isen,iphi,:],axnames3)

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
        m.add_dataset('cdist_up (0-)', tabDistFinal[UP0M,:,isen,iphi,:],axnames2)
        if hist : m.add_dataset('disth_up (0-)', tabHistFinal[UP0M,:,:,isen,iphi,:],axnames3)

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
        m.add_dataset('cdist_down (0-)', tabDistFinal[DOWN0M,:,isen,iphi,:],axnames2)
        if hist : m.add_dataset('disth_down (0-)', tabHistFinal[DOWN0M,:,:,isen,iphi,:],axnames3)

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
        m.add_dataset('cdist_up (0+)', tabDistFinal[UP0P,:,isen,iphi,:],axnames2)
        if hist : m.add_dataset('disth_up (0+)', tabHistFinal[UP0P,:,:,isen,iphi,:],axnames3)

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
        m.add_dataset('cdist_down (B)', tabDistFinal[DOWNB,:,isen,iphi,:],axnames2)
        if hist : m.add_dataset('disth_down (B)', tabHistFinal[DOWNB,:,:,isen,iphi,:],axnames3)


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
        
        
    if (cats is not None):
        m.add_dataset('catWeightPh', np.array([cats[0], cats[4], cats[8], cats[12], cats[16], cats[20],
                                               cats[24], cats[28]], dtype=np.float64), ['Categories'])
        m.add_dataset('catNbPh', np.array([cats[1], cats[5], cats[9], cats[13], cats[17], cats[21],
                                           cats[25], cats[29]], dtype=np.float64), ['Categories'])
        m.add_dataset('catErrP', np.array([cats[2], cats[6], cats[10], cats[14], cats[18], cats[22], 
                                           cats[26], cats[30]], dtype=np.float64), ['Categories'])
        m.add_dataset('catErrAbs', np.array([cats[3], cats[7], cats[11], cats[15], cats[19], cats[23], 
                                             cats[27], cats[31]], dtype=np.float64), ['Categories'])
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
    #P11 = T_demi*(DELTA+DELTA_PRIM)
    P12 = T_demi*DELTA_PRIM
    P33bis = T_demi*DELTA
    P44bis = P33bis*DELTA_SECO

    # parameters equally spaced in scattering probabiliy [0, 1]
    # pha['p_P11'][:] = P11
    # pha['p_P12'][:] = P12
    # pha['p_P22'][:] = T_demi*(DELTA*cTh2[:] + DELTA_PRIM)
    # pha['p_P33'][:] = P33bis*cTh[:] # U
    # pha['p_P44'][:] = P44bis*cTh[:] # V
    # pha['p_ang'][:] = theta[:] # angle

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

    # pha['a_P11'][:] = P11
    # pha['a_P12'][:] = P12
    # pha['a_P22'][:] = T_demi*(DELTA*cTh2LE[:] + DELTA_PRIM) 
    # pha['a_P33'][:] = P33bis*cThLE[:]  # U
    # pha['a_P44'][:] = P44bis*cThLE[:]  # V

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


def InitConst(surf, env, NATM, NOCE, mod,
              NBPHOTONS, NBLOOP, THVDEG, DEPO,
              XBLOCK, XGRID,NLAM, SIM, NF,
              NBTHETA, NBPHI, OUTPUT_LAYERS,
              RTER, LE, ZIP, FLUX, NLVL, NPSTK, NWLPROBA, BEER, RR, 
              WEIGHTRR, NLOW, NJAC, NSENSOR, REFRAC, HORIZ, SZA_MAX, SUN_DISC, cusForward, nObj,
              Pmin_x, Pmin_y, Pmin_z, Pmax_x, Pmax_y, Pmax_z, IsAtm, TC, nbCx, nbCy, HIST) :
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
    copy_to_device('ZIPd', ZIP, np.int32)
    copy_to_device('FLUXd', FLUX, np.int32)
    #copy_to_device('MId', MI, np.int32)
    copy_to_device('NLVLd', NLVL, np.int32)
    copy_to_device('NPSTKd', NPSTK, np.int32)
    copy_to_device('BEERd', BEER, np.int32)
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
        copy_to_device('Pmin_x', Pmin_x, np.float32)
        copy_to_device('Pmin_y', Pmin_y, np.float32)
        copy_to_device('Pmin_z', Pmin_z, np.float32)
        copy_to_device('Pmax_x', Pmax_x, np.float32)
        copy_to_device('Pmax_y', Pmax_y, np.float32)
        copy_to_device('Pmax_z', Pmax_z, np.float32)
        copy_to_device('IsAtm', IsAtm, np.int32)
        if TC is not None:
            copy_to_device('TCd', TC, np.float32)
            copy_to_device('nbCx', nbCx, np.int32)
            copy_to_device('nbCy', nbCy, np.int32)
        if (  (cusForward != None) and (cusForward.dict['LMODE'] == "RF")  ):
            copy_to_device('LMODEd', 1, np.int32)
        if (  (cusForward != None) and (cusForward.dict['LMODE'] == "FF")  ):
            copy_to_device('CFXd', cusForward.dict['CFX'], np.float32)
            copy_to_device('CFYd', cusForward.dict['CFY'], np.float32)
            copy_to_device('CFTXd', cusForward.dict['CFTX'], np.float32)
            copy_to_device('CFTYd', cusForward.dict['CFTY'], np.float32)
            copy_to_device('LMODEd', 2, np.int32)
        
def init_profile(wl, prof, kind):
    '''
    take the profile as a MLUT, and setup the gpu structure

    kind = 'atm' or 'oc' for atmosphere or ocean
    '''

    # reformat to smartg format

    NLAY = len(prof.axis('z_'+kind)) - 1
    shp = (len(wl), NLAY+1)
    prof_gpu = np.zeros(shp, dtype=type_Profile, order='C')

    if kind == "oc":
        prof_gpu['z'][0,:] = prof.axis('z_'+kind)  * 1e-3 # to Km
        prof_gpu['n'][0,:] = 1.34;
    else:
        prof_gpu['z'][0,:] = prof.axis('z_'+kind)
        prof_gpu['n'][:,:] = prof['n_'+kind].data[...]
    prof_gpu['z'][1:,:] = -999.      # other wavelengths are NaN

    prof_gpu['OD'][:,:] = prof['OD_'+kind].data[...]
    prof_gpu['OD_sca'][:] = prof['OD_sca_'+kind].data[...]
    prof_gpu['OD_abs'][:] = prof['OD_abs_'+kind].data[...]
    prof_gpu['pmol'][:] = prof['pmol_'+kind].data[...]
    prof_gpu['ssa'][:] = prof['ssa_'+kind].data[...]
    #NEW !!!
    prof_gpu['pine'][:] = prof['pine_'+kind].data[...]
    prof_gpu['FQY1'][:] = prof['FQY1_'+kind].data[...]
    #NEW !!!
    if 'iphase_'+kind in prof.datasets():
        prof_gpu['iphase'][:] = prof['iphase_'+kind].data[...]

    return to_gpu(prof_gpu)



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

 
def loop_kernel(NBPHOTONS, faer, foce, NLVL, NATM, NOCE, MAX_HIST, NLOW,
                NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                NLAM, NSENSOR, double, kern, kern2, p, X0, le, tab_sensor, spectrum,
                prof_atm, prof_oc, wl_proba_icdf, stdev, rng, myObjects0, TC, nbCx, nbCy, hist=False):
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
    
    # Initializations linked to objects
    if TC is not None:
        nbPhCat = gpuzeros(8, dtype=np.uint64) # vector to fill the number of photons for  each categories
        if double:
            wPhCat = gpuzeros(8, dtype=np.float64)  # vector to fill the weight of photons for each categories
            tabObjInfo = gpuzeros((9, nbCx, nbCy), dtype=np.float64)
        else:
            wPhCat = gpuzeros(8, dtype=np.float32)
            tabObjInfo = gpuzeros((9, nbCx, nbCy), dtype=np.float32)
        tabMatRecep = np.zeros((9, nbCx, nbCy), dtype=np.float64)  
        # vecteur comprenant : weightPhotons, nbPhoton, err% et errAbs pour
        # les 8 categories donc 4 x 8 valeurs = 32. vecCat[0], [1], [2] et [3]
        # pour la categorie 1 et ainsi de suite...
        vecCats = np.zeros((32), dtype=np.float64) 
    else:
        nbPhCat = gpuzeros(1, dtype=np.uint64)
        if double:
            wPhCat = gpuzeros(1, dtype=np.float64)
            tabObjInfo = gpuzeros((1, 1, 1), dtype=np.float64)
        else:
            wPhCat = gpuzeros(1, dtype=np.float32)
            tabObjInfo = gpuzeros((1, 1, 1), dtype=np.float32)
            
    # Initialize the array for error counting
    NERROR = 32
    errorcount = gpuzeros(NERROR, dtype='uint64')

    
    if (NATM+NOCE >0) : tabDistTot = gpuzeros((NLVL,NATM+NOCE,NSENSOR,NBTHETA,NBPHI), dtype=np.float64)
    else : tabDistTot = gpuzeros((NLVL,1,NSENSOR,NBTHETA,NBPHI), dtype=np.float64)
    if hist : tabHistTot = gpuzeros((NLVL,MAX_HIST,(NATM+NOCE+NPSTK+NLOW),NSENSOR,NBTHETA,NBPHI), dtype=np.float32)
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
        if (NATM+NOCE >0) : tabDist = gpuzeros((NLVL,NATM+NOCE,NSENSOR,NBTHETA,NBPHI), dtype=np.float64)
        else : tabDist = gpuzeros((NLVL,1,NSENSOR,NBTHETA,NBPHI), dtype=np.float64)
    else:
        tabPhotons = gpuzeros((NLVL,NPSTK,NSENSOR,NLAM,NBTHETA,NBPHI), dtype=np.float32)
        if (NATM+NOCE >0) : tabDist = gpuzeros((NLVL,NATM+NOCE,NSENSOR,NBTHETA,NBPHI), dtype=np.float32)
        else : tabDist = gpuzeros((NLVL,1,NSENSOR,NBTHETA,NBPHI), dtype=np.float32)

    if hist : tabHist = gpuzeros((NLVL,MAX_HIST,(NATM+NOCE+NPSTK+NLOW),NSENSOR,NBTHETA,NBPHI), dtype=np.float32)
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
    while(np.sum(NPhotonsInTot.get()) < NBPHOTONS):
        tabPhotons.fill(0.)
        NPhotonsOut.fill(0)
        NPhotonsIn.fill(0)
        Counter.fill(0)
        # en rapport avec les objets
        tabObjInfo.fill(0)
        wPhCat.fill(0)
        
        start_cuda_clock = cuda.Event()
        end_cuda_clock = cuda.Event()
        start_cuda_clock.record()

        # kernel launch

        if myObjects0 is not None:
            kern(spectrum, X0, faer, foce,
                 errorcount, nThreadsActive, tabPhotons, tabDist, tabHist,
                 Counter, NPhotonsIn, NPhotonsOut, tabthv, tabphi, tab_sensor,
                 prof_atm, prof_oc, wl_proba_icdf, rng.state, tabObjInfo, myObjects0, nbPhCat, wPhCat,
                 block=(XBLOCK, 1, 1), grid=(XGRID, 1, 1))
        else:
            kern(spectrum, X0, faer, foce,
                 errorcount, nThreadsActive, tabPhotons, tabDist, tabHist,
                 Counter, NPhotonsIn, NPhotonsOut, tabthv, tabphi, tab_sensor,
                 prof_atm, prof_oc, wl_proba_icdf, rng.state,
                 block=(XBLOCK, 1, 1), grid=(XGRID, 1, 1))

        end_cuda_clock.record()
        end_cuda_clock.synchronize()
        secs_cuda_clock = secs_cuda_clock + start_cuda_clock.time_till(end_cuda_clock)

        cuda.Context.synchronize()
        np.set_printoptions(precision=5, linewidth=150)

        if TC is not None:
            # Tableau de la repartition des poids (photons) sur la surface du recepteur
            tabMatRecep += tabObjInfo[:, :, :].get()
            for i in range (0, 8):
                # Comptage des poids pour chaque categories
                vecCats[i*4] += wPhCat[i].get();

        
        L = NPhotonsIn   # number of photons launched by last kernel
        NPhotonsInTot += L

        NPhotonsOutTot += NPhotonsOut
        S = tabPhotons   # sum of weights for the last kernel
        tabPhotonsTot += S
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!
        T = tabDist
        tabDistTot += T
        if hist :
            H = tabHist
            tabHistTot = H
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        if (myObjects0 is not None):
            import sys
            print ("Avancement... NPhotonsIn host (smartg) is:", NPhotonsInTot, file = sys.stderr)

        N_simu += 1
        if stdev:
            (NSENSOR,NLAM) = NPhotonsIn.shape
            L = L.reshape((1,1,NSENSOR,NLAM,1,1))   # broadcast to tabPhotonsTot
            #warn('stdev is activated: it is known to slow down the code considerably.')
            SoverL = S.get()/L.get()
            sum_x += SoverL
            sum_x2 += (SoverL)**2

        # update of the progression Bar
        sphot = np.sum(NPhotonsInTot.get())
        p.update(sphot,
                'Launched {:.3g} photons'.format(sphot))
    secs_cuda_clock = secs_cuda_clock*1e-3

    if TC is not None:
        for i in range (0, 8):
            # # Comptage des poids pour chaque categories
            # vecCats[i*4] = wPhCat[i].get();
            # Comptage du nombre de photons pour chaque categories
            vecCats[(i*4)+1] = nbPhCat[i].get();
            # Erreur relatives et absolues pour chaque categories
            if (vecCats[i*4] == 0 or vecCats[(i*4)+1] == 0):
                vecCats[(i*4)+2] = 0.
                vecCats[(i*4)+3] = 0.
            else:    
                vecCats[(i*4)+2] = float((100.*(1./vecCats[(i*4)+1]**0.5)));
                vecCats[(i*4)+3] = vecCats[i*4]*(1./vecCats[(i*4)+1]**0.5);
    else:
        tabMatRecep = None
        vecCats = None

    if stdev:
        # finalize the calculation of the standard deviation
        sigma = np.sqrt(sum_x2/N_simu - (sum_x/N_simu)**2)

        # extrapolate in 1/sqrt(N_simu)
        sigma /= np.sqrt(N_simu)
    else:
        sigma = None

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return NPhotonsInTot.get(), tabPhotonsTot.get(), tabDistTot.get(), tabHistTot.get(), errorcount, \
        NPhotonsOutTot.get(), sigma, N_simu, secs_cuda_clock, tabMatRecep, vecCats
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #return NPhotonsInTot.get(), tabPhotonsTot.get(), errorcount, NPhotonsOutTot.get(), sigma, N_simu, secs_cuda_clock


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
