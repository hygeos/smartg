#!/usr/bin/env python
# encoding: utf-8


'''
SMART-G
Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU
'''



import numpy as np
import time
from datetime import datetime
from numpy import pi
from tools.profile.profil import AeroOPAC, Profile, KDIS, KDIS_IBAND, REPTRAN, REPTRAN_IBAND, CloudOPAC
from tools.cdf import ICDF
from tools.water.iop_spm import IOP_SPM
from tools.water.iop_mm import IOP_MM
from tools.water.iop_AOS_water import IOP_AOS_WATER
from tools.water.iop_PandR import IOP_PandR
from tools.water.phase_functions import PhaseFunction
from os.path import dirname, realpath, join, basename, exists
from warnings import warn
from tools.progress import Progress
from tools.luts import merge, read_lut_hdf, read_mlut_hdf, LUT, MLUT
from scipy.interpolate import interp1d
import subprocess
from collections import OrderedDict
from pycuda.gpuarray import to_gpu, zeros as gpuzeros
import pycuda.driver as cuda


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
UPTOA = 0
DOWN0P = 1
DOWN0M = 2
UP0P = 3
UP0M = 4

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
    ('z',      'float32'),   # // altitude
    ('tau',    'float32'),   # // cumulated optical thickness (from top)
    ('pmol',   'float32'),   # // probability of pure Rayleigh scattering event
    ('ssa',    'float32'),   # // single scattering albedo (scatterer only)
    ('abs',    'float32'),   # // absorption coefficient
    ('iphase', 'int32'),    # // phase function index
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
    '''
    def __init__(self, ENV=0, ENV_SIZE=0., X0=0., Y0=0., ALB=0.5):
        self.dict = {
                'ENV': ENV,
                'ENV_SIZE': ENV_SIZE,
                'X0': X0,
                'Y0': Y0,
                'SURFALB': ALB,
                }

    def __str__(self):
        return 'ENV={ENV_SIZE}-X={X0:.1f}-Y={Y0:.1f}'.format(**self.dict)


def smartg(*args, **kwargs):

    warn('function smartg will be deprecated. Please use Smartg(<compilation_option>).run(<runtime_options>)')

    comp_kwargs = {}
    for k in ['pp', 'debug', 'alt_move', 'debug_photon', 'double']:
        if k in kwargs:
            comp_kwargs[k] = kwargs.pop(k)

    return Smartg(**comp_kwargs).run(*args, **kwargs)


class Smartg(object):

    def __init__(self, pp=True, debug=False,
                 alt_move=False, debug_photon=False,
                 double=False):
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
        '''
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        from pycuda.driver import module_from_buffer

        self.pp = pp
        self.double = double

        #
        # compilation option
        #
        options = []
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

        #
        # compile the kernel or load binary
        #
        time_before_compilation = datetime.now()
        if exists(src_device):

            # load device.cu
            src_device_content = open(src_device).read()

            # kernel compilation
            self.mod = SourceModule(src_device_content,
                               nvcc='/usr/local/cuda/bin/nvcc',
                               options=options,
                               no_extern_c=True,
                               cache_dir='/tmp/',
                               include_dirs=[dir_src,
                                   join(dir_src, 'incRNGs/Random123/')])
        elif exists(binnames[pp]):
            # load existing binary
            print 'read binary', binnames[pp]
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
             NBPHOTONS=1e9, DEPO=0.0279, THVDEG=0., SEED=-1,
             RTER=6371., wl_proba=None,
             NBTHETA=45, NBPHI=90, NF=1e6,
             OUTPUT_LAYERS=0, XBLOCK=256, XGRID=256,
             NBLOOP=None, progress=True,
             le=None, flux=None, stdev=False):
        '''
        Run a SMART-G simulation

        Arguments:

            - wl: wavelength in nm (float)
                  or: a list/array of wavelengths
                  or: a list of IBANDs (reptran, kdis)

            - atm: Profile object
                default None (no atmosphere)
                Example:
                    # clear atmosphere, AFGL midlatitude summer
                    Profile('afglms')
                    # AFGL tropical with maritime clear aerosols AOT(550)=0.3
                    Profile('afglms', aer=AeroOPAC('maritime_clean', 0.3, 550., ))

            - surf: Surface object
                default None (no surface)
                RoughSurface(WIND=5.)  # wind-roughened ocean surface
                FlatSurface()          # flat air-water interface
                LambSurface(ALB=0.1)   # Lambertian surface of albedo 0.1

            - water: Iop object, providing options relative to the ocean surface
                default None (no ocean)

            - env: environment effect parameters (dictionary)
                default None (no environment effect)

            - NBPHOTONS: number of photons launched

            - DEPO: Rayleigh depolarization ratio

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
                     or a Queue object to store the progress as (max_value), then (current_value, message), finally 'message'

            - le: Local Estimate method activation
                  Provide output geometries in radians like so:
                  {'th':<float32 array>, 'phi': <float32 array>}
                  default None: cone sampling
                NOTE: Overwrite NBPHI and NBTHETA

            - flux: if specified output is 'planar' or 'spherical' flux instead of radiance

            - stdev: calculate the standard deviation between each kernel run


        Return value:
        ------------

        Returns a MLUT object containing:
            - the polarized reflectance (I,Q,U,V) at the different layers
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

        # number of output levels
        # warning! values defined in communs.h should be < LVL
        NLVL = 5

        # number of Stokes parameters of the radiation field
        NPSTK = 4

        t0 = datetime.now()

        attrs = OrderedDict()
        attrs.update({'processing started at': t0})
        attrs.update({'VZA': THVDEG})
        attrs.update({'MODE': {True: 'PPA', False: 'SSA'}[self.pp]})
        attrs.update({'XBLOCK': XBLOCK})
        attrs.update({'XGRID': XGRID})

        if SEED == -1:
            # SEED is based on clock
            SEED = np.uint32((datetime.now()
                - datetime.utcfromtimestamp(0)).total_seconds()*1000)

        assert isinstance(wl, (float, list, np.ndarray))
        if isinstance(wl, list):
            if (False not in map(lambda x: isinstance(x, (REPTRAN_IBAND, KDIS_IBAND)), wl)):
                # wl is a list of REPTRAN_IBANDs or KDIS_IBANDS
                wavelengths = map(lambda x: x.w, wl)
            else:
                # wl is a list of floats
                assert (False not in map(lambda x: isinstance(x, float), wl))
                wavelengths = wl
        else:
            wavelengths = wl
        NLAM = np.array(wavelengths).size

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
        # get the phase function and the atmospheric profiles

        prof_atm, phasesAtm, NATM, HATM, taumol, tauaer = get_profAtm(wl,atm)

        # computation of the impact point
        x0, y0, z0, tabTransDir = impactInit(self.pp, HATM, NATM, NLAM, prof_atm, THVDEG, RTER)
        X0 = to_gpu(np.array([x0, y0, z0], dtype='float32'))


        #
        # surface
        #
        if surf is None:
            # default surface parameters
            surf = FlatSurface()

        #
        # ocean profile
        # get the phase function and oceanic profile
        prof_oc, phasesOc, NOCE = get_profOc(wavelengths, water, NLAM)


        #
        # albedo and adjacency effect
        #
        spectrum = np.zeros(NLAM, dtype=type_Spectrum)
        spectrum['lambda'] = np.array(wavelengths)
        if env is None:
            # default values (no environment effect)
            env = Environment()
            if 'SURFALB' in surf.dict:
                spectrum['alb_surface'] = surf.dict['SURFALB']
            else:
                spectrum['alb_surface'] = -999.
        else:
            spectrum['alb_surface'] = env.alb.get(wavelengths)

        if water is None:
            spectrum['alb_seafloor'] = -999.
        else:
            spectrum['alb_seafloor'] = water.alb
        spectrum = to_gpu(spectrum)

        # Local Estimate option
        LE = 0
        if le!=None:
            LE = 1
            NBTHETA =  le['th'].shape[0]
            NBPHI   =  le['phi'].shape[0]
        
        FLUX = 0
        if flux == 'spherical' : FLUX = 1

        # computation of the phase functions
        if(SIM == 0 or SIM == 2 or SIM == 3):
            foce = calculF(phasesOc, NF, DEPO)
        else:
            foce = gpuzeros(1, dtype='float32')

        if(SIM == -2 or SIM == 1 or SIM == 2):
            faer = calculF(phasesAtm, NF, DEPO)
        else:
            faer = gpuzeros(1, dtype='float32')

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
                       NWLPROBA)


        # Initialize the progress bar
        p = Progress(NBPHOTONS, progress)


        # Loop and kernel call
        (NPhotonsInTot,
                tabPhotonsTot, errorcount, NPhotonsOutTot, sigma, Nkernel, secs_cuda_clock
                ) = loop_kernel(NBPHOTONS, faer, foce,
                                NLVL, NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                                NLAM, self.double, self.kernel, p, X0, le, spectrum,
                                to_gpu(prof_atm), to_gpu(prof_oc),
                                wl_proba_icdf, SEED, stdev)
        attrs['kernel time (s)'] = secs_cuda_clock
        attrs['number of kernel iterations'] = Nkernel
        attrs.update(self.common_attrs)

        # finalization
        output = finalize(tabPhotonsTot, wavelengths, NPhotonsInTot, errorcount, NPhotonsOutTot,
                           OUTPUT_LAYERS, tabTransDir, SIM, attrs, prof_atm, phasesAtm,
                           taumol, tauaer, sigma, le=le, flux=flux)
        output.set_attr('processing time (s)', (datetime.now() - t0).total_seconds())

        p.finish('Done! | Received {:.1%} of {:.3g} photons ({:.1%})'.format(
            np.sum(NPhotonsOutTot[0,...])/float(np.sum(NPhotonsInTot)),
            np.sum(NPhotonsInTot),
            np.sum(NPhotonsInTot)/float(NBPHOTONS),
            ))

        return output


def reptran_merge(m, ibands, verbose=True):
    '''
    merge (average) several correlated-k bands in the dimension 'Wavelength'

    '''
    from collections import Counter

    # count how many ibands share the same band
    c = Counter(map(lambda x: x.band, ibands))
    nc = len(c)

    assert len(ibands) == len(m.axes['Wavelength'])

    mmerged = MLUT()
    mmerged.add_axis('Azimuth angles', m.axes['Azimuth angles'])
    mmerged.add_axis('Zenith angles', m.axes['Zenith angles'])

    # wavelength axis
    i = 0
    wl = []
    for _ in xrange(nc):
        b = ibands[i].band
        wl.append(np.average(b.awvl, weights=b.awvl_weight))
        i += c[b]

    mmerged.add_axis('Wavelength', wl)

    # for each dataset
    for (name, data, axnames) in m.data:

        if axnames != ['Wavelength', 'Azimuth angles', 'Zenith angles']:
            if verbose: print 'Skipping dataset', name
            continue

        _, nphi, ntheta = data.shape

        mdata = np.zeros((nc, nphi, ntheta), dtype=data.dtype)

        i0 = 0
        for i in xrange(nc):  # loop on the merged bands
            S, norm = 0., 0.
            b = ibands[i0].band
            for j in xrange(c[b]): # loop on the internal bands
                weight = ibands[i0+j].weight
                extra = ibands[i0+j].extra
                S += data[i0+j] * weight * extra
                norm += weight * extra
            i0 += c[b]

            mdata[i,:,:] = S/norm

        mmerged.add_dataset(name, mdata, ['Wavelength', 'Azimuth angles', 'Zenith angles'])

    mmerged.set_attrs(m.attrs)

    if verbose:
        print 'Merged {} wavelengths down to {}'.format(len(ibands), nc)

    return mmerged


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
        OUTPUT_LAYERS, tabTransDir, SIM, attrs, prof_atm, phasesAtm,
        taumol, tauaer, sigma, le=None, flux=None):
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
        m.add_axis('Wavelength', wl)
        ilam = slice(None)
        axnames.insert(0, 'Wavelength')
    else:
        m.set_attr('Wavelength', str(wl))
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

    # direct transmission
    if NLAM > 1:
        m.add_dataset('direct transmission', tabTransDir,
                axnames=['Wavelength'])
    else:
        m.set_attr('direct transmission', str(tabTransDir[0]))

    # write atmospheric profiles
    if SIM in [-2, 1, 2]:
        m.add_axis('ALT', prof_atm['z'][0,:])
        for (key,_) in type_Profile:
            if key == 'z':
                continue
            if NLAM == 1:
                m.add_dataset(key, prof_atm[key].ravel(), ['ALT'])
            else:
                m.add_dataset(key, prof_atm[key], ['Wavelength', 'ALT'])

        if (taumol is not None) and (tauaer is not None):
            if NLAM == 1:
                m.add_dataset('taumol', taumol.ravel(), ['ALT'])
                m.add_dataset('tauaer', tauaer.ravel(), ['ALT'])
            else:
                m.add_dataset('taumol', taumol, ['Wavelength', 'ALT'])
                m.add_dataset('tauaer', tauaer, ['Wavelength', 'ALT'])

    # write atmospheric phase functions
    if len(phasesAtm) > 0:
        npstk = 4
        nscat = len(phasesAtm[0].ang)
        npha = len(phasesAtm)
        pha = np.zeros((npha, nscat, npstk), dtype='float')
        m.add_axis('SCAT_ANGLE_ATM', phasesAtm[0].ang_in_rad()*180./pi)

        for i in xrange(npha):
            assert phasesAtm[i].phase.shape == phasesAtm[0].phase.shape
            pha[i, ] = phasesAtm[i].phase[:]
        m.add_dataset('phases_atm', pha, ['IPHA', 'SCAT_ANGLE_ATM', 'NPSTK'])

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



def calculF(phases, N, DEPO):

    """
    Compute CDF of scattering phase matrices

    Arguments :
        - phases : list of phase functions
        - N : Number of discrete values of the phase function
    --------------------------------------------------
    Returns :
        - phase_H : cumulative distribution of the phase function and phase function
    """
    nmax, nphases = 0, 1

    # define the number of phases functions and the maximal number of angles describing each of them
    for idx, phase in enumerate(phases):
        if idx > 0:
            if phase.N>nmax:
                nmax = phase.N
        nphases += 1

    # Initialize the cumulative distribution function
    phase_H = np.zeros((nphases, N), dtype=type_Phase, order='C')


    # Set Rayleigh phase function
    idx = 0
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
    cTh2 = cTh*cTh
    theta = np.arccos(cTh)
    cThLE = np.cos(thetaLE)
    cTh2LE = cThLE*cThLE
    T_demi = (3.0/2.0)
    P11 = T_demi*(DELTA+DELTA_PRIM)
    P12 = T_demi*DELTA_PRIM
    P33bis = T_demi*DELTA
    P44bis = P33bis*DELTA_SECO

    # parameters equally spaced in scattering probabiliyu
    phase_H['p_P11'][idx, :] = P11
    phase_H['p_P12'][idx, :] = P12
    phase_H['p_P22'][idx, :] = T_demi*(DELTA*cTh2[:] + DELTA_PRIM)
    phase_H['p_P33'][idx, :] = P33bis*cTh[:] # U
    phase_H['p_P44'][idx, :] = P44bis*cTh[:] # V
    phase_H['p_ang'][idx, :] = theta[:] # angle

    phase_H['a_P11'][idx, :] = P11
    phase_H['a_P12'][idx, :] = P12
    phase_H['a_P22'][idx, :] = T_demi*(DELTA*cTh2LE[:] + DELTA_PRIM) 
    phase_H['a_P33'][idx, :] = P33bis*cThLE[:]  # U
    phase_H['a_P44'][idx, :] = P44bis*cThLE[:]  # V

    for idx, phase in enumerate(phases):
        # conversion en gradiant
        angles = phase.ang_in_rad()

        scum = [0]
        dtheta = np.diff(angles)
        pm = phase.phase[:, 1] + phase.phase[:, 0]
        sin = np.sin(angles)
        tmp = dtheta * ((sin[:-1] * pm[:-1] + sin[1:] * pm[1:]) / 3. +  (sin[:-1] * pm[1:] + sin[1:] * pm[:-1])/6. )* np.pi * 2.
        scum = np.append(scum,tmp)
        scum = np.cumsum(scum)
        scum /= scum[phase.N-1]

        # probability between 0 and 1
        z = (np.arange(N, dtype='float64')+1)/N
        angN = (np.arange(N, dtype='float64'))/(N-1)*np.pi
        f1 = interp1d(phase.ang_in_rad(), phase.phase[:,1])
        f2 = interp1d(phase.ang_in_rad(), phase.phase[:,0])
        f3 = interp1d(phase.ang_in_rad(), phase.phase[:,2])
        f4 = interp1d(phase.ang_in_rad(), phase.phase[:,3])

        # parameters equally spaced in scattering probabiliyu
        phase_H['p_P11'][idx+1, :] = interp1d(scum, phase.phase[:,1])(z)  # I par P11
        phase_H['p_P22'][idx+1, :] = interp1d(scum, phase.phase[:,0])(z)  # I per P22
        phase_H['p_P33'][idx+1, :] = interp1d(scum, phase.phase[:,2])(z)  # U P33
        phase_H['p_P43'][idx+1, :] = interp1d(scum, phase.phase[:,3])(z)  # V P43
        phase_H['p_ang'][idx+1, :] = interp1d(scum, phase.ang_in_rad())(z) # angle

        # parameters equally spaced in scattering angle [0, 180]
        phase_H['a_P11'][idx+1, :] = f1(angN)  # I par P11
        phase_H['a_P22'][idx+1, :] = f2(angN)  # I per P22
        phase_H['a_P33'][idx+1, :] = f3(angN)  # U P33
        phase_H['a_P43'][idx+1, :] = f4(angN)  # V P43

    return to_gpu(phase_H)


def InitConst(surf, env, NATM, NOCE, mod,
                   NBPHOTONS, NBLOOP, THVDEG, DEPO,
                   XBLOCK, XGRID,NLAM, SIM, NF,
                   NBTHETA, NBPHI, OUTPUT_LAYERS,
                   RTER, LE, FLUX, NLVL, NPSTK, NWLPROBA) :

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

def get_profAtm(wl, atm):

    """
    get the atmospheric profile, the altitude of the top of Atmosphere, the number of layers of the atmosphere

    Arguments :
        - wl : wavelength
        - atm : Profile object
         default None (no atmosphere)
    -----------------------------------------------------------------------------------------------------------
    Returns :
        - phasesAtm : Atmospheric phase functions
        - nprofilesAtm : List of atmospheric profiles set contiguously
        - NATM : Number of layers of the atmosphere
        - HATM : Altitude of the Top of Atmosphere

    """

    if atm is not None:
        # write the profile
        if isinstance(wl, (float, int, REPTRAN_IBAND, KDIS_IBAND)):
            wl = [wl]
        profilesAtm, phasesAtm = atm.calc_bands(wl)
        # the altitude is get only by the first atmospheric profile
        # remove the key Altitude from the list of keys
        NATM = len(profilesAtm[0])-1
        HATM = profilesAtm[0]['ALT'][0]

        #
        # reformat
        #
        shp = (len(wl), NATM+1)
        prof_atm = np.zeros(shp, dtype=type_Profile)
        taumol = np.zeros(shp, dtype=np.float32)
        tauaer = np.zeros(shp, dtype=np.float32)
        prof_atm['z'][0,:] = profilesAtm[0]['ALT']    # only for first wavelength
        prof_atm['z'][1:,:] = -999.                 # other wavelengths are NaN
        for i, profile in enumerate(profilesAtm):
            prof_atm['tau'][i,:] = profile['H']
            prof_atm['pmol'][i,:] = profile['YDEL']
            prof_atm['ssa'][i,:] = profile['XSSA']
            prof_atm['abs'][i,:] = profile['percent_abs']
            prof_atm['iphase'][i,:] = profile['IPHA']
            taumol[i,:] = profile['hmol']
            tauaer[i,:] = profile['haer']
    else:
        # no atmosphere
        phasesAtm = []
        NATM = 0
        HATM = 0
        taumol = None
        tauaer = None

        prof_atm = np.zeros(1, dtype=type_Profile)

    return prof_atm, phasesAtm, NATM, HATM, taumol, tauaer

def get_profOc(wl, water, NLAM):

    """
    get the oceanic profile, the altitude of the top of Atmosphere, the number of layers of the atmosphere
    Arguments :
        - wl : wavelengths
        - water : Profile object
            default None (no atmosphere)
        - D : Dictionary containing all the parameters required to launch the simulation by the kernel
        - NLAM : Number of wavelengths
    -------------------------------------------------------------------------------------------------------
    Returns :
        - phasesOc : Oceanic phase functions
        - nprofilesOc : List of oceanic profiles set contiguously
        - NOCE : Number of layers of the ocean

    """

    if water is None:
            # use default water values
        phasesOc = []
        NOCE = 0

        prof_oc = np.zeros(1, dtype=type_Profile)
    else:
        if isinstance(wl, (float, int, REPTRAN_IBAND, KDIS_IBAND)):
            wl = [wl]
        profilesOc, phasesOc = water.calc_bands(wl)
        NOCE = 1
        if water.depth is None:
            DEPTH = 10000.
        else:
            DEPTH = water.depth
        #
        # switch to new format
        #
        shp = (len(wl), 2)
        prof_oc = np.zeros(shp, dtype=type_Profile)
        prof_oc['z'][0,:] = 0.      # only for first wavelength
        prof_oc['z'][1:,:] = -999.  # otherwise -999.
        prof_oc['pmol'][:,:] = 0.    # use only CDF
        prof_oc['abs'][:,:] = 0.   # no absorption
        for ilam in xrange(0, NLAM):
            prof_oc['tau'][ilam, 0] = 0.
            prof_oc['ssa'][ilam, 0] = 1.
            prof_oc['iphase'][ilam, 0] = 0

            prof_oc['tau'][ilam, 1] = - (profilesOc[ilam][1]+profilesOc[ilam][0]) * DEPTH
            prof_oc['ssa'][ilam, 1] = profilesOc[ilam][1]/(profilesOc[ilam][1]+profilesOc[ilam][0])
            prof_oc['iphase'][ilam, 1] = profilesOc[ilam][2]


    return prof_oc, phasesOc, NOCE


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
                NLAM, double , kern, p, X0, le, spectrum,
                prof_atm, prof_oc, wl_proba_icdf, SEED, stdev):
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
        - kern : kernel launching the transfert radiative simulation
        - p: progress bar object
        - X0: initial coordinates of the photon entering the atmosphere
    --------------------------------------------------------------
    Returns :
        - nbPhotonsTot : Total number of photons processed   # FIXME description
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

    # RNG status and configuration
    philox_data = np.zeros(XBLOCK*XGRID+1, dtype='uint32')
    philox_data[0] = SEED
    philox_data = to_gpu(philox_data)

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
                prof_atm, prof_oc, wl_proba_icdf, philox_data,
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


def impactInit(pp, Hatm, NATM, NLAM, prof_atm, THVDEG, Rter):
    """
    Calculate the coordinates of the entry point in the atmosphere
    and direct transmission of the atmosphere

    Arguments :
        - pp: plane parallel/spherical mode
        - Hatm : Altitude of the Top of Atmosphere
        - NATM : Number of layers of the atmosphere
        - NLAM : Number of wavelengths
        - ALT : Altitude profile of the atmosphere
        - H : optical thickness profile of the atmosphere
        - THVDEG : View Zenith Angle in degree
        - Rter: earth radius

    Returns :
        - (x0, y0, z0) : cartesian coordinates
    """

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
            for ilam in xrange(NLAM):
                tautot[ilam] = prof_atm['tau'][ilam, NATM]/np.cos(THVDEG*pi/180.)
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
        for i in xrange(1, NATM+1):
            # V is the direction vector, X is the position vector, D is the
            # distance to the next layer and R is the position vector at the
            # next layer
            # we have: R = X + V.D
            # R² = X² + (V.D)² + 2XVD
            # where R is Rter+ALT[i]
            # solve for D:
            delta = 4.*(vx*xph + vy*yph + vz*zph)**2 - 4*((xph**2 + yph**2 + zph**2) - (Rter + prof_atm['z'][0,i])**2)

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

            for ilam in xrange(NLAM):
                # optical thickness of the layer in vertical direction
                hlay0 = abs(prof_atm['tau'][ilam, i] - prof_atm['tau'][ilam, i - 1])

                # thickness of the layer
                D0 = abs(prof_atm['z'][0,i-1] - prof_atm['z'][0,i])

                # optical thickness of the layer at current wavelength
                hlay = hlay0*D/D0

                # cumulative optical thickness
                tautot[ilam] += hlay

    return x0, y0, z0, np.exp(-tautot)

