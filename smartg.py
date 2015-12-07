#!/usr/bin/env python
# encoding: utf-8


'''
SMART-G
Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU
'''



import numpy as np
from tools.gpustruct import GPUStruct
import time
from datetime import datetime
from numpy import pi
from tools.profile.profil import AeroOPAC, Profile, REPTRAN, REPTRAN_IBAND, CloudOPAC
from tools.water.iop_spm import IOP_SPM
from tools.water.iop_mm import IOP_MM
from tools.water.iop_AOS_water import IOP_AOS_WATER
from tools.water.phase_functions import PhaseFunction
from os.path import dirname, realpath, join, basename, exists
from warnings import warn
from tools.progress import Progress
from tools.luts import merge, read_lut_hdf, read_mlut_hdf, LUT, MLUT
from scipy.interpolate import interp1d
import subprocess
from collections import OrderedDict


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


def smartg_thr(*args, **kwargs):
    '''
    A wrapper around smartg running it in a thread
    This prevents blocking the context in an interactive environment

    Args and returns: identical as smartg
    '''
    from multiprocessing import Process, Queue
    q = Queue()  # queue to store the result
    qpro = Queue()  # queue to store the progress

    if ('progress' not in kwargs):
        kwargs['progress'] = True  # default value
    if kwargs['progress']:
        kwargs['progress'] = qpro

    p = Process(target=_smartg_thread, args=(q, qpro, args, kwargs))
    p.start()
    print 'Started thread', p.pid
    pro = None

    while q.empty():
        if qpro.empty():
            time.sleep(0.1)
        else:
            if pro is None:
                pro = Progress(qpro.get(), True)
            else:
                ret = qpro.get()
                if isinstance(ret, tuple):
                    pro.update(ret[0], ret[1])

    res = q.get()
    if isinstance(res, Exception):
        raise res

    while not qpro.empty():
        ret = qpro.get()
        if isinstance(ret, str):
            pro.finish(ret)

    return res


def _smartg_thread(q, qpro, args, kwargs):
    '''
    the thread function calling smartg
    (called by smartg_thr)
    '''
    try:
        m = smartg(*args, **kwargs)
        q.put(m)
    except Exception as ex:
        # in case of exception, put the exception in the queue instead of the
        # result
        q.put(ex)


def smartg(wl, pp=True,
           atm=None, surf=None, water=None, env=None,
           NBPHOTONS=1e9, DEPO=0.0279, THVDEG=0., SEED=-1,
           RTER=6371.,
           NBTHETA=45, NBPHI=90,
           NFAER=1000000, NFOCE=1000000,
           OUTPUT_LAYERS=0, XBLOCK=256, XGRID=256,
           NBLOOP=None, progress=True, debug=False,
           alt_move=False, debug_photon=False):
        '''
        Run a SMART-G simulation

        Arguments:

            - wl: wavelength in nm (float)
                  or: a list/array of wavelengths
                  or: a list of IBANDs (reptran)

            - pp:
                True: use plane parallel geometry (default)
                False: use spherical shell geometry

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

            - NBTHETA: number of zenith angles in output

            - NBPHI: number of azimuth angles in output

            - NFAER: number of discretization of the inversed aerosol phase functions

            - NFOCE: number of discretization of the inversed ocean phase functions

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

            - debug: set to True to activate debug mode (optional stdout if problems are detected)
                NOTE: compilation flag, not available if the kernel is provided as a binary

            - alt_move: set to true to activate the alternate move scheme in move_sp.
                NOTE: compilation flag, not available if the kernel is provided as a binary 

            - debug_photon: activate the display of photon path for the thread 0
                NOTE: compilation flag, not available if the kernel is provided as a binary 


        Return value:
        ------------

        Returns a MLUT object containing:
            - the polarized reflectance (I,Q,U,V) at the different layers
            - the number of photons (N) received at each layer
            - the profiles and phase functions
            - attributes

        Example:
            M = smartg(wl=400., NBPHOTONS=1e7, atm=Profile('afglt'))
            M['I_up (TOA)'][:,:] contains the top of atmosphere reflectance
        '''
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        from pycuda.driver import module_from_buffer

        #
        # initialization
        #

        if NBPHI%2 == 1:
            warn('Odd number of azimuth')

        if NBLOOP is None:
            NBLOOP = min(NBPHOTONS/30, 1e8)

        # number of output levels
        # warning! should be identical to the value defined in communs.h
        NLVL = 5

        # number of Stokes parameters of the radiation field + 1 for number of photons
        # warning! still hardcoded in device.cu and profil.py (FIXME)
#        NPSTK = 4
        NPSTK = 5

        attrs = OrderedDict()
        attrs.update({'device': pycuda.autoinit.device.name()})
        attrs.update({'processing started at': datetime.now()})
        attrs.update({'VZA': THVDEG})
        attrs.update({'MODE': {True: 'PPA', False: 'SSA'}[pp]})
        attrs.update({'XBLOCK': XBLOCK})
        attrs.update({'XGRID': XGRID})
        attrs.update(get_git_attrs())

        if SEED == -1:
            # SEED is based on clock
            SEED = np.uint32((datetime.now()
                - datetime.utcfromtimestamp(0)).total_seconds()*1000)

        assert isinstance(wl, (float, list, np.ndarray))
        if isinstance(wl, list):
            if (False not in map(lambda x: isinstance(x, REPTRAN_IBAND), wl)):
                # wl is a list of REPTRAN_IBANDs
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

        nprofilesAtm, phasesAtm, NATM, HATM = get_profAtm(wl,atm)

        # computation of the impact point
        x0, y0, z0, tabTransDir = impactInit(pp, HATM, NATM, NLAM, nprofilesAtm['ALT'], nprofilesAtm['H'], THVDEG, RTER)


        #
        # surface
        #
        if surf is None:
            # default surface parameters
            surf = FlatSurface()

        #
        # ocean profile
        # get the phase function and oceanic profile
        nprofilesOc, phasesOc, NOCE = get_profOc(wavelengths, water, NLAM)

        #
        # environment effect
        #
        if env is None:
            # default values (no environment effect)
            env = Environment()

        #
        # albedo
        #
        if 'SURFALB' in surf.dict:
            surf_alb = surf.dict['SURFALB']
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

        # compilation option
        # list of compilation flag :
        #     - DRANDPHILOX4x32_7 : Utilisation du random Philox-4x32-7
        #     - DPROGRESSION : Calcul et affichage de la progression de la simulation
        #     - DSPHERIQUE : Calcul en sphérique
        #     - DDEBUG : Ajout de tests intermédiaires utilisés lors du débugage

        options = ['-DRANDPHILOX4x32_7','-DPROGRESSION']
        # options.extend(['-DPARAMETRES','-DPROGRESSION'])
        if not pp:
            options.append('-DSPHERIQUE')
        if debug:
            options.append('-DDEBUG')
        if alt_move:
            options.append('-DALT_MOVE')
        if debug_photon:
            options.append('-DDEBUG_PHOTON')

        #
        # compile or load the kernel
        #
        if exists(src_device):

            # load device.cu
            src_device_content = open(src_device).read()

            # kernel compilation
            mod = SourceModule(src_device_content,
                               nvcc='/usr/local/cuda/bin/nvcc',
                               options=options,
                               no_extern_c=True,
                               cache_dir='/tmp/',
                               include_dirs=[dir_src,
                                   join(dir_src, 'incRNGs/Random123/')])
        elif exists(binnames[pp]):
            # load existing binary
            print 'read binary', binnames[pp]
            mod = module_from_buffer(open(binnames[pp], 'rb').read())

        else:
            raise IOError('Could not find {} or {}.'.format(src_device, binnames[pp]))

        # get the kernel
        kern = mod.get_function('lancementKernelPy')

        # computation of the phase functions
        if(SIM == 0 or SIM == 2 or SIM == 3):
            if phasesOc != []:
                foce = calculF(phasesOc, NFOCE)
            else:
                foce = [0]
        else:
            foce = [0], [0]

        if(SIM == -2 or SIM == 1 or SIM == 2):
            if phasesAtm != []:
                faer = calculF(phasesAtm, NFAER)
            else:
                faer = [0]
        else:
            faer = [0]

        # write the input variables into data structures
        Tableau, Var, Init = InitSD(nprofilesAtm, nprofilesOc, NLAM,
                                NLVL, NPSTK, NBTHETA, NBPHI, faer, foce,
                                albedo, wavelengths, x0, y0, z0,
                                XBLOCK, XGRID, SEED, options)

        # initialization of the constants
        InitConstantes(surf, env, NATM, NOCE, mod,
                       NBPHOTONS, NBLOOP, THVDEG, DEPO,
                       XBLOCK, XGRID, NLAM, SIM, NFAER,
                       NFOCE, NBTHETA, NBPHI, OUTPUT_LAYERS, RTER)


        # Initialize the progress bar
        p = Progress(NBPHOTONS, progress)


        # Loop and kernel call
        (nbPhotonsTot, nbPhotonsTotInter, nbPhotonsTotInter,
                nbPhotonsSorTot, tabPhotonsTot) = loop_kernel(NBPHOTONS, Tableau, Var, Init,
                                                              NLVL, NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                                                              NLAM, options, kern, p)


        # finalization
        output = finalize(tabPhotonsTot, wavelengths, nbPhotonsTotInter,
                           OUTPUT_LAYERS, tabTransDir, SIM, attrs, nprofilesAtm, phasesAtm)

        p.finish('Done! (used {}) | '.format(attrs['device']) + afficheProgress(nbPhotonsTot, NBPHOTONS, nbPhotonsSorTot))

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


def calculOmega(NBTHETA, NBPHI):
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


def finalize(tabPhotonsTot2, wl, nbPhotonsTotInter, OUTPUT_LAYERS, tabTransDir, SIM, attrs, nprofilesAtm, phasesAtm):
    '''
    create and return the final output
    '''
    (NLVL,NPSTK,NLAM,NBTHETA,NBPHI) = tabPhotonsTot2.shape
    tabFinal = np.zeros_like(tabPhotonsTot2, dtype='float64')
    tabTh, tabPhi, tabOmega = calculOmega(NBTHETA, NBPHI)

    # normalization
    # (broadcast to dimensions (LVL, LAM, THETA, PHI))
    norm = 2.0 * tabOmega.reshape((1,1,-1,1)) * np.cos(tabTh).reshape((1,1,-1,1)) * nbPhotonsTotInter.reshape((1,-1,1,1))

    # I
    tabFinal[:,0,:,:,:] = (tabPhotonsTot2[:,0,:,:,:] + tabPhotonsTot2[:,1,:,:,:])/norm

    # Q
    tabFinal[:,1,:,:,:] = (tabPhotonsTot2[:,0,:,:,:] - tabPhotonsTot2[:,1,:,:,:])/norm

    # U
    tabFinal[:,2,:,:,:] = tabPhotonsTot2[:,2,:,:,:]/norm

    # V
    tabFinal[:,3,:,:,:] = tabPhotonsTot2[:,3,:,:,:]/norm

    # N
    tabFinal[:,4,:,:,:] = tabPhotonsTot2[:,4,:,:,:]


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

    # swapaxes : (th, phi) -> (phi, theta)
    m.add_dataset('I_up (TOA)', tabFinal.swapaxes(3,4)[UPTOA,0,ilam,:,:], axnames)
    m.add_dataset('Q_up (TOA)', tabFinal.swapaxes(3,4)[UPTOA,1,ilam,:,:], axnames)
    m.add_dataset('U_up (TOA)', tabFinal.swapaxes(3,4)[UPTOA,2,ilam,:,:], axnames)
    m.add_dataset('V_up (TOA)', tabFinal.swapaxes(3,4)[UPTOA,3,ilam,:,:], axnames)
    m.add_dataset('N_up (TOA)', tabFinal.swapaxes(3,4)[UPTOA,4,ilam,:,:], axnames)

    if OUTPUT_LAYERS & 1:
        m.add_dataset('I_down (0+)', tabFinal.swapaxes(3,4)[DOWN0P,0,ilam,:,:], axnames)
        m.add_dataset('Q_down (0+)', tabFinal.swapaxes(3,4)[DOWN0P,1,ilam,:,:], axnames)
        m.add_dataset('U_down (0+)', tabFinal.swapaxes(3,4)[DOWN0P,2,ilam,:,:], axnames)
        m.add_dataset('V_down (0+)', tabFinal.swapaxes(3,4)[DOWN0P,3,ilam,:,:], axnames)
        m.add_dataset('N_down (0+)', tabFinal.swapaxes(3,4)[DOWN0P,4,ilam,:,:], axnames)

        m.add_dataset('I_up (0-)', tabFinal.swapaxes(3,4)[UP0M,0,ilam,:,:], axnames)
        m.add_dataset('Q_up (0-)', tabFinal.swapaxes(3,4)[UP0M,1,ilam,:,:], axnames)
        m.add_dataset('U_up (0-)', tabFinal.swapaxes(3,4)[UP0M,2,ilam,:,:], axnames)
        m.add_dataset('V_up (0-)', tabFinal.swapaxes(3,4)[UP0M,3,ilam,:,:], axnames)
        m.add_dataset('N_up (0-)', tabFinal.swapaxes(3,4)[UP0M,4,ilam,:,:], axnames)

    if OUTPUT_LAYERS & 2:
        m.add_dataset('I_down (0-)', tabFinal.swapaxes(3,4)[DOWN0M,0,ilam,:,:], axnames)
        m.add_dataset('Q_down (0-)', tabFinal.swapaxes(3,4)[DOWN0M,1,ilam,:,:], axnames)
        m.add_dataset('U_down (0-)', tabFinal.swapaxes(3,4)[DOWN0M,2,ilam,:,:], axnames)
        m.add_dataset('V_down (0-)', tabFinal.swapaxes(3,4)[DOWN0M,3,ilam,:,:], axnames)
        m.add_dataset('N_down (0-)', tabFinal.swapaxes(3,4)[DOWN0M,4,ilam,:,:], axnames)

        m.add_dataset('I_up (0+)', tabFinal.swapaxes(3,4)[UP0P,0,ilam,:,:], axnames)
        m.add_dataset('Q_up (0+)', tabFinal.swapaxes(3,4)[UP0P,1,ilam,:,:], axnames)
        m.add_dataset('U_up (0+)', tabFinal.swapaxes(3,4)[UP0P,2,ilam,:,:], axnames)
        m.add_dataset('V_up (0+)', tabFinal.swapaxes(3,4)[UP0P,3,ilam,:,:], axnames)
        m.add_dataset('N_up (0+)', tabFinal.swapaxes(3,4)[UP0P,4,ilam,:,:], axnames)

    # direct transmission
    if NLAM > 1:
        m.add_dataset('direct transmission', tabTransDir,
                axnames=['Wavelength'])
    else:
        m.set_attr('direct transmission', str(tabTransDir[0]))

    # write atmospheric profiles
    if SIM in [-2, 1, 2]:
        m.add_axis('ALT', nprofilesAtm['ALT'])
        for key in nprofilesAtm.keys():
            if key == 'ALT':
                continue
            if NLAM == 1:
                m.add_dataset(key, nprofilesAtm[key], ['ALT'])
            else:
                m.add_dataset(key, nprofilesAtm[key].reshape((NLAM, -1)), ['Wavelength', 'ALT'])

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


    # write attributes
    attrs['processing duration (s)'] = (datetime.now()
                                        - attrs['processing started at']).total_seconds()
    for k, v in attrs.items():
        m.set_attr(k, str(v))

    return m


def afficheProgress(nbPhotonsTot, NBPHOTONS, nbPhotonsSorTot):
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
    chaine += 'Launched %.2e photons (%3d%%)' % (nbPhotonsTot, pourcent)
    chaine += ' - received %.2e ' % (nbPhotonsSorTot);

    return chaine


def calculF(phases, N):

    """
    Compute CDF of scattering phase matrices

    Arguments :
        - phases : list of phase functions
        - N : Number of discrete values of the phase function
    --------------------------------------------------
    Returns :
        - phases_list : list of phase function set contiguously
        - phase_H : cumulative distribution of the phase functions
    """
    nmax, nphases, imax=0, 0, 0
    phases_list = []

    # define the number of phases functions and the maximal number of angles describing each of them
    for idx, phase in enumerate(phases):
        if phase.N>nmax:
            imax, nmax = idx, phase.N
        nphases += 1

    # Initialize the cumulative distribution function
    phase_H = np.zeros((nphases, N, 5), dtype=np.float32)

    for idx, phase in enumerate(phases):
        if idx != imax:
            # resizing the attributes of the phase object following the nmax
            phase.ang.resize(nmax)
            phase.phase.resize(nmax, 4)
        tmp = np.append(phase.ang, phase.phase)
        phases_list = np.append(phases_list, tmp)
        scum = np.zeros(phase.N)
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
        phase_H[idx, :, 4] = interp1d(scum, phase.ang_in_rad())(z)  # angle
        phase_H[idx, :, 0] = interp1d(scum, phase.phase[:,1])(z)  # I par P11
        phase_H[idx, :, 1] = interp1d(scum, phase.phase[:,0])(z)  # I per P22
        phase_H[idx, :, 2] = interp1d(scum, phase.phase[:,2])(z)  # U P33
        phase_H[idx, :, 3] = interp1d(scum, phase.phase[:,3])(z)  # V P43                                          

    return phase_H


def InitConstantes(surf, env, NATM, NOCE, mod,
                   NBPHOTONS, NBLOOP, THVDEG, DEPO,
                   XBLOCK, XGRID,NLAM, SIM, NFAER,
                   NFOCE, NBTHETA, NBPHI, OUTPUT_LAYERS, RTER) :

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

    import pycuda.driver as cuda
    D = {}

    # compute some needed constants
    THV = THVDEG * np.pi/180.
    STHV = np.sin(THV)
    CTHV = np.cos(THV)
    GAMAbis = DEPO / (2- DEPO)
    DELTAbis = (1.0 - GAMAbis) / (1.0 + 2.0 *GAMAbis)
    DELTA_PRIMbis = GAMAbis / (1.0 + 2.0*GAMAbis)
    DELTA_SECObis = (1.0 - 3.0*GAMAbis) / (1.0 - GAMAbis);
    BETAbis  = 3./2. * DELTA_PRIMbis
    ALPHAbis = 1./8. * DELTAbis
    Abis = 1. + BETAbis / (3.0 * ALPHAbis)
    ACUBEbis = Abis * Abis* Abis

    # converting the constants into arrays + dictionary update
    D.update(NBLOOP=np.array([NBLOOP], dtype=np.uint32))
    D.update(NOCE=np.array([NOCE], dtype=np.int32))
    D.update(OUTPUT_LAYERS=np.array([OUTPUT_LAYERS], dtype=np.uint32))
    D.update(NFAER=np.array([NFAER], dtype=np.uint32))
    D.update(NFOCE=np.array([NFOCE], dtype=np.uint32))
    D.update(NATM=np.array([NATM], dtype=np.int32))
    D.update(XBLOCK=np.array([XBLOCK], dtype=np.int32))
    D.update(YBLOCK=np.array([1], dtype=np.int32))
    D.update(XGRID=np.array([XGRID], dtype=np.int32))
    D.update(YGRID=np.array([1], dtype=np.int32))
    D.update(NBTHETA=np.array([NBTHETA], dtype=np.int32))
    D.update(NBPHI=np.array([NBPHI], dtype=np.int32))
    D.update(NLAM=np.array([NLAM], dtype=np.int32))
    D.update(SIM=np.array([SIM], dtype=np.int32))
    if surf != None:
        D.update(SUR=np.array(surf.dict['SUR'], dtype=np.int32))
        D.update(DIOPTRE=np.array(surf.dict['DIOPTRE'], dtype=np.int32))
        D.update(WINDSPEED=np.array(surf.dict['WINDSPEED'], dtype=np.float32))
        D.update(NH2O=np.array(surf.dict['NH2O'], dtype=np.float32))
    if env != None:
        D.update(ENV=np.array(env.dict['ENV'], dtype=np.int32))
        D.update(ENV_SIZE=np.array(env.dict['ENV_SIZE'], dtype=np.float32))
        D.update(X0=np.array(env.dict['X0'], dtype=np.float32))
        D.update(Y0=np.array(env.dict['Y0'], dtype=np.float32))
    D.update(STHV=np.array([STHV], dtype=np.float32))
    D.update(CTHV=np.array([CTHV], dtype=np.float32))
    D.update(DELTA=np.array([DELTAbis], dtype=np.float32))
    D.update(DELTA_PRIM=np.array([DELTA_PRIMbis], dtype=np.float32))
    D.update(DELTA_SECO=np.array([DELTA_SECObis], dtype=np.float32))
    D.update(BETA=np.array([BETAbis], dtype=np.float32))
    D.update(ALPHA=np.array([ALPHAbis], dtype=np.float32))
    D.update(A=np.array([Abis], dtype=np.float32))
    D.update(ACUBE=np.array([ACUBEbis], dtype=np.float32))

    # copy the constants into the device memory
    for key in D.keys():
        a,_ = mod.get_global('%sd'%key)
        cuda.memcpy_htod(a, D[key])

    cuda.memcpy_htod(mod.get_global('RTER')[0], np.array([RTER], dtype=np.float32))


def InitSD(nprofilesAtm, nprofilesOc, NLAM,
           NLVL, NPSTK, NBTHETA, NBPHI, faer,
           foce, albedo, wl, x0, y0,
           z0,XBLOCK, XGRID, SEED, options):

    """
    Initialize the principles data structures in python and send them the device memory

    Arguments:

        - nprofilesAtm: Atmospheric profile
        - nprofilesOc : Oceanic profile
        - NLAM: Number of wavelength
        - NLVL : Number of output levels
        - NPSTK : Number of stockes parameter + 1 for number of photons
        - NBTHETA : Number of intervals in zenith
        - NBPHI : Number of intervals in azimuth angle
        - faer : CDF of scattering phase matrices (Atmosphere)
        - foce : CDF of scattering phase matrices (Ocean)
        - albedo : Spectral Albedo
        - wl: wavelength
        - (x0,y0,z0) : Initial coordinates of the photon
        - XBLOCK: Block Size
        - XGRID : Grid Size
        - SEED: random number generator seed
        - options: compilation options

    -----------------------------------------------------------

    Returns the following GPUStruct Class:
        * Tableau : Class containing the arrays sent to the device
            Attributes :
                - nbPhotonsInter : number of photons injected by interval of NLAM
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
                - lambda : wavelength
                - z : altitudes level in the atmosphere
            optional:
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
                - x0,y0,z0 : cartesian coordinates of the photon at the initialization
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
    tmp = [(np.uint64, '*nbPhotonsInter', np.zeros(NLAM, dtype=np.uint64)),
           (np.float32, '*tabPhotons', np.zeros(NLVL * NPSTK * NBTHETA * NBPHI * NLAM, dtype=np.float32)),
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

    if '-DRANDPHILOX4x32_7' in options:
        tmp += [(np.uint32, '*etat', np.zeros(XBLOCK*1*XGRID*1, dtype=np.uint32)), (np.uint32, 'config', SEED)]


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

    nprofilesOc = {}
    nprofilesOc['HO'], nprofilesOc['SSO'], nprofilesOc['IPO'] = np.zeros(NLAM*2, dtype=np.float32), np.zeros(NLAM*2, dtype=np.float32), np.zeros(NLAM*2, dtype=np.float32)
    if water is None:
            # use default water values
        nprofilesOc['HO'], nprofilesOc['SSO'], nprofilesOc['IPO'],phasesOc = [0], [0], [0], []
        NOCE = 0
    else:
        if isinstance(wl, (float, int, REPTRAN_IBAND)):
            wl = [wl]
        profilesOc, phasesOc = water.calc_bands(wl)
        for ilam in xrange(0, NLAM):
            nprofilesOc['HO'][ilam*2] = 0
            nprofilesOc['SSO'][ilam*2] = 1
            nprofilesOc['IPO'][ilam*2] = 0
            nprofilesOc['HO'][ilam*2+1] = -1.e10
            nprofilesOc['SSO'][ilam*2+1] = profilesOc[ilam][1]/(profilesOc[ilam][1]+profilesOc[ilam][0])
            nprofilesOc['IPO'][ilam*2+1] = profilesOc[ilam][2]
            # parametrer les indices
        NOCE = 1

    return nprofilesOc, phasesOc, NOCE


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


def loop_kernel(NBPHOTONS, Tableau, Var, Init, NLVL,
                NPSTK, XBLOCK, XGRID, NBTHETA, NBPHI,
                NLAM, options , kern, p):
    """
    launch the kernel several time until the targeted number of photons injected is reached

    Arguments:
        - NBPHOTONS : Number of photons injected
        - Tableau : Class containing the arrays sent to the device
        - Var : Class containing the variables sent to the device
        - Init : Class containing the initial parameters of the photon
        - NLVL : Number of output levels
        - NPSTK : Number of Stokes parameters + 1 for number of photons
        - BLOCK : Block dimension
        - XGRID : Grid dimension
        - NBTHETA : Number of intervals in zenith
        - NLAM : Number of wavelengths
        - options : compilation options
        - kern : kernel launching the transfert radiative simulation
        - p: progress bar object
    --------------------------------------------------------------
    Returns :
        - nbPhotonsTot : Total number of photons processed
        - nbPhotonsTotInter : Total number of photons processed by interval
        - nbPhotonsSorTot : Total number of outgoing photons
        - tabPhotonsTot : Total weight of all outgoing photons

    """

    # Initialize of the parameters
    nbPhotonsTot = 0
    nbPhotonsTotInter = np.zeros(NLAM, dtype=np.uint64)
    nbPhotonsSorTot = 0
    tabPhotonsTot = np.zeros((NLVL,NPSTK,NLAM,NBTHETA,NBPHI), dtype=np.float32)

    # skip List used to avoid transfering arrays already sent into the device
    skipTableau = ['faer', 'foce', 'ho', 'sso', 'ipo', 'h', 'pMol', 'ssa', 'abs', 'ip', 'alb', 'lambda', 'z']
    skipVar = ['erreurtheta', 'erreurpoids', 'nThreadsActive', 'nbThreads', 'erreurvxy', 'erreurvy', 'erreurcase']

    while(nbPhotonsTot < NBPHOTONS):

        Tableau.tabPhotons = np.zeros(NLVL*NPSTK*NBTHETA * NBPHI * NLAM, dtype=np.float32)
        Tableau.nbPhotonsInter = np.zeros(NLAM, dtype=np.int32)
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
        tabPhotonsTot += Tableau.tabPhotons.reshape(tabPhotonsTot.shape)

        for ilam in xrange(0, NLAM):
            nbPhotonsTotInter[ilam] += Tableau.nbPhotonsInter[ilam]

        if '-DPROGRESSION' in options:
            nbPhotonsSorTot += Var.nbPhotonsSor;

        # update of the progression Bar
        p.update(nbPhotonsTot, afficheProgress(nbPhotonsTot, NBPHOTONS, nbPhotonsSorTot))

    return nbPhotonsTot, nbPhotonsTotInter , nbPhotonsTotInter, nbPhotonsSorTot, tabPhotonsTot


def impactInit(pp, Hatm, NATM, NLAM, ALT, H, THVDEG, Rter):
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
                tautot[ilam] = H[NATM+ilam*(NATM+1)]/np.cos(THVDEG*pi/180.)
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
            delta = 4.*(vx*xph + vy*yph + vz*zph)**2 - 4*((xph**2 + yph**2 + zph**2) - (Rter + ALT[i])**2)

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
                hlay0 = abs(H[i + ilam*(NATM+1)] - H[i - 1 + ilam*(NATM+1)])

                # thickness of the layer
                D0 = abs(ALT[i-1] - ALT[i])

                # optical thickness of the layer at current wavelength
                hlay = hlay0*D/D0

                # cumulative optical thickness
                tautot[ilam] += hlay

    return x0, y0, z0, np.exp(-tautot)

