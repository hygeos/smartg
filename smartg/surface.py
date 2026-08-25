"""Surface and environment definitions for SMART-G simulations.

Key Classes
-----------
FlatSurface
    Definition of a flat sea surface.
RoughSurface
    Definition of a roughened sea surface (Cox & Munk).
LambSurface
    Definition of a lambertian reflector.
RTLSSurface
    Definition of a Ross-Thick Li-Sparse reflector.
RPVSurface
    Definition of a RPV reflector.
Environment
    Adjacency (environment) effect parameters.
"""

from __future__ import annotations

from copy import deepcopy
from warnings import warn

from smartg.albedo import AlbedoCst, AlbedoSpeclib, AlbedoSpectrum, AlbedoMap


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
        self.alb=None
        self.kp=None
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
    ALB : AlbedoCst, | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap, optional
        The albedo spectral model.
    """
    def __init__(self, ALB=AlbedoCst(0.5)):

        if (not isinstance(ALB, (AlbedoCst, AlbedoSpeclib, AlbedoSpectrum, AlbedoMap))):
            raise ValueError('The parameter ALB must be one of the following objects: AlbedoCst, ' +
                             'AlbedoSpeclib, AlbedoSpectrum or AlbedoMap.')
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

        * k0 : AlbedoCst | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap
            -> The spectral albedo of the isotropic (lambertian) kernel
        * k1p: AlbedoCst | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap
           -> The relative weight of the F1 (geometric) kernel (=K1/K0)
        * k2p: AlbedoCst | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap
           -> The relative weight of the F2 (volumetric) kernel (=K2/K0)
    k0 : None | AlbedoCst | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap, optional
        The spectral albedo of the isotropic (lambertian) kernel. Default AlbedoCst(0.5).
    k1p : None | AlbedoCst | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap, optional
        The relative weight of the F1 (geometric) kernel (=K1/K0). Default AlbedoCst(0.5).
    k2p : None | AlbedoCst | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap, optional
        The relative weight of the F2 (volumetric) kernel (=K2/K0). Default AlbedoCst(0.5).

    Notes
    -----
    The parameter kp is depracated. Use k0, k1p and k2p instead. Providing parameters k0, k1p
    and k2p with kp will circumvent kp values.
    """
    def __init__(self, kp=None, k0=None, k1p=None, k2p=None):

        kp_bis = (AlbedoCst(0.5), AlbedoCst(0.0), AlbedoCst(0.0))
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
        self.kp = kp_bis+(AlbedoCst(0.0),)
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

        * r0 : AlbedoCst
            -> Normalization.
        * k : AlbedoCst
            -> Minnaert exponent.
        * bt : AlbedoCst
            -> Henyey-Greenstein asymetry parameter.
        * rc : AlbedoCst
            -> Hotspot parameter.

    r0 : None | AlbedoCst | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap, optional
       Normalization.
    k : None | AlbedoCst | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap, optional
        Minnaert exponent.
    bt : None | AlbedoCst | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap, optional
        Henyey-Greenstein asymetry parameter.
    rc : None | AlbedoCst | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap, optional
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

        kp_bis = (AlbedoCst(0.5), AlbedoCst(0.0), AlbedoCst(0.0), AlbedoCst(0.0))
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

        * -1 -> Activate the disk mode, i.e., an horizontal disk centered at X0,Y0 of radius `ENV_SIZE`.
                The surface profile used inside the disk is a LambSurface(ALB) object using the Environment
                parameter `ALB`. The surface profile used outside the disk is the parameter surf of the
                Smartg method run.
        *  0 -> Deactivated. The default value.
        *  1 -> Same as -1 but the opposite.
        *  2 -> Activate the gaussian mode. A LambSurface(ALB) object using the Environment parameter `ALB`
                is used and corrected by a gaussian centred on X0,Y0 with a maximum of ALB_SURF and an
                asymptotic value of ALB of the environement. The square of the sigma is ENV_SIZE.
                The form of the gaussian:

                    - :math:`exp(- ((x-X0)**2 + (y-Y0)**2))/ENV_SIZE)`
        *  3 -> ALB map2D modulated by checkerboard spatial function
        *  4 -> Same as 1 but for a band defined as Abs(X) <= ENV_SIZE, -4 for Abs(X)>= ENV_SIZE
        *  5 -> 2D horizontal map of albedos for the whole surface, need ALB to be ALbedo_map object
                in that case the surface keyword of Smartg run method is not unused

    ENV_SIZE : float, optional
        Definitions:

            - The radius (in km) of the disk for ENV = -1 or 1.
            - The square (in km) of the sigma of the gaussian for ENV = 2
            - The size of the spatial pattern (in km) for ENV = 3
    X0 : float, optional
        The X origin position
    Y0 : float, optional
        The Y origin position
    ALB : AlbedoCst | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap, optional
        The albedo spectral model
    NENV : int, optional
        In progress...
    NXENVMAP : int, optional
        In progress...
    NYENVMAP : int, optional
        In progress...

    """
    def __init__(self, ENV=0, ENV_SIZE=1.e6, X0=0., Y0=0., ALB=AlbedoCst(0.0), NENV=1,
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
