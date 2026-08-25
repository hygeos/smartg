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


def _albedo_str(alb):
    """Compact display of a spectral albedo model."""
    if isinstance(alb, AlbedoCst):
        return str(alb.alb)
    return type(alb).__name__


class FlatSurface(object):
    """
    Definition of a flat sea surface

    Parameters
    ----------
    sur : int, optional
        The processes at the surface dioptre. 2 choices:
            - 1 -> Force reflection
            - 3 -> Reflection and transmission
    nh2o : float, optional
        The relative refarctive index air/water
    """
    def __init__(self, sur=3, nh2o=1.33):
        self.dict = {
                'SUR': sur,
                'DIOPTRE': 0,
                'WINDSPEED': -999.,
                'NH2O': nh2o,
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
    wind : float, optional
        The wind speed (m/s)
    sur : int, optional
        The processes at the surface dioptre. 2 choices:
            - 1 -> Force reflection
            - 3 -> Reflection and transmission
    nh2o : float, optional
        The relative refarctive index air/water
    wave_shadow : bool, optional
        Include wave shadowing effect. Default False.
    brdf : bool, optional
        Replace slope sampling by Cox & Munk BRDF, no ocean, just reflection
    single : bool, optional
        Deactivate multiple reflections/refractions at the interface. Default False.
    """
    def __init__(self, wind=5., sur=3, nh2o=1.33, wave_shadow=False, brdf=False, single=False):

        self.dict = {
                'SUR': sur if not brdf else 1,
                'DIOPTRE': 1,
                'WINDSPEED': wind,
                'NH2O': nh2o,
                'WAVE_SHADOW': 1 if wave_shadow else 0,
                'BRDF': 1 if brdf else 0,
                'SINGLE': 1 if single else 0,
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
    alb : AlbedoCst, | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap, optional
        The albedo spectral model.
    """
    def __init__(self, alb=AlbedoCst(0.5)):

        if (not isinstance(alb, (AlbedoCst, AlbedoSpeclib, AlbedoSpectrum, AlbedoMap))):
            raise ValueError('The parameter alb must be one of the following objects: AlbedoCst, ' +
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
        self.alb = alb
    def __str__(self):
        return 'LAMBSUR-ALB={}'.format(_albedo_str(self.alb))


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

        kp_bis = [AlbedoCst(0.5), AlbedoCst(0.0), AlbedoCst(0.0)]
        if kp is not None:
            warn_message = "\nThe use of parameter `kp` is deprecated as of SMART-G 1.1.0 " + \
                           "and will be removed in one of the next release.\n" + \
                           "Please use k0, k1p and k2p instead."
            warn(warn_message, DeprecationWarning)
            kp_bis = list(deepcopy(kp))

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
        self.kp = tuple(kp_bis) + (AlbedoCst(0.0),)
        self.alb= None
    def __str__(self):
        return 'RTLS-K={}'.format(
            '/'.join(_albedo_str(a) for a in self.kp[:3]))


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

        kp_bis = [AlbedoCst(0.5), AlbedoCst(0.0), AlbedoCst(0.0), AlbedoCst(0.0)]
        if kp is not None:
            warn_message = "\nThe use of parameter `kp` is deprecated as of SMART-G 1.1.0 " + \
                           "and will be removed in one of the next release.\n" + \
                           "Please use r0, k, bt and rc instead."
            warn(warn_message, DeprecationWarning)
            kp_bis = list(deepcopy(kp))

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
        self.kp = tuple(kp_bis)
        self.alb= None
    def __str__(self):
        return 'RPV-KP={}'.format(
            '/'.join(_albedo_str(a) for a in self.kp))


class Environment(object):
    """
    Stores the smartg parameters relative the the environment effect

    Parameters
    ----------
    env : int, optional
        The type of environment to consider. Possibilities are:

        * -1 -> Activate the disk mode, i.e., an horizontal disk centered at x0,y0 of radius `env_size`.
                The surface profile used inside the disk is a LambSurface(alb) object using the Environment
                parameter `alb`. The surface profile used outside the disk is the parameter surf of the
                Smartg method run.
        *  0 -> Deactivated. The default value.
        *  1 -> Same as -1 but the opposite.
        *  2 -> Activate the gaussian mode. A LambSurface(alb) object using the Environment parameter `alb`
                is used and corrected by a gaussian centred on x0,y0 with a maximum of ALB_SURF and an
                asymptotic value of alb of the environement. The square of the sigma is env_size.
                The form of the gaussian:

                    - :math:`exp(- ((x-x0)**2 + (y-y0)**2))/env_size)`
        *  3 -> alb map2D modulated by checkerboard spatial function
        *  4 -> Same as 1 but for a band defined as Abs(X) <= env_size, -4 for Abs(X)>= env_size
        *  5 -> 2D horizontal map of albedos for the whole surface, need alb to be an AlbedoMap object
                in that case the surface keyword of Smartg run method is not unused

    env_size : float, optional
        Definitions:

            - The radius (in km) of the disk for env = -1 or 1.
            - The square (in km) of the sigma of the gaussian for env = 2
            - The size of the spatial pattern (in km) for env = 3
    x0 : float, optional
        The X origin position
    y0 : float, optional
        The Y origin position
    alb : AlbedoCst | AlbedoSpeclib | AlbedoSpectrum | AlbedoMap, optional
        The albedo spectral model
    nenv : int, optional
        In progress...
    nxenvmap : int, optional
        In progress...
    nyenvmap : int, optional
        In progress...

    """
    def __init__(self, env=0, env_size=1.e6, x0=0., y0=0., alb=AlbedoCst(0.0), nenv=1,
                nxenvmap=0, nyenvmap=0):
        self.dict = {
                'ENV': env,
                'ENV_SIZE': env_size,
                'X0': x0,
                'Y0': y0,
                }
        self.alb = alb
        self.nenv = nenv
        self.nxenvmap = nxenvmap
        self.nyenvmap = nyenvmap

    def __str__(self):
        return 'ENV={ENV_SIZE}-X={X0:.1f}-Y={Y0:.1f}'.format(**self.dict)
