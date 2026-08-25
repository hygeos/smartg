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
from typing import get_args
from warnings import warn

from smartg.albedo import AlbedoCst, AlbedoLike


def _albedo_str(alb: AlbedoLike) -> str:
    """Compact display of a spectral albedo model."""
    if isinstance(alb, AlbedoCst):
        return str(alb.alb)
    return type(alb).__name__


class FlatSurface:
    """
    Definition of a flat sea surface.

    Parameters
    ----------
    sur : int, optional
        The processes at the surface dioptre. 2 choices:

            - 1 -> Force reflection
            - 3 -> Reflection and transmission
    nh2o : float, optional
        The relative refractive index air/water.
    """

    def __init__(
        self,
        sur: int = 3,
        nh2o: float = 1.33,
    ) -> None:
        self.dict = {
            'SUR': sur,
            'DIOPTRE': 0,
            'WINDSPEED': -999.,
            'NH2O': nh2o,
            'WAVE_SHADOW': 0,
            'BRDF': 0,
            'SINGLE': 1,
        }
        self.alb: AlbedoLike | None = None
        self.kp: tuple[AlbedoLike, ...] | None = None

    def __str__(self) -> str:
        return 'FLATSURF-SUR={SUR}'.format(**self.dict)


class RoughSurface:
    """
    Definition of a roughened sea surface.

    Parameters
    ----------
    wind : float, optional
        The wind speed (m/s).
    sur : int, optional
        The processes at the surface dioptre. 2 choices:

            - 1 -> Force reflection
            - 3 -> Reflection and transmission
    nh2o : float, optional
        The relative refractive index air/water.
    wave_shadow : bool, optional
        Include wave shadowing effect. Default False.
    brdf : bool, optional
        Replace slope sampling by Cox & Munk BRDF, no ocean,
        just reflection.
    single : bool, optional
        Deactivate multiple reflections/refractions at the
        interface. Default False.
    """

    def __init__(
        self,
        wind: float = 5.,
        sur: int = 3,
        nh2o: float = 1.33,
        wave_shadow: bool = False,
        brdf: bool = False,
        single: bool = False,
    ) -> None:
        self.dict = {
            'SUR': sur if not brdf else 1,
            'DIOPTRE': 1,
            'WINDSPEED': wind,
            'NH2O': nh2o,
            'WAVE_SHADOW': 1 if wave_shadow else 0,
            'BRDF': 1 if brdf else 0,
            'SINGLE': 1 if single else 0,
        }
        self.alb: AlbedoLike | None = None
        self.kp: tuple[AlbedoLike, ...] | None = None

    def __str__(self) -> str:
        return ('ROUGHSUR={SUR}-WIND={WINDSPEED}-DI={DIOPTRE}'
                '-WAVE_SHADOW={WAVE_SHADOW}-BRDF={BRDF}'
                '-SINGLE={SINGLE}').format(**self.dict)


class LambSurface:
    """
    Definition of a lambertian reflector.

    Parameters
    ----------
    alb : AlbedoLike, optional
        The albedo spectral model. Default AlbedoCst(0.5).
    """

    def __init__(
        self,
        alb: AlbedoLike | None = None,
    ) -> None:
        if alb is None:
            alb = AlbedoCst(0.5)
        if not isinstance(alb, get_args(AlbedoLike)):
            raise ValueError(
                'The parameter alb must be one of the following '
                'objects: AlbedoCst, AlbedoSpeclib, AlbedoSpectrum '
                'or AlbedoMap.')
        self.dict = {
            'SUR': 1,
            'DIOPTRE': 3,
            'WINDSPEED': -999.,
            'NH2O': -999.,
            'WAVE_SHADOW': 0,
            'BRDF': 1,
            'SINGLE': 1,
        }
        self.alb: AlbedoLike = alb

    def __str__(self) -> str:
        return 'LAMBSUR-ALB={}'.format(_albedo_str(self.alb))


class RTLSSurface:
    """
    Definition of a Ross-Thick Li-Sparse reflector.

    Parameters
    ----------
    kp : None | tuple, optional
        The Ross-Thick Li-Sparse coefficients (deprecated, see
        notes). Form of the tuple:

        * k0 : AlbedoLike
            -> The spectral albedo of the isotropic (lambertian)
               kernel
        * k1p : AlbedoLike
            -> The relative weight of the F1 (geometric) kernel
               (=K1/K0)
        * k2p : AlbedoLike
            -> The relative weight of the F2 (volumetric) kernel
               (=K2/K0)
    k0 : None | AlbedoLike, optional
        The spectral albedo of the isotropic (lambertian) kernel.
        Default AlbedoCst(0.5).
    k1p : None | AlbedoLike, optional
        The relative weight of the F1 (geometric) kernel (=K1/K0).
        Default AlbedoCst(0.0).
    k2p : None | AlbedoLike, optional
        The relative weight of the F2 (volumetric) kernel
        (=K2/K0). Default AlbedoCst(0.0).

    Notes
    -----
    The parameter kp is deprecated. Use k0, k1p and k2p instead.
    Providing parameters k0, k1p and k2p with kp will circumvent
    kp values.
    """

    def __init__(
        self,
        kp: tuple[AlbedoLike, AlbedoLike, AlbedoLike] | None = None,
        k0: AlbedoLike | None = None,
        k1p: AlbedoLike | None = None,
        k2p: AlbedoLike | None = None,
    ) -> None:
        kp_bis: list[AlbedoLike] = [
            AlbedoCst(0.5), AlbedoCst(0.0), AlbedoCst(0.0)]
        if kp is not None:
            warn(
                '\nThe use of parameter `kp` is deprecated as of '
                'SMART-G 1.1.0 and will be removed in one of the '
                'next release.\n'
                'Please use k0, k1p and k2p instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            kp_bis = list(deepcopy(kp))

        if k0 is not None:
            kp_bis[0] = k0
        if k1p is not None:
            kp_bis[1] = k1p
        if k2p is not None:
            kp_bis[2] = k2p

        self.dict = {
            'SUR': 1,
            'DIOPTRE': 4,
            'WINDSPEED': -999.,
            'NH2O': -999.,
            'WAVE_SHADOW': 0,
            'BRDF': 1,
            'SINGLE': 1,
        }
        self.kp: tuple[AlbedoLike, ...] = (
            tuple(kp_bis) + (AlbedoCst(0.0),))
        self.alb: AlbedoLike | None = None

    def __str__(self) -> str:
        return 'RTLS-K={}'.format(
            '/'.join(_albedo_str(a) for a in self.kp[:3]))


class RPVSurface:
    """
    Definition of a RPV reflector.

    Parameters
    ----------
    kp : tuple, optional
        The RPV coefficients (deprecated, see notes). Form of the
        tuple:

        * r0 : AlbedoLike
            -> Normalization.
        * k : AlbedoLike
            -> Minnaert exponent.
        * bt : AlbedoLike
            -> Henyey-Greenstein asymetry parameter.
        * rc : AlbedoLike
            -> Hotspot parameter.
    r0 : None | AlbedoLike, optional
        Normalization. Default AlbedoCst(0.5).
    k : None | AlbedoLike, optional
        Minnaert exponent. Default AlbedoCst(0.0).
    bt : None | AlbedoLike, optional
        Henyey-Greenstein asymetry parameter. Default
        AlbedoCst(0.0).
    rc : None | AlbedoLike, optional
        Hotspot parameter. Default AlbedoCst(0.0).

    Notes
    -----
    The parameter kp is deprecated. Use r0, k, bt and rc instead.
    Providing parameters r0, k, bt and rc with kp will circumvent
    kp values.

    References
    ----------
    Rahman, H., M. M. Verstraete, and B. Pinty, (1993) Coupled
    surface-atmosphere reflectance (CSAR) model. 1. Model
    description and inversion on synthetic data, JGR, 98,
    20,779-20,789.
    """

    def __init__(
        self,
        kp: tuple[
            AlbedoLike, AlbedoLike, AlbedoLike, AlbedoLike,
        ] | None = None,
        r0: AlbedoLike | None = None,
        k: AlbedoLike | None = None,
        bt: AlbedoLike | None = None,
        rc: AlbedoLike | None = None,
    ) -> None:
        kp_bis: list[AlbedoLike] = [
            AlbedoCst(0.5), AlbedoCst(0.0), AlbedoCst(0.0),
            AlbedoCst(0.0)]
        if kp is not None:
            warn(
                '\nThe use of parameter `kp` is deprecated as of '
                'SMART-G 1.1.0 and will be removed in one of the '
                'next release.\n'
                'Please use r0, k, bt and rc instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            kp_bis = list(deepcopy(kp))

        if r0 is not None:
            kp_bis[0] = r0
        if k is not None:
            kp_bis[1] = k
        if bt is not None:
            kp_bis[2] = bt
        if rc is not None:
            kp_bis[3] = rc

        self.dict = {
            'SUR': 1,
            'DIOPTRE': 5,
            'WINDSPEED': -999.,
            'NH2O': -999.,
            'WAVE_SHADOW': 0,
            'BRDF': 1,
            'SINGLE': 1,
        }
        self.kp: tuple[AlbedoLike, ...] = tuple(kp_bis)
        self.alb: AlbedoLike | None = None

    def __str__(self) -> str:
        return 'RPV-KP={}'.format(
            '/'.join(_albedo_str(a) for a in self.kp))


class Environment:
    """
    Stores the smartg parameters relative to the environment
    effect.

    Parameters
    ----------
    env : int, optional
        The type of environment to consider. Possibilities are:

        * -1 -> Activate the disk mode, i.e., an horizontal disk
                centered at x0,y0 of radius `env_size`. The
                surface profile used inside the disk is a
                LambSurface(alb) object using the Environment
                parameter `alb`. The surface profile used outside
                the disk is the parameter surf of the Smartg
                method run.
        *  0 -> Deactivated. The default value.
        *  1 -> Same as -1 but the opposite.
        *  2 -> Activate the gaussian mode. A LambSurface(alb)
                object using the Environment parameter `alb` is
                used and corrected by a gaussian centred on x0,y0
                with a maximum of ALB_SURF and an asymptotic
                value of alb of the environment. The square of
                the sigma is env_size. The form of the gaussian:

                    - :math:`exp(- ((x-x0)**2 + (y-y0)**2))/env_size)`
        *  3 -> alb map2D modulated by checkerboard spatial
                function
        *  4 -> Same as 1 but for a band defined as
                Abs(X) <= env_size, -4 for Abs(X) >= env_size
        *  5 -> 2D horizontal map of albedos for the whole
                surface, need alb to be an AlbedoMap object; in
                that case the surface keyword of the Smartg run
                method is not used

    env_size : float, optional
        Definitions:

            - The radius (in km) of the disk for env = -1 or 1.
            - The square (in km) of the sigma of the gaussian
              for env = 2
            - The size of the spatial pattern (in km) for env = 3
    x0 : float, optional
        The X origin position.
    y0 : float, optional
        The Y origin position.
    alb : AlbedoLike, optional
        The albedo spectral model. Default AlbedoCst(0.0).
    nenv : int, optional
        In progress...
    nxenvmap : int, optional
        In progress...
    nyenvmap : int, optional
        In progress...
    """

    def __init__(
        self,
        env: int = 0,
        env_size: float = 1.e6,
        x0: float = 0.,
        y0: float = 0.,
        alb: AlbedoLike | None = None,
        nenv: int = 1,
        nxenvmap: int = 0,
        nyenvmap: int = 0,
    ) -> None:
        if alb is None:
            alb = AlbedoCst(0.0)
        self.dict = {
            'ENV': env,
            'ENV_SIZE': env_size,
            'X0': x0,
            'Y0': y0,
        }
        self.alb: AlbedoLike = alb
        self.nenv = nenv
        self.nxenvmap = nxenvmap
        self.nyenvmap = nyenvmap

    def __str__(self) -> str:
        return 'ENV={ENV_SIZE}-X={X0:.1f}-Y={Y0:.1f}'.format(
            **self.dict)
