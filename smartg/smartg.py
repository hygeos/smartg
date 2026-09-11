#!/usr/bin/env python
# encoding: utf-8


"""
SMART-G: Speed-up Monte carlo Advanced Radiative Transfer code using
GPU.

This module hosts the Smartg class, whose constructor compiles the CUDA
kernel with the requested options and whose run method performs the
radiative transfer simulations.

Key Classes
-----------
Smartg
    Main simulation class. The constructor compiles the CUDA kernel
    with the requested options (plane-parallel or spherical, forward
    or backward, ALIS, 3D objects, ...); the run method launches the
    Monte Carlo radiative transfer simulation and returns the results
    as an xarray Dataset.
StdevLim
    Adaptive stopping criterion for Smartg.run based on the standard
    deviation of the results.

Key Functions
-------------
multi_profiles
    Reorganize a list of atmosphere or ocean profiles into a single
    multi-profile table, so several profile configurations can be
    simulated in a single run.
reduce_diff
    Post-process ALIS finite-difference runs into sensitivities.
"""

import os
import numpy as np
from datetime import datetime, timezone
from numpy import pi
from smartg.atmosphere import Atmosphere, od2k, blackbody_radiance
from smartg.sensor import Sensor
from smartg.phase import THETA_GRID_KINDS, convert_phase_to_iparper
from smartg.phase import theta_grid as _make_theta_grid
from smartg.water import Water
from warnings import warn
from smartg.surface import Environment
from smartg.progress import progress as make_progress
from smartg.cdf import icdf_2d
from smartg.environ import modified_environ
from smartg.xarray import drop_axes
from scipy.interpolate import interp1d

# from scipy.integrate import simpson
import subprocess
from collections import OrderedDict
from pycuda import gpuarray
from pycuda.gpuarray import GPUArray, to_gpu
import pycuda.driver as cuda
from smartg.bandset import BandSet
from pycuda.compiler import SourceModule

# bellow necessary for object incorporation
from smartg.objects3d import (
    Mirror, Plane, Spheric, LambMirror, Matte, CusForward, CusBackward,
)
import xarray as xr
import geoclide as gc
import tempfile
from collections.abc import Callable, Sequence
from typing import cast

# pycuda ships no type stubs: gpuarray.zeros infers its dtype
# parameter as type[float64] from the default value, which flags
# every non-float64 call. The cast keeps the true signature
# usable without changing the runtime object.
gpuzeros = cast('Callable[..., GPUArray]', gpuarray.zeros)


# set up directories
from smartg.config import DIR_ROOT

DIR_SRC = DIR_ROOT / 'smartg' / 'src'
SRC_DEVICE = DIR_SRC / 'device.cu'
# constants definition
# (should match #defines in src/communs.h)
SPACE = 0
ATMOS = 1
SURF0P = 2  # surface (air side)
SURF0M = 3  # surface (water side)
ABSORBED = 4
NONE = 5
OCEAN = 6
SEAFLOOR = 7
OBJSURF = 8

# constants definition
# (should match #defines in src/communs.h)
UPTOA = 0
DOWN0P = 1
DOWN0M = 2
UP0P = 3
UP0M = 4
DOWNB = 5

#
MAX_NREF = 100

#
# type definitions (should match cuda struct definitions)
#
# mirrors struct Phase of communs.h: one phase matrix tabulated on the
# scattering angle grid of its medium. The angles a random walk draws
# its deflection from live in a separate table, TYPE_PCDF below, whose
# length is independent of this one.
TYPE_PHASE = [
    ('a_P11', 'float32'),  # \
    ('a_P12', 'float32'),  # |
    ('a_P22', 'float32'),  # | tabulated on the
    ('a_P33', 'float32'),  # | scattering angle grid
    ('a_P43', 'float32'),  # |
    ('a_P44', 'float32'),  # /
]

# the cumulative distribution of each phase function at the nodes of
# its angle grid, from 0 to 1, integrated exactly for the tabulated
# matrix: F11 linear in theta between nodes times the true sin(theta)
TYPE_PCDF = 'float32'

# mirrors struct AGrid of communs.h: align=True reproduces the 4 bytes
# of padding the compiler inserts before the 8-byte pointer
TYPE_AGRID = np.dtype(
    [
        ('n', 'uint32'),
        ('mode', 'int32'),
        ('log2n', 'int32'),
        ('ang', 'uint64'),
    ],
    align=True,
)

# mirrors struct PGrid of phase_grid.h: the two ints fill the 8 bytes
# before the pointer, so align=True changes nothing here
TYPE_PGRID = np.dtype(
    [
        ('n', 'uint32'),
        ('log2n', 'int32'),
        ('cdf', 'uint64'),
    ],
    align=True,
)

TYPE_SPECTRUM = np.dtype(
    [
        ('lambda', 'float32'),
        ('alb_surface', 'float32'),
        ('alb_seafloor', 'float32'),
        ('alb_env', 'float32'),
        ('k1p_surface', 'float32'),
        ('k2p_surface', 'float32'),
        ('k3p_surface', 'float32'),
        ('alb_envs', 'float32', MAX_NREF),
    ]
)

TYPE_ENV_MAP = np.dtype(
    [
        ('x', 'float32'),  # // x coordinate on the ground
        ('y', 'float32'),  # // y coordinate on the ground
        ('env_index', 'int32'),  # // environment index map
    ]
)

TYPE_PROFILE = [
    ('z', 'float32'),  # // altitude
    ('n', 'float32'),  # // refractive index
    ('T', 'float32'),  # // temperature
    ('OD', 'float32'),  # // cumulated extinction optical thickness (from top)
    # // cumulated scattering optical thickness (from top)
    ('OD_sca', 'float32'),
    # // cumulated absorption optical thickness (from top)
    ('OD_abs', 'float32'),
    ('pmol', 'float32'),  # // probability of pure Rayleigh scattering event
    ('ssa', 'float32'),  # // layer single scattering albedo
    ('pine', 'float32'),  # // layer fraction of inelastic scattering
    ('FQY1', 'float32'),  # // layer Fluorescence Quantum Yield of 1st specie
    ('iphase', 'int32'),  # // phase function index
]

TYPE_CELL = [
    ('iopt', 'int32'),  # // Optical scattering properties index
    ('iabs', 'int32'),  # // Optical absorbing properties index
    ('pminx', 'float32'),  # // Box point pmin.x
    ('pminy', 'float32'),  # // Box point pmin.y
    ('pminz', 'float32'),  # // Box point pmin.z
    ('pmaxx', 'float32'),  # // Box point pmax.x
    ('pmaxy', 'float32'),  # // Box point pmax.y
    ('pmaxz', 'float32'),  # // Box point pmax.z
    ('neighbour1', 'int32'),  # // neighbour box index +X
    ('neighbour2', 'int32'),  # // neighbour box index -X
    ('neighbour3', 'int32'),  # // neighbour box index +Y
    ('neighbour4', 'int32'),  # // neighbour box index -Y
    ('neighbour5', 'int32'),  # // neighbour box index +Z
    ('neighbour6', 'int32'),  # // neighbour box index -Z
]

TYPE_SENSOR = [
    ('pos_x', 'float32'),  # // X position of the sensor
    ('pos_y', 'float32'),  # // Y position of the sensor
    # // Z position of the sensor (from Earth's center in spherical,
    # from the ground in PP)
    ('pos_z', 'float32'),
    # // zenith angle of viewing direction (Zenith> 90 for downward
    # looking, <90 for upward, default Zenith)
    ('th_deg', 'float32'),
    ('ph_deg', 'float32'),  # // azimut angle of viewing direction
    # // localization (ATMOS=1, ...), see constant definitions in
    # communs.h
    ('loc', 'int32'),
    ('fov', 'float32'),  # // sensor FOV (degree)
    # // sensor type: Radiance (0), Planar flux (1), Spherical Flux (2),
    # default 0
    ('sensor_type', 'int32'),
    ('icell', 'int32'),  # // Box in which the sensor is
    # // Wavelength start index that the sensor 'sees' (default -1 :
    # all)
    ('ilam_0', 'int32'),
    # // Wavelength stop  index that the sensor 'sees' (default -1 :
    # all)
    ('ilam_1', 'int32'),
]

TYPE_SPECTRUM_OBJ = [
    ('reflectAV', 'float32'),
    ('reflectAR', 'float32'),
]

TYPE_IOBJECTS = [
    ('geo', 'int32'),  # 1 = sphere, 2 = plane, ...
    ('materialAV', 'int32'),  # 1 = LambMirror, 2 = Matte,
    ('materialAR', 'int32'),  # 3 = Mirror, ... (AV = avant, AR = Arriere)
    ('type', 'int32'),  # 1 = reflector, 2 = receiver
    ('reflectAV', 'float32'),  # reflectivity of materialAV
    ('reflectAR', 'float32'),  # reflectivity of materialAR
    ('roughAV', 'float32'),  # roughness of materialAV
    ('roughAR', 'float32'),  # roughness of materialAR
    ('shdAV', 'int32'),  # shadow option of materialAV, 0=false, 1=true
    ('shdAR', 'int32'),  # shadow option of materialAR
    ('nindAV', 'float32'),  # refractive index of materialAV
    ('nindAR', 'float32'),  # refractive index of materialAR
    ('distAV', 'int32'),  # distribution used for materialAV, 1=Beck, 2=GGX
    ('distAR', 'int32'),  # distribution used for materialAR
    ('p0x', 'float32'),  # \            \
    ('p0y', 'float32'),  # | point p0   \
    ('p0z', 'float32'),  # /              \
    #                 |
    ('p1x', 'float32'),  # \               |
    ('p1y', 'float32'),  # | point p1     |
    ('p1z', 'float32'),  # /               |
    #                 | Plane Object
    ('p2x', 'float32'),  # \               |
    ('p2y', 'float32'),  # | point p2     |
    ('p2z', 'float32'),  # /               |
    #                 |
    ('p3x', 'float32'),  # \              /
    ('p3y', 'float32'),  # | point p3   /
    ('p3z', 'float32'),  # /            /
    ('myRad', 'float32'),  # \
    ('z0', 'float32'),  # | Sperical Object
    ('z1', 'float32'),  # |
    ('phi', 'float32'),  # /
    ('mvRx', 'float32'),  # \
    ('mvRy', 'float32'),  # | Transformation type rotation
    ('mvRz', 'float32'),  # /
    ('rotOrder', 'int32'),  # rotation order: 1=XYZ; 2=XZY;...
    ('mvTx', 'float32'),  # \
    ('mvTy', 'float32'),  # | tranformation type translation
    ('mvTz', 'float32'),  # /
    ('nBx', 'float32'),  # \
    ('nBy', 'float32'),  # | normalBase de l'obj apres trans
    ('nBz', 'float32'),  # /
]

TYPE_GOBJ = [
    ('nObj', 'int32'),  # Number of objects in this group
    ('index', 'int32'),  # Index at the table of IObjects where
    # we start to fill the objects of the
    # group
    ('bPminx', 'float32'),  # \
    ('bPminy', 'float32'),  # |
    ('bPminz', 'float32'),  # | Bounding box of the group
    ('bPmaxx', 'float32'),  # |
    ('bPmaxy', 'float32'),  # |
    ('bPmaxz', 'float32'),  # /
]


class StdevLim(object):
    """
    Adaptive stopping criterion for Smartg.run based on the standard
    deviation of the results.

    Parameters
    ----------
    err_abs_min : float, optional
        The minimum absolute error. Stop the simulation if max abs error
        <= err_abs_min.
    err_rel_min : float, optional
        The minimum relative error in percentage.
    n_loop_min : int, optional
        The minimum kernel loop number before allowing to stop the
        simulation.
    stk : int, optional
        The Stokes component to consider. Choices are:

            * 0 -> I Stokes component (Default)
            * 1 -> Q Stokes component
            * 2 -> U Stokes component
            * 3 -> V Stokes component
    level : int, optional
        The level to use to analyse the standard deviations. Six
        choices:

            * 0 -> UPTOA (Default)
            * 1 -> DOWN0P
            * 2 -> DOWN0M
            * 3 -> UP0P
            * 4 -> UP0M
            * 5 -> DOWNB
    verbose : bool, optional
        Activate verbose mode to print the max absolute and relative
        errors at each kernel loop.
    fmt : str, optional
        The verbose print format for abs and rel max values.

    Notes
    -----
    For the moment, it does not work correctly with kdis and reptran.
    """

    def __init__(
        self,
        err_abs_min: float = 0.0,
        err_rel_min: float = 0.0,
        n_loop_min: int = 10,
        stk: int = 0,
        level: int = 0,
        verbose: bool = False,
        fmt: str = ".5e",
    ) -> None:

        self.dict = {
            'err_abs_min': err_abs_min,
            'err_rel_min': err_rel_min,
            'n_loop_min': n_loop_min,
            'stk': stk,
            'level': level,
            'verbose': verbose,
            'format': fmt,
        }

    def __str__(self) -> str:
        return self.dict.__str__()

    def __repr__(self) -> str:
        return 'Stdevlim dict: %s' % self.dict.__repr__()


class Smartg(object):
    """
    Initialization of the Smartg object

    Performs the compilation and loading of the kernel. This class is
    designed so split compilation and kernel loading from the code
    execution: in case of successive smartg executions, the kernel
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
        Use the ALIS method (Emde et al. 2010) for treating gaseous
        absorption and perturbed profile. The parameter alt_pp must be
        set to True.
    back : bool, optional
        Activate backward mode (else forward)
    bias : bool, optional
        Use the bias sampling scheme
    alt_pp : bool, optional
        Use a plane parallel propagation scheme following the photon at
        each layer. Increase the computational time, but allow the use
        of the ALIS method
    obj3d : bool, optional
        Allow 3D objects
    opt3d : bool, optional
        Activate the 3D atmosphere mode
    device : int | str, optional
        The device number / GPU to use. The GPU numbers can be obtained
        with the command `nvidia-smi`.
    sif : bool, optional
        Include the Sun Induced Fluorescence
    thermal : bool, optional
        Still in dev...
    rng : str, optional
        The pseudo-random number generator to use, only 2 choice:
        - PHILOX
        - CURAND_PHILOX
    cache_dir : str | Path, optional
        Path to the directory where the cache files are stored.
    keep_context : None | bool, optional
        Only in case autoinit is set to False. This parameter allows to
        keep or not the context after the use of the run method. By
        default (for the case autoinit=False) kill the context after the
        use of the run method.
    amf_variance : bool, optional, default=False
        Enable storage of the second moment of photon path lengths
        (⟨D²⟩) in tabDist, for Jensen bias correction of the mean-path
        AMF approximation. Requires ``alis=True`` since tabDist and per-
        photon cumulative distances (ph->cdist) are only available under
        the ALIS method. When enabled, the output ``cdist`` datasets
        have an ``iAMF`` axis of size 3 instead of 2:

        * iAMF=0: Σ(w · I)  — intensity-weighted count  (I = Stokes I =
          Ix+Iy)
        * iAMF=1: Σ(d · w · I) — intensity-weighted path length
        * iAMF=2: Σ(d² · w · I) — intensity-weighted squared path length

    cdist_wabs : bool, optional, default=False
        When True, include the absorption weight (Tabs = exp(-τ_abs)) in
        the intensity weight ``w_n`` used for the ``cdist`` (tabDist)
        accumulation.  By default the cdist moments are accumulated with
        ``w_n = weight · I`` (scattering weight only); with this option
        ``w_n = weight · Tabs · I``.  Requires ``alis=True``.

    nscl : int, optional, default=1
        Number of scatter classes for AMF decomposition (Approach 2).
        When ``nscl=1`` (default), no classification is performed and
        the output is identical to the standard AMF. When ``nscl>1``,
        photons are classified according to the ``scatter_classes``
        mode, and the ``cdist`` output datasets gain an extra ``iSCL``
        dimension of size ``nscl``. Typically set to ``NATM_ABS`` (one
        class per absorption layer) for ``'last_scattering_layer'``
        mode, or to the maximum expected scattering order for
        ``'scattering_order'`` mode, or to ``NATM_ABS * norders`` for
        ``'scattering_order_per_layer'`` mode. Requires ``alis=True``.

    scatter_classes : str, optional, default='last_scattering_layer'
        Mode for defining scatter classes when ``nscl>1``:

        * ``'none'`` — no classification (forces ``nscl=1``).
        * ``'last_scattering_layer'`` — photons are classified by the
          atmospheric layer in which their last scattering event
          occurred (original Approach 2 behaviour).
        * ``'scattering_order'`` — photons are classified by their total
          number of scattering events (``ph->nint``). Class index is
          ``min(nint, nscl) - 1``, so the last class collects all
          photons with ``nint >= nscl``.
        * ``'scattering_order_per_layer'`` — combined classification by
          both the last scattering layer and the scattering order. Class
          index is ``layer * norders + order``. Requires ``norders >=
          1``. Set ``nscl = NATM_ABS * norders``.

    norders : int, optional, default=1
        Number of scattering order bins per layer for the
        ``'scattering_order_per_layer'`` mode. The last bin collects all
        photons with ``nint >= norders``. Ignored for other modes.

    Raises
    ------
    ValueError
        If amf_variance=True is used without alis=True. If nscl>1 is
        used without alis=True. If scatter_classes is not one of the
        accepted values. If scatter_classes='scattering_order_per_layer'
        and norders < 1.
    """

    def __init__(
        self,
        pp: bool = True,
        debug: bool = False,
        autoinit: bool = True,
        verbose_photon: bool = False,
        double: bool = True,
        alis: bool = False,
        back: bool = False,
        bias: bool = True,
        alt_pp: bool = False,
        obj3d: bool = False,
        opt3d: bool = False,
        device: int | None = None,
        sif: bool = False,
        thermal: bool = False,
        rng: str = 'PHILOX',
        cache_dir: str | None = None,
        keep_context: bool | None = None,
        amf_variance: bool = False,
        cdist_wabs: bool = False,
        nscl: int = 1,
        scatter_classes: str | list = 'last_scattering_layer',
        norders: int = 1,
    ) -> None:
        assert not ((device is not None) and ('CUDA_DEVICE' in os.environ)), (
            "Can not use the 'device' option while the CUDA_DEVICE is set"
        )

        if device is not None:
            env_modif = {'CUDA_DEVICE': str(device)}
        else:
            env_modif = {}

        if not autoinit:
            self.keep_context = (
                keep_context if keep_context is not None else False
            )
        else:
            if keep_context is not None:
                raise ValueError(
                    "The parameter keep_context can be defined only "
                    "if 'autoinit' is False."
                )
            self.keep_context = True

        if cache_dir is None:
            cache_dir = tempfile.gettempdir()

        # Bind the name on every path: the imports inside the
        # branches below only bind it conditionally.
        import pycuda

        if autoinit:
            with modified_environ(**env_modif):
                try:
                    import pycuda.autoinit

                    self.ctx = pycuda.autoinit.context
                except Exception:
                    # In case cuda context has been manually popped
                    from importlib import reload, import_module

                    pycuda.autoinit = import_module('pycuda.autoinit')
                    reload(pycuda.autoinit)
                    self.ctx = pycuda.autoinit.context
        else:
            cuda.init()  # pyright: ignore[reportAttributeAccessIssue]
            from pycuda.tools import make_default_context

            self.ctx = make_default_context()

        self.autoinit = autoinit
        self.pp = pp
        self.double = double
        self.alis = alis
        if amf_variance and not alis:
            raise ValueError(
                'amf_variance=True requires alis=True '
                '(tabDist and ph->cdist need ALIS)'
            )
        self.amf_variance = amf_variance
        if cdist_wabs and not alis:
            raise ValueError(
                'cdist_wabs=True requires alis=True '
                '(tabDist and ph->cdist need ALIS)'
            )
        self.cdist_wabs = cdist_wabs
        _valid_scatter_classes = (
            'none',
            'last_scattering_layer',
            'scattering_order',
            'scattering_order_per_layer',
        )
        if scatter_classes not in _valid_scatter_classes:
            raise ValueError(
                f'scatter_classes must be one of {_valid_scatter_classes}, '
                f'got {scatter_classes!r}'
            )
        if scatter_classes == 'none':
            nscl = 1
        if scatter_classes == 'scattering_order_per_layer':
            if norders < 1:
                raise ValueError(
                    f'norders must be >= 1 for scattering_order_per_layer '
                    f'mode, got {norders}'
                )
        if nscl > 1 and not alis:
            raise ValueError(
                'nscl>1 requires alis=True '
                '(scatter-class decomposition needs ALIS cdist)'
            )
        self.nscl = int(nscl)
        self.scatter_classes = scatter_classes
        self.norders = int(norders)
        # SCL_MODE: 0=none, 1=last_scattering_layer, 2=scattering_order,
        # 3=scattering_order_per_layer
        self._scl_mode = _valid_scatter_classes.index(scatter_classes)
        self.rng = _init_rng(rng)
        self.back = back
        self.thermal = thermal
        self.obj3d = obj3d
        self.opt3d = opt3d

        #
        # compilation option
        #
        options = []
        # options = ['-G']
        # options = ['-g', '-G']
        if not pp:
            # spherical shell calculation
            # automatically with ALT_PP (for eventually ocean
            # propagation)
            options.append('-DSPHERIQUE')
            options.append('-DALT_PP')
        if alt_pp:
            # new Plane Parallel propagation scheme
            options.append('-DALT_PP')
        if opt3d:
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
        if amf_variance:
            options.append(
                '-DAMF_VARIANCE'
            )  # Store cdist² for Jensen bias correction
        if cdist_wabs:
            options.append(
                '-DCDIST_WABS'
            )  # Include absorption weight in cdist accumulation
        if sif:
            options.append('-DSIF')
        if thermal:
            # thermal source
            options.append('-DTHERMAL')
        if back:
            # backward mode
            options.append('-DBACK')
        if bias:
            # bias sampling scheme for scattering and
            # reflection/transmission
            options.append('-DBIAS')
        if obj3d:
            # 3D Object mode
            options.append('-DOBJ3D')
        options.append('-D' + rng)
        # options.append('-lineinfo')

        #
        # compile the kernel or load binary
        #
        time_before_compilation = datetime.now()

        # load device.cu
        src_device_content = open(
            SRC_DEVICE, encoding='ascii', errors='ignore'
        ).read()

        # kernel compilation
        self.mod = SourceModule(
            src_device_content,
            nvcc='nvcc',
            options=options,
            no_extern_c=True,
            cache_dir=str(cache_dir),
            include_dirs=[
                str(DIR_SRC),
                str(DIR_SRC / 'incRNGs' / 'Random123'),
            ],
        )

        # load the kernel
        self.kernel = self.mod.get_function('launchKernel')
        # self.kernel2 = self.mod.get_function('launchKernel2')
        self.kernel2 = self.mod.get_function('reduce_absorption_gpu')

        #
        # common attributes
        #
        self.common_attrs = OrderedDict()
        self.common_attrs['compilation_time'] = (
            datetime.now() - time_before_compilation
        ).total_seconds()
        if autoinit:
            self.common_attrs['device'] = pycuda.autoinit.device.name()
            try:
                attr = (
                    pycuda
                    ._driver  # pyright: ignore[reportAttributeAccessIssue]
                    .device_attribute
                )
                self.common_attrs['device_number'] = (
                    pycuda.autoinit.device.get_attributes()[
                        attr.MULTI_GPU_BOARD_GROUP_ID
                    ]
                )
            except AttributeError:
                self.common_attrs['device_number'] = 'undefined'
        else:
            assert self.ctx is not None
            self.common_attrs['device'] = self.ctx.get_device().name()
            try:
                attr = (
                    pycuda
                    ._driver  # pyright: ignore[reportAttributeAccessIssue]
                    .device_attribute
                )
                self.common_attrs['device_number'] = (
                    self.ctx.get_device().get_attributes()[
                        attr.MULTI_GPU_BOARD_GROUP_ID
                    ]
                )
            except Exception:
                self.common_attrs['device_number'] = 'undefined'
        self.common_attrs['pycuda_version'] = pycuda.VERSION_TEXT
        cuda_version = (
            cuda.get_version()  # pyright: ignore[reportAttributeAccessIssue]
        )
        self.common_attrs['cuda_version'] = '.'.join(
            [str(x) for x in cuda_version]
        )
        self.common_attrs.update(_get_git_attrs())

    def clear_context(self) -> None:
        """
        Manually kill the CUDA context.

        Notes
        -----
        Once this method has been called, the run method can no longer
        be used: the Smartg object must be reinitialized.
        """
        try:
            assert self.ctx is not None
            self.ctx.pop()
            self.ctx.detach()
            self.ctx = None
            from pycuda.tools import clear_context_caches

            clear_context_caches()
            if self.autoinit:
                # In case of autoinit delete pycuda.autoinit
                import pycuda.autoinit

                del pycuda.autoinit
        except Exception:
            print("There is no current context to clear.")

    def run(
        self,
        wavelength,
        atmosphere=None,
        surface=None,
        water=None,
        environment=None,
        alis_options: dict | None = None,
        n_photons: float = 1e9,
        depo: float = 0.0279,
        depo_water: float = 0.0906,
        th_deg: float = 0.0,
        ph_deg: float = 0.0,
        seed: int = -1,
        earth_radius: float = 6371.0,
        wavelength_proba: np.ndarray | None = None,
        sensor_proba: np.ndarray | None = None,
        cell_proba=None,
        n_theta: int = 45,
        n_phi: int = 90,
        n_icdf: float = 1e6,
        theta_grid: str | NDArray[np.floating] | None = None,
        output_layers: int = 0,
        xblock: int = 256,
        xgrid: int = 256,
        n_loop: float | None = None,
        progress: bool = True,
        le: dict | None = None,
        flux: str | None = None,
        stdev: bool = False,
        stdev_lim: StdevLim | None = None,
        beer: int = 1,
        russian_roulette: int = 0,
        russian_roulette_weight: float = 0.1,
        sza_max: float = 90.0,
        sun_disc: float = 0.0,
        le_fov: float = 0.0,
        sensor=None,
        refraction: bool = False,
        reflectance: bool = True,
        my_objects=None,
        interval=None,
        is_atm: int | None = 1,
        cus_l=None,
        s_min: float = 0,
        s_max: float = 1e6,
        r_min: float = 0,
        r_max: float = 1e6,
        ffs: bool = False,
        direct: bool = False,
        ocean_interaction: bool | None = None,
        polarization: bool = True,
        no_aer_output: bool = False,
    ) -> xr.Dataset:
        """
        Run a SMART-G simulation

        Parameters
        ----------
        wavelength : float | list | 1-D ndarray
            Wavelength(s) in nm. It can be a list of ReptranIband or
            KdisIband objects.
        atmosphere : None | Atm1D | MLUT, optional
            The atmosphere profile. If None, there is no atmosphere.
        surface : None | RoughSurface | FlatSurface | LambSurface, optional
            The surface profile, see `smartg.surface`. If None, there is
            no surface.
        water : None | Water1D | MLUT, optional
            The water profile. If None, there is no water.
        environment : None | Environment, optional
            The environment (adjacency effect) profile. If None, there
            is no environment.
        alis_options : None | dict, optional
            The alis options (the compilation option alis must be set to
            True). The dictionary keys:

            * 'nlow' : int
                -> The number of low spectral resolution computation. If
                nlow = -1 select all wavelengths.
            * 'hist' : bool, optional
                -> Activate history. If the key does not exist the
                history mode is not activated.
            * 'max_hist' : int, optional
                -> The max number of history (only if hist is True).
                Default 8e6.
            * 'njac' : int, optional
                -> The number of perturbed profiles. Default no
                Jacobian.
            * 'njac_abs' : bool, optional
                -> If True, Jacobians are for absorption only.
                ``weight_sca`` is computed
                   only for the reference wavelength group (allowing a
                   small ``nlow``), and is then reused (interpolated)
                   for all perturbed groups. The scattering correction
                   for perturbed wavelengths is taken from the reference
                   group, while their absorption is recomputed from the
                   perturbed profile. Requires ``njac`` > 0. Default
                   False.

            Note: Optional for the dictionary keys indicate that the key
            is not required to be present.
        n_photons : int, optional
            The total number of photons used for the simulation. Default
            1e9.
        depo : float, optional
            The Rayleigh depolarization factor (air). Default 0.0279.
        depo_water : float, optional
            The Rayleigh depolarization factor (water). Default 0.0906.
        th_deg : float, optional
            The sun/viewing zenith angle in forward/backward mode, in
            degrees. This parameter is ignored if the parameter `sensor`
            is used. If the parameter `cus_l` is a CusBackward with a
            v_sun vector, that vector gives the sun direction instead
            of th_deg and ph_deg, which then only define the
            atmosphere impact point and the VZA attribute.
        ph_deg : float, optional
            The sun/viewing azimuth angle in forward/backward mode, in
            degrees. This parameter is ignored if the parameter `sensor`
            is used. See also th_deg.
        seed : int, optional
            The seed used to initiate the series of random numbers.
            Default based on clock time.
        earth_radius : float, optional
            The earth radius in km
        wavelength_proba : None | 1-D ndarray, optional
            The inversed cumulative distribution function for wavelength
            selection. It is for example the result of function
            icdf(proba, n).
        sensor_proba : None | 1-D ndarray, optional
           The inversed cumulative distribution function for sensor
           selection. It is for example the result of function
           icdf(proba, n).
        cell_proba : None | 2-D ndarray, optional
            The inversed cumulative distribution function for cell
            selection. It is for example the result of function
            icdf_2d(proba, n).
        n_theta : int, optional
            The number of viewing/sun zenith angles in forward/backward
            for the cone sampling. This parameter is ignored if the
            parameter `le` is used.
        n_phi : int, optional
            The number of viewing/sun azimuth angles in forward/backward
            for the cone sampling. This parameter is ignored if the
            parameter `le` is used.
        n_icdf : int, optional
            The number of scattering angles of the atmospheric and
            oceanic phase tables uploaded to the GPU. Ignored when
            `theta_grid` supplies its own grid. Each angle costs 28
            bytes per phase function, 24 for the matrix and 4 for its
            cumulative probability, so the 1e6 default costs 28 MB per
            phase function.
        theta_grid : str or ndarray, optional
            Distribution of those scattering angles:

            - None -> equally spaced (default)
            - 'phase' -> adopt the grid the phase matrices already
              carry, i.e. their `theta_atm` and `theta_oc` axes; with
              a profile computed with `n_theta='native'`, this keeps
              the union of the angles the components' tables carry
            - 'uniform', 'chebyshev', 'lobatto' or 'peak' -> generate
              `n_icdf` angles of that kind, see `smartg.phase.theta_grid`
            - an array of angles in degrees, from 0 to 180

            Clustering the angles towards 0 and 180 degrees resolves
            the forward diffraction peak of large particles with far
            fewer of them. On a cloud phase function, 1801 Lobatto
            angles are more accurate than 18001 equally spaced ones,
            for a tenth of the memory. Clustering is not free though:
            it thins the middle of the range, where 1801 Lobatto
            angles are 3 times less accurate than 1801 equally spaced
            ones. Use 'peak' to choose that trade-off explicitly.
        output_layers : int, optional
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

            Note: Consider only the needed layers may reduce
            significantly the computational time.
        xblock : int, optional
            The number of cuda blocks.
        xgrid : int, optional
            The number of cuda grids.
        n_loop : None | float, optional
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
                -> The zenith angles in degrees. Only if 'th' is not
                provided.
            * 'phi_deg' : 1-D ndarray | list, optional
                -> The azimuth angles in degrees. Only if 'phi' is not
                provided.
            * 'zip' : bool, optional
               -> If True, then 'th' and 'phi' covary and the output is
               only one-dimensional n_theta, but user should verify
               that n_phi==n_theta.
            * 'count_level' : 1-D ndarray | list, optional
                -> The level to consider. Possibilities: -2(all),
                -1(none), 0(UPTOA), 1(DOWN0P), 2(DOWN0M),
                   3(UP0P), 4(UP0M) or 5(DOWNB). The level to consider
                   may change only with th/th_deg. The array must be of
                   length n_theta. If the key is not present it will be
                   the same as count_level = np.full_like(th/th_deg, -2,
                   dtype=np.int32).

            Note: Optional for the dictionary keys indicate that the key
            is not required to be present. If th/phi are not provided,
            th_deg/phi_deg must be given.
        flux : None | str, optional
            Activate the flux mode (instead of radiance). Only 2
            choices:
                - 'planar'
                - 'spherical'
        stdev : bool, optional
            Activate the calculation of the standard deviation (between
            each kernel run).
        stdev_lim : None | StdevLim, optional
            To stop the computation if the standard deviation is above a
            certain limit. Only if stdev is True.
        beer : int, optional
            If beer=1 compute absorption using Beer-Lambert law,
            otherwise compute it with the Single scattering albedo. beer
            automatically set to 1 if ALIS is True.
        russian_roulette : int, optional
            Activate the Russian Roulette. ON = 1 and OFF = 0.
        russian_roulette_weight : float, optional
            The threshold weight to apply to the Russian Roulette.
        sza_max : float, optional
            The maximum SZA value for solar BOXES in case a Regulard
            grid and cone sampling.
        sun_disc : float, optional
            The angular size of the Sun disc in degrees, 0 (default
            means no angular size). In the B and BR modes the angular
            size of the Sun is given by the sun_fov parameter of
            CusBackward instead, or by le_fov under local estimate,
            and sun_disc has no effect on the signal collected by the
            receiver.
        le_fov : float, optional
            The half-angle in degrees of the cone sampled around each
            local estimate direction, 0 (default) meaning the exact
            directions. It gives its angular size to the source seen
            by the local estimate: 0.266 for the Sun disc, whose
            angular radius it is. It requires the parameter le, and
            it is the local estimate counterpart of the sun_fov
            parameter of CusBackward, which applies only without le.
        sensor : None | Sensor | list, optional
            The light source / sensor (Sensor object or list of Sensor
            objects) in forward / backward mode.
        refraction : bool, optional
            If True include atmospheric refraction.
        reflectance : bool, optional
           Convert output to reflectance units, otherwise in radiance
           units with Solar irradiance set to PI. Only of flux is None
           and for plane parallel atmosphere.
        my_objects : None | list, optional
            A list of 3d objects (Entity objects) that will be used in
            the simulation. Currently sphere and plane objects are
            considered. The compilation option `obj3d` must be set to
            True.
        interval : None | list, optional
            A principal bounding box in case 3d objects are
            incorporated. It must be a list composed of 2 lists with the
            bbox min and max values [[xmin, ymin, zmin], [xmax, ymax,
            zmax]].
        is_atm : int, optional
            If is_atm=0 provide more robust test with 3d objects in case
            the atmosphere we remove the atmosphere.
        cus_l : None | CusForward | CusBackward, optional
            Use the RF, FF (CusForward) or B, BR (CusBackward) launching
            modes. The compilation option `obj3d` must be set to True.
            A CusBackward can also carry the sun direction as a vector
            in its v_sun parameter (see th_deg) and the angular size
            of the sun in its sun_fov parameter (see sun_disc), which
            applies only without the parameter le (see le_fov).
        s_min : int, optional
            The minimum number of interactions (scattering/reflection).
            Default 0.
        s_max : int, optional
            The maximum number of interactions (scattering/reflection).
            Default 1e6.
        r_min : int, optional
            The minimum number of reflections (by surface only, not
            environment). Default 0.
        r_max : int, optional
            The maximum number of reflections (by surface only, not
            environment). Default 1e6
        ffs : bool, optional
            Forced First Scattering (for use in spherical limb geometry
            only). Default False.
        direct : bool, optional
            Include directly transmitted photons. Default False.
        ocean_interaction : None | int, optional
            If ocean_interaction=1 select photons that interact with
            ocean. Default None, no selection.
        polarization : bool, optional
            Consider (if True) the polarization of the light. Default
            True.
        no_aer_output : bool, optional
            Add output where only photons not scattered by aerosols are
            considered. Default False. For example, next to the output
            variable m['I_up (TOA)'] we will also get m['I_up (TOA),
            no_aer'].

        Returns
        -------
        out : xr.Dataset
            A Dataset containing the simulation results and more,
            e.g.:
            - the polarized dimensionless reflectance (I,Q,U,V) at the
              different layers
            - the number of photons (N) received at each layer
            - the profiles and phase functions
            - attributes
            - ...

        Notes
        -----

        In cone sampling, the sun/sensor is targeting the origin (0,0,0)
        in forward/backward.

        Examples
        --------
        >>> from smartg.smartg import Smartg
        >>> from smartg.surface import RoughSurface
        >>> from smartg.atmosphere import Atm1D, AerOPAC
        >>> from smartg.water import Water1D, HydrosolPR
        >>> aer = AerOPAC('maritime_clean', 0.5, 550.)
        >>> atmosphere = Atm1D('afglt', comp=[aer])
        >>> water = Water1D(grid=[0, -5.], comp=[HydrosolPR(chl=0.5)])
        >>> surface = RoughSurface(wind=5., nh2o=1.34)
        >>> m = Smartg().run(wavelength=550., atmosphere=atmosphere,
        ...                  water=water, surface=surface)
        >>> # Look at the TOA radiance/reflectance ('I_up (TOA)')
        >>> m['I_up (TOA)'].dims
        ('Azimuth angles', 'Zenith angles')
        >>> m['I_up (TOA)'].values  # doctest: +SKIP
        array([[0.15751, 0.14705, ..., 0.11015, 0.07649],
               ...,
               [0.15648, 0.14717, ..., 0.10935, 0.07744]])

        """

        if not self.pp and water is not None:
            raise ValueError(
                "Ocean + spherical atm is not allowed! Still in progress..."
            )

        if output_layers not in (np.arange(9, dtype=np.int32) - 1):
            raise ValueError(
                'The output_layers value must be an integer between -1 and 7.'
            )

        # Check the cone of the local estimate: it is sampled around
        # the directions given by the le parameter
        if le_fov < 0. or le_fov >= 90.:
            raise ValueError(
                'The le_fov value must be in [0, 90[ degrees'
            )
        if le_fov > 0. and le is None:
            raise ValueError(
                'The parameter le_fov can be used only with the '
                'parameter le'
            )

        # Check the custom launching mode and the 3D objects against
        # the compilation options: the launching code of the forward
        # modes (RF, FF) is compiled only without the back option, and
        # the one of the backward modes (B, BR) only with it
        if cus_l is not None:
            if my_objects is None:
                raise ValueError(
                    'The parameter cus_l can be used only if parameter '
                    'my_objects is provided.'
                )
            if not isinstance(cus_l, (CusForward, CusBackward)):
                raise ValueError(
                    'The cus_l parameter must be a CusForward or a '
                    'CusBackward'
                )
            if isinstance(cus_l, CusBackward) and not self.back:
                raise ValueError(
                    'CusBackward can be used only with the compilation '
                    'option back=True'
                )
            if isinstance(cus_l, CusForward) and self.back:
                raise ValueError(
                    'CusForward can be used only with the compilation '
                    'option back=False'
                )
            if sensor is not None:
                raise ValueError(
                    'The use of sensor(s) and a custom launching mode'
                    + ' (cusForward or cusBackward) is prohibited!'
                )
            # sun_fov and le_fov give the same angular size to the
            # sun on paths that never meet: sun_fov only without le
            # (the solar cone of countPhotonObj3D and the
            # normalization of the receiver signal), le_fov only
            # with it
            if (
                isinstance(cus_l, CusBackward)
                and le is not None
                and le_fov == 0.
            ):
                warn(
                    'The sun_fov of the CusBackward ({} degrees) has '
                    'no effect with the le parameter, where the '
                    'angular size of the source is given by le_fov, '
                    'left at 0: the local estimate looks at a point '
                    'source. Set le_fov=0.266 for the solar '
                    'disc.'.format(cus_l.dict['sun_fov']),
                    stacklevel=2,
                )
        if my_objects is not None and not self.obj3d:
            raise ValueError(
                'The parameter my_objects can be used only with the '
                'compilation option obj3d=True'
            )

        # Compute the sun direction as vector, given either by the
        # v_sun attribute of CusBackward or by th_deg and ph_deg
        if cus_l is not None and cus_l.dict.get('v_sun') is not None:
            v_sun = cus_l.dict['v_sun']
        else:
            v_sun = gc.ang2vec(th_deg, ph_deg, vec_view='nadir')
        v_sun = gc.normalize(v_sun)

        surf_lph = 0
        if cus_l is not None:
            if cus_l.dict['mode'] == "B":
                sensor = Sensor(
                    pos_x=cus_l.dict['position'].x,
                    pos_y=cus_l.dict['position'].y,
                    pos_z=cus_l.dict['position'].z,
                    th_deg=cus_l.dict['th_deg'],
                    ph_deg=cus_l.dict['ph_deg'],
                    loc='ATMOS',
                    fov=0.0,
                    sensor_type=0,
                )
                # fov=cus_l.dict['receiver_fov'],
                # sensor_type=cus_l.dict['sampling_code'])
            elif cus_l.dict['mode'] == "BR":
                sensor = Sensor(
                    pos_x=cus_l.dict['receiver'].transformation.transx,
                    pos_y=cus_l.dict['receiver'].transformation.transy,
                    pos_z=cus_l.dict['receiver'].transformation.transz,
                    th_deg=cus_l.dict['th_deg'],
                    ph_deg=cus_l.dict['ph_deg'],
                    loc='ATMOS',
                    fov=0.0,
                    sensor_type=0,
                )
                # fov=cus_l.dict['receiver_fov'],
                # sensor_type=cus_l.dict['sampling_code'])
            elif cus_l.dict['mode'] == "FF":
                # The projected surface at TOA where the photons are
                # launched
                dot_nn = gc.dot(v_sun * -1, gc.Vector(0.0, 0.0, 1.0))
                if (
                    cus_l.dict['sampling_code'] == 2
                    and cus_l.dict['fov'] > 1e-6
                ):  # isotropic
                    surf_lph = float(cus_l.dict['cfx']) * float(
                        cus_l.dict['cfy']
                    )
                else:
                    surf_lph = (
                        float(cus_l.dict['cfx'])
                        * float(cus_l.dict['cfy'])
                        * dot_nn
                    )

        #
        # initialization
        #

        # Begin initialization with OBJ ============================
        if my_objects is not None:
            # Main bounding box initialization
            if interval is not None:
                p_min_x = interval[0][0]
                p_min_y = interval[0][1]
                p_min_z = interval[0][2]
                p_max_x = interval[1][0]
                p_max_y = interval[1][1]
                p_max_z = interval[1][2]
            else:
                p_min_x = -100000
                p_min_y = -100000
                p_min_z = 0
                p_max_x = 100000
                p_max_y = 100000
                p_max_z = 120

            # Initialize all the parameters linked with 3D objects
            (
                n_gobj,
                n_obj,
                n_robj,
                surf_lph_rf,
                n_h,
                z_alt_h,
                tot_s_h,
                tc,
                n_cx,
                n_cy,
                my_objects0,
                my_gobj0,
                my_robj0,
                my_spect_obj0,
                n_cos,
            ) = _init_obj(
                lgobj=my_objects, v_sun=v_sun, wavelength=wavelength,
                cus_l=cus_l,
            )

            # If we are in RF mode don't forget to update the value of
            # surf_lph
            if surf_lph_rf is not None:
                surf_lph = surf_lph_rf

        else:
            my_objects0 = gpuzeros(1, dtype=np.uint32)
            # my_objects0 = gpuzeros(1, dtype='int32')
            my_gobj0 = gpuzeros(1, dtype='int32')
            my_robj0 = gpuzeros(1, dtype='int32')
            my_spect_obj0 = gpuzeros(
                1, dtype='int32'
            )  # normally 2 dims: obj dim + wavelength dim
            n_obj = 0
            n_gobj = 0
            n_robj = 0
            p_min_x = None
            p_min_y = None
            p_min_z = None
            p_max_x = None
            p_max_y = None
            p_max_z = None
            is_atm = None
            tc = None
            n_cx = 10
            n_cy = 10
            n_h = 0
            z_alt_h = None
            tot_s_h = None
            n_cos = None
        # END OBJ ===================================================

        if n_phi % 2 == 1:
            warn('Odd number of azimuth', stacklevel=2)

        if (n_loop is None) and (n_obj <= 0):
            n_loop = min(n_photons / 30, 1e6)
        elif (n_loop is None) and (n_obj > 0):
            n_loop = min(n_photons / 10, 1e6)

        n_icdf = int(n_icdf)

        # number of output levels
        # warning! values defined in communs.h should be < LVL
        n_lvl = 6

        # warning! values defined in communs.h
        # Maximum number of photons histories (alis=True and
        # alis_options['hist'] = True), otherwise 0 (no histories)
        max_hist = np.int64(1)
        max_nlow = 801

        # number of Stokes parameters of the radiation field
        n_pstk = 4

        t0 = datetime.now()

        attrs = OrderedDict()
        attrs.update({'processing started at': t0})
        attrs.update({'VZA': th_deg})
        attrs.update({'MODE': {True: 'PPA', False: 'SSA'}[self.pp]})
        attrs.update({'XBLOCK': xblock})
        attrs.update({'XGRID': xgrid})
        attrs.update({'NPHOTONS': '{:g}'.format(n_photons)})

        if not isinstance(wavelength, BandSet):
            wavelength = BandSet(wavelength)
        n_lam = wavelength.size

        n_low = 0
        hist = False
        hist_code = 0
        n_jac = 0
        n_jac_abs = 0
        if alis_options is not None:
            if 'hist' in alis_options.keys():
                if alis_options['hist']:
                    hist = True
                    if 'max_hist' in alis_options.keys():
                        max_hist = np.int64(alis_options['max_hist'])
                    else:
                        max_hist = np.int64(8000000)
            if 'njac' in alis_options.keys():
                n_jac = alis_options['njac']
            if alis_options.get('njac_abs', False):
                n_jac_abs = 1
            if alis_options['nlow'] == -1:
                n_low = n_lam
            else:
                n_low = alis_options['nlow']
            beer = 1
            assert n_low <= max_nlow

        if hist:
            hist_code = 1

        if surface is not None:
            if surface.dict['BRDF'] != 0:
                water = None  # special case BRDF, water is shortcut

        # determine sim
        if (atmosphere is not None) and (surface is None) and (water is None):
            sim = -2  # atmosphere only
        elif (
            (atmosphere is None) and (surface is not None)
            and (water is None)
        ):
            sim = -1  # surface only
        elif (
            (atmosphere is None) and (surface is not None)
            and (water is not None)
        ):
            sim = 0  # ocean + dioptre
        elif (
            (atmosphere is not None) and (surface is not None)
            and (water is None)
        ):
            sim = 1  # atmosphere + dioptre
        elif (
            (atmosphere is not None) and (surface is not None)
            and (water is not None)
        ):
            sim = 2  # atmosphere + dioptre + ocean
        elif (
            (atmosphere is None) and (surface is None)
            and (water is not None)
        ):
            sim = 3  # ocean only
        else:
            raise ValueError('Error in SIM')

        #
        # atmosphere
        #
        if isinstance(atmosphere, Atmosphere):
            prof_atm = atmosphere.calc(wavelength)
        elif isinstance(atmosphere, xr.Dataset) or (atmosphere is None):
            prof_atm = atmosphere
        elif hasattr(atmosphere, 'to_xarray'):
            prof_atm = atmosphere.to_xarray()
        else:
            raise ValueError(
                'atmosphere must be an Atmosphere class, an xr.Dataset, an '
                'MLUT-like object or equal to None!'
            )

        if prof_atm is not None and hasattr(prof_atm, 'to_xarray'):
            prof_atm = prof_atm.to_xarray()

        if prof_atm is None:
            z_toa = 120.0
        elif 'z_atm' in prof_atm.coords:
            z_toa = prof_atm.coords['z_atm'].to_numpy()[0]
        else:
            # 3D profile: no vertical axis, and ZTOAd is not used by
            # the OPT3D kernel path
            z_toa = 0.0

        if prof_atm is not None:
            ang_atm, agrid_atm = _resolve_agrid(
                theta_grid, n_icdf, prof_atm, 'atm'
            )
            # caer is reachable from the kernel only through a pointer
            # in constant memory, which does not keep it alive: it has
            # to stay referenced here until the kernel is done
            faer, caer = _calc_phase_gpu(
                prof_atm,
                n_theta=agrid_atm[0],
                depo=depo,
                kind='atm',
                polarization=polarization,
                # mode 0 keeps the historical grid of each builder
                ang_a=None if agrid_atm[1] == 0 else ang_atm,
            )
            pgrid_atm = (caer.shape[-1], caer)
            prof_atm_gpu, cell_atm_gpu = _init_profile(
                wavelength, prof_atm, 'atm'
            )
            if 'z_atm' in prof_atm.coords:
                n_atm = len(prof_atm.coords['z_atm']) - 1
            else:
                n_atm = prof_atm.sizes['iopt'] - 1
            if self.opt3d:
                n_atm_abs = np.int32(prof_atm['iabs_atm'].to_numpy().max())
            else:
                n_atm_abs = n_atm
        else:
            faer = gpuzeros(1, dtype='float32')
            caer = gpuzeros(1, dtype=TYPE_PCDF)
            agrid_atm = (n_icdf, 0, None)
            pgrid_atm = (1, caer)
            prof_atm_gpu = to_gpu(np.zeros(1, dtype=TYPE_PROFILE))
            cell_atm_gpu = to_gpu(np.zeros(1, dtype=TYPE_CELL))
            n_atm = 0
            n_atm_abs = 0

        # computation of the impact point
        # x0, _ = _impact_init(prof_atm, n_lam, th_deg, earth_radius,
        # self.pp)
        x0, tab_trans_dir_analytic = _impact_init(
            prof_atm, n_lam, th_deg, earth_radius, self.pp
        )

        # sensor definition
        if sensor is None:
            # by defaut sensor in forward mode, with ZA=180.-th_deg,
            # ph_deg=180., fov=0.
            if sim == 3:
                sensor2 = [
                    Sensor(
                        th_deg=180.0 - th_deg,
                        ph_deg=ph_deg + 180.0,
                        loc='OCEAN',
                    )
                ]
            elif (sim == -1) or (sim == 0):
                sensor2 = [
                    Sensor(
                        th_deg=180.0 - th_deg,
                        ph_deg=ph_deg + 180.0,
                        loc='SURF0P',
                    )
                ]
            else:
                if cus_l is not None:  # for FF mode
                    sensor2 = [
                        Sensor(
                            pos_x=x0.get()[0],
                            pos_y=x0.get()[1],
                            pos_z=x0.get()[2],
                            th_deg=180.0 - th_deg,
                            ph_deg=ph_deg + 180.0,
                            loc='ATMOS',
                        )
                    ]
                    # fov=0.0, sensor_type=0)]
                    # fov=cus_l.dict['fov'],
                    # sensor_type=cus_l.dict['sampling_code'])]
                else:
                    sensor2 = [
                        Sensor(
                            pos_x=x0.get()[0],
                            pos_y=x0.get()[1],
                            pos_z=x0.get()[2],
                            th_deg=180.0 - th_deg,
                            ph_deg=ph_deg + 180.0,
                            loc='ATMOS',
                        )
                    ]
        elif isinstance(sensor, Sensor):
            sensor2 = [sensor]
        elif isinstance(sensor, list):
            sensor2 = sensor
        else:
            raise ValueError(
                'sensor must be a Sensor class, a list or Sensor '
                'classes or equal to None!'
            )

        n_sensor = len(sensor2)

        tab_sensor = np.zeros(n_sensor, dtype=TYPE_SENSOR, order='C')
        for i, s in enumerate(sensor2):
            for k in s.dict.keys():
                tab_sensor[i][k] = s.dict[k]
        tab_sensor = to_gpu(tab_sensor)

        # Auto-set sun_disc from sensor FOV if not explicitly set
        # This ensures sensor cone angle is available in kernel for
        # direct beam tolerance
        if sun_disc == 0:
            for sens in sensor2:
                if sens.dict['sensor_type'] == 1 and sens.dict['fov'] > 1e-6:
                    sun_disc = sens.dict['fov']
                    break  # Use first sensor with cone FOV

        # The min and max posx and posy of sensors. Useful for forward
        # mode in 3d atm
        sxmin = np.inf
        sxmax = -np.inf
        symin = np.inf
        symax = -np.inf
        for sens in sensor2:
            if sens.cell_size > 0:
                half_csize = 0.5 * sens.cell_size
                sxmin = min(sxmin, sens.dict['pos_x'] - half_csize)
                sxmax = max(sxmax, sens.dict['pos_x'] + half_csize)
                symin = min(symin, sens.dict['pos_y'] - half_csize)
                symax = max(symax, sens.dict['pos_y'] + half_csize)
            else:
                sxmin = min(sxmin, sens.dict['pos_x'])
                sxmax = max(sxmax, sens.dict['pos_x'])
                symin = min(symin, sens.dict['pos_y'])
                symax = max(symax, sens.dict['pos_y'])
        if sensor2[0].cell_size > 0:
            nbsx = round((sxmax - sxmin) / sensor2[0].cell_size)
            nbsy = round((symax - symin) / sensor2[0].cell_size)
        else:
            nbsx = 0
            nbsy = 0

        #
        # ocean
        #
        if isinstance(water, Water):
            prof_oc = water.calc(wavelength)
        elif isinstance(water, xr.Dataset) or (water is None):
            prof_oc = water
        elif hasattr(water, 'to_xarray'):
            prof_oc = water.to_xarray()
        else:
            raise ValueError(
                'water must be a Water class, an xr.Dataset, an '
                'MLUT-like object or equal to None!'
            )

        if prof_oc is not None and hasattr(prof_oc, 'to_xarray'):
            prof_oc = prof_oc.to_xarray()

        if prof_oc is not None:
            ang_oc, agrid_oc = _resolve_agrid(
                theta_grid, n_icdf, prof_oc, 'oc'
            )
            # coce, like caer above, is kept alive by this reference
            foce, coce = _calc_phase_gpu(
                prof_oc,
                n_theta=agrid_oc[0],
                depo=depo_water,
                kind='oc',
                polarization=polarization,
                ang_a=None if agrid_oc[1] == 0 else ang_oc,
            )
            pgrid_oc = (coce.shape[-1], coce)
            prof_oc_gpu, cell_oc_gpu = _init_profile(wavelength, prof_oc, 'oc')
            n_oce = len(prof_oc.coords['z_oc']) - 1
            if self.opt3d:
                n_oce_abs = np.int32(prof_oc['iabs_oc'].to_numpy().max())
            else:
                n_oce_abs = n_oce
        else:
            foce = gpuzeros(1, dtype='float32')
            coce = gpuzeros(1, dtype=TYPE_PCDF)
            agrid_oc = (n_icdf, 0, None)
            pgrid_oc = (1, coce)
            prof_oc_gpu = to_gpu(np.zeros(1, dtype=TYPE_PROFILE))
            cell_oc_gpu = to_gpu(np.zeros(1, dtype=TYPE_CELL))
            n_oce = 0
            n_oce_abs = 0

        #
        # albedo and adjacency effect
        #
        spectrum = np.zeros(n_lam, dtype=TYPE_SPECTRUM)
        envmap = np.zeros(1, dtype=TYPE_ENV_MAP)
        spectrum['lambda'] = wavelength[:]
        if environment is None:
            # default values (no environment effect)
            environment = Environment()
            if surface is not None:
                if surface.alb is not None:
                    spectrum['alb_surface'] = surface.alb.get(wavelength[:])
                elif surface.kp is not None:
                    spectrum['alb_surface'] = surface.kp[0].get(wavelength[:])
                    spectrum['k1p_surface'] = surface.kp[1].get(wavelength[:])
                    spectrum['k2p_surface'] = surface.kp[2].get(wavelength[:])
                    spectrum['k3p_surface'] = surface.kp[3].get(wavelength[:])
                else:
                    spectrum['alb_surface'] = -999.0
            else:
                spectrum['alb_surface'] = -999.0
        else:
            assert surface is not None
            if surface.alb is not None:
                spectrum['alb_surface'] = surface.alb.get(wavelength[:])
            elif surface.kp is not None:
                spectrum['alb_surface'] = surface.kp[0].get(wavelength[:])
                spectrum['k1p_surface'] = surface.kp[1].get(wavelength[:])
                spectrum['k2p_surface'] = surface.kp[2].get(wavelength[:])
                spectrum['k3p_surface'] = surface.kp[3].get(wavelength[:])
            albenv = environment.alb.get(wavelength[:])
            if albenv.ndim == 2:
                environment.nenv = albenv.shape[1]
                spectrum['alb_envs'][:, : environment.nenv] = albenv
                shp = environment.alb.map.data.shape
                environment.nxenvmap = shp[0]
                environment.nyenvmap = shp[1]
                envmap = np.zeros(shp, dtype=TYPE_ENV_MAP)
                x_map, y_map = np.meshgrid(
                    environment.alb.map.axis('X'),
                    environment.alb.map.axis('Y'),
                    indexing='ij',
                )
                envmap['x'] = x_map
                envmap['y'] = y_map
                envmap['env_index'] = environment.alb.get_map(x_map, y_map)
            else:
                spectrum['alb_env'] = albenv

        if water is None:
            spectrum['alb_seafloor'] = -999.0
        else:
            assert prof_oc is not None
            spectrum['alb_seafloor'] = prof_oc['albedo_seafloor'].data[...]

        envmap = to_gpu(envmap)
        spectrum = to_gpu(spectrum)

        # Local Estimate option
        le_code = 0
        zip_code = 0
        if le is not None:
            le_code = 1
            if 'th' not in le:
                le['th'] = (
                    np.array(le['th_deg'], dtype='float32').ravel()
                    * np.pi
                    / 180.0
                )
            else:
                le['th'] = np.array(le['th'], dtype='float32').ravel()
            if 'phi' not in le:
                le['phi'] = (
                    np.array(le['phi_deg'], dtype='float32').ravel()
                    * np.pi
                    / 180.0
                )
            else:
                le['phi'] = np.array(le['phi'], dtype='float32').ravel()

            n_theta = le['th'].shape[0]
            n_phi = le['phi'].shape[0]

            if 'zip' in le:
                if le['zip']:
                    assert n_phi == n_theta
                    zip_code = 1
                    n_phi = 1

            if 'count_level' in le:
                le['count_level'] = np.array(
                    le['count_level'], dtype='int32'
                ).ravel()
                assert len(le['count_level']) == n_theta

        flux_code = 0
        if flux is not None:
            le_code = 0
            if flux == 'planar':
                flux_code = 1
            if flux == 'spherical':
                flux_code = 2
            if flux == 'tilted planar':
                flux_code = 3

        if wavelength_proba is not None:
            assert wavelength_proba.dtype == 'int64'
            wavelength_proba_icdf = to_gpu(wavelength_proba)
            n_wavelength_proba = len(wavelength_proba_icdf)
        else:
            wavelength_proba_icdf = gpuzeros(1, dtype='int64')
            n_wavelength_proba = 0

        if sensor_proba is not None:
            assert sensor_proba.dtype == 'int64'
            sensor_proba_icdf = to_gpu(sensor_proba)
            n_sensor_proba = len(sensor_proba_icdf)
        else:
            sensor_proba_icdf = gpuzeros(1, dtype='int64')
            n_sensor_proba = 0

        if cell_proba is not None:
            if (cell_proba == 'auto') and not self.back and self.thermal:
                assert prof_atm is not None
                if 'z_atm' not in prof_atm.coords:
                    raise ValueError(
                        "cell_proba='auto' requires a 1D atmosphere "
                        "profile (the 3D profile has no z_atm axis "
                        "and no temperature profile)"
                    )
                kabs = od2k(prof_atm, 'OD_abs_atm')
                z = -prof_atm.coords['z_atm'].to_numpy()
                B = blackbody_radiance(
                    wavelength[:][:, None],
                    prof_atm['T_atm'].to_numpy()[None, :],
                )
                emission = xr.DataArray(
                    kabs * B,
                    dims=['wavelength', 'z_atm'],
                    coords={'wavelength': wavelength[:], 'z_atm': z},
                )
                norm_emission = (4 * np.pi) * emission.sum(dim='z_atm')
                p_emission = emission * (4 * np.pi) / norm_emission
                cell_proba_icdf = to_gpu(icdf_2d(p_emission.to_numpy()).T)
                n_cell_proba = cell_proba_icdf.shape[0]
            else:
                assert not isinstance(cell_proba, str)
                assert cell_proba.shape[1] == n_lam
                cell_proba_icdf = to_gpu(cell_proba)
                n_cell_proba = cell_proba.shape[0]
        else:
            cell_proba_icdf = gpuzeros(1, dtype='int64')
            n_cell_proba = 0

        refrac = 0
        if refraction:
            refrac = 1

        horiz = 1
        if not self.pp and not reflectance:
            horiz = 0

        # initialization of the constants
        _init_const(
            surface,
            environment,
            n_atm,
            n_atm_abs,
            n_oce,
            n_oce_abs,
            self.mod,
            n_loop,
            th_deg,
            xblock,
            xgrid,
            n_lam,
            sim,
            agrid_atm,
            agrid_oc,
            pgrid_atm,
            pgrid_oc,
            n_theta,
            n_phi,
            output_layers,
            earth_radius,
            le_code,
            zip_code,
            flux_code,
            ffs,
            direct,
            ocean_interaction,
            n_lvl,
            n_pstk,
            n_wavelength_proba,
            n_sensor_proba,
            n_cell_proba,
            beer,
            s_min,
            s_max,
            r_min,
            r_max,
            russian_roulette,
            russian_roulette_weight,
            n_low,
            n_jac,
            n_sensor,
            refrac,
            horiz,
            sza_max,
            sun_disc,
            le_fov,
            cus_l,
            n_obj,
            n_gobj,
            n_robj,
            p_min_x,
            p_min_y,
            p_min_z,
            p_max_x,
            p_max_y,
            p_max_z,
            is_atm,
            tc,
            n_cx,
            n_cy,
            v_sun,
            hist_code,
            z_toa,
            sensor2[0].cell_size,
            sxmin,
            sxmax,
            symin,
            symax,
            nbsx,
            nbsy,
            no_aer_output,
            n_scl=self.nscl,
            scl_mode=self._scl_mode,
            n_orders=self.norders,
            n_jac_abs=n_jac_abs,
        )

        # Initialize the progress bar
        p = make_progress(n_photons, progress)

        # Initialize the RNG
        seed = self.rng.setup(seed, xblock, xgrid)

        # Loop and kernel call
        (
            n_photons_in_tot,
            tab_photons_tot,
            tab_photons_tot_no_aer,
            tab_dist_tot,
            tab_hist_tot,
            tab_trans_dir,
            errorcount,
            n_photons_out_tot,
            n_photons_out_tot_no_aer,
            sigma,
            n_kernel,
            secs_cuda_clock,
            c_mat_visu_recep,
            mat_cats,
            mat_loss,
            w_ph_cats,
            w_ph_cats2,
        ) = _loop_kernel(
            n_photons,
            faer,
            foce,
            n_lvl,
            n_atm,
            n_atm_abs,
            n_oce,
            n_oce_abs,
            max_hist,
            n_low,
            n_pstk,
            xblock,
            xgrid,
            n_theta,
            n_phi,
            n_lam,
            n_sensor,
            self.double,
            self.kernel,
            p,
            x0,
            le,
            tab_sensor,
            envmap,
            spectrum,
            prof_atm_gpu,
            prof_oc_gpu,
            cell_atm_gpu,
            cell_oc_gpu,
            wavelength_proba_icdf,
            sensor_proba_icdf,
            cell_proba_icdf,
            stdev,
            stdev_lim,
            self.rng,
            self.alis,
            my_objects0,
            tc,
            n_cx,
            n_cy,
            my_gobj0,
            my_robj0,
            my_spect_obj0,
            hist=hist,
            amf_variance=self.amf_variance,
            nscl=self.nscl,
            le_fov=le_fov,
        )

        attrs['kernel time (s)'] = secs_cuda_clock
        attrs['number of kernel iterations'] = n_kernel
        attrs['seed'] = seed
        attrs.update(self.common_attrs)

        # If there is a receiver -> normalization of the signal
        # collected
        n_cte = None
        if tc is not None:
            c_mat_visu_recep, mat_cats, n_cte = _normalize_rec(
                c_mat_visu_recep=c_mat_visu_recep,
                mat_cats=mat_cats,
                n_cx=n_cx,
                n_cy=n_cy,
                n_photons=float(np.sum(n_photons_in_tot)),
                surf_lph=surf_lph,
                cell_size=tc,
                cus_l=cus_l,
                le=le_code,
            )

        if n_h > 0 and tc is not None and cus_l is not None:
            assert z_alt_h is not None
            mz_alt_h = z_alt_h / n_h
            s_rec = tc * tc * n_cx * n_cy  # ; weight_r=mat_cats[2, 1]
            # dic_stp : tuple incorporating parameters for Solar Tower
            # Power applications
            if self.back:
                receiver_fov = cus_l.dict['receiver_fov']
            else:
                receiver_fov = 0.0
            dic_stp = {
                "nb_H": n_h,
                "n_cos": n_cos,
                "totS_H": tot_s_h,
                "surfTOA": surf_lph,
                "MZAlt_H": mz_alt_h,
                "vSun": v_sun,
                "wRec": mat_cats[2, 1],
                "SREC": s_rec,
                "TC": tc,
                "LPH": cus_l.dict['lph'],
                "LPR": cus_l.dict['lpr'],
                "prog": progress,
                "n_cte": n_cte,
                "receiver_fov": receiver_fov,
            }
        # If there are no heliostats --> no analyses of optical losses
        elif tc is not None and cus_l is not None:
            s_rec = tc * tc * n_cx * n_cy
            mat_loss = None  # ;weight_r=mat_cats[2, 1]
            if self.back:
                receiver_fov = cus_l.dict['receiver_fov']
            else:
                receiver_fov = 0.0
            dic_stp = {
                "vSun": v_sun,
                "wRec": mat_cats[2, 1],
                "SREC": s_rec,
                "TC": tc,
                "LPH": cus_l.dict['lph'],
                "LPR": cus_l.dict['lpr'],
                "prog": progress,
                "n_cte": n_cte,
                "receiver_fov": receiver_fov,
            }
        elif tc is not None:
            s_rec = tc * tc * n_cx * n_cy
            mat_loss = None
            dic_stp = {"vSun": v_sun, "SREC": s_rec, "TC": tc, "n_cte": n_cte}
        # If there are no heliostats and receiver --> there is no STP
        else:
            dic_stp = None
            mat_loss = None  # ; weight_r=0

        # finalization
        output = _finalize(
            tab_photons_tot,
            tab_photons_tot_no_aer,
            tab_dist_tot,
            tab_hist_tot,
            wavelength[:],
            n_photons_in_tot,
            errorcount,
            n_photons_out_tot,
            n_photons_out_tot_no_aer,
            output_layers,
            tab_trans_dir,
            tab_trans_dir_analytic,
            attrs,
            prof_atm,
            prof_oc,
            sigma,
            horiz,
            le=le,
            flux=flux,
            back=self.back,
            sza_max=sza_max,
            sun_disc=sun_disc,
            hist=hist,
            c_mat_visu_recep=c_mat_visu_recep,
            dic_stp=dic_stp,
            mat_cats=mat_cats,
            mat_loss=mat_loss,
            w_ph_cats=w_ph_cats,
            w_ph_cats2=w_ph_cats2,
            no_aer_output=no_aer_output,
        )

        output.attrs['processing time (s)'] = (
            datetime.now() - t0
        ).total_seconds()

        if self.alis:
            p.finish(
                'Done! | Received {:.1%} of {:.3g} photons ({:.1%})'.format(
                    np.sum(n_photons_out_tot[0, ...])
                    / float(np.sum(n_photons_in_tot)),
                    np.sum(n_photons_in_tot) / float(n_lam),
                    np.sum(n_photons_in_tot)
                    / float(n_photons)
                    / float(n_lam),
                )
            )
        else:
            p.finish(
                'Done! | Received {:.1%} of {:.3g} photons ({:.1%})'.format(
                    np.sum(n_photons_out_tot[0, ...])
                    / float(np.sum(n_photons_in_tot)),
                    np.sum(n_photons_in_tot),
                    np.sum(n_photons_in_tot) / float(n_photons),
                )
            )

        if wavelength.scalar:
            output = drop_axes(output, 'wavelength')
            output.attrs['wavelength'] = wavelength[:]

        if not self.autoinit and not self.keep_context:
            assert self.ctx is not None
            self.ctx.pop()
            self.ctx.detach()
            self.ctx = None
            from pycuda.tools import clear_context_caches

            clear_context_caches()

        return output


def _calc_solid_angles(
    n_theta: int,
    n_phi: int,
    sza_max: float = 90.0,
    sun_disc: float = 0,
) -> tuple:
    """
    Compute zenith angles, azimuth angles, and solid angles for the
    sensor grid.

    Parameters
    ----------
    n_theta : int
        Number of zenith angle bins.
    n_phi : int
        Number of azimuth angle bins.
    sza_max : float, optional
        Maximum zenith angle in degrees. Default is ``90.``.
    sun_disc : float, optional
        Half-angle of the solar disc in degrees. When non-zero, all
        solid angles are set to the solid angle subtended by the solar
        disc. Default is ``0``.

    Returns
    -------
    tab_th : numpy.ndarray
        Array of shape ``(n_theta,)`` containing the zenith angles in
        radians, centred within each bin.
    tab_phi : numpy.ndarray
        Array of shape ``(n_phi,)`` containing the azimuth angles in
        radians, starting at ``0`` and spaced by ``2π / n_phi``.
    tab_omega : numpy.ndarray
        Array of shape ``(n_theta,)`` containing the normalized solid
        angles. When ``sun_disc != 0``, all elements are set to the
        solid angle of the solar disc ``2π(1 - cos(sun_disc))``.
    """

    # zenith angles
    dth = (sza_max / 180.0 * np.pi) / n_theta
    tab_th = np.linspace(
        dth / 2, sza_max / 180.0 * np.pi - dth / 2, n_theta, dtype='float64'
    )

    # azimuth angles
    dphi = 2 * np.pi / n_phi
    tab_phi = np.linspace(0.0, 2 * np.pi - dphi, n_phi, dtype='float64')

    # solid angles
    tab_ds = np.sin(tab_th) * dth * dphi

    # normalize to 1
    tab_omega = tab_ds / (sum(tab_ds) * n_phi)
    if sun_disc != 0:
        tab_omega[:] = 2 * np.pi * (1.0 - np.cos(sun_disc * np.pi / 180))

    return tab_th, tab_phi, tab_omega


def _add_variable(ds, name, data, dims, attrs=None):
    """
    Attach an array to the Dataset under explicit dimension names.

    Mirror of the legacy MLUT.add_dataset: the variable binds to the
    dataset coordinates by dimension name only, so no index alignment
    can ever occur, and a size mismatch raises immediately.
    """
    ds[name] = xr.Variable(dims, data, attrs=attrs)


def _add_level_output(
    ds,
    direction,
    lvl,
    axnames,
    tab_final,
    tab_final_no_aer,
    n_photons_out_tot,
    n_photons_out_tot_no_aer,
    sigma,
    tab_dist_final,
    zip_flag,
    cdist_axnames_zip,
    cdist_axnames_full,
    isen,
    ilam,
    iphi,
    no_aer_output,
):
    """
    Add the radiometric variables of one output level to the Dataset.

    The variables are named after the level direction, for instance
    'I_up (TOA)' or 'N_down (0+)', following the historical MLUT
    dataset names.
    """
    for i, stk in enumerate(('I', 'Q', 'U', 'V')):
        _add_variable(
            ds,
            f'{stk}_{direction}',
            tab_final[lvl, i, isen, ilam, iphi, :],
            axnames,
        )
    if sigma is not None:
        for i, stk in enumerate(('I', 'Q', 'U', 'V')):
            _add_variable(
                ds,
                f'{stk}_stdev_{direction}',
                sigma[lvl, i, isen, ilam, iphi, :],
                axnames,
            )
    _add_variable(
        ds,
        f'N_{direction}',
        n_photons_out_tot[lvl, isen, ilam, iphi, :],
        axnames,
    )
    if no_aer_output:
        for i, stk in enumerate(('I', 'Q', 'U', 'V')):
            _add_variable(
                ds,
                f'{stk}_{direction}, no_aer',
                tab_final_no_aer[lvl, i, isen, ilam, iphi, :],
                axnames,
            )
        _add_variable(
            ds,
            f'N_{direction}, no_aer',
            n_photons_out_tot_no_aer[lvl, isen, ilam, iphi, :],
            axnames,
        )
    if len(tab_dist_final) > 1:
        if zip_flag:
            _add_variable(
                ds,
                f'cdist_{direction}',
                np.squeeze(tab_dist_final[lvl, :, isen]),
                cdist_axnames_zip,
            )
        else:
            _add_variable(
                ds,
                f'cdist_{direction}',
                tab_dist_final[lvl, :, isen],
                cdist_axnames_full,
            )


def _finalize(
    tab_photons_tot: np.ndarray,
    tab_photons_tot_no_aer: np.ndarray,
    tab_dist_tot: np.ndarray,
    tab_hist_tot,
    wavelength: np.ndarray,
    n_photons_in_tot: np.ndarray,
    errorcount: GPUArray,
    n_photons_out_tot: np.ndarray,
    n_photons_out_tot_no_aer: np.ndarray,
    output_layers: int,
    tab_trans_dir: np.ndarray,
    tab_trans_dir_analytic: np.ndarray | None,
    attrs: dict,
    prof_atm,
    prof_oc,
    sigma: np.ndarray | None,
    horiz: int,
    le: dict | None = None,
    flux: str | None = None,
    back: bool = False,
    sza_max: float = 90.0,
    sun_disc: float = 0,
    hist: bool = False,
    c_mat_visu_recep: np.ndarray | None = None,
    dic_stp: dict | None = None,
    mat_cats: np.ndarray | None = None,
    mat_loss: np.ndarray | None = None,
    w_ph_cats: np.ndarray | None = None,
    w_ph_cats2: np.ndarray | None = None,
    no_aer_output: bool = False,
) -> xr.Dataset:
    """
    Create and return the final output of a simulation.

    Parameters
    ----------
    tab_photons_tot : np.ndarray
        Accumulated photon weights of shape (level, stk, sensor, lam,
        theta, phi).
    tab_photons_tot_no_aer : np.ndarray
        Same as tab_photons_tot but without the aerosol scattering
        contributions (see the no_aer_output option of run).
    tab_dist_tot : np.ndarray
        Accumulated ALIS path-length distances.
    tab_hist_tot : np.ndarray | None
        Accumulated photon histories (hist mode).
    wavelength : np.ndarray
        The wavelengths in nm.
    n_photons_in_tot : pycuda.gpuarray.GPUArray
        Number of launched photons per sensor and wavelength.
    errorcount : np.ndarray
        Kernel error counters.
    n_photons_out_tot : np.ndarray
        Number of photons counted in each output box.
    n_photons_out_tot_no_aer : np.ndarray
        Same as n_photons_out_tot without the aerosol scattering
        contributions.
    output_layers : int
        The output layers flag of run.
    tab_trans_dir : np.ndarray
        Direct transmission accumulated by the kernel.
    tab_trans_dir_analytic : np.ndarray | None
        Analytic direct (Beer-Lambert) transmission.
    attrs : dict
        Attributes to attach to the output Dataset.
    prof_atm, prof_oc : xr.Dataset | None
        Atmospheric and oceanic profiles, stored in the output.
    sigma : np.ndarray | None
        Standard deviation estimate (stdev mode).
    horiz : int
        Horizontal irradiance normalization flag.
    le : dict | None, optional
        The local estimate dictionary of run.
    flux : str | None, optional
        The flux mode of run ('planar', 'spherical', ...).
    back : bool, optional
        Backward mode flag.
    sza_max : float, optional
        Maximum solar zenith angle in degrees (sun_disc mode).
    sun_disc : int, optional
        Sun discretization flag of run.
    hist : bool, optional
        If True, add the photon-history datasets.
    c_mat_visu_recep : np.ndarray | None, optional
        Receiver visualization matrix (3D-object mode).
    dic_stp : dict | None, optional
        Solar Tower Power parameters (3D-object mode).
    mat_cats, mat_loss : np.ndarray | None, optional
        Receiver category and optical-loss matrices (3D-object mode).
    w_ph_cats, w_ph_cats2 : np.ndarray | None, optional
        Receiver photon weights (and their squares) per category.
    no_aer_output : bool, optional
        If True, add the no-aerosol variables to the output.

    Returns
    -------
    xr.Dataset
        The simulation results.
    """
    if hasattr(prof_atm, 'to_xarray'):
        prof_atm = prof_atm.to_xarray()
    if hasattr(prof_oc, 'to_xarray'):
        prof_oc = prof_oc.to_xarray()

    (_, _, n_sensor, n_lam, n_theta, n_phi) = tab_photons_tot.shape

    # normalization in case of radiance
    # (broadcast everything to dimensions
    # (LVL,n_pstk,SENSOR,LAM,THETA,PHI))
    norm_npho = n_photons_in_tot.reshape((1, 1, n_sensor, n_lam, 1, 1))
    zip_flag = False
    if flux is None:
        if le is not None:
            tab_th = le['th']
            tab_phi = le['phi']
            if 'zip' not in le.keys():
                zip_flag = False
            else:
                zip_flag = le['zip']
            norm_geo = 1.0
        else:
            tab_th, tab_phi, tab_omega = _calc_solid_angles(
                n_theta, n_phi, sza_max=sza_max, sun_disc=sun_disc
            )
            if horiz == 1:
                norm_geo = (
                    2.0
                    * tab_omega.reshape((1, 1, -1, 1))
                    * np.cos(tab_th).reshape((1, 1, -1, 1))
                )
            else:
                norm_geo = 2.0 * tab_omega.reshape((1, 1, -1, 1))
    else:
        norm_geo = 1.0
        tab_th, tab_phi, _ = _calc_solid_angles(
            n_theta, n_phi, sza_max=sza_max, sun_disc=sun_disc
        )

    # normalization
    tab_final = tab_photons_tot.astype('float64') / (norm_geo * norm_npho)
    tab_final_no_aer = tab_photons_tot_no_aer.astype('float64') / (
        norm_geo * norm_npho
    )
    tab_dist_final = tab_dist_tot.astype('float64')
    # if hist : tab_hist_final = tab_hist_tot

    # swapaxes : (th, phi) -> (phi, theta)
    tab_final = tab_final.swapaxes(4, 5)
    tab_final_no_aer = tab_final_no_aer.swapaxes(4, 5)
    if len(tab_dist_final) > 1:
        tab_dist_final = tab_dist_final.swapaxes(3, 4)
    if hist:
        tab_hist_tot = tab_hist_tot.swapaxes(3, 4)
    n_photons_out_tot = n_photons_out_tot.swapaxes(3, 4)
    n_photons_out_tot_no_aer = n_photons_out_tot_no_aer.swapaxes(3, 4)
    if sigma is not None:
        sigma /= norm_geo
        sigma = sigma.swapaxes(4, 5)

    #
    # create the output Dataset
    #
    ds = xr.Dataset()

    # add the axes
    axnames = ['Zenith angles']
    if hist:
        _add_variable(
            ds, 'Nphotons_in', n_photons_in_tot,
            ('sensor_in', 'wavelength_in'),
        )

    iphi = slice(None)
    ds.attrs['zip'] = 'False'
    ds.attrs['NPhotonIn_sum'] = np.sum(n_photons_in_tot)

    if le is not None:
        ds.attrs['LE'] = int(1)
    else:
        ds.attrs['LE'] = int(0)

    if le is not None:
        if 'zip' in le:
            if le['zip']:
                ds.attrs['zip'] = 'True'
                iphi = 0
            else:
                axnames.insert(0, 'Azimuth angles')
        else:
            axnames.insert(0, 'Azimuth angles')
    else:
        axnames.insert(0, 'Azimuth angles')

    ds.coords['Zenith angles'] = tab_th * 180.0 / np.pi
    ds.coords['Azimuth angles'] = tab_phi * 180.0 / np.pi

    axnames4 = []
    if n_lam > 1:
        ds.coords['wavelength'] = wavelength
        ilam = slice(None)
        axnames.insert(0, 'wavelength')
        axnames4.insert(0, 'wavelength')
    else:
        ds.attrs['wavelength'] = str(wavelength)
        ilam = 0

    if n_sensor > 1:
        ds.coords['sensor index'] = np.arange(n_sensor)
        isen = slice(None)
        axnames.insert(0, 'sensor index')
        axnames4.insert(0, 'sensor index')
    else:
        isen = 0

    write_uptoa = output_layers in (0, 1, 2, 3, 7)
    write_down0p = output_layers in (1, 3, 4, 7)
    write_down0m = output_layers in (2, 3, 5, 6)
    write_up0p = output_layers in (2, 3, 5, 6)
    write_up0m = output_layers in (1, 3, 4)
    write_downb = output_layers in (2, 3, 5)

    # Build dimension names for cdist variables (ALIS mode)
    # Shape after swapaxes: (n_lvl, N_LAYERS, n_sensor, n_phi,
    # n_theta, [NSCL,] n_iamf)
    # When NSCL=1, squeeze it away for backward compatibility
    if len(tab_dist_final) > 1 and tab_dist_final.shape[-2] == 1:
        tab_dist_final = tab_dist_final[..., 0, :]  # remove trivial NSCL dim
        _has_scl = False
    elif len(tab_dist_final) > 1:
        _has_scl = True
    else:
        _has_scl = False

    if _has_scl:
        cdist_axnames_zip = ['cdist_layer', 'Zenith angles', 'iSCL', 'iAMF']
        cdist_axnames_full = ['cdist_layer']
        if n_sensor > 1:
            cdist_axnames_full.append('sensor index')
        cdist_axnames_full.extend(
            ['Azimuth angles', 'Zenith angles', 'iSCL', 'iAMF']
        )
    else:
        cdist_axnames_zip = ['cdist_layer', 'Zenith angles', 'iAMF']
        cdist_axnames_full = ['cdist_layer']
        if n_sensor > 1:
            cdist_axnames_full.append('sensor index')
        cdist_axnames_full.extend(
            ['Azimuth angles', 'Zenith angles', 'iAMF']
        )

    level_kwargs = {
        'axnames': axnames,
        'tab_final': tab_final,
        'tab_final_no_aer': tab_final_no_aer,
        'n_photons_out_tot': n_photons_out_tot,
        'n_photons_out_tot_no_aer': n_photons_out_tot_no_aer,
        'sigma': sigma,
        'tab_dist_final': tab_dist_final,
        'zip_flag': zip_flag,
        'cdist_axnames_zip': cdist_axnames_zip,
        'cdist_axnames_full': cdist_axnames_full,
        'isen': isen,
        'ilam': ilam,
        'iphi': iphi,
        'no_aer_output': no_aer_output,
    }

    if write_uptoa:
        _add_level_output(ds, 'up (TOA)', UPTOA, **level_kwargs)

    if hist:
        _add_variable(
            ds,
            'histories',
            tab_hist_tot,
            (
                'hist_level',
                'hist_photon',
                'hist_record',
                'hist_theta',
                'hist_sensor',
                'hist_phi',
            ),
        )

    if write_down0p:
        _add_level_output(ds, 'down (0+)', DOWN0P, **level_kwargs)
    if write_up0m:
        _add_level_output(ds, 'up (0-)', UP0M, **level_kwargs)
    if write_down0m:
        _add_level_output(ds, 'down (0-)', DOWN0M, **level_kwargs)
    if write_up0p:
        _add_level_output(ds, 'up (0+)', UP0P, **level_kwargs)
    if write_downb:
        _add_level_output(ds, 'down (B)', DOWNB, **level_kwargs)

    # write atmospheric profiles
    if prof_atm is not None:
        # direct transmission
        if tab_trans_dir_analytic is not None:
            _add_variable(
                ds,
                'direct transmission',
                tab_trans_dir_analytic,
                ['wavelength'],
            )
        _add_variable(
            ds,
            'direct transmission (dev)',
            np.exp(-tab_trans_dir[isen, ilam]),
            axnames4,
        )

        for axis_name in prof_atm.coords:
            if axis_name not in ds.coords:
                coord = prof_atm.coords[axis_name]
                ds.coords[axis_name] = (coord.dims, coord.to_numpy())

        for name in [
            'n_atm',
            'T_atm',
            'OD_r',
            'OD_p',
            'OD_g',
            'OD_atm',
            'OD_sca_atm',
            'OD_abs_atm',
            'pmol_atm',
            'ssa_atm',
            'ssa_p_atm',
        ]:
            da = prof_atm[name]
            _add_variable(ds, name, da.to_numpy(), da.dims, attrs=da.attrs)
        if 'phase_atm' in prof_atm.data_vars:
            for name in ['phase_atm', 'iphase_atm']:
                da = prof_atm[name]
                # Rename 'iphase' and 'nphamat' per domain to
                # avoid sharing dimensions across atm/oc ('stk' is
                # the legacy name of the nphamat dimension)
                dims = [
                    'phase_index_atm' if d == 'iphase'
                    else 'nphamat_atm' if d in ('nphamat', 'stk')
                    else d
                    for d in da.dims
                ]
                _add_variable(ds, name, da.to_numpy(), dims, attrs=da.attrs)
        if 'pine_atm' in prof_atm.data_vars:
            for name in ['pine_atm', 'FQY1_atm']:
                da = prof_atm[name]
                dims = [
                    'phase_index_atm' if d == 'iphase'
                    else 'nphamat_atm' if d in ('nphamat', 'stk')
                    else d
                    for d in da.dims
                ]
                _add_variable(ds, name, da.to_numpy(), dims, attrs=da.attrs)

        if 'neighbour_atm' in prof_atm.data_vars:
            for name in [
                'iopt_atm',
                'iabs_atm',
                'pmin_atm',
                'pmax_atm',
                'neighbour_atm',
            ]:
                da = prof_atm[name]
                _add_variable(
                    ds, name, da.to_numpy(), da.dims, attrs=da.attrs
                )

    # write ocean profiles
    if prof_oc is not None:
        for axis_name in prof_oc.coords:
            if axis_name not in ds.coords:
                coord = prof_oc.coords[axis_name]
                ds.coords[axis_name] = (coord.dims, coord.to_numpy())

        for name in [
            'T_oc',
            'OD_w',
            'OD_p_oc',
            'OD_y',
            'OD_oc',
            'OD_sca_oc',
            'OD_abs_oc',
            'pmol_oc',
            'ssa_oc',
            'albedo_seafloor',
        ]:
            da = prof_oc[name]
            _add_variable(ds, name, da.to_numpy(), da.dims, attrs=da.attrs)
        if 'ssa_w' in prof_oc.data_vars:
            da = prof_oc['ssa_w']
            _add_variable(ds, 'ssa_w', da.to_numpy(), da.dims, attrs=da.attrs)
        if 'ssa_p_oc' in prof_oc.data_vars:
            da = prof_oc['ssa_p_oc']
            _add_variable(
                ds, 'ssa_p_oc', da.to_numpy(), da.dims, attrs=da.attrs
            )
        if 'phase_oc' in prof_oc.data_vars:
            for name in ['phase_oc', 'iphase_oc']:
                da = prof_oc[name]
                # Rename 'iphase' and 'nphamat' per domain to
                # avoid sharing dimensions across atm/oc ('stk' is
                # the legacy name of the nphamat dimension)
                dims = [
                    'phase_index_oc' if d == 'iphase'
                    else 'nphamat_oc' if d in ('nphamat', 'stk')
                    else d
                    for d in da.dims
                ]
                _add_variable(ds, name, da.to_numpy(), dims, attrs=da.attrs)
        if 'pine_oc' in prof_oc.data_vars:
            for name in ['pine_oc', 'FQY1_oc']:
                da = prof_oc[name]
                dims = [
                    'phase_index_oc' if d == 'iphase'
                    else 'nphamat_oc' if d in ('nphamat', 'stk')
                    else d
                    for d in da.dims
                ]
                _add_variable(ds, name, da.to_numpy(), dims, attrs=da.attrs)

        if 'neighbour_oc' in prof_oc.data_vars:
            for name in [
                'iopt_oc',
                'iabs_oc',
                'pmin_oc',
                'pmax_oc',
                'neighbour_oc',
            ]:
                da = prof_oc[name]
                _add_variable(
                    ds, name, da.to_numpy(), da.dims, attrs=da.attrs
                )

    # write the error count
    err = errorcount.get()
    for i, d in enumerate(
        [
            'ERROR_THETA',
            'ERROR_CASE',
            'ERROR_VXY',
            'ERROR_MAX_LOOP',
        ]
    ):
        ds.attrs[d] = err[i]

    # write attributes
    for k, v in list(attrs.items()):
        ds.attrs[k] = str(v)

    # fluxes post-processing
    if flux is not None:
        ds.attrs['flux'] = flux
        for d in list(map(str, ds.data_vars)):
            if (
                ('_stdev_' in d)
                or (d.startswith('Q_'))
                or (d.startswith('U_'))
                or (d.startswith('V_'))
            ):
                ds = ds.drop_vars([d])
            elif d.startswith('I_') or d.startswith('N_'):
                flux_var = (
                    ds[d]
                    .sum(dim='Azimuth angles', keep_attrs=True)
                    .sum(dim='Zenith angles', keep_attrs=True)
                )
                ds = ds.drop_vars([d])
                ds[d.replace('I_', 'flux_')] = flux_var.variable

    if c_mat_visu_recep is not None:
        # a receiver implies the Solar Tower Power parameters
        assert dic_stp is not None
        # Indice 0 = Sum of all Cats, then cat1 to cat8, def of cats ->
        # see Moulana et al, 2019
        ds.coords['Categories'] = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32
        )
        var_x, var_y = np.shape(c_mat_visu_recep[0][:][:])
        x_indices = np.arange(var_x)
        y_indices = np.arange(var_y)
        ds.coords['X_Cell_Index'] = x_indices
        ds.coords['Y_Cell_Index'] = y_indices
        _add_variable(
            ds,
            'C_Receiver',
            c_mat_visu_recep[:][:][:],
            ['Categories', 'X_Cell_Index', 'Y_Cell_Index'],
        )
        ds.attrs['S_Receiver'] = str(
            dic_stp["SREC"]
        )  # Receiver surface in km²
        ds.attrs['S_Cell'] = str(dic_stp["TC"])  # Cell surface in km²
        # half-angle of the receiver solid angle
        if back:
            ds.attrs['ALDEG'] = str(dic_stp["receiver_fov"])
        else:
            ds.attrs['ALDEG'] = str(90)

    if mat_cats is not None:
        assert dic_stp is not None
        assert w_ph_cats is not None and w_ph_cats2 is not None
        _add_variable(ds, 'cat_PhNb', mat_cats[:, 0], ['Categories'])
        _add_variable(ds, 'cat_w', mat_cats[:, 1], ['Categories'])
        _add_variable(ds, 'cat_w2', mat_cats[:, 2], ['Categories'])
        _add_variable(ds, 'cat_irr', mat_cats[:, 3], ['Categories'])
        _add_variable(ds, 'cat_errAbs', mat_cats[:, 4], ['Categories'])
        _add_variable(ds, 'cat_err%', mat_cats[:, 5], ['Categories'])

        arr_wc = np.zeros((9, n_lam), dtype=np.float64)
        arr_wc2 = np.zeros((9, n_lam), dtype=np.float64)

        arr_wc[0, ilam] = np.sum(w_ph_cats[:, ilam], axis=0)
        arr_wc[1:, ilam] = w_ph_cats[:, ilam]

        arr_wc2[0, ilam] = np.sum(w_ph_cats2[:, ilam], axis=0)
        arr_wc2[1:, ilam] = w_ph_cats2[:, ilam]

        axe_w_ph = ['Categories']
        if n_lam > 1:
            axe_w_ph.append('wavelength')

        _add_variable(ds, 'wPhCats', arr_wc[:, ilam], axe_w_ph)
        _add_variable(ds, 'wPhCats2', arr_wc2[:, ilam], axe_w_ph)
        _add_variable(
            ds, 'norm_npho', norm_npho[0, 0, 0, :, 0, 0], ['wavelength']
        )

        ds.attrs['n_cte'] = str(dic_stp["n_cte"])

    if mat_loss is not None:
        assert dic_stp is not None
        assert prof_atm is not None
        _add_variable(
            ds, 'wLoss', np.array(mat_loss[:, 0], dtype=np.float64), ['index']
        )
        _add_variable(
            ds, 'wLoss2', np.array(mat_loss[:, 1], dtype=np.float64), ['index']
        )
        ds.attrs['n_cos'] = str(dic_stp["n_cos"])

        # To consider also the multispectral case
        if n_lam > 1:
            n_wavelength = len(wavelength)
        else:
            n_wavelength = 1

        # ======== Find the extinction between TOA and heliostats
        tau_ext = np.zeros(n_wavelength, dtype=np.float64)
        tr_tau = np.zeros(n_wavelength, dtype=np.float64)
        p_pyt = np.zeros(n_wavelength, dtype=np.float64)

        # find the atm layer where the mean heliostats z altitude is
        # located
        if 'z_atm' not in prof_atm.coords:
            raise ValueError(
                "the STP optical efficiencies require a 1D "
                "atmosphere profile (the 3D profile has no z_atm "
                "axis)"
            )
        ci = 0
        zatm = prof_atm.coords['z_atm'].to_numpy()
        od_atm = prof_atm['OD_atm'].to_numpy()
        while zatm[ci] > dic_stp["MZAlt_H"]:
            ci += 1

        for i in range(0, n_wavelength):
            tau_ext[i] = (od_atm[i, ci] - od_atm[i, ci - 1]) * (
                dic_stp["MZAlt_H"] / zatm[ci - 1]
            )
            tau_ext[i] = od_atm[i, ci] - tau_ext[i]
            # Beer-Lamber law to find the transmisttance
            tr_tau[i] = np.exp(-abs(tau_ext[i] / -dic_stp["vSun"].z))
            # theoric computation of the total power collected by all
            # the heliostats
            p_pyt[i] = (
                tr_tau[i] * dic_stp["totS_H"] * 1e6
            )  # mult by 1e6 to convert km² to m²
        # Save results
        _add_variable(ds, 'n_tr', tr_tau, ['wavelength'])
        _add_variable(ds, 'powc_H', p_pyt, ['wavelength'])
        # ========

        # === Here allows the calculation of the analytical approx of
        # n_atm in backward ->
        if (
            back
            and (dic_stp["LPH"] is not None)
            and (dic_stp["LPR"] is not None)
        ):
            naatm = np.zeros(n_wavelength, dtype=np.float64)
            p = make_progress(n_wavelength - 1, dic_stp["prog"])
            for j in range(0, n_wavelength):
                sum_naatm = 0
                p.update(
                    j + 1,
                    'n_aatm computed : {:.3g} / {:.3g}'.format(
                        j + 1, n_wavelength
                    ),
                )
                for i in range(len(dic_stp["LPH"])):
                    sum_naatm += _find_extinction(
                        dic_stp["LPH"][i], dic_stp["LPR"][0], prof_atm, j
                    )
                naatm[j] = sum_naatm / len(dic_stp["LPH"])
            p.finish(
                'Done! | Analytic approx of n_atm computed for '
                '{:.3g} wavelengths'.format(
                    n_wavelength
                )
            )
            _add_variable(ds, 'n_aatm', naatm, ['wavelength'])
        # ===

    return ds


def _isotropic(
    n_theta: int,
    ang_a: NDArray[np.float64] | None = None,
) -> np.ndarray:
    """
    Build the isotropic phase-function lookup table.

    Computes a uniform phase matrix with cumulative distribution
    function sampling over scattering angles.

    Parameters
    ----------
    n_theta : int
        Theta discretization used to build the sampling lookup tables.
        In CUDA, phase values are sampled over this angular
        discretization. A finer angular discretization improves sampling
        precision but increases GPU memory usage.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_theta,)`` and dtype ``TYPE_PHASE``. Contains
        the isotropic phase-function lookup table ready to be indexed by
        phase lookup routines.

    Warnings
    --------
    This function has not been validated yet.
    """
    phase_H = np.zeros(n_theta, dtype=TYPE_PHASE, order='C')
    angles = np.linspace(
        0.0, pi, int(n_theta), endpoint=True, dtype=np.float64
    )
    norm = 0.5
    phase = np.zeros((4, n_theta), dtype='float64')
    phase[0, :] = 0.5 / norm
    phase[1, :] = 0.5 / norm
    phase[2, :] = 0.5 / norm
    phase[3, :] = 0.5 / norm

    angN = _uniform_angles(n_theta) if ang_a is None else ang_a
    f1 = interp1d(angles, phase[0, :])
    f2 = interp1d(angles, phase[1, :])
    f3 = interp1d(angles, phase[2, :])
    f4 = interp1d(angles, phase[3, :])

    # parameters equally spaced in scattering angle [0, 180]
    phase_H['a_P11'][:] = f1(angN)  # I par P11
    phase_H['a_P22'][:] = f2(angN)  # I per P22
    phase_H['a_P33'][:] = f3(angN)  # U P33
    phase_H['a_P43'][:] = f4(angN)  # V P43
    phase_H['a_P44'][:] = f3(angN)  # V P44=P33

    return phase_H
    return phase_H, cdf


def _rayleigh(
    n_theta: int,
    depo: float,
    polarization: bool = True,
    ang_a: NDArray[np.float64] | None = None,
) -> np.ndarray:
    """
    Build the Rayleigh phase-function lookup table.

    Computes the Rayleigh phase matrix (polarized or scalar) with
    cumulative distribution function sampling over scattering angles.

    Parameters
    ----------
    n_theta : int
        Theta discretization used to build the sampling lookup tables.
        In CUDA, phase values are sampled over this angular
        discretization. A finer angular discretization improves sampling
        precision but increases GPU memory usage.
    depo : float
        Molecular depolarization factor. Generates the Rayleigh phase
        entry. If negative, an isotropic phase function is used instead
        of Rayleigh.
    polarization : bool, optional
        If ``False``, build scalar-equivalent phase tables with
        polarization disabled. If ``True``, keep the polarized phase-
        matrix terms required by the vector radiative transfer kernels.
        Default is ``True``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_theta,)`` and dtype ``TYPE_PHASE``. Contains
        the Rayleigh phase-function lookup table ready to be indexed by
        phase lookup routines.
    """
    pha = np.zeros(n_theta, dtype=TYPE_PHASE, order='C')

    gama = depo / (2 - depo)
    delta = np.float32((1.0 - gama) / (1.0 + 2.0 * gama))
    delta_prim = np.float32(gama / (1.0 + 2.0 * gama))
    beta = np.float32(3.0 / 2.0 * delta_prim)
    alpha = np.float32(1.0 / 8.0 * delta)
    a_coeff = np.float32(1.0 + beta / (3.0 * alpha))

    theta_le = (
        np.linspace(0.0, pi, int(n_theta), endpoint=True, dtype=np.float64)
        if ang_a is None
        else ang_a
    )
    c_th_le = np.cos(theta_le)
    c_th2_le = c_th_le * c_th_le

    delta_seco = np.float32((1.0 - 3.0 * gama) / (1.0 - gama))
    t_half = 3.0 / 2.0
    p22 = t_half * (delta + delta_prim)
    p12 = t_half * delta_prim
    p33bis = t_half * delta
    p44bis = p33bis * delta_seco

    if not polarization:
        # P(theta) -> phase matrix in Iperpar convention
        # F(theta) -> phase matrix in IQUV convention
        # from IQUV to IperIpar (in the case only IQUV F11 != 0 i.e. no
        # polarisation)
        # a_p11 = ((3./8.)*delta*(c_th2_le[:]-1)) + 0.5
        a_p11 = t_half * (delta * c_th2_le[:] + delta_prim)
        a_f11 = 0.5 * (a_p11 + 2 * p12 + p22)

        pha['a_P11'][:] = 0.5 * a_f11
        pha['a_P12'][:] = 0.5 * a_f11
        pha['a_P22'][:] = 0.5 * a_f11
    else:
        # parameters equally spaced in scattering angle [0, 180]
        pha['a_P11'][:] = t_half * (delta * c_th2_le[:] + delta_prim)
        pha['a_P12'][:] = p12
        pha['a_P22'][:] = p22
        pha['a_P33'][:] = p33bis * c_th_le[:]  # U
        pha['a_P44'][:] = p44bis * c_th_le[:]  # V

    return pha


def _uniform_angles(n: int) -> NDArray[np.float64]:
    """Equally spaced angles over ``[0, pi]``, the historical grid."""
    return np.arange(n, dtype='float64') / (n - 1) * np.pi


def _resolve_agrid(
    theta_grid, n_icdf: int, profile, kind: str
) -> tuple[NDArray[np.float64], tuple]:
    """
    Pick the angular grid of the equal-angle half of a phase table.

    The grid is classified into the cheapest device lookup that
    reproduces it exactly, and returned in its canonical form so that
    the table and the kernel always agree on the node positions.

    Parameters
    ----------
    theta_grid : None, str or array_like
        ``None`` for the equally spaced grid of *n_icdf* angles, the
        string ``'phase'`` to adopt the grid the phase matrix already
        carries, one of ``smartg.phase.THETA_GRID_KINDS`` to generate
        *n_icdf* angles of that kind, or an explicit array of angles in
        degrees.
    n_icdf : int
        Number of angles, ignored when the grid comes from the phase
        matrix or is given explicitly.
    profile : xr.Dataset or None
        Optical profile holding the ``theta_<kind>`` coordinate.
    kind : str
        ``'atm'`` or ``'oc'``.

    Returns
    -------
    ang : ndarray
        The angles in radians, from 0 to pi.
    agrid : tuple
        ``(n, mode, ang_gpu)``, where *mode* is the device lookup
        (0 equally spaced, 1 tabulated) and *ang_gpu* the device copy
        of *ang* for mode 1, else ``None``.
    """
    if theta_grid is None:
        ang = _uniform_angles(n_icdf)
    elif isinstance(theta_grid, str) and theta_grid == 'phase':
        name = 'theta_' + kind
        if profile is not None and name in profile.coords:
            ang = profile.coords[name].to_numpy() * np.pi / 180.0
        else:
            # no tabulated phase matrix: only the analytic molecular
            # phase functions are needed, on any grid
            ang = _uniform_angles(n_icdf)
    elif isinstance(theta_grid, str):
        if theta_grid not in THETA_GRID_KINDS:
            raise ValueError(
                "Choices for the theta_grid parameter are: 'phase', "
                f"{THETA_GRID_KINDS}, or an array of angles in "
                f"degrees, got {theta_grid!r}."
            )
        ang = _make_theta_grid(n_icdf, theta_grid, unit='rad')
    else:
        ang = np.deg2rad(np.asarray(theta_grid, dtype='float64'))

    ang = np.ascontiguousarray(ang, dtype='float64')
    n = len(ang)
    if n < 2:
        raise ValueError(f"The angular grid needs >= 2 angles, got {n}.")
    if np.any(np.diff(ang) <= 0):
        raise ValueError("The angular grid must be strictly increasing.")
    if not (np.isclose(ang[0], 0.0) and np.isclose(ang[-1], np.pi)):
        raise ValueError(
            "The angular grid must span 0 to 180 degrees, got "
            f"{np.rad2deg(ang[0])} to {np.rad2deg(ang[-1])}."
        )

    # classify, and snap onto the canonical nodes of the chosen mode so
    # that the table and the kernel cannot disagree
    canonical = _uniform_angles(n)
    if np.allclose(ang, canonical, rtol=0.0, atol=1e-9):
        return canonical, (n, 0, None)

    return ang, (n, 1, to_gpu(ang.astype('float32')))


def _agrid_struct(agrid) -> np.void:
    """
    Pack a phase-table angle grid into its ``TYPE_AGRID`` record.

    Parameters
    ----------
    agrid : tuple
        ``(n, mode, ang_gpu)`` as returned by ``_resolve_agrid``, where
        *ang_gpu* is the device copy of the angle axis for mode 1, and
        ``None`` otherwise.

    Returns
    -------
    numpy.void
        A single ``TYPE_AGRID`` record ready to be copied to the
        ``AGAERd`` / ``AGOCEd`` device constants.
    """
    n, mode, ang_gpu = agrid
    rec = np.zeros(1, dtype=TYPE_AGRID)
    rec['n'] = n
    rec['mode'] = mode
    # trip count of the tabulated binary search
    rec['log2n'] = int(np.floor(np.log2(n - 2))) if n > 2 else 0
    rec['ang'] = 0 if ang_gpu is None else int(ang_gpu.gpudata)
    return rec[0]


def _pgrid_struct(pgrid) -> np.void:
    """
    Pack a cumulative distribution into its ``TYPE_PGRID`` record.

    Parameters
    ----------
    pgrid : tuple
        ``(n, cdf_gpu)``, where *cdf_gpu* is the device copy of the
        cumulative distribution at the ``n`` nodes of the angle grid.

    Returns
    -------
    numpy.void
        A single ``TYPE_PGRID`` record ready to be copied to the
        ``PGAERd`` / ``PGOCEd`` device constants.

    Notes
    -----
    Only the address of *cdf_gpu* is copied to the device, so nothing
    on the device keeps that allocation alive: the caller has to hold
    a reference to it for as long as the kernel runs.
    """
    n, cdf_gpu = pgrid
    rec = np.zeros(1, dtype=TYPE_PGRID)
    rec['n'] = n
    # trip count of the binary search, as for the angle grid
    rec['log2n'] = int(np.floor(np.log2(n - 2))) if n > 2 else 0
    rec['cdf'] = int(cdf_gpu.gpudata)
    return rec[0]


def _calc_phase_host(
    profile,
    n_theta: int,
    depo: float,
    kind: str,
    polarization: bool = True,
    ang_a: NDArray[np.float64] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the phase-function lookup table uploaded to the GPU.

    This routine converts the phase information stored in an atmospheric
    or oceanic profile into the structured ``TYPE_PHASE`` table expected
    by the CUDA kernels. The returned table always reserves:

    - index 0 for the molecular phase function (Rayleigh, or isotropic
      when
        ``depo < 0``),
    - index 1 for the VRS phase function,
    - subsequent indices for the tabulated particle phase functions
      found in
        ``phase_<kind>``.

    For each phase entry, two tables are precomputed on the same
    scattering angle grid over [0, pi]:

    - the ``a_*`` fields of the phase matrix, which every reader of
      a phase matrix goes through,
    - its cumulative distribution at those very nodes, integrated
      exactly for the tabulated matrix (F11 linear in theta between
      nodes, times the true sin(theta)), which a random walk draws
      its deflection from by inverting one bin in closed form. The
      drawn deflection is therefore distributed exactly as the
      matrix the walk then reads, whatever the grid.

    The profile phase matrices are first normalized to the internal
    I-parallel/I-perpendicular representation with
    ``convert_phase_to_iparper``. When ``polarization`` is disabled,
    tabulated phase matrices are reduced to their scalar intensity
    equivalent before the lookup tables are built.

    Parameters
    ----------
    profile : xr.Dataset
        Atmospheric or oceanic optical profile.
    n_theta : int
        Theta discretization used to build the sampling lookup tables.
        In CUDA, phase values are sampled over this ``n_theta`` angular
        discretization. A finer angular discretization improves sampling
        precision but increases GPU memory usage.
    depo : float
        Molecular depolarization factor used to generate the Rayleigh
        phase entry. If negative, an isotropic phase function is used
        instead of Rayleigh.
    kind : str
        Profile family identifier. Must be either ``'atm'`` (atmosphere)
        or ``'oc'`` (ocean).
    polarization : bool, optional
        If ``False``, build scalar-equivalent phase tables with
        polarization disabled. If ``True``, keep the polarized phase-
        matrix terms required by the vector radiative transfer kernels.
    ang_a : ndarray, optional
        Scattering angle grid in radians the phase matrix is sampled
        on. Defaults to ``n_theta`` equally spaced angles.

    Returns
    -------
    (ndarray, ndarray)
        The phase matrix table, of shape ``(n_phase_entries,
        n_theta)`` and dtype ``TYPE_PHASE``, and the cumulative
        distribution, of the same shape and dtype ``TYPE_PCDF``. Both
        are indexed by ``iphase_<kind>`` in the profile uploaded by
        ``_init_profile``.

    Notes
    -----
    The scattering-angle coordinate from ``theta_<kind>`` is converted
    from degrees to radians internally, and the cumulative scattering
    probability is obtained by integrating the phase terms over solid
    angle before interpolation.
    """

    if hasattr(profile, 'to_xarray'):
        profile = profile.to_xarray()

    name_phase = 'phase_{}'.format(kind)
    if name_phase in profile.data_vars:
        nphases = profile[name_phase].shape[0]
    else:
        nphases = 0

    nphases += 2  # include Rayleigh and VRS phase function
    # nphases += 1   # include Rayleigh phase function

    nrows = nphases if nphases > 0 else 1
    phase_H = np.zeros((nrows, n_theta), dtype=TYPE_PHASE, order='C')
    angN = _uniform_angles(n_theta) if ang_a is None else ang_a

    # Set Rayleigh phase function or isotropic if depo <0
    if depo >= 0:
        phase_H[0, :] = _rayleigh(
            n_theta, depo, polarization=polarization, ang_a=ang_a
        )
    # no polarization switch in isotropic because the function needs
    # first to be corrected
    else:
        phase_H[0, :] = _isotropic(n_theta, ang_a=ang_a)
    if 'theta_' + kind in profile.coords:
        angles = profile.coords['theta_' + kind].to_numpy() * pi / 180.0
        assert angles[-1] < 3.15  # assert that angles are in radians
    else:
        angles = None

    # Set VRS phase function
    phase_H[1, :] = _rayleigh(n_theta, 0.17, ang_a=ang_a)

    idx = 2
    # idx = 1
    for ipha in range(nphases - 2):
        # for ipha in range(nphases-1):
        assert angles is not None

        # (ipha, nphamat, theta)
        phase = profile[name_phase][ipha, :, :].to_numpy()

        phase = convert_phase_to_iparper(phase)

        if not polarization:
            if len(phase[:, 0]) == 4:
                raise ValueError(
                    "old profiles with only 4 phase matrix terms are not "
                    "supported without polarization"
                )
            # back to IQUV convention to obtain F11
            F11 = 0.5 * (phase[0, :] + 2 * phase[1, :] + phase[4, :])
            # reset all values to 0
            phase[:, :] = 0.0
            # reconvert to Iperpar but without considering polarization
            phase[0, :] = 0.5 * F11
            phase[1, :] = 0.5 * F11
            phase[4, :] = 0.5 * F11

        # f1 = interp1d(angles, phase[1,:])
        # f2 = interp1d(angles, phase[0,:])
        f1 = interp1d(angles, phase[0, :])
        f2 = interp1d(angles, phase[1, :])
        f3 = interp1d(angles, phase[2, :])
        f4 = interp1d(angles, phase[3, :])

        if len(phase[:, 0]) == 4:  # spherical particle
            # parameters equally spaced in scattering angle [0, 180]
            phase_H['a_P11'][idx, :] = f1(angN)  # I par P11
            phase_H['a_P22'][idx, :] = f2(angN)  # I per P22
            phase_H['a_P33'][idx, :] = f3(angN)  # U P33
            phase_H['a_P43'][idx, :] = f4(angN)  # V P43
            phase_H['a_P44'][idx, :] = f3(angN)  # V P44=P33
        else:  # non spherical particle
            f5 = interp1d(angles, phase[4, :])
            f6 = interp1d(angles, phase[5, :])

            phase_H['a_P11'][idx, :] = f1(angN)  # I par P11
            phase_H['a_P22'][idx, :] = f5(angN)  # I per P22
            phase_H['a_P12'][idx, :] = f2(angN)  # I per P22
            phase_H['a_P33'][idx, :] = f3(angN)  # U P33
            phase_H['a_P43'][idx, :] = f4(angN)  # V P43
            phase_H['a_P44'][idx, :] = f6(angN)  # V P44=P33
            # phase_H['a_P33'][idx, :] = f6(angN)  # V P44=P33

        idx += 1

    return phase_H, _cdf_of_table(phase_H, angN)


def _cdf_of_table(
    phase_H: np.ndarray, ang: NDArray[np.float64]
) -> np.ndarray:
    """
    Cumulative distribution of every row of a phase table, at the
    nodes of its angle grid.

    The mass of a bin is the exact integral of the tabulated phase
    function over it, F11 linear in theta between the two nodes times
    the true sin(theta), which is the density the kernel's ``pSample``
    inverts in closed form. The table's own float32 values are used,
    so that the host and the device describe the same function.

    Parameters
    ----------
    phase_H : ndarray
        Table of dtype ``TYPE_PHASE`` and shape ``(nrows, n)``.
    ang : ndarray
        The ``n`` angles in radians, from 0 to pi.

    Returns
    -------
    ndarray
        ``(nrows, n)`` of dtype ``TYPE_PCDF``, each row from 0 to 1.
    """
    f11 = 0.5 * (
        phase_H['a_P11'].astype(np.float64)
        + phase_H['a_P22'].astype(np.float64)
        + 2.0 * phase_H['a_P12'].astype(np.float64)
    )
    th0 = ang[:-1]
    th1 = ang[1:]
    dth = th1 - th0
    f0 = f11[:, :-1]
    df = f11[:, 1:] - f0
    # int_th0^th1 (f0 + df (th-th0)/dth) sin(th) dth
    mass = (
        f0 * (np.cos(th0) - np.cos(th1))
        + df * ((np.sin(th1) - np.sin(th0)) / dth - np.cos(th1))
    )
    cdf = np.zeros(phase_H.shape, dtype=np.float64)
    cdf[:, 1:] = np.cumsum(mass, axis=1)
    total = cdf[:, -1:]
    # a row of zeros, e.g. a VRS entry of a profile that has none,
    # keeps a cumulative distribution of zeros and is never drawn
    np.divide(cdf, total, out=cdf, where=total > 0)
    return cdf.astype(TYPE_PCDF)


def _calc_phase_gpu(
    profile,
    n_theta: int,
    depo: float,
    kind: str,
    polarization: bool = True,
    ang_a: NDArray[np.float64] | None = None,
) -> tuple[GPUArray, GPUArray]:
    """
    Upload the phase matrix table and the cumulative distribution
    built by :func:`_calc_phase_host` to the GPU.

    They are built on the host so that they can be checked without a
    GPU; see :func:`_calc_phase_host` for the parameters and for the
    layout of the returned tables.

    Returns
    -------
    (pycuda.gpuarray.GPUArray, pycuda.gpuarray.GPUArray)
        The phase matrix table, of shape ``(n_phase_entries,
        n_theta)`` and dtype ``TYPE_PHASE``, and the cumulative
        distribution, of the same shape and dtype ``TYPE_PCDF``.
    """
    phase_H, cdf_H = _calc_phase_host(
        profile, n_theta, depo, kind, polarization=polarization, ang_a=ang_a,
    )

    return to_gpu(phase_H), to_gpu(cdf_H)


def _init_const(
    surface,
    environment,
    n_atm: int,
    n_atm_abs: int | np.integer,
    n_oce: int,
    n_oce_abs: int | np.integer,
    mod: SourceModule,
    n_loop: float | None,
    th_deg: float,
    xblock: int,
    xgrid: int,
    n_lam: int,
    sim: int,
    agrid_atm: tuple,
    agrid_oc: tuple,
    pgrid_atm: tuple,
    pgrid_oc: tuple,
    n_theta: int,
    n_phi: int,
    output_layers: int,
    earth_radius: float,
    le: int,
    zip_mode: int,
    flux: int,
    ffs: bool,
    direct: bool,
    ocean_interaction: bool | None,
    n_lvl: int,
    n_pstk: int,
    n_wavelength_proba: int,
    n_sensor_proba: int,
    n_cell_proba: int,
    beer: int,
    s_min: float,
    s_max: float,
    r_min: float,
    r_max: float,
    russian_roulette: int,
    russian_roulette_weight: float,
    n_low: int,
    n_jac: int,
    n_sensor: int,
    refrac: int,
    horiz: int,
    sza_max: float,
    sun_disc: float,
    le_fov: float,
    cus_l,
    n_obj: int,
    n_gobj: int,
    n_robj: int,
    p_min_x: float | None,
    p_min_y: float | None,
    p_min_z: float | None,
    p_max_x: float | None,
    p_max_y: float | None,
    p_max_z: float | None,
    is_atm: int | None,
    tc: float | None,
    n_cx: int,
    n_cy: int,
    v_sun,
    hist: int,
    z_toa: float,
    cell_size,
    sx_min: float,
    sx_max: float,
    sy_min: float,
    sy_max: float,
    n_sx: int,
    n_sy: int,
    no_aer_output: bool,
    n_scl: int = 1,
    scl_mode: int = 0,
    n_orders: int = 1,
    n_jac_abs: int = 0,
) -> None:
    """Upload the simulation constants to the CUDA device globals.

    This routine computes a few derived geometric quantities and copies
    all scalar simulation settings to the global constants defined in
    the CUDA module.

    Parameters
    ----------
    surface : object | None
        Surface configuration object from ``smartg.surface``
        (FlatSurface, RoughSurface, LambSurface, RTLSSurface or
        RPVSurface) exposing a ``dict`` attribute with keys required
        by SMART-G (for example
        ``SUR``, ``BRDF``, ``DIOPTRE``, ``WINDSPEED``, ``NH2O``,
        ``WAVE_SHADOW``, ``SINGLE``).
    environment : Environment | None
        Environment configuration object exposing a ``dict`` attribute
        and geometry metadata (for example ``nenv``, ``nxenvmap``,
        ``nyenvmap``). If ``None``, environment-related constants are
        not updated.
    n_atm, n_atm_abs, n_oce, n_oce_abs : int
        Numbers of atmospheric/oceanic layers and absorbing layers.
    mod : pycuda.compiler.SourceModule
        Compiled CUDA module containing global symbols to update.
    agrid_atm, agrid_oc : tuple
        Angle grid descriptors ``(n, mode, ang_gpu)`` of the
        atmospheric and oceanic phase tables, see ``_resolve_agrid``.
    pgrid_atm, pgrid_oc : tuple
        Cumulative distribution descriptors ``(n, cdf_gpu)`` of the
        atmosphere and the ocean, see ``_pgrid_struct``. The caller
        keeps *cdf_gpu* alive; only its address reaches the device.
    n_loop, xblock, xgrid, n_lam, sim : int
        Main Monte Carlo control parameters.
    n_theta, n_phi, output_layers : int
        Output-grid control parameters.
    th_deg, earth_radius, sza_max, sun_disc, le_fov, z_toa : float
        Angular and physical scalar settings.
    cell_size, sx_min, sx_max, sy_min, sy_max : float
        Spatial scalar settings.
    le, zip_mode, flux, direct, beer : int
        Integer flags controlling radiative-transfer modes.
    n_lvl, n_pstk, n_theta, n_phi, n_lam : int
        Angular/spectral and Stokes discretization controls.
    n_wavelength_proba, n_sensor_proba, n_cell_proba : int
        Sampling configuration parameters.
    s_min, s_max, r_min, r_max, russian_roulette, n_low : int
        Path-length limits and Russian-roulette configuration.
    n_jac, hist, n_sensor, refrac, horiz : int
        Jacobian/history, sensor, and geometry/refraction control flags.
    n_obj, n_gobj, n_robj, n_cx, n_cy, n_scl, scl_mode, n_orders : int
        Object-scene and acceleration/grid scaling configuration.
    ffs : bool
        If ``True``, enable forward-flux mode constant.
    ocean_interaction : bool or None
        Ocean-interaction flag. If ``None``, the dedicated device
        constant is set to ``-1``.
    russian_roulette_weight : float
        Weight associated with Russian roulette.
    cus_l : CusForward | CusBackward | None
        Optional custom launch/view configuration object exposing
        ``dict``.
    p_min_x, p_min_y, p_min_z, p_max_x, p_max_y, p_max_z : float
        Bounding-box limits for object handling.
    is_atm : int
        Flag indicating atmospheric context for object processing.
    tc : float or None
        Receiver cell size.
    v_sun : gc.Vector
        Sun-direction vector with ``x``, ``y``, and ``z`` attributes.
    n_sx, n_sy : int
        Number of horizontal bins for aerosol-related outputs.
    no_aer_output : bool
        Add output where only photons not scattered by aerosols are
        considered. Default False.

    Returns
    -------
    None
    """

    # compute some needed constants
    th_v = th_deg * np.pi / 180.0
    s_th_v = np.sin(th_v)
    c_th_v = np.cos(th_v)

    if (cus_l is not None) and (cus_l.dict['mode'] == "FF"):
        pz_d = z_toa + cus_l.dict['cftz']
    else:
        pz_d = z_toa
    t_temp = pz_d / -v_sun.z
    px_d = -v_sun.x * t_temp
    py_d = -v_sun.y * t_temp

    def copy_to_device(name: str, scalar, dtype) -> None:
        cuda.memcpy_htod(  # pyright: ignore[reportAttributeAccessIssue]
            mod.get_global(name)[0], np.array([scalar], dtype=dtype)
        )

    # copy constants to device
    copy_to_device('NBLOOPd', n_loop, np.uint32)
    copy_to_device('NOCEd', n_oce, np.int32)
    copy_to_device('NOCE_ABSd', n_oce_abs, np.int32)
    copy_to_device('OUTPUT_LAYERSd', output_layers, np.int32)
    copy_to_device('AGAERd', _agrid_struct(agrid_atm), TYPE_AGRID)
    copy_to_device('AGOCEd', _agrid_struct(agrid_oc), TYPE_AGRID)
    copy_to_device('PGAERd', _pgrid_struct(pgrid_atm), TYPE_PGRID)
    copy_to_device('PGOCEd', _pgrid_struct(pgrid_oc), TYPE_PGRID)
    copy_to_device('NATMd', n_atm, np.int32)
    copy_to_device('NATM_ABSd', n_atm_abs, np.int32)
    copy_to_device('XBLOCKd', xblock, np.int32)
    copy_to_device('YBLOCKd', 1, np.int32)
    copy_to_device('XGRIDd', xgrid, np.int32)
    copy_to_device('YGRIDd', 1, np.int32)
    copy_to_device('NBTHETAd', n_theta, np.int32)
    copy_to_device('NBPHId', n_phi, np.int32)
    copy_to_device('NLAMd', n_lam, np.int32)
    copy_to_device('SIMd', sim, np.int32)
    copy_to_device('LEd', le, np.int32)
    copy_to_device('ZIPd', zip_mode, np.int32)
    copy_to_device('FLUXd', flux, np.int32)
    copy_to_device('FFSd', 1 if ffs else 0, np.int32)
    copy_to_device('DIRECTd', 1 if direct else 0, np.int32)
    copy_to_device('cell_sized', cell_size, np.float32)
    copy_to_device('sxmind', sx_min, np.float32)
    copy_to_device('sxmaxd', sx_max, np.float32)
    copy_to_device('symind', sy_min, np.float32)
    copy_to_device('symaxd', sy_max, np.float32)
    copy_to_device('nbsxd', n_sx, np.uint32)
    copy_to_device('nbsyd', n_sy, np.uint32)
    copy_to_device('no_aer_outd', int(no_aer_output), np.int32)
    if ocean_interaction is None:
        copy_to_device('OCEAN_INTERACTIONd', -1, np.int32)
    else:
        copy_to_device(
            'OCEAN_INTERACTIONd', 1 if ocean_interaction else 0, np.int32
        )
    # copy_to_device('MId', MI, np.int32)
    copy_to_device('NLVLd', n_lvl, np.int32)
    copy_to_device('NPSTKd', n_pstk, np.int32)
    copy_to_device('BEERd', beer, np.int32)
    copy_to_device('SMINd', s_min, np.int32)
    copy_to_device('SMAXd', s_max, np.int32)
    copy_to_device('RMINd', r_min, np.int32)
    copy_to_device('RMAXd', r_max, np.int32)
    copy_to_device('RRd', russian_roulette, np.int32)
    copy_to_device('WEIGHTRRd', russian_roulette_weight, np.float32)
    copy_to_device('NLOWd', n_low, np.int32)
    copy_to_device('NJACd', n_jac, np.int32)
    copy_to_device('NJACABSd', n_jac_abs, np.int32)
    copy_to_device('HISTd', hist, np.int32)
    copy_to_device('NSENSORd', n_sensor, np.int32)
    copy_to_device('NSCLd', n_scl, np.int32)
    copy_to_device('SCL_MODEd', scl_mode, np.int32)
    copy_to_device('NORDERSd', n_orders, np.int32)
    if surface is not None:
        copy_to_device('SURd', surface.dict['SUR'], np.int32)
        copy_to_device('BRDFd', surface.dict['BRDF'], np.int32)
        copy_to_device('DIOPTREd', surface.dict['DIOPTRE'], np.int32)
        copy_to_device('WINDSPEEDd', surface.dict['WINDSPEED'], np.float32)
        copy_to_device('NH2Od', surface.dict['NH2O'], np.float32)
        copy_to_device('WAVE_SHADOWd', surface.dict['WAVE_SHADOW'], np.int32)
        copy_to_device('SINGLEd', surface.dict['SINGLE'], np.int32)
    if environment is not None:
        copy_to_device('ENVd', environment.dict['ENV'], np.int32)
        copy_to_device('ENV_SIZEd', environment.dict['ENV_SIZE'], np.float32)
        copy_to_device('X0d', environment.dict['X0'], np.float32)
        copy_to_device('Y0d', environment.dict['Y0'], np.float32)
        copy_to_device('NENVd', environment.nenv, np.int32)
        copy_to_device('NXENVMAPd', environment.nxenvmap, np.int32)
        copy_to_device('NYENVMAPd', environment.nyenvmap, np.int32)
    copy_to_device('STHVd', s_th_v, np.float32)
    copy_to_device('CTHVd', c_th_v, np.float32)
    copy_to_device('RTER', earth_radius, np.float32)
    copy_to_device('NWLPROBA', n_wavelength_proba, np.int32)
    copy_to_device('NSENSORPROBA', n_sensor_proba, np.int32)
    copy_to_device('NCELLPROBA', n_cell_proba, np.int32)
    copy_to_device('REFRACd', refrac, np.int32)
    copy_to_device('HORIZd', horiz, np.int32)
    copy_to_device('SZA_MAXd', sza_max, np.float32)
    copy_to_device('SUN_DISCd', sun_disc, np.float32)
    copy_to_device('LE_FOVd', le_fov, np.float32)
    # copy en rapport avec les objets :
    if n_obj != 0:
        copy_to_device('nObj', n_obj, np.int32)
        copy_to_device('nGObj', n_gobj, np.int32)
        copy_to_device('nRObj', n_robj, np.int32)
        copy_to_device('Pmin_x', p_min_x, np.float32)
        copy_to_device('Pmin_y', p_min_y, np.float32)
        copy_to_device('Pmin_z', p_min_z, np.float32)
        copy_to_device('Pmax_x', p_max_x, np.float32)
        copy_to_device('Pmax_y', p_max_y, np.float32)
        copy_to_device('Pmax_z', p_max_z, np.float32)
        copy_to_device('IsAtm', is_atm, np.int32)
        copy_to_device('DIRSXd', v_sun.x, np.float64)
        copy_to_device('DIRSYd', v_sun.y, np.float64)
        copy_to_device('DIRSZd', v_sun.z, np.float64)
        copy_to_device('PXd', px_d, np.float32)
        copy_to_device('PYd', py_d, np.float32)
        copy_to_device('PZd', pz_d, np.float32)
        copy_to_device('ZTOAd', z_toa, np.float32)
        if tc is not None:
            copy_to_device('TCd', tc, np.float32)
            copy_to_device('nbCx', n_cx, np.int32)
            copy_to_device('nbCy', n_cy, np.int32)
        if (cus_l is not None) and (cus_l.dict['mode'] == "RF"):
            copy_to_device('LMODEd', 1, np.int32)
        if (cus_l is not None) and (cus_l.dict['mode'] == "FF"):
            copy_to_device('CFXd', cus_l.dict['cfx'], np.float32)
            copy_to_device('CFYd', cus_l.dict['cfy'], np.float32)
            copy_to_device('CFTXd', cus_l.dict['cftx'], np.float32)
            copy_to_device('CFTYd', cus_l.dict['cfty'], np.float32)
            copy_to_device('ALDEGd', cus_l.dict['fov'], np.float32)
            copy_to_device('TYPEd', cus_l.dict['sampling_code'], np.int32)
            copy_to_device('LMODEd', 2, np.int32)
        if (cus_l is not None) and (
            cus_l.dict['mode'] == "B" or cus_l.dict['mode'] == "BR"
        ):
            copy_to_device('THDEGd', cus_l.dict['th_deg'], np.float32)
            copy_to_device('PHDEGd', cus_l.dict['ph_deg'], np.float32)
            copy_to_device('ALDEGd', cus_l.dict['receiver_fov'], np.float32)
            copy_to_device('TYPEd', cus_l.dict['sampling_code'], np.int32)
            copy_to_device('CBACK_SFOVd', cus_l.dict['sun_fov'], np.float32)
        if (cus_l is not None) and (cus_l.dict['mode'] == "B"):
            copy_to_device('LMODEd', 3, np.int32)
        if (cus_l is not None) and (cus_l.dict['mode'] == "BR"):
            copy_to_device('LMODEd', 4, np.int32)
        if cus_l is None:
            copy_to_device('LMODEd', 0, np.int32)


def _init_profile(wavelength, prof, kind: str) -> tuple:
    """Prepare profile and cell arrays on the GPU.

    Convert an atmospheric or oceanic profile into the internal SMART-G
    structured arrays and upload them to GPU memory.

    Parameters
    ----------
    wavelength : 1-D ndarray
        Wavelength grid used for the simulation. Its length defines the
        first dimension of the generated profile array.
    prof : xr.Dataset
        Atmospheric or oceanic profile.
    kind : str
        Profile family identifier. Must be either ``'atm'`` (atmosphere)
        or ``'oc'`` (ocean).

    Returns
    -------
    tuple
        Two GPU arrays ``(prof_gpu, cell_gpu)`` where:

        - ``prof_gpu`` contains the profile 1-D optical properties,
        - ``cell_gpu`` contains the profile 3-D optical properties.
    """

    if kind not in ('atm', 'oc'):
        raise ValueError("kind must be either 'atm' or 'oc'.")

    if hasattr(prof, 'to_xarray'):
        prof = prof.to_xarray()

    # NREF = len(prof.axis('z_'+kind))
    # reformat to smartg format
    if 'iopt_' + kind in prof.data_vars:
        NLAY = len(prof['OD_' + kind].to_numpy()[0, :])
    else:
        NLAY = len(prof.coords['z_' + kind])
    shp = (len(wavelength), NLAY)
    prof_gpu = np.zeros(shp, dtype=TYPE_PROFILE, order='C')

    if kind == "oc":
        if 'iopt_oc' not in prof.data_vars:
            prof_gpu['z'][0, :] = prof.coords['z_' + kind].to_numpy()
            # prof_gpu['z'][0,:] = prof.coords['z_'+kind].to_numpy()  *
            # 1e-3 # to Km
            prof_gpu['T'][0, :] = prof['T_' + kind].to_numpy()
            cell_gpu = np.zeros(1, dtype=TYPE_CELL)
        else:
            cell_gpu = np.zeros(
                len(prof['iopt_oc'].to_numpy()), dtype=TYPE_CELL
            )
        prof_gpu['n'][0, :] = 1.34
    else:
        if 'iopt_atm' not in prof.data_vars:
            prof_gpu['z'][0, :] = prof.coords['z_' + kind].to_numpy()
            prof_gpu['T'][0, :] = prof['T_' + kind].to_numpy()
            prof_gpu['n'][:, :] = prof['n_' + kind].to_numpy()
            cell_gpu = np.zeros(1, dtype=TYPE_CELL)
        else:
            cell_gpu = np.zeros(
                len(prof['iopt_atm'].to_numpy()), dtype=TYPE_CELL
            )
    prof_gpu['z'][1:, :] = -999.0  # other wavelengths are NaN

    prof_gpu['OD'][:, :] = prof['OD_' + kind].to_numpy()
    prof_gpu['OD_sca'][:] = prof['OD_sca_' + kind].to_numpy()
    prof_gpu['OD_abs'][:] = prof['OD_abs_' + kind].to_numpy()
    prof_gpu['pmol'][:] = prof['pmol_' + kind].to_numpy()
    prof_gpu['ssa'][:] = prof['ssa_' + kind].to_numpy()
    prof_gpu['pine'][:] = prof['pine_' + kind].to_numpy()
    prof_gpu['FQY1'][:] = prof['FQY1_' + kind].to_numpy()
    if 'iphase_' + kind in prof.data_vars:
        prof_gpu['iphase'][:] = prof['iphase_' + kind].to_numpy()

    if len(cell_gpu) > 1:
        cell_gpu['iopt'][:] = prof['iopt_' + kind].to_numpy()
        cell_gpu['iabs'][:] = prof['iabs_' + kind].to_numpy()
        pmin = prof['pmin_' + kind].to_numpy()
        pmax = prof['pmax_' + kind].to_numpy()
        neighbour = prof['neighbour_' + kind].to_numpy()
        cell_gpu['pminx'][:] = pmin[0, :]
        cell_gpu['pminy'][:] = pmin[1, :]
        cell_gpu['pminz'][:] = pmin[2, :]
        cell_gpu['pmaxx'][:] = pmax[0, :]
        cell_gpu['pmaxy'][:] = pmax[1, :]
        cell_gpu['pmaxz'][:] = pmax[2, :]
        cell_gpu['neighbour1'][:] = neighbour[0, :]
        cell_gpu['neighbour2'][:] = neighbour[1, :]
        cell_gpu['neighbour3'][:] = neighbour[2, :]
        cell_gpu['neighbour4'][:] = neighbour[3, :]
        cell_gpu['neighbour5'][:] = neighbour[4, :]
        cell_gpu['neighbour6'][:] = neighbour[5, :]

    return to_gpu(prof_gpu), to_gpu(cell_gpu)


def multi_profiles(profs: list, kind: str = 'atm') -> xr.Dataset:
    """Reorganize a list of profiles into a single multi-profile table.

    This helper concatenates compatible profile fields so several
    atmosphere or ocean profile configurations can be simulated in a
    single SMART-G run. It can also be used in workflows such as finite-
    difference sensitivity or Jacobian computations, but it is not
    limited to those use cases.

    Parameters
    ----------
    profs : list of xr.Dataset
        Profiles returned by atmospheric or oceanic profile builders
        (for example ``atm.calc()`` or ``water.calc()``). MLUT-like
        objects are converted with ``to_xarray()`` when available.
        DataArray inputs are converted to single-variable datasets.
    kind : str, default='atm'
        Profile family to process. Allowed values are:

        - ``'atm'`` for atmospheric profiles.
        - ``'oc'`` for oceanic profiles.

    Returns
    -------
    xr.Dataset
        Reorganized profile dataset where compatible variables from all
        input profiles are concatenated, with phase-function indexing
        adjusted to remain unique across concatenated blocks.
    """

    xprofs = []
    for prof in profs:
        if hasattr(prof, 'to_xarray'):
            prof = prof.to_xarray()
        if not isinstance(prof, xr.Dataset):
            raise TypeError('Each profile must be an xr.Dataset.')
        xprofs.append(prof)

    first = xprofs[0]
    pro = xr.Dataset(attrs=first.attrs)

    for d in first.data_vars:
        if 'iphase' in d:
            imax = 0
            chunks = []
            for M in xprofs:
                da = M[d]
                chunks.append(da + imax)
                imax += np.unique(da.data).max() + 1
            pro[d] = xr.concat(chunks, dim=chunks[0].dims[0])
        elif d == ('phase_' + kind):
            pro[d] = xr.concat([M[d] for M in xprofs], dim=first[d].dims[0])
        elif d == ('T_' + kind):
            pro[d] = first[d]
        else:
            pro[d] = xr.concat([M[d] for M in xprofs], dim=first[d].dims[0])

    return pro


def reduce_diff(
    ds_sg: xr.Dataset,
    varnames,
    delta: float | Sequence[float] | np.ndarray | None = None,
) -> xr.Dataset:
    """Post-process ALIS finite-difference runs into sensitivities.

    The input lookup tables are expected to be packed along the
    wavelength axis as one reference block followed by one perturbed
    block per variable: ``[ref, var1, var2, ...]``. For each radiometric
    quantity, this function keeps the reference LUT and appends one
    finite-difference LUT per variable.

    Parameters
    ----------
    ds_sg : xr.DataArray
        SMART-G output produced in ALIS finite-difference mode.
    varnames : sequence of str
        Names of perturbed variables, in the same order as their
        wavelength blocks in ``ds_sg``.
    delta : sequence of float, optional
        Perturbation amplitude for each variable. If provided, finite
        differences are divided by ``delta[k]`` and the outputs are
        Jacobians. If omitted, raw finite-difference sensitivities are
        returned.

    Returns
    -------
    xr.Dataset
        Dataset containing:

        - the original radiometric variables over the reference
          wavelength block,
        - one derived variable per perturbation containing either
          the sensitivity ``f(x+dx)-f(x)`` or the Jacobian
          ``(f(x+dx)-f(x))/dx``.

    Notes
    -----
    Only variables whose names contain one of ``'I_'``, ``'Q_'``,
    ``'U_'``, ``'V_'``, ``'transmission'``, or ``'flux'`` are processed.
    """

    if hasattr(ds_sg, 'to_xarray'):
        ds_sg = ds_sg.to_xarray()

    if isinstance(ds_sg, xr.DataArray):
        data_name = ds_sg.name if ds_sg.name is not None else 'data'
        ds_sg = ds_sg.to_dataset(name=data_name)

    if not isinstance(ds_sg, xr.Dataset):
        raise TypeError(
            'reduce_diff expects MLUT/LUT or xarray Dataset/DataArray input.'
        )

    if 'wavelength' not in ds_sg.dims:
        raise ValueError("Input must define a 'wavelength' dimension.")

    n_diff = len(varnames)
    n_wavelength_total = ds_sg.sizes['wavelength']
    block_size = int(n_wavelength_total / (n_diff + 1))
    if block_size * (n_diff + 1) != n_wavelength_total:
        raise ValueError(
            'wavelength size is not compatible with the number of '
            'perturbation blocks.'
        )

    if delta is not None:
        if np.isscalar(delta):
            delta_val = float(cast('float', delta))
            delta = np.full(n_diff, delta_val, dtype=np.float64)
        else:
            delta = np.asarray(delta)
        if delta.shape[0] != n_diff:
            raise ValueError('delta must have the same length as varnames.')

    wavelength_ref = ds_sg['wavelength'].isel(wavelength=slice(0, block_size))
    prefixes = ('I_', 'Q_', 'U_', 'V_', 'transmission', 'flux')

    out_vars = OrderedDict()
    for var_name, da in ds_sg.data_vars.items():
        if 'wavelength' not in da.dims:
            continue
        if not any(pref in str(var_name) for pref in prefixes):
            continue

        ref_da = da.isel(wavelength=slice(0, block_size)).assign_coords(
            wavelength=wavelength_ref
        )
        out_vars[var_name] = ref_da

        for k, pert_name in enumerate(varnames):
            pert_da = da.isel(
                wavelength=slice((k + 1) * block_size, (k + 2) * block_size)
            ).assign_coords(wavelength=wavelength_ref)
            diff_da = pert_da - ref_da
            if delta is not None:
                diff_da = diff_da / delta[k]
                deriv_name = f'd{var_name}/d{pert_name}'
            else:
                deriv_name = f'd{var_name}->({pert_name})'
            diff_da.attrs = dict(da.attrs)
            out_vars[deriv_name] = diff_da

    res = xr.Dataset(data_vars=out_vars, attrs=dict(ds_sg.attrs))
    return res


def _loop_kernel(
    n_photons: float,
    faer: GPUArray | None,
    foce: GPUArray | None,
    n_level: int,
    n_atm: int,
    n_atm_abs: int | np.integer,
    n_oce: int,
    n_oce_abs: int | np.integer,
    max_hist: int | np.integer,
    n_low: int,
    n_pstk: int,
    xblock: int,
    xgrid: int,
    n_theta: int,
    n_phi: int,
    n_lam: int,
    n_sensor: int,
    double: bool,
    kernel,
    progress,
    x0: GPUArray | None,
    le: dict | None,
    tab_sensor: GPUArray,
    envmap: GPUArray,
    spectrum: GPUArray,
    prof_atm: GPUArray,
    prof_oc: GPUArray,
    cell_atm: GPUArray,
    cell_oc: GPUArray,
    wavelength_proba_icdf: GPUArray | None,
    sensor_proba_icdf: GPUArray | None,
    cell_proba_icdf: GPUArray | None,
    stdev: bool,
    stdev_lim: StdevLim | None,
    rng,
    alis: bool,
    lobj_gpu: GPUArray | None,
    receiver_cell_size: float | None,
    n_cx: int,
    n_cy: int,
    lgobj_gpu: GPUArray | None,
    lrobj_gpu: GPUArray | None,
    lobj_spect: GPUArray | None,
    hist: bool = False,
    amf_variance: bool = False,
    nscl: int = 1,
    le_fov: float = 0.0,
) -> tuple:
    """Run the transport kernel until the requested photon budget.

    This function repeatedly launches the GPU kernel, accumulates
    radiometric outputs, optional ALIS path-length diagnostics, optional
    history buffers, and optional receiver-object diagnostics until the
    stopping criterion is reached.

    Parameters
    ----------
    n_photons : int
        Target number of launched photons.
    faer, foce : pycuda.gpuarray.GPUArray
        Atmospheric and oceanic phase-function lookup tables (see
        _calc_phase_gpu).
    n_level : int
        Number of output levels.
    n_atm, n_atm_abs : int
        Number of atmospheric layers and number of atmospheric absorbing
        layers.
    n_oce, n_oce_abs : int
        Number of ocean layers and number of ocean absorbing layers.
    max_hist : int
        Maximum history length when ``hist=True``.
    n_low : int
        Number of wavelengths in ALIS low-resolution mode.
    n_pstk : int
        Number of Stokes components plus one accumulator component.
    xblock, xgrid : int
        CUDA launch dimensions (threads per block and number of blocks).
    n_theta, n_phi : int
        Number of angular bins in zenith and azimuth.
    n_lam : int
        Number of wavelengths.
    n_sensor : int
        Number of sensors.
    double : bool
        If True, use double precision for kernel accumulators.
    kernel : callable
        Main GPU kernel entry point.
    progress : progress
        Progress-bar-like object exposing ``update(value, message)``.
    x0 : pycuda.gpuarray.GPUArray
        Initial photon position.
    le : dict or None
        Local estimate configuration, or None.
    tab_sensor, envmap, spectrum : pycuda.gpuarray.GPUArray
        Sensor table, environment map, and spectrum arrays on device.
    prof_atm, prof_oc : pycuda.gpuarray.GPUArray
        Atmospheric and ocean profile tables.
    cell_atm, cell_oc : pycuda.gpuarray.GPUArray
        Atmospheric and ocean cell lookup tables.
    wavelength_proba_icdf, sensor_proba_icdf, cell_proba_icdf : GPUArray
        Inverse-CDF tables for wavelength, sensor, and cell sampling.
    stdev : bool
        If True, estimate standard deviation of normalized outputs.
    stdev_lim : object or None
        Optional adaptive stopping criterion based on absolute/relative
        error.
    rng : object
        Random-number generator backend with a ``state`` GPU buffer.
    alis : bool
        Whether ALIS mode is active.
    lobj_gpu, lgobj_gpu, lrobj_gpu, lobj_spect : GPUArray
        Object, object-group, receiver-object, and object-spectrum GPU
        tables.
    receiver_cell_size : float or None
        Receiver cell size. If None, receiver diagnostics are disabled.
    n_cx, n_cy : int
        Receiver grid dimensions in x and y.
    hist : bool, optional
        If True, accumulate photon histories.
    amf_variance : bool, optional
        If True, allocate extra AMF variance channel in ALIS distances.
    nscl : int, optional
        Number of ALIS scaling channels.

    Returns
    -------
    tuple
        Tuple containing, in order:

        1. ``n_photons_in_tot`` (ndarray)
        2. ``tab_photons_tot`` (ndarray)
        3. ``tab_photons_tot_no_aer`` (ndarray)
        4. ``tab_dist_tot`` (ndarray)
        5. ``tab_hist_tot`` (ndarray)
        6. ``tab_trans_dir`` (ndarray)
        7. ``errorcount`` (pycuda.gpuarray.GPUArray)
        8. ``n_photons_out_tot`` (ndarray)
        9. ``n_photons_out_tot_no_aer`` (ndarray)
        10. ``sigma`` (ndarray or None)
        11. ``n_simu`` (int)
        12. ``secs_cuda_clock`` (float)
        13. ``tab_mat_recep`` (ndarray or None)
        14. ``mat_cats`` (ndarray or None)
        15. ``mat_loss`` (ndarray or None)
        16. ``w_ph_cat_tot`` (ndarray)
        17. ``w_ph_cat2_tot`` (ndarray)
    """
    # Initializations
    n_threads_active = gpuzeros(1, dtype=np.uint32)
    counter = gpuzeros(1, dtype=np.uint64)

    if double:
        fdtype = np.float64
    else:
        fdtype = np.float32

    # If a receiver object is used then: initialize matrix and vectors
    # for gains and losses
    if receiver_cell_size is not None:
        n_ph_cat = gpuzeros(
            8, dtype=np.uint64
        )  # number of photons in each category
        w_ph_cat = gpuzeros(
            (8, n_lam), dtype=fdtype
        )  # photon weight for each category
        w_ph_cat_tot = gpuzeros((8, n_lam), dtype=fdtype)
        w_ph_cat2 = gpuzeros(
            (8, n_lam), dtype=fdtype
        )  # squared photon weights per category
        w_ph_cat2_tot = gpuzeros((8, n_lam), dtype=fdtype)
        tab_obj_info = gpuzeros((9, n_cx, n_cy), dtype=fdtype)
        w_ph_loss = gpuzeros(7, dtype=fdtype)
        w_ph_loss2 = gpuzeros(7, dtype=fdtype)
        tab_mat_recep = np.zeros((9, n_cx, n_cy), dtype=np.float64)

        # Matrix where lines: l0 = sumCats, l1=cat1, l2=cat2, ...
        # l8=cat8
        # and columns: c0=nbPhotons, c1=weight, c2=weight2, c3=flux (W),
        # c4=errAbs, c5=err%
        mat_cats = np.zeros((9, 6), dtype=np.float64)

        # Matrix where: M[0,0]=W_I, M[1,0]=W_rhoM, ..., M[6,0]=W_SP
        # and: M[0,1]=W_I^2, M[1,1]=W_rhoM^2, ..., M[6,1]=W_SP^2
        mat_loss = np.zeros((7, 2), dtype=np.float64)
    else:
        n_ph_cat = gpuzeros((1, 1), dtype=np.uint64)
        w_ph_cat = gpuzeros((1, 1), dtype=fdtype)
        w_ph_cat2 = gpuzeros((1, 1), dtype=fdtype)
        w_ph_cat_tot = gpuzeros((1, 1), dtype=fdtype)
        w_ph_cat2_tot = gpuzeros((1, 1), dtype=fdtype)
        w_ph_loss = gpuzeros(1, dtype=fdtype)
        w_ph_loss2 = gpuzeros(1, dtype=fdtype)
        tab_obj_info = gpuzeros((1, 1, 1), dtype=fdtype)
        tab_mat_recep = None
        mat_cats = None
        mat_loss = None

    # Scratch holding the local estimate directions sampled inside
    # the cone, one slice per thread. Its azimuth part is n_theta
    # long when the directions are zipped, since the kernel then
    # takes iph from ith; the kernel slices it by the same rule.
    if le_fov > 0:
        n_phi_le = (
            n_theta if (le is not None and le.get('zip', False)) else n_phi
        )
        tab_dir_le = gpuzeros(
            xblock * xgrid * (n_theta + n_phi_le), dtype='float32'
        )
    else:
        tab_dir_le = gpuzeros(1, dtype='float32')

    # Initialize the array for error counting
    n_error = 32
    errorcount = gpuzeros(n_error, dtype='uint64')

    if n_atm > 0:
        tab_trans_dir = gpuzeros((n_sensor, n_lam), dtype=np.float64)
    else:
        tab_trans_dir = gpuzeros((1, 1), dtype=np.float64)

    if (n_atm + n_oce > 0) and (n_atm_abs + n_oce_abs < 500) and alis:
        n_iamf = 3 if amf_variance else 2
        n_scl = nscl
        tab_dist_tot = gpuzeros(
            (
                n_level,
                n_atm_abs + n_oce_abs,
                n_sensor,
                n_theta,
                n_phi,
                n_scl,
                n_iamf,
            ),
            dtype=np.float64,
        )
    else:
        n_scl = 1
        n_iamf = None
        tab_dist_tot = gpuzeros((1), dtype=np.float64)

    # Initialize accumulators
    tab_photons_tot = gpuzeros(
        (n_level, n_pstk, n_sensor, n_lam, n_theta, n_phi), dtype=np.float64
    )
    tab_photons_tot_no_aer = gpuzeros(
        (n_level, n_pstk, n_sensor, n_lam, n_theta, n_phi), dtype=np.float64
    )
    n_simu = 0
    # Accumulate normalized quantities and their squares to estimate
    # sigma (only filled when stdev is enabled).
    sum_x = 0.0
    sum_x2 = 0.0
    format_std = ''
    max_aerr = None
    max_rerr = None

    # Arrays for counting launched photons (per wavelength)
    n_photons_in = gpuzeros((n_sensor, n_lam), dtype=np.uint64)
    n_photons_in_tot = gpuzeros((n_sensor, n_lam), dtype=np.uint64)

    # Arrays for counting output photons
    n_photons_out = gpuzeros(
        (n_level, n_sensor, n_lam, n_theta, n_phi), dtype=np.uint64
    )
    n_photons_out_no_aer = gpuzeros(
        (n_level, n_sensor, n_lam, n_theta, n_phi), dtype=np.uint64
    )
    n_photons_out_tot = gpuzeros(
        (n_level, n_sensor, n_lam, n_theta, n_phi), dtype=np.uint64
    )
    n_photons_out_tot_no_aer = gpuzeros(
        (n_level, n_sensor, n_lam, n_theta, n_phi), dtype=np.uint64
    )

    if double:
        tab_photons = gpuzeros(
            (n_level, n_pstk, n_sensor, n_lam, n_theta, n_phi),
            dtype=np.float64,
        )
        tab_photons_no_aer = gpuzeros(
            (n_level, n_pstk, n_sensor, n_lam, n_theta, n_phi),
            dtype=np.float64,
        )
        if (n_atm + n_oce > 0) and (n_atm_abs + n_oce_abs < 500) and alis:
            tab_dist = gpuzeros(
                (
                    n_level,
                    n_atm_abs + n_oce_abs,
                    n_sensor,
                    n_theta,
                    n_phi,
                    n_scl,
                    n_iamf,
                ),
                dtype=np.float64,
            )
        else:
            tab_dist = gpuzeros((1), dtype=np.float64)
    else:
        tab_photons = gpuzeros(
            (n_level, n_pstk, n_sensor, n_lam, n_theta, n_phi),
            dtype=np.float32,
        )
        tab_photons_no_aer = gpuzeros(
            (n_level, n_pstk, n_sensor, n_lam, n_theta, n_phi),
            dtype=np.float32,
        )
        if (n_atm + n_oce > 0) and (n_atm_abs + n_oce_abs < 500) and alis:
            tab_dist = gpuzeros(
                (
                    n_level,
                    n_atm_abs + n_oce_abs,
                    n_sensor,
                    n_theta,
                    n_phi,
                    n_scl,
                    n_iamf,
                ),
                dtype=np.float32,
            )
        else:
            tab_dist = gpuzeros((1), dtype=np.float32)

    if hist:
        _n_cols_hist = n_atm_abs + n_oce_abs + n_pstk + n_low + 7
        tab_hist_tot = gpuzeros(
            (2, max_hist, _n_cols_hist, n_sensor, n_theta, n_phi),
            dtype=np.float32,
        )
        _hist_bytes = int(cast('int', tab_hist_tot.nbytes))
        print(
            f"[ALIS hist] tabHist allocated — "
            f"shape: (2, {max_hist:,}, {_n_cols_hist}, {n_sensor}, "
            f"{n_theta}, {n_phi})  "
            f"| record: {_n_cols_hist} float32 "
            f"({n_atm_abs + n_oce_abs} path-lengths + {n_pstk} Stokes "
            f"+ {n_low} ALIS weights + 7 scalars)  "
            f"| GPU: {_hist_bytes / 1024**2:.1f} MB  "
            f"| CPU (on transfer): {_hist_bytes / 1024**2:.1f} MB"
        )
    else:
        tab_hist_tot = gpuzeros((1), dtype=np.float32)

    # Local estimate angles
    if le is not None:
        tab_thv = to_gpu(le['th'].astype('float32'))
        tab_phi = to_gpu(le['phi'].astype('float32'))
        if 'count_level' in le:
            tab_level = to_gpu(le['count_level'].astype('int32'))
        else:
            tab_level = to_gpu(np.full((n_theta), -2).astype('int32'))
    else:
        tab_thv = gpuzeros(1, dtype='float32')
        tab_phi = gpuzeros(1, dtype='float32')
        tab_level = to_gpu(np.array([-2]).astype('int32'))

    secs_cuda_clock = 0.0
    alis_norm = n_lam if n_low != 0 else 1
    n_photons_target = n_photons
    while (np.sum(n_photons_in_tot.get()) / alis_norm) < n_photons_target:
        tab_photons.fill(0.0)
        tab_photons_no_aer.fill(0.0)
        n_photons_out.fill(0)
        n_photons_out_no_aer.fill(0)
        n_photons_in.fill(0)
        counter.fill(0)
        tab_obj_info.fill(0)
        w_ph_cat.fill(0)
        w_ph_cat2.fill(0)
        w_ph_loss.fill(0)
        w_ph_loss2.fill(0)
        n_threads_active.fill(xblock * xgrid)

        start_cuda_clock = (
            cuda.Event()  # pyright: ignore[reportAttributeAccessIssue]
        )
        end_cuda_clock = (
            cuda.Event()  # pyright: ignore[reportAttributeAccessIssue]
        )
        start_cuda_clock.record()

        # Kernel launch
        kernel(
            envmap,
            spectrum,
            x0,
            faer,
            foce,
            errorcount,
            n_threads_active,
            tab_photons,
            tab_dist,
            tab_hist_tot,
            max_hist,
            tab_photons_no_aer,
            tab_trans_dir,
            counter,
            n_photons_in,
            n_photons_out,
            n_photons_out_no_aer,
            tab_thv,
            tab_phi,
            tab_level,
            tab_sensor,
            prof_atm,
            prof_oc,
            cell_atm,
            cell_oc,
            wavelength_proba_icdf,
            sensor_proba_icdf,
            cell_proba_icdf,
            rng.state,
            tab_obj_info,
            lobj_gpu,
            lgobj_gpu,
            lrobj_gpu,
            lobj_spect,
            n_ph_cat,
            w_ph_cat,
            w_ph_cat2,
            w_ph_loss,
            w_ph_loss2,
            tab_dir_le,
            block=(xblock, 1, 1),
            grid=(xgrid, 1, 1),
        )

        end_cuda_clock.record()
        end_cuda_clock.synchronize()
        secs_cuda_clock += start_cuda_clock.time_till(end_cuda_clock)

        (
            cuda.Context  # pyright: ignore[reportAttributeAccessIssue]
        ).synchronize()
        np.set_printoptions(precision=5, linewidth=150)

        if receiver_cell_size is not None:
            assert tab_mat_recep is not None
            assert mat_cats is not None
            assert mat_loss is not None
            # Matrix with the photon weight distribution on the receiver
            # surface.
            tab_mat_recep += tab_obj_info[:, :, :].get()
            # Fill loss matrix with photon weights used for loss
            # estimates.
            mat_loss[:, 0] += w_ph_loss[:].get()
            mat_loss[:, 1] += w_ph_loss2[:].get()
            # Fill category matrix.
            mat_cats[0, 1] += np.sum(w_ph_cat[:, :].get())
            mat_cats[0, 2] += np.sum(w_ph_cat2[:, :].get())
            w_ph_cat_tot += w_ph_cat
            w_ph_cat2_tot += w_ph_cat2
            for i in range(0, 8):
                mat_cats[i + 1, 1] += np.sum(w_ph_cat[i, :].get())
                mat_cats[i + 1, 2] += np.sum(w_ph_cat2[i, :].get())

        launched_last = n_photons_in
        n_photons_in_tot += launched_last

        n_photons_out_tot += n_photons_out
        sum_weights = tab_photons

        n_photons_out_tot_no_aer += n_photons_out_no_aer
        sum_weights_no_aer = tab_photons_no_aer

        if not hist:
            tab_photons_tot += sum_weights
            tab_photons_tot_no_aer += sum_weights_no_aer

        tab_dist_tot += tab_dist

        n_simu += 1

        sphot = np.sum(n_photons_in_tot.get()) / alis_norm
        if stdev:
            n_sensor_cur, n_lam_cur = n_photons_in.shape
            launched_last = launched_last.reshape(
                (1, 1, n_sensor_cur, n_lam_cur, 1, 1)
            )
            s_over_l = sum_weights.get() / launched_last.get()
            sum_x += s_over_l
            sum_x2 += s_over_l**2

            if stdev_lim is not None:
                sigma_bis = np.sqrt(sum_x2 / n_simu - (sum_x / n_simu) ** 2)
                sigma_bis /= np.sqrt(n_simu)
                sigma_bis[np.isnan(sigma_bis)] = 0

                abs_min = stdev_lim.dict['err_abs_min']
                rel_min = stdev_lim.dict['err_rel_min']
                min_loop = stdev_lim.dict['n_loop_min']
                stk_stdev = stdev_lim.dict['stk']
                level_stdev = stdev_lim.dict['level']
                format_std = stdev_lim.dict['format']

                avg = sum_x / n_simu
                err_rel = (sigma_bis / avg) * 100
                err_rel[np.isnan(err_rel)] = 0
                max_rerr = np.max(err_rel[level_stdev, stk_stdev, :, :, :, :])
                max_aerr = np.max(
                    sigma_bis[level_stdev, stk_stdev, :, :, :, :]
                )

                if stdev_lim.dict['verbose']:
                    print(
                        f"max rel_err = {max_rerr:{format_std}}; "
                        f"max abs_err = {max_aerr:{format_std}}"
                    )

                if (n_simu >= min_loop and max_aerr <= abs_min) or (
                    n_simu >= min_loop and max_rerr <= rel_min
                ):
                    progress.update(
                        sphot,
                        f"Launched {sphot:.3g} photons; "
                f"err[abs] = {max_aerr:{format_std}}; "
                f"err[rel] = {max_rerr:{format_std}};",
                    )
                    break

        if receiver_cell_size is not None and stdev_lim is not None:
            assert mat_cats is not None
            n_photons_tmp = np.sum(n_photons_in_tot.get())
            n_bis = n_photons_tmp / (n_photons_tmp - 1)
            sum_2z = (mat_cats[0, 1] * mat_cats[0, 1]) / n_photons_tmp
            sum_z2 = mat_cats[0, 2]
            if le is None:
                num = (n_bis * (sum_z2 - sum_2z)) ** 0.5
            else:
                num = (n_bis * abs(sum_z2 - sum_2z)) ** 0.5
            den = mat_cats[0, 1]
            err_p_tmp = (num / den) * 100
            min_loop = stdev_lim.dict['n_loop_min']
            rel_min = stdev_lim.dict['err_rel_min']
            format_std = stdev_lim.dict['format']

            if stdev_lim.dict['verbose']:
                print(f"relative_err = {err_p_tmp:{format_std}}")

            progress.update(
                sphot,
                f"Launched {sphot:.3g} photons; "
                f"err[rel] = {err_p_tmp:{format_std}};",
            )

            if n_simu >= min_loop and err_p_tmp <= rel_min:
                n_photons_target = n_photons_tmp
                break
        elif stdev and stdev_lim is not None:
            progress.update(
                sphot,
                f"Launched {sphot:.3g} photons; "
                f"err[abs] = {max_aerr:{format_std}}; "
                f"err[rel] = {max_rerr:{format_std}};",
            )
        else:
            progress.update(sphot, 'Launched {:.3g} photons'.format(sphot))

    # END WHILE LOOP
    secs_cuda_clock *= 1e-3

    if receiver_cell_size is not None:
        assert mat_cats is not None
        n_bis = n_photons_target / (n_photons_target - 1)
        # Count the total number of received photons and for each
        # category.
        mat_cats[0, 0] = np.sum(n_ph_cat[:].get())
        for i in range(0, 8):
            mat_cats[i + 1, 0] = n_ph_cat[i].get()

        # Relative and absolute error for sum of categories and per-
        # category values.
        for i in range(0, 9):
            if mat_cats[i, 0] != 0 and mat_cats[i, 1] != 0:
                sum_2z = (mat_cats[i, 1] * mat_cats[i, 1]) / n_photons_target
                sum_z2 = mat_cats[i, 2]
                if le is None:
                    mat_cats[i, 4] = (n_bis * (sum_z2 - sum_2z)) ** 0.5
                else:
                    mat_cats[i, 4] = (n_bis * abs(sum_z2 - sum_2z)) ** 0.5
                mat_cats[i, 5] = (mat_cats[i, 4] / mat_cats[i, 1]) * 100

    if stdev:
        sigma = np.sqrt(sum_x2 / n_simu - (sum_x / n_simu) ** 2)
        sigma /= np.sqrt(n_simu)
    else:
        sigma = None

    return (
        n_photons_in_tot.get(),
        tab_photons_tot.get(),
        tab_photons_tot_no_aer.get(),
        tab_dist_tot.get(),
        tab_hist_tot.get(),
        tab_trans_dir.get(),
        errorcount,
        n_photons_out_tot.get(),
        n_photons_out_tot_no_aer.get(),
        sigma,
        n_simu,
        secs_cuda_clock,
        tab_mat_recep,
        mat_cats,
        mat_loss,
        w_ph_cat_tot.get(),
        w_ph_cat2_tot.get(),
    )


def _get_git_attrs() -> dict:
    """Retrieve git repository metadata as output attributes.

    Queries the current git repository for the HEAD commit hash and
    working tree status. Returns an empty dict silently if git is
    unavailable or the current directory is not inside a git repository.

    Returns
    -------
    dict
        Dictionary with zero or more of the following keys:

        ``'git_commit_ref'`` : bytes
            SHA-1 hash of the current HEAD commit.
        ``'git_dirty_repo'`` : int
            1 if the working tree has uncommitted tracked-file changes,
            0 otherwise.
    """
    attrs = {}

    # Try to find git executable
    import shutil

    git_cmd = shutil.which('git')
    if git_cmd is None:
        # Git not found in PATH, try common locations
        for git_path in ['/usr/bin/git', '/usr/local/bin/git', '/bin/git']:
            if os.path.exists(git_path):
                git_cmd = git_path
                break

    if git_cmd is None:
        # Git not available, return empty dict
        return {}

    # check current commit
    p = subprocess.Popen(
        [git_cmd, 'rev-parse', 'HEAD'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.wait():
        return {}
    else:
        shasum = p.communicate()[0].strip()
        attrs.update({'git_commit_ref': shasum})

    # check if repo is dirty
    p = subprocess.Popen(
        [git_cmd, 'status', '--porcelain', '--untracked-files=no'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.wait():
        return {}
    else:
        is_dirty = len(p.communicate()[0]) != 0
        attrs.update({'git_dirty_repo': int(is_dirty)})
    return attrs


def _impact_init(
    prof_atm,
    nlam: int,
    thv_deg: float,
    earth_radius: float,
    pp: bool,
) -> tuple:
    """Compute the TOA entry point and the direct transmittance.

    Calculates the cartesian coordinates of the photon entry point at
    the top of atmosphere and the direct (Beer-Lambert) transmittance
    through the atmosphere for each wavelength.

    Parameters
    ----------
    prof_atm : xr.Dataset | MLUT | None
        Atmospheric profile containing ``z_atm`` coordinates (km) and
        ``OD_atm`` optical depth array (a 3D profile carries the
        ``iopt`` optical-property axis instead of ``z_atm``). If
        None, no atmosphere is assumed.
    nlam : int
        Number of wavelengths.
    thv_deg : float
        Solar/viewing zenith angle in degrees.
    earth_radius : float
        Earth radius in km.
    pp : bool
        If True, use plane-parallel geometry; if False, use spherical
        geometry.

    Returns
    -------
    x0_gpu : pycuda.gpuarray.GPUArray
        GPU array of shape (3,) containing the cartesian coordinates
        ``[x0, y0, z0]`` (float32) of the atmosphere entry point.
    tab_trans_dir : numpy.ndarray
        Array of shape (nlam,) with the direct transmittance
        ``exp(-tau_total)`` for each wavelength.
    """
    if prof_atm is None:
        h_atm = 0.0
        natm = 0
        z_atm = None
        od_atm = None
    else:
        if hasattr(prof_atm, 'to_xarray'):
            prof_atm = prof_atm.to_xarray()
        if 'z_atm' in prof_atm.coords:
            z_atm = prof_atm.coords['z_atm'].to_numpy()
            h_atm = z_atm[0]
            natm = len(z_atm) - 1
        else:
            # 3D profile: no vertical axis, the iopt axis indexes the
            # unique optical properties
            z_atm = None
            h_atm = 0.0
            natm = prof_atm.sizes['iopt'] - 1
        od_atm = prof_atm['OD_atm'].to_numpy()

    vx = -np.sin(thv_deg * np.pi / 180)
    vy = 0.0
    vz = -np.cos(thv_deg * np.pi / 180)
    earth_radius = np.double(earth_radius)

    tautot = np.zeros(nlam, dtype=np.float64)

    if pp:
        z0 = h_atm
        x0 = h_atm * np.tan(thv_deg * np.pi / 180.0)
        y0 = 0.0

        if natm != 0:
            assert prof_atm is not None and od_atm is not None
            for ilam in range(nlam):
                if prof_atm['OD_atm'].ndim == 2:
                    # lam, z
                    # tautot[ilam] = prof_atm['OD_atm'][ilam,
                    # natm]/np.cos(thv_deg*pi/180.)
                    tautot[ilam] = od_atm[ilam, -1] / np.cos(
                        thv_deg * np.pi / 180.0
                    )
                elif prof_atm['OD_atm'].ndim == 1:
                    # z
                    # tautot[ilam] =
                    # prof_atm['OD_atm'][natm]/np.cos(thv_deg*pi/180.)
                    tautot[ilam] = od_atm[-1] / np.cos(thv_deg * np.pi / 180.0)
                else:
                    raise ValueError(
                        'invalid number of dimensions in prof_atm'
                    )
    else:
        tanthv = np.tan(thv_deg * np.pi / 180.0)

        # Pythagorean theorem in right triangle OMZ, where:
        # * O is the center of the earth
        # * M is the entry point in the atmosphere, has cartesian
        # coordinates (x0, y0, earth_radius+z0)
        #     (origin is at the surface)
        # * Z is the projection of M on z axis
        # tan(thv) = x0/z0
        # earth_radius is the radius of the earth and h_atm the
        # thickness of the atmosphere
        # solve the equation x0^2 + (earth_radius+z0)^2 =
        # (earth_radius+h_atm)^2 for z0
        delta = 4 * earth_radius**2 + 4 * (tanthv**2 + 1) * (
            h_atm**2 + 2 * h_atm * earth_radius
        )
        z0 = (-2.0 * earth_radius + np.sqrt(delta)) / (2 * (tanthv**2 + 1.0))
        x0 = z0 * tanthv
        y0 = 0.0
        z0 += earth_radius

        # loop over the NATM atmosphere layers to find the total optical
        # thickness
        xph = x0
        yph = y0
        zph = z0
        for i in range(1, natm + 1):
            # natm > 0 implies that a profile was given
            assert z_atm is not None and od_atm is not None
            # V is the direction vector, X is the position vector, d is
            # the
            # distance to the next layer and R is the position vector at
            # the
            # next layer
            # we have: R = X + V.d
            # R² = X² + (V.d)² + 2XVD
            # where R is earth_radius+ALT[i]
            # solve for d:
            delta = 4.0 * (vx * xph + vy * yph + vz * zph) ** 2 - 4 * (
                (xph**2 + yph**2 + zph**2) - (earth_radius + z_atm[i]) ** 2
            )

            # the 2 solutions are:
            d1 = 0.5 * (
                -2.0 * (vx * xph + vy * yph + vz * zph) + np.sqrt(delta)
            )
            d2 = 0.5 * (
                -2.0 * (vx * xph + vy * yph + vz * zph) - np.sqrt(delta)
            )

            # the solution is the smallest positive one
            if d1 > 0:
                if d2 > 0:
                    d = min(d1, d2)
                else:
                    d = d1
            else:
                if d2 > 0:
                    d = d2
                else:
                    raise RuntimeError('No solution in _impact_init')

            # photon moves forward
            xph += vx * d
            yph += vy * d
            zph += vz * d

            for ilam in range(nlam):
                # optical thickness of the layer in vertical direction
                hlay0 = abs(od_atm[ilam, i] - od_atm[ilam, i - 1])

                # thickness of the layer
                d0 = abs(z_atm[i - 1] - z_atm[i])

                # optical thickness of the layer at current wavelength
                hlay = hlay0 * d / d0

                # cumulative optical thickness
                tautot[ilam] += hlay

    return to_gpu(np.array([x0, y0, z0], dtype='float32')), np.exp(-tautot)


def _init_rng(rng: str) -> '_RngPhilox | _RngCurandPhilox':
    """
    Return the RNG backend instance for the given RNG name.

    Parameters
    ----------
    rng : str
        The random-number generator name: 'PHILOX' or 'CURAND_PHILOX'.

    Returns
    -------
    _RngPhilox | _RngCurandPhilox
        The RNG backend.
    """
    if rng == 'PHILOX':
        return _RngPhilox()
    elif rng == 'CURAND_PHILOX':
        return _RngCurandPhilox()
    else:
        raise ValueError('Invalid RNG "{}"'.format(rng))


class _RngPhilox(object):
    """Philox random-number generator backend.

    This helper manages the RNG seed and state buffer for Philox-based
    random number generation on the GPU.
    """

    def __init__(self) -> None:
        pass

    def setup(self, seed: int, xblock: int, xgrid: int) -> int:
        """Initialize Philox RNG state on GPU.

        Parameters
        ----------
        seed : int
            Seed value for the random-number generator. If -1, seed is
            derived from current UTC time (rounded to nearest second).
        xblock : int
            Number of threads per block in the GPU kernel launch.
        xgrid : int
            Number of blocks (grid size) for GPU kernel launch.

        Returns
        -------
        int
            The seed value used to initialize the RNG state. If input
            was -1, returns the generated timestamp-based seed;
            otherwise returns the input seed.

        Notes
        -----
        This method allocates GPU memory for the RNG state buffer and
        transfers it to device. The state buffer has size
        ``xblock*xgrid+1`` elements.
        """
        if seed == -1:
            # seed is based on clock
            # A multiply by 1000 has been removed to avoid OverflowError
            # due to uint32 limit
            seed = int(
                np.uint32(
                    (
                        datetime.now(tz=timezone.utc)
                        - datetime(1970, 1, 1, tzinfo=timezone.utc)
                    ).total_seconds()
                )
            )

        state = np.zeros(xblock * xgrid + 1, dtype='uint32')
        state[0] = seed
        self.state = to_gpu(state)

        return seed


class _RngCurandPhilox(object):
    """CURAND Philox random-number generator backend.

    This helper wraps a tiny CUDA module that initializes
    ``curandStatePhilox4_32_10_t`` states on device memory for all
    active threads.
    """

    def __init__(self) -> None:
        # build module containing the initialization functions
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
            int idx = (blockIdx.x * YGRIDd + blockIdx.y)
                      * XBLOCKd * YBLOCKd
                      + (threadIdx.x * YBLOCKd + threadIdx.y);
            curand_init(SEEDd, idx, 0, &state[idx]);
        }
        }
        '''
        self.mod = SourceModule(source, no_extern_c=True)

        # get state size
        s = gpuzeros(1, dtype=np.uint32)
        self.mod.get_function('get_state_size')(
            s, block=(1, 1, 1), grid=(1, 1, 1)
        )
        self.STATE_SIZE = int(np.squeeze(s.get()))  # size in bytes

    def setup(self, seed: int, xblock: int, xgrid: int) -> int:
        """Initialize CURAND Philox RNG state on GPU.

        Parameters
        ----------
        seed : int
            Seed value for the random-number generator. If -1, seed is
            derived from current UTC time (rounded to nearest second).
        xblock : int
            Number of threads per block in the GPU kernel launch.
        xgrid : int
            Number of blocks (grid size) for GPU kernel launch.

        Returns
        -------
        int
            The seed value used to initialize the RNG state. If input
            was -1, returns the generated timestamp-based seed;
            otherwise returns the input seed.

        Notes
        -----
        This method initializes ``curandStatePhilox4_32_10_t`` states on
        device memory for all threads in the GPU grid. It configures GPU
        global variables (XBLOCKd, XGRIDd, SEEDd) and launches the setup
        kernel to initialize the RNG state buffer.
        """
        if seed == -1:
            # seed is based on clock
            seed = int(
                np.uint32(
                    (
                        datetime.now(tz=timezone.utc)
                        - datetime(1970, 1, 1, tzinfo=timezone.utc)
                    ).total_seconds()
                )
            )

        cuda.memcpy_htod(  # pyright: ignore[reportAttributeAccessIssue]
            self.mod.get_global('XBLOCKd')[0],
            np.array([xblock], dtype=np.int32),
        )
        cuda.memcpy_htod(  # pyright: ignore[reportAttributeAccessIssue]
            self.mod.get_global('XGRIDd')[0], np.array([xgrid], dtype=np.int32)
        )
        cuda.memcpy_htod(  # pyright: ignore[reportAttributeAccessIssue]
            self.mod.get_global('SEEDd')[0], np.array([seed], dtype=np.int32)
        )

        # setup RNG
        self.state = gpuzeros(self.STATE_SIZE * xblock * xgrid, dtype='uint8')
        setup = self.mod.get_function('setup')
        setup(self.state, block=(xblock, 1, 1), grid=(xgrid, 1, 1))

        return seed


def _init_obj(lgobj, v_sun, wavelength, cus_l=None) -> tuple:
    """Initialize object-related GPU buffers and receiver metadata.

    Parameters
    ----------
    lgobj : list
        List of object groups/entities used by the 3-D object mode.
    v_sun : gc.Vector
        Sun direction vector, used in restricted-forward (``RF``) mode.
    wavelength : float or array-like or BandSet
        Wavelength definition in nm. It can also be a list of
        REPTRAN/KDIS bands and will be converted to ``BandSet`` when
        needed.
    cus_l : object, optional
        Custom launching mode object (for example ``CusForward`` or
        ``CusBackward``). Default is ``None``.

    Returns
    -------
    tuple
        ``(n_gobj, n_obj, n_robj, surf_lph, n_h, z_alt_h, tot_s_h, tc,
        n_cx, n_cy, lobj_gpu, lgobj_gpu, lrobj_gpu, lobj_spect,
        n_cos)``.
    """

    index_offset = 0
    lobj = []
    n_gobj = len(lgobj)
    ind_robj = []
    lgobj_gpu = np.zeros(n_gobj, dtype=TYPE_GOBJ, order='C')

    # Build a flat list of entities and a GPU table of object-group
    # parameters.
    for i in range(0, n_gobj):
        lgobj_gpu['index'][i] = index_offset
        lgobj_gpu['bPminx'][i] = lgobj[i].bbox_pmin.x
        lgobj_gpu['bPminy'][i] = lgobj[i].bbox_pmin.y
        lgobj_gpu['bPminz'][i] = lgobj[i].bbox_pmin.z
        lgobj_gpu['bPmaxx'][i] = lgobj[i].bbox_pmax.x
        lgobj_gpu['bPmaxy'][i] = lgobj[i].bbox_pmax.y
        lgobj_gpu['bPmaxz'][i] = lgobj[i].bbox_pmax.z
        if lgobj[i].check == "GroupE":
            lgobj_gpu['nObj'][i] = lgobj[i].nob
            index_offset += lgobj[i].nob
            lobj.extend(lgobj[i].le)
        elif lgobj[i].check == "Entity":
            lgobj_gpu['nObj'][i] = 1
            index_offset += 1
            lobj.append(lgobj[i])
        else:
            raise ValueError(
                'In the my_objects list, only Entity and GroupE '
                'classes are authorized!'
            )

    lgobj_gpu = to_gpu(lgobj_gpu)
    n_obj = len(lobj)

    if cus_l is not None and cus_l.dict['mode'] == "BR":
        lobj_gpu = np.zeros(n_obj + 1, dtype=TYPE_IOBJECTS, order='C')
        tc = cus_l.dict['receiver'].tc
        size_x_min = min(
            cus_l.dict['receiver'].geo.p1.x,
            cus_l.dict['receiver'].geo.p2.x,
            cus_l.dict['receiver'].geo.p3.x,
            cus_l.dict['receiver'].geo.p4.x,
        )
        size_x_max = max(
            cus_l.dict['receiver'].geo.p1.x,
            cus_l.dict['receiver'].geo.p2.x,
            cus_l.dict['receiver'].geo.p3.x,
            cus_l.dict['receiver'].geo.p4.x,
        )
        size_x = size_x_max - size_x_min
        size_y_min = min(
            cus_l.dict['receiver'].geo.p1.y,
            cus_l.dict['receiver'].geo.p2.y,
            cus_l.dict['receiver'].geo.p3.y,
            cus_l.dict['receiver'].geo.p4.y,
        )
        size_y_max = max(
            cus_l.dict['receiver'].geo.p1.y,
            cus_l.dict['receiver'].geo.p2.y,
            cus_l.dict['receiver'].geo.p3.y,
            cus_l.dict['receiver'].geo.p4.y,
        )
        size_y = size_y_max - size_y_min
        n_cx = int(size_x / tc)
        n_cy = int(size_y / tc)
        lobj_gpu['mvRx'][n_obj] = cus_l.dict['receiver'].transformation.rotx
        lobj_gpu['mvRy'][n_obj] = cus_l.dict['receiver'].transformation.roty
        lobj_gpu['mvRz'][n_obj] = cus_l.dict['receiver'].transformation.rotz
        if cus_l.dict['receiver'].transformation.rot_order == "XYZ":
            lobj_gpu['rotOrder'][n_obj] = 1
        elif cus_l.dict['receiver'].transformation.rot_order == "XZY":
            lobj_gpu['rotOrder'][n_obj] = 2
        elif cus_l.dict['receiver'].transformation.rot_order == "YXZ":
            lobj_gpu['rotOrder'][n_obj] = 3
        elif cus_l.dict['receiver'].transformation.rot_order == "YZX":
            lobj_gpu['rotOrder'][n_obj] = 4
        elif cus_l.dict['receiver'].transformation.rot_order == "ZXY":
            lobj_gpu['rotOrder'][n_obj] = 5
        elif cus_l.dict['receiver'].transformation.rot_order == "ZYX":
            lobj_gpu['rotOrder'][n_obj] = 6
        else:
            raise ValueError('Unknown rotation order')
        lobj_gpu['mvTx'][n_obj] = cus_l.dict['receiver'].transformation.transx
        lobj_gpu['mvTy'][n_obj] = cus_l.dict['receiver'].transformation.transy
        lobj_gpu['mvTz'][n_obj] = cus_l.dict['receiver'].transformation.transz

        ind_robj.append(n_obj)  # For creating a receiver-only GPU table.
    else:
        lobj_gpu = np.zeros(n_obj, dtype=TYPE_IOBJECTS, order='C')
        tc = None
        n_cx = int(0)
        n_cy = int(0)

    # Account for spectral variability of object reflectivity.
    n_obj_total = lobj_gpu.size
    if not isinstance(wavelength, BandSet):
        wavelength = BandSet(wavelength)
    nlam = wavelength.size
    lobj_spect = np.zeros(
        (n_obj_total * nlam), dtype=TYPE_SPECTRUM_OBJ, order='C'
    )

    # Initialization before object loop.
    pp1 = 0.0
    pp2 = 0.0
    pp3 = 0.0
    pp4 = 0.0
    n_h = 0
    z_alt_h = 0.0
    tot_s_h = 0.0
    ncos = 0.0
    if cus_l is not None and cus_l.dict['mode'] == "RF":
        surf_lph = 0
    else:
        surf_lph = None

    # Iterate over all objects.
    for i in range(0, n_obj):
        normal_base = gc.Vector(0.0, 0.0, 1.0)
        if isinstance(lobj[i].geo, Spheric):
            lobj_gpu['geo'][i] = 1
            lobj_gpu['myRad'][i] = lobj[i].geo.radius
            lobj_gpu['z0'][i] = lobj[i].geo.z0
            lobj_gpu['z1'][i] = lobj[i].geo.z1
            lobj_gpu['phi'][i] = lobj[i].geo.phi
        elif isinstance(lobj[i].geo, Plane):
            lobj_gpu['geo'][i] = 2
            lobj_gpu['p0x'][i] = lobj[i].geo.p1.x
            lobj_gpu['p0y'][i] = lobj[i].geo.p1.y
            lobj_gpu['p0z'][i] = lobj[i].geo.p1.z
            lobj_gpu['p1x'][i] = lobj[i].geo.p2.x
            lobj_gpu['p1y'][i] = lobj[i].geo.p2.y
            lobj_gpu['p1z'][i] = lobj[i].geo.p2.z
            lobj_gpu['p2x'][i] = lobj[i].geo.p3.x
            lobj_gpu['p2y'][i] = lobj[i].geo.p3.y
            lobj_gpu['p2z'][i] = lobj[i].geo.p3.z
            lobj_gpu['p3x'][i] = lobj[i].geo.p4.x
            lobj_gpu['p3y'][i] = lobj[i].geo.p4.y
            lobj_gpu['p3z'][i] = lobj[i].geo.p4.z

            # Normal of the plane object after applying rotation
            # transform.
            normal_base = gc.Vector(0, 0, 1)
            tp_rx0 = gc.get_rotate_x_tf(lobj[i].transformation.rotation[0])
            tp_ry0 = gc.get_rotate_y_tf(lobj[i].transformation.rotation[1])
            tp_rz0 = gc.get_rotate_z_tf(lobj[i].transformation.rotation[2])
            if lobj[i].transformation.rot_order == "XYZ":
                tp_t0 = tp_rx0 * tp_ry0 * tp_rz0
            elif lobj[i].transformation.rot_order == "XZY":
                tp_t0 = tp_rx0 * tp_rz0 * tp_ry0
            elif lobj[i].transformation.rot_order == "YXZ":
                tp_t0 = tp_ry0 * tp_rx0 * tp_rz0
            elif lobj[i].transformation.rot_order == "YZX":
                tp_t0 = tp_ry0 * tp_rz0 * tp_rx0
            elif lobj[i].transformation.rot_order == "ZXY":
                tp_t0 = tp_rz0 * tp_rx0 * tp_ry0
            elif lobj[i].transformation.rot_order == "ZYX":
                tp_t0 = tp_rz0 * tp_ry0 * tp_rx0
            else:
                raise ValueError('Unknown rotation order')

            normal_base = tp_t0(normal_base)
            normal_base = gc.normalize(normal_base)
            lobj_gpu['nBx'][i] = normal_base.x
            lobj_gpu['nBy'][i] = normal_base.y
            lobj_gpu['nBz'][i] = normal_base.z
        else:
            raise ValueError(
                "Your geometry can be only spheric or plane, please "
                "choose between Spheric or Plane classes!"
            )

        # Apply transformation parameters.
        lobj_gpu['mvRx'][i] = lobj[i].transformation.rotx
        lobj_gpu['mvRy'][i] = lobj[i].transformation.roty
        lobj_gpu['mvRz'][i] = lobj[i].transformation.rotz
        if lobj[i].transformation.rot_order == "XYZ":
            lobj_gpu['rotOrder'][i] = 1
        elif lobj[i].transformation.rot_order == "XZY":
            lobj_gpu['rotOrder'][i] = 2
        elif lobj[i].transformation.rot_order == "YXZ":
            lobj_gpu['rotOrder'][i] = 3
        elif lobj[i].transformation.rot_order == "YZX":
            lobj_gpu['rotOrder'][i] = 4
        elif lobj[i].transformation.rot_order == "ZXY":
            lobj_gpu['rotOrder'][i] = 5
        elif lobj[i].transformation.rot_order == "ZYX":
            lobj_gpu['rotOrder'][i] = 6
        else:
            raise ValueError('Unknown rotation order')
        lobj_gpu['mvTx'][i] = lobj[i].transformation.transx
        lobj_gpu['mvTy'][i] = lobj[i].transformation.transy
        lobj_gpu['mvTz'][i] = lobj[i].transformation.transz

        # Front material (AV).
        lobj_gpu['materialAV'][i] = 0
        lobj_gpu['shdAV'][i] = 0
        lobj_gpu['nindAV'][i] = 1
        lobj_gpu['distAV'][i] = 0
        lobj_gpu['reflectAV'][i] = 0
        if np.array(lobj[i].material_front.reflectivity).size == 1:
            lobj_spect['reflectAV'][(i * nlam) : ((i * nlam) + nlam)] = (
                np.full((nlam), lobj[i].material_front.reflectivity)
            )
        elif lobj[i].material_front.reflectivity.size != nlam:
            raise ValueError(
                'The number of reflectivities must be equal to the '
                'number of wavelengths!'
            )
        else:
            lobj_spect['reflectAV'][(i * nlam) : ((i * nlam) + nlam)] = lobj[
                i
            ].material_front.reflectivity[:]

        if isinstance(lobj[i].material_front, LambMirror):
            lobj_gpu['materialAV'][i] = 1
            lobj_gpu['roughAV'][i] = 0.0
        elif isinstance(lobj[i].material_front, Matte):
            lobj_gpu['materialAV'][i] = 2
            lobj_gpu['roughAV'][i] = lobj[i].material_front.roughness
        elif isinstance(lobj[i].material_front, Mirror):
            lobj_gpu['materialAV'][i] = 3
            lobj_gpu['shdAV'][i] = int(lobj[i].material_front.shadow)
            lobj_gpu['nindAV'][i] = lobj[i].material_front.nind
            lobj_gpu['distAV'][i] = lobj[i].material_front.distribution
            lobj_gpu['roughAV'][i] = lobj[i].material_front.roughness
        else:
            raise ValueError('Unknown material AV')

        # Back material (AR).
        lobj_gpu['materialAR'][i] = 0
        lobj_gpu['shdAR'][i] = 0
        lobj_gpu['nindAR'][i] = 1
        lobj_gpu['distAR'][i] = 0
        lobj_gpu['reflectAR'][i] = 0
        if np.array(lobj[i].material_back.reflectivity).size == 1:
            lobj_spect['reflectAR'][(i * nlam) : ((i * nlam) + nlam)] = (
                np.full((nlam), lobj[i].material_back.reflectivity)
            )
        elif lobj[i].material_back.reflectivity.size != nlam:
            raise ValueError(
                'The number of reflectivities must be equal to the '
                'number of wavelengths!'
            )
        else:
            lobj_spect['reflectAR'][(i * nlam) : ((i * nlam) + nlam)] = lobj[
                i
            ].material_back.reflectivity[:]

        if isinstance(lobj[i].material_back, LambMirror):
            lobj_gpu['materialAR'][i] = 1
            lobj_gpu['roughAR'][i] = 0.0
        elif isinstance(lobj[i].material_back, Matte):
            lobj_gpu['materialAR'][i] = 2
            lobj_gpu['roughAR'][i] = lobj[i].material_back.roughness
        elif isinstance(lobj[i].material_back, Mirror):
            lobj_gpu['materialAR'][i] = 3
            lobj_gpu['shdAR'][i] = int(lobj[i].material_back.shadow)
            lobj_gpu['nindAR'][i] = lobj[i].material_back.nind
            lobj_gpu['distAR'][i] = lobj[i].material_back.distribution
            lobj_gpu['roughAR'][i] = lobj[i].material_back.roughness
        else:
            raise ValueError('Unknown material AR')

        # Object role: reflector, receiver, or environment.
        if lobj[i].name == "reflector":
            lobj_gpu['type'][i] = 1

            if isinstance(lobj[i].geo, Plane) and (
                isinstance(lobj[i].material_back, Mirror)
                or isinstance(lobj[i].material_front, Mirror)
            ):
                n_h += 1
                z_alt_h += lobj[i].transformation.transz
                tot_s_h += abs(lobj[i].geo.p1.x) * abs(lobj[i].geo.p1.y) * 4
                ncos += gc.dot(
                    normal_base, gc.Vector(-v_sun.x, -v_sun.y, -v_sun.z)
                )

            if cus_l is not None and cus_l.dict['mode'] == "RF":
                pp1 = lobj[i].geo.p1
                pp2 = lobj[i].geo.p2
                pp3 = lobj[i].geo.p3
                pp4 = lobj[i].geo.p4
                dot_p = gc.dot(v_sun * -1, normal_base)
                two_aa_bis = abs((pp1.x - pp4.x) * (pp2.y - pp3.y)) + abs(
                    (pp2.x - pp3.x) * (pp1.y - pp4.y)
                )
                surf_lph_bis = (two_aa_bis / 2.0) * dot_p
                surf_lph += surf_lph_bis
        elif lobj[i].name == "receiver":
            lobj_gpu['type'][i] = 2
            tc = lobj[i].tc
            size_x_min = min(
                lobj[i].geo.p1.x,
                lobj[i].geo.p2.x,
                lobj[i].geo.p3.x,
                lobj[i].geo.p4.x,
            )
            size_x_max = max(
                lobj[i].geo.p1.x,
                lobj[i].geo.p2.x,
                lobj[i].geo.p3.x,
                lobj[i].geo.p4.x,
            )
            size_x = size_x_max - size_x_min
            size_y_min = min(
                lobj[i].geo.p1.y,
                lobj[i].geo.p2.y,
                lobj[i].geo.p3.y,
                lobj[i].geo.p4.y,
            )
            size_y_max = max(
                lobj[i].geo.p1.y,
                lobj[i].geo.p2.y,
                lobj[i].geo.p3.y,
                lobj[i].geo.p4.y,
            )
            size_y = size_y_max - size_y_min
            n_cx = int(size_x / tc)
            n_cy = int(size_y / tc)
            ind_robj.append(i)
        elif lobj[i].name == "environment":
            lobj_gpu['type'][i] = 3
        else:
            raise ValueError(
                'You have to specify if your object is a reflector '
                'or a receiver!'
            )

    # Create receiver-only GPU table.
    n_robj = len(ind_robj)
    if n_robj > 0:
        lrobj_gpu = np.zeros(n_robj, dtype=TYPE_IOBJECTS, order='C')
        for i in range(0, n_robj):
            lrobj_gpu[:][i] = lobj_gpu[:][ind_robj[i]]
    else:
        lrobj_gpu = np.zeros(1, dtype=TYPE_IOBJECTS, order='C')

    lobj_gpu = to_gpu(lobj_gpu)
    lrobj_gpu = to_gpu(lrobj_gpu)
    lobj_spect = to_gpu(lobj_spect)
    if n_h > 0:
        n_cos = ncos / n_h
    else:
        n_cos = 1

    return (
        n_gobj,
        n_obj,
        n_robj,
        surf_lph,
        n_h,
        z_alt_h,
        tot_s_h,
        tc,
        n_cx,
        n_cy,
        lobj_gpu,
        lgobj_gpu,
        lrobj_gpu,
        lobj_spect,
        n_cos,
    )


def _normalize_rec(
    c_mat_visu_recep: np.ndarray,
    mat_cats: np.ndarray,
    n_cx: int,
    n_cy: int,
    n_photons: float,
    surf_lph,
    cell_size: float,
    cus_l,
    le: int,
) -> tuple:
    """
    Normalize receiver signal.

    This function normalizes the signal collected by a 3d object
    receiver. Multiplication by the solar irradiance at the top of
    atmosphere is still needed.

    Parameters
    ----------
    c_mat_visu_recep : ndarray
        3D array containing the signal weight collected by each cell of
        the receiver.
    mat_cats : ndarray
        2D array containing total signal and per-category breakdowns.
        Rows correspond to categories, columns to the per-category
        weight sums.
    n_cx : int
        Number of receiver cells in the x direction.
    n_cy : int
        Number of receiver cells in the y direction.
    n_photons : float
        Total number of launched photons in the simulation.
    surf_lph : float
        Illuminated surface area (km²) for launching mode "FF" or "RF".
    cell_size : float
        Side length (km) of a square receiver cell (taille cellule).
    cus_l : object or None
        Custom launching mode object with attributes like
        ``dict['mode']``, ``dict['fov']`` and, in the "B" and "BR"
        modes, ``dict['sun_fov']``, the half-angle (degrees) of the cone
        subtended by the sun. If `None`, no normalization is applied.
    le : bool
        Flag indicating whether the local estimate mode is enabled.

    Returns
    -------
    tuple of (ndarray, ndarray, float)
        - **c_mat_visu_recep** : normalized receiver signal matrix
        - **mat_cats** : normalized category matrix
        - **norm_c** : normalization constant (dimensionless)
    """
    s_rec = cell_size * cell_size * n_cx * n_cy  # receiver surface in km²
    s_rec_m = s_rec * 1e6  # receiver surface in m²

    # Normalize intensities such that only a mult by E_TOA is still
    # needed to obtain power unit
    if cus_l is None:
        norm_c = 1.0
        # norm_c = 1./n_photons
        # # Weights -> propor to w/m², mult by s_rec_m is needed to get
        # something propor to watt unit
        # norm_c *= s_rec_m
        # c_mat_visu_recep[:][:][:] = c_mat_visu_recep[:][:][:]*norm_c
        # for i in range (0, 9):
        #     mat_cats[i,3] = mat_cats[i,1]*norm_c # intensity
        #     mat_cats[i,4] *= norm_c # Absolute err
    elif cus_l.dict['mode'] == "FF" or cus_l.dict['mode'] == "RF":
        # Here results are already propor to watt unit
        norm_c = (
            surf_lph * 1e6
        ) / n_photons  # Here multiply by 1e6 to convert km² to m²
        norm_ff = 1.0
        # lambertian sampling normalization
        if (
            cus_l.dict['mode'] == "FF"
            and cus_l.dict['sampling_code'] == 1
            and cus_l.dict['fov'] > 1e-6
        ):
            norm_ff = (1 - np.cos(np.radians(2 * cus_l.dict['fov']))) / (
                4 * (1 - np.cos(np.radians(cus_l.dict['fov'])))
            )
        # isotropic sampling normalization
        elif (
            cus_l.dict['mode'] == "FF"
            and cus_l.dict['sampling_code'] == 2
            and cus_l.dict['fov'] > 1e-6
        ):
            norm_ff = 1.0
        norm_c *= norm_ff
        for i in range(0, 9):
            c_mat_visu_recep[i][:][:] = c_mat_visu_recep[i][:][:] * norm_c
            mat_cats[i, 3] = mat_cats[i, 1] * norm_c
            mat_cats[i, 4] *= norm_c
    elif cus_l.dict['mode'] == "B" or cus_l.dict['mode'] == "BR":
        norm_br = 2
        # lambertian sampling normalization
        if cus_l.dict['sampling_code'] == 1:
            norm_br = (
                1 - np.cos(np.radians(2 * cus_l.dict['receiver_fov']))
            ) / 2.0
        # isotropic sampling normalization
        elif cus_l.dict['sampling_code'] == 2:
            norm_br = 2 * (1 - np.cos(np.radians(cus_l.dict['receiver_fov'])))

        if not le:
            norm_c = norm_br / (
                n_photons
                * 2
                * (1 - np.cos(np.radians(cus_l.dict['sun_fov'])))
            )
        else:
            norm_c = norm_br / n_photons

        # Weights -> propor to w/m², mult by s_rec_m is needed to get
        # something propor to watt unit
        norm_c *= s_rec_m
        c_mat_visu_recep[:][:][:] = c_mat_visu_recep[:][:][:] * norm_c
        for i in range(0, 9):
            mat_cats[i, 3] = mat_cats[i, 1] * norm_c
            mat_cats[i, 4] *= norm_c
    else:
        raise ValueError('Unknown launching mode!')

    return c_mat_visu_recep, mat_cats, norm_c


def _find_extinction(ip, fp, prof_atm, w_ind: int = 0):
    """
    Compute the atmospheric extinction along a segment between two
    points.

    The extinction is computed as :math:`e^{-|\\Delta\\tau|}`, where
    :math:`\\Delta\\tau` is the cumulated optical depth along the path
    from `ip` to `fp`.

    .. note::
        Only valid for 1-D plane-parallel atmospheres.

    Parameters
    ----------
    ip : gc.Point
        Initial position.
    fp : gc.Point
        Final position.
    prof_atm : xarray.Dataset or object with ``to_xarray``
        Atmospheric profile containing coordinates ``z_atm`` and
        variable ``OD_atm`` (cumulated extinction optical depth from the
        top).
    w_ind : int, optional
        Wavelength index into ``OD_atm``. Default is 0.

    Returns
    -------
    float
        Extinction factor between `ip` and `fp` (dimensionless, in [0,
        1]).
    """
    # Be sure ip and fp are Point classes
    if not all(isinstance(i, gc.Point) for i in [ip, fp]):
        raise ValueError('Both ip and fp must be Point classes!')

    # If there is no atm then there are no scattering and abs -> n_ext =
    # 1
    if prof_atm is None:
        n_ext = 1
        return n_ext

    if hasattr(prof_atm, 'to_xarray'):
        prof_atm = prof_atm.to_xarray()

    if 'z_atm' not in prof_atm.coords:
        raise ValueError(
            "_find_extinction requires a 1D plane-parallel "
            "atmosphere profile (the 3D profile has no z_atm axis)"
        )
    zatm = prof_atm.coords['z_atm'].to_numpy()
    od_atm = prof_atm['OD_atm'].to_numpy()

    # Vector/direction from ip to fp
    vec = fp - ip

    # Find the atm layer of the initial location
    lay = int(0)
    while zatm[lay] > ip.z:
        lay += int(1)

    # Initialization
    tau_hit = 0.0  # Optical depth distance (from ip to fp)
    ilayer2 = lay

    # Case with only 1 layer: n = 1
    if fp.z >= zatm[ilayer2] and fp.z < zatm[ilayer2 - 1]:
        # delta_i is: Delta(tau)1 = |tau(i-1) - tau(i)|
        delta_i = abs(od_atm[w_ind, ilayer2 - 1] - od_atm[w_ind, ilayer2])
        # tau_hit = (Delta(D1)/Delta(Z1))*delta_i
        tau_hit += (
            (ip - fp).Length() / abs(zatm[ilayer2 - 1] - zatm[ilayer2])
        ) * delta_i
    else:  # Case with several layers: n >= 2
        # Find the layer where there is intersection
        ilayer2 = int(1)
        while zatm[ilayer2] > fp.z and zatm[ilayer2] > 0.0:
            ilayer2 += int(1)

        higher = False
        ilayer = lay
        old_p = ip

        # Check if the photon come from higher or lower layer
        if ilayer < ilayer2:  # true if the photon come from higher layer
            higher = True

        while ilayer != ilayer2:
            if higher:
                time_t = abs(zatm[ilayer] - old_p.z) / abs(vec.z)
            else:
                time_t = abs(zatm[ilayer - 1] - old_p.z) / abs(vec.z)
            new_p = old_p + (vec * time_t)
            delta_i = abs(od_atm[w_ind, ilayer] - od_atm[w_ind, ilayer - 1])
            tau_hit += (
                (new_p - old_p).Length() / abs(zatm[ilayer - 1] - zatm[ilayer])
            ) * delta_i

            if higher:  # the photon come from higher layer
                ilayer += int(1)
            else:  # the photon come from lower layer
                ilayer -= int(1)
            old_p = new_p  # Update the position of the photon

        # Calculate and add the last tau distance when ilayer is equal
        # to ilayer2
        delta_i = abs(od_atm[w_ind, ilayer2] - od_atm[w_ind, ilayer2 - 1])
        tau_hit += (
            (fp - old_p).Length() / abs(zatm[ilayer2 - 1] - zatm[ilayer2])
        ) * delta_i

    n_ext = np.exp(-abs(tau_hit))

    return n_ext
