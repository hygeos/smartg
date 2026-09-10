#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Non-regression test of the 3D atmosphere mode (opt3d=True) using the
# IPRT phase B cumulus cloud case (C3), with aerosols. Unlike the C2
# cubic cloud, this case mixes a realistic 100x100x53 cloud field with
# a 1D Rayleigh + aerosol profile, which is the configuration used for
# the 3MI scenes.
# Tested with the following GPUs: 5070 Ti
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartg import conftest
from smartg.atmosphere import (
    AerOPAC,
    Atm1D,
    Atm3D,
    Cloud3D,
    read_i3rc_cloud,
)
from smartg.config import DIR_AUXDATA
from smartg.diff import diff1
from smartg.grid3d import Grid3D
from smartg.iprt.iprt import compute_deltam, group_iquv
from smartg.sensor import get_sensors_grid
from smartg.view import satellite_view
from smartg.phase import read_phase_cdf
from smartg.albedo import AlbedoCst
from smartg.surface import LambSurface
from smartg.smartg import Smartg
from smartg.truncation import GT_trunc

# *********************** Global variable(s) ***************************
# Fixed seed: seed=-1 would derive it from the clock, giving a new
# noise realisation at every run. The reference delta_m values below
# were measured with this seed.
SEED = 1234

# The IPRT C3 benchmark uses 1e10 photons (see the report
# others/rapport_simulateur_3MI_HYGEOS_v1.2.pdf, section 4.5). A tenth
# of that is enough here because the sensor grid is also reduced, see
# NSENSORS below: the number of photons per sensor stays higher than in
# the benchmark.
NBPHOTONS = 1e9
NBLOOP = 1e7

# Every test runs in two tiers. The slow one uses the photon count
# above, a tenth of the IPRT benchmark, and is what the reference
# delta_m values below were measured with: it is deselected by default
# (see pytest.ini). The fast one divides it by PHOTON_DIVIDER and is
# the one that runs routinely.
#
# Dividing the photons multiplies the MC noise, and delta_m is then
# dominated by it: a small systematic bias would hide inside the
# DELTAM_TOL band. This is why the fast tier does not rely on delta_m
# alone, see MEAN_TOL below.
TIERS = ["fast", pytest.param("slow", marks=pytest.mark.slow)]
PHOTON_DIVIDER = {"fast": 30, "slow": 1}

# Two sided fractional band around the reference delta_m. The fast tier
# gets a wider one because dividing the photons by 30 multiplies its MC
# noise by sqrt(30). See the C2 test file for the measurement.
DELTAM_TOL = {"fast": 0.4, "slow": 0.25}

# Second observable, and the sensitive one at the fast tier, where
# delta_m is dominated by the MC noise and a small systematic bias would
# hide inside its band. The spatial mean of a Stokes component over the
# 2500 sensors averages that noise down by a factor ~50, and being a
# linear functional it is an unbiased estimator: its expected value
# depends neither on the photon count nor on the noise realisation. One
# reference therefore serves both tiers.
#
# It is compared with an absolute tolerance of MEAN_TOL times the mean
# of I, for all four components: the means of Q, U and V are small
# compared with the one of I, so a relative band on them would be
# meaningless.
MEAN_TOL = 0.01

# A component whose mean absolute value falls below SIGNAL_FLOOR times
# the one of I carries no usable signal in these configurations: it is
# Monte Carlo noise, which is why the IPRT benchmark itself reports
# delta_m of 300 to 900% on V. Such a component is logged but not
# asserted, by any of the checks. See the C2 test file for the
# measurements behind this.
SIGNAL_FLOOR = 1e-3

# Number of scattering angles of the phase matrices. The benchmark uses
# 18001 but, unlike the C2 cubic cloud which has a single cloudy cell,
# the cumulus field mixes the 1D aerosol with each of its 20435 cloudy
# cells, so one phase matrix per cell is built: 18001 angles would need
# about 18 GB of host memory per atmosphere, and half of that on the
# GPU, against 1.8 GB here.
NTH = 1801

# The reference wavelength of the whole case, in nm: the cloud, the
# aerosol and the molecular optical properties are all given at 670 nm.
W_REF = 670.0

# CUDA block/grid: the optimal pair is GPU-dependent and could be
# measured at runtime, but the RNG is seeded per thread index over a
# XBLOCK*XGRID state buffer (smartg/smartg.py:3820), so changing the
# pair changes the noise realisation even at fixed SEED. The pair is
# therefore pinned, and the search is kept for benchmarking only.
FIND_OPTIMAL_XB_XG = False
XB = [32, 64, 128]  # candidate XBLOCK values
XG = [512, 1024]  # candidate XGRID values
CHECK_NBPHOTONS = 1e8  # short runs used only for timing
CHECK_NBLOOP = 1e7
XBLOCK = 128  # used when FIND_OPTIMAL_XB_XG is False
XGRID = 1024  # (values accepted by most GPUs after 10xx)

SCALE = 1  # can be useful for grid with very small cells
ROOTPATH = Path(__file__).resolve().parent.parent

# The IPRT C3 sensors cover the NCELLS x NCELLS cells of the cumulus
# field. Only the central NSENSORS x NSENSORS of them are used here,
# which divides the cost by 4. The MYSTIC reference is cropped the same
# way.
NCELLS = 100
NSENSORS = 50

# GT truncation, as in Iwabuchi and Suzuki (2009), with the parameters
# of the notebook notebooks/demo_notebook.ipynb: simple GT truncation
# without correction, i.e. scheme S of the paper. The photon count is
# left unchanged, so that the difference between the two tests below is
# the truncation bias alone and not a difference of MC noise.
#
# The truncation is applied to one phase matrix at a time by Atm3D.calc
# and, with aerosols, the C3 field holds one
# mixed matrix per cloudy cell, so pytrunc.gt_phase_approx is called
# 20489 times. Its cost per call, measured with pytrunc 1.1.0 at this
# NTH on a Ryzen 9 5950X, the loop being single threaded:
#
#     method     angle                 ms/call   total
#     lobatto    searched, th_tol=20      25.0    8.5 min
#     lobatto    forced 8 deg              2.6    0.9 min
#     trapezoid  searched, th_tol=20      30.2   10.3 min
#     trapezoid  forced 8 deg              2.6    0.9 min
#
# The Lobatto rows are the ones with lobatto_optimization, as in the C2
# test; without it the search costs 8.9 s per call. The angle is imposed
# here, and not only for the 8.5 min a build that it saves: measured on
# the slow tier, searching it degrades the delta_m of Q by 4.0 % and
# that of U by 8.9 %, for 2.2 % gained on the one of I. The fast tier
# reverses that verdict, but wrongly, its Q and U being noise dominated
# and a change of truncation drawing a different noise.
THETA_TR = 8.0
GT_TRUNC = GT_trunc(
    trunc_frac=0.435,
    theta_tol=20,  # unused, the angle below is imposed
    theta_tr=THETA_TR,
    integral_method="lobatto",
    lobatto_optimization=True,
)

# The single scattering albedo of the 1D aerosol at W_REF, as given by
# the IPRT C3 description.
SSA_AER_1D = 0.931184

# Reference delta_m values (in percent) of I, Q, U and V, measured with
# the settings above (SEED, XBLOCK, XGRID, NBPHOTONS, NSENSORS) for the
# backward case 4. Regenerate them with the same settings from the log
# if the physics legitimately changes.
#
# The benchmark of the report (table 6, RTX 4090 backward, case 4, with
# aerosols) gives 0.86 / 16.21 / 67.46 / 297.41 over the full 100x100
# sensor grid and with 10 times more photons. These values are noise
# dominated, and here every sensor gets 4e5 photons instead of 1e6, so
# they are expected to grow by about sqrt(2.5) = 1.58: 1.36 for I and
# 470 for V, against the 1.308 and 442.742 measured below. The
# remaining difference comes from the cropped sensor grid, which does
# not see the same part of the cumulus field.
DELTAM_REF_AER_B = {
    "slow": {4: (1.308, 36.348, 92.629, 442.742)},
    "fast": {
        4: (6.516, 155.448, 455.981, 2287.887),
    },
}

# Same, with the GT truncated phase matrices. The photon count being
# unchanged, the difference with the table above is the truncation
# alone, and it goes both ways: Q, U and V improve by a factor 2.5 to 4
# because they are noise dominated and the truncation is a variance
# reduction, while I degrades from 1.308 to 3.758. That degradation is
# a bias, not noise: the noise demonstrably went down, as the three
# other components show. It is the price of the uncorrected GT scheme S
# with the truncation angle imposed at THETA_TR.
DELTAM_REF_AER_B_GT = {
    "slow": {4: (3.758, 14.663, 28.646, 102.196)},
    "fast": {
        4: (3.978, 41.559, 109.220, 462.115),
    },
}

# Reference spatial means of I, Q, U and V, shared by the two tiers,
# see MEAN_TOL above. They are measured on the slow tier, which is the
# most precise estimate available, and the fast tier is required to
# reproduce them.
MEAN_REF_AER_B = {
    4: (1.073469e-01, -9.372497e-04, 3.520986e-05, 1.951276e-06),
}
MEAN_REF_AER_B_GT = {
    4: (1.081335e-01, -8.164488e-04, 3.969879e-05, -3.099173e-07),
}

# Mean absolute value of each Stokes component, measured on the fast
# tier. These are not checked: they are the signal scale against
# which SIGNAL_FLOOR above decides which components are worth
# asserting at all.
SIGNAL_REF_AER_B = {
    4: (1.073562e-01, 1.789515e-03, 1.212019e-03, 1.675291e-04),
}
SIGNAL_REF_AER_B_GT = {
    4: (1.081096e-01, 1.143670e-03, 5.018751e-04, 5.779069e-05),
}

# Viewing and sun geometry of the 9 IPRT C3 cases:
# (POSZ key, THETA, PHI, THETA_0). PHI_0 is 180. everywhere. Only the
# case 4 is tested, it is the fastest one, but the whole table is kept
# to document the benchmark and to make an extension straightforward.
CASES = {
    1: ("bottom", 40.0, 0.0, 20.0),
    2: ("bottom", 40.0, 60.0, 20.0),
    3: ("bottom", 40.0, 120.0, 20.0),
    4: ("bottom", 40.0, 180.0, 20.0),
    5: ("top", 180.0, 0.0, 40.0),
    6: ("top", 140.0, 0.0, 40.0),
    7: ("top", 140.0, 60.0, 40.0),
    8: ("top", 140.0, 120.0, 40.0),
    9: ("top", 140.0, 180.0, 40.0),
}
PHI_0 = 180.0
BACKWARD_CASES = (4,)
# **********************************************************************

# **************************** logging *********************************
# Create log file
log_dir = ROOTPATH / "tests" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Create a named logger
logger = logging.getLogger("test_phase_b_c3")
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

# Set the formatter for the console handler
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

# Create a file handler
file_handler = logging.FileHandler(
    ROOTPATH / "tests" / "logs" / "iprt_phase_b_c3.log", mode="w"
)
file_handler.setLevel(logging.INFO)

# Set the formatter for the file handler
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
# **********************************************************************


def _crop_cells():
    """
    Slice of the central NSENSORS cells of a NCELLS axis
    """
    i0 = (NCELLS - NSENSORS) // 2

    return slice(i0, i0 + NSENSORS)


def _crop_edges():
    """
    Slice of the NSENSORS+1 edges bounding the cells of _crop_cells
    """
    cells = _crop_cells()

    return slice(cells.start, cells.stop + 1)


def _k_from_cumulated_od(od_layers, zgrid_desc):
    """
    Convert the layer optical depths of the IPRT C3 profile into
    extinction (or scattering) coefficients

    The IPRT file gives one optical depth per layer, from the top of the
    atmosphere downwards, while SMART-G expects coefficients on the
    SMART-G altitude grid, hence the cumulated sum and the derivative.
    """
    dz = diff1(zgrid_desc)
    od_cumulated = np.concatenate(([0.0], np.cumsum(od_layers))).reshape(
        1, len(dz)
    )
    k = abs(diff1(od_cumulated, axis=1) / dz)
    k[np.isnan(k)] = 0

    return k


def _build_cloud_c3():
    """
    Read the IPRT C3 cumulus cloud field (100 x 100 x 53 cells) and its
    phase matrices

    Both atmospheres below share it: the phase matrices sampled over NTH
    angles are by far the most expensive part of their construction.

    Returns
    -------
    (cloud3, grid3)
    """
    dir_phase_b = DIR_AUXDATA / "IPRT" / "phaseB"

    cld_phase = read_phase_cdf(
        dir_phase_b / "opt_prop" / "watercloud_670.mie.cdf",
        n_theta=NTH,
        normalize=False,
        output_sg_ready=False,
    )
    cloud3 = Cloud3D(
        "wc",
        w_ref=W_REF,
        ds=read_i3rc_cloud(
            dir_phase_b / "grids" / "cumulus.dat", loc_xgrid=0, loc_ygrid=0
        ),
        phase=cld_phase,
        reff_acc=1,
        reff_min=5,
    )

    xgrid, ygrid, zgrid = cloud3.get_xyz_grid()
    grid3 = Grid3D(xgrid * SCALE, ygrid * SCALE, zgrid * SCALE, periodic=True)

    return cloud3, grid3


def _build_atm_c3(cloud_c3, truncation=None, with_aer=True):
    """
    Build the IPRT C3 cumulus cloud atmosphere

    The 3D cumulus field comes from cloud_c3, the Rayleigh scattering,
    the molecular absorption and, if with_aer, the aerosol extinction
    come from the 1D IPRT profile. truncation is the scattering phase
    truncation, applied to the phase matrices by Atm3D.calc.

    Returns
    -------
    (pro, grid3, surface, wavelengths)
    """
    cloud3, grid3 = cloud_c3
    dir_phase_b = DIR_AUXDATA / "IPRT" / "phaseB"

    # ========= 1D molecular and aerosol profiles
    file_mol = dir_phase_b / "opt_prop" / "atmos_tau_cu.dat"
    # Columns of the IPRT file: bottom, top, temperature, then the
    # molecular absorption at 0.67, 2.13 and 11.0 um, the Rayleigh
    # scattering at 0.67 um and the aerosol extinction at 0.67 and
    # 2.13 um. Only the 0.67 um ones are used here.
    od = pd.read_csv(
        file_mol,
        comment="!",
        header=None,
        usecols=[3, 6, 7],
        sep=r"\s+",
        dtype=float,
    ).values
    zgrid_desc = grid3.zGRID[::-1]
    mol_abs = _k_from_cumulated_od(od[:, 0], zgrid_desc)
    mol_sca = _k_from_cumulated_od(od[:, 1], zgrid_desc)

    comp = []
    atm3_kwargs = {}
    if with_aer:
        ext_aer = _k_from_cumulated_od(od[:, 2], zgrid_desc)

        # 1D aerosol phase matrix
        phase_waso = read_phase_cdf(
            dir_phase_b / "opt_prop" / "waso_670.mie.cdf",
            n_theta=NTH,
            normalize=False,
            output_sg_ready=False,
        )
        aer = AerOPAC(
            "continental_clean",
            0.5,
            w_ref=550.0,
            phase=phase_waso.isel(wavelength_phase=0, reff=0),
        )
        comp = [aer]
        atm3_kwargs = {
            "aer_ext_1d": ext_aer,
            "aer_ssa_1d": np.full_like(ext_aer, SSA_AER_1D),
        }

    # ========= profiles computations
    wavelengths = np.array([W_REF])
    atm3 = Atm3D(
        atm_1d=Atm1D("afglt", comp=comp),
        grid_3d=grid3,
        comp_3d=[cloud3],
        wavelength_phase=[W_REF],
        mol_sca_1d=mol_sca,
        mol_abs_1d=mol_abs,
        **atm3_kwargs,
    )
    pro = atm3.calc(wavelengths, n_theta=NTH, truncation=truncation)

    surface = LambSurface(alb=AlbedoCst(0.2))

    return pro, grid3, surface, wavelengths


@pytest.fixture(scope="module")
def s3db():
    """
    Backward compilation in 3D
    """
    return Smartg(
        opt3d=True, alt_pp=True, alis=False, back=True, double=True, bias=True
    )


@pytest.fixture(scope="module")
def cloud_c3():
    """
    IPRT C3 cumulus cloud field, shared by the two atmospheres
    """
    return _build_cloud_c3()


@pytest.fixture(scope="module")
def atm_c3_aer(cloud_c3):
    """
    IPRT C3 atmosphere with the 1D aerosol
    """
    return _build_atm_c3(cloud_c3)


@pytest.fixture(scope="module")
def atm_c3_aer_gt(cloud_c3):
    """
    Same as atm_c3_aer, with the GT truncated phase matrices
    """
    return _build_atm_c3(cloud_c3, truncation=GT_TRUNC)


@pytest.fixture(scope="module")
def sensor_grid(cloud_c3):
    """
    The central NSENSORS x NSENSORS sensors of the IPRT C3 grid

    The sensors are the cell centers of the grid given here, so the grid
    is the central part of the atmosphere grid. Its z axis is the one of
    the atmosphere: only its first and last values are used, to place
    the sensors.
    """
    _cloud3, grid3 = cloud_c3

    return Grid3D(
        grid3.xgrid[_crop_edges()],
        grid3.ygrid[_crop_edges()],
        grid3.zgrid,
        periodic=True,
    )


def _resolve_posz(sensor_grid, key):
    """
    Altitude where the sensors are placed
    """
    if key == "bottom":
        return sensor_grid.zGRID[0]
    elif key == "top":
        return sensor_grid.zGRID[-1] - 1e-6 * SCALE
    raise NameError(f"Unknown POSZ key '{key}'!")


def _find_optimal_xb_xg(sg, **run_kwargs):
    """
    Find the number of CUDA blocks and grids giving the shortest kernel
    time, using short runs with the same geometry as the real one.
    """
    if not FIND_OPTIMAL_XB_XG:
        return XBLOCK, XGRID

    k_time = np.inf
    best_xb = XBLOCK
    best_xg = XGRID
    for xg in XG:
        for xb in XB:
            m_test = sg.run(
                **run_kwargs,
                nb_photons=CHECK_NBPHOTONS,
                nb_loop=CHECK_NBLOOP,
                xblock=xb,
                xgrid=xg,
                progress=False,
            )
            time_s = float(m_test.attrs["kernel time (s)"])
            if time_s < k_time:
                k_time = time_s
                best_xb = xb
                best_xg = xg
            logger.info(f"time (s) = {time_s}; xblock = {xb}; xgrid = {xg}")
    logger.info(f"Best xblock = {best_xb}; best xgrid = {best_xg}")

    return best_xb, best_xg


def _run_case_backward(s3db, atm_c3, sensor_grid, case, nbphotons=NBPHOTONS):
    """
    Run one backward IPRT C3 case

    Returns
    -------
    (m, norm)
    """
    pro, grid3, surface, wavelengths = atm_c3
    posz_key, theta, phi, theta_0 = CASES[case]
    posz = _resolve_posz(sensor_grid, posz_key)

    # !!!! grid3 is different than the sensors grid !!!
    sensors = get_sensors_grid(
        sensor_grid.xgrid,
        sensor_grid.ygrid,
        pos_z=posz,
        th_deg=theta,
        ph_deg=phi,
        fov=0.0,
        loc="ATMOS",
        cell_size=sensor_grid.xgrid[1] - sensor_grid.xgrid[0],
        grid_3d=grid3,
    )

    # count_level = 0 -> only COUNT TOA
    le = {
        "th_deg": np.array([theta_0]),
        "phi_deg": np.array([PHI_0]),
        "count_level": np.array([0]),
    }

    kw = dict(
        wavelength=wavelengths,
        atmosphere=pro,
        sensor=sensors,
        le=le,
        surface=surface,
        n_icdf=NTH,
        stdev=True,
    )
    xb, xg = _find_optimal_xb_xg(s3db, **kw)

    m = s3db.run(
        **kw,
        nb_photons=nbphotons,
        nb_loop=NBLOOP,
        xblock=xb,
        xgrid=xg,
        seed=SEED,
    )

    return m, np.cos(np.radians(theta_0)) / np.pi


def _smartg_iquv(m, norm, U_sign=1, V_sign=-1):
    """
    Extract the normalized I, Q, U and V (NSENSORS, NSENSORS) matrices
    """
    shape = (NSENSORS, NSENSORS)

    return (
        m["I_up (TOA)"].values[:, 0, 0].reshape(shape) * norm,
        m["Q_up (TOA)"].values[:, 0, 0].reshape(shape) * norm,
        m["U_up (TOA)"].values[:, 0, 0].reshape(shape) * norm * U_sign,
        m["V_up (TOA)"].values[:, 0, 0].reshape(shape) * norm * V_sign,
    )


def _mystic_iquv(tcase):
    """
    Read the MYSTIC I, Q, U and V matrices of a given case, cropped to
    the central NSENSORS x NSENSORS sensors

    tcase is the MYSTIC case number: it is the C3 case number without
    aerosol, and the C3 case number + 9 with aerosols.
    """
    file_res = (
        DIR_AUXDATA / "IPRT" / "phaseB" / "mystic_res" / "iprt_case_C3_mystic.dat"
    )
    nrows = NCELLS * NCELLS
    read_res = pd.read_csv(
        file_res,
        skiprows=(nrows * (tcase - 1)) + 3,
        nrows=nrows,
        header=None,
        sep=r"\s+",
        dtype=float,
    ).values

    crop = _crop_cells()

    return tuple(
        read_res[:, istk].reshape(NCELLS, NCELLS).T[crop, crop]
        for istk in (7, 8, 9, 10)
    )


def _plot_case(request, m, iquv_sg, iquv_my, case, sensor_grid, title_suffix):
    """
    Save the SMART-G maps and the SMART-G - MYSTIC differences in the
    pytest html report
    """
    i_sg, q_sg, u_sg, v_sg = iquv_sg
    i_my, q_my, u_my, v_my = iquv_my

    stk = ["I", "Q", "U", "V"]
    wavelength = m.coords["wavelength"].values
    xgrid = sensor_grid.xgrid
    ygrid = sensor_grid.ygrid
    max_i = np.max(i_sg)
    max_q = np.max(np.abs(q_sg))
    max_u = np.max(np.abs(u_sg))
    max_v = np.max(np.abs(v_sg))

    satellite_view(
        m,
        xgrid,
        ygrid,
        wavelength,
        "none",
        ["jet", "coolwarm", "coolwarm", "coolwarm"],
        figsize=(10.5, 7),
        fontsize=16,
        vmin=[0.0, -max_q, -max_u, -max_v],
        vmax=[max_i, max_q, max_u, max_v],
        scale=False,
        stokes=stk,
        matrices=[i_sg, q_sg, u_sg, v_sg],
        cbar_shrink=1,
        cbar_sci_format=True,
        title=f"C3 - case {case} - SMART-G - {title_suffix}",
    )
    conftest.savefig(request, bbox_inches="tight")

    lim = [max_i * 0.05, max_q * 0.05, max_u * 0.05, max_v * 0.05]
    satellite_view(
        m,
        xgrid,
        ygrid,
        wavelength,
        "none",
        ["coolwarm", "coolwarm", "coolwarm", "coolwarm"],
        figsize=(10.5, 7),
        fontsize=16,
        vmin=[-val for val in lim],
        vmax=lim,
        scale=False,
        stokes=stk,
        matrices=[i_sg - i_my, q_sg - q_my, u_sg - u_my, v_sg - v_my],
        cbar_shrink=1,
        cbar_sci_format=True,
        title=f"C3 - case {case} - dif(SMART-G - MYSTIC) - {title_suffix}",
    )
    conftest.savefig(request, bbox_inches="tight")


def _is_significant(signal_ref, istk):
    """
    Whether a Stokes component carries enough signal to be asserted on

    See SIGNAL_FLOOR. A signal_ref of None, i.e. not yet measured, keeps
    every component so that a new reference gets fully logged.
    """
    if signal_ref is None:
        return True

    return signal_ref[istk] > SIGNAL_FLOOR * signal_ref[0]


def _skipped(signal_ref):
    """
    Names of the components left unasserted, for the log
    """
    return [
        stk
        for istk, stk in enumerate(["I", "Q", "U", "V"])
        if not _is_significant(signal_ref, istk)
    ]


def _check_deltam(delta_m_ref, signal_ref, iquv_my, iquv_sg, label, tol):
    """
    Compute the delta_m values and compare them with the previous saved
    validated ones

    Returns the list of the failure messages (empty if the case is ok).

    delta_m_ref can be None: the calculated values are then logged and
    the case is reported as a failure, which is how a new reference is
    measured before being written in the tables above.
    """
    iquv_mystic = group_iquv(
        i_list=[iquv_my[0]],
        q_list=[iquv_my[1]],
        u_list=[iquv_my[2]],
        v_list=[iquv_my[3]],
    )
    iquv_smartg = group_iquv(
        i_list=[iquv_sg[0]],
        q_list=[iquv_sg[1]],
        u_list=[iquv_sg[2]],
        v_list=[iquv_sg[3]],
    )

    delta_m = compute_deltam(obs=iquv_mystic, mod=iquv_smartg, print_res=False)

    if delta_m_ref is not None:
        logger.info(
            f"{label} - I={delta_m_ref[0]:.3f}; Q={delta_m_ref[1]:.3f}; "
            + f"U={delta_m_ref[2]:.3f}; V={delta_m_ref[3]:.3f} - ref delta_m:"
        )
    logger.info(
        f"{label} - I={delta_m[0]:.3f}; Q={delta_m[1]:.3f}; "
        + f"U={delta_m[2]:.3f}; V={delta_m[3]:.3f} - calculated delta_m"
    )

    if delta_m_ref is None:
        return [f"{label}: no reference delta_m, see the log for the values"]

    # Check if the test is ok by comparing the ref delta_m and the
    # calculated one
    skipped = _skipped(signal_ref)
    if skipped:
        logger.info(
            f"{label} - {', '.join(skipped)} below SIGNAL_FLOOR, "
            + "not asserted"
        )

    errors = []
    iquv_name = ["I", "Q", "U", "V"]
    for istk, stk in enumerate(iquv_name):
        if not _is_significant(signal_ref, istk):
            continue
        ref = delta_m_ref[istk]
        if abs(delta_m[istk] - ref) > tol * ref:
            errors.append(
                f"{label}: problem with {stk} values, get "
                + f"{delta_m[istk]:.5f}. {stk} must be within "
                + f"[{(1-tol)*ref:.5f}, {(1+tol)*ref:.5f}]"
            )

    return errors


def _check_means(mean_ref, signal_ref, iquv_sg, label):
    """
    Compare the spatial mean of each Stokes component with its previous
    saved validated one

    Unlike delta_m, this averages the Monte Carlo noise out, so it is
    the observable that keeps the fast tier sensitive to a systematic
    bias. Same contract as _check_deltam: returns the list of the
    failure messages, and a mean_ref of None logs the calculated values
    and reports a failure, which is how a new reference is measured.
    """
    means = tuple(float(np.mean(stk)) for stk in iquv_sg)

    if mean_ref is not None:
        logger.info(
            f"{label} - I={mean_ref[0]:.6e}; Q={mean_ref[1]:.6e}; "
            + f"U={mean_ref[2]:.6e}; V={mean_ref[3]:.6e} - ref mean:"
        )
    logger.info(
        f"{label} - I={means[0]:.6e}; Q={means[1]:.6e}; "
        + f"U={means[2]:.6e}; V={means[3]:.6e} - calculated mean"
    )

    if mean_ref is None:
        return [f"{label}: no reference mean, see the log for the values"]

    # The mean of I sets the scale of the four tolerances, the means of
    # Q, U and V being much smaller than it
    tol = MEAN_TOL * abs(mean_ref[0])

    errors = []
    for istk, stk in enumerate(["I", "Q", "U", "V"]):
        if not _is_significant(signal_ref, istk):
            continue
        ref = mean_ref[istk]
        if abs(means[istk] - ref) > tol:
            errors.append(
                f"{label}: problem with the mean of {stk}, get "
                + f"{means[istk]:.6e}. It must be within "
                + f"[{ref-tol:.6e}, {ref+tol:.6e}]"
            )

    return errors


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize(
    "case", BACKWARD_CASES, ids=[f"case{i}" for i in BACKWARD_CASES]
)
def test_c3_aer_backward(request, s3db, atm_c3_aer, sensor_grid, case, tier):
    """
    IPRT phase B, cumulus cloud C3, backward, with aerosols
    """
    print(f"=== Test C3 case {case} - backward - with aerosols - {tier}")

    m, norm = _run_case_backward(
        s3db,
        atm_c3_aer,
        sensor_grid,
        case,
        nbphotons=NBPHOTONS / PHOTON_DIVIDER[tier],
    )
    iquv_sg = _smartg_iquv(m, norm)
    iquv_my = _mystic_iquv(case + 9)

    _plot_case(
        request,
        m,
        iquv_sg,
        iquv_my,
        case,
        sensor_grid,
        title_suffix="with aer",
    )

    label = f"C3 - case {case} - aer - {tier}"
    signal_ref = SIGNAL_REF_AER_B.get(case)
    errors = _check_deltam(
        DELTAM_REF_AER_B[tier].get(case),
        signal_ref,
        iquv_my,
        iquv_sg,
        label,
        DELTAM_TOL[tier],
    )
    errors += _check_means(
        MEAN_REF_AER_B.get(case), signal_ref, iquv_sg, label
    )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize(
    "case", BACKWARD_CASES, ids=[f"case{i}" for i in BACKWARD_CASES]
)
def test_c3_aer_backward_gt(
    request, s3db, atm_c3_aer_gt, sensor_grid, case, tier
):
    """
    IPRT phase B, cumulus cloud C3, backward, with aerosols, with the GT
    truncated phase matrices
    """
    print(
        f"=== Test C3 case {case} - backward - with aerosols - GT trunc"
        + f" - {tier}"
    )

    m, norm = _run_case_backward(
        s3db,
        atm_c3_aer_gt,
        sensor_grid,
        case,
        nbphotons=NBPHOTONS / PHOTON_DIVIDER[tier],
    )
    iquv_sg = _smartg_iquv(m, norm)
    iquv_my = _mystic_iquv(case + 9)

    _plot_case(
        request,
        m,
        iquv_sg,
        iquv_my,
        case,
        sensor_grid,
        title_suffix="with aer - GT trunc",
    )

    label = f"C3 - case {case} - aer GT - {tier}"
    signal_ref = SIGNAL_REF_AER_B_GT.get(case)
    errors = _check_deltam(
        DELTAM_REF_AER_B_GT[tier].get(case),
        signal_ref,
        iquv_my,
        iquv_sg,
        label,
        DELTAM_TOL[tier],
    )
    errors += _check_means(
        MEAN_REF_AER_B_GT.get(case), signal_ref, iquv_sg, label
    )
    assert not errors, "\n".join(errors)
