#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Non-regression test of the 3D atmosphere mode (opt3D=True) using the
# IPRT phase B cubic cloud case (C2).
# Tested with the following GPUs: 3090
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartg import conftest
from smartg.atmosphere import Atm1D
from smartg.config import DIR_AUXDATA
from smartg.iprt.iprt import compute_deltam, groupIQUV
from smartg.libATM3D import (
    Atm3D,
    Cloud3D,
    Grid3D,
    create_sensors,
    read_cld_nth_cte,
    satellite_view,
)
from smartg.smartg import AlbedoCst, LambSurface, Smartg

# *********************** Global variable(s) ***************************
# Fixed seed: SEED=-1 would derive it from the clock, giving a new
# noise realisation at every run. The reference delta_m values below
# were measured with this seed.
SEED = 1234
DELTAM_TOL = 0.25  # two-sided fractional band around the ref delta_m
NBPHOTONS = 49e9  # notebook values: required for the reference
NBLOOP = 1e8  # delta_m values below to be reproducible
NTH = 18001  # 1801 is not enough for case 6

# CUDA block/grid: the optimal pair is GPU-dependent and could be
# measured at runtime, but the RNG is seeded per thread index over a
# XBLOCK*XGRID state buffer (smartg/smartg.py:3820), so changing the
# pair changes the noise realisation even at fixed SEED. The pair is
# therefore pinned, and the search is kept for benchmarking only.
FIND_OPTIMAL_XB_XG = False
XB = [32, 64, 128]  # candidate XBLOCK values
XG = [512, 1024]  # candidate XGRID values
CHECK_NBPHOTONS = 1e8  # short runs used only for timing
CHECK_NBLOOP = 1e8
XBLOCK = 128  # used when FIND_OPTIMAL_XB_XG is False
XGRID = 1024  # (values accepted by most GPUs after 10xx)

SCALE = 1  # can be useful for grid with very small cells
ROOTPATH = Path(__file__).resolve().parent.parent

# Reference delta_m values (in percent) of I, Q, U and V, measured with
# the settings above (SEED, XBLOCK, XGRID, NBPHOTONS). They are within
# a few percent of the values of the notebook
# notebooks/validation_SMARTG_IPRT_phaseB-C2.ipynb, whose outputs are
# stripped in the repository. Regenerate them with the same settings
# from the log if the physics legitimately changes.
DELTAM_REF_NOATM_B = {
    1: (0.507, 2.126, 65.313, 404.483),
    2: (0.481, 2.267, 2.644, 304.021),
    3: (0.458, 3.689, 2.070, 248.201),
    4: (0.393, 4.605, 32.565, 315.668),
    5: (0.142, 1.081, 24.303, 262.352),
    6: (0.255, 30.129, 89.739, 981.808),
    7: (0.177, 1.507, 1.391, 70.308),
    8: (0.157, 17.658, 6.626, 365.500),
    9: (0.176, 11.334, 69.407, 424.826),
}

# Viewing and sun geometry of the 9 IPRT C2 cases:
# (POSZ key, THETA, PHI, THETA_0). PHI_0 is 180. everywhere.
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
# **********************************************************************

# **************************** logging *********************************
# Create log file
log_dir = ROOTPATH / "tests" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Create a named logger
logger = logging.getLogger("test_phase_b_c2")
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
    ROOTPATH / "tests" / "logs" / "iprt_phase_b_c2.log", mode="w"
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


def _build_atm_c2(**atm3_kwargs):
    """
    Build the IPRT C2 cubic cloud atmosphere.

    Only the molecular arguments of Atm3D differ between the with and
    without atmosphere sections, hence the **atm3_kwargs.

    Returns
    -------
    (pro, grid3, surf, wls)
    """
    # ========= phase matrix
    file_cld_phase = (
        DIR_AUXDATA / "IPRT" / "phaseB" / "opt_prop" / "watercloud_800.mie.cdf"
    )
    cld_phase = read_cld_nth_cte(filename=file_cld_phase, nb_theta=NTH)

    # ========= grid (reduced grid = faster)
    xgrid = np.array([0.0, 3.0, 4.0, 7.0]) * SCALE
    ygrid = np.array([0.0, 3.0, 4.0, 7.0]) * SCALE
    zgrid = np.array([0.0, 2.0, 3.0, 5.0]) * SCALE
    grid3 = Grid3D(xgrid, ygrid, zgrid, periodic=True)

    # ========= cloud
    # First column x, second y and third z. We follow the IPRT
    # convention for indices (start at 1 instead of 0): the cubic cloud
    # is between 3 and 4 km in x and y, and between 2 and 3 km in z.
    cloud_indices = np.zeros((1, 3), dtype=np.int32)
    cloud_indices[0, :] = np.array([2, 2, 2])
    cld_ext_coeff = np.zeros(1, dtype=np.float64)
    cld_ext_coeff[0] = 10.0
    reff = np.zeros_like(cld_ext_coeff, dtype=np.float64)
    reff[0] = 10.0
    cloud3 = Cloud3D(
        "wc",
        w_ref=800.0,
        ext_ref=cld_ext_coeff,
        xyz_grids=[grid3.xGRID, grid3.yGRID, grid3.zGRID],
        cell_indices=cloud_indices,
        reff=reff,
        phase=cld_phase,
    )

    atm3 = Atm3D(
        "afglt",
        grid3,
        wls=np.array([800.0]),
        wl_ref=800.0,
        cloud_3d=cloud3,
        **atm3_kwargs,
    )

    grid = atm3.get_grid()
    prof_ray = atm3.get_glob_molecular_sca()  # Rayleigh
    prof_abs = atm3.get_glob_molecular_abs()

    ext_aer = atm3.get_glob_aer_ext() * (1 / SCALE)
    # ssa forced to 1, must have the same form as ext_cld3D
    ssa_aer = np.ones_like(ext_aer)
    prof_aer = (ext_aer, ssa_aer)

    prof_phases = atm3.get_glob_aer_phase(wl_phase=[800.0], n_theta=NTH)

    cells = atm3.get_cells_info()

    # ========= profiles computations
    atm3d = Atm1D(
        "ATM3D",
        grid=grid,
        prof_ray=prof_ray,
        prof_abs=prof_abs,
        prof_aer=prof_aer,
        prof_phases=prof_phases,
        cells=cells,
    )
    pro = atm3d.calc(atm3.wls, n_theta=NTH)

    surf = LambSurface(ALB=AlbedoCst(0.2))

    return pro, grid3, surf, atm3.wls


@pytest.fixture(scope="module")
def s3db():
    """
    Backward compilation in 3D
    """
    return Smartg(
        opt3D=True, alt_pp=True, alis=False, back=True, double=True, bias=True
    )


@pytest.fixture(scope="module")
def atm_c2_noatm():
    """
    IPRT C2 atmosphere without the molecular contribution
    """
    return _build_atm_c2(tauR=0.0, NO2=False, O3=0.0, H2O=0.0)


@pytest.fixture(scope="module")
def sensor_grid():
    """
    The 70x70 sensor grid, identical for the 9 cases
    """
    return Grid3D(
        np.linspace(0.0, 7.0, 71) * SCALE,
        np.linspace(0.0, 7.0, 71) * SCALE,
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) * SCALE,
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
                NBPHOTONS=CHECK_NBPHOTONS,
                NBLOOP=CHECK_NBLOOP,
                XBLOCK=xb,
                XGRID=xg,
                progress=False,
            )
            time_s = float(m_test.attrs["kernel time (s)"])
            if time_s < k_time:
                k_time = time_s
                best_xb = xb
                best_xg = xg
            logger.info(
                f"time (s) = {time_s}; xblock = {xb}; xgrid = {xg}"
            )
    logger.info(f"Best xblock = {best_xb}; best xgrid = {best_xg}")

    return best_xb, best_xg


def _run_case_backward(s3db, atm_c2, sensor_grid, case):
    """
    Run one backward IPRT C2 case

    Returns
    -------
    (m, norm)
    """
    pro, grid3, surf, wls = atm_c2
    posz_key, theta, phi, theta_0 = CASES[case]
    posz = _resolve_posz(sensor_grid, posz_key)

    # !!!! grid3 is different than the sensors grid !!!
    _x0, _y0, sensors, _icells = create_sensors(
        sensor_grid,
        POSZ=posz,
        THDEG=theta,
        PHDEG=phi,
        FOV=0.0,
        LOC="ATMOS",
        CELL_SIZE=sensor_grid.xgrid[1] - sensor_grid.xgrid[0],
        grid3D_atm=grid3,
    )

    # count_level = 0 -> only COUNT TOA
    le = {
        "th_deg": np.array([theta_0]),
        "phi_deg": np.array([PHI_0]),
        "count_level": np.array([0]),
    }

    kw = dict(
        wl=wls,
        atm=pro,
        sensor=sensors,
        le=le,
        surf=surf,
        NF=NTH,
        stdev=True,
    )
    xb, xg = _find_optimal_xb_xg(s3db, **kw)

    m = s3db.run(
        **kw,
        NBPHOTONS=NBPHOTONS,
        NBLOOP=NBLOOP,
        XBLOCK=xb,
        XGRID=xg,
        SEED=SEED,
    )

    return m, np.cos(np.radians(theta_0)) / np.pi


def _smartg_iquv(
    m, norm, U_sign=1, V_sign=-1, mI=None, mQ=None, mU=None, mV=None
):
    """
    Extract the normalized I, Q, U and V (70, 70) matrices

    The mI, mQ, mU and mV arguments allow to force the values, for
    example when a single forward run holds several cases.
    """
    if mI is None:
        mI = m["I_up (TOA)"][:, 0, 0]
    if mQ is None:
        mQ = m["Q_up (TOA)"][:, 0, 0]
    if mU is None:
        mU = m["U_up (TOA)"][:, 0, 0]
    if mV is None:
        mV = m["V_up (TOA)"][:, 0, 0]

    return (
        mI.reshape(70, 70) * norm,
        mQ.reshape(70, 70) * norm,
        mU.reshape(70, 70) * norm * U_sign,
        mV.reshape(70, 70) * norm * V_sign,
    )


def _mystic_iquv(tcase):
    """
    Read the MYSTIC I, Q, U and V (70, 70) matrices of a given case

    tcase is the MYSTIC case number: it is the C2 case number without
    atmosphere, and the C2 case number + 9 with atmosphere.
    """
    file_res = (
        DIR_AUXDATA / "IPRT" / "phaseB" / "mystic_res" / "iprt_case_C2_mystic.dat"
    )
    read_res = pd.read_csv(
        file_res,
        skiprows=(4900 * (tcase - 1)) + 3,
        nrows=4900,
        header=None,
        sep=r"\s+",
        dtype=float,
    ).values

    return (
        read_res[:, 7].reshape(70, 70).T,
        read_res[:, 8].reshape(70, 70).T,
        read_res[:, 9].reshape(70, 70).T,
        read_res[:, 10].reshape(70, 70).T,
    )


def _plot_case(
    request,
    m,
    iquv_sg,
    iquv_my,
    case,
    sensor_grid,
    title_suffix,
    i_vmin,
    v_diff_frac,
):
    """
    Save the SMART-G maps and the SMART-G - MYSTIC differences in the
    pytest html report
    """
    i_sg, q_sg, u_sg, v_sg = iquv_sg
    i_my, q_my, u_my, v_my = iquv_my

    stk = ["I", "Q", "U", "V"]
    wl = m.axes["wavelength"]
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
        wl,
        "none",
        ["jet", "coolwarm", "coolwarm", "coolwarm"],
        fig_size=(10.5, 7),
        font_size=int(16),
        vmin=[i_vmin, -max_q, -max_u, -max_v],
        vmax=[max_i, max_q, max_u, max_v],
        scale=False,
        save_file=None,
        stk=stk,
        factor=None,
        mat_force=[i_sg, q_sg, u_sg, v_sg],
        cb_shrink=1,
        cb_sform=True,
        fig_title=f"C2 - case {case} - SMART-G - {title_suffix}",
    )
    conftest.savefig(request, bbox_inches="tight")

    lim = [
        max_i * 0.05,
        max_q * 0.05,
        max_u * 0.05,
        max_v * v_diff_frac,
    ]
    satellite_view(
        m,
        xgrid,
        ygrid,
        wl,
        "none",
        ["coolwarm", "coolwarm", "coolwarm", "coolwarm"],
        fig_size=(10.5, 7),
        font_size=int(16),
        vmin=[-val for val in lim],
        vmax=lim,
        scale=False,
        save_file=None,
        stk=stk,
        factor=None,
        mat_force=[i_sg - i_my, q_sg - q_my, u_sg - u_my, v_sg - v_my],
        cb_shrink=1,
        cb_sform=True,
        fig_title=(
            f"C2 - case {case} - dif(SMART-G - MYSTIC) - {title_suffix}"
        ),
    )
    conftest.savefig(request, bbox_inches="tight")


def _check_deltam(delta_m_ref, iquv_my, iquv_sg, label):
    """
    Compute the delta_m values and compare them with the previous saved
    validated ones
    """
    iquv_mystic = groupIQUV(
        lI=[iquv_my[0]], lQ=[iquv_my[1]], lU=[iquv_my[2]], lV=[iquv_my[3]]
    )
    iquv_smartg = groupIQUV(
        lI=[iquv_sg[0]], lQ=[iquv_sg[1]], lU=[iquv_sg[2]], lV=[iquv_sg[3]]
    )

    delta_m = compute_deltam(
        obs=iquv_mystic, mod=iquv_smartg, print_res=False
    )

    logger.info(
        f"{label} - I={delta_m_ref[0]:.3f}; Q={delta_m_ref[1]:.3f}; "
        + f"U={delta_m_ref[2]:.3f}; V={delta_m_ref[3]:.3f} - ref delta_m:"
    )
    logger.info(
        f"{label} - I={delta_m[0]:.3f}; Q={delta_m[1]:.3f}; "
        + f"U={delta_m[2]:.3f}; V={delta_m[3]:.3f} - calculated delta_m"
    )

    # Check if the test is ok by comparing the ref delta_m and the
    # calculated one
    iquv_name = ["I", "Q", "U", "V"]
    for istk, stk in enumerate(iquv_name):
        ref = delta_m_ref[istk]
        assert abs(delta_m[istk] - ref) <= DELTAM_TOL * ref, (
            f"Problem with {stk} values, get {delta_m[istk]:.5f}."
            + f" {stk} must be within [{(1-DELTAM_TOL)*ref:.5f}, "
            + f"{(1+DELTAM_TOL)*ref:.5f}]"
        )


@pytest.mark.parametrize(
    "case", list(CASES), ids=[f"case{i}" for i in CASES]
)
def test_c2_noatm_backward(request, s3db, atm_c2_noatm, sensor_grid, case):
    """
    IPRT phase B, cubic cloud C2, backward, without atmosphere
    """
    print(f"=== Test C2 case {case} - backward - without atmosphere")

    m, norm = _run_case_backward(s3db, atm_c2_noatm, sensor_grid, case)
    iquv_sg = _smartg_iquv(m, norm)
    iquv_my = _mystic_iquv(case)

    _plot_case(
        request,
        m,
        iquv_sg,
        iquv_my,
        case,
        sensor_grid,
        title_suffix="without atm",
        i_vmin=np.min(np.abs(iquv_sg[0])),
        v_diff_frac=0.015,
    )

    _check_deltam(
        DELTAM_REF_NOATM_B[case], iquv_my, iquv_sg, f"C2 - case {case}"
    )
