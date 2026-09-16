"""Non-regression tests of the 3D mode on the IPRT C3 cumulus cloud.

The IPRT phase B cumulus cloud case (C3) is run with opt3d=True, in
backward mode, with aerosols, with and without the GT truncation, and
compared with MYSTIC. Unlike the C2 cubic cloud, this case mixes a
realistic 100x100x53 cloud field with a 1D Rayleigh and aerosol
profile, which is the configuration of the 3MI scenes. The
atmospheres, the sensors, the runs and the plots come from
smartg.iprt.phase_b.

Tested with the following GPUs: 5070 Ti
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from smartg import conftest
from smartg.atmosphere import Cloud3D
from smartg.grid3d import Grid3D
from smartg.iprt.common import compute_deltam
from smartg.iprt.phase_b import (
    ATM_CASE_OFFSET,
    MYSTIC_RES_C3,
    PhaseBAtmosphere,
    backward_run_kwargs,
    build_atm_c3,
    build_cloud_c3,
    central_slice,
    find_optimal_xb_xg,
    plot_camera_difference,
    plot_camera_iquv,
    read_iprt_iquv,
    run_case_backward,
    sensor_grid_c3,
    smartg_iquv,
)
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
# N_SENSORS below: the number of photons per sensor stays higher than in
# the benchmark.
N_PHOTONS = 1e9
N_LOOP = 1e7

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
N_THETA = 1801

# CUDA block/grid: the optimal pair is GPU-dependent and could be
# measured at runtime, but the RNG is seeded per thread index over a
# XBLOCK*XGRID state buffer (smartg/smartg.py:3820), so changing the
# pair changes the noise realisation even at fixed SEED. The pair is
# therefore pinned, and the search is kept for benchmarking only.
FIND_OPTIMAL_XB_XG = False
X_BLOCKS = [32, 64, 128]  # candidate XBLOCK values
X_GRIDS = [512, 1024]  # candidate XGRID values
CHECK_N_PHOTONS = 1e8  # short runs used only for timing
CHECK_N_LOOP = 1e7
XBLOCK = 128  # used when FIND_OPTIMAL_XB_XG is False
XGRID = 1024  # (values accepted by most GPUs after 10xx)

SCALE = 1  # can be useful for grid with very small cells
ROOT_PATH = Path(__file__).resolve().parent.parent

# The IPRT C3 sensors cover the N_CELLS x N_CELLS cells of the cumulus
# field. Only the central N_SENSORS x N_SENSORS of them are used here,
# which divides the cost by 4. The MYSTIC reference is cropped the same
# way.
N_CELLS = 100
N_SENSORS = 50

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
# N_THETA on a Ryzen 9 5950X, the loop being single threaded:
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

# Reference delta_m values (in percent) of I, Q, U and V, measured with
# the settings above (SEED, XBLOCK, XGRID, N_PHOTONS, N_SENSORS) for the
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

# Only the case 4 of the 9 phase_b.CASES is tested, it is the fastest
# one.
BACKWARD_CASES = (4,)
# **********************************************************************

# **************************** logging *********************************
LOG_DIR = ROOT_PATH / "tests" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "iprt_phase_b_c3.log"
LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)

# Errors on the console, everything in the log file
logger = logging.getLogger("test_phase_b_c3")
logger.setLevel(logging.INFO)
for handler, level in (
    (logging.StreamHandler(), logging.ERROR),
    (logging.FileHandler(LOG_FILE, mode="w"), logging.INFO),
):
    handler.setLevel(level)
    handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(handler)
# **********************************************************************


@pytest.fixture(scope="module")
def s3db() -> Smartg:
    """Backward compilation in 3D."""
    return Smartg(
        opt3d=True, alt_pp=True, alis=False, back=True, double=True, bias=True
    )


@pytest.fixture(scope="module")
def cloud_c3() -> tuple[Cloud3D, Grid3D]:
    """The IPRT C3 cumulus field, shared by the two atmospheres."""
    return build_cloud_c3(n_theta=N_THETA, scale=SCALE)


@pytest.fixture(scope="module")
def atm_c3_aer(cloud_c3: tuple[Cloud3D, Grid3D]) -> PhaseBAtmosphere:
    """IPRT C3 atmosphere with aerosols."""
    return build_atm_c3(cloud_c3, n_theta=N_THETA)


@pytest.fixture(scope="module")
def atm_c3_aer_gt(cloud_c3: tuple[Cloud3D, Grid3D]) -> PhaseBAtmosphere:
    """Same as atm_c3_aer, with the GT truncated phase matrices."""
    return build_atm_c3(cloud_c3, truncation=GT_TRUNC, n_theta=N_THETA)


@pytest.fixture(scope="module")
def sensor_grid(cloud_c3: tuple[Cloud3D, Grid3D]) -> Grid3D:
    """The central N_SENSORS x N_SENSORS sensors of the C3 field."""
    return sensor_grid_c3(cloud_c3[1], N_SENSORS)


def _xblock_xgrid(sg: Smartg, **run_kwargs: Any) -> tuple[int, int]:
    """Return the CUDA block and grid sizes of a run.

    They are XBLOCK and XGRID, unless FIND_OPTIMAL_XB_XG asks for the
    fastest pair, measured with short runs of the same geometry.

    Parameters
    ----------
    sg : Smartg
        The compiled SMART-G.
    **run_kwargs
        The Smartg.run arguments of the run, without n_photons, n_loop,
        xblock and xgrid.

    Returns
    -------
    xblock, xgrid : int
        The CUDA block and grid sizes.
    """
    if not FIND_OPTIMAL_XB_XG:
        return XBLOCK, XGRID
    return find_optimal_xb_xg(
        sg, X_BLOCKS, X_GRIDS, CHECK_N_PHOTONS, CHECK_N_LOOP, **run_kwargs
    )


def _run_case_backward(
    s3db: Smartg,
    atm: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    case: int,
    n_photons: float = N_PHOTONS,
) -> tuple[xr.Dataset, float]:
    """Run one backward C3 case with the pinned settings.

    Parameters
    ----------
    s3db : Smartg
        The backward compilation.
    atm : PhaseBAtmosphere
        The atmosphere.
    sensor_grid : Grid3D
        The sensor grid.
    case : int
        The case number.
    n_photons : float
        Number of photons.

    Returns
    -------
    ds : xr.Dataset
        The output of the run.
    norm : float
        The normalisation of the maps.
    """
    options: dict[str, Any] = {"n_icdf": N_THETA}
    xblock, xgrid = _xblock_xgrid(
        s3db, **backward_run_kwargs(atm, sensor_grid, case), **options
    )
    return run_case_backward(
        s3db, atm, sensor_grid, case, n_photons, n_loop=N_LOOP,
        xblock=xblock, xgrid=xgrid, seed=SEED, **options,
    )


def _read_mystic(case: int) -> tuple[np.ndarray, ...]:
    """Read the MYSTIC maps of a case, cropped to the sensor grid.

    Parameters
    ----------
    case : int
        The MYSTIC case number: the C3 case number without aerosols,
        plus ATM_CASE_OFFSET with aerosols.

    Returns
    -------
    tuple of ndarray
        The I, Q, U and V maps of the central N_SENSORS x N_SENSORS
        sensors.
    """
    crop = central_slice(N_CELLS, N_SENSORS)
    return tuple(
        stk[crop, crop]
        for stk in read_iprt_iquv(MYSTIC_RES_C3, case, N_CELLS)
    )


def _plot_case(
    request: pytest.FixtureRequest,
    iquv_sg: tuple[np.ndarray, ...],
    iquv_my: tuple[np.ndarray, ...],
    case: int,
    sensor_grid: Grid3D,
    title_suffix: str,
    i_vmin: float | None = 0.0,
    v_diff_frac: float = 0.05,
) -> None:
    """Save the SMART-G maps and their differences with MYSTIC.

    The two figures go to the pytest html report.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request of the test.
    iquv_sg, iquv_my : tuple of ndarray
        The SMART-G and the MYSTIC I, Q, U and V maps.
    case : int
        The case number, for the titles.
    sensor_grid : Grid3D
        The sensor grid.
    title_suffix : str
        The end of the titles.
    i_vmin : float, optional
        The lower bound of the I colour scale. By default the minimum
        of abs(I).
    v_diff_frac : float
        The bound of the V difference colour scale, as a fraction of
        the maximum of abs(V).
    """
    head = f"C3 - case {case}"
    plot_camera_iquv(
        iquv_sg, sensor_grid.xgrid, sensor_grid.ygrid,
        title=f"{head} - SMART-G - {title_suffix}", i_vmin=i_vmin,
    )
    conftest.savefig(request, bbox_inches="tight")
    plot_camera_difference(
        iquv_sg, iquv_my, sensor_grid.xgrid, sensor_grid.ygrid,
        title=f"{head} - dif(SMART-G - MYSTIC) - {title_suffix}",
        v_diff_frac=v_diff_frac,
    )
    conftest.savefig(request, bbox_inches="tight")


def _is_significant(signal_ref: tuple[float, ...] | None, istk: int
                    ) -> bool:
    """Tell whether a Stokes component is worth asserting on.

    See SIGNAL_FLOOR.

    Parameters
    ----------
    signal_ref : tuple of float, optional
        The mean absolute values of I, Q, U and V. None, i.e. not yet
        measured, keeps every component so that a new reference gets
        fully logged.
    istk : int
        The index of the component, 0 for I.

    Returns
    -------
    bool
        True if the component carries enough signal.
    """
    if signal_ref is None:
        return True

    return signal_ref[istk] > SIGNAL_FLOOR * signal_ref[0]


def _skipped(signal_ref: tuple[float, ...] | None) -> list[str]:
    """Return the names of the components left unasserted, for the log.

    Parameters
    ----------
    signal_ref : tuple of float, optional
        The mean absolute values of I, Q, U and V.

    Returns
    -------
    list of str
        The names of the components below SIGNAL_FLOOR.
    """
    return [
        stk
        for istk, stk in enumerate(["I", "Q", "U", "V"])
        if not _is_significant(signal_ref, istk)
    ]


def _check_deltam(
    delta_m_ref: tuple[float, ...] | None,
    signal_ref: tuple[float, ...] | None,
    iquv_my: tuple[np.ndarray, ...],
    iquv_sg: tuple[np.ndarray, ...],
    label: str,
    tol: float,
) -> list[str]:
    """Compare the delta_m values with the saved validated ones.

    The failure messages are returned instead of asserted, so that a
    test can report every case of a forward group instead of stopping
    at the first one.

    Parameters
    ----------
    delta_m_ref : tuple of float, optional
        The reference delta_m of I, Q, U and V. With None the
        calculated values are logged and the case is reported as a
        failure, which is how a new reference is measured before being
        written in the tables above.
    signal_ref : tuple of float, optional
        The mean absolute values of I, Q, U and V, see SIGNAL_FLOOR.
    iquv_my, iquv_sg : tuple of ndarray
        The MYSTIC and the SMART-G I, Q, U and V maps.
    label : str
        The case label, for the log and the messages.
    tol : float
        The two sided fractional band around the reference.

    Returns
    -------
    list of str
        The failure messages, empty if the case is ok.
    """
    delta_m = compute_deltam(
        obs=list(iquv_my), mod=list(iquv_sg), print_res=False
    )

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
    for istk, stk in enumerate(["I", "Q", "U", "V"]):
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


def _check_means(
    mean_ref: tuple[float, ...] | None,
    signal_ref: tuple[float, ...] | None,
    iquv_sg: tuple[np.ndarray, ...],
    label: str,
) -> list[str]:
    """Compare the spatial mean of each component with the saved one.

    Unlike delta_m, the mean averages the Monte Carlo noise out, so it
    is the observable that keeps the fast tier sensitive to a
    systematic bias. Same contract as _check_deltam.

    Parameters
    ----------
    mean_ref : tuple of float, optional
        The reference means of I, Q, U and V. With None the calculated
        values are logged and the case is reported as a failure.
    signal_ref : tuple of float, optional
        The mean absolute values of I, Q, U and V, see SIGNAL_FLOOR.
    iquv_sg : tuple of ndarray
        The SMART-G I, Q, U and V maps.
    label : str
        The case label, for the log and the messages.

    Returns
    -------
    list of str
        The failure messages, empty if the case is ok.
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

    # The mean of I sets the scale of the four tolerances, the means
    # of Q, U and V being much smaller than it
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
def test_c3_aer_backward(
    request: pytest.FixtureRequest,
    s3db: Smartg,
    atm_c3_aer: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    case: int,
    tier: str,
) -> None:
    """IPRT phase B, cumulus cloud C3, backward, with aerosols."""
    print(f"=== Test C3 case {case} - backward - with aerosols - {tier}")

    ds, norm = _run_case_backward(
        s3db,
        atm_c3_aer,
        sensor_grid,
        case,
        n_photons=N_PHOTONS / PHOTON_DIVIDER[tier],
    )
    iquv_sg = smartg_iquv(ds, norm, N_SENSORS)
    iquv_my = _read_mystic(case + ATM_CASE_OFFSET)

    _plot_case(request, iquv_sg, iquv_my, case, sensor_grid,
               title_suffix="with aer")

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
    request: pytest.FixtureRequest,
    s3db: Smartg,
    atm_c3_aer_gt: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    case: int,
    tier: str,
) -> None:
    """IPRT phase B, cumulus cloud C3, backward, with aerosols.

    With the GT truncated phase matrices.
    """
    print(
        f"=== Test C3 case {case} - backward - with aerosols - GT trunc"
        + f" - {tier}"
    )

    ds, norm = _run_case_backward(
        s3db,
        atm_c3_aer_gt,
        sensor_grid,
        case,
        n_photons=N_PHOTONS / PHOTON_DIVIDER[tier],
    )
    iquv_sg = smartg_iquv(ds, norm, N_SENSORS)
    iquv_my = _read_mystic(case + ATM_CASE_OFFSET)

    _plot_case(request, iquv_sg, iquv_my, case, sensor_grid,
               title_suffix="with aer - GT trunc")

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
