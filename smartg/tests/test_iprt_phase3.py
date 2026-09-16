"""Non-regression tests of the IPRT phase 3 cases.

The spherical geometry cases of smartg.iprt.phase3, the one layer
cases D1 to D6 and the vertically inhomogeneous ones E1 to E5, are
compared with saved SMART-G results. E6, the camera at 300 000 km, is
not covered yet.

Tested with the following GPUs: 5070 Ti
"""

import importlib
import logging
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import xarray as xr

from smartg import conftest
from smartg.config import DIR_AUXDATA

# *********************** Global variable(s) ***************************
# Fixed seed: seed=-1 would derive it from the clock, giving a new
# noise realisation at every run. XBLOCK and XGRID, which also pin the
# noise realisation, are set by run_sim of the module (64 and 1024).
SEED = 1234

# Every case runs in two tiers. The slow one uses the photon count of
# the saved results, 1e8 per viewing direction, about 3.8 hours for
# the 11 cases, and is deselected by default (see conftest.py). The
# fast one, 1e6 per viewing direction, takes about 5 minutes for the
# file and is the one that runs routinely.
TIERS = ["fast", pytest.param("slow", marks=pytest.mark.slow)]
N_PHOTONS = {"fast": 1e6, "slow": 1e8}

# The cases, by the suffix of their case_ function in the module and
# of their output file iprt_phase3_<case>.nc. The aerosol and cloud
# cases (D3, D4, D5, E3, E4, E5) run on the native scattering angles
# of their files, the saved results were computed on 18001 equally
# spaced angles: the two agree within the Monte Carlo noise.
CASE_NAMES = ["d1", "d2", "d3", "d4", "d5", "d6",
              "e1", "e2", "e3", "e4", "e5"]

# Saved results: 1e8 photons per viewing direction, the v4 run of
# October 2025 (res_iprt_phase3_1e8photons_v4), in the IPRT output
# format written by to_iprt_output: radiance and std on (zout, sza,
# saa, vza, vaa, stokes).
REF_DIR = DIR_AUXDATA / "IPRT" / "phase3" / "smartg_ref_res"

# Both the test run and the saved result estimate their Monte Carlo
# noise (stdev=True), so each direction is compared in units of the
# combined sigma: z = (test - ref) / sqrt(sigma_test^2 + sigma_ref^2).
# Two observables are asserted, per output level and Stokes parameter:
#
# - the bias of the mean over the 8 x 19 x 19 directions, relative to
#   the mean |I| of the saved result. It averages the noise down and,
#   being linear, is unbiased whatever the photon count, so it is what
#   catches a systematic error: at the fast tier, 1e6 photons give a
#   noise of about 1e-3 of I per direction in the Rayleigh cases, and
#   the biases measured against the saved results are below 0.1 %.
# - the fraction of the directions with |z| > 3, which catches an
#   error confined to a few directions (a sun position, the horizon)
#   that the mean would dilute.
#
# The per direction statistics are not those of a unit normal at low
# photon counts: the local estimate is right skewed at TOA and behind
# the forward peak of the cloud phase functions (D5, E5), so a low
# count run and its sigma both come out a little low, and |z| > 3 is
# reached by 2 to 10 % of the directions at TOA in the Rayleigh and
# aerosol cases, 11 % in D5 and 23 % in E5, at the fast tier. The
# excess shrinks with the photon count. The tolerances are therefore
# measured at the tier's photon count, see the tables below, and not
# derived from a unit normal.
#
# A Stokes parameter whose mean |value| in the saved result falls
# below SIGNAL_FLOOR times the one of I is noise (V everywhere, 1e-4
# of I at most; it is even exactly zero, with a zero sigma, in the
# Rayleigh cases): it is logged, not asserted.
SIGNAL_FLOOR = 1e-3
Z_OUTLIER = 3.0

# Tolerance on the bias of the mean, as a fraction of the mean |I| of
# the saved result, and on the fraction of |z| > Z_OUTLIER. The "*"
# entry is the default, the cloud cases (D5, E5) and the rough ocean
# case (D6) get their own, wider ones: the forward peak of the phase
# functions and the sun glint make the noise of a few directions 10 to
# 100 times larger than in the Rayleigh cases, and those directions
# move the mean.
#
# Fast tier, measured with SEED and with a second seed (4321). The
# fractions are the same with both seeds to 0.01: at most 0.099 (E3,
# E4 and E1 at TOA), 0.109 (D5) and 0.232 (E5 at TOA). The biases are
# below 7.5e-4 except D5 (1.4e-3), E5 (5.1e-3 on U at TOA) and D6
# (5.1e-3 on I at TOA with the second seed, 5.7e-4 with SEED: the
# glint). The tolerances leave a margin of about 1.5 on the fractions
# and 3 on the biases. Of the 7.5e-4, D3 keeps -7e-4 with both seeds
# (mean z -0.27 at BOA): the waso.mie.cdf table has 68 native angles,
# on which its normalisation integral differs slightly from the one
# on the 18001 angles of the saved result.
#
# Slow tier: not measured (about 3.8 hours), the values are estimates.
# With the same photon count on both sides the skew is the same on
# both sides too, so the bias should fall to a few 1e-4 (D3 keeping
# its -7e-4) and the fraction close to the 2 to 6 % of the Rayleigh
# cases at the fast tier. Measure them from the log if they fire.
MEAN_TOL = {
    "fast": {"*": 0.003, "d5": 0.005, "d6": 0.015, "e5": 0.015},
    "slow": {"*": 0.002, "d5": 0.003, "d6": 0.005, "e5": 0.005},
}
FRAC_TOL = {
    "fast": {"*": 0.15, "d5": 0.20, "e5": 0.35},
    "slow": {"*": 0.10, "d5": 0.15, "e5": 0.15},
}

# Figures of the html report: the Stokes parameters of the test run at
# this sun zenith angle, in polar view as the notebook draws them
# (radiances times NORM), and the map of z at the same angle, on a
# colour scale of +-Z_SCALE.
PLOT_SZA = 60.0
NORM = 1.0 / np.pi
Z_SCALE = 4.0

ZOUT_NAMES = ["BOA", "TOA"]
STOKES = ["I", "Q", "U", "V"]
ROOT_PATH = Path(__file__).resolve().parent.parent
# **********************************************************************

# **************************** logging *********************************
LOG_DIR = ROOT_PATH / "tests" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "iprt_phase3.log"
LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)

# Errors on the console, everything in the log file
logger = logging.getLogger("test_phase3")
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
def phase3() -> ModuleType:
    """The module of the phase 3 cases.

    It compiles its two kernels when imported, so the import is
    deferred from the collection to the first test.
    """
    return importlib.import_module("smartg.iprt.phase3")


def _tolerance(table: dict[str, dict[str, float]], tier: str, case: str
               ) -> float:
    """Return the tolerance of a case at a tier.

    Parameters
    ----------
    table : dict
        MEAN_TOL or FRAC_TOL.
    tier : str
        'fast' or 'slow'.
    case : str
        The case name, e.g. 'd1'.

    Returns
    -------
    float
        The entry of the case, or the default one, "*".
    """
    return table[tier].get(case, table[tier]["*"])


def _open(folder: str | Path, case: str
          ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return the radiance and std arrays of a case.

    Parameters
    ----------
    folder : str or Path
        The folder of the IPRT output file iprt_phase3_<case>.nc.
    case : str
        The case name, e.g. 'd1'.

    Returns
    -------
    radiance, std : ndarray
        The arrays on (zout, sza, vza, vaa, stokes), the saa axis, of
        length 1, being dropped.
    coords : dict of ndarray
        The zout, sza, vza and vaa coordinates, for the checks and the
        figures.
    """
    ds = xr.open_dataset(Path(folder) / f"iprt_phase3_{case}.nc").load()
    coords = {
        name: ds[f"{case}_{name}"].values
        for name in ["zout", "sza", "vza", "vaa"]
    }
    return (
        ds[f"radiance_{case}"].values[:, :, 0],
        ds[f"std_{case}"].values[:, :, 0],
        coords,
    )


def _compare(
    test: np.ndarray,
    sig_test: np.ndarray,
    ref: np.ndarray,
    sig_ref: np.ndarray,
    ref_i: np.ndarray,
) -> dict[str, float]:
    """Return the statistics of one Stokes parameter at one level.

    Parameters
    ----------
    test, sig_test : ndarray
        The values of the test run and their standard deviations.
    ref, sig_ref : ndarray
        The values of the saved result and their standard deviations.
    ref_i : ndarray
        The I values of the saved result at the same level.

    Returns
    -------
    dict
        The bias of the mean relative to the mean abs(ref_i) ("bias"),
        and, over the directions where the combined sigma is not zero,
        their number ("n"), the mean, rms and largest abs(z) ("mean_z",
        "rms_z", "max_z") and the fraction of abs(z) > Z_OUTLIER
        ("frac"). The number of directions with a zero combined sigma
        is "zero".
    """
    sig = np.sqrt(sig_test**2 + sig_ref**2)
    ok = sig > 0
    stats = {
        "bias": float((test.mean() - ref.mean()) / np.abs(ref_i).mean()),
        "n": int(ok.sum()),
        "zero": int((~ok).sum()),
        "mean_z": np.nan,
        "rms_z": np.nan,
        "max_z": np.nan,
        "frac": np.nan,
    }
    if ok.any():
        z = (test[ok] - ref[ok]) / sig[ok]
        stats.update(
            mean_z=float(z.mean()),
            rms_z=float(np.sqrt(np.mean(z**2))),
            max_z=float(np.abs(z).max()),
            frac=float(np.mean(np.abs(z) > Z_OUTLIER)),
        )
    return stats


def _plot(
    request: pytest.FixtureRequest,
    phase3: ModuleType,
    case: str,
    rad: np.ndarray,
    sig: np.ndarray,
    ref: np.ndarray,
    sig_ref: np.ndarray,
    coords: dict[str, np.ndarray],
    tier: str,
) -> None:
    """Save the polar views of the test run and of z in the html report.

    One figure of each per output level, at the sun zenith angle
    PLOT_SZA, drawn with plot_polar_iprt as the notebook does: the
    viewing azimuth columns at 360 - vaa.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request of the test.
    phase3 : ModuleType
        The smartg.iprt.phase3 module.
    case : str
        The case name, e.g. 'd1'.
    rad, sig : ndarray
        The radiances of the test run and their standard deviations.
    ref, sig_ref : ndarray
        The same, for the saved result.
    coords : dict of ndarray
        The zout, sza, vza and vaa coordinates.
    tier : str
        'fast' or 'slow', for the titles.
    """
    isza = int(np.argmin(np.abs(coords["sza"] - PLOT_SZA)))
    sza = coords["sza"][isza]
    vza = coords["vza"]
    phis = coords["vaa"][::-1] + 180.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.nan_to_num(
            (rad - ref) / np.sqrt(sig**2 + sig_ref**2),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
    for iz, zname in enumerate(ZOUT_NAMES):
        zout = coords["zout"][iz]
        head = f"IPRT case {case.upper()} - SZA = {sza:.0f} - {zout:.0f}km"
        stokes = [rad[iz, isza, :, :, k] * NORM for k in range(4)]
        phase3.plot_polar_iprt(
            *stokes, thetas=vza, phis=phis,
            title=f"{head} - SMART-G {tier} tier ({zname})",
        )
        conftest.savefig(request, bbox_inches="tight")

        stokes = [z[iz, isza, :, :, k] for k in range(4)]
        phase3.plot_polar_iprt(
            *stokes, thetas=vza, phis=phis,
            min_i=-Z_SCALE, max_i=Z_SCALE, max_q=Z_SCALE, max_u=Z_SCALE,
            max_v=Z_SCALE, cmap_i="RdBu_r",
            title=f"{head} - z = (test - ref) / sigma ({zname})",
        )
        conftest.savefig(request, bbox_inches="tight")


def _log_times(folder: str | Path, case: str, label: str) -> None:
    """Log the processing times of the BOA and TOA runs.

    Parameters
    ----------
    folder : str or Path
        The folder of the intermediate files of the case.
    case : str
        The case name, e.g. 'd1'.
    label : str
        The label of the case in the log.
    """
    for zname in ["boa", "toa"]:
        ds = xr.open_dataset(Path(folder) / f"iprt_phase3_{case}_{zname}.nc")
        logger.info(
            f"{label} - {zname.upper()} - "
            f"{float(ds.attrs['processing time (s)']):.1f} s - "
            f"{int(ds.attrs['NPhotonIn_sum']):.2e} photons"
        )
        ds.close()


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("case", CASE_NAMES)
def test_phase3(
    request: pytest.FixtureRequest,
    phase3: ModuleType,
    case: str,
    tier: str,
    tmp_path: Path,
) -> None:
    """IPRT phase 3, one case against its saved result."""
    print(f"=== Test phase 3 case {case.upper()} - {tier}")
    label = f"{case.upper()} - {tier}"

    run_case = getattr(phase3, f"case_{case}")
    run_case(
        nphotons=N_PHOTONS[tier], overwrite=True, output_dir=tmp_path,
        seed=SEED,
    )
    _log_times(tmp_path, case, label)

    rad, sig, coords = _open(tmp_path, case)
    ref, sig_ref, coords_ref = _open(REF_DIR, case)
    for name, values in coords.items():
        assert np.array_equal(values, coords_ref[name]), name
    assert rad.shape == ref.shape

    _plot(request, phase3, case, rad, sig, ref, sig_ref, coords, tier)

    mean_tol = _tolerance(MEAN_TOL, tier, case)
    frac_tol = _tolerance(FRAC_TOL, tier, case)
    errors = []
    for iz, zname in enumerate(ZOUT_NAMES):
        ref_i = ref[iz, ..., 0]
        signal = np.abs(ref[iz]).mean(axis=(0, 1, 2)) / np.abs(ref_i).mean()
        for istk, stk in enumerate(STOKES):
            stats = _compare(
                rad[iz, ..., istk], sig[iz, ..., istk],
                ref[iz, ..., istk], sig_ref[iz, ..., istk], ref_i,
            )
            asserted = signal[istk] > SIGNAL_FLOOR and stats["n"] > 0
            logger.info(
                f"{label} - {zname} - {stk}: bias={stats['bias']:+.3e}; "
                f"mean z={stats['mean_z']:+.3f}; rms z={stats['rms_z']:.3f}; "
                f"max |z|={stats['max_z']:.2f}; "
                f"frac |z|>{Z_OUTLIER:.0f}={stats['frac']:.4f}; "
                f"n={stats['n']}; zero sigma={stats['zero']}; "
                f"signal={signal[istk]:.2e}"
                + ("" if asserted else " - below SIGNAL_FLOOR, not asserted")
            )
            if not asserted:
                continue
            if abs(stats["bias"]) > mean_tol:
                errors.append(
                    f"{label} - {zname}: problem with the mean of {stk}, "
                    f"get a bias of {stats['bias']:.3e} of the mean |I|. "
                    f"It must be within +-{mean_tol:.1e}"
                )
            if stats["frac"] > frac_tol:
                errors.append(
                    f"{label} - {zname}: problem with the {stk} values, "
                    f"{stats['frac']:.4f} of the directions differ by more "
                    f"than {Z_OUTLIER:.0f} sigma. The fraction must be "
                    f"below {frac_tol:.2f}"
                )
    assert not errors, "\n".join(errors)
