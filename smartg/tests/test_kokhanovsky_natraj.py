"""Validation of SMART-G on Rayleigh, aerosol and cloud benchmarks.

- Natraj and Hovenier (2012), "Polarized light reflected and
  transmitted by thick Rayleigh scattering atmospheres", ApJ 748, 28:
  the Stokes parameters of Rayleigh slabs of optical thickness 1 to
  1024 over a Lambertian ground (validation/Rayleigh_Slab_Natraj_2011
  in the auxdata);
- Kokhanovsky et al. (2010), "Benchmark results in vector atmospheric
  radiative transfer", JQSRT 111, 1931: a Rayleigh layer, an aerosol
  layer and a water cloud (validation/*_refl_N_*.dat).

They are the Rayleigh, aerosol and cloud validations of the SMART-G
paper (Ramon et al. 2019, JQSRT 222-223, 89), run with SMART-G and
compared direction by direction with the benchmark values.

Tested with the following GPUs: RTX 5070 Ti
"""

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from smartg.albedo import AlbedoCst
from smartg.atmosphere import AerOPAC, Atm1D
from smartg.config import DIR_AUXDATA
from smartg.phase import read_phase_dat
from smartg.smartg import LocalEstimate, Smartg
from smartg.surface import LambSurface

# *********************** Global variable(s) ***************************
# Fixed seed: seed=-1 would derive it from the clock, giving a new
# noise realisation at every run. The default xblock and xgrid pin the
# rest of it.
SEED = 1234

# Every case runs in two tiers. The slow one is deselected by default
# (see conftest.py). On an RTX 5070 Ti the fast tier takes 32 s of runs
# (the cloud 15 s) and the slow one 4.3 minutes (the cloud 2.1).
TIERS = ["fast", pytest.param("slow", marks=pytest.mark.slow)]
N_PHOTONS = {
    "fast": {"natraj": 1e7, "natraj_ground": 1e7, "rayleigh": 1e7,
             "aerosol": 1e7, "cloud": 1e7},
    "slow": {"natraj": 1e8, "natraj_ground": 1e8, "rayleigh": 1e8,
             "aerosol": 1e8, "cloud": 1e8},
}

VALIDATION = DIR_AUXDATA / "validation"
ROOT_PATH = Path(__file__).resolve().parent.parent

# The Natraj and Hovenier tables: I, Q and U leaving the top of the
# slab (UP), for the grounds of albedo NATRAJ_ALBEDO, the sun at the 50
# cosines NATRAJ_MU0, the 51 view cosines NATRAJ_MU and the relative
# azimuths NATRAJ_PHI, normalised to an incident flux of pi per unit
# area normal to the sun: SMART-G's reflectance is the table divided
# by mu0. Their azimuth is SMART-G's 180 - phi, with the same signs of
# Q and U. The cases are the one of the SMART-G paper, an optical
# thickness of 4 over a black ground under the sun at mu0 = 0.5, and an
# optical thickness of 1 over a ground of albedo 0.8, which couples the
# slab and the ground. The view cosines go from 1 to 0.04 by 0.06, the
# paper's subset, leaving out the horizon. Rayleigh scattering without
# depolarization, conservative and strongly polarizing, is where the
# biased sampling of the scattering angle fails (the notebook of the
# paper showed a bias and a larger variance): these cases are run with
# bias=False.
NATRAJ_ALBEDO = [0.0, 0.25, 0.8]
NATRAJ_MU0 = np.round(np.arange(1, 51) * 0.02, 2)
NATRAJ_MU = np.round(np.arange(0, 51) * 0.02, 2)
NATRAJ_PHI = np.arange(0.0, 181.0, 30.0)
NATRAJ_CASES = {
    "natraj": {"tau": 4, "albedo": 0.0, "mu0": 0.5},
    "natraj_ground": {"tau": 1, "albedo": 0.8, "mu0": 0.5},
}
NATRAJ_VIEW_MU = np.round(np.arange(1.0, 0.03, -0.06), 2)

# The Kokhanovsky et al. benchmarks: the reflection function at the top
# of a layer over a black ground, SMART-G's reflectance, for the sun at
# 60 degrees, view zenith angles of 0 to 89 degrees by 1 and relative
# azimuths of 0, 90 and 180 degrees, all conservative and without
# depolarization. Their azimuth is SMART-G's 180 - phi, with the signs
# of KOKHANOVSKY_SIGN; U and V change sign in the other half plane. The
# paper compared the view zenith angles of 0 to 88 degrees by 4
# (Rayleigh) or by 2 (aerosol and cloud), so are they here. The phase
# matrices are the ARTDECO ones of the benchmark (opt_kokha_*_standard
# .dat: the scattering angle, then F11, F21, F33 and F34), normalized
# again below, see _kokhanovsky_phase.
KOKHANOVSKY_SZA = 60.0
KOKHANOVSKY_WAVELENGTH = 409.71678
KOKHANOVSKY_CASES = {
    "rayleigh": {"tau": 0.3262, "phase": None,
                 "fname": "Rayleigh_refl_N_60.dat", "step": 4},
    "aerosol": {"tau": 0.3262, "phase": "opt_kokha_aer_standard.dat",
                "fname": "aerosol_refl_N_240.dat", "step": 2},
    "cloud": {"tau": 5.0, "phase": "opt_kokha_cl_standard.dat",
              "fname": "cloud_refl_N_360.dat", "step": 2},
}
KOKHANOVSKY_SIGN = {"I": 1.0, "Q": -1.0, "U": -1.0, "V": -1.0}

# Root mean square, over the compared directions, of the difference
# between the run and the benchmark, in units of the standard deviation
# of the run, to which REL_FLOOR times the benchmark value is added in
# quadrature, and ABS_FLOOR for the values that cross 0. It is about 1
# for a run that only differs by its Monte Carlo noise. The floor
# absorbs the 1e-3 by which GPU runs of the same seed differ from one
# process to the next. A Stokes parameter whose benchmark values are
# 0 but for round-off (V in the Rayleigh cases, up to 8e-15) is not
# compared.
#
# Measured on 2026-09-28 on an RTX 5070 Ti, over the compared Stokes
# parameters of each case. Fast tier, three seeds: 0.12 to 1.59, the
# largest being the U of the slab over a ground. Slow tier, one seed:
# 0.04 to 1.03. The bound of 2 fails an I scaled by 1 + e or 1 - e from
# e = 0.3 % in the Rayleigh cases, whose Q and U it catches from 0.3 to
# 0.5 %, 0.75 % (fast) and 0.5 % (slow) in the aerosol, and 2 % in the
# cloud at the fast tier. The U and V of the aerosol and of the cloud,
# and the Q of the cloud, are too noisy for a scaling of 5 % to fire:
# the forward peak makes the local estimates noisy (Buras and Mayer,
# 2011), which is why the cloud z_rms stays close to 1 at 1e8 photons.
REL_FLOOR = 1e-3
ABS_FLOOR = 1e-5
Z_RMS_MAX = {"fast": 2.0, "slow": 2.0}
# **********************************************************************

# **************************** logging *********************************
LOG_DIR = ROOT_PATH / "tests" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "kokhanovsky_natraj.log"
LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)

# Errors on the console, everything in the log file
logger = logging.getLogger("test_kokhanovsky_natraj")
logger.setLevel(logging.INFO)
for handler, level in (
    (logging.StreamHandler(), logging.ERROR),
    (logging.FileHandler(LOG_FILE, mode="w"), logging.INFO),
):
    handler.setLevel(level)
    handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(handler)
# **********************************************************************


def _read_natraj(stokes: str, tau: int) -> xr.DataArray:
    """Read a Natraj and Hovenier table of the light leaving the top.

    Parameters
    ----------
    stokes : str
        'I', 'Q' or 'U'.
    tau : int
        The optical thickness of the slab, 1 or more.

    Returns
    -------
    DataArray
        The table on (albedo, mu0, mu, phi).
    """
    fname = VALIDATION / "Rayleigh_Slab_Natraj_2011" / f"{stokes}_UP_TAU_{tau}"
    rows = [
        line.split() for line in fname.read_text().splitlines()
        if line.strip() and line.split()[0][0].isdigit()
    ]
    shape = (len(NATRAJ_ALBEDO), len(NATRAJ_MU0), len(NATRAJ_MU), 9)
    table = np.array(rows, dtype=np.float64).reshape(shape)
    return xr.DataArray(
        table[..., 2:], dims=("albedo", "mu0", "mu", "phi"),
        coords={"albedo": NATRAJ_ALBEDO, "mu0": NATRAJ_MU0,
                "mu": NATRAJ_MU, "phi": NATRAJ_PHI},
    )


def _read_kokhanovsky(fname: str) -> xr.DataArray:
    """Read a Kokhanovsky et al. benchmark table.

    Parameters
    ----------
    fname : str
        The file in the validation folder: the view zenith angle, then
        I, Q, U and V at the relative azimuths 0, 90 and 180 degrees.

    Returns
    -------
    DataArray
        The table on (stokes, phi, vza).
    """
    table = np.loadtxt(VALIDATION / fname)
    return xr.DataArray(
        table[:, 1:].reshape(-1, 3, 4).transpose(2, 1, 0),
        dims=("stokes", "phi", "vza"),
        coords={"stokes": list("IQUV"), "phi": [0.0, 90.0, 180.0],
                "vza": table[:, 0]},
    )


def _kokhanovsky_phase(fname: str) -> xr.DataArray:
    """Read the phase matrix of a Kokhanovsky et al. benchmark.

    The kernel draws the scattering angles from the table renormalized,
    but weighs the local estimates with its values as they are: they
    are normalized here to 2 over the cosine of the scattering angle,
    F11 interpolated linearly in the angle as the kernel does. The
    file, and read_phase_dat, which normalizes with the trapezoidal
    rule in the cosine, leave 2.0018 and 1.9982 for the cloud.

    Parameters
    ----------
    fname : str
        The file in the validation folder.

    Returns
    -------
    DataArray
        The matrix on ('nphamat', 'theta_atm'), as AerOPAC takes it.
    """
    pha = read_phase_dat(VALIDATION / fname).isel(
        wavelength_phase=0, z_phase=0, drop=True
    )
    theta = pha["theta_atm"].values
    fine = np.union1d(np.linspace(0.0, 180.0, 400001), theta)
    f11 = np.interp(fine, theta, pha.values[0])
    t = np.radians(fine)
    return pha * (2.0 / np.trapezoid(f11 * np.sin(t), t))


def _z_rms(
    run: np.ndarray, sd: np.ndarray, reference: np.ndarray, label: str
) -> float:
    """Compare a Stokes parameter with the benchmark, see Z_RMS_MAX.

    Parameters
    ----------
    run, sd, reference : ndarray
        The run, its standard deviation and the benchmark values over
        the compared directions.
    label : str
        The case and the Stokes parameter, for the log.

    Returns
    -------
    float
        The root mean square of the differences in standard deviations,
        NaN if a standard deviation is.
    """
    sigma = np.sqrt(sd**2 + (REL_FLOOR * reference) ** 2 + ABS_FLOOR**2)
    z = (run - reference) / sigma
    z_rms = float(np.sqrt(np.mean(z**2)))
    bias = np.mean(run - reference) / np.mean(np.abs(reference))
    logger.info(
        f"{label}: z_rms={z_rms:.3f}, max |z|={np.max(np.abs(z)):.2f}, "
        f"rel bias={bias:+.2e}"
    )
    return z_rms


def _run_case(
    smartg: Smartg, case: str, n_photons: float, seed: int = SEED
) -> dict[str, float]:
    """Run a benchmark case and compare it, see Z_RMS_MAX.

    Parameters
    ----------
    smartg : Smartg
        The kernel, compiled with bias=False for the Natraj cases.
    case : str
        A key of NATRAJ_CASES or KOKHANOVSKY_CASES.
    n_photons : float
        Number of photons.
    seed : int, optional
        The seed of the run.

    Returns
    -------
    dict
        The z_rms of each compared Stokes parameter.
    """
    if case in NATRAJ_CASES:
        cfg = NATRAJ_CASES[case]
        theta = np.degrees(np.arccos(NATRAJ_VIEW_MU))
        phi = NATRAJ_PHI
        m = smartg.run(
            wavelength=320.0, th_deg=float(np.degrees(np.arccos(cfg["mu0"]))),
            atmosphere=Atm1D(
                "afglt", tau_r=np.array([cfg["tau"]], dtype=float),
                tco3=0.0, no2=False,
            ),
            surface=(
                LambSurface(alb=AlbedoCst(cfg["albedo"]))
                if cfg["albedo"] else None
            ),
            le=LocalEstimate(th_deg=theta, phi_deg=phi),
            n_photons=n_photons, depol=0.0, output_layers=0, stdev=True,
            seed=seed, progress=False,
        )
        expected = {
            stokes: _read_natraj(stokes, cfg["tau"]).sel(
                albedo=cfg["albedo"], mu0=cfg["mu0"], mu=NATRAJ_VIEW_MU,
                phi=180.0 - phi,
            ).transpose("phi", "mu").values / cfg["mu0"]
            for stokes in "IQU"
        }
    else:
        cfg = KOKHANOVSKY_CASES[case]
        ref = _read_kokhanovsky(cfg["fname"])
        theta = np.arange(0.0, 89.0, cfg["step"])
        phi = np.array([0.0, 90.0, 180.0, 270.0])
        comp = []
        tau_r = cfg["tau"]
        if cfg["phase"] is not None:
            comp = [AerOPAC(
                "maritime_clean", cfg["tau"], KOKHANOVSKY_WAVELENGTH,
                ssa=1.0, phase=_kokhanovsky_phase(cfg["phase"]),
            )]
            tau_r = 0.0
        m = smartg.run(
            wavelength=KOKHANOVSKY_WAVELENGTH, th_deg=KOKHANOVSKY_SZA,
            atmosphere=Atm1D(
                "afglt", comp=comp, tau_r=np.array([tau_r]), tco3=0.0,
                no2=False,
            ),
            le=LocalEstimate(th_deg=theta, phi_deg=phi),
            n_photons=n_photons, depol=0.0, output_layers=0, stdev=True,
            seed=seed, progress=False,
        )
        expected = {}
        for stokes in "IQUV":
            rows = []
            for p in phi:
                # the benchmark azimuth, folded into [0, 180]
                q = (180.0 - p) % 360.0
                sign = KOKHANOVSKY_SIGN[stokes]
                if q > 180.0:
                    q = 360.0 - q
                    if stokes in "UV":
                        sign = -sign
                rows.append(sign * ref.sel(stokes=stokes, phi=q, vza=theta))
            expected[stokes] = np.array(rows)
    result = {}
    scale = np.max(np.abs(expected["I"]))
    for stokes, reference in expected.items():
        if np.max(np.abs(reference)) < 1e-9 * scale:
            continue
        result[stokes] = _z_rms(
            m[f"{stokes}_up (TOA)"].values.ravel(),
            m[f"{stokes}_stdev_up (TOA)"].values.ravel(),
            reference.ravel(), f"{case}, {n_photons:.0e} - {stokes}",
        )
    return result


@pytest.fixture(scope="module")
def s_biased() -> Smartg:
    """Forward compilation in 1D, biased scattering sampling."""
    return Smartg(double=True, bias=True)


@pytest.fixture(scope="module")
def s_unbiased() -> Smartg:
    """Forward compilation in 1D, unbiased scattering sampling."""
    return Smartg(double=True, bias=False)


def _assert(case: str, tier: str, result: dict[str, float]) -> None:
    """Assert every z_rms of a case under the bound of its tier."""
    # a NaN, from a direction without its standard deviation, would
    # pass a plain comparison
    failures = [
        f"{stokes}: z_rms {z:.3f}" for stokes, z in result.items()
        if not z <= Z_RMS_MAX[tier]
    ]
    assert not failures, (
        f"{case}, {tier}: above the z_rms bound of {Z_RMS_MAX[tier]}: "
        + "; ".join(failures)
    )


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("case", list(NATRAJ_CASES))
def test_natraj(s_unbiased: Smartg, case: str, tier: str) -> None:
    """Compare a Rayleigh slab with Natraj and Hovenier (2012)."""
    result = _run_case(s_unbiased, case, N_PHOTONS[tier][case])
    _assert(case, tier, result)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("case", list(KOKHANOVSKY_CASES))
def test_kokhanovsky(s_biased: Smartg, case: str, tier: str) -> None:
    """Compare a benchmark of Kokhanovsky et al. (2010)."""
    result = _run_case(s_biased, case, N_PHOTONS[tier][case])
    _assert(case, tier, result)
