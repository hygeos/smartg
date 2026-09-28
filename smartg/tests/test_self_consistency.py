"""Self-consistency validations of SMART-G.

The same quantity, computed in two different ways, must agree within
the Monte Carlo noise of the two runs:

- the equivalence theorem: the absorption as a Beer-Lambert weight
  along the path (beer=1) or as the single scattering albedo at each
  collision (beer=0), in each move mode;
- ALIS (Emde, Buras and Mayer 2011, JQSRT 112, 1622) against the
  standard run, with water;
- the Jacobians that ALIS computes on perturbed profiles (n_jac)
  against the finite differences of standard runs.

They are sections 3 and 4 of the validation notebook of the SMART-G
paper (Validation_smartg_compilation.ipynb, removed in 22daaf3), which
compared the runs by eye.

Tested with the following GPUs: RTX 5070 Ti
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from smartg.atmosphere import AerOPAC, Atm1D
from smartg.smartg import (
    Alis,
    LocalEstimate,
    Smartg,
    multi_profiles,
    reduce_diff,
)
from smartg.surface import RoughSurface
from smartg.water import DEFAULT_WATER_TRUNC, HydrosolPR, Water1D

# *********************** Global variable(s) ***************************
# Fixed seed: seed=-1 would derive it from the clock, giving a new
# noise realisation at every run. The two runs compared take SEED and
# SEED + 1, so that their noises are independent. The default xblock
# and xgrid pin the rest of it.
SEED = 1234

# Every case runs in two tiers. The slow one is deselected by default
# (see conftest.py). The ALIS runs, through the water, take half as
# many photons as the equivalence theorem; their standard runs take
# as many per wavelength (and profile). On an RTX 5070 Ti the fast
# tier takes 3.5 minutes and the slow one 30, the Jacobians half of
# it.
TIERS = ["fast", pytest.param("slow", marks=pytest.mark.slow)]
N_PHOTONS = {
    "fast": {"equivalence": 1e7, "alis": 5e6},
    "slow": {"equivalence": 1e8, "alis": 5e7},
}

ROOT_PATH = Path(__file__).resolve().parent.parent
LEVELS = ["up (TOA)", "down (0+)", "up (0+)", "down (0-)", "up (0-)",
          "down (B)"]
# The levels above the water, where Q, U and V of ALIS are compared,
# see test_alis_with_water
ABOVE_WATER = ["up (TOA)", "down (0+)", "up (0+)"]

# The move modes and the kernel options that select them
MODES = {
    "pp": {},
    "alt_pp": {"alt_pp": True},
    "sp": {"pp": False},
}

# Root mean square, over the directions (and wavelengths) of a Stokes
# parameter at a level, of the difference between the two runs in
# units of its standard deviation, both runs' deviations added in
# quadrature. It is about 1 when the runs only differ by their noise,
# but the directions of a local estimate share their photons: the
# noise of I is mostly common to all of them, so that the z_rms of I is
# closer to a single draw than to a mean over the directions.
#
# Measured on 2026-09-28 on an RTX 5070 Ti, over the compared outputs
# of each test. Fast tier, four seeds: at most 1.91 (the equivalence
# theorem over the water), 1.41 (ALIS) and 1.25 (the Jacobians). Slow
# tier, two seeds: 1.97, 1.44 and 1.29. The bound of 3 fails a relative
# difference of I of about three standard deviations of the
# difference: in the fast tier of the equivalence theorem 0.3 % above
# the water and 2 to 5 % below it, three times less in the slow tier,
# and 40 % more in the ALIS tests, which take half the photons.
Z_RMS_MAX = {"fast": 3.0, "slow": 3.0}
# **********************************************************************

# **************************** logging *********************************
LOG_DIR = ROOT_PATH / "tests" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "self_consistency.log"
LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)

# Errors on the console, everything in the log file
logger = logging.getLogger("test_self_consistency")
logger.setLevel(logging.INFO)
for handler, level in (
    (logging.StreamHandler(), logging.ERROR),
    (logging.FileHandler(LOG_FILE, mode="w"), logging.INFO),
):
    handler.setLevel(level)
    handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(handler)
# **********************************************************************


def _water(chl: float = 0.3) -> Water1D:
    """Return 10 m of water of a chlorophyll concentration in mg/m3.

    Five layers over a black bottom, the phase matrix truncated as by
    default: a shallow water keeps the runs short.
    """
    return Water1D(
        grid=np.linspace(0.0, -10.0, 6),
        comp=[HydrosolPR(chl, truncation=DEFAULT_WATER_TRUNC)],
    )


def _z_rms(
    run: np.ndarray, sd: np.ndarray, other: np.ndarray,
    sd_other: np.ndarray, label: str,
) -> float:
    """Compare two runs of a Stokes parameter, see Z_RMS_MAX.

    Parameters
    ----------
    run, sd, other, sd_other : ndarray
        The two runs and their standard deviations, over the compared
        directions.
    label : str
        The case, the level and the Stokes parameter, for the log.

    Returns
    -------
    float
        The root mean square of the differences in standard deviations,
        infinite if a difference meets a zero or undefined deviation.
    """
    sigma = np.sqrt(sd**2 + sd_other**2)
    diff = run - other
    # a direction where both runs are the same without any spread
    # agrees, any other difference over a zero or NaN deviation fails
    z = np.divide(
        diff, sigma, out=np.where(diff == 0.0, 0.0, np.inf), where=sigma > 0
    )
    z_rms = float(np.sqrt(np.mean(z**2)))
    scale = np.mean(np.abs(other))
    logger.info(
        f"{label}: z_rms={z_rms:.3f}, max |z|={np.max(np.abs(z)):.2f}, "
        f"rel diff={np.mean(diff) / scale if scale else 0.0:+.2e}"
    )
    return z_rms


def _compare(
    run: xr.Dataset, other: xr.Dataset, label: str,
    polarized_levels: list[str] | None = None,
) -> dict[str, float]:
    """Compare every Stokes parameter at every level of two runs.

    Parameters
    ----------
    run, other : Dataset
        The two SMART-G outputs, with stdev=True.
    label : str
        The case, for the log.
    polarized_levels : list of str, optional
        The levels where Q, U and V are compared, all by default; I is
        compared at every level.

    Returns
    -------
    dict
        The z_rms of each compared level and Stokes parameter. A
        level that one run leaves out, or where I is 0 in both, is
        skipped, and so is a Stokes parameter that is 0 but for
        round-off in both.
    """
    result = {}
    for level in LEVELS:
        if f"I_{level}" not in run or f"I_{level}" not in other:
            continue
        scale = max(
            np.max(np.abs(run[f"I_{level}"].values)),
            np.max(np.abs(other[f"I_{level}"].values)),
        )
        if scale == 0.0:
            continue
        for stokes in "IQUV":
            if (
                stokes != "I" and polarized_levels is not None
                and level not in polarized_levels
            ):
                continue
            values = [
                m[f"{stokes}{infix}{level}"].values.ravel()
                for m in (run, other) for infix in ("_", "_stdev_")
            ]
            peak = max(np.max(np.abs(values[0])), np.max(np.abs(values[2])))
            if peak < 1e-9 * scale:
                continue
            result[f"{stokes}_{level}"] = _z_rms(
                values[0], values[1], values[2], values[3],
                f"{label} - {stokes}_{level}",
            )
    return result


def _assert(label: str, tier: str, result: dict[str, float]) -> None:
    """Assert every z_rms of a case under the bound of its tier."""
    assert result, f"{label}: nothing compared"
    # a NaN, from a direction without its standard deviation, would
    # pass a plain comparison
    failures = [
        f"{key}: z_rms {z:.3f}" for key, z in result.items()
        if not z <= Z_RMS_MAX[tier]
    ]
    assert not failures, (
        f"{label}, {tier}: above the z_rms bound of {Z_RMS_MAX[tier]}: "
        + "; ".join(failures)
    )


@pytest.fixture(scope="module")
def kernels() -> dict[str, Smartg]:
    """Hold the Smartg objects, compiled on first use."""
    return {}


def _kernel(kernels: dict[str, Smartg], name: str) -> Smartg:
    """Return the Smartg of a move mode or of ALIS, compiled once."""
    if name not in kernels:
        if name == "alis":
            options = MODES["alt_pp"] | {"alis": True}
        else:
            options = MODES[name]
        kernels[name] = Smartg(**options)
    return kernels[name]


# *********************** equivalence theorem **************************
# The equivalence theorem: the absorption as a weight along the path
# (beer=1, the default) or at the collisions (beer=0). The first case
# is the one of the notebook, the Chappuis band of ozone at 580 nm in
# a molecular atmosphere, in each of the three move modes; the second
# adds the absorption of an urban aerosol, whose single scattering
# albedo is about 0.8, and of the water, below a rough sea, in the two
# plane parallel modes, the spherical one refusing a water body.
EQUIVALENCE_CASES = [
    ("chappuis", "pp"), ("chappuis", "alt_pp"), ("chappuis", "sp"),
    ("ocean", "pp"), ("ocean", "alt_pp"),
]


def _equivalence_setup(scene: str) -> dict[str, Any]:
    """Return the run keywords of a scene of the equivalence theorem."""
    if scene == "chappuis":
        return {
            "wavelength": 580.0,
            "atmosphere": Atm1D(
                "afglt", grid=np.linspace(100.0, 0.0, 101), tco3=300.0,
                no2=False,
            ),
            "output_layers": 1,
        }
    return {
        "wavelength": 550.0,
        "atmosphere": Atm1D(
            "afglt", grid=np.linspace(100.0, 0.0, 26), tco3=300.0,
            comp=[AerOPAC("urban", 0.3, 550.0)],
        ),
        "surface": RoughSurface(wind=5.0),
        "water": _water(),
        "output_layers": 3,
    }


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize(("scene", "mode"), EQUIVALENCE_CASES)
def test_equivalence_theorem(
    kernels: dict[str, Smartg], scene: str, mode: str, tier: str
) -> None:
    """Compare the absorption at the collisions and along the path."""
    smartg = _kernel(kernels, mode)
    runs = [
        smartg.run(
            th_deg=45.0, n_photons=N_PHOTONS[tier]["equivalence"],
            beer=beer,
            le=LocalEstimate(
                th_deg=np.linspace(0.0, 80.0, 9), phi_deg=[0.0, 90.0, 180.0]
            ),
            stdev=True, seed=SEED + beer, progress=False,
            **_equivalence_setup(scene),
        )
        for beer in (0, 1)
    ]
    label = (
        f"equivalence {scene} {mode}, "
        f"{N_PHOTONS[tier]['equivalence']:.0e}"
    )
    _assert(label, tier, _compare(runs[0], runs[1], label))


# ****************************** ALIS **********************************
# ALIS draws the wavelength of each photon among the wavelengths of the
# run, then weighs its path at every other one with the ratio of their
# absorption and scattering coefficients and of their phase functions.
# That correction is a scalar: the Q, U and V of a wavelength carry the
# polarization of the wavelength drawn, which is exact only if the
# polarizing properties of the medium do not change across the band.
# Those of the water do, its molecular scattering falling as the
# wavelength to the -4.3 and that of the particles as its inverse: over
# the 50 nm of these runs, Q and U below the surface are off by 5 to
# 10 % at the ends of the band (up (0-), 2026-09-28), above the water
# they are not. So I is compared at every level, and Q, U and V above
# the water only.
ALIS_WAVELENGTHS = np.linspace(500.0, 550.0, 6)
ALIS_LE = LocalEstimate(th_deg=[0.0, 30.0, 45.0, 60.0],
                        phi_deg=[0.0, 90.0, 180.0])


def _alis_setup() -> dict[str, Any]:
    """Return the keywords shared by the ALIS and standard runs."""
    return {
        "th_deg": 65.0, "le": ALIS_LE, "surface": RoughSurface(wind=12.0),
        "output_layers": 3, "stdev": True, "progress": False,
    }


@pytest.mark.parametrize("tier", TIERS)
def test_alis_with_water(kernels: dict[str, Smartg], tier: str) -> None:
    """Compare ALIS with the standard run, the notebook's water case.

    An atmosphere of the US standard gases over 10 m of water below a
    sea roughened by a 12 m/s wind, from 500 to 550 nm: the absorption
    of the water nearly triples across the band, its scattering drops
    by a third. ALIS selects every wavelength (n_low=-1), which leaves
    no interpolation of its corrections; the standard run takes as
    many photons per wavelength as the ALIS one in total.
    """
    n_photons = N_PHOTONS[tier]["alis"]
    common: dict[str, Any] = _alis_setup() | {
        "atmosphere": Atm1D("afglus", grid=np.linspace(100.0, 0.0, 51)),
        "water": _water(),
    }
    alis = _kernel(kernels, "alis").run(
        ALIS_WAVELENGTHS, n_photons=n_photons, seed=SEED,
        alis_options=Alis(n_low=-1), **common,
    )
    standard = _kernel(kernels, "alt_pp").run(
        ALIS_WAVELENGTHS, n_photons=n_photons * ALIS_WAVELENGTHS.size,
        seed=SEED + 1, **common,
    )
    label = f"ALIS with water, {n_photons:.0e}"
    _assert(label, tier, _compare(alis, standard, label, ABOVE_WATER))


# **************************** Jacobians *******************************
# The Jacobians ALIS computes along the same paths for perturbed
# profiles, packed after the reference one on the wavelength axis, as
# the notebook did for ozone and chlorophyll: n_jac=2 with an ozone
# column raised from 300 to 450 DU and a chlorophyll concentration
# raised from 0.3 to 0.6 mg/m3; n_jac_abs, which reuses the scattering
# corrections of the reference profile and so fits the absorption
# alone, with the ozone and n_low=2, the wavelength in the middle
# taking interpolated corrections. The perturbations are large so that
# the finite differences of the standard runs, whose noises do not
# cancel, stand out of them: at 1e7 photons, 20 to 80 standard
# deviations for the ozone Jacobians of I but that of the upwelling
# below the surface, 14 (up (0-)) to 460 (down (B)) for the
# chlorophyll ones below the surface.
#
# ALIS weighs a perturbed profile as it weighs another wavelength, with
# a scalar (see test_alis_with_water): the Jacobians of I are exact,
# those of Q, U and V only for a perturbation of the absorption, which
# leaves the polarization of the scattering as it is. So Q, U and V are
# compared above the water, for the ozone only: the Jacobians of Q and
# U with respect to the chlorophyll, which changes the share of the
# molecular scattering in the water, were 2 to 2.8 standard deviations
# off above the water at 5e7 photons (2026-09-28).
JACOBIAN_WAVELENGTHS = np.array([500.0, 525.0, 550.0])
TCO3, D_TCO3 = 300.0, 150.0
CHL, D_CHL = 0.3, 0.3
JACOBIAN_CASES = {
    "o3_chl": {
        "profiles": [(TCO3, CHL), (TCO3 + D_TCO3, CHL), (TCO3, CHL + D_CHL)],
        "names": ["O3", "Chl"], "delta": [D_TCO3, D_CHL],
        "absorption": [True, False],
        "alis": Alis(n_low=-1, n_jac=2),
    },
    "o3_abs": {
        "profiles": [(TCO3, CHL), (TCO3 + D_TCO3, CHL)],
        "names": ["O3"], "delta": [D_TCO3], "absorption": [True],
        "alis": Alis(n_low=2, n_jac=1, n_jac_abs=True),
    },
}


def _jacobian(
    m: xr.Dataset, case: str, key: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return the Jacobians of an output and their standard deviations.

    The deviation of a Jacobian adds those of its two blocks in
    quadrature, as for independent runs: for ALIS, whose blocks share
    their paths, that overestimates it.
    """
    cfg = JACOBIAN_CASES[case]
    stokes, level = key.split("_", 1)
    diff = reduce_diff(m, cfg["names"], delta=cfg["delta"])
    n_wl = JACOBIAN_WAVELENGTHS.size
    sd = m[f"{stokes}_stdev_{level}"].values
    jacobians, sds = [], []
    for k, (name, delta) in enumerate(zip(cfg["names"], cfg["delta"]), 1):
        jacobians.append(diff[f"d{key}/d{name}"].values.ravel())
        sds.append((np.sqrt(
            sd[k * n_wl:(k + 1) * n_wl] ** 2 + sd[:n_wl] ** 2
        ) / delta).ravel())
    return jacobians, sds


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("case", list(JACOBIAN_CASES))
def test_alis_jacobians(
    kernels: dict[str, Smartg], case: str, tier: str
) -> None:
    """Compare the ALIS Jacobians with finite differences.

    The profiles are those of test_alis_with_water, with the ozone
    column of the tropical atmosphere; the standard run takes the same
    multi-profile atmosphere and water, as many photons per wavelength
    and profile as the ALIS one in total. Q, U and V are compared above
    the water, for the perturbations of the absorption only, see
    JACOBIAN_CASES.
    """
    cfg = JACOBIAN_CASES[case]
    n_photons = N_PHOTONS[tier]["alis"]
    profiles = cfg["profiles"]
    wavelengths = np.tile(JACOBIAN_WAVELENGTHS, len(profiles))
    common: dict[str, Any] = _alis_setup() | {
        "atmosphere": multi_profiles([
            Atm1D(
                "afglt", tco3=tco3, grid=np.linspace(100.0, 0.0, 51)
            ).calc(JACOBIAN_WAVELENGTHS)
            for tco3, _ in profiles
        ], kind="atm"),
        "water": multi_profiles(
            [_water(chl).calc(JACOBIAN_WAVELENGTHS) for _, chl in profiles],
            kind="oc",
        ),
    }
    alis = _kernel(kernels, "alis").run(
        wavelengths, n_photons=n_photons, seed=SEED,
        alis_options=cfg["alis"], **common,
    )
    standard = _kernel(kernels, "alt_pp").run(
        wavelengths, n_photons=n_photons * wavelengths.size, seed=SEED + 1,
        **common,
    )
    label = f"Jacobians {case}, {n_photons:.0e}"
    result = {}
    for level in LEVELS:
        if f"I_{level}" not in alis:
            continue
        for stokes in "IQUV" if level in ABOVE_WATER else "I":
            key = f"{stokes}_{level}"
            j_alis, sd_alis = _jacobian(alis, case, key)
            j_std, sd_std = _jacobian(standard, case, key)
            for name, absorption, a, sa, b, sb in zip(
                cfg["names"], cfg["absorption"], j_alis, sd_alis, j_std,
                sd_std,
            ):
                if stokes != "I" and not absorption:
                    continue
                result[f"d{key}/d{name}"] = _z_rms(
                    a, sa, b, sb, f"{label} - d{key}/d{name}"
                )
    _assert(label, tier, result)
