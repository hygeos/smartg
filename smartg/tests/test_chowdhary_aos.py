"""Validation of SMART-G on the AOS testbed of Chowdhary et al. (2020).

Chowdhary, Zhai, Xu, Frouin and Ramon (2020), "Testbed results for
scalar and vector radiative transfer computations of light in
atmosphere-ocean systems", JQSRT 242, 106717 (accepted manuscript:
https://ntrs.nasa.gov/citations/20190034124), give the upwelling
reflectances I, Q and U just above a wind-ruffled sea, up (0+), and at
the top of the atmosphere, up (TOA), at 350, 450, 550 and 650 nm, for
the sun at 30 and 60 degrees, 13 view zenith angles and 4 azimuths, of
four atmosphere-ocean systems:

- AOS-I: a Rayleigh atmosphere over a rough sea, no ocean body;
- AOS-II: a rough sea over 100 m of pure sea water and a black bottom,
  no atmosphere;
- AOS-III: both;
- AOS-IV: AOS-III with forward-peaked hydrosols in the water.

The four are run with SMART-G and compared, direction by direction,
with the testbed values of validation/chowdhary2020_AOS/reference in
the auxdata, the reflectances of the supplementary Tables S2 to S5 of
the paper.

Tested with the following GPUs: RTX PRO 6000 Blackwell, RTX 5070 Ti
"""

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from smartg.atmosphere import Atm1D
from smartg.config import DIR_AUXDATA
from smartg.smartg import LocalEstimate, Smartg
from smartg.surface import RoughSurface
from smartg.water import Hydrosol, Water1D

# *********************** Global variable(s) ***************************
# Fixed seed: seed=-1 would derive it from the clock, giving a new
# noise realisation at every run. The default xblock and xgrid pin the
# rest of it.
SEED = 1234

# Every case runs in two tiers. The slow one, 1e8 photons, is
# deselected by default (see conftest.py). The fast one takes 10 times
# fewer photons, 100 times fewer for AOS-IV, whose hydrosols scatter
# about 80 times along the 100 m of water at 550 and 650 nm. The
# fast tier takes 3 minutes and the slow one an hour and a half on
# an RTX 5070 Ti (95 s and 37 minutes of kernel on an RTX PRO 6000
# Blackwell), AOS-IV taking most of it.
TIERS = ["fast", pytest.param("slow", marks=pytest.mark.slow)]
N_PHOTONS = {
    "fast": {"I": 1e7, "II": 1e7, "III": 1e7, "IV": 1e6},
    "slow": {"I": 1e8, "II": 1e8, "III": 1e8, "IV": 1e8},
}

AOS_DIR = DIR_AUXDATA / "validation" / "chowdhary2020_AOS"
ROOT_PATH = Path(__file__).resolve().parent.parent

# The testbed set-up. The Rayleigh optical thickness of the atmosphere
# at the four wavelengths, without depolarization and without
# absorption; the ocean body of 100 m over a black bottom, with the
# scattering coefficient of pure sea water BW, its absorption A_PURE
# (AOS-II and AOS-III), and for AOS-IV the bulk absorption A_BLK and
# the particulate scattering coefficient BP, for a chlorophyll
# concentration of 0.03 mg/m3 at 350 and 450 nm and 3 mg/m3 at 550 and
# 650 nm, whose scattering matrices are in PHASE_TABLES. Neither the
# air nor the water depolarizes. The sea is the isotropic Cox and
# Munk one of a 7 m/s wind, with a refractive index of 1.34.
WAVELENGTHS = np.array([350.0, 450.0, 550.0, 650.0])
TAU_R = np.array([0.63031, 0.22111, 0.097069, 0.049188])
BW = np.array([0.0134, 0.0045, 0.0019, 0.0010])
A_PURE = np.array([0.0204, 0.0092, 0.0565, 0.3400])
A_BLK = np.array([0.0215, 0.0144, 0.1065, 0.3787])
BP = np.array([0.0422, 0.0335, 0.8050, 0.8050])
WIND = 7.0
NH2O = 1.34
DEPTH = 100.0

# The hydrosol matrices of AOS-IV are the supplementary Tables S1a
# and S1b of the paper, Table_S1a.txt and Table_S1b.txt in the auxdata:
# the scattering angle in degrees, then F11, F21, F33 and F43 divided
# by 4 pi, every 0.1 degree up to 10 degrees and every degree beyond.
# The paper normalizes F11/(4 pi) to 1 over the sphere (its Eq. 8):
# interpolated linearly in the angle, as SMART-G does, the tables
# integrate to 1.0023 and 1.0025, and give the asymmetry parameters and
# the backscattering ratios of the paper, see test_testbed_phase. Two
# things are adapted: the fourth column, F43, is negated into the F34
# that the four-term tables of SMART-G take (convert_phase_to_iparper),
# a sign that only concerns V; and the paper prints the F43 of S1a at
# 25 degrees with a decimal comma, corrected in the file. Table_4a.txt
# and Table_4b.txt, next to them, are their 1 degree extracts, which
# miss the diffraction peak: the first degree holds 10.5 % and 12 % of
# the scattering, but linearly interpolated between the 1191 of 0
# degree and the 75 of 1 degree it holds 3.6 times as much, and
# normalizing such a table scales the rest of the matrix by 0.77: the
# radiance leaving the water came out 16 to 21 % low at 550 and 650 nm.
PHASE_TABLES = ["Table_S1a.txt", "Table_S1b.txt"]

# The azimuths of the testbed put the specular plane at 0 degree where
# SMART-G puts it at 180 degrees: the reference at phi compares with
# SMART-G at 180 - phi, i.e. the azimuth axis [0, 60, 120, 180]
# reversed. For U the paper's azimuths are 0, 60, 180 and 240, and the
# column labelled 120 in the files is the paper's 240, the mirror image
# of 120, where U changes sign. V is not compared: the files hold 0.
U_SIGN = np.array([1.0, 1.0, -1.0, 1.0])[None, :, None]

# The testbed files hold two copy errors of the paper's supplementary
# tables, which the published version keeps (checked against its
# supplementary material), and whose blocks are left out of the
# comparison, as (model, sza, level, Stokes parameters, wavelength
# index, azimuth indices of the files):
# - AOS-II, sun at 60 degrees, up (0+), 450 nm (Table S3b): the U of
#   the principal plane, 0 and 180 degrees, is a copy of I, where it
#   should be 0;
# - AOS-III, sun at 30 degrees, up (0+), 450 nm, azimuths 60 and 120
#   of the files (Table S4a, the paper's 60 and 240): the values are
#   those of 350 nm. SMART-G agrees there with an older version of the
#   file, which the final one replaced.
MASKED = [
    ("II", 60, "up (0+)", "U", 1, [0, 3]),
    ("III", 30, "up (0+)", "IQU", 1, [1, 2]),
]

# The sea surface. The testbed uses the Cox and Munk reflection without
# shadowing, without multiple reflections between facets and without
# renormalizing it to conserve energy (the paper's Tables 1 and 4).
# SMART-G reproduces it with RoughSurface(brdf=True), to the noise level
# in every direction, which is how the paper compared SMART-G, on AOS-I
# alone (its Sec. 4.4): that surface only reflects. AOS-II to IV need
# the slope sampling, which also transmits into the water, with
# single=True, the facets reflecting once. The slope sampling draws a
# facet among those facing the photon, weighted by their projected
# area, so that shadowing and the conservation of energy are built in
# (the paper's Table 6), and it reflects less of the light meeting the
# sea at grazing incidence than the testbed's formula. Run on AOS-I, it
# gives up (0+) 0 to 0.4 % below the testbed on average up to a view
# zenith angle of 30 degrees, but 0.5 to 2 % at 45 degrees and 8 to
# 9.5 % at 60, and a TOA radiance 0.1 to 0.45 % below it up to 50
# degrees, but 1 % for I and 1.5 to 1.8 % for Q at 55 and 60. In
# AOS-III under the sun at 60 degrees, whose blue sky light meets the
# sea at grazing incidence, the deficit just above it reaches 0.7 to
# 0.85 % at 350 and 450 nm. It is a difference between the models, not
# a regression: SMART-G v1.0.8 and v1.2.0 depart from the testbed the
# same way at large view zenith angles (up to 11 % at 60 degrees), and
# sit up to 0.3 % above it at small ones where 2.0 sits up to 0.3 %
# below (AOS-III, 1e8 photons), 2.0 having corrected their too large
# weight at grazing incidence. AOS-II to IV are therefore compared up
# to VZA_MAX, 30 degrees just above the sea and 50 degrees at TOA, with
# a relative floor of REL_FLOOR, see Z_RMS_MAX. AOS-I is compared
# everywhere.
VZA_MAX = {
    "up (0+)": {"I": 60.0, "II": 30.0, "III": 30.0, "IV": 30.0},
    "up (TOA)": {"I": 60.0, "II": 50.0, "III": 50.0, "IV": 50.0},
}

# Root mean square, over the compared directions, of the difference
# between the run and the testbed, in units of the standard deviation
# of the run, to which REL_FLOOR times the testbed value is added in
# quadrature, and 1e-5 for the values that cross 0. It is about 1 for a
# run that only differs by its Monte Carlo noise. The floor of AOS-I
# absorbs the 1e-3 by which GPU runs of the same seed differ from one
# process to the next; the one of AOS-II to IV the surface model
# difference above.
#
# Measured on 2026-09-27 and 28, over I, Q and U at both levels. Fast
# tier, three seeds per case: 0.56 to 1.63, the largest being the I of
# AOS-IV just above the sea for the sun at 60 degrees. Slow tier, with
# SEED: 0.40 to 1.02. Compared up to 60 degrees at TOA, as AOS-I is,
# the Q at TOA of AOS-III and AOS-IV reached 1.53 to 1.76 at the slow
# tier, from the surface model difference at 55 and 60 degrees alone,
# which the 50 degrees of VZA_MAX leave out.
#
# Scaling a Stokes parameter of the run by 1 + e or 1 - e, the bound
# of 2 fails AOS-I from e = 0.5 % for I and Q at the fast tier (0.3 %
# for I at the slow one) and 0.5 to 1.5 % for U. The floor of 0.5 %
# sets what it catches in AOS-II to IV, much the same at both tiers:
# 1.5 to 3 % of I and of Q and 2 to 5 % of U in AOS-III and AOS-IV;
# in AOS-II, which the glint dominates, 1.5 to 2 % of I, but 5 % for
# the sun at 60 degrees at the fast tier.
REL_FLOOR = {"I": 1e-3, "II": 5e-3, "III": 5e-3, "IV": 5e-3}
ABS_FLOOR = 1e-5
Z_RMS_MAX = {"fast": 2.0, "slow": 2.0}
# **********************************************************************

# **************************** logging *********************************
LOG_DIR = ROOT_PATH / "tests" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "chowdhary_aos.log"
LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)

# Errors on the console, everything in the log file
logger = logging.getLogger("test_chowdhary_aos")
logger.setLevel(logging.INFO)
for handler, level in (
    (logging.StreamHandler(), logging.ERROR),
    (logging.FileHandler(LOG_FILE, mode="w"), logging.INFO),
):
    handler.setLevel(level)
    handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(handler)
# **********************************************************************


def _testbed_phase(fname: Path) -> xr.DataArray:
    """Read a hydrosol matrix of the testbed, see PHASE_TABLES.

    Parameters
    ----------
    fname : Path
        Table_S1a.txt or Table_S1b.txt: the scattering angle in
        degrees, then F11, F21, F33 and F43 divided by 4 pi.

    Returns
    -------
    DataArray
        The matrix on the dimensions of read_phase_dat, F34 = -F43 in
        place of F43, scaled by 4 pi to the normalization of SMART-G,
        2 over the cosine of the scattering angle.
    """
    table = np.loadtxt(fname)
    pha = 4.0 * np.pi * table[:, 1:] * np.array([1.0, 1.0, 1.0, -1.0])
    return xr.DataArray(
        pha.T[None, None],
        coords=[[0.0], [0.0], np.arange(4), table[:, 0]],
        dims=["wavelength_phase", "z_phase", "nphamat", "theta_oc"],
        name="phase_oc",
    )


def _column(values: np.ndarray) -> np.ndarray:
    """Profile of a homogeneous ocean body on the grid [0, -DEPTH]."""
    return np.stack([np.zeros_like(values), values], axis=1)


def _run_aos(
    smartg: Smartg, model: str, sza: int, n_photons: float,
    theta: np.ndarray, phi: np.ndarray,
) -> xr.Dataset:
    """Run an AOS model of the testbed.

    Parameters
    ----------
    smartg : Smartg
        The compiled kernel.
    model : str
        'I', 'II', 'III' or 'IV'.
    sza : int
        Solar zenith angle in degrees.
    n_photons : float
        Number of photons.
    theta, phi : ndarray
        View zenith and azimuth angles of the local estimate, degrees.

    Returns
    -------
    Dataset
        The SMART-G output, with standard deviations.
    """
    atmosphere = None
    if model != "II":
        atmosphere = Atm1D("afglms", tau_r=TAU_R, tco3=0.0, no2=False)
    water = None
    if model == "I":
        surface = RoughSurface(wind=WIND, nh2o=NH2O, sur=1, brdf=True)
    else:
        surface = RoughSurface(wind=WIND, nh2o=NH2O, sur=3, single=True)
        comp, absorption = [], A_PURE
        if model == "IV":
            pha_a, pha_b = (
                _testbed_phase(AOS_DIR / "hydrosol_phase" / table)
                for table in PHASE_TABLES
            )
            pha = xr.concat(
                [pha_a, pha_a, pha_b, pha_b], dim="wavelength_phase"
            ).assign_coords(wavelength_phase=WAVELENGTHS)
            comp, absorption = [Hydrosol(phase=pha, bp=_column(BP))], A_BLK
        water = Water1D(
            grid=[0.0, -DEPTH], aw=_column(absorption), bw=_column(BW),
            comp=comp,
        )
    return smartg.run(
        wavelength=WAVELENGTHS, th_deg=float(sza), atmosphere=atmosphere,
        surface=surface, water=water,
        le=LocalEstimate(th_deg=theta, phi_deg=phi), n_photons=n_photons,
        depol=0.0, depol_water=0.0, output_layers=3, stdev=True,
        seed=SEED, progress=False,
    )


@pytest.mark.parametrize(
    ("table", "g_paper", "bb_ratio"),
    [(PHASE_TABLES[0], 0.95, 0.0108), (PHASE_TABLES[1], 0.97, 0.0058)],
)
def test_testbed_phase(table: str, g_paper: float, bb_ratio: float) -> None:
    """The hydrosol matrices are those of the paper, see PHASE_TABLES.

    Interpolated linearly in the angle, F11 integrates to 1 over the
    sphere within 0.3 %, and gives the asymmetry parameter of the paper
    and the backscattering ratio of its Table 3, at the chlorophyll
    concentration of the table, 0.03 or 3 mg/m3.
    """
    pha = _testbed_phase(AOS_DIR / "hydrosol_phase" / table)
    theta = pha["theta_oc"].values
    fine = np.union1d(np.linspace(0.0, 180.0, 400001), theta)
    t = np.radians(fine)
    f11 = np.interp(fine, theta, pha.values[0, 0, 0])
    w = f11 * np.sin(t) / 2.0
    norm = np.trapezoid(w, t)
    assert norm == pytest.approx(1.0, abs=0.003)
    g = np.trapezoid(w * np.cos(t), t) / norm
    assert g == pytest.approx(g_paper, abs=0.005)
    back = fine >= 90.0
    bb = np.trapezoid(w[back], t[back]) / norm
    assert bb == pytest.approx(bb_ratio, rel=0.01)


@pytest.fixture(scope="module")
def s1d() -> Smartg:
    """Forward compilation in 1D."""
    return Smartg(double=True)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("sza", [30, 60])
@pytest.mark.parametrize("model", ["I", "II", "III", "IV"])
def test_aos(s1d: Smartg, model: str, sza: int, tier: str) -> None:
    """Compare an AOS model with the testbed, see Z_RMS_MAX."""
    ref = xr.open_dataset(AOS_DIR / "reference" / f"ml{sza}_AOS_{model}.nc")
    theta = ref["Zenith angles"].values
    phi = ref["Azimuth angles"].values
    m = _run_aos(s1d, model, sza, N_PHOTONS[tier][model], theta, phi)
    label = f"AOS-{model}, sun at {sza}, {tier}"
    failures = []
    for level in ("up (0+)", "up (TOA)"):
        if f"I_{level}" not in ref:
            continue
        compared = np.ones(ref[f"I_{level}"].shape, dtype=bool)
        compared &= (theta <= VZA_MAX[level][model])[None, None, :]
        for stokes in "IQU":
            ok = compared.copy()
            for mod, s, lev, stks, iwl, iphi in MASKED:
                if (mod, s, lev) == (model, sza, level) and stokes in stks:
                    ok[iwl, iphi, :] = False
            testbed = ref[f"{stokes}_{level}"].values[ok]
            run = m[f"{stokes}_{level}"].values[:, ::-1, :]
            if stokes == "U":
                run = run * U_SIGN
            sd = m[f"{stokes}_stdev_{level}"].values[:, ::-1, :][ok]
            sigma = np.sqrt(
                sd**2 + (REL_FLOOR[model] * testbed) ** 2 + ABS_FLOOR**2
            )
            z = (run[ok] - testbed) / sigma
            z_rms = float(np.sqrt(np.mean(z**2)))
            bias = np.mean(run[ok] - testbed) / np.mean(np.abs(testbed))
            logger.info(
                f"{label} - {stokes} {level}: z_rms={z_rms:.3f}, "
                f"max |z|={np.max(np.abs(z)):.2f}, rel bias={bias:+.2e}"
            )
            # a NaN, from a direction without its standard deviation,
            # would pass a plain comparison
            if not z_rms <= Z_RMS_MAX[tier]:
                failures.append(f"{stokes} {level}: z_rms {z_rms:.3f}")
    assert not failures, (
        f"{label}: above the z_rms bound of {Z_RMS_MAX[tier]}: "
        + "; ".join(failures)
    )
