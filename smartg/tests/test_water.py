"""Tests of the ocean side of SMART-G.

They cover the comparison with HydroLight, the water profiles, the
move of the photons under water and the truncation of the hydrosol
phase matrices.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.typing import NDArray

from smartg.albedo import AlbedoCst, AlbedoMap
from smartg.atmosphere import AerOPAC, Atm1D
from smartg.config import DIR_AUXDATA
from smartg.phase import integ_phase, read_phase
from smartg.sensor import Sensor
from smartg.smartg import LocalEstimate, Smartg
from smartg.surface import Environment, RoughSurface
from smartg.truncation import DMTrunc, GTTrunc
from smartg.water import (
    DEFAULT_WATER_TRUNC,
    Hydrosol,
    HydrosolPR,
    HydrosolZhai,
    Water1D,
    WaterRw,
)

SmartgRun = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]
RwRun = list[
    tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]
]

# -------------------------------------------------
# Logging
# -------------------------------------------------
ROOTPATH = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOTPATH / "smartg" / "tests" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("test_water")
logger.setLevel(logging.INFO)

# Console handler (errors only)
_ch = logging.StreamHandler()
_ch.setLevel(logging.ERROR)
_fmt = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s -"
    " %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

# File handler (info and above)
_fh = logging.FileHandler(LOG_DIR / "water.log", mode="w")
_fh.setLevel(logging.INFO)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)


# -----------------------------------------------------------------
# Raw data: irradiance reflectance R = Eu/Ed from a
# HydroLight / Petzold run
# Columns:  in-air  | depth 0 m | depth 1 m | depth 5 m
# -----------------------------------------------------------------
RAW_HL_PW = """\
440.0   1.0766E-01   1.4224E-01   1.4228E-01   1.4243E-01
495.0   4.9788E-02   3.1616E-02   3.1620E-02   3.1634E-02
550.0   3.6312E-02   5.8226E-03   5.8218E-03   5.8187E-03
575.0   3.4924E-02   3.5155E-03   3.5147E-03   3.5115E-03
600.0   3.3475E-02   1.0209E-03   1.0201E-03   1.0173E-03
"""

# HydroLight Lu/Ed (1/sr) reference
# Columns: in-air | depth 0 m | depth 1 m | depth 5 m
RAW_HL_PW_RRS = """\
440.0   2.9509E-02   4.2502E-02   4.2502E-02   4.2498E-02
495.0   1.0841E-02   9.8604E-03   9.8603E-03   9.8601E-03
550.0   6.5506E-03   1.8459E-03   1.8461E-03   1.8467E-03
575.0   6.1567E-03   1.1166E-03   1.1168E-03   1.1173E-03
600.0   5.7323E-03   3.2492E-04   3.2506E-04   3.2552E-04
"""

# Column metadata (Eu/Ed)
COLUMN_NAMES = [
    "r_depth_0_plus",
    "r_depth_0_minus",
    "r_depth_1",
    "r_depth_5",
]

COLUMN_LONG_NAMES = [
    "R=Eu/Ed, in air (just above sea surface)",
    "R=Eu/Ed, just below sea surface (0 m)",
    "R=Eu/Ed, 1 m below sea surface",
    "R=Eu/Ed, 5 m below sea surface",
]

# Column metadata (Lu/Ed)
RRS_COLUMN_NAMES = [
    "rrs_depth_0_plus",
    "rrs_depth_0_minus",
    "rrs_depth_1",
    "rrs_depth_5",
]

RRS_COLUMN_LONG_NAMES = [
    "Lu/Ed, in air (just above sea surface)",
    "Lu/Ed, just below sea surface (0 m)",
    "Lu/Ed, 1 m below sea surface",
    "Lu/Ed, 5 m below sea surface",
]


def _parse_hl_table(
    raw: str,
    col_names: list[str],
    col_long_names: list[str],
) -> xr.Dataset:
    """Parse a HydroLight text table into an xr.Dataset.

    Parameters
    ----------
    raw : str
        Multi-line text block.  First column is wavelength
        (nm), remaining columns are the data values.
    col_names : list[str]
        Variable names for each data column.
    col_long_names : list[str]
        ``long_name`` attribute for each data column.

    Returns
    -------
    xr.Dataset
        One data variable per column, each with a
        ``wavelength`` dimension, ``long_name`` and ``units``
        attributes.
    """
    wavelengths: list[float] = []
    rows: list[list[float]] = []

    for line in raw.strip().splitlines():
        tokens = line.split()
        wavelengths.append(float(tokens[0]))
        rows.append([float(tok) for tok in tokens[1:]])

    data = np.array(rows)  # shape (n_wavelengths, n_columns)

    data_vars: dict[str, xr.DataArray] = {}
    for idx, (name, long_name) in enumerate(
        zip(col_names, col_long_names, strict=True)
    ):
        data_vars[name] = xr.DataArray(
            data[:, idx],
            dims=["wavelength"],
            coords={"wavelength": wavelengths},
            attrs={
                "long_name": long_name,
                "units": "dimensionless",
            },
        )

    return xr.Dataset(data_vars)


@pytest.fixture(scope="module")
def hl_pw() -> xr.Dataset:
    """Return parsed HydroLight Eu/Ed dataset."""
    return _parse_hl_table(
        RAW_HL_PW, COLUMN_NAMES, COLUMN_LONG_NAMES,
    )


@pytest.fixture(scope="module")
def hl_pw_rrs() -> xr.Dataset:
    """Return parsed HydroLight Lu/Ed dataset."""
    return _parse_hl_table(
        RAW_HL_PW_RRS, RRS_COLUMN_NAMES, RRS_COLUMN_LONG_NAMES,
    )


# -----------------------------------------------------------------
# SMART-G simulation fixtures
# -----------------------------------------------------------------

WAVELENGTHS = [440.0, 550.0, 600.0]
SZA_DEG = 30.0
WATER_GRID = [0, -9990, -10000]  # metres; sea bottom at 10 km ≈ ∞
SEED = 1234


def _build_water_iop(pure_water_path: Path) -> Water1D:
    """Build a custom pure-water IOP profile for SMART-G.

    Pure water is treated as a general scatterer with the analytic
    phase function from Mobley (*Light and Water*, ch. 3, eq. 3.30)
    on top of the Petzold tabulation, and zero absorption. The phase
    function is written to pure_water_path, out of the auxdata.
    """
    hydrolight_dir = DIR_AUXDATA / "validation" / "HYDROLIGHT"

    aph = np.array(
        pd.read_csv(
            hydrolight_dir / "aw1.dat",
            sep=r"\s+",
            header=None,
        )
    )
    bph = np.array(
        pd.read_csv(
            hydrolight_dir / "bw1.dat",
            sep=r"\s+",
            header=None,
        )
    )
    aw = np.zeros_like(aph)
    ag = np.zeros_like(aph)
    bw = np.zeros_like(bph)

    # Read Petzold tabulation, overwrite with Mobley analytic phase
    petzold_path = hydrolight_dir / "petzold.dat"

    data = np.loadtxt(petzold_path)
    theta = np.radians(data[:, 0])
    data[:, 1] = 0.06225 * (1.0 + 0.835 * np.cos(theta) ** 2)
    np.savetxt(pure_water_path, data)

    phase = read_phase(pure_water_path, kind="oc")

    return Water1D(
        grid=WATER_GRID,
        aw=aw,
        bw=bw,
        comp=[Hydrosol(phase=phase, ap=aph, acdom=ag, bp=bph)],
    )


@pytest.fixture(scope="module")
def _water_iop(tmp_path_factory: pytest.TempPathFactory) -> Water1D:
    pure_water_path = tmp_path_factory.mktemp("hydrolight") / "pure_water.dat"
    return _build_water_iop(pure_water_path)


@pytest.fixture(scope="module")
def _atm() -> Atm1D:
    aer = AerOPAC("maritime_clean", 0.2, 550)
    return Atm1D("afglms", tco3=300.0, no2=True, p0=1024.0, comp=[aer])


@pytest.fixture(scope="module")
def _surf() -> RoughSurface:
    return RoughSurface(wind=5.0, nh2o=1.34)


@pytest.fixture(scope="module")
def _smartg_run(
    _water_iop: Water1D,
    _atm: Atm1D,
    _surf: RoughSurface,
) -> SmartgRun:
    """Run SMART-G and return R = Eu/Ed, Lu/Ed and stdev.

    Two runs are performed:

    1. **Flux run** (``flux='planar'``) — gives R = Eu/Ed.
    2. **Local estimate run** — gives Lu/Ed with MC stdev.

    Returns
    -------
    r_smartg : np.ndarray  (3,)  — R = Eu/Ed
    rrs_smartg : np.ndarray  (3,)  — Lu/Ed (1/sr)
    rrs_stdev : np.ndarray  (3,)  — MC stdev of Lu/Ed
    """
    sg = Smartg(double=True)

    # --- Irradiance run (planar flux, 5x photons) ---
    m_flux = sg.run(
        wavelength=WAVELENGTHS,
        th_deg=SZA_DEG,
        atmosphere=_atm,
        surface=_surf,
        water=_water_iop,
        n_photons=5e7,
        n_loop=1e6,
        seed=SEED,
        xblock=64,
        xgrid=1024,
        output_layers=3,
        flux="planar",
    )
    r_smartg = (
        m_flux["flux_up (0-)"].values
        / m_flux["flux_down (0-)"].values
    )

    # --- Local estimate run (Lu/Ed) ---
    local_est = LocalEstimate(
        th_deg=np.array([0.0]),
        phi_deg=np.array([0.0]),
        count_level=np.array([4]),
    )
    m_le = sg.run(
        wavelength=WAVELENGTHS,
        th_deg=SZA_DEG,
        atmosphere=_atm,
        surface=_surf,
        water=_water_iop,
        n_photons=1e7,
        n_loop=1e6,
        seed=SEED,
        xblock=64,
        xgrid=1024,
        output_layers=4,
        n_icdf=1e3,
        le=local_est,
        stdev=True,
    )

    i_le_up = m_le["I_up (0-)"].values
    i_le_up_sd = m_le["I_stdev_up (0-)"].values
    fd_le = m_flux["flux_down (0-)"].values

    rrs_smartg = i_le_up[:, 0, 0] / fd_le / np.pi
    # Ed is considered perfect, so stdev comes from Lu only
    rrs_stdev = i_le_up_sd[:, 0, 0] / fd_le / np.pi

    return r_smartg, rrs_smartg, rrs_stdev


# -----------------------------------------------------------------
# tests
# -----------------------------------------------------------------


MAX_DIFF_PCT = 1.0  # maximum allowed diff in %


def test_hydrolight(
    hl_pw: xr.Dataset,
    hl_pw_rrs: xr.Dataset,
    _smartg_run: SmartgRun,
) -> None:
    """SMART-G must agree with HydroLight within criteria.

    Eu/Ed: relative difference < 1.0%.
    Lu/Ed: absolute difference < 4 * MC stdev (Ed perfect).
    """
    r_smartg, rrs_smartg, rrs_stdev = _smartg_run

    # --- Eu/Ed at depth 0- ---
    r_hl_ref = hl_pw["r_depth_0_minus"]
    r_hl = np.array(
        [
            r_hl_ref.sel(wavelength=440.0).values,
            r_hl_ref.sel(wavelength=550.0).values,
            r_hl_ref.sel(wavelength=600.0).values,
        ]
    )

    logger.info("---- Eu/Ed ----")
    for i, wavelength in enumerate(WAVELENGTHS):
        pct = abs(r_smartg[i] - r_hl[i]) / abs(r_hl[i]) * 100.0
        status = "PASS" if pct < MAX_DIFF_PCT else "FAIL"
        logger.info(
            f"wavelength={wavelength:.0f}nm - "
            f"SMART-G={r_smartg[i]:.4E} - "
            f"HydroLight={r_hl[i]:.4E} - "
            f"diff(%)={pct:.3f} - "
            f"{status}"
        )

    np.testing.assert_array_less(
        np.abs(r_smartg - r_hl) / np.abs(r_hl) * 100.0,
        MAX_DIFF_PCT,
        err_msg=(
            "SMART-G R=Eu/Ed differs from HydroLight"
            " by more than 1.0%\n"
            f"  SMART-G  : {r_smartg}\n"
            f"  HydroLight: {r_hl}"
        ),
    )

    # --- Lu/Ed at depth 0- ---
    rrs_hl_ref = hl_pw_rrs["rrs_depth_0_minus"]
    rrs_hl = np.array(
        [
            rrs_hl_ref.sel(wavelength=440.0).values,
            rrs_hl_ref.sel(wavelength=550.0).values,
            rrs_hl_ref.sel(wavelength=600.0).values,
        ]
    )

    logger.info("---- Lu/Ed ----")
    for i, wavelength in enumerate(WAVELENGTHS):
        diff = abs(rrs_smartg[i] - rrs_hl[i])
        pct = diff / abs(rrs_hl[i]) * 100.0
        threshold = 4.0 * rrs_stdev[i]
        pct_sigma = threshold / abs(rrs_hl[i]) * 100.0
        status = "PASS" if diff < threshold else "FAIL"
        logger.info(
            f"wavelength={wavelength:.0f}nm - "
            f"SMART-G={rrs_smartg[i]:.4E} - "
            f"HydroLight={rrs_hl[i]:.4E} - "
            f"diff(%)={pct:.3f} - "
            f"4*sigma(%)={pct_sigma:.3f} - "
            f"{status}"
        )

    np.testing.assert_array_less(
        np.abs(rrs_smartg - rrs_hl),
        4.0 * rrs_stdev,
        err_msg=(
            "SMART-G Lu/Ed differs from HydroLight"
            " by more than 4 sigma\n"
            f"  SMART-G  : {rrs_smartg}\n"
            f"  HydroLight: {rrs_hl}\n"
            f"  |diff|    : {np.abs(rrs_smartg - rrs_hl)}\n"
            f"  4*sigma   : {4.0 * rrs_stdev}"
        ),
    )


# -----------------------------------------------------------------
# WaterRw vs Water1D equivalence
# -----------------------------------------------------------------

RW_ALBEDO = 0.5

# Variables of a water profile that Smartg.run requires and copies to
# its output, see the ocean profile in `smartg.smartg._finalize`. The
# kernel also receives pine_oc and FQY1_oc, left out: WaterRw sets them
# to 1 where Water1D gives 0, harmless in a layer of null thickness
KERNEL_WATER_VARS = [
    "T_oc",
    "OD_w",
    "OD_p_oc",
    "OD_y",
    "OD_oc",
    "OD_sca_oc",
    "OD_abs_oc",
    "pmol_oc",
    "ssa_oc",
    "albedo_seafloor",
]


def test_waterrw_profile_matches_water1d() -> None:
    """WaterRw must match the equivalent Water1D profile.

    WaterRw is a fast path for a lambertian reflector placed just
    below the air-water interface.  It is the degenerate case of a
    Water1D profile holding no hydrosol and no water column, and must
    stay numerically identical to it for every variable of
    KERNEL_WATER_VARS.
    """
    alb = AlbedoCst(RW_ALBEDO)
    pro_rw = WaterRw(alb=alb).calc(WAVELENGTHS)
    pro_w1d = Water1D(grid=[0.0, 0.0], comp=[], alb=alb).calc(WAVELENGTHS)

    for name in KERNEL_WATER_VARS:
        assert name in pro_rw.data_vars, f"{name} missing from WaterRw"
        assert name in pro_w1d.data_vars, f"{name} missing from Water1D"
        np.testing.assert_allclose(
            pro_rw[name].values,
            pro_w1d[name].values,
            rtol=1e-12,
            atol=0.0,
            err_msg=(
                f"WaterRw and Water1D disagree on '{name}'\n"
                f"  WaterRw : {pro_rw[name].values}\n"
                f"  Water1D : {pro_w1d[name].values}"
            ),
        )

    # neither model scatters, so no phase matrix is expected
    for pro, label in ((pro_rw, "WaterRw"), (pro_w1d, "Water1D")):
        assert "phase_oc" not in pro.data_vars, (
            f"{label} should not define a phase matrix"
        )


STOKES = ["I", "Q", "U", "V"]

# Absolute floor added to the 4-sigma criterion. V stays exactly zero
# for a Rayleigh atmosphere over a lambertian reflector, and so does its
# stdev, so a purely relative criterion would compare 0 < 0 and fail.
# The floor is orders of magnitude below the smallest non-zero Stokes
# component, so it masks nothing real.
STOKES_ATOL = 1e-9


@pytest.fixture(scope="module")
def _atm_rayleigh() -> Atm1D:
    """Rayleigh-only atmosphere: no aerosol, no cloud, no absorption."""
    return Atm1D("afglt", tco3=0.0, tcwp=0.0, no2=False)


@pytest.fixture(scope="module")
def _rw_vs_w1d_run(_atm_rayleigh: Atm1D, _surf: RoughSurface) -> RwRun:
    """Run SMART-G with WaterRw and with the equivalent Water1D.

    Both runs use a Rayleigh atmosphere above a rough ocean surface, so
    the comparison exercises the air-water interface, including its
    effect on polarization.

    The local estimate is taken out of the principal plane, so that U
    does not vanish by symmetry.

    Returns
    -------
    list of (dict, dict)
        One (values, stdev) pair per water model, each keyed by Stokes
        component and holding one value per wavelength.
    """
    sg = Smartg()
    alb = AlbedoCst(RW_ALBEDO)
    local_est = LocalEstimate(
        th_deg=np.array([30.0]),
        phi_deg=np.array([45.0]),
    )
    out = []
    for water in (
        WaterRw(alb=alb),
        Water1D(grid=[0.0, 0.0], comp=[], alb=alb),
    ):
        m = sg.run(
            wavelength=WAVELENGTHS,
            th_deg=SZA_DEG,
            atmosphere=_atm_rayleigh,
            surface=_surf,
            water=water,
            n_photons=2e7,
            n_loop=1e6,
            le=local_est,
            stdev=True,
        )
        out.append(
            (
                {s: m[f"{s}_up (TOA)"].values[:, 0, 0] for s in STOKES},
                {s: m[f"{s}_stdev_up (TOA)"].values[:, 0, 0] for s in STOKES},
            )
        )
    return out


def test_atm_rayleigh_is_purely_scattering(_atm_rayleigh: Atm1D) -> None:
    """The test atmosphere must hold no absorption and no particle."""
    pro = _atm_rayleigh.calc(WAVELENGTHS, phase=False)
    assert np.abs(pro["OD_g"].values).max() == 0.0, "gaseous absorption left"
    assert np.abs(pro["OD_p"].values).max() == 0.0, "particles left"
    assert (pro["OD_r"].values[:, -1] > 0).all(), "no Rayleigh scattering"


@pytest.mark.parametrize("stokes", STOKES)
def test_waterrw_simulation_matches_water1d(
    _rw_vs_w1d_run: RwRun,
    stokes: str,
) -> None:
    """Both models must give the same Stokes vector within MC noise.

    The two runs are independent, so the difference is compared to the
    quadratic sum of their stdevs.
    """
    (val_rw, sd_rw), (val_w1d, sd_w1d) = _rw_vs_w1d_run
    a, b = val_rw[stokes], val_w1d[stokes]
    sigma = np.hypot(sd_rw[stokes], sd_w1d[stokes])
    tol = 4.0 * sigma + STOKES_ATOL

    logger.info(f"---- WaterRw vs Water1D, {stokes}_up (TOA) ----")
    for i, wavelength in enumerate(WAVELENGTHS):
        diff = abs(a[i] - b[i])
        status = "PASS" if diff < tol[i] else "FAIL"
        logger.info(
            f"wavelength={wavelength:.0f}nm - "
            f"WaterRw={a[i]:.4E} - "
            f"Water1D={b[i]:.4E} - "
            f"diff={diff:.3E} - "
            f"tol={tol[i]:.3E} - "
            f"{status}"
        )

    np.testing.assert_array_less(
        np.abs(a - b),
        tol,
        err_msg=(
            f"WaterRw and Water1D {stokes} differ by more than 4 sigma\n"
            f"  WaterRw : {a}\n"
            f"  Water1D : {b}\n"
            f"  |diff|  : {np.abs(a - b)}\n"
            f"  tol     : {tol}"
        ),
    )


def _sea_strip_run(
    codes: list[int], alt_pp: bool, back: bool, seed: int
) -> xr.Dataset:
    """Run clear water 10 m deep under an albedo map of three strips.

    The strips are cut across x at -1 and 1 km. A cell coded -1 is sea,
    whose seafloor has the albedo 1 of the map list, white; a cell
    coded 0 is land, below which the seafloor keeps the black albedo of
    the water profile.
    """
    albedo_map = AlbedoMap(
        np.array([[code] for code in codes]),
        np.array([-1.0, 1.0, 1e8]),
        np.array([1e8]),
        [AlbedoCst(0.0), AlbedoCst(1.0)],
    )
    kwargs: dict[str, Any] = {}
    if back:
        kwargs["sensor"] = Sensor(th_deg=150.0, ph_deg=0.0, loc="SURF0P")
    else:
        kwargs["th_deg"] = 30.0
    return Smartg(alt_pp=alt_pp, back=back).run(
        450.0,
        surface=RoughSurface(wind=2.0),
        water=Water1D(grid=[0.0, -10.0], comp=[], alb=AlbedoCst(0.0)),
        environment=Environment(env=5, alb=albedo_map),
        le=LocalEstimate(th_deg=[30.0], phi_deg=[90.0], count_level=[0]),
        n_photons=1e6,
        stdev=True,
        seed=seed,
        progress=False,
        **kwargs,
    )


@pytest.mark.parametrize("alt_pp", [False, True])
@pytest.mark.parametrize("back", [False, True])
def test_photons_move_under_water_in_kilometres(
    alt_pp: bool, back: bool
) -> None:
    """Check the horizontal move of the photons under water.

    The altitudes of the water profile are in metres, the horizontal
    positions in kilometres. The light enters the sea at the origin, in
    the middle of a strip 2 km wide between two lands, and reaches its
    seafloor 10 m below, some metres away: the radiance must be the one
    of a map of sea only. The photons moved horizontally in kilometres
    the distance they travel in metres, and reached the black seafloor
    below the land.
    """
    coast = _sea_strip_run([0, -1, 0], alt_pp, back, seed=21)
    sea = _sea_strip_run([-1, -1, -1], alt_pp, back, seed=22)
    level = "up (TOA)"
    z = (coast[f"I_{level}"].values - sea[f"I_{level}"].values) / np.hypot(
        coast[f"I_stdev_{level}"].values, sea[f"I_stdev_{level}"].values
    )
    assert np.all(np.abs(z) < 5)


def _fresnel_reflectance(
    mu: float | NDArray[np.float64], n: float
) -> NDArray[np.float64]:
    """Return the Fresnel reflectance of a flat air-water interface.

    Parameters
    ----------
    mu : float or ndarray
        Cosine of the incidence angle in the air.
    n : float
        Relative refractive index water/air.

    Returns
    -------
    ndarray
        Reflectance for unpolarized light coming from the air.
    """
    mu = np.asarray(mu, dtype=float)
    mu_t = np.sqrt(1.0 - (1.0 - mu**2) / n**2)
    r_par = (n * mu - mu_t) / (n * mu + mu_t)
    r_per = (mu - n * mu_t) / (mu + n * mu_t)
    return 0.5 * (r_par**2 + r_per**2)


def _seafloor_reflectances(
    th_sun: float, th_view: float, albedo: float, n: float
) -> tuple[float, float]:
    """Return the analytic radiances over a Lambertian seafloor.

    The water is transparent, the interface flat and the sky black.
    With t = 1 - R the transmittance of the interface seen from the
    air and r_e its white-sky reflectance, the seafloor receives the
    irradiance E = mu_0 F_0 t(th_sun) / (1 - A r_i), where
    r_i = 1 - (1 - r_e) / n^2 is the reflectance of the interface to
    the isotropic radiance A E / pi of the seafloor (reciprocity). That
    radiance leaves the water multiplied by t(th_view) / n^2.

    Parameters
    ----------
    th_sun, th_view : float
        Solar and viewing zenith angles in the air, in degrees.
    albedo : float
        Albedo A of the seafloor.
    n : float
        Relative refractive index water/air.

    Returns
    -------
    rho_air, rho_water : float
        pi L / (mu_0 F_0) of the upwelling radiance just above the
        surface, at th_view, and just below it.
    """
    mu = np.linspace(0.0, 1.0, 20001)
    r_e = 2.0 * np.trapezoid(_fresnel_reflectance(mu, n) * mu, mu)
    t_sun, t_view = 1.0 - _fresnel_reflectance(
        np.cos(np.radians([th_sun, th_view])), n
    )
    denominator = n**2 * (1.0 - albedo) + albedo * (1.0 - r_e)
    rho_water = albedo * t_sun * n**2 / denominator
    return float(rho_water * t_view / n**2), float(rho_water)


def _clear_water(albedo: float) -> Water1D:
    """Return 10 m of water that neither absorbs nor scatters."""
    zeros = np.zeros((1, 2))
    return Water1D(
        grid=[0.0, -10.0], comp=[], aw=zeros, bw=zeros,
        alb=AlbedoCst(albedo),
    )


def _radiance(m: xr.Dataset, level: str) -> tuple[float, float]:
    """Return the radiance of a one-direction run and its stdev."""
    return (
        float(m[f"I_{level}"].values.ravel()[0]),
        float(m[f"I_stdev_{level}"].values.ravel()[0]),
    )


@pytest.mark.parametrize("side", ["air", "water"])
@pytest.mark.parametrize("alt_pp", [False, True])
def test_backward_radiance_crosses_the_interface(
    alt_pp: bool, side: str
) -> None:
    """Forward and backward agree on the analytic seafloor radiance.

    No atmosphere, 10 m of transparent water over a Lambertian seafloor
    of albedo 0.5, a slightly rough interface; sun and view 30 degrees
    from the zenith and 90 degrees apart in azimuth, far from the glint.
    The radiance is looked at just above the surface and just below
    it. A backward photon carries a radiance, which the interface
    divides by n^2 on the way down and multiplies by n^2 on the way up:
    without that factor, and with the refraction local estimate of the
    forward photon, the backward radiance above the water was 1.9 times
    the forward one and the analytic value, and 1.08 times below.
    """
    th, albedo = 30.0, 0.5
    surface = RoughSurface(wind=2.0)
    ref = _seafloor_reflectances(
        th, th, albedo, surface.dict["NH2O"]
    )[side == "water"]
    le = LocalEstimate(th_deg=[th], phi_deg=[90.0], count_level=[0])
    if side == "air":
        sensor = Sensor(th_deg=180.0 - th, ph_deg=0.0, loc="SURF0P")
        fw_kwargs: dict[str, Any] = {"le": le}
        fw_level = "up (TOA)"
    else:
        sensor = Sensor(
            th_deg=180.0 - th, ph_deg=0.0, loc="OCEAN", pos_z=-1e-3
        )
        fw_kwargs = {
            "le": LocalEstimate(
                th_deg=[th], phi_deg=[90.0], count_level=[4]
            ),
            "output_layers": 3,
        }
        fw_level = "up (0-)"
    common: dict[str, Any] = {
        "surface": surface,
        "water": _clear_water(albedo),
        "n_photons": 1e7,
        "stdev": True,
        "seed": 41,
        "progress": False,
    }
    fw, fw_sd = _radiance(
        Smartg(alt_pp=alt_pp).run(
            450.0, th_deg=th, **fw_kwargs, **common
        ),
        fw_level,
    )
    bw, bw_sd = _radiance(
        Smartg(alt_pp=alt_pp, back=True).run(
            450.0, sensor=sensor, le=le, **common
        ),
        "up (TOA)",
    )
    msg = (
        f"forward {fw:.5f}+-{fw_sd:.5f}, backward {bw:.5f}+-{bw_sd:.5f},"
        f" analytic {ref:.5f}"
    )
    logger.info("interface %s alt_pp=%s: %s", side, alt_pp, msg)
    assert abs(fw - bw) < 4 * np.hypot(fw_sd, bw_sd), msg
    # 0.5 % for the rough interface against the flat one of the
    # analytic value
    for value, sd in ((fw, fw_sd), (bw, bw_sd)):
        assert abs(value - ref) < 4 * sd + 5e-3 * ref, msg


@pytest.mark.parametrize("wind", [0.0])
def test_backward_sky_seen_from_under_water(wind: float) -> None:
    """Forward and backward agree on the sky seen from under water.

    A Rayleigh atmosphere, transparent water over a black seafloor, the
    sun 30 degrees from the zenith; the downwelling radiance just below
    the surface, 40 degrees from the zenith and 90 degrees in azimuth
    from the sun, is sky light only. The backward photons leave the
    water to scatter in the atmosphere, and the interface multiplies
    their radiance by n^2: without it the backward radiance was
    1 / n^2 = 0.57 times the forward one.
    """
    common: dict[str, Any] = {
        "atmosphere": Atm1D("afglt"),
        "surface": RoughSurface(wind=wind),
        "water": _clear_water(0.0),
        "stdev": True,
        "seed": 45,
        "progress": False,
    }
    fw, fw_sd = _radiance(
        Smartg().run(
            450.0,
            th_deg=30.0,
            le=LocalEstimate(
                th_deg=[40.0], phi_deg=[90.0], count_level=[2]
            ),
            output_layers=3,
            n_photons=1e8,
            **common,
        ),
        "down (0-)",
    )
    bw, bw_sd = _radiance(
        Smartg(back=True).run(
            450.0,
            sensor=Sensor(
                th_deg=40.0, ph_deg=0.0, loc="OCEAN", pos_z=-1e-3
            ),
            le=LocalEstimate(
                th_deg=[30.0], phi_deg=[90.0], count_level=[0]
            ),
            n_photons=1e7,
            **common,
        ),
        "up (TOA)",
    )
    msg = f"forward {fw:.5f}+-{fw_sd:.5f}, backward {bw:.5f}+-{bw_sd:.5f}"
    logger.info("sky under water, wind %s: %s", wind, msg)
    assert abs(fw - bw) < 4 * np.hypot(fw_sd, bw_sd), msg


def test_hydrosol_calc_phase_truncation() -> None:
    """GPU-free checks of the pytrunc truncation of the derived phase.

    The phase matrices derived from the backscattering ratio must be
    normalized to 2, non-negative, with only F11 and F22 = F11
    non-null, and the scattering factor must be the fraction of the
    forward peak the grid resolves, times 1 - f with a truncation.
    Configurations yielding a negative truncated phase must be
    rejected.
    """
    wavelength = np.array([440.0, 550.0])
    z = np.array([0.0, -10.0])
    bbp = np.full((2, 2), 0.01)

    def calc(**kwargs: Any) -> NDArray[np.float64]:
        h = Hydrosol(bp=0.1, bbp_ratio=0.01, n_theta=721, **kwargs)
        pha, coef = h.calc_phase(wavelength, z, bbp)
        ang = np.deg2rad(pha["theta_oc"].values)
        p = pha.values
        assert not np.isnan(p).any()
        assert (p >= 0.0).all()
        np.testing.assert_allclose(
            integ_phase(ang, p[:, :, 0, :]), 2.0, rtol=2e-3
        )
        np.testing.assert_array_equal(p[:, :, 4, :], p[:, :, 0, :])
        assert (p[:, :, [1, 2, 3, 5], :] == 0.0).all()
        return coef.values

    # no truncation by default: the 721 angles resolve 72 % of the
    # forward peak of the mixture
    resolved = calc()
    np.testing.assert_allclose(resolved, 0.722, atol=1e-3)

    # the recommended GT truncation: 1 - trunc_frac of the resolved part
    np.testing.assert_allclose(
        calc(truncation=DEFAULT_WATER_TRUNC), 0.7 * resolved, rtol=1e-12
    )

    # GT truncation with a searched truncation angle
    coef = calc(
        truncation=GTTrunc(
            trunc_frac=0.5, theta_tol=30.0, lobatto_optimization=True
        )
    )
    np.testing.assert_allclose(coef, 0.5 * resolved, rtol=1e-12)

    # a truncation fraction larger than the energy of the truncated
    # peak gives a negative truncated phase, as does the Legendre
    # ringing of Delta-M on the Fournier-Forand mixtures
    with pytest.raises(ValueError, match="negative"):
        Hydrosol(
            bp=0.1,
            bbp_ratio=0.03,
            n_theta=721,
            truncation=GTTrunc(trunc_frac=0.5, theta_tr=5.0),
        ).calc_phase(wavelength, z, np.full((2, 2), 0.03))
    with pytest.raises(ValueError, match="negative"):
        Hydrosol(
            bp=0.1,
            bbp_ratio=0.01,
            n_theta=721,
            truncation=DMTrunc(n_streams=8),
        ).calc_phase(wavelength, z, bbp)


def test_chlorophyll_hydrosols_scatter_their_whole_bp() -> None:
    """HydrosolPR and HydrosolZhai scatter their whole bp.

    Their scattering coefficient is scaled by the factor their phase
    matrices come with, and by nothing else: they used to halve it
    whenever the phase matrices were calculated, as in every run, but
    not with `phase=False`. HydrosolPR gives the profile of a Hydrosol
    built from its own inherent optical properties.
    """
    wavelength = np.array([443.0, 550.0])
    grid = np.array([0.0, -5.0, -10.0])
    for hydrosol in (
        HydrosolPR(chl=1.0, n_theta=721),
        HydrosolZhai(chl_surf=1.0, n_theta=721),
    ):
        iop = hydrosol.iop(wavelength, grid)
        assert iop["bbp_ratio"] is not None
        _, coef = hydrosol.calc_phase(wavelength, grid, iop["bbp_ratio"])
        np.testing.assert_allclose(
            hydrosol.coeffs(wavelength, grid)["bp"],
            hydrosol.coeffs(wavelength, grid, phase=False)["bp"]
            * coef.values,
            rtol=1e-12,
        )

    pr = HydrosolPR(chl=1.0, n_theta=721)
    iop = pr.iop(wavelength, grid)
    user = Hydrosol(
        bp=iop["bp"], ap=iop["ap"], acdom=iop["acdom"],
        bbp_ratio=iop["bbp_ratio"], n_theta=721,
    )
    pro_pr = Water1D(grid=grid, comp=[pr]).calc(wavelength)
    pro_user = Water1D(grid=grid, comp=[user]).calc(wavelength)
    for var in ["OD_sca_oc", "OD_abs_oc", "phase_oc"]:
        np.testing.assert_allclose(
            pro_pr[var].values, pro_user[var].values, rtol=1e-12,
            err_msg=var,
        )


def _backscattered_fraction(
    f11: NDArray[np.float64], theta_deg: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the backscattered fraction of tabulated phase functions.

    The mass of each angular bin is the one the kernel samples, F11
    linear in theta between the nodes times the true sin(theta), see
    `smartg.smartg._cdf_of_table`. The grid must hold 90 degrees.
    """
    ang = np.deg2rad(theta_deg)
    th0, th1 = ang[:-1], ang[1:]
    f0 = f11[..., :-1]
    df = f11[..., 1:] - f0
    mass = f0 * (np.cos(th0) - np.cos(th1)) + df * (
        (np.sin(th1) - np.sin(th0)) / (th1 - th0) - np.cos(th1)
    )
    back = theta_deg[:-1] >= 90.0
    return mass[..., back].sum(axis=-1) / mass.sum(axis=-1)


@pytest.mark.parametrize("truncation", [None, DEFAULT_WATER_TRUNC])
@pytest.mark.parametrize(
    "make",
    [
        lambda t: Hydrosol(bp=0.1, bbp_ratio=0.005, truncation=t),
        lambda t: Hydrosol(bp=0.1, bbp_ratio=0.02, truncation=t),
        lambda t: HydrosolPR(chl=0.5, truncation=t),
        lambda t: HydrosolZhai(chl_surf=0.5, truncation=t),
    ],
    ids=["Hydrosol-0.005", "Hydrosol-0.02", "HydrosolPR", "HydrosolZhai"],
)
def test_derived_phase_backscattering(
    make: Any, truncation: GTTrunc | None
) -> None:
    """The derived phase gives the requested particle backscattering.

    The scattering coefficient of the profile times the backscattered
    fraction of the phase matrix the kernel samples must be
    `bbp_ratio * bp`, at the default `n_theta` of each class, with and
    without truncation. The forward peak the angular grid does not
    resolve is counted as unscattered: it used to be spread over all
    the angles, backscattering included, by +10 % to +51 %. The phase
    matrix must be non-negative, which the mixture of HydrosolZhai,
    whose ratio of 0.04 weights a Fournier-Forand function negatively,
    was not close to 0 deg.

    The tolerance covers the Park & Ruddick weights, derived from
    backscattered fractions of 0.030 and 0.002 where the two
    Fournier-Forand functions have 0.029963 and 0.0019976: -0.12 %.
    The GT truncation of `DEFAULT_WATER_TRUNC` adds a bias of its own
    on a coarse grid, see the tolerance below.
    """
    hydrosol = make(truncation)
    rtol = 2e-3
    if truncation is not None and hydrosol.n_theta < 7201:
        # GT integrates its plateau as a step up to theta_tr and the
        # kernel as a ramp over the last bin before it, so that the
        # truncated fraction of the table exceeds f: +2.3 % of
        # backscattering at 721 angles (+0.2 % at 7201) for a ratio
        # of 0.005
        rtol = 3e-2
    wavelength = np.array([443.0, 550.0])
    grid = np.array([0.0, -5.0, -10.0])
    pro = Water1D(grid=grid, comp=[hydrosol]).calc(wavelength)
    assert (pro["phase_oc"].values >= 0.0).all()
    iop = hydrosol.iop(wavelength, grid)
    assert iop["bbp_ratio"] is not None
    frac = _backscattered_fraction(
        pro["phase_oc"].values[:, 0, :], pro["theta_oc"].values
    )[pro["iphase_oc"].values]
    bb = hydrosol.coeffs(wavelength, grid)["bp"] * frac
    np.testing.assert_allclose(
        bb[:, 1:], (iop["bbp_ratio"] * iop["bp"])[:, 1:], rtol=rtol
    )


def test_hydrosol_zhai_warns_nothing() -> None:
    """HydrosolZhai evaluates without a floating point warning.

    Its non-algal particles, whose concentration is null, used to take
    the logarithm of 0.
    """
    with np.errstate(all="raise"):
        iop = HydrosolZhai(chl_surf=1.0).iop(
            np.array([443.0, 550.0]), np.array([0.0, -5.0, -10.0])
        )
    assert np.isfinite(iop["bp"]).all()


def test_hydrosol_arrays_interpolated_onto_wavelength_phase() -> None:
    """Arrays over the profile wavelengths work with wavelength_phase.

    The coefficients supplied over the wavelengths of the profile are
    interpolated linearly onto `wavelength_phase` to tabulate the phase
    matrices, where they used to be refused, or paired by position
    with the tabulation wavelengths.
    """
    wavelength = np.array([443.0, 550.0])
    grid = np.array([0.0, -5.0, -10.0])
    ones = np.ones((1, len(grid)))
    hydrosol = Hydrosol(
        bp=np.array([[0.1], [0.2]]) * ones,
        bbp_ratio=np.array([[0.01], [0.02]]) * ones,
        n_theta=721,
        wavelength_phase=[500.0],
    )
    pro = Water1D(grid=grid, comp=[hydrosol]).calc(wavelength)

    # the backscattering ratio interpolated at 500 nm
    bbp_500 = 0.01 + 0.01 * (500.0 - 443.0) / (550.0 - 443.0)
    ref = Hydrosol(
        bp=0.1, bbp_ratio=bbp_500, n_theta=721, wavelength_phase=[500.0]
    )
    pro_ref = Water1D(grid=grid, comp=[ref]).calc(wavelength)
    np.testing.assert_allclose(
        pro["phase_oc"].values, pro_ref["phase_oc"].values, rtol=1e-12
    )
    # the scattering of each wavelength, scaled by the factor of the
    # phase matrix at 500 nm
    np.testing.assert_allclose(
        pro["OD_p_oc"].values,
        pro_ref["OD_p_oc"].values * np.array([[1.0], [2.0]]),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: [
            HydrosolPR(chl=1.0, n_theta=721),
            HydrosolZhai(chl_surf=1.0, n_theta=721),
        ],
        lambda: [
            HydrosolPR(chl=1.0, n_theta=7201),
            Hydrosol(bp=0.1, bbp_ratio=0.01),
        ],
        lambda: [
            HydrosolPR(chl=1.0, n_theta=721, wavelength_phase=[500.0]),
            HydrosolPR(chl=0.2, n_theta=721),
        ],
    ],
    ids=["z_phase", "theta_oc", "wavelength_phase"],
)
def test_water1d_mixes_hydrosols_of_other_grids(make: Any) -> None:
    """Hydrosols tabulated on different grids are mixed.

    A hydrosol constant with depth, tabulated on a single depth, with
    one that varies, hydrosols of different `n_theta` or
    `wavelength_phase`: Water1D used to refuse to mix them. At each
    wavelength and depth, the mixed phase matrix must be the average of
    the phase matrices of the hydrosols alone, weighted by their
    scattering coefficients, each interpolated linearly onto the
    angles of the mixture.
    """
    wavelength = np.array([443.0, 550.0])
    grid = np.array([0.0, -5.0, -10.0])
    pro = Water1D(grid=grid, comp=make()).calc(wavelength)
    theta = pro["theta_oc"].values

    total: Any = 0.0
    bsca: Any = 0.0
    for hydrosol in make():
        alone = Water1D(grid=grid, comp=[hydrosol]).calc(wavelength)
        pha = alone["phase_oc"].values[alone["iphase_oc"].values]
        pha = np.apply_along_axis(
            lambda row, t=alone["theta_oc"].values: np.interp(theta, t, row),
            -1, pha,
        )
        bp = hydrosol.coeffs(wavelength, grid)["bp"]
        total = total + pha * bp[:, :, None, None]
        bsca = bsca + bp
    np.testing.assert_allclose(
        pro["phase_oc"].values[pro["iphase_oc"].values],
        total / bsca[:, :, None, None],
        rtol=1e-12, atol=1e-12,
    )


def test_water1d_mixes_a_hydrosol_given_arrays() -> None:
    """A Hydrosol given its phase and an array bp is mixed.

    Its scattering coefficient, given over the wavelengths and depths
    of the profile, weights its phase matrix at the tabulation
    wavelength of the mixture and at every depth; it used to be
    evaluated on the single wavelength and depth of the phase matrix,
    and refused.
    """
    wavelength = np.array([443.0, 550.0])
    grid = np.array([0.0, -5.0, -10.0])
    pha, _ = Hydrosol(bp=0.1, bbp_ratio=0.015, n_theta=721).calc_phase(
        np.array([550.0]), np.array([0.0]), np.full((1, 1), 0.015)
    )
    bp = np.array([[0.1, 0.1, 0.1], [0.2, 0.3, 0.4]])
    chl = HydrosolPR(chl=0.5, n_theta=721, wavelength_phase=[550.0])
    pro = Water1D(
        grid=grid, comp=[Hydrosol(phase=pha, bp=bp), chl]
    ).calc(wavelength)

    # both are tabulated at 550 nm, which weights every wavelength
    alone = Water1D(grid=grid, comp=[chl]).calc(wavelength)
    pha_chl = alone["phase_oc"].values[alone["iphase_oc"].values[1]]
    bp_chl = chl.coeffs(wavelength, grid)["bp"][1][:, None, None]
    expected = (pha.values[0, 0] * bp[1][:, None, None]
                + pha_chl * bp_chl) / (bp[1][:, None, None] + bp_chl)
    for i in range(len(wavelength)):
        np.testing.assert_allclose(
            pro["phase_oc"].values[pro["iphase_oc"].values[i]], expected,
            rtol=1e-12, atol=1e-12,
        )


def test_hydrosol_1d_coefficient_is_a_depth_profile() -> None:
    """A 1-D coefficient is a depth profile, refused when ambiguous.

    With as many wavelengths as depths, a 1-D array could be either:
    it is refused, where it used to be read silently as a depth
    profile. A spectrum is given the shape (n_wavelength, 1).
    """
    wavelength = np.array([443.0, 550.0])

    def od_p(grid: list[float], bp: Any) -> NDArray[np.float64]:
        """Particle optical thickness of a hydrosol of that `bp`."""
        hydrosol = Hydrosol(bp=bp, bbp_ratio=0.01, n_theta=721)
        pro = Water1D(grid=grid, comp=[hydrosol]).calc(wavelength, phase=False)
        return pro["OD_p_oc"].values

    with pytest.raises(ValueError, match="ambiguous"):
        od_p([0.0, -10.0], [0.1, 0.2])
    with pytest.raises(ValueError, match=r"spectrum of shape \(2, 1\)"):
        od_p([0.0, -5.0, -10.0], [0.1, 0.2])
    np.testing.assert_allclose(
        od_p([0.0, -10.0], [[0.1], [0.2]]), [[0.0, -1.0], [0.0, -2.0]]
    )
    np.testing.assert_allclose(
        od_p([0.0, -10.0], [[0.1, 0.2]]), [[0.0, -2.0], [0.0, -2.0]]
    )
    np.testing.assert_allclose(
        od_p([0.0, -5.0, -10.0], [0.0, 0.1, 0.2]),
        [[0.0, -0.5, -1.5], [0.0, -0.5, -1.5]],
    )


def test_hydrosol_zhai_tabulates_a_single_depth() -> None:
    """A phase matrix constant with depth is tabulated once.

    HydrosolZhai varies its scattering coefficient with depth, but not
    its backscattering ratio, the only input of its phase matrix: it
    used to tabulate the same matrix at every depth.
    """
    wavelength = np.array([443.0, 550.0])
    grid = np.linspace(0.0, -100.0, 11)
    pro = Water1D(
        grid=grid, comp=[HydrosolZhai(chl_surf=0.5, n_theta=721)]
    ).calc(wavelength)
    assert pro["phase_oc"].shape[0] == len(wavelength)
    np.testing.assert_array_equal(
        pro["iphase_oc"].values,
        np.repeat([[0], [1]], len(grid), axis=1),
    )
