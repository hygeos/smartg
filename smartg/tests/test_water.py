#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import logging
from pathlib import Path

from smartg.albedo import AlbedoCst
from smartg.atmosphere import AerOPAC, Atm1D
from smartg.config import DIR_AUXDATA
from smartg.phase import integ_phase, read_phase
from smartg.smartg import Smartg
from smartg.surface import RoughSurface
from smartg.truncation import DM_trunc, GT_trunc
from smartg.water import Hydrosol, Water1D, WaterRw

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


def _build_water_iop() -> Water1D:
    """Build a custom pure-water IOP profile for SMART-G.

    Pure water is treated as a general scatterer with the analytic
    phase function from Mobley (*Light and Water*, ch. 3, eq. 3.30)
    on top of the Petzold tabulation, and zero absorption.
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
    pure_water_path = hydrolight_dir / "pure_water.dat"

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
def _water_iop():
    return _build_water_iop()


@pytest.fixture(scope="module")
def _atm():
    aer = AerOPAC("maritime_clean", 0.2, 550)
    return Atm1D("afglms", tco3=300.0, no2=True, p0=1024.0, comp=[aer])


@pytest.fixture(scope="module")
def _surf():
    return RoughSurface(WIND=5.0, NH2O=1.34)


@pytest.fixture(scope="module")
def _smartg_run(_water_iop, _atm, _surf):
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
        wl=WAVELENGTHS,
        THVDEG=SZA_DEG,
        atm=_atm,
        surf=_surf,
        water=_water_iop,
        NBPHOTONS=5e7,
        NBLOOP=1e6,
        XBLOCK=64,
        XGRID=1024,
        alis_options={"nlow": -1, "njac": 0},
        OUTPUT_LAYERS=3,
        flux="planar",
    )
    m_flux = m_flux.to_xarray()
    r_smartg = (
        m_flux["flux_up (0-)"].values
        / m_flux["flux_down (0-)"].values
    )

    # --- Local estimate run (Lu/Ed) ---
    local_est = {
        "th_deg": np.array([0.0]),
        "phi_deg": np.array([0.0]),
        "count_level": np.array([4]),
    }
    m_le = sg.run(
        wl=WAVELENGTHS,
        THVDEG=SZA_DEG,
        atm=_atm,
        surf=_surf,
        water=_water_iop,
        NBPHOTONS=1e7,
        NBLOOP=1e6,
        XBLOCK=64,
        XGRID=1024,
        alis_options={"nlow": -1, "njac": 0},
        OUTPUT_LAYERS=4,
        NF=1e3,
        le=local_est,
        stdev=True,
    )
    m_le = m_le.to_xarray()

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


def test_hydrolight(hl_pw, hl_pw_rrs, _smartg_run):
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
    for i, wl in enumerate(WAVELENGTHS):
        pct = abs(r_smartg[i] - r_hl[i]) / abs(r_hl[i]) * 100.0
        status = "PASS" if pct < MAX_DIFF_PCT else "FAIL"
        logger.info(
            f"wl={wl:.0f}nm - "
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
    for i, wl in enumerate(WAVELENGTHS):
        diff = abs(rrs_smartg[i] - rrs_hl[i])
        pct = diff / abs(rrs_hl[i]) * 100.0
        threshold = 4.0 * rrs_stdev[i]
        pct_sigma = threshold / abs(rrs_hl[i]) * 100.0
        status = "PASS" if diff < threshold else "FAIL"
        logger.info(
            f"wl={wl:.0f}nm - "
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

# Variables the CUDA kernel reads from a water profile, see the
# prof_oc handling in Smartg.run
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


def test_waterrw_profile_matches_water1d():
    """WaterRw must match the equivalent Water1D profile.

    WaterRw is a fast path for a lambertian reflector placed just
    below the air-water interface.  It is the degenerate case of a
    Water1D profile holding no hydrosol and no water column, and must
    stay numerically identical to it for every variable the kernel
    reads.
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
def _atm_rayleigh():
    """Rayleigh-only atmosphere: no aerosol, no cloud, no absorption."""
    return Atm1D("afglt", tco3=0.0, tcwp=0.0, no2=False)


@pytest.fixture(scope="module")
def _rw_vs_w1d_run(_atm_rayleigh, _surf):
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
    local_est = {
        "th_deg": np.array([30.0]),
        "phi_deg": np.array([45.0]),
    }
    out = []
    for water in (
        WaterRw(alb=alb),
        Water1D(grid=[0.0, 0.0], comp=[], alb=alb),
    ):
        m = sg.run(
            wl=WAVELENGTHS,
            THVDEG=SZA_DEG,
            atm=_atm_rayleigh,
            surf=_surf,
            water=water,
            NBPHOTONS=2e7,
            NBLOOP=1e6,
            le=local_est,
            stdev=True,
        ).to_xarray()
        out.append(
            (
                {s: m[f"{s}_up (TOA)"].values[:, 0, 0] for s in STOKES},
                {s: m[f"{s}_stdev_up (TOA)"].values[:, 0, 0] for s in STOKES},
            )
        )
    return out


def test_atm_rayleigh_is_purely_scattering(_atm_rayleigh):
    """The test atmosphere must hold no absorption and no particle."""
    pro = _atm_rayleigh.calc(WAVELENGTHS, phase=False)
    assert np.abs(pro["OD_g"].values).max() == 0.0, "gaseous absorption left"
    assert np.abs(pro["OD_p"].values).max() == 0.0, "particles left"
    assert (pro["OD_r"].values[:, -1] > 0).all(), "no Rayleigh scattering"


@pytest.mark.parametrize("stokes", STOKES)
def test_waterrw_simulation_matches_water1d(_rw_vs_w1d_run, stokes):
    """Both models must give the same Stokes vector within MC noise.

    The two runs are independent, so the difference is compared to the
    quadratic sum of their stdevs.
    """
    (val_rw, sd_rw), (val_w1d, sd_w1d) = _rw_vs_w1d_run
    a, b = val_rw[stokes], val_w1d[stokes]
    sigma = np.hypot(sd_rw[stokes], sd_w1d[stokes])
    tol = 4.0 * sigma + STOKES_ATOL

    logger.info(f"---- WaterRw vs Water1D, {stokes}_up (TOA) ----")
    for i, wl in enumerate(WAVELENGTHS):
        diff = abs(a[i] - b[i])
        status = "PASS" if diff < tol[i] else "FAIL"
        logger.info(
            f"wl={wl:.0f}nm - "
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


def test_hydrosol_calc_phase_truncation():
    """GPU-free checks of the pytrunc truncation of the derived phase.

    The phase matrices derived from the backscattering ratio must be
    normalized to 2, non-negative, with only F11 and F22 = F11
    non-null, and the truncation factor must be 1 - f (1 without
    truncation). Configurations yielding a negative truncated phase
    must be rejected.
    """
    wav = np.array([440.0, 550.0])
    z = np.array([0.0, -10.0])
    bbp = np.full((2, 2), 0.01)

    def calc(**kwargs):
        h = Hydrosol(bp=0.1, bbp_ratio=0.01, n_theta=721, **kwargs)
        pha, coef = h.calc_phase(wav, z, bbp)
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

    # default GT truncation: coef_trunc = 1 - trunc_frac
    np.testing.assert_allclose(calc(), 0.7)

    # no truncation
    np.testing.assert_array_equal(calc(truncation=None), 1.0)

    # GT truncation with a searched truncation angle
    coef = calc(
        truncation=GT_trunc(
            trunc_frac=0.5, theta_tol=30.0, lobatto_optimization=True
        )
    )
    np.testing.assert_allclose(coef, 0.5)

    # a truncation fraction larger than the energy of the truncated
    # peak gives a negative truncated phase, as does the Legendre
    # ringing of Delta-M on the Fournier-Forand mixtures
    with pytest.raises(ValueError, match="negative"):
        Hydrosol(
            bp=0.1,
            bbp_ratio=0.03,
            n_theta=721,
            truncation=GT_trunc(trunc_frac=0.5, theta_tr=5.0),
        ).calc_phase(wav, z, np.full((2, 2), 0.03))
    with pytest.raises(ValueError, match="negative"):
        Hydrosol(
            bp=0.1,
            bbp_ratio=0.01,
            n_theta=721,
            truncation=DM_trunc(nb_streams=8),
        ).calc_phase(wav, z, bbp)
