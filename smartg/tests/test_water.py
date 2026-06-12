#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import logging
from pathlib import Path

from smartg.atmosphere import AerOPAC, Atm1D
from smartg.config import DIR_AUXDATA
from smartg.phase import read_phase
from smartg.smartg import RoughSurface, Smartg
from smartg.water import IOP

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

# Column metadata
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


def _parse_hl_pw(raw: str) -> xr.Dataset:
    """Parse a HydroLight Petzold-Water text block into an xr.Dataset.

    Parameters
    ----------
    raw : str
        Multi-line text block.  First column is wavelength (nm),
        remaining columns are the data values.

    Returns
    -------
    xr.Dataset
        One data variable per column, each with a ``wavelength``
        dimension, ``long_name`` and ``units`` attributes.
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
        zip(COLUMN_NAMES, COLUMN_LONG_NAMES, strict=True)
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
    """Return parsed HydroLight irradiance reflectance dataset."""
    return _parse_hl_pw(RAW_HL_PW)


# -----------------------------------------------------------------
# SMART-G simulation fixtures
# -----------------------------------------------------------------

WAVELENGTHS = [440.0, 550.0, 600.0]
SZA_DEG = 30.0
WATER_GRID = [0, -9990, -10000]  # metres; sea bottom at 10 km ≈ ∞


def _build_water_iop() -> IOP:
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

    return IOP(
        phase=phase,
        aw=aw,
        ap=aph,
        aCDOM=ag,
        bw=bw,
        bp=bph,
        Z=WATER_GRID,
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
    """Run SMART-G and return R = Eu/Ed and its MC standard deviation.

    Two runs are performed:

    1. **Flux run** (``flux='planar'``) — gives R = Eu/Ed values.
    2. **Radiance run** (no flux, ``stdev=True``) — gives per-bin
       radiance stdevs which are propagated to the irradiance stdev
       of R.

    Returns
    -------
    r_smartg : np.ndarray  (3,)  — R = Eu/Ed  at 3 wavelengths
    r_stdev  : np.ndarray  (3,)  — MC standard deviation of R
    """
    sg = Smartg(double=True)

    # --- Irradiance run (planar flux, stdev stripped) ---
    m_flux = sg.run(
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
        OUTPUT_LAYERS=3,
        flux="planar",
        stdev=True,
    )
    m_flux = m_flux.to_xarray()
    r_smartg = (
        m_flux["flux_up (0-)"].values
        / m_flux["flux_down (0-)"].values
    )

    # --- Radiance run (no flux → stdev preserved) ---
    m_rad = sg.run(
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
        OUTPUT_LAYERS=3,
        stdev=True,
    )
    m_rad = m_rad.to_xarray()

    i_up = m_rad["I_up (0-)"].values
    i_down = m_rad["I_down (0-)"].values
    i_up_sd = m_rad["I_stdev_up (0-)"].values
    i_down_sd = m_rad["I_stdev_down (0-)"].values

    # Irradiance ≈ sum of radiance over angular bins
    fu = i_up.sum(axis=(1, 2))
    fd = i_down.sum(axis=(1, 2))
    fu_sd = np.sqrt((i_up_sd**2).sum(axis=(1, 2)))
    fd_sd = np.sqrt((i_down_sd**2).sum(axis=(1, 2)))

    r_stdev = r_smartg * np.sqrt((fu_sd / fu) ** 2 + (fd_sd / fd) ** 2)

    return r_smartg, r_stdev


# -----------------------------------------------------------------
# tests
# -----------------------------------------------------------------


def test_hydrolight(hl_pw, _smartg_run):
    """SMART-G R = Eu/Ed must agree with HydroLight within 4Δ.

    The absolute difference between SMART-G and the HydroLight
    reference (from ``RAW_HL_PW``, column ``r_depth_0_minus``) must
    be smaller than four times the SMART-G standard deviation, at
    each of the three matching wavelengths (440, 550, 600 nm).
    """
    r_smartg, r_stdev = _smartg_run

    # HydroLight reference at depth 0- for the 3 test wavelengths
    hl_ref = hl_pw["r_depth_0_minus"]
    r_hl = np.array(
        [
            hl_ref.sel(wavelength=440.0).values,
            hl_ref.sel(wavelength=550.0).values,
            hl_ref.sel(wavelength=600.0).values,
        ]
    )

    for i, wl in enumerate(WAVELENGTHS):
        diff = abs(r_smartg[i] - r_hl[i])
        pct = diff / abs(r_hl[i]) * 100.0
        threshold = 4.0 * r_stdev[i]
        pct_sigma = threshold / abs(r_hl[i]) * 100.0
        status = "PASS" if diff < threshold else "FAIL"
        logger.info(
            f"wl={wl:.0f}nm - "
            f"SMART-G={r_smartg[i]:.4E} - "
            f"HydroLight={r_hl[i]:.4E} - "
            f"diff(%)={pct:.3f} - "
            f"4*sigma(%)={pct_sigma:.3f} - "
            f"{status}"
        )

    np.testing.assert_array_less(
        np.abs(r_smartg - r_hl),
        4.0 * r_stdev,
        err_msg=(
            "SMART-G R=Eu/Ed differs from HydroLight"
            " by more than 4 Δ\n"
            f"  SMART-G  : {r_smartg}\n"
            f"  HydroLight: {r_hl}\n"
            f"  |diff|    : {np.abs(r_smartg - r_hl)}\n"
            f"  4*Δ       : {4.0 * r_stdev}"
        ),
    )
