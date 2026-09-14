#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Angular grid of the phase matrix on a 1D ice cloud, end to end.

test_phase_grid.py checks the host tables and the device lookup where
they are deterministic. This test runs the kernel: the Iwabuchi and
Suzuki (2009) figure 3 setup of notebooks/demo_notebook.py, with
the truncation off and the water cloud replaced by the Baum aggregated
solid column ice table, the sharpest phase function of the three ice
tables in auxdata. The radiance reflected at TOA and transmitted at
the surface is computed at two viewing angles for every angle grid
kind and size, and compared with the values saved below, which were
measured with the same seed and photon count, within the Monte Carlo
noise the run estimates for itself (stdev=True), so that the check
holds on any GPU model.

The angle grid is what is under test, so the grids are not judged
against the file grid but against a uniform grid fine enough to hold
every node of the file: the kernel indexes a uniform grid analytically
and every other grid by bisection in a table, so the reference and the
candidates go through different code paths. That reference was
measured with 1e10 photons and is only logged, the assertion is
against the saved 1e8 values. See
others/iwabuchi_angular_grid_benchmark.md for the study.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from smartg.atmosphere import Atm1D, Cloud
from smartg.phase import theta_grid
from smartg.smartg import Smartg

# ************************ Global variable(s) **************************
# Fixed seed and CUDA block/grid: seed=-1 would derive the seed from
# the clock, and the RNG is seeded per thread over a XBLOCK*XGRID
# state buffer, so all three pin the noise realisation of the values
# saved below.
SEED = 1234
XBLOCK = 128
XGRID = 1024

# The saved values below were measured with this photon count. The
# photons are launched in 10 kernel loops so that Smartg.run estimates
# the Monte Carlo noise from the spread between loops (stdev=True);
# the noise of the total is that of a single launch of NBPHOTONS.
NBPHOTONS = 1e8
NBLOOP = 1e7

# Iwabuchi and Suzuki (2009) figure 3: a cloud of optical thickness 5
# at 500 nm between 0 and 1 km, effective radius 8 um, conservative
# scattering, no other scatterer or absorber, sun at 60 degrees in
# the principal plane. The radiance is counted by local estimate at
# two viewing angles in that plane, 60 and 27.5 degrees from nadir.
WAVELENGTH = 500.0
CLOUD = "ic_baum_asc"
REFF = 8.0
Z_BOTTOM = 0.0
Z_TOP = 1.0
TAU = 5.0
SZA = 60.0
VZA = np.array([-60.0, -27.5])

# Band around the saved values, in units of the noise of a difference
# of two independent realisations, sqrt(2) times the sigma the run
# estimates for its own value. The saved values are one realisation:
# the same seed gives another one on another GPU model, since the
# threads are scheduled differently, so the band must hold whatever
# the GPU. Measured with stdev=True at the settings above, the
# relative sigma is 0.1% (transmission at 60 degrees) to 0.5%
# (reflection); over 96 values from two GPU models and two seeds, the
# worst deviation from the saved values was 1.04%, 2.4 sigma of the
# difference. Four sigma keeps a false failure below 1e-4 per value
# with a sigma estimated from 10 loops, and still tells apart the
# grid effects of the study: the coarse uniform-451 grid at +3% in
# reflection and +9% in transmission, and the +1% of uniform-1801 in
# transmission at 60 degrees, where the band is 0.6%.
NSIGMA = 4.0

# The angle grids under test: "kind-n" for the generators of
# smartg.phase.theta_grid, "native" for the grid of the file.
GRIDS = [
    "uniform-18001",
    "uniform-451",
    "uniform-1801",
    "lobatto-451",
    "lobatto-1801",
    "peak-451",
    "peak-1801",
    "native",
]

# Reference of the study, in the order of VZA: uniform-18001 with 1e10
# photons, seed 1234. Every node of the file grid lies on a 0.01 degree
# lattice, so this grid reproduces the file exactly through the
# analytic index path of the kernel. Not asserted, since 1e8 photons
# cannot resolve it: the relative difference of every grid against it
# is written to the log.
REF_18001 = {
    "I_up (TOA)": (7.850642e-01, 5.050812e-01),
    "I_down (0+)": (2.570866e01, 5.736278e-01),
}

# Values measured with the settings above (SEED, XBLOCK, XGRID,
# NBPHOTONS, NBLOOP), in the order of VZA, per grid. Regenerate them
# from the log with the same settings if the physics legitimately
# changes.
SAVED = {
    "uniform-18001": {
        "I_up (TOA)": (7.837796e-01, 5.031467e-01),
        "I_down (0+)": (2.571826e01, 5.765969e-01),
    },
    "uniform-451": {
        "I_up (TOA)": (8.095347e-01, 5.153202e-01),
        "I_down (0+)": (2.803431e01, 6.122661e-01),
    },
    "uniform-1801": {
        "I_up (TOA)": (7.861330e-01, 5.025746e-01),
        "I_down (0+)": (2.598059e01, 5.778464e-01),
    },
    "lobatto-451": {
        "I_up (TOA)": (7.892513e-01, 5.034006e-01),
        "I_down (0+)": (2.587028e01, 5.762447e-01),
    },
    "lobatto-1801": {
        "I_up (TOA)": (7.848452e-01, 5.032166e-01),
        "I_down (0+)": (2.572425e01, 5.765339e-01),
    },
    "peak-451": {
        "I_up (TOA)": (7.874768e-01, 5.042025e-01),
        "I_down (0+)": (2.577740e01, 5.753914e-01),
    },
    "peak-1801": {
        "I_up (TOA)": (7.845503e-01, 5.036387e-01),
        "I_down (0+)": (2.572139e01, 5.767912e-01),
    },
    "native": {
        "I_up (TOA)": (7.845163e-01, 5.029800e-01),
        "I_down (0+)": (2.571825e01, 5.760760e-01),
    },
}

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# **************************** logging *********************************
logger = logging.getLogger("test_phase_grid_ice")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.ERROR)
    _ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(_ch)
    _fh = logging.FileHandler(LOG_DIR / "phase_grid_ice.log", mode="w")
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(_fh)


# ***************************** helpers ********************************
def _cloud():
    return Cloud(CLOUD, REFF, Z_BOTTOM, Z_TOP, TAU, WAVELENGTH, ssa=1.0)


def grid_of(name):
    """The scattering angles of a GRIDS entry, in degrees."""
    if name == "native":
        return np.unique(_cloud().ds_mix["theta"].values.astype(float))
    kind, n = name.rsplit("-", 1)
    return theta_grid(int(n), kind)


def atm_on(name):
    """The Iwabuchi atmosphere with its phase matrix on a grid."""
    return Atm1D(
        "afglt",
        comp=[_cloud()],
        grid=np.array([Z_TOP, Z_BOTTOM]),
        tco3=0.0,
        no2=False,
        tcwp=0.0,
        tau_r=0.0,
    ).calc(WAVELENGTH, n_theta=grid_of(name))


def run(sg, name, n_photons=NBPHOTONS, n_loop=NBLOOP, seed=SEED):
    """The radiances of REF_18001's keys and their Monte Carlo sigma.

    Two dicts keyed like REF_18001, each value in the order of VZA.
    """
    m = sg.run(
        wavelength=WAVELENGTH,
        atmosphere=atm_on(name),
        ph_deg=180.0,
        th_deg=SZA,
        le={"th_deg": VZA, "phi_deg": np.array([180.0])},
        n_loop=n_loop,
        n_photons=n_photons,
        output_layers=7,
        reflectance=False,
        theta_grid="phase",
        seed=seed,
        xblock=XBLOCK,
        xgrid=XGRID,
        progress=False,
        stdev=True,
    )

    def values(key):
        return np.squeeze(m[key].values).ravel()[: len(VZA)]

    return (
        {k: values(k) for k in REF_18001},
        {k: values(k.replace("I_", "I_stdev_")) for k in REF_18001},
    )


@pytest.fixture(scope="module")
def sg():
    return Smartg(pp=True, double=True)


# ****************************** tests *********************************
@pytest.mark.parametrize("name", GRIDS)
def test_radiance_on_grid(sg, name):
    got, sigma = run(sg, name)
    logger.info("---- %s (%d angles) ----", name, len(grid_of(name)))
    for k, ref in REF_18001.items():
        saved = np.asarray(SAVED[name][k])
        band = NSIGMA * np.sqrt(2.0) * sigma[k]
        logger.info(
            "%s at VZA %s: %s; saved %s; sigma %s; vs 1e10 reference %s %%",
            k,
            " / ".join("%g" % v for v in VZA),
            " / ".join("%.6e" % v for v in got[k]),
            " / ".join("%.6e" % v for v in saved),
            " / ".join("%.2e" % v for v in sigma[k]),
            " / ".join("%+.2f" % (100 * (v / r - 1))
                       for v, r in zip(got[k], ref)),
        )
        assert np.all(np.isfinite(sigma[k])) and np.all(sigma[k] > 0), (
            f"{name} {k}: no Monte Carlo sigma, {sigma[k]}"
        )
        assert np.all(np.abs(got[k] - saved) <= band), (
            f"{name} {k}: {got[k]} vs saved {saved}, "
            f"{(got[k] - saved) / (np.sqrt(2.0) * sigma[k])} sigma of "
            f"the difference, band {NSIGMA} sigma"
        )
