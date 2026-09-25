"""Tests of the OPAC aerosols against reference values.

Each test builds an AerOPAC, computes its profile and compares the
optical thickness and the single scattering albedo at 400 and 700
nm with the reference values.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from smartg import conftest
from smartg.atmosphere import (
    AerOPAC,
    AerUser,
    Atm1D,
    atm_pro_from_aeronet,
    read_aeronet_pfn,
)
from smartg.config import DIR_AUXDATA, DIR_ROOT

# ************************ Global variable(s) **************************
MIXTURES = [
    "continental_clean",
    "continental_average",
    "continental_polluted",
    "urban",
    "desert_spheric",
    "desert",
    "maritime_clean",
    "maritime_polluted",
    "maritime_tropical",
    "arctic",
    "antarctic_spheric",
    "antarctic",
]

SPECIES = [
    "miam",
    "micm",
    "minm",
    "mian",
    "micn",
    "minn",
    "sscm",
    "ssam",
    "inso",
    "soot",
    "suso",
    "waso",
]
# **********************************************************************

# ***************************** logging ********************************
# Create log file
Path(DIR_ROOT / "smartg" / "tests" / "logs").mkdir(parents=True, exist_ok=True)

# Create a named logger
logger = logging.getLogger("test_aerosol")
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
    DIR_ROOT / "smartg" / "tests" / "logs" / "aerosol.log", mode="w"
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


@pytest.mark.parametrize("mix", MIXTURES)
def test_aer_mixtures(request: pytest.FixtureRequest, mix: str) -> None:
    """Check one OPAC mixture at 400 and 700 nm."""
    wavelengths = np.array([400.0, 700.0])
    aer = AerOPAC(
        mix,
        1.0,
        550.0,
        h_free_min=0.0,
        h_stra_max=0,
        h_stra_min=0.0,
        h_free_max=0.0,
    )
    pro = Atm1D("afglt", comp=[aer]).calc(wavelengths)

    ref_fname = (
        DIR_AUXDATA / "aerosols" / "test_ref" / f"atm_afglt_{mix}.nc"
    )
    pro_ref = xr.open_dataset(ref_fname)

    tau_aer_400 = pro["OD_p"][0, -1].values
    tau_aer_ref_400 = pro_ref["OD_p"][0, -1].values
    tau_aer_700 = pro["OD_p"][1, -1].values
    tau_aer_ref_700 = pro_ref["OD_p"][1, -1].values
    ssa_aer_400 = pro["ssa_p_atm"][0, -1].values
    ssa_aer_ref_400 = pro_ref["ssa_p_atm"][0, -1].values
    ssa_aer_700 = pro["ssa_p_atm"][1, -1].values
    ssa_aer_ref_700 = pro_ref["ssa_p_atm"][1, -1].values

    logger.info(
        f"{mix} - 400nm - tau_ref={tau_aer_ref_400:.3f} - "
        + f"tau_calc={tau_aer_400:.3f}"
    )
    logger.info(
        f"{mix} - 700nm - tau_ref={tau_aer_ref_700:.3f} - "
        + f"tau_calc={tau_aer_700:.3f}"
    )
    logger.info(
        f"{mix} - 400nm - ssa_ref={ssa_aer_ref_400:.3f} - "
        + f"ssa_calc={ssa_aer_400:.3f}"
    )
    logger.info(
        f"{mix} - 700nm - ssa_ref={ssa_aer_ref_700:.3f} - "
        + f"ssa_calc={ssa_aer_700:.3f}"
    )

    assert np.isclose(tau_aer_400, tau_aer_ref_400, atol=2e-3), (
        f"Problem with {mix} tau value at 400nm, get {tau_aer_400:.5f} "
        + f"instead of {tau_aer_ref_400:.5f}"
    )
    assert np.isclose(tau_aer_700, tau_aer_ref_700, atol=2e-3), (
        f"Problem with {mix} tau value at 700nm, get {tau_aer_700:.5f} "
        + f"instead of {tau_aer_ref_700:.5f}"
    )

    assert np.isclose(ssa_aer_400, ssa_aer_ref_400, atol=2e-3), (
        f"Problem with {mix} ssa value at 400nm, get {ssa_aer_400:.5f} "
        + f"instead of {ssa_aer_ref_400:.5f}"
    )
    assert np.isclose(ssa_aer_700, ssa_aer_ref_700, atol=2e-3), (
        f"Problem with {mix} ssa value at 700nm, get {ssa_aer_700:.5f} "
        + f"instead of {ssa_aer_ref_700:.5f}"
    )

    plt.close("all")
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle("Phase function at 400nm and z=0km")
    stk_labels = ["F11", "F12", "F33", "F34", "F22", "F44"]
    stk_indices = [0, 1, 2, 3, 4, 5]
    for idx, (label, istk) in enumerate(
        zip(stk_labels, stk_indices, strict=True)
    ):
        ax = axes[idx // 2, idx % 2]
        ax.plot(
            pro_ref["theta_atm"].values,
            pro_ref["phase_atm"].values[0, istk, :],
            "-k",
            label="reference",
        )
        ax.plot(
            pro["theta_atm"].values,
            pro["phase_atm"].values[0, istk, :],
            "--r",
            label="calculated",
        )
        ax.set_title(label)
        ax.set_xlabel(r"$\theta$ (°)")
        ax.grid()
        ax.set_xlim([0, 180])
        if idx == 0:
            ax.set_yscale("log")
            ax.legend()
    fig.tight_layout()
    conftest.savefig(request, bbox_inches="tight")

    plt.close("all")
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle("Diff phase function (ref - calc) at 400nm and z=0km")
    for idx, (label, istk) in enumerate(
        zip(stk_labels, stk_indices, strict=True)
    ):
        ax = axes[idx // 2, idx % 2]
        diff = (
            pro_ref["phase_atm"].values[0, istk, :]
            - pro["phase_atm"].values[0, istk, :]
        )
        ax.plot(pro_ref["theta_atm"].values, diff, "-b")
        ax.set_title(label)
        ax.set_xlabel(r"$\theta$ (°)")
        ax.grid()
        ax.set_xlim([0, 180])
    fig.tight_layout()
    conftest.savefig(request, bbox_inches="tight")

    assert np.all(
        np.isclose(
            pro["phase_atm"].values[:, :4, :],
            pro_ref["phase_atm"].values[:, :4, :],
            atol=1e-5,
            rtol=1e-3,
        )
    ), f"Problem with {mix} phase function"


@pytest.mark.parametrize("spe", SPECIES)
def test_aer_species(request: pytest.FixtureRequest, spe: str) -> None:
    """Check one OPAC species at 400 and 700 nm."""
    wavelengths = np.array([400.0, 700.0])
    aer = AerOPAC(
        spe,
        1.0,
        550.0,
        h_free_min=0.0,
        h_stra_max=0,
        h_stra_min=0.0,
        h_free_max=0.0,
    )
    pro = Atm1D("afglt", comp=[aer]).calc(wavelengths)

    ref_fname = (
        DIR_AUXDATA / "aerosols" / "test_ref" / f"atm_afglt_{spe}.nc"
    )
    pro_ref = xr.open_dataset(ref_fname)

    tau_aer_400 = pro["OD_p"][0, -1].values
    tau_aer_ref_400 = pro_ref["OD_p"][0, -1].values
    tau_aer_700 = pro["OD_p"][1, -1].values
    tau_aer_ref_700 = pro_ref["OD_p"][1, -1].values
    ssa_aer_400 = pro["ssa_p_atm"][0, -1].values
    ssa_aer_ref_400 = pro_ref["ssa_p_atm"][0, -1].values
    ssa_aer_700 = pro["ssa_p_atm"][1, -1].values
    ssa_aer_ref_700 = pro_ref["ssa_p_atm"][1, -1].values

    logger.info(
        f"{spe} - 400nm - tau_ref={tau_aer_ref_400:.3f} - "
        + f"tau_calc={tau_aer_400:.3f}"
    )
    logger.info(
        f"{spe} - 700nm - tau_ref={tau_aer_ref_700:.3f} - "
        + f"tau_calc={tau_aer_700:.3f}"
    )
    logger.info(
        f"{spe} - 400nm - ssa_ref={ssa_aer_ref_400:.3f} - "
        + f"ssa_calc={ssa_aer_400:.3f}"
    )
    logger.info(
        f"{spe} - 700nm - ssa_ref={ssa_aer_ref_700:.3f} - "
        + f"ssa_calc={ssa_aer_700:.3f}"
    )

    assert np.isclose(tau_aer_400, tau_aer_ref_400, atol=2e-3), (
        f"Problem with {spe} tau value at 400nm, get {tau_aer_400:.5f} "
        + f"instead of {tau_aer_ref_400:.5f}"
    )
    assert np.isclose(tau_aer_700, tau_aer_ref_700, atol=2e-3), (
        f"Problem with {spe} tau value at 700nm, get {tau_aer_700:.5f} "
        + f"instead of {tau_aer_ref_700:.5f}"
    )

    assert np.isclose(ssa_aer_400, ssa_aer_ref_400, atol=2e-3), (
        f"Problem with {spe} ssa value at 400nm, get {ssa_aer_400:.5f} "
        + f"instead of {ssa_aer_ref_400:.5f}"
    )
    assert np.isclose(ssa_aer_700, ssa_aer_ref_700, atol=2e-3), (
        f"Problem with {spe} ssa value at 700nm, get {ssa_aer_700:.5f} "
        + f"instead of {ssa_aer_ref_700:.5f}"
    )

    plt.close("all")
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle("Phase function at 400nm and z=0km")
    stk_labels = ["F11", "F12", "F33", "F34", "F22", "F44"]
    stk_indices = [0, 1, 2, 3, 4, 5]
    for idx, (label, istk) in enumerate(
        zip(stk_labels, stk_indices, strict=True)
    ):
        ax = axes[idx // 2, idx % 2]
        ax.plot(
            pro_ref["theta_atm"].values,
            pro_ref["phase_atm"].values[0, istk, :],
            "-k",
            label="reference",
        )
        ax.plot(
            pro["theta_atm"].values,
            pro["phase_atm"].values[0, istk, :],
            "--r",
            label="calculated",
        )
        ax.set_title(label)
        ax.set_xlabel(r"$\theta$ (°)")
        ax.grid()
        ax.set_xlim([0, 180])
        if idx == 0:
            ax.set_yscale("log")
            ax.legend()
    fig.tight_layout()
    conftest.savefig(request, bbox_inches="tight")

    plt.close("all")
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle("Diff phase function (ref - calc) at 400nm and z=0km")
    for idx, (label, istk) in enumerate(
        zip(stk_labels, stk_indices, strict=True)
    ):
        ax = axes[idx // 2, idx % 2]
        diff = (
            pro_ref["phase_atm"].values[0, istk, :]
            - pro["phase_atm"].values[0, istk, :]
        )
        ax.plot(pro_ref["theta_atm"].values, diff, "-b")
        ax.set_title(label)
        ax.set_xlabel(r"$\theta$ (°)")
        ax.grid()
        ax.set_xlim([0, 180])
    fig.tight_layout()
    conftest.savefig(request, bbox_inches="tight")

    assert np.all(
        np.isclose(
            pro["phase_atm"].values[:, 0:4, :],
            pro_ref["phase_atm"].values[:, 0:4, :],
            atol=1e-5,
            rtol=1e-3,
        )
    ), f"Problem with {spe} phase function"


def test_desert_free_stra(request: pytest.FixtureRequest) -> None:
    """Check the desert mixture with free and stratospheric layers."""
    wavelengths = np.array([400.0, 700.0])
    aer = AerOPAC("desert", 1.0, 550.0)
    pro = Atm1D(
        "afglt", comp=[aer], pfgrid=[100.0, 12.0, 6.0, 0.0]
    ).calc(wavelengths)

    ref_fname = (
        DIR_AUXDATA / "aerosols" / "test_ref" / "atm_afglt_desert_free_stra.nc"
    )
    pro_ref = xr.open_dataset(ref_fname)

    tau_aer_400 = pro["OD_p"][0, -1].values
    tau_aer_ref_400 = pro_ref["OD_p"][0, -1].values
    tau_aer_700 = pro["OD_p"][1, -1].values
    tau_aer_ref_700 = pro_ref["OD_p"][1, -1].values
    ssa_aer_400 = pro["ssa_p_atm"][0, -1].values
    ssa_aer_ref_400 = pro_ref["ssa_p_atm"][0, -1].values
    ssa_aer_700 = pro["ssa_p_atm"][1, -1].values
    ssa_aer_ref_700 = pro_ref["ssa_p_atm"][1, -1].values

    logger.info(
        f"desert free stra - 400nm - tau_ref={tau_aer_ref_400:.3f} - "
        + f"tau_calc={tau_aer_400:.3f}"
    )
    logger.info(
        f"desert free stra - 700nm - tau_ref={tau_aer_ref_700:.3f} - "
        + f"tau_calc={tau_aer_700:.3f}"
    )
    logger.info(
        f"desert free stra - 400nm - ssa_ref={ssa_aer_ref_400:.3f} - "
        + f"ssa_calc={ssa_aer_400:.3f}"
    )
    logger.info(
        f"desert free stra - 700nm - ssa_ref={ssa_aer_ref_700:.3f} - "
        + f"ssa_calc={ssa_aer_700:.3f}"
    )

    assert np.isclose(tau_aer_400, tau_aer_ref_400, atol=2e-3), (
        "Problem with desert free stra tau value at 400nm, "
        + f"get {tau_aer_400:.5f} instead of {tau_aer_ref_400:.5f}"
    )
    assert np.isclose(tau_aer_700, tau_aer_ref_700, atol=2e-3), (
        "Problem with desert free stra tau value at 700nm, "
        + f"get {tau_aer_700:.5f} instead of {tau_aer_ref_700:.5f}"
    )

    assert np.isclose(ssa_aer_400, ssa_aer_ref_400, atol=2e-3), (
        "Problem with desert free stra ssa value at 400nm, "
        + f"get {ssa_aer_400:.5f} instead of {ssa_aer_ref_400:.5f}"
    )
    assert np.isclose(ssa_aer_700, ssa_aer_ref_700, atol=2e-3), (
        "Problem with desert free stra ssa value at 700nm, "
        + f"get {ssa_aer_700:.5f} instead of {ssa_aer_ref_700:.5f}"
    )

    iph = pro["iphase_atm"][0, -1].values
    iph_ref = pro_ref["iphase_atm"][0, -1].values
    plt.close("all")
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle("Phase function at 400nm and z=0km")
    stk_labels = ["F11", "F12", "F33", "F34", "F22", "F44"]
    stk_indices = [0, 1, 2, 3, 4, 5]
    for idx, (label, istk) in enumerate(
        zip(stk_labels, stk_indices, strict=True)
    ):
        ax = axes[idx // 2, idx % 2]
        ax.plot(
            pro_ref["theta_atm"].values,
            pro_ref["phase_atm"].values[iph_ref, istk, :],
            "-k",
            label="reference",
        )
        ax.plot(
            pro["theta_atm"].values,
            pro["phase_atm"].values[iph, istk, :],
            "--r",
            label="calculated",
        )
        ax.set_title(label)
        ax.set_xlabel(r"$\theta$ (°)")
        ax.grid()
        ax.set_xlim([0, 180])
        if idx == 0:
            ax.set_yscale("log")
            ax.legend()
    fig.tight_layout()
    conftest.savefig(request, bbox_inches="tight")

    plt.close("all")
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle("Diff phase function (ref - calc) at 400nm and z=0km")
    for idx, (label, istk) in enumerate(
        zip(stk_labels, stk_indices, strict=True)
    ):
        ax = axes[idx // 2, idx % 2]
        diff = (
            pro_ref["phase_atm"].values[iph_ref, istk, :]
            - pro["phase_atm"].values[iph, istk, :]
        )
        ax.plot(pro_ref["theta_atm"].values, diff, "-b")
        ax.set_title(label)
        ax.set_xlabel(r"$\theta$ (°)")
        ax.grid()
        ax.set_xlim([0, 180])
    fig.tight_layout()
    conftest.savefig(request, bbox_inches="tight")

    assert np.all(
        np.isclose(
            pro["phase_atm"].values[:, 0:4, :],
            pro_ref["phase_atm"].values[:, 0:4, :],
            atol=1e-5,
            rtol=1e-3,
        )
    ), "Problem with desert free stra phase function"


def test_dd_cc_mixture(request: pytest.FixtureRequest) -> None:
    """Check a desert and a continental clean aerosol together."""
    wavelengths = np.array([400.0, 700.0])
    pfgrid = [100.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    aer1 = AerOPAC(
        "desert",
        1.0,
        550.0,
        h_free_min=0.0,
        h_stra_max=0,
        h_stra_min=0.0,
        h_free_max=0.0,
    )
    aer2 = AerOPAC(
        "continental_clean",
        1.0,
        550.0,
        h_free_min=0.0,
        h_stra_max=0,
        h_stra_min=0.0,
        h_free_max=0.0,
    )
    pro = Atm1D("afglt", comp=[aer1, aer2], pfgrid=pfgrid).calc(wavelengths)

    ref_fname = (
        DIR_AUXDATA
        / "aerosols"
        / "test_ref"
        / "atm_afglt_desert_cont_clean_mix.nc"
    )
    pro_ref = xr.open_dataset(ref_fname)

    tau_aer_400 = pro["OD_p"][0, -1].values
    tau_aer_ref_400 = pro_ref["OD_p"][0, -1].values
    tau_aer_700 = pro["OD_p"][1, -1].values
    tau_aer_ref_700 = pro_ref["OD_p"][1, -1].values
    ssa_aer_400 = pro["ssa_p_atm"][0, -1].values
    ssa_aer_ref_400 = pro_ref["ssa_p_atm"][0, -1].values
    ssa_aer_700 = pro["ssa_p_atm"][1, -1].values
    ssa_aer_ref_700 = pro_ref["ssa_p_atm"][1, -1].values

    logger.info(
        f"dd + cc - 400nm - tau_ref={tau_aer_ref_400:.3f} - "
        + f"tau_calc={tau_aer_400:.3f}"
    )
    logger.info(
        f"dd + cc - 700nm - tau_ref={tau_aer_ref_700:.3f} - "
        + f"tau_calc={tau_aer_700:.3f}"
    )
    logger.info(
        f"dd + cc - 400nm - ssa_ref={ssa_aer_ref_400:.3f} - "
        + f"ssa_calc={ssa_aer_400:.3f}"
    )
    logger.info(
        f"dd + cc - 700nm - ssa_ref={ssa_aer_ref_700:.3f} - "
        + f"ssa_calc={ssa_aer_700:.3f}"
    )

    assert np.isclose(tau_aer_400, tau_aer_ref_400, atol=2e-3), (
        "Problem with dd + cc tau value at 400nm, "
        + f"get {tau_aer_400:.5f} instead of {tau_aer_ref_400:.5f}"
    )
    assert np.isclose(tau_aer_700, tau_aer_ref_700, atol=2e-3), (
        "Problem with dd + cc tau value at 700nm, "
        + f"get {tau_aer_700:.5f} instead of {tau_aer_ref_700:.5f}"
    )

    assert np.isclose(ssa_aer_400, ssa_aer_ref_400, atol=2e-3), (
        "Problem with dd + cc ssa value at 400nm, "
        + f"get {ssa_aer_400:.5f} instead of {ssa_aer_ref_400:.5f}"
    )
    assert np.isclose(ssa_aer_700, ssa_aer_ref_700, atol=2e-3), (
        "Problem with dd + cc ssa value at 700nm, "
        + f"get {ssa_aer_700:.5f} instead of {ssa_aer_ref_700:.5f}"
    )

    iph = pro["iphase_atm"][0, -1].values
    iph_ref = pro_ref["iphase_atm"][0, -1].values
    plt.close("all")
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle("Phase function at 400nm and z=0km")
    stk_labels = ["F11", "F12", "F33", "F34", "F22", "F44"]
    stk_indices = [0, 1, 2, 3, 4, 5]
    for idx, (label, istk) in enumerate(
        zip(stk_labels, stk_indices, strict=True)
    ):
        ax = axes[idx // 2, idx % 2]
        ax.plot(
            pro_ref["theta_atm"].values,
            pro_ref["phase_atm"].values[iph_ref, istk, :],
            "-k",
            label="reference",
        )
        ax.plot(
            pro["theta_atm"].values,
            pro["phase_atm"].values[iph, istk, :],
            "--r",
            label="calculated",
        )
        ax.set_title(label)
        ax.set_xlabel(r"$\theta$ (°)")
        ax.grid()
        ax.set_xlim([0, 180])
        if idx == 0:
            ax.set_yscale("log")
            ax.legend()
    fig.tight_layout()
    conftest.savefig(request, bbox_inches="tight")

    plt.close("all")
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle("Diff phase function (ref - calc) at 400nm and z=0km")
    for idx, (label, istk) in enumerate(
        zip(stk_labels, stk_indices, strict=True)
    ):
        ax = axes[idx // 2, idx % 2]
        diff = (
            pro_ref["phase_atm"].values[iph_ref, istk, :]
            - pro["phase_atm"].values[iph, istk, :]
        )
        ax.plot(pro_ref["theta_atm"].values, diff, "-b")
        ax.set_title(label)
        ax.set_xlabel(r"$\theta$ (°)")
        ax.grid()
        ax.set_xlim([0, 180])
    fig.tight_layout()
    conftest.savefig(request, bbox_inches="tight")

    assert np.all(
        np.isclose(
            pro["phase_atm"].values[:, 0:4, :],
            pro_ref["phase_atm"].values[:, 0:4, :],
            atol=1e-5,
            rtol=1e-3,
        )
    ), "Problem with dd + cc phase function"


def test_desert_one_wavelength(request: pytest.FixtureRequest) -> None:
    """Check the desert mixture at a single wavelength."""
    wavelength = 400.0
    aer = AerOPAC(
        "desert",
        1.0,
        550.0,
        h_free_min=0.0,
        h_stra_max=0,
        h_stra_min=0.0,
        h_free_max=0.0,
    )
    pro = Atm1D("afglt", comp=[aer]).calc(wavelength)

    ref_fname = DIR_AUXDATA / "aerosols" / "test_ref" / "atm_afglt_desert.nc"
    pro_ref = xr.open_dataset(ref_fname)

    tau_aer_400 = pro["OD_p"][0, -1].values
    tau_aer_ref_400 = pro_ref["OD_p"][0, -1].values
    ssa_aer_400 = pro["ssa_p_atm"][0, -1].values
    ssa_aer_ref_400 = pro_ref["ssa_p_atm"][0, -1].values

    logger.info(
        f"desert one wavelength - 400nm - tau_ref={tau_aer_ref_400:.3f} - "
        + f"tau_calc={tau_aer_400:.3f}"
    )
    logger.info(
        f"desert one wavelength - 400nm - ssa_ref={ssa_aer_ref_400:.3f} - "
        + f"ssa_calc={ssa_aer_400:.3f}"
    )

    assert np.isclose(tau_aer_400, tau_aer_ref_400, atol=2e-3), (
        "Problem with desert one wavelength tau value at 400nm, "
        + f"get {tau_aer_400:.5f} instead of {tau_aer_ref_400:.5f}"
    )

    assert np.isclose(ssa_aer_400, ssa_aer_ref_400, atol=2e-3), (
        "Problem with desert one wavelength ssa value at 400nm, "
        + f"get {ssa_aer_400:.5f} instead of {ssa_aer_ref_400:.5f}"
    )

    iph = pro["iphase_atm"][0, -1].values
    iph_ref = pro_ref["iphase_atm"][0, -1].values
    plt.close("all")
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle("Phase function at 400nm and z=0km")
    stk_labels = ["F11", "F12", "F33", "F34", "F22", "F44"]
    stk_indices = [0, 1, 2, 3, 4, 5]
    for idx, (label, istk) in enumerate(
        zip(stk_labels, stk_indices, strict=True)
    ):
        ax = axes[idx // 2, idx % 2]
        ax.plot(
            pro_ref["theta_atm"].values,
            pro_ref["phase_atm"].values[iph_ref, istk, :],
            "-k",
            label="reference",
        )
        ax.plot(
            pro["theta_atm"].values,
            pro["phase_atm"].values[iph, istk, :],
            "--r",
            label="calculated",
        )
        ax.set_title(label)
        ax.set_xlabel(r"$\theta$ (°)")
        ax.grid()
        ax.set_xlim([0, 180])
        if idx == 0:
            ax.set_yscale("log")
            ax.legend()
    fig.tight_layout()
    conftest.savefig(request, bbox_inches="tight")

    plt.close("all")
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle("Diff phase function (ref - calc) at 400nm and z=0km")
    for idx, (label, istk) in enumerate(
        zip(stk_labels, stk_indices, strict=True)
    ):
        ax = axes[idx // 2, idx % 2]
        diff = (
            pro_ref["phase_atm"].values[iph_ref, istk, :]
            - pro["phase_atm"].values[iph, istk, :]
        )
        ax.plot(pro_ref["theta_atm"].values, diff, "-b")
        ax.set_title(label)
        ax.set_xlabel(r"$\theta$ (°)")
        ax.grid()
        ax.set_xlim([0, 180])
    fig.tight_layout()
    conftest.savefig(request, bbox_inches="tight")

    ipha = pro["iphase_atm"][0, :].values
    ipha_ref = pro_ref["iphase_atm"][0, :].values
    assert np.all(
        np.isclose(
            pro["phase_atm"].values[ipha, 0:4, :],
            pro_ref["phase_atm"].values[ipha_ref, 0:4, :],
            atol=1e-5,
            rtol=1e-3,
        )
    ), "Problem with desert one wavelength phase function"


# ************************* component inputs ***************************
Z_LEVELS = np.linspace(100.0, 0.0, 101)
WAVELENGTHS = np.array([440.0, 550.0, 1020.0])


@pytest.mark.parametrize(
    "tau_ref",
    [[0.1], (0.1,), np.array([0.1]), np.array(0.1)],
    ids=["list", "tuple", "array", "0-d"],
)
def test_tau_ref_holding_one_value(tau_ref: object) -> None:
    """An array holding one optical thickness is taken as a number.

    It was silently ignored, the component keeping the optical
    thickness of the OPAC number densities.
    """
    ref, _ = AerOPAC("continental_clean", 0.1, 550.0).dtau_ssa(
        WAVELENGTHS, Z_LEVELS, 50.0
    )
    dtau, _ = AerOPAC("continental_clean", tau_ref, 550.0).dtau_ssa(
        WAVELENGTHS, Z_LEVELS, 50.0
    )
    np.testing.assert_array_equal(dtau, ref)
    assert np.isclose(dtau[1].sum(), 0.1, rtol=1e-5)


def test_tau_ref_of_several_values_refused() -> None:
    """An array of several optical thicknesses has no meaning."""
    with pytest.raises(TypeError, match="tau_ref"):
        AerOPAC("continental_clean", np.array([0.1, 0.2]), 550.0)


def _henyey_greenstein(theta: np.ndarray, g: float = 0.7) -> np.ndarray:
    """Return the Henyey-Greenstein phase function, 2 over mu."""
    mu = np.cos(np.radians(theta))
    return (1.0 - g**2) / (1.0 + g**2 - 2.0 * g * mu) ** 1.5


def test_read_aeronet_pfn_angles_increase(tmp_path: Path) -> None:
    """The AERONET phase function comes back on increasing angles.

    The files list them from 180 to 0 degrees.
    """
    angles = [180.0, 90.0, 0.0]
    lines = ["preamble\n"] * 6
    lines.append(
        "Site,Date(dd:mm:yyyy),Day_of_Year(Fraction),"
        + ",".join(f"{a:.6f}[440nm]" for a in angles)
        + ",Phase_Function_Mode\n"
    )
    lines.append("Site,10:04:2020,101.5,0.2,0.3,25.0,Total\n")
    fname = tmp_path / "site.pfn"
    fname.write_text("".join(lines))
    pfn = read_aeronet_pfn(fname, 2020)
    np.testing.assert_array_equal(pfn["theta_atm"].values, [0.0, 90.0, 180.0])
    np.testing.assert_array_equal(pfn.values[0, 0], [25.0, 0.3, 0.2])


def test_aeronet_phase_matrix_does_not_polarize() -> None:
    """A scalar AERONET phase function scatters without polarizing.

    Its matrix is F11 = F22 = F33 = F44 = pfn and F21 = F34 = 0, in the
    right angular order even when the angles decrease, as those of the
    AERONET files do.
    """
    days = np.array([101.0, 102.0])
    wavelengths = np.array([440.0, 675.0])
    theta = np.linspace(180.0, 0.0, 181)
    coords = {"Day_of_Year(Fraction)": days, "wavelength": wavelengths}
    aod = xr.DataArray(np.full((2, 2), 0.2), coords=coords)
    ssa = xr.DataArray(np.full((2, 2), 0.9), coords=coords)
    pfn = xr.DataArray(
        np.broadcast_to(_henyey_greenstein(theta), (2, 2, theta.size)),
        coords={**coords, "theta_atm": theta},
    )
    pro = atm_pro_from_aeronet(
        "2020-04-10", "12:00:00", aod, ssa, pfn, [550.0]
    )
    pha = pro["phase_atm"].values[0]
    theta_atm = pro["theta_atm"].values
    f11 = pha[0]
    np.testing.assert_allclose(
        f11[[0, -1]], _henyey_greenstein(theta_atm[[0, -1]]), rtol=1e-5
    )
    assert f11[0] > 100.0 * f11[-1]
    for term in (2, 4, 5):  # F33, F22, F44
        np.testing.assert_array_equal(pha[term], f11)
    for term in (1, 3):  # F21, F34
        np.testing.assert_array_equal(pha[term], 0.0)


def test_aer_user_sorts_its_angles() -> None:
    """AerUser sorts decreasing angles along with the phase matrix."""
    theta = np.linspace(0.0, 180.0, 181)
    pfn = _henyey_greenstein(theta)
    zeros = np.zeros_like(pfn)
    phase = np.stack([pfn, zeros, pfn, zeros])[None, None]
    args = (np.full((1, 1), 0.2), np.full((1, 1), 0.9))
    hum, wavelength = np.array([0.0]), np.array([550.0])
    ref = AerUser(*args, phase, hum, wavelength, theta)
    rev = AerUser(*args, phase[..., ::-1], hum, wavelength, theta[::-1])
    rh = np.zeros(2)
    np.testing.assert_array_equal(
        rev.phase(wavelength, np.array([100.0, 0.0]), rh).values,
        ref.phase(wavelength, np.array([100.0, 0.0]), rh).values,
    )
    with pytest.raises(ValueError, match="distinct"):
        AerUser(*args, phase, hum, wavelength, np.full(181, 90.0))


GRID = np.array([100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.0])


def test_forced_ssa_over_wavelength_and_altitude() -> None:
    """A forced ssa DataArray over (wavelength, z) is interpolated.

    Onto every grid Atm1D evaluates the component on: the profile, the
    pfgrid of the phase matrices and their union.
    """
    wavelength = np.array([500.0, 700.0])
    values = 0.8 + 0.01 * GRID[None, :] / 100.0 + 0.1 * np.array(
        [[0.0], [1.0]]
    )
    ssa = xr.DataArray(
        values, coords={"wavelength": wavelength, "z": GRID}
    )
    pro = Atm1D(
        "afglt",
        comp=[AerOPAC("continental_clean", 0.1, 550.0, ssa=ssa)],
        grid=GRID,
        pfgrid=[100.0, 1.5, 0.0],
    ).calc(np.array([500.0, 600.0]))
    expected = np.stack([values[0], 0.5 * (values[0] + values[1])])
    scatters = np.diff(pro["OD_p"].values, axis=1, prepend=0.0) > 0.0
    np.testing.assert_allclose(
        pro["ssa_p_atm"].values[scatters], expected[scatters], rtol=1e-6
    )


def test_forced_ssa_arrays_off_the_grid_refused() -> None:
    """A forced ssa array not matching the grid gives a clear error.

    A 1-D array holds one value per wavelength of the calculation, and
    a 2-D one per wavelength and level of the profile grid, neither of
    which the phase matrices are mixed on here.
    """
    aer = AerOPAC("continental_clean", 0.1, 550.0, ssa=[0.9, 0.8])
    atm = Atm1D("afglt", comp=[aer], wavelength_phase=[600.0])
    with pytest.raises(ValueError, match="DataArray over wavelength"):
        atm.calc([500.0, 700.0])
    aer = AerOPAC(
        "continental_clean", 0.1, 550.0, ssa=np.full((2, GRID.size), 0.9)
    )
    atm = Atm1D("afglt", comp=[aer], grid=GRID)
    atm.calc([500.0, 700.0], phase=False)
    with pytest.raises(ValueError, match="wavelength and the altitude"):
        atm.calc([500.0, 700.0])


def test_opac_file_without_default_heights() -> None:
    """mineral_transported needs its heights, then it works.

    Its file gives 'None' to all its heights and scale heights, which
    made the constructor fail on float('None').
    """
    with pytest.raises(ValueError, match="h_mix_min, h_mix_max and z_mix"):
        AerOPAC("mineral_transported", 0.1, 550.0)
    with pytest.raises(ValueError, match="z_mix"):
        AerOPAC(
            "mineral_transported", 0.1, 550.0, h_mix_min=1.5, h_mix_max=3.5
        )
    aer = AerOPAC(
        "mineral_transported", 0.1, 550.0,
        h_mix_min=1.5, h_mix_max=3.5, z_mix=1.0,
    )
    assert aer.h_min == [1.5] and aer.h_max == [3.5]
    dtau, _ = aer.dtau_ssa(np.array([550.0]), Z_LEVELS, 50.0)
    assert np.isclose(dtau.sum(), 0.1, rtol=1e-5)
    np.testing.assert_array_equal(dtau[0, (Z_LEVELS > 4.0)], 0.0)


def test_opac_list_holds_mixtures_only() -> None:
    """AerOPAC.list leaves out the species and the layer files."""
    names = AerOPAC.list()
    assert "desert" in names and "mineral_transported" in names
    for name in ("free_troposphere", "stratosphere", "waso", "inso"):
        assert name not in names
