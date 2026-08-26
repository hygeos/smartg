#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Must be set before importing JAX so only the CPU backend
# is initialised. This prevents JAX from creating a CUDA context
# that conflicts with PyCUDA's at process exit
# ("context::pop failed: invalid device context").
os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import pytest
import logging
from gc import collect

jax = pytest.importorskip(
    "jax", reason="cannot test this since the jax package is not installed."
)
from smartg.histories import get_histories, BigSum, Si, Si2
import numpy as np
import matplotlib.pyplot as plt
from smartg.smartg import Smartg
from smartg.surface import LambSurface
from smartg.albedo import AlbedoCst
from smartg.atmosphere import Atm1D, AerOPAC, od2k
from smartg.diff import diff1
from luts import LUT
from smartg.view import mdesc
from smartg import conftest
from pathlib import Path
from smartg.config import DIR_AUXDATA
from smartg.xarray import drop_axes


# ***************************** logging ********************************
ROOTPATH = Path(__file__).resolve().parent.parent.parent
Path(ROOTPATH / "smartg" / "tests" / "logs").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("test_smartg_jax")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(
    ROOTPATH / "smartg" / "tests" / "logs" / "smartg_jax.log", mode="w"
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# **********************************************************************


# Clean up JAX memory and stale PyCUDA atexit handlers after each test
@pytest.fixture(scope="function", autouse=True)
def cleanup_after_each_test():
    yield
    jax.clear_caches()
    collect()
    # Each Smartg() import of pycuda.autoinit registers a new
    # _finish_up atexit callback. After sg.clear_context() the
    # context is already gone, so the callback
    # raises "context::pop failed". Unregister it here while we
    # still have control, so the error is never printed.
    try:
        import atexit
        import pycuda.autoinit as _pai

        atexit.unregister(_pai._finish_up)
    except Exception:
        pass


@pytest.mark.parametrize("n_wl_abs", [301])
@pytest.mark.parametrize("wmax", [350.0])
@pytest.mark.parametrize("wmin", [320.0])
def test_smartg_jax2(
    n_wl_abs, wmin, wmax, request, nb_photons=5e4, max_hist=1e6
):
    alb_snow = AlbedoCst(0.6)
    alb_hist = AlbedoCst(1.0)
    wl_sca = np.linspace(wmin, wmax, num=11)
    wl_abs = np.linspace(wmin, wmax, num=n_wl_abs)
    alb = alb_snow.get(wl_abs)
    lez = {"th_deg": np.array([0.0]), "phi_deg": np.array([0.0]), "zip": False}

    for aod, fmt1 in zip(
        np.linspace(0.1, 0.5, num=2), ["-m", "-c"], strict=True
    ):
        level = 0  # 1: BOA downward reflectance, 0 : TOA
        atm = Atm1D(
            "afglms",
            comp=[AerOPAC("urban", aod, 550.0)],
            grid=np.linspace(50.0, 0.0, num=40),
        )
        sigma = od2k(atm.calc(wl_abs), "OD_abs_atm")[:, 1:]
        sg = Smartg(alis=True, alt_pp=True)
        m = (
            sg.run(
                seed=0,
                th_v_deg=45.0,
                wl=wl_sca,
                surf=LambSurface(alb_hist),
                le=lez,
                beer=0,
                atm=atm.calc(wl_sca),
                alis_options={
                    "nlow": wl_sca.size,
                    "hist": True,
                    "max_hist": np.int64(max_hist),
                },
                nb_photons=nb_photons,
                nb_loop=nb_photons,
                n_f=1e3,
            )
        )
        m = drop_axes(m, "Zenith angles", "Azimuth angles")
        m0 = (
            sg.run(
                seed=0,
                th_v_deg=45.0,
                wl=wl_abs,
                surf=LambSurface(alb_snow),
                le=lez,
                beer=0,
                atm=atm.calc(wl_abs),
                alis_options={"nlow": wl_sca.size, "hist": False},
                nb_photons=nb_photons,
                n_f=1e3,
            )
        )
        m0 = drop_axes(m0, "Zenith angles", "Azimuth angles")
        sg.clear_context()

        with jax.default_device(
            jax.devices("cpu")[0]
        ):  # run on CPU to avoid slow GPU XLA compilation
            N, S, D, w, _, nref, _, _, _, _, _ = get_histories(
                m, LEVEL=level, verbose=False
            )
            stk_i = (
                np.array(
                    BigSum(Si, only_I=True)(
                        wl_abs, sigma, alb, S[:, 0], w, D, nref, wl_sca
                    ).sum(axis=0)
                )
                / N
            )
            stk_i2 = (
                np.array(
                    BigSum(Si2, only_I=True)(
                        wl_abs, sigma, alb, S[:, 0], w, D, nref, wl_sca
                    ).sum(axis=0)
                )
                / N
            )
        std = np.sqrt((stk_i2 - stk_i**2) / N)
        upper = stk_i + 1.95 * std
        lower = stk_i - 1.95 * std
        p = plt.plot(
            wl_abs,
            stk_i,
            fmt1,
            label="AOD@550: {:.1f}; NBPH={:.0e}; NBHIST={:.0e}".format(
                aod, np.int64(nb_photons), np.int64(max_hist)
            ),
        )
        col = p[0].get_color()
        print(stk_i2, std)
        plt.fill_between(
            wl_abs,
            lower,
            upper,
            facecolor=col,
            edgecolor=col,
            alpha=0.4,
            label="95 percent confidence",
        )
        plt.plot(
            wl_abs,
            m0["I_up (TOA)"][:],
            marker="+",
            ls="",
            label="AOD@550: {:.1f}; NBPH={:.0e}; NO HIST".format(
                aod, np.int64(nb_photons)
            ),
            color=p[0].get_color(),
        )
        plt.xlabel(r"$\lambda (nm)$")
        plt.ylabel("TOA reflectance")
        # plt.ylim(0.,0.7)
        plt.title("Urban aerosols, SZA=45°, nadir viewing, snow albedo")
        plt.grid()
    plt.legend()
    conftest.savefig(request)


def test_validation_artdeco(request, nb_photons=5e5, valpath=DIR_AUXDATA):
    """
    Validation of SMART-G with ARTDECO validation data
    """
    typ = "desert"  # tau=0.25
    ####################""""""
    fgas = Path(valpath) / "validation" / f"cTauGas_ray_{typ}_O2.dat"
    gas_valid = diff1(np.loadtxt(fgas, skiprows=7)[:, 1:].T, axis=1)
    z_valid = np.loadtxt(fgas, skiprows=7)[:, 0]
    w_valid = np.array(open(fgas).readlines()[5].split()).astype(float)
    fray = Path(valpath) / "validation" / f"cTauRay_ray_{typ}_O2.dat"
    ray_valid = diff1(np.loadtxt(fray, skiprows=7)[:, 1:].T, axis=1)
    faer_abs = Path(valpath) / "validation" / f"cTauAbs_ptcle_ray_{typ}_O2.dat"
    aer_abs_valid = diff1(np.loadtxt(faer_abs, skiprows=7)[:, 1:].T, axis=1)
    faer_sca = Path(valpath) / "validation" / f"cTauSca_ptcle_ray_{typ}_O2.dat"
    aer_sca_valid = diff1(np.loadtxt(faer_sca, skiprows=7)[:, 1:].T, axis=1)
    # aerosols phase matrix import
    faer_phase = Path(valpath) / "validation" / f"phasemat_ray_{typ}_O2.dat"
    n = int(np.genfromtxt(faer_phase, usecols=range(1), max_rows=1, dtype=int))
    pfwav = []
    npf = 3
    data = np.zeros((npf, 1, n, 5), dtype=np.float32)
    for k in range(npf):
        pfwav.append(
            np.genfromtxt(
                faer_phase,
                usecols=range(1),
                skip_header=(1 + (2 + n) * k),
                max_rows=1,
            )
        )
        # pizero=np.genfromtxt(faer_phase, usecols=range(1),
        #                     skip_header=(1+(2+n)*k+1), max_rows=1)
        data[k, 0, :, :] = np.genfromtxt(
            faer_phase,
            usecols=range(5),
            skip_header=(1 + (2 + n) * k + 2),
            max_rows=n,
        )
    data = data.swapaxes(2, 3)

    # From iparper to standard phase convention
    pha_data = data[:, :, 1:, :].copy()
    pha_data[:, :, 0, :] = (data[:, :, 1, :] + data[:, :, 2, :]) * 0.5
    pha_data[:, :, 1, :] = (data[:, :, 1, :] - data[:, :, 2, :]) * 0.5

    phase_valid = LUT(
        pha_data,
        names=["wav_phase", "z_phase", "stk", "theta_atm"],
        axes=[pfwav, [0], None, data[0, 0, 0, :]],
    )
    data_valid = np.loadtxt(
        Path(valpath) / "validation" / f"artdeco_lbl_nstr_32_ray_{typ}_O2.dat"
    )
    aer_ext_valid = aer_sca_valid + aer_abs_valid
    aer_ssa_valid = aer_sca_valid / aer_ext_valid
    aer_ssa_valid[aer_ext_valid == 0] = 1.0
    comp = [AerOPAC("desert", 0.5, 550.0, phase=phase_valid)]
    atm_valid = Atm1D(
        "afglmw",
        grid=z_valid,
        tco3=0.0,
        no2=False,
        pfwav=pfwav,
        comp=comp,
        prof_ray=ray_valid,
        prof_aer=(aer_ext_valid, aer_ssa_valid),
        prof_abs=gas_valid,
    )
    sigma_valid = od2k(atm_valid.calc(w_valid), "OD_abs_atm")[:, 1:]
    ###############

    le = {
        "th_deg": np.array([20.0]),
        "phi_deg": np.array([180.0]),
        "zip": False,
    }
    nlow = 3
    wl_lr = np.linspace(w_valid.min(), w_valid.max(), num=nlow)

    sg = Smartg(alis=True, alt_pp=True)
    m1 = (
        sg.run(
            seed=0,
            th_v_deg=30.0,
            wl=w_valid,
            surf=None,
            le=le,
            beer=0,
            atm=atm_valid.calc(w_valid),
            depo=0.0,
            alis_options={"nlow": nlow, "hist": False},
            nb_photons=nb_photons,
            nb_loop=nb_photons,
            n_f=1e3,
        )
    )
    m1 = drop_axes(m1, "Zenith angles", "Azimuth angles")
    m2 = (
        sg.run(
            seed=0,
            th_v_deg=30.0,
            wl=w_valid,
            surf=None,
            le=le,
            beer=0,
            atm=atm_valid.calc(w_valid),
            depo=0.0,
            alis_options={
                "nlow": nlow,
                "hist": True,
                "max_hist": np.int64(1e7),
            },
            nb_photons=nb_photons,
            nb_loop=nb_photons,
            n_f=1e3,
        )
    )
    m2 = drop_axes(m2, "Zenith angles", "Azimuth angles")
    sg.clear_context()
    print("GPU time no hist: %.4f" % float(m1.attrs["kernel time (s)"]), "s")
    print("GPU time hist: %.4f" % float(m2.attrs["kernel time (s)"]), "s")

    with jax.default_device(
        jax.devices("cpu")[0]
    ):  # run on CPU to avoid slow GPU XLA compilation
        N, S, D, w, _, nref, _, _, _, _, _ = get_histories(
            m2, LEVEL=0, verbose=True
        )
        stk_i = (
            np.array(
                BigSum(Si, only_I=True)(
                    w_valid,
                    sigma_valid,
                    np.zeros_like(w_valid),
                    S[:, 0],
                    w,
                    D,
                    nref,
                    wl_lr,
                ).sum(axis=0)
            )
            / N
        )

    #####################
    i_valid = data_valid[:, 1]

    # Mask for "significant" reference values where relative error is meaningful
    sig = np.abs(i_valid) > 1e-3

    # --- no-hist ---
    abs_err_no_hist = np.abs(m1["I_up (TOA)"][:] - i_valid)
    rel_err_no_hist = abs_err_no_hist[sig] / np.abs(i_valid[sig])
    logger.info(
        f"no-hist vs ARTDECO (|ref|>1e-3) - "
        f"max rel err = {np.max(rel_err_no_hist) * 100:.6f}% - "
        f"max abs err = {np.max(abs_err_no_hist):.6e}"
    )
    assert np.all(rel_err_no_hist < 0.02), (
        f"SMART-G no-hist exceeds 2% relative error vs ARTDECO reference "
        f"(|ref|>1e-3): max rel err = {rel_err_no_hist.max():.4%}"
    )

    # --- hist+jax ---
    abs_err_hist = np.abs(stk_i - i_valid)
    rel_err_hist = abs_err_hist[sig] / np.abs(i_valid[sig])
    logger.info(
        f"hist+jax vs ARTDECO (|ref|>1e-3) - "
        f"max rel err = {np.max(rel_err_hist) * 100:.6f}% - "
        f"max abs err = {np.max(abs_err_hist):.6e}"
    )
    assert np.all(rel_err_hist < 0.02), (
        f"SMART-G hist+jax exceeds 2% relative error vs ARTDECO reference "
        f"(|ref|>1e-3): max rel err = {rel_err_hist.max():.4%}"
    )

    plt.figure(figsize=(12, 4))
    plt.plot(w_valid, i_valid, "r", label="Doubling Adding: 32 streams")
    plt.plot(w_valid, m1["I_up (TOA)"], "c", label="SMART-G no hist.")
    plt.plot(w_valid, stk_i, "b", label="SMART-G, hist. with jax")
    plt.legend()
    plt.ylabel(mdesc("I_up (TOA)"))
    plt.grid()
    conftest.savefig(request)
    ##
    plt.figure(figsize=(12, 4))
    df = stk_i - i_valid
    dff = df / i_valid * 100
    plt.plot(w_valid, dff, "b-")
    df1 = m1["I_up (TOA)"][:] - i_valid
    dff1 = df1 / i_valid * 100
    plt.plot(w_valid, dff1, "c-")
    plt.ylim(-2, 2)
    plt.grid()
    plt.ylabel(
        mdesc("I_up (TOA)") + " relative difference to Doubling Adding (%)"
    )
    conftest.savefig(request)
