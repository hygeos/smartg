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
from gc import collect

try:
    from smartg.histories import get_histories, BigSum, Si, Si2
    import jax

    SKIP = False
except ModuleNotFoundError:
    SKIP = True
import numpy as np
import matplotlib.pyplot as plt
from smartg.smartg import Smartg
from smartg.smartg import LambSurface, AlbedoCst
from smartg.atmosphere import Atm1D, AerOPAC, od2k, diff1
from smartg.albedo import AlbedoCst
from luts import LUT
from smartg.tools.smartg_view import mdesc
from smartg import conftest
from pathlib import Path
from smartg.config import DIR_AUXDATA


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


@pytest.mark.parametrize("N_WL_ABS", [301])
@pytest.mark.parametrize("WMAX", [350.0])
@pytest.mark.parametrize("WMIN", [320.0])
@pytest.mark.skipif(
    SKIP, reason="cannot test this since the jax package is not installed."
)
def test_smartg_jax2(
    N_WL_ABS, WMIN, WMAX, request, NBPHOTONS=5e4, MAX_HIST=1e6
):
    ALB_SNOW = AlbedoCst(0.6)
    ALB_HIST = AlbedoCst(1.0)
    wl_sca = np.linspace(WMIN, WMAX, num=11)
    wl_abs = np.linspace(WMIN, WMAX, num=N_WL_ABS)
    alb = ALB_SNOW.get(wl_abs)
    lez = {"th_deg": np.array([0.0]), "phi_deg": np.array([0.0]), "zip": False}

    for AOD, fmt1 in zip(np.linspace(0.1, 0.5, num=2), ["-m", "-c"]):
        LEVEL = 0  # 1: BOA downward reflectance, 0 : TOA
        atm = Atm1D(
            "afglms",
            comp=[AerOPAC("urban", AOD, 550.0)],
            grid=np.linspace(50.0, 0.0, num=40),
        )
        sigma = od2k(atm.calc(wl_abs), "OD_abs_atm")[:, 1:]
        sg = Smartg(alis=True, alt_pp=True)
        m = (
            sg.run(
                SEED=0,
                THVDEG=45.0,
                wl=wl_sca,
                surf=LambSurface(ALB_HIST),
                le=lez,
                BEER=0,
                atm=atm.calc(wl_sca),
                alis_options={
                    "nlow": wl_sca.size,
                    "hist": True,
                    "max_hist": np.int64(MAX_HIST),
                },
                NBPHOTONS=NBPHOTONS,
                NBLOOP=NBPHOTONS,
                NF=1e3,
            )
            .dropaxis("Zenith angles")
            .dropaxis("Azimuth angles")
        )
        m0 = (
            sg.run(
                SEED=0,
                THVDEG=45.0,
                wl=wl_abs,
                surf=LambSurface(ALB_SNOW),
                le=lez,
                BEER=0,
                atm=atm.calc(wl_abs),
                alis_options={"nlow": wl_sca.size, "hist": False},
                NBPHOTONS=NBPHOTONS,
                NF=1e3,
            )
            .dropaxis("Zenith angles")
            .dropaxis("Azimuth angles")
        )
        sg.clear_context()

        with jax.default_device(
            jax.devices("cpu")[0]
        ):  # run on CPU to avoid slow GPU XLA compilation
            N, S, D, w, _, nref, _, _, _, _, _ = get_histories(
                m, LEVEL=LEVEL, verbose=False
            )
            I = (
                np.array(
                    BigSum(Si, only_I=True)(
                        wl_abs, sigma, alb, S[:, 0], w, D, nref, wl_sca
                    ).sum(axis=0)
                )
                / N
            )
            I2 = (
                np.array(
                    BigSum(Si2, only_I=True)(
                        wl_abs, sigma, alb, S[:, 0], w, D, nref, wl_sca
                    ).sum(axis=0)
                )
                / N
            )
        Std = np.sqrt((I2 - I**2) / N)
        upper = I + 1.95 * Std
        lower = I - 1.95 * Std
        p = plt.plot(
            wl_abs,
            I,
            fmt1,
            label="AOD@550: {:.1f}; NBPH={:.0e}; NBHIST={:.0e}".format(
                AOD, np.int64(NBPHOTONS), np.int64(MAX_HIST)
            ),
        )
        col = p[0].get_color()
        print(I2, Std)
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
                AOD, np.int64(NBPHOTONS)
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


@pytest.mark.skipif(
    SKIP, reason="cannot test this since the jax package is not installed."
)
def test_validation_artdeco(request, NB=5e5, VALPATH=DIR_AUXDATA):
    """
    Validation of SMART-G with ARTDECO validation data
    """
    typ = "desert"  # tau=0.25
    ####################""""""
    fgas = Path(VALPATH) / "validation" / f"cTauGas_ray_{typ}_O2.dat"
    gas_valid = diff1(np.loadtxt(fgas, skiprows=7)[:, 1:].T, axis=1)
    z_valid = np.loadtxt(fgas, skiprows=7)[:, 0]
    w_valid = np.array(open(fgas).readlines()[5].split()).astype(float)
    fray = Path(VALPATH) / "validation" / f"cTauRay_ray_{typ}_O2.dat"
    ray_valid = diff1(np.loadtxt(fray, skiprows=7)[:, 1:].T, axis=1)
    faer_abs = Path(VALPATH) / "validation" / f"cTauAbs_ptcle_ray_{typ}_O2.dat"
    aer_abs_valid = diff1(np.loadtxt(faer_abs, skiprows=7)[:, 1:].T, axis=1)
    faer_sca = Path(VALPATH) / "validation" / f"cTauSca_ptcle_ray_{typ}_O2.dat"
    aer_sca_valid = diff1(np.loadtxt(faer_sca, skiprows=7)[:, 1:].T, axis=1)
    # aerosols phase matrix import
    faer_phase = Path(VALPATH) / "validation" / f"phasemat_ray_{typ}_O2.dat"
    f = open(faer_phase, "r")
    N = np.genfromtxt(faer_phase, usecols=range(1), max_rows=1, dtype=int)
    pfwav = []
    Npf = 3
    data = np.zeros((Npf, 1, N, 5), dtype=np.float32)
    for k in range(Npf):
        pfwav.append(
            np.genfromtxt(
                faer_phase,
                usecols=range(1),
                skip_header=(1 + (2 + N) * k),
                max_rows=1,
            )
        )
        # pizero=np.genfromtxt(faer_phase, usecols=range(1),
        #                     skip_header=(1+(2+N)*k+1), max_rows=1)
        data[k, 0, :, :] = np.genfromtxt(
            faer_phase,
            usecols=range(5),
            skip_header=(1 + (2 + N) * k + 2),
            max_rows=N,
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
        Path(VALPATH) / "validation" / f"artdeco_lbl_nstr_32_ray_{typ}_O2.dat"
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
    NLOW = 3
    wl_lr = np.linspace(w_valid.min(), w_valid.max(), num=NLOW)

    sg = Smartg(alis=True, alt_pp=True)
    m1 = (
        sg.run(
            SEED=0,
            THVDEG=30.0,
            wl=w_valid,
            surf=None,
            le=le,
            BEER=0,
            atm=atm_valid.calc(w_valid),
            DEPO=0.0,
            alis_options={"nlow": NLOW, "hist": False},
            NBPHOTONS=NB,
            NBLOOP=NB,
            NF=1e3,
        )
        .dropaxis("Zenith angles")
        .dropaxis("Azimuth angles")
    )
    m2 = (
        sg.run(
            SEED=0,
            THVDEG=30.0,
            wl=w_valid,
            surf=None,
            le=le,
            BEER=0,
            atm=atm_valid.calc(w_valid),
            DEPO=0.0,
            alis_options={
                "nlow": NLOW,
                "hist": True,
                "max_hist": np.int64(1e7),
            },
            NBPHOTONS=NB,
            NBLOOP=NB,
            NF=1e3,
        )
        .dropaxis("Zenith angles")
        .dropaxis("Azimuth angles")
    )
    sg.clear_context()
    print("GPU time no hist: %.4f" % float(m1.attrs["kernel time (s)"]), "s")
    print("GPU time hist: %.4f" % float(m2.attrs["kernel time (s)"]), "s")

    with jax.default_device(
        jax.devices("cpu")[0]
    ):  # run on CPU to avoid slow GPU XLA compilation
        N, S, D, w, _, nref, _, _, _, _, _ = get_histories(
            m2, LEVEL=0, verbose=True
        )
        I = (
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
    plt.figure(figsize=(12, 4))
    plt.plot(w_valid, i_valid, "r", label="Doubling Adding: 32 streams")
    m1["I_up (TOA)"].plot("c", label="SMART-G no hist.")
    plt.plot(w_valid, I, "b", label="SMART-G, hist. with jax")
    plt.legend()
    plt.ylabel(mdesc("I_up (TOA)"))
    plt.grid()
    conftest.savefig(request)
    ##
    plt.figure(figsize=(12, 4))
    df = I - i_valid
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
