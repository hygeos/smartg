"""Tests of the photon histories, replayed with jax.

JAX runs on the CPU here, see the environment set above the
imports.
"""
import os

# Must be set before importing JAX so only the CPU backend
# is initialised. This prevents JAX from creating a CUDA context
# that conflicts with PyCUDA's at process exit
# ("context::pop failed: invalid device context").
os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import logging
from contextlib import suppress
from gc import collect

import pytest

jax = pytest.importorskip(
    "jax", reason="cannot test this since the jax package is not installed."
)
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from smartg import conftest
from smartg.albedo import AlbedoCst
from smartg.atmosphere import AerOPAC, Atm1D, od2k
from smartg.config import DIR_AUXDATA
from smartg.diff import diff1
from smartg.histories import (
    big_sum,
    get_histories,
    get_photon_index,
    photon_mean_std,
    si,
)
from smartg.smartg import Alis, LocalEstimate, Smartg
from smartg.surface import LambSurface
from smartg.view import mdesc
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


# Largest relative difference between the replay of the histories and
# the direct run of test_smartg_jax2, see its docstring
REPLAY_RTOL = 1e-2
# ARTDECO comparison: ALIS draws the photon paths once for all the
# wavelengths, so the relative errors of the ~1000 wavelengths come from
# one noise realization. Its maximum, measured at 1.7 to 2.3 % on gpu4,
# gets a noise margin, while the mean relative error (0.1 to 0.3 %)
# catches a bias over the spectrum
ARTDECO_MAX_RTOL = 3e-2
ARTDECO_MEAN_RTOL = 5e-3


# Clean up JAX memory and stale PyCUDA atexit handlers after each test
@pytest.fixture(scope="function", autouse=True)
def cleanup_after_each_test() -> Iterator[None]:
    """Clear the jax caches and the pycuda exit hook after each test."""
    yield
    jax.clear_caches()
    collect()
    # Each Smartg() import of pycuda.autoinit registers a new
    # _finish_up atexit callback. After sg.clear_context() the
    # context is already gone, so the callback
    # raises "context::pop failed". Unregister it here while we
    # still have control, so the error is never printed. The
    # suppress covers a pycuda without autoinit or _finish_up.
    with suppress(ImportError, AttributeError):
        import atexit

        import pycuda.autoinit as _pai

        atexit.unregister(_pai._finish_up)


@pytest.mark.parametrize("n_wavelength_abs", [301])
@pytest.mark.parametrize("wmax", [350.0])
@pytest.mark.parametrize("wmin", [320.0])
def test_smartg_jax2(
    n_wavelength_abs: int,
    wmin: float,
    wmax: float,
    request: pytest.FixtureRequest,
    n_photons: float = 5e4,
    max_hist: float = 1e6,
) -> None:
    """Replay the histories of a run with jax, on the CPU.

    The two runs, with and without histories, draw the same photon
    paths (same seed), so the replay, which applies the absorption and
    the albedo of the direct run, must give its TOA reflectance back:
    within 0.3 % measured on gpu4 on 2026-09-25, up to the float32
    weights of the kernel. REPLAY_RTOL leaves room for that, and fails
    a replay that would keep the unit albedo of the histories or miss
    the absorption.
    """
    alb_snow = AlbedoCst(0.6)
    alb_hist = AlbedoCst(1.0)
    wavelength_sca = np.linspace(wmin, wmax, num=11)
    wavelength_abs = np.linspace(wmin, wmax, num=n_wavelength_abs)
    alb = alb_snow.get(wavelength_abs)
    lez = LocalEstimate(
        th_deg=np.array([0.0]), phi_deg=np.array([0.0]), zip=False
    )

    max_rel_diff = []
    for aod, fmt1 in zip(
        np.linspace(0.1, 0.5, num=2), ["-m", "-c"], strict=True
    ):
        level = 0  # 1: BOA downward reflectance, 0 : TOA
        atmosphere = Atm1D(
            "afglms",
            comp=[AerOPAC("urban", aod, 550.0)],
            grid=np.linspace(50.0, 0.0, num=40),
        )
        sigma = od2k(atmosphere.calc(wavelength_abs), "OD_abs_atm")[:, 1:]
        sg = Smartg(alis=True, alt_pp=True)
        m = (
            sg.run(
                seed=0,
                th_deg=45.0,
                wavelength=wavelength_sca,
                surface=LambSurface(alb_hist),
                le=lez,
                beer=0,
                atmosphere=atmosphere.calc(wavelength_sca),
                alis_options=Alis(
                    n_low=wavelength_sca.size,
                    hist=True,
                    max_hist=int(max_hist),
                ),
                n_photons=n_photons,
                n_loop=n_photons,
                n_icdf=1e3,
            )
        )
        m = drop_axes(m, "Zenith angles", "Azimuth angles")
        m0 = (
            sg.run(
                seed=0,
                th_deg=45.0,
                wavelength=wavelength_abs,
                surface=LambSurface(alb_snow),
                le=lez,
                beer=0,
                atmosphere=atmosphere.calc(wavelength_abs),
                alis_options=Alis(
                    n_low=wavelength_sca.size, hist=False
                ),
                n_photons=n_photons,
                n_icdf=1e3,
            )
        )
        m0 = drop_axes(m0, "Zenith angles", "Azimuth angles")
        sg.clear_context()

        with jax.default_device(
            jax.devices("cpu")[0]
        ):  # run on CPU to avoid slow GPU XLA compilation
            n, s, d, w, _, nref, _, _, _, _, _ = get_histories(
                m, level=level, verbose=False
            )
            values = np.array(
                big_sum(si, only_i=True)(
                    wavelength_abs, sigma, alb, s[:, 0], w, d,
                    nref, wavelength_sca
                )
            )
        # the records of a photon are summed before the variance
        stk_i, std = photon_mean_std(
            values, get_photon_index(m, level=level), n
        )
        i_direct = m0["I_up (TOA)"].to_numpy()
        max_rel_diff.append(float(np.max(np.abs(stk_i / i_direct - 1))))
        logger.info(
            f"jax2 AOD {aod:.1f}: replay vs direct run, max relative "
            f"difference {max_rel_diff[-1]:.2e}"
        )
        upper = stk_i + 1.95 * std
        lower = stk_i - 1.95 * std
        p = plt.plot(
            wavelength_abs,
            stk_i,
            fmt1,
            label=f"AOD@550: {aod:.1f}; NBPH={np.int64(n_photons):.0e}; "
            f"NBHIST={int(max_hist):.0e}",
        )
        col = p[0].get_color()
        print(stk_i, std)
        plt.fill_between(
            wavelength_abs,
            lower,
            upper,
            facecolor=col,
            edgecolor=col,
            alpha=0.4,
            label="95 percent confidence",
        )
        plt.plot(
            wavelength_abs,
            m0["I_up (TOA)"][:],
            marker="+",
            ls="",
            label=f"AOD@550: {aod:.1f}; NBPH={np.int64(n_photons):.0e}; "
            "NO HIST",
            color=p[0].get_color(),
        )
        plt.xlabel(r"$\lambda (nm)$")
        plt.ylabel("TOA reflectance")
        # plt.ylim(0.,0.7)
        plt.title("Urban aerosols, SZA=45°, nadir viewing, snow albedo")
        plt.grid()
    plt.legend()
    conftest.savefig(request)

    assert max(max_rel_diff) < REPLAY_RTOL, (
        f"the replay of the histories differs from the direct run by "
        f"up to {max(max_rel_diff):.2e}, more than {REPLAY_RTOL}"
    )


def test_histories_several_loops(n_photons: float = 5e4) -> None:
    """Check the histories of several loops and directions.

    Each loop used to overwrite the records of the previous ones while
    Nphotons_in counted them all, so that the radiance rebuilt from
    the histories came out about n_loops times too small. With two
    directions, the records overlapped each other.
    """
    wavelength = np.linspace(320.0, 350.0, num=11)
    atmosphere = Atm1D(
        "afglms", grid=np.linspace(50.0, 0.0, num=21)
    ).calc(wavelength)
    kabs = od2k(atmosphere, "OD_abs_atm")[:, 1:]
    common = {
        "seed": 0,
        "th_deg": 45.0,
        "wavelength": wavelength,
        "surface": LambSurface(AlbedoCst(1.0)),
        "le": LocalEstimate(th_deg=[0.0, 30.0], phi_deg=[0.0]),
        "atmosphere": atmosphere,
        "n_photons": n_photons,
        # few enough threads for several loops of n_photons / 5
        "xblock": 64,
        "xgrid": 16,
    }
    max_hist = int(1e6)
    sg = Smartg(alis=True, alt_pp=True)
    m_hist = sg.run(
        alis_options=Alis(
            n_low=wavelength.size, hist=True, max_hist=max_hist
        ),
        n_loop=n_photons / 5,
        **common,
    )
    m_ref = sg.run(alis_options=Alis(n_low=wavelength.size), **common)
    sg.clear_context()
    assert int(m_hist.attrs["number of kernel iterations"]) >= 5
    assert int(m_hist.attrs["hist records"]) <= max_hist

    # (wavelength, Zenith angles)
    i_ref = np.squeeze(m_ref["I_up (TOA)"].values)
    for idir in range(2):
        with jax.default_device(jax.devices("cpu")[0]):
            n, s, d, w, _, nref, _, _, _, _, _ = get_histories(
                m_hist, idir=idir
            )
            values = np.array(
                big_sum(si, only_i=True)(
                    wavelength, kabs, np.ones_like(wavelength),
                    s[:, 0], w, d, nref, wavelength
                )
            )
        # the photons are numbered over all the loops
        index = get_photon_index(m_hist, idir=idir)
        assert index.max() < n
        stk_i, std = photon_mean_std(values, index, n)
        rel_diff = np.max(np.abs(stk_i / i_ref[:, idir] - 1))
        n_sigma = np.max(np.abs(stk_i - i_ref[:, idir]) / std)
        logger.info(
            f"several loops, direction {idir}: max rel diff = "
            f"{rel_diff:.4f}, {n_sigma:.2f} std, rel std "
            f"{np.max(std / stk_i):.4f}"
        )
        np.testing.assert_allclose(stk_i, i_ref[:, idir], rtol=0.05)
        # the reference has about the same noise as the histories
        assert np.all(np.isfinite(std)) and np.all(std > 0.0)
        assert n_sigma < 6.0 * np.sqrt(2.0)


def test_validation_artdeco(
    request: pytest.FixtureRequest,
    n_photons: float = 5e5,
    valpath: Path = DIR_AUXDATA,
) -> None:
    """Check SMART-G against the ARTDECO validation data."""
    typ = "desert"  # tau=0.25
    fgas = Path(valpath) / "validation" / f"cTauGas_ray_{typ}_O2.dat"
    gas_valid = diff1(np.loadtxt(fgas, skiprows=7)[:, 1:].T, axis=1)
    z_valid = np.loadtxt(fgas, skiprows=7)[:, 0]
    w_valid = np.array(fgas.read_text().splitlines()[5].split()).astype(float)
    fray = Path(valpath) / "validation" / f"cTauRay_ray_{typ}_O2.dat"
    ray_valid = diff1(np.loadtxt(fray, skiprows=7)[:, 1:].T, axis=1)
    faer_abs = Path(valpath) / "validation" / f"cTauAbs_ptcle_ray_{typ}_O2.dat"
    aer_abs_valid = diff1(np.loadtxt(faer_abs, skiprows=7)[:, 1:].T, axis=1)
    faer_sca = Path(valpath) / "validation" / f"cTauSca_ptcle_ray_{typ}_O2.dat"
    aer_sca_valid = diff1(np.loadtxt(faer_sca, skiprows=7)[:, 1:].T, axis=1)
    # aerosols phase matrix import
    faer_phase = Path(valpath) / "validation" / f"phasemat_ray_{typ}_O2.dat"
    n_theta = int(
        np.genfromtxt(faer_phase, usecols=range(1), max_rows=1, dtype=int)
    )
    wavelength_phase = []
    npf = 3
    data = np.zeros((npf, 1, n_theta, 5), dtype=np.float32)
    for k in range(npf):
        wavelength_phase.append(
            np.genfromtxt(
                faer_phase,
                usecols=range(1),
                skip_header=(1 + (2 + n_theta) * k),
                max_rows=1,
            )
        )
        # pizero=np.genfromtxt(faer_phase, usecols=range(1),
        #                     skip_header=(1+(2+n)*k+1), max_rows=1)
        data[k, 0, :, :] = np.genfromtxt(
            faer_phase,
            usecols=range(5),
            skip_header=(1 + (2 + n_theta) * k + 2),
            max_rows=n_theta,
        )
    data = data.swapaxes(2, 3)

    # From iparper to standard phase convention
    pha_data = data[:, :, 1:, :].copy()
    pha_data[:, :, 0, :] = (data[:, :, 1, :] + data[:, :, 2, :]) * 0.5
    pha_data[:, :, 1, :] = (data[:, :, 1, :] - data[:, :, 2, :]) * 0.5

    phase_valid = xr.DataArray(
        pha_data,
        dims=["wavelength_phase", "z_phase", "nphamat", "theta_atm"],
        coords={
            "wavelength_phase": wavelength_phase,
            "z_phase": [0],
            "theta_atm": data[0, 0, 0, :],
        },
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
        wavelength_phase=wavelength_phase,
        comp=comp,
        prof_ray=ray_valid,
        prof_aer=(aer_ext_valid, aer_ssa_valid),
        prof_abs=gas_valid,
    )
    sigma_valid = od2k(atm_valid.calc(w_valid), "OD_abs_atm")[:, 1:]
    ###############

    le = LocalEstimate(
        th_deg=np.array([20.0]),
        phi_deg=np.array([180.0]),
        zip=False,
    )
    nlow = 3
    wavelength_lr = np.linspace(w_valid.min(), w_valid.max(), num=nlow)

    sg = Smartg(alis=True, alt_pp=True)
    m1 = (
        sg.run(
            seed=0,
            th_deg=30.0,
            wavelength=w_valid,
            surface=None,
            le=le,
            beer=0,
            atmosphere=atm_valid.calc(w_valid),
            depol=0.0,
            alis_options=Alis(n_low=nlow, hist=False),
            n_photons=n_photons,
            n_loop=n_photons,
            n_icdf=1e3,
        )
    )
    m1 = drop_axes(m1, "Zenith angles", "Azimuth angles")
    m2 = (
        sg.run(
            seed=0,
            th_deg=30.0,
            wavelength=w_valid,
            surface=None,
            le=le,
            beer=0,
            atmosphere=atm_valid.calc(w_valid),
            depol=0.0,
            alis_options=Alis(
                n_low=nlow,
                hist=True,
                max_hist=int(1e7),
            ),
            n_photons=n_photons,
            n_loop=n_photons,
            n_icdf=1e3,
        )
    )
    m2 = drop_axes(m2, "Zenith angles", "Azimuth angles")
    sg.clear_context()
    print(f"GPU time no hist: {float(m1.attrs['kernel time (s)']):.4f}", "s")
    print(f"GPU time hist: {float(m2.attrs['kernel time (s)']):.4f}", "s")

    with jax.default_device(
        jax.devices("cpu")[0]
    ):  # run on CPU to avoid slow GPU XLA compilation
        n, s, d, w, _, nref, _, _, _, _, _ = get_histories(
            m2, level=0, verbose=True
        )
        stk_i = (
            np.array(
                big_sum(si, only_i=True)(
                    w_valid,
                    sigma_valid,
                    np.zeros_like(w_valid),
                    s[:, 0],
                    w,
                    d,
                    nref,
                    wavelength_lr,
                ).sum(axis=0)
            )
            / n
        )

    #####################
    # Only I is compared. ALIS weighs the other wavelengths with a
    # scalar, which is exact for I, while Q, U and V keep the
    # polarization of the wavelength drawn: right as long as the
    # scattering properties do not change across the band, as over
    # this O2-A band, where only the absorption does (see
    # test_self_consistency.py for the limitation and a fix)
    i_valid = data_valid[:, 1]

    # Mask for "significant" reference values, where the relative
    # error is meaningful
    sig = np.abs(i_valid) > 1e-3

    # --- no-hist ---
    abs_err_no_hist = np.abs(m1["I_up (TOA)"][:] - i_valid)
    rel_err_no_hist = abs_err_no_hist[sig] / np.abs(i_valid[sig])
    logger.info(
        f"no-hist vs ARTDECO (|ref|>1e-3) - "
        f"max rel err = {np.max(rel_err_no_hist) * 100:.6f}% - "
        f"mean rel err = {np.mean(rel_err_no_hist) * 100:.6f}% - "
        f"max abs err = {np.max(abs_err_no_hist):.6e}"
    )
    assert np.max(rel_err_no_hist) < ARTDECO_MAX_RTOL, (
        f"SMART-G no-hist exceeds {ARTDECO_MAX_RTOL:.0%} relative error vs "
        f"ARTDECO (|ref|>1e-3): max rel err = {rel_err_no_hist.max():.4%}"
    )
    assert np.mean(rel_err_no_hist) < ARTDECO_MEAN_RTOL, (
        f"SMART-G no-hist mean relative error vs ARTDECO exceeds "
        f"{ARTDECO_MEAN_RTOL:.1%}: {float(np.mean(rel_err_no_hist)):.4%}"
    )

    # --- hist+jax ---
    abs_err_hist = np.abs(stk_i - i_valid)
    rel_err_hist = abs_err_hist[sig] / np.abs(i_valid[sig])
    logger.info(
        f"hist+jax vs ARTDECO (|ref|>1e-3) - "
        f"max rel err = {np.max(rel_err_hist) * 100:.6f}% - "
        f"mean rel err = {np.mean(rel_err_hist) * 100:.6f}% - "
        f"max abs err = {np.max(abs_err_hist):.6e}"
    )
    assert np.max(rel_err_hist) < ARTDECO_MAX_RTOL, (
        f"SMART-G hist+jax exceeds {ARTDECO_MAX_RTOL:.0%} relative error vs "
        f"ARTDECO (|ref|>1e-3): max rel err = {rel_err_hist.max():.4%}"
    )
    assert np.mean(rel_err_hist) < ARTDECO_MEAN_RTOL, (
        f"SMART-G hist+jax mean relative error vs ARTDECO exceeds "
        f"{ARTDECO_MEAN_RTOL:.1%}: {float(np.mean(rel_err_hist)):.4%}"
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
