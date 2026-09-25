"""Validation of SMART-G on the 1D cases of the IPRT phase A.

The cases A1 (a Rayleigh layer), A2 (a Rayleigh layer on a Lambertian
surface) and A5 (a water cloud layer, in the principal plane and in
the almucantar) are run with SMART-G, converted to the IPRT phase A
ASCII format, and compared with the MYSTIC results.

Tested with the following GPUs: 3090
"""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from smartg import conftest
from smartg.albedo import AlbedoCst
from smartg.atmosphere import Atm1D
from smartg.config import DIR_AUXDATA
from smartg.iprt.common import compute_deltam, group_iquv
from smartg.iprt.phase_a import (
    convert_sgout_to_iprtout,
    select_and_plot_polar_iprt,
    select_iprt_iquv,
)
from smartg.phase import calc_iphase, read_phase
from smartg.sensor import Sensor
from smartg.smartg import LocalEstimate, Smartg
from smartg.surface import LambSurface
from smartg.view import plot_iquv_comparison
from smartg.xarray import drop_axes

# *********************** Global variable(s) ***************************
# a fixed seed, with the launch geometry pinned in each run, so that a
# run repeats its noise realization rather than drawing a new one
SEED = 1234
# The delta_m of a run against MYSTIC must not exceed the one of the
# saved SMART-G reference shifted by STDFAC times its standard
# deviation. Shifting every direction at once, this band is loose:
# it accepts a sign-flipped V or a Q off by 15 % in A5. It is kept as
# the IPRT benchmark check, and the run is also compared with the
# saved reference direction by direction, see Z_RMS_MAX. The V of
# MYSTIC being zero in A1 and A2, where compute_deltam returns 0, only
# the latter checks V there.
STDFAC = 4
# Root mean square, over the directions, of the difference between
# the run and the saved SMART-G reference in units of their combined
# standard deviation, to which REL_FLOOR times the reference value is
# added. It is about 1 for a run that only differs by its Monte Carlo
# noise. The floor absorbs the 1e-3 by which GPU runs of the same seed
# differ from one process to the next, which reaches twice the
# standard deviation of the I of A1 and A2: without it, the z_rms of
# that I went from 1.31 to 1.75 between two runs. Measured on gpu4 on
# 2026-09-25, three runs: 0.38 to 1.22 for every Stokes component of
# every case. The bound of 2 then fails an I scaled by 0.3 % in A1 and
# A2 and by 1.2 % in A5, a Q or U of the A5 almucantar scaled by 12 %
# (the band above accepted 15 to 23 %), zeroed or of the wrong sign.
# The U and V of A5 in the principal plane, and its V in the
# almucantar, are pure noise at this photon count and cannot be
# checked beyond their level.
Z_RMS_MAX = 2.0
REL_FLOOR = 1e-3
ROOT_PATH = Path(__file__).resolve().parent.parent
# **********************************************************************

# **************************** logging *********************************
LOG_DIR = ROOT_PATH / "tests" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "iprt_phaseA.log"
LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)

# Errors on the console, everything in the log file
logger = logging.getLogger("test_phaseA")
logger.setLevel(logging.INFO)
for handler, level in (
    (logging.StreamHandler(), logging.ERROR),
    (logging.FileHandler(LOG_FILE, mode="w"), logging.INFO),
):
    handler.setLevel(level)
    handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(handler)
# **********************************************************************


def _check_against_reference(
    run: np.ndarray, ref: np.ndarray, label: str
) -> None:
    """Compare a run with the saved reference, direction by direction.

    Both are IPRT phase A tables with standard deviations: the geometry
    in the columns 1 to 5 (the depolarization of column 0 is not always
    written in the reference), I, Q, U and V in 6 to 9 and their
    standard deviations in 10 to 13. The differences are compared with
    the combined standard deviation and REL_FLOOR, see Z_RMS_MAX. A
    component whose standard deviations are all zero, the V of a
    Rayleigh atmosphere, must be the same in both.
    """
    assert run.shape == ref.shape, f"{label}: {run.shape} vs {ref.shape}"
    assert np.array_equal(run[:, 1:6], ref[:, 1:6]), (
        f"{label}: the directions differ from the reference"
    )
    atol = 1e-6 * np.max(np.abs(ref[:, 6]))
    for istk, stk in enumerate(["I", "Q", "U", "V"]):
        diff = run[:, 6 + istk] - ref[:, 6 + istk]
        sigma = np.sqrt(
            run[:, 10 + istk] ** 2
            + ref[:, 10 + istk] ** 2
            + (REL_FLOOR * ref[:, 6 + istk]) ** 2
        )
        noisy = sigma > 0
        assert np.all(np.abs(diff[~noisy]) <= atol), (
            f"{label}: {stk} differs from the reference where both have "
            "no noise"
        )
        if not noisy.any():
            continue
        z_rms = float(np.sqrt(np.mean((diff[noisy] / sigma[noisy]) ** 2)))
        logger.info(f"{label} - {stk}: z_rms={z_rms:.3f} against the ref")
        assert z_rms <= Z_RMS_MAX, (
            f"{label}: {stk} differs from the reference by {z_rms:.3f} "
            f"standard deviations (root mean square), above {Z_RMS_MAX}"
        )


@pytest.fixture(scope="module")
def s1df() -> Smartg:
    """Forward compilation in 1D."""
    return Smartg(alt_pp=True, back=False, double=True, bias=True)


@pytest.fixture(scope="module")
def s1db() -> Smartg:
    """Backward compilation in 1D."""
    return Smartg(alt_pp=True, back=True, double=True, bias=True)


def test_a1(
    request: pytest.FixtureRequest, s1df: Smartg, s1db: Smartg
) -> None:
    """IPRT phase A, case A1: a Rayleigh layer without surface."""
    print("=== Test A1")
    mol_sca = np.array([0.0, 0.5])[None, :]
    mol_abs = np.array([0.0, 0.0])[None, :]
    z = np.array([1.0, 0.0])
    atmosphere = Atm1D(
        "afglt", grid=z, prof_ray=mol_sca, prof_abs=mol_abs
    ).calc(550.0)
    surface = None

    # *************************** DEPOL = 0 ***************************
    sza = 0.0
    saa = 65.0
    phi_0 = 180.0 - saa  # To follow MYSTIC convention
    le = LocalEstimate(
        th_deg=np.array([sza]),
        phi_deg=np.array([phi_0]),
        count_level=np.array([0]),
    )

    # BOA radiances
    vza_min = 0.0
    vza_max = 80.0
    vza_inc = 5.0
    vza = np.arange(vza_min, vza_max + vza_inc, vza_inc)

    # vaa from 0. to 180.
    vaa_min = 0.0
    vaa_max = 360.0
    vaa_inc = 5.0
    vaa = np.arange(vaa_min, vaa_max + vaa_inc, vaa_inc)

    lsensors = []
    n_vza = len(vza)
    n_vaa = len(vaa)
    n_dir = round(n_vaa * n_vza)
    for _iza, za in enumerate(vza):
        for _iaa, aa in enumerate(vaa):
            phi = -aa + 180
            lsensors.append(
                Sensor(pos_z=np.min(z), th_deg=za, ph_deg=phi, loc="ATMOS")
            )

    m_a1_b = s1db.run(
        wavelength=550.0,
        n_photons=1e7 * n_dir,
        n_loop=1e7,
        atmosphere=atmosphere,
        sensor=lsensors,
        output_layers=0,
        le=le,
        surface=surface,
        xblock=64,
        xgrid=1024,
        seed=SEED,
        beer=1,
        depol=0.0,
        stdev=True,
        progress=True,
    )
    m_a1_b = drop_axes(m_a1_b, "Azimuth angles", "Zenith angles")

    for name in list(m_a1_b.data_vars):
        if "sensor index" in m_a1_b[name].dims:
            mat_tmp = np.swapaxes(
                m_a1_b[name].values.reshape(len(vza), len(vaa)), 0, 1
            )
            attrs_tmp = m_a1_b[name].attrs
            m_a1_b = m_a1_b.drop_vars([name])
            m_a1_b[name] = xr.Variable(
                ("Azimuth angles", "Zenith angles"),
                mat_tmp,
                attrs=attrs_tmp,
            )
    m_a1_b = m_a1_b.assign_coords(
        {"Azimuth angles": -vaa + 180.0, "Zenith angles": vza}
    )

    m_a1_b_boa_dep0 = drop_axes(m_a1_b, "sensor index")

    # TOA radiances
    # We use the previous vaa
    vza_min = 100.0
    vza_max = 180.0
    vza_inc = 5.0
    vza = np.arange(vza_min, vza_max + vza_inc, vza_inc)

    lsensors = []
    n_vza = len(vza)
    n_vaa = len(vaa)
    n_dir = round(n_vaa * n_vza)
    for _iza, za in enumerate(vza):
        for _iaa, aa in enumerate(vaa):
            phi = -aa + 180
            lsensors.append(
                Sensor(pos_z=np.max(z), th_deg=za, ph_deg=phi, loc="ATMOS")
            )

    m_a1_b = s1db.run(
        wavelength=550.0,
        n_photons=1e7 * n_dir,
        n_loop=1e7,
        atmosphere=atmosphere,
        sensor=lsensors,
        output_layers=0,
        le=le,
        surface=surface,
        xblock=64,
        xgrid=1024,
        seed=SEED,
        beer=1,
        depol=0.0,
        stdev=True,
        progress=True,
    )
    m_a1_b = drop_axes(m_a1_b, "Azimuth angles", "Zenith angles")

    for name in list(m_a1_b.data_vars):
        if "sensor index" in m_a1_b[name].dims:
            mat_tmp = np.swapaxes(
                m_a1_b[name].values.reshape(len(vza), len(vaa)), 0, 1
            )
            attrs_tmp = m_a1_b[name].attrs
            m_a1_b = m_a1_b.drop_vars([name])
            m_a1_b[name] = xr.Variable(
                ("Azimuth angles", "Zenith angles"),
                mat_tmp,
                attrs=attrs_tmp,
            )
    m_a1_b = m_a1_b.assign_coords(
        {"Azimuth angles": -vaa + 180.0, "Zenith angles": vza}
    )

    m_a1_b_toa_dep0 = drop_axes(m_a1_b, "sensor index")
    # *****************************************************************

    # ************************* DEPOL = 0.03 **************************
    # We use the previous vaa and vza
    # SMART-G Forward th and phi using local estimate (anticlockwise)
    # conversion with vza and vaa MYSTIC (clockwise)
    th = 180.0 - vza
    phi = -vaa
    th[th == 0] = 1e-6  # avoid problem due to special case of 0
    le = LocalEstimate(th_deg=th, phi_deg=phi)  # , zip=True
    sza = 30.0
    saa = 0.0
    phi_0 = (
        180.0 - saa
    )  # SMART-G anticlockwise converted to be consistent with MYSTIC
    m_a1_f_dep003 = s1df.run(
        th_deg=sza,
        ph_deg=phi_0,
        wavelength=550.0,
        n_photons=1e7,
        n_loop=1e5,
        atmosphere=atmosphere,
        output_layers=7,
        le=le,
        surface=surface,
        xblock=64,
        xgrid=1024,
        seed=SEED,
        beer=1,
        depol=0.03,
        stdev=True,
    )

    # ************************* DEPOL = 0.1 **************************
    # We use the previous vaa, vza and le
    sza = 30.0
    saa = 65.0
    phi_0 = (
        180.0 - saa
    )  # SMART-G anticlockwise converted to be consistent with MYSTIC
    m_a1_f_dep01 = s1df.run(
        th_deg=sza,
        ph_deg=phi_0,
        wavelength=550.0,
        n_photons=1e7,
        n_loop=1e5,
        atmosphere=atmosphere,
        output_layers=7,
        le=le,
        surface=surface,
        xblock=64,
        xgrid=1024,
        seed=SEED,
        beer=1,
        depol=0.1,
        stdev=True,
    )
    # *****************************************************************

    with TemporaryDirectory() as tmpdir:
        # === Convert smartg output to iprt ascii output format
        # (Forward, U must be multiplied by -1)
        tmp_file_a1 = Path(tmpdir) / "a1.dat"

        vza_boa_dep0 = m_a1_b_boa_dep0.coords["Zenith angles"].values
        vaa_boa_dep0 = 180.0 - m_a1_b_boa_dep0.coords["Azimuth angles"].values
        vza_toa_dep0 = m_a1_b_toa_dep0.coords["Zenith angles"].values
        vaa_toa_dep0 = 180.0 - m_a1_b_toa_dep0.coords["Azimuth angles"].values

        vza_dep003 = 180.0 - m_a1_f_dep003.coords["Zenith angles"].values
        vaa_dep003 = -m_a1_f_dep003.coords["Azimuth angles"].values
        vza_dep01 = 180.0 - m_a1_f_dep01.coords["Zenith angles"].values
        vaa_dep01 = -m_a1_f_dep01.coords["Azimuth angles"].values

        # convert
        convert_sgout_to_iprtout(
            datasets=[
                m_a1_b_boa_dep0,
                m_a1_b_toa_dep0,
                m_a1_f_dep003,
                m_a1_f_dep003,
                m_a1_f_dep01,
                m_a1_f_dep01,
            ],
            u_signs=[1.0, 1, -1, -1, -1, -1],
            case_name="A1",
            depols=[0.0, 0.0, 0.03, 0.03, 0.1, 0.1],
            altitudes=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            szas=[0.0, 0.0, 30.0, 30.0, 30.0, 30.0],
            saas=[65.0, 65.0, 0.0, 0.0, 65.0, 65.0],
            vzas=[
                vza_boa_dep0,
                vza_toa_dep0,
                180.0 - vza_dep003,
                vza_dep003,
                180.0 - vza_dep01,
                vza_dep01,
            ],
            vaas=[
                vaa_boa_dep0,
                vaa_toa_dep0,
                vaa_dep003,
                vaa_dep003,
                vaa_dep01,
                vaa_dep01,
            ],
            fname=tmp_file_a1,
            output_layer=[
                "_up (TOA)",
                "_up (TOA)",
                "_down (0+)",
                "_up (TOA)",
                "_down (0+)",
                "_up (TOA)",
            ],
        )

        smartg_a1 = pd.read_csv(
            tmp_file_a1, header=None, sep=r"\s+", dtype=float, comment="#"
        ).values
        mystic_a1 = pd.read_csv(
            DIR_AUXDATA
            / "IPRT"
            / "phaseA"
            / "mystic_res"
            / "iprt_case_a1_mystic.dat",
            header=None,
            sep=r"\s+",
            dtype=float,
            comment="#",
        ).values
        avoid_p = False

        l_dep = [0.0, 0.03, 0.1]
        l_sza = [0.0, 30.0, 30.0]
        l_saa = [65.0, 0.0, 65.0]
        l_alt = [0.0, 1.0]
        l_invth = [False, True]
        # ============ 0km of altitude
        iquv_smartg_tot = None
        iquv_mystic_tot = None
        for isim in range(3):
            imgs = []
            for ialt in range(2):
                title = (
                    f"IPRT case A1 - depol = {l_dep[isim]} - "
                    + f"sza = {l_sza[isim]:.0f} - saa = {l_saa[isim]:.0f} - "
                    + f"{l_alt[ialt]:.0f}km - SMARTG"
                )
                tmp_filename = (
                    f"a1_dep{l_dep[isim]:.0e}_{l_alt[ialt]:.0f}km_smartg.png"
                )
                i_smartg, q_smartg, u_smartg, v_smartg = (
                    select_and_plot_polar_iprt(
                        smartg_a1,
                        z_alti=l_alt[ialt],
                        depol=l_dep[isim],
                        title=title,
                        change_u_sign=True,
                        inv_thetas=l_invth[ialt],
                        sym=False,
                        output_iquv=True,
                        avoid_plot=avoid_p,
                        save_fig=Path(tmpdir) / tmp_filename,
                    )
                )
                imgs.append(mpimg.imread(Path(tmpdir) / tmp_filename))

                tmp_filename = (
                    f"a1_dep{l_dep[isim]}_{l_alt[ialt]:.0f}km_mystic.png"
                )
                title = (
                    f"IPRT case A1 - depol = {l_dep[isim]} - "
                    + f"sza = {l_sza[isim]:.0f} - saa = {l_saa[isim]:.0f} - "
                    + f"{l_alt[ialt]:.0f}km - MYSTIC"
                )
                i_mystic, q_mystic, u_mystic, v_mystic = (
                    select_and_plot_polar_iprt(
                        mystic_a1,
                        z_alti=l_alt[ialt],
                        depol=l_dep[isim],
                        title=title,
                        change_u_sign=True,
                        inv_thetas=l_invth[ialt],
                        sym=False,
                        output_iquv=True,
                        avoid_plot=avoid_p,
                        save_fig=Path(tmpdir) / tmp_filename,
                    )
                )
                imgs.append(mpimg.imread(Path(tmpdir) / tmp_filename))

                if isim == 0 and ialt == 0:
                    iquv_smartg_tot = group_iquv(
                        i_list=[i_smartg],
                        q_list=[q_smartg],
                        u_list=[u_smartg],
                        v_list=[v_smartg],
                    )
                    iquv_mystic_tot = group_iquv(
                        i_list=[i_mystic],
                        q_list=[q_mystic],
                        u_list=[u_mystic],
                        v_list=[v_mystic],
                    )
                else:
                    assert iquv_smartg_tot is not None
                    assert iquv_mystic_tot is not None
                    iquv_smartg_tot = np.concatenate(
                        (
                            iquv_smartg_tot,
                            group_iquv(
                                i_list=[i_smartg],
                                q_list=[q_smartg],
                                u_list=[u_smartg],
                                v_list=[v_smartg],
                            ),
                        ),
                        axis=1,
                    )
                    iquv_mystic_tot = np.concatenate(
                        (
                            iquv_mystic_tot,
                            group_iquv(
                                i_list=[i_mystic],
                                q_list=[q_mystic],
                                u_list=[u_mystic],
                                v_list=[v_mystic],
                            ),
                        ),
                        axis=1,
                    )

                i_val = i_mystic - i_smartg
                q_val = q_mystic - q_smartg
                u_val = u_mystic - u_smartg
                v_val = v_mystic - v_smartg
                max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
                max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
                max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
                max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
                tmp_filename = (
                    f"a1_dep{l_dep[isim]}_0{l_alt[ialt]:.0f}m_dif.png"
                )
                title = (
                    f"IPRT case A1 - depol = {l_dep[isim]}  - "
                    + f"sza = {l_sza[isim]:.0f} - saa = {l_saa[isim]:.0f} - "
                    + f"{l_alt[ialt]:.0f}km - dif (MYSTIC-SMARTG)"
                )
                select_and_plot_polar_iprt(
                    mystic_a1,
                    z_alti=l_alt[ialt],
                    depol=l_dep[isim],
                    title=title,
                    force_iquv=[i_val, q_val, u_val, v_val],
                    max_i=max_i,
                    max_q=max_q,
                    max_u=max_u,
                    max_v=max_v,
                    cmap_i="RdBu_r",
                    avoid_plot=avoid_p,
                    save_fig=Path(tmpdir) / tmp_filename,
                )
                imgs.append(mpimg.imread(Path(tmpdir) / tmp_filename))

            plt.close("all")
            fig, axs = plt.subplots(6, 1, figsize=(12, 24))
            for i in range(6):
                axs[i].axis("off")
                axs[i].imshow(imgs[i])
            fig.tight_layout()
            conftest.savefig(request, bbox_inches="tight")

    # === Compute the delta_m values and analyse them with the previous
    # saved validated ones
    # SMARTG ref results
    smartg_a1_ref = pd.read_csv(
        DIR_AUXDATA
        / "IPRT"
        / "phaseA"
        / "smartg_ref_res"
        / "iprt_output_format"
        / "iprt_case_a1_smartg_ref.dat",
        header=None,
        sep=r"\s+",
        dtype=float,
        comment="#",
    ).values
    iquv_smartg_ref_tot = None
    iquv_smartg_std_ref_tot = None
    for isim in range(3):
        for ialt in range(2):
            (
                i_smartg_ref,
                q_smartg_ref,
                u_smartg_ref,
                v_smartg_ref,
                i_smartg_std_ref,
                q_smartg_std_ref,
                u_smartg_std_ref,
                v_smartg_std_ref,
            ) = select_and_plot_polar_iprt(
                smartg_a1_ref,
                z_alti=l_alt[ialt],
                depol=l_dep[isim],
                change_u_sign=True,
                inv_thetas=l_invth[ialt],
                sym=False,
                output_iquv=True,
                output_iquv_std=True,
                avoid_plot=True,
            )

            if isim == 0 and ialt == 0:
                iquv_smartg_ref_tot = group_iquv(
                    i_list=[i_smartg_ref],
                    q_list=[q_smartg_ref],
                    u_list=[u_smartg_ref],
                    v_list=[v_smartg_ref],
                )
                iquv_smartg_std_ref_tot = group_iquv(
                    i_list=[i_smartg_std_ref],
                    q_list=[q_smartg_std_ref],
                    u_list=[u_smartg_std_ref],
                    v_list=[v_smartg_std_ref],
                )
            else:
                assert iquv_smartg_ref_tot is not None
                assert iquv_smartg_std_ref_tot is not None
                iquv_smartg_ref_tot = np.concatenate(
                    (
                        iquv_smartg_ref_tot,
                        group_iquv(
                            i_list=[i_smartg_ref],
                            q_list=[q_smartg_ref],
                            u_list=[u_smartg_ref],
                            v_list=[v_smartg_ref],
                        ),
                    ),
                    axis=1,
                )
                iquv_smartg_std_ref_tot = np.concatenate(
                    (
                        iquv_smartg_std_ref_tot,
                        group_iquv(
                            i_list=[i_smartg_std_ref],
                            q_list=[q_smartg_std_ref],
                            u_list=[u_smartg_std_ref],
                            v_list=[v_smartg_std_ref],
                        ),
                    ),
                    axis=1,
                )

    assert iquv_mystic_tot is not None
    assert iquv_smartg_tot is not None
    assert iquv_smartg_ref_tot is not None
    assert iquv_smartg_std_ref_tot is not None

    # Compute the delta_m values from the ref smartg results
    delta_m_ref = compute_deltam(
        obs=iquv_mystic_tot, mod=iquv_smartg_ref_tot, print_res=False
    )
    logger.info(
        f"A1 - I={delta_m_ref[0]:.3f}; Q={delta_m_ref[1]:.3f}; "
        + f"U={delta_m_ref[2]:.3f}; V={delta_m_ref[3]:.3f} - ref delta_m:"
    )

    # Compute the delta_m values from the ref smartg results +- err
    delta_m_ref_p = compute_deltam(
        obs=iquv_mystic_tot,
        mod=iquv_smartg_ref_tot + STDFAC * iquv_smartg_std_ref_tot,
        print_res=False,
    )
    delta_m_ref_m = compute_deltam(
        obs=iquv_mystic_tot,
        mod=iquv_smartg_ref_tot - STDFAC * iquv_smartg_std_ref_tot,
        print_res=False,
    )

    # Compute the delta_m values from the smartg test results
    delta_m = compute_deltam(
        obs=iquv_mystic_tot, mod=iquv_smartg_tot, print_res=False
    )
    logger.info(
        f"A1 - I={delta_m[0]:.3f}; Q={delta_m[1]:.3f}; "
        + f"U={delta_m[2]:.3f}; V={delta_m[3]:.3f} - calculated delta_m"
    )

    # Check if the the test is ok by comparing smartg ref and smartg
    # test
    iquv_name = ["I", "Q", "U", "V"]
    for istk, stk in enumerate(iquv_name):
        max_val = max(
            delta_m_ref[istk], delta_m_ref_p[istk], delta_m_ref_m[istk]
        )
        assert not (delta_m[istk] > max_val), (
            f"Problem with {stk} values, get {delta_m[istk]:.5f}."
            + f" {stk} must be < to {max_val:.5f}"
        )

    _check_against_reference(smartg_a1, smartg_a1_ref, "A1")


def test_a2(request: pytest.FixtureRequest, s1df: Smartg) -> None:
    """IPRT phase A, case A2: Rayleigh layer on a Lambertian surface."""
    print("=== Test A2:")
    # === Atmosphere profile
    mol_sca = np.array([0.0, 0.1])[None, :]
    mol_abs = np.array([0.0, 0.0])[None, :]
    z = np.array([1.0, 0.0])
    atmosphere = Atm1D(
        "afglt", grid=z, prof_ray=mol_sca, prof_abs=mol_abs
    ).calc(550.0)
    surface = LambSurface(alb=AlbedoCst(0.3))

    # === Illumination conditions
    vza_min = 100.0
    vza_max = 180.0
    vza_inc = 5.0
    vza = np.arange(vza_min, vza_max + vza_inc, vza_inc)

    vaa_min = 0.0
    vaa_max = 180.0
    vaa_inc = 5.0
    vaa = np.arange(vaa_min, vaa_max + vaa_inc, vaa_inc)

    # SMART-G Forward th and phi using local estimate (anticlockwise)
    # conversion with vza and vaa MYSTIC (clockwise)
    th = 180.0 - vza
    phi = -vaa
    th[th == 0] = 1e-6  # avoid problem due to special case of 0
    le = LocalEstimate(th_deg=th, phi_deg=phi)

    sza = 50.0
    saa = 0.0
    phi_0 = (
        180.0 - saa
    )  # SMART-G anticlockwise converted to be consistent with MYSTIC

    # === Simulation
    m_a2_f = s1df.run(
        th_deg=sza,
        ph_deg=phi_0,
        wavelength=550.0,
        n_photons=1e7,
        n_loop=1e6,
        atmosphere=atmosphere,
        output_layers=7,
        le=le,
        surface=surface,
        xblock=64,
        xgrid=1024,
        beer=1,
        depol=0.03,
        stdev=True,
        seed=SEED,
    )

    with TemporaryDirectory() as tmpdir:
        # === Convert smartg output to iprt ascii output format
        # (Forward, U must be multiplied by -1)
        tmp_file_a2 = Path(tmpdir) / "a2.dat"
        convert_sgout_to_iprtout(
            datasets=[m_a2_f, m_a2_f],
            u_signs=[-1, -1],
            case_name="A2",
            depols=[0.03, 0.03],
            altitudes=[0.0, 1.0],
            szas=[50.0, 50.0],
            saas=[0.0, 0.0],
            vzas=[180.0 - vza, vza],
            vaas=[vaa, vaa],
            fname=tmp_file_a2,
            output_layer=["_down (0+)", "_up (TOA)"],
        )

        # === Plot and comparison with MYSTIC (to save in the report)
        smartg_a2 = pd.read_csv(
            tmp_file_a2, header=None, sep=r"\s+", dtype=float, comment="#"
        ).values
        mystic_a2 = pd.read_csv(
            DIR_AUXDATA
            / "IPRT"
            / "phaseA"
            / "mystic_res"
            / "iprt_case_a2_mystic.dat",
            header=None,
            sep=r"\s+",
            dtype=float,
            comment="#",
        ).values
        avoid_p = False
        # 0km of altitude
        title = "IPRT case A2 - depol = 0.03 - 0km - SMARTG"
        i_smartg_0km, q_smartg_0km, u_smartg_0km, v_smartg_0km = (
            select_and_plot_polar_iprt(
                smartg_a2,
                0.0,
                title=title,
                change_u_sign=True,
                sym=True,
                output_iquv=True,
                avoid_plot=avoid_p,
                save_fig=Path(tmpdir) / "a2_0km_smartg.png",
            )
        )

        title = "IPRT case A2 - depol = 0.03 - 0km - MYSTIC"
        i_mystic_0km, q_mystic_0km, u_mystic_0km, v_mystic_0km = (
            select_and_plot_polar_iprt(
                mystic_a2,
                0.0,
                title=title,
                change_u_sign=True,
                sym=True,
                output_iquv=True,
                avoid_plot=avoid_p,
                save_fig=Path(tmpdir) / "a2_0km_mystic.png",
            )
        )

        i_val = i_mystic_0km - i_smartg_0km
        q_val = q_mystic_0km - q_smartg_0km
        u_val = u_mystic_0km - u_smartg_0km
        v_val = v_mystic_0km - v_smartg_0km
        max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
        max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
        max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
        max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
        title = "IPRT case A2 - depol = 0.03 - 0km - dif (MYSTIC-SMARTG)"
        select_and_plot_polar_iprt(
            mystic_a2,
            0.0,
            title=title,
            force_iquv=[i_val, q_val, u_val, v_val],
            max_i=max_i,
            max_q=max_q,
            max_u=max_u,
            max_v=max_v,
            cmap_i="RdBu_r",
            avoid_plot=avoid_p,
            save_fig=Path(tmpdir) / "a2_0km_dif.png",
        )

        imgs = []
        imgs.append(mpimg.imread(Path(tmpdir) / "a2_0km_smartg.png"))
        imgs.append(mpimg.imread(Path(tmpdir) / "a2_0km_mystic.png"))
        imgs.append(mpimg.imread(Path(tmpdir) / "a2_0km_dif.png"))

    plt.close("all")
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    for i in range(3):
        axs[i].axis("off")
        axs[i].imshow(imgs[i])
    fig.tight_layout()
    conftest.savefig(request, bbox_inches="tight")

    with TemporaryDirectory() as tmpdir:
        # 1km of altitude (TOA)
        title = "IPRT case A2 - depol = 0.03 - 1km - SMARTG"
        i_smartg_1km, q_smartg_1km, u_smartg_1km, v_smartg_1km = (
            select_and_plot_polar_iprt(
                smartg_a2,
                1.0,
                title=title,
                change_u_sign=True,
                inv_thetas=True,
                sym=True,
                output_iquv=True,
                avoid_plot=avoid_p,
                save_fig=Path(tmpdir) / "a2_1km_smartg.png",
            )
        )

        title = "IPRT case A2 - depol = 0.03 - 1km - MYSTIC"
        i_mystic_1km, q_mystic_1km, u_mystic_1km, v_mystic_1km = (
            select_and_plot_polar_iprt(
                mystic_a2,
                1.0,
                title=title,
                change_u_sign=True,
                inv_thetas=True,
                sym=True,
                output_iquv=True,
                avoid_plot=avoid_p,
                save_fig=Path(tmpdir) / "a2_1km_mystic.png",
            )
        )

        i_val = i_mystic_1km - i_smartg_1km
        q_val = q_mystic_1km - q_smartg_1km
        u_val = u_mystic_1km - u_smartg_1km
        v_val = v_mystic_1km - v_smartg_1km
        max_i = max(np.abs(np.min(i_val)), np.abs(np.max(i_val)))
        max_q = max(np.abs(np.min(q_val)), np.abs(np.max(q_val)))
        max_u = max(np.abs(np.min(u_val)), np.abs(np.max(u_val)))
        max_v = max(np.abs(np.min(v_val)), np.abs(np.max(v_val)))
        title = "IPRT case A2 - depol = 0.03 - 1km - dif (MYSTIC-SMARTG)"
        select_and_plot_polar_iprt(
            mystic_a2,
            1.0,
            title=title,
            force_iquv=[i_val, q_val, u_val, v_val],
            max_i=max_i,
            max_q=max_q,
            max_u=max_u,
            max_v=max_v,
            cmap_i="RdBu_r",
            avoid_plot=avoid_p,
            save_fig=Path(tmpdir) / "a2_1km_dif.png",
        )

        imgs = []
        imgs.append(mpimg.imread(Path(tmpdir) / "a2_1km_smartg.png"))
        imgs.append(mpimg.imread(Path(tmpdir) / "a2_1km_mystic.png"))
        imgs.append(mpimg.imread(Path(tmpdir) / "a2_1km_dif.png"))

    plt.close("all")
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))  # 12,8

    for i in range(3):
        axs[i].axis("off")
        axs[i].imshow(imgs[i])
    fig.tight_layout()
    conftest.savefig(request, bbox_inches="tight")

    # === Compute the delta_m values and analyse them with the previous
    # saved validated ones
    # MYSTIC and calculated SMART-G total IQUV
    iquv_smartg_tot = group_iquv(
        i_list=[i_smartg_0km, i_smartg_1km],
        q_list=[q_smartg_0km, q_smartg_1km],
        u_list=[u_smartg_0km, u_smartg_1km],
        v_list=[v_smartg_0km, v_smartg_1km],
    )
    iquv_mystic_tot = group_iquv(
        i_list=[i_mystic_0km, i_mystic_1km],
        q_list=[q_mystic_0km, q_mystic_1km],
        u_list=[u_mystic_0km, u_mystic_1km],
        v_list=[v_mystic_0km, v_mystic_1km],
    )

    # SMARTG ref results
    smartg_a2_ref = pd.read_csv(
        DIR_AUXDATA
        / "IPRT"
        / "phaseA"
        / "smartg_ref_res"
        / "iprt_output_format"
        / "iprt_case_a2_smartg_ref.dat",
        header=None,
        sep=r"\s+",
        dtype=float,
        comment="#",
    ).values
    (
        i_smartg_0km_ref,
        q_smartg_0km_ref,
        u_smartg_0km_ref,
        v_smartg_0km_ref,
        i_smartg_std_0km_ref,
        q_smartg_std_0km_ref,
        u_smartg_std_0km_ref,
        v_smartg_std_0km_ref,
    ) = select_and_plot_polar_iprt(
        smartg_a2_ref,
        0.0,
        change_u_sign=True,
        output_iquv=True,
        output_iquv_std=True,
        avoid_plot=True,
    )
    (
        i_smartg_1km_ref,
        q_smartg_1km_ref,
        u_smartg_1km_ref,
        v_smartg_1km_ref,
        i_smartg_std_1km_ref,
        q_smartg_std_1km_ref,
        u_smartg_std_1km_ref,
        v_smartg_std_1km_ref,
    ) = select_and_plot_polar_iprt(
        smartg_a2_ref,
        1.0,
        change_u_sign=True,
        inv_thetas=True,
        output_iquv_std=True,
        output_iquv=True,
        avoid_plot=True,
    )

    iquv_smartg_ref_tot = group_iquv(
        i_list=[i_smartg_0km_ref, i_smartg_1km_ref],
        q_list=[q_smartg_0km_ref, q_smartg_1km_ref],
        u_list=[u_smartg_0km_ref, u_smartg_1km_ref],
        v_list=[v_smartg_0km_ref, v_smartg_1km_ref],
    )
    iquv_smartg_std_ref_tot = group_iquv(
        i_list=[i_smartg_std_0km_ref, i_smartg_std_1km_ref],
        q_list=[q_smartg_std_0km_ref, q_smartg_std_1km_ref],
        u_list=[u_smartg_std_0km_ref, u_smartg_std_1km_ref],
        v_list=[v_smartg_std_0km_ref, v_smartg_std_1km_ref],
    )

    # Compute the delta_m values from the ref smartg results
    delta_m_ref = compute_deltam(
        obs=iquv_mystic_tot, mod=iquv_smartg_ref_tot, print_res=False
    )
    logger.info(
        f"A2 - I={delta_m_ref[0]:.3f}; Q={delta_m_ref[1]:.3f}; "
        + f"U={delta_m_ref[2]:.3f}; V={delta_m_ref[3]:.3f} - ref delta_m:"
    )

    # Compute the delta_m values from the ref smartg results +- err
    delta_m_ref_p = compute_deltam(
        obs=iquv_mystic_tot,
        mod=iquv_smartg_ref_tot + STDFAC * iquv_smartg_std_ref_tot,
        print_res=False,
    )
    delta_m_ref_m = compute_deltam(
        obs=iquv_mystic_tot,
        mod=iquv_smartg_ref_tot - STDFAC * iquv_smartg_std_ref_tot,
        print_res=False,
    )

    # Compute the delta_m values from the smartg test results
    delta_m = compute_deltam(
        obs=iquv_mystic_tot, mod=iquv_smartg_tot, print_res=False
    )
    logger.info(
        f"A2 - I={delta_m[0]:.3f}; Q={delta_m[1]:.3f}; "
        + f"U={delta_m[2]:.3f}; V={delta_m[3]:.3f} - calculated delta_m"
    )

    # Check if the the test is ok by comparing smartg ref and smartg
    # test
    iquv_name = ["I", "Q", "U", "V"]
    for istk, stk in enumerate(iquv_name):
        max_val = max(
            delta_m_ref[istk], delta_m_ref_p[istk], delta_m_ref_m[istk]
        )
        assert not (delta_m[istk] > max_val), (
            f"Problem with {stk} values, get {delta_m[istk]:.5f}."
            + f" {stk} must be < to {max_val:.5f}"
        )

    _check_against_reference(smartg_a2, smartg_a2_ref, "A2")


def test_a5_pp(request: pytest.FixtureRequest, s1df: Smartg) -> None:
    """IPRT phase A, case A5: a water cloud, in the principal plane."""
    print("=== Test A5 principal plane:")
    # === Atmosphere profile
    z = np.array([1.0, 0.0])
    mol_sca = np.array([0.0, 0.0])[None, :]
    mol_abs = np.array([0.0, 0.0])[None, :]
    cld_tau_ext = np.full_like(mol_sca, 5.0, dtype=np.float32)
    cld_tau_ext[:, 0] = 0.0  # dtau TOA equal to 0
    cld_ssa = np.full_like(mol_sca, 0.999979, dtype=np.float32)
    prof_aer = (cld_tau_ext, cld_ssa)
    # The water cloud has a phase function with a non-negligible
    # peak, hence a fine angular resolution there. Its native grid
    # keeps the 0.01 degree step of the peak and coarsens to 3 degrees
    # where the function is smooth, 450 angles instead of the 18001
    # equally spaced ones the automatic grid would build. The n_icdf
    # of the run below is a different quantity, the number of points
    # of the inverted phase function the scattering angles are drawn
    # from, and keeps its value
    file_cld_phase = (
        DIR_AUXDATA / "IPRT" / "phaseA" / "opt_prop" / "watercloud.mie.cdf"
    )
    cld_phase = read_phase(fname=file_cld_phase, n_theta="native")
    theta_cld = cld_phase.coords["theta_atm"].to_numpy()
    pha_atm, ipha_atm = calc_iphase(cld_phase, np.array([800.0]), z)
    lpha = []
    for i in range(pha_atm.shape[0]):
        lpha.append(
            xr.DataArray(
                pha_atm[i, :, :],
                dims=["nphamat", "theta_atm"],
                coords={"theta_atm": theta_cld},
            )
        )
    atmosphere = Atm1D(
        "afglt",
        grid=z,
        prof_ray=mol_sca,
        prof_abs=mol_abs,
        prof_aer=prof_aer,
        prof_phases=(ipha_atm, lpha),
    )
    pro = atmosphere.calc(800.0, phase=False)
    surface = None

    # === Illumination conditions
    sza = 50.0
    saa = 0.0
    phi_0 = (
        180.0 - saa
    )  # SMART-G anticlockwise converted to be consistent with MYSTIC

    vza_min = 100.0
    vza_max = 180.0
    vza_inc = 1.0
    vza = np.arange(vza_min, vza_max + vza_inc, vza_inc)

    vaa = np.array([0.0, 180.0])

    # SMART-G Forward th and phi using local estimate (anticlockwise)
    # conversion with vza and vaa MYSTIC (clockwise)
    th = 180.0 - vza
    phi = -vaa
    th[th == 0] = 1e-6  # avoid problem due to special case of 0
    le = LocalEstimate(th_deg=th, phi_deg=phi)  # , zip=True

    # === Simulation
    m_a5_f_pp = s1df.run(
        th_deg=sza,
        ph_deg=phi_0,
        wavelength=800.0,
        n_photons=1e7,
        n_loop=1e6,
        n_icdf=18001,
        atmosphere=pro,
        output_layers=7,
        le=le,
        surface=surface,
        xblock=64,
        xgrid=1024,
        beer=1,
        depol=0.03,
        stdev=True,
        seed=SEED,
    )

    with TemporaryDirectory() as tmpdir:
        # === Convert smartg output to iprt ascii output format
        tmp_file_a5_pp = Path(tmpdir) / "a5_pp.dat"
        convert_sgout_to_iprtout(
            datasets=[m_a5_f_pp, m_a5_f_pp],
            u_signs=[-1, -1],
            case_name="A5",
            depols=[0.03, 0.03],
            altitudes=[0.0, 1.0],
            szas=[50.0, 50.0],
            saas=[0.0, 0.0],
            vzas=[vza, vza],
            vaas=[vaa, vaa],
            fname=tmp_file_a5_pp,
            output_layer=["_down (0+)", "_up (TOA)"],
        )

        # === Plot and comparison with MYSTIC (to save in the report)
        smartg_a5_pp = pd.read_csv(
            tmp_file_a5_pp, header=None, sep=r"\s+", dtype=float, comment="#"
        ).values
    mystic_a5_pp = pd.read_csv(
        DIR_AUXDATA
        / "IPRT"
        / "phaseA"
        / "mystic_res"
        / "iprt_case_a5_pp_mystic.dat",
        header=None,
        sep=r"\s+",
        dtype=float,
        comment="#",
    ).values
    vza_n = np.sort(np.concatenate((vza - 180, 180 - vza)))
    nvza = len(vza_n)

    # Reflectance
    iquvs_with_std = select_iprt_iquv(
        smartg_a5_pp, 1.0, change_u_sign=False, inv_thetas=True, stdev=True
    )
    iquvm_with_std = select_iprt_iquv(
        mystic_a5_pp,
        1.0,
        change_u_sign=False,
        inv_thetas=True,
        i_index=5,
        va_index=3,
        phi_index=4,
        z_index=0,
        stdev=True,
    )

    # MYSTIC IQUV and stdev IQUV
    iquvm_pp = np.zeros((4, nvza), dtype=np.float32)
    iquvstdm_pp = np.zeros((4, nvza), dtype=np.float32)
    iquvs_pp = np.zeros((4, nvza), dtype=np.float32)
    iquvstds_pp = np.zeros((4, nvza), dtype=np.float32)
    for i in range(4):
        iquvm_pp[i, :] = np.concatenate(
            (iquvm_with_std[i][:, 1], iquvm_with_std[i][::-1, 0])
        )
        iquvstdm_pp[i, :] = np.concatenate(
            (iquvm_with_std[i + 4][:, 1], iquvm_with_std[i + 4][::-1, 0])
        )
        iquvs_pp[i, :] = np.concatenate(
            (iquvs_with_std[i][:, 1], iquvs_with_std[i][::-1, 0])
        )
        iquvstds_pp[i, :] = np.concatenate(
            (iquvs_with_std[i + 4][:, 1], iquvs_with_std[i + 4][::-1, 0])
        )

    iquvs_pp_tot = iquvs_pp.copy()
    iquvm_pp_tot = iquvm_pp.copy()

    iquvy_min = [0.0, -2e-2, -1.2e-4, -1e-5]
    iquvy_max = [2.5e-1, 1.5e-2, 6e-5, 1e-5]
    plot_iquv_comparison(
        iquv_obs=iquvm_pp,
        iquv_mod=iquvs_pp,
        iquv_std_obs=iquvstdm_pp,
        iquv_std_mod=iquvstds_pp,
        xaxis=vza_n,
        xlabel="vza [deg]",
        iquv_ymin=iquvy_min,
        iquv_ymax=iquvy_max,
        title="reflectance  MYSTIC-red SMARTG-blue",
    )
    conftest.savefig(request, bbox_inches="tight")

    # Transmittance
    iquvs_with_std = select_iprt_iquv(
        smartg_a5_pp, 0.0, change_u_sign=False, inv_thetas=True, stdev=True
    )
    iquvm_with_std = select_iprt_iquv(
        mystic_a5_pp,
        0.0,
        change_u_sign=False,
        inv_thetas=True,
        i_index=5,
        va_index=3,
        phi_index=4,
        z_index=0,
        stdev=True,
    )

    # MYSTIC IQUV and stdev IQUV
    for i in range(4):
        iquvm_pp[i, :] = np.concatenate(
            (iquvm_with_std[i][:, 1], iquvm_with_std[i][::-1, 0])
        )
        iquvstdm_pp[i, :] = np.concatenate(
            (iquvm_with_std[i + 4][:, 1], iquvm_with_std[i + 4][::-1, 0])
        )
        iquvs_pp[i, :] = np.concatenate(
            (iquvs_with_std[i][:, 1], iquvs_with_std[i][::-1, 0])
        )
        iquvstds_pp[i, :] = np.concatenate(
            (iquvs_with_std[i + 4][:, 1], iquvs_with_std[i + 4][::-1, 0])
        )

    iquvy_min = [0.0, -3e-3, -1.5e-4, -2e-5]
    iquvy_max = [3.5, 4e-3, 2e-4, 3e-5]

    plot_iquv_comparison(
        iquv_obs=iquvm_pp,
        iquv_mod=iquvs_pp,
        iquv_std_obs=iquvstdm_pp,
        iquv_std_mod=iquvstds_pp,
        xaxis=vza_n,
        xlabel="vza [deg]",
        iquv_ymin=iquvy_min,
        iquv_ymax=iquvy_max,
        title="transmittance  MYSTIC-red SMARTG-blue",
    )
    conftest.savefig(request, bbox_inches="tight")

    iquvs_pp_tot = np.concatenate((iquvs_pp_tot, iquvs_pp), axis=1)
    iquvm_pp_tot = np.concatenate((iquvm_pp_tot, iquvm_pp), axis=1)

    # === Compute the delta_m values and analyse them with the previous
    # saved validated ones
    # SMARTG ref results
    smartg_a5_pp_ref = pd.read_csv(
        DIR_AUXDATA
        / "IPRT"
        / "phaseA"
        / "smartg_ref_res"
        / "iprt_output_format"
        / "iprt_case_a5_smartg_pp_ref.dat",
        header=None,
        sep=r"\s+",
        dtype=float,
        comment="#",
    ).values
    iquvs_with_std_ref = select_iprt_iquv(
        smartg_a5_pp_ref, 1.0, change_u_sign=False, inv_thetas=True, stdev=True
    )
    iquvs_pp_ref = np.zeros((4, nvza), dtype=np.float32)
    iquvs_pp_std_ref = np.zeros((4, nvza), dtype=np.float32)
    for i in range(4):
        iquvs_pp_ref[i, :] = np.concatenate(
            (iquvs_with_std_ref[i][:, 1], iquvs_with_std_ref[i][::-1, 0])
        )
        iquvs_pp_std_ref[i, :] = np.concatenate(
            (
                iquvs_with_std_ref[i + 4][:, 1],
                iquvs_with_std_ref[i + 4][::-1, 0],
            )
        )
    iquvs_pp_ref_tot = iquvs_pp_ref.copy()
    iquvs_pp_std_ref_tot = iquvs_pp_std_ref.copy()
    iquvs_with_std_ref = select_iprt_iquv(
        smartg_a5_pp_ref, 0.0, change_u_sign=False, inv_thetas=True, stdev=True
    )
    for i in range(4):
        iquvs_pp_ref[i, :] = np.concatenate(
            (iquvs_with_std_ref[i][:, 1], iquvs_with_std_ref[i][::-1, 0])
        )
        iquvs_pp_std_ref[i, :] = np.concatenate(
            (
                iquvs_with_std_ref[i + 4][:, 1],
                iquvs_with_std_ref[i + 4][::-1, 0],
            )
        )
    iquvs_pp_ref_tot = np.concatenate((iquvs_pp_ref_tot, iquvs_pp_ref), axis=1)
    iquvs_pp_std_ref_tot = np.concatenate(
        (iquvs_pp_std_ref_tot, iquvs_pp_std_ref), axis=1
    )

    # Compute the delta_m values from the ref smartg results
    delta_m_ref = compute_deltam(
        obs=iquvm_pp_tot, mod=iquvs_pp_ref_tot, print_res=False
    )
    logger.info(
        f"A5_pp - I={delta_m_ref[0]:.3f}; Q={delta_m_ref[1]:.3f}; "
        + f"U={delta_m_ref[2]:.3f}; V={delta_m_ref[3]:.3f} - ref delta_m:"
    )

    # Compute the delta_m values from the ref smartg results +- err
    delta_m_ref_p = compute_deltam(
        obs=iquvm_pp_tot,
        mod=iquvs_pp_ref_tot + STDFAC * iquvs_pp_std_ref_tot,
        print_res=False,
    )
    delta_m_ref_m = compute_deltam(
        obs=iquvm_pp_tot,
        mod=iquvs_pp_ref_tot - STDFAC * iquvs_pp_std_ref_tot,
        print_res=False,
    )

    # Compute the delta_m values from the smartg test results
    delta_m = compute_deltam(
        obs=iquvm_pp_tot, mod=iquvs_pp_tot, print_res=False
    )
    logger.info(
        f"A5_pp - I={delta_m[0]:.3f}; Q={delta_m[1]:.3f}; "
        + f"U={delta_m[2]:.3f}; V={delta_m[3]:.3f} - calculated delta_m:"
    )

    # Check if the the test is ok by comparing smartg ref and smartg
    # test
    iquv_name = ["I", "Q", "U", "V"]
    for istk, stk in enumerate(iquv_name):
        max_val = max(
            delta_m_ref[istk], delta_m_ref_p[istk], delta_m_ref_m[istk]
        )
        assert not (delta_m[istk] > max_val), (
            f"Problem with {stk} values, get {delta_m[istk]:.5f}."
            + f" {stk} must be < to {max_val:.5f}"
        )

    _check_against_reference(smartg_a5_pp, smartg_a5_pp_ref, "A5_pp")


def test_a5_al(request: pytest.FixtureRequest, s1df: Smartg) -> None:
    """IPRT phase A, case A5: a water cloud, in the almucantar."""
    print("=== Test A5 almucantar:")
    # === Atmosphere profile
    z = np.array([1.0, 0.0])
    mol_sca = np.array([0.0, 0.0])[None, :]
    mol_abs = np.array([0.0, 0.0])[None, :]
    cld_tau_ext = np.full_like(mol_sca, 5.0, dtype=np.float32)
    cld_tau_ext[:, 0] = 0.0  # dtau TOA equal to 0
    cld_ssa = np.full_like(mol_sca, 0.999979, dtype=np.float32)
    prof_aer = (cld_tau_ext, cld_ssa)
    # The water cloud has a phase function with a non-negligible
    # peak, hence a fine angular resolution there. Its native grid
    # keeps the 0.01 degree step of the peak and coarsens to 3 degrees
    # where the function is smooth, 450 angles instead of the 18001
    # equally spaced ones the automatic grid would build. The n_icdf
    # of the run below is a different quantity, the number of points
    # of the inverted phase function the scattering angles are drawn
    # from, and keeps its value
    file_cld_phase = (
        DIR_AUXDATA / "IPRT" / "phaseA" / "opt_prop" / "watercloud.mie.cdf"
    )
    cld_phase = read_phase(fname=file_cld_phase, n_theta="native")
    theta_cld = cld_phase.coords["theta_atm"].to_numpy()
    pha_atm, ipha_atm = calc_iphase(cld_phase, np.array([800.0]), z)
    lpha = []
    for i in range(pha_atm.shape[0]):
        lpha.append(
            xr.DataArray(
                pha_atm[i, :, :],
                dims=["nphamat", "theta_atm"],
                coords={"theta_atm": theta_cld},
            )
        )
    atmosphere = Atm1D(
        "afglt",
        grid=z,
        prof_ray=mol_sca,
        prof_abs=mol_abs,
        prof_aer=prof_aer,
        prof_phases=(ipha_atm, lpha),
    )
    pro = atmosphere.calc(800.0, phase=False)
    surface = None

    # === Illumination conditions
    sza = 50.0
    saa = 0.0
    phi_0 = (
        180.0 - saa
    )  # SMART-G anticlockwise converted to be consistent with MYSTIC

    vza = np.array([130.0])

    vaa_min = 0.0
    vaa_max = 180.0
    vaa_inc = 1.0
    vaa = np.arange(vaa_min, vaa_max + vaa_inc, vaa_inc)

    # SMART-G Forward th and phi using local estimate (anticlockwise)
    # conversion with vza and vaa MYSTIC (clockwise)
    th = 180.0 - vza
    phi = -vaa
    th[th == 0] = 1e-6  # avoid problem due to special case of 0
    le = LocalEstimate(th_deg=th, phi_deg=phi)

    # === Simulation
    m_a5_f_al = s1df.run(
        th_deg=sza,
        ph_deg=phi_0,
        wavelength=800.0,
        n_photons=1e7,
        n_loop=1e6,
        n_icdf=18001,
        atmosphere=pro,
        output_layers=7,
        le=le,
        surface=surface,
        xblock=64,
        xgrid=1024,
        beer=1,
        depol=0.03,
        stdev=True,
        seed=SEED,
    )

    with TemporaryDirectory() as tmpdir:
        # === Convert smartg output to iprt ascii output format
        tmp_file_a5_al = Path(tmpdir) / "a5_al.dat"
        convert_sgout_to_iprtout(
            datasets=[m_a5_f_al, m_a5_f_al],
            u_signs=[-1, -1],
            case_name="A5",
            depols=[0.03, 0.03],
            altitudes=[0.0, 1.0],
            szas=[50.0, 50.0],
            saas=[0.0, 0.0],
            vzas=[vza, vza],
            vaas=[vaa, vaa],
            fname=tmp_file_a5_al,
            output_layer=["_down (0+)", "_up (TOA)"],
        )

        # === Plot and comparison with MYSTIC (to save in the report)
        smartg_a5_al = pd.read_csv(
            tmp_file_a5_al, header=None, sep=r"\s+", dtype=float, comment="#"
        ).values
    mystic_a5_al = pd.read_csv(
        DIR_AUXDATA
        / "IPRT"
        / "phaseA"
        / "mystic_res"
        / "iprt_case_a5_al_mystic.dat",
        header=None,
        sep=r"\s+",
        dtype=float,
        comment="#",
    ).values
    vaa_n = vaa
    nvaa = len(vaa_n)

    # Reflectance
    iquvs_with_std = select_iprt_iquv(
        smartg_a5_al, 1.0, change_u_sign=False, inv_thetas=True, stdev=True
    )
    iquvm_with_std = select_iprt_iquv(
        mystic_a5_al,
        1.0,
        change_u_sign=False,
        inv_thetas=True,
        i_index=5,
        va_index=3,
        phi_index=4,
        z_index=0,
        stdev=True,
    )

    # MYSTIC IQUV and stdev IQUV
    iquvm_al = np.zeros((4, nvaa), dtype=np.float32)
    iquvstdm_al = np.zeros((4, nvaa), dtype=np.float32)
    iquvs_al = np.zeros((4, nvaa), dtype=np.float32)
    iquvstds_al = np.zeros((4, nvaa), dtype=np.float32)
    for i in range(4):
        iquvm_al[i, :] = iquvm_with_std[i][0, :]
        iquvstdm_al[i, :] = iquvm_with_std[i + 4][0, :]
        iquvs_al[i, :] = iquvs_with_std[i][0, :]
        iquvstds_al[i, :] = iquvs_with_std[i + 4][0, :]

    iquvs_al_tot = iquvs_al.copy()
    iquvm_al_tot = iquvm_al.copy()

    iquvy_min = [6e-2, -1e-2, -2e-3, -5e-5]
    iquvy_max = [1.2e-1, 2e-2, 1.2e-2, 2e-5]
    plot_iquv_comparison(
        iquv_obs=iquvm_al,
        iquv_mod=iquvs_al,
        iquv_std_obs=iquvstdm_al,
        iquv_std_mod=iquvstds_al,
        xaxis=vaa_n,
        xlabel="vza [deg]",
        iquv_ymin=iquvy_min,
        iquv_ymax=iquvy_max,
        title="reflectance  MYSTIC-red SMARTG-blue",
    )
    conftest.savefig(request, bbox_inches="tight")

    # Transmittance
    iquvs_with_std = select_iprt_iquv(
        smartg_a5_al, 0.0, change_u_sign=False, inv_thetas=True, stdev=True
    )
    iquvm_with_std = select_iprt_iquv(
        mystic_a5_al,
        0.0,
        change_u_sign=False,
        inv_thetas=True,
        i_index=5,
        va_index=3,
        phi_index=4,
        z_index=0,
        stdev=True,
    )

    # MYSTIC IQUV and stdev IQUV
    for i in range(4):
        iquvm_al[i, :] = iquvm_with_std[i][0, :]
        iquvstdm_al[i, :] = iquvm_with_std[i + 4][0, :]
        iquvs_al[i, :] = iquvs_with_std[i][0, :]
        iquvstds_al[i, :] = iquvs_with_std[i + 4][0, :]

    iquvy_min = [0.0, -3.5e-3, -3e-3, -1.5e-5]
    iquvy_max = [3.5, 5e-4, 5e-4, 2.5e-5]

    plot_iquv_comparison(
        iquv_obs=iquvm_al,
        iquv_mod=iquvs_al,
        iquv_std_obs=iquvstdm_al,
        iquv_std_mod=iquvstds_al,
        xaxis=vaa_n,
        xlabel="vza [deg]",
        iquv_ymin=iquvy_min,
        iquv_ymax=iquvy_max,
        title="transmittance  MYSTIC-red SMARTG-blue",
    )
    conftest.savefig(request, bbox_inches="tight")

    iquvs_al_tot = np.concatenate((iquvs_al_tot, iquvs_al), axis=1)
    iquvm_al_tot = np.concatenate((iquvm_al_tot, iquvm_al), axis=1)

    # === Compute the delta_m values and analyse them with the previous
    # saved validated ones
    # SMARTG ref results
    smartg_a5_al_ref = pd.read_csv(
        DIR_AUXDATA
        / "IPRT"
        / "phaseA"
        / "smartg_ref_res"
        / "iprt_output_format"
        / "iprt_case_a5_smartg_al_ref.dat",
        header=None,
        sep=r"\s+",
        dtype=float,
        comment="#",
    ).values
    iquvs_with_std_ref = select_iprt_iquv(
        smartg_a5_al_ref, 1.0, change_u_sign=False, inv_thetas=True, stdev=True
    )
    iquvs_al_ref = np.zeros((4, nvaa), dtype=np.float32)
    iquvs_al_std_ref = np.zeros((4, nvaa), dtype=np.float32)
    for i in range(4):
        iquvs_al_ref[i, :] = iquvs_with_std_ref[i][0, :]
        iquvs_al_std_ref[i, :] = iquvs_with_std_ref[i + 4][0, :]
    iquvs_al_ref_tot = iquvs_al_ref.copy()
    iquvs_al_std_ref_tot = iquvs_al_std_ref.copy()
    iquvs_with_std_ref = select_iprt_iquv(
        smartg_a5_al_ref, 0.0, change_u_sign=False, inv_thetas=True, stdev=True
    )
    for i in range(4):
        iquvs_al_ref[i, :] = iquvs_with_std_ref[i][0, :]
        iquvs_al_std_ref[i, :] = iquvs_with_std_ref[i + 4][0, :]
    iquvs_al_ref_tot = np.concatenate((iquvs_al_ref_tot, iquvs_al_ref), axis=1)
    iquvs_al_std_ref_tot = np.concatenate(
        (iquvs_al_std_ref_tot, iquvs_al_std_ref), axis=1
    )

    # Compute the delta_m values from the ref smartg results
    delta_m_ref = compute_deltam(
        obs=iquvm_al_tot, mod=iquvs_al_ref_tot, print_res=False
    )
    logger.info(
        f"A5_al - I={delta_m_ref[0]:.3f}; Q={delta_m_ref[1]:.3f}; "
        + f"U={delta_m_ref[2]:.3f}; V={delta_m_ref[3]:.3f} - ref delta_m:"
    )

    # Compute the delta_m values from the ref smartg results +- err
    delta_m_ref_p = compute_deltam(
        obs=iquvm_al_tot,
        mod=iquvs_al_ref_tot + STDFAC * iquvs_al_std_ref_tot,
        print_res=False,
    )
    delta_m_ref_m = compute_deltam(
        obs=iquvm_al_tot,
        mod=iquvs_al_ref_tot - STDFAC * iquvs_al_std_ref_tot,
        print_res=False,
    )

    # Compute the delta_m values from the smartg test results
    delta_m = compute_deltam(
        obs=iquvm_al_tot, mod=iquvs_al_tot, print_res=False
    )
    logger.info(
        f"A5_al - I={delta_m[0]:.3f}; Q={delta_m[1]:.3f}; "
        + f"U={delta_m[2]:.3f}; V={delta_m[3]:.3f} - calculated delta_m:"
    )

    # Check if the the test is ok by comparing smartg ref and smartg
    # test
    iquv_name = ["I", "Q", "U", "V"]
    for istk, stk in enumerate(iquv_name):
        max_val = max(
            delta_m_ref[istk], delta_m_ref_p[istk], delta_m_ref_m[istk]
        )
        assert not (delta_m[istk] > max_val), (
            f"Problem with {stk} values, get {delta_m[istk]:.5f}."
            + f" {stk} must be < to {max_val:.5f}"
        )

    _check_against_reference(smartg_a5_al, smartg_a5_al_ref, "A5_al")
