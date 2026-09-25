"""Tests of the 1D atmosphere profiles.

Each one builds an Atm1D, with or without components, on the
default grid or on a given one, and computes it at one or several
wavelengths.
"""
import numpy as np
import pytest
from numpy.typing import NDArray

from smartg.atmosphere import (
    AerOPAC,
    Atm1D,
    Cloud,
    n_air_co2,
    refractivity,
    strgrid_to_numpy,
)

Wavelength = float | list[float] | NDArray[np.float64]

"""Tests for the ``Atm1D`` profile calculation.

The ``wavelength`` fixture parametrizes the wavelength input
across the formats
accepted by :meth:`Atm1D.calc` (float, 0-d array, list, multi-element
array).
Each ``test_profile*`` exercises a different ``Atm1D`` configuration:
default ``afglt``, ``afglms`` with explicit ``grid``/``pfgrid``, with
aerosol or aerosol+cloud components, and with overridden ``tau_r`` or
``ssa``.
"""


@pytest.fixture(
    params=[
        500.0,
        np.array(500.0),
        [400.0],
        np.array([400.0, 800.0]),
    ]
)
def wavelength(request: pytest.FixtureRequest) -> Wavelength:
    """Give a wavelength as a float, a 0-d array, a list, an array."""
    return request.param


def test_profile1(wavelength: Wavelength) -> None:
    """Compute the default atmosphere."""
    atmosphere = Atm1D("afglt")
    atmosphere.calc(wavelength)


def test_profile2(wavelength: Wavelength) -> None:
    """Compute an atmosphere on given altitude and phase grids."""
    atmosphere = Atm1D(
        "afglms",
        grid=[100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.0],
        pfgrid=[100.0, 10.0, 0.0],
    )
    atmosphere.calc(wavelength)


def test_profile3(wavelength: Wavelength) -> None:
    """Compute an atmosphere holding an aerosol, on a string grid."""
    atmosphere = Atm1D(
        "afglms",
        comp=[AerOPAC("desert", 0.1, 550.0)],
        grid="100[20]10[1]0",
        pfgrid=[100.0, 10.0, 0.0],
    )
    atmosphere.calc(wavelength)


def test_profile4(wavelength: Wavelength) -> None:
    """Compute an atmosphere holding an aerosol and a cloud."""
    atmosphere = Atm1D(
        "afglms",
        comp=[
            AerOPAC("desert", 0.1, 550.0),
            Cloud("wc", 12.68, 2, 3, 10.0, 550.0),
        ],
        grid=[100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.0],
        pfgrid=[100.0, 10.0, 0.0],
    )
    atmosphere.calc(wavelength)


def test_profile5() -> None:
    # set tauray
    """Check that tau_r sets the Rayleigh optical depth."""
    pro = Atm1D("afglms", grid=[100, 20, 0.0], tau_r=0.14).calc(500.0)
    assert np.isclose(pro["OD_r"][0, -1], 0.14)

    Atm1D("afglms", grid=[100, 20, 0.0], tau_r=0.14).calc([490.0, 500.0])
    Atm1D("afglms", grid=[100, 20, 0.0], tau_r=[0.15, 0.14]).calc(
        [490.0, 500.0]
    )


def test_profile6() -> None:
    # set ssa
    """Compute an aerosol with its albedo overridden."""
    Atm1D(
        "afglms",
        grid=[100, 20, 0.0],
        comp=[AerOPAC("urban", 0.1, 550.0, ssa=0.8)],
    ).calc(400.0)

    Atm1D(
        "afglms",
        grid=[100, 20, 0.0],
        comp=[AerOPAC("urban", 0.1, 550.0, ssa=0.8)],
    ).calc([400.0, 500.0, 600.0])

    Atm1D(
        "afglms",
        grid=[100, 20, 0.0],
        comp=[AerOPAC("urban", 0.1, 550.0, ssa=[0.76, 0.77, 0.78])],
    ).calc([400.0, 500.0, 600.0])


def test_refractivity_of_standard_air() -> None:
    """Check the refractive index of air at 15 °C and 1013.25 hPa.

    The updated Edlén equation (Birch and Downs 1993) is written so
    that n - 1 at these standard conditions is the standard value of
    `n_air_co2`, about 2.78e-4 at 550 nm. A pressure taken in hPa
    instead of Pa makes it 100 times too small.
    """
    n = refractivity(0.55, 1013.25, 288.15, 400.0)
    assert 2.7e-4 < n[0, 0] - 1.0 < 2.9e-4
    np.testing.assert_allclose(
        n - 1.0, n_air_co2(0.55, 400.0) - 1.0, rtol=1e-4
    )
    # the profile carries it: the US standard surface is close to
    # the standard conditions
    pro = Atm1D("afglus").calc(550.0)
    assert 2.7e-4 < pro["n_atm"].values[0, -1] - 1.0 < 2.9e-4


def test_calc_split_without_phase_matrices() -> None:
    """calc_split gives no phase profile when there is no matrix.

    With phase=False, or without any component: it raised a KeyError
    on 'iphase_atm'. Its per-layer optical thicknesses rebuild the
    profile, but for the optical thickness above the top level.
    """
    atm = Atm1D("afglus", comp=[AerOPAC("desert", 0.1, 550.0)])
    prof_abs, prof_ray, prof_aer, prof_phases = atm.calc_split(
        500.0, phase=False
    )
    assert prof_phases is None
    assert Atm1D("afglus").calc_split(500.0)[3] is None

    pro = atm.calc(500.0, phase=False)
    rebuilt = Atm1D(
        "afglus", prof_abs=prof_abs, prof_ray=prof_ray, prof_aer=prof_aer,
        prof_phases=prof_phases,
    ).calc(500.0, phase=False)
    for name in ("OD_p", "OD_r", "OD_g"):
        np.testing.assert_allclose(
            rebuilt[name].values, pro[name].values, rtol=1e-5, atol=1e-8
        )


@pytest.mark.parametrize(
    "grid", ["0[1]100", np.arange(0.0, 101.0), [100.0, 2.0, 5.0, 0.0]],
    ids=["string", "array", "unsorted"],
)
def test_grid_not_decreasing_refused(grid: object) -> None:
    """A grid that does not run from TOA to BOA is refused.

    It went through and gave NaN particle optical thicknesses.
    """
    with pytest.raises(ValueError, match="grid must decrease strictly"):
        Atm1D("afglus", comp=[AerOPAC("desert", 0.1, 550.0)], grid=grid)
    if not isinstance(grid, str):
        with pytest.raises(ValueError, match="pfgrid must decrease"):
            Atm1D("afglus", pfgrid=grid)


def test_strgrid_error_gives_a_toa_to_boa_example() -> None:
    """The parse error of strgrid_to_numpy shows a TOA to BOA grid."""
    with pytest.raises(ValueError, match=r'"500\[10\]100\[1\]0"'):
        strgrid_to_numpy("100")


def test_grid_beyond_the_profile_refused() -> None:
    """A grid above the top of the profile file is refused.

    There is no air above it, and the Rayleigh optical thickness came
    out NaN from a 0 / 0 CO2 ratio, which profile() now guards.
    """
    with pytest.raises(ValueError, match="from 0 to 120 km"):
        Atm1D("afglus", grid="150[10]120[5]0")
    # with the Rayleigh scattering forced it may, as the camera level
    # of the IPRT case E6 does
    Atm1D("afglus", grid=[3e5, 120.0, 0.0], prof_ray=np.zeros((1, 3)))
    atm = Atm1D("afglus")
    pro = atm.profile(
        550.0, prof=atm.prof.regrid(np.array([150.0, 120.0, 50.0, 0.0]))
    )
    assert np.isfinite(pro["OD_r"].values).all()
    assert np.isfinite(pro["OD_sca_atm"].values).all()
