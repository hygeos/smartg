"""Tests of the 1D atmosphere profiles.

Each one builds an Atm1D, with or without components, on the
default grid or on a given one, and computes it at one or several
wavelengths.
"""
import numpy as np
import pytest
from numpy.typing import NDArray

from smartg.atmosphere import AerOPAC, Atm1D, Cloud

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
