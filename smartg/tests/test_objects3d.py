"""GPU tests of the 3D objects on a solar tower power (STP) scene.

The quick start scene of demo_notebook_objects.py, four heliostats
reflecting the sun on a receiver over a desert aerosol and a
Lambertian ground, run in the restricted forward (RF) mode at two
wavelengths. The tests check the bookkeeping of the receiver: the
receiver image of each photon category against the category weights,
per wavelength and summed. The slow tier runs them again with the
DatomicAdd fallback of the GPUs without a double precision atomicAdd,
forced on the current GPU.
"""

from collections.abc import Iterator
from typing import Any

import geoclide as gc
import numpy as np
import pytest
import xarray as xr

import smartg.smartg as smartg_mod
from smartg.albedo import AlbedoCst
from smartg.atmosphere import AerOPAC, Atm1D
from smartg.objects3d import (
    CusForward,
    Entity,
    Matte,
    Mirror,
    Plane,
    Transformation,
)
from smartg.smartg import Smartg
from smartg.surface import LambSurface

SEED = 1234
XBLOCK = 256
XGRID = 256
SZA = 14.3
N_PHOTONS = 2e6
WAVELENGTHS = np.array([500.0, 600.0])

# half-widths of the heliostats and of the receiver, in km
W_MX, W_MY = 0.004725, 0.00642
W_RX, W_RY = 0.006, 0.007
# y rotation and x position of the four heliostats, aimed at the
# receiver for the sun zenith angle SZA
HELIOSTATS = [
    (20.281725, -0.05),
    (29.460753, -0.1),
    (35.129831, -0.15),
    (38.715473, -0.2),
]


def _plane(w_x: float, w_y: float) -> Plane:
    """Return a horizontal rectangle centred on the origin."""
    return Plane(
        p1=gc.Point(-w_x, -w_y, 0.0),
        p2=gc.Point(w_x, -w_y, 0.0),
        p3=gc.Point(-w_x, w_y, 0.0),
        p4=gc.Point(w_x, w_y, 0.0),
    )


def _scene(reflectivity: float = 0.88) -> list[Entity]:
    """Return the four heliostats and the receiver of the scene."""
    objects = [
        Entity(
            name="reflector",
            material_front=Mirror(reflectivity=reflectivity),
            material_back=Matte(),
            geo=_plane(W_MX, W_MY),
            transformation=Transformation(
                rotation=np.array([0.0, rot_y, 0.0]),
                translation=np.array([x, 0.0, 0.00517]),
            ),
        )
        for rot_y, x in HELIOSTATS
    ]
    objects.append(
        Entity(
            name="receiver",
            tc=0.0005,
            material_front=Matte(reflectivity=0.0),
            material_back=Matte(reflectivity=0.0),
            geo=_plane(W_RX, W_RY),
            transformation=Transformation(
                rotation=np.array([0.0, -101.5, 0.0]),
                translation=np.array([0.0, 0.0, 0.1065]),
            ),
        )
    )
    return objects


def _run(sg: Smartg, wavelength: float | np.ndarray) -> xr.Dataset:
    """Run the scene at the given wavelengths."""
    w2 = 0.5
    atmosphere = Atm1D(
        "afglms", comp=[AerOPAC("desert", 0.25, 550.0)], p0=877, tcwp=1.2
    )
    return sg.run(
        wavelength=wavelength,
        atmosphere=atmosphere,
        surface=LambSurface(alb=AlbedoCst(0.25)),
        th_deg=SZA,
        n_photons=N_PHOTONS,
        my_objects=_scene(),
        interval=[[-w2, -w2, -0.005], [w2, w2, 0.125]],
        cus_l=CusForward(mode="RF"),
        seed=SEED,
        xblock=XBLOCK,
        xgrid=XGRID,
        progress=False,
    )


@pytest.fixture(
    scope="module",
    params=["native", pytest.param("datomicadd", marks=pytest.mark.slow)],
)
def sg(request: pytest.FixtureRequest) -> Iterator[Smartg]:
    """Compile the kernel, natively or with the DatomicAdd fallback."""
    if request.param == "native":
        yield Smartg(double=True, obj3d=True)
        return
    source_module = smartg_mod.SourceModule

    def forced(*args: Any, **kwargs: Any) -> Any:
        options = list(kwargs.pop("options", []))
        return source_module(
            *args, options=[*options, "-DFORCE_DATOMICADD"], **kwargs
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(smartg_mod, "SourceModule", forced)
        yield Smartg(double=True, obj3d=True)


@pytest.fixture(scope="module")
def two_bands(sg: Smartg) -> xr.Dataset:
    """Run the scene at the two wavelengths."""
    return _run(sg, WAVELENGTHS)


def test_receiver_image_matches_categories(two_bands: xr.Dataset) -> None:
    """The receiver image of each category sums to its irradiance."""
    image = two_bands["C_Receiver"].sum(("X_Cell_Index", "Y_Cell_Index"))
    np.testing.assert_allclose(
        image.values, two_bands["cat_irr"].values, rtol=1e-8
    )
    # the total (category 0) is the sum of the eight categories
    np.testing.assert_allclose(
        two_bands["cat_w"].values[0],
        two_bands["cat_w"].values[1:].sum(),
        rtol=1e-8,
    )


def test_band_weights_sum_to_categories(two_bands: xr.Dataset) -> None:
    """The per wavelength category weights add up to the totals."""
    w_ph_cats = two_bands["wPhCats"]
    assert w_ph_cats.dims == ("Categories", "wavelength")
    np.testing.assert_allclose(
        w_ph_cats.sum("wavelength").values,
        two_bands["cat_w"].values,
        rtol=1e-8,
    )
    # both wavelengths reach the receiver
    assert np.all(w_ph_cats.values[0] > 0.0)


def test_every_counted_category_is_weighted(two_bands: xr.Dataset) -> None:
    """A category counting photons also counts their weight."""
    counts = two_bands["cat_PhNb"].values
    weights = two_bands["cat_w"].values
    # the scene reaches category 7 (environment and surface), which
    # the DatomicAdd fallback used to count without its weight
    assert counts[7] > 0
    np.testing.assert_array_equal(weights[counts > 0] > 0.0, True)
