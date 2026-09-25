"""GPU tests of the 3D objects on a solar tower power (STP) scene.

The quick start scene of demo_notebook_objects.py, four heliostats
reflecting the sun on a receiver over a desert aerosol and a
Lambertian ground, run in the restricted forward (RF) mode at two
wavelengths. The tests check the bookkeeping of the receiver: the
receiver image of each photon category against the category weights,
per wavelength and summed, and the weights of the optical losses at
the heliostats, per wavelength, down to the efficiencies of
nopt_view. The slow tier runs them again with the
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
from smartg.view import nopt_view

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


def _scene(reflectivity: float | np.ndarray = 0.88) -> list[Entity]:
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


def _transparent() -> Atm1D:
    """Return an atmosphere that neither scatters nor absorbs."""
    return Atm1D("afglt", tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0)


def _receiver(
    half: float,
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tc: float = 0.001,
) -> Entity:
    """Return a black square receiver of the given half-width."""
    return Entity(
        name="receiver",
        tc=tc,
        material_front=Matte(reflectivity=0.0),
        material_back=Matte(reflectivity=0.0),
        geo=_plane(half, half),
        transformation=Transformation(
            rotation=np.array(rotation), translation=np.array(translation)
        ),
    )


def _run_ff(
    sg: Smartg,
    objects: list[Entity],
    atmosphere: Atm1D,
    field: float,
    centre: tuple[float, float] = (0.0, 0.0),
    wavelength: float = 550.0,
    **kwargs: Any,
) -> xr.Dataset:
    """Run a scene in the FF mode, zenith sun and black ground.

    The photons are launched over a square of side field centred on
    centre, and the direct ones are counted by the receivers.
    """
    return sg.run(
        wavelength=wavelength,
        atmosphere=atmosphere,
        surface=LambSurface(alb=AlbedoCst(0.0)),
        th_deg=0.0,
        n_photons=1e6,
        my_objects=objects,
        cus_l=CusForward(
            cfx=field, cfy=field, cftx=centre[0], cfty=centre[1], mode="FF"
        ),
        direct=True,
        seed=SEED,
        xblock=XBLOCK,
        xgrid=XGRID,
        progress=False,
        **kwargs,
    )


def _run(
    sg: Smartg,
    wavelength: float | np.ndarray,
    atmosphere: Atm1D | None = None,
    reflectivity: float | np.ndarray = 0.88,
) -> xr.Dataset:
    """Run the scene at the given wavelengths.

    Over a desert aerosol by default.
    """
    w2 = 0.5
    if atmosphere is None:
        atmosphere = Atm1D(
            "afglms", comp=[AerOPAC("desert", 0.25, 550.0)], p0=877,
            tcwp=1.2,
        )
    return sg.run(
        wavelength=wavelength,
        atmosphere=atmosphere,
        surface=LambSurface(alb=AlbedoCst(0.25)),
        th_deg=SZA,
        n_photons=N_PHOTONS,
        my_objects=_scene(reflectivity),
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


def _check_loss_identities(
    w_loss: np.ndarray, reflectivity: float | np.ndarray
) -> None:
    """Check the loss weights of one wavelength against each other.

    The incident weight splits into the absorbed and the reflected
    ones, the reflected one into the blocked and the unblocked ones,
    and the absorbed share is 1 - reflectivity for every photon.
    """
    w_i, w_rho_m, w_rho_p, w_b_m, w_b_p = w_loss[:5]
    np.testing.assert_allclose(w_rho_m + w_rho_p, w_i, rtol=1e-9)
    np.testing.assert_allclose(w_b_m + w_b_p, w_rho_p, rtol=1e-9)
    np.testing.assert_allclose(1.0 - w_rho_m / w_i, reflectivity, rtol=1e-6)


def test_loss_weights_single_band(sg: Smartg) -> None:
    """A single wavelength keeps the flat loss weights."""
    ds = _run(sg, 550.0)
    assert ds["wLoss"].dims == ("index",)
    assert ds["wLoss"].shape == (7,)
    _check_loss_identities(ds["wLoss"].values, 0.88)


def test_loss_weights_per_band(two_bands: xr.Dataset) -> None:
    """Each wavelength has its own loss weights."""
    w_loss = two_bands["wLoss"]
    assert w_loss.dims == ("index", "wavelength")
    assert w_loss.shape == (7, WAVELENGTHS.size)
    assert two_bands["wLoss2"].dims == ("index", "wavelength")
    for ilam in range(WAVELENGTHS.size):
        _check_loss_identities(w_loss.values[:, ilam], 0.88)


def _efficiencies(text: str) -> dict[str, float]:
    """Read the efficiencies nopt_view printed."""
    values = {}
    for line in text.splitlines():
        name, _, rest = line.partition(" =")
        if rest and name.startswith("n"):
            values[name] = float(rest.split(",")[0])
    return values


def test_nopt_view_weights_the_bands(
    sg: Smartg, capsys: pytest.CaptureFixture[str]
) -> None:
    """nopt_view weights the loss weights of each band by mtoa.

    The mirrors reflect 0.9 and 0.5 of the two wavelengths, so the
    reflection efficiency is the mean of the two, each weighted by
    the mtoa share and by the mean incident weight of its photons.
    Over a molecular atmosphere, the latter differ between the bands.
    """
    reflectivity = np.array([0.9, 0.5])
    ds = _run(sg, WAVELENGTHS, Atm1D("afglt"), reflectivity)
    w_loss = ds["wLoss"].values
    for ilam in range(WAVELENGTHS.size):
        _check_loss_identities(w_loss[:, ilam], reflectivity[ilam])
    incident = w_loss[0] / ds["norm_npho"].values

    def nref(mtoa: np.ndarray) -> float:
        share = mtoa / mtoa.sum() * incident
        return float(np.sum(share * reflectivity) / np.sum(share))

    capsys.readouterr()
    for mtoa in (np.array([3.0, 1.0]), None):
        nopt_view(ds, mtoa=mtoa)
        printed = _efficiencies(capsys.readouterr().out)
        expected = nref(np.ones(2) if mtoa is None else mtoa)
        np.testing.assert_allclose(printed["nref"], expected, atol=2e-6)
    with pytest.raises(ValueError, match="mtoa"):
        nopt_view(ds, mtoa=np.array([1.0, 2.0, 3.0]))


def test_transparent_atmosphere(sg: Smartg) -> None:
    """Without extinction the scene has finite weights, and no losses.

    A layer that does not scatter made the weight of every photon
    reaching an object NaN: the fraction of the layer above the hit was
    computed from its scattering optical depth, 0/0.
    """
    ds = _run(sg, 550.0, _transparent())
    w_loss = ds["wLoss"].values
    assert np.all(np.isfinite(w_loss))
    # nearly every photon launched toward a heliostat reaches one with
    # its full weight, and every one reflected toward the receiver
    # reaches it
    assert w_loss[0] > 0.9 * float(ds["norm_npho"].sum())
    _check_loss_identities(w_loss, 0.88)
    np.testing.assert_allclose(ds["cat_w"].values[2], w_loss[6], rtol=1e-6)

