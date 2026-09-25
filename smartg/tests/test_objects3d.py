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


def _mirror(
    half: float,
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reflectivity: float = 1.0,
    corner_z: float = 0.0,
) -> Entity:
    """Return a square heliostat of the given half-width.

    Its corners are at corner_z in its own frame.
    """
    return Entity(
        name="reflector",
        material_front=Mirror(reflectivity=reflectivity),
        material_back=Matte(),
        geo=Plane(
            p1=gc.Point(-half, -half, corner_z),
            p2=gc.Point(half, -half, corner_z),
            p3=gc.Point(-half, half, corner_z),
            p4=gc.Point(half, half, corner_z),
        ),
        transformation=Transformation(
            rotation=np.array(rotation), translation=np.array(translation)
        ),
    )


def _run_rf(
    sg: Smartg,
    objects: list[Entity],
    th_deg: float = 0.0,
    n_photons: float = 1e6,
) -> xr.Dataset:
    """Run a scene in the RF mode in a transparent atmosphere.

    Over a black ground.
    """
    return sg.run(
        wavelength=550.0,
        atmosphere=_transparent(),
        surface=LambSurface(alb=AlbedoCst(0.0)),
        th_deg=th_deg,
        n_photons=n_photons,
        my_objects=objects,
        cus_l=CusForward(mode="RF"),
        seed=SEED,
        xblock=XBLOCK,
        xgrid=XGRID,
        progress=False,
    )


def _two_heliostats(
    reflectivity: float,
) -> tuple[list[Entity], float]:
    """Return a large and a small heliostat, and a receiver.

    Under a zenith sun, the large heliostat, 10 m wide and tilted by
    22.5 degrees, reflects on the receiver 50 m away; the small one,
    2 m wide and tilted by -30 degrees, reflects away from it. Also
    return the area the large one projects toward the sun, in m².
    """
    tilt = 22.5
    large = _mirror(0.005, (0.0, 0.0, 0.005), (0.0, tilt, 0.0), reflectivity)
    small = _mirror(
        0.001, (-0.1, 0.0, 0.005), (0.0, -30.0, 0.0), reflectivity
    )
    # the receiver faces the beam of the large heliostat
    beam = np.array(
        [np.sin(np.radians(2 * tilt)), 0.0, np.cos(np.radians(2 * tilt))]
    )
    centre = np.array([0.0, 0.0, 0.005]) + 0.05 * beam
    facing = float(np.degrees(np.arctan2(-beam[0], -beam[2])))
    receiver = _receiver(
        0.008,
        (float(centre[0]), float(centre[1]), float(centre[2])),
        (0.0, facing, 0.0),
        tc=0.002,
    )
    projected = 10.0**2 * float(np.cos(np.radians(tilt)))
    return [large, small, receiver], projected


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


def test_translation_along_x_only(sg: Smartg) -> None:
    """An object translated along x only is intersected where it is.

    The kernel applied the translation of an object only when its y or
    its z translation was positive: this receiver on the ground was
    intersected at the origin, out of the launch field.
    """
    half = 0.002
    receiver = _receiver(half, (0.05, 0.0, 0.0))
    ds = _run_ff(sg, [receiver], _transparent(), 4 * half, (0.05, 0.0))
    area = (2 * half * 1e3) ** 2
    np.testing.assert_allclose(ds["cat_irr"].values[0], area, rtol=0.01)


def test_run_without_objects_after_objects(sg: Smartg) -> None:
    """A run with objects leaves no trace in a run without objects.

    The object constants of the module stayed from one run to the next:
    after the RF run, a run without objects kept looking for objects in
    its empty tables, and killed every photon leaving TOA unhit.
    """

    def clear_sky() -> xr.Dataset:
        return sg.run(
            wavelength=550.0,
            atmosphere=Atm1D("afglt"),
            th_deg=SZA,
            n_photons=1e5,
            seed=SEED,
            xblock=XBLOCK,
            xgrid=XGRID,
            progress=False,
        )

    before = clear_sky()["I_up (TOA)"].values
    _run(sg, 550.0)
    after = clear_sky()["I_up (TOA)"].values
    assert np.all(np.isfinite(after))
    assert after.mean() > 0.0
    np.testing.assert_allclose(after.mean(), before.mean(), rtol=0.02)


def test_rf_reflector_on_the_ground(sg: Smartg) -> None:
    """The RF launch aims at a heliostat at z = 0 too.

    The offset from the heliostat up to TOA was computed only for a
    heliostat with a z translation: at z = 0, the launch positions were
    left uninitialised in double precision.
    """
    mirror = _mirror(0.002, (0.05, 0.02, 0.0))
    receiver = _receiver(0.002, (0.05, -0.3, 0.05))
    ds = _run_rf(sg, [mirror, receiver], th_deg=30.0, n_photons=1e5)
    # every photon reaches the heliostat, but for the rounding of the
    # float launch positions 70 km away along the sun direction
    incident = ds["wLoss"].values[0] / float(ds["norm_npho"].sum())
    assert 0.99 < incident <= 1.0 + 1e-9


def test_rf_draws_heliostats_by_projected_area(sg: Smartg) -> None:
    """The RF launch draws each heliostat by its projected area.

    As the host normalises every photon by the sum of the projected
    areas. Drawn with the same probability, the large heliostat, 96 %
    of the projected area, got half of the photons, and the receiver
    about half of the power it reflects.
    """
    reflectivity = 0.9
    objects, projected = _two_heliostats(reflectivity)
    ds = _run_rf(sg, objects)
    np.testing.assert_allclose(
        ds["cat_irr"].values[0], reflectivity * projected, rtol=2e-3
    )


def test_cosine_efficiency_weighs_the_heliostat_areas(sg: Smartg) -> None:
    """The cosine efficiency n_cos weighs each heliostat by its area.

    It was the plain mean of the cosines: 0.895 instead of 0.922 for
    the large and the small heliostat, whose projected areas are 96 %
    of the total.
    """
    objects, _ = _two_heliostats(1.0)
    ds = _run_rf(sg, objects, n_photons=1e4)
    areas = np.array([10.0, 2.0]) ** 2
    cosines = np.cos(np.radians([22.5, 30.0]))
    np.testing.assert_allclose(
        float(ds.attrs["n_cos"]),
        np.sum(areas * cosines) / np.sum(areas),
        rtol=1e-6,
    )
