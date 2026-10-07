"""GPU tests of the 3D objects on a solar tower power (STP) scene.

The quick start scene of demo_notebook_objects.py, four heliostats
reflecting the sun on a receiver over a desert aerosol and a
Lambertian ground, run in the restricted forward (RF) mode at two
wavelengths. The tests check the bookkeeping of the receiver: the
receiver image of each photon category against the category weights,
per wavelength and summed, and the weights of the optical losses at
the heliostats, per wavelength, down to the efficiencies of
nopt_view. Every test runs with the fast plane-parallel move and with
the alternative one (alt_pp), which follows the photons layer by
layer. The slow tier runs them again with the DatomicAdd fallback of
the GPUs without a double precision atomicAdd, forced on the current
GPU, in the fast move.
"""

from typing import Any

import geoclide as gc
import numpy as np
import pytest
import xarray as xr
from pycuda.compiler import SourceModule

import smartg.smartg as smartg_mod
from smartg.albedo import AlbedoCst
from smartg.atmosphere import AerOPAC, Atm1D
from smartg.objects3d import (
    CusBackward,
    CusForward,
    Entity,
    LambMirror,
    MaterialType,
    Matte,
    Mirror,
    Plane,
    Spheric,
    Transformation,
)
from smartg.smartg import LocalEstimate, Smartg, _od_at_altitude
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


def _transparent(grid: list[float] | None = None) -> Atm1D:
    """Return an atmosphere that neither scatters nor absorbs."""
    return Atm1D(
        "afglt", grid=grid, tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0
    )


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
    cftz: float = 0.0,
    th_deg: float = 0.0,
    **kwargs: Any,
) -> xr.Dataset:
    """Run a scene in the FF mode, zenith sun by default, black ground.

    The photons are launched over a square of side field, whose direct
    beam is centred on centre at the ground, cftz km above TOA, and the
    direct ones are counted by the receivers.
    """
    return sg.run(
        wavelength=wavelength,
        atmosphere=atmosphere,
        surface=LambSurface(alb=AlbedoCst(0.0)),
        th_deg=th_deg,
        n_photons=1e6,
        my_objects=objects,
        cus_l=CusForward(
            cfx=field,
            cfy=field,
            cftx=centre[0],
            cfty=centre[1],
            cftz=cftz,
            mode="FF",
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
    sun_disc: float = 0.0,
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
        sun_disc=sun_disc,
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
    seed: int = SEED,
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
        seed=seed,
        xblock=XBLOCK,
        xgrid=XGRID,
        progress=False,
    )


@pytest.fixture(
    scope="module",
    params=[
        "native",
        "alt_pp",
        pytest.param("datomicadd", marks=pytest.mark.slow),
    ],
)
def sg(request: pytest.FixtureRequest) -> Smartg:
    """Compile the kernel, in the fast or the alternative (alt_pp) move.

    The fast one natively or with the DatomicAdd fallback. The kernel is
    compiled when the Smartg is built, so the compiler is patched for
    that alone, and never while a native one is built.
    """
    if request.param in ("native", "alt_pp"):
        assert smartg_mod.SourceModule is SourceModule
        return Smartg(
            double=True, obj3d=True, alt_pp=request.param == "alt_pp"
        )
    source_module = smartg_mod.SourceModule

    def forced(*args: Any, **kwargs: Any) -> Any:
        options = list(kwargs.pop("options", []))
        return source_module(
            *args, options=[*options, "-DFORCE_DATOMICADD"], **kwargs
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(smartg_mod, "SourceModule", forced)
        sg = Smartg(double=True, obj3d=True)
    return sg


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


@pytest.mark.parametrize("is_atm", [1, 0])
def test_objects_need_an_atmosphere(sg: Smartg, is_atm: int) -> None:
    """A scene without atmosphere is refused before the kernel runs.

    The run stopped on an assertion once the kernel was done, and with
    is_atm=0 the kernel never ended.
    """
    with pytest.raises(ValueError, match="need an atmosphere"):
        sg.run(
            wavelength=550.0,
            surface=LambSurface(alb=AlbedoCst(0.0)),
            my_objects=[_receiver(0.002, (0.0, 0.0, 0.0))],
            is_atm=is_atm,
            progress=False,
        )


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


@pytest.mark.parametrize(
    ("height", "corner_z", "sun_disc"),
    [(0.0, 0.0, 0.0), (0.005, 0.01, 0.0), (0.0, 0.0, 0.266)],
    ids=["ground", "raised corners", "ground sun disc"],
)
def test_rf_reflector_on_the_ground(
    sg: Smartg, height: float, corner_z: float, sun_disc: float
) -> None:
    """The RF launch aims at a heliostat at z = 0 too.

    The offset from the heliostat up to TOA was computed only for a
    heliostat with a z translation: at z = 0, the launch positions were
    left uninitialised in double precision. The launch also aims at the
    corners of a heliostat raised in its own frame. With the sun disc,
    12 % of the hits at z = 0 were lost: the hit, computed along the
    ray from TOA, came out below the ground tolerance.
    """
    mirror = _mirror(0.002, (0.05, 0.02, height), corner_z=corner_z)
    receiver = _receiver(0.002, (0.05, -0.3, 0.05))
    ds = _run_rf(
        sg, [mirror, receiver], th_deg=30.0, n_photons=1e5, sun_disc=sun_disc
    )
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


def test_plane_corners_above_their_origin(sg: Smartg) -> None:
    """A plane is intersected at the z of its corners.

    The kernel dropped the z of the corners: this blocker, 15 m high
    with its corners 10 m above its origin, was intersected 5 m high,
    below the receiver it shades.
    """
    receiver = _receiver(0.002, (0.0, 0.0, 0.01))
    half = 0.004
    blocker = Entity(
        name="environment",
        material_front=Matte(reflectivity=0.0),
        material_back=Matte(reflectivity=0.0),
        geo=Plane(
            p1=gc.Point(-half, -half, 0.01),
            p2=gc.Point(half, -half, 0.01),
            p3=gc.Point(-half, half, 0.01),
            p4=gc.Point(half, half, 0.01),
        ),
        transformation=Transformation(
            translation=np.array([0.0, 0.0, 0.005])
        ),
    )
    lit = _run_ff(sg, [receiver], _transparent(), 2 * half)
    shaded = _run_ff(sg, [receiver, blocker], _transparent(), 2 * half)
    np.testing.assert_allclose(lit["cat_irr"].values[0], 16.0, rtol=0.01)
    assert shaded["cat_irr"].values[0] == 0.0


def _sphere_shadow(
    sg: Smartg,
    rotation: tuple[float, float, float],
    front: MaterialType,
    back: MaterialType,
    height: float = 0.01,
    half: float = 0.003,
) -> tuple[xr.Dataset, float]:
    """Run a sphere over a receiver; return the output and the shadow.

    A sphere of radius 2 m, its centre at height, over a square
    receiver of half-width half 10 cm above the ground, under a zenith
    sun: the direct sun reaches the receiver but for the shadow of the
    sphere, a disc of radius 2 m. Also return the direct power expected
    on the receiver, in m².
    """
    radius = 0.002
    sphere = Entity(
        name="environment",
        material_front=front,
        material_back=back,
        geo=Spheric(radius=radius),
        transformation=Transformation(
            rotation=np.array(rotation),
            translation=np.array([0.0, 0.0, height]),
        ),
    )
    receiver = _receiver(half, (0.0, 0.0, 0.0001))
    ds = _run_ff(sg, [sphere, receiver], _transparent(), 2 * half)
    expected = (2 * half * 1e3) ** 2 - np.pi * (radius * 1e3) ** 2
    return ds, expected


def test_rotated_sphere_casts_its_whole_shadow(sg: Smartg) -> None:
    """A sphere rotated about y is intersected on its whole surface.

    The kernel bounded the transformed box of a sphere by 5 of its 8
    corners: rotated by 30 degrees about y, the part of the sphere
    beyond 0.37 of its radius along x was never tested, 27 % of its
    shadow under a zenith sun.
    """
    ds, expected = _sphere_shadow(
        sg, (0.0, 30.0, 0.0), Matte(reflectivity=0.0), Matte(reflectivity=0.0)
    )
    np.testing.assert_allclose(ds["cat_irr"].values[0], expected, rtol=5e-3)


def test_sphere_reflects_with_its_front(sg: Smartg) -> None:
    """The outside of a sphere is its front.

    The base normal of a sphere was left at zero, so every hit took
    its back material: this white Lambertian sphere with a black back
    absorbed everything. Its light reaches the receiver around its
    shadow, 50 cm below it: 2.5 % of the power it intercepts, 0.315
    m², by a Monte Carlo integration of its Lambertian reflection
    around the normal of each hit.
    """
    ds, expected = _sphere_shadow(
        sg,
        (0.0, 0.0, 0.0),
        LambMirror(reflectivity=1.0),
        Matte(),
        height=0.0025,
        half=0.006,
    )
    cat_irr = ds["cat_irr"].values
    np.testing.assert_allclose(cat_irr[1], expected, rtol=5e-3)
    np.testing.assert_allclose(cat_irr[3], 0.315, rtol=0.1)


def _flipped_plane(
    back: MaterialType, tilt: float, translation: tuple[float, float, float]
) -> Entity:
    """Return a 10 m square facing down, its back tilted by tilt."""
    return Entity(
        name="environment",
        material_front=Matte(reflectivity=0.0),
        material_back=back,
        geo=_plane(0.005, 0.005),
        transformation=Transformation(
            rotation=np.array([0.0, 180.0 + tilt, 0.0]),
            translation=np.array(translation),
        ),
    )


def test_lambertian_back_face_reflects_back(sg: Smartg) -> None:
    """The back of a Lambertian plane reflects to the back side.

    It was sampled around the front normal: the light hitting this
    upward back face went through the plane, onto the receiver below.
    """
    plane = _flipped_plane(LambMirror(reflectivity=1.0), 0.0, (0, 0, 0.01))
    receiver = _receiver(0.005, (0.0, 0.0, 0.001))
    ds = _run_ff(sg, [plane, receiver], _transparent(), 0.01)
    assert ds["cat_irr"].values[0] == 0.0


def test_rough_mirror_back_face_reflects(sg: Smartg) -> None:
    """The back of a rough mirror reflects like its front.

    Its microfacets were drawn around the front normal, so that no
    reflection was ever found and every photon hitting it was lost.
    The back face, tilted by 22.5 degrees, sends the zenith sun onto
    the receiver 50 m away.
    """
    tilt = 22.5
    back = Mirror(reflectivity=1.0, roughness=0.02)
    plane = _flipped_plane(back, tilt, (0.0, 0.0, 0.005))
    objects, projected = _two_heliostats(1.0)
    receiver = objects[2]
    ds = _run_ff(sg, [plane, receiver], _transparent(), 0.012)
    power = ds["cat_irr"].values[0]
    assert 0.95 * projected < power < 1.005 * projected


@pytest.mark.parametrize("beer", [0, 1])
def test_direct_sun_on_a_receiver_in_absorbing_air(
    sg: Smartg, beer: int
) -> None:
    """The direct sun reaches a receiver attenuated by exp(-tau).

    With either treatment of the absorption. With beer=0 a photon
    reaching an object before any collision was still multiplied by
    the single scattering albedo of its layer, 0.81 at the ground in
    this urban aerosol.
    """
    atmosphere = Atm1D("afglt", comp=[AerOPAC("urban", 0.5, 550.0)])
    od = atmosphere.calc(550.0)["OD_atm"].values.ravel()[-1]
    half = 0.002
    receiver = _receiver(half, (0.0, 0.0, 0.0))
    ds = _run_ff(sg, [receiver], atmosphere, 4 * half, beer=beer)
    area = (2 * half * 1e3) ** 2
    np.testing.assert_allclose(
        ds["cat_irr"].values[1], area * np.exp(-od), rtol=0.015
    )


def test_ff_launch_below_toa(sg: Smartg) -> None:
    """A FF launch moved down by cftz starts at the optical depth there.

    It started at the position PZd but with the layer and the optical
    depth of TOA: the direct sun on the ground was attenuated by the
    whole atmosphere, and more, instead of the 1.5 km below the launch.
    """
    atmosphere = Atm1D("afglt")
    profile = atmosphere.calc(450.0)
    z_atm = profile["z_atm"].values
    od_atm = profile["OD_atm"].values
    launch = 1.5
    od = od_atm[0, -1] - _od_at_altitude(z_atm, od_atm, launch)[0]
    half = 0.002
    receiver = _receiver(half, (0.0, 0.0, 0.0))
    ds = _run_ff(
        sg,
        [receiver],
        atmosphere,
        4 * half,
        wavelength=450.0,
        cftz=launch - z_atm[0],
    )
    area = (2 * half * 1e3) ** 2
    np.testing.assert_allclose(
        ds["cat_irr"].values[1], area * np.exp(-od), rtol=0.01
    )


@pytest.fixture(scope="module", params=[False, True], ids=["fast", "alt_pp"])
def sg_back(request: pytest.FixtureRequest) -> Smartg:
    """Compile the backward kernel, in both plane-parallel moves."""
    return Smartg(double=True, obj3d=True, back=True, alt_pp=request.param)


def _absorbing(
    grid: list[float], od_abs: np.ndarray, wavelength: float = 550.0
) -> xr.Dataset:
    """Return a profile that only absorbs.

    od_abs is the absorption optical depth from TOA down to each level
    of the grid.
    """
    profile = _transparent(grid).calc(wavelength)
    for name in ("OD_atm", "OD_abs_atm"):
        profile[name].values[:] = od_abs
    profile["ssa_atm"].values[0, 1:][np.diff(od_abs) > 0] = 0.0
    return profile


def _absorbing_layer(
    grid: list[float], layer: int, od_abs: float, wavelength: float = 550.0
) -> xr.Dataset:
    """Return a profile that only absorbs, od_abs in one of its layers.

    The layer counts from the top, the first one is 1.
    """
    od = np.zeros(len(grid))
    od[layer:] = od_abs
    return _absorbing(grid, od, wavelength)


def _br_facing_the_sun(
    sg: Smartg,
    profile: xr.Dataset,
    sza: float,
    half: float,
    centre_z: float,
    seed: int = SEED,
) -> xr.Dataset:
    """Run a BR receiver facing the sun, its photons sent to the sun."""
    v_sun = gc.ang2vec(sza, 0.0, vec_view="nadir")
    to_sun = -gc.normalize(v_sun)
    tilt = float(np.degrees(np.arctan2(to_sun.x, to_sun.z)))
    normal = gc.normalize(gc.get_rotate_y_tf(tilt)(gc.Vector(0.0, 0.0, 1.0)))
    np.testing.assert_allclose(
        [normal.x, normal.y, normal.z], [to_sun.x, to_sun.y, to_sun.z],
        atol=1e-12,
    )
    receiver = _receiver(half, (0.0, 0.0, centre_z), (0.0, tilt, 0.0))
    return sg.run(
        wavelength=550.0,
        atmosphere=profile,
        surface=LambSurface(alb=AlbedoCst(0.0)),
        n_photons=1e6,
        my_objects=[receiver],
        cus_l=CusBackward(
            normal=normal, receiver_fov=0.0, mode="BR", receiver=receiver,
            v_sun=v_sun, sun_fov=0.266,
        ),
        direct=True,
        seed=seed,
        xblock=XBLOCK,
        xgrid=XGRID,
        progress=False,
    )


def test_br_starts_in_the_layer_of_its_receiver_point(sg_back: Smartg) -> None:
    """A BR photon starts with the optical depth of its point.

    The receiver faces the sun across the bottom of a layer that only
    absorbs, above clear air, and every photon goes straight to the sun:
    its weight is the transmission from its point of the receiver, whose
    mean is the one of exp(-tau(z)/mu) over the receiver. The photons
    started with the layer and the optical depth of the receiver centre,
    on the boundary: the fast move gave all of them the transmission of
    the centre, the alternative one gave the lower half the absorption
    of the layer above.
    """
    sza, half, centre_z = 60.0, 0.01, 0.1
    grid = [120.0, 0.2, 0.1, 0.0]
    profile = _absorbing_layer(grid, 2, 2.0)
    ds = _br_facing_the_sun(sg_back, profile, sza, half, centre_z)
    mean_weight = ds["cat_w"].values[1] / ds["cat_PhNb"].values[1]
    # the altitude is uniform over the tilted receiver, and the optical
    # depth from TOA linear with it inside a layer
    mu = np.cos(np.radians(sza))
    height = half * np.sin(np.radians(sza))
    z = centre_z + np.linspace(-height, height, 20001)
    od = np.interp(z, np.array(grid)[::-1], profile["OD_atm"].values[0, ::-1])
    expected = np.exp(-od / mu).mean()
    np.testing.assert_allclose(mean_weight, expected, rtol=2e-3)


# levels of a profile that absorbs more and more toward the ground, and
# its absorption optical depth from TOA, linear inside each layer
GRID_ABS = [120.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.0]


def _od_abs(grid: list[float]) -> np.ndarray:
    """Return the optical depth of GRID_ABS at the levels of grid."""
    levels = np.array(GRID_ABS)
    od = 3.0 * (np.exp(-levels / 8.0) - np.exp(-levels[0] / 8.0))
    return np.interp(grid, levels[::-1], od[::-1])


@pytest.mark.parametrize("split", [False, True], ids=["grid", "split"])
@pytest.mark.parametrize(
    "altitude", [0.0, 0.35, 0.5], ids=["ground", "in layer", "on level"]
)
def test_direct_sun_through_absorbing_layers(
    sg: Smartg, altitude: float, split: bool
) -> None:
    """The weight of a direct photon is the transmission to its object.

    Under a sun at 60 degrees, in a profile that only absorbs, every
    photon launched inside a black receiver reaches it with the same
    weight, exp(-tau(z)/mu), through all the layers above it. The
    same with every layer split in two, which the alternative move
    walks one by one.
    """
    grid = GRID_ABS
    if split:
        middles = (np.array(GRID_ABS[:-1]) + np.array(GRID_ABS[1:])) / 2
        grid = sorted([*GRID_ABS, *middles], reverse=True)
    profile = _absorbing(grid, _od_abs(grid))
    sza, half = 60.0, 0.002
    # the direct beam on the receiver is centred on it
    v_sun = gc.normalize(gc.ang2vec(sza, 0.0, vec_view="nadir"))
    centre = (v_sun.x * altitude / -v_sun.z, v_sun.y * altitude / -v_sun.z)
    ds = _run_ff(
        sg, [_receiver(half, (0.0, 0.0, altitude))], profile, half,
        centre=centre, th_deg=sza,
    )
    weight = ds["cat_w"].values[1] / ds["cat_PhNb"].values[1]
    expected = np.exp(-_od_abs([altitude])[0] / np.cos(np.radians(sza)))
    np.testing.assert_allclose(weight, expected, rtol=1e-5)


def test_fast_and_alt_pp_moves_agree_on_the_receiver() -> None:
    """The two plane-parallel moves give the same receiver categories.

    The RF scene over its desert aerosol, each move with its own seed:
    the categories agree within the Monte Carlo error of the receiver
    counts (cat_errAbs), exact in this mode, at 5 sigma.
    """
    fast = _run(Smartg(double=True, obj3d=True), 550.0, seed=11)
    alt = _run(Smartg(double=True, obj3d=True, alt_pp=True), 550.0, seed=12)
    err = np.hypot(fast["cat_errAbs"].values, alt["cat_errAbs"].values)
    counted = err > 0
    # the sum and the light reflected by the heliostats alone
    assert counted[0] and counted[2]
    z = (fast["cat_irr"].values - alt["cat_irr"].values)[counted]
    assert np.all(np.abs(z / err[counted]) < 5)


def _horizontal_view_of_a_wall(sg: Smartg, distance: float) -> float:
    """Return the mean weight of the local estimate of a wall.

    A BR receiver at 0.5 km sends its photons along its normal
    (receiver_fov=0), horizontally (v.z = 6e-17), to a Lambertian wall
    at the given distance, whose local estimate looks at the sun. The
    atmosphere only absorbs, uniformly below 1 km, with 1 per km.
    """
    altitude, sza = 0.5, 60.0
    v_sun = gc.normalize(gc.ang2vec(sza, 0.0, vec_view="nadir"))
    # the wall faces the receiver and the sun
    side = 1.0 if v_sun.x > 0 else -1.0
    receiver = _receiver(0.002, (0.0, 0.0, altitude), (0.0, side * 90.0, 0.0))
    wall = Entity(
        name="environment",
        material_front=LambMirror(reflectivity=0.5),
        material_back=Matte(),
        geo=_plane(0.005, 0.005),
        transformation=Transformation(
            rotation=np.array([0.0, -side * 90.0, 0.0]),
            translation=np.array([side * distance, 0.0, altitude]),
        ),
    )
    normal = gc.normalize(
        gc.get_rotate_y_tf(side * 90.0)(gc.Vector(0.0, 0.0, 1.0))
    )
    ds = sg.run(
        wavelength=550.0,
        atmosphere=_absorbing_layer([120.0, 1.0, 0.0], 2, 1.0),
        surface=LambSurface(alb=AlbedoCst(0.0)),
        n_photons=1e6,
        my_objects=[wall, receiver],
        cus_l=CusBackward(
            normal=normal, receiver_fov=0.0, mode="BR", receiver=receiver,
            v_sun=v_sun,
        ),
        le=LocalEstimate(th_deg=[sza], phi_deg=[0.0]),
        le_fov=0.266,
        direct=True,
        seed=SEED,
        xblock=XBLOCK,
        xgrid=XGRID,
        progress=False,
    )
    return float(ds["cat_w"].values[0] / ds["cat_PhNb"].values[0])


def test_horizontal_ray_to_a_wall_in_absorbing_air(sg_back: Smartg) -> None:
    """A horizontal photon reaches an object through its absorption.

    Moving the wall 0.5 km away multiplies the light of its local
    estimate by exactly exp(-0.5), the absorption of the longer path.
    The fast move divided the difference of the absorption optical
    depths at the two ends, at the same altitude, by v.z = 6e-17: the
    weight came out 0 or 1.
    """
    near = _horizontal_view_of_a_wall(sg_back, 0.1)
    far = _horizontal_view_of_a_wall(sg_back, 0.6)
    assert near > 0.0
    # within the noise of the cone of the local estimate, 2e-4
    np.testing.assert_allclose(far / near, np.exp(-0.5), rtol=1e-3)


def _roof(half: float, altitude: float) -> Entity:
    """Return a black horizontal square, of half-width half, in km."""
    return Entity(
        name="environment",
        material_front=Matte(reflectivity=0.0),
        material_back=Matte(reflectivity=0.0),
        geo=_plane(half, half),
        transformation=Transformation(
            rotation=np.array([0.0, 0.0, 0.0]),
            translation=np.array([0.0, 0.0, altitude]),
        ),
    )


@pytest.mark.parametrize("roof", [True, False], ids=["roof", "no roof"])
def test_local_estimate_masked_by_a_black_roof(sg: Smartg, roof: bool) -> None:
    """A black roof over the scene masks every local estimate to TOA.

    The photons are launched at 20 km, below a black roof at 30 km, into
    a dusty atmosphere over a bright ground: every path of the local
    estimate to TOA crosses the roof, so the TOA radiance is exactly 0.
    The roof spans 1e5 km: at that altitude a photon scattered sideways
    travels about 1000 km, and some went round a roof of 2000 km. The
    alternative move tested the mask once the virtual photon was at TOA,
    from where the ray missed the roof. Without the roof (a small object
    far away) the radiance is not 0.
    """
    obstacle = (
        _roof(5e4, 30.0) if roof else _receiver(0.001, (500.0, 500.0, 0.0))
    )
    ds = sg.run(
        wavelength=550.0,
        atmosphere=Atm1D("afglt", comp=[AerOPAC("desert", 0.5, 550.0)]),
        surface=LambSurface(alb=AlbedoCst(0.3)),
        th_deg=0.0,
        n_photons=1e5,
        my_objects=[obstacle],
        cus_l=CusForward(cfx=1.0, cfy=1.0, cftz=20.0 - 120.0, mode="FF"),
        le=LocalEstimate(th_deg=[0.0, 30.0, 60.0], phi_deg=[0.0]),
        seed=SEED,
        xblock=XBLOCK,
        xgrid=XGRID,
        progress=False,
    )
    radiance = ds["I_up (TOA)"].values
    if roof:
        np.testing.assert_array_equal(radiance, 0.0)
    else:
        assert np.all(radiance > 0.0)


def _br_local_estimate(alt_pp: bool, seeds: list[int]) -> np.ndarray:
    """Return the receiver flux of the BR scene with a local estimate.

    One run per seed, the local estimate toward the sun with its disc.
    """
    sg = Smartg(double=True, obj3d=True, back=True, alt_pp=alt_pp)
    objects = _scene()
    normal = gc.normalize(
        gc.get_rotate_y_tf(-101.5)(gc.Vector(0.0, 0.0, 1.0))
    )
    w2 = 0.5
    atmosphere = Atm1D(
        "afglms", comp=[AerOPAC("desert", 0.25, 550.0)], p0=877, tcwp=1.2
    )
    flux = []
    for seed in seeds:
        ds = sg.run(
            wavelength=550.0,
            atmosphere=atmosphere,
            surface=LambSurface(alb=AlbedoCst(0.25)),
            n_photons=5e5,
            my_objects=objects,
            interval=[[-w2, -w2, -0.005], [w2, w2, 0.125]],
            cus_l=CusBackward(
                normal=normal, receiver_fov=90.0, mode="BR",
                receiver=objects[-1],
                v_sun=gc.ang2vec(SZA, 0.0, vec_view="nadir"),
            ),
            le=LocalEstimate(th_deg=[SZA], phi_deg=[0.0]),
            le_fov=0.266,
            direct=True,
            seed=seed,
            xblock=XBLOCK,
            xgrid=XGRID,
            progress=False,
        )
        flux.append(ds["cat_irr"].values[0])
    return np.array(flux)


def test_br_local_estimate_moves_agree() -> None:
    """The local estimate of the BR mode is the same in both moves.

    The receiver of the RF scene, seen from the sun by local estimate
    through the dust, 5 runs per move with their own seeds. The
    alternative move attenuated the virtual photon on its way to TOA
    and again in countPhotonObj3D, from a stale optical depth: about a
    third too little light here.
    """
    fast = _br_local_estimate(False, [21, 22, 23, 24, 25])
    alt = _br_local_estimate(True, [31, 32, 33, 34, 35])
    err = np.hypot(fast.std(ddof=1), alt.std(ddof=1)) / np.sqrt(5)
    assert fast.mean() > 0.0
    assert abs(fast.mean() - alt.mean()) < 5 * err


def _br_horizon(alt_pp: bool, seeds: list[int]) -> np.ndarray:
    """Return the light of a BR receiver facing the horizon, per seed.

    Its photons leave horizontally (receiver_fov=0, v.z = 6e-17) into an
    urban aerosol, which absorbs (beer=1), and the local estimates of
    their collisions look at the sun. The light is the sum of their
    weights per photon launched.
    """
    sg = Smartg(double=True, obj3d=True, back=True, alt_pp=alt_pp)
    receiver = _receiver(0.002, (0.0, 0.0, 0.5), (0.0, 90.0, 0.0))
    normal = gc.normalize(
        gc.get_rotate_y_tf(90.0)(gc.Vector(0.0, 0.0, 1.0))
    )
    atmosphere = Atm1D("afglt", comp=[AerOPAC("urban", 0.5, 550.0)])
    light = []
    for seed in seeds:
        ds = sg.run(
            wavelength=550.0,
            atmosphere=atmosphere,
            surface=LambSurface(alb=AlbedoCst(0.0)),
            n_photons=2e5,
            my_objects=[receiver],
            cus_l=CusBackward(
                normal=normal, receiver_fov=0.0, mode="BR",
                receiver=receiver,
                v_sun=gc.ang2vec(60.0, 0.0, vec_view="nadir"),
            ),
            le=LocalEstimate(th_deg=[60.0], phi_deg=[0.0]),
            le_fov=0.266,
            seed=seed,
            xblock=XBLOCK,
            xgrid=XGRID,
            progress=False,
        )
        light.append(ds["cat_w"].values[0] / float(ds["norm_npho"].sum()))
    return np.array(light)


def test_horizontal_photons_collide_along_their_layer() -> None:
    """A horizontal photon collides at tauR along its layer, absorbed.

    The light of the local estimates of photons leaving horizontally is
    the same in both moves, 5 runs each. The fast move divided their
    change of vertical optical depth, a few float ulps, by v.z = 6e-17:
    their distance and their absorption were rounding noise.
    """
    fast = _br_horizon(False, [41, 42, 43, 44, 45])
    alt = _br_horizon(True, [51, 52, 53, 54, 55])
    err = np.hypot(fast.std(ddof=1), alt.std(ddof=1)) / np.sqrt(5)
    assert fast.mean() > 0.0
    assert abs(fast.mean() - alt.mean()) < 5 * err


def test_receiver_cells_tile_the_receiver(sg: Smartg) -> None:
    """Every hit of a receiver lands in one of its cells.

    0.0006 / 0.0001 truncated to 5 cells: the hits past them got the
    index -1, written before the flux map or into the previous
    category. The 6 x 6 cells share the sun evenly.
    """
    half = 0.0003
    receiver = _receiver(half, (0.0, 0.0, 0.0), tc=0.0001)
    ds = _run_ff(sg, [receiver], _transparent(), 4 * half)
    image = ds["C_Receiver"]
    assert image.shape == (9, 6, 6)
    np.testing.assert_allclose(
        image.sum(("X_Cell_Index", "Y_Cell_Index")).values,
        ds["cat_irr"].values,
        rtol=1e-8,
    )
    area = (2 * half * 1e3) ** 2
    np.testing.assert_allclose(ds["cat_irr"].values[0], area, rtol=0.01)
    np.testing.assert_allclose(image.values[0], area / 36, rtol=0.1)
