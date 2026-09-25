"""GPU-free tests of the 3D objects.

The geometry classes, the heliostat generators and the helpers of
smartg.objects3d, and the host side of the object setup of
smartg.smartg, which checks the scene before anything is copied to
the GPU. The kernel side is tested by test_objects3d.py.
"""

import geoclide as gc
import numpy as np
import pytest

from smartg.objects3d import (
    CusBackward,
    CusForward,
    Entity,
    GroupE,
    Heliostat,
    Mirror,
    Plane,
    Spheric,
    Transformation,
    generate_h_a,
    generate_h_p,
    rotate_vector,
)
from smartg.smartg import (
    _check_object_roles,
    _od_at_altitude,
    _receiver_grid,
    _rf_launch_cdf,
)


def _xyz(point: gc.Vector | gc.Point) -> np.ndarray:
    """Return the coordinates of a geoclide vector or point."""
    return np.array([point.x, point.y, point.z], dtype=np.float64)


@pytest.mark.parametrize("order", ["XYZ", "xyz", "zyx", "ZxY"])
def test_rotate_vector_accepts_either_case(order: str) -> None:
    """The rotation order is read in upper or lower case."""
    rotated = rotate_vector(gc.Vector(0.0, 0.0, 1.0), 90.0, 0.0, 0.0, order)
    np.testing.assert_allclose(_xyz(rotated), [0.0, -1.0, 0.0], atol=1e-12)


def test_rotate_vector_default_order() -> None:
    """The default order is XYZ."""
    vector = gc.Vector(1.0, 2.0, 3.0)
    default = rotate_vector(vector, 10.0, 20.0, 30.0)
    xyz = rotate_vector(vector, 10.0, 20.0, 30.0, "XYZ")
    np.testing.assert_allclose(_xyz(default), _xyz(xyz))


def test_rotate_vector_unknown_order() -> None:
    """An order that is not a permutation of XYZ raises ValueError."""
    with pytest.raises(ValueError, match="rotation_order"):
        rotate_vector(gc.Vector(0.0, 0.0, 1.0), 0.0, 0.0, 0.0, "XXY")


def test_entity_copy_keeps_alpha_color() -> None:
    """A copy of an Entity keeps every property, alpha_color too."""
    entity = Entity(color="red", alpha_color=0.1)
    copy = Entity(entity)
    assert copy.color == "red"
    assert copy.alpha_color == 0.1


@pytest.mark.parametrize(
    ("corners", "message"),
    [
        # a trapezoid with every corner on its side of the axes
        (
            {"p3": gc.Point(-0.5, 0.5, 0.0), "p4": gc.Point(0.5, 0.7, 0.0)},
            "rectangle",
        ),
        # a rectangle below the x axis
        (
            {"p3": gc.Point(-0.5, -0.2, 0.0), "p4": gc.Point(0.5, -0.2, 0.0)},
            r"p3\.y > 0, p4\.y > 0",
        ),
        ({"p4": gc.Point(-0.5, 0.5, 0.0)}, r"p4\.x > 0$"),
    ],
)
def test_plane_names_the_violated_condition(
    corners: dict[str, gc.Point], message: str
) -> None:
    """An invalid Plane raises a ValueError naming what is wrong."""
    with pytest.raises(ValueError, match=message):
        Plane(**corners)


@pytest.mark.parametrize(
    "rotation", [[0.0, 0.0, 45.0], [30.0, 0.0, 0.0], [0.0, 60.0, 20.0]]
)
def test_spheric_bbox_holds_the_rotated_sphere(rotation: list[float]) -> None:
    """The bounding box of a rotated sphere holds the whole sphere.

    Without a user box, and after set_transformation too.
    """
    radius, centre = 1.0, np.array([5.0, 0.0, 2.0])
    transformation = Transformation(
        rotation=np.array(rotation), translation=centre
    )
    entity = Entity(geo=Spheric(radius=radius), transformation=transformation)
    moved = Entity(geo=Spheric(radius=radius))
    moved.set_transformation(transformation)
    for sphere in (entity, moved):
        assert np.all(_xyz(sphere.bbox_pmin) <= centre - radius + 1e-12)
        assert np.all(_xyz(sphere.bbox_pmax) >= centre + radius - 1e-12)


def _facet_mirrors(objects: list) -> list:
    """Return the front materials of every facet of the heliostats."""
    return [
        entity.material_front
        for group in objects
        for entity in (group.le if isinstance(group, GroupE) else [group])
    ]


@pytest.mark.parametrize("generator", ["h_p", "h_a"])
def test_generators_take_the_optics_of_heliostat_type(generator: str) -> None:
    """The template reflectivity and roughness reach the facets.

    Unless the generator is given its own.
    """
    template = Heliostat(
        helio_size_x=0.01, helio_size_y=0.01, reflectivity=0.8, roughness=0.01
    )
    receiver = gc.Point(0.0, 0.0, 0.1)

    def generate(
        reflectivity: float | None = None, roughness: float | None = None
    ) -> list:
        if generator == "h_p":
            return generate_h_p(
                heliostat_pos_list=[gc.Point(0.1, 0.0, 0.005)],
                receiver_pos=receiver,
                heliostat_type=template,
                reflectivity=reflectivity,
                roughness=roughness,
            )
        return generate_h_a(
            receiver_pos=receiver,
            min_ang_deg=0.0,
            max_ang_deg=0.0,
            n_heliostats=1,
            heliostat_type=template,
            reflectivity=reflectivity,
            roughness=roughness,
        )

    mirrors = _facet_mirrors(generate())
    assert len(mirrors) == template.n_facets_x * template.n_facets_y
    for mirror in mirrors:
        assert (mirror.reflectivity, mirror.roughness) == (0.8, 0.01)
    for mirror in _facet_mirrors(generate(reflectivity=0.5, roughness=0.0)):
        assert (mirror.reflectivity, mirror.roughness) == (0.5, 0.0)


def test_generators_default_optics() -> None:
    """Without a template, the heliostats are perfect mirrors."""
    (heliostat,) = generate_h_p(
        heliostat_pos_list=[gc.Point(0.1, 0.0, 0.005)],
        receiver_pos=gc.Point(0.0, 0.0, 0.1),
    )
    assert isinstance(heliostat, Entity)
    mirror = heliostat.material_front
    assert isinstance(mirror, Mirror)
    assert (mirror.reflectivity, mirror.roughness) == (1.0, 0.0)


def test_od_at_altitude_interpolates_the_layer() -> None:
    """The optical depth to the heliostats is linear in their layer.

    Between the levels around them, whatever the altitude of the level
    below; an altitude on a level gives its optical depth.
    """
    z_atm = np.array([100.0, 10.0, 2.0, 1.0, 0.0])
    od_atm = np.array(
        [[0.0, 0.05, 0.15, 0.2, 0.3], [0.0, 0.1, 0.3, 0.4, 0.6]]
    )
    for altitude, expected in (
        (1.5, [0.175, 0.35]),
        (0.5, [0.25, 0.5]),
        (2.0, od_atm[:, 2]),
    ):
        np.testing.assert_allclose(
            _od_at_altitude(z_atm, od_atm, altitude), expected
        )
    np.testing.assert_allclose(_od_at_altitude(z_atm, od_atm[:1], 0.0), [0.3])


def test_od_at_altitude_outside_the_profile() -> None:
    """Heliostats below the bottom of the profile raise ValueError."""
    z_atm = np.array([100.0, 50.0, 20.0, 10.0, 5.0, 3.4])
    od_atm = np.linspace(0.0, 0.3, z_atm.size)[None, :]
    with pytest.raises(ValueError, match="outside the atmosphere profile"):
        _od_at_altitude(z_atm, od_atm, 0.005)


@pytest.mark.parametrize(
    ("name", "cus_l"),
    [
        ("receiver", None),
        ("receiver", CusForward(mode="FF")),
        ("reflector", CusForward(mode="RF")),
    ],
)
def test_spheric_receiver_or_rf_reflector_is_refused(
    name: str, cus_l: CusForward | None
) -> None:
    """A receiver, or a reflector in the RF mode, must be a Plane."""
    sphere = Entity(name=name, geo=Spheric(radius=0.01))
    with pytest.raises(ValueError, match="Plane geometry"):
        _check_object_roles([sphere], cus_l)


def test_spheric_reflector_outside_rf_is_accepted() -> None:
    """A sphere reflects in the FF mode and without a launching mode."""
    sphere = Entity(name="reflector", geo=Spheric(radius=0.01))
    for cus_l in (None, CusForward(mode="FF")):
        _check_object_roles([sphere, Entity(name="receiver")], cus_l)


def test_spheric_br_receiver_is_refused() -> None:
    """The receiver of the BR mode must be a Plane."""
    receiver = Entity(name="receiver", geo=Spheric(radius=0.01))
    with pytest.raises(ValueError, match="Plane geometry"):
        CusBackward(receiver=receiver, mode="BR")


def _receiver(x_low: float, x_high: float, half_y: float, tc: float) -> Entity:
    """Return a receiver from x_low to x_high and -half_y to half_y."""
    return Entity(
        name="receiver",
        tc=tc,
        geo=Plane(
            p1=gc.Point(x_low, -half_y, 0.0),
            p2=gc.Point(x_high, -half_y, 0.0),
            p3=gc.Point(x_low, half_y, 0.0),
            p4=gc.Point(x_high, half_y, 0.0),
        ),
    )


def test_receiver_grid_is_the_receiver() -> None:
    """The cells tile the receiver, whatever the rounding of size / tc.

    0.0006 / 0.0001 is 5.999999999999999, which int() truncated.
    """
    assert 0.0006 / 0.0001 < 6.0
    small = _receiver(-0.0003, 0.0003, 0.0003, 0.0001)
    assert _receiver_grid(small) == (6, 6)
    scene = _receiver(-0.006, 0.006, 0.007, 0.0005)
    assert _receiver_grid(scene) == (24, 28)


@pytest.mark.parametrize(
    ("receiver", "message"),
    [
        (_receiver(-0.001, 0.003, 0.001, 0.001), "centred"),
        (_receiver(-0.006, 0.006, 0.007, 0.0007), "multiple"),
    ],
)
def test_receiver_grid_refuses_what_the_kernel_cannot_bin(
    receiver: Entity, message: str
) -> None:
    """An off-centre receiver, or one tc does not divide, raises."""
    with pytest.raises(ValueError, match=message):
        _receiver_grid(receiver)


def test_rf_launch_cdf_follows_the_projected_areas() -> None:
    """Each reflector is drawn with the probability of its area.

    The receiver in the middle of the objects is never drawn, and the
    last reflector closes the table at exactly 1.
    """
    area = np.array([3.0, 0.0, 1.0, 0.0])
    is_reflector = np.array([True, False, True, False])
    cdf = _rf_launch_cdf(area, is_reflector)
    assert cdf.dtype == np.float32
    np.testing.assert_array_equal(cdf[[0, 2]], [0.75, 1.0])
    # a uniform draw picks the first reflector whose cdf reaches it
    draws = np.linspace(0.0, 1.0, 100001)[1:]
    drawn = np.flatnonzero(is_reflector)[
        np.searchsorted(cdf[is_reflector], draws)
    ]
    np.testing.assert_allclose(np.mean(drawn == 0), 0.75, atol=1e-4)


def test_rf_launch_cdf_without_projected_area() -> None:
    """Reflectors edge-on to the sun are equally likely."""
    cdf = _rf_launch_cdf(np.zeros(3), np.array([True, True, True]))
    np.testing.assert_allclose(cdf, [1 / 3, 2 / 3, 1.0], rtol=1e-6)
