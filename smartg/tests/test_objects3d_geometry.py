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
