"""GPU-free tests of the 3D objects.

The geometry classes, the heliostat generators and the helpers of
smartg.objects3d, and the host side of the object setup of
smartg.smartg, which checks the scene before anything is copied to
the GPU. The kernel side is tested by test_objects3d.py.
"""

import geoclide as gc
import numpy as np
import pytest

from smartg.objects3d import Entity, rotate_vector


@pytest.mark.parametrize("order", ["XYZ", "xyz", "zyx", "ZxY"])
def test_rotate_vector_accepts_either_case(order: str) -> None:
    """The rotation order is read in upper or lower case."""
    rotated = rotate_vector(gc.Vector(0.0, 0.0, 1.0), 90.0, 0.0, 0.0, order)
    np.testing.assert_allclose(
        [rotated.x, rotated.y, rotated.z], [0.0, -1.0, 0.0], atol=1e-12
    )


def test_rotate_vector_default_order() -> None:
    """The default order is XYZ."""
    vector = gc.Vector(1.0, 2.0, 3.0)
    default = rotate_vector(vector, 10.0, 20.0, 30.0)
    xyz = rotate_vector(vector, 10.0, 20.0, 30.0, "XYZ")
    np.testing.assert_allclose(
        [default.x, default.y, default.z], [xyz.x, xyz.y, xyz.z]
    )


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
