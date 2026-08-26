"""3D scene objects for SMART-G.

Defines materials (Mirror, LambMirror, Matte), primitive shapes
(Plane, Spheric), and the Entity/Heliostat/GroupE object model used
to describe reflectors and receivers (e.g. heliostat fields) for
SMART-G. Also provides related helpers: heliostat-field generation and
facet curvature, rotation/reflection computations, and reading heliostat
position files.

Key Classes
-----------
Entity
    3D object representation with geometry and material
    properties.
Heliostat
    Composite heliostat assembly consisting of multiple facets.
GroupE
    Container for grouping multiple Entity objects.
Mirror, LambMirror, Matte
    Material surface models (specular mirror, Lambertian mirror,
    matte).
Plane, Spheric
    Primitive surface shapes.
Transformation
    Rotation and translation applied to objects.
CusForward
    Custom rectangular forward launching mode of surface X*Y.
CusBackward
    Backward launching mode from a point or a plane receiver.

Key Functions
-------------
generate_h_p
    Generate well-oriented Heliostats from their positions.
generate_h_a
    Generate well-oriented Heliostats arranged in an angular
    sector around the receiver.
generate_box
    Create a 3D box/building composed of six planar faces.
convert_lg_to_le
    Convert a mixed list of Entity and GroupE objects to Entity
    objects only.
extract_points
    Extract heliostat coordinates from a file.
"""

from __future__ import annotations

import geoclide as gc
import numpy as np
import re
from itertools import dropwhile
from pathlib import Path
from scipy import interpolate
from warnings import warn


class Mirror(object):
    """
    Glossy/specular mirror material surface model.

    Represents glossy/specular reflective materials such as pure and
    highly polished aluminum, silver-backed glass mirrors, and similar
    surfaces. Uses microfacet theory with configurable roughness
    distribution models.

    Attributes
    ----------
    reflectivity : float, optional
        Albedo (reflectance) of the object. Must be between 0 and 1.
        Default: 1.0
    roughness : float, optional
        Surface roughness parameter (alpha) according to Walter et
        al. 2007. Characterizes the distribution of microfacet
        slopes. Default: 0.0
    shadow : bool, optional
        Whether to include shadowing-masking effects from surface
        roughness. Default: False
    nind : float or None, optional
        Relative refractive index (air/material). If None,
        represents a perfect mirror (nind = infinity). The internal
        value becomes -1 for perfect mirrors.
        Default: None
    distribution : str, optional
        Microfacet distribution model. Options are:
        - "Beckmann": Beckmann distribution (internally value 1)
        - "GGX": GGX/Trowbridge-Reitz distribution (internally value 2)
        Default: "Beckmann"

    References
    ----------
    Walter, B., Marschner, S. R., Li, H., & Torrance, K. E. (2007).
    Microfacet models for refraction through rough surfaces.
    """

    def __init__(
        self,
        reflectivity: float = 1.0,
        roughness: float = 0.0,
        shadow: bool = False,
        nind: float | None = None,
        distribution: str = "Beckmann",
    ) -> None:
        self.reflectivity = reflectivity
        self.roughness = roughness
        self.shadow = shadow
        if nind is None:
            self.nind = -1
        else:
            self.nind = nind
        if distribution == "Beckmann":
            self.distribution = 1
        elif distribution == "GGX":
            self.distribution = 2
        else:
            NameError(
                "Please choose a distribution between str(Beckmann) or "
                "str(GGX)"
            )

    def __str__(self) -> str:
        return (
            "Material -> Mirror : "
            "reflectivity="
            + str(self.reflectivity)
            + ", roughness="
            + str(self.roughness)
            + ", shadow="
            + str(self.shadow)
            + ", nind="
            + str(self.nind)
            + ", distribution="
            + str(self.distribution)
        )


class LambMirror(object):
    """
    Lambertian mirror material surface model.

    Represents a Lambertian reflective material with equal probability
    of reflection in all directions within the hemisphere normal to the
    object surface

    Parameters
    ----------
    reflectivity : float, optional
        Albedo (reflectance) of the object. Must be between 0 and 1.
        Controls the fraction of incident light that is reflected.
        Default: 0.5
    """

    def __init__(self, reflectivity: float = 0.5) -> None:
        self.reflectivity = reflectivity

    def __str__(self) -> str:
        return "Material -> Lambertian Mirror : reflectivity=" + str(
            self.reflectivity
        )


class Matte(object):
    """
    Matte material surface model.

    Represents matte materials such as concrete, plastic, dust,
    and similar surfaces with diffuse reflectance properties.

    Parameters
    ----------
    reflectivity : float, optional
        Albedo (reflectance) of the object. Must be between 0 and 1.
        Default: 0.0
    roughness : float, optional
        Surface roughness parameter.
        Default: 0.0

    Notes
    -----
    Be careful !!

    - For the moment this material is only used for totally
      absorbant surfaces.
    """

    def __init__(
        self, reflectivity: float = 0.0, roughness: float = 0.0
    ) -> None:
        self.reflectivity = reflectivity
        self.roughness = roughness

    def __str__(self) -> str:
        return (
            "Material -> Matte : "
            "reflectivity="
            + str(self.reflectivity)
            + ", roughness="
            + str(self.roughness)
        )


class Plane(object):
    """
    Planar surface defined by four corner points.

    Defines a rectangular plane surface constructed from four corner
    points. The plane must satisfy specific coordinate constraints
    for each point.

    Parameters
    ----------
    p1 : gc.Point, optional
        Bottom-left corner point (x negative, y negative).
        Default: gc.Point(-0.5, -0.5, 0.)
    p2 : gc.Point, optional
        Bottom-right corner point (x positive, y negative).
        Default: gc.Point(0.5, -0.5, 0.)
    p3 : gc.Point, optional
        Top-left corner point (x negative, y positive).
        Default: gc.Point(-0.5, 0.5, 0.)
    p4 : gc.Point, optional
        Top-right corner point (x positive, y positive).
        Default: gc.Point(0.5, 0.5, 0.)

    Notes
    -----
    The plane geometry requires:
    - p1 and p3 have the same negative x-coordinate
    - p2 and p4 have the same positive x-coordinate
    - p1 and p2 have the same negative y-coordinate
    - p3 and p4 have the same positive y-coordinate
    """

    def __init__(
        self,
        p1: gc.Point | None = None,
        p2: gc.Point | None = None,
        p3: gc.Point | None = None,
        p4: gc.Point | None = None,
    ) -> None:
        if p1 is None:
            p1 = gc.Point(-0.5, -0.5, 0.0)
        if p2 is None:
            p2 = gc.Point(0.5, -0.5, 0.0)
        if p3 is None:
            p3 = gc.Point(-0.5, 0.5, 0.0)
        if p4 is None:
            p4 = gc.Point(0.5, 0.5, 0.0)
        if (
            isinstance(p1, gc.Point)
            and isinstance(p2, gc.Point)
            and isinstance(p3, gc.Point)
            and isinstance(p4, gc.Point)
        ):
            if (
                ((p1.x == p3.x) and (p1.x < 0))
                and ((p2.x == p4.x) and (p2.x > 0))
                and ((p1.y == p2.y) and (p1.y < 0))
                and ((p3.y == p4.y) and (p3.y > 0))
            ):
                self.p1 = p1
                self.p2 = p2
                self.p3 = p3
                self.p4 = p4
            elif (p1.x >= 0) or (p2.x <= 0) or (p1.y >= 0) or (p3.y >= 0):
                raise NameError(
                    "Those conditions must be filled! : "
                    + "p1.x < 0 , p1.y < 0 ,"
                    + "p2.x > 0 , p2.y < 0 ,"
                    + "p3.x < 0 , p3.y > 0 ,"
                    + "p4.x > 0 , p4.y > 0"
                )
            elif (
                (p1.x != p3.x)
                or (p2.x != p4.x)
                or (p1.y != p2.y)
                or (p3.y != p4.y)
            ):
                raise NameError(
                    "Your plane geometry must be at leat a rectangle!"
                )
            else:
                NameError("Unknown error in Plane class!")
        else:
            raise NameError("All arguments must be Point type!")

    def __str__(self) -> str:
        return (
            "Coordinates of the Plane :\n"
            "-> p1=("
            + str(self.p1.x)
            + ", "
            + str(self.p1.y)
            + ", "
            + str(self.p1.z)
            + ")\n"
            + "-> p2=("
            + str(self.p2.x)
            + ", "
            + str(self.p2.y)
            + ", "
            + str(self.p2.z)
            + ")\n"
            + "-> p3=("
            + str(self.p3.x)
            + ", "
            + str(self.p3.y)
            + ", "
            + str(self.p3.z)
            + ")\n"
            + "-> p4=("
            + str(self.p4.x)
            + ", "
            + str(self.p4.y)
            + ", "
            + str(self.p4.z)
            + ")"
        )


class Spheric(object):
    """
    Spherical surface model.

    Represents a spherical (or partial spherical) surface defined by
    radius and optional height constraints. Can represent a full
    sphere or a partial sphere.

    Parameters
    ----------
    radius : float, optional
        Radius of the sphere. Must be positive.
        Default: 10.0
    z0 : float or None, optional
        Minimum height (bottom) of the spherical surface. If None,
        defaults to -radius (full sphere from bottom). For partial
        spheres, specify custom z0 value.
        Default: None (becomes -radius)
    z1 : float or None, optional
        Maximum height (top) of the spherical surface. If None,
        defaults to +radius (full sphere to top). For partial
        spheres, specify custom z1 value.
        Default: None (becomes +radius)
    phi : float, optional
        Azimuthal angle range in degrees. 360 degrees represents a full
        sphere; smaller values create a partial spherical sector.
        Default: 360.0

    Notes
    -----
    For a full sphere, use default values: z0 = -radius, z1 = +radius,
    phi = 360°
    """

    def __init__(
        self,
        radius: float = 10.0,
        z0: float | None = None,
        z1: float | None = None,
        phi: float = 360.0,
    ) -> None:
        self.radius = radius
        self.phi = phi
        if z0 is None:
            self.z0 = -1.0 * radius
        else:
            self.z0 = z0
        if z1 is None:
            self.z1 = 1.0 * radius
        else:
            self.z1 = z1

    def __str__(self) -> str:
        return (
            "Sphere with the following caracteristics :\n"
            + "-> radius = "
            + str(self.radius)
            + "\n"
            + "-> z0 = "
            + str(self.z0)
            + "\n"
            + "-> z1 = "
            + str(self.z1)
            + "\n"
            + "-> phi = "
            + str(self.phi)
        )


class Transformation:
    """
    Apply rotation and translation transformations to objects.

    Enables flexible transformation of objects through rotation and
    translation operations. Supports multiple rotation order
    conventions for specifying the sequence of rotations around
    different axes.

    Parameters
    ----------
    rotation : 1-D ndarray, optional
        An array with 3 elements specifying rotation angles (in
        degrees) around the x, y, and z axes respectively.
        Default: np.zeros(3, dtype=float) (no rotation)
    translation : 1-D ndarray, optional
        An array with 3 elements specifying translation distances
        (in kilometers) along the x, y, and z axes respectively.
        Default: np.zeros(3, dtype=float) (no translation)
    rotation_order : str, optional
        Specifies the order in which rotations are applied. Options are:
        - "XYZ": Rotate around X, then Y, then Z
        - "XZY": Rotate around X, then Z, then Y
        - "YXZ": Rotate around Y, then X, then Z
        - "YZX": Rotate around Y, then Z, then X
        - "ZXY": Rotate around Z, then X, then Y
        - "ZYX": Rotate around Z, then Y, then X
        Default: "XYZ"
    """

    def __init__(
        self,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
        rotation_order: str = "XYZ",
    ) -> None:
        if rotation is None:
            rotation = np.zeros(3, dtype=float)
        if translation is None:
            translation = np.zeros(3, dtype=float)
        self.rotation = rotation
        self.rotx = rotation[0]
        self.roty = rotation[1]
        self.rotz = rotation[2]
        self.rot_order = rotation_order
        self.translation = translation
        self.transx = translation[0]
        self.transy = translation[1]
        self.transz = translation[2]

    def __str__(self) -> str:
        return (
            "Transformation : rotation=("
            + str(self.rotx)
            + ", "
            + str(self.roty)
            + ", "
            + str(self.rotz)
            + ") and translation =("
            + str(self.transx)
            + ", "
            + str(self.transy)
            + ", "
            + str(self.transz)
            + ")"
        )


MaterialType = Mirror | LambMirror | Matte
GeometryType = Plane | Spheric


class Entity(object):
    """
    3D object representation with geometry and material properties.

    Enables the creation and management of 3D objects with customizable
    geometry, materials, transformations, and visualization properties.
    Objects can be either reflectors or receivers. Receivers will have
    their flux distribution tracked during simulations.

    Parameters
    ----------
    entity : Entity or None, optional
        Existing Entity object to copy. If provided, all properties are
        copied from the source entity. If None, properties are set
        individually from other parameters.
        Default: None
    name : str, optional
        Object type. Options are:
        - "reflector": Passive reflecting surface
        - "receiver": Active receiver that tracks flux distribution
        Default: "reflector"
    tc : float, optional
        Cell size for flux distribution calculation (Taille Cellules in
        km). Defines the spatial resolution for flux binning.
        Default: 0.01
    material_front : Material, optional
        Material for the object's front surface (above-view side).
        Default: Matte()
    material_back : Material, optional
        Material for the object's back surface (reverse side).
        Default: Matte()
    geo : Geometry, optional
        Geometric shape of the object (e.g., Plane, Spheric).
        Default: Plane()
    transformation : Transformation, optional
        Rotation and translation transformation to apply to the object.
        Default: Transformation() (identity transformation)
    bbox_pmin : None | gc.Point, optional
        Minimum corner of the bounding box (in development).
        Default: None
    bbox_pmax : None | gc.Point, optional
        Maximum corner of the bounding box (in development).
        Default: None
    color : str, optional
        Color for visualization/rendering.
        Default: 'grey'
    alpha_color : float, optional
        Transparency alpha value for visualization (0.0 to 1.0).
        Default: 0.5
    """

    def __init__(
        self,
        entity: Entity | None = None,
        name: str = "reflector",
        tc: float = 0.01,
        material_front: MaterialType | None = None,
        material_back: MaterialType | None = None,
        geo: GeometryType | None = None,
        transformation: Transformation | None = None,
        bbox_pmin: gc.Point | None = None,
        bbox_pmax: gc.Point | None = None,
        color: str = "grey",
        alpha_color: float = 0.5,
    ) -> None:
        if material_front is None:
            material_front = Matte()
        if material_back is None:
            material_back = Matte()
        if geo is None:
            geo = Plane()
        if transformation is None:
            transformation = Transformation()
        if isinstance(entity, Entity):
            self.name = entity.name
            self.tc = entity.tc
            self.material_front = entity.material_front
            self.material_back = entity.material_back
            self.geo = entity.geo
            self.transformation = entity.transformation
            # TODO: Compute automatically bbox_pmin and bbox_pmax
            # from geo and transformation
            self.bbox_pmin = entity.bbox_pmin
            self.bbox_pmax = entity.bbox_pmax
            self.color = entity.color
            self.alpha_color = alpha_color
        else:
            if not isinstance(geo, (Plane, Spheric)):
                raise NameError(
                    "For the moment only Plane or a Spheric geo are accepted."
                )

            self.name = name
            self.tc = tc
            self.material_front = material_front
            self.material_back = material_back
            self.geo = geo
            self.transformation = transformation

            # if bbox pmin and pmax are not provided compute them
            # automatically based on the geometry and transformation
            if bbox_pmin is None or bbox_pmax is None:
                box = gc.BBox()
                entity_tf = self.get_transformation()
                if isinstance(self.geo, Plane):
                    box = box.union(entity_tf(self.geo.p1))
                    box = box.union(entity_tf(self.geo.p2))
                    box = box.union(entity_tf(self.geo.p3))
                    box = box.union(entity_tf(self.geo.p4))
                elif isinstance(self.geo, Spheric):
                    p1 = entity_tf(
                        gc.Point(
                            -self.geo.radius, -self.geo.radius, self.geo.z0
                        )
                    )
                    p2 = entity_tf(
                        gc.Point(self.geo.radius, self.geo.radius, self.geo.z1)
                    )
                    box = box.union(p1)
                    box = box.union(p2)
                if bbox_pmin is None:
                    bbox_pmin = box.pmin
                if bbox_pmax is None:
                    bbox_pmax = box.pmax

            self.bbox_pmin = bbox_pmin
            self.bbox_pmax = bbox_pmax
            self.color = color
            self.alpha_color = alpha_color
        self.check = "Entity"

    def __str__(self) -> str:
        return (
            "The entity is a "
            + str(self.name)
            + " with the following carac:\n"
            + str(self.material_front)
            + "\n"
            + str(self.geo)
            + "\n"
            + str(self.transformation)
        )

    def get_transformation(self) -> gc.Transform:
        """
        Compute the combined transformation matrix for the entity.

        Returns
        -------
        out : gc.Transform
            Combined transformation matrix (translation * rotations
            in specified order). The rotation order is determined by
            the entity's transformation.rot_order attribute (e.g.,
            "XYZ", "ZYX", etc.).

        Notes
        -----
        The transformation is applied as::

            combined = Translation * Rotation_sequence

        where Rotation_sequence depends on rot_order:
        - "XYZ": Rx * Ry * Rz
        - "XZY": Rx * Rz * Ry
        - "YXZ": Ry * Rx * Rz
        - "YZX": Ry * Rz * Rx
        - "ZXY": Rz * Rx * Ry
        - "ZYX": Rz * Ry * Rx
        """
        trans = gc.get_translate_tf(
            gc.Vector(
                self.transformation.transx,
                self.transformation.transy,
                self.transformation.transz,
            )
        )
        rot_x = gc.get_rotate_x_tf(self.transformation.rotx)
        rot_y = gc.get_rotate_y_tf(self.transformation.roty)
        rot_z = gc.get_rotate_z_tf(self.transformation.rotz)

        # total tt of all transform together
        tt = None
        if self.transformation.rot_order == "XYZ":
            tt = trans * rot_x * rot_y * rot_z
        elif self.transformation.rot_order == "XZY":
            tt = trans * rot_x * rot_z * rot_y
        elif self.transformation.rot_order == "YXZ":
            tt = trans * rot_y * rot_x * rot_z
        elif self.transformation.rot_order == "YZX":
            tt = trans * rot_y * rot_z * rot_x
        elif self.transformation.rot_order == "ZXY":
            tt = trans * rot_z * rot_x * rot_y
        elif self.transformation.rot_order == "ZYX":
            tt = trans * rot_z * rot_y * rot_x
        else:
            raise NameError("Unknown rotation order")

        return tt

    def set_transformation(
        self, transformation: Transformation, recompute_bbox: bool = True
    ) -> None:
        """
        Update the entity's transformation and optionally recompute
        bounding box.

        Parameters
        ----------
        transformation : Transformation
            New transformation object containing rotation angles
            (rotx, roty, rotz), rotation order (rot_order), and
            translation components (transx, transy, transz).
        recompute_bbox : bool, optional
            If True (default), recompute the bounding box (bbox_pmin
            and bbox_pmax) based on the new transformation and the
            entity's geometry. If False, keep the existing bounding
            box values.
            Default: True

        Notes
        -----
        The bounding box is automatically recomputed by transforming
        all geometry points using the new transformation matrix and
        computing their extent.

        Examples
        --------
        >>> entity = Entity(
        ...     geo=Plane(...), transformation=Transformation()
        ... )
        >>> new_tf = Transformation(translation=np.array([1., 2., 3.]))
        >>> entity.set_transformation(
        ...     new_tf
        ... )  # Update position and recompute bbox
        >>> entity.set_transformation(
        ...     new_tf, recompute_bbox=False
        ... )  # Update without bbox update
        """
        self.transformation = transformation

        if recompute_bbox:
            # Recompute bounding box based on new transformation
            box = gc.BBox()
            entity_tf = self.get_transformation()

            if isinstance(self.geo, Plane):
                box = box.union(entity_tf(self.geo.p1))
                box = box.union(entity_tf(self.geo.p2))
                box = box.union(entity_tf(self.geo.p3))
                box = box.union(entity_tf(self.geo.p4))
            elif isinstance(self.geo, Spheric):
                p1 = entity_tf(
                    gc.Point(-self.geo.radius, -self.geo.radius, self.geo.z0)
                )
                p2 = entity_tf(
                    gc.Point(self.geo.radius, self.geo.radius, self.geo.z1)
                )
                box = box.union(p1)
                box = box.union(p2)

            self.bbox_pmin = box.pmin
            self.bbox_pmax = box.pmax


class Heliostat(object):
    """
    Composite heliostat assembly consisting of multiple facets.

    Represents a heliostat composed of multiple individual facets
    arranged in a grid pattern.

    Parameters
    ----------
    pos : gc.Point, optional
        Heliostat position (center point) stored as a Point class.
        Default: gc.Point(0., 0., 0.)
    n_facets_x : int, optional
        Number of facet divisions in the x direction. Controls how
        many times the heliostat is split along the x-axis. Must be
        >= 1 (total facets >= 2).
        Default: 2
    n_facets_y : int, optional
        Number of facet divisions in the y direction. Controls how
        many times the heliostat is split along the y-axis. Must be
        >= 1 (total facets >= 2).
        Default: 2
    helio_size_x : float, optional
        Heliostat size in the x direction (meters).
        Default: 0.02
    helio_size_y : float, optional
        Heliostat size in the y direction (meters).
        Default: 0.02
    curve_focal_length : float | None, optional
        Focal length (in km) for curvature. If None, the focal length
        is computed automatically based on the distance to the
        receiver. A virtual value of infinity means a flat heliostat
        with no curvature.
        Default: None
    reflectivity : float, optional
        Reflectivity of the heliostat (between 0 and 1). Represents the
        fraction of incident radiation that is reflected.
        Default: 1.0
    roughness : float, optional
        Surface roughness of the heliostat facets.
        Default: 0
    """

    def __init__(
        self,
        pos: gc.Point | None = None,
        n_facets_x: int = int(2),
        n_facets_y: int = int(2),
        helio_size_x: float = 0.02,
        helio_size_y: float = 0.02,
        curve_focal_length: float | None = None,
        reflectivity: float = 1.0,
        roughness: float = 0,
    ) -> None:
        if pos is None:
            pos = gc.Point(0.0, 0.0, 0.0)
        # Be sure that we split a heliostat by at least 2
        if n_facets_x * n_facets_y < 2:
            raise Exception("The number of facets must be >= 2!")
        # Be sure that n_facets_x and n_facets_y are integer values
        if not (isinstance(n_facets_x, int) and isinstance(n_facets_y, int)):
            raise Exception("n_facets_x and n_facets_y must be integers")
        self.pos = pos
        self.n_facets_x = n_facets_x
        self.n_facets_y = n_facets_y
        self.helio_size_x = helio_size_x
        self.helio_size_y = helio_size_y
        self.curve_focal_length = curve_focal_length
        self.ref = reflectivity
        self.rough = roughness

    def __str__(self) -> str:
        return (
            "POS="
            + str(self.pos)
            + "; "
            + "SPX="
            + str(self.n_facets_x)
            + "; "
            + "SPY="
            + str(self.n_facets_y)
            + "; "
            + "HSX="
            + str(self.helio_size_x)
            + "; "
            + "HSY="
            + str(self.helio_size_y)
            + "; "
            + "CURVE_FL="
            + str(self.curve_focal_length)
            + "; "
            + "REF="
            + str(self.ref)
            + "; "
            + "ROUGH="
            + str(self.rough)
        )


class GroupE(object):
    """Container for grouping multiple Entity objects.

    A GroupE instance represents a collection of Entity objects with
    a shared bounding box. This is useful for managing related
    geometric objects as a single unit, such as a set of heliostats
    or building components.

    Parameters
    ----------
    entities : list, optional
        List of Entity objects to group. Default is [Entity()].
    bbox : None | list, optional
        Custom bounding box as [Pmin, Pmax] where Pmin and Pmax are
        geoclide.Point objects. If None (default), bounding box is
        computed from entities[0].
    """

    def __init__(
        self,
        entities: list[Entity] | None = None,
        bbox: list[gc.Point] | None = None,
    ) -> None:
        if entities is None:
            entities = [Entity()]
        self.le = entities
        self.nob = len(entities)
        if bbox is None:
            box = gc.BBox(entities[0].bbox_pmin, entities[0].bbox_pmax)
            for i in range(1, self.nob):
                box = box.union(entities[i].bbox_pmin)
                box = box.union(entities[i].bbox_pmax)
            self.bbox_pmin = box.pmin
            self.bbox_pmax = box.pmax
        else:
            self.bbox_pmin = bbox[0]
            self.bbox_pmax = bbox[1]
        self.check = "GroupE"


def find_rots(
    dir_in: gc.Vector | gc.Normal | None = None,
    dir_out: gc.Vector | gc.Normal | None = None,
    normal: gc.Vector | gc.Normal | None = None,
) -> list:
    """Compute rotation angles to reflect an incoming ray toward an
    outgoing direction.

    Determines the Y and Z rotation angles necessary to orient a surface
    so that it reflects an incoming ray (dir_in) toward an outgoing
    direction (-dir_out). Can work with either incoming/outgoing ray
    directions or a pre-computed surface normal.

    Parameters
    ----------
    dir_in : gc.Vector | gc.Normal, optional
        Direction vector of the incoming ray or sun direction
        (geoclide.Vector). Required unless normal is provided. Default
        is None.
    dir_out : gc.Vector | gc.Normal, optional
        Direction vector of the outgoing ray, typically from receiver to
        facet center. The surface will be oriented to reflect dir_in
        toward -dir_out. Required unless normal is provided.
        Default is None.
    normal : gc.Vector | gc.Normal, optional
        Pre-computed normal vector of the reflection surface
        (geoclide.Vector). If provided, dir_in and dir_out are not used.
        Allows direct specification of the desired surface normal.
        Default is None.

    Raises
    ------
    ValueError
        If normal is None and either dir_in or dir_out is also None.

    Returns
    -------
    list
        A list containing rotation information:

        - **list[0]** : rot_y_deg (float)
            Rotation angle around Y-axis in radians
        - **list[1]** : rot_z_deg (float)
            Rotation angle around Z-axis in radians
        - **list[2]** : combined_tf (gc.Transform)
            Combined rotation transformation (geoclide.Transform
            object) that applies both rotations to orient the surface
            normal from (0, 0, 1) to the target direction

    Notes
    -----
    The function uses an iterative method to find rotation angles that
    align the initial surface normal (0, 0, 1) with the target normal
    computed from dir_in and dir_out. The algorithm applies Y-rotation
    first, then Z-rotation to achieve the desired reflection geometry.

    If normal is provided, it takes precedence and dir_in/dir_out are
    ignored.
    """
    # 1)Find the normal of the facet but filled in a vector class
    if normal is not None:
        facet_normal = gc.Vector(normal)
    elif dir_in is None or dir_out is None:
        raise ValueError(
            "find_rots needs either the normal parameter, or both the dir_in "
            "and dir_out parameters"
        )
    else:
        facet_normal = (gc.Vector(dir_in) + gc.Vector(dir_out)) * (-0.5)
    facet_normal = gc.normalize(facet_normal)
    facet_normal.z = np.clip(
        facet_normal.z, -1, 1
    )  # Avoid nan value in next operations

    # 2) Apply the inverse rotation operations to find the necessary
    # angles
    # 2.a) Initialisation
    loop = int(0)
    rot_y = 0
    rot_z = 0
    ope_z = 0
    # Values returned if no rotation is needed, i.e. if the while
    # loop below is never entered (identity transform and no
    # rotation in y and z)
    rot_y_deg = 0.0
    rot_z_deg = 0.0
    combined_tf = gc.Transform()
    # The initial value of the facet normal is (0, 0, 1) but forced
    # to (0, 0, 0) to be sure to activate the while loop below
    initial_normal = gc.Vector(0.0, 0.0, 0.0)

    # 2.b) Rotations are found in the loop bellow, at the end we check
    #      if after applying the transform to the initial normal of the
    #      facet 'initial_normal' we have the same value as the known
    #      well oriented facet normal 'facet_normal'. If no rotation has
    #      been found an error message will appear
    while (
        abs(initial_normal.x - facet_normal.x) > 1e-4
        or abs(initial_normal.y - facet_normal.y) > 1e-4
        or abs(initial_normal.z - facet_normal.z) > 1e-4
    ):
        loop += int(1)
        if loop > 4:
            raise NameError("No rotation has been found!")

        if loop == 1:
            rot_y = np.arccos(facet_normal.z)
            if facet_normal.x == 0 and rot_y == 0:
                ope_z = 0
            else:
                ope_z = facet_normal.x / np.sin(rot_y)
            ope_z = np.clip(ope_z, -1, 1)
            rot_z = np.arccos(ope_z)
        elif loop == 2:
            rot_y = np.arccos(facet_normal.z)
            if facet_normal.x == 0 and rot_y == 0:
                ope_z = 0
            else:
                ope_z = facet_normal.x / np.sin(rot_y)
            ope_z = np.clip(ope_z, -1, 1)
            rot_z = -np.arccos(ope_z)
        elif loop == 3:
            rot_y = -np.arccos(facet_normal.z)
            if facet_normal.x == 0 and rot_y == 0:
                ope_z = 0
            else:
                ope_z = facet_normal.x / np.sin(rot_y)
            ope_z = np.clip(ope_z, -1, 1)
            rot_z = np.arccos(ope_z)
        elif loop == 4:
            rot_y = -np.arccos(facet_normal.z)
            if facet_normal.x == 0 and rot_y == 0:
                ope_z = 0
            else:
                ope_z = facet_normal.x / np.sin(rot_y)
            ope_z = np.clip(ope_z, -1, 1)
            rot_z = -np.arccos(ope_z)

        rot_y_deg = np.degrees(rot_y)
        rot_z_deg = np.degrees(rot_z)
        tt_z = gc.get_rotate_z_tf(rot_z_deg)
        tt_y = gc.get_rotate_y_tf(rot_y_deg)
        combined_tf = tt_z * tt_y
        initial_normal = gc.normalize(combined_tf(gc.Vector(0.0, 0.0, 1.0)))

    return [rot_y_deg, rot_z_deg, combined_tf]


def generate_mtf(
    heliostat: Heliostat | None = None, receiver_pos: gc.Point | None = None
) -> np.ndarray:
    """Compute transformations for curved heliostat facet orientation.

    Generates transformation matrices for each facet of a heliostat to
    enable facet curvature. Each facet is oriented such that it
    reflects solar rays toward the center of a specified receiver
    position.

    Parameters
    ----------
    heliostat : Heliostat, optional
        A Heliostat class object defining the base heliostat geometry
        and segmentation.
        Default is Heliostat().
    receiver_pos : gc.Point, optional
        Position of the receiver center as a geoclide Point object.
        Facets are oriented to focus reflected rays toward this point.
        Default is gc.Point(0., 0., 0.).

    Returns
    -------
    facet_transforms : 2-D ndarray of Transform
        2D array of transformation matrices (geoclide.Transform
        objects) of shape (n_facets_x, n_facets_y), one for each
        facet. Each transformation positions and orients the
        corresponding facet.
    """
    if heliostat is None:
        heliostat = Heliostat()
    if receiver_pos is None:
        receiver_pos = gc.Point(0.0, 0.0, 0.0)
    # Heliostat is splited in facets in x and y directions
    n_facets_x = heliostat.n_facets_x
    n_facets_y = heliostat.n_facets_y
    # Size in x and y of a given facet
    facet_size_x = heliostat.helio_size_x / n_facets_x
    facet_size_y = heliostat.helio_size_y / n_facets_y
    half_facet_x = facet_size_x / 2
    half_facet_y = facet_size_y / 2  # Size of a facet divided by 2

    heliostat_pos = gc.Point(heliostat.pos.x, heliostat.pos.y, heliostat.pos.z)
    assumed_receiver_pos = gc.Point(
        0.0, 0.0, 0.0 + gc.Vector(heliostat_pos - receiver_pos).length()
    )

    # Find the positions of facets and store them in matrix
    # facet_points[i][j]
    facet_points = np.zeros(
        (n_facets_x, n_facets_y), dtype="object"
    )  # Matrix of Point object of each facets
    for i in range(0, n_facets_x):
        for j in range(0, n_facets_y):
            facet_points[i][j] = gc.Point(
                -(heliostat.helio_size_x / 2.0)
                + (i * facet_size_x)
                + half_facet_x,
                -(heliostat.helio_size_y / 2.0)
                + (j * facet_size_y)
                + half_facet_y,
                0.0,
            )

    # Find transform as function of focal length (for the curve)
    facet_transforms = np.zeros(
        (n_facets_x, n_facets_y), dtype="object"
    )  # Matrix of Transform object of each facets
    for i in range(0, n_facets_x):
        for j in range(0, n_facets_y):
            dir_in = gc.Point(0.0, 0.0, 0.0) - assumed_receiver_pos
            dir_in = gc.normalize(dir_in)
            dir_out = facet_points[i][j] - assumed_receiver_pos
            dir_out = gc.normalize(dir_out)
            rot_info = find_rots(dir_in=dir_in, dir_out=dir_out)
            facet_transforms[i][j] = gc.Transform(rot_info[2])

    return facet_transforms


def generate_le_h(
    heliostat: Heliostat | None = None,
    receiver_pos: gc.Point | None = None,
    theta_deg: float = 0.0,
    phi_deg: float = 0.0,
    facet_transforms: np.ndarray | None = None,
) -> list[Entity]:
    """Convert a heliostat to well-oriented plane facets for receiver
    reflection.

    Generates a list of properly oriented planar entity/facets from a
    heliostat object. Each facet is independently oriented to reflect
    solar rays toward a given receiver. This function manages the
    conversion of curved or segmented heliostats into their constituent
    facet entities.

    The facet indexing follows a matrix convention based on the
    heliostat's segmentation in x and y directions. See Notes section
    for the indexing convention.

    Parameters
    ----------
    heliostat : Heliostat, optional
        A Heliostat class object representing the heliostat to be
        converted.
        Default is Heliostat().
    receiver_pos : gc.Point, optional
        Position of the receiver as a geoclide.Point object. Used to
        orient facets toward the target. If None, a default point is
        used. Default is None.
    theta_deg : float, optional
        Solar zenith angle in degrees. Default is 0.
    phi_deg : float, optional
        Solar azimuth angle in degrees. Default is 0.
    facet_transforms : None | 2-D ndarray, optional
        A 2D ndarray of Transform objects of dim (n_facets_x,
        n_facets_y) representing the orientation of each facet. If
        None, The transforms are computed automatically based on the
        heliostat and receiver positions.

    Returns
    -------
    out : list
        List of plane Entity objects, each representing a facet
        properly oriented to reflect solar rays toward the receiver.

    Notes
    -----
    **Facet indexing convention:**

    Each facet is identified by a two-index notation **fij** where:

    - **i** is the row index (0 to n_facets_x-1), representing
      position along the x-direction
    - **j** is the column index (0 to n_facets_y-1), representing
      position along the y-direction

    Example with 4x4 segmentation::

                j0   j1   j2   j3
              +----+----+----+----+
        i0   |f00 |f01 |f02 |f03 |
              +----+----+----+----+
        i1   |f10 |f11 |f12 |f13 |
              +----+----+----+----+
        i2   |f20 |f21 |f22 |f23 |
              +----+----+----+----+
        i3   |f30 |f31 |f32 |f33 |
              +----+----+----+----+
                   ↑ y
              ← x

    The first row contains f00, f01, f02, f03; the second row contains
    f10, f11, f12, f13, and so on. This row-major ordering allows easy
    identification of any facet from its position in the segmented
    heliostat grid.
    """
    if heliostat is None:
        heliostat = Heliostat()
    # Be sure that the correct agrs have been given
    if not isinstance(heliostat, Heliostat):
        raise Exception("heliostat must be a Heliostat class!")
    if not isinstance(receiver_pos, gc.Point):
        raise Exception(
            "The receiver position 'receiver_pos' must be a Point class!"
        )

    # Direction of the sun (from (x,y,z) to (0,0,0))
    sun_dir = gc.ang2vec(theta_deg, phi_deg, vec_view="nadir")
    # Heliostat is splited in facets in x and y directions
    n_facets_x = heliostat.n_facets_x
    n_facets_y = heliostat.n_facets_y
    # Size in x and y of a given facet
    facet_size_x = heliostat.helio_size_x / n_facets_x
    facet_size_y = heliostat.helio_size_y / n_facets_y
    # Focal length or distance between heliostat and receiver
    focal_length = heliostat.curve_focal_length
    # Position of the heliostat
    heliostat_pos = gc.Point(heliostat.pos.x, heliostat.pos.y, heliostat.pos.z)
    # Receiver assumed position or the assumed focal length point.
    # Needed to curve the heliostat
    if focal_length is not None:
        assumed_receiver_pos = gc.Point(0.0, 0.0, 0.0 + focal_length)
    else:
        heliostat_pos_copy = gc.Point(heliostat_pos)
        receiver_distance = gc.Vector(
            heliostat_pos_copy - receiver_pos
        ).length()
        assumed_receiver_pos = gc.Point(0.0, 0.0, 0.0 + receiver_distance)
    # For the bounding box
    bbox_dist = (
        np.sqrt(
            heliostat.helio_size_x * heliostat.helio_size_x
            + heliostat.helio_size_y * heliostat.helio_size_y
        )
        / 2
    )

    # Initialisation
    facets = []  # List of facets
    half_facet_x = facet_size_x / 2
    half_facet_y = facet_size_y / 2  # Size of a facet divided by 2
    # Create one facet to be ready to clone other facets
    base_facet = Entity(
        name="reflector",
        material_front=Mirror(
            reflectivity=heliostat.ref, roughness=heliostat.rough
        ),
        material_back=Matte(),
        geo=Plane(
            p1=gc.Point(-half_facet_x, -half_facet_y, 0.0),
            p2=gc.Point(half_facet_x, -half_facet_y, 0.0),
            p3=gc.Point(-half_facet_x, half_facet_y, 0.0),
            p4=gc.Point(half_facet_x, half_facet_y, 0.0),
        ),
        transformation=Transformation(
            rotation=np.array([0.0, 0.0, 0.0]),
            translation=np.array([0.0, 0.0, 0.0]),
        ),
    )

    # Find the positions of facets and store them in matrix
    # facet_points[i][j]
    facet_points = np.zeros(
        (n_facets_x, n_facets_y), dtype="object"
    )  # Matrix of Point object of each facets
    for i in range(0, n_facets_x):
        for j in range(0, n_facets_y):
            facet_points[i][j] = gc.Point(
                -(heliostat.helio_size_x / 2.0)
                + (i * facet_size_x)
                + half_facet_x,
                -(heliostat.helio_size_y / 2.0)
                + (j * facet_size_y)
                + half_facet_y,
                0.0,
            )

    # Find transform as function of focal length (for the curve)
    if facet_transforms is None:
        facet_transforms = np.zeros(
            (n_facets_x, n_facets_y), dtype="object"
        )  # Matrix of Transform object of each facets
        for i in range(0, n_facets_x):
            for j in range(0, n_facets_y):
                dir_in = gc.Point(0.0, 0.0, 0.0) - assumed_receiver_pos
                dir_in = gc.normalize(dir_in)
                dir_out = facet_points[i][j] - assumed_receiver_pos
                dir_out = gc.normalize(dir_out)
                rot_info = find_rots(dir_in=dir_in, dir_out=dir_out)
                facet_transforms[i][j] = gc.Transform(rot_info[2])

    # Find the general heliostat rotation transform (like helistat is
    # a unique facet)
    dir_in = gc.Vector(sun_dir.x, sun_dir.y, sun_dir.z)
    dir_out = heliostat_pos - receiver_pos
    dir_in = gc.normalize(dir_in)
    dir_out = gc.normalize(dir_out)
    heliostat_rot_info = find_rots(dir_in=dir_in, dir_out=dir_out)
    heliostat_tf = heliostat_rot_info[2]

    # Apply the general rotation transform to each facet point and then
    # apply translation. This gives the final position of each facet
    # after rotation and translation of the heliostat, stored in the
    # matrix transformed_facet_points
    transformed_facet_points = np.zeros(
        (n_facets_x, n_facets_y), dtype="object"
    )  # equals to facet_points after application of transform
    for i in range(0, n_facets_x):
        for j in range(0, n_facets_y):
            tmp_point = gc.Point(facet_points[i][j])
            tmp_point = heliostat_tf(tmp_point)
            tmp_point.x += heliostat_pos.x
            tmp_point.y += heliostat_pos.y
            tmp_point.z += heliostat_pos.z
            transformed_facet_points[i][j] = gc.Point(tmp_point)

    # Write the initial coordinate system in term of vectors (x, y
    # and z)
    vec_x = gc.Vector(1.0, 0.0, 0.0)
    vec_y = gc.Vector(0.0, 1.0, 0.0)
    vec_z = gc.Vector(0.0, 0.0, 1.0)

    # Apply the general rotation transform to find the new coordinate
    # system of the heliostat
    vec_x = heliostat_tf(vec_x)
    vec_y = heliostat_tf(vec_y)
    vec_z = heliostat_tf(vec_z)
    vec_x = gc.normalize(vec_x)
    vec_y = gc.normalize(vec_y)
    vec_z = gc.normalize(vec_z)

    # Create the transformation matrix allowing to move between the 2
    # coordinate systems
    nn1 = vec_x
    nn2 = vec_y
    nn3 = vec_z
    mm2 = np.zeros((4, 4), dtype=np.float64)
    # Fill the transformation matrix (nn3 is the new z axis)
    mm2[0, 0] = nn1.x
    mm2[0, 1] = nn2.x
    mm2[0, 2] = nn3.x
    mm2[0, 3] = 0.0
    mm2[1, 0] = nn1.y
    mm2[1, 1] = nn2.y
    mm2[1, 2] = nn3.y
    mm2[1, 3] = 0.0
    mm2[2, 0] = nn1.z
    mm2[2, 1] = nn2.z
    mm2[2, 2] = nn3.z
    mm2[2, 3] = 0.0
    mm2[3, 0] = 0.0
    mm2[3, 1] = 0.0
    mm2[3, 2] = 0.0
    mm2[3, 3] = 1.0
    # Now create the transform object with the transformation matrix and
    # its inverse
    mm2_inv = np.transpose(mm2)
    world_to_obj = gc.Transform(
        m=mm2, m_inv=mm2_inv
    )  # move from world/initial to object∕new basis
    obj_to_world = gc.Transform(
        m=mm2_inv, m_inv=mm2
    )  # move from object∕new to world/initial basis

    # The normal of the heliostat heliostat_normal = z axis of the new
    # coordinate system
    heliostat_normal = gc.Vector(
        vec_z
    )  # stored as a vector for transformation purposes
    for i in range(0, n_facets_x):
        for j in range(0, n_facets_y):
            # come back to the initial coordinate system
            facet_normal = obj_to_world(heliostat_normal)
            # apply the transform of the facet to consider the curve
            # effect
            facet_normal = facet_transforms[i][j](facet_normal)
            # Now we return to the new coordinate system, which gives
            # then the normal of the facet (not heliostat) stored in
            # facet_transforms[i][j]
            facet_normal = world_to_obj(facet_normal)
            facet_normal = gc.normalize(facet_normal)

            # Find the rotation transform
            facet_rot_info = find_rots(normal=facet_normal)

            # Once the rotation angles have been found, create the facet
            # as entity object
            facet_entity = Entity(base_facet)
            facet_entity.transformation = Transformation(
                rotation=np.array([0.0, facet_rot_info[0], facet_rot_info[1]]),
                translation=np.array(
                    [
                        transformed_facet_points[i][j].x,
                        transformed_facet_points[i][j].y,
                        transformed_facet_points[i][j].z,
                    ]
                ),
                rotation_order="ZYX",
            )
            facet_center_pos = gc.Point(heliostat_pos)

            facet_entity.bbox_pmin = gc.Point(
                facet_center_pos.x - bbox_dist,
                facet_center_pos.y - bbox_dist,
                facet_center_pos.z - bbox_dist,
            )
            facet_entity.bbox_pmax = gc.Point(
                facet_center_pos.x + bbox_dist,
                facet_center_pos.y + bbox_dist,
                facet_center_pos.z + bbox_dist,
            )
            facets.append(facet_entity)

    return facets


def generate_box(
    dim_xyz: list[float] | None = None,
    pos: gc.Point | None = None,
    material_front: str | list[MaterialType] = "LambMirror",
    reflectivity: list[float] | None = None,
    roughness: list[float] | None = None,
    rot_z: float = 0.0,
    gap: float = 0.0001,
    obj_type: str = "environment",
    colors: list[str] | None = None,
    alpha_color: list[float] | None = None,
) -> GroupE:
    """Create a 3D box/building composed of six planar faces.

    Generates a box with six faces following Didier's 3D atmosphere
    convention in SMART-G. Each face can have different materials and
    properties. The origin is located at the center of the bottom face
    (Face 5), not at the center of the box.

    Face convention and orientation:

    - Face 0: Right   - In face: top Y+, right Z-
    - Face 1: Left    - In face: top Y+, right Z+
    - Face 2: Back    - In face: top Z-, right X+
    - Face 3: Front   - In face: top Z+, right X+
    - Face 4: Top     - In face: top Y+, right X+
    - Face 5: Bottom  - In face: top Y+, right X-

    Parameters
    ----------
    dim_xyz : list, optional
        Dimensions of the box in [x, y, z] in kilometers. Default is
        [0.05, 0.05, 0.05].
    pos : gc.Point, optional
        Position of the box center. Origin is at the center of Face 5
        (bottom).
        Default is gc.Point(0., 0., 0.).
    material_front : str | list, optional
        Material for the front side of faces. Either:

        - "LambMirror" : Lambertian mirror for all faces (constant
          reflectivity)
        - "Mirror" : Specular mirror for all faces (with roughness)
        - list : List containing 6 material objects (e.g., Matte,
          LambMirror, Mirror) for each face

        Default is "LambMirror".
    reflectivity : list, optional
        Reflectivity values for each face when material_front is
        "Mirror" or "LambMirror". List of 6 floats, one per face.
        Default is [1., 1., 1., 1., 1., 1.]. Else ignored if
        material_front is a list of material objects.
    roughness : list, optional
        Surface roughness for each face when material_front is "Mirror".
        List of 6 floats, one per face. Default is [0.2, 0.2, 0.2,
        0.2, 0.2, 0.2]. Else ignored if material_front is
        "LambMirror" or a list of material objects.
    rot_z : float, optional
        Global rotation angle in degrees around the Z-axis. Default
        is 0.
    gap : float, optional
        Gap to add to the global bounding box, useful for very small
        objects. Default is 0.0001.
    obj_type : str, optional
        Type of object. Choices are: 'environment', 'reflector', or
        'receiver'. Default is 'environment'.
    colors : list, optional
        List of str colors for each of the 6 faces. If None, all faces
        are colored grey. Default is None.
    alpha_color : list, optional
        List of transparency float values (0-1) for each of the 6
        faces. If None, all faces have 0.5. Default is None.

    Returns
    -------
    out : GroupE
        A group object (GroupE class) composed of six plane objects
        representing the box faces.

    Notes
    -----
    - Origin is at the center of Face 5 (bottom), NOT at the center of
      the box.
    - Global rotation in Z-axis only (other rotations not yet enabled).
    - Front side of each face uses the specified material
      (material_front); back side is always Matte (totally absorptive).
    """
    if dim_xyz is None:
        dim_xyz = [0.05, 0.05, 0.05]
    if pos is None:
        pos = gc.Point(0.0, 0.0, 0.0)
    if reflectivity is None:
        reflectivity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    if roughness is None:
        roughness = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    # Material AV = front part (i.e. part outside the box) of Face 0
    # to Face 5, back part (i.e. part inside the box) will be definite
    # as matte (totally absorbant)
    material_front_list: list[MaterialType] = []
    if material_front == "Mirror":
        for i in range(0, 6):
            material_front_list.append(
                Mirror(reflectivity=reflectivity[i], roughness=roughness[i])
            )
    elif material_front == "LambMirror":
        for i in range(0, 6):
            material_front_list.append(
                LambMirror(reflectivity=reflectivity[i])
            )
    elif isinstance(material_front, str):
        raise NameError(
            "Unknown material_front value: '"
            + material_front
            + "'. It must be 'Mirror', 'LambMirror', or a list of 6 material"
            + " objects"
        )
    else:
        material_front_list = material_front

    # colors
    if colors is None:
        colors = ["grey", "grey", "grey", "grey", "grey", "grey"]
    if alpha_color is None:
        alpha_color = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # === Commun parameters ===
    # Compute the half dimensions in X, Y and Z
    half_dim_x = dim_xyz[0] / 2.0
    half_dim_y = dim_xyz[1] / 2.0
    half_dim_z = dim_xyz[2] / 2.0

    # With the global Z rotation, 4 translations are needed in the
    # direction after the rotation, for Face 0 to 3
    tt = gc.get_rotate_z_tf(rot_z)
    offset_x = gc.Vector(1.0, 0.0, 0.0)
    offset_x = tt(offset_x)
    offset_x = gc.normalize(offset_x) * half_dim_x
    offset_y = gc.Vector(0.0, 1.0, 0.0)
    offset_y = tt(offset_y)
    offset_y = gc.normalize(offset_y) * half_dim_y

    # Initialize a numpy array list of Points (p1 to p4 to construct a
    # face) for all faces (from face 0 to 5)
    p1_faces = np.empty(6, dtype=object)
    p2_faces = np.empty(6, dtype=object)
    p3_faces = np.empty(6, dtype=object)
    p4_faces = np.empty(6, dtype=object)

    # Initialisze rotation needed to orient correctly each face
    rot_x_faces = np.zeros(6, dtype="float64")
    rot_y_faces = np.zeros(6, dtype="float64")
    rot_z_faces = np.full(
        6, rot_z
    )  # for Z rotation it is the same value for all faces

    # Initialize translation variables of all faces
    trans_x_faces = np.zeros(6, dtype="float64")
    trans_y_faces = np.zeros(6, dtype="float64")
    trans_z_faces = np.zeros(6, dtype="float64")
    # === End commun parameters ===

    # Face 0 unique parameters
    p1_faces[0] = gc.Point(-half_dim_z, -half_dim_y, 0.0)
    p2_faces[0] = gc.Point(half_dim_z, -half_dim_y, 0.0)
    p3_faces[0] = gc.Point(-half_dim_z, half_dim_y, 0.0)
    p4_faces[0] = gc.Point(half_dim_z, half_dim_y, 0.0)
    rot_x_faces[0] = 0.0
    rot_y_faces[0] = 90.0
    trans_x_faces[0] = pos.x + offset_x.x
    trans_y_faces[0] = pos.y + offset_x.y
    trans_z_faces[0] = pos.z + half_dim_z

    # Face 1 unique parameters
    p1_faces[1] = gc.Point(-half_dim_z, -half_dim_y, 0.0)
    p2_faces[1] = gc.Point(half_dim_z, -half_dim_y, 0.0)
    p3_faces[1] = gc.Point(-half_dim_z, half_dim_y, 0.0)
    p4_faces[1] = gc.Point(half_dim_z, half_dim_y, 0.0)
    rot_x_faces[1] = 0.0
    rot_y_faces[1] = -90.0
    trans_x_faces[1] = pos.x - offset_x.x
    trans_y_faces[1] = pos.y - offset_x.y
    trans_z_faces[1] = pos.z + half_dim_z

    # Face 2 unique parameters
    p1_faces[2] = gc.Point(-half_dim_x, -half_dim_z, 0.0)
    p2_faces[2] = gc.Point(half_dim_x, -half_dim_z, 0.0)
    p3_faces[2] = gc.Point(-half_dim_x, half_dim_z, 0.0)
    p4_faces[2] = gc.Point(half_dim_x, half_dim_z, 0.0)
    rot_x_faces[2] = -90.0
    rot_y_faces[2] = 0.0
    trans_x_faces[2] = pos.x + offset_y.x
    trans_y_faces[2] = pos.y + offset_y.y
    trans_z_faces[2] = pos.z + half_dim_z

    # Face 3 unique parameters
    p1_faces[3] = gc.Point(-half_dim_x, -half_dim_z, 0.0)
    p2_faces[3] = gc.Point(half_dim_x, -half_dim_z, 0.0)
    p3_faces[3] = gc.Point(-half_dim_x, half_dim_z, 0.0)
    p4_faces[3] = gc.Point(half_dim_x, half_dim_z, 0.0)
    rot_x_faces[3] = 90.0
    rot_y_faces[3] = 0.0
    trans_x_faces[3] = pos.x - offset_y.x
    trans_y_faces[3] = pos.y - offset_y.y
    trans_z_faces[3] = pos.z + half_dim_z

    # Face 4 unique parameters
    p1_faces[4] = gc.Point(-half_dim_x, -half_dim_y, 0.0)
    p2_faces[4] = gc.Point(half_dim_x, -half_dim_y, 0.0)
    p3_faces[4] = gc.Point(-half_dim_x, half_dim_y, 0.0)
    p4_faces[4] = gc.Point(half_dim_x, half_dim_y, 0.0)
    rot_x_faces[4] = 0.0
    rot_y_faces[4] = 0.0
    trans_x_faces[4] = pos.x
    trans_y_faces[4] = pos.y
    trans_z_faces[4] = pos.z + 2 * half_dim_z

    # Face 5 unique parameters
    p1_faces[5] = gc.Point(-half_dim_x, -half_dim_y, 0.0)
    p2_faces[5] = gc.Point(half_dim_x, -half_dim_y, 0.0)
    p3_faces[5] = gc.Point(-half_dim_x, half_dim_y, 0.0)
    p4_faces[5] = gc.Point(half_dim_x, half_dim_y, 0.0)
    rot_x_faces[5] = 0.0
    rot_y_faces[5] = 180.0
    trans_x_faces[5] = pos.x
    trans_y_faces[5] = pos.y
    trans_z_faces[5] = pos.z

    # Create the faces and incorporate them in a list
    faces = []
    for i in range(0, 6):
        face = Entity(
            name=obj_type,
            color=colors[i],
            alpha_color=alpha_color[i],
            material_front=material_front_list[i],
            material_back=Matte(),
            geo=Plane(
                p1=p1_faces[i], p2=p2_faces[i], p3=p3_faces[i], p4=p4_faces[i]
            ),
            transformation=Transformation(
                rotation=np.array(
                    [rot_x_faces[i], rot_y_faces[i], rot_z_faces[i]]
                ),
                translation=np.array(
                    [trans_x_faces[i], trans_y_faces[i], trans_z_faces[i]]
                ),
                rotation_order="ZXY",
            ),
        )
        faces.append(face)

    # Create a group of object with a global bounding box (can improve
    # significantly the computational time!)
    max_xy = max(pos.x, 2 * max(half_dim_x, half_dim_y))
    p_min = gc.Point(pos.x - max_xy - gap, pos.y - max_xy - gap, pos.z - gap)
    p_max = gc.Point(
        pos.x + max_xy + gap,
        pos.y + max_xy + gap,
        pos.z + 2 * half_dim_z + gap,
    )
    box_group = GroupE(entities=faces, bbox=[p_min, p_max])

    return box_group


def ref_fresnel(dir_in: gc.Vector, geo_transform: gc.Transform) -> gc.Vector:
    """Calculate Fresnel reflection direction for a ray on a
    transformed surface.

    Computes the direction of a reflected ray using simple Fresnel
    reflection based on the incident ray direction and the surface
    transformation.

    Parameters
    ----------
    dir_in : gc.Vector
        Direction vector of the incident ray entering the reflecting
        surface.
    geo_transform : gc.Transform
        Transformation (rotation and translation) of the surface where
        reflection occurs.

    Returns
    -------
    out : gc.Vector
        Direction vector of the reflected ray.
    """
    if isinstance(dir_in, gc.Vector):
        incident_dir = dir_in
    else:
        raise Exception("the dir_in argument must be a Vector class")
    if isinstance(geo_transform, gc.Transform):
        surf_tf = geo_transform
    else:
        raise Exception("the geo_transform argument must be a Transform class")

    # Default value of the surface plane normal
    normal_vec = gc.Vector(0.0, 0.0, 1)

    # Real value of the normal after considering transformation
    tt = surf_tf
    normal_vec = tt(normal_vec)

    # Information needed from the incoming ray
    ray_dir = incident_dir
    ray_dir = gc.Vector(-ray_dir.x, -ray_dir.y, -ray_dir.z)

    # Use the equation of Fresnel reflection (plenty explained in
    # pbrtv3 book)
    ray_dir = incident_dir + normal_vec * (2 * gc.dot(normal_vec, ray_dir))

    # Be sure ray_dir is normalized
    ray_dir = gc.Vector(gc.normalize(ray_dir))

    return ray_dir


def generate_h_p(
    theta_deg: float = 0.0,
    phi_deg: float = 0.0,
    heliostat_pos_list: list[gc.Point] | None = None,
    receiver_pos: gc.Point | None = None,
    helio_size_x: float = 0.001,
    helio_size_y: float = 0.001,
    reflectivity: float = 1,
    roughness: float = 0,
    heliostat_type: Heliostat | None = None,
    facet_transforms_list: list[np.ndarray] | None = None,
) -> list[Entity | GroupE]:
    """Generate well-oriented Heliostats from their positions.

    Generates a list of heliostat entities oriented to reflect sun rays
    toward a receiver. Can handle either planar heliostats or curved
    (faceted) heliostats depending on the heliostat_type parameter.

    Parameters
    ----------
    theta_deg : float, optional
        Sun zenith angle in degrees. Default is 0.
    phi_deg : float, optional
        Sun azimuth angle in degrees. Default is 0.
    heliostat_pos_list : list of Point, optional
        Coordinates of the center of heliostats. List of Point objects
        (geoclide). Default is [gc.Point(0., 0., 0.)].
    receiver_pos : Point, optional
        Coordinate of the center of the receiver (geoclide Point
        object). Default is gc.Point(0., 0., 0.).
    helio_size_x : float, optional
        Heliostat size in x-axis in kilometers. Default is 0.001.
    helio_size_y : float, optional
        Heliostat size in y-axis in kilometers. Default is 0.001.
    reflectivity : float, optional
        Reflectivity of the heliostats. Default is 1.
    roughness : float, optional
        Surface roughness of the heliostats. Default is 0.
    heliostat_type : Heliostat or None, optional
        If specified, must be a Heliostat class instance for generating
        curved (faceted) heliostats. If None (default), generates planar
        heliostats.
    facet_transforms_list : None or object, optional
        Under development. Default is None.

    Returns
    -------
    out : list
        List of Entity or GroupE objects, each properly oriented to
        reflect solar rays towards the receiver.
    """
    if heliostat_pos_list is None:
        heliostat_pos_list = [gc.Point(0.0, 0.0, 0.0)]
    if receiver_pos is None:
        receiver_pos = gc.Point(0.0, 0.0, 0.0)
    pos_list_copy = heliostat_pos_list.copy()
    obj_list = []

    # Case where the heliostat is totally plane
    if heliostat_type is None:
        # compute the sun direction vector
        sun_dir = gc.ang2vec(theta_deg, phi_deg, vec_view="nadir")
        bbox_dist = (
            np.sqrt(helio_size_x * helio_size_x + helio_size_y * helio_size_y)
            / 2
        )

        half_helio_x = helio_size_x / 2
        half_helio_y = helio_size_y / 2
        template_entity = Entity(
            name="reflector",
            material_front=Mirror(
                reflectivity=reflectivity, roughness=roughness
            ),
            material_back=Matte(reflectivity=0.0),
            geo=Plane(
                p1=gc.Point(-half_helio_x, -half_helio_y, 0.0),
                p2=gc.Point(half_helio_x, -half_helio_y, 0.0),
                p3=gc.Point(-half_helio_x, half_helio_y, 0.0),
                p4=gc.Point(half_helio_x, half_helio_y, 0.0),
            ),
            transformation=Transformation(
                rotation=np.array([0.0, 0.0, 0.0]),
                translation=np.array([0.0, 0.0, 0.0]),
            ),
        )

        for i in range(0, len(heliostat_pos_list)):
            # 1) Find the normalized vector colinear (and same dir) to
            # the normal of heliostat surface
            dir_to_receiver = pos_list_copy[i] - receiver_pos
            dir_to_receiver = gc.normalize(dir_to_receiver)

            # 2) Find the necessary rotations to apply on the heliostat
            # to reflect to the receiver
            rot_info = find_rots(dir_in=sun_dir, dir_out=dir_to_receiver)
            rot_y_deg = rot_info[0]
            rot_z_deg = rot_info[1]

            # 3) Once the rotation angles have been found, create
            # heliostat objects
            heliostat_entity = Entity(template_entity)
            heliostat_entity.bbox_pmin = gc.Point(
                pos_list_copy[i].x - bbox_dist,
                pos_list_copy[i].y - bbox_dist,
                pos_list_copy[i].z - bbox_dist,
            )
            heliostat_entity.bbox_pmax = gc.Point(
                pos_list_copy[i].x + bbox_dist,
                pos_list_copy[i].y + bbox_dist,
                pos_list_copy[i].z + bbox_dist,
            )
            heliostat_entity.transformation = Transformation(
                rotation=np.array([0.0, rot_y_deg, rot_z_deg]),
                translation=np.array(
                    [
                        pos_list_copy[i].x,
                        pos_list_copy[i].y,
                        pos_list_copy[i].z,
                    ]
                ),
                rotation_order="ZYX",
            )
            obj_list.append(heliostat_entity)
    # Case where the heliostat is composed by facets (i.g. to consider
    # the curvature)
    else:
        # Take the commun parameters of all heliostats
        n_facets_x = heliostat_type.n_facets_x
        n_facets_y = heliostat_type.n_facets_y
        helio_size_x = heliostat_type.helio_size_x
        helio_size_y = heliostat_type.helio_size_y
        curve_focal_length = heliostat_type.curve_focal_length

        # Generate all the facets and store them as entity object in
        # a list
        for i in range(0, len(heliostat_pos_list)):
            heliostat_obj = Heliostat(
                n_facets_x=n_facets_x,
                n_facets_y=n_facets_y,
                helio_size_x=helio_size_x,
                helio_size_y=helio_size_y,
                curve_focal_length=curve_focal_length,
                pos=pos_list_copy[i],
                reflectivity=reflectivity,
                roughness=roughness,
            )
            if facet_transforms_list is None:
                facet_entities = generate_le_h(
                    heliostat=heliostat_obj,
                    receiver_pos=receiver_pos,
                    theta_deg=theta_deg,
                    phi_deg=phi_deg,
                )
            else:
                facet_entities = generate_le_h(
                    heliostat=heliostat_obj,
                    receiver_pos=receiver_pos,
                    theta_deg=theta_deg,
                    phi_deg=phi_deg,
                    facet_transforms=facet_transforms_list[i],
                )
            facet_group = GroupE(entities=facet_entities)
            obj_list.append(facet_group)

    return obj_list


def generate_h_a(
    theta_deg: float = 0.0,
    phi_deg: float = 0.0,
    receiver_pos: gc.Point | None = None,
    min_ang_deg: float = 0.0,
    max_ang_deg: float = 360.0,
    gap_ang_deg: float = 5.0,
    first_dist: float = 0.1,
    n_heliostats: int = 10,
    gap_dist: float = 0.01,
    helio_size_x: float = 0.001,
    helio_size_y: float = 0.001,
    pillar_height: float = 0.006,
    reflectivity: float = 1,
    roughness: float = 0,
    heliostat_type: Heliostat | None = None,
    facet_transforms_list: list[np.ndarray] | None = None,
    return_positions: bool = False,
) -> list[Entity | GroupE] | tuple[list[Entity | GroupE], list[gc.Point]]:
    """Generate well-oriented Heliostats arranged in an angular sector
    around receiver.

    Generates heliostats positioned between min_ang_deg and max_ang_deg
    angles, properly oriented to reflect sun rays toward a central
    receiver. Heliostats are arranged in concentric patterns with
    specified angular and radial gaps.

    The angular coordinate system is defined as:

    .. code-block:: text

        y
        ^
        |/) ANG
        ---> x

    where ANG is measured from the positive x-axis.

    Parameters
    ----------
    theta_deg : float, optional
        Sun zenith angle in degrees. Default is 0.
    phi_deg : float, optional
        Sun azimuth angle in degrees. Default is 0.
    receiver_pos : Point, optional
        Coordinate of the center of the receiver (geoclide Point
        object). Heliostats are filled between min_ang_deg and
        max_ang_deg around this receiver. Default is gc.Point(0.,
        0., 50.).
    min_ang_deg : float, optional
        Minimum angular position in degrees. Default is 0.
    max_ang_deg : float, optional
        Maximum angular position in degrees. Default is 360.
    gap_ang_deg : float, optional
        Angular spacing in degrees for placing heliostats between
        min_ang_deg and max_ang_deg. Default is 5.
    first_dist : float, optional
        First distance between receiver and heliostat center in
        kilometers. Default is 0.1.
    n_heliostats : int, optional
        Number of heliostats to place at each angular position (radial
        direction). Default is 10.
    gap_dist : float, optional
        Radial gap between heliostats in kilometers after the first
        distance first_dist. Default is 0.01.
    helio_size_x : float, optional
        Heliostat size in x-axis in kilometers. Default is 0.001.
    helio_size_y : float, optional
        Heliostat size in y-axis in kilometers. Default is 0.001.
    pillar_height : float, optional
        Pillar height (distance from ground to heliostat) in kilometers.
        Default is 0.006.
    reflectivity : float, optional
        Reflectivity of the heliostats. Default is 1.
    roughness : float, optional
        Surface roughness of the heliostats. Default is 0.
    heliostat_type : Heliostat or None, optional
        If specified, must be a Heliostat class instance for generating
        curved (faceted) heliostats. If None (default), generates planar
        heliostats.
    facet_transforms_list : None or object, optional
        Under development. Default is None.
    return_positions : bool, optional
        If True, also return the list of heliostat positions. Default
        is False.

    Returns
    -------
    out1 : list
        List of heliostat Entity or GroupE objects arranged in the
        angular sector.
    out2 : list
        If return_positions is True, also returns the list of heliostat
        center positions (geoclide Point objects).
    """
    if receiver_pos is None:
        receiver_pos = gc.Point(0.0, 0.0, 50.0)

    # I) Find the position of all heliostats
    total_positions = int(
        ((max_ang_deg - min_ang_deg) / gap_ang_deg) * n_heliostats
    )

    # To avoid a given bug
    if (
        max_ang_deg - min_ang_deg < 360.000000001
        and max_ang_deg - min_ang_deg > 359.999999999
    ):
        n_rings = int(total_positions / n_heliostats)
    else:
        n_rings = int(total_positions / n_heliostats) + 1

    print("Total number of Heliostats = ", n_rings * n_heliostats)

    heliostat_positions = []
    current_ang_deg = min_ang_deg

    if min_ang_deg != max_ang_deg:
        for _i in range(0, n_rings):
            radial_dist = first_dist
            for _j in range(0, n_heliostats):
                tmp_pos = gc.Point(radial_dist, 0.0, 0.0)
                rot_z_tf = gc.get_rotate_z_tf(current_ang_deg)
                tmp_pos = rot_z_tf(tmp_pos)
                heliostat_positions.append(
                    gc.Point(tmp_pos.x, tmp_pos.y, tmp_pos.z + pillar_height)
                )
                radial_dist += gap_dist
            current_ang_deg += gap_ang_deg
    else:
        radial_dist = first_dist
        rot_z_tf = gc.get_rotate_z_tf(current_ang_deg)
        for _j in range(0, n_heliostats):
            tmp_pos = gc.Point(radial_dist, 0.0, 0.0)
            tmp_pos = rot_z_tf(tmp_pos)
            heliostat_positions.append(
                gc.Point(tmp_pos.x, tmp_pos.y, tmp_pos.z + pillar_height)
            )
            radial_dist += gap_dist

    # II) Creation of heliostats
    obj_list = []

    # Case where the heliostat is totally plane
    if heliostat_type is None:
        # calculate the sun direction vector
        sun_dir = gc.ang2vec(theta_deg, phi_deg, vec_view="nadir")
        bbox_dist = (
            np.sqrt(helio_size_x * helio_size_x + helio_size_y * helio_size_y)
            / 2
        )

        half_helio_x = helio_size_x / 2
        half_helio_y = helio_size_y / 2
        template_entity = Entity(
            name="reflector",
            material_front=Mirror(
                reflectivity=reflectivity, roughness=roughness
            ),
            material_back=Matte(),
            geo=Plane(
                p1=gc.Point(-half_helio_x, -half_helio_y, 0.0),
                p2=gc.Point(half_helio_x, -half_helio_y, 0.0),
                p3=gc.Point(-half_helio_x, half_helio_y, 0.0),
                p4=gc.Point(half_helio_x, half_helio_y, 0.0),
            ),
            transformation=Transformation(
                rotation=np.array([0.0, 0.0, 0.0]),
                translation=np.array([0.0, 0.0, 0.0]),
            ),
        )

        for i in range(0, len(heliostat_positions)):
            # 1) The vector of the photon after a reflection (here the
            # opposite direction)
            dir_to_receiver = heliostat_positions[i] - receiver_pos
            dir_to_receiver = gc.normalize(dir_to_receiver)

            # 2) The incoming (sun_dir) and outcoming
            #    (dir_to_receiver) directions are known then find the
            #    rotation angles
            rot_info = find_rots(dir_in=sun_dir, dir_out=dir_to_receiver)
            rot_y_deg = rot_info[0]
            rot_z_deg = rot_info[1]

            # 3) Once the rotation angles have been found, create
            # heliostat objects
            heliostat_entity = Entity(template_entity)
            heliostat_entity.bbox_pmin = gc.Point(
                heliostat_positions[i].x - bbox_dist,
                heliostat_positions[i].y - bbox_dist,
                heliostat_positions[i].z - bbox_dist,
            )
            heliostat_entity.bbox_pmax = gc.Point(
                heliostat_positions[i].x + bbox_dist,
                heliostat_positions[i].y + bbox_dist,
                heliostat_positions[i].z + bbox_dist,
            )
            heliostat_entity.transformation = Transformation(
                rotation=np.array([0.0, rot_y_deg, rot_z_deg]),
                translation=np.array(
                    [
                        heliostat_positions[i].x,
                        heliostat_positions[i].y,
                        heliostat_positions[i].z,
                    ]
                ),
                rotation_order="ZYX",
            )
            obj_list.append(heliostat_entity)

    # Case where the heliostat is composed by facets (i.g. to consider
    # the curvature)
    else:
        # Take the commun parameters of all heliostats
        n_facets_x = heliostat_type.n_facets_x
        n_facets_y = heliostat_type.n_facets_y
        helio_size_x = heliostat_type.helio_size_x
        helio_size_y = heliostat_type.helio_size_y
        curve_focal_length = heliostat_type.curve_focal_length

        # Generate all the facets and store them as entity object in
        # a list
        for i in range(0, len(heliostat_positions)):
            heliostat_obj = Heliostat(
                n_facets_x=n_facets_x,
                n_facets_y=n_facets_y,
                helio_size_x=helio_size_x,
                helio_size_y=helio_size_y,
                curve_focal_length=curve_focal_length,
                pos=heliostat_positions[i],
                reflectivity=reflectivity,
                roughness=roughness,
            )
            if facet_transforms_list is None:
                facet_entities = generate_le_h(
                    heliostat=heliostat_obj,
                    receiver_pos=receiver_pos,
                    theta_deg=theta_deg,
                    phi_deg=phi_deg,
                )
            else:
                facet_entities = generate_le_h(
                    heliostat=heliostat_obj,
                    receiver_pos=receiver_pos,
                    theta_deg=theta_deg,
                    phi_deg=phi_deg,
                    facet_transforms=facet_transforms_list[i],
                )
            facet_group = GroupE(entities=facet_entities)
            obj_list.append(facet_group)

    if return_positions:
        return obj_list, heliostat_positions
    else:
        return obj_list


def convert_lg_to_le(obj_list: list[Entity | GroupE]) -> list[Entity]:
    """Convert a mixed list of Entity and GroupE objects to Entity
    objects only.

    Flattens groups by expanding all GroupE objects into their
    constituent Entity objects, resulting in a list containing only
    Entity objects.

    Parameters
    ----------
    obj_list : list
        List containing Entity and/or GroupE objects to be converted.

    Returns
    -------
    out : list
        Flattened list containing only Entity objects. GroupE objects
        are converted into their constituent Entity objects.
    """
    n_objs = len(obj_list)
    flat_list: list[Entity] = []

    for i in range(0, n_objs):
        obj = obj_list[i]
        if isinstance(obj, GroupE):
            flat_list.extend(obj.le)
        elif isinstance(obj, Entity):
            flat_list.append(obj)
        else:
            raise NameError(
                "In the list, only Entity and GroupE classes are autorised!"
            )

    return flat_list


def rotate_vector(
    vector: gc.Vector,
    rot_x: float,
    rot_y: float,
    rot_z: float,
    rotation_order: str = "xyz",
) -> gc.Vector:
    """
    Definition of the function rotate_vector

    coordinate system convention:

      y
      ^   x : right; y : front; z : top
      |
    z X -- > x

    Given a vector and rotations to perform to this vector in degrees
    in the x,y,z axes, with in option the rotation order

    Arg:
    v              : A direction described by Vector class object
    rotx,y,z       : Rotations in x,y and z in degrees
    rotation_order : str with the order of rotations i.g. 'xyz', zxy',
                     ...

    Return:
    rotated_vector : The rotated (normalized) direction (also
                     a Vector class)
    """
    tt = gc.Transform()
    tr_x = gc.get_rotate_x_tf(rot_x)
    tr_y = gc.get_rotate_y_tf(rot_y)
    tr_z = gc.get_rotate_z_tf(rot_z)
    if rotation_order == "XYZ":
        tt = tr_x * tr_y * tr_z
    elif rotation_order == "XZY":
        tt = tr_x * tr_z * tr_y
    elif rotation_order == "YXZ":
        tt = tr_y * tr_x * tr_z
    elif rotation_order == "YZX":
        tt = tr_y * tr_z * tr_x
    elif rotation_order == "ZXY":
        tt = tr_z * tr_x * tr_y
    elif rotation_order == "ZYX":
        tt = tr_z * tr_y * tr_x
    else:
        raise NameError("Unknown rotation_order value!")
    rotated_vector = tt(vector)
    rotated_vector = gc.Vector(gc.normalize(rotated_vector))

    return rotated_vector


def interpolate_refls_from_wls(
    wavelengths: np.ndarray | list[float],
    reflectivities: np.ndarray | list[float],
    new_wavelengths: np.ndarray | list[float],
    extrapolate: bool = False,
) -> np.ndarray:
    """
        Definition: Giving a set of wavelengths (wavelengths) and
                    reflectivities (reflectivities), get the
                    interpolated reflectivities folowing the new set of
                    wavelengths (new_wavelengths)

    ==== ARGS:
    wavelengths     : List/array of wavelengths
    reflectivities  : List/array with reflectivities at each wavelength
                      of wavelengths
    new_wavelengths : List/array of the new wavelengths where we want
                      to interpolate

    ==== RETURN:
    refls_new : numpy array with the interpolated reflectivities
    """

    # type: ignore -> fill_value is documented as accepting an
    # array-like, a 2 element tuple or "extrapolate", but its stub only
    # allows a float
    if extrapolate:
        f = interpolate.interp1d(
            wavelengths,
            reflectivities,
            fill_value="extrapolate",  # type: ignore
        )
    else:
        f = interpolate.interp1d(
            wavelengths,
            reflectivities,
            fill_value=(reflectivities[0], reflectivities[-1]),  # type: ignore
            bounds_error=False,
        )

    refls_new = f(new_wavelengths)

    # Ensure relfectivities are between 0 and 1
    refls_new[refls_new < 0] = 0
    refls_new[refls_new > 1] = 1

    return refls_new


def is_comment(line: str) -> bool:
    """
    function to check if a line starts with some character. Here #
    for comment
    """
    # return true if a line starts with #
    return line.startswith("#")


def extract_points(filename: str | Path) -> list[gc.Point]:
    """Extract heliostat coordinates from a file.

    Reads a file and extracts the (x, y, z) coordinates of each
    heliostat, returning them as geoclide Point objects.

    The input file must follow this format:

    - First line: comment line beginning with '#'
    - Second line: empty line
    - Subsequent lines: x, y, and z coordinates of each heliostat,
      separated by commas

    Parameters
    ----------
    filename : str | pathlib.Path
        Path to the file containing the heliostat coordinates.

    Returns
    -------
    out : list
        List of geoclide.Point objects, each containing the x, y, and z
        coordinates of a heliostat.
    """

    # First check if filename is an str type
    file_content = ""
    try:
        with open(filename, "r") as file:
            for _curline in dropwhile(is_comment, file):
                file_content = file.read()
    except FileNotFoundError:
        print(str(filename) + " has been not found")
    except IOError:
        print("Enter/Exit error with " + str(filename))

    # Looking for a float and fill it in values
    values = re.findall(r"-?[0-9]+\.?[0-9]*", file_content)

    # Number of dimension and number of heliostats
    n_dims = 3  # x, y and z --> 3 dim
    n_heliostats = int(len(values) / n_dims)

    # # Fill the x, y and z coordinates into a list of Point classes
    points = []
    for i in range(0, n_heliostats):
        points.append(
            gc.Point(
                float(values[i * n_dims]),
                float(values[(i * n_dims) + 1]),
                float(values[(i * n_dims) + 2]),
            )
        )

    return points


class CusForward:
    """
    Custom rectangular forward launching mode of surface X*Y.

    Parameters
    ----------
    cfx : float, optional
        The size along the x axis (only for the FF lmode).
    cfy : float, optional
        The size along the y axis (only for the FF lmode).
    cftx : float, optional
        The translation to apply in x axis (only for the FF
        lmode).
    cfty : float, optional
        The translation to apply in y axis (only for the FF
        lmode).
    cftz : float, optional
        The translation to apply in z axis (only for the FF
        lmode).
    fov : float, optional
        The field of view or half-angle of the sun (only for the
        FF lmode).
    sampling : str, optional
        The sampling type (only for the FF lmode). Choices:

            * 'lambertian'
            * 'isotropic'
            * 'disk' (in development)
    lmode : str, optional
        The launching mode. Two choices:

            * 'RF' -> Restricted Forward. Launch the photons such
              that the direct beams fill only reflector objects.
            * 'FF' -> Full Forward. Launch the photons in a
              rectangle from TOA where the beams at the center
              target the origin point (0, 0, 0).
    lph : object, optional
        In progress...
    lpr : object, optional
        In progress...
    """
    def __init__(
        self,
        cfx: float = 0.,
        cfy: float = 0.,
        cftx: float = 0.,
        cfty: float = 0.,
        cftz: float = 0.,
        fov: float = 0.,
        sampling: str = "isotropic",
        lmode: str = "RF",
        lph: object | None = None,
        lpr: object | None = None,
    ) -> None:
        if sampling == "lambertian":
            sampling_code = 1
        elif sampling == "isotropic":
            sampling_code = 2
        elif sampling == "disk":  # in development
            sampling_code = 3
        else:
            raise ValueError(
                'You must choose lambertian or isotropic sampling')

        self.dict = {
            'CFX':   cfx,
            'CFY':   cfy,
            'CFTX':  cftx,
            'CFTY':  cfty,
            'CFTZ':  cftz,
            'FOV':   fov,
            'TYPE':  sampling_code,
            'LMODE': lmode,
            # under development ->
            'LPH':   lph,
            'LPR':   lpr,
        }

    def __str__(self) -> str:
        return (
            'CusForward=-CFX{CFX}-CFY{CFY}-CFTX{CFTX}-CFTY{CFTY}'
            '-CFTZ{CFTZ}-FOV{FOV}-TYPE{TYPE}'
            '-LMODE{LMODE}'.format(**self.dict)
        )


class CusBackward:
    """
    Backward launching mode from a point or a plane receiver.

    Parameters
    ----------
    pos : Point, optional
        The position (X, Y, Z) in cartesian coordinates. The
        default is Point(0., 0., 0.).
    thdeg : float, optional
        The zenith angle in degrees.
    phdeg : float, optional
        The azimuth angle in degrees.
    v : Vector, optional
        The normal vector of the receiver. If provided,
        circumvent thdeg and phdeg.
    aldeg : float, optional
        Launch in a solid angle where alpha is the half-angle of
        the cone.
    rec : Entity, optional
        The receiver object to be used in 'BR' mode. It must be a
        plane Entity object of type 'receiver'. The photon
        position is sampled at the receiver surface.
    sampling : str, optional
        The sampling type (only for the BR lmode). 2 choices:

            * 'lambertian'
            * 'isotropic'
    lmode : str, optional
        The launching mode. 2 choices:

            * 'B' -> Basic backward (deprecated, see notes).
              Launch the photons from a given point in a given
              direction with a field of view aldeg.
            * 'BR' -> Backward with receiver. Launch the photons
              from a given receiver (plane object) in a given
              direction with a field of view aldeg. Default
              value.
    lph : object, optional
        In progress...
    lpr : object, optional
        In progress...

    Notes
    -----
    The 'B' mode is deprecated and may lead to wrong results. Use
    instead the Sensor class.
    """
    def __init__(
        self,
        pos: gc.Point | None = None,
        thdeg: float = 0.,
        phdeg: float = 0.,
        v: gc.Vector | None = None,
        aldeg: float = 0.,
        rec: Entity | None = None,
        sampling: str = "lambertian",
        lmode: str = "BR",
        lph: object | None = None,
        lpr: object | None = None,
    ) -> None:
        if pos is None:
            pos = gc.Point(0., 0., 0.)
        if isinstance(v, gc.Vector):
            th, ph = gc.vec2ang(v)
            thdeg, phdeg = float(th), float(ph)
        elif v is not None:
            raise ValueError('The v argument must be a Vector')
        if lmode == "BR" and not isinstance(rec, Entity):
            raise ValueError(
                'In the BR lmode you have to specify a receiver!')
        if sampling == "lambertian":
            sampling_code = 1
        elif sampling == "isotropic":
            sampling_code = 2
        else:
            raise ValueError(
                'You must choose lambertian or isotropic sampling')

        if lmode == "B":
            warn(
                "\nThe lmode `B` is deprecated as of SMART-G 1.1.0 "
                "and will be removed in one of the next release.\n"
                "Please use the lmode `BR` or the class Sensor "
                "instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.dict = {
            'POS':   pos,
            'THDEG': thdeg,
            'PHDEG': phdeg,
            'ALDEG': aldeg,
            'REC':   rec,
            'TYPE':  sampling_code,
            'LMODE': lmode,
            # under development ->
            'LPH':   lph,
            'LPR':   lpr,
        }

    def __str__(self) -> str:
        return (
            'CusBackward:-POS={POS}-THDEG={THDEG}-PHDEG={PHDEG}'
            '-ALDEG={ALDEG}-TYPE={TYPE}'
            '-LMODE={LMODE}'.format(**self.dict)
        )
