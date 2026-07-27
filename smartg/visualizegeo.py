

import geoclide as gc
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as mp3d
from matplotlib import colors as mcolors
import re
from itertools import dropwhile
from scipy import interpolate


class Mirror(object):
    """
    Glossy/specular mirror material surface model.

    Represents glossy/specular reflective materials such as pure and highly 
    polished aluminum, silver-backed glass mirrors, and similar surfaces. Uses 
    microfacet theory with configurable roughness distribution models.

    Attributes
    ----------
    reflectivity : float, optional
        Albedo (reflectance) of the object. Must be between 0 and 1.
        Default: 1.0
    roughness : float, optional
        Surface roughness parameter (alpha) according to Walter et al. 2007.
        Characterizes the distribution of microfacet slopes. Default: 0.0
    shadow : bool, optional
        Whether to include shadowing-masking effects from surface roughness.
        Default: False
    nind : float or None, optional
        Relative refractive index (air/material). If None, represents a perfect 
        mirror (nind = infinity). The internal value becomes -1 for perfect mirrors.
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
    def __init__(self, reflectivity = 1., roughness = 0., shadow = False, nind = None,
                 distribution = "Beckmann"):
        self.reflectivity = reflectivity
        self.roughness    = roughness
        self.shadow       = shadow
        if nind is None:
            self.nind     = -1
        else:
            self.nind     = nind
        if distribution == "Beckmann":
            self.distribution = 1
        elif distribution == "GGX":
            self.distribution = 2
        else:
            NameError('Please choose a distribution between str(Beckmann) or str(GGX)')

    def __str__(self):
        return 'Material -> Mirror : ' \
            'reflectivity=' + str(self.reflectivity) + ', roughness=' + str(self.roughness) \
            + ', shadow=' + str(self.shadow) + ', nind=' + str(self.nind) \
            + ', distribution=' + str(self.distribution)


class LambMirror(object):
    """
    Lambertian mirror material surface model.

    Represents a Lambertian reflective material with equal probability of reflection 
    in all directions within the hemisphere normal to the object surface

    Parameters
    ----------
    reflectivity : float, optional
        Albedo (reflectance) of the object. Must be between 0 and 1.
        Controls the fraction of incident light that is reflected.
        Default: 0.5
    """
    def __init__(self, reflectivity = 0.5):
        self.reflectivity = reflectivity
        

    def __str__(self):
        return 'Material -> Lambertian Mirror : ' \
            'reflectivity=' + str(self.reflectivity)


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

    - For the moment this material is only used for totally absorbant surfaces.
    """
    def __init__(self, reflectivity = 0., roughness = 0.):
        self.reflectivity = reflectivity
        self.roughness = roughness
        
    def __str__(self):
        return 'Material -> Matte : ' \
            'reflectivity=' + str(self.reflectivity) + ', roughness=' + str(self.roughness)


class Plane(object):
    """
    Planar surface defined by four corner points.

    Defines a rectangular plane surface constructed from four corner points.
    The plane must satisfy specific coordinate constraints for each point.

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
    def __init__(self, p1 = gc.Point(-0.5, -0.5, 0.), p2 = gc.Point(0.5, -0.5, 0.), \
                 p3 = gc.Point(-0.5, 0.5, 0.), p4 = gc.Point(0.5, 0.5, 0.)):
        if (isinstance(p1, gc.Point) and isinstance(p2, gc.Point) and \
            isinstance(p3, gc.Point) and isinstance(p4, gc.Point)):
            if (  ( (p1.x == p3.x) and (p1.x < 0) )  and \
                  ( (p2.x == p4.x) and (p2.x > 0) )  and \
                  ( (p1.y == p2.y) and (p1.y < 0) )  and \
                  ( (p3.y == p4.y) and (p3.y > 0) )   ):
                self.p1 = p1
                self.p2 = p2
                self.p3 = p3
                self.p4 = p4
            elif ( (p1.x >= 0) or (p2.x <= 0) or (p1.y >= 0) or (p3.y >= 0) ):
                raise NameError( 'Those conditions must be filled! : ' + \
                                'p1.x < 0 , p1.y < 0 ,' + \
                                'p2.x > 0 , p2.y < 0 ,' + \
                                'p3.x < 0 , p3.y > 0 ,' + \
                                'p4.x > 0 , p4.y > 0' )
            elif ( (p1.x != p3.x) or (p2.x != p4.x) or (p1.y != p2.y) or (p3.y != p4.y) ):
                raise NameError('Your plane geometry must be at leat a rectangle!')
            else:
                NameError('Unknown error in Plane class!')
        else:
            raise NameError('All arguments must be Point type!')

    def __str__(self):
        return 'Coordinates of the Plane :\n' \
            '-> p1=(' + str(self.p1.x) + ', ' + str(self.p1.y) + ', ' + str(self.p1.z) + ')\n' + \
            '-> p2=(' + str(self.p2.x) + ', ' + str(self.p2.y) + ', ' + str(self.p2.z) + ')\n' + \
            '-> p3=(' + str(self.p3.x) + ', ' + str(self.p3.y) + ', ' + str(self.p3.z) + ')\n' + \
            '-> p4=(' + str(self.p4.x) + ', ' + str(self.p4.y) + ', ' + str(self.p4.z) + ')'

class Spheric(object):
    """
    Spherical surface model.

    Represents a spherical (or partial spherical) surface defined by radius 
    and optional height constraints. Can represent a full sphere or a partial 
    sphere.

    Parameters
    ----------
    radius : float, optional
        Radius of the sphere. Must be positive.
        Default: 10.0
    z0 : float or None, optional
        Minimum height (bottom) of the spherical surface. If None, defaults 
        to -radius (full sphere from bottom). For partial spheres, specify 
        custom z0 value.
        Default: None (becomes -radius)
    z1 : float or None, optional
        Maximum height (top) of the spherical surface. If None, defaults 
        to +radius (full sphere to top). For partial spheres, specify 
        custom z1 value.
        Default: None (becomes +radius)
    phi : float, optional
        Azimuthal angle range in degrees. 360 degrees represents a full 
        sphere; smaller values create a partial spherical sector.
        Default: 360.0

    Notes
    -----
    For a full sphere, use default values: z0 = -radius, z1 = +radius, phi = 360°
    """
    def __init__(self, radius = 10., z0 = None, z1 = None, phi = 360.):
        self.radius = radius
        self.phi = phi
        if (z0 == None):
            self.z0 = -1.*radius
        else:
            self.z0 = z0
        if (z1 == None):
            self.z1 = 1.*radius
        else:
            self.z1 = z1

    def __str__(self):
        return 'Sphere with the following caracteristics :\n' + \
            '-> radius = ' + str(self.radius) + '\n' + \
            '-> z0 = ' + str(self.z0) + '\n' + \
            '-> z1 = ' + str(self.z1) + '\n' + \
            '-> phi = ' + str(self.phi)


class Transformation():
    """
    Apply rotation and translation transformations to objects.

    Enables flexible transformation of objects through rotation and translation 
    operations. Supports multiple rotation order conventions for specifying 
    the sequence of rotations around different axes.

    Parameters
    ----------
    rotation : 1-D ndarray, optional
        An array with 3 elements specifying rotation angles (in degrees) 
        around the x, y, and z axes respectively.
        Default: np.zeros(3, dtype=float) (no rotation)
    translation : 1-D ndarray, optional
        An array with 3 elements specifying translation distances (in kilometers) 
        along the x, y, and z axes respectively.
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
    def __init__(self, rotation = np.zeros(3, dtype=float), translation=np.zeros(3, dtype=float), \
                 rotation_order = "XYZ"):
        self.rotation = rotation
        self.rotx = rotation[0]
        self.roty = rotation[1]
        self.rotz = rotation[2]
        self.rotOrder = rotation_order
        self.translation = translation
        self.transx = translation[0]
        self.transy = translation[1]
        self.transz = translation[2]

    def __str__(self):
        return 'Transformation : rotation=(' + str(self.rotx) + ', ' + str(self.roty) + ', ' + \
            str(self.rotz) + ') and translation =(' + str(self.transx) + ', ' + \
            str(self.transy) + ', ' + str(self.transz) + ')'
    
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
        Cell size for flux distribution calculation (Taille Cellules in km).
        Defines the spatial resolution for flux binning.
        Default: 0.01
    material_av : Material, optional
        Material for the object's front surface (above-view side).
        Default: Matte()
    material_ar : Material, optional
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
    def __init__(self, entity = None, name="reflector", tc = 0.01, material_av=Matte(), \
                 material_ar=Matte(), geo=Plane(), transformation=Transformation(), \
                 bbox_pmin = None, bbox_pmax = None, color = 'grey', alpha_color = 0.5):
        if isinstance(entity, Entity) :
            self.name = entity.name; self.TC = entity.TC; self.materialAV = entity.materialAV
            self.materialAR = entity.materialAR; self.geo = entity.geo
            self.transformation = entity.transformation
            #TODO: Compute automatically bboxGPmin and bboxGPmax from geo and transformation
            self.bboxGPmin = entity.bboxGPmin; self.bboxGPmax = entity.bboxGPmax
            self.color = entity.color; self.alpha_color = alpha_color
        else:
            if not isinstance(geo, (Plane, Spheric)):
                raise NameError('For the moment only Plane or a Spheric geo are accepted.')

            self.name = name
            self.TC = tc
            self.materialAV = material_av
            self.materialAR = material_ar
            self.geo = geo
            self.transformation = transformation

            # if bbox pmin and pmax are not provided compute them automatically
            # based on the geometry and transformation
            if bbox_pmin is None or bbox_pmax is None:
                box = gc.BBox()
                E_tf = self.get_transformation()
                if isinstance(self.geo, Plane):
                    box = box.union(E_tf(self.geo.p1))
                    box = box.union(E_tf(self.geo.p2))
                    box = box.union(E_tf(self.geo.p3))
                    box = box.union(E_tf(self.geo.p4))
                elif isinstance(self.geo, Spheric):
                    p1 = E_tf(gc.Point(-self.geo.radius, -self.geo.radius, self.geo.z0))
                    p2 = E_tf(gc.Point(self.geo.radius, self.geo.radius, self.geo.z1))
                    box = box.union(p1)
                    box = box.union(p2)
                if bbox_pmin is None: bbox_pmin = box.pmin
                if bbox_pmax is None: bbox_pmax = box.pmax

            self.bboxGPmin = bbox_pmin
            self.bboxGPmax = bbox_pmax
            self.color = color
            self.alpha_color = alpha_color
        self.check = "Entity"

    def __str__(self):
        return 'The entity is a ' + str(self.name) + ' with the following carac:\n' + \
            str(self.materialAV) + '\n' + \
            str(self.geo) + '\n' + \
            str(self.transformation)
    
    def get_transformation(self):
        """
        Compute the combined transformation matrix for the entity.

        Returns
        -------
        out : gc.Transform
            Combined transformation matrix (translation * rotations in specified order).
            The rotation order is determined by the entity's transformation.rotOrder 
            attribute (e.g., "XYZ", "ZYX", etc.).

        Notes
        -----
        The transformation is applied as::

            combined = Translation * Rotation_sequence

        where Rotation_sequence depends on rotOrder:
        - "XYZ": Rx * Ry * Rz
        - "XZY": Rx * Rz * Ry
        - "YXZ": Ry * Rx * Rz
        - "YZX": Ry * Rz * Rx
        - "ZXY": Rz * Rx * Ry
        - "ZYX": Rz * Ry * Rx
        """
        Trans = gc.get_translate_tf(gc.Vector(self.transformation.transx, self.transformation.transy, \
                                              self.transformation.transz))
        Rotx = gc.get_rotateX_tf(self.transformation.rotx)
        Roty = gc.get_rotateY_tf(self.transformation.roty)
        Rotz = gc.get_rotateZ_tf(self.transformation.rotz)

        # total tt of all transform together
        tt = None
        if   (self.transformation.rotOrder == "XYZ"): tt = Trans*Rotx*Roty*Rotz
        elif (self.transformation.rotOrder == "XZY"): tt = Trans*Rotx*Rotz*Roty
        elif (self.transformation.rotOrder == "YXZ"): tt = Trans*Roty*Rotx*Rotz
        elif (self.transformation.rotOrder == "YZX"): tt = Trans*Roty*Rotz*Rotx
        elif (self.transformation.rotOrder == "ZXY"): tt = Trans*Rotz*Rotx*Roty
        elif (self.transformation.rotOrder == "ZYX"): tt = Trans*Rotz*Roty*Rotx
        else: raise NameError('Unknown rotation order')

        return tt

    def set_transformation(self, transformation, recompute_bbox=True):
        """
        Update the entity's transformation and optionally recompute bounding box.

        Parameters
        ----------
        transformation : Transformation
            New transformation object containing rotation angles (rotx, roty, rotz),
            rotation order (rotOrder), and translation components (transx, transy, transz).
        recompute_bbox : bool, optional
            If True (default), recompute the bounding box (bboxGPmin and bboxGPmax)
            based on the new transformation and the entity's geometry.
            If False, keep the existing bounding box values.
            Default: True

        Notes
        -----
        The bounding box is automatically recomputed by transforming all geometry
        points using the new transformation matrix and computing their extent.

        Examples
        --------
        >>> entity = Entity(geo=Plane(...), transformation=Transformation())
        >>> new_tf = Transformation(translation=np.array([1., 2., 3.]))
        >>> entity.set_transformation(new_tf)  # Update position and recompute bbox
        >>> entity.set_transformation(new_tf, recompute_bbox=False)  # Update without bbox update
        """
        self.transformation = transformation

        if recompute_bbox:
            # Recompute bounding box based on new transformation
            box = gc.BBox()
            E_tf = self.get_transformation()
            
            if isinstance(self.geo, Plane):
                box = box.union(E_tf(self.geo.p1))
                box = box.union(E_tf(self.geo.p2))
                box = box.union(E_tf(self.geo.p3))
                box = box.union(E_tf(self.geo.p4))
            elif isinstance(self.geo, Spheric):
                p1 = E_tf(gc.Point(-self.geo.radius, -self.geo.radius, self.geo.z0))
                p2 = E_tf(gc.Point(self.geo.radius, self.geo.radius, self.geo.z1))
                box = box.union(p1)
                box = box.union(p2)
            
            self.bboxGPmin = box.pmin
            self.bboxGPmax = box.pmax


class Heliostat(object):
    """
    Composite heliostat assembly consisting of multiple facets.

    Represents a heliostat composed of multiple individual facets arranged 
    in a grid pattern.

    Parameters
    ----------
    pos : gc.Point, optional
        Heliostat position (center point) stored as a Point class.
        Default: gc.Point(0., 0., 0.)
    n_facets_x : int, optional
        Number of facet divisions in the x direction. Controls how many times
        the heliostat is split along the x-axis. Must be >= 1 (total facets >= 2).
        Default: 2
    n_facets_y : int, optional
        Number of facet divisions in the y direction. Controls how many times
        the heliostat is split along the y-axis. Must be >= 1 (total facets >= 2).
        Default: 2
    helio_size_x : float, optional
        Heliostat size in the x direction (meters).
        Default: 0.02
    helio_size_y : float, optional
        Heliostat size in the y direction (meters).
        Default: 0.02
    curve_focal_length : float | None, optional
        Focal length (in km) for curvature. If None, the focal length is computed
        automatically based on the distance to the receiver. A virtual value of
        infinity means a flat heliostat with no curvature.
        Default: None
    reflectivity : float, optional
        Reflectivity of the heliostat (between 0 and 1). Represents the
        fraction of incident radiation that is reflected.
        Default: 1.0
    roughness : float, optional
        Surface roughness of the heliostat facets.
        Default: 0
    """
    def __init__(self, pos = gc.Point(0., 0., 0.), n_facets_x=int(2), n_facets_y=int(2), helio_size_x=0.02,
                 helio_size_y=0.02, curve_focal_length=None, reflectivity=1., roughness=0):
        # Be sure that we split a heliostat by at least 2
        if (n_facets_x*n_facets_y < 2):
            raise Exception("The number of facets must be >= 2!")
        # Be sure that n_facets_x and n_facets_y are integer values
        if not ( isinstance(n_facets_x, int) and isinstance(n_facets_y, int) ):
            raise Exception("n_facets_x and n_facets_y must be integers")
        self.pos = pos
        self.sPx = n_facets_x
        self.sPy = n_facets_y
        self.hSx = helio_size_x
        self.hSy = helio_size_y
        self.curveFL = curve_focal_length
        self.ref = reflectivity
        self.rough = roughness

    def __str__(self):
        return "POS=" + str(self.pos) + '; ' + "SPX=" + str(self.sPx) + '; ' + \
                "SPY=" + str(self.sPy) + '; ' + "HSX=" + str(self.hSx) + '; ' + \
                "HSY=" + str(self.hSy)  + '; ' + "CURVE_FL=" + str(self.curveFL) + \
                '; ' + "REF=" + str(self.ref) + '; ' + "ROUGH=" + str(self.rough)


class GroupE(object):
    """Container for grouping multiple Entity objects.

    A GroupE instance represents a collection of Entity objects with a shared
    bounding box. This is useful for managing related geometric objects as a
    single unit, such as a set of heliostats or building components.

    Parameters
    ----------
    entities : list, optional
        List of Entity objects to group. Default is [Entity()].
    bbox : None | list, optional
        Custom bounding box as [Pmin, Pmax] where Pmin and Pmax are geoclide.Point
        objects. If None (default), bounding box is computed from entities[0].
    """
    def __init__(self, entities=[Entity()], bbox=None):
        self.le  = entities
        self.nob = len(entities)
        if bbox is None:
            box = gc.BBox(entities[0].bboxGPmin, entities[0].bboxGPmax)
            for i in range (1, self.nob):
                box = box.union(entities[i].bboxGPmin)
                box = box.union(entities[i].bboxGPmax)
            self.bboxGPmin = box.pmin
            self.bboxGPmax = box.pmax
        else:
            self.bboxGPmin = bbox[0]
            self.bboxGPmax = bbox[1]
        self.check = "GroupE"


def find_rots(dir_in=None, dir_out=None, normal=None):
    """Compute rotation angles to reflect an incoming ray toward an outgoing direction.

    Determines the Y and Z rotation angles necessary to orient a surface so that
    it reflects an incoming ray (dir_in) toward an outgoing direction (-dir_out). Can work
    with either incoming/outgoing ray directions or a pre-computed surface normal.

    Parameters
    ----------
    dir_in : gc.Vector, optional
        Direction vector of the incoming ray or sun direction (geoclide.Vector).
        Required unless normal is provided. Default is None.
    dir_out : gc.Vector, optional
        Direction vector of the outgoing ray, typically from receiver to facet center.
        The surface will be oriented to reflect dir_in toward -dir_out.
        Required unless normal is provided. Default is None.
    normal : gc.Vector, optional
        Pre-computed normal vector of the reflection surface (geoclide.Vector).
        If provided, dir_in and dir_out are not used. Allows direct specification of the
        desired surface normal. Default is None.

    Returns
    -------
    list
        A list containing rotation information:

        - **list[0]** : rotYD (float)
            Rotation angle around Y-axis in radians
        - **list[1]** : rotZD (float)
            Rotation angle around Z-axis in radians
        - **list[2]** : TTT (gc.Transform)
            Combined rotation transformation (geoclide.Transform object) that applies
            both rotations to orient the surface normal from (0, 0, 1) to the target direction

    Notes
    -----
    The function uses an iterative method to find rotation angles that align the
    initial surface normal (0, 0, 1) with the target normal computed from dir_in and dir_out.
    The algorithm applies Y-rotation first, then Z-rotation to achieve the desired
    reflection geometry.

    If normal is provided, it takes precedence and dir_in/dir_out are ignored.
    """
    # 1)Find the normal of the facet but filled in a vector class
    if normal is not None: vNF = gc.Vector(normal)
    else: vNF = (dir_in + dir_out)*(-0.5)
    vNF = gc.normalize(vNF)
    vNF.z = np.clip(vNF.z, -1, 1) # Avoid nan value in next operations

    # 2) Apply the inverse rotation operations to find the necessary angles
    # 2.a) Initialisation
    loop=int(0); rotY=0; rotZ=0; opeZ=0;
    # The initial value of the facet normal is (0, 0, 1) but forced to (0, 0, 0)
    # to be sure to activate the while loop below
    vNF_initial = gc.Vector(0., 0., 0.)

    # 2.b) Rotations are found in the loop bellow, at the end we check if after applying
    #      the transform to the initial normal of the facet 'vNF_initial' we have the same
    #      value as the known well oriented facet normal 'vecNF'. If no rotation has been
    #      found an error message will appear
    while (abs(vNF_initial.x - vNF.x) > 1e-4 or abs(vNF_initial.y - vNF.y) > 1e-4 or 
           abs(vNF_initial.z - vNF.z) > 1e-4):
        loop += int(1)
        if loop > 4:
            raise NameError('No rotation has been found!')

        if (loop == 1):
            rotY = np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0): opeZ = 0
            else: opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = np.arccos(opeZ)
        elif(loop == 2):
            rotY = np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0): opeZ = 0
            else: opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = -np.arccos(opeZ)
        elif(loop == 3):
            rotY = -np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0): opeZ = 0
            else: opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = np.arccos(opeZ)
        elif(loop == 4):
            rotY = -np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0): opeZ = 0
            else: opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = -np.arccos(opeZ)
 
        rotYD = np.degrees(rotY); rotZD = np.degrees(rotZ);
        TTZ = gc.get_rotateZ_tf(rotZD); TTY = gc.get_rotateY_tf(rotYD);
        TTT = TTZ*TTY
        vNF_initial = gc.normalize(TTT(gc.Vector(0., 0., 1.)))

    return [rotYD, rotZD, TTT]


def generate_mtf(heliostat=Heliostat(), receiver_pos = gc.Point(0., 0., 0.)):
    """Compute transformations for curved heliostat facet orientation.

    Generates transformation matrices for each facet of a heliostat to enable
    facet curvature. Each facet is oriented such that it reflects solar rays
    toward the center of a specified receiver position.

    Parameters
    ----------
    heliostat : Heliostat, optional
        A Heliostat class object defining the base heliostat geometry and segmentation.
        Default is Heliostat().
    receiver_pos : gc.Point, optional
        Position of the receiver center as a geoclide Point object.
        Facets are oriented to focus reflected rays toward this point.
        Default is gc.Point(0., 0., 0.).

    Returns
    -------
    MTF : 2-D ndarray of Transform
        2D array of transformation matrices (geoclide.Transform objects) of shape (SPX, SPY),
        one for each facet. Each transformation positions and orients the corresponding facet.
    """
    # Heliostat is splited in facets in x and y directions
    SPX = heliostat.sPx; SPY = heliostat.sPy
    # Size in x and y of a given facet
    SFX = heliostat.hSx/SPX; SFY = heliostat.hSy/SPY
    wMx = SFX/2; wMy = SFY/2 # Size of a facet divided by 2

    POSH = gc.Point(heliostat.pos.x, heliostat.pos.y, heliostat.pos.z)
    APOSR = gc.Point(0., 0., 0.+(POSH - receiver_pos).Length())

    # Find the positions of facets and store them in matrix MPF[i][j]
    MPF = np.zeros((SPX, SPY), dtype="object") # Matrix of Point object of each facets
    for i in range (0, SPX):
        for j in range (0, SPY):
            MPF[i][j] = gc.Point(-(heliostat.hSx/2.) + (i*SFX) + wMx, -(heliostat.hSy/2.) + (j*SFY) + wMy, 0.)

    # Find transform as function of focal length (for the curve)
    MTF = np.zeros((SPX, SPY), dtype="object") # Matrix of Transform object of each facets
    for i in range (0, SPX):
        for j in range (0, SPY):
            UI = gc.Point(0., 0., 0.) - APOSR
            UI = gc.normalize(UI)
            UO = MPF[i][j] - APOSR
            UO = gc.normalize(UO)
            RINF  = find_rots(dir_in=UI, dir_out=UO)
            MTF[i][j] = gc.Transform(RINF[2])

    return MTF


def generate_lef_h(heliostat = Heliostat(), receiver_pos = None, theta_deg = 0., phi_deg = 0., facet_transforms=None):
    """Convert a heliostat to well-oriented plane facets for receiver reflection.

    Generates a list of properly oriented planar entity/facets from a heliostat object.
    Each facet is independently oriented to reflect solar rays toward a given receiver.
    This function manages the conversion of curved or segmented heliostats into their
    constituent facet entities.

    The facet indexing follows a matrix convention based on the heliostat's segmentation
    in x and y directions. See Notes section for the indexing convention.

    Parameters
    ----------
    heliostat : Heliostat, optional
        A Heliostat class object representing the heliostat to be converted.
        Default is Heliostat().
    receiver_pos : gc.Point, optional
        Position of the receiver as a geoclide.Point object. Used to orient facets
        toward the target. If None, a default point is used. Default is None.
    theta_deg : float, optional
        Solar zenith angle in degrees. Default is 0.
    phi_deg : float, optional
        Solar azimuth angle in degrees. Default is 0.
    facet_transforms : None | 2-D ndarray, optional
        A 2D ndarray of Transform objects of dim (SPX, SPY) representing the orientation
        of each facet. If None, The transforms are computed automatically based on the
        heliostat and receiver positions.

    Returns
    -------
    out : list
        List of plane Entity objects, each representing a facet properly oriented
        to reflect solar rays toward the receiver.

    Notes
    -----
    **Facet indexing convention:**

    Each facet is identified by a two-index notation **fij** where:

    - **i** is the row index (0 to SPX-1), representing position along the x-direction
    - **j** is the column index (0 to SPY-1), representing position along the y-direction

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

    The first row contains f00, f01, f02, f03; the second row contains f10, f11, f12, f13,
    and so on. This row-major ordering allows easy identification of any facet
    from its position in the segmented heliostat grid.
    """
    # Be sure that the correct agrs have been given
    if not isinstance(heliostat, Heliostat):
        raise Exception("heliostat must be a Heliostat class!")
    if not isinstance(receiver_pos, gc.Point):
        raise Exception("The receiver position 'receiver_pos' must be a Point class!")

    # Direction of the sun (from (x,y,z) to (0,0,0))
    vSun = gc.ang2vec(theta_deg, phi_deg, vec_view="nadir")
    # Heliostat is splited in facets in x and y directions
    SPX = heliostat.sPx; SPY = heliostat.sPy;
    # Size in x and y of a given facet
    SFX = heliostat.hSx/SPX; SFY = heliostat.hSy/SPY
    # Focal length or distance between heliostat and receiver
    FL = heliostat.curveFL
    # Position of the heliostat
    POSH = gc.Point(heliostat.pos.x, heliostat.pos.y, heliostat.pos.z)
    # Receiver assumed position or the assumed focal length point.
    # Needed to curve the heliostat
    if (FL is not None):
        APOSR = gc.Point(0., 0., 0.+FL)
    else:
        PHTEMP = gc.Point(POSH)
        DTEMP = (PHTEMP - receiver_pos).length()
        APOSR = gc.Point(0., 0., 0.+DTEMP)
    # For the bounding box
    bboxDist = np.sqrt(heliostat.hSx*heliostat.hSx + heliostat.hSy*heliostat.hSy)/2

    # Initialisation
    LF = [] # List of facets
    wMx = SFX/2; wMy = SFY/2 # Size of a facet divided by 2
    # Create one facet to be ready to clone other facets
    F1 = Entity(name = "reflector", \
                material_av = Mirror(reflectivity = heliostat.ref, roughness = heliostat.rough), \
                material_ar = Matte(), \
                geo = Plane( p1 = gc.Point(-wMx, -wMy, 0.),
                             p2 = gc.Point(wMx, -wMy, 0.),
                             p3 = gc.Point(-wMx, wMy, 0.),
                             p4 = gc.Point(wMx, wMy, 0.) ), \
                transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                 translation = np.array([0., 0., 0.]) ))

    # Find the positions of facets and store them in matrix MPF[i][j]
    MPF = np.zeros((SPX, SPY), dtype="object") # Matrix of Point object of each facets
    for i in range (0, SPX):
        for j in range (0, SPY):
            MPF[i][j] = gc.Point(-(heliostat.hSx/2.) + (i*SFX) + wMx, -(heliostat.hSy/2.) + (j*SFY) + wMy, 0.)

    # Find transform as function of focal length (for the curve)
    if facet_transforms is None:
        facet_transforms = np.zeros((SPX, SPY), dtype="object") # Matrix of Transform object of each facets
        for i in range (0, SPX):
            for j in range (0, SPY):
                UI = gc.Point(0., 0., 0.) - APOSR
                UI = gc.normalize(UI)
                UO = MPF[i][j] - APOSR
                UO = gc.normalize(UO)
                RINF  = find_rots(dir_in=UI, dir_out=UO)
                facet_transforms[i][j] = gc.Transform(RINF[2])


    # Find the general heliostat rotation transform (like helistat is a unique facet)
    UI = gc.Vector(vSun.x, vSun.y, vSun.z); UO = POSH - receiver_pos;
    UI = gc.normalize(UI); UO = gc.normalize(UO);
    RINF2  = find_rots(dir_in=UI, dir_out=UO)
    TTZY = RINF2[2]

    # Apply the general rotation transform to each facet point and then apply translation.
    # This gives the final position of each facet after rotation and translation of
    # the heliostat, stored in the matrix MPFAT 
    MPFAT = np.zeros((SPX, SPY), dtype="object") # equals to MPF after application of transform
    for i in range (0, SPX):
        for j in range (0, SPY):
            tempP = gc.Point(MPF[i][j])
            tempP = TTZY(tempP)
            tempP.x += POSH.x; tempP.y += POSH.y; tempP.z += POSH.z;
            MPFAT[i][j] = gc.Point(tempP)

    # Write the initial coordinate system in term of vectors (x, y and z)
    vecX = gc.Vector(1., 0., 0.); vecY = gc.Vector(0., 1., 0.); vecZ = gc.Vector(0., 0., 1.);

    # Apply the general rotation transform to find the new coordinate system of the heliostat
    vecX = TTZY(vecX); vecY = TTZY(vecY); vecZ = TTZY(vecZ);
    vecX = gc.normalize(vecX); vecY = gc.normalize(vecY); vecZ = gc.normalize(vecZ);

    # Create the transformation matrix allowing to move between the 2 coordinate systems
    nn1 = vecX; nn2 = vecY;nn3 = vecZ; 
    mm2 = np.zeros((4,4), dtype=np.float64)
    # Fill the transformation matrix (nn3 is the new z axis)
    mm2[0,0] = nn1.x ; mm2[0,1] = nn2.x ; mm2[0,2] = nn3.x ; mm2[0,3] = 0. ;
    mm2[1,0] = nn1.y ; mm2[1,1] = nn2.y ; mm2[1,2] = nn3.y ; mm2[1,3] = 0. ;
    mm2[2,0] = nn1.z ; mm2[2,1] = nn2.z ; mm2[2,2] = nn3.z ; mm2[2,3] = 0. ;
    mm2[3,0] = 0.    ; mm2[3,1] = 0.    ; mm2[3,2] = 0.    ; mm2[3,3] = 1. ;
    # Now create the transform object with the transformation matrix and its inverse
    mm2Inv = np.transpose(mm2)
    wTo = gc.Transform(m = mm2, mInv = mm2Inv) # move from world/initial to object∕new basis
    oTw = gc.Transform(m = mm2Inv, mInv = mm2) # move from object∕new to world/initial basis

    # The normal of the heliostat vecNH = z axis of the new coordinate system
    vecNH = gc.Vector(vecZ) # stored as a vector for transformation purposes
    for i in range (0, SPX):
        for j in range (0, SPY):
            # come back to the initial coordinate system
            vecNF = oTw(vecNH)
            # apply the transform of the facet to consider the curve effect
            vecNF = facet_transforms[i][j](vecNF)
            # Now we return to the new coordinate system, which gives
            # then the normal of the facet (not heliostat) stored in facet_transforms[i][j]
            vecNF = wTo(vecNF)
            vecNF = gc.normalize(vecNF)

            # Find the rotation transform
            RINF3 = find_rots(normal=vecNF)

            # Once the rotation angles have been found, create the facet as entity object
            tempF1 = Entity(F1)
            tempF1.transformation = Transformation( rotation = np.array([0., RINF3[0], RINF3[1]]), \
                                                    translation = np.array([MPFAT[i][j].x, MPFAT[i][j].y, MPFAT[i][j].z]), \
                                                    rotation_order = "ZYX")
            tempPP = gc.Point(POSH)
            
            tempF1.bboxGPmin = gc.Point(tempPP.x-bboxDist, tempPP.y-bboxDist, tempPP.z-bboxDist)
            tempF1.bboxGPmax = gc.Point(tempPP.x+bboxDist, tempPP.y+bboxDist, tempPP.z+bboxDist)
            LF.append(tempF1)

    return LF


def generate_box(dim_xyz=[0.05, 0.05, 0.05], pos=gc.Point(0., 0., 0.), material_av = "LambMirror",
        reflectivity=[1., 1., 1., 1., 1., 1.], roughness=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2], rot_z = 0., gap=0.0001,
        obj_type="environment", colors=None, alpha_color=None):
    """Create a 3D box/building composed of six planar faces.

    Generates a box with six faces following Didier's 3D atmosphere convention in SMART-G.
    Each face can have different materials and properties. The origin is located at the
    center of the bottom face (Face 5), not at the center of the box.

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
        Dimensions of the box in [x, y, z] in kilometers. Default is [0.05, 0.05, 0.05].
    pos : gc.Point, optional
        Position of the box center. Origin is at the center of Face 5 (bottom).
        Default is gc.Point(0., 0., 0.).
    material_av : str | list, optional
        Material for the front side of faces. Either:

        - "LambMirror" : Lambertian mirror for all faces (constant reflectivity)
        - "Mirror" : Specular mirror for all faces (with roughness)
        - list : List containing 6 material objects (e.g., Matte, LambMirror, Mirror)
                 for each face

        Default is "LambMirror".
    reflectivity : list, optional
        Reflectivity values for each face when material_av is "Mirror" or "LambMirror".
        List of 6 floats, one per face. Default is [1., 1., 1., 1., 1., 1.].
        Else ignored if material_av is a list of material objects.
    roughness : list, optional
        Surface roughness for each face when material_av is "Mirror".
        List of 6 floats, one per face. Default is [0.2, 0.2, 0.2, 0.2, 0.2, 0.2].
        Else ignored if material_av is "LambMirror" or a list of material objects.
    rot_z : float, optional
        Global rotation angle in degrees around the Z-axis. Default is 0.
    gap : float, optional
        Gap to add to the global bounding box, useful for very small objects.
        Default is 0.0001.
    obj_type : str, optional
        Type of object. Choices are: 'environment', 'reflector', or 'receiver'.
        Default is 'environment'.
    colors : list, optional
        List of str colors for each of the 6 faces. If None, all faces are colored grey.
        Default is None.
    alpha_color : list, optional
        List of transparency float values (0-1) for each of the 6 faces. If None, all faces have 0.5.
        Default is None.

    Returns
    -------
    out : GroupE
        A group object (GroupE class) composed of six plane objects representing
        the box faces.

    Notes
    -----
    - Origin is at the center of Face 5 (bottom), NOT at the center of the box.
    - Global rotation in Z-axis only (other rotations not yet enabled).
    - Front side of each face uses the specified material (material_av);
      back side is always Matte (totally absorptive).
    """
    # Material AV = front part (i.e. part outside the box) of Face 0 to Face 5,
    # back part (i.e. part inside the box) will be definite as matte (totally absorbant)
    matAVL = []
    if (material_av == "Mirror") :
        for i in range (0, 6):
            matAVL.append(Mirror(reflectivity = reflectivity[i], roughness=roughness[i]))
    elif (material_av == "LambMirror") :
        for i in range (0, 6):
            matAVL.append(LambMirror(reflectivity = reflectivity[i]))
    else :
        matAVL = material_av

    # colors
    if colors is None : colors = ['grey', 'grey', 'grey', 'grey', 'grey', 'grey']
    if alpha_color is None : alpha_color = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # === Commun parameters ===
    # Compute the half dimensions in X, Y and Z
    wMx = dim_xyz[0]/2.; wMy = dim_xyz[1]/2.; wMz = dim_xyz[2]/2.

    # With the global Z rotation, 4 translations are needed in the direction after the rotation, for Face 0 to 3
    TT = gc.get_rotateZ_tf(rot_z)
    TX = gc.Vector(1., 0., 0.); TX = TT(TX); TX = gc.normalize(TX)*wMx
    TY = gc.Vector(0., 1., 0.); TY = TT(TY); TY = gc.normalize(TY)*wMy

    # Initialize a numpy array list of Points (p1 to p4 to construct a face) for all faces (from face 0 to 5)
    p1_F = np.empty(6, dtype=object); p2_F = np.empty(6, dtype=object); p3_F = np.empty(6, dtype=object); p4_F = np.empty(6, dtype=object)

    # Initialisze rotation needed to orient correctly each face
    rotX_F = np.zeros(6, dtype='float64'); rotY_F = np.zeros(6, dtype='float64')
    rotZ_F = np.full(6, rot_z) # for Z rotation it is the same value for all faces
    
    # Initialize translation variables of all faces
    transX_F = np.zeros(6, dtype='float64'); transY_F = np.zeros(6, dtype='float64'); transZ_F = np.zeros(6, dtype='float64')
    # === End commun parameters ===
    
    
    # Face 0 unique parameters
    p1_F[0] = gc.Point(-wMz, -wMy, 0.); p2_F[0] = gc.Point(wMz, -wMy, 0.); p3_F[0] = gc.Point(-wMz, wMy, 0.); p4_F[0] = gc.Point(wMz, wMy, 0.)
    rotX_F[0] = 0.; rotY_F[0] = 90.
    transX_F[0] = pos.x+TX.x; transY_F[0] = pos.y+TX.y; transZ_F[0] = pos.z + wMz
    
    # Face 1 unique parameters
    p1_F[1] = gc.Point(-wMz, -wMy, 0.); p2_F[1] = gc.Point(wMz, -wMy, 0.); p3_F[1] = gc.Point(-wMz, wMy, 0.); p4_F[1] = gc.Point(wMz, wMy, 0.)
    rotX_F[1] = 0.; rotY_F[1] = -90.
    transX_F[1] = pos.x-TX.x; transY_F[1] = pos.y-TX.y; transZ_F[1] = pos.z + wMz
    
    # Face 2 unique parameters
    p1_F[2] = gc.Point(-wMx, -wMz, 0.); p2_F[2] = gc.Point(wMx, -wMz, 0.); p3_F[2] = gc.Point(-wMx, wMz, 0.); p4_F[2] = gc.Point(wMx, wMz, 0.)
    rotX_F[2] = -90.; rotY_F[2] = 0.
    transX_F[2] = pos.x+TY.x; transY_F[2] = pos.y+TY.y; transZ_F[2] = pos.z + wMz
    
    # Face 3 unique parameters
    p1_F[3] = gc.Point(-wMx, -wMz, 0.); p2_F[3] = gc.Point(wMx, -wMz, 0.); p3_F[3] = gc.Point(-wMx, wMz, 0.); p4_F[3] = gc.Point(wMx, wMz, 0.)
    rotX_F[3] = 90.; rotY_F[3] = 0.
    transX_F[3] = pos.x-TY.x; transY_F[3] = pos.y-TY.y; transZ_F[3] = pos.z + wMz
    
    # Face 4 unique parameters
    p1_F[4] = gc.Point(-wMx, -wMy, 0.); p2_F[4] = gc.Point(wMx, -wMy, 0.); p3_F[4] = gc.Point(-wMx, wMy, 0.); p4_F[4] = gc.Point(wMx, wMy, 0.)
    rotX_F[4] = 0.; rotY_F[4] = 0.
    transX_F[4] = pos.x; transY_F[4] = pos.y; transZ_F[4] = pos.z + 2*wMz
    
    # Face 5 unique parameters
    p1_F[5] = gc.Point(-wMx, -wMy, 0.); p2_F[5] = gc.Point(wMx, -wMy, 0.); p3_F[5] = gc.Point(-wMx, wMy, 0.); p4_F[5] = gc.Point(wMx, wMy, 0.)
    rotX_F[5] = 0.; rotY_F[5] = 180.
    transX_F[5] = pos.x; transY_F[5] = pos.y; transZ_F[5] = pos.z
    
    # Create the faces and incorporate them in a list
    LOBJ = []
    for i in range (0, 6):
        F = Entity(name = obj_type, \
                   color = colors[i], \
                   alpha_color = alpha_color[i], \
                   material_av = matAVL[i], \
                   material_ar = Matte(), \
                   geo = Plane( p1 = p1_F[i], p2 = p2_F[i], p3 = p3_F[i], p4 = p4_F[i] ), \
                   transformation = Transformation( rotation = np.array([rotX_F[i], rotY_F[i], rotZ_F[i]]),
                                                    translation = np.array([transX_F[i], transY_F[i], transZ_F[i]]), rotation_order="ZXY" ))
        LOBJ.append(F)

    # Create a group of object with a global bounding box (can improve significantly the computational time!)
    maxXY = max(pos.x, 2*max(wMx, wMy))
    p_min = gc.Point( pos.x - maxXY - gap, pos.y - maxXY - gap, pos.z - gap)
    p_max = gc.Point( pos.x + maxXY + gap, pos.y + maxXY + gap, pos.z + 2*wMz + gap )
    GOBJ = GroupE(entities = LOBJ, bbox = [p_min, p_max])
    
    return GOBJ


def ref_fresnel(dir_in, geo_transform):
    """Calculate Fresnel reflection direction for a ray on a transformed surface.

    Computes the direction of a reflected ray using simple Fresnel reflection
    based on the incident ray direction and the surface transformation.

    Parameters
    ----------
    dir_in : gc.Vector
        Direction vector of the incident ray entering the reflecting surface.
    geo_transform : gc.Transform
        Transformation (rotation and translation) of the surface where reflection occurs.

    Returns
    -------
    out : gc.Vector
        Direction vector of the reflected ray.
    """
    if isinstance(dir_in, gc.Vector) :
        dirE = dir_in
    else :
        raise Exception("the dir_in argument must be a Vector class")
    if isinstance(geo_transform, gc.Transform) :
        geoT = geo_transform
    else :
        raise Exception("the geo_transform argument must be a Transform class")

    # Default value of the surface plane normal
    NN = gc.Vector(0., 0., 1)
    
    # Real value of the normal after considering transformation
    TT = geoT
    NN = TT(NN)

    # Information needed from the incoming ray
    V = dirE
    V = gc.Vector(-V.x, -V.y, -V.z)
    
    # Use the equation of Fresnel reflection (plenty explained in pbrtv3 book)
    V = dirE + NN*(2*gc.dot(NN, V))

    # Be sure V is normalized
    V = gc.normalize(V)
    
    return V


def visualize_entity(entities, theta_deg = 0., phi_deg = 0., draw_method = 'SM', ray_color = 'r',
                     sr_view=1, xyz_limit = None, show_rays=True, rs_fac = 1):
    """Enable a 3D visualization of created objects.

    Parameters
    ----------
    entities : list | Entity
        A list of Entity objects to visualize.
    theta_deg : float, optional
        The zenith angle of the sun in degrees. Default is 0.
    phi_deg : float, optional
        The azimuth angle of the sun in degrees. Default is 0.
    draw_method : str, optional
        The drawing method. 'SM' (Second Method) is the default and recommended.
        'FM' (First Method) is useful for debugging issues.
    ray_color : str, optional
        Sun rays color, e.g., 'r', 'b', 'g', etc. Default is 'r'.
    sr_view : int, optional
        Number of sun rays that can be seen in the figure. Default is 1.
    xyz_limit : dict, optional
        Dictionary specifying x, y, z view limits in km. If None (default),
        limits are automatically chosen. Example format:
        {'x_min': 0., 'x_max': 10., 'y_min': 0., 'y_max': 10., 
         'z_min': 0., 'z_max': 10.}
    show_rays : bool, optional
        Whether to show sun rays. Default is True.
    rs_fac : float, optional
        Ray scale factor. Default is 1.

    Returns
    -------
    out : matplotlib.figure.Figure
        A matplotlib figure object containing the 3D visualization.
    """

    if not isinstance(entities, (list)): entities = [entities]

    if not (all(isinstance(x, (Entity, GroupE)) for x in entities)):
        raise NameError('The only objects accepted for entities parameter are: Entity or GroupE')

    # ensure we have only Entity objects (converts if necessary GroupE to Entity objects)
    entities = convert_lg_to_le(entities)

    E = entities
    E_tf = []
    box = gc.BBox()
    for i in range(0, len(E)):
        E_tf.append(E[i].get_transformation())
        box = box.union((E[i].bboxGPmin))
        box = box.union((E[i].bboxGPmax))
 
    box_center = box.pmin + 0.5*(box.pmax - box.pmin)
    box_max_size = gc.vmax(box.pmax - box.pmin)
    pmin_n = gc.Point(box_center.x - 0.5*box_max_size, 
                      box_center.y - 0.5*box_max_size, 
                      box_center.z - 0.5*box_max_size)
    pmax_n = gc.Point(box_center.x + 0.5*box_max_size, 
                      box_center.y + 0.5*box_max_size, 
                      box_center.z + 0.5*box_max_size)
    box_n = gc.BBox(pmin_n, pmax_n)

    # calculate the sun direction vector
    vSun = gc.ang2vec(theta_deg, phi_deg, vec_view='nadir')
    wsx = -vSun.x; wsy=-vSun.y; wsz=-vSun.z

    ltmesh = []
    lMir_int = int(0)
    E_rec = []; E_ref = []
    E_rec_tf = []; E_ref_tf = []
    for i in range(0, len(E)):
        if (E[i].name == "reflector"): 
            E_ref.append(E[i])
            E_ref_tf.append(E_tf[i])
        if (E[i].name == "receiver") : 
            E_rec.append(E[i])
            E_rec_tf.append(E_tf[i])

    nbRef = len(E_ref)
    xr = [None]*nbRef; yr = [None]*nbRef; zr = [None]*nbRef
    atLeastOneInt = [False]*nbRef
    TabPhoton2 = []

    for k in range (0, len(E_ref)):
        # Get the transformation
        tt = E_ref_tf[k]

        photon_pos = gc.Point(wsx+E_ref[k].transformation.transx, wsy+E_ref[k].transformation.transy, wsz+E_ref[k].transformation.transz)
        photon = gc.Ray(o = photon_pos, d = vSun, maxt = 1200.)
    
        if isinstance(E_ref[k].geo, Plane):
           # Vertex triangle indices
            vi = np.array([np.array([0, 1, 2]),                   # indices or triangle 1
                           np.array([2, 3, 1])], dtype=np.int32)  # indices of triangle 2

            # List of points of the plane
            P = np.array([np.array([E_ref[k].geo.p1.x, E_ref[k].geo.p1.y, E_ref[k].geo.p1.z]),
                          np.array([E_ref[k].geo.p2.x, E_ref[k].geo.p2.y, E_ref[k].geo.p2.z]),
                          np.array([E_ref[k].geo.p3.x, E_ref[k].geo.p3.y, E_ref[k].geo.p3.z]),
                          np.array([E_ref[k].geo.p4.x, E_ref[k].geo.p4.y, E_ref[k].geo.p4.z])], dtype = np.float64)
            
            tmesh = gc.TriangleMesh(vertices=P, faces=vi)
        elif isinstance(E_ref[k].geo, Spheric):
            sphere = gc.Sphere(E_ref[k].geo.radius, E_ref[k].geo.z0, E_ref[k].geo.z1, E_ref[k].geo.phi)
            tmesh = sphere.to_trianglemesh()
        else: 
            raise NameError('This geometry is unknown or not yet accepted!')
        
        tmesh.apply_tf(tt)
        ltmesh.append(tmesh)

        ds = gc.calc_intersection(tmesh, photon)
        if(ds['is_intersection'].values and ds['thit'].values < float('inf')):
            atLeastOneInt[k] = True
            lMir_int += int(1)
            p_hit = gc.Point(ds['phit'].values)
            t_hit = ds['thit'].values
            tr = np.linspace(t_hit*0.98*(1/rs_fac), t_hit, 100)
            xr[k] = photon.o.x + tr*photon.d.x
            yr[k] = photon.o.y + tr*photon.d.y
            zr[k] = photon.o.z + tr*photon.d.z
            vecTemp = ref_fresnel(dir_in = photon.d, geo_transform = tt)
            TabPhoton2 = np.append(TabPhoton2, gc.Ray(o=p_hit, d=vecTemp, maxt=120))


    xr2 = [None]*lMir_int; yr2 = [None]*lMir_int; zr2 = [None]*lMir_int
    atLeastOneInt2 = [False]*lMir_int

    for k in range (0, len(E_rec)):
        # Get the transformation
        tt = E_rec[k].get_transformation()

        if isinstance(E_rec[k].geo, Plane):
            # Vertex triangle indices
            vi = np.array([np.array([0, 1, 2]),                   # indices or triangle 1
                           np.array([2, 3, 1])], dtype=np.int32)  # indices of triangle 2

            # List of points of the plane
            P = np.array([np.array([E_rec[k].geo.p1.x, E_rec[k].geo.p1.y, E_rec[k].geo.p1.z]),
                          np.array([E_rec[k].geo.p2.x, E_rec[k].geo.p2.y, E_rec[k].geo.p2.z]),
                          np.array([E_rec[k].geo.p3.x, E_rec[k].geo.p3.y, E_rec[k].geo.p3.z]),
                          np.array([E_rec[k].geo.p4.x, E_rec[k].geo.p4.y, E_rec[k].geo.p4.z])], dtype = np.float64)
            
            tmesh = gc.TriangleMesh(vertices=P, faces=vi)
        elif isinstance(E_rec[k].geo, Spheric):
            sphere = gc.Sphere(E_rec[k].geo.radius, E_rec[k].geo.z0, E_rec[k].geo.z1, E_rec[k].geo.phi)
            tmesh = sphere.to_trianglemesh()
        else:
            raise NameError('This geometry is unknown or not yet accepted!')
        tmesh.apply_tf(tt)
        ltmesh.append(tmesh)

        for i in range(0, lMir_int):
            ds = gc.calc_intersection(tmesh, TabPhoton2[i])
            if(ds['is_intersection'].values and ds['thit'].values < float('inf')):
                atLeastOneInt2[i] = True
                p_hit = gc.Point(ds['phit'].values)
                t_hit = ds['thit'].values
                tr = np.linspace(TabPhoton2[i].mint, t_hit, 100)
                xr2[i] = TabPhoton2[i].o.x + tr*TabPhoton2[i].d.x
                yr2[i] = TabPhoton2[i].o.y + tr*TabPhoton2[i].d.y
                zr2[i] = TabPhoton2[i].o.z + tr*TabPhoton2[i].d.z

    # create the matplotlib figure
    fig = plt.figure()#figsize=[128, 96])
    ax = fig.add_subplot(111, projection=Axes3D.name)
    ax.scatter([-1,1], [-1,1], [-1,1], alpha=0.0)

    for itmesh, tmesh in enumerate(ltmesh):
        # Triangles mesh parameters for plot
        # First method (draw even if there is error with an object, useful for debug):
        # ----------------------------->
        if (draw_method == 'FM'):
            for itri in range(0, tmesh.ntriangles):
                p0 = gc.Point(tmesh.vertices[tmesh.faces[itri,0],:])
                p1 = gc.Point(tmesh.vertices[tmesh.faces[itri,1],:])
                p2 = gc.Point(tmesh.vertices[tmesh.faces[itri,2],:])
                Mat = np.array([[p0.x, p0.y, p0.z], \
                                [p1.x, p1.y, p1.z], \
                                [p2.x, p2.y, p2.z]])
                face1 = mp3d.art3d.Poly3DCollection([Mat], alpha = E[itmesh].alpha_color, linewidths=0.2)
                face1.set_facecolor(mcolors.to_rgba(E[itmesh].color))
                ax.add_collection3d(face1)

        # Second method (better visual, avoid some matplotlib bugs):
        # ----------------------------->
        if (draw_method == 'SM'):
            p0_t0 = gc.Point(tmesh.vertices[tmesh.faces[0,0],:])
            p1_t0 = gc.Point(tmesh.vertices[tmesh.faces[0,1],:])
            p2_t0 = gc.Point(tmesh.vertices[tmesh.faces[0,2],:])
            p0_t1 = gc.Point(tmesh.vertices[tmesh.faces[1,0],:])
            p1_t1 = gc.Point(tmesh.vertices[tmesh.faces[1,1],:])
            p2_t1 = gc.Point(tmesh.vertices[tmesh.faces[1,2],:])
            Mat = np.array([[p0_t0.x, p0_t0.y, p0_t0.z], \
                            [p1_t0.x, p1_t0.y, p1_t0.z], \
                            [p2_t0.x, p2_t0.y, p2_t0.z], \
                            [p0_t1.x, p0_t1.y, p0_t1.z], \
                            [p1_t1.x, p1_t1.y, p1_t1.z], \
                            [p2_t1.x, p2_t1.y, p2_t1.z]])
            
            if (np.array_equal(Mat[:,0], np.full((6), Mat[0,0]))):
                yy, zz = np.meshgrid(Mat[:,0], Mat[:,2])
                xx = np.full((6,6), Mat[0,0])
                ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba(E[itmesh].color), alpha = E[itmesh].alpha_color, \
                                linewidth=0.2, antialiased=True)
            elif (np.array_equal(Mat[:,1], np.full((6), Mat[0,1]))):
                xx, zz = np.meshgrid(Mat[:,0], Mat[:,2])
                yy = np.full((6,6), Mat[0,1])
                ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba(E[itmesh].color), alpha = E[itmesh].alpha_color, \
                                linewidth=0.2, antialiased=True)
            elif (np.array_equal(Mat[:,2], np.full((6), Mat[0,2]))): # need to be verified
                xx, yy = np.meshgrid(Mat[:,0], Mat[:,1])
                zz = np.full((6,6), Mat[0,2])
                ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba(E[itmesh].color), alpha = E[itmesh].alpha_color, \
                                linewidth=0.2, antialiased=True)
            else:
                ax.plot_trisurf(Mat[:,0], Mat[:,1], Mat[:,2], color = mcolors.to_rgba(E[itmesh].color), \
                                alpha = 0.5, linewidth=0.2, antialiased=True)

    # ==============================================
    # plot all the geometries
    if (show_rays):
        for i in range(0, nbRef):
            if (atLeastOneInt[i] and i%sr_view ==0): ax.plot(xr[i], yr[i], zr[i], color=ray_color, linewidth=1*rs_fac)

        for i in range(0, lMir_int):
            if (atLeastOneInt2[i] and i%sr_view ==0): ax.plot(xr2[i], yr2[i], zr2[i], color=ray_color, linewidth=1*rs_fac)

    if (xyz_limit is not None):
        ax.set_xlim3d(xyz_limit['x_min'], xyz_limit['x_max'])
        ax.set_ylim3d(xyz_limit['y_min'], xyz_limit['y_max'])
        ax.set_zlim3d(xyz_limit['z_min'], xyz_limit['z_max'])
    else: # generic local visualization
        ax.set_xlim3d(box_n.pmin.x, box_n.pmax.x)
        ax.set_ylim3d(box_n.pmin.y, box_n.pmax.y)
        ax.set_zlim3d(box_n.pmin.z, box_n.pmax.z)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the geometries
    fig = ax.get_figure()
    return fig


def generate_h_p(theta_deg=0., phi_deg = 0., heliostat_pos_list = [gc.Point(0., 0., 0.)], receiver_pos = gc.Point(0., 0., 0.), \
                helio_size_x = 0.001, helio_size_y = 0.001, reflectivity = 1, roughness=0, heliostat_type = None, facet_transforms_list = None):
    """Generate well-oriented Heliostats from their positions.

    Generates a list of heliostat entities oriented to reflect sun rays toward
    a receiver. Can handle either planar heliostats or curved (faceted) heliostats
    depending on the heliostat_type parameter.

    Parameters
    ----------
    theta_deg : float, optional
        Sun zenith angle in degrees. Default is 0.
    phi_deg : float, optional
        Sun azimuth angle in degrees. Default is 0.
    heliostat_pos_list : list of Point, optional
        Coordinates of the center of heliostats. List of Point objects (geoclide).
        Default is [gc.Point(0., 0., 0.)].
    receiver_pos : Point, optional
        Coordinate of the center of the receiver (geoclide Point object).
        Default is gc.Point(0., 0., 0.).
    helio_size_x : float, optional
        Heliostat size in x-axis in kilometers. Default is 0.001.
    helio_size_y : float, optional
        Heliostat size in y-axis in kilometers. Default is 0.001.
    reflectivity : float, optional
        Reflectivity of the heliostats. Default is 1.
    roughness : float, optional
        Surface roughness of the heliostats. Default is 0.
    heliostat_type : Heliostat or None, optional
        If specified, must be a Heliostat class instance for generating curved
        (faceted) heliostats. If None (default), generates planar heliostats.
    facet_transforms_list : None or object, optional
        Under development. Default is None.

    Returns
    -------
    out : list
        List of Entity or GroupE objects, each properly oriented to
        reflect solar rays towards the receiver.
    """
    PH_ = heliostat_pos_list.copy()
    lObj = []

    # Case where the heliostat is totally plane
    if (heliostat_type is None):
        # compute the sun direction vector
        vSun = gc.ang2vec(theta_deg, phi_deg, vec_view='nadir')
        bboxDist = np.sqrt(helio_size_x*helio_size_x + helio_size_y*helio_size_y)/2

        Hxx = helio_size_x/2; Hyy = helio_size_y/2
        objM = Entity(name = "reflector", \
                      material_av = Mirror(reflectivity = reflectivity, roughness = roughness), \
                      material_ar = Matte(reflectivity = 0.), \
                      geo = Plane( p1 = gc.Point(-Hxx, -Hyy, 0.),
                                   p2 = gc.Point(Hxx, -Hyy, 0.),
                                   p3 = gc.Point(-Hxx, Hyy, 0.),
                                   p4 = gc.Point(Hxx, Hyy, 0.) ), \
                      transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                       translation = np.array([0., 0., 0.]) ))


        for i in range (0, len(heliostat_pos_list)):
            # 1) Find the normalized vector colinear (and same dir) to the normal of heliostat surface
            vecHR = PH_[i]-receiver_pos
            vecHR = gc.normalize(vecHR)

            # 2) Find the necessary rotations to apply on the heliostat to reflect to the receiver
            rInfo = find_rots(dir_in=vSun, dir_out=vecHR)
            rotYD = rInfo[0]; rotZD = rInfo[1];

            # 3) Once the rotation angles have been found, create heliostat objects
            objMi = Entity(objM);
            objMi.bboxGPmin = gc.Point(PH_[i].x-bboxDist, PH_[i].y-bboxDist, PH_[i].z-bboxDist)
            objMi.bboxGPmax = gc.Point(PH_[i].x+bboxDist, PH_[i].y+bboxDist, PH_[i].z+bboxDist)
            objMi.transformation = Transformation( rotation = np.array([0., rotYD, rotZD]), \
                                                   translation = np.array([PH_[i].x, PH_[i].y, PH_[i].z]), \
                                                   rotation_order = "ZYX")
            lObj.append(objMi)
    # Case where the heliostat is composed by facets (i.g. to consider the curvature)
    else:
        # Take the commun parameters of all heliostats
        SPX = heliostat_type.sPx; SPY = heliostat_type.sPy; helio_size_x = heliostat_type.hSx; helio_size_y = heliostat_type.hSy; CURVE_FL = heliostat_type.curveFL;

        # Generate all the facets and store them as entity object in a list
        for i in range (0, len(heliostat_pos_list)):
            H0 = Heliostat(n_facets_x=SPX, n_facets_y=SPY, helio_size_x=helio_size_x, helio_size_y=helio_size_y, curve_focal_length=CURVE_FL, pos=PH_[i], reflectivity=reflectivity, roughness=roughness)
            if facet_transforms_list is None: TLE = generate_lef_h(heliostat=H0, receiver_pos=receiver_pos, theta_deg=theta_deg, phi_deg=phi_deg)
            else: TLE = generate_lef_h(heliostat=H0, receiver_pos=receiver_pos, theta_deg=theta_deg, phi_deg=phi_deg, facet_transforms = facet_transforms_list[i])
            GTEMP = GroupE(entities = TLE)
            lObj.append(GTEMP)

    return lObj


def generate_h_a(theta_deg=0., phi_deg = 0., receiver_pos = gc.Point(0., 0., 50.), min_ang_deg=0., \
                max_ang_deg=360., gap_ang_deg = 5., first_dist = 0.1, n_heliostats = 10, gap_dist = 0.01, \
                helio_size_x = 0.001, helio_size_y = 0.001, pillar_height = 0.006, reflectivity = 1, roughness=0,
                heliostat_type=None, facet_transforms_list = None, return_positions = False):
    """Generate well-oriented Heliostats arranged in an angular sector around receiver.

    Generates heliostats positioned between min_ang_deg and max_ang_deg angles, properly
    oriented to reflect sun rays toward a central receiver. Heliostats are arranged
    in concentric patterns with specified angular and radial gaps.

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
        Coordinate of the center of the receiver (geoclide Point object).
        Heliostats are filled between min_ang_deg and max_ang_deg around this receiver.
        Default is gc.Point(0., 0., 50.).
    min_ang_deg : float, optional
        Minimum angular position in degrees. Default is 0.
    max_ang_deg : float, optional
        Maximum angular position in degrees. Default is 360.
    gap_ang_deg : float, optional
        Angular spacing in degrees for placing heliostats between min_ang_deg and max_ang_deg.
        Default is 5.
    first_dist : float, optional
        First distance between receiver and heliostat center in kilometers.
        Default is 0.1.
    n_heliostats : int, optional
        Number of heliostats to place at each angular position (radial direction).
        Default is 10.
    gap_dist : float, optional
        Radial gap between heliostats in kilometers after the first distance first_dist.
        Default is 0.01.
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
        If specified, must be a Heliostat class instance for generating curved
        (faceted) heliostats. If None (default), generates planar heliostats.
    facet_transforms_list : None or object, optional
        Under development. Default is None.
    return_positions : bool, optional
        If True, also return the list of heliostat positions. Default is False.

    Returns
    -------
    out1 : list
        List of heliostat Entity or GroupE objects arranged in the angular sector.
    out2 : list
        If return_positions is True, also returns the list of heliostat center positions
        (geoclide Point objects).
    """
    # I) Find the position of all heliostats
    lenpH = int(  ( (max_ang_deg-min_ang_deg)/gap_ang_deg )*n_heliostats  )

    # To avoid a given bug
    if (max_ang_deg-min_ang_deg < 360.000000001 and max_ang_deg-min_ang_deg > 359.999999999):
        nbI = int(lenpH/n_heliostats)
    else:
        nbI = int(lenpH/n_heliostats) + 1

    print("Total number of Heliostats = ", nbI*n_heliostats)

    pH = []
    myRotZ = min_ang_deg

    if (min_ang_deg != max_ang_deg):
        for i in range (0, nbI):
            Dhr = first_dist
            for j in range (0, n_heliostats):
                myP = gc.Point(Dhr, 0., 0.)
                RotZT = gc.get_rotateZ_tf(myRotZ)
                myP=RotZT(myP)
                pH.append( gc.Point(myP.x, myP.y, myP.z+pillar_height) )
                Dhr += gap_dist
            myRotZ += gap_ang_deg
    else:
        Dhr = first_dist
        RotZT = gc.get_rotateZ_tf(myRotZ)
        for j in range (0, n_heliostats):
            myP = gc.Point(Dhr, 0., 0.)
            myP=RotZT(myP)
            pH.append( gc.Point(myP.x, myP.y, myP.z+pillar_height) )
            Dhr += gap_dist


    # II) Creation of heliostats
    lObj = []

    # Case where the heliostat is totally plane
    if (heliostat_type is None):
        # calculate the sun direction vector
        vSun = gc.ang2vec(theta_deg, phi_deg, vec_view='nadir')
        bboxDist = np.sqrt(helio_size_x*helio_size_x + helio_size_y*helio_size_y)/2

        Hxx = helio_size_x/2; Hyy = helio_size_y/2
        objM = Entity(name = "reflector", \
                      material_av = Mirror(reflectivity = reflectivity, roughness = roughness), \
                      material_ar = Matte(), \
                      geo = Plane( p1 = gc.Point(-Hxx, -Hyy, 0.),
                                   p2 = gc.Point(Hxx, -Hyy, 0.),
                                   p3 = gc.Point(-Hxx, Hyy, 0.),
                                   p4 = gc.Point(Hxx, Hyy, 0.) ), \
                      transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                       translation = np.array([0., 0., 0.]) ))

        for i in range (0, len(pH)):
            # 1) The vector of the photon after a reflection (here the opposite direction)
            vecHR = pH[i]-receiver_pos
            vecHR = gc.normalize(vecHR)

            # 2) The incoming (vSun) and outcoming (vecHR) directions are known then find
            #    the rotation angles
            rInfo = find_rots(dir_in=vSun, dir_out=vecHR)
            rotYD = rInfo[0]; rotZD = rInfo[1]

            # 3) Once the rotation angles have been found, create heliostat objects
            objMi = Entity(objM)
            objMi.bboxGPmin = gc.Point(pH[i].x-bboxDist, pH[i].y-bboxDist, pH[i].z-bboxDist)
            objMi.bboxGPmax = gc.Point(pH[i].x+bboxDist, pH[i].y+bboxDist, pH[i].z+bboxDist)
            objMi.transformation = Transformation( rotation = np.array([0., rotYD, rotZD]), \
                                                   translation = np.array([pH[i].x, pH[i].y, pH[i].z]), \
                                                   rotation_order = "ZYX")
            lObj.append(objMi)

    # Case where the heliostat is composed by facets (i.g. to consider the curvature)
    else:
        # Take the commun parameters of all heliostats
        SPX = heliostat_type.sPx; SPY = heliostat_type.sPy; helio_size_x = heliostat_type.hSx; helio_size_y = heliostat_type.hSy; CURVE_FL = heliostat_type.curveFL

        # Generate all the facets and store them as entity object in a list
        for i in range (0, len(pH)):
            H0 = Heliostat(n_facets_x=SPX, n_facets_y=SPY, helio_size_x=helio_size_x, helio_size_y=helio_size_y, curve_focal_length=CURVE_FL, pos=pH[i], reflectivity=reflectivity, roughness=roughness)
            if facet_transforms_list is None: TLE = generate_lef_h(heliostat=H0, receiver_pos=receiver_pos, theta_deg=theta_deg, phi_deg=phi_deg)
            else: TLE = generate_lef_h(heliostat=H0, receiver_pos=receiver_pos, theta_deg=theta_deg, phi_deg=phi_deg, facet_transforms = facet_transforms_list[i])
            GTEMP = GroupE(entities = TLE)
            lObj.append(GTEMP)

    if (return_positions):
        return lObj, pH
    else:
        return lObj
    

def convert_lg_to_le(obj_list):
    """Convert a mixed list of Entity and GroupE objects to Entity objects only.

    Flattens groups by expanding all GroupE objects into their constituent
    Entity objects, resulting in a list containing only Entity objects.

    Parameters
    ----------
    obj_list : list
        List containing Entity and/or GroupE objects to be converted.

    Returns
    -------
    out : list
        Flattened list containing only Entity objects. GroupE objects are
        converted into their constituent Entity objects.
    """
    nGObj=len(obj_list)
    LOBJ=[]

    for i in range (0, nGObj):
        if isinstance(obj_list[i], GroupE):
            LOBJ.extend(obj_list[i].le)
        elif isinstance(obj_list[i], Entity):
            LOBJ.append(obj_list[i])
        else:
            raise NameError('In the list, only Entity and GroupE classes are autorised!')

    return LOBJ


def rotate_vector(vector, rot_x, rot_y, rot_z, rotation_order="xyz"):
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
    rotation_order : str with the order of rotations i.g. 'xyz', zxy', ...

    Return:
    rotated_vector : The rotated (normalized) direction (also a Vector class)
    """
    TT = gc.Transform()
    tr_x = gc.get_rotateX_tf(rot_x)
    tr_y = gc.get_rotateY_tf(rot_y)
    tr_z = gc.get_rotateZ_tf(rot_z)
    if rotation_order == "XYZ":
        TT = tr_x*tr_y*tr_z
    elif rotation_order == "XZY":
        TT = tr_x*tr_z*tr_y
    elif rotation_order == "YXZ":
        TT = tr_y*tr_x*tr_z
    elif rotation_order == "YZX":
        TT = tr_y*tr_z*tr_x
    elif rotation_order == "ZXY":
        TT = tr_z*tr_x*tr_y
    elif rotation_order == "ZYX":
        TT = tr_z*tr_y*tr_x
    else:
        raise NameError("Unknown rotation_order value!")
    rotated_vector = TT(vector)
    rotated_vector = gc.normalize(rotated_vector)

    return rotated_vector


def interpolate_refls_from_wls (wavelengths, reflectivities, new_wavelengths, extrapolate=False):
    """
        Definition: Giving a set of wavelengths (wavelengths) and reflectivities (reflectivities),
                    get the interpolated reflectivities folowing the new set of wavelengths (new_wavelengths)

    ==== ARGS:
    wavelengths     : List/array of wavelengths
    reflectivities  : List/array with reflectivities at each wavelength of wavelengths
    new_wavelengths : List/array of the new wavelengths where we want to interpolate

    ==== RETURN:
    refls_new : numpy array with the interpolated reflectivities
    """

    if extrapolate: f = interpolate.interp1d(wavelengths, reflectivities, fill_value='extrapolate')
    else : f = interpolate.interp1d(wavelengths, reflectivities, fill_value=(reflectivities[0],reflectivities[-1]), bounds_error=False)

    refls_new = f(new_wavelengths)

    # Ensure relfectivities are between 0 and 1
    refls_new[refls_new<0] = 0
    refls_new[refls_new>1] = 1

    return refls_new


def is_comment(line):
    """
    function to check if a line
    starts with some character.
    Here # for comment
    """
    # return true if a line starts with #
    return line.startswith('#')


def extract_points(filename):
    """Extract heliostat coordinates from a file.

    Reads a file and extracts the (x, y, z) coordinates of each heliostat,
    returning them as geoclide Point objects.

    The input file must follow this format:

    - First line: comment line beginning with '#'
    - Second line: empty line
    - Subsequent lines: x, y, and z coordinates of each heliostat, separated by commas

    Parameters
    ----------
    filename : str | pathlib.Path
        Path to the file containing the heliostat coordinates.

    Returns
    -------
    out : list
        List of geoclide.Point objects, each containing the x, y, and z coordinates
        of a heliostat.
    """

    # First check if filename is an str type
    try:
        with open(filename, "r") as file:
            for curline in dropwhile(is_comment, file):
                insideFile = file.read()
    except FileNotFoundError:
        print(str(filename) + ' has been not found')
    except IOError:
        print("Enter/Exit error with " + str(filename))
            
    # Looking for a float and fill it in listVal
    listVal = re.findall(r"-?[0-9]+\.?[0-9]*", insideFile)
        
    # Number of dimension and number of heliostats
    nbDim = 3 # x, y and z --> 3 dim
    nbH = int(len(listVal)/nbDim)

    # # Fill the x, y and z coordinates into a list of Point classes
    lPH = []
    for i in range (0, nbH):
        lPH.append(  gc.Point( float(listVal[i*nbDim]), float(listVal[(i*nbDim)+1]),
                               float(listVal[(i*nbDim)+2]) )  )

    return lPH
