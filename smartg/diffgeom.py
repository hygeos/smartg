#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This module is deprecated. It will be removed in one of the next versions of smartg.
# All the classes/functions are now available in the geoclide package.

from smartg.geometry import Vector, Point, Normal, Ray
from smartg.geometry import Dot, Cross, Normalize, CoordinateSystem, \
    Distance, FaceForward
from warnings import warn


#####################################################################################
class DifferentialGeometry(object):
    '''
    Declare common parameters, enable to shade some parameters and then declare in
    other files several shapes without the need to distinguish them.
    '''
    def __init__(self, p, dpdu, dpdv, uu, vv, shape = None):
        warn_message = 'DifferentialGeometry is depracated as of smartg v1.1.0. ' + \
                       'and will be removed in one of the next versions of smartg.'
        warn(warn_message, DeprecationWarning)
        if isinstance(p, Point) and isinstance(dpdu, Vector) and \
           isinstance(dpdv, Vector):
            self.p = p
            self.dpdu = dpdu
            self.dpdv = dpdv
            # self.dndu = dndu
            # self.dndv = dndv
            self.nn = Normal(Normalize(Cross(self.dpdu, self.dpdv)))
            self.u = uu
            self.v = vv
            # self.dudx = 0
            # self.dvdx = 0
            # self.dudy = 0
            # self.dvdy = 0
            self.shape = shape
        else:
            raise NameError('Problem with the type of argument(s)')
#####################################################################################

if __name__ == '__main__':
    
    a = Vector(10, 2, 3)
    b = Point(2, 4, 6)
    c = Vector(b)*2
    p1 = Point(4, 8, 12)
    print("a =", a)
    print("b =", b)
    print("c =", c)
    print("Dot(a, c) =", Dot(a, c))
    print("Dot(a, b) =", Dot(a, Vector(b)))
    print("Cross(a, c) =", Cross(a, c))
    print("Lengh(a) =", a.Lengh())
    print("normalize(a) =", Normalize(a))
    print("Lengh of Normalize(a) =", Normalize(a).Lengh())

    v1 = Vector(2, 1, 0)
    v2, v3 = CoordinateSystem(v1)

    print("v1 =", v1)
    print("v2 =", v2)
    print("v3 =", v3)
    print("p2 =", p1-b)
    print("type(p2) =", type(p1-b))
    print("Distance(b, p1) =",  Distance(b, p1))
    print("a*2 =", a*2)

    vec1 = Vector (-2, 4, -1)
    vec2 = Vector (5, 1, 3)

    print("FaceForward(vec1, vec2) =", FaceForward(vec1, vec2))

    rayon = Ray(p1, vec2)
    print("p1 =", rayon.o)
    print("vec2 =", rayon.d)
    print("rayon(t=1) =", rayon[1])
    print("rayon(t=2) =", rayon[2])
