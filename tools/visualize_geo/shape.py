#!/usr/bin/env python
# -*- coding: utf-8 -*-

import geometry
from geometry import Vector, Point, Normal, Ray, BBox
from geometry import Dot, Cross, Normalize, CoordinateSystem, \
    Distance, FaceForward
import diffgeom
from diffgeom import DifferentialGeometry
import transform
from transform import Transform
import math 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#####################################################################################
class Shape(object):
    '''
    Creation of the class Shape
    '''
    indShape = 0
    def __init__(self, ObjectToWorld, WorldToObject):
        Shape.indShape += 1
        self.oTw = ObjectToWorld
        self.wTo = WorldToObject
#####################################################################################

#####################################################################################
class Sphere(Shape):
    '''
    Creation of the class Sphere
    '''
    def __init__(self, oTw, wTo, rad, z0, z1, pm):
        Shape.__init__(self, ObjectToWorld = oTw, WorldToObject = wTo)
        self.radius = rad
        self.zmin = Clamp(min(z0, z1), -self.radius, self.radius)
        self.zmax = Clamp(max(z0, z1), -self.radius, self.radius)
        self.thetaMin = math.acos(Clamp(self.zmin/self.radius, -1, 1))
        self.thetaMax = math.acos(Clamp(self.zmax/self.radius, -1, 1))
        self.phiMax = math.radians(Clamp(pm, 0, 360))
        Shape.indShape += 1

    def Intersect(self, r):

        ray = Ray(r)
        ray.o = self.wTo[r.o]
        ray.d = self.wTo[r.d]

        print "MRAYY", ray.o
        self.thit = None; self.dg = None

        # Compute quadratic sphere coefficients
        A = ray.d.x*ray.d.x + ray.d.y*ray.d.y + ray.d.z*ray.d.z
        B = 2 * (ray.d.x*ray.o.x + ray.d.y*ray.o.y + ray.d.z*ray.o.z)
        C = ray.o.x*ray.o.x + ray.o.y*ray.o.y + ray.o.z*ray.o.z - \
            self.radius*self.radius

        # Solve quadratic equation for _t_ values
        exist = None; t0 = None; t1 = None;
        exist, t0, t1 = Quadratic(exist, t0, t1, A, B, C)
        if (exist == False): return False
        
        # Compute intersection distance along ray
        if (t0 > ray.maxt or t1 < ray.mint): return False
        self.thit = t0
        if (t0 < ray.mint):
            self.thit = t1
            if (self.thit > ray.maxt): return False

        # Compute sphere hit position and $\phi$
        phit = ray[self.thit]
        if (phit.x == 0 and phit.y == 0): phit.x = 1e-5 * self.radius
        phi = math.atan2(phit.y, phit.x)
        if (phi < 0): phi += 2*math.pi

        #Test sphere intersection against clipping parameters
        if ((self.zmin > -self.radius and phit.z < self.zmin) or \
            (self.zmax <  self.radius and phit.z > self.zmax) or phi > self.phiMax):
            if (self.thit == t1): return False
            if (t1 > ray.maxt): return False
            self.thit = t1
            # Compute sphere hit position and $\phi$
            phit = ray[self.thit]
            if (phit.x == 0 and phit.y == 0): phit.x = 1e-5 * self.radius
            phi = math.atan2(phit.y, phit.x)
            if (phi < 0): phi += 2*math.pi
            if ((self.zmin > -self.radius and phit.z < self.zmin) or \
                (self.zmax <  self.radius and phit.z > self.zmax) or \
                phi > self.phiMax):
                return False;

        #Find parametric representation of sphere hit
        u = phi / self.phiMax
        theta = math.acos(Clamp(phit.z / self.radius, -1, 1))
        v = (theta - self.thetaMin) / (self.thetaMax - self.thetaMin)

        # Compute sphere $\dpdu$ and $\dpdv$
        zradius = math.sqrt(phit.x*phit.x + phit.y*phit.y)
        invzradius = 1 / zradius
        cosphi = phit.x * invzradius
        sinphi = phit.y * invzradius
        dpdu = Vector(-self.phiMax * phit.y, self.phiMax * phit.x, 0);
        dpdv = Vector(Vector(phit.z * cosphi, phit.z * sinphi, -self.radius \
                        * math.sin(theta)) * (self.thetaMax-self.thetaMin))

        # Initialize _DifferentialGeometry_ from parametric information
        self.dg = DifferentialGeometry(self.oTw[phit], self.oTw[dpdu], self.oTw[dpdv], u, v, self)

        return True

    def Area(self):
        return (self.phiMax * self.radius * (self.zmax-self.zmin))
#####################################################################################

#####################################################################################
class Triangle(Shape):
    '''
    Creation of the class Sphere
    '''
    def __init__(self, oTw, wTo, a = Point(), b = Point(), c = Point()):
        Shape.__init__(self, ObjectToWorld = oTw, WorldToObject = wTo)
        self.p1 = a; self.p2 = b; self.p3 = c;

    def Intersect(self, r):
        ray = Ray(r)
        e1 = self.p2 - self.p1
        e2 = self.p3 - self.p1
        s1 = Cross(ray.d, e2)
        divisor = Dot(s1, e1)

        if (divisor == 0):
            return False
        invDivisor = 1./divisor

        # compute the first barycentric coordinate
        s = ray.o - self.p1
        b1 = Dot(s, s1) * invDivisor
        if (b1 < 0 or  b1 > 1):
            return False

        # compute the second barycentric coordinate
        s2 = Cross(s, e1)
        b2 = Dot(ray.d, s2) * invDivisor
        if (b2 < 0 or  b1+b2 > 1):
            return False

        # compute the time at the intersection point
        t = Dot(e2, s2) * invDivisor
        if (t < ray.mint or t > ray.maxt):
            return False

        # compute triangle partial derivatives
        uvs = np.array([[0., 0.], [1., 0.], [1., 1.]])

        # compute deltas for triangle partial derivatives
        du1 = uvs[0][0] - uvs[2][0]
        du2 = uvs[1][0] - uvs[2][0]
        dv1 = uvs[0][1] - uvs[2][1]
        dv2 = uvs[1][1] - uvs[2][1]
        dp1 = self.p1 - self.p3
        dp2 = self.p2 - self.p3
        determinant = du1 * dv2 - dv1 * du2

        if (determinant == 0):
            dpdu, dpdv = CoordinateSystem(Normalize(Cross(e2, e1)))
        else:
            invdet = 1./determinant
            myV1 = dp1*dv2
            myV2 = dp2*dv1
            print type(myV1), type(myV2)
            dpdu = ( dp1*dv2   - dp2*dv1) * invdet
            dpdv = (dp1*(-du2) + dp2*du1) * invdet
        
        # interpolate $(u,v)$ triangle parametric coordinates
        b0 = 1 - b1 - b2
        tu = b0*uvs[0][0] + b1*uvs[1][0] + b2*uvs[2][0]
        tv = b0*uvs[0][1] + b1*uvs[1][1] + b2*uvs[2][1]

        # fill the DifferentialGeometry and thit
        self.dg = DifferentialGeometry(ray[t], dpdu, dpdv, tu, tv, self)
        self.thit = t

        return True
#####################################################################################

#####################################################################################
class TriangleM(Shape):
    '''
    Creation of the class Sphere
    '''
    def __init__(self, oTw, wTo, a = Point(), b = Point(), c = Point()):
        Shape.__init__(self, ObjectToWorld = oTw, WorldToObject = wTo)
        self.p1 = a; self.p2 = b; self.p3 = c;

    def Intersect(self, r):
        ray = Ray(r)
        e1 = self.p2 - self.p1
        e2 = self.p3 - self.p1
        s1 = Cross(ray.d, e2)
        divisor = Dot(s1, e1)

        if (divisor == 0):
            return False
        invDivisor = 1./divisor

        # compute the first barycentric coordinate
        s = ray.o - self.p1
        b1 = Dot(s, s1) * invDivisor
        if (b1 < 0 or  b1 > 1):
            return False

        # compute the second barycentric coordinate
        s2 = Cross(s, e1)
        b2 = Dot(ray.d, s2) * invDivisor
        if (b2 < 0 or  b1+b2 > 1):
            return False

        # compute the time at the intersection point
        t = Dot(e2, s2) * invDivisor
        if (t < ray.mint or t > ray.maxt):
            return False

        # compute triangle partial derivatives
        uvs = np.array([[0., 0.], [1., 0.], [1., 1.]])

        # compute deltas for triangle partial derivatives
        du1 = uvs[0][0] - uvs[2][0]
        du2 = uvs[1][0] - uvs[2][0]
        dv1 = uvs[0][1] - uvs[2][1]
        dv2 = uvs[1][1] - uvs[2][1]
        dp1 = self.p1 - self.p3
        dp2 = self.p2 - self.p3
        determinant = du1 * dv2 - dv1 * du2

        if (determinant == 0):
            dpdu, dpdv = CoordinateSystem(Normalize(Cross(e2, e1)))
        else:
            invdet = 1./determinant
            myV1 = dp1*dv2
            myV2 = dp2*dv1
            print type(myV1), type(myV2)
            dpdu = ( dp1*dv2   - dp2*dv1) * invdet
            dpdv = (dp1*(-du2) + dp2*du1) * invdet
        
        # interpolate $(u,v)$ triangle parametric coordinates
        b0 = 1 - b1 - b2
        tu = b0*uvs[0][0] + b1*uvs[1][0] + b2*uvs[2][0]
        tv = b0*uvs[0][1] + b1*uvs[1][1] + b2*uvs[2][1]

        # fill the DifferentialGeometry and thit
        self.dg = DifferentialGeometry(ray[t], dpdu, dpdv, tu, tv, self)
        self.thit = t

        return True
#####################################################################################

#####################################################################################
class TriangleMesh(Shape):
    '''
    Creation of the class Sphere
    '''
    def __init__(self, oTw, wTo, vi, P):
        Shape.__init__(self, ObjectToWorld = oTw, WorldToObject = wTo)
        if isinstance(vi, np.ndarray) and isinstance(P, np.ndarray):
            self.VertexIndex = vi
            self.p = P
            self.ntris = len(np.atleast_1d(vi))/3
            self.nverts = np.amax(self.VertexIndex) + 1
            self.reftri = [None]*self.ntris
            for i in xrange(0, self.ntris):
                PointA = Point(self.p[self.VertexIndex[3*i]*3], self.p[self.VertexIndex[3*i]*3 + 1],
                               self.p[self.VertexIndex[3*i]*3 + 2])
                PointB = Point(self.p[self.VertexIndex[3*i + 1]*3], self.p[self.VertexIndex[3*i + 1]*3 + 1],
                               self.p[self.VertexIndex[3*i + 1]*3 + 2])
                PointC = Point(self.p[self.VertexIndex[3*i + 2]*3], self.p[self.VertexIndex[3*i + 2]*3 + 1],
                               self.p[self.VertexIndex[3*i + 2]*3 + 2])
                self.reftri[i] = TriangleM(Transform(), Transform(), PointA, PointB, PointC)
        else:
            raise NameError('error: vi or/and P are not numpy array***')

    def Intersect(self, r):
        self.dg = None
        self.thit = float("inf")
        for i in xrange(0, self.ntris):
            mybool = self.reftri[i].Intersect(r)
            if mybool:
                TriThit = self.reftri[i].thit
                if self.thit > TriThit:
                    self.thit = TriThit
                    self.dg = self.reftri[i].dg
        
        if self.dg == None:
            return False
            
        return True
#####################################################################################
def swap(a, b):
    return b, a

def Clamp(val, low, high):
    if (val < low): return low
    elif (val > high): return high
    else: return val

def Quadratic(exist, t0, t1, A, B, C):
    #  Find quadratic discriminant
    discrim = (B * B) - (4 * A * C)
    if (discrim < 0): return False, None, None
    rootDiscrim = math.sqrt(discrim)

    # Compute quadratic _t_ values
    if (B < 0):
        q = -0.5 * (B - rootDiscrim)
    else:
        q = -0.5 * (B + rootDiscrim)
    if (A != 0):
        t0 = q / A
    else:
        t0 = C / q
    t1 = C / q
    if (t0 > t1): t0, t1 = swap(t0, t1)
    return True, t0, t1
#####################################################################################

#####################################################################################
if __name__ == '__main__':

    uvs = np.array([[0, 0], [1, 0], [1, 1]])
    print "uvs[0, 0] =", uvs[0, 0], "uvs[0, 1] =", uvs[0, 1]
    print "uvs[1, 0] =", uvs[1, 0], "uvs[1, 1] =", uvs[1, 1]
    print "uvs[2, 0] =", uvs[2, 0], "uvs[2, 1] =", uvs[2, 1]
    
    aVec = np.array(30)
    aVec2 = np.zeros(5, dtype=np.float32)
    aVec2[0] = 1
    aVec2[2] = 368
    aVec2[3] = 68
    print "lenght =", len(np.atleast_1d(aVec)), type(aVec)
    print "lenght =", len(np.atleast_1d(aVec2)), type(aVec2), "max=", np.amax(aVec2), "min=",np.amin(aVec2)
 
    ARAY = Ray(Point(0, 0, 10), Vector(-0.3, 0.5, -1), 0, 0, 30)
    ARAY2 = Ray(Point(0, 0, 10), Vector(-1.8, 1.9, -1), 0, 0, 30)
    vi = np.array([0, 1, 2,
                   2, 3, 1], dtype=np.int32)
    P = np.array([0,   0,  0,
                  -30, 0,  0,
                  0,   30, 0,
                  -30, 30, 0], dtype=np.float32)
    trimesh = TriangleMesh(Transform(), Transform(), vi, P)
    bool2 = trimesh.Intersect(ARAY2)
    if bool2:
        print "There is an intersection with the triangle Mesh***"
        print "At p = (", trimesh.dg.p.x, ", ", trimesh.dg.p.y, ", ", trimesh.dg.p.z, ")"
        tHitmesh = trimesh.thit
    else:
        print "No intersection with the triangle Mesh***"


    A1 = Point(0, 0, 0) ; A2 = Point(-30, 0, 0); A3 = Point(0, 30, 0);
    tri =  Triangle(Transform(), Transform(), A1, A2, A3)
    bool1 = tri.Intersect(ARAY)
    if bool1:
        print "There is an intersection with the triangle***"
        print "At p = (", tri.dg.p.x, ", ", tri.dg.p.y, ", ", tri.dg.p.z, ")"
        tHit = tri.thit
    else:
        print "No intersection with the triangle***"

    # transform needed for the first sphere
    TSph1 = Transform()

    TRX = TSph1.rotateX(90)
    TSph1 = TRX
    invTSph1 = TSph1.inverse(TSph1)

    # create the first sphere
    Sph1 = Sphere(TSph1, invTSph1, 60, -60, 60, 29.9)
    # ================================================
    # transform needed for the second sphere
    TSph2 = Transform()

    TTrans = TSph2.translate(Vector(-110, 0, 0))
    TSph2 = TTrans
    invTSph2 = TSph2.inverse(TSph2)

    # create the second sphere
    Sph2 = Sphere(TSph2, invTSph2, 50, -50, 50, 360)


    # create the matplotlib figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # parameters of the first sphere
    u1 = np.linspace(0, Sph1.phiMax, 20)
    v1 = np.linspace(Sph1.thetaMin, Sph1.thetaMax, 20)
    myPoint1 = TSph1[Point(Sph1.radius * np.outer(np.cos(u1), np.sin(v1)), Sph1.radius * np.outer(np.sin(u1), np.sin(v1)), Sph1.radius * np.outer(np.ones(np.size(u1)), np.cos(v1)))]
    x1 = myPoint1.x
    y1 = myPoint1.y
    z1 = myPoint1.z

    # parameters of the second sphere
    u2 = np.linspace(0, Sph2.phiMax, 20)
    v2 = np.linspace(Sph2.thetaMin, Sph2.thetaMax, 20)
    myPoint2 = TSph2[Point(Sph2.radius * np.outer(np.cos(u2), np.sin(v2)), Sph2.radius * np.outer(np.sin(u2), np.sin(v2)), Sph2.radius * np.outer(np.ones(np.size(u2)), np.cos(v2)))]
    x2 = myPoint2.x
    y2 = myPoint2.y
    z2 = myPoint2.z

    # Photon paramaters plot thanks to straight equation (SunZ = A*SunX + b)
    # where SunY is constant and equal to 0
    tan_S = np.tan(60*(np.pi / 180.))      # zenith angle at 60°
    SunX = np.linspace(0, 120*tan_S, 100)
    xs = SunX
    ys = 0 * SunX                          # multiply by SunX to get an array
    zs = (1./tan_S) * SunX

    # Triangles parameters
    # v1 = [0, 0, 0]
    # v2 = [-30, 0, 0]
    # v3 = [0, 30, 0]

    # x = [v1[0], v2[0], v3[0]]
    # y = [v1[1], v2[1], v3[1]]
    # z = [v1[2], v2[2], v3[2]]
    # verts = [zip(x, y, z)]
    # ax.add_collection3d(Poly3DCollection(verts, facecolors='orange', linewidths=2))

    # Triangles mesh parameters
    TrV1 = [None]*trimesh.ntris
    TrV2 = [None]*trimesh.ntris
    TrV3 = [None]*trimesh.ntris
    Trx = [None]*trimesh.ntris
    Try = [None]*trimesh.ntris
    Trz = [None]*trimesh.ntris
    for i in xrange(0, trimesh.ntris):
        print "p1 = (", trimesh.reftri[i].p1.x, ", ", trimesh.reftri[i].p1.y, ", ", trimesh.reftri[i].p1.z, ")"
        print "p2 = (", trimesh.reftri[i].p2.x, ", ", trimesh.reftri[i].p2.y, ", ", trimesh.reftri[i].p2.z, ")"
        print "p3 = (", trimesh.reftri[i].p3.x, ", ", trimesh.reftri[i].p3.y, ", ", trimesh.reftri[i].p3.z, ")"
        TrV1[i] = [trimesh.reftri[i].p1.x, trimesh.reftri[i].p1.y, trimesh.reftri[i].p1.z]
        TrV2[i] = [trimesh.reftri[i].p2.x, trimesh.reftri[i].p2.y, trimesh.reftri[i].p2.z]
        TrV3[i] = [trimesh.reftri[i].p3.x, trimesh.reftri[i].p3.y, trimesh.reftri[i].p3.z]
        Trx[i] = [TrV1[i][0], TrV2[i][0], TrV3[i][0]]
        Try[i] = [TrV1[i][1], TrV2[i][1], TrV3[i][1]]
        Trz[i] = [TrV1[i][2], TrV2[i][2], TrV3[i][2]]
        Trverts = [zip(Trx[i], Try[i], Trz[i])]
        ax.add_collection3d(Poly3DCollection(Trverts, facecolors='orange', linewidths=2))



    if bool1:
        t3 = np.linspace(ARAY.mint, tHit, 100)
        x3 = ARAY.o.x + t3*ARAY.d.x
        y3 = ARAY.o.y + t3*ARAY.d.y
        z3 = ARAY.o.z + t3*ARAY.d.z
        ax.plot(x3, y3, z3, color='r', linewidth=2)
        tnn = np.linspace(0, 1, 20)
        P1 = tri.dg.p ; N1 = tri.dg.nn;
        N1 = FaceForward(N1, ARAY.d * -1)
        x5 = P1.x + tnn * N1.x
        y5 = P1.y + tnn * N1.y
        z5 = P1.z + tnn * N1.z
        ax.plot(x5, y5, z5, color='g', linewidth=4)

    if bool2:
        tm1 = np.linspace(ARAY2.mint, tHitmesh, 100)
        xm1 = ARAY2.o.x + t3*ARAY2.d.x
        ym1 = ARAY2.o.y + t3*ARAY2.d.y
        zm1 = ARAY2.o.z + t3*ARAY2.d.z
        ax.plot(xm1, ym1, zm1, color='r', linewidth=2)
        tnnmesh = np.linspace(0, 1, 20)
        Pmesh = trimesh.dg.p ; Nmesh = trimesh.dg.nn;
        Nmesh = FaceForward(Nmesh, ARAY.d * -1)
        xm2 = Pmesh.x + tnnmesh * Nmesh.x
        ym2 = Pmesh.y + tnnmesh * Nmesh.y
        zm2 = Pmesh.z + tnnmesh * Nmesh.z
        ax.plot(xm2, ym2, zm2, color='g', linewidth=4)

    # plot all the geometries
    ax.scatter([0],[0],[0],color="g",s=140)
    ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, color='b')
    ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, color='b')
    ax.plot(xs, ys, zs, color='y', linewidth=4)
    ax.set_xlim3d(-120, 120)
    ax.set_ylim3d(-120, 120)
    ax.set_zlim3d(-120, 120)
    # Show the geometries
    plt.show()
