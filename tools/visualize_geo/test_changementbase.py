#!/usr/bin/env python
# -*- coding: utf-8 -*-

import geometry
from geometry import Vector, Point, Normal, Ray, BBox
from geometry import Dot, Cross, Normalize, CoordinateSystem, \
    Distance, FaceForward
import diffgeom
from diffgeom import DifferentialGeometry
import transform
from transform import Transform, Aff
import shape
from shape import Shape, Sphere, TriangleM, TriangleMesh, swap, \
    Clamp, Quadratic
import math 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


if __name__ == '__main__':

    N = Vector(1, 1, 1) # normal
    N = Normalize(N)
    V1, V2 = CoordinateSystem(N)

    e1 = Vector(1, 0, 0)
    e2 = Vector(0, 1, 0)
    e3 = Vector(0, 0, 1)

    print "V1 =", V1, "V2 =", V2, "N =", N
    print "norm V1 =", V1.Lengh(), "normV2 =", V2.Lengh(), "normN =", N.Lengh()

    M = np.array([[V1.x, V2.x, N.x, 0],  # row 1
                  [V1.y, V2.y, N.y, 0],  # row 2
                  [V1.z, V2.z, N.z, 0],  # row 2
                  [   0,    0,   0, 1]])
    tM = np.transpose(M)
             
    Aff(M, 'M')
    Aff(tM, 'tM')
    
    wTo = Transform(tM, M)
    oTw = wTo.inverse(wTo)
    P1 = Point(N.x, N.y, N.z)
    Vec1 = Vector(1, 0, 0)

    print "repère 1 P1 =", P1
    print "repère 1 Vec1 =", Vec1
    P1 = wTo[P1]
    Vec1 = wTo[Vec1]
    print "repère 2 P1 =", P1
    print "repère 2 Vec1 =", Vec1
    P1 = oTw[P1]
    Vec1 = oTw[Vec1]
    print "retour repère 1 P1 =", P1
    print "retour repère 1 Vec1 =", Vec1

    # print "M[0, 0] =", M[0, 0], "M[0, 1] =", M[0, 1], "M[0, 2] =", M[0, 2], "M[0, 3] =", M[0, 3]
    # print "M[1, 0] =", M[1, 0], "M[1, 1] =", M[1, 1], "M[1, 2] =", M[1, 2], "M[1, 3] =", M[1, 3]
    # print "M[2, 0] =", M[2, 0], "M[2, 1] =", M[2, 1], "M[2, 2] =", M[2, 2], "M[2, 3] =", M[2, 3]
    # print "M[3, 0] =", M[3, 0], "M[3, 1] =", M[3, 1], "M[3, 2] =", M[3, 2], "M[3, 3] =", M[3, 3]
    # print "========================="
    # print "tM[0, 0] =", tM[0, 0], "tM[0, 1] =", tM[0, 1], "tM[0, 2] =", tM[0, 2], "tM[0, 3] =", tM[0, 3]
    # print "tM[1, 0] =", tM[1, 0], "tM[1, 1] =", tM[1, 1], "tM[1, 2] =", tM[1, 2], "tM[1, 3] =", tM[1, 3]
    # print "tM[2, 0] =", tM[2, 0], "tM[2, 1] =", tM[2, 1], "tM[2, 2] =", tM[2, 2], "tM[2, 3] =", tM[2, 3]
    # print "tM[3, 0] =", tM[3, 0], "tM[3, 1] =", tM[3, 1], "tM[3, 2] =", tM[3, 2], "tM[3, 3] =", tM[3, 3]

    # uvs = np.array([[0, 0], [1, 0], [1, 1]])
    # print "uvs[0, 0] =", uvs[0, 0], "uvs[0, 1] =", uvs[0, 1]
    # print "uvs[1, 0] =", uvs[1, 0], "uvs[1, 1] =", uvs[1, 1]
    # print "uvs[2, 0] =", uvs[2, 0], "uvs[2, 1] =", uvs[2, 1]
    
    # aVec = np.array(30)
    # aVec2 = np.zeros(5, dtype=np.float32)
    # aVec2[0] = 1
    # aVec2[2] = 368
    # aVec2[3] = 68
    # print "lenght =", len(np.atleast_1d(aVec)), type(aVec)
    # print "lenght =", len(np.atleast_1d(aVec2)), type(aVec2), "max=", np.amax(aVec2), "min=",np.amin(aVec2)
 
    # ARAY = Ray(Point(0, 0, 10), Vector(-0.3, 0.5, -1), 0, 0, 30)
    # ARAY2 = Ray(Point(0, 0, 10), Vector(-1.8, 1.9, -1), 0, 0, 30)
    # vi = np.array([0, 1, 2, 2, 3, 1], dtype=np.int32)
    # P = np.array([0, 0, 0, -30, 0, 0, 0, 30, 0, -30, 30, 0], dtype=np.float32)
    # trimesh = TriangleMesh(Transform(), Transform(), vi, P)
    # bool2 = trimesh.Intersect(ARAY2)
    # if bool2:
    #     print "There is an intersection with the triangle Mesh***"
    #     print "At p = (", trimesh.dg.p.x, ", ", trimesh.dg.p.y, ", ", trimesh.dg.p.z, ")"
    #     tHitmesh = trimesh.thit
    # else:
    #     print "No intersection with the triangle Mesh***"


    # A1 = Point(0, 0, 0) ; A2 = Point(-30, 0, 0); A3 = Point(0, 30, 0);
    # tri =  Triangle(Transform(), Transform(), A1, A2, A3)
    # bool1 = tri.Intersect(ARAY)
    # if bool1:
    #     print "There is an intersection with the triangle***"
    #     print "At p = (", tri.dg.p.x, ", ", tri.dg.p.y, ", ", tri.dg.p.z, ")"
    #     tHit = tri.thit
    # else:
    #     print "No intersection with the triangle***"

    # # transform needed for the first sphere
    # TSph1 = Transform()

    # TRX = TSph1.rotateX(90)
    # TSph1 = TRX
    # invTSph1 = TSph1.inverse(TSph1)

    # # create the first sphere
    # Sph1 = Sphere(TSph1, invTSph1, 60, -60, 60, 29.9)
    # # ================================================
    # # transform needed for the second sphere
    # TSph2 = Transform()

    # TTrans = TSph2.translate(Vector(-110, 0, 0))
    # TSph2 = TTrans
    # invTSph2 = TSph2.inverse(TSph2)

    # # create the second sphere
    # Sph2 = Sphere(TSph2, invTSph2, 50, -50, 50, 360)


    # # create the matplotlib figure
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # parameters of the first sphere
    # u1 = np.linspace(0, Sph1.phiMax, 20)
    # v1 = np.linspace(Sph1.thetaMin, Sph1.thetaMax, 20)
    # myPoint1 = TSph1[Point(Sph1.radius * np.outer(np.cos(u1), np.sin(v1)), Sph1.radius * np.outer(np.sin(u1), np.sin(v1)), Sph1.radius * np.outer(np.ones(np.size(u1)), np.cos(v1)))]
    # x1 = myPoint1.x
    # y1 = myPoint1.y
    # z1 = myPoint1.z

    # # parameters of the second sphere
    # u2 = np.linspace(0, Sph2.phiMax, 20)
    # v2 = np.linspace(Sph2.thetaMin, Sph2.thetaMax, 20)
    # myPoint2 = TSph2[Point(Sph2.radius * np.outer(np.cos(u2), np.sin(v2)), Sph2.radius * np.outer(np.sin(u2), np.sin(v2)), Sph2.radius * np.outer(np.ones(np.size(u2)), np.cos(v2)))]
    # x2 = myPoint2.x
    # y2 = myPoint2.y
    # z2 = myPoint2.z

    # # Photon paramaters plot thanks to straight equation (SunZ = A*SunX + b)
    # # where SunY is constant and equal to 0
    # tan_S = np.tan(60*(np.pi / 180.))      # zenith angle at 60°
    # SunX = np.linspace(0, 120*tan_S, 100)
    # xs = SunX
    # ys = 0 * SunX                          # multiply by SunX to get an array
    # zs = (1./tan_S) * SunX

    # # Triangles parameters
    # # v1 = [0, 0, 0]
    # # v2 = [-30, 0, 0]
    # # v3 = [0, 30, 0]

    # # x = [v1[0], v2[0], v3[0]]
    # # y = [v1[1], v2[1], v3[1]]
    # # z = [v1[2], v2[2], v3[2]]
    # # verts = [zip(x, y, z)]
    # # ax.add_collection3d(Poly3DCollection(verts, facecolors='orange', linewidths=2))

    # # Triangles mesh parameters
    # TrV1 = [None]*trimesh.ntris
    # TrV2 = [None]*trimesh.ntris
    # TrV3 = [None]*trimesh.ntris
    # Trx = [None]*trimesh.ntris
    # Try = [None]*trimesh.ntris
    # Trz = [None]*trimesh.ntris
    # for i in xrange(0, trimesh.ntris):
    #     print "p1 = (", trimesh.reftri[i].p1.x, ", ", trimesh.reftri[i].p1.y, ", ", trimesh.reftri[i].p1.z, ")"
    #     print "p2 = (", trimesh.reftri[i].p2.x, ", ", trimesh.reftri[i].p2.y, ", ", trimesh.reftri[i].p2.z, ")"
    #     print "p3 = (", trimesh.reftri[i].p3.x, ", ", trimesh.reftri[i].p3.y, ", ", trimesh.reftri[i].p3.z, ")"
    #     TrV1[i] = [trimesh.reftri[i].p1.x, trimesh.reftri[i].p1.y, trimesh.reftri[i].p1.z]
    #     TrV2[i] = [trimesh.reftri[i].p2.x, trimesh.reftri[i].p2.y, trimesh.reftri[i].p2.z]
    #     TrV3[i] = [trimesh.reftri[i].p3.x, trimesh.reftri[i].p3.y, trimesh.reftri[i].p3.z]
    #     Trx[i] = [TrV1[i][0], TrV2[i][0], TrV3[i][0]]
    #     Try[i] = [TrV1[i][1], TrV2[i][1], TrV3[i][1]]
    #     Trz[i] = [TrV1[i][2], TrV2[i][2], TrV3[i][2]]
    #     Trverts = [zip(Trx[i], Try[i], Trz[i])]
    #     ax.add_collection3d(Poly3DCollection(Trverts, facecolors='orange', linewidths=2))



    # if bool1:
    #     t3 = np.linspace(ARAY.mint, tHit, 100)
    #     x3 = ARAY.o.x + t3*ARAY.d.x
    #     y3 = ARAY.o.y + t3*ARAY.d.y
    #     z3 = ARAY.o.z + t3*ARAY.d.z
    #     ax.plot(x3, y3, z3, color='r', linewidth=2)
    #     tnn = np.linspace(0, 1, 20)
    #     P1 = tri.dg.p ; N1 = tri.dg.nn;
    #     N1 = FaceForward(N1, ARAY.d * -1)
    #     x5 = P1.x + tnn * N1.x
    #     y5 = P1.y + tnn * N1.y
    #     z5 = P1.z + tnn * N1.z
    #     ax.plot(x5, y5, z5, color='g', linewidth=4)

    # if bool2:
    #     tm1 = np.linspace(ARAY2.mint, tHitmesh, 100)
    #     xm1 = ARAY2.o.x + t3*ARAY2.d.x
    #     ym1 = ARAY2.o.y + t3*ARAY2.d.y
    #     zm1 = ARAY2.o.z + t3*ARAY2.d.z
    #     ax.plot(xm1, ym1, zm1, color='r', linewidth=2)
    #     tnnmesh = np.linspace(0, 1, 20)
    #     Pmesh = trimesh.dg.p ; Nmesh = trimesh.dg.nn;
    #     Nmesh = FaceForward(Nmesh, ARAY.d * -1)
    #     xm2 = Pmesh.x + tnnmesh * Nmesh.x
    #     ym2 = Pmesh.y + tnnmesh * Nmesh.y
    #     zm2 = Pmesh.z + tnnmesh * Nmesh.z
    #     ax.plot(xm2, ym2, zm2, color='g', linewidth=4)

    # # plot all the geometries
    # ax.scatter([0],[0],[0],color="g",s=140)
    # ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, color='b')
    # ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, color='b')
    # ax.plot(xs, ys, zs, color='y', linewidth=4)
    # ax.set_xlim3d(-120, 120)
    # ax.set_ylim3d(-120, 120)
    # ax.set_zlim3d(-120, 120)
    # # Show the geometries
    # plt.show()
