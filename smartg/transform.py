#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import geometry
from .geometry import Vector, Point, Normal, Ray, BBox
from .geometry import Dot, Cross, Normalize, CoordinateSystem, \
    Distance, FaceForward
import math 
import numpy as np
from numpy.linalg import inv

#####################################################################################
class Transform(object):
    '''
    Creation of the class Transform
    '''

    def __init__(self, m = np.identity(4), mInv = None):
        self.m = m
        if (mInv is None):
            self.mInv = inv(self.m)
        else:
            self.mInv = mInv

    def __eq__(self, t):
        if isinstance(t, Transform):
            self.m = t.m
            self.mInv = t.mInv

    def inverse(self, t):
        return Transform(t.mInv, t.m)

    def isIdentity(self):
        return (self.m[0,0] == 1) and (self.m[0,1] == 0) and (self.m[0,2] == 0) and \
            (self.m[0,3] == 0) and (self.m[1,0] == 0) and (self.m[1,1] == 1) and \
            (self.m[1,2] == 0) and (self.m[1,3] == 0) and (self.m[2,0] == 0) and \
            (self.m[2,1] == 0) and (self.m[2,2] == 1) and (self.m[2,3] == 0) and \
            (self.m[3,0] == 0) and (self.m[3,1] == 0) and (self.m[3,2] == 0) and \
            (self.m[3,3] == 1)

    def translate(self, v):
        myM = np.identity(4)
        myM[0,3] = v.x; myM[1,3] = v.y; myM[2,3] = v.z;
        myMInv = np.identity(4)
        myMInv[0,3] = (v.x)*-1; myMInv[1,3] = (v.y)*-1; myMInv[2,3] = (v.z)*-1;
        return Transform(myM, myMInv)

    def scale(self, x, y, z):
        myM = np.identity(4)
        myM[0,0] = x; myM[1,1] = y; myM[2,2] = z;
        myMInv = np.identity(4)
        myMInv[0,0] = 1./x; myMInv[1,1] = 1./y; myMInv[2,2] = 1./z;
        return Transform(myM, myMInv)

    def rotateX(self, angle):
        sin_t = np.sin(angle*(np.pi / 180.))
        cos_t = np.cos(angle*(np.pi / 180.))
        myM = np.identity(4)
        myM[1,1] = cos_t; myM[1,2] = -1.*sin_t;
        myM[2,1] = sin_t; myM[2,2] = cos_t;
        return Transform(myM, np.transpose(myM))

    def rotateY(self, angle):
        sin_t = np.sin(angle*(np.pi / 180.))
        cos_t = np.cos(angle*(np.pi / 180.))
        myM = np.identity(4)
        myM[0,0] = cos_t; myM[2,0] = -1.*sin_t;
        myM[0,2] = sin_t; myM[2,2] = cos_t;
        return Transform(myM, np.transpose(myM))

    def rotateZ(self, angle):
        sin_t = np.sin(angle*(np.pi / 180.))
        cos_t = np.cos(angle*(np.pi / 180.))
        myM = np.identity(4)
        myM[0,0] = cos_t; myM[0,1] = -1.*sin_t;
        myM[1,0] = sin_t; myM[1,1] = cos_t;
        return Transform(myM, np.transpose(myM))

    def rotate(self, angle, axis):
        a = Vector(Normalize(axis))
        s = np.sin(angle*(np.pi / 180.))
        c = np.cos(angle*(np.pi / 180.))
        myM = np.identity(4)

        myM[0,0] = a.x*a.x+(1-a.x*a.x)*c;
        myM[0,1] = a.x*a.y*(1-c)-a.z*s;
        myM[0,2] = a.x*a.z*(1-c)+a.y*s;

        myM[1,0] = a.x*a.y*(1-c)+a.z*s;
        myM[1,1] = a.y*a.y+(1-a.y*a.y)*c;
        myM[1,2] = a.y*a.z*(1-c)-a.x*s;

        myM[2,0] = a.x*a.z*(1-c)-a.y*s;
        myM[2,1] = a.y*a.z*(1-c)+a.x*s;
        myM[2,2] = a.z*a.z+(1-a.z*a.z)*c;
        
        return Transform(myM, np.transpose(myM))

    def __mul__(self, T): 
        if (type(T) == Transform):
            return Transform(np.dot(self.m, T.m), np.dot(T.mInv, self.mInv))
        else:
            raise NameError('mul accepted only with Transform')

    def __getitem__(self, c):
        if isinstance(c, Point):
            xp = self.m[0,0]*c.x + self.m[0,1]*c.y + self.m[0,2]*c.z + self.m[0,3]
            yp = self.m[1,0]*c.x + self.m[1,1]*c.y + self.m[1,2]*c.z + self.m[1,3]
            zp = self.m[2,0]*c.x + self.m[2,1]*c.y + self.m[2,2]*c.z + self.m[2,3]
            wp = self.m[3,0]*c.x + self.m[3,1]*c.y + self.m[3,2]*c.z + self.m[3,3]
            if (wp is 1):
                return Point(xp, yp, zp)
            else: 
                return Point(xp, yp, zp)/wp
        elif isinstance(c, Vector):
            xv = self.m[0,0]*c.x + self.m[0,1]*c.y + self.m[0,2]*c.z
            yv = self.m[1,0]*c.x + self.m[1,1]*c.y + self.m[1,2]*c.z
            zv = self.m[2,0]*c.x + self.m[2,1]*c.y + self.m[2,2]*c.z
            return Vector(xv, yv, zv)
        elif isinstance(c, Normal):
            xn = self.mInv[0,0]*c.x + self.mInv[1,0]*c.y + self.mInv[2,0]*c.z
            yn = self.mInv[0,1]*c.x + self.mInv[1,1]*c.y + self.mInv[2,1]*c.z
            zn = self.mInv[0,2]*c.x + self.mInv[1,2]*c.y + self.mInv[2,2]*c.z
            return Normal(xn, yn, zn)
        elif isinstance(c, Ray):
            R = Ray(c.o, c.d)
            R.o = self[R.o]
            R.d = self[R.d]
            return R
        elif isinstance(c, BBox):
            P1 = self[Point(c.pMin.x, c.pMin.y, c.pMin.z)]
            V = self[Vector(c.pMax.x-c.pMin.x, c.pMax.y-c.pMin.y, c.pMax.z-c.pMin.z)]
            P2 = P1 + V
            # print "V =", V, "P1=", P1, "P2=", P2
            ret = BBox(P1, P2)
            # print "minB =", ret.pMin, "maxB =", ret.pMax
            # ret = BBox(self[Point(c.pMin.x, c.pMin.y, c.pMin.z)]) 
            # ret = ret.union(ret, self[Point(c.pMax.x, c.pMin.y, c.pMin.z)])
            # ret = ret.union(ret, self[Point(c.pMin.x, c.pMax.y, c.pMin.z)])
            # ret = ret.union(ret, self[Point(c.pMin.x, c.pMin.y, c.pMax.z)])
            # ret = ret.union(ret, self[Point(c.pMin.x, c.pMax.y, c.pMax.z)])
            # ret = ret.union(ret, self[Point(c.pMax.x, c.pMax.y, c.pMin.z)])
            # ret = ret.union(ret, self[Point(c.pMax.x, c.pMin.y, c.pMax.z)])
            # ret = ret.union(ret, self[Point(c.pMax.x, c.pMax.y, c.pMax.z)])
            return ret
        else:
            raise NameError('Unkonwn type for transformations')
#####################################################################################

#####################################################################################
def Aff(Mat, myString = None):
    if (myString == None):
        myString = 'My Matrix'
    myString2 = ' '
    for i in range(0, len(myString)-1):
        myString2 = myString2 + ' '

    print(myString + ' = (' + str(Mat[0,0]) + ', ' + str(Mat[0,1]) + ', ' \
        + str(Mat[0,2]) + ', ' + str(Mat[0,3])+ ')')
    print(myString2 + '   (' + str(Mat[1,0]) + ', ' + str(Mat[1,1]) + ', ' \
        + str(Mat[1,2]) + ', ' + str(Mat[1,3])+ ')')
    print(myString2 + '   (' + str(Mat[2,0]) + ', ' + str(Mat[2,1]) + ', ' \
        + str(Mat[2,2]) + ', ' + str(Mat[2,3])+ ')')
    print(myString2 + '   (' + str(Mat[3,0]) + ', ' + str(Mat[3,1]) + ', ' \
        + str(Mat[3,2]) + ', ' + str(Mat[3,3])+ ')')
#####################################################################################

##################################################################################### 
if __name__ == '__main__':

    A = np.array([[1, 0, 0, 2], [0, 1, 0, 2], [0, 0, 1, 2], [0, 0, 0, 1]])
    P1 = Point(10, 0, 2)
    V1 = Vector(1, 2, 3)
    R1 = Ray(P1, V1)

    print("P1 =", P1, "V1 =", V1, "R1[1] =", R1[1])

    MyT=Transform()
    Aff(MyT.m, 'Mat')
    Aff(MyT.mInv, 'InvMat')
    Trans = MyT.translate(Vector(10, 0, 0))
    Tinv = Trans.inverse(Trans)
    P2 = Tinv[P1]
    V2 = Tinv[V1]
    R2 = Tinv[R1]
    
    print("translated point =", P2, ", vector =", V2, ", ray =", R2[1])

    muSca = MyT.scale(2,2,2)
    muScaInv = muSca.inverse(muSca)
    TS = Trans*muSca #Transform(np.dot(Trans.m, muSca.m), np.dot(muSca.mInv, Trans.mInv))
    TSInv = TS.inverse(TS)

    P3 = muScaInv[P2]
    V3 = muScaInv[V2]
    R3 = muScaInv[R2]

    print("trans + scale : point =", P3, ", vector =", V3, ", ray =", R3[1])

    P4 = TSInv[P1]
    V4 = TSInv[V1]
    R4 = TSInv[R1]

    print("trans + scale 2: point =", P4, ", vector =", V4, ", ray =", R4[1])
    # Aff(MyT.mInv, 'MatInv')
    # MyT = MyT.Inverse(MyT)
    # Aff(MyT.m, 'Mat')
    # Aff(MyT.mInv, 'MatInv')
