#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math 
import numpy as np

#####################################################################################
class Vector(object):
    '''
    Definition of Vector

    Creation of the class vector

    args : x, y and z directions
    '''
    def __init__(self, x = 0, y = 0, z = 0):
        if isinstance(x, Vector) or isinstance(x, Point) or isinstance(x, Normal):
            self.x = x.x; self.y = x.y; self.z = x.z;
        elif isinstance(x, np.ndarray):
            self.x = x[0]; self.y = x[1]; self.z = x[2]
        else: 
            self.x = x; self.y = y; self.z = z;

    def __eq__(self, v2):
        if isinstance(v2, Vector):
            return (self.x==v2.x) and (self.y==v2.y) and (self.z==v2.z)
        else:
            raise NameError('Equality with a Vector must be with another Vector')

    def __add__(self, v2):
        if isinstance(v2, Vector):
            return Vector(self.x+v2.x, self.y+v2.y, self.z+v2.z) 
        else:
            raise NameError('Addition with a Vector must be with another Vector')

    def __sub__(self, v2):
        if isinstance(v2, Vector):
            return Vector(self.x-v2.x, self.y-v2.y, self.z-v2.z)
        else:
            raise NameError('Substraction with a Vector must be with another Vector')

    def __truediv__(self, sca):
        if (type(sca) != Vector) and (type(sca) != Point) and (type(sca) != Normal):
            return Vector(self.x/sca, self.y/sca, self.z/sca) 
        else:
            raise NameError('div accepted only with scalar')
    def __mul__(self, sca): 
        if (type(sca) != Vector) and (type(sca) != Point) and (type(sca) != Normal):
            return Vector(sca*self.x, sca*self.y, sca*self.z)
        else:
            raise NameError('mul accepted only with scalar')

    def __getitem__(self, ind):
        if ind == 0:
            return self.x;
        elif ind == 1:
            return self.y;
        elif ind == 2 :
            return self.z;
        else:
            raise NameError('Indice out of range!')

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'

    def Length(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z) # L2 norm

    def asarr(self):
        return np.array([self.x, self.y, self.z])
    
#####################################################################################

#####################################################################################
class Point(object):
    '''
    Definition of Point

    Creation of the class Point

    args : x, y and z coordinates
    '''
    def __init__(self, x = 0, y = 0, z = 0):
        if isinstance(x, Vector) or isinstance(x, Point) or isinstance(x, Normal):
            self.x = x.x; self.y = x.y; self.z = x.z;
        else:
            self.x = x; self.y = y; self.z = z;

    def __eq__(self, p2):
        if isinstance(p2, Point):
            return (self.x==p2.x) and (self.y==p2.y) and (self.z==p2.z)
        else:
            raise NameError('Equality with a Point must be with another Point')

    def __add__(self, v):
        if isinstance(v, Vector):
            return Point(self.x+v.x, self.y+v.y, self.z+v.z)
        else:
            raise NameError('Addition with a Point must be with a Vector')

    def __sub__(self, vp2):
        if isinstance(vp2, Vector):
            return Point(self.x-vp2.x, self.y-vp2.y, self.z-vp2.z)
        elif isinstance(vp2, Point):
            return Vector(self.x-vp2.x, self.y-vp2.y, self.z-vp2.z)
        else:
            raise NameError('Subs with a Point must be with another Point or a Vector')

    def __truediv__(self, sca):
        if (type(sca) != Vector) and (type(sca) != Point) and (type(sca) != Normal):
            return Point(self.x/sca, self.y/sca, self.z/sca)
        else:
            raise NameError('div accepted only with scalar')

    def __mul__(self, sca):
        if (type(sca) != Vector) and (type(sca) != Point) and (type(sca) != Normal):
            return Point(sca*self.x, sca*self.y, sca*self.z)
        else:
            raise NameError('mul accepted only with scalar')

    def __getitem__(self, ind):
        if ind == 0:
            return self.x;
        elif ind == 1:
            return self.y;
        elif ind == 2 :
            return self.z;
        else:
            raise NameError('indice out of range!')

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'
#####################################################################################

#####################################################################################
class Normal(object):
    '''
    Definition of Normal

    Creation of the class Normal

    args : x, y and z directions
    '''
    def __init__(self, x = 0, y = 0, z = 0):
        if isinstance(x, Vector) or isinstance(x, Point) or isinstance(x, Normal):
            self.x = x.x; self.y = x.y; self.z = x.z;
        else:
            self.x = x; self.y = y; self.z = z;

    def __eq__(self, n2):
        if isinstance(n2, Normal):
            return (self.x==n2.x) and (self.y==n2.y) and (self.z==n2.z)
        else:
            raise NameError('Equality with a Normal must be with another Normal')

    def __add__(self, n2):
        if isinstance(n2, Normal):
            return Normal(self.x+n2.x, self.y+n2.y, self.z+n2.z) 
        else:
            raise NameError('Addition with a Normal must be with another Normal')

    def __sub__(self, n2):
        if isinstance(n2, Normal):
            return Vector(self.x-n2.x, self.y-n2.y, self.z-n2.z)
        elif n2 == 0:
            return (self.x==-1*self.x) and (self.y==-1*self.y) and (self.z==-1*self.z)
        else:
            raise NameError('Substraction with a Normal must be with another Normal')

    def __truediv__(self, sca):
        if (type(sca) != Vector) and (type(sca) != Point) and (type(sca) != Normal):
            return Vector(self.x/sca, self.y/sca, self.z/sca) 
        else:
            raise NameError('div accepted only with scalar')

    def __mul__(self, sca): 
        if (type(sca) != Vector) and (type(sca) != Point) and (type(sca) != Normal):
            return Vector(sca*self.x, sca*self.y, sca*self.z)
        else:
            raise NameError('mul accepted only with scalar')

    def __getitem__(self, ind):
        if ind == 0:
            return self.x;
        elif ind == 1:
            return self.y;
        elif ind == 2 :
            return self.z;
        else:
            raise NameError('Indice out of range!')

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'

    def Length(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z) # L2 norm
#####################################################################################

#####################################################################################
class Ray(object):
    '''
    Definition of Ray

    Creation of the class Ray

    o     : Reference point of the ray (Point class)
    d     : Direction of the ray (Vector class)
    t     : Initial time for the initial position of the ray
            t = 0 means o = the initial position
    start : Min value of t
    end   : Max value of t

    '''
    def __init__(self, o = Point(), d = Vector(), t = 0, start = 0, end = float("inf")):
        if isinstance(o, Ray):
            self.o = o.o
            self.d = o.d
            self.t = o.t
            self.mint = o.mint
            self.maxt = o.maxt
        else:
            self.o = o
            self.d = d
            self.t = t
            self.mint = start
            self.maxt = end

    def __getitem__(self, t):
        if (type(t) != Vector) and (type(t) != Point) and (type(t) != Normal):
            return (Point(self.o) + Vector(self.d * t))
        else:
            raise NameError('t must be a scalar')

#####################################################################################

#####################################################################################
class BBox(object):
    '''
    Definition of Bounding Box

    Creation of the bounding box class thanks to 2 points
    
    p1 : First point (Point class)
    p2 : Second point (Point class)
    '''
    def __init__(self, p1 = None, p2 = None):
        if (isinstance(p1, Point) or (p1 is None)) and \
           (isinstance(p2, Point) or (p2 is None)):
            if ((p1 is None) and (p2 is None)):
                self.pMax = Point(float("inf"), float("inf"), float("inf"))
                self.pMin = Point(float("-inf"), float("-inf"), float("-inf"))
            elif (p2 is None):
                self.pMax = Point(p1)
                self.pMin = Point(p1)    
            elif (p1 is None):
                self.pMax = Point(p2)
                self.pMin = Point(p2) 
            else:
                self.pMax = Point(max(p1.x, p2.x), max(p1.y, p2.y), max(p1.z, p2.z))
                self.pMin = Point(min(p1.x, p2.x), min(p1.y, p2.y), min(p1.z, p2.z))
        else:
            raise NameError('Bounding Box constructor accepts only points')
        
    def union(self, arg1, arg2):
        if isinstance(arg1, BBox):
            ret = BBox()
            if isinstance(arg2, Point):
                ret.pMin.x = min(arg1.pMin.x, arg2.x);
                ret.pMin.y = min(arg1.pMin.y, arg2.y);
                ret.pMin.z = min(arg1.pMin.z, arg2.z);
                ret.pMax.x = max(arg1.pMax.x, arg2.x);
                ret.pMax.y = max(arg1.pMax.y, arg2.y);
                ret.pMax.z = max(arg1.pMax.z, arg2.z);
            elif isinstance(arg2, BBox):
                ret.pMin.x = min(b.pMin.x, arg2.pMin.x);
                ret.pMin.y = min(b.pMin.y, arg2.pMin.y);
                ret.pMin.z = min(b.pMin.z, arg2.pMin.z);
                ret.pMax.x = max(b.pMax.x, arg2.pMax.x);
                ret.pMax.y = max(b.pMax.y, arg2.pMax.y);
                ret.pMax.z = max(b.pMax.z, arg2.pMax.z);
            else:
                raise NameError('arg2 of BBox union have to be BBox or Point class')
            return ret
        else:
            raise NameError('arg1 of BBox union have to be BBox class')            
            
    def get_vertices(self):
        '''
        get the 8 vertices of a BBox as a list of points.
        p0==pMin, then next 3 points are in the plane
        pMin.z,  the order being anti-clockwise.
        Next 4 points are in the plane pMax.z, starting with point
        just above p0, so pMax==p6
        '''
        return [
            Point(self.pMin.x,self.pMin.y,self.pMin.z),
            Point(self.pMax.x,self.pMin.y,self.pMin.z),
            Point(self.pMax.x,self.pMax.y,self.pMin.z),
            Point(self.pMin.x,self.pMax.y,self.pMin.z),
            Point(self.pMin.x,self.pMin.y,self.pMax.z),
            Point(self.pMax.x,self.pMin.y,self.pMax.z),
            Point(self.pMax.x,self.pMax.y,self.pMax.z),
            Point(self.pMin.x,self.pMax.y,self.pMax.z)
        ]

    def include_point(self, P):
        '''
        test if Point P is included in BBox
        '''
        return (P.x >= self.pMin.x) and (P.x <= self.pMax.x) and \
               (P.y >= self.pMin.y) and (P.y <= self.pMax.y) and \
               (P.z >= self.pMin.z) and (P.z <= self.pMax.z)


    def IntersectP(self, Ray) :
        '''
        test if Ray Ray intersects the BBox and finds both intersection points
        '''
        t0 = 0.
        epsi = 1e-32 * 0.5
        t1 = 1e32;
        gamma3 = (3*epsi)/(1 - 3*epsi)
        for i in range(3):
            if Ray.d[i]!= 0 :
                invRayDir = 1. / Ray.d[i]
            else: invRayDir=1e32
            tNear = (self.pMin[i] - Ray.o[i]) * invRayDir
            tFar  = (self.pMax[i] - Ray.o[i]) * invRayDir
            if (tNear > tFar):
                tmp  = tNear
                tNear= tFar
                tFar = tmp
            tFar *= 1 + 2*gamma3
            t0 = tNear if tNear > t0 else t0
            t1 = tFar  if  tFar < t1 else t1
            if (t0 > t1) : return 0, 0, False
        return t0, t1, True


def CommonVertices(BBox1, BBox2):
    ''' 
    return a list of boolean checking if vertices are common between BBoxes 
    '''
    return np.array(list((map(lambda x: x in BBox2.get_vertices(), BBox1.get_vertices()))))

def CommonFace(BBox1, BBox2, Fill_value=None):
    ''' 
    return the face index of th BBox1 which is common to BBox2 with
    the convention of index from 0 to 5, for +X,-X,+Y,-Y,+Z,-Z faces
    # (en.wikipedia.org/wiki/Cube_mapping)
    return Fill if there is no common face
    '''
    ok = CommonVertices(BBox1, BBox2)
    if ok.sum()==4:
        n  = np.arange(8)[ok]
        if   np.array_equal(n, np.array([1,2,5,6])):
            return 0
        elif np.array_equal(n, np.array([0,3,4,7])):
            return 1
        elif np.array_equal(n, np.array([2,3,6,7])):
            return 2
        elif np.array_equal(n, np.array([0,1,4,5])):
            return 3
        elif np.array_equal(n, np.array([4,5,6,7])):
            return 4
        elif np.array_equal(n, np.array([0,1,2,3])):
            return 5
        else: return Fill_value

    else :
        return Fill_value

#####################################################################################
def Dot(a, b):
    if (isinstance(a, Vector) or isinstance(a, Normal)) and \
       (isinstance(b, Vector) or isinstance(b, Normal)):
        return (a.x*b.x + a.y*b.y + a.z*b.z)
    else:
        raise NameError('Dot argments have to be Vector or Normal classes')

def Cross(a, b):
    if (isinstance(a, Vector) and isinstance(b, Vector)) or \
    (isinstance(a, Vector) and isinstance(b, Normal)) or \
    (isinstance(a, Normal) and isinstance(b, Vector)):
        return Vector((a.y*b.z)-(a.z*b.y), (a.z*b.x)-(a.x*b.z), (a.x*b.y)-(a.y*b.x))
    elif isinstance(a, Normal) and isinstance(b, Normal):
        raise NameError('Only 1 Normal (Cross args) is tolerate not 2')
    else:
        raise NameError('Cross args must be Vector or Normal')

def Normalize(v):
    if isinstance(v, Vector) or isinstance(v, Normal):
        return v / v.Length()
    else:
        raise NameError('Normalize argument have to be Vector or Normal class')

def CoordinateSystem(v1):
    '''
    Create an orthogonal coordinate system from 1 vector (v1)
    '''
    if (abs(v1.x) > abs(v1.y)):
        invLen = 1/ math.sqrt(v1.x*v1.x + v1.z*v1.z)
        v2 = Vector(-v1.z*invLen, 0, v1.x*invLen)
    else:
        invLen = 1/ math.sqrt(v1.y*v1.y + v1.z*v1.z)
        v2 = Vector(0, v1.z*invLen, -v1.y*invLen)
    v3 = Cross(v1, v2)
    return v2, v3

def Distance(p1, p2):
    if isinstance(p1, Point) and isinstance(p2, Point):
        return (p1 - p2).Length()
    else:
        raise NameError('Distance arguments have to be Point classes)')

def FaceForward(a, b):
    '''
    Flip the Vector/Normal a if the Vector/Normal b is in the opposite direction.
    For exemple, it can be useful to flip a surface normal so that it lies in the
    same hemisphere as a given vector.
    Args : Vector or Normal a, b
    Output : Possibly fliped Vector or Normal a
    '''
    if (isinstance(a, Vector) or isinstance(a, Normal)) and \
    (isinstance(b, Vector) or isinstance(b, Normal)):
        return (a*-1) if (Dot(a, b) < 0) else a
    else:
        raise NameError('FaceForward args have to be Vector or Normal classes')

def rotation3D(theta, u):
    ''' 
    rotation matrix of an angle theta in degree around unit vector u
    '''
    # Rodrigues rotation formula
    ct = np.cos(np.radians(theta))
    st = np.sqrt(1-ct*ct)
    A = np.zeros((3,3))
    for k in range(3) : A[k,k] = 1.
    B = np.zeros((3,3))
    B[0,1] = -u.z
    B[0,2] =  u.y
    B[1,0] =  u.z
    B[1,2] = -u.x
    B[2,0] = -u.y
    B[2,1] =  u.x
    C = B.dot(B)
    return A + B*st + C*(1-ct)


def MaxDimension(v):
    if isinstance(v, Vector):
        val = 0
        if (v.x > v.y):
            if(v.x > v.z):
                val = 0
            else:
                val = 2
        else:
            if(v.y > v.z):
                val = 1
            else:
                val = 2
        return int(val)
    else:
        raise NameError('v argument must be a Vector')
        
def Permute(v, xx, yy, zz):
    if isinstance(v, Vector):
        return Vector(v[xx], v[yy], v[zz])
    else:
        raise NameError('v argument must be a Vector')

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
    print("Length(a) =", a.Length())
    print("normalize(a) =", Normalize(a))
    print("Length of Normalize(a) =", Normalize(a).Length())

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
