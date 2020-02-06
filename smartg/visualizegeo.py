#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import geometry
from .geometry import Vector, Point, Normal, Ray, BBox, rotation3D
from .geometry import Dot, Cross, Normalize, CoordinateSystem, \
    Distance, FaceForward
from . import diffgeom
from .diffgeom import DifferentialGeometry
from . import transform
from .transform import Transform, Aff
from . import shape
from .shape import Shape, Sphere, TriangleM, TriangleMesh, swap, \
    Clamp, Quadratic, Triangle
import math 
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as mp3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors as mcolors

import re, six
from itertools import dropwhile

def receiver_view(disMatrix, w = False, logI=False, nameFile = None, MTOA = 1320,
                  vmin=None, vmax=None, interpol='none'):

    '''
    Definition of receiver_view

    Display the distribution of the radiant flux at a given receiver in the relative
    coordinates i.e. the coordinates specific to the object (which move with the
    object if there is a use of transformation)

        ^ x
        |     Print with the following cordinate system
    y <--    

    disMatrix : 2D numpy array with flux distribution at the receiver
    w         : The receiver size, in kilometer. Can be a scalar or a list with x
                and y values
    logI      : Enable log interval
    nameFile  : By default None. If not None create a pdf file in auxdata directory
                of the current print with the specified name
    MTOA      : Radiant exitance at TOA (W/m2)
    vmin      : Minimal distribution value (W/m2), not for log print
    vmax      : Maximal distribution value (W/m2), not for log print
    interpol  : Interpolations for imshow/matshow, i.e. nearest, bilinear, bicubic, ...
    '''
    if w==False :
        raise Exception("In receiver_view(), the receiver size w must be specified!")
        
    if isinstance(w, (list, tuple)):
        wx= w[0]
        wy= w[1]
    else:
        wx = w
        wy = w

    m = disMatrix

    plt.figure()

    if logI == False :
        cax = plt.imshow(m*MTOA, cmap=plt.get_cmap('jet'), interpolation=interpol, \
                         vmin=vmin, vmax=vmax, extent = [(wy*1000),-(wy*1000),-(wx*1000),(wx*1000)])
    else:
        m2 = m
        if (np.amin(m2) < 0.00001):
            valmin = 0.00001
        else:
            valmin = np.amin(m2)
            
        cax = plt.imshow(m*MTOA, cmap=plt.get_cmap('jet'), \
                         norm=mcolors.LogNorm(vmin=valmin*MTOA, vmax=np.amax(m*MTOA)), \
                         interpolation=interpol, extent = [(wy*1000),-(wy*1000),-(wx*1000),(wx*1000)])

    cbar = plt.colorbar()
    cbar.remove()
    cbar = plt.colorbar(cax)
    cbar.set_label(r'Irradiance (W m$^{-2}$)', fontsize = 12)
    plt.xlabel(r'Position (m) in relative y axis')
    plt.ylabel(r'Position (m) in relative x axis')
    plt.title('Receiver surface')
    if (nameFile is not None):
        plt.savefig(nameFile + '.pdf')  


def cat_view(mlut, view = 'all', acc = 6, MTOA = 1320):
    '''
    Definition of cat_view

    mlut : mlut table
    view : Several print choices : all, weight, number, errP and errAbs
    acc  : Accuracy, number of decimal points to show (integer)
    '''
    m = mlut
    lP = ["(  D  )", "(  H  )", "(  E  )", "(  A  )", "( H+A )", "( H+E )", "( E+A )", "(H+E+A)"]
    intAcc = int(acc)
    strAcc = str(intAcc)
    # strAcc = "{0:." + strAcc + "}"
    strAcc = "%." + strAcc + "f"

    print("**************************************************")
    print("Intensity in power unit (Watt) for each categories")
    print("Irradiance(W/m²)=Intensity(W)/receiverSurface(m²)")
    print("**************************************************")
    for i in range (0, 8):
        if (view == 'all'):
            print("CAT",i+1, lP[i], ": weight=", strAcc % (mlut['catWeightPh'][i]*MTOA), " number=", np.uint64(mlut['catNbPh'][i]),
                  " err(%)=", strAcc % mlut['catErrP'][i], " errAbs=", strAcc % (mlut['catErrAbs'][i]*MTOA))
        elif (view == 'weight'):
            print("CAT",i+1,": weight=", strAcc % (mlut['catWeightPh'][i]*MTOA))
        elif (view == 'number'):
            print("CAT",i+1,": number=", np.uint64(mlut['catNbPh'][i]))
        elif (view == 'errP'):
            print("CAT",i+1,": err(%)=", strAcc % mlut['catErrP'][i])
        elif (view == 'errAbs'):
            print("CAT",i+1,": errAbs=", strAcc % (mlut['catErrAbs'][i]*MTOA))

class Mirror(object):
    '''
    Definition of Mirror

    Glossy/specular material as pure and highly polished aluminum, silver
    behind glass mirror, ...

    reflectivity : The albedo of the object
    roughness    : Equal to alpha parameter according to Walter et al. 2007
    shadow       : Shadowing-Masking effect, by default not considered
    nind         : Relative refractive index air/material, by default
                   is None -> case of perfect mirror (nind = infinity)
    distribution : Two choices --> "Beckmann" or "GGX"
    '''
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
    '''
    Definition of LambMirror

    Lambertian material, same probability of reflection in all the direction
    inside the hemisphere of the normal of the object surface

    reflectivity : The albedo of the object
    '''
    def __init__(self, reflectivity = 0.5):
        self.reflectivity = reflectivity
        

    def __str__(self):
        return 'Material -> Lambertian Mirror : ' \
            'reflectivity=' + str(self.reflectivity)

class Matte(object):
    '''
    Definition of Matte

    Diffuse material as Concrete, plastic, dust, ...

    reflectivity : The albedo of the object
    roughness    : Not yet available
    '''
    def __init__(self, reflectivity = 0., roughness = 0.):
        self.reflectivity = reflectivity
        self.roughness = roughness
        
    def __str__(self):
        return 'Material -> Matte : ' \
            'reflectivity=' + str(self.reflectivity) + ', roughness=' + str(self.roughness)

class Plane(object):
    '''
    Definition of Plane

    Plane constructed with 4 points : p1, p2, p3, p4

    p1 : x --> negative and y --> negative
    p2 : x --> positive and y --> negative
    p3 : x --> negative and y --> positive
    p4 : x --> positive and y --> positive
    '''
    def __init__(self, p1 = Point(-0.5, -0.5, 0.), p2 = Point(0.5, -0.5, 0.), \
                 p3 = Point(-0.5, 0.5, 0.), p4 = Point(0.5, 0.5, 0.)):
        if (isinstance(p1, Point) and isinstance(p2, Point) and \
            isinstance(p3, Point) and isinstance(p4, Point)):
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
    '''
    Definition of Spheric

    Sphere constructed with --->

    radius   : The radius of th e sphere
    radiusZ0 : Take into account all the sphere -> radiusZ0 = -radius
    radiusZ1 : Take into account all the sphere -> radiusZ1 = +radius
    phi      : The value of phi, 360 degrees is the value of a full sphere
    '''
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
    '''
    Definition of Transformation

    Enable to move, rotate a given object

    rotation      : 1D np array, 3 values for rotation in x, y and z (degree)
    translation   : 1D np array, 3 values for translation in x, y and z (kilometer)
    rotationOrder : Order of rotation, 6 choices : XYZ, XZY, YXZ, YZX, ZXY, ZYX
    '''
    def __init__(self, rotation = np.zeros(3, dtype=float), translation=np.zeros(3, dtype=float), \
                 rotationOrder = "XYZ"):
        self.rotation = rotation
        self.rotx = rotation[0]
        self.roty = rotation[1]
        self.rotz = rotation[2]
        self.rotOrder = rotationOrder
        self.translation = translation
        self.transx = translation[0]
        self.transy = translation[1]
        self.transz = translation[2]

    def __str__(self):
        return 'Transformation : rotation=(' + str(self.rotx) + ', ' + str(self.roty) + ', ' + \
            str(self.rotz) + ') and translation =(' + str(self.transx) + ', ' + \
            str(self.transy) + ', ' + str(self.transz) + ')'
    
class Entity(object):
    '''
    Definition of Entity

    This class enables the creation a 3D object

    entity : By default None. But useful in case where we need a copy of a given
             object
    name   : 2 choices --> reflector or receiver.
             If receiver is chosen, smartg will count the distribution flux
    TC     : Taille Cellules --> size of cells for the flux distribution (kilometer)
    bboxGPmin/max : in development...
    '''
    def __init__(self, entity = None, name="reflector", TC = 0.01, materialAV=Matte(), \
                 materialAR=Matte(), geo=Plane(), transformation=Transformation(), \
                 bboxGPmin = Point(-100000., -100000., 0.), bboxGPmax = Point(100000., 100000., 120.)):
        if isinstance(entity, Entity) :
            self.name = entity.name; self.TC = entity.TC; self.materialAV = entity.materialAV;
            self.materialAR = entity.materialAR; self.geo = entity.geo ;
            self.transformation = entity.transformation;
            self.bboxGPmin = entity.bboxGPmin; self.bboxGPmax = entity.bboxGPmax;
        else:
            self.name = name
            self.TC = TC
            self.materialAV = materialAV
            self.materialAR = materialAR
            self.geo = geo
            self.transformation = transformation
            self.bboxGPmin = bboxGPmin
            self.bboxGPmax = bboxGPmax

    def __str__(self):
        return 'The entity is a ' + str(self.name) + ' with the following carac:\n' + \
            str(self.material) + '\n' + \
            str(self.geo) + '\n' + \
            str(self.transformation)

class Heliostat(object):
    '''
    Definition of Heliostat
    
    This class enables the creation of heliostats i.e. a group of facets:

    POS           : Heliostat position stored in a Point class
    SPX, SPY      : The heliostat is splited in facets : SPx -> the number of time
                    we split in the x direction, SPy -> the same in y direction
    HSX           : Heliostat size in x direction
    HSY           : Heliostat size in y direction
    CURVE_FL      : Focal length : A curved heliostat is possible if curveFL is given
    REF           : Reflectivity of the heliostat
    '''
    def __init__(self, POS = Point(0., 0., 0.), SPX=int(2), SPY=int(2), HSX=0.02,
                 HSY=0.02, CURVE_FL=None, REF=1.):
        # Be sure that we split a heliostat by at least 2
        if (SPX*SPY < 2):
            raise Exception("The number of facets must be >= 2!")
        # Be sure that SPX and SPY are integer values
        if not ( isinstance(SPX, int) and isinstance(SPY, int) ):
            raise Exception("SPx and SPy must be integers")
        self.pos = POS
        self.sPx = SPX
        self.sPy = SPY
        self.hSx = HSX
        self.hSy = HSY
        self.curveFL = CURVE_FL
        self.ref = REF

    def __str__(self):
        return "POS=" + str(self.pos) + '; ' + "SPX=" + str(self.sPx) + '; ' + \
                "SPY=" + str(self.sPy) + '; ' + "HSX=" + str(self.hSx) + '; ' + \
                "HSY=" + str(self.hSy)  + '; ' + "CURVE_FL=" + str(self.curveFL) + \
                '; ' + "REF=" + str(self.ref)

class GroupE(object):
    '''
    Definition of GroupE

    This class is a group of entity objects

    LE   : List of entity objects
    BBOX : List with Pmin and Pmax (Point classes) to construct a custum
           bounding box, if None -> take Pmin and Pmax of LE[0]
    '''
    def __init__(self, LE=[Entity()], BBOX=None):
        self.le  = LE
        self.nob = len(LE)
        if BBOX is None:
            self.bboxGPmin = LE[0].bboxGPmin
            self.bboxGPmax = LE[0].bboxGPmax
        else:
            self.bboxGPmin = BBOX[0]
            self.bboxGPmax = BBOX[1]           

def findRots(UI=None, UO=None, vecNF=None):
    '''
    Description of the function findRots:

    ===ARGS:
    UI    : Direction of the incoming ray or sun direction
    UO    : Opposite direction of the outcoming ray / direction from receiver to facet
    vecNF : Normal of the reflection surface, if given UI and UO are not needed

    ===RETURN:
    Return a list with rotation information, to reflect UI to -UO:
    list[0] -> rotYD : Rotation in Y direction
    list[0] -> rotZD : Rotation in Z direction
    list[2] -> TTT   : Rotation transform object 
    '''
    # 1)Find the normal of the facet but filled in a vector class
    if vecNF is not None:
        vNF = Vector(vecNF)
    else:
        vNF = (UI + UO)*(-0.5)
    vNF = Normalize(vNF)
    vNF.z = np.clip(vNF.z, -1, 1) # Avoid nan value in next operations

    # 2) Apply the inverse rotation operations to find the necessary angles
    # 2.a) Initialisation
    loop = int(0); TT = Transform(); rotY = 0; rotZ =0; opeZ=0;
    # The initial value of the facet normal is (0, 0, 1) but forced to (0, 0, 0)
    # to be sure to activate the while loop below
    vNF_initial = Vector(0., 0., 0.)

    # 2.b) Rotations are found in the loop bellow, at the end we check if after applying
    #      the transform to the initial normal of the facet 'vNF_initial' we have the same
    #      value as the known well oriented facet normal 'vecNF'. If no rotation has been
    #      found an error message will appear
    while (abs(vNF_initial.x - vNF.x) > 1e-4 or abs(vNF_initial.y - vNF.y) > 1e-4 or 
           abs(vNF_initial.z - vNF.z) > 1e-4):
        loop += int(1)
        if loop > 5:
            raise NameError('No rotation has been found!')
        if (loop == 1):
            rotY = np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0):
                opeZ = 0
            else:
                opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = np.arccos(opeZ)
        elif(loop == 2):
            rotY = np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0):
                opeZ = 0
            else:
                opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = -np.arccos(opeZ)
        elif(loop == 3):
            rotY = -np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0):
                opeZ = 0
            else:
                opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = np.arccos(opeZ)
        elif(loop == 4):
            rotY = -np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0):
                opeZ = 0
            else:
                opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = -np.arccos(opeZ)
        rotYD = np.degrees(rotY)
        rotZD = np.degrees(rotZ)
        TTZ = TT.rotateZ(rotZD)
        TTY = TT.rotateY(rotYD)
        TTT = TTZ*TTY
        vNF_initial = Vector(0., 0., 1.)
        vNF_initial = TTT[vNF_initial]
        vNF_initial = Normalize(vNF_initial)

    return [rotYD, rotZD, TTT]

def generateLEfH(HELIO = Heliostat(), PR = None, THEDEG = 0., PHIDEG = 0.):
    '''
    Definition of the function generateLEfH
    This function enables the conversion of an object heliostat to a list of 
    well oriented plane entity / facets to reflect to a given receiver

    ===ARGS:
    HELIO          : A heliostat class object
    THEDEG, PHIDEG : The theta and phi angles in degrees for the sun direction
    PR             : A class Point with the position of the receiver receiver
    
    ===RETURN:
    List of all well oriented plane entity / facets

    Convention -> here an example of a heliostat splited in 4 in x and y
    directions, the matrices used below follow this:

         j0   j1   j2   j3
       ---------------------
    i0 |f00 |f01 |f02 |f03 |    i, j           : matrix indices
       ---------------------    f00, f10, ...  : facet 0, 1, ...
    i1 |f10 |f11 |f12 |f13 |  
       -----------------------> y
    i2 |f20 |f21 |f22 |f23 |
       ----------|----------
    i3 |f30 |f31 |f32 |f33 |
       ----------|----------
                 x
    '''

    # Be sure that the correct agrs have been given
    if not isinstance(HELIO, Heliostat):
        raise Exception("HELIO must be a Heliostat class!")
    if not isinstance(PR, Point):
        raise Exception("The receiver position 'PR' must be a Point class!")

    # Direction of the sun (from (x,y,z) to (0,0,0))
    vSun = convertAnglestoV(THETA=THEDEG, PHI=PHIDEG, TYPE="Sun")
    # Heliostat is splited in facets in x and y directions
    SPX = HELIO.sPx; SPY = HELIO.sPy;
    # Size in x and y of a given facet
    SFX = HELIO.hSx/SPX; SFY = HELIO.hSy/SPY
    # Focal length or distance between heliostat and receiver
    FL = HELIO.curveFL
    # Position of the heliostat
    POSH = Point(HELIO.pos.x, HELIO.pos.y, HELIO.pos.z)
    # Receiver assumed position or the assumed focal length point.
    # Needed to curve the heliostat
    if (FL is not None):
        APOSR = Point(0., 0., 0.+FL)
    else:
        PHTEMP = Point(POSH)
        DTEMP = (PHTEMP - PR).Length()
        APOSR = Point(0., 0., 0.+DTEMP)
    # For the bounding box
    bboxDist = np.sqrt(HELIO.hSx*HELIO.hSx + HELIO.hSy*HELIO.hSy)/2
        
    # Initialisation
    LF = [] # List of facets
    wMx = SFX/2; wMy = SFY/2 # Size of a facet divided by 2
    # Create one facet to be ready to clone other facets
    F1 = Entity(name = "reflector", \
                materialAV = Mirror(reflectivity = HELIO.ref), \
                materialAR = Matte(), \
                geo = Plane( p1 = Point(-wMx, -wMy, 0.),
                             p2 = Point(wMx, -wMy, 0.),
                             p3 = Point(-wMx, wMy, 0.),
                             p4 = Point(wMx, wMy, 0.) ), \
                transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                 translation = np.array([0., 0., 0.]) ))
    
    # Find the positions of facets and store them in matrix MPF[i][j]
    MPF = np.zeros((SPX, SPY), dtype="object") # Matrix of Point object of each facets
    for i in range (0, SPX):
        for j in range (0, SPY):
            MPF[i][j] = Point(-(HELIO.hSx/2.) + (i*SFX) + wMx, -(HELIO.hSy/2.) + (j*SFY) + wMy, 0.)

    # Find transform in function of focal length (for the curve)
    MTF = np.zeros((SPX, SPY), dtype="object") # Matrix of Transform object of each facets
    for i in range (0, SPX):
        for j in range (0, SPY):
            UI = Point(0., 0., 0.) - APOSR
            UI = Normalize(UI)
            UO = MPF[i][j] - APOSR
            UO = Normalize(UO)
            RINF  = findRots(UI=UI, UO=UO)
            MTF[i][j] = Transform(RINF[2])


    # Find the general heliostat rotation transform (like helistat is a unique facet)
    TT = Transform()
    UI = Vector(vSun.x, vSun.y, vSun.z); UO = POSH - PR;
    UI = Normalize(UI); UO = Normalize(UO);
    RINF2  = findRots(UI=UI, UO=UO)
    TTZY = RINF2[2]

    # Apply the general rotation transform to each facet point and then apply translation.
    # This gives the final position of each facet after rotation and translation of
    # the heliostat, stored in the matrix MPFAT 
    MPFAT = np.zeros((SPX, SPY), dtype="object") # equals to MPF after application of transform
    for i in range (0, SPX):
        for j in range (0, SPY):
            tempP = Point(MPF[i][j])
            tempP = TTZY[tempP]
            tempP.x += POSH.x; tempP.y += POSH.y; tempP.z += POSH.z;
            MPFAT[i][j] = Point(tempP)

    # Write the initial coordinate system in term of vectors (x, y and z)
    vecX = Vector(1., 0., 0.); vecY = Vector(0., 1., 0.); vecZ = Vector(0., 0., 1.);

    # Apply the general rotation transform to find the new coordinate system of the heliostat
    vecX = TTZY[vecX]; vecY = TTZY[vecY]; vecZ = TTZY[vecZ];
    vecX = Normalize(vecX); vecY = Normalize(vecY); vecZ = Normalize(vecZ);

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
    wTo = Transform(m = mm2, mInv = mm2Inv) # move from world/initial to object∕new basis
    oTw = Transform(m = mm2Inv, mInv = mm2) # move from object∕new to world/initial basis

    # The normal of the heliostat vecNH = z axis of the new coordinate system
    vecNH = Vector(vecZ) # stored as a vector for transformation purposes
    for i in range (0, SPX):
        for j in range (0, SPY):
            # come back to the initial coordinate system
            vecNF = oTw[vecNH]
            # apply the transform of the facet to consider the curve effect
            vecNF = MTF[i][j][vecNF]
            # Now we return to the new coordinate system, which gives
            # then the normal of the facet (not heliostat) stored in MTF[i][j]
            vecNF = wTo[vecNF]
            vecNF = Normalize(vecNF)
    
            # Find the rotation transform
            RINF3 = findRots(vecNF=vecNF)

            # Once the rotation angles have been found, create the facet as entity object
            tempF1 = Entity(F1)
            tempF1.transformation = Transformation( rotation = np.array([0., RINF3[0], RINF3[1]]), \
                                                    translation = np.array([MPFAT[i][j].x, MPFAT[i][j].y, MPFAT[i][j].z]), \
                                                    rotationOrder = "ZYX")
            tempPP = Point(POSH)
            
            tempF1.bboxGPmin = Point(tempPP.x-bboxDist, tempPP.y-bboxDist, tempPP.z-bboxDist)
            tempF1.bboxGPmax = Point(tempPP.x+bboxDist, tempPP.y+bboxDist, tempPP.z+bboxDist)
            LF.append(tempF1)

    return LF

def Ref_Fresnel(dirEnt, geoTrans):
    '''
    Definition of Ref_Fresnel

    Simple Fresnel reflection

    dirE     : Direction of the ray entering on the surface of reflection
    geoTrans : Transformation of the surface where there is reflection

    return a Vector class containing the direction of the reflected ray
    '''
    if isinstance(dirEnt, Vector) :
        dirE = dirEnt
    else :
        raise Exception("the dirEnt argument must be a Vector class")
    if isinstance(geoTrans, Transform) :
        geoT = geoTrans
    else :
        raise Exception("the geoTrans argument must be a Transform class")

    # Default value of the surface plane normal
    NN = Vector(0., 0., 1)
    
    # Real value of the normal after considering transformation
    TT = geoT
    NN = TT[NN]

    # Information needed from the incoming ray
    V = dirE
    V = Vector(-V.x, -V.y, -V.z)
    
    # Use the equation of Fresnel reflection (plenty explained in pbrtv3 book)
    V = dirE + NN*(2*Dot(NN, V))

    # Be sure V is normalized
    V = Normalize(V)
    
    return V


def Analyse_create_entity(ENTITY, THEDEG = 0., PHIDEG = 0., PLANEDM = 'SM'):
    '''
    Definition of Analyse_create_entity

    Enable a 3D visualization of the created objects

    ENTITY  : A list of objects (Entity classes)
    THEDEG  : The zenith angle of the sun
    PHIDEG  : The azimuth angle of the sun
    PlaneDM : Plane Draw method, two choices 'FM' (First Method) or 'SM'(seconde
              Method). By default 'SM', 'FM' is useful for debug issues

    return a matplotlib fig
    '''
    if (isinstance(ENTITY, Entity)):
        E = []
        E = np.append(E, ENTITY)
        # Enable generic local visualization (part1)
        if isinstance(E[0].geo, Plane):
            GLXmin = min(E[0].geo.p1.x, E[0].geo.p2.x, E[0].geo.p3.x, E[0].geo.p4.x)
            GLYmin = min(E[0].geo.p1.y, E[0].geo.p2.y, E[0].geo.p3.y, E[0].geo.p4.y)
            GLZmin = min(E[0].geo.p1.z, E[0].geo.p2.z, E[0].geo.p3.z, E[0].geo.p4.z)
            GLXmax = max(E[0].geo.p1.x, E[0].geo.p2.x, E[0].geo.p3.x, E[0].geo.p4.x)
            GLYmax = max(E[0].geo.p1.y, E[0].geo.p2.y, E[0].geo.p3.y, E[0].geo.p4.y)
            GLZmax = max(E[0].geo.p1.z, E[0].geo.p2.z, E[0].geo.p3.z, E[0].geo.p4.z)
            GLEcaX = abs(GLXmin-GLXmax); GLEcaY = abs(GLYmin-GLYmax); GLEcaZ = abs(GLZmin-GLZmax);
            GLEcaM = max(GLEcaX, GLEcaY, GLEcaZ)
        # End (part1)
            
    elif (all(isinstance(x, Entity) for x in ENTITY)):
        E = ENTITY
        # Enable generic local visualization (part2)
        # Be carful, if the local is greater than 100km the below need to be modified!
        GLXmin = 100.; GLYmin = 100.; GLZmin = 100.; GLXmax = -100.; GLYmax = -100.; GLZmax = -100.;
        for i in range(0, len(E)):
            if E[i].transformation.transx < GLXmin :
                GLXmin = E[i].transformation.transx
            if E[i].transformation.transx > GLXmax :
                GLXmax = E[i].transformation.transx
            if E[i].transformation.transy < GLYmin :
                GLYmin = E[i].transformation.transy
            if E[i].transformation.transy > GLYmax :
                GLYmax = E[i].transformation.transy
            if E[i].transformation.transz < GLZmin :
                GLZmin = E[i].transformation.transz
            if E[i].transformation.transz > GLZmax :
                GLZmax = E[i].transformation.transz
        GLEcaX = abs(GLXmin-GLXmax); GLEcaY = abs(GLYmin-GLYmax); GLEcaZ = abs(GLZmin-GLZmax);
        GLEcaM = max(GLEcaX, GLEcaY, GLEcaZ)
        # End (part2)
    else:
        raise NameError('ENTITY argument needs to be an Entity object or a list' + \
                        ' of Entity Objects ')

    # calculate the sun direction vector
    vSun = convertAnglestoV(THETA=THEDEG, PHI=PHIDEG, TYPE="Sun")
    
    wsx = -vSun.x; wsy=-vSun.y; wsz=-vSun.z;
    xs = np.linspace(0, 0.1*wsx, 100)
    ys = np.linspace(0, 0.1*wsy, 100)
    zs = np.linspace(0, 0.1*wsz, 100)

    sunDirection = vSun
    LMir = 0; LMir2 = 0;
    TabPhoton = []; atLeastOneInt = []; xn = []; yn = []; zn = []; xr = []; yr = []; zr = [];
    TabPhoton2 = [];atLeastOneInt2 = []; xr2 = []; yr2 = []; zr2 = [];

    for i in range(0, len(E)):
        if (E[i].name == "reflector"):
            LMir += 1
            atLeastOneInt = np.append(atLeastOneInt, False)
            xn = np.append(xn, None); yn = np.append(yn, None); zn = np.append(zn, None);
            xr = np.append(xr, None); yr = np.append(yr, None); zr = np.append(zr, None);      
            TabPhoton = np.append(TabPhoton, Ray(o = Point(wsx+E[i].transformation.transx, wsy+E[i].transformation.transy, wsz+E[i].transformation.transz), \
                                                 d = Vector( sunDirection.x, sunDirection.y, sunDirection.z ), end = 1200.))

    # create the matplotlib figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=Axes3D.name)
    ax.scatter([-1,1], [-1,1], [-1,1], alpha=0.0)
    
    for k in range (0, len(E)):
        # ===================================================================================
        # En commun (!!reinitialized for each loop!!)
        # ===================================================================================
        # all transform separetly       
        tr = Transform()
        Trans = tr.translate(Vector(E[k].transformation.transx, E[k].transformation.transy, \
                                    E[k].transformation.transz))
        Rotx = tr.rotateX(E[k].transformation.rotx)
        Roty = tr.rotateY(E[k].transformation.roty)
        Rotz = tr.rotateZ(E[k].transformation.rotz)

        # total tt of all transform together
        tt = None
        if (E[k].transformation.rotOrder == "XYZ"):
            tt = Trans*Rotx*Roty*Rotz
        elif (E[k].transformation.rotOrder == "XZY"):
            tt = Trans*Rotx*Rotz*Roty
        elif (E[k].transformation.rotOrder == "YXZ"):
            tt = Trans*Roty*Rotx*Rotz
        elif (E[k].transformation.rotOrder == "YZX"):
            tt = Trans*Roty*Rotz*Rotx
        elif (E[k].transformation.rotOrder == "ZXY"):
            tt = Trans*Rotz*Rotx*Roty
        elif (E[k].transformation.rotOrder == "ZYX"):
            tt = Trans*Rotz*Roty*Rotx
        else:
            raise NameError('Unknown rotation order')
        
        tt_inv = tt.inverse(tt)
        
        # ===================================================================================
        
        if isinstance(E[k].geo, Plane):
            # Vertex triangle indices
            vi = np.array([0, 1, 2,                   # indices or triangle 1
                           2, 3, 1], dtype=np.int32)  # indices of triangle 2

            # List of points of the plane
            P = np.array([Point(E[k].geo.p1.x, E[k].geo.p1.y, E[k].geo.p1.z),
                          Point(E[k].geo.p2.x, E[k].geo.p2.y, E[k].geo.p2.z),
                          Point(E[k].geo.p3.x, E[k].geo.p3.y, E[k].geo.p3.z),
                          Point(E[k].geo.p4.x, E[k].geo.p4.y, E[k].geo.p4.z)], dtype = Point)
            
            PlaneMesh = TriangleMesh(tt, tt_inv, vi, P)

            if(E[k].name == "reflector" and THEDEG != None):
                for i in range(0, LMir):
                    t_hit = float('inf')
                    if(PlaneMesh.Intersect(TabPhoton[i])):
                        if (PlaneMesh.thit < t_hit):
                            atLeastOneInt[i] = True
                            LMir2 += 1
                            xr2 = np.append(xr2, None); yr2 = np.append(yr2, None); zr2 = np.append(zr2, None); 
                            atLeastOneInt2 = np.append(atLeastOneInt2, False)
                            p_hit = PlaneMesh.dg.p
                            t_hit = PlaneMesh.thit
                            sunDistance = sunDirection*t_hit
                            tnn = np.linspace(0, 0.001, 20)
                            P1 = PlaneMesh.dg.p ; N1 = PlaneMesh.dg.nn;
                            N1 = FaceForward(N1, sunDirection * -1)
                            # For ploting the normal and the red ray
                            xn[i] = P1.x + tnn * N1.x
                            yn[i] = P1.y + tnn * N1.y
                            zn[i] = P1.z + tnn * N1.z
                            #tr = np.linspace(Photon.mint, t_hit, 100)
                            tr = np.linspace(t_hit*0.98, t_hit, 100)
                            xr[i] = TabPhoton[i].o.x + tr*TabPhoton[i].d.x
                            yr[i] = TabPhoton[i].o.y + tr*TabPhoton[i].d.y
                            zr[i] = TabPhoton[i].o.z + tr*TabPhoton[i].d.z
                            vecTemp = Ref_Fresnel(dirEnt = TabPhoton[i].d, geoTrans = tt)
                            # print("TabPhoton.d = (", TabPhoton[i].d.x, ", ", TabPhoton[i].d.y, ", ", TabPhoton[i].d.z, ")")
                            # print("vecTemp = (", vecTemp.x, ", ", vecTemp.y, ", ", vecTemp.z, ")")
                            TabPhoton2 = np.append(TabPhoton2, Ray(o=p_hit, d=vecTemp, end=120))
                                               
            if (E[k].name == "receiver" and THEDEG != None):
                for i in range(0, LMir2):
                    t_hit = float('inf')
                    if(PlaneMesh.Intersect(TabPhoton2[i])):
                        atLeastOneInt2[i] = True
                        if (PlaneMesh.thit < t_hit):
                            p_hit = PlaneMesh.dg.p
                            t_hit = PlaneMesh.thit
                            tr = np.linspace(TabPhoton2[i].mint, t_hit, 100)
                            xr2[i] = TabPhoton2[i].o.x + tr*TabPhoton2[i].d.x
                            yr2[i] = TabPhoton2[i].o.y + tr*TabPhoton2[i].d.y
                            zr2[i] = TabPhoton2[i].o.z + tr*TabPhoton2[i].d.z
                        
            # Triangles mesh parameters for plot
            # First method (draw even if there is error with an object, useful for debug):
            # ----------------------------->
            if (PLANEDM == 'FM'):
                for i in range(0, PlaneMesh.ntris):
                    Mat = np.array([[PlaneMesh.reftri[i].p1.x, PlaneMesh.reftri[i].p1.y, PlaneMesh.reftri[i].p1.z], \
                                    [PlaneMesh.reftri[i].p2.x, PlaneMesh.reftri[i].p2.y, PlaneMesh.reftri[i].p2.z], \
                                    [PlaneMesh.reftri[i].p3.x, PlaneMesh.reftri[i].p3.y, PlaneMesh.reftri[i].p3.z]])
                    face1 = mp3d.art3d.Poly3DCollection([Mat], alpha = 0.75, linewidths=0.2)
                    face1.set_facecolor(mcolors.to_rgba('grey'))
                    ax.add_collection3d(face1)

            # Second method (better visual, avoid some matplotlib bugs):
            # ----------------------------->
            if (PLANEDM == 'SM'):
                Mat = np.array([[PlaneMesh.reftri[0].p1.x, PlaneMesh.reftri[0].p1.y, PlaneMesh.reftri[0].p1.z], \
                                [PlaneMesh.reftri[0].p2.x, PlaneMesh.reftri[0].p2.y, PlaneMesh.reftri[0].p2.z], \
                                [PlaneMesh.reftri[0].p3.x, PlaneMesh.reftri[0].p3.y, PlaneMesh.reftri[0].p3.z], \
                                [PlaneMesh.reftri[1].p1.x, PlaneMesh.reftri[1].p1.y, PlaneMesh.reftri[1].p1.z], \
                                [PlaneMesh.reftri[1].p2.x, PlaneMesh.reftri[1].p2.y, PlaneMesh.reftri[1].p2.z], \
                                [PlaneMesh.reftri[1].p3.x, PlaneMesh.reftri[1].p3.y, PlaneMesh.reftri[1].p3.z]])
                
                if (np.array_equal(Mat[:,0], np.full((6), Mat[0,0]))):
                    yy, zz = np.meshgrid(Mat[:,0], Mat[:,2])
                    xx = np.full((6,6), Mat[0,0])
                    ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba('grey'), alpha = 0.15, \
                                    linewidth=0.2, antialiased=True)
                elif (np.array_equal(Mat[:,1], np.full((6), Mat[0,1]))):
                    xx, zz = np.meshgrid(Mat[:,0], Mat[:,2])
                    yy = np.full((6,6), Mat[0,1])
                    ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba('grey'), alpha = 0.15, \
                                    linewidth=0.2, antialiased=True)
                elif (np.array_equal(Mat[:,2], np.full((6), Mat[0,2]))): # need to be verified
                    xx, yy = np.meshgrid(Mat[:,0], Mat[:,1])
                    zz = np.full((6,6), Mat[0,2])
                    ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba('grey'), alpha = 0.15, \
                                    linewidth=0.2, antialiased=True)
                else:
                    ax.plot_trisurf(Mat[:,0], Mat[:,1], Mat[:,2], color = mcolors.to_rgba('grey'), \
                                    alpha = 0.5, linewidth=0.2, antialiased=True)

        elif isinstance(E[k].geo, Spheric):

            S = Sphere(tt, tt_inv, E[k].geo.radius, E[k].geo.z0, E[k].geo.z1, E[k].geo.phi)

            # Plot parameters
            u1 = np.linspace(0, S.phiMax, 20)
            v1 = np.linspace(S.thetaMin, S.thetaMax, 20)
            myP = tt[Point(S.radius * np.outer(np.cos(u1), np.sin(v1)), \
                           S.radius * np.outer(np.sin(u1), np.sin(v1)), \
                           S.radius * np.outer(np.ones(np.size(u1)), np.cos(v1)))]
            x1 = myP.x
            y1 = myP.y
            z1 = myP.z
            
            ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, color='b', alpha=0.5)

            for i in range(0, LMir):
                if(S.Intersect(TabPhoton[i])):
                    atLeastOneInt[i] = True
                    if (S.thit < t_hit):
                        p_hit = S.dg.p
                        t_hit = S.thit
                        sunDistance = sunDirection*t_hit
                        tnn = np.linspace(0, 0.001, 20)
                        P1 = S.dg.p ; N1 = S.dg.nn;
                        N1 = FaceForward(N1, sunDirection * -1)
                        # For ploting the normal and the red ray
                        xn[i] = P1.x + tnn * N1.x
                        yn[i] = P1.y + tnn * N1.y
                        zn[i] = P1.z + tnn * N1.z
                        #tr = np.linspace(Photon.mint, t_hit, 100)
                        tr = np.linspace(t_hit*0.98, t_hit, 100)
                        xr[i] = TabPhoton[i].o.x + tr*TabPhoton[i].d.x
                        yr[i] = TabPhoton[i].o.y + tr*TabPhoton[i].d.y
                        zr[i] = TabPhoton[i].o.z + tr*TabPhoton[i].d.z
        else:
            raise NameError('This geometry is unknown!')


    # ==============================================
    # plot all the geometries
    for i in range(0, LMir):
        if (atLeastOneInt[i]):
            ax.plot(xr[i], yr[i], zr[i], color='r', linewidth=1)

    for i in range(0, LMir2):
        if (atLeastOneInt2[i]):
            ax.plot(xr2[i], yr2[i], zr2[i], color='r', linewidth=1)

    # Enable generic local visualization (part3)
    if (len(E) == 1):
        if (GLEcaZ == GLEcaM):
            ax.set_zlim3d(E[0].transformation.transz+GLZmin, E[0].transformation.transz+GLZmax)
        else:
            TempVal = GLEcaM - GLEcaZ
            ax.set_zlim3d(E[0].transformation.transz + GLZmin-(0.5*GLEcaM), E[0].transformation.transz + GLZmax+(0.5*GLEcaM))
        if (GLEcaX == GLEcaM):
            ax.set_xlim3d(E[0].transformation.transx+GLXmin, E[0].transformation.transx+GLXmax)
        else:
            TempVal = GLEcaM - GLEcaX
            ax.set_xlim3d(E[0].transformation.transx+GLXmin-(0.5*GLEcaM), E[0].transformation.transx+GLXmax+(0.5*GLEcaM))
        if (GLEcaY == GLEcaM):
            ax.set_ylim3d(E[0].transformation.transy+GLYmin, E[0].transformation.transy+GLYmax)
        else:
            TempVal = GLEcaM - GLEcaY
            ax.set_ylim3d(E[0].transformation.transy+GLYmin-(0.5*GLEcaM), E[0].transformation.transy+GLYmax+(0.5*GLEcaM)) 
    else:
        ax.set_zlim3d(0, GLEcaM)
        if (GLEcaX == GLEcaM):
            ax.set_xlim3d(GLXmin, GLXmax)
        else:
            TempVal = GLEcaM - GLEcaX
            ax.set_xlim3d(GLXmin-(0.5*GLEcaM), GLXmax+(0.5*GLEcaM))
        if (GLEcaY == GLEcaM):
            ax.set_ylim3d(GLYmin, GLYmax)
        else:
            TempVal = GLEcaM - GLEcaY
            ax.set_ylim3d(GLYmin-(0.5*GLEcaM), GLYmax+(0.5*GLEcaM)) 
    # End (part3)    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # Show the geometries
    return ax

def generateHfP(THEDEG=0., PHIDEG = 0., PH = [Point(0., 0., 0.)], PR = Point(0., 0., 0.), \
                HSX = 0.001, HSY = 0.001, REF = 1, HTYPE = None):
    '''
    Definition of generateHfP

    Enable to generate well oriented Heliostats from their positions

    THEDEG  : Sun zenith angle (degree)
    PHIDEG  : Sun azimuth angle (degree)
    PH : Coordinates of the center of heliostats (list of point classes)
    PR : Coordinate of the center of the receiver (point class)
    HSX : Heliostat size in x axis (kilometer)
    HSY : Heliostat size in y axis (kilometer)
    REF : reflectivity of the heliostats
    HTYPE : If specified must be a class heliostat

    return a list of entity object
    '''
    
    lObj = []

    # Case where the heliostat is totally plane
    if (HTYPE is None):
        # compute the sun direction vector
        vSun = convertAnglestoV(THETA=THEDEG, PHI=PHIDEG, TYPE="Sun")

        Hxx = HSX/2; Hyy = HSY/2
        objM = Entity(name = "reflector", \
                      materialAV = Mirror(reflectivity = REF), \
                      materialAR = Matte(reflectivity = 0.), \
                      geo = Plane( p1 = Point(-Hxx, -Hyy, 0.),
                                   p2 = Point(Hxx, -Hyy, 0.),
                                   p3 = Point(-Hxx, Hyy, 0.),
                                   p4 = Point(Hxx, Hyy, 0.) ), \
                      transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                       translation = np.array([0., 0., 0.]) ))


        for i in range (0, len(PH)):
            # 1) Find the normalized vector colinear (and same dir) to the normal of heliostat surface
            vecHR = PH[i]-PR
            vecHR = Normalize(vecHR)

            # 2) Find the necessary rotations to apply on the heliostat to reflect to the receiver
            rInfo = findRots(UI=vSun, UO=vecHR)
            rotYD = rInfo[0]; rotZD = rInfo[1];

            # 3) Once the rotation angles have been found, create heliostat objects
            objMi = Entity(objM);
            objMi.transformation = Transformation( rotation = np.array([0., rotYD, rotZD]), \
                                                   translation = np.array([PH[i].x, PH[i].y, PH[i].z]), \
                                                   rotationOrder = "ZYX")
            lObj.append(objMi)
    # Case where the heliostat is composed by facets (i.g. to consider the curvature)
    elif(isinstance(HTYPE, Heliostat)):
        # Take the commun parameters of all heliostats
        SPX = HTYPE.sPx; SPY = HTYPE.sPy; HSX = HTYPE.hSx; HSY = HTYPE.hSy; CURVE_FL = HTYPE.curveFL;
        
        # Generate all the facets and store them as entity object in a list 
        for i in range (0, len(PH)):
            H0 = Heliostat(SPX=SPX, SPY=SPY, HSX=HSX, HSY=HSY, CURVE_FL=CURVE_FL, POS=PH[i], REF=REF)
            TLE = generateLEfH(HELIO=H0, PR=PR, THEDEG=THEDEG, PHIDEG=PHIDEG)
            lObj.extend(TLE)
    else:
        raise NameError('If HTYPE is specified it must be a Heliostat class!')

    return lObj


def generateHfA(THEDEG=0., PHIDEG = 0., PR = Point(0., 0., 50.), MINANG=0., \
                MAXANG=360., GAPDEG = 5., FDRH = 0.1, NBH = 10, GAPDIST = 0.01, \
                HSX = 0.001, HSY = 0.001, PILLH = 0.006, REF = 1, HTYPE=None):
    '''
    Definition of generateHfA

    Enable to generate well oriented Heliostats from two angles [MINANG, MAXANG] 
 
          y
          ^ 
          |/) ANG
          ---> x

    THEDEG  : Sun zenith angle (degree)
    PHIDEG  : Sun azimuth angle (degree)
    PR      : Coordinate of the center of the receiver (point class)
            # Heliostats are filled between MINANG and MAXANG
    MINANG  : min value of ANG (degree)
    MAXANG  : max value of ANG (degree)
    GAPDEG  : Fill heliostats every GAPDEG inside [MINANG, MAXANG] (degree)
    FDRH    : First Distance Receiver-Heliostat (kilometer)
    NBH     : number of heliostats to put every GAPDEG
    GAPDIST : After FDRH, the gap between heliostats (kilometer)
    HSX     : Heliostat size in x axis (kilometer)
    HSY     : Heliostat size in y axis (kilometer)
    PILLH   : Pillar height, distance Ground-Heliostat (kilometer)
    REF     : reflectivity of the heliostats
    HTYPE : If specified must be a class heliostat

    return a list of objects
    '''

    # I) Find the position of all heliostats
    lenpH = int(  ( (MAXANG-MINANG)/GAPDEG )*NBH  )
    
    # To avoid a given bug
    if (MAXANG-MINANG < 360.000000001 and MAXANG-MINANG > 359.999999999):
        nbI = int(lenpH/NBH)
    else:
        nbI = int(lenpH/NBH) + 1

    print("Total number of Heliostats = ", nbI*NBH)
    
    pH = []
    myRotZ = MINANG
    RotZT = Transform()

    if (MINANG != MAXANG):
        for i in range (0, nbI):
            Dhr = FDRH
            for j in range (0, NBH):
                myP = Point(Dhr, 0., 0.)
                RotZT = RotZT.rotateZ(myRotZ)
                myP=RotZT[myP]
                pH.append( Point(myP.x, myP.y, myP.z+PILLH) )
                Dhr += GAPDIST
            myRotZ += GAPDEG
    else:
        Dhr = FDRH
        for j in range (0, NBH):
            myP = Point(Dhr, 0., 0.)
            RotZT = RotZT.rotateZ(myRotZ)
            myP=RotZT[myP]
            pH.append( Point(myP.x, myP.y, myP.z+PILLH) )
            Dhr += GAPDIST      


    # II) Creation of heliostats
    lObj = []

    # Case where the heliostat is totally plane
    if (HTYPE is None):
        # calculate the sun direction vector
        vSun = convertAnglestoV(THETA=THEDEG, PHI=PHIDEG, TYPE="Sun")

        Hxx = HSX/2; Hyy = HSY/2
        objM = Entity(name = "reflector", \
                      materialAV = Mirror(reflectivity = REF), \
                      materialAR = Matte(), \
                      geo = Plane( p1 = Point(-Hxx, -Hyy, 0.),
                                   p2 = Point(Hxx, -Hyy, 0.),
                                   p3 = Point(-Hxx, Hyy, 0.),
                                   p4 = Point(Hxx, Hyy, 0.) ), \
                      transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                       translation = np.array([0., 0., 0.]) ))

        for i in range (0, len(pH)):
            # 1) The vector of the photon after a reflection (here the opposite direction) 
            vecHR = pH[i]-PR
            vecHR = Normalize(vecHR)

            # 2) The incoming (vSun) and outcoming (vecHR) directions are known then find
            #    the rotation angles
            rInfo = findRots(UI=vSun, UO=vecHR)
            rotYD = rInfo[0]; rotZD = rInfo[1];

            # 3) Once the rotation angles have been found, create heliostat objects 
            objMi = Entity(objM);
            objMi.transformation = Transformation( rotation = np.array([0., rotYD, rotZD]), \
                                                   translation = np.array([pH[i].x, pH[i].y, pH[i].z]), \
                                                   rotationOrder = "ZYX")
            lObj.append(objMi)
        
    # Case where the heliostat is composed by facets (i.g. to consider the curvature)
    elif(isinstance(HTYPE, Heliostat)):
        # Take the commun parameters of all heliostats
        SPX = HTYPE.sPx; SPY = HTYPE.sPy; HSX = HTYPE.hSx; HSY = HTYPE.hSy; CURVE_FL = HTYPE.curveFL;
        
        # Generate all the facets and store them as entity object in a list 
        for i in range (0, len(pH)):
            H0 = Heliostat(SPX=SPX, SPY=SPY, HSX=HSX, HSY=HSY, CURVE_FL=CURVE_FL, POS=pH[i], REF=REF)
            TLE = generateLEfH(HELIO=H0, PR=PR, THEDEG=THEDEG, PHIDEG=PHIDEG)
            lObj.extend(TLE)
    else:
        raise NameError('If HTYPE is specified it must be a Heliostat class!')
        
    return lObj


def convertVtoAngles(v, TYPE="Sensor", verbose=False):
    """
    Definition of the function convertVtoAngles

    coordinate system convention:

      y
      ^   x : right; y : front; z : top
      |
    z X -- > x

    The description of a direction by a vector v is
    converted by the description by 2 angles: Theta and Phi

    Arg:
    v     : A direction described by Vector class object

    TYPE  : By default TYPE=str(Sensor) where we look from (0,0,0) to
            (x,y,z) but we can be in Sun case i.e. TYPE=str(Sun) and
            where we look at the opposite side : from (x,y,z) to (0,0,0)

    Return Theta and Phi in degrees:
    Theta : Zenith angle, start at Z+ in plane ZX going
            in the trigonometric direction arround y axis

    Phi   : Azimuth angle, start at X+ in plane XY going in
            the trigonométric direction arround z axis
    """
    if isinstance(v, Vector):
        # First be sure that the vector v is normalized
        if (TYPE == "Sensor"):
            v = Normalize(v)
        elif (TYPE == "Sun"):
            v = Normalize(Vector(-v.x, -v.y, -v.z)) # Sun we look at the oposite side
        else:
            raise NameError('TYPE arg must be str(Sensor) or str(Sun)')

        # Find rotations in Y and Z direction (see the function findRots)
        rInfo = findRots(vecNF=v)
        Theta = rInfo[0]; Phi = rInfo[1];

        if verbose : print("Theta=", Theta, "Phi=", Phi)
        
        return Theta, Phi
    else:
        raise NameError('v argument must be a Vector')

def convertAnglestoV(THETA=0., PHI=0., TYPE="Sensor"):
    """
    Definition of the function convertVtoAngles

    coordinate system convention:

      y
      ^   x : right; y : front; z : top
      |
    z X -- > x

    The description of a direction by 2 angles THETA and PHI is
    converted by the description a vector v

    Arg:
    Theta : Zenith angle in degree, start at Z+ in plane ZX going
            in the trigonometric direction arround y axis

    Phi   : Azimuth angle in degree, start at X+ in plane XY going in
            the trigonométric direction arround z axis

    TYPE  : By default TYPE=str(Sensor) where we look from (0,0,0) to
            (x,y,z) but we can be in Sun case i.e. TYPE=str(Sun) and
            where we look at the opposite side : from (x,y,z) to (0,0,0)

    Return a normalized vector v:
    v     : A direction described by Vector class object
    """
    if (TYPE == "Sensor"):
        # By default the vector v = (0, 0, 1) for THETA=0 and PHI=0
        v = Vector(0, 0, 1)
    elif (TYPE == "Sun"):
        # By default the vector v = (0, 0, -1) for THETA=0 and PHI=0
        v = Vector(0, 0, -1)
    else:
        raise NameError('TYPE arg must be str(Sensor) or str(Sun)')
    
    # Creation of the transform object
    TT = Transform()

    # Take the zenith angle for the first rotation in Y axis
    v = TT.rotateY(THETA)[v]

    # Take the azimuth angle for the second rotation in Z axis
    v = TT.rotateZ(PHI)[v]

    # Be sure v is normalized
    v = Normalize(v)
    
    return v

def is_comment(s):
    """
    function to check if a line
    starts with some character.
    Here # for comment
    """
    # return true if a line starts with #
    return s.startswith('#')

def extractPoints(filename):
    """
    Definition of the function extractPoints

    filename : Name of the file (str type) and its location (absolute of
               relative path) containing the x, y and z positions of
               heliostats. The format of the file must contain at least a
               first comment begining by '#', then an empty line and
               finally each lines with the x, y and z coordinates of each
               heliostats seperated by a comma.

    return : List of class point containing the coordinates of heliostats
    """
    # First check if filename is an str type
    if not isinstance(filename, six.string_types) :
        raise NameError('filename must be an str type!')

    # Check if filename can be read, if yes read it
    try:
        with open(filename, "r") as file:
            for curline in dropwhile(is_comment, file):
                insideFile = file.read()
    except FileNotFoundError:
        print(filename + ' has been not found')
    except IOError:
        print("Enter/Exit error with " + filename)
            
    # Looking for a float and fill it in listVal
    listVal = re.findall(r"-?[0-9]+\.?[0-9]*", insideFile)
        
    # Number of dimension and number of heliostats
    nbDim = 3 # x, y and z --> 3 dim
    nbH = int(len(listVal)/nbDim)

    # # Fill the x, y and z coordinates into a list of Point classes
    lPH = []
    for i in range (0, nbH):
        lPH.append(  Point( float(listVal[i*nbDim]), float(listVal[(i*nbDim)+1]),
                            float(listVal[(i*nbDim)+2]) )  )

    return lPH

def random_equal_area_geometries(theta_in_degrees, phi_in_degrees, fov_radius_in_degrees=0.265, N=1):	# central direction
    '''
    equal area geometries inside a circle of radius fov_radius centred 
    phis, thetas are the angular coordinates in degrees w.r.t. the center of the circle		
    '''
    mum = np.cos(np.radians(fov_radius_in_degrees))
    # random sampling of theta according to equal area sampling
    ct=np.sqrt(1. - np.random.rand(N) * (1-mum**2))
    t = np.degrees(np.arccos(ct))
    # uniform sampling for azimuth
    p=np.random.rand(N) * 360.
    # unit vector around which to rotate all previous directions
    u=Normalize(Cross(convertAnglestoV(), convertAnglestoV(THETA=theta_in_degrees, PHI=phi_in_degrees)))
    # rotation matrix calculation
    R=rotation3D(theta_in_degrees,u)
    # new directions
    t2 = np.zeros_like(t)
    p2 = np.zeros_like(p)
    for k in range(N):
        v = convertAnglestoV(THETA=t[k],PHI=p[k]).asarr()
        vv = Vector(R.dot(v))
        t2[k],p2[k] = convertVtoAngles(vv)

    return {'th_deg': t2, 'phi_deg': p2, 'zip':True}


def packed_geometries(theta_in_degrees, phi_in_degrees, fov_radius_in_degrees=0.265):	# central direction
    '''
    optimal packing of 19 equal small circles in a unit circle
    xs, ys are the coordinates of the small circle centers
    phis, thetas are the angular coordinates in degrees w.r.t. the center of the unit circle		
    '''
    xs = np.array([-0.205604646759568224693193969093,
        0.205604646759568224693193969093,
        -0.561722341219392118229847722909,
        0.561722341219392118229847722909,
        -0.205604646759568224693193969093,
        0.205604646759568224693193969093,
        -0.767326987978960342923041692002,
        0.767326987978960342923041692002,
        -0.411209293519136449386387938185,
        0.000000000000000000000000000000,
        0.411209293519136449386387938185,
        -0.767326987978960342923041692002,
        0.767326987978960342923041692002,
        -0.205604646759568224693193969093,
        0.205604646759568224693193969093,
        -0.561722341219392118229847722909,
        0.561722341219392118229847722909,
        -0.205604646759568224693193969093,
        0.205604646759568224693193969093
        ])
    phis = phi_in_degrees + fov_radius_in_degrees*xs			    

    ys = np.array([-0.767326987978960342923041692002, 
        -0.767326987978960342923041692002,
        -0.561722341219392118229847722909,
        -0.561722341219392118229847722909,
        -0.356117694459823893536653753817,
        -0.356117694459823893536653753817,
        -0.205604646759568224693193969093,
        -0.205604646759568224693193969093,
        0.000000000000000000000000000000,
        0.000000000000000000000000000000,
        0.000000000000000000000000000000,
        0.205604646759568224693193969093,
        0.205604646759568224693193969093,
        0.356117694459823893536653753817,
        0.356117694459823893536653753817,
        0.561722341219392118229847722909,
        0.561722341219392118229847722909,
        0.767326987978960342923041692002,
        0.767326987978960342923041692002
        ])
    thetas = theta_in_degrees + fov_radius_in_degrees*ys
    le={'th_deg':thetas, 'phi_deg':phis, 'zip':True}

    return le


    
if __name__ == '__main__':

    Heliostat1 = Entity(name = "reflector", \
                       material = Mirror(reflectivity = 1., roughness = 0.1), \
                       geo = Plane( p1 = Point(-10., -10., 0.),
                                    p2 = Point(-10., 10., 0.),
                                    p3 = Point(10., -10., 0.),
                                    p4 = Point(10., 10., 0.) ), \
                       transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                        translation = np.array([0., 0., 0.]) ))

    Recepteur1 = Entity(name = "receiver", \
                        material = Mirror(reflectivity = 1., roughness = 0.1), \
                        geo = Plane( p1 = Point(-10., -10., 0.),
                                     p2 = Point(-10., 10., 0.),
                                     p3 = Point(10., -10., 0.),
                                     p4 = Point(10., 10., 0.) ), \
                        transformation = Transformation( rotation = np.array([45., 0., 0.]), \
                                                         translation = np.array([0., -10., 80.]) ))
    Heliostat2 = Entity(name = "reflector", \
                        material = Mirror(reflectivity = 1., roughness = 0.1), \
                        geo = Spheric( radius = 20.,
                                       z0 = -0.,
                                       z1 = 20.,
                                       phi = 360. ), \
                        transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                         translation = np.array([0., 15., 30.]) ))


    print("Helio1 :", Heliostat1)
    print("Recept1 :", Recepteur1)
    print("Helio2 :", Heliostat2)
    
    fig = Analyse_create_entity([Heliostat1, Recepteur1, Heliostat2], THEDEG = 0.)

    plt.show(fig)
