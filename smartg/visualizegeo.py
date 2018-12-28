#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import geometry
from .geometry import Vector, Point, Normal, Ray, BBox
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

def receiver_view(disMatrix, w = False, logI=False, nameFile = None):

    '''
    disMatrix : numpy array with flux distribution at the receiver
    w : the receiver size, in meter. Can be a scalar or a list with x and y values
    logI : enable log interval
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
        cax = plt.imshow(m*1320, cmap=plt.get_cmap('jet'), interpolation='None', \
                         extent = [-(wy*1000),(wy*1000),-(wx*1000),(wx*1000)])
    else:
        m2 = m
        if (np.amin(m2) < 0.00001):
            valmin = 0.00001
        else:
            valmin = np.amin(m2)
            
        cax = plt.imshow(m*1320, cmap=plt.get_cmap('jet'), \
                         norm=mcolors.LogNorm(vmin=valmin*1320, vmax=np.amax(m*1320)), \
                         interpolation='None', extent = [-(wy*1000),(wy*1000),-(wx*1000),(wx*1000)])

    cbar = plt.colorbar()
    cbar.remove()
    cbar = plt.colorbar(cax)
    cbar.set_label(r'Irradiance (W m$^{-2}$)', fontsize = 12)

    plt.xlabel(r'Horizontale position (m) -y axis-')
    plt.ylabel(r'Verticale position (m) -x axis-')
    plt.title('Receiver surface')
    if (nameFile is not None):
        plt.savefig(nameFile + '.pdf')  


def cat_view(mlut, view = 'all'):
    '''
    mlut : mlut table
    view : what to print ? choice between : all, weight, number, errP and errAbs
    '''
    m = mlut
    for i in range (0, 8):
        if (view == 'all'):
            print("CAT",i+1,": weight=", mlut['catWeightPh'][i], " number=", np.uint64(mlut['catNbPh'][i]),
                  " err(%)=", mlut['catErrP'][i], " errAbs=", mlut['catErrAbs'][i])
        elif (view == 'weight'):
            print("CAT",i+1,": weight=", mlut['catWeightPh'][i])
        elif (view == 'number'):
            print("CAT",i+1,": number=", np.uint64(mlut['catNbPh'][i]))
        elif (view == 'errP'):
            print("CAT",i+1,": err(%)=", mlut['catErrP'][i])
        elif (view == 'errAbs'):
            print("CAT",i+1,": errAbs=", mlut['catErrAbs'][i])


class Mirror(object):
    '''
    definition...
    '''
    def __init__(self, reflectivity = 1., rugosity = 0.):
        self.reflectivity = reflectivity
        self.rugosity = rugosity

    def __str__(self):
        return 'Material -> Mirror : ' \
            'reflectivity=' + str(self.reflectivity) + ', rugosity=' + str(self.rugosity)

class LambMirror(object):
    '''
    definition...
    '''
    def __init__(self, reflectivity = 0.5):
        self.reflectivity = reflectivity

    def __str__(self):
        return 'Material -> Lambertian Mirror : ' \
            'reflectivity=' + str(self.reflectivity)

class Matte(object):
    '''
    definition...
    '''
    def __init__(self, reflectivity = 0.):
        self.reflectivity = reflectivity

    def __str__(self):
        return 'Material -> Matte : ' \
            'reflectivity=' + str(self.reflectivity)

class Plane(object):
    '''
    Plane constructed with 4 points : p1, p2, p3, p4
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
    Sphere constructed with --->
    radius : the radius of th e sphere
    radiusZ0 : take into account all the sphere -> radiusZ0 = -radius
    radiusZ1 : take into account all the sphere -> radiusZ1 = +radius
    phi : the value of phi, 360 degrees is the value of a full sphere
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
    rotation in x, y and z in degrees
    translation in x, y and z in meters
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
    definition...
    '''
    def __init__(self, entity = None, name="reflector", TC = 0.01, materialAV=Matte(), \
                 materialAR=Matte(), geo=Plane(), transformation=Transformation()):
        if isinstance(entity, Entity) :
            self.name = entity.name; self.TC = entity.TC; self.materialAV = entity.materialAV;
            self.materialAR = entity.materialAR; self.geo = entity.geo ; self.transformation = entity.transformation;
        else:
            self.name = name
            self.TC = TC
            self.materialAV = materialAV
            self.materialAR = materialAR
            self.geo = geo
            self.transformation = transformation

    def __str__(self):
        return 'The entity is a ' + str(self.name) + ' with the following carac:\n' + \
            str(self.material) + '\n' + \
            str(self.geo) + '\n' + \
            str(self.transformation)

    
def Ref_Fresnel(dirEnt, geoTrans):
    '''
    Simple Fresnel reflection
    dirE : direction of the ray entering on the surface of reflection
    geoTrans : transformation of the surface where there is reflection
    '''
    if isinstance(dirEnt, Vector) :
        dirE = dirEnt
    else :
        raise Exception("the dirEnt argument must be a Vector class")
    if isinstance(geoTrans, Transform) :
        geoT = geoTrans
    else :
        raise Exception("the geoTrans argument must be a Transform class")

    MyT = Transform()
    RotPi = MyT.rotateZ(180.)
    TT = geoT
    invTT = TT.inverse(TT)
    V1 = dirE
    V2 = invTT[V1]
    V2 = RotPi[V2]
    V2 = Vector(V2.x*-1., V2.y*-1., V2.z*-1.)
    V2 = TT[V2]
    return V2
    
def Analyse_create_entity(entity, Theta):
    '''
    definition
    '''
    if (isinstance(entity, Entity)):
        E = []
        E = np.append(E, entity)
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
            
    elif (all(isinstance(x, Entity) for x in entity)):
        E = entity
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
        raise NameError('entity argument need to be an Entity objet or a list' + \
                        ' of Entity Object ')

    #atLeastOneInt = False
    #t_hit = 9999.
    # wsx = 120.*np.sin(Theta*(np.pi/180.)); wsy = 0.; wsz = 120.*np.cos(Theta*(np.pi/180.));
    #wsx += E[0].transformation.transx; wsy += E[0].transformation.transy; wsz += E[0].transformation.transz;
    vSun = Vector(0., 0., -1.)
    tSunTheta = Transform(); tSunPhi = Transform(); tSunThethaPhi = Transform();
    tSunTheta = tSunThethaPhi.rotateY(Theta) 
    tSunPhi = tSunThethaPhi.rotateZ(0)   # pas vérifié car valeur gene = 0
    tSunThethaPhi = tSunTheta * tSunPhi
    vSun = tSunThethaPhi[vSun]
    vSun = Normalize(vSun)
    
    wsx = -vSun.x; wsy=-vSun.y; wsz=-vSun.z;
    xs = np.linspace(0, 0.1*wsx, 100)
    ys = np.linspace(0, 0.1*wsy)
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
            
            for i in range(0, LMir):
                t_hit = 9999.
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
                                               
            if (E[k].name == "receiver"):
                for i in range(0, LMir2):
                    t_hit = 9999.
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

            # First method :
            # ----------------------------->
            # for i in xrange(0, PlaneMesh.ntris):
            #     Mat = np.array([[PlaneMesh.reftri[i].p1.x, PlaneMesh.reftri[i].p1.y, PlaneMesh.reftri[i].p1.z], \
            #                     [PlaneMesh.reftri[i].p2.x, PlaneMesh.reftri[i].p2.y, PlaneMesh.reftri[i].p2.z], \
            #                     [PlaneMesh.reftri[i].p3.x, PlaneMesh.reftri[i].p3.y, PlaneMesh.reftri[i].p3.z]])
            #     face1 = mp3d.art3d.Poly3DCollection([Mat], alpha = 0.5, linewidths=0.2)
            #     face1.set_facecolor(mcolors.to_rgba('grey'))
            #     ax.add_collection3d(face1)

            # Second method (better visual, avoid some matplotlib bugs):
            # ----------------------------->           
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

        
if __name__ == '__main__':

    Heliostat1 = Entity(name = "reflector", \
                       material = Mirror(reflectivity = 1., rugosity = 0.1), \
                       geo = Plane( p1 = Point(-10., -10., 0.),
                                    p2 = Point(-10., 10., 0.),
                                    p3 = Point(10., -10., 0.),
                                    p4 = Point(10., 10., 0.) ), \
                       transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                        translation = np.array([0., 0., 0.]) ))

    Recepteur1 = Entity(name = "receiver", \
                        material = Mirror(reflectivity = 1., rugosity = 0.1), \
                        geo = Plane( p1 = Point(-10., -10., 0.),
                                     p2 = Point(-10., 10., 0.),
                                     p3 = Point(10., -10., 0.),
                                     p4 = Point(10., 10., 0.) ), \
                        transformation = Transformation( rotation = np.array([45., 0., 0.]), \
                                                         translation = np.array([0., -10., 80.]) ))
    Heliostat2 = Entity(name = "reflector", \
                        material = Mirror(reflectivity = 1., rugosity = 0.1), \
                        geo = Spheric( radius = 20.,
                                       z0 = -0.,
                                       z1 = 20.,
                                       phi = 360. ), \
                        transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                         translation = np.array([0., 15., 30.]) ))


    print("Helio1 :", Heliostat1)
    print("Recept1 :", Recepteur1)
    print("Helio2 :", Heliostat2)
    
    fig = Analyse_create_entity([Heliostat1, Recepteur1, Heliostat2], Theta = 0.)

    plt.show(fig)
