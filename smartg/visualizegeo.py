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

def receiver_view(mlut, w = False, logI=False):

    '''
    mlut : mlut table
    w : the receiver size, in meter
    logI : enable log interval
    '''
    if w==False :
        raise Exception("In receiver_view(), the receiver size w must be specified!")
        
    m = mlut

    if logI == False :
        cax = plt.imshow(m['C_Receptor'][:,:]*1320, cmap=plt.get_cmap('jet'), interpolation='None', \
                         extent = [-(w*1000),(w*1000),-(w*1000),(w*1000)])
    else:
        #print(("npmin=", np.amin(m['C_Receptor'][:,:])*1320)) 
        m2 = m['C_Receptor'][:,:]#*1000
        if (np.amin(m2) < 0.00001):
            valmin = 0.00001
        else:
            valmin = np.amin(m2)
        #print(("maxval=", np.amax(m2)))
        inte = m2.sum()/1e7
        #print(("sum", m2.sum()))
        #print(("inte=", inte))
        cax = plt.imshow(m['C_Receptor'][:,:]*1360, cmap=cm.jet, norm=LogNorm(vmin=valmin, vmax=np.amax(m['C_Receptor'][:,:]*1360)), interpolation='None', \
                         extent = [-(w*1000),(w*1000),-(w*1000),(w*1000)])
    
    cbar = plt.colorbar(cax)
    cbar.set_label(r'Irradiance (W m$^{-2}$)', fontsize = 12)
    plt.xlabel(r'Horizontale position (m)')
    plt.ylabel(r'Verticale position (m)')
    plt.title('Receiver surface')
    plt.savefig('plot.pdf')  



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
    def __init__(self, rotation = np.zeros(3, dtype=float), translation=np.zeros(3, dtype=float)):
        self.rotation = rotation
        self.rotx = rotation[0]
        self.roty = rotation[1]
        self.rotz = rotation[2]
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
    def __init__(self, name="reflector", TC = 0.01, materialAV=Matte(), materialAR=Matte(), geo=Plane(), \
                 transformation=Transformation()):
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


def Analyse_create_entity(entity, Theta):
    '''
    definition
    '''
    if (isinstance(entity, Entity)):
        E = []
        E = np.append(E, entity)
    elif (all(isinstance(x, Entity) for x in entity)):
        E = entity
    else:
        raise NameError('entity argument need to be an Entity objet or a list' + \
                        ' of Entity Object ')

    atLeastOneInt = False
    t_hit = 9999.
    wsx = 120.*np.sin(Theta*(np.pi/180.)); wsy = 0.; wsz = 120.*np.cos(Theta*(np.pi/180.));
    xs = np.linspace(0, wsx, 100)
    ys = np.linspace(0, wsy, 100)
    zs = np.linspace(0, wsz, 100)

    sunDirection = Normalize(Vector(-wsx, wsy, -wsz))

    Photon = Ray(o = Point(wsx, wsy, wsz), \
                 d = Vector( sunDirection.x, \
                             sunDirection.y, \
                             sunDirection.z ), \
                 end = 120.)
    
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
        tt = Trans*Rotx*Roty*Rotz
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

            if(PlaneMesh.Intersect(Photon)):
                atLeastOneInt = True
                if (PlaneMesh.thit < t_hit):
                    p_hit = PlaneMesh.dg.p
                    t_hit = PlaneMesh.thit
                    sunDistance = sunDirection*t_hit
                    tnn = np.linspace(0, 1, 20)
                    P1 = PlaneMesh.dg.p ; N1 = PlaneMesh.dg.nn;
                    N1 = FaceForward(N1, sunDirection * -1)
                    # For ploting the normal and the red ray
                    xn = P1.x + tnn * N1.x
                    yn = P1.y + tnn * N1.y
                    zn = P1.z + tnn * N1.z
                    tr = np.linspace(Photon.mint, t_hit, 100)
                    xr = Photon.o.x + tr*Photon.d.x
                    yr = Photon.o.y + tr*Photon.d.y
                    zr = Photon.o.z + tr*Photon.d.z

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
            
            if(S.Intersect(Photon)):
                atLeastOneInt = True
                if (S.thit < t_hit):
                    p_hit = S.dg.p
                    t_hit = S.thit
                    sunDistance = sunDirection*t_hit
                    tnn = np.linspace(0, 1, 20)
                    P1 = S.dg.p ; N1 = S.dg.nn;
                    N1 = FaceForward(N1, sunDirection * -1)
                    # For ploting the normal and the red ray
                    xn = P1.x + tnn * N1.x
                    yn = P1.y + tnn * N1.y
                    zn = P1.z + tnn * N1.z
                    tr = np.linspace(Photon.mint, t_hit, 100)
                    xr = Photon.o.x + tr*Photon.d.x
                    yr = Photon.o.y + tr*Photon.d.y
                    zr = Photon.o.z + tr*Photon.d.z
        else:
            raise NameError('This geometry is unknown!')


    # ==============================================
    # plot all the geometries
    if (atLeastOneInt):
        ax.plot(xn, yn, zn, color='g', linewidth=4)
        ax.plot(xr, yr, zr, color='r', linewidth=2)
    else:
        ax.plot(xs, ys, zs, color='y', linewidth=4)
        
    ax.scatter([0],[0],[0],color="g",s=10)
    ax.set_xlim3d(-20, 120)
    ax.set_ylim3d(-20, 120)
    ax.set_zlim3d(-20, 120)
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

    Recepteur1 = Entity(name = "receptor", \
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
