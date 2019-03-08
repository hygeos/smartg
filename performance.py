#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
SMART-G performance tests
'''
# matplotlib
import matplotlib
#matplotlib.use('Agg') # particular use
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib.colors import LogNorm

# smartg
from smartg.smartg import Smartg
from smartg.smartg import LambSurface, RoughSurface
from smartg.atmosphere import AtmAFGL, AeroOPAC, CloudOPAC
from luts.luts import merge
from smartg.water import IOP_1
from smartg.tools.smartg_view import smartg_view, input_view

# other import (for comparaison between commits, ...)
import numpy as np
import sys
import imp
import math
import os
import subprocess

# other import in function of python version
try:
    # Python 2.7
    reload(sys)
    sys.setdefaultencoding('utf8')
except NameError:
    try:
        # Python 3.4+
        from importlib import reload
    except ImportError:
        # Python 3.0 - 3.3
        from imp import reload

# *************************************************************************
#                !!!  MODIFICATION --> GO PART 2 !!!
# *************************************************************************

#==========================================================================
# PART1 CAN BE MODIFIED BUT NOT MANDATORY:
#==========================================================================  
def test_rayleigh(**kwargv):
    '''
    Basic Rayleigh example
    '''
    if option_cuda_time == False:
        print("==============================================")
        print("Basic Rayleigh")
        print("==============================================")
    m = Smartg(debug_photon=False, double = False).run(NF=1e6, wl=400., NBPHOTONS=1e8,
                     atm=AtmAFGL('afglt'), progress=False, **kwargv)
    if option_cuda_time == False:
        print(choose_attrs + " :", m.attrs[choose_attrs])
        print(m.attrs['device'])
    return m.attrs

def test_rayleighLE(**kwargv):
    '''
    LE Basic Rayleigh example
    '''
    if option_cuda_time == False:
        print("==============================================")
        print("Basic Rayleigh with LE")
        print("==============================================")
    loc = {'phi':np.array([0]), 'th':np.array([1.57])}
    m = Smartg(double = False).run(NF=1e6, le=loc, wl=400., NBPHOTONS=1e9,
                     atm=AtmAFGL('afglt'), progress=False, **kwargv)
    if option_cuda_time == False:
        print(choose_attrs + " :", m.attrs[choose_attrs])
        print(m.attrs['device'])
    return m.attrs

def test_sp(**kwargv):
    '''
    Basic test in spherical
    '''
    if option_cuda_time == False:
        print("==============================================")
        print("Basic Rayleigh in spherical ")
        print("==============================================")
    m = Smartg(pp=False, double = False).run(NF=1e6, wl=400., NBPHOTONS=1e9, 
                             atm=AtmAFGL('afglt'), progress=False, **kwargv)
    if option_cuda_time == False:
        print(choose_attrs + " :", m.attrs[choose_attrs])
        print(m.attrs['device'])
    return m.attrs

def test_rayleigh_grid(**kwargv):
    '''
    Use a custom atmosphere grid
    '''
    if option_cuda_time == False:
        print("==============================================")
        print("Rayleigh custom atmosphere grid ")
        print("==============================================")
    pro = AtmAFGL('afglt', grid='100[75]25[5]10[1]0')
    m = Smartg(pp=False, double = False).run(NF=1e6, wl=500., NBPHOTONS=1e9, 
                     atm=pro, progress=False, **kwargv)
    if option_cuda_time == False:
        print(choose_attrs + " :", m.attrs[choose_attrs])
        print(m.attrs['device'])
    return m.attrs

def test_aerosols(**kwargv):
    '''
    test with aerosols
    '''
    if option_cuda_time == False:
        print("==============================================")
        print("test with aerosols")
        print("==============================================")
    aer = AeroOPAC('maritime_clean', 0.4, 550.)
    pro = AtmAFGL('afglms', comp=[aer])
    m = Smartg(pp=False, double = False).run(NF=1e6, wl=490., atm=pro, NBPHOTONS=1e9,
                     progress=False, **kwargv)
    if option_cuda_time == False:
        print(choose_attrs + " :", m.attrs[choose_attrs])
        print(m.attrs['device'])
    return m.attrs

def test_aerosols2(**kwargv):
    '''
    test with aerosols2
    '''
    if option_cuda_time == False:
        print("==============================================")
        print("test with aerosols2")
        print("==============================================")
    aer = AeroOPAC('desert', 0.4, 550.)
    pro = AtmAFGL('afglms', comp=[aer])
    m = Smartg(double = False).run(NF=1e6, wl=490., atm=pro, NBPHOTONS=1e9,
                     progress=False, **kwargv)
    if option_cuda_time == False:
        print(choose_attrs + " :", m.attrs[choose_attrs])
        print(m.attrs['device'])
    return m.attrs

# def test_aerocloud(**kwargv):
#     '''
#     test with aerocloud
#     '''
#     if option_cuda_time == False:
#         print("==============================================")
#         print("test with aerocloud")
#         print("==============================================")
#     aer = AeroOPAC('maritime_clean', 0.1, 550.)
#     cloud = CloudOPAC('CUMA',[('wc.sol.mie',1.,12.68,2.,3.)], 5., 550.)
#     pro = AtmAFGL('afglss.dat', comp=[aer], cloud=cloud, grid='100[25]25[5]5[1]0')
#     m = Smartg(pp=False).run(NF=1e6, wl=490., atm=pro, NBPHOTONS=1e9,
#                      progress=False, **kwargv)
#     if option_cuda_time == False:
#         print(choose_attrs + " :", m.attrs[choose_attrs])
#         print(m.attrs['device'])
#     return m.attrs

def test_atm_surf(**kwargv):
    '''
    atmosphere + lambertian surface of albedo 10%
    '''
    if option_cuda_time == False:
        print("==============================================")
        print("atmosphere + lambertian surface of albedo 10%")
        print("==============================================")
    m = Smartg(pp=False, double = False).run(wl=490., NF=1e6, NBPHOTONS=1e9, atm=AtmAFGL('afglms'),
                     surf=LambSurface(ALB=0.1), progress=False, **kwargv)
    if option_cuda_time == False:
        print(choose_attrs + " :", m.attrs[choose_attrs])
        print(m.attrs['device'])
    return m.attrs

# def test_atm_surf_ocean(**kwargv):
#     if option_cuda_time == False:
#         print("==============================================")
#         print("test_atm_surf_ocean")
#         print("==============================================")
#     m = Smartg(pp=False).run(wl=490., NF=1e6, NBPHOTONS=1e9,
#                      atm=Profile('afglms', aer=AeroOPAC('maritime_clean', 0.2, 550)),
#                      surf=RoughSurface(), NBTHETA=30, water=IOP_MM(chl=1., NANG=1000),
#                      progress=False, **kwargv)
#     if option_cuda_time == False:
#         print(choose_attrs + " :", m.attrs[choose_attrs])
#         print(m.attrs['device'])
#     return m.attrs

# def test_surf_ocean(**kwargv):
#     if option_cuda_time == False:
#         print("==============================================")
#         print("test_surf_ocean")
#         print("==============================================")
#     m = Smartg(pp=False).run(wl=490., NF=1e6, THVDEG=30., NBPHOTONS=1e9,
#                      surf=RoughSurface(), water=IOP_MM(1., pfwav=[400.]),
#                      progress=False, **kwargv)
#     if option_cuda_time == False:
#         print(choose_attrs + " :", m.attrs[choose_attrs])
#         print(m.attrs['device'])
#     return m.attrs

# def test_ocean(**kwargv):
#     if option_cuda_time == False:
#         print("==============================================")
#         print("test_ocean")
#         print("==============================================")
#     m = Smartg(pp=False).run(wl=560., NF=1e6, THVDEG=30., water=IOP_SPM(100.),
#                      NBPHOTONS=1e9, progress=False, **kwargv)
#     if option_cuda_time == False:
#         print(choose_attrs + " :", m.attrs[choose_attrs])
#         print(m.attrs['device'])
#     return m.attrs

# def test_oceanLE(**kwargv):
#     if option_cuda_time == False:
#         print("==============================================")
#         print("test_ocean with LE")
#         print("==============================================")
#     loc = {'phi':np.array([0]), 'th':np.array([1.57])}
#     m = Smartg().run(wl=560., NF=1e4, THVDEG=30., water=IOP_SPM(100.),
#                      NBPHOTONS=4e7, progress=False, le = loc, **kwargv)
#     if option_cuda_time == False:
#         print(choose_attrs + " :", m.attrs[choose_attrs])
#         print(m.attrs['device'])
#     return m.attrs

def test_objForward(**kwargv):
    '''
    Basic Rayleigh example
    '''
    # smartg for 3D object
    from smartg.smartg import CusForward
    import smartg.geometry
    from smartg.geometry import Vector, Point, Normal, Ray, BBox
    import smartg.transform
    from smartg.transform import Transform, Aff
    import smartg.visualizegeo
    from smartg.visualizegeo import receiver_view, Mirror, Plane, Spheric, Transformation, \
        Entity, Analyse_create_entity, LambMirror, Matte, cat_view
    
    if option_cuda_time == False:
        print("==============================================")
        print("Basic Object in Forward mode")
        print("==============================================")

    # Creation des objets MirA, B, C et D et du recepteur Recept1
    wRx = 0.006; wRy = 0.007
    wMx = 0.004725; wMy = 0.00642
    MirA = Entity(name = "reflector", \
                  materialAV = Mirror(reflectivity = 0.88), \
                  materialAR = Matte(reflectivity = 0.), \
                  geo = Plane( p1 = Point(-wMx, -wMy, 0.),
                               p2 = Point(wMx, -wMy, 0.),
                               p3 = Point(-wMx, wMy, 0.),
                               p4 = Point(wMx, wMy, 0.) ), \
                  transformation = Transformation( rotation = np.array([0., 20.281725, 0.]), \
                                                   translation = np.array([0.95, 0., 0.00517]) ))
    

    MirA = Entity(MirA);
    MirA.transformation = Transformation( rotation = np.array([0., 20.281725, 0.]), \
                                          translation = np.array([0.95, 0., 0.00517]) )
    MirB = Entity(MirA);
    MirB.transformation = Transformation( rotation = np.array([0., 29.460753, 0.]), \
                                          translation = np.array([0.9, 0., 0.00517]) )
    MirC = Entity(MirA);
    MirC.transformation = Transformation( rotation = np.array([0., 35.129831, 0.]), \
                                          translation = np.array([0.85, 0., 0.00517]) )
    MirD = Entity(MirA);
    MirD.transformation = Transformation( rotation = np.array([0., 38.715473, 0.]), \
                                          translation = np.array([0.8, 0., 0.00517]) )
    
    Recept1 = Entity(name = "receiver", TC = 0.0005, \
                     materialAV = Matte(reflectivity = 0.), \
                     materialAR = Matte(reflectivity = 0.), \
                     geo = Plane( p1 = Point(-wRx, -wRy, 0.),
                                  p2 = Point(wRx, -wRy, 0.),
                                  p3 = Point(-wRx, wRy, 0.),
                                  p4 = Point(wRx, wRy, 0.) ), \
                     transformation = Transformation( rotation = np.array([0., -101.5, 0.]), \
                                                      translation = np.array([1., 0., 0.1065]) ))
    
    listobjs = [MirD, MirC, MirB, MirA, Recept1]
    
    # Creation d'un interval d'étude (BBox permettant d'éviter certains tests d'intersection)
    Pmin = [-120., -120., -0.5]
    Pmax = [120., 120., 121]
    interval0 = [Pmin, Pmax]
    
    # angle zenithal du soleil 
    solarDir = 14.3

    # permet de lancer les photons depuis un rectangle en TOA
    custumF = CusForward(CFX=0.5, CFY=0.05, CFTX=1., CFTY=0., LMODE = "FF")
    
    aer = AeroOPAC('desert', 0.25, 550.)
    pro = AtmAFGL('afglms', comp=[aer], P0 = 877, H2O=1.2)
    
    m = Smartg(debug_photon=False, double = True,
               obj3D = True).run(surf = LambSurface(ALB=0.25),
                                 THVDEG=solarDir, NF=1e6, wl=550., NBPHOTONS=1e8, NBLOOP = 2e7,
                                 atm=pro, progress=False, myObjects=listobjs, BEER=1,
                                 interval = interval0, IsAtm = 1, PHVDEG = 0., cusForward=custumF, **kwargv)
    
    if option_cuda_time == False:
        print(choose_attrs + " :", m.attrs[choose_attrs])
        print(m.attrs['device'])
    
    return m.attrs
#==========================================================================

#==========================================================================
# PART 2 MODIFICATION HERE:
#==========================================================================
if __name__ == '__main__' and '__file__' in globals(): 
    #==========================================================
    # Some options:
    #==========================================================
    # If true launch a test with different grid/block sizes 
    option_cuda_time = False
    # if option_cuda_time=true use the following sizes 512
    list_SG = [128, 256, 384, 512]
    list_SB = [32, 64, 128, 256, 384]
    
    if (len(sys.argv) == 1):
        # Default size of XGRID(SG) and XBLOCK(SB)
        # Generaly best performance: SG=128 and SB=64
        SG = 256 
        SB = 256 
    else:
        SG = int(sys.argv[1])
        SB = int(sys.argv[2])
    #==========================================================

    #==========================================================
    # List of commits (list_commits[] if no commit):
    #==========================================================
    list_commits = []
    # example: list_commits = ['d57f450e255','a5c8f787bf4']
    #==========================================================

    #==========================================================
    # List of test cases (at least one test)):
    #==========================================================
    list_tests = [test_rayleigh]
    # example : list_tests = [test_sp, test_ocean]
    #==========================================================

    #==========================================================
    # Choose between 0, 1 and 2 (my_attrs[0, 1 or 2])
    #==========================================================
    my_attrs = ['kernel time (s)', 'compilation_time', 'processing time (s)']
    choose_attrs = my_attrs[0]
    #==========================================================
#==========================================================================

    def test_rayleigh(**kwargv):
        '''
        Basic Rayleigh example
        '''
        if option_cuda_time == False:
            print("==============================================")
            print("Basic Rayleigh")
            print("==============================================")
        m = Smartg(debug_photon=False, double = False).run(NF=1e6, wl=400., NBPHOTONS=1e8,
                         atm=AtmAFGL('afglt'), progress=False, **kwargv)
        if option_cuda_time == False:
            print(choose_attrs + " :", m.attrs[choose_attrs])
            print(m.attrs['device'])
        return m.attrs

#==========================================================================
# PART 3 NO MODIFICATION IS NEEDED HERE:
#==========================================================================
    #==========================================================
    # Some functions:
    #==========================================================
    def prepare_measure():
        '''
        Avoids some loss due to the kernel launch  
        '''
        print("==============================================")
        print("computation without measurement")
        print("==============================================")
        # m = Smartg(debug_photon=False).run(NF=1e5, wl=400., NBPHOTONS=1e6,
        #                  atm=AtmAFGL('afglt'), progress=False)

    def cuda_block_time(var_list):
        '''
        Time test with different block/grid sizes
        '''
        list_x=[]
        list_y=[]
        list_z=[]
        # enumerate the list of XGRID sizes
        for indice, SG in enumerate(list_SG):
            print("==============================================")
            print("XGRID = ", SG)
            print("==============================================")
            list_y.append(list_SB)
            # enumerate the list of XBLOCK sizes
            for indice2, SB in enumerate(list_SB):
                list_x.append(list_SG[indice])
                m2 = var_list(XGRID=SG, XBLOCK=SB)
                list_z.append(m2[choose_attrs])
                if indice == 0 and indice2 == 0:
                    var1 = m2[choose_attrs]
                if var1 > m2[choose_attrs]:
                    var1 = m2[choose_attrs]
                print(choose_attrs + "(xblock", SB, ") =", m2[choose_attrs])
        return var1, list_x, list_y, list_z
    #==========================================================

    print("Number of commit given: ", len(list_commits))

    # La commande 'git rev-parse --abbrev-ref HEAD' permet de savoir
    # l'où on se trouve ? Dans quelle branche ?
    my_branch = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']

    # Marche impec sur python 2 -->
    if sys.version[0] == '2':
        output = subprocess.Popen( my_branch, stdout=subprocess.PIPE )\
                           .communicate()[0].rstrip('\n\r')
    # Sur python 3 -->    
    else:
        # Créer une classe avec ce que renvoie la commande se trouvant
        # dans la liste my_branch
        output = subprocess.Popen(my_branch,stdout=subprocess.PIPE)
        # Récupération du renvoie de la commande
        output = output.stdout.read()
        # Conversion en str car renvoie en binaire
        output = output.decode("utf-8") 
        # Enlever les espaces
        output = output.rstrip('\n\r')
        
    number_tests = len(list_tests)
    number_commits = len(list_commits)
    os.system("rm -f data.txt temp.pdf")
    dic_x = {}
    dic_y = {}
    dic_z = {}

    #==========================================================
    # If there are no commit:
    #==========================================================
    if number_commits == 0:
        print("No commit given, default use")
        if output == "HEAD":
            print("You're in a detached branch...")
        else:
            print("you're in: ", output)
            prepare_measure()
            x= [None] * number_tests
            for i in range(0, number_tests):
                if option_cuda_time == False:
                    x[i] = list_tests[i](XGRID=SG, XBLOCK=SB)[choose_attrs]
                else:
                    (x[i], dic_x[i], dic_y[i], dic_z[i]) = cuda_block_time(list_tests[i])
                    #==========================================
                    # Save plot in pdf
                    #==========================================
                    # define normalization
                    from matplotlib import colors
                    linearNorm = colors.Normalize(vmin=0.0001,vmax=1.0)

                    fig2 = plt.figure()
                    ax2 = fig2.add_axes([0.1,0.1,0.88,0.88])
                    # cmap=cm.gray
                    # cmap=cm.hsv_r
                    # cmap = cm.RdBu_r
                    # cmap = cm.Reds
                    cmap = cm.pink

                    xs = np.asarray(dic_x[i])
                    ys = np.asarray(dic_y[i])
                    zs = np.asarray(dic_z[i]).astype(np.float)
                    maxi = zs.max()
                    mini = zs.min()
                    # zs = (zs - mini)/(maxi - mini)
                    zs = ((zs*100)/mini)-100
                    maxi = zs.max()
                    mini = zs.min()

                    line1=ax2.scatter(xs,ys,s=140,c=zs,marker='o',cmap=cmap, vmin=mini, vmax=maxi)

                    # colorbar simple
                    cb = plt.colorbar(line1,ax=ax2,fraction=0.1,pad=0.03,aspect=20)
                    cb.ax.tick_params(width=1.4,labelsize=8)
                    cb.set_label('percentage of loss (' + choose_attrs +')',fontsize=12)

                    ax2.text(.5, .92, list_tests[i].__name__,
                             horizontalalignment='center',
                             transform=ax2.transAxes,
                             fontsize=18, fontweight='bold')

                    ax2.set_yticks(np.asarray(list_SB))
                    ax2.set_xticks(np.asarray(list_SG))
                    ax2.set_ylabel('XBLOCK')
                    ax2.set_xlabel('XGRID')
                    plt.savefig(list_tests[i].__name__ + '.pdf')
    #==========================================================

    #==========================================================
    # If there are only one commit
    #==========================================================
    elif number_commits == 1:
        print("One commit Given")
        if output == "HEAD":
            print("you're in a detached branch...")
        else:
            print("you're in: ", output)
            nimp = os.system("git checkout " + list_commits[0])

            for attr in dir(sys.modules['smartg.atmosphere']):
                if attr not in ('__name__', '__file__'):
                    delattr(sys.modules['smartg.atmosphere'], attr)
            reload(sys.modules['smartg.atmosphere'])

            for attr in dir(sys.modules['smartg.water']):
                if attr not in ('__name__', '__file__'):
                    delattr(sys.modules['smartg.water'], attr)
            reload(sys.modules['smartg.water'])

            for attr in dir(sys.modules['smartg.tools.luts']):
                if attr not in ('__name__', '__file__'):
                    delattr(sys.modules['smartg.tools.luts'], attr)
            reload(sys.modules['smartg.tools.luts'])

            for attr in dir(sys.modules['smartg.tools.smartg_view']):
                if attr not in ('__name__', '__file__'):
                    delattr(sys.modules['smartg.tools.smartg_view'], attr)
            reload(sys.modules['smartg.tools.smartg_view'])

            for attr in dir(sys.modules['smartg.smartg']):
                if attr not in ('__name__', '__file__'):
                    delattr(sys.modules['smartg.smartg'], attr)  
            reload(sys.modules['smartg.smartg'])
                
            from smartg.smartg import Smartg, LambSurface, RoughSurface
            from smartg.atmosphere import AtmAFGL, AeroOPAC, CloudOPAC
            from luts.luts import merge
            from smartg.water import IOP_1
            from smartg.tools.smartg_view import smartg_view, input_view
                
            if nimp == 0:
                prepare_measure()
                x = [None] * number_tests
                for i in range(0, number_tests):
                    if option_cuda_time == False:
                        x[i] = list_tests[i](XGRID=SG, XBLOCK=SB)[choose_attrs]
                    else:
                        (x[i], dic_x[i], dic_y[i], dic_z[i]) = cuda_block_time(list_tests[i])
                        #==========================================
                        # Save plot in pdf
                        #==========================================
                        # define normalization
                        from matplotlib import colors
                        linearNorm = colors.Normalize(vmin=0.0001,vmax=1.0)

                        fig2 = plt.figure()
                        ax2 = fig2.add_axes([0.1,0.1,0.88,0.88])

                        cmap = cm.pink

                        xs = np.asarray(dic_x[i])
                        ys = np.asarray(dic_y[i])
                        zs = np.asarray(dic_z[i]).astype(np.float)
                        maxi = zs.max()
                        mini = zs.min()
                        zs = ((zs*100)/mini)-100
                        maxi = zs.max()
                        mini = zs.min()

                        line1=ax2.scatter(xs,ys,s=140,c=zs,marker='o',cmap=cmap, vmin=mini, vmax=maxi)

                        # colorbar simple
                        cb = plt.colorbar(line1,ax=ax2,fraction=0.1,pad=0.03,aspect=20)
                        cb.ax.tick_params(width=1.4,labelsize=8)
                        cb.set_label('percentage of loss (' + choose_attrs +')',fontsize=12)

                        ax2.text(.5, .92, list_tests[i].__name__,
                                 horizontalalignment='center',
                                 transform=ax2.transAxes,
                                 fontsize=18, fontweight='bold')

                        ax2.set_yticks(np.asarray(list_SB))
                        ax2.set_xticks(np.asarray(list_SG))
                        ax2.set_ylabel('XBLOCK')
                        ax2.set_xlabel('XGRID')
                        plt.savefig(list_tests[i].__name__ + '.pdf')

                os.system("git checkout " + output)
            else:
                raise Exception("ERROR : Use git stash.")
                os.system("git checkout " + output)
    #==========================================================

    #==========================================================
    # If there are several commits
    #==========================================================
    else:
        print("several commits")
        if output == "HEAD":
            print("you're in a detached branch...")
        else:
            #==========================================
            # begining of the code
            #==========================================
            print("you're in: ", output)
            if (os.path.isfile("perf_data.txt")):
                os.system("rm perf_data.txt")
            size_x = number_tests * number_commits
            x = [None] * size_x
            for i in range(0, size_x, number_tests):
                nimp = os.system("git checkout " + list_commits[int(i/number_tests)])

                for attr in dir(sys.modules['smartg.atmosphere']):
                    if attr not in ('__name__', '__file__'):
                        delattr(sys.modules['smartg.atmosphere'], attr)
                reload(sys.modules['smartg.atmosphere'])

                for attr in dir(sys.modules['smartg.water']):
                    if attr not in ('__name__', '__file__'):
                        delattr(sys.modules['smartg.water'], attr)
                reload(sys.modules['smartg.water'])

                for attr in dir(sys.modules['smartg.tools.luts']):
                    if attr not in ('__name__', '__file__'):
                        delattr(sys.modules['smartg.tools.luts'], attr)
                reload(sys.modules['smartg.tools.luts'])

                for attr in dir(sys.modules['smartg.tools.smartg_view']):
                    if attr not in ('__name__', '__file__'):
                        delattr(sys.modules['smartg.tools.smartg_view'], attr)
                reload(sys.modules['smartg.tools.smartg_view'])

                for attr in dir(sys.modules['smartg.smartg']):
                    if attr not in ('__name__', '__file__'):
                        delattr(sys.modules['smartg.smartg'], attr)  
                reload(sys.modules['smartg.smartg'])
                
                from smartg.smartg import Smartg, LambSurface, RoughSurface
                from smartg.atmosphere import AtmAFGL, AeroOPAC, CloudOPAC
                from luts.luts import merge
                from smartg.water import IOP_1
                from smartg.tools.smartg_view import smartg_view, input_view
                
                if nimp == 0:
                    # reload(smartg)
                    # if i == 0:
                    prepare_measure()
                    for j in range(i, i+number_tests):
                        if option_cuda_time == False:
                            # x[0,1,2,...,size_x] and list_tests[0,1,...,number_tests]
                            x[j] = list_tests[j-i](XGRID=SG, XBLOCK=SB)[choose_attrs]
                        else:
                            (x[j], dic_x[j], dic_y[j], dic_z[j]) = cuda_block_time(list_tests[j-i])
                            #==========================================
                            # Save plot in pdf
                            #==========================================
                            # define normalization
                            from matplotlib import colors
                            linearNorm = colors.Normalize(vmin=0.0001,vmax=1.0)

                            fig2 = plt.figure()
                            ax2 = fig2.add_axes([0.1,0.1,0.88,0.88])

                            cmap = cm.pink

                            xs = np.asarray(dic_x[j])
                            ys = np.asarray(dic_y[j])
                            zs = np.asarray(dic_z[j]).astype(np.float)
                            maxi = zs.max()
                            mini = zs.min()
                            zs = ((zs*100)/mini)-100
                            maxi = zs.max()
                            mini = zs.min()

                            line1=ax2.scatter(xs,ys,s=140,c=zs,marker='o',cmap=cmap, vmin=mini, vmax=maxi)

                            # colorbar simple
                            cb = plt.colorbar(line1,ax=ax2,fraction=0.1,pad=0.03,aspect=20)
                            cb.ax.tick_params(width=1.4,labelsize=8)
                            cb.set_label('percentage of loss (' + choose_attrs +')',fontsize=12)

                            ax2.text(.5, .92, list_tests[j-i].__name__,
                                     horizontalalignment='center',
                                     transform=ax2.transAxes,
                                     fontsize=18, fontweight='bold')

                            ax2.set_yticks(np.asarray(list_SB))
                            ax2.set_xticks(np.asarray(list_SG))
                            ax2.set_ylabel('XBLOCK')
                            ax2.set_xlabel('XGRID')
                            plt.savefig('commit' + str((i/number_tests)+1) + '_' + list_tests[j-i].__name__ + '.pdf')
                    
                else:
                    raise Exception("ERROR : Use git stash.")
                    os.system("git checkout " + output)
            os.system("git checkout " + output)

            #==========================================
            # write results in data.txt
            #==========================================
            fichier = open("perf_data.txt", "a")
            fichier.write("commits")
            fichier.writelines([" %s" % item.__name__  for item  in list_tests])
            fichier.write("\n")
            for i in range(0, number_commits):
                fichier.write(str(i+1))
                for j in range(i, i+number_tests):
                    fichier.write(" ")   
                    fichier.write(x[j+i*(number_tests-1)]) # x[1,2,3...]
                fichier.write("\n")
            fichier.close()

            #==========================================
            # Save plot in pdf
            #==========================================
            data = np.genfromtxt('perf_data.txt', delimiter=' ', names=True, dtype=None)
            ax = plt.figure().gca()
            for item in list_tests:
                plt.plot(data['commits'], data[item.__name__], label=item.__name__)
            plt.grid()
            plt.legend(loc='best', fancybox=True, framealpha=0.3)
            plt.axis(xmin=1, ymin=0, xmax=number_commits)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel('Commits')
            ax.set_ylabel('Time(s)')
            plt.savefig('perf_time.pdf')

            #==========================================
            # Plot the pourcentage of gain
            #==========================================
            # Create a matrix Mat[i, j] with i = indice of test cases, j = indice of commits
            Mat = np.zeros((number_tests, number_commits), dtype=np.float64)
            # Take all the time values from x and convert them as float numpy array (xbis)
            xbis = np.array(x, dtype = np.float64)
            idi = np.arange(int(number_tests), dtype = int)
            # Fill correctly the matrix Mat[i,j] with time values
            for j in range(0, number_commits):
                Mat[idi, j] = xbis[(j*number_tests)+idi]
            # Loop bellow = tranformation in poucentage with reference = last commit
            Ref = Mat[idi, (number_commits-1)]
            for j in range(0, number_commits):
                #Mat[idi, j] = ((Ref[idi]*100)/Mat[idi, j])-100
                Mat[idi, j] = ((Mat[idi, j]*100)/Ref[idi])-100

            commits_array = np.arange(1, number_commits+1, dtype = int)
            
            # Plot all tests and save in pdf
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
            for i in range(0, number_tests):
                line3 = ax3.plot(commits_array, Mat[i, :], label=list_tests[i].__name__)
            plt.grid()
            plt.legend(loc='best', fancybox=True, framealpha=0.3)
            plt.axis(xmin=1, ymin = np.amin(Mat), xmax=number_commits, ymax = np.amax(Mat))
            ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax3.set_xlabel('Commits')
            ax3.set_ylabel('Coputational time evolution (' + choose_attrs + ')')
            ax3.set_yticks(list(plt.yticks()[0]) + [np.amin(Mat), np.amax(Mat)])
            ax3.axhline(0, color='black', lw=2)
            plt.savefig('perf_time_percentage.pdf')
    #==========================================================
#==========================================================================
