#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Script de visualisation des résultats
'''


import os
import sys
import warnings
warnings.simplefilter("ignore",DeprecationWarning)
import pyhdf.SD
from optparse import OptionParser, Option, OptionValueError
from copy import copy
import subprocess




######################################################
##                PARSE OPTIONS                     ##
######################################################
def check_tuple2(option, opt, value):
    try:
        if len(value.split(','))!=2 :
            raise OptionValueError(
            "option %s: invalid tuple value int,float: %r" % (opt, value))
        else : return (int(value.split(',')[0]),float(value.split(',')[1]))
    except ValueError:
        raise OptionValueError(
            "option %s: invalid tuple value int,float: %r" % (opt, value))
            
def check_tuple3(option, opt, value):
    try:
        if len(value.split(','))!=3 :
            raise OptionValueError(
            "option %s: invalid tuple value string float,float: %r" % (opt, value))
        else : return (value.split(',')[0],float(value.split(',')[1]),float(value.split(',')[2]))
    except ValueError:
        raise OptionValueError(
            "option %s: invalid tuple value string,float,float: %r" % (opt, value))  
            
class MyOption (Option):
    TYPES = Option.TYPES + ("tuple2","tuple3",)
    TYPE_CHECKER = copy(Option.TYPE_CHECKER)
    TYPE_CHECKER["tuple2"] = check_tuple2
    TYPE_CHECKER["tuple3"] = check_tuple3
    
parser = OptionParser(option_class=MyOption,usage='%prog [options] hdf_file [hdf_file2]'
        + ' Polar Plots of MC output hdf file, eventually run and display SOS results\n'    
        + ' In case of a second hdf file, differences between MC file 1 and MC file 2 are plotted\n'
        + ' Options ...\n'
        + '-d --down : downward radiance at BOA (default upward TOA)\n'
        + '-R --Relative : Plot relative differences in percent instead of absolute\n'
        + '-a --aerosol : aerosol model (mandatory if -c is chosen), should be U80 or M80 or T70 or 0 (forcing no aerosol)\n'
        + '-s --savefile : output graphics file name\n'
        + '-r --rmax : maximum reflectance for color scale, in percent in case of relative differences see -R option\n'
        + '-p --percent : choose Polarization Ratio (default polarized reflectance) and set maximum PR for color scale\n'
        + '-t --transect : Add a transect below 2D plot for the closest azimuth to phi0 or theta0 in case of the list option activated\n'
        + '-P --points : optional filename containing data points to be added to the transect, format txt file with columns (phi theta I Q U)\n'
        + '-e --error : choose relative error instead of polarized reflectance and maximum error for color scale\n'
        + '-l --list : the hdf_file argument is replaced by a file contaning a list of input filenames (the int and float separated by a comma x,y given in this option specify first,\n\t\tthe dimension number to be taken as constant in 2D plots :\n\t\t0 : theta\n\t\t1 : phi\n\t\t2 : other\n \t and second,\n \t\t its value\n'
        + '-o --other : string for the 3 dimension ex LAMBDA , WINDSPEED and 2 floats for inf and sup separated by commas; in case of -list option\n'
        + '\nexamples : ---------\n\n\tpython runOSfromMC.py -p 100 -r 0.5 -t 0 out.hdf : polar plot(theta,phi) of the Stokes parameters and reflectance scaled to 0.5 \n\tand polarization ratio scale to 100 percent and transect of azimuth 0 plottted under polar plots\n\n'
        + "\tpython runOSfromMC.py -o LAMBDA,400,700 -l 0,60 list.txt : polar plot(lambda,phi) of the Stokes parameters with the list of simulations output files in the txt file lits.txt\n, where the 3rd dimension is the varying 'LAMBDA' parameter plotted between 400 and 700 and the SZA beeing taken constant at 60 deg. ")

parser.add_option('-d','--down',
            dest='down',
            action="store_true", default=False,
            help =  '-d downward radiance at BOA (default upward TOA)\n')
parser.add_option('-R','--Relative',
            dest='rel',
            action="store_true", default=False,
            help =  '-R --Relative : Plot relative differences in percent instead of absolute\n')
parser.add_option('-s', '--savefile',
            dest='filename',
            help='-s output graphics file name\n'
            )
parser.add_option('-a', '--aerosol',
            dest='aerosol',
            type='string',
            help='-a aerosol model (mandatory if -c is chosen), should be U80 or M80 or T70 or 0 (forcing no aerosol)\n'
            )
parser.add_option('-r', '--rmax',
            type='float',
            dest='rmax',
            help='-r maximum reflectance for color scale, in percent in case of relative differences see -R option'
            )
parser.add_option('-p', '--percent',
            dest='percent',
            type='float',
            help='-p choose Polarization Ratio (default polarized reflectance) and maximum PR for color scale (default 100)\n'
            )
parser.add_option('-t', '--transect',
            dest='phi0',
            type='float',
            help = '-t --transect : Add a transect below 2D plot for the closest azimuth to phi0 or theta0 in case of the list option activated \n'
            )
parser.add_option('-P', '--points',
            dest='points',
            help = '-P --points : optional filename containing data points to be added to the transect, format txt file with columns (phi theta I Q U)\n'
            )
parser.add_option('-e', '--error',
            dest='error',
            type='float',
            help='-e choose relative error instead of polarized reflectance and maximum error for color scale\n'
            )
parser.add_option('-o', '--others',
            dest='others',
            type='tuple3',
            help='-o : string for the 3rd dimension ex LAMBDA , WINDSPEED and 2 floats for inf and sup separated by commas; in case of -list option\n'
            )
parser.add_option('-l', '--list',
            dest='list',
            type='tuple2',
            help='-l the hdf_file argument is replaced by a file contaning a list of input filenames (the int and float separated by a comma x,y given in this option specify first,\n\t\tthe dimension number to be taken as constant in 2D plots :\n\t\t0 : theta\n\t\t1 : phi\n\t\t2 : other\n \t and second,\n \t\t its value'
            )
parser.add_option('-Q', '--QU',
            dest='QU',
            action="store_true", default=False,
            help='-Show Q and U also'
            )
(options, args) = parser.parse_args()
if len(args) != 1 and len(args) != 2:
        parser.print_usage()
        exit(1)

path_cuda = args[0]
if options.list == None:
        path_cuda = args[0]
        if len(args) == 2 :
           path_cuda2 = args[1]
else:
        path_cudal = open(args[0],"r").readlines()
        if len(args) == 2 :
           path_cudal2 = open(args[1],"r").readlines()

if len(args) == 2 :
         path_cuda2 = args[1]

import matplotlib
if options.filename != None:
    matplotlib.use('Agg')
from pylab import savefig, show, figure, subplot, cm
import numpy as np
np.seterr(invalid='ignore', divide='ignore') # ignore division by zero errors
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
#from  mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear

#----------------------------------------------------------------------------
# axes semi polaires
#----------------------------------------------------------------------------
def setup_axes3(fig, rect, options=None):
    """
    Sometimes, things like axis_direction need to be adjusted.
    """

    # rotate a bit for better orientation
    tr_rotate = Affine2D().translate(0, 0)

    # scale degree to radians
    tr_scale = Affine2D().scale(np.pi/180., 1.)

    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

    angle_ticks1 = [(0, r"$0$"),
                   (45, r"$45$"),
                   (90, r"$90$"),
                   (135, r"$135$"),
                   (180, r"$180$"),]
                   
    grid_locator1 = FixedLocator([v for v, s in angle_ticks1])
    tick_formatter1 = DictFormatter(dict(angle_ticks1))
    
    if options==None :
        angle_ticks2 = [(0, r"$0$"),
                   (30, r"$30$"),
                   (60, r"$60$"),
                   (90, r"$90$")]

    else :
       ti=np.linspace(options.others[1],options.others[2],num=4,endpoint=True)
       angle_ticks2 = [(0, r"$%.1f$"%ti[0]),
                   (30, r"$%.1f$"%ti[1]),
                   (60, r"$%.1f$"%ti[2]),
                   (90, r"$%.1f$"%ti[3])]
       
    grid_locator2 = FixedLocator([v for v, s in angle_ticks2])
    tick_formatter2 = DictFormatter(dict(angle_ticks2))


    ra0, ra1 = 0.,180. 
    cz0, cz1 = 0, 90.
    grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                        extremes=(ra0, ra1, cz0, cz1),
                                        grid_locator1=grid_locator1,
                                        grid_locator2=grid_locator2,
                                        tick_formatter1=tick_formatter1,
                                        tick_formatter2=tick_formatter2,
                                        )

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    # adjust axis
    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["right"].set_axis_direction("top")
    ax1.axis["bottom"].set_visible(False)
    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")
    
    if options==None:
        ax1.axis["left"].label.set_text(r"$\theta_{s}$")
        ax1.axis["top"].label.set_text(r"$\phi$")
        
    else :
        if options.others[0]=='LAMBDA':
           ax1.axis["left"].label.set_text(r"$\lambda (nm)$")
        else :
           ax1.axis["left"].label.set_text(options.others[0])
        if options.list[0]==1 : ax1.axis["top"].label.set_text(r"$\theta_{s}$")
        else : ax1.axis["top"].label.set_text(r"$\phi$")
    ax1.grid(True)


    # create a parasite axes whose transData in RA, cz
    aux_ax = ax1.get_aux_axes(tr)

    aux_ax.patch = ax1.patch # for aux_ax to have a clip path as in ax
    ax1.patch.zorder=0.9 # but this has a side effect that the patch is
                        # drawn twice, and possibly over some other
                        # artists. So, we decrease the zorder a bit to
                        # prevent this.

    return ax1, aux_ax
#----------------------------------------------------------------------------



#----------------------------------------------------------------------------
# plot 2D 
#----------------------------------------------------------------------------
def plot_2D_parameter(fig, rect, theta , phi, data, Vdata, Vdatat=None, title=None, label=None, iphi0=-1, sub=None, points=None, method='pcolormesh', options=None) :

    '''
    Contour and eventually transect of 2D parameter (theta,phi)
    fig : destiantion figure
    rect: string 'ABC' soecifiying position in the plot grid (see matplotlib multiple plots)
    theta: 1D vector for zenith angles
    phi: 1D vector for azimut angles
    data: 2D vector parameter
    Vdata: 1D parameter of iso-contour values
    Vdatat: keyword for optional ticks value of the colorbar (if not set: no colorbar added)
    tit : keyword of plot title (default no title)
    lab : keyword of colorbar title (default no label)
    iphi0: keyword of the value of the azimut angle for the transect plot (default : iphi0 < 0, no transect)
    sub : keyword for the position of the transect plot (similar type as rect)
    points : keyword for the data added to the transect coming from an additional txt file (array with phi, theta and data columns)
    method : keyword for choosing plotting method (contour/pcolormesh)
    other :  keyword for specifying that phi is representing another parameter
    '''

    # grille 2D des angles
    r , t = np.meshgrid(theta,phi)
    NN = len(phi)
    if options==None : 
        ticks = np.array([-90,-75,-60,-30,0,30,60,75,90])
        ax3, aux_ax3 = setup_axes3(fig, rect)
    else : 
        if options.list[0]==2: ticks = np.array([-90,-75,-60,-30,0,30,60,75,90])
        else: ticks = np.linspace(options.others[1],options.others[2],num=4,endpoint=True)
        ax3, aux_ax3 = setup_axes3(fig, rect, options=options)

    if iphi0 >= 0 : ax = subplot(sub)
    cmap = cm.jet
    cmap.set_under('black')
    cmap.set_over('white')
    cmap.set_bad('0.5') # grey 50%
    masked_data = np.ma.masked_where(np.isnan(data) | np.isinf(data), data)
    if method == 'contour':
        cax3 = aux_ax3.contourf(t,r,masked_data,Vdata, cmap=cmap)
    else:
        cax3 = aux_ax3.pcolormesh(t,r,masked_data ,cmap=cmap, vmin=Vdata[0], vmax=Vdata[-1])
       
    if title != None : ax3.set_title(title,weight='bold',position=(0.15,0.9))
    if iphi0 >= 0 :
          vertex0 = np.array([[0,0],[phi[iphi0],90]])
          vertex1 = np.array([[0,0],[phi[NN-1-iphi0],90]])
          aux_ax3.plot(vertex0[:,0],vertex0[:,1],'w')
          if options==None:
              aux_ax3.plot(vertex1[:,0],vertex1[:,1],'w--')
              ax.plot(theta,data[iphi0,:],'k-')
              ax.plot(-theta,data[NN-1-iphi0,:],'k--')
              ax.set_xlim(-90,90)
          else :
              if options.list[0]==2 :
                  aux_ax3.plot(vertex1[:,0],vertex1[:,1],'w--')
                  ax.plot(theta,data[iphi0,:],'k-')
                  ax.plot(-theta,data[NN-1-iphi0,:],'k--')
                  ax.set_xlim(-90,90)
              else: 
                  ax.plot(theta/90.*(options.others[2]-options.others[1])+ options.others[1],data[iphi0,:],'k-+')
                  ax.set_xlim(options.others[1],options.others[2])
              
          ax.set_ylim(Vdata[0],Vdata[-1])
          ax.set_xticks(ticks)
          ax.grid(True)
          if points != None :
              phi_txt = points[:,0]
              theta_txt = points[:,1]
              data_txt = points[:,2]
              i=0

              for p in phi_txt : 
                  if p >=90 : sign=-1
                  else : sign=1
                  ax.plot(theta_txt[i]*sign,data_txt[i],'*', ms=7, alpha=0.7, mfc='red')
                  i+=1

    if Vdatat != None :
        cb3=fig.colorbar(cax3,orientation='horizontal',ticks=Vdatat,extend='both')
        if label != None  : cb3.set_label(label)

#----------------------------------------------------------------------------
def setXYZ(theta,phi,other,options,data,data2,SYM='I'):
            
    if options.list == None :
        XX = theta
        YY = phi
        Z = data
        opt=None
        if (data2!=None) :
            Z2 = data2
    else:
        if options.list[0]==0 :
            opt=options
            XX = other
            YY = phi
            Z = data.transpose()
            if (data2!=None) : Z2 = data2.transpose()
   
        if options.list[0]==1 :
            opt=options
            XX = other
            YY = theta
            Z = data.transpose()
            if (data2!=None) : Z2 = data2.transpose()
        if options.list[0]==2 :
            opt=None
            XX = theta
            YY = phi
            Z = data
            if (data2!=None) : Z2 = data2

    if (data2!=None) :
        if options.rel==True :
            ZZ = (Z-Z2)/Z2*100
            TITLE='({}1-{}2)/{}2[%]'.format(SYM,SYM,SYM)
        else:
            ZZ = Z-Z2
            TITLE='{}1-{}2'.format(SYM,SYM)
    else: 
        ZZ = Z
        TITLE='{}'.format(SYM)
    
    return XX,YY,ZZ,TITLE,opt


#----------------------------------------------------------------------------
def main():
    ##########################################################
    ##                DONNEES FICHIER CUDA                  ##
    ##########################################################

    # if a list of inputs files check only existence of the first
    if options.list != None :
        path_cuda = path_cudal[0].rstrip()
        if len(args) == 2 :
           path_cuda2 = path_cudal2[0].rstrip()
        NL = len(path_cudal)
    else :
        path_cuda = args[0]
        if len(args) == 2 :
            path_cuda2 = args[1]
    
    # verification de l'existence du fichier hdf
    if os.path.exists(path_cuda):
        # lecture du fichier hdf
        sd_cuda = pyhdf.SD.SD(path_cuda)
        # lecture du nombre de valeurs de phi
        NBPHI_cuda = getattr(sd_cuda,'NBPHI')
        NBTHETA_cuda = getattr(sd_cuda,'NBTHETA')
        thetas = getattr(sd_cuda,'VZA (deg.)')
        mus = np.cos(thetas * np.pi / 180.)
        WINDSPEED = getattr(sd_cuda,'WINDSPEED')
        W0LAM = getattr(sd_cuda,'W0LAM')
        LAMBDA = getattr(sd_cuda,'LAMBDA')
        NH2O = getattr(sd_cuda,'NH2O')
        SIM = getattr(sd_cuda,'SIM')
        DIOPTRE = getattr(sd_cuda,'DIOPTRE')
        MODE = getattr(sd_cuda,'MODE')

        # Récupération des valeurs de theta
        name = "Zenith angles"
        hdf_theta = sd_cuda.select(name)
        theta = hdf_theta.get()

        # Récupération des valeurs de phi
        name = "Azimut angles"
        hdf_phi = sd_cuda.select(name)
        phi = hdf_phi.get()
        # ATTENTION version uniquement valide pour SOS trafiqué permettant des pas en phi de 0.5 deg
        #Dphi = phi[1] - phi[0] 

        # Vrai pour SOS_V5.1 officiel
        Dphi = (phi[1] - phi[0]) / 2.

        if options.down == True  :
            sds_cuda = sd_cuda.select("I_down (0+)")
            dataI = sds_cuda.get()
            sds_cuda = sd_cuda.select("Q_down (0+)")
            dataQ = sds_cuda.get()
            sds_cuda = sd_cuda.select("U_down (0+)")
            dataU = sds_cuda.get()
            sds_cuda = sd_cuda.select("Numbers of photons")
            dataN = sds_cuda.get()
        else :
            sds_cuda = sd_cuda.select("I_up (TOA)")
            dataI = sds_cuda.get()
            sds_cuda = sd_cuda.select("Q_up (TOA)")
            dataQ = sds_cuda.get()
            sds_cuda = sd_cuda.select("U_up (TOA)")
            dataU = sds_cuda.get()
            sds_cuda = sd_cuda.select("Numbers of photons")
            dataN = sds_cuda.get()


    else:
        sys.stdout.write("Pas de fichier "+path_cuda+"\n")
        sys.exit()

    if len(args) == 2 :
     if os.path.exists(path_cuda2):
        sd_cuda = pyhdf.SD.SD(path_cuda2)
        # lecture du nombre de valeurs de phi
        NBPHI_cuda2 = getattr(sd_cuda,'NBPHI')
        NBTHETA_cuda2 = getattr(sd_cuda,'NBTHETA')
        if options.down == True  :
            sds_cuda = sd_cuda.select("I_down (0+)")
            dataI2 = sds_cuda.get()
            sds_cuda = sd_cuda.select("Q_down (0+)")
            dataQ2 = sds_cuda.get()
            sds_cuda = sd_cuda.select("U_down (0+)")
            dataU2 = sds_cuda.get()
            sds_cuda = sd_cuda.select("Numbers of photons")
            dataN2 = sds_cuda.get()
        else :
            sds_cuda = sd_cuda.select("I_up (TOA)")
            dataI2 = sds_cuda.get()
            sds_cuda = sd_cuda.select("Q_up (TOA)")
            dataQ2 = sds_cuda.get()
            sds_cuda = sd_cuda.select("U_up (TOA)")
            dataU2 = sds_cuda.get()
            sds_cuda = sd_cuda.select("Numbers of photons")
            dataN2 = sds_cuda.get()

     else:
        sys.stdout.write("Pas de fichier "+path_cuda2+"\n")
        sys.exit()

    # if a list of inputs files read SDS of remaining files
    if options.list != None :
        # close first file
        sd_cuda.end()
        
        # build the 3D array containing all SDS
        dataI = np.zeros((NL,NBPHI_cuda,NBTHETA_cuda), dtype=np.float32)
        dataQ = np.zeros((NL,NBPHI_cuda,NBTHETA_cuda), dtype=np.float32)
        dataU = np.zeros((NL,NBPHI_cuda,NBTHETA_cuda), dtype=np.float32)
        dataN = np.zeros((NL,NBPHI_cuda,NBTHETA_cuda), dtype=np.float32)
        other = []
        # loop on input files
        for i in range(NL):
            sd_cuda = pyhdf.SD.SD(path_cudal[i].rstrip())
            other.append(getattr(sd_cuda,options.others[0]))
            if options.down == True  :
                sds_cuda = sd_cuda.select("I_down (0+)")
                dataI[i,:,:] = sds_cuda.get()
                sds_cuda = sd_cuda.select("Q_down (0+)")
                dataQ[i,:,:] = sds_cuda.get()
                sds_cuda = sd_cuda.select("U_down (0+)")
                dataU[i,:,:] = sds_cuda.get()
                sds_cuda = sd_cuda.select("Numbers of photons")
                dataN[i,:,:] = sds_cuda.get()
            else :
                sds_cuda = sd_cuda.select("I_up (TOA)")
                dataI[i,:,:] = sds_cuda.get()
                sds_cuda = sd_cuda.select("Q_up (TOA)")
                dataQ[i,:,:] = sds_cuda.get()
                sds_cuda = sd_cuda.select("U_up (TOA)")
                dataU[i,:,:] = sds_cuda.get()
                sds_cuda = sd_cuda.select("Numbers of photons")
                dataN[i,:,:] = sds_cuda.get()
            sd_cuda.end()

        if len(args)==2:
        # build the 3D array containing all SDS
            dataI2 = np.zeros((NL,NBPHI_cuda,NBTHETA_cuda), dtype=np.float32)
            dataQ2 = np.zeros((NL,NBPHI_cuda,NBTHETA_cuda), dtype=np.float32)
            dataU2 = np.zeros((NL,NBPHI_cuda,NBTHETA_cuda), dtype=np.float32)
            dataN2 = np.zeros((NL,NBPHI_cuda,NBTHETA_cuda), dtype=np.float32)
            other = []
            # loop on input files
            for i in range(NL):
                sd_cuda = pyhdf.SD.SD(path_cudal2[i].rstrip())
                other.append(getattr(sd_cuda,options.others[0]))
                if options.down == True  :
                    sds_cuda = sd_cuda.select("I_down (0+)")
                    dataI2[i,:,:] = sds_cuda.get()
                    sds_cuda = sd_cuda.select("Q_down (0+)")
                    dataQ2[i,:,:] = sds_cuda.get()
                    sds_cuda = sd_cuda.select("U_down (0+)")
                    dataU2[i,:,:] = sds_cuda.get()
                    sds_cuda = sd_cuda.select("Numbers of photons")
                    dataN2[i,:,:] = sds_cuda.get()
                else :
                    sds_cuda = sd_cuda.select("I_up (TOA)")
                    dataI2[i,:,:] = sds_cuda.get()
                    sds_cuda = sd_cuda.select("Q_up (TOA)")
                    dataQ2[i,:,:] = sds_cuda.get()
                    sds_cuda = sd_cuda.select("U_up (TOA)")
                    dataU2[i,:,:] = sds_cuda.get()
                    sds_cuda = sd_cuda.select("Numbers of photons")
                    dataN2[i,:,:] = sds_cuda.get()
                sd_cuda.end()
    ##################################################################################
    ##              CREATION/CHOIX/MODIFICATION DE CERTAINES DONNES                 ##
    ##################################################################################

    #---------------------------------------------------------
    # Sauvegarde de la grandeur désirée
    #---------------------------------------------------------
    # if a list of inputs files read SDS of remaining files
    if options.list != None :
        
        if options.list[0]==0:
            ith1 = (np.abs(theta-options.list[1])).argmin() 
            data_cudaI = np.zeros((NL,NBPHI_cuda), dtype=float)
            data_cudaI = dataI[0:NL,:,ith1]
            data_cudaQ = np.zeros((NL, NBPHI_cuda), dtype=float)
            data_cudaQ = dataQ[0:NL,:,ith1]
            data_cudaU = np.zeros((NL, NBPHI_cuda), dtype=float)
            data_cudaU = dataU[0:NL,:,ith1]
            data_cudaIP = np.sqrt(data_cudaQ*data_cudaQ + data_cudaU*data_cudaU)
            data_cudaPR = data_cudaIP/data_cudaI * 100
            data_cudaN = np.zeros((NL, NBPHI_cuda), dtype=float)
            data_cudaN = 100./ np.sqrt(dataN[0:NL,:,ith1])
            if len(args)==2:
                data_cudaI2 = np.zeros((NL,NBPHI_cuda), dtype=float)
                data_cudaI2 = dataI2[0:NL,:,ith1]
                data_cudaQ2 = np.zeros((NL, NBPHI_cuda), dtype=float)
                data_cudaQ2 = dataQ2[0:NL,:,ith1]
                data_cudaU2 = np.zeros((NL, NBPHI_cuda), dtype=float)
                data_cudaU2 = dataU2[0:NL,:,ith1]
                data_cudaIP2 = np.sqrt(data_cudaQ2*data_cudaQ2 + data_cudaU2*data_cudaU2)
                data_cudaPR2 = data_cudaIP2/data_cudaI2 * 100
                data_cudaN2  = np.zeros((NL, NBPHI_cuda), dtype=float)
                data_cudaN2  = 100./ np.sqrt(dataN2[0:NL,:,ith1])
            
        if options.list[0]==1:
            iphi1 = (np.abs(phi-options.list[1])).argmin() 
            data_cudaI = np.zeros((NL,NBTHETA_cuda), dtype=float)
            data_cudaI = dataI[0:NL,iphi1,:]
            data_cudaQ = np.zeros((NL, NBTHETA_cuda), dtype=float)
            data_cudaQ = dataQ[0:NL,iphi1,:]
            data_cudaU = np.zeros((NL, NBTHETA_cuda), dtype=float)
            data_cudaU = dataU[0:NL,iphi1,:]
            data_cudaIP = np.sqrt(data_cudaQ*data_cudaQ + data_cudaU*data_cudaU)
            data_cudaPR = data_cudaIP/data_cudaI * 100
            data_cudaN = np.zeros((NL, NBTHETA_cuda), dtype=float)
            data_cudaN = 100./ np.sqrt(dataN[0:NL,iphi1,:])
            if len(args)==2:
                data_cudaI2 = np.zeros((NL,NBTHETA_cuda), dtype=float)
                data_cudaI2 = dataI2[0:NL,iphi1,:]
                data_cudaQ2 = np.zeros((NL, NBTHETA_cuda), dtype=float)
                data_cudaQ2 = dataQ2[0:NL,iphi1,:]
                data_cudaU2 = np.zeros((NL, NBTHETA_cuda), dtype=float)
                data_cudaU2 = dataU2[0:NL,iphi1,:]
                data_cudaIP2 = np.sqrt(data_cudaQ2*data_cudaQ2 + data_cudaU2*data_cudaU2)
                data_cudaPR2 = data_cudaIP2/data_cudaI2 * 100
                data_cudaN2  = np.zeros((NL, NBPHI_cuda), dtype=float)
                data_cudaN2  = 100./ np.sqrt(dataN2[0:NL,iphi1,:])
            
        if options.list[0]==2:
            io1 = (np.abs(np.array(other)-options.list[1])).argmin() 
            data_cudaI = np.zeros((NBPHI_cuda,NBTHETA_cuda), dtype=float)
            data_cudaI = dataI[io1,:,:]
            data_cudaQ = np.zeros((NBPHI_cuda,NBTHETA_cuda), dtype=float)
            data_cudaQ = dataQ[io1,:,:]
            data_cudaU = np.zeros((NBPHI_cuda,NBTHETA_cuda), dtype=float)
            data_cudaU = dataU[io1,:,:]
            data_cudaIP = np.sqrt(data_cudaQ*data_cudaQ + data_cudaU*data_cudaU)
            data_cudaPR = data_cudaIP/data_cudaI * 100
            data_cudaN = np.zeros((NBPHI_cuda,NBTHETA_cuda), dtype=float)
            data_cudaN = 100./ np.sqrt(dataN[io1,:,:])
            if len(args)==2:
                data_cudaI2 = np.zeros((NBPHI_cuda,NBTHETA_cuda), dtype=float)
                data_cudaI2 = dataI2[io1,:,:]
                data_cudaQ2 = np.zeros((NBPHI_cuda,NBTHETA_cuda), dtype=float)
                data_cudaQ2 = dataQ2[io1,:,:]
                data_cudaU2 = np.zeros((NBPHI_cuda,NBTHETA_cuda), dtype=float)
                data_cudaU2 = dataU2[io1,:,:]
                data_cudaIP2 = np.sqrt(data_cudaQ2*data_cudaQ2 + data_cudaU2*data_cudaU2)
                data_cudaPR2 = data_cudaIP2/data_cudaI2 * 100
                data_cudaN2  = np.zeros((NL, NBPHI_cuda), dtype=float)
                data_cudaN2  = 100./ np.sqrt(dataN2[io1,:,:])
    else:  
        data_cudaI = np.zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)
        data_cudaI = dataI[0:NBPHI_cuda,:]
        data_cudaQ = np.zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)
        data_cudaQ = dataQ[0:NBPHI_cuda,:]
        data_cudaU = np.zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)
        data_cudaU = dataU[0:NBPHI_cuda,:]
        data_cudaIP = np.sqrt(data_cudaQ*data_cudaQ + data_cudaU*data_cudaU)
        data_cudaPR = data_cudaIP/data_cudaI * 100
        data_cudaN = np.zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)
        data_cudaN = 100./ np.sqrt(dataN[0:NBPHI_cuda,:])

        if len(args) == 2 :
          if NBPHI_cuda==NBPHI_cuda2 and NBTHETA_cuda==NBTHETA_cuda2 :
            data_cudaI2 = np.zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)
            data_cudaI2 = dataI2[0:NBPHI_cuda,:]
            data_cudaQ2 = np.zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)
            data_cudaQ2 = dataQ2[0:NBPHI_cuda,:]
            data_cudaU2 = np.zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)
            data_cudaU2 = dataU2[0:NBPHI_cuda,:]
            data_cudaIP2 = np.sqrt(data_cudaQ2*data_cudaQ2 + data_cudaU2*data_cudaU2)
            data_cudaPR2 = data_cudaIP2/data_cudaI2 * 100
          else:
            sys.stdout.write("Dimensions incompatibles entre "+path_cuda+"et " +path_cuda2 +"\n")
       

    #---------------------------------------------------------
    if options.rmax == None:
        max=0.1
    else:
        max=options.rmax

    if options.error == None:
        maxe=1.0
    else:
        maxe=options.error

    if options.percent == None:
        maxp=100.0
    else:
        maxp=options.percent

    #---------------------------------------------------------
    # Recuperation d eventuels points de simulation venant d un fichier txt (comme libradtran par exemple)
    ###
    # pour l'instant un transect seulement
    # format: phi theta I Q U
    # pas de difference possible
    ###
    #---------------------------------------------------------

    if (options.points != None) :
      fichier_txt = open(options.points, "r")
      data=np.loadtxt(fichier_txt,comments="#")
      #data=np.array(data)
      ang = data[:,0:2]
      I_txt = data[:,2].reshape((-1,1))
      Q_txt = data[:,3].reshape((-1,1))
      U_txt = data[:,4].reshape((-1,1)) 
      IP_txt = np.sqrt(Q_txt*Q_txt + U_txt*U_txt)
      PR_txt = IP_txt/I_txt * 100
      I_txt = np.concatenate((ang,I_txt),axis=1)
      Q_txt = np.concatenate((ang,Q_txt),axis=1)
      U_txt = np.concatenate((ang,U_txt),axis=1)
      IP_txt = np.concatenate((ang,IP_txt),axis=1)
      PR_txt = np.concatenate((ang,PR_txt),axis=1)

    ##########################################################
    ##              CREATION DES GRAPHIQUES 2D              ##
    ##########################################################

    #---------------------------------------------------------
    # Calcul pour l'ergonomie des graphiques 2D
    #---------------------------------------------------------
    if (len(args)==2) :
      VI = np.linspace(-max,max,50) # levels des contours
      VIt = np.linspace(-max,max,6) # ticks des color bars associees
    else:
      VI = np.linspace(0.,max,50) # levels des contours
      VIt = np.linspace(0.,max,6) # ticks des color bars associees
    VQ = np.linspace(-max,max,50)
    VQt = np.linspace(-max,max,5)
    VU = np.linspace(-max,max,50)
    VUt = np.linspace(-max,max,5)
    if (len(args)==2) :
      VIP = np.linspace(-max,max,50)
      VIPt = np.linspace(-max,max,6)
    else:
      VIP = np.linspace(0.,max,50)
      VIPt = np.linspace(0.,max,6)
    if (len(args)==2) :
      VPR = np.linspace(-maxp,maxp,50)
      VPRt = np.linspace(-maxp,maxp,6)
    else:
      VPR = np.linspace(0.,maxp,50)
      VPRt = np.linspace(0.,maxp,6)
    if (len(args)==2) :
      VN = np.linspace(-maxe,maxe,50)
      VNt = np.linspace(-maxe,maxe,6)
    else:
      VN = np.linspace(0.,maxe,50)
      VNt = np.linspace(0.,maxe,6)
        

    #---------------------------------------------------------
    #choix des tableaux a tracer
    #---------------------------------------------------------

    #---------------------------------------------------------
    #   Creation
    #---------------------------------------------------------

    
    if options.QU==True:
        fig = figure(1, figsize=(9, 9))
    else:
        fig = figure(1, figsize=(9, 4.5))
        
    fig.text(.5,.95, r"$\theta_{v}=%.2f$" %float(thetas),fontsize='14', ha='center')
    fig.text(.5,.85, "Geometry : %s" %MODE,fontsize='14', ha='center')
    
    # if a list of inputs files read SDS of remaining files
    # substitute theta to phi
    # other dimension replace theta, so scale other limits to match the interval 0,90 for plotting
    if options.list != None :
        other=np.array(other)
        
        if options.list[0]==0 :
            fig.text(.5,.90, r"$\theta_{s}=%.2f$" %theta[ith1],fontsize='14', ha='center')
            other = (other-options.others[1])/(options.others[2]-options.others[1])* 90.
        if options.list[0]==1 :
            fig.text(.5,.90, r"$\phi=%.2f$" %phi[iphi1],fontsize='14', ha='center')
            other = (other-options.others[1])/(options.others[2]-options.others[1])* 90.
        if options.list[0]==2 :
            if options.others[0]=='LAMBDA':     
                fig.text(.5,.90, r"$\lambda=%.2f nm$"%other[io1],fontsize='14', ha='center')
            else:
                fig.text(.5,.90, options.others[0]+"$=%.2f$"%other[io1],fontsize='14', ha='center')
            other = (other-options.others[1])/(options.others[2]-options.others[1])* 90.
            
    else:
        fig.text(.5,.90, r"$\lambda=%.2f nm$"%(float(LAMBDA)),fontsize='14', ha='center')
    fig.subplots_adjust(wspace=0.3, hspace=0.3, left=0.05, right=0.95)
    #fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)

    if options.QU==True:
        sub =  [423,424,427,428]
    else:
        sub =  [223,0,0,224]

    #r , t = np.meshgrid(theta,phi)

    if options.phi0 != None :
        if options.QU==True:
            rect = [421,422,425,426] 
        else:
            rect = [221,0,0,222]
        
        if options.list != None : 
            if options.list[0]==0 : 
                iphi0 = (np.abs(phi-options.phi0)).argmin()
                fig.text(.5,.80, r"$\phi^{0}=%.2f$"%phi[iphi0],fontsize='14', ha='center')
            if options.list[0]==1 : 
                iphi0 = (np.abs(theta-options.phi0)).argmin()
                fig.text(.5,.80, r"$\theta_{s}^{0}=%.2f$"%theta[iphi0],fontsize='14', ha='center')
            if options.list[0]==2 : 
                iphi0 = (np.abs(phi-options.phi0)).argmin()
                fig.text(.5,.80, r"$\phi^{0}=%.2f$"%phi[iphi0],fontsize='14', ha='center')
        else :
            iphi0 = (np.abs(phi-options.phi0)).argmin()

    else:
        if options.QU==True:
            rect = [221,222,223,224]
        else:
            rect = [121,0,0,122]
        iphi0 = -1
        
        
    if options.points == None :
        I_txt = None
        Q_txt = None
        U_txt = None
        IP_txt = None
        PR_txt = None
        
    if options.list == None :
        other = None
        
    if len(args)==1:
        data_cudaI2=None
        data_cudaQ2=None
        data_cudaU2=None
        data_cudaIP2=None
        data_cudaPR2=None
        data_cudaN2=None
        
        
    # first quarter I
        
    XX,YY,ZZ,TITLE,opt = setXYZ(theta,phi,other,options,data_cudaI,data_cudaI2,SYM='I')       
    lab=''
    if options.QU==False :
        lab='Reflectance'
       
    plot_2D_parameter(fig, rect[0], XX , YY, ZZ, VI, Vdatat=VIt,title=TITLE, iphi0=iphi0, sub=sub[0], options=opt, points=I_txt, label=lab) 
                

    if options.QU==True:
        # 2nd quarter Q
        XX,YY,ZZ,TITLE,opt = setXYZ(theta,phi,other,options,data_cudaQ,data_cudaQ2,SYM='Q')
        lab=''
        plot_2D_parameter(fig, rect[1], XX , YY, ZZ, VQ, Vdatat=VQt,title=TITLE, iphi0=iphi0, sub=sub[1], options=opt, points=Q_txt, label=lab)
            
    
        # 3rd quarter U
        XX,YY,ZZ,TITLE,opt = setXYZ(theta,phi,other,options,data_cudaU,data_cudaU2,SYM='U')
        lab='Reflectance'
        plot_2D_parameter(fig, rect[2], XX , YY, ZZ, VU, Vdatat=VUt,title=TITLE, iphi0=iphi0, sub=sub[2], options=opt, points=U_txt, label=lab)
        

    # 4th quarter
    # Polarization ratio   
    if (options.percent >= 0.) and (options.error == None):
        XX,YY,ZZ,TITLE,opt = setXYZ(theta,phi,other,options,data_cudaPR,data_cudaPR2,SYM='P')
        lab='Polarization Ratio'
        plot_2D_parameter(fig, rect[3], XX , YY, ZZ, VPR, Vdatat=VPRt,title=TITLE, iphi0=iphi0, sub=sub[3], options=opt, points=PR_txt, label=lab) 
        
#
#    # or Error
    if options.percent == None and options.error >= 0.:
        XX,YY,ZZ,TITLE,opt = setXYZ(theta,phi,other,options,data_cudaN,data_cudaN2,SYM=r"$\Delta$")
        lab='Relative Error'
        plot_2D_parameter(fig, rect[3], XX , YY, ZZ, VN, Vdatat=VNt,title=TITLE, iphi0=iphi0, sub=sub[3], options=opt, points=None, label=lab)
#
#    # or Polarized reflectance
    if options.percent == None  and options.error == None:
        XX,YY,ZZ,TITLE,opt = setXYZ(theta,phi,other,options,data_cudaIP,data_cudaIP2,SYM='IP')
        lab='Polarized Reflectance'
        plot_2D_parameter(fig, rect[3], XX , YY, ZZ, VIP, Vdatat=VIPt,title=TITLE, iphi0=iphi0, sub=sub[3], options=opt, points=IP_txt, label=lab)

    if options.filename == None:
        show()
    else:
        savefig(options.filename)


if __name__ == '__main__':
    main()
