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
from optparse import OptionParser

######################################################
##                PARSE OPTIONS                     ##
######################################################
# NOTE: before importing matplotlib, to switch to agg backend if savefile is
# selected

parser = OptionParser(usage='%prog [options] hdf_file [hdf_file2]')
parser.add_option('-s', '--savefile',
        dest='filename',
        help='output file name',
        )
parser.add_option('-r', '--rmax',
        type='float',
        dest='rmax',
        help='maximum reflectance for color scale',
        )
parser.add_option('-p', '--percent',
        dest='percent',
        type='float',
        help='choose polarization ratio instead of polarized reflectance and maximum PR for color scale',
        )
parser.add_option('-e', '--error',
        dest='error',
        type='float',
        help='choose relative error instead of polarized reflectance and maximum error for color scale',
        )
(options, args) = parser.parse_args()
if len(args) != 1 and len(args) != 2:
    parser.print_usage()
    exit(1)

path_cuda = args[0]

if len(args) == 2 :
     path_cuda2 = args[1]

import matplotlib
if options.filename != None:
    matplotlib.use('Agg')
import numpy as np
np.seterr(invalid='ignore', divide='ignore') # ignore division by zero errors
from pylab import savefig, show, figure
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter

#----------------------------------------------------------------------------
# axes semi polaires
#----------------------------------------------------------------------------
def setup_axes3(fig, rect):
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

    angle_ticks2 = [(0, r"$0$"),
                   (30, r"$30$"),
                   (60, r"$60$"),
                   (90, r"$90$")]
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

    ax1.axis["left"].label.set_text(r"$\theta_{v}$")
    ax1.axis["top"].label.set_text(r"$\phi$")
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

def main():



    ##########################################################
    ##                DONNEES FICHIER CUDA                  ##
    ##########################################################

    # verification de l'existence du fichier hdf
    if os.path.exists(path_cuda):

        # lecture du fichier hdf
        sd_cuda = pyhdf.SD.SD(path_cuda)
        # lecture du nombre de valeurs de phi
        NBPHI_cuda = getattr(sd_cuda,'NBPHI')
        NBTHETA_cuda = getattr(sd_cuda,'NBTHETA')
        thetas = getattr(sd_cuda,'VZA (deg.)')

        # Récupération des valeurs de theta
        name = "Zenith angles"
        hdf_theta = sd_cuda.select(name)
        theta = hdf_theta.get()

        # Récupération des valeurs de phi
        name = "Azimut angles"
        hdf_phi = sd_cuda.select(name)
        phi = hdf_phi.get()

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



    ##################################################################################
    ##              CREATION/CHOIX/MODIFICATION DE CERTAINES DONNES                 ##
    ##################################################################################

    # Sauvegarde de la grandeur désirée
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
        data_cudaN2 = np.zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)
        data_cudaN2 = 100./ np.sqrt(dataN2[0:NBPHI_cuda,:])
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

    # Calcul pour l'ergonomie des graphiques
    if len(args) == 2:
      VI = np.linspace(-max,max,50) # levels des contours
      VIt = np.linspace(-max,max,6) # ticks des color bars associees
    else:
      VI = np.linspace(0.,max,50) # levels des contours
      VIt = np.linspace(0.,max,6) # ticks des color bars associees
    VQ = np.linspace(-max,max,50)
    VQt = np.linspace(-max,max,5)
    VU = np.linspace(-max,max,50)
    VUt = np.linspace(-max,max,5)
    if len(args) == 2:
      VIP = np.linspace(-max,max,50)
      VIPt = np.linspace(-max,max,6)
    else:
      VIP = np.linspace(0.,max,50)
      VIPt = np.linspace(0.,max,6)
    if len(args) == 2:
      VPR = np.linspace(-maxp,maxp,50)
      VPRt = np.linspace(-maxp,maxp,6)
    else:
      VPR = np.linspace(0.,maxp,50)
      VPRt = np.linspace(0.,maxp,6)

    VN = np.linspace(0.,maxe,50)
    VNt = np.linspace(0.,maxe,6)

    ##########################################################
    ##              CREATION DES GRAPHIQUES 2D              ##
    ##########################################################

    # grille 2D des angles
    r , t = np.meshgrid(theta,phi)


    fig = figure(1, figsize=(9, 9))
    fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)

    ax3, aux_ax3 = setup_axes3(fig, 221)
    if len(args)==2:
        cax3 = aux_ax3.contourf(t,r,data_cudaI-data_cudaI2,VI)
        ax3.set_title("I1-I2",weight='bold',position=(0.25,1.0))
    else:
        cax3 = aux_ax3.contourf(t,r,data_cudaI,VI)
        ax3.set_title("I",weight='bold',position=(0.25,1.0))
    cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VIt)
    cb3.set_label("Reflectance")

    ax3, aux_ax3 = setup_axes3(fig, 222)
    if len(args)==2:
        cax3 = aux_ax3.contourf(t,r,data_cudaQ-data_cudaQ2,VQ)
        ax3.set_title("Q1-Q2",weight='bold',position=(0.25,1.0))
    else:
        cax3 = aux_ax3.contourf(t,r,data_cudaQ,VQ)
        ax3.set_title("Q",weight='bold',position=(0.25,1.0))
    cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VQt)
    #cb3.set_label("Reflectance")

    ax3, aux_ax3 = setup_axes3(fig, 223)
    if len(args)==2:
        cax3 = aux_ax3.contourf(t,r,data_cudaU-data_cudaU2,VU)
        ax3.set_title("U1-U2",weight='bold',position=(0.25,1.0))
    else:
        cax3 = aux_ax3.contourf(t,r,data_cudaU,VU)
        ax3.set_title("U",weight='bold',position=(0.25,1.0))
    cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VUt)
    #cb3.set_label("Reflectance")

    ax3, aux_ax3 = setup_axes3(fig, 224)

    if (options.percent >= 0.) and (options.error == None):
      if len(args)==2:
        cax3 = aux_ax3.contourf(t,r,data_cudaPR-data_cudaPR2,VPR)
        ax3.set_title("P1-P2[%]",weight='bold',position=(0.25,1.0))
      else:
        cax3 = aux_ax3.contourf(t,r,data_cudaPR,VPR)
        ax3.set_title("P[%]",weight='bold',position=(0.25,1.0))
      cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VPRt)
      cb3.set_label("Polarization Ratio")
    if options.percent == None and options.error >= 0.:
      cax3 = aux_ax3.contourf(t,r,data_cudaN,VN)
      ax3.set_title(r"$\Delta$ [%]",weight='bold',position=(0.25,1.0))
      cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VNt)
      cb3.set_label("Relative Error")
    if options.percent == None  and options.error == None:
      if len(args)==2:
        cax3 = aux_ax3.contourf(t,r,data_cudaIP-data_cudaIP2,VIP)
        ax3.set_title("IP1-IP2",weight='bold',position=(0.25,1.0))
      else:
        cax3 = aux_ax3.contourf(t,r,data_cudaIP,VIP)
        ax3.set_title("IP",weight='bold',position=(0.25,1.0))
      cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VIPt)
      cb3.set_label("Polarized Reflectance")


    if options.filename == None:
        show()
    else:
        savefig(options.filename)


if __name__ == '__main__':
    main()
