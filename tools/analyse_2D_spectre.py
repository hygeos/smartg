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
import matplotlib
#matplotlib.use('Agg')
import numpy as np
np.seterr(invalid='ignore', divide='ignore') # ignore division by zero errors
#from pylab import savefig, show, figure
from pylab import *
from optparse import OptionParser
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
import matplotlib.gridspec as gridspec

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


    ######################################################
    ##                PARSE OPTIONS                     ##
    ######################################################

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
    parser.add_option('-n', '--nlambda',
            type='int',
            dest='NI',
            help='Number of wavelengths sample',
            )
    (options, args) = parser.parse_args()
    if len(args) != 1 and len(args) != 2:
        parser.print_usage()
        exit(1)

    path_cuda = args[0]

    if len(args) == 2 :
         path_cuda2 = args[1]


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

        # Récupération des valeurs de lamnda
        name = "Valeurs de Lambda"
        hdf_lam = sd_cuda.select(name)
        lam = hdf_lam.get()

        sds_cuda = sd_cuda.select("I_up (TOA)")
        dataI = sds_cuda.get()
        sds_cuda = sd_cuda.select("Q_up (TOA)")
        dataQ = sds_cuda.get()
        sds_cuda = sd_cuda.select("U_up (TOA)")
        dataU = sds_cuda.get()

        dataIP = np.sqrt(dataQ*dataQ + dataU*dataU)
        dataPR = dataIP/dataI * 100

    else:
        sys.stdout.write("Pas de fichier "+path_cuda+"\n")
        sys.exit()


    ##################################################################################
    ##              CREATION/CHOIX/MODIFICATION DE CERTAINES DONNES                 ##
    ##################################################################################

    # Sauvegarde de la grandeur désirée
    data_cudaI = np.zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)
    data_cudaI = dataI[0,0:NBPHI_cuda,:]
    data_cudaQ = np.zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)
    data_cudaQ = dataQ[0,0:NBPHI_cuda,:]
    data_cudaU = np.zeros((NBPHI_cuda, NBTHETA_cuda), dtype=float)
    data_cudaU = dataU[0,0:NBPHI_cuda,:]
    data_cudaIP = np.sqrt(data_cudaQ*data_cudaQ + data_cudaU*data_cudaU)
    data_cudaPR = data_cudaIP/data_cudaI * 100


    #---------------------------------------------------------
    if options.rmax == None:
        max=0.2
    else:
        max=options.rmax

    if options.NI == None:
        NI=3
    else:
        NI=options.NI

    if (NI >= 10) :
      fig = figure(1, figsize=(2*NI, 9))
      if (options.filename == None) :
         options.filename = 'analyse'
         print "Sortie dans 2 fichiers images : 'analyse_2D.png et analyse_spectre.png' car N >= 10\n"
    else :
      fig = figure(1, figsize=(18, 9))


    if options.error == None:
        maxe=1.0
    else:
        maxe=options.error

    if options.percent == None:
        maxp=100.0
    else:
        maxp=options.percent

    # Calcul pour l'ergonomie des graphiques
    VI = np.linspace(0.,max,50) # levels des contours
    VIt = np.linspace(0.,max,6) # ticks des color bars associees
    VQ = np.linspace(-max,max,50)
    VQt = np.linspace(-max,max,5)
    VU = np.linspace(-max,max,50)
    VUt = np.linspace(-max,max,5)
    VIP = np.linspace(0.,max,50)
    VIPt = np.linspace(0.,max,6)
    VPR = np.linspace(0.,maxp,50)
    VPRt = np.linspace(0.,maxp,6)

    ##########################################################
    ##  CREATION DES GRAPHIQUES 2D  : 1) polaire            ##
    ##########################################################

    # grille 2D des angles
    r , t = np.meshgrid(theta,phi)


    fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)
    NJ=3 # 3 parametres de stokes
    NLAM=lam.shape
    step=NLAM[0]/NI
    G = gridspec.GridSpec(NJ, NI)

    for i in range(NI):
     for j in range(NJ):
         s=j*NI+i+1
         l =  i*step
         title = r"$\lambda$ : %.3f nm" % lam[l]
         #axes = subplot(G[i, j])
         rect=NI*10+NJ*100+s

         if NI <= 3 :
             ax3, aux_ax3 = setup_axes3(fig, rect)
         else :
             ax3 = subplot(G[j, i])
         #cax3 = aux_ax3.contourf(t,r,data_cudaQ,VQ)
         #x3.set_title("Q",weight='bold',position=(0.25,1.0))
         #b3=fig.colorbar(cax3,orientation='horizontal',ticks=VQt)
         #cax3 = aux_ax3.contourf(t,r,data_cudaPR,VPR)
         #ax3.set_title("P",weight='bold',position=(0.25,1.0))
         #cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VPRt)
         if j==0 :
           if NI <= 3 :
             cax3 = aux_ax3.contourf(t,r,dataI[l,:,:],VI)
           else :
             cax3 = ax3.contourf(t,r,dataI[l,:,:],VI)
           ax3.set_title("I",weight='bold',position=(0.25,1.0))
           cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VIt)
           #cb3.set_label(r"$\rho$")

         if j==1 :
           if NI <= 3 :
             cax3 = aux_ax3.contourf(t,r,dataIP[l,:,:],VIP)
           else :
             cax3 = ax3.contourf(t,r,dataIP[l,:,:],VI)
           ax3.set_title("IP",weight='bold',position=(0.25,1.0))
           cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VQt)
           #cb3.set_label(r"$\rho_p$")

         if j==2 :
           if NI <= 3 :
             cax3 = aux_ax3.contourf(t,r,dataPR[l,:,:],VPR)
           else :
             cax3 = ax3.contourf(t,r,dataPR[l,:,:],VPR)
           ax3.set_title("P[%]",weight='bold',position=(0.25,1.0))
           cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VPRt)
           cb3.set_label(title)

         #ax3=subplot(G[i, j])
         #cax3 = ax3.contourf(t,r,dataI[s,:,:],VI)

         #ax3, aux_ax3 = setup_axes3(fig, 221)
         #cax3 = aux_ax3.contourf(t,r,data_cudaI,VI)
         #ax3.set_title("I",weight='bold',position=(0.25,1.0))
         #cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VIt)
         #cb3.set_label("Reflectance")


         #ax3, aux_ax3 = setup_axes3(fig, 222)
         #cax3 = aux_ax3.contourf(t,r,data_cudaQ,VQ)
         #ax3.set_title("Q",weight='bold',position=(0.25,1.0))
         #cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VQt)

         #ax3, aux_ax3 = setup_axes3(fig, 223)
         #cax3 = aux_ax3.contourf(t,r,data_cudaU,VU)
         #ax3.set_title("U",weight='bold',position=(0.25,1.0))
         #cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VUt)

         #ax3, aux_ax3 = setup_axes3(fig, 224)

         #if (options.percent >= 0.) and (options.error == None):
         #   cax3 = aux_ax3.contourf(t,r,data_cudaPR,VPR)
         #   ax3.set_title("P[%]",weight='bold',position=(0.25,1.0))
         #   cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VPRt)
         #   cb3.set_label("Polarization Ratio")
         #if options.percent == None and options.error >= 0.:
         #   cax3 = aux_ax3.contourf(t,r,data_cudaN,VN)
         #   ax3.set_title(r"$\Delta$ [%]",weight='bold',position=(0.25,1.0))
         #   cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VNt)
         #   cb3.set_label("Relative Error")
         #if options.percent == None  and options.error == None:
         #   cax3 = aux_ax3.contourf(t,r,data_cudaIP,VIP)
         #   ax3.set_title("IP",weight='bold',position=(0.25,1.0))
         #   cb3=fig.colorbar(cax3,orientation='horizontal',ticks=VIPt)
         #   cb3.set_label("Polarized Reflectance")

    if options.filename == None:
        show()
    else:
        savefig(options.filename + '_2D.png') 
    ##########################################################
    ##  CREATION DES GRAPHIQUES 2D  : 2) spectres           ##
    ##########################################################
    step = 3
    fig = figure(2, figsize=(10*NBTHETA_cuda/step, 10*NBPHI_cuda/step))
    fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)

    G = gridspec.GridSpec(NBTHETA_cuda/step, NBPHI_cuda/step)

    for i in range(NBTHETA_cuda/step):
     for j in range(NBPHI_cuda/step):
             ax3 = subplot(G[j, i])
             plot(lam,dataI[:,j*step,i*step],'r')
             #plot(lam,dataQ[:,j*step,i*step],'b')
             plot(lam,dataIP[:,j*step,i*step],'b')
             if options.rmax != None :
                 ylim([0,max])
             if i==(NBTHETA_cuda/step -1) :
                 ax3.set_title(r"$\phi$=%.2f"%phi[j*step],horizontalalignment='left')
             if j==(NBPHI_cuda/step -1) :
                 ax3.text(lam[0],max*0.95,r"$\theta_{v}$=%.2f"%theta[i*step],horizontalalignment='left')
                 #ax3.set_title(r"$\theta_{v}$=%.2f"%theta[i*step],horizontalalignment='right')

    if options.filename == None:
        show()
    else:
        savefig(options.filename + '_spectre.png')

    print 'Done.'


if __name__ == '__main__':
    main()
