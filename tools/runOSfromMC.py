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
#matplotlib.use('Agg')
import numpy as np
np.seterr(invalid='ignore', divide='ignore') # ignore division by zero errors
from pylab import savefig, show, figure, plot, subplot
from optparse import OptionParser
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
from  mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
import subprocess

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

#----------------------------------------------------------------------------
# plot 2D 
#----------------------------------------------------------------------------
def plot_2D_parameter(fig, rect, theta , phi, data, Vdata, max, Vdatat=None, title=None, label=None, iphi0=-1, sub=None) :

    '''
    Contour and eventually transect of 2D parameter (theta,phi)
    fig : destiantion figure
    rect: string 'ABC' soecifiying position in the plot grid (see matplotlib multiple plots)
    theta: 1D vector for zenith angles
    phi: 1D vector for azimut angles
    data: 2D vector parameter
    Vdata: 1D parameter of iso-contour values
    max : maximum value of parameter for color scale
    Vdatat: keyword for optional ticks value of the colorbar (if not set: no colorbar added)
    tit : keyword of plot title (default no title)
    lab : keyword of colorbar title (default no label)
    iphi0: keyword of the value of the azimut angle for the transect plot (default : iphi0 < 0, no transect)
    sub : keyword for the position of the transect plot (similar type as rect)
    '''

    # grille 2D des angles
    r , t = np.meshgrid(theta,phi)
    NN = len(phi)
    ticks = np.array([-90,-75,-60,-30,0,30,60,75,90])

    ax3, aux_ax3 = setup_axes3(fig, rect)
    if iphi0 >= 0 : ax = subplot(sub)
    cax3 = aux_ax3.contourf(t,r,data,Vdata)
    if title != None : ax3.set_title(title,weight='bold',position=(0.15,0.9))
    if iphi0 >= 0 :
          vertex0 = np.array([[0,0],[phi[iphi0],90]])
          vertex1 = np.array([[0,0],[phi[NN-1-iphi0],90]])
          aux_ax3.plot(vertex0[:,0],vertex0[:,1],'w')
          aux_ax3.plot(vertex1[:,0],vertex1[:,1],'w--')
          ax.plot(theta,data[iphi0,:],'k-')
          ax.plot(-theta,data[NN-1-iphi0,:],'k--')
          if data.min() < 0  : 
              ax.set_ylim(-max,max)
          else :
              ax.set_ylim(0,max)
          ax.set_xlim(-90,90)
          ax.set_xticks(ticks)
          ax.grid=True

    if Vdatat != None :
        cb3=fig.colorbar(cax3,orientation='horizontal',ticks=Vdatat)
        if label != None  : cb3.set_label(label)

#----------------------------------------------------------------------------

def main():


    ######################################################
    ##                PARSE OPTIONS                     ##
    ######################################################

    parser = OptionParser(usage='%prog [options] hdf_file [hdf_file2]'
        + ' Polar Plots of MC output hdf file, eventually run and display SOS results\n'    
        + ' In case of a second hdf file, differences between MC file 1 and MC file 2 are plotted\n'
        + ' Options ...\n'
        + '-d --down : downward radiance at BOA (default upward TOA)\n'
        + '-S --ShowSOS : Plot SOS result (default MC hdf file)\n'
        + '-D --DiffSOS : Plot differences between MC and SOS result\n'
        + '-c --computeSOS : compute SOS result (default False: start from ./SOS_Up.txt and ./SOS_Down.txt files)\n'
        + '-a --aerosol : aerosol model (mandatory if -c is chosen), should be U80 or M80\n'
        + '-s --savefile : output graphics file name\n'
        + '-r --rmax : maximum reflectance for color scale\n'
        + '-p --percent : choose Polarization Ratio (default polarized reflectance) and set maximum PR for color scale\n'
        + '-t --transect : Add a transect below 2D plot for the closest azimuth to phi0 \n'
        + '-e --error : choose relative error instead of polarized reflectance and maximum error for color scale\n')

    parser.add_option('-d','--down',
            dest='down',
            action="store_true", default=False,
            help =  '-d downward radiance at BOA (default upward TOA)\n')
    parser.add_option('-S','--ShowSOS',
            dest='sos',
            action="store_true", default=False,
            help =  '-S Draw SOS result (default MC hdf file)\n')
    parser.add_option('-D','--DiffSOS',
            dest='diff',
            action="store_true", default=False,
            help =  '-D --DiffSOS : Plot differences between MC and SOS result\n')
    parser.add_option('-c','--compute',
            dest='compute',
            action="store_true", default=False,
            help =  '-c compute SOS result (default False start from ./SOS_Up.txt and ./SOS_Down.txt files)\n')
    parser.add_option('-s', '--savefile',
            dest='filename',
            help='-s output graphics file name\n'
            )
    parser.add_option('-a', '--aerosol',
            dest='aerosol',
            type='string',
            help='-a aerosol model (mandatory if -c is chosen), should be U80 or M80\n'
            )
    parser.add_option('-r', '--rmax',
            type='float',
            dest='rmax',
            help='-r maximum reflectance for color scale'
            )
    parser.add_option('-p', '--percent',
            dest='percent',
            type='float',
            help='-p choose Polarization Ratio (default polarized reflectance) and maximum PR for color scale (default 100)\n'
            )
    parser.add_option('-t', '--transect',
            dest='phi0',
            type='float',
            help = '-t --transect : Add a transect below 2D plot for the closest azimuth to phi0 \n'
            )
    parser.add_option('-e', '--error',
            dest='error',
            type='float',
            help='-e choose relative error instead of polarized reflectance and maximum error for color scale\n'
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
        TAURAY = getattr(sd_cuda,'TAURAY')
        TAUAER = getattr(sd_cuda,'TAUAER')
        WINDSPEED = getattr(sd_cuda,'WINDSPEED')
        W0LAM = getattr(sd_cuda,'W0LAM')
        HR = getattr(sd_cuda,'HR')
        LAMBDA = getattr(sd_cuda,'LAMBDA')
        NH2O = getattr(sd_cuda,'NH2O')
        SIM = getattr(sd_cuda,'SIM')

        # Récupération des valeurs de theta
        name = "Zenith angles"
        hdf_theta = sd_cuda.select(name)
        theta = hdf_theta.get()

        # Récupération des valeurs de phi
        name = "Azimut angles"
        hdf_phi = sd_cuda.select(name)
        phi = hdf_phi.get()
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



    ##################################################################################
    ##              CREATION/CHOIX/MODIFICATION DE CERTAINES DONNES                 ##
    ##################################################################################

    #---------------------------------------------------------
    # Sauvegarde de la grandeur désirée
    #---------------------------------------------------------
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

    if options.diff == True:
        options.sos = False
    ##########################################################
    ##              PARAMETRAGE ET RUN DES SOS              ##
    ##########################################################

    #---------------------------------------------------------
    # Parametres des SOS
    #---------------------------------------------------------
    # ecritue du fichier d'angle utilisateur pour OS V5.1
    fname = "./MC_angle_%i.txt" % NBTHETA_cuda
    fangle = open(fname,"w")
    for i in range(NBTHETA_cuda) :
        fangle.write("%9.5f\n" % theta[i])

    #------------------------------------
    # conversion en string pour passge au ksh
    LAMBDA = "%.3f" % (LAMBDA/1000.)
    Dphi_s = "%4i" % Dphi
    thetas = "%.2f" % thetas
    TAURAY = "%.5f" % TAURAY
    TAUAER = "%.5f" % TAUAER
    W0LAM = "%.2f" % W0LAM
    NH2O = "%.2f" % NH2O
    WINDSPEED = "%.1f" % WINDSPEED
    HR = "%.1f" % HR

    ### !! ###
    ZMIN='0'
    ZMAX='1'
    ### !! ###

    if SIM == -2 : # Black surface
       SURF_TYPE = '0'
       W0LAM = '0.00'
    else : 
       SURF_TYPE = '1'

    #b = subprocess.call("pwd",shell=True) 
    thisDir = "/home/did/RTC/MCCuda/tools"

    if options.compute == True : 
      if (options.aerosol != 'U80') & (options.aerosol != 'M80') : 
         parser.print_usage()
         exit(1)
      if options.aerosol == 'U80' :
         SOS_AER_MODEL = '2'
         SOS_AER_SF_MODEL = '2'
         SOS_AER_RH = '80.'
      if options.aerosol == 'M80' :
         SOS_AER_MODEL = '2'
         SOS_AER_SF_MODEL = '3'
         SOS_AER_RH = '80.'

      #---------------------------------------------------------
      # lancement des SOS
      #---------------------------------------------------------
      cmd = "$RACINE/exe/main_SOS.ksh -SOS.Wa "+LAMBDA+"	\
         -ANG.Rad.NbGauss 40    -ANG.Rad.ResFile "+thisDir+"/tmp_SOS_UsedAngles.txt \
         -ANG.Rad.UserAngFile "+thisDir+"/"+fname +"\
	   -ANG.Aer.NbGauss 40 \
         -ANG.Log "+thisDir+"/tmp_Angles.Log \
         -ANG.Aer.ResFile "+thisDir+"/tmp_AER_UsedAngles.txt \
         -ANG.Thetas "+thetas+" -SOS.View 2 -SOS.View.Dphi "+Dphi_s+"  \
         -SOS.IGmax 30 \
       -SOS.ResFileUp.UserAng "+thisDir+"/SOS_Up.txt  -SOS.ResFileDown.UserAng "+thisDir+"/SOS_Down.txt \
         -SOS.ResBin "+thisDir+"/tmp_SOS_Result.bin \
         -SOS.ResFileUp "+thisDir+"/tmp_Up.txt -SOS.ResFileDown "+thisDir+"/tmp_Down.txt -SOS.Log "+thisDir+"/tmp_SOS.log\
	   -SOS.Config "+thisDir+"/tmp_SOS_config.txt \
	   -SOS.Trans "+thisDir+"/tmp_SOS_transm.txt \
       -SOS.MDF 0.0279 \
	   -AP.MOT "+TAURAY+" -AP.HR "+HR+" \
         -AP.Type 2	\
         -AP.AerLayer.Zmin "+ZMIN+" -AP.AerLayer.Zmax "+ZMAX+" \
       -AP.ResFile "+thisDir+"/tmp_Profile.txt -AP.Log "+thisDir+"/tmp_Profile.Log \
	   -AER.Waref "+LAMBDA+" -AER.AOTref "+TAUAER+" \
	   -AER.ResFile "+thisDir+"/tmp_Aerosols.txt -AER.Log "+thisDir+"/tmp_Aerosols.Log -AER.MieLog 0 \
       -SURF.Log "+thisDir+"/tmp_Surface.Log -SURF.File DEFAULT \
         -SURF.Type "+SURF_TYPE+" -SURF.Alb "+W0LAM+" -SURF.Ind "+NH2O+" \
         -SURF.Glitter.Wind "+WINDSPEED+" \
	   -AER.Model "+SOS_AER_MODEL+" -AER.Tronca 0 \
         -AER.SF.Model "+SOS_AER_SF_MODEL+" -AER.SF.RH "+SOS_AER_RH

      b = subprocess.call("echo " + cmd + " > ./cmd",shell=True) 
      b = subprocess.call("cat ./runOS_template.ksh ./cmd > ./tmp.ksh",shell=True) 
      b = subprocess.call("chmod +x ./tmp.ksh",shell=True) 
      b = subprocess.call("./tmp.ksh",shell=True) 
      #b = subprocess.call("rm tmp* cmd",shell=True)

    #---------------------------------------------------------
    # Recuperation des SOS
    #---------------------------------------------------------

    I_sos = np.zeros((NBPHI_cuda,NBTHETA_cuda),dtype=float)
    Q_sos = np.zeros((NBPHI_cuda,NBTHETA_cuda),dtype=float)
    U_sos = np.zeros((NBPHI_cuda,NBTHETA_cuda),dtype=float)

    if options.down == False :
       fichier_sos = open(thisDir+"/SOS_Up.txt", "r")
    else :
       fichier_sos = open(thisDir+"/SOS_Down.txt", "r")

    data=np.loadtxt(fichier_sos,comments="#")
    I_sos = data[:,2]
    Q_sos = data[:,3]
    U_sos = data[:,4] 

    # Pour les OS le Dphi lance est 2 fois plus petit que pour Cuda afin d'avoir des valeurs de Phi qui coincident
    # De plus les 0S couvrent la gamme 0 - 360 et il nous faut uniquement une demi espace
    # NPhi(OS) = 4 * Nphi(MC) + 1
    phiAll = data[:,0].reshape(-1)
    OK = np.where(( (phiAll.astype(int) % int(2*Dphi)) == int(Dphi) )  & (phiAll < 180) )

    I_sos = I_sos[OK].reshape((-1,NBTHETA_cuda))  
    Q_sos = Q_sos[OK].reshape((-1,NBTHETA_cuda)) 
    U_sos = U_sos[OK].reshape((-1,NBTHETA_cuda))
    # retournement de Phi, convention OS
    I_sos = I_sos[::-1,:]
    Q_sos = Q_sos[::-1,:]
    U_sos = U_sos[::-1,:]

    ##########################################################
    ##              CREATION DES GRAPHIQUES 2D              ##
    ##########################################################

    #---------------------------------------------------------
    # Calcul pour l'ergonomie des graphiques 2D
    #---------------------------------------------------------
    if (len(args)==2) | (options.diff==True):
      VI = np.linspace(-max,max,50) # levels des contours
      VIt = np.linspace(-max,max,6) # ticks des color bars associees
    else:
      VI = np.linspace(0.,max,50) # levels des contours
      VIt = np.linspace(0.,max,6) # ticks des color bars associees
    VQ = np.linspace(-max,max,50)
    VQt = np.linspace(-max,max,5)
    VU = np.linspace(-max,max,50)
    VUt = np.linspace(-max,max,5)
    if (len(args)==2) | (options.diff==True):
      VIP = np.linspace(-max,max,50)
      VIPt = np.linspace(-max,max,6)
    else:
      VIP = np.linspace(0.,max,50)
      VIPt = np.linspace(0.,max,6)
    if (len(args)==2) | (options.diff==True):
      VPR = np.linspace(-maxp,maxp,50)
      VPRt = np.linspace(-maxp,maxp,6)
    else:
      VPR = np.linspace(0.,maxp,50)
      VPRt = np.linspace(0.,maxp,6)

    VN = np.linspace(0.,maxe,50)
    VNt = np.linspace(0.,maxe,6)

    #---------------------------------------------------------
    #choix des tableaux a tracer
    #---------------------------------------------------------
    if options.sos==True :
        data_cudaI = I_sos
        data_cudaQ = Q_sos
        data_cudaU = U_sos
        data_cudaIP = np.sqrt(data_cudaQ*data_cudaQ + data_cudaU*data_cudaU)
        data_cudaPR = data_cudaIP/data_cudaI * 100
    if options.diff==True :
        data_cudaI2 = I_sos
        data_cudaQ2 = Q_sos
        data_cudaU2 = U_sos
        data_cudaIP2 = np.sqrt(data_cudaQ2*data_cudaQ2 + data_cudaU2*data_cudaU2)
        data_cudaPR2 = data_cudaIP2/data_cudaI2 * 100


    #---------------------------------------------------------
    #   Creation
    #---------------------------------------------------------

    fig = figure(1, figsize=(9, 9))
    fig.subplots_adjust(wspace=0.3, hspace=0.3, left=0.05, right=0.95)
    #fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)

    sub =  [423,424,427,428]

    r , t = np.meshgrid(theta,phi)

    if options.phi0 != None :
        rect = [421,422,425,426]
        iphi0 = (np.abs(phi-options.phi0)).argmin()
    else:
        rect = [221,222,223,224]
        iphi0 = -1

    # first quarter
    if (len(args)==2) | (options.diff==True):
         plot_2D_parameter(fig, rect[0], theta , phi, data_cudaI-data_cudaI2, VI,  max, Vdatat=VIt,title='I1-I2', iphi0=iphi0, sub=sub[0])
    else:
         plot_2D_parameter(fig, rect[0], theta , phi, data_cudaI, VI,  max, Vdatat=VIt,title='I', iphi0=iphi0, sub=sub[0])


    # 2nd quarter
    if (len(args)==2) | (options.diff==True):
         plot_2D_parameter(fig, rect[1], theta , phi, data_cudaQ-data_cudaQ2, VQ,  max, Vdatat=VQt,title='Q1-Q2', iphi0=iphi0, sub=sub[1])
    else:
         plot_2D_parameter(fig, rect[1], theta , phi, data_cudaQ,  VQ, max, Vdatat=VQt,  title='Q', iphi0=iphi0, sub=sub[1])

    # 3rd quarter
    if (len(args)==2) | (options.diff==True):
         plot_2D_parameter(fig, rect[2], theta , phi, data_cudaU-data_cudaU2, VU,  max, Vdatat=VUt,title='U1-U2', label='Reflectance', iphi0=iphi0, sub=sub[2])
    else:
         plot_2D_parameter(fig, rect[2], theta , phi, data_cudaU,  VU, max, Vdatat=VUt,  title='U', label='Reflectance', iphi0=iphi0, sub=sub[2])

    # 4th quarter
    # Polarization ratio
    if (options.percent >= 0.) and (options.error == None):
      if (len(args)==2) | (options.diff==True):
         plot_2D_parameter(fig, rect[3], theta , phi, data_cudaPR-data_cudaPR2, VPR,  maxp, Vdatat=VPRt,title='P1-P2[%]', label='Polarization Ratio', iphi0=iphi0, sub=sub[3])
      else:
         plot_2D_parameter(fig, rect[3], theta , phi, data_cudaPR, VPR,  maxp, Vdatat=VPRt,title='P[%]', label='Polarization Ratio', iphi0=iphi0, sub=sub[3])

    # or Error
    if options.percent == None and options.error >= 0.:
         plot_2D_parameter(fig, rect[3], theta , phi, data_cudaN, VN,  maxe, Vdatat=VNt,title=r"$\Delta$ [%]", label='Relative Error', iphi0=iphi0, sub=sub[3])

    # or Polaried reflectance
    if options.percent == None  and options.error == None:
      if (len(args)==2) | (options.diff==True):
         plot_2D_parameter(fig, rect[3], theta , phi, data_cudaIP-data_cudaIP2, VIP,  max, Vdatat=VIPt,title='IP1-IP2', label='Polarized Reflectance', iphi0=iphi0, sub=sub[3])
      else:
         plot_2D_parameter(fig, rect[3], theta , phi, data_cudaIP, VIP,  max, Vdatat=VIPt,title='IP', label='Polarized Reflectance', iphi0=iphi0, sub=sub[3])


    if options.filename == None:
        show()
    else:
        savefig(options.filename)


if __name__ == '__main__':
    main()