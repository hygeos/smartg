# encoding:utf-8
import subprocess
import sys
sys.path.append("profile")
sys.path.append("water")
import numpy as np
import pyhdf.SD
#from prosail import run_prosail
from profil import profil,readREPTRAN
from water_spm_model import iop

####################################
install_dir = "/home/did/RTC/SMART-G/"
output_dir = install_dir + "resultat/"


class Job(object):
    def __init__(self,install_dir):
        self.install_dir=install_dir
        self.dict={'NBPHOTONS':1e7,'THVDEG':0.,'LAMBDA':443.,'TAURAY':0.23541,'TAUAER':0.,'W0AER':1.,\
                    'PATHDIFFAER':'/home/did/RTC/SMART-G/fic/pf_M80_443nm.txt','SIM':-2,'PROFIL':0,'PATHPROFILATM':'/home/did/RTC/SMART-G/profil/443nm_aot0.10540_O3mls_100layers.dat',\
                    'HA':1.,'HR':8.,'ZMIN':0.,'ZMAX':1.,'NATM':20,'HATM':100.,'SUR':1,'DIOPTRE':2,'W0LAM':0.,\
                    'WINDSPEED':5.,'NH2O':1.33,'ATOT':0.05,'BTOT':0.1,'PATHDIFFOCE':'/home/did/RTC/SMART-G/fic/pf_ocean_1.0_443nm.txt',\
                    'ENV':0,'ENV_SIZE':10.,'X0':0.,'Y0':0.,'SEED':-1,'NBTHETA':45,'NBPHI':45,'LSAAER':72001,\
                    'NFAER':1000000,'LSAOCE':72001,'NFOCE':10000000,'PATHRESULTATSHDF':'/home/did/RTC/SMART-G/resultat/out.hdf',\
                    'WRITE_PERIOD':-1,'OUTPUT_LAYERS':1,'XBLOCK':256,'YBLOCK':1,'XGRID':256,'YGRID':1,'NBLOOP':5000}
        self.atotlist=[self.dict["ATOT"]]
        self.btotlist=[self.dict["BTOT"]]
        self.iopnamelist=[self.dict["PATHDIFFOCE"]]
        self.str_Parametre="""
# Nombre de photons a lancer (unsigned long long) (limite par le tableau du poids des photons en unsigned long long)
NBPHOTONS = {NBPHOTONS}

# Angle zenithal de visée en degrés (float)
THVDEG = {THVDEG}

# Longueur d'onde [nm] (float)
LAMBDA = {LAMBDA}

# Epaisseur optique moleculaire (Rayleigh) (float)
TAURAY = {TAURAY}

# Epaisseur optique aerosol (float)
TAUAER = {TAUAER}

# Albedo simple de diffusion des aerosols (float)
W0AER = {W0AER}

# Nom absolu du fichier de la matrice de phase des aérosol
	# Données commencant sur la première ligne
	# %lf\t%lf\t%lf\t%lf\t%lf => Correspondant à angle	p2	p1	p3	p4
PATHDIFFAER = {PATHDIFFAER}

# Type de simulation
	# -2 pour atmosphere seule
	# -1 pour dioptre seul
	# 0 pour ocean et dioptre
	# 1 pour atmosphere et dioptre
	# 2 pour atmosphere dioptre et ocean
	# 3 pour ocean seul
SIM = {SIM}

# Type de profil atmospherique
	# 0 : profil standard avec echelle de hauteur
	# 1 : profil à 2 ou 3 couches (comme dans les SOS)
	# 2 : profil utilisateurs, Attention au bon format -> commencer les données sur la 2ème ligne à la couche 0
PROFIL = {PROFIL}

# Nom absolu du fichier du profil vertical atmosphérique utilisateur (utile uniquement si PROFIL = 2)
	# Le format du fichier doit être le suivant 
	# I	ALT		hmol(I)		haer(I)		H(I)		XDEL(I)		YDEL(I)  ZABS(I) => Première ligne indicative, pas de données dessus
	# %d\t%f\t%f\t%f\t%f\t%f\t%f\t%f  => Format pour toutes les autres lignes
PATHPROFILATM = {PATHPROFILATM}

# Echelle de hauteur aerosol [km] (utilise si PROFIL=0) (float)
HA = {HA}

# Echelle de hauteur moleculaire [km] (float)
HR = {HR}

# Altitude basse de la couche aerosol [km] (utilise si PROFIL=3) (float)
ZMIN = {ZMIN}

# Altitude haute de la couche aerosol [km] (utilise si PROFIL=3) (float)
ZMAX = {ZMAX}

# Nombre de couche+1
NATM={NATM}

# Altitude max de l'atmosphère
HATM={HATM}

# Type de reflexion de la surface
	# 1 pour reflexion forcee sur le dioptre
	# 2 transmission forcee
	# 3 reflexion et transmission
SUR = {SUR}

# Type de dioptre 
	# 0 = plan
	# 1 = agite avec reflexion multiple
	# 2 = agite sans reflexion multiple
	# 3 = surface lambertienne (uniquement sans océan)
	# 4 = glitter + surface lambertienne (2 en reflexion + 3 pour transmission) - Use SUR=3 in this case
DIOPTRE = {DIOPTRE}

# Albedo simple de diffusion de la surface lambertienne (float)
W0LAM = {W0LAM}

# Vitesse du vent [m/s] (utilise si idioptre=1) (modele de Cox et Munk) (float)
WINDSPEED = {WINDSPEED}

# Indice de refraction relatif air/eau (float)
NH2O = {NH2O}

# Coefficients d'absorption et de diffusion totaux de l'eau
# et fonction de phase totale pour la diffusion dans l'eau
ATOT = {ATOT}
BTOT = {BTOT}
PATHDIFFOCE = {PATHDIFFOCE}

# Effet d environnement
# 0 pas d effet
# 1 effet d environnement de type cible circulaire
# ENV_SIZE rayon de la cible en km
# X0 decalage en km entre la coordonnee X du centre de la cible et le point visee
# Y0 decalage en km entre la coordonnee Y du centre de la cible et le point visee
# !! l environnement est lambertien avec un albedo W0LAM
ENV= {ENV}
ENV_SIZE= {ENV_SIZE}
X0= {X0}
Y0= {Y0}

###
## Paramètres autres de la simulation ##
#_____________________________________#

# Graine avec laquelle on initialise les generateurs aleatoires (int)
# si SEED est positif on l'utilise comme graine (cela permet d'avoir les memes nombres aleatoires d'une simulation a l'autre, et
# donc une meme simulation)
# si SEED=-1 on crée une graine aleatoirement (donc toutes les simulations sont differentes les unes des autres)
SEED = {SEED}

# Le demi-espace dans lequel on récupère les photons est divisé en boites
# theta parcourt 0..PI
# phi parcourt 0..PI
NBTHETA = {NBTHETA}
NBPHI = {NBPHI}

# Nombre d'échantillons pour la matrice de phase des aérosols (int)
LSAAER = {LSAAER}

# 
NFAER = {NFAER}

#
LSAOCE = {LSAOCE}

#
NFOCE = {NFOCE} 

###
## Controle des sorties ##
#_______________________#        

# Chemin du fichiers de sortie et témoin
PATHRESULTATSHDF = {PATHRESULTATSHDF}

# Période d'écriture du fichier témoin en min (-1 pour désactiver)
WRITE_PERIOD = {WRITE_PERIOD}

# Output layers (binary flags)
# 1 -> BOA (0+) downward
OUTPUT_LAYERS    {OUTPUT_LAYERS}     

###
## Paramètres de la carte graphique  ##
#____________________________________#

# Les threads sont rangés dans des blocks de taille XBLOCK*YBLOCK (limite par watchdog il faut tuer X + limite XBLOCK*YBLOCK =< 256
#ou 512)
# XBLOCK doit être un multiple de 32
# à laisser à 256 (tests d'optimisation effectues)
XBLOCK = {XBLOCK} 
YBLOCK = {YBLOCK} 

# et les blocks sont eux-même rangés dans un grid de taille XGRID*YGRID (limite par watchdog il faut tuer X + limite XGRID<65535 et YGRID<65535)
XGRID = {XGRID} 
YGRID = {YGRID} 

# Nombre de boucles dans le kernel (unsigned int) (limite par watchdog, il faut tuer X)
# Ne pas mettre NBLOOP à une valeur trop importante, ce qui conduit à des erreurs dans le résultats de sorti
# Si les résultats manquent de lissage pour les valeurs importantes de réflectance, réduire ce chiffre, relancer et comparer
# Une valeur de 5000 semble satisfaisante
NBLOOP = {NBLOOP} 
        """
    
    def setParams(self,**kwargs):
        self.dict.update(kwargs)
        
    def run(self,options,reptran=None):
        outlist=[]
        if reptran!=None: 
            outname='/home/did/RTC/SMART-G/resultat/out-'+reptran.filename+'-'+reptran.band_name
            for i in range(reptran.nband):
                fname='/home/did/RTC/SMART-G/resultat/o-'+reptran.filename+'-'+reptran.band_name+'-%iof%i'%(i+1,reptran.nband)
                self.setParams(LAMBDA=reptran.awvl[i],PATHPROFILATM=self.profilenamelist[i],PATHRESULTATSHDF=fname)
                if self.dict["SIM"] in [0,2,3]: self.setParams(PATHDIFFOCE=self.iopnamelist[i],ATOT=self.atotlist[i],BTOT=self.btotlist[i])
                fo=open(output_dir+"MyParametre%i.txt"%i,"w")
                fo.write(self.str_Parametre.format(**self.dict))
                fo.close()
                cmd="/home/did/RTC/SMART-G/SMART-G-" + options.geo + " " + output_dir+"MyParametre%i.txt"%i
                subprocess.call(cmd,shell=True)
                outlist.append(fname)
                iband,w,weight,extra,crs_source=reptran.getwvl_param(i)
                sd=pyhdf.SD.SD(outlist[i])
                sds = sd.select("I_up (TOA)")
                dataI = sds.get()
                if i==0:
                    Isum=np.zeros_like(dataI)
                    Qsum=np.zeros_like(dataI)
                    Usum=np.zeros_like(dataI)
                    Nsum=np.zeros_like(dataI)
                    norm=0.
                sds = sd.select("Q_up (TOA)")
                dataQ = sds.get()
                sds = sd.select("U_up (TOA)")
                dataU = sds.get()
                sds = sd.select("Numbers of photons")
                dataN = sds.get()
                sd.end()
                Isum+=dataI*weight*extra
                Qsum+=dataQ*weight*extra
                Usum+=dataU*weight*extra
                Nsum+=dataN*weight*extra
                norm+=weight*extra
            Isum/=norm
            Qsum/=norm
            Usum/=norm
            Nsum/=norm
            cmd='cp %s %s'%(outlist[0],outname) # on duplique le 1er fichier de sortie hdf pour servir de base au fichier de sortie final
            subprocess.call(cmd,shell=True)
            sd = pyhdf.SD.SD(outname,pyhdf.SD.SDC.WRITE)
            sds = sd.select("I_up (TOA)")
            sds[:]=Isum
            sds.endaccess()
            sds = sd.select("Q_up (TOA)")
            sds[:]=Qsum
            sds.endaccess()
            sds = sd.select("U_up (TOA)")
            sds[:]=Usum
            sds.endaccess()
            sds = sd.select("Numbers of photons")
            sds[:]=Nsum
            sds.endaccess()

        else :
            if self.dict["PROFIL"]!=0 : 
                self.setParams(PATHPROFILATM=self.profilenamelist[0])
            self.setParams(PATHDIFFOCE=self.iopnamelist[0],ATOT=self.atotlist[0],BTOT=self.btotlist[0],LAMBDA=options.w)
            fo=open(output_dir+"MyParametre.txt","w") 
            fo.write(self.str_Parametre.format(**self.dict))
            fo.close()
            cmd="/home/did/RTC/SMART-G/SMART-G-" + options.geo + " " + output_dir+"MyParametre.txt"
            subprocess.call(cmd,shell=True)
            outname=self.dict["PATHRESULTATSHDF"]
        self.outname=outname
            
    def setProfile(self,profile):
        self.dict.update(NATM=profile.natm,HATM=profile.hatm,PROFIL=2,LSAAER=profile.NSCAER)
        self.profilenamelist=profile.namelist
    
    def setIop(self,iop):
        self.atotlist=iop.atotlist
        self.btotlist=iop.btotlist
        self.iopnamelist=iop.namelist
        self.dict.update(LSAOCE=iop.NSCOCE)
                
class Profile(object):  
    def __init__(self,install_dir,atmfile):
        self.install_dir=install_dir
        self.atmfile=atmfile
        self.args=[self.install_dir+'tools/profile/'+ atmfile+'.dat' ,self.install_dir+'tools/profile/crs_O3_UBremen_cf.dat']
        
    def run(self,options):
        self.natm,self.hatm,self.namelist = profil(options,self.args)
        self.NSCAER=options.NSCAER
 
class Iop(object):  
    def __init__(self,install_dir):
        self.install_dir=install_dir
        
    def run(self,options):
        self.atotlist,self.btotlist,self.namelist = iop(options)
        self.NSCOCE=options.NSCOCE
        
class Options():
    def  __init__(self):
        self.dict={'geo':'pp','aot':0.,'phase':False,'grid':None,'lat':45.,'w':550.,'noabs':False,'o3':None,'rep':None,'channel':None,\
                'SPM':1.,'NSCAER':72001,'NSCOCE':72001,'ang_trunc':5.,'gamma':0.5,'alpha':1.,'nbp':1.15,\
                'opac':None,'wref':550.,'Ha':1.}
        self.geo=self.dict["geo"]
        self.aot=self.dict["aot"]
        self.phase=self.dict["phase"]
        self.grid=self.dict["grid"]
        self.lat=self.dict["lat"]
        self.w=self.dict["w"]
        self.noabs=self.dict["noabs"]
        self.o3=self.dict["o3"]
        self.rep=self.dict["rep"]
        self.channel=self.dict["channel"]
        self.SPM=self.dict["SPM"]
        self.NSCAER=self.dict["NSCAER"]
        self.NSCOCE=self.dict["NSCOCE"]
        self.ang_trunc=self.dict["ang_trunc"]
        self.gamma=self.dict["gamma"]
        self.alpha=self.dict["alpha"]
        self.nbp=self.dict["nbp"]
        self.opac=self.dict["opac"]
        self.wref=self.dict["wref"]
        self.Ha=self.dict["Ha"]
        
    def setOptions(self,**kwargs):
        self.dict.update(kwargs)
        self.geo=self.dict["geo"]
        self.aot=self.dict["aot"]
        self.phase=self.dict["phase"]
        self.grid=self.dict["grid"]
        self.lat=self.dict["lat"]
        self.w=self.dict["w"]
        self.noabs=self.dict["noabs"]
        self.o3=self.dict["o3"]
        self.rep=self.dict["rep"]
        self.channel=self.dict["channel"]
        self.SPM=self.dict["SPM"]
        self.NSCAER=self.dict["NSCAER"]
        self.NSCOCE=self.dict["NSCOCE"]
        self.ang_trunc=self.dict["ang_trunc"]
        self.gamma=self.dict["gamma"]
        self.alpha=self.dict["alpha"]
        self.nbp=self.dict["nbp"]
        self.opac=self.dict["opac"]
        self.wref=self.dict["wref"]
        self.Ha=self.dict["Ha"]

def outname(job,profile,options):
    if job.dict["SIM"] in [-2,3]:
        strBASE='SIM%i'%job.dict["SIM"]
    else:
        strBASE='SIM%i_DI%i'%(job.dict["SIM"],job.dict["DIOPTRE"])
    if job.dict["ENV"]==0 or (job.dict["SIM"] not in [1,2]):
        strENV=''
    else:
        strENV='ENV%.1f-%i-X%.1f-Y%.1f'%(job.dict["W0LAM"],job.dict["ENV_SIZE"],job.dict["X0"],job.dict["Y0"])
    
    if options.rep!=None:
        strCHA=options.channel
    else:
        strCHA='%.2f'%options.w
    
    strVIEW='THV%.1f'%(job.dict["THVDEG"])
    strATM= profile.atmfile
    strAER=''
    if options.aot >0.:
        if options.opac!=None:
            strAER+=options.opac+'-'
        strAER+='AOT%.2f'%options.aot
    if job.dict["SIM"] in [0,2,3]:
        strOCE='SPM%.1f'%options.SPM
    else:
        strOCE=''
    strNOABS=''
    if options.noabs : strNOABS='noabs'
    strO3=''
    if options.o3 !=None : strO3='%.0fDU'%options.o3
    name=strBASE+'-'+strENV+'-'+strCHA+'-'+strVIEW+'-'+strATM+'-'+strAER+'-'+strOCE+'-'+options.geo+'-'+ strNOABS+'-'+strO3+'.hdf'
    return name     

def main():   


###########################################################################################################################################################
#   EXAMPLE 1
#    reptran_filename='reptran_solar_msg'
#    reptran_band_list=[('msg1_seviri_ch008',0.4)]
#    grid='100[75]25[5]10[1]0' # see profil.py help
#    atmfile='afglt'
#    aerosol_model='maritime_polluted'
#    output_dir=install_dir + 'resultat/'
#    lam_phase_index=0
#    layer_phase=13
#    
#    # Atmospheric Profile Preparation:
#    # starts with standard atmosphere name
#    myprofile=Profile(install_dir,atmfile)
#    # OPTIONS
#    myoptions=Options()
#    #---------
#    # Atmospheric Profile Preparation:
#    # Needs wavelength (monochromatic or Reptran parametrization), standard atmosphere name and eventually new vertical grid
#    # Needs aerosol AOT and eventually OPAC aerosol model name and reference wavelength for AOT, with optionnaly Phase MAtrix computation
#    #
#    # Example : We compute first scattering matrices (quite long) for further use: depends on atmopsheric profile wavelength and aerosol model
#    # we are using REPTRAN so read th correlated K parameters into the appropriate structure, 
#    # !! set aot>0 but its value has no importance for phase matrix computation
#    reptran=readREPTRAN(reptran_filename)
#    #---------    
#    for reptran_bandname,W0LAM in reptran_band_list:
#        reptran.selectBand(reptran.Bandname2Band(reptran_bandname)) 
#        lam_phase=reptran.awvl[lam_phase_index]
#        myoptions.setOptions(grid=grid,aot=0.2,opac=aerosol_model,phase=True,NSCAER=7201,rep=reptran_filename,channel=reptran_bandname)
#        # run the profile computation with selected options (see profil.py for help)
#        myprofile.run(myoptions)
#        
#    # Example : We then continue to set up the profile depending on AOT, including gaseous absorption,no more phase matrices computation
#    #---------          
#    for reptran_bandname,W0LAM in reptran_band_list:
#        reptran.selectBand(reptran.Bandname2Band(reptran_bandname)) 
#        myoptions.setOptions(geo='sp',wref=550.,aot=0.2,rep=reptran_filename,channel=reptran_bandname,noabs=True,phase=False) 
#        myprofile.run(myoptions)
#        
#        
#    # Optionnaly Introduce the IOp of ocean 
#    # Set the corresponding options (see water_spm_model.py for help)
#        myiop=Iop(install_dir)
#        myoptions.setOptions(SPM=1.,NSCOCE=72001)
#        myiop.run(myoptions)
#        
#     # Finally fir each time a simulation is launched, initiate a Job and set basic Parameters of the simulation 
#     # Don't forget to link the job with the vertical profil and ocean iop's
#        myjob=Job(install_dir)
#        myjob.setParams(SIM=2,THVDEG=65.,SUR=3,DIOPTRE=2,WINDSPEED=5.,NBPHOTONS=1e8,NBTHETA=30,NBPHI=30)
#        myjob.setProfile(myprofile)
#        myjob.setIop(myiop)
#     # You have to manually set the aerosol phase matrix to use for the job run (TODO : update smart-g for automatic set up)
#        myjob.setParams(PATHDIFFAER='/home/did/RTC/SMART-G/fic/pf_'+aerosol_model+'_%inm_layer-%i.txt'%(lam_phase,layer_phase)) 
#
#     # launch the job, !! you have to pass the reptran structure as a keyword if needed
#        myjob.run(myoptions,reptran=reptran)
#        
#        
#     # build the output filename depending on parametres and options and move it to the output directory 
#        cmd="mv %s %s"%(myjob.outname,output_dir+outname(myjob,myprofile,myoptions))
#        print '#--------------------------------------------------------------------------------------------------------#'
#        print cmd
#        print '#--------------------------------------------------------------------------------------------------------#'
#        subprocess.call(cmd,shell=True)
#        
#        flist=[]
#        for dist in np.linspace(-1.1,9.9,num=1):
#       # for dist in np.linspace(-9.9,9.9,num=10):
#            myjob.setParams(W0LAM=W0LAM,ENV=1,ENV_SIZE=10.,X0=dist)
#            myjob.run(myoptions,reptran=reptran)
#            oname=output_dir+outname(myjob,myprofile,myoptions) 
#            flist.append(oname)
#            cmd="mv %s %s"%(myjob.outname,oname)
#            print '#--------------------------------------------------------------------------------------------------------#'
#            print cmd
#            print '#--------------------------------------------------------------------------------------------------------#'
#            subprocess.call(cmd,shell=True)
#        fo=open("list.txt",'w')
#        for f in flist : fo.write('%s\n'%f)
#        fo.close()
###########################################################################################################################################################
        
###########################################################################################################################################################
#   EXAMPLE 2
#    grid='100[75]25[5]10[1]0' # see profil.py help
#    atmfile='afglms'
#    aerosol_model='desert_high'
#    output_dir=install_dir + 'resultat/'
##    lam_phase_index=0
#    lam_phase=550.
#    layer_phase=10
#    
#    myprofile=Profile(install_dir,atmfile)
#    myoptions=Options()
#    #---------    
#        
#    flist=[]
#    for aerosol_model in ['desert_high','desert']:
#        myoptions.setOptions(wref=550.,w=lam_phase,grid=grid,aot=1,opac=aerosol_model,phase=True,NSCAER=7201,geo='sp')
#        myprofile.run(myoptions)
#        
#        for noabs in [True,False]:
#            myoptions.setOptions(noabs=noabs,phase=False)
#            myprofile.run(myoptions)
#            #---------     
#            myjob=Job(install_dir)
#            myjob.setParams(SIM=1,THVDEG=65.,SUR=3,DIOPTRE=2,WINDSPEED=5.,\
#                        NBPHOTONS=1e8,NBTHETA=30,NBPHI=30)
#            myjob.setParams(PATHDIFFAER='/home/did/RTC/SMART-G/fic/pf_'+aerosol_model+'_%inm_layer-%i.txt'%(lam_phase,layer_phase)) 
#            myjob.setProfile(myprofile)  
#            
#        
#        #    for dist in np.linspace(-9.9,9.9,num=2):
#        #        myjob.setParams(ENV=1,ENV_SIZE=10.,X0=dist)
#            myjob.run(myoptions)
#            oname=output_dir+outname(myjob,myprofile,myoptions) 
#            flist.append(oname)
#            cmd="mv %s %s"%(myjob.outname,oname)
#            print '#--------------------------------------------------------------------------------------------------------#'
#            print cmd
#            print '#--------------------------------------------------------------------------------------------------------#'
#            subprocess.call(cmd,shell=True)
#    fo=open("list.txt",'w')
#    for f in flist : fo.write('%s\n'%f)
#    fo.close()
        

############################################################################################################################################################
##   EXAMPLE 3 : Just Rayleigh : kokhanovsky test case, modify DEPOL=0 in  SMART-G commun.h and recompile
#    grid='100[75]25[5]10[1]0' # see profil.py help
#    atmfile='afglt'
#    output_dir=install_dir + 'resultat/'
##    lam_phase_index=0
#    
#    myprofile=Profile(install_dir,atmfile)
#    myoptions=Options()
#    #---------    
#    myoptions.setOptions(w=409.45,aot=0,grid=grid,geo='pp',noabs=True,lat=45)
#    myprofile.run(myoptions)
#    #---------     
#    myjob=Job(install_dir)
#    myjob.setParams(SIM=-2,THVDEG=60.,NBPHOTONS=1e9,NBTHETA=180,NBPHI=180)
#    myjob.setProfile(myprofile)  
#
#    myjob.run(myoptions)
#    oname=output_dir+outname(myjob,myprofile,myoptions) 
#    cmd="mv %s %s"%(myjob.outname,oname)
#    print '#--------------------------------------------------------------------------------------------------------#'
#    print cmd
#    print '#--------------------------------------------------------------------------------------------------------#'
#    subprocess.call(cmd,shell=True)

###########################################################################################################################################################       

###########################################################################################################################################################
#   EXAMPLE 4 : Ozone Gaseous transmission for MERIS
#    reptran_filename='reptran_solar_envisat'
    reptran_filename='reptran_solar_sentinel'
    

    reptran_band_gen='sentinel3_olci_b'
#    reptran_band_list=['sentinel3_olci_b05']
    reptran_band_list=map(lambda x: reptran_band_gen+str(x).zfill(2),np.arange(20)+2)
#    reptran_band_gen='envisat_meris_ch'
#    reptran_band_list=map(lambda x: reptran_band_gen+str(x).zfill(2),np.arange(15)+1)
#    grid='100[25]50[2]20[1]0' # see profil.py help
    grid_simu=None # grille verticale de base pour la simu
    grid_phase='50[20]10[5]0'# grille verticale pour le calcul des fonctions de phase
    atmfile='afglsw'
    aerosol_model='maritime_polluted'
#    aerosol_model='mycloud'
    output_dir=install_dir + 'resultat/'
    lam_phase_index=0 # en cas de reptran la fonction de phase utilisee est celle du debut de la bande
    

    myprofile=Profile(install_dir,atmfile)
    # OPTIONS
    myoptions=Options()
    #---------

    reptran=readREPTRAN(reptran_filename)
    
    #---------  
    
    for DU in [300,0.]:
      for noabs in [False]:
        flist=[]
        i=0
        for reptran_bandname in reptran_band_list:
            reptran.selectBand(reptran.Bandname2Band(reptran_bandname)) 
            lam_phase=reptran.awvl[lam_phase_index] # longuer d onde  calcul de la focntion de phase pour chaque bande
            
            # on calcule les fonctions de phase sur une grille grid_phase (il faut metrre quand meme aot >0)
            if i==0 : myoptions.setOptions(grid=grid_phase,aot=.3,opac=aerosol_model,phase=True,NSCAER=7201,rep=reptran_filename,channel=reptran_bandname)
            myprofile.run(myoptions)  
            layer_phase=myprofile.natm -1 # on prend la fonction de phase de la deuxieme couche pres du sol (a modifier selon besoin)
            print 'couche  N: %i'%layer_phase
            
            # on repasse sur une grille fine et on calcule les proprietes optiques du profil, on fixe le reste des options
            myoptions.setOptions(geo='sp',wref=550,grid=grid_simu,aot=.3,opac=aerosol_model,phase=False,NSCAER=7201,rep=reptran_filename,channel=reptran_bandname,o3=DU,noabs=noabs)
            myprofile.run(myoptions)
           
            myjob=Job(install_dir)           
            ## simulation sol lambertien avec albedo de neige spectralement constant de 0.8
            # simulation glitter
            myjob.setParams(SIM=1,THVDEG=40.,SUR=3,DIOPTRE=2,W0LAM=0.8,NBPHOTONS=1e7,NBTHETA=30,NBPHI=30)
            myjob.setProfile(myprofile)
            # on finit par specifier la fonction de phase aerosol/nuage utilisee
            myjob.setParams(PATHDIFFAER='/home/did/RTC/SMART-G/fic/pf_'+aerosol_model+'_%inm_layer-%i.txt'%(lam_phase,layer_phase)) 
    
            myjob.run(myoptions,reptran=reptran)

            cmd="mv %s %s"%(myjob.outname,output_dir+outname(myjob,myprofile,myoptions))
            print '#--------------------------------------------------------------------------------------------------------#'
            print cmd
            print '#--------------------------------------------------------------------------------------------------------#'
            subprocess.call(cmd,shell=True)
            oname=output_dir+outname(myjob,myprofile,myoptions)
            flist.append(oname)
        i=i+1

        fo=open("list-%s-glint_maritime_%.0fDU_NOABS-%s.txt"%(reptran_filename,DU,noabs),'w')
        for f in flist : fo.write('%s\n'%f)
        fo.close()
###########################################################################################################################################################
         
       
if __name__ == '__main__':
    main()

