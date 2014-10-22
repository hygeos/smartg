# encoding:utf-8
import subprocess
import sys
sys.path.append("profile")
import numpy as np
import pyhdf.SD
#from prosail import run_prosail
from profil import profil,readREPTRAN

####################################
install_dir = "/home/did/RTC/SMART-G/"
#afgl_name= install_dir + "tools/profile/afglms.dat"
#o3_name= install_dir + "tools/profile/crs_O3_UBremen_cf.dat"
output_dir = install_dir + "resultat/"
profil_name = output_dir + "profil.tmp"
#output_name= output_dir + "ENV-%s_%i_%.0f_thv-%s_AOT-%s_HA-%s_%s_X0%.1f.hdf"
#param_name= output_dir + "MyParametre.txt"
####################################
## Water model
#CHL= 1. # mg/l
#LSAOCE=72001 # nb d'angles de mla fonction de phase de sortie
#ANG_TRUNC=5. # angle de troncature de la fonction de phase
#
## Vegetation model
## Fixed in prosail, reflectance at nadir for thetas=45 deg, for LAI=2
#
## Aerosols
#
## Example for the simulation of MERIS bands
##
##geo_list=['sp','pp']
#abs_flag_list =["","--noabs"]
#albedo_list=[0.7]
#dioptre_list=[3]
#thv_list=[25]
#HA_list=[1.]
##X0_list=[-25.,-11.5,-10.5,-9.5,-8.5,-5,0.,5.,8.5,9.5,10.5,11.5,25.]
#X0_list=[0.,9.5]
#Y0=0.
#ENV_SIZE=10.
#ENV=1
##HA_list=[0.1,1.,3.]
#AOT_list=[0.2]
#geo_list=['sp']
##abs_flag_list =[""]
## lambda ref of MERIS as in Reference Model
#lam_ref=np.array([412.5,442.5,490.,510.,560.,620.,665.,681.25,708.75,753.75,761.875,778.75,865.,885.,900.])
#index_BOUVET_ATBD = np.array([1,2,3,4,5,6,7,8,12,13])
##lam_list=lam_ref[index_BOUVET_ATBD-1]
#lam_list=[600.]


#def runProfil(grid=grid,lat=lat,reptran=None,reptran_band=None, AOT=0., w=w, HA=HA):
#    return

class Job(object):
    def __init__(self,install_dir):
        self.install_dir=install_dir
        self.outlist=[]
        self.dict={'NBPHOTONS':1e7,'THVDEG':0.,'LAMBDA':443.,'TAURAY':0.23541,'TAUAER':0.,'W0AER':1.,\
                    'PATHDIFFAER':'/home/did/RTC/SMART-G/fic/pf_M80_443nm.txt','SIM':-2,'PROFIL':0,'PATHPROFILATM':'/home/did/RTC/SMART-G/profil/443nm_aot0.10540_O3mls_100layers.dat',\
                    'HA':1.,'HR':8.,'ZMIN':0.,'ZMAX':1.,'NATM':20,'HATM':100.,'SUR':1,'DIOPTRE':2,'W0LAM':0.,\
                    'WINDSPEED':5.,'NH2O':1.33,'ATOT':0.05,'BTOT':0.1,'PATHDIFFOCE':'/home/did/RTC/SMART-G/fic/pf_ocean_1.0_443nm.txt',\
                    'ENV':0,'ENV_SIZE':10.,'X0':0.,'Y0':0.,'SEED':-1,'NBTHETA':45,'NBPHI':45,'LSAAER':72001,\
                    'NFAER':1000000,'LSAOCE':72001,'NFOCE':10000000,'PATHRESULTATSHDF':'/home/did/RTC/SMART-G/resultat/out.hdf',\
                    'WRITE_PERIOD':-1,'OUTPUT_LAYERS':0,'XBLOCK':256,'YBLOCK':1,'XGRID':256,'YGRID':1,'NBLOOP':5000}
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
    
    def run(self,reptran=None):
        outlist=[]
        if reptran!=None: 
            outname='/home/did/RTC/SMART-G/resultat/out-'+reptran.filename+'-'+reptran.band_name
            for i in range(reptran.nband):
                fname='/home/did/RTC/SMART-G/resultat/out-'+reptran.filename+'-'+reptran.band_name+'-%iof%i'%(i+1,reptran.nband)
                self.setParams(LAMBDA=reptran.awvl[i],PATHPROFILATM=self.profilenamelist[i],PATHRESULTATSHDF=fname)            
                fo=open(output_dir+"MyParametre%i.txt"%i,"w")
                fo.write(self.str_Parametre.format(**self.dict))
                fo.close()
                cmd="/home/did/RTC/SMART-G/SMART-G " + output_dir+"MyParametre%i.txt"%i
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
            fo=open(output_dir+"MyParametre.txt","w") 
            fo.write(self.str_Parametre.format(**self.dict))
            fo.close()
            cmd="/home/did/RTC/SMART-G/SMART-G " + output_dir+"MyParametre.txt"
            subprocess.call(cmd,shell=True)
            outlist.append(self.dict["PATHRESULTATSHDF"])
        self.outlist=outlist
            
    def setProfile(self,profile):
        self.dict.update(NATM=profile.natm,HATM=profile.hatm,PROFIL=2)
        self.profilenamelist=profile.namelist
                
class Profile(object):  
    def __init__(self,install_dir,atmfile):
        self.install_dir=install_dir
        self.args=[self.install_dir+'tools/profile/'+ atmfile ,self.install_dir+'tools/profile/crs_O3_UBremen_cf.dat']
        self.options=Options()
        
    def run(self):
        self.natm,self.hatm,self.namelist = profil(self.options,self.args) 
      
class Options():
    def  __init__(self):
        self.dict={'aer':0.,'grid':None,'lat':45.,'w':550.,'noabs':False,'rep':None,'channel':None}
        self.aer=self.dict["aer"]
        self.grid=self.dict["grid"]
        self.lat=self.dict["lat"]
        self.w=self.dict["w"]
        self.noabs=self.dict["noabs"]
        self.rep=self.dict["rep"]
        self.channel=self.dict["channel"]
        
    def setOptions(self,**kwargs):
        self.dict.update(kwargs)
        self.aer=self.dict["aer"]
        self.grid=self.dict["grid"]
        self.lat=self.dict["lat"]
        self.w=self.dict["w"]
        self.noabs=self.dict["noabs"]
        self.rep=self.dict["rep"]
        self.channel=self.dict["channel"]


def main():   
#    reptran_filename='reptran_solar_envisat'
#    reptran_bandname='envisat_meris_ch09'
    reptran_filename='reptran_solar_msg'
    reptran_bandname='msg1_seviri_ch008'
    grid='100[25]25[5]10[1]0'
    atmfile='afglt.dat'
    myjob=Job(install_dir)
    myprofile=Profile(install_dir,atmfile)
    myprofile.options.setOptions(grid=grid,rep=reptran_filename,channel=reptran_bandname,aer=0.1)
    myprofile.run()
    myjob.setProfile(myprofile)
    myjob.setParams(SIM=2,THVDEG=60.,SUR=3,DIOPTRE=2,WINDSPEED=2.,W0LAM=0.3,ENV=1,ENV_SIZE=10000.,X0=9995.)
    myjob.setParams(ATOT=0.6,BTOT=1.8,PATHDIFFOCE='/home/did/RTC/SMART-G/fic/pf_ocean_1_700nm.txt')
    reptran=readREPTRAN(reptran_filename)
    Nint,Rint=reptran.selectBand(reptran.Bandname2Band(reptran_bandname))
 
    myjob.run(reptran=reptran)
    outname=install_dir+'resultat/out-'+ reptran_filename+'-'+reptran_bandname
    cmd='cp %s %s'%(myjob.outlist[0],outname)
    subprocess.call(cmd,shell=True)
    for i in range(len(myjob.outlist)):
        iband,w,weight,extra,crs_source=reptran.getwvl_param(i)
        sd=pyhdf.SD.SD(myjob.outlist[i])
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
        

if __name__ == '__main__':
    main()
#    fp=open(param_name,"w")
#    fp.write(str_Parametre.format(PROFIL_NAME=profil_name,OUTPUT_NAME=output_name%(abs_flag,dioptre_list[0],lam,thv_list[0],AOT_list[0],HA,geo_list[0],X0),THV=thv_list[0],DIOPTRE=dioptre_list[0],ALBEDO=albedo_list[0],LAMBDA=lam,TAUAER=AOT_list[0],HA=HA,ENV=ENV,ENV_SIZE=ENV_SIZE,Y0=Y0,X0=X0))
#    fp.close()


#for abs_flag in abs_flag_list:
# for HA in HA_list:
#  for X0 in X0_list:
#    for lam in lam_list:
#      cmd="python %s/tools/profile/profil.py %s -w %f -g '100[10]50[5]10[1]0' --lat 45 -A %f -H %f %s %s> %s" % (install_dir,abs_flag,lam,AOT_list[0],HA,afgl_name,o3_name,profil_name)
#      print cmd
#      b = subprocess.call(cmd,shell=True) 
#      fp=open(param_name,"w")
#      fp.write(str_Parametre.format(PROFIL_NAME=profil_name,OUTPUT_NAME=output_name%(abs_flag,dioptre_list[0],lam,thv_list[0],AOT_list[0],HA,geo_list[0],X0),THV=thv_list[0],DIOPTRE=dioptre_list[0],ALBEDO=albedo_list[0],LAMBDA=lam,TAUAER=AOT_list[0],HA=HA,ENV=ENV,ENV_SIZE=ENV_SIZE,Y0=Y0,X0=X0))
#      fp.close()
#      cmd="/home/did/RTC/SMART-G/SMART-G-" + geo_list[0] +" " + param_name
#      b = subprocess.call(cmd,shell=True)
