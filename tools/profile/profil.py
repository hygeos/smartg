import os
from string  import count,split
import numpy as np
from optparse import OptionParser
from scipy.interpolate import interp1d
from scipy.constants import codata
import netCDF4
from scipy.ndimage import map_coordinates

class readREPTRAN(object): # Read the channels reptran cdf file, either a list of sensor channels or a full solar coarse medium or fine grid
    def __init__(self,filename):
        self.filename=filename
        self._readFileGeneral()
        self.band=0
        self.species=['H2O','CO2','O3','N2O','CO','CH4','O2','N2']

    def _readFileGeneral(self):
        indir = '/home/did/RTC/REPTRAN/data/correlated_k/reptran/'
        nc=netCDF4.Dataset(indir+self.filename+'.cdf')
        self.wvl=nc.variables['wvl'][:] # the wavelength grid
        self.extra=nc.variables['extra'][:] # the extra terrestrial solar irradiance for the walength grid
        self.wvl_integral=nc.variables['wvl_integral'][:] # the wavelength integral (width) of each sensor channel
        self.nwvl_in_band=nc.variables['nwvl_in_band'][:] # the number of internal bands (representative bands) in each sensor channel
        self.band_names=nc.variables['band_name'][:] # the name of the sensor channel
        self.iwvl=nc.variables['iwvl'][:] # the indices of the internal bands within the wavelength grid for each sensor channel
        self.iwvl_weight=nc.variables['iwvl_weight'][:] # the weight associated to each internal band
        self.cross_section_source=nc.variables['cross_section_source'][:] # for each internal band, the list of species that participated to the absorption computation 
                                                                            # (1: abs, 0 : no abs)
    def selectBand(self,band): # select a particular sensor channel by its number (from 0)
        self.band=band
        self.nband=self.nwvl_in_band[self.band] # the number of internal bands (representative bands) in this channel
        self.iband=self.iwvl[:self.nband,self.band] # the indices of the internal bands within the wavelength grid for this channel
        self.awvl=self.wvl[self.iband-1] # the corresponsing wavelenghts of the internal bands
        self.awvl_weight=self.iwvl_weight[:self.nband,self.band] # the weights of the internal bands for this channel
        self.aextra=self.extra[self.iband-1] # the extra terrestrial solar irradiance of the internal bands for this channel
        self.across_section_source=self.cross_section_source[self.iband-1] # the source of absorption by species of the internal bands for this channel
        self.band_name=self.band_names[band,:].tostring().replace(" ","")
        return self.nband,self.wvl_integral[self.band]
        
    def Bandname2Band(self,bandname):
        l=[]
        for k in range(len(self.wvl_integral)):
            l.append(self.band_names[k,:].tostring().replace(" ",""))
        return l.index(bandname)
        

    def getwvl_param(self,i): # for a particular internal band, after selecting a sensor channel, get the idex number,wavelegnth,weight an extratreterial solar irradiance 
        return  self.iband[i],self.awvl[i], self.awvl_weight[i],self.aextra[i],self.across_section_source[i,:]

    def getabs_param(self,ispecies,i): #for a particular internal band, after selecting a sensor channel, and for a particular specie, get the absorption cross section LUT
        crs=readCRS(self.filename+'.lookup.'+self.species[ispecies],self.iband[i])
        return crs


class readCRS(object):
    def __init__(self,filename,iband):
        self.filename=filename
        self._readFileGeneral(iband)

    def _readFileGeneral(self,iband):
        indir = '/home/did/RTC/REPTRAN/data/correlated_k/reptran/'
        nc=netCDF4.Dataset(indir+self.filename+'.cdf')
        self.wvl_index=nc.variables['wvl_index'][:]
        ii=list(self.wvl_index).index(iband)
        dat=nc.variables['xsec'][:]
        self.xsec=dat[:,:,ii,:]
        self.pressure=nc.variables['pressure'][:]
        self.t_ref=nc.variables['t_ref'][:]
        self.t_pert=nc.variables['t_pert'][:]
        self.vmrs=nc.variables['vmrs'][:]

def interp3(x, y, z, v, xi, yi, zi, **kwargs):
    """Sample a 3D array "v" with pixel corner locations at "x","y","z" at the
    points in "xi", "yi", "zi" using linear interpolation. Additional kwargs
    are passed on to ``scipy.ndimage.map_coordinates``."""
    assert v.ndim==3
    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
    xi, yi, zi = np.atleast_1d(xi, yi, zi)
    for arr in [xi, yi, zi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y, z], [xi, yi, zi])]

    map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)

def interp2(x, y, v, xi, yi, **kwargs):
    """Sample a 2D array "v" with pixel corner locations at "x","y", at the
    points in "xi", "yi",  using linear interpolation. Additional kwargs
    are passed on to ``scipy.ndimage.map_coordinates``."""
    assert v.ndim==2
    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
    xi, yi = np.atleast_1d(xi, yi)
    for arr in [xi, yi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y], [xi, yi])]

    map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)

# Rayleigh Optical thickness for Bodhaine et al. 1999
# Ozone Chappuis band cross section data from University of Bremen
# Atmospheric profile readers and Gridd parsing routine adapted from the Py4CATS soffware package

def from_this_dir(filename):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

def isnumeric(x) :
    try : 
        float(x)
        return True
    except ValueError :
        return False

def rho(F) :
    return (6.*F-6)/(3+7.*F)

def FF(rho):
    return (6+3*rho)/(6-7*rho)

# gravity acceleration at the ground
# lat : deg

def g0(lat) : 
    return 980.6160 * (1. - 0.0026372 * np.cos(2*lat*np.pi/180.) + 0.0000059 * np.cos(2*lat*np.pi/180.)**2)

# gravity acceleration at altitude z
# lat : deg
# z : m
def g(lat,z) :
    return g0(lat) - (3.085462 * 1.e-4 + 2.27 * 1.e-7 * np.cos(2*lat*np.pi/180.)) * z \
            + (7.254 * 1e-11 + 1e-13 * np.cos(2*lat*np.pi/180.)) * z**2  \
            - (1.517 * 1e-17 + 6 * 1e-20 * np.cos(2*lat*np.pi/180.)) * z**3

# effective mass weighted altitude from US statndard
# z : m
def zc(z) :
    return 0.73737 * z + 5517.56

# depolarisation factor of N2
# lam : um
def FN2(lam) : 
    return 1.034 + 3.17 *1e-4 *lam**(-2)

# depolarisation factor of O2
# lam : um
def FO2(lam) : 
    return 1.096 + 1.385 *1e-3 *lam**(-2) + 1.448 *1e-4 *lam**(-4)

# depolarisation factor of air for 360 ppm CO2
def Fair360(lam) : 
    return (78.084 * FN2(lam) + 20.946 * FO2(lam) +0.934 + 0.036 *1.15)/(78.084+20.946+0.934+0.036)

def PR(theta,rho):
    gam=rho/(2-rho)
    return 1/4./np.pi * 3/(4*(1+2*gam)) * ((1-gam)*np.cos(theta*np.pi/180.)**2 + (1+3*gam))

# depolarisation factor of air for CO2
# lam : um
# co2 : ppm
def Fair(lam,co2) : 
    return (78.084 * FN2(lam) + 20.946 * FO2(lam) + 0.934 + co2*1e-4 *1.15)/(78.084+20.946+0.934+co2*1e-4)

# molecular volume
# co2 : ppm
def ma(co2):
    return 15.0556 * co2*1e-6 + 28.9595

# index of refraction of dry air  (300 ppm CO2)
# lam : um
def n300(lam):
    return 1e-8 * ( 8060.51 + 2480990/(132.274 - lam**(-2)) + 17455.7/(39.32957 - lam**(-2))) + 1.

# index of refraction odf dry air
# lam : um
# co2 : ppm
def n(lam,co2):
    return (n300(lam)-1) * (1 + 0.54*(co2*1e-6 - 0.0003)) + 1.

# Rayleigh cross section
# lam : um
# co2 : ppm
def raycrs(lam,co2):
    Avogadro = codata.value('Avogadro constant')
    Ns = Avogadro/22.4141 * 273.15/288.15 * 1e-3
    nn2 = n(lam,co2)**2
    return 24*np.pi**3 * (nn2-1)**2 /(lam*1e-4)**4/Ns**2/(nn2+2)**2 * Fair(lam,co2)

# Rayleigh optical depth
# lam : um
# co2 : ppm
# lat : deg
# z : m
# P : hPa
def rod(lam,co2,lat,z,P):
    Avogadro = codata.value('Avogadro constant')
    return raycrs(lam,co2) * P*1e3 * Avogadro/ma(co2) /g(lat,z)

####################################################################################################################################

def change_altitude_grid (zOld, gridSpec):
        """ Setup a new altitude grid and interpolate profiles to new grid. """
        zFirst, zLast =  zOld[0], zOld[-1]
        #specs = re.split ('[-\s:,]+',gridSpec)
        if count(gridSpec,'[')+count(gridSpec,']')==0:
            if count(gridSpec,',')==0:
                try:                deltaZ = float(gridSpec)
                except ValueError:  raise SystemExit, 'z grid spacing not a number!'
                # set up new altitude grid
                zNew = np.arange(zFirst, zLast+deltaZ, deltaZ)
            elif count(gridSpec,',')==1:
                try:                zLow,zHigh = map(float,split(gridSpec,','))
                except ValueError:  raise SystemExit, 'z grid spacing not a pair of floats!'
                # for new grid simply extract old grid points within given bounds (also include altitudes slightly outside)
                eps  = min( zOld[1:]-zOld[:-1] ) / 10.
                zNew = np.compress(np.logical_and(np.greater_equal(zOld,zLow-eps), np.less_equal(zOld,zHigh+eps)), zOld)
            elif count(gridSpec,',')==2:
                try:                zLow,zHigh,deltaZ = map(float,split(gridSpec,','))
                except ValueError:  raise SystemExit, 'z grid spacing not a triple of floats (zLow.zHigh,deltaZ)!'
                # set up new altitude grid
                zNew = np.arange(max(zLow,zFirst), min(zHigh,zLast)+deltaZ, deltaZ)
            elif count(gridSpec,',')>2:
                try:                zNew = np.array(map(float, split(gridSpec,',')))
                except ValueError:  raise SystemExit, 'z grid not a set of floats separated by commas!'
        elif count(gridSpec,'[')==count(gridSpec,']') > 0:
              zNew = parseGridSpec (gridSpec)
        if not zFirst <= zNew[0] < zNew[-1] <= zLast:
            pass 
            #raise SystemExit, '%s  %f %f  %s  %f %f' % ('ERROR: new zGrid', zNew[0],zNew[-1], ' outside old grid', zFirst, zLast)
        else:
               raise SystemExit, 'New altitude not specified correctly\n' + \
                     'either simply give altitude step size, a pair of lower,upper limits,  or "start(step)stop"!'
        return zNew

####################################################################################################################################
def parseGridSpec (gridSpec):
    """ Set up (altitude) grid specified in format 'start[step1]stop1[step2]stop' or similar. """
    # get indices of left and right brackets
    lp = [];  rp = []
    for i in xrange(len(gridSpec)):
        if   (gridSpec[i]=='['):  lp.append(i)
        elif (gridSpec[i]==']'):  rp.append(i)
        else:                     pass
    if len(lp)==len(rp):
        gridStart = [];  gridStop = [];  gridStep = []
        for i in xrange(len(lp)):
            if i>0:  start=rp[i-1]+1
            else:    start=0
            if i<len(lp)-1: stop=lp[i+1]
            else:           stop=len(gridSpec)

            try:
                gridStart.append(float(gridSpec[start:lp[i]]))
            except ValueError:
                print 'cannot parse grid start specification\nstring not a number!'
                raise SystemExit
            try:
                gridStep.append(float(gridSpec[lp[i]+1:rp[i]]))
            except ValueError:
                print 'cannot parse grid step specification\nstring not a number!'
                raise SystemExit
            try:
                gridStop.append(float(gridSpec[rp[i]+1:stop]))
            except ValueError:
                print 'cannot parse grid stop specification\nstring not a number!'
                raise SystemExit
            if i==0:
                if gridStop[0]<=gridStart[0]: newGrid = gridStart[0]+gridStop[0] - (np.arange(gridStop[0], gridStart[0]+gridStep[0], gridStep[0]))
                if gridStop[0]>=gridStart[0]: newGrid = np.arange(gridStart[0], gridStop[0]+gridStep[0], gridStep[0])
            if i>0:
                if gridStop[i]<=gridStart[i]: newGrid = np.concatenate((newGrid[:-1],gridStart[i]+gridStop[i]- (np.arange(gridStop[i], gridStart[i]+gridStep[i], gridStep[i]))))
                if gridStop[i]>=gridStart[i]: newGrid = np.concatenate((newGrid[:-1],np.arange(gridStart[i], gridStop[i]+gridStep[i], gridStep[i])))
    else:
        print 'cannot parse grid specification\nnumber of opening and closing braces differs!\nUse format start[step]stop'
        raise SystemExit
    # set up new altitude grid
    return newGrid



####################################################################################################################################

def profil(options,args):

    if options.lat==None :  lat=45.
    else : lat=options.lat

    # lecture du fichier atmosphere AFGL
    data=np.loadtxt(args[0],comments="#")
    z=data[:,0] # en km
    P=data[:,1] # en hPa
    T=data[:,2] # en K
    air=data[:,3] # Air density en cm-3
    o3=data[:,4] # Ozone density en cm-3
    o2=data[:,5] # O2 density en cm-3
    h2o=data[:,6] # H2O density en cm-3
    co2=data[:,7] # CO2 density en cm-3
    no2=data[:,8] # NO2 density en cm-3
    # lecture des fichiers US Standard atmosphere pour les autres gaz
    datach4=np.loadtxt('/home/did/RTC/SMART-G/tools/profile/afglus_ch4_vmr.dat',comments="#")
    ch4=datach4[:,1] * air # CH4 density en cm-3
    dataco=np.loadtxt('/home/did/RTC/SMART-G/tools/profile/afglus_co_vmr.dat',comments="#")
    co=dataco[:,1] * air # CO density en cm-3
    datan2o=np.loadtxt('/home/did/RTC/SMART-G/tools/profile/afglus_n2o_vmr.dat',comments="#")
    n2o=datan2o[:,1] * air # N2O density en cm-3
    datan2=np.loadtxt('/home/did/RTC/SMART-G/tools/profile/afglus_n2_vmr.dat',comments="#")
    n2=datan2[:,1] * air # N2 density en cm-3
    # lecture du fichier crs de l'ozone dans les bandes de Chappuis
    crs_chappuis=np.loadtxt(args[1],comments="#")

    #-------------------------------------------
    # optionnal regrid
    if options.grid !=None :
        znew= change_altitude_grid(z,options.grid)
        f=interp1d(z,P)
        P=f(znew)
        f=interp1d(z,T)
        T=f(znew)
        f=interp1d(z,air)
        airnew=f(znew)
        f=interp1d(z,o3)
        o3=f(znew)
        f=interp1d(z,o2)
        o2=f(znew)
        f=interp1d(z,h2o)
        h2o=f(znew)
        f=interp1d(z,co2)
        co2=f(znew)
        f=interp1d(z,no2)
        no2=f(znew)
        f=interp1d(z,ch4/air)
        ch4=f(znew)*airnew
        f=interp1d(z,co/air)
        co=f(znew)*airnew
        f=interp1d(z,n2o/air)
        n2o=f(znew)*airnew
        f=interp1d(z,n2/air)
        n2=f(znew)*airnew
        z=znew
        air=airnew
    #-------------------------------------------
    M=len(z) # Final number of layer
    datamol = np.zeros(M, np.float)
    datao3  = np.zeros(M, np.float)
    datamol  = np.zeros(M, np.float)
    dataray  = np.zeros(M, np.float)
    dataaer  = np.zeros(M, np.float)
    Ha = 1. # Hauteur d'echelle des aerosols en km
    xh2o=h2o/air # h2o vmr
    namelist=[] # list of profiles generated
    #-------------------------------------------
    if (options.rep!=None and  options.channel!=None):
        Nmol=8
        densmol  = np.zeros((M, Nmol), np.float)
        densmol[:,0]=h2o
        densmol[:,1]=co2
        densmol[:,2]=o3
        densmol[:,3]=n2o
        densmol[:,4]=co
        densmol[:,5]=ch4
        densmol[:,6]=o2
        densmol[:,7]=n2
        filename=options.rep
        reptran_bandname=options.channel
        reptran=readREPTRAN(filename)
        Nint,Rint = reptran.selectBand(reptran.Bandname2Band(reptran_bandname)) # selection d'une bande en particulier dans le fichier reptran, lecture du nombre de bandes internes et de l'integrale en nm de la bande
    
        for iint in np.arange(Nint): # pour chaque bande interne de la correlated_k (1 fichier de sortie par lambda)
            iband,w,weight,extra,crs_source=reptran.getwvl_param(iint)# on recupere les parametres de la bande: numero,lambda,poids,irradiance solaire, et tableau des gaz absorbants 
            datamol[:]=0.
            fname="/home/did/RTC/SMART-G/profil/pro-"+filename+'-'+reptran_bandname+'-%iof%i'%(iint+1,Nint)
            outfile=open(fname,'w')
            for ig in np.arange(Nmol): # pour chaque gaz 
                if crs_source[ig]==1: # si le gaz est absorbant a cette lambda
                    crs_mol=reptran.getabs_param(ig,iint) # on recupere la LUT d'absorption
                    f=interp1d(crs_mol.pressure,crs_mol.t_ref) #interpolation du profil vertical de temperature de reference dans les LUT
                    dT = T - f(P*100) # ecart en temperature par rapport au profil de reference (ou P de reference est en Pa et P AFGL en hPa)
                    if ig==0 :# si h2o
                        datamol += interp3(crs_mol.t_pert,crs_mol.vmrs,crs_mol.pressure,crs_mol.xsec,dT,xh2o,P*100) * densmol[:,ig] * 1e-11# interpolation dans la LUT d'absorption en fonction de
                                          # pression, ecart en temperature et vmr de h2o et mutiplication par la densite, calcul de reptran avec LUT en 10^(-20) m2, passage en km-1
                    else:
                        tab=crs_mol.xsec
                        datamol += interp2(crs_mol.t_pert,crs_mol.pressure,np.squeeze(tab),dT,P*100) * densmol[:,ig] * 1e-11  # interpolation dans la LUT d'absorption en fonction de
                                          # pression, ecart en temperature et mutiplication par la densite, calcul de reptran avec LUT en 10^(-20) m2, passage en km-1 
    
            outstr= "# I   ALT               hmol(I)         haer(I)         H(I)            XDEL(I)         YDEL(I)         percent_abs  LAM=  %7.2f nm, WEIGHT= %7.5f, E0=%9.3f, Rint=%8.3f\n" % (w,weight,extra,Rint)
            outfile.write(outstr)
            print "# I   ALT               hmol(I)         haer(I)         H(I)            XDEL(I)         YDEL(I)         percent_abs  abs LAM=  %7.2f nm, WEIGHT= %7.5f, E0=%9.3f, Rint=%8.3f" % (w,weight,extra,Rint)
            for m in range(M):
                #### Chappuis bands
                # SIGMA = (C0 + C1*(T-T0) + C2*(T-T0)^2) * 1.E-20 cm^2
                T0 = 273.15 #in K
                datao3[m] =  np.interp(w,crs_chappuis[:,0],crs_chappuis[:,1]) + np.interp(w,crs_chappuis[:,0],crs_chappuis[:,2])*(T[m]-T0) + \
                            np.interp(w,crs_chappuis[:,0],crs_chappuis[:,3])*(T[m]-T0)**2
                datao3[m] *= o3[m]* 1e-15 # calcul de Chapuis avec LUT en 10^(-20) cm2, passage en km-1 
                dataray[m] =  rod(w*1e-3,co2[m]/air[m]*1e6,lat,z[m]*1e3,P[m])
                dataaer[m] = options.aer * np.exp(-z[m]/Ha)
                if m==0 : 
                    dz=0.
                    hg=0.
                    taur_prec=0.
                    taua_prec=0.
                    st0= "%d\t" % m
                    st1= "%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t" % (0., 0., 0. , 0., 1., 0., 0.)
                    st2= "%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t" % (0., 0., 0. , 0., 1., 0.)
                    print ''.join(st0),'%7.2f\t' % z[m],''.join(st1)
                    outstr=  ''.join(st0)+'%7.2f\t' % z[m]+''.join(st2)+'\n'
                    outfile.write(outstr)
                else : 
                    dz = z[m-1]-z[m]
                    taur = dataray[m] - taur_prec
                    taur_prec = dataray[m]
                    taua = dataaer[m] - taua_prec
                    taua_prec = dataaer[m]
                    taug = datao3[m]*dz
                    taug += datamol[m]*dz
                    if options.noabs==True:
                        taug=0.
                    tau = taur+taua+taug
                    abs = taug/tau
                    xdel = taua/(tau*(1-abs))
                    ydel = taur/(tau*(1-abs))
                    hg += taug
                    htot = dataray[m]+dataaer[m]+hg
                    st0= "%d\t" % m
                    st1= "%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t" % (dataray[m], dataaer[m], htot , xdel, ydel, abs, taug)
                    st2= "%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t" % (dataray[m], dataaer[m], htot , xdel, ydel, abs)
                    print ''.join(st0),'%7.2f\t' % z[m],''.join(st1)
                    outstr= ''.join(st0)+'%7.2f\t' % z[m]+''.join(st2)+'\n'
                    outfile.write(outstr)
            namelist.append(fname)
                    #-------------------------------------------
    else : 
        fname="/home/did/RTC/SMART-G/profil/profil.tmp"
        outfile=open(fname,"w")
        w=options.w
        outstr= "# I   ALT               hmol(I)         haer(I)         H(I)            XDEL(I)         YDEL(I)         percent_abs  LAM=  %7.2f nm\n" % w
        outfile.write(outstr)
        print "# I   ALT               hmol(I)         haer(I)         H(I)            XDEL(I)         YDEL(I)         percent_abs  abs LAM=  %7.2f nm" % w
        for m in range(M):
            #### Chappuis bands
            # SIGMA = (C0 + C1*(T-T0) + C2*(T-T0)^2) * 1.E-20 cm^2
            T0 = 273.15 #in K
            datao3[m] =  np.interp(w,crs_chappuis[:,0],crs_chappuis[:,1]) + np.interp(w,crs_chappuis[:,0],crs_chappuis[:,2])*(T[m]-T0) + \
                        np.interp(w,crs_chappuis[:,0],crs_chappuis[:,3])*(T[m]-T0)**2
            datao3[m] *= o3[m]* 1e-15 # calcul de Chapuis avec LUT en 10^(-20) cm2, passage en km-1 
            dataray[m] =  rod(w*1e-3,co2[m]/air[m]*1e6,lat,z[m]*1e3,P[m])
            dataaer[m] = options.aer * np.exp(-z[m]/Ha)
            if m==0 : 
                dz=0.
                hg=0.
                taur_prec=0.
                taua_prec=0.
                st0= "%d\t" % m
                st1= "%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t" % (0., 0., 0. , 0., 1., 0., 0.)
                st2= "%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t" % (0., 0., 0. , 0., 1., 0.)
                print ''.join(st0),'%7.2f\t' % z[m],''.join(st1)
                outstr=  ''.join(st0)+'%7.2f\t' % z[m]+''.join(st2)+'\n'
                outfile.write(outstr)
            else : 
                dz = z[m-1]-z[m]
                taur = dataray[m] - taur_prec
                taur_prec = dataray[m]
                taua = dataaer[m] - taua_prec
                taua_prec = dataaer[m]
                taug = datao3[m]*dz
                taug += datamol[m]*dz
                if options.noabs==True:
                    taug=0.
                tau = taur+taua+taug
                abs = taug/tau
                xdel = taua/(tau*(1-abs))
                ydel = taur/(tau*(1-abs))
                hg += taug
                htot = dataray[m]+dataaer[m]+hg
                st0= "%d\t" % m
                st1= "%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t" % (dataray[m], dataaer[m], htot , xdel, ydel, abs, taug)
                st2= "%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t" % (dataray[m], dataaer[m], htot , xdel, ydel, abs)
                print ''.join(st0),'%7.2f\t' % z[m],''.join(st1)
                outstr= ''.join(st0)+'%7.2f\t' % z[m]+''.join(st2)+'\n'
                outfile.write(outstr)
        namelist.append(fname)
                #-------------------------------------------
    return M-1,z[0],namelist # return number of layers, altitute of TOA and list of profiles generated
    
if __name__ == '__main__':
    ####################################################################################################################################     
    parser = OptionParser(usage='%prog [options] file_in_atm file_in_crsO3\n Type %prog -h for help\n')
    parser.add_option('-n','--noabs',
                dest='noabs',
                action='store_true',
                default=False,
                help='no gaseous absorption'
                )
    parser.add_option('-w', '--wavel',
                dest='w',
                type='float',
                default=550.,
                help='wavelength (nm), default 550 nm' 
                )
    parser.add_option('-l', '--lat',
                dest='lat',
                type='float',
                help='latitude (deg)' 
                )
    parser.add_option('-g', '--grid',
                dest='grid',
                type='string',
                help='vertical grid format : start[step]Z1[step1].....[stepN]stop (km) with start>stop' 
                )
    parser.add_option('-A', '--AOT',
                dest='aer',
                type='float',
                default=0.,
                help='Aerosol Optical Thickness , default 0. (no aerosols)' 
                )
    parser.add_option('-R', '--REPTRAN',
                dest='rep',
                type='string',
                help='REPTRAN molecular absorption file' 
                )
    parser.add_option('-C', '--CHANNEL',
                dest='channel',
                type='string',
                help='Sensor channel name (use with REPTRAN)' 
                )
                
    (options, args) = parser.parse_args()
    if len(args) != 2 :
        parser.print_usage()
        exit(1)
    profil(options,args)
