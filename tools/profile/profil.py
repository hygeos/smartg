import os
from string  import count,split
import numpy as np
from optparse import OptionParser
from scipy.interpolate import interp1d
from scipy.constants import codata

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
    n2 = n(lam,co2)**2
    return 24*np.pi**3 * (n2-1)**2 /(lam*1e-4)**4/Ns**2/(n2+2)**2 * Fair(lam,co2)

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
            #if i==0:
            #    if gridStop[0]<=gridStart[0]: print 'incorrect grid specification:  Stop < Start'; raise SystemExit
            #    newGrid = np.arange(gridStart[0], gridStop[0]+gridStep[0], gridStep[0])
            #else:
            #    if gridStop[i]<=gridStart[i]: print 'incorrect grid specification:  Stop < Start'; raise SystemExit
            #    newGrid = np.concatenate((newGrid[:-1],np.arange(gridStart[i], gridStop[i]+gridStep[i], gridStep[i])))
    else:
        print 'cannot parse grid specification\nnumber of opening and closing braces differs!\nUse format start[step]stop'
        raise SystemExit
    # set up new altitude grid
    return newGrid

####################################################################################################################################
 
parser = OptionParser(usage='%prog [options] file_in_atm file_in_crsO3\n Type %prog -h for help\n')
parser.add_option('-n','--noabs',
            dest='noabs',
            action='store_true',
            default=False,
            help='no gaseous absorption'
            )
parser.add_option('-z',
            dest='zcol',
            type='int',
            help='number of the altitude column (starting from 1)'
            )
parser.add_option('-p',
            dest='pcol',
            type='int',
            help='number of the pressure column (starting from 1)'
            )
parser.add_option('-t',
            dest='tcol',
            type='int',
            help='number of the temperature column (starting from 1)'
            )
parser.add_option('-a',
            dest='acol',
            type='int',
            help='number of the air density column (starting from 1)'
            )
parser.add_option('-o',
            dest='ocol',
            type='int',
            help='number of ozone density colmun (starting from 1)' 
            )
parser.add_option('-c', 
            dest='ccol',
            type='int',
            help='number of co2 density colmun (starting from 1)' 
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
parser.add_option('-H', '--Ha',
            dest='Ha',
            type='float',
            default=1.,
            help='Aerosol Scale Height (km), default 1. km' 
            )
(options, args) = parser.parse_args()
if len(args) != 2 :
   parser.print_usage()
   exit(1)
if options.zcol==None :  ss=0
else : ss= options.zcol-1
if options.pcol==None :  sp=1
else : sp= options.pcol-1
if options.tcol==None :  st=2
else : st= options.tcol-1
if options.acol==None :  sa=3
else : sa= options.acol-1
if options.ocol==None : so=4
else : so= options.ocol-1
if options.ccol==None : sc=7
else : sc= options.ccol-1
w=options.w
if options.lat==None :  lat=45.
else : lat=options.lat

fi_atm = open(args[0], "r")
lignes=fi_atm.readlines()
data=np.loadtxt(args[0],comments="#")
####
# SIGMA = (C0 + C1*(T-T0) + C2*(T-T0)^2) * 1.E-20 cm^2
T0 = 273.15 #in K
crs=np.loadtxt(args[1],comments="#")


if ss<0 or so >=data.shape[1] :
   parser.print_usage()
   exit(1)

z=data[:,ss]
oz=data[:,so]
T=data[:,st]
P=data[:,sp]
co2=data[:,sc]
air=data[:,sa]
#-------------------------------------------
# optionnal regrid
if options.grid !=None :
  znew= change_altitude_grid(z,options.grid)
  f=interp1d(z,data[:,so])
  oz = f(znew)
  f=interp1d(z,data[:,st])
  T = f(znew)
  f=interp1d(z,data[:,sp])
  P = f(znew)
  f=interp1d(z,data[:,sc])
  co2 = f(znew)
  f=interp1d(z,data[:,sa])
  air = f(znew)
  z=znew
#-------------------------------------------

M=len(z)
dataoz  = np.zeros(M, np.float)
dataray  = np.zeros(M, np.float)
dataaer  = np.zeros(M, np.float)
Ha = 30.

print "# I   ALT               hmol(I)         haer(I)         H(I)            XDEL(I)         YDEL(I)         percent  abs O3   LAM=  %7.2f nm" % w
for m in range(M):
    if not options.noabs : 
        dataoz[m] =  np.interp(w,crs[:,0],crs[:,1]) + np.interp(w,crs[:,0],crs[:,2])*(T[m]-T0) + np.interp(w,crs[:,0],crs[:,3])*(T[m]-T0)**2
    dataoz[m] *= oz[m]* 1e-15 
    dataray[m] =  rod(w*1e-3,co2[m]/air[m]*1e6,lat,z[m]*1e3,P[m])
    dataaer[m] = options.aer * np.exp(-z[m]/Ha)
    #if options.os==True :
    if m==0 : 
        dz=0.
        ho=0.
        taur_prec=0.
        taua_prec=0.
        #st1=  "%11.5E %11.5E" %  (0.,0.)
        #print '%7.2f\t' % z[m],''.join(st1)
        st0= "%d\t" % m
        st1= "%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t" % (0., 0., 0. , 0., 1., 0.)
        print ''.join(st0),'%7.2f\t' % z[m],''.join(st1)
    else : 
        dz = z[m-1]-z[m]
        taur = dataray[m] - taur_prec
        taur_prec = dataray[m]
        taua = dataaer[m] - taua_prec
        taua_prec = dataaer[m]
        tauo = dataoz[m]*dz
        tau = taur+taua+tauo
        abs = tauo/tau
        xdel = taua/(tau*(1-abs))
        ydel = taur/(tau*(1-abs))
        ho += tauo
        htot = dataray[m]+dataaer[m]+ho
        st0= "%d\t" % m
        st1= "%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t%11.5E\t" % (dataray[m], dataaer[m], htot , xdel, ydel, abs)
        print ''.join(st0),'%7.2f\t' % z[m],''.join(st1)
