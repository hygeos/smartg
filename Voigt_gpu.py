import sys
sys.path.insert(0, '/home/did/RTC/HITRAN/')
import hapi
import numpy as np

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from warnings import warn


mod = SourceModule("""
  
        __global__ void voigt_hui_gpu(int n, double *vgt, double *v, double *v0, double *gammaL, double *gammaD, double *S, double *a6, double *b6)
                {
              const int i = threadIdx.x + blockDim.x * blockIdx.x;
              const double sqrtLn2=0.832554611158;
              const double sqrtLn2overPi=0.46971863935;
              double x,y,K,xx,pRe,pIm,qRe,qIm,q2inv;
              double cr0,cr2,cr4,cr6,dr0,dr2,dr4,dr6;
              double ci1,ci3,ci5,di1,di3,di5;
              int j;
              vgt[i]=0.;
              for (j=0;j<n;j++) {
	               y = sqrtLn2 * gammaL[j] / gammaD[j];
	               cr0 = (((((a6[6]*y+a6[5])*y+a6[4])*y+a6[3])*y+a6[2])*y+a6[1])*y+a6[0];
	               cr2 = -((((15.*a6[6]*y +10.*a6[5])*y +6.*a6[4])*y +3.*a6[3])*y +a6[2]);
	               cr4 = (15.*a6[6]*y +5.*a6[5])*y +a6[4];
	               cr6 = -a6[6];
	               ci1 = -(((((6.*a6[6]*y +5.*a6[5])*y +4.*a6[4])*y +3.*a6[3])*y +2.*a6[2])*y+a6[1]);
	               ci3 = ((20.*a6[6]*y +10.*a6[5])*y +4.*a6[4])*y +a6[3];
	               ci5 = -(6.*y*a6[6] +a6[5]);
	               dr0 = ((((((y+b6[6])*y +b6[5])*y +b6[4])*y +b6[3])*y +b6[2])*y +b6[1])*y +b6[0];
	               dr2 = -(((((21.*y +15.*b6[6])*y +10.*b6[5])*y +6.*b6[4])*y +3.*b6[3])*y+b6[2]);
	               dr4 = ((35.*y +15.*b6[6])*y +5.*b6[5])*y +b6[4];
	               dr6 = -(7.*y +b6[6]);
	               di1 = -((((((7.*y+6.*b6[6])*y+5.*b6[5])*y+4.*b6[4])*y+3.*b6[3])*y+2.*b6[2])*y+b6[1]);
	               di3 = (((35.*y +20.*b6[6])*y +10.*b6[5])*y +4.*b6[4])*y +b6[3];
	               di5 = -((21.*y +6.*b6[6])*y +b6[5]);
	               x = sqrtLn2 * (v[i]-v0[j])/ gammaD[j];
	               xx = x*x;
                     pRe = ((cr6*xx+cr4)*xx+cr2)*xx+cr0;
                     pIm = ((ci5*xx+ci3)*xx+ci1)*x;
                     qRe = ((dr6*xx+dr4)*xx+dr2)*xx+dr0;
                     qIm = (((xx+di5)*xx+di3)*xx+di1)*x;
                     q2inv = 1./(qRe*qRe + qIm*qIm);
                     K = q2inv * (pRe*qRe + pIm*qIm);
                    /* printf("%d %d %f\n",threadIdx.x + blockDim.x * blockIdx.x, j,K);*/
                     vgt[i] = vgt[i] + (sqrtLn2overPi/gammaD[j])*S[j]*K;
                }
              }
        """)

mod2 = SourceModule("""
        #include <pycuda-complex.hpp>
        __global__ void voigt_hum1wei24_gpu(int n, float *vgt, float *v, float *v0, float *gammaL, float *gammaD, float *S, float *wB, float *NA, float *A)
                {
              
              const int i = threadIdx.x + blockDim.x * blockIdx.x;
              float K,x,y,norm;
              const float sqrtLn2=0.832554611158;
              const float recSqrtPi=0.564189583548;
              const float sqrtLn2overPi=0.46971863935;
              const float one=1.,halfone=0.5,two=2.;
              const float wL24=4.11953428781; /* sqrt(24)/2**(0.25) */
              pycuda::complex<float> r;
              pycuda::complex<float> t,iz;
              pycuda::complex<float> w;
              pycuda::complex<float> lpiz;
              pycuda::complex<float> lmiz;
              pycuda::complex<float> recLmiZ;
              pycuda::complex<float> jj(0.,1.);
              int j;
              vgt[i]=0.;
              for (j=0;j<n;j++) {
                    x = sqrtLn2 * (v[i]-v0[j])/ gammaD[j];
	              y = sqrtLn2 * gammaL[j] / gammaD[j];
                    norm = abs(x)+y;
	              t   = y - jj*x ;
	              iz=-t;  
	             if (y<15.0 && norm<15.0) {
		               lpiz = wL24 + iz;  
		               lmiz = wL24 - iz; 
		               recLmiZ  = one / lmiz ;
		               r       = lpiz * recLmiZ ;
		               w = recLmiZ * (recSqrtPi + two*recLmiZ*(wB[24]+(wB[23]+(wB[22]+(wB[21]+(wB[20]+ \
                              (wB[19]+(wB[18]+(wB[17]+(wB[16]+(wB[15]+(wB[14]+(wB[13]+(wB[12]+(wB[11]+(wB[10]+ \
                              (wB[9]+(wB[8]+(wB[7]+(wB[6]+(wB[5]+(wB[4]+(wB[3]+(wB[2]+wB[1]*r) \
                              *r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r));
                          }
                    else {
                           w = t * recSqrtPi / (halfone + t*t);
                         } 
               
	           K=w.real();
                 vgt[i] = vgt[i] + (sqrtLn2overPi/gammaD[j])*S[j]*NA[j]*A[j]*K;
                
                }
               }
        """)
        
mod3 = SourceModule("""
        #include <pycuda-complex.hpp>
        __global__ void voigt_hum1wei24_gpu64(int n, double *vgt, double *v, double *v0, double *gammaL, double *gammaD, double *S, double *wB, double *NA, double *A)
                {
              
              const int i = threadIdx.x + blockDim.x * blockIdx.x;
              double K,x,y,norm;
              const double sqrtLn2=0.832554611158;
              const double recSqrtPi=0.564189583548;
              const double sqrtLn2overPi=0.46971863935;
              const double one=1.,halfone=0.5,two=2.;
              const double wL24=4.11953428781; /* sqrt(24)/2**(0.25) */
              pycuda::complex<double> r;
              pycuda::complex<double> t,iz;
              pycuda::complex<double> w;
              pycuda::complex<double> lpiz;
              pycuda::complex<double> lmiz;
              pycuda::complex<double> recLmiZ;
              pycuda::complex<double> jj(0.,1.);
              int j;
              vgt[i]=0.;
              for (j=0;j<n;j++) {
                    x = sqrtLn2 * (v[i]-v0[j])/ gammaD[j];
	              y = sqrtLn2 * gammaL[j] / gammaD[j];
                    norm = abs(x)+y;
	              t   = y - jj*x ;
	              iz=-t;  
	             if (y<15.0 && norm<15.0) {
		               lpiz = wL24 + iz;  
		               lmiz = wL24 - iz; 
		               recLmiZ  = one / lmiz ;
		               r       = lpiz * recLmiZ ;
		               w = recLmiZ * (recSqrtPi + two*recLmiZ*(wB[24]+(wB[23]+(wB[22]+(wB[21]+(wB[20]+ \
                              (wB[19]+(wB[18]+(wB[17]+(wB[16]+(wB[15]+(wB[14]+(wB[13]+(wB[12]+(wB[11]+(wB[10]+ \
                              (wB[9]+(wB[8]+(wB[7]+(wB[6]+(wB[5]+(wB[4]+(wB[3]+(wB[2]+wB[1]*r) \
                              *r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r)*r));
                          }
                    else {
                           w = t * recSqrtPi / (halfone + t*t);
                         } 
               
	           K=w.real();
                 vgt[i] = vgt[i] + (sqrtLn2overPi/gammaD[j])*S[j]*NA[j]/A[j]*K;
                
                }
               }
        """)


def absorptionCoefficient_Voigt_gpu(Components=None,SourceTables=None,partitionFunction=hapi.PYTIPS,
                                Environment=None,OmegaRange=None,OmegaStep=None,OmegaWing=None,
                                IntensityThreshold=hapi.DefaultIntensityThreshold,
                                OmegaWingHW=hapi.DefaultOmegaWingHW,
                                ParameterBindings=hapi.DefaultParameterBindings,
                                EnvironmentDependencyBindings=hapi.DefaultEnvironmentDependencyBindings,
                                GammaL='gamma_air', HITRAN_units=True, LineShift=True,
                                File=None, Format=None, OmegaGrid=None):   
    """
    INPUT PARAMETERS: 
        Components:  list of tuples [(M,I,D)], where
                        M - HITRAN molecule number,
                        I - HITRAN isotopologue number,
                        D - abundance (optional)
        SourceTables:  list of tables from which to calculate cross-section   (optional)
        partitionFunction:  pointer to partition function (default is PYTIPS) (optional)
        Environment:  dictionary containing thermodynamic parameters.
                        'p' - pressure in atmospheres,
                        'T' - temperature in Kelvin
                        Default={'p':1.,'T':296.}
        OmegaRange:  wavenumber range to consider.
        OmegaStep:   wavenumber step to consider. 
        OmegaWing:   absolute wing for calculating a lineshape (in cm-1) 
        IntensityThreshold:  threshold for intensities
        OmegaWingHW:  relative wing for calculating a lineshape (in halfwidths)
        GammaL:  specifies broadening parameter ('gamma_air' or 'gamma_self')
        HITRAN_units:  use cm2/molecule (True) or cm-1 (False) for absorption coefficient
        File:   write output to file (if specified)
        Format:  c-format of file output (accounts significant digits in OmegaStep)
    OUTPUT PARAMETERS: 
        Omegas: wavenumber grid with respect to parameters OmegaRange and OmegaStep
        Xsect: absorption coefficient calculated on the grid
    ---
    DESCRIPTION:
        Calculate absorption coefficient using Voigt profile.
        Absorption coefficient is calculated at arbitrary temperature and pressure.
        User can vary a wide range of parameters to control a process of calculation
        (such as OmegaRange, OmegaStep, OmegaWing, OmegaWingHW, IntensityThreshold).
        The choise of these parameters depends on properties of a particular linelist.
        Default values are a sort of guess which gives a decent precision (on average) 
        for a reasonable amount of cpu time. To increase calculation accuracy,
        user should use a trial and error method.
    ---
    EXAMPLE OF USAGE:
        nu,coef = absorptionCoefficient_Voigt(((2,1),),'co2',OmegaStep=0.01,
                                              HITRAN_units=False,GammaL='gamma_self')
    ---
    """

    # warn user about too large omega step
    if OmegaStep>0.1: warn('Too small omega step: possible accuracy decline')

    # "bug" with 1-element list
    Components = hapi.listOfTuples(Components)
    SourceTables = hapi.listOfTuples(SourceTables)
    
    # determine final input values
    Components,SourceTables,Environment,OmegaRange,OmegaStep,OmegaWing,\
    IntensityThreshold,Format = \
       hapi.getDefaultValuesForXsect(Components,SourceTables,Environment,OmegaRange,
                                OmegaStep,OmegaWing,IntensityThreshold,Format)
    
    # get uniform linespace for cross-section
    #number_of_points = (OmegaRange[1]-OmegaRange[0])/OmegaStep + 1
    #Omegas = linspace(OmegaRange[0],OmegaRange[1],number_of_points)
    if OmegaGrid is not None:
        Omegas = hapi.npsort(OmegaGrid)
    else:
        Omegas = hapi.arange(OmegaRange[0],OmegaRange[1],OmegaStep)
    number_of_points = len(Omegas)
    Xsect = np.zeros(number_of_points)
       
    # reference temperature and pressure
    Tref = hapi.__FloatType__(296.) # K
    pref = hapi.__FloatType__(1.) # atm
    
    # actual temperature and pressure
    T = Environment['T'] # K
    p = Environment['p'] # atm
       
    # create dictionary from Components
    ABUNDANCES = {}
    NATURAL_ABUNDANCES = {}
    for Component in Components:
        M = Component[0]
        I = Component[1]
        if len(Component) >= 3:
            ni = Component[2]
        else:
            try:
                ni = hapi.ISO[(M,I)][hapi.ISO_INDEX['abundance']]
            except KeyError:
                raise Exception('cannot find component M,I = %d,%d.' % (M,I))
        ABUNDANCES[(M,I)] = ni
        NATURAL_ABUNDANCES[(M,I)] = hapi.ISO[(M,I)][hapi.ISO_INDEX['abundance']]
        
    # precalculation of volume concentration
    if HITRAN_units:
        factor = hapi.__FloatType__(1.0)
    else:
        factor = hapi.volumeConcentration(p,T) 
        
        #------------------------ GPU -------------------------
    #voigt_hum1wei24_gpu = mod2.get_function("voigt_hum1wei24_gpu")
    voigt_hum1wei24_gpu64 = mod3.get_function("voigt_hum1wei24_gpu64")
    positions  = []
    strengths = []
    NA = []
    A = []
    gammaL = []
    gammaD = []
        #------------------------ /GPU ------------------------


    # SourceTables contain multiple tables
    for TableName in SourceTables:

        # get line centers
        nline = hapi.LOCAL_TABLE_CACHE[TableName]['header']['number_of_rows']
        
        # loop through line centers (single stream)
        for RowID in range(nline):
            
            # get basic line parameters (lower level)
            LineCenterDB = hapi.LOCAL_TABLE_CACHE[TableName]['data']['nu'][RowID]
            LineIntensityDB = hapi.LOCAL_TABLE_CACHE[TableName]['data']['sw'][RowID]
            LowerStateEnergyDB = hapi.LOCAL_TABLE_CACHE[TableName]['data']['elower'][RowID]
            MoleculeNumberDB = hapi.LOCAL_TABLE_CACHE[TableName]['data']['molec_id'][RowID]
            IsoNumberDB = hapi.LOCAL_TABLE_CACHE[TableName]['data']['local_iso_id'][RowID]
            #Gamma0DB = hapi.LOCAL_TABLE_CACHE[TableName]['data']['gamma_air'][RowID]
            #Gamma0DB = hapi.LOCAL_TABLE_CACHE[TableName]['data']['gamma_self'][RowID]
            Gamma0DB = hapi.LOCAL_TABLE_CACHE[TableName]['data'][GammaL][RowID]
            TempRatioPowerDB = hapi.LOCAL_TABLE_CACHE[TableName]['data']['n_air'][RowID]
            #TempRatioPowerDB = 1.0 # for planar molecules
            if LineShift:
                Shift0DB = hapi.LOCAL_TABLE_CACHE[TableName]['data']['delta_air'][RowID]
            else:
                Shift0DB = 0
            
            # filter by molecule and isotopologue
            if (MoleculeNumberDB,IsoNumberDB) not in ABUNDANCES: continue
            
            # partition functions for T and Tref
            # TODO: optimize
            SigmaT = partitionFunction(MoleculeNumberDB,IsoNumberDB,T)
            SigmaTref = partitionFunction(MoleculeNumberDB,IsoNumberDB,Tref)
            
            # get all environment dependences from voigt parameters
            
            #   intensity
            LineIntensity = hapi.EnvironmentDependency_Intensity(LineIntensityDB,T,Tref,SigmaT,SigmaTref,
                                                            LowerStateEnergyDB,LineCenterDB)
            
            #   FILTER by LineIntensity: compare it with IntencityThreshold
            # TODO: apply wing narrowing instead of filtering, this would be more appropriate
            if LineIntensity < IntensityThreshold: continue
            
            #   doppler broadening coefficient (GammaD)
            # V1 >>>
            #GammaDDB = cSqrtLn2*LineCenterDB/cc*sqrt(2*cBolts*T/molecularMass(MoleculeNumberDB,IsoNumberDB))
            #GammaD = EnvironmentDependency_GammaD(GammaDDB,T,Tref)
            # V2 >>>
            cMassMol = 1.66053873e-27 # hapi
            #cMassMol = 1.6605402e-27 # converter
            m = hapi.molecularMass(MoleculeNumberDB,IsoNumberDB) * cMassMol * 1000
            GammaD = np.sqrt(2*hapi.cBolts*T*np.log(2)/m/hapi.cc**2)*LineCenterDB
            
            #   lorentz broadening coefficient
            Gamma0 = hapi.EnvironmentDependency_Gamma0(Gamma0DB,T,Tref,p,pref,TempRatioPowerDB)
            
            #   get final wing of the line according to Gamma0, OmegaWingHW and OmegaWing
            # XXX min or max?
            #OmegaWingF = max(OmegaWing,OmegaWingHW*Gamma0,OmegaWingHW*GammaD)

            #   shift coefficient
            Shift0 = Shift0DB*p/pref
            
            # XXX other parameter (such as Delta0, Delta2, anuVC etc.) will be included in HTP version
            
            #PROFILE_VOIGT(sg0,GamD,Gam0,sg)
            #      sg0           : Unperturbed line position in cm-1 (Input).
            #      GamD          : Doppler HWHM in cm-1 (Input)
            #      Gam0          : Speed-averaged line-width in cm-1 (Input).
            #      sg            : Current WaveNumber of the Computation in cm-1 (Input).
            
            # XXX time?
            #BoundIndexLower = bisect(Omegas,LineCenterDB-OmegaWingF)
            #BoundIndexUpper = bisect(Omegas,LineCenterDB+OmegaWingF)
            #lineshape_vals = PROFILE_VOIGT(LineCenterDB+Shift0,GammaD,Gamma0,Omegas[BoundIndexLower:BoundIndexUpper])[0]
            #Xsect[BoundIndexLower:BoundIndexUpper] += factor / NATURAL_ABUNDANCES[(MoleculeNumberDB,IsoNumberDB)] * \
            #                                          ABUNDANCES[(MoleculeNumberDB,IsoNumberDB)] * \
            #                                          LineIntensity * lineshape_vals
#
        #------------------------ GPU -------------------------
            positions.append(LineCenterDB+Shift0)
            gammaL.append(Gamma0)
            gammaD.append(GammaD)
            strengths.append(LineIntensity)
            NA.append(NATURAL_ABUNDANCES[(MoleculeNumberDB,IsoNumberDB)])
            A.append(ABUNDANCES[(MoleculeNumberDB,IsoNumberDB)])

        vGrid = Omegas
        #vGrid = Omegas[BoundIndexLower:BoundIndexUpper]
        N     = int(round(np.sqrt(len(vGrid))))
        XS    = np.zeros(len(vGrid),np.float64)
        L     = len(positions) 
        wBd = np.array([0.000000000000e+00,  -1.513746165453e-10,   4.904821733949e-09,   1.331046162178e-09,  -3.008282275137e-08,
                        -1.912225850812e-08,   1.873834348705e-07,   2.568264134556e-07,  -1.085647578974e-06,  -3.038893183936e-06,
                        4.139461724840e-06,   3.047106608323e-05,   2.433141546260e-05,  -2.074843151142e-04,  -7.816642995614e-04,
                        -4.936426901280e-04,   6.215006362950e-03,   3.372336685532e-02,   1.083872348457e-01,   2.654963959881e-01,
                        5.361139535729e-01,   9.257087138589e-01,   1.394819673379e+00,   1.856286499206e+00,   2.197858936532e+00],np.float64)

        voigt_hum1wei24_gpu64(np.int64(L),\
                drv.Out(XS), \
                drv.In(np.array(vGrid,np.float64)),\
                drv.In(np.array(positions,np.float64)), \
                drv.In(np.array(gammaL,np.float64)), \
                drv.In(np.array(gammaD,np.float64)), \
                drv.In(np.array(strengths,np.float64)), \
                drv.In(wBd),\
                drv.In(np.array(NA,np.float64)),\
                drv.In(np.array(A,np.float64)),\
                block=(N,1,1),grid=(N,1)
                )
        Xsect = factor * XS

    if File: hapi.save_to_file(File,Format,Omegas,Xsect)
    return Omegas,Xsect
