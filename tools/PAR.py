import numpy as np
from smartg import Smartg, RoughSurface, LambSurface, FlatSurface
from profile.profil import Profile, AeroOPAC, CloudOPAC, REPTRAN, REPTRAN_IBAND
from luts import MLUT, LUT, Idx, merge, read_lut_hdf, read_mlut_hdf
from water.iop_spm import IOP_SPM
from water.iop_mm import IOP_MM
from smartg_view import semi_polar, smartg_view
#from IPython.html.widgets.interaction import interact, interact_manual
#from IPython.display import clear_output
from scipy.integrate import simps, trapz
import scipy.constants as cst
from pylab import *
import ephem

def Irr(R):
    #---------------------------------------------------------
    # Compute Irradiance from a 2D(phi,theta) Reflectance LUT from SMART-G
    #---------------------------------------------------------
    phi = R.axes[0]
    the = R.axes[1]
    mu  = np.cos(the*np.pi/180.)
    Tab = np.zeros(len(phi),dtype=float)
    for iphi in range(len(phi)):
        Tab[iphi] =  simps(mu *  R[iphi,:], -mu)
    return 2 * simps(Tab, phi*np.pi/180.) / np.pi  
    
def SpherIrr(R):
    #---------------------------------------------------------
    # Compute Spherical Irradiance from a 2D(phi,theta) Reflectance LUT from SMART-G
    #---------------------------------------------------------
    phi = R.axes[0]
    the = R.axes[1]
    mu  = np.cos(the*np.pi/180.)
    Tab = np.zeros(len(phi),dtype=float)
    for iphi in range(len(phi)):
        Tab[iphi] =  simps(R[iphi,:], -mu)
    return 2 * simps(Tab, phi*np.pi/180.) / np.pi  
    
def ReadREPTRAN_bands(repname, BAND=None, LMIN=None, LMAX=None, SAMPLING=100000, FULL=False):
    rep = REPTRAN(repname+'.cdf')
    L = rep.band_names
    wi_l=[]
    we_l=[]
    ex_l=[]
    dl_l=[]
    wb_l=[]
    ii_l=[]
    ni_l=[]
    ib_l=[]
    if FULL :band_l=[]
    if BAND==None:
        istart=0
    else:
        istart=rep.band(BAND).band
    for i in range(istart,len(L),SAMPLING):
        band = rep.band(L[i])
        if LMIN != None:
            if band.awvl[0] < LMIN : continue
        if LMAX != None:
            if band.awvl[-1] > LMAX : break
        
        for iband in band.ibands():
            wi = iband.band.awvl[iband.index] # wvl of internal band
            wi_l.append(wi)
            we = iband.band.awvl_weight[iband.index] # weight of internal band
            we_l.append(we)
            ex = iband.band.aextra[iband.index] # E0 of internal band
            ex_l.append(ex)
            dl = iband.band.Rint # bandwidth
            dl_l.append(dl)
            wb = np.mean(iband.band.awvl[:])
            wb_l.append(wb)
            ii = iband.index
            ii_l.append(ii)
            ni = iband.band.nband
            ni_l.append(ni)
            ib = iband.band.band
            ib_l.append(ib)
            if FULL:band_l.append(iband)
            
    wi=LUT(np.array(wi_l),axes=[wi_l],names=['Wavelengths'],desc='Wavelengths internal band')
    wb=LUT(np.array(wb_l),axes=[wi_l],names=['Wavelengths'],desc='Wavelengths central band')
    we=LUT(np.array(we_l),axes=[wi_l],names=['Wavelengths'],desc='Weight')
    ex=LUT(np.array(ex_l),axes=[wi_l],names=['Wavelengths'],desc='E0')
    dl=LUT(np.array(dl_l),axes=[wi_l],names=['Wavelengths'],desc='Dlambda')
    if FULL : return wi, wb, we, ex, dl, band_l
    else : return wi, wb, we, ex, dl

    
def Int(wi, wb, ex, we, dl, M=None, field=None, lim=[400.,700.]):
    '''
    all input vectors have same length, coming from REPTRAN
    wi : input wavelengths of internal bands (nm)
    wb : input wavelengths of bands (nm)
    ex : input extra-terrestrial irradiances at internal bands (W m-2 nm-1)
    we : input weights of internal bands
    dl : input bandwidths of bands (nm)
    M  : optional LUT or MLUT with 3D (lambda,phi,theta) or 1D (lambda) radiative field to spectrally integrate
    field : optional field of MLUT (ex: 'I_up (TOA'), if None, the extraterrestrial irradiance is integrated
    lim: spectral boundaries for integration
    
    returns
    spectrally integrated intensity and averaged intensity
    spectrally integrated daily quanta and average daily quanta
    '''
    ok=np.where((wb.data >=lim[0]) & (wb.data <lim[1]))[0]
    if (M != None) :
        if (field != None) :
            L = M[field]
            tab = L.data
        else : 
            tab = M.data
        
        if tab.ndim == 3 :
            R = np.rollaxis(tab,0,3)
            E = LUT(sum(R[:,:,ok] * ex.data[ok] * we.data[ok] * dl.data[ok], axis=2), \
                axes=[L.axes[1], L.axes[2]], desc='E_'+L.desc, \
                names=[L.names[1], L.names[2]], attrs=L.attrs)
            Q = LUT(sum(R[:,:,ok] * wi.data[ok] * ex.data[ok] * we.data[ok] * dl.data[ok], axis=2) , \
                axes=[L.axes[1], L.axes[2]], desc='Q_'+L.desc, \
                names=[L.names[1], L.names[2]], attrs=L.attrs)
        else:
            E = sum(tab[ok] * ex.data[ok] * we.data[ok] * dl.data[ok])
            Q = sum(tab[ok] * wi.data[ok] * ex.data[ok] * we.data[ok] * dl.data[ok])
    else:
        E = sum(ex.data[ok] * we.data[ok] * dl.data[ok])
        Q = sum(ex.data[ok] * wi.data[ok] * we.data[ok] * dl.data[ok])
    
    norm = sum(we.data[ok] * dl.data[ok])
    E *= 1e-3 # from mW/m2/nm to W/m2/nm 
    Eavg = E/norm
    Q *= 1e-12/(cst.h*cst.c*cst.Avogadro) * 3600*24 # from mW/m2/nm*nm to W/m2/m*m (1e-12) and then to
        # Einstein/m2/day

    Qavg = Q/norm
        
    return E, Eavg, Q, Qavg


def SpecInt(wi, wb, ex, we, dl, M=None, field=None, lim=[400.,700.]):
    '''
    all input vectors have same length, coming from REPTRAN
    wi : input wavelengths of internal bands (nm)
    wb : input wavelengths of bands (nm)
    ex : input extra-terrestrial irradiances at internal bands (W m-2 nm-1)
    we : input weights of internal bands
    dl : input bandwidths of bands (nm)
    M  : optional LUT or MLUT with 3D (lambda,phi,theta) or 1D (lambda) radiative field to spectrally integrate
    lim: spectral boundaries for integration
    
    returns
    spectrally integrated intensity
    '''
    ok=np.where((wb.data >=lim[0]) & (wb.data <lim[1]))[0]
    if (M != None) :
        if (field != None) :
            L = M[field]
            tab = L.data
        else :
            tab = M.data

        if tab.ndim == 3 :
            R = np.rollaxis(tab,0,3)
            E = LUT(sum(R[:,:,ok] * we.data[ok] * dl.data[ok], axis=2), \
                axes=[L.axes[1], L.axes[2]], desc=L.desc, \
                names=[L.names[1], L.names[2]], attrs={'LAMBDA':lim[0]})
        else:
            E = sum(tab[ok] * we.data[ok] * dl.data[ok])
    else:
        E = sum(we.data[ok] * dl.data[ok])

    norm = sum(we.data[ok] * dl.data[ok])
    Eavg = E/norm

    return E, Eavg


def SpecInt2(wi, wb, ex, we, dl, M=None, field=None, Irradiance=False, lim=None, DL=None):
    Eavg=[] 
    if DL == None:
        wu = np.unique(wb.data)
        DL = 1e-5
    else:
        if lim == None:
            lim = [wi.data.min(),wi.data.max()]
        wu = np.linspace(lim[0],lim[1]-DL,endpoint=True,num=(lim[1]-lim[0])/DL)
    for linf in wu:
        E1,E2 = SpecInt(wi, wb, ex, we, dl, M=M, field=field,lim=[linf,linf+DL])
        if Irradiance :
            Eavg.append(Irr(E2))
        else:
            Eavg.append(E2)
    Mavg = merge(Eavg, ['LAMBDA'])
    return  Mavg

def Int2(wi, wb, ex, we, dl, M=None, field=None, lim=[400.,700.], DL=1., Irradiance=False):
    l=[]
    Qavg=[]
    Eavg=[]
    Qint=[]
    Eint=[]   
    for linf in np.linspace(lim[0],lim[1]-DL,endpoint=True,num=(lim[1]-lim[0])/DL):
        E1,E2,Q1,Q2 = Int(wi, wb, ex, we, dl, M=M, field=field,lim=[linf,linf+DL])
        l.append(linf+DL/2.)
        if Irradiance :
            Eint.append(Irr(E1))
            Eavg.append(Irr(E2))
            Qint.append(Irr(Q1))
            Qavg.append(Irr(Q2))
        else:
            Eint.append(E1)
            Eavg.append(E2)
            Qint.append(Q1)
            Qavg.append(Q2)
    return l, Eint, Eavg, Qint, Qavg

def viewPAR(fi, fsp, SZA=None, RAA=None, verbose=False):
    repname='reptran_solar_coarse'
    #repname='reptran_solar_sentinel'
    LMIN=400. # nm
    LMAX=700. # nm
    SAMPLING=1 # reptran file undersampling (1 : all bands included)
    DL=5. #nm spectral interval for integration
    wi,wb,we,ex,dl = ReadREPTRAN_bands(repname,LMIN=LMIN,LMAX=LMAX,SAMPLING=SAMPLING)       
    M=read_mlut_hdf(fi)
    transdir=read_lut_hdf(fsp,'Direct Transmission',axnames=['Wavelengths'])
    transdir.axes[0]=wi.data
    transdir.names[0]='Wavelengths'

    # transform MLUT with an explicit wavelength axis
    # -------------------------------------------
    MM_l=[]
    for k in range(len(M.luts)):
        MM_tmp = M.luts[k]
        MM=LUT(MM_tmp.data,axes=[wi.data,MM_tmp.axes[1],MM_tmp.axes[2]],desc=MM_tmp.desc,
               names=['Wavelengths',MM_tmp.names[1],MM_tmp.names[2]],attrs=MM_tmp.attrs)
        MM_l.append(MM)
    M2 = MLUT(MM_l)
    # -------------------------------------------  
    iza = M2['I_up (TOA)'].attrs['VZA (deg.)']
    mus = np.cos(iza * np.pi/180.)
    if verbose :fig,ax=subplots()
    if verbose :ax.set_ylabel('Irradiance (mW/cm2/mic)', color='b')
    if verbose :ax.set_xlabel('Wavelength (nm)')
    if verbose : print '*****************************'
    if verbose : print '* %.2f - %.2f nm'%(LMIN,LMAX)
    if SZA!=None:
        if verbose : print '******* VZA : %5.2f  ********'%iza
    else:
        if verbose : print '******* SZA : %5.2f  ********'%iza
        if verbose : print '       Level             | PAR(Ein./m2/day) R/T | Irr.(mW/cm2/mic) R/T'

        #### Extra-terrestrial ##########
        lp,sEint_toad,sEavg_toad,sQint_toad,sQavg_toad = Int2(wi, wb, ex, we, dl, M=None, field=None, lim=[LMIN,LMAX], DL=DL)
        if verbose :ax.plot(np.array(lp),np.array(sEavg_toad)*mus*1e2,'k+-',label='TOAd')

        Eint_toad,Eavg_toad,Qint_toad,Qavg_toad = Int(wi, wb, ex, we, dl, M=None, field=None, lim=[LMIN,LMAX])
        if verbose : print '(1) TOA incident         |  %7.3f     1.000   |  %7.3f     1.000   '%(Qint_toad*mus,Eavg_toad*mus*1e2) # 1e2 conversion\
                            # from W m-2 nm-1 to mW cm-2 mic-1

        #### TOA ########## 
        field = 'I_up (TOA)'
        lp, sEint_toau,sEavg_toau,sQint_toau,sQavg_toau = Int2(wi, wb, ex, we, dl, M=M2, field=field, \
                    lim=[LMIN,LMAX], DL=DL, Irradiance=True)
        if verbose :ax.plot(np.array(lp),np.array(sEavg_toau)*mus*1e2,'r',label='TOAu')


        Eint_toau,Eavg_toau,Qint_toau,Qavg_toau = Int(wi, wb, ex, we, dl, M=M2, field=field, \
                    lim=[LMIN,LMAX])
        QQ=Irr(Qint_toau)
        EE=Irr(Eavg_toau)
        if verbose : print '(2) TOA up               |  %7.3f     %5.3f   |  %7.3f     %5.3f   '%(QQ*mus,QQ/Qint_toad,EE*mus*1e2,EE/Eavg_toad)       


        #### Surface direct ##########
        lp, sEdirint_boad,sEdiravg_boad,sQdirint_boad,sQdiravg_boad = Int2(wi, wb, ex, we, dl, M=transdir, field=None, \
                    lim=[LMIN,LMAX], DL=DL)
        if verbose :ax.plot(np.array(lp),np.array(sEdiravg_boad)*mus*1e2,'b',label='BOAd_dir')

        Edirint_boad,Ediravg_boad,Qdirint_boad,Qdiravg_boad = Int(wi, wb, ex, we, dl, M=transdir, field=None, lim=[LMIN,LMAX])
        if verbose : print '(3) BOA (0+) down  (dir) |  %7.3f     %5.3f   |  %7.3f     %5.3f   '%(Qdirint_boad*mus,Qdirint_boad/Qint_toad,Ediravg_boad*mus*1e2,Ediravg_boad/Eavg_toad)
        #### Surface diffuse ##########
        field = 'I_down (0+)'
        lp, sEdifint_boad,sEdifavg_boad,sQdifint_boad,sQdifavg_boad = Int2(wi, wb, ex, we, dl, M=M2, field=field, \
                    lim=[LMIN,LMAX], DL=DL, Irradiance=True)
        if verbose :ax.plot(np.array(lp),np.array(sEdifavg_boad)*mus*1e2,'c',label='BOAd_dif')

        Edifint_boad,Edifavg_boad,Qdifint_boad,Qdifavg_boad = Int(wi, wb, ex, we, dl, M=M2, field=field, lim=[LMIN,LMAX])
        QQ=Irr(Qdifint_boad)
        EE=Irr(Edifavg_boad)
        if verbose : print '(4) BOA (0+) down  (dif) |  %7.3f     %5.3f   |  %7.3f     %5.3f   '%(QQ*mus,QQ/Qint_toad,EE*mus*1e2,EE/Eavg_toad)
        Eavg_boad = EE + Ediravg_boad
        Qint_boad = QQ + Qdirint_boad

        #### Surface total ##########
        if verbose :ax.plot(np.array(lp),(np.array(sEdifavg_boad)+np.array(sEdiravg_boad))*mus*1e2,'k',label='BOAd_tot')
        if verbose : print '(5) BOA (0+) down  (tot) |  %7.3f     %5.3f   |  %7.3f     %5.3f   '\
            %(Qint_boad*mus,Qint_boad/Qint_toad,Eavg_boad*mus*1e2,Eavg_boad/Eavg_toad)

        #### Surface Up ##########
        field = 'I_up (0+)'
        lp, sEint_boau,sEavg_boau,sQint_boau,sQavg_boau = Int2(wi, wb, ex, we, dl, M=M2, field=field, \
                    lim=[LMIN,LMAX], DL=DL, Irradiance=True)
        if verbose :ax.plot(np.array(lp),np.array(sEavg_boau)*mus*1e2,'g',label='BOAu') 

        Eint_boau,Eavg_boau,Qint_boau,Qavg_boau = Int(wi, wb, ex, we, dl, M=M2, field=field, \
                    lim=[LMIN,LMAX])
        QQ=Irr(Qint_boau)
        EE=Irr(Eavg_boau)
        if verbose : print '(6) BOA (0+) up          |  %7.3f     %5.3f   |  %7.3f     %5.3f   '%(QQ*mus,QQ/Qint_boad,EE*mus*1e2,EE/Eavg_boad)

        #### Water down ##########
        field = 'I_down (0-)'
        lp, sEint_tood,sEavg_tood,sQint_tood,sQavg_tood = Int2(wi, wb, ex, we, dl, M=M2, field=field, \
                    lim=[LMIN,LMAX], DL=DL, Irradiance=True)
        if verbose :ax.plot(np.array(lp),np.array(sEavg_tood)*mus*1e2,'y',label='TOOd') 

        Eint_tood,Eavg_tood,Qint_tood,Qavg_tood = Int(wi, wb, ex, we, dl, M=M2, field=field, \
                    lim=[LMIN,LMAX])
        QQ=Irr(Qint_tood)
        EE=Irr(Eavg_tood)
        if verbose : print '(7) BOA (0-) down        |  %7.3f     %5.3f   |  %7.3f     %5.3f   '%(QQ*mus,QQ/Qint_boad,EE*mus*1e2,EE/Eavg_boad)


        #### Water up ##########
        field = 'I_up (0-)'
        lp, sEint_toou,sEavg_toou,sQint_toou,sQavg_toou = Int2(wi, wb, ex, we, dl, M=M2, field=field, \
                    lim=[LMIN,LMAX], DL=DL, Irradiance=True)
        if verbose :ax.plot(np.array(lp),np.array(sEavg_toou)*mus*1e2,'k',label='TOOu') 

        Eint_toou,Eavg_toou,Qint_toou,Qavg_toou = Int(wi, wb, ex, we, dl, M=M2, field=field, \
                    lim=[LMIN,LMAX])
        QQ=Irr(Qint_toou)
        EE=Irr(Eavg_toou)
        if verbose : print '(8) BOA (0-) up          |  %7.3f     %5.3f   |  %7.3f     %5.3f   '%(QQ*mus,QQ/Irr(Qint_tood),EE*mus*1e2,EE/Irr(Eavg_tood))


        #### Water spherical ##########
        QQ=SpherIrr(Qint_tood) + SpherIrr(Qint_toou)
        EE=SpherIrr(Eavg_tood) + SpherIrr(Eavg_toou)
        if verbose : print '(9) BOA (0-) spher.      |  %7.3f             |  %7.3f             '%(QQ*mus,EE*mus*1e2)


        if verbose :legend(bbox_to_anchor=(-0.15, 1), loc='upper right', borderaxespad=0.)
    
    # TOA radiances in particular geometry
    if (SZA!=None) and (RAA!=None):
        field = 'I_up (TOA)'
        if verbose :ax2 = ax.twinx()
        if verbose :ax2.set_ylabel('Radiance (mW/cm2/mic/sr)', color='r')
        lp, sEint_toau,sEavg_toau,sQint_toau,sQavg_toau = Int2(wi, wb, ex, we, dl, M=M2, field=field, \
                    lim=[LMIN,LMAX], DL=DL, Irradiance=False)
        
        if verbose : print '(0) TOA up sat. SZA:%5.1f RAA:%5.1f | Radiance (mW/cm2/mic/sr)'%(SZA,RAA) 
        sRad = []
        musObs=cos(SZA*np.pi/180.)
        for k,Rad in enumerate(sEavg_toau):
            r = Rad[Idx(RAA),Idx(SZA)]*musObs*1e2/np.pi
            #r = Rad[Idx(RAA),Idx(SZA)]*mus*1e2/np.pi
            sRad.append(r)
            if verbose :print '(0) ----------- %5.1f(nm)| %7.3f '%(lp[k],r) 
        if verbose :ax2.plot(np.array(lp),np.array(sRad),'r+-',label='TOAu')

        if verbose : print '** VIIRS channels'
        lviirs=[410.,443.,486.,551.,671.]
        r_l=[]
        for lam in lviirs:
            lp, sEint_toau,sEavg_toau,sQint_toau,sQavg_toau = Int2(wi, wb, ex, we, dl, M=M2, field=field, \
                    lim=[lam-DL/2,lam+DL/2], DL=DL, Irradiance=False)
            Rad = sEavg_toau[0]
            r = Rad[Idx(RAA),Idx(SZA)]*musObs*1e2/np.pi
            #r = Rad[Idx(RAA),Idx(SZA)]*mus*1e2/np.pi
            if verbose : print '--------------- %5.1f(nm)| %7.3f '%(lam,r)
            r_l.append(r)
            #semi_polar(Rad)
        if verbose : legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.)
        return r_l
    else :
        return Eavg_boad*mus*1e2, Qint_boad*mus
    
    
def dailyPAR(fpp_l, fsp_l, dt_l ,verbose=False):
    for i,fpp in enumerate(fpp_l):
        fsp = fsp_l[i]
        dt = dt_l[i]
        PAR=[0.] # PAR at sunrise
        PAR2=[0.] # PAR at sunrise
        if verbose :print '------------------------------------------------------------ '
        if verbose :print '------------------- date number :%i ------------------------ '%i
        if verbose :print '------------------------------------------------------------ '
        for k,f in enumerate(fpp):
            E, Q = viewPAR(f, fsp[k],verbose=verbose)
            PAR.append(E)
            PAR2.append(Q)
        PAR.append(0.) # PAR at sunset
        PAR2.append(0.) # PAR at sunset
        dPAR = simps(PAR,dt)
        dPAR2 = simps(PAR2,dt)
        dlength=dt[-1] - dt[0]

        print 'dlength (day) : %.4f , daily PAR: %8.3f (mW/cm2/mic) %8.3f (E/m2/day), daytime PAR: %8.3f (mW/cm2/mic)'\
                %(dlength,dPAR,dPAR2,dPAR/dlength)
            
def ObsPAR(fi_l, fsp_l, SZA_l, raa_l,verbose=False):
    for i,fpp in enumerate(fi_l):
        fsp = fsp_l[i]
        SZA = SZA_l[i]
        raa = raa_l[i]
        if verbose :print '------------------------------------------------------------ '
        if verbose :print '------------------- date number :%i ------------------------ '%i
        if verbose :print '------------------------------------------------------------ '
        for k,f in enumerate(fpp):
            print '%8.4f %8.4f %8.4f %8.4f %8.4f'%tuple(viewPAR(f, fsp[k], SZA=SZA, RAA=raa,verbose=verbose))
   
def simulatePAR(CALC=True, verbose=False) :
    #--------------------------
    # spectral and spatial sampling
    #--------------------------
    LMIN=400. # nm
    LMAX=700. # nm
    PFRES=50. # spectral resolution for phase function (nm)
    SAMPLING = 1 # Reptran band undersampling rate
    grid='100[30]10[5]10[1]0' # altitude grid for scattering and absorption coefficients
    pfgrid=[100, 20, 10, 8., 6., 3., 2., 0.] # altitude grid for phase functions
    pfwav=np.linspace(LMIN, LMAX, num=(LMAX-LMIN)/PFRES+1, endpoint=True) # wavelength grid for phase functions
    repname='reptran_solar_coarse' # k distribution 
    #repname='reptran_solar_sentinel'
    rep = REPTRAN(repname+'.cdf')
    o3=300. # DU
    wref=862. # wavelength for AOT and COT reference
    aot=0.1
    #atmmodel, wvc = ('afglms', 2.96) # g/cm2
    #atmmodel, wvc = ('afglt', 4.18) # g/cm2
    atmmodel, wvc, SurfPre = ('afglmw', 0.89, 1018.) # g/cm2, mbar
    #atmmodel, wvc = ('afglsw', 0.42) # g/cm2
    #atmmodel, wvc = ('afglss', 2.11) # g/cm2
       
    
    aermodel_l=['maritime_clean','continental_average']
    aermodel=aermodel_l[1]
    cloudscatt='wc.sol.mie' # scattering properties of water cloud in the solar spectrum through Mie calculations
    cloudname='Stratocumulus'
    #cloudname='Cirrus'
    wc = 1 # water cloud droplets arbitrary concentration
    cot = 1. 
    reff = 12. # cloud droplets effective radius (mic)
    Zcmin = 9. # cloud bottom height (km)
    Zcmax = 10. # cloud top height (km)
    chl=0.1 # mg/l
    ws=5. # m/s
    lat=45.
    lon=-40.
    da_l=['2015/7/15','2015/1/15'] # date of acquisition

    vza_l=[0., 30., 60.]
    raa=90. # Sun VIIRS relative azimuth
    time_step=120. # time step for daily PAR integration (minute)
    
    wi,wb,we,ex,dl,ibands = ReadREPTRAN_bands(repname,LMIN=LMIN,LMAX=LMAX,SAMPLING=SAMPLING,FULL=True)
    
    aer = AeroOPAC(aermodel, aot, wref)
    if CALC : 
            pro = Profile(atmmodel,O3=o3,
                grid=grid,  # optional, otherwise use default grid
                #pfgrid=pfgrid,   # optional, otherwise use a single band 100-0
                pfwav=pfwav # optional, otherwise phase functions are calculated at all bands
                #,aer=aer
                ,cloud=CloudOPAC(cloudname,[(cloudscatt, wc, reff , Zcmin, Zcmax)], cot, wref)
                )
    else :
            pro = Profile(atmmodel,O3=o3
                #,aer=aer
                ,cloud=CloudOPAC(cloudname,[(cloudscatt, wc, reff , Zcmin, Zcmax)], cot, wref)
                )
    '''
    pro.calc(665.)            
    aer.calc(665.)
    t665=aer.tau_tot
    aer.calc(865.)
    t865=aer.tau_tot
    Angstrom = -np.log(t665/t865)/np.log(665./865.)
    '''
    Angstrom=0.
    
    surf=RoughSurface(SUR=3, WIND=ws, NH2O=1.34)
    #water=IOP_SPM(SPM=100.,pfwav=pfwav,NANG=7201)
    water=IOP_MM(chl,pfwav=pfwav,NANG=7201)
    
    
    #---------------------------
    # Geometry
    #---------------------------
    fpp_l_l  = []
    fsp_l_l  = []
    fpp2_l_l = []
    fsp2_l_l = []
    dt_l_l   = []
    SZA_l = []
    raa_l = []
    for idate,da in enumerate(da_l):
    
        natl=ephem.Observer()
        natl.lon=str(lon)
        natl.lat=str(lat)
        natl.date=da
        sun=ephem.Sun()
        drise=natl.next_rising(sun,start=da)
        dset=natl.next_setting(sun,start=da)
        #daylength=dset- drise
        dnoon=natl.next_transit(sun,start=da) # new date for sun position at local noon
        dviirs=ephem.Date(dnoon+90*ephem.minute) # viirs overpass at 13:30 local time
        natl.date=dviirs # observer date set to viirs overpass
        sun.compute(natl)
        if verbose:
            print '#--------- VIIRS PAR SIMULATION ------------------'
            print '# Lmin=%6.3f nm, Lmax=%6.3f nm'%(LMIN,LMAX)
            print '# Spectral parametrization:%s'%repname
            print '# Lat=%6.3f, Lon=%6.3f  '%(lat,lon)
            print '# sunrise:%s, sunset:%s , viirs overpass:%s (noon+1h30)'%(drise,dset,dviirs)
            print '# Aerosols:%s, AOT(%.2f nm)=%.2f, Ang.(665/865):%5.3f'%(aermodel,wref,aot,Angstrom)
            print '# Atmosphere:%s, O3=%.1f DU, H2O=%.2f g/cm2, SurfPre=%7.1f mbar'%(atmmodel,o3,wvc,SurfPre)
            print '# Windspeed=%4.1f m/s, Chlorophyll=%5.2f mg/l '%(ws,chl)
            print '# Cloud=%s, COT=%5.1f, Zmin=%5.1f, Zmax=%5.1f '%(cloudname,cot,Zcmin,Zcmax)

        # first runs with SZA's for irradiance calculations
        fpp_l=[]
        fsp_l=[]
        dt_l =[drise]

        start = ephem.Date(drise+20*ephem.minute) # we start 20 min after sunrise
        stop = ephem.Date(dset-20*ephem.minute) # we stop 20 min before sunset
        for date in np.linspace(start,stop,endpoint=True,num=8):
            natl.date = date
            sun.compute(natl)
            SZA = 90.-float(sun.alt)*180/np.pi     
            dt_l.append(date)

            if CALC : fpp_l.append(Smartg('SMART-G-PP', wl = ibands, THVDEG=SZA,
            #if CALC : fpp_l.append(Smartg('SMART-G-PP', wl = list(wi.data), THVDEG=SZA,
                      atm=pro, dir='/home/did/RTC/SMART-G/tools/PAR/VIIRS3',
                      NBPHOTONS=1e7,OUTPUT_LAYERS=3,
                      surf=surf, water=water, overwrite=True, NFAER=10000,NFOCE=100000).output)

            if CALC : fsp_l.append(Smartg('SMART-G-SP', wl = ibands, THVDEG=SZA,
            #if CALC : fsp_l.append(Smartg('SMART-G-SP', wl = list(wi.data), THVDEG=SZA,
                      atm=pro, dir='/home/did/RTC/SMART-G/tools/PAR/VIIRS3',
                      NBPHOTONS=1e7,OUTPUT_LAYERS=0,
                      surf=surf, water=water, overwrite=True, NFAER=10000,NFOCE=100000).output)
        dt_l.append(dset)

        fpp_l_l.append(fpp_l)
        fsp_l_l.append(fsp_l)
        dt_l_l.append(dt_l)
        # second runs with VZA's for VIIRS observations simulation
        natl.date=dviirs # observer date set to viirs overpass
        sun.compute(natl) 
        SZA = 90.-float(sun.alt)*180/np.pi  
        if verbose: print 'VIIRS overpass Sun position: SZA:%5.1f RAA:%5.1f'%(SZA,raa)
        SZA_l.append(SZA)
        raa_l.append(raa)
        year = dviirs.tuple()[0]
        month= dviirs.tuple()[1]
        day = dviirs.tuple()[2]
        hour_dec = (dviirs.triple()[2]-day)*24
        HEADER = '%4i %2i %2i %7.3f %7.3f %7.3f'%(year,month,day,hour_dec,lat,lon)
        
        if (idate==0 or CALC==False): # one run only per date for VIIRS observation (it contains all SZA and RAA)
            fpp2_l=[]
            fsp2_l=[]  
            for VZA in vza_l:
                TRAIL  = '%7.3f %7.3f %7.3f %7.3f %7.3f %7.1f %7.3f %7.1f'%(SZA,VZA,raa,aot,Angstrom,o3,wvc,SurfPre)
                if CALC : fpp2_l.append(Smartg('SMART-G-PP', wl = ibands, THVDEG=VZA,
                #if CALC : fpp2_l.append(Smartg('SMART-G-PP', wl = list(wi.data), THVDEG=VZA,
                          atm=pro, dir='/home/did/RTC/SMART-G/tools/PAR/VIIRS3',
                          NBPHOTONS=1e8,OUTPUT_LAYERS=3,
                          surf=surf, water=water, overwrite=True, NFAER=10000,NFOCE=100000).output)

                #if CALC : fsp2_l.append(Smartg('SMART-G-SP', wl = list(wi.data), THVDEG=VZA,
                if CALC : fsp2_l.append(Smartg('SMART-G-SP', wl = ibands, THVDEG=VZA,
                          atm=pro, dir='/home/did/RTC/SMART-G/tools/PAR/VIIRS3',
                          NBPHOTONS=1e7,OUTPUT_LAYERS=0,
                          surf=surf, water=water, overwrite=True, NFAER=10000,NFOCE=100000).output)
                print HEADER,TRAIL
        fpp2_l_l.append(fpp2_l)
        fsp2_l_l.append(fsp2_l)

    return fpp_l_l, fsp_l_l, fpp2_l_l, fsp2_l_l, dt_l_l, SZA_l, raa_l 
