import numpy as np
from smartg.smartg import Smartg
from smartg.atmosphere import AtmAFGL
from smartg.tools.smartg_view import smartg_view

from smartg.histories import get_histories, BigSum, Si
from smartg.smartg import Smartg, with_cuda_context
from smartg.smartg import LambSurface, Albedo_cst
from smartg.atmosphere import AtmAFGL, AerOPAC, od2k
from smartg.albedo import Albedo_cst

from matplotlib import pyplot as plt



@with_cuda_context
def Run1(atm, wl_sca, NB=1e4,  lez = {'th_deg':np.array([0.]), 'phi_deg':np.array([0.]), 'zip':False}, MAX_HIST=1e6, hist=True, ALB=1.):
    return Smartg(alis=True, alt_pp=True).run(THVDEG=45., wl=wl_sca, atm=atm, NBLOOP=NB, NBPHOTONS=NB, BEER=0,
                        surf=LambSurface(ALB=Albedo_cst(ALB)), 
                        alis_options={'nlow':wl_sca.size, 'hist':hist, 'max_hist':np.int64(MAX_HIST)}, 
                        progress=False,  NF=1e3, le=lez, OUTPUT_LAYERS=1)
    
@with_cuda_context
def GetI(m, wl_sca, wl_abs, sigma, alb, LEVEL):
    N, S, D, w, _, nref, _, _, _, _ = get_histories(m, LEVEL=LEVEL, verbose=False)
    I       = BigSum(Si, only_I=True)(wl_abs, sigma, alb, S[:,0], w, D, nref, wl_sca).sum(axis=0)/N
    return I

def test_smartg_jax(N_WL_ABS, NBPHOTONS, MAX_HIST=1e6, fmt='-'):
    ALB = 0.8 # snow albedo
    wl_sca       = np.linspace(320., 350., num=11)
    wl_abs       = np.linspace(320., 350., num=N_WL_ABS)
    alb          = np.full_like(wl_abs, fill_value=ALB)

    for AOD in np.linspace(0.3, 0.3, num=1):
        LEVEL=0 # BOA dowanward reflectance, 0 : TOA
        atm = AtmAFGL('afglms', comp=[AerOPAC('urban',AOD, 550.)], grid=np.linspace(50., 0., num=40))
        sigma = od2k(atm.calc(wl_abs), 'OD_g')[:,1:]
        m=Run1(atm, wl_sca, NB=NBPHOTONS, MAX_HIST=MAX_HIST)
        m0=Run1(atm, wl_abs, NB=NBPHOTONS, MAX_HIST=MAX_HIST, hist=False, ALB=ALB).dropaxis('Zenith angles').dropaxis('Azimuth angles')
        I=GetI(m, wl_sca, wl_abs, sigma, alb, LEVEL)
        p=plt.plot(wl_abs, I, fmt, label='AOD@550: {:.1f}; NBPH={:.0e}; NBHIST={:.0e}'.format(AOD, np.int64(NBPHOTONS), np.int64(MAX_HIST)))
        m0['I_up (TOA)'].plot(color=p[0].get_color(), linestyle='--', label='reference')
        plt.xlabel(r'$\lambda (nm)$')
        plt.ylabel('TOA reflectance')
        plt.ylim(0,0.8)
        plt.title('Urban aerosols, SZA=45°, nadir viewing, snow albedo') 
        plt.grid()
    plt.legend()
    
    
    
if __name__ == "__main__":
    
    print("Running SMARTG photons paths histories example with JAX")
    """ UV TOA spectrum with aerosols, lambertian surface and constant albedo """
    N_WL_ABS = 200
    test_smartg_jax(N_WL_ABS, NBPHOTONS=1e4, MAX_HIST=1e6, fmt='r-')    
    #test_smartg_jax(N_WL_ABS, NBPHOTONS=1e4, MAX_HIST=1e7, fmt='r+-')
    #test_smartg_jax(N_WL_ABS, NBPHOTONS=1e5, MAX_HIST=1e6, fmt='b-')
    test_smartg_jax(N_WL_ABS, NBPHOTONS=1e5, MAX_HIST=1e7, fmt='b-')
    plt.savefig("fig.png")