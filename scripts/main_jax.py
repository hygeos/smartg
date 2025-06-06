import numpy as np
from smartg.smartg import Smartg
from smartg.atmosphere import AtmAFGL
from smartg.tools.smartg_view import smartg_view

from smartg.histories import get_histories, BigSum, Si, Si2
from smartg.smartg import Smartg
from smartg.smartg import LambSurface, Albedo_cst
from smartg.atmosphere import AtmAFGL, AerOPAC, od2k
from smartg.albedo import Albedo_cst

from matplotlib import pyplot as plt   
    
    
def test_smartg_jax2(N_WL_ABS, WMIN, WMAX, NBPHOTONS=2e4, MAX_HIST=2e7):
    ALB_SNOW     = Albedo_cst(0.6)
    ALB_HIST     = Albedo_cst(1.0)
    wl_sca       = np.linspace(WMIN, WMAX, num=11)
    wl_abs       = np.linspace(WMIN, WMAX, num=N_WL_ABS)
    alb          = ALB_SNOW.get(wl_abs)
    lez          = {'th_deg':np.array([0.]), 'phi_deg':np.array([0.]), 'zip':False}

    for AOD, fmt1 in zip(np.linspace(0.1, 0.3, num=2), ['-m', '-c']):
        LEVEL=0 # 1: BOA downward reflectance, 0 : TOA
        atm = AtmAFGL('afglms', comp=[AerOPAC('urban',AOD, 550.)], grid=np.linspace(50., 0., num=40))
        sigma = od2k(atm.calc(wl_abs), 'OD_abs_atm')[:,1:]
        m  = Smartg(alis=True, alt_pp=True). run(THVDEG=45., wl=wl_sca, surf=LambSurface(ALB_HIST), le=lez, BEER=0, atm=atm.calc(wl_sca), 
            alis_options={'nlow':wl_sca.size,'hist':True, 'max_hist':np.int64(MAX_HIST)},
            NBPHOTONS=NBPHOTONS, NBLOOP=NBPHOTONS, NF=1e3).dropaxis('Zenith angles').dropaxis('Azimuth angles')
        m0 = Smartg(alis=True, alt_pp=True). run(THVDEG=45., wl=wl_abs, surf=LambSurface(ALB_SNOW), le=lez, BEER=0, atm=atm.calc(wl_abs), 
            alis_options={'nlow':wl_sca.size,'hist':False},
            NBPHOTONS=NBPHOTONS, NF=1e3).dropaxis('Zenith angles').dropaxis('Azimuth angles')
        N, S, D, w, _, nref, _, _, _, _ = get_histories(m, LEVEL=LEVEL, verbose=False)
        I  = BigSum(Si,  only_I=True) (wl_abs, sigma, alb, S[:,0], w, D, nref, wl_sca).sum(axis=0)/N
        I2 = BigSum(Si2, only_I=True) (wl_abs, sigma, alb, S[:,0], w, D, nref, wl_sca).sum(axis=0)/N
        Std     = np.sqrt((I2-I**2)/N)
        upper   = I + 1.95*Std
        lower   = I - 1.95*Std
        p=plt.plot(wl_abs, I, fmt1, label='AOD@550: {:.1f}; NBPH={:.0e}; NBHIST={:.0e}'.format(AOD, np.int64(NBPHOTONS), np.int64(MAX_HIST)))
        col = p[0].get_color()
        print(I2-I**2)
        plt.fill_between(wl_abs, lower, upper, facecolor=col, edgecolor=col, alpha=0.4, label='95 percent confidence')
        plt.plot(wl_abs, m0['I_up (TOA)'][:], marker='+', ls='', label='AOD@550: {:.1f}; NBPH={:.0e}; NO HIST'.format(AOD, np.int64(NBPHOTONS)), color=p[0].get_color())
        plt.xlabel(r'$\lambda (nm)$')
        plt.ylabel('TOA reflectance')
        #plt.ylim(0.,0.7)
        plt.title('Urban aerosols, SZA=45°, nadir viewing, snow albedo') 
        plt.grid()
        plt.legend()


    
if __name__ == "__main__":
    
    print("Running SMARTG photons paths histories example with JAX")
    """ UV TOA spectrum with aerosols, lambertian surface and constant albedo """
    N_WL_ABS = 201
    WMIN, WMAX = 520., 550.
    test_smartg_jax2(N_WL_ABS, WMIN, WMAX, NBPHOTONS=2e5, MAX_HIST=2e7)
    plt.savefig("fig.png")