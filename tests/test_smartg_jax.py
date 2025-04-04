
import numpy as np
import pytest
import matplotlib.pyplot as plt
from smartg.histories import get_histories, BigSum, Si
from smartg.smartg import Smartg, with_cuda_context
from smartg.smartg import LambSurface, Albedo_cst
from smartg.atmosphere import AtmAFGL, AerOPAC, od2k
from smartg.albedo import Albedo_cst
from tests import conftest


@with_cuda_context
def Run1(atm, wl_sca, NB=1e4,  lez = {'th_deg':np.array([30.]), 'phi_deg':np.array([0.]), 'zip':False}):
    return Smartg(alis=True, alt_pp=True).run(THVDEG=45., wl=wl_sca, atm=atm, NBLOOP=NB, NBPHOTONS=NB, BEER=0,
                        surf=LambSurface(ALB=Albedo_cst(0.1)), alis_options={'nlow':wl_sca.size, 'hist':True}, 
                        progress=False,  NF=1e3, le=lez, OUTPUT_LAYERS=1)
    
@with_cuda_context
def GetI(m, wl_sca, wl_abs, sigma, alb, LEVEL):
    N, S, D, w, _, nref, _, _, _, _ = get_histories(m, LEVEL=LEVEL, verbose=False)
    I       = BigSum(Si, only_I=True)(wl_abs, sigma, alb, S[:,0], w, D, nref, wl_sca).sum(axis=0)/N
    return I

@pytest.mark.parametrize('N_WL_ABS', [100,1000])
def test_smartg_jax(N_WL_ABS, request):
    wl_sca       = np.linspace(320., 550., num=11)
    wl_abs       = np.linspace(320., 550., num=N_WL_ABS)
    alb          = np.zeros_like(wl_abs)

    for AOD in np.linspace(0., 0.3, num=2):
        atm = AtmAFGL('afglms', comp=[AerOPAC('urban',AOD, 550.)], grid=np.linspace(50., 0., num=40))
        sigma = od2k(atm.calc(wl_abs), 'OD_g')[:,1:]
        m=Run1(atm, wl_sca, NB=1E4)
        I=GetI(m, wl_sca, wl_abs, sigma, alb, 1)
        p=plt.plot(wl_abs, I, label='{:.1f}'.format(AOD))
    plt.legend()
    conftest.savefig(request)