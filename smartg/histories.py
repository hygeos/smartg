import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, vmap, jit
import xarray

def get_histories(m, LEVEL=0, verbose=False):
    ''' 
    Return photons histories main outputs
    
    Input
        m : a MLUT (or xarray) SMART-G output with the ALIS option and hist=True having been set
        
    Keyword 
        LEVEL : 0 or 1 (up TOA or down 0+ levels only)
        verbose : print the Number of injected photons (N), 
                  Number of Local Estimate virtual photons (NLE), 
                  Number of Low Resolution wavelengths recorded (NLR)
                  Number of vertical layers (NL)
        
    Output
        a tuple consisting of 
            N : the number of injected photons
            S : A ndarray of size (NLE, 4) for 4 Stokes components
            D : A ndarray of size (NLE, NL) for cumulative distances traveled in layers
            w : A ndarray of size (NLE, NLR) for corrective scattering weights for the different LR wavelengths
            nrrs : A ndarray of size (NLE) of Rotational Raman Scattering event flag (1 : RRS, 0: no RRS)
            nref : A ndarray of size (NLE) of number of reflection on the surface (as described by the keyword surf in the run method)
            nrrs : A ndarray of size (NLE) of Sun Induced Fluorescence event flag (1 : SIF, 0: no SIF)
            nvrs : A ndarray of size (NLE) of Vibrational Raman Scattering event flag (1 : VRS, 0: no VRS)
            nenv : A ndarray of size (NLE) of reflection on the environement (as described by the keyword env in the run method)
            ith  : A ndarray of size (NLE) of index of the Zenith angle LE direction of the virtual photon
    '''
    NL=m.axis('z_atm').size-1 if not isinstance(m, xarray.Dataset) else m['z_atm'].size-1
    tabHist_ = np.squeeze(m['histories'].data)
    tabHist = tabHist_[LEVEL, :,:]
    if verbose : print (tabHist.shape)
    w0      = tabHist[:, NL+4:-6] 
    #D0      = tabHist[:,0]
    good    = w0[:,0]!=0
    ngood   = np.sum(good)
    N = m['Nphotons_in'].data[0,0]
    ###################
    S       = np.zeros((ngood,4),dtype=np.float32) 
    D       = tabHist[good,     :NL  ]
    S[:,:4] = tabHist[good, NL:NL+4  ]
    w       = tabHist[good, NL+4:-6  ]
    nrrs    = tabHist[good,      -6  ]
    nref    = tabHist[good,      -5  ]
    nsif    = tabHist[good,      -4  ]
    nvrs    = tabHist[good,      -3  ]
    nenv    = tabHist[good,      -2  ]
    ith     = tabHist[good,      -1  ]
    #
    if verbose : print('Number of photons in : {}\nNumber of LE photons : {}\nNumber of LR wavelengths : {}\nNumber of Layers : {}'.format(N, *w.shape, NL))
    
    return N, S, D, w, nrrs, nref, nsif, nvrs, nenv, ith




def Si(lam, kabs, alb, sik, wi_lr, Dij, Ki, lam_lr_grid):
    '''
    JAX based computation ONE Stoke component of ONE virtual photon for ONE High Resolution wavelength
    
    Input
        lam : current HR wavelength (nm)
        kabs: A ndarray of size (NL) of gaseous absorption coefficient for the current wavelength and for all layers
        alb: surface albedo for the current wavelength
        sik : virtual LE photons Stokes component k
        wi_lr: A ndarray of size (NLR) virtual LE photons corrective scattering weights for the different LR wavelengths
        Dij  : A ndarray of size (NL) of the virtual LE photons cumulative distances traveled in layers
        Ki   : Number of reflection on the surface
        lam_lr_grid : A ndarray of size (NLR) LR wavelengths grid
    '''
    
    # interpolation of scattering weights at low spectral resolution to current lambda
    wi = jnp.interp(lam, lam_lr_grid, wi_lr)
    
    return sik * wi * jnp.exp(- jnp.sum(Dij * kabs)) * alb**Ki
    
    
def Si2(lam, kabs, alb, sik, wi_lr, Dij, Ki, lam_lr_grid):
    '''
    JAX based computation of the square of ONE Stoke component of ONE virtual photon for ONE High Resolution wavelength
    
    Input
        lam : current HR wavelength (nm)
        kabs: A ndarray of size (NL) of gaseous absorption coefficient for the current wavelength and for all layers
        alb: surface albedo for the current wavelength
        sik : virtual LE photons Stokes component k
        wi_lr: A ndarray of size (NLR) virtual LE photons corrective scattering weights for the different LR wavelengths
        Dij  : A ndarray of size (NL) of the virtual LE photons cumulative distances traveled in layers
        Ki   : Number of reflection on the surface
        lam_lr_grid : A ndarray of size (NLR) LR wavelengths grid
    '''
    
    # interpolation of scattering weights at low spectral resolution to current lambda
    wi = jnp.interp(lam, lam_lr_grid, wi_lr)

    return (sik * wi * jnp.exp(- jnp.sum(Dij * kabs)) * alb**Ki)**2



def BigSum(S, grad=None, only_I=False):
    '''
    JAX based function for computing ALL the Stokes vectors for ALL High Resolution wavelengths and for ALL LE photons
    '''
    if grad is not None : S = value_and_grad(S, argnums=grad)
    f1m = vmap(S,  in_axes=(0   ,    0,    0, None, None, None, None, None))  # co varying wavelengths inputs
    f2m = vmap(f1m,in_axes=(None, None, None,    0,    0,    0,    0, None)) # co vaying LE photons inputs
    f3m = vmap(f2m,in_axes=(None, None, None,    1, None, None, None, None)) # co varying Stoke components inputs

    if only_I : return jit(f2m)
    else : return jit(f3m)