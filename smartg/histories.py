"""Photon histories post-processing for ALIS simulations.

This module analyses the photon histories recorded by Smartg.run
when the ALIS option is used with alis_options['hist'] = True: it
rebuilds high-resolution Stokes vectors from the recorded events
(with JAX) and derives air mass factor (AMF) statistics.

Key Functions
-------------
get_histories
    Return the main outputs of the recorded photon histories.
compute_cdist_hist
    Compute cdist (tabDist) moments from ALIS photon histories.
amf_from_cdist
    Derive AMF statistics from a raw cdist moments array.
compute_amf
    Compute AMF from a Smartg MLUT, for both hist=False and
    hist=True runs.
"""

import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, vmap, jit
import xarray
import jax

def get_histories(m, LEVEL=0, IDIR=0,verbose=False):
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
            nref : A ndarray of size (NLE) of number of reflection on the surface (as described by the keyword surface in the run method)
            nsif : A ndarray of size (NLE) of Sun Induced Fluorescence event flag (1 : SIF, 0: no SIF)
            nvrs : A ndarray of size (NLE) of Vibrational Raman Scattering event flag (1 : VRS, 0: no VRS)
            nenv : A ndarray of size (NLE) of reflection on the environement (as described by the keyword environment in the run method)
            nint : A ndarray of size (NLE) of number of reflection or scattering
            nlscl : A ndarray of size (NLE) of last-scattering layer index (-1 = surface/unscattered)
    '''
    NL=m.axis('z_atm').size-1 if not isinstance(m, xarray.Dataset) else m['z_atm'].size-1
    tabHist_ = np.squeeze(m['histories'].data)
    tabHist = tabHist_[LEVEL, :,:]
    if verbose : print (tabHist.shape)
    w0      = tabHist[:, NL+4:-7] 
    #D0      = tabHist[:,0]
    good    = w0[:,0]!=0
    ngood   = np.sum(good)
    max_hist = tabHist.shape[0]
    if ngood >= max_hist:
        # Use print rather than warnings.warn: Python's default warning filter
        # deduplicates per call-site, so the message would silently disappear
        # on the second call in the same session.
        print(
            f"\033[1;33m[ALIS hist WARNING] History buffer saturated: "
            f"{ngood:,}/{max_hist:,} slots used ({100.*ngood/max_hist:.0f}%). "
            "Photons beyond max_hist were NOT recorded — results will be biased. "
            "→ Increase max_hist or reduce nb_photons per loop.\033[0m"
        )
    N = m['Nphotons_in'].data[0,0]
    ###################
    S       = np.zeros((ngood,4),dtype=np.float32) 
    D       = tabHist[good,     :NL  ]
    S[:,:4] = tabHist[good, NL:NL+4  ]
    w       = tabHist[good, NL+4:-7  ]
    nrrs    = tabHist[good,      -7  ]
    nref    = tabHist[good,      -6  ]
    nsif    = tabHist[good,      -5  ]
    nvrs    = tabHist[good,      -4  ]
    nenv    = tabHist[good,      -3  ]
    nint    = tabHist[good,      -2  ]
    nlscl   = tabHist[good,      -1  ]
    #
    if verbose : print('Number of photons in : {}\nNumber of LE photons : {}\nNumber of LR wavelengths : {}\nNumber of Layers : {}'.format(N, *w.shape, NL))
    
    return N, S, D, w, nrrs, nref, nsif, nvrs, nenv, nint, nlscl




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
    return Si(lam, kabs, alb, sik, wi_lr, Dij, Ki, lam_lr_grid)**2



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


# ─────────────────────────────────────────────────────────────────────────────
# Post-hoc AMF computation from ALIS photon histories
# ─────────────────────────────────────────────────────────────────────────────

def compute_cdist_hist(
    D_h, S_h, w_h, nref_h, nint_h, nlscl_h,
    wavelength_lr_r, wavelength_ref, alb_ref, natm_abs,
    *,
    amf_variance    = True,
    nscl            = 1,
    scatter_classes = 'last_scattering_layer',
    norders         = 1,
    cdist_wabs      = False,
    kabs_ref        = None,
):
    """
    Compute cdist (tabDist) moments from ALIS photon histories.

    Replicates the GPU tabDist accumulation for Smartg options:
        amf_variance, nscl, scatter_classes, norders, cdist_wabs.

    Parameters
    ----------
    D_h      : (NLE, NL) array  – path lengths per absorption layer [km]
    S_h      : (NLE, NStokes)   – Stokes components from pure-scattering MC run
    w_h      : (NLE, NLR)       – ALIS LR scattering-correction weights
    nref_h   : (NLE,)           – surface-reflection count per photon
    nint_h   : (NLE,) int       – scattering order (total interaction count)
    nlscl_h  : (NLE,) int       – last-scattering layer index (-1 = surface/none)
    wavelength_lr_r  : (NLR,)           – LR wavelength axis [nm]
    wavelength_ref   : float            – reference wavelength [nm]
                                          (weight evaluation)
    alb_ref  : float            – surface albedo at wavelength_ref
    natm_abs : int              – number of atmospheric absorption layers
    amf_variance : bool         – store 3rd moment Σ d²·w
    nscl     : int              – number of scatter classes (1 = no decomposition)
    scatter_classes : str       – 'last_scattering_layer' | 'scattering_order'
                                  | 'scattering_order_per_layer'
    norders  : int              – scattering-order bins per layer (mode 3 only)
    cdist_wabs : bool           – include Beer-Lambert transmittance in w_n
    kabs_ref : (natm_abs,) or None – kabs [km⁻¹] at wavelength_ref
                                     (for cdist_wabs=True)

    Returns
    -------
    cdist : ndarray, float64
        shape (natm_abs, niamf)          when nscl == 1
              (natm_abs, nscl, niamf)    when nscl >  1
        niamf = 3 if amf_variance else 2
    """
    niamf = 3 if amf_variance else 2
    NLE   = int(D_h.shape[0])

    wavelength_lr_j  = jnp.array(wavelength_lr_r, dtype=jnp.float32)
    wavelength_ref_j = jnp.float32(wavelength_ref)

    # 1. ALIS scattering-correction weight at wavelength_ref
    w_h_j    = jnp.array(w_h,  dtype=jnp.float32)
    w_scalar = vmap(
        lambda wi: jnp.interp(wavelength_ref_j, wavelength_lr_j, wi)
    )(w_h_j)

    # 2. Optional Beer-Lambert transmittance at wavelength_ref
    D_abs = jnp.array(D_h[:, :natm_abs], dtype=jnp.float32)
    if cdist_wabs and kabs_ref is not None:
        k_ref_j = jnp.array(kabs_ref[:natm_abs], dtype=jnp.float32)
        Tabs = jnp.exp(-jnp.sum(D_abs * k_ref_j[None, :], axis=1))
    else:
        Tabs = jnp.ones(NLE, dtype=jnp.float32)

    # 3. Effective photon weight  w_n = S_I · wsca · alb^Ki · Tabs
    safe_alb = jnp.float32(alb_ref if float(alb_ref) > 0. else 1.)
    nref_j   = jnp.array(nref_h, dtype=jnp.float32)
    S_h_j    = jnp.array(S_h,    dtype=jnp.float32)
    w_n = S_h_j[:, 0] * w_scalar * jnp.power(safe_alb, nref_j) * Tabs

    # 4. Scatter class index (mirrors SCL_MODE in device.cu)
    nint_np  = np.asarray(nint_h,  dtype=np.int32)
    nlscl_np = np.asarray(nlscl_h, dtype=np.int32)

    if nscl <= 1:
        cls_np = np.zeros(NLE, dtype=np.int32)
    elif scatter_classes == 'last_scattering_layer':
        cls_np = np.where(
            nlscl_np >= 0,
            np.minimum((nlscl_np * nscl) // natm_abs, nscl - 1),
            0,
        ).astype(np.int32)
    elif scatter_classes == 'scattering_order':
        cls_np = np.clip(np.minimum(nint_np, nscl) - 1, 0, nscl - 1).astype(np.int32)
    elif scatter_classes == 'scattering_order_per_layer':
        ilayer = np.where(nlscl_np >= 0, np.minimum(nlscl_np, natm_abs - 1), 0)
        iorder = np.where(nint_np  >  0, np.minimum(nint_np - 1, norders - 1), 0)
        cls_np = np.minimum(ilayer * norders + iorder, nscl - 1).astype(np.int32)
    else:
        raise ValueError(
            f"Unknown scatter_classes={scatter_classes!r}. "
            "Choose from: 'last_scattering_layer', 'scattering_order', "
            "'scattering_order_per_layer'."
        )

    # 5. Accumulate moments
    if nscl <= 1:
        W_tot = float(jnp.sum(w_n))
        Dw    = np.array(jnp.sum(D_abs * w_n[:, None], axis=0))
        cdist_out = np.empty((natm_abs, niamf), dtype=np.float64)
        cdist_out[:, 0] = W_tot
        cdist_out[:, 1] = Dw
        if amf_variance:
            cdist_out[:, 2] = np.array(jnp.sum(D_abs ** 2 * w_n[:, None], axis=0))
    else:
        cls_j  = jnp.array(cls_np, dtype=jnp.int32)
        cls_oh = (jnp.arange(nscl, dtype=jnp.int32)[None, :] == cls_j[:, None]).astype(jnp.float32)
        w_cls  = w_n[:, None] * cls_oh
        W_cls  = np.array(jnp.sum(w_cls, axis=0))
        DW     = np.array(jnp.einsum('il,ic->lc', D_abs, w_cls))
        cdist_out = np.empty((natm_abs, nscl, niamf), dtype=np.float64)
        cdist_out[:, :, 0] = W_cls[None, :]
        cdist_out[:, :, 1] = DW
        if amf_variance:
            D2W = np.array(jnp.einsum('il,ic->lc', D_abs ** 2, w_cls))
            cdist_out[:, :, 2] = D2W

    return cdist_out


def amf_from_cdist(cdist, thick):
    """
    Derive AMF statistics from a raw cdist moments array.

    Shared by the hist=False (GPU tabDist) and hist=True (post-hoc) paths.

    Parameters
    ----------
    cdist : (NL, niamf) or (NL, nscl, niamf) ndarray
        iAMF=0: Σ w,   iAMF=1: Σ d·w,   iAMF=2: Σ d²·w
    thick : (NL,) array – layer thicknesses [km]

    Returns
    -------
    dict with keys:
        'AMF'         : (NL,)       – total AMF per layer
        'std_AMF'     : (NL,)       – σ(AMF) (zeros when niamf < 3)
        'mean_dist'   : (NL,)       – mean path length [km]
        'W'           : (NL,)       – total weight
        -- only when cdist.ndim == 3 (nscl > 1): --
        'AMF_cls'     : (NL, nscl)  – per-class AMF
        'std_AMF_cls' : (NL, nscl)  – per-class σ(AMF)
        'W_cls'       : (NL, nscl)  – per-class weight
        'var_within'  : (NL,)       – within-class variance
        'var_between' : (NL,)       – between-class variance
    """
    thick   = np.asarray(thick, dtype=np.float64)
    niamf   = cdist.shape[-1]
    has_scl = (cdist.ndim == 3)

    if has_scl:
        W_cls         = cdist[:, :, 0]
        W             = W_cls.sum(axis=1)
        mean_dist_cls = cdist[:, :, 1] / np.where(W_cls > 0, W_cls, 1.)
        mean_dist     = cdist[:, :, 1].sum(axis=1) / np.where(W > 0, W, 1.)
        AMF           = mean_dist / thick
        AMF_cls       = mean_dist_cls / thick[:, None]
        result = dict(AMF=AMF, W=W, mean_dist=mean_dist, W_cls=W_cls, AMF_cls=AMF_cls)
        if niamf >= 3:
            mean_dist2_cls = cdist[:, :, 2] / np.where(W_cls > 0, W_cls, 1.)
            var_cls        = mean_dist2_cls - mean_dist_cls ** 2
            frac_cls       = W_cls / np.where(W > 0, W, 1.)[:, None]
            var_within     = (frac_cls * var_cls).sum(axis=1)
            var_between    = (frac_cls * (mean_dist_cls - mean_dist[:, None]) ** 2).sum(axis=1)
            std_AMF        = np.sqrt(np.maximum(var_within + var_between, 0.)) / thick
            std_AMF_cls    = np.sqrt(np.maximum(var_cls, 0.)) / thick[:, None]
            result.update(std_AMF=std_AMF, std_AMF_cls=std_AMF_cls,
                          var_within=var_within, var_between=var_between)
        else:
            result.update(std_AMF=np.zeros_like(AMF),
                          std_AMF_cls=np.zeros_like(AMF_cls))
    else:
        W         = cdist[:, 0]
        mean_dist = cdist[:, 1] / np.where(W > 0, W, 1.)
        AMF       = mean_dist / thick
        result    = dict(AMF=AMF, W=W, mean_dist=mean_dist)
        if niamf >= 3:
            mean_dist2    = cdist[:, 2] / np.where(W > 0, W, 1.)
            result['std_AMF'] = np.sqrt(np.maximum(mean_dist2 - mean_dist**2, 0.)) / thick
        else:
            result['std_AMF'] = np.zeros_like(AMF)

    return result


def compute_amf(m, *, wavelength_lr_r=None, wavelength_ref=None,
                alb_ref=None, natm_abs=None,
                amf_variance=True, nscl=1,
                scatter_classes='last_scattering_layer',
                norders=1, cdist_wabs=False, kabs_ref=None):
    """
    Compute AMF from a Smartg output — works transparently for hist=False and hist=True.

    Dispatches on the output content:
      • 'cdist_up (TOA)' present  →  hist=False: reads GPU tabDist directly
      • 'histories'       present  →  hist=True:  calls compute_cdist_hist()

    Parameters
    ----------
    m : xr.Dataset – Smartg.run() output (MLUT input is deprecated)
    wavelength_lr_r : (NLR,) array, optional
        LR wavelength axis [nm].  Defaults to m['wavelength'].
    wavelength_ref : float, optional
        Reference wavelength [nm].  Defaults to the median of
        wavelength_lr_r.
    alb_ref : float
        Surface albedo at wavelength_ref (required for hist=True path).
    natm_abs : int, optional
        Number of atmospheric absorption layers.
        Defaults to m['z_atm'].size - 1.
    amf_variance : bool   – include 3rd moment Σ d²·w (for σ(AMF))
    nscl : int            – number of scatter classes
    scatter_classes : str – 'last_scattering_layer' | 'scattering_order'
                            | 'scattering_order_per_layer'
    norders : int         – scatter-order bins per layer (mode 3 only)
    cdist_wabs : bool     – include Beer-Lambert transmittance weight in w_n
    kabs_ref : (natm_abs,) array – kabs [km⁻¹] at wavelength_ref
                                   (for cdist_wabs=True)

    Returns
    -------
    amf_dict : dict  – output of amf_from_cdist()
        Keys always present: 'AMF', 'std_AMF', 'W', 'mean_dist'
        Extra keys when nscl > 1: 'AMF_cls', 'std_AMF_cls', 'W_cls',
                                   'var_within', 'var_between'
    thick : (NL,) ndarray – layer thicknesses [km]
    cdist : (NL, niamf) or (NL, nscl, niamf) ndarray – raw moments
    """
    if hasattr(m, 'to_xarray'):  # legacy MLUT input
        m = m.to_xarray()
    thick = np.abs(np.diff(m['z_atm'].values))

    # Auto-fill optional parameters from the output
    if wavelength_lr_r is None:
        wavelength_lr_r  = m['wavelength'].values
    if wavelength_ref is None:
        wavelength_ref   = float(np.median(wavelength_lr_r))
    if natm_abs is None:
        natm_abs = int(m['z_atm'].size) - 1

    # Dispatch on the output content.
    # Check for 'histories' FIRST: a hist=True run also stores a basic
    # cdist_up (TOA) (nscl=1, niamf=2), so testing cdist first would
    # silently ignore the richer post-hoc computation.
    has_hist = True
    try:
        m['histories']
    except Exception:
        has_hist = False

    if has_hist:
        # hist=True path: compute cdist post-hoc from photon histories
        if alb_ref is None:
            raise ValueError(
                "compute_amf: alb_ref is required for the hist=True path "
                "(m contains 'histories')."
            )
        _, S, D, w, _, nref, _, _, _, nint, nlscl = get_histories(m)
        cdist = compute_cdist_hist(
            D, S, w, nref, nint, nlscl,
            wavelength_lr_r, wavelength_ref, alb_ref, natm_abs,
            amf_variance    = amf_variance,
            nscl            = nscl,
            scatter_classes = scatter_classes,
            norders         = norders,
            cdist_wabs      = cdist_wabs,
            kabs_ref        = kabs_ref,
        )
    else:
        # hist=False path: read GPU tabDist directly from the output
        try:
            da    = m['cdist_up (TOA)']
        except Exception:
            raise ValueError(
                "compute_amf: m contains neither 'histories' (hist=True) "
                "nor 'cdist_up (TOA)' (hist=False)."
            )
        names = list(da.dims)
        arr   = da.data
        idx   = [slice(None)] * arr.ndim
        for i, nm in enumerate(names):
            if nm in ('Azimuth angles', 'Zenith angles'):
                idx[i] = 0
        cdist = arr[tuple(idx)]                        # (NL, [nscl,] niamf)

    return amf_from_cdist(cdist, thick), thick, cdist