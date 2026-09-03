"""Photon histories post-processing for ALIS simulations.

This module analyses the photon histories recorded by Smartg.run
when the ALIS option is used with alis_options['hist'] = True: it
rebuilds high-resolution Stokes vectors from the recorded events
(with JAX) and derives air mass factor (AMF) statistics.

Functions
---------
get_histories
    Return the main outputs of the recorded photon histories.
si
    Beer-Lambert weight of one Stokes component, one virtual
    photon and one high-resolution wavelength.
si2
    Square of `si`, for the variance of the rebuilt Stokes
    component.
big_sum
    Vectorize and JIT-compile a `si`-like function over
    wavelengths, photons and, optionally, Stokes components.
compute_cdist_hist
    Compute cdist (tabDist) moments from ALIS photon histories.
amf_from_cdist
    Derive AMF statistics from a raw cdist moments array.
compute_amf
    Compute AMF from a Smartg output, for both hist=False and
    hist=True runs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, vmap, jit
import xarray as xr
import jax
from luts.luts import MLUT
from numpy.typing import NDArray

def get_histories(
    m: MLUT | xr.Dataset,
    level: int = 0,
    idir: int = 0,
    verbose: bool = False,
) -> tuple[
    int,
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """Return the main outputs of the recorded photon histories.

    Parameters
    ----------
    m : MLUT or xarray.Dataset
        A Smartg output with the ALIS option and hist=True set.
    level : int, optional
        The output level: 0 for TOA (up), 1 for downward at the 0+
        level. Default 0.
    idir : int, optional
        Currently unused. Default 0.
    verbose : bool, optional
        If True, print the number of injected photons (N), of
        Local Estimate virtual photons (NLE), of Low Resolution
        wavelengths recorded (NLR) and of vertical layers (NL).
        Default False.

    Returns
    -------
    n : int
        The number of injected photons.
    s : ndarray of shape (NLE, 4)
        The 4 Stokes components of the virtual photons.
    d : ndarray of shape (NLE, NL)
        The cumulative distances traveled in each layer.
    w : ndarray of shape (NLE, NLR)
        The corrective scattering weights for the different LR
        wavelengths.
    nrrs : ndarray of shape (NLE,)
        The Rotational Raman Scattering event flag (1: RRS, 0: no
        RRS).
    nref : ndarray of shape (NLE,)
        The number of reflections on the surface (as described by
        the surface keyword of the run method).
    nsif : ndarray of shape (NLE,)
        The Sun Induced Fluorescence event flag (1: SIF, 0: no
        SIF).
    nvrs : ndarray of shape (NLE,)
        The Vibrational Raman Scattering event flag (1: VRS, 0: no
        VRS).
    nenv : ndarray of shape (NLE,)
        The number of reflections on the environment (as described
        by the environment keyword of the run method).
    nint : ndarray of shape (NLE,)
        The number of reflections or scatterings.
    nlscl : ndarray of shape (NLE,)
        The last-scattering layer index (-1 for surface/unscattered
        photons).
    """
    nl=m.axis('z_atm').size-1 if not isinstance(m, xr.Dataset) else m['z_atm'].size-1
    tab_hist_ = np.squeeze(m['histories'].data)
    tab_hist = tab_hist_[level, :,:]
    if verbose : print (tab_hist.shape)
    w0      = tab_hist[:, nl+4:-7]
    #d0      = tab_hist[:,0]
    good    = w0[:,0]!=0
    ngood   = np.sum(good)
    max_hist = tab_hist.shape[0]
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
    n = m['Nphotons_in'].data[0,0]
    ###################
    s       = np.zeros((ngood,4),dtype=np.float32)
    d       = tab_hist[good,     :nl  ]
    s[:,:4] = tab_hist[good, nl:nl+4  ]
    w       = tab_hist[good, nl+4:-7  ]
    nrrs    = tab_hist[good,      -7  ]
    nref    = tab_hist[good,      -6  ]
    nsif    = tab_hist[good,      -5  ]
    nvrs    = tab_hist[good,      -4  ]
    nenv    = tab_hist[good,      -3  ]
    nint    = tab_hist[good,      -2  ]
    nlscl   = tab_hist[good,      -1  ]
    #
    if verbose : print('Number of photons in : {}\nNumber of LE photons : {}\nNumber of LR wavelengths : {}\nNumber of Layers : {}'.format(n, *w.shape, nl))

    return n, s, d, w, nrrs, nref, nsif, nvrs, nenv, nint, nlscl




def si(
    lam: float,
    kabs: NDArray[np.floating],
    alb: float,
    sik: float,
    wi_lr: NDArray[np.floating],
    dij: NDArray[np.floating],
    ki: float,
    lam_lr_grid: NDArray[np.floating],
) -> jax.Array:
    """Beer-Lambert weight of one photon's Stokes component.

    JAX based computation, for one virtual photon, one Stokes
    component and one high-resolution wavelength.

    Parameters
    ----------
    lam : float
        The current high-resolution wavelength (nm).
    kabs : ndarray of shape (NL,)
        The gaseous absorption coefficient for the current
        wavelength and for all layers.
    alb : float
        The surface albedo for the current wavelength.
    sik : float
        The virtual LE photon's Stokes component k.
    wi_lr : ndarray of shape (NLR,)
        The virtual LE photon's corrective scattering weights for
        the different LR wavelengths.
    dij : ndarray of shape (NL,)
        The virtual LE photon's cumulative distances traveled in
        layers.
    ki : int
        The number of reflections on the surface.
    lam_lr_grid : ndarray of shape (NLR,)
        The LR wavelengths grid.

    Returns
    -------
    float
        The Beer-Lambert weighted Stokes component.
    """
    # interpolation of scattering weights at low spectral resolution to current lambda
    wi = jnp.interp(lam, lam_lr_grid, wi_lr)

    return sik * wi * jnp.exp(- jnp.sum(dij * kabs)) * alb**ki


def si2(
    lam: float,
    kabs: NDArray[np.floating],
    alb: float,
    sik: float,
    wi_lr: NDArray[np.floating],
    dij: NDArray[np.floating],
    ki: float,
    lam_lr_grid: NDArray[np.floating],
) -> jax.Array:
    """Square of `si`, the Beer-Lambert weighted Stokes component.

    JAX based computation, for one virtual photon, one Stokes
    component and one high-resolution wavelength. Used to
    accumulate the second moment needed for the Monte-Carlo
    variance.

    Parameters
    ----------
    lam : float
        The current high-resolution wavelength (nm).
    kabs : ndarray of shape (NL,)
        The gaseous absorption coefficient for the current
        wavelength and for all layers.
    alb : float
        The surface albedo for the current wavelength.
    sik : float
        The virtual LE photon's Stokes component k.
    wi_lr : ndarray of shape (NLR,)
        The virtual LE photon's corrective scattering weights for
        the different LR wavelengths.
    dij : ndarray of shape (NL,)
        The virtual LE photon's cumulative distances traveled in
        layers.
    ki : int
        The number of reflections on the surface.
    lam_lr_grid : ndarray of shape (NLR,)
        The LR wavelengths grid.

    Returns
    -------
    float
        The square of `si` for the same arguments.
    """
    return si(lam, kabs, alb, sik, wi_lr, dij, ki, lam_lr_grid)**2



def big_sum(
    s: Callable[..., Any],
    grad: int | None = None,
    only_i: bool = False,
) -> Callable[..., Any]:
    """Vectorize and JIT-compile a Stokes-component function.

    Wraps a per-photon, per-wavelength function such as `si` or
    `si2` with `jax.vmap`, so that it can be called once with full
    wavelength, photon and Stokes-component arrays instead of a
    Python loop, then JIT-compiles the result.

    Parameters
    ----------
    s : callable
        A function with the signature of `si`:
        ``s(lam, kabs, alb, sik, wi_lr, dij, ki, lam_lr_grid)``.
    grad : int, optional
        If given, `s` is first replaced with its value and
        gradient with respect to its `grad`-th positional argument
        (`jax.value_and_grad`). Default None (no differentiation).
    only_i : bool, optional
        If True, vectorize only over wavelengths and photons, for
        an `s` that only takes the I Stokes component (`sik` and
        the output are then scalar per photon). If False, also
        vectorize over the Stokes-components axis of `sik`.
        Default False.

    Returns
    -------
    callable
        A JIT-compiled, `jax.vmap`-vectorized version of `s`,
        taking the same arguments with an added leading wavelength
        axis (axis 0 of `lam`, `kabs`, `alb`) and an added photon
        axis (axis 0 of `sik`, `wi_lr`, `dij`, `ki`), plus, unless
        `only_i` is True, a Stokes-components axis (axis 1 of
        `sik`).
    """
    if grad is not None : s = value_and_grad(s, argnums=grad)
    f1m = vmap(s,  in_axes=(0   ,    0,    0, None, None, None, None, None))  # co varying wavelengths inputs
    f2m = vmap(f1m,in_axes=(None, None, None,    0,    0,    0,    0, None)) # co vaying LE photons inputs
    f3m = vmap(f2m,in_axes=(None, None, None,    1, None, None, None, None)) # co varying Stoke components inputs

    if only_i : return jit(f2m)
    else : return jit(f3m)


# ─────────────────────────────────────────────────────────────────────────────
# Post-hoc AMF computation from ALIS photon histories
# ─────────────────────────────────────────────────────────────────────────────

def compute_cdist_hist(
    d_h: NDArray[np.floating],
    s_h: NDArray[np.floating],
    w_h: NDArray[np.floating],
    nref_h: NDArray[np.floating],
    nint_h: NDArray[np.floating],
    nlscl_h: NDArray[np.floating],
    wavelength_lr_r: NDArray[np.floating],
    wavelength_ref: float,
    alb_ref: float,
    natm_abs: int,
    *,
    amf_variance: bool           = True,
    nscl: int                    = 1,
    scatter_classes: str         = 'last_scattering_layer',
    norders: int                 = 1,
    cdist_wabs: bool             = False,
    kabs_ref: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Compute cdist (tabDist) moments from ALIS photon histories.

    Replicates the GPU tabDist accumulation for the Smartg options
    amf_variance, nscl, scatter_classes, norders and cdist_wabs.

    Parameters
    ----------
    d_h : ndarray of shape (NLE, NL)
        Path lengths per absorption layer (km).
    s_h : ndarray of shape (NLE, NStokes)
        Stokes components from the pure-scattering MC run.
    w_h : ndarray of shape (NLE, NLR)
        ALIS LR scattering-correction weights.
    nref_h : ndarray of shape (NLE,)
        Surface-reflection count per photon.
    nint_h : ndarray of shape (NLE,)
        Scattering order (total interaction count), integer-valued.
    nlscl_h : ndarray of shape (NLE,)
        Last-scattering layer index (-1 for surface/none),
        integer-valued.
    wavelength_lr_r : ndarray of shape (NLR,)
        The LR wavelength axis (nm).
    wavelength_ref : float
        The reference wavelength (nm) used to evaluate the
        scattering-correction weight.
    alb_ref : float
        The surface albedo at wavelength_ref.
    natm_abs : int
        The number of atmospheric absorption layers.
    amf_variance : bool, optional
        If True, also store the 3rd moment Σ d²·w. Default True.
    nscl : int, optional
        The number of scatter classes. 1 disables the
        decomposition. Default 1.
    scatter_classes : str, optional
        The scatter-class definition, one of
        'last_scattering_layer', 'scattering_order' or
        'scattering_order_per_layer'. Default
        'last_scattering_layer'.
    norders : int, optional
        The number of scattering-order bins per layer, only used
        when scatter_classes is 'scattering_order_per_layer'.
        Default 1.
    cdist_wabs : bool, optional
        If True, include the Beer-Lambert transmittance in the
        photon weight w_n. Default False.
    kabs_ref : ndarray of shape (natm_abs,) or None, optional
        The absorption coefficient (km⁻¹) at wavelength_ref, used
        when cdist_wabs is True. Default None.

    Returns
    -------
    ndarray, float64
        The cdist moments, of shape (natm_abs, niamf) when
        nscl == 1 or (natm_abs, nscl, niamf) when nscl > 1, where
        niamf is 3 if amf_variance else 2.

    Raises
    ------
    ValueError
        If scatter_classes is not one of the known scatter-class
        definitions.
    """
    niamf = 3 if amf_variance else 2
    nle   = int(d_h.shape[0])

    wavelength_lr_j  = jnp.array(wavelength_lr_r, dtype=jnp.float32)
    wavelength_ref_j = jnp.float32(wavelength_ref)

    # 1. ALIS scattering-correction weight at wavelength_ref
    w_h_j    = jnp.array(w_h,  dtype=jnp.float32)
    w_scalar = vmap(
        lambda wi: jnp.interp(wavelength_ref_j, wavelength_lr_j, wi)
    )(w_h_j)

    # 2. Optional Beer-Lambert transmittance at wavelength_ref
    d_abs = jnp.array(d_h[:, :natm_abs], dtype=jnp.float32)
    if cdist_wabs and kabs_ref is not None:
        k_ref_j = jnp.array(kabs_ref[:natm_abs], dtype=jnp.float32)
        t_abs = jnp.exp(-jnp.sum(d_abs * k_ref_j[None, :], axis=1))
    else:
        t_abs = jnp.ones(nle, dtype=jnp.float32)

    # 3. Effective photon weight  w_n = S_I · wsca · alb^Ki · Tabs
    safe_alb = jnp.float32(alb_ref if float(alb_ref) > 0. else 1.)
    nref_j   = jnp.array(nref_h, dtype=jnp.float32)
    s_h_j    = jnp.array(s_h,    dtype=jnp.float32)
    w_n = s_h_j[:, 0] * w_scalar * jnp.power(safe_alb, nref_j) * t_abs

    # 4. Scatter class index (mirrors SCL_MODE in device.cu)
    nint_np  = np.asarray(nint_h,  dtype=np.int32)
    nlscl_np = np.asarray(nlscl_h, dtype=np.int32)

    if nscl <= 1:
        cls_np = np.zeros(nle, dtype=np.int32)
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
        w_tot = float(jnp.sum(w_n))
        dw    = np.array(jnp.sum(d_abs * w_n[:, None], axis=0))
        cdist_out = np.empty((natm_abs, niamf), dtype=np.float64)
        cdist_out[:, 0] = w_tot
        cdist_out[:, 1] = dw
        if amf_variance:
            cdist_out[:, 2] = np.array(jnp.sum(d_abs ** 2 * w_n[:, None], axis=0))
    else:
        cls_j  = jnp.array(cls_np, dtype=jnp.int32)
        cls_oh = (jnp.arange(nscl, dtype=jnp.int32)[None, :] == cls_j[:, None]).astype(jnp.float32)
        w_cls  = w_n[:, None] * cls_oh
        w_cls_sum = np.array(jnp.sum(w_cls, axis=0))
        dw_cls = np.array(jnp.einsum('il,ic->lc', d_abs, w_cls))
        cdist_out = np.empty((natm_abs, nscl, niamf), dtype=np.float64)
        cdist_out[:, :, 0] = w_cls_sum[None, :]
        cdist_out[:, :, 1] = dw_cls
        if amf_variance:
            d2w_cls = np.array(jnp.einsum('il,ic->lc', d_abs ** 2, w_cls))
            cdist_out[:, :, 2] = d2w_cls

    return cdist_out


def amf_from_cdist(
    cdist: NDArray[np.float64],
    thick: NDArray[np.floating],
) -> dict[str, NDArray[np.float64]]:
    """Derive AMF statistics from a raw cdist moments array.

    Shared by the hist=False (GPU tabDist) and hist=True (post-hoc)
    paths.

    Parameters
    ----------
    cdist : ndarray of shape (NL, niamf) or (NL, nscl, niamf)
        The raw moments: index 0 along the last axis is Σ w, index
        1 is Σ d·w and, when niamf == 3, index 2 is Σ d²·w.
    thick : ndarray of shape (NL,)
        The layer thicknesses (km).

    Returns
    -------
    dict
        A dictionary with the following keys, always present:

        'AMF' : ndarray of shape (NL,)
            The total AMF per layer.
        'std_AMF' : ndarray of shape (NL,)
            The standard deviation of the AMF (zeros when
            niamf < 3).
        'mean_dist' : ndarray of shape (NL,)
            The mean path length (km).
        'W' : ndarray of shape (NL,)
            The total weight.

        and, only when cdist.ndim == 3 (nscl > 1):

        'AMF_cls' : ndarray of shape (NL, nscl)
            The per-class AMF.
        'std_AMF_cls' : ndarray of shape (NL, nscl)
            The per-class standard deviation of the AMF.
        'W_cls' : ndarray of shape (NL, nscl)
            The per-class weight.
        'var_within' : ndarray of shape (NL,)
            The within-class variance.
        'var_between' : ndarray of shape (NL,)
            The between-class variance.
    """
    thick   = np.asarray(thick, dtype=np.float64)
    niamf   = cdist.shape[-1]
    has_scl = (cdist.ndim == 3)

    if has_scl:
        w_cls         = cdist[:, :, 0]
        w             = w_cls.sum(axis=1)
        mean_dist_cls = cdist[:, :, 1] / np.where(w_cls > 0, w_cls, 1.)
        mean_dist     = cdist[:, :, 1].sum(axis=1) / np.where(w > 0, w, 1.)
        amf           = mean_dist / thick
        amf_cls       = mean_dist_cls / thick[:, None]
        result = dict(AMF=amf, W=w, mean_dist=mean_dist, W_cls=w_cls, AMF_cls=amf_cls)
        if niamf >= 3:
            mean_dist2_cls = cdist[:, :, 2] / np.where(w_cls > 0, w_cls, 1.)
            var_cls        = mean_dist2_cls - mean_dist_cls ** 2
            frac_cls       = w_cls / np.where(w > 0, w, 1.)[:, None]
            var_within     = (frac_cls * var_cls).sum(axis=1)
            var_between    = (frac_cls * (mean_dist_cls - mean_dist[:, None]) ** 2).sum(axis=1)
            std_amf        = np.sqrt(np.maximum(var_within + var_between, 0.)) / thick
            std_amf_cls    = np.sqrt(np.maximum(var_cls, 0.)) / thick[:, None]
            result.update(std_AMF=std_amf, std_AMF_cls=std_amf_cls,
                          var_within=var_within, var_between=var_between)
        else:
            result.update(std_AMF=np.zeros_like(amf),
                          std_AMF_cls=np.zeros_like(amf_cls))
    else:
        w         = cdist[:, 0]
        mean_dist = cdist[:, 1] / np.where(w > 0, w, 1.)
        amf       = mean_dist / thick
        result    = dict(AMF=amf, W=w, mean_dist=mean_dist)
        if niamf >= 3:
            mean_dist2    = cdist[:, 2] / np.where(w > 0, w, 1.)
            result['std_AMF'] = np.sqrt(np.maximum(mean_dist2 - mean_dist**2, 0.)) / thick
        else:
            result['std_AMF'] = np.zeros_like(amf)

    return result


def compute_amf(
    m: MLUT | xr.Dataset,
    *,
    wavelength_lr_r: NDArray[np.floating] | None = None,
    wavelength_ref: float | None = None,
    alb_ref: float | None = None,
    natm_abs: int | None = None,
    amf_variance: bool = True,
    nscl: int = 1,
    scatter_classes: str = 'last_scattering_layer',
    norders: int = 1,
    cdist_wabs: bool = False,
    kabs_ref: NDArray[np.floating] | None = None,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64], NDArray[np.float64]]:
    """Compute AMF from a Smartg output.

    Works transparently for hist=False and hist=True runs, by
    dispatching on the output content:

    * 'histories' present -> hist=True: calls compute_cdist_hist.
    * 'cdist_up (TOA)' present -> hist=False: reads the GPU tabDist
      directly.

    Parameters
    ----------
    m : xarray.Dataset
        A Smartg.run() output. MLUT input is deprecated and
        converted with `to_xarray`.
    wavelength_lr_r : ndarray of shape (NLR,), optional
        The LR wavelength axis (nm). Defaults to
        m['wavelength'].values.
    wavelength_ref : float, optional
        The reference wavelength (nm). Defaults to the median of
        wavelength_lr_r.
    alb_ref : float, optional
        The surface albedo at wavelength_ref. Required for the
        hist=True path.
    natm_abs : int, optional
        The number of atmospheric absorption layers. Defaults to
        m['z_atm'].size - 1.
    amf_variance : bool, optional
        If True, include the 3rd moment Σ d²·w, for σ(AMF).
        Default True.
    nscl : int, optional
        The number of scatter classes. Default 1.
    scatter_classes : str, optional
        The scatter-class definition, one of
        'last_scattering_layer', 'scattering_order' or
        'scattering_order_per_layer'. Default
        'last_scattering_layer'.
    norders : int, optional
        The number of scatter-order bins per layer, only used when
        scatter_classes is 'scattering_order_per_layer'. Default 1.
    cdist_wabs : bool, optional
        If True, include the Beer-Lambert transmittance weight in
        w_n. Default False.
    kabs_ref : ndarray of shape (natm_abs,), optional
        The absorption coefficient (km⁻¹) at wavelength_ref, used
        when cdist_wabs is True.

    Returns
    -------
    amf_dict : dict
        The output of `amf_from_cdist`. Keys always present:
        'AMF', 'std_AMF', 'W', 'mean_dist'. Extra keys when
        nscl > 1: 'AMF_cls', 'std_AMF_cls', 'W_cls', 'var_within',
        'var_between'.
    thick : ndarray of shape (NL,)
        The layer thicknesses (km).
    cdist : ndarray of shape (NL, niamf) or (NL, nscl, niamf)
        The raw moments.

    Raises
    ------
    ValueError
        If alb_ref is not given for a hist=True output, or if m
        contains neither 'histories' nor 'cdist_up (TOA)'.
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
        _, s, d, w, _, nref, _, _, _, nint, nlscl = get_histories(m)
        cdist = compute_cdist_hist(
            d, s, w, nref, nint, nlscl,
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
