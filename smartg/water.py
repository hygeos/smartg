#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Preprocessing of oceanic optical properties for SMART-G simulations.

This module provides tools to build and preprocess water column profiles
for use as input to SMART-G radiative transfer simulations. Pure water
absorption and scattering are always present and computed intrinsically
(the water-equivalent of Rayleigh scattering in the atmosphere);
hydrosols (particles, CDOM, phytoplankton...) are added on top of it to
build a complete water model.

Workflow
--------
Typical usage involves:
1. Create a water profile using model classes (e.g., Water1D)
2. Add hydrosol components (chlorophyll-driven models, user-supplied
   inherent optical properties) as needed
3. (Optional) Call the profile's `calc()` method to compute optical
   properties with specific parameters (if using optional parameters not
   set by default)
4. Pass the resulting profile object as the `water` parameter to
   `smartg.run()`

Key Classes
-----------
Water1D
    1D water column profile model. Provides the depth grid, the pure
    water absorption and scattering coefficients (read from auxiliary
    data and from the standard spectral law), and the seafloor albedo.
    Hydrosols can be added to build a complete water model.

Hydrosol
    Hydrosol optical properties supplied directly by the user, i.e. the
    particle and CDOM absorption coefficients, the particle scattering
    coefficient, and either the phase matrices or the backscattering
    ratio from which a Fournier-Forand phase matrix is derived.

HydrosolPR
    Chlorophyll-driven hydrosol model using the Park & Ruddick
    parameterization. Like Hydrosol, but the absorption, scattering and
    backscattering ratio are derived from a chlorophyll concentration
    instead of being supplied by the user.

HydrosolZhai
    Chlorophyll-driven hydrosol model described in Zhai et al. (2017).
    Like HydrosolPR, but the chlorophyll concentration varies with depth,
    following the stratified trophic profile of Uitz et al. (2006).

WaterRw
    Model of water reflectance (lambertian reflector under the surface),
    without any water column optics.
"""

from __future__ import print_function, division, absolute_import
import numpy as np
import xarray as xr
from smartg.atmosphere import diff1
from smartg.albedo import AlbedoCst
from smartg.phase import integ_phase, calc_iphase, fournier_forand
from smartg.bandset import BandSet
from smartg.config import DIR_AUXDATA as dir_aux
from smartg.interp import interp_1d_coord
from smartg.typing import PathType, NumericArrayLike
from pathlib import Path
from numpy.typing import NDArray


def diff2(x: NumericArrayLike) -> NDArray:
    """
    Compute the discrete difference of an array with a zero appended at the end.

    This is equivalent to ``numpy.ediff1d(x, to_end=[0.])``, which computes
    ``x[i+1] - x[i]`` for each element and appends a zero as the final value,
    so the output has the same length as the input.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    ndarray
        The discrete difference array, same shape as `x`, with a zero
        appended at the end.
    """
    return np.ediff1d(x, to_end=[0.])


def _read_aw(dir_aux: PathType) -> xr.DataArray:
    """
    Read pure water absorption coefficient.

    Combines data from [1]_ for wavelengths <= 725 nm and
    [2]_ for wavelengths > 725 nm.
    Values are converted from cm^-1 to m^-1.

    Parameters
    ----------
    dir_aux : path-like
        Path to the auxiliary data directory. Must contain
        ``water/pope97.dat`` and ``water/palmer74.dat``.

    Returns
    -------
    aw : DataArray
        Pure water absorption coefficient [m^-1] as a function of
        wavelength [nm], with dimension ``('wavelength',)``.

    References
    ----------
    .. [1] R. M. Pope and E. S. Fry, "Absorption spectrum (380-700 nm)
       of pure water. II. Integrating cavity measurements,"
       Appl. Opt. 36, 8710-8723 (1997).
       https://doi.org/10.1364/AO.36.008710
    .. [2] K. F. Palmer and D. Williams, "Optical properties of water
       in the near infrared," J. Opt. Soc. Am. 64, 1107-1110 (1974).
       https://doi.org/10.1364/JOSA.64.001107
    """

    # Pope&Fry
    with open(Path(dir_aux) / "water" / "pope97.dat", "rb") as fp:
        for i in range(6):
            fp.readline()  # skip the first 6 lines
        data_pf = np.genfromtxt(fp)
    aw_pf = data_pf[:, 1] * 100  #  convert from cm-1 to m-1
    lam_pf = data_pf[:, 0]
    ok_pf = lam_pf <= 725

    # Palmer&Williams
    data_pw = np.genfromtxt(
        Path(dir_aux) / "water" / "palmer74.dat", skip_header=5
    )
    aw_pw = data_pw[::-1, 1] * 100  #  convert from cm-1 to m-1
    lam_pw = data_pw[::-1, 0]
    ok_pw = lam_pw > 725

    aw = xr.DataArray(
        np.array(list(aw_pf[ok_pf]) + list(aw_pw[ok_pw])),
        dims=["wavelength"],
        coords={
            "wavelength": np.array(list(lam_pf[ok_pf]) + list(lam_pw[ok_pw]))
        },
    )

    return aw


def _expand_phase_4_to_6(phase):
    """
    Convert a 4-term (F11, F21, F33, F34) phase matrix into its 6-term
    equivalent by duplicating F22 = F11 and F44 = F33. Returns the input
    unchanged if it already has 6 terms, or is None.
    """
    if phase is None:
        return None
    if hasattr(phase, 'to_xarray'):
        phase = phase.to_xarray()
    if phase.shape[2] != 4:
        return phase

    pha_6 = np.zeros((phase.shape[0], phase.shape[1], 6, phase.shape[3]), dtype=np.float64)
    pha_6[:,:,0:4,:] = phase[:,:,:,:].copy() # F11, F12, F33, F34
    pha_6[:,:,4,:] = phase[:,:,0,:].copy() # F22 = F11
    pha_6[:,:,5,:] = phase[:,:,2,:].copy() # F44 = F33
    axes = list(phase.dims)
    coords = {}
    for i, dim in enumerate(axes):
        if i == 2:
            coords[dim] = np.arange(6)
        elif dim in phase.coords and phase.coords[dim].size == pha_6.shape[i]:
            coords[dim] = phase.coords[dim].values
        else:
            coords[dim] = np.arange(pha_6.shape[i])
    return xr.DataArray(pha_6, dims=axes, coords=coords)


class Hydrosol(object):
    '''
    Initialize the user-defined hydrosol model

    Parameters
    ----------
    phase : None or DataArray or LUT, optional
        Phase matrices with dimensions [nwav, nz, stk, angle]. If None,
        the phase matrices are derived from `bbp_ratio` (see notes).
    bp : None or 2-D ndarray, optional
        Particle scattering coefficient in m-1, dimensions [nwav, nz]
    ap : None or 2-D ndarray, optional
        Particle absorption coefficient in m-1, dimensions [nwav, nz]
    acdom : None or 2-D ndarray, optional
        CDOM absorption coefficient in m-1, dimensions [nwav, nz]
    bbp_ratio : None or 2-D ndarray, optional
        Backscattering ratio, dimensions [nwav, nz]. Only used if `phase`
        is not provided.
    n_theta : int, optional
        Number of angles of the derived phase matrices
    theta_trunc : float, optional
        Truncation angle in degrees of the derived phase matrices
    pfwav : None or array_like, optional
        Wavelengths at which the phase matrices are calculated. If None,
        they are calculated at all wavelengths.

    Notes
    -----
    When `phase` is not provided, the phase matrices are derived from the
    backscattering ratio `bbp_ratio` following Park & Ruddick (2005), as a
    mixture of two Fournier-Forand phase functions. Their forward peak is
    truncated at `theta_trunc`, and the scattering coefficient `bp` is
    scaled accordingly.

    The pure water absorption and scattering coefficients are not defined
    here but in the Water1D profile, since pure water is always present.

    References
    ----------
    .. [1] Y.-J. Park and K. Ruddick, "Model of remote-sensing
       reflectance including bidirectional effects for case 1 and case 2
       waters," Appl. Opt. 44, 1236-1249 (2005).
    '''

    def __init__(self, phase=None, bp=None, ap=None, acdom=None,
                 bbp_ratio=None, n_theta=721, theta_trunc=5., pfwav=None):
        self.bp = bp
        self.ap = ap
        self.acdom = acdom
        self.bbp_ratio = bbp_ratio
        self._phase = _expand_phase_4_to_6(phase)
        self.n_theta = n_theta
        self.theta_trunc = theta_trunc
        self.pfwav = None if pfwav is None else np.array(pfwav)

        self._pha = None
        self._coef_trunc = None
        self._bsca = None

    def iop(self, wav, z):
        '''
        Inherent optical properties of the hydrosol at the given
        wavelengths and depths.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm
        z : ndarray
            Profile of depths in m

        Returns
        -------
        dict
            Coefficients in m-1 with dimensions [len(wav), len(z)]:
            'ap' and 'acdom' (particle and CDOM absorption), 'bp'
            (particle scattering, before truncation correction), 'aphy'
            (fluorescing absorption) and 'fqyc' (fluorescence quantum
            yield), plus 'bbp_ratio' (backscattering ratio) or None.
        '''
        shp = (len(wav), len(z))
        zeros = np.zeros(shp, dtype='float')

        def as_2d(x):
            if x is None:
                return zeros.copy()
            x = np.asarray(x, dtype='float')
            try:
                return np.broadcast_to(x, shp).copy()
            except ValueError:
                raise ValueError(
                    'Cannot evaluate the hydrosol coefficients over '
                    + f'{len(wav)} wavelengths and {len(z)} depths: the '
                    + f'provided arrays have shape {x.shape}.')

        ap = as_2d(self.ap)
        return {'ap': ap,
                'bp': as_2d(self.bp),
                'acdom': as_2d(self.acdom),
                'bbp_ratio': (None if self.bbp_ratio is None
                              else as_2d(self.bbp_ratio)),
                'aphy': ap,
                'fqyc': zeros.copy(),
                }

    def _trunc_scaling(self):
        '''
        Factor applied to the scattering coefficient to account for the
        truncation of the phase matrix forward peak.
        '''
        return 1.

    def calc_phase(self, wav, z, bbp_ratio):
        '''
        Calculate the phase matrices and the associated truncation
        factor, as a mixture of two Fournier-Forand phase functions
        weighted by the backscattering ratio.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm
        z : ndarray
            Profile of depths in m
        bbp_ratio : 2-D ndarray
            Backscattering ratio, dimensions [len(wav), len(z)]

        Returns
        -------
        pha_da : DataArray
            Phase matrices with dimensions [wav_phase, z_phase, stk,
            theta_oc]
        coef_trunc : DataArray
            Truncation factor with dimensions [wav_phase, z_phase]
        '''
        nwav = len(wav)
        nz = len(z)

        # particles phase function
        # see Park & Ruddick, 05
        # https://odnature.naturalsciences.be/downloads/publications/park_appliedoptics_2005.pdf
        ang = np.linspace(0, np.pi, self.n_theta, dtype='float64')    # angle in radians
        ff1 = fournier_forand(ang, 1.117,3.695)[None,None,:]
        ff2 = fournier_forand(ang, 1.05, 3.259)[None,None,:]

        itronc = int(self.n_theta * self.theta_trunc/180.)
        pha = np.zeros((nwav, nz, 6, self.n_theta), dtype='float64')
        r1 = ((bbp_ratio - 0.002)/0.028)[:,:,None]

        pha[:,:,0,:] = r1*ff1 + (1-r1)*ff2

        # truncate
        pha[:,:,0,:itronc] = pha[:,:,0,itronc][:,:,None]

        pha[:,:,1,:] = 0.
        pha[:,:,2,:] = 0.
        pha[:,:,3,:] = 0.
        pha[:,:,4,:] = pha[:,:,0,:].copy() # P22 = P11
        pha[:,:,5,:] = pha[:,:,2,:].copy() # P44 = P33

        pha[:,:,:,0] = 0.

        # normalize
        integ_ff = integ_phase(ang, pha[:,:,0,:])
        pha *= 2./integ_ff[:,:,None,None]

        pha_da = xr.DataArray(pha,
            dims=['wav_phase', 'z_phase', 'stk', 'theta_oc'],
            coords={'wav_phase': wav, 'z_phase': z, 'theta_oc': np.rad2deg(ang)},
           )
        coef_trunc = xr.DataArray(integ_ff*0.5, dims=['wav_phase', 'z_phase'],
                                  coords={'wav_phase': wav, 'z_phase': z})

        return pha_da, coef_trunc

    def phase(self, wav, z, use_old_calc_iphase=False):
        '''
        Phase matrices of the hydrosol, with dimensions [wav_phase,
        z_phase, stk, theta_oc]. Returns None if the hydrosol does not
        scatter.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm
        z : ndarray
            Profile of depths in m
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).
        '''
        if self._phase is not None:
            return self._phase

        iop = self.iop(wav, z)
        if not (np.asarray(iop['bp']) > 0).any():
            return None
        if iop['bbp_ratio'] is None:
            raise Exception('No phase function nor bbp_ratio has been provided, but bp>0')

        self._resolve_truncation(wav, z, use_old_calc_iphase)
        return self._pha

    def _resolve_truncation(self, wav, z, use_old_calc_iphase=False):
        '''
        Compute the phase matrices at the tabulation wavelengths `pfwav`,
        along with the associated truncation factor. The result is
        memoized, so that the scattering coefficient and the phase
        matrices stay consistent whichever is requested first.
        '''
        if self._coef_trunc is not None:
            return

        wav_pha = wav if self.pfwav is None else self.pfwav
        z = np.asarray(z, dtype='float')
        iop = self.iop(wav_pha, z)
        bbp_ratio, bp = iop['bbp_ratio'], iop['bp']

        # tabulate a single depth if neither the phase matrices nor the
        # scattering coefficient vary vertically, to avoid duplicating
        # the phase matrices
        if (np.allclose(bbp_ratio, bbp_ratio[:,:1])
                and np.allclose(bp, bp[:,:1])):
            sl = slice(0, 1)
        else:
            sl = slice(None)

        self._pha, self._coef_trunc = self.calc_phase(wav_pha, z[sl],
                                                      bbp_ratio[:,sl])
        self._bsca = bp[:,sl] * self._coef_trunc.values * self._trunc_scaling()

    def _coef_trunc_on(self, wav, z, use_old_calc_iphase=False):
        '''
        Truncation factor mapped from the tabulation grid of the phase
        matrices onto the given wavelength and depth grids.
        '''
        # index with ipha, so that each wavelength/depth gets the factor
        # of the phase matrix it is actually assigned to
        _, ipha = calc_iphase(self._pha, np.asarray(wav), np.asarray(z),
                              use_old_calc_iphase)
        return self._coef_trunc.values.ravel()[ipha]

    def scattering(self, pha):
        '''
        Scattering coefficient in m-1 of the hydrosol, on the tabulation
        grid of the given phase matrices `pha`, i.e. with dimensions
        [wav_phase, z_phase]. Used to weight the hydrosols when averaging
        their phase matrices.
        '''
        if self._phase is None:
            return self._bsca
        return self.iop(pha.coords['wav_phase'].values,
                        pha.coords['z_phase'].values)['bp']

    def coeffs(self, wav, z, phase=True, use_old_calc_iphase=False):
        '''
        Inherent optical properties of the hydrosol, with the scattering
        coefficient corrected for the phase matrix truncation.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm
        z : ndarray
            Profile of depths in m
        phase : bool, optional
            Whether the phase matrices are calculated. If False, no
            truncation correction is applied.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        dict
            Same as the `iop` method, with 'bp' scaled by the truncation
            factor.
        '''
        iop = self.iop(wav, z)

        if (self._phase is None) and (np.asarray(iop['bp']) > 0).any():
            if iop['bbp_ratio'] is None:
                raise Exception('No phase function nor bbp_ratio has been provided, but bp>0')
            if phase:
                self._resolve_truncation(wav, z, use_old_calc_iphase)
                coef_trunc = self._coef_trunc_on(wav, z, use_old_calc_iphase)
                iop['bp'] = iop['bp'] * coef_trunc * self._trunc_scaling()

        return iop


class HydrosolPR(Hydrosol):
    '''
    Initialize the chlorophyll-driven hydrosol model, using a similar IOP
    parameterization as Polymer's PR model.

    Parameters
    ----------
    chl : float
        Chlorophyll concentration in mg/m3
    n_theta : int, optional
        Number of angles of the derived phase matrices
    theta_trunc : float, optional
        Truncation angle in degrees of the derived phase matrices
    pfwav : None or array_like, optional
        Wavelengths at which the phase matrices are calculated. If None,
        they are calculated at all wavelengths.
    fqyc : float, optional
        Chlorophyll a fluorescence quantum yield

    Examples
    --------
    >>> from smartg.water import Water1D, HydrosolPR
    >>> water = Water1D(grid=[0, -5.], comp=[HydrosolPR(chl=0.5)])
    '''

    def __init__(self, chl, n_theta=72001, theta_trunc=5., pfwav=None,
                 fqyc=0.0):
        super().__init__(n_theta=n_theta, theta_trunc=theta_trunc, pfwav=pfwav)
        self.chl = chl
        self.fqyc = fqyc

        # Bricaud (98)
        ap_bricaud = np.genfromtxt(dir_aux / 'water' / 'aph_bricaud_1998.txt',
                                   delimiter=',', skip_header=12)  # header is lambda,Ap,Ep,Aphi,Ephi
        self.bricaud = xr.Dataset()
        self.bricaud = self.bricaud.assign_coords(wav=ap_bricaud[:,0])
        self.bricaud['A'] = xr.DataArray(ap_bricaud[:,1], dims=['wav'])
        self.bricaud['E'] = xr.DataArray(1-ap_bricaud[:,2], dims=['wav'])

    def _trunc_scaling(self):
        return 0.5

    def iop(self, wav, z):
        '''
        Inherent optical properties calculation. The chlorophyll
        concentration does not vary with depth, so the coefficients are
        broadcast over the depth profile.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm
        z : ndarray
            Profile of depths in m
        '''
        wav = np.asarray(wav, dtype='float')
        chl = self.chl

        # phytoplankton absorption
        aphy = (interp_1d_coord(self.bricaud['A'], 'wav', wav, extrema=True)
            * (chl**interp_1d_coord(self.bricaud['E'], 'wav', wav, extrema=True)))

        # chlorophyll fluorescence (scattering coefficient)
        fqyc = np.full_like(aphy, self.fqyc) # Fluorescence Quantum Yield for Chlorophyll
        fqyc[wav<370.]=0.
        fqyc[wav>690.]=0.

        # CDM absorption central value
        # from Bricaud et al GBC, 2012 (data from nov 2007)
        fa = 1.
        acdm443 = fa * 0.069 * (chl**1.070)

        s_cdom = 0.00262*(acdm443**(-0.448))
        if (s_cdom > 0.025): s_cdom=0.025
        if (s_cdom < 0.011): s_cdom=0.011

        acdm = acdm443 * np.exp(-s_cdom*(wav - 443))

        bp = 0.416*(chl**0.766)*550./wav

        #
        # backscattering coefficient
        #
        if chl < 2:
            v = 0.5*(np.log10(chl) - 0.3)
        else:
            v = 0
        bbp_ratio = 0.002 + 0.01*( 0.5-0.25*np.log10(chl))*((wav/550.)**v)

        shp = (len(wav), len(z))

        def as_2d(x):
            return np.broadcast_to(x[:,None], shp).copy()

        aphy_2d = as_2d(aphy)
        return {'ap': aphy_2d,
                'bp': as_2d(bp),
                'acdom': as_2d(acdm),
                'bbp_ratio': as_2d(bbp_ratio),
                'aphy': aphy_2d,
                'fqyc': as_2d(fqyc),
                }


class HydrosolZhai(Hydrosol):
    '''
    Initialize the chlorophyll-driven hydrosol model described in Zhai et
    al. (2017), where the chlorophyll concentration varies with depth.

    Parameters
    ----------
    chl_surf : float
        Chlorophyll concentration in mg/m3 at the surface. The
        concentration at depth is derived from it (see notes), so this is
        the surface value only, unlike the depth-independent `chl` of
        HydrosolPR.
    n_theta : int, optional
        Number of angles of the derived phase matrices
    theta_trunc : float, optional
        Truncation angle in degrees of the derived phase matrices
    pfwav : None or array_like, optional
        Wavelengths at which the phase matrices are calculated. If None,
        they are calculated at all wavelengths.
    euphotic_depth : float or None, optional
        Euphotic depth in m, noted Z_eu in [2]_. This is a positive
        depth, measured downwards from the surface, and not a z
        coordinate: unlike the `grid` of Water1D, it does not follow the
        convention of being negative below the surface. If None, it is
        computed from the chlorophyll climatology of [2]_.
    mixed : bool, optional
        Mixed or stratified waters
    fqyc : float, optional
        Chlorophyll a fluorescence quantum yield

    Notes
    -----
    The chlorophyll profile is a continuous function of depth, so the
    coefficients are evaluated at whichever depths the Water1D profile
    provides.

    References
    ----------
    .. [1] P.-W. Zhai, Y. Hu, D. M. Winker, B. A. Franz, J. Werdell, and
       Y. Chen, "Vector radiative transfer model for coupled
       atmosphere and ocean systems including inelastic sources in
       ocean waters," Opt. Express 25, A223-A239 (2017).
    .. [2] J. Uitz, H. Claustre, A. Morel, and S. B. Hooker, "Vertical
       distribution of phytoplankton communities in open ocean: An
       assessment based on surface chlorophyll," J. Geophys. Res. 111,
       C08005 (2006).
    '''

    def __init__(self, chl_surf, n_theta=7201, theta_trunc=5., pfwav=None,
                 euphotic_depth=None, mixed=False, fqyc=0.):
        super().__init__(n_theta=n_theta, theta_trunc=theta_trunc, pfwav=pfwav)
        self.chl_surf = chl_surf
        self.fqyc = fqyc

        # Bricaud (98)
        # Absorption of the phytoplankton
        ap_bricaud = np.genfromtxt(dir_aux / 'water' / 'aph_bricaud_1998.txt',
                                   delimiter=',', skip_header=12)  # header is lambda,Ap,Ep,Aphi,Ephi
        # Add extension to 360 nm (Wei et al., 2016)
        # spectral slope of aph is symetrical wrt 440 nm in the 360-520 spectral range
        w_uv      = np.linspace(360., 398., num=20)
        a_bricaud = ap_bricaud[:,1]
        e_bricaud = 1.-ap_bricaud[:,2]
        w         = ap_bricaud[:,0]
        ii        = np.where((w<=520.) & (w>480.))
        a_uv      = np.zeros_like(w_uv)
        e_uv      = np.zeros_like(w_uv)
        a_uv[::-1] = a_bricaud[ii]
        e_uv[::-1] = e_bricaud[ii]
        self.bricaud = xr.Dataset()
        self.bricaud = self.bricaud.assign_coords(wav=np.concatenate((w_uv,ap_bricaud[:,0])))
        self.bricaud['A'] = xr.DataArray(np.concatenate((a_uv,a_bricaud)),
                                         dims=['wav'])
        self.bricaud['E'] = xr.DataArray(np.concatenate((e_uv,e_bricaud)),
                                         dims=['wav'])

        # Chlorophyll integrated over the euphotic column, from which the
        # euphotic depth is derived when it is not provided
        if euphotic_depth is None:
            if not mixed:
                if (chl_surf > 1.): chl_euphotic = 37.7*chl_surf**0.615
                else: chl_euphotic = 36.1*chl_surf**0.357
            else:
                chl_euphotic = 42.1*chl_surf**0.538
            euphotic_depth = 568.2*chl_euphotic**(-0.746)
        self.euphotic_depth = euphotic_depth

        # Reduced concentration chi and reduced depth zeta,
        # stratified trophic case 1 parametrization (Uitz et al., 2006)
        self.chi_b    = 0.471
        self.s        = 0.135
        self.chi_max  = 1.572
        self.zeta_max = 0.969
        self.dzeta    = 0.393

    def chi(self, zeta):
        return self.chi_b - self.s*zeta + self.chi_max*np.exp(-((zeta-self.zeta_max)/self.dzeta)**2)

    def chl(self, z):
        '''
        Chlorophyll concentration in mg/m3 at the depths z (in m)
        '''
        zeta = np.abs(np.asarray(z, dtype='float')/self.euphotic_depth)
        chl = self.chl_surf*self.chi(zeta)/self.chi(0.)
        return np.where(chl < 0., 1e-8, chl)

    def _trunc_scaling(self):
        return 0.5

    def iop(self, wav, z, p1=0.33, r1=0.5, r2=0.5):
        '''
        Inherent optical properties calculation

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm
        z : ndarray
            Profile of depths in m
        p1, r1 : float, optional
            Parameters related to particles extinction, see Zhai et al. 2017
        r2 : float, optional
            Parameter related to CDOM absorption, see Zhai et al. 2017
        '''
        wav = np.asarray(wav, dtype='float')
        chl2, wav2 = np.meshgrid(self.chl(z), wav)

        # specific phytoplankton absorption
        chl2star=np.full_like(chl2, 1.)
        aphystar = (interp_1d_coord(self.bricaud['A'], 'wav', wav2, extrema=True)
            * (chl2star**interp_1d_coord(self.bricaud['E'], 'wav', wav2, extrema=True)))
        aphy = aphystar * chl2
        aphystar440 = (interp_1d_coord(self.bricaud['A'], 'wav', 440., extrema=True)
            * (chl2star**interp_1d_coord(self.bricaud['E'], 'wav', wav2, extrema=True)))
        aphy440 = aphystar440 * chl2

        # phytoplankton covariant particles extinction
        piz440=0.68
        bp440 = aphy440 * piz440/(1-piz440)
        bp = bp440 *(wav2/440.)**(-1.)

        # chlorophyll fluorescence (scattering coefficient)
        fqyc = np.full_like(aphy, self.fqyc) # Fluorescence Quantum Yield for Chlorophyll
        fqyc[wav2<370.]=0.
        fqyc[wav2>690.]=0.

        # CDOM covariant absorption
        acdm440 = 0.24*aphy440**0.43
        s_cdom=0.02
        acdom = acdm440 * np.exp(-s_cdom*(wav2 - 440))

        # non-algal particles backscattering
        spm = 0. # g/m3
        gamma=0.5
        bbpnap650 = 10**(1.03*np.log10(spm) - 2.06) # Neukermans et al 2012
        bbpnap = bbpnap650*(wav2/650.)**(-gamma)
        bbp_ratio_nap = np.zeros_like(aphy)
        bbp_ratio_nap[:] = 0.04
        bpnap  = bbpnap/bbp_ratio_nap
        bp += bpnap

        return {'ap': aphy,
                'bp': bp,
                'acdom': acdom,
                'bbp_ratio': bbp_ratio_nap,
                'aphy': aphy,
                'fqyc': fqyc,
                }


class Water(object):
    """Base class for water."""

    pass


class Water1D(Water):
    '''
    1D water column profile definition

    Pure water absorption and scattering are always present and computed
    here; hydrosols are added through the `comp` parameter.

    Parameters
    ----------
    grid : array_like, optional
        Vertical grid of the water column, in m, from the surface down to
        the sea floor. These are z coordinates, not depths: z is 0 at the
        surface and becomes more negative downwards, so the grid must be
        decreasing (e.g. [0., -2.5, -5.]). This is the oceanic
        counterpart of the `grid` parameter of `Atm1D`, and it defines
        the `z_oc` coordinate of the profile returned by `calc()`.
        Note that the first item of the grid is not used.
    comp : list, optional
        Hydrosols to consider, i.e. a list of Hydrosol, HydrosolPR or/and
        HydrosolZhai objects.
    aw : None or 2-D ndarray, optional
        Force the pure water absorption coefficient in m-1, with
        dimensions [nwav, nz]. If None, it is read from the auxiliary
        data (see `_read_aw`).
    bw : None or 2-D ndarray, optional
        Force the pure water scattering coefficient in m-1, with
        dimensions [nwav, nz]. If None, it is computed as
        19.3e-4*(wav/550)**-4.3.
    alb : albedo object, optional
        Albedo of the sea floor, i.e. of the reflector placed at the
        bottom of the water column, at the deepest level of `grid`. This
        is the reflectance of the sea bottom seen from within the water,
        and it is not related to the albedo of the air-water interface:
        the latter is set by the `surf` parameter of `smartg.run()`
        (e.g. `LambSurface(ALB=...)` or `RoughSurface(...)`). It fills
        the `albedo_seafloor` variable of the profile returned by
        `calc()`.

    Examples
    --------
    >>> from smartg.water import Water1D, HydrosolPR
    >>> water = Water1D(comp=[HydrosolPR(chl=0.5)])
    '''

    def __init__(self, grid=[0, -10000], comp=None, aw=None, bw=None,
                 alb=AlbedoCst(0.)):
        self.grid = np.array(grid, dtype='float')
        self.comp = [] if comp is None else comp
        self.aw = aw
        self.bw = bw
        self.alb = alb

        self.aw_table = _read_aw(dir_aux)

    def calc(self, wav, phase=True, use_old_calc_iphase=False):
        '''
        Profile and phase matrix calculation at bands / wav

        Parameters
        ----------
        wav : array_like or BandSet
            Wavelengths at which to calculate the profile.
        phase : bool, optional
            Whether to calculate the phase matrices.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        out : Dataset
            An xarray Dataset object with the profile and (if phase =
            True) the phase matrices.
        '''
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)
        wav = np.array(wav)

        z = self.grid
        shp = (len(wav), len(z))
        wav2 = np.stack([wav]*len(z), axis=1)

        #
        # pure water absorption and scattering
        #
        if self.aw is None:
            aw = interp_1d_coord(self.aw_table, 'wavelength', wav2)
        else:
            aw = self.aw

        if self.bw is None:
            bw = 19.3e-4*((wav2/550.)**-4.3)
        else:
            bw = self.bw

        #
        # hydrosols absorption and scattering
        #
        ap = np.zeros(shp, dtype='float')
        bp = np.zeros(shp, dtype='float')
        acdom = np.zeros(shp, dtype='float')
        aphy_fluo = np.zeros(shp, dtype='float')
        aphy = np.zeros(shp, dtype='float')

        for comp in self.comp:
            iop = comp.coeffs(wav, z, phase=phase,
                              use_old_calc_iphase=use_old_calc_iphase)
            ap += iop['ap']
            bp += iop['bp']
            acdom += iop['acdom']
            aphy += iop['aphy']
            aphy_fluo += iop['aphy'] * iop['fqyc']

        # the fluorescing fraction of the phytoplankton absorption is
        # counted as (inelastic) scattering instead of absorption
        atot = aw + ap - aphy_fluo + acdom
        btot = bw + bp + aphy_fluo

        pro = xr.Dataset()
        pro = pro.assign_coords(wavelength=wav[:], z_oc=z)

        pro['T_oc'] = xr.DataArray(np.array([280.]*len(z), dtype='float32'), dims=['z_oc'])

        #
        # phase matrices
        #
        if phase:
            pha = self.phase(wav, use_old_calc_iphase=use_old_calc_iphase)

            if pha is not None:
                pha_, ipha = calc_iphase(pha, pro.coords['wavelength'].values,
                                         pro.coords['z_oc'].values, use_old_calc_iphase)

                pro = pro.assign_coords(theta_oc=pha.coords['theta_oc'].values)
                pro['phase_oc'] = xr.DataArray(pha_, dims=['iphase', 'stk', 'theta_oc'])
                pro['iphase_oc'] = xr.DataArray(ipha, dims=['wavelength', 'z_oc'])

        dz = - diff1(z)
        tau_w   = - (aw   + bw  ) * dz
        tau_p   = - (ap   + bp  ) * dz
        tau_y   = - (acdom      ) * dz
        tau_tot = - (atot + btot) * dz
        tau_sca = - (btot       ) * dz
        tau_abs = - (atot       ) * dz
        tau_ine = - (aphy_fluo  ) * dz
        tau_phy = - (aphy       ) * dz

        with np.errstate(invalid='ignore'):
            ssa_w   = bw/(aw   + bw)
        ssa_w[np.isnan(ssa_w)] = 1.

        with np.errstate(invalid='ignore'):
            ssa_p   = bp/(ap + bp)
        ssa_p[np.isnan(ssa_p)] = 1.

        with np.errstate(invalid='ignore'):
            pmol    = bw/(bw   + bp)
        pmol[np.isnan(pmol)]   = 1.
        pmol[~np.isfinite(pmol)] = 1.

        with np.errstate(invalid='ignore', divide='ignore'):
            pine = tau_ine/tau_sca
        pine[np.isnan(pine)] = 0.
        pine[~np.isfinite(pine)] = 0.

        with np.errstate(invalid='ignore'):
            ssa = tau_sca/tau_tot
        ssa[np.isnan(ssa)] = 1.

        # ratio of the fluorescing to the total phytoplankton absorption,
        # i.e. the fluorescence quantum yield weighted over the hydrosols
        with np.errstate(invalid='ignore', divide='ignore'):
            fqy1 = tau_ine/tau_phy
        fqy1[np.isnan(fqy1)] = 0.
        fqy1[~np.isfinite(fqy1)] = 0.

        pro['OD_w'] = xr.DataArray(np.cumsum(tau_w, out=tau_w, axis=1),
                        dims=['wavelength', 'z_oc'],
                        attrs={'description':
                               'Cumulated water optical thickness at each wavelength'})

        pro['OD_p_oc'] = xr.DataArray(np.cumsum(tau_p, out=tau_p, axis=1),
                        dims=['wavelength', 'z_oc'],
                        attrs={'description':
                               'Cumulated oceanic particles optical thickness at each wavelength'})

        pro['OD_y'] = xr.DataArray(np.cumsum(tau_y, out=tau_y, axis=1),
                        dims=['wavelength', 'z_oc'],
                        attrs={'description':
                               'Cumulated CDOM optical thickness at each wavelength'})

        pro['OD_oc'] = xr.DataArray(np.cumsum(tau_tot, out=tau_tot, axis=1),
                        dims=['wavelength', 'z_oc'])

        pro['OD_sca_oc'] = xr.DataArray(np.cumsum(tau_sca, out=tau_sca, axis=1),
                        dims=['wavelength', 'z_oc'])

        pro['OD_abs_oc'] = xr.DataArray(np.cumsum(tau_abs, out=tau_abs, axis=1),
                        dims=['wavelength', 'z_oc'])

        pro['pine_oc'] = xr.DataArray(pine,
                        dims=['wavelength', 'z_oc'])

        pro['pmol_oc'] = xr.DataArray(pmol,
                        dims=['wavelength', 'z_oc'])

        pro['ssa_oc'] = xr.DataArray(ssa,
                        dims=['wavelength', 'z_oc'])

        pro['ssa_p_oc'] = xr.DataArray(ssa_p,
                        dims=['wavelength', 'z_oc'])
        pro['ssa_w'] = xr.DataArray(ssa_w,
                        dims=['wavelength', 'z_oc'])

        pro['FQY1_oc'] = xr.DataArray(fqy1,
                        dims=['wavelength', 'z_oc'])

        pro['albedo_seafloor'] = xr.DataArray(self.alb.get(wav),
                        dims=['wavelength'])

        return pro

    def phase(self, wav, use_old_calc_iphase=False):
        '''
        Calculate the phase matrices of the hydrosols, averaged over the
        hydrosols and weighted by their scattering coefficient.

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (deprecated).

        Returns
        -------
        out : DataArray or None
            The phase matrices with dimensions [wav_phase, z_phase, stk,
            theta_oc], or None if no hydrosol scatters.
        '''
        z = self.grid

        phases = []
        for comp in self.comp:
            pha = comp.phase(wav, z, use_old_calc_iphase=use_old_calc_iphase)
            if pha is not None:
                phases.append((comp, pha))

        if len(phases) == 0:
            return None
        if len(phases) == 1:
            return phases[0][1]

        ref = phases[0][1]
        for _, pha in phases[1:]:
            for dim in ['wav_phase', 'z_phase', 'theta_oc']:
                if not np.array_equal(pha.coords[dim].values,
                                      ref.coords[dim].values):
                    raise ValueError(
                        'The phase matrices of the hydrosols must share the '
                        + f'same {dim} grid to be averaged. Use a common '
                        + 'pfwav, or provide the phase matrices directly.')

        pha_tot = 0.
        bsca = 0.
        for comp, pha in phases:
            # weight each hydrosol by its scattering coefficient, on the
            # tabulation grid of its phase matrices
            bsca_ = xr.DataArray(
                comp.scattering(pha),
                dims=['wav_phase', 'z_phase'],
                coords={'wav_phase': pha.coords['wav_phase'].values,
                        'z_phase': pha.coords['z_phase'].values})
            bsca = bsca + bsca_
            pha_tot = pha_tot + pha * bsca_

        with np.errstate(divide='ignore', invalid='ignore'):
            pha_tot = pha_tot/bsca
        return pha_tot.fillna(0.)


class WaterRw(Water):
    '''
    Initialize the water reflectance model

    The water is defined as a lambertian reflector placed just below the
    air-water interface, without any water column: the water body has
    neither geometric nor optical thickness.

    Parameters
    ----------
    alb : albedo object
        Albedo of the lambertian reflector, i.e. the water reflectance
        just below the surface. Although it is passed as an albedo (it is
        implemented as a lambertian reflector), the quantity to supply
        here is the subsurface irradiance reflectance R(0-) =
        Eu(0-)/Ed(0-). Do not supply a water-leaving reflectance such as
        rho_w or Rrs: those are defined above the interface, at the 0+
        level, and the air-water transmission would then be counted
        twice (see notes). Use an AlbedoSpectrum object to supply a
        spectrally varying R(0-).

    Notes
    -----
    This gives the reflectance at the 0- level, just below the interface,
    which is not the same as a lambertian surface at the 0+ level, just
    above it: here the photons still cross the air-water interface, so
    the Fresnel transmission and the total internal reflection of the
    upwelling light are still accounted for by the `surf` parameter of
    `smartg.run()`.

    Note that, unlike the `alb` of Water1D, this reflector is not a sea
    floor: it stands for the water body itself, and it is placed at the
    top of the water column rather than at its bottom.

    Being lambertian, the reflector is isotropic, whereas the upwelling
    field of real water is not (Q = Eu/Lu differs from pi). The angular
    shape of the reflected light is therefore an approximation, even
    when the magnitude of R(0-) is exact.

    The same model can be obtained with an empty Water1D profile of null
    thickness:

    >>> Water1D(grid=[0., 0.], comp=[], alb=alb)

    Both give the same optical thicknesses, single scattering albedo and
    seafloor albedo (pure water drops out on its own, since the layer has
    no thickness), but WaterRw is faster: it neither reads the pure water
    absorption auxiliary data nor computes any phase matrix.

    Examples
    --------
    >>> from smartg.water import WaterRw
    >>> from smartg.albedo import AlbedoCst
    >>> water = WaterRw(alb=AlbedoCst(0.05))
    '''

    def __init__(self, alb):
        self.alb = alb

    def calc(self, wav):
        '''
        Profile and phase function calculation at bands wav (nm)
        '''
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)
        wav = np.array(wav)

        pro = xr.Dataset()
        pro = pro.assign_coords(wavelength=wav[:], z_oc=np.zeros(2))
        shp = (len(wav), 2)

        pro['T_oc'] = xr.DataArray(np.array([280., 280.], dtype='float32'), dims=['z_oc'])
        pro['OD_oc'] = xr.DataArray(np.zeros(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['OD_w'] = xr.DataArray(np.zeros(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['OD_p_oc'] = xr.DataArray(np.zeros(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['OD_sca_oc'] = xr.DataArray(np.zeros(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['OD_abs_oc'] = xr.DataArray(np.zeros(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['OD_y'] = xr.DataArray(np.zeros(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['pmol_oc'] = xr.DataArray(np.ones(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['pine_oc'] = xr.DataArray(np.ones(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['FQY1_oc'] = xr.DataArray(np.ones(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['ssa_oc'] = xr.DataArray(np.ones(shp, dtype='float32'), dims=['wavelength', 'z_oc'])
        pro['albedo_seafloor'] = xr.DataArray(self.alb.get(wav), dims=['wavelength'])

        return pro
