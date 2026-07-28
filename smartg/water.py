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


class Water(object):
    """Base class for water."""

    pass


class Hydrosol(object):
    '''
    Initialize the user-defined hydrosol model

    Parameters
    ----------
    phase : None or DataArray or LUT, optional
        Phase matrices with dimensions [nwav, nz, stk, angle]. If None,
        the phase matrices are derived from `Bp` (see notes).
    bp : None or 2-D ndarray, optional
        Particle scattering coefficient in m-1, dimensions [nwav, nz]
    ap : None or 2-D ndarray, optional
        Particle absorption coefficient in m-1, dimensions [nwav, nz]
    aCDOM : None or 2-D ndarray, optional
        CDOM absorption coefficient in m-1, dimensions [nwav, nz]
    Bp : None or 2-D ndarray, optional
        Backscattering ratio, dimensions [nwav, nz]. Only used if `phase`
        is not provided.
    NANG : int, optional
        Number of angles of the derived phase matrices
    ang_trunc : float, optional
        Truncation angle in degrees of the derived phase matrices
    pfwav : None or array_like, optional
        Wavelengths at which the phase matrices are calculated. If None,
        they are calculated at all wavelengths.

    Notes
    -----
    When `phase` is not provided, the phase matrices are derived from the
    backscattering ratio `Bp` following Park & Ruddick (2005), as a
    mixture of two Fournier-Forand phase functions. Their forward peak is
    truncated at `ang_trunc`, and the scattering coefficient `bp` is
    scaled accordingly.

    The pure water absorption and scattering coefficients are not defined
    here but in the Water1D profile, since pure water is always present.

    References
    ----------
    .. [1] Y.-J. Park and K. Ruddick, "Model of remote-sensing
       reflectance including bidirectional effects for case 1 and case 2
       waters," Appl. Opt. 44, 1236-1249 (2005).
    '''

    def __init__(self, phase=None, bp=None, ap=None, aCDOM=None, Bp=None,
                 NANG=721, ang_trunc=5., pfwav=None):
        self.bp = bp
        self.ap = ap
        self.aCDOM = aCDOM
        self.Bp = Bp
        self._phase = _expand_phase_4_to_6(phase)
        self.NANG = NANG
        self.ang_trunc = ang_trunc
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
            'ap' and 'aCDOM' (particle and CDOM absorption), 'bp'
            (particle scattering, before truncation correction), 'aphy'
            (fluorescing absorption) and 'FQYC' (fluorescence quantum
            yield), plus 'Bp' (backscattering ratio) or None.
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
                'aCDOM': as_2d(self.aCDOM),
                'Bp': None if self.Bp is None else as_2d(self.Bp),
                'aphy': ap,
                'FQYC': zeros.copy(),
                }

    def _trunc_scaling(self):
        '''
        Factor applied to the scattering coefficient to account for the
        truncation of the phase matrix forward peak.
        '''
        return 1.

    def calc_phase(self, wav, z, Bp):
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
        Bp : 2-D ndarray
            Backscattering ratio, dimensions [len(wav), len(z)]

        Returns
        -------
        P : DataArray
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
        ang = np.linspace(0, np.pi, self.NANG, dtype='float64')    # angle in radians
        ff1 = fournier_forand(ang, 1.117,3.695)[None,None,:]
        ff2 = fournier_forand(ang, 1.05, 3.259)[None,None,:]

        itronc = int(self.NANG * self.ang_trunc/180.)
        pha = np.zeros((nwav, nz, 6, self.NANG), dtype='float64')
        r1 = ((Bp - 0.002)/0.028)[:,:,None]

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

        P = xr.DataArray(pha,
            dims=['wav_phase', 'z_phase', 'stk', 'theta_oc'],
            coords={'wav_phase': wav, 'z_phase': z, 'theta_oc': np.rad2deg(ang)},
           )
        coef_trunc = xr.DataArray(integ_ff*0.5, dims=['wav_phase', 'z_phase'],
                                  coords={'wav_phase': wav, 'z_phase': z})

        return P, coef_trunc

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
        if iop['Bp'] is None:
            raise Exception('No phase function nor Bp has been provided, but bp>0')

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
        Bp, bp = iop['Bp'], iop['bp']

        # tabulate a single depth if neither the phase matrices nor the
        # scattering coefficient vary vertically, to avoid duplicating
        # the phase matrices
        if np.allclose(Bp, Bp[:,:1]) and np.allclose(bp, bp[:,:1]):
            sl = slice(0, 1)
        else:
            sl = slice(None)

        self._pha, self._coef_trunc = self.calc_phase(wav_pha, z[sl], Bp[:,sl])
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
            if iop['Bp'] is None:
                raise Exception('No phase function nor Bp has been provided, but bp>0')
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
    NANG : int, optional
        Number of angles of the derived phase matrices
    ang_trunc : float, optional
        Truncation angle in degrees of the derived phase matrices
    pfwav : None or array_like, optional
        Wavelengths at which the phase matrices are calculated. If None,
        they are calculated at all wavelengths.
    FQYC : float, optional
        Chlorophyll a fluorescence quantum yield

    Examples
    --------
    >>> from smartg.water import Water1D, HydrosolPR
    >>> water = Water1D(Z=[0, -5.], comp=[HydrosolPR(chl=0.5)])
    '''

    def __init__(self, chl, NANG=72001, ang_trunc=5., pfwav=None, FQYC=0.0):
        super().__init__(NANG=NANG, ang_trunc=ang_trunc, pfwav=pfwav)
        self.chl = chl
        self.FQYC = FQYC

        # Bricaud (98)
        ap_bricaud = np.genfromtxt(dir_aux / 'water' / 'aph_bricaud_1998.txt',
                                   delimiter=',', skip_header=12)  # header is lambda,Ap,Ep,Aphi,Ephi
        self.BRICAUD = xr.Dataset()
        self.BRICAUD = self.BRICAUD.assign_coords(wav=ap_bricaud[:,0])
        self.BRICAUD['A'] = xr.DataArray(ap_bricaud[:,1], dims=['wav'])
        self.BRICAUD['E'] = xr.DataArray(1-ap_bricaud[:,2], dims=['wav'])

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
        aphy = (interp_1d_coord(self.BRICAUD['A'], 'wav', wav, extrema=True)
            * (chl**interp_1d_coord(self.BRICAUD['E'], 'wav', wav, extrema=True)))

        # chlorophyll fluorescence (scattering coefficient)
        FQYC = np.full_like(aphy, self.FQYC) # Fluorescence Quantum Yield for Chlorophyll
        FQYC[wav<370.]=0.
        FQYC[wav>690.]=0.

        # CDM absorption central value
        # from Bricaud et al GBC, 2012 (data from nov 2007)
        fa = 1.
        aCDM443 = fa * 0.069 * (chl**1.070)

        S = 0.00262*(aCDM443**(-0.448))
        if (S > 0.025): S=0.025
        if (S < 0.011): S=0.011

        aCDM = aCDM443 * np.exp(-S*(wav - 443))

        bp = 0.416*(chl**0.766)*550./wav

        #
        # backscattering coefficient
        #
        if chl < 2:
            v = 0.5*(np.log10(chl) - 0.3)
        else:
            v = 0
        Bp = 0.002 + 0.01*( 0.5-0.25*np.log10(chl))*((wav/550.)**v)

        shp = (len(wav), len(z))

        def as_2d(x):
            return np.broadcast_to(x[:,None], shp).copy()

        aphy_2d = as_2d(aphy)
        return {'ap': aphy_2d,
                'bp': as_2d(bp),
                'aCDOM': as_2d(aCDM),
                'Bp': as_2d(Bp),
                'aphy': aphy_2d,
                'FQYC': as_2d(FQYC),
                }


class HydrosolZhai(Hydrosol):
    '''
    Initialize the chlorophyll-driven hydrosol model described in Zhai et
    al. (2017), where the chlorophyll concentration varies with depth.

    Parameters
    ----------
    chls : float
        Chlorophyll concentration in mg/m3 at the surface
    NANG : int, optional
        Number of angles of the derived phase matrices
    ang_trunc : float, optional
        Truncation angle in degrees of the derived phase matrices
    pfwav : None or array_like, optional
        Wavelengths at which the phase matrices are calculated. If None,
        they are calculated at all wavelengths.
    Zeu : float or None, optional
        Euphotic depth in m. If None, computed from climatology.
    MIXED : bool, optional
        Mixed or stratified waters
    FQYC : float, optional
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

    def __init__(self, chls, NANG=7201, ang_trunc=5., pfwav=None,
                 Zeu=None, MIXED=False, FQYC=0.):
        super().__init__(NANG=NANG, ang_trunc=ang_trunc, pfwav=pfwav)
        self.chls = chls
        self.FQYC = FQYC

        # Bricaud (98)
        # Absorption of the phytoplankton
        ap_bricaud = np.genfromtxt(dir_aux / 'water' / 'aph_bricaud_1998.txt',
                                   delimiter=',', skip_header=12)  # header is lambda,Ap,Ep,Aphi,Ephi
        # Add extension to 360 nm (Wei et al., 2016)
        # spectral slope of aph is symetrical wrt 440 nm in the 360-520 spectral range
        wUV = np.linspace(360., 398., num=20)
        A   = ap_bricaud[:,1]
        B   = 1.-ap_bricaud[:,2]
        w   = ap_bricaud[:,0]
        ii  = np.where((w<=520.) & (w>480.))
        AUV = np.zeros_like(wUV)
        BUV = np.zeros_like(wUV)
        AUV[::-1] = A[ii]
        BUV[::-1] = B[ii]
        self.BRICAUD = xr.Dataset()
        self.BRICAUD = self.BRICAUD.assign_coords(wav=np.concatenate((wUV,ap_bricaud[:,0])))
        self.BRICAUD['A'] = xr.DataArray(np.concatenate((AUV,A)), dims=['wav'])
        self.BRICAUD['E'] = xr.DataArray(np.concatenate((BUV,B)), dims=['wav'])

        # Determine chlorophyll integrated column until euphotic depth Zeu
        if Zeu is None:
            if not MIXED:
                if (chls > 1.): chl_zeu = 37.7*chls**0.615
                else: chl_zeu = 36.1*chls**0.357
            else:
                chl_zeu = 42.1*chls**0.538
            Zeu = 568.2*chl_zeu**(-0.746)
        self.Zeu = Zeu

        # Reduced concentration chi and reduced depth zeta,
        # stratified trophic case 1 parametrization (Uitz et al., 2006)
        self.chi_b    = 0.471
        self.s        = 0.135
        self.chi_max  = 1.572
        self.zeta_max = 0.969
        self.Dzeta    = 0.393

    def chi(self, zeta):
        return self.chi_b - self.s*zeta + self.chi_max*np.exp(-((zeta-self.zeta_max)/self.Dzeta)**2)

    def chl(self, z):
        '''
        Chlorophyll concentration in mg/m3 at the depths z (in m)
        '''
        zeta = np.abs(np.asarray(z, dtype='float')/self.Zeu)
        chl = self.chls*self.chi(zeta)/self.chi(0.)
        return np.where(chl < 0., 1e-8, chl)

    def _trunc_scaling(self):
        return 0.5

    def iop(self, wav, z, p1=0.33, R1=0.5, R2=0.5):
        '''
        Inherent optical properties calculation

        Parameters
        ----------
        wav : ndarray
            Wavelengths in nm
        z : ndarray
            Profile of depths in m
        p1, R1 : float, optional
            Parameters related to particles extinction, see Zhai et al. 2017
        R2 : float, optional
            Parameter related to CDOM absorption, see Zhai et al. 2017
        '''
        wav = np.asarray(wav, dtype='float')
        chl2, wav2 = np.meshgrid(self.chl(z), wav)

        # specific phytoplankton absorption
        chl2star=np.full_like(chl2, 1.)
        aphystar = (interp_1d_coord(self.BRICAUD['A'], 'wav', wav2, extrema=True)
            * (chl2star**interp_1d_coord(self.BRICAUD['E'], 'wav', wav2, extrema=True)))
        aphy = aphystar * chl2
        aphystar440 = (interp_1d_coord(self.BRICAUD['A'], 'wav', 440., extrema=True)
            * (chl2star**interp_1d_coord(self.BRICAUD['E'], 'wav', wav2, extrema=True)))
        aphy440 = aphystar440 * chl2

        # phytoplankton covariant particles extinction
        piz440=0.68
        bp440 = aphy440 * piz440/(1-piz440)
        bp = bp440 *(wav2/440.)**(-1.)

        # chlorophyll fluorescence (scattering coefficient)
        FQYC = np.full_like(aphy, self.FQYC) # Fluorescence Quantum Yield for Chlorophyll
        FQYC[wav2<370.]=0.
        FQYC[wav2>690.]=0.

        # CDOM covariant absorption
        aCDM440 = 0.24*aphy440**0.43
        S=0.02
        aCDOM = aCDM440 * np.exp(-S*(wav2 - 440))

        # non-algal particles backscattering
        SPM = 0. # g/m3
        gamma=0.5
        bbpnap650 = 10**(1.03*np.log10(SPM) - 2.06) # Neukermans et al 2012
        bbpnap = bbpnap650*(wav2/650.)**(-gamma)
        Bpnap = np.zeros_like(aphy)
        Bpnap[:] = 0.04
        bpnap  = bbpnap/Bpnap
        bp += bpnap

        return {'ap': aphy,
                'bp': bp,
                'aCDOM': aCDOM,
                'Bp': Bpnap,
                'aphy': aphy,
                'FQYC': FQYC,
                }


class Water1D(Water):
    '''
    1D water column profile definition

    Pure water absorption and scattering are always present and computed
    here; hydrosols are added through the `comp` parameter.

    Parameters
    ----------
    Z : array_like, optional
        Profile of depths in m, from the surface to the sea floor.
        Note that the first item of Z is not used.
    comp : list, optional
        Hydrosols to consider, i.e. a list of Hydrosol, HydrosolPR or/and
        HydrosolZhai objects.
    aw : None or 2-D ndarray, optional
        Force the pure water absorption coefficient in m-1, with
        dimensions [nwav, nZ]. If None, it is read from the auxiliary
        data (see `_read_aw`).
    bw : None or 2-D ndarray, optional
        Force the pure water scattering coefficient in m-1, with
        dimensions [nwav, nZ]. If None, it is computed as
        19.3e-4*(wav/550)**-4.3.
    ALB : albedo object, optional
        Sea floor albedo

    Examples
    --------
    >>> from smartg.water import Water1D, HydrosolPR
    >>> water = Water1D(comp=[HydrosolPR(chl=0.5)])
    '''

    def __init__(self, Z=[0, -10000], comp=None, aw=None, bw=None,
                 ALB=AlbedoCst(0.)):
        self.Z = np.array(Z, dtype='float')
        self.comp = [] if comp is None else comp
        self.aw = aw
        self.bw = bw
        self.ALB = ALB

        self.AW = _read_aw(dir_aux)

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

        z = self.Z
        shp = (len(wav), len(z))
        wav2 = np.stack([wav]*len(z), axis=1)

        #
        # pure water absorption and scattering
        #
        if self.aw is None:
            aw = interp_1d_coord(self.AW, 'wavelength', wav2)
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
        aCDOM = np.zeros(shp, dtype='float')
        aphy_fluo = np.zeros(shp, dtype='float')
        aphy = np.zeros(shp, dtype='float')

        for comp in self.comp:
            iop = comp.coeffs(wav, z, phase=phase,
                              use_old_calc_iphase=use_old_calc_iphase)
            ap += iop['ap']
            bp += iop['bp']
            aCDOM += iop['aCDOM']
            aphy += iop['aphy']
            aphy_fluo += iop['aphy'] * iop['FQYC']

        # the fluorescing fraction of the phytoplankton absorption is
        # counted as (inelastic) scattering instead of absorption
        atot = aw + ap - aphy_fluo + aCDOM
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
        tau_y   = - (aCDOM      ) * dz
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

        pro['albedo_seafloor'] = xr.DataArray(self.ALB.get(wav),
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
        z = self.Z

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

        P_tot = 0.
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
            P_tot = P_tot + pha * bsca_

        with np.errstate(divide='ignore', invalid='ignore'):
            P_tot = P_tot/bsca
        return P_tot.fillna(0.)


class WaterRw(Water):
    def __init__(self, ALB):
        '''
        Defines a model of water reflectance (lambertian under the surface)

        ALB: albedo object of the lambertian reflector
        '''
        self.ALB = ALB

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
        pro['albedo_seafloor'] = xr.DataArray(self.ALB.get(wav), dims=['wavelength'])

        return pro
