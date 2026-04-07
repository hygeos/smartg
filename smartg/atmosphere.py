#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from luts.luts import MLUT, LUT, Idx, read_mlut, read_mlut_hdf5
from smartg.tools.phase import calc_iphase
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy import constants
from scipy.constants import speed_of_light, Planck, Boltzmann
from smartg.bandset import BandSet
from smartg.config import DIR_AUXDATA
import pandas as pd

import xarray as xr
from tempfile import TemporaryDirectory


class AerOPAC(object):
    """
    Initialize the Aerosol OPAC model

    Parameters
    ----------
    filename : str
        Complete path to the aerosol file or filename for aerosols located in "auxdata/aerosols/OPAC/mixtures/".  
        Available auxdata aerosols: antarctic, antarctic_spheric, arctic, continental_average,  
        continental_clean, continental_polluted, desert, desert_spheric, maritime_clean,  
        maritime_polluted, mineral_transported, maritime_tropical and urban
    tau_ref : float
        Optical thickness at reference wavelength w_ref
    w_ref : float
        Wavelength in nanometers at reference optical depth tau_ref
    H_mix_min : float, optional
        Force min altitude of the mixture
    H_mix_max : float, optional
        Force max altitude of the mixture
    H_free_min : float, optional
        Force min altitude of the free troposphere
    H_free_max : float, optional
        Force max altitude of the free troposphere
    H_stra_min : float, optional
        Force min altitude of the stratosphere
    H_stra_max : float, optional
        Force max altitude of the stratosphere
    Z_mix : float, optional
        Force scale height (see notes) of the mixture
    Z_free : float, optional
        Force scale height (see notes) of the free troposphere
    Z_stra : float, optional
        Force scale height (see notes) of the stratosphere
    ssa : None | float | list | 1-D ndarray | 2-D ndarray | LUT, optional
        Force particle single scattering albedo. Default None.
        
        - if float -> same value for all wavelengths and altitudes
        - if list -> it will be converted into a 1-D ndarray.
        - if 1-D ndarray -> only wavelength dependence is considered 
        - if 2-D ndarray -> wavelength and altitude dependence is considered
        - if LUT -> wavelength and altitude dependence is considered

        Note that LUT is more flexible since it allows interpolation if wavelengths  
        in calc method are different (but not the case for the altitude axis).
    phase : None | luts.LUT, optional
        Phase matrix F as function of wavelength, altitude, stoke components and scattering angle    
        The variable names must be:  
        If 4-D matrix -> wav_phase, z_phase, stk, theta  
        If 2-D matrix (assumed monochromatic and contant vertically) -> stk, theta   
        Where:  
        - wav_phase is the wavelength. It must be equal to the `pfwav` parameter of AtmAFGL  
          if defined, else `wav` parameter vavelengths of the AtmAFGL calc method.
        - z_phase is the phase altitude. It must be equal to the `pfgrid[1:]` parameter
          of AtmAFGL  
        - stk the phase matrix unique terms.  
        - theta the scattering angle.  

        The phase matrix terms (IQUV convention) must be given in the folowing order: 
        - F11, F21, F33 and F34 if only 4 terms are given (only for spherical particles)
        - F11, F21, F33, F34, F22 and F44 if 6 terms are given (for both spherical and non-spherical particles)
    rh_mix/free/stra : None | float, optional
        Force relative humidity of mixture/free tropo/strato. Default None.

    Notes
    -----
    The scale height (see Hess et al. 2004) is the variable Z in the following equation:

    - :math:`N(h) = N(0)exp(-h/Z)`

    with N the number density and h the altitude

    Examples
    --------
    >>> from smartg.atmosphere import AerOPAC
    >>> aer_mc = AeroOPAC('maritime_clean', 0.1, 550.)
    >>> aer_mc.mixture.describe()
    <luts.luts.MLUT object at 0x7fbadd61d250>
    Datasets:
    [0] ext (float32 in [0.00384, 0.485]), axes=('hum', 'wav')
        Attributes:
        _FillValue: nan
        description: extinction coefficient in km^-1
    [1] ssa (float32 in [0.436, 1]), axes=('hum', 'wav')
        Attributes:
        _FillValue: nan
        description: single scattering albedo
    [2] phase (float32 in [-0.818, 5.79e+03]), axes=('hum', 'wav', 'stk', 'theta')
        Attributes:
        _FillValue: nan
        description: scattering phase matrix
    Axes:
    [0] hum: 8 values in [0.0, 99.0]
    [1] wav: 26 values in [250, 4500]
    [2] theta: 1801 values in [0.0, 180.0]
    Attributes:
    name : maritime_clean
    H_mix_min : 0
    H_mix_max : 2
    H_free_min : 2
    H_free_max : 12
    H_stra_min : 12
    H_stra_max : 35
    Z_mix : 1
    Z_free : 8
    Z_stra : 99
    date : 2024-03-19
    source : Created by HYGEOS using MOPSMAP v1.0.
    <luts.luts.MLUT at 0x7fbadd61d250>
    """

    def __init__(self, filename, tau_ref, w_ref, H_mix_min=None, H_mix_max=None, 
                 H_free_min=None, H_free_max=None, H_stra_min=None, H_stra_max=None,
                 Z_mix=None, Z_free=None, Z_stra=None, ssa=None, phase=None,
                 rh_mix=None, rh_free=None, rh_stra=None):
        
        self.tau_ref = tau_ref
        if (np.isscalar(w_ref) or
            (isinstance(w_ref, np.ndarray) and w_ref.ndim == 0) ) : self.w_ref = np.array([w_ref])
        else                                                      : self.w_ref = np.array(w_ref)
        self._phase = phase

        if ssa is None : self.ssa = None
        else           :
            if (isinstance(ssa, list)) :
                ssa = np.array(ssa)
            if ( np.isscalar(ssa)                                 or
                 (isinstance(ssa, np.ndarray) and (ssa.ndim <=2)) or
                 isinstance(ssa, LUT) ):
                self.ssa = ssa
            else:
                raise ValueError ("The ssa variable must a scalar, a list, an ndarray of dim <= 2, or a LUT.")
                    
        filename = Path(filename)
        if filename.parent == Path('.'):  # no directory given
            filename = DIR_AUXDATA / 'aerosols' / 'OPAC' / 'mixtures' / filename.name

        # Add extension if needed
        if "_sol" not in filename.name and not filename.suffix == ".nc":
            filename = filename.with_name(filename.name + "_sol.nc")
        elif filename.suffix != ".nc":
            filename = filename.with_name(filename.name + ".nc")

        if not filename.exists():
            raise FileNotFoundError(f"{filename} does not exist")

        self.filename = filename

        self.mixture = read_mlut(self.filename)
        # check if hum dim size == 1 (to avoid lut sub bug)
        if (self.mixture.axes['hum'].size == 1):
            from copy import deepcopy
            from luts.luts import merge
            hum_v1 = self.mixture.axes['hum'][0]
            hum_v2 = hum_v1 + 1
            m1 = deepcopy(self.mixture).sub({'hum':0.})
            m2 = deepcopy(m1)
            m1.set_attr('hum',hum_v1)
            m2.set_attr('hum',hum_v2)
            m3 = merge([m1,m2], ['hum'])
            self.mixture = m3
        self.hum_or_reff = "hum"
        self.free_tropo = None
        self.strato = None

        if H_mix_min is None : H_mix_min = float(self.mixture.attrs['H_mix_min'])
        if H_mix_max is None : H_mix_max = float(self.mixture.attrs['H_mix_max'])
        if H_free_min is None : H_free_min = float(self.mixture.attrs['H_free_min'])
        if H_free_max is None : H_free_max = float(self.mixture.attrs['H_free_max'])
        if H_stra_min is None : H_stra_min = float(self.mixture.attrs['H_stra_min'])
        if H_stra_max is None : H_stra_max = float(self.mixture.attrs['H_stra_max'])

        if Z_mix is None : Z_mix = float(self.mixture.attrs['Z_mix'])
        if Z_free is None : Z_free = float(self.mixture.attrs['Z_free'])
        if Z_stra is None :
            if self.mixture.attrs['Z_stra'] == '99' : Z_stra = 1e6 # -> OPAC Z=99 for constant vertical dist
            else                                    : Z_stra = float(self.mixture.attrs['Z_stra'])


        self.force_rh = [rh_mix, rh_free, rh_stra]
        self.vert_content = []
        self.H_min = []
        self.H_max =[]
        self.Z_sh =[]

        if (H_mix_max-H_mix_min > 1e-6):
            self.vert_content.append(self.mixture)
            self.H_min.append(H_mix_min)
            self.H_max.append(H_mix_max)
            self.Z_sh.append(Z_mix)
        if (H_free_max-H_free_min > 1e-6):
            filename_tmp = DIR_AUXDATA / 'aerosols' / 'OPAC' / 'free_troposphere' / 'free_troposphere_sol.nc'
            self.free_tropo = read_mlut(filename_tmp)
            # check we have the same wl dim than previous aer pro in vert_content
            if len(self.vert_content) > 0:
                aer_prev = self.vert_content[-1]
                w_cur = self.free_tropo.axes['wav']
                w_prev = aer_prev.axes['wav']
                nwcur = len(w_cur)
                nwprev = len(w_prev)
                if (nwcur != nwprev or (nwcur == nwprev and not np.array_equal(w_cur, w_prev)) ):
                    self.free_tropo = self.free_tropo.sub({'wav': Idx(w_prev, fill_value='extrema,warn')})
            self.vert_content.append(self.free_tropo)
            self.H_min.append(H_free_min)
            self.H_max.append(H_free_max)
            self.Z_sh.append(Z_free)
        if (H_stra_max-H_stra_min > 1e-6):
            filename_tmp = DIR_AUXDATA / 'aerosols' / 'OPAC' / 'stratosphere' / 'stratosphere_sol.nc'
            self.strato = read_mlut(filename_tmp)
            # check we have the same wl dim than previous aer pro in vert_content
            if len(self.vert_content) > 0:
                aer_prev = self.vert_content[-1]
                w_cur = self.strato.axes['wav']
                w_prev = aer_prev.axes['wav']
                nwcur = len(w_cur)
                nwprev = len(w_prev)
                if (nwcur != nwprev or (nwcur == nwprev and not np.array_equal(w_cur, w_prev)) ):
                    self.strato = self.strato.sub({'wav': Idx(w_prev, fill_value='extrema,warn')})
            self.vert_content.append(self.strato)
            self.H_min.append(H_stra_min)
            self.H_max.append(H_stra_max)
            self.Z_sh.append(Z_stra)
        
    def dtau_ssa(self, wav, Z, rh):
        '''
        Calculate optical depth and single scattering albedo.
        
        Computes the spectral optical depth (dtau) and single scattering albedo (ssa)
        for aerosol/cloud layers at specified wavelengths and altitudes. This method
        works with both AerOPAC (aerosol) and Cloud classes (which inherits from AerOPAC).
        Handles vertical profiles (mixtures, free troposphere, stratosphere) and optional
        scaling/forcing of optical properties.
        
        Parameters
        ----------
        wav : array-like
            Wavelengths (in nm) at which to calculate optical properties
        Z : array-like
            Altitude profile (in km) for which to calculate optical properties
        rh : float or array-like, optional
            Relative humidity (0-100). Only used with AerOPAC class; ignored for Cloud.
            Also ignored for specific vertical layers if their corresponding layer-specific
            humidity values (rh_mix, rh_free, rh_stra) are set to non-None during initialization.
            For example, if only rh_mix is specified, rh is ignored only in the mixture layer.
            
        Returns
        -------
        dtau : ndarray
            Optical depth with shape (len(wav), len(Z))
        ssa : ndarray
            Single scattering albedo with shape (len(wav), len(Z))
        '''
        dtau = np.zeros((len(wav), len(Z)), dtype=np.float32)
        dtau_ref = np.zeros((1, len(Z)), dtype=np.float32)
        ssa = np.zeros_like(dtau)

        if (self.hum_or_reff == 'hum'):
            hum_or_reff_val = rh
        elif (self.hum_or_reff == 'reff'):
            hum_or_reff_val = self.reff
        else:
            raise NameError("ext and ssa must varies as function of hum or reff.")
        
        if (np.isscalar(hum_or_reff_val) or
            (isinstance(hum_or_reff_val, np.ndarray) and hum_or_reff_val.ndim == 0) ) : hum_or_reff_val = np.array([hum_or_reff_val])
        else                                                                          : hum_or_reff_val = np.array(hum_or_reff_val)
        
        ext_ = np.zeros_like(dtau)
        ext_ref_ = np.zeros_like(dtau_ref)
        ssa_ = np.zeros_like(dtau)
        for icont, cont in enumerate(self.vert_content):
            if ((self.hum_or_reff == 'hum') and (self.force_rh[icont] is not None)) : rh_reff = np.full_like(hum_or_reff_val, self.force_rh[icont])
            else                                                                    : rh_reff = hum_or_reff_val
            if (len(rh_reff) == 1):
                ext_tmp = cont['ext'].swapaxes(self.hum_or_reff, 'wav').sub()[:,Idx(rh_reff[:], fill_value='extrema,warn')][Idx(wav),:]
                ext_ref_tmp = cont['ext'].swapaxes(self.hum_or_reff, 'wav').sub()[:,Idx(rh_reff[:], fill_value='extrema,warn')][Idx(self.w_ref),:]
                ssa_tmp = cont['ssa'].swapaxes(self.hum_or_reff, 'wav').sub()[:,Idx(rh_reff[:], fill_value='extrema,warn')][Idx(wav),:]
                for iz in range (0, len(Z)):
                    ext_[:,iz] = ext_tmp[:,0]
                    ext_ref_[:,iz] = ext_ref_tmp[:,0]
                    ssa_[:,iz] = ssa_tmp[:,0]
            else:      
                ext_ = cont['ext'].swapaxes(self.hum_or_reff, 'wav').sub()[:,Idx(rh_reff[:], fill_value='extrema,warn')][Idx(wav),:]
                ext_ref_ = cont['ext'].swapaxes(self.hum_or_reff, 'wav').sub()[:,Idx(rh_reff[:], fill_value='extrema,warn')][Idx(self.w_ref),:]
                ssa_ = cont['ssa'].swapaxes(self.hum_or_reff, 'wav').sub()[:,Idx(rh_reff[:], fill_value='extrema,warn')][Idx(wav),:]
            dtau_ = np.zeros_like(dtau)
            dtau_ref_ = np.zeros_like(dtau_ref)
            h1 = np.maximum(self.H_min[icont], Z[1:])
            h2 = np.minimum(self.H_max[icont], Z[:-1])
            cond = h2>h1
            dtau_[:,1:][:,cond] = ext_[:,1:][:,cond] * get_aer_dist_integral(self.Z_sh[icont], h1[cond], h2[cond])
            dtau += dtau_
            ssa += dtau_*ssa_
            dtau_ref_[:,1:][:,cond] = ext_ref_[:,1:][:,cond] * get_aer_dist_integral(self.Z_sh[icont], h1[cond], h2[cond])
            dtau_ref += dtau_ref_

        ssa[dtau!=0] /= dtau[dtau!=0]

        #apply scaling factor to get the required optical thickness at the
        # specified wavelength or force tau for all wavelengths
        if self.tau_ref is not None: 
            if (isinstance(self.tau_ref, np.ndarray) and self.tau_ref.ndim == 0) or np.isscalar(self.tau_ref):
                dtau *= self.tau_ref/np.sum(dtau_ref)
            else:
                assert isinstance(self.tau_ref, LUT)
                dtau *= (self.tau_ref[Idx(wav)]/np.sum(dtau, axis=1))[:,None]

        # force ssa
        if self.ssa is not None:
            if np.isscalar(self.ssa): # scalar
                ssa[:,:] = float(self.ssa)
            elif isinstance(self.ssa, np.ndarray): # ndarray with dim <= 2
                if self.ssa.ndim == 0: ssa[:,:] = self.ssa
                elif self.ssa.ndim == 1: ssa[:,:] = self.ssa[:,None] # If 1d array -> consider only wl variability
                elif self.ssa.ndim == 2: ssa[:,:] = self.ssa[:,:]
            else: # LUT
                ssa[:,:] = self.ssa[Idx(wav)][:,None]
        return dtau, ssa
    
    def phase(self, wav, Z, rh, NBTHETA=721, conv_Iparper=True):
        """
        Calculate phase matrix for aerosols and clouds.
        
        Computes the phase matrix at specified wavelengths and altitudes for aerosol/cloud
        layers. This method works with both AerOPAC (aerosol) and Cloud classes (which 
        inherits from AerOPAC). Handles vertical profiles (mixtures, free troposphere, 
        stratosphere) and performs angle resampling. Supports both spherical (4 Stokes 
        components) and non-spherical (6 components) particles.
        
        Parameters
        ----------
        wav : array-like
            Wavelengths (in nm) at which to calculate phase matrix
        Z : array-like
            Altitude profile (in km) for which to calculate phase matrix
        rh : array-like
            Relative humidity (%). Must have size similar to Z (altitude profile). 
            Only used with AerOPAC class; ignored for Cloud.
            Relative humidity can be greater than 100% (supersaturation).
            Also ignored for specific vertical layers if their corresponding layer-specific
            humidity values (rh_mix, rh_free, rh_stra) are set to non-None during initialization.
            For example, if only rh_mix is specified, rh is ignored only in the mixture layer.
        NBTHETA : int, optional
            Number of scattering angles for angle resampling. Default is 721.
        conv_Iparper : bool, optional
            If True (default), converts the phase matrix from I/Q convention to Ipar/Iper
            convention. This applies general conversion formulas valid for both spherical
            and non-spherical particles.
            
        Returns
        -------
        phase_matrix : LUT
            Lookup table containing the phase matrix with axes [wav_phase, z_phase, stk, theta_atm].
            Shape is (len(wav), len(Z)-1, nphamat, NBTHETA) where:
            - nphamat = 4 for spherical particles only (phase matrix unique terms P11, P21, P33, P34)
            - nphamat = 6 for spherical and non-spherical particles (additional phase matrix unique terms P22, P44)
            - theta_atm: scattering angles from 0° to 180°
            
        Notes
        -----
        **AerOPAC only (aerosols):** The method handles vertical averaging based on the input altitude profile Z 
        and the defined aerosol layer altitudes (mixture, free troposphere, stratosphere). 
        If the provided Z profile has a resolution such that multiple input layers fall within 
        a single internal zgrid interval, the phase matrix is averaged across those layers 
        according to the vertical distribution of aerosols in each of the three stratospheric 
        layers (mixture, free troposphere, and stratosphere). This ensures proper vertical 
        integration when the requested altitude resolution is coarser than the internal grid.
        """

        if self._phase is not None:
            if self._phase.ndim == 2:
                # convert to 4-dim by inserting empty dimensions wav_phase
                # and z_phase
                assert self._phase.names == ['stk', 'theta_atm']

                if conv_Iparper: pha_ = pha2Iparperconv(self._phase.data[:,:])
                else: pha_ = self._phase.data[:,:]
                pha = LUT(pha_[None,None,:,:],
                          names = ['wav_phase', 'z_phase'] + self._phase.names,
                          axes = [np.array([wav[0]]), np.array([0.])] + self._phase.axes,
                         )

                return pha
            else:
                if conv_Iparper:
                    pha_ = pha2Iparperconv(self._phase.data[:,:,:,:]) # be careful, if nstk=4 convert to nstk=6
                    pha = LUT(pha_,names = self._phase.names,axes = self._phase.axes)
                    return pha
                else:
                    return self._phase

        theta = np.linspace(0., 180., num=NBTHETA)
        lam_tabulated = np.array(self.mixture.axis('wav'))
        nwav = len(wav)

        P_tot = 0.
        dssa = 0.
        for icont, cont in enumerate(self.vert_content):    
            # Number of independant components of the phase Matrix
            # Spheric particles -> 4, non spheric particles -> 6
            nphamat = cont['phase'].shape[2]

            if ( (np.max(wav) > np.max(lam_tabulated)) or
                (np.min(wav) < np.min(lam_tabulated)) ):
                phase_bis = cont['phase'].swapaxes('wav', self.hum_or_reff).sub()[Idx(wav),:,:,:]
            else:
                # The optimisation consists to not interpolate at all wavelengths of lam_tabulated,
                # but only the wavelengths of lam_tabulated closely in the range of np.min(wav) and np.max(wav)
                range_ind = np.array([np.argwhere((lam_tabulated <= np.min(wav)))[-1][0],
                                    np.argwhere((lam_tabulated >= np.max(wav)))[0][0]])
                ilam_tabulated = np.arange(len(lam_tabulated), dtype=int)
                ilam_opti = np.concatenate(np.argwhere((ilam_tabulated >= range_ind[0]) &
                                                    (ilam_tabulated <= range_ind[1])))

                if len(ilam_opti) > 1 : phase_bis = cont['phase'].swapaxes('wav', self.hum_or_reff).sub()[ilam_opti,:,:,:].sub()[Idx(wav),:,:,:]
                else                  : phase_bis = cont['phase'].swapaxes('wav', self.hum_or_reff).sub()[ilam_opti,:,:,:]

            if (NBTHETA != len(phase_bis.axes[3])): phase_bis = phase_bis.sub()[:,:,:,Idx(theta)]

            nphamat_ = 6
            if (self.hum_or_reff == 'hum'):
                if (self.force_rh[icont] is not None) : hum_or_reff_val = np.full_like(rh, self.force_rh[icont])
                else                                  : hum_or_reff_val = rh

                P = LUT(
                    np.zeros((nwav, len(rh)-1, nphamat_, NBTHETA), dtype='float32')+np.nan,
                    axes=[wav, None, None, theta],
                    names=['wav_phase', 'z_phase', 'stk', 'theta_atm'],
                    )  # nlam_tabulated, nrh, stk, NBTHETA
                
                for irh_, rh_ in enumerate(hum_or_reff_val[1:]):
                    irh = Idx(rh_, fill_value='extrema')
                    #irh = Idx(rh_, fill_value='extrema')
                    P.data[:,irh_,0:nphamat,:] = phase_bis.sub()[:,irh,:,:].data
            elif (self.hum_or_reff == 'reff'):
                P = LUT(
                    np.zeros((nwav, 1, nphamat_, NBTHETA), dtype='float32')+np.nan,
                    axes=[wav, None, None, theta],
                    names=['wav_phase', 'z_phase', 'stk', 'theta_atm'],
                    )  # nlam_tabulated, nrh, stk, NBTHETA
                
                irh = Idx(self.reff).index(cont['phase'].axes[0])
                #irh = Idx(self.reff).index(cont['phase'].axes[0])
                P.data[:,0,0:nphamat,:] = phase_bis[:,irh,:,:].data
                hum_or_reff_val = self.reff
            else:
                raise NameError("Phase matrix must varies as function of hum or reff.")
            
            if (np.isscalar(hum_or_reff_val) or
            (isinstance(hum_or_reff_val, np.ndarray) and hum_or_reff_val.ndim == 0) ) : hum_or_reff_val = np.array([hum_or_reff_val])
            else                                                                      : hum_or_reff_val = np.array(hum_or_reff_val)

            if (nphamat == 4): # only for spherical particles
                P.data[:,:,4,:] = P.data[:,:,0,:].copy() # F22 = F11
                P.data[:,:,5,:] = P.data[:,:,2,:].copy() # F44 = F33

            if conv_Iparper:
                # convert I, Q into Ipar, Iper ; Fij -> Pij
                # use general formulas (valid for both spherical and non-spherical particles)
                # P33=F33, P34=F34 and P44=F44
                F11 = P.data[:,:,0,:].copy()
                F21 = P.data[:,:,1,:].copy()
                F22 = P.data[:,:,4,:].copy()
                P.data[:,:,0,:] = 0.5*(F11+2*F21+F22) # P11
                P.data[:,:,1,:] = 0.5*(F11-F22)       # P21
                P.data[:,:,4,:] = 0.5*(F11-2*F21+F22) # P22

            dtau_ =  np.zeros((len(wav), len(Z)), dtype=np.float32)
            ext_ = np.zeros_like(dtau_)
            ssa_ = np.zeros_like(dtau_)
            if (len(hum_or_reff_val) == 1):
                ext_tmp = cont['ext'].swapaxes(self.hum_or_reff, 'wav').sub()[:,Idx(hum_or_reff_val[:], fill_value='extrema,warn')][Idx(wav),:]
                ssa_tmp = cont['ssa'].swapaxes(self.hum_or_reff, 'wav').sub()[:,Idx(hum_or_reff_val[:], fill_value='extrema,warn')][Idx(wav),:]
                for iz in range (0, len(Z)):
                    ext_[:,iz] = ext_tmp[:,0]
                    ssa_[:,iz] = ssa_tmp[:,0]
            else:      
                ext_ = cont['ext'].swapaxes(self.hum_or_reff, 'wav').sub()[:,Idx(hum_or_reff_val[:], fill_value='extrema,warn')][Idx(wav),:]
                ssa_ = cont['ssa'].swapaxes(self.hum_or_reff, 'wav').sub()[:,Idx(hum_or_reff_val[:], fill_value='extrema,warn')][Idx(wav),:]
            h1 = np.maximum(self.H_min[icont], Z[1:])
            h2 = np.minimum(self.H_max[icont], Z[:-1])
            cond = h2>h1
            dtau_[:,1:][:,cond] = ext_[:,1:][:,cond] * get_aer_dist_integral(self.Z_sh[icont], h1[cond], h2[cond])
            dssa_ = dtau_*ssa_ # NLAM, ALTITUDE
            dssa_ = dssa_[:,1:,None,None]
            dssa += dssa_
            P_tot+= P*dssa_

        
        with np.errstate(divide='ignore'):
            P_tot.data /= dssa
        P_tot.data[np.isnan(P_tot.data)] = 0.
        P_tot.axes[1] = Z[1:]
        return P_tot
    
    @staticmethod
    def list():
        """
        List available standard OPAC aerosol mixture files.
        
        Returns
        -------
        list of str
            List of available OPAC aerosol mixture filenames (without suffix).
            
        Examples
        --------
        >>> from smartg.atmosphere import AerOPAC
        >>> AerOPAC.list()
        ['antarctic', 'antarctic_spheric', 'arctic', 'continental_average', ...]
        """
        base_dir = DIR_AUXDATA / 'aerosols' / 'OPAC' / 'mixtures'
        files = list(base_dir.glob("*.nc"))
        return sorted([f.stem.replace('_sol', '') for f in files])


class Cloud(AerOPAC):
    """
    Initialize the cloud model

    Parameters
    ----------
    filename : str,
        Complete path to the cloud file or filename for clouds located in "auxdata/clouds/"  
        Available auxdata clouds: wc, ic_baum_ghm, ic_baum_asc and ic_baum_sc
    reff : float
        Effective radius in micrometers
    zmin : float,
        Minimum altitude of the cloud
    zmax : float,
        Maximum altitude of the cloud
    tau_ref : float,
        Optical thickness at reference wavelength w_ref
    w_ref : float
        Wavelength in nanometers at reference optical thickness tau_ref
    ssa : None | float | list | 1-D ndarray | 2-D ndarray | LUT, optional
        Force particle single scattering albedo. 
        
        - if float -> same value for all wavelengths and altitudes
        - if list -> it will be converted into a 1-D ndarray.
        - if 1-D ndarray -> only wavelength dependence is considered 
        - if 2-D ndarray -> wavelength and altitude dependence is considered
        - if LUT -> wavelength and altitude dependence is considered

        Note that LUT is more flexible since it allows interpolation if wavelengths  
        in calc method are different (but not the case for the altitude axis).
    phase : None | luts.LUT, optional
        Phase matrix F as function of wavelength, altitude, stoke components and scattering angle    
        The variable names must be:  
        If 4-D matrix -> wav_phase, z_phase, stk, theta  
        If 2-D matrix (assumed monochromatic and contant vertically) -> stk, theta   
        Where:  
        - wav_phase is the wavelength. It must be equal to the `pfwav` parameter of AtmAFGL  
          if defined, else `wav` parameter wavelengths of the AtmAFGL calc method.
        - z_phase is the phase altitude. It must be equal to the `pfgrid[1:]` parameter
          of AtmAFGL  
        - stk the phase matrix unique terms.  
        - theta the scattering angle.  

        The phase matrix terms (IQUV convention) must be given in the folowing order: 
        - F11, F21, F33 and F34 if only 4 terms are given (only for spherical particles)
        - F11, F21, F33, F34, F22 and F44 if 6 terms are given (for both spherical and non-spherical particles)

    Examples
    --------
    >>> from smartg.atmophere import Cloud
    >>> cld_wc = Cloud('wc', 12.68, 2, 3, 10., 550.)
    >>> cld_wc.mixture.describe(show_attrs=True)
    <luts.luts.MLUT object at 0x7fbb4c74eb10>
    Datasets:
    [0] phase (float32 in [-111, 3.05e+05]), axes=('reff', 'wav', 'stk', 'theta')
        Attributes:
        description: phase matrix integral normalized to 2. stk order: p11, p21, p33 and p34
    [1] ext (float64 in [123, 4.62e+03]), axes=('reff', 'wav')
        Attributes:
        description: extinction coefficient in km^-1
    [2] ssa (float64 in [0.476, 1]), axes=('reff', 'wav')
        Attributes:
        description: single scattering albedo
    Axes:
    [0] reff: 26 values in [5.0, 30.0]
    [1] wav: 209 values in [253.0570068359375, 4441.29296875]
    [2] stk: 4 values in [0, 3]
    [3] theta: 594 values in [0.0, 180.0]
    Attributes:
    veff : 0.1
    <luts.luts.MLUT at 0x7fbb4c74eb10>
    
    """

    def __init__(self, filename, reff, zmin, zmax, tau_ref, w_ref, ssa=None,
                 phase=None):
        self.reff = reff
        self.tau_ref = tau_ref
        if (np.isscalar(w_ref) or
            (isinstance(w_ref, np.ndarray) and w_ref.ndim == 0) ) : self.w_ref = np.array([w_ref])
        else                                                      : self.w_ref = np.array(w_ref)

        if ssa is None : self.ssa = None
        else           :
            if (isinstance(ssa, list)) :
                ssa = np.array(ssa)
            if ( np.isscalar(ssa)                                 or
                 (isinstance(ssa, np.ndarray) and (ssa.ndim <=2)) or
                 isinstance(ssa, LUT) ):
                self.ssa = ssa
            else:
                raise ValueError ("The ssa variable must a scalar, a list, an ndarray of dim <= 2, or a LUT.")

        filename = Path(filename)
        if filename.parent == Path('.'):  # no directory given
            base_dir = Path(DIR_AUXDATA) / 'clouds'
            filename = base_dir / filename.name

        if "_sol" not in filename.name and not filename.suffix == ".nc":
            filename = filename.with_name(filename.name + "_sol.nc")
        elif filename.suffix != ".nc":
            filename = filename.with_name(filename.name + ".nc")

        if not filename.exists():
            raise FileNotFoundError(f"{filename} does not exist")

        self.filename = filename

        self.mixture = read_mlut(self.filename)
        self.hum_or_reff = "reff"
        self.free_tropo = None
        self.strato = None

        self.vert_content = []
        self.H_min = []
        self.H_max =[]
        self.Z_sh =[]

        if (zmax-zmin > 1e-6):
            self.vert_content.append(self.mixture)
            self.H_min.append(zmin)
            self.H_max.append(zmax)
            self.Z_sh.append(1e6) # constant dist

        self._phase = phase

    @staticmethod
    def list():
        """
        List available standard cloud model files.
        
        Returns
        -------
        list of str
            List of available cloud model filenames (without suffix).
            
        Examples
        --------
        >>> from smartg.atmosphere import Cloud
        >>> Cloud.list()
        ['ic_baum_asc', 'ic_baum_ghm', 'ic_baum_sc', 'wc']
        """
        base_dir = Path(DIR_AUXDATA) / "clouds"
        files = list(base_dir.glob("*.nc"))
        return sorted([f.stem.replace('_sol', '') for f in files])
        

class AerUser(AerOPAC):
    """
    Initialize the user-defined aerosol model

    Parameters
    ----------
    aod : 2-D ndarray
        aerosol optical depth values with shape (len(hum), len(wav))
    ssa : 2-D ndarray
        Single scattering albedo values with shape (len(hum), len(wav))
    phase : 4-D ndarray
        Phase function values with shape (len(hum), len(wav), len(stk), len(theta)).

        Where len(stk) is the number of unique phase terms.

        The phase matrix terms must be given in the folowing order: 
        - F11, F21, F33 and F34 if only 4 terms are given (only for spherical particles)
        - F11, F21, F33, F34, F22 and F44 if 6 terms are given (for both spherical and non-spherical particles)
    hum : 1-D ndarray
        Relative humidity values in percentage
    wav : 1-D ndarray
        Wavelength values in nanometers
    theta : 1-D ndarray
        Scattering angle values in degrees
    H_mix_min : float, optional
        Force min altitude of the mixture
    H_mix_max : float, optional
        Force max altitude of the mixture
    Z_mix : float, optional
        Force scale height (see notes) of the mixture

    Notes
    -----
    The scale height (see Hess et al. 2004) is the variable Z in the following equation:

    - :math:`N(h) = N(0)exp(-h/Z)`

    with N the number density and h the altitude
    
    """

    def __init__(self, aod, ssa, phase, hum, wav, theta,  
                 H_mix_min=0., H_mix_max=2., Z_mix=2):
        

        self.filename = 'none'
        self.tau_ref = None
        ext = aod / (Z_mix * (np.exp(-H_mix_min/Z_mix) - np.exp(-H_mix_max/Z_mix)))
        
        # Create an xarray Dataset to hold the mixture data
        ds = xr.Dataset( {'ext': (('hum', 'wav'), ext),
                        'ssa': (('hum', 'wav'), ssa),
                        'phase': (('hum', 'wav', 'stk', 'theta'), phase)},
                        coords={'hum': hum, 'wav': wav, 'theta': theta,
                                'stk': np.arange(phase.shape[2])}
        )

        ds.attrs['name'] = 'none'
        ds.attrs['H_mix_min'] = str(H_mix_min)
        ds.attrs['H_mix_max'] = str(H_mix_max)
        ds.attrs['Z_mix'] = str(Z_mix)

        with TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir)/'tmp_lut.nc'
            ds.to_netcdf(tmp_file)
            ds = read_mlut(tmp_file)

        self.w_ref = np.array([ds.axes['wav'][0]])
        self.ssa = None

        self.mixture = ds
        # check if hum dim size == 1 (to avoid lut sub bug)
        if (self.mixture.axes['hum'].size == 1):
            from copy import deepcopy
            from luts.luts import merge
            hum_v1 = self.mixture.axes['hum'][0]
            hum_v2 = hum_v1 + 1
            m1 = deepcopy(self.mixture).sub({'hum':0.})
            m2 = deepcopy(m1)
            m1.set_attr('hum',hum_v1)
            m2.set_attr('hum',hum_v2)
            m3 = merge([m1,m2], ['hum'])
            self.mixture = m3

        self.hum_or_reff = "hum"
        self.free_tropo = None
        self.strato = None

        self.force_rh = [None]
        self.vert_content = []
        self.H_min = []
        self.H_max =[]
        self.Z_sh =[]

        if (H_mix_max-H_mix_min > 1e-6):
            self.vert_content.append(self.mixture)
            self.H_min.append(H_mix_min)
            self.H_max.append(H_mix_max)
            self.Z_sh.append(Z_mix)

        self._phase = None

    @staticmethod
    def list():
        """
        """
        raise NotImplementedError(
            "The list() method is not available for user-defined aerosols. "
            "User-defined aerosols are custom configurations and do not have "
            "a predefined list of available files."
        )



class Atmosphere(object):
    ''' Base class for atmosphere '''
    pass


class AtmAFGL(Atmosphere):
    """
    Atmospheric profile definition using AFGL data

    Parameters
    ----------

    atm_filename : str
        The AFGL atmosphere profile to use. Choice are:
            - 'afglms' for Mid-Latitude Summer (45N July)
            - 'afglmw' for Mid-Latitude Winter (45N Jan)
            - 'afglss' for Sub Arctic Summer (60N July)
            - 'afglsw' for Sub Arctic Winter (60N Jan)
            - 'afglt' for Tropic (15N Annual Average)
            - 'afglus' for U.S. Standard (1976)
        
        File format: If a full path is not provided (only filename), the atmospheric 
        auxdata directory is automatically prepended to the path. The file extension 
        defaults to '.nc' if not specified. Only '.nc' (NetCDF) and '.dat' file 
        formats are accepted. For '.dat' files, the libratran atmosphere file 
        convention is used.
    comp:  list, optional
        Components particles (aerosols or clouds) to consider, i.e. a list of aerOPAC or/and Cloud objects.
    grid : None | 1-D array-like, optional
      The vertical grid (from TOA to BOA). The optical properties of the atmosphere are recalculated following 
      the new grid. If None, the AFGL grid is kept.
    lat : float, optional
        The latitude used for Rayleigh optical depth calculation. Default=45.
    P0:  None | float, optional
        The sea surface pressure. If None take P0 from the AFGL profil.
    O3 : None | float, optional
        The total ozone column in Dobson units. If None keep the total ozone content of the chosen atmospheric 
        profile.
    H2O : None | float, optional
        The total water vapor column in g.cm-2. If None keep the total water vapor content of the chosen 
        atmospheric profile.
    NO2: bool, optional
        Activate NO2 absorption (default True)
    O3_H2O_alt : None | float, optional
        Altitude (km) at which the specified O3 and H2O values apply. When specified,
        the O3 and H2O profiles are scaled such that the column amount from TOA to this
        altitude matches the provided O3 and H2O values. The full gaseous distribution
        from TOA to ground is preserved; only the scaling factor is adjusted to match
        the constraint at this reference altitude.
        Default: None
    tauR : None | float, optional
        Force the Rayleigh optical thickness. If None, computed from atmospheric profile and wavelength.
    pfwav : None | list, optional
        The list of wavelengths over which the phase matrices are calculated. Then use the nearest wavelength 
        during cuda simulation. Useful to reduce the memory. If None, compute the phase matrix at all wavelengths.
    pfgrid : list, optional
        The vertcial grid (from TOA to BOA) over which the phase matrices are calculated. This parameter can help 
        reduce the memory but must be used with care. If misused, it may lead to innaccurate results especially when 
        multiple aerosols are mixed. The default value is [100, 0], meaning a single phase matrix is calculated over 
        the entire column from 0 to 100 km. This is effcient and accurate when only 1 type of aerosol is present. 
        However, if multiple aerosols are mixed (with different vertical distributions), a single phase matrix may 
        introduce an important bias.
    prof_abs : None | 2-D ndarray, optional
        - In 1D atm mode -> force the gaseous absorption optical thickness vertical profile (NWavelength,NZ),
        it shortcuts any further gaseous absorption computation.
        - In 3D atm mode -> just an optical properties index, it must be completed by the cells grid
    prof_ray : None | 2-D ndarray, optional
        - In 1D atm mode -> force the Rayleigh scattering optical thickness vertical profile (NWavelength,NZ),
        it shortcuts any further Rayleigh scattering computation.
        - In 3D atm mode -> just an optical properties index, it must be completed by the cells grid
    prof_aer : None | tuple, optional
        - In 1D atm mode - > A tuple (ext,ssa) with the aerosol extinction optical thickness profile (ext) and 
        single scattering albedo arrays (ssa), it shortcuts any further particles scattering computation.
        - In 3D atm mode -> just an optical properties index, it must be completed by the cells grid
    prof_phases : None | tuple, optional
        A tuple (iphase, phases ) where iphase is the phase matrix indices profile (NWavelength,NZ), 
        and  phases is a list of phase matrices LUT (as outputs of the `read_phase` utility).
    RH_cst : None | float, optional
        Force relative humidity to be constant. If None calculated depending on H2O vertical profile.
    O3_acs : str, optional
        Path to ozone netcdf4 file with absorption coefficient cross section (SIGMA = 1E-20 * [C0 + C1*T + C2*T^2], 
        in cm^2, and where T is in degrees Celcius). If only filename is given automatically look at "auxdata/acs/".
        By default use Bogumil Version 3.0 data. Available files in auxdata:
            - 'O3_acs_BogumilV3.0_coeffs.nc'
            - 'O3_acs_Chehade(Bogumil_revised)V4.1_coeffs.nc'
            - 'O3_acs_SerdyuchenkoV2.0_coeffs.nc'
    NO2_acs : str, optional
        Path to NO2 netcdf4 file with absorption coefficient cross section (SIGMA = 1E-20 * [C0 + C1*T + C2*T^2], 
        in cm^2, and where T is in degrees Celcius). If only filename is given automatically look at "auxdata/acs/".
        By default use Bogumil Version 1.0 data. Available files in auxdata:
            - 'NO2_acs_BogumilV1.0_coeffs.nc'
            - 'NO2_acs_Bingen_coeffs.nc'
    cells : None | tuple, optional 
        If cells is given, then we are in 3D mode. Definitions:
           - 'iopt' gives the number of the optical property corresponding to the cells. iopt(Ncell)
           - 'iabs' gives the number of the absorption property corresponding to the cells. iabs(Ncell)
           - Bounding Boxes(1 Point Bottom Left pmin, 1 Point Top Right pmax) of the cells. pmin(3,Ncell). pmax(3,Ncell) 
           and 6 neighbours index (positive X, negative X, positive Y, negative Y, positive Z, negative Z). neighbour(6,Ncell)
           it returns coefficients in (km-1) instead of optical thicknesses
    """
    def __init__(self, atm_filename, comp=[],
                 grid=None, lat=45.,
                 P0=None, O3=None, H2O=None, NO2=True,
                 O3_H2O_alt=None,
                 tauR=None,
                 pfwav=None, pfgrid=[100., 0.], prof_abs=None,
                 prof_ray=None, prof_aer=None, prof_phases=None,
                 RH_cst=None, US=True,
                 cells=None,
                 O3_acs = 'O3_acs_BogumilV3.0_coeffs',
                 NO2_acs = 'NO2_acs_BogumilV1.0_coeffs'):

        self.lat = lat
        self.comp = comp
        self.pfwav = pfwav
        self.pfgrid = np.array(pfgrid)
        self.prof_abs = prof_abs
        self.prof_ray = prof_ray
        self.prof_aer = prof_aer
        self.prof_phases = prof_phases
        self.RH_cst = RH_cst
        self.US = US
        self.OPT3D = cells is not None
        if self.OPT3D : self.cells = cells

        self.tauR = tauR
        if tauR is not None:
            self.tauR = np.array(tauR)

        assert (np.diff(pfgrid) < 0.).all()

        atm_filename = Path(atm_filename)

        #
        # init directories and read atm file
        #
        if atm_filename.name == "ATM3D":
            Nopt = grid.size
            prof = Profile_base(None)
            prof.z = np.arange(Nopt, dtype=np.float32)[::-1]
            attr_names = ["P", "T", "dens_air", "dens_h2o", "dens_o3", "dens_n2o", 
                          "dens_co", "dens_ch4", "dens_co2", "dens_o2", "dens_n2", 
                          "dens_no2", "dens_so2"]
            for attr_name in attr_names:
                setattr(prof, attr_name, np.zeros(Nopt, dtype=np.float32))
            prof.RH_cst = RH_cst
        else:
            if atm_filename.parent == Path('.'):
                atm_filename = DIR_AUXDATA / 'atmospheres' / atm_filename.name
            # By default if no suffix is given consider it as a netcdf file
            if not atm_filename.exists() and atm_filename.suffix == '':
                atm_filename = atm_filename.with_name(atm_filename.name + ".nc")
            
            if atm_filename.suffix == '.nc' or atm_filename.suffix == '.dat':
                prof = Profile_base(atm_filename, O3=O3, H2O=H2O, NO2=NO2, P0=P0, 
                                    RH_cst=RH_cst, US=US, O3_H2O_alt=O3_H2O_alt)
            else:
                raise NameError("This file format is not supported. Only '.nc' and" + \
                                " '.dat' are supported.")
                

        #
        # read gaseous acs
        #
        O3_acs_path = Path(O3_acs)
        if O3_acs_path.parent == Path('.'):
            O3_acs_path = DIR_AUXDATA / 'acs' / O3_acs_path.name
        if not O3_acs_path.exists() and O3_acs_path.suffix != '.nc':
            O3_acs_path = O3_acs_path.with_name(O3_acs_path.name + ".nc")
        self.acs_o3 = read_mlut(O3_acs_path)
        self.acs_o3.rename_axis('wav', 'wavelength')

        NO2_acs_path = Path(NO2_acs)
        if NO2_acs_path.parent == Path('.'):
            NO2_acs_path = DIR_AUXDATA / 'acs' / NO2_acs_path.name
        if not NO2_acs_path.exists() and NO2_acs_path.suffix != '.nc':
            NO2_acs_path = NO2_acs_path.with_name(NO2_acs_path.name + ".nc")
        self.acs_no2 = read_mlut(NO2_acs_path)
        self.acs_no2.rename_axis('wav', 'wavelength')


        #
        # regrid profile if required
        #
        if grid is None:
            self.prof = prof
        else:
            if isinstance(grid, str):
                grid = str2grid_array(grid)
            self.prof = prof.regrid(np.array(grid))

        #
        # calculate reduced profile
        # (for phase function blending)
        #
        self.prof_red = prof.regrid(pfgrid)



    def calc(self, wav, phase=True, NBTHETA=721, conv_Iparper=True, use_old_calc_iphase=False,
             truncation=None):
        """
        Profile and phase matrix calculation at bands / wav

        Parameters
        ----------
        wav : float | 1-D ndarray | BandSet | list
            Wavelengths at which to calculate the profile. It can be a list of REPTRAN_IBAND or KDIS_IBAND.
        NBTHETA : int, optional
            The number of angles to be considered for the phase matrix.
        conv_Iparper : bool, optional
            Convert to I parallel I perpendicular convention.
        use_old_calc_iphase : bool, optional
            Use the old way to compute iphase (depracated).
        truncation : None | DM_trunc | GT_trunc, optional
            The scattering phase truncation to use.

        Returns
        -------
        out : MLUT
            An MLUT object with the profile and (if phase = True) the phase matrices.
        """
        
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)
            
        profile = self.profile(wav)
        
        if phase:
            if self.pfwav is None:
                wav_pha = wav[:]
            else:
                wav_pha = self.pfwav
            pha = self.phase(wav_pha, NBTHETA=NBTHETA, conv_Iparper=False)
            is_Iparper = False

            pro_var = profile.datasets()
            if (  pha is not None  or 
                  ( self.OPT3D and ('phase_atm' in pro_var) and truncation )  ):
                
                if pha is not None:
                    pha_, ipha = calc_iphase(pha, profile.axis('wavelength'), profile.axis('z_atm'), use_old_calc_iphase)
                    if pha_.shape[1] == 4:
                        # if only 4 components extend to 6 to use general formulas
                        pha_tmp = np.zeros((pha_.shape[0], 6, pha_.shape[2]), dtype=np.float64)
                        pha_tmp[:,0:4,:] = pha_.copy()
                        pha_tmp[:,4,:] = pha_tmp[:,0,:]
                        pha_tmp[:,5,:] = pha_tmp[:,2,:]
                        pha_ = pha_tmp
                else: # 3D ATM
                    pha_ = profile['phase_atm'].data
                    is_Iparper = True

                nphase = pha_.shape[0]

                # if Iparper convention come back to IQUV for truncation
                if is_Iparper:
                    for iph in range (nphase):
                        pha_[iph,:,:] = pha2Iparperconv(pha_[iph,:,:])

                # If truncation parameter is given compute truncated phase function
                if truncation is not None:
                    from pytrunc.truncation import delta_m_phase_approx, gt_phase_approx
                    if self.OPT3D: theta = profile.axis('theta_atm')
                    else: theta = pha.axes[-1]
                    pha_tr = np.zeros(pha_.shape, dtype=np.float64)
                    nphac = pha_.shape[1]
                    if (truncation.tr_method == 'DM'):
                        m_max = truncation.m_max
                    elif (truncation.tr_method == 'GT'):
                        f_ = truncation.trunc_frac
                        th_tol = truncation.theta_tol
                        l_opti = truncation.lobatto_optimization
                        th_f = truncation.theta_tr
                    else:
                        raise ValueError("truncation method not recognized")
                    method = truncation.integral_method
                    f_pha = np.zeros(nphase, dtype=np.float64)
                    for iph in range (nphase):
                        if (truncation.tr_method == 'DM'):
                            ds_pha = delta_m_phase_approx(pha_[iph,0,:], theta, m_max, method=method)
                            
                        elif (truncation.tr_method == 'GT'):
                            ds_pha = gt_phase_approx(pha_[iph,0,:], theta, f_, method=method,
                                                     th_tol=th_tol, th_f=th_f, lobatto_optimization=l_opti)
                        f11_tr = ds_pha['phase_tr'].values
                        f = ds_pha['f'].values
                        f_pha[iph] = f
                        # Ensure for the moment only 1 unique truncation factor
                        if iph > 0 and not np.isclose(f_pha[iph], f_pha[0], atol=1e-6):
                            raise ValueError("Several truncation factors f is not yet authorized")

                        pha_tr[iph,0,:] = f11_tr
                        beta = pha_tr[iph,0,:]/pha_[iph,0,:]
                        for icomp in range(1, nphac):
                            pha_tr[iph,icomp,:] = pha_[iph,icomp,:]*beta
                        if truncation.pha_scale_method == 2:
                            beta2 = 1. / (1 - f)
                            pha_tr[iph,1,:] = pha_[iph,1,:] * beta2
                            pha_tr[iph,3,:] = pha_[iph,3,:] * beta2

                if conv_Iparper or is_Iparper:
                    for iph in range (nphase):
                        pha_[iph,:,:] = pha2Iparperconv(pha_[iph,:,:])
                        if truncation is not None:
                            pha_tr[iph,:,:] = pha2Iparperconv(pha_tr[iph,:,:])

                if not self.OPT3D:
                    profile.add_axis('theta_atm', pha.axes[-1])
                    profile.add_dataset('phase_atm', pha_, ['iphase', 'stk', 'theta_atm'])
                    profile.add_dataset('iphase_atm', ipha, ['wavelength', 'z_atm'])
                else :
                    attrs_tmp = profile['phase_atm'].attrs
                    profile.rm_lut('phase_atm')
                    profile.add_dataset('phase_atm', pha_, ['iphase', 'stk', 'theta_atm'],
                                        attrs=attrs_tmp)

                if truncation is not None:
                    # profile.add_dataset('phase_atm_tr', pha_tr, axnames=['iphase', 'stk', 'theta_atm'])
                    attrs_tmp = profile['phase_atm'].attrs
                    profile.rm_lut('phase_atm')
                    profile.add_dataset('phase_atm', pha_tr, axnames=['iphase', 'stk', 'theta_atm'])

                    # case tau instead of coeff (1D atm)
                    if not self.OPT3D:
                        dtau_p = diff1(profile['OD_p'].data, axis=1)
                        dtau_p_tr = (1 - f*profile['ssa_p_atm'].data) * dtau_p
                        tau_p_tr = np.cumsum(dtau_p_tr, axis=1)
                        ssa_p_atm_tr = profile['ssa_p_atm'].data * ( (1-f) / (1 - f*profile['ssa_p_atm'].data) )
                        tau_atm_tr = tau_p_tr + profile['OD_r'].data + profile['OD_g'].data
                        dtau_r = diff1(profile['OD_r'].data, axis=1)
                        tau_sca_tr = np.cumsum(dtau_r + dtau_p_tr*ssa_p_atm_tr, axis=1)
                        with np.errstate(invalid='ignore', divide='ignore'):
                            ssa_atm_tr = (dtau_r+ dtau_p_tr*ssa_p_atm_tr)/diff1(tau_atm_tr, axis=1)
                        ssa_atm_tr[np.isnan(ssa_atm_tr)] = 1.
                        with np.errstate(invalid='ignore', divide='ignore'):
                            pmol_tr = dtau_r/(dtau_r + dtau_p_tr*ssa_p_atm_tr)
                        pmol_tr[np.isnan(pmol_tr)] = 1.
                        
                        attrs_tmp = profile['OD_p'].attrs
                        profile.rm_lut('OD_p')
                        profile.add_dataset('OD_p', tau_p_tr, axnames=['wavelength', 'z_atm'],
                                            attrs=attrs_tmp)
                        attrs_tmp = profile['ssa_p_atm'].attrs
                        profile.rm_lut('ssa_p_atm')
                        profile.add_dataset('ssa_p_atm', ssa_p_atm_tr, axnames=['wavelength', 'z_atm'],
                                            attrs=attrs_tmp)
                        attrs_tmp = profile['OD_atm'].attrs
                        profile.rm_lut('OD_atm')
                        profile.add_dataset('OD_atm', tau_atm_tr, axnames=['wavelength', 'z_atm'],
                                            attrs=attrs_tmp)
                        attrs_tmp = profile['OD_sca_atm'].attrs
                        profile.rm_lut('OD_sca_atm')
                        profile.add_dataset('OD_sca_atm', tau_sca_tr, axnames=['wavelength', 'z_atm'],
                                            attrs=attrs_tmp)
                        attrs_tmp = profile['ssa_atm'].attrs
                        profile.rm_lut('ssa_atm')
                        profile.add_dataset('ssa_atm', ssa_atm_tr, axnames=['wavelength', 'z_atm'],
                                            attrs=attrs_tmp)
                        attrs_tmp = profile['pmol_atm'].attrs
                        profile.rm_lut('pmol_atm')
                        profile.add_dataset('pmol_atm', pmol_tr, axnames=['wavelength', 'z_atm'],
                                            attrs=attrs_tmp)
                    # case coeff instead of tau (3D atm)
                    # sig for coeficients
                    else:
                        sig_p = profile['OD_p'].data
                        sig_p_tr = (1 - f*profile['ssa_p_atm'].data) * sig_p
                        ssa_p_atm_tr = profile['ssa_p_atm'].data * ( (1-f) / (1 - f*profile['ssa_p_atm'].data) )
                        sig_atm_tr = sig_p_tr + profile['OD_r'].data + profile['OD_g'].data
                        sig_sca_tr = profile['OD_r'].data  + sig_p_tr*ssa_p_atm_tr
                        with np.errstate(invalid='ignore', divide='ignore'):
                            ssa_atm_tr = (profile['OD_r'].data + sig_p_tr*ssa_p_atm_tr)/sig_atm_tr
                        ssa_atm_tr[np.isnan(ssa_atm_tr)] = 1.
                        sig_r = profile['OD_r'].data
                        with np.errstate(invalid='ignore', divide='ignore'):
                            pmol_tr = sig_r/(sig_r + sig_p_tr*ssa_p_atm_tr)
                        pmol_tr[np.isnan(pmol_tr)] = 1.
                        
                        attrs_tmp = profile['OD_p'].attrs
                        profile.rm_lut('OD_p')
                        profile.add_dataset('OD_p', sig_p_tr, axnames=['wavelength', 'iopt'],
                                            attrs=attrs_tmp)
                        attrs_tmp = profile['ssa_p_atm'].attrs
                        profile.rm_lut('ssa_p_atm')
                        profile.add_dataset('ssa_p_atm', ssa_p_atm_tr, axnames=['wavelength', 'iopt'],
                                            attrs=attrs_tmp)
                        attrs_tmp = profile['OD_atm'].attrs
                        profile.rm_lut('OD_atm')
                        profile.add_dataset('OD_atm', sig_atm_tr, axnames=['wavelength', 'iopt'],
                                            attrs=attrs_tmp)
                        attrs_tmp = profile['OD_sca_atm'].attrs
                        profile.rm_lut('OD_sca_atm')
                        profile.add_dataset('OD_sca_atm', sig_sca_tr, axnames=['wavelength', 'iopt'],
                                            attrs=attrs_tmp)
                        attrs_tmp = profile['ssa_atm'].attrs
                        profile.rm_lut('ssa_atm')
                        profile.add_dataset('ssa_atm', ssa_atm_tr, axnames=['wavelength', 'iopt'],
                                            attrs=attrs_tmp)
                        attrs_tmp = profile['pmol_atm'].attrs
                        profile.rm_lut('pmol_atm')
                        profile.add_dataset('pmol_atm', pmol_tr, axnames=['wavelength', 'iopt'],
                                            attrs=attrs_tmp)

        return profile

    def profile(self, wav, prof=None):
        """
        Calculate the profile of optical properties at given wavelengths.
        
        Computes atmospheric optical properties (extinction, scattering, absorption) 
        as a function of wavelength and altitude, including contributions from 
        Rayleigh scattering, aerosols, and gaseous absorbers (O3, NO2, and molecular gases).
        
        Parameters
        ----------
        wav : array-like or BandSet
            Wavelengths at which to calculate optical properties [nm].
            If not a BandSet, it will be converted to one.
        prof : Profile_base, optional
            Atmospheric profile containing altitude grids, temperature, pressure, 
            and density profiles. Default is None; uses self.prof if not provided.
            
        Returns
        -------
        profile : MLUT
            Multi-dimensional lookup table containing atmospheric optical properties 
            with dimensions as a function of wavelength and altitude (or iopt grid for 3D mode).
            
            Key datasets included:
            
            - **n_atm**: Atmospheric refractive index [wavelength, z_atm]
            - **T_atm**: Temperature profile [z_atm] (K)
            - **OD_r**: Rayleigh cumulated optical thickness [wavelength, z_atm] 
              or scattering coefficient (km⁻¹) in 3D mode
            - **OD_p**: Particles cumulated optical thickness [wavelength, z_atm]
              or extinction coefficient (km⁻¹) in 3D mode
            - **ssa_p_atm**: Particle single scattering albedo [wavelength, z_atm] or [wavelength, iopt]
            - **OD_g**: Cumulated gaseous absorption optical thickness [wavelength, z_atm]
              or absorption coefficient (km⁻¹) in 3D mode
            - **OD_atm**: Total cumulated optical thickness [wavelength, z_atm]
              or total extinction coefficient (km⁻¹) in 3D mode
            - **OD_sca_atm**: Cumulated scattering optical thickness [wavelength, z_atm]
            - **OD_abs_atm**: Cumulated absorption optical thickness [wavelength, z_atm]
            - **ssa_atm**: Total single scattering albedo [wavelength, z_atm]
            
        Notes
        -----
        The method can operate in two modes:
        
        - **1D Mode (OPT3D=False)**: Returns cumulated optical thicknesses with axes 
          [wavelength, z_atm]
        - **3D Mode (OPT3D=True)**: Returns extinction/absorption coefficients with axes 
          [wavelength, iopt] for use in 3D radiative transfer calculations
          
        Optical properties include:
        
        - Rayleigh scattering (from self.prof_ray or computed using Rayleigh optical depth)
        - Aerosol scattering and absorption from aerosol components
        - Gaseous absorption from ozone, NO2, and molecular gases (H2O, O2, CO2)
          using cross-sections (acs_o3, acs_no2) or REPTRAN/KDIS spectral databases
          
        Single scattering albedo is calculated as the ratio of scattering to extinction 
        optical thicknesses for each layer.
        """
        if not isinstance(wav, BandSet):
            wav = BandSet(wav)

        if prof is None:
            prof = self.prof

        dz = -diff1(prof.z)

        pro = MLUT()
        pro.add_axis('z_atm', prof.z)
        pro.add_axis('wavelength', wav[:])

        # refractive index
        n = refractivity(wav[:]*1e-3, prof.P, prof.T,prof.dens_co2/prof.dens_air*1e6)
        pro.add_dataset('n_atm', n, axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'atmospheric refractive index'})


        pro.add_dataset('T_atm', self.prof.T, axnames=['z_atm'],
                        attrs={'description':
                               'temperature (K)'})
        
        #
        # Rayleigh optical thickness
        #
        # cumulated Rayleigh optical thickness (wav, z)
        if self.prof_ray is None :
            tauray = rod(wav[:]*1e-3, prof.dens_co2/prof.dens_air*1e6, self.lat,
                     prof.z*1e3, prof.P)
            dtaur  = diff1(tauray, axis=1)
        else : 
            dtaur = self.prof_ray
            tauray= np.cumsum(dtaur,axis=1)

        if self.tauR is not None:
            # scale Rayleigh optical thickness
            if self.tauR.ndim == 1:
                # for each wavelength
                tauray *= self.tauR[:,None]/tauray[:,-1:]
            else:
                # scalar
                tauray *= self.tauR/tauray[:,-1:]

        assert tauray.ndim == 2

        # Rayleigh optical thickness
        dtaur = diff1(tauray, axis=1)
        if not self.OPT3D : 
            pro.add_dataset('OD_r', tauray, axnames=['wavelength', 'z_atm'],
            attrs={'description':
            'Cumulated rayleigh optical thickness'})
        else:
            if self.prof_ray is None:
                ray_coef = abs(dtaur/dz)
                ray_coef[~np.isfinite(ray_coef)] = 0.
            else:
                ray_coef = self.prof_ray
            pro.add_dataset('OD_r', ray_coef, axnames=['wavelength', 'iopt'],
            attrs={'description':
            'rayleigh scattering coefficient (km-1)'})

        #
        # Aerosol optical thickness and single scattering albedo
        #
        if self.prof_aer is None :
            dtaua = np.zeros((len(wav), len(prof.z)), dtype='float32')
            ssa_p = np.zeros((len(wav), len(prof.z)), dtype='float32')
            for comp in self.comp:
                dtau_, ssa_ = comp.dtau_ssa(wav[:], prof.z, prof.relative_humidity())
                dtaua += dtau_
                ssa_p+= dtau_ * ssa_
            ssa_p[dtaua!=0] /= dtaua[dtaua!=0]
            ssa_p[dtaua==0] = 1.
            taua = np.cumsum(dtaua, axis=1)

        else:
            (dtaua, ssa_p) = self.prof_aer
            taua= np.cumsum(dtaua,axis=1)

        if not self.OPT3D : 
            pro.add_dataset('OD_p', taua,
            axnames=['wavelength', 'z_atm'],
            attrs={'description':
            'Cumulated particles optical thickness at each wavelength'})
        else:
            if self.prof_aer is None:
                aer_coef = abs(dtaua/dz)
                aer_coef[~np.isfinite(aer_coef)] = 0.
            else : (aer_coef, ssa_p) = self.prof_aer
            pro.add_dataset('OD_p', aer_coef,
            axnames=['wavelength', 'iopt'],
            attrs={'description':
            'particles extinction coefficient (km-1)'})

        if not self.OPT3D:
            pro.add_dataset('ssa_p_atm', ssa_p, axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Particles single scattering albedo of the layer'})
        else :
            pro.add_dataset('ssa_p_atm', ssa_p, axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'Particles single scattering albedo of the layer'})


            
        if self.prof_abs is None:
            #
            # Ozone optical thickness
            #
            
            # Consider gaseous from reptran/kdis
            use_o3_acs  = True
            use_no2_acs = True
            if wav.use_reptran_kdis:
                tau_mol = wav.calc_profile(self.prof) * dz
                # If not reptran (i.e. Kdis case) we set 03 and NO2 to 0 (already calculated in Kdis)
                if not (str(wav.type_wav) == "<class 'smartg.reptran.REPTRAN_IBAND'>"):
                    all_kdis_gas = wav.data[0].band.kdis.species + wav.data[0].band.kdis.species_c
                    if 'no2' in all_kdis_gas :
                        use_no2_acs = False
                        tau_no2 = LUT(np.zeros((len(wav), len(prof.z)), dtype='float32') , axes=[wav[:], None], names=['wavelength', 'z_atm'])
                    if 'o3' in all_kdis_gas  :
                        use_o3_acs  = False
                        tau_o3 = LUT(np.zeros((len(wav), len(prof.z)), dtype='float32') , axes=[wav[:], None], names=['wavelength', 'z_atm'])
            else:
                tau_mol = np.zeros((len(wav), len(prof.z)), dtype='float32') * dz


            # Compute o3 and no2 (if kdis only compute them if not already computed)
            if use_no2_acs or use_o3_acs:
                # Commun part           
                T0 = 273.15  # in K
                T = LUT(prof.T, axes=[None], names=['z_atm'])# temperature variability in z
                if use_o3_acs:
                    # O3 optical thickness
                    min_wl = np.min(self.acs_o3.axes['wavelength'])
                    max_wl = np.max(self.acs_o3.axes['wavelength'])
                    C0 = self.acs_o3['O3_C0'].sub({'wavelength':Idx(wav[:], round=True, fill_value='extrema')})
                    C1 = self.acs_o3['O3_C1'].sub({'wavelength':Idx(wav[:], round=True, fill_value='extrema')})
                    C2 = self.acs_o3['O3_C2'].sub({'wavelength':Idx(wav[:], round=True, fill_value='extrema')})
                    tau_o3 = C0 + C1*(T - T0) + C2*(T - T0)*(T - T0)
                    tau_o3.data[~np.logical_and(wav[:]>min_wl, wav[:]<max_wl)] = 0.
                    tau_o3 *= prof.dens_o3 * 1e-15  # LUT in 10^(-20) cm2, convert in km-1
                    tau_o3 *= dz
                    tau_o3.data[tau_o3.data < 0] = 0
                if use_no2_acs:
                    # NO2 optical thickness
                    min_wl = np.min(self.acs_no2.axes['wavelength'])
                    max_wl = np.max(self.acs_no2.axes['wavelength'])
                    C0 = self.acs_no2['NO2_C0'].sub({'wavelength':Idx(wav[:], round=True, fill_value='extrema')})
                    C1 = self.acs_no2['NO2_C1'].sub({'wavelength':Idx(wav[:], round=True, fill_value='extrema')})
                    C2 = self.acs_no2['NO2_C2'].sub({'wavelength':Idx(wav[:], round=True, fill_value='extrema')})
                    tau_no2 = C0 + C1*(T - T0) + C2*(T - T0)*(T - T0)
                    tau_no2.data[~np.logical_and(wav[:]>min_wl, wav[:]<max_wl)] = 0.
                    tau_no2 *= prof.dens_no2 * 1e-15  # LUT in 10^(-20) cm2, convert in km-1
                    tau_no2 *= dz
                    tau_no2.data[tau_no2.data < 0] = 0
                
            #
            # Total gaseous optical thickness
            #
            dtaug = tau_o3 + tau_no2 + tau_mol
            taug = dtaug.apply(lambda x: np.cumsum(x, axis=1))

            if not self.OPT3D:
                pro.add_dataset('OD_g', taug.data,
                axnames=['wavelength', 'z_atm'],
                attrs={'description': 'Cumulated gaseous absorption optical thickness'})
            else:
                abs_coef = abs(dtaug.data/dz)
                abs_coef[~np.isfinite(abs_coef)] = 0.
                pro.add_dataset('OD_g', abs_coef, axnames=['wavelength', 'iopt'],
                  attrs={'description':
                         'gaseous absorption coefficient (km-1)'})

        else:
            dtaug = self.prof_abs
            taug  = np.cumsum(dtaug,axis=1)
            if not self.OPT3D:
                pro.add_dataset('OD_g', taug, axnames=['wavelength', 'z_atm'],
                  attrs={'description':
                         'Cumulated gaseous absorption optical thickness'})

            else: 
                abs_coef = self.prof_abs
                pro.add_dataset('OD_g', abs_coef, axnames=['wavelength', 'iopt'],
                  attrs={'description':
                         'gaseous absorption coefficient (km-1)'})

                
        #
        # Total optical thickness and other parameters
        #
        if not self.OPT3D:
            tau_tot = tauray + taua + taug[:,:]
            pro.add_dataset('OD_atm', tau_tot,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Cumulated extinction optical thickness'})

            tau_sca = np.cumsum(dtaur + dtaua*ssa_p, axis=1)
            pro.add_dataset('OD_sca_atm', tau_sca,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Cumulated scattering optical thickness'})

            tau_abs = np.cumsum(dtaug[:,:] + dtaua*(1-ssa_p), axis=1)
            pro.add_dataset('OD_abs_atm', tau_abs,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Cumulated absorption optical thickness'})

            with np.errstate(invalid='ignore', divide='ignore'):
                ssa = (dtaur+ dtaua*ssa_p)/diff1(tau_tot, axis=1)
            ssa[np.isnan(ssa)] = 1.
            pro.add_dataset('ssa_atm', ssa,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Single scattering albedo of the layer'})


        else:
            tot_coef = ray_coef + aer_coef + abs_coef[:,:]
            pro.add_dataset('OD_atm', tot_coef,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'extinction coefficient (km-1)'})

            sca_coef = ray_coef + aer_coef*ssa_p
            pro.add_dataset('OD_sca_atm', sca_coef,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'scattering coefficient (km-1)'})

            tabs_coef = abs_coef + aer_coef*(1.-ssa_p)
            pro.add_dataset('OD_abs_atm', tabs_coef,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'total absorption coefficient (km-1)'})

            with np.errstate(invalid='ignore', divide='ignore'):
                ssa = (ray_coef+ aer_coef*ssa_p)/tot_coef
            ssa[np.isnan(ssa)] = 1.
            pro.add_dataset('ssa_atm', ssa,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'Single scattering albedo of the layer'})

        with np.errstate(invalid='ignore', divide='ignore'):
            pmol = dtaur/(dtaur + dtaua*ssa_p)
        pmol[np.isnan(pmol)] = 1.
        if not self.OPT3D:
            pro.add_dataset('pmol_atm', pmol,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'Ratio of molecular scattering to total scattering of the layer'})
        else :
            pro.add_dataset('pmol_atm', pmol,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'Ratio of molecular scattering to total scattering of the layer'})

            
        pine = np.zeros_like(ssa)
        FQY1 = np.zeros_like(ssa)
        if not self.OPT3D:
            pro.add_dataset('pine_atm', pine,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'fraction of inelastic scattering of the layer'})
            pro.add_dataset('FQY1_atm', FQY1,
                        axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'fluoresence quantum yield of the layer'})
        else :
            pro.add_dataset('pine_atm', pine,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'fraction of inelastic scattering of the layer'})
            pro.add_dataset('FQY1_atm', FQY1,
                        axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'fluoresence quantum yield of the layer'})


        if self.prof_phases is not None:
            ipha, phases = self.prof_phases
            if not self.OPT3D:
                pro.add_dataset('iphase_atm', ipha, axnames=['wavelength', 'z_atm'],
                        attrs={'description':
                               'index of phase matrix'})
            else :
                pro.add_dataset('iphase_atm', ipha, axnames=['wavelength', 'iopt'],
                        attrs={'description':
                               'index of phase matrix'})

            # set the number of scattering angles to the maximum
            ip  = np.array([p.axis('theta_atm').size for p in phases]).argmax()
            theta = phases[ip].axis('theta_atm')
            pha = np.stack([p[:,Idx(theta)] for p in phases])
            pro.add_axis('theta_atm', theta)
            pro.add_dataset('phase_atm', pha, axnames=['iphase', 'stk', 'theta_atm'],
                    attrs={'description':
                           'phase matrices'})
        # Pure 3D
        #
        if self.OPT3D:
            (iopt, iabs, pmin, pmax, neighbour) = self.cells
            pro.add_dataset('iopt_atm', iopt, axnames=['icell'])
            pro.add_dataset('iabs_atm', iabs, axnames=['icell'])
            pro.add_dataset('pmin_atm', pmin, axnames=['xyz', 'icell'])
            pro.add_dataset('pmax_atm', pmax, axnames=['xyz', 'icell'])
            pro.add_dataset('neighbour_atm', neighbour, axnames=['faces', 'icell'])

        return pro

    def phase(self, wav, NBTHETA=721, conv_Iparper=True):
        """
        Calculate phase matrix of aerosols and clouds at specified wavelengths.
        
        Computes weighted average phase functions for all aerosol components 
        using the reduced atmospheric profile. Each component's contribution 
        is weighted by its optical depth and single scattering albedo.
        
        Parameters
        ----------
        wav : scalar or array-like
            Wavelengths at which to calculate phase matrix [nm].
            If scalar, will be converted to 1-D array.
        NBTHETA : int, optional
            Number of scattering angles for angle resampling. Default is 721,
            corresponding to angles from 0° to 180°.
        conv_Iparper : bool, optional
            If True (default), converts the phase matrix from I/Q Stokes convention 
            to Ipar/Iper convention. This applies general conversion formulas valid 
            for both spherical and non-spherical particles.
            
        Returns
        -------
        phase_matrix : LUT or None
            Lookup table containing the weighted average phase matrix with axes 
            [wav_phase, z_phase, stk, theta_atm] if aerosol components are present.
            Shape is (len(wav), nz, nphamat, NBTHETA) where:
            - nz: number of altitude levels in the reduced profile (self.pfgrid)
            - nphamat = 4 for spherical particles only (phase matrix unique terms P11, P21, P33, P34)
            - nphamat = 6 for spherical and non-spherical particles (additional terms P22, P44)
            - theta_atm: scattering angles from 0° to 180°
            
            Returns None if no aerosol components are defined (self.comp is empty).
            
        Notes
        -----
        **Weighted averaging:** The phase matrix is computed as a weighted average 
        across all aerosol components defined in the comp attribute:
        
        pha_total = [∑_i (pha_i x Δτ_i x ssa_i)) / (∑_i (Δτ_i x ssa_i)]
        
        where:
        
        - pha_i is the phase matrix of component i
        - Δτ_i is the optical depth of component i  
        - ssa_i is the single scattering albedo of component i
        
        The relative humidity used for calculations is obtained from 
        the reduced profile (self.prof_red).
        """
        wav = np.atleast_1d(wav)
        pha = 0.
        norm = 0.
        rh = self.prof_red.relative_humidity()

        for comp in self.comp:
            dtau, ssa_p = comp.dtau_ssa(wav, self.pfgrid, rh=rh)
            dtau = dtau[:,1:][:,:,None,None]
            ssa_p = ssa_p[:,1:][:,:,None,None]
            pha += comp.phase(wav, self.pfgrid, rh, NBTHETA=NBTHETA, conv_Iparper=conv_Iparper)*dtau*ssa_p
            norm += dtau*ssa_p
        if len(self.comp) > 0:
            pha /= norm
            pha.data[np.isnan(pha.data)] = 0.

            return pha
        else:
            return None

    def calc_split(self, wav, phase=True, NBTHETA=721):
        """
        Computes atmospheric optical properties at specified wavelengths and 
        separates them into decomposed components (absorption, Rayleigh scattering, 
        aerosols, and phase functions). These returned profiles can be used as 
        alternative inputs to initialize a new AtmAFGL instance.
        
        Parameters
        ----------
        wav : scalar or array-like
            Wavelengths at which to calculate optical properties [nm].
        phase : bool, optional
            If True (default), calculates phase functions. Set to False to skip 
            phase function computations for faster execution.
        NBTHETA : int, optional
            Number of scattering angles for phase function resampling. Default is 721,
            corresponding to angles from 0° to 180°. Only used if phase=True.
            
        Returns
        -------
        prof_abs : ndarray
            Gaseous absorption coefficient [wavelength, altitude] (km⁻¹).
            Differential optical thickness for absorption from cumulated profile.
        prof_ray : ndarray
            Rayleigh scattering coefficient [wavelength, altitude] (km⁻¹).
            Differential optical thickness for Rayleigh from cumulated profile.
        (prof_aer, ssa_aer) : tuple
            Aerosol profiles with:
            
            - prof_aer: Aerosol extinction coefficient [wavelength, altitude] (km⁻¹)
            - ssa_aer: Particle single scattering albedo [wavelength, altitude]
            
        (pro_iphase, pro_phases) : tuple
            Phase function profiles with:
            
            - pro_iphase: Phase matrix indices array [wavelength, altitude]
            - pro_phases: List of phase matrix LUT objects, one for each phase index
            
        Notes
        -----
        This method is useful for decomposing atmospheric optical properties into 
        separate components. The returned profiles can be used to recreate the 
        atmospheric model by passing them as alternative inputs:
        
        - prof_abs: passed as the prof_abs parameter
        - prof_ray: passed as the prof_ray parameter  
        - (prof_aer, ssa_aer): passed as the prof_aer parameter
        - (pro_iphase, pro_phases): passed to phase parameter handling
        
        All returned arrays are cast to float32 for memory efficiency.
        
        Examples
        --------
        >>> atm = AtmAFGL(...)
        >>> prof_abs, prof_ray, (prof_aer, ssa_aer), (pro_iphase, pro_phases) = atm.calc_split(wav=500.)
        """
        pro = self.calc(wav=wav, phase=phase, NBTHETA=NBTHETA)
        pro_aer = diff1(pro['OD_p'].data.astype(np.float32), axis=1)
        ssa_aer = pro['ssa_p_atm'].data
        pro_ray = diff1(pro['OD_r'].data.astype(np.float32), axis=1)
        pro_abs = diff1(pro['OD_g'].data.astype(np.float32), axis=1)
        pro_iphase = pro['iphase_atm'].data
        pro_phases = [pro['phase_atm'].sub({'iphase':i}) for i in range(pro_iphase.max()+1)]

        return pro_abs, pro_ray, (pro_aer, ssa_aer), (pro_iphase, pro_phases)


def read_phase(filename, standard=False, kind='atm'):
    '''
    Read phase function from filename as a LUT

    standard: standard phase function definition, otherwise Smart-g definition
    '''
    data2 = pd.read_csv(filename, sep=r'\s+', header=None)

    theta = np.array(data2[0])
    pha   = np.array(data2[[1,2,3,4]])

    if standard:
        pha[:,0] = data2[1] + data2[2]
        pha[:,1] = data2[1] - data2[2]
        pha[:,2] = data2[3]
        pha[:,3] = data2[4]

    # Normalization to Sum_-1_+1 P(mu) dmu = 2.
    f = (pha[:,0] + pha[:,1])/2.
    mu= np.cos(np.radians(theta))
    Norm = np.trapezoid(f,-mu)
    pha *= (2./abs(Norm))

    P = LUT(pha.swapaxes(0, 1),  # stk, theta
            axes=[None, theta],
            names=['stk', 'theta_'+kind],
           )

    return P


class Profile_base(object):
    """
    Atmospheric profile with physical properties.

    Reads and processes atmospheric profiles from files (NetCDF or libratran format).
    Allows customization of ozone, water vapor, and pressure profiles. Automatically
    scales gaseous constituents to specified total column amounts.

    Parameters
    ----------
    atm_filename : str | Path
        Path to atmospheric profile file. Accepts .nc (NetCDF) or .dat (libratran) formats.
        If only filename is provided (no path), the auxdata directory is automatically prepended.
        If no suffix is provided, .nc is assumed by default.
    O3 : float | None, optional
        Total ozone column in Dobson units (DU). If None, uses the value from the
        atmospheric profile. The O3 profile is scaled to match this column amount.
        Default: None
    H2O : float | None, optional
        Total water vapor column in g/cm². If None, uses the value from the
        atmospheric profile. The H2O profile is scaled to match this column amount.
        Default: None
    NO2 : bool | None, optional
        Include NO2 absorption. If False, NO2 density is set to zero.
        Default: True
    P0 : float | None, optional
        Sea surface (bottom layer) pressure in hPa. If None, uses the pressure
        from the atmospheric profile. Scales all pressure values proportionally.
        Default: None
    RH_cst : float | None, optional
        Force relative humidity to be constant at this value. If None, relative
        humidity is recalculated from the temperature and water vapor profiles.
        Default: None
    US : bool | None, optional
        Use U.S. Standard atmosphere convention. Application-specific flag.
        Default: True
    O3_H2O_alt : float | None, optional
        Altitude (km) at which the specified O3 and H2O values apply. When specified,
        the O3 and H2O profiles are scaled such that the column amount from TOA to this
        altitude matches the provided O3 and H2O values. The full gaseous distribution
        from TOA to ground is preserved; only the scaling factor is adjusted to match
        the constraint at this reference altitude.
        Default: None

    Notes
    -----
    File format support:
    - .nc (NetCDF): Expects variables 'P', 'T', 'dens', 'H2O', 'O3', etc. with
      dimension 'z_atm' for altitude
    - .dat (libratran): Text format with header line containing variable names
      (e.g., 'z(km) p(mb) T(K) air(cm-3) o3(cm-3) ...')
    """
    def __init__(self, atm_filename, O3=None, H2O=None, NO2=True, P0=None, RH_cst=None, US=True, O3_H2O_alt=None):

        if atm_filename is None:
            return
        atm_filename = Path(atm_filename)
        self.atm_filename = atm_filename

        if not atm_filename.is_file():
            raise FileNotFoundError(f"Atmospheric profile file not found: {atm_filename}")

        if atm_filename.suffix == '.dat':
            with open(atm_filename) as f:
                lines = f.readlines()

            desc = None
            desc = ''
            n=0
            for line in lines:
                if ('z(km)' in line) and ('p(mb)' in line) and ('T(K)' in line) and ('air(cm-3)' in line) :
                    desc = line
                    break
                else:
                    n+=1
            if desc=='' : n = 0

            if desc is not None:
                #data = np.loadtxt(atm_filename, dtype=np.float32, comments="#", skiprows=n)
                data = pd.read_csv(atm_filename, comment="#", header=None, sep=r'\s+', dtype=np.float32, skiprows=n).values
                self.z        = data[:,0] # Altitude in km
                self.P        = data[:,1] # pressure in hPa
                self.T        = data[:,2] # temperature in K
                self.dens_air = data[:,3] # Air density in cm-3
                data2 = np.zeros((data.shape[0], 5))
                for i,gas in enumerate(['o3','o2','h2o','co2','no2']):
                    try : 
                        ind = desc.split().index(gas+'(cm-3)')
                        data2[:,i] = data[:, ind-1]
                    except ValueError:
                        data2[:,i] = 0.
                self.dens_o3  = data2[:,0] # Ozone density in cm-3
                self.dens_o2  = data2[:,1] # O2 density in cm-3
                self.dens_h2o = data2[:,2] # H2O density in cm-3
                self.dens_co2 = data2[:,3] # CO2 density in cm-3
                self.dens_no2 = data2[:,4] # NO2 density in cm-3
                nz = data.shape[0]
                self.dens_ch4 = np.zeros(nz, dtype=np.float32)
                self.dens_co = np.zeros(nz, dtype=np.float32)
                self.dens_n2o = np.zeros(nz, dtype=np.float32)
                self.dens_n2 = np.zeros(nz, dtype=np.float32)
                self.dens_so2 = np.zeros(nz, dtype=np.float32)
            else:
                raise NameError('Invalid atmospheric file format')
        elif atm_filename.suffix == '.nc':
            data = read_mlut(atm_filename)
            self.z        = data.axes['z_atm'] # Altitude in km
            self.P        = data['P'].data     # pressure in hPa
            self.T        = data['T'].data     # temperature in K
            self.dens_air = data['dens'].data  # Air density in cm-3
            self.dens_h2o = data['H2O'].data   # H2O density in cm-3
            self.dens_o3 = data['O3'].data     # O3 density in cm-3
            self.dens_n2o = data['N2O'].data   # N2O density in cm-3
            self.dens_co = data['CO'].data     # CO density in cm-3
            self.dens_ch4 = data['CH4'].data   # CH4 density in cm-3
            self.dens_co2 = data['CO2'].data   # CO2 density in cm-3
            self.dens_o2 = data['O2'].data     # O2 density in cm-3
            self.dens_n2 = data['N2'].data     # N2 density in cm-3
            self.dens_no2 = data['NO2'].data   # NO2 density in cm-3
            self.dens_so2 = data['SO2'].data   # SO2 density in cm-3

        self.RH_cst   = RH_cst

        # scale to specified total O3 content
        if O3 is not None:
            if O3_H2O_alt is None:
                self.dens_o3 *= 2.69e16 * O3 / (simpson(y=self.dens_o3, x=-self.z) * 1e5)
            else:
                f_dens_o3 = interp1d(self.z, self.dens_o3, fill_value='extrapolate')
                z_alt = np.append(self.z[self.z>O3_H2O_alt], O3_H2O_alt)
                dens_o3_alt = f_dens_o3(z_alt)
                o3_afgl = (simpson(dens_o3_alt, -z_alt) * 1e5)/2.69e16
                self.dens_o3 *= O3/o3_afgl
            if O3==0 : self.dens_o3[:] = 0.

        # scale to total H2O content
        if H2O is not None:
            M_H2O = 18.015 # g/mol
            Avogadro = constants.value('Avogadro constant')
            if O3_H2O_alt is None:
                self.dens_h2o *= H2O/ M_H2O * Avogadro / (simpson(y=self.dens_h2o, x=-self.z) * 1e5)
            else:
                f_dens_h2o = interp1d(self.z, self.dens_h2o, fill_value='extrapolate')
                z_alt = np.append(self.z[self.z>O3_H2O_alt], O3_H2O_alt)
                dens_h2o_alt = f_dens_h2o(z_alt)
                h2o_afgl = (simpson(y=dens_h2o_alt, x=-z_alt) * 1e5 * M_H2O)/Avogadro
                self.dens_h2o *= H2O/h2o_afgl
            if H2O==0 : self.dens_h2o[:] = 0.

        if P0 is not None:
            self.P *= P0/self.P[-1]

        if not NO2:
            self.dens_no2[:] = 0.

    def regrid(self, znew):
        """
        Regrid atmospheric profile to a new altitude grid.

        Interpolates all atmospheric properties (pressure, temperature, and gas densities)
        from the current altitude grid to a new altitude grid using linear interpolation.
        Special boundary conditions are applied for pressure (using bounds_error=False with
        specific fill values) and temperature (using extrapolation).

        Parameters
        ----------
        znew : 1-D ndarray
            New altitude grid in kilometers. Must be a 1-D array of altitude values.
            The new grid can be coarser, finer, or irregular compared to the original grid.

        Returns
        -------
        Profile_base
            New Profile_base object with all atmospheric properties interpolated to the
            new altitude grid `znew`. The following attributes are interpolated:
            - z: altitude (km)
            - P: pressure (hPa)
            - T: temperature (K)
            - dens_air: air density (molecule/cm³)
            - dens_o3: ozone density (molecule/cm³)
            - dens_o2: oxygen density (molecule/cm³)
            - dens_h2o: water vapor density (molecule/cm³)
            - dens_co2: carbon dioxide density (molecule/cm³)
            - dens_no2: nitrogen dioxide density (molecule/cm³)
            - dens_ch4: methane density (molecule/cm³)
            - dens_co: carbon monoxide density (molecule/cm³)
            - dens_n2o: nitrous oxide density (molecule/cm³)
            - dens_n2: nitrogen density (molecule/cm³)
            - dens_so2: sulfur dioxide density (molecule/cm³)
            - RH_cst: constant relative humidity (None | float)
        """

        prof = Profile_base(None)
        z = self.z
        prof.z = znew
        try:
            prof.P = interp1d(z, self.P, bounds_error=False, fill_value=(1012., 1e-5))(znew)
            #prof.P = np.interp(znew, z, self.P, right=1012., left=1e-5)
        except ValueError:
            print('Error interpolating ({}, {}) -> ({}, {})'.format(z[0], z[-1], znew[0], znew[-1]))
            print('atm_filename = {}'.format(self.atm_filename))
            raise
        prof.T = interp1d(z, self.T, fill_value='extrapolate')(znew) # No found np.interp with extrapolate

        prof.dens_air = interp1d(z, self.dens_air, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_o3  = interp1d(z, self.dens_o3, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_o2  = interp1d(z, self.dens_o2, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_h2o = interp1d(z, self.dens_h2o, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_co2 = interp1d(z, self.dens_co2, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_no2 = interp1d(z, self.dens_no2, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_ch4 = interp1d(z, self.dens_ch4, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_co  = interp1d(z, self.dens_co, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_n2o = interp1d(z, self.dens_n2o, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_n2  = interp1d(z, self.dens_n2, bounds_error=False, fill_value=(0., 0.))  (znew)
        prof.dens_so2 = interp1d(z, self.dens_so2, bounds_error=False, fill_value=(0., 0.))  (znew)

        prof.RH_cst   = self.RH_cst

        return prof

    def relative_humidity(self):
        """
        Calculate relative humidity profile for each atmospheric layer.
        
        Computes the relative humidity at all altitude levels based on the atmospheric 
        profile's water vapor density, air density, pressure, and temperature. 
        
        Returns
        -------
        rh : ndarray
            Relative humidity profile [%] with shape matching altitude grid.
            Values can exceed 100% if atmospheric conditions are supersaturated.
            
        Notes
        -----
        The relative humidity is calculated as:
        
        rh = (p_H₂O / p_sat) x 100
        
        where:
        
        - p_H₂O is the partial pressure of water vapor (from density ratio)
        - p_sat is the saturation vapor pressure at the given temperature
        
        If RH_cst (constant relative humidity) was set during initialization, 
        that constant value is returned for all layers instead of calculating 
        from the density/temperature profile.
        
        The saturation pressure calculation accounts for both water and ice phases 
        using temperature-dependent formulas.
        """
        if self.RH_cst is not None : 
            rh[:] = self.RH_cst
        else:
            p_h2o = (self.dens_h2o / self.dens_air) * self.P
            p_sat = saturation_pressure(self.T)
            rh = (p_h2o / p_sat) * 100 

        return rh


def saturation_pressure(T):
    """
    Calculate saturation vapor pressure for water and ice phases.
    
    Uses the Huang (2018) empirical formula, which provides accurate
    saturation vapor pressure calculations for both liquid water and ice phases.
    
    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin [K]
        
    Returns
    -------
    sat_press : float or numpy.ndarray
        Saturation vapor pressure [hPa]
        
    Notes
    -----
    The function automatically selects the appropriate formula based on temperature:
    - For T > 273.15 K (0°C): liquid water phase formula
    - For T ≤ 273.15 K (0°C): ice phase formula
    
    References
    ----------
    Huang, J. (2018). A Simple Accurate Formula for Calculating Saturation 
    Vapor Pressure of Water and Ice. Journal of Applied Meteorology and Climatology, 57(6), 1265-1272.
    """
    tc = T-273.15 # temperature in C°
    sat_press = np.zeros_like(tc)
    
    is_water = tc > 0
    is_ice = np.logical_not(is_water)
    
    sat_press[is_water] = ( np.exp(34.494 - 4924.99 / (tc[is_water] + 237.1)) ) / \
                          ( (tc[is_water] + 105)**1.57 )

    sat_press[is_ice] = ( np.exp(43.494 - (6545.8 / (tc[is_ice] + 278))) ) / \
                        ( (tc[is_ice] + 868)**2 )
    return sat_press * 1e-2


def FN2(lam):
    """
    Compute the depolarization factor of N2 as a function of wavelength.

    Parameters
    ----------
    lam : float | ndarray
        Wavelength in micrometers (μm).

    Returns
    -------
    float | ndarray
        Depolarization factor of N2. Same shape as input `lam`.
    """
    return 1.034 + 3.17 *1e-4 *lam**(-2)


def FO2(lam):
    """
    Compute the depolarization factor of O2 as a function of wavelength.

    Parameters
    ----------
    lam : float | ndarray
        Wavelength in micrometers (μm).

    Returns
    -------
    float | ndarray
        Depolarization factor of O2. Same shape as input `lam`.
    """
    return 1.096 + 1.385 *1e-3 *lam**(-2) + 1.448 *1e-4 *lam**(-4)


def Fair(lam, co2):
    """
    Calculates the depolarization factor for air using a composite formula based on
    the depolarization factors of N2 and O2, and the CO2 concentration. Produces
    a 2-D array with one value per wavelength-layer combination.

    Parameters
    ----------
    lam : 1-D ndarray
        Wavelength values in micrometers (μm). Shape: (N,)
    co2 : 1-D ndarray
        CO2 concentration in parts per million (ppm). Shape: (M,)

    Returns
    -------
    ndarray
        Depolarization factor of air. Shape: (N, M), where N is the number of
        wavelengths and M is the number of layers.
    """
    _FN2 = FN2(lam).reshape((-1,1))
    _FO2 = FO2(lam).reshape((-1,1))
    _CO2 = co2.reshape((1,-1))

    return ((78.084 * _FN2 + 20.946 * _FO2 + 0.934 +
            _CO2*1e-4 *1.15)/(78.084+20.946+0.934+_CO2*1e-4))


def n300(lam):
    ''' index of refraction of dry air  (300 ppm CO2)
        lam : um
    '''
    return 1e-8 * ( 8060.51 + 2480990/(132.274 - lam**(-2)) + 17455.7/(39.32957 - lam**(-2))) + 1.


def n_air(lam, co2):
    ''' index of refraction of dry air (N wavelengths x M layers)
        lam : um (N)
        co2 : ppm (M)
    '''
    N300 = n300(lam).reshape((-1,1))
    CO2 = co2.reshape((1,-1))
    return ((N300 - 1) * (1 + 0.54*(CO2*1e-6 - 0.0003)) + 1.)

def ma(co2):
    ''' molecular volume
        co2 : ppm
    '''
    return 15.0556 * co2*1e-6 + 28.9595

def raycrs(lam, co2):
    """
    Compute the Rayleigh cross section 
    
    Parameters:
    -----------
    lam : 1-D ndarray
        The wavelength(s) in um
    co2 : float | 1-D ndarray
        CO2 concentration(s) in ppm

    Returns:
    out : 2-D ndarray
        The Rayleigh cross section (N wavelengths x M layers)
    """

    if not isinstance(lam, np.ndarray):
        raise ValueError("The parameter lam must be a 1-D np.ndarray.")
    if not np.isscalar(co2) and not isinstance(co2, np.ndarray):
        raise ValueError("The parameter co2 must be a scalar or a 1-D ndarray.")

    # Ensure float64 due to numpy 2
    lam = lam.astype(np.float64)
    co2 = np.float64(co2)

    Avogadro = constants.value('Avogadro constant')
    Ns = Avogadro/22.4141 * 273.15/288.15 * 1e-3
    nn2 = n_air(lam, co2)**2

    return (24*np.pi**3 * (nn2-1)**2/(lam[:,None]*1e-4)**4/Ns**2/(nn2+2)**2 * Fair(lam, co2))

def g0(lat):
    ''' gravity acceleration at the ground
        lat : deg
    '''
    assert isnumeric(lat)
    return (980.6160 * (1. - 0.0026372 * np.cos(2*lat*np.pi/180.)
            + 0.0000059 * np.cos(2*lat*np.pi/180.)**2))

def g(lat, z) :
    ''' gravity acceleration at altitude z
        lat : deg (scalar)
        z : m
    '''
    assert isnumeric(lat)
    return (g0(lat) - (3.085462 * 1.e-4 + 2.27 * 1.e-7 * np.cos(2*lat*np.pi/180.)) * z
            + (7.254 * 1e-11 + 1e-13 * np.cos(2*lat*np.pi/180.)) * z**2
            - (1.517 * 1e-17 + 6 * 1e-20 * np.cos(2*lat*np.pi/180.)) * z**3)

def rod(lam, co2=400., lat=45., z=0., P=1013.25, pressure='surface'):
    """
    Rayleigh optical depth from Bodhaine et al, 99 (N wavelengths x M layers)
        lam : wavelength in um (N)
        co2 : ppm (M)
        lat : deg (scalar)
        z : altitude in m (M)
        P : pressure in hPa (M)
            (surface or sea-level)
        pressure: str
            - 'surface': P provided at altitude z
            - 'sea-level': P provided at altitude 0
    """
    Avogadro = constants.value('Avogadro constant')
    zs = 0.73737 * z + 5517.56  # effective mass-weighted altitude
    G = g(lat, zs)
    # air pressure at the pixel (i.e. at altitude) in hPa
    if pressure == 'sea-level':
        Psurf = (P * (1. - 0.0065 * z / 288.15) ** 5.255) * 1000.  # air pressure at pixel location in dyn / cm2, which is hPa * 1000
    elif pressure == 'surface':
        Psurf = P * 1000.  # convert to dyn/cm2
    else:
        raise ValueError(f'Invalid pressure type ({pressure})')

    return raycrs(lam, co2) * Psurf * Avogadro/ma(co2)/G

def refractivity(lam,P,T,co2):
    ''' Refractivity of air
        lam : um (N)
        P   : hPa (M)
        T   : K (M)
        co2 : ppm (M)
    '''
    p= P*100.
    t = T-273.15
    Ntp = 1 + (n_air(lam[:],co2) - 1) * p * (1.+p*(60.1-0.972*t)*1e-10)\
        /(96095.43 * (1 + 0.003661 * t))
    return Ntp

def diff1(A, axis=0, samesize=True):
    if samesize:
        B = np.zeros_like(A)
        key = [slice(None)]*A.ndim
        key[axis] = slice(1, None, None)
        B[tuple(key)] = np.diff(A, axis=axis)[:]
        return B
    else:
        return np.diff(A, axis=axis)

def average(A):
    '''
    returns average value within each interval

    A: input array, size N
    returns averaged array of size N-1
    '''
    return 0.5*(A[1:] + A[:-1])


def isiterable(x):
    return hasattr(x, '__iter__')

def isnumeric(x):
    try:
        float(x)
        return True
    except TypeError:
        return False
    

def od2k(prof, dataset, axis=1, zreverse=False):
    '''
    From integrated Optical Depth to vertical coefficient in km-1)

    Inputs:
        prof : atmosphere profile (MLUT) as computed by calc method of AtmAFGL
        dataset : name of the dataset to be processed

    Keywords:
        axis : number of the vertical dimension, default 1
        zreverse : invert the vertical axis, default False

    Outputs:
        2D array (NW, NZ) of vertical coefficient (km-1)
    '''
    ot = diff1(prof[dataset].data.astype(np.float32), axis=axis)
    #dz = diff1(prof.axis('z_atm')).astype(np.float32)
    zz = prof.axis('z_atm') if not isinstance(prof, xr.Dataset) else prof['z_atm']
    dz = diff1(zz).astype(np.float32)
    
    
    k  = abs(ot/dz)
    k[np.isnan(k)] = 0
    sl = slice(None,None,-1 if zreverse else 1)
    
    return k[:,sl]


def BPlanck(wav, T):
    a = 2.0*Planck*speed_of_light**2
    b = Planck*speed_of_light/(wav*Boltzmann*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity


def get_aer_dist_integral(Z, H_min, H_max):
    return (-(Z)*np.exp(-H_max/Z) + (Z)*np.exp(-H_min/Z))


def check_date(dates, year):
    """
    Validate that all dates are from a single year and match the provided year.

    Parameters
    ----------
    dates : 1d-array | list 
        Dates in format "dd:mm:yyyy" (numpy array or list)
    year : int
        Expected year in format yyyy

    Raises
    ------
    ValueError
        If dates contain multiple years or if the year doesn't match the expected year

    Returns
    -------
    None
    """
    if len(dates) == 0:
        raise ValueError("dates cannot be empty")

    # Extract years from dates using list comprehension
    years = np.unique([int(date.split(':')[-1]) for date in dates])

    if years.size != 1:
        raise ValueError(
            f"Multiple years found in dates: {years}. "
            "Data spanning multiple years is not supported for 'day of year' dimension."
        )

    extracted_year = years[0]
    if extracted_year != year:
        raise ValueError(
            f"Date year ({extracted_year}) does not match expected year ({year})."
        )


def read_Aeronet_AOD(file, year):
    """
    Extract AOD data from Aeronet file

    Parameters
    ----------
    file : str | Pathlike
        Extinction AOD aeronet file path
    year : int
        The year for 'Day_of_Year(Fraction)' dimension creation

    Returns
    -------
    out : xr.DataArray
        Lookup table with extinction AOD as function of Day_of_Year(Fraction) and wavelength
    """

    AOD = pd.read_csv(file, sep=',', skiprows=6)
    NTIME_AOD = AOD.index.size

    check_date (dates=AOD["Date(dd:mm:yyyy)"].values, year=year)

    wav_ext = []
    for key in AOD.keys():
        if 'AOD_Extinction-Total' in key:
            str_bis = key.split('[')
            wav_ext.append(float(str_bis[1][:-3]))
    wav_ext = np.unique(wav_ext)        
    NWAV_EXT = len(wav_ext)

    mat_ext = np.zeros((NTIME_AOD, NWAV_EXT), dtype=np.float64)
    for itime in range (0, NTIME_AOD):
        for iwav, wav in enumerate(wav_ext):
            key = 'AOD_Extinction-Total[' + str(int(wav)) + 'nm]'
            mat_ext[itime, iwav] = AOD.iloc[itime][key]

    AOD_ext_lut = xr.DataArray(
                               mat_ext,
                               coords={
                                        'Day_of_Year(Fraction)': AOD["Day_of_Year(Fraction)"].values,
                                        'wavelength': wav_ext
                                       },
                               dims=['Day_of_Year(Fraction)', 'wavelength'],
                               name='aod'
                              )

    return AOD_ext_lut


def read_Aeronet_SSA(file, year):
    """
    Extract SSA data from Aeronet file

    Parameters
    ----------
    file : str | Pathlike
        Single scattering albedo aeronet file path
    year : int
        The year for 'Day_of_Year(Fraction)' dimension creation

    Returns
    -------
    out : xr.DataArray
        Lookup table with single scattering albedo as function of Day_of_Year(Fraction) 
        and wavelength
    """
    SSA = pd.read_csv(file, sep=',', skiprows=6)
    NTIME_SSA = SSA.index.size

    check_date (dates=SSA["Date(dd:mm:yyyy)"].values, year=year)

    wav_ssa = []
    for key in SSA.keys():
        if 'Single_Scattering_Albedo' in key:
            str_bis = key.split('[')
            wav_ssa.append(float(str_bis[1][:-3]))
    wav_ssa = np.unique(wav_ssa)        
    NWAV_SSA = len(wav_ssa)

    mat_ssa = np.zeros((NTIME_SSA, NWAV_SSA), dtype=np.float64)
    for itime in range (0, NTIME_SSA):
        for iwav, wav in enumerate(wav_ssa):
            key = 'Single_Scattering_Albedo[' + str(int(wav)) + 'nm]'
            mat_ssa[itime, iwav] = SSA.iloc[itime][key]

    SSA_lut = xr.DataArray(
                           mat_ssa,
                           coords={
                                    'Day_of_Year(Fraction)': SSA["Day_of_Year(Fraction)"].values,
                                    'wavelength': wav_ssa
                                   },
                           dims=['Day_of_Year(Fraction)', 'wavelength'],
                           name='ssa'
                           )

    return SSA_lut


def read_Aeronet_PFN(file, year):
    """
    Extract PFN data from Aeronet file

    Parameters
    ----------
    file : str | Pathlike
        Phase matrix aeronet file path
    year : int
        The year for 'Day_of_Year(Fraction)' dimension creation

    Returns
    -------
    out : xr.DataArray
        Lookup table with phase function matrix as function of Day_of_Year(Fraction), 
        wavelength and theta_atm
    """
    PFN = pd.read_csv(file, sep=',', skiprows=6)
    PFN = PFN[PFN['Phase_Function_Mode']=='Total'] # take only total of fine + coarse
    NTIME_PFN = PFN.index.size

    check_date (dates=PFN["Date(dd:mm:yyyy)"].values, year=year)

    ang = []; wav_pfn = []
    for key in PFN.keys():
        if '0000' in key:
            str_bis = key.split('[')
            ang.append(float(str_bis[0]))
            wav_pfn.append(float(str_bis[1][:-3]))
    ang = np.unique(ang)[::-1]
    wav_pfn = np.unique(wav_pfn)
    NANG = len(ang)
    NWAV_PFN = len(wav_pfn)

    mat_pfn = np.zeros((NTIME_PFN, NWAV_PFN, NANG), dtype=np.float64)
    for itime in range (0, NTIME_PFN):
        for iwav, wav in enumerate(wav_pfn):
            for iang, ag in enumerate(ang):
                ang_str = "%.6f" % float(ag)
                key = ang_str + "[" + str(int(wav)) + 'nm]'
                mat_pfn[itime, iwav, iang] = PFN.iloc[itime][key]
    
    phase_lut = xr.DataArray(
                              mat_pfn,
                              coords={
                                       'Day_of_Year(Fraction)': PFN["Day_of_Year(Fraction)"].values,
                                       'wavelength': wav_pfn,
                                       'theta_atm': ang
                                      },
                              dims=['Day_of_Year(Fraction)', 'wavelength', 'theta_atm'],
                              name='pfn'
                            )

    return phase_lut


def atm_pro_from_aeronet(date, time, aod_file, ssa_file, pfn_file, b_wav, 
                         pfwav=None, grid=None,  
                         atm_name="afglt", 
                         P0=None, O3=None, H2O=None, O3_H2O_alt=None,
                         H_mix_min=0., H_mix_max=2., Z_mix=8):
    """
    Create an atmosphere profil from aeronet files

    Parameters
    ----------
    date : str
        Date in the following format -> "yyyy-mm-dd"
    time : str
        Time in the following format -> "hh:mm:ss"
    aod_file : str | LUT
        Extinction AOD aeronet file (finishing by .aod) or aod LUT
    ssa_file : str | LUT
        Single scattering albedo aeronet file (finishing by .ssa) or ssa LUT
    pfn_file : str | LUT
        Phase matrix aeronet file (finishing by .pfn) or pfn LUT
    b_wav : list | BandSet
        Kdis bands or list of wavelenghts
    pfwav : list
        List of wavelenghts where the phase functions are computed
    grid : array-like
        Altitude grid profil
    atm_name : str
        The atmAFGL atmosphere used
    P0 : float
        Surface pressure
    O3 : float
        Scale ozone vertical column (Dobson units)
    H2O : float
        Scale Water vertical column
    O3_H2O_alt : float
        Altitude of H2O and O3 values, by default None and scale from z=0km
    H_mix_min : float, optional
        Force min altitude of the mixture
    H_mix_max : float, optional
        Force max altitude of the mixture
    Z_mix : float, optional
        Force scale height (see notes) of the mixture

    Returns
    -------
    out : MLUT
        The atmophere profil. Similar to the output of the calc method of AtmAFGL.

    Notes
    -----
    The scale height (see Hess et al. 2004) is the variable Z in the following equation:

    - :math:`N(h) = N(0)exp(-h/Z)`

    with N the number density and h the altitude
    """

    pd_date = pd.Timestamp(date + " " + time)
    nb_sec_day = 24*60*60 # number of seconds in one day
    day_frac = 1 - ( (nb_sec_day - (pd_date.hour*60*60 + pd_date.minute*60 + pd_date.second)) / nb_sec_day )
    day_year_frac = pd_date.day_of_year + day_frac; print('day_year_frac =', day_year_frac)
    year = pd_date.year

    if isinstance(aod_file, xr.DataArray): aod_lut = aod_file
    else: aod_lut = read_Aeronet_AOD(aod_file, year=year)
    if isinstance(ssa_file, xr.DataArray): ssa_lut = ssa_file
    else: ssa_lut = read_Aeronet_SSA(ssa_file, year=year)
    if isinstance(pfn_file, xr.DataArray): pfn_lut = pfn_file
    else: pfn_lut = read_Aeronet_PFN(pfn_file, year=year)

    if not isinstance(b_wav, BandSet): b_wav_BS = BandSet(b_wav)
    else : b_wav_BS = b_wav
    b_wav_unique = np.unique(b_wav_BS)
    if (pfwav is None): pf_wav = b_wav_unique
    else: pf_wav = pfwav

    fv_time = 'extrapolate'
    aod_lut = aod_lut.interp(
        {'Day_of_Year(Fraction)': day_year_frac, 'wavelength': b_wav_unique}, 
        method='linear', 
        kwargs={'fill_value': fv_time}
                            ).drop_vars('Day_of_Year(Fraction)')
    ssa_lut = ssa_lut.interp(
        {'Day_of_Year(Fraction)': day_year_frac, 'wavelength': b_wav_unique}, 
        method='linear', 
        kwargs={'fill_value': fv_time}
                            ).drop_vars('Day_of_Year(Fraction)')
    pfn_lut = pfn_lut.interp(
        {'Day_of_Year(Fraction)': day_year_frac, 'wavelength': b_wav_unique}, 
        method='linear', 
        kwargs={'fill_value': fv_time}
                            ).drop_vars('Day_of_Year(Fraction)')


    aod_lut = aod_lut.where(aod_lut >= 0, 0)
    ssa_lut = ssa_lut.where((ssa_lut >= 0) & (ssa_lut <= 1), np.clip(ssa_lut, 0, 1))
    pfn_lut = pfn_lut.where(pfn_lut >= 0, 0)


    pfn_val = pfn_lut.values
    pfn_val = np.stack([pfn_val[:,:]]*4, axis=1)
    pfn_val[:,2:3,:]=0.
    pfn_lut = xr.DataArray(pfn_val, 
                        dims=['wavelength', 'stk', 'theta_atm'],
                        coords={'wavelength': pfn_lut.wavelength,
                                'stk': np.arange(4),
                                'theta_atm': pfn_lut.theta_atm})

    hum = np.array([0.])
    wav = aod_lut.wavelength.values.copy()
    theta = pfn_lut.theta_atm.values.copy()
    aod = aod_lut.values[None,:]
    ssa = ssa_lut.values[None,:]
    phase = pfn_lut.values[None,:,:,:]

    aer = AerUser(aod, ssa, phase, hum, wav, theta, 
                  H_mix_min=H_mix_min, H_mix_max=H_mix_max, Z_mix=Z_mix)
    pro = AtmAFGL(atm_name, comp=[aer], grid=grid, P0=P0, O3=O3, H2O=H2O, pfwav=pf_wav, 
                  O3_H2O_alt=O3_H2O_alt).calc(b_wav_BS)

    return pro


def artdeco_to_smartg_cld(input_path, output_path=None, h5_group=None, normalize=True, overwrite=False, veff = None, wl_max = 4500):
    """
    Convert ARTDECO cloud HDF5 file to SMART-G NetCDF file format.

    Reads cloud optical properties from an ARTDECO HDF5 file and converts them 
    to MLUT format.

    Parameters
    ----------
    input_path : str | Path
        Path to the ARTDECO cloud HDF5 file.
    output_path : str | Path, optional
        Output path for saving the converted SMART-G cloud NetCDF file.
        If None, the converted data is not saved to disk. Default: None
    h5_group : str, optional
        Group name within the HDF5 file to open. If None and the file contains
        only one group, that group is automatically selected. If the file contains
        multiple groups, a group name must be specified. Default: None
    normalize : bool, optional
        If True (default), normalize the p11 phase matrix component integral to 2.
        Default: True
    overwrite : bool, optional
        If True and output_path is given, overwrite existing file. Default: False
    veff : float, optional
        Effective volume fraction. Required if cloud properties are dependent on veff.
        Default: None
    wl_max : float, optional
        Maximum wavelength in nanometers. Only wavelengths <= wl_max are included.
        Default: 4500

    Returns
    -------
    m : MLUT
        Multi-dimensional lookup table (MLUT) object containing cloud optical properties.
        Includes axes:
        
        - reff: effective radius
        - wav: wavelength (nm)
        - stk: Stokes components (4 or 6 terms)
        - theta: scattering angle (degrees)
        
        And datasets:
        
        - phase: phase matrix (normalized to 2 if normalize=True)
        - ext: extinction coefficient (km⁻¹)
        - ssa: single scattering albedo
    """
    import netCDF4  # noqa: F401 - must be imported before h5py to avoid HDF5 library conflicts
    import h5py

    # Deals with the case where h5_group is not provided 
    if h5_group is None:
        with h5py.File(input_path, "r") as f:
            keys = list(f.keys())
        if len(keys) == 1 :
            h5_group = keys[0]
        elif len(keys) > 1:
            raise NameError("The h5 file has more than one group. Please choose one group between: " + ', '.join(keys))

    art_cld = read_mlut_hdf5(input_path, group=h5_group)

    # If p22 doesn't exist --> convention with 4 stk components
    # Care, phase_comp elements are sorted in a specific way
    try:    
        art_cld["p22_phase_function"]
        nstk = int(6)
        phase_comp = ['p11_phase_function', 'p21_phase_function', 'p33_phase_function',
                    'p34_phase_function', 'p22_phase_function', 'p44_phase_function']
    except:
        nstk = int(4)
        # here p33 = p44
        phase_comp = ['p11_phase_function', 'p21_phase_function', 'p44_phase_function', 'p34_phase_function']

    # check if the cloud properties are dependant of veff
    is_veff = False
    try:
        art_cld.axes["veff"]
        is_veff = True
    except:
        pass

    if is_veff and veff is None:
        veff_min = str(np.min(art_cld.axes["veff"]))
        veff_max = str(np.max(art_cld.axes["veff"]))
        raise NameError ("The cloud file is dependant of veff. Please give a veff value between: " + veff_min + " and " + veff_max)
    elif is_veff and veff is not None:   
        art_cld = art_cld.sub({'veff':Idx(veff)})

    m = MLUT()
    reff = np.array(art_cld.axes['reff'], dtype=np.float32)
    m.add_axis('reff', reff)
    nreff = len(m.axes['reff'])

    wav = np.array(np.round(art_cld.axes['wavelengths']*1e3, decimals=3), dtype=np.float32)
    m.add_axis('wav', wav[wav<=wl_max])
    nwav = len(m.axes['wav'])

    stk = np.arange(nstk, dtype=np.int16)
    m.add_axis('stk', stk)

    theta = np.array(np.rad2deg(np.arccos(np.float64(art_cld.axes['mu']))), dtype=np.float64)
    m.add_axis('theta', np.sort(theta))
    ntheta = len(m.axes['theta'])

    phase = np.zeros((nreff, nwav, nstk, ntheta), dtype=np.float32)
    for ipc, pc in enumerate(phase_comp):
        # reorder axes, from 'mu', 'reff', 'wavelengths' to 'reff', 'wavelengths', 'mu'
        phac = art_cld[pc].swapaxes('mu', 'reff').swapaxes('mu', 'wavelengths')

        # only take wavelengths less than wl_max
        phac = phac[:,np.arange(nwav),:]

        # sort theta (since mu = cos(theta) may be sorted differently)
        phac = phac[:,:, np.argsort(theta)]
        phase[:,:,ipc,:] = phac.data

    if nstk == 6 : pha_desc = 'phase matrix integral normalized to 2. stk order: p11, p21, p33, p34, p22 and p44'
    if nstk == 4 : pha_desc = 'phase matrix integral normalized to 2. stk order: p11, p21, p33 and p34'

    # integral of P11 must be equal to 2
    if normalize:
        for iwav in range (0, nwav):
            for ireff in range (0, nreff):  
                f = phase[ireff, iwav, 0, :] # P11
                Norm = np.trapezoid(f,-art_cld.axes['mu'][::-1])
                phase[ireff, iwav, :, :] *= 2./abs(Norm)

    m.add_dataset('phase', phase, axnames=['reff', 'wav', 'stk', 'theta'], attrs={'description':pha_desc})

    ext = np.array(art_cld["Cext"][:,np.arange(nwav)], np.float64)
    m.add_dataset('ext', ext, axnames=['reff', 'wav'], attrs={'description':'extinction coefficient in km^-1'})

    ssa = np.array(art_cld["single_scattering_albedo"][:,np.arange(nwav)], np.float64)
    m.add_dataset('ssa', ssa, axnames=['reff', 'wav'], attrs={'description':'single scattering albedo'})

    if veff is not None: m.set_attr('veff', veff)

    if output_path is not None: m.save(output_path, overwrite=overwrite)

    return m

    
def extract_split(m):
    """
    Use a SMART-G run results' MLUT object to compute atmospheric optical 
    properties at specified wavelengths and separates them into decomposed 
    components (absorption, Rayleigh scattering, aerosols, and phase functions).
    These returned profiles can be used as alternative inputs to initialize a 
    new AtmAFGL instance.

    Parameters
    ----------
    m : MLUT
        An MLUT object containing results of a SMART-G run. 
        Must include the following datasets:
        
        - OD_p: particulate optical depth
        - OD_r: Rayleigh optical depth
        - OD_g: gaseous optical depth
        - ssa_p_atm: single scattering albedo of particles
        - iphase_atm: phase function indices
        - phase_atm: phase matrix function

    Returns
    -------
    prof_abs : ndarray
        Gaseous absorption optical depth profile.
    prof_ray : ndarray
        Rayleigh optical depth profile.
    prof_aer : tuple of (ndarray, ndarray)
        Tuple containing:
        
        - prof_aer[0]: Aerosol optical depth profile
        - prof_aer[1]: Single scattering albedo profile of aerosols
    prof_phase : tuple of (ndarray, list)
        Tuple containing:
        
        - prof_phase[0]: Phase function indices (iphase_atm) 
        - prof_phase[1]: List of phase matrix functions for each index

    Examples
    --------
    >>> from smartg.atmosphere import extract_split, AtmAFGL
    >>> prof_abs, prof_ray, prof_aer, prof_phases = extract_split(mlut_result)
    >>> new_atm = AtmAFGL('afglt', prof_abs=prof_abs, prof_ray=prof_ray, 
    ...     prof_aer=prof_aer, prof_phases=prof_phases)
    """
    pro_aer = diff1(m['OD_p'].data.astype(np.float32), axis=1)
    ssa_aer = m['ssa_p_atm'].data
    pro_ray = diff1(m['OD_r'].data.astype(np.float32), axis=1)
    pro_abs = diff1(m['OD_g'].data.astype(np.float32), axis=1)
    pro_iphase = m['iphase_atm'].data
    pro_phases = [m['phase_atm'].sub({'iphase':i}) for i in range(pro_iphase.max()+1)]

    return pro_abs, pro_ray, (pro_aer, ssa_aer), (pro_iphase, pro_phases)


def pha2Iparperconv(pha):
    """
    Convert phase to I parallel/perpendicular convention

    Parameters
    ----------
    pha : 2-D ndarray | 4-D ndarray
        The phase matrix to be converted. In 2-D, stk in in dim 0, and in 4-D in dim3.
    
    Returns
    -------
    out : 2-D ndarray | 4-D ndarray
        The phase matrix converted.
    """

    ndim = len(pha.shape)
    if (ndim != 2 and ndim != 4):
        raise ValueError("The phase matrix dimension must be 2 or 4!")
    
    if ndim == 2:
        nstk = pha.shape[0]
        nth = pha.shape[1]
    else :
        nstk = pha.shape[2]
        nth = pha.shape[3]
    
    if (nstk != 4 and nstk != 6):
        raise ValueError("The number of phase matrix terms must be equal to 4 or 6!")


    if ndim == 2:
        pha_converted = np.zeros((6,nth), dtype=np.float64)
        if (nstk == 4): # spherical particles
            pha_converted[0:4,:] = pha.copy()
            pha_converted[4,:] = pha[0,:].copy()
            pha_converted[5,:] = pha[2,:].copy()
            p0 = pha_converted[0,:].copy()
            p1 = pha_converted[1,:].copy()
            p4 = pha_converted[4,:].copy()
            pha_converted[0,:] = 0.5*(p0+2*p1+p4) # P11
            pha_converted[1,:] = 0.5*(p0-p4)      # P12=P21
            pha_converted[4,:] = 0.5*(p0-2*p1+p4) # P22
        elif (nstk == 6): # non spherical particles
            pha_converted[:,:] = pha.copy()
            p0 = pha_converted[0,:].copy()
            p1 = pha_converted[1,:].copy()
            p4 = pha_converted[4,:].copy()
            pha_converted[0,:] = 0.5*(p0+2*p1+p4) # P11
            pha_converted[1,:] = 0.5*(p0-p4)      # P12=P21
            pha_converted[4,:] = 0.5*(p0-2*p1+p4) # P22
    else: # ndim = 4
        pha_converted = np.zeros((pha.shape[0],pha.shape[1],6,nth), dtype=np.float64)
        if (nstk == 4): # spherical particles
            pha_converted[:,:,0:4,:] = pha.copy()
            pha_converted[:,:,4,:] = pha[:,:,0,:].copy()
            pha_converted[:,:,5,:] = pha[:,:,2,:].copy()
            p0 = pha_converted[:,:,0,:].copy()
            p1 = pha_converted[:,:,1,:].copy()
            p4 = pha_converted[:,:,4,:].copy()
            pha_converted[:,:,0,:] = 0.5*(p0+2*p1+p4) # P11
            pha_converted[:,:,1,:] = 0.5*(p0-p4)      # P12=P21
            pha_converted[:,:,4,:] = 0.5*(p0-2*p1+p4) # P22
        elif (nstk == 6): # non spherical particles
            pha_converted[:,:,:,:] = pha.copy()
            p0 = pha_converted[:,:,0,:].copy()
            p1 = pha_converted[:,:,1,:].copy()
            p4 = pha_converted[:,:,4,:].copy()
            pha_converted[:,:,0,:] = 0.5*(p0+2*p1+p4) # P11
            pha_converted[:,:,1,:] = 0.5*(p0-p4)      # P12=P21
            pha_converted[:,:,4,:] = 0.5*(p0-2*p1+p4) # P22
    return pha_converted


def str2grid_array(str_grid):
    """
    Convert altitude grid specification string to numpy array.

    This function adopts py4cats' compact grid specification format, providing
    py4cats users with familiar syntax for altitude grid definition in SMARTG.

    Parameters
    ----------
    str_grid : str
        Compact grid specification string describing a piecewise-linear altitude
        grid. Format: 'start[step1]stop1[step2]stop2[step3]stop3...'
        
        Each segment is defined by:
        - start: starting altitude value (float, int, or scientific notation)
        - [step]: step size enclosed in square brackets
        - stop: ending altitude value (float, int, or scientific notation)
        
        Supports both positive and negative steps. Results are always monotonic
        across all segments.

    Returns
    -------
    ndarray
        1D array of altitude values. The array is sorted and contains the
        generated grid points covering all specified segments.

    Notes
    -----
    - Each segment creates a uniformly spaced array using numpy.linspace
    - The final endpoint is always included in the output
    - Intermediate endpoints between segments are included with their exact value
    - Step sizes can be positive or negative
    - Supports scientific notation (e.g., 1e-3, 2.5E+2)
    

    Examples
    --------
    Simple grid from TOA to ground (100 km to 0 km with step 1 km):
    
    >>> grid = str2grid_array('100[1]0')
    >>> grid
    array([100.,  99.,  98., ...,   2.,   1.,   0.])
    >>> len(grid)
    101

    Multi-segment grid with varying resolution (TOA to ground):
    
    >>> grid = str2grid_array('500[10]100[1]0')
    >>> grid[:5]
    array([500., 490., 480., 470., 460.])
    >>> grid[40:43]
    array([100.,  99.,  98.])
    
    Grid with scientific notation:
    
    >>> grid = str2grid_array('1[1e-1]1e-1[1e-2]0')
    >>> grid
    array([1.  , 0.9 , 0.8 , 0.7 , 0.6 , 0.5 , 0.4 , 0.3 , 0.2 , 0.1 ,
           0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.  ])
    >>> len(grid)
    20
    """
    import re

    # Split by bracketed steps to extract numbers and steps separately
    # re.split with capturing group keeps the steps
    # Result: [start, step1, stop1, step2, stop2, ...]
    parts = re.split(r'\[([\d.eE+-]+)\]', str_grid)

    if len(parts) < 3 or len(parts) % 2 == 0:
        raise ValueError(f'Cannot parse grid specification: "{str_grid}"\n'
                         'Expected format: start[step]stop[step]stop...\n'
                         'Example: "0[1]100[10]500"')

    # Extract numbers (at even indices) and steps (at odd indices)
    numbers_str = [parts[i] for i in range(0, len(parts), 2)]
    steps_str = [parts[i] for i in range(1, len(parts), 2)]

    # Validate we have sensible input
    if not numbers_str or not steps_str:
        raise ValueError(f'Cannot parse grid specification: "{str_grid}"\n'
                         'Expected format: start[step]stop[step]stop...\n'
                         'Example: "0[1]100[10]500"')

    # Convert to floats
    try:
        numbers = [float(x) for x in numbers_str]
        steps = [float(x) for x in steps_str]
    except ValueError as e:
        raise ValueError(f'Invalid numeric value in grid specification: {e}')

    # Validate steps are non-zero
    if any(step == 0 for step in steps):
        raise ValueError('Step size cannot be zero')

    # Convert to numpy arrays for vectorized operations
    numbers = np.asarray(numbers)
    steps = np.asarray(steps)

    # Vectorized calculation of points per segment
    n_array = np.round(np.abs(np.diff(numbers) / steps)).astype(int)
    n_array[-1] += 1  # Ensure final endpoint is included

    # Build piecewise linear grid with list comprehension
    return np.concatenate([np.linspace(numbers[i], numbers[i + 1], n, endpoint=(i == len(steps) - 1))
                           for i, n in enumerate(n_array)])
