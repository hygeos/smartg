#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from smartg.rrs import l2d_inv
from smartg.vrs import V2d_inv
from luts.luts import LUT
import scipy.constants as cst
from scipy.interpolate import interp1d


class BandSet(object):
    def __init__(self, wav):
        """Initialize a BandSet from wavelength band definitions.

        Common object for formatting input band definitions. Accepts a
        scalar float, a 1-D array of wavelengths, or a list of KDIS or
        RepTran ``IBandS`` objects (detected automatically via the
        ``calc_profile`` attribute of the first element).

        The wavelengths are stored internally as a ``float32`` NumPy
        array. When the input is a scalar, it is reshaped to a 1-element
        array and ``scalar`` is set to ``True``.

        Parameters
        ----------
        wav : float, list, ndarray, or list of IBandS
            Wavelength band definition. A float or 1-D array of
            wavelengths (in nm), or a list of KDIS/RepTran ``IBandS``
            objects whose ``w`` attribute gives the band wavelength.

        Notes
        -----
        When ``wav`` is a list of ``IBandS`` objects,
        ``use_reptran_kdis`` is set to ``True``, ``data`` holds the
        original objects, and ``calc_profile`` delegates to each band's
        own ``calc_profile`` method. Otherwise ``data`` is ``None`` and
        ``calc_profile`` returns zeros.
        """
        try:
            self.use_reptran_kdis = hasattr(wav[0], "calc_profile")
        except:
            self.use_reptran_kdis = False

        self.type_wav = False
        if self.use_reptran_kdis:
            self.wav = [x.w for x in wav]
            self.data = wav
            self.type_wav = type(wav[0])
        else:
            self.wav = wav
            self.data = None
            self.type_wav = None

        assert isinstance(self.wav, (float, list, np.ndarray))
        self.wav = np.array(self.wav, dtype="float32")
        self.scalar = self.wav.ndim == 0
        if self.scalar:
            self.wav = self.wav.reshape(1)
        self.size = self.wav.size

    def __getitem__(self, key):
        """Return the wavelength(s) at the given index or slice.

        Parameters
        ----------
        key : int, slice, or array-like
            Index, slice, or fancy index into the internal wavelength
            array.

        Returns
        -------
        ndarray or float
            The wavelength value(s) selected from the internal
            ``float32`` wavelength array.
        """
        return self.wav[key]

    def __len__(self):
        """Return the number of bands in the set.

        Returns
        -------
        int
            Number of wavelength bands (``self.size``).
        """
        return self.size

    def calc_profile(self, prof):
        """Compute the molecular absorption profile for each band.

        For each band, calculate the absorption optical depth profile
        over the altitude grid of ``prof``. When the BandSet was built
        from KDIS/RepTran ``IBandS`` objects
        (``use_reptran_kdis is True``), each band's own
        ``calc_profile`` method is invoked. Otherwise an array of zeros
        is returned (no molecular absorption handled at the band level).

        Parameters
        ----------
        prof : Profile
            Atmospheric profile providing the altitude grid ``prof.z``
            (in km) on which the optical depth is evaluated.

        Returns
        -------
        tau_mol : ndarray
            Molecular absorption optical depth with shape
            ``(size, len(prof.z))`` and ``float32`` dtype.
        """
        tau_mol = np.zeros((self.size, len(prof.z)), dtype="float32")

        if self.use_reptran_kdis:
            for i, w in enumerate(self.data):
                tau_mol[i, :] = w.calc_profile(prof)

        return tau_mol


def spectral_grids(
    lmin, lmax, datas, dl=None, dls=None, Raman="RRS", unit="mW/m2/nm"
):
    """Build spectral grids for absorption and scattering computations.

    Construct the high-resolution wavelength grid used for absorption
    features (``wl``), the low-resolution grid used for scattering
    features (``wls``), and the Raman excitation grid (``wl_RS``). A
    solar irradiance LUT (``Es_LUT``) is built over the union of the
    scattering and Raman-shifted ranges, and precomputed interpolation
    parameters (``iwls_in``, ``wwls_in``) map each high-resolution
    wavelength onto the low-resolution grid by 1-D linear
    interpolation.

    When ``dl`` is ``None``, the high-resolution grid is taken directly
    from the solar spectrum samples falling within ``[lmin, lmax]``;
    otherwise it is a uniform grid from ``lmin`` to ``lmax`` with step
    ``dl``. When ``dls`` is ``None``, the low-resolution grid is
    collapsed to a single point (``NWS = 1``); otherwise it is a
    uniform grid with step ``dls`` whose size is forced to be odd.

    Parameters
    ----------
    lmin : float
        Minimum wavelength of the spectral range (nm).
    lmax : float
        Maximum wavelength of the spectral range (nm).
    datas : ndarray
        Solar spectrum data with shape ``(N, 2)``: column 0 holds the
        wavelengths (nm) and column 1 the irradiance values.
    dl : float, optional
        High spectral resolution step for absorption features (nm).
        If ``None`` (default), the solar spectrum sampling within
        ``[lmin, lmax]`` is used as the high-resolution grid.
    dls : float, optional
        Low spectral resolution step for scattering features (nm).
        If ``None`` (default), the low-resolution grid is reduced to a
        single point (``NWS = 1``).
    Raman : {'RRS', 'VRS'}, optional
        Raman scattering type: ``'RRS'`` (rotational Raman, default)
        uses a 90 deg scattering angle and 243 K temperature;
        ``'VRS'`` (vibrational Raman) uses ``V2d_inv``.
    unit : {'mW/m2/nm', 'photons/cm2/s/nm'}, optional
        Unit of the solar irradiance in ``datas``. If
        ``'photons/cm2/s/nm'``, the values are converted from
        ``mW/m2/nm`` to photon flux. Default is ``'mW/m2/nm'``.

    Returns
    -------
    wl : ndarray
        High-resolution wavelength grid (nm) for absorption features.
    wls : ndarray
        Low-resolution wavelength grid (nm) for scattering features
        (single point if ``dls`` is ``None``).
    wl_RS : ndarray
        Raman excitation wavelength grid (nm) corresponding to ``wl``.
    Es_LUT : LUT
        Solar irradiance look-up table over the union of ``wls`` and
        ``wl_RS`` ranges, indexed by wavelength.
    iwls_in : ndarray of int8
        Index of the lower ``wls`` value used to linearly interpolate
        each ``wl`` onto the low-resolution grid.
    wwls_in : ndarray of float32
        Floating-point weight (in ``[0, 1]``) between ``iwls_in`` and
        ``iwls_in + 1`` for the linear interpolation of ``wl`` in
        ``wls``.
    """
    ## Solar spectrum input data ##
    wl0 = datas[:, 0]
    E0 = datas[:, 1]
    # from mW/m2/nm to photons/cm2/s/nm
    if unit == "photons/cm2/s/nm":
        E0 *= 1e-3 * 1e-4 / (cst.h * cst.c) * (wl0 * 1e-9)

    ## High spectral resolution grid (for absorption features)
    if dl is None:
        # Solar grid
        ii = np.where((wl0 >= lmin) & (wl0 <= lmax))
        wl = wl0[ii]
        NW = wl.size
        # Solar resolution
        dl = (wl[-1] - wl[0]) / wl.size
    else:
        NW = int((lmax - lmin) / dl) + 1
        wl = np.linspace(lmin, lmax, num=NW)  # wavelength grid

    if Raman == "RRS":
        ## RRS excitation wavelength grid for scattering angle of 90 deg and 243°K
        wl_RS, _ = l2d_inv(wl, 90.0, 243.0)
    else:
        ## VRS excitation wavelength grid
        wl_RS, _ = V2d_inv(wl)

    lmin_RS = min(wl_RS.min(), wl.min())
    lmax_RS = max(wl_RS.max(), wl.max())

    # Solar spectrum LUT building
    ii = np.where((wl0 >= lmin_RS) & (wl0 <= lmax_RS))
    Es_LUT = LUT(E0[ii], axes=[wl0[ii]], names=["wavelength"], desc="Es")

    # low spectral resolution for scattering computations # nm
    if dls is not None:
        NWS = int((lmax_RS - lmin_RS) / dls)
        # ensures NWS is odd
        NWS = NWS if (NWS & 1) else NWS + 1
    else:
        NWS = 1
    wls = np.linspace(lmin_RS, lmax_RS, num=NWS)

    # parameters for 1D linear interpolation of wl in wls
    if NWS > 1:
        f = interp1d(wls, np.linspace(0, NWS - 1, num=NWS))
        iw = f(wl)
        iwls_in = np.floor(iw).astype(
            np.int8
        )  # index of lower wls value in the wls array,
        wwls_in = (iw - iwls_in).astype(
            np.float32
        )  # floating proportion between iwls and iwls+1
        # special case for NWS
        ii = np.where(iwls_in == (NWS - 1))
        iwls_in[ii] = NWS - 2
        wwls_in[ii] = 1.0
    else:
        iwls_in = np.array([0], dtype=np.int8)
        wwls_in = np.array([0], dtype=np.float32)

    return wl, wls, wl_RS, Es_LUT, iwls_in, wwls_in
