#!/usr/bin/env python
# -*- coding: utf-8 -*-


import geoclide as gc
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as mp3d
from matplotlib import colors as mcolors
from typing import Literal, Sequence
import re
from itertools import dropwhile
from scipy import interpolate


def receiver_view(
    ds_sg_out: xr.Dataset,
    cat: int | Sequence[int] = 0,
    log_color_scale: bool = False,
    save_path: str | None = None,
    mtoa: float = 1320,
    vmin: float | None = None,
    vmax: float | None = None,
    interpolation: str = 'none',
    flux_unit: Literal['W', 'kW', 'MW'] = 'W',
) -> None:

    """
    Plot receiver irradiance from a SMART-G simulation output.

    The function reads receiver weights from ``ds_sg_out['C_Receiver']``,
    optionally selecting and summing one or more categories,
    converts the cell size from km to m using ``ds_sg_out.attrs['S_Cell']``, normalizes
    by cell area, multiplies by ``mtoa``, applies the selected power ``flux_unit``, and
    displays the 2-D map with :func:`matplotlib.pyplot.imshow`.

    The displayed axes are labeled as relative receiver coordinates (m):
    ``x`` points upward and ``y`` points to the left.

    Parameters
    ----------
    ds_sg_out : xr.Dataset
        SMART-G output Dataset (obtained via ``mlut.to_xarray()``).
    cat : int or sequence of int, default=0
        Receiver category index as defined in [1]_.

        - ``0``: sum of all categories (scalar only).
        - ``1``-``8``: a single specific category.
        - A list / tuple / array of ints in ``1``-``8``: the selected
          categories are summed together. ``0`` is not allowed in this case.
    log_color_scale : bool, optional
        If ``True``, use a logarithmic color normalization.
        Default: False
    save_path : str, optional
        Output filename (without extension). If provided, the figure is saved as
        ``<save_path>.pdf``.
        Default: None
    mtoa : float, optional
        Solar flux at TOA (W/m²). Multiplicative factor applied to the
        receiver weights before display. Typically the TOA solar irradiance for
        physical units, but can be set to any value to rescale monochromatic
        simulation outputs.
        Default: 1320
    vmin : float, optional
        Lower color limit for linear scale. Ignored when
        ``log_color_scale=True``.
        Default: None
    vmax : float, optional
        Upper color limit for linear scale. Ignored when
        ``log_color_scale=True``.
        Default: None
    interpolation : str, optional
        Default: 'none'
    flux_unit : str, optional
        Power unit used for displayed irradiance values. Choices are 'W' (Watt),
        'kW' (kiloWatt), 'MW' (MegaWatt).
        Default: 'W'.


    Returns
    -------
    None
        This function creates a matplotlib figure and colorbar, and optionally
        saves the figure to disk.

    References
    ----------
    .. [1] Moulana, M., Elias, T., Cornet, C., & Ramon, D. (2019).
           First results to evaluate losses and gains in solar radiation
           collected by solar tower plants.
           *SOLARPACES 2018: International Conference on Concentrating Solar
           Power and Chemical Energy Systems*.
           https://doi.org/10.1063/1.5117709
    """

    if np.isscalar(cat):
        m = ds_sg_out['C_Receiver'].isel(Categories=cat).values
    else:
        cat_list = list(cat)
        if 0 in cat_list:
            raise ValueError(
                "Category index 0 (sum of all) is not allowed when specifying "
                "multiple categories. Use individual indices 1-8."
            )
        if any(c < 1 or c > 8 for c in cat_list):
            raise ValueError("Category indices must be in the range 1-8.")
        m = ds_sg_out['C_Receiver'].isel(Categories=cat_list).sum(
            dim='Categories').values
    # Cell size: S_Cell attribute is in km, convert to m
    cell_size = float(ds_sg_out.attrs['S_Cell']) * 1e3
    half_x = (ds_sg_out.dims['X_Cell_Index'] * cell_size) / 2.
    half_y = (ds_sg_out.dims['Y_Cell_Index'] * cell_size) / 2.
    cell_area = cell_size * cell_size

    if flux_unit == "W":
        unit_scale = 1.
        unit_label = "W"
    elif flux_unit == "kW":
        unit_scale = 1e-3
        unit_label = "kW"
    elif flux_unit == "MW":
        unit_scale = 1e-6
        unit_label = "MW"
    else:
        raise NameError('Unknown argument for unit!')

    plt.figure()

    if not log_color_scale:
        im = plt.imshow((unit_scale * m * mtoa) / cell_area,
                        cmap=plt.get_cmap('jet'), interpolation=interpolation,
                        vmin=vmin, vmax=vmax, extent=[half_y, -half_y, -half_x, half_x])
    else:
        log_vmin = 0.00001 if np.amin(m) < 0.00001 else np.amin(m)
        im = plt.imshow((unit_scale * m * mtoa) / cell_area,
                        cmap=plt.get_cmap('jet'),
                        norm=mcolors.LogNorm(vmin=log_vmin * mtoa,
                                            vmax=np.amax(m * mtoa)),
                        interpolation=interpolation,
                        extent=[half_y, -half_y, -half_x, half_x])

    cbar = plt.colorbar()
    cbar.remove()
    cbar = plt.colorbar(im)
    cbar.set_label(r'Irradiance (' + unit_label + r'.m$^{-2}$)', fontsize=12)
    plt.xlabel(r'Position (m) in relative y axis')
    plt.ylabel(r'Position (m) in relative x axis')
    plt.title('Receiver surface')
    if (save_path is not None):
        plt.savefig(save_path + '.pdf')  


def cat_view(
    ds_sg_out: xr.Dataset,
    mtoa: float | np.ndarray = 1320,
    ncl: Literal["68%", "87%", "95%", "99%", "99.99%"] = "68%",
    output_unit: Literal["FLUX", "FLUX_DENSITY", "RADIANCE"] = "FLUX_DENSITY",
    flux_unit: Literal["uW", "mW", "W", "kW", "MW"] = "W",
    length_unit: Literal["mm", "cm", "dm", "m", "km"] = "m",
    print_results: bool = True,
    accuracy: int = 6,
    kdis_rep_bands: object | None = None,
) -> xr.Dataset:
    """
    Normalize photon weights from a SMART-G simulation output to flux, flux
    density, or radiance with error estimates.

    Processes receiver weights from ``ds_sg_out['wPhCats']`` and
    ``ds_sg_out['wPhCats2']``, applies the specified ``output_unit``, multiplies
    by ``mtoa``, applies the selected ``flux_unit``, and returns a new Dataset
    with normalized intensity and error estimates for all 8 receiver categories.

    Parameters
    ----------
    ds_sg_out : xr.Dataset
        SMART-G output Dataset containing receiver photon weights and metadata.
    mtoa : float | 1-D ndarray, optional
        Solar flux at TOA (W/m²). If there is a wavelength dimension, provide
        an np.array with the flux as a function of wavelength.
        Default: 1320
    ncl : str, optional
        Nominal Confidence Limit for the error estimation.
        Default: "68%"
    output_unit : str, optional
        Output unit type. Choices are:
        - 'FLUX' (Watt)
        - 'FLUX_DENSITY' (Watt/meter²)
        - 'RADIANCE' (Watt/meter²/sr)
        Default: "FLUX_DENSITY"
    flux_unit : str, optional
        Power unit used for displayed irradiance values. Choices are 'W' (Watt),
        'kW' (kiloWatt), 'MW' (MegaWatt).
        Default: 'W'.
    length_unit : str, optional
        Length unit for display. Choices are "cm" (centimeter), "m" (meter),
        "km" (kilometer), etc.
        Default: "m"
    print_results : bool, optional
        If True, print results. If there is a wavelength dimension, prints
        the spectrally integrated results.
        Default: True
    accuracy : int, optional
        Accuracy: number of decimal points to display when printing.
        Default: 6
    kdis_rep_bands : KDIS_IBAND_LIST | REPTRAN_IBAND_LIST, optional
        Band information object. Used for spectral processing.
        Default: None

    Returns
    -------
    output : xr.Dataset
        Dataset containing intensity (flux, flux density, or radiance) with
        associated error estimates for each category.
    """

    m = ds_sg_out

    # Initialize the output Dataset
    output = xr.Dataset()

    # Add the Categories dimension (See Moulana et al. 2019 for 8 Categories)
    categories = np.arange(9, dtype=np.float64)
    output = output.assign_coords(Categories=categories)

    # Parameters not dependant on the wavelength
    aldeg = float(m.attrs['ALDEG'])

    # Parameters needed in case kdis or reptran is used
    if kdis_rep_bands is not None:
        _, _, _, _, norm, norm_dl = kdis_rep_bands.get_weights(output_type='DataArray')

    # Check if there is a dimension wavelength
    is_wave_axis = 'wavelength' in m['wPhCats'].dims

    # Fill needed parameters considering the case with and without the wl
    # dimension
    if is_wave_axis:
        nph = m['norm_npho'].values
        nph_int = float(m.attrs['NPHOTONS'])
    else:
        nph = float(m.attrs['NPHOTONS'])

    # LUT with sum of photon weight (and squared weight) in function of
    # Categories and (if there is wl dim) wavelength
    mf = m['wPhCats']
    mf2 = m['wPhCats2']

    # The desired unit of measurement between Watt, kiloWatt, MegaWatt...
    if flux_unit == "uW":
        k = 1e6
        flux_unit_long = "microWatt"
    elif flux_unit == "mW":
        k = 1e3
        flux_unit_long = "milliWatt"
    elif flux_unit == "W":
        k = 1.0
        flux_unit_long = "Watt"
    elif flux_unit == "kW":
        k = 1e-3
        flux_unit_long = "kiloWatt"
    elif flux_unit == "MW":
        k = 1e-6
        flux_unit_long = "MegaWatt"
    else:
        raise NameError('Unknown argument for flux_unit!')

    # The desired unit of measurement of length (centimeter, meter, ...)
    if length_unit == "mm":
        kl = 1e-3 * 1e-3
        length_unit_long = "millimeter"
    elif length_unit == "cm":
        kl = 1e-2 * 1e-2
        length_unit_long = "centimeter"
    elif length_unit == "dm":
        kl = 1e-1 * 1e-1
        length_unit_long = "decimeter"
    elif length_unit == "m":
        kl = 1.0
        length_unit_long = "meter"
    elif length_unit == "km":
        kl = 1e3 * 1e3
        length_unit_long = "kilometer"
    else:
        raise NameError('Unknown argument for length_unit!')

    if output_unit == "FLUX":
        cst = 1.0 * k
        str_print = f"Flux in {flux_unit_long} for each categories"
        str_type = "flux"
    elif output_unit == "FLUX_DENSITY":
        cst = (1.0 * k * kl) / (float(m.attrs['S_Receiver']) * 1e6)
        str_print = (f"Irradiance in {flux_unit_long}/"
                     f"{length_unit_long}² for each categories")
        str_type = "irradiance"
    elif output_unit == "RADIANCE":
        cst = (1.0 * k * kl) / (float(m.attrs['S_Receiver']) * 1e6)
        cst *= 2.0 / (np.pi * (1 - np.cos(np.radians(2 * aldeg))))
        str_print = (f"Radiance in {flux_unit_long}/"
                     f"{length_unit_long}²/sr for each categories")
        str_type = "radiance"
    else:
        raise NameError('Unknown argument for output_unit!')

    if is_wave_axis:
        cst *= float(m.attrs['n_cte'])
        cst *= np.sum(nph) / nph
    else:
        cst *= float(m.attrs['n_cte'])

    # Normalized intensity
    if is_wave_axis:
        if kdis_rep_bands is not None:
            # Group wavelengths by band structure and sum within each band
            mf_n = (mf * cst * mtoa).groupby('wavelength').sum(dim='wavelength')
            mf_n_int = mf_n / norm

            mf_2_n_int = (mf2 * (cst * mtoa) * (cst * mtoa)).groupby(
                'wavelength').sum(dim='wavelength')
            mf_2_n_int /= norm

            # Convert to DataArray with proper coordinates
            mf_n_int = xr.DataArray(
                mf_n_int.values,
                dims=["Categories", "wavelength"],
                coords={
                    "Categories": np.arange(9, dtype=np.float64),
                    "wavelength": mf_n_int.wavelength,
                },
            )
            mf_n /= norm_dl
        else:
            mf_n = mf * cst * mtoa
            mf_2_n = mf2 * (cst * mtoa) * (cst * mtoa)

        # For non-grouped case, wrap as DataArray if needed
        if not isinstance(mf_n, xr.DataArray):
            mf_n = xr.DataArray(
                mf_n.values if hasattr(mf_n, 'values') else mf_n,
                dims=["Categories", "wavelength"],
                coords={
                    "Categories": np.arange(9, dtype=np.float64),
                    "wavelength": m.wavelength,
                },
            )

        # Add the wavelength dimension in the output Dataset
        if "wavelength" not in output.coords:
            output = output.assign_coords(
                wavelength=mf_n.wavelength
            )
    else:
        mf_n = mf * cst * mtoa

    # Nominal confidence limit factor for error calculation
    if ncl == "68%":
        ld = 1
    elif ncl == "87%":
        ld = 1.5
    elif ncl == "95%":
        ld = 2
    elif ncl == "99%":
        ld = 3
    elif ncl == "99.99%":
        ld = 4

    # Absolute error calculation and normalization
    if is_wave_axis:
        s_wl = len(m.wavelength)
        abs_err = np.zeros((9, s_wl), dtype="float64")
        sum_2_z = np.zeros((9, s_wl), dtype="float64")
        sum_z_2 = np.zeros((9, s_wl), dtype="float64")

        n_bis = nph / (nph - 1)

        sum_2_z[:, :] = (mf.values[:, :] * mf.values[:, :]) / nph
        sum_z_2 = mf2.values[:, :]
        abs_err[:, :] = (n_bis * np.abs(sum_z_2 - sum_2_z)) ** 0.5
        abs_err_lut = xr.DataArray(
            abs_err[:, :],
            dims=["Categories", "wavelength"],
            coords={
                "Categories": np.arange(9, dtype=np.float64),
                "wavelength": m.wavelength,
            },
        )
        if kdis_rep_bands is not None:
            # Group by bands and sum within each band
            abs_err_lut_n = (abs_err_lut * cst * mtoa * ld).groupby(
                'wavelength').sum(dim='wavelength')
            abs_err_lut_n /= norm_dl
        else:
            abs_err_lut_n = abs_err_lut.values[:, :] * cst * mtoa * ld
        abs_err_lut_n = xr.DataArray(
            abs_err_lut_n if isinstance(abs_err_lut_n, np.ndarray)
            else abs_err_lut_n.values,
            dims=["Categories", "wavelength"],
            coords={
                "Categories": np.arange(9, dtype=np.float64),
                "wavelength": (
                    abs_err_lut_n.wavelength
                    if hasattr(abs_err_lut_n, "wavelength")
                    else m.wavelength
                ),
            },
        )

        abs_err_int = np.zeros(9, dtype="float64")
        sum_2_z_int = np.zeros(9, dtype="float64")
        sum_z_2_int = np.zeros(9, dtype="float64")

        n_bis_int = nph_int / (nph_int - 1)

        if kdis_rep_bands is not None:
            mf_int = np.sum(mf_n_int.values[:, :], axis=1)
            mf_2_int = np.sum(mf_2_n_int.values[:, :], axis=1)
        else:
            mf_int = np.sum(mf_n.values[:, :], axis=1)
            mf_2_int = np.sum(mf_2_n.values[:, :], axis=1)

        sum_2_z_int[:] = (mf_int[:] * mf_int[:]) / nph_int
        sum_z_2_int = mf_2_int[:]
        abs_err_int[:] = (n_bis_int * np.abs(sum_z_2_int - sum_2_z_int)) ** 0.5
        abs_err_lut_int = xr.DataArray(
            abs_err_int[:],
            dims=["Categories"],
            coords={"Categories": np.arange(9, dtype=np.float64)},
        )
        abs_err_lut_n_int = abs_err_lut_int

    else:
        abs_err = np.zeros(9, dtype="float64")
        sum_2_z = np.zeros(9, dtype="float64")
        sum_z_2 = np.zeros(9, dtype="float64")

        n_bis = nph / (nph - 1)

        sum_2_z[:] = (mf.values[:] * mf.values[:]) / nph
        sum_z_2 = mf2.values[:]
        abs_err[:] = (n_bis * np.abs(sum_z_2 - sum_2_z)) ** 0.5
        abs_err_lut = xr.DataArray(
            abs_err[:],
            dims=["Categories"],
            coords={"Categories": np.arange(9, dtype=np.float64)},
        )
        abs_err_lut_n = abs_err_lut * cst * mtoa * ld
    # Relative error calculation
    rel_err_lut_n = (abs_err_lut_n / mf_n) * 100
    
    # Create DataArray for the number of photons in function of Categories
    nb_ph_lut = xr.DataArray(
        m['cat_PhNb'].values,
        dims=["Categories"],
        coords={"Categories": np.arange(9, dtype=np.float64)},
    )
    
    # Add descriptions and DataArrays to output Dataset
    mf_n.attrs['description'] = str_print
    nb_ph_lut.attrs['description'] = (
        "Number of photons in function of Categories"
    )
    abs_err_lut_n.attrs['description'] = f'Absolute error of {output_unit}'
    rel_err_lut_n.attrs['description'] = (
        f'Relative error in percentage of {output_unit}'
    )

    output[output_unit] = mf_n
    output['NbPhotons'] = nb_ph_lut
    output['AbsoluteErr'] = abs_err_lut_n
    output['RelativeErr'] = rel_err_lut_n

    if kdis_rep_bands is not None:
        output[output_unit + "_int"] = mf_n_int
        mf_n_tot = xr.DataArray(
            np.sum(mf_n_int.values[:, :], axis=1),
            dims=["Categories"],
            coords={"Categories": np.arange(9, dtype=np.float64)},
        )
        output[output_unit + "_tot"] = mf_n_tot
        output["AbsoluteErr_tot"] = abs_err_lut_int

    # Print results if requested
    if print_results:
        l_p = ["(  D  )", "(  H  )", "(  E  )", "(  A  )",
               "( H+A )", "( H+E )", "( E+A )", "(H+E+A)"]
        int_acc = int(accuracy)
        str_acc = str(int_acc)
        str_acc = "%." + str_acc + "f"

        mat = np.zeros((9, 4), dtype="float64")
        if is_wave_axis:
            if kdis_rep_bands is not None:
                mat[:, 0] = np.sum(mf_n_int.values[:, :], axis=1)
            else:
                mat[:, 0] = np.sum(mf_n.values[:, :], axis=1)
            mat[:, 1] = m['cat_PhNb'].values
            mat[:, 2] = abs_err_lut_n_int.values
            mat[:, 3] = (mat[:, 2] / mat[:, 0]) * 100
        else:
            mat[:, 0] = mf_n.values
            mat[:, 1] = m['cat_PhNb'].values
            mat[:, 2] = abs_err_lut_n.values
            mat[:, 3] = rel_err_lut_n.values
            
        print("**********************************************************")
        print(str_print)
        print("**********************************************************")
        print("SUM_CATS      " + ": " + str_type + "=",
              str_acc % (mat[0,0]), " number_ph=",
              np.uint64(mat[0,1]), " errAbs=",
              str_acc % (mat[0,2]), " err(%)=",
              str_acc % (mat[0,3]*ld))
        for i in range (0, 8):
            print("CAT", i+1, l_p[i], ": " + str_type + "=",
                  str_acc % (mat[i+1,0]), " number_ph=",
                  np.uint64(mat[i+1,1]), " errAbs=",
                  str_acc % (mat[i+1,2]), " err(%)=",
                  str_acc % (mat[i+1,3]*ld))
    return output


def nopt_view(
    ds_sg_out: xr.Dataset,
    back: bool = False,
    acc: int = 6,
    ncl: Literal["68%", "87%", "95%", "99%", "99.99%"] = "68%",
    mtoa: None | np.ndarray = None,
    natm_approx: bool = False,
) -> None:
    """
    Calculate and display the detailed optical efficiencies with
    associated error estimates of a Solar Tower Power simulated with
    SMART-G.

    Parameters
    ----------
    ds_sg_out : xr.Dataset
        SMART-G output Dataset containing simulation results.
    back : bool, optional
        False for forward mode (default), True for backward mode.
        Determines which efficiency metrics are calculated and displayed.
        Default: False
    acc : int, optional
        Accuracy: number of decimal points to display in the output.
        Default: 6
    ncl : str, optional
        Nominal Confidence Limit for error estimation. Options are:
        - "68%" (1 sigma)
        - "87%" (1.5 sigma)
        - "95%" (2 sigma)
        - "99%" (3 sigma)
        - "99.99%" (4 sigma)
        Default: "68%"
    mtoa : None | 1-D ndarray, optional
        Solar flux at TOA for each wavelength band. If None, uses the
        total power. If provided, weights the calculation by flux
        per band. Default: None
    natm_approx : bool, optional
        If True, calculate and display the analytic approximation of
        atmospheric transmission (natm_approx) in backward mode. Ignored
        in forward mode. Default: False

        
    Notes
    -----
    In forward mode, displays:
    - nopt: Total optical efficiency
    - ncos: Cosine efficiency
    - nsha: Shading efficiency
    - nref: Reflection efficiency
    - nblo: Blocking efficiency
    - nspi: Spillage efficiency
    - natm: Atmospheric transmission

    In backward mode, displays:
    - nopt: Total optical efficiency
    - ncos: Cosine efficiency
    - nref: Reflection efficiency
    - nsbsa: Product of blocking, shading, and atmospheric efficiencies

    Each metric includes an estimate of absolute error and relative error.
    """
    ds = ds_sg_out
    # Number of photons launched
    nph = float(ds.attrs['NPHOTONS'])
    # n/(n-1)
    nbis = nph / (nph - 1)

    if(mtoa is None):
        powc_h = ds['powc_H'].values
    else:
        powc_h = 0.
        for i in range(0, len(mtoa)):
            powc_h += ds['powc_H'].values[i] * mtoa[i]
        powc_h /= np.sum(mtoa)

    k = float(ds.attrs['n_cte']) / powc_h

    int_acc = int(acc)
    str_acc = str(int_acc)
    str_acc = "%." + str_acc + "f"
    if (ncl == "68%"):
        ld = 1
    elif (ncl == "87%"):
        ld = 1.5
    elif (ncl == "95%"):
        ld = 2
    elif (ncl == "99%"):
        ld = 3
    elif (ncl == "99.99%"):
        ld = 4

    print("**********************************************")
    print(" Optical Efficiencies")
    print("**********************************************")

    if(back == False):  # Forward mode ->
        # Sum of weights
        # w0=wI, w1=wrhoM, w2=wrhoP, w3=wBM, w4=wBP, w5=wSM, w6=wSP
        # w7=wREC
        w0 = ds['wLoss'].values[0]
        w1 = ds['wLoss'].values[1]
        w2 = ds['wLoss'].values[2]
        w3 = ds['wLoss'].values[3]
        w4 = ds['wLoss'].values[4]
        w5 = ds['wLoss'].values[5]
        w6 = ds['wLoss'].values[6]
        w7 = ds['cat_w'].values[2]
        # Sum of (weights²)
        w0_2 = ds['wLoss2'].values[0]
        w1_2 = ds['wLoss2'].values[1]
        w2_2 = ds['wLoss2'].values[2]
        w3_2 = ds['wLoss2'].values[3]
        w4_2 = ds['wLoss2'].values[4]
        w5_2 = ds['wLoss2'].values[5]
        w6_2 = ds['wLoss2'].values[6]
        w7_2 = ds['cat_w2'].values[2]
        # (Sum of weights)² divided by the number of photons
        sum_z_bar2 = [
            (w0*w0)/nph, (w1*w1)/nph, (w2*w2)/nph, (w3*w3)/nph,
            (w4*w4)/nph, (w5*w5)/nph, (w6*w6)/nph, (w7*w7)/nph
        ]
        # Sum of (weights²)
        sum_z2_bar = [w0_2, w1_2, w2_2, w3_2, w4_2, w5_2, w6_2, w7_2]
        dw = []
        for i in range(0, len(sum_z_bar2)):
            dw_temp = ld * nbis * (sum_z2_bar[i] - sum_z_bar2[i]) ** 0.5
            dw.append(dw_temp)
        
        nopt = gc.clamp(k*w7, 0, 1)
        k_s = k / float(ds.attrs['n_cos'])
        ncos = float(ds.attrs['n_cos'])
        nsha = gc.clamp(k_s*w0, 0, 1)
        nref = gc.clamp(1-(w1/w0), 0, 1)
        nblo = gc.clamp(1-(w3/w2), 0, 1)
        nspi = gc.clamp(1-(w5/w4), 0, 1)
        natm = gc.clamp(w7/w6, 0, 1)

        d_nopt = abs(k)*dw[7]
        d_ncos = 0.
        d_nsha = abs(k_s)*dw[0]
        d_nref = abs(-1./w0)*dw[1] + abs(w1/w0**2)*dw[0]
        d_nblo = abs(-1./w2)*dw[3] + abs(w3/w2**2)*dw[2]
        d_nspi = abs(-1./w4)*dw[5] + abs(w5/w4**2)*dw[4]
        d_natm = abs(1./w6)*dw[7] + abs(w7/w6**2)*dw[6]

        print("nopt =", str_acc % nopt, ", errAbs =", str_acc % d_nopt,
              ", err% =", str_acc % ((d_nopt/nopt)*100))
        print("ncos =", str_acc % ncos, ", errAbs =", str_acc % d_ncos,
              ", err% =", str_acc % ((d_ncos/ncos)*100))
        print("nsha =", str_acc % nsha, ", errAbs =", str_acc % d_nsha,
              ", err% =", str_acc % ((d_nsha/nsha)*100))
        print("nref =", str_acc % nref, ", errAbs =", str_acc % d_nref,
              ", err% =", str_acc % ((d_nref/nref)*100))
        print("nblo =", str_acc % nblo, ", errAbs =", str_acc % d_nblo,
              ", err% =", str_acc % ((d_nblo/nblo)*100))
        print("nspi =", str_acc % nspi, ", errAbs =", str_acc % d_nspi,
              ", err% =", str_acc % ((d_nspi/nspi)*100))
        print("natm =", str_acc % natm, ", errAbs =", str_acc % d_natm,
              ", err% =", str_acc % ((d_natm/natm)*100))
    else:  # Backward mode ->
        # Sum of weights
        # w0=wI, w1=wrhoM, w2=wREC
        w0 = ds['wLoss'].values[0]
        w1 = ds['wLoss'].values[1]
        w2 = ds['cat_w'].values[2]
        # Sum of (weights²)
        w0_2 = ds['wLoss2'].values[0]
        w1_2 = ds['wLoss2'].values[1]
        w2_2 = ds['cat_w2'].values[2]
        # (Sum of weights)² divided by the number of photons
        sum_z_bar2 = [(w0*w0)/nph, (w1*w1)/nph, (w2*w2)/nph]
        # Sum of (weights²)
        sum_z2_bar = [w0_2, w1_2, w2_2]
        dw = []
        for i in range(0, len(sum_z_bar2)):
            dw_temp = ld * nbis * (sum_z2_bar[i] - sum_z_bar2[i]) ** 0.5
            dw.append(dw_temp)
        nopt = gc.clamp(k*w2, 0, 1)
        ncos = float(ds.attrs['n_cos'])
        nref = gc.clamp(1-(w1/w0), 0, 1)
        nsbsa = gc.clamp((k*w2)/(ncos*nref), 0, 1)

        d_nopt = abs(k)*dw[2]
        d_ncos = 0.
        d_nref = abs(-1./w0)*dw[1] + abs(w1/w0**2)*dw[0]

        d_nsbsa = (abs(k/(ncos*(1-(w1/w0))))*dw[2] +
                   abs((k*w2)/(ncos*w0*(1-(w1/w0))**2))*dw[1] +
                   abs((-k*w2*w1)/(ncos*w0*w0*(1-(w1/w0))**2)))

        print("nopt =", str_acc % nopt, ", errAbs =", str_acc % d_nopt,
              ", err% =", str_acc % ((d_nopt/nopt)*100))
        print("ncos =", str_acc % ncos, ", errAbs =", str_acc % d_ncos,
              ", err% =", str_acc % ((d_ncos/ncos)*100))
        print("nref =", str_acc % nref, ", errAbs =", str_acc % d_nref,
              ", err% =", str_acc % ((d_nref/nref)*100))
        print("nsbsa =", str_acc % nsbsa, ", errAbs =",
              str_acc % d_nsbsa, ", err% =",
              str_acc % ((d_nsbsa/nsbsa)*100))

        if (natm_approx):
            if(mtoa is None):
                naatm = ds['n_aatm'].values
            else:
                naatm = 0.
                for i in range(0, len(mtoa)):
                    naatm += ds['n_aatm'].values[i] * mtoa[i]
                naatm /= np.sum(mtoa)
            print("naatm =", str_acc % naatm,
                  " -> analytic approx of natm")


class Mirror(object):
    """
    Glossy/specular mirror material surface model.

    Represents glossy/specular reflective materials such as pure and highly 
    polished aluminum, silver-backed glass mirrors, and similar surfaces. Uses 
    microfacet theory with configurable roughness distribution models.

    Attributes
    ----------
    reflectivity : float, optional
        Albedo (reflectance) of the object. Must be between 0 and 1.
        Default: 1.0
    roughness : float, optional
        Surface roughness parameter (alpha) according to Walter et al. 2007.
        Characterizes the distribution of microfacet slopes. Default: 0.0
    shadow : bool, optional
        Whether to include shadowing-masking effects from surface roughness.
        Default: False
    nind : float or None, optional
        Relative refractive index (air/material). If None, represents a perfect 
        mirror (nind = infinity). The internal value becomes -1 for perfect mirrors.
        Default: None
    distribution : str, optional
        Microfacet distribution model. Options are:
        - "Beckmann": Beckmann distribution (internally value 1)
        - "GGX": GGX/Trowbridge-Reitz distribution (internally value 2)
        Default: "Beckmann"

    References
    ----------
    Walter, B., Marschner, S. R., Li, H., & Torrance, K. E. (2007).
    Microfacet models for refraction through rough surfaces.
    """
    def __init__(self, reflectivity = 1., roughness = 0., shadow = False, nind = None,
                 distribution = "Beckmann"):
        self.reflectivity = reflectivity
        self.roughness    = roughness
        self.shadow       = shadow
        if nind is None:
            self.nind     = -1
        else:
            self.nind     = nind
        if distribution == "Beckmann":
            self.distribution = 1
        elif distribution == "GGX":
            self.distribution = 2
        else:
            NameError('Please choose a distribution between str(Beckmann) or str(GGX)')

    def __str__(self):
        return 'Material -> Mirror : ' \
            'reflectivity=' + str(self.reflectivity) + ', roughness=' + str(self.roughness) \
            + ', shadow=' + str(self.shadow) + ', nind=' + str(self.nind) \
            + ', distribution=' + str(self.distribution)


class LambMirror(object):
    """
    Lambertian mirror material surface model.

    Represents a Lambertian reflective material with equal probability of reflection 
    in all directions within the hemisphere normal to the object surface

    Parameters
    ----------
    reflectivity : float, optional
        Albedo (reflectance) of the object. Must be between 0 and 1.
        Controls the fraction of incident light that is reflected.
        Default: 0.5
    """
    def __init__(self, reflectivity = 0.5):
        self.reflectivity = reflectivity
        

    def __str__(self):
        return 'Material -> Lambertian Mirror : ' \
            'reflectivity=' + str(self.reflectivity)


class Matte(object):
    """
    Matte material surface model.

    Represents matte materials such as concrete, plastic, dust, 
    and similar surfaces with diffuse reflectance properties.

    Parameters
    ----------
    reflectivity : float, optional
        Albedo (reflectance) of the object. Must be between 0 and 1.
        Default: 0.0
    roughness : float, optional
        Surface roughness parameter.
        Default: 0.0

    Notes
    -----
    Be careful !!

    - For the moment this material is only used for totally absorbant surfaces.
    """
    def __init__(self, reflectivity = 0., roughness = 0.):
        self.reflectivity = reflectivity
        self.roughness = roughness
        
    def __str__(self):
        return 'Material -> Matte : ' \
            'reflectivity=' + str(self.reflectivity) + ', roughness=' + str(self.roughness)


class Plane(object):
    """
    Planar surface defined by four corner points.

    Defines a rectangular plane surface constructed from four corner points.
    The plane must satisfy specific coordinate constraints for each point.

    Parameters
    ----------
    p1 : gc.Point, optional
        Bottom-left corner point (x negative, y negative).
        Default: gc.Point(-0.5, -0.5, 0.)
    p2 : gc.Point, optional
        Bottom-right corner point (x positive, y negative).
        Default: gc.Point(0.5, -0.5, 0.)
    p3 : gc.Point, optional
        Top-left corner point (x negative, y positive).
        Default: gc.Point(-0.5, 0.5, 0.)
    p4 : gc.Point, optional
        Top-right corner point (x positive, y positive).
        Default: gc.Point(0.5, 0.5, 0.)

    Notes
    -----
    The plane geometry requires:
    - p1 and p3 have the same negative x-coordinate
    - p2 and p4 have the same positive x-coordinate
    - p1 and p2 have the same negative y-coordinate
    - p3 and p4 have the same positive y-coordinate
    """
    def __init__(self, p1 = gc.Point(-0.5, -0.5, 0.), p2 = gc.Point(0.5, -0.5, 0.), \
                 p3 = gc.Point(-0.5, 0.5, 0.), p4 = gc.Point(0.5, 0.5, 0.)):
        if (isinstance(p1, gc.Point) and isinstance(p2, gc.Point) and \
            isinstance(p3, gc.Point) and isinstance(p4, gc.Point)):
            if (  ( (p1.x == p3.x) and (p1.x < 0) )  and \
                  ( (p2.x == p4.x) and (p2.x > 0) )  and \
                  ( (p1.y == p2.y) and (p1.y < 0) )  and \
                  ( (p3.y == p4.y) and (p3.y > 0) )   ):
                self.p1 = p1
                self.p2 = p2
                self.p3 = p3
                self.p4 = p4
            elif ( (p1.x >= 0) or (p2.x <= 0) or (p1.y >= 0) or (p3.y >= 0) ):
                raise NameError( 'Those conditions must be filled! : ' + \
                                'p1.x < 0 , p1.y < 0 ,' + \
                                'p2.x > 0 , p2.y < 0 ,' + \
                                'p3.x < 0 , p3.y > 0 ,' + \
                                'p4.x > 0 , p4.y > 0' )
            elif ( (p1.x != p3.x) or (p2.x != p4.x) or (p1.y != p2.y) or (p3.y != p4.y) ):
                raise NameError('Your plane geometry must be at leat a rectangle!')
            else:
                NameError('Unknown error in Plane class!')
        else:
            raise NameError('All arguments must be Point type!')

    def __str__(self):
        return 'Coordinates of the Plane :\n' \
            '-> p1=(' + str(self.p1.x) + ', ' + str(self.p1.y) + ', ' + str(self.p1.z) + ')\n' + \
            '-> p2=(' + str(self.p2.x) + ', ' + str(self.p2.y) + ', ' + str(self.p2.z) + ')\n' + \
            '-> p3=(' + str(self.p3.x) + ', ' + str(self.p3.y) + ', ' + str(self.p3.z) + ')\n' + \
            '-> p4=(' + str(self.p4.x) + ', ' + str(self.p4.y) + ', ' + str(self.p4.z) + ')'

class Spheric(object):
    """
    Spherical surface model.

    Represents a spherical (or partial spherical) surface defined by radius 
    and optional height constraints. Can represent a full sphere or a partial 
    sphere.

    Parameters
    ----------
    radius : float, optional
        Radius of the sphere. Must be positive.
        Default: 10.0
    z0 : float or None, optional
        Minimum height (bottom) of the spherical surface. If None, defaults 
        to -radius (full sphere from bottom). For partial spheres, specify 
        custom z0 value.
        Default: None (becomes -radius)
    z1 : float or None, optional
        Maximum height (top) of the spherical surface. If None, defaults 
        to +radius (full sphere to top). For partial spheres, specify 
        custom z1 value.
        Default: None (becomes +radius)
    phi : float, optional
        Azimuthal angle range in degrees. 360 degrees represents a full 
        sphere; smaller values create a partial spherical sector.
        Default: 360.0

    Notes
    -----
    For a full sphere, use default values: z0 = -radius, z1 = +radius, phi = 360°
    """
    def __init__(self, radius = 10., z0 = None, z1 = None, phi = 360.):
        self.radius = radius
        self.phi = phi
        if (z0 == None):
            self.z0 = -1.*radius
        else:
            self.z0 = z0
        if (z1 == None):
            self.z1 = 1.*radius
        else:
            self.z1 = z1

    def __str__(self):
        return 'Sphere with the following caracteristics :\n' + \
            '-> radius = ' + str(self.radius) + '\n' + \
            '-> z0 = ' + str(self.z0) + '\n' + \
            '-> z1 = ' + str(self.z1) + '\n' + \
            '-> phi = ' + str(self.phi)


class Transformation():
    """
    Apply rotation and translation transformations to objects.

    Enables flexible transformation of objects through rotation and translation 
    operations. Supports multiple rotation order conventions for specifying 
    the sequence of rotations around different axes.

    Parameters
    ----------
    rotation : 1-D ndarray, optional
        An array with 3 elements specifying rotation angles (in degrees) 
        around the x, y, and z axes respectively.
        Default: np.zeros(3, dtype=float) (no rotation)
    translation : 1-D ndarray, optional
        An array with 3 elements specifying translation distances (in kilometers) 
        along the x, y, and z axes respectively.
        Default: np.zeros(3, dtype=float) (no translation)
    rotationOrder : str, optional
        Specifies the order in which rotations are applied. Options are:
        - "XYZ": Rotate around X, then Y, then Z
        - "XZY": Rotate around X, then Z, then Y
        - "YXZ": Rotate around Y, then X, then Z
        - "YZX": Rotate around Y, then Z, then X
        - "ZXY": Rotate around Z, then X, then Y
        - "ZYX": Rotate around Z, then Y, then X
        Default: "XYZ"
    """
    def __init__(self, rotation = np.zeros(3, dtype=float), translation=np.zeros(3, dtype=float), \
                 rotationOrder = "XYZ"):
        self.rotation = rotation
        self.rotx = rotation[0]
        self.roty = rotation[1]
        self.rotz = rotation[2]
        self.rotOrder = rotationOrder
        self.translation = translation
        self.transx = translation[0]
        self.transy = translation[1]
        self.transz = translation[2]

    def __str__(self):
        return 'Transformation : rotation=(' + str(self.rotx) + ', ' + str(self.roty) + ', ' + \
            str(self.rotz) + ') and translation =(' + str(self.transx) + ', ' + \
            str(self.transy) + ', ' + str(self.transz) + ')'
    
class Entity(object):
    """
    3D object representation with geometry and material properties.

    Enables the creation and management of 3D objects with customizable 
    geometry, materials, transformations, and visualization properties. 
    Objects can be either reflectors or receivers. Receivers will have 
    their flux distribution tracked during simulations.

    Parameters
    ----------
    entity : Entity or None, optional
        Existing Entity object to copy. If provided, all properties are 
        copied from the source entity. If None, properties are set 
        individually from other parameters.
        Default: None
    name : str, optional
        Object type. Options are:
        - "reflector": Passive reflecting surface
        - "receiver": Active receiver that tracks flux distribution
        Default: "reflector"
    TC : float, optional
        Cell size for flux distribution calculation (Taille Cellules in km).
        Defines the spatial resolution for flux binning.
        Default: 0.01
    materialAV : Material, optional
        Material for the object's front surface (above-view side).
        Default: Matte()
    materialAR : Material, optional
        Material for the object's back surface (reverse side).
        Default: Matte()
    geo : Geometry, optional
        Geometric shape of the object (e.g., Plane, Spheric).
        Default: Plane()
    transformation : Transformation, optional
        Rotation and translation transformation to apply to the object.
        Default: Transformation() (identity transformation)
    bboxGPmin : None | gc.Point, optional
        Minimum corner of the bounding box (in development).
        Default: None
    bboxGPmax : None | gc.Point, optional
        Maximum corner of the bounding box (in development).
        Default: None
    color : str, optional
        Color for visualization/rendering.
        Default: 'grey'
    alpha_color : float, optional
        Transparency alpha value for visualization (0.0 to 1.0).
        Default: 0.5
    """
    def __init__(self, entity = None, name="reflector", TC = 0.01, materialAV=Matte(), \
                 materialAR=Matte(), geo=Plane(), transformation=Transformation(), \
                 bboxGPmin = None, bboxGPmax = None, color = 'grey', alpha_color = 0.5):
        if isinstance(entity, Entity) :
            self.name = entity.name; self.TC = entity.TC; self.materialAV = entity.materialAV
            self.materialAR = entity.materialAR; self.geo = entity.geo 
            self.transformation = entity.transformation
            #TODO: Compute automatically bboxGPmin and bboxGPmax from geo and transformation
            self.bboxGPmin = entity.bboxGPmin; self.bboxGPmax = entity.bboxGPmax
            self.color = entity.color; self.alpha_color = alpha_color
        else:
            if not isinstance(geo, (Plane, Spheric)):
                raise NameError('For the moment only Plane or a Spheric geo are accepted.')
            
            self.name = name
            self.TC = TC
            self.materialAV = materialAV
            self.materialAR = materialAR
            self.geo = geo
            self.transformation = transformation

            # if bbox pmin and pmax are not provided compute them automatically 
            # based on the geometry and transformation 
            if bboxGPmin is None or bboxGPmax is None:
                box = gc.BBox()
                E_tf = self.get_transformation()
                if isinstance(self.geo, Plane):
                    box = box.union(E_tf(self.geo.p1))
                    box = box.union(E_tf(self.geo.p2))
                    box = box.union(E_tf(self.geo.p3))
                    box = box.union(E_tf(self.geo.p4))
                elif isinstance(self.geo, Spheric):
                    p1 = E_tf(gc.Point(-self.geo.radius, -self.geo.radius, self.geo.z0))
                    p2 = E_tf(gc.Point(self.geo.radius, self.geo.radius, self.geo.z1))
                    box = box.union(p1)
                    box = box.union(p2)
                if bboxGPmin is None: bboxGPmin = box.pmin
                if bboxGPmax is None: bboxGPmax = box.pmax

            self.bboxGPmin = bboxGPmin
            self.bboxGPmax = bboxGPmax
            self.color = color
            self.alpha_color = alpha_color
        self.check = "Entity"

    def __str__(self):
        return 'The entity is a ' + str(self.name) + ' with the following carac:\n' + \
            str(self.materialAV) + '\n' + \
            str(self.geo) + '\n' + \
            str(self.transformation)
    
    def get_transformation(self):
        """
        Compute the combined transformation matrix for the entity.

        Returns
        -------
        out : gc.Transform
            Combined transformation matrix (translation * rotations in specified order).
            The rotation order is determined by the entity's transformation.rotOrder 
            attribute (e.g., "XYZ", "ZYX", etc.).

        Notes
        -----
        The transformation is applied as::

            combined = Translation * Rotation_sequence

        where Rotation_sequence depends on rotOrder:
        - "XYZ": Rx * Ry * Rz
        - "XZY": Rx * Rz * Ry
        - "YXZ": Ry * Rx * Rz
        - "YZX": Ry * Rz * Rx
        - "ZXY": Rz * Rx * Ry
        - "ZYX": Rz * Ry * Rx
        """
        Trans = gc.get_translate_tf(gc.Vector(self.transformation.transx, self.transformation.transy, \
                                              self.transformation.transz))
        Rotx = gc.get_rotateX_tf(self.transformation.rotx)
        Roty = gc.get_rotateY_tf(self.transformation.roty)
        Rotz = gc.get_rotateZ_tf(self.transformation.rotz)

        # total tt of all transform together
        tt = None
        if   (self.transformation.rotOrder == "XYZ"): tt = Trans*Rotx*Roty*Rotz
        elif (self.transformation.rotOrder == "XZY"): tt = Trans*Rotx*Rotz*Roty
        elif (self.transformation.rotOrder == "YXZ"): tt = Trans*Roty*Rotx*Rotz
        elif (self.transformation.rotOrder == "YZX"): tt = Trans*Roty*Rotz*Rotx
        elif (self.transformation.rotOrder == "ZXY"): tt = Trans*Rotz*Rotx*Roty
        elif (self.transformation.rotOrder == "ZYX"): tt = Trans*Rotz*Roty*Rotx
        else: raise NameError('Unknown rotation order')

        return tt

    def set_transformation(self, transformation, recompute_bbox=True):
        """
        Update the entity's transformation and optionally recompute bounding box.

        Parameters
        ----------
        transformation : Transformation
            New transformation object containing rotation angles (rotx, roty, rotz),
            rotation order (rotOrder), and translation components (transx, transy, transz).
        recompute_bbox : bool, optional
            If True (default), recompute the bounding box (bboxGPmin and bboxGPmax)
            based on the new transformation and the entity's geometry.
            If False, keep the existing bounding box values.
            Default: True

        Notes
        -----
        The bounding box is automatically recomputed by transforming all geometry
        points using the new transformation matrix and computing their extent.

        Examples
        --------
        >>> entity = Entity(geo=Plane(...), transformation=Transformation())
        >>> new_tf = Transformation(translation=np.array([1., 2., 3.]))
        >>> entity.set_transformation(new_tf)  # Update position and recompute bbox
        >>> entity.set_transformation(new_tf, recompute_bbox=False)  # Update without bbox update
        """
        self.transformation = transformation

        if recompute_bbox:
            # Recompute bounding box based on new transformation
            box = gc.BBox()
            E_tf = self.get_transformation()
            
            if isinstance(self.geo, Plane):
                box = box.union(E_tf(self.geo.p1))
                box = box.union(E_tf(self.geo.p2))
                box = box.union(E_tf(self.geo.p3))
                box = box.union(E_tf(self.geo.p4))
            elif isinstance(self.geo, Spheric):
                p1 = E_tf(gc.Point(-self.geo.radius, -self.geo.radius, self.geo.z0))
                p2 = E_tf(gc.Point(self.geo.radius, self.geo.radius, self.geo.z1))
                box = box.union(p1)
                box = box.union(p2)
            
            self.bboxGPmin = box.pmin
            self.bboxGPmax = box.pmax


class Heliostat(object):
    """
    Composite heliostat assembly consisting of multiple facets.

    Represents a heliostat composed of multiple individual facets arranged 
    in a grid pattern.

    Parameters
    ----------
    POS : gc.Point, optional
        Heliostat position (center point) stored as a Point class.
        Default: gc.Point(0., 0., 0.)
    SPX : int, optional
        Number of facet divisions in the x direction. Controls how many times 
        the heliostat is split along the x-axis. Must be >= 1 (total facets >= 2).
        Default: 2
    SPY : int, optional
        Number of facet divisions in the y direction. Controls how many times 
        the heliostat is split along the y-axis. Must be >= 1 (total facets >= 2).
        Default: 2
    HSX : float, optional
        Heliostat size in the x direction (meters).
        Default: 0.02
    HSY : float, optional
        Heliostat size in the y direction (meters).
        Default: 0.02
    CURVE_FL : float | None, optional
        Focal length (in km) for curvature. If None, the focal length is computed
        automatically based on the distance to the receiver. A virtual value of 
        infinity means a flat heliostat with no curvature.
        Default: None
    REF : float, optional
        Reflectivity of the heliostat (between 0 and 1). Represents the 
        fraction of incident radiation that is reflected.
        Default: 1.0
    ROUGH : float, optional
        Surface roughness of the heliostat facets.
        Default: 0
    """
    def __init__(self, POS = gc.Point(0., 0., 0.), SPX=int(2), SPY=int(2), HSX=0.02,
                 HSY=0.02, CURVE_FL=None, REF=1., ROUGH=0):
        # Be sure that we split a heliostat by at least 2
        if (SPX*SPY < 2):
            raise Exception("The number of facets must be >= 2!")
        # Be sure that SPX and SPY are integer values
        if not ( isinstance(SPX, int) and isinstance(SPY, int) ):
            raise Exception("SPx and SPy must be integers")
        self.pos = POS
        self.sPx = SPX
        self.sPy = SPY
        self.hSx = HSX
        self.hSy = HSY
        self.curveFL = CURVE_FL
        self.ref = REF
        self.rough = ROUGH

    def __str__(self):
        return "POS=" + str(self.pos) + '; ' + "SPX=" + str(self.sPx) + '; ' + \
                "SPY=" + str(self.sPy) + '; ' + "HSX=" + str(self.hSx) + '; ' + \
                "HSY=" + str(self.hSy)  + '; ' + "CURVE_FL=" + str(self.curveFL) + \
                '; ' + "REF=" + str(self.ref) + '; ' + "ROUGH=" + str(self.rough)


class GroupE(object):
    """Container for grouping multiple Entity objects.

    A GroupE instance represents a collection of Entity objects with a shared
    bounding box. This is useful for managing related geometric objects as a
    single unit, such as a set of heliostats or building components.

    Parameters
    ----------
    LE : list, optional
        List of Entity objects to group. Default is [Entity()].
    BBOX : None | list, optional
        Custom bounding box as [Pmin, Pmax] where Pmin and Pmax are geoclide.Point
        objects. If None (default), bounding box is computed from LE[0].
    """
    def __init__(self, LE=[Entity()], BBOX=None):
        self.le  = LE
        self.nob = len(LE)
        if BBOX is None:
            box = gc.BBox(LE[0].bboxGPmin, LE[0].bboxGPmax)
            for i in range (1, self.nob):
                box = box.union(LE[i].bboxGPmin)
                box = box.union(LE[i].bboxGPmax)
            self.bboxGPmin = box.pmin
            self.bboxGPmax = box.pmax
        else:
            self.bboxGPmin = BBOX[0]
            self.bboxGPmax = BBOX[1]
        self.check = "GroupE"


def findRots(UI=None, UO=None, vecNF=None):
    """Compute rotation angles to reflect an incoming ray toward an outgoing direction.

    Determines the Y and Z rotation angles necessary to orient a surface so that
    it reflects an incoming ray (UI) toward an outgoing direction (-UO). Can work
    with either incoming/outgoing ray directions or a pre-computed surface normal.

    Parameters
    ----------
    UI : gc.Vector, optional
        Direction vector of the incoming ray or sun direction (geoclide.Vector).
        Required unless vecNF is provided. Default is None.
    UO : gc.Vector, optional
        Direction vector of the outgoing ray, typically from receiver to facet center.
        The surface will be oriented to reflect UI toward -UO.
        Required unless vecNF is provided. Default is None.
    vecNF : gc.Vector, optional
        Pre-computed normal vector of the reflection surface (geoclide.Vector).
        If provided, UI and UO are not used. Allows direct specification of the
        desired surface normal. Default is None.

    Returns
    -------
    list
        A list containing rotation information:

        - **list[0]** : rotYD (float)
            Rotation angle around Y-axis in radians
        - **list[1]** : rotZD (float)
            Rotation angle around Z-axis in radians
        - **list[2]** : TTT (gc.Transform)
            Combined rotation transformation (geoclide.Transform object) that applies
            both rotations to orient the surface normal from (0, 0, 1) to the target direction

    Notes
    -----
    The function uses an iterative method to find rotation angles that align the
    initial surface normal (0, 0, 1) with the target normal computed from UI and UO.
    The algorithm applies Y-rotation first, then Z-rotation to achieve the desired
    reflection geometry.

    If vecNF is provided, it takes precedence and UI/UO are ignored.
    """
    # 1)Find the normal of the facet but filled in a vector class
    if vecNF is not None: vNF = gc.Vector(vecNF)
    else: vNF = (UI + UO)*(-0.5)
    vNF = gc.normalize(vNF)
    vNF.z = np.clip(vNF.z, -1, 1) # Avoid nan value in next operations

    # 2) Apply the inverse rotation operations to find the necessary angles
    # 2.a) Initialisation
    loop=int(0); rotY=0; rotZ=0; opeZ=0;
    # The initial value of the facet normal is (0, 0, 1) but forced to (0, 0, 0)
    # to be sure to activate the while loop below
    vNF_initial = gc.Vector(0., 0., 0.)

    # 2.b) Rotations are found in the loop bellow, at the end we check if after applying
    #      the transform to the initial normal of the facet 'vNF_initial' we have the same
    #      value as the known well oriented facet normal 'vecNF'. If no rotation has been
    #      found an error message will appear
    while (abs(vNF_initial.x - vNF.x) > 1e-4 or abs(vNF_initial.y - vNF.y) > 1e-4 or 
           abs(vNF_initial.z - vNF.z) > 1e-4):
        loop += int(1)
        if loop > 4:
            raise NameError('No rotation has been found!')

        if (loop == 1):
            rotY = np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0): opeZ = 0
            else: opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = np.arccos(opeZ)
        elif(loop == 2):
            rotY = np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0): opeZ = 0
            else: opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = -np.arccos(opeZ)
        elif(loop == 3):
            rotY = -np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0): opeZ = 0
            else: opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = np.arccos(opeZ)
        elif(loop == 4):
            rotY = -np.arccos(vNF.z)
            if (vNF.x == 0 and rotY == 0): opeZ = 0
            else: opeZ = vNF.x/np.sin(rotY)
            opeZ = np.clip(opeZ, -1, 1)
            rotZ = -np.arccos(opeZ)
 
        rotYD = np.degrees(rotY); rotZD = np.degrees(rotZ);
        TTZ = gc.get_rotateZ_tf(rotZD); TTY = gc.get_rotateY_tf(rotYD);
        TTT = TTZ*TTY
        vNF_initial = gc.normalize(TTT(gc.Vector(0., 0., 1.)))

    return [rotYD, rotZD, TTT]

def generateMTF(HELIO=Heliostat(), PR = gc.Point(0., 0., 0.)):
    """Compute transformations for curved heliostat facet orientation.

    Generates transformation matrices for each facet of a heliostat to enable
    facet curvature. Each facet is oriented such that it reflects solar rays
    toward the center of a specified receiver position.

    Parameters
    ----------
    HELIO : Heliostat, optional
        A Heliostat class object defining the base heliostat geometry and segmentation.
        Default is Heliostat().
    PR : gc.Point, optional
        Position of the receiver center as a geoclide Point object.
        Facets are oriented to focus reflected rays toward this point.
        Default is gc.Point(0., 0., 0.).

    Returns
    -------
    MTF : 2-D ndarray of Transform
        2D array of transformation matrices (geoclide.Transform objects) of shape (SPX, SPY),
        one for each facet. Each transformation positions and orients the corresponding facet.
    """
    # Heliostat is splited in facets in x and y directions
    SPX = HELIO.sPx; SPY = HELIO.sPy
    # Size in x and y of a given facet
    SFX = HELIO.hSx/SPX; SFY = HELIO.hSy/SPY
    wMx = SFX/2; wMy = SFY/2 # Size of a facet divided by 2

    POSH = gc.Point(HELIO.pos.x, HELIO.pos.y, HELIO.pos.z)
    APOSR = gc.Point(0., 0., 0.+(POSH - PR).Length())

    # Find the positions of facets and store them in matrix MPF[i][j]
    MPF = np.zeros((SPX, SPY), dtype="object") # Matrix of Point object of each facets
    for i in range (0, SPX):
        for j in range (0, SPY):
            MPF[i][j] = gc.Point(-(HELIO.hSx/2.) + (i*SFX) + wMx, -(HELIO.hSy/2.) + (j*SFY) + wMy, 0.)

    # Find transform in function of focal length (for the curve)
    MTF = np.zeros((SPX, SPY), dtype="object") # Matrix of Transform object of each facets
    for i in range (0, SPX):
        for j in range (0, SPY):
            UI = gc.Point(0., 0., 0.) - APOSR
            UI = gc.normalize(UI)
            UO = MPF[i][j] - APOSR
            UO = gc.normalize(UO)
            RINF  = findRots(UI=UI, UO=UO)
            MTF[i][j] = gc.Transform(RINF[2])

    return MTF


def generateLEfH(HELIO = Heliostat(), PR = None, THEDEG = 0., PHIDEG = 0., MTF=None):
    """Convert a heliostat to well-oriented plane facets for receiver reflection.

    Generates a list of properly oriented planar entity/facets from a heliostat object.
    Each facet is independently oriented to reflect solar rays toward a given receiver.
    This function manages the conversion of curved or segmented heliostats into their
    constituent facet entities.

    The facet indexing follows a matrix convention based on the heliostat's segmentation
    in x and y directions. See Notes section for the indexing convention.

    Parameters
    ----------
    HELIO : Heliostat, optional
        A Heliostat class object representing the heliostat to be converted.
        Default is Heliostat().
    PR : gc.Point, optional
        Position of the receiver as a geoclide.Point object. Used to orient facets
        toward the target. If None, a default point is used. Default is None.
    THEDEG : float, optional
        Solar zenith angle in degrees. Default is 0.
    PHIDEG : float, optional
        Solar azimuth angle in degrees. Default is 0.
    MTF : None | 2-D ndarray, optional
        A 2D ndarray of Transform objects of dim (SPX, SPY) representing the orientation 
        of each facet. If None, The transforms are computed automatically based on the 
        heliostat and receiver positions.

    Returns
    -------
    out : list
        List of plane Entity objects, each representing a facet properly oriented
        to reflect solar rays toward the receiver.

    Notes
    -----
    **Facet indexing convention:**

    Each facet is identified by a two-index notation **fij** where:

    - **i** is the row index (0 to SPX-1), representing position along the x-direction
    - **j** is the column index (0 to SPY-1), representing position along the y-direction

    Example with 4x4 segmentation::

                j0   j1   j2   j3
              +----+----+----+----+
        i0   |f00 |f01 |f02 |f03 |
              +----+----+----+----+
        i1   |f10 |f11 |f12 |f13 |
              +----+----+----+----+
        i2   |f20 |f21 |f22 |f23 |
              +----+----+----+----+
        i3   |f30 |f31 |f32 |f33 |
              +----+----+----+----+
                   ↑ y
              ← x

    The first row contains f00, f01, f02, f03; the second row contains f10, f11, f12, f13,
    and so on. This row-major ordering allows easy identification of any facet
    from its position in the segmented heliostat grid.
    """
    # Be sure that the correct agrs have been given
    if not isinstance(HELIO, Heliostat):
        raise Exception("HELIO must be a Heliostat class!")
    if not isinstance(PR, gc.Point):
        raise Exception("The receiver position 'PR' must be a Point class!")

    # Direction of the sun (from (x,y,z) to (0,0,0))
    vSun = gc.ang2vec(THEDEG, PHIDEG, vec_view="nadir")
    # Heliostat is splited in facets in x and y directions
    SPX = HELIO.sPx; SPY = HELIO.sPy;
    # Size in x and y of a given facet
    SFX = HELIO.hSx/SPX; SFY = HELIO.hSy/SPY
    # Focal length or distance between heliostat and receiver
    FL = HELIO.curveFL
    # Position of the heliostat
    POSH = gc.Point(HELIO.pos.x, HELIO.pos.y, HELIO.pos.z)
    # Receiver assumed position or the assumed focal length point.
    # Needed to curve the heliostat
    if (FL is not None):
        APOSR = gc.Point(0., 0., 0.+FL)
    else:
        PHTEMP = gc.Point(POSH)
        DTEMP = (PHTEMP - PR).length()
        APOSR = gc.Point(0., 0., 0.+DTEMP)
    # For the bounding box
    bboxDist = np.sqrt(HELIO.hSx*HELIO.hSx + HELIO.hSy*HELIO.hSy)/2
        
    # Initialisation
    LF = [] # List of facets
    wMx = SFX/2; wMy = SFY/2 # Size of a facet divided by 2
    # Create one facet to be ready to clone other facets
    F1 = Entity(name = "reflector", \
                materialAV = Mirror(reflectivity = HELIO.ref, roughness = HELIO.rough), \
                materialAR = Matte(), \
                geo = Plane( p1 = gc.Point(-wMx, -wMy, 0.),
                             p2 = gc.Point(wMx, -wMy, 0.),
                             p3 = gc.Point(-wMx, wMy, 0.),
                             p4 = gc.Point(wMx, wMy, 0.) ), \
                transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                 translation = np.array([0., 0., 0.]) ))
    
    # Find the positions of facets and store them in matrix MPF[i][j]
    MPF = np.zeros((SPX, SPY), dtype="object") # Matrix of Point object of each facets
    for i in range (0, SPX):
        for j in range (0, SPY):
            MPF[i][j] = gc.Point(-(HELIO.hSx/2.) + (i*SFX) + wMx, -(HELIO.hSy/2.) + (j*SFY) + wMy, 0.)

    # Find transform in function of focal length (for the curve)
    if MTF is None:
        MTF = np.zeros((SPX, SPY), dtype="object") # Matrix of Transform object of each facets
        for i in range (0, SPX):
            for j in range (0, SPY):
                UI = gc.Point(0., 0., 0.) - APOSR
                UI = gc.normalize(UI)
                UO = MPF[i][j] - APOSR
                UO = gc.normalize(UO)
                RINF  = findRots(UI=UI, UO=UO)
                MTF[i][j] = gc.Transform(RINF[2])


    # Find the general heliostat rotation transform (like helistat is a unique facet)
    UI = gc.Vector(vSun.x, vSun.y, vSun.z); UO = POSH - PR;
    UI = gc.normalize(UI); UO = gc.normalize(UO);
    RINF2  = findRots(UI=UI, UO=UO)
    TTZY = RINF2[2]

    # Apply the general rotation transform to each facet point and then apply translation.
    # This gives the final position of each facet after rotation and translation of
    # the heliostat, stored in the matrix MPFAT 
    MPFAT = np.zeros((SPX, SPY), dtype="object") # equals to MPF after application of transform
    for i in range (0, SPX):
        for j in range (0, SPY):
            tempP = gc.Point(MPF[i][j])
            tempP = TTZY(tempP)
            tempP.x += POSH.x; tempP.y += POSH.y; tempP.z += POSH.z;
            MPFAT[i][j] = gc.Point(tempP)

    # Write the initial coordinate system in term of vectors (x, y and z)
    vecX = gc.Vector(1., 0., 0.); vecY = gc.Vector(0., 1., 0.); vecZ = gc.Vector(0., 0., 1.);

    # Apply the general rotation transform to find the new coordinate system of the heliostat
    vecX = TTZY(vecX); vecY = TTZY(vecY); vecZ = TTZY(vecZ);
    vecX = gc.normalize(vecX); vecY = gc.normalize(vecY); vecZ = gc.normalize(vecZ);

    # Create the transformation matrix allowing to move between the 2 coordinate systems
    nn1 = vecX; nn2 = vecY;nn3 = vecZ; 
    mm2 = np.zeros((4,4), dtype=np.float64)
    # Fill the transformation matrix (nn3 is the new z axis)
    mm2[0,0] = nn1.x ; mm2[0,1] = nn2.x ; mm2[0,2] = nn3.x ; mm2[0,3] = 0. ;
    mm2[1,0] = nn1.y ; mm2[1,1] = nn2.y ; mm2[1,2] = nn3.y ; mm2[1,3] = 0. ;
    mm2[2,0] = nn1.z ; mm2[2,1] = nn2.z ; mm2[2,2] = nn3.z ; mm2[2,3] = 0. ;
    mm2[3,0] = 0.    ; mm2[3,1] = 0.    ; mm2[3,2] = 0.    ; mm2[3,3] = 1. ;
    # Now create the transform object with the transformation matrix and its inverse
    mm2Inv = np.transpose(mm2)
    wTo = gc.Transform(m = mm2, mInv = mm2Inv) # move from world/initial to object∕new basis
    oTw = gc.Transform(m = mm2Inv, mInv = mm2) # move from object∕new to world/initial basis

    # The normal of the heliostat vecNH = z axis of the new coordinate system
    vecNH = gc.Vector(vecZ) # stored as a vector for transformation purposes
    for i in range (0, SPX):
        for j in range (0, SPY):
            # come back to the initial coordinate system
            vecNF = oTw(vecNH)
            # apply the transform of the facet to consider the curve effect
            vecNF = MTF[i][j](vecNF)
            # Now we return to the new coordinate system, which gives
            # then the normal of the facet (not heliostat) stored in MTF[i][j]
            vecNF = wTo(vecNF)
            vecNF = gc.normalize(vecNF)
    
            # Find the rotation transform
            RINF3 = findRots(vecNF=vecNF)

            # Once the rotation angles have been found, create the facet as entity object
            tempF1 = Entity(F1)
            tempF1.transformation = Transformation( rotation = np.array([0., RINF3[0], RINF3[1]]), \
                                                    translation = np.array([MPFAT[i][j].x, MPFAT[i][j].y, MPFAT[i][j].z]), \
                                                    rotationOrder = "ZYX")
            tempPP = gc.Point(POSH)
            
            tempF1.bboxGPmin = gc.Point(tempPP.x-bboxDist, tempPP.y-bboxDist, tempPP.z-bboxDist)
            tempF1.bboxGPmax = gc.Point(tempPP.x+bboxDist, tempPP.y+bboxDist, tempPP.z+bboxDist)
            LF.append(tempF1)

    return LF


def generateBox(dimXYZ=[0.05, 0.05, 0.05], pos=gc.Point(0., 0., 0.), matAV = "LambMirror",
        ref=[1., 1., 1., 1., 1., 1.], rough=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2], rotZ = 0., gap=0.0001,
        obj_type="environment", colors=None, alpha_color=None):
    """Create a 3D box/building composed of six planar faces.

    Generates a box with six faces following Didier's 3D atmosphere convention in SMART-G.
    Each face can have different materials and properties. The origin is located at the
    center of the bottom face (Face 5), not at the center of the box.

    Face convention and orientation:
    
    - Face 0: Right   - In face: top Y+, right Z-
    - Face 1: Left    - In face: top Y+, right Z+
    - Face 2: Back    - In face: top Z-, right X+
    - Face 3: Front   - In face: top Z+, right X+
    - Face 4: Top     - In face: top Y+, right X+
    - Face 5: Bottom  - In face: top Y+, right X-

    Parameters
    ----------
    dimXYZ : list, optional
        Dimensions of the box in [x, y, z] in kilometers. Default is [0.05, 0.05, 0.05].
    pos : gc.Point, optional
        Position of the box center. Origin is at the center of Face 5 (bottom).
        Default is gc.Point(0., 0., 0.).
    matAV : str | list, optional
        Material for the front side of faces. Either:
        
        - "LambMirror" : Lambertian mirror for all faces (constant reflectivity)
        - "Mirror" : Specular mirror for all faces (with roughness)
        - list : List containing 6 material objects (e.g., Matte, LambMirror, Mirror) 
                 for each face
        
        Default is "LambMirror".
    ref : list, optional
        Reflectivity values for each face when matAV is "Mirror" or "LambMirror".
        List of 6 floats, one per face. Default is [1., 1., 1., 1., 1., 1.].
        Else ignored if matAV is a list of material objects.
    rough : list, optional
        Surface roughness for each face when matAV is "Mirror".
        List of 6 floats, one per face. Default is [0.2, 0.2, 0.2, 0.2, 0.2, 0.2].
        Else ignored if matAV is "LambMirror" or a list of material objects.
    rotZ : float, optional
        Global rotation angle in degrees around the Z-axis. Default is 0.
    gap : float, optional
        Gap to add to the global bounding box, useful for very small objects.
        Default is 0.0001.
    obj_type : str, optional
        Type of object. Choices are: 'environment', 'reflector', or 'receiver'.
        Default is 'environment'.
    colors : list, optional
        List of str colors for each of the 6 faces. If None, all faces are colored grey.
        Default is None.
    alpha_color : list, optional
        List of transparency float values (0-1) for each of the 6 faces. If None, all faces have 0.5.
        Default is None.

    Returns
    -------
    out : GroupE
        A group object (GroupE class) composed of six plane objects representing
        the box faces.

    Notes
    -----
    - Origin is at the center of Face 5 (bottom), NOT at the center of the box.
    - Global rotation in Z-axis only (other rotations not yet enabled).
    - Front side of each face uses the specified material (matAV);
      back side is always Matte (totally absorptive).
    """
    # Material AV = front part (i.e. part outside the box) of Face 0 to Face 5,
    # back part (i.e. part inside the box) will be definite as matte (totally absorbant)
    matAVL = []
    if (matAV == "Mirror") :
        for i in range (0, 6):
            matAVL.append(Mirror(reflectivity = ref[i], roughness=rough[i]))
    elif (matAV == "LambMirror") :
        for i in range (0, 6):
            matAVL.append(LambMirror(reflectivity = ref[i]))
    else :
        matAVL = matAV

    # colors
    if colors is None : colors = ['grey', 'grey', 'grey', 'grey', 'grey', 'grey']
    if alpha_color is None : alpha_color = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    # === Commun parameters ===
    # Compute the half dimensions in X, Y and Z
    wMx = dimXYZ[0]/2.; wMy = dimXYZ[1]/2.; wMz = dimXYZ[2]/2.
    
    # With the global Z rotation, 4 translations are needed in the direction after the rotation, for Face 0 to 3 
    TT = gc.get_rotateZ_tf(rotZ)
    TX = gc.Vector(1., 0., 0.); TX = TT(TX); TX = gc.normalize(TX)*wMx
    TY = gc.Vector(0., 1., 0.); TY = TT(TY); TY = gc.normalize(TY)*wMy
    
    # Initialize a numpy array list of Points (p1 to p4 to construct a face) for all faces (from face 0 to 5)
    p1_F = np.empty(6, dtype=object); p2_F = np.empty(6, dtype=object); p3_F = np.empty(6, dtype=object); p4_F = np.empty(6, dtype=object)
    
    # Initialisze rotation needed to orient correctly each face
    rotX_F = np.zeros(6, dtype='float64'); rotY_F = np.zeros(6, dtype='float64')
    rotZ_F = np.full(6, rotZ) # for Z rotation it is the same value for all faces
    
    # Initialize translation variables of all faces
    transX_F = np.zeros(6, dtype='float64'); transY_F = np.zeros(6, dtype='float64'); transZ_F = np.zeros(6, dtype='float64')
    # === End commun parameters ===
    
    
    # Face 0 unique parameters
    p1_F[0] = gc.Point(-wMz, -wMy, 0.); p2_F[0] = gc.Point(wMz, -wMy, 0.); p3_F[0] = gc.Point(-wMz, wMy, 0.); p4_F[0] = gc.Point(wMz, wMy, 0.)
    rotX_F[0] = 0.; rotY_F[0] = 90.
    transX_F[0] = pos.x+TX.x; transY_F[0] = pos.y+TX.y; transZ_F[0] = pos.z + wMz
    
    # Face 1 unique parameters
    p1_F[1] = gc.Point(-wMz, -wMy, 0.); p2_F[1] = gc.Point(wMz, -wMy, 0.); p3_F[1] = gc.Point(-wMz, wMy, 0.); p4_F[1] = gc.Point(wMz, wMy, 0.)
    rotX_F[1] = 0.; rotY_F[1] = -90.
    transX_F[1] = pos.x-TX.x; transY_F[1] = pos.y-TX.y; transZ_F[1] = pos.z + wMz
    
    # Face 2 unique parameters
    p1_F[2] = gc.Point(-wMx, -wMz, 0.); p2_F[2] = gc.Point(wMx, -wMz, 0.); p3_F[2] = gc.Point(-wMx, wMz, 0.); p4_F[2] = gc.Point(wMx, wMz, 0.)
    rotX_F[2] = -90.; rotY_F[2] = 0.
    transX_F[2] = pos.x+TY.x; transY_F[2] = pos.y+TY.y; transZ_F[2] = pos.z + wMz
    
    # Face 3 unique parameters
    p1_F[3] = gc.Point(-wMx, -wMz, 0.); p2_F[3] = gc.Point(wMx, -wMz, 0.); p3_F[3] = gc.Point(-wMx, wMz, 0.); p4_F[3] = gc.Point(wMx, wMz, 0.)
    rotX_F[3] = 90.; rotY_F[3] = 0.
    transX_F[3] = pos.x-TY.x; transY_F[3] = pos.y-TY.y; transZ_F[3] = pos.z + wMz
    
    # Face 4 unique parameters
    p1_F[4] = gc.Point(-wMx, -wMy, 0.); p2_F[4] = gc.Point(wMx, -wMy, 0.); p3_F[4] = gc.Point(-wMx, wMy, 0.); p4_F[4] = gc.Point(wMx, wMy, 0.)
    rotX_F[4] = 0.; rotY_F[4] = 0.
    transX_F[4] = pos.x; transY_F[4] = pos.y; transZ_F[4] = pos.z + 2*wMz
    
    # Face 5 unique parameters
    p1_F[5] = gc.Point(-wMx, -wMy, 0.); p2_F[5] = gc.Point(wMx, -wMy, 0.); p3_F[5] = gc.Point(-wMx, wMy, 0.); p4_F[5] = gc.Point(wMx, wMy, 0.)
    rotX_F[5] = 0.; rotY_F[5] = 180.
    transX_F[5] = pos.x; transY_F[5] = pos.y; transZ_F[5] = pos.z
    
    # Create the faces and incorporate them in a list
    LOBJ = []
    for i in range (0, 6):
        F = Entity(name = obj_type, \
                   color = colors[i], \
                   alpha_color = alpha_color[i], \
                   materialAV = matAVL[i], \
                   materialAR = Matte(), \
                   geo = Plane( p1 = p1_F[i], p2 = p2_F[i], p3 = p3_F[i], p4 = p4_F[i] ), \
                   transformation = Transformation( rotation = np.array([rotX_F[i], rotY_F[i], rotZ_F[i]]),
                                                    translation = np.array([transX_F[i], transY_F[i], transZ_F[i]]), rotationOrder="ZXY" ))
        LOBJ.append(F)
    
    # Create a group of object with a global bounding box (can improve significantly the computational time!)
    maxXY = max(pos.x, 2*max(wMx, wMy))
    p_min = gc.Point( pos.x - maxXY - gap, pos.y - maxXY - gap, pos.z - gap)
    p_max = gc.Point( pos.x + maxXY + gap, pos.y + maxXY + gap, pos.z + 2*wMz + gap )
    GOBJ = GroupE(LE = LOBJ, BBOX = [p_min, p_max])
    
    return GOBJ


def Ref_Fresnel(dirEnt, geoTrans):
    """Calculate Fresnel reflection direction for a ray on a transformed surface.

    Computes the direction of a reflected ray using simple Fresnel reflection
    based on the incident ray direction and the surface transformation.

    Parameters
    ----------
    dirEnt : gc.Vector
        Direction vector of the incident ray entering the reflecting surface.
    geoTrans : gc.Transform
        Transformation (rotation and translation) of the surface where reflection occurs.

    Returns
    -------
    out : gc.Vector
        Direction vector of the reflected ray.
    """
    if isinstance(dirEnt, gc.Vector) :
        dirE = dirEnt
    else :
        raise Exception("the dirEnt argument must be a Vector class")
    if isinstance(geoTrans, gc.Transform) :
        geoT = geoTrans
    else :
        raise Exception("the geoTrans argument must be a Transform class")

    # Default value of the surface plane normal
    NN = gc.Vector(0., 0., 1)
    
    # Real value of the normal after considering transformation
    TT = geoT
    NN = TT(NN)

    # Information needed from the incoming ray
    V = dirE
    V = gc.Vector(-V.x, -V.y, -V.z)
    
    # Use the equation of Fresnel reflection (plenty explained in pbrtv3 book)
    V = dirE + NN*(2*gc.dot(NN, V))

    # Be sure V is normalized
    V = gc.normalize(V)
    
    return V


def visualize_entity(entities, th_deg = 0., ph_deg = 0., draw_method = 'SM', ray_color = 'r', 
                     sr_view=1, xyz_limit = None, show_rays=True, rs_fac = 1):
    """Enable a 3D visualization of created objects.

    Parameters
    ----------
    entities : list | Entity
        A list of Entity objects to visualize.
    th_deg : float, optional
        The zenith angle of the sun in degrees. Default is 0.
    ph_deg : float, optional
        The azimuth angle of the sun in degrees. Default is 0.
    draw_method : str, optional
        The drawing method. 'SM' (Second Method) is the default and recommended.
        'FM' (First Method) is useful for debugging issues.
    ray_color : str, optional
        Sun rays color, e.g., 'r', 'b', 'g', etc. Default is 'r'.
    sr_view : int, optional
        Number of sun rays that can be seen in the figure. Default is 1.
    xyz_limit : dict, optional
        Dictionary specifying x, y, z view limits in km. If None (default),
        limits are automatically chosen. Example format:
        {'x_min': 0., 'x_max': 10., 'y_min': 0., 'y_max': 10., 
         'z_min': 0., 'z_max': 10.}
    show_rays : bool, optional
        Whether to show sun rays. Default is True.
    rs_fac : float, optional
        Ray scale factor. Default is 1.

    Returns
    -------
    out : matplotlib.figure.Figure
        A matplotlib figure object containing the 3D visualization.
    """

    if not isinstance(entities, (list)): entities = [entities]

    if not (all(isinstance(x, (Entity, GroupE)) for x in entities)):
        raise NameError('The only objects accepted for entities parameter are: Entity or GroupE')

    # ensure we have only Entity objects (converts if necessary GroupE to Entity objects)
    entities = convertLGtoLE(entities)

    E = entities
    E_tf = []
    box = gc.BBox()
    for i in range(0, len(E)):
        E_tf.append(E[i].get_transformation())
        box = box.union((E[i].bboxGPmin))
        box = box.union((E[i].bboxGPmax))
 
    box_center = box.pmin + 0.5*(box.pmax - box.pmin)
    box_max_size = gc.vmax(box.pmax - box.pmin)
    pmin_n = gc.Point(box_center.x - 0.5*box_max_size, 
                      box_center.y - 0.5*box_max_size, 
                      box_center.z - 0.5*box_max_size)
    pmax_n = gc.Point(box_center.x + 0.5*box_max_size, 
                      box_center.y + 0.5*box_max_size, 
                      box_center.z + 0.5*box_max_size)
    box_n = gc.BBox(pmin_n, pmax_n)

    # calculate the sun direction vector
    vSun = gc.ang2vec(th_deg, ph_deg, vec_view='nadir')
    wsx = -vSun.x; wsy=-vSun.y; wsz=-vSun.z

    ltmesh = []
    lMir_int = int(0)
    E_rec = []; E_ref = []
    E_rec_tf = []; E_ref_tf = []
    for i in range(0, len(E)):
        if (E[i].name == "reflector"): 
            E_ref.append(E[i])
            E_ref_tf.append(E_tf[i])
        if (E[i].name == "receiver") : 
            E_rec.append(E[i])
            E_rec_tf.append(E_tf[i])

    nbRef = len(E_ref)
    xr = [None]*nbRef; yr = [None]*nbRef; zr = [None]*nbRef
    atLeastOneInt = [False]*nbRef
    TabPhoton2 = []

    for k in range (0, len(E_ref)):
        # Get the transformation
        tt = E_ref_tf[k]

        photon_pos = gc.Point(wsx+E_ref[k].transformation.transx, wsy+E_ref[k].transformation.transy, wsz+E_ref[k].transformation.transz)
        photon = gc.Ray(o = photon_pos, d = vSun, maxt = 1200.)
    
        if isinstance(E_ref[k].geo, Plane):
           # Vertex triangle indices
            vi = np.array([np.array([0, 1, 2]),                   # indices or triangle 1
                           np.array([2, 3, 1])], dtype=np.int32)  # indices of triangle 2

            # List of points of the plane
            P = np.array([np.array([E_ref[k].geo.p1.x, E_ref[k].geo.p1.y, E_ref[k].geo.p1.z]),
                          np.array([E_ref[k].geo.p2.x, E_ref[k].geo.p2.y, E_ref[k].geo.p2.z]),
                          np.array([E_ref[k].geo.p3.x, E_ref[k].geo.p3.y, E_ref[k].geo.p3.z]),
                          np.array([E_ref[k].geo.p4.x, E_ref[k].geo.p4.y, E_ref[k].geo.p4.z])], dtype = np.float64)
            
            tmesh = gc.TriangleMesh(vertices=P, faces=vi)
        elif isinstance(E_ref[k].geo, Spheric):
            sphere = gc.Sphere(E_ref[k].geo.radius, E_ref[k].geo.z0, E_ref[k].geo.z1, E_ref[k].geo.phi)
            tmesh = sphere.to_trianglemesh()
        else: 
            raise NameError('This geometry is unknown or not yet accepted!')
        
        tmesh.apply_tf(tt)
        ltmesh.append(tmesh)

        ds = gc.calc_intersection(tmesh, photon)
        if(ds['is_intersection'].values and ds['thit'].values < float('inf')):
            atLeastOneInt[k] = True
            lMir_int += int(1)
            p_hit = gc.Point(ds['phit'].values)
            t_hit = ds['thit'].values
            tr = np.linspace(t_hit*0.98*(1/rs_fac), t_hit, 100)
            xr[k] = photon.o.x + tr*photon.d.x
            yr[k] = photon.o.y + tr*photon.d.y
            zr[k] = photon.o.z + tr*photon.d.z
            vecTemp = Ref_Fresnel(dirEnt = photon.d, geoTrans = tt)
            TabPhoton2 = np.append(TabPhoton2, gc.Ray(o=p_hit, d=vecTemp, maxt=120))


    xr2 = [None]*lMir_int; yr2 = [None]*lMir_int; zr2 = [None]*lMir_int
    atLeastOneInt2 = [False]*lMir_int

    for k in range (0, len(E_rec)):
        # Get the transformation
        tt = E_rec[k].get_transformation()

        if isinstance(E_rec[k].geo, Plane):
            # Vertex triangle indices
            vi = np.array([np.array([0, 1, 2]),                   # indices or triangle 1
                           np.array([2, 3, 1])], dtype=np.int32)  # indices of triangle 2

            # List of points of the plane
            P = np.array([np.array([E_rec[k].geo.p1.x, E_rec[k].geo.p1.y, E_rec[k].geo.p1.z]),
                          np.array([E_rec[k].geo.p2.x, E_rec[k].geo.p2.y, E_rec[k].geo.p2.z]),
                          np.array([E_rec[k].geo.p3.x, E_rec[k].geo.p3.y, E_rec[k].geo.p3.z]),
                          np.array([E_rec[k].geo.p4.x, E_rec[k].geo.p4.y, E_rec[k].geo.p4.z])], dtype = np.float64)
            
            tmesh = gc.TriangleMesh(vertices=P, faces=vi)
        elif isinstance(E_rec[k].geo, Spheric):
            sphere = gc.Sphere(E_rec[k].geo.radius, E_rec[k].geo.z0, E_rec[k].geo.z1, E_rec[k].geo.phi)
            tmesh = sphere.to_trianglemesh()
        else:
            raise NameError('This geometry is unknown or not yet accepted!')
        tmesh.apply_tf(tt)
        ltmesh.append(tmesh)

        for i in range(0, lMir_int):
            ds = gc.calc_intersection(tmesh, TabPhoton2[i])
            if(ds['is_intersection'].values and ds['thit'].values < float('inf')):
                atLeastOneInt2[i] = True
                p_hit = gc.Point(ds['phit'].values)
                t_hit = ds['thit'].values
                tr = np.linspace(TabPhoton2[i].mint, t_hit, 100)
                xr2[i] = TabPhoton2[i].o.x + tr*TabPhoton2[i].d.x
                yr2[i] = TabPhoton2[i].o.y + tr*TabPhoton2[i].d.y
                zr2[i] = TabPhoton2[i].o.z + tr*TabPhoton2[i].d.z

    # create the matplotlib figure
    fig = plt.figure()#figsize=[128, 96])
    ax = fig.add_subplot(111, projection=Axes3D.name)
    ax.scatter([-1,1], [-1,1], [-1,1], alpha=0.0)

    for itmesh, tmesh in enumerate(ltmesh):
        # Triangles mesh parameters for plot
        # First method (draw even if there is error with an object, useful for debug):
        # ----------------------------->
        if (draw_method == 'FM'):
            for itri in range(0, tmesh.ntriangles):
                p0 = gc.Point(tmesh.vertices[tmesh.faces[itri,0],:])
                p1 = gc.Point(tmesh.vertices[tmesh.faces[itri,1],:])
                p2 = gc.Point(tmesh.vertices[tmesh.faces[itri,2],:])
                Mat = np.array([[p0.x, p0.y, p0.z], \
                                [p1.x, p1.y, p1.z], \
                                [p2.x, p2.y, p2.z]])
                face1 = mp3d.art3d.Poly3DCollection([Mat], alpha = E[itmesh].alpha_color, linewidths=0.2)
                face1.set_facecolor(mcolors.to_rgba(E[itmesh].color))
                ax.add_collection3d(face1)

        # Second method (better visual, avoid some matplotlib bugs):
        # ----------------------------->
        if (draw_method == 'SM'):
            p0_t0 = gc.Point(tmesh.vertices[tmesh.faces[0,0],:])
            p1_t0 = gc.Point(tmesh.vertices[tmesh.faces[0,1],:])
            p2_t0 = gc.Point(tmesh.vertices[tmesh.faces[0,2],:])
            p0_t1 = gc.Point(tmesh.vertices[tmesh.faces[1,0],:])
            p1_t1 = gc.Point(tmesh.vertices[tmesh.faces[1,1],:])
            p2_t1 = gc.Point(tmesh.vertices[tmesh.faces[1,2],:])
            Mat = np.array([[p0_t0.x, p0_t0.y, p0_t0.z], \
                            [p1_t0.x, p1_t0.y, p1_t0.z], \
                            [p2_t0.x, p2_t0.y, p2_t0.z], \
                            [p0_t1.x, p0_t1.y, p0_t1.z], \
                            [p1_t1.x, p1_t1.y, p1_t1.z], \
                            [p2_t1.x, p2_t1.y, p2_t1.z]])
            
            if (np.array_equal(Mat[:,0], np.full((6), Mat[0,0]))):
                yy, zz = np.meshgrid(Mat[:,0], Mat[:,2])
                xx = np.full((6,6), Mat[0,0])
                ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba(E[itmesh].color), alpha = E[itmesh].alpha_color, \
                                linewidth=0.2, antialiased=True)
            elif (np.array_equal(Mat[:,1], np.full((6), Mat[0,1]))):
                xx, zz = np.meshgrid(Mat[:,0], Mat[:,2])
                yy = np.full((6,6), Mat[0,1])
                ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba(E[itmesh].color), alpha = E[itmesh].alpha_color, \
                                linewidth=0.2, antialiased=True)
            elif (np.array_equal(Mat[:,2], np.full((6), Mat[0,2]))): # need to be verified
                xx, yy = np.meshgrid(Mat[:,0], Mat[:,1])
                zz = np.full((6,6), Mat[0,2])
                ax.plot_surface(xx, yy, zz, color = mcolors.to_rgba(E[itmesh].color), alpha = E[itmesh].alpha_color, \
                                linewidth=0.2, antialiased=True)
            else:
                ax.plot_trisurf(Mat[:,0], Mat[:,1], Mat[:,2], color = mcolors.to_rgba(E[itmesh].color), \
                                alpha = 0.5, linewidth=0.2, antialiased=True)

    # ==============================================
    # plot all the geometries
    if (show_rays):
        for i in range(0, nbRef):
            if (atLeastOneInt[i] and i%sr_view ==0): ax.plot(xr[i], yr[i], zr[i], color=ray_color, linewidth=1*rs_fac)

        for i in range(0, lMir_int):
            if (atLeastOneInt2[i] and i%sr_view ==0): ax.plot(xr2[i], yr2[i], zr2[i], color=ray_color, linewidth=1*rs_fac)

    if (xyz_limit is not None):
        ax.set_xlim3d(xyz_limit['x_min'], xyz_limit['x_max'])
        ax.set_ylim3d(xyz_limit['y_min'], xyz_limit['y_max'])
        ax.set_zlim3d(xyz_limit['z_min'], xyz_limit['z_max'])
    else: # generic local visualization
        ax.set_xlim3d(box_n.pmin.x, box_n.pmax.x)
        ax.set_ylim3d(box_n.pmin.y, box_n.pmax.y)
        ax.set_zlim3d(box_n.pmin.z, box_n.pmax.z)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the geometries
    fig = ax.get_figure()
    return fig


def generateHfP(THEDEG=0., PHIDEG = 0., PH = [gc.Point(0., 0., 0.)], PR = gc.Point(0., 0., 0.), \
                HSX = 0.001, HSY = 0.001, REF = 1, ROUGH=0, HTYPE = None, LMTF = None):
    """Generate well-oriented Heliostats from their positions.

    Generates a list of heliostat entities oriented to reflect sun rays toward
    a receiver. Can handle either planar heliostats or curved (faceted) heliostats
    depending on the HTYPE parameter.

    Parameters
    ----------
    THEDEG : float, optional
        Sun zenith angle in degrees. Default is 0.
    PHIDEG : float, optional
        Sun azimuth angle in degrees. Default is 0.
    PH : list of Point, optional
        Coordinates of the center of heliostats. List of Point objects (geoclide).
        Default is [gc.Point(0., 0., 0.)].
    PR : Point, optional
        Coordinate of the center of the receiver (geoclide Point object).
        Default is gc.Point(0., 0., 0.).
    HSX : float, optional
        Heliostat size in x-axis in kilometers. Default is 0.001.
    HSY : float, optional
        Heliostat size in y-axis in kilometers. Default is 0.001.
    REF : float, optional
        Reflectivity of the heliostats. Default is 1.
    ROUGH : float, optional
        Surface roughness of the heliostats. Default is 0.
    HTYPE : Heliostat or None, optional
        If specified, must be a Heliostat class instance for generating curved
        (faceted) heliostats. If None (default), generates planar heliostats.
    LMTF : None or object, optional
        Under development. Default is None.

    Returns
    -------
    out : list
        List of Entity or GroupE objects, each properly oriented to
        reflect solar rays towards the receiver.
    """
    PH_ = PH.copy()
    lObj = []

    # Case where the heliostat is totally plane
    if (HTYPE is None):
        # compute the sun direction vector
        vSun = gc.ang2vec(THEDEG, PHIDEG, vec_view='nadir')
        bboxDist = np.sqrt(HSX*HSX + HSY*HSY)/2

        Hxx = HSX/2; Hyy = HSY/2
        objM = Entity(name = "reflector", \
                      materialAV = Mirror(reflectivity = REF, roughness = ROUGH), \
                      materialAR = Matte(reflectivity = 0.), \
                      geo = Plane( p1 = gc.Point(-Hxx, -Hyy, 0.),
                                   p2 = gc.Point(Hxx, -Hyy, 0.),
                                   p3 = gc.Point(-Hxx, Hyy, 0.),
                                   p4 = gc.Point(Hxx, Hyy, 0.) ), \
                      transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                       translation = np.array([0., 0., 0.]) ))


        for i in range (0, len(PH)):
            # 1) Find the normalized vector colinear (and same dir) to the normal of heliostat surface
            vecHR = PH_[i]-PR
            vecHR = gc.normalize(vecHR)

            # 2) Find the necessary rotations to apply on the heliostat to reflect to the receiver
            rInfo = findRots(UI=vSun, UO=vecHR)
            rotYD = rInfo[0]; rotZD = rInfo[1];

            # 3) Once the rotation angles have been found, create heliostat objects
            objMi = Entity(objM);
            objMi.bboxGPmin = gc.Point(PH_[i].x-bboxDist, PH_[i].y-bboxDist, PH_[i].z-bboxDist)
            objMi.bboxGPmax = gc.Point(PH_[i].x+bboxDist, PH_[i].y+bboxDist, PH_[i].z+bboxDist)
            objMi.transformation = Transformation( rotation = np.array([0., rotYD, rotZD]), \
                                                   translation = np.array([PH_[i].x, PH_[i].y, PH_[i].z]), \
                                                   rotationOrder = "ZYX")
            lObj.append(objMi)
    # Case where the heliostat is composed by facets (i.g. to consider the curvature)
    else:
        # Take the commun parameters of all heliostats
        SPX = HTYPE.sPx; SPY = HTYPE.sPy; HSX = HTYPE.hSx; HSY = HTYPE.hSy; CURVE_FL = HTYPE.curveFL;
        
        # Generate all the facets and store them as entity object in a list 
        for i in range (0, len(PH)):
            H0 = Heliostat(SPX=SPX, SPY=SPY, HSX=HSX, HSY=HSY, CURVE_FL=CURVE_FL, POS=PH_[i], REF=REF, ROUGH=ROUGH)
            if LMTF is None: TLE = generateLEfH(HELIO=H0, PR=PR, THEDEG=THEDEG, PHIDEG=PHIDEG)
            else: TLE = generateLEfH(HELIO=H0, PR=PR, THEDEG=THEDEG, PHIDEG=PHIDEG, MTF = LMTF[i])
            GTEMP = GroupE(LE = TLE)
            lObj.append(GTEMP)

    return lObj


def generateHfA(THEDEG=0., PHIDEG = 0., PR = gc.Point(0., 0., 50.), MINANG=0., \
                MAXANG=360., GAPDEG = 5., FDRH = 0.1, NBH = 10, GAPDIST = 0.01, \
                HSX = 0.001, HSY = 0.001, PILLH = 0.006, REF = 1, ROUGH=0,
                HTYPE=None, LMTF = None, RLPH = False):
    """Generate well-oriented Heliostats arranged in an angular sector around receiver.

    Generates heliostats positioned between MINANG and MAXANG angles, properly
    oriented to reflect sun rays toward a central receiver. Heliostats are arranged
    in concentric patterns with specified angular and radial gaps.

    The angular coordinate system is defined as:

    .. code-block:: text

        y
        ^ 
        |/) ANG
        ---> x

    where ANG is measured from the positive x-axis.

    Parameters
    ----------
    THEDEG : float, optional
        Sun zenith angle in degrees. Default is 0.
    PHIDEG : float, optional
        Sun azimuth angle in degrees. Default is 0.
    PR : Point, optional
        Coordinate of the center of the receiver (geoclide Point object).
        Heliostats are filled between MINANG and MAXANG around this receiver.
        Default is gc.Point(0., 0., 50.).
    MINANG : float, optional
        Minimum angular position in degrees. Default is 0.
    MAXANG : float, optional
        Maximum angular position in degrees. Default is 360.
    GAPDEG : float, optional
        Angular spacing in degrees for placing heliostats between MINANG and MAXANG.
        Default is 5.
    FDRH : float, optional
        First distance between receiver and heliostat center in kilometers.
        Default is 0.1.
    NBH : int, optional
        Number of heliostats to place at each angular position (radial direction).
        Default is 10.
    GAPDIST : float, optional
        Radial gap between heliostats in kilometers after the first distance FDRH.
        Default is 0.01.
    HSX : float, optional
        Heliostat size in x-axis in kilometers. Default is 0.001.
    HSY : float, optional
        Heliostat size in y-axis in kilometers. Default is 0.001.
    PILLH : float, optional
        Pillar height (distance from ground to heliostat) in kilometers.
        Default is 0.006.
    REF : float, optional
        Reflectivity of the heliostats. Default is 1.
    ROUGH : float, optional
        Surface roughness of the heliostats. Default is 0.
    HTYPE : Heliostat or None, optional
        If specified, must be a Heliostat class instance for generating curved
        (faceted) heliostats. If None (default), generates planar heliostats.
    LMTF : None or object, optional
        Under development. Default is None.
    RLPH : bool, optional
        If True, also return the list of heliostat positions. Default is False.

    Returns
    -------
    out1 : list
        List of heliostat Entity or GroupE objects arranged in the angular sector.
    out2 : list
        If RLPH is True, also returns the list of heliostat center positions
        (geoclide Point objects).
    """
    # I) Find the position of all heliostats
    lenpH = int(  ( (MAXANG-MINANG)/GAPDEG )*NBH  )
    
    # To avoid a given bug
    if (MAXANG-MINANG < 360.000000001 and MAXANG-MINANG > 359.999999999):
        nbI = int(lenpH/NBH)
    else:
        nbI = int(lenpH/NBH) + 1

    print("Total number of Heliostats = ", nbI*NBH)
    
    pH = []
    myRotZ = MINANG

    if (MINANG != MAXANG):
        for i in range (0, nbI):
            Dhr = FDRH
            for j in range (0, NBH):
                myP = gc.Point(Dhr, 0., 0.)
                RotZT = gc.get_rotateZ_tf(myRotZ)
                myP=RotZT(myP)
                pH.append( gc.Point(myP.x, myP.y, myP.z+PILLH) )
                Dhr += GAPDIST
            myRotZ += GAPDEG
    else:
        Dhr = FDRH
        RotZT = gc.get_rotateZ_tf(myRotZ)
        for j in range (0, NBH):
            myP = gc.Point(Dhr, 0., 0.)
            myP=RotZT(myP)
            pH.append( gc.Point(myP.x, myP.y, myP.z+PILLH) )
            Dhr += GAPDIST      


    # II) Creation of heliostats
    lObj = []

    # Case where the heliostat is totally plane
    if (HTYPE is None):
        # calculate the sun direction vector
        vSun = gc.ang2vec(THEDEG, PHIDEG, vec_view='nadir')
        bboxDist = np.sqrt(HSX*HSX + HSY*HSY)/2

        Hxx = HSX/2; Hyy = HSY/2
        objM = Entity(name = "reflector", \
                      materialAV = Mirror(reflectivity = REF, roughness = ROUGH), \
                      materialAR = Matte(), \
                      geo = Plane( p1 = gc.Point(-Hxx, -Hyy, 0.),
                                   p2 = gc.Point(Hxx, -Hyy, 0.),
                                   p3 = gc.Point(-Hxx, Hyy, 0.),
                                   p4 = gc.Point(Hxx, Hyy, 0.) ), \
                      transformation = Transformation( rotation = np.array([0., 0., 0.]), \
                                                       translation = np.array([0., 0., 0.]) ))

        for i in range (0, len(pH)):
            # 1) The vector of the photon after a reflection (here the opposite direction) 
            vecHR = pH[i]-PR
            vecHR = gc.normalize(vecHR)

            # 2) The incoming (vSun) and outcoming (vecHR) directions are known then find
            #    the rotation angles
            rInfo = findRots(UI=vSun, UO=vecHR)
            rotYD = rInfo[0]; rotZD = rInfo[1]

            # 3) Once the rotation angles have been found, create heliostat objects 
            objMi = Entity(objM)
            objMi.bboxGPmin = gc.Point(pH[i].x-bboxDist, pH[i].y-bboxDist, pH[i].z-bboxDist)
            objMi.bboxGPmax = gc.Point(pH[i].x+bboxDist, pH[i].y+bboxDist, pH[i].z+bboxDist)
            objMi.transformation = Transformation( rotation = np.array([0., rotYD, rotZD]), \
                                                   translation = np.array([pH[i].x, pH[i].y, pH[i].z]), \
                                                   rotationOrder = "ZYX")
            lObj.append(objMi)
        
    # Case where the heliostat is composed by facets (i.g. to consider the curvature)
    else:
        # Take the commun parameters of all heliostats
        SPX = HTYPE.sPx; SPY = HTYPE.sPy; HSX = HTYPE.hSx; HSY = HTYPE.hSy; CURVE_FL = HTYPE.curveFL
        
        # Generate all the facets and store them as entity object in a list 
        for i in range (0, len(pH)):
            H0 = Heliostat(SPX=SPX, SPY=SPY, HSX=HSX, HSY=HSY, CURVE_FL=CURVE_FL, POS=pH[i], REF=REF, ROUGH=ROUGH)
            if LMTF is None: TLE = generateLEfH(HELIO=H0, PR=PR, THEDEG=THEDEG, PHIDEG=PHIDEG)
            else: TLE = generateLEfH(HELIO=H0, PR=PR, THEDEG=THEDEG, PHIDEG=PHIDEG, MTF = LMTF[i])
            GTEMP = GroupE(LE = TLE)
            lObj.append(GTEMP)
    
    if (RLPH):
        return lObj, pH
    else:
        return lObj
    

def convertLGtoLE(LGOBJ):
    """Convert a mixed list of Entity and GroupE objects to Entity objects only.

    Flattens groups by expanding all GroupE objects into their constituent
    Entity objects, resulting in a list containing only Entity objects.

    Parameters
    ----------
    LGOBJ : list
        List containing Entity and/or GroupE objects to be converted.

    Returns
    -------
    out : list
        Flattened list containing only Entity objects. GroupE objects are
        converted into their constituent Entity objects.
    """
    nGObj=len(LGOBJ)
    LOBJ=[]

    for i in range (0, nGObj):
        if isinstance(LGOBJ[i], GroupE):
            LOBJ.extend(LGOBJ[i].le)
        elif isinstance(LGOBJ[i], Entity):
            LOBJ.append(LGOBJ[i])
        else:
            raise NameError('In the list, only Entity and GroupE classes are autorised!')

    return LOBJ


def rotate_vector(vector, rot_x, rot_y, rot_z, rot_order="xyz"):
    """
    Definition of the function rotate_vector

    coordinate system convention:

      y
      ^   x : right; y : front; z : top
      |
    z X -- > x

    Given a vector and rotations to perform to this vector in degrees
    in the x,y,z axes, with in option the rotation order

    Arg:
    v         : A direction described by Vector class object
    rotx,y,z  : Rotations in x,y and z in degrees
    rot_order : str with the order of rotations i.g. 'xyz', zxy', ...

    Return:
    rotated_vector : The rotated (normalized) direction (also a Vector class)
    """
    TT = gc.Transform()
    tr_x = gc.get_rotateX_tf(rot_x)
    tr_y = gc.get_rotateY_tf(rot_y)
    tr_z = gc.get_rotateZ_tf(rot_z) 
    if rot_order == "XYZ":
        TT = tr_x*tr_y*tr_z
    elif rot_order == "XZY":
        TT = tr_x*tr_z*tr_y
    elif rot_order == "YXZ":
        TT = tr_y*tr_x*tr_z
    elif rot_order == "YZX":
        TT = tr_y*tr_z*tr_x
    elif rot_order == "ZXY":
        TT = tr_z*tr_x*tr_y
    elif rot_order == "ZYX":
        TT = tr_z*tr_y*tr_x
    else:
        raise NameError("Unknown rot_order value!")
    rotated_vector = TT(vector)
    rotated_vector = gc.normalize(rotated_vector)

    return rotated_vector


def interpolate_refls_from_wls (wls, refls, wls_new, extrapolate=False):
    """
        Definition: Giving a set of wavelengths (wls) and reflectivities (refls),
                    get the interpolated reflectivities folowing the new set of wavelengths (wls_total)
    
    ==== ARGS:
    wls     : List/array of wavelengths
    refls   : List/array with reflectivities at each wavelength of wls 
    wls_new : List/array of the new wavelengths where we want to interpolate

    ==== RETURN:
    refls_new : numpy array with the interpolated reflectivities
    """

    if extrapolate: f = interpolate.interp1d(wls, refls, fill_value='extrapolate')
    else : f = interpolate.interp1d(wls, refls, fill_value=(refls[0],refls[-1]), bounds_error=False)

    refls_new = f(wls_new)

    # Ensure relfectivities are between 0 and 1
    refls_new[refls_new<0] = 0
    refls_new[refls_new>1] = 1

    return refls_new


def is_comment(s):
    """
    function to check if a line
    starts with some character.
    Here # for comment
    """
    # return true if a line starts with #
    return s.startswith('#')

def extractPoints(filename):
    """Extract heliostat coordinates from a file.

    Reads a file and extracts the (x, y, z) coordinates of each heliostat,
    returning them as geoclide Point objects.

    The input file must follow this format:

    - First line: comment line beginning with '#'
    - Second line: empty line
    - Subsequent lines: x, y, and z coordinates of each heliostat, separated by commas

    Parameters
    ----------
    filename : str | pathlib.Path
        Path to the file containing the heliostat coordinates.

    Returns
    -------
    out : list
        List of geoclide.Point objects, each containing the x, y, and z coordinates
        of a heliostat.
    """

    # First check if filename is an str type
    try:
        with open(filename, "r") as file:
            for curline in dropwhile(is_comment, file):
                insideFile = file.read()
    except FileNotFoundError:
        print(str(filename) + ' has been not found')
    except IOError:
        print("Enter/Exit error with " + str(filename))
            
    # Looking for a float and fill it in listVal
    listVal = re.findall(r"-?[0-9]+\.?[0-9]*", insideFile)
        
    # Number of dimension and number of heliostats
    nbDim = 3 # x, y and z --> 3 dim
    nbH = int(len(listVal)/nbDim)

    # # Fill the x, y and z coordinates into a list of Point classes
    lPH = []
    for i in range (0, nbH):
        lPH.append(  gc.Point( float(listVal[i*nbDim]), float(listVal[(i*nbDim)+1]),
                               float(listVal[(i*nbDim)+2]) )  )

    return lPH
