#!/usr/bin/env python
# encoding: utf-8


from __future__ import print_function, division, absolute_import

import warnings
warnings.simplefilter("ignore",DeprecationWarning)
from pylab import figure, subplot2grid, tight_layout, setp, subplots, xlabel, ylabel, FormatStrFormatter
import numpy as np
np.seterr(invalid='ignore', divide='ignore') # ignore division by zero errors
import xarray as xr
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import floating_axes
from matplotlib.projections import PolarAxes
from matplotlib import cm
from luts.luts import plot_polar, transect2D as _transect2D_orig, Idx, Idx_base, MLUT
from smartg.atmosphere import diff1
from smartg.water import diff2


def transect2D(lut, index=None, vmin=None, vmax=None, sym=True, swap='auto', fig=None, sub=121, color='k', percent=False, fmt='-'):
    '''
    Transect of 2D LUT - Fixed version that reuses existing axes
    
    This is a patched version of luts.transect2D that properly reuses existing
    subplot axes when overlaying multiple transects on the same figure.
    '''
    assert lut.ndim == 2

    if fig is None:
        fig = figure(figsize=(4.5, 2.5))

    if swap == 'auto':
        if ('azi' in lut.names[1].lower()) and ('azi' not in lut.names[0].lower()):
            swap = True
        else:
            swap = False

    # ax1 is angle, ax2 is radius
    if swap:
        ax1, ax2 = lut.axes[1], lut.axes[0]
        name1, name2 = lut.names[1], lut.names[0]
        data = np.swapaxes(lut.data, 0, 1)
    else:
        ax1, ax2 = lut.axes[0], lut.axes[1]
        name1, name2 = lut.names[0], lut.names[1]
        data = lut.data

    if vmin is None:
        vmin = np.amin(lut.data[~np.isnan(lut.data)])
    if vmax is None:
        vmax = np.amax(lut.data[~np.isnan(lut.data)])
    if vmin == vmax:
        vmin -= 0.001
        vmax += 0.001
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    if percent:
        vmin = 0.
        vmax = 100.

    ax1_scaled = ax1
    label2 = name2

    # convert Idx instance to index if necessary
    if isinstance(index, Idx_base):
        idx_result = np.around(index.index(ax1))
        index = int(idx_result.flat[0])
    mirror_index = (ax1_scaled.shape[0] // 2 + index) % ax1_scaled.shape[0]

    if swap:
        title = lut.axes[1][index]
    else:
        title = lut.axes[0][index]

    ax2_min = np.amin(ax2)
    ax2_max = np.amax(ax2)
    label1 = name1 + ' {:7.2f}'.format(title)

    # Parse subplot specification
    nrows = sub // 100
    ncols = (sub // 10) % 10
    idx = (sub % 10) - 1  # 0-based index

    # Check if subplot already exists by using a marker attribute
    ax_cart = None
    marker_name = f'_transect2D_sub_{sub}'
    if hasattr(fig, marker_name):
        ax_cart = getattr(fig, marker_name)
    
    is_new_axes = ax_cart is None
    if is_new_axes:
        ax_cart = fig.add_subplot(sub)
        setattr(fig, marker_name, ax_cart)  # Store reference
        ax_cart.grid(True)
        ax_cart.set_xlabel(label2)
        if sym:
            ax_cart.set_xlim(-ax2_max, ax2_max)
        else:
            ax_cart.set_xlim(ax2_min, ax2_max)
        ax_cart.set_ylim(vmin, vmax)
        ax_cart._transect2D_first = True
    else:
        # Expand ylim to accommodate new data
        current_ylim = ax_cart.get_ylim()
        new_vmin = min(current_ylim[0], vmin)
        new_vmax = max(current_ylim[1], vmax)
        ax_cart.set_ylim(new_vmin, new_vmax)
    
    ax_cart.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    # Plot transects
    ax_cart.plot(ax2, data[index, :], fmt, color=color)
    if sym:
        ax_cart.plot(-ax2, data[mirror_index, :], fmt, color=color)

    if lut.desc is not None:
        ax_cart.set_title(lut.desc)


def mdesc(desc, logI=False):
    """
    Format Stokes parameter description for display with LaTeX notation.

    Parses a description string to extract Stokes parameter, direction, and other 
    components, then formats them with proper LaTeX notation including directional 
    arrows (up/down).

    Parameters
    ----------
    desc : str
        Description string in format 'Stokes_direction(component)_info' 
        (e.g., 'I_up(TOA)', 'Q_down(0+)').
    logI : bool, optional
        If True and Stokes parameter is 'I', prepends 'log10' to the output. 
        Default is False.

    Returns
    -------
    str
        Formatted LaTeX string with Stokes parameter, directional arrow, 
        component subscripts, and optional log scale notation.

    Examples
    --------
    >>> mdesc('I_up(TOA)')
    '$I^{\\uparrow}_{TOA}$'
    >>> mdesc('I_up(TOA)', logI=True)
    '$log_{10} I^{\\uparrow}_{TOA}$'
    >>> mdesc('Q_down(0+)')
    '$Q^{\\downarrow}_{0+}$'
    """
    sep1=desc.find('_')
    sep2=desc.find('(')
    sep3=desc.find(')')
    if sep1 == 1 : stokes=desc[0:1]
    elif sep1 == 2 : stokes=desc[sep1-2:sep1]
    elif sep1 == 4 : stokes=desc[sep1-4:sep1]
    else : stokes=desc[0:sep1]
    dir=desc[sep1+1:sep2-1]

    if logI and stokes=='I':
        pref=r'$log_{10} '
    else:
        pref=r'$'
        
    if dir == 'up':
        return pref + stokes + r'^{\uparrow}' + '_{'+desc[sep2+1:sep3]+'}' + desc[sep3+1:] +'$'
    else:
        return pref + stokes + r'^{\downarrow}' + '_{'+desc[sep2+1:sep3]+'}' + desc[sep3+1:] +'$'


def lut_to_xr(lut):
    """
    Convert a LUT object to xr.DataArray, preserving the desc as long_name.
    
    Parameters
    ----------
    lut : LUT
        A LUT object with `.to_xarray()` method and optional `.desc` attribute
        
    Returns
    -------
    xr.DataArray
        DataArray with dimensions and description preserved
    """
    da = lut.to_xarray()
    if hasattr(lut, 'desc') and lut.desc is not None:
        da.attrs['long_name'] = lut.desc
    return da
    

def smartg_view(mlut, logI=False, QU=False, Circ=False, full=False, field='up (TOA)', prefix='', ind=[0], cmap=None, fig=None, subdict=None,
        Imin=None, Imax=None, Pmin=0, Pmax=100):
    '''
    visualization of a smartg MLUT

    Options:
        logI: shows log10 of I
        Circ: shows Circular polarization 
        QU:  shows Q U and DoP
        field: level of output
        prefix: eventually a prefix for field
        ind: list of indices of azimutal planes
        full: shows all
        cmap: color map
        fig: already existing figure
        subdict: dictionnary of LUT subsetter (see LUT class , sub() method)
        Imin: min value of I
        Imax: max value of I
        Pmin: min value of DOP
        Pmax: max value of DOP

    Outputs:
    if full is False, it returns 1 figure
    if full is True,  it returns 2 figures
    '''

    I = mlut[prefix+'I_' + field]
    Q = mlut[prefix+'Q_' + field]
    U = mlut[prefix+'U_' + field]
    V = mlut[prefix+'V_' + field]

    if subdict is not None :
        I = I.sub(d=subdict)
        Q = Q.sub(d=subdict)
        U = U.sub(d=subdict)
        V = V.sub(d=subdict)

    # Linearly polarized reflectance
    IPL = (Q*Q + U*U).apply(np.sqrt, 'Lin. Pol. ref.')
    
    # Polarized reflectance
    IP = (Q*Q + U*U + V*V).apply(np.sqrt, 'Pol. ref.')

    # Degree of Linear Polarization (%)
    DoLP = 100*IPL/I
    DoLP.desc = prefix+r'$DoLP$'
    
    # Angle of Linear Polarization (deg)
    AoLP = (U/Q)
    AoLP.apply(np.arctan)*90/np.pi
    AoLP.desc = prefix+r'$AoLP$'
    
    # Degree of Circular Polarization (%)
    DoCP = 100*V.apply(abs)/I
    DoCP.desc = prefix+r'$DoCP$'

    # Degree of Polarization (%)
    DoP = 100*IP/I
    DoP.desc = prefix+r'$DoP\,(\%)$'

    if isinstance(ind, Idx_base):
        ind = np.round(ind.index(mlut.axes['Azimuth angles'])).astype(int)

    if not full:
        if QU:
            if fig is None: fig = figure(figsize=(9, 9))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                plot_polar_xr(lut_to_xr(lI),  index=ind, rect=421, sub=423, fig=fig, cmap=cmap, vmin=Imin, vmax=Imax)
            else:
                I.desc = mdesc(I.desc)
                plot_polar_xr(lut_to_xr(I),  index=ind, rect=421, sub=423, fig=fig, cmap=cmap, vmin=Imin, vmax=Imax)
            Q.desc = mdesc(Q.desc)
            U.desc = mdesc(U.desc)
            plot_polar_xr(lut_to_xr(Q),  index=ind, rect=422, sub=424, fig=fig, cmap=cmap)
            plot_polar_xr(lut_to_xr(U),  index=ind, rect=425, sub=427, fig=fig, cmap=cmap)
            if Circ:
                V.desc = mdesc(V.desc)
                plot_polar_xr(lut_to_xr(V), index=ind, rect=426, sub=428, fig=fig, cmap=cmap)
            else:
                plot_polar_xr(lut_to_xr(DoP), index=ind, rect=426, sub=428, fig=fig, vmin=Pmin, vmax=Pmax, cmap=cmap)
        else:
            # show only I and PR
            if fig is None: fig = figure(figsize=(9, 4.5))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                plot_polar_xr(lut_to_xr(lI),  index=ind, rect=221, sub=223, fig=fig, cmap=cmap, vmin=Imin, vmax=Imax)
            else:
                I.desc = mdesc(I.desc)
                plot_polar_xr(lut_to_xr(I),  index=ind, rect=221, sub=223, fig=fig, cmap=cmap, vmin=Imin, vmax=Imax)

            if Circ:
                plot_polar_xr(lut_to_xr(DoCP), index=ind, rect=222, sub=224, fig=fig, vmin=0, vmax=Pmax, cmap=cmap)
            else:
                plot_polar_xr(lut_to_xr(DoP), index=ind, rect=222, sub=224, fig=fig, vmin=Pmin, vmax=Pmax, cmap=cmap)

        return fig


    else:
        # full plots
        fig1 = figure(figsize=(16, 4))
        lI=I.apply(np.log10)
        lI.desc = mdesc(I.desc,logI=True)
        I.desc = mdesc(I.desc)
        Q.desc = mdesc(Q.desc)
        U.desc = mdesc(U.desc)
        V.desc = mdesc(V.desc)
        plot_polar_xr(lut_to_xr(I),  index=ind, rect=241, sub=245, fig=fig1, cmap=cmap, vmin=Imin, vmax=Imax)
        plot_polar_xr(lut_to_xr(Q),  index=ind, rect=242, sub=246, fig=fig1, cmap=cmap)
        plot_polar_xr(lut_to_xr(U),  index=ind, rect=243, sub=247, fig=fig1, cmap=cmap)
        plot_polar_xr(lut_to_xr(V),  index=ind, rect=244, sub=248, fig=fig1, cmap=cmap)
        
        fig2 = figure(figsize=(16, 4))
        Q.desc = mdesc(Q.desc)
        U.desc = mdesc(U.desc)
        V.desc = mdesc(V.desc)
        plot_polar_xr(lut_to_xr(lI),  index=ind, rect=241, sub=245, fig=fig2, cmap=cmap)
        plot_polar_xr(lut_to_xr(DoLP),  index=ind, rect=242, sub=246, fig=fig2, vmin=Pmin, vmax=Pmax, cmap=cmap)
        plot_polar_xr(lut_to_xr(DoCP),  index=ind, rect=243, sub=247, fig=fig2, vmin=Pmin, vmax=Pmax, cmap=cmap)
        plot_polar_xr(lut_to_xr(DoP),  index=ind, rect=244, sub=248, fig=fig2, vmin=Pmin, vmax=Pmax, cmap=cmap)
        #plot_polar_xr(lut_to_xr(AoLP),  index=ind, rect='244', sub='248', fig=fig2, vmin=-180, vmax=180, cmap=cmap)

        return fig1, fig2

def transect_view(mlut, logI=False, QU=False, Circ=False, full=False, field='up (TOA)', prefix='', ind=[0], fig=None, color='k', subdict=None, 
         **kwargs):
    '''
    visualization of a smartg MLUT

    Options:
        logI: shows log10 of I
        Circ: shows Circular polarization 
        QU:  shows Q U and DoP
        field: level of output
        ind: list of indices of azimutal planes
        full: shows all
        color: color of the transect
        subdict: dictionnary of LUT subsetter (see LUT class , sub() method)

    Outputs:
    if full is False, it returns 1 figure
    if full is True,  it returns 2 figures
    '''

    I = mlut[prefix+'I_' + field]
    Q = mlut[prefix+'Q_' + field]
    U = mlut[prefix+'U_' + field]
    V = mlut[prefix+'V_' + field]

    if subdict is not None :
        I = I.sub(d=subdict)
        Q = Q.sub(d=subdict)
        U = U.sub(d=subdict)
        V = V.sub(d=subdict)

    # Linearly polarized reflectance
    IPL = (Q*Q + U*U).apply(np.sqrt, 'Lin. Pol. ref.')
    
    # Polarized reflectance
    IP = (Q*Q + U*U +V*V).apply(np.sqrt, 'Pol. ref.')

    # Degree of Linear Polarization (%)
    DoLP = 100*IPL/I
    DoLP.desc = prefix+r'$DoLP$'
    
    # Angle of Linear Polarization (deg)
    AoLP = (U/Q)
    AoLP.apply(np.arctan)*90/np.pi
    AoLP.desc = prefix+r'$AoLP$'
    
    # Degree of Circular Polarization (%)
    DoCP = 100*V.apply(abs)/I
    DoCP.desc = prefix+r'$DoCP$'

    # Degree of Polarization (%)
    DoP = 100*IP/I
    DoP.desc = prefix+r'$DoP$'

    if isinstance(ind, Idx_base):
        ind = np.round(ind.index(mlut.axes['Azimuth angles'])).astype(int)

    if not full:
        if QU:
            if fig is None: fig = figure(figsize=(8, 8))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                transect2D_xr(lut_to_xr(lI),  index=ind, sub=221, fig=fig, color=color,  **kwargs)
            else:
                I.desc = mdesc(I.desc)
                transect2D_xr(lut_to_xr(I),  index=ind, sub=221, fig=fig, color=color,   **kwargs)
            Q.desc = mdesc(Q.desc)
            U.desc = mdesc(U.desc)
            transect2D_xr(lut_to_xr(Q),  index=ind, sub=222, fig=fig, color=color, **kwargs)
            transect2D_xr(lut_to_xr(U),  index=ind, sub=223, fig=fig, color=color, **kwargs)
            if Circ:
                V.desc = mdesc(V.desc)
                transect2D_xr(lut_to_xr(V), index=ind, sub=224, fig=fig, color=color, **kwargs)
            else:
                transect2D_xr(lut_to_xr(DoP), index=ind, sub=224, fig=fig,  color=color, percent=True, **kwargs)
        else:
            # show only I and PR
            if fig is None: fig = figure(figsize=(8, 4))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                transect2D_xr(lut_to_xr(lI),  index=ind, sub=121, fig=fig, color=color,   **kwargs)
            else:
                I.desc = mdesc(I.desc)
                transect2D_xr(lut_to_xr(I),  index=ind, sub=121, fig=fig, color=color,  **kwargs)

            if Circ:
                transect2D_xr(lut_to_xr(DoCP), index=ind, sub=122, fig=fig,  color=color, percent=True, **kwargs)
            else:
                transect2D_xr(lut_to_xr(DoP), index=ind, sub=122, fig=fig, color=color, percent=True, **kwargs)

        return fig


    else:
        # full plots
        if fig is None: 
            fig1 = figure(figsize=(16, 4))
            fig2 = figure(figsize=(16, 4))
        else : fig1,fig2 = fig
        lI=I.apply(np.log10)
        lI.desc = mdesc(I.desc,logI=True)
        I.desc = mdesc(I.desc)
        Q.desc = mdesc(Q.desc)
        U.desc = mdesc(U.desc)
        V.desc = mdesc(V.desc)
        transect2D_xr(lut_to_xr(I),  index=ind,  sub=141, fig=fig1, color=color,  **kwargs)
        transect2D_xr(lut_to_xr(Q),  index=ind,  sub=142, fig=fig1, color=color, **kwargs)
        transect2D_xr(lut_to_xr(U),  index=ind, sub=143, fig=fig1, color=color, **kwargs)
        transect2D_xr(lut_to_xr(V),  index=ind, sub=144, fig=fig1, color=color, **kwargs)
        
        Q.desc = mdesc(Q.desc)
        U.desc = mdesc(U.desc)
        V.desc = mdesc(V.desc)
        transect2D_xr(lut_to_xr(lI),  index=ind, sub=141,fig=fig2, color=color, **kwargs)
        transect2D_xr(lut_to_xr(DoLP),  index=ind, sub=142, fig=fig2, color=color, percent=True, **kwargs)
        transect2D_xr(lut_to_xr(DoCP),  index=ind, sub=143, fig=fig2, color=color, percent=True, **kwargs)
        transect2D_xr(lut_to_xr(DoP),  index=ind,  sub=144, fig=fig2, color=color, percent=True, **kwargs)
        #transect2D(AoLP, index=ind,  sub=144, fig=fig2, color=color, **kwargs)

        return fig1, fig2


def spectrum(da, vmin=None, vmax=None, sub='111', fig=None, color='k', percent=False, fmt='-'):
    """
    Plot spectrum of a 1D DataArray.

    Parameters
    ----------
    da : xr.DataArray
        One-dimensional xarray DataArray with 'wavelength' dimension.
    vmin, vmax : float, optional
        Range of values. If None (default), determined from data min/max.
    sub : str, optional
        Subplot specification. Default is '111'.
    fig : matplotlib.figure.Figure, optional
        Destination figure. If None, creates a new figure.
    color : str, optional
        Color of the plot line. Default is 'k' (black).
    percent : bool, optional
        If True, scale y-axis to 0-100%. Default is False.
    fmt : str, optional
        Line format. Default is '-'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the spectrum plot.
    """
    from pylab import figure

    if isinstance(da, object) and hasattr(da, 'names') and hasattr(da, 'axes'):
        warn_message = "\nUsing an LUT for da is deprecated, use an xarray.DataArray instead."
        warnings.warn(warn_message, DeprecationWarning)
        da = da.to_xarray()

    assert 'wavelength' in da.dims, "DataArray must have 'wavelength' dimension"

    if fig is None:
        fig = figure(figsize=(4.5, 2.5))

    ax1 = da.coords['wavelength'].values
    data = da.values

    if vmin is None:
        vmin = np.amin(data[~np.isnan(data)])
    if vmax is None:
        vmax = np.amax(data[~np.isnan(data)])
    if vmin == vmax:
        vmin -= 0.001
        vmax += 0.001
    if vmin > vmax: 
        vmin, vmax = vmax, vmin
    if percent:
        vmin = 0.
        vmax = 100.

    ax1_min = np.amin(ax1)
    ax1_max = np.amax(ax1)

    # Check if subplot already exists by using a marker attribute
    marker_name = f'_spectrum_sub_{sub}'
    ax_cart = None
    is_new_axes = True
    if hasattr(fig, marker_name):
        ax_cart = getattr(fig, marker_name)
        is_new_axes = False
    
    if is_new_axes:
        ax_cart = fig.add_subplot(sub)
        setattr(fig, marker_name, ax_cart)  # Store reference
        ax_cart.grid(True)
        ax_cart.set_xlim(ax1_min, ax1_max)
        ax_cart.set_ylim(vmin, vmax)
        ax_cart.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax_cart.set_xlabel(r'$\lambda$ (nm)')
    else:
        # Extend ylimits if needed
        current_ylim = ax_cart.get_ylim()
        new_vmin = min(current_ylim[0], vmin)
        new_vmax = max(current_ylim[1], vmax)
        ax_cart.set_ylim(new_vmin, new_vmax)

    # Plot
    ax_cart.plot(ax1, data[:], fmt, color=color)

    # Add title
    title = da.attrs.get('long_name', da.name)
    if title is not None:
        ax_cart.set_title(title)

    return fig


def spectrum_view(mlut, logI=False, QU=False, Circ=False, full=False, field='up (TOA)', prefix='', fig=None, color='k', subdict=None, 
         **kwargs):
    '''
    visualization of a smartg MLUT

    Options:
        logI: shows log10 of I
        Circ: shows Circular polarization 
        QU:  shows Q U and DoP
        field: level of output
        full: shows all
        color: color of the transect
        subdict: dictionnary of LUT subsetter (see LUT class , sub() method)

    Outputs:
    if full is False, it returns 1 figure
    if full is True,  it returns 2 figures
    '''

    I = mlut[prefix+'I_' + field]
    Q = mlut[prefix+'Q_' + field]
    U = mlut[prefix+'U_' + field]
    V = mlut[prefix+'V_' + field]

    if subdict is not None :
        I = I.sub(d=subdict)
        Q = Q.sub(d=subdict)
        U = U.sub(d=subdict)
        V = V.sub(d=subdict)

    # Linearly polarized reflectance
    IPL = (Q*Q + U*U).apply(np.sqrt, 'Lin. Pol. ref.')
    
    # Polarized reflectance
    IP = (Q*Q + U*U +V*V).apply(np.sqrt, 'Pol. ref.')

    # Degree of Linear Polarization (%)
    DoLP = 100*IPL/I
    DoLP.desc = prefix+r'$DoLP$'
    
    # Angle of Linear Polarization (deg)
    AoLP = (U/Q)
    AoLP.apply(np.arctan)*90/np.pi
    AoLP.desc = prefix+r'$AoLP$'
    
    # Degree of Circular Polarization (%)
    DoCP = 100*V.apply(abs)/I
    DoCP.desc = prefix+r'$DoCP$'

    # Degree of Polarization (%)
    DoP = 100*IP/I
    DoP.desc = prefix+r'$DoP$'

    if not full:
        if QU:
            if fig is None: fig = figure(figsize=(8, 8))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                spectrum(lI, sub=221, fig=fig, color=color,  **kwargs)
            else:
                I.desc = mdesc(I.desc)
                spectrum(I,  sub=221, fig=fig, color=color,   **kwargs)
            Q.desc = mdesc(Q.desc)
            U.desc = mdesc(U.desc)
            spectrum(Q, sub=222, fig=fig, color=color, **kwargs)
            spectrum(U, sub=223, fig=fig, color=color, **kwargs)
            if Circ:
                V.desc = mdesc(V.desc)
                spectrum(V, sub=224, fig=fig, color=color, **kwargs)
            else:
                spectrum(DoP, sub=224, fig=fig,  color=color, percent=True, **kwargs)
        else:
            # show only I and PR
            if fig is None: fig = figure(figsize=(8, 4))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                spectrum(lI, sub=121, fig=fig, color=color,   **kwargs)
            else:
                I.desc = mdesc(I.desc)
                spectrum(I, sub=121, fig=fig, color=color,  **kwargs)

            if Circ:
                spectrum(DoCP, sub=122, fig=fig,  color=color, percent=True, **kwargs)
            else:
                spectrum(DoP, sub=122, fig=fig, color=color, percent=True, **kwargs)

        return fig

    else:
        # full plots
        if fig is None: 
            fig1 = figure(figsize=(16, 4))
            fig2 = figure(figsize=(16, 4))
        else : fig1,fig2 = fig
        lI=I.apply(np.log10)
        lI.desc = mdesc(I.desc,logI=True)
        I.desc = mdesc(I.desc)
        Q.desc = mdesc(Q.desc)
        U.desc = mdesc(U.desc)
        V.desc = mdesc(V.desc)
        spectrum(I, sub=141, fig=fig1, color=color,  **kwargs)
        spectrum(Q, sub=142, fig=fig1, color=color, **kwargs)
        spectrum(U, sub=143, fig=fig1, color=color, **kwargs)
        spectrum(V, sub=144, fig=fig1, color=color, **kwargs)
        
        spectrum(lI, sub=141, fig=fig2, color=color, **kwargs)
        spectrum(DoLP, sub=142, fig=fig2, color=color, percent=True, **kwargs)
        spectrum(DoCP, sub=143, fig=fig2, color=color, percent=True, **kwargs)
        spectrum(DoP, sub=144, fig=fig2, color=color, percent=True, **kwargs)
        #spectrum(AoLP, index=ind,  sub=144, fig=fig2, color=color, **kwargs)

        return fig1, fig2
        
def phase_view(ds_sg, ipha=None, fig=None, axarr=None, iw=0, kind='atm',
               show_trunc=False, force_4stk=False):
    """
    Visualization of SMART-G phase function.

    Parameters
    ----------
    ds_sg : xr.Dataset
        An xarray Dataset from SMART-G, can be from simulation results or smartg input
        profile, containing phase function data with variables 'phase_atm' or 'phase_oc', 
        and 'OD_atm' or 'OD_oc'.
    ipha : int, optional
        Absolute index of the phase function coming from Profile.
        If None, uses all unique indices from iphase_kind.
    fig : matplotlib.figure.Figure, optional
        Figure object. If None, creates a new figure.
    axarr : numpy.ndarray, optional
        2D array of matplotlib axes. If None, creates appropriate subplot grid.
    iw : int, optional
        Wavelength index for multi-wavelength simulations. Default is 0.
    kind : {'atm', 'oc'}, optional
        Phase function type: 'atm' for atmospheric, 'oc' for oceanic. Default is 'atm'.
    show_trunc : bool, optional
        If True, also plots truncated phase function. Default is False.
    force_4stk : bool, optional
        If True, forces 2x2 subplot layout even for 6-stokes. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the phase function plots.
    axarr : numpy.ndarray
        Array of matplotlib axes.
    """

    if isinstance(ds_sg, MLUT):
        warn_message = "\nUsing an MLUT for ds_sg is deprecated, use an xarray.Dataset instead."
        warnings.warn(warn_message, DeprecationWarning)
        ds_sg = ds_sg.to_xarray()

    od_key = 'OD_'+kind
    phase_key = 'phase_'+kind
    theta_key = 'theta_'+kind
    
    # Handle multi-wavelength case
    od_data = ds_sg[od_key]
    nd = len(od_data.dims)
    
    if nd > 1:
        # Find wavelength dimension index
        if 'wavelength' in od_data.dims:
            wavelength = ds_sg.coords['wavelength'].values
            labw = r' at $%.1f nm$' % wavelength[iw]
        else:
            labw = ''
    else:
        labw = ''

    phase = ds_sg[phase_key].values
    if show_trunc:
        phase_tr = ds_sg['phase_'+kind+'_tr'].values
    
    ang = ds_sg.coords[theta_key].values
    nstk = phase.shape[1]
    
    if (axarr is None):
        if nstk == 4 or force_4stk:
            fig, axarr = subplots(2, 2)
            fig.set_size_inches(10, 6)
        elif nstk == 6:
            fig, axarr = subplots(nrows=3, ncols=2)
            fig.set_size_inches(10, 9)
        
    if ipha is None:
        iphase_key = 'iphase_'+kind
        if iphase_key in ds_sg:
            iphase_data = ds_sg[iphase_key].values
            if nd > 1:
                ni = np.unique(iphase_data)
            else:
                ni = np.unique(iphase_data)
        else:
            ni = [0]
    else:
        ni = [ipha]
    
    for i in ni:
        if nstk == 4:
            P11 = 0.5*(phase[i,0,:]+phase[i,1,:])
            P12 = 0.5*(phase[i,0,:]-phase[i,1,:])
            P33 = phase[i,2,:]
            P43 = phase[i,3,:]
            if show_trunc : 
                P11_tr = 0.5*(phase_tr[i,0,:]+phase_tr[i,1,:])
                P12_tr = 0.5*(phase_tr[i,0,:]-phase_tr[i,1,:])
                P33_tr = phase_tr[i,2,:]
                P43_tr = phase_tr[i,3,:]
        
            if (np.max(P11[:]) > 0.) :
                axarr[0,0].semilogy(ang, P11,label='%3i'%i)
                if show_trunc : axarr[0,0].semilogy(ang, P11_tr, 'k--')
            axarr[0,0].set_title(r'$P_{11}$'+labw)
            axarr[0,0].grid()
            axarr[0,0].set_xlim([0,180])
            axarr[0,0].set_xticks([0,30,60,90,120,150,180])
            
            if (np.max(P11[:]) > 0.) :
                axarr[0,1].plot(ang, -P12/P11)
                if show_trunc : axarr[0,1].plot(ang, -P12_tr/P11, 'k--')
            axarr[0,1].set_title(r'-$P_{12}/P_{11}$')
            axarr[0,1].grid()
            axarr[0,1].set_xlim([0,180])
            axarr[0,1].set_xticks([0,30,60,90,120,150,180])

            
            if (np.max(P11[:]) > 0.) :
                axarr[1,0].plot(ang, P33/P11)
                if show_trunc : axarr[1,0].plot(ang, P33_tr/P11, 'k--')
            axarr[1,0].set_title(r'$P_{33}/P_{11}$')
            axarr[1,0].grid()
            axarr[1,0].set_xlim([0,180])
            axarr[1,0].set_xlabel(r'$\theta$')
            axarr[1,0].set_xticks([0,30,60,90,120,150,180])
                    
            if (np.max(P11[:]) > 0.) :
                axarr[1,1].plot(ang, P43/P11)
                if show_trunc : axarr[1,1].plot(ang, P43_tr/P11, 'k--')
            axarr[1,1].set_title(r'$P_{43}/P_{11}$')
            axarr[1,1].grid()
            axarr[1,1].set_xlim([0,180])
            axarr[1,1].set_xlabel(r'$\theta$')
            axarr[1,1].set_xticks([0,30,60,90,120,150,180])
        elif nstk == 6:
            F0 = phase[i,0,:] # F11
            F1 = phase[i,1,:] # F12 = F21
            F2 = phase[i,2,:] # F33
            F3 = phase[i,3,:] # F34 = -F43
            F4 = phase[i,4,:] # F22
            F5 = phase[i,5,:] # F44

            P11 = 0.5*(F0+2*F1+F4)
            P12 = 0.5*(F0-F4)
            P22 = 0.5*(F0-2*F1+F4)
            P33 = F2
            P34 = F3
            P44 = F5
            if show_trunc : 
                F0_tr = phase_tr[i,0,:] # F11
                F1_tr = phase_tr[i,1,:] # F12 = F21
                F2_tr = phase_tr[i,2,:] # F33
                F3_tr = phase_tr[i,3,:] # F34 = -F43
                F4_tr = phase_tr[i,4,:] # F22
                F5_tr = phase_tr[i,5,:] # F44

                P11_tr = 0.5*(F0_tr+2*F1_tr+F4_tr)
                P12_tr = 0.5*(F0_tr-F4_tr)
                P22_tr = 0.5*(F0_tr-2*F1_tr+F4_tr)
                P33_tr = F2_tr
                P34_tr = F3_tr
                P44_tr = F5_tr
        
            if (np.max(P11[:]) > 0.) :
                axarr[0,0].semilogy(ang, P11,label='%3i'%i)
                if show_trunc : axarr[0,0].semilogy(ang, P11_tr, 'k--')
            axarr[0,0].set_title(r'$P_{11}$'+labw)
            axarr[0,0].grid()
            axarr[0,0].set_xlim([0,180])
            axarr[0,0].set_xticks([0,30,60,90,120,150,180])
            
            if (np.max(P11[:]) > 0.) :
                axarr[0,1].plot(ang, -P12/P11)
                if show_trunc : axarr[0,1].plot(ang, -P12_tr/P11, 'k--')
            axarr[0,1].set_title(r'-$P_{12}/P_{11}$')
            axarr[0,1].grid()
            axarr[0,1].set_xlim([0,180])
            axarr[0,1].set_xticks([0,30,60,90,120,150,180])

            
            if (np.max(P11[:]) > 0.) :
                axarr[1,0].plot(ang, P33/P11)
                if show_trunc : axarr[1,0].plot(ang, P33_tr/P11, 'k--')
            axarr[1,0].set_title(r'$P_{33}/P_{11}$')
            axarr[1,0].grid()
            axarr[1,0].set_xlim([0,180])
            axarr[1,0].set_xticks([0,30,60,90,120,150,180])
            if force_4stk:
                axarr[1,0].set_xlabel(r'$\theta$')
                
                    
            if (np.max(P11[:]) > 0.) :
                axarr[1,1].plot(ang, P34/P11)
                if show_trunc : axarr[1,1].plot(ang, P34_tr/P11, 'k--')
            axarr[1,1].set_title(r'$P_{34}/P_{11}$')
            axarr[1,1].grid()
            axarr[1,1].set_xlim([0,180])
            axarr[1,1].set_xticks([0,30,60,90,120,150,180])
            if force_4stk:
                axarr[1,1].set_xlabel(r'$\theta$')
               

            if not force_4stk:
                if (np.max(P11[:]) > 0.) :
                    axarr[2,0].plot(ang, P22/P11)
                    if show_trunc : axarr[2,0].plot(ang, P22_tr/P11, 'k--')
                axarr[2,0].set_title(r'$P_{22}/P_{11}$')
                axarr[2,0].grid()
                axarr[2,0].set_xlim([0,180])
                axarr[2,0].set_xlabel(r'$\theta$')
                axarr[2,0].set_xticks([0,30,60,90,120,150,180])
                        
                if (np.max(P11[:]) > 0.) :
                    axarr[2,1].plot(ang, P44/P11)
                    if show_trunc : axarr[2,1].plot(ang, P44_tr/P11, 'k--')
                axarr[2,1].set_title(r'$P_{44}/P_{11}$')
                axarr[2,1].grid()
                axarr[2,1].set_xlim([0,180])
                axarr[2,1].set_xlabel(r'$\theta$')
                axarr[2,1].set_xticks([0,30,60,90,120,150,180])
                setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
    
    setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    axarr[0,0].legend(loc='upper center',fontsize = 'medium',labelspacing=0.01)

    return fig, axarr

    
def profile_view(ds_sg, fig=None, ax=None, iw=0, kind='atm', zmax=None):
    """
    Visualization of SMART-G vertical profile.

    Parameters
    ----------
    ds_sg : xr.Dataset
        An xarray Dataset from SMART-G, can be from simulation results or smartg input profile,
        containing optical depth and other profile data with variables 'OD_atm' or 'OD_oc', and 
        related optical properties.
    fig : matplotlib.figure.Figure, optional
        Figure object. If None, creates a new figure.
    ax : matplotlib.axes.Axes, optional
        Axes object. If None, creates a new axes.
    iw : int, optional
        Wavelength index for multi-wavelength simulations. Default is 0.
    kind : {'atm', 'oc'}, optional
        Profile type: 'atm' for atmospheric, 'oc' for oceanic. Default is 'atm'.
    zmax : float, optional
        Maximum altitude (for 'atm') or depth (for 'oc') to plot. 
        If None, automatically determined from data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the profile plot.
    ax : matplotlib.axes.Axes
        Axes object containing the profile plot.
    """

    if isinstance(ds_sg, MLUT):
        warn_message = "\nUsing an MLUT for ds_sg is deprecated, use an xarray.Dataset instead."
        warnings.warn(warn_message, DeprecationWarning)
        ds_sg = ds_sg.to_xarray()

    if (ax is None):
        fig, ax = subplots(1, 1)
        fig.set_size_inches(5, 5)
    
    od_key = 'OD_'+kind
    z_key = 'z_'+kind
    
    od_data = ds_sg[od_key]
    nd = len(od_data.dims)
    
    # Handle multi-wavelength case
    labw = ''
    if nd > 1 and 'wavelength' in od_data.dims:
        wavelength = ds_sg.coords['wavelength'].values
        labw = r' at $%.1f nm$' % wavelength[iw]

    z = ds_sg.coords[z_key].values
    if kind == 'oc': 
        sign = -1.
        func = diff2
    else:
        sign = 1.    
        func = diff1
    
    Dz = np.abs(func(z))
    
    # Extract and compute optical depths
    Dtau = sign * func(ds_sg[od_key].values)
    Dtau_Sca = sign * func(ds_sg['OD_sca_'+kind].values)
    Dtau_Abs = sign * func(ds_sg['OD_abs_'+kind].values)
    if kind == 'atm':
        Dtau_ExtA = sign * func(ds_sg['OD_p'].values)
        Dtau_ScaR = sign * func(ds_sg['OD_r'].values)
        Dtau_AbsG = sign * func(ds_sg['OD_g'].values)
        ssa_p = ds_sg['ssa_p_'+kind].values
        Dtau_ScaA = Dtau_ExtA * ssa_p
        Dtau_AbsA = Dtau_ExtA * (1. - ssa_p)
        if (np.max(Dtau_AbsA) > 0.) : ax.semilogx((Dtau_AbsA/Dz), z, 'r--',label=r'$\sigma_{abs}^{a+c}$')
        if (np.max(Dtau_ScaA) > 0.) : ax.semilogx((Dtau_ScaA/Dz), z, 'r',  label=r'$\sigma_{sca}^{a+c}$')
        if (np.max(Dtau_AbsG) > 0.) : ax.semilogx((Dtau_AbsG/Dz), z, 'g--',  label=r'$\sigma_{abs}^{gas}$')
        ax.semilogx((Dtau_ScaR/Dz), z, 'b', label=r'$\sigma_{sca}^{R}$' )
        ax.set_xlim(1e-6,10)
        xlabel('Vertical profile'+labw + r' $(km^{-1})$')
        ylabel(r'$z (km)$')
        if zmax is None : zmax = max(100., z.max())
        ax.set_ylim(0, zmax)
    else :
        Dtau_ExtP = sign * func(ds_sg['OD_p_oc'].values)
        Dtau_ExtW = sign * func(ds_sg['OD_w'].values)
        Dtau_AbsY = sign * func(ds_sg['OD_y'].values)
        ssa_p = ds_sg['ssa_p_'+kind].values
        ssa_w = ds_sg['ssa_w'].values
        pine  = ds_sg['pine_oc'].values
        Dtau_ScaP = Dtau_ExtP * ssa_p
        Dtau_AbsP = Dtau_ExtP * (1. - ssa_p)
        Dtau_ScaW = Dtau_ExtW * ssa_w
        Dtau_AbsW = Dtau_ExtW * (1. - ssa_w)
        Dtau_Ine  = Dtau_Sca  * pine
        if (np.max(Dtau_AbsP) > 0.) : ax.semilogx((Dtau_AbsP/Dz), z, 'r--',label=r'$\sigma_{abs}^{p}$')
        if (np.max(Dtau_ScaP) > 0.) : ax.semilogx((Dtau_ScaP/Dz), z, 'r',  label=r'$\sigma_{sca}^{p}$')
        if (np.max(Dtau_AbsW) > 0.) : ax.semilogx((Dtau_AbsW/Dz), z, 'b--',label=r'$\sigma_{abs}^{w}$')
        if (np.max(Dtau_ScaW) > 0.) : ax.semilogx((Dtau_ScaW/Dz), z, 'b',  label=r'$\sigma_{sca}^{w}$')
        if (np.max(Dtau_AbsY) > 0.) : ax.semilogx((Dtau_AbsY/Dz), z, 'y--',label=r'$\sigma_{abs}^{y}$')
        if (np.max(Dtau_Ine) > 0.) : ax.semilogx((Dtau_Ine/Dz), z, 'm:' ,label=r'$\sigma_{ine}^{}$')
        ax.set_xlim(1e-4,10)
        xlabel('Vertical profile'+labw + r' $(m^{-1})$')
        ylabel(r'$z (m)$')
        if zmax is None : zmax = min(-100., z.min())
        ax.set_ylim(zmax, 0)
    ax.semilogx((Dtau/Dz), z, 'k.-', label=r'$\sigma_{ext}^{tot}$')
    ax.semilogx((Dtau_Abs/Dz), z, 'k.--', label=r'$\sigma_{abs}^{tot}$')
    #ax.set_title('Vertical profile'+labw)
    ax.grid()
    ax.legend()

    try :
        ax2 = ax.twiny()
        nf = ds_sg['iphase_'+kind].values
        z_vals = ds_sg.coords[z_key].values
        ax2.plot(nf[1:], z_vals[1:], 'm-', drawstyle='steps-post', label='i')
        ax2.set_xlabel('Phase Matrix index', color='m')
        ax2.tick_params('x', colors='m')
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%i'))
        return fig, ax
    
    except:
        return fig, ax
    
    
def input_view(ds_sg, iw=0, kind='atm', zmax=None, ipha=None):
    """
    Visualization of SMART-G input profile and phase functions.

    Parameters
    ----------
    ds_sg : xr.Dataset
        An xarray Dataset from SMART-G, can be from simulation results or smartg input profile,
        containing phase function data and optical depth profiles.
    iw : int, optional
        Wavelength index for multi-wavelength simulations. Default is 0.
    kind : {'atm', 'oc'}, optional
        Profile type: 'atm' for atmospheric, 'oc' for oceanic. Default is 'atm'.
    zmax : float, optional
        Maximum altitude (for 'atm') or depth (for 'oc') to plot.
        If None, automatically determined from data.
    ipha : int, optional
        Absolute index of the phase function coming from Profile.
        If None, uses all unique indices.
    """

    if isinstance(ds_sg, MLUT):
        warn_message = "\nUsing an MLUT for ds_sg is deprecated, use an xarray.Dataset instead."
        warnings.warn(warn_message, DeprecationWarning)
        ds_sg = ds_sg.to_xarray()

    if 'phase_'+kind in ds_sg:
        fig = figure()
        phase_data = ds_sg['phase_'+kind].values
        nstk = phase_data.shape[1]
        if nstk == 4:
            fig.set_size_inches(12, 6)
            ax1 = subplot2grid((2, 3), (0, 0))
            ax2 = subplot2grid((2, 3), (0, 1))
            ax3 = subplot2grid((2, 3), (1, 0))
            ax4 = subplot2grid((2, 3), (1, 1))
        
            axarr = np.array([[ax1, ax2], [ax3, ax4]])
        
            _,_ = phase_view(ds_sg, iw=iw, axarr=axarr, kind=kind, ipha=ipha)
            
            ax5 = subplot2grid((2, 3), (0, 2), rowspan=2, colspan=1)
            
            profile_view(ds_sg, iw=iw, ax=ax5, kind=kind, zmax=zmax)
        else:
            fig.set_size_inches(12, 9)
            ax1 = subplot2grid((3, 3), (0, 0))
            ax2 = subplot2grid((3, 3), (0, 1))
            ax3 = subplot2grid((3, 3), (1, 0))
            ax4 = subplot2grid((3, 3), (1, 1))
            ax5 = subplot2grid((3, 3), (2, 0))
            ax6 = subplot2grid((3, 3), (2, 1))
        
            axarr = np.array([[ax1, ax2], [ax3, ax4], [ax5, ax6]])
        
            _,_ = phase_view(ds_sg, iw=iw, axarr=axarr, kind=kind, ipha=ipha)
            
            ax7 = subplot2grid((3, 3), (0, 2), rowspan=2, colspan=1)
            
            profile_view(ds_sg, iw=iw, ax=ax7, kind=kind, zmax=zmax)
    else:
        fig, _ = profile_view(ds_sg, iw=iw, kind=kind, zmax=zmax)

    tight_layout()


def compare(mlut, mref, field='up (TOA)',errb=False, logI=False, U_sign=1, same_U_convention=True, U_symetry=True,
                  Nparam=4, vmax=None, vmin=None, emax=None, ermax=None, same_azimuth_convention=True,
                  azimuth=[0.,90.], title='', SZA_MAX=89., zenith_title=r'$SZA (°)$', errref=None):
    '''
    compare the results of two smartg runs : mlut vs mref in two different azimuth planes
    outputs: a figure
    keywords:
        field : name of the output level to be compared 
        errb  : error bar visible for mlut : should have been run with the stdev option
        LogI  : plot Intensity in log scale
        U_sign: change sign for U
        same_U_convention: mlut and mref have the same convention for U
        U_symetry:  U changes sign convention for the  two halves of the plane
        Nparam: number of parameters :  by defaut 4 for I,Q,U, DoLP; 5 adds V:, 2 keeps only I and DoLP 
        vmin,vmax: min and max values for parameters: list of length Nparam
        emax: max absolute error scale : length Nparam
        ermax: max relative error scale (in percent) : length Nparam
        same_azimuth_convention: mlut and mref have the same azimuth convention
        azimuth: list of azimuths
        title: plot title
        SZA_MAX: SZA max for the comparison
        errref : eventually intensity absolute error on refence points
    '''

    from pylab import subplots
    if vmax is None : vmax=[0.1]*Nparam 
    if vmin is None : vmin=[-0.1]*Nparam 
    if emax is None : emax=[0.1]*Nparam
    if ermax is None : ermax=[0.1]*Nparam
    stokesT = ['I','Q','U','V']
    stokes=stokesT[:Nparam-1]
    signT = [1,1,U_sign*1,1,1] # sign convention for both mluts
    sign=signT[:Nparam-1]+[1]
    if same_U_convention: diffsignT = [1,1,1,1,1]    # sign convention difference
    else: diffsignT = [1,1,-1,1,1]
    diffsign=diffsignT[:Nparam-1]+[1]
    if U_symetry: symetryT=[1,1,1,1,1]
    else: symetryT=[1,1,-1,1,1]
    symetry=symetryT[:Nparam-1]+[1]
    fig,ax = subplots(3,Nparam, sharey=False,sharex=True,gridspec_kw=dict(hspace=0.2,wspace=0.3))
    fig.set_size_inches(Nparam*3,8)
    fig.set_dpi=600
    fig.suptitle(title)
    
    for i in range(Nparam):
        if i!=Nparam-1 :
            S = mlut[stokes[i] + '_' + field]
            if S.names.index('Azimuth angles') == 0: th = S.axes[1]   
            else: th = S.axes[0]
            Sref = mref[stokes[i] + '_' + field]
            S.desc = mdesc(S.desc)
            if errb : E = mlut[stokes[i] + '_' + 'stdev' + '_' + field]
            if logI and stokes[i]=='I':
                S=S.apply(np.log10)
                Sref=Sref.apply(np.log10)
                S.desc=r'$log_{10}$ '+S.desc
        else:
            I=mlut['I' + '_' + field]
            Q=mlut['Q' + '_' + field]
            U=mlut['U' + '_' + field]
            Ip= ((Q*Q+U*U).apply(np.sqrt))
            S= (Ip/I) * 100
            S.desc= r'$DoLP' + I.desc[1:3] + I.desc[4:] + '$'
            S.desc= 'DoLP' + I.desc[1:]
            S.desc = mdesc(S.desc)
            
            if errb: 
                dI=mlut['I' + '_' + 'stdev' + '_' + field]
                dQ=mlut['Q' + '_' + 'stdev' + '_' + field]
                dU=mlut['U' + '_' + 'stdev' + '_' + field]
                dIp= ((dQ*dQ+dU*dU).apply(np.sqrt))
                E = (dI/I + dIp/Ip) * S
            Iref=mref['I' + '_' + field]
            Qref=mref['Q' + '_' + field]
            Uref=mref['U' + '_' + field]
            Sref= (((Qref*Qref+Uref*Uref).apply(np.sqrt))/Iref) * 100           
     
        vmi=vmin[i]
        vma=vmax[i]
        ema=emax[i]
        erma=ermax[i]

        for phi0,sym1,sym2,labref in [(azimuth[0],'r','-','ref.'),(azimuth[1],'g','-','')]:
        #for phi0,sym1,sym2,labref in [(azimuth[0],'r','.','ref.'),(azimuth[1],'g','.','')]:

            # both points at their own abscissas
            if same_azimuth_convention:

                if S.names.index('Azimuth angles') == 0: # check right order of axes
                    refp = sign[i]*Sref[Idx(phi0,round=True),:] # reference for >0 view angle
                    refm = sign[i]*Sref[Idx(180.-phi0,round=True),:] #      reference for <0 view angle
                    sp   = diffsign[i]*sign[i]*S[Idx(phi0),:]       #     simulation for >0 view angle
                    sm   = symetry[i]*diffsign[i]*sign[i]*S[Idx(180-phi0),:]
                    if errb:
                        dsp  = E[Idx(phi0),:]         #     simulation error for >0 view angle
                        dsm  = E[Idx(180.-phi0),:]
                    else:
                        (dsp,dsm) = (0,0)
                else:
                    refm = sign[i]*Sref.swapaxes(0,1)[Idx(phi0,round=True),:] # reference for >0 view angle
                    refp = sign[i]*Sref.swapaxes(0,1)[Idx(180.-phi0,round=True),:] #      reference for <0 view angle
                    sp   = diffsign[i]*sign[i]*S.swapaxes(0,1)[Idx(phi0),:]       #     simulation for >0 view angle
                    sm   = symetry[i]*diffsign[i]*sign[i]*S.swapaxes(0,1)[Idx(180-phi0),:]
                    if errb:
                        dsp  = E.swapaxes(0,1)[Idx(phi0),:]         #     simulation error for >0 view angle
                        dsm  = E.swapaxes(0,1)[Idx(180.-phi0),:]
                    else:
                        (dsp,dsm) = (0,0)
                        
            else:
                refp = sign[i]*Sref[Idx(180.-phi0,round=True),:] # reference for >0 view angle
                refm = sign[i]*Sref[Idx(phi0,round=True),:] #      reference for <0 view angle
                sp   = diffsign[i]*sign[i]*S[Idx(phi0),:]       #     simulation for >0 view angle
                sm   = symetry[i]*diffsign[i]*sign[i]*S[Idx(180-phi0),:]
                if errb:
                    dsp  = E[Idx(phi0),:]         #     simulation error for >0 view angle
                    dsm  = E[Idx(180-phi0),:]
                else:
                    (dsp,dsm) = (0,0)
                    
            ax[0,i].plot(th, refp,'k'+'.')
            ax[0,i].plot(-th,refm,'k'+'.',label=labref)
            ax[0,i].errorbar(th, sp, fmt=sym1+'')
            ax[0,i].errorbar(-th,sm, fmt=sym1+'', \
                        label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0))
            ax[0,i].set_ylim([vmi, vma])
            ax[0,i].set_xlim([-SZA_MAX, SZA_MAX])
            ax[0,i].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
            
            if logI and i==0:
                if errb:
                    ax[1,i].errorbar(th,10**sp-10**refp, yerr=dsp,\
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor='k',capsize=2)
                    ax[1,i].errorbar(-th,10**sm-10**refm,yerr=dsm,fmt=sym1+sym2,ecolor='k',capsize=2) 
                else:
                    ax[1,i].errorbar(th,10**sp-10**refp, \
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor='k',capsize=2)
                    ax[1,i].errorbar(-th,10**sm-10**refm,fmt=sym1+sym2,ecolor='k',capsize=2) 
    
            else:
                if errb:
                    ax[1,i].errorbar(th,sp-refp, yerr=dsp,\
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor=sym1,capsize=2)
                    ax[1,i].errorbar(-th,sm-refm,yerr=dsm,fmt=sym1+sym2,ecolor=sym1,capsize=2) 
                else:
                    ax[1,i].errorbar(th,sp-refp, \
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor=sym1,capsize=2)
                    ax[1,i].errorbar(-th,sm-refm,fmt=sym1+sym2,ecolor=sym1,capsize=2) 
            ax[1,i].set_ylim([-1*ema,ema])
            ax[1,i].set_xlim([-SZA_MAX,SZA_MAX])  

            if errb:
                ax[2,i].errorbar(th,(sp-refp)/refp*100, yerr=dsp/abs(refp)*100, \
                             fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor=sym1,capsize=2)
                ax[2,i].errorbar(-th,(sm-refm)/refm*100, yerr= dsm/abs(refm)*100,fmt=sym1+sym2,ecolor=sym1,capsize=2)  
                if (i==0 and errref is not None):
                    ax[2,0].plot(th,errref/refp*100,sym1+'-.')
                    ax[2,0].plot(th,-errref/refp*100,sym1+'-.')
                    ax[2,0].plot(-th,errref/refm*100,sym1+'-.')
                    ax[2,0].plot(-th,-errref/refm*100,sym1+'-.')
            else:
                ax[2,i].errorbar(th,(sp-refp)/refp*100,\
                             fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor='k',capsize=2)
                ax[2,i].errorbar(-th,(sm-refm)/refm*100,fmt=sym1+sym2,ecolor='k',capsize=2)  
            
            if i!=Nparam-1 : ax[2,i].set_ylim([-1*erma,erma])
            else : ax[2,i].set_ylim([-1*erma,erma])
                
            ax[2,i].set_xlim([-SZA_MAX, SZA_MAX])    
            ax[1,i].plot([-SZA_MAX,SZA_MAX],[0.,0.],'k--')
            ax[2,i].plot([-SZA_MAX,SZA_MAX],[0.,0.],'k--')
            ax[1,i].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

            ax[0,i].set_title(S.desc)   
            if i==0: 
 
                ax[0,i].legend(loc='upper center',fontsize = 8,labelspacing=0.0)
                #ax[1,i].text(-50.,ema*0.75,r'$N_{\Phi}$:%i, $N_{\theta}$:%i'%\
                #         (S.axes[0].shape[0],S.axes[1].shape[0]))
                ax[1,i].set_ylabel(r'$\Delta$')
                ax[2,i].set_ylabel(r'$\Delta (\%)$')
            ax[2,i].set_xlabel(zenith_title)
    return fig


def compare_spectrum(mlut, mref, field='up (TOA)',errb=False, logI=False, U_sign=1, same_U_convention=True, U_symetry=True,
                  Nparam=4, vmax=None, vmin=None, emax=None, ermax=None, same_azimuth_convention=True,
                  azimuth=[0.,90.], title='', LMIN=None, LMAX=None, lambda_title=r'$\lambda$ (nm)', errref=None):
    '''
    compare the results of two smartg runs : mlut vs mref in two different azimuth planes
    outputs: a figure
    keywords:
        field : name of the output level to be compared 
        errb  : error bar visible for mlut : should have been run with the stdev option
        LogI  : plot Intensity in log scale
        U_sign: change sign for U
        same_U_convention: mlut and mref have the same convention for U
        U_symetry:  U changes sign convention for the  two halves of the plane
        Nparam: number of parameters :  by defaut 4 for I,Q,U, DoLP; 5 adds V:, 2 keeps only I and DoLP 
        vmin,vmax: min and max values for parameters: list of length Nparam
        emax: max absolute error scale : length Nparam
        ermax: max relative error scale (in percent) : length Nparam
        same_azimuth_convention: mlut and mref have the same azimuth convention
        azimuth: list of azimuths
        title: plot title
        SZA_MAX: SZA max for the comparison
        errref : eventually intensity absolute error on refence points
    '''

    from pylab import subplots
    if vmax is None : vmax=[0.1]*Nparam 
    if vmin is None : vmin=[-0.1]*Nparam 
    if emax is None : emax=[0.1]*Nparam
    if ermax is None : ermax=[0.1]*Nparam
    stokesT = ['I','Q','U','V']
    stokes=stokesT[:Nparam-1]
    signT = [1,1,U_sign*1,1,1] # sign convention for both mluts
    sign=signT[:Nparam-1]+[1]
    if same_U_convention: diffsignT = [1,1,1,1,1]    # sign convention difference
    else: diffsignT = [1,1,-1,1,1]
    diffsign=diffsignT[:Nparam-1]+[1]
    if U_symetry: symetryT=[1,1,1,1,1]
    else: symetryT=[1,1,-1,1,1]
    symetry=symetryT[:Nparam-1]+[1]
    fig,ax = subplots(3,Nparam, sharey=False,sharex=True,gridspec_kw=dict(hspace=0.2,wspace=0.3))
    fig.set_size_inches(Nparam*3,8)
    fig.set_dpi=600
    fig.suptitle(title)
    
    for i in range(Nparam):
        if i!=Nparam-1 :
            S = mlut[stokes[i] + '_' + field]
            wav = S.axis('wavelength')   
            Sref = mref[stokes[i] + '_' + field]
            S.desc = mdesc(S.desc)
            if errb : E = mlut[stokes[i] + '_' + 'stdev' + '_' + field]
            if logI and stokes[i]=='I':
                S=S.apply(np.log10)
                Sref=Sref.apply(np.log10)
                S.desc=r'$log_{10}$ '+S.desc
        else:
            I=mlut['I' + '_' + field]
            Q=mlut['Q' + '_' + field]
            U=mlut['U' + '_' + field]
            Ip= ((Q*Q+U*U).apply(np.sqrt))
            S= (Ip/I) * 100
            S.desc= r'$DoLP' + I.desc[1:3] + I.desc[4:] + '$'
            S.desc= 'DoLP' + I.desc[1:]
            S.desc = mdesc(S.desc)
            
            if errb: 
                dI=mlut['I' + '_' + 'stdev' + '_' + field]
                dQ=mlut['Q' + '_' + 'stdev' + '_' + field]
                dU=mlut['U' + '_' + 'stdev' + '_' + field]
                dIp= ((dQ*dQ+dU*dU).apply(np.sqrt))
                E = (dI/I + dIp/Ip) * S
            Iref=mref['I' + '_' + field]
            Qref=mref['Q' + '_' + field]
            Uref=mref['U' + '_' + field]
            Sref= (((Qref*Qref+Uref*Uref).apply(np.sqrt))/Iref) * 100           
     
        if LMIN is None:
            LMIN=wav.min()
        if LMAX is None:
            LMAX=wav.max()
        vmi=vmin[i]
        vma=vmax[i]
        ema=emax[i]
        erma=ermax[i]

        sym1s = ['r', 'g','b','m']
        sym2s = ['-','-','-','-']
        labrefs=['ref.','','','']
        for k in range(len(azimuth)):
            phi0 = azimuth[k]
            sym1 = sym1s[k]
            sym2 = sym2s[k]
            labref = labrefs[k]

            # both points at their own abscissas
            if same_azimuth_convention:
                if S.names.index('Azimuth angles') == 0: # check right order of axes
                    refp = sign[i]*Sref[Idx(phi0,round=True),:] # reference for >0 view angle
                    sp   = diffsign[i]*sign[i]*S[Idx(phi0),:]       #     simulation for >0 view angle
                    if errb:
                        dsp  = E[Idx(phi0),:]         #     simulation error for >0 view angle
                    else:
                        dsp = 0
                else:
                    refp = sign[i]*Sref.swapaxes(0,1)[Idx(phi0,round=True),:] #      reference for <0 view angle
                    sp   = diffsign[i]*sign[i]*S.swapaxes(0,1)[Idx(phi0),:]       #     simulation for >0 view angle
                    if errb:
                        dsp  = E.swapaxes(0,1)[Idx(phi0),:]         #     simulation error for >0 view angle
                    else:
                        dsp = 0
                        
            else:
                refp = sign[i]*Sref[Idx(180.-phi0,round=True),:] # reference for >0 view angle
                sp   = diffsign[i]*sign[i]*S[Idx(phi0),:]       #     simulation for >0 view angle
                if errb:
                    dsp  = E[Idx(phi0),:]         #     simulation error for >0 view angle
                else:
                    dsp = 0
                    
            ax[0,i].plot(wav,refp,'k'+'.',label=labref)
            ax[0,i].errorbar(wav,sp, fmt=sym1+'', \
                        label=r'$\Phi=%.0f$'%(phi0))
            ax[0,i].set_ylim([vmi, vma])
            ax[0,i].set_xlim([LMIN,LMAX])  
            ax[0,i].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
            
            if logI and i==0:
                if errb:
                    ax[1,i].errorbar(wav,10**sp-10**refp, yerr=dsp,\
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f$'%(phi0),ecolor='k',capsize=2)
                else:
                    ax[1,i].errorbar(wav,10**sp-10**refp, \
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f$'%(phi0),ecolor='k',capsize=2)
    
            else:
                if errb:
                    ax[1,i].errorbar(wav,sp-refp, yerr=dsp,\
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f$'%(phi0),ecolor=sym1,capsize=2)
                else:
                    ax[1,i].errorbar(wav,sp-refp, \
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f$'%(phi0),ecolor=sym1,capsize=2)
            ax[1,i].set_ylim([-1*ema,ema])
            ax[1,i].set_xlim([LMIN,LMAX])  

            if errb:
                ax[2,i].errorbar(wav,(sp-refp)/refp*100, yerr=dsp/abs(refp)*100, \
                             fmt=sym1+sym2,label=r'$\Phi=%.0f$'%(phi0),ecolor=sym1,capsize=2)
                if (i==0 and errref is not None):
                    ax[2,0].plot(wav,errref/refp*100,sym1+'-.')
                    ax[2,0].plot(wav,-errref/refp*100,sym1+'-.')
            else:
                ax[2,i].errorbar(wav,(sp-refp)/refp*100,\
                             fmt=sym1+sym2,label=r'$\Phi=%.0f$'%(phi0),ecolor='k',capsize=2)
            
            if i!=Nparam-1 : ax[2,i].set_ylim([-1*erma,erma])
            else : ax[2,i].set_ylim([-1*erma,erma])
                
            ax[2,i].set_xlim([LMIN, LMAX])    
            ax[1,i].plot([LMIN,LMAX],[0.,0.],'k--')
            ax[2,i].plot([LMIN,LMAX],[0.,0.],'k--')
            ax[1,i].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

            ax[0,i].set_title(S.desc)   
            if i==0: 
 
                ax[0,i].legend(loc='upper center',fontsize = 8,labelspacing=0.0)
                ax[1,i].set_ylabel(r'$\Delta$')
                ax[2,i].set_ylabel(r'$\Delta (\%)$')
            ax[2,i].set_xlabel(lambda_title)
    return fig


# ============================================================================
# xarray-compatible versions of plotting functions
# ============================================================================

def bin_edges(x, min=None, max=None):
    """Helper function to compute bin edges from bin centers"""
    edges = np.zeros(len(x) + 1)
    edges[1:-1] = (x[1:] + x[:-1]) / 2.0
    edges[0] = 2 * x[0] - edges[1]
    edges[-1] = 2 * x[-1] - edges[-2]
    if min is not None:
        edges[0] = min
    if max is not None:
        edges[-1] = max
    return edges


def plot_polar_xr(da, index=None, vmin=None, vmax=None, rect=211, sub=212,
                  sym=True, swap='auto', fig=None, cmap=None, semi=False):
    """
    Contour and optionally transect of 2D DataArray on a semi-polar plot.
    
    xarray version of luts.plot_polar, compatible with xr.DataArray objects.

    Parameters
    ----------
    da : xr.DataArray
        2D data array with dimensions (angle, radius) or similar
        Angle is assumed to be in degrees and is not scaled
    index : int, array, or list, optional
        Index/indices of the item to transect in the first dimension
        If None (default), no transect
    vmin, vmax : float, optional
        Range of values. If None, determined from data
    rect : int
        Subplot position of the main plot (111 for example)
    sub : int
        Subplot position of the transect
    sym : bool
        If True, the transect uses symmetrical axis
    swap : bool or 'auto'
        If True or 'auto', swap the order of the 2 axes
        If 'auto', searches for 'azi' in both dimension names
    fig : matplotlib.figure.Figure, optional
        Destination figure. If None, create a new figure
    cmap : matplotlib.cm.Colormap, optional
        Color map to use
    semi : bool
        If True, use semi-polar (180 deg), otherwise polar (360 deg)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    
    # Initialization
    Phimax = 360.
    if semi:
        Phimax = 180.

    assert da.ndim == 2, "DataArray must be 2D"

    show_sub = index is not None
    if fig is None:
        if show_sub:
            fig = figure(figsize=(4.5, 4.5))
        else:
            fig = figure(figsize=(4.5, 6))

    # Get dimension names
    dim_names = list(da.dims)
    dim0_name, dim1_name = dim_names[0], dim_names[1]

    # Determine if we need to swap axes
    if swap == 'auto':
        if ('azi' in dim1_name.lower()) and ('azi' not in dim0_name.lower()):
            swap = True
        else:
            swap = False

    # Get axes values and data
    if swap:
        ax1_name, ax2_name = dim1_name, dim0_name
        ax1 = da.coords[dim1_name].values
        ax2 = da.coords[dim0_name].values
        data = da.values.T  # Transpose to get (angle, radius)
    else:
        ax1_name, ax2_name = dim0_name, dim1_name
        ax1 = da.coords[dim0_name].values
        ax2 = da.coords[dim1_name].values
        data = da.values

    # Determine axis labels
    label1 = da.coords[ax1_name].attrs.get('long_name', ax1_name)
    label2 = da.coords[ax2_name].attrs.get('long_name', ax2_name)

    # Determine min/max values
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    if vmin == vmax:
        vmin -= 0.001
        vmax += 0.001
    if vmin > vmax:
        vmin, vmax = vmax, vmin

    # Semi-polar axis setup
    ax1_scaled = ax1
    ax2_min = np.amin(ax2)
    ax2_max = np.amax(ax2)
    ax2_scaled = (ax2 - ax2_min) / (ax2_max - ax2_min) * 90.

    # Setup angle and radius axis locators/formatters
    grid_locator1 = angle_helper.LocatorDMS({True: 4, False: 8}[semi], include_last=False)
    tick_formatter1 = angle_helper.FormatterDMS()

    class Locator(object):
        def __call__(self, *args):
            return [np.array([0, 30, 60, 90]), 4, 1.0]

    class Formatter(object):
        def __call__(self, *args):
            return list(map(lambda x: '{:.3g}'.format(x), np.linspace(ax2_min, ax2_max, 4)))

    # Radius axis locator/formatter
    if ((ax2_min < 10.) and (ax2_min >= 0)
            and (ax2_max <= 90) and (ax2_max > 80)):
        grid_locator2 = angle_helper.LocatorDMS(4)
        tick_formatter2 = angle_helper.FormatterDMS()
    else:
        grid_locator2 = Locator()
        tick_formatter2 = Formatter()

    # Setup transform
    tr_rotate = Affine2D().translate(0, 0)
    tr_scale = Affine2D().scale(np.pi / 180., 1.)
    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

    # Create grid helper and floating subplot
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr,
        extremes=(0., Phimax, 0., 90.),
        grid_locator1=grid_locator1,
        grid_locator2=grid_locator2,
        tick_formatter1=tick_formatter1,
        tick_formatter2=tick_formatter2,
    )

    ax_polar = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax_polar)

    # Adjust polar axis
    ax_polar.grid(True)
    ax_polar.axis["left"].set_axis_direction("bottom")
    ax_polar.axis["right"].set_axis_direction("top")
    ax_polar.axis["bottom"].set_visible(False)
    ax_polar.axis["top"].set_axis_direction("bottom")
    ax_polar.axis["top"].toggle(ticklabels=True, label=True)
    ax_polar.axis["top"].major_ticklabels.set_axis_direction("top")
    ax_polar.axis["top"].label.set_axis_direction("top")

    ax_polar.axis["top"].axes.text(0.72, 0.98, label1,
                                    transform=ax_polar.transAxes,
                                    ha='left', va='bottom')
    ax_polar.axis["left"].axes.text(0.10, -0.03, label2,
                                    transform=ax_polar.transAxes,
                                    ha='center', va='top')

    # Create auxiliary polar axes
    aux_ax_polar = ax_polar.get_aux_axes(tr)
    aux_ax_polar.patch = ax_polar.patch
    ax_polar.patch.zorder = 0.9

    # Initialize cartesian axis for transect
    if show_sub:
        ax_cart = fig.add_subplot(sub)
        if sym:
            ax_cart.set_xlim(-ax2_max, ax2_max)
        else:
            ax_cart.set_xlim(ax2_min, ax2_max)
        ax_cart.set_ylim(vmin, vmax)
        ax_cart.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax_cart.grid(True)

    # Setup colormap
    if cmap is None:
        cmap = cm.rainbow.copy()
        cmap.set_under('black')
        cmap.set_over('white')
        cmap.set_bad('0.5')

    # Draw colormesh
    r, t = np.meshgrid(bin_edges(ax2_scaled, min=0, max=90), bin_edges(ax1_scaled))
    masked_data = np.ma.masked_where(np.isnan(data) | np.isinf(data), data)
    im = aux_ax_polar.pcolormesh(t, r, masked_data, cmap=cmap, vmin=vmin, vmax=vmax)

    # Draw transects if requested
    if show_sub:
        # Ensure index is array-like
        if isinstance(index, (int, np.integer)):
            indexes = [index]
        elif isinstance(index, (list, tuple)):
            indexes = list(index)
        else:
            indexes = np.atleast_1d(index).astype(int)

        for ii, idx in enumerate(indexes):
            if semi:
                mirror_index = -1 - idx
            else:
                mirror_index = (ax1_scaled.shape[0] // 2 + idx) % ax1_scaled.shape[0]

            # Draw line over colormesh
            vertex0 = np.array([[0, 0], [ax1_scaled[idx], ax2_max]])
            vertex1 = np.array([[0, 0], [ax1_scaled[mirror_index], ax2_max]])
            aux_ax_polar.plot(vertex0[:, 0], vertex0[:, 1], 'w')
            if sym:
                aux_ax_polar.plot(vertex1[:, 0], vertex1[:, 1], 'w--', linewidth=2)

            # Plot transects
            color = ['k', 'r', 'g', 'b', 'm', 'y'][ii % 6]
            ax_cart.plot(ax2, data[idx, :], '-' + color)
            if sym:
                ax_cart.plot(-ax2, data[mirror_index, :], '--' + color)

    # Add colorbar
    fig.colorbar(im, orientation='horizontal',
                 extend='both', ticks=np.linspace(vmin, vmax, 5),
                 shrink=0.7)

    # Add title
    title = da.attrs.get('long_name', da.name)
    if title is not None:
        ax_polar.set_title(title, weight='bold', position=(0.05, 0.97))

    return fig


def transect2D_xr(da, index=None, vmin=None, vmax=None, sym=True, swap='auto', 
                  fig=None, sub=121, color='k', percent=False, fmt='-'):
    """
    xarray-compatible version of transect2D.
    
    Transect of 2D DataArray - Fixed version that reuses existing axes.

    Parameters
    ----------
    da : xr.DataArray
        2D data array to display
    index : int or array-like, optional
        Index/indices to transect
    vmin, vmax : float, optional
        Value range
    sym : bool
        Use symmetrical axis
    swap : bool or 'auto'
        Swap axes if needed
    fig : matplotlib.figure.Figure, optional
        Destination figure
    sub : int
        Subplot position
    color : str
        Color for the plot
    percent : bool
        If True, set scale to 0-100%
    fmt : str
        Plot format string

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    assert da.ndim == 2, "DataArray must be 2D"

    if fig is None:
        fig = figure(figsize=(4.5, 2.5))

    # Get dimension names
    dim_names = list(da.dims)
    
    if swap == 'auto':
        if ('azi' in dim_names[1].lower()) and ('azi' not in dim_names[0].lower()):
            swap = True
        else:
            swap = False

    # Get axes and data
    if swap:
        ax1 = da.coords[dim_names[1]].values
        ax2 = da.coords[dim_names[0]].values
        name1 = dim_names[1]
        name2 = dim_names[0]
        data = da.values.T
    else:
        ax1 = da.coords[dim_names[0]].values
        ax2 = da.coords[dim_names[1]].values
        name1 = dim_names[0]
        name2 = dim_names[1]
        data = da.values

    # Determine value range
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    if vmin == vmax:
        vmin -= 0.001
        vmax += 0.001
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    if percent:
        vmin = 0.
        vmax = 100.

    ax1_scaled = ax1
    label2 = da.coords[name2].attrs.get('long_name', name2)

    # Ensure index is an integer
    if index is not None:
        if isinstance(index, (list, tuple)):
            index = int(index[0])
        else:
            index = int(index)
    
    if index is None:
        index = 0

    mirror_index = (ax1_scaled.shape[0] // 2 + index) % ax1_scaled.shape[0]

    ax2_min = np.amin(ax2)
    ax2_max = np.amax(ax2)
    label1 = name1 + ' {:7.2f}'.format(ax1_scaled[index])

    # Parse subplot specification
    nrows = sub // 100
    ncols = (sub // 10) % 10
    idx = (sub % 10) - 1

    # Check if subplot already exists
    ax_cart = None
    marker_name = f'_transect2D_sub_{sub}'
    if hasattr(fig, marker_name):
        ax_cart = getattr(fig, marker_name)

    is_new_axes = ax_cart is None
    if is_new_axes:
        ax_cart = fig.add_subplot(sub)
        setattr(fig, marker_name, ax_cart)
        ax_cart.grid(True)
        ax_cart.set_xlabel(label2)
        if sym:
            ax_cart.set_xlim(-ax2_max, ax2_max)
        else:
            ax_cart.set_xlim(ax2_min, ax2_max)
        ax_cart.set_ylim(vmin, vmax)
        ax_cart._transect2D_first = True
    else:
        # Expand ylim to accommodate new data
        current_ylim = ax_cart.get_ylim()
        new_vmin = min(current_ylim[0], vmin)
        new_vmax = max(current_ylim[1], vmax)
        ax_cart.set_ylim(new_vmin, new_vmax)

    ax_cart.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    # Plot transects
    ax_cart.plot(ax2, data[index, :], fmt, color=color)
    if sym:
        ax_cart.plot(-ax2, data[mirror_index, :], fmt, color=color)

    # Add title
    title = da.attrs.get('long_name', da.name)
    if title is not None:
        ax_cart.set_title(title)

    return fig