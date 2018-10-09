#!/usr/bin/env python
# encoding: utf-8


from __future__ import print_function, division, absolute_import

import warnings
warnings.simplefilter("ignore",DeprecationWarning)
from pylab import figure, subplot2grid, tight_layout, setp, subplots, xlabel, ylabel
import numpy as np
np.seterr(invalid='ignore', divide='ignore') # ignore division by zero errors
from smartg.tools.luts import plot_polar, transect2D, Idx
from smartg.atmosphere import diff1
from smartg.water import diff2


def mdesc(desc, logI=False):
    sep1=desc.find('_')
    sep2=desc.find('(')
    sep3=desc.find(')')
    if sep1 == 1 : stokes=desc[0:1]
    else : stokes=desc[sep1-2:sep1]
    #stokes=desc[0:sep1]
    dir=desc[sep1+1:sep2-1]

    if logI and stokes=='I':
        pref=r'$log_{10} '
    else:
        pref=r'$'
        
    if dir == 'up':
        return pref + stokes + r'^{\uparrow}' + '_{'+desc[sep2+1:sep3]+'}' + desc[sep3+1:] +'$'
    else:
        return pref + stokes + r'^{\downarrow}' + '_{'+desc[sep2+1:sep3]+'}' + desc[sep3+1:] +'$'
    

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

    if not full:
        if QU:
            if fig is None: fig = figure(figsize=(9, 9))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                plot_polar(lI,  index=ind, rect='421', sub='423', fig=fig, cmap=cmap, vmin=Imin, vmax=Imax)
            else:
                I.desc = mdesc(I.desc)
                plot_polar(I,  index=ind, rect='421', sub='423', fig=fig, cmap=cmap, vmin=Imin, vmax=Imax)
            Q.desc = mdesc(Q.desc)
            U.desc = mdesc(U.desc)
            plot_polar(Q,  index=ind, rect='422', sub='424', fig=fig, cmap=cmap)
            plot_polar(U,  index=ind, rect='425', sub='427', fig=fig, cmap=cmap)
            if Circ:
                V.desc = mdesc(V.desc)
                plot_polar(V, index=ind, rect='426', sub='428', fig=fig, cmap=cmap)
            else:
                plot_polar(DoP, index=ind, rect='426', sub='428', fig=fig, vmin=Pmin, vmax=Pmax, cmap=cmap)
        else:
            # show only I and PR
            if fig is None: fig = figure(figsize=(9, 4.5))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                plot_polar(lI,  index=ind, rect='221', sub='223', fig=fig, cmap=cmap, vmin=Imin, vmax=Imax)
            else:
                I.desc = mdesc(I.desc)
                plot_polar(I,  index=ind, rect='221', sub='223', fig=fig, cmap=cmap, vmin=Imin, vmax=Imax)

            if Circ:
                plot_polar(DoCP, index=ind, rect='222', sub='224', fig=fig, vmin=0, vmax=Pmax, cmap=cmap)
            else:
                plot_polar(DoP, index=ind, rect='222', sub='224', fig=fig, vmin=Pmin, vmax=Pmax, cmap=cmap)

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
        plot_polar(I,  index=ind, rect='241', sub='245', fig=fig1, cmap=cmap, vmin=Imin, vmax=Imax)
        plot_polar(Q,  index=ind, rect='242', sub='246', fig=fig1, cmap=cmap)
        plot_polar(U,  index=ind, rect='243', sub='247', fig=fig1, cmap=cmap)
        plot_polar(V,  index=ind, rect='244', sub='248', fig=fig1, cmap=cmap)
        
        fig2 = figure(figsize=(16, 4))
        Q.desc = mdesc(Q.desc)
        U.desc = mdesc(U.desc)
        V.desc = mdesc(V.desc)
        plot_polar(lI,  index=ind, rect='241', sub='245', fig=fig2, cmap=cmap)
        plot_polar(DoLP,  index=ind, rect='242', sub='246', fig=fig2, vmin=Pmin, vmax=Pmax, cmap=cmap)
        plot_polar(DoCP,  index=ind, rect='243', sub='247', fig=fig2, vmin=Pmin, vmax=Pmax, cmap=cmap)
        plot_polar(DoP,  index=ind, rect='244', sub='248', fig=fig2, vmin=Pmin, vmax=Pmax, cmap=cmap)
        #plot_polar(AoLP,  index=ind, rect='244', sub='248', fig=fig2, vmin=-180, vmax=180, cmap=cmap)

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

    if not full:
        if QU:
            if fig is None: fig = figure(figsize=(8, 8))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                transect2D(lI,  index=ind, sub=221, fig=fig, color=color,  **kwargs)
            else:
                I.desc = mdesc(I.desc)
                transect2D(I,  index=ind, sub=221, fig=fig, color=color,   **kwargs)
            Q.desc = mdesc(Q.desc)
            U.desc = mdesc(U.desc)
            transect2D(Q,  index=ind, sub=222, fig=fig, color=color, **kwargs)
            transect2D(U,  index=ind, sub=223, fig=fig, color=color, **kwargs)
            if Circ:
                V.desc = mdesc(V.desc)
                transect2D(V, index=ind, sub=224, fig=fig, color=color, **kwargs)
            else:
                transect2D(DoP, index=ind, sub=224, fig=fig,  color=color, percent=True, **kwargs)
        else:
            # show only I and PR
            if fig is None: fig = figure(figsize=(8, 4))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                transect2D(lI,  index=ind, sub=121, fig=fig, color=color,   **kwargs)
            else:
                I.desc = mdesc(I.desc)
                transect2D(I,  index=ind, sub=121, fig=fig, color=color,  **kwargs)

            if Circ:
                transect2D(DoCP, index=ind, sub=122, fig=fig,  color=color, percent=True, **kwargs)
            else:
                transect2D(DoP, index=ind, sub=122, fig=fig, color=color, percent=True, **kwargs)

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
        transect2D(I,  index=ind,  sub=141, fig=fig1, color=color,  **kwargs)
        transect2D(Q,  index=ind,  sub=142, fig=fig1, color=color, **kwargs)
        transect2D(U,  index=ind, sub=143, fig=fig1, color=color, **kwargs)
        transect2D(V,  index=ind, sub=144, fig=fig1, color=color, **kwargs)
        
        Q.desc = mdesc(Q.desc)
        U.desc = mdesc(U.desc)
        V.desc = mdesc(V.desc)
        transect2D(lI,  index=ind, sub=141,fig=fig2, color=color, **kwargs)
        transect2D(DoLP,  index=ind, sub=142, fig=fig2, color=color, percent=True, **kwargs)
        transect2D(DoCP,  index=ind, sub=143, fig=fig2, color=color, percent=True, **kwargs)
        transect2D(DoP,  index=ind,  sub=144, fig=fig2, color=color, percent=True, **kwargs)
        #transect2D(AoLP, index=ind,  sub=144, fig=fig2, color=color, **kwargs)

        return fig1, fig2

def spectrum(lut, vmin=None, vmax=None, sub='111', fig=None, color='k', percent=False, fmt='-'):
    '''
    spectrum of a 1D LUT

    lut:  1D look-up table to display
            with axis 'wavelength'
    vmin, vmax: range of values
                default None: determine min/max from values
    fig : destination figure. If None (default), create a new figure.
    color : color of the transect
    percent: if True set scale to 0 to 100%
    '''
    from pylab import figure

    assert 'wavelength' in lut.names

    if fig is None:
        fig = figure(figsize=(4.5, 2.5))

    ax1   = lut.axes[0]
    data  = lut.data

    if vmin is None:
        vmin = np.amin(lut.data[~np.isnan(lut.data)])
    if vmax is None:
        vmax = np.amax(lut.data[~np.isnan(lut.data)])
    if vmin == vmax:
        vmin -= 0.001
        vmax += 0.001
    if vmin > vmax: vmin, vmax = vmax, vmin
    if percent:
        vmin=0.
        vmax=100.

    ax1_min = np.amin(ax1)
    ax1_max = np.amax(ax1)
    #

    ax_cart = fig.add_subplot(sub)
    ax_cart.grid(True)

    ax_cart.set_xlim(ax1_min, ax1_max)
    ax_cart.set_ylim(vmin, vmax)
    ax_cart.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    #ax_cart.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    ax_cart.grid(True)
    ax_cart.set_xlabel(r'$\lambda$ (nm)')

    #
    # plot
    #
    ax_cart.plot(ax1 , data[:], fmt, color=color)

    if lut.desc is not None:
        ax_cart.set_title(lut.desc)


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
                spectrum(lI, sub='221', fig=fig, color=color,  **kwargs)
            else:
                I.desc = mdesc(I.desc)
                spectrum(I,  sub='221', fig=fig, color=color,   **kwargs)
            Q.desc = mdesc(Q.desc)
            U.desc = mdesc(U.desc)
            spectrum(Q, sub='222', fig=fig, color=color, **kwargs)
            spectrum(U, sub='223', fig=fig, color=color, **kwargs)
            if Circ:
                V.desc = mdesc(V.desc)
                spectrum(V, sub='224', fig=fig, color=color, **kwargs)
            else:
                spectrum(DoP, sub='224', fig=fig,  color=color, percent=True, **kwargs)
        else:
            # show only I and PR
            if fig is None: fig = figure(figsize=(8, 4))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                spectrum(lI, sub='121', fig=fig, color=color,   **kwargs)
            else:
                I.desc = mdesc(I.desc)
                spectrum(I, sub='121', fig=fig, color=color,  **kwargs)

            if Circ:
                spectrum(DoCP, sub='122', fig=fig,  color=color, percent=True, **kwargs)
            else:
                spectrum(DoP, sub='122', fig=fig, color=color, percent=True, **kwargs)

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
        spectrum(I, sub='141', fig=fig1, color=color,  **kwargs)
        spectrum(Q, sub='142', fig=fig1, color=color, **kwargs)
        spectrum(U, sub='143', fig=fig1, color=color, **kwargs)
        spectrum(V, sub='144', fig=fig1, color=color, **kwargs)
        
        Q.desc = mdesc(Q.desc)
        U.desc = mdesc(U.desc)
        V.desc = mdesc(V.desc)
        spectrum(lI, sub='141', fig=fig2, color=color, **kwargs)
        spectrum(DoLP, sub='142', fig=fig2, color=color, percent=True, **kwargs)
        spectrum(DoCP, sub='143', fig=fig2, color=color, percent=True, **kwargs)
        spectrum(DoP, sub='144', fig=fig2, color=color, percent=True, **kwargs)
        #spectrum(AoLP, index=ind,  sub=144, fig=fig2, color=color, **kwargs)

        return fig1, fig2
        
def phase_view(mlut, ipha=None, fig=None, axarr=None, iw=0, kind='atm'):
    '''
    visualization of a smartg MLUT phase function from output

    Options:
        ipha: absolute index of the phase function coming from Profile
        fig : fig object to be created or included in
        axarr : system of axes (2,2) to be created on used
        iw : in case of multi wavelength simulation, index of wavelength to be plotted
        kind : atmopsheric 'atm' or oceanic 'oc' phase function
    '''

    nd = mlut['OD_'+kind].ndim
    if nd>1:
        wi = mlut['OD_'+kind].names.index('wavelength') # Wavelength index
        key = [slice(None)]*nd
        key[wi] = iw
        key=tuple(key)
        labw=r' at $%.1f nm$'%mlut.axes['wavelength'][iw]
    else:
        key=tuple([slice(None)])
        labw=''

    phase = mlut['phase_'+kind]
    ang = phase.axis('theta_'+kind)
    if (axarr is None):
        fig, axarr = subplots(2, 2)
        fig.set_size_inches(10, 6)
        
    if (ipha is None) : ni= np.unique(mlut['iphase_'+kind].__getitem__(key)) 
    else:ni=[ipha]
    
    for i in ni:
        
        P11 = 0.5*(phase[i,0,:]+phase[i,1,:])
        P12 = 0.5*(phase[i,0,:]-phase[i,1,:])
        P33 = phase[i,2,:]
        P43 = phase[i,3,:]
    
        if (np.max(P11[:]) > 0.) :axarr[0,0].semilogy(ang, P11,label='%3i'%i)
        axarr[0,0].set_title(r'$P_{11}$'+labw)
        axarr[0,0].grid()
        
        if (np.max(P11[:]) > 0.) :axarr[0,1].plot(ang, -P12/P11)
        axarr[0,1].set_title(r'-$P_{12}/P_{11}$')
        axarr[0,1].grid()
        
        if (np.max(P11[:]) > 0.) :axarr[1,0].plot(ang, P33/P11)
        axarr[1,0].set_title(r'$P_{33}/P_{11}$')
        axarr[1,0].grid()
                
        if (np.max(P11[:]) > 0.) :axarr[1,1].plot(ang, P43/P11)
        axarr[1,1].set_title(r'$P_{43}/P_{11}$')
        axarr[1,1].grid()
    
    setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    axarr[0,0].legend(loc='upper center',fontsize = 'medium',labelspacing=0.01)

    return fig, axarr

    
def profile_view(mlut, fig=None, ax=None, iw=0, kind='atm', zmax=None):
    '''
    visualization of a smartg MLUT profile from output

    Options:
        fig : fig object to be created or included in
        axarr : system of axes (2,2) to be created or used
        iw : in case of multi wavelength simulation, index of wavelength to be plotted
        kind : atmopsheric 'atm' or oceanic 'oc' profile
        zmax: max altitude or depth of the plot
    '''

    if (ax is None):
        fig, ax = subplots(1, 1)
        fig.set_size_inches(5, 5)
    
    nd = mlut['OD_'+kind].ndim
    if nd>1:
        wi = mlut['OD_'+kind].names.index('wavelength') # Wavelength index
        key = [slice(None)]*nd
        key[wi] = iw
        key=tuple(key)
        labw=r' at $%.1f nm$'%mlut.axes['wavelength'][iw]
        
    else:
        key=tuple([slice(None)])
        labw=''

    z = mlut.axis('z_'+kind, aslut=True)
    if kind=='oc': 
        sign=-1.
        func = diff2
    else:
        sign=1.    
        func = diff1
    Dz = z.apply(func)
    Dz = Dz.apply(abs,'Dz')
    Dtau     = sign * mlut['OD_'+kind    ].sub().__getitem__(key).apply(func,'Dtau')
    Dtau_Sca = sign * mlut['OD_sca_'+kind    ].sub().__getitem__(key).apply(func,'Dtau_Sca')
    if kind=='atm':
        Dtau_ExtA = sign * mlut['OD_p'].sub().__getitem__(key).apply(func,'Dtau_ExtA')
        Dtau_ScaR = sign * mlut['OD_r'].sub().__getitem__(key).apply(func,'Dtau_ScaR')
        Dtau_AbsG = sign * mlut['OD_g'].sub().__getitem__(key).apply(func,'Dtau_AbsG')
        ssa_p = mlut['ssa_p_'+kind].sub().__getitem__(key)
        Dtau_ScaA = Dtau_ExtA * ssa_p
        Dtau_AbsA = Dtau_ExtA * (1. - ssa_p)
        if (np.max(Dtau_AbsA[:]) > 0.) : ax.semilogx((Dtau_AbsA/Dz)[:], z[:], 'r--',label=r'$\sigma_{abs}^{a+c}$')
        if (np.max(Dtau_ScaA[:]) > 0.) : ax.semilogx((Dtau_ScaA/Dz)[:], z[:], 'r',  label=r'$\sigma_{sca}^{a+c}$')
        if (np.max(Dtau_AbsG[:]) > 0.) : ax.semilogx((Dtau_AbsG/Dz)[:], z[:], 'g--',  label=r'$\sigma_{abs}^{gas}$')
        ax.semilogx((Dtau_ScaR/Dz)[:], z[:], 'b', label=r'$\sigma_{sca}^{R}$' )
        ax.set_xlim(1e-6,10)
        xlabel(r'$(km^{-1})$')
        ylabel(r'$z (km)$')
        if zmax is None : zmax = max(100.,z.data.max())
        ax.set_ylim(0,zmax)
    else :
        Dtau_ExtP = sign * mlut['OD_p_oc'].sub().__getitem__(key).apply(func,'Dtau_ExtP')
        Dtau_ExtW = sign * mlut['OD_w'].sub().__getitem__(key).apply(func,'Dtau_ExtW')
        Dtau_AbsY = sign * mlut['OD_y'].sub().__getitem__(key).apply(func,'Dtau_AbsY')
        ssa_p = mlut['ssa_p_'+kind].sub().__getitem__(key)
        ssa_w = mlut['ssa_w'].sub().__getitem__(key)
        pine  = mlut['pine_oc'].sub().__getitem__(key)
        Dtau_ScaP = Dtau_ExtP * ssa_p
        Dtau_AbsP = Dtau_ExtP * (1. - ssa_p)
        Dtau_ScaW = Dtau_ExtW * ssa_w
        Dtau_AbsW = Dtau_ExtW * (1. - ssa_w)
        Dtau_Ine  = Dtau_Sca  * pine
        if (np.max(Dtau_AbsP[:]) > 0.) : ax.semilogx((Dtau_AbsP/Dz)[:], z[:], 'r--',label=r'$\sigma_{abs}^{p}$')
        if (np.max(Dtau_ScaP[:]) > 0.) : ax.semilogx((Dtau_ScaP/Dz)[:], z[:], 'r',  label=r'$\sigma_{sca}^{p}$')
        if (np.max(Dtau_AbsW[:]) > 0.) : ax.semilogx((Dtau_AbsW/Dz)[:], z[:], 'b--',label=r'$\sigma_{abs}^{w}$')
        if (np.max(Dtau_ScaW[:]) > 0.) : ax.semilogx((Dtau_ScaW/Dz)[:], z[:], 'b',  label=r'$\sigma_{sca}^{w}$')
        if (np.max(Dtau_AbsY[:]) > 0.) : ax.semilogx((Dtau_AbsY/Dz)[:], z[:], 'y--',label=r'$\sigma_{abs}^{y}$')
        if (np.max(Dtau_Ine[:])  > 0.) : ax.semilogx((Dtau_Ine/Dz)[:] , z[:], 'm:' ,label=r'$\sigma_{ine}^{}$')
        ax.set_xlim(1e-4,10)
        xlabel(r'$(m^{-1})$')
        ylabel(r'$z (m)$')
        if zmax is None : zmax = min(-100.,z.data.min())
        ax.set_ylim(zmax,0)
    ax.semilogx((Dtau/Dz)[:], z[:], 'k^-', label=r'$\sigma_{ext}^{tot}$')
    ax.set_title('Vertical profile'+labw)
    ax.grid()
    ax.legend()
    
    try:        
        i=0
        if kind=='atm':
            xy=(1e-5,zmax*0.9)
        else:
            xy=(1e-4,zmax*0.9)
        ax.annotate('%i'%(mlut['iphase_'+kind].sub().__getitem__(key)[0]),xy=xy)
        for k in range(mlut.axes['z_'+kind].shape[0]):
            i0 = mlut['iphase_'+kind].sub().__getitem__(key)[k]
            if i0 != i :
                zl = mlut.axes['z_'+kind][k]
                if kind=='atm':
                    ax.plot([1e-6,10],[zl,zl],'k--')
                    #ax.plot([1e-6,10],[zl+1,zl+1],'k--')
                    ax.annotate('%i'%i0,xy=(1e-5,zl-1))
                else:
                    ax.plot([1e-4,10],[zl,zl],'k--')
                    #ax.plot([1e-6,10],[zl+1,zl+1],'k--')
                    ax.annotate('%i'%i0,xy=(1e-4,zl-1))
                i = i0
            
    except:
        return fig, ax
        
    return fig, ax
    
    
def input_view(mlut, iw=0, kind='atm', zmax=None, ipha=None):
    ''' 
    visualization of a smartg MLUT profile and phase functions from output

    Options:
        iw : in case of multi wavelength simulation, index of wavelength to be plotted
        kind : atmopsheric 'atm' or oceanic 'oc' profile
        zmax: max altitude or depth of the plot
        ipha: absolute index of the phase function coming from Profile
    '''
    fig = figure()
    fig.set_size_inches(12,6)
    try:
        mlut['phase_'+kind]
        ax1 = subplot2grid((2,3),(0,0))
        ax2 = subplot2grid((2,3),(0,1))
        ax3 = subplot2grid((2,3),(1,0))
        ax4 = subplot2grid((2,3),(1,1))
    
        axarr = np.array([[ax1,ax2],[ax3,ax4]])
    
        _,_= phase_view(mlut, iw=iw, axarr=axarr, kind=kind, ipha=ipha)
        
        ax5 = subplot2grid((2,3),(0,2),rowspan=2,colspan=1)
        
        _,_= profile_view(mlut, iw=iw, ax=ax5, kind=kind, zmax=zmax)
        
        tight_layout()
        
    except:
        _,_= profile_view(mlut, iw=iw, kind=kind, zmax=zmax)

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
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor='k')
                    ax[1,i].errorbar(-th,10**sm-10**refm,yerr=dsm,fmt=sym1+sym2,ecolor='k') 
                else:
                    ax[1,i].errorbar(th,10**sp-10**refp, \
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor='k')
                    ax[1,i].errorbar(-th,10**sm-10**refm,fmt=sym1+sym2,ecolor='k') 
    
            else:
                if errb:
                    ax[1,i].errorbar(th,sp-refp, yerr=dsp,\
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor=sym1)
                    ax[1,i].errorbar(-th,sm-refm,yerr=dsm,fmt=sym1+sym2,ecolor=sym1) 
                else:
                    ax[1,i].errorbar(th,sp-refp, \
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor=sym1)
                    ax[1,i].errorbar(-th,sm-refm,fmt=sym1+sym2,ecolor=sym1) 
            ax[1,i].set_ylim([-1*ema,ema])
            ax[1,i].set_xlim([-SZA_MAX,SZA_MAX])  

            if errb:
                ax[2,i].errorbar(th,(sp-refp)/refp*100, yerr=dsp/abs(refp)*100, \
                             fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor=sym1)
                ax[2,i].errorbar(-th,(sm-refm)/refm*100, yerr= dsm/abs(refm)*100,fmt=sym1+sym2,ecolor=sym1)  
                if (i==0 and errref is not None):
                    ax[2,0].plot(th,errref/refp*100,sym1+'-.')
                    ax[2,0].plot(th,-errref/refp*100,sym1+'-.')
                    ax[2,0].plot(-th,errref/refm*100,sym1+'-.')
                    ax[2,0].plot(-th,-errref/refm*100,sym1+'-.')
            else:
                ax[2,i].errorbar(th,(sp-refp)/refp*100,\
                             fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor='k')
                ax[2,i].errorbar(-th,(sm-refm)/refm*100,fmt=sym1+sym2,ecolor='k')  
            
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

