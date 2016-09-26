#!/usr/bin/env python
# encoding: utf-8

import warnings
warnings.simplefilter("ignore",DeprecationWarning)
from pylab import figure
import numpy as np
np.seterr(invalid='ignore', divide='ignore') # ignore division by zero errors
from luts import plot_polar, LUT, transect2D


def mdesc(desc, logI=False):
    sep1=desc.find('_')
    sep2=desc.find('(')
    sep3=desc.find(')')
    stokes=desc[0:sep1]
    dir=desc[sep1+1:sep2-1]

    if logI and stokes=='I':
        pref=r'$log_{10} '
    else:
        pref=r'$'
        
    if dir == 'up':
        return pref + stokes + '^{\uparrow}' + '_{'+desc[sep2+1:sep3]+'}$'
    else:
        return pref + stokes + '^{\downarrow}' + '_{'+desc[sep2+1:sep3]+'}$'
    

def smartg_view(mlut, logI=False, QU=False, Circ=False, full=False, field='up (TOA)', ind=0, cmap=None, fig=None):
    '''
    visualization of a smartg MLUT

    Options:
        logI: shows log10 of I
        Circ: shows Circular polarization 
        QU:  shows Q U and DoP
        field: level of output
        ind: index of azimutal plane
        full: shows all
        cmap: color map
    '''

    I = mlut['I_' + field]
    Q = mlut['Q_' + field]
    U = mlut['U_' + field]
    V = mlut['V_' + field]

    # Linearly polarized reflectance
    IPL = (Q*Q + U*U).apply(np.sqrt, 'Lin. Pol. ref.')
    
    # Polarized reflectance
    IP = (Q*Q + U*U +V*V).apply(np.sqrt, 'Pol. ref.')

    # Degree of Linear Polarization (%)
    DoLP = 100*IPL/I
    DoLP.desc = r'$DoLP$'
    
    # Angle of Linear Polarization (deg)
    AoLP = (U/Q)
    AoLP.apply(np.arctan)*90/np.pi
    AoLP.desc = r'$AoLP$'
    
    # Degree of Circular Polarization (%)
    DoCP = 100*V.apply(abs)/I
    DoCP.desc = r'$DoCP$'

    # Degree of Polarization (%)
    DoP = 100*IP/I
    DoP.desc = r'$DoP$'

    if not full:
        if QU:
            if fig is None: fig = figure(figsize=(9, 9))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                plot_polar(lI,  ind, rect='421', sub='423', fig=fig, cmap=cmap)
            else:
                I.desc = mdesc(I.desc)
                plot_polar(I,  ind, rect='421', sub='423', fig=fig, cmap=cmap)
            Q.desc = mdesc(Q.desc)
            U.desc = mdesc(U.desc)
            plot_polar(Q,  ind, rect='422', sub='424', fig=fig, cmap=cmap)
            plot_polar(U,  ind, rect='425', sub='427', fig=fig, cmap=cmap)
            if Circ:
                V.desc = mdesc(V.desc)
                plot_polar(V, ind, rect='426', sub='428', fig=fig, cmap=cmap)
            else:
                plot_polar(DoP, ind, rect='426', sub='428', fig=fig, vmin=0, vmax=100, cmap=cmap)
        else:
            # show only I and PR
            if fig is None: fig = figure(figsize=(9, 4.5))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                plot_polar(lI,  ind, rect='221', sub='223', fig=fig, cmap=cmap)
            else:
                I.desc = mdesc(I.desc)
                plot_polar(I,  ind, rect='221', sub='223', fig=fig, cmap=cmap)

            if Circ:
                plot_polar(DoCP, ind, rect='222', sub='224', fig=fig, vmin=0, vmax=100, cmap=cmap)
            else:
                plot_polar(DoP, ind, rect='222', sub='224', fig=fig, vmin=0, vmax=100, cmap=cmap)

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
        plot_polar(I,  ind, rect='241', sub='245', fig=fig1, cmap=cmap)
        plot_polar(Q,  ind, rect='242', sub='246', fig=fig1, cmap=cmap)
        plot_polar(U,  ind, rect='243', sub='247', fig=fig1, cmap=cmap)
        plot_polar(V,  ind, rect='244', sub='248', fig=fig1, cmap=cmap)
        
        fig2 = figure(figsize=(16, 4))
        Q.desc = mdesc(Q.desc)
        U.desc = mdesc(U.desc)
        V.desc = mdesc(V.desc)
        plot_polar(lI,  ind, rect='241', sub='245', fig=fig2, cmap=cmap)
        plot_polar(DoLP,  ind, rect='242', sub='246', fig=fig2, vmin=0, vmax=100, cmap=cmap)
        plot_polar(DoCP,  ind, rect='243', sub='247', fig=fig2, vmin=0, vmax=100, cmap=cmap)
        plot_polar(DoP,  ind, rect='244', sub='248', fig=fig2, vmin=0, vmax=100, cmap=cmap)

        return fig1, fig2

def transect_view(mlut, logI=False, QU=False, Circ=False, full=False, field='up (TOA)', ind=0, fig=None, color='k'):
    '''
    visualization of a smartg MLUT

    Options:
        logI: shows log10 of I
        Circ: shows Circular polarization 
        QU:  shows Q U and DoP
        field: level of output
        ind: index of azimutal plane
        full: shows all
        color: color of the transect
    '''

    I = mlut['I_' + field]
    Q = mlut['Q_' + field]
    U = mlut['U_' + field]
    V = mlut['V_' + field]

    # Linearly polarized reflectance
    IPL = (Q*Q + U*U).apply(np.sqrt, 'Lin. Pol. ref.')
    
    # Polarized reflectance
    IP = (Q*Q + U*U +V*V).apply(np.sqrt, 'Pol. ref.')

    # Degree of Linear Polarization (%)
    DoLP = 100*IPL/I
    DoLP.desc = r'$DoLP$'
    
    # Angle of Linear Polarization (deg)
    AoLP = (U/Q)
    AoLP.apply(np.arctan)*90/np.pi
    AoLP.desc = r'$AoLP$'
    
    # Degree of Circular Polarization (%)
    DoCP = 100*V.apply(abs)/I
    DoCP.desc = r'$DoCP$'

    # Degree of Polarization (%)
    DoP = 100*IP/I
    DoP.desc = r'$DoP$'

    if not full:
        if QU:
            if fig is None: fig = figure(figsize=(8, 8))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                transect2D(lI,  ind, sub=221, fig=fig, color=color)
            else:
                I.desc = mdesc(I.desc)
                transect2D(I,  ind, sub=221, fig=fig, color=color)
            Q.desc = mdesc(Q.desc)
            U.desc = mdesc(U.desc)
            transect2D(Q,  ind, sub=222, fig=fig, color=color)
            transect2D(U,  ind, sub=223, fig=fig, color=color)
            if Circ:
                V.desc = mdesc(V.desc)
                transect2D(V, ind, sub=224, fig=fig, color=color)
            else:
                transect2D(DoP, ind, sub=224, fig=fig, vmin=0, vmax=100, color=color)
        else:
            # show only I and PR
            if fig is None: fig = figure(figsize=(8, 4))
            if logI:
                lI=I.apply(np.log10)
                lI.desc = mdesc(I.desc, logI=logI)
                transect2D(lI,  ind, sub=121, fig=fig, color=color)
            else:
                I.desc = mdesc(I.desc)
                transect2D(I,  ind, sub=121, fig=fig, color=color)

            if Circ:
                transect2D(DoCP, ind, sub=122, fig=fig, vmin=0, vmax=100, color=color)
            else:
                transect2D(DoP, ind, sub=122, fig=fig, vmin=0, vmax=100, color=color)

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
        transect2D(I,  ind,  sub=141, fig=fig1, color=color)
        transect2D(Q,  ind,  sub=142, fig=fig1, color=color)
        transect2D(U,  ind, sub=143, fig=fig1, color=color)
        transect2D(V,  ind, sub=144, fig=fig1, color=color)
        
        Q.desc = mdesc(Q.desc)
        U.desc = mdesc(U.desc)
        V.desc = mdesc(V.desc)
        transect2D(lI,  ind, sub=141,fig=fig2, color=color)
        transect2D(DoLP,  ind, sub=142, fig=fig2, vmin=0, vmax=100, color=color)
        transect2D(DoCP,  ind, sub=143, fig=fig2, vmin=0, vmax=100, color=color)
        transect2D(DoP,  ind,  sub=144, fig=fig2, vmin=0, vmax=100, color=color)

        return fig1, fig2
        
def phase_view(mlut, ipha=None, fig= None, axarr=None, iw=0):
    '''
    visualization of a smartg MLUT phase function from output

    Options:
        ipha: sabsolute index of the phase function coming from Profile
        fig : fig object to be created or included in
        axarr : system of axes (2,2) to be created on used
    '''

    from pylab import subplots, setp
    from numpy import unique
    
    nd = mlut['tau'].ndim
    Linlabw=''
    if nd>1:
        wi = mlut['tau'].names.index('Wavelength') # Wavelength index
        key = [slice(None)]*nd
        key[wi] = iw
        key=tuple(key)
        labw=r' at $%.1f nm$'%mlut.axes['Wavelength'][iw]
    else:
        key=tuple([slice(None)])
        labw=''

    phase = mlut['phases_atm']
    ang = phase.axes[1]

    if (axarr==None):
        fig, axarr = subplots(2, 2)
        fig.set_size_inches(10, 6)
        
    if ipha==None : ni= unique(mlut['iphase'].__getitem__(key)) 
    else:ni=[ipha]
    
    for i in ni:
        
        P11 = 0.5*(phase[i,:,0]+phase[i,:,1])
        P12 = 0.5*(phase[i,:,0]-phase[i,:,1])
        P33 = phase[i,:,2]
        P43 = phase[i,:,3]
    
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
    
def atm_view(mlut, ipha=None, fig=None, ax=None, iw=0):
    '''
    visualization of a smartg MLUT atmospheric profile from output

    Options:
        ipha: sabsolute index of the phase function coming from Profile
        fig : fig object to be created or included in
        axarr : system of axes (2,2) to be created on used
        iw : in case of multi wavelength simulation, index of wavelength to be plotted
    '''

    from pylab import subplots, xlabel, ylabel

    if (ax==None):
        fig, ax = subplots(1, 1)
        fig.set_size_inches(5, 5)
    
    nd = mlut['tau'].ndim
    if nd>1:
        wi = mlut['tau'].names.index('Wavelength') # Wavelength index
        key = [slice(None)]*nd
        key[wi] = iw
        key=tuple(key)
        labw=r' at $%.1f nm$'%mlut.axes['Wavelength'][iw]
        
    else:
        key=tuple([slice(None)])
        labw=''
        
    z = mlut['tau'].axis('ALT',aslut=True)
    Dz = z.apply(np.gradient)
    Dz = Dz.apply(abs,'Dz')
    Dtau = mlut['tau'].sub().__getitem__(key).apply(np.gradient,'Dtau')
    Tot = (Dtau / Dz)
    Ext = Tot * (1. -mlut['abs'].sub().__getitem__(key))
    Gas = Tot * mlut['abs'].sub().__getitem__(key)
    DtauR = mlut['taumol'].sub().__getitem__(key).apply(np.gradient,'DtauR')
    ExtR = (DtauR / Dz)
    ExtA = Ext * (1-mlut['pmol']).sub().__getitem__(key)
    ScaA = ExtA * mlut['ssa'].sub().__getitem__(key)
    AbsA = ExtA * (1.-mlut['ssa'].sub().__getitem__(key))
    if (np.max(Gas[:]) > 0.) : ax.semilogx(Gas[:],z[:], 'g:',linewidth=3,label=r'$\sigma_{abs}^{gas}$')
    ax.semilogx(ExtR[:],z[:], 'b-.',linewidth=2, label=r'$\sigma_{sca}^{R}$' )
    if (np.max(AbsA[:]) > 0.) : ax.semilogx(AbsA[:],z[:],'r-',label=r'$\sigma_{abs}^{a+c}$')
    if (np.max(ScaA[:]) > 0.) : ax.semilogx(ScaA[:],z[:],'y',label=r'$\sigma_{sca}^{a+c}$')
    ax.semilogx(Tot[:],z[:],'k',label=r'$\sigma_{ext}^{tot}$')
    ax.grid()
    ax.set_xlim(1e-6,10)
    xlabel(r'$(km^{-1})$')
    ylabel(r'$z (km)$')
    ax.set_ylim(0,50)
    ax.set_title('Vertical profile'+labw)
    ax.legend()
    
    try:        
        i=0
        ax.annotate('%i'%(mlut['iphase'].sub().__getitem__(key)[0]),xy=(1e-5,47))
        for k in range(mlut.axes['ALT'].shape[0]):
            i0 = mlut['iphase'].sub().__getitem__(key)[k]
            if i0 != i :
                zl = mlut.axes['ALT'][k]
                ax.plot([1e-6,10],[zl+1,zl+1],'k--')
                ax.annotate('%i'%i0,xy=(1e-5,zl-1))
                i = i0
            
    except:
        return fig, ax
        
    return fig, ax
    
    
def input_view(mlut, iw=0):
    
    import matplotlib.pyplot as plt
    from numpy import array
    fig = plt.figure()
    fig.set_size_inches(12,6)
    try:
        mlut['phases_atm']
        ax1 = plt.subplot2grid((2,3),(0,0))
        ax2 = plt.subplot2grid((2,3),(0,1))
        ax3 = plt.subplot2grid((2,3),(1,0))
        ax4 = plt.subplot2grid((2,3),(1,1))
    
        axarr = array([[ax1,ax2],[ax3,ax4]])
    
        _,_= phase_view(mlut, iw=iw, axarr=axarr)
        
        ax5 = plt.subplot2grid((2,3),(0,2),rowspan=2,colspan=1)
        
        _,_= atm_view(mlut, iw=iw, ax=ax5)
        
        plt.tight_layout()
        
    except:
        _,_= atm_view(mlut, iw=iw)
