#!/usr/bin/env python
# encoding: utf-8


from __future__ import print_function, division, absolute_import

import warnings
warnings.simplefilter("ignore",DeprecationWarning)
from pylab import figure, subplot2grid, tight_layout, setp, subplots, xlabel, ylabel
import numpy as np
np.seterr(invalid='ignore', divide='ignore') # ignore division by zero errors
from smartg.tools.luts import plot_polar, transect2D
from smartg.atmosphere import diff1


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
        return pref + stokes + r'^{\uparrow}' + '_{'+desc[sep2+1:sep3]+'}$'
    else:
        return pref + stokes + r'^{\downarrow}' + '_{'+desc[sep2+1:sep3]+'}$'
    

def smartg_view(mlut, logI=False, QU=False, Circ=False, full=False, field='up (TOA)', ind=[0], cmap=None, fig=None, subdict=None,
        Imin=None, Imax=None):
    '''
    visualization of a smartg MLUT

    Options:
        logI: shows log10 of I
        Circ: shows Circular polarization 
        QU:  shows Q U and DoP
        field: level of output
        ind: list of indices of azimutal planes
        full: shows all
        cmap: color map
        fig: already existing figure
        subdict: dictionnary of LUT subsetter (see LUT class , sub() method)
        Imin: min value of I
        Imax: max value of I

    Outputs:
    if full is False, it returns 1 figure
    if full is True,  it returns 2 figures
    '''

    I = mlut['I_' + field]
    Q = mlut['Q_' + field]
    U = mlut['U_' + field]
    V = mlut['V_' + field]

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
    DoP.desc = r'$DoP\,(\%)$'

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
                plot_polar(DoP, index=ind, rect='426', sub='428', fig=fig, vmin=0, vmax=100, cmap=cmap)
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
                plot_polar(DoCP, index=ind, rect='222', sub='224', fig=fig, vmin=0, vmax=100, cmap=cmap)
            else:
                plot_polar(DoP, index=ind, rect='222', sub='224', fig=fig, vmin=0, vmax=100, cmap=cmap)

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
        plot_polar(DoLP,  index=ind, rect='242', sub='246', fig=fig2, vmin=0, vmax=100, cmap=cmap)
        plot_polar(DoCP,  index=ind, rect='243', sub='247', fig=fig2, vmin=0, vmax=100, cmap=cmap)
        plot_polar(DoP,  index=ind, rect='244', sub='248', fig=fig2, vmin=0, vmax=100, cmap=cmap)

        return fig1, fig2

def transect_view(mlut, logI=False, QU=False, Circ=False, full=False, field='up (TOA)', ind=[0], fig=None, color='k', subdict=None, 
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

    I = mlut['I_' + field]
    Q = mlut['Q_' + field]
    U = mlut['U_' + field]
    V = mlut['V_' + field]

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

        return fig1, fig2
        
def phase_view(mlut, ipha=None, fig= None, axarr=None, iw=0, kind='atm'):
    '''
    visualization of a smartg MLUT phase function from output

    Options:
        ipha: sabsolute index of the phase function coming from Profile
        fig : fig object to be created or included in
        axarr : system of axes (2,2) to be created on used
    '''

    nd = mlut['OD_'+kind].ndim
    #Linlabw=''
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
    
def atm_view(mlut, ipha=None, fig=None, ax=None, iw=0, kind='atm'):
    '''
    visualization of a smartg MLUT atmospheric profile from output

    Options:
        ipha: absolute index of the phase function coming from Profile
        fig : fig object to be created or included in
        axarr : system of axes (2,2) to be created or used
        iw : in case of multi wavelength simulation, index of wavelength to be plotted
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

    sign=1.    
    if kind=='oc': sign=-1.
    z = mlut.axis('z_'+kind, aslut=True)
    Dz = z.apply(diff1)
    Dz = Dz.apply(abs,'Dz')
    Dtau     = sign * mlut['OD_'+kind    ].sub().__getitem__(key).apply(diff1,'Dtau')
    Dtau_Sca = sign * mlut['OD_sca_'+kind].sub().__getitem__(key).apply(diff1,'Dtau_sca')
    Dtau_Abs = sign * mlut['OD_abs_'+kind].sub().__getitem__(key).apply(diff1,'Dtau_abs')

    pmol     = mlut['pmol_'+kind].sub().__getitem__(key)
    Dtau_ScaR = Dtau_Sca * pmol
    Dtau_ScaA = Dtau_Sca - Dtau_ScaR
    if kind=='atm':
        ssa_p = mlut['ssa_p_'+kind].sub().__getitem__(key)
        Dtau_ExtA = Dtau_ScaA / ssa_p
        Dtau_AbsA = Dtau_ExtA * (1. - ssa_p)
        Dtau_AbsG = Dtau_Abs - Dtau_AbsA
        if (np.max(Dtau_AbsA[:]) > 0.) : ax.semilogx((Dtau_AbsA/Dz)[:], z[:], 'r--',label=r'$\sigma_{abs}^{a+c}$')
        if (np.max(Dtau_ScaA[:]) > 0.) : ax.semilogx((Dtau_ScaA/Dz)[:], z[:], 'r',  label=r'$\sigma_{sca}^{a+c}$')
        if (np.max(Dtau_AbsG[:]) > 0.) : ax.semilogx((Dtau_AbsG/Dz)[:], z[:], 'g',  label=r'$\sigma_{abs}^{gas}$')
        ax.semilogx((Dtau_ScaR/Dz)[:], z[:], 'b', label=r'$\sigma_{sca}^{R}$' )
        ax.set_xlim(1e-6,10)
        xlabel(r'$(km^{-1})$')
        ylabel(r'$z (km)$')
        #ax.set_ylim(0,50)
        ax.set_ylim(0,max(100.,z.data.max()))
    else :
        Dtau_ScaA.data[0] = Dtau_ScaA.data[1]
        Dtau_Abs.data[0] = Dtau_Abs.data[1]
        Dtau_ScaR.data[0] = Dtau_ScaR.data[1]
        Dtau.data[0] = Dtau.data[1]
        Dz.data[0] = Dz.data[1]
        if (np.max(Dtau_ScaA[:]) > 0.) : ax.semilogx((Dtau_ScaA/Dz)[:], z[:],'r^-', label=r'$\sigma_{sca}^{p}$')
        if (np.max(Dtau_Abs[:]) > 0.)  : ax.semilogx((Dtau_Abs/Dz)[:], z[:], 'g^-', label=r'$\sigma_{abs}^{tot}$')
        ax.semilogx((Dtau_ScaR/Dz)[:], z[:], 'b^-', label=r'$\sigma_{sca}^{w}$' )
        ax.set_xlim(1e-3,3)
        xlabel(r'$(m^{-1})$')
        ylabel(r'$z (m)$')
        #ax.set_ylim(-50,1)
        ax.set_ylim(0,min(-100.,z.data.min()))
    ax.semilogx((Dtau/Dz)[:], z[:], 'k^-', label=r'$\sigma_{ext}^{tot}$')
    ax.set_title('Vertical profile'+labw)
    ax.grid()
    ax.legend()
    
    try:        
        i=0
        ax.annotate('%i'%(mlut['iphase_'+kind].sub().__getitem__(key)[0]),xy=(1e-5,51))
        for k in range(mlut.axes['z_'+kind].shape[0]):
            i0 = mlut['iphase_'+kind].sub().__getitem__(key)[k]
            if i0 != i :
                zl = mlut.axes['z_'+kind][k]
                ax.plot([1e-6,10],[zl,zl],'k--')
                #ax.plot([1e-6,10],[zl+1,zl+1],'k--')
                ax.annotate('%i'%i0,xy=(1e-5,zl-1))
                i = i0
            
    except:
        return fig, ax
        
    return fig, ax
    
    
def input_view(mlut, iw=0, kind='atm'):
    
    fig = figure()
    fig.set_size_inches(12,6)
    try:
        mlut['phase_'+kind]
        ax1 = subplot2grid((2,3),(0,0))
        ax2 = subplot2grid((2,3),(0,1))
        ax3 = subplot2grid((2,3),(1,0))
        ax4 = subplot2grid((2,3),(1,1))
    
        axarr = np.array([[ax1,ax2],[ax3,ax4]])
    
        _,_= phase_view(mlut, iw=iw, axarr=axarr, kind=kind)
        
        ax5 = subplot2grid((2,3),(0,2),rowspan=2,colspan=1)
        
        _,_= atm_view(mlut, iw=iw, ax=ax5, kind=kind)
        
        tight_layout()
        
    except:
        _,_= atm_view(mlut, iw=iw, kind=kind)
