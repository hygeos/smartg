#!/usr/bin/env python
# encoding: utf-8


import warnings
warnings.simplefilter("ignore",DeprecationWarning)
from pylab import figure
import numpy as np
np.seterr(invalid='ignore', divide='ignore') # ignore division by zero errors
from luts import plot_polar, LUT


def smartg_view(mlut, QU=False, field='up (TOA)'):
    '''
    visualization of a smartg MLUT

    Options:
        QU: show Q and U also
    '''

    I = mlut['I_' + field]
    Q = mlut['Q_' + field]
    U = mlut['U_' + field]

    # polarized reflectance
    IP = (Q*Q + U*U).apply(np.sqrt, 'Pol. ref.')

    # polarization ratio
    PR = 100*IP/I
    PR.desc = 'Pol. ratio. (%)'

    if QU:
        fig = figure(figsize=(9, 9))
        plot_polar(I,  0, rect='421', sub='423', fig=fig)
        plot_polar(Q,  0, rect='422', sub='424', fig=fig)
        plot_polar(U,  0, rect='425', sub='427', fig=fig)
        plot_polar(PR, 0, rect='426', sub='428', fig=fig, vmin=0, vmax=100)
    else:
        # show only I and PR
        fig = figure(figsize=(9, 4.5))
        plot_polar(I,  0, rect='221', sub='223', fig=fig)
        plot_polar(PR, 0, rect='222', sub='224', fig=fig, vmin=0, vmax=100)
        
def phase_view(mlut, ipha=None, fig= None, axarr=None):
    '''
    visualization of a smartg MLUT phase function from output

    Options:
        ipha: sabsolute index of the phase function coming from Profile
        fig : fig object to be created or included in
        axarr : system of axes (2,2) to be created on used
    '''

    from pylab import subplots, setp
    
    phase = mlut['phases_atm']
    ang = phase.axes[1]
    Npha = phase[:,0,0].shape[0]
    if (fig==None or axarr==None):
        fig, axarr = subplots(2, 2)
    fig.set_size_inches(12, 6)
    if ipha==None : ni=range(Npha)
    else:ni=[ipha]
    
    for i in ni:
        P11 = 0.5*(phase[i,:,0]+phase[i,:,1])
        P12 = 0.5*(phase[i,:,0]-phase[i,:,1])
        P33 = phase[i,:,2]
        P43 = phase[i,:,3]
    
        axarr[0,0].semilogy(ang, P11,label='%3i'%i)
        axarr[0,0].set_title(r'$P_{11}$')
        axarr[0,0].grid()
        
        axarr[0,1].plot(ang, -P12/P11)
        axarr[0,1].set_title(r'-$P_{12}/P_{11}$')
        axarr[0,1].grid()
        
        axarr[1,0].plot(ang, P33/P11)
        axarr[1,0].set_title(r'$P_{33}/P_{11}$')
        axarr[1,0].grid()
                
        axarr[1,1].plot(ang, P43/P11)
        axarr[1,1].set_title(r'$P_{43}/P_{11}$')
        axarr[1,1].grid()
    
    setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    axarr[0,0].legend(loc='upper center',fontsize = 'medium',labelspacing=0.01)

    return fig, axarr
    
def atm_view(mlut, ipha=None, fig=None, ax=None):
    '''
    visualization of a smartg MLUT atmospheric profile from output

    Options:
        ipha: sabsolute index of the phase function coming from Profile
        fig : fig object to be created or included in
        axarr : system of axes (2,2) to be created on used
    '''

    from pylab import subplots, xlabel, ylabel
    
    if (fig==None or ax==None):
        fig, ax = subplots(1, 1)
    fig.set_size_inches(5, 5)
    
    z = LUT(mlut.axes['ALT'],axes=[mlut.axes['ALT']],names=['ALT'])
    Dz = z.apply(np.gradient)
    Dz = Dz.apply(abs,'Dz')
    Dtau = mlut['H'].apply(np.gradient,'Dtau')
    Tot = (Dtau / Dz)
    Ext = Tot * (1. -mlut['percent_abs'])
    Gas = Tot * mlut['percent_abs']
    DtauR = mlut['hmol'].apply(np.gradient,'DtauR')
    ExtR = (DtauR / Dz)
    ExtA = Ext - ExtR
    ExtA = ExtA.apply(abs)
    ScaA = ExtA * mlut['XSSA']
    AbsA = ExtA * (1.-mlut['XSSA'])
    if (np.max(Gas[:]) > 0.) : ax.semilogx(Gas[:],z[:], 'g',label=r'$\sigma_{abs}^{gas}$')
    ax.semilogx(ExtR[:],z[:], 'b',label=r'$\sigma_{sca}^{R}$' )
    if (np.max(AbsA[:]) > 0.) : ax.semilogx(AbsA[:],z[:],'r',label=r'$\sigma_{abs}^{a+c}$')
    if (np.max(ScaA[:]) > 0.) : ax.semilogx(ScaA[:],z[:],'y',label=r'$\sigma_{sca}^{a+c}$')
    ax.semilogx(Tot[:],z[:],'k',label=r'$\sigma_{ext}^{tot}$')
    ax.grid()
    ax.set_xlim(1e-10,10)
    xlabel(r'$(km^{-1})$')
    ylabel(r'$z (km)$')
    ax.set_ylim(0,50)
    ax.set_title('Vertical profile')
    ax.legend()
    i=0
    for k in range(len(list(mlut['IPHA'].axes[0]))):
        if mlut['IPHA'][k]!=i :
            zl = mlut['IPHA'].axes[0][k]
            ax.plot([1e-10,10],[zl+1,zl+1],'k--')
            ax.annotate('%i'%mlut['IPHA'][k],xy=(1e-8,zl-1))
            i=mlut['IPHA'][k]
    return fig, ax
