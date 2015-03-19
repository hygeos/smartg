#!/usr/bin/env python
# encoding: utf-8


import warnings
warnings.simplefilter("ignore",DeprecationWarning)
from pylab import cm, figure
import numpy as np
np.seterr(invalid='ignore', divide='ignore') # ignore division by zero errors
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
from luts import Idx
#from  mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear



def semi_polar(lut, index=None, vmin=None, vmax=None, rect='211', sub='212',
               sym=None, swap=False, fig=None):
    '''
    Contour and eventually transect of 2D LUT on a semi polar plot, with
    dimensions (angle, radius)

    lut: 2D look-up table to display
    index: index of the item to transect in the 'angle' dimension
           if None (default), no transect
    vmin, vmax: range of values
                default None: determine min/max from values
    rect: subplot position of the main plot ('111' for example)
    sub: subplot position of the transect
    sym: the transect uses symmetrical axis (boolean)
         if None (default), use symmetry iff axis is 'zenith'
    swap: swap the order of the 2 axes to (radius, angle)
    fig : destination figure. If None (default), create a new figure.
    '''

    #
    # initialization
    #

    assert lut.ndim == 2

    show_sub = index is not None
    if fig is None:
        if show_sub:
            fig = figure(figsize=(4.5, 4.5))
        else:
            fig = figure(figsize=(4.5, 6))

    if swap:
        ax1, ax2 = lut.axes[1], lut.axes[0]
        name1, name2 = lut.names[1], lut.names[0]
    else:
        ax1, ax2 = lut.axes[0], lut.axes[1]
        name1, name2 = lut.names[0], lut.names[1]

    if vmin is None:
        vmin = np.amin(lut.data)
    if vmax is None:
        vmax = np.amax(lut.data)
    if vmin == vmax:
        vmin -= 0.001
        vmax += 0.001
    if vmin > vmax: vmin, vmax = vmax, vmin

    #
    # semi polar axis
    #
    if 'azimu' in name1.lower():
        ax1_min, ax1_max = 0.,180.
        ax1_ticks = [0, 45, 90, 135, 180]
        label1 = r'$\phi$'
    else:
        ax1_min, ax1_max = ax1[0], ax1[-1]
        ax1_ticks = np.linspace(ax1_min, ax1_max, 5)
        label1 = name1

    if 'zenit' in name2.lower():
        ax2_min, ax2_max = 0, 90.
        if sym is None: sym=True
        ax2_ticks = [0, 30, 60, 90]
        if sym: ax2_ticks = [-90, -60, -30, 0, 30, 60, 90]
        label2 = r'$\theta$'
    else:
        ax2_min, ax2_max = ax2[0], ax2[-1]
        ax2_ticks = np.linspace(ax2_min, ax2_max, 5)
        if sym is None: sym=False
        if sym:
            ax2_ticks = list(-ax2_ticks[1:][::-1]) + list(ax2_ticks)
        label2 = name2

    # 1st axis
    grid_locator1 = FixedLocator(ax1_ticks )
    tick_formatter1 = DictFormatter(dict(zip(ax1_ticks,
            map(lambda x: r'${}$'.format(x), ax1_ticks))))

    # 2nd axis
    grid_locator2 = FixedLocator(ax2_ticks)
    tick_formatter2 = DictFormatter(dict(zip(ax2_ticks,
            map(lambda x: r'${}$'.format(x), ax2_ticks))))

    tr_rotate = Affine2D().translate(0, 0)  # orientation
    tr_scale = Affine2D().scale(np.pi/ax1_max, 1.)  # scale to radians

    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

    grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                    extremes=(ax1_min, ax1_max, ax2_min, ax2_max),
                                    grid_locator1=grid_locator1,
                                    grid_locator2=grid_locator2,
                                    tick_formatter1=tick_formatter1,
                                    tick_formatter2=tick_formatter2,
                            )

    ax_polar = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax_polar)

    # adjust axis
    ax_polar.grid(True)
    ax_polar.axis["left"].set_axis_direction("bottom")
    ax_polar.axis["right"].set_axis_direction("top")
    ax_polar.axis["bottom"].set_visible(False)
    ax_polar.axis["top"].set_axis_direction("bottom")
    ax_polar.axis["top"].toggle(ticklabels=True, label=True)
    ax_polar.axis["top"].major_ticklabels.set_axis_direction("top")
    ax_polar.axis["top"].label.set_axis_direction("top")

    ax_polar.axis["top"].label.set_text(label1)
    ax_polar.axis["left"].axes.text(0.25, -0.03, label2,
                                    transform=ax_polar.transAxes,
                                   ha='center',
                                   va='top')

    # create a parasite axes whose transData in RA, cz
    aux_ax_polar = ax_polar.get_aux_axes(tr)

    aux_ax_polar.patch = ax_polar.patch # for aux_ax to have a clip path as in ax
    ax_polar.patch.zorder=0.9 # but this has a side effect that the patch is
                        # drawn twice, and possibly over some other
                        # artists. So, we decrease the zorder a bit to
                        # prevent this.

    #
    # initialize the cartesian axis below the semipolar
    #
    if show_sub:
        ax_cart = fig.add_subplot(sub)
        if sym:
            ax_cart.set_xlim(-ax2_max, ax2_max)
        else:
            ax_cart.set_xlim(ax2_min, ax2_max)
        ax_cart.set_ylim(vmin, vmax)
        ax_cart.grid(True)
        ax_cart.set_xticks(ax2_ticks)

    #
    # draw colormesh
    #
    cmap = cm.jet
    cmap.set_under('black')
    cmap.set_over('white')
    cmap.set_bad('0.5') # grey 50%
    r, t = np.meshgrid(ax2, ax1)
    if swap:
        data = np.swapaxes(lut.data, 0, 1)
    else:
        data = lut.data
    masked_data = np.ma.masked_where(np.isnan(data) | np.isinf(data), data)
    im = aux_ax_polar.pcolormesh(t, r, masked_data, cmap=cmap, vmin=vmin, vmax=vmax)

    if show_sub:
        
        # convert Idx instance to index if necessarry
        if isinstance(index, Idx):
            index = int(round(index.index(ax1)))

        # draw line over colormesh
        vertex0 = np.array([[0,0],[ax1[index],ax2_max]])
        vertex1 = np.array([[0,0],[ax1[-1-index],ax2_max]])
        aux_ax_polar.plot(vertex0[:,0],vertex0[:,1], 'w')
        if sym:
            aux_ax_polar.plot(vertex1[:,0],vertex1[:,1],'w--')

        #
        # plot transects
        #
        ax_cart.plot(ax2, data[index,:],'k-')
        if sym:
            ax_cart.plot(-ax2, data[-1-index,:],'k--')

    # add colorbar
    cb = fig.colorbar(im, orientation='horizontal', extend='both', ticks=np.linspace(vmin, vmax, 5))
    if lut.desc is not None:
        cb.set_label(lut.desc)
