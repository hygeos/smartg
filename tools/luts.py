#!/usr/bin/env python
# encoding: utf-8

'''
Several tools for look-up tables management and interpolation

Provides:
    - LUT class: extends ndarrays for generic multi-dimensional interpolation
    - Idx class: find the index of values, for LUT interpolation
    - merge: look-up tables merging
    - read_lut_hdf: read LUTs from HDF files
'''

import numpy as np
from pyhdf.SD import SD, SDC
from scipy.interpolate import interp1d
from os.path import exists
from os import remove
from collections import OrderedDict, Iterable

class CMN_MLUT_LUT(object):
    '''
    Base class for MLUTs and LUTS
    Should not be used as-is
    '''
    def __add__(self, other):
        return self.__binary_operation__(other, lambda x, y: x+y)

    def __radd__(self, other):
        return self.__binary_operation__(other, lambda x, y: x+y)

    def __sub__(self, other):
        return self.__binary_operation__(other, lambda x, y: x-y)

    def __rsub__(self, other):
        return self.__binary_operation__(other, lambda x, y: y-x)

    def __mul__(self, other):
        return self.__binary_operation__(other, lambda x, y: x*y)

    def __rmul__(self, other):
        return self.__binary_operation__(other, lambda x, y: x*y)

    def __div__(self, other):
        return self.__binary_operation__(other, lambda x, y: x/y)

    def __rdiv__(self, other):
        return self.__binary_operation__(other, lambda x, y: y/x)

    def __eq__(self, other):
        return self.equal(other)

    def __neq__(self, other):
        return not self.__eq__(other)


class LUT(CMN_MLUT_LUT):
    '''
    Look-up table storage with generic multi-dimensional interpolation.
    Extends the __getitem__ method of ndarrays to float and float arrays (index
    tables with floats)
    The LUT axes can be optionally provided so that values can be interpolated
    into float indices in a first step, using the Idx class.

    Arguments:
        * data is a n-dimension array containing the LUT data
        * axes is a list representing each dimension, containing for each
          dimension:
            - a 1-d array or list containing the tabulated values of the dimension
            - or None, if there are no values associated with this dimension
        * names is an optional list of names for each axis (default: None)
          it is necessary for storing the LUT
        * attr is a dictionary of additional attributes, useful for merging LUTs
        * desc is a string describing the parameter stored (default None)
          when using save(), desc is used as the hdf dataset name to store the LUT

    Attributes: axes, shape, data, ndim, attrs, names, desc

    Example 1
    ---------
    Basic usage, without axes and in 1D:
    >>> data = np.arange(10)**2
    >>> L = LUT(data)
    >>> L[1], L[-1], L[::2]    # standard indexing is possible
    (1, 81, array([ 0,  4, 16, 36, 64]))
    >>> L[1.5]    # Indexing with a float: interpolation
    2.5
    >>> L[np.array([[0.5, 1.5], [2.5, 9.]])]    # interpolate several values at once
    array([[  0.5,   2.5],
           [  6.5,  81. ]])

    Example 2
    ---------
    Interpolation of the atmospheric pressure
    >>> z = np.linspace(0, 120., 80)
    >>> P0 = np.linspace(980, 1030, 6)
    >>> Pdata = P0.reshape(1,-1)*np.exp(-z.reshape(-1,1)/8) # dimensions (z, P0)

    A 2D LUT with attached axes.  Axes names can optionally be provided.
    >>> P = LUT(Pdata, axes=[z, P0], names=['z', 'P0'])
    >>> P[Idx(8.848), Idx(1013.)]  # standard pressure at mount Everest
    336.09126751112842
    >>> z = np.random.rand(50, 50)  # now z is a 2D array of elevations between 0 and 1 km
    >>> P0 = 1000+20*np.random.rand(50, 50)  # and P0 is a 2D array of pressures at z=0
    >>> P[Idx(z), Idx(P0)].shape    # returns a 2D array of corresponding pressures
    (50, 50)

    In Idx, the values can optionally be passed using keyword notation.
    In this case, there is a verification that the argument corresponds to the right axis name.
    >>> P[:, Idx(1013., 'P0')].shape   # returns the (shape of) standard vertical pressure profile
    (80,)
    '''

    def __init__(self, data, axes=None, names=None, desc=None, attrs=None):
        self.data = data
        self.desc = desc
        if attrs is None:
            self.attrs = {}
        else:
            self.attrs = attrs
        self.ndim = self.data.ndim
        self.shape = data.shape
        self.sub = Subsetter(self)

        # check axes
        if axes is None:
            self.axes = self.ndim * [None]
        else:
            self.axes = axes
            assert len(axes) == self.ndim
            for ax in axes:
                if isinstance(ax, np.ndarray):
                    assert ax.ndim == 1
                elif isinstance(ax, list): pass
                elif ax is None: pass
                else:
                    raise Exception('Invalid axis type {}'.format(ax.__class__))

        # check names
        if names is None:
            self.names = self.ndim * [None]
        else:
            self.names = names
            assert len(names) == self.ndim

    def axis(self, a):
        '''
        returns axis referred to by a (string or integer)
        '''
        if isinstance(a, str):
            index = self.names.index(a)
            return self.axes[index]
        elif isinstance(a, int):
            return self.axes[a]
        else:
            raise TypeError('argument of LUT.axis() should be int or string')

    def print_info(self, prepend=''):
        if self.desc is None:
            print prepend+'LUT ({}):'.format(self.data.dtype)
        else:
            print prepend+'LUT "{}" ({}):'.format(self.desc, self.data.dtype)

        for i in xrange(self.data.ndim):
            if self.names[i] is None:
                name = 'NoName'
            else:
                name = self.names[i]
            if self.axes[i] is None:
                print prepend+'  Dim {} ({}): No axis attached'.format(i, name)
            else:
                print prepend+'  Dim {} ({}): {} values betweeen {} and {}'.format(
                        i, name,
                        len(self.axes[i]),
                        self.axes[i][0],
                        self.axes[i][-1])

    def __getitem__(self, keys):
        '''
        Get items from the LUT, with possible interpolation

        Indexing works mostly like a standard array indexing, with the following differences:
            - float indexes can be provided, they will result in an
              interpolation between bracketing integer indices for this dimension
            - float arrays can be provided, they will result in an
              interpolation between bracketing integer arrays for this dimension
            - Idx objects can be provided, they are first converted into indices
            - basic indexing and slicing, and advanced indexing (indexing with
              an ndarray of int or bool) can still be used
              (see http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)
            - if arrays are passed as arguments, they must all have the same shape
            - Unlike ndarrays, the number of dimensions in keys should be
              identical to the dimension of the LUT
              >>> LUT(np.zeros((2, 2)))[:]
              Traceback (most recent call last):
              ...
              Exception: Incorrect number of dimensions in __getitem__

        Returns: a scalar or ndarray
        '''

        if not isinstance(keys, tuple):
            keys = (keys,)
        keys = list(keys)
        N = len(keys)

        if N != self.ndim:
            raise Exception('Incorrect number of dimensions in __getitem__')


        # convert the Idx keys to float indices
        for i in xrange(N):
            k = keys[i]
            if isinstance(k, Idx):
                # k is an Idx instance
                # convert it to float indices for the current axis
                if k.name not in [None, self.names[i]]:
                    msg = 'Error, wrong parameter passed at position {}, expected {}, got {}'
                    raise Exception(msg.format(i, self.names[i], k.name))
                keys[i] = k.index(self.axes[i])

        # determine the dimensions of the result (for broadcasting coef)
        dims_array = None
        dims_result = []
        for i in xrange(N):
            k = keys[i]
            if isinstance(k, np.ndarray):
                if dims_array is None:
                    dims_array = k.shape
                    dims_result.extend([slice(None)]*k.ndim)
                else:
                    assert dims_array == k.shape, 'LUTS.__getitem__: all arrays must have same shape ({} != {})'.format(str(dims_array), str(k.shape))
            else:
                dims_result.append(None)

        # determine the interpolation axes
        # and for those axes, determine the lower index (inf) and the weight
        # (x) between lower and upper index
        interpolate_axis = []   # indices of the interpolated axes
        inf_list = []       # index of the lower elements for the interpolated axes
        x_list = []         # weights, for the interpolated axes
        for i in xrange(N):
            k = keys[i]

            # floating-point indices should be interpolated
            interpolate = False
            if isinstance(k, np.ndarray) and (k.dtype in [np.dtype('float')]):
                interpolate = True
                inf = k.astype('int')
                inf[inf == self.data.shape[i]-1] -= 1
                x = k-inf
                if k.ndim > 0:
                    x = x.__getitem__(dims_result)
            elif isinstance(k, float):
                interpolate = True
                inf = int(k)
                if inf == self.data.shape[i]-1:
                    inf -= 1
                x = k-inf
            if interpolate:
                # current axis needs interpolation
                inf_list.append(inf)
                x_list.append(x)
                interpolate_axis.append(i)

        # loop over the 2^n bracketing elements
        # (cartesian product of [0, 1] over n dimensions)
        n = len(interpolate_axis)
        result = 0
        for b in xrange(2**n):

            # coefficient attributed to the current item
            # and adjust the indices
            # for the interpolated dimensions
            coef = 1
            for i in xrange(n):
                # bb is the ith bit in b (0 or 1)
                bb = ((1<<i)&b)>>i
                x = x_list[i]
                if bb:
                    coef *= x
                else:
                    coef *= 1-x

                keys[interpolate_axis[i]] = inf_list[i] + bb

            result += coef * self.data[tuple(keys)]

        return result


    def equal(self, other, strict=True, show_diff=True):
        assert isinstance(other, LUT)
        return self.to_mlut().equal(other.to_mlut(), strict=strict, show_diff=show_diff)

    def __binary_operation__(self, other, fn):
        if isinstance(other, LUT):
            return fn(self.to_mlut(), other.to_mlut())[0]
        else:
            return fn(self.to_mlut(), other)[0]

    def to_mlut(self):
        '''
        convert to a MLUT
        '''
        m = MLUT()

        # axes
        if self.axes is not None:
            for i in xrange(len(self.axes)):
                name = self.names[i]
                axis = self.axes[i]
                if (name is None) or (axis is None):
                    continue
                m.add_axis(name, axis)

        # datasets
        m.add_dataset(self.desc, self.data, axnames=self.names)

        # attributes
        m.set_attrs(self.attrs)

        return m

    def save(self, filename, **kwargs):
        return self.to_mlut().save(filename, **kwargs)


    def plot(self, *args, **kwargs):
        if self.ndim == 1:
            self.__plot_1d()
        elif self.ndim == 2:
            return semi_polar(self, *args, **kwargs)
        else:
            raise Exception('No plot defined for {} dimensions'.format(self.ndim))


    def __plot_1d(self):
        '''
        plot a 1-dimension LUT
        '''
        from pylab import plot, xlabel, ylabel, grid
        x = self.axes[0]
        y = self.data
        plot(x, y)
        if self.names[0] is not None:
            xlabel(self.names[0])
        if self.desc is not None:
            ylabel(self.desc)
        grid()


class Subsetter(object):
    '''
    A conveniency class to use the syntax like:
    LUT.sub[:,:,0]
    for subsetting LUTs
    '''
    def __init__(self, LUT):
        self.LUT = LUT

    def __getitem__(self, keys):
        '''
        subset parent LUT
        '''
        axes = []
        names = []
        attrs = self.LUT.attrs
        desc = self.LUT.desc

        for i in xrange(self.LUT.ndim):
            if keys[i] == slice(None):
                axes.append(self.LUT.axes[i])
                names.append(self.LUT.names[i])

        data = self.LUT.__getitem__(keys)

        return LUT(data, axes=axes, names=names, attrs=attrs, desc=desc)


class Idx(object):
    '''
    Calculate the indices of values by interpolation in a LUT axis
    The index method is typically called when indexing a dimension of a LUT
    object by a Idx object.
    The round attribute (boolean) indicates whether the resulting index should
    be rounded to the closest integer.

    Example: find the float index of 35. in an array [0, 10, ..., 100]
    >>> Idx(35.).index(np.linspace(0, 100, 11))
    array(3.5)

    Find the indices of several values in the array [0, 10, ..., 100]
    >>> Idx(np.array([32., 45., 72.])).index(np.linspace(0, 100, 11))
    array([ 3.2,  4.5,  7.2])

    Optionally, the name of the parameter can be provided as a keyword
    argument.
    Example: Idx(3., 'a') instead of Idx(3.)
    This allows verifying that the parameter is used in the right axis.
    '''
    def __init__(self, value, name=None, round=False, bounds_error=True, fill_value=np.NaN):
        if value is not None:
            self.value = value
            self.name = name
            self.round = round
            self.bounds_error = bounds_error
            self.fill_value = fill_value

    def index(self, axis):
        '''
        Return the floating point index of the values in the axis
        '''
        if len(axis) == 1:
            if not np.allclose(np.array(self.value), axis[0]):
                raise ValueError("(Idx) Out of axis value (value={}, axis={})".format(self.value, axis))
            return 0
        else:
            # axis is scalar or ndarray: interpolate
            res = interp1d(axis, np.arange(len(axis)),
                    bounds_error=self.bounds_error,
                    fill_value=self.fill_value)(self.value)
            if self.round:
                if isinstance(res, np.ndarray):
                    res = res.round().astype(int)
                else:
                    res = round(res)
            return res


def semi_polar(lut, index=None, vmin=None, vmax=None, rect='211', sub='212',
               sym=None, swap=False, fig=None, cmap=None):
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
    from pylab import figure, cm
    from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
    from matplotlib.transforms import Affine2D
    from mpl_toolkits.axisartist import floating_axes
    from matplotlib.projections import PolarAxes

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
        data = np.swapaxes(lut.data, 0, 1)
    else:
        ax1, ax2 = lut.axes[0], lut.axes[1]
        name1, name2 = lut.names[0], lut.names[1]
        data = lut.data

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
    ax1_ticks = [0, 45, 90, 135, 180]
    if 'azimu' in name1.lower():
        ax1_min, ax1_max = 0., 180.
        ax1_ticks = dict(zip(ax1_ticks, map(str, ax1_ticks)))
        label1 = r'$\phi$'
        ax1_scaled = ax1
    else:
        ax1_min, ax1_max = ax1[0], ax1[-1]
        ax1_ticks = dict(zip(ax1_ticks,
                         map(lambda x: '{:.1f}'.format(x), np.linspace(ax1_min, ax1_max, len(ax1_ticks)))))
        label1 = name1

        # rescale ax1 to (0, 180)
        ax1_scaled = (ax1-ax1_min)/(ax1_max-ax1_min)*180.

    ax2_ticks = [0, 30, 60, 90]
    if 'zenit' in name2.lower():
        ax2_min, ax2_max = 0, 90.
        if sym is None: sym=True
        ax2_ticks = dict(zip(ax2_ticks, map(str, ax2_ticks)))
        label2 = r'$\theta$'
        ax2_scaled = ax2
    else:
        ax2_min, ax2_max = ax2[0], ax2[-1]
        ax2_ticks = dict(zip(ax2_ticks,
                             map(lambda x: '{:.1f}'.format(x), np.linspace(ax2_min, ax2_max, len(ax2_ticks)))))
        if sym is None: sym=False
        label2 = name2

        # rescale to (0, 90)
        ax2_scaled = (ax2-ax2_min)/(ax2_max-ax2_min)*90.

    # 1st axis
    grid_locator1 = FixedLocator(ax1_ticks.keys())
    tick_formatter1 = DictFormatter(ax1_ticks)

    # 2nd axis
    grid_locator2 = FixedLocator(ax2_ticks.keys())
    tick_formatter2 = DictFormatter(ax2_ticks)

    tr_rotate = Affine2D().translate(0, 0)  # orientation
    tr_scale = Affine2D().scale(np.pi/180., 1.)  # scale to radians

    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

    grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                    extremes=(0., 180., 0., 90.),
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

    ax_polar.axis["top"].axes.text(0.70, 0.92, label1,
                                   transform=ax_polar.transAxes,
                                   ha='left',
                                   va='bottom')
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

    #
    # draw colormesh
    #
    if cmap is None:
        cmap = cm.jet
        cmap.set_under('black')
        cmap.set_over('white')
        cmap.set_bad('0.5') # grey 50%
    r, t = np.meshgrid(ax2_scaled, ax1_scaled)
    masked_data = np.ma.masked_where(np.isnan(data) | np.isinf(data), data)
    im = aux_ax_polar.pcolormesh(t, r, masked_data, cmap=cmap, vmin=vmin, vmax=vmax)

    if show_sub:
        # convert Idx instance to index if necessarry
        if isinstance(index, Idx):
            index = int(round(index.index(ax1)))

        # draw line over colormesh
        vertex0 = np.array([[0,0],[ax1_scaled[index],ax2_max]])
        vertex1 = np.array([[0,0],[ax1_scaled[-1-index],ax2_max]])
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
    fig.colorbar(im, orientation='horizontal', extend='both', ticks=np.linspace(vmin, vmax, 5))
    if lut.desc is not None:
        ax_polar.set_title(lut.desc, weight='bold', position=(0.15,0.9))


def merge(L, axes):
    '''
    Merge several luts or mluts

    Arguments:
        - L is a list of LUT or MLUT objects to merge
        - axes is a list of axes names to merge
          these names should be present in each LUT attribute

    Returns a LUT or MLUT for which each dataset has new axes as defined in list axes
    (list of strings)
    The attributes of the merged mlut consists of the common attributes with
    identical values.

    Example:

    >>> np.random.seed(0)

    create four 2D LUTs with identical axes and 2 attributes
    >>> ax1 = np.arange(4)     # 0..3
    >>> ax2 = np.arange(5)+10  # 10..14
    >>> L1 = LUT(np.random.rand(4, 5), axes=[ax1, ax2], names=['C','D'], attrs={'A':11, 'B': 1})
    >>> L2 = LUT(np.random.rand(4, 5), axes=[ax1, ax2], names=['C','D'], attrs={'A':11, 'B': 2})
    >>> Lmerged = merge([L1, L2], ['B'])  # merge 2 luts (1 new dimension 'B')
    >>> Lmerged.shape, Lmerged.attrs    # should contain 'A', which has not been merged
    ((2, 4, 5), OrderedDict([('A', 11)]))

    create 2 more LUTs and merge 4 luts over 2 dimensions
    merge them and create two new axes 'A' and 'B', using the attributes
    provided in each LUT attributes
    >>> L3 = LUT(np.random.rand(4, 5), axes=[ax1, ax2], names=['C','D'], attrs={'A':10, 'B': 1})
    >>> L4 = LUT(np.random.rand(4, 5), axes=[ax1, ax2], names=['C','D'], attrs={'A':10, 'B': 2})
    >>> Lmerged = merge([L1, L2, L3, L4], ['A', 'B'])
    >>> Lmerged.shape
    (2, 2, 4, 5)

    Merge two MLUTS using attribute to LUT promotion
    >>> M1 = MLUT()
    >>> M1.add_dataset('a', np.arange(10))
    >>> M1.set_attrs({'b':1, 'c':10})
    >>> M1.promote_attr('b')  # attribute 'a' is converted to a scalar LUT
    >>> M2 = MLUT()
    >>> M2.add_dataset('a', np.arange(10))
    >>> M2.set_attrs({'b':2, 'c':20})
    >>> M2.promote_attr('b')
    >>> merged = merge([M1, M2], ['c'])
    >>> merged.print_info(show_self=False)
     Datasets:
      [0] a (float64 between 0 and 9), axes=('c', None)
      [1] b (float64 between 1 and 2), axes=('c', None)
     Axes:
      [0] c: 2 values between 10 and 20
    '''

    # LUT merging: convert to mlut
    if isinstance(L[0], LUT):
        return merge(map(lambda x: x.to_mlut(), L), axes)[0]

    # else: MLUT merging

    m = MLUT()
    first = L[0]

    # check mluts compatibility
    for i in xrange(1, len(L)):
        assert first.equal(L[i], strict=False)

    # add old axes
    for (axname, axis) in first.axes.items():
        m.add_axis(axname, axis)

    # determine the new axes from the attributes of all mluts
    newaxes = []  # new axes
    newaxnames = []
    for axname in axes:
        axis = []
        for mlut in L:
            value = mlut.attrs[axname]
            if value not in axis:
                axis.append(value)
        m.add_axis(axname, axis)
        newaxes.append(axis)
        newaxnames.append(axname)

    # dataset loop
    for i in xrange(len(first.datasets())):

        # build new data
        new_shape = tuple(map(len, newaxes))+first.data[i][1].shape
        dtype = first.data[i][1].dtype
        newdata = np.zeros(new_shape, dtype=dtype)+np.NaN
        for mlut in L:

            # find the index of the attributes in the new LUT
            index = ()
            for j in xrange(len(axes)):
                a = axes[j]
                index += (newaxes[j].index(mlut.attrs[a]),)
            index += (slice(None),)

            newdata[index] = mlut.data[i][1]

        name = first.data[i][0]
        axnames = first.data[i][2]
        if axnames is None:
            m.add_dataset(name, newdata)
        else:
            m.add_dataset(name, newdata, newaxnames+axnames)

    # fill with common arguments
    for k, v in first.attrs.items():
        if False in map(lambda x: k in x.attrs, L):
            continue
        if isinstance(v, np.ndarray):
            if False in map(lambda x: np.allclose(v, x.attrs[k]), L):
                continue
        else:
            if False in map(lambda x: v == x.attrs[k], L):
                continue
        m.set_attr(k, v)

    return m


class MLUT(CMN_MLUT_LUT):
    '''
    A class to store and manage multiple look-up tables

    How to create a MLUT:
    >>> m = MLUT()
    >>> m.add_axis('a', np.linspace(100, 150, 5))
    >>> m.add_axis('b', np.linspace(5, 8, 6))
    >>> m.add_axis('c', np.linspace(0, 1, 7))
    >>> np.random.seed(0)
    >>> m.add_dataset('data1', np.random.randn(5, 6), ['a', 'b'])
    >>> m.add_dataset('data2', np.random.randn(5, 6, 7), ['a', 'b', 'c'])
    >>> # Add a dataset without associated axes
    >>> m.add_dataset('data3', np.random.randn(10, 12))
    >>> m.set_attr('x', 12)   # set MLUT attributes
    >>> m.set_attrs({'y':15, 'z':8})
    >>> m.print_info(show_self=False)
     Datasets:
      [0] data1 (float64 between -2.55 and 2.27), axes=('a', 'b')
      [1] data2 (float64 between -2.22 and 2.38), axes=('a', 'b', 'c')
      [2] data3 (float64 between -2.77 and 2.3), axes=(None, None)
     Axes:
      [0] a: 5 values between 100.0 and 150.0
      [1] b: 6 values between 5.0 and 8.0
      [2] c: 7 values between 0.0 and 1.0

    Use bracket notation to extract a LUT
    Note that you can use a string or integer.
    data1 is the first dataset in this case, we could use m[0]
    >>> m['data1'].print_info()  # or m[0]
    LUT "data1" (float64):
      Dim 0 (a): 5 values betweeen 100.0 and 150.0
      Dim 1 (b): 6 values betweeen 5.0 and 8.0

    Binary operations are possible between two MLUTs or a MLUT and a float or
    int
    >>> (m*2)['data1'][0,0] == m['data1'][0,0]*2
    True

    MLUTs can be stored using the save(filename) method.
    '''
    def __init__(self):
        # axes
        self.axes = OrderedDict()
        # data: a list of (name, array, axnames)
        self.data = []
        # attributes
        self.attrs = OrderedDict()

    def datasets(self):
        ''' returns a list of the datasets names '''
        return map(lambda x: x[0], self.data)

    def add_axis(self, name, axis):
        ''' Add an axis to the MLUT '''
        assert isinstance(name, str)
        if isinstance(axis, list):
            ax = np.array(axis)
        else:
            ax = axis
        assert ax.ndim == 1

        self.axes[name] = ax

    def add_dataset(self, name, dataset, axnames=None):
        ''' Add a dataset to the LUT '''
        assert name not in map(lambda x: x[0], self.data)
        if axnames is not None:
            # check axes consistency
            assert len(axnames) == dataset.ndim
            for i, ax in enumerate(axnames):
                if ax is None: continue
                assert ax in self.axes
                assert dataset.shape[i] == len(self.axes[ax])
        else:
            axnames = [None]*dataset.ndim

        self.data.append((name, dataset, axnames))

    def save(self, filename, overwrite=False, verbose=False):
        '''
        Save a MLUT to a hdf file
        '''
        typeconv = {
                    np.dtype('float32'): SDC.FLOAT32,
                    np.dtype('float64'): SDC.FLOAT64,
                    }
        if exists(filename):
            if overwrite:
                remove(filename)
            else:
                ex = Exception('File {} exists'.format(filename))
                setattr(ex, 'filename', filename)
                raise ex

        if verbose:
            print 'Writing "{}" to "{}"'.format(self.desc, filename)
        hdf = SD(filename, SDC.WRITE | SDC.CREATE)

        # write axes
        if self.axes is not None:
            for name, ax in self.axes.items():
                if verbose:
                    print '   Write axis "{}" in "{}"'.format(name, filename)
                type = typeconv[ax.dtype]
                sds = hdf.create(name, type, ax.shape)
                sds.setcompress(SDC.COMP_DEFLATE, 9)
                sds[:] = ax[:]
                sds.endaccess()

        # write datasets
        for name, data, axnames in self.data:
            if verbose:
                print '   Write data "{}"'.format(name)
            type = typeconv[data.dtype]
            sds = hdf.create(name, type, data.shape)
            sds.setcompress(SDC.COMP_DEFLATE, 9)
            sds[:] = data[:]
            if axnames is not None:
                setattr(sds, 'dimensions', ','.join(map(str, axnames)))
            sds.endaccess()

        # write attributes
        if verbose:
            print '   Write {} attributes'.format(len(self.attrs))
        for k, v in self.attrs.items():
            setattr(hdf, k, v)

        hdf.end()

    def set_attr(self, key, value):
        self.attrs[key] = value

    def set_attrs(self, attributes):
        self.attrs.update(attributes)

    def print_info(self, show_range=True, show_self=True):
        if show_self:
            print str(self)
        print ' Datasets:'
        for i, (name, dataset, axes) in enumerate(self.data):
            if axes is None:
                axdesc = ''
            else:
                axdesc = ', axes='+ str(tuple(axes))
            if show_range:
                rng = ' between {:.3g} and {:.3g}'.format(np.amin(dataset), np.amax(dataset))
            else:
                rng = ''
            print '  [{}] {} ({}{})'.format(i, name, dataset.dtype, rng, dataset.shape) + axdesc
        print ' Axes:'
        for i, (name, values) in enumerate(self.axes.items()):
            print '  [{}] {}: {} values between {} and {}'.format(i, name, len(values), values[0], values[-1])

    def promote_attr(self, name):
        '''
        Create a new dataset from attribute name
        '''
        assert isinstance(name, str)
        assert name in self.attrs
        value = self.attrs[name]
        if isinstance(value, Iterable):
            value = np.array(value)
        else:
            value = np.array([value])

        self.add_dataset(name, value)

    def __getitem__(self, key):
        '''
        return the LUT corresponding to key (int or string)
        '''
        if isinstance(key, str):
            index = -1
            self.data
            for i, (name, _, _) in enumerate(self.data):
                if key == name:
                    index = i
                    break
            if index == -1:
                raise Exception('Cannot find dataset {}'.format(key))
        elif isinstance(key, int):
            index = key
        else:
            raise Exception('multi-dimensional LUTs should only be indexed with strings or integers')

        name, dataset, axnames = self.data[index]
        if axnames is None:
            axes = None
        else:
            axes = []
            names = []
            for ax in axnames:
                if ax is None:
                    axes.append(None)
                else:
                    axes.append(self.axes[ax])
                names.append(ax)

        return LUT(desc=name, data=dataset, axes=axes, names=names, attrs=self.attrs)

    def __binary_operation_scalar__(self, other, fn):
        '''
        Binary operation between a MLUT and a scalar
        '''
        m = MLUT()
        m.attrs = self.attrs
        m.axes = self.axes

        for i, (name, data, axnames) in enumerate(self.data):
            m.add_dataset(name, fn(data, other), axnames=axnames)

        return m

    def __binary_operation_mlut__(self, other, fn):
        '''
        Binary operation between two MLUTS
        '''
        m = MLUT()

        # check and include axes
        for ax in self.axes:
            assert ax in other.axes
            assert len(self.axes[ax]) == len(other.axes[ax])
            assert np.allclose(self.axes[ax], other.axes[ax])
            m.add_axis(ax, self.axes[ax])

        # include datasets
        assert len(self.data) == len(other.data),  'Binary operation between inconsistent MLUTs'
        for i, (name, data, axnames) in enumerate(self.data):
            m.add_dataset(name, fn(data, other.data[i][1]), axnames=axnames)

        # include common attributes
        for k in self.attrs:
            # check that the attributes are equal
            if not (k in other.attrs):
                continue
            if isinstance(self.attrs[k], np.ndarray):
                if not isinstance(other.attrs[k], np.ndarray):
                    continue
                if not np.allclose(self.attrs[k], other.attrs[k]):
                    continue
            else:
                if self.attrs[k] != other.attrs[k]:
                    continue

            m.set_attr(k, self.attrs[k])
        return m

    def __binary_operation__(self, other, fn):
        if isinstance(other, MLUT):
            return self.__binary_operation_mlut__(other, fn)
        else:
            return self.__binary_operation_scalar__(other, fn)

    def equal(self, other, strict=True, show_diff=False):
        '''
        Test equality between two MLUTs
        Arguments:
         * show_diff: print their differences
         * strict:
            True: use strict equality
            False: MLUTs compatibility but not strict equality
                -> same axes
                -> same datasets names and shapes, but not content
                -> attributes may be different
        '''
        msg = 'MLUTs diff:'
        if not isinstance(other, MLUT):
            msg += '  other is not a MLUT'
            print msg
            return False

        eq = True

        # check axes
        for k in set(self.axes).union(other.axes):
            if (k not in other.axes) or (k not in self.axes):
                msg += '  axis {} missing in either\n'.format(k)
                eq = False
            if self.axes[k].shape != other.axes[k].shape:
                msg += '  axis {} shape mismatch\n'.format(k)
                eq = False
            if not np.allclose(self.axes[k], other.axes[k]):
                msg += '  axis {} is different\n'.format(k)
                eq = False

        # check datasets
        for i, (name0, data0, axnames0) in enumerate(self.data):
            name1, data1, axnames1 = other.data[i]
            if ((name0 != name1)
                    or (strict and not np.allclose(data0, data1))
                    or (axnames0 != axnames1)):
                msg += '  dataset {} is different\n'.format(name0)
                eq = False

        # check attributes
        if strict:
            for a in set(self.attrs.keys()).union(other.attrs.keys()):
                if (a not in self.attrs) or (a not in other.attrs):
                    msg += '  attribute {} missing in either MLUT\n'.format(a)
                    eq = False
                    continue
                if (self.attrs[a] != other.attrs[a]):
                    msg += '  value of attribute {} differs ({} and {})\n'.format(a, self.attrs[a], other.attrs[a])
                    eq = False
                    continue

        if show_diff and not eq:
            print msg

        return eq


def read_mlut_hdf(filename, datasets=None):
    '''
    read a MLUT from a hdf file (filename)
    datasets: list of datasets to read:
        * None (default): read all datasets, including axes as indicated by the
          attribute 'dimensions'
        * a list of:
            - dataset names (string)
            - or a tuple (dataset_name, axes) where axes is a list of
              dimensions (strings), overriding the attribute 'dimensions'
    '''
    hdf = SD(filename)

    # read the datasets
    ls_axes = []
    ls_datasets = []
    if datasets is None:
        datasets = xrange(len(hdf.datasets()))

    for i in datasets:
        if isinstance(i, tuple):
            (name, axes) = i
            sds = hdf.select(name)
        else:
            axes = None
            sds = hdf.select(i)
        sdsname = sds.info()[0]

        if (axes is None) and ('dimensions' in sds.attributes()):
            axes = sds.attributes()['dimensions'].split(',')
            axes = map(lambda x: x.strip(), axes)

            # replace 'None's by None
            axes = map(lambda x: {True: None, False: x}[x == 'None'], axes)

        if axes is not None:
            ls_axes.extend(axes)

        ls_datasets.append((sdsname, sds.get(), axes))

    # remove 'None' axes
    while None in ls_axes:
        ls_axes.remove(None)

    # transfer the axes from ls_datasets to the new MLUT
    m = MLUT()
    for ax in set(ls_axes):

        # read the axis of not done already
        if ax not in map(lambda x: x[0], ls_datasets):
            sds = hdf.select(ax)
            m.add_axis(ax, sds.get())
        else:
            i = map(lambda x: x[0], ls_datasets).index(ax)
            (name, data, axnames) = ls_datasets.pop(i)
            m.add_axis(name, data)

    # add the datasets
    for (name, data, axnames) in ls_datasets:
        m.add_dataset(name, data, axnames)

    # read the attributes
    for k, v in hdf.attributes().items():
        m.set_attr(k, v)

    return m


def read_lut_hdf(filename, dataset, axnames=None):
    '''
    read a hdf file as a LUT, using axis list axnames
    if axnames is None, read the axes names in the attribute 'dimensions' of dataset
    '''
    # TODO: deprecate this one
    axes = []
    names = []
    data = None
    dimensions = None

    hdf = SD(filename)

    # load dataset
    if dataset in hdf.datasets():
        sds = hdf.select(dataset)
        data = sds.get()
        if 'dimensions' in sds.attributes():
            dimensions = sds.attributes()['dimensions'].split(',')
            dimensions = map(lambda x: x.strip(), dimensions)
    else:
        print 'dataset "{}" not available.'.format(dataset)
        print '{} contains the following datasets:'.format(filename)
        for d in hdf.datasets():
            print '  *', d
        raise Exception('Missing dataset')

    if axnames == None:
        assert dimensions is not None, 'Error, dimension names have not been provided'
        axnames = dimensions
    else:
        if dimensions is not None:
            assert axnames == dimensions, 'Error in dimensions, expected {}, found {}'.format(axnames, dimensions)

    # load axes
    for d in axnames:
        if not d in hdf.datasets():
            axes.append(None)
            names.append(d)
            continue

        sds = hdf.select(d)
        (sdsname, rank, shp, dtype, nattr) = sds.info()

        assert rank == 1

        axis = sds.get()
        axes.append(axis)
        names.append(sdsname)

    assert data.ndim == len(axes)
    for i in xrange(data.ndim):
        assert (axes[i] is None) or (len(axes[i]) == data.shape[i])

    # read global attributes
    attrs = {}
    for a in hdf.attributes():
        attrs.update({a: hdf.attributes()[a]})

    return LUT(data, axes=axes, names=names, desc=dataset, attrs=attrs)





if __name__ == '__main__':
    import doctest
    doctest.testmod()
