#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sensor creation and visualization helpers for the SMART-G 3D
atmosphere mode.

The 3D atmosphere itself is built with the Atm3D, Cloud3D (module
smartg.atmosphere) and Grid3D (module smartg.grid3d) classes.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from luts.luts import Idx

from smartg.grid3d import locate_voxel_index
from smartg.smartg import Sensor


class OOMFormatter(ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%2.2f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format

def find_order(mat):
    return math.floor(math.log(np.max(np.abs(mat)), 10))

def find_order_or_none(mat, cb_sform):
    if cb_sform : return OOMFormatter(find_order(mat))
    else        : return None

def get_tv(vmin, vmax, mat):
    if vmin is None: vmintv = np.min(mat)
    else: vmintv = vmin
    if vmax is None: vmaxtv = np.max(mat)
    else: vmaxtv = vmax
    tv = np.linspace(vmintv, vmaxtv, 9, endpoint=True)
    return tv

def satellite_view(mlut, xgrid, ygrid, wl, interp_name='none',
                   color_bar='Blues_r', color_reverse=False, fig_size=(8,8), font_size=int(18),
                   vmin = None, vmax = None, scale=False, save_file=None, stk="I", factor=1,
                   mat_force=None, cb_shrink=0.9, cb_sform = True, fig_title=None,
                   xlim=None, ylim=None):
    """
    Description: The function give a 'satellite' 2D image of the SMART-G 3D atm return results

    ===Parameters:
    mlut         : SMART-G return MLUT object
    xgrid, ygrid : Numpy array with grid profil in the x and y axes
    wl           : Wavelength
    interp_name  : Interpolations for imshow/matshow, i.e. nearest, bilinear, bicubic, ...
    color_bar    : Bar color of the figure, i.g 'Blues_r', 'jet', ...
    fig_size     : A tuple with width and height of the figure in inches
    vmin, vmax   : Color bar interval
    scale        : If True scale between 0 and 1 (or between vmin and vmax if not None)
    font_size    : Font size of the figure
    save_file    : If not None, save the generated image in pdf, i.g. save_file = 'test', save as 'test.pdf'
    mat_force    : Force matrix = mat_force (can be a list of matrix, max len = 4)
    cb_shrink    : Color bar shrink value
    cb_sform     : Use scientific form for color bar values
    fig_title    : add a title to the figure
    """

    if not isinstance(stk, list):
        stk = [stk]

    stokes_name = []
    for stokes in stk:
        # Choose between I, Q, U and V
        if (stokes == "I"):
            stokes_name.append('I_up (TOA)')
        elif (stokes == "Q"):
            stokes_name.append('Q_up (TOA)')
        elif (stokes == "U"):
            stokes_name.append('U_up (TOA)')
        elif (stokes == "V"):
            stokes_name.append('V_up (TOA)')
        else: 
            raise NameError("Unknown stk!")
        

    Nx = xgrid.size-1; Ny = ygrid.size-1 # Number of sensors in x and y axis
    if mat_force is None:
        # First check the Azimuth and Zenith angles dimensions exist, if not the case return an error message
        if ("Azimuth angles" or "Zenith angles") not in mlut[stokes_name[0]].names:
            raise NameError("The Azimuth angles and/or Zenith angles dimension(s) are/is missing")

        axis_number = len(mlut[stokes_name[0]].names)

        ind = [slice(None)]*axis_number # if axis_number = 3, tuple(ind) equivalent to [:,:,:]

        # Two last indices for axis Azimuth angles and Zenith angles forced to 0 (consider we have only one sun position)
        ind[-1] = 0; ind[-2] = 0 # TODO Consider also the case where several sun position are given

        # if Azimuth dim or Zenith dim > 1 -> return an error message. TODO to remove once the option above is added
        if ((mlut.axes["Azimuth angles"].size or mlut.axes["Zenith angles"].size) > 1):
            raise NameError("Dimension size > 1 is not authorized for both Azimuth and Zenith angles")

        # If we have the wavelength dimension
        if "wavelength" in mlut[stokes_name[0]].names:
            ind[-3] = Idx(wl) # TODO Enable a default value, for example for the monochromatique case

        if "sensor index" in mlut[stokes_name[0]].names:
            sensor_number = mlut.axes["sensor index"].size
        else:
            sensor_number = int(1)
    else:
        sensor_number = int(mat_force[0].shape[0]*mat_force[0].shape[1])

    # Check if the product of Nx and Ny is equal to the number of sensors
    if (Nx*Ny != sensor_number):
        raise NameError("The product of Nx and Ny must be equal to the number of sensors!")

    # Convert the 1D results to a 2D matrix. The order of Nx and Ny below is very important!
    if (mat_force is None):
        matrix = []
        for name in stokes_name:
            matrix.append(mlut[name][tuple(ind)].reshape(Ny,Nx)*factor)
    else:
        matrix = mat_force
        if not isinstance(matrix, list): matrix = [matrix]
    # The variable matrix is now in the following form:
    # - x0 ... xn
    # y0
    #  :
    # yn

    if not isinstance(vmin, list): vmin = [vmin]
    if not isinstance(vmax, list): vmax = [vmax]
    if len(vmin) == 1: vmin = [vmin[0], vmin[0], vmin[0], vmin[0]]
    if len(vmax) == 1: vmax = [vmax[0], vmax[0], vmax[0], vmax[0]]

    # Deal with all the possibilties where vmin, vmax and scale are used
    if scale:
        for idm, mat in enumerate (matrix):
            vmin_scale = 0.; vmax_scale = 1.
            if vmin[idm] is not None : vmin_scale = vmin[idm]
            if vmax[idm] is not None : vmax_scale = vmax[idm]
            matrix[idm] = np.interp(mat, (mat.min(), mat.max()), (vmin_scale, vmax_scale))

    plt.rcParams.update({'font.size':font_size})
    if not isinstance(color_bar, list): color_bar = [color_bar]
    if not isinstance(color_reverse, list): color_reverse = [color_reverse]
    if len(color_bar) == 1: color_bar = [color_bar[0], color_bar[0], color_bar[0], color_bar[0]]
    if len(color_reverse) == 1: color_reverse = [color_reverse[0], color_reverse[0], color_reverse[0], color_reverse[0]]
    cmaps = []
    for idcb, cbar in enumerate(color_bar):
        if not isinstance(cbar, str):
            cmaps.append(cbar)
        else:
            cmaps.append(plt.get_cmap(cbar))
        if (color_reverse[idcb]): cmaps[idcb] = plt.get_cmap(cbar).reversed()
        cmaps[idcb].set_bad('white',1.)

    if len(matrix) == 1:
        if fig_size is None: fig_size = (6,4)
        plt.figure(figsize=fig_size, constrained_layout=True)
        if fig_title is not None : plt.title(fig_title)


        # By default in the imshow function, the origin (origin='upper') i.e matrix[0,0] is at the upper left,
        # and we want the origin at bottom left (origin='lower).
        if (is_same_cell_size(xgrid) and is_same_cell_size(ygrid)):
            img = plt.imshow(matrix[0], vmin=vmin[0], vmax=vmax[0], origin='lower', cmap=cmaps[0],
                             interpolation=interp_name, extent=[xgrid.min(),xgrid.max(),ygrid.min(),ygrid.max()])
        else:
            if (interp_name != 'none'):
                print("Warning: the interp_name variable cannot be used (and then ignored) when using pcolormesh!" + 
                " i.e. when we have a cell size varying along the x or y axis.")
            img = plt.pcolormesh(xgrid, ygrid, matrix[0], vmin=vmin[0], vmax=vmax[0], cmap=cmaps[0])
            plt.axis('scaled') # x and y axes with the same scaling
        
        cbar = plt.colorbar(img, shrink=cb_shrink, orientation='vertical',
                            format=find_order_or_none(matrix[0][~np.isnan(matrix[0])], cb_sform),
                            ticks=get_tv(vmin[0], vmax[0], matrix[0][~np.isnan(matrix[0])]))
        cbar.set_label(r''+ stokes_name[0], fontsize = font_size)

        if xlim is not None: plt.xlim(xlim[0], xlim[1])
        if ylim is not None: plt.ylim(ylim[0], ylim[1])
        plt.xlabel(r'X (km)')
        plt.ylabel(r'Y (km)')

    elif len(matrix) == 2:
        if fig_size is None: fig_size = (12,4)
        fig, axs = plt.subplots(1,2, figsize=fig_size, constrained_layout=True, sharex=True, sharey=True)
        if fig_title is not None : fig.suptitle(fig_title)

        cax1 = axs[0].imshow(matrix[0], vmin=vmin[0], vmax=vmax[0], origin='lower', cmap=cmaps[0],
                    interpolation=interp_name, extent=[xgrid.min(),xgrid.max(),ygrid.min(),ygrid.max()])
        
        cbar1 = fig.colorbar(cax1, ax=axs[0], shrink=cb_shrink, orientation='vertical', format=find_order_or_none(matrix[0], cb_sform), ticks=get_tv(vmin[0], vmax[0], matrix[0]))
        cbar1.set_label(r''+ stokes_name[0], fontsize = font_size)

        cax2 = axs[1].imshow(matrix[1], vmin=vmin[1], vmax=vmax[1], origin='lower', cmap=cmaps[1],
                    interpolation=interp_name, extent=[xgrid.min(),xgrid.max(),ygrid.min(),ygrid.max()])
        
        cbar2 = fig.colorbar(cax2, ax=axs[1], shrink=cb_shrink, orientation='vertical', format=find_order_or_none(matrix[1], cb_sform), ticks=get_tv(vmin[1], vmax[1], matrix[1]))
        cbar2.set_label(r''+ stokes_name[1], fontsize = font_size)

        axs[0].set_xlim(xgrid[0], xgrid[-1])
        axs[1].set_ylim(ygrid[0], ygrid[-1])

        axs[0].set_ylabel(r'Y (km)')
        fig.supxlabel(r'X (km)')

    elif len(matrix) == 3:
        if fig_size is None: fig_size = (12,8)
        fig = plt.figure(figsize=fig_size)
        gs = gridspec.GridSpec(4, 4, figure=fig)
        if fig_title is not None : fig.suptitle(fig_title)

        ax1 = plt.subplot(gs[:2, :2])
        cax1 = ax1.imshow(matrix[0], vmin=vmin[0], vmax=vmax[0], origin='lower', cmap=cmaps[0],
                        interpolation=interp_name, extent=[xgrid.min(),xgrid.max(),ygrid.min(),ygrid.max()])
        cbar1 = plt.colorbar(cax1, ax=ax1, shrink=cb_shrink, orientation='vertical', format=find_order_or_none(matrix[0], cb_sform), ticks=get_tv(vmin[0], vmax[0], matrix[0]))
        cbar1.set_label(r''+ stokes_name[0], fontsize = font_size)
        ax1.set_xlim(xgrid[0], xgrid[-1])

        ax2 = plt.subplot(gs[:2, 2:], sharey=ax1)
        plt.setp(ax2.get_yticklabels(), visible=False)
        cax2 = ax2.imshow(matrix[1], vmin=vmin[1], vmax=vmax[1], origin='lower', cmap=cmaps[1],
                        interpolation=interp_name, extent=[xgrid.min(),xgrid.max(),ygrid.min(),ygrid.max()])
        cbar2 = plt.colorbar(cax2, ax=ax2, shrink=cb_shrink, orientation='vertical', format=find_order_or_none(matrix[1], cb_sform), ticks=get_tv(vmin[1], vmax[1], matrix[1]))
        cbar2.set_label(r''+ stokes_name[1], fontsize = font_size)
        ax2.set_xlim(xgrid[0], xgrid[-1])

        ax3 = plt.subplot(gs[2:4, 1:3])
        cax3 = ax3.imshow(matrix[2], vmin=vmin[2], vmax=vmax[2], origin='lower', cmap=cmaps[2],
                        interpolation=interp_name, extent=[xgrid.min(),xgrid.max(),ygrid.min(),ygrid.max()])
        cbar3 = plt.colorbar(cax3, ax=ax3, shrink=cb_shrink, orientation='vertical', format=find_order_or_none(matrix[2], cb_sform), ticks=get_tv(vmin[2], vmax[2], matrix[2]))
        cbar3.set_label(r''+ stokes_name[2], fontsize = font_size)
        ax3.set_xlim(xgrid[0], xgrid[-1])

        ax1.set_ylabel(r'Y (km)')
        ax3.set_ylabel(r'Y (km)')
        ax3.set_xlabel(r'X (km)')

        gs.tight_layout(fig)


    elif len(matrix) == 4:
        if fig_size is None: fig_size = (12,8)
        fig, axs = plt.subplots(2,2, figsize=fig_size, constrained_layout=True, sharex=True, sharey=True)
        if fig_title is not None : fig.suptitle(fig_title)
        plt.rcParams.update({'font.size':font_size})
        for i in range (0, 2):
            for j in range (0, 2):
                cax = axs[i,j].imshow(matrix[i*2+j], vmin=vmin[i*2+j], vmax=vmax[i*2+j], origin='lower', cmap=cmaps[i*2+j],
                        interpolation=interp_name, extent=[xgrid.min(),xgrid.max(),ygrid.min(),ygrid.max()])
                cbar = fig.colorbar(cax, ax=axs[i,j], shrink=cb_shrink, orientation='vertical', format=find_order_or_none(matrix[i*2+j], cb_sform), ticks=get_tv(vmin[i*2+j], vmax[i*2+j], matrix[i*2+j]))
                cbar.set_label(r''+ stokes_name[i*2+j], fontsize = font_size)
                #axs[i,j].set_xlim(xgrid[0], xgrid[-1])
                if xlim is not None: axs[i,j].set_xlim(xlim[0], xlim[1])
                if ylim is not None: axs[i,j].set_ylim(ylim[0], ylim[1])

        fig.supxlabel(r'X (km)')
        fig.supylabel(r'Y (km)')
    else:
        raise NameError("Give more than 4 stk is not authorized!")
    
    # If the option save_file is used, save the file in pdf
    if (save_file is not None):
        # Deal with the case where we have not specified the '.pdf' at the end
        if ( not save_file.endswith('.pdf') and (not save_file.endswith('.png')) ):
            save_file += '.pdf'
        plt.savefig(save_file)



def get_sensors_pos_icells_from_3Dgrid(grid3D, POSZ):
    g = grid3D
    # Arrays with sizes of cells in x and y axes
    sizes_x = (g.xgrid[1:] - g.xgrid[:-1])/2.
    sizes_y = (g.ygrid[1:] - g.ygrid[:-1])/2.

    # x and y sensors position into the 3Dgrid
    x0  = g.xgrid[:-1] + sizes_x
    y0  = g.ygrid[:-1] + sizes_y

    xx,yy  = np.meshgrid(x0, y0)
    zz     = np.zeros_like(xx) + POSZ
    icells = locate_voxel_index(g.xGRID, g.yGRID, g.zGRID, xx.ravel(), yy.ravel(), zz.ravel())

    return x0, y0, xx, yy, icells

def find_id(val, grid):
    id = None
    for i in range (0, len(grid)-1):
        if (val > grid[i] and val <= grid[i+1]): id = i
    if id is None: raise NameError("An id cannot be found!")
    return id

def create_sensors(grid3D, POSZ=120., THDEG=180., PHDEG=180., FOV=0., TYPE=0., LOC='ATMOS', CELL_SIZE=-1, grid3D_atm=None):
        """
        Description : Create a list of sensors

        === Parameters:
        grid3D       : A Grid3D class
        POSZ         : Altitude where the sensors will be placed
        THDEG, PHDEG : viewing angles
        FOV          : Field of View (deg, default 0.)
        TYPE         : Radiance (0), Planar flux (1), Spherical Flux (2), default 0
        LOC          : Localization (default ATMOS)
        grid3D_atm   : To know the true intial bbox of sensors in case we have a different grid for atm

        === Return:
        List of Sensor classes
        """
        # TODO consider a possible variability between sensors (positions, viewing angles, etc.)
        x0, y0, xx, yy, icells = get_sensors_pos_icells_from_3Dgrid(grid3D, POSZ)

        sensors=[]
        for POSX,POSY,ICELL in zip(xx.ravel(), yy.ravel(), icells):
            sensors.append(Sensor(POSX=POSX, POSY=POSY, POSZ=POSZ, FOV=FOV, TYPE=TYPE,
                                  THDEG=THDEG, PHDEG=PHDEG, LOC=LOC, ICELL=ICELL, CELL_SIZE=CELL_SIZE))
            
        if grid3D_atm is not None:
            _, _, xx_atm, yy_atm, icells_atm = get_sensors_pos_icells_from_3Dgrid(grid3D_atm, POSZ)
            sensors_atm=[]
            for POSX,POSY,ICELL in zip(xx_atm.ravel(), yy_atm.ravel(), icells_atm):
                sensors_atm.append(Sensor(POSX=POSX, POSY=POSY, POSZ=POSZ, FOV=FOV, TYPE=TYPE,
                                          THDEG=THDEG, PHDEG=PHDEG, LOC=LOC, ICELL=ICELL, CELL_SIZE=CELL_SIZE))
            sensors_new = []
            for isens in range (0, len(sensors)):
                posx = sensors[isens].dict['POSX'].copy()
                idx = find_id(posx, grid3D_atm.xgrid)
                posy = sensors[isens].dict['POSY'].copy()
                idy = find_id(posy, grid3D_atm.ygrid)
                sens_tmp = Sensor()
                sens_tmp.dict = sensors_atm[idx+grid3D_atm.Nx*idy].dict.copy()
                sens_tmp.dict['POSX'] = posx
                sens_tmp.dict['POSY'] = posy
                sens_tmp.cell_size = sensors_atm[idx+grid3D_atm.Nx*idy].cell_size
                sensors_new.append(sens_tmp)

            # replace now the sensor list by the corrected one
            sensors = sensors_new.copy()
        
        return x0, y0, sensors, icells
