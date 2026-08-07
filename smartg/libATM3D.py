#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from luts.luts import Idx, MLUT, LUT, from_xarray
import pandas as pd
from pathlib import Path
import os

from smartg.atmosphere import Atm1D, Cloud, od2k
from smartg.smartg import Sensor
from smartg.grid3d import (
    Grid3D,
    Get_3Dcells,
    Get_3Dcells_indices,
    Get_3Dcells_neighbours,
    locate_3Dregular_cells,
    locate_voxel_index,
    create_1d_grid,
    extend_1d_grid,
    is_sorted,
    is_same_cell_size,
)
from smartg.phase import read_cld_nth_cte

import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from smartg.config import DIR_AUXDATA
from luts.luts import read_mlut

import xarray as xr

from warnings import warn
import math


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

class Cloud3D(object):
    """
    Description: Represent a 3D cloud profil

    === Attribut:
    filename             : Cloud smartg filename, choice between: wc, ic_baum_asc, ic_baum_ghm and ic_baum_sc
    w_ref                : Reference wavelength of ext_ref or of the IPRT file
    reff                 : Numpy 1D array with the cloud effective radii at wavelength w_ref
    ext_ref              : Numpy 1D array with the cloud extinction coefficient at wavelength w_ref
    xyz_grids            : List with in the indices 0, 1 and 2 the 1D arrays with respectively the x, y and z grid profils
    cell_indices         : Numpy 3D array with the cloud xyz indices
    ext_reff_filename    : File name with path location. File following the convention of IPRT cloud files
                           with exctintion coefficient and reff of each cloud cell.
    reff_acc             : Interger with decimal accuracy of reff. By default None then do not replace the read reff values
    reff_min, reff_max   : The reff values less than reff_min will replaced by reff_min. The same for values greater than reff_max
    phase                : LUT object with the cloud phase Matrix depending on wl, reff, stk, and theta

    === Be careful !! If reff, ext_ref, xyz_grids or cell_indices are given circumvent variables read from ext_reff_filename !!

    """

    def __init__(self, filename, w_ref,
                 reff=None, ext_ref=None, xyz_grids=None, cell_indices=None,
                 ext_reff_filename=None,
                 reff_acc = None, reff_min = None, reff_max = None,
                 phase=None):
        
        filename = Path(filename)
        if filename.parent == Path('.'): filename = Path(DIR_AUXDATA) / 'clouds' / filename

        if "_sol" not in filename.name and filename.suffix != ".nc":
            filename = filename.with_name(filename.stem + "_sol.nc")
        elif filename.suffix != ".nc":
            filename = filename.with_name(filename.name + ".nc")

        if not filename.exists():
            raise FileNotFoundError(f"{filename} does not exist")
        
        self.filename = filename
        self.cld_mlut = read_mlut(self.filename)
        self.w_ref = w_ref

        # Check (if not set to None) that the file exists and is readable
        if (ext_reff_filename is None):
            self.ext_reff_filename = None
        elif (not Path(ext_reff_filename).exists()):
            raise NameError("The given file does not exists!")
        elif (not os.access(Path(ext_reff_filename), os.R_OK)):
            raise NameError("The given file cannot be read!")
        else:
            self.ext_reff_filename = Path(ext_reff_filename)
        
        # If file_name is None, all other variables must be specified
        if  ( (ext_reff_filename is None)
              and ( reff is None or ext_ref is None or xyz_grids is None or cell_indices is None ) ):
            raise NameError("If file_name is set to None reff, ext_ref, xyz_grids and cell_indices must be given!")

        # TODO adds checks on the varaibles bellow (if we have np.arrays, ...)
        self.xyz_grids    = xyz_grids
        self.ext_ref    = ext_ref
        self.cell_indices = cell_indices
        self.reff         = reff
        self.reff_acc     = reff_acc
        self.reff_min     = reff_min
        self.reff_max     = reff_max

        if (phase is None):
            self.phase = phase
        # Check if phase is a LUT object with the correct axes
        elif (not isinstance(phase, LUT)):
            raise NameError("phase must be a LUT object!")
        elif (not all([item in phase.names for item in ['wav_phase', 'reff', 'stk', 'theta_atm']])):
            raise NameError("Phase matrix must have 4 dimensions: wav_phase, reff, stk and theta_atm")
        else:
            self.phase = phase


    def get_xyz_grid(self, loc_xgrid = "centered", loc_ygrid = "centered"):
        """
        Description: Get the x, y and z 1D grid profils.

        In case xyz_grids is None and IPRT_filename is provided, we can choose where to place the x and y grids by
        specifiying the values of loc_xgrid and loc_ygrid -->

        loc_xgrid, loc_ygrid : Grid location. By default an str: "centered" i.e. the grid center is at coordinate 0.
                               Or give a scalar with the starting position of the grid.
        === Return:
        xgrid, ygrid, zgrid : Three 1D arrays with the x, y and z grid profils.
        """

        # First check if we have already the xyz grids
        if (self.xyz_grids is not None): return self.xyz_grids[0], self.xyz_grids[1], self.xyz_grids[2]

        # Read only the needed information, the two first rows.
        # Be careful ! The second row have a greater dimension than the first one. Then -> two steps of reading.
        contentA = pd.read_csv(self.ext_reff_filename, skiprows = 1, nrows = 1, header=None, sep=r'\s+', dtype=float).values
        contentB = pd.read_csv(self.ext_reff_filename, skiprows = 2, nrows = 1, header=None, sep=r'\s+', dtype=float).values

        # If there are empty dimensions remove them
        contentA = np.squeeze(contentA)
        contentB = np.squeeze(contentB)

        # Number of cells in x and y axes
        Nx = int(contentA[0])
        Ny = int(contentA[1])

        # Cell sizes in x and y axes
        Dx = contentB[0]
        Dy = contentB[1]

        # Create x and y grid
        xgrid = create_1d_grid(Nx, Dx, loc=loc_xgrid)
        ygrid = create_1d_grid(Ny, Dy, loc=loc_ygrid)

        # Grid in the z axis can be directly read from the file
        zgrid = contentB[2:]

        return xgrid, ygrid, zgrid

    def get_ext_ref(self):

        # First check if we have already the ext_coeff
        if (self.ext_ref is not None) : return self.ext_ref

        # Read only the disired column
        ext_ref = pd.read_csv(self.ext_reff_filename, skiprows = 3, header=None, usecols=[3], sep=r'\s+', dtype=float).values

        # If there are empty dimensions remove them
        ext_ref = np.squeeze(ext_ref)

        return ext_ref

    def get_cell_indices(self):

        # First check if we have already the cell_indices
        if (self.cell_indices is not None): return self.cell_indices

        # Read only the disired column
        cell_indices = pd.read_csv(self.ext_reff_filename, skiprows = 3, header=None, usecols=[0,1,2], sep=r'\s+', dtype=float).values

        # If there are empty dimensions remove them and ensure that we have interger type
        cell_indices = np.squeeze(cell_indices.astype(np.int32))

        # We need to have 2 dimensions, a numpy array as list of another numpy arrays with the x, y and z indices
        if(cell_indices.ndim == 1): cell_indices = np.array([cell_indices])

        return cell_indices

    def get_reff(self):

        # First check if we have already reff
        if (self.reff is not None): return self.reff

        # Read only the disired column
        reff = pd.read_csv(self.ext_reff_filename, skiprows = 3, header=None, usecols=[4], sep=r'\s+', dtype=float).values
        if self.reff_acc is not None: reff = np.around(reff, decimals=self.reff_acc)

        # If there are empty dimensions remove them
        reff = np.squeeze(reff)

        if self.reff_min is not None: reff[reff<self.reff_min ] = self.reff_min
        if self.reff_max is not None: reff[reff>self.reff_max ] = self.reff_max

        return reff
    
    def get_phase(self, n_theta=721, conv_Iparper=True):

        # First check if we have already phase
        if (self.phase is not None):
            if (len(self.phase.axes[3]) == n_theta):
                return self.phase
            else:
                theta = np.linspace(0., 180., n_theta)
                return self.phase.sub()[:, :, :, Idx(theta)]

        theta = np.linspace(0., 180., n_theta)
        pha = self.cld_mlut['phase'].swapaxes('reff', 'wav')[:,:,:,Idx(theta)]
        nwav = pha.shape[0]
        nreff = pha.shape[1]
        nstklut = pha.shape[2]

        pha_ = np.zeros((nwav, nreff, 6, n_theta), dtype=np.float64)
        pha_[:,:,:nstklut,:] = pha

        P = LUT(pha_, axes=[self.cld_mlut.axes['wav'], self.cld_mlut.axes['reff'], np.arange(6), theta],
                names=['wav_phase', 'reff', 'stk', 'theta_atm']) 

        if conv_Iparper:
            if (nstklut == 4): # spherical particles
                P.data[:,:,4,:] = P.data[:,:,0,:].copy()
                P.data[:,:,5,:] = P.data[:,:,2,:].copy()
                P0 = P.data[:,:,0,:].copy()
                P1 = P.data[:,:,1,:].copy()
                P4 = P.data[:,:,4,:].copy()
                P.data[:,:,0,:] = 0.5*(P0+2*P1+P4) # P11
                P.data[:,:,1,:] = 0.5*(P0-P4)      # P12=P21
                P.data[:,:,4,:] = 0.5*(P0-2*P1+P4) # P22
            elif (nstklut == 6): # non spherical particles
                # note: the sign of P43/P34 affects only the sign of V,
                # since V=0 for rayleigh scattering it does not matter 
                P0 = P.data[:,:,0,:].copy()
                P1 = P.data[:,:,1,:].copy()
                P4 = P.data[:,:,4,:].copy()
                P.data[:,:,0,:] = 0.5*(P0+2*P1+P4) # P11
                P.data[:,:,1,:] = 0.5*(P0-P4)      # P12=P21
                P.data[:,:,4,:] = 0.5*(P0-2*P1+P4) # P22

        return P



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

    
class Atm3D(object):
    """
    In progress...
    If wls.size is equal to 1 and wl_ref is None, take wls as reference wavelength
    """

    def __init__(self, atm_filename, grid_3d, wls, comp=[], cloud_3d=None, wl_ref = None,
    lat=45, P0=None, O3=None, H2O=None, NO2=True, tauR=None, mol_sca_1d=None, mol_abs_1d=None,
    aer_ext_1d=None, aer_ssa_1d = None, nth_aer_1d=721, phase_aer_1d=None):

        possible_atm_filename = ['afglms', 'afglmw', 'afglss', 'afglsw', 'afglt', 'afglus']

        if (atm_filename not in possible_atm_filename):
            raise NameError('Unknown atmosphere file name!')

        if (not isinstance(grid_3d, Grid3D)):
            raise NameError('grid_3d variable must be a Grid3D object!')

        if (cloud_3d is None):
            self.cloud_3d      = None
            self.cld_reff      = None
            self.cld_ext_ref   = None
            self.cloud_indices = None
        elif (not isinstance(cloud_3d, Cloud3D)):
            raise NameError('cloud_3d variable must be a Cloud3D object!')
        else:
            self.cloud_3d      = cloud_3d
            self.cld_reff      = cloud_3d.get_reff()
            self.cld_ext_ref   = cloud_3d.get_ext_ref()
            self.cloud_indices = cloud_3d.get_cell_indices() # update if performed bellow

            # === Cell indices in SMART-G + consider boundaries:
            # The "-1" is here because IPRT input cloud file indices start at 1 intead of 0 for SMART-G
            cloud_indices_new = self.cloud_indices-1
            # Look if we have xy boundaries
            # Below we look on the x axis, but works also if we check on the y axis
            is_boundaryxy = grid_3d.Nx < grid_3d.NX

            # Update the cloud_indices (for the x and y axes) if they are xy boundaries
            if (is_boundaryxy): cloud_indices_new[:,:2]+=1

            # Finally update the attribut
            self.cloud_indices = cloud_indices_new

        if (not isinstance(wls, np. ndarray)):
            raise NameError('wls must be a numpy array!')
        
        if (wls.ndim > 1):
            raise NameError('wls must be a 1d numpy array!')

        if (wls.size > 1 and wl_ref is None):
            raise NameError('wl_ref must be specified if wls contains more than 1 wavelength!')

        self.atm_filename = atm_filename
        self.grid_3d      = grid_3d
        self.wls          = wls
        if (wls.size == 1 and wl_ref is None): self.wl_ref   = wls[0]
        else                                 : self.wl_ref   = wl_ref

        # Calculate the 1d optical properties
        self.ipha_aer_1d = None
        self.pha_aer_1d  = None
        if ( (mol_sca_1d is None) or    # if molecular abs is not given
             (mol_abs_1d is None) or    # if rayleigh coeff is not given
             (aer_ext_1d is None) or    # if aer coeff is not given
             (aer_ssa_1d is None)       or    # 
             (phase_aer_1d is None) ):          # if aer_ssa is not given

            if len(comp) > 0 : pha_ = True # compute aer phase only if a list of aer is given
            else             : pha_ = False

            znew = grid_3d.zGRID[::-1]
            if pha_ and comp[0].phase is not None : zpf = [100, 0]
            else                                  : zpf = znew

            atm_1d = Atm1D(atm_filename, comp=comp, lat=lat, p0=P0, tco3=O3, tcwp=H2O, no2=NO2,
                             tau_r=tauR, grid=znew, pfgrid=zpf).calc(wls, phase=pha_, n_theta=nth_aer_1d)
            self.ssa_aer_1d = atm_1d['ssa_p_atm']
            if (pha_):
                self.ipha_aer_1d = atm_1d['iphase_atm']
                self.pha_aer_1d  = atm_1d['phase_atm']    
    
        if mol_sca_1d is not None : self.molecular_sca_1d = mol_sca_1d
        else                      : self.molecular_sca_1d = od2k(atm_1d, 'OD_r')
        
        if mol_abs_1d is not None : self.molecular_abs_1d = mol_abs_1d
        else                      : self.molecular_abs_1d = od2k(atm_1d, 'OD_g')

        if aer_ext_1d is not None : self.ext_aer_1d = aer_ext_1d
        else                      : self.ext_aer_1d = od2k(atm_1d, 'OD_p')   

        if phase_aer_1d is not None:
            self.ipha_aer_1d = phase_aer_1d[0] 
            self.pha_aer_1d  = phase_aer_1d[1] 

        if aer_ssa_1d is not None : self.ssa_aer_1d = aer_ssa_1d
        else                      : self.ssa_aer_1d = atm_1d['ssa_p_atm'][:,:]

        # Atm1D.calc returns xarray objects, while the 3D optical properties are
        # numpy arrays and LUT. The 1d aerosol ones are converted once here, so
        # that the get_glob_aer_* methods can mix the 1d and the 3d ones.
        if isinstance(self.pha_aer_1d, xr.DataArray):
            self.pha_aer_1d = from_xarray(self.pha_aer_1d)
        for attr_name in ['ipha_aer_1d', 'ext_aer_1d', 'ssa_aer_1d']:
            attr = getattr(self, attr_name)
            if isinstance(attr, xr.DataArray): setattr(self, attr_name, attr.values)
        # ===

    def get_glob_molecular_sca(self):
        """
        Get global (1d+3d) molecular scattering (Rayleigh) coefficients
        2 dimensions : wavelength, iopt (number 1d layers + unique 3d cells)
        """
        if (self.cloud_3d is None):
            molecular_sca_glob = self.molecular_sca_1d
        else:
            cloud_indices = self.cloud_indices

            # 1d xyz indices where there are clouds
            cloud_1d_indices = np.ravel_multi_index((cloud_indices[:,0], cloud_indices[:,1], cloud_indices[:,2]),
                                                     dims=(self.grid_3d.NX, self.grid_3d.NY, self.grid_3d.NZ))

            # calculate the 3d rayleigh extinction coefficient
            molecular_sca_glob = np.concatenate([self.molecular_sca_1d,
                self.molecular_sca_1d[:,self.grid_3d.NZ-self.grid_3d.idz[cloud_1d_indices]]], axis=1)

        return molecular_sca_glob

    def get_glob_molecular_abs(self):
        """
        Get global (1d+3d) molecular absorption coefficients
        2 dimensions : wavelength, iopt (number 1d layers + unique 3d cells)
        """

        if (self.cloud_3d is None):
            mol_abs_glob = self.molecular_abs_1d
        else: # if there are clouds
            cloud_indices = self.cloud_indices

            # 1d xyz indices where there are clouds
            cloud_1d_indices = np.ravel_multi_index((cloud_indices[:,0], cloud_indices[:,1], cloud_indices[:,2]),
                                                    dims=(self.grid_3d.NX, self.grid_3d.NY, self.grid_3d.NZ))

            # calculate the 3d molecular coefficient
            mol_abs_glob = np.concatenate([self.molecular_abs_1d,
                self.molecular_abs_1d[:,self.grid_3d.NZ-self.grid_3d.idz[cloud_1d_indices]]], axis=1)

        return mol_abs_glob

    def get_grid(self):

        if( self.cloud_3d is None):
            Nopt = self.grid_3d.NZ + 1
        else:
            nb_unique_cells = self.cloud_indices.shape[0]
            Nopt = self.grid_3d.NZ + 1 + nb_unique_cells

        return np.arange(Nopt)
    
    
    def get_glob_aer_ext_ssa(self):
        """
        Get global (1d+3d) aerosol extinction coefficients and aerosol single scattering albedos
        2 dimensions : wavelength, iopt (number 1d layers + unique 3d cells)
        """

        ext_cld = self.cld_ext_ref
        cld_reff = self.cld_reff
        w_ref = self.cloud_3d.w_ref

        wav = self.wls
        nwav = len(wav)
        n_unique_cell = len(ext_cld)
        ext_mix_3d = np.zeros((nwav, n_unique_cell), dtype=np.float64)
        ssa_mix_3d = np.ones((nwav, n_unique_cell), dtype=np.float64)


        if (self.pha_aer_1d is None):
            ext_cld_ref0 = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(w_ref)]
            for iw in range (0, nwav):
                ext_factor = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(wav[iw])]/ext_cld_ref0
                ext_mix_3d[iw,:] = ext_cld * ext_factor
                ssa_mix_3d[iw,:] = self.cloud_3d.cld_mlut['ssa'][Idx(cld_reff), Idx(wav[iw])]
        else:                    
            cld_reff = self.cld_reff
            for iwav in range (0, nwav):
                idz_atm = []
                for icell in range (0, n_unique_cell):
                    idz = self.cloud_indices[icell,2]
                    id_zatm = self.grid_3d.NZ+1 - idz
                    idz_atm.append(id_zatm)

                ext_cld_ref0 = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(w_ref)]
                ext_factor = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(wav[iwav])]/ext_cld_ref0
                ext_cld_tmp = self.cld_ext_ref * ext_factor
                ssa_cld_tmp = self.cloud_3d.cld_mlut['ssa'][Idx(cld_reff), Idx(wav[iwav])]

                ssa_aer_tmp = self.ssa_aer_1d[iwav,idz_atm]
                ext_aer_tmp = self.ext_aer_1d[iwav,idz_atm]
                ext_mix_tmp = ext_aer_tmp + ext_cld_tmp
                ext_mix_3d[iwav,:] = ext_mix_tmp
                ssa_mix_3d[iwav,:] = (ext_aer_tmp*ssa_aer_tmp + ext_cld_tmp*ssa_cld_tmp) / ext_mix_tmp

        # Create a table with only the cloud properties but in global shape i.e. for each cells
        #  not sharing the same opt prop, and other commun cells in z, following the plan parallel
        #  1D atm philosophy
        ext_aer_glob = np.concatenate([self.ext_aer_1d, ext_mix_3d], axis=1)
        ssa_aer_glob = np.concatenate([self.ssa_aer_1d[:,:], ssa_mix_3d], axis=1)

        return ext_aer_glob, ssa_aer_glob
    

    def get_glob_aer_ssa(self):
        """
        Get global (1d+3d) aerosol single scattering albedos
        2 dimensions : wavelength, iopt (number 1d layers + unique 3d cells)
        """

        ext_cld = self.cld_ext_ref
        cld_reff = self.cld_reff
        w_ref = self.cloud_3d.w_ref

        wav = self.wls
        nwav = len(wav)
        n_unique_cell = len(ext_cld)
        ssa_mix_3d = np.ones((nwav, n_unique_cell), dtype=np.float64)

        ext_cld_ref0 = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(w_ref)]

        if (self.pha_aer_1d is None):
            for iw in range (0, nwav):
                ssa_mix_3d[iw,:] = self.cloud_3d.cld_mlut['ssa'][Idx(cld_reff), Idx(wav[iw])]
        else:
            cld_reff = self.cld_reff
            for iwav in range (0, nwav):
                idz_atm = []
                for icell in range (0, n_unique_cell):
                    idz = self.cloud_indices[icell,2]
                    id_zatm = self.grid_3d.NZ+1 - idz
                    idz_atm.append(id_zatm)

                ext_cld_ref0 = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(w_ref)]
                ext_factor = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(wav[iwav])]/ext_cld_ref0
                ext_cld_tmp = self.cld_ext_ref * ext_factor
                ssa_cld_tmp = self.cloud_3d.cld_mlut['ssa'][Idx(cld_reff), Idx(wav[iwav])]

                ssa_aer_tmp = self.ssa_aer_1d[iwav,idz_atm]
                ext_aer_tmp = self.ext_aer_1d[iwav,idz_atm]
                ext_mix_tmp = ext_aer_tmp + ext_cld_tmp
                ssa_mix_3d[iwav,:] = (ext_aer_tmp*ssa_aer_tmp + ext_cld_tmp*ssa_cld_tmp) / ext_mix_tmp
                        
        # Create a table with only the cloud properties but in global shape i.e. for each cells
        #  not sharing the same opt prop, and other commun cells in z, following the plan parallel
        #  1D atm philosophy
        ssa_aer_glob = np.concatenate([self.ssa_aer_1d[:,:], ssa_mix_3d], axis=1)

        return ssa_aer_glob
    
    
    def get_glob_aer_ext(self):
        """
        Get global (1d+3d) aerosol extinction coefficients
        2 dimensions : wavelength, iopt (number 1d layers + unique 3d cells)
        """
                
        ext_cld = self.cld_ext_ref
        cld_reff = self.cld_reff
        w_ref = self.cloud_3d.w_ref

        wav = self.wls
        nwav = len(wav)
        n_unique_cell = len(ext_cld)
        ext_mix_3d = np.zeros((nwav, n_unique_cell), dtype=np.float64)


        if (self.pha_aer_1d is None):
            ext_cld_ref0 = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(w_ref)]
            for iw in range (0, nwav):
                ext_factor = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(wav[iw])]/ext_cld_ref0
                ext_mix_3d[iw,:] = ext_cld * ext_factor
        else:
            cld_reff = self.cld_reff
            for iwav in range (0, nwav):
                idz_atm = []
                for icell in range (0, n_unique_cell):
                    idz = self.cloud_indices[icell,2]
                    id_zatm = self.grid_3d.NZ+1 - idz
                    idz_atm.append(id_zatm)

                ext_cld_ref0 = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(w_ref)]
                ext_factor = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(wav[iwav])]/ext_cld_ref0
                ext_cld_tmp = self.cld_ext_ref * ext_factor

                ext_aer_tmp = self.ext_aer_1d[iwav,idz_atm]
                ext_mix_tmp = ext_aer_tmp + ext_cld_tmp
                ext_mix_3d[iwav,:] = ext_mix_tmp

        # Create a table with only the cloud properties but in global shape i.e. for each cells
        #  not sharing the same opt prop, and other commun cells in z, following the plan parallel
        #  1D atm philosophy
        ext_aer_glob = np.concatenate([self.ext_aer_1d, ext_mix_3d], axis=1)

        return ext_aer_glob
    

    def get_glob_aer_phase(self, wl_phase=None, n_theta=721, conv_Iparper=True):
        """
        Get global (1d+3d) aerosol phase matrices

        === return:
        tuple -> (ipha3D, luts)
        where luts is a list of unique phase luts (1d + 3d)
        and where ipha3D is a 2d numpy matrix(wavelength, iopt) with the lut phase index of list luts
        """

        # ===== 1) Calcul des matrices de phases
        # Get the cloud effective radii
        cld_reff = self.cld_reff
        cld_reff_unique = np.unique(cld_reff)
        nreff_unique = cld_reff_unique.size

        if wl_phase is not None: wav = wl_phase
        else: wav = self.wls
        nwav = len(wav)

        # Get the phase matrix
        # Here dim : wav,reff,stk,theta
        phase = self.cloud_3d.get_phase(n_theta=n_theta, conv_Iparper=conv_Iparper)

        luts = []

        if self.pha_aer_1d is not None: # case list of 1d aer is given
            # Dim of pha_aer: wav, stk, theta.
            # 3d phase : phase.axes[3] -> theta dim
            # 1d phase : self.pha_aer.axes[2] -> theta dim
            if (len(self.pha_aer_1d.axes[2]) == n_theta):
                phase_aer_1d = self.pha_aer_1d
            else:
                phase_aer_1d = self.pha_aer_1d.sub()[:, :, Idx(phase.axes[3])]

            # First plan parallel phase 
            for iwav in range (0, nwav):
                for iz in range (0, self.grid_3d.NZ+1):
                    luts.append(phase_aer_1d.sub()[self.ipha_aer_1d[iwav,iz], :, :])

            # Second 3d mix phase
            ext_cld = self.cld_ext_ref
            n_unique_cell = len(ext_cld)
            for iwav in range (0, nwav):
                idz_atm = []
                for icell in range (0, n_unique_cell):
                    idz = self.cloud_indices[icell,2]
                    id_zatm = self.grid_3d.NZ+1 - idz
                    idz_atm.append(id_zatm)

                ext_cld_ref0 = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(self.cloud_3d.w_ref)]
                ext_factor = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(wav[iwav])]/ext_cld_ref0
                ext_cld_tmp = self.cld_ext_ref * ext_factor
                ssa_cld_tmp = self.cloud_3d.cld_mlut['ssa'][Idx(cld_reff), Idx(wav[iwav])]

                ssa_aer_tmp = self.ssa_aer_1d[iwav,idz_atm]
                ext_aer_tmp = self.ext_aer_1d[iwav,idz_atm]
                ext_tot = ext_aer_tmp + ext_cld_tmp

                for icell in range (0, n_unique_cell):
                    pha_cld_tmp = phase.sub()[Idx(wav[iwav]),Idx(cld_reff[icell]), :, :]
                    pha_aer_tmp = phase_aer_1d.sub()[self.ipha_aer_1d[iwav,idz_atm[icell]], :, :]
                    pha_tot = ( (pha_aer_tmp*ext_aer_tmp[icell]*ssa_aer_tmp[icell]) +
                                (pha_cld_tmp*ext_cld_tmp[icell]*ssa_cld_tmp[icell]) ) / ext_tot[icell]
                    luts.append(pha_tot)

            # Third concatenate plan parallel + 3d optical prop (first without considering wl)
            nbz = self.ext_aer_1d.shape[1] # nb of z_atm
            phase_glob_indices_w0 = np.arange(nbz+n_unique_cell, dtype=np.int32)
        else: # case no 1d aer given
            for iwav in range (0, nwav):
                # Loop only on the unique cld_reff
                for ireff in range (0, nreff_unique):
                    luts.append(phase.sub()[Idx(wav[iwav]),Idx(cld_reff_unique[ireff]), :, :])

            # ===== 2) phase matrix indices to take for all cloud cells
            # Obtain the correct indices from the unique radii phase matrix
            phase_3d_indices_w0 = np.full(cld_reff.size, np.nan, dtype=np.int32)
            for ireff in range (0, nreff_unique):
                phase_3d_indices_w0[np.squeeze(np.argwhere( cld_reff == cld_reff_unique[ireff]))] = ireff

            # Concatenate plan parallel + 3d optical prop (first without considering wl)
            phase_glob_indices_w0 = np.concatenate([np.zeros(self.grid_3d.NZ+1, dtype=np.int32), phase_3d_indices_w0[:]])


        # Now consider the wl dimension
        phase_glob_indices = np.zeros((nwav, phase_glob_indices_w0.size), dtype=np.int32)

        for iwav in range (0, nwav):
            phase_glob_indices[iwav,:] = phase_glob_indices_w0[:] + (iwav*nreff_unique)

        ipha3D = phase_glob_indices

        return (ipha3D, luts)
    

    def get_glob_aer_phase_ext_ssa(self, wl_phase=None, n_theta=721, conv_Iparper=True):
        """
        Get global (1d+3d) aerosol phase matrices, extinction coefficients and single scattering albedos

        === return:
        (ipha3D, luts), ext_aer_glob, ssa_aer_glob
        where luts is a list of unique phase luts (1d + 3d)
        and where ipha3D is a 2d numpy matrix(wavelength, iopt) with the lut phase index of list luts
        """

        # ===== 1) Calcul des matrices de phases
        # Get the cloud effective radii
        cld_reff = self.cld_reff
        cld_reff_unique = np.unique(cld_reff)
        nreff_unique = cld_reff_unique.size
        w_ref = self.cloud_3d.w_ref

        if wl_phase is not None: wav = wl_phase
        else: wav = self.wls
        nwav = len(wav)

        # Get the phase matrix
        # Here dim : wav,reff,stk,theta
        phase = self.cloud_3d.get_phase(n_theta=n_theta, conv_Iparper=conv_Iparper)

        luts = []

        n_unique_cell = len(self.cld_ext_ref)
        ext_mix_3d = np.zeros((nwav, n_unique_cell), dtype=np.float64)
        ssa_mix_3d = np.ones((nwav, n_unique_cell), dtype=np.float64)

        if self.pha_aer_1d is not None: # case list of 1d aer is given
            # Dim of pha_aer: wav, stk, theta.
            # 3d phase : phase.axes[3] -> theta dim
            # 1d phase : self.pha_aer.axes[2] -> theta dim
            if (len(self.pha_aer_1d.axes[2]) == n_theta):
                phase_aer_1d = self.pha_aer_1d
            else:
                phase_aer_1d = self.pha_aer_1d.sub()[:, :, Idx(phase.axes[3])]

            # First plan parallel phase 
            for iwav in range (0, nwav):
                for iz in range (0, self.grid_3d.NZ+1):
                    luts.append(phase_aer_1d.sub()[self.ipha_aer_1d[iwav,iz], :, :])

            # Second 3d mix phase
            for iwav in range (0, nwav):
                idz_atm = []
                for icell in range (0, n_unique_cell):
                    idz = self.cloud_indices[icell,2]
                    id_zatm = self.grid_3d.NZ+1 - idz
                    idz_atm.append(id_zatm)

                ext_cld_ref0 = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(w_ref)]
                ext_factor = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(wav[iwav])]/ext_cld_ref0
                ext_cld_tmp = self.cld_ext_ref * ext_factor
                ssa_cld_tmp = self.cloud_3d.cld_mlut['ssa'][Idx(cld_reff), Idx(wav[iwav])]

                ssa_aer_tmp = self.ssa_aer_1d[iwav,idz_atm]
                ext_aer_tmp = self.ext_aer_1d[iwav,idz_atm]
                ext_mix_tmp = ext_aer_tmp + ext_cld_tmp
                ext_mix_3d[iwav,:] = ext_mix_tmp
                ssa_mix_3d[iwav,:] = (ext_aer_tmp*ssa_aer_tmp + ext_cld_tmp*ssa_cld_tmp) / ext_mix_tmp

                for icell in range (0, n_unique_cell):
                    pha_cld_tmp = phase.sub()[Idx(wav[iwav]),Idx(cld_reff[icell]), :, :]
                    pha_aer_tmp = phase_aer_1d.sub()[self.ipha_aer_1d[iwav,idz_atm[icell]], :, :]
                    pha_tot = ( (pha_aer_tmp*ext_aer_tmp[icell]*ssa_aer_tmp[icell]) +
                                (pha_cld_tmp*ext_cld_tmp[icell]*ssa_cld_tmp[icell]) ) / ext_mix_tmp[icell]
                    luts.append(pha_tot)

            # Third concatenate plan parallel + 3d optical prop (first without considering wl)
            nbz = self.ext_aer_1d.shape[1] # nb of z_atm
            tot_phase_indices = np.arange(nbz+n_unique_cell, dtype=np.int32)
        else: # case no 1d aer given
            ext_cld_ref0 = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(w_ref)]
            for iwav in range (0, nwav):
                ext_factor = self.cloud_3d.cld_mlut['ext'][Idx(cld_reff), Idx(wav[iwav])]/ext_cld_ref0
                ext_mix_3d[iwav,:] = self.cld_ext_ref * ext_factor
                ssa_mix_3d[iwav,:] = self.cloud_3d.cld_mlut['ssa'][Idx(cld_reff), Idx(wav[iwav])]
                # Loop only on the unique cld_reff
                for ireff in range (0, nreff_unique):
                    luts.append(phase.sub()[Idx(wav[iwav]),Idx(cld_reff_unique[ireff]), :, :])

            # ===== 2) phase matrix indices to take for all cloud cells
            # Obtain the correct indices from the unique radii phase matrix
            tot_phase_indices = np.full(cld_reff.size, np.nan, dtype=np.int32)
            for ireff in range (0, nreff_unique):
                tot_phase_indices[np.squeeze(np.argwhere( cld_reff == cld_reff_unique[ireff]))] = ireff

            # Concatenate plan parallel + 3d optical prop (first without considering wl)
            tot_phase_indices = np.concatenate([np.zeros(self.grid_3d.NZ+1, dtype=np.int32), tot_phase_indices[:]])

        
        # Create a table with only the cloud properties but in global shape i.e. for each cells
        #  not sharing the same opt prop, and other commun cells in z, following the plan parallel
        #  1D atm philosophy
        ext_aer_glob = np.concatenate([self.ext_aer_1d, ext_mix_3d], axis=1)
        ssa_aer_glob = np.concatenate([self.ssa_aer_1d[:,:], ssa_mix_3d], axis=1)

        # Now consider the wl dimension
        tot_phase_indices_wl = np.zeros((nwav, tot_phase_indices.size), dtype=np.int32)

        for iwav in range (0, nwav):
            tot_phase_indices_wl[iwav,:] = tot_phase_indices[:] + (iwav*nreff_unique)

        ipha3D = tot_phase_indices_wl

        return (ipha3D, luts), ext_aer_glob, ssa_aer_glob


    def get_cells_info(self):

        if (self.cloud_3d is None):
            Nopt = self.grid_3d.NZ + 1
        else:
            cloud_indices = self.cloud_indices

            # 1d xyz indices where there are clouds
            cloud_1d_indices = np.ravel_multi_index((cloud_indices[:,0], cloud_indices[:,1], cloud_indices[:,2]),
                                                    dims=(self.grid_3d.NX, self.grid_3d.NY, self.grid_3d.NZ))

            nb_unique_cells = cloud_indices.shape[0]
            Nopt = self.grid_3d.NZ + 1 + nb_unique_cells

        iopt        = np.zeros(self.grid_3d.NCELL, dtype=np.int32)
        iabs        = np.zeros_like(iopt)
        iopt[:]     = np.arange(Nopt)[self.grid_3d.NZ-self.grid_3d.idz] # Scattering depending on Z for clear atmosphere (Rayleigh)
        iabs[:]     = np.arange(Nopt)[self.grid_3d.NZ-self.grid_3d.idz] # Absorption depending on Z only

        if (self.cloud_3d is not None): iopt[cloud_1d_indices] = self.grid_3d.NZ + 1 + np.arange(cloud_1d_indices.size)

        return (iopt, iabs, self.grid_3d.pmin, self.grid_3d.pmax, self.grid_3d.neigh)