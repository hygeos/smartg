#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from luts.luts import Idx, MLUT, LUT
import pandas as pd
from pathlib import Path
import os

from smartg.atmosphere import AtmAFGL, CloudOPAC, od2k
from smartg.smartg import Sensor

import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

import xarray as xr

def is_sorted(arr):
    """
    Description : Check if the numpy array values are in the ascending order.

    === Paramerter:
    arr : Numpy 1D array

    === Return:
    Boolean
    """
    return np.all(np.diff(arr) >= 0)

def is_same_cell_size(grid):
    """
    Description: Check if a given grid constains cells with the same size

    === Paramerter:
    grid : Numpy 1D array

    === Return:
    Boolean
    """

    return (np.max(grid[1:]-grid[:-1]) - np.min(grid[1:]-grid[:-1]) < 10e-6)

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
    return np.math.floor(np.math.log(np.max(np.abs(mat)), 10))

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

def create_1d_grid(cell_number, cell_size, loc="centered"):
    """
    Description : Create a 1 dimensional grid profil

    === Parameters:
    cell_size   : Size of a cell.
    cell_number : The number of cells.
    loc         : Grid location. By default an str: "centered" i.e. the grid center is at coordinate 0.
                  Or give a scalar with the starting position of the grid.

    === Return:
    Numpy array with a 1D grid profil
    """

    if loc == "centered":
        half_grid_size = cell_number*cell_size/2.
        grid = np.linspace(-half_grid_size, half_grid_size, num=cell_number+1)
    elif np.isscalar(loc):
        loc = float(loc)
        grid_size = cell_number*cell_size
        grid = np.linspace(loc, loc+grid_size, num=cell_number+1)
    else:
        raise NameError("Unkown argument for the variable loc!")

    return grid

def extend_1d_grid(grid, extend_value, type='length'):
    """
    Description: Extend a 1D grid

    === Parameters:
    grid                 : 1D numpy array to be extended.
    extend_value         : Extend value.
    type                 : "length" -> extend the grid by the extend value length
                           "limit"  -> extend the grid until a given limit

    === Return:
    extended_grid : The 1D array grid after the extend

    Example1:
    grid = np.array([0., 10.])
    extended_grid = extend_1d_grid(grid, extend_value=20., type='length')
    return -> extended_grid = np.array([-20., 0., 10., 30.])

    Example2:
    grid = np.array([0., 10.])
    extended_grid = extend_1d_grid(grid, extend_value=20., type='limit')
    return -> extended_grid = np.array([0., 10., 20.])
    """

    if type == "length":
        N = len(grid)
        N_extended = N+2
        extended_grid = np.zeros((N_extended), dtype=np.float32)
        extended_grid[0] = grid[0]-extend_value
        extended_grid[-1] = grid[-1]+extend_value
        extended_grid[1:-1] = grid[:]
    elif type == "limit":
        if (extend_value >= grid[0] and extend_value <= grid[-1]):
            raise NameError("The extend limit value must be outside the range of the initial grid!")
        else:
            extend_value = np.array([float(extend_value)])
            extended_grid = np.concatenate([grid, extend_value])
            extended_grid = np.sort(extended_grid)
    else:
        raise NameError("Unkown extend type!")

    return extended_grid


def read_cld_nth_cte(filename, nb_theta=721):
        """
        Description: Read libRatran water cloud files (i.g. wc.sol.mie.cdf) or monochromatic IPRT netcdf cloud file,
        and convert to LUT object with a constant theta discretisation i.e. nb_theta = cte.

        === Parameters:
        filename : File name with path location of netcdf cloud file.
        nb_theta : Number of theta discretization between 0 and 180 degrees.
        === Return:
        LUT object with the cloud phase matrix but with a constant theta number = nb_theta
        """

        ds = xr.open_dataset(filename)
        
        # Phase matrix (wl=670nm, reff, stk, ntheta)
        phase = ds["phase"][:, :, :, :].data

        NBSTK   = 4
        NBTHETA = nb_theta
        NBREFF  = ds["reff"].size
        NWAV    = ds["wavelen"].size
        theta = np.linspace(0., 180., num=NBTHETA)
        reff = ds["reff"].data
        wavelength = ds["wavelen"].data*1e3

        P = LUT( np.full((NWAV, NBREFF, NBSTK, NBTHETA), np.NaN, dtype=np.float32),
                 axes=[wavelength, reff, None, theta],
                 names=['wav_phase', 'reff', 'stk', 'theta_atm'],
                 desc="phase_atm" )

        for iwav in range (0, NWAV):
            for ireff in range(NBREFF):
                for istk in range (NBSTK):
                    # ntheta (wl, reff, stk)
                    nth = ds["ntheta"][iwav, ireff, istk].data

                    # theta (wl, reff, stk, ntheta)
                    th = ds["theta"][iwav, ireff, istk, :].data

                    P.data[iwav, ireff, istk, :] = np.interp(theta, th[:nth], phase[iwav,ireff,istk,:nth],  period=np.inf)

        # convert I, Q into Ipar, Iper
        P0 = P.data[:,:,0,:].copy()
        P1 = P.data[:,:,1,:].copy()
        P.data[:,:,0,:] = P0+P1
        P.data[:,:,1,:] = P0-P1

        return P


class Cloud3D(object):
    """
    Description: Represent a 3D cloud profil

    === Attribut:
    phase                : LUT object with the cloud phase Matrix depending on wl, reff, stk, and theta
    file_name            : File name with path location. File following the convention of IPRT cloud files
                           with exctintion coefficient and reff of each cloud cell.
    loc_xgrid, loc_ygrid : Grid location. By default an str: "centered" i.e. the grid center is at coordinate 0.
                           Or give a scalar with the starting position of the grid.
    reff_acc             : interger with decimal accuracy of reff. By default None then do not replace the read reff values

    === Others if given circumvent variables read from file_name
    xyz_grids            : List with in the indices 0, 1 and 2 the 1D arrays with respectively the x, y and z grid profils
    ext_coeff            : Numpy 1D array with the cloud extinction coefficient
    cell_indices         : Numpy 3D array with the cloud xyz indices
    reff                 : Numpy 1D array with the cloud effective radii
    """

    def __init__(self, phase=None, file_name=None, loc_xgrid = "centered", loc_ygrid = "centered", reff_acc = None,
                 xyz_grids=None, ext_coeff=None, cell_indices=None, reff=None):

        if (phase is None):
            self.phase = phase
        # Check if phase is a LUT object with the correct axes
        elif (not isinstance(phase, LUT)):
            raise NameError("phase must be a LUT object!")
        elif (not all([item in phase.names for item in ['wav_phase', 'reff', 'stk', 'theta_atm']])):
            raise NameError("Phase matrix must have 4 dimensions: wav_phase, reff, stk and theta_atm")
        else:
            self.phase = phase

        # Check (if not set to None) that the file exists and is readable
        if (file_name is None):
            self.file_name = None
        elif (not Path(file_name).exists()):
            raise NameError("The given file does not exists!")
        elif (not os.access(file_name, os.R_OK)):
            raise NameError("The given file cannot be read!")
        else:
            self.file_name = file_name
        
        # If file_name is None, all other variables must be specified
        if  ( (file_name is None)
              and ( xyz_grids is None or ext_coeff is None or cell_indices is None or reff is None) ):
            raise NameError("If file_name is set to None all other variables must be given!")

        # TODO adds checks on the varaibles bellow (if we have np.arrays, ...)
        self.xyz_grids    = xyz_grids
        self.loc_xgrid    = loc_xgrid
        self.loc_ygrid    = loc_ygrid
        self.ext_coeff    = ext_coeff
        self.cell_indices = cell_indices
        self.reff         = reff
        self.reff_acc     = reff_acc

    def get_xyz_grid(self):
        """
        Description: Get the x, y and z 1D grid profils.

        === Return:
        xgrid, ygrid, zgrid : Three 1D arrays with the x, y and z grid profils.
        """

        # First check if we have already the xyz grids
        if (self.xyz_grids is not None): return self.xyz_grids[0], self.xyz_grids[1], self.xyz_grids[2]

        # Read only the needed information, the two first rows.
        # Be careful ! The second row have a greater dimension than the first one. Then -> two steps of reading.
        contentA = pd.read_csv(self.file_name, skiprows = 1, nrows = 1, header=None, sep='\s+', dtype=float).values
        contentB = pd.read_csv(self.file_name, skiprows = 2, nrows = 1, header=None, sep='\s+', dtype=float).values

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
        xgrid = create_1d_grid(Nx, Dx, loc=self.loc_xgrid)
        ygrid = create_1d_grid(Ny, Dy, loc=self.loc_ygrid)

        # Grid in the z axis can be directly read from the file
        zgrid = contentB[2:]

        return xgrid, ygrid, zgrid

    def get_ext_coeff(self):
        """
        Description: Get the cloud extinction coefficient of each cell indices given in the input file

        === Return:
        ext_coeff : Numpy 1D array with the cloud extinction coefficient
        """

        # First check if we have already the ext_coeff
        if (self.ext_coeff is not None) : return self.ext_coeff

        # Read only the disired column
        ext_coeff = pd.read_csv(self.file_name, skiprows = 3, header=None, usecols=[3], sep='\s+', dtype=float).values

        # If there are empty dimensions remove them
        ext_coeff = np.squeeze(ext_coeff)

        return ext_coeff

    def get_cell_indices(self):
        """
        Description: Get the cloud cell indices where there are clouds

        === Return:
        cell_indices : Numpy 3D array with the cloud xyz indices
        """

        # First check if we have already the cell_indices
        if (self.cell_indices is not None): return self.cell_indices

        # Read only the disired column
        cell_indices = pd.read_csv(self.file_name, skiprows = 3, header=None, usecols=[0,1,2], sep='\s+', dtype=float).values

        # If there are empty dimensions remove them and ensure that we have interger type
        cell_indices = np.squeeze(cell_indices.astype(np.int32))

        # We need to have 2 dimensions, a numpy array as list of another numpy arrays with the x, y and z indices
        if(cell_indices.ndim == 1): cell_indices = np.array([cell_indices])

        return cell_indices

    def get_reff(self):
        """
        In progress...
        """

        # First check if we have already reff
        if (self.reff is not None): return self.reff

        # Read only the disired column
        reff = pd.read_csv(self.file_name, skiprows = 3, header=None, usecols=[4], sep='\s+', dtype=float).values
        if self.reff_acc is not None: reff = np.around(reff, decimals=self.reff_acc)

        # If there are empty dimensions remove them
        reff = np.squeeze(reff)

        return reff


def Get_3Dcells_indices(NX, NY, NZ):
    '''
    set up a rectangular regular 3D grid indices
    
    Inputs:
        Number of grid cells in each dimension
        
    Ouputs:
        triplet of 3D indices
    '''
    Ncell = NX*NY*NZ
    # from cell number to x,y and z indices
    return np.unravel_index(np.arange(Ncell, dtype=np.int32), (NX, NY, NZ), order='C')


def Get_3Dcells_neighbours(NX, NY, NZ, BOUNDARY_ABS=-5, periodic=False,
                          BOUNDARY_BOA=-2, BOUNDARY_TOA=-1):
    '''
    Computes the 3D neighbouring cells indices, one for each of the 6 cube faces
    
    Inputs:
        Number of grid cells in each dimension
        
    Keyword:
        - periodic : the neighbours are horizontally periodic, otherwise it is an absorbing boundary
        
    Outputs:
        2D array (6, Ncell) containing the neighbouring cell indices for each of the 6 cuboid faces,
            with the convention order, +X,-X,+Y,-Y,+Z,-Z

    '''
    idx, idy, idz  = Get_3Dcells_indices(NX, NY, NZ)
    # indices of neighbouring cells in rectangular grid
    neigh_idx      = np.vstack((idx+1, idx-1, idx  , idx  , idx  , idx  )) # by convention POSITIVE first
    neigh_idy      = np.vstack((idy  , idy  , idy+1, idy-1, idy  , idy  ))
    neigh_idz      = np.vstack((idz  , idz  , idz  , idz  , idz+1, idz-1))
    
    if periodic: 
        neigh = np.ravel_multi_index((neigh_idx, neigh_idy, neigh_idz), 
                                       dims=(NX, NY, NZ), mode=('wrap','wrap','clip'))
    else:
        neigh = np.ravel_multi_index((neigh_idx, neigh_idy, neigh_idz), 
                                       dims=(NX, NY, NZ), mode=('clip','clip','clip'))
    ## boundaries neighbouring
    # with 'clip' mode, if outside the domain then the neighbour index is the same as the cell index
    neigh[np.equal(neigh , np.arange(NX*NY*NZ, dtype=np.int32))] = BOUNDARY_ABS 
    # by definition -Z neighbour at the domain boundary is BOA
    neigh[5, np.where(neigh[5,:]==BOUNDARY_ABS)] = BOUNDARY_BOA 
    # by definition +Z neighbour at the domain boundary is TOA
    neigh[4, np.where(neigh[4,:]==BOUNDARY_ABS)] = BOUNDARY_TOA
    # by convention +Z neighbour at the domain boundary -1 is also TOA
    neigh[4, np.where(neigh[4,:]%NZ==        0)] = BOUNDARY_TOA 
    
    return neigh.astype(np.int32)


def Get_3Dcells(Nx=1, Ny=1, Nz=50, Dx=1., Dy=1., Dz=1.,x=None, y=None, z=None, periodic=False,
                HORIZ_EXTENT_LENGTH=0, SAT_ALTITUDE=1e3):
    '''
    return the cells geometrical properties for use in 3D atmospheric profile object

    Keywords:
        - Nx, Ny and Nz are the number of cells in each dimension
        - Dx, Dy and Dz are the cells dimensions in km (default 1.)
        - x(Nx), y(Ny), z(Nz), coordinates can be provided instead, it erase Ni and Di
        - HORIZ_EXTENT_LENGTH: in km, is not 0, then one cell before and one after in X and Y 
            are added with a specific length of HORIZ_EXTENT_LENGTH. it results in a total
            number of cells being NX = Nx+2 and NY = Ny+2
        - SAT_ALTITUDE: Max altitude in km for sensors location within the 3D grid
        - periodic : the neighbours are horizontally periodic, otherwise it is an absorbing boundary
    '''
    # CELLS INDEXING
    DN             = 2 if HORIZ_EXTENT_LENGTH !=0 else 0
    sl             = slice(DN//2, -DN//2) if DN==2 else slice(None, None)
        
    if x is None:
        NX   = Nx + DN  # Number of cells in x
        # cells boundaries coordinates 
        # horizontal 
        Hx   = Nx*Dx/2.  # x central domain half length (km)
        x    = np.zeros((NX+1), dtype=np.float32)
        x[0] = -(Hx + HORIZ_EXTENT_LENGTH)
        x[-1]= (Hx  + HORIZ_EXTENT_LENGTH)
        x[sl]= np.linspace(-Hx, Hx, num=Nx+1)
    else:
        NX   = x.size-1
        
    if y is None:
        NY   = Ny + DN  # Number of cells in x
        # cells boundaries coordinates 
        # horizontal 
        Hy   = Ny*Dy/2.  # y central domain half length (km)
        y    = np.zeros((NY+1), dtype=np.float32)
        y[0] = -(Hy + HORIZ_EXTENT_LENGTH)
        y[-1]= (Hy  + HORIZ_EXTENT_LENGTH)
        y[sl]= np.linspace(-Hy, Hy, num=Ny+1)
    else:
        NY   = y.size-1
        
    # vertical boundaries
    if z is None :
        # we add a empty very thin cell above TOA (just for interfacing purposes)
        NZ    = Nz + 1
        z     = np.zeros((NZ+1), dtype=np.float32)
        z[:-1]= np.linspace(0, Nz*Dz, num=NZ)
        z[-1] = SAT_ALTITUDE # above TOA , sensor max level
    else:
        NZ    = z.size-1
    
    # from cell number to x,y and z indices
    idx, idy, idz  = Get_3Dcells_indices(NX, NY, NZ)
    # indices of neighbouring cells in rectangular grid
    neigh          = Get_3Dcells_neighbours(NX, NY, NZ, periodic=periodic)
    # Bounding boxes, lower left and upper right corners
    pmin           = np.zeros((3, NX*NY*NZ), dtype=np.float32)
    pmax           = np.zeros_like(pmin)
    pmin[0,:]      = x[idx]
    pmax[0,:]      = x[idx+1]
    pmin[1,:]      = y[idy]
    pmax[1,:]      = y[idy+1]
    pmin[2,:]      = z[idz] # ! from bottom to top
    pmax[2,:]      = z[idz+1]

    return (idx,idy,idz), (NX,NY,NZ), (x,y,z), neigh, pmin, pmax


def locate_3Dregular_cells(xgrid,ygrid,zgrid,x,y,z):
    '''
    return the cells indices corresponding the the coordinates x,y,z
    in a regular grid whose limits are defined by xgrid,ygrid and zgrid
    '''
    return  np.ravel_multi_index(( \
            np.floor(interp1d(xgrid, np.arange(len(xgrid)))(x)).astype(int) ,
            np.floor(interp1d(ygrid, np.arange(len(ygrid)))(y)).astype(int) ,
            np.floor(interp1d(zgrid, np.arange(len(zgrid)))(z)).astype(int)),
                     dims = (len(xgrid)-1, len(ygrid)-1, len(zgrid)-1))


def satellite_view(mlut, xgrid, ygrid, wl, interp_name='none',
                   color_bar='Blues_r', color_reverse=False, fig_size=(8,8), font_size=int(18),
                   vmin = None, vmax = None, scale=False, save_file=None, stk="I", factor=1,
                   mat_force=None, cb_shrink=0.9, cb_sform = True, fig_title=None):
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


    # Check if the product of Nx and Ny is equal to the number of sensors
    Nx = xgrid.size-1; Ny = ygrid.size-1 # Number of sensors in x and y axis
    if "sensor index" in mlut[stokes_name[0]].names:
        sensor_number = mlut.axes["sensor index"].size
    else:
        sensor_number = int(1)
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
        
        cbar = plt.colorbar(img, shrink=cb_shrink, orientation='vertical', format=find_order_or_none(matrix[0], cb_sform), ticks=get_tv(vmin[0], vmax[0], matrix[0]))
        cbar.set_label(r''+ stokes_name[0], fontsize = font_size)

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
                        interpolation=interp_name)#, extent=[xgrid.min(),xgrid.max(),ygrid.min(),ygrid.max()])
                cbar = fig.colorbar(cax, ax=axs[i,j], shrink=cb_shrink, orientation='vertical', format=find_order_or_none(matrix[i*2+j], cb_sform), ticks=get_tv(vmin[i*2+j], vmax[i*2+j], matrix[i*2+j]))
                cbar.set_label(r''+ stokes_name[i*2+j], fontsize = font_size)
                #axs[i,j].set_xlim(xgrid[0], xgrid[-1])

        fig.supxlabel(r'X (km)')
        fig.supylabel(r'Y (km)')
    else:
        raise NameError("Give more than 4 stk is not authorized!")
    
    # If the option save_file is used, save the file in pdf
    if (save_file is not None):
        # Deal with the case where we have not specified the '.pdf' at the end
        if not save_file.endswith('.pdf'):
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
    icells = locate_3Dregular_cells(g.xGRID, g.yGRID, g.zGRID, xx.ravel(), yy.ravel(), zz.ravel())

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

    
class Grid3D(object):
    """
    The class Grid3D represent the 3D grid necessary to represent the 3D atmosphere


    === Attributs:
    xgrid, ygrid, zgrid : Numpy array with grid profil in the x, y and z axes.
    periodic            : Boolean to know if the periodic condition for the x and y axes will be used.
    horiz_extend_length : Set x, y boundaries (only if periodic is true) with a given extend length.
    vert_extend_limit   : In progress...

    === Other attributs calculated:
    Nx, Ny, Nz          : The number of cells in the x, y and z axes without boundaries.
    NX, NY, NZ          : The number of cells in the x, y and z axes considering the boundaries.
    NCELL               : Total number of cells, equal to NX*NY*NZ.
    xGRID, yGRID, zGRID : Numpy array with grid profil in the x, y and z axes considering the boundaries. 
    idx, idy, idz       : x, y and z indices of the 3D grid matrix.
    neigh               : In progress...
    pmin, pmax          : In progress...
    """
    def __init__ (self, xgrid, ygrid, zgrid, periodic=False, horiz_extend_length=None, vert_extend_limit=None):

        # Ensure xgrid, ygrid or zgrid are 1D numpy arrays
        if (   (not isinstance(xgrid, np. ndarray))
            or (not isinstance(ygrid, np. ndarray))
            or (not isinstance(zgrid, np. ndarray))  ):
            raise NameError('xgrid, ygrid and zgrid must be numpy arrays!')
        elif (xgrid.ndim > 1 or ygrid.ndim > 1 or zgrid.ndim > 1):
            raise NameError('xgrid, ygrid and zgrid must be 1D numpy arrays!')

        # Check that xgrid, ygrid and zgrid are sorted
        if (   not is_sorted(xgrid)
            or not is_sorted(ygrid)
            or not is_sorted(zgrid)  ):
            raise NameError('Check xgrid, ygrid or zgrid! Values must be in the ascending order.')

        # If there is a periodic condition the horizontal boundaries are prohibited
        if (periodic and horiz_extend_length is not None):
            raise NameError('If periodic is set to True, the variable horiz_extend_length must be equal to None!')

        # ==== calculation of the other attributs
        # Consider the boundaries if horiz_extend_length or vert_extend_limit is given
        if horiz_extend_length is not None:
            xgrid_with_boundary = extend_1d_grid(xgrid, horiz_extend_length, type='length')
            ygrid_with_boundary = extend_1d_grid(ygrid, horiz_extend_length, type='length')
        else:
            xgrid_with_boundary = xgrid
            ygrid_with_boundary = ygrid
        if vert_extend_limit is not None:
            # Check that the extend value limit is strictly greater than the max value of zgrid
            if ( vert_extend_limit > np.max(zgrid) ):
                zgrid_with_boundary = extend_1d_grid(zgrid, vert_extend_limit, type='limit')
            else:
                raise NameError('The vertical extend limit must be strictly greater than the max value of zgrid!')
        else:
            zgrid_with_boundary = zgrid

        (idx,idy,idz), (NX,NY,NZ), (xGRID, yGRID, zGRID), neigh, pmin, pmax = \
            Get_3Dcells(x=xgrid_with_boundary, y=ygrid_with_boundary, z=zgrid_with_boundary,
             SAT_ALTITUDE=zgrid_with_boundary[-1], periodic=periodic)
        # =====

        self.Nx = len(xgrid)-1
        self.Ny = len(ygrid)-1
        self.Nz = len(zgrid)-1
        self.periodic = periodic
        self.horiz_extend_length = horiz_extend_length
        self.vert_extend_limit = vert_extend_limit
        self.NX = NX
        self.NY = NY
        self.NZ = NZ
        self.NCELL = NX*NY*NZ
        self.idx = idx
        self.idy = idy
        self.idz = idz
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.zgrid = zgrid
        self.xGRID = xGRID
        self.yGRID = yGRID
        self.zGRID = zGRID
        self.neigh = neigh
        self.pmin = pmin
        self.pmax = pmax

    # Print all the attributs using the function Print()
    def __str__(self):
        attributs = {attr: getattr(self, attr) for attr in dir(self) if not attr.startswith("__") and not callable(getattr(self, attr))}
        return str(attributs)



class Atm3D(object):
    """
    In progress...
    If wls.size is equal to 1 and wl_ref is None, take wls as reference wavelength
    """

    def __init__(self, atm_filename, grid_3d, wls, cloud_3d=None, wl_ref = None,
    lat=45, P0=None, O3=None, H2O=None, NO2=True, tauR=None, mol_sca_coeff=None, mol_abs_coeff=None):

        possible_atm_filename = ['afglms', 'afglmw', 'afglss', 'afglsw', 'afglt', 'afglus',
         'afglus_ch4_vmr', 'afglus_co_vmr', 'afglus_n2_vmr', 'afglus_n2o_vmr', 'afglus_no2',
          'mcclams', 'mcclamw']

        if (atm_filename not in possible_atm_filename):
            raise NameError('Unknown atmosphere file name!')

        if (not isinstance(grid_3d, Grid3D)):
            raise NameError('grid_3d variable must be a Grid3D object!')

        if (cloud_3d is None):
            self.phase         = None
            self.cloud_3d      = None
            self.cld_ext_coeff = None
            self.cld_reff      = None
            self.cloud_indices = None
        elif (not isinstance(cloud_3d, Cloud3D)):
            raise NameError('cloud_3d variable must be a Cloud3D object!')
        else:
            self.phase         = cloud_3d.phase
            self.cloud_3d      = cloud_3d
            self.cld_ext_coeff = cloud_3d.get_ext_coeff()
            self.cld_reff      = cloud_3d.get_reff()
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

        # Calculate the 1d extinction rayleigh coefficient and store it as attribut
        znew = grid_3d.zGRID[::-1]
        atm_1d = AtmAFGL(atm_filename, lat=lat, P0=P0, O3=O3, H2O=H2O, NO2=NO2, tauR=tauR,
                               grid=znew).calc(wls, phase=False)
        ext_ray = od2k(atm_1d, 'OD_r')
        mol_abs = od2k(atm_1d, 'OD_g')
        self.ext_rayleigh = ext_ray
        self.molecular_abs_coeff = mol_abs

        if mol_sca_coeff is not None : self.ext_rayleigh = mol_sca_coeff
        if mol_abs_coeff is not None : self.molecular_abs_coeff = mol_abs_coeff
        # ===

        # TODO -> calulate the 1d aer extinction coeff ??

    def get_3d_rayleigh_ext_coeff(self):
        """
        Consider all the grid cells with non commun opt property + 1d atm
        In progress...
        """
        if (self.cloud_3d is None):
            ext_rayleigh_3d = self.ext_rayleigh
        else:
            cloud_indices = self.cloud_indices

            # 1d xyz indices where there are clouds
            cloud_1d_indices = np.ravel_multi_index((cloud_indices[:,0], cloud_indices[:,1], cloud_indices[:,2]),
                                                    dims=(self.grid_3d.NX, self.grid_3d.NY, self.grid_3d.NZ))

            # calculate the 3d rayleigh extinction coefficient
            ext_rayleigh_3d = np.concatenate([self.ext_rayleigh,
                self.ext_rayleigh[:,self.grid_3d.NZ-self.grid_3d.idz[cloud_1d_indices]]], axis=1)

        return ext_rayleigh_3d

    def get_molecular_abs(self):
        """
        Consider all the grid cells with non commun opt property + 1d atm
        In progress...
        """

        if (self.cloud_3d is None):
            mol_abs_coeff_3d = self.molecular_abs_coeff
        else: # if there are clouds
            cloud_indices = self.cloud_indices

            # 1d xyz indices where there are clouds
            cloud_1d_indices = np.ravel_multi_index((cloud_indices[:,0], cloud_indices[:,1], cloud_indices[:,2]),
                                                    dims=(self.grid_3d.NX, self.grid_3d.NY, self.grid_3d.NZ))

            # calculate the 3d molecular coefficient
            mol_abs_coeff_3d = np.concatenate([self.molecular_abs_coeff,
                self.molecular_abs_coeff[:,self.grid_3d.NZ-self.grid_3d.idz[cloud_1d_indices]]], axis=1)

        return mol_abs_coeff_3d

    def get_grid(self):

        if( self.cloud_3d is None):
            Nopt = self.grid_3d.NZ + 1
        else:
            nb_unique_cells = self.cloud_indices.shape[0]
            Nopt = self.grid_3d.NZ + 1 + nb_unique_cells

        return np.arange(Nopt)

    def get_3d_cld_ext_coeff(self, cloud_MLUT=None, reff=5.):
        """
        cloud_LUT : To consider variation of the ext coeff in function of wls
        """

        if (cloud_MLUT is not None and not isinstance(cloud_MLUT, MLUT)):
            raise NameError('The given cloud_MLUT variable is not an MLUT object!')

        ext_cld = self.cld_ext_coeff

        if cloud_MLUT is None:
            # In this case, we just take the same ext coeff for each wl
            ext_cld_wls = ext_cld
            for i in range (0, self.wls.size-1):
                ext_cld_wls = np.concatenate([ext_cld_wls, ext_cld])
            ext_cld_wls = ext_cld_wls.reshape((self.wls.size, ext_cld.size))
        else:
            cloud_sub    = cloud_MLUT.sub({'effective_radius':Idx(reff),'effective_variance':0})
            wls_micro    = self.wls*1e-3
            wl_ref_micro = self.wl_ref*1e-3
            cloud_wls    = cloud_sub['normed_ext_coeff' ][Idx(wls_micro, fill_value='extrema')]
            cloud_wl_ref = cloud_sub['normed_ext_coeff' ][Idx(wl_ref_micro, fill_value='extrema')]
            eps_cld      = cloud_wls/cloud_wl_ref
            ext_cld_wls  = ext_cld[None, :] * eps_cld[:, None]

        # Create a table with only the cloud properties but in global shape i.e. for each cells
        #  not sharing the same opt prop, and other commun cells in z, following the plan parallel
        #  1D atm philosophy
        ext_cld_3d = np.concatenate([np.zeros_like(self.ext_rayleigh), ext_cld_wls], axis=1)

        return ext_cld_3d

    def get_phase_prof_old(self, species='wc.sol', reff=2., wl_phase=None):
        """
        For the moment only one phase matrix for all cells containing clouds
        In progress..
        return a tuple with the phase matrix indices to choose, and a list of phase matrix LUT.
        """

        nb_unique_cells = self.cloud_indices.shape[0]
        Nopt = self.grid_3d.NZ + 1 + nb_unique_cells
        
        # TODO s'inspirer de la fonction calc_iphase dans tools/phase.py
        ipha3D = np.zeros((self.wls.size, Nopt), dtype=np.int32) # only one matrix for all cells, then always index 0

        cld = CloudOPAC(species, reff, 2., 3., 1., self.wl_ref) # reff = 2 here but previouly we took 5, why ?
        # Create phase matrix and store it in LUT object (function of stokes and theta)
        pha_cld_tmp = AtmAFGL(self.atm_filename, comp=[cld], pfwav=wl_phase).calc(self.wls)['phase_atm']
        pha_cld = []

        # if wl_phase is not None:
        #     for i in range (0, wl_phase.size):
        #         pha_cld.append(pha_cld_tmp.sub()[Idx(wl_phase[i]),:,:])
        # else:
        #     for i in range (0, self.wls.size):
        #         pha_cld.append(pha_cld_tmp.sub()[i,:,:])

        # Calculate the phase for each wl_phase if given else for each wls
        if wl_phase is not None: nwav = len(wl_phase)
        else: nwav = len(self.wls)
        
        for i in range (0, nwav):
            pha_cld.append(pha_cld_tmp.sub()[i,:,:])
        


        return (ipha3D, pha_cld)

    def get_phase_prof_OPAC(self, species='wc.sol', wl_phase=None, phaseOpti=False, nb_theta=721, AOD=1.):
        """
        return a tuple with the phase matrix indices to choose, and a list of phase matrix LUT.
        """

        cld_reff = self.cld_reff
        cld_reff_unique = np.unique(cld_reff)
        nreff_unique = cld_reff_unique.size

        if wl_phase is not None: wav = wl_phase
        else: wav = self.wls
        nwav = len(wav)

        # Get the LUT of the phase matrix in function of the effective radius and theta
        # Loop only on the unique cld_reff
        pha_cld = []
        for iwav in range (0, nwav):
            for ireff in range (0, nreff_unique):
                cld = CloudOPAC(species, cld_reff_unique[ireff], 2., 3., AOD, self.wl_ref)
                pha_cld.append(AtmAFGL(self.atm_filename, comp=[cld]).calc([wav[iwav]],
                 phaseOpti=phaseOpti, NBTHETA=nb_theta)['phase_atm'].sub()[0,:,:])
    
        # ===== 2) phase matrix indice to take for all cloud cells
        # Obtain the correct indices from the unique radii phase matrix
        cld_phase_indices = np.full(cld_reff.size, np.NaN, dtype=np.int32)
        for ireff in range (0, nreff_unique):
            cld_phase_indices[np.squeeze(np.argwhere( cld_reff == cld_reff_unique[ireff]))] = ireff

        # Concatenate Rayleigh plan parallel + cloud
        cld_phase_indices = np.concatenate([np.zeros(self.grid_3d.NZ+1, dtype=np.int32), cld_phase_indices[:]])

        # Consider the wl dimension
        cld_phase_indices_wl = np.zeros((nwav, cld_phase_indices.size), dtype=np.int32)

        for iwav in range (0, nwav):
            cld_phase_indices_wl[iwav,:] = cld_phase_indices[:] + (iwav*nreff_unique)

        ipha3D = cld_phase_indices_wl

        return (ipha3D, pha_cld)

    def get_phase_prof_IPRT(self):
        """
        In progress...
        """

        # ===== 1) Calcul des matrices de phases
        # Get the cloud effective radii
        cld_reff = self.cld_reff
        cld_reff_unique = np.unique(cld_reff)
        nreff_unique = cld_reff_unique.size

        # Get the MLUT of the monochromatic phase matrix in function of the effective radius and theta
        phase_670 = self.phase
        luts = []

        # Loop only on the unique cld_reff
        for ireff in range (0, nreff_unique):
            luts.append(phase_670.sub()[0,Idx(cld_reff_unique[ireff]), :, :])

        # ===== 2) phase matrix indices to take for all cloud cells
        # Obtain the correct indices from the unique radii phase matrix
        cld_phase_indices = np.full(cld_reff.size, np.NaN, dtype=np.int32)
        for ireff in range (0, nreff_unique):
            cld_phase_indices[np.squeeze(np.argwhere( cld_reff == cld_reff_unique[ireff]))] = ireff

        # Concatenate Rayleigh plan parallel + cloud
        cld_phase_indices = np.concatenate([np.zeros(self.grid_3d.NZ+1, dtype=np.int32), cld_phase_indices[:]])

        # Consider the wl dimension (even if its equal to 1)
        cld_phase_indices_wl = np.zeros((1, cld_phase_indices.size), dtype=np.int32)
        cld_phase_indices_wl[0,:] = cld_phase_indices[:]

        return (cld_phase_indices_wl, luts)

    def get_phase_prof(self, wl_phase=None):
        """
        In progress...
        """

        # ===== 1) Calcul des matrices de phases
        # Get the cloud effective radii
        cld_reff = self.cld_reff
        cld_reff_unique = np.unique(cld_reff)
        nreff_unique = cld_reff_unique.size

        if wl_phase is not None: wav = wl_phase
        else: wav = self.wls
        nwav = len(wav)

        # Get the MLUT of the monochromatic phase matrix in function of the effective radius and theta
        phase = self.phase

        if (phase is None):
            raise NameError("Phase matrix is none. This method cannot be used! Use instead: get_phase_prof_OPAC().")

        luts = []

        for iwav in range (0, nwav):
            # Loop only on the unique cld_reff
            for ireff in range (0, nreff_unique):
                luts.append(phase.sub()[Idx(wav[iwav]),Idx(cld_reff_unique[ireff]), :, :])

        # ===== 2) phase matrix indices to take for all cloud cells
        # Obtain the correct indices from the unique radii phase matrix
        cld_phase_indices = np.full(cld_reff.size, np.NaN, dtype=np.int32)
        for ireff in range (0, nreff_unique):
            cld_phase_indices[np.squeeze(np.argwhere( cld_reff == cld_reff_unique[ireff]))] = ireff

        # Concatenate Rayleigh plan parallel + cloud
        cld_phase_indices = np.concatenate([np.zeros(self.grid_3d.NZ+1, dtype=np.int32), cld_phase_indices[:]])

        # Consider the wl dimension (even if its equal to 1)
        cld_phase_indices_wl = np.zeros((nwav, cld_phase_indices.size), dtype=np.int32)

        for iwav in range (0, nwav):
            cld_phase_indices_wl[iwav,:] = cld_phase_indices[:] + (iwav*nreff_unique)

        ipha3D = cld_phase_indices_wl

        return (ipha3D, luts)


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