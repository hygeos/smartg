"""Cartesian voxel grid geometry for the SMART-G 3D atmosphere mode.

This module defines the regular 3D grid used to describe a 3D
atmosphere: cell indexing, neighbouring relations (including the
boundary sentinels understood by the CUDA kernel), bounding boxes and
voxel location utilities.

Key Classes
-----------
Grid3D
    Regular 3D grid defined by its x, y and z cell-boundary arrays,
    with optional horizontal periodicity or boundary extensions.
"""

from __future__ import annotations

from typing import cast
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from smartg.typing import NumericArrayLike, RealNumber


def is_sorted(arr: NDArray) -> bool:
    """Check if the numpy array values are in the ascending order.

    Parameters
    ----------
    arr : NDArray
        Numpy 1D array.

    Returns
    -------
    bool
        True if the values are in the ascending order.
    """
    return bool(np.all(np.diff(arr) >= 0))


def is_same_cell_size(grid: NDArray) -> bool:
    """Check if a given grid contains cells with the same size.

    Parameters
    ----------
    grid : NDArray
        Numpy 1D array.

    Returns
    -------
    bool
        True if all the cells have the same size.
    """
    return bool(np.max(grid[1:]-grid[:-1]) - np.min(grid[1:]-grid[:-1]) < 10e-6)


def create_1d_grid(
    cell_number: int,
    cell_size: float,
    loc: str | RealNumber = "centered",
) -> NDArray[np.floating]:
    """Create a 1 dimensional grid profil.

    Parameters
    ----------
    cell_number : int
        The number of cells.
    cell_size : float
        Size of a cell.
    loc : str | RealNumber, optional
        Grid location. By default an str: "centered" i.e. the grid
        center is at coordinate 0. Or give a scalar with the starting
        position of the grid.

    Returns
    -------
    NDArray
        Numpy array with a 1D grid profil.
    """
    if loc == "centered":
        half_grid_size = cell_number*cell_size/2.
        grid = np.linspace(-half_grid_size, half_grid_size, num=cell_number+1)
    elif np.isscalar(loc):
        loc = float(cast(RealNumber, loc))
        grid_size = cell_number*cell_size
        grid = np.linspace(loc, loc+grid_size, num=cell_number+1)
    else:
        raise NameError("Unkown argument for the variable loc!")

    return grid


def extend_1d_grid(
    grid: NDArray[np.number],
    extend_value: float,
    type: str = 'length',
) -> NDArray[np.floating]:
    """Extend a 1D grid.

    Parameters
    ----------
    grid : NDArray
        1D numpy array to be extended.
    extend_value : float
        Extend value.
    type : str, optional
        "length" -> extend the grid by the extend value length.
        "limit"  -> extend the grid until a given limit.

    Returns
    -------
    NDArray
        The 1D array grid after the extend.

    Examples
    --------
    >>> grid = np.array([0., 10.])
    >>> extend_1d_grid(grid, extend_value=20., type='length')
    array([-20.,   0.,  10.,  30.], dtype=float32)
    >>> extend_1d_grid(grid, extend_value=20., type='limit')
    array([ 0., 10., 20.])
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
            raise NameError(
                "The extend limit value must be outside the range of the "
                "initial grid!"
            )
        else:
            limit = np.array([float(extend_value)])
            extended_grid = np.concatenate([grid, limit])
            extended_grid = np.sort(extended_grid)
    else:
        raise NameError("Unkown extend type!")

    return extended_grid


def Get_3Dcells_indices(
    NX: int, NY: int, NZ: int
) -> tuple[NDArray[np.integer], NDArray[np.integer], NDArray[np.integer]]:
    '''
    set up a rectangular regular 3D grid indices

    Inputs:
        Number of grid cells in each dimension

    Ouputs:
        triplet of 3D indices
    '''
    Ncell = NX*NY*NZ
    # from cell number to x,y and z indices
    return cast(
        "tuple[NDArray[np.integer], NDArray[np.integer], NDArray[np.integer]]",
        np.unravel_index(np.arange(Ncell, dtype=np.int32), (NX, NY, NZ), order='C'),
    )


def Get_3Dcells_neighbours(
    NX: int, NY: int, NZ: int,
    BOUNDARY_ABS: int = -5,
    periodic: bool = False,
    BOUNDARY_BOA: int = -2,
    BOUNDARY_TOA: int = -1,
) -> NDArray[np.int32]:
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


def Get_3Dcells(
    Nx: int = 1, Ny: int = 1, Nz: int = 50,
    Dx: float = 1., Dy: float = 1., Dz: float = 1.,
    x: NDArray[np.number] | None = None,
    y: NDArray[np.number] | None = None,
    z: NDArray[np.number] | None = None,
    periodic: bool = False,
    HORIZ_EXTENT_LENGTH: float = 0,
    SAT_ALTITUDE: float = 1e3,
) -> tuple[
    tuple[NDArray[np.integer], NDArray[np.integer], NDArray[np.integer]],
    tuple[int, int, int],
    tuple[NDArray[np.number], NDArray[np.number], NDArray[np.number]],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
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


def locate_3Dregular_cells(
    xgrid: NDArray, ygrid: NDArray, zgrid: NDArray,
    x: NumericArrayLike, y: NumericArrayLike, z: NumericArrayLike,
) -> NDArray[np.integer]:
    '''
    return the cells indices corresponding the the coordinates x,y,z
    in a regular grid whose limits are defined by xgrid,ygrid and zgrid
    '''
    warn_message = "\nlocate_3Dregular_cells is deprecated as of SMART-G 1.3.0 " + \
                   "and will be removed in one of the next release.\n" + \
                   "Please use locate_voxel_index instead (more robust and faster)."
    warn(warn_message, DeprecationWarning)

    return  cast(
        "NDArray[np.integer]",
        np.ravel_multi_index(( \
            np.floor(interp1d(xgrid, np.arange(len(xgrid)))(x)).astype(int) ,
            np.floor(interp1d(ygrid, np.arange(len(ygrid)))(y)).astype(int) ,
            np.floor(interp1d(zgrid, np.arange(len(zgrid)))(z)).astype(int)),
                     dims = (len(xgrid)-1, len(ygrid)-1, len(zgrid)-1)),
    )


def locate_voxel_index(
    xgrid: NDArray, ygrid: NDArray, zgrid: NDArray,
    x: NumericArrayLike, y: NumericArrayLike, z: NumericArrayLike,
) -> int | NDArray[np.integer]:
    '''
    Locate voxel index for given coordinates.

    Parameters
    ----------
    xgrid : 1D ndarray
        The x-axis grid boundaries of the voxels.
    ygrid : 1D ndarray
        The y-axis grid boundaries of the voxels.
    zgrid : 1D ndarray
        The z-axis grid boundaries of the voxels.
    x, y, z : float | 1D ndarray
        The coordinates for which to locate the voxel index.
        Either all scalars or all 1D arrays of same size.

    Returns
    -------
    out : int | 1D ndarray
        Flat index of the voxel(s) containing the given coordinates.
        Returns int if input coordinates are scalars, ndarray if arrays.
    '''
    # check input types and shapes
    is_x_scalar = np.isscalar(x)
    is_y_scalar = np.isscalar(y)
    is_z_scalar = np.isscalar(z)

    # verify all coordinates are same type
    if not (is_x_scalar and is_y_scalar and is_z_scalar) and \
       not (not is_x_scalar and not is_y_scalar and not is_z_scalar):
        raise TypeError("Coordinates must be either all scalars or all arrays")

    # deals only with numpy arrays for consistent handling
    x_arr = np.atleast_1d(x)
    y_arr = np.atleast_1d(y)
    z_arr = np.atleast_1d(z)

    # check that arrays have same size
    if not (x_arr.ndim == 1 and y_arr.ndim == 1 and z_arr.ndim == 1):
        raise TypeError("Coordinates must be scalars or 1D arrays")

    if not (x_arr.size == y_arr.size == z_arr.size):
        raise TypeError(f"Coordinate arrays must have same size")

    # check if coordinates are within grid boundaries
    if np.any(x_arr < xgrid[0]) or np.any(x_arr > xgrid[-1]):
        raise ValueError(f"x coordinates outside range [{xgrid[0]}, {xgrid[-1]}]")
    if np.any(y_arr < ygrid[0]) or np.any(y_arr > ygrid[-1]):
        raise ValueError(f"y coordinates outside range [{ygrid[0]}, {ygrid[-1]}]")
    if np.any(z_arr < zgrid[0]) or np.any(z_arr > zgrid[-1]):
        raise ValueError(f"z coordinates outside range [{zgrid[0]}, {zgrid[-1]}]")

    # find cell indices using binary search (faster than interp1d)
    ix = np.searchsorted(xgrid, x_arr, side='right') - 1
    iy = np.searchsorted(ygrid, y_arr, side='right') - 1
    iz = np.searchsorted(zgrid, z_arr, side='right') - 1

    # handle out-of-bounds coordinates
    ix = np.clip(ix, 0, len(xgrid) - 2)
    iy = np.clip(iy, 0, len(ygrid) - 2)
    iz = np.clip(iz, 0, len(zgrid) - 2)

    result = cast(
        "NDArray[np.integer]",
        np.ravel_multi_index((ix, iy, iz),
                             dims=(len(xgrid)-1, len(ygrid)-1, len(zgrid)-1)),
    )

    # Return scalar if input was scalar
    return cast(int, result[0]) if is_x_scalar else result


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
    def __init__(
        self,
        xgrid: NDArray[np.number],
        ygrid: NDArray[np.number],
        zgrid: NDArray[np.number],
        periodic: bool = False,
        horiz_extend_length: float | None = None,
        vert_extend_limit: float | None = None,
    ) -> None:

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
    def __str__(self) -> str:
        attributs = {attr: getattr(self, attr) for attr in dir(self) if not attr.startswith("__") and not callable(getattr(self, attr))}
        return str(attributs)
