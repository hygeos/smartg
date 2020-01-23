import numpy as np
from scipy.interpolate import interp1d

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

