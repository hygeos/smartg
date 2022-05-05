import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from luts.luts import Idx

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


def satellite_view(mlut, Nx, Ny, x0, y0, wl, interp_name='none',
                   color_bar='Blues_r', fig_size=(8,8), font_size=int(18),
                   vmin = None, vmax = None, scale=False, save_file=None):
    """
    Description: The function give a 'satellite' 2D image of the SMART-G 3D atm return results

    ===Parameters:
    mlut        : SMART-G return MLUT object
    Nx, Ny      : Number of sensors in x and y axis
    x0, y0      : x and y arrays with x and y cell boundary positions of the domain
    wl          : Wavelength
    interp_name : Interpolations for imshow/matshow, i.e. nearest, bilinear, bicubic, ...
    color_bar   : Bar color of the figure, i.g 'Blues_r', 'jet', ...
    fig_size    : A tuple with width and height of the figure in inches
    vmin, vmax  : Color bar interval
    scale       : If True scale between 0 and 1 (or between vmin and vmax if not None)
    font_size   : Font size of the figure
    save_file   : If not None, save the generated image in pdf, i.g. save_file = 'test', save as 'test.pdf'
    """
    # Force the use of only 'I_up (TOA), TODO add the option to choose between I, Q, U, V
    stokes_name = 'I_up (TOA)'

    # First check the Azimuth and Zenith angles dimensions exist, if not the case return an error message
    if ("Azimuth angles" or "Zenith angles") not in mlut[stokes_name].names:
        raise NameError("The Azimuth angles and/or Zenith angles dimension(s) are/is missing")

    # The number of dimensions in the LUT
    axis_number = len(mlut[stokes_name].names)
    
    ind = [slice(None)]*axis_number # if axis_number = 3, tuple(ind) equivalent to [:,:,:]

    # Two last indices for axis Azimuth angles and Zenith angles forced to 0 (consider we have only one sun position)
    ind[-1] = 0; ind[-2] = 0 # TODO Consider also the case where several sun position are given

    # if Azimuth dim or Zenith dim > 1 -> return an error message. TODO to remove once the option above is added
    if ((mlut.axes["Azimuth angles"].size or mlut.axes["Zenith angles"].size) > 1):
        raise NameError("Dimension size > 1 is not authorized for both Azimuth and Zenith angles")

    # If we have the wavelength dimension
    if "wavelength" in mlut[stokes_name].names:
        ind[-3] = Idx(wl) # TODO Enable a default value, for example for the monochromatique case


    # Check if the product of Nx and Ny is equal to the number of sensors
    if "sensor index" in mlut[stokes_name].names:
        sensor_number = mlut.axes["sensor index"].size
    else:
        sensor_number = int(1)
    if (Nx*Ny != sensor_number):
        raise NameError("The product of Nx and Ny must be equal to the number of sensors!")


    # Convert the 1D results to a 2D matrix. The order of Nx and Ny below is very important!
    matrix = mlut[stokes_name][tuple(ind)].reshape(Ny,Nx)
    # The variable matrix is now in the following form:
    # - x0 ... xn
    # y0
    #  :
    # yn

    # By default in the imshow function, the origin (origin='upper') i.e matrix[0,0] is at the upper left,
    # and we want the origin at bottom left (origin='lower).
    plt.figure(figsize=fig_size)
    plt.rcParams.update({'font.size':font_size})

    # Deal with all the possibilties where vmin, vmax and scale are used
    if scale:
        vmin_scale = 0.; vmax_scale = 1.
        if vmin is not None : vmin_scale = vmin
        if vmax is not None : vmax_scale = vmax
        matrix = np.interp(matrix, (matrix.min(), matrix.max()), (vmin_scale, vmax_scale))

    # Now call the function imshow for the 2D drawing
    img = plt.imshow(matrix, vmin=vmin, vmax=vmax, origin='lower', cmap=plt.get_cmap(color_bar),
                     interpolation=interp_name, extent=[x0.min(),x0.max(),y0.min(),y0.max()])

    # Color bar parametrization
    cbar = plt.colorbar(img, shrink=0.7, orientation='vertical')
    cbar.set_label(r''+ stokes_name, fontsize = font_size)

    # Labels for x and y axes
    plt.xlabel(r'X (km)')
    plt.ylabel(r'Y (km)')

    # If the option save_file is used, save the file in pdf
    if (save_file is not None):
        # Deal with the case where we have not specified the '.pdf' at the end
        if not save_file.endswith('.pdf'):
            save_file += '.pdf'
        plt.savefig(save_file)