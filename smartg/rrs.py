import scipy.constants as cst
import numpy as np

# Atmosphere model: mixing ratio (mol/mol)
X_N2 = 0.788
X_O2 = 0.212


# Bates, Planel. Space Sa., Vol.32, No.6, pp. 785-790. 1984
def fk_n2(lam):
    '''
    lam in nm
    '''
    return 1.034 + 3.17*1e-4/((lam*1e-3)**2)

def epsilon_n2(lam):
    '''
    lam in nm
    '''
    return (fk_n2(lam)-1) * 4.5

def fk_o2(lam):
    '''
    lam in nm
    '''
    return 1.096 + 1.385*1e-3/((lam*1e-3)**2) + 1.448*1e-4/((lam*1e-3)**4)

def epsilon_o2(lam):
    '''
    lam in nm
    '''
    return (fk_o2(lam)-1) * 4.5

def epsilon_air(lam):
    '''
    lam in nm
    '''
    return epsilon_n2(lam) * X_N2 + epsilon_o2(lam) * X_O2

#Kattawar, Astrophysical Journal, Part 1, vol. 243, Feb. 1, 1981, p. 1049-1057.
def f0_air(lam, theta):
    '''
    lam in nm
    theta in deg
    '''
    eps = epsilon_air(lam)
    c2  = np.cos(np.radians(theta))**2
    num = (180.+13.*eps) + (180.+eps)*c2
    den = (180.+52.*eps) + (180.+4.*eps)*c2
    return num/den
    
def f0_n2(lam, theta):
    '''
    lam in nm
    theta in deg
    '''
    eps = epsilon_n2(lam)
    c2  = np.cos(np.radians(theta))**2
    num = (180.+13.*eps) + (180.+eps)*c2
    den = (180.+52.*eps) + (180.+4.*eps)*c2
    return num/den

def f0_o2(lam, theta):
    '''
    lam in nm
    theta in deg
    '''
    eps = epsilon_o2(lam)
    c2  = np.cos(np.radians(theta))**2
    num = (180.+13.*eps) + (180.+eps)*c2
    den = (180.+52.*eps) + (180.+4.*eps)*c2
    return num/den

# Joiner, J., Bhartia, P. K., Cebula, R. P., Hilsenrath, E., McPeters, R. D., & Park, H. (1995). Rotational Raman scattering (Ring effect) 
# in satellite backscatter ultraviolet measurements. Applied Optics, 34(21), 4513. doi:10.1364/ao.34.004513
## !!!! Erreur dans le papier original sur les coeffs de Placzek-Teller Anti Stokes !!!

def k_ratio(lam, theta):
    '''
    lam in nm
    theta in deg
    '''
    return (1.-f0_o2(lam,theta))/(1.-f0_n2(lam,theta))

def bjm_plus(J):
    return 3.*(J+1)*(J+2)/2./(2*J+1)/(2*J+3)

def bjm_minus(J):
    b = 3.*J*(J-1)/2./(2*J+1)/(2*J-1)
    b[J<=1] = 0.
    return b
    
def l_o2(T):
    '''
    O2 Rotational Raman Spectrum

    T in K
    '''
    B0 = 1.4378  # cm-1
    J = np.linspace(0, 36, num=37, dtype=np.int32)
    # 1 if J is odd, 0 if even. Faster than j % 2 != 0 (but only int!)
    gj = J & 1
    # B0 translated in m-1!!!
    Ej = J * (J + 1) * cst.h * cst.c * B0 * 100
    Fj = gj * (2 * J + 1) * np.exp(-Ej / (cst.k * T))
    
    lj_stk  = Fj * bjm_plus(J)
    dnu_stk = -(4*J+6)*B0
    is_nonzero  = lj_stk != 0.
    lj_stk  = lj_stk[is_nonzero]
    dnu_stk = dnu_stk[is_nonzero]
    lj_astk  = Fj * bjm_minus(J)
    is_nonzero  = lj_astk != 0.
    dnu_astk =  (4*J-2)*B0
    lj_astk  = lj_astk[is_nonzero]
    dnu_astk = dnu_astk[is_nonzero]

    norm = lj_stk.sum() + lj_astk.sum()
    return dnu_stk, lj_stk/norm, dnu_astk, lj_astk/norm

def l_n2(T):
    '''
    N2 Rotational Raman Spectrum

    T in K
    '''
    B0 = 1.9897  # cm-1
    J = np.linspace(0, 36, num=37, dtype=np.int32)
    # Bitwise AND with 1 selects odd J -> gj takes the odd branch value
    gj = np.where(J & 1, 3, 6)
    # B0 translated in m-1 !!!
    Ej = J * (J + 1) * cst.h * cst.c * B0 * 100
    Fj = gj * (2 * J + 1) * np.exp(-Ej / (cst.k * T))
    
    lj_stk  = Fj * bjm_plus(J)
    dnu_stk = -(4*J+6)*B0
    is_nonzero  = lj_stk != 0.
    lj_stk  = lj_stk[is_nonzero]
    dnu_stk = dnu_stk[is_nonzero]
    lj_astk  = Fj * bjm_minus(J)
    is_nonzero  = lj_astk != 0.
    dnu_astk =  (4*J-2)*B0
    lj_astk  = lj_astk[is_nonzero]
    dnu_astk = dnu_astk[is_nonzero]

    norm = lj_stk.sum() + lj_astk.sum()
    return dnu_stk, lj_stk/norm, dnu_astk, lj_astk/norm

def l_air(lam, theta, T):
    '''
    Air Rotational Raman Spectrum
    
    lam central wavelength in nm
    theta in deg
    T in K
    '''
    dnu_stk_n2, lj_stk_n2, dnu_astk_n2, lj_astk_n2 = l_n2(T)
    dnu_stk_o2, lj_stk_o2, dnu_astk_o2, lj_astk_o2 = l_o2(T)
    lj_stk_n2 *= X_N2
    lj_astk_n2 *= X_N2
    lj_stk_o2 *= X_O2*k_ratio(lam, theta)
    lj_astk_o2 *= X_O2*k_ratio(lam, theta)
    
    nu0 = 1e7/(lam) # nu0 in cm-1
    # compute output lamnda in nm
    lam_stk_n2 = 1e7/(nu0+dnu_stk_n2)
    lam_astk_n2 = 1e7/(nu0+dnu_astk_n2)
    lam_stk_o2 = 1e7/(nu0+dnu_stk_o2)
    lam_astk_o2 = 1e7/(nu0+dnu_astk_o2)
    
    norm = lj_stk_n2.sum() + lj_astk_n2.sum() + lj_stk_o2.sum() + lj_astk_o2.sum() 
    
    lam_out= np.concatenate([lam_astk_n2, lam_astk_o2, lam_stk_n2, lam_stk_o2])
    l_out  = np.concatenate([lj_astk_n2, lj_astk_o2, lj_stk_n2, lj_stk_o2])/norm
    ii = np.argsort(lam_out)
    
    # return spectrum with increasing wavelengths
    return lam_out[ii], l_out[ii]


def l2d(lam, theta, T):
    '''
    Air Rotational Raman Spectrum
    
    lam central wavelength in nm
    theta in deg
    T in K
    '''
    kk = k_ratio(lam, theta)
    nlam = lam.size
    dnu_stk_n2, lj_stk_n2, dnu_astk_n2, lj_astk_n2 = l_n2(T)
    dnu_stk_o2, lj_stk_o2, dnu_astk_o2, lj_astk_o2 = l_o2(T) 
    norm_n2 = lj_stk_n2.sum() + lj_astk_n2.sum()
    norm_o2 = lj_stk_o2.sum() + lj_astk_o2.sum()
    lj_stk_n2/=norm_n2
    lj_astk_n2/=norm_n2
    lj_stk_o2/=norm_o2
    lj_astk_o2/=norm_o2
    lj_stk_n2 = np.stack([lj_stk_n2]*nlam) * X_N2
    lj_astk_n2 = np.stack([lj_astk_n2]*nlam) * X_N2
    lj_stk_o2 = lj_stk_o2[np.newaxis, :] * X_O2 * kk[:, np.newaxis]
    lj_astk_o2 = lj_astk_o2[np.newaxis, :] * X_O2 * kk[:, np.newaxis]
    norm = np.sum(lj_stk_n2, axis=1) + np.sum(lj_astk_n2, axis=1) + np.sum(lj_stk_o2, axis=1) + np.sum(lj_astk_o2, axis=1)
    lj_stk_n2/=norm[:, np.newaxis]
    lj_astk_n2/=norm[:, np.newaxis]
    lj_stk_o2/=norm[:, np.newaxis]
    lj_astk_o2/=norm[:, np.newaxis]
    #we add also negative unity impulse at zero for removal of elastic
    #l_out  = np.concatenate([lj_astk_n2, lj_astk_o2, lj_stk_n2, lj_stk_o2, np.stack([np.array([0])]*nlam)], axis=1)
    #dnu_out= np.concatenate([dnu_astk_n2, dnu_astk_o2, dnu_stk_n2, dnu_stk_o2, np.array([0.])])
    l_out  = np.concatenate([lj_astk_n2, lj_astk_o2, lj_stk_n2, lj_stk_o2], axis=1)
    dnu_out= np.concatenate([dnu_astk_n2, dnu_astk_o2, dnu_stk_n2, dnu_stk_o2])
    
    
    # reorganization with lambda instead od Dnu and increasing order
    nu0 = 1e7/lam
    lam_out = 1e7/(nu0[:, np.newaxis]+dnu_out[np.newaxis,:])
    ii  = np.argsort(lam_out, axis=1)
    lam_out = np.take_along_axis(lam_out, ii, axis=1)
    l_out   = np.take_along_axis(l_out,   ii, axis=1)
    
    return lam_out, l_out


def l2d_inv(lam, theta, T):
    '''
    Air Inverse Rotational Raman Spectrum
    
    lam central wavelength in nm
    theta in deg
    T in K
    '''
    
    dnu_stk_n2, lj_stk_n2, dnu_astk_n2, lj_astk_n2 = l_n2(T)
    dnu_stk_o2, lj_stk_o2, dnu_astk_o2, lj_astk_o2 = l_o2(T) 
    # reorganization with lambda instead od Dnu
    dnu_in= np.concatenate([dnu_astk_n2, dnu_astk_o2, dnu_stk_n2, dnu_stk_o2])
    nu0 = 1e7/lam
    lam_in1 = 1e7/(nu0[:, np.newaxis]-dnu_stk_o2[np.newaxis,:])
    lam_in2 = 1e7/(nu0[:, np.newaxis]-dnu_astk_o2[np.newaxis,:])
    nlam = lam.shape[0]
    kk1 = k_ratio(lam_in1, theta)
    kk2 = k_ratio(lam_in2, theta)
    norm_n2 = lj_stk_n2.sum() + lj_astk_n2.sum()
    norm_o2 = lj_stk_o2.sum() + lj_astk_o2.sum()
    lj_stk_n2/=norm_n2
    lj_astk_n2/=norm_n2
    lj_stk_o2/=norm_o2
    lj_astk_o2/=norm_o2
    lj_stk_n2 = np.stack([lj_stk_n2]*nlam) * X_N2
    lj_astk_n2 = np.stack([lj_astk_n2]*nlam) * X_N2
    lj_stk_o2 = lj_stk_o2[np.newaxis, :] * X_O2 * kk1
    lj_astk_o2 = lj_astk_o2[np.newaxis, :] * X_O2 * kk2
    norm = np.sum(lj_stk_n2, axis=1) + np.sum(lj_astk_n2, axis=1) + np.sum(lj_stk_o2, axis=1) + np.sum(lj_astk_o2, axis=1)
    lj_stk_n2/=norm[:, np.newaxis]
    lj_astk_n2/=norm[:, np.newaxis]
    lj_stk_o2/=norm[:, np.newaxis]
    lj_astk_o2/=norm[:, np.newaxis]
    l_in    = np.concatenate([lj_astk_n2, lj_astk_o2, lj_stk_n2, lj_stk_o2], axis=1)

    lam_in = 1e7/(nu0[:, np.newaxis]-dnu_in[np.newaxis,:])
    ii  = np.argsort(lam_in, axis=1)
    lam_in = np.take_along_axis(lam_in, ii, axis=1)
    l_in   = np.take_along_axis(l_in,   ii, axis=1)
    
    return lam_in, l_in
