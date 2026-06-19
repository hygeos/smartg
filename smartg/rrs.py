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
    
    Lj_S  = Fj * bjm_plus(J)
    Dnu_S = -(4*J+6)*B0
    not0  = Lj_S != 0.
    Lj_S  = Lj_S[not0]
    Dnu_S = Dnu_S[not0]
    Lj_A  = Fj * bjm_minus(J)
    not0  = Lj_A != 0.
    Dnu_A =  (4*J-2)*B0
    Lj_A  = Lj_A[not0]
    Dnu_A = Dnu_A[not0]

    norm = Lj_S.sum() + Lj_A.sum()
    return Dnu_S, Lj_S/norm, Dnu_A, Lj_A/norm

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
    
    Lj_S  = Fj * bjm_plus(J)
    Dnu_S = -(4*J+6)*B0
    not0  = Lj_S != 0.
    Lj_S  = Lj_S[not0]
    Dnu_S = Dnu_S[not0]
    Lj_A  = Fj * bjm_minus(J)
    not0  = Lj_A != 0.
    Dnu_A =  (4*J-2)*B0
    Lj_A  = Lj_A[not0]
    Dnu_A = Dnu_A[not0]

    norm = Lj_S.sum() + Lj_A.sum()
    return Dnu_S, Lj_S/norm, Dnu_A, Lj_A/norm

def l_air(lam, theta, T):
    '''
    Air Rotational Raman Spectrum
    
    lam central wavelength in nm
    theta in deg
    T in K
    '''
    Dnu_S_N2, Lj_S_N2, Dnu_A_N2, Lj_A_N2 = l_n2(T)
    Dnu_S_O2, Lj_S_O2, Dnu_A_O2, Lj_A_O2 = l_o2(T)
    Lj_S_N2 *= X_N2
    Lj_A_N2 *= X_N2
    Lj_S_O2 *= X_O2*k_ratio(lam, theta)
    Lj_A_O2 *= X_O2*k_ratio(lam, theta)
    
    nu0 = 1e7/(lam) # nu0 in cm-1
    # compute output lamnda in nm
    lam_S_N2 = 1e7/(nu0+Dnu_S_N2)
    lam_A_N2 = 1e7/(nu0+Dnu_A_N2)
    lam_S_O2 = 1e7/(nu0+Dnu_S_O2)
    lam_A_O2 = 1e7/(nu0+Dnu_A_O2)
    
    norm = Lj_S_N2.sum() + Lj_A_N2.sum() + Lj_S_O2.sum() + Lj_A_O2.sum() 
    
    lam_out= np.concatenate([lam_A_N2, lam_A_O2, lam_S_N2, lam_S_O2])
    L_out  = np.concatenate([Lj_A_N2, Lj_A_O2, Lj_S_N2, Lj_S_O2])/norm
    ii = np.argsort(lam_out)
    
    # return spectrum with increasing wavelengths
    return lam_out[ii], L_out[ii]


def l2d(lam, theta, T):
    '''
    Air Rotational Raman Spectrum
    
    lam central wavelength in nm
    theta in deg
    T in K
    '''
    KK = k_ratio(lam, theta)
    nlam = lam.size
    Dnu_S_N2, Lj_S_N2, Dnu_A_N2, Lj_A_N2 = l_n2(T)
    Dnu_S_O2, Lj_S_O2, Dnu_A_O2, Lj_A_O2 = l_o2(T) 
    norm_N2 = Lj_S_N2.sum() + Lj_A_N2.sum()
    norm_O2 = Lj_S_O2.sum() + Lj_A_O2.sum()
    Lj_S_N2/=norm_N2
    Lj_A_N2/=norm_N2
    Lj_S_O2/=norm_O2
    Lj_A_O2/=norm_O2
    Lj_S_N2 = np.stack([Lj_S_N2]*nlam) * X_N2
    Lj_A_N2 = np.stack([Lj_A_N2]*nlam) * X_N2
    Lj_S_O2 = Lj_S_O2[np.newaxis, :] * X_O2 * KK[:, np.newaxis]
    Lj_A_O2 = Lj_A_O2[np.newaxis, :] * X_O2 * KK[:, np.newaxis]
    norm = np.sum(Lj_S_N2, axis=1) + np.sum(Lj_A_N2, axis=1) + np.sum(Lj_S_O2, axis=1) + np.sum(Lj_A_O2, axis=1)
    Lj_S_N2/=norm[:, np.newaxis]
    Lj_A_N2/=norm[:, np.newaxis]
    Lj_S_O2/=norm[:, np.newaxis]
    Lj_A_O2/=norm[:, np.newaxis]
    #we add also negative unity impulse at zero for removal of elastic
    #L_out  = np.concatenate([Lj_A_N2, Lj_A_O2, Lj_S_N2, Lj_S_O2, np.stack([np.array([0])]*nlam)], axis=1)
    #Dnu_out= np.concatenate([Dnu_A_N2, Dnu_A_O2, Dnu_S_N2, Dnu_S_O2, np.array([0.])])
    L_out  = np.concatenate([Lj_A_N2, Lj_A_O2, Lj_S_N2, Lj_S_O2], axis=1)
    Dnu_out= np.concatenate([Dnu_A_N2, Dnu_A_O2, Dnu_S_N2, Dnu_S_O2])
    
    
    # reorganization with lambda instead od Dnu and increasing order
    nu0 = 1e7/lam
    lam_out = 1e7/(nu0[:, np.newaxis]+Dnu_out[np.newaxis,:])
    ii  = np.argsort(lam_out, axis=1)
    lam_out = np.take_along_axis(lam_out, ii, axis=1)
    L_out   = np.take_along_axis(L_out,   ii, axis=1)
    
    return lam_out, L_out


def l2d_inv(lam, theta, T):
    '''
    Air Inverse Rotational Raman Spectrum
    
    lam central wavelength in nm
    theta in deg
    T in K
    '''
    
    Dnu_S_N2, Lj_S_N2, Dnu_A_N2, Lj_A_N2 = l_n2(T)
    Dnu_S_O2, Lj_S_O2, Dnu_A_O2, Lj_A_O2 = l_o2(T) 
    # reorganization with lambda instead od Dnu
    Dnu_in= np.concatenate([Dnu_A_N2, Dnu_A_O2, Dnu_S_N2, Dnu_S_O2])
    nu0 = 1e7/lam
    lam_in1 = 1e7/(nu0[:, np.newaxis]-Dnu_S_O2[np.newaxis,:])
    lam_in2 = 1e7/(nu0[:, np.newaxis]-Dnu_A_O2[np.newaxis,:])
    nlam = lam.shape[0]
    KK1 = k_ratio(lam_in1, theta)
    KK2 = k_ratio(lam_in2, theta)
    norm_N2 = Lj_S_N2.sum() + Lj_A_N2.sum()
    norm_O2 = Lj_S_O2.sum() + Lj_A_O2.sum()
    Lj_S_N2/=norm_N2
    Lj_A_N2/=norm_N2
    Lj_S_O2/=norm_O2
    Lj_A_O2/=norm_O2
    Lj_S_N2 = np.stack([Lj_S_N2]*nlam) * X_N2
    Lj_A_N2 = np.stack([Lj_A_N2]*nlam) * X_N2
    Lj_S_O2 = Lj_S_O2[np.newaxis, :] * X_O2 * KK1
    Lj_A_O2 = Lj_A_O2[np.newaxis, :] * X_O2 * KK2
    norm = np.sum(Lj_S_N2, axis=1) + np.sum(Lj_A_N2, axis=1) + np.sum(Lj_S_O2, axis=1) + np.sum(Lj_A_O2, axis=1)
    Lj_S_N2/=norm[:, np.newaxis]
    Lj_A_N2/=norm[:, np.newaxis]
    Lj_S_O2/=norm[:, np.newaxis]
    Lj_A_O2/=norm[:, np.newaxis]
    L_in    = np.concatenate([Lj_A_N2, Lj_A_O2, Lj_S_N2, Lj_S_O2], axis=1)

    lam_in = 1e7/(nu0[:, np.newaxis]-Dnu_in[np.newaxis,:])
    ii  = np.argsort(lam_in, axis=1)
    lam_in = np.take_along_axis(lam_in, ii, axis=1)
    L_in   = np.take_along_axis(L_in,   ii, axis=1)
    
    return lam_in, L_in
