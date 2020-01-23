import scipy.constants as cst
import numpy as np

# Atmosphere model: mixing ratio (mol/mol)
X_N2 = 0.788
X_O2 = 0.212

def is_odd(j):
    return j % 2 != 0

# Bates, Planel. Space Sa., Vol.32, No.6, pp. 785-790. 1984 
def Fk_N2(lam):
    '''
    lam in nm
    '''
    return 1.034 + 3.17*1e-4/((lam*1e-3)**2)

def Epsilon_N2(lam):
    '''
    lam in nm
    '''
    return (Fk_N2(lam)-1) * 4.5

def Fk_O2(lam):
    '''
    lam in nm
    '''
    return 1.096 + 1.385*1e-3/((lam*1e-3)**2) + 1.448*1e-4/((lam*1e-3)**4)

def Epsilon_O2(lam):
    '''
    lam in nm
    '''
    return (Fk_O2(lam)-1) * 4.5

def Epsilon_air(lam):
    '''
    lam in nm
    '''
    return Epsilon_N2(lam) * X_N2 + Epsilon_O2(lam) * X_O2

#Kattawar, Astrophysical Journal, Part 1, vol. 243, Feb. 1, 1981, p. 1049-1057.
def f0_air(lam, theta):
    '''
    lam in nm
    theta in deg
    '''
    eps = Epsilon_air(lam)
    c2  = np.cos(np.radians(theta))**2
    num = (180.+13.*eps) + (180.+eps)*c2
    den = (180.+52.*eps) + (180.+4.*eps)*c2
    return num/den
    
def f0_N2(lam, theta):
    '''
    lam in nm
    theta in deg
    '''
    eps = Epsilon_N2(lam)
    c2  = np.cos(np.radians(theta))**2
    num = (180.+13.*eps) + (180.+eps)*c2
    den = (180.+52.*eps) + (180.+4.*eps)*c2
    return num/den

def f0_O2(lam, theta):
    '''
    lam in nm
    theta in deg
    '''
    eps = Epsilon_O2(lam)
    c2  = np.cos(np.radians(theta))**2
    num = (180.+13.*eps) + (180.+eps)*c2
    den = (180.+52.*eps) + (180.+4.*eps)*c2
    return num/den

# Joiner, J., Bhartia, P. K., Cebula, R. P., Hilsenrath, E., McPeters, R. D., & Park, H. (1995). Rotational Raman scattering (Ring effect) 
# in satellite backscatter ultraviolet measurements. Applied Optics, 34(21), 4513. doi:10.1364/ao.34.004513
## !!!! Erreur dans le papier original sur les coeffs de Placzek-Teller Anti Stokes !!!

def K(lam, theta):
    '''
    lam in nm
    theta in deg
    '''
    return (1.-f0_O2(lam,theta))/(1.-f0_N2(lam,theta))

def bjp(J):
    return 3.*(J+1)*(J+2)/2./(2*J+1)/(2*J+3)

def bjm(J):
    b = 3.*J*(J-1)/2./(2*J+1)/(2*J-1)
    b[J<=1] = 0.
    return b
    
def L_O2(T):
    '''
    O2 Rotational Raman Spectrum

    T in K
    '''
    B0 = 1.4378 # cm-1
    J  = np.linspace(0, 36, num=37, dtype=int)
    gj = np.array([1 if is_odd(j) else 0 for j in J])
    Ej = J*(J+1)*cst.h*cst.c*B0*100 # B0 translated in m-1 !!!
    Fj = gj*(2*J+1) * np.exp(-Ej/(cst.k*T))
    
    Lj_S  = Fj * bjp(J)
    Dnu_S = -(4*J+6)*B0
    not0  = Lj_S != 0.
    Lj_S  = Lj_S[not0]
    Dnu_S = Dnu_S[not0]
    Lj_A  = Fj * bjm(J)
    not0  = Lj_A != 0.
    Dnu_A =  (4*J-2)*B0
    Lj_A  = Lj_A[not0]
    Dnu_A = Dnu_A[not0]

    norm = Lj_S.sum() + Lj_A.sum()
    return Dnu_S, Lj_S/norm, Dnu_A, Lj_A/norm

def L_N2(T):
    '''
    N2 Rotational Raman Spectrum

    T in K
    '''
    B0 = 1.9897 # cm-1
    J  = np.linspace(0, 36, num=37, dtype=int)
    gj = np.array([3 if is_odd(j) else 6 for j in J])
    Ej = J*(J+1)*cst.h*cst.c*B0*100 # B0 translated in m-1 !!!
    Fj = gj*(2*J+1) * np.exp(-Ej/(cst.k*T))
    
    Lj_S  = Fj * bjp(J)
    Dnu_S = -(4*J+6)*B0
    not0  = Lj_S != 0.
    Lj_S  = Lj_S[not0]
    Dnu_S = Dnu_S[not0]
    Lj_A  = Fj * bjm(J)
    not0  = Lj_A != 0.
    Dnu_A =  (4*J-2)*B0
    Lj_A  = Lj_A[not0]
    Dnu_A = Dnu_A[not0]

    norm = Lj_S.sum() + Lj_A.sum()
    return Dnu_S, Lj_S/norm, Dnu_A, Lj_A/norm

def L(lam, theta, T):
    '''
    Air Rotational Raman Spectrum
    
    lam central wavelength in nm
    theta in deg
    T in K
    '''
    Dnu_S_N2, Lj_S_N2, Dnu_A_N2, Lj_A_N2 = L_N2(T)
    Dnu_S_O2, Lj_S_O2, Dnu_A_O2, Lj_A_O2 = L_O2(T)
    Lj_S_N2 *= X_N2
    Lj_A_N2 *= X_N2
    Lj_S_O2 *= X_O2*K(lam, theta)
    Lj_A_O2 *= X_O2*K(lam, theta)
    
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


def L2d(lam, theta, T):
    '''
    Air Rotational Raman Spectrum
    
    lam central wavelength in nm
    theta in deg
    T in K
    '''
    KK = K(lam, theta)
    nlam = lam.size
    Dnu_S_N2, Lj_S_N2, Dnu_A_N2, Lj_A_N2 = L_N2(T)
    Dnu_S_O2, Lj_S_O2, Dnu_A_O2, Lj_A_O2 = L_O2(T) 
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


def L2d_inv(lam, theta, T):
    '''
    Air Inverse Rotational Raman Spectrum
    
    lam central wavelength in nm
    theta in deg
    T in K
    '''
    
    Dnu_S_N2, Lj_S_N2, Dnu_A_N2, Lj_A_N2 = L_N2(T)
    Dnu_S_O2, Lj_S_O2, Dnu_A_O2, Lj_A_O2 = L_O2(T) 
    # reorganization with lambda instead od Dnu
    Dnu_in= np.concatenate([Dnu_A_N2, Dnu_A_O2, Dnu_S_N2, Dnu_S_O2])
    nu0 = 1e7/lam
    lam_in1 = 1e7/(nu0[:, np.newaxis]-Dnu_S_O2[np.newaxis,:])
    lam_in2 = 1e7/(nu0[:, np.newaxis]-Dnu_A_O2[np.newaxis,:])
    nlam = lam.shape[0]
    KK1 = K(lam_in1, theta)
    KK2 = K(lam_in2, theta)
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
