import numpy as np

def Gauss(ks, Aj, kj, Dkj):

    return Aj * 1./Dkj * np.exp(-4*np.log(2)*(ks-kj)**2/Dkj**2)

def fR(ks):
    A  = np.array([0.41, 0.39, 0.10, 0.10])
    k  = np.array([3250., 3425., 3530., 3625.])
    Dk = np.array([210., 175., 140., 140.])
    norm = np.sum(A) * np.sqrt(np.pi/4/np.log(2))
    norm = 1./norm
    Su=np.zeros_like(ks)
    for j in range(4):
        Su+= Gauss(ks, A[j], k[j], Dk[j])

    return Su*norm


def V2d(lam, Nl=16):
    '''
    Ocean Vibrational Raman Spectrum

    lam central wavelength in nm
    '''
    k   = 1e7/lam   # cm-1
    k0  = k - 2950. # cm-1
    k1  = k - 3850. # cm-1
    w0  = 1e7/k0
    w1  = 1e7/k1
    wgrid = np.linspace(w0, w1, num=Nl, dtype=np.float32)
    ks    = 1e7*(1./lam[np.newaxis,:]-1./wgrid)
    
    return wgrid, 1e7/wgrid**2 * fR(ks)


def V2d_inv(lam, Nl=16):
    '''
    Inverse Ocean Vibrational Raman Spectrum

    lam central wavelength in nm
    '''
    k   = 1e7/lam # cm-1
    k0  = k + 3850. # cm-1
    k1  = k + 2950. # cm-1
    w0  = 1e7/k0
    w1  = 1e7/k1
    wgrid = np.linspace(w0, w1, num=Nl, dtype=np.float32).T
    ks    = 1e7*(1./wgrid - 1./lam[:,np.newaxis])
    
    return wgrid, 1e7/wgrid**2 * fR(ks)
