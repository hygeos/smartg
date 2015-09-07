#!/usr/bin/env python
# encoding: utf-8


from numpy import sin, cos, pi, allclose
from os.path import exists, dirname
from os import makedirs, remove


def fournierForand(ang, n, mu):
    '''
    Fournier-Forand phase function
    Parameters:
        ang: angle in radians
        n: index of refraction of the particles
        mu: slope parameter of the hyperbolic dustribution
    Normalized to 4pi
    See http://www.oceanopticsbook.info/view/scattering/the_fournierforand_phase_function
    '''
    v = (3-mu)/2
    delta = 4/( 3*(n-1)*(n-1) )*sin(ang/2)*sin(ang/2)
    delta180 = 4/( 3*(n-1)*(n-1) )*sin(pi/2)*sin(pi/2)

    res = 1/( 4*pi*(1-delta)*(1-delta)*(delta**v) )*( v*(1-delta) - (1-(delta**v)) + ( delta*(1-(delta**v)) - v*(1-delta) )*1/(sin(ang/2)*sin(ang/2)) ) + (1-(delta180**v))/(16*pi*(delta180-1)*(delta180**v)) * (3*cos(ang)*cos(ang) - 1)
    res *= 4*pi

    return res

def fournierForandB(n, mu):
    '''
    backscatter fraction of the Fournier-Forand phase function
    '''
    d90 = 4./(3.*(n-1.)**2)*(sin(pi/4.)**2)
    v = (3.-mu)/2.
    B   = 1 - (1 - d90**(v+1) - 0.5*(1-d90**v))/((1-d90)*d90**v)
    return B

def henyeyGreenstein(angle, g):
    '''
    Henyey-Greenstein phase function
    Parameters:
        angle: angle in radians
        g: asymmetry coefficient
           (0: isotropic ; 1: highly peaked)
    Normalized to 4pi
    See http://www.oceanopticsbook.info/view/scattering/the_henyeygreenstein_phase_function
    '''
    return (1 - g*g)/((1 + g*g - 2*g*cos(angle))**1.5)


class PhaseFunction(object):
    '''
    Store a phase function

    Arguments:
        ang: array of angles
        phase: phase function (shape (N, 4))
        header: list of strings added at the header of the output file
        degrees: angle is in degrees instead of radians

    '''
    def __init__(self, ang, phase, header=[], degrees=False, coef_trunc=1.):
        self.ang = ang
        self.phase = phase
        self.header = header
        self.N = ang.shape[0]
        self.degrees = degrees
        self.coef_trunc = coef_trunc
        assert phase.shape == (self.N, 4)

    def ang_in_rad(self):
        if self.degrees == True:
            return self.ang * (pi/180.)
        else:
            return self.ang

    def write(self, filename, overwrite=False):
        '''
        write the phase function to a text file filename
        '''

        if exists(filename):
            if overwrite:
                print 'INFO: overwriting {}'.format(filename)
                remove(filename)
            else:
                raise Exception('File {} exists'.format(filename))

        if not exists(dirname(filename)):
            makedirs(dirname(filename))

        #
        # write the phase function
        #
        fo = open(filename, 'w')

        # write the header
        for line in self.header:
            if not line.startswith('#'):
                line = '#'+line
            if not line.endswith('\n'):
                line = line+'\n'
            fo.write(line)

        for i in xrange(self.N):
            ang = self.ang[i]
            if not self.degrees:
                ang *= 180./pi
            fo.write('{:11.5E} {:11.5E} {:11.5E} {:11.5E} {:11.5E}\n'.format(
                    ang,
                    self.phase[i,0],
                    self.phase[i,1],
                    self.phase[i,2],
                    self.phase[i,3],
                    ))
        fo.close()

    def check_compatible(self, other):
        assert self.N == other.N
        assert self.degrees == other.degrees
        assert allclose(self.ang, other.ang)

    def __add__(self, other):
        if isinstance(other, PhaseFunction):
            self.check_compatible(other)
            return PhaseFunction(self.ang, self.phase+other.phase,
                    header=self.header+other.header, degrees=self.degrees,
                    coef_trunc=None)
        elif isinstance(other, float):
            return PhaseFunction(self.ang, self.phase+other,
                    header=self.header+other.header, degrees=self.degrees,
                    coef_trunc=None)
        else:
            raise Exception('Error')

    def __radd__(self, other):
        if isinstance(other, PhaseFunction):
            self.check_compatible(other)
            return PhaseFunction(self.ang, self.phase+other.phase,
                    header=self.header+other.header, degrees=self.degrees,
                    coef_trunc=None)
        elif isinstance(other, float):
            return PhaseFunction(self.ang, self.phase+other,
                    header=self.header, degrees=self.degrees,
                    coef_trunc=None)
        else:
            raise Exception('Error')

    def __rmul__(self, other):
        if isinstance(other, float):
            return PhaseFunction(self.ang, self.phase*other,
                    header=self.header, degrees=self.degrees,
                    coef_trunc=None)
        elif isinstance(other, PhaseFunction):
            self.check_compatible(other)
            return PhaseFunction(self.ang, self.phase*other.phase,
                    header=self.header+other.header, degrees=self.degrees,
                    coef_trunc=None)
        else:
            raise Exception('Error')

    def __div__(self, other):
        if isinstance(other, float):
            return PhaseFunction(self.ang, self.phase/other,
                    header=self.header, degrees=self.degrees,
                    coef_trunc=None)
        else:
            raise Exception('Error')

    def plot(self):
        from pylab import semilogy, grid
        ang = self.ang
        if self.degrees:
            semilogy(ang, self.phase[:,0]+self.phase[:,1])
        else:
            semilogy(ang*180.3/pi, self.phase[:,0]+self.phase[:,1])
        grid()

def test():
    from pylab import show, semilogy, grid

    th = linspace(1e-7, pi, 1e6)
    semilogy(th, fournierForand(th, 1.15, 3.5), label='FF')
    semilogy(th, henyeyGreenstein(th, 0.9), label='HG')
    grid()
    show()

if __name__ == '__main__':
    from numpy import trapz, linspace

    th = linspace(1e-7, pi, 1e6)
    print 'Normalization'
    print 'HG', trapz(henyeyGreenstein(th, 0.5)*sin(th), x=th)
    print 'FF', trapz(fournierForand(th, 1.15, 3.5)*sin(th), x=th)

    # test()


