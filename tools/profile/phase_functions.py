#!/usr/bin/env python
# encoding: utf-8


from numpy import sin, cos, pi
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
    def __init__(self, ang, phase, header=[], degrees=False):
        self.ang = ang
        self.phase = phase
        self.header = header
        self.N = ang.shape[0]
        self.degrees = degrees
        assert phase.shape == (self.N, 4)

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
            fo.write('{:7.2E} {:11.5E} {:11.5E} {:11.5E} {:11.5E}\n'.format(
                    ang,
                    self.phase[i,0],
                    self.phase[i,1],
                    self.phase[i,2],
                    self.phase[i,3],
                    ))
        fo.close()


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


