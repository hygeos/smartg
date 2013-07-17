#!/usr/bin/env python
# vim:fileencoding=utf-8




'''
Calculate the ocean water absorption, scattering coefficients and phase
function using the wavelength and the chlorophyll concentration
'''


from numpy import sin, cos, pi, array, exp
from numpy import arange, zeros, log10, sqrt
from sys import argv


wl_1 = array([350., 355., 360., 365., 370., 375., 380., 385., 390., 395.,
              400., 405., 410., 415., 420., 425., 430., 435., 440., 445.,
              450., 455., 460., 465., 470., 475., 480., 485., 490., 495.,
              500., 505., 510., 515., 520., 525., 530., 535., 540., 545.,
              550., 555., 560., 565., 570., 575., 580., 585., 590., 595.,
              600., 605., 610., 615., 620., 625., 630., 635., 640., 645.,
              650., 655., 660., 665., 670., 675., 680., 685., 690., 695., 700.]);

ah2o = array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01137, 0.00941, 0.00851, 0.00813, 0.00663, 0.0053, 0.00473,
              0.00444, 0.00454, 0.00478, 0.00495, 0.0053,   0.00635, 0.00751, 0.00922, 0.00962, 0.00979,
              0.01011, 0.0106, 0.0114, 0.0127, 0.0136, 0.015, 0.0173, 0.0204, 0.0256, 0.0325, 0.0396,
              0.0409, 0.0417, 0.0434, 0.0452, 0.0474, 0.0511, 0.0565, 0.0596, 0.0619, 0.0642, 0.0695,
              0.0772, 0.0896, 0.11, 0.1351, 0.1672, 0.2224, 0.2577, 0.2644, 0.2678, 0.2755, 0.2834,
              0.2916, 0.3012, 0.3108, 0.325, 0.34, 0.371, 0.41, 0.429 , 0.439, 0.448, 0.465, 0.486,
              0.516, 0.559, 0.624 ] )

A_bricaud95 = array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0263, 0.0285, 0.0313, 0.03375, 0.0356, 0.03655, 0.0386, 0.0397, 0.0403, 0.03865,
                     0.0371, 0.0356, 0.035, 0.0341, 0.0332, 0.0315, 0.0301, 0.02875, 0.0274, 0.02535,
                     0.023, 0.0204, 0.018, 0.01595, 0.0143, 0.01285, 0.0117, 0.0106, 0.0097, 0.0088 ,
                     0.008, 0.007, 0.0062, 0.0056, 0.0053, 0.0052, 0.0053, 0.0055, 0.0056, 0.0056 ,
                     0.0054, 0.0055, 0.0057, 0.0061, 0.0065, 0.00675, 0.0071, 0.00745, 0.0077, 0.00795,
                     0.0083, 0.0092, 0.0115, 0.01525, 0.0189, 0.0201, 0.0182, 0.01345, 0.0083, 0.0049, 0.003])

B_bricaud95 = array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.282, 0.2815, 0.283 , 0.292 , 0.299 , 0.3145, 0.314 , 0.326 , 0.332 , 0.3515,
                     0.359, 0.3665, 0.365 , 0.3685, 0.368 , 0.3765, 0.377 , 0.3745, 0.361 , 0.345 ,
                     0.321, 0.294, 0.26  , 0.2305, 0.196 , 0.1675, 0.139 , 0.114 , 0.09  , 0.0695,
                     0.052, 0.0315, 0.016 , 0.0085, 0.005 , 0.02  , 0.035 , 0.053 , 0.073 , 0.0905,
                     0.092, 0.084, 0.071 , 0.0645, 0.064 , 0.0725, 0.078 , 0.086 , 0.098 , 0.116 ,
                     0.124, 0.123, 0.121 , 0.134 , 0.149 , 0.158 , 0.155 , 0.131 , 0.086 , 0.0285, -0.034])

ee = array([0.77800, 0.76700, 0.75600, 0.73700, 0.72000, 0.70000, 0.68500, 0.67300, 0.67000, 0.66000,
          0.64358, 0.64776, 0.65175, 0.65555, 0.65917, 0.66259, 0.66583, 0.66889, 0.67175, 0.67443,
          0.67692, 0.67923, 0.68134, 0.68327, 0.68501, 0.68657, 0.68794, 0.68903, 0.68955, 0.68947,
          0.68880, 0.68753, 0.68567, 0.68320, 0.68015, 0.67649, 0.67224, 0.66739, 0.66195, 0.65591,
          0.64927, 0.64204, 0.64000, 0.63000, 0.62300, 0.61500, 0.61000, 0.61400, 0.61800, 0.62200,
          0.62600, 0.63000, 0.63400, 0.63800, 0.64200, 0.64700, 0.65300, 0.65800, 0.66300, 0.66700,
          0.67200, 0.67700, 0.68200, 0.68700, 0.69500, 0.69700, 0.69300, 0.66500, 0.64000,0.62000,0.60000 ])

Chi = array([0.15300, 0.14900, 0.14400, 0.14000, 0.13600, 0.13100, 0.12700, 0.12300, 0.11900, 0.11800,
          0.11748, 0.12066, 0.12259, 0.12326, 0.12269, 0.12086, 0.11779, 0.11372, 0.10963, 0.10560,
          0.10165, 0.09776, 0.09393, 0.09018, 0.08649, 0.08287, 0.07932, 0.07584, 0.07242, 0.06907,
          0.06579, 0.06257, 0.05943, 0.05635, 0.05341, 0.05072, 0.04829, 0.04611, 0.04419, 0.04253,
          0.04111, 0.03996, 0.03900, 0.03750, 0.03600, 0.03400, 0.03300, 0.03280, 0.03250, 0.03300,
          0.03400, 0.03500, 0.03600, 0.03750, 0.03850, 0.04000, 0.04200, 0.04300, 0.04400, 0.04450,
          0.04500, 0.04600, 0.04750, 0.04900, 0.05150, 0.05200, 0.05050, 0.04400, 0.03900,0.03400,0.03000])

Kw = array([0.02710, 0.02380, 0.02160, 0.01880, 0.01770, 0.01595, 0.01510, 0.01376, 0.01271, 0.01208,
          0.01042, 0.00890, 0.00812, 0.00765, 0.00758, 0.00768, 0.00770, 0.00792, 0.00885, 0.00990,
          0.01148, 0.01182, 0.01188, 0.01211, 0.01251, 0.01320, 0.01444, 0.01526, 0.01660, 0.01885,
          0.02188, 0.02701, 0.03385, 0.04090, 0.04214, 0.04287, 0.04454, 0.04630, 0.04846, 0.05212,
          0.05746, 0.06053, 0.06280, 0.06507, 0.07034, 0.07801, 0.09038, 0.11076, 0.13584, 0.16792,
          0.22310, 0.25838, 0.26506, 0.26843, 0.27612, 0.28400, 0.29218, 0.30176, 0.31134, 0.32553,
          0.34052, 0.37150, 0.41048, 0.42947, 0.43946, 0.44844, 0.46543, 0.48642, 0.51640,0.55939,0.62438])


def fournierForand(ang, n, mu):
    '''
    Fournier-Forand phase function
    '''
    v = (3-mu)/2
    delta = 4/( 3*(n-1)*(n-1) )*sin(ang/2)*sin(ang/2)
    delta180 = 4/( 3*(n-1)*(n-1) )*sin(pi/2)*sin(pi/2)

    res = 1/( 4*pi*(1-delta)*(1-delta)*(delta**v) )*( v*(1-delta) - (1-(delta**v)) + ( delta*(1-(delta**v)) - v*(1-delta) )*1/(sin(ang/2)*sin(ang/2)) ) + (1-(delta180**v))/(16*pi*(delta180-1)*(delta180**v)) * (3*cos(ang)*cos(ang) - 1)
    res *= 4*pi

    return res

def henyeyGreenstein(asym, angle):
    '''
    Henyey-Greenstein phase function
    '''
    return (1 - asym*asym)/((1 + asym*asym - 2*asym*cos(angle))**1.5)


def main():

    #
    # read the arguments
    #
    if len(argv) != 5:
        print 'Arguments: water_model.py chl wl NANG tronc_ang'
        print 'where:'
        print '    chl is the chlorophyll concentration in mg/m3'
        print '    wl is the wavelength in nm'
        print '    NANG is the number if angles for the phase function'
        print '    tronc_ang is the truncation angle in degrees'
        print
        print 'Example: water_model.py 0.1 500 72001 5.'
        exit(1)

    chl = float(argv[1])
    wl = float(argv[2])
    NANG = int(argv[3])
    ang_trunc = float(argv[4])

    #
    # wavelength index
    #
    if (wl < wl_1[0]) or (wl > wl_1[-1]):
        print 'Error, wavelength {} is out of range ({}, {})'.format(wl, wl_1[0], wl_1[-1])
        exit(1)
    i1 = int((wl - wl_1[0])/(wl_1[1] - wl_1[0]))


    #
    # pure water coefficients
    #
    a0 = ah2o[i1]
    b0 = 19.3e-4*((wl_1[i1]/550.)**-4.3)


    #
    # phytoplankton
    #
    anap440 = 0.0124*(chl**0.724)
    anap = anap440*exp(-0.011*(wl-440.) )
    aphi = A_bricaud95[i1]*(chl**(1.-B_bricaud95[i1]))
    a1 = anap + aphi
    b1 = 0.416*(chl**0.766)*550./wl


    #
    # backscattering coefficient
    #
    if chl < 2:
        v = 0.5*(log10(chl) - 0.3)
    else:
        v = 0
    bb1 = 0.002 + 0.01*( 0.5-0.25*log10(chl))*((wl/550)**v)
    r1 = (bb1 - 0.002)/0.028

    #
    # phase function
    #
    ang = pi * arange(NANG, dtype='float64')/(NANG-1)    # angle in radians

    # pure water
    pf0 = zeros((NANG, 4), dtype='float64') # pure water phase function
    pf0[:,0] = 0.75
    pf0[:,1] = 0.75 * cos(ang)**2
    pf0[:,2] = 0.75 * cos(ang)
    pf0[:,3] = 0.

    # particles (troncature)
    itronc = int(NANG * ang_trunc/180.)
    pf1 = zeros((NANG, 4), dtype='float64') # pure water phase function
    pf1[itronc:,0] = 0.5*(r1*fournierForand(ang[itronc:],1.117,3.695) +(1-r1)*fournierForand(ang[itronc:],1.05,3.259))
    pf1[:itronc,0] = 0.5*(r1*fournierForand(ang[itronc ],1.117,3.695) +(1-r1)*fournierForand(ang[itronc ],1.05,3.259))
    pf1[:,1] = pf1[:,0]
    pf1[:,2] = 0.
    pf1[:,3] = 0.

    # normalization after truncation
    integ_ff = 0
    for iang in xrange(1, NANG):
        dtheta = ang[iang] - ang[iang-1]
        pm1 = pf1[iang-1,0] + pf1[iang-1,1]
        pm2 = pf1[iang,0] + pf1[iang,1]
        sin1 = sin(ang[iang-1])
        sin2 = sin(ang[iang])
        integ_ff += dtheta*((sin1*pm1+sin2*pm2)/3. + (sin1*pm2+sin2*pm1)/6.)
    rat1 = integ_ff/2.
    pf1 *= 1/rat1
    b1 *= rat1


    #
    # total coefficients
    #
    btot = b0 + b1

    if False:
        # 1) absorption calculated from individual absorption coefficients (Bricaud et al)
        atot = a0 + a1
    else:
        # 2) absorption deduced from Morel's total attenuation and Kirk's formula
        Kd = Kw[i1] + Chi[i1]*(chl**ee[i1])
        delta = (0.256*(b0+b1/rat1))*(0.256*(b0+b1/rat1)) + 4*Kd*Kd
        atot = 0.5*(-0.256*(b0+b1/rat1) + sqrt(delta))

    # total scattering function
    pf = (b0*pf0 + b1*pf1)/btot


    #
    # display results
    #
    print '# chlorophyll concentration: {}'.format(chl)
    print '# wavelength: {}'.format(wl)
    print '# total absorption coefficient: {}'.format(atot)
    print '# total scattering coefficient: {}'.format(btot)
    print '# truncating at {} deg'.format(ang_trunc*180/pi)
    for i in xrange(NANG):
        print '{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                ang[i] * 180/pi,
                pf[i,0],
                pf[i,1],
                pf[i,2],
                pf[i,3],
                )




if __name__ == '__main__':
    main()
