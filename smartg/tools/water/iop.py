#!/usr/bin/env python
# encoding: utf-8


import numpy as np
import tempfile


class IOP(object):
    '''
    abstract base class for IOP models
    '''
    def write(self, wl,  dir_profile, dir_phases, dir_list_phases, Zbottom=10000., Nlayer=2):
        '''
        write profiles and phase functions at bands wl (list)
        returns a tuple (profile, phases) where
        - profiles is the filename containing the concatenated profiles
        - phase is a file containing the list pf phase functions
        -Zbottom is the depth of the bottom in m
        -Nlayer is the number of layer in the ocean
        '''
        # convert to list if wl is a scalar
        if isinstance(wl, (float, int)):
            wl = [wl]

        iops, phases = self.calc_bands(wl)

        # write the profiles
        profil_oce = tempfile.mktemp(dir=dir_profile, prefix='profil_oce_')
        f = open(profil_oce, 'w')
        for (atot, btot, ipha) in iops:
                f.write('# I   DEPTH    H(I)    SSA(I)  IPHA\n')
                f.write('0 0. 0. 1. 0\n')
                for n in range(Nlayer):
                    z= (n+1.)*Zbottom/Nlayer
                    tau=z*(atot+btot) 
                    f.write('{} {} {} {} {}\n'.format(n+1,z,-tau,btot/(atot+btot), ipha))
                #f.write('1 1000. -1.e10 {} {}\n'.format(btot/(atot+btot), ipha))
        f.close()

        # write the phase functions and list of phase functions
        file_list_pf_ocean = tempfile.mktemp(dir=dir_list_phases, prefix='list_pf_ocean_')
        fp = open(file_list_pf_ocean, 'w')
        for p in phases:
            file_phase = tempfile.mktemp(dir=dir_phases, prefix='pf_ocean_')
            p.write(file_phase)
            fp.write(file_phase+'\n')
        fp.close()

        return profil_oce, file_list_pf_ocean


    def calc_bands(self, wl):
        '''
        calculate atot, btot at bands wl, and phase function at pfwav (or wl if
        pfwav is None)

        returns a list of (a, b, iphase) and a list of phase functions
        '''

        if (self.last is not None) and (self.last[0] == list(wl)):
            return (self.last[1], self.last[2])

        #
        # calculate the phase functions
        #
        if self.pfwav is None:
            pfwav = wl
        else:
            pfwav = self.pfwav

        phases_1 = []
        phases_2 = []
        phases_tot = []
        for w in pfwav:
            if self.verbose: print 'Calculate ocean phase function at', w

            _, [(b1, P1), (b2, P2)] = self.calc(w)

            phases_1.append(P1)
            phases_2.append(P2)
            ptot = (b1*P1 + b2*P2)/(b1+b2)
            phases_tot.append(ptot)

        #
        # calculate the IOPs
        #
        iops = []
        for w in wl:
            ipha = np.abs(w - np.array(pfwav)).argmin()

            a, [(b1, _), (b2, _)] = self.calc(w, skip_phase=True)

            # adjust the scattering coefficients according to the truncation coefficient
            b = b1*phases_1[ipha].coef_trunc + b2*phases_2[ipha].coef_trunc

            iops.append((a, b, ipha))

        self.last = (list(wl), iops, phases_tot)

        return iops, phases_tot


