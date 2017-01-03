#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from os.path import join, dirname, realpath, isdir
from subprocess import check_call

'''
SMART-G project-wide configuration

Execute this module to install libRadtran data files
'''

dir_root = dirname(realpath(__file__))

# number of Stokes parameters of the radiation field
NPSTK = 4

#
# auxiliary files (from libradtran)
# http://www.libradtran.org/
#
dir_auxdata = join(dir_root, 'auxdata')
dir_libradtran = join(dir_auxdata, 'libRadtran')
dir_libradtran_reptran =  join(dir_libradtran, 'data/correlated_k/reptran/')
dir_libradtran_opac =  join(dir_libradtran, 'data/aerosol/OPAC/')
dir_libradtran_atmmod = join(dir_libradtran, 'data/atmmod/')
dir_libradtran_crs = join(dir_libradtran, 'data/crs/')



#
# setup
#
def check_libradtran():
    if (isdir(dir_libradtran)
            and isdir(dir_libradtran_reptran)
            and isdir(dir_libradtran_opac)
            and isdir(dir_libradtran_atmmod)
            and (dir_libradtran_crs)):
        return

    cmds = []
    cmds.append(('wget http://www.libradtran.org/download/libRadtran-2.0.1.tar.gz', dir_auxdata))
    cmds.append(('tar xzvf libRadtran-2.0.1.tar.gz', dir_auxdata))
    cmds.append(('rm -fv libRadtran-2.0.1.tar.gz', dir_auxdata))
    cmds.append(('ln -sfn libRadtran-2.0.1 libRadtran', dir_auxdata))

    cmds.append(('wget http://www.meteo.physik.uni-muenchen.de/\~libradtran/lib/exe/fetch.php\?media\=download:optprop_v2.1.tar.gz -O optprop_v2.1.tar.gz', dir_libradtran))
    cmds.append(('tar xzvf optprop_v2.1.tar.gz', dir_libradtran))
    cmds.append(('rm -fv optprop_v2.1.tar.gz', dir_libradtran))
    cmds.append(('wget http://www.meteo.physik.uni-muenchen.de/~libradtran/lib/exe/fetch.php?media=download:reptran_2015_all.tar.gz -O reptran_2015_all.tar.gz', dir_libradtran))
    cmds.append(('tar xzvf reptran_2015_all.tar.gz', dir_libradtran))
    cmds.append(('rm -fv reptran_2015_all.tar.gz', dir_libradtran))
    cmds.append(('cp -v data/wc/mie/*.cdf data/aerosol/OPAC/optprop/', dir_libradtran))

    print('Libradtran base directory does not exist:')
    print(dir_libradtran)
    print()
    print('Either link to an existing libradtran installation, or')
    print('create it with the following commands:')
    print()

    dir_prev = ''
    for cmd, d in cmds:
        if d != dir_prev:
            print('cd '+d)
            dir_prev = d
        print(cmd)

    print()
    r = input('Execute these commands ? (y/n) ')
    if r == 'y':
        for cmd, d in cmds:
            print(cmd)
            if check_call([cmd], cwd=d, shell=True):
                print('command failed, aborting')
                break


if __name__ == "__main__":
    check_libradtran()

