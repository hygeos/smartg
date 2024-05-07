#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join, dirname, realpath

'''
SMART-G constant variables

'''

dir_root = dirname(dirname(realpath(__file__)))
NPSTK = 4
dir_auxdata = join(dir_root, 'auxdata')

# Deprecated, will be removed in one of the next release
dir_libradtran = join(dir_auxdata, 'libRadtran')
dir_libradtran_atmmod = join(dir_libradtran, 'data/atmmod/')