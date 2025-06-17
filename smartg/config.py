#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ
from os.path import join, dirname, realpath, isdir
from dotenv import load_dotenv
import warnings

'''
SMART-G constant variables

'''

dir_root = dirname(dirname(realpath(__file__)))
NPSTK = 4
dir_auxdata_old = join(dir_root, 'auxdata')

load_dotenv(dir_root) # To consider .env file

# Allow old way in v1.1, but remove it in next releases
try:
    dir_auxdata_new = environ['SMARTG_DIR_AUXDATA']
    dir_auxdata = dir_auxdata_new
except KeyError:
    if not (isdir(dir_auxdata_old)):
        raise NameError("The environment variable 'SMARTG_DIR_AUXDATA' does not exist!")
    else:
        warnings.simplefilter('always', DeprecationWarning)
        warn_message = "\nThe 'SMARTG_DIR_AUXDATA' variable does not exist, but the folder auxdata has \n" + \
                       "been found in the root directory and is used instead. This is deprecated, and will \n" + \
                       "not be allowed in next releases. Please give the auxdata directory path by \n" + \
                       "using the environment variable 'SMARTG_DIR_AUXDATA' instead."
        warnings.warn(warn_message, DeprecationWarning)
        dir_auxdata = dir_auxdata_old

# TODO for constant variable must use uppercase 
dir_libradtran = join(dir_auxdata, 'libRadtran')
dir_libradtran_atmmod = join(dir_libradtran, 'data/atmmod/')