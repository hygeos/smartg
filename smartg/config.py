#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ
from pathlib import Path
from dotenv import load_dotenv
import warnings

'''
SMART-G constant variables

'''

dir_root = Path(__file__).resolve().parent.parent
NPSTK = 4
dir_auxdata_old = dir_root / 'auxdata'

load_dotenv(dir_root / '.env') # To consider .env file

# Allow old way in v1.1, but remove it in next releases
try:
    dir_auxdata_new = environ['SMARTG_DIR_AUXDATA']
    DIR_AUXDATA = Path(dir_auxdata_new)
except KeyError:
    if not dir_auxdata_old.is_dir():
        raise NameError("The environment variable 'SMARTG_DIR_AUXDATA' does not exist!")
    else:
        warnings.simplefilter('always', DeprecationWarning)
        warn_message = "\nThe 'SMARTG_DIR_AUXDATA' variable does not exist, but the folder auxdata has \n" + \
                       "been found in the root directory and is used instead. This is deprecated, and will \n" + \
                       "not be allowed in next releases. Please give the auxdata directory path by \n" + \
                       "using the environment variable 'SMARTG_DIR_AUXDATA' instead."
        warnings.warn(warn_message, DeprecationWarning)
        DIR_AUXDATA = dir_auxdata_old

# TODO for constant variable must use uppercase 
dir_libradtran = DIR_AUXDATA / 'libRadtran'
dir_libradtran_atmmod = dir_libradtran / 'data' / 'atmmod'