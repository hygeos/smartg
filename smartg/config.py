#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join, dirname, realpath
from pathlib import Path
from textwrap import dedent
from core import env, log

'''
SMART-G constant variables

'''

dir_root = dirname(dirname(realpath(__file__)))
NPSTK = 4
# dir_auxdata = join(dir_root, 'auxdata')
dir_auxdata = env.getdir("SMARTG_DIR_DATA", Path("auxdata"))

if not dir_auxdata.is_dir():
    log.error(dedent(
        """Missing auxdata dir in environment variables (SMARTG_DIR_DATA)
        define the environment variable SMARTG_DIR_DATA
        and run $ smartg download_auxdata
        to download the require ancillary data.
        """), e=EnvironmentError)

# Deprecated, will be removed in one of the next release
dir_libradtran = join(dir_auxdata, 'libRadtran')
dir_libradtran_atmmod = join(dir_libradtran, 'data/atmmod/')