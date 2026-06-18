#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ
from pathlib import Path
from dotenv import load_dotenv

"""
SMART-G constant variables

"""

DIR_ROOT = Path(__file__).resolve().parent.parent
NPSTK = 4

load_dotenv(DIR_ROOT / ".env")  # To consider .env file


try:
    dir_auxdata = environ["SMARTG_DIR_AUXDATA"]
    DIR_AUXDATA = Path(dir_auxdata)
except KeyError as err:
    raise NameError(
        "The environment variable 'SMARTG_DIR_AUXDATA' does not exist!"
    ) from err
