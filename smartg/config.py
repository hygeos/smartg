"""
SMART-G module-level constants.

This module centralizes configuration values shared across the SMART-G
package. It is imported lazily by submodules that need these values, so
that simply importing :mod:`smartg` does not trigger side effects.

The module reads ``SMARTG_DIR_AUXDATA`` from the process environment
(or a project-level ``.env`` file located at the repository root) at
import time. Importing this module will raise :class:`NameError` if the
variable is not set, with a chained ``KeyError`` indicating the missing
environment variable.

Exposes
-------
DIR_ROOT : pathlib.Path
    Absolute path to the SMART-G repository root (parent of the
    ``smartg`` package).
DIR_AUXDATA : pathlib.Path
    Absolute path to the directory holding SMART-G auxiliary data files
    (aerosols, surface BRDFs, etc.). Resolved from the
    ``SMARTG_DIR_AUXDATA`` environment variable.
"""

from os import environ
from pathlib import Path
from dotenv import load_dotenv


DIR_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(DIR_ROOT / ".env")  # To consider .env file


try:
    dir_auxdata = environ["SMARTG_DIR_AUXDATA"]
    DIR_AUXDATA = Path(dir_auxdata)
except KeyError as err:
    raise NameError(
        "The environment variable 'SMARTG_DIR_AUXDATA' does not exist!"
    ) from err
