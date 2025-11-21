#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
from os.path import join, dirname, realpath
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import tarfile

dir_root = dirname(dirname(realpath(__file__)))

# auxdata source: HYGEOS
AER_URL = "https://docs.hygeos.com/s/8PnKXFXQbmYyTte/download"
ACS_URL = "https://docs.hygeos.com/s/HwotAHPstdCCKcJ/download"
ATM_URL = "https://docs.hygeos.com/s/z6MRf9g66WmWeBA/download"
STP_URL = "https://docs.hygeos.com/s/NW42DNPtKw3NNW7/download"
VALID_URL = "https://docs.hygeos.com/s/6EPBqwebn94NYPq/download"
WATER_URL = "https://docs.hygeos.com/s/3NKP5tMsHKnNRpt/download"
KDIS_URL = "https://docs.hygeos.com/s/CHTFFgHe6to39CR/download"
CLOUD_URL = "https://docs.hygeos.com/s/agDWDy998j64SHf/download"

# some data (mystic res and opt_prop) are taken from: 
# https://www.meteo.physik.uni-muenchen.de/~iprt/doku.php?id=intercomparisons:intercomparisons
IPRT_URL = "https://docs.hygeos.com/s/i4QaxtpjSfjwtNk/download"

# reptran source: http://www.libradtran.org
# the libradtran url is better to be sure to get the last versions of reptran look-up tables
REPTRAN_URL = "http://www.meteo.physik.uni-muenchen.de/~libradtran/lib/exe/fetch.php?media=download:reptran_2017_all.tar.gz"
# since the above url is not always stable, we provide an alternative url (but can be outdated!!)
REPTRAN_URL_HYG = "https://docs.hygeos.com/s/jHKMcZZmkf6xy7D/download"

AUXDATA_DICT = {
    "aer": AER_URL,
    "acs": ACS_URL,
    "atm": ATM_URL,
    "STP": STP_URL,
    "valid": VALID_URL,
    "water": WATER_URL,
    "kdis": KDIS_URL,
    "cld": CLOUD_URL,
    "IPRT": IPRT_URL,
    "reptran": REPTRAN_URL,
}


def safe_download(url, outfile):
    """Cross-platform download with progress."""
    import urllib.request

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            print(f"\rDownloading {outfile}: {percent}%", end="")
        else:
            downloaded = count * block_size
            print(f"\rDownloaded {downloaded/1024/1024:.1f} MB...", end="")

    print(f"Downloading {url} → {outfile}")
    urlretrieve(url, outfile, reporthook)
    print("\nDownload complete.")


def extract_zip(zfile, dest):
    """Cross-platform unzip (verbose)."""
    with zipfile.ZipFile(zfile, 'r') as z:
        print(f"Extracting ZIP {zfile} → {dest}")
        for name in z.namelist():
            print("  extracting:", name)
        z.extractall(dest)


def extract_tar(tfile, dest):
    """Cross-platform untar (verbose)."""
    with tarfile.open(tfile, "r:gz") as tar:
        print(f"Extracting TAR {tfile} → {dest}")
        for member in tar.getmembers():
            print("  extracting:", member.name)
        tar.extractall(dest)


def download(savepath, data_type="all"):
    """
    Dowload the SMART-G auxiliary data

    Parameters
    ----------
    savepath : str
        The path where the data will be saved
    data_type : str, optional
        The type of data to download, can be: "all", "aer", "acs", "atm", "STP", "valid", 
        "water", "kdis", "cld", "IPRT". Default is "all". Definitions:

        * all -> all the available data
        * aer -> aerosols data
        * acs -> absorption cross section coefficients data
        * atm -> atmosphere profils
        * STP -> STP (Solar Power Tower) files with heliostat positions
        * valid -> validation files
        * water -> water files needed for some simulations including the ocean
        * kdis -> k-distribution
        * cld -> cloud data
        * IPRT -> some data from IPRT (International working group on Polarized Radiative Transfer)

    Examples
    --------
    >>> from pathlib import Path
    >>> from smartg.auxdata import download
    >>> savepath = Path("/dir/where/to/save/data")
    >>> download(savepath, data_type="all")
    """

    list_kind = ["all"] + list(AUXDATA_DICT.keys())

    if data_type not in list_kind:
        raise ValueError("Invalid value for 'kind'. Must be one of: " + ", ".join(list_kind))
    
    Path(savepath).mkdir(parents=True, exist_ok=True)

    if data_type == "all": names = list(AUXDATA_DICT.keys())
    else                 : names = [data_type]

    for name in names:
        try:
            print(f"Trying to download {name} auxiliary data in {savepath}...\n")

            if name == "reptran":
                out = join(savepath, name + ".tar.gz")
                safe_download(AUXDATA_DICT[name], out)
                extract_tar(out, savepath)
                Path(out).unlink(missing_ok=True)

            else:
                out = join(savepath, name + ".zip")
                safe_download(AUXDATA_DICT[name] + "/" + name + ".zip", out)
                extract_zip(out, savepath)
                Path(out).unlink(missing_ok=True)

            print(f"{name} auxiliary data downloaded and extracted successfully. ✅\n")

        except Exception as e1:
            print(f"Error during download and/or extraction: {e1}. ❌\n")

            if name == "reptran":
                print("Another url is available for reptran, trying again...\n")
                try:
                    out = join(savepath, name + ".zip")
                    safe_download(REPTRAN_URL_HYG + "/" + name + ".zip", out)
                    extract_zip(out, savepath)
                    Path(out).unlink(missing_ok=True)
                    print(f"{name} auxiliary data downloaded and extracted successfully. ✅\n")
                except Exception as e2:
                    print(f"Error during download and/or extraction: {e2}. ❌\n")