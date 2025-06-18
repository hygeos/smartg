#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
from os.path import join, dirname, realpath
from pathlib import Path


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

# some data (mystic res and opt_prop) are taken from: https://www.meteo.physik.uni-muenchen.de/~iprt/doku.php?id=intercomparisons:intercomparisons
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
    >>> import smartg.auxdata.download as download
    >>> download('/dir/where/to/save/data/', data_type="all")
    """

    list_kind = ["all"] + list(AUXDATA_DICT.keys())

    if data_type not in list_kind:
        raise ValueError("Invalid value for 'kind'. Must be one of: " + ", ".join(list_kind))
    
    Path(savepath).mkdir(parents=True, exist_ok=True)

    if data_type == "all": names = list(AUXDATA_DICT.keys())
    else            : names = [data_type]

    for name in names:
        if name == "reptran":
            command1 = ['wget', '-c', savepath, AUXDATA_DICT[name], '-O', join(savepath,name+'.tar.gz')]
            command2 = ['tar', '-xvzf', join(savepath,name+'.tar.gz'), '--strip-components=2', '-C', savepath]
            command3 = ['rm', '-f', join(savepath,name+'.tar.gz')]
        else:
            command1 = ['wget', '-c', '-P', savepath, AUXDATA_DICT[name]+'/'+name+'.zip']
            command2 = ['unzip', '-o', join(savepath+'/'+name+'.zip'), '-d', savepath]
            command3 = ['rm', '-f', join(savepath+'/'+name+'.zip')]
        try:
            print(f"Trying to download {name} auxiliary data in {savepath}...\n")
            if name == "reptran":
                res = subprocess.run(command1, check=False)
                if res.returncode != 0 and res.returncode != 1:
                    raise Exception(f"{res.stderr.decode('utf-8')}")
            else:
                subprocess.run(command1, check=True)
            subprocess.run(command2, check=True)
            subprocess.run(command3, check=True)
            print(f"{name} auxiliary data downloaded and extracted successfully. ✅\n")
        except Exception as e1:
            print(f"Error during download and/or extraction: {e1}. ❌\n")
            if name == "reptran":
                print("Another url is available for reptran, trying again...\n")
                try:
                    subprocess.run(['wget', '-c', '-P', savepath, REPTRAN_URL_HYG+"/"+name+".zip"], check=True)
                    subprocess.run(['unzip', '-o', savepath+"/"+name+".zip", "-d", savepath], check=True)
                    subprocess.run(['rm', '-f', savepath+"/"+name+".zip"], check=True)
                    print(f"{name} auxiliary data downloaded and extracted successfully. ✅\n")
                except subprocess.CalledProcessError as e2:
                    print(f"Error during download and/or extraction: {e2}. ❌\n")
