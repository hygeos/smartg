#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import tarfile
from typing import Optional
from smartg.typing import PathType


# auxdata source: HYGEOS
AER_URL = "https://docs.hygeos.com/s/8PnKXFXQbmYyTte/download"
ACS_URL = "https://docs.hygeos.com/s/HwotAHPstdCCKcJ/download"
ATM_URL = "https://docs.hygeos.com/s/z6MRf9g66WmWeBA/download"
STP_URL = "https://docs.hygeos.com/s/NW42DNPtKw3NNW7/download"
VALID_URL = "https://docs.hygeos.com/s/6EPBqwebn94NYPq/download"
WATER_URL = "https://docs.hygeos.com/s/3NKP5tMsHKnNRpt/download"
KDIS_URL = "https://docs.hygeos.com/s/CHTFFgHe6to39CR/download"
CLOUD_URL = "https://docs.hygeos.com/s/agDWDy998j64SHf/download"

# some data (mystic res and opt_prop) are taken from the IPRT site:
# https://www.meteo.physik.uni-muenchen.de/~iprt/doku.php?id=
# intercomparisons:intercomparisons
IPRT_URL = "https://docs.hygeos.com/s/i4QaxtpjSfjwtNk/download"

# reptran source: http://www.libradtran.org
# Use the libradtran URL to obtain the latest reptran look-up
# tables when possible.
REPTRAN_URL = (
    "http://www.meteo.physik.uni-muenchen.de/~libradtran/lib/exe/"
    + "fetch.php?media=download:reptran_2017_all.tar.gz"
)
# The HYGEOS-hosted URL is an alternative if the upstream link fails.
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


def safe_download(url: str, outfile: PathType) -> None:

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = (
                int(count * block_size * 100 / total_size)
                if total_size > 0
                else 0
            )
            print(f"\rDownloading {outfile}: {percent}%", end="")
        else:
            downloaded = count * block_size
            print(f"\rDownloaded {downloaded / 1024 / 1024:.1f} MB...", end="")

    print(f"Downloading {url} → {outfile}")
    urlretrieve(url, outfile, reporthook)
    print("\nDownload complete.")


def extract_zip(zfile: PathType, dest: PathType) -> None:
    with zipfile.ZipFile(zfile, "r") as z:
        print(f"Extracting ZIP {zfile} → {dest}")
        for name in z.namelist():
            print("  extracting:", name)
        z.extractall(dest)


def extract_tar(
    tfile: PathType, dest: PathType, target_folder: Optional[str] = None
) -> None:
    with tarfile.open(tfile, "r:gz") as tar:
        if target_folder is None:
            print(f"Extracting TAR {tfile} → {dest}")
            for member in tar.getmembers():
                print("  extracting:", member.name)
            tar.extractall(dest)
        else:
            print(
                f"Extracting TAR {tfile} → {dest} (only folder"
                + f" '{target_folder}')"
            )
            for member in tar.getmembers():
                if target_folder in member.name:
                    parts = member.name.split("/")
                    try:
                        idx = parts.index(target_folder)
                        # rewrite to keep only the path starting at
                        # target_folder
                        member.name = "/".join(parts[idx:])
                        print("  extracting:", member.name)
                        tar.extract(member, dest)
                    except ValueError:
                        continue


def download(dname: PathType, data_type: str = "all") -> None:
    """Download the SMART-G auxiliary data.

    Parameters
    ----------
    dname : str or path-like
    Directory path where the data will be saved.
    data_type : str, optional
            Type of data to download. Can be one of:
            "all", "aer", "acs", "atm", "STP", "valid",
            "water", "kdis", "cld", "IPRT". Default is
            "all". Definitions:

            * all -> all the available data
            * aer -> aerosols data
            * acs -> absorption cross section coefficients
                data
            * atm -> atmosphere profiles
            * STP -> STP (Solar Power Tower) files with
                heliostat positions
            * valid -> validation files
            * water -> water files needed for some
                simulations, including the ocean
            * kdis -> k-distribution
            * cld -> cloud data
            * IPRT -> some data from IPRT (International
                working group on Polarized Radiative Transfer)

    Examples
    --------
    >>> from pathlib import Path
    >>> from smartg.auxdata import download
    >>> dname = Path("/dir/where/to/save/data")
    >>> download(dname, data_type="all")
    """

    list_kind = ["all"] + list(AUXDATA_DICT.keys())

    if data_type not in list_kind:
        raise ValueError(
            "Invalid value for 'kind'. Must be one of: " + ", ".join(list_kind)
        )

    dname = Path(dname)
    dname.mkdir(parents=True, exist_ok=True)

    if data_type == "all":
        names = list(AUXDATA_DICT.keys())
    else:
        names = [data_type]

    for name in names:
        try:
            print(
                f"Trying to download {name} auxiliary data in {dname}...\n"
            )

            if name == "reptran":
                out = dname / f"{name}.tar.gz"
                safe_download(AUXDATA_DICT[name], out)
                extract_tar(out, dname, target_folder="reptran")
                Path(out).unlink(missing_ok=True)

            else:
                out = dname / f"{name}.zip"
                safe_download(AUXDATA_DICT[name] + "/" + name + ".zip", out)
                extract_zip(out, dname)
                Path(out).unlink(missing_ok=True)

            print(
                f"{name} auxiliary data downloaded and extracted "
                + "successfully. ✅\n"
            )

        except Exception as e1:
            print(f"Error during download and/or extraction: {e1}. ❌\n")

            if name == "reptran":
                print(
                    "Another url is available for reptran, trying again...\n"
                )
                try:
                    out = dname / f"{name}.zip"
                    safe_download(f"{REPTRAN_URL_HYG}/{name}.zip", out)
                    extract_zip(out, dname)
                    Path(out).unlink(missing_ok=True)
                    print(
                        f"{name} auxiliary data downloaded and extracted "
                        + "successfully. ✅\n"
                    )
                except Exception as e2:
                    print(
                        f"Error during download and/or extraction: {e2}. ❌\n"
                    )
