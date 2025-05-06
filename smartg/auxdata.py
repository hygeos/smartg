import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Literal
from core import log, env

from core.download import download_url, uncompress
from core.fileutils import mdir

# Helper class extracted from internal tool 'hyp'
# TODO remove class, replace by proper implementation in core module
# cf joackim.orciere@hygeos.com
class prompt:
        
    def _concat_mess(*args):
        message = ""
        for arg in args:
            message += str(arg)
        return message + str(log.rgb.default)
        
    def msg(*args, **kwargs):
        
        msg = prompt._concat_mess(*args, log.rgb.default)
        return input(msg)

    yes_no_def_true = f"[Y/n]"
    yes_no_def_false = f"[y/N]"

    class questions:
        
        def proceed(default=True):
            
            a = None
            while a is None:
                i = None
                if default == True:
                    i = prompt.msg(log.rgb.orange("Proceed "), prompt.yes_no_def_true, " ? ")
                elif default == False:
                    i = prompt.msg(log.rgb.orange("Proceed "), prompt.yes_no_def_false, " ? ")
                else:
                    raise RuntimeError()
                
                a = prompt.parse.yes_or_no(i, default=default)
                if a is None:
                    log.warning("Invalid input")
            
            if a == True:
                return True
            else:
                log.info("> Aborting ")
                exit(-1)
                return False
    
    class parse:
            
        def yes_or_no(s: str, default:Literal["y", "n"]|None=None):
            """
            returns True if yes
            returns False if no
            returns None if input is not valid
            """
            
            if s == "": return default
            
            s = s.lower()
            
            yes = ["y", "yes"]
            no = ["n", "no"]
            
            if s in yes: return True
            if s in no: return False
            return None

def download_auxdata_from_nextcloud(output_dir: Path | str):
    """
    Function for downloading data from Nextcloud contained in the data/eoread directory

    Args:
        product_name (str): Name of the product with the extension
        output_dir (Path | str): Directory where to store downloaded data
        input_dir (Path | str, optional): Sub repository in which the product are stored. Defaults to ''.

    Returns:
        Path: Output path of the downloaded data
    """
    
    base_sharelink = "https://docs.hygeos.com/s/fkpWLxnyw59T3Cz/download?path=%2F&files="
    
    output_dir = mdir(output_dir)
    archive = "auxdata_smartg.tar.gz"
    
    url = f"{base_sharelink}{archive}"
    # download_url(url, output_dir, wget_opts="", verbose=verbose, if_exists=if_exists)
    
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        cmd = f'wget "{url}" -O {tmpdir/archive}'
        # print(cmd)
        res = subprocess.run(cmd, shell=True)
        
        # uncompressing and moving content
        cmd = f"tar -xzvf {tmpdir/archive} -C {output_dir}"
        res = subprocess.run(cmd, shell=True)
        
    
    # Uncompress downloaded file 
    # if product_name.split('.')[-1] in ['zip','gz']:
    # return uncompress(output_dir/"download?.zip", output_dir)
        
    return output_dir
        


def download(args):
    
    # download path management 
    if args.download_dir is not None:
        path = Path(args.download_dir)
    else:
        path = env.getvar("SMARTG_DIR_DATA", default="None")
        
        if path == "None":
            default = Path("auxdata")
            log.info(f"SMARTG_DIR_DATA is not set, defaulting to ./{default}")
            ok = prompt.questions.proceed()
            if not ok: exit(-1)
            
            path = default
            
    path=Path(path) # convert str to Path
    if not path.is_dir():
        log.error(f"Target directory {path} for SMARTG_DIR_DATA does not exit", e=FileNotFoundError)
        
    log.info(f"Downloading SMART-G auxdata to {path.resolve()}")
        
    download_auxdata_from_nextcloud(path)