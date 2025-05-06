import argparse
from textwrap import dedent

from smartg import auxdata

def entry(args=None):
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='SMART-G cli, facilitate auxiliary data download')

    subs = parser.add_subparsers(dest="command", required=True)
    
    help_msg = dedent("""
        Downloads the auxiliary data required to run SMART-G.
        requires the environment variable SMARTG_DIR_DATA to be set.
        Defaults to ./auxdata
        """)
    
    cmd_name = "download_auxdata"
    
    # auxdata download command
    cmd = subs.add_parser(help=help_msg, name=cmd_name)
    cmd.add_argument("download_dir", action="store", help=str("Override auxdata download dir path"), nargs="?", default=None)
    
    
    
    args = parser.parse_args()
    
    if args.command == cmd_name:
        auxdata.download(args)