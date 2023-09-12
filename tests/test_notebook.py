import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# import os
from os.path import join, dirname, realpath
ROOTPATH = dirname(dirname(realpath(__file__)))

def test_demo_notebook():
    """
    Execute the demo notebook
    """
    with open('notebooks/demo_notebook.ipynb') as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': join(ROOTPATH, 'notebooks/')}})