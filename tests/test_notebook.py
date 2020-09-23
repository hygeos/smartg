import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_demo_notebook():
    """
    Execute the demo notebook
    """
    with open('notebooks/demo_notebook.ipynb') as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb)