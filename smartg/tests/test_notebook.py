import papermill as pm

# import os
from os.path import join, dirname, realpath
ROOTPATH = dirname(dirname(dirname(realpath(__file__))))

def test_demo_notebook():
    """
    Execute the demo notebook
    """
    print("\nTesting demo_notebook.ipynb...")
    nb_path = join(ROOTPATH, "notebooks/demo_notebook.ipynb")
    nb_output_path = join(ROOTPATH, "smartg/tests/logs/demo_notebook_log.ipynb")
    pm.execute_notebook(str(nb_path), str(nb_output_path), cwd=ROOTPATH)

def test_demo_notebook_objects():
    """
    Execute the demo notebook objects
    """
    print("\nTesting demo_notebook_objects.ipynb...")
    nb_path = join(ROOTPATH, "notebooks/demo_notebook_objects.ipynb")
    nb_output_path = join(ROOTPATH, "smartg/tests/logs/demo_notebook_objects_log.ipynb")
    pm.execute_notebook(str(nb_path), str(nb_output_path), cwd=ROOTPATH)