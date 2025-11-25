import papermill as pm

from pathlib import Path
ROOTPATH =  Path(__file__).resolve().parent.parent.parent


def test_demo_notebook():
    """
    Execute the demo notebook
    """
    print("\nTesting demo_notebook.ipynb...")
    nb_path = ROOTPATH / 'smartg' / 'notebooks' / 'demo_notebook.ipynb'
    nb_output_path = ROOTPATH / 'smartg' / 'tests' / 'logs' / 'demo_notebook_log.ipynb'
    pm.execute_notebook(nb_path, nb_output_path, cwd=ROOTPATH)

def test_demo_notebook_objects():
    """
    Execute the demo notebook objects
    """
    print("\nTesting demo_notebook_objects.ipynb...")
    nb_path = ROOTPATH / 'smartg' / 'notebooks' / 'demo_notebook_objects.ipynb'
    nb_output_path = ROOTPATH / 'smartg' / 'tests' / 'logs' / 'demo_notebook_objects_log.ipynb'
    pm.execute_notebook(nb_path, nb_output_path, cwd=ROOTPATH)