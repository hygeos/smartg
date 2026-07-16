from importlib.util import find_spec
from pathlib import Path

import papermill as pm
import pytest

SKIP = find_spec("jax") is None

ROOTPATH = Path(__file__).resolve().parent.parent.parent


def test_demo_notebook():
    """
    Execute the demo notebook
    """
    print("\nTesting demo_notebook.ipynb...")
    nb_path = ROOTPATH / "smartg" / "notebooks" / "demo_notebook.ipynb"
    nb_output_path = (
        ROOTPATH / "smartg" / "tests" / "logs" / "demo_notebook_log.ipynb"
    )
    pm.execute_notebook(nb_path, nb_output_path, cwd=ROOTPATH)


def test_demo_notebook_objects():
    """
    Execute the demo notebook objects
    """
    print("\nTesting demo_notebook_objects.ipynb...")
    nb_path = ROOTPATH / "smartg" / "notebooks" / "demo_notebook_objects.ipynb"
    nb_output_path = (
        ROOTPATH
        / "smartg"
        / "tests"
        / "logs"
        / "demo_notebook_objects_log.ipynb"
    )
    pm.execute_notebook(nb_path, nb_output_path, cwd=ROOTPATH)


@pytest.mark.skipif(
    SKIP, reason="cannot test this since the jax package is not installed."
)
def test_demo_notebook_photons_histories():
    """
    Execute the photon histories demo notebook
    """
    print("\nTesting demo_notebook_photons_histories.ipynb...")
    nb_path = (
        ROOTPATH
        / "smartg"
        / "notebooks"
        / "demo_notebook_photons_histories.ipynb"
    )
    nb_output_path = (
        ROOTPATH
        / "smartg"
        / "tests"
        / "logs"
        / "demo_notebook_photons_histories_log.ipynb"
    )
    pm.execute_notebook(nb_path, nb_output_path, cwd=ROOTPATH)
