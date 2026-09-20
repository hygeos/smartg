"""Execute the demo notebooks through papermill.

They are tracked as jupytext percent scripts, so each one is read
as a notebook first, and the executed copies land in tests/logs.
"""
from importlib.util import find_spec
from pathlib import Path

import jupytext
import papermill as pm
import pytest

SKIP = find_spec("jax") is None

ROOTPATH = Path(__file__).resolve().parent.parent.parent


def execute_notebook(nb_path: Path, nb_output_path: Path) -> None:
    """Execute a notebook tracked as a jupytext percent script.

    The scripts hold no kernelspec, hence the explicit kernel name.
    """
    pm.execute_notebook(
        jupytext.read(nb_path),
        nb_output_path,
        cwd=ROOTPATH,
        kernel_name="python3",
    )


def test_demo_notebook() -> None:
    """Execute the demo notebook."""
    print("\nTesting demo_notebook.py...")
    nb_path = ROOTPATH / "smartg" / "notebooks" / "demo_notebook.py"
    nb_output_path = (
        ROOTPATH / "smartg" / "tests" / "logs" / "demo_notebook_log.ipynb"
    )
    execute_notebook(nb_path, nb_output_path)


def test_demo_notebook_objects() -> None:
    """Execute the 3D objects demo notebook."""
    print("\nTesting demo_notebook_objects.py...")
    nb_path = ROOTPATH / "smartg" / "notebooks" / "demo_notebook_objects.py"
    nb_output_path = (
        ROOTPATH
        / "smartg"
        / "tests"
        / "logs"
        / "demo_notebook_objects_log.ipynb"
    )
    execute_notebook(nb_path, nb_output_path)


@pytest.mark.skipif(
    SKIP, reason="cannot test this since the jax package is not installed."
)
def test_demo_notebook_photons_histories() -> None:
    """Execute the photon histories demo notebook."""
    print("\nTesting demo_notebook_photons_histories.py...")
    nb_path = (
        ROOTPATH
        / "smartg"
        / "notebooks"
        / "demo_notebook_photons_histories.py"
    )
    nb_output_path = (
        ROOTPATH
        / "smartg"
        / "tests"
        / "logs"
        / "demo_notebook_photons_histories_log.ipynb"
    )
    execute_notebook(nb_path, nb_output_path)
