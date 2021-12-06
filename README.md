# SMART-G
_Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU_

SMART-G is a radiative transfer code using a Monte-Carlo technique to simulate the propagation of the polarized light in the atmosphere and/or ocean, and using GPU acceleration.

## Installation
SMART-G is written in python and cuda, and uses the [pycuda](http://mathema.tician.de/software/pycuda/) library. Python3 is recommended.

We recommend to use the [anaconda python distribution](https://www.anaconda.com/download/). You can create a dedicated [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) for SMART-G, or use your current environment.

### Using conda

The smartg dependencies can be installed on anaconda with the following command:

```
  # create a new environment and activate it (optional but recommended)
  conda create -n smartg -c conda-forge mamba
  conda activate smartg

  # install all SMART-G dependencies
  mamba env update -f environment.yml
```

### Using poetry

The dependencies can also be installed with [python-poetry](https://python-poetry.org/), using the following command:

```
poetry install --no-root
```

The dependencies are installed in the current environment from the definitions included in `pyproject.toml`


## Auxiliary data
Atmospheric profiles created in atmosphere.py are based on data included in the [libRadtran](http://www.libradtran.org/) library.
This auxiliary data can be automatically installed using the config.py script (`python smartg/config.py`)

## Examples
Examples are provided in the [sample notebooks](notebooks).

[jupyter notebook](http://jupyter.org) has nice possibilities for interactive development and visualization, in particular if you are using a remote cuda computer. Sample notebooks are provided in the folder [notebooks](notebooks).

## Tests
Run the command `pytest tests` or the provided notebooks.

## Hardware tested
GeForce GTX 660 Ti (unused for a while), GeForce GeForce GTX 970, GeForce GTX 1070, GeForce TITAN V, Quadro P2000, GeForce RTX 2080 Ti

## Documentation

Use the provided makefile:

* `make help` will provide sphinx help

* `make html` will build the doc in html

* `make serve` will run a small python http server to view the doc

The sphinx extensions [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) is used so that [google](https://google.github.io/styleguide/pyguide.html) or [numpy](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) style docstring can be used.