# SMART-G
_Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU_

SMART-G is a radiative transfer code using a Monte-Carlo technique to simulate the propagation of the polarized light in the atmosphere and/or ocean, and using GPU acceleration.

# Dependencies
SMART-G is written in python and cuda. Its dependencies are:
* python 2 or 3
* [pycuda](http://mathema.tician.de/software/pycuda/)
* numpy
* scipy

Optional dependencies:
* [ipython notebook](http://ipython.org/notebook.html) has nice possibilities for interactive development and visualization, in particular if you are using a remote cuda computer. Sample notebooks are provided in the folder [notebooks](notebooks).
* [nose](http://nose.readthedocs.org/) for testing

# Auxiliary data
Atmospheric profiles created in atmosphere.py are based on data included in the [libRadtran](http://www.libradtran.org/) library.
This auxiliary data can be automatically installed using the config.py script (`python smartg/config.py`)


# Examples
Examples are provided in the script `examples_tests.py` and in the [sample notebooks](notebooks).

# Tests
Run the command `nosetests` or the provided notebooks.
