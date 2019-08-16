# SMART-G
_Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU_

SMART-G is a radiative transfer code using a Monte-Carlo technique to simulate the propagation of the polarized light in the atmosphere and/or ocean, and using GPU acceleration.

# Installation
SMART-G is written in python and cuda, and uses the [pycuda](http://mathema.tician.de/software/pycuda/) library. Python3 is recommended.

We recommend to use the [anaconda python distribution](https://www.anaconda.com/download/). You can optionnally create a dedicated [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) (`conda create -n smartg python=3`) and then activate it. The smartg dependencies can be installed on anaconda with the following command:

```
conda install -c conda-forge lukepfister::pycuda numpy scipy nose notebook matplotlib netcdf4 progressbar2 pyhdf
```

# Auxiliary data
Atmospheric profiles created in atmosphere.py are based on data included in the [libRadtran](http://www.libradtran.org/) library.
This auxiliary data can be automatically installed using the config.py script (`python smartg/config.py`)


# Examples
Examples are provided in the script `examples_tests.py` and in the [sample notebooks](notebooks).

[jupyter notebook](http://jupyter.org) has nice possibilities for interactive development and visualization, in particular if you are using a remote cuda computer. Sample notebooks are provided in the folder [notebooks](notebooks).

# Tests
Run the command `pytest` or the provided notebooks.
