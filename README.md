<p align="center">
<img src="https://hygeos.com/wp-content/uploads/2023/02/SMART-G-300x300.png" width="200">
</p>

SMART-G
=======

[![image](https://img.shields.io/pypi/v/smartg.svg)](https://pypi.python.org/pypi/smartg)
[![image](https://img.shields.io/github/v/tag/hygeos/smartg?label=github&color=blue)](https://github.com/hygeos/smartg)
[![image](https://pepy.tech/badge/smartg)](https://pepy.tech/project/smartg)
[![image](https://img.shields.io/badge/DOI-10.1038%2Fs41586--020--2649--2-blue)](
https://doi.org/10.1016/j.jqsrt.2018.10.017)

SMART-G (Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU) is a radiative transfer code using a Monte-Carlo technique to simulate the propagation of the polarized light in the atmosphere and/or ocean, and using GPU acceleration.

Didier Ramon  
Mustapha Moulana  
François Steinmetz  
Dominique Jolivet  
Mathieu Compiègne  
[HYGEOS](https://hygeos.com/en/)

----------------------------------------------------------------------  


## 1. Installation

### 1.1 Dependencies
### 1.1.1 Pixi (recommended)
[Pixi](https://pixi.sh/) is recommended for its fast dependency resolution and robust environment management. Unlike Conda, which only considers Conda packages during conflict resolution, Pixi consider both Conda and pip package versions when solving dependencies.

To create and activate the environment, use the following command:

```bash
  pixi shell
```

To consider all extra dependencies (e.g. jax), use instead:

```bash
  pixi shell --environment extra
```



### 1.1.2 Anaconda/Miniconda (alternative)

With Anaconda/Miniconda, use the following command:

```bash
  conda create -n smartg-env -f environment.yml
  conda activate smartg-env
```

For a full installation (extra dependencies), replace `environment.yml` by `environment-extra.yml`.

### 1.2 Auxiliary data

The auxiliary data can be downloaded as follow:

```python
>>> # Example to download all the data. See the docstring for more details.
>>> from smartg.auxdata import download
>>> download('dir/path/where/to/save/data/', data_type='all')
```

The environment variable `SMARTG_DIR_AUXDATA` have to be defined.

For example, in the `.bashrc` / `.zshrc` file the following can be added:

```
export SMARTG_AUXDATA_DIR="dir/path/where/to/save/data/"
```

or (not recommended) in a `.env` file in the SMART-G root directory:

```
SMARTG_AUXDATA_DIR=dir/path/where/to/save/data/
```


## 2. Examples

Examples are provided in the [sample notebooks](smartg/notebooks).

[jupyter notebook](http://jupyter.org) has nice possibilities for interactive development and visualization, in particular if you are using a remote cuda computer. Sample notebooks are provided in the folder [notebooks](smartg/notebooks).

## 3. Tests

To check that SMART-G is running correctly, run the following command at the root of the project:

```bash
pytest smartg/tests/test_cuda.py smartg/tests/test_profile.py smartg/tests/test_smartg.py -s -v
```

A full testing is recommended in dev:

```bash
pytest smartg/tests/ -s -v
```

To avoid repeating some pytest arguments, a `pytest.ini` file can be created (in the root directory). The following is an example of the contents of such a file:
```
[pytest]
addopts= --html=test_reportv1.html --self-contained-html -s -v
```
The arguments "--html=test_reportv1.html --self-contained-html" are used to generate an html report containing the results of the tests (sometime with more details e.g. plots), named "test_reportv1.html".

## 4. Hardware tested

GeForce GTX 1070, GeForce TITAN V, GeForce RTX 2080 Ti, Geforce RTX 3070, Geforce RTX 3090, Geforce RTX 4090, A100, Geforce RTX 5070 ti

The use of GPUs before 10xx series (Pascal) is depracated as of SMART-G 1.0.0

## 5. Licensing information

This software is available under the SMART-G license v1.0, available in the LICENSE.TXT file.

## 6. Referencing

When acknowledging the use of SMART-G for scientific papers, reports etc please cite the following reference:

* Ramon, D., Steinmetz, F., Jolivet, D., Compiègne, M., & Frouin, R. (2019). Modeling polarized radiative
  transfer in the ocean-atmosphere system with the GPU-accelerated SMART-G Monte Carlo code.
  Journal of Quantitative Spectroscopy and Radiative Transfer, 222, 89-107. https://doi.org/10.1016/j.jqsrt.2018.10.017
