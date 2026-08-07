<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="https://raw.githubusercontent.com/hygeos/smartg/master/images/smartg-horizontal-c-transparent-on-dark.svg">
    <img alt="SMART-G" width="360"
         src="https://raw.githubusercontent.com/hygeos/smartg/master/images/smartg-horizontal-c-transparent-on-light.svg">
  </picture>
</p>

<p align="center">
  <em><strong>S</strong>peed-up <strong>M</strong>onte Carlo <strong>A</strong>dvanced <strong>R</strong>adiative <strong>T</strong>ransfer Code using <strong>G</strong>PU</em>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/smartg"><img alt="PyPI version" src="https://img.shields.io/pypi/v/smartg.svg"></a>
  <a href="https://anaconda.org/conda-forge/smartg"><img alt="conda-forge version" src="https://img.shields.io/conda/vn/conda-forge/smartg.svg"></a>
  <a href="https://pepy.tech/project/smartg"><img alt="Downloads" src="https://static.pepy.tech/badge/smartg"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img alt="Requires a CUDA GPU" src="https://img.shields.io/badge/GPU-CUDA-76B900?logo=nvidia&logoColor=white"></a>
  <a href="https://doi.org/10.1016/j.jqsrt.2018.10.017"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1016%2Fj.jqsrt.2018.10.017-blue"></a>
</p>

SMART-G is a radiative transfer code using a Monte-Carlo technique to simulate the propagation of the polarized light in the atmosphere and/or ocean, and using GPU acceleration.

Didier Ramon  
Mustapha Moulana  
François Steinmetz  
Dominique Jolivet  
Mathieu Compiègne  
[HYGEOS](https://hygeos.com/en/)

----------------------------------------------------------------------  


## 1. Features

* Polarized (I, Q, U, V) Monte-Carlo radiative transfer, accelerated on NVIDIA GPUs
* Coupled ocean-atmosphere system, or atmosphere only / ocean only
* 1D atmospheric profiles (AFGL standard atmospheres or user-provided), aerosols (OPAC or user-defined) and clouds
* 3D atmospheres (`opt3D=True`), validated against MYSTIC on the IPRT phase B benchmark
* 3D objects and concentrated solar flux geometries (heliostat fields, solar towers)
* Flat, rough or Lambertian surfaces, and 1D ocean profiles
* Spectral integration with the k-distribution and REPTRAN parameterizations
* Rotational (Ring effect) and vibrational Raman scattering
* Forward and backward modes, local estimate, and photon-history tracking
* Results convertible to an `xarray.Dataset`, with built-in visualization helpers


## 2. Installation

### 2.1 PyPI

To install SMART-G from PyPI:

```bash
pip install smartg
```

To include extra dependencies use instead:

```bash
pip install smartg[extra]
```


### 2.2 conda-forge

Use the command:

```bash
conda install -c conda-forge smartg
```

If you need extra dependencies (jax with cuda) we recommend the installation with pip instead.



### 2.3 github clone (for development)
<details>
  <summary>Click here</summary>

  First clone the repository:

  ```bash
  git clone https://github.com/hygeos/smartg.git
  ```

  You can now choose between Pixi or Conda for your development environment.

  #### 2.3.1 Pixi (recommended)
  [Pixi](https://pixi.sh/) is recommended for its fast dependency resolution and robust environment management. Unlike Conda, which only considers Conda packages during conflict resolution, Pixi considers both Conda and pip package versions when solving dependencies.

  To create and activate the environment, use the following command:

  ```bash
  pixi shell
  ```

  To consider all extra dependencies (e.g. jax), use instead:

  ```bash
  pixi shell --environment extra
  ```

  #### 2.3.2 Anaconda/Miniconda (alternative)

  With Anaconda/Miniconda, use the following command:

  ```bash
  conda env create -n smartg-env -f environment.yml
  conda activate smartg-env
  ```

  For a full installation (extra dependencies), replace `environment.yml` by `environment-extra.yml`.

</details>

## 3. Nvidia driver and CUDA
An installation guide is available in the nvidia website: [installation-guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/introduction.html).

You can also install CUDA using conda:

```bash
conda install nvidia::cuda
```

## 4. Auxiliary data

The auxiliary data can be downloaded as follow:

```python
>>> # Example to download all the data. See the docstring for more details.
>>> from smartg.auxdata import download
>>> from pathlib import Path
>>> download(Path('dir/path/where/to/save/data/'), data_type='all')
```

The environment variable `SMARTG_DIR_AUXDATA` must be defined.

For example, in the `.bashrc` / `.zshrc` file the following can be added:

```bash
export SMARTG_DIR_AUXDATA="dir/path/where/to/save/data/"
```

or (not recommended) in a `.env` file in the SMART-G root directory:

```
SMARTG_DIR_AUXDATA=dir/path/where/to/save/data/
```


## 5. Quick start

Once the auxiliary data are in place, a first simulation takes three lines:

```python
from smartg.smartg import Smartg
from smartg.atmosphere import Atm1D
from smartg.smartg_view import smartg_view

# 1e8 photons at 500 nm, tropical atmosphere, sun at a 30° zenith angle
res = Smartg().run(wl=500., THVDEG=30., NBPHOTONS=1e8, atm=Atm1D('afglt'))

ds = res.to_xarray()  # 'I_up (TOA)', 'Q_up (TOA)', 'U_up (TOA)', 'V_up (TOA)', ...
smartg_view(ds)       # polar view of the reflectance and of the polarization
```

The first call compiles the CUDA kernel; subsequent runs reuse it.

## 6. Examples

Sample notebooks are provided in the [notebooks](smartg/notebooks) folder, and [jupyter notebook](http://jupyter.org) has nice possibilities for interactive development and visualization, in particular if you are using a remote cuda computer. Good entry points are:

* [`demo_notebook.ipynb`](smartg/notebooks/demo_notebook.ipynb) — general usage: atmosphere, ocean, surface, outputs and visualization
* [`demo_notebook_objects.ipynb`](smartg/notebooks/demo_notebook_objects.ipynb) — 3D objects and concentrated solar flux
* [`demo_notebook_photons_histories.ipynb`](smartg/notebooks/demo_notebook_photons_histories.ipynb) — tracking the photon paths

## 7. Tests

To check that SMART-G is running correctly, run the following command at the root of the project:

```bash
pytest smartg/tests/test_cuda.py smartg/tests/test_profile.py smartg/tests/test_smartg.py -s -v
```

A full testing is recommended in dev:

```bash
pytest smartg/tests/ -s -v
```

With Pixi, both are available as tasks:

```bash
pixi run test-basic  # the three files above
pixi run test-all    # the whole test suite
```

To avoid repeating some pytest arguments, a `pytest.ini` file can be created at the root of the project. The following is an example of the contents of such a file:

```
[pytest]
addopts= --html=test_report.html --self-contained-html -s -v -rs
```

The arguments `--html=test_report.html --self-contained-html` generate an html report containing the results of the tests (sometimes with more details e.g. plots), named `test_report.html`, and `-rs` lists the reasons why tests have been skipped.

### 7.1 The IPRT phase B tests

`test_iprt_phase_b_c2.py` (cubic cloud) and `test_iprt_phase_b_c3.py` (cumulus cloud with aerosols) check the 3D atmosphere mode (`opt3D=True`) against the MYSTIC reference of the IPRT phase B benchmark. Reproducing the benchmark photon counts takes hours, so each of their tests exists in two tiers: a fast one, run by default, and a slow one selected with `-m slow`.

```bash
pytest smartg/tests/test_iprt_phase_b_c3.py           # fast, ~4 min
pytest -m slow smartg/tests/test_iprt_phase_b_c3.py   # slow, ~38 min
```

Both files together take 7 min in the fast tier and 1 h 23 in the slow one. These durations were measured on a Ryzen 9 5950X with a GeForce RTX 5070 Ti; the CPU counts as much as the GPU for C3, whose atmosphere is built by a single threaded loop over the cloudy cells.

The fast tier detects a 5 % error on the cloud optical properties, the slow one 1 %: run it before a release, or after a change to the 3D kernel, to the phase matrices or to the truncation.

## 8. Hardware tested

GeForce GTX 1070, GeForce TITAN V, GeForce RTX 2080 Ti, GeForce RTX 3070, GeForce RTX 3090, GeForce RTX 4090, GeForce RTX 5070 Ti, A100, RTX PRO 6000 Blackwell (Workstation Edition)

The use of GPUs before 10xx series (Pascal) is deprecated as of SMART-G 1.0.0

## 9. Licensing information

This software is available under the SMART-G license v1.0, available in the [LICENSE.TXT](LICENSE.TXT) file. It can be used for free for non-commercial purposes; for commercial use, please [contact HYGEOS](https://hygeos.com/en/contact/).

## 10. Referencing

When acknowledging the use of SMART-G for scientific papers, reports etc please cite the following reference(s):

* Ramon, D., Steinmetz, F., Jolivet, D., Compiègne, M., & Frouin, R. (2019). Modeling polarized radiative
  transfer in the ocean-atmosphere system with the GPU-accelerated SMART-G Monte Carlo code.
  Journal of Quantitative Spectroscopy and Radiative Transfer, 222, 89-107. https://doi.org/10.1016/j.jqsrt.2018.10.017

* Moulana, M., Cornet, C., Elias, T., Ramon, D., Caliot, C., & Compiègne, M. (2024). Concentrated solar flux
  modeling in solar power  towers with a 3D objects-atmosphere hybrid system to consider atmospheric and environmental
  gains. Solar Energy, 277, 112675. https://doi.org/10.1016/j.solener.2024.112675

## 11. Getting help

* Changes between versions: [CHANGELOG.md](CHANGELOG.md)
* Bug reports and feature requests: [github issues](https://github.com/hygeos/smartg/issues)
* Anything else: [contact HYGEOS](https://hygeos.com/en/contact/)
