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
* 3D atmospheres (`opt3d=True`), validated against MYSTIC on the IPRT phase B benchmark
* Spherical atmosphere (`pp=False`), validated on the IPRT phase 3 benchmark
* 3D objects and concentrated solar flux geometries (heliostat fields, solar towers)
* Flat, rough or Lambertian surfaces, and 1D ocean profiles
* Spectral integration with the k-distribution and REPTRAN parameterizations
* Rotational (Ring effect) and vibrational Raman scattering
* Forward and backward modes, local estimate, and photon-history tracking
* Results returned as an `xarray.Dataset`, with built-in visualization helpers


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
>>> # Example to download all the data. See the docstrings for more details.
>>> from smartg.auxdata import download
>>> from pathlib import Path
>>> download(Path('dir/path/where/to/save/data/'), data_type='all')
>>> # Only some datasets, in the SMARTG_DIR_AUXDATA directory
>>> download(data_type=['aer', 'cld'])
```

The datasets already on disk are skipped. Each download is recorded in
a `.smartg_auxdata.json` manifest inside the auxdata directory, which
allows checking and refreshing the data when it changes on the server:

```python
>>> from smartg.auxdata import check_update, update
>>> check_update()  # queries the remote versions, downloads nothing
Auxiliary data in /dir/path/where/to/save/data
dataset  status      downloaded (UTC)  remote (UTC)      source / note
aer      up to date  2026-09-15 14:03  2026-04-27 14:37  hygeos
acs      outdated    2026-09-15 14:03  2026-09-16 09:12  hygeos
kdis     missing     -                 2024-05-02 14:13  hygeos
...
update() would download: acs, kdis
>>> update()  # downloads only the missing, outdated or unrecorded datasets
>>> update(data_type='reptran', force=True)  # download again whatever the status
```

The manifest also records the checksum of every downloaded file, so
`check_update()` lists the files modified or deleted locally (status
`modified`). The remote data is the reference: `restore()` shows these
files and, after confirmation, replaces them by the remote copies
without downloading the whole dataset again. Extra files are kept.

```python
>>> from smartg.auxdata import restore
>>> restore()  # lists the modified files and asks before replacing them
>>> restore(data_type='aer', yes=True)  # no question asked
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

Once the auxiliary data are in place, a first simulation takes a few lines:

```python
from smartg.smartg import Smartg
from smartg.atmosphere import Atm1D
from smartg.view import smartg_view

# 1e8 photons at 500 nm, tropical atmosphere, sun at a 30° zenith angle
ds = Smartg().run(wavelength=500., th_deg=30., n_photons=1e8,
                  atmosphere=Atm1D('afglt'))

# ds is an xarray.Dataset holding 'I_up (TOA)', 'Q_up (TOA)',
# 'U_up (TOA)', 'V_up (TOA)', ...
smartg_view(ds)  # polar view of the reflectance and of the polarization
```

The first call compiles the CUDA kernel; subsequent runs reuse it.

## 6. Examples

Sample notebooks are provided in the [notebooks](smartg/notebooks) folder, and [jupyter notebook](http://jupyter.org) has nice possibilities for interactive development and visualization, in particular if you are using a remote cuda computer. Good entry points are:

* [`demo_notebook.py`](smartg/notebooks/demo_notebook.py) — general usage: atmosphere, ocean, surface, outputs and visualization
* [`demo_notebook_objects.py`](smartg/notebooks/demo_notebook_objects.py) — simulations involving 3D objects, e.g., solar power towers
* [`demo_notebook_photons_histories.py`](smartg/notebooks/demo_notebook_photons_histories.py) — tracking the photon paths (needs the extra dependencies, e.g. `pixi shell --environment extra`)
* [`validation_smartg_iprt_phase_a.py`](smartg/notebooks/validation_smartg_iprt_phase_a.py) — the 1D cases of the IPRT phase A, compared with MYSTIC
* [`validation_smartg_iprt_phase_b_c2.py`](smartg/notebooks/validation_smartg_iprt_phase_b_c2.py) — the cubic cloud (C2) of the IPRT phase B, in 3D mode, compared with MYSTIC
* [`validation_smartg_iprt_phase3.py`](smartg/notebooks/validation_smartg_iprt_phase3.py) — the spherical cases D1 to E6 of the IPRT phase 3

### 6.1 Notebooks as percent scripts

The notebooks are stored as [jupytext](https://jupytext.readthedocs.io) percent scripts: plain Python files in which `# %%` starts a code cell and `# %% [markdown]` a markdown cell. They hold neither outputs nor editor metadata, so their diffs and merges are readable. Jupyter magics are commented (`# %%time`) so that the scripts stay valid Python, and are restored when a script is opened as a notebook.

jupytext is installed with the other dependencies. Its configuration, in the `[tool.jupytext]` section of `pyproject.toml`, pairs each script with a `.ipynb` notebook of the same name, which keeps the outputs locally and is ignored by git.

* **JupyterLab / Jupyter Notebook** (`pixi run jupyter lab`): right-click a script, then *Open With → Jupytext Notebook*. Saving updates both the script and its paired notebook. Run `pixi run jupytext-config set-default-viewer python` once to open the scripts as notebooks with a double click.
* **VS Code**: the `# %%` cells of a script run as is in the Interactive Window. To use the notebook editor, open the paired `.ipynb` (created by the task below) and either install the [Jupytext Sync](https://marketplace.visualstudio.com/items?itemName=caenrigen.jupytext-sync) extension, which updates the script on save, or run the task before committing.
* **Synchronization**: after a `git pull`, or after editing one file of a pair outside Jupyter, run

  ```bash
  pixi run sync-notebooks
  ```

  The cells are taken from the most recently modified file of each pair, and the outputs from the notebook. Synchronize before pulling as well, so that edits made in a notebook are not overwritten by a newer script.
* **New notebook**: save it as `smartg/notebooks/<name>.ipynb` from Jupyter, which creates its script, or run `pixi run jupytext --sync smartg/notebooks/<name>.ipynb`. Only the `.py` script is committed.

## 7. Tests

To check that SMART-G is running correctly, run the following command at the root of the project:

```bash
pytest smartg/tests/test_cuda.py smartg/tests/test_profile.py \
       smartg/tests/test_atm3d.py smartg/tests/test_smartg.py -s -v
```

A full testing is recommended in dev:

```bash
pytest smartg/tests/ -s -v
```

With Pixi, both are available as tasks:

```bash
pixi run test-basic  # the four files above
pixi run test-all    # the whole test suite
```

`test_smartg_jax.py` needs the jax dependencies of the `extra`
environment. Without them it skips itself with a message rather than
failing, so run it with `pixi run -e extra pytest
smartg/tests/test_smartg_jax.py`.

To avoid repeating some pytest arguments, a `pytest.ini` file can be created at the root of the project. The following is an example of the contents of such a file:

```
[pytest]
addopts= --html=test_report.html --self-contained-html -s -v -rs
```

The arguments `--html=test_report.html --self-contained-html` generate an html report containing the results of the tests (sometimes with more details e.g. plots), named `test_report.html`, and `-rs` lists the reasons why tests have been skipped.

### 7.1 The IPRT tests

Four files compare SMART-G with the [IPRT](https://www.meteo.physik.uni-muenchen.de/~iprt/doku.php?id=start) (International Polarized Radiative Transfer) model intercomparison, one per phase of the benchmark. The durations below were measured on a Ryzen 9 5950X with a GeForce RTX 5070 Ti.

**Phase A** — `test_quick_iprt_phase_a.py` runs the 1D cases A1 (a Rayleigh layer), A2 (a Rayleigh layer over a Lambertian surface) and A5 (a water cloud, in the principal plane and in the almucantar), and compares them with MYSTIC. Four tests, about 2 min 30, no slow tier.

**Phase B** — `test_iprt_phase_b_c2.py` (cubic cloud) and `test_iprt_phase_b_c3.py` (cumulus cloud with aerosols) check the 3D atmosphere mode (`opt3d=True`) against the MYSTIC reference. Reproducing the benchmark photon counts takes hours, so each of their tests exists in two tiers: a fast one, run by default, and a slow one selected with `-m slow`.

```bash
pytest smartg/tests/test_iprt_phase_b_c3.py           # fast, ~4 min
pytest -m slow smartg/tests/test_iprt_phase_b_c3.py   # slow, ~38 min
```

Both files together take 7 min in the fast tier and 1 h 23 in the slow one; the CPU counts as much as the GPU for C3, whose atmosphere is built by a single threaded loop over the cloudy cells.

The fast tier detects a 5 % error on the cloud optical properties, the slow one 1 %: run it before a release, or after a change to the 3D kernel, to the phase matrices or to the truncation.

**Phase 3** — `test_iprt_phase3.py` checks the spherical geometry (`pp=False`) on the one-layer cases D1 to D6 and the vertically inhomogeneous cases E1 to E5, against saved SMART-G results computed with 1e8 photons per viewing direction. It has the same two tiers: the fast one uses 1e6 photons per direction and takes about 5 min for the file, the slow one reproduces the 1e8 of the references and takes about 3 h 50 for the 11 cases. E6, the camera at 300 000 km, is not covered yet.

```bash
pytest smartg/tests/test_iprt_phase3.py           # fast, ~5 min
pytest -m slow smartg/tests/test_iprt_phase3.py   # slow, ~3 h 50
```

Because a GPU run is not reproducible bit for bit, the phase 3 comparison is statistical: the tolerances on the mean bias and on the fraction of directions beyond three combined standard deviations were measured per tier rather than taken from a normal distribution.

## 8. Naming conventions

Identifiers follow PEP 8: `snake_case` for modules, functions, variables and
parameters, `PascalCase` for classes and `UPPER_SNAKE_CASE` for constants.
Where several spellings of one concept coexisted, `smartg/smartg.py` is the
reference and the rest of the package follows it.

| concept | use | not |
|---|---|---|
| number of photons | `n_photons` | `nphotons`, `NBPHOTONS` |
| photons per kernel loop | `n_loop` | `NBLOOP` |
| wavelength | `wavelength` | `wav`, `wvl`, `lam` |
| number of scattering angles | `n_theta` | `NBTHETA` |
| depolarization factor | `depo` | `depol` |
| file path or file name | `fname` | `filename`, `file_name` |
| output folder | `output_dir` | `dir_output` |

File formats keep their own spelling: a variable read from or written to a
netCDF, an OPAC or an IPRT file stays `wav`, `wavelen` or `wvl` when that is
what the format calls it. The phase matrix wavelength axis is
`wavelength_phase`.

The symbols of the equations a module implements stay as the paper writes them
in docstrings and comments, but the code around them is lower case: `p_tot`,
not `P_tot`.

Line length is 79 columns for code and 72 for docstrings and comments. Both
are checked by ruff, together with PEP 8, the naming rules, the numpy
docstring convention and the annotations, configured in the
`[tool.ruff]` and `[tool.pyright]` sections of `pyproject.toml`.
`smartg/obselete_files`, whose unused Python 2 modules no longer parse,
is excluded from both.

ruff and pyright are not project dependencies, so install them
separately; the configuration was written for ruff 0.16 and pyright
1.1.411, and a different ruff may select a different set of rules by
default. Lint the tracked files with:

```bash
ruff check $(git ls-files '*.py')
pyright
```

Passing the tracked files explicitly keeps untracked scratch modules out
of the report.

## 9. Hardware tested

GeForce GTX 1070, GeForce TITAN V, GeForce RTX 2080 Ti, GeForce RTX 3070, GeForce RTX 3090, GeForce RTX 4090, GeForce RTX 5070 Ti, A100, RTX PRO 6000 Blackwell (Workstation Edition)

The use of GPUs before 10xx series (Pascal) is deprecated as of SMART-G 1.0.0

## 10. Licensing information

This software is available under the SMART-G license v1.0, available in the [LICENSE.TXT](LICENSE.TXT) file. It can be used for free for non-commercial purposes; for commercial use, please [contact HYGEOS](https://hygeos.com/en/contact/).

## 11. Referencing

When acknowledging the use of SMART-G for scientific papers, reports etc please cite the following reference(s):

* Ramon, D., Steinmetz, F., Jolivet, D., Compiègne, M., & Frouin, R. (2019). Modeling polarized radiative
  transfer in the ocean-atmosphere system with the GPU-accelerated SMART-G Monte Carlo code.
  Journal of Quantitative Spectroscopy and Radiative Transfer, 222, 89-107. https://doi.org/10.1016/j.jqsrt.2018.10.017

* Moulana, M., Cornet, C., Elias, T., Ramon, D., Caliot, C., & Compiègne, M. (2024). Concentrated solar flux
  modeling in solar power  towers with a 3D objects-atmosphere hybrid system to consider atmospheric and environmental
  gains. Solar Energy, 277, 112675. https://doi.org/10.1016/j.solener.2024.112675

## 12. Getting help

* Changes between versions: [CHANGELOG.md](CHANGELOG.md)
* Bug reports and feature requests: [github issues](https://github.com/hygeos/smartg/issues)
* Anything else: [contact HYGEOS](https://hygeos.com/en/contact/)
