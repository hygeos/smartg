  SMART-G
  =======

  SMART-G (Speed-up Monte Carlo Advanced Radiative Transfer Code using GPU) is a radiative transfer code using a Monte-Carlo technique to simulate the propagation of the polarized light in the atmosphere and/or ocean, and using GPU acceleration.

  Didier Ramon  
  Mustapha Moulana  
  François Steinmetz  
  Dominique Jolivet  
  Mathieu Compiègne  
  [HYGEOS](www.hygeos.com)

----------------------------------------------------------------------  


## 1. Installation

### 1.1 Dependencies

The smartg dependencies can be installed on anaconda with the following command:

```
  # create a new environment and activate it (optional but recommended)
  conda create -n smartg -c conda-forge mamba
  conda activate smartg

  # install all SMART-G dependencies
  mamba env update -f environment.yml
```

### 1.2 Auxiliary data

The auxiliary data can be downloaded with the following command:
```
$ make auxdata_all
```

## 2. Examples

Examples are provided in the [sample notebooks](notebooks).

[jupyter notebook](http://jupyter.org) has nice possibilities for interactive development and visualization, in particular if you are using a remote cuda computer. Sample notebooks are provided in the folder [notebooks](notebooks).

## 3. Tests

Example of pytest.ini file:
```
[pytest]
addopts= --html=test_reportv1.html --self-contained-html -s -v
```
Run the command `pytest tests/test_cuda.py tests/test_profile.py tests/test_smartg.py` to check that SMART-G is correctly running.

## 4. Hardware tested

GeForce GTX 1070, GeForce TITAN V, GeForce RTX 2080 Ti, Geforce RTX 3070, Geforce RTX 3090, Geforce RTX 4090, A100

The use of GPUs before 10xx series (Pascal) is depracated as of SMART-G 1.0.0

## 5. Licencing information

This software is available under the SMART-G licence v1.0, available in the LICENCE.TXT file.

## 6. Referencing

When acknowledging the use of SMART-G for scientific papers, reports etc please cite the following reference:

* Ramon, D., Steinmetz, F., Jolivet, D., Compiègne, M., & Frouin, R. (2019). Modeling polarized radiative
  transfer in the ocean-atmosphere system with the GPU-accelerated SMART-G Monte Carlo code.
  Journal of Quantitative Spectroscopy and Radiative Transfer, 222, 89-107. https://doi.org/10.1016/j.jqsrt.2018.10.017
