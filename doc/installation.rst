Installation
============

SMART-G is written in python and cuda, and uses the
[pycuda](http://mathema.tician.de/software/pycuda/) library. Python3 is
recommended.

Using the Anaconda distribution
-------------------------------

We recommend to use the `anaconda python distribution <https://www.anaconda.com/download/>`_ . You can optionnally create
a dedicated
`environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_
(:code:`conda create -n smartg python=3`) and then activate it. The smartg
dependencies can be installed on anaconda with the following command:

.. code::

    conda install -c conda-forge lukepfister::pycuda numpy scipy nose notebook matplotlib netcdf4 progressbar2 pyhdf

Dependencies
------------

Atmospheric profiles created in atmosphere.py are based on data included in
the `libRadtran <http://www.libradtran.org/>`_ library. This auxiliary data
can be automatically installed using the config.py script (:code:`python
smartg/config.py`)

