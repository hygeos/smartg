Usage
=====

.. currentmodule:: smartg.smartg

Smart-G is run using a pythonic interface. First instantiate a :class:`Smartg` class::

    from smartg.smartg import Smartg
    from smartg.atmosphere import AtmAFGL

    sg = Smartg()

During instanciation, the cuda code is being compiled with the compilation options passed as arguments.

Then use the :func:`Smartg.run` method to actually run a simulation::

    m = sg.run(500., atm=AtmAFGL('afglms'))

    m['Rtoa']   # contains the simulated TOA reflectance


Examples are provided in the sample notebooks (see directory `notebooks`).

`jupyter notebooks <http://jupyter.org>`_ have nice possibilities for
interactive development and visualization, in particular if you are using a
remote cuda computer. Sample notebooks are provided in the folder
`notebooks`.



