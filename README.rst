X-Ray Timing Made Easy
=======================
+------------------+-------------------------+----------------------------+
| Master           | |Build Status Master|   | |Coverage Status Master|   |
+------------------+-------------------------+----------------------------+

We are writing a general-purpose timing package for X-ray time series.

Contents
--------

The contents will be:

- make a light curve from event data
- make periodograms in Leahy and rms normalization
- average periodograms
- cross spectra and lags (time vs energy, time vs frequency)
- maximum likelihood fitting of periodograms/parametric models
- bispectra (?)
- cross correlation functions, coherence
- spectral-timing functionality
- Bayesian QPO searches
- power colours
- rms spectra

Documentation
-------------

Is generated using `Sphinx`_. Try::

   $ sphinx-build docs docs/_build

Then open ``./docs/_build/index.html`` in the browser of your choice.

.. _Sphinx: http://sphinx-doc.org

Test suite
----------

Try::

   $ nosetests

Copyright
---------

All content © 2015 the authors. The code is distributed under the MIT license.

Pull requests are welcome! If you are interested in the further development of
this project, please `get in touch via the issues
<https://github.com/dhuppenkothen/stingray/issues>`_!

.. |Build Status Master| image:: https://travis-ci.org/StingraySoftware/stingray.svg?branch=master
    :target: https://travis-ci.org/StingraySoftware/stingray
.. |Coverage Status Master| image:: https://coveralls.io/repos/github/StingraySoftware/stingray/badge.svg?branch=master
    :target: https://coveralls.io/github/StingraySoftware/stingray?branch=master
