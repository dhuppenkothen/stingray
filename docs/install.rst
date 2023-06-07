Dependencies
============
A **minimal installation** of Stingray requires the following dependencies:

+ astropy>=4.0
+ numpy>=1.17.0
+ scipy>=1.1.0
+ matplotlib>=3.0,!=3.4.0

In **typical** uses, requiring input/output, caching of results, and faster processing, we **recommend the following dependencies**:

+ numba (**highly** recommended)
+ tbb (needed by numba)
+ tqdm (for progress bars, always useful)
+ pyfftw (for the fastest FFT in the West)
+ h5py (for input/output)
+ pyyaml (for input/output)
+ emcee (for MCMC analysis, e.g. for PSD fitting)
+ corner (for the plotting of MCMC results)
+ statsmodels (for some statistical analysis)

For **pulsar searches and timing**, we recommend installing

+ pint-pulsar

Some of the dependencies are available in ``conda``, the others via ``pip``.
To install all required and recommended dependencies in a recent installation, you should be good running the following command:

    $ pip install astropy scipy matplotlib numpy h5py tqdm numba pint-pulsar emcee corner statsmodels pyfftw tbb

For development work, you will need the following extra libraries:

+ pytest
+ pytest-astropy
+ tox
+ jinja2<=3.0.0
+ docutils
+ sphinx-astropy
+ nbsphinx>=0.8.3,!=0.8.8
+ pandoc
+ ipython
+ jupyter
+ notebook
+ towncrier<22.12.0
+ black

Which can be installed with the following command:

    $ pip install pytest pytest-astropy jinja2<=3.0.0 docutils sphinx-astropy nbsphinx pandoc ipython jupyter notebook towncrier<22.12.0 tox black

Downloading and Installing Stingray
===================================

There are currently two ways to install Stingray:

* via ``conda``
* via ``pip``
* from source

Installing via ``conda``
------------------------

If you manage your Python installation and packages
via Anaconda or miniconda, you can install ``stingray``
via the ``conda-forge`` build: ::

    $ conda install -c conda-forge stingray

That should be all you need to do! Just remember to :ref:`run the tests <testsuite>` before
you use it!

Installing via ``pip``
----------------------

``pip``-installing Stingray is easy! Just do::

    $ pip install stingray

And you should be done! Just remember to :ref:`run the tests <testsuite>` before you use it!

Installing from source (bleeding edge version)
----------------------------------------------

For those of you wanting to install the bleeding-edge development version from
source (it *will* have bugs; you've been warned!), first clone
`our repository <https://github.com/StingraySoftware/stingray>`_ on GitHub: ::

    $ git clone --recursive https://github.com/StingraySoftware/stingray.git

Now ``cd`` into the newly created ``stingray`` directory.
Finally, install ``stingray`` itself: ::

    $ pip install -e "."

Installing development environment (for new contributors)
---------------------------------------------------------

For those of you wanting to contribute to the project, install the bleeding-edge development version from
source. First fork
`our repository <https://github.com/StingraySoftware/stingray>`_ on GitHub and clone the forked repository using: ::

    $ git clone --recursive https://github.com/<your github username>/stingray.git

Now, navigate to this folder and run
the following command to add an upstream remote that's linked to Stingray's main repository.
(This will be necessary when submitting PRs later.): ::

    $ cd stingray
    $ git remote add upstream https://github.com/StingraySoftware/stingray.git

Now, install the necessary dependencies::

    $ pip install astropy scipy matplotlib numpy pytest pytest-astropy h5py tqdm

Finally, install ``stingray`` itself::

    $ pip install -e "."

.. _testsuite:

Test Suite
----------

Please be sure to run the test suite before you use the package, and please report anything
you think might be bugs on our GitHub `Issues page <https://github.com/StingraySoftware/stingray/issues>`_.

Stingray uses `py.test <https://pytest.org>`_ and `tox
<https://tox.readthedocs.io>`_ for testing. To run the tests, try::

   $ tox -e test

You may need to install tox first::

   $ pip install tox

To run a specific test file (e.g., test_io.py), try::

    $ cd stingray
    $ py.test tests/test_io.py

If you have installed Stingray via pip or conda, the source directory might
not be easily accessible. Once installed, you can also run the tests using::

   $ python -c 'import stingray; stingray.test()'

or from within a python interpreter:

.. doctest-skip::

   >>> import stingray
   >>> stingray.test()

Documentation
-------------

The documentation including tutorials is hosted `here <https://docs.stingray.science/>`_.
The documentation uses `sphinx <https://www.sphinx-doc.org/en/stable/>`_ to build and requires the extensions `sphinx-astropy <https://pypi.org/project/sphinx-astropy/>`_ and `nbsphinx <https://pypi.org/project/nbsphinx/>`_.

You can build the API reference yourself by going into the ``docs`` folder within the ``stingray`` root
directory and running the ``Makefile``: ::

    $ cd stingray/docs
    $ make html

If that doesn't work on your system, you can invoke ``sphinx-build`` itself from the stingray source directory: ::

    $ cd stingray
    $ sphinx-build docs docs/_build

The documentation should be located in ``stingray/docs/_build``. Try opening ``./docs/_build/index.rst`` from
the stingray source directory.

