# Contributing to stingray

> All great things have small beginnings.

Hello there! We love and appreciate every small contribution you can make to improve stingray!
We are proudly open source and believe our(yes! yours as well) work will help enhance the quality of research around the world.  We want to make contributing to stingray as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

A successful project is not just built by amazing programmers but by an amalgamation of coders, testers,  reviewers and even documentation writers. There are a few guidelines that we need all contributors to follow so that we can have a chance of keeping on top of things.

## Contribution Guidelines

Contributions from everyone, experienced and inexperienced, are welcome! If you don't know where to start, look at the [Open Issues](https://github.com/StingraySoftware/stingray/issues) and/or get involved in our [Slack channel](http://slack-invite.timelabtechnologies.com/) . This code is written in Python 3.5+. Tests run at each commit during Pull Requests, so it is easy to single out points in the code that break this compatibility.

- **Branches:**
    - Don't use your master branch(forked) for anything. **Consider deleting your master branch.
    - Make a new branch, called a feature branch, for each separable set of changes: “one task, one branch”.
    - Start that new feature branch from the most current development version of stingray.
    - Name of branch should be the purpose of change eg. *bugfix-for-issue20* or *refactor-lightcurve-code.*
    - Never merge changes from stingray/master into your feature branch. If changes in the development version require changes to our code you can rebase, but only if asked.
- **Commits:**
    - Make frequent commits.
    - One commit per logical change in the code-base.
    - Add commit message.
- **Naming Conventions:**
    - Change name of the remote origin(*yourusername/stingray*) to your *github-username.*
    - Name the remote that is the primary stingray repository( *StingraySoftware/stingray*) as  stingray.

### Contribution Workflow

These, conceptually, are the steps you will follow in contributing to stingray. These steps keep work well organized, with readable history. This in turn makes it easier for project maintainers (that might be you) to see what you’ve done, and why you did it:

1. Regularly fetch latest stingray development version `stingray/master` from GitHub.
2. Make a new feature branch. **Recommended:** Use virtual environments to work on branch.
3. Editing Workflow:
    1. One commit per logical changes.
    2. Run tests to make sure that changes don't break existing code.
    3. Code should have appropriate docstring.
    4. Update appropriate documentation if necessary and test it on sphinx.
    5. Write a test for code changes.
    6. If  modifications require more than one commit, break changes into smaller commits.
    7. Push the code on your remote(forked) repository.
4. All code changes should be submitted via PR's(i.e. fork, branch, work on stuff, just submit pull request).
Code Reviews are super-useful: another contributor can review the code, which means both the contributor and reviewer will be up to date with how everything fits together, and can get better by reading each other's code! :)
5. Take feedback and make changes/revise the PR as asked.
6. Don't merge using web interface if your branch falls behind master. Fetch and rebase.

## Coding Guidelines

### Compatibility and Dependencies

- **Compatibility:** All code must be compatible with **Python 3.5** **or later**. Usage of six, __future__, and 2to3 is not longer acceptable.
- **Dependency Management:**
    - The core package and affiliated packages should be importable with no dependencies other than the [Python Standard Library](https://docs.python.org/3/library/index.html), [astropy](https://docs.astropy.org/en/stable/), [numpy](https://numpy.org/doc/stable/)>=1.11.0, [scipy](https://www.scipy.org/docs.html)>=0.18.0, [matplotlib](https://matplotlib.org/contents.html)>=1.3.
    - Additional dependencies - such as SciPy, Matplotlib, or other third-party packages - are allowed for sub-modules or in function calls, but they must be noted in the package documentation and should only affect the relevant component. In functions and methods, the optional dependency should use a normal `import` statement, which will raise an `ImportError` if the dependency is not available.

### Coding Style and Conventions

- **Style Guide:**
    - Follow the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/). Follow the existing coding style within the sub-package and avoid changes that are purely stylistic.
    - Indentation should be **ONLY** with **four spaces** no mixing of tabs-and-spaces.
    - Maximum line length should be **79** characters unless doing so makes the code unreadable, ugly.
    - Functions and methods should be lower-case only, and separated by a '_'  in case of multiple words eg. `my_new_method`.
    - Use verbose variable names(readability > economy). Only loop iteration variables are allowed to be a single letter.
    - Classes start with an upper-case letter and use CamelCase eg. `MyNewClass`.
    - Line comments should start with a single #.
- **Formatting Style:**  The new Python 3 formatting style should be used, i.e. f-strings `f"{variable_name}"`  or  `"{0}".format(variable_name}`should be used instead of `"%s" % (variable_name)`.
- **Linter/Style Guide Checker:** Our testing infrastructure currently enforces a subset of the PEP8 style guide. You can check locally whether your changes have followed these by running [flake8](https://pypi.org/project/flake8/) with the following command:

    `flake8 astropy --count --select=E101,W191,W291,W292,W293,W391,E111,E112,E113,E30,E502,E722,E901,E902,E999,F822,F823`

- **Code Formatters:** The use of automatic code formatters (e.g., Black) is strongly discouraged.
- **Imports:**
    - Absolute imports are to be used in general. The exception to this is relative imports of the form `from . import modulename`, this convention makes it clearer what code is from the current sub-module as opposed to from another. It is best to use when referring to files within the same sub-module.
    - The import `numpy as np`, `import scipy as sp`, `import matplotlib as mpl`, and `import matplotlib.pyplot as plt` naming conventions should be used wherever relevant. `from packagename import *` should never be used, except as a tool to flatten the namespace of a module.
- **Variable access in Classes:** Classes should either use direct variable access, or Python’s property mechanism for setting object instance variables. `get_value/set_value` style methods should be used only when getting and setting the values requires a computationally-expensive operation.
- **super( ) function:** Classes should use the built-in `super()` function when making calls to methods in their super-class(es) unless there are specific reasons not to. `super()` should be used consistently in all sub-classes since it does not work otherwise.
- **Multiple Inheritance:** Multiple inheritance should be avoided in general without good reason.
- **__init__.py:** The `__init__.py` files for modules should not contain any significant implementation code.  `__init__.py` can contain docstrings and code for organizing the module layout, however if a module is small enough that it fits in one file, it should simply be a single file, rather than a directory with an  `__init__.py` file.

### Standard output, warnings, and errors

- **Print Statement:** Used only for outputs in methods and scenarios explicitly requested by the user
- **Errors and Exceptions:**  Always use the `raise` with built-in or custom exception classes. The nondescript `Exception` class should be avoided as much as possible, in favor of more specific exceptions (*IOError, ValueError* etc.).
- **Warnings:**  Always use the `warnings.warn(message, warning_class)`for warnings. These get redirected to `log.warning()` by default, but one can still use the standard warning-catching mechanism and custom warning classes.
- **Debugging and  Informational messages:**  Always use `log.info(message)` and `log.debug(message)`. The logging system uses the built-in Python logging module.

### Data and Configuration

- **Storing Data:**
    - Packages can include data in a directory named *data* inside a subpackage source directory as long as it is less than about 100 kB.
    - If the data exceeds this size, it should be hosted outside the source code repository, either at a third-party location on the internet.

### Documentation and Testing

- **Docstrings:**
    - Docstrings must be provided for all public classes, methods, and functions.
    - Docstrings should follow the [numpydoc style](https://numpydoc.readthedocs.io/en/latest/format.html) and reStructured Text format.
    - Write usage examples in the docstrings of all classes and functions whenever possible. These examples should be short and simple to reproduce. Users should be able to copy them verbatim and run them.
- **Unit tests:**  Provided for as many public methods and functions as possible, and should adhere to the standards set in the Testing Guidelines.
- **Building Documentation:**
    - Use sphinx to build the documentation.
    - All extra documentation should go into a /docs sub-directory under the main stingray directory.

## Testing Guidelines

The testing framework used by stingray is the pytest framework, it is accessed through the python [setup.py](http://setup.py/) test command. To run the tests, you will need to make sure you have the pytest package (version 3.1 or later) installed.

- All tests should be py.test compliant: [http://pytest.org/latest/](http://pytest.org/latest/).
- Keep all tests in a /tests subdirectory under the main stingray directory.
- Write one test script per module in the package.
- Extra examples can go into an /examples folder in the main stingray directory, scripts that gather various data analysis tasks into longer procedures into a /scripts folder in the same location.

## Community Guidelines

- Be kind and respectful to others.
- Give and accept constructive criticism.
<!--- Something from CODE_OF_CONDUCT.md?? --->
