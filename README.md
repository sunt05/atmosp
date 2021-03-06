atmosp
======

- [atmosp](#atmosp)
    - [Note](#note)
  - [An atmospheric sciences utility library](#an-atmospheric-sciences-utility-library)
    - [Features](#features)
    - [Dependencies](#dependencies)
    - [Installation](#installation)
    - [Development version](#development-version)
    - [Examples](#examples)
    - [License](#license)


### Note


This is a fork version of [atmos-python/atmos]([atmos_gh](https://github.com/atmos-python/atmos)). Modifications
are made here to enable its installation via `pip install atmos`


An atmospheric sciences utility library
---------------------------------------

**atmos** is a library of Python programming utilities for the
atmospheric sciences. It is in ongoing development. If you have an idea
for a feature or have found a bug, please post it on the [GitHub issue
tracker](https://github.com/mcgibbon/atmos/issues).

Information on how to use the module can be found predominantly by using
the built-in help() function in Python. Many docstrings are
automatically generated by the module and so information may appear to
be missing in the source code. There is also [HTML
documentation](http://www.pythonhosted.org/atmos) available.

This module is currently alpha. The API of components at the base module
level should stay backwards-compatible, but sub-modules are subject to
change. In particular, features in the util module are likely to be
changed or removed entirely.


### Features

-   defined constants used in atmospheric science
-   functions for common atmospheric science equations
-   a simple calculate() interface function for accessing equations
-   no need to remember equation function names or argument order
-   fast calculation of quantities using numexpr
-   skew-T plots integrated into matplotlib

### Dependencies

This module is tested to work with Python versions 2.6, 2.7, 3.3, and
3.4 on Unix. Support is given for all platforms. If there are bugs on
your particular version of Python, please submit it to the [GitHub issue
tracker](https://github.com/mcgibbon/atmos/issues).

Package dependencies:

-   numpy
-   numexpr
-   six
-   nose

### Installation

To install this module, download and run the following:

``` {.sourceCode .bash}
$ python setup.py install
```

If you would like to edit and develop the code, you can instead install
in develop mode

``` {.sourceCode .bash}
$ python setup.py develop
```

If you are running Anaconda, you can install using conda:

``` {.sourceCode .bash}
$ conda install -c mcgibbon atmos
```

You can also install using pip:

``` {.sourceCode .bash}
$ pip install atmos
```

### Development version

The most recent development version can be found in the [GitHub develop
branch](https://github.com/mcgibbon/atmos/tree/develop).

### Examples

Calculating pressure from virtual temperature and air density:

``` {.sourceCode .python}
>>> import atmos
>>> atmos.calculate('p', Tv=273., rho=1.27)
    99519.638400000011
```

### License

This module is available under an MIT license. Please see `LICENSE.txt`.
