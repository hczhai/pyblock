
<img src="https://raw.githubusercontent.com/sanshar/Block/master/README_Examples/block_logo.jpg" width="60px" height="60px" />

[![Documentation Status](https://readthedocs.org/projects/pyblock-dmrg/badge/?version=latest)](https://pyblock-dmrg.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/hczhai/pyblock.svg?branch=master)](https://travis-ci.org/hczhai/pyblock)
[![codecov](https://codecov.io/gh/hczhai/pyblock/branch/master/graph/badge.svg)](https://codecov.io/gh/hczhai/pyblock)

-------------------------------------------------

`pyblock` is a python interface build upon `BLOCK 1.5.3` (https://github.com/sanshar/StackBlock).

Installation
------------

`cmake` (version >= 3.0) can be used to compile C++ part of the code, as follows:

    mkdir build
    cd build
    cmake .. -DBUILD_LIB=ON <extra options>
    make

There are several options available for `cmake ..`.

When `-DBUILD_LIB=ON` is given, the python extension library
will be built. Otherwise, the `block 1.5.3` executable will be built.

If MKL Math library is installed in the system, add `-DUSE_MKL=ON` to use MKL library.

If boost (version < 1.56) is used, add `-DBOOST_OLD=ON`. For new boost version no extra options are required.

The root path and `./build` path are required to be added to `PYTHONPATH` so that the python can locate `block` and `pyblock` modules. A cleaner way to run tests is

    cd tests/hubbard-1d
    PYTHONPATH=../..:../../build python3 dmrg.py

This will only temporarily change the `PYTHONPATH`. One can also install the pyblock by running (in package root directory)

    pip3 install .

BLOCK
-----

`BLOCK` implements the density matrix renormalization group (DMRG) algorithm for quantum chemistry.

How to cite `Block`
-------------------

`Block` is distributed under the GNU GPL license which is reproduced in the file LICENSE.
In addition, `Block` contains a full copy of the Newmat C++ matrix library by Robert Davies.

We would appreciate if you cite the following papers in publications resulting from the
use of `Block`:

* G. K.-L. Chan and M. Head-Gordon, J. Chem. Phys. 116, 4462 (2002),
* G. K.-L. Chan, J. Chem. Phys. 120, 3172 (2004),
* D. Ghosh, J. Hachmann, T. Yanai, and G. K.-L. Chan, J. Chem. Phys., 128, 144117 (2008),
* S. Sharma and G. K-.L. Chan, J. Chem. Phys. 136, 124121 (2012).

In addition, a useful list of DMRG references relevant to quantum chemistry can be found
in the article above by Sharma and Chan.

Documentation
-------------

The online documentation of `block 1.5.3` is available at [https://sanshar.github.io/Block](https://sanshar.github.io/Block).

