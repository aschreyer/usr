# USR: Ultrafast Shape Recognition

This package contains a Python implementation of the _Ultrafast Shape Recognition 
algorithm_ that was published by Ballester and Richards in 2007.

Ballester, P.J. and Richards, W.G. (2007), Ultrafast shape recognition to search 
compound databases for similar molecular shapes. _J. Comput. Chem._, 28: 1711-1723. 
[doi:10.1002/jcc.20681](https://doi.org/10.1002/jcc.20681)

## Installation

The **USR** package requires [RDKit](rdkit.org), which can only be installed directly 
through the conda Python package manager.

**USR** can simply be installed with `pip` from PyPI using `pip install rd-usr`.

which will install the latest stable release. For other versions, the version can be 
explicitly specified using the  normal pip syntax, e.g. `pip install rd-usr==1.0`. 

### Developer installation

For developers who directly work on the source code, it is recommended to install 
**USR** in development mode. For that, please clone the repository first with

    $ git clone git@github.org:aschreyer/usr.git

and then install the repo in `pip` development mode:

    $ pip install  .[tests]

This step will also install all the dependencies required for running the test suite.

### Testing

**USR** comes with a comprehensive test suite that can be started by calling `$ py.test` 
in the source directory.
