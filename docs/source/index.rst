.. cpnet documentation master file, created by
   sphinx-quickstart on Tue Jan 26 16:30:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

A Python package for detecting core-periphery structure in networks
===================================================================

This package contains algorithms for detecting core-periphery structure in networks. 
All algorithms are implemented in Python, with speed accelerations by numba, and can be used with a small coding effort.


Installation
************

Before installing this package, make sure that you have a **Python with version 3.6 or above**.

There are two ways to install this package, *conda* or *pip*. conda is recommended if you have a conda environment. Otherwise, use pip.  

For conda::

    bash
    conda install -c conda-forge -c skojaku cpnet 

For pip::

    pip install cpnet

This package is under active development. If you have issues and feature requests, please raise them through `Github <https://github.com/skojaku/core-periphery-detection>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Examples
**********
- `Example 1 (Detection of core-periphery structure) <https://github.com/skojaku/core-periphery-detection/blob/master/notebooks/example1.ipynb>`_
- `Example 2 (Statistical test) <https://github.com/skojaku/core-periphery-detection/blob/master/notebooks/example2.ipynb>`_
- `Example 3 (Case study: Pilitical blog network <https://github.com/skojaku/core-periphery-detection/blob/master/notebooks/example3.ipynb>`_
- `Example 4 (Case study: Airport network) <https://github.com/skojaku/core-periphery-detection/blob/master/notebooks/example4.ipynb>`_


Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
