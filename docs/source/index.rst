.. cpnet documentation master file, created by
   sphinx-quickstart on Tue Jan 26 16:30:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

A Python package for detecting core-periphery structure in networks
===================================================================

This package contains algorithms for detecting core-periphery structure in networks.
All algorithms are implemented in Python, with speed accelerations by numba, and can be used with a small coding effort.

See the `project page <https://github.com/skojaku/core-periphery-detection>`_ for the usage of this package.


Installation
************

Before installing this package, make sure that you have a **Python with version 3.6 or above**.

pip is the most easiest way to install:

.. code-block:: bash

    pip install cpnet

For conda users, although the package can be install using pip without problem in conda environement, you may want to avoid mixing pip with conda. In this case, we recommend making a link to the package:

.. code-block:: bash

    git clone https://github.com/skojaku/core-periphery-detection
    cd core-periphery-detection
    conda develop .

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
