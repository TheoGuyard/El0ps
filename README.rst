.. El0ps documentation master file, created by
   sphinx-quickstart on Fri Oct 13 13:46:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====
El0ps
=====
*-An Exact L0-Problem Solver-*


|Test Status| |Codecov| |Documentation| |Python 3.8+| |PyPI version| |License|

``el0ps`` is a Python package allowing to solve L0-penalized problems.
It is designed to be numerically efficient and supports a wide range of loss and penalty functions.
You can pick from ``el0ps``'s already-made estimators or customize your own by building on to of the available datafits and penalties template classes.


Installation
------------

``el0ps`` is available on `pipy <https://pypi.org>`_. 
Get the latest version of the package by running the following command.

.. code-block:: shell

   $ pip install el0ps


Cite
----

``el0ps`` is distributed under
`AGPL v3 license <https://github.com/TheoGuyard/El0ps/blob/main/LICENSE>`_.
Please cite the package as follows:

.. todo:: Add citation

.. .. code-block:: bibtex

..    @inproceedings{skglm,
..       title     = {},
..       author    = {},
..       booktitle = {},
..       year      = {},
..    }

.. |Test Status| image:: https://github.com/el0ps/el0ps/actions/workflows/test.yml/badge.svg
   :target: https://github.com/el0ps/el0ps/actions/workflows/test.yml
.. |Codecov| image:: https://codecov.io/gh/el0ps/el0ps/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/el0ps/el0ps
.. |Documentation| image:: https://img.shields.io/badge/documentation-latest-blue
   :target: https://el0ps.github.io
.. |Python 3.8+| image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/release/python-380/
.. |License| image:: https://img.shields.io/badge/License--AGPL-v3-blue.svg
   :target: https://github.com/benchopt/benchopt/blob/main/LICENSE
.. |PyPI version| image:: https://badge.fury.io/py/el0ps.svg
   :target: https://pypi.org/project/el0ps/
