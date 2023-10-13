.. El0ps documentation master file, created by
   sphinx-quickstart on Fri Oct 13 13:46:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====
El0ps
=====
*-An Exact L0-Problem Solver-*


|Python 3.8+| |Documentation| |Test Status| |Codecov| |PyPI version| |License|

``el0ps`` is a Python package allowing to solve L0-penalized problems.
It is designed to be numerically efficient and supports a wide range of loss and penalty functions.
You can pick from ``el0ps``'s already-made estimators or customize your own by building on to of the available datafits and penalties template classes.


Quick start
-----------

``el0ps`` is available on `pipy <https://pypi.org>`_. 
Get the latest version of the package by running the following command.

.. code-block:: shell

   $ pip install el0ps

You are now ready to take your first steps with ``el0ps`` in the :ref:`Getting started section <getting_started>`.
Typical use-cases and workflows are prented in the :ref:`Examples section <examples>`.
Further details on how the package can be found in the :ref:`User guide section <user_guide>`.
You can also learn how to customize your own estimators and solvers in the :ref:`Custom estimators and solvers <custom>`.


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


Documentation tree
------------------

.. toctree::
   :maxdepth: 1

   getting_started.rst
   user_guide.rst
   custom.rst
   examples.rst
   api.rst


.. |Python 3.8+| image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/release/python-380/
.. |Documentation| image:: https://img.shields.io/badge/documentation-latest-blue
   :target: https://el0ps.github.io
.. |Test Status| image:: https://github.com/TheoGuyard/el0ps/actions/workflows/test.yml/badge.svg
   :target: https://github.com/TheoGuyard/el0ps/actions/workflows/test.yml
.. |Codecov| image:: https://codecov.io/gh/TheoGuyard/El0ps/graph/badge.svg?token=H2IA4O67X6
   :target: https://codecov.io/gh/TheoGuyard/El0ps
.. |PyPI version| image:: https://badge.fury.io/py/el0ps.svg
   :target: https://pypi.org/project/el0ps/
.. |License| image:: https://img.shields.io/badge/License-AGPL--v3-red.svg
   :target: https://github.com/benchopt/benchopt/blob/main/LICENSE
