.. El0ps documentation master file, created by
   sphinx-quickstart on Fri Oct 13 13:46:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================================
El0ps: An Exact L0-Problem Solver
=================================

-----------------


``el0ps`` is a Python package providing utilities to handle **L0-regularized** problems expressed as

.. math::

   \textstyle\min_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{Ax}) + \lambda\|\mathbf{x}\|_0 + h(\mathbf{x})

appearing in several applications.
These problems aim at minimizing a trade-off between a data-fidelity function :math:`f` composed with a matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` and the L0-norm which counts the number of non-zeros in its argument to promote sparse solutions.
The additional penalty function :math:`h` can be used to enforce other desirable properties on the solutions and is involved in the construction of efficient solution methods.
The package includes:

- A **flexible** framework with built-in problem instances and the possibility to define custom ones,

- A **state-of-the-art** solver based on a specialized Branch-and-Bound algorithm,

- An **interface** to `scikit-learn <https://scikit-learn.org>`_ providing linear model estimators based on L0-regularized problems.

.. important::

   Get a quick tour of the package with the :ref:`getting_started` page.


Installation
------------

``el0ps`` is available on `pypi <https://pypi.org>`_. The latest version of the package can be installed as follows:

.. prompt:: shell $

   pip install el0ps

Contribute
----------

Feel free to contribute by report any bug on the `issue <https://github.com/TheoGuyard/El0ps/issues>`_ page or by opening a `pull request <https://github.com/TheoGuyard/El0ps/pulls>`_.
Any feedback or contribution is welcome.
Check out the :ref:`contribute` page for more information.


Cite
----

``el0ps`` is distributed under
`AGPL v3 license <https://github.com/TheoGuyard/El0ps/blob/main/LICENSE>`_.
Please cite it as follows:

.. code-block:: bibtex

   @inproceedings{el0ps2024guyard,
      title        = {A New Branch-and-Bound Pruning Framework for L0-Regularized Problems},
      author       = {Guyard, Th{\'e}o and Herzet, C{\'e}dric and Elvira, Cl{\'e}ment and Ayse-Nur Arslan},
      booktitle    = {International Conference on Machine Learning (ICML)},
      year         = {2024},
      organization = {PMLR},
   }


.. |Python 3.9+| image:: https://img.shields.io/badge/python-3.9%2B-blue
   :target: https://www.python.org/downloads/release/python-390/
.. |Documentation| image:: https://img.shields.io/badge/documentation-latest-blue
   :target: https://el0ps.github.io
.. |PyPI version| image:: https://badge.fury.io/py/el0ps.svg
   :target: https://pypi.org/project/el0ps/
.. |Test Status| image:: https://github.com/TheoGuyard/el0ps/actions/workflows/test.yml/badge.svg
   :target: https://github.com/TheoGuyard/el0ps/actions/workflows/test.yml
.. |Codecov| image:: https://codecov.io/gh/TheoGuyard/El0ps/graph/badge.svg?token=H2IA4O67X6
   :target: https://codecov.io/gh/TheoGuyard/El0ps
.. |License| image:: https://img.shields.io/badge/License-AGPL--v3-red.svg
   :target: https://github.com/benchopt/benchopt/blob/main/LICENSE


.. toctree::
   :maxdepth: 1
   :hidden:
   :includehidden:

   getting_started.rst
   custom.rst
   examples/examples.rst
   api.rst
   contribute.rst
   news.rst
