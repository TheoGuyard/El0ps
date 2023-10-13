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
You can pick from ``el0ps``'s already-made estimators or customize your own by building on top of the available datafits and penalties template classes.


Installation
------------

``el0ps`` is available on `pipy <https://pypi.org>`_. 
Get the latest version of the package by running the following command.

.. code-block:: shell

   $ pip install el0ps


Getting started
---------------

Here is a simple example of how to use ``el0ps`` to solve an L0-penalized problem.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_regression
    from el0ps import Problem, Path
    from el0ps.datafit import Leastsquares
    from el0ps.penalty import Bigm
    from el0ps.solver import BnbSolver

    # Generate sparse regression data
    A, y, x = make_regression(n_samples=50, n_features=100, coef=True)
    M = 10. * np.linalg.norm(x, np.inf)

    # Instantiate the datafit and penalty functions
    datafit = Leastsquares(y)
    penalty = Bigm(M)

    # Solve the problem for a fixed lambda using el0ps's BnB solver
    lmbd = 10.
    problem = Problem(datafit, penalty, A, lmbd)
    solver = BnbSolver()
    result = solver.solve(problem)

    # Fit a regularization path by varying the lambda parameter
    path = Path()
    fit_data = path.fit(solver, datafit, penalty, A)


Cite
----

``el0ps`` is distributed under
`AGPL v3 license <https://github.com/TheoGuyard/El0ps/blob/main/LICENSE>`_.
Please cite the package as follows:

..

    Todo : Add citation

.. .. code-block:: bibtex

..    @inproceedings{skglm,
..       title     = {},
..       author    = {},
..       booktitle = {},
..       year      = {},
..    }


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
