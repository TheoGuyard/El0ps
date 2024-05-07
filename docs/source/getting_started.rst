.. _getting_started:

===============
Getting started
===============

This page provides a quick starting guide to ``el0ps``.

What's in the box
-----------------

Mathematically speaking, ``el0ps`` addresses optimization problems expressed as

.. math::

   \textstyle\min_{x \in \mathbb{R}^{n}} f(Ax) + \lambda\|x\|_0 + h(x)

where :math:`f` is a **datafit** function, :math:`h` is a **penalty** function, :math:`A \in \mathbb{R}^{m \times n}` is a matrix and :math:`\lambda>0` is an hyperparameter.
The package provides efficient solvers for this problem, methods to fit regularization paths and other utilities.
Check out the :ref:`Ingredients<ingredients>` page for more details.

.. important::

    ``el0ps`` comes with already-made datafits and penalties. It is also designed to be modular and allows users to define their own.
    This can be make through convinient templates classes. Check out the :ref:`Customize you own problem<custom>` page for more details.



Creating and solving a problem
------------------------------

Here is a simple example showing how to solve an instance of L0-penalized problem with a Least-squares datafit and a Big-M constraint penalty for a fixed :math:`\lambda`.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_regression
    from el0ps.datafits import Leastsquares
    from el0ps.penalties import Bigm
    from el0ps.solvers import BnbSolver

    # Generate sparse regression data using scikit-learn
    A, y, x = make_regression(n_samples=50, n_features=100, coef=True)
    M = 10. * np.linalg.norm(x, np.inf)

    # Instantiate the problem and solve it using el0ps's BnB solver
    datafit = Leastsquares(y)
    penalty = Bigm(M)
    solver = BnbSolver()
    result = solver.solve(datafit, penalty, A, lmbd=10.)

.. Scikit-learn estimators
.. -----------------------


.. Regularization path
.. -------------------

.. You can also fit a regularization path, i.e., solve the problem over a range of :math:`\lambda`, as simply as follows.

.. .. code-block:: python

..     from el0ps import Path
..     path = Path()
..     data = path.fit(solver, datafit, penalty, A)

.. The documentation references ``el0ps``'s already-made datafits and penalties other than the :class:`.datafit.Leastsquares` and :class:`.penalty.Bigm` ones.
