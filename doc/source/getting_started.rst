.. _getting_started:

===============
Getting started
===============

This page provides a quick starting guide to ``el0ps``.

What's in the box
-----------------

``el0ps`` addresses optimization problems expressed as

.. math::

   \tag{$\mathcal{P}$}\textstyle\min_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{Ax}) + \lambda\|\mathbf{x}\|_0 + h(\mathbf{x})

where :math:`f(\cdot)` is a `datafit` function, :math:`h(\cdot)` is a `penalty` function, :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` is a matrix, :math:`\|\cdot\|_0` is the so-called :math:`\ell_0`-norm defined for all :math:`\mathbf{x} \in \mathbb{R}^n` as

.. math:: \|\mathbf{x}\|_0 = \mathrm{card}(\{i \in 1,\dots,n \mid x_i \neq 0\})

and :math:`\lambda>0` is an hyperparameter.
The package provides efficient solvers for this family of problems, methods to fit regularization paths, bindings for `scikit-learn <https://scikit-learn.org>`_ estimators and other utilities.
Check out the :ref:`Ingredients<ingredients>` page for more details.

.. tip::

    ``el0ps`` comes with already-made datafits and penalties. It is also designed to be modular and allows users to define their own.
    Check out the :ref:`User-defined problems<custom>` page for more details.


Solving a problem
-----------------

Here is a simple example showing how to solve an instance of problem :math:`(\mathcal{P})` with a least-squares datafit and an :math:`\ell_2`-norm penalty function.

.. code-block:: python

    import numpy as np
    from el0ps.datafit import Leastsquares
    from el0ps.penalty import L2norm
    from el0ps.solver import BnbSolver

    # Generate sparse regression data
    np.random.seed(0)
    x = np.zeros(100)
    s = np.random.randint(100, size=5)
    x[s] = 1.
    A = np.random.randn(50, 100)
    A /= np.linalg.norm(A, ord=2)
    y = A @ x
    e = np.random.randn(50)
    e *= np.sqrt((y @ y) / (10. * (e @ e)))
    y += e

    # Instantiate the function f(Ax) = (1/2) * ||y - Ax||_2^2
    datafit = Leastsquares(y)

    # Instantiate the function h(x) = beta * ||x||_2^2
    penalty = L2norm(beta=0.1)
    
    # Solve the problem with el0ps' Branch-and-Bound solver
    solver = BnbSolver()
    result = solver.solve(datafit, penalty, A, lmbd=0.01)

You can pass various options to the solver (see the :class:`.solvers.BnbOptions` documentation).
Once the problem is solved, you can recover different quantities such as the solver status, the solution or the optimal value of the problem from the ``result`` variable (see the :class:`.solvers.Result` documentation).


Fitting regularization paths
----------------------------

You can also fit a regularization path where problem :math:`(\mathcal{P})` is solved over a grid of :math:`\lambda`.
Fitting a path with ``lmbd_num`` different values of this parameter logarithmically spaced from some ``lmbd_max`` to some ``lmbd_min`` can be simply done as follows.

.. code-block:: python

    from el0ps.path import Path

    path = Path(lmbd_max=1e-0, lmbd_min=1e-2, lmbd_num=20)
    data = path.fit(solver, datafit, penalty, A)

Once the path is fitted, you can recover different statistics ``data`` variable such as the number of non-zeros in the solution, the datafit value or the solution time.
Various other options can be passed to the :class:`.Path` class (see the :class:`.path.Path` documentation).
An option of interest is the ``lmbd_scaled`` which is ``False`` by default.
When setting ``lmbd_scaled=True``, the values of the parameter :math:`\lambda` are scaled so that the first solution constructed in the path when ``lmbd=lmbd_max`` correponds to the all-zero vector. 

Scikit-Learn estimators
-----------------------

``el0ps`` also provides `scikit-learn <https://scikit-learn.org>`_ compatible estimators based on problem :math:`(\mathcal{P})`.
They can be used similarly to any other estimator in the package pipeline as follows.

.. code-block:: python

    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from el0ps.estimator import L0Regressor

    # Generate sparse regression data
    A, y = make_regression(n_informative=5, n_samples=100, n_features=200)
    
    # Split training and testing sets
    A_train, A_test, y_train, y_test = train_test_split(A, y)

    # Initialize a regerssor with L0-norm regularization with Big-M constraint
    estimator = L0Regressor(lmbd=0.1, M=1.)

    # Fit and score the estimator manually ...
    estimator.fit(A_train, y_train)
    estimator.score(A_test, y_test)

    # ... or in a pipeline
    pipeline = Pipeline([('estimator', estimator)])
    pipeline.fit(A_train, y_train)
    pipeline.score(A_test, y_test)

Like datafit and penalty functions, you can customize your own estimators.