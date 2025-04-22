.. _getting_started:

===============
Getting started
===============

``el0ps`` provides utilities to handle L0-regularized problems expressed as

.. math::

   \textstyle\min_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{Ax}) + \lambda\|\mathbf{x}\|_0 + h(\mathbf{x})

where :math:`f` is a datafit function, :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` is a matrix, :math:`\lambda > 0` is a parameter, :math:`\|\cdot\|_0` counts the number of non-zero entries in its input, and :math:`h` is a penalty function.
This section explains how to instantiate a problem, solve it, construct regularization paths, and integrate ``el0ps`` with the `scikit-learn <https://scikit-learn.org>`_ pipeline ecosystem.

Instantiating a problem
-----------------------

To define a problem instance, the package provides built-in datafit function :math:`f` and penalty function :math:`h`.
Moreover, any `numpy <https://numpy.org>`_-compatible object for the matrix :math:`\mathbf{A}` and standard float for the parameter :math:`\lambda` can be used.
The example below illustrates how to instantiate the components of an L0-regularized problem with a least-squares datafit and an L2-norm penalty.

.. code-block:: python

    from sklearn.datasets import make_regression
    from el0ps.datafit import Leastsquares
    from el0ps.penalty import L2norm

    # Generate sparse regression data using scikit-learn
    A, y = make_regression(n_informative=5, n_samples=100, n_features=200)

    datafit = Leastsquares(y)   # datafit function f(Ax) = (1/2) * ||y - Ax||_2^2
    penalty = L2norm(beta=0.1)  # penalty function h(x) = beta * ||x||_2^2
    lmbd = 0.01                 # L0-norm weight

All the built-in datafit and penalty functions are listed in the :ref:`api_references` page.
Besides, custom ones can also be defined by users based on template classes to better suit application needs. 
Check-out the :ref:`custom` page for more details.

Solving a problem
-----------------

.. currentmodule:: el0ps.solver

With all problem components defined, the resulting L0-regularized problem can be solved using one of the solvers included in the package.  
This can be done as simply as in the following example.

.. code-block:: python

    from el0ps.solver import BnbSolver

    solver = BnbSolver()
    result = solver.solve(datafit, penalty, A, lmbd)

The :class:`BnbSolver` class accepts various options to control its behavior and stopping criteria.  
Its ``solve`` method returns a :class:`Result` object containing the solution and useful information on the solving process.
Alternative solvers are also available.
Check-out the :ref:`api_references` for more details.

Constructing regularization paths
---------------------------------

.. currentmodule:: el0ps.path

A common task when working with L0-regularized problems is the construction of regularization paths, that is, the compilation of the problem solutions for multiple values of parameter :math:`\lambda`.  
The package provides a dedicated utility to automate this process as follows.

.. code-block:: python

    from el0ps.path import Path

    lmbds = [0.1, 0.01, 0.001]
    path = Path(lmbds=lmbds)
    results = path.fit(solver, datafit, penalty, A)

This code builds a regularization path for all values of :math:`\lambda` in ``[0.1, 0.01, 0.001]``.  
Options of the :class:`Path` class also allow to automatically constrict a suitable parameter grid.  
The ``fit`` method returns a dictionary mapping each value of :math:`\lambda` to its corresponding result object.

Scikit-Learn estimators
-----------------------

The package also offers `linear model <https://scikit-learn.org/stable/modules/linear_model.html#linear-model>`_ estimators correspond to solutions of L0-regularized problems.  
These estimators are fully compatible with the `scikit-learn <https://scikit-learn.org>`_ ecosystem.
They can be used as follows.

.. code-block:: python

    from el0ps.estimator import L0L2Regressor

    # Estimator corresponding to a solution of the L0-regularized problem
    # min_{x in R^n} (1/2) ||y - Ax||_2^2 + lmbd ||x||_0 + beta ||x||_2^2
    estimator = L0L2Regressor(lmbd=0.1, beta=0.01)
    
    # Split training and testing sets
    A_train, A_test, y_train, y_test = train_test_split(A, y)

    # Standard fit/score workflow of scikit-learn
    estimator.fit(A_train, y_train)
    estimator.score(A_test, y_test)

    # Standard hyperparameter tuning pipeline of scikit-learn
    params = {'lmbd': [0.1, 1.], 'beta': [0.1, 1.]}
    grid_search_cv = GridSearchCV(reg, params)
    grid_search_cv.fit(A, y)
    best_params = grid_search_cv.best_params_

All the built-in estimators are listed in the :ref:`api_references` page.
Besides, custom ones can also be defined by users based on template classes to better suit application needs. 
Check-out the :ref:`custom` page for more details.