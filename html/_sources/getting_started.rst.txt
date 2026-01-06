.. _getting_started:

===============
Getting started
===============

``el0ps`` provides utilities to handle L0-regularized problems expressed as

.. math::

   \textstyle\min_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{Ax}) + \lambda\|\mathbf{x}\|_0 + h(\mathbf{x})

where :math:`f` is a datafit function, :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` is a matrix, :math:`\lambda > 0` is a parameter, the L0-norm :math:`\|\cdot\|_0` counts the number of non-zero entries in its input, and :math:`h` is a penalty function.
This section explains how to:
 
- Instantiate a problem,
- Solve a problem,
- Construct a regularization path,
- Integrate ``el0ps`` with the `scikit-learn <https://scikit-learn.org>`_ ecosystem.

Instantiating a problem
-----------------------

To define a problem instance, the package provides built-in datafit functions :math:`f` and penalty functions :math:`h`.
Moreover, any `numpy <https://numpy.org>`_-compatible object for the matrix :math:`\mathbf{A}` and standard float for the parameter :math:`\lambda` can be used.
The example below illustrates how to instantiate the components of an L0-regularized problem expressed as

.. math::

   \textstyle\min_{\mathbf{x} \in \mathbb{R}^{n}} \tfrac{1}{2}\|\mathbf{y} - \mathbf{Ax}\|_2^2 + \lambda\|\mathbf{x}\|_0 + \beta\|\mathbf{x}\|_2^2


involving a least-squares datafit and an L2-norm penalty.

.. code-block:: python

    from sklearn.datasets import make_regression
    from el0ps.datafit import Leastsquares
    from el0ps.penalty import L2norm

    # Generate sparse regression data using scikit-learn
    A, y = make_regression(n_informative=5, n_samples=100, n_features=200)

    datafit = Leastsquares(y)   # datafit f(Ax) = (1/2) * ||y - Ax||_2^2
    penalty = L2norm(beta=0.1)  # penalty h(x) = beta * ||x||_2^2
    lmbd = 0.01                 # L0-norm weight parameter

All the built-in datafit and penalty functions are listed in the :ref:`api_references` page.
Other instances not natively provided by the package can also be defined by users based on template classes to better suit application needs. 
Check-out the :ref:`custom` page for more details.

Solving a problem
-----------------

.. currentmodule:: el0ps.solver

With all problem components defined, the resulting L0-regularized problem can be solved using one of the solvers included in the package.  
This can be done as simply as follows.

.. code-block:: python

    from el0ps.solver import BnbSolver

    solver = BnbSolver()
    result = solver.solve(datafit, penalty, A, lmbd)

The :class:`BnbSolver` class accepts various options to control its behavior and stopping criteria.  
Its ``solve`` method returns a :class:`Result` object containing the problem solution as well as useful information on the solving process.
Alternatively, :class:`MipSolver` and :class:`OaSolver` solvers can be use, but are usually less efficient.
Check-out the :ref:`api_references` for more details.

Constructing a regularization path
----------------------------------

.. currentmodule:: el0ps.path

A common task when working with L0-regularized problems is the construction of a regularization path, that is, the compilation of the problem solutions for multiple values of parameter :math:`\lambda`.  
The package provides a dedicated utility to automate this process as follows.

.. code-block:: python

    from el0ps.path import Path

    path = Path(lmbds=[0.1, 0.01, 0.001])
    results = path.fit(solver, datafit, penalty, A)

This code constructs a regularization path for values of :math:`\lambda` in ``[0.1, 0.01, 0.001]``.  
Other options of the :class:`Path` class also be used to automatically construct a suitable parameter grid.  
The ``fit`` method returns a dictionary mapping each value of :math:`\lambda` to its corresponding result object.

Scikit-learn estimators
-----------------------

The package offers `linear model <https://scikit-learn.org/stable/modules/linear_model.html#linear-model>`_ estimators corresponding to solutions of L0-regularized problems.  
For instance, assume that one has access to some observation

.. math::

   \mathbf{y} = \mathbf{A}\mathbf{x}^{\dagger} + \boldsymbol{\epsilon}

of some vector :math:`\mathbf{x}^{\dagger} \in \mathbb{R}^n` through the linear mapping :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` and corrupted by some noise :math:`\boldsymbol{\epsilon} \in \mathbb{R}^m`.
Then, an estimator of this vector can be obtained as a solution of the L0-regularized problem with a least-squares datafit :math:`f(\mathbf{Ax}) = \tfrac{1}{2}\|\mathbf{y} - \mathbf{Ax}\|_2^2` and where the expression of :math:`h` and the value of :math:`\lambda` depends on the distribution of the noise.
``el0ps`` provides a set of estimators fully compatible with the `scikit-learn <https://scikit-learn.org>`_ ecosystem that are made by combining a given datafit, penalty, and value of :math:`\lambda`.
They can be used as follows.

.. code-block:: python

    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split, GridSearchCV
    from el0ps.estimator import L0L2Regressor

    # Linear model generation
    A, y = make_regression(n_informative=5, n_samples=100, n_features=200)

    # L0-problem-based estimator with
    #   - f(Ax) = 0.5 * ||y - Ax||_2^2
    #   - lmbd = 0.1
    #   - h(x) = 0.01 * ||x||_2^2
    estimator = L0L2Regressor(lmbd=0.1, beta=0.01)
    
    # Split training and testing sets
    A_train, A_test, y_train, y_test = train_test_split(A, y)

    # Standard fit/score workflow of scikit-learn
    estimator.fit(A_train, y_train)
    estimator.score(A_test, y_test)

    # Estimation of the underlying linear model vector
    x_estimated = estimator.coef_

    # Standard hyperparameter tuning pipeline of scikit-learn
    params = {'lmbd': [0.1, 1.], 'beta': [0.1, 1.]}
    grid_search_cv = GridSearchCV(reg, params)
    grid_search_cv.fit(A, y)
    best_params = grid_search_cv.best_params_

All the built-in estimators are listed in the :ref:`api_references` page.
Other instances not natively provided by the package can also be defined by users based on template classes to better suit application needs. 
Check-out the :ref:`custom` page for more details.