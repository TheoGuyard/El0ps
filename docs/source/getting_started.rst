.. _getting_started:

===============
Getting started
===============

Here is a simple example showing how to solve an instance of L0-penalized problem with a Least-squares datafit and a Big-M constraint penalty for a fixed :math:`\lambda`.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_regression
    from el0ps.datafit import Leastsquares
    from el0ps.penalty import Bigm
    from el0ps.solver import BnbSolver

    # Generate sparse regression data using scikit-learn
    A, y, x = make_regression(n_samples=50, n_features=100, coef=True)
    M = 10. * np.linalg.norm(x, np.inf)

    # Instantiate the problem and solve it using el0ps's BnB solver
    datafit = Leastsquares(y)
    penalty = Bigm(M)
    solver = BnbSolver()
    result = solver.solve(datafit, penalty, A, lmbd=10.)

You can also fit a regularization path, i.e., solve the problem over a range of :math:`\lambda`, as simply as follows.

.. code-block:: python

    from el0ps import Path
    path = Path()
    data = path.fit(solver, datafit, penalty, A)

The documentation references ``el0ps``'s already-made datafits and penalties other than the :class:`.datafit.Leastsquares` and :class:`.penalty.Bigm` ones.
