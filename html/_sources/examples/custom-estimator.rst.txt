.. _examples-custom-estimator:

Custom estimator
----------------

The following example shows how to implement a custom estimator in el0ps, as explained in the :ref:`custom-estimator` section.

.. code-block:: python
    
    from el0ps.datafit import Logcosh
    from el0ps.penalty import L2norm
    from el0ps.estimator import L0Estimator

    class L0L2LogcoshEstimator(L0Estimator):
        """
        L0-problem-based estimator corresponding to the solution of

            min_{x in R^n} LogCosh(Ax) + lambda * ||x||_0 + beta ||x||_2^2

        for some lambda > 0, beta > 0, and where
        
            LogCosh(w) = sum_{i=1,...,m} log(cosh(yi - wi))

        for some y in R^m.
        """

        def __init__(self, lmbd: float, beta: float) -> None:
            # The matrix A and the datafit attribute y are set when calling the
            # fit method of the L0Estimator class. They need not be initialized
            # here.
            datafit = Logcosh()
            penalty = L2norm(beta)
            super().__init__(datafit, penalty, lmbd)
        