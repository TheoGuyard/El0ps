"""Base classes for L0-norm classifier estimators."""

import numpy as np
from sklearn.base import ClassifierMixin
from el0ps.solver import BaseSolver, BnbSolver
from el0ps.datafit import Logistic
from .base import L0Estimator
from .utils import select_bigml1l2_penalty


class L0L1L2Classifier(L0Estimator, ClassifierMixin):
    """Scikit-learn-compatible `LinearModel` classifier estimator
    corresponding to a solution of L0-regularized problems expressed as

        `min_{||w||_{infty} <= M} 0.5 * sum(log(1 + exp(-(Xw * y)))) + lmbd * ||w||_0 + alpha * ||w||_1 + beta * ||w||_2^2`

    where `alpha >= 0`, `beta >= 0` and `M > 0`. Setting `alpha = 0`,
    `beta = 0` or `M = infty` is allowed, but not simultaneously.


    Parameters
    ----------
    lmbd: float
        L0-norm weight.
    alpha: float
        L1-norm weight.
    beta: float
        L2-norm weight.
    M: float, default=np.inf
        Big-M value.
    fit_intercept: bool, default=False
        Whether to fit an intercept term.
    solver: BaseSolver, default=BnbSolver()
        Solver for the estimator associated problem.
    """  # noqa: E501

    def __init__(
        self,
        lmbd: float,
        alpha: float,
        beta: float,
        M: float = np.inf,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        self.lmbd = lmbd
        self.alpha = alpha
        self.beta = beta
        self.M = M
        datafit = Logistic(np.zeros(0))
        penalty = select_bigml1l2_penalty(alpha, beta, M)
        super().__init__(datafit, penalty, lmbd, fit_intercept, solver)


class L0Classifier(L0L1L2Classifier):
    """Substitute for :class:`.estimators.L0L1L2Classifier` with parameters
    `alpha=0` and `beta=0`."""

    def __init__(
        self,
        lmbd: float,
        M: float,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, 0.0, 0.0, M, fit_intercept, solver)


class L0L1Classifier(L0L1L2Classifier):
    """Substitute for :class:`.estimators.L0L1L2Classifier` with parameter
    `beta=0`."""

    def __init__(
        self,
        lmbd: float,
        alpha: float,
        M: float = np.inf,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, alpha, 0.0, M, fit_intercept, solver)


class L0L2Classifier(L0L1L2Classifier):
    """Substitute for :class:`.estimators.L0L1L2Classifier` with parameter
    `alpha=0`."""

    def __init__(
        self,
        lmbd: float,
        beta: float,
        M: float = np.inf,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, 0.0, beta, M, fit_intercept, solver)
