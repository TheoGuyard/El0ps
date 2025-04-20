"""Base classes for L0-norm regression estimators."""

import numpy as np
from sklearn.base import RegressorMixin
from el0ps.solver import BaseSolver, BnbSolver
from el0ps.datafit import Leastsquares
from .base import L0Estimator
from .utils import select_bigml1l2_penalty


class L0L1L2Regressor(L0Estimator, RegressorMixin):
    """Scikit-learn-compatible `LinearModel` regression estimator corresponding
    to a solution of L0-regularized problems expressed as

        `min_{||w||_{infty} <= M} 0.5 * ||Xw - y||_2^2 + lmbd * ||w||_0 + alpha * ||w||_1 + beta * ||w||_2^2`

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
        datafit = Leastsquares(np.zeros(0))
        penalty = select_bigml1l2_penalty(alpha, beta, M)
        super().__init__(datafit, penalty, lmbd, fit_intercept, solver)


class L0Regressor(L0L1L2Regressor):
    """Substitute for :class:`.estimators.L0L1L2Regressor` with parameters
    `alpha=0` and `beta=0`."""

    def __init__(
        self,
        lmbd: float,
        M: float,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, 0.0, 0.0, M, fit_intercept, solver)


class L0L1Regressor(L0L1L2Regressor):
    """Substitute for :class:`.estimators.L0L1L2Regressor` with parameter
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


class L0L2Regressor(L0L1L2Regressor):
    """Substitute for :class:`.estimators.L0L1L2Regressor` with parameters
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
