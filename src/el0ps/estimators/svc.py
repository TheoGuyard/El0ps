"""Base classes for L0-norm SVC estimators."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin
from el0ps.solvers import BaseSolver, BnbSolver
from el0ps.datafits import Squaredhinge
from .base import BaseL0Estimator, select_bigml1l2_penalty, _fit


class BaseL0SVC(BaseL0Estimator, ClassifierMixin):
    """Base class for L0-norm SVC estimators."""

    pass


class L0L1L2SVC(BaseL0SVC):
    r"""Sparse SVC with L0L1L2-norm regularization.

    The optimization problem solved is

    .. math::
        min     sum(max(1 - y * (X @ w), 0)^2) + lmbd ||w||_0 + alpha ||w||_1 + beta ||w||_2^2
        s.t.    ||w||_inf <= M

    The parameters `alpha` and `beta` can be set to zero and an inifite value
    of `M` is allowed. However, setting `alpha=0`, `beta=0` and `M=np.inf`
    simulteanously is not allowed.


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
        super().__init__(lmbd, fit_intercept, solver)
        self.alpha = alpha
        self.beta = beta
        self.M = M

    def fit(self, X: ArrayLike, y: ArrayLike):
        datafit = Squaredhinge(y)
        penalty = select_bigml1l2_penalty(self.alpha, self.beta, self.M)
        return _fit(self, datafit, penalty, X, self.lmbd, self.solver)


class L0SVC(L0L1L2SVC):
    """Substitute for :class:`estimators.L0L1L2SVC` with parameters `alpha=0`
    and `beta=0`."""

    def __init__(
        self,
        lmbd: float,
        M: float,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, 0.0, 0.0, M, fit_intercept, solver)


class L0L1SVC(L0L1L2SVC):
    """Substitute for :class:`estimators.L0L1L2SVC` with parameter `beta=0`."""

    def __init__(
        self,
        lmbd: float,
        alpha: float,
        M: float = np.inf,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, alpha, 0.0, M, fit_intercept, solver)


class L0L2SVC(L0L1L2SVC):
    """Substitute for :class:`estimators.L0L1L2SVC` with parameter
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
