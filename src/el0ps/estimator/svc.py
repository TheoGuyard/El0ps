"""Base classes for L0-norm SVC estimators."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin
from el0ps.solver import BaseSolver, BnbSolver
from el0ps.datafit import Squaredhinge
from .base import BaseL0Estimator, select_bigml1l2_penalty, _fit


class BaseL0SVC(BaseL0Estimator, ClassifierMixin):
    """Base class for L0-norm SVC estimators."""

    pass


class L0L1L2SVC(BaseL0SVC):
    r"""Sparse SVC with L0L1L2-norm regularization and Big-M constraint.

    The optimization problem solved is

    .. math::

        \min        & \ \ \textstyle \|[1 - y \odot Xw]_+\|_2^2 + \lambda \|w\|_0 + \alpha \|w\|_1 + \beta \|w\|_2^2 \\
        \text{s.t.} & \ \ \|w\|_{\infty} \leq M

    where :math:`\alpha \geq 0`, :math:`\beta \geq 0` and :math:`M > 0`.
    Setting :math:`alpha=0`, :math:`beta=0` or :math:`M=\infty` is allowed, but
    not simulteanously.

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
    """Substitute for :class:`.estimators.L0L1L2SVC` with parameters
    ``alpha=0`` and ``beta=0``."""

    def __init__(
        self,
        lmbd: float,
        M: float,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, 0.0, 0.0, M, fit_intercept, solver)


class L0L1SVC(L0L1L2SVC):
    """Substitute for :class:`.estimators.L0L1L2SVC` with parameter
    ``beta=0``."""

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
    """Substitute for :class:`.estimators.L0L1L2SVC` with parameter
    ``alpha=0``."""

    def __init__(
        self,
        lmbd: float,
        beta: float,
        M: float = np.inf,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, 0.0, beta, M, fit_intercept, solver)
