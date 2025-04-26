"""Base classes for L0-norm regression estimators."""

import numpy as np
from sklearn.linear_model._base import RegressorMixin

from el0ps.solver import BaseSolver, BnbSolver
from el0ps.datafit import Leastsquares
from el0ps.estimator.base import L0Estimator
from el0ps.estimator.utils import select_bigml1l2_penalty


class L0L1L2Regressor(L0Estimator, RegressorMixin):
    r"""Scikit-learn-compatible `linear model <https://scikit-learn.org/stable/api/sklearn.linear_model.html>`_
    regression estimators with L0L1L2-regularization.

    The estimator corresponds to a solution of the problem

    .. math::

        \textstyle\min_{\|\mathbf{x}\|_{\infty} \leq M} f(\mathbf{Ax}) + \lambda\|\mathbf{x}\|_0 + \alpha\|\mathbf{x}\|_1 + \beta\|\mathbf{x}\|_2^2

    where :math:`f` is a :class:`el0ps.datafit.Leastsquares` function,
    :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` is a matrix,
    :math:`\lambda > 0` is a parameter, the L0-norm :math:`\|\cdot\|_0` counts
    the number of non-zero entries in its input, and :math:`h` is a penalty
    function."""  # noqa: E501

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
        datafit = Leastsquares(np.zeros(0))
        penalty = select_bigml1l2_penalty(alpha, beta, M)
        super().__init__(datafit, penalty, lmbd, fit_intercept, solver)


class L0Regressor(L0L1L2Regressor):
    """Substitute for :class:`L0L1L2Regressor` with parameters ``alpha=0`` and
    ``beta=0``."""

    def __init__(
        self,
        lmbd: float,
        M: float,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, 0.0, 0.0, M, fit_intercept, solver)


class L0L1Regressor(L0L1L2Regressor):
    """Substitute for :class:`L0L1L2Regressor` with parameter ``beta=0``."""

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
    """Substitute for :class:`L0L1L2Regressor` with parameters ``alpha=0``."""

    def __init__(
        self,
        lmbd: float,
        beta: float,
        M: float = np.inf,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, 0.0, beta, M, fit_intercept, solver)
