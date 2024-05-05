"""Base classes for L0-norm classification estimators."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin
from el0ps.solvers import BaseSolver, BnbSolver
from el0ps.datafits import Logistic
from el0ps.penalties import (
    Bigm,
    BigmL1norm,
    BigmL2norm,
    L1L2norm,
    L1norm,
    L2norm,
)
from .base import BaseL0Estimator, TargetClass, _fit


class BaseL0Classification(BaseL0Estimator, ClassifierMixin):
    """Base class for L0-norm classification estimators."""

    def target_class(self):
        return TargetClass.BINARY


class L0Classification(BaseL0Classification):
    r"""Sparse classification with L0-norm regularization. The formulation
    includes a Big-M constraint to enable mixed-integer programming solution
    methods.

    The optimization problem solved is

    .. math::
        min     sum(log(1 + exp(-y * (X @ w)))) + lmbd ||w||_0
        s.t.    ||w||_inf <= M

    Parameters
    ----------
    lmbd: float
        L0-norm weight.
    M: float
        Big-M value.
    fit_intercept: bool, default=False
        Whether to fit an intercept term.
    solver: BaseSolver, default=BnbSolver()
        Solver for the estimator associated problem.
    """

    def __init__(
        self,
        lmbd: float,
        M: float,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, fit_intercept, solver)
        self.M = M

    def fit(self, X: ArrayLike, y: ArrayLike):
        datafit = Logistic(y)
        penalty = Bigm(self.M)
        return _fit(self, datafit, penalty, X, self.lmbd, self.solver)


class L0L1Classification(BaseL0Classification):
    r"""Sparse classification with L0L1-norm regularization. The formulation
    can include a Big-M constraint to strengthen mixed-integer programming
    solution methods.

    The optimization problem solved is

    .. math::
        min     sum(log(1 + exp(-y * (X @ w)))) + lmbd ||w||_0 + alpha ||w||_1
        s.t.    ||w||_inf <= M


    Parameters
    ----------
    lmbd: float
        L0-norm weight.
    alpha: float
        L1-norm weight.
    M: float, default=np.inf
        Big-M value (no Big-M constraint is enforced when M=np.inf).
    fit_intercept: bool, default=False
        Whether to fit an intercept term.
    solver: BaseSolver, default=BnbSolver()
        Solver for the estimator associated problem.
    """

    def __init__(
        self,
        lmbd: float,
        alpha: float,
        M: float = np.inf,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, fit_intercept, solver)
        self.alpha = alpha
        self.M = M

    def fit(self, X: ArrayLike, y: ArrayLike):
        datafit = Logistic(y)
        if self.M == np.inf:
            penalty = L1norm(self.alpha)
        else:
            penalty = BigmL1norm(self.alpha, self.M)
        return _fit(self, datafit, penalty, X, self.lmbd, self.solver)


class L0L2Classification(BaseL0Classification):
    r"""Sparse classification with L0L2-norm regularization. The formulation
    can include a Big-M constraint to strengthen mixed-integer programming
    solution methods.

    The optimization problem solved is

    .. math::
        min     sum(log(1 + exp(-y * (X @ w)))) + lmbd ||w||_0 + alpha ||w||_2^2
        s.t.    ||w||_inf <= M


    Parameters
    ----------
    lmbd: float
        L0-norm weight.
    alpha: float
        L2-norm weight.
    M: float, default=np.inf
        Big-M value (no Big-M constraint is enforced when M=np.inf).
    fit_intercept: bool, default=False
        Whether to fit an intercept term.
    solver: BaseSolver, default=BnbSolver()
        Solver for the estimator associated problem.
    """  # noqa: E501

    def __init__(
        self,
        lmbd: float,
        alpha: float,
        M: float = np.inf,
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, fit_intercept, solver)
        self.alpha = alpha
        self.M = M

    def fit(self, X: ArrayLike, y: ArrayLike):
        datafit = Logistic(y)
        if self.M == np.inf:
            penalty = L2norm(self.alpha)
        else:
            penalty = BigmL2norm(self.alpha, self.M)
        return _fit(self, datafit, penalty, X, self.lmbd, self.solver)


class L0L1L2Classification(BaseL0Classification):
    r"""Sparse classification with L0L1L2-norm regularization.

    The optimization problem solved is

    .. math::
        min sum(log(1 + exp(-y * (X @ w)))) + lmbd ||w||_0 + alpha ||w||_1 + beta ||w||_2^2


    Parameters
    ----------
    lmbd: float
        L0-norm weight.
    alpha: float
        L1-norm weight.
    beta: float
        L2-norm weight.
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
        fit_intercept: bool = False,
        solver: BaseSolver = BnbSolver(),
    ):
        super().__init__(lmbd, fit_intercept, solver)
        self.alpha = alpha
        self.beta = beta

    def fit(self, X: ArrayLike, y: ArrayLike):
        datafit = Logistic(y)
        penalty = L1L2norm(self.alpha, self.beta)
        return _fit(self, datafit, penalty, X, self.lmbd, self.solver)
