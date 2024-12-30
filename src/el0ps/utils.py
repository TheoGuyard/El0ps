"""Miscellaneous utilities."""

import numpy as np
from numpy.typing import ArrayLike
from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty


def compute_lmbd_max(
    datafit: BaseDatafit, penalty: BasePenalty, A: ArrayLike
) -> float:
    """Return a value of ``lmbd`` above which the all-zero vector is always a
    solution of the L0-penalized problem.

    Parameters
    ----------
    datafit: BaseDatafit
        Datafit function.
    penalty: BasePenalty
        Penalty function.
    A: ArrayLike
        Linear operator.

    Returns
    -------
    lmbd_max: float
        A value of ``lmbd`` above which the all-zero vector is always a
        solution of the L0-penalized problem.
    """
    w = np.zeros(A.shape[0])
    v = np.abs(A.T @ datafit.gradient(w))
    i = np.argmax(v)
    return penalty.conjugate_scalar(i, v[i])


def compute_param_slope_pos_scalar(
    penalty: BasePenalty,
    i: int,
    lmbd: float,
    tol: float = 1e-8,
    maxit: int = 100,
) -> float:
    """Utility function to compute the value of ``param_slope_pos_scalar`` in a
    :class:`.penalty.BasePenalty` instance when no closed-form is available.

    Parameters
    ----------
    penalty: BasePenalty
        The penalty instance.
    i: int
        Index of the splitting term.
    lmbd: float
        L0-regularization parameter.
    tol: float = 1e-4
        Bisection tolerance.
    maxit: int = 100
        Maximum number of bisection iterations.
    """
    a = 0.0
    b = 1.0
    while penalty.conjugate_scalar(i, b) < lmbd:
        b *= 2.0
        if b > 1e12:
            return np.inf
    for _ in range(maxit):
        c = (a + b) / 2.0
        fa = penalty.conjugate_scalar(i, a) - lmbd
        fc = penalty.conjugate_scalar(i, c) - lmbd
        if (-tol <= fc <= tol) or (b - a < 0.5 * tol):
            return c
        elif fc * fa >= 0.0:
            a = c
        else:
            b = c
    return c


def compute_param_slope_neg_scalar(
    penalty: BasePenalty,
    i: int,
    lmbd: float,
    tol: float = 1e-8,
    maxit: int = 100,
) -> float:
    """Utility function to compute the value of ``param_slope_neg_scalar`` in a
    :class:`.penalty.BasePenalty` instance when no closed-form is available.

    Parameters
    ----------
    penalty: BasePenalty
        The penalty instance.
    i: int
        Index of the splitting term.
    lmbd: float
        L0-regularization parameter.
    tol: float = 1e-4
        Bisection tolerance.
    maxit: int = 100
        Maximum number of bisection iterations.
    """
    a = -1.0
    b = 0.0
    while penalty.conjugate_scalar(i, a) < lmbd:
        a *= 2.0
        if a < -1e12:
            return -np.inf
    for _ in range(maxit):
        c = (a + b) / 2.0
        fa = penalty.conjugate_scalar(i, a) - lmbd
        fc = penalty.conjugate_scalar(i, c) - lmbd
        if (-tol <= fc <= tol) or (b - a < 0.5 * tol):
            return c
        elif fc * fa >= 0.0:
            a = c
        else:
            b = c
    return c


def compute_param_limit_pos_scalar(
    penalty: BasePenalty, i: int, lmbd: float
) -> float:
    """Utility function to compute the value of ``param_limit_pos_scalar`` in a
    :class:`.penalty.BasePenalty` instance when no closed-form is available.

    Parameters
    ----------
    penalty: BasePenalty
        The penalty instance.
    i: int
        Index of the splitting term.
    lmbd: float
        L0-regularization parameter.
    """
    param_slope_scalar = penalty.param_slope_pos_scalar(i, lmbd)
    return penalty.conjugate_subdiff_scalar(i, param_slope_scalar)[1]


def compute_param_limit_neg_scalar(
    penalty: BasePenalty, i: int, lmbd: float
) -> float:
    """Utility function to compute the value of ``param_limit_neg_scalar`` in a
    :class:`.penalty.BasePenalty` instance when no closed-form is available.

    Parameters
    ----------
    penalty: BasePenalty
        The penalty instance.
    i: int
        Index of the splitting term.
    lmbd: float
        L0-regularization parameter.
    """
    param_slope_scalar = penalty.param_slope_neg_scalar(i, lmbd)
    return penalty.conjugate_subdiff_scalar(i, param_slope_scalar)[0]
