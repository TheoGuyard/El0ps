"""Miscellaneous utilities."""

import numpy as np
from functools import lru_cache
from numba.experimental import jitclass
from numpy.typing import ArrayLike
from el0ps.datafits import SmoothDatafit
from el0ps.penalties import BasePenalty


def compute_lmbd_max(
    datafit: SmoothDatafit, penalty: BasePenalty, A: ArrayLike
) -> float:
    """Return a value of ``lmbd`` above which the all-zero vector is always a
    solution of the L0-penalized problem.

    Parameters
    ----------
    datafit: SmoothDatafit
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
    p = penalty.conjugate_scalar(i, v[i])
    if p < np.inf:
        lmbd_max = p
    else:
        lmbd_max = penalty.param_maxval_scalar(i)
    return lmbd_max


def compute_param_slope_scalar(
    penalty: BasePenalty,
    i: int,
    lmbd: float,
    tol: float = 1e-4,
    maxit: int = 100,
) -> float:
    """Utility function to compute the value of ``param_slope`` in a
    :class:`.penalties.BasePenalty` instance when no closed-form is available.

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
    if penalty.param_maxval_scalar(i) < lmbd:
        return penalty.param_maxdom_scalar(i)
    a = 0.0
    b = 1.0
    while penalty.conjugate_scalar(i, b) < lmbd:
        b *= 2.0
    for _ in range(maxit):
        c = (a + b) / 2.0
        fa = penalty.conjugate_scalar(i, a) - lmbd
        fc = penalty.conjugate_scalar(i, c) - lmbd
        if (-tol <= fc <= tol) or (b - a < 0.5 * tol):
            return c
        if fc * fa >= 0.0:
            a = c
        else:
            b = c
    return c


def compute_param_limit_scalar(
    penalty: BasePenalty, i: int, lmbd: float
) -> float:
    """Utility function to compute the value of ``param_limit_scalar`` in a
    :class:`.penalties.BasePenalty` instance when no closed-form is available.

    Parameters
    ----------
    penalty: BasePenalty
        The penalty instance.
    i: int
        Index of the splitting term.
    lmbd: float
        L0-regularization parameter.
    """
    param_slope_scalar = penalty.param_slope_scalar(i, lmbd)
    return np.max(penalty.conjugate_subdiff_scalar(i, param_slope_scalar))


def compute_param_maxval_scalar(
    penalty: BasePenalty,
    i: int,
    tol: float = 1e-4,
    maxit: int = 100,
) -> float:
    """Utility function to compute the value of ``param_maxval_scalar`` in a
    :class:`.penalties.BasePenalty` instance when no closed-form is available.

    Parameters
    ----------
    penalty: BasePenalty
        The penalty instance.
    i: int
        Index of the splitting term.
    tol: float = 1e-4
        Bisection tolerance.
    maxit: int = 100
        Maximum number of bisection iterations.
    """
    a = 0.0
    b = 1.0
    while penalty.conjugate_scalar(i, b) < np.inf:
        b *= 2.0
    for _ in range(maxit):
        if b - a < tol:
            return penalty.conjugate_scalar(i, a)
        c = (a + b) / 2.0
        fc = penalty.conjugate_scalar(i, c)
        if fc < np.inf:
            a = c
        else:
            b = c
    return penalty.conjugate_scalar(i, a)


def compute_param_maxdom_scalar(
    penalty: BasePenalty,
    i: int,
    tol: float = 1e-4,
    maxit: int = 100,
) -> float:
    """Utility function to compute the value of ``param_maxdom_scalar`` in a
    :class:`.penalties.BasePenalty` instance when no closed-form is available.

    Parameters
    ----------
    penalty: BasePenalty
        The penalty instance.
    i: int
        Index of the splitting term.
    tol: float = 1e-4
        Bisection tolerance.
    maxit: int = 100
        Maximum number of bisection iterations.
    """
    a = 0.0
    b = 1.0
    while penalty.conjugate_scalar(i, b) < np.inf:
        b *= 2.0
    for _ in range(maxit):
        c = (a + b) / 2.0
        if b - a < tol:
            return c
        fc = penalty.conjugate_scalar(i, c)
        if fc < np.inf:
            a = c
        else:
            b = c
    return c


@lru_cache()
def compiled_clone(instance):
    """Compile a class instance to a ``jitclass``. Credits: ``skglm`` package.

    Parameters
    ----------
    instance: object
        Instance to compile.

    Returns
    -------
    compiled_instance: jitclass
        Compiled instance.
    """
    cls = instance.__class__
    spec = instance.get_spec()
    params = instance.params_to_dict()
    return jitclass(spec)(cls)(**params)
