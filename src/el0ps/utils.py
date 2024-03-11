"""Miscellaneous utilities."""

import numpy as np
from functools import lru_cache
from numba.experimental import jitclass
from numpy.typing import ArrayLike
from el0ps.datafit import SmoothDatafit
from el0ps.penalty import BasePenalty


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
    v = A.T @ datafit.gradient(w)
    r = np.max(np.abs(v))
    if penalty.conjugate(r) == 0.0:
        lmbd_max = 0.0
    elif penalty.conjugate(r) < np.inf:
        lmbd_max = penalty.conjugate(r)
    else:
        lmbd_max = penalty.param_maxval()

    return lmbd_max


def compute_param_slope(
    penalty: BasePenalty, lmbd: float, tol: float = 1e-4, maxit: int = 100
) -> float:
    """Utility function to compute the value of `param_slope` in a
    ``BasePenalty`` instance when no closed-form is available.

    Parameters
    ----------
    tol: float = 1e-4
        Bisection tolerance.
    maxit: int = 100
        Maximum number of bisection iterations.
    """
    if penalty.param_maxval() < lmbd:
        return penalty.param_maxdom()
    a = 0.0
    b = 1.0
    while penalty.conjugate(b) < lmbd:
        b *= 2.0
    for _ in range(maxit):
        c = (a + b) / 2.0
        fa = penalty.conjugate(a) - lmbd
        fc = penalty.conjugate(c) - lmbd
        if (-tol <= fc <= tol) or (b - a < 0.5 * tol):
            return c
        if fc * fa >= 0.0:
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
