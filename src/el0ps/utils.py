"""Miscellaneous utilities."""

import numpy as np
from numpy.typing import NDArray

from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty


def compute_lmbd_max(
    datafit: BaseDatafit, penalty: BasePenalty, A: NDArray
) -> float:
    """Return a value ``lmbd_max`` such that the all-zero vector is a solution
    of the problem

    ``min_{x in R^n} f(Ax) + lmbd * ||x||_0 + h(x)``

    whenever ``lmbd >= lmbd_max`` for any datafit function ``f`` and penalty
    function ``h``.

    Parameters
    ----------
    datafit: BaseDatafit
        Problem datafit function.
    penalty: BasePenalty
        Problem penalty function.
    A: ArrayLike
        Problem matrix.

    Returns
    -------
    lmbd_max: float
        The value ``lmbd_max`` ensuring an all-zero solution.
    """
    w = np.zeros(A.shape[0])
    v = np.abs(A.T @ datafit.gradient(w))
    i = np.argmax(v)
    return penalty.conjugate(i, v[i])
