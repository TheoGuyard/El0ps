"""Miscellaneous utilities."""

import numpy as np
from functools import lru_cache
from numba.experimental import jitclass
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
    v = A.T @ datafit.gradient(w)
    r = np.max(np.abs(v))
    if penalty.conjugate(r) == 0.0:
        lmbd_max = 0.0
    elif penalty.conjugate(r) < np.inf:
        lmbd_max = penalty.conjugate(r)
    else:
        lmbd_max = penalty.param_maxval()

    return lmbd_max


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
