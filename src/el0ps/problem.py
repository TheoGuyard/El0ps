"""Base class for L0-penalized problems."""

import numpy as np
from functools import lru_cache
from typing import Union
from numba.experimental import jitclass
from numpy.typing import NDArray
from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty


class Problem:
    r"""L0-penalized problem.

    The problem is defined as

    .. math:: \textstyle\min_{x} f(Ax) + \lambda \|x\|_0 + h(x)

    where :math:`f(\cdot)` is a datafit function, :math:`A` is a linear
    operator, :math:`\lambda>0` is the L0-regularization weight and
    :math:`h(\cdot)` is a penalty function.

    Parameters
    ----------
    datafit: BaseDatafit
        Datafit function.
    penalty: BasePenalty
        Penalty function.
    A: NDArray
        Linear operator.
    lmbd: float, positive
        L0-regularization weight.

    Attributes
    ----------
    m: int
        Number of rows in ``A``.
    n: int
        Number of columns in ``A``.
    """

    def __init__(
        self,
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: NDArray,
        lmbd: float,
    ) -> None:
        if not isinstance(datafit, BaseDatafit):
            raise ValueError(
                "Parameter `datafit` must derive from `BaseDatafit`."
            )
        if not isinstance(penalty, BasePenalty):
            raise ValueError(
                "Parameter `penalty` must derive from `BasePenalty`."
            )
        if not isinstance(A, np.ndarray):
            raise ValueError("Parameter `A` must derive from `np.ndarray`.")
        if not isinstance(lmbd, float):
            raise ValueError("Parameter `lmbd` must derive from `float`.")
        if A.ndim != 2:
            raise ValueError("Parameter `A` must be a two-dimensional array.")
        if not A.flags.f_contiguous:
            A = np.array(A, order="F")
        if lmbd < 0.0:
            raise ValueError("Parameter `lmbd` must be positive.")

        self.datafit = compiled_clone(datafit)
        self.penalty = compiled_clone(penalty)
        self.A = A
        self.lmbd = lmbd
        self.m, self.n = A.shape

    def __str__(self) -> str:
        s = ""
        s += "L0-penalized problem\n"
        s += "  Datafit : {}\n".format(self.datafit)
        s += "  Penalty : {}\n".format(self.penalty)
        s += "  Dims    : {} x {}\n".format(self.m, self.n)
        s += "  Lambda  : {:.4e}\n".format(self.lmbd)
        return s

    def value(self, x: NDArray, Ax: Union[NDArray, None] = None) -> float:
        """Value of the objective function of the problem at ``x``.

        Parameters
        ----------
        x: NDArray
            Vector at which the objective function evaluated.
        Ax: Union[NDArray, None] = None
            Value of ``A @ x`` if it is already computed, allows to save
            computations.

        Returns
        -------
        value: float
            The value of the problem objective function at ``x``.
        """
        if Ax is None:
            Ax = self.A @ x
        return (
            self.datafit.value(Ax)
            + self.lmbd * np.linalg.norm(x, 0)
            + sum(self.penalty.value(xi) for xi in x)
        )


def compute_lmbd_max(
    datafit: BaseDatafit, penalty: BasePenalty, A: NDArray
) -> float:
    """Return a value of ``lmbd`` above which the all-zero vector is always a
    solution of the corresponding :class:`.Problem`.

    Parameters
    ----------
    datafit: BaseDatafit
        Datafit function.
    penalty: BasePenalty
        Penalty function.
    A: NDArray
        Linear operator.

    Returns
    -------
    lmbd_max: float
        A value of ``lmbd`` above which the all-zero vector is always a
        solution of the corresponding :class:`.Problem`.
    """

    if not isinstance(A, np.ndarray):
        raise ValueError("Parameter `A` must be a `np.ndarray`.")
    if A.ndim != 2:
        raise ValueError("Parameter `A` must be a two-dimensional array.")
    if A.size == 0:
        raise ValueError("Parameter `A` is empty.")

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
