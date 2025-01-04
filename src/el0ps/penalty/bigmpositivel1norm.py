import numpy as np
from numpy.typing import ArrayLike
from numba import float64

from el0ps.compilation import CompilableClass

from .base import BasePenalty


class BigmPositiveL1norm(CompilableClass, BasePenalty):
    r"""Positive L1-norm penalty function with Big-M constraint.

    The function is defined as

    .. math:: h(x) = \alpha \|x\|_1 + \text{Indicator}(0 \leq x \leq M)

    where :math:`\alpha > 0`, :math:`\M > 0`, and
    :math:`\text{Indicator}(\cdot)` is the convex indicator function.

    Parameters
    ----------
    M: float, positive
        Big-M value.
    alpha: float, positive
        L1-norm weight.
    """

    def __init__(self, M: float, alpha: float) -> None:
        self.M = M
        self.alpha = alpha

    def __str__(self) -> str:
        return "BigmPositiveL1norm"

    def get_spec(self) -> tuple:
        spec = (
            ("M", float64),
            ("alpha", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M, alpha=self.alpha)

    def value_scalar(self, i: int, x: float) -> float:
        return self.alpha * x if (x >= 0.0) and (x <= self.M) else np.inf

    def conjugate_scalar(self, i: int, x: float) -> float:
        return self.M * np.maximum(x - self.alpha, 0.0)

    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        return np.maximum(0.0, np.minimum(x - eta * self.alpha, self.M))

    def subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        if x == 0.0:
            return [-np.inf, self.alpha]
        elif 0.0 < x < self.M:
            s = self.alpha
            return [s, s]
        elif x == self.M:
            return [self.alpha, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        if x < self.alpha:
            return [0.0, 0.0]
        elif x == self.alpha:
            return [0.0, self.M]
        else:
            s = np.sign(x) * self.M
            return [s, s]

    def param_slope_pos_scalar(self, i: int, lmbd: float) -> float:
        return (lmbd / self.M) + self.alpha

    def param_slope_neg_scalar(self, i: int, lmbd: float) -> float:
        return -np.inf

    def param_limit_pos_scalar(self, i: int, lmbd: float) -> float:
        return self.M

    def param_limit_neg_scalar(self, i: int, lmbd: float) -> float:
        return 0.0
