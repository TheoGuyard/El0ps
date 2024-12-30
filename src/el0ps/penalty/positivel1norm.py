import numpy as np
from numpy.typing import ArrayLike
from numba import float64

from el0ps.compilation import CompilableClass

from .base import BasePenalty


class PositiveL1norm(CompilableClass, BasePenalty):
    r"""Positive L1-norm penalty function.

    The function is defined as

    .. math:: h(x) = \alpha \|x\|_1 + \text{Indicator}(x \geq 0)

    where :math:`\alpha > 0` and :math:`\text{Indicator}(\cdot)` is the convex
    indicator function.

    Parameters
    ----------
    alpha: float, positive
        L1-norm weight.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __str__(self) -> str:
        return "PositiveL1norm"

    def get_spec(self) -> tuple:
        spec = (("alpha", float64),)
        return spec

    def params_to_dict(self) -> dict:
        return dict(alpha=self.alpha)

    def value_scalar(self, i: int, x: float) -> float:
        return self.alpha * x if x >= 0.0 else np.inf

    def conjugate_scalar(self, i: int, x: float) -> float:
        return 0.0 if x <= self.alpha else np.inf

    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        return np.maximum(0.0, x - eta * self.alpha)

    def subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        if x == 0:
            return [-np.inf, self.alpha]
        elif x > 0:
            s = self.alpha
            return [s, s]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        if x < self.alpha:
            return [0.0, 0.0]
        elif x == self.alpha:
            return [0.0, np.inf]
        else:
            return [np.nan, np.nan]

    def param_slope_pos_scalar(self, i: int, lmbd: float) -> float:
        return self.alpha

    def param_slope_neg_scalar(self, i: int, lmbd: float) -> float:
        return -np.inf

    def param_limit_pos_scalar(self, i: int, lmbd: float) -> float:
        return np.inf

    def param_limit_neg_scalar(self, i: int, lmbd: float) -> float:
        return 0.0
