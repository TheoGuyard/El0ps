import numpy as np
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass

from .base import BasePenalty


class PositiveL1norm(CompilableClass, BasePenalty):
    r"""Positive L1-norm :class:`BasePenalty` penalty function.

    The splitting terms are expressed as

    .. math::
        h_i(x_i) = \begin{cases}
        \alpha x_i & \text{if } x_i \geq 0 \\
        +\infty & \text{otherwise}
        \end{cases}

    for some :math:`\alpha > 0`.

    Parameters
    ----------
    alpha: float
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

    def value(self, i: int, x: float) -> float:
        return self.alpha * x if x >= 0.0 else np.inf

    def conjugate(self, i: int, x: float) -> float:
        return 0.0 if x <= self.alpha else np.inf

    def prox(self, i: int, x: float, eta: float) -> float:
        return np.maximum(0.0, x - eta * self.alpha)

    def subdiff(self, i: int, x: float) -> NDArray:
        if x == 0:
            return [-np.inf, self.alpha]
        elif x > 0:
            s = self.alpha
            return [s, s]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        if x < self.alpha:
            return [0.0, 0.0]
        elif x == self.alpha:
            return [0.0, np.inf]
        else:
            return [np.nan, np.nan]

    def param_slope_pos(self, i: int, lmbd: float) -> float:
        return self.alpha

    def param_slope_neg(self, i: int, lmbd: float) -> float:
        return -np.inf

    def param_limit_pos(self, i: int, lmbd: float) -> float:
        return np.inf

    def param_limit_neg(self, i: int, lmbd: float) -> float:
        return 0.0

    def param_bndry_pos(self, i, lmbd):
        return self.alpha

    def param_bndry_neg(self, i, lmbd):
        return -np.inf
