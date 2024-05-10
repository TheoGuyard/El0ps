import numpy as np
from numpy.typing import ArrayLike
from numba import float64
from .base import BasePenalty, MipPenalty


class L1norm(BasePenalty, MipPenalty):
    r"""L1-norm penalty function.

    The function is defined as

    .. math:: h(x) = \alpha \|x\|_1

    where :math:`\alpha > 0`.

    Parameters
    ----------
    alpha: float, positive
        L1-norm weight.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __str__(self) -> str:
        return "L1norm"

    def get_spec(self) -> tuple:
        spec = (("alpha", float64),)
        return spec

    def params_to_dict(self) -> dict:
        return dict(alpha=self.alpha)

    def value_scalar(self, i: int, x: float) -> float:
        return self.alpha * np.abs(x)

    def conjugate_scalar(self, i: int, x: float) -> float:
        return 0.0 if np.abs(x) <= self.alpha else np.inf

    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        return np.sign(x) * np.maximum(0.0, np.abs(x) - eta * self.alpha)

    def subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        if x == 0:
            return [-self.alpha, self.alpha]
        else:
            s = self.alpha * np.sign(x)
            return [s, s]

    def conjugate_subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        if np.abs(x) < self.alpha:
            return [0.0, 0.0]
        elif x == -self.alpha:
            return [-np.inf, 0.0]
        elif x == self.alpha:
            return [0.0, np.inf]
        else:
            return [np.nan, np.nan]

    def param_slope_scalar(self, i: int, lmbd: float) -> float:
        return self.alpha

    def param_limit_scalar(self, i: int, lmbd: float) -> float:
        return np.inf

    def param_maxval_scalar(self, i: int) -> float:
        return 0.0

    def param_maxdom_scalar(self, i: int) -> float:
        return self.alpha
