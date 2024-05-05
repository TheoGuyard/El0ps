import numpy as np
from numpy.typing import ArrayLike
from numba import float64
from .base import BasePenalty


class L1norm(BasePenalty):
    r"""L1-norm penalty function given by :math:`h(x) = \alpha |x|`, with
    :math:`\alpha>0`.

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

    def value(self, i: int, x: float) -> float:
        return self.alpha * np.abs(x)

    def conjugate(self, i: int, x: float) -> float:
        return 0.0 if np.abs(x) <= self.alpha else np.inf

    def prox(self, i: int, x: float, eta: float) -> float:
        return np.sign(x) * np.maximum(0.0, np.abs(x) - eta * self.alpha)

    def subdiff(self, i: int, x: float) -> ArrayLike:
        if x == 0:
            return [-self.alpha, self.alpha]
        else:
            s = self.alpha * np.sign(x)
            return [s, s]

    def conjugate_subdiff(self, i: int, x: float) -> ArrayLike:
        if np.abs(x) < self.alpha:
            return [0.0, 0.0]
        elif x == -self.alpha:
            return [-np.inf, 0.0]
        elif x == self.alpha:
            return [0.0, np.inf]
        else:
            return [np.nan, np.nan]

    def param_slope(self, i: int, lmbd: float) -> float:
        return self.alpha

    def param_limit(self, i: int, lmbd: float) -> float:
        return np.inf

    def param_maxval(self, i: int) -> float:
        return 0.0

    def param_maxdom(self, i: int) -> float:
        return self.alpha
