import numpy as np
from numpy.typing import ArrayLike
from numba import float64
from .base import BasePenalty


class L2norm(BasePenalty):
    r"""L2-norm penalty function given by :math:`h(x) = \alpha x^2`, with
    :math:`\alpha>0`.

    Parameters
    ----------
    alpha: float, positive
        L2-norm weight.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __str__(self) -> str:
        return "L2norm"

    def get_spec(self) -> tuple:
        spec = (("alpha", float64),)
        return spec

    def params_to_dict(self) -> dict:
        return dict(alpha=self.alpha)

    def value(self, i: int, x: float) -> float:
        return self.alpha * x**2

    def conjugate(self, i: int, x: float) -> float:
        return x**2 / (4.0 * self.alpha)

    def prox(self, i: int, x: float, eta: float) -> float:
        return x / (1.0 + 2.0 * eta * self.alpha)

    def subdiff(self, i: int, x: float) -> ArrayLike:
        s = 2.0 * self.alpha * x
        return [s, s]

    def conjugate_subdiff(self, i: int, x: float) -> ArrayLike:
        s = x / (2.0 * self.alpha)
        return [s, s]

    def param_slope(self, i: int, lmbd: float) -> float:
        return 2.0 * np.sqrt(lmbd * self.alpha)

    def param_limit(self, i: int, lmbd: float) -> float:
        return np.sqrt(lmbd / self.alpha)

    def param_maxval(self, i: int) -> float:
        return np.inf

    def param_maxdom(self, i: int) -> float:
        return np.inf
