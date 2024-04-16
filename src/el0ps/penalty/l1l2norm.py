import numpy as np
from numpy.typing import ArrayLike
from numba import float64
from .base import BasePenalty


class L1L2norm(BasePenalty):
    r"""L1L2-norm penalty function given by
    :math:`h(x) = \alpha |x| + \beta x^2`, with :math:`\alpha>0` and
    :math:`\beta>0`.

    Parameters
    ----------
    alpha: float, positive
        L1-norm weight.
    beta: float, positive
        L2-norm weight.
    """

    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta

    def __str__(self) -> str:
        return "L1L2norm"

    def get_spec(self) -> tuple:
        spec = (
            ("alpha", float64),
            ("beta", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(alpha=self.alpha, beta=self.beta)

    def value(self, i: int, x: float) -> float:
        return self.alpha * np.abs(x) + self.beta * x**2

    def conjugate(self, i: int, x: float) -> float:
        return np.maximum(np.abs(x) - self.alpha, 0.0) ** 2 / (4.0 * self.beta)

    def prox(self, i: int, x: float, eta: float) -> float:
        v = np.sign(x) / (1.0 + 2.0 * eta * self.beta)
        return v * np.maximum(np.abs(x) - eta * self.alpha, 0.0)

    def subdiff(self, i: int, x: float) -> ArrayLike:
        if x == 0:
            return [-self.alpha, self.alpha]
        else:
            s = self.alpha * np.sign(x) + 2.0 * self.beta * x
            return [s, s]

    def conjugate_subdiff(self, i: int, x: float) -> ArrayLike:
        if np.abs(x) < self.alpha:
            return [0.0, 0.0]
        elif x == -self.alpha:
            return [(x + self.alpha) / (2.0 * self.beta), 0.0]
        elif x == self.alpha:
            return [0.0, (x - self.alpha) / (2.0 * self.beta)]
        else:
            s = (x - self.alpha * np.sign(x)) / (2.0 * self.beta)
            return [s, s]

    def param_slope(self, i: int, lmbd: float) -> float:
        return self.alpha + np.sqrt(4.0 * self.beta * lmbd)

    def param_limit(self, i: int, lmbd: float) -> float:
        return np.sqrt(lmbd / self.beta)

    def param_maxval(self, i: int) -> float:
        return np.inf

    def param_maxdom(self, i: int) -> float:
        return np.inf
