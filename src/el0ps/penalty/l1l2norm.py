import numpy as np
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

    def value(self, x: float) -> float:
        return self.alpha * np.abs(x) + self.beta * x**2

    def conjugate(self, x: float) -> float:
        return np.maximum(np.abs(x) - self.alpha, 0.0) ** 2 / (4.0 * self.beta)

    def param_slope(self, lmbd: float) -> float:
        return self.alpha + np.sqrt(4.0 * self.beta * lmbd)

    def param_limit(self, lmbd: float) -> float:
        return np.sqrt(lmbd / self.beta)

    def param_maxval(self) -> float:
        return np.inf

    def param_maxzer(self) -> float:
        return self.alpha

    def prox(self, x: float, eta: float) -> float:
        return (np.sign(x) / (1.0 + 2.0 * eta * self.beta)) * np.maximum(
            np.abs(x) - eta * self.alpha, 0.0
        )
