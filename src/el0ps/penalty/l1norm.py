import numpy as np
from numba import float64
from .base import ProximablePenalty


class L1norm(ProximablePenalty):
    """L1-norm penalty function given by

    .. math:: f(x) = alpha * |x|

    where `alpha` is a positive hyperparameter.

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

    def value(self, x: float) -> float:
        return self.alpha * np.abs(x)

    def conjugate(self, x: float) -> float:
        return 0.0 if np.abs(x) <= self.alpha else np.inf

    def prox(self, x: float, eta: float) -> float:
        return np.sign(x) * np.maximum(0.0, np.abs(x) - eta * self.alpha)

    def param_slope(self, lmbd: float) -> float:
        return self.alpha

    def param_limit(self, lmbd: float) -> float:
        return np.inf

    def param_maxval(self) -> float:
        return 0.0
