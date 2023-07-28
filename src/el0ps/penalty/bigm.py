import numpy as np
from numba import float64
from .base import ProximablePenalty


class Bigm(ProximablePenalty):
    """Big-M penalty function given by

    .. math:: h(x) = 0 if |x| <= M and +inf otherwise

    where `M` is a positive hyperparameter.

    Parameters
    ----------
    M: float
        Big-M value.
    """

    def __init__(self, M: float) -> None:
        self.M = M

    def __str__(self) -> str:
        return "Bigm"

    def get_spec(self) -> tuple:
        spec = (("M", float64),)
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M)

    def value(self, x: float) -> float:
        return 0.0 if np.abs(x) <= self.M else np.inf

    def conjugate(self, x: float) -> float:
        return self.M * np.abs(x)

    def prox(self, x: float, eta: float) -> float:
        return np.maximum(np.minimum(x, self.M), -self.M)

    def param_slope(self, lmbd: float) -> float:
        return lmbd / self.M

    def param_limit(self, lmbd: float) -> float:
        return self.M

    def param_maxval(self) -> float:
        return np.inf
