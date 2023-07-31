import numpy as np
from numba import float64
from .base import ProximablePenalty


class BigmL1norm(ProximablePenalty):
    """Big-M constraint plus L1-norm penalty function given by

    .. math:: h(x) = alpha * |x| if |x| <= M and +inf otherwise

    where `M` and `alpha` are positive hyperparameters.

    Parameters
    ----------
    M: float
        Big-M value.
    alpha: float
        L1-norm weight.
    """

    def __init__(self, M: float, alpha: float) -> None:
        self.M = M
        self.alpha = alpha

    def __str__(self) -> str:
        return "BigmL1norm"

    def get_spec(self) -> tuple:
        spec = (
            ("M", float64),
            ("alpha", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M, alpha=self.alpha)

    def value(self, x: float) -> float:
        xabs = np.abs(x)
        return self.alpha * xabs if xabs <= self.M else np.inf

    def conjugate(self, x: float) -> float:
        return self.M * np.maximum(np.abs(x) - self.alpha, 0.0)

    def prox(self, x: float, eta: float) -> float:
        return np.sign(x) * np.maximum(
            np.minimum(np.abs(x) - eta * self.alpha, self.M), 0.0
        )
    
    def conjugate_scaling_factor(self, x: float) -> float:
        return 1.

    def param_slope(self, lmbd: float) -> float:
        return (lmbd / self.M) + self.alpha

    def param_limit(self, lmbd: float) -> float:
        return self.M

    def param_maxval(self) -> float:
        return np.inf
