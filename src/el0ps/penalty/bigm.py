import numpy as np
from numba import float64
from .base import ProximablePenalty


class Bigm(ProximablePenalty):
    r"""Big-M penalty function given by

    .. math:: h(x) = 0 \ \ \text{if} \ \ |x| \leq M \ \ \text{and} \ \ h(x) = +\infty \ \ \text{otherwise}

    with :math:`M>0`.

    Parameters
    ----------
    M: float
        Big-M value.
    """  # noqa: E501

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

    def conjugate_scaling_factor(self, x: float) -> float:
        return 1.0

    def param_slope(self, lmbd: float) -> float:
        return lmbd / self.M

    def param_limit(self, lmbd: float) -> float:
        return self.M

    def param_maxval(self) -> float:
        return np.inf

    def param_maxzer(self) -> float:
        return 0.0
