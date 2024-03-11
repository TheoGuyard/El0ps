import numpy as np
from numba import float64
from .base import BasePenalty


class BigmL1norm(BasePenalty):
    r"""Big-M constraint plus L1-norm penalty function given by
    :math:`h(x) = \alpha |x|` when :math:`|x| <= M` and
    :math:`h(x) = +\infty` otherwise, with :math:`M > 0` and
    :math:`\alpha > 0`.

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
        v = np.abs(x) - eta * self.alpha
        return np.sign(x) * np.maximum(np.minimum(v, self.M), 0.0)

    def subdiff(self, x: float) -> tuple[float]:
        if x == 0.0:
            return (-self.alpha, self.alpha)
        elif np.abs(x) < self.M:
            return (self.alpha, self.alpha)
        elif x == -self.M:
            return (-np.inf, -self.alpha)
        elif x == self.M:
            return (self.alpha, np.inf)
        else:
            return ()

    def conjugate_subdiff(self, x: float) -> tuple[float]:
        if np.abs(x) < self.alpha:
            return (0.0, 0.0)
        elif x == self.alpha:
            return (0.0, self.M)
        elif x == -self.alpha:
            return (-self.M, 0.0)
        else:
            s = np.sign(x) * self.M
            return (s, s)

    def param_slope(self, lmbd: float) -> float:
        return (lmbd / self.M) + self.alpha

    def param_limit(self, lmbd: float) -> float:
        return self.M

    def param_maxval(self) -> float:
        return np.inf

    def param_maxdom(self) -> float:
        return np.inf
