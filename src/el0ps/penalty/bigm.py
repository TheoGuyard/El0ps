import numpy as np
from numba import float64
from .base import BasePenalty


class Bigm(BasePenalty):
    r"""Big-M penalty function given by :math:`h(x) = 0` when :math:`|x| <= M`
    and :math:`h(x) = +\infty` otherwise, with :math:`M > 0`.

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

    def subdiff(self, x: float) -> tuple[float]:
        if np.abs(x) < self.M:
            return (0.0, 0.0)
        elif x == -self.M:
            return (-np.inf, 0.0)
        elif x == self.M:
            return (0.0, np.inf)
        else:
            return ()

    def conjugate_subdiff(self, x: float) -> tuple[float]:
        if x == 0.0:
            return (-self.M, self.M)
        else:
            s = np.sign(x) * self.M
            return (s, s)

    def param_slope(self, lmbd: float) -> float:
        return lmbd / self.M

    def param_limit(self, lmbd: float) -> float:
        return self.M

    def param_maxval(self) -> float:
        return np.inf

    def param_maxdom(self) -> float:
        return np.inf
