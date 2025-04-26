import numpy as np
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import SymmetricPenalty


class L1norm(CompilableClass, SymmetricPenalty):
    r"""L1-norm :class:`BasePenalty` penalty function.

    The splitting terms are expressed as

    .. math::
        h_i(x_i) = \alpha|x_i|

    for some :math:`\alpha > 0`.

    Parameters
    ----------
    alpha: float
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

    def subdiff(self, i: int, x: float) -> NDArray:
        if x == 0:
            return [-self.alpha, self.alpha]
        else:
            s = self.alpha * np.sign(x)
            return [s, s]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
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

    def param_bndry(self, i, lmbd):
        return np.inf
