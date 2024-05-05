import numpy as np
import pyomo.kernel as pmo
from numpy.typing import ArrayLike
from numba import float64
from .base import MipPenalty


class BigmL2norm(MipPenalty):
    r"""Big-M constraint plus L2-norm penalty function given by
    :math:`h(x) = \alpha x^2` when :math:`|x| <= M` and
    :math:`h(x) = +\infty` otherwise, with :math:`M > 0` and
    :math:`\alpha > 0`.

    Parameters
    ----------
    M: float
        Big-M value.
    alpha: float
        L2-norm weight.
    """

    def __init__(self, M: float, alpha: float) -> None:
        self.M = M
        self.alpha = alpha

    def __str__(self) -> str:
        return "BigmL2norm"

    def get_spec(self) -> tuple:
        spec = (
            ("M", float64),
            ("alpha", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M, alpha=self.alpha)

    def value(self, i: int, x: float) -> float:
        return self.alpha * x**2 if np.abs(x) <= self.M else np.inf

    def conjugate(self, i: int, x: float) -> float:
        r = np.maximum(np.minimum(x / (2.0 * self.alpha), self.M), -self.M)
        return x * r - self.alpha * r**2

    def prox(self, i: int, x: float, eta: float) -> float:
        v = x / (1.0 + 2.0 * eta * self.alpha)
        return np.maximum(np.minimum(v, self.M), -self.M)

    def subdiff(self, i: int, x: float) -> ArrayLike:
        if np.abs(x) < self.M:
            s = 2.0 * self.alpha * x
            return [s, s]
        elif x == -self.M:
            return [-np.inf, 2.0 * self.alpha * x]
        elif x == self.M:
            return [2.0 * self.alpha * x, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff(self, i: int, x: float) -> ArrayLike:
        s = np.maximum(np.minimum(x / (2.0 * self.alpha), self.M), -self.M)
        return [s, s]

    def param_slope(self, i: int, lmbd: float) -> float:
        if lmbd < self.alpha * self.M**2:
            return np.sqrt(4.0 * lmbd * self.alpha)
        else:
            return (lmbd / self.M) + self.alpha * self.M

    def param_limit(self, i: int, lmbd: float) -> float:
        if lmbd < self.alpha * self.M**2:
            return np.sqrt(lmbd / self.alpha)
        else:
            return self.M

    def param_maxval(self, i: int) -> float:
        return np.inf

    def param_maxdom(self, i: int) -> float:
        return np.inf

    def bind_model(self, model: pmo.block, lmbd: float) -> None:

        model.g1_var = pmo.variable_dict()
        for i in model.N:
            model.g1_var[i] = pmo.variable(domain=pmo.NonNegativeReals)

        model.gpos_con = pmo.constraint_dict()
        model.gneg_con = pmo.constraint_dict()
        model.g1_con = pmo.constraint_dict()
        for i in model.N:
            model.gpos_con[i] = pmo.constraint(
                model.x[i] <= self.M * model.z[i]
            )
            model.gneg_con[i] = pmo.constraint(
                model.x[i] >= -self.M * model.z[i]
            )
            model.g1_con[i] = pmo.constraint(
                model.x[i] ** 2 <= model.g1_var[i] * model.z[i]
            )
        model.g_con = pmo.constraint(
            model.g
            >= (
                lmbd * sum(model.z[i] for i in model.N)
                + self.alpha * sum(model.g1_var[i] for i in model.N)
            )
        )
