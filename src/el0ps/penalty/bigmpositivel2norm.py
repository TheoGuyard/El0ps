import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import BasePenalty, MipPenalty


class BigmPositiveL2norm(CompilableClass, BasePenalty, MipPenalty):
    """Positive big-M plus L2-norm penalty function expressed as

    ``h(x) = sum_{i = 1,...,n} hi(xi)``

    where ``hi(xi) = beta * xi^2`` if ``0 <= xi <= M`` and ``hi(xi) = inf``
    otherwise for some ``M > 0`` and ``beta > 0``.

    Parameters
    ----------
    M: float
        Big-M value.
    beta: float
        L2-norm weight.
    """

    def __init__(self, M: float, beta: float) -> None:
        self.M = M
        self.beta = beta

    def __str__(self) -> str:
        return "BigmPositiveL2norm"

    def get_spec(self) -> tuple:
        spec = (
            ("M", float64),
            ("beta", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M, beta=self.beta)

    def value(self, i: int, x: float) -> float:
        return self.beta * x**2 if (x >= 0.) and (x <= self.M) else np.inf

    def conjugate(self, i: int, x: float) -> float:
        r = np.maximum(np.minimum(x / (2.0 * self.beta), self.M), 0.)
        return x * r - self.beta * r**2

    def prox(self, i: int, x: float, eta: float) -> float:
        v = x / (1.0 + 2.0 * eta * self.beta)
        return np.maximum(np.minimum(v, self.M), 0.)

    def subdiff(self, i: int, x: float) -> NDArray:
        if x == 0.:
            return [-np.inf, 0.]
        elif 0. < x < self.M:
            s = 2.0 * self.beta * x
            return [s, s]
        elif x == self.M:
            return [2.0 * self.beta * x, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        s = np.maximum(np.minimum(x / (2.0 * self.beta), self.M), 0.)
        return [s, s]

    def param_slope_pos(self, i: int, lmbd: float) -> float:
        if lmbd < self.beta * self.M**2:
            return np.sqrt(4.0 * lmbd * self.beta)
        else:
            return (lmbd / self.M) + self.beta * self.M
        
    def param_slope_neg(self, i: int, lmbd: float) -> float:
        return -np.inf

    def param_limit_pos(self, i: int, lmbd: float) -> float:
        if lmbd < self.beta * self.M**2:
            return np.sqrt(lmbd / self.beta)
        else:
            return self.M
        
    def param_limit_neg(self, i: int, lmbd: float) -> float:
        return 0.

    def param_bndry_pos(self, i, lmbd):
        if lmbd < self.beta * self.M**2:
            return np.sqrt(4.0 * lmbd * self.beta)
        else:
            return np.inf
        
    def param_bndry_neg(self, i, lmbd):
        return -np.inf

    def bind_model(self, model: pmo.block, lmbd: float) -> None:

        model.g1_var = pmo.variable_dict()
        for i in model.N:
            model.g1_var[i] = pmo.variable(
                domain=pmo.NonNegativeReals, ub=self.M**2
            )

        model.gpos_con = pmo.constraint_dict()
        model.gneg_con = pmo.constraint_dict()
        model.g1_con = pmo.constraint_dict()
        for i in model.N:
            model.gpos_con[i] = pmo.constraint(
                model.x[i] <= self.M * model.z[i]
            )
            model.gneg_con[i] = pmo.constraint(model.x[i] >= 0.)
            model.g1_con[i] = pmo.conic.rotated_quadratic(
                model.g1_var[i], model.z[i], [model.x[i]]
            )
        model.g_con = pmo.constraint(
            model.g
            >= (
                lmbd * sum(model.z[i] for i in model.N)
                + 2.0 * self.beta * sum(model.g1_var[i] for i in model.N)
            )
        )
