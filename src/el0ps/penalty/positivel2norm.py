import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass

from .base import BasePenalty, MipPenalty


class PositiveL2norm(CompilableClass, BasePenalty, MipPenalty):
    """Positive L2-norm penalty function expressed as 

    ``h(x) = sum_{i = 1,...,n} hi(xi)``

    where ``hi(xi) = beta * xi^2`` if ``xi >= 0.`` and ``hi(xi) = inf``
    otherwise for some ``beta > 0``.

    Parameters
    ----------
    beta: float
        L2-norm weight.
    """

    def __init__(self, beta: float) -> None:
        self.beta = beta

    def __str__(self) -> str:
        return "PositiveL2norm"

    def get_spec(self) -> tuple:
        spec = (("beta", float64),)
        return spec

    def params_to_dict(self) -> dict:
        return dict(beta=self.beta)

    def value(self, i: int, x: float) -> float:
        return self.beta * x**2 if x >= 0.0 else np.inf

    def conjugate(self, i: int, x: float) -> float:
        return np.maximum(x, 0.0) ** 2 / (4.0 * self.beta)

    def prox(self, i: int, x: float, eta: float) -> float:
        return np.maximum(x, 0.0) / (1.0 + 2.0 * self.beta * eta)

    def subdiff(self, i: int, x: float) -> NDArray:
        if x == 0.0:
            return [-np.inf, 0.0]
        elif x > 0.0:
            s = 2.0 * self.beta * x
            return [s, s]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        if x > 0.0:
            s = x / (2.0 * self.beta)
            return [s, s]
        else:
            s = 0.0
            return [s, s]

    def param_slope_pos(self, i: int, lmbd: float) -> float:
        return 2.0 * np.sqrt(lmbd * self.beta)

    def param_slope_neg(self, i: int, lmbd: float) -> float:
        return -np.inf

    def param_limit_pos(self, i: int, lmbd: float) -> float:
        return np.sqrt(lmbd / self.beta)

    def param_limit_neg(self, i: int, lmbd: float) -> float:
        return 0.0
    
    def param_bndry_pos(self, i, lmbd):
        return 2.0 * np.sqrt(lmbd * self.beta)
    
    def param_bndry_neg(self, i, lmbd):
        return -np.inf

    def bind_model(self, model: pmo.block, lmbd: float) -> None:
        model.g1_var = pmo.variable_dict()
        for i in model.N:
            model.g1_var[i] = pmo.variable(domain=pmo.NonNegativeReals)

        model.g1_con = pmo.constraint_dict()
        model.gpos_con = pmo.constraint_dict()
        for i in model.N:
            model.g1_con[i] = pmo.conic.rotated_quadratic(
                model.g1_var[i], model.z[i], [model.x[i]]
            )
            model.gpos_con[i] = pmo.constraint(model.x[i] >= 0.0)
        model.g_con = pmo.constraint(
            model.g
            >= (
                lmbd * sum(model.z[i] for i in model.N)
                + 2.0 * self.beta * sum(model.g1_var[i] for i in model.N)
            )
        )
