import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import BasePenalty, MipPenalty


class Bounds(CompilableClass, BasePenalty, MipPenalty):
    """Bound constraint penalty function expressed as

    ``h(x) = sum_{i = 1,...,n} hi(xi)``

    where ``hi(xi) = 0`` if ``x_lb[i] <= xi <= x_ub[i]`` and ``hi(xi) = inf``
    otherwise for vectors ``x_lb <= 0`` and ``x_ub >= 0`` of lower and upper
    bounds.

    Parameters
    ----------
    x_lb: NDArray
        Vector or lower bounds.
    x_ub: NDArray
        Vector of upper bounds.
    """

    def __init__(self, x_lb: NDArray, x_ub: NDArray) -> None:
        self.x_lb = x_lb
        self.x_ub = x_ub

    def __str__(self) -> str:
        return "Bounds"

    def get_spec(self) -> tuple:
        spec = (("x_lb", float64[:]), ("x_ub", float64[:]))
        return spec

    def params_to_dict(self) -> dict:
        return dict(x_lb=self.x_lb, x_ub=self.x_ub)

    def value(self, i: int, x: float) -> float:
        return 0.0 if (self.x_lb[i] <= x <= self.x_ub[i]) else np.inf

    def conjugate(self, i: int, x: float) -> float:
        if x >= 0.0:
            return self.x_ub[i] * x
        else:
            return self.x_lb[i] * x

    def prox(self, i: int, x: float, eta: float) -> float:
        return np.maximum(np.minimum(x, self.x_ub[i]), self.x_lb[i])

    def subdiff(self, i: int, x: float) -> NDArray:
        if self.x_lb[i] < x < self.x_ub[i]:
            return [0.0, 0.0]
        elif x == self.x_lb[i]:
            return [-np.inf, 0.0]
        elif x == self.x_ub[i]:
            return [0.0, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        if x == 0.0:
            return [self.x_lb[i], self.x_ub[i]]
        elif x > 0.0:
            s = self.x_ub[i]
            return [s, s]
        else:
            s = self.x_lb[i]
            return [s, s]

    def param_slope_pos(self, i: int, lmbd: float) -> float:
        if self.x_ub[i] == 0.0:
            return np.inf
        return lmbd / self.x_ub[i]

    def param_slope_neg(self, i: int, lmbd: float) -> float:
        if self.x_lb[i] == 0.0:
            return -np.inf
        return lmbd / self.x_lb[i]

    def param_limit_pos(self, i: int, lmbd: float) -> float:
        return self.x_ub[i]

    def param_limit_neg(self, i: int, lmbd: float) -> float:
        return self.x_lb[i]

    def param_bndry_pos(self, i, lmbd):
        return np.inf

    def param_bndry_neg(self, i, lmbd):
        return -np.inf

    def bind_model(self, model: pmo.block, lmbd: float) -> None:
        model.gpos_con = pmo.constraint_dict()
        model.gneg_con = pmo.constraint_dict()
        for i in model.N:
            model.gpos_con[i] = pmo.constraint(
                model.x[i] <= self.x_ub[i] * model.z[i]
            )
            model.gneg_con[i] = pmo.constraint(
                model.x[i] >= self.x_lb[i] * model.z[i]
            )
        model.g_con = pmo.constraint(
            model.g >= lmbd * sum(model.z[i] for i in model.N)
        )
