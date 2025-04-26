import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import BasePenalty, MipPenalty


class Bounds(CompilableClass, BasePenalty, MipPenalty):
    r"""Bound-constraint :class:`BasePenalty` penalty function.

    The splitting terms are expressed as

    .. math::
        h_i(x_i) = \begin{cases}
        0 & \text{if } x^{\text{lb}}_i \leq x_i \leq x^{\text{ub}}_i \\
        +\infty & \text{otherwise}
        \end{cases}

    for some :math:`x^{\text{lb}}_i < 0` and :math:`x^{\text{ub}}_i > 0`.

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

    def bind_model(self, model: pmo.block) -> None:
        model.hpos_con = pmo.constraint_dict()
        model.hneg_con = pmo.constraint_dict()
        for i in model.N:
            model.hpos_con[i] = pmo.constraint(
                model.x[i] <= self.x_ub[i] * model.z[i]
            )
            model.hneg_con[i] = pmo.constraint(
                model.x[i] >= self.x_lb[i] * model.z[i]
            )
        model.h_con = pmo.constraint(model.h >= 0.0)
