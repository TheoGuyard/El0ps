import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import BasePenalty, MipPenalty


class BigmPositiveL2norm(CompilableClass, BasePenalty, MipPenalty):
    r"""Positive big-M plus L2-norm :class:`BasePenalty` penalty function.

    The splitting terms are expressed as

    .. math::
        h_i(x_i) = \begin{cases}
        \beta x_i^2 & \text{if } 0 \leq x_i \leq M \\
        +\infty & \text{otherwise}
        \end{cases}

    for some :math:`M > 0` and :math:`\beta > 0`.

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
        return self.beta * x**2 if (x >= 0.0) and (x <= self.M) else np.inf

    def conjugate(self, i: int, x: float) -> float:
        r = np.maximum(np.minimum(x / (2.0 * self.beta), self.M), 0.0)
        return x * r - self.beta * r**2

    def prox(self, i: int, x: float, eta: float) -> float:
        v = x / (1.0 + 2.0 * eta * self.beta)
        return np.maximum(np.minimum(v, self.M), 0.0)

    def subdiff(self, i: int, x: float) -> NDArray:
        if x == 0.0:
            return [-np.inf, 0.0]
        elif 0.0 < x < self.M:
            s = 2.0 * self.beta * x
            return [s, s]
        elif x == self.M:
            return [2.0 * self.beta * x, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        s = np.maximum(np.minimum(x / (2.0 * self.beta), self.M), 0.0)
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
        return 0.0

    def param_bndry_pos(self, i, lmbd):
        if lmbd < self.beta * self.M**2:
            return np.sqrt(4.0 * lmbd * self.beta)
        else:
            return np.inf

    def param_bndry_neg(self, i, lmbd):
        return -np.inf

    def bind_model(self, model: pmo.block) -> None:

        model.h1_var = pmo.variable_dict()
        for i in model.N:
            model.h1_var[i] = pmo.variable(
                domain=pmo.NonNegativeReals, ub=self.M**2
            )

        model.hpos_con = pmo.constraint_dict()
        model.hneg_con = pmo.constraint_dict()
        model.h1_con = pmo.constraint_dict()
        for i in model.N:
            model.hpos_con[i] = pmo.constraint(
                model.x[i] <= self.M * model.z[i]
            )
            model.hneg_con[i] = pmo.constraint(model.x[i] >= 0.0)
            model.h1_con[i] = pmo.conic.rotated_quadratic(
                model.h1_var[i], model.z[i], [model.x[i]]
            )
        model.h_con = pmo.constraint(
            model.h >= 2.0 * self.beta * sum(model.h1_var[i] for i in model.N)
        )
