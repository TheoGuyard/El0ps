import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass

from .base import BasePenalty, MipPenalty


class PositiveL2norm(CompilableClass, BasePenalty, MipPenalty):
    r"""Positive L2-norm :class:`BasePenalty` penalty function.

    The splitting terms are expressed as

    .. math::
        h_i(x_i) = \begin{cases}
        \beta x_i^2 & \text{if } x_i \geq 0 \\
        +\infty & \text{otherwise}
        \end{cases}

    for some :math:`\beta > 0`.

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

    def bind_model(self, model: pmo.block) -> None:
        model.h1_var = pmo.variable_dict()
        for i in model.N:
            model.h1_var[i] = pmo.variable(domain=pmo.NonNegativeReals)

        model.h1_con = pmo.constraint_dict()
        model.hpos_con = pmo.constraint_dict()
        for i in model.N:
            model.h1_con[i] = pmo.conic.rotated_quadratic(
                model.h1_var[i], model.z[i], [model.x[i]]
            )
            model.hpos_con[i] = pmo.constraint(model.x[i] >= 0.0)
        model.h_con = pmo.constraint(
            model.h >= 2.0 * self.beta * sum(model.h1_var[i] for i in model.N)
        )
