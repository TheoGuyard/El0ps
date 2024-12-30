import numpy as np
import pyomo.kernel as pmo
from numpy.typing import ArrayLike
from numba import float64

from el0ps.compilation import CompilableClass

from .base import BasePenalty, MipPenalty


class PositiveL2norm(CompilableClass, BasePenalty, MipPenalty):
    r"""Positive L2-norm penalty function.

    The function is defined as

    .. math:: h(x) = \beta \|x\|_2^2 + \text{Indicator}(x \geq 0)

    where :math:`\ beta > 0` and :math:`\text{Indicator}(\cdot)` is the convex
    indicator function.

    Parameters
    ----------
    beta: float, positive
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

    def value_scalar(self, i: int, x: float) -> float:
        return self.beta * x**2 if x >= 0.0 else np.inf

    def conjugate_scalar(self, i: int, x: float) -> float:
        return np.maximum(x, 0.0) ** 2 / (4.0 * self.beta)

    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        return np.maximum(x, 0.0) / (1.0 + 2.0 * self.beta * eta)

    def subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        if x == 0.0:
            return [-np.inf, 0.0]
        elif x > 0.0:
            s = 2.0 * self.beta * x
            return [s, s]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        if x > 0.0:
            s = x / (2.0 * self.beta)
            return [s, s]
        else:
            s = 0.0
            return [s, s]

    def param_slope_pos_scalar(self, i: int, lmbd: float) -> float:
        return 2.0 * np.sqrt(lmbd * self.beta)

    def param_slope_neg_scalar(self, i: int, lmbd: float) -> float:
        return -np.inf

    def param_limit_pos_scalar(self, i: int, lmbd: float) -> float:
        return np.sqrt(lmbd / self.beta)

    def param_limit_neg_scalar(self, i: int, lmbd: float) -> float:
        return 0.0

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
