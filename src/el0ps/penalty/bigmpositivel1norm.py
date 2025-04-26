import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import BasePenalty, MipPenalty


class BigmPositiveL1norm(CompilableClass, BasePenalty, MipPenalty):
    r"""Positive big-M plus L1-norm :class:`BasePenalty` penalty function.

    The splitting terms are expressed as

    .. math::
        h_i(x_i) = \begin{cases}
        \alpha x_i & \text{if } 0 \leq x_i \leq M \\
        +\infty & \text{otherwise}
        \end{cases}

    for some :math:`M > 0` and :math:`\alpha > 0`.

    Parameters
    ----------
    M: float
        Big-M value.
    alpha: float
        L1-norm weight.
    """

    def __init__(self, M: float, alpha: float) -> None:
        self.M = M
        self.alpha = alpha

    def __str__(self) -> str:
        return "BigmPositiveL1norm"

    def get_spec(self) -> tuple:
        spec = (
            ("M", float64),
            ("alpha", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M, alpha=self.alpha)

    def value(self, i: int, x: float) -> float:
        return self.alpha * x if (x >= 0.0) and (x <= self.M) else np.inf

    def conjugate(self, i: int, x: float) -> float:
        return self.M * np.maximum(x - self.alpha, 0.0)

    def prox(self, i: int, x: float, eta: float) -> float:
        return np.maximum(0.0, np.minimum(x - eta * self.alpha, self.M))

    def subdiff(self, i: int, x: float) -> NDArray:
        if x == 0.0:
            return [-np.inf, self.alpha]
        elif 0.0 < x < self.M:
            s = self.alpha
            return [s, s]
        elif x == self.M:
            return [self.alpha, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        if x < self.alpha:
            return [0.0, 0.0]
        elif x == self.alpha:
            return [0.0, self.M]
        else:
            s = np.sign(x) * self.M
            return [s, s]

    def param_slope_pos(self, i: int, lmbd: float) -> float:
        return (lmbd / self.M) + self.alpha

    def param_slope_neg(self, i: int, lmbd: float) -> float:
        return -np.inf

    def param_limit_pos(self, i: int, lmbd: float) -> float:
        return self.M

    def param_limit_neg(self, i: int, lmbd: float) -> float:
        return 0.0

    def param_bndry_pos(self, i: int, lmbd: float) -> float:
        return np.inf

    def param_bndry_neg(self, i: int, lmbd: float) -> float:
        return -np.inf

    def bind_model(self, model: pmo.block) -> None:

        model.h1_var = pmo.variable_dict()
        for i in model.N:
            model.h1_var[i] = pmo.variable(domain=pmo.NonNegativeReals)

        model.hpos_con = pmo.constraint_dict()
        model.hneg_con = pmo.constraint_dict()
        model.h1pos_con = pmo.constraint_dict()
        model.h1neg_con = pmo.constraint_dict()
        for i in model.N:
            model.hpos_con[i] = pmo.constraint(
                model.x[i] <= self.M * model.z[i]
            )
            model.hneg_con[i] = pmo.constraint(model.x[i] >= 0)
            model.h1pos_con[i] = pmo.constraint(model.h1_var[i] >= model.x[i])
            model.h1neg_con[i] = pmo.constraint(model.h1_var[i] >= 0.0)
        model.h_con = pmo.constraint(
            model.h >= self.alpha * sum(model.h1_var[i] for i in model.N)
        )
