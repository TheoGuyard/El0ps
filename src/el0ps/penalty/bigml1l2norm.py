import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import SymmetricPenalty, MipPenalty


class BigmL1L2norm(CompilableClass, SymmetricPenalty, MipPenalty):
    r"""Big-M plus L1L2-norm :class:`BasePenalty` penalty function.

    The splitting terms are expressed as

    .. math::
        h_i(x_i) = \begin{cases}
        \alpha |x_i| + \beta x_i^2 & \text{if } |x_i| \leq M \\
        +\infty & \text{otherwise}
        \end{cases}

    for some :math:`M > 0`, :math:`\alpha > 0`, and :math:`\beta > 0`.

    Parameters
    ----------
    M: float
        Big-M value.
    alpha: float
        L1-norm weight.
    beta: float
        L2-norm weight.
    """

    def __init__(self, M: float, alpha: float, beta: float) -> None:
        self.M = M
        self.alpha = alpha
        self.beta = beta

    def __str__(self) -> str:
        return "BigmL1L2norm"

    def get_spec(self) -> tuple:
        spec = (
            ("M", float64),
            ("alpha", float64),
            ("beta", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M, alpha=self.alpha, beta=self.beta)

    def value(self, i: int, x: float) -> float:
        if np.abs(x) <= self.M:
            return self.alpha * np.abs(x) + self.beta * x**2
        else:
            return np.inf

    def conjugate(self, i: int, x: float) -> float:
        if np.abs(x) <= self.alpha + 2.0 * self.beta * self.M:
            return np.maximum(np.abs(x) - self.alpha, 0.0) ** 2 / (
                4.0 * self.beta
            )
        else:
            return self.M * (np.abs(x) - self.alpha) - self.beta * self.M**2

    def prox(self, i: int, x: float, eta: float) -> float:
        if np.abs(x) <= eta * self.alpha + self.M * (
            2.0 * eta * self.beta + 1
        ):
            return (
                np.sign(x)
                * np.maximum(np.abs(x) - eta * self.alpha, 0.0)
                / (2.0 * eta * self.beta + 1.0)
            )
        else:
            return np.sign(x) * self.M

    def subdiff(self, i: int, x: float) -> NDArray:
        if x == 0.0:
            return [-self.alpha, self.alpha]
        elif np.abs(x) < self.M:
            s = self.alpha * np.sign(x) + 2.0 * self.beta * x
            return [s, s]
        elif x == -self.M:
            return [-np.inf, -self.alpha + 2.0 * self.beta * x]
        elif x == self.M:
            return [self.alpha + 2.0 * self.beta * x, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        if np.abs(x) <= self.alpha + 2.0 * self.beta * self.M:
            return np.sign(x) * (np.abs(x) - self.alpha) / (2.0 * self.beta)
        else:
            return self.M * np.sign(x)

    def param_slope(self, i: int, lmbd: float) -> float:
        if lmbd <= self.beta * self.M**2:
            return self.alpha + np.sqrt(4.0 * self.beta * lmbd)
        else:
            return self.alpha + (lmbd / self.M) + self.beta * self.M

    def param_limit(self, i: int, lmbd: float) -> float:
        if lmbd <= self.beta * self.M**2:
            return np.sqrt(lmbd / self.beta)
        else:
            return self.M

    def param_bndry(self, i, lmbd):
        if lmbd <= self.beta * self.M**2:
            return self.alpha + np.sqrt(4.0 * self.beta * lmbd)
        else:
            return np.inf

    def bind_model(self, model: pmo.block) -> None:

        model.h1_var = pmo.variable_dict()
        model.h2_var = pmo.variable_dict()
        for i in model.N:
            model.h1_var[i] = pmo.variable(
                domain=pmo.NonNegativeReals, ub=self.M
            )
            model.h2_var[i] = pmo.variable(
                domain=pmo.NonNegativeReals, ub=self.M**2
            )

        model.hpos_con = pmo.constraint_dict()
        model.hneg_con = pmo.constraint_dict()
        model.h1pos_con = pmo.constraint_dict()
        model.h1neg_con = pmo.constraint_dict()
        model.h2_con = pmo.constraint_dict()
        for i in model.N:
            model.hpos_con[i] = pmo.constraint(
                model.x[i] <= self.M * model.z[i]
            )
            model.hneg_con[i] = pmo.constraint(
                model.x[i] >= -self.M * model.z[i]
            )
            model.h1pos_con[i] = pmo.constraint(model.x[i] <= model.h1_var[i])
            model.h1neg_con[i] = pmo.constraint(-model.x[i] <= model.h1_var[i])
            model.h2_con[i] = pmo.conic.rotated_quadratic(
                model.h2_var[i], model.z[i], [model.x[i]]
            )
        model.h_con = pmo.constraint(
            model.h
            >= self.alpha * sum(model.h1_var[i] for i in model.N)
            + 2.0 * self.beta * sum(model.h2_var[i] for i in model.N)
        )
