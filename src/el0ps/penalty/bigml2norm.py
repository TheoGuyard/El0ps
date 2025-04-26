import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import SymmetricPenalty, MipPenalty


class BigmL2norm(CompilableClass, SymmetricPenalty, MipPenalty):
    r"""Big-M plus L2-norm :class:`BasePenalty` penalty function.

    The splitting terms are expressed as

    .. math::
        h_i(x_i) = \begin{cases}
        \beta x_i^2 & \text{if } |x_i| \leq M \\
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
        return "BigmL2norm"

    def get_spec(self) -> tuple:
        spec = (
            ("M", float64),
            ("beta", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M, beta=self.beta)

    def value(self, i: int, x: float) -> float:
        return self.beta * x**2 if np.abs(x) <= self.M else np.inf

    def conjugate(self, i: int, x: float) -> float:
        r = np.maximum(np.minimum(x / (2.0 * self.beta), self.M), -self.M)
        return x * r - self.beta * r**2

    def prox(self, i: int, x: float, eta: float) -> float:
        v = x / (1.0 + 2.0 * eta * self.beta)
        return np.maximum(np.minimum(v, self.M), -self.M)

    def subdiff(self, i: int, x: float) -> NDArray:
        if np.abs(x) < self.M:
            s = 2.0 * self.beta * x
            return [s, s]
        elif x == -self.M:
            return [-np.inf, 2.0 * self.beta * x]
        elif x == self.M:
            return [2.0 * self.beta * x, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        s = np.maximum(np.minimum(x / (2.0 * self.beta), self.M), -self.M)
        return [s, s]

    def param_slope(self, i: int, lmbd: float) -> float:
        if lmbd < self.beta * self.M**2:
            return np.sqrt(4.0 * lmbd * self.beta)
        else:
            return (lmbd / self.M) + self.beta * self.M

    def param_limit(self, i: int, lmbd: float) -> float:
        if lmbd < self.beta * self.M**2:
            return np.sqrt(lmbd / self.beta)
        else:
            return self.M

    def param_bndry(self, i, lmbd):
        if lmbd < self.beta * self.M**2:
            return np.sqrt(4.0 * lmbd * self.beta)
        else:
            return np.inf

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
            model.hneg_con[i] = pmo.constraint(
                model.x[i] >= -self.M * model.z[i]
            )
            model.h1_con[i] = pmo.conic.rotated_quadratic(
                model.h1_var[i], model.z[i], [model.x[i]]
            )
        model.h_con = pmo.constraint(
            model.h >= 2.0 * self.beta * sum(model.h1_var[i] for i in model.N)
        )
