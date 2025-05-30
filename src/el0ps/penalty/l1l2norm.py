import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import SymmetricPenalty, MipPenalty


class L1L2norm(CompilableClass, SymmetricPenalty, MipPenalty):
    r"""L1L2-norm :class:`BasePenalty` penalty function.

    The splitting terms are expressed as

    .. math::
        h_i(x_i) = \alpha|x_i| + \beta x_i^2

    for some :math:`\alpha > 0` and :math:`\beta > 0`.

    Parameters
    ----------
    alpha: float
        L1-norm weight.
    beta: float
        L2-norm weight.
    """

    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta

    def __str__(self) -> str:
        return "L1L2norm"

    def get_spec(self) -> tuple:
        spec = (
            ("alpha", float64),
            ("beta", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(alpha=self.alpha, beta=self.beta)

    def value(self, i: int, x: float) -> float:
        return self.alpha * np.abs(x) + self.beta * x**2

    def conjugate(self, i: int, x: float) -> float:
        return np.maximum(np.abs(x) - self.alpha, 0.0) ** 2 / (4.0 * self.beta)

    def prox(self, i: int, x: float, eta: float) -> float:
        v = np.sign(x) / (1.0 + 2.0 * eta * self.beta)
        return v * np.maximum(np.abs(x) - eta * self.alpha, 0.0)

    def subdiff(self, i: int, x: float) -> NDArray:
        if x == 0:
            return [-self.alpha, self.alpha]
        else:
            s = self.alpha * np.sign(x) + 2.0 * self.beta * x
            return [s, s]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        if np.abs(x) < self.alpha:
            return [0.0, 0.0]
        elif x == -self.alpha:
            return [(x + self.alpha) / (2.0 * self.beta), 0.0]
        elif x == self.alpha:
            return [0.0, (x - self.alpha) / (2.0 * self.beta)]
        else:
            s = (x - self.alpha * np.sign(x)) / (2.0 * self.beta)
            return [s, s]

    def param_slope(self, i: int, lmbd: float) -> float:
        return self.alpha + np.sqrt(4.0 * self.beta * lmbd)

    def param_limit(self, i: int, lmbd: float) -> float:
        return np.sqrt(lmbd / self.beta)

    def param_bndry(self, i, lmbd):
        return (self.alpha + 2.0 * self.beta) * np.sqrt(lmbd / self.beta)

    def bind_model(self, model: pmo.block) -> None:

        model.h1_var = pmo.variable_dict()
        model.h2_var = pmo.variable_dict()
        for i in model.N:
            model.h1_var[i] = pmo.variable(domain=pmo.NonNegativeReals)
            model.h2_var[i] = pmo.variable(domain=pmo.NonNegativeReals)

        model.h1pos_con = pmo.constraint_dict()
        model.h1neg_con = pmo.constraint_dict()
        model.h2_con = pmo.constraint_dict()
        for i in model.N:
            model.h1pos_con[i] = pmo.constraint(model.h1_var[i] >= model.x[i])
            model.h1neg_con[i] = pmo.constraint(model.h1_var[i] >= -model.x[i])
            model.h2_con[i] = pmo.conic.rotated_quadratic(
                model.h2_var[i], model.z[i], [model.x[i]]
            )
        model.h_con = pmo.constraint(
            model.h
            >= (
                self.alpha * sum(model.h1_var[i] for i in model.N)
                + 2.0 * self.beta * sum(model.h2_var[i] for i in model.N)
            )
        )
