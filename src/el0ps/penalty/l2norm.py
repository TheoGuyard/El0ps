import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import SymmetricPenalty, MipPenalty


class L2norm(CompilableClass, SymmetricPenalty, MipPenalty):
    r"""L2-norm :class:`BasePenalty` penalty function.

    The splitting terms are expressed as

    .. math::
        h_i(x_i) = \beta x_i^2

    for some :math:`\beta > 0`.

    Parameters
    ----------
    beta: float
        L2-norm weight.
    """

    def __init__(self, beta: float) -> None:
        self.beta = beta

    def __str__(self) -> str:
        return "L2norm"

    def get_spec(self) -> tuple:
        spec = (("beta", float64),)
        return spec

    def params_to_dict(self) -> dict:
        return dict(beta=self.beta)

    def value(self, i: int, x: float) -> float:
        return self.beta * x**2

    def conjugate(self, i: int, x: float) -> float:
        return x**2 / (4.0 * self.beta)

    def prox(self, i: int, x: float, eta: float) -> float:
        return x / (1.0 + 2.0 * self.beta * eta)

    def subdiff(self, i: int, x: float) -> NDArray:
        s = 2.0 * self.beta * x
        return [s, s]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        s = x / (2.0 * self.beta)
        return [s, s]

    def param_slope(self, i: int, lmbd: float) -> float:
        return 2.0 * np.sqrt(lmbd * self.beta)

    def param_limit(self, i: int, lmbd: float) -> float:
        return np.sqrt(lmbd / self.beta)

    def param_bndry(self, i, lmbd):
        return 2.0 * np.sqrt(lmbd * self.beta)

    def bind_model(self, model: pmo.block) -> None:
        model.h1_var = pmo.variable_dict()
        for i in model.N:
            model.h1_var[i] = pmo.variable(domain=pmo.NonNegativeReals)

        model.h1_con = pmo.constraint_dict()
        for i in model.N:
            model.h1_con[i] = pmo.conic.rotated_quadratic(
                model.h1_var[i], model.z[i], [model.x[i]]
            )
        model.h_con = pmo.constraint(
            model.h >= 2.0 * self.beta * sum(model.h1_var[i] for i in model.N)
        )
