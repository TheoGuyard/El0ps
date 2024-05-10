import numpy as np
import pyomo.kernel as pmo
from numpy.typing import ArrayLike
from numba import float64
from .base import BasePenalty, MipPenalty


class L2norm(BasePenalty, MipPenalty):
    r"""L2-norm penalty function.

    The function is defined as

    .. math:: h(x) = \beta \|x\|_2^2

    where :math:`\beta > 0`.

    Parameters
    ----------
    beta: float, positive
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

    def value_scalar(self, i: int, x: float) -> float:
        return self.beta * x**2

    def conjugate_scalar(self, i: int, x: float) -> float:
        return x**2 / (4.0 * self.beta)

    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        return x / (1.0 + 2.0 * eta * self.beta)

    def subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        s = 2.0 * self.beta * x
        return [s, s]

    def conjugate_subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        s = x / (2.0 * self.beta)
        return [s, s]

    def param_slope_scalar(self, i: int, lmbd: float) -> float:
        return 2.0 * np.sqrt(lmbd * self.beta)

    def param_limit_scalar(self, i: int, lmbd: float) -> float:
        return np.sqrt(lmbd / self.beta)

    def param_maxval_scalar(self, i: int) -> float:
        return np.inf

    def param_maxdom_scalar(self, i: int) -> float:
        return np.inf

    def bind_model(self, model: pmo.block, lmbd: float) -> None:
        model.g1_var = pmo.variable_dict()
        for i in model.N:
            model.g1_var[i] = pmo.variable(domain=pmo.Reals)

        model.g1_con = pmo.constraint_dict()
        for i in model.N:
            model.g1_con[i] = pmo.constraint(
                model.x[i] ** 2 <= model.g1_var[i] * model.z[i]
            )
        model.g_con = pmo.constraint(
            model.g
            >= (
                lmbd * sum(model.z[i] for i in model.N)
                + self.beta * sum(model.g1_var[i] for i in model.N)
            )
        )
