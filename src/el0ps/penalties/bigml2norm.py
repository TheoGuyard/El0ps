import numpy as np
import pyomo.kernel as pmo
from numpy.typing import ArrayLike
from numba import float64
from .base import BasePenalty, MipPenalty


class BigmL2norm(BasePenalty, MipPenalty):
    r"""Big-M constraint plus L2-norm penalty function.

    The function is defined as

    .. math:: h(x) = \beta \|x\|_2^2 + \mathbb{I}(\|x\|_{\infty} \leq M)

    where :math:`\mathbb{I}(\cdot)` is the convex indicator function,
    :math:`\beta > 0` and :math:`M > 0`.


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

    def value_scalar(self, i: int, x: float) -> float:
        return self.beta * x**2 if np.abs(x) <= self.M else np.inf

    def conjugate_scalar(self, i: int, x: float) -> float:
        r = np.maximum(np.minimum(x / (2.0 * self.beta), self.M), -self.M)
        return x * r - self.beta * r**2

    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        v = x / (1.0 + 2.0 * eta * self.beta)
        return np.maximum(np.minimum(v, self.M), -self.M)

    def subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        if np.abs(x) < self.M:
            s = 2.0 * self.beta * x
            return [s, s]
        elif x == -self.M:
            return [-np.inf, 2.0 * self.beta * x]
        elif x == self.M:
            return [2.0 * self.beta * x, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        s = np.maximum(np.minimum(x / (2.0 * self.beta), self.M), -self.M)
        return [s, s]

    def param_slope_scalar(self, i: int, lmbd: float) -> float:
        if lmbd < self.beta * self.M**2:
            return np.sqrt(4.0 * lmbd * self.beta)
        else:
            return (lmbd / self.M) + self.beta * self.M

    def param_limit_scalar(self, i: int, lmbd: float) -> float:
        if lmbd < self.beta * self.M**2:
            return np.sqrt(lmbd / self.beta)
        else:
            return self.M

    def param_maxval_scalar(self, i: int) -> float:
        return np.inf

    def param_maxdom_scalar(self, i: int) -> float:
        return np.inf

    def bind_model(self, model: pmo.block, lmbd: float) -> None:

        model.g1_var = pmo.variable_dict()
        for i in model.N:
            model.g1_var[i] = pmo.variable(domain=pmo.NonNegativeReals)

        model.gpos_con = pmo.constraint_dict()
        model.gneg_con = pmo.constraint_dict()
        model.g1_con = pmo.constraint_dict()
        for i in model.N:
            model.gpos_con[i] = pmo.constraint(
                model.x[i] <= self.M * model.z[i]
            )
            model.gneg_con[i] = pmo.constraint(
                model.x[i] >= -self.M * model.z[i]
            )
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
