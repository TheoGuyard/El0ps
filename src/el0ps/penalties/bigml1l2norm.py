import numpy as np
import pyomo.kernel as pmo
from numpy.typing import ArrayLike
from numba import float64
from .base import BasePenalty, MipPenalty


class BigmL1L2norm(BasePenalty, MipPenalty):
    r"""Big-M constraint plus L1L2-norm penalty function.

    The function is defined as

    .. math:: h(x) = \alpha \|x\|_2^2 + \beta \|x\|_2^2 + \mathbb{I}(\|x\|_{\infty} \leq M)

    where :math:`\mathbb{I}(\cdot)` is the convex indicator function,
    :math:`\alpha > 0`, :math:`\beta > 0` and :math:`M > 0`.


    Parameters
    ----------
    M: float
        Big-M value.
    alpha: float
        L1-norm weight.
    beta: float
        L2-norm weight.
    """  # noqa: E501

    def __init__(self, M: float, alpha: float, beta: float) -> None:
        self.M = M
        self.alpha = alpha
        self.beta = beta

    def __str__(self) -> str:
        return "BigmL1L2norm"

    def get_spec(self) -> tuple:
        spec = (("M", float64), ("alpha", float64), ("beta", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M, alpha=self.alpha, beta=self.beta)

    def value_scalar(self, i: int, x: float) -> float:
        z = np.abs(x)
        if z <= self.M:
            return self.alpha * z + self.beta * x**2
        else:
            return np.inf

    def conjugate_scalar(self, i: int, x: float) -> float:
        z = np.sign(x) * np.minimum(
            np.maximum((np.abs(x) - self.alpha) / (2.0 * self.beta), 0.0),
            self.M,
        )
        return x * z - self.alpha * z - self.beta * z**2

    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        p = np.sign(x) * np.minimum(
            np.maximum(
                (np.abs(x) - eta * self.alpha) / (1.0 + 2.0 * eta * self.beta),
                0.0,
            ),
            self.M,
        )
        return p

    def subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        if x == 0.0:
            return [-self.alpha, self.alpha]
        if np.abs(x) < self.M:
            s = self.alpha * np.sign(x) + 2.0 * self.beta * x
            return [s, s]
        elif x == -self.M:
            return [-np.inf, -self.alpha - 2.0 * self.beta * self.M]
        elif x == self.M:
            return [self.alpha + 2.0 * self.beta * self.M, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        s = np.sign(x) * np.minimum(
            np.maximum((np.abs(x) - self.alpha) / (2.0 * self.beta), 0.0),
            self.M,
        )
        return [s, s]

    def param_slope_scalar(self, i: int, lmbd: float) -> float:
        if lmbd <= self.beta * self.M**2:
            return self.alpha + np.sqrt(4.0 * lmbd * self.beta)
        else:
            return self.alpha + (lmbd / self.M) + self.beta * self.M

    def param_limit_scalar(self, i: int, lmbd: float) -> float:
        if lmbd <= self.beta * self.M**2:
            return np.sqrt(lmbd / self.beta)
        else:
            return self.M

    def param_maxval_scalar(self, i: int) -> float:
        return np.inf

    def param_maxdom_scalar(self, i: int) -> float:
        return np.inf

    def bind_model(self, model: pmo.block, lmbd: float) -> None:

        model.g1_var = pmo.variable_dict()
        model.g2_var = pmo.variable_dict()
        for i in model.N:
            model.g1_var[i] = pmo.variable(domain=pmo.NonNegativeReals)
            model.g2_var[i] = pmo.variable(domain=pmo.NonNegativeReals)

        model.gpos_con = pmo.constraint_dict()
        model.gneg_con = pmo.constraint_dict()
        model.g1pos_con = pmo.constraint_dict()
        model.g1neg_con = pmo.constraint_dict()
        model.g2_con = pmo.constraint_dict()
        for i in model.N:
            model.gpos_con[i] = pmo.constraint(
                model.x[i] <= self.M * model.z[i]
            )
            model.gneg_con[i] = pmo.constraint(
                model.x[i] >= -self.M * model.z[i]
            )
            model.g1pos_con[i] = pmo.constraint(model.g1_var[i] >= model.x[i])
            model.g1neg_con[i] = pmo.constraint(model.g1_var[i] >= -model.x[i])
            model.g2_con[i] = pmo.constraint(
                model.x[i] ** 2 <= model.g2_var[i] * model.z[i]
            )
        model.g_con = pmo.constraint(
            model.g
            >= (
                lmbd * sum(model.z[i] for i in model.N)
                + self.alpha * sum(model.g1_var[i] for i in model.N)
                + self.beta * sum(model.g2_var[i] for i in model.N)
            )
        )
