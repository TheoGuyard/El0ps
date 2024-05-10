import numpy as np
import pyomo.kernel as pmo
from numpy.typing import ArrayLike
from numba import float64
from .base import BasePenalty, MipPenalty


class BigmL1norm(BasePenalty, MipPenalty):
    r"""Big-M constraint plus L1-norm penalty function.

    The function is defined as

    .. math:: h(x) = \alpha \|x\|_1 + \mathbb{I}(\|x\|_{\infty} \leq M)

    where :math:`\mathbb{I}(\cdot)` is the convex indicator function,
    :math:`\alpha > 0` and :math:`M > 0`.

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
        return "BigmL1norm"

    def get_spec(self) -> tuple:
        spec = (
            ("M", float64),
            ("alpha", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M, alpha=self.alpha)

    def value_scalar(self, i: int, x: float) -> float:
        xabs = np.abs(x)
        return self.alpha * xabs if xabs <= self.M else np.inf

    def conjugate_scalar(self, i: int, x: float) -> float:
        return self.M * np.maximum(np.abs(x) - self.alpha, 0.0)

    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        v = np.abs(x) - eta * self.alpha
        return np.sign(x) * np.maximum(np.minimum(v, self.M), 0.0)

    def subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        if x == 0.0:
            return [-self.alpha, self.alpha]
        elif np.abs(x) < self.M:
            return [self.alpha, self.alpha]
        elif x == -self.M:
            return [-np.inf, -self.alpha]
        elif x == self.M:
            return [self.alpha, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        if np.abs(x) < self.alpha:
            return [0.0, 0.0]
        elif x == self.alpha:
            return [0.0, self.M]
        elif x == -self.alpha:
            return [-self.M, 0.0]
        else:
            s = np.sign(x) * self.M
            return [s, s]

    def param_slope_scalar(self, i: int, lmbd: float) -> float:
        return (lmbd / self.M) + self.alpha

    def param_limit_scalar(self, i: int, lmbd: float) -> float:
        return self.M

    def param_maxval_scalar(self, i: int) -> float:
        return np.inf

    def param_maxdom_scalar(self, i: int) -> float:
        return np.inf

    def bind_model(self, model: pmo.block, lmbd: float) -> None:

        model.g1_var = pmo.variable_dict()
        for i in model.N:
            model.g1_var[i] = pmo.variable(domain=pmo.Reals)

        model.gpos_con = pmo.constraint_dict()
        model.gneg_con = pmo.constraint_dict()
        model.g1pos_con = pmo.constraint_dict()
        model.g1neg_con = pmo.constraint_dict()
        for i in model.N:
            model.gpos_con[i] = pmo.constraint(
                model.x[i] <= self.M * model.z[i]
            )
            model.gneg_con[i] = pmo.constraint(
                model.x[i] >= -self.M * model.z[i]
            )
            model.g1pos_con[i] = pmo.constraint(model.g1_var[i] >= model.x[i])
            model.g1neg_con[i] = pmo.constraint(model.g1_var[i] >= -model.x[i])
        model.g_con = pmo.constraint(
            model.g
            >= (
                lmbd * sum(model.z[i] for i in model.N)
                + self.alpha * sum(model.g1_var[i] for i in model.N)
            )
        )
