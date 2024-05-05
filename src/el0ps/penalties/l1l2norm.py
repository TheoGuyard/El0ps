import numpy as np
import pyomo.kernel as pmo
from numpy.typing import ArrayLike
from numba import float64
from .base import MipPenalty


class L1L2norm(MipPenalty):
    r"""L1L2-norm penalty function given by
    :math:`h(x) = \alpha |x| + \beta x^2`, with :math:`\alpha>0` and
    :math:`\beta>0`.

    Parameters
    ----------
    alpha: float, positive
        L1-norm weight.
    beta: float, positive
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

    def subdiff(self, i: int, x: float) -> ArrayLike:
        if x == 0:
            return [-self.alpha, self.alpha]
        else:
            s = self.alpha * np.sign(x) + 2.0 * self.beta * x
            return [s, s]

    def conjugate_subdiff(self, i: int, x: float) -> ArrayLike:
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

    def param_maxval(self, i: int) -> float:
        return np.inf

    def param_maxdom(self, i: int) -> float:
        return np.inf

    def bind_model(self, model: pmo.block, lmbd: float) -> None:

        model.g1_var = pmo.variable_dict()
        model.g2_var = pmo.variable_dict()
        for i in model.N:
            model.g1_var[i] = pmo.variable(domain=pmo.Reals)
            model.g2_var[i] = pmo.variable(domain=pmo.Reals)

        model.g1pos_con = pmo.constraint_dict()
        model.g1neg_con = pmo.constraint_dict()
        model.g2_con = pmo.constraint_dict()
        for i in model.N:
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
