import pyomo.environ as pyo
import numpy as np
from numba import float64
from .base import ModelablePenalty, ProximablePenalty


class L2norm(ModelablePenalty, ProximablePenalty):
    r"""L2-norm penalty function given by

    .. math:: f(x) = \alpha x^2

    with :math:`\alpha>0`.

    Parameters
    ----------
    alpha: float, positive
        L2-norm weight.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __str__(self) -> str:
        return "L2norm"

    def get_spec(self) -> tuple:
        spec = (("alpha", float64),)
        return spec

    def params_to_dict(self) -> dict:
        return dict(alpha=self.alpha)

    def value(self, x: float) -> float:
        return self.alpha * x**2

    def conjugate(self, x: float) -> float:
        return x**2 / (4.0 * self.alpha)

    def conjugate_scaling_factor(self, x: float) -> float:
        return 1.0

    def param_slope(self, lmbd: float) -> float:
        return 2.0 * np.sqrt(lmbd * self.alpha)

    def param_limit(self, lmbd: float) -> float:
        return np.sqrt(lmbd / self.alpha)

    def param_maxval(self) -> float:
        return np.inf

    def param_maxzer(self) -> float:
        return 0.0

    def bind_model(self, model: pyo.Model, lmbd: float) -> None:
        def g1_con_rule(model: pyo.Model, i: int):
            return model.x[i] ** 2 <= model.g1[i] * model.z[i]

        def g_con_rule(model: pyo.Model):
            return model.g >= (
                lmbd * sum(model.z[i] for i in model.N)
                + self.alpha * sum(model.g1[i] for i in model.N)
                + self.beta * sum(model.g2[i] for i in model.N)
            )

        model.g1 = pyo.Var(model.N, within=pyo.Reals)
        model.g1_con = pyo.Constraint(model.N, rule=g1_con_rule)
        model.g_con = pyo.Constraint(rule=g_con_rule)

    def prox(self, x: float, eta: float) -> float:
        return x / (1.0 + 2.0 * eta * self.alpha)
