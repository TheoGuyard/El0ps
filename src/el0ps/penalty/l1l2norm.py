import pyomo.environ as pyo
import numpy as np
from numba import float64
from .base import ModelablePenalty, ProximablePenalty


class L1L2norm(ModelablePenalty, ProximablePenalty):
    r"""L1L2-norm penalty function given by

    .. math:: h(x) = \alpha|x| + \beta x^2

    with :math:`\alpha>0` and :math:`\beta>0`.

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

    def value(self, x: float) -> float:
        return self.alpha * np.abs(x) + self.beta * x**2

    def conjugate(self, x: float) -> float:
        return np.maximum(np.abs(x) - self.alpha, 0.0) ** 2 / (4.0 * self.beta)

    def conjugate_scaling_factor(self, x: float) -> float:
        return 1.0

    def param_slope(self, lmbd: float) -> float:
        return self.alpha + np.sqrt(4.0 * self.beta * lmbd)

    def param_limit(self, lmbd: float) -> float:
        return np.sqrt(lmbd / self.beta)

    def param_maxval(self) -> float:
        return np.inf

    def param_maxzer(self) -> float:
        return self.alpha

    def bind_model(self, model: pyo.Model, lmbd: float) -> None:
        def g1pos_con_rule(model: pyo.Model, i: int):
            return model.g1[i] >= model.x[i]

        def g1neg_con_rule(model: pyo.Model, i: int):
            return model.g1[i] >= -model.x[i]

        def g2_con_rule(model: pyo.Model, i: int):
            return model.x[i] ** 2 <= model.g2[i] * model.z[i]

        def g_con_rule(model: pyo.Model):
            return model.g >= (
                lmbd * sum(model.z[i] for i in model.N)
                + self.alpha * sum(model.g1[i] for i in model.N)
                + self.beta * sum(model.g2[i] for i in model.N)
            )

        model.g1 = pyo.Var(model.N, within=pyo.Reals)
        model.g2 = pyo.Var(model.N, within=pyo.Reals)
        model.g1pos_con = pyo.Constraint(model.N, rule=g1pos_con_rule)
        model.g2neg_con = pyo.Constraint(model.N, rule=g1neg_con_rule)
        model.g2_con = pyo.Constraint(model.N, rule=g2_con_rule)
        model.g_con = pyo.Constraint(rule=g_con_rule)

    def prox(self, x: float, eta: float) -> float:
        return (np.sign(x) / (1.0 + 2.0 * eta * self.beta)) * np.maximum(
            np.abs(x) - eta * self.alpha, 0.0
        )
