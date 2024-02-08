import pyomo.environ as pyo
import numpy as np
from numba import float64
from .base import ModelablePenalty, ProximablePenalty


class BigmL2norm(ModelablePenalty, ProximablePenalty):
    r"""Big-M constraint plus L2-norm penalty function given by

    .. math:: h(x) = \alpha x^2 \ \ \text{if} \ \ |x| \leq M \ \ \text{and} \ \ h(x)=+\infty \ \ \text{otherwise}

    with :math:`M>0` and :math:`\alpha>0`.

    Parameters
    ----------
    M: float
        Big-M value.
    alpha: float
        L2-norm weight.
    """  # noqa: E501

    def __init__(self, M: float, alpha: float) -> None:
        self.M = M
        self.alpha = alpha

    def __str__(self) -> str:
        return "BigmL2norm"

    def get_spec(self) -> tuple:
        spec = (
            ("M", float64),
            ("alpha", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M, alpha=self.alpha)

    def value(self, x: float) -> float:
        return self.alpha * x**2 if np.abs(x) <= self.M else np.inf

    def conjugate(self, x: float) -> float:
        r = np.maximum(np.minimum(x / (2.0 * self.alpha), self.M), -self.M)
        return x * r - self.alpha * r**2

    def conjugate_scaling_factor(self, x: float) -> float:
        return 1.0

    def param_slope(self, lmbd: float) -> float:
        if lmbd < self.alpha * self.M**2:
            return np.sqrt(4.0 * lmbd * self.alpha)
        else:
            return (lmbd / self.M) + self.alpha * self.M

    def param_limit(self, lmbd: float) -> float:
        if 1.0 < self.alpha * self.M**2:
            return np.sqrt(lmbd / self.alpha)
        else:
            return self.M

    def param_maxval(self) -> float:
        return np.inf

    def param_maxzer(self) -> float:
        return 0.0

    def bind_model(self, model: pyo.Model, lmbd: float) -> None:
        def gpos_con_rule(model: pyo.Model, i: int):
            return model.x[i] <= self.M * model.z[i]

        def gneg_con_rule(model: pyo.Model, i: int):
            return model.x[i] >= -self.M * model.z[i]

        def g1_con_rule(model: pyo.Model, i: int):
            return model.x[i] ** 2 <= model.g1[i] * model.z[i]

        def g_con_rule(model: pyo.Model):
            return model.g >= (
                lmbd * sum(model.z[i] for i in model.N)
                + self.alpha * sum(model.g1[i] for i in model.N)
            )

        model.g1 = pyo.Var(model.N, within=pyo.Reals)
        model.gpos_con = pyo.Constraint(model.N, rule=gpos_con_rule)
        model.gneg_con = pyo.Constraint(model.N, rule=gneg_con_rule)
        model.g1_con = pyo.Constraint(model.N, rule=g1_con_rule)
        model.g_con = pyo.Constraint(rule=g_con_rule)

    def prox(self, x: float, eta: float) -> float:
        return np.maximum(
            np.minimum(x / (1.0 + 2.0 * eta * self.alpha), self.M), -self.M
        )
