import pyomo.environ as pyo
import numpy as np
from numba import float64
from .base import ModelablePenalty, ProximablePenalty


class Bigm(ModelablePenalty, ProximablePenalty):
    r"""Big-M penalty function given by

    .. math:: h(x) = 0 \ \ \text{if} \ \ |x| \leq M \ \ \text{and} \ \ h(x) = +\infty \ \ \text{otherwise}

    with :math:`M>0`.

    Parameters
    ----------
    M: float
        Big-M value.
    """  # noqa: E501

    def __init__(self, M: float) -> None:
        self.M = M

    def __str__(self) -> str:
        return "Bigm"

    def get_spec(self) -> tuple:
        spec = (("M", float64),)
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M)

    def value(self, x: float) -> float:
        return 0.0 if np.abs(x) <= self.M else np.inf

    def conjugate(self, x: float) -> float:
        return self.M * np.abs(x)

    def conjugate_scaling_factor(self, x: float) -> float:
        return 1.0

    def param_slope(self, lmbd: float) -> float:
        return lmbd / self.M

    def param_limit(self, lmbd: float) -> float:
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

        def g_con_rule(model: pyo.Model):
            return model.g >= lmbd * sum(model.z[i] for i in model.N)

        model.gpos_con = pyo.Constraint(model.N, rule=gpos_con_rule)
        model.gneg_con = pyo.Constraint(model.N, rule=gneg_con_rule)
        model.g_con = pyo.Constraint(rule=g_con_rule)

    def prox(self, x: float, eta: float) -> float:
        return np.maximum(np.minimum(x, self.M), -self.M)
