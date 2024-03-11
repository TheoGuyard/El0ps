import pyomo.environ as pyo
import numpy as np
from numba import int32, float64
from numpy.typing import ArrayLike
from .base import ModelableDatafit, SmoothDatafit


class Squaredhinge(ModelableDatafit, SmoothDatafit):
    r"""Squared-Hinge datafit function given by

    .. math:: f(x) = \frac{1}{m}\sum_{j=1}^m\max(1 - y_j x_j, 0)^2

    where ``m`` is the size of the vector ``y``.

    Parameters
    ----------
    y: ArrayLike
        Data vector.
    """

    def __init__(self, y: ArrayLike) -> None:
        self.y = y
        self.m = y.size
        self.L = 2.0 / y.size

    def __str__(self) -> str:
        return "Squaredhinge"

    def get_spec(self) -> tuple:
        spec = (("y", float64[::1]), ("m", int32), ("L", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y)

    def value(self, x: ArrayLike) -> float:
        return (
            np.linalg.norm(np.maximum(1.0 - self.y * x, 0.0), 2) ** 2 / self.m
        )

    def conjugate(self, x: ArrayLike) -> float:
        return (
            (0.5 * self.m) * np.dot(x, x)
            + np.dot(self.y, x)
            - np.linalg.norm(np.maximum(-0.5 * self.m * (self.y * x), 0.0), 2)
            ** 2
            / self.m
        )

    def bind_model(self, model: pyo.Model) -> None:
        def f1_con_rule(model: pyo.Model, j: int):
            return model.f1[j] >= 1.0 - self.y[j] * model.w[j]

        def f_con_rule(model: pyo.Model):
            return model.f >= sum(model.f1[j] ** 2 for j in model.M) / self.m

        model.f1 = pyo.Var(model.M, within=pyo.NonNegativeReals)
        model.f1_con = pyo.Constraint(model.M, rule=f1_con_rule)
        model.f_con = pyo.Constraint(rule=f_con_rule)

    def gradient(self, x: ArrayLike) -> ArrayLike:
        return -self.y * np.maximum(1.0 - self.y * x, 0.0) / (0.5 * self.m)
