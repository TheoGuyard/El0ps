import pyomo.environ as pyo
import numpy as np
from numba import int32, float64
from numpy.typing import ArrayLike
from .base import ModelableDatafit, ProximableDatafit, SmoothDatafit


class Leastsquares(ModelableDatafit, ProximableDatafit, SmoothDatafit):
    r"""Least-squares datafit function given by

    .. math:: f(x) = \frac{1}{2m} \|x - y\|_2^2

    where ``m`` is the size of the vector ``y``.

    Parameters
    ----------
    y: ArrayLike
        Data vector.
    """

    def __init__(self, y: ArrayLike) -> None:
        self.y = y
        self.m = y.size
        self.L = 1.0 / y.size

    def __str__(self) -> str:
        return "Leastsquares"

    def get_spec(self) -> tuple:
        spec = (("y", float64[::1]), ("m", int32), ("L", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y)

    def value(self, x: ArrayLike) -> float:
        return np.linalg.norm(x - self.y, 2) ** 2 / (2.0 * self.m)

    def conjugate(self, x: ArrayLike) -> float:
        return (0.5 * self.m) * np.dot(x, x) + np.dot(x, self.y)

    def bind_model(self, model: pyo.Model) -> None:
        def f_con_rule(model: pyo.Model):
            return model.f >= sum(
                (model.w[j] - self.y[j]) ** 2 for j in model.M
            ) / (2.0 * self.m)

        model.f_con = pyo.Constraint(rule=f_con_rule)

    def prox(self, x: ArrayLike, eta: float) -> ArrayLike:
        return (x + (eta / self.m) * self.y) / (1.0 + eta / self.m)

    def gradient(self, x: ArrayLike) -> ArrayLike:
        return (x - self.y) / self.m
