import numpy as np
import pyomo.kernel as pmo
from numba import float64
from numpy.typing import ArrayLike
from .base import MipDatafit, SmoothDatafit


class Squaredhinge(SmoothDatafit, MipDatafit):
    r"""Squared-Hinge datafit function.

    The function is defined as

    .. math:: f(x) = \textstyle \sum_j \max(1 - y_j x_j, 0)^2

    Parameters
    ----------
    y: ArrayLike
        Data vector.
    """

    def __init__(self, y: ArrayLike) -> None:
        self.y = y
        self.L = 2.0

    def __str__(self) -> str:
        return "Squaredhinge"

    def get_spec(self) -> tuple:
        spec = (("y", float64[::1]), ("L", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y)

    def value(self, x: ArrayLike) -> float:
        v = np.maximum(1.0 - self.y * x, 0.0)
        return np.dot(v, v)

    def conjugate(self, x: ArrayLike) -> float:
        v = np.maximum(-0.5 * (self.y * x), 0.0)
        return 0.5 * np.dot(x, x) + np.dot(self.y, x) - np.dot(v, v)

    def lipschitz_constant(self) -> float:
        return self.L

    def gradient(self, x: ArrayLike) -> ArrayLike:
        return -2.0 * self.y * np.maximum(1.0 - self.y * x, 0.0)

    def bind_model(self, model: pmo.block) -> None:
        model.f1_var = pmo.variable_dict()
        for j in model.M:
            model.f1_var[j] = pmo.variable(domain=pmo.NonNegativeReals)
        model.f1_con = pmo.constraint_dict()
        for j in model.M:
            model.f1_con[j] = pmo.constraint(
                model.f1_var[j] >= 1.0 - self.y[j] * model.w[j]
            )
        model.f_con = pmo.constraint(
            model.f >= sum(model.f1_var[j] ** 2 for j in model.M)
        )
