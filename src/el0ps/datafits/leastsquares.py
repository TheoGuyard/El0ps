import numpy as np
import pyomo.kernel as pmo
from numba import float64
from numpy.typing import ArrayLike
from .base import MipDatafit, SmoothDatafit


class Leastsquares(SmoothDatafit, MipDatafit):
    r"""Least-squares datafit function.

    The function is defined as

    .. math:: f(x) = \textstyle \frac{1}{2} \sum_j (y_j - x_j)^2

    Parameters
    ----------
    y: ArrayLike
        Data vector.
    """

    def __init__(self, y: ArrayLike) -> None:
        self.y = y
        self.L = 1.0

    def __str__(self) -> str:
        return "Leastsquares"

    def get_spec(self) -> tuple:
        spec = (("y", float64[::1]), ("L", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y)

    def value(self, x: ArrayLike) -> float:
        v = x - self.y
        return 0.5 * np.dot(v, v)

    def conjugate(self, x: ArrayLike) -> float:
        return 0.5 * np.dot(x, x) + np.dot(x, self.y)

    def lipschitz_constant(self) -> float:
        return self.L

    def gradient(self, x: ArrayLike) -> ArrayLike:
        return x - self.y

    def bind_model(self, model: pmo.block) -> None:
        model.f_con = pmo.constraint(
            model.f
            >= sum((model.w[j] - self.y[j]) ** 2 for j in model.M) / 2.0
        )
