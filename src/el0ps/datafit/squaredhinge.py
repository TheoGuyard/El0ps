import numpy as np
from numba import int32, float64
from numpy.typing import ArrayLike
from .base import SmoothDatafit


class Squaredhinge(SmoothDatafit):
    r"""Squared-Hinge datafit function given by

    .. math:: f(x) = \sum_j \max(1 - y_j * x_j, 0)^2

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
        return -2. * self.y * np.maximum(1.0 - self.y * x, 0.0)
