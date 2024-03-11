import numpy as np
from numba import int32, float64
from numpy.typing import ArrayLike
from .base import SmoothDatafit


class Squaredhinge(SmoothDatafit):
    r"""Squared-Hinge datafit function given by

    .. math:: f(x) = 1 / m \sum_(j=1)^m \max(1 - y_j * x_j, 0)^2

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
        v = np.maximum(1.0 - self.y * x, 0.0)
        return np.dot(v, v) / self.m

    def conjugate(self, x: ArrayLike) -> float:
        v = np.maximum(-0.5 * self.m * (self.y * x), 0.0)
        return (
            (0.5 * self.m) * np.dot(x, x)
            + np.dot(self.y, x)
            - np.dot(v, v) / self.m
        )

    def lipschitz_constant(self) -> float:
        return self.L

    def gradient(self, x: ArrayLike) -> ArrayLike:
        return -self.y * np.maximum(1.0 - self.y * x, 0.0) / (0.5 * self.m)
