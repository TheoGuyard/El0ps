import numpy as np
from numba import int32, float64
from numpy.typing import ArrayLike
from .base import SmoothDatafit


class Logistic(SmoothDatafit):
    r"""Logistic datafit function given by

    .. math:: f(x) = \frac{1}{m} \sum_{j=1}^m \log(1 + \exp(-y_j x_j))

    where ``m`` is the size of the vector ``y``.

    Parameters
    ----------
    y: ArrayLike
        Data vector.
    """

    def __init__(self, y: ArrayLike) -> None:
        self.y = y
        self.m = y.size
        self.L = 1.0 / (4.0 * y.size)

    def __str__(self) -> str:
        return "Logistic"

    def get_spec(self) -> tuple:
        spec = (("y", float64[::1]), ("m", int32), ("L", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y)

    def value(self, x: ArrayLike) -> float:
        return np.sum(np.log(1.0 + np.exp(-self.y * x))) / self.m

    def conjugate(self, x: ArrayLike) -> float:
        c = -(x * self.y) * self.m
        if not np.all((0.0 < c) & (c < 1.0)):
            return np.inf
        r = 1.0 - c
        return (np.dot(c, np.log(c)) + np.dot(r, np.log(r))) / self.m

    def gradient(self, x: ArrayLike) -> ArrayLike:
        return -self.y / (self.m * (1.0 + np.exp(self.y * x)))
