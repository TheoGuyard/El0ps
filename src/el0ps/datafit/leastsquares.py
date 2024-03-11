import numpy as np
from numba import int32, float64
from numpy.typing import ArrayLike
from .base import SmoothDatafit


class Leastsquares(SmoothDatafit):
    r"""Least-squares datafit function given by

    .. math:: f(x) = 1 / 2m ||x - y||_2^2

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
        v = x - self.y
        return np.dot(v, v) / (2.0 * self.m)

    def conjugate(self, x: ArrayLike) -> float:
        return (0.5 * self.m) * np.dot(x, x) + np.dot(x, self.y)

    def lipschitz_constant(self) -> float:
        return self.L
    
    def gradient(self, x: ArrayLike) -> ArrayLike:
        return (x - self.y) / self.m
