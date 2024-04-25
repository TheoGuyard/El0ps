import numpy as np
from numba import int32, float64
from numpy.typing import ArrayLike
from .base import SmoothDatafit


class Logistic(SmoothDatafit):
    r"""Logistic datafit function given by

    .. math:: f(x) = \sum_j \log(1 + \exp(-y_j * x_j))

    Parameters
    ----------
    y: ArrayLike
        Data vector.
    """

    def __init__(self, y: ArrayLike) -> None:
        self.y = y
        self.L = 0.25

    def __str__(self) -> str:
        return "Logistic"

    def get_spec(self) -> tuple:
        spec = (("y", float64[::1]), ("L", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y)

    def value(self, x: ArrayLike) -> float:
        return np.sum(np.log(1.0 + np.exp(-self.y * x)))

    def conjugate(self, x: ArrayLike) -> float:
        c = -(x * self.y)
        if not np.all((0.0 < c) & (c < 1.0)):
            return np.inf
        r = 1.0 - c
        return np.dot(c, np.log(c)) + np.dot(r, np.log(r))

    def lipschitz_constant(self) -> float:
        return self.L

    def gradient(self, x: ArrayLike) -> ArrayLike:
        return -self.y / (1.0 + np.exp(self.y * x))
