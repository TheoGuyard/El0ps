import numpy as np
from numba import int32, float64
from numpy.typing import NDArray
from .base import SmoothDatafit


class Logcosh(SmoothDatafit):
    r"""Logcosh datafit function given by

    .. math:: f(x) = \frac{1}{m} \sum_{j=1}^m \log(\cosh(x_j - y_j))

    where ``m`` is the size of the vector ``y``.

    Parameters
    ----------
    y: NDArray[np.float64]
        Data vector.
    """  # noqa: E501

    def __init__(self, y: NDArray[np.float64]) -> None:
        self.y = y
        self.m = y.size
        self.L = 1.0 / y.size

    def __str__(self) -> str:
        return "Logcosh"

    def get_spec(self) -> tuple:
        spec = (("y", float64[::1]), ("m", int32), ("L", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y)

    def value(self, x: NDArray[np.float64]) -> float:
        return np.sum(np.log(np.cosh(x - self.y))) / self.m

    def conjugate(self, x: NDArray[np.float64]) -> float:
        if np.max(np.abs(x)) > 1. / self.m:
            return np.inf
        else:
            z = np.arctanh(self.m * x) + self.y
        return np.sum(z * x) - np.sum(np.log(np.cosh(z - self.y))) / self.m

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.tanh(x - self.y) / self.m
