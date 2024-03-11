import numpy as np
from numba import int32, float64
from numpy.typing import ArrayLike
from .base import SmoothDatafit


class Kullbackleibler(SmoothDatafit):
    r"""Kullback-Leibler datafit function given by

    .. math:: f(x) = 1 / m \sum_(j=1)^m y_ij \log(y_j / (x_j + e)) + x_j + e - y_j

    where ``m`` is the size of the vector ``y`` and ``e`` is a smoothing
    parameter.

    Parameters
    ----------
    y: ArrayLike
        Data vector.
    e: float = 1e-6
        Smoothing parameter, positive.
    """  # noqa: E501

    def __init__(self, y: ArrayLike, e: float = 1e-6) -> None:
        self.y = y
        self.e = e
        self.m = y.size
        self.L = np.max(y) / (self.m * e**2)
        self.log_yy = np.log(y * y)

    def __str__(self) -> str:
        return "Kullbackleibler"

    def get_spec(self) -> tuple:
        spec = (
            ("y", float64[::1]),
            ("e", float64),
            ("m", int32),
            ("L", float64),
            ("log_yy", float64[::1]),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y, e=self.e)

    def value(self, x: ArrayLike) -> float:
        z = np.maximum(x, 0.0) + self.e
        return np.sum(self.y * np.log(self.y / z) + z - self.y) / self.m

    def conjugate(self, x: ArrayLike) -> float:
        u = self.m * x
        v = 1.0 - u
        if np.any(v <= 0.0):
            return np.inf
        return np.sum(self.y * (self.log_yy - np.log(v)) - self.e * u) / self.m

    def lipschitz_constant(self) -> float:
        return self.L

    def gradient(self, x: ArrayLike) -> ArrayLike:
        z = np.maximum(x, 0.0) + self.e
        return (1.0 - self.y / z) / self.m
