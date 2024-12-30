import numpy as np
from numba import float64
from numpy.typing import ArrayLike
from el0ps.compilation import CompilableClass

from .base import BaseDatafit


class KullbackLeibler(CompilableClass, BaseDatafit):
    r"""Kullback-Leibler datafit function.

    The function is defined as

    .. math:: f(w) = y^T (\log(\frac{y}{w + e}) + (w + e) - y)

    where :math:`e > 0` is a smoothing parameter.

    Parameters
    ----------
    y: ArrayLike
        Data vector.
    e: float = 1e-8
        Positive smoothing parameter.
    """  # noqa: E501

    def __init__(self, y: ArrayLike, e: float = 1e-8) -> None:
        self.y = y
        self.e = e
        self.L = np.max(y) / e**2
        self.log_yy = np.log(y * y)

    def __str__(self) -> str:
        return "KullbackLeibler"

    def get_spec(self) -> tuple:
        spec = (
            ("y", float64[::1]),
            ("e", float64),
            ("L", float64),
            ("log_yy", float64[::1]),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y, e=self.e)

    def value(self, w: ArrayLike) -> float:
        z = np.maximum(w, 0.0) + self.e
        return np.sum(self.y * np.log(self.y / z) + z - self.y)

    def conjugate(self, w: ArrayLike) -> float:
        u = w
        v = 1.0 - u
        if np.any(v <= 0.0):
            return np.inf
        return np.sum(self.y * (self.log_yy - np.log(v)) - self.e * u)

    def gradient(self, w: ArrayLike) -> ArrayLike:
        z = np.maximum(w, 0.0) + self.e
        return 1.0 - self.y / z

    def gradient_lipschitz_constant(self) -> float:
        return self.L
