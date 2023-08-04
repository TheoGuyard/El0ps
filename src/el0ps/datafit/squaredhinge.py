import numpy as np
from numba import int32, float64
from numpy.typing import NDArray
from .base import SmoothDatafit


class Squaredhinge(SmoothDatafit):
    """Squared-Hinge data-fidelity function given by

    .. math:: f(x) = ||max(1 - y * x, 0.)||_2^2 / m

    where `m` is the size of vector `y` and `*` is the element-wise product.

    Parameters
    ----------
    y: NDArray[np.float64]
        Data vector.
    """

    def __init__(self, y: NDArray[np.float64]) -> None:
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

    def value(self, x: NDArray[np.float64]) -> float:
        return (
            np.linalg.norm(np.maximum(1.0 - self.y * x, 0.0), 2) ** 2 / self.m
        )

    def conjugate(self, x: NDArray[np.float64]) -> float:
        return (
            (0.5 * self.m) * np.dot(x, x)
            + np.dot(self.y, x)
            - np.linalg.norm(np.maximum(-0.5 * self.m * (self.y * x), 0.0), 2)
            ** 2
            / self.m
        )

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return -self.y * np.maximum(1.0 - self.y * x, 0.0) / (0.5 * self.m)
