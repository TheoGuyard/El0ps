import numpy as np
from numba import int32, float64
from numpy.typing import NDArray
from .base import ProximableDatafit, SmoothDatafit


class Quadratic(ProximableDatafit, SmoothDatafit):
    """Quadratic data-fidelity function

    .. math:: f(x) = ||x - y||_2^2 / 2m

    where `m` is the size of vector `y`.

    Parameters
    ----------
    y : NDArray
        Target vector.
    """

    def __init__(self, y: NDArray) -> None:
        self.y = y
        self.m = y.size
        self.L = 1.0 / y.size

    def __str__(self) -> str:
        return "Quadratic"

    def get_spec(self) -> tuple:
        spec = (("y", float64[::1]), ("m", int32), ("L", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y)

    def value(self, x: NDArray) -> float:
        return np.linalg.norm(x - self.y, 2) ** 2 / (2.0 * self.m)

    def conjugate(self, x: NDArray) -> float:
        return (0.5 * self.m) * np.dot(x, x) + np.dot(x, self.y)

    def prox(self, x: NDArray, eta: float) -> NDArray:
        return (x + (eta / self.m) * self.y) / (1.0 + eta / self.m)

    def gradient(self, x: NDArray) -> NDArray:
        return (x - self.y) / self.m
