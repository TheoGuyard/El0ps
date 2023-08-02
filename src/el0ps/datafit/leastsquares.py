import numpy as np
from numba import int32, float64
from numpy.typing import NDArray
from .base import ProximableDatafit, SmoothDatafit


class Leastsquares(ProximableDatafit, SmoothDatafit):
    """Least-squares data-fidelity function given by

    .. math:: f(x) = ||x - y||_2^2 / 2m

    where `m` is the size of vector `y`.

    Parameters
    ----------
    y: NDArray[np.float64]
        Data vector.
    """

    def __init__(self, y: NDArray[np.float64]) -> None:
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

    def value(self, x: NDArray[np.float64]) -> float:
        return np.linalg.norm(x - self.y, 2) ** 2 / (2.0 * self.m)

    def conjugate(self, x: NDArray[np.float64]) -> float:
        return (0.5 * self.m) * np.dot(x, x) + np.dot(x, self.y)

    def prox(self, x: NDArray[np.float64], eta: float) -> NDArray[np.float64]:
        return (x + (eta / self.m) * self.y) / (1.0 + eta / self.m)

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (x - self.y) / self.m