import numpy as np
from numba import float64
from numpy.typing import NDArray
from .base import ProximablePenalty


class Bigm(ProximablePenalty):
    """Big-M penalty function.

    The Big-M penalty function reads:

    .. math:: h(x) = 0 if ||x||_inf <= M and +inf otherwise

    where `M` is a positive hyperparameter.

    Parameters
    ----------
    M : float
        Big-M value.
    """

    def __init__(self, M: float) -> None:
        self.M = M

    def __str__(self) -> str:
        return "Bigm"

    def get_spec(self) -> tuple:
        spec = (("M", float64),)
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M)

    def value_scalar(self, i: int, x: float) -> float:
        return 0.0 if np.abs(x) <= self.M else np.inf

    def conjugate_scalar(self, i: int, x: float) -> float:
        return self.M * np.abs(x)

    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        return np.maximum(np.minimum(x, self.M), -self.M)

    # Overload `value` function for faster evaluation
    def value(self, x: NDArray) -> float:
        return 0.0 if np.linalg.norm(x, np.inf) <= self.M else np.inf

    # Overload `conjugate` function for faster evaluation
    def conjugate(self, x: NDArray) -> float:
        return self.M * np.linalg.norm(x, 1)

    # Overload `prox` function for faster evaluation
    def prox(self, x: NDArray, eta: float) -> NDArray:
        return np.maximum(np.minimum(x, self.M), -self.M)

    def param_zerlimit(self, i: int) -> float:
        return 0.0

    def param_domlimit(self, i: int) -> float:
        return np.inf

    def param_vallimit(self, i: int) -> float:
        return np.inf

    def param_levlimit(self, i: int, lmbd: float) -> float:
        return lmbd / self.M

    def param_sublimit(self, i: int, lmbd: float) -> float:
        return self.M
