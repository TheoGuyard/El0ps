import gurobipy as gp
from gurobipy import MVar, Model, Var
import numpy as np
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

    L = np.nan

    def __init__(self, y: NDArray) -> None:
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.ndim != 1:
            raise ValueError("Vector `y` must be a one-dimensional array.")
        if y.size == 0:
            raise ValueError("Vector `y` is empty.")

        self.y = y
        self.m = y.shape[0]
        self.L = 1.0 / y.shape[0]

    def __str__(self) -> str:
        return "Quadratic"

    def value(self, x: NDArray) -> float:
        return np.linalg.norm(x - self.y, 2) ** 2 / (2.0 * self.m)

    def conjugate(self, x: NDArray) -> float:
        return 0.5 * self.m * np.dot(x, x) + np.dot(self.y, x)

    def prox(self, x: NDArray, eta: float) -> NDArray:
        return (x + (eta / self.m) * self.y) / (1.0 + eta / self.m)

    def gradient(self, x: NDArray) -> NDArray:
        return (x - self.y) / self.m

    def bind_model_cost(
        self, model: Model, A: NDArray, x_var: MVar, f_var: Var
    ) -> None:
        r_var = self.y - A @ x_var
        model.addConstr(
            f_var >= gp.quicksum(ri * ri for ri in r_var) / (2.0 * self.y.size)
        )
