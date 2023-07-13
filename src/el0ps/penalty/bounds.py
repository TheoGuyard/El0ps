from gurobipy import MVar, Model, Var
import numpy as np
from numpy.typing import NDArray
from .base import ProximablePenalty


class Bigm(ProximablePenalty):
    """Big-M penalty function.

    The Big-M penalty function reads:

    .. math:: f(x) = 0 if ||x||_inf <= M and +inf otherwise

    where `M` is a positive hyperparameter.

    Parameters
    ----------
    M : float, positive
        Big-M value.
    """

    def __init__(self, M: float) -> None:
        if not isinstance(M, float):
            M = float(M)
        if M <= 0.0:
            raise ValueError("Parameter `M` must be positive.")
        self.M = M

    def __str__(self) -> str:
        return "Big-M"

    def value_scalar(self, i: int, x: float) -> float:
        return 0.0 if np.abs(x) <= self.M else np.inf

    def conjugate_scalar(self, i: int, x: float) -> float:
        return self.M * np.abs(x)

    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        return np.clip(x, -self.M, self.M)

    # Overload `value` function for faster evaluation
    def value(self, x: NDArray) -> float:
        return 0.0 if np.linalg.norm(x, np.inf) <= self.M else np.inf

    # Overload `conjugate` function for faster evaluation
    def conjugate(self, x: NDArray) -> float:
        return self.M * np.linalg.norm(x, 1)

    # Overload `prox` function for faster evaluation
    def prox(self, x: NDArray, eta: float) -> NDArray:
        return np.clip(x, -self.M, self.M)

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

    def bind_model_cost(
        self, model: Model, lmbd: float, x_var: MVar, z_var: MVar, g_var: Var
    ) -> None:
        model.addConstr(g_var >= lmbd * sum(z_var))
        model.addConstr(x_var <= self.M * z_var)
        model.addConstr(x_var >= -self.M * z_var)
