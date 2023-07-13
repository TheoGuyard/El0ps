import gurobipy as gp
import numpy as np
from numpy.typing import NDArray
from gurobipy import MVar, Model, Var
from .base import ProximablePenalty


class L1norm(ProximablePenalty):
    """L1-norm penalty function.

    The L1-norm penalty function reads:

    .. math:: f(x) = alpha * ||x||_1

    where `alpha` is a positive hyperparameter.

    Parameters
    ----------
    alpha : float, positive
        L1-norm weight.
    """

    def __init__(self, alpha: float) -> None:
        if not isinstance(alpha, float):
            alpha = float(alpha)
        if alpha <= 0.0:
            raise ValueError("Parameter `alpha` must be positive.")
        self.alpha = alpha

    def __str__(self) -> str:
        return "L1-norm"

    def value_scalar(self, i: int, x: float) -> float:
        return self.alpha * np.abs(x)

    def conjugate_scalar(self, i: int, x: float) -> float:
        return 0.0 if np.abs(x) <= self.alpha else np.inf

    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        return np.sign(x) * np.maximum(0.0, np.abs(x) - eta)

    # Overload `value` function for faster evaluation
    def value(self, x: NDArray) -> float:
        return self.alpha * np.sum(np.abs(x))

    # Overload `conjugate` function for faster evaluation
    def conjugate(self, x: NDArray) -> float:
        return 0.0 if np.all(np.abs(x) <= self.alpha) else np.inf

    # Overload `prox` function for faster evaluation
    def prox(self, x: NDArray, eta: float) -> NDArray:
        return np.sign(x) * np.maximum(0.0, np.abs(x) - eta)

    def param_zerlimit(self, i: int) -> float:
        return self.alpha

    def param_domlimit(self, i: int) -> float:
        return self.alpha

    def param_vallimit(self, i: int) -> float:
        return 0.0

    def param_levlimit(self, i: int, lmbd: float) -> float:
        return self.alpha

    def param_sublimit(self, i: int, lmbd: float) -> float:
        return np.inf

    def bind_model_cost(
        self, model: Model, lmbd: float, x_var: MVar, z_var: MVar, g_var: Var
    ) -> None:
        raise NotImplementedError


class L2norm(ProximablePenalty):
    """L2-norm penalty function.

    The L2-norm penalty function reads:

    .. math:: f(x) = alpha * ||x||_2^2

    where `alpha` is a positive hyperparameter.

    Parameters
    ----------
    alpha : float, positive
        L2-norm weight.
    param_slope_scalar : float
        Working value.
    param_limit_scalar : float
        Working value.
    """

    def __init__(self, alpha: float) -> None:
        if not isinstance(alpha, float):
            alpha = float(alpha)
        if alpha <= 0.0:
            raise ValueError("Parameter `alpha` must be positive.")
        self.alpha = alpha

    def __str__(self) -> str:
        return "L2-norm"

    def value_scalar(self, i: int, x: float) -> float:
        return self.alpha * x**2

    def conjugate_scalar(self, i: int, x: float) -> float:
        return x**2 / (4.0 * self.alpha)

    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        return x / (1.0 + 2.0 * eta * self.alpha)

    # Overload `value` function for faster evaluation
    def value(self, x: NDArray) -> float:
        return self.alpha * np.dot(x, x)

    # Overload `conjugate` function for faster evaluation
    def conjugate(self, x: NDArray) -> float:
        return np.dot(x, x) / (4.0 * self.alpha)

    # Overload `prox` function for faster evaluation
    def prox(self, x: NDArray, eta: float) -> NDArray:
        return x / (1.0 + 2.0 * eta * self.alpha)

    def param_zerlimit(self, i: int) -> float:
        return 0.0

    def param_domlimit(self, i: int) -> float:
        return np.inf

    def param_vallimit(self, i: int) -> float:
        return np.inf

    def param_levlimit(self, i: int, lmbd: float) -> float:
        return 2.0 * np.sqrt(lmbd * self.alpha)

    def param_sublimit(self, i: int, lmbd: float) -> float:
        return np.sqrt(lmbd / self.alpha)

    def bind_model_cost(
        self, model: Model, lmbd: float, x_var: MVar, z_var: MVar, g_var: Var
    ) -> None:
        s_var = model.addMVar(x_var.size, vtype=gp.GRB.CONTINUOUS, name="s")
        model.addConstr(s_var >= 0.0)
        model.addConstr(x_var * x_var <= s_var * z_var)
        model.addConstr(g_var >= lmbd * sum(z_var) + self.alpha * sum(s_var))
