import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import BasePenalty, MipPenalty


class BigmPositiveL1norm(CompilableClass, BasePenalty, MipPenalty):
    """Positive big-M plus L1-norm penalty function expressed as

    ``h(x) = sum_{i = 1,...,n} hi(xi)``

    where ``hi(x) = alpha * xi`` if ``0 <= xi <= M`` and ``h_i(xi) = inf``
    otherwise for some ``M > 0`` and ``alpha > 0``.

    Parameters
    ----------
    M: float
        Big-M value.
    alpha: float
        L1-norm weight.
    """

    def __init__(self, M: float, alpha: float) -> None:
        self.M = M
        self.alpha = alpha

    def __str__(self) -> str:
        return "BigmPositiveL1norm"

    def get_spec(self) -> tuple:
        spec = (
            ("M", float64),
            ("alpha", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M, alpha=self.alpha)

    def value(self, i: int, x: float) -> float:
        return self.alpha * x if (x >= 0.0) and (x <= self.M) else np.inf

    def conjugate(self, i: int, x: float) -> float:
        return self.M * np.maximum(x - self.alpha, 0.0)

    def prox(self, i: int, x: float, eta: float) -> float:
        return np.maximum(0.0, np.minimum(x - eta * self.alpha, self.M))

    def subdiff(self, i: int, x: float) -> NDArray:
        if x == 0.0:
            return [-np.inf, self.alpha]
        elif 0.0 < x < self.M:
            s = self.alpha
            return [s, s]
        elif x == self.M:
            return [self.alpha, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        if x < self.alpha:
            return [0.0, 0.0]
        elif x == self.alpha:
            return [0.0, self.M]
        else:
            s = np.sign(x) * self.M
            return [s, s]

    def param_slope_pos(self, i: int, lmbd: float) -> float:
        return (lmbd / self.M) + self.alpha

    def param_slope_neg(self, i: int, lmbd: float) -> float:
        return -np.inf

    def param_limit_pos(self, i: int, lmbd: float) -> float:
        return self.M

    def param_limit_neg(self, i: int, lmbd: float) -> float:
        return 0.0

    def param_bndry_pos(self, i: int, lmbd: float) -> float:
        return np.inf

    def param_bndry_neg(self, i: int, lmbd: float) -> float:
        return -np.inf

    def bind_model(self, model: pmo.block, lmbd: float) -> None:

        model.g1_var = pmo.variable_dict()
        for i in model.N:
            model.g1_var[i] = pmo.variable(domain=pmo.NonNegativeReals)

        model.gpos_con = pmo.constraint_dict()
        model.gneg_con = pmo.constraint_dict()
        model.g1pos_con = pmo.constraint_dict()
        model.g1neg_con = pmo.constraint_dict()
        for i in model.N:
            model.gpos_con[i] = pmo.constraint(
                model.x[i] <= self.M * model.z[i]
            )
            model.gneg_con[i] = pmo.constraint(model.x[i] >= 0)
            model.g1pos_con[i] = pmo.constraint(model.g1_var[i] >= model.x[i])
            model.g1neg_con[i] = pmo.constraint(model.g1_var[i] >= 0.0)
        model.g_con = pmo.constraint(
            model.g
            >= (
                lmbd * sum(model.z[i] for i in model.N)
                + self.alpha * sum(model.g1_var[i] for i in model.N)
            )
        )
