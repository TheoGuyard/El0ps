import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import SymmetricPenalty, MipPenalty


class BigmL1norm(CompilableClass, SymmetricPenalty, MipPenalty):
    r"""Big-M plus L1-norm :class:`BasePenalty` penalty function.

    The splitting terms are expressed as

    .. math::
        h_i(x_i) = \begin{cases}
        \alpha|x_i| & \text{if } |x_i| \leq M \\
        +\infty & \text{otherwise}
        \end{cases}

    for some :math:`M > 0` and :math:`\alpha > 0`.

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
        return "BigmL1norm"

    def get_spec(self) -> tuple:
        spec = (
            ("M", float64),
            ("alpha", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(M=self.M, alpha=self.alpha)

    def value(self, i: int, x: float) -> float:
        xabs = np.abs(x)
        return self.alpha * xabs if xabs <= self.M else np.inf

    def conjugate(self, i: int, x: float) -> float:
        return self.M * np.maximum(np.abs(x) - self.alpha, 0.0)

    def prox(self, i: int, x: float, eta: float) -> float:
        v = np.abs(x) - eta * self.alpha
        return np.sign(x) * np.maximum(np.minimum(v, self.M), 0.0)

    def subdiff(self, i: int, x: float) -> NDArray:
        if x == 0.0:
            return [-self.alpha, self.alpha]
        elif np.abs(x) < self.M:
            return [self.alpha, self.alpha]
        elif x == -self.M:
            return [-np.inf, -self.alpha]
        elif x == self.M:
            return [self.alpha, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        if np.abs(x) < self.alpha:
            return [0.0, 0.0]
        elif x == self.alpha:
            return [0.0, self.M]
        elif x == -self.alpha:
            return [-self.M, 0.0]
        else:
            s = np.sign(x) * self.M
            return [s, s]

    def param_slope(self, i: int, lmbd: float) -> float:
        return (lmbd / self.M) + self.alpha

    def param_limit(self, i: int, lmbd: float) -> float:
        return self.M

    def param_bndry(self, i, lmbd):
        return np.inf

    def bind_model(self, model: pmo.block) -> None:

        model.h1_var = pmo.variable_dict()
        for i in model.N:
            model.h1_var[i] = pmo.variable(domain=pmo.NonNegativeReals)

        model.hpos_con = pmo.constraint_dict()
        model.hneg_con = pmo.constraint_dict()
        model.h1pos_con = pmo.constraint_dict()
        model.h1neg_con = pmo.constraint_dict()
        for i in model.N:
            model.hpos_con[i] = pmo.constraint(
                model.x[i] <= self.M * model.z[i]
            )
            model.hneg_con[i] = pmo.constraint(
                model.x[i] >= -self.M * model.z[i]
            )
            model.h1pos_con[i] = pmo.constraint(model.h1_var[i] >= model.x[i])
            model.h1neg_con[i] = pmo.constraint(model.h1_var[i] >= -model.x[i])
        model.h_con = pmo.constraint(
            model.h >= self.alpha * sum(model.h1_var[i] for i in model.N)
        )
