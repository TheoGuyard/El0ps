import numpy as np
import pyomo.kernel as pmo
from numpy.typing import NDArray
from numba import float64

from el0ps.compilation import CompilableClass
from el0ps.penalty.base import SymmetricPenalty, MipPenalty


class Bigm(CompilableClass, SymmetricPenalty, MipPenalty):
    r"""Big-M :class:`BasePenalty` penalty function.

    The splitting terms are expressed as

    .. math::
        h_i(x_i) = \begin{cases}
        0 & \text{if } |x_i| \leq M \\
        +\infty & \text{otherwise}
        \end{cases}

    for some :math:`M > 0`.

    Parameters
    ----------
    M: float
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

    def value(self, i: int, x: float) -> float:
        return 0.0 if np.abs(x) <= self.M else np.inf

    def conjugate(self, i: int, x: float) -> float:
        return self.M * np.abs(x)

    def prox(self, i: int, x: float, eta: float) -> float:
        return np.maximum(np.minimum(x, self.M), -self.M)

    def subdiff(self, i: int, x: float) -> NDArray:
        if np.abs(x) < self.M:
            return [0.0, 0.0]
        elif x == -self.M:
            return [-np.inf, 0.0]
        elif x == self.M:
            return [0.0, np.inf]
        else:
            return [np.nan, np.nan]

    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        if x == 0.0:
            return [-self.M, self.M]
        else:
            s = np.sign(x) * self.M
            return [s, s]

    def param_slope(self, i: int, lmbd: float) -> float:
        return lmbd / self.M

    def param_limit(self, i: int, lmbd: float) -> float:
        return self.M

    def param_bndry(self, i, lmbd):
        return np.inf

    def bind_model(self, model: pmo.block) -> None:
        model.hpos_con = pmo.constraint_dict()
        model.hneg_con = pmo.constraint_dict()
        for i in model.N:
            model.hpos_con[i] = pmo.constraint(
                model.x[i] <= self.M * model.z[i]
            )
            model.hneg_con[i] = pmo.constraint(
                model.x[i] >= -self.M * model.z[i]
            )
        model.h_con = pmo.constraint(model.h >= 0.0)
