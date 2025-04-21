import numpy as np
import pyomo.kernel as pmo
from numba import float64
from numpy.typing import NDArray

from el0ps.compilation import CompilableClass
from el0ps.datafit.base import BaseDatafit, MipDatafit


class Leastsquares(CompilableClass, BaseDatafit, MipDatafit):
    """Least-squares datafit function defined as

    ``f(w) = sum_{i=1,...,m} 0.5 * (yi - wi)^2``

    where ``y in R^m`` is a vector.

    Parameters
    ----------
    y : NDArray
        Data vector.
    """

    def __init__(self, y: NDArray) -> None:
        self.y = y
        self.L = 1.0

    def __str__(self) -> str:
        return "Leastsquares"

    def get_spec(self) -> tuple:
        spec = (("y", float64[::1]), ("L", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y)

    def value(self, x: NDArray) -> float:
        v = x - self.y
        return 0.5 * np.dot(v, v)

    def conjugate(self, x: NDArray) -> float:
        return 0.5 * np.dot(x, x) + np.dot(x, self.y)

    def gradient(self, x: NDArray) -> NDArray:
        return x - self.y

    def gradient_lipschitz_constant(self) -> float:
        return self.L

    def bind_model(self, model: pmo.block) -> None:
        model.c_var = pmo.variable()
        model.c_con = pmo.constraint(model.c_var == 1.0)
        model.r_var = pmo.variable_dict()
        model.r_con = pmo.constraint_dict()
        for j in model.M:
            model.r_var[j] = pmo.variable(domain=pmo.Reals)
            model.r_con[j] = pmo.constraint(
                model.r_var[j] == model.w[j] - self.y[j]
            )
        model.f_con = pmo.conic.rotated_quadratic(
            model.f,
            model.c_var,
            [model.r_var[j] for j in model.M],
        )
