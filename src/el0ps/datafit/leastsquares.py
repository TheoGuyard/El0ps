import numpy as np
import pyomo.kernel as pmo
from numba import float64
from numpy.typing import NDArray

from el0ps.compilation import CompilableClass
from el0ps.datafit.base import BaseDatafit, MipDatafit


class Leastsquares(CompilableClass, BaseDatafit, MipDatafit):
    r"""Least-squares datafit function.

    The function is defined as

    .. math::

        f(\mathbf{w}) = \sum_{i=1}^m \tfrac{1}{2}(y_i - w_i)^2

    for some :math:`\mathbf{y} \in \mathbb{R}^m`.

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

    def value(self, w: NDArray) -> float:
        v = w - self.y
        return 0.5 * np.dot(v, v)

    def conjugate(self, w: NDArray) -> float:
        return 0.5 * np.dot(w, w) + np.dot(w, self.y)

    def gradient(self, w: NDArray) -> NDArray:
        return w - self.y

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
