import numpy as np
import pyomo.kernel as pmo
from numba import float64
from numpy.typing import NDArray

from el0ps.compilation import CompilableClass
from el0ps.datafit.base import BaseDatafit, MipDatafit


class Squaredhinge(CompilableClass, BaseDatafit, MipDatafit):
    r"""Squared-hinge datafit function.

    The function is defined as

    .. math::

        f(\mathbf{w}) = \sum_{i=1}^m \max(1 - y_i w_i, 0)^2

    for some :math:`\mathbf{y} \in \mathbb{R}^m`.

    Parameters
    ----------
    y : NDArray
        Data vector.
    """

    def __init__(self, y: NDArray) -> None:
        self.y = y
        self.L = 2.0

    def __str__(self) -> str:
        return "Squaredhinge"

    def get_spec(self) -> tuple:
        spec = (("y", float64[::1]), ("L", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y)

    def value(self, w: NDArray) -> float:
        v = np.maximum(1.0 - self.y * w, 0.0)
        return np.dot(v, v)

    def conjugate(self, w: NDArray) -> float:
        v = np.maximum(-0.5 * (self.y * w), 0.0)
        return 0.5 * np.dot(w, w) + np.dot(self.y, w) - np.dot(v, v)

    def gradient(self, w: NDArray) -> NDArray:
        return -2.0 * self.y * np.maximum(1.0 - self.y * w, 0.0)

    def gradient_lipschitz_constant(self) -> float:
        return self.L

    def bind_model(self, model: pmo.block) -> None:
        model.f1_var = pmo.variable_dict()
        model.f1_con = pmo.constraint_dict()
        for j in model.M:
            model.f1_var[j] = pmo.variable(domain=pmo.NonNegativeReals)
            model.f1_con[j] = pmo.constraint(
                model.f1_var[j] >= 1.0 - self.y[j] * model.w[j]
            )
        model.c_var = pmo.variable()
        model.c_con = pmo.constraint(model.c_var == 0.5)
        model.f_con = pmo.conic.rotated_quadratic(
            model.f,
            model.c_var,
            [model.f1_var[j] for j in model.M],
        )
