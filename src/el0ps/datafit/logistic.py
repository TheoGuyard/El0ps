import numpy as np
import pyomo.kernel as pmo
from numba import float64
from numpy.typing import ArrayLike

from el0ps.compilation import CompilableClass

from .base import BaseDatafit, MipDatafit


class Logistic(CompilableClass, BaseDatafit, MipDatafit):
    r"""Logistic datafit function.

    The function is defined as

    .. math:: f(x) = 1^T \log(1 + \exp(y \odot x))

    where :math:`\odot` denotes the elementwise product.

    Parameters
    ----------
    y: ArrayLike
        Data vector.
    """

    def __init__(self, y: ArrayLike) -> None:
        self.y = y
        self.L = 0.25

    def __str__(self) -> str:
        return "Logistic"

    def get_spec(self) -> tuple:
        spec = (("y", float64[::1]), ("L", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y)

    def value(self, x: ArrayLike) -> float:
        return np.sum(np.log(1.0 + np.exp(-self.y * x)))

    def conjugate(self, x: ArrayLike) -> float:
        c = -(x * self.y)
        if not np.all((0.0 < c) & (c < 1.0)):
            return np.inf
        r = 1.0 - c
        return np.dot(c, np.log(c)) + np.dot(r, np.log(r))

    def gradient(self, x: ArrayLike) -> ArrayLike:
        return -self.y / (1.0 + np.exp(self.y * x))

    def gradient_lipschitz_constant(self) -> float:
        return self.L

    def bind_model(self, model: pmo.block) -> None:
        model.c1_var = pmo.variable_dict()
        model.c2_var = pmo.variable_dict()
        model.c3_var = pmo.variable_dict()
        model.f1_var = pmo.variable_dict()
        model.f2_var = pmo.variable_dict()
        model.f3_var = pmo.variable_dict()
        for j in model.M:
            model.c1_var[j] = pmo.variable(domain=pmo.Reals)
            model.c2_var[j] = pmo.variable(domain=pmo.Reals)
            model.c3_var[j] = pmo.variable(domain=pmo.Reals)
            model.f1_var[j] = pmo.variable(domain=pmo.Reals)
            model.f2_var[j] = pmo.variable(domain=pmo.Reals)
            model.f3_var[j] = pmo.variable(domain=pmo.Reals)

        model.c1_con = pmo.constraint_dict()
        model.c2_con = pmo.constraint_dict()
        model.c3_con = pmo.constraint_dict()
        model.f1f2_con = pmo.constraint_dict()
        model.f1f3_con = pmo.constraint_dict()
        model.f2f3_con = pmo.constraint_dict()
        for j in model.M:
            model.c1_con[j] = pmo.constraint(model.c1_var[j] == 1.0)
            model.c2_con[j] = pmo.constraint(
                model.c2_var[j] == self.y[j] * model.w[j] - model.f1_var[j]
            )
            model.c3_con[j] = pmo.constraint(
                model.c3_var[j] == -model.f1_var[j]
            )
            model.f1f2_con[j] = pmo.conic.primal_exponential(
                model.f2_var[j],
                model.c1_var[j],
                model.c2_var[j],
            )
            model.f1f3_con[j] = pmo.conic.primal_exponential(
                model.f3_var[j],
                model.c1_var[j],
                model.c3_var[j],
            )
            model.f2f3_con[j] = pmo.constraint(
                model.f2_var[j] + model.f3_var[j] <= 1.0
            )
        model.f_con = pmo.constraint(
            model.f >= sum(model.f1_var[j] for j in model.M)
        )
