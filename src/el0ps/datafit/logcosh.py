import numpy as np
from numba import float64
from numpy.typing import NDArray

from el0ps.compilation import CompilableClass
from el0ps.datafit.base import BaseDatafit


class Logcosh(CompilableClass, BaseDatafit):
    r"""Logcosh datafit function.

    The function is defined as

    .. math::

        f(\mathbf{w}) = \sum_{i=1}^m \log(\cosh(y_i - w_i))

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
        return "Logcosh"

    def get_spec(self) -> tuple:
        spec = (("y", float64[::1]), ("L", float64))
        return spec

    def params_to_dict(self) -> dict:
        return dict(y=self.y)

    def value(self, w: NDArray) -> float:
        return np.sum(np.log(np.cosh(w - self.y)))

    def conjugate(self, w: NDArray) -> float:
        if np.max(np.abs(w)) > 1.0:
            return np.inf
        else:
            z = np.arctanh(w) + self.y
        return np.sum(z * w) - np.sum(np.log(np.cosh(z - self.y)))

    def gradient(self, w: NDArray) -> NDArray:
        return np.tanh(w - self.y)

    def gradient_lipschitz_constant(self) -> float:
        return self.L
