"""Miscellaneous utilities."""

import numpy as np
from numpy.typing import NDArray

from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty


def compute_lmbd_max(
    datafit: BaseDatafit, penalty: BasePenalty, A: NDArray
) -> float:
    r"""
    Return a value :math:`\lambda_{\max}` such that the all-zero vector is
    a solution of an L0-regularized problem whenever
    :math:`\lambda \geq \lambda_{\max}`.

    The problem is expressed as

    .. math::

        \textstyle\min_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{Ax}) + \lambda\|\mathbf{x}\|_0 + h(\mathbf{x})

    where :math:`f` is a :class:`el0ps.datafit.BaseDatafit` function,
    :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` is a matrix, :math:`h` is a
    :class:`el0ps.penalty.BasePenalty` function, and :math:`\lambda` is a
    positive scalar.

    Parameters
    ----------
    datafit: BaseDatafit
        Problem datafit function.
    penalty: BasePenalty
        Problem penalty function.
    A: NDArray
        Problem matrix.

    Returns
    -------
    lmbd_max: float
        The value ``lmbd_max`` ensuring an all-zero solution.
    """  # noqa: E501
    w = np.zeros(A.shape[0])
    v = np.abs(A.T @ datafit.gradient(w))
    i = np.argmax(v)
    return penalty.conjugate(i, v[i])
