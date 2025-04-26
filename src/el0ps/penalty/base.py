"""Base classes for penalty functions and related utilities."""

import numpy as np
import pyomo.kernel as pmo
from abc import abstractmethod
from numpy.typing import NDArray


class BasePenalty:
    r"""Base class for penalty defined as separable functions.

    This class represent separable mathematical functions expressed as

    .. math::
        \begin{align*}
            h : \mathbb{R}^n &\rightarrow \mathbb{R} \cup \{+\infty\} \\
            \mathbf{x} &\mapsto h(\mathbf{x}) = \textstyle\sum_{i=1}^n h_i(x_i)
        \end{align*}


    where each splitting term :math:`h_i` is proper, lower-semicontinuous,
    convex, coercive, non-negative, and minimized at :math:`x_i = 0`.
    """

    @abstractmethod
    def value(self, i: int, x: float) -> float:
        """Value of the i-th splitting term of the penalty function at ``x``.

        Parameters
        ----------
        i : int
            Index of the splitting term.
        x : float
            Value at which the splitting term is evaluated.

        Returns
        -------
        value : float
            The value of the i-th splitting term of the function at ``x``.
        """
        ...

    @abstractmethod
    def conjugate(self, i: int, x: float) -> float:
        """Value of the convex conjugate of the i-th splitting term of the
        penalty function at ``x``.

        Parameters
        ----------
        i : int
            Index of the splitting term.
        x : float
            Value at which the convex conjugate is evaluated.

        Returns
        -------
        value : float
            The value of the convex conjugate.
        """
        ...

    @abstractmethod
    def prox(self, i: int, x: float, eta: float) -> float:
        """Proximity operator of the i-th splitting term of the penalty
        function weighted by ``eta`` at ``x``.

        Parameters
        ----------
        i : int
            Index of the splitting term.
        x : float
            Value at which the proximal operator is evaluated.
        eta : float, positive
            Multiplicative weight.

        Returns
        -------
        value : float
            The value of the proximity operator.
        """
        ...

    @abstractmethod
    def subdiff(self, i: int, x: float) -> NDArray:
        """Subdifferential of the i-th splitting term of the penalty function
        at ``x``, returned as an interval.

        Parameters
        ----------
        i : int
            Index of the splitting term.
        x : float
            Value at which the subdifferential is evaluated.

        Returns
        -------
        value : NDArray
            1D-array of size 2 containing the lower and upper bounds of the
            interval corresponding to the subdifferential.
        """
        ...

    @abstractmethod
    def conjugate_subdiff(self, i: int, x: float) -> NDArray:
        """Subdifferential of the conjugate of the i-th splitting term of the
        penalty function at ``x``, returned as an interval.

        Parameters
        ----------
        i : int
            Index of the splitting term.
        x : float
            Value at which the subdifferential of the conjugate is evaluated.

        Returns
        -------
        value : NDArray
            1D-array of size 2 containing the lower and upper bounds of the
            interval corresponding to the conjugate subdifferential.
        """
        ...

    def param_slope_pos(self, i: int, lmbd: float) -> float:
        """Supremum of the set ``{x in R | self.conjugate(i, x) <= lmbd}``.

        Parameters
        ----------
        i : int
            Index of the splitting term in the set definition.
        lmbd : float
            Threshold value in the set definition.

        Returns
        -------
        value : float
            The supremum of the set.
        """
        return compute_param_slope_pos(self, i, lmbd)

    def param_slope_neg(self, i: int, lmbd: float) -> float:
        """Infimum of the set ``{x in R | self.conjugate(i, x) <= lmbd}``.

        Parameters
        ----------
        i : int
            Index of the splitting term in the set definition.
        lmbd : float
            Threshold value in the set definition.

        Returns
        -------
        value : float
            The infimum of the set.
        """
        return compute_param_slope_neg(self, i, lmbd)

    def param_limit_pos(self, i: int, lmbd: float) -> float:
        """Supremum of the set ``self.conjugate_subdiff(i, tau)`` where
        ``tau = self.param_slope_pos(i, lmbd)``.

        Parameters
        ----------
        i : int
            Index of the splitting term in the definition of ``tau``.
        lmbd : float
            Threshold value in the definition of ``tau``.

        Returns
        -------
        value : float
            The supremum of the set.
        """
        s = self.conjugate_subdiff(i, self.param_slope_pos(i, lmbd))
        return np.inf if np.all(np.isnan(s)) else s[1]

    def param_limit_neg(self, i: int, lmbd: float) -> float:
        """Infimum of the set ``self.conjugate_subdiff(i, tau)`` where
        ``tau = self.param_slope_neg(i, lmbd)``.

        Parameters
        ----------
        i : int
            Index of the splitting term in the definition of ``tau``.
        lmbd : float
            Threshold value in the definition of ``tau``.

        Returns
        -------
        value : float
            The infimum of the set.
        """
        s = self.conjugate_subdiff(i, self.param_slope_neg(i, lmbd))
        return -np.inf if np.all(np.isnan(s)) else s[0]

    def param_bndry_pos(self, i: int, lmbd: float) -> float:
        """Supremum of the set ``self.subdiff(i, tau)`` where
        ``tau = self.param_limit_pos(i, lmbd)``.

        Parameters
        ----------
        i : int
            Index of the splitting term in the definition of ``tau``.
        lmbd : float
            Threshold value in the definition of ``tau``.

        Returns
        -------
        value : float
            The supremum of the set.
        """
        tau = self.param_limit_pos(i, lmbd)
        return np.inf if tau == np.inf else self.subdiff(i, tau)[1]

    def param_bndry_neg(self, i: int, lmbd: float) -> float:
        """Infimum of the set ``self.subdiff(i, tau)`` where
        ``tau = self.param_limit_neg(i, lmbd)``.

        Parameters
        ----------
        i : int
            Index of the splitting term in the definition of ``tau``.
        lmbd : float
            Threshold value in the definition of ``tau``.

        Returns
        -------
        value : float
            The supremum of the set.
        """
        tau = self.param_limit_neg(i, lmbd)
        return -np.inf if tau == -np.inf else self.subdiff(i, tau)[0]


class SymmetricPenalty(BasePenalty):
    """Base class for symmetric :class:`BasePenalty` functions."""

    @abstractmethod
    def param_slope(self, i: int, lmbd: float) -> float:
        """Supremum of the set ``{x in R | self.conjugate(i, x) <= lmbd}``.

        Parameters
        ----------
        i : int
            Index of the splitting term in the set definition.
        lmbd : float
            Threshold value in the set definition.

        Returns
        -------
        value : float
            The supremum of the set.
        """
        ...

    def param_slope_pos(self, i: int, lmbd: float) -> float:
        return self.param_slope(i, lmbd)

    def param_slope_neg(self, i: int, lmbd: float) -> float:
        return -self.param_slope(i, lmbd)

    def param_limit(self, i: int, lmbd: float) -> float:
        s = self.conjugate_subdiff(i, self.param_slope(i, lmbd))
        return np.inf if np.all(np.isnan(s)) else s[1]

    def param_limit_pos(self, i, lmbd):
        return self.param_limit(i, lmbd)

    def param_limit_neg(self, i, lmbd):
        return -self.param_limit(i, lmbd)

    def param_bndry(self, i: int, lmbd: float) -> float:
        tau = self.param_limit(i, lmbd)
        return np.inf if tau < np.inf else self.subdiff(i, tau)[1]

    def param_bndry_pos(self, i: int, lmbd: float) -> float:
        return self.param_bndry(i, lmbd)

    def param_bndry_neg(self, i: int, lmbd: float) -> float:
        return -self.param_bndry(i, lmbd)


class MipPenalty:
    """Base class for penalty functions that can be modeled into
    `pyomo <https://pyomo.readthedocs.io/en/stable/>`_."""

    @abstractmethod
    def bind_model(self, model: pmo.block) -> None:
        """Impose an constraint associated with the penalty function in a
        `pyomo <https://pyomo.readthedocs.io/en/stable/>`_ model.

        Given a pyomo.kernel.block ``model`` object containing a real scalar
        variable ``model.h`` and a real vector variable ``model.x`` of size
        ``model.N``, this function is intended to impose the relations

        ``model.h >= self.value(model.x)``

        and

        ``model.z[i] = 0 ==> model.x[i] = 0 for all i in model.N``

        using ``pyomo`` expressions.

        Parameters
        ----------
        model : pmo.block
            The pyomo kernel model.
        lmbd : float
            The scalar parameter involved in the expressions to be modeled.
        """
        ...


def compute_param_slope_pos(
    penalty: BasePenalty,
    i: int,
    lmbd: float,
    tol: float = 1e-8,
    maxit: int = 100,
) -> float:
    """Utility to approximate the value of the function
    ``penalty.param_slope_pos`` for a given :class:`BasePenalty` instance using
    a bisection method.

    Parameters
    ----------
    penalty : BasePenalty
        The penalty function.
    i : int
        Parameter involved in the ``penalty.param_slope_pos`` function.
    lmbd : float
        Parameter involved in the ``penalty.param_slope_pos`` function.
    tol : float = 1e-4
        Bisection approximation tolerance.
    maxit : int = 100
        Maximum number of bisection iterations.
    """
    a = 0.0
    b = 1.0
    while penalty.conjugate(i, b) < lmbd:
        b *= 2.0
        if b > 1e12:
            return np.inf
    for _ in range(maxit):
        c = (a + b) / 2.0
        fa = penalty.conjugate(i, a) - lmbd
        fc = penalty.conjugate(i, c) - lmbd
        if (-tol <= fc <= tol) or (b - a < 0.5 * tol):
            return c
        elif fc * fa >= 0.0:
            a = c
        else:
            b = c
    return c


def compute_param_slope_neg(
    penalty: BasePenalty,
    i: int,
    lmbd: float,
    tol: float = 1e-8,
    maxit: int = 100,
) -> float:
    """Utility to approximate the value of the function
    ``penalty.param_slope_neg`` for a given :class:`BasePenalty` instance using
    a bisection method.

    Parameters
    ----------
    penalty : BasePenalty
        The penalty function.
    i : int
        Parameter involved in the ``penalty.param_slope_neg`` function.
    lmbd : float
        Parameter involved in the ``penalty.param_slope_neg`` function.
    tol : float = 1e-4
        Bisection approximation tolerance.
    maxit : int = 100
        Maximum number of bisection iterations.
    """
    a = -1.0
    b = 0.0
    while penalty.conjugate(i, a) < lmbd:
        a *= 2.0
        if a < -1e12:
            return -np.inf
    for _ in range(maxit):
        c = (a + b) / 2.0
        fa = penalty.conjugate(i, a) - lmbd
        fc = penalty.conjugate(i, c) - lmbd
        if (-tol <= fc <= tol) or (b - a < 0.5 * tol):
            return c
        elif fc * fa >= 0.0:
            a = c
        else:
            b = c
    return c
