"""Base classes for penalty functions and related utilities."""

import numpy as np
import pyomo.kernel as pmo
from abc import abstractmethod
from numpy.typing import ArrayLike


class BasePenalty:
    """Base class for penalty functions. Penalty functions are assumed to be
    proper, closed, convex, separable, even, coercive, non-negative and
    minimized at 0.
    """

    @abstractmethod
    def value_scalar(self, i: int, x: float) -> float:
        """Value of the i-th splitting term of the function at ``x``.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        x: float
            Value at which the function is evaluated.

        Returns
        -------
        value: float
            The value of the function at ``x``.
        """
        ...

    @abstractmethod
    def conjugate_scalar(self, i: int, x: float) -> float:
        """Value of the conjugate of the i-th splitting term of the function
        at ``x``.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        x: float
            Value at which the conjugate is evaluated.

        Returns
        -------
        value: float
            The value of the conjugate of the function at ``x``.
        """
        ...

    @abstractmethod
    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        """Proximity operator of ``eta`` times the i-th splitting term of the
        function at ``x``.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        x: float
            Value at which the prox is evaluated.
        eta: float, positive
            Multiplicative factor of the function.

        Returns
        -------
        p: float
            The proximity operator of ``eta`` times the function at ``x``.
        """
        ...

    @abstractmethod
    def subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        """Subdifferential operator of the i-th splitting term of the function
        at ``x``.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        x: float
            Value at which the prox is evaluated.

        Returns
        -------
        s: ArrayLike
            The subdifferential (interval) of the function at ``x``.
        """
        ...

    @abstractmethod
    def conjugate_subdiff_scalar(self, i: int, x: float) -> ArrayLike:
        """Subdifferential operator of the i-th splitting term of the function
        conjugate at ``x``.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        x: float
            Value at which the prox is evaluated.

        Returns
        -------
        s: ArrayLike
            The subdifferential (interval) of the function conjugate at ``x``.
        """
        ...

    @abstractmethod
    def param_slope_pos_scalar(self, i: int, lmbd: float) -> float:
        """Value of

        .. math:: sup_x { x in R | h^*_i(x) <= lmbd }

        where :math:`h^*_i` is the conjugate of the i-th splitting term of the
        penalty function.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        lmbd: float
            Threshold value.

        Returns
        -------
        value: float
            The maximum value of ``x`` such that the conjugate function is
            below ``lmbd``.
        """
        ...

    @abstractmethod
    def param_slope_neg_scalar(self, i: int, lmbd: float) -> float:
        """Value of

        .. math:: inf_x { x in R | h^*_i(x) <= lmbd }

        where :math:`h^*_i` is the conjugate of the i-th splitting term of the
        penalty function.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        lmbd: float
            Threshold value.

        Returns
        -------
        value: float
            The minimum value of ``x`` such that the conjugate function is
            below ``lmbd``.
        """
        ...

    @abstractmethod
    def param_limit_pos_scalar(self, i: int, lmbd: float) -> float:
        """Value of

        .. math:: sup_x { x in subdiff(h^*_i)(t) }

        where :math:`h^*_i` is the conjugate of the i-th splitting term of the
        penalty function and :math:`t` is the value of
        `self.param_slope_pos_scalar(i, lmbd)`.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        lmbd: float
            Argument of the function `self.param_slope_pos_scalar`.

        Returns
        -------
        value: float
            The maximum element of the subdifferential of the i-th splitting
            term of the conjugate function at
            self.param_slope_pos_scalar(i, lmbd).
        """
        ...

    @abstractmethod
    def param_limit_neg_scalar(self, i: int, lmbd: float) -> float:
        """Value of

        .. math:: inf_x { x in subdiff(h^*_i)(t) }

        where :math:`h^*_i` is the conjugate of the i-th splitting term of the
        penalty function and :math:`t` is the value of
        `self.param_slope_neg_scalar(i, lmbd)`.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        lmbd: float
            Argument of the function `self.param_slope_neg_scalar`.

        Returns
        -------
        value: float
            The minimum element of the subdifferential of the i-th splitting
            term of the conjugate function at
            self.param_slope_neg_scalar(i, lmbd).
        """
        ...

    def value(self, x: ArrayLike) -> float:
        """Value of the function at ``x``.

        Parameters
        ----------
        x: ArrayLike
            Value at which the function is evaluated.

        Returns
        -------
        value: float
            The value of the function at ``x``.
        """
        v = np.zeros_like(x)
        for i, xi in enumerate(x):
            v[i] = self.value_scalar(i, xi)
        return v.sum()

    def conjugate(self, x: ArrayLike) -> float:
        """Value of the conjugate of the function at ``x``.

        Parameters
        ----------
        x: ArrayLike
            Value at which the conjugate is evaluated.

        Returns
        -------
        value: float
            The value of the conjugate of the function at ``x``.
        """
        v = np.zeros_like(x)
        for i, xi in enumerate(x):
            v[i] = self.conjugate_scalar(i, xi)
        return v.sum()

    def prox(self, x: ArrayLike, eta: float) -> ArrayLike:
        """Proximity operator of ``eta`` times the function at ``x``.

        Parameters
        ----------
        x: ArrayLike
            Value at which the prox is evaluated.
        eta: float, positive
            Multiplicative factor of the function.

        Returns
        -------
        p: ArrayLike
            The proximity operator of ``eta`` times the function at ``x``.
        """
        p = np.zeros_like(x)
        for i, xi in enumerate(x):
            p[i] = self.prox_scalar(i, xi, eta)
        return p

    def subdiff(self, x: ArrayLike) -> ArrayLike:
        """Subdifferential operator of the function at ``x``.

        Parameters
        ----------
        x: ArrayLike
            Value at which the prox is evaluated.

        Returns
        -------
        s: ArrayLike
            The subdifferential (interval) of the function at ``x``.
        """
        s = np.zeros((x.size, 2))
        for i, xi in enumerate(x):
            s[i, :] = self.subdiff_scalar(i, xi)
        return s

    def conjugate_subdiff(self, x: ArrayLike) -> ArrayLike:
        """Subdifferential operator of the function conjugate at ``x``.

        Parameters
        ----------
        x: ArrayLike
            Value at which the prox is evaluated.

        Returns
        -------
        s: ArrayLike
            The subdifferential (interval) of the function conjugate at ``x``.
        """
        s = np.zeros((x.size, 2))
        for i, xi in enumerate(x):
            s[i, :] = self.conjugate_subdiff_scalar(i, xi)
        return s

    def param_slope_pos(self, lmbd: float, idx: list) -> ArrayLike:
        # TODO: documentation
        return np.array([self.param_slope_pos_scalar(i, lmbd) for i in idx])

    def param_slope_neg(self, lmbd: float, idx: list) -> ArrayLike:
        # TODO: documentation
        return np.array([self.param_slope_neg_scalar(i, lmbd) for i in idx])

    def param_limit_pos(self, lmbd: float, idx: list) -> ArrayLike:
        # TODO: documentation
        return np.array([self.param_limit_pos_scalar(i, lmbd) for i in idx])

    def param_limit_neg(self, lmbd: float, idx: list) -> ArrayLike:
        # TODO: documentation
        return np.array([self.param_limit_neg_scalar(i, lmbd) for i in idx])


class SymmetricPenalty(BasePenalty):

    @abstractmethod
    def param_slope_scalar(self, i: int, lmbd: float) -> float:
        """Value of

        .. math:: sup_x { x in R | h^*_i(x) <= lmbd }

        where :math:`h^*_i` is the conjugate of the i-th splitting term of the
        penalty function.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        lmbd: float
            Threshold value.

        Returns
        -------
        value: float
            The maximum value of ``x`` such that the conjugate function is
            below ``lmbd``.
        """
        ...

    def param_slope_pos_scalar(self, i: int, lmbd: float) -> float:
        return self.param_slope_scalar(i, lmbd)

    def param_slope_neg_scalar(self, i: int, lmbd: float) -> float:
        return -self.param_slope_scalar(i, lmbd)

    @abstractmethod
    def param_limit_scalar(self, i: int, lmbd: float) -> float:
        """Value of

        .. math:: sup_x { x in subdiff(h^*_i)(t) }

        where :math:`h^*_i` is the conjugate of the i-th splitting term of the
        penalty function and :math:`t` is the value of
        `self.param_slope_pos_scalar(i, lmbd)`.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        lmbd: float
            Argument of the function `self.param_slope_pos_scalar`.

        Returns
        -------
        value: float
            The maximum element of the subdifferential of the i-th splitting
            term of the conjugate function at
            self.param_slope_pos_scalar(i, lmbd).
        """
        ...

    def param_limit_pos_scalar(self, i: int, lmbd: float) -> float:
        return self.param_limit_scalar(i, lmbd)

    def param_limit_neg_scalar(self, i: int, lmbd: float) -> float:
        return -self.param_limit_scalar(i, lmbd)


class MipPenalty:
    """Base class for penalty functions that can be modeled into pyomo."""

    @abstractmethod
    def bind_model(self, model: pmo.block, lmbd: float) -> None:
        """Bind the L0-regularization together with the penalty function into a
        pyomo kernel model. The model should contain a scalar and
        unconstrained variable `model.g`, a variable `model.x` with size
        `model.N` and a variable `model.z` with size `model.N`. The
        `bind_model` function binds the following epigraph formulation:

        .. math:: model.g >= lmbd * sum(model.z) + self.value(model.x)

        and must ensures that `model.z[i] = 0` implies `model.x[i] = 0`.

        Parameters
        ----------
        model: pmo.block
            The pyomo mixed-integer programming model (kernel model).
        lmbd: float
            The L0-regularization weight.
        """
        ...
