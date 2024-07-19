"""Base classes for penalty functions and related utilities."""

import numpy as np
import pyomo.kernel as pmo
from abc import abstractmethod
from numpy.typing import ArrayLike


class BasePenalty:
    """Base class for penalty functions."""

    @abstractmethod
    def get_spec(self) -> tuple:
        """Specify the numba types of the class attributes.

        Returns
        -------
        spec: Tuple of (attr_name, dtype)
            Specs to be passed to Numba jitclass to compile the class.
        """
        ...

    @abstractmethod
    def params_to_dict(self) -> dict:
        """Get the parameters to initialize an instance of the class.

        Returns
        -------
        dict_of_params: dict
            The parameters to instantiate an object of the class.
        """
        ...

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
    def param_slope_scalar(self, i: int, lmbd: float) -> float:
        """Maximum value of ``x`` such that the i-th splitting term of the
        function is below ``lmbd``.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        lmbd: float
            Threshold value.

        Returns
        -------
        value: float
            The maximum value of ``x`` such that the function is below
            ``lmbd``.
        """
        ...

    @abstractmethod
    def param_limit_scalar(self, i: int, lmbd: float) -> float:
        """Minimum value of ``x`` such that ``x`` is in the i-th splitting term
        of the subdifferential of the conjugate of the function at
        ``self.param_slope(lmbd)``.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        lmbd: float
            Argument of the function `self.param_slope`.

        Returns
        -------
        value: float
            The minimum value of `x` such that `x` is in the subdifferential of
            the conjugate of the function at `self.param_slope(lmbd)`.
        """
        return ...

    @abstractmethod
    def param_maxval_scalar(self, i: int) -> float:
        """Maximum value of the i-th splitting term of the conjugate of the
        function over its domain.

        Parameters
        ----------
        i: int
            Index of the splitting term.

        Returns
        -------
        value: float
            The maximum value of the conjugate of the function over its domain.
        """
        ...

    @abstractmethod
    def param_maxdom_scalar(self, i: int) -> float:
        """Right boundary of the i-th splitting term of the conjugate domain.

        Parameters
        ----------
        i: int
            Index of the splitting term.

        Returns
        -------
        value: float
            The right boundary of the conjugate domain.
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
        s = np.zeros_like(x)
        for i, xi in enumerate(x):
            s[i] = self.subdiff_scalar(i, xi)
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
        s = np.zeros_like(x)
        for i, xi in enumerate(x):
            s[i] = self.conjugate_subdiff_scalar(i, xi)
        return s.sum()


class MipPenalty:
    """Base class for penalty functions that can be modeled into a
    Mixed-Integer Program."""

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
