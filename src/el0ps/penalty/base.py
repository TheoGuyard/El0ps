"""Base classes for penalty functions and related utilities."""

import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray


class BasePenalty:
    """Base class for penalty functions. These functions are assumed to be
    separable and splitting terms are identified by the index `i`, i.e, h(x) =
    sum_i hi(xi)."""

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
        dict_of_params : dict
            The parameters to instantiate an object of the class.
        """

    @abstractmethod
    def value_scalar(self, i: int, x: float) -> float:
        """Value of h_i(.) at x.

        Parameters
        ----------
        i : int
            Index of the splitting term.
        x : float
            Value at which h_i(.) is evaluated.

        Returns
        -------
        value_scalar : float
            The value of h_i(.) at x.
        """

    @abstractmethod
    def conjugate_scalar(self, i: int, x: float) -> float:
        """Value of the convex-conjugate of h_i(.) at x.

        Parameters
        ----------
        i : int
            Index of the splitting term.
        x : float
            Value at which the convex-conjugate of h_i(.) is evaluated.

        Returns
        -------
        value : float
            The value of the convex-conjugate of h_i(.) at x.
        """

    def value(self, x: NDArray) -> float:
        """Value of function at vector x.

        Parameters
        ----------
        x : NDArray, shape (n,)
            Vector at which the function is evaluated.

        Returns
        -------
        value : float
            The function value at vector x.
        """

        value = np.sum(self.value_scalar(i, xi) for i, xi in enumerate(x))
        return value

    def conjugate(self, x: NDArray) -> float:
        """Value of the convex-conjugate of the function at vector x.

        Parameters
        ----------
        x : NDArray, shape (n,)
            Vector at which the convex-conjugate is evaluated.

        Returns
        -------
        value : float
            The convex-conjugate value at vector x.
        """

        value = np.sum(self.conjugate_scalar(i, xi) for i, xi in enumerate(x))
        return value

    @abstractmethod
    def param_zerlimit(self, i: int) -> float:
        """Maximum value of `x` such that the convex-conjugate `h_i(.)` at `x`
        is zero.

        Parameters
        ----------
        i : int
            Index of the splitting term.

        Returns
        -------
        param_zerlimit : float
            The maximum value of `x` such that the convex-conjugate `h_i(.)` at
            `x` is zero.
        """

    @abstractmethod
    def param_domlimit(self, i: int) -> float:
        """Maximum value of `x` such that the convex-conjugate of `h_i(.)` at
        `x` is finite.

        Parameters
        ----------
        i : int
            Index of the splitting term.

        Returns
        -------
        param_zerlimit : float
            The maximum value of `x` such that the convex-conjugate of `h_i(.)`
            at `x` is finite.
        """

    @abstractmethod
    def param_vallimit(self, i: int) -> float:
        """Maximum value of the convex-conjugate of `h_i(.)` over its domain.

        Parameters
        ----------
        i : int
            Index of the splitting term.

        Returns
        -------
        param_zerlimit : float
            The maximum value of the convex-conjugate of `h_i(.)` over its
            domain.
        """

    @abstractmethod
    def param_levlimit(self, i: int, lmbd: float) -> float:
        """Maximum value of `x` such that the convex-conjugate of `h_i(.)`
        at `x` is geater or equal to the value `lmbd`.

        Parameters
        ----------
        i : int
            Index of the splitting term.
        lmbd : float
            Threshold value.

        Returns
        -------
        param_levlimit : float
            The maximum value of `x` such that the convex-conjugate of `h_i(.)`
            is greater or equal to `lmbd`.
        """

    @abstractmethod
    def param_sublimit(self, i: int, lmbd: float) -> float:
        """Minimum value of `x` such that `x` belongs to the subdifferential of
        the convex-conjugate of `h_i(.)` at `self.param_slope(i, lmbd)`.

        Parameters
        ----------
        i : int
            Index of the splitting term.
        lmbd : float
            Value threshold in the function `self.param_slope`.

        Returns
        -------
        param_sublimit : float
            The minimum value of `x` such that `x` belongs to the
            subdifferential of the convex-conjugate of `h_i(.)` at
            `self.param_slope(i, lmbd)`.
        """


class ProximablePenalty(BasePenalty):
    """Base class for proximable penalty functions. These functions are assumed
    to be separable."""

    @abstractmethod
    def prox_scalar(self, i: int, x: float, eta: float) -> float:
        """Prox of `eta` times the i-th splitting-term of the function at x.

        Parameters
        ----------
        i : int
            Index of the splitting term.
        x : float
            Value at which the i-th splitting term of the prox is evaluated.
        eta : float, positive
            Multiplicative factor in front of the i-th splitting term of the
            function.

        Returns
        -------
        p : float
            The proximity operator of the i-th splitting term of the function
            at x.
        """

    def prox(self, x: NDArray, eta: float) -> NDArray:
        """Prox of `eta` times the function evaluated at vector x.

        Parameters
        ----------
        x : NDArray, shape (n,)
            Vector at which the prox is evaluated.
        eta : float, positive
            Multiplicative factor in front of the function.

        Returns
        -------
        p : NDArray, shape (n,)
            The proximity operator at vector x.
        """

        p = np.array([self.prox_scalar(i, xi, eta) for i, xi in enumerate(x)])
        return p
