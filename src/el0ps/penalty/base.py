"""Base classes for penalty functions and related utilities."""

import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray


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
        dict_of_params : dict
            The parameters to instantiate an object of the class.
        """
        ...

    @abstractmethod
    def value(self, x: float) -> float:
        """Value of the function at x.

        Parameters
        ----------
        x : float
            Value at which the function is evaluated.

        Returns
        -------
        value : float
            The value of the function at x.
        """
        ...

    @abstractmethod
    def conjugate(self, x: float) -> float:
        """Value of the conjugate of the function at x.

        Parameters
        ----------
        x : float
            Value at which the conjugate is evaluated.

        Returns
        -------
        value : float
            The value of the conjugate of the function at x.
        """
        ...

    @abstractmethod
    def param_slope(self, lmbd: float) -> float:
        """Maximum value of `x` such that `h(x) <= lmbd`.
        
        Parameters
        ----------
        lmbd : float
            Threshold value.

        Returns
        -------
        value : float
            The maximum value of `x` such that `h(x) <= lmbd`.
        """
        ...

    @abstractmethod
    def param_limit(self, lmbd: float) -> float:
        """Minimum value of `x` such that `x` is in the subdifferential of the 
        conjugate of the function at `self.param_slope(lmbd)`.

        Parameters
        ----------
        lmbd : float
            Argument of the function `self.param_slope`.

        Returns
        -------
        value : float
            The minimum value of `x` such that `x` is in the subdifferential of
            the conjugate of the function at `self.param_slope(lmbd)`.
        """
        ...

    @abstractmethod
    def param_maxval(self) -> float:
        """Maximum value of the conjugate of the function over its domain.

        Returns
        -------
        value : float
            The maximum value of the conjugate of the function over its domain.
        """
        ...


class ProximablePenalty(BasePenalty):
    """Base class for proximable penalty functions."""

    @abstractmethod
    def prox(self, x: float, eta: float) -> float:
        """Proximity operator of `eta` times the function at x.

        Parameters
        ----------
        x : float
            Value at which the prox is evaluated.
        eta : float, positive
            Multiplicative factor in front of the function.

        Returns
        -------
        p : float
            The proximity operator of `eta` times the function at x.
        """
        ...