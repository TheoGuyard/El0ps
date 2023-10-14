"""Base classes for data-fidelity functions and related utilities."""

import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray


class BaseDatafit:
    """Base class for datafit functions."""

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
    def value(self, x: NDArray[np.float64]) -> float:
        """Value of function at ``x``.

        Parameters
        ----------
        x: NDArray[np.float64]
            Vector at which the function is evaluated.

        Returns
        -------
        value: float
            The function value at ``x``.
        """
        ...

    @abstractmethod
    def conjugate(self, x: NDArray[np.float64]) -> float:
        """Value of the conjugate of the function at ``x``.

        Parameters
        ----------
        x: NDArray[np.float64]
            Vector at which the conjugate is evaluated.

        Returns
        -------
        value: float
            The conjugate value at ``x``.
        """
        ...


class ProximableDatafit(BaseDatafit):
    """Base class for proximable :class:`.datafit.BaseDatafit` functions."""

    @abstractmethod
    def prox(self, x: NDArray[np.float64], eta: float) -> NDArray[np.float64]:
        """Prox of ``eta`` times the function at ``x``.

        Parameters
        ----------
        x: NDArray[np.float64]
            Vector at which the prox is evaluated.
        eta: float, positive
            Multiplicative factor in front of the function.

        Returns
        -------
        p: NDArray[np.float64]
            The proximity operator at ``x``.
        """
        ...


class SmoothDatafit(BaseDatafit):
    """Base class for differentiable :class:`.datafit.BaseDatafit` functions
    with a Lipschitz-continuous gradient. Functions deriving from this class
    must set an attribute ``L`` giving the gradient Lipschitz constant
    value."""

    @abstractmethod
    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Value of gradient at ``x``.

        Parameters
        ----------
        x: NDArray[np.float64]
            Vector at which the gradient is evaluated.

        Returns
        -------
        g: NDArray[np.float64]
            The gradient at ``x``.
        """
        ...
