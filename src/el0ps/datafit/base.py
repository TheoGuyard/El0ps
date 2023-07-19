"""Base classes for data-fidelity functions and related utilities."""

from abc import abstractmethod
from numpy.typing import NDArray


class BaseDatafit:
    """Base class for data-fidelity functions."""

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

    @abstractmethod
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


class ProximableDatafit(BaseDatafit):
    """Base class for proximable data-fidelity functions."""

    @abstractmethod
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


class SmoothDatafit(BaseDatafit):
    """Base class for differentiable data-fidelity functions with a
    Lipschitz-continuous gradient. Functions deriving from this class must
    set an attribute `L` giving the gradient Lipschitz constant value."""

    @abstractmethod
    def gradient(self, x: NDArray) -> NDArray:
        """Value of gradient at vector x.

        Parameters
        ----------
        x : NDArray, shape (n,)
            Vector at which the gradient is evaluated.

        Returns
        -------
        g : NDArray, shape (n,)
            The gradient at vector x.
        """
        ...
