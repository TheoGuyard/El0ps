"""Base classes for data-fidelity functions and related utilities."""

from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray
from gurobipy import Model, Var, MVar


class BaseDatafit(metaclass=ABCMeta):
    """Base class for data-fidelity functions."""

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

    @abstractmethod
    def bind_model_cost(
        self, model: Model, A: NDArray, x_var: MVar, f_var: Var
    ) -> None:
        """#TODO: Doc"""
        ...


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
    Lipschitz-continuous gradient."""

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

    @property
    @abstractmethod
    def L(self) -> float:
        """Value of the gradient Lipchitz constant.

        Returns
        -------
        L : float
            The gradient Lipchitz constant value.
        """
