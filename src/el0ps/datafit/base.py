"""Base classes for datafit functions and related utilities."""

import pyomo.kernel as pmo
from abc import abstractmethod
from numpy.typing import NDArray


class BaseDatafit:
    """Base class for datafit functions, assumed proper, lower-semicontinuous,
    convex, and differentiable with a Lipschitz-continuous gradient."""

    @abstractmethod
    def value(self, x: NDArray) -> float:
        """Value of the function at ``x``.

        Parameters
        ----------
        x : NDArray
            Vector at which the function is evaluated.

        Returns
        -------
        value : float
            The function value at ``x``.
        """
        ...

    @abstractmethod
    def conjugate(self, x: NDArray) -> float:
        """Value of the convex conjugate of the function at ``x``.

        Parameters
        ----------
        x : NDArray
            Vector at which the conjugate is evaluated.

        Returns
        -------
        value : float
            The conjugate value at ``x``.
        """
        ...

    @abstractmethod
    def gradient(self, x: NDArray) -> NDArray:
        """Value of gradient at ``x``.

        Parameters
        ----------
        x : NDArray
            Vector at which the gradient is evaluated.

        Returns
        -------
        value : NDArray
            The gradient at ``x``.
        """
        ...

    @abstractmethod
    def gradient_lipschitz_constant(self) -> float:
        """Lipschitz constant of the gradient.

        Returns
        -------
        value : float
            The Lipschitz constant of the gradient.
        """
        ...


class MipDatafit:
    """Base class for datafit functions that can be modeled into pyomo."""

    @abstractmethod
    def bind_model(self, model: pmo.block) -> None:
        """In a pyomo model containing a real scalar variable `model.f` and a
        real vector variable `model.w` of size `model.M`, bind the relation

        ``model.f >= self.value(model.w)``

        using ``pyomo`` expressions.

        Parameters
        ----------
        model : pmo.block
            The pyomo kernel model.
        """
        ...
