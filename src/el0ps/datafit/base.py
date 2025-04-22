"""Base classes for datafit functions and related utilities."""

import pyomo.kernel as pmo
from abc import abstractmethod
from numpy.typing import NDArray


class BaseDatafit:
    """Base class for datafit functions. These functions are assumed proper,
    lower-semicontinuous, convex, and differentiable with a
    Lipschitz-continuous gradient."""

    @abstractmethod
    def value(self, w: NDArray) -> float:
        """Value of the function at ``w``.

        Parameters
        ----------
        w : NDArray
            Vector at which the function is evaluated.

        Returns
        -------
        value : float
            The function value at ``w``.
        """
        ...

    @abstractmethod
    def conjugate(self, w: NDArray) -> float:
        """Value of the convex conjugate of the function at ``w``.

        Parameters
        ----------
        w : NDArray
            Vector at which the conjugate is evaluated.

        Returns
        -------
        value : float
            The conjugate value at ``w``.
        """
        ...

    @abstractmethod
    def gradient(self, w: NDArray) -> NDArray:
        """Value of gradient at ``w``.

        Parameters
        ----------
        w : NDArray
            Vector at which the gradient is evaluated.

        Returns
        -------
        value : NDArray
            The gradient at ``w``.
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
        """In a pyomo model containing a real scalar variable ``model.f`` and a
        real vector variable ``model.w`` of size ``model.M``, bind the relation

        ``model.f >= self.value(model.w)``

        using ``pyomo`` expressions.

        Parameters
        ----------
        model : pmo.block
            The pyomo kernel model.
        """
        ...
