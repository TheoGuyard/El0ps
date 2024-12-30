"""Base classes for datafit functions and related utilities."""

import pyomo.kernel as pmo
from abc import abstractmethod
from numpy.typing import ArrayLike


class BaseDatafit:
    """Base class for datafit functions. The function takes a real vector as
    input and returns an extended-real value in ]-infty,+infty]. It is assumed
    to be proper, lower semicontinuous, convex, and differentiable with a
    Lipschitz-continuous gradient."""

    @abstractmethod
    def value(self, x: ArrayLike) -> float:
        """Value of the function at ``x``.

        Parameters
        ----------
        x: ArrayLike
            Vector at which the function is evaluated.

        Returns
        -------
        value: float
            The function value at ``x``.
        """
        ...

    @abstractmethod
    def conjugate(self, x: ArrayLike) -> float:
        """Value of the conjugate of the function at ``x``.

        Parameters
        ----------
        x: ArrayLike
            Vector at which the conjugate is evaluated.

        Returns
        -------
        value: float
            The conjugate value at ``x``.
        """
        ...

    @abstractmethod
    def gradient(self, x: ArrayLike) -> ArrayLike:
        """Value of gradient at ``x``.

        Parameters
        ----------
        x: ArrayLike
            Vector at which the gradient is evaluated.

        Returns
        -------
        g: ArrayLike
            The gradient at ``x``.
        """
        ...

    @abstractmethod
    def gradient_lipschitz_constant(self) -> float:
        """Lipschitz constant of the gradient.

        Returns
        -------
        L: float
            The Lipschitz constant of the gradient.
        """
        ...


class MipDatafit:
    """Base class for datafit functions that can be modeled into pyomo."""

    @abstractmethod
    def bind_model(self, model: pmo.block) -> None:
        """Bind the datafit function into a pyomo kernel model. The model
        should contain a scalar and unconstrained variable `model.f` as well as
        a variable `model.w` with size `model.M`. The `bind_model` function
        binds the epigraph formulation `model.f >= self.value(model.w)`.

        Arguments
        ---------
        model: pmo.block
            The pyomo mixed-integer programming model (kernel model).
        """
        ...
