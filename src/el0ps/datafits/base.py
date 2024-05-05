"""Base classes for datafit functions and related utilities."""

import pyomo.kernel as pmo
from abc import abstractmethod
from numpy.typing import ArrayLike


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


class MipDatafit(BaseDatafit):
    """Base class for :class:`.datafit.BaseDatafit` that can be modeled into
    a Mixed-Integer Program."""

    @abstractmethod
    def bind_model(self, model: pmo.block) -> None:
        """Bind the datafit function into a pyomo `kernel` model. The model
        should contain a scalar and unconstrained variable `model.f` as well as
        a variable `model.w` with size `model.M`. The `bind_model` function
        binds the following epigraph formulation:

        .. math:: model.f >= self.value(model.w)

        Arguments
        ---------
        model: pmo.block
            The pyomo mixed-integer programming model (kernel model).
        """
        ...


class SmoothDatafit(BaseDatafit):
    """Base class for differentiable :class:`.datafit.BaseDatafit` functions
    with a Lipschitz-continuous gradient."""

    @abstractmethod
    def lipschitz_constant(self) -> float:
        """Lipschitz constant of the gradient.

        Returns
        -------
        L: float
            The Lipschitz constant of the gradient.
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


class StronglyConvexDatafit(BaseDatafit):
    """Base class for strongly convex :class:`.datafit.BaseDatafit`
    functions."""

    @abstractmethod
    def strong_convexity_constant(self) -> float:
        """Strong convexity constant of the function.

        Returns
        -------
        S: float
            The strong convexity constant of the function.
        """
        ...
