"""Base classes for datafit functions and related utilities."""

import pyomo.kernel as pmo
from abc import abstractmethod
from numpy.typing import NDArray


class BaseDatafit:
    r"""Base class for datafit functions.

    This class represent mathematical functions expressed as

    .. math::
        \begin{align*}
            f : \mathbb{R}^m &\rightarrow \mathbb{R} \cup \{+\infty\} \\
            \mathbf{w} &\mapsto f(\mathbf{w})
        \end{align*}

    that are proper, lower-semicontinuous, convex, and differentiable with a
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
    """Base class for datafit functions that can be modeled into
    `pyomo <https://pyomo.readthedocs.io/en/stable/>`_."""

    @abstractmethod
    def bind_model(self, model: pmo.block) -> None:
        """Impose a constraint associated with the datafit function in a
        `pyomo <https://pyomo.readthedocs.io/en/stable/>`_ model.

        Given a pyomo.kernel.block ``model`` object containing a real scalar
        variable ``model.f`` and a real vector variable ``model.w`` of size
        ``model.M``, this function is intended to imposer the relation

        ``model.f >= self.value(model.w)``

        using ``pyomo`` expressions.

        Parameters
        ----------
        model : pmo.block
            The pyomo kernel model.
        """
        ...
