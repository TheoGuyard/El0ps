"""Base classes for penalty functions and related utilities."""

from abc import abstractmethod


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
        dict_of_params: dict
            The parameters to instantiate an object of the class.
        """
        ...

    @abstractmethod
    def value(self, x: float) -> float:
        """Value of the function at ``x``.

        Parameters
        ----------
        x: float
            Value at which the function is evaluated.

        Returns
        -------
        value: float
            The value of the function at ``x``.
        """
        ...

    @abstractmethod
    def conjugate(self, x: float) -> float:
        """Value of the conjugate of the function at ``x``.

        Parameters
        ----------
        x: float
            Value at which the conjugate is evaluated.

        Returns
        -------
        value: float
            The value of the conjugate of the function at ``x``.
        """
        ...

    @abstractmethod
    def conjugate_scaling_factor(self, x: float) -> float:
        """Return a scalar ``s`` such that ``s * x`` is within the domain of
        the conjugate of the function.

        Parameters
        ----------
        x: float
            Value to be scaled.

        Returns
        -------
        s: float
            The scaling factor.
        """
        ...

    @abstractmethod
    def param_slope(self, lmbd: float) -> float:
        """Maximum value of ``x`` such that the function is below ``lmbd``.

        Parameters
        ----------
        lmbd: float
            Threshold value.

        Returns
        -------
        value: float
            The maximum value of ``x`` such that the function is below
            ``lmbd``.
        """
        ...

    @abstractmethod
    def param_limit(self, lmbd: float) -> float:
        """Minimum value of `x` such that `x` is in the subdifferential of the
        conjugate of the function at `self.param_slope(lmbd)`.

        Parameters
        ----------
        lmbd: float
            Argument of the function `self.param_slope`.

        Returns
        -------
        value: float
            The minimum value of `x` such that `x` is in the subdifferential of
            the conjugate of the function at `self.param_slope(lmbd)`.
        """
        ...

    @abstractmethod
    def param_maxval(self) -> float:
        """Maximum value of the conjugate of the function over its domain.

        Returns
        -------
        value: float
            The maximum value of the conjugate of the function over its domain.
        """
        ...

    @abstractmethod
    def param_maxzer(self) -> float:
        """Maximum value of x such that the conjugate of the function at x
        equals zero.

        Returns
        -------
        value: float
            The maximum value of x such that the conjugate of the function at x
            equals zero.
        """
        ...

    def approximate_param_slope(
        self, lmbd: float, tol: float = 1e-4, maxit: int = 100
    ) -> float:
        """Utility function to approximate the value of :func:`.param_slope`
        when no closed-form is available.

        Parameters
        ----------
        tol: float = 1e-4
            Tolerance for the approximation.
        maxit: int = 100
            Maximum number of bisection iterations.
        """
        a = 0.0
        b = 1.0
        c = 0.5
        while self.conjugate(b) < lmbd:
            b *= 2.0
        for _ in range(maxit):
            c = (a + b) / 2.0
            fa = self.conjugate(a) - lmbd
            fc = self.conjugate(c) - lmbd
            if (-tol <= fc <= tol) or (b - a < 0.5 * tol):
                return c
            if fc * fa >= 0.0:
                a = c
            else:
                b = c
        return c


class ProximablePenalty(BasePenalty):
    """Base class for proximable penalty functions."""

    @abstractmethod
    def prox(self, x: float, eta: float) -> float:
        """Proximity operator of ``eta`` times the function at ``x``.

        Parameters
        ----------
        x: float
            Value at which the prox is evaluated.
        eta: float, positive
            Multiplicative factor in front of the function.

        Returns
        -------
        p: float
            The proximity operator of ``eta`` times the function at ``x``.
        """
        ...
