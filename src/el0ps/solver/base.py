"""Base classes for L0-penalized problem solvers and related utilities."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from numpy.typing import NDArray
from typing import Optional
from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty


class Status(Enum):
    """Solver status.

    Attributes
    ----------
    UNKNOWN: str
        Unknown solver status.
    ITER_LIMIT: str
        The solver reached the iteration limit.
    TIME_LIMIT: str
        The solver reached the time limit.
    MEMORY_LIMIT: str
        The solver reached the memory limit.
    RUNNING: str
        The solver is running.
    INFEASIBLE: str
        The problem is infeasible.
    UNBOUNDED: str
        The problem is unbounded.
    OPTIMAL: str
        The solver found an optimal solution.
    """

    def __str__(self):
        return str(self.value)

    UNKNOWN = "unknown"
    ITER_LIMIT = "iter_limit"
    TIME_LIMIT = "time_limit"
    MEMORY_LIMIT = "memory_limit"
    RUNNING = "running"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    OPTIMAL = "optimal"


@dataclass
class Result:
    """Solver results.

    Attributes
    ----------
    status: Status
        Solver status.
    solve_time: float
        Solve time in seconds.
    iter_count: int
        Iteration count.
    x: NDArray
        Problem solution.
    objective_value: float
        Objective value.
    trace: dict, default=None
        Solver trace if available, otherwise `None`.
    """

    status: Status
    solve_time: float
    iter_count: int
    x: NDArray
    objective_value: float
    trace: Optional[dict]

    def __str__(self) -> str:
        s = ""
        s += "Result\n"
        s += "  Status     : {}\n".format(self.status.value)
        s += "  Solve time : {:f} seconds\n".format(self.solve_time)
        s += "  Iter count : {:d}\n".format(self.iter_count)
        s += "  Objective  : {:f}\n".format(self.objective_value)
        s += "  Non-zeros  : {:d}".format(sum(self.x != 0.0))
        return s


class BaseSolver:
    r"""Base class for solvers of L0-regularized problem.

    The problem is expressed as

    .. math::

        \textstyle\min_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{Ax}) + \lambda\|\mathbf{x}\|_0 + h(\mathbf{x})

    where :math:`f` is a :class:`el0ps.datafit.BaseDatafit` function,
    :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` is a matrix, :math:`h` is a
    :class:`el0ps.penalty.BasePenalty` function, and :math:`\lambda` is a
    positive scalar."""  # noqa: E501

    @property
    def accept_jitclass(self) -> bool:
        """Return whether if the solver accepts a
        `jitclass <https://numba.readthedocs.io/en/stable/user/jitclass.html>`_
        for the datafit and penalty function. This method can be overridden in
        derived classes to provide a more specific implementation."""
        return False

    @abstractmethod
    def solve(
        self,
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: NDArray,
        lmbd: float,
        x_init: Optional[NDArray] = None,
    ) -> Result:
        r"""Solve an L0-regularized problem.

        Parameters
        ----------
        datafit: BaseDatafit
            Problem datafit function.
        penalty: BasePenalty
            Problem penalty function.
        A: NDArray
            Problem matrix.
        lmbd: float, positive
            Problem L0-norm weight.
        x_init: NDArray, default=None
            Stating point for the solver.

        Returns
        -------
        result: Result
            Solver results.
        """
        ...
