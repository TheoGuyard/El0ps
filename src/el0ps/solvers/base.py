"""Base classes for L0-penalized problem solvers and related utilities."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from numba.experimental.jitclass.base import JitClassType
from numpy.typing import ArrayLike
from typing import Union
from el0ps.datafits import BaseDatafit
from el0ps.penalties import BasePenalty


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
    RUNNING: str
        The solver is running.
    OPTIMAL: str
        The solver found an optimal solution.
    """

    def __str__(self):
        return str(self.value)

    UNKNOWN = "unknown"
    ITER_LIMIT = "iter_limit"
    TIME_LIMIT = "time_limit"
    RUNNING = "running"
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
    rel_gap: float
        Relative gap.
    x: ArrayLike
        Problem solution.
    objective_value: float
        Objective value.
    n_nnz: int
        Number of non-zeros in the solution.
    trace: Union[dict, None]
        Solver trace if available, otherwise `None`.
    """

    status: Status
    solve_time: float
    iter_count: int
    rel_gap: float
    x: ArrayLike
    objective_value: float
    n_nnz: int
    trace: Union[dict, None]

    def __str__(self) -> str:
        s = ""
        s += "Result\n"
        s += "  Status     : {}\n".format(self.status.value)
        s += "  Solve time : {:.6f} seconds\n".format(self.solve_time)
        s += "  Iter count : {}\n".format(self.iter_count)
        s += "  Rel. gap   : {:.2e}\n".format(self.rel_gap)
        s += "  Objective  : {:.6f}\n".format(self.objective_value)
        s += "  Non-zeros  : {}".format(self.n_nnz)
        return s


class BaseSolver:
    """Base class for L0-regularized problem solvers.

    The optimization problem solved is

    .. math:: \min f(Xw) + \lambda \|w\|_0 + h(w)

    where :math:`f` is a datafit term, :math:`h` is a penalty term and
    :math:`\lambda` is the L0-norm weight.
    """  # noqa: W605

    @abstractmethod
    def solve(
        self,
        datafit: Union[BaseDatafit, JitClassType],
        penalty: Union[BasePenalty, JitClassType],
        A: ArrayLike,
        lmbd: float,
        x_init: Union[ArrayLike, None] = None,
    ):
        r"""Solve an L0-regularized problem.

        Parameters
        ----------
        datafit: Union[BaseDatafit, JitClassType]
            Data-fitting function, can be compiled as a jitclass.
        penalty: Union[BasePenalty, JitClassType]
            Penalty function, can be compiled as a jitclass.
        A: ArrayLike
            Linear operator.
        lmbd: float, positive
            L0-norm weight.
        x_init: Union[ArrayLike, None] = None
            Stating point.

        Returns
        -------
        result: Result
            Solver results.
        """
        ...
