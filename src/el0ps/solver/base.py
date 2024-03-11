"""Base classes for L0-penalized problem solvers and related utilities."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from numpy.typing import ArrayLike
from typing import Union
from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty


class Status(Enum):
    """:class:`solver.BaseSolver` status.

    Attributes
    ----------
    UNKNOWN: str
        Unknown solver status.
    NODE_LIMIT: str
        The solver reached the node limit.
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
    NODE_LIMIT = "nodelim"
    TIME_LIMIT = "timelim"
    RUNNING = "running"
    OPTIMAL = "optimal"


@dataclass
class Result:
    """:class:`solver.BaseSolver` results.

    Attributes
    ----------
    status: Status
        Solver status.
    solve_time: float
        Solver solve time in seconds.
    iter_count: int
        Solver iter count.
    rel_gap: float
        Solver relative gap.
    x: ArrayLike
        Problem solution.
    z: ArrayLike
        Binary vector coding where non-zeros are located in ``x``.
    objective_value: float
        Problem objective value.
    n_nnz: int
        Number of non-zeros in ``x``.
    trace: dict
        Solver trace.
    """

    status: Status
    solve_time: float
    iter_count: int
    rel_gap: float
    x: ArrayLike
    z: ArrayLike
    objective_value: float
    n_nnz: int
    trace: dict

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
    """Base class for L0-penalized problem solvers."""

    @abstractmethod
    def solve(
        self,
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: ArrayLike,
        lmbd: float,
        x_init: Union[ArrayLike, None] = None,
    ):
        r"""Solve an L0-penalized problem of the form

        .. math:: \textstyle\min_{x} f(Ax) + \lambda \|x\|_0 + h(x)

        where :math:`f(\cdot)` is a datafit function, :math:`A` is a matrix,
        :math:`\lambda>0` is the L0-regularization weight and :math:`h(\cdot)`
        is a penalty function.

        Parameters
        ----------
        datafit: BaseDatafit
            Datafit function.
        penalty: BasePenalty
            Penalty function.
        A: ArrayLike
            Linear operator.
        lmbd: float, positive
            L0-norm weight.
        x_init: Union[ArrayLike, None] = None
            Stating value of ``x``.

        Returns
        -------
        result: Result
            Solver results.
        """
        ...
