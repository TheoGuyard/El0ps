"""Base classes for L0-penalized problem solvers and related utilities."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Union
from numpy.typing import NDArray
from el0ps.problem import Problem


class Status(Enum):
    OTHER_LIMIT = "OTHER_LIMIT"
    NODE_LIMIT = "NODE_LIMIT"
    TIME_LIMIT = "TIME_LIMIT"
    RUNNING = "RUNNING"
    OPTIMAL = "OPTIMAL"


@dataclass
class Results:
    """Solver results."""

    status: Status
    solve_time: float
    node_count: int
    objective_value: float
    rel_gap: float
    x: NDArray
    z: NDArray
    trace: dict

    def __str__(self) -> str:
        s = ""
        s += "Results\n"
        s += "  Status     : {}\n".format(self.status.value)
        s += "  Solve time : {:.6f} seconds\n".format(self.solve_time)
        s += "  Node count : {}\n".format(self.node_count)
        s += "  Objective  : {:.6f}\n".format(self.objective_value)
        s += "  Rel. gap   : {:.2%}\n".format(self.rel_gap)
        s += "  Non-zeros  : {}".format(int(sum(self.z)))
        return s


@dataclass
class BaseSolver:
    """Base class for L0-penalized problem solvers."""

    @abstractmethod
    def solve(
        self,
        problem: Problem,
        x_init: Union[NDArray, None] = None,
        S0_init: Union[NDArray, None] = None,
        S1_init: Union[NDArray, None] = None,
    ):
        """Solve an L0-penalized problem.

        Parameters
        ----------
        problem : Problem
            Problem to solve.
        x_init : Union[NDArray, None]
            Stating value of x.
        S0_init : Union[NDArray, None]
            Set of indices of x forced to be zero.
        S1_init : Union[NDArray, None]
            Set of indices of x forced to be non-zero.

        Returns
        -------
        result : Result
            Solver results.
        """
        ...
