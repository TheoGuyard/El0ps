"""Base classes for L0-penalized problem solvers and related utilities."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from numpy.typing import NDArray
from el0ps.problem import Problem


class Status(Enum):
    SOLVE_NOT_CALLED = -10
    OTHER_LIMIT = -4
    NODE_LIMIT = -3
    TIME_LIMIT = -2
    RUNNING = -1
    OPTIMAL = 0


@dataclass
class Results:
    """Solver results."""

    termination_status: Status
    solve_time: float
    node_count: int
    objective_value: float
    lower_bound: float
    upper_bound: float
    x: NDArray
    z: NDArray
    trace: dict | None


@dataclass
class BaseSolver(metaclass=ABCMeta):
    """Base class for L0-penalized problem solvers."""

    @abstractmethod
    def solve(
        self,
        problem: Problem,
        x_init: NDArray | None = None,
        S0_init: NDArray | None = None,
        S1_init: NDArray | None = None,
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
