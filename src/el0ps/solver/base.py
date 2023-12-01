"""Base classes for L0-penalized problem solvers and related utilities."""

import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Union
from numba import njit
from numpy.typing import NDArray
from el0ps.problem import Problem


@njit
def rel_gap(ub, lb):
    return np.abs(ub - lb) / (np.abs(ub) + 1e-16)


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
class Results:
    """:class:`solver.BaseSolver` results.

    Attributes
    ----------
    status: Status
        Solver status.
    solve_time: float
        Solver solve time in seconds.
    node_count: int
        Solver node count.
    rel_gap: float
        Solver relative gap.
    x: NDArray
        Problem solution.
    z: NDArray
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
    node_count: int
    rel_gap: float
    x: NDArray
    z: NDArray
    objective_value: float
    n_nnz: int
    trace: dict

    def __str__(self) -> str:
        s = ""
        s += "Results\n"
        s += "  Status     : {}\n".format(self.status.value)
        s += "  Solve time : {:.6f} seconds\n".format(self.solve_time)
        s += "  Node count : {}\n".format(self.node_count)
        s += "  Rel. gap   : {:.2%}\n".format(self.rel_gap)
        s += "  Objective  : {:.6f}\n".format(self.objective_value)
        s += "  Non-zeros  : {}".format(self.n_nnz)
        return s


@dataclass
class BaseSolver:
    """Base class for :class:`.Problem` solvers."""

    @abstractmethod
    def solve(
        self,
        problem: Problem,
        x_init: Union[NDArray, None] = None,
        S0_init: Union[NDArray, None] = None,
        S1_init: Union[NDArray, None] = None,
    ):
        """Solve a :class:`.Problem`.

        Parameters
        ----------
        problem: Problem
            :class:`.Problem` to solve.
        x_init: Union[NDArray, None] = None
            Stating value of ``x``.
        S0_init: Union[NDArray, None] = None
            Indices in ``x`` forced to be zero.
        S1_init: Union[NDArray, None] = None
            Indices in ``x`` forced to be non-zero.

        Returns
        -------
        result: Result
            Solver results.
        """
        ...
