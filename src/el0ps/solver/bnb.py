"""Branch-and-Bound algorithm solver for L0-penalized problems."""

import numpy as np
import time
import sys
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Union
from numpy.typing import NDArray
from el0ps.problem import Problem
from .base import BaseSolver, Results, Status, _REL_GAP_EPS
from .node import BnbNode
from .bounding import BnbBoundingSolver, CdBoundingSolver


class BnbExplorationStrategy(Enum):
    """:class:`.solver.BnbSolver` exploration strategy.

    Attributes
    ----------
    BFS: str
        Breadth-first search.
    DFS: str
        Depth-first search.
    BBS: str
        Best-bound search.
    WBS: str
        Worst-bound search.
    """

    BFS = "BFS"
    DFS = "DFS"
    BBS = "BBS"
    WBS = "WBS"
    # TODO mixed BFS-DFS


class BnbBranchingStrategy(Enum):
    """:class:`.solver.BnbSolver` exploration strategy.

    Attributes
    ----------
    LARGEST: str
        Select the largest entry in absolute value in the relaxation solution
        among indices that are still unfixed.
    """

    LARGEST = "LARGEST"


@dataclass
class BnbOptions:
    """:class:`.solver.BnbSolver` options.

    Parameters
    ----------
    bounding_solver: BnbBoundingSolver
        Bounding solver.
    exploration_strategy: BnbExplorationStrategy
        Branch-and-Bound exploration strategy.
    exploration_depth_switch: int
        Depth switch for the BnbExplorationStrategy.MIX exploration strategy.
    branching_strategy: BnbBranchingStrategy
        Branch-and-Bound branching strategy.
    time_limit: float
        Branch-and-Bound time limit in seconds.
    node_limit: int
        Branch-and-Bound node limit.
    rel_tol: float
        Relative MIP tolerance.
    int_tol: float
        Integrality tolerance for a float.
    dualpruning: bool
        Whether to use dual-pruning.
    l1screening: bool
        Whether to use screening acceleration.
    l0screening: bool
        Whether to use node-screening acceleration.
    verbose: bool
        Whether to toggle solver verbosity.
    trace: bool
        Whether to store the solver trace.
    """

    bounding_solver: BnbBoundingSolver = CdBoundingSolver()
    exploration_strategy: BnbExplorationStrategy = BnbExplorationStrategy.DFS
    exploration_depth_switch: int = 0
    branching_strategy: BnbBranchingStrategy = BnbBranchingStrategy.LARGEST
    time_limit: float = float(sys.maxsize)
    node_limit: int = sys.maxsize
    rel_tol: float = 1e-4
    int_tol: float = 1e-8
    dualpruning: bool = True
    l1screening: bool = True
    l0screening: bool = True
    verbose: bool = False
    trace: bool = False

    def _validate_types(self):
        for field_name, field_def in self.__dataclass_fields__.items():
            actual_type = type(getattr(self, field_name))
            if not issubclass(actual_type, field_def.type):
                raise ValueError(
                    "Expected '{}' for argument '{}', got '{}'.".format(
                        field_def.type, field_name, actual_type
                    )
                )

    def __post_init__(self):
        self._validate_types()


class BnbSolver(BaseSolver):
    """Branch-and-Bound solver for :class:`.Problem`."""

    _trace_keys = [
        "solve_time",
        "iter_count",
        "lower_bound",
        "upper_bound",
        "node_lower_bound",
        "node_upper_bound",
        "node_time_lower_bound",
        "node_time_upper_bound",
        "node_card_S0",
        "node_card_S1",
        "node_card_Sb",
    ]

    def __init__(self, **kwargs) -> None:
        self.options = BnbOptions(**kwargs)
        self.status = Status.RUNNING
        self.start_time = None
        self.queue = None
        self.iter_count = None
        self.x = None
        self.lower_bound = None
        self.upper_bound = None
        self.trace = None

    def __str__(self):
        return "BnbSolver"

    @property
    def abs_gap(self):
        """Absolute gap between the lower and upper bounds."""
        return self.upper_bound - self.lower_bound

    @property
    def rel_gap(self):
        """Relative gap between the lower and upper bounds."""
        return (self.upper_bound - self.lower_bound) / (
            np.abs(self.upper_bound) + _REL_GAP_EPS
        )

    @property
    def solve_time(self):
        """Solver solve time in seconds."""
        return time.time() - self.start_time

    @property
    def queue_length(self):
        """Length of the Branch-and-Bound queue."""
        return len(self.queue)

    def _setup(
        self,
        problem: Problem,
        x_init: Union[NDArray[np.float64], None] = None,
        S0_init: Union[NDArray[np.float64], None] = None,
        S1_init: Union[NDArray[np.float64], None] = None,
    ):
        # Sanity checks
        if x_init is None:
            x_init = np.zeros(problem.n)
        w_init = problem.A @ x_init
        if S0_init is None:
            S0_init = np.zeros(problem.n, dtype=np.bool_)
        if S1_init is None:
            S1_init = np.zeros(problem.n, dtype=np.bool_)
        if not np.all(x_init[S0_init] == 0.0):
            raise ValueError("Arguments `x_init` and `S0_init` missmatch.")
        if not np.all(x_init[S1_init] != 0.0):
            raise ValueError("Arguments `x_init` and `S1_init` missmatch.")

        # Initialize internal solver attributes
        self.status = Status.RUNNING
        self.start_time = time.time()
        self.queue = []
        self.iter_count = 0
        self.x = np.copy(x_init)
        self.lower_bound = -np.inf
        self.upper_bound = problem.value(x_init, w_init)
        if self.options.trace:
            self.trace = {key: [] for key in self._trace_keys}

        # Root node
        root = BnbNode(
            -1,
            np.zeros(problem.n, dtype=np.bool_),
            np.zeros(problem.n, dtype=np.bool_),
            np.ones(problem.n, dtype=np.bool_),
            -np.inf,
            np.inf,
            0.0,
            0.0,
            x_init,
            w_init,
            -problem.datafit.gradient(w_init),
            np.zeros(problem.n),
        )

        # Add initial fixing constraints to root
        for idx in np.flatnonzero(S0_init):
            root.fix_to(problem, idx, 0)
        for idx in np.flatnonzero(S1_init):
            root.fix_to(problem, idx, 1)

        # Initialize the queue with root node
        self.queue.append(root)

        # Initialize bounding solver
        self.options.bounding_solver.setup(problem, x_init, S0_init, S1_init)

    def _print_header(self):
        s = "-" * 68 + "\n"
        s += "|"
        s += " {:>6}".format("Iters")
        s += " {:>6}".format("Timer")
        s += " {:>5}".format("S0")
        s += " {:>5}".format("S1")
        s += " {:>5}".format("Sb")
        s += " {:>6}".format("Lower")
        s += " {:>6}".format("Upper")
        s += " {:>9}".format("Abs gap")
        s += " {:>9}".format("Rel gap")
        s += "|" + "\n"
        s += "-" * 68
        print(s)

    def _print_progress(self, node: BnbNode):
        s = "|"
        s += " {:>6d}".format(self.iter_count)
        s += " {:>6.2f}".format(self.solve_time)
        s += " {:>5d}".format(node.card_S0)
        s += " {:>5d}".format(node.card_S1)
        s += " {:>5d}".format(node.card_Sb)
        s += " {:>6.2f}".format(self.lower_bound)
        s += " {:>6.2f}".format(self.upper_bound)
        s += " {:>9.2e}".format(self.abs_gap)
        s += " {:>9.2e}".format(self.rel_gap)
        s += "|"
        print(s)

    def _print_footer(self):
        s = "-" * 68
        print(s)

    def _can_continue(self):
        if self.solve_time >= self.options.time_limit:
            self.status = Status.TIME_LIMIT
        elif self.iter_count >= self.options.node_limit:
            self.status = Status.NODE_LIMIT
        elif len(self.queue) == 0:
            self.status = Status.OPTIMAL
        elif self.rel_gap < self.options.rel_tol:
            self.status = Status.OPTIMAL

        return self.status == Status.RUNNING

    def _compute_lower_bound(self, problem: Problem, node: BnbNode):
        self.options.bounding_solver.bound(
            problem,
            node,
            self.upper_bound,
            self.options.rel_tol,
            self.options.dualpruning,
            self.options.l1screening,
            self.options.l0screening,
        )

    def _compute_upper_bound(self, problem: Problem, node: BnbNode):
        self.options.bounding_solver.bound(
            problem,
            node,
            self.upper_bound,
            self.options.rel_tol,
            False,
            False,
            False,
            True,
        )

    def _next_node(self):
        if self.options.exploration_strategy == BnbExplorationStrategy.DFS:
            _next_node = self.queue.pop()
        elif self.options.exploration_strategy == BnbExplorationStrategy.BFS:
            _next_node = self.queue.pop(0)
        elif self.options.exploration_strategy == BnbExplorationStrategy.BBS:
            _next_node = self.queue.pop(
                np.argmax([qnode.lower_bound for qnode in self.queue])
            )
        elif self.options.exploration_strategy == BnbExplorationStrategy.WBS:
            _next_node = self.queue.pop(
                np.argmin([qnode.lower_bound for qnode in self.queue])
            )
        else:
            raise NotImplementedError
        self.iter_count += 1
        return _next_node

    def _prune(self, node: BnbNode):
        return self.upper_bound < node.lower_bound

    def _has_feasible_solution(self, problem: Problem, node: BnbNode):
        return np.all(
            np.logical_or(
                np.abs(node.x[node.Sb]) <= self.options.int_tol,
                np.abs(node.x[node.Sb])
                >= problem.penalty.param_limit(problem.lmbd),
            )
        )

    def _has_tight_relaxation(self, problem: Problem, node: BnbNode):
        if node.rel_gap <= self.options.rel_tol:
            return True
        elif np.all(
            np.logical_or(
                np.abs(node.x[node.Sb]) <= self.options.int_tol,
                np.abs(node.x[node.Sb])
                >= problem.penalty.param_limit(problem.lmbd),
            )
        ):
            return True
        return False

    def _update_trace(self, node: BnbNode):
        for key in self._trace_keys:
            if key.startswith("node_"):
                self.trace[key].append(getattr(node, key[5:]))
            else:
                self.trace[key].append(getattr(self, key))

    def _update_bounds(self, node):
        if node.upper_bound < self.upper_bound:
            self.upper_bound = node.upper_bound
            self.x = np.copy(node.x_inc)
            for qnode in self.queue:
                if self._prune(qnode):
                    self.queue.remove(qnode)
        if len(self.queue) != 0:
            self.lower_bound = min([qnode.lower_bound for qnode in self.queue])
        else:
            self.lower_bound = self.upper_bound

    def _branch(self, problem: Problem, node: BnbNode):
        if not np.any(node.Sb):
            return
        if self.options.branching_strategy == BnbBranchingStrategy.LARGEST:
            jSb = np.argmax(np.abs(node.x[node.Sb]))
            j = np.arange(node.x.size)[node.Sb][jSb]

        node0 = copy(node)
        node0.fix_to(problem, j, 0)
        self.queue.append(node0)

        node1 = copy(node)
        node1.fix_to(problem, j, 1)
        self.queue.append(node1)

    def solve(
        self,
        problem: Problem,
        x_init: Union[NDArray[np.float64], None] = None,
        S0_init: Union[NDArray[np.bool_], None] = None,
        S1_init: Union[NDArray[np.bool_], None] = None,
    ):
        self._setup(problem, x_init, S0_init, S1_init)

        if self.options.verbose:
            self._print_header()

        while self._can_continue():
            node = self._next_node()
            self._compute_lower_bound(problem, node)
            if not self._prune(node):
                self._compute_upper_bound(problem, node)
                if not self._has_feasible_solution(problem, node):
                    self._branch(problem, node)
            self._update_bounds(node)
            if self.options.trace:
                self._update_trace(node)
            if self.options.verbose:
                self._print_progress(node)
            del node

        if self.options.verbose:
            self._print_footer()

        return Results(
            self.status,
            self.solve_time,
            self.iter_count,
            self.rel_gap,
            self.x,
            np.array(self.x != 0.0, dtype=float),
            problem.value(self.x),
            np.sum(np.abs(self.x) > self.options.int_tol),
            self.trace,
        )
