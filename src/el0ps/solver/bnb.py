"""Branch-and-Bound algorithm solver for L0-penalized problems."""

import numpy as np
import time
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Union
from numpy.typing import NDArray
from el0ps.problem import Problem
from .base import BaseSolver, Results, Status
from .node import BnbNode
from .bounding import BoundingSolver


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
    workingsets: bool
        Whether to use working sets during the bounding process.
    l1screening: bool
        Whether to use screening acceleration.
    l0screening: bool
        Whether to use node-screening acceleration.
    verbose: bool
        Whether to toggle solver verbosity.
    trace: bool
        Whether to store the solver trace.
    """

    bounding_solver: BoundingSolver = BoundingSolver()
    exploration_strategy: BnbExplorationStrategy = BnbExplorationStrategy.DFS
    exploration_depth_switch: int = 0
    branching_strategy: BnbBranchingStrategy = BnbBranchingStrategy.LARGEST
    time_limit: float = float(sys.maxsize)
    node_limit: int = sys.maxsize
    rel_tol: float = 1e-4
    int_tol: float = 1e-8
    workingsets: bool = True
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
        "queue_length",
        "lower_bound",
        "upper_bound",
        "abs_gap",
        "rel_gap",
        "supp_left",
        "node_lower_bound",
        "node_upper_bound",
        "node_time_lower_bound",
        "node_time_upper_bound",
        "node_card_S0",
        "node_card_S1",
        "node_card_Sb",
        "node_depth",
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
            np.abs(self.upper_bound) + 1e-16
        )

    @property
    def solve_time(self):
        """Solver solve time in seconds."""
        return time.time() - self.start_time

    @property
    def queue_length(self):
        """Length of the Branch-and-Bound queue."""
        return len(self.queue)

    @property
    def supp_left(self):
        if len(self.queue) == 0:
            return 0.0
        return sum(
            [2 ** np.count_nonzero(qnode.Sb) for qnode in self.queue]
        ) / (2**self.x.size)

    def setup(
        self,
        problem: Problem,
        x_init: Union[NDArray[np.float64], None] = None,
        S0_init: Union[NDArray[np.float64], None] = None,
        S1_init: Union[NDArray[np.float64], None] = None,
    ):
        # Sanity checks
        if x_init is None:
            x_init = np.zeros(problem.n)
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
        self.upper_bound = problem.value(x_init)
        self.trace = {key: [] for key in self._trace_keys}

        # Initialize the bounding solver
        self.options.bounding_solver.setup(problem)

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
            np.zeros(problem.n),
            np.zeros(problem.m),
            np.zeros(problem.n),
        )

        # Add initial fixing constraints to root
        for idx in np.flatnonzero(S0_init):
            root.fix_to(problem, idx, 0)
        for idx in np.flatnonzero(S1_init):
            root.fix_to(problem, idx, 1)

        # Initialize the queue with root node
        self.queue.append(root)

    def print_header(self):
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

    def print_progress(self, node: BnbNode):
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

    def print_footer(self):
        s = "-" * 68
        print(s)

    def can_continue(self):
        if self.solve_time >= self.options.time_limit:
            self.status = Status.TIME_LIMIT
        elif self.iter_count >= self.options.node_limit:
            self.status = Status.NODE_LIMIT
        elif len(self.queue) == 0:
            self.status = Status.OPTIMAL
        elif self.rel_gap < self.options.rel_tol:
            self.status = Status.OPTIMAL

        return self.status == Status.RUNNING

    def compute_lower_bound(self, node: BnbNode):
        self.options.bounding_solver.bound(
            node,
            self.upper_bound,
            self.options.rel_tol,
            self.options.workingsets,
            self.options.dualpruning,
            self.options.l1screening,
            self.options.l0screening,
            upper=False,
        )

    def compute_upper_bound(self, node: BnbNode):
        if node.category == 0.:
            return
        self.options.bounding_solver.bound(
            node,
            self.upper_bound,
            self.options.rel_tol,
            False,
            False,
            False,
            False,
            upper=True,
        )

    def next_node(self):
        if self.options.exploration_strategy == BnbExplorationStrategy.DFS:
            next_node = self.queue.pop()
        elif self.options.exploration_strategy == BnbExplorationStrategy.BFS:
            next_node = self.queue.pop(0)
        elif self.options.exploration_strategy == BnbExplorationStrategy.BBS:
            next_node = self.queue.pop(
                np.argmax([qnode.lower_bound for qnode in self.queue])
            )
        elif self.options.exploration_strategy == BnbExplorationStrategy.WBS:
            next_node = self.queue.pop(
                np.argmin([qnode.lower_bound for qnode in self.queue])
            )
        else:
            raise NotImplementedError
        self.iter_count += 1
        return next_node

    def prune(self, node: BnbNode):
        return self.upper_bound < node.lower_bound

    def is_feasible(self, problem: Problem, node: BnbNode):
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

    def update_trace(self, node: BnbNode):
        for key in self._trace_keys:
            if key.startswith("node_"):
                self.trace[key].append(getattr(node, key[5:]))
            else:
                self.trace[key].append(getattr(self, key))

    def update_bounds(self, node):
        if node.upper_bound < self.upper_bound:
            self.upper_bound = node.upper_bound
            self.x = np.copy(node.x_inc)
            for qnode in self.queue:
                if self.prune(qnode):
                    self.queue.remove(qnode)
        if len(self.queue) != 0:
            self.lower_bound = min([qnode.lower_bound for qnode in self.queue])
        else:
            self.lower_bound = self.upper_bound

    def branch(self, problem: Problem, node: BnbNode):
        if not np.any(node.Sb):
            return
        if self.options.branching_strategy == BnbBranchingStrategy.LARGEST:
            jSb = np.argmax(np.abs(node.x[node.Sb]))
            j = np.arange(node.x.size)[node.Sb][jSb]

        node0 = node.child(problem, j, 0)
        self.queue.append(node0)

        node1 = node.child(problem, j, 1)
        self.queue.append(node1)

    def solve(
        self,
        problem: Problem,
        x_init: Union[NDArray[np.float64], None] = None,
        S0_init: Union[NDArray[np.bool_], None] = None,
        S1_init: Union[NDArray[np.bool_], None] = None,
    ):
        self.setup(problem, x_init, S0_init, S1_init)

        if self.options.verbose:
            self.print_header()

        while self.can_continue():
            node = self.next_node()
            self.compute_lower_bound(node)
            if not self.prune(node):
                self.compute_upper_bound(node)
                if not self.is_feasible(problem, node):
                    self.branch(problem, node)
            self.update_bounds(node)
            if self.options.trace:
                self.update_trace(node)
            if self.options.verbose:
                self.print_progress(node)
            del node

        if self.options.verbose:
            self.print_footer()

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
