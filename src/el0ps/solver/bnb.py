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
from .base import BaseSolver, Results, Status
from .node import BnbNode
from .bounding import BnbBoundingSolver, CdBoundingSolver


class BnbExplorationStrategy(Enum):
    BFS = "BFS"
    DFS = "DFS"
    # TODO mixed BFS-DFS


class BnbBranchingStrategy(Enum):
    LARGEST = "LARGEST"


@dataclass
class BnbOptions:
    """Branch-and-Bound solver options.

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
    l1screening: bool = True
    l0screening: bool = True
    verbose: bool = False
    trace: bool = False

    def validate_types(self):
        for field_name, field_def in self.__dataclass_fields__.items():
            actual_type = type(getattr(self, field_name))
            if not issubclass(actual_type, field_def.type):
                raise ValueError(
                    "Expected '{}' for argument '{}', got '{}'.".format(
                        field_def.type, field_name, actual_type
                    )
                )

    def __post_init__(self):
        self.validate_types()


class BnbSolver(BaseSolver):
    """Branch-and-Bound solver for L0-penalized problems."""

    _trace_keys = [
        "solve_time",
        "node_count",
        "queue_size",
        "lower_bound",
        "upper_bound",
        "node_lower_bound",
        "node_upper_bound",
        "node_card_S0",
        "node_card_S1",
        "node_card_Sb",
    ]

    def __init__(self, **kwargs) -> None:
        self.options = BnbOptions(**kwargs)
        self.status = Status.RUNNING
        self.start_time = None
        self.queue = None
        self.node_count = None
        self.x = None
        self.lower_bound = None
        self.upper_bound = None
        self.trace = None

    def __str__(self):
        return "BnbSolver"

    @property
    def rel_gap(self):
        return np.abs(self.upper_bound - self.lower_bound) / (
            np.abs(self.upper_bound) + 1e-16
        )

    @property
    def solve_time(self):
        return time.time() - self.start_time

    @property
    def queue_length(self):
        return len(self.queue)

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
        self.node_count = 0
        self.x = np.copy(x_init)
        self.lower_bound = -np.inf
        self.upper_bound = problem.value(x_init, w_init)
        self.trace = {key: [] for key in self._trace_keys}

        # Root node
        root = BnbNode(
            -1,
            np.zeros(problem.n, dtype=np.bool_),
            np.zeros(problem.n, dtype=np.bool_),
            np.ones(problem.n, dtype=np.bool_),
            -np.inf,
            np.inf,
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

    def print_header(self):
        s = "-" * 58 + "\n"
        s += "|"
        s += " {:>6}".format("Nodes")
        s += " {:>6}".format("Timer")
        s += " {:>5}".format("S0")
        s += " {:>5}".format("S1")
        s += " {:>5}".format("Sb")
        s += " {:>6}".format("Lower")
        s += " {:>6}".format("Upper")
        s += " {:>9}".format("Rel gap")
        s += "|" + "\n"
        s += "-" * 58
        print(s)

    def print_progress(self, node: BnbNode):
        s = "|"
        s += " {:>6d}".format(self.node_count)
        s += " {:>6.2f}".format(self.solve_time)
        s += " {:>5d}".format(node.card_S0)
        s += " {:>5d}".format(node.card_S1)
        s += " {:>5d}".format(node.card_Sb)
        s += " {:>6.2f}".format(self.lower_bound)
        s += " {:>6.2f}".format(self.upper_bound)
        s += " {:>9.2e}".format(self.rel_gap)
        s += "|"
        print(s)

    def print_footer(self):
        s = "-" * 58
        print(s)

    def can_continue(self):
        """Update the solver status and return whether it can continue. Before
        calling this function, the solver status is Status.RUNNING."""

        if self.solve_time >= self.options.time_limit:
            self.status = Status.TIME_LIMIT
        elif self.node_count >= self.options.node_limit:
            self.status = Status.NODE_LIMIT
        elif self.rel_gap < self.options.rel_tol:
            self.status = Status.OPTIMAL
        elif not any(self.queue):
            self.status = Status.OPTIMAL

        return self.status == Status.RUNNING

    def next_node(self):
        if self.options.exploration_strategy == BnbExplorationStrategy.DFS:
            next_node = self.queue.pop()
        elif self.options.exploration_strategy == BnbExplorationStrategy.BFS:
            next_node = self.queue.pop(0)
        else:
            raise NotImplementedError
        self.node_count += 1
        return next_node

    def prune(self, node: BnbNode):
        """Check whether the exploration should continue below the node. The
        exploration will be stopped if the node is pruned by bound or if its
        relaxation is infeasible.

        Parameters
        ----------
        node : BnbNode
            The node to test.

        Returns
        -------
        decision : bool
            Whether the exploration should continue below the node.
        """
        return node.lower_bound > self.upper_bound

    def has_tight_relaxation(self, node: BnbNode):
        return node.rel_gap <= self.options.rel_tol

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
            for node in self.queue:
                if node.lower_bound > self.upper_bound:
                    self.queue.remove(node)
        if self.queue:
            self.lower_bound = np.min(
                [qnode.lower_bound for qnode in self.queue]
            )
        else:
            self.lower_bound = self.upper_bound

    def branch(self, problem: Problem, node: BnbNode):
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
        """Solve a `Problem`.

        Parameters
        ----------
        problem: Problem
            Problem to solve.
        x_init: Union[NDArray[np.float64], None]
            Warm-start value of x.
        S0_init: Union[NDArray[np.bool_], None]
            Set of indices of x forced to be zero from the beginning.
        S1_init: Union[NDArray[np.bool_], None]
            Set of indices of x forced to be non-zero from the beginning.
        """

        self.setup(problem, x_init, S0_init, S1_init)
        bounding_solver = self.options.bounding_solver
        bounding_solver.setup(problem, x_init, S0_init, S1_init)

        if self.options.verbose:
            self.print_header()

        while self.can_continue():
            node = self.next_node()
            bounding_solver.bound(
                problem,
                node,
                self.upper_bound,
                self.options.rel_tol,
                self.options.l1screening,
                self.options.l0screening,
            )
            if not self.prune(node):
                bounding_solver.bound(
                    problem,
                    node,
                    self.upper_bound,
                    self.options.rel_tol,
                    self.options.l1screening,
                    self.options.l0screening,
                    incumbent=True,
                )
                if not self.has_tight_relaxation(node):
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
            self.node_count,
            problem.value(self.x),
            self.rel_gap,
            self.x,
            np.array(self.x != 0.0, dtype=float),
            self.trace,
        )
