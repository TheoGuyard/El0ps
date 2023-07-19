"""Branch-and-Bound algorithm solver for L0-penalized problems."""

import numpy as np
import time
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Union
from numpy.typing import NDArray
from el0ps.problem import Problem
from .base import BaseSolver, Results, Status
from .bounding import BnbBoundingSolver, GurobiBoundingSolver


class BnbExplorationStrategy(Enum):
    BFS = "BFS"
    DFS = "DFS"
    # TODO mixed BFS-DFS


class BnbBranchingStrategy(Enum):
    LARGEST = "LARGEST"


class BnbNode:
    """Branch-and-Bound tree node.

    Parameters
    ----------
    S0 : NDArray
        Set of indices forced to be zero.
    S1 : NDArray
        Set of indices forced to be non-zero.
    Sb : NDArray
        Set of free indices.
    lower_bound : float
        BnbNode lower bound.
    upper_bound : float
        BnbNode upper bound.
    x : NDArray
        Relaxation solution.
    w : NDArray
        Value of `problem.A @ self.x`, where `problem` is the problem to be
        solved.
    u : NDArray
        Value of `-problem.datafit.gradient(self.w)`, where `problem` is the
        problem to be solved.
    x_inc : NDArray
        Incumbent solution computed at the node.
    """

    def __init__(
        self,
        S0: NDArray,
        S1: NDArray,
        Sb: NDArray,
        lower_bound: float,
        upper_bound: float,
        x: NDArray,
        w: NDArray,
        u: NDArray,
        x_inc: NDArray,
    ) -> None:
        self.S0 = S0
        self.S1 = S1
        self.Sb = Sb
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.x = x
        self.w = w
        self.u = u
        self.x_inc = x_inc

    def __str__(self) -> str:
        s = ""
        s += "BnbNode\n"
        s += "  S0 / S1 / Sb : {} / {} / {}\n".format(
            np.sum(self.S0), np.sum(self.S1), np.sum(self.Sb)
        )
        s += "  Lower bound  : {:.4f}\n".format(self.lower_bound)
        s += "  Upper bound  : {:.4f}\n".format(self.upper_bound)
        return s

    def __copy__(self):
        return BnbNode(
            np.copy(self.S0),
            np.copy(self.S1),
            np.copy(self.Sb),
            self.lower_bound,
            self.upper_bound,
            np.copy(self.x),
            np.copy(self.w),
            np.copy(self.u),
            np.copy(self.x_inc),
        )

    def depth(self):
        return np.sum(np.logical_or(self.S0, self.S1))

    @property
    def abs_gap(self):
        return self.upper_bound - self.lower_bound

    @property
    def rel_gap(self):
        return self.abs_gap / (np.abs(self.upper_bound) + 1e-16)

    @property
    def card_S0(self):
        return np.sum(self.S0)

    @property
    def card_S1(self):
        return np.sum(self.S1)

    @property
    def card_Sb(self):
        return np.sum(self.Sb)

    def fix_to(self, problem: Problem, idx: int, val: bool):
        if not (isinstance(val, bool) or val in [0, 1]):
            raise ValueError("Argument `val` must be a `bool`.")
        if not self.Sb[idx]:
            raise ValueError(f"Index {idx} is already fixed.")

        self.Sb[idx] = False
        if val:
            self.S1[idx] = True
        else:
            self.S0[idx] = True
            if self.x[idx] != 0.0:
                self.w -= self.x[idx] * problem.A[:, idx]
                self.u = -problem.datafit.gradient(self.w)
                self.x[idx] = 0.0


@dataclass
class BnbOptions:
    """Branch-and-Bound solver options.

    Parameters
    ----------
    bounding_solver : Any
        Bounding solver.
    exploration_strategy : BnbExplorationStrategy
        Branch-and-Bound exploration strategy.
    exploration_depth_switch : int
        Depth switch for the BnbExplorationStrategy.MIX exploration strategy.
    branching_strategy : BnbBranchingStrategy
        Branch-and-Bound branching strategy.
    time_limit : float
        Branch-and-Bound time limit in seconds.
    node_limit : int
        Branch-and-Bound node limit.
    abs_tol : float
        Target absolute tolerance `|ub - lb|`.
    rel_tol : float
        Target relative tolerance `|ub - lb| / max(|ub|, |lb|)`.
    int_tol : float
        Integrality tolerance for a float.
    prune_tol : float
        Pruning test tolerance.
    verbose : bool
        Whether to toggle solver verbosity.
    trace : bool
        Whether to store the solver trace.
    """

    bounding_solver: BnbBoundingSolver = GurobiBoundingSolver(
        {"OutputFlag": 0}
    )
    exploration_strategy: BnbExplorationStrategy = BnbExplorationStrategy.DFS
    exploration_depth_switch: int = 0
    branching_strategy: BnbBranchingStrategy = BnbBranchingStrategy.LARGEST
    time_limit: float = 60.0
    node_limit: int = 1_000
    abs_tol: float = 1e-8
    rel_tol: float = 1e-4
    int_tol: float = 1e-8
    prune_tol: float = 0.0
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
        self.status = Status.SOLVE_NOT_CALLED
        self.start_time = None
        self.queue = None
        self.node_count = None
        self.x = None
        self.lower_bound = None
        self.upper_bound = None
        self.trace = None

    def _print_header(self):
        s = "-" * 62 + "\n"
        s += "|"
        s += " {:>9}".format("Timer")
        s += " {:>9}".format("Nodes")
        s += " {:>9}".format("Lower")
        s += " {:>9}".format("Upper")
        s += " {:>9}".format("A-gap")
        s += " {:>9}".format("R-gap")
        s += "|"
        print(s)

    def _print_info(self):
        s = "|"
        s += " {:>9.2f}".format(self.solve_time)
        s += " {:>9d}".format(self.node_count)
        s += " {:>9.2f}".format(self.lower_bound)
        s += " {:>9.2f}".format(self.upper_bound)
        s += " {:>9.2e}".format(self.abs_gap)
        s += " {:>9.2e}".format(self.rel_gap)
        s += "|"
        print(s)

    def _print_footer(self):
        s = "-" * 62
        print(s)

    @property
    def abs_gap(self):
        return self.upper_bound - self.lower_bound

    @property
    def rel_gap(self):
        return self.abs_gap / (np.abs(self.upper_bound) + 1e-16)

    @property
    def solve_time(self):
        return time.time() - self.start_time

    @property
    def queue_length(self):
        return len(self.queue)

    def _can_continue(self):
        """Update the solver status and return whether it can continue. Before
        calling this function, the solver status is Status.RUNNING or
        Status.SOLVE_NOT_CALLED."""

        if self.solve_time >= self.options.time_limit:
            self.status = Status.TIME_LIMIT
        elif self.node_count >= self.options.node_limit:
            self.status = Status.NODE_LIMIT
        elif (self.rel_gap < self.options.rel_tol) and (
            self.abs_gap < self.options.abs_tol
        ):
            self.status = Status.OPTIMAL
        elif not any(self.queue):
            self.status = Status.OPTIMAL

        return self.status in [Status.SOLVE_NOT_CALLED, Status.RUNNING]

    def _next_node(self):
        if self.options.exploration_strategy == BnbExplorationStrategy.DFS:
            next_node = self.queue.pop()
        elif self.options.exploration_strategy == BnbExplorationStrategy.BFS:
            next_node = self.queue.pop(0)
        else:
            raise NotImplementedError
        self.node_count += 1
        return next_node

    def _continue_below_node(self, node: BnbNode):
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
        return node.lower_bound <= self.upper_bound + self.options.prune_tol

    def _prune_queue(self):
        """Prune all queue nodes whose lower bound is greater than best upper
        bound.
        """
        for node in self.queue:
            if node.lower_bound > self.upper_bound + self.options.prune_tol:
                self.queue.remove(node)

    def _setup(
        self,
        problem: Problem,
        x_init: Union[NDArray, None] = None,
        S0_init: Union[NDArray, None] = None,
        S1_init: Union[NDArray, None] = None,
    ):
        """Initialize internal attributes of the solver and set the warm-start.

        Parameters
        ----------
        problem : Problem
            Problem to solve.
        x_init : Union[NDArray, None]
            Warm-start value of x.
        S0_init : Union[NDArray, None]
            Set of indices of x forced to be zero.
        S1_init : Union[NDArray, None]
            Set of indices of x forced to be non-zero.
        """

        # Initialize warm-start defaults
        if x_init is None:
            x_init = np.zeros(problem.n)
        if S0_init is None:
            S0_init = np.zeros(problem.n, dtype=bool)
        if S1_init is None:
            S1_init = np.zeros(problem.n, dtype=bool)

        # Sanity check for argument type and compatibilities
        x_init = np.array(x_init)
        S0_init = np.array(S0_init)
        S1_init = np.array(S1_init)
        if np.any(np.logical_and(S0_init, x_init != 0.0)):
            raise ValueError(
                "Arguments `x_init` and `S0_init` are incompatible."
            )

        # Initialize internal solver attributes
        self.status = Status.RUNNING
        self.start_time = time.time()
        self.queue = []
        self.node_count = 0
        self.x = np.copy(x_init)
        self.lower_bound = -np.inf
        self.upper_bound = problem.value(x_init)
        self.trace = {key: [] for key in self._trace_keys}

        # Root node
        root = BnbNode(
            np.zeros(problem.n, dtype=bool),
            np.zeros(problem.n, dtype=bool),
            np.ones(problem.n, dtype=bool),
            self.lower_bound,
            self.upper_bound,
            np.copy(self.x),
            problem.A @ self.x,
            -problem.datafit.gradient(problem.A @ self.x),
            np.copy(self.x),
        )

        # Add initial fixing constraints to root
        for idx in np.where(S0_init)[0]:
            root.fix_to(problem, idx, 0)
        for idx in np.where(S1_init)[0]:
            root.fix_to(problem, idx, 1)

        # Initialize the queue with root node
        self.queue.append(root)

    def _has_integer_solution(self, problem: Problem, node: BnbNode):
        return (node.rel_gap <= self.options.rel_tol) and (
            node.abs_gap <= self.options.abs_tol
        )

    def _update_trace(self, node: BnbNode):
        for key in self._trace_keys:
            if key.startswith("node_"):
                self.trace[key].append(getattr(node, key[5:]))
            else:
                self.trace[key].append(getattr(self, key))

    def _branch(self, problem: Problem, node: BnbNode):
        # If end of branch, do not create children
        if not np.any(node.Sb):
            return

        # Select branching index
        if self.options.branching_strategy == BnbBranchingStrategy.LARGEST:
            idx = np.argmax(np.abs(node.x * node.Sb))

        node0 = copy(node)
        node0.fix_to(problem, idx, 0)

        node1 = copy(node)
        node1.fix_to(problem, idx, 1)

        self.queue.append(node0)
        self.queue.append(node1)

    def solve(
        self,
        problem: Problem,
        x_init: Union[NDArray, None] = None,
        S0_init: Union[NDArray, None] = None,
        S1_init: Union[NDArray, None] = None,
    ):
        if self.options.verbose:
            self._print_header()

        self._setup(problem, x_init, S0_init, S1_init)

        bounding_solver = self.options.bounding_solver
        bounding_solver.setup(problem, x_init, S0_init, S1_init)

        while self._can_continue():
            # 1) Choose a new node
            node = self._next_node()

            # 2) Solve the relaxed problem
            bounding_solver.bound(problem, node, self, "lower")

            # 3) Check infeasibility and pruning condition
            if self._continue_below_node(node):
                # 3a) Construct a feasible solution
                bounding_solver.bound(problem, node, self, "upper")

                # 3b) Update the upper bound and prune queue nodes if needed
                if node.upper_bound < self.upper_bound:
                    self.upper_bound = node.upper_bound
                    self.x = node.x_inc
                    self._prune_queue()

                # 3c) Branch and update the best lower bound if the relaxation
                # solution is not integer
                if not self._has_integer_solution(problem, node):
                    self._branch(problem, node)
                    self.lower_bound = min(
                        [queue_node.lower_bound for queue_node in self.queue]
                    )

            if self.options.trace:
                self._update_trace(node)
            if self.options.verbose:
                self._print_info()

            # Free memory
            del node

        if self.options.verbose:
            self._print_footer()

        return Results(
            self.status,
            self.solve_time,
            self.node_count,
            problem.value(self.x),
            self.lower_bound,
            self.upper_bound,
            self.x,
            np.array(self.x != 0.0, dtype=float),
            self.trace,
        )
