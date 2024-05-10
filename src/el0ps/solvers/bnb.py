"""Branch-and-Bound solver for L0-regularized problems."""

import numpy as np
import time
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Union
from numba.experimental.jitclass.base import JitClassType
from numpy.typing import ArrayLike
from el0ps.datafits import BaseDatafit
from el0ps.penalties import BasePenalty
from el0ps.solvers import BaseSolver, Result, Status
from el0ps.utils import compiled_clone
from .bnb_node import BnbNode
from .bnb_bound import BnbBoundingSolver


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
    MIX: str
        Starts with a DFS strategy and switches to a BBS strategy when the
        relative mip gap is domain a factor `mix_threshold` from the target
        one.
    """

    BFS = "BFS"
    DFS = "DFS"
    BBS = "BBS"
    MIX = "MIX"

    def mix_threshold(self):
        return 1e-1


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
    """:class:`.solvers.BnbSolver` options.

    Parameters
    ----------
    bounding_solver: BnbBnbBoundingSolver
        Bounding solver.
    exploration_strategy: BnbExplorationStrategy
        Branch-and-Bound exploration strategy.
    branching_strategy: BnbBranchingStrategy
        Branch-and-Bound branching strategy.
    time_limit: float
        Branch-and-Bound time limit in seconds.
    iter_limit: int
        Branch-and-Bound iteration limit (number of nodes explored).
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
    simpruning: bool
        Whether to use node-screening acceleration.
    verbose: bool
        Whether to toggle solver verbosity.
    trace: bool
        Whether to store the solver trace.
    """

    exploration_strategy: BnbExplorationStrategy = BnbExplorationStrategy.MIX
    branching_strategy: BnbBranchingStrategy = BnbBranchingStrategy.LARGEST
    bounding_regfunc_type: str = "convex"
    bounding_maxiter_inner: int = 1_000
    bounding_maxiter_outer: int = 100
    bounding_skip_setup: bool = False
    time_limit: float = float(sys.maxsize)
    iter_limit: int = sys.maxsize
    rel_tol: float = 1e-4
    int_tol: float = 1e-8
    workingsets: bool = True
    dualpruning: bool = True
    l1screening: bool = True
    simpruning: bool = True
    verbose: bool = False
    trace: bool = False


class BnbSolver(BaseSolver):
    """Branch-and-Bound solver for L0-penalized problems."""

    _trace_keys = [
        "timer",
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
        self.bounding_solver = None

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
            np.abs(self.upper_bound) + 1e-10
        )

    @property
    def timer(self):
        """Elapsed time from the start time."""
        return time.time() - self.start_time

    @property
    def queue_length(self):
        """Length of the Branch-and-Bound queue."""
        return len(self.queue)

    @property
    def supp_left(self):
        """Number of support left to explore."""
        if len(self.queue) == 0:
            return 0.0
        return sum(
            [2 ** np.count_nonzero(qnode.Sb) for qnode in self.queue]
        ) / (2**self.x.size)

    def setup(
        self,
        datafit: Union[BaseDatafit, JitClassType],
        penalty: Union[BasePenalty, JitClassType],
        A: ArrayLike,
        lmbd: float,
        x_init: Union[ArrayLike, None],
    ):

        if isinstance(datafit, BaseDatafit):
            datafit = compiled_clone(datafit)
        if isinstance(penalty, BasePenalty):
            penalty = compiled_clone(penalty)
        if not A.flags.f_contiguous:
            A = np.array(A, order="F")
        if x_init is None:
            x_init = np.zeros(A.shape[1])

        # Initialize attributes and state
        self.m, self.n = A.shape
        self.datafit = datafit
        self.penalty = penalty
        self.A = A
        self.lmbd = lmbd
        self.status = Status.RUNNING
        self.queue = []
        self.iter_count = 0
        self.x = np.copy(x_init)
        self.lower_bound = -np.inf
        self.upper_bound = (
            datafit.value(A @ self.x)
            + lmbd * np.linalg.norm(x_init, 0)
            + penalty.value(self.x)
        )
        self.trace = {key: [] for key in self._trace_keys}

        # Initialize the bounding solver
        if (
            not self.options.bounding_skip_setup
            or self.bounding_solver is None
        ):
            self.bounding_solver = BnbBoundingSolver(
                regfunc_type=self.options.bounding_regfunc_type,
                maxiter_inner=self.options.bounding_maxiter_inner,
                maxiter_outer=self.options.bounding_maxiter_outer,
            )
            self.bounding_solver.setup(datafit, penalty, A)

        # Root node
        root = BnbNode(
            -1,
            np.zeros(self.n, dtype=np.bool_),
            np.zeros(self.n, dtype=np.bool_),
            np.ones(self.n, dtype=np.bool_),
            -np.inf,
            np.inf,
            0.0,
            0.0,
            np.zeros(self.n),
            np.zeros(self.m),
            np.zeros(self.n),
        )
        self.queue.append(root)

        self.start_time = time.time()

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
        s += " {:>6.2f}".format(self.timer)
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
        if self.timer >= self.options.time_limit:
            self.status = Status.TIME_LIMIT
        elif self.iter_count >= self.options.iter_limit:
            self.status = Status.ITER_LIMIT
        elif len(self.queue) == 0:
            self.status = Status.OPTIMAL
        elif self.rel_gap < self.options.rel_tol:
            self.status = Status.OPTIMAL

        return self.status == Status.RUNNING

    def compute_lower_bound(self, node: BnbNode):
        self.bounding_solver.bound(
            node,
            self.lmbd,
            self.upper_bound,
            self.options.rel_tol,
            self.options.workingsets,
            self.options.dualpruning,
            self.options.l1screening,
            self.options.simpruning,
            upper=False,
        )

    def compute_upper_bound(self, node: BnbNode):
        if node.category == 0.0:
            return
        self.bounding_solver.bound(
            node,
            self.lmbd,
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
                np.argmin([qnode.lower_bound for qnode in self.queue])
            )
        elif self.options.exploration_strategy == BnbExplorationStrategy.MIX:
            if (
                self.rel_gap / self.options.rel_tol
            ) < self.options.exploration_strategy.mix_threshold():
                next_node = self.queue.pop(
                    np.argmin([qnode.lower_bound for qnode in self.queue])
                )
            else:
                next_node = self.queue.pop()
        else:
            raise NotImplementedError
        self.iter_count += 1
        return next_node

    def prune(self, node: BnbNode):
        return self.upper_bound < node.lower_bound

    def is_feasible(self, node: BnbNode):
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
            for qnode in self.queue:
                if self.prune(qnode):
                    self.queue.remove(qnode)
        if len(self.queue) != 0:
            self.lower_bound = min([qnode.lower_bound for qnode in self.queue])
        else:
            self.lower_bound = self.upper_bound

    def branch(self, node: BnbNode):
        if not np.any(node.Sb):
            return
        if self.options.branching_strategy == BnbBranchingStrategy.LARGEST:
            jSb = np.argmax(np.abs(node.x[node.Sb]))
            j = np.arange(node.x.size)[node.Sb][jSb]

        node0 = node.child(j, 0, self.A)
        self.queue.append(node0)

        node1 = node.child(j, 1, self.A)
        self.queue.append(node1)

    def solve(
        self,
        datafit: Union[BaseDatafit, JitClassType],
        penalty: Union[BasePenalty, JitClassType],
        A: ArrayLike,
        lmbd: float,
        x_init: Union[ArrayLike, None] = None,
    ):

        self.setup(datafit, penalty, A, lmbd, x_init)

        if self.options.verbose:
            self.print_header()

        while self.can_continue():
            node = self.next_node()
            self.compute_lower_bound(node)
            if not self.prune(node):
                self.compute_upper_bound(node)
                if not self.is_feasible(node):
                    self.branch(node)
            self.update_bounds(node)
            if self.options.trace:
                self.update_trace(node)
            if self.options.verbose:
                self.print_progress(node)
            del node

        if self.options.verbose:
            self.print_footer()

        return Result(
            self.status,
            self.timer,
            self.iter_count,
            self.rel_gap,
            self.x,
            self.upper_bound,
            np.sum(np.abs(self.x) > self.options.int_tol),
            self.trace,
        )
