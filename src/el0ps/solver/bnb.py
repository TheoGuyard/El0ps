import sys
import pybnb as pb
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from numba import boolean, float64, int64
from numba.experimental.jitclass.base import JitClassType
from numpy.typing import ArrayLike
from typing import Union

from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty
from el0ps.compilation import CompilableClass, compiled_clone
from el0ps.solver import Status, Result


@dataclass
class NodeState:
    category: int
    S0: ArrayLike
    S1: ArrayLike
    Sb: ArrayLike
    x_lower: ArrayLike
    x_upper: ArrayLike
    upper_bound: float
    lower_bound: float

    def fix_to(self, idx: int, val: bool):
        self.Sb[idx] = False
        if val:
            self.category = 1
            self.S1[idx] = True
        else:
            self.category = 0
            self.S0[idx] = True
            self.x_lower[idx] = 0.0
            self.x_upper[idx] = 0.0


@dataclass
class TreeState:
    best_upper_bound: float


class BoundSolver(CompilableClass):

    def __init__(
        self,
        iter_limit: int = sys.maxsize,
        relative_tol: float = 1e-4,
        workingset: bool = True,
        dualpruning: bool = True,
        screening: bool = True,
        simpruning: bool = True,
    ) -> None:

        # Solver parameters
        self.iter_limit = iter_limit
        self.relative_tol = relative_tol
        self.workingset = workingset
        self.dualpruning = dualpruning
        self.screening = screening
        self.simpruning = simpruning

        # Working values
        self.S0 = np.empty(0, dtype=np.bool_)
        self.S1 = np.empty(0, dtype=np.bool_)
        self.Sb = np.empty(0, dtype=np.bool_)
        self.Sbi = np.empty(0, dtype=np.bool_)
        self.Ws = np.empty(0, dtype=np.bool_)
        self.x = np.empty(0, dtype=np.float64)
        self.w = np.empty(0, dtype=np.float64)
        self.u = np.empty(0, dtype=np.float64)
        self.v = np.empty(0, dtype=np.float64)
        self.pv = np.inf
        self.pv_old = np.inf
        self.dv = np.nan
        self.ub = np.inf
        self.lipschitz = np.nan
        self.A_colnorm = np.empty(0, dtype=np.float64)
        self.stepsize = np.empty(0, dtype=np.float64)
        self.param_slope_pos = np.empty(0, dtype=np.float64)
        self.param_slope_neg = np.empty(0, dtype=np.float64)
        self.param_limit_pos = np.empty(0, dtype=np.float64)
        self.param_limit_neg = np.empty(0, dtype=np.float64)
        self.inner_relative_tol = 0.1 * self.relative_tol
        self.flag_Ws_update = False
        self.upper = False

    def get_spec(self) -> tuple:
        spec = (
            ("iter_limit", int64),
            ("relative_tol", float64),
            ("workingset", boolean),
            ("dualpruning", boolean),
            ("screening", boolean),
            ("simpruning", boolean),
            ("S0", boolean[:]),
            ("S1", boolean[:]),
            ("Sb", boolean[:]),
            ("Sbi", boolean[:]),
            ("Ws", boolean[:]),
            ("x", float64[::1]),
            ("w", float64[::1]),
            ("u", float64[::1]),
            ("v", float64[::1]),
            ("pv", float64),
            ("pv_old", float64),
            ("dv", float64),
            ("ub", float64),
            ("lipschitz", float64),
            ("A_colnorm", float64[:]),
            ("stepsize", float64[:]),
            ("param_slope_pos", float64[:]),
            ("param_slope_neg", float64[:]),
            ("param_limit_pos", float64[:]),
            ("param_limit_neg", float64[:]),
            ("inner_relative_tol", float64),
            ("flag_Ws_update", boolean),
            ("upper", boolean),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(
            iter_limit=self.iter_limit,
            relative_tol=self.relative_tol,
            workingset=self.workingset,
            dualpruning=self.dualpruning,
            screening=self.screening,
            simpruning=self.simpruning,
        )

    def setup(
        self,
        datafit: JitClassType,
        penalty: JitClassType,
        A: ArrayLike,
        lmbd: float,
    ) -> None:
        self.lipschitz = datafit.gradient_lipschitz_constant()
        self.A_colnorm = np.sqrt(np.sum(A**2, axis=0))
        self.stepsize = 1.0 / (self.lipschitz * self.A_colnorm**2)
        self.param_slope_pos = penalty.param_slope_pos(lmbd, range(A.shape[1]))
        self.param_slope_neg = penalty.param_slope_neg(lmbd, range(A.shape[1]))
        self.param_limit_pos = penalty.param_limit_pos(lmbd, range(A.shape[1]))
        self.param_limit_neg = penalty.param_limit_neg(lmbd, range(A.shape[1]))

    def bound(
        self,
        datafit: JitClassType,
        penalty: JitClassType,
        A: ArrayLike,
        lmbd: float,
        S0: ArrayLike,
        S1: ArrayLike,
        Sb: ArrayLike,
        x: ArrayLike,
        ub: float,
        upper: bool,
    ):

        # ----- Initialize working values ----- #

        self.S0 = S0
        self.S1 = S1
        self.Sb = Sb
        self.Sbi = np.copy(Sb)

        if self.workingset and not upper:
            self.Ws = self.S1 | (x != 0.0)
        else:
            self.Ws = self.S1 | self.Sbi

        self.x = x
        self.w = np.dot(A[:, self.Ws], x[self.Ws])
        self.u = -datafit.gradient(self.w)
        self.v = np.empty_like(x)

        self.pv = np.inf
        self.pv_old = np.inf
        self.dv = np.nan
        self.ub = ub

        self.inner_relative_tol = 0.1 * self.relative_tol
        self.flag_Ws_update = False
        self.upper = upper

        while True:

            self.v = np.empty_like(x)

            for _ in range(self.iter_limit):

                self.pv_old = self.pv
                self.inner_loop(datafit, penalty, A)
                self.update_pv(datafit, penalty, lmbd)
                if self.inner_stopping_criterion():
                    break

            self.workingset_update(A)
            self.update_dv(datafit, penalty, A, lmbd)

            if upper or self.outer_stopping_criterion():
                break

            if self.dualpruning:
                if self.dv > ub:
                    break
            if self.screening:
                self.screening_tests(datafit, A)
            if self.simpruning:
                self.simpruning_tests(datafit, penalty, A, lmbd)

    def update_pv(
        self,
        datafit: JitClassType,
        penalty: JitClassType,
        lmbd: float,
    ):
        pv = datafit.value(self.w)
        for i in np.flatnonzero(self.S1):
            pv += lmbd + penalty.value_scalar(i, self.x[i])
        for i in np.flatnonzero(self.Sbi):
            if 0.0 <= self.x[i] <= self.param_limit_pos[i]:
                pv += self.param_slope_pos[i] * self.x[i]
            elif 0.0 >= self.x[i] >= self.param_limit_neg[i]:
                pv += self.param_slope_neg[i] * self.x[i]
            else:
                pv += lmbd + penalty.value_scalar(i, self.x[i])
        self.pv = pv

    def update_dv(
        self,
        datafit: JitClassType,
        penalty: JitClassType,
        A: ArrayLike,
        lmbd: float,
    ):
        dv = -datafit.conjugate(-self.u)
        for i in np.flatnonzero(self.S1):
            self.v[i] = np.dot(A[:, i], self.u)
            dv -= penalty.conjugate_scalar(i, self.v[i]) - lmbd
        for i in np.flatnonzero(self.Sbi):
            self.v[i] = np.dot(A[:, i], self.u)
            dv -= np.maximum(
                penalty.conjugate_scalar(i, self.v[i]) - lmbd, 0.0
            )
        self.dv = dv

    def inner_loop(
        self,
        datafit: JitClassType,
        penalty: JitClassType,
        A: ArrayLike,
    ):
        for i in np.flatnonzero(self.Ws):
            ai = A[:, i]
            xi = self.x[i]
            ci = xi + self.stepsize[i] * np.dot(ai, self.u)
            if self.Sb[i]:
                sni = self.stepsize[i] * self.param_slope_neg[i]
                spi = self.stepsize[i] * self.param_slope_pos[i]
                if sni <= ci <= spi:
                    self.x[i] = 0.0
                elif sni + self.param_limit_neg[i] <= ci < sni:
                    self.x[i] = ci - sni
                elif spi + self.param_limit_pos[i] >= ci > spi:
                    self.x[i] = ci - spi
                else:
                    self.x[i] = penalty.prox_scalar(i, ci, self.stepsize[i])
            elif self.S1[i]:
                self.x[i] = penalty.prox_scalar(i, ci, self.stepsize[i])
            if self.x[i] != xi:
                self.w += (self.x[i] - xi) * ai
                self.u = -datafit.gradient(self.w)

    def inner_stopping_criterion(self):
        if (
            np.abs(self.pv - self.pv_old) / np.abs(self.pv)
            <= self.inner_relative_tol
        ):
            return True
        return False

    def outer_stopping_criterion(self):
        if not self.flag_Ws_update:
            if (
                np.abs(self.pv - self.dv) / np.abs(self.pv)
                <= self.relative_tol
            ):
                return True
            if self.inner_relative_tol <= 1e-12:
                return True
            self.inner_relative_tol *= 1e-2
        return False

    def workingset_update(self, A: ArrayLike):
        self.flag_Ws_update = False
        for i in np.flatnonzero(np.logical_not(self.Ws) & self.Sbi):
            self.v[i] = np.dot(A[:, i], self.u)
            if (
                self.v[i] >= self.param_slope_pos[i]
                or self.v[i] <= self.param_slope_neg[i]
            ):
                self.Ws[i] = True
                self.flag_Ws_update = True

    def screening_tests(self, datafit: JitClassType, A: ArrayLike):
        flag = False
        r = np.sqrt(2.0 * np.abs(self.pv - self.dv) * self.lipschitz)
        for i in np.flatnonzero(self.Sbi):
            r_neg = self.param_slope_neg[i] + r * self.A_colnorm[i]
            r_pos = self.param_slope_pos[i] - r * self.A_colnorm[i]
            if r_neg < self.v[i] < r_pos:
                if self.x[i] != 0.0:
                    self.w -= self.x[i] * A[:, i]
                    self.x[i] = 0.0
                    flag = True
                self.Ws[i] = False
                self.Sbi[i] = False
        if flag:
            self.u = -datafit.gradient(self.w)

    def simpruning_tests(
        self,
        datafit: JitClassType,
        penalty: JitClassType,
        A: ArrayLike,
        lmbd: float,
    ):
        flag_update = False
        flag_passed = True
        while flag_passed:
            flag_passed = False
            for i in np.flatnonzero(self.Sb):
                p = penalty.conjugate_scalar(i, self.v[i]) - lmbd
                pp = np.maximum(p, 0.0)
                pn = np.maximum(-p, 0.0)
                if self.dv + pn > self.ub:
                    flag_passed = True
                    self.Sb[i] = False
                    self.S0[i] = True
                    self.Ws[i] = False
                    self.Sbi[i] = False
                    if self.x[i] != 0.0:
                        self.w -= self.x[i] * A[:, i]
                        self.x[i] = 0.0
                        flag_update = True
                    self.dv += pp
                elif self.dv + pp > self.ub:
                    flag_passed = True
                    self.Sb[i] = False
                    self.S1[i] = True
                    self.Ws[i] = True
                    self.Sbi[i] = False
                    self.dv += pn
        if flag_update:
            self.u = -datafit.gradient(self.w)


class ProblemWrapper(pb.Problem):

    def __init__(
        self,
        datafit: Union[BaseDatafit, JitClassType],
        penalty: Union[BaseDatafit, JitClassType],
        A: ArrayLike,
        lmbd: float,
        x_init: Union[ArrayLike, None] = None,
        bound_solver: Union[BoundSolver, JitClassType] = BoundSolver(),
    ):

        # Prepare problem data
        if isinstance(datafit, BaseDatafit) and isinstance(
            datafit, CompilableClass
        ):
            datafit = compiled_clone(datafit)
        if isinstance(penalty, BasePenalty) and isinstance(
            penalty, CompilableClass
        ):
            penalty = compiled_clone(penalty)
        if not A.flags.f_contiguous:
            A = np.array(A, order="F")
        if isinstance(bound_solver, BoundSolver):
            bound_solver = compiled_clone(bound_solver)
        if x_init is None:
            x_init = np.zeros(A.shape[1])

        # Problem data
        self.datafit = datafit
        self.penalty = penalty
        self.A = A
        self.lmbd = lmbd
        self.m, self.n = A.shape

        # Solver state
        self.tree_state = TreeState(np.inf)

        # Node state
        self.node_state = NodeState(
            -1,
            np.zeros(self.n, dtype=bool),
            np.zeros(self.n, dtype=bool),
            np.ones(self.n, dtype=bool),
            x_init,
            np.zeros(self.n),
            -np.inf,
            np.inf,
        )

        # Bounding solver
        self.bound_solver = bound_solver
        self.bound_solver.setup(datafit, penalty, A, lmbd)

        # Solver trace
        self.trace = dict()

    def sense(self):
        return pb.minimize

    def objective(self) -> float:

        if self.node_state.category == 0:
            return self.node_state.upper_bound

        if not np.any(self.node_state.Sb):
            self.node_state.x_upper = np.copy(self.node_state.x_lower)
            self.node_state.upper_bound = self.node_state.lower_bound
            return self.node_state.upper_bound

        if np.all(
            (self.node_state.x_lower[self.node_state.Sb] == 0.0)
            | (
                self.node_state.x_lower[self.node_state.Sb]
                >= self.bound_solver.param_limit_pos[self.node_state.Sb]
            )
            | (
                self.node_state.x_lower[self.node_state.Sb]
                <= self.bound_solver.param_limit_neg[self.node_state.Sb]
            )
        ):
            self.node_state.x_upper = np.copy(self.node_state.x_lower)
            self.node_state.upper_bound = self.node_state.lower_bound
            return self.node_state.upper_bound

        self.bound_solver.bound(
            self.datafit,
            self.penalty,
            self.A,
            self.lmbd,
            self.node_state.S0 | self.node_state.Sb,
            self.node_state.S1,
            np.zeros(self.n, dtype=bool),
            self.node_state.x_upper,
            self.tree_state.best_upper_bound,
            True,
        )

        self.node_state.x_upper = self.bound_solver.x
        self.node_state.upper_bound = self.bound_solver.pv

        return self.node_state.upper_bound

    def bound(self) -> float:

        self.bound_solver.bound(
            self.datafit,
            self.penalty,
            self.A,
            self.lmbd,
            self.node_state.S0,
            self.node_state.S1,
            self.node_state.Sb,
            self.node_state.x_lower,
            self.tree_state.best_upper_bound,
            False,
        )

        self.node_state.S0 = self.bound_solver.S0
        self.node_state.S1 = self.bound_solver.S1
        self.node_state.Sb = self.bound_solver.Sb
        self.node_state.x_lower = self.bound_solver.x
        self.node_state.lower_bound = self.bound_solver.dv

        return self.node_state.lower_bound

    def save_state(self, node: pb.Node) -> None:
        node.state = self.node_state

    def load_state(self, node: pb.Node) -> None:
        self.node_state = node.state

    def branch(self):

        if not np.any(self.node_state.Sb):
            return

        # TODO: make the two following lines more readable
        jSb = np.argmax(np.abs(self.node_state.x_lower[self.node_state.Sb]))
        j = np.arange(self.n)[self.node_state.Sb][jSb]

        child = pb.Node()
        child.state = deepcopy(self.node_state)
        child.state.fix_to(j, 0)
        yield child

        child = pb.Node()
        child.state = deepcopy(self.node_state)
        child.state.fix_to(j, 1)
        yield child

    def notify_solve_begins(
        self,
        comm,
        worker_comm,
        convergence_checker,
    ):
        pass

    def notify_new_best_node(
        self,
        node: pb.Node,
        current: bool,
    ):
        if current:
            self.tree_state.best_upper_bound = self.node_state.upper_bound

    def notify_solve_finished(
        self,
        comm,
        worker_comm,
        results,
    ):
        pass


class BnbSolver:

    def __init__(
        self,
        relative_gap: float = 1e-8,
        absolute_gap: float = 0.0,
        node_limit: int = sys.maxsize,
        time_limit: float = np.inf,
        queue_limit: int = sys.maxsize,
        queue_strategy: str = "bound",
        verbose: bool = True,
        **kwargs,
    ):
        self.solver = pb.Solver()
        self.relative_gap = relative_gap
        self.absolute_gap = absolute_gap
        self.node_limit = node_limit
        self.time_limit = time_limit
        self.queue_limit = queue_limit
        self.queue_strategy = queue_strategy
        self.verbose = verbose

        self.bound_solver = compiled_clone(BoundSolver(**kwargs))

    def package_results(self, problem, results):

        status = Status.UNKNOWN
        if results.solution_status == pb.SolutionStatus.optimal:
            status = Status.OPTIMAL
        elif results.solution_status == pb.SolutionStatus.invalid:
            absolute_gap = np.abs(results.objective - results.bound)
            relative_gap = np.abs(results.objective - results.bound) / max(
                1.0, np.abs(results.objective)
            )
            if (
                absolute_gap <= self.absolute_gap
                or relative_gap <= self.relative_gap
            ):
                status = Status.OPTIMAL
            if (
                results.termination_condition
                == pb.TerminationCondition.optimality
            ):
                status = Status.OPTIMAL
        elif results.solution_status == pb.SolutionStatus.feasible:
            if (
                results.termination_condition
                == pb.TerminationCondition.node_limit
            ):
                status = Status.ITER_LIMIT
            elif (
                results.termination_condition
                == pb.TerminationCondition.time_limit
            ):
                status = Status.TIME_LIMIT
            elif (
                results.termination_condition
                == pb.TerminationCondition.queue_limit
            ):
                status = Status.MEMORY_LIMIT
            else:
                status = Status.OPTIMAL
        elif results.solution_status == pb.SolutionStatus.infeasible:
            status = Status.INFEASIBLE
        elif results.solution_status == pb.SolutionStatus.unbounded:
            status = Status.UNBOUNDED

        return Result(
            status,
            results.wall_time,
            results.nodes,
            results.best_node.state.x_upper,
            results.objective,
            problem.trace,
        )

    def solve(
        self,
        datafit: Union[BaseDatafit, JitClassType],
        penalty: Union[BaseDatafit, JitClassType],
        A: ArrayLike,
        lmbd: float,
        x_init: Union[ArrayLike, None] = None,
    ):

        problem = ProblemWrapper(
            datafit, penalty, A, lmbd, x_init, self.bound_solver
        )

        results = self.solver.solve(
            problem,
            absolute_gap=self.absolute_gap,
            relative_gap=self.relative_gap,
            node_limit=self.node_limit,
            time_limit=self.time_limit,
            queue_limit=self.queue_limit,
            queue_strategy=self.queue_strategy,
            log=pb.solver._notset if self.verbose else None,
        )

        return self.package_results(problem, results)
