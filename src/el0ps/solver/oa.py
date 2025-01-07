"""Outer-Approximation solver for L0-regularized problems."""

import numpy as np
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
from time import time
from typing import Union
from numba.experimental.jitclass.base import JitClassType
from numpy.typing import ArrayLike
from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty
from el0ps.solver import BaseSolver, Result, Status
from el0ps.solver.mip import _mip_optim_bindings


class OaSolver(BaseSolver):
    """Outer Approximation solver for L0-regularized problems.

    Parameters
    ----------
    optimizer_name: str = "gurobi"
        Mixed-Integer Programming optimizer to use. Available options are
        "cplex", "gurobi", and "mosek".
    relative_gap: float
        Relative Mixed-Integer Programming tolerance.
    absolute_gap: float
        Absolute Mixed-Integer Programming tolerance.
    time_limit: float
        Outer-Approximation solver time limit in seconds.
    iter_limit: float
        Outer-Approximation solver iteration limit.
    inner_iter_limit: float
        Inner solver iteration limit.
    inner_rel_tol: float
        Inner solver tolerance.
    verbose: bool
        Whether to toggle solver verbosity.
    keeptrace: bool
        Whether to store the solver trace.
    """

    _trace_keys = [
        "timer",
        "timer_outer",
        "timer_inner",
        "iter_count",
        "upper_bound",
        "lower_bound",
    ]

    def __init__(
        self,
        optimizer_name: str = "gurobi",
        relative_gap: float = 1e-8,
        absolute_gap: float = 0.0,
        time_limit: float = float(sys.maxsize),
        iter_limit: int = sys.maxsize,
        inner_iter_limit: int = sys.maxsize,
        inner_rel_tol: float = 1e-4,
        verbose: bool = False,
        keeptrace: bool = False,
    ) -> None:
        self.optimizer_name = optimizer_name
        self.relative_gap = relative_gap
        self.absolute_gap = absolute_gap
        self.time_limit = time_limit
        self.iter_limit = iter_limit
        self.inner_iter_limit = inner_iter_limit
        self.inner_rel_tol = inner_rel_tol
        self.verbose = verbose
        self.keeptrace = keeptrace

    def __str__(self):
        return "OaSolver"

    def initialize_optimizer(self):
        if self.optimizer_name in _mip_optim_bindings:
            bindings = _mip_optim_bindings[self.optimizer_name]
            optim = pyo.SolverFactory(bindings["optimizer_name"])
            optim.options[bindings["time_limit"]] = self.time_limit
            optim.options[bindings["relative_gap"]] = self.relative_gap
            optim.options[bindings["absolute_gap"]] = self.absolute_gap
            optim.options[bindings["verbose"]] = int(self.verbose)
        else:
            raise ValueError(
                "Solver {} not supported. Available ones are: {}".format(
                    self.optimizer_name,
                    _mip_optim_bindings.keys(),
                )
            )
        return optim

    def initialize_inner_solver(self):
        self.lipschitz = self.datafit.gradient_lipschitz_constant()
        self.A_colnorm = np.linalg.norm(self.A, ord=2, axis=0)
        self.stepsize = 1.0 / (self.lipschitz * self.A_colnorm**2)

    def build_model(self) -> None:
        model = pmo.block()
        model.N = range(self.A.shape[1])
        model.z = pmo.variable_dict()
        model.t = pmo.variable(domain=pmo.Reals)
        for i in model.N:
            model.z[i] = pmo.variable(domain=pmo.Binary)
        model.obj = pmo.objective(
            self.lmbd * sum(model.z[i] for i in model.N) + model.t
        )
        model.t_con = pmo.constraint_dict()
        return model

    def add_cut(
        self, optim, model, z: ArrayLike, f: float, g: ArrayLike
    ) -> None:
        self.cuts_count += 1
        model.t_con[self.cuts_count] = pmo.constraint(
            model.t >= f + sum(g[i] * (model.z[i] - z[i]) for i in model.N)
        )
        optim.add_constraint(model.t_con[self.cuts_count])

    def outer_solve(self, optim, model):
        start_time_outer = time()
        result = optim.solve(model, warmstart=True, save_results=False)
        self.timer_outer += time() - start_time_outer
        zk = np.array([model.z[i].value for i in model.N])
        vk = result.problem.lower_bound
        return zk, vk

    def inner_solve(self, z):

        start_time_inner = time()

        Z = z != 0.0
        x = np.zeros(self.A.shape[1])
        x[Z] = np.copy(self.x[Z])
        w = self.A[:, Z] @ x[Z]
        u = -self.datafit.gradient(w)

        pv = np.inf
        for _ in range(self.inner_iter_limit):
            pv_old = pv
            for i in np.flatnonzero(Z):
                ai = self.A[:, i]
                xi = x[i]
                ci = xi + self.stepsize[i] * np.dot(ai, u)
                x[i] = self.penalty.prox_scalar(i, ci, self.stepsize[i])
                if x[i] != xi:
                    w += (x[i] - xi) * ai
                    u[:] = -self.datafit.gradient(w)
            pv = self.datafit.value(w) + self.penalty.value(x)
            rtol = np.abs(pv - pv_old) / (np.abs(pv) + 1e-10)
            if rtol < self.inner_rel_tol:
                break

        v = self.A.T @ u
        g = np.array(
            [self.penalty.conjugate_scalar(i, vi) for i, vi in enumerate(v)]
        )
        dv = -self.datafit.conjugate(-u) - np.dot(z, g)
        self.timer_inner += time() - start_time_inner

        return x, dv, -g

    @property
    def abs_gap(self):
        return self.upper_bound - self.lower_bound

    @property
    def rel_gap(self):
        return (self.upper_bound - self.lower_bound) / (
            np.abs(self.upper_bound) + 1e-10
        )

    @property
    def timer(self):
        return time() - self.start_time

    def print_header(self):
        s = "-" * 50 + "\n"
        s += "|"
        s += " {:>6}".format("Iters")
        s += " {:>6}".format("Timer")
        s += " {:>6}".format("Lower")
        s += " {:>6}".format("Upper")
        s += " {:>9}".format("Abs gap")
        s += " {:>9}".format("Rel gap")
        s += "|" + "\n"
        s += "-" * 50
        print(s)

    def print_progress(self):
        s = "|"
        s += " {:>6d}".format(self.iter_count)
        s += " {:>6.2f}".format(self.timer)
        s += " {:>6.2f}".format(self.lower_bound)
        s += " {:>6.2f}".format(self.upper_bound)
        s += " {:>9.2e}".format(self.abs_gap)
        s += " {:>9.2e}".format(self.rel_gap)
        s += "|"
        print(s)

    def print_footer(self):
        s = "-" * 50
        print(s)

    def update_bounds(self, zk, vk, xk, fk) -> None:
        if self.lower_bound < vk:
            self.lower_bound = vk
        if self.upper_bound > fk + self.lmbd * np.sum(zk):
            self.upper_bound = fk + self.lmbd * np.sum(zk)
            self.x = xk

    def update_trace(self) -> None:
        for key in self._trace_keys:
            self.trace[key].append(getattr(self, key))

    def can_continue(self):
        if self.timer >= self.time_limit:
            self.status = Status.TIME_LIMIT
        elif self.iter_count >= self.iter_limit:
            self.status = Status.ITER_LIMIT
        elif self.rel_gap < self.relative_gap:
            self.status = Status.OPTIMAL
        return self.status == Status.RUNNING

    def solve(
        self,
        datafit: Union[BaseDatafit, JitClassType],
        penalty: Union[BasePenalty, JitClassType],
        A: ArrayLike,
        lmbd: float,
        x_init: Union[ArrayLike, None] = None,
    ):

        self.datafit = datafit
        self.penalty = penalty
        self.A = A
        self.lmbd = lmbd

        self.status = Status.RUNNING
        self.iter_count = 0
        self.cuts_count = 0
        self.timer_outer = 0.0
        self.timer_inner = 0.0
        self.x = x_init if x_init is not None else np.zeros(self.A.shape[1])
        self.upper_bound = (
            self.datafit.value(self.A @ self.x)
            + self.lmbd * np.linalg.norm(self.x, 0)
            + self.penalty.value(self.x)
        )
        self.lower_bound = -np.inf
        self.trace = {k: [] for k in self._trace_keys}

        self.start_time = time()

        optim = self.initialize_optimizer()
        model = self.build_model()
        optim.set_instance(model)

        self.initialize_inner_solver()

        if self.verbose:
            self.print_header()

        z0 = np.array(self.x != 0.0, dtype=int)
        _, f0, g0 = self.inner_solve(z0)
        self.add_cut(optim, model, z0, f0, g0)

        while self.can_continue():
            self.iter_count += 1
            zk, vk = self.outer_solve(optim, model)
            xk, fk, gk = self.inner_solve(zk)
            self.add_cut(optim, model, zk, fk, gk)
            self.update_bounds(zk, vk, xk, fk)
            if self.verbose:
                self.print_progress()
            if self.keeptrace:
                self.update_trace()

        if self.verbose:
            self.print_footer()

        return Result(
            self.status,
            self.timer,
            self.iter_count,
            self.x,
            self.upper_bound,
            self.trace,
        )
