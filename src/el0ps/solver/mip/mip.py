"""Generic Mixed-Integer Programming solver for L0-penalized problems."""

import numpy as np
import pyomo.environ as pyo
import sys
from dataclasses import dataclass
from typing import Literal, Union
from pyomo.opt.results import SolverResults
from numpy.typing import NDArray
from el0ps.datafit import ModelableDatafit
from el0ps.penalty import ModelablePenalty
from el0ps.solver import BaseSolver, Result, Status


@dataclass
class MipOptions:
    """:class:`.solver.MipSolver` options.

    Parameters
    ----------
    time_limit: float
        Mixed-Integer Programming solver time limit in seconds.
    rel_tol: float
        Relative Mixed-Integer Programming tolerance.
    int_tol: float
        Integrality tolerance for a float.
    verbose: bool
        Whether to toggle solver verbosity.
    """

    optimizer_name: Literal["cplex", "gurobi", "mosek"] = "gurobi"
    time_limit: float = float(sys.maxsize)
    rel_tol: float = 1e-4
    int_tol: float = 1e-8
    verbose: bool = False


class MipSolver(BaseSolver):
    """Mixed-Integer Programming solver for L0-penalized problems."""

    def __init__(self, **kwargs) -> None:
        self.options = MipOptions(**kwargs)

    def __str__(self):
        return "MipSolver"

    def initialize_optimizer(self):
        if self.options.optimizer_name == "cplex":
            optim = pyo.SolverFactory("cplex_direct")
            optim.options["timelimit"] = self.options.time_limit
            optim.options["mip.tolerances.mipgap"] = self.options.rel_tol
            optim.options["mip.tolerances.integrality"] = self.options.int_tol
            optim.options["mip.display"] = int(self.options.verbose)
        elif self.options.optimizer_name == "gurobi":
            optim = pyo.SolverFactory("gurobi_direct")
            optim.options["TimeLimit"] = self.options.time_limit
            optim.options["MIPGap"] = self.options.rel_tol
            optim.options["IntFeasTol"] = self.options.int_tol
            optim.options["OutputFlag"] = self.options.verbose
        elif self.options.optimizer_name == "mosek":
            optim = pyo.SolverFactory("mosek_direct")
            optim.options["dparam.mio_max_time"] = self.options.time_limit
            optim.options["dparam.mio_tol_rel_gap"] = self.options.rel_tol
            optim.options[
                "dparam.mio_tol_abs_relax_int"
            ] = self.options.int_tol
            optim.options["iparam.log"] = int(self.options.verbose)
        else:
            raise ValueError(
                "Unsupported optimizer '{}'.".format(
                    self.options.optimizer_name
                )
            )
        return optim

    def build_model(
        self,
        datafit: ModelableDatafit,
        penalty: ModelablePenalty,
        A: NDArray,
        lmbd: float,
    ):
        model = pyo.ConcreteModel()
        model.M = pyo.RangeSet(0, A.shape[0] - 1)
        model.N = pyo.RangeSet(0, A.shape[1] - 1)
        model.A = pyo.Param(model.M, model.N, within=pyo.Reals)
        model.x = pyo.Var(model.N, within=pyo.Reals)
        model.w = pyo.Var(model.M, within=pyo.Reals)
        model.z = pyo.Var(model.N, within=pyo.Binary)
        model.f = pyo.Var(within=pyo.Reals)
        model.g = pyo.Var(within=pyo.Reals)

        def w_con_rule(model: pyo.Model, j: int):
            return model.w[j] == sum(A[j, i] * model.x[i] for i in model.N)

        model.w_con = pyo.Constraint(model.M, rule=w_con_rule)
        datafit.bind_model(model)
        penalty.bind_model(model, lmbd)

        model.obj = pyo.Objective(expr=model.f + model.g)
        return model

    def package_result(self, model: pyo.Model, result: SolverResults):
        if result.solver.termination_condition == "optimal":
            status = Status.OPTIMAL
        elif result.solver.termination_condition == "maxIterations":
            status = Status.NODE_LIMIT
        elif result.solver.termination_condition == "maxTimeLimit":
            status = Status.TIME_LIMIT
        else:
            status = Status.UNKNOWN

        upper_bound = result.problem.upper_bound
        lower_bound = result.problem.lower_bound
        iter_count = -1
        solve_time = result.solver.wallclock_time

        x = np.array([model.x[i].value for i in model.N])
        z = np.array([model.z[i].value for i in model.N])

        return Result(
            status,
            solve_time,
            iter_count,
            np.abs(upper_bound - lower_bound) / (np.abs(upper_bound) + 1e-10),
            x,
            z,
            upper_bound,
            np.sum(np.abs(x) > self.options.int_tol),
            {},
        )

    def solve(
        self,
        datafit: ModelableDatafit,
        penalty: ModelablePenalty,
        A: NDArray,
        lmbd: float,
        x_init: Union[NDArray, None] = None,
    ):
        if x_init is None:
            x_init = np.zeros(A.shape[1])

        optim = self.initialize_optimizer()
        model = self.build_model(datafit, penalty, A, lmbd)

        for i, xi in enumerate(x_init):
            model.x[i] = xi
            model.z[i] = 0.0 if xi == 0.0 else 1.0

        result = optim.solve(model)

        return self.package_result(model, result)
