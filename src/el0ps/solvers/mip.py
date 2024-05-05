"""Mixed-Integer Programming solver for L0-regularized problems."""

import numpy as np
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
from dataclasses import dataclass
from typing import Literal, Union
from numpy.typing import ArrayLike
from pyomo.opt.results import SolverResults
from el0ps.datafits import MipDatafit
from el0ps.penalties import MipPenalty
from el0ps.solvers import BaseSolver, Result, Status


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
    """Mixed-Integer Programming solver for L0-regularized problems."""

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
            optim.options["dparam.mio_tol_abs_relax_int"] = (
                self.options.int_tol
            )
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
        datafit: MipDatafit,
        penalty: MipPenalty,
        A: ArrayLike,
        lmbd: float,
    ):
        model = pmo.block()
        model.M = range(A.shape[0])
        model.N = range(A.shape[1])
        model.x = pmo.variable_dict()
        model.z = pmo.variable_dict()
        for i in model.N:
            model.x[i] = pmo.variable(domain=pmo.Reals)
            model.z[i] = pmo.variable(domain=pmo.Binary)
        model.w = pmo.variable_dict()
        for j in model.M:
            model.w[j] = pmo.variable(domain=pmo.Reals)
        model.f = pmo.variable(domain=pmo.Reals)
        model.g = pmo.variable(domain=pmo.Reals)

        model.w_con = pmo.constraint_dict()
        for j in model.M:
            model.w_con[j] = pmo.constraint(
                model.w[j] == sum(A[j, i] * model.x[i] for i in model.N)
            )

        datafit.bind_model(model)
        penalty.bind_model(model, lmbd)

        model.obj = pmo.objective(model.f + model.g)
        return model

    def package_result(self, model: pmo.block, result: SolverResults):
        if result.solver.termination_condition == "optimal":
            status = Status.OPTIMAL
        elif result.solver.termination_condition == "maxIterations":
            status = Status.ITER_LIMIT
        elif result.solver.termination_condition == "maxTimeLimit":
            status = Status.TIME_LIMIT
        else:
            status = Status.UNKNOWN

        upper_bound = result.problem.upper_bound
        lower_bound = result.problem.lower_bound
        iter_count = -1  # TODO: recover number of iterations
        solve_time = result.solver.wallclock_time
        abs_gap = np.abs(upper_bound - lower_bound)
        rel_gap = abs_gap / (np.abs(upper_bound) + 1e-10)
        x = np.array([model.x[i].value for i in model.N])
        n_nnz = np.sum(np.abs(x) > self.options.int_tol)

        return Result(
            status,
            solve_time,
            iter_count,
            rel_gap,
            x,
            upper_bound,
            n_nnz,
            None,
        )

    def solve(
        self,
        datafit: MipDatafit,
        penalty: MipPenalty,
        A: ArrayLike,
        lmbd: float,
        x_init: Union[ArrayLike, None] = None,
    ):

        optim = self.initialize_optimizer()
        model = self.build_model(datafit, penalty, A, lmbd)

        if x_init is not None:
            assert len(x_init) == A.shape[1]
            for i, xi in enumerate(x_init):
                model.x[i].set_value(xi)

        result = optim.solve(model, warmstart=True)

        return self.package_result(model, result)
