"""Mixed-Integer Programming solver for L0-regularized problems."""

import numpy as np
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
from dataclasses import dataclass
from typing import Union
from numpy.typing import ArrayLike
from pyomo.opt.results import SolverResults
from el0ps.datafits import MipDatafit
from el0ps.penalties import MipPenalty
from el0ps.solvers import BaseSolver, Result, Status

_optim_bindings = {
    "cplex": {
        "optimizer_name": "cplex_direct",
        "time_limit": "timelimit",
        "rel_tol": "mip.tolerances.mipgap",
        "int_tol": "mip.tolerances.integrality",
        "verbose": "mip.display",
    },
    "gurobi": {
        "optimizer_name": "gurobi_direct",
        "time_limit": "TimeLimit",
        "rel_tol": "MIPGap",
        "int_tol": "IntFeasTol",
        "verbose": "OutputFlag",
    },
    "mosek": {
        "optimizer_name": "mosek_direct",
        "time_limit": "dparam.mio_max_time",
        "rel_tol": "dparam.mio_tol_rel_gap",
        "int_tol": "dparam.mio_tol_abs_relax_int",
        "verbose": "iparam.log",
    },
}


@dataclass
class MipOptions:
    """:class:`.solvers.MipSolver` options.

    Parameters
    ----------
    optimizer_name: str = "gurobi"
        Mixed-Integer Programming optimizer to use. Available options are
        "cplex", "gurobi", and "mosek".
    time_limit: float
        Mixed-Integer Programming solver time limit in seconds.
    rel_tol: float
        Relative Mixed-Integer Programming tolerance.
    int_tol: float
        Integrality tolerance for a float.
    verbose: bool
        Whether to toggle solver verbosity.
    """

    optimizer_name: str = "gurobi"
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
        if self.options.optimizer_name in _optim_bindings:
            bindings = _optim_bindings[self.options.optimizer_name]
            optim = pyo.SolverFactory(bindings["optimizer_name"])
            optim.options[bindings["time_limit"]] = self.options.time_limit
            optim.options[bindings["rel_tol"]] = self.options.rel_tol
            optim.options[bindings["int_tol"]] = self.options.int_tol
            optim.options[bindings["verbose"]] = int(self.options.verbose)
        else:
            raise ValueError(
                "Solver {} not supported. Available ones are: {}".format(
                    self.options.optimizer_name,
                    _optim_bindings.keys(),
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
        iter_count = np.nan  # TODO: how to recover this with pyomo?
        solve_time = result.solver.wallclock_time
        abs_gap = np.abs(upper_bound - lower_bound)
        rel_gap = abs_gap / (np.abs(upper_bound) + 1e-10)
        x = np.array(
            [
                model.x[i].value if model.x[i].value is not None else 0.0
                for i in model.N
            ]
        )
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
