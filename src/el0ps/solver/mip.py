"""Mixed-Integer Programming solver for L0-regularized problems."""

import numpy as np
import pyomo.environ as pyo
import pyomo.kernel as pmo
import sys
from typing import Union
from numpy.typing import ArrayLike
from pyomo.opt.results import SolverResults
from el0ps.datafit import MipDatafit
from el0ps.penalty import MipPenalty
from el0ps.solver import BaseSolver, Result, Status


_mip_optim_bindings = {
    "cplex": {
        "optimizer_name": "cplex_persistent",
        "relative_gap": "mip_tolerances_mipgap",
        "absolute_gap": "mip_tolerances_absmipgap",
        "node_limit": "mip_limits_nodes",
        "time_limit": "timelimit",
        "verbose": "mip_display",
    },
    "gurobi": {
        "optimizer_name": "gurobi_persistent",
        "relative_gap": "MIPGap",
        "absolute_gap": "MIPGapAbs",
        "node_limit": "NodeLimit",
        "time_limit": "TimeLimit",
        "verbose": "OutputFlag",
    },
    "mosek": {
        "optimizer_name": "mosek_persistent",
        "relative_gap": "dparam.mio_tol_rel_gap",
        "absolute_gap": "dparam.mio_tol_abs_gap",
        "node_limit": "dparam.mio_max_nodes",
        "time_limit": "dparam.mio_max_time",
        "verbose": "iparam.log",
    },
}


class MipSolver(BaseSolver):
    """Mixed-Integer Programming solver for L0-regularized problems.

    Parameters
    ----------
    optimizer_name: str = "gurobi"
        Mixed-Integer Programming optimizer to use. Available options are
        "cplex", "gurobi", and "mosek".
    relative_gap: float
        Relative Mixed-Integer Programming tolerance.
    absolute_gap: float
        Absolute Mixed-Integer Programming tolerance.
    node_limit: int
        Mixed-Integer Programming number of nodes explored limit.
    time_limit: float
        Mixed-Integer Programming solver time limit in seconds.
    queue_limit: int
        Mixed-Integer Programming queue size limit.
    verbose: bool
        Whether to toggle solver verbosity.
    """

    def __init__(
        self,
        optimizer_name: str = "gurobi",
        relative_gap: float = 1e-8,
        absolute_gap: float = 0.0,
        node_limit: int = sys.maxsize,
        time_limit: float = float(sys.maxsize),
        verbose: bool = False,
    ) -> None:
        self.optimizer_name = optimizer_name
        self.node_limit = node_limit
        self.time_limit = time_limit
        self.relative_gap = relative_gap
        self.absolute_gap = absolute_gap
        self.verbose = verbose

    def __str__(self):
        return "MipSolver"

    def initialize_optimizer(self):
        if self.optimizer_name in _mip_optim_bindings:
            bindings = _mip_optim_bindings[self.optimizer_name]
            optim = pyo.SolverFactory(bindings["optimizer_name"])
            optim.options[bindings["relative_gap"]] = self.relative_gap
            optim.options[bindings["absolute_gap"]] = self.absolute_gap
            # Raises errors with cplex
            # optim.options[bindings["node_limit"]] = self.node_limit
            optim.options[bindings["time_limit"]] = self.time_limit
            optim.options[bindings["verbose"]] = int(self.verbose)
        else:
            raise ValueError(
                "Solver {} not supported. Available ones are: {}".format(
                    self.optimizer_name,
                    _mip_optim_bindings.keys(),
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
        iter_count = -1  # TODO: how to recover this with pyomo?
        solve_time = result.solver.wallclock_time
        x = np.array(
            [
                model.x[i].value if model.x[i].value is not None else 0.0
                for i in model.N
            ]
        )

        return Result(
            status,
            solve_time,
            iter_count,
            x,
            upper_bound,
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
        optim.set_instance(model)

        if x_init is not None:
            assert len(x_init) == A.shape[1]
            for i, xi in enumerate(x_init):
                model.x[i].set_value(xi)

        result = optim.solve(model, warmstart=True)

        return self.package_result(model, result)
