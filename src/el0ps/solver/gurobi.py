import gurobipy as gp
import numpy as np
from numpy.typing import NDArray
from el0ps import Problem
from .base import BaseSolver, Results, Status


class GurobiSolver(BaseSolver):
    def __init__(self, options: dict):
        self.options = options

    def solve(
        self,
        problem: Problem,
        x_init: NDArray | None = None,
        S0_init: NDArray | None = None,
        S1_init: NDArray | None = None,
    ):
        n = problem.A.shape[1]
        model = gp.Model()
        x_var = model.addMVar(n, vtype=gp.GRB.CONTINUOUS, name="x")
        z_var = model.addMVar(n, vtype=gp.GRB.BINARY, name="z")
        f_var = model.addVar(vtype=gp.GRB.CONTINUOUS, name="f")
        g_var = model.addVar(vtype=gp.GRB.CONTINUOUS, name="g")
        model.setObjective(f_var + g_var, gp.GRB.MINIMIZE)

        problem.datafit.bind_model_cost(model, problem.A, x_var, f_var)
        problem.penalty.bind_model_cost(
            model, problem.lmbd, x_var, z_var, g_var
        )

        if S0_init is not None:
            for i in S0_init:
                model.addConstr(x_var[i] == 0.0)
                model.addConstr(z_var[i] == 0.0)
        if S1_init is not None:
            for i in S1_init:
                model.addConstr(z_var[i] == 1.0)
        if x_init is not None:
            for i in range(n):
                x_var[i].Start = x_init[i]

        for k, v in self.options.items():
            model.setParam(k, v)

        model.optimize()

        if model.Status == gp.GRB.OPTIMAL:
            termination_status = Status.OPTIMAL
        elif model.Status == gp.GRB.NODE_LIMIT:
            termination_status = Status.NODE_LIMIT
        elif model.Status == gp.GRB.TIME_LIMIT:
            termination_status = Status.TIME_LIMIT
        else:
            termination_status = Status.OTHER_LIMIT

        return Results(
            termination_status,
            model.Runtime,
            model.NodeCount,
            model.ObjVal,
            model.ObjVal,
            model.ObjBound,
            np.array(x_var.X),
            np.array(z_var.X),
            None,
        )
