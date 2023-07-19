import gurobipy as gp
import numpy as np
from typing import Union
from numpy.typing import NDArray
from el0ps import Problem
from el0ps.datafit import Quadratic
from el0ps.penalty import Bigm, L2norm
from .base import BaseSolver, Results, Status


class GurobiSolver(BaseSolver):
    """Gurobi solver for L0-penalized problems.

    Parameters
    ----------
    options: dict
        Gurobi options set with `model.setParam(option_name, option_value)`.
        See `https://www.gurobi.com/documentation` for more details.
    """

    _default_options = {
        "OutputFlag": 0.0,
        "MIPGap": 1e-4,
        "MIPGapAbs": 1e-8,
        "IntFeasTol": 1e-8,
    }

    def __init__(self, options: dict = _default_options):
        self.options = options

    def _bind_model_f_var(
        self,
        problem: Problem,
        model: gp.Model,
        x_var: gp.MVar,
        f_var: gp.Var,
    ) -> None:
        if isinstance(problem.datafit, Quadratic):
            r = problem.datafit.y - problem.A @ x_var
            model.addConstr(
                f_var >= gp.quicksum(r * r) / (2.0 * problem.datafit.y.size)
            )
        else:
            raise NotImplementedError(
                f"`GurobiSolver` does not support a `{type(problem.datafit)}`"
                "data-fidelity function yet."
            )

    def _bind_model_g_var(
        self,
        problem: Problem,
        model: gp.Model,
        x_var: gp.MVar,
        z_var: gp.MVar,
        g_var: gp.Var,
    ) -> None:
        if isinstance(problem.penalty, Bigm):
            model.addConstr(g_var >= problem.lmbd * sum(z_var))
            model.addConstr(x_var <= problem.penalty.M * z_var)
            model.addConstr(x_var >= -problem.penalty.M * z_var)
        elif isinstance(problem.penalty, L2norm):
            s_var = model.addMVar(
                x_var.size, vtype=gp.GRB.CONTINUOUS, name="s"
            )
            model.addConstr(s_var >= 0.0)
            model.addConstr(x_var * x_var <= s_var * z_var)
            model.addConstr(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(s_var)
            )
        else:
            raise NotImplementedError(
                f"`GurobiSolver` does not support a `{type(problem.penalty)}`"
                "penalty function yet."
            )

    def build_model(self, problem: Problem, relax: bool = False) -> None:
        """Build the following MIP model of the L0-penalized problem

            min f_var + g_var                               (1)
            st  f_var >= f(A * x_var)                       (2)
                g_var >= lmbd * norm(x_var, 0) + h(x_var)   (3)
                f_var real, g_var real, x_var vector        (4)

        Constraint (2) is set by `self._bind_model_f_var()` and constraint (3)
        is set by `self._bind_model_g_var()`, depending on the Problem data
        fidelity and penalty functions.

        Paramaters
        ----------
        problem: Problem
            The L0-penalized problem to be solved.
        relax: bool = False
            Whether to relax integrality constraints on the binary variable
            coding the nullity in x.
        """

        m, n = problem.A.shape
        model = gp.Model()
        f_var = model.addVar(vtype=gp.GRB.CONTINUOUS, name="f")
        g_var = model.addVar(vtype=gp.GRB.CONTINUOUS, name="g")
        x_var = model.addMVar(n, vtype=gp.GRB.CONTINUOUS, name="x")
        if relax:
            z_var = model.addMVar(
                n, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z"
            )  # noqa
        else:
            z_var = model.addMVar(n, vtype=gp.GRB.BINARY, name="z")
        model.setObjective(f_var + g_var, gp.GRB.MINIMIZE)
        self._bind_model_f_var(problem, model, x_var, f_var)
        self._bind_model_g_var(problem, model, x_var, z_var, g_var)

        self.model = model
        self.x_var = x_var
        self.z_var = z_var
        self.f_var = f_var
        self.g_var = g_var

    def solve(
        self,
        problem: Problem,
        x_init: Union[NDArray, None] = None,
        S0_init: Union[NDArray, None] = None,
        S1_init: Union[NDArray, None] = None,
    ) -> Results:
        self.build_model(problem)

        if S0_init is not None:
            for i in S0_init:
                self.model.addConstr(self.x_var[i] == 0.0)
                self.model.addConstr(self.z_var[i] == 0.0)
        if S1_init is not None:
            for i in S1_init:
                self.model.addConstr(self.z_var[i] == 1.0)
        if x_init is not None:
            for i in range(problem.A.shape[1]):
                self.x_var[i].Start = x_init[i]

        for k, v in self.options.items():
            self.model.setParam(k, v)

        self.model.optimize()

        if self.model.Status == gp.GRB.OPTIMAL:
            status = Status.OPTIMAL
        elif self.model.Status == gp.GRB.NODE_LIMIT:
            status = Status.NODE_LIMIT
        elif self.model.Status == gp.GRB.TIME_LIMIT:
            status = Status.TIME_LIMIT
        else:
            status = Status.OTHER_LIMIT

        return Results(
            status,
            self.model.Runtime,
            int(self.model.NodeCount),
            self.model.ObjVal,
            self.model.ObjVal,
            self.model.ObjBound,
            np.array(self.x_var.X),
            np.array(self.z_var.X),
            None,
        )
