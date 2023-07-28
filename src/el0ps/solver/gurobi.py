import gurobipy as gp
import numpy as np
from typing import Union
from numpy.typing import NDArray
from el0ps import Problem
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

    def bind_model_f_var(
        self,
        problem: Problem,
        model: gp.Model,
        x_var: gp.MVar,
        f_var: gp.Var,
    ) -> None:
        if str(problem.datafit) == "Quadratic":
            r_var = problem.datafit.y - problem.A @ x_var
            model.addConstr(f_var >= (r_var @ r_var) / (2.0 * problem.m))
        else:
            raise NotImplementedError(
                "`GurobiSolver` does not support `{}` yet.".format(
                    type(problem.datafit)
                )
            )

    def bind_model_g_var(
        self,
        problem: Problem,
        model: gp.Model,
        x_var: gp.MVar,
        z_var: gp.MVar,
        g_var: gp.Var,
    ) -> None:
        if str(problem.penalty) == "Bigm":
            model.addConstr(g_var >= problem.lmbd * sum(z_var))
            model.addConstr(x_var <= problem.penalty.M * z_var)
            model.addConstr(x_var >= -problem.penalty.M * z_var)
        elif str(problem.penalty) == "L2norm":
            s_var = model.addMVar(problem.n, vtype="C", name="s")
            model.addConstr(s_var >= 0.0)
            model.addConstr(x_var * x_var <= s_var * z_var)
            model.addConstr(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(s_var)
            )
        else:
            raise NotImplementedError(
                "`GurobiSolver` does not support `{}` yet.".format(
                    type(problem.penalty)
                )
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

        model = gp.Model()
        f_var = model.addVar(vtype="C", name="f")
        g_var = model.addVar(vtype="C", name="g")
        x_var = model.addMVar(problem.n, vtype="C", name="x", lb=-np.inf)
        z_var = model.addMVar(problem.n, vtype="B", name="z")
        self.bind_model_f_var(problem, model, x_var, f_var)
        self.bind_model_g_var(problem, model, x_var, z_var, g_var)
        model.setObjective(f_var + g_var, gp.GRB.MINIMIZE)
        if relax:
            model.relax()

        self.model = model
        self.x_var = x_var
        self.z_var = z_var
        self.f_var = f_var
        self.g_var = g_var

    def set_init(
        self,
        x_init: Union[NDArray, None] = None,
        S0_init: Union[NDArray, None] = None,
        S1_init: Union[NDArray, None] = None,
    ) -> None:
        if S0_init is not None:
            for i in S0_init:
                self.model.addConstr(self.x_var[i] == 0.0)
                self.model.addConstr(self.z_var[i] == 0.0)
        if S1_init is not None:
            for i in S1_init:
                self.model.addConstr(self.z_var[i] == 1.0)
        if x_init is not None:
            for i in range(x_init.shape):
                self.x_var[i].Start = x_init[i]

    def set_options(self) -> None:
        for k, v in self.options.items():
            self.model.setParam(k, v)

    def get_status(self) -> Status:
        if self.model.Status == gp.GRB.OPTIMAL:
            status = Status.OPTIMAL
        elif self.model.Status == gp.GRB.NODE_LIMIT:
            status = Status.NODE_LIMIT
        elif self.model.Status == gp.GRB.TIME_LIMIT:
            status = Status.TIME_LIMIT
        else:
            status = Status.OTHER_LIMIT
        return status

    def solve(
        self,
        problem: Problem,
        x_init: Union[NDArray, None] = None,
        S0_init: Union[NDArray, None] = None,
        S1_init: Union[NDArray, None] = None,
    ) -> Results:
        self.build_model(problem)
        self.set_init(x_init, S0_init, S1_init)
        self.set_options()
        self.model.optimize()

        return Results(
            self.get_status(),
            self.model.Runtime,
            int(self.model.NodeCount),
            self.model.ObjVal,
            self.model.ObjBound,
            np.array(self.x_var.X),
            np.array(self.z_var.X),
            None,
        )
