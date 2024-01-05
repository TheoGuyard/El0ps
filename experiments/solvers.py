import re
import sys
import gurobipy as gp
import mosek.fusion as msk
import numpy as np
import osqp
from docplex.mp.model import Model
from docplex.mp.dvar import Var
from typing import Union
from numpy.typing import NDArray
from l0bnb import BNBTree
from scipy import sparse
from el0ps import Problem
from el0ps.solver import (
    BaseSolver,
    BnbSolver,
    BnbNode,
    Status,
    Results,
    BnbBranchingStrategy,
    BnbExplorationStrategy,
)
from el0ps.solver.bounding import BoundingSolver


class CplexSolver(BaseSolver):
    """Cplex solver for L0-penalized problems."""

    def __init__(
        self,
        time_limit: float = float(sys.maxsize),
        rel_tol: float = 1e-4,
        int_tol: float = 1e-8,
        verbose: bool = False,
    ):
        self.options = {
            "time_limit": time_limit,
            "rel_tol": rel_tol,
            "int_tol": int_tol,
            "verbose": verbose,
        }

    def __str__(self):
        return "CplexSolver"

    def bind_model_f_var(
        self,
        problem: Problem,
        model: Model,
        x_var: Var,
        f_var: Var,
    ) -> None:
        if str(problem.datafit) == "Leastsquares":
            f1_var = model.continuous_var_list(
                problem.m, name="f1", lb=-np.inf
            )
            model.add_constraints(
                f1_var[j]
                == problem.datafit.y[j] - model.dot(x_var, problem.A[j, :])
                for j in range(problem.m)
            )
            model.add_constraint(
                f_var >= model.sumsq(f1_var) / (2.0 * problem.m)
            )
        elif str(problem.datafit) == "Squaredhinge":
            f1_var = model.continuous_var_list(
                problem.m, name="f1", lb=-np.inf
            )
            f2_var = model.continuous_var_list(problem.m, name="f2")
            model.add_constraints(
                f1_var[j]
                == 1.0
                - problem.datafit.y[j]
                - model.dot(x_var, problem.A[j, :])
                for j in range(problem.m)
            )
            model.add_constraints(
                f2_var[j] >= f1_var[j] for j in range(problem.m)
            )
            model.add_constraints(f2_var[j] >= 0.0 for j in range(problem.m))
            model.add_constraint(f_var >= model.sumsq(f2_var) / problem.m)
        else:
            raise NotImplementedError(
                "`CplexSolver` does not support `{}` yet.".format(
                    type(problem.datafit)
                )
            )

    def bind_model_g_var(
        self,
        problem: Problem,
        model: Model,
        x_var: Var,
        z_var: Var,
        g_var: Var,
    ) -> None:
        if str(problem.penalty) == "Bigm":
            model.add_constraints(
                x_var[i] <= problem.penalty.M * z_var[i]
                for i in range(problem.n)
            )
            model.add_constraints(
                x_var[i] >= -problem.penalty.M * z_var[i]
                for i in range(problem.n)
            )
            model.add_constraint(g_var >= problem.lmbd * sum(z_var))
        elif str(problem.penalty) == "BigmL1norm":
            g1_var = model.continuous_var_list(problem.n, name="g1")
            model.add_constraints(
                g1_var[i] >= x_var[i] for i in range(problem.n)
            )
            model.add_constraints(
                g1_var[i] >= -x_var[i] for i in range(problem.n)
            )
            model.add_constraints(
                x_var[i] <= problem.penalty.M * z_var[i]
                for i in range(problem.n)
            )
            model.add_constraints(
                x_var[i] >= -problem.penalty.M * z_var[i]
                for i in range(problem.n)
            )
            model.add_constraint(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(g1_var)
            )
        elif str(problem.penalty) == "BigmL2norm":
            g1_var = model.continuous_var_list(problem.n, name="g1")
            model.add_constraints(
                x_var[i] <= problem.penalty.M * z_var[i]
                for i in range(problem.n)
            )
            model.add_constraints(
                x_var[i] >= -problem.penalty.M * z_var[i]
                for i in range(problem.n)
            )
            model.add_quadratic_constraints(
                x_var[i] * x_var[i] <= g1_var[i] * z_var[i]
                for i in range(problem.n)
            )
            model.add_constraint(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(g1_var)
            )
        elif str(problem.penalty) == "L2norm":
            g1_var = model.continuous_var_list(problem.n, name="g1")
            model.add_quadratic_constraints(
                x_var[i] * x_var[i] <= g1_var[i] * z_var[i]
                for i in range(problem.n)
            )
            model.add_constraint(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(g1_var)
            )
        elif str(problem.penalty) == "L1L2norm":
            g1_var = model.continuous_var_list(problem.n, name="g1")
            g2_var = model.continuous_var_list(problem.n, name="g1")
            model.add_quadratic_constraints(
                z_var[i] * g1_var[i] >= x_var[i] for i in range(problem.n)
            )
            model.add_quadratic_constraints(
                z_var[i] * g1_var[i] >= -x_var[i] for i in range(problem.n)
            )
            model.add_quadratic_constraints(
                x_var[i] * x_var[i] <= g2_var[i] * z_var[i]
                for i in range(problem.n)
            )
            model.add_constraint(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(g1_var)
                + problem.penalty.beta * sum(g2_var)
            )
        else:
            raise NotImplementedError(
                "`CplexSolver` does not support `{}` yet.".format(
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

        model = Model()
        f_var = model.continuous_var(name="f", lb=-np.inf)
        g_var = model.continuous_var(name="g", lb=-np.inf)
        x_var = model.continuous_var_list(problem.n, name="x", lb=-np.inf)
        if relax:
            z_var = model.continuous_var_list(
                problem.n, name="z", lb=0.0, ub=1.0
            )
        else:
            z_var = model.binary_var_list(problem.n, name="z")
        self.bind_model_f_var(problem, model, x_var, f_var)
        self.bind_model_g_var(problem, model, x_var, z_var, g_var)
        model.minimize(f_var + g_var)

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
                self.model.add_constraint(self.x_var[i] == 0.0)
                self.model.add_constraint(self.z_var[i] == 0.0)
        if S1_init is not None:
            for i in S1_init:
                self.model.add_constraint(self.z_var[i] == 1.0)
        if x_init is not None:
            warmstart = self.model.new_solution()
            for i, xi in enumerate(x_init):
                warmstart.add_var_value(self.x_var[i], xi)
                if xi == 0.0:
                    warmstart.add_var_value(self.z_var[i], 0.0)
                else:
                    warmstart.add_var_value(self.z_var[i], 1.0)
            self.model.add_mip_start(warmstart)

    def set_options(self) -> None:
        self.model.parameters.mip.display = int(self.options["verbose"])
        self.model.parameters.timelimit = self.options["time_limit"]
        self.model.parameters.mip.tolerances.mipgap = self.options["rel_tol"]
        self.model.parameters.mip.tolerances.integrality = self.options[
            "int_tol"
        ]

    def get_status(self) -> Status:
        if self.model.get_solve_status().value == 2:
            status = Status.OPTIMAL
        elif self.model.solve_details.time > self.options["time_limit"]:
            status = Status.TIME_LIMIT
        else:
            status = Status.UNKNOWN
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
        self.model.solve()
        self.status = self.get_status()
        if self.status == Status.OPTIMAL:
            self.x = np.array([xi.solution_value for xi in self.x_var])
            self.z = np.array([zi.solution_value for zi in self.z_var])
        else:
            self.x = np.zeros(problem.n)
            self.z = np.zeros(problem.n)
        self.x *= self.z > self.options["int_tol"]

        return Results(
            self.status,
            self.model.solve_details.time,
            int(self.model.solve_details.nb_iterations),
            self.model.solve_details.mip_relative_gap,
            self.x,
            self.z,
            problem.value(self.x),
            np.sum(np.abs(self.x) > self.options["int_tol"]),
            None,
        )


class GurobiSolver(BaseSolver):
    """Gurobi solver for L0-penalized problems."""

    def __init__(
        self,
        time_limit: float = float(sys.maxsize),
        rel_tol: float = 1e-4,
        int_tol: float = 1e-8,
        verbose: bool = False,
    ):
        self.options = {
            "time_limit": time_limit,
            "rel_tol": rel_tol,
            "int_tol": int_tol,
            "verbose": verbose,
        }

    def __str__(self):
        return "GurobiSolver"

    def bind_model_f_var(
        self,
        problem: Problem,
        model: gp.Model,
        x_var: gp.MVar,
        f_var: gp.Var,
    ) -> None:
        if str(problem.datafit) == "Leastsquares":
            f1_var = model.addMVar(problem.m, vtype="C", name="f1", lb=-np.inf)
            model.addConstr(f1_var == problem.datafit.y - problem.A @ x_var)
            model.addConstr(f_var >= (f1_var @ f1_var) / (2.0 * problem.m))
        elif str(problem.datafit) == "Logistic":
            f1_var = model.addMVar(problem.m, vtype="C", name="f1", lb=-np.inf)
            f2_var = model.addMVar(problem.m, vtype="C", name="f2", lb=-np.inf)
            f3_var = model.addMVar(problem.m, vtype="C", name="f3", lb=0.0)
            f4_var = model.addMVar(problem.m, vtype="C", name="f4", lb=-np.inf)
            model.addConstr(f1_var >= -f2_var)
            model.addConstr(f4_var == problem.datafit.y * (problem.A @ x_var))
            for i in range(problem.m):
                model.addGenConstrLog(
                    f3_var[i],
                    f2_var[i],
                    "FuncPieces=-2 FuncPieceError=1e-8 FuncPieceRatio=0",
                )
                model.addGenConstrLogistic(
                    f4_var[i],
                    f3_var[i],
                    "FuncPieces=-2 FuncPieceError=1e-8 FuncPieceRatio=0",
                )
            model.addConstr(f_var >= gp.quicksum(f1_var) / problem.m)
        elif str(problem.datafit) == "Squaredhinge":
            f1_var = model.addMVar(problem.m, vtype="C", name="f1", lb=-np.inf)
            f2_var = model.addMVar(problem.m, vtype="C", name="f2", lb=-np.inf)
            f3_var = model.addMVar(problem.m, vtype="C", name="f3")
            model.addConstr(f1_var == problem.A @ x_var)
            model.addConstr(f2_var == 1.0 - problem.datafit.y * f1_var)
            model.addConstr(f3_var >= f2_var)
            model.addConstr(f3_var >= 0.0)
            model.addConstr(f_var >= (f3_var @ f3_var) / problem.m)
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
            model.addConstr(x_var <= problem.penalty.M * z_var)
            model.addConstr(x_var >= -problem.penalty.M * z_var)
            model.addConstr(g_var >= problem.lmbd * sum(z_var))
        elif str(problem.penalty) == "BigmL1norm":
            g1_var = model.addMVar(problem.n, vtype="C", name="g1")
            model.addConstr(g1_var >= x_var)
            model.addConstr(g1_var >= -x_var)
            model.addConstr(x_var <= problem.penalty.M * z_var)
            model.addConstr(x_var >= -problem.penalty.M * z_var)
            model.addConstr(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(g1_var)
            )
        elif str(problem.penalty) == "BigmL2norm":
            g1_var = model.addMVar(problem.n, vtype="C", name="g1")
            model.addConstr(g1_var >= 0.0)
            model.addConstr(x_var <= problem.penalty.M * z_var)
            model.addConstr(x_var >= -problem.penalty.M * z_var)
            model.addConstr(x_var * x_var <= g1_var * z_var)
            model.addConstr(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(g1_var)
            )
        elif str(problem.penalty) == "L2norm":
            g1_var = model.addMVar(problem.n, vtype="C", name="g1")
            model.addConstr(g1_var >= 0.0)
            model.addConstr(x_var * x_var <= g1_var * z_var)
            model.addConstr(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(g1_var)
            )
        elif str(problem.penalty) == "L1L2norm":
            g1_var = model.addMVar(problem.n, vtype="C", name="g1")
            g2_var = model.addMVar(problem.n, vtype="C", name="g2")
            model.addConstr(z_var * g1_var >= x_var)
            model.addConstr(z_var * g1_var >= -x_var)
            model.addConstr(g2_var >= 0.0)
            model.addConstr(x_var * x_var <= g2_var * z_var)
            model.addConstr(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(g1_var)
                + problem.penalty.beta * sum(g2_var)
            )
        elif str(problem.penalty) == "NeglogTriangular":
            g1_var = model.addMVar(problem.n, vtype="C", name="g1")
            g2_var = model.addMVar(problem.n, vtype="C", name="g2")
            g3_var = model.addMVar(problem.n, vtype="C", name="g3", lb=-np.inf)
            model.addConstr(z_var * g1_var >= x_var)
            model.addConstr(z_var * g1_var >= -x_var)
            model.addConstr(x_var <= problem.penalty.sigma * z_var)
            model.addConstr(x_var >= -problem.penalty.sigma * z_var)
            model.addConstr(g2_var == 1.0 - g1_var / problem.penalty.sigma)
            for i in range(problem.n):
                model.addGenConstrLog(g2_var[i], g3_var[i])
            model.addConstr(
                g_var
                >= problem.lmbd * sum(z_var)
                - problem.penalty.alpha * sum(g3_var)
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
        f_var = model.addVar(vtype="C", name="f", lb=-np.inf)
        g_var = model.addVar(vtype="C", name="g")
        x_var = model.addMVar(problem.n, vtype="C", name="x", lb=-np.inf)
        z_var = model.addMVar(problem.n, vtype="B", name="z")
        self.bind_model_f_var(problem, model, x_var, f_var)
        self.bind_model_g_var(problem, model, x_var, z_var, g_var)
        model.setObjective(f_var + g_var, gp.GRB.MINIMIZE)
        if relax:
            model.relax()
        model.update()

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
            for i, xi in enumerate(x_init):
                self.x_var[i].Start = xi

    def set_options(self) -> None:
        self.model.setParam("OutputFlag", self.options["verbose"])
        self.model.setParam("TimeLimit", self.options["time_limit"])
        self.model.setParam("MIPGap", self.options["rel_tol"])
        self.model.setParam("IntFeasTol", self.options["int_tol"])

    def get_status(self) -> Status:
        if self.model.Status == gp.GRB.OPTIMAL:
            status = Status.OPTIMAL
        elif self.model.Status == gp.GRB.TIME_LIMIT:
            status = Status.TIME_LIMIT
        else:
            status = Status.UNKNOWN
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
        self.status = self.get_status()
        if self.status == Status.OPTIMAL:
            self.x = np.array(self.x_var.X)
            self.z = np.array(self.z_var.X)
        else:
            self.x = np.zeros(problem.n)
            self.z = np.zeros(problem.n)
        self.x *= self.z > self.options["int_tol"]

        return Results(
            self.status,
            self.model.Runtime,
            int(self.model.NodeCount),
            self.model.MIPGap,
            self.x,
            self.z,
            problem.value(self.x),
            np.sum(np.abs(self.x) > self.options["int_tol"]),
            None,
        )


class MosekSolver(BaseSolver):
    """Mosek solver for L0-penalized problems."""

    def __init__(
        self,
        time_limit: float = float(sys.maxsize),
        rel_tol: float = 1e-4,
        int_tol: float = 1e-8,
        verbose: bool = False,
    ):
        self.options = {
            "time_limit": time_limit,
            "rel_tol": rel_tol,
            "int_tol": int_tol,
            "verbose": verbose,
        }

    def __str__(self):
        return "MosekSolver"

    def bind_model_f_var(
        self,
        problem: Problem,
        model: gp.Model,
        x_var: gp.MVar,
        f_var: gp.Var,
    ) -> None:
        if str(problem.datafit) == "Leastsquares":
            f1_var = model.variable("f1", problem.m, msk.Domain.unbounded())
            model.constraint(
                msk.Expr.hstack(
                    msk.Expr.constTerm(np.ones(problem.m)),
                    f1_var,
                    msk.Expr.sub(
                        problem.datafit.y, msk.Expr.mul(problem.A, x_var)
                    ),
                ),
                msk.Domain.inRotatedQCone(),
            )
            model.constraint(
                msk.Expr.sub(
                    f_var, msk.Expr.mul(1.0 / problem.m, msk.Expr.sum(f1_var))
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(problem.datafit) == "Logistic":
            f1_var = model.variable("f1", problem.m, msk.Domain.unbounded())
            f2_var = model.variable("f2", problem.m, msk.Domain.unbounded())
            f3_var = model.variable("f3", problem.m, msk.Domain.unbounded())
            model.constraint(
                msk.Expr.add(f2_var, f3_var),
                msk.Domain.lessThan(1.0),
            )
            model.constraint(
                msk.Expr.hstack(
                    f2_var,
                    msk.Expr.constTerm(np.ones(problem.m)),
                    msk.Expr.sub(
                        msk.Expr.mulElm(
                            problem.datafit.y, msk.Expr.mul(problem.A, x_var)
                        ),
                        f1_var,
                    ),
                ),
                msk.Domain.inPExpCone(),
            )
            model.constraint(
                msk.Expr.hstack(
                    f3_var,
                    msk.Expr.constTerm(np.ones(problem.m)),
                    msk.Expr.sub(0.0, f1_var),
                ),
                msk.Domain.inPExpCone(),
            )
            model.constraint(
                msk.Expr.sub(
                    f_var, msk.Expr.mul(1.0 / problem.m, msk.Expr.sum(f1_var))
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(problem.datafit) == "Squaredhinge":
            f1_var = model.variable("f1", problem.m, msk.Domain.unbounded())
            f2_var = model.variable("f2", problem.m, msk.Domain.unbounded())
            f3_var = model.variable("f3", problem.m, msk.Domain.unbounded())
            model.constraint(
                msk.Expr.sub(
                    f1_var,
                    msk.Expr.sub(
                        1.0,
                        msk.Expr.mulElm(
                            problem.datafit.y, msk.Expr.mul(problem.A, x_var)
                        ),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
            model.constraint(
                msk.Expr.sub(f2_var, f1_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(f2_var, msk.Domain.greaterThan(0.0))
            model.constraint(
                msk.Expr.hstack(
                    msk.Expr.constTerm(0.5 * np.ones(problem.m)),
                    f3_var,
                    f2_var,
                ),
                msk.Domain.inRotatedQCone(),
            )
            model.constraint(
                msk.Expr.sub(
                    f_var, msk.Expr.mul(1.0 / problem.m, msk.Expr.sum(f3_var))
                ),
                msk.Domain.greaterThan(0.0),
            )
        else:
            raise NotImplementedError(
                "`MosekSolver` does not support `{}` yet.".format(
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
            model.constraint(
                msk.Expr.sub(x_var, msk.Expr.mul(problem.penalty.M, z_var)),
                msk.Domain.lessThan(0.0),
            )
            model.constraint(
                msk.Expr.add(x_var, msk.Expr.mul(problem.penalty.M, z_var)),
                msk.Domain.greaterThan(0.0),
            )
            model.constraint(
                msk.Expr.sub(
                    g_var, msk.Expr.mul(problem.lmbd, msk.Expr.sum(z_var))
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(problem.penalty) == "BigmL1norm":
            g1_var = model.variable(
                "g1", problem.n, msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.sub(g1_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.add(g1_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.sub(x_var, msk.Expr.mul(problem.penalty.M, z_var)),
                msk.Domain.lessThan(0.0),
            )
            model.constraint(
                msk.Expr.add(x_var, msk.Expr.mul(problem.penalty.M, z_var)),
                msk.Domain.greaterThan(0.0),
            )
            model.constraint(
                msk.Expr.sub(
                    g_var,
                    msk.Expr.add(
                        msk.Expr.mul(problem.lmbd, msk.Expr.sum(z_var)),
                        msk.Expr.mul(
                            problem.penalty.alpha, msk.Expr.sum(g1_var)
                        ),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(problem.penalty) == "BigmL2norm":
            g1_var = model.variable(
                "g1", problem.n, msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.hstack(msk.Expr.mul(0.5, g1_var), z_var, x_var),
                msk.Domain.inRotatedQCone(),
            )
            model.constraint(
                msk.Expr.sub(x_var, msk.Expr.mul(problem.penalty.M, z_var)),
                msk.Domain.lessThan(0.0),
            )
            model.constraint(
                msk.Expr.add(x_var, msk.Expr.mul(problem.penalty.M, z_var)),
                msk.Domain.greaterThan(0.0),
            )
            model.constraint(
                msk.Expr.sub(
                    g_var,
                    msk.Expr.add(
                        msk.Expr.mul(problem.lmbd, msk.Expr.sum(z_var)),
                        msk.Expr.mul(
                            problem.penalty.alpha, msk.Expr.sum(g1_var)
                        ),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(problem.penalty) == "L2norm":
            g1_var = model.variable(
                "g1", problem.n, msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.hstack(msk.Expr.mul(0.5, g1_var), z_var, x_var),
                msk.Domain.inRotatedQCone(),
            )
            model.constraint(
                msk.Expr.sub(
                    g_var,
                    msk.Expr.add(
                        msk.Expr.mul(problem.lmbd, msk.Expr.sum(z_var)),
                        msk.Expr.mul(
                            problem.penalty.alpha, msk.Expr.sum(g1_var)
                        ),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(problem.penalty) == "L1L2norm":
            g1_var = model.variable(
                "g1", problem.n, msk.Domain.greaterThan(0.0)
            )
            g2_var = model.variable(
                "g2", problem.n, msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.sub(g1_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.add(g1_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.hstack(msk.Expr.mul(0.5, g2_var), z_var, x_var),
                msk.Domain.inRotatedQCone(),
            )
            model.constraint(
                msk.Expr.sub(
                    g_var,
                    msk.Expr.add(
                        msk.Expr.mul(problem.lmbd, msk.Expr.sum(z_var)),
                        msk.Expr.add(
                            msk.Expr.mul(
                                problem.penalty.alpha, msk.Expr.sum(g1_var)
                            ),
                            msk.Expr.mul(
                                problem.penalty.beta, msk.Expr.sum(g2_var)
                            ),
                        ),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(problem.penalty) == "NeglogTriangular":
            g1_var = model.variable("g1", problem.n, msk.Domain.unbounded())
            g2_var = model.variable("g2", problem.n, msk.Domain.unbounded())
            g3_var = model.variable("g3", problem.n, msk.Domain.unbounded())
            model.constraint(
                msk.Expr.sub(g1_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.add(g1_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.sub(
                    x_var, msk.Expr.mul(problem.penalty.sigma, z_var)
                ),
                msk.Domain.lessThan(0.0),
            )
            model.constraint(
                msk.Expr.add(
                    x_var, msk.Expr.mul(problem.penalty.sigma, z_var)
                ),
                msk.Domain.greaterThan(0.0),
            )
            model.constraint(
                msk.Expr.sub(
                    msk.Expr.constTerm(np.ones(problem.n)),
                    msk.Expr.add(
                        msk.Expr.mul(1.0 / problem.penalty.sigma, g1_var),
                        g2_var,
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
            model.constraint(
                msk.Expr.hstack(
                    g2_var,
                    msk.Expr.constTerm(np.ones(problem.n)),
                    g3_var,
                ),
                msk.Domain.inPExpCone(),
            )
            model.constraint(
                msk.Expr.add(
                    g_var,
                    msk.Expr.add(
                        msk.Expr.mul(
                            problem.penalty.alpha, msk.Expr.sum(g3_var)
                        ),
                        msk.Expr.mul(-problem.lmbd, msk.Expr.sum(z_var)),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        else:
            raise NotImplementedError(
                "`MosekSolver` does not support `{}` yet.".format(
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

        model = msk.Model()
        f_var = model.variable("f", 1, msk.Domain.unbounded())
        g_var = model.variable("g", 1, msk.Domain.greaterThan(0.0))
        x_var = model.variable("x", problem.n, msk.Domain.unbounded())
        if relax:
            z_var = model.variable(
                "z", problem.n, msk.Domain.inRange(0.0, 1.0)
            )
        else:
            z_var = model.variable("z", problem.n, msk.Domain.binary())
        self.bind_model_f_var(problem, model, x_var, f_var)
        self.bind_model_g_var(problem, model, x_var, z_var, g_var)
        model.objective(
            msk.ObjectiveSense.Minimize, msk.Expr.add([f_var, g_var])
        )

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
                self.model.constraint(self.x_var[i], msk.Domain.equalsTo(0.0))
                self.model.constraint(self.z_var[i], msk.Domain.equalsTo(0.0))
        if S1_init is not None:
            for i in S1_init:
                self.model.addConstr(self.z_var[i], msk.Domain.equalsTo(1.0))
        if x_init is not None:
            self.x_var.setLevel(x_init)

    def set_options(self) -> None:
        self.model.setSolverParam("mioMaxTime", self.options["time_limit"])
        self.model.setSolverParam("mioTolRelGap", self.options["rel_tol"])
        self.model.setSolverParam("mioTolAbsRelaxInt", self.options["int_tol"])
        self.model.setSolverParam("log", int(self.options["verbose"]))

    def get_status(self) -> Status:
        if self.model.getPrimalSolutionStatus() == msk.SolutionStatus.Optimal:
            status = Status.OPTIMAL
        elif (
            self.model.getSolverDoubleInfo("mioTime")
            >= self.options["time_limit"]
        ):
            status = Status.TIME_LIMIT
        else:
            status = Status.UNKNOWN
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
        self.model.solve()
        self.status = self.get_status()
        if self.status == Status.OPTIMAL:
            self.x = np.array(self.x_var.level())
            self.z = np.array(self.z_var.level())
        else:
            self.x = np.zeros(problem.n)
            self.z = np.zeros(problem.n)
        self.x *= self.z > self.options["int_tol"]

        return Results(
            self.status,
            self.model.getSolverDoubleInfo("mioTime"),
            self.model.getSolverIntInfo("mioNumBranch"),
            self.model.getSolverDoubleInfo("mioObjRelGap"),
            self.x,
            self.z,
            problem.value(self.x),
            np.sum(np.abs(self.x) > self.options["int_tol"]),
            None,
        )


class OsqpBoundingSolver(BoundingSolver):
    def __init__(self, eps_rel: float = 1e-4):
        self.eps_rel = eps_rel

    def setup(self, problem: Problem) -> None:
        if str(problem.datafit) != "Leastsquares":
            raise ValueError("Only a `Leastsquares` datafit is supported.")
        if str(problem.penalty) != "Bigm":
            raise ValueError("Only a `Bigm` penalty is supported.")

        m = problem.m
        n = problem.n
        A = problem.A
        lmbd = problem.lmbd
        y = problem.datafit.y
        M = problem.penalty.M
        On = sparse.eye(n)
        Om = sparse.eye(m)
        Zn = sparse.csc_matrix((n, n))
        on = np.ones(n)
        zm = np.zeros(m)
        zn = np.zeros(n)
        P = sparse.block_diag([Om, Zn, Zn], format="csc")
        q = np.hstack([zm, (m * lmbd / M) * on, zn])
        Q = sparse.bmat(
            [
                [-Om, None, A],
                [None, On, None],
                [None, On, -On],
                [None, On, On],
            ],
            format="csc",
        )
        lc = np.hstack([y, -M * on, zn, zn])
        uc = np.hstack([y, M * on, np.inf * on, np.inf * on])
        self.model = osqp.OSQP()
        self.model.setup(P, q, Q, lc, uc, verbose=False, eps_rel=self.eps_rel)

    def bound(
        self,
        problem: Problem,
        node: BnbNode,
        ub: float,
        rel_tol: float,
        workingsets: bool,
        dualpruning: bool,
        l1screening: bool,
        l0screening: bool,
        upper: bool = False,
    ):
        # Handle the root case and case where the upper-bounding problem yields
        # the same solutiona s the parent node.
        if upper:
            if not np.any(node.S1):
                node.x_inc = np.zeros(problem.n)
                node.upper_bound = problem.datafit.value(np.zeros(problem.m))
                return
            elif node.category == 0:
                return

        # Node data
        if upper:
            S1 = node.S1
            Sb = np.zeros(node.Sb.shape, dtype=np.bool_)
        else:
            S1 = node.S1
            Sb = node.Sb

        # Relaxation construction
        m = problem.m
        n = problem.n
        y = problem.datafit.y
        M = problem.penalty.M
        A = problem.A
        lmbd = problem.lmbd
        on = np.ones(n)
        zm = np.zeros(m)
        zn = np.zeros(n)
        q_new = np.hstack([zm, (m * lmbd / M) * Sb, zn])
        lc_new = np.hstack([y, -M * (Sb | S1), zn, zn])
        uc_new = np.hstack([y, M * (Sb | S1), np.inf * on, np.inf * on])
        self.model.update(q=q_new, l=lc_new, u=uc_new)
        result = self.model.solve()
        x = result.x[-n:]
        w = A[:, S1 | Sb] @ x[S1 | Sb]

        if upper:
            node.x_inc = np.copy(x) * S1
            node.upper_bound = problem.datafit.value(w) + lmbd * np.sum(S1)
        else:
            node.x = np.copy(x)
            node.w = np.copy(w)
            node.u = -problem.datafit.gradient(w)
            node.lower_bound = (
                -problem.datafit.conjugate(-node.u)
                - np.sum(
                    [
                        np.maximum(
                            problem.penalty.conjugate(A[:, i].T @ node.u)
                            - lmbd,
                            0.0,
                        )
                        for i in np.flatnonzero(Sb)
                    ]
                )
                - np.sum(
                    [
                        problem.penalty.conjugate(A[:, i].T @ node.u) - lmbd
                        for i in np.flatnonzero(S1)
                    ]
                )
            )


class SbnbSolver(BaseSolver):
    """Sbnb solver for L0-penalized problems."""

    def __init__(
        self,
        time_limit: float = float(sys.maxsize),
        rel_tol: float = 1e-4,
        int_tol: float = 1e-8,
        verbose: bool = False,
    ):
        self.options = {
            "time_limit": time_limit,
            "rel_tol": rel_tol,
            "int_tol": int_tol,
            "verbose": verbose,
        }
        self.solver = BnbSolver(
            verbose=verbose,
            time_limit=time_limit,
            rel_tol=rel_tol,
            int_tol=int_tol,
            bounding_solver=OsqpBoundingSolver(),
        )

    def __str__(self):
        return "SbnbSolver"

    def solve(
        self,
        problem: Problem,
        x_init: Union[NDArray, None] = None,
        S0_init: Union[NDArray, None] = None,
        S1_init: Union[NDArray, None] = None,
    ) -> Results:
        return self.solver.solve(problem, x_init, S0_init, S1_init)


class L0bnbSolver(BaseSolver):
    """L0bnb solver for L0-penalized problems."""

    def __init__(
        self,
        time_limit: float = float(sys.maxsize),
        rel_tol: float = 1e-4,
        int_tol: float = 1e-8,
        verbose: bool = False,
    ):
        self.options = {
            "time_limit": time_limit,
            "rel_tol": rel_tol,
            "int_tol": int_tol,
            "verbose": verbose,
        }

    def __str__(self):
        return "L0bnbSolver"

    def solve(
        self,
        problem: Problem,
        x_init: Union[NDArray, None] = None,
        S0_init: Union[NDArray, None] = None,
        S1_init: Union[NDArray, None] = None,
    ) -> Results:
        if S0_init is not None:
            raise ValueError("`S0_init` argument not supported yet.")
        if S1_init is not None:
            raise ValueError("`S1_init` argument not supported yet.")

        if str(problem.datafit) != "Leastsquares":
            raise NotImplementedError(
                "`L0bnbSolver` does not support `{}` yet.".format(
                    type(problem.datafit)
                )
            )

        if str(problem.penalty) == "Bigm":
            l0 = problem.m * problem.lmbd
            l2 = 0.0
            M = problem.penalty.M
        elif str(problem.penalty) == "L2norm":
            l0 = problem.m * problem.lmbd
            l2 = problem.m * problem.penalty.alpha
            M = np.inf
        elif str(problem.penalty) == "BigmL2norm":
            l0 = problem.m * problem.lmbd
            l2 = problem.m * problem.penalty.alpha
            M = problem.penalty.M
        else:
            raise NotImplementedError(
                "`L0bnbSolver` does not support `{}` yet.".format(
                    type(problem.penalty)
                )
            )

        solver = BNBTree(
            problem.A,
            problem.datafit.y,
            self.options["int_tol"],
            self.options["rel_tol"],
        )
        result = solver.solve(
            l0,
            l2,
            M,
            gap_tol=self.options["rel_tol"],
            warm_start=x_init,
            verbose=self.options["verbose"],
            time_limit=self.options["time_limit"],
        )

        if result.sol_time < self.options["time_limit"]:
            status = Status.OPTIMAL
        else:
            status = Status.TIME_LIMIT

        return Results(
            status,
            result.sol_time,
            solver.number_of_nodes,
            np.abs(result.gap),
            np.array(result.beta),
            np.array(result.beta != 0.0, dtype=float),
            problem.value(result.beta),
            np.sum(np.abs(result.beta) > self.options["int_tol"]),
            None,
        )


def extract_extra_options(solver_name):
    if solver_name.startswith("el0ps"):
        pattern = r"\[([^]]*)\]"
        match = re.search(pattern, solver_name)
        if match:
            options_str = match.group(1)
            if options_str:
                option_pairs = options_str.split(",")
                options_dict = {}
                for pair in option_pairs:
                    k, v = pair.split("=")
                    if k in [
                        "dualpruning",
                        "l1screening",
                        "l0screening",
                        "verbose",
                        "trace",
                    ]:
                        options_dict[k] = v in ["true", "True"]
                    elif k == "exploration_strategy":
                        options_dict[
                            "exploration_strategy"
                        ] = BnbExplorationStrategy[v]
                    elif k == "exploration_depth_switch":
                        options_dict["exploration_depth_switch"] = int(v)
                    elif k == "branching_strategy":
                        options_dict[
                            "branching_strategy"
                        ] = BnbBranchingStrategy[v]
                return options_dict
    return {}


def get_solver(solver_name, options={}):
    if solver_name.startswith("el0ps"):
        return BnbSolver(**{**options, **extract_extra_options(solver_name)})
    elif solver_name == "sbnb":
        return SbnbSolver(**options)
    elif solver_name == "l0bnb":
        return L0bnbSolver(**options)
    elif solver_name == "cplex":
        return CplexSolver(**options)
    elif solver_name == "gurobi":
        return GurobiSolver(**options)
    elif solver_name == "mosek":
        return MosekSolver(**options)
    else:
        raise ValueError("Unknown solver name {}".format(solver_name))


def can_handle(solver_name, datafit_name, penalty_name):
    if solver_name.startswith("el0ps"):
        handle_datafit = True
        handle_penalty = True
    elif solver_name == "sbnb":
        handle_datafit = datafit_name in ["Leastsquares"]
        handle_penalty = penalty_name in ["Bigm"]
    elif solver_name == "l0bnb":
        handle_datafit = datafit_name in ["Leastsquares"]
        handle_penalty = penalty_name in ["Bigm", "BigmL2norm", "L2norm"]
    elif solver_name == "cplex":
        handle_datafit = datafit_name in [
            "Leastsquares",
            "Squaredhinge",
        ]
        handle_penalty = penalty_name in [
            "Bigm",
            "BigmL1norm",
            "BigmL2norm",
            "L2norm",
            "L1L2norm",
        ]
    elif solver_name == "gurobi":
        handle_datafit = datafit_name in [
            "Leastsquares",
            "Logistic",
            "Squaredhinge",
        ]
        handle_penalty = penalty_name in [
            "Bigm",
            "BigmL1norm",
            "BigmL2norm",
            "L2norm",
            "L1L2norm",
            "NeglogTriangular",
        ]
    elif solver_name == "mosek":
        handle_datafit = datafit_name in [
            "Leastsquares",
            "Logistic",
            "Squaredhinge",
        ]
        handle_penalty = penalty_name in [
            "Bigm",
            "BigmL1norm",
            "BigmL2norm",
            "L2norm",
            "L1L2norm",
            "NeglogTriangular",
        ]
    else:
        raise ValueError("Unknown solver name {}".format(solver_name))
    return handle_datafit and handle_penalty
