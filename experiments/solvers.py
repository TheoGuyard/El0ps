import re
import sys
import time
import numpy as np
from docplex.mp.model import Model
from docplex.mp.dvar import Var
import gurobipy as gp
import mosek.fusion as msk
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from typing import Union, get_type_hints
from numpy.typing import NDArray
from l0bnb import BNBTree
from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty
from el0ps.solver import (
    BaseSolver,
    BnbSolver,
    BnbOptions,
    Status,
    Result,
    BnbBranchingStrategy,
    BnbExplorationStrategy,
)


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
        datafit: BaseDatafit,
        A: NDArray,
        model: Model,
        x_var: Var,
        f_var: Var,
    ) -> None:
        m, _ = A.shape
        if str(datafit) == "Leastsquares":
            f1_var = model.continuous_var_list(m, name="f1", lb=-np.inf)
            model.add_constraints(
                f1_var[j] == datafit.y[j] - model.dot(x_var, A[j, :])
                for j in range(m)
            )
            model.add_constraint(f_var >= 0.5 * model.sumsq(f1_var))
        elif str(datafit) == "Squaredhinge":
            f1_var = model.continuous_var_list(m, name="f1", lb=-np.inf)
            f2_var = model.continuous_var_list(m, name="f2")
            model.add_constraints(
                f1_var[j] == 1.0 - datafit.y[j] - model.dot(x_var, A[j, :])
                for j in range(m)
            )
            model.add_constraints(f2_var[j] >= f1_var[j] for j in range(m))
            model.add_constraints(f2_var[j] >= 0.0 for j in range(m))
            model.add_constraint(f_var >= model.sumsq(f2_var))
        else:
            raise NotImplementedError(
                "`CplexSolver` does not support `{}` yet.".format(str(datafit))
            )

    def bind_model_g_var(
        self,
        penalty: BasePenalty,
        lmbd: NDArray,
        model: Model,
        x_var: Var,
        z_var: Var,
        g_var: Var,
    ) -> None:
        n = len(x_var)
        if str(penalty) == "Bigm":
            model.add_constraints(
                x_var[i] <= penalty.M * z_var[i] for i in range(n)
            )
            model.add_constraints(
                x_var[i] >= -penalty.M * z_var[i] for i in range(n)
            )
            model.add_constraint(g_var >= lmbd * sum(z_var))
        elif str(penalty) == "BigmL1norm":
            g1_var = model.continuous_var_list(n, name="g1")
            model.add_constraints(g1_var[i] >= x_var[i] for i in range(n))
            model.add_constraints(g1_var[i] >= -x_var[i] for i in range(n))
            model.add_constraints(
                x_var[i] <= penalty.M * z_var[i] for i in range(n)
            )
            model.add_constraints(
                x_var[i] >= -penalty.M * z_var[i] for i in range(n)
            )
            model.add_constraint(
                g_var >= lmbd * sum(z_var) + penalty.alpha * sum(g1_var)
            )
        elif str(penalty) == "BigmL2norm":
            g1_var = model.continuous_var_list(n, name="g1")
            model.add_constraints(
                x_var[i] <= penalty.M * z_var[i] for i in range(n)
            )
            model.add_constraints(
                x_var[i] >= -penalty.M * z_var[i] for i in range(n)
            )
            model.add_quadratic_constraints(
                x_var[i] * x_var[i] <= g1_var[i] * z_var[i] for i in range(n)
            )
            model.add_constraint(
                g_var >= lmbd * sum(z_var) + penalty.alpha * sum(g1_var)
            )
        elif str(penalty) == "L2norm":
            g1_var = model.continuous_var_list(n, name="g1")
            model.add_quadratic_constraints(
                x_var[i] * x_var[i] <= g1_var[i] * z_var[i] for i in range(n)
            )
            model.add_constraint(
                g_var >= lmbd * sum(z_var) + penalty.alpha * sum(g1_var)
            )
        elif str(penalty) == "L1L2norm":
            g1_var = model.continuous_var_list(n, name="g1")
            g2_var = model.continuous_var_list(n, name="g1")
            model.add_quadratic_constraints(
                z_var[i] * g1_var[i] >= x_var[i] for i in range(n)
            )
            model.add_quadratic_constraints(
                z_var[i] * g1_var[i] >= -x_var[i] for i in range(n)
            )
            model.add_quadratic_constraints(
                x_var[i] * x_var[i] <= g2_var[i] * z_var[i] for i in range(n)
            )
            model.add_constraint(
                g_var
                >= lmbd * sum(z_var)
                + penalty.alpha * sum(g1_var)
                + penalty.beta * sum(g2_var)
            )
        else:
            raise NotImplementedError(
                "`CplexSolver` does not support `{}` yet.".format(str(penalty))
            )

    def build_model(
        self,
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: NDArray,
        lmbd: float,
    ) -> None:

        _, n = A.shape
        model = Model()
        f_var = model.continuous_var(name="f", lb=-np.inf)
        g_var = model.continuous_var(name="g", lb=-np.inf)
        x_var = model.continuous_var_list(n, name="x", lb=-np.inf)
        z_var = model.binary_var_list(n, name="z")
        self.bind_model_f_var(datafit, A, model, x_var, f_var)
        self.bind_model_g_var(penalty, lmbd, model, x_var, z_var, g_var)
        model.minimize(f_var + g_var)

        self.model = model
        self.x_var = x_var
        self.z_var = z_var
        self.f_var = f_var
        self.g_var = g_var

    def set_init(self, x_init: Union[NDArray, None] = None) -> None:
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
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: NDArray,
        lmbd: float,
        x_init: Union[NDArray, None] = None,
    ) -> Result:
        self.build_model(datafit, penalty, A, lmbd)
        self.set_init(x_init)
        self.set_options()
        self.model.solve()
        self.status = self.get_status()
        if self.status == Status.OPTIMAL:
            self.x = np.array([xi.solution_value for xi in self.x_var])
            self.z = np.array([zi.solution_value for zi in self.z_var])
        else:
            self.x = np.zeros(A.shape[1])
            self.z = np.zeros(A.shape[1])
        self.x *= self.z > self.options["int_tol"]

        objective_value = (
            datafit.value(A @ self.x)
            + lmbd * np.linalg.norm(self.x, 0)
            + sum(penalty.value(i, xi) for i, xi in enumerate(self.x))
        )

        return Result(
            self.status,
            self.model.solve_details.time,
            int(self.model.solve_details.nb_iterations),
            self.model.solve_details.mip_relative_gap,
            self.x,
            self.z,
            objective_value,
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
        datafit: BaseDatafit,
        A: NDArray,
        model: gp.Model,
        x_var: gp.MVar,
        f_var: gp.Var,
    ) -> None:
        m, _ = A.shape
        if str(datafit) == "Leastsquares":
            f1_var = model.addMVar(m, vtype="C", name="f1", lb=-np.inf)
            model.addConstr(f1_var == datafit.y - A @ x_var)
            model.addConstr(f_var >= 0.5 * (f1_var @ f1_var))
        elif str(datafit) == "Squaredhinge":
            f1_var = model.addMVar(m, vtype="C", name="f1", lb=-np.inf)
            f2_var = model.addMVar(m, vtype="C", name="f2", lb=-np.inf)
            f3_var = model.addMVar(m, vtype="C", name="f3")
            model.addConstr(f1_var == A @ x_var)
            model.addConstr(f2_var == 1.0 - datafit.y * f1_var)
            model.addConstr(f3_var >= f2_var)
            model.addConstr(f3_var >= 0.0)
            model.addConstr(f_var >= (f3_var @ f3_var))
        else:
            raise NotImplementedError(
                "`GurobiSolver` does not support `{}` yet.".format(
                    str(datafit)
                )
            )

    def bind_model_g_var(
        self,
        penalty: BasePenalty,
        lmbd: float,
        model: gp.Model,
        x_var: gp.MVar,
        z_var: gp.MVar,
        g_var: gp.Var,
    ) -> None:
        n = x_var.size
        if str(penalty) == "Bigm":
            model.addConstr(x_var <= penalty.M * z_var)
            model.addConstr(x_var >= -penalty.M * z_var)
            model.addConstr(g_var >= lmbd * sum(z_var))
        elif str(penalty) == "BigmL1norm":
            g1_var = model.addMVar(n, vtype="C", name="g1")
            model.addConstr(g1_var >= x_var)
            model.addConstr(g1_var >= -x_var)
            model.addConstr(x_var <= penalty.M * z_var)
            model.addConstr(x_var >= -penalty.M * z_var)
            model.addConstr(
                g_var >= lmbd * sum(z_var) + penalty.alpha * sum(g1_var)
            )
        elif str(penalty) == "BigmL2norm":
            g1_var = model.addMVar(n, vtype="C", name="g1")
            model.addConstr(g1_var >= 0.0)
            model.addConstr(x_var <= penalty.M * z_var)
            model.addConstr(x_var >= -penalty.M * z_var)
            model.addConstr(x_var * x_var <= g1_var * z_var)
            model.addConstr(
                g_var >= lmbd * sum(z_var) + penalty.alpha * sum(g1_var)
            )
        elif str(penalty) == "L2norm":
            g1_var = model.addMVar(n, vtype="C", name="g1")
            model.addConstr(g1_var >= 0.0)
            model.addConstr(x_var * x_var <= g1_var * z_var)
            model.addConstr(
                g_var >= lmbd * sum(z_var) + penalty.alpha * sum(g1_var)
            )
        elif str(penalty) == "L1L2norm":
            g1_var = model.addMVar(n, vtype="C", name="g1")
            g2_var = model.addMVar(n, vtype="C", name="g2")
            model.addConstr(z_var * g1_var >= x_var)
            model.addConstr(z_var * g1_var >= -x_var)
            model.addConstr(g2_var >= 0.0)
            model.addConstr(x_var * x_var <= g2_var * z_var)
            model.addConstr(
                g_var
                >= lmbd * sum(z_var)
                + penalty.alpha * sum(g1_var)
                + penalty.beta * sum(g2_var)
            )
        elif str(penalty) == "NeglogTriangular":
            g1_var = model.addMVar(n, vtype="C", name="g1")
            g2_var = model.addMVar(n, vtype="C", name="g2")
            g3_var = model.addMVar(n, vtype="C", name="g3", lb=-np.inf)
            model.addConstr(z_var * g1_var >= x_var)
            model.addConstr(z_var * g1_var >= -x_var)
            model.addConstr(x_var <= penalty.sigma * z_var)
            model.addConstr(x_var >= -penalty.sigma * z_var)
            model.addConstr(g2_var == 1.0 - g1_var / penalty.sigma)
            for i in range(n):
                model.addGenConstrLog(g2_var[i], g3_var[i])
            model.addConstr(
                g_var >= lmbd * sum(z_var) - penalty.alpha * sum(g3_var)
            )
        else:
            raise NotImplementedError(
                "`GurobiSolver` does not support `{}` yet.".format(
                    str(penalty)
                )
            )

    def build_model(
        self,
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: NDArray,
        lmbd: float,
    ) -> None:
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
        _, n = A.shape
        model = gp.Model()
        f_var = model.addVar(vtype="C", name="f", lb=-np.inf)
        g_var = model.addVar(vtype="C", name="g")
        x_var = model.addMVar(n, vtype="C", name="x", lb=-np.inf)
        z_var = model.addMVar(n, vtype="B", name="z")
        self.bind_model_f_var(datafit, A, model, x_var, f_var)
        self.bind_model_g_var(penalty, lmbd, model, x_var, z_var, g_var)
        model.setObjective(f_var + g_var, gp.GRB.MINIMIZE)
        model.update()

        self.model = model
        self.x_var = x_var
        self.z_var = z_var
        self.f_var = f_var
        self.g_var = g_var

    def set_init(self, x_init: Union[NDArray, None] = None) -> None:
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
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: NDArray,
        lmbd: float,
        x_init: Union[NDArray, None] = None,
    ) -> Result:
        self.build_model(datafit, penalty, A, lmbd)
        self.set_init(x_init)
        self.set_options()
        self.model.optimize()
        self.status = self.get_status()
        if self.status == Status.OPTIMAL:
            self.x = np.array(self.x_var.X)
            self.z = np.array(self.z_var.X)
        else:
            self.x = np.zeros(A.shape[1])
            self.z = np.zeros(A.shape[1])
        self.x *= self.z > self.options["int_tol"]

        objective_value = (
            datafit.value(A @ self.x)
            + lmbd * np.linalg.norm(self.x, 0)
            + sum(penalty.value(i, xi) for i, xi in enumerate(self.x))
        )

        return Result(
            self.status,
            self.model.Runtime,
            int(self.model.NodeCount),
            self.model.MIPGap,
            self.x,
            self.z,
            objective_value,
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
        datafit: BaseDatafit,
        A: NDArray,
        model: msk.Model,
        x_var: msk.Variable,
        f_var: msk.Variable,
    ) -> None:
        m, _ = A.shape
        if str(datafit) == "Leastsquares":
            f1_var = model.variable("f1", m, msk.Domain.unbounded())
            model.constraint(
                msk.Expr.hstack(
                    msk.Expr.constTerm(np.ones(m)),
                    f1_var,
                    msk.Expr.sub(datafit.y, msk.Expr.mul(A, x_var)),
                ),
                msk.Domain.inRotatedQCone(),
            )
            model.constraint(
                msk.Expr.sub(f_var, msk.Expr.sum(f1_var)),
                msk.Domain.greaterThan(0.0),
            )
        elif str(datafit) == "Logistic":
            f1_var = model.variable("f1", m, msk.Domain.unbounded())
            f2_var = model.variable("f2", m, msk.Domain.unbounded())
            f3_var = model.variable("f3", m, msk.Domain.unbounded())
            model.constraint(
                msk.Expr.add(f2_var, f3_var),
                msk.Domain.lessThan(1.0),
            )
            model.constraint(
                msk.Expr.hstack(
                    f2_var,
                    msk.Expr.constTerm(np.ones(m)),
                    msk.Expr.sub(
                        msk.Expr.mulElm(datafit.y, msk.Expr.mul(A, x_var)),
                        f1_var,
                    ),
                ),
                msk.Domain.inPExpCone(),
            )
            model.constraint(
                msk.Expr.hstack(
                    f3_var,
                    msk.Expr.constTerm(np.ones(m)),
                    msk.Expr.sub(0.0, f1_var),
                ),
                msk.Domain.inPExpCone(),
            )
            model.constraint(
                msk.Expr.sub(f_var, msk.Expr.sum(f1_var)),
                msk.Domain.greaterThan(0.0),
            )
        elif str(datafit) == "Squaredhinge":
            f1_var = model.variable("f1", m, msk.Domain.unbounded())
            f2_var = model.variable("f2", m, msk.Domain.unbounded())
            f3_var = model.variable("f3", m, msk.Domain.unbounded())
            model.constraint(
                msk.Expr.sub(
                    f1_var,
                    msk.Expr.sub(
                        1.0,
                        msk.Expr.mulElm(datafit.y, msk.Expr.mul(A, x_var)),
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
                    msk.Expr.constTerm(0.5 * np.ones(m)),
                    f3_var,
                    f2_var,
                ),
                msk.Domain.inRotatedQCone(),
            )
            model.constraint(
                msk.Expr.sub(f_var, msk.Expr.sum(f3_var)),
                msk.Domain.greaterThan(0.0),
            )
        else:
            raise NotImplementedError(
                "`MosekSolver` does not support `{}` yet.".format(str(datafit))
            )

    def bind_model_g_var(
        self,
        penalty: BasePenalty,
        lmbd: float,
        model: msk.Model,
        x_var: msk.Variable,
        z_var: msk.Variable,
        g_var: msk.Variable,
    ) -> None:
        n = x_var.getSize()
        if str(penalty) == "Bigm":
            model.constraint(
                msk.Expr.sub(x_var, msk.Expr.mul(penalty.M, z_var)),
                msk.Domain.lessThan(0.0),
            )
            model.constraint(
                msk.Expr.add(x_var, msk.Expr.mul(penalty.M, z_var)),
                msk.Domain.greaterThan(0.0),
            )
            model.constraint(
                msk.Expr.sub(g_var, msk.Expr.mul(lmbd, msk.Expr.sum(z_var))),
                msk.Domain.greaterThan(0.0),
            )
        elif str(penalty) == "BigmL1norm":
            g1_var = model.variable("g1", n, msk.Domain.greaterThan(0.0))
            model.constraint(
                msk.Expr.sub(g1_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.add(g1_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.sub(x_var, msk.Expr.mul(penalty.M, z_var)),
                msk.Domain.lessThan(0.0),
            )
            model.constraint(
                msk.Expr.add(x_var, msk.Expr.mul(penalty.M, z_var)),
                msk.Domain.greaterThan(0.0),
            )
            model.constraint(
                msk.Expr.sub(
                    g_var,
                    msk.Expr.add(
                        msk.Expr.mul(lmbd, msk.Expr.sum(z_var)),
                        msk.Expr.mul(penalty.alpha, msk.Expr.sum(g1_var)),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(penalty) == "BigmL2norm":
            g1_var = model.variable("g1", n, msk.Domain.greaterThan(0.0))
            model.constraint(
                msk.Expr.hstack(msk.Expr.mul(0.5, g1_var), z_var, x_var),
                msk.Domain.inRotatedQCone(),
            )
            model.constraint(
                msk.Expr.sub(x_var, msk.Expr.mul(penalty.M, z_var)),
                msk.Domain.lessThan(0.0),
            )
            model.constraint(
                msk.Expr.add(x_var, msk.Expr.mul(penalty.M, z_var)),
                msk.Domain.greaterThan(0.0),
            )
            model.constraint(
                msk.Expr.sub(
                    g_var,
                    msk.Expr.add(
                        msk.Expr.mul(lmbd, msk.Expr.sum(z_var)),
                        msk.Expr.mul(penalty.alpha, msk.Expr.sum(g1_var)),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(penalty) == "L2norm":
            g1_var = model.variable("g1", n, msk.Domain.greaterThan(0.0))
            model.constraint(
                msk.Expr.hstack(msk.Expr.mul(0.5, g1_var), z_var, x_var),
                msk.Domain.inRotatedQCone(),
            )
            model.constraint(
                msk.Expr.sub(
                    g_var,
                    msk.Expr.add(
                        msk.Expr.mul(lmbd, msk.Expr.sum(z_var)),
                        msk.Expr.mul(penalty.alpha, msk.Expr.sum(g1_var)),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(penalty) == "L1L2norm":
            g1_var = model.variable("g1", n, msk.Domain.greaterThan(0.0))
            g2_var = model.variable("g2", n, msk.Domain.greaterThan(0.0))
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
                        msk.Expr.mul(lmbd, msk.Expr.sum(z_var)),
                        msk.Expr.add(
                            msk.Expr.mul(penalty.alpha, msk.Expr.sum(g1_var)),
                            msk.Expr.mul(penalty.beta, msk.Expr.sum(g2_var)),
                        ),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(penalty) == "NeglogTriangular":
            g1_var = model.variable("g1", n, msk.Domain.unbounded())
            g2_var = model.variable("g2", n, msk.Domain.unbounded())
            g3_var = model.variable("g3", n, msk.Domain.unbounded())
            model.constraint(
                msk.Expr.sub(g1_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.add(g1_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.sub(x_var, msk.Expr.mul(penalty.sigma, z_var)),
                msk.Domain.lessThan(0.0),
            )
            model.constraint(
                msk.Expr.add(x_var, msk.Expr.mul(penalty.sigma, z_var)),
                msk.Domain.greaterThan(0.0),
            )
            model.constraint(
                msk.Expr.sub(
                    msk.Expr.constTerm(np.ones(n)),
                    msk.Expr.add(
                        msk.Expr.mul(1.0 / penalty.sigma, g1_var),
                        g2_var,
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
            model.constraint(
                msk.Expr.hstack(
                    g2_var,
                    msk.Expr.constTerm(np.ones(n)),
                    g3_var,
                ),
                msk.Domain.inPExpCone(),
            )
            model.constraint(
                msk.Expr.add(
                    g_var,
                    msk.Expr.add(
                        msk.Expr.mul(penalty.alpha, msk.Expr.sum(g3_var)),
                        msk.Expr.mul(-lmbd, msk.Expr.sum(z_var)),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        else:
            raise NotImplementedError(
                "`MosekSolver` does not support `{}` yet.".format(str(penalty))
            )

    def build_model(
        self,
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: NDArray,
        lmbd: float,
    ) -> None:
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
        _, n = A.shape
        model = msk.Model()
        f_var = model.variable("f", 1, msk.Domain.unbounded())
        g_var = model.variable("g", 1, msk.Domain.greaterThan(0.0))
        x_var = model.variable("x", n, msk.Domain.unbounded())
        z_var = model.variable("z", n, msk.Domain.binary())
        self.bind_model_f_var(datafit, A, model, x_var, f_var)
        self.bind_model_g_var(penalty, lmbd, model, x_var, z_var, g_var)
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
    ) -> None:
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
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: NDArray,
        lmbd: float,
        x_init: Union[NDArray, None] = None,
    ) -> Result:
        self.build_model(datafit, penalty, A, lmbd)
        self.set_init(x_init)
        self.set_options()
        self.model.solve()
        self.status = self.get_status()
        if self.status == Status.OPTIMAL:
            self.x = np.array(self.x_var.level())
            self.z = np.array(self.z_var.level())
        else:
            self.x = np.zeros(A.shape[1])
            self.z = np.zeros(A.shape[1])
        self.x *= self.z > self.options["int_tol"]

        objective_value = (
            datafit.value(A @ self.x)
            + lmbd * np.linalg.norm(self.x, 0)
            + sum(penalty.value(i, xi) for i, xi in enumerate(self.x))
        )

        return Result(
            self.status,
            self.model.getSolverDoubleInfo("mioTime"),
            self.model.getSolverIntInfo("mioNumBranch"),
            self.model.getSolverDoubleInfo("mioObjRelGap"),
            self.x,
            self.z,
            objective_value,
            np.sum(np.abs(self.x) > self.options["int_tol"]),
            None,
        )


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
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: NDArray,
        lmbd: float,
        x_init: Union[NDArray, None] = None,
    ) -> Result:
        if str(datafit) != "Leastsquares":
            raise NotImplementedError(
                "`L0bnbSolver` does not support `{}` yet.".format(str(datafit))
            )

        m, _ = A.shape
        if str(penalty) == "Bigm":
            l0 = lmbd
            l2 = 0.0
            M = penalty.M
        elif str(penalty) == "L2norm":
            l0 = lmbd
            l2 = penalty.alpha
            M = np.inf
        elif str(penalty) == "BigmL2norm":
            l0 = lmbd
            l2 = penalty.alpha
            M = penalty.M
        else:
            raise NotImplementedError(
                "`L0bnbSolver` does not support `{}` yet.".format(str(penalty))
            )

        solver = BNBTree(
            A,
            datafit.y,
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

        self.x = np.array(result.beta)
        self.z = np.array(result.beta != 0.0, dtype=float)

        objective_value = (
            datafit.value(A @ self.x)
            + lmbd * np.linalg.norm(self.x, 0)
            + sum(penalty.value(i, xi) for i, xi in enumerate(self.x))
        )

        return Result(
            status,
            result.sol_time,
            solver.number_of_nodes,
            np.abs(result.gap) if not np.isnan(result.gap) else 0.0,
            self.x,
            self.z,
            objective_value,
            np.sum(np.abs(self.x) > self.options["int_tol"]),
            None,
        )


class OmpPath:

    def __init__(self, max_nnz=10) -> None:
        self.max_nnz = max_nnz

    def __str__(self) -> str:
        return "OmpPath"

    def fit(self, datafit, A):
        assert str(datafit) == "Leastsquares"

        fit_data = {
            "status": [],
            "solve_time": [],
            "x": [],
            "datafit_value": [],
            "n_nnz": [],
        }

        start_time = time.time()

        n = A.shape[1]
        y = datafit.y
        s = []
        r = y
        for _ in range(self.max_nnz):
            u = np.dot(A.T, r)
            i = np.argmax(np.abs(u))
            s.append(i)
            x = np.zeros(n)
            x[s] = np.linalg.lstsq(A[:, s], y, rcond=None)[0]
            w = A[:, s] @ x[s]
            r = y - w

            fit_data["status"].append(Status.OPTIMAL)
            fit_data["solve_time"].append(time.time() - start_time)
            fit_data["x"].append(x)
            fit_data["datafit_value"].append(datafit.value(w))
            fit_data["n_nnz"].append(len(s))

        return fit_data


class LassoPath:

    def __init__(
        self,
        lmbd_ratio_max=1.0,
        lmbd_ratio_min=1e-3,
        lmbd_ratio_num=31,
        max_nnz=10,
        stop_if_not_optimal=True,
    ) -> None:
        self.lmbd_ratio_max = lmbd_ratio_max
        self.lmbd_ratio_min = lmbd_ratio_min
        self.lmbd_ratio_num = lmbd_ratio_num
        self.max_nnz = max_nnz
        self.stop_if_not_optimal = stop_if_not_optimal

    def __str__(self) -> str:
        return "LassoPath"

    def fit(self, datafit, A):
        assert str(datafit) == "Leastsquares"

        fit_data = {
            "status": [],
            "solve_time": [],
            "x": [],
            "datafit_value": [],
            "n_nnz": [],
        }

        m = A.shape[0]
        y = datafit.y

        lmbd_ratio_grid = np.logspace(
            np.log10(self.lmbd_ratio_max),
            np.log10(self.lmbd_ratio_min),
            self.lmbd_ratio_num,
        )
        lmbd_max = np.linalg.norm(A.T @ y, np.inf)

        start_time = time.time()

        for lmbd_ratio in lmbd_ratio_grid:
            lmbd = lmbd_ratio * lmbd_max
            lasso = Lasso(alpha=lmbd, max_iter=int(1e5), fit_intercept=False)
            lasso.fit(A, y)
            x = lasso.coef_
            w = A @ x
            s = np.where(x != 0)[0]

            if len(s) > self.max_nnz:
                break

            fit_data["status"].append(Status.OPTIMAL)
            fit_data["solve_time"].append(time.time() - start_time)
            fit_data["x"].append(np.copy(x))
            fit_data["datafit_value"].append(datafit.value(w))
            fit_data["n_nnz"].append(len(s))

        return fit_data


class EnetPath:

    def __init__(
        self,
        lmbd_ratio_max=1.0,
        lmbd_ratio_min=1e-3,
        lmbd_ratio_num=31,
        max_nnz=10,
        stop_if_not_optimal=True,
    ) -> None:
        self.lmbd_ratio_max = lmbd_ratio_max
        self.lmbd_ratio_min = lmbd_ratio_min
        self.lmbd_ratio_num = lmbd_ratio_num
        self.max_nnz = max_nnz
        self.stop_if_not_optimal = stop_if_not_optimal

    def __str__(self) -> str:
        return "EnetPath"

    def fit(self, datafit, A):
        assert str(datafit) == "Leastsquares"

        fit_data = {
            "status": [],
            "solve_time": [],
            "x": [],
            "datafit_value": [],
            "n_nnz": [],
        }

        m = A.shape[0]
        y = datafit.y

        lmbd_ratio_grid = np.logspace(
            np.log10(self.lmbd_ratio_max),
            np.log10(self.lmbd_ratio_min),
            self.lmbd_ratio_num,
        )
        lmbd_max = np.linalg.norm(A.T @ y, np.inf)

        # Calibrate L1 ratio
        param_grid = {
            "alpha": lmbd_max * np.logspace(-2, 1, 4),
            "l1_ratio": np.linspace(0.1, 0.9, 9),
        }
        grid_search = GridSearchCV(
            estimator=ElasticNet(), param_grid=param_grid, cv=5
        )
        grid_search.fit(A, y)
        l1_ratio = grid_search.best_estimator_.l1_ratio

        start_time = time.time()

        for lmbd_ratio in lmbd_ratio_grid:
            lmbd = lmbd_ratio * lmbd_max
            lasso = ElasticNet(
                alpha=lmbd,
                l1_ratio=l1_ratio,
                max_iter=int(1e5),
                fit_intercept=False,
            )
            lasso.fit(A, y)
            x = lasso.coef_
            w = A @ x
            s = np.where(x != 0)[0]

            if len(s) > self.max_nnz:
                break

            fit_data["status"].append(Status.OPTIMAL)
            fit_data["solve_time"].append(time.time() - start_time)
            fit_data["x"].append(np.copy(x))
            fit_data["datafit_value"].append(datafit.value(w))
            fit_data["n_nnz"].append(len(s))

        return fit_data


def extract_extra_options(solver_name):
    if solver_name.startswith("el0ps"):
        option_types = get_type_hints(BnbOptions)
        pattern = r"\[([^]]*)\]"
        match = re.search(pattern, solver_name)
        if match:
            options_str = match.group(1)
            if options_str:
                option_pairs = options_str.split(",")
                options_dict = {}
                for pair in option_pairs:
                    k, v = pair.split("=")
                    if k == "exploration_strategy":
                        options_dict[k] = BnbExplorationStrategy(v)
                    elif k == "branching_strategy":
                        options_dict[k] = BnbBranchingStrategy(v)
                    elif option_types[k] in [str, int, float]:
                        options_dict[k] = option_types[k](v)
                    elif option_types[k] == bool:
                        options_dict[k] = v in ["true", "True"]
                return options_dict
    return {}


def get_solver(solver_name, options={}):
    if solver_name.startswith("el0ps"):
        return BnbSolver(**{**options, **extract_extra_options(solver_name)})
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


def get_relaxed_path(solver_name, path_opts={}):
    if solver_name == "Omp":
        return OmpPath(max_nnz=path_opts["max_nnz"])
    elif solver_name == "Lasso":
        return LassoPath(
            lmbd_ratio_max=path_opts["lmbd_ratio_max"],
            lmbd_ratio_min=path_opts["lmbd_ratio_min"],
            lmbd_ratio_num=path_opts["lmbd_ratio_num"],
            max_nnz=path_opts["max_nnz"],
            stop_if_not_optimal=path_opts["stop_if_not_optimal"],
        )
    elif solver_name == "Enet":
        return EnetPath(
            lmbd_ratio_max=path_opts["lmbd_ratio_max"],
            lmbd_ratio_min=path_opts["lmbd_ratio_min"],
            lmbd_ratio_num=path_opts["lmbd_ratio_num"],
            max_nnz=path_opts["max_nnz"],
            stop_if_not_optimal=path_opts["stop_if_not_optimal"],
        )
    else:
        raise ValueError("Unknown solver name: {}".format(solver_name))


def can_handle_instance(solver_name, datafit_name, penalty_name):
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


def can_handle_compilation(solver_name):
    return solver_name.startswith("el0ps")
