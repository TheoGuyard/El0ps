import sys
import gurobipy as gp
import mosek.fusion as msk
import numpy as np
from typing import Union
from numpy.typing import NDArray
from l0bnb import BNBTree
from el0ps import Problem
from el0ps.solver import BaseSolver, BnbSolver, Status, Results


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
            "OutputFlag": int(verbose),
            "TimeLimit": time_limit,
            "MIPGap": rel_tol,
            "IntFeasTol": int_tol,
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
            r_var = model.addMVar(problem.m, vtype="C", name="r", lb=-np.inf)
            model.addConstr(r_var == problem.datafit.y - problem.A @ x_var)
            model.addConstr(f_var >= (r_var @ r_var) / (2.0 * problem.m))
        elif str(problem.datafit) == "Logistic":
            r_var = model.addMVar(problem.m, vtype="C", name="r", lb=-np.inf)
            l_var = model.addMVar(problem.m, vtype="C", name="l", lb=-np.inf)
            u_var = model.addMVar(problem.m, vtype="C", name="u", lb=0.0)
            v_var = model.addMVar(problem.m, vtype="C", name="v", lb=1.0)
            model.addConstr(r_var == -problem.datafit.y * (problem.A @ x_var))
            model.addConstr(v_var == 1.0 + u_var)
            for i in range(problem.m):
                model.addGenConstrExp(r_var[i], u_var[i])
                model.addGenConstrExp(l_var[i], v_var[i])
            model.addConstr(f_var >= sum(l_var) / problem.m)
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
            s_var = model.addMVar(problem.n, vtype="C", name="s")
            model.addConstr(s_var >= x_var)
            model.addConstr(s_var >= -x_var)
            model.addConstr(x_var <= problem.penalty.M * z_var)
            model.addConstr(x_var >= -problem.penalty.M * z_var)
            model.addConstr(z_var * s_var >= x_var)
            model.addConstr(z_var * s_var >= -x_var)
            model.addConstr(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(s_var)
            )
        elif str(problem.penalty) == "BigmL2norm":
            s_var = model.addMVar(problem.n, vtype="C", name="s")
            model.addConstr(s_var >= 0.0)
            model.addConstr(x_var <= problem.penalty.M * z_var)
            model.addConstr(x_var >= -problem.penalty.M * z_var)
            model.addConstr(x_var * x_var <= s_var * z_var)
            model.addConstr(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(s_var)
            )
        elif str(problem.penalty) == "L1norm":
            s_var = model.addMVar(problem.n, vtype="C", name="s")
            model.addConstr(z_var * s_var >= x_var)
            model.addConstr(z_var * s_var >= -x_var)
            model.addConstr(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(s_var)
            )
        elif str(problem.penalty) == "L2norm":
            s_var = model.addMVar(problem.n, vtype="C", name="s")
            model.addConstr(s_var >= 0.0)
            model.addConstr(x_var * x_var <= s_var * z_var)
            model.addConstr(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(s_var)
            )
        elif str(problem.penalty) == "L1L2norm":
            s1_var = model.addMVar(problem.n, vtype="C", name="s1")
            s2_var = model.addMVar(problem.n, vtype="C", name="s2")
            model.addConstr(z_var * s1_var >= x_var)
            model.addConstr(z_var * s1_var >= -x_var)
            model.addConstr(s2_var >= 0.0)
            model.addConstr(x_var * x_var <= s2_var * z_var)
            model.addConstr(
                g_var
                >= problem.lmbd * sum(z_var)
                + problem.penalty.alpha * sum(s1_var)
                + problem.penalty.beta * sum(s2_var)
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
            for i, xi in enumerate(x_init):
                self.x_var[i].Start = xi

    def set_options(self) -> None:
        for k, v in self.options.items():
            self.model.setParam(k, v)

    def get_status(self) -> Status:
        if self.model.Status == gp.GRB.OPTIMAL:
            status = Status.OPTIMAL
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
        self.status = self.get_status()
        if self.status == Status.OPTIMAL:
            self.x = np.array(self.x_var.X)
            self.z = np.array(self.z_var.X)
        else:
            self.x = np.zeros(problem.n)
            self.z = np.zeros(problem.n)

        return Results(
            self.status,
            self.model.Runtime,
            int(self.model.NodeCount),
            problem.value(self.x),
            self.model.MIPGap,
            self.x,
            self.z,
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
            "log": int(verbose),
            "mioMaxTime": time_limit,
            "mioTolRelGap": rel_tol,
            "mioTolAbsRelaxInt": int_tol,
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
            r_var = model.variable("r", problem.m, msk.Domain.unbounded())
            model.constraint(
                msk.Expr.hstack(
                    msk.Expr.constTerm(np.ones(problem.m)),
                    r_var,
                    msk.Expr.sub(
                        problem.datafit.y, msk.Expr.mul(problem.A, x_var)
                    ),
                ),
                msk.Domain.inRotatedQCone(),
            )
            model.constraint(
                msk.Expr.sub(
                    f_var, msk.Expr.mul(1.0 / problem.m, msk.Expr.sum(r_var))
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(problem.datafit) == "Logistic":
            r_var = msk.Expr.mulElm(
                -problem.datafit.y, msk.Expr.mul(problem.A, x_var)
            )
            l_var = model.variable("l", problem.m, msk.Domain.unbounded())
            u_var = model.variable("u", problem.n, msk.Domain.greaterThan(0.0))
            model.constraint(
                msk.Expr.hstack(
                    u_var,
                    msk.Expr.constTerm(np.ones(problem.n)),
                    r_var,
                ),
                msk.Domain.inPExpCone(),
            )
            model.constraint(
                msk.Expr.hstack(
                    msk.Expr.add(1.0, u_var),
                    msk.Expr.constTerm(np.ones(problem.n)),
                    r_var,
                ),
                msk.Domain.inPExpCone(),
            )
            model.constraint(
                msk.Expr.sub(
                    f_var, msk.Expr.mul(1.0 / problem.m, msk.Expr.sum(l_var))
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
            s_var = model.variable("s", problem.n, msk.Domain.unbounded())
            model.constraint(
                msk.Expr.sub(s_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.add(s_var, x_var), msk.Domain.greaterThan(0.0)
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
                            problem.penalty.alpha, msk.Expr.sum(s_var)
                        ),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(problem.penalty) == "BigmL2norm":
            s_var = model.variable("s", problem.n, msk.Domain.greaterThan(0.0))
            model.constraint(
                msk.Expr.hstack(msk.Expr.mul(0.5, s_var), z_var, x_var),
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
                            problem.penalty.alpha, msk.Expr.sum(s_var)
                        ),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(problem.penalty) == "L2norm":
            s_var = model.variable("s", problem.n, msk.Domain.greaterThan(0.0))
            model.constraint(
                msk.Expr.hstack(msk.Expr.mul(0.5, s_var), z_var, x_var),
                msk.Domain.inRotatedQCone(),
            )
            model.constraint(
                msk.Expr.sub(
                    g_var,
                    msk.Expr.add(
                        msk.Expr.mul(problem.lmbd, msk.Expr.sum(z_var)),
                        msk.Expr.mul(
                            problem.penalty.alpha, msk.Expr.sum(s_var)
                        ),
                    ),
                ),
                msk.Domain.greaterThan(0.0),
            )
        elif str(problem.penalty) == "L1L2norm":
            s1_var = model.variable(
                "s1", problem.n, msk.Domain.greaterThan(0.0)
            )
            s2_var = model.variable(
                "s2", problem.n, msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.sub(s1_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.add(s1_var, x_var), msk.Domain.greaterThan(0.0)
            )
            model.constraint(
                msk.Expr.hstack(msk.Expr.mul(0.5, s2_var), z_var, x_var),
                msk.Domain.inRotatedQCone(),
            )
            model.constraint(
                msk.Expr.sub(
                    g_var,
                    msk.Expr.add(
                        msk.Expr.mul(problem.lmbd, msk.Expr.sum(z_var)),
                        msk.Expr.add(
                            msk.Expr.mul(
                                problem.penalty.alpha, msk.Expr.sum(s1_var)
                            ),
                            msk.Expr.mul(
                                problem.penalty.beta, msk.Expr.sum(s2_var)
                            ),
                        ),
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
        g_var = model.variable("g", 1, msk.Domain.unbounded())
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
        for k, v in self.options.items():
            self.model.setSolverParam(k, v)

    def get_status(self) -> Status:
        if self.model.getPrimalSolutionStatus() == msk.SolutionStatus.Optimal:
            status = Status.OPTIMAL
        elif (
            self.model.getSolverDoubleInfo("mioTime")
            >= self.options["mioMaxTime"]
        ):
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
        self.model.solve()
        self.status = self.get_status()
        if self.status == Status.OPTIMAL:
            self.x = np.array(self.x_var.level())
            self.z = np.array(self.z_var.level())
        else:
            self.x = np.zeros(problem.n)
            self.z = np.zeros(problem.n)

        return Results(
            self.status,
            self.model.getSolverDoubleInfo("mioTime"),
            self.model.getSolverIntInfo("mioNumBranch"),
            self.model.getSolverDoubleInfo("mioObjInt"),
            self.model.getSolverDoubleInfo("mioObjRelGap"),
            self.x,
            self.z,
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
            "verbose": int(verbose),
            "time_limit": time_limit,
            "rel_tol": rel_tol,
            "int_tol": int_tol,
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
            problem.value(result.beta),
            result.gap,
            np.array(result.beta),
            np.array(result.beta != 0.0, dtype=float),
            None,
        )


def precompile(problem, solver):
    time_limit_precompile = 5.0
    if isinstance(solver, BnbSolver):
        solver.options.time_limit = time_limit_precompile
    elif isinstance(solver, L0bnbSolver):
        solver.options["time_limit"] = time_limit_precompile
    elif isinstance(solver, GurobiSolver):
        solver.options["TimeLimit"] = time_limit_precompile
    elif isinstance(solver, MosekSolver):
        solver.options["mioMaxTime"] = time_limit_precompile
    else:
        raise ValueError("Unknown solver {}".format(solver))
    solver.solve(problem)


def get_solver(solver_name, options={}):
    if solver_name == "el0ps":
        return BnbSolver(**options)
    elif solver_name == "l0bnb":
        return L0bnbSolver(**options)
    elif solver_name == "gurobi":
        return GurobiSolver(**options)
    elif solver_name == "mosek":
        return MosekSolver(**options)
    else:
        raise ValueError("Unknown solver name {}".format(solver_name))


def can_handle(solver_name, datafit_name, penalty_name):
    if solver_name == "el0ps":
        handle_datafit = datafit_name in ["Leastsquares", "Logistic"]
        handle_penalty = penalty_name in [
            "Bigm",
            "BigmL1norm",
            "BigmL2norm",
            "L1norm",
            "L2norm",
            "L1L2norm",
        ]
    elif solver_name == "l0bnb":
        handle_datafit = datafit_name in ["Leastsquares"]
        handle_penalty = penalty_name in ["Bigm", "BigmL2norm", "L2norm"]
    elif solver_name == "gurobi":
        handle_datafit = datafit_name in ["Leastsquares", "Logistic"]
        handle_penalty = penalty_name in [
            "Bigm",
            "BigmL1norm",
            "BigmL2norm",
            "L1norm",
            "L2norm",
            "L1L2norm",
        ]
    elif solver_name == "mosek":
        handle_datafit = datafit_name in ["Leastsquares", "Logistic"]
        handle_penalty = penalty_name in [
            "Bigm",
            "BigmL1norm",
            "BigmL2norm",
            "L2norm",
            "L1L2norm",
        ]
    else:
        raise ValueError("Unknown solver name {}".format(solver_name))
    return handle_datafit and handle_penalty
