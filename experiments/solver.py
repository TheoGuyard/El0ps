import sys
import numpy as np
from typing import Union
from numpy.typing import ArrayLike
from el0ps.datafit import Leastsquares
from el0ps.penalty import Bigm, L2norm, BigmL2norm
from el0ps.solver import (
    BaseSolver,
    Status,
    Result,
    BnbSolver,
    MipSolver,
    OaSolver,
)
from l0bnb import BNBTree
from numba.experimental.jitclass.base import JitClassType


class L0bnbSolver(BaseSolver):

    def __init__(
        self,
        integrality_tol: float = 0.0,
        relative_gap: float = 1e-8,
        absolute_gap: float = 0.0,
        time_limit: float = float(sys.maxsize),
        verbose: bool = False,
    ):
        self.integrality_tol = integrality_tol
        self.relative_gap = relative_gap
        self.absolute_gap = absolute_gap
        self.time_limit = time_limit
        self.verbose = verbose

    def __str__(self):
        return "L0bnbSolver"

    def solve(
        self,
        datafit: Leastsquares,
        penalty: Union[Bigm, L2norm, BigmL2norm],
        A: ArrayLike,
        lmbd: float,
        x_init: Union[ArrayLike, None] = None,
    ) -> Result:

        assert isinstance(datafit, Leastsquares)
        assert (
            isinstance(penalty, Bigm)
            or isinstance(penalty, L2norm)
            or isinstance(penalty, BigmL2norm)
        )

        if isinstance(penalty, Bigm):
            l0 = lmbd
            l2 = 0.0
            M = penalty.M
        elif isinstance(penalty, L2norm):
            l0 = lmbd
            l2 = penalty.beta
            M = sys.maxsize
        elif isinstance(penalty, BigmL2norm):
            l0 = lmbd
            l2 = penalty.beta
            M = penalty.M

        solver = BNBTree(
            A,
            datafit.y,
            self.integrality_tol,
            self.relative_gap,
        )

        result = solver.solve(
            l0,
            l2,
            M,
            gap_tol=self.relative_gap,
            warm_start=x_init,
            verbose=self.verbose,
            time_limit=self.time_limit,
        )

        if result.sol_time < self.time_limit:
            status = Status.OPTIMAL
        else:
            status = Status.TIME_LIMIT

        solution = np.array(result.beta)

        objective_value = (
            datafit.value(A @ solution)
            + lmbd * np.linalg.norm(solution, ord=0)
            + penalty.value(solution)
        )

        return Result(
            status,
            result.sol_time,
            solver.number_of_nodes,
            solution,
            objective_value,
            None,
        )


def get_solver(solver_name: str, solver_opts: dict) -> BaseSolver:
    if solver_name == "el0ps":
        return BnbSolver(**solver_opts)
    elif solver_name == "mip":
        return MipSolver(**solver_opts)
    elif solver_name == "oa":
        return OaSolver(**solver_opts)
    elif solver_name == "l0bnb":
        return L0bnbSolver(**solver_opts)
    else:
        raise ValueError(f"Unknown solver {solver_name}.")


def can_handle_instance(
    solver_name: str,
    solver_opts: dict,
    datafit_name: str,
    penalty_name: str,
) -> bool:
    if solver_name == "el0ps":
        return True
    elif solver_name == "mip":
        optim_name = solver_opts["optimizer_name"]
        if optim_name == "cplex":
            return datafit_name in [
                "Leastsquares",
                "Squaredhinge",
            ] and penalty_name in [
                "Bigm",
                "BigmL1norm",
                "BigmL2norm",
                "Bounds",
                "L2norm",
                "L1L2norm",
                "PositiveL2norm",
            ]
        elif optim_name == "gurobi":
            return datafit_name in [
                "Leastsquares",
                "Squaredhinge",
            ] and penalty_name in [
                "Bigm",
                "BigmL1norm",
                "BigmL2norm",
                "Bounds",
                "L2norm",
                "L1L2norm",
                "PositiveL2norm",
            ]
        elif optim_name == "mosek":
            return datafit_name in [
                "Leastsquares",
                "Logistic",
                "Squaredhinge",
            ] and penalty_name in [
                "Bigm",
                "BigmL1norm",
                "BigmL2norm",
                "Bounds",
                "L2norm",
                "L1L2norm",
                "PositiveL2norm",
            ]
        else:
            raise ValueError(f"Unknown optimizer {optim_name}.")
    elif solver_name == "oa":
        return True
    elif solver_name == "l0bnb":
        return datafit_name in ["Leastsquares"] and penalty_name in [
            "Bigm",
            "BigmL2norm",
            "L2norm",
        ]
    else:
        raise ValueError(f"Unknown solver {solver_name}.")


def can_handle_compilation(solver_name: str) -> bool:
    return solver_name in ["el0ps", "oa"]


def precompile_solver(
    solver: BaseSolver,
    datafit: JitClassType,
    penalty: JitClassType,
    A: ArrayLike,
    lmbd: float,
    precompile_time: float = 5.0,
) -> None:

    time_limit = solver.time_limit
    solver.time_limit = precompile_time
    solver.solve(datafit, penalty, A, lmbd)
    solver.time_limit = time_limit
