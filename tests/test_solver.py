import numpy as np
import pytest
from pyomo.opt.base.solvers import check_available_solvers
from el0ps.datafit import Leastsquares
from el0ps.penalty import Bigm
from el0ps.utils import compute_lmbd_max
from el0ps.solver import Status, BnbSolver, MipSolver, OaSolver
from el0ps.solver.mip import _mip_optim_bindings
from .utils import make_regression


A, y, x_true = make_regression(3, 15, 20)
M = 1.5 * np.max(np.abs(x_true))
datafit = Leastsquares(y)
penalty = Bigm(M)
lmbd = 0.1 * compute_lmbd_max(datafit, penalty, A)
x_init = np.zeros(x_true.shape)

solvers = [
    pytest.param(solver, id=f"{solver.__class__.__name__}")
    for solver in [
        BnbSolver(),
        MipSolver(optimizer_name="cplex"),
        MipSolver(optimizer_name="gurobi"),
        MipSolver(optimizer_name="mosek"),
        OaSolver(optimizer_name="cplex"),
        OaSolver(optimizer_name="gurobi"),
        OaSolver(optimizer_name="mosek"),
    ]
]


@pytest.mark.parametrize("solver", solvers)
def test_solver(solver):

    if isinstance(solver, (MipSolver, OaSolver)):
        if not check_available_solvers(
            _mip_optim_bindings[solver.optimizer_name]["optimizer_name"]
        ):
            pytest.skip(f"{solver.optimizer_name} not available")

    result = solver.solve(datafit, penalty, A, lmbd, x_init=x_init)
    assert result.status == Status.OPTIMAL
