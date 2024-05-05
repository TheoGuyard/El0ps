import numpy as np
from pyomo.opt.base.solvers import check_available_solvers
from el0ps.datafits import Leastsquares
from el0ps.penalties import Bigm
from el0ps.utils import compute_lmbd_max
from el0ps.solvers import Status, BnbSolver, MipSolver
from .utils import make_regression


def test_solver():
    A, y, x_true = make_regression(3, 30, 50)
    M = 1.5 * np.max(np.abs(x_true))

    datafit = Leastsquares(y)
    penalty = Bigm(M)
    lmbd = 0.1 * compute_lmbd_max(datafit, penalty, A)
    x_init = np.zeros(x_true.shape)

    solver = BnbSolver()
    result = solver.solve(datafit, penalty, A, lmbd, x_init=x_init)
    assert result.status == Status.OPTIMAL

    for optimizer_name in ["cplex", "gurobi", "mosek"]:
        if check_available_solvers(optimizer_name + "_direct"):
            solver = MipSolver(optimizer_name=optimizer_name)
            result = solver.solve(datafit, penalty, A, lmbd, x_init=x_init)
            assert result.status == Status.OPTIMAL
