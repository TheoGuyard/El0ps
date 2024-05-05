import numpy as np
from pyomo.opt.base.solvers import check_available_solvers
from el0ps.datafits import Leastsquares
from el0ps.penalties import Bigm
from el0ps.utils import compute_lmbd_max
from el0ps.solvers import Status, BnbSolver, MipSolver


def test_solver():
    k, m, n = 3, 30, 50
    x = np.zeros(n)
    s = np.array(np.floor(np.linspace(0, n - 1, num=k)), dtype=int)
    x[s] = np.sign(np.random.randn(k))
    A = np.random.randn(m, n)
    y = A @ x
    e = np.random.randn(m)
    e *= np.sqrt((y @ y) / (10.0 * (e @ e)))
    y += e
    M = 1.5 * np.max(np.abs(x))

    datafit = Leastsquares(y)
    penalty = Bigm(M)
    lmbd = 0.1 * compute_lmbd_max(datafit, penalty, A)

    solver = BnbSolver()
    result = solver.solve(datafit, penalty, A, lmbd, x_init=x)
    assert result.status == Status.OPTIMAL

    for optimizer_name in ["cplex", "gurobi", "mosek"]:
        if check_available_solvers(optimizer_name + "_direct"):
            solver = MipSolver(optimizer_name=optimizer_name)
            result = solver.solve(datafit, penalty, A, lmbd, x_init=x)
            assert result.status == Status.OPTIMAL
