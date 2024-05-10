import numpy as np

from el0ps.datafits import Leastsquares
from el0ps.penalties import Bigm
from el0ps.solvers import BnbSolver, Status
from el0ps.path import Path
from .utils import make_regression


def test_path():
    A, y, x_true = make_regression(3, 150, 200)
    M = 1.5 * np.max(np.abs(x_true))

    datafit = Leastsquares(y)
    penalty = Bigm(M)
    solver = BnbSolver()

    path = Path(lmbd_max=1e-0, lmbd_min=1e-1, lmbd_num=10, lmbd_scaled=True)
    fit = path.fit(solver, datafit, penalty, A)
    assert np.all(status == Status.OPTIMAL for status in fit["status"])
