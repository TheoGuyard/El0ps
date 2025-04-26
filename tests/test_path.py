import numpy as np

from el0ps.datafit import Leastsquares
from el0ps.penalty import Bigm
from el0ps.solver import BnbSolver, Status
from el0ps.path import Path
from .utils import make_regression


def test_path():
    A, y, x_true = make_regression(3, 150, 200)
    M = 1.5 * np.max(np.abs(x_true))

    datafit = Leastsquares(y)
    penalty = Bigm(M)
    solver = BnbSolver(verbose=False)

    path = Path(
        lmbd_max=1e-0, lmbd_min=1e-1, lmbd_num=10, lmbd_normalized=True
    )
    fit = path.fit(solver, datafit, penalty, A)
    assert np.all(result.status == Status.OPTIMAL for _, result in fit.items())
