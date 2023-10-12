import numpy as np

from el0ps.datafit import Leastsquares
from el0ps.penalty import Bigm
from el0ps.solver import BnbSolver, Status
from el0ps.path import Path


def test_path():
    k, m, n = 3, 20, 30
    x = np.zeros(n)
    s = np.array(np.floor(np.linspace(0, n - 1, num=k)), dtype=int)
    x[s] = np.random.randn(k)
    A = np.random.randn(m, n)
    y = A @ x
    y += np.random.randn(m) * 0.1 * (np.linalg.norm(y) ** 2 / m)
    M = 1.5 * np.max(np.abs(x))

    datafit = Leastsquares(y)
    penalty = Bigm(M)
    solver = BnbSolver()

    path = Path()
    fit = path.fit(solver, datafit, penalty, A)
    assert np.all(status == Status.OPTIMAL for status in fit["status"])
