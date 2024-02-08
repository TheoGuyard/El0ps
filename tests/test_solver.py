import numpy as np

from el0ps.datafit import Leastsquares
from el0ps.penalty import Bigm
from el0ps.utils import compute_lmbd_max
from el0ps.solver import Status, BnbNode, BnbSolver


def test_solver():
    k, m, n = 3, 50, 100
    x = np.zeros(n)
    s = np.array(np.floor(np.linspace(0, n - 1, num=k)), dtype=int)
    x[s] = np.random.randn(k)
    A = np.random.randn(m, n)
    y = A @ x
    y += 0.01 * np.random.randn(m) * np.linalg.norm(y) ** 2
    M = 1.5 * np.max(np.abs(x))

    datafit = Leastsquares(y)
    penalty = Bigm(M)
    lmbd = 0.1 * compute_lmbd_max(datafit, penalty, A)

    S0 = np.zeros(n, dtype=bool)
    S1 = np.zeros(n, dtype=bool)
    Sb = np.ones(n, dtype=bool)
    x = np.random.randn(n)
    w = A @ x

    node = BnbNode(-1, S0, S1, Sb, -np.inf, +np.inf, 0., 0., x, w, np.copy(x))
    assert isinstance(node, BnbNode)
    assert isinstance(node.__str__(), str)

    node.fix_to(0, 0, A)
    node.fix_to(1, 1, A)
    assert node.x[0] == 0.0
    assert not node.Sb[0]
    assert node.S0[0]
    assert not node.Sb[1]
    assert node.S1[1]
    assert np.allclose(node.w, A @ node.x)

    solver = BnbSolver()
    result = solver.solve(datafit, penalty, A, lmbd, x_init=x)

    assert result.status == Status.OPTIMAL
