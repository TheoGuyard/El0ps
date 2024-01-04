import numpy as np

from el0ps.datafit import Leastsquares
from el0ps.penalty import Bigm
from el0ps.problem import Problem, compute_lmbd_max
from el0ps.solver import Status, BnbNode, BnbSolver


def test_solver():
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
    lmbd = 0.01 * compute_lmbd_max(datafit, penalty, A)
    problem = Problem(datafit, penalty, A, lmbd)

    S0 = np.zeros(n, dtype=bool)
    S1 = np.zeros(n, dtype=bool)
    Sb = np.ones(n, dtype=bool)
    x = np.random.randn(n)
    w = problem.A @ x
    u = -problem.datafit.gradient(w)

    node = BnbNode(-1, S0, S1, Sb, -np.inf, +np.inf, 0., 0., x, w, u, np.copy(x))
    assert isinstance(node, BnbNode)
    assert isinstance(node.__str__(), str)

    node.fix_to(problem, 0, 0)
    node.fix_to(problem, 1, 1)
    assert node.x[0] == 0.0
    assert not node.Sb[0]
    assert node.S0[0]
    assert not node.Sb[1]
    assert node.S1[1]
    assert np.allclose(node.w, problem.A @ node.x)
    assert np.allclose(node.u, -problem.datafit.gradient(w))

    solver = BnbSolver()
    result = solver.solve(problem)

    assert result.status == Status.OPTIMAL
