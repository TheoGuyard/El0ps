import numpy as np
import pytest

from el0ps.datafit import Quadratic
from el0ps.penalty import Bigm
from el0ps.problem import Problem, compute_lmbd_max
from el0ps.solver import Status, BnbNode, BnbSolver, GurobiSolver


def test_solver():
    k, m, n = 3, 20, 30
    x = np.zeros(n)
    s = np.array(np.floor(np.linspace(0, n - 1, num=k)), dtype=int)
    x[s] = np.random.randn(k)
    A = np.random.randn(m, n)
    y = A @ x
    y += np.random.randn(m) * 0.1 * (np.linalg.norm(y)**2 / m)
    M = 1.5 * np.max(np.abs(x))

    datafit = Quadratic(y)
    penalty = Bigm(M)
    lmbd = 0.01 * compute_lmbd_max(datafit, penalty, A)
    problem = Problem(datafit, penalty, A, lmbd)

    S0 = np.zeros(n, dtype=bool)
    S1 = np.zeros(n, dtype=bool)
    Sb = np.ones(n, dtype=bool)
    x = np.random.randn(n)
    w = problem.A @ x
    u = -problem.datafit.gradient(w)

    node = BnbNode(S0, S1, Sb, -np.inf, +np.inf, x, w, u, np.copy(x))
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

    with pytest.raises(ValueError):
        node.fix_to(problem, 0, 1)
    with pytest.raises(ValueError):
        node.fix_to(problem, 2, 2)

    bnb_solver = BnbSolver()
    mip_solver = GurobiSolver()

    bnb_result = bnb_solver.solve(problem)
    mip_result = mip_solver.solve(problem)

    assert bnb_result.status == Status.OPTIMAL
    assert mip_result.status == Status.OPTIMAL
    assert np.allclose(bnb_result.objective_value, mip_result.objective_value)
    assert np.allclose(bnb_result.z, mip_result.z)
