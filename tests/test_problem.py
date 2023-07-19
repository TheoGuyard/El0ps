import numpy as np
import pytest

from el0ps.datafit import Quadratic
from el0ps.penalty import L1norm, L2norm
from el0ps.problem import Problem, compute_lmbd_max


m = 100
n = 150
A = np.random.randn(m, n)
y = np.random.randn(m)
x = np.random.randn(n)
w = A @ x
lmbd = np.random.rand()
alpha = np.random.rand()
datafit = Quadratic(y)
penalty = L2norm(alpha)


def test_problem():
    with pytest.raises(ValueError):
        problem = Problem(datafit, penalty, A, "lmbd")

    with pytest.raises(ValueError):
        problem = Problem(datafit, penalty, A, -1.0)

    with pytest.raises(ValueError):
        problem = Problem(datafit, penalty, "A", lmbd)

    with pytest.raises(ValueError):
        problem = Problem(datafit, penalty, np.zeros(0), lmbd)

    problem = Problem(datafit, penalty, A, lmbd)
    assert isinstance(problem.__str__(), str)
    obj1 = problem.value(x)
    obj2 = problem.value(x, w)
    assert isinstance(obj1, float)
    assert isinstance(obj2, float)
    assert obj1 == pytest.approx(obj2)

    lmbd_max = compute_lmbd_max(datafit, penalty, A)
    assert lmbd_max >= 0.0
    assert lmbd_max <= np.inf

    alpha = np.linalg.norm(A.T @ y, np.inf)
    lmbd_max = compute_lmbd_max(datafit, L1norm(alpha), A)
    assert lmbd_max == 0.0
