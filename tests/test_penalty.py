import numpy as np
import pytest

from el0ps.penalty import (
    ProximablePenalty,
    Bigm,
    BigmL1norm,
    BigmL2norm,
    L1norm,
    L2norm,
    L1L2norm,
)

n = 100
x = np.random.randn(n)
u = np.random.randn(n)
lmbd = np.random.rand()
bigm = np.linalg.norm(x, np.inf)
alpha = np.random.rand()
beta = np.random.rand()
penalties = [
    Bigm(bigm),
    BigmL1norm(bigm, alpha),
    BigmL2norm(bigm, alpha),
    L1norm(alpha),
    L2norm(alpha),
    L1L2norm(alpha, beta),
]


@pytest.mark.parametrize("penalty", penalties)
def test_instances(penalty):
    assert isinstance(penalty.__str__(), str)
    assert penalty.value(0.0) == 0.0
    for xi, ui in zip(x, u):
        assert penalty.value(xi) >= 0.0
        assert penalty.value(xi) == penalty.value(-xi)
        assert penalty.conjugate(ui) >= 0.0
        assert penalty.value(xi) + penalty.conjugate(ui) >= xi * ui - 1e-10
    slope = penalty.param_slope(lmbd)
    limit = penalty.param_limit(lmbd)
    maxval = penalty.param_maxval()
    maxzer = penalty.param_maxzer()
    slope_approx = penalty.approximate_param_slope(lmbd, tol=1e-8)
    assert slope >= 0.0
    assert limit >= 0.0
    assert maxval >= 0.0
    assert maxzer >= 0.0
    assert slope == pytest.approx(slope_approx)
    if limit < np.inf:
        assert penalty.conjugate(slope) == pytest.approx(lmbd)
        assert (
            penalty.value(limit) + penalty.conjugate(slope)
            >= limit * slope - 1e-10
        )
    else:
        assert penalty.conjugate(slope) < lmbd
    if maxval < np.inf:
        assert penalty.conjugate(maxval) < np.inf
    else:
        assert penalty.conjugate(maxval) == np.inf
    assert penalty.conjugate(maxzer) == 0.0

    if isinstance(penalty, ProximablePenalty):
        eta = np.random.rand()
        for xi in x:
            pi = penalty.prox(xi, eta)
            v1 = 0.5 * (pi - xi) ** 2 + eta * penalty.value(pi)
            v2 = eta * penalty.value(xi)
            assert v1 <= v2
