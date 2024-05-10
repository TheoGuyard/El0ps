import numpy as np
import pytest

from el0ps.penalties import (
    Bigm,
    BigmL1norm,
    BigmL2norm,
    BigmL1L2norm,
    L1norm,
    L2norm,
    L1L2norm,
)
from el0ps.utils import compute_param_slope_scalar, compute_param_limit_scalar

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
    BigmL1L2norm(bigm, alpha, beta),
    L1norm(alpha),
    L2norm(alpha),
    L1L2norm(alpha, beta),
]


@pytest.mark.parametrize("penalty", penalties)
def test_instances(penalty):
    assert isinstance(penalty.__str__(), str)
    for i, (xi, ui) in enumerate(zip(x, u)):
        assert penalty.value_scalar(i, 0.0) == 0.0
        assert penalty.value_scalar(i, xi) >= 0.0
        assert penalty.value_scalar(i, xi) == penalty.value_scalar(i, -xi)
        assert penalty.conjugate_scalar(i, ui) >= 0.0
        assert (
            penalty.value_scalar(i, xi) + penalty.conjugate_scalar(i, ui)
            >= xi * ui - 1e-10
        )
        slope = penalty.param_slope_scalar(i, lmbd)
        limit = penalty.param_limit_scalar(i, lmbd)
        maxval = penalty.param_maxval_scalar(i)
        maxdom = penalty.param_maxdom_scalar(i)
        slope_approx = compute_param_slope_scalar(penalty, i, lmbd, tol=1e-8)
        limit_approx = compute_param_limit_scalar(penalty, i, lmbd)
        assert slope >= 0.0
        assert limit >= 0.0
        assert maxval >= 0.0
        assert maxdom >= 0.0
        assert slope == pytest.approx(slope_approx)
        assert limit == pytest.approx(limit_approx)
        if limit < np.inf:
            assert penalty.conjugate_scalar(i, slope) == pytest.approx(lmbd)
            assert (
                penalty.value_scalar(i, limit)
                + penalty.conjugate_scalar(i, slope)
                >= limit * slope - 1e-10
            )
        else:
            assert penalty.conjugate_scalar(i, slope) < lmbd
        if maxval < np.inf:
            assert penalty.conjugate_scalar(i, maxval) < np.inf
        else:
            assert penalty.conjugate_scalar(i, maxval) == np.inf
        if maxdom < np.inf:
            assert penalty.conjugate_scalar(i, maxdom) < np.inf
        eta = np.random.rand()
        pi = penalty.prox_scalar(i, xi, eta)
        v1 = 0.5 * (pi - xi) ** 2 + eta * penalty.value_scalar(i, pi)
        v2 = eta * penalty.value_scalar(i, xi)
        assert v1 <= v2
