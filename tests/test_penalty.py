import numpy as np
import pytest

from el0ps.penalty import (
    BasePenalty,
    SymmetricPenalty,
    Bigm,
    BigmL1norm,
    BigmL2norm,
    BigmPositiveL1norm,
    Bounds,
    L1norm,
    L2norm,
    L1L2norm,
    PositiveL1norm,
    PositiveL2norm,
)
from el0ps.utils import (
    compute_param_slope_pos_scalar,
    compute_param_slope_neg_scalar,
    compute_param_limit_pos_scalar,
    compute_param_limit_neg_scalar,
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
    BigmPositiveL1norm(bigm, alpha),
    Bounds(-2 * np.ones(n), 2 * np.ones(n)),
    L1norm(alpha),
    L2norm(alpha),
    L1L2norm(alpha, beta),
    PositiveL1norm(alpha),
    PositiveL2norm(beta),
]


@pytest.mark.parametrize("penalty", penalties)
def test_instances(penalty: BasePenalty):
    assert isinstance(penalty.__str__(), str)
    for i, (xi, ui) in enumerate(zip(x, u)):
        assert penalty.value_scalar(i, 0.0) == 0.0
        assert penalty.value_scalar(i, xi) >= 0.0
        assert penalty.conjugate_scalar(i, ui) >= 0.0
        assert (
            penalty.value_scalar(i, xi) + penalty.conjugate_scalar(i, ui)
            >= xi * ui - 1e-10
        )
        slope_pos = penalty.param_slope_pos_scalar(i, lmbd)
        slope_neg = penalty.param_slope_neg_scalar(i, lmbd)
        limit_pos = penalty.param_limit_pos_scalar(i, lmbd)
        limit_neg = penalty.param_limit_neg_scalar(i, lmbd)
        slope_pos_approx = compute_param_slope_pos_scalar(penalty, i, lmbd)
        slope_neg_approx = compute_param_slope_neg_scalar(penalty, i, lmbd)
        limit_pos_approx = compute_param_limit_pos_scalar(penalty, i, lmbd)
        limit_neg_approx = compute_param_limit_neg_scalar(penalty, i, lmbd)
        assert slope_pos >= 0.0
        assert slope_neg <= 0.0
        assert limit_pos >= 0.0
        assert limit_neg <= 0.0
        assert slope_pos == pytest.approx(slope_pos_approx)
        assert slope_neg == pytest.approx(slope_neg_approx)
        assert limit_pos == pytest.approx(limit_pos_approx)
        assert limit_neg == pytest.approx(limit_neg_approx)
        if limit_pos == 0.0:
            assert slope_pos == np.inf
        elif limit_pos < np.inf:
            assert penalty.conjugate_scalar(i, slope_pos) == pytest.approx(
                lmbd
            )
            assert (
                penalty.value_scalar(i, limit_pos)
                + penalty.conjugate_scalar(i, slope_pos)
                >= limit_pos * slope_pos - 1e-10
            )
        else:
            assert penalty.conjugate_scalar(i, slope_pos) < lmbd
        if limit_neg == 0.0:
            assert slope_neg == -np.inf
        elif limit_neg > -np.inf:
            assert penalty.conjugate_scalar(i, slope_neg) == pytest.approx(
                lmbd
            )
            assert (
                penalty.value_scalar(i, limit_neg)
                + penalty.conjugate_scalar(i, slope_neg)
                >= limit_neg * slope_neg - 1e-10
            )
        else:
            assert penalty.conjugate_scalar(i, slope_neg) < lmbd
        eta = np.random.rand()
        pi = penalty.prox_scalar(i, xi, eta)
        v1 = 0.5 * (pi - xi) ** 2 + eta * penalty.value_scalar(i, pi)
        v2 = eta * penalty.value_scalar(i, xi)
        assert v1 <= v2
    assert penalty.value(x) >= 0.0
    assert penalty.conjugate(u) >= 0.0
    assert penalty.value(x) + penalty.conjugate(u) >= x @ u - 1e-10
    assert penalty.prox(x, 1.0).shape == x.shape
    assert penalty.subdiff(x).shape == (x.size, 2)
    assert penalty.conjugate_subdiff(u).shape == (u.size, 2)
    assert len(penalty.param_slope_pos(lmbd, range(x.size))) == x.size
    assert len(penalty.param_slope_neg(lmbd, range(x.size))) == x.size
    assert len(penalty.param_limit_pos(lmbd, range(x.size))) == x.size
    assert len(penalty.param_limit_neg(lmbd, range(x.size))) == x.size
    if isinstance(penalty, SymmetricPenalty):
        sp = penalty.param_slope_pos(lmbd, range(x.size))
        sn = penalty.param_slope_neg(lmbd, range(x.size))
        assert np.allclose(sp, -sn)
        lp = penalty.param_limit_pos(lmbd, range(x.size))
        ln = penalty.param_limit_neg(lmbd, range(x.size))
        assert np.allclose(lp, -ln)
