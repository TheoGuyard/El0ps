import numpy as np
import pytest

from el0ps.compilation import CompilableClass, compiled_clone
from el0ps.penalty import (
    BasePenalty,
    SymmetricPenalty,
    Bigm,
    BigmL1norm,
    BigmL2norm,
    BigmPositiveL1norm,
    BigmPositiveL2norm,
    Bounds,
    L1norm,
    L2norm,
    L1L2norm,
    PositiveL1norm,
    PositiveL2norm,
    compute_param_slope_pos,
    compute_param_slope_neg,
)

n = 100
x = np.random.randn(n)
u = np.random.randn(n)
lmbd = np.random.rand()
bigm = np.linalg.norm(x, np.inf)
alpha = np.random.rand()
beta = np.random.rand()
penalties = [
    pytest.param(penalty, id=f"{penalty.__class__.__name__}")
    for penalty in [
        Bigm(bigm),
        BigmL1norm(bigm, alpha),
        BigmL2norm(bigm, alpha),
        BigmPositiveL1norm(bigm, alpha),
        BigmPositiveL2norm(bigm, beta),
        Bounds(-2 * np.ones(n), 2 * np.ones(n)),
        L1norm(alpha),
        L2norm(alpha),
        L1L2norm(alpha, beta),
        PositiveL1norm(alpha),
        PositiveL2norm(beta),
    ]
]


@pytest.mark.parametrize("penalty", penalties)
def test_penalty(penalty: BasePenalty):

    if isinstance(penalty, CompilableClass):
        penalty = compiled_clone(penalty)

    assert isinstance(penalty.__str__(), str)

    for i, (xi, ui) in enumerate(zip(x, u)):

        assert penalty.value(i, 0.0) == 0.0
        assert penalty.value(i, xi) >= 0.0
        assert penalty.conjugate(i, ui) >= 0.0
        assert (
            penalty.value(i, xi) + penalty.conjugate(i, ui) >= xi * ui - 1e-10
        )

        tol = 1e-8
        slope_pos = penalty.param_slope_pos(i, lmbd)
        slope_neg = penalty.param_slope_neg(i, lmbd)
        slope_pos_approx = compute_param_slope_pos(penalty, i, lmbd, tol=tol)
        slope_neg_approx = compute_param_slope_neg(penalty, i, lmbd, tol=tol)
        assert slope_pos >= 0.0
        assert slope_neg <= 0.0
        assert slope_pos == pytest.approx(slope_pos_approx, abs=tol)
        assert slope_neg == pytest.approx(slope_neg_approx, abs=tol)

        limit_pos = penalty.param_limit_pos(i, lmbd)
        limit_neg = penalty.param_limit_neg(i, lmbd)
        if limit_pos == 0.0:
            assert slope_pos == np.inf
        elif limit_pos < np.inf:
            assert penalty.conjugate(i, slope_pos) == pytest.approx(lmbd)
            assert (
                penalty.value(i, limit_pos) + penalty.conjugate(i, slope_pos)
                >= limit_pos * slope_pos - 1e-10
            )
        else:
            assert penalty.conjugate(i, slope_pos) < lmbd
        if limit_neg == 0.0:
            assert slope_neg == -np.inf
        elif limit_neg > -np.inf:
            assert penalty.conjugate(i, slope_neg) == pytest.approx(lmbd)
            assert (
                penalty.value(i, limit_neg) + penalty.conjugate(i, slope_neg)
                >= limit_neg * slope_neg - 1e-10
            )
        else:
            assert penalty.conjugate(i, slope_neg) < lmbd

        if isinstance(penalty, SymmetricPenalty):
            assert slope_pos == pytest.approx(-slope_neg)
            assert limit_pos == pytest.approx(-limit_neg)

        eta = np.random.rand()
        pi = penalty.prox(i, xi, eta)
        v1 = 0.5 * (pi - xi) ** 2 + eta * penalty.value(i, pi)
        v2 = eta * penalty.value(i, xi)
        assert v1 <= v2
