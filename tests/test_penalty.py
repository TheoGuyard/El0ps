import numpy as np
import pytest

from el0ps.penalty import ProximablePenalty, Bigm, L1norm, L2norm

n = 100
x = np.random.randn(n)
u = np.random.randn(n)
lmbd = np.random.rand()
bigm = np.linalg.norm(x, np.inf)
alpha = np.random.rand()
penalties = [Bigm(bigm), L1norm(alpha), L2norm(alpha)]


@pytest.mark.parametrize("penalty", penalties)
def test_instances(penalty):
    assert isinstance(penalty.__str__(), str)
    assert penalty.value(x) + penalty.conjugate(u) >= np.dot(x, u)
    for i, (xi, ui) in enumerate(zip(x, u)):
        assert penalty.value_scalar(i, xi) >= 0.0
        assert penalty.value_scalar(i, xi) == penalty.value_scalar(i, -xi)
        assert penalty.conjugate_scalar(i, ui) >= 0.0
        assert (
            penalty.value_scalar(i, xi) + penalty.conjugate_scalar(i, ui)
            >= xi * ui
        )
        zi = penalty.param_zerlimit(i)
        di = penalty.param_domlimit(i)
        vi = penalty.param_vallimit(i)
        li = penalty.param_levlimit(i, lmbd)
        si = penalty.param_sublimit(i, lmbd)
        assert zi >= 0.0
        assert di >= 0.0
        assert vi >= 0.0
        assert li >= 0.0
        assert si >= 0.0
        assert penalty.conjugate_scalar(i, di) == vi
        if di < np.inf:
            assert penalty.conjugate_scalar(i, di) < np.inf
        else:
            assert penalty.conjugate_scalar(i, di) == np.inf
        assert penalty.conjugate_scalar(i, di) == vi
        if si < np.inf:
            assert penalty.conjugate_scalar(i, li) == pytest.approx(lmbd)
            assert penalty.value_scalar(i, si) + penalty.conjugate_scalar(
                i, li
            ) == pytest.approx(si * li)
        else:
            assert penalty.conjugate_scalar(i, li) < lmbd

    assert penalty.value(np.zeros(n)) == 0.0
    assert penalty.value_scalar(i, 0.0) == 0.0

    if isinstance(penalty, ProximablePenalty):
        eta = np.random.rand()
        p = penalty.prox(x, eta)
        assert p.shape == x.shape
        for i, xi in enumerate(x):
            assert penalty.prox_scalar(i, xi, eta) == p[i]
