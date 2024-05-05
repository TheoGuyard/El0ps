import numpy as np
import pytest

from el0ps.datafits import (
    BaseDatafit,
    SmoothDatafit,
    Kullbackleibler,
    Leastsquares,
    Logcosh,
    Logistic,
    Squaredhinge,
)

m = 100
y = np.random.randn(m)
x = np.random.randn(m)
u = np.random.randn(m)
base_datafits = [BaseDatafit, SmoothDatafit]
datafits = [
    Kullbackleibler(np.abs(y)),
    Leastsquares(y),
    Logcosh(y),
    Logistic(2.0 * (y > 0.0) - 1.0),
    Squaredhinge(2.0 * (y > 0.0) - 1.0),
]


@pytest.mark.parametrize("datafit", datafits)
def test_instances(datafit):
    assert isinstance(datafit.__str__(), str)
    assert datafit.value(x) + datafit.conjugate(u) >= np.dot(x, u)
    if isinstance(datafit, SmoothDatafit):
        assert datafit.L >= 0.0
        assert datafit.gradient(x).shape == x.shape
        assert datafit.value(x) >= datafit.value(u) + np.dot(
            datafit.gradient(u), x - u
        )
