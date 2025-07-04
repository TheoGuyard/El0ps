import numpy as np
import pytest

from el0ps.datafit import (
    BaseDatafit,
    KullbackLeibler,
    Leastsquares,
    Logcosh,
    Logistic,
    Squaredhinge,
)

m = 100
y = np.random.randn(m)
datafits = [
    pytest.param(datafit, id=f"{datafit.__class__.__name__}")
    for datafit in [
        KullbackLeibler(np.ceil(10 * np.abs(y))),
        Leastsquares(y),
        Logcosh(y),
        Logistic(2.0 * (y > 0.0) - 1.0),
        Squaredhinge(2.0 * (y > 0.0) - 1.0),
    ]
]


@pytest.mark.parametrize("datafit", datafits)
def test_datafit(datafit: BaseDatafit):
    assert isinstance(datafit.__str__(), str)
    x = np.random.randn(m)
    u = np.random.randn(m)
    if isinstance(datafit, KullbackLeibler):
        x = np.abs(x) + 2. * datafit.e
        u = np.abs(u) + 2. * datafit.e
    assert datafit.value(x) + datafit.conjugate(u) >= np.dot(x, u)
    assert datafit.gradient_lipschitz_constant() >= 0.0
    assert datafit.gradient(x).shape == x.shape
    assert datafit.value(x) >= datafit.value(u) + np.dot(
        datafit.gradient(u), x - u
    )
