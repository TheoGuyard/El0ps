import numpy as np
import pytest

from el0ps.datafit import (
    BaseDatafit,
    ProximableDatafit,
    SmoothDatafit,
    Quadratic,
)

m = 100
y = np.random.randn(m)
x = np.random.randn(m)
u = np.random.randn(m)
base_datafits = [BaseDatafit, ProximableDatafit, SmoothDatafit]
datafits = [Quadratic(y)]


@pytest.mark.parametrize("base_datafit", base_datafits)
def test_base(base_datafit):
    class NewDatafitClass(base_datafit):
        pass

    with pytest.raises(TypeError):
        datafit = NewDatafitClass()  # noqa: F841


@pytest.mark.parametrize("datafit", datafits)
def test_instances(datafit):
    assert isinstance(datafit.__str__(), str)
    assert datafit.value(x) + datafit.conjugate(u) >= np.dot(x, u)

    if isinstance(datafit, ProximableDatafit):
        eta = np.random.rand()
        assert datafit.prox(x, eta).shape == x.shape

    if isinstance(datafit, SmoothDatafit):
        assert datafit.L >= 0.0
        assert datafit.gradient(x).shape == x.shape
        assert datafit.value(x) >= datafit.value(u) + np.dot(
            datafit.gradient(u), x - u
        )
