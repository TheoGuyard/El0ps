import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from el0ps.datafits import Leastsquares, Logistic, Squaredhinge
from el0ps.estimators import L0L1L2Regressor, L0L1L2Classifier, L0L1L2SVC
from el0ps.penalties import L1L2norm
from el0ps.solvers import Status
from el0ps.utils import compute_lmbd_max
from .utils import make_classification, make_regression, make_svc

k, m, n = 3, 50, 50
lmbd_factor = 0.1
alpha = 0.2
beta = 0.3

A_cls, y_cls, coef_true_cls = make_classification(k, m, n)
lmbd_cls = lmbd_factor * compute_lmbd_max(
    Logistic(y_cls), L1L2norm(alpha, beta), A_cls
)

A_reg, y_reg, coef_true_reg = make_regression(k, m, n)
lmbd_reg = lmbd_factor * compute_lmbd_max(
    Leastsquares(y_reg), L1L2norm(alpha, beta), A_reg
)

A_svc, y_svc, coef_true_svc = make_svc(k, m, n)
lmbd_svc = lmbd_factor * compute_lmbd_max(
    Squaredhinge(y_svc), L1L2norm(alpha, beta), A_svc
)

test_data = [
    (L0L1L2Classifier(lmbd_cls, alpha, beta), A_cls, y_cls),
    (L0L1L2Regressor(lmbd_reg, alpha, beta), A_reg, y_reg),
    (L0L1L2SVC(lmbd_cls, alpha, beta), A_svc, y_svc),
]


@pytest.mark.parametrize("estimator,A,y", test_data)
def test_estimator(estimator, A, y):
    checks = check_estimator(estimator, generate_only=True)
    assert np.all(checks)
    estimator.fit(A, y)
    y_pred = estimator.predict(A)
    assert estimator.fit_result_.status == Status.OPTIMAL
    assert y_pred.shape == y.shape
