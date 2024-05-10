import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from el0ps.datafits import Leastsquares, Logistic, Squaredhinge
from el0ps.estimators import L0Regressor, L0Classifier, L0SVC
from el0ps.penalties import Bigm
from el0ps.solvers import Status
from el0ps.utils import compute_lmbd_max
from .utils import make_classification, make_regression, make_svc

k, m, n = 3, 50, 50
M_factor = 1.5
lmbd_factor = 0.1

A_cls, y_cls, coef_true_cls = make_classification(k, m, n)
M_cls = M_factor * np.max(np.abs(coef_true_cls))
lmbd_cls = lmbd_factor * compute_lmbd_max(
    Leastsquares(y_cls), Bigm(M_cls), A_cls
)

A_reg, y_reg, coef_true_reg = make_regression(k, m, n)
M_reg = M_factor * np.max(np.abs(coef_true_reg))
lmbd_reg = lmbd_factor * compute_lmbd_max(Logistic(y_reg), Bigm(M_reg), A_reg)

A_svc, y_svc, coef_true_svc = make_svc(k, m, n)
M_svc = M_factor * np.max(np.abs(coef_true_svc))
lmbd_svc = lmbd_factor * compute_lmbd_max(
    Squaredhinge(y_svc), Bigm(M_svc), A_svc
)

# TODO: test other estimators
test_data = [
    (L0Classifier(lmbd_cls, M_cls), A_cls, y_cls),
    (L0Regressor(lmbd_reg, M_reg), A_reg, y_reg),
    (L0SVC(lmbd_svc, M_svc), A_svc, y_svc),
]


@pytest.mark.parametrize("estimator,A,y", test_data)
def test_estimator(estimator, A, y):
    checks = check_estimator(estimator, generate_only=True)
    assert np.all(checks)
    estimator.fit(A, y)
    y_pred = estimator.predict(A)
    assert estimator.fit_result_.status == Status.OPTIMAL
    assert y_pred.shape == y.shape
