import warnings
import l0learn
from numba import float64
import numpy as np
import openml as oml
from sklearn.linear_model import ElasticNet, LogisticRegression
from el0ps.datafit import Leastsquares, Logistic, Squaredhinge
from el0ps.penalty import (
    ProximablePenalty,
    Bigm,
    BigmL1norm,
    BigmL2norm,
    L1norm,
    L2norm,
)


class NeglogTriangular(ProximablePenalty):
    """Negative log-likelyhood of the triangular distribution penalty given by

    .. math:: h(x) = -alpha * ln(1 - |x|/sigma) if |x| <= sigma and +inf
    otherwise

    where `alpha` and `sigma` are positive hyperparameters.

    Parameters
    ----------
    alpha: float
        Penalty weight.
    sigma: float
        Distribution spead.
    """

    def __init__(self, alpha: float, sigma: float) -> None:
        self.alpha = alpha
        self.sigma = sigma

    def __str__(self) -> str:
        return "NeglogTriangular"

    def get_spec(self) -> tuple:
        spec = (
            ("alpha", float64),
            ("sigma", float64),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(alpha=self.alpha, sigma=self.sigma)

    def value(self, x: float) -> float:
        if np.abs(x) <= self.sigma:
            return -self.alpha * np.log(1.0 - np.abs(x) / self.sigma)
        else:
            return np.inf

    def conjugate(self, x: float) -> float:
        z = np.maximum((self.sigma / self.alpha) * np.abs(x) - 1.0, 0.0)
        return self.alpha * (z - np.log(z + 1.0))

    def prox(self, x: float, eta: float) -> float:
        w = self.sigma - np.sqrt(
            (self.sigma - np.abs(x)) ** 2 + 4.0 * eta * self.alpha
        )
        z = 0.5 * (x + np.sign(x) * w)
        return np.maximum(np.minimum(z, self.sigma), -self.sigma)

    def conjugate_scaling_factor(self, x: float) -> float:
        return 1.0

    def param_slope(self, lmbd: float) -> float:
        return self.approximate_param_slope(lmbd)

    def param_limit(self, lmbd: float) -> float:
        return self.sigma - 1.0 / self.param_slope(lmbd)

    def param_maxval(self) -> float:
        return np.inf


def f1_score(x_true, x):
    s = x != 0.0
    s_true = x_true != 0.0
    i = np.sum(s & s_true)
    p = 1.0 if not np.any(s) else i / np.sum(s)
    r = 1.0 if not np.any(s_true) else i / np.sum(s_true)
    f = 0.0 if (p + r) == 0.0 else 2.0 * p * r / (p + r)
    return f


def synthetic_x(k, n):
    x = np.zeros(n)
    s = np.array(np.floor(np.linspace(0, n - 1, num=k)), dtype=int)
    x[s] = np.random.randn(k)
    x[s] += np.sign(x[s])
    return x


def synthetic_A(m, n, rho, normalize):
    M = np.zeros(n)
    N1 = np.repeat(np.arange(n).reshape(n, 1), n).reshape(n, n)
    N2 = np.repeat(np.arange(n).reshape(1, n), n).reshape(n, n).T
    K = np.power(rho, np.abs(N1 - N2))
    A = np.random.multivariate_normal(M, K, size=m)
    if normalize:
        A /= np.linalg.norm(A, axis=0, ord=2)
    return A


def synthetic_y(datafit_name, x, A, m, snr):
    if datafit_name == "Leastsquares":
        y = A @ x
        e = np.random.randn(m)
        e *= np.sqrt((y @ y) / (snr * (e @ e)))
        y += e
    elif datafit_name == "Logistic":
        p = 1.0 / (1.0 + np.exp(-snr * (A @ x)))
        y = 2.0 * np.random.binomial(1, p, size=m) - 1.0
    elif datafit_name == "Squaredhinge":
        p = 1.0 / (1.0 + np.exp(-snr * (A @ x)))
        y = 2.0 * (p > 0.5) - 1.0
    else:
        raise ValueError(f"Unsupported data-fidelity function {datafit_name}")
    return y


def calibrate_objective(datafit_name, penalty_name, A, y, x_true=None):
    bindings = {
        "Leastsquares": "SquaredError",
        "Logistic": "Logistic",
        "Squaredhinge": "SquaredHinge",
        "Bigm": "L0",
        "BigmL1norm": "L0L1",
        "BigmL2norm": "L0L2",
        "L1norm": "L0L1",
        "L2norm": "L0L2",
    }

    assert datafit_name in bindings.keys()
    assert penalty_name in bindings.keys()

    m, n = A.shape

    # Datafit
    if datafit_name == "Leastsquares":
        datafit = Leastsquares(y)
    elif datafit_name == "Logistic":
        datafit = Logistic(y)
    elif datafit_name == "Squaredhinge":
        datafit = Squaredhinge(y)

    # Calibrate the penalty parameters w.r.t x_truth when it is known
    gamma_true = None
    if x_true is not None:
        s_true = x_true != 0.0
        A_true = A[:, s_true]
        clf = None
        if datafit_name == "Leastsquares":
            if penalty_name in ["L1norm", "BigmL1norm"]:
                clf = ElasticNet(l1_ratio=1.0, fit_intercept=False)
            elif penalty_name in ["L2norm", "BigmL2norm"]:
                clf = ElasticNet(l1_ratio=0.0, fit_intercept=False)
        elif datafit_name == "Logistic":
            if penalty_name in ["L1norm", "BigmL1norm"]:
                clf = LogisticRegression(penalty="l1", fit_intercept=False)
            elif penalty_name in ["L2norm", "BigmL2norm"]:
                clf = LogisticRegression(penalty="l2", fit_intercept=False)
        if clf is not None:
            c_best = (
                np.linalg.norm(
                    A_true.T @ datafit.gradient(np.zeros(m)), np.inf
                )
                / m
            )
            v_best = np.inf
            coef_grid = np.logspace(4, -4, 50) * c_best
            for coef in coef_grid:
                if isinstance(clf, ElasticNet):
                    clf.set_params(alpha=coef)
                elif isinstance(clf, LogisticRegression):
                    clf.set_params(C=1.0 / coef)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(A_true, y)
                x_test = clf.coef_.ravel().copy()
                v_test = np.linalg.norm(x_true[s_true] - x_test, 2)
                if v_test < v_best:
                    v_best = v_test
                    c_best = coef
            gamma_true = c_best

    # Fit regularization path with L0Learn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cvfit = l0learn.cvfit(
            A,
            y,
            bindings[datafit_name],
            bindings[penalty_name],
            intercept=False,
            num_gamma=1 if penalty_name == "Bigm" or gamma_true else 10,
            gamma_max=m * gamma_true if gamma_true else m * 10,
            gamma_min=m * gamma_true if gamma_true else m * 0.0001,
        )

    # Penalty and L0-norm parameters calibration from L0learn path
    best_M = None
    best_lmbda = None
    best_gamma = None
    best_cv = np.inf
    best_f1 = 0.0
    for i, gamma in enumerate(cvfit.gamma):
        for j, lmbda in enumerate(cvfit.lambda_0[i]):
            x = cvfit.coeff(lmbda, gamma)
            x = np.array(x.todense()).reshape(n + 1)[1:]
            cv = cvfit.cv_means[i][j]
            f1 = 0.0 if x_true is None else f1_score(x_true, x)
            if f1 >= best_f1 and cv <= best_cv:
                best_M = 1.5 * np.max(np.abs(x))
                best_lmbda = lmbda / m
                best_gamma = gamma / m
                best_cv = cv
                best_f1 = f1

    if penalty_name == "Bigm":
        penalty = Bigm(best_M)
    elif penalty_name == "BigmL1norm":
        penalty = BigmL1norm(best_M, best_gamma)
    elif penalty_name == "BigmL2norm":
        penalty = BigmL2norm(best_M, best_gamma)
    elif penalty_name == "L1norm":
        penalty = L1norm(best_gamma)
    elif penalty_name == "L2norm":
        penalty = L2norm(best_gamma)

    return datafit, penalty, best_lmbda


def get_data_synthetic(
    datafit_name, penalty_name, k, m, n, rho, snr, normalize
):
    x_true = synthetic_x(k, n)
    A = synthetic_A(m, n, rho, normalize)
    y = synthetic_y(datafit_name, x_true, A, m, snr)
    datafit, penalty, lmbd = calibrate_objective(
        datafit_name, penalty_name, A, y, x_true
    )
    return datafit, penalty, A, lmbd, x_true


def get_data_openml(
    datafit_name, penalty_name, dataset_id, dataset_target, normalize
):
    dataset = oml.datasets.get_dataset(
        dataset_id,
        download_data=False,
        download_qualities=False,
        download_features_meta_data=False,
    )
    dataset = dataset.get_data(target=dataset_target)
    A = dataset[0].to_numpy()
    y = dataset[1].to_numpy().flatten()
    assert A.ndim == 2
    assert y.ndim == 1
    A = A[:, np.linalg.norm(A, axis=0, ord=2) != 0.0]
    if normalize:
        A /= np.linalg.norm(A, axis=0, ord=2)
    if datafit_name in ["Logistic", "Squaredhinge"]:
        y_cls = np.unique(y)
        assert y_cls.size == 2
        y_idx0 = y == y_cls[0]
        y_idx1 = y == y_cls[1]
        y = np.zeros(y.size, dtype=float)
        y[y_idx0] = -1.0
        y[y_idx1] = 1.0
    datafit, penalty, lmbd = calibrate_objective(
        datafit_name, penalty_name, A, y
    )
    return datafit, penalty, A, lmbd, None


def get_data_atp(nnz_proba, m, n, snr, t):
    s = np.random.binomial(1, nnz_proba, size=n).astype(bool)
    k = np.flatnonzero(s).size
    x_true = np.zeros(n)
    x_true[s] = np.random.triangular(-t, 0.0, t, size=k)
    A = synthetic_A(m, n, 0.0, True)
    y = synthetic_y("Leastsquares", x_true, A, m, snr)
    e = np.var(y - A @ x_true)
    datafit = Leastsquares(y)
    penalty = NeglogTriangular(e / m, t)
    lmbd = (e / m) * np.log((1.0 - nnz_proba) / nnz_proba)
    return datafit, penalty, A, lmbd, x_true


def get_data(dataset):
    if "datatype" not in dataset.keys():
        raise ValueError("Key `datatype` not found.")

    if dataset["datatype"] == "synthetic":
        return get_data_synthetic(
            dataset["datafit_name"],
            dataset["penalty_name"],
            dataset["k"],
            dataset["m"],
            dataset["n"],
            dataset["rho"],
            dataset["snr"],
            dataset["normalize"],
        )
    elif dataset["datatype"] == "openml":
        return get_data_openml(
            dataset["datafit_name"],
            dataset["penalty_name"],
            dataset["dataset_id"],
            dataset["dataset_target"],
            dataset["normalize"],
        )
    elif dataset["datatype"] == "atp":
        return get_data_atp(
            dataset["nnz_proba"],
            dataset["m"],
            dataset["n"],
            dataset["snr"],
            dataset["t"],
        )
    else:
        raise ValueError("Unknown datatype {}".format(dataset["datatype"]))
