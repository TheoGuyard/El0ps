import pathlib
import l0learn
import numpy as np
import openml as oml
from libsvmdata import fetch_libsvm
from scipy import sparse
from ucimlrepo import fetch_ucirepo
from el0ps.datafit import *  # noqa
from el0ps.penalty import Bigm, BigmL1norm, BigmL2norm, L1norm, L2norm


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
    x[s] = np.sign(np.random.randn(s.size))
    return x


def synthetic_A(model, m, n, rho):
    M = np.zeros(n)
    N1 = np.repeat(np.arange(n).reshape(n, 1), n).reshape(n, n)
    N2 = np.repeat(np.arange(n).reshape(1, n), n).reshape(n, n).T
    K = np.power(rho, np.abs(N1 - N2))
    A = np.random.multivariate_normal(M, K, size=m)
    if model == "poisson":
        A = np.abs(A)
    return A


def synthetic_y(model, x, A, m, snr):
    if model == "linear":
        y = A @ x
        e = np.random.randn(m)
        e *= (y @ y) / (np.sqrt(snr) * (e @ e))
        y += e
    elif model == "logistic":
        p = 1.0 / (1.0 + np.exp(-snr * (A @ x)))
        y = 2.0 * np.random.binomial(1, p, size=m) - 1.0
    elif model == "svm":
        p = 1.0 / (1.0 + np.exp(-snr * (A @ x)))
        y = 2.0 * (p > 0.5) - 1.0
    elif model == "poisson":
        y = np.random.poisson(-snr * (A @ x), m)
    elif model == "random":
        y = np.random.normal(0.0, snr, size=m)
    else:
        raise ValueError(f"Unsupported model {model}")
    return y


def get_data_synthetic(model, k, m, n, rho, snr, normalize=False):
    x = synthetic_x(k, n)
    A = synthetic_A(model, m, n, rho)
    if normalize:
        A /= np.linalg.norm(A, axis=0, ord=2)
    y = synthetic_y(model, x, A, m, snr)
    if model == "random":
        x = None
    return A, y, x


def get_data_libsvm(dataset_name):
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    A, y = fetch_libsvm(dataset_name)
    return A, y, None


def get_data_openml(dataset_id, dataset_target):
    dataset = oml.datasets.get_dataset(
        dataset_id,
        download_data=False,
        download_qualities=False,
        download_features_meta_data=False,
    )
    dataset = dataset.get_data(target=dataset_target)
    A = dataset[0].to_numpy().astype(float)
    y = dataset[1].to_numpy().flatten().astype(float)
    return A, y, None


def get_data_uciml(dataset_id):
    dataset = fetch_ucirepo(id=dataset_id)
    A = dataset.data.features
    y = dataset.data.targets
    A = A.to_numpy().astype(float)
    y = y.to_numpy().flatten().astype(float)
    return A, y, None


def get_data_hardcoded(dataset_name):
    A_path = (
        pathlib.Path(__file__)
        .parent.joinpath("datasets", dataset_name + "_A")
        .with_suffix(".npy")
    )
    A = np.load(A_path)

    y_path = (
        pathlib.Path(__file__)
        .parent.joinpath("datasets", dataset_name + "_y")
        .with_suffix(".npy")
    )
    y = np.load(y_path)

    x_path = A_path = (
        pathlib.Path(__file__)
        .parent.joinpath("datasets", dataset_name + "_x")
        .with_suffix(".npy")
    )
    x = None if not x_path.exists() else np.load(x_path)

    return A, y, x


def process_data(datafit_name, A, y, x_true, interactions, center, normalize):
    if sparse.issparse(A):
        A = A.todense()
    if not A.flags["F_CONTIGUOUS"] or not A.flags["OWNDATA"]:
        A = np.array(A, order="F")
    if interactions:
        t = np.triu_indices(A.shape[1], k=1)
        A = np.multiply(A[:, t[0]], A[:, t[1]])
        x_true = None
    zero_columns = np.abs(np.linalg.norm(A, axis=0)) < 1e-7
    if np.any(zero_columns):
        A = np.array(A[:, np.logical_not(zero_columns)], order="F")
    if center:
        A -= np.mean(A, axis=0)
        y -= np.mean(y)
    if normalize:
        A /= np.linalg.norm(A, axis=0, ord=2)
        y /= np.linalg.norm(y, ord=2)
    if datafit_name in ["Logistic", "Squaredhinge"]:
        y_cls = np.unique(y)
        assert y_cls.size == 2
        y_cls0 = y == y_cls[0]
        y_cls1 = y == y_cls[1]
        y = np.zeros(y.size, dtype=float)
        y[y_cls0] = -1.0
        y[y_cls1] = 1.0
    return A, y, x_true


def get_data(dataset):
    get_data_func = "get_data_" + dataset["dataset_type"]
    A, y, x_true = eval(get_data_func)(**dataset["dataset_opts"])
    A, y, x_true = process_data(
        dataset["datafit_name"], A, y, x_true, **dataset["process_opts"]
    )
    return A, y, x_true


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
    datafit = eval(datafit_name)(y)

    # Fit regularization path with L0Learn
    cvfit = l0learn.cvfit(
        A,
        y,
        bindings[datafit_name],
        bindings[penalty_name],
        intercept=False,
        num_gamma=1 if bindings[penalty_name] == "L0" else 40,
        gamma_max=0.0 if bindings[penalty_name] == "L0" else m * 1e2,
        gamma_min=0.0 if bindings[penalty_name] == "L0" else m * 1e-4,
        num_folds=5,
    )

    # Penalty and L0-norm parameters calibration from L0learn path
    best_M = None
    best_lmbda = None
    best_gamma = None
    best_cv = np.inf
    best_f1 = 0.0
    best_x = None
    for i, gamma in enumerate(cvfit.gamma):
        for j, lmbda in enumerate(cvfit.lambda_0[i]):
            x = cvfit.coeff(lmbda, gamma, include_intercept=False)
            x = np.array(x.todense()).reshape(n)
            cv = cvfit.cv_means[i][j][0]
            f1 = 0.0 if x_true is None else f1_score(x_true, x)
            if f1 >= best_f1 and cv < best_cv:
                best_M = 1.5 * np.max(np.abs(x))
                best_lmbda = lmbda / m
                best_gamma = gamma / m
                best_cv = cv
                best_f1 = f1
                best_x = np.copy(x)

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

    return datafit, penalty, best_lmbda, best_x
