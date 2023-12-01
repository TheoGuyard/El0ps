import pathlib
import l0learn
import warnings
import numpy as np
import openml as oml
from libsvmdata import fetch_libsvm
from scipy import sparse
from el0ps.datafit import Kullbackleibler, Leastsquares, Logistic, Squaredhinge
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


def synthetic_A(datafit_name, m, n, rho, normalize):
    M = np.zeros(n)
    N1 = np.repeat(np.arange(n).reshape(n, 1), n).reshape(n, n)
    N2 = np.repeat(np.arange(n).reshape(1, n), n).reshape(n, n).T
    K = np.power(rho, np.abs(N1 - N2))
    A = np.random.multivariate_normal(M, K, size=m)
    if datafit_name == "Kullbackleibler":
        A = np.abs(A)
    if normalize:
        A /= np.linalg.norm(A, axis=0, ord=2)
    return A


def synthetic_y(datafit_name, x, A, m, snr):
    if datafit_name == "Kullbackleibler":
        y = np.random.poisson(-snr * (A @ x), m)
    elif datafit_name == "Leastsquares":
        y = A @ x
        e = np.random.randn(m)
        e *= (y @ y) / (np.sqrt(snr) * (e @ e))
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

    # Fit regularization path with L0Learn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cvfit = l0learn.cvfit(
            A,
            y,
            bindings[datafit_name],
            bindings[penalty_name],
            intercept = False,
            num_gamma = 1 if bindings[penalty_name] == "L0" else 100,
            gamma_max = 0.0 if bindings[penalty_name] == "L0" else m * 1e4,
            gamma_min = 0.0 if bindings[penalty_name] == "L0" else m * 1e-4,
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
            cv = cvfit.cv_means[i][j][0]
            f1 = 0.0 if x_true is None else f1_score(x_true, x)
            if f1 >= best_f1 and cv < best_cv:
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
    A = synthetic_A(datafit_name, m, n, rho, normalize)
    y = synthetic_y(datafit_name, x_true, A, m, snr)
    datafit, penalty, lmbd = calibrate_objective(
        datafit_name, penalty_name, A, y, x_true
    )
    return datafit, penalty, A, lmbd, x_true


def get_data_libsvm(datafit_name, penalty_name, dataset_name, normalize):
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    A, y = fetch_libsvm(dataset_name)
    if sparse.issparse(A):
        A = A.todense()
    A = A.reshape(*A.shape, order="F")
    zero_columns = np.abs(np.linalg.norm(A, axis=0)) < 1e-7
    if np.any(zero_columns):
        A = np.array(A[:, np.logical_not(zero_columns)])
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
    A = dataset[0].to_numpy().astype(float)
    y = dataset[1].to_numpy().flatten().astype(float)
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


def get_data_lattice(datafit_name, penalty_name, normalize=False):
    A_path = pathlib.Path(__file__).parent.joinpath(
        "datasets", "lattice_A.npy"
    )
    y_path = pathlib.Path(__file__).parent.joinpath(
        "datasets", "lattice_y.npy"
    )
    A = np.load(A_path)
    y = np.load(y_path)
    if normalize:
        A /= np.linalg.norm(A, axis=0, ord=2)
    datafit, penalty, lmbd = calibrate_objective(
        datafit_name, penalty_name, A, y
    )
    return datafit, penalty, A, lmbd, None


def get_data(dataset):
    if "dataset_type" not in dataset.keys():
        raise ValueError("Key `dataset_type` not found.")
    if dataset["dataset_type"] == "synthetic":
        return get_data_synthetic(
            dataset["datafit_name"],
            dataset["penalty_name"],
            dataset["dataset_opts"]["k"],
            dataset["dataset_opts"]["m"],
            dataset["dataset_opts"]["n"],
            dataset["dataset_opts"]["rho"],
            dataset["dataset_opts"]["snr"],
            dataset["dataset_opts"]["normalize"],
        )
    elif dataset["dataset_type"] == "libsvm":
        return get_data_libsvm(
            dataset["datafit_name"],
            dataset["penalty_name"],
            dataset["dataset_opts"]["dataset_name"],
            dataset["dataset_opts"]["normalize"],
        )
    elif dataset["dataset_type"] == "openml":
        return get_data_openml(
            dataset["datafit_name"],
            dataset["penalty_name"],
            dataset["dataset_opts"]["dataset_id"],
            dataset["dataset_opts"]["dataset_target"],
            dataset["dataset_opts"]["normalize"],
        )
    elif dataset["dataset_type"] == "lattice":
        return get_data_lattice(
            dataset["datafit_name"],
            dataset["penalty_name"],
            dataset["dataset_opts"]["normalize"],
        )
    else:
        raise ValueError("Unknown datatype {}".format(dataset["dataset_type"]))
