import pathlib
import l0learn
import numpy as np
import openml as oml
from libsvmdata import fetch_libsvm
from scipy import sparse
from scipy.fftpack import dct
from ucimlrepo import fetch_ucirepo
from el0ps.datafit import *  # noqa
from el0ps.penalty import Bigm, BigmL1norm, BigmL2norm, L1norm, L2norm


def f1_score(x_true, x):
    """Compute the F1 support recovery score of x with respect to x_true."""
    s = x != 0.0
    s_true = x_true != 0.0
    i = np.sum(s & s_true)
    p = 1.0 if not np.any(s) else i / np.sum(s)
    r = 1.0 if not np.any(s_true) else i / np.sum(s_true)
    f = 0.0 if (p + r) == 0.0 else 2.0 * p * r / (p + r)
    return f


def synthetic_x(k, n):
    """Generate a k-sparse vector or size n with evenly-spaced non-zero entries
    of unit amplitude and ramdom sign."""
    x = np.zeros(n)
    s = np.array(np.floor(np.linspace(0, n - 1, num=k)), dtype=int)
    x[s] = np.sign(np.random.randn(s.size))
    return x


def synthetic_A(matrix, m, n, normalize):
    """Generate a matrix A of size (m, n). If matrix=="correlated(r)", each
    row is an independent realization of a multivariate normal distribution
    with zero mean and covariance matrix K with each entry defined as
    K[i,j]=r^|i-j| for some r in [0, 1). If matrix=="dct", the rows are the DCT
    basis functions. If matrix=="toeplitz", the rows are shifted gaussian."""
    if matrix.startswith("correlated"):
        r = float(matrix.split("(")[1].split(")")[0])
        M = np.zeros(n)
        N1 = np.repeat(np.arange(n).reshape(n, 1), n).reshape(n, n)
        N2 = np.repeat(np.arange(n).reshape(1, n), n).reshape(n, n).T
        K = np.power(r, np.abs(N1 - N2))
        A = np.random.multivariate_normal(M, K, size=m)
    elif matrix == "dct":
        A = dct(np.eye(np.maximum(m, n)))
        A = A[np.random.permutation(m), :]
        A = A[:, :n]
    elif matrix == "toeplitz":
        ranget = np.linspace(-10, 10, m)
        offset = 3.0
        rangev = np.linspace(-10 + offset, 10 - offset, n)
        A = np.zeros((m, n))
        for j in range(n):
            A[:, j] = np.exp(-0.5 * ((ranget - rangev[j]) ** 2))
    else:
        raise ValueError(f"Unsupported matrix type {matrix}")
    if normalize:
        A /= np.linalg.norm(A, axis=0, ord=2)
    return A


def synthetic_y(model, x, A, m, s):
    """Generate an output y ~ model(Ax,e) where A is an (m, n) matrix, x is a
    k-sparse vector and e is a noise that depends on the model considered."""
    if model == "linear":
        y = A @ x
        e = np.random.randn(m)
        e *= np.sqrt((y @ y) / (s * (e @ e)))
        y += e
    elif model == "logistic":
        p = 1.0 / (1.0 + np.exp(-s * (A @ x)))
        y = 2.0 * np.random.binomial(1, p, size=m) - 1.0
    elif model == "svm":
        p = 1.0 / (1.0 + np.exp(-s * (A @ x)))
        y = 2.0 * (p > 0.5) - 1.0
    elif model == "poisson":
        y = np.random.poisson(-s * (A @ x), m)
    elif model == "random":
        y = np.random.normal(0.0, s, size=m)
    else:
        raise ValueError(f"Unsupported model {model}")
    return y


def get_data_synthetic(matrix, model, k, m, n, s, normalize=False):
    """Generate synthetic data for sparse problems."""
    x = synthetic_x(k, n)
    A = synthetic_A(matrix, m, n, normalize)
    if model == "poisson":
        A = np.abs(A)
    y = synthetic_y(model, x, A, m, s)
    if model == "random":
        x = None
    return A, y, x


def get_data_libsvm(dataset_name):
    """Extract a dataset from LIBSVM, see the `libsvm` package for more
    details."""
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    A, y = fetch_libsvm(dataset_name)
    return A, y, None


def get_data_openml(dataset_id, dataset_target):
    """Extract a dataset from OpenML, see the `openml` package for more
    details."""
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
    """Extract a dataset from the UCI ML repository, see the `ucimlrepo`
    package for more details."""
    dataset = fetch_ucirepo(id=dataset_id)
    A = dataset.data.features
    y = dataset.data.targets
    A = A.to_numpy().astype(float)
    y = y.to_numpy().flatten().astype(float)
    return A, y, None


def get_data_hardcoded(dataset_name):
    """Extract a dataset from the folder experiments/datasets/."""
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


def process_data(
    datafit_name, penalty_name, A, y, x_true, center=False, normalize=False
):
    """Process the problem data."""
    if sparse.issparse(A):
        A = A.todense()
    if not A.flags["F_CONTIGUOUS"] or not A.flags["OWNDATA"]:
        A = np.array(A, order="F")
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
    if dataset["dataset_type"] == "synthetic":
        A, y, x_true = get_data_synthetic(**dataset["dataset_opts"])
    elif dataset["dataset_type"] == "libsvm":
        A, y, x_true = get_data_libsvm(**dataset["dataset_opts"])
    elif dataset["dataset_type"] == "openml":
        A, y, x_true = get_data_openml(**dataset["dataset_opts"])
    elif dataset["dataset_type"] == "uciml":
        A, y, x_true = get_data_uciml(**dataset["dataset_opts"])
    elif dataset["dataset_type"] == "hardcoded":
        A, y, x_true = get_data_hardcoded(**dataset["dataset_opts"])
    else:
        raise ValueError(f"Unsupported dataset type {dataset['dataset_type']}")
    A, y, x_true = process_data(
        dataset["datafit_name"],
        dataset["penalty_name"],
        A,
        y,
        x_true,
        **dataset["process_opts"],
    )
    return A, y, x_true


def calibrate_parameters(datafit_name, penalty_name, A, y, x_true=None):
    """Give some data (A,y), datafit and penalty, use `l0learn` to find an
    appropriate the L0-norm regularization weight and suitable hyperparameters
    in the penalty function."""

    # Binding for datafit and penalty names between El0ps and L0learn
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

    # Datafit instanciation
    datafit = eval(datafit_name)(y)

    # Fit an approximate regularization path with L0Learn
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

    # Penalty and L0-norm parameters calibration from L0learn path. Select the
    # hyperparameters with the best cross-validation score among those with the
    # best support recovery F1 score.
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
            if (f1 > best_f1) or (x_true is None):
                if cv < best_cv:
                    best_M = 1.5 * np.max(np.abs(x))
                    best_lmbda = lmbda / m
                    best_gamma = gamma / m
                    best_cv = cv
                    best_f1 = f1
                    best_x = np.copy(x)

    # Penalty instanciation
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
