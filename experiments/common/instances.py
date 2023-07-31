import warnings
import cvxpy as cp
import numpy as np
import l0learn
from el0ps.datafit import Leastsquares, Logistic
from el0ps.penalty import Bigm, BigmL1norm, BigmL2norm, L1norm, L2norm

def f1_score(x_true, x):
    s = x != 0.
    s_true = x_true != 0.
    i = np.sum(s & s_true)
    p = 1.0 if not np.any(s) else i / np.sum(s)
    r = 1.0 if not np.any(s_true) else i / np.sum(s_true)
    f = 0.0 if (p + r) == 0.0 else 2.0 * p * r / (p + r)
    return f

def synthetic_x(k, n):
    x = np.zeros(n)
    s = np.array(np.floor(np.linspace(0, n - 1, num=k)), dtype=int)
    x[s] = np.random.randn(k)
    return x

def synthetic_A(m, n, rho, normalize):
    M = np.zeros(n)
    N1 = np.repeat(np.arange(n).reshape(n, 1), n).reshape(n, n)
    N2 = np.repeat(np.arange(n).reshape(1, n), n).reshape(n, n).T
    K = np.power(rho, np.abs(N1 - N2))
    V = np.random.standard_normal(m * n).reshape(m, n)
    W = M + V @ np.linalg.cholesky(K).T
    A = np.reshape(W, (n, m)).T
    if normalize:
        A /= np.linalg.norm(A, axis=0, ord=2)
    return A

def synthetic_y(datafit_name, x, A, m, snr):
    if datafit_name == "Leastsquares":
        y = A @ x
        e = np.random.normal(0, np.std(y) / snr, size=m)
        y += e
    elif datafit_name == "Logistic":
        p = 1.0 / (1.0 + np.exp(-snr * (A @ x)))
        y = 2.0 * np.random.binomial(1, p, size=m) - 1.0
    else:
        raise ValueError(f"Unsupported data-fidelity function {datafit_name}")
    return y

def calibrate_objective(datafit_name, penalty_name, A, y, x_true=None):
    
    bindings = {
        'Leastsquares': 'SquaredError',
        'Logistic': 'Logistic',
        'Bigm': 'L0',
        'BigmL1norm': 'L0L1',
        'BigmL2norm': 'L0L2',
        'L1norm': 'L0L1',
        'L2norm': 'L0L2',
    }

    assert datafit_name in bindings.keys()
    assert penalty_name in bindings.keys()

    m, n = A.shape

    # Datafit
    if datafit_name == "Leastsquares":
        datafit = Leastsquares(y)
    elif datafit_name == "Logistic":
        datafit = Logistic(y)

    # Calibrate the penalty parameters w.r.t x_truth when it is known
    gamma_true = None
    # if penalty_name != "Bigm" and x_true is not None:
    #     s_true = x_true != 0.
    #     A_true = A[:, s_true]
    #     cp_x = cp.Variable(np.sum(s_true))
    #     cp_gamma = cp.Parameter(nonneg=True)
    #     if datafit_name == "Leastsquares":
    #         cp_datafit = cp.sum_squares(y - A_true @ cp_x) / m
    #     elif datafit_name == "Logistic":
    #         cp_datafit = cp.sum(cp.logistic(-y * (A_true @ cp_x))) / m
    #     if penalty_name in ["BigmL1norm", "L1norm"]:
    #         cp_penalty = cp.norm(cp_x, 1)
    #     elif penalty_name in ["BigmL2norm", "L2norm"]:
    #         cp_penalty = cp.sum_squares(cp_x)
    #     cp_objective = cp.Minimize(cp_datafit + cp_gamma * cp_penalty)
    #     cp_problem = cp.Problem(cp_objective)
    #     gamma_grid = m * np.logspace(-4, 1, 50)
    #     best_norm = np.inf
    #     for gamma_val in gamma_grid:
    #         cp_gamma.value = gamma_val
    #         cp_problem.solve()
    #         test_norm = np.linalg.norm(cp_x.value - x_true[s_true], 2) ** 2
    #         if test_norm < best_norm:
    #             best_norm = test_norm
    #             gamma_true = gamma_val

    # Fit regularization path with L0Learn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cvfit = l0learn.cvfit(
            A, 
            y, 
            bindings[datafit_name], 
            bindings[penalty_name], 
            intercept = False, 
            num_gamma = 1 if penalty_name == "Bigm" or gamma_true else 10,
            gamma_max = m * gamma_true if gamma_true else m * 10,
            gamma_min = m * gamma_true if gamma_true else m * 0.0001,
        )

    # Penalty and L0-norm parameters calibration from L0learn path
    best_x = None
    best_M = None
    best_lmbda = None
    best_gamma = None
    best_cv = np.inf
    best_f1 = 0.
    for i, gamma in enumerate(cvfit.gamma):
        for j, lmbda in enumerate(cvfit.lambda_0[i]):
            x = np.array(cvfit.coeff(lmbda, gamma).todense()).reshape(n+1)[1:]
            cv = cvfit.cv_means[i][j]
            f1 = 0. if x_true is None else f1_score(x_true, x)
            if f1 >= best_f1 and cv <= best_cv:
                best_x = np.copy(x)
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

def synthetic_data(datafit_name, penalty_name, k, m, n, rho, snr, normalize):
    x_true = synthetic_x(k, n)
    A = synthetic_A(m, n, rho, normalize)
    y = synthetic_y(datafit_name, x_true, A, m, snr)
    datafit, penalty, lmbd = calibrate_objective(datafit_name, penalty_name, A, y, x_true)
    return datafit, penalty, A, lmbd, x_true
