import re
import sys
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model._base import LinearModel, RegressorMixin
from skglm import Lasso, ElasticNet, MCPRegression
from skglm.datafits import Quadratic
from skglm.estimators import _glm_fit
from skglm.penalties import L0_5, SCAD
from skglm.utils.jit_compilation import compiled_clone
from skglm.solvers import AndersonCD
from typing import Union, get_type_hints
from numpy.typing import NDArray
from l0bnb import BNBTree
from el0ps.datafits import BaseDatafit
from el0ps.penalties import BasePenalty
from el0ps.solvers import (
    BaseSolver,
    BnbSolver,
    BnbOptions,
    MipOptions,
    MipSolver,
    Status,
    Result,
    BnbBranchingStrategy,
    BnbExplorationStrategy,
)


class L0bnbSolver(BaseSolver):
    """L0bnb solver for L0-penalized problems."""

    def __init__(
        self,
        time_limit: float = float(sys.maxsize),
        rel_tol: float = 1e-4,
        int_tol: float = 1e-8,
        verbose: bool = False,
    ):
        self.options = {
            "time_limit": time_limit,
            "rel_tol": rel_tol,
            "int_tol": int_tol,
            "verbose": verbose,
        }

    def __str__(self):
        return "L0bnbSolver"

    def solve(
        self,
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: NDArray,
        lmbd: float,
        x_init: Union[NDArray, None] = None,
    ) -> Result:
        if str(datafit) != "Leastsquares":
            raise NotImplementedError(
                "`L0bnbSolver` does not support `{}` yet.".format(str(datafit))
            )

        m, _ = A.shape
        if str(penalty) == "Bigm":
            l0 = lmbd
            l2 = 0.0
            M = penalty.M
        elif str(penalty) == "L2norm":
            l0 = lmbd
            l2 = penalty.alpha
            M = np.inf
        elif str(penalty) == "BigmL2norm":
            l0 = lmbd
            l2 = penalty.alpha
            M = penalty.M
        else:
            raise NotImplementedError(
                "`L0bnbSolver` does not support `{}` yet.".format(str(penalty))
            )

        solver = BNBTree(
            A,
            datafit.y,
            self.options["int_tol"],
            self.options["rel_tol"],
        )
        result = solver.solve(
            l0,
            l2,
            M,
            gap_tol=self.options["rel_tol"],
            warm_start=x_init,
            verbose=self.options["verbose"],
            time_limit=self.options["time_limit"],
        )

        if result.sol_time < self.options["time_limit"]:
            status = Status.OPTIMAL
        else:
            status = Status.TIME_LIMIT

        self.x = np.array(result.beta)
        self.z = np.array(result.beta != 0.0, dtype=float)

        objective_value = (
            datafit.value(A @ self.x)
            + lmbd * np.linalg.norm(self.x, 0)
            + penalty.value(self.x)
        )

        return Result(
            status,
            result.sol_time,
            solver.number_of_nodes,
            np.abs(result.gap) if not np.isnan(result.gap) else 0.0,
            self.x,
            objective_value,
            np.sum(np.abs(self.x) > self.options["int_tol"]),
            None,
        )


class OmpPath:

    def __init__(self, max_nnz=10) -> None:
        self.max_nnz = max_nnz

    def __str__(self) -> str:
        return "OmpPath"

    def fit(self, datafit, A):
        assert str(datafit) == "Leastsquares"

        fit_data = {
            "status": [],
            "solve_time": [],
            "x": [],
            "datafit_value": [],
            "n_nnz": [],
        }

        start_time = time.time()

        n = A.shape[1]
        y = datafit.y
        s = []
        r = y
        for _ in range(self.max_nnz):
            u = np.dot(A.T, r)
            i = np.argmax(np.abs(u))
            s.append(i)
            x = np.zeros(n)
            x[s] = np.linalg.lstsq(A[:, s], y, rcond=None)[0]
            w = A[:, s] @ x[s]
            r = y - w

            fit_data["status"].append(Status.OPTIMAL)
            fit_data["solve_time"].append(time.time() - start_time)
            fit_data["x"].append(x)
            fit_data["datafit_value"].append(datafit.value(w))
            fit_data["n_nnz"].append(len(s))

        return fit_data


class LassoPath:

    def __init__(
        self,
        lmbd_max=1.0,
        lmbd_min=1e-3,
        lmbd_num=31,
        lmbd_scaled=False,
        max_nnz=10,
        stop_if_not_optimal=True,
    ) -> None:
        self.lmbd_max = lmbd_max
        self.lmbd_min = lmbd_min
        self.lmbd_num = lmbd_num
        self.lmbd_scaled = lmbd_scaled
        self.max_nnz = max_nnz
        self.stop_if_not_optimal = stop_if_not_optimal

    def __str__(self) -> str:
        return "LassoPath"

    def fit(self, datafit, A):
        assert str(datafit) == "Leastsquares"

        fit_data = {
            "status": [],
            "solve_time": [],
            "x": [],
            "datafit_value": [],
            "n_nnz": [],
        }

        y = datafit.y

        lmbd_grid = np.logspace(
            np.log10(self.lmbd_max),
            np.log10(self.lmbd_min),
            self.lmbd_num,
        )
        if self.lmbd_scaled:
            lmbd_grid *= np.linalg.norm(A.T @ y, np.inf)

        start_time = time.time()

        for lmbd in lmbd_grid:
            est = Lasso(alpha=lmbd, max_iter=int(1e5), fit_intercept=False)
            est.fit(A, y)
            x = est.coef_
            w = A @ x
            s = np.where(x != 0)[0]

            if len(s) > self.max_nnz:
                break

            fit_data["status"].append(Status.OPTIMAL)
            fit_data["solve_time"].append(time.time() - start_time)
            fit_data["x"].append(np.copy(x))
            fit_data["datafit_value"].append(datafit.value(w))
            fit_data["n_nnz"].append(len(s))

        return fit_data


class EnetPath:

    def __init__(
        self,
        lmbd_max=1.0,
        lmbd_min=1e-3,
        lmbd_num=31,
        lmbd_scaled=False,
        max_nnz=10,
        stop_if_not_optimal=True,
    ) -> None:
        self.lmbd_max = lmbd_max
        self.lmbd_min = lmbd_min
        self.lmbd_num = lmbd_num
        self.lmbd_scaled = lmbd_scaled
        self.max_nnz = max_nnz
        self.stop_if_not_optimal = stop_if_not_optimal

    def __str__(self) -> str:
        return "EnetPath"

    def fit(self, datafit, A):
        assert str(datafit) == "Leastsquares"

        fit_data = {
            "status": [],
            "solve_time": [],
            "x": [],
            "datafit_value": [],
            "n_nnz": [],
        }

        y = datafit.y

        lmbd_grid = np.logspace(
            np.log10(self.lmbd_max),
            np.log10(self.lmbd_min),
            self.lmbd_num,
        )
        lmbd_max = np.linalg.norm(A.T @ y, np.inf)
        if self.lmbd_scaled:
            lmbd_grid *= lmbd_max

        # Calibrate L1 ratio
        param_grid = {
            "alpha": lmbd_max * np.logspace(-2, 1, 4),
            "l1_ratio": np.linspace(0.1, 0.9, 9),
        }
        grid_search = GridSearchCV(
            estimator=ElasticNet(), param_grid=param_grid, cv=5
        )
        grid_search.fit(A, y)
        l1_ratio = grid_search.best_estimator_.l1_ratio

        start_time = time.time()

        for lmbd in lmbd_grid:
            est = ElasticNet(
                alpha=lmbd,
                l1_ratio=l1_ratio,
                max_iter=int(1e5),
                fit_intercept=False,
            )
            est.fit(A, y)
            x = est.coef_
            w = A @ x
            s = np.where(x != 0)[0]

            if len(s) > self.max_nnz:
                break

            fit_data["status"].append(Status.OPTIMAL)
            fit_data["solve_time"].append(time.time() - start_time)
            fit_data["x"].append(np.copy(x))
            fit_data["datafit_value"].append(datafit.value(w))
            fit_data["n_nnz"].append(len(s))

        return fit_data


class L05Regression(LinearModel, RegressorMixin):

    def __init__(
        self,
        alpha=1.0,
        gamma=3,
        weights=None,
        max_iter=50,
        max_epochs=50_000,
        p0=10,
        verbose=0,
        tol=1e-4,
        positive=False,
        fit_intercept=True,
        warm_start=False,
        ws_strategy="subdiff",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.verbose = verbose
        self.tol = tol
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.ws_strategy = ws_strategy

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        penalty = compiled_clone(L0_5(self.alpha))
        datafit = compiled_clone(Quadratic(), to_float32=X.dtype == np.float32)
        solver = AndersonCD(
            self.max_iter,
            self.max_epochs,
            self.p0,
            tol=self.tol,
            ws_strategy=self.ws_strategy,
            fit_intercept=self.fit_intercept,
            warm_start=self.warm_start,
            verbose=self.verbose,
        )
        return solver.path(
            X, y, datafit, penalty, alphas, coef_init, return_n_iter
        )

    def fit(self, X, y):
        penalty = L0_5(self.alpha)
        solver = AndersonCD(
            self.max_iter,
            self.max_epochs,
            self.p0,
            tol=self.tol,
            ws_strategy=self.ws_strategy,
            fit_intercept=self.fit_intercept,
            warm_start=self.warm_start,
            verbose=self.verbose,
        )
        return _glm_fit(X, y, self, Quadratic(), penalty, solver)


class L05Path:

    def __init__(
        self,
        lmbd_max=1.0,
        lmbd_min=1e-3,
        lmbd_num=31,
        lmbd_scaled=False,
        max_nnz=10,
        stop_if_not_optimal=True,
    ) -> None:
        self.lmbd_max = lmbd_max
        self.lmbd_min = lmbd_min
        self.lmbd_num = lmbd_num
        self.lmbd_scaled = lmbd_scaled
        self.max_nnz = max_nnz
        self.stop_if_not_optimal = stop_if_not_optimal

    def __str__(self) -> str:
        return "L05Path"

    def fit(self, datafit, A):
        assert str(datafit) == "Leastsquares"

        fit_data = {
            "status": [],
            "solve_time": [],
            "x": [],
            "datafit_value": [],
            "n_nnz": [],
        }

        y = datafit.y

        lmbd_grid = np.logspace(
            np.log10(self.lmbd_max),
            np.log10(self.lmbd_min),
            self.lmbd_num,
        )
        if self.lmbd_scaled:
            lmbd_grid *= np.linalg.norm(A.T @ y, np.inf)

        start_time = time.time()

        for lmbd in lmbd_grid:
            est = L05Regression(
                alpha=lmbd,
                max_iter=int(1e5),
                fit_intercept=False,
            )
            est.fit(A, y)
            x = est.coef_
            w = A @ x
            s = np.where(x != 0)[0]

            if len(s) > self.max_nnz:
                break

            fit_data["status"].append(Status.OPTIMAL)
            fit_data["solve_time"].append(time.time() - start_time)
            fit_data["x"].append(np.copy(x))
            fit_data["datafit_value"].append(datafit.value(w))
            fit_data["n_nnz"].append(len(s))

        return fit_data


class McpPath:

    def __init__(
        self,
        lmbd_max=1.0,
        lmbd_min=1e-3,
        lmbd_num=31,
        lmbd_scaled=False,
        max_nnz=10,
        stop_if_not_optimal=True,
    ) -> None:
        self.lmbd_max = lmbd_max
        self.lmbd_min = lmbd_min
        self.lmbd_num = lmbd_num
        self.lmbd_scaled = lmbd_scaled
        self.max_nnz = max_nnz
        self.stop_if_not_optimal = stop_if_not_optimal

    def __str__(self) -> str:
        return "McpPath"

    def fit(self, datafit, A):
        assert str(datafit) == "Leastsquares"

        fit_data = {
            "status": [],
            "solve_time": [],
            "x": [],
            "datafit_value": [],
            "n_nnz": [],
        }

        y = datafit.y

        lmbd_grid = np.logspace(
            np.log10(self.lmbd_max),
            np.log10(self.lmbd_min),
            self.lmbd_num,
        )
        lmbd_max = np.linalg.norm(A.T @ y, np.inf)
        if self.lmbd_scaled:
            lmbd_grid *= lmbd_max

        # Calibrate MCP ratio
        param_grid = {
            "alpha": lmbd_max * np.logspace(-2, 1, 4),
            "gamma": np.logspace(-2, 1, 4),
        }
        grid_search = GridSearchCV(
            estimator=MCPRegression(), param_grid=param_grid, cv=5
        )
        grid_search.fit(A, y)
        gamma = grid_search.best_estimator_.gamma

        start_time = time.time()

        for lmbd in lmbd_grid:
            est = MCPRegression(
                alpha=lmbd,
                gamma=gamma,
                max_iter=int(1e5),
                fit_intercept=False,
            )
            est.fit(A, y)
            x = est.coef_
            w = A @ x
            s = np.where(x != 0)[0]

            if len(s) > self.max_nnz:
                break

            fit_data["status"].append(Status.OPTIMAL)
            fit_data["solve_time"].append(time.time() - start_time)
            fit_data["x"].append(np.copy(x))
            fit_data["datafit_value"].append(datafit.value(w))
            fit_data["n_nnz"].append(len(s))

        return fit_data


class SCADRegression(LinearModel, RegressorMixin):

    def __init__(
        self,
        alpha=1.0,
        gamma=3,
        weights=None,
        max_iter=50,
        max_epochs=50_000,
        p0=10,
        verbose=0,
        tol=1e-4,
        positive=False,
        fit_intercept=True,
        warm_start=False,
        ws_strategy="subdiff",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.verbose = verbose
        self.tol = tol
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.ws_strategy = ws_strategy

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        penalty = compiled_clone(SCAD(self.alpha, self.gamma, self.positive))
        datafit = compiled_clone(Quadratic(), to_float32=X.dtype == np.float32)
        solver = AndersonCD(
            self.max_iter,
            self.max_epochs,
            self.p0,
            tol=self.tol,
            ws_strategy=self.ws_strategy,
            fit_intercept=self.fit_intercept,
            warm_start=self.warm_start,
            verbose=self.verbose,
        )
        return solver.path(
            X, y, datafit, penalty, alphas, coef_init, return_n_iter
        )

    def fit(self, X, y):
        penalty = SCAD(self.alpha, self.gamma)
        solver = AndersonCD(
            self.max_iter,
            self.max_epochs,
            self.p0,
            tol=self.tol,
            ws_strategy=self.ws_strategy,
            fit_intercept=self.fit_intercept,
            warm_start=self.warm_start,
            verbose=self.verbose,
        )
        return _glm_fit(X, y, self, Quadratic(), penalty, solver)


class ScadPath:

    def __init__(
        self,
        lmbd_max=1.0,
        lmbd_min=1e-3,
        lmbd_num=31,
        lmbd_scaled=False,
        max_nnz=10,
        stop_if_not_optimal=True,
    ) -> None:
        self.lmbd_max = lmbd_max
        self.lmbd_min = lmbd_min
        self.lmbd_num = lmbd_num
        self.lmbd_scaled = lmbd_scaled
        self.max_nnz = max_nnz
        self.stop_if_not_optimal = stop_if_not_optimal

    def __str__(self) -> str:
        return "ScadPath"

    def fit(self, datafit, A):
        assert str(datafit) == "Leastsquares"

        fit_data = {
            "status": [],
            "solve_time": [],
            "x": [],
            "datafit_value": [],
            "n_nnz": [],
        }

        y = datafit.y

        lmbd_grid = np.logspace(
            np.log10(self.lmbd_max),
            np.log10(self.lmbd_min),
            self.lmbd_num,
        )
        lmbd_max = np.linalg.norm(A.T @ y, np.inf)
        if self.lmbd_scaled:
            lmbd_grid *= lmbd_max

        param_grid = {
            "alpha": lmbd_max * np.logspace(-2, 1, 4),
            "gamma": np.logspace(-2, 1, 4),
        }
        grid_search = GridSearchCV(
            estimator=SCADRegression(), param_grid=param_grid, cv=5
        )
        grid_search.fit(A, y)
        gamma = grid_search.best_estimator_.gamma

        start_time = time.time()

        for lmbd in lmbd_grid:
            est = SCADRegression(
                alpha=lmbd,
                gamma=gamma,
                max_iter=int(1e5),
                fit_intercept=False,
            )
            est.fit(A, y)
            x = est.coef_
            w = A @ x
            s = np.where(x != 0)[0]

            if len(s) > self.max_nnz:
                break

            fit_data["status"].append(Status.OPTIMAL)
            fit_data["solve_time"].append(time.time() - start_time)
            fit_data["x"].append(np.copy(x))
            fit_data["datafit_value"].append(datafit.value(w))
            fit_data["n_nnz"].append(len(s))

        return fit_data


def extract_bnb_options(solver_name):
    option_types = get_type_hints(BnbOptions)
    pattern = r"\[([^]]*)\]"
    match = re.search(pattern, solver_name)
    if match:
        options_str = match.group(1)
        if options_str:
            option_pairs = options_str.split(",")
            options_dict = {}
            for pair in option_pairs:
                k, v = pair.split("=")
                if k == "exploration_strategy":
                    options_dict[k] = BnbExplorationStrategy(v)
                elif k == "branching_strategy":
                    options_dict[k] = BnbBranchingStrategy(v)
                elif option_types[k] in [str, int, float]:
                    options_dict[k] = option_types[k](v)
                elif option_types[k] == bool:
                    options_dict[k] = v in ["true", "True"]
            return options_dict
    return {}


def extract_mip_options(solver_name):
    option_types = get_type_hints(MipOptions)
    pattern = r"\[([^]]*)\]"
    match = re.search(pattern, solver_name)
    if match:
        options_str = match.group(1)
        if options_str:
            option_pairs = options_str.split(",")
            options_dict = {}
            for pair in option_pairs:
                k, v = pair.split("=")
                if option_types[k] in [str, int, float]:
                    options_dict[k] = option_types[k](v)
                elif option_types[k] == bool:
                    options_dict[k] = v in ["true", "True"]
            return options_dict
    return {}


def get_solver(solver_name, options={}):
    if solver_name.startswith("el0ps"):
        return BnbSolver(**{**options, **extract_bnb_options(solver_name)})
    elif solver_name.startswith("mip"):
        return MipSolver(**{**options, **extract_mip_options(solver_name)})
    elif solver_name == "l0bnb":
        return L0bnbSolver(**options)
    else:
        raise ValueError("Unknown solver name {}".format(solver_name))


def get_relaxed_path(solver_name, path_opts={}):
    if solver_name == "Omp":
        return OmpPath(max_nnz=path_opts["max_nnz"])
    elif solver_name == "Lasso":
        return LassoPath(
            lmbd_max=path_opts["lmbd_max"],
            lmbd_min=path_opts["lmbd_min"],
            lmbd_num=path_opts["lmbd_num"],
            lmbd_scaled=path_opts["lmbd_scaled"],
            max_nnz=path_opts["max_nnz"],
            stop_if_not_optimal=path_opts["stop_if_not_optimal"],
        )
    elif solver_name == "Enet":
        return EnetPath(
            lmbd_max=path_opts["lmbd_max"],
            lmbd_min=path_opts["lmbd_min"],
            lmbd_num=path_opts["lmbd_num"],
            lmbd_scaled=path_opts["lmbd_scaled"],
            max_nnz=path_opts["max_nnz"],
            stop_if_not_optimal=path_opts["stop_if_not_optimal"],
        )
    elif solver_name == "L05":
        return L05Path(
            lmbd_max=path_opts["lmbd_max"],
            lmbd_min=path_opts["lmbd_min"],
            lmbd_num=path_opts["lmbd_num"],
            lmbd_scaled=path_opts["lmbd_scaled"],
            max_nnz=path_opts["max_nnz"],
            stop_if_not_optimal=path_opts["stop_if_not_optimal"],
        )
    elif solver_name == "Mcp":
        return McpPath(
            lmbd_max=path_opts["lmbd_max"],
            lmbd_min=path_opts["lmbd_min"],
            lmbd_num=path_opts["lmbd_num"],
            lmbd_scaled=path_opts["lmbd_scaled"],
            max_nnz=path_opts["max_nnz"],
            stop_if_not_optimal=path_opts["stop_if_not_optimal"],
        )
    elif solver_name == "Scad":
        return ScadPath(
            lmbd_max=path_opts["lmbd_max"],
            lmbd_min=path_opts["lmbd_min"],
            lmbd_num=path_opts["lmbd_num"],
            lmbd_scaled=path_opts["lmbd_scaled"],
            max_nnz=path_opts["max_nnz"],
            stop_if_not_optimal=path_opts["stop_if_not_optimal"],
        )
    else:
        raise ValueError("Unknown solver name: {}".format(solver_name))


def can_handle_instance(solver_name, datafit_name, penalty_name):
    if solver_name.startswith("el0ps"):
        handle_datafit = True
        handle_penalty = True
    elif solver_name.startswith("mip"):
        solver_opts = extract_mip_options(solver_name)
        if solver_opts["optimizer_name"] == "cplex":
            handle_datafit = datafit_name in [
                "Leastsquares",
                "Squaredhinge",
            ]
            handle_penalty = penalty_name in [
                "Bigm",
                "BigmL1norm",
                "BigmL2norm",
                "L2norm",
                "L1L2norm",
            ]
        elif solver_opts["optimizer_name"] == "gurobi":
            handle_datafit = datafit_name in [
                "Leastsquares",
                "Squaredhinge",
            ]
            handle_penalty = penalty_name in [
                "Bigm",
                "BigmL1norm",
                "BigmL2norm",
                "L2norm",
                "L1L2norm",
                "NeglogTriangular",
            ]
        elif solver_opts["optimizer_name"] == "mosek":
            handle_datafit = datafit_name in [
                "Leastsquares",
                "Logistic",
                "Squaredhinge",
            ]
            handle_penalty = penalty_name in [
                "Bigm",
                "BigmL1norm",
                "BigmL2norm",
                "L2norm",
                "L1L2norm",
                "NeglogTriangular",
            ]
        else:
            raise ValueError(
                "Unknown mip optimizer {}".format(
                    solver_opts["optimizer_name"]
                )
            )
    elif solver_name == "l0bnb":
        handle_datafit = datafit_name in ["Leastsquares"]
        handle_penalty = penalty_name in ["Bigm", "BigmL2norm", "L2norm"]
    else:
        raise ValueError("Unknown solver name {}".format(solver_name))
    return handle_datafit and handle_penalty


def can_handle_compilation(solver_name):
    return solver_name.startswith("el0ps")
