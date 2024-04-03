import pathlib
import pickle
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from abc import abstractmethod
from copy import deepcopy
from datetime import datetime
from el0ps.path import Path
from el0ps.solver import Status
from el0ps.utils import compiled_clone, compute_lmbd_max
from sklearn.model_selection import train_test_split

sys.path.append(pathlib.Path(__file__).parent.parent.absolute())
from experiments.instances import (
    calibrate_parameters,
    f1_score,
    get_data,
)  # noqa
from experiments.solvers import (  # noqa
    can_handle_compilation,
    can_handle_instance,
    get_solver,
    get_relaxed_path,
)


class Experiment:
    name = "experiment"
    results_dir = pathlib.Path(__file__).parent.absolute().joinpath("results")
    saves_dir = pathlib.Path(__file__).parent.absolute().joinpath("saves")

    def __init__(self, config_path):
        self.config_path = config_path
        self.results = None

    def setup(self):
        print("Loading config...")
        with open(pathlib.Path(self.config_path), "r") as file:
            self.config = yaml.load(file, Loader=yaml.Loader)
        assert self.config["expname"] == self.name

    def load_problem(self):
        print("Loading data...")
        A, y, x_true = get_data(self.config["dataset"])
        print("  A shape: {}".format(A.shape))
        print("  y shape: {}".format(y.shape))
        print("  x shape: {}".format(None if x_true is None else x_true.shape))

        print("Calibrating parameters...")
        datafit, penalty, lmbd, x_cal = calibrate_parameters(
            self.config["dataset"]["datafit_name"],
            self.config["dataset"]["penalty_name"],
            A,
            y,
            x_true,
        )
        lmbd_max = compute_lmbd_max(datafit, penalty, A)
        print("  num nz: {}".format(sum(x_cal != 0.0)))
        print("  lratio: {}".format(lmbd / lmbd_max))
        for param_name, param_value in penalty.params_to_dict().items():
            print("  {}\t: {}".format(param_name, param_value))
        self.x_true = x_true
        self.datafit = datafit
        self.penalty = penalty
        self.A = A
        self.lmbd = lmbd

    def precompile(self):
        print("Precompiling...")
        self.compiled_datafit = compiled_clone(self.datafit)
        self.compiled_penalty = compiled_clone(self.penalty)
        solver_opts = deepcopy(self.config["solvers"]["solvers_opts"])
        solver_opts["time_limit"] = 5.0
        solver = get_solver("el0ps", solver_opts)
        solver.solve(
            self.compiled_datafit, self.compiled_penalty, self.A, self.lmbd
        )

    @abstractmethod
    def run(self):
        pass

    def save_results(self):
        print("Saving results...")
        result_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        result_file = "{}_{}.pickle".format(self.name, result_uuid)
        result_path = pathlib.Path(self.results_dir, result_file)
        with open(result_path, "wb") as file:
            data = {"config": self.config, "results": self.results}
            pickle.dump(data, file)
        print("  File name: {}".format(result_file))

    @abstractmethod
    def load_results(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def save_plot(self):
        pass


class Perfprofile(Experiment):
    name = "perfprofile"

    def run(self):
        results = {}
        for solver_name in self.config["solvers"]["solvers_name"]:
            if can_handle_instance(
                solver_name,
                self.config["dataset"]["datafit_name"],
                self.config["dataset"]["penalty_name"],
            ):
                print("Running {}...".format(solver_name))
                solver_opts = self.config["solvers"]["solvers_opts"]
                solver = get_solver(solver_name, solver_opts)
                if can_handle_compilation(solver_name):
                    result = solver.solve(
                        self.compiled_datafit,
                        self.compiled_penalty,
                        self.A,
                        self.lmbd,
                    )
                else:
                    result = solver.solve(
                        self.datafit, self.penalty, self.A, self.lmbd
                    )
            else:
                print("Skipping {}".format(solver_name))
                result = None
            results[solver_name] = result
            if result is not None:
                print(result)
        self.results = results

    def load_results(self):
        print("Loading results...")
        found, match, notcv = 0, 0, 0
        times = {s: [] for s in self.config["solvers"]["solvers_name"]}
        for result_path in self.results_dir.glob(self.name + "_*.pickle"):
            found += 1
            with open(result_path, "rb") as file:
                file_data = pickle.load(file)
                if file_data["config"] == self.config:
                    match += 1
                    for solver_name, result in file_data["results"].items():
                        if result.status == Status.OPTIMAL:
                            times[solver_name].append(result.solve_time)
                        else:
                            notcv += 1
        print("  {} files found".format(found))
        print("  {} files matched".format(match))
        print("  {} files not converged".format(notcv))

        if match == 0:
            self.grid_times = None
            self.curve_times = None
            return

        print("Computing statistics...")
        for solver_name, solver_times in times.items():
            print("  {}".format(solver_name))
            print("     num : {}".format(len(solver_times)))
            print("     mean: {}".format(np.mean(solver_times)))
            print("     std : {}".format(np.std(solver_times)))

        min_times = np.min([np.min(v) for v in times.values()])
        max_times = np.max([np.max(v) for v in times.values()])
        self.grid_times = np.logspace(
            np.floor(np.log10(min_times)),
            np.ceil(np.log10(max_times)),
            100,
        )
        self.curve_times = {
            solver_name: [np.sum(stats <= g) for g in self.grid_times]
            for solver_name, stats in times.items()
        }

    def plot(self):
        if self.curve_times is None or self.grid_times is None:
            return
        _, axs = plt.subplots()
        for solver_name in self.config["solvers"]["solvers_name"]:
            axs.plot(
                self.grid_times,
                self.curve_times[solver_name],
                label=solver_name,
            )
        axs.grid(visible=True, which="major", axis="both")
        axs.grid(visible=True, which="minor", axis="both", alpha=0.2)
        axs.minorticks_on()
        axs.set_xscale("log")
        axs.set_xlabel("Time budget")
        axs.set_ylabel("Inst. solved")
        plt.show()

    def save_plot(self):
        if self.curve_times is None or self.grid_times is None:
            return
        print("Saving data...")
        save_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        table = pd.DataFrame({"grid_times": self.grid_times})
        for solver_name in self.config["solvers"]["solvers_name"]:
            table[solver_name] = self.curve_times[solver_name]
        file_name = "{}_{}".format(self.name, save_uuid)
        save_path = self.saves_dir.joinpath("{}.csv".format(file_name))
        table.to_csv(save_path, index=False)


class Regpath(Experiment):
    name = "regpath"

    def run(self):
        results = {}
        for solver_name in self.config["solvers"]["solvers_name"]:
            if can_handle_instance(
                solver_name,
                self.config["dataset"]["datafit_name"],
                self.config["dataset"]["penalty_name"],
            ):
                solver_opts = self.config["solvers"]["solvers_opts"]
                solver = get_solver(solver_name, solver_opts)
                print("Running {}...".format(solver_name))
                path = Path(**self.config["path_opts"])
                if can_handle_compilation(solver_name):
                    result = path.fit(
                        solver,
                        self.compiled_datafit,
                        self.compiled_penalty,
                        self.A,
                    )
                else:
                    result = path.fit(
                        solver, self.datafit, self.penalty, self.A
                    )
                del result["x"]
            else:
                print("Skipping {}...".format(solver_name))
                result = None
            results[solver_name] = result
        self.results = results

    def load_results(self):
        print("Loading results...")
        self.lmbd_ratio_grid = np.logspace(
            np.log10(self.config["path_opts"]["lmbd_ratio_max"]),
            np.log10(self.config["path_opts"]["lmbd_ratio_min"]),
            self.config["path_opts"]["lmbd_ratio_num"],
        )
        self.stats_specs = {
            "solve_time": {"log": True},
            "objective_value": {"log": False},
            "datafit_value": {"log": False},
            "n_nnz": {"log": False},
        }
        stats = {
            stat_key: {
                solver_name: {i: [] for i in range(self.lmbd_ratio_grid.size)}
                for solver_name in self.config["solvers"]["solvers_name"]
            }
            for stat_key in self.stats_specs.keys()
        }

        found, match, empty, notcv = 0, 0, 0, 0
        for result_path in self.results_dir.glob(self.name + "_*.pickle"):
            found += 1
            with open(result_path, "rb") as file:
                file_data = pickle.load(file)
                if self.config == file_data["config"]:
                    match += 1
                    if not any(file_data["results"].values()):
                        empty += 1
                        continue
                    for solver_name, result in file_data["results"].items():
                        if result is not None:
                            for i in range(self.lmbd_ratio_grid.size):
                                if len(result["status"]) > i:
                                    if result["status"][i] == Status.OPTIMAL:
                                        for (
                                            stat_name,
                                            stat_values,
                                        ) in stats.items():
                                            stat_values[solver_name][i].append(
                                                result[stat_name][i]
                                            )
                                    else:
                                        notcv += 1

        print("  {} files found".format(found))
        print("  {} files matched".format(match))
        print("  {} empty results".format(empty))
        print("  {} not converged".format(notcv))

        if (match == 0) or (match == empty):
            return

        self.mean_stats = {
            stat_key: {
                solver_key: [
                    (
                        np.mean(solver_values[i])
                        if len(solver_values[i])
                        else np.nan
                    )
                    for i in range(self.lmbd_ratio_grid.size)
                ]
                for solver_key, solver_values in stat_values.items()
            }
            for stat_key, stat_values in stats.items()
        }

    def plot(self):
        _, axs = plt.subplots(1, len(self.mean_stats), squeeze=False)
        for i, (stat_name, stat_values) in enumerate(self.mean_stats.items()):
            for solver_name, solver_values in stat_values.items():
                axs[0, i].plot(
                    self.lmbd_ratio_grid,
                    solver_values,
                    label=solver_name,
                )
            axs[0, i].set_xscale("log")
            axs[0, i].grid(visible=True, which="major", axis="both")
            axs[0, i].grid(visible=True, which="minor", axis="both", alpha=0.2)
            axs[0, i].minorticks_on()
            axs[0, i].set_xlabel("lmbd/lmbd_max")
            axs[0, i].set_ylabel(stat_name)
            axs[0, i].invert_xaxis()
            if self.stats_specs[stat_name]["log"]:
                axs[0, i].set_yscale("log")
        axs[0, 0].legend()
        plt.show()

    def save_plot(self):
        print("Saving data...")
        save_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        save_file = "{}_{}.csv".format(self.name, save_uuid)
        save_path = self.saves_dir.joinpath(save_file)
        table = pd.DataFrame({"lmbd_ratio_grid": self.lmbd_ratio_grid})
        for stat_name, stat_values in self.mean_stats.items():
            for solver_name, solver_values in stat_values.items():
                table[solver_name + "_" + stat_name] = solver_values
        table.to_csv(save_path, index=False)


class Statistics(Experiment):
    name = "statistics"

    def run(self):

        # Train/test splitting
        print("Splitting train/test data...")
        A_train, A_test, y_train, y_test = train_test_split(
            self.A,
            self.datafit.y,
            test_size=self.config["dataset"]["test_size"],
        )

        print("Calibrating parameters on train/test data...")
        datafit_train, penalty_train, _, _ = calibrate_parameters(
            self.config["dataset"]["datafit_name"],
            self.config["dataset"]["penalty_name"],
            A_train,
            y_train,
            self.x_true,
        )
        datafit_test, penalty_test, _, _ = calibrate_parameters(
            self.config["dataset"]["datafit_name"],
            self.config["dataset"]["penalty_name"],
            A_test,
            y_test,
            self.x_true,
        )
        datafit_train_compiled = compiled_clone(datafit_train)
        penalty_train_compiled = compiled_clone(penalty_train)

        results = {}

        # Relax methods
        for solver_name in self.config["relaxed_solvers"]:
            print("Running {}...".format(solver_name))
            path = get_relaxed_path(solver_name, self.config["path_opts"])
            result = path.fit(datafit_train, A_train)
            result["f1_score"] = []
            result["train_error"] = []
            result["test_error"] = []
            for x_res in result["x"]:
                s = np.where(x_res != 0)[0]
                x = np.zeros(x_res.size)
                x[s] = np.linalg.lstsq(A_train[:, s], y_train, rcond=None)[0]
                w_train = A_train @ x
                w_test = A_test @ x
                result["f1_score"].append(f1_score(self.x_true, x))
                result["train_error"].append(datafit_train.value(w_train))
                result["test_error"].append(datafit_test.value(w_test))
            results[solver_name] = result

        # L0 methods
        for solver_name in self.config["solvers"]["solvers_name"]:
            if can_handle_instance(
                solver_name,
                self.config["dataset"]["datafit_name"],
                self.config["dataset"]["penalty_name"],
            ):
                solver_opts = self.config["solvers"]["solvers_opts"]
                solver = get_solver(solver_name, solver_opts)
                print("Running {}...".format(solver_name))
                path = Path(**self.config["path_opts"])
                if can_handle_compilation(solver_name):
                    result = path.fit(
                        solver,
                        datafit_train_compiled,
                        penalty_train_compiled,
                        A_train,
                    )
                else:
                    result = path.fit(
                        solver, datafit_train, penalty_train, A_train
                    )
                result["f1_score"] = []
                result["train_error"] = []
                result["test_error"] = []
                for x_res in result["x"]:
                    s = np.where(x_res != 0)[0]
                    x = np.zeros(x_res.size)
                    x[s] = np.linalg.lstsq(A_train[:, s], y_train, rcond=None)[
                        0
                    ]
                    w_train = A_train @ x
                    w_test = A_test @ x
                    result["f1_score"].append(f1_score(self.x_true, x))
                    result["train_error"].append(datafit_train.value(w_train))
                    result["test_error"].append(datafit_test.value(w_test))
            else:
                print("Skipping {}...".format(solver_name))
                result = None
            results[solver_name] = result
        self.results = results

    def load_results(self):
        print("Loading results...")

        self.nnz_grid = np.array(range(self.config["path_opts"]["max_nnz"]))
        self.stats_specs = {
            "solve_time": {"log": True},
            "f1_score": {"log": False},
            "train_error": {"log": True},
            "test_error": {"log": True},
        }

        stats = {
            stat_key: {
                solver_name: {i: [] for i in range(self.nnz_grid.size)}
                for solver_name in (
                    self.config["solvers"]["solvers_name"]
                    + self.config["relaxed_solvers"]
                )
            }
            for stat_key in self.stats_specs.keys()
        }

        found, match, empty, notcv = 0, 0, 0, 0
        for result_path in self.results_dir.glob(self.name + "_*.pickle"):
            found += 1
            with open(result_path, "rb") as file:
                file_data = pickle.load(file)
                if self.config == file_data["config"]:
                    match += 1
                    if not any(file_data["results"].values()):
                        empty += 1
                        continue
                    for solver_name, result in file_data["results"].items():
                        if result is not None:
                            for k in self.nnz_grid:
                                idx = [
                                    (i, train_error)
                                    for i, train_error in enumerate(
                                        result["train_error"]
                                    )
                                    if result["status"][i] == Status.OPTIMAL
                                    and result["n_nnz"][i] == k
                                ]
                                if len(idx) == 0:
                                    notcv += 1
                                    continue
                                i = min(idx, key=lambda x: x[1])[0]
                                for (
                                    stat_name,
                                    stat_values,
                                ) in stats.items():
                                    stat_values[solver_name][k].append(
                                        result[stat_name][i]
                                    )

        print("  {} files found".format(found))
        print("  {} files matched".format(match))
        print("  {} empty results".format(empty))
        print("  {} not converged".format(notcv))

        if (match == 0) or (match == empty):
            return

        self.mean_stats = {
            stat_key: {
                solver_key: np.array(
                    [
                        (
                            np.mean(solver_values[i])
                            if len(solver_values[i])
                            else np.nan
                        )
                        for i in range(self.nnz_grid.size)
                    ]
                )
                for solver_key, solver_values in stat_values.items()
            }
            for stat_key, stat_values in stats.items()
        }

        for stat_key, stat_values in self.mean_stats.items():
            if stat_key in ["train_error", "test_error"]:
                for solver_key, solver_values in stat_values.items():
                    self.mean_stats[stat_key][solver_key] /= np.nanmax(
                        solver_values
                    )

    def plot(self):
        _, axs = plt.subplots(
            1,
            len(self.mean_stats),
            squeeze=False,
            figsize=(3 * len(self.mean_stats), 3),
        )
        for i, (stat_name, stat_values) in enumerate(self.mean_stats.items()):
            for solver_name, solver_values in stat_values.items():
                if (
                    stat_name != "solve_time"
                    and solver_name != "el0ps"
                    and solver_name not in self.config["relaxed_solvers"]
                ):
                    solver_values = np.full(self.nnz_grid.shape, np.nan)
                axs[0, i].plot(
                    self.nnz_grid,
                    solver_values,
                    label=solver_name,
                )
            axs[0, i].grid(visible=True, which="major", axis="both")
            axs[0, i].grid(visible=True, which="minor", axis="both", alpha=0.2)
            axs[0, i].minorticks_on()
            axs[0, i].set_xlabel("n_nnz")
            axs[0, i].set_ylabel(stat_name)
            if self.stats_specs[stat_name]["log"]:
                axs[0, i].set_yscale("log")
        axs[0, 0].legend()
        plt.tight_layout()
        plt.show()

    def save_plot(self):
        print("Saving data...")
        save_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        save_file = "{}_{}.csv".format(self.name, save_uuid)
        save_path = self.saves_dir.joinpath(save_file)
        table = pd.DataFrame({"nnz_grid": self.nnz_grid})
        for stat_name, stat_values in self.mean_stats.items():
            for solver_name, solver_values in stat_values.items():
                table[solver_name + "_" + stat_name] = solver_values
        table.to_csv(save_path, index=False)
