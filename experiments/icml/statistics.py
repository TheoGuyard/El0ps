import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
import pickle
import sys
from datetime import datetime
from el0ps.datafit import *  # noqa
from el0ps.path import Path
from el0ps.solver import Status
from el0ps.utils import compiled_clone
from sklearn.model_selection import train_test_split

sys.path.append(pathlib.Path(__file__).parent.parent.parent.absolute())
from experiments.experiment import Experiment  # noqa
from experiments.solvers import (  # noqa
    OmpPath,
    LassoPath,
    EnetPath,
    can_handle_compilation,
    can_handle_instance,
    get_solver,
    get_path,
)
from experiments.instances import calibrate_parameters, f1_score  # noqa


class Statistics(Experiment):
    name = "statistics"

    relaxed_solver_names = ["OmpPath", "LassoPath", "EnetPath"]

    def run(self):

        # Train/test splitting
        print("Splitting train/test data...")
        A_train, A_test, y_train, y_test = train_test_split(
            self.A,
            self.datafit.y,
            test_size=self.config["dataset"]["test_size"],
        )

        print("Calibrating parameters on train data...")
        datafit_train, penalty_train, _, _ = calibrate_parameters(
            self.config["dataset"]["datafit_name"],
            self.config["dataset"]["penalty_name"],
            A_train,
            y_train,
            self.x_true,
        )
        datafit_test = eval(self.config["dataset"]["datafit_name"])(y_test)
        datafit_train_compiled = compiled_clone(datafit_train)
        penalty_train_compiled = compiled_clone(penalty_train)

        results = {}

        # Relax methods
        for solver_name in self.relaxed_solver_names:
            print("Running {}...".format(solver_name))
            path = get_path(solver_name, self.config["path_opts"])
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
                    + self.relaxed_solver_names
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
        _, axs = plt.subplots(1, len(self.mean_stats), squeeze=False)
        for i, (stat_name, stat_values) in enumerate(self.mean_stats.items()):
            for solver_name, solver_values in stat_values.items():
                axs[0, i].plot(
                    self.nnz_grid,
                    solver_values,
                    label=solver_name,
                    marker="o",
                )
            axs[0, i].grid(visible=True, which="major", axis="both")
            axs[0, i].grid(visible=True, which="minor", axis="both", alpha=0.2)
            axs[0, i].minorticks_on()
            axs[0, i].set_xlabel("n_nnz")
            axs[0, i].set_ylabel(stat_name)
            if self.stats_specs[stat_name]["log"]:
                axs[0, i].set_yscale("log")
            axs[0, i].legend()
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
