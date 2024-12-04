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
from el0ps.datafits import *  # noqa
from el0ps.penalties import *  # noqa
from el0ps.path import Path
from el0ps.solvers import Status
from el0ps.utils import compiled_clone, compute_lmbd_max
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV, ElasticNet
from numba import njit
from numpy.typing import ArrayLike
import time

sys.path.append(pathlib.Path(__file__).parent.parent.absolute())
from experiments.instances import (  # noqa
    calibrate_parameters,
    acc_score,
    fdr_score,
    f1_score,
    get_data,
)
from experiments.solvers import (  # noqa
    can_handle_compilation,
    can_handle_instance,
    get_solver,
    get_path,
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
        self.A = A
        self.y = y
        self.x_true = x_true

    def calibrate_parameters(self):
        flag_calibration = False
        if self.config["dataset"]["dataset_type"] == "hardcoded":
            file_name = self.config["dataset"]["dataset_opts"]["dataset_name"]
            file_dir = pathlib.Path(__file__).parent.joinpath("datasets")
            file_path = file_dir.joinpath(file_name).with_suffix(".pkl")
            with open(file_path, "rb") as dataset_file:
                data = pickle.load(dataset_file)
                if "calibrations" in data.keys():
                    for calibration in data["calibrations"]:
                        if (
                            str(calibration["datafit"]) == self.config["dataset"]["datafit_name"] and
                            str(calibration["penalty"]) == self.config["dataset"]["penalty_name"]
                        ):
                            print("Calibration found")
                            self.datafit = calibration["datafit"]
                            self.penalty = calibration["penalty"]
                            self.lmbd = calibration["lmbd"]
                            self.x_cal = calibration["x_cal"]
                            flag_calibration = True
                            break
        if not flag_calibration:
            print("Calibrating parameters...")
            self.datafit, self.penalty, self.lmbd, self.x_cal = (
                calibrate_parameters(
                    self.config["dataset"]["datafit_name"],
                    self.config["dataset"]["penalty_name"],
                    self.A,
                    self.y,
                    self.x_true,
                )
            )

        lmbd_max = compute_lmbd_max(self.datafit, self.penalty, self.A)
        print("  num nz: {}".format(sum(self.x_cal != 0.0)))
        print("  lratio: {}".format(self.lmbd / lmbd_max))
        for param_name, param_value in self.penalty.params_to_dict().items():
            print("  {}\t: {}".format(param_name, param_value))

        if "calibration_opts" in self.config["dataset"].keys():
            calibration_opts = self.config["dataset"]["calibration_opts"]
            if "factor_lmbd" in calibration_opts.keys():
                print("  factor for lmbd: {}".format(calibration_opts["factor_lmbd"]))
                self.lmbd *= calibration_opts["factor_lmbd"]
            if "factor_penalty_params" in calibration_opts.keys():
                for param_name, param_factor in calibration_opts["factor_penalty_params"].items():
                    print("  factor for {}: {}".format(param_name, param_factor))
                    param_val = getattr(self.penalty, param_name)
                    setattr(self.penalty, param_name, param_factor * param_val)

    def precompile(self):
        print("Precompiling datafit and penalty...")
        self.compiled_datafit = compiled_clone(self.datafit)
        self.compiled_penalty = compiled_clone(self.penalty)
        if "solvers" in self.config.keys():
            solver_opts = deepcopy(self.config["solvers"]["solvers_opts"])
        else:
            solver_opts = {}
        solver_opts["time_limit"] = 5.0
        solver = get_solver("el0ps", solver_opts)
        solver.solve(
            self.compiled_datafit,
            self.compiled_penalty,
            self.A,
            self.lmbd,
        )

    def precompile_solver(self, solver):
        print(f"Precompiling solver {solver}...")
        time_limit = solver.options.time_limit
        solver.options.time_limit = 5.0
        solver.solve(
            self.compiled_datafit,
            self.compiled_penalty,
            self.A,
            self.lmbd,
        )
        solver.options.time_limit = time_limit
        solver.options.bounding_skip_setup = True

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
                    self.precompile_solver(solver)
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
        nodes = {s: [] for s in self.config["solvers"]["solvers_name"]}
        for result_path in self.results_dir.glob(self.name + "_*.pickle"):
            found += 1
            with open(result_path, "rb") as file:
                file_data = pickle.load(file)
                if file_data["config"] == self.config:
                    match += 1
                    for solver_name, result in file_data["results"].items():
                        if result.status == Status.OPTIMAL:
                            times[solver_name].append(result.solve_time)
                            nodes[solver_name].append(result.iter_count)
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
        min_times = np.nanmin([np.nanmin(v) if len(v) else np.nan for v in times.values()])
        max_times = np.nanmax([np.nanmax(v) if len(v) else np.nan for v in times.values()])
        self.grid_times = np.logspace(
            np.floor(np.log10(min_times)),
            np.ceil(np.log10(max_times)),
            100,
        )
        self.curve_times = {
            solver_name: [np.mean(stats <= g) for g in self.grid_times]
            for solver_name, stats in times.items()
        }

        min_nodes = np.nanmin([np.nanmin(v) if len(v) else np.nan for v in nodes.values()])
        max_nodes = np.nanmax([np.nanmax(v) if len(v) else np.nan for v in nodes.values()])
        self.grid_nodes = np.logspace(
            np.floor(np.log10(min_nodes)),
            np.ceil(np.log10(max_nodes)),
            100,
        )
        self.curve_nodes = {
            solver_name: [np.mean(stats <= g) for g in self.grid_nodes]
            for solver_name, stats in nodes.items()
        }

        #######
        mean_times = {s: np.mean(v) if len(v) else '' for s, v in times.items()}
        for solver_name in [
            "el0ps[simpruning=False]",
            "el0ps",
            "el0ps[simpruning=False,peeling=True]",
            "el0ps[peeling=True]",
            "mip[optimizer_name=cplex]",
            "mip[optimizer_name=gurobi]",
            "mip[optimizer_name=mosek]",
            "oa",
            "l0bnb",
        ]:
            print("{},".format(mean_times[solver_name]), end="")
        print()
        #######

    def plot(self):

        return

        if self.curve_times is None or self.curve_nodes is None:
            return
        _, axs = plt.subplots(1, 2)
        for solver_name in self.config["solvers"]["solvers_name"]:
            axs[0].plot(
                self.grid_times,
                self.curve_times[solver_name],
                label=solver_name,
            )
            axs[1].plot(
                self.grid_nodes,
                self.curve_nodes[solver_name],
                label=solver_name,
            )
        for ax in axs:
            ax.grid(visible=True, which="major", axis="both")
            ax.grid(visible=True, which="minor", axis="both", alpha=0.2)
            ax.minorticks_on()
            ax.set_xscale("log")
            ax.set_ylabel("Inst. solved")
        axs[0].set_xlabel("Time budget")
        axs[1].set_xlabel("Node budget")
        axs[1].legend()
        plt.show()

    def save_plot(self):
        print("Saving data...")
        save_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        table_times = pd.DataFrame({"grid_times": self.grid_times})
        table_nodes = pd.DataFrame({"grid_nodes": self.grid_nodes})
        for solver_name in self.config["solvers"]["solvers_name"]:
            table_times[solver_name] = self.curve_times[solver_name]
            table_nodes[solver_name] = self.curve_nodes[solver_name]
        file_times_name = "{}_{}_{}".format(self.name, "times", save_uuid)
        file_nodes_name = "{}_{}_{}".format(self.name, "nodes", save_uuid)
        save_times_path = self.saves_dir.joinpath(
            "{}.csv".format(file_times_name)
        )
        save_nodes_path = self.saves_dir.joinpath(
            "{}.csv".format(file_nodes_name)
        )
        table_times.to_csv(save_times_path, index=False)
        table_nodes.to_csv(save_nodes_path, index=False)


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
                    self.precompile_solver(solver)
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
            np.log10(self.config["path_opts"]["lmbd_max"]),
            np.log10(self.config["path_opts"]["lmbd_min"]),
            self.config["path_opts"]["lmbd_num"],
        )
        self.stats_specs = {
            "solve_time": {"log": True},
            # "objective_value": {"log": False},
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
                # if self.config == file_data["config"]:
                if (
                    self.config["expname"] == file_data["config"]["expname"]
                    and self.config["dataset"]
                    == file_data["config"]["dataset"]
                    and np.all(
                        np.isin(
                            file_data["config"]["solvers"]["solvers_name"],
                            self.config["solvers"]["solvers_name"],
                        )
                    )
                    and self.config["path_opts"]
                    == file_data["config"]["path_opts"]
                ):
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
            self.mean_stats = None
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
        if self.mean_stats is None:
            return
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

        datafit_name = self.config["dataset"]["datafit_name"]
        datafit_train = eval(datafit_name)(y_train)
        datafit_test = eval(datafit_name)(y_test)

        results = {}

        # Relax methods
        for method_name in self.config["methods_name"]:
            print("Running {}...".format(method_name))
            path = get_path(method_name, self.config["path_opts"])
            result = path.fit(datafit_name, A_train, y_train)
            result["f1_score"] = []
            result["acc_score"] = []
            result["fdr_score"] = []
            result["train_error"] = []
            result["test_error"] = []
            for x in result["x"]:
                result["f1_score"].append(f1_score(self.x_true, x))
                result["acc_score"].append(acc_score(self.x_true, x))
                result["fdr_score"].append(fdr_score(self.x_true, x))
                result["train_error"].append(datafit_train.value(A_train @ x))
                result["test_error"].append(datafit_test.value(A_test @ x))
            results[method_name] = result
        self.results = results

    def load_results(self):
        print("Loading results...")

        self.nnz_grid = np.array(range(self.config["path_opts"]["max_nnz"]))
        self.stats_specs = {
            "solve_time": {"log": True},
            # "acc_score": {"log": False},
            # "fdr_score": {"log": False},
            "f1_score": {"log": False},
            "train_error": {"log": True},
            # "test_error": {"log": False},
        }

        stats = {
            stat_key: {
                method_name: {i: [] for i in range(self.nnz_grid.size)}
                for method_name in self.config["methods_name"]
            } for stat_key in self.stats_specs.keys()
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
                    for method_name, result in file_data["results"].items():
                        if result is not None:
                            for k in self.nnz_grid:
                                mask = np.ones(len(result["n_nnz"]))
                                mask[np.logical_or(
                                    np.array(result["n_nnz"]) > k,
                                    np.array(result["status"]) != Status.OPTIMAL
                                )] = np.inf
                                if np.all(mask == np.inf):
                                    notcv += 1
                                    continue
                                idx = np.argmin(np.array(result["test_error"]) * mask)
                                for stat_name, stat_values in stats.items():
                                    stat_values[method_name][k].append(
                                        result[stat_name][idx]
                                    )

        print("  {} files found".format(found))
        print("  {} files matched".format(match))
        print("  {} empty results".format(empty))
        print("  {} not converged".format(notcv))

        if (match == 0) or (match == empty):
            return

        self.mean_stats = {
            stat_name: {
                method_name: np.array(
                    [
                        (
                            np.mean(solver_values[i])
                            if len(solver_values[i])
                            else np.nan
                        )
                        for i in range(self.nnz_grid.size)
                    ]
                )
                for method_name, solver_values in stat_values.items()
            }
            for stat_name, stat_values in stats.items()
        }

    def plot(self):
        _, axs = plt.subplots(
            1,
            len(self.mean_stats),
            squeeze=False,
            figsize=(3 * len(self.mean_stats), 3),
        )
        for i, (stat_name, stat_values) in enumerate(self.mean_stats.items()):
            for method_name, solver_values in stat_values.items():
                axs[0, i].plot(
                    self.nnz_grid,
                    solver_values,
                    label=method_name,
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
            for method_name, solver_values in stat_values.items():
                table[method_name + "_" + stat_name] = solver_values
        table.to_csv(save_path, index=False)


class Simpruning(Experiment):
    name = "simpruning"

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
                    self.precompile_solver(solver)
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
        self.grid_depth = [d for d in range(self.config["dataset"]["dataset_opts"]["n"] + 1)]
        prop_depth = {
            solver_name: {d: [] for d in self.grid_depth}
            for solver_name in self.config["solvers"]["solvers_name"]
        }
        for result_path in self.results_dir.glob(self.name + "_*.pickle"):
            found += 1
            with open(result_path, "rb") as file:
                file_data = pickle.load(file)
                if file_data["config"] == self.config:
                    match += 1
                    for solver_name, result in file_data["results"].items():
                        if result.status == Status.OPTIMAL:
                            for d in self.grid_depth:
                                prop_depth[solver_name][d].append(
                                    np.sum(np.array(result.trace["node_depth"]) == d) / len(result.trace["node_depth"])
                                )
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
        self.mean_prop_depth = {
            solver_name: [
                np.mean(prop_depth[solver_name][d]) if 
                len(prop_depth[solver_name][d]) else np.nan
                for d in self.grid_depth
            ] for solver_name in self.config["solvers"]["solvers_name"]
        }

    def plot(self):
        _, axs = plt.subplots(1, 1)
        for solver_name in self.config["solvers"]["solvers_name"]:
            axs.plot(
                self.grid_depth,
                self.mean_prop_depth[solver_name],
                label=solver_name,
            )
        axs.grid(visible=True, which="major", axis="both")
        axs.grid(visible=True, which="minor", axis="both", alpha=0.2)
        axs.minorticks_on()
        axs.set_ylabel("Prop.")
        axs.set_xlabel("Depth")
        # axs.set_xscale("log")
        axs.legend()
        plt.show()

    def save_plot(self):
        print("Saving data...")
        save_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        table = pd.DataFrame({"grid_depth": self.grid_depth})
        for solver_name in self.config["solvers"]["solvers_name"]:
            table[solver_name] = self.mean_prop_depth[solver_name]
        file_name = "{}_{}".format(self.name, save_uuid)
        save_path = self.saves_dir.joinpath("{}.csv".format(file_name))
        table.to_csv(save_path, index=False)


class Tightening(Experiment):
    name = "tightening"

    def run(self):
        results = {}
        results["bigm"] = np.maximum(
            np.max(self.penalty.x_ub),
            -np.min(self.penalty.x_lb),
        )
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
                    self.precompile_solver(solver)
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
        self.grid_depth = [d for d in range(self.config["dataset"]["dataset_opts"]["n"] + 1)]
        bound_spread = {
            solver_name: {d: [] for d in self.grid_depth}
            for solver_name in self.config["solvers"]["solvers_name"]
        }
        num_nodes = {
            solver_name: {d: [] for d in self.grid_depth}
            for solver_name in self.config["solvers"]["solvers_name"]
        }
        for result_path in self.results_dir.glob(self.name + "_*.pickle"):
            found += 1
            with open(result_path, "rb") as file:
                file_data = pickle.load(file)
                if file_data["config"] == self.config:
                    match += 1
                    for solver_name, result in file_data["results"].items():
                        if solver_name != "bigm":
                            if result.status == Status.OPTIMAL:
                                for i, d in enumerate(result.trace["node_depth"]):
                                    if "peeling" in solver_name:
                                        bound_spread[solver_name][d].append(
                                            result.trace["node_bound_spread"][i] / 
                                            (2 * file_data["results"]["bigm"])
                                        )
                                    else:
                                        bound_spread[solver_name][d].append(1.)
                                    num_nodes[solver_name][d].append(np.sum(result.trace["node_depth"] == d))
                        else:
                            notcv += 1
        print("  {} files found".format(found))
        print("  {} files matched".format(match))
        print("  {} files not converged".format(notcv))

        if match == 0:
            self.grid_depth = None
            return

        print("Computing statistics...")
        self.mean_bound_spread = {
            solver_name: [
                np.mean(bound_spread[solver_name][d]) if 
                len(bound_spread[solver_name][d]) else np.nan
                for d in self.grid_depth
            ] for solver_name in self.config["solvers"]["solvers_name"]
        }
        self.mean_num_nodes = {
            solver_name: [
                np.mean(num_nodes[solver_name][d]) if 
                len(num_nodes[solver_name][d]) else np.nan
                for d in self.grid_depth
            ] for solver_name in self.config["solvers"]["solvers_name"]
        }

    def plot(self):
        _, axs = plt.subplots(1, 2)
        for solver_name in self.config["solvers"]["solvers_name"]:
            axs[0].plot(
                self.grid_depth,
                self.mean_bound_spread[solver_name],
                label=solver_name,
            )
            axs[1].plot(
                self.grid_depth,
                self.mean_num_nodes[solver_name],
                label=solver_name,
            )
        axs[0].grid(visible=True, which="major", axis="both")
        axs[0].grid(visible=True, which="minor", axis="both", alpha=0.2)
        axs[0].minorticks_on()
        axs[0].set_ylabel("Bound spread")
        axs[0].set_xlabel("Depth")
        axs[0].legend()
        axs[1].grid(visible=True, which="major", axis="both")
        axs[1].grid(visible=True, which="minor", axis="both", alpha=0.2)
        axs[1].minorticks_on()
        axs[1].set_ylabel("Num nodes")
        axs[1].set_xlabel("Depth")
        axs[1].legend()
        plt.show()

    def save_plot(self):
        print("Saving data...")
        save_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        table = pd.DataFrame({"grid_depth": self.grid_depth})
        for solver_name in self.config["solvers"]["solvers_name"]:
            table[solver_name] = self.mean_bound_spread[solver_name]
        file_name = "{}_{}".format(self.name, save_uuid)
        save_path = self.saves_dir.joinpath("{}.csv".format(file_name))
        table.to_csv(save_path, index=False)


class Screening(Experiment):
    name = "screening"

    def load_problem(self):
        pass

    def calibrate_parameters(self):
        pass

    def precompile(self):
        pass

    @staticmethod
    @njit
    def cg_linsys(A, s, b, x0=None, tol=1e-8):
        """Solves the square linear system

        .. math:: (A.T @ A + s * I)x = b

        using a conjugate gradient method.

        Arguments
        ---------
        A : ArrayLike, shape (m, n)
            Linear system matrix.
        s : float
            Identity matrix scaling factor.
        b : ArrayLike, shape (n,)
            Linear system right-hand-side.
        x0 : ArrayLike, shape (n,)
            Initial point.
        tol : float
            Conjugate gradient tolerance criterion.

        Returns
        -------
        x : NDArray, shape (n,)
            The linear system solution.
        """
        n = b.size
        x = x0.copy() if x0 is not None else np.zeros(n)
        r = b - np.dot(A.T, np.dot(A, x)) - s * x
        p = r.copy()
        for _ in range(n):
            g = np.dot(A.T, np.dot(A, p)) + s * p
            q = np.dot(p, g)
            a = np.dot(p, r) / q
            x += a * p
            r = b - np.dot(A.T, np.dot(A, x)) - s * x
            if np.dot(r, r) < tol:
                break
            d = np.dot(r, g) / q
            p *= -d
            p += r
        return x
    
    @staticmethod
    @njit
    def update_H1(H1, A, S1, i, l2):
        ai = A[:, i]
        A1 = A[:, S1]
        b1 = np.dot(A1.T, ai)
        d = np.dot(H1, b1)
        c = np.dot(ai, ai) + l2 - np.dot(b1, d)
        e = -d / c

        Hw = H1 + np.outer(d, d / c)

        k = np.sum(S1[:i])
        H1_new = np.zeros((H1.shape[0] + 1, H1.shape[1] + 1))
        H1_new[:k, :k] = Hw[:k, :k]
        H1_new[:k, k] = e[:k]
        H1_new[k, :k] = e[:k]
        H1_new[k, k] = 1. / c
        H1_new[k+1:, :k] = Hw[k:, :k]
        H1_new[:k, k+1:] = Hw[:k, k:]
        H1_new[k+1:, k+1:] = Hw[k:, k:]
        H1_new[k, k+1:] = e[k:]
        H1_new[k+1:, k] = e[k:]

        return H1_new

    def solve_enet(
        self,
        scr: bool = False,
        smt: bool = False,
        tol: float = 1e-10,
        max_time: float = 600.,
    ):
        
        A = self.A
        y = self.y
        l1 = self.l1
        l2 = self.l2
        a = self.a
        L = self.L

        _, n = A.shape
        x = np.zeros(n)
        w = A @ x
        u = y - w
        v = A.T @ u
        q = np.zeros(n)

        S0 = np.zeros(n, dtype=bool)
        S1 = np.zeros(n, dtype=bool)
        Sb = np.ones(n, dtype=bool)
        sg = np.zeros(n)
        H1 = np.zeros((0, 0))

        trace = {"timer": [], "pv": [], "dv": []}

        t0 = time.time()

        while True:
            h1 = v[S1] - l1 * sg[S1] - l2 * x[S1]
            x[S1] += H1 @ h1
            c = x[Sb] + v[Sb] / L
            x[Sb] = np.sign(c) * np.maximum(np.abs(c) - l1 / L, 0.) / (1. + l2 / L)
            w = A[:, Sb | S1] @ x[Sb | S1]
            u = y - w
            v[Sb | S1] = A[:, Sb | S1].T @ u
            q[Sb | S1] = np.maximum(np.abs(v[Sb | S1]) - l1, 0.)

            pv = 0.5 * np.dot(u, u) + l1 * np.sum(np.abs(x)) + 0.5 * l2 * np.dot(x, x)
            dv = -0.5 * np.dot(u, u) + np.dot(y, u) - np.dot(q, q) / (2. * l2)
            gv = np.maximum(pv - dv, 0.)

            if scr:
                test_S0 = (np.abs(v) < l1 - np.sqrt(gv) * a) & Sb
                S0[test_S0] = True
                Sb[test_S0] = False
                q[test_S0] = 0.
                change_S0 = (x != 0.) & S0
                if np.any(change_S0):
                    w -= A[:, change_S0] @ x[change_S0]
                    x[change_S0] = 0.
                    u = y - w
                    v = A.T @ u
            if smt:
                test_Sp = (v > l1 + np.sqrt(gv) * a) & Sb
                test_Sn = (v < -l1 - np.sqrt(gv) * a) & Sb
                for i in np.flatnonzero(test_Sp):
                    H1 = self.update_H1(H1, A, S1, i, l2)
                    S1[i] = True
                    Sb[i] = False
                    sg[i] = 1.
                for i in np.flatnonzero(test_Sn):
                    H1 = self.update_H1(H1, A, S1, i, l2)
                    S1[i] = True
                    Sb[i] = False
                    sg[i] = -1.

            trace["timer"].append(time.time() - t0)
            trace["pv"].append(pv)
            trace["dv"].append(dv)
            if pv - dv < tol:
                break
            if time.time() - t0 > max_time:
                break
            
        return x, trace

    def run(self):
        ratios = {}
        times = {}
        iters = {}
        base_opts = self.config["base_opts"]
        for param_name, param_values in self.config["variations"].items():
            ratios[param_name] = {}
            times[param_name] = {}
            iters[param_name] = {}
            for param_value in param_values:
                ratios[param_name][param_value] = {}
                times[param_name][param_value] = {}
                iters[param_name][param_value] = {}

                print("param {}: {}".format(param_name, param_value))
                opts = deepcopy(base_opts)
                opts[param_name] = param_value

                self.A, self.y, self.x_true = get_data_synthetic(**opts)

                print("calibrating hyperparameters...")
                # enet_cv = ElasticNetCV(cv=5, random_state=42, fit_intercept=False)
                # enet_cv.fit(self.A, self.y)
                lmax = np.linalg.norm(self.A.T @ self.y, np.inf)
                self.l1 = opts["l1_ratio"] * 0.5 * (0.3 * lmax)
                self.l2 = opts["l2_ratio"] * 0.5 * (1. - 0.5) * (0.3 * lmax)

                # self.l1 = opts["l1_ratio"] * self.A.shape[0] * enet_cv.l1_ratio_ * enet_cv.alpha_
                # self.l2 = opts["l2_ratio"] * self.A.shape[0] * (1 - enet_cv.l1_ratio_) * enet_cv.alpha_
                self.a = np.linalg.norm(self.A, axis=0)
                self.L = np.linalg.eigvalsh(self.A.T @ self.A)[-1]

                print("precompiling...")
                self.solve_enet(scr=True, smt=True, max_time=5.)

                print("solving problem...")
                x = None
                grid_tol = np.logspace(
                    np.log10(self.config["max_tol"]),
                    np.log10(self.config["min_tol"]),
                    self.config["num_tol"]
                )
                for scr in [False, True]:
                    for smt in [False, True]:
                        x, trace = self.solve_enet(
                            scr=scr, 
                            smt=smt, 
                            tol=self.config["min_tol"], 
                            max_time=self.config["max_time"]
                        )
                        pgname = "pg"
                        if scr:
                            pgname += "+scr"
                        if smt:
                            pgname += "+smt"
                        gap = np.maximum(trace["pv"][-1] - trace["dv"][-1], 0.)
                        print("  {}".format(pgname))
                        print("    time:", trace["timer"][-1])
                        print("    iter:", len(trace["timer"]))
                        print("    gap :", gap)
                        subopt = np.maximum(np.array(trace["pv"]) - trace["dv"][-1], 0.)
                        times[param_name][param_value][pgname] = np.full(len(grid_tol), np.nan)
                        iters[param_name][param_value][pgname] = np.full(len(grid_tol), np.nan)
                        for i, ti in enumerate(grid_tol):
                            idx = np.where(subopt <= ti)[0]
                            if len(idx):
                                times[param_name][param_value][pgname][i] = trace["timer"][idx[0]]
                                iters[param_name][param_value][pgname][i] = idx[0]

                print("computing rates...")
                m = self.A.shape[0]
                w = self.A @ x
                u = self.y - w
                v = self.A.T @ u
                q = np.maximum(np.abs(v) - self.l1, 0.)
                d = np.abs(np.abs(v) - self.l1) / self.a
                dmin = np.min(d)
                dmax = np.max(d)
                radii = np.linspace(dmin - 1e-8, dmax + 1e-8, self.config["radii_num"])
                num_S0 = np.sum(np.abs(x) == 0.)
                num_S1 = np.sum(np.abs(x) != 0.)
                ratio_S0 = np.empty(radii.shape)
                ratio_S1 = np.empty(radii.shape)
                for i, r in enumerate(radii):
                    ratio_S0[i] = (np.sum(r < (self.l1 - np.abs(v)) / self.a) / num_S0)
                    ratio_S1[i] = (np.sum(r < (np.abs(v) - self.l1) / self.a) / num_S1)

                ratios[param_name][param_value]["ratios_S0"] = ratio_S0
                ratios[param_name][param_value]["ratios_S1"] = ratio_S1

                print()
                
        self.results = {"ratios": ratios, "times": times, "iters": iters}

    def load_results(self):
        print("Loading results...")
        found, match, notcv = 0, 0, 0
        ratios_S0 = {
            param_name: {
                param_value: {
                    r: [] for r in range(self.config["radii_num"])
                } for param_value in param_values
            } for param_name, param_values in self.config["variations"].items()
        }
        ratios_S1 = {
            param_name: {
                param_value: {
                    r: [] for r in range(self.config["radii_num"])
                } for param_value in param_values
            } for param_name, param_values in self.config["variations"].items()
        }
        times_pg = {
            param_name: {
                param_value: {
                    pgname: [[] for _ in range(self.config["num_tol"])]
                    for pgname in ["pg", "pg+scr", "pg+smt", "pg+scr+smt"]
                } for param_value in param_values
            } for param_name, param_values in self.config["variations"].items()
        }
        for result_path in self.results_dir.glob(self.name + "_*.pickle"):
            found += 1
            with open(result_path, "rb") as file:
                file_data = pickle.load(file)
                if file_data["config"] == self.config:
                    match += 1
                    for param_name, param_values in file_data["results"]["ratios"].items():
                        for param_value, ratios in param_values.items():
                            for r, ratio in enumerate(ratios["ratios_S0"]):
                                ratios_S0[param_name][param_value][r].append(ratio)
                            for r, ratio in enumerate(ratios["ratios_S1"]):
                                ratios_S1[param_name][param_value][r].append(ratio)
                    for param_name, param_values in file_data["results"]["times"].items():
                        for param_value, times in param_values.items():
                            for pgname, pgtimes in times.items():
                                for i in range(self.config["num_tol"]):
                                    times_pg[param_name][param_value][pgname][i].append(pgtimes[i])
        print("  {} files found".format(found))
        print("  {} files matched".format(match))
        print("  {} files not converged".format(notcv))

        if match == 0:
            self.grid_times = None
            self.curve_times = None
            return

        print("Computing statistics...")
        self.grid_radii = np.linspace(0, 1, self.config["radii_num"])
        self.mean_ratios_S0 = {
            param_name: {
                param_value: [np.mean(ratio) for ratio in ratios.values()]
                for param_value, ratios in param_values.items()
            } for param_name, param_values in ratios_S0.items()
        }
        self.mean_ratios_S1 = {
            param_name: {
                param_value: [np.mean(ratio) for ratio in ratios.values()]
                for param_value, ratios in param_values.items()
            } for param_name, param_values in ratios_S1.items()
        }
        self.mean_times_pg = {
            param_name: {
                param_value: {
                    pgname: [np.mean(time) for time in times]
                    for pgname, times in pgtimes.items()
                } for param_value, pgtimes in param_values.items()
            } for param_name, param_values in times_pg.items()
        }

    def plot(self):
        _, axs = plt.subplots(2, len(self.config["variations"]), squeeze=False)
        colors = ["blue", "green", "orange", "red"]
        for i, (param_name, param_values) in enumerate(self.config["variations"].items()):
            for j, param_value in enumerate(param_values):
                axs[0, i].plot(
                    self.grid_radii,
                    self.mean_ratios_S0[param_name][param_value],
                    label="{}: {}".format(param_name, param_value),
                )
                axs[0, i].plot(
                    self.grid_radii,
                    self.mean_ratios_S1[param_name][param_value],
                    label="{}: {}".format(param_name, param_value),
                )
                grid_tol = np.logspace(
                    np.log10(self.config["max_tol"]),
                    np.log10(self.config["min_tol"]),
                    self.config["num_tol"]
                )
                for pgname, pgtimes in self.mean_times_pg[param_name][param_value].items():
                    if not ("scr" in pgname) and not ("smt" in pgname):
                        linestyle = "solid"
                    elif ("scr" in pgname) and not ("smt" in pgname):
                        linestyle = "dashed"
                    elif not ("scr" in pgname) and ("smt" in pgname):
                        linestyle = "dotted"
                    elif ("scr" in pgname) and ("smt" in pgname):
                        linestyle = "dashdot"
                    axs[1, i].plot(
                        pgtimes,
                        grid_tol,
                        label=pgname,
                        color=colors[j],
                        linestyle=linestyle,
                    )
            axs[0, i].grid(visible=True, which="major", axis="both")
            axs[0, i].minorticks_on()
            axs[0, i].set_xlabel("gamma")
            axs[0, i].set_ylabel("identif")
            axs[1, i].grid(visible=True, which="major", axis="both")
            axs[1, i].minorticks_on()
            axs[1, i].set_ylabel("subopt")
            axs[1, i].set_xlabel("time")
            axs[1, i].set_xscale("log")
            axs[1, i].set_yscale("log")
        axs[0, -1].legend()
        axs[1, -1].legend()
        plt.show()

    def save_plot(self):
        print("Saving data...")
        table = pd.DataFrame({"radii": self.grid_radii})
        for param_name, param_values in self.config["variations"].items():
            for param_value in param_values:
                table[f"{param_name}-{param_value}-S0"] = self.mean_ratios_S0[param_name][param_value]
                table[f"{param_name}-{param_value}-S1"] = self.mean_ratios_S1[param_name][param_value]
        file_name = "screening_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
            self.config["base_opts"]["supp_val"],
            self.config["base_opts"]["k"],
            self.config["base_opts"]["m"],
            self.config["base_opts"]["n"],
            self.config["base_opts"]["matrix"],
            self.config["base_opts"]["s"],
            self.config["base_opts"]["l1_ratio"],
            self.config["base_opts"]["l2_ratio"],
        )
        file_path = self.saves_dir.joinpath(file_name)
        table.to_csv(file_path, index=False)
