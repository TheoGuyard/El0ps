import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
import pickle
import sys
from datetime import datetime
from el0ps.path import Path
from el0ps.solver import Status

sys.path.append(pathlib.Path(__file__).parent.parent.parent.absolute())
from experiments.base import Experiment  # noqa
from experiments.solvers import can_handle, get_solver  # noqa


class Realworld(Experiment):
    name = "realworld"

    def run(self):
        results = {}
        for solver_name in self.config["solvers"]["solvers_name"]:
            if can_handle(
                solver_name,
                self.config["dataset"]["datafit_name"],
                self.config["dataset"]["penalty_name"],
            ):
                solver = get_solver(
                    solver_name, self.config["solvers"]["solvers_opts"]
                )
                print("Running {}...".format(solver_name))
                path = Path(**self.config["path_opts"])
                result = path.fit(solver, self.datafit, self.penalty, self.A)
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
                    np.mean(solver_values[i])
                    if len(solver_values[i])
                    else np.nan
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
