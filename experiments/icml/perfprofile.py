import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
import pickle
import sys
from datetime import datetime
from el0ps.solver import Status

sys.path.append(pathlib.Path(__file__).parent.parent.parent.absolute())
from experiments.experiment import Experiment  # noqa
from experiments.solvers import (  # noqa
    can_handle_compilation,
    can_handle_instance,
    get_solver,
)


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
            return

        print("Computing statistics...")
        for solver_name, solver_times in times.items():
            print("  {}".format(solver_name))
            print("     num : {}".format(len(solver_times)))
            print("     mean: {}".format(np.mean(solver_times)))
            print("     std : {}".format(np.std(solver_times)))

        min_times = np.min([np.min(v) for v in times.values()])
        max_times = np.max([np.max(v) for v in times.values()])
        min_nodes = np.min([np.min(v) for v in nodes.values()])
        max_nodes = np.max([np.max(v) for v in nodes.values()])
        grid_times = np.logspace(
            np.floor(np.log10(min_times)),
            np.ceil(np.log10(max_times)),
            100,
        )
        grid_nodes = np.logspace(
            np.floor(np.log10(min_nodes)),
            np.ceil(np.log10(max_nodes)),
            100,
        )
        curve_times = {
            solver_name: [np.sum(stats <= g) for g in grid_times]
            for solver_name, stats in times.items()
        }
        curve_nodes = {
            solver_name: [np.sum(stats <= g) for g in grid_nodes]
            for solver_name, stats in nodes.items()
        }
        self.profiles = {
            "times": {"grid": grid_times, "curve": curve_times},
            "nodes": {"grid": grid_nodes, "curve": curve_nodes},
        }

    def plot(self):
        _, axs = plt.subplots(1, 2)
        for solver_name in self.config["solvers"]["solvers_name"]:
            axs[0].plot(
                self.profiles["times"]["grid"],
                self.profiles["times"]["curve"][solver_name],
                label=solver_name,
            )
            axs[1].plot(
                self.profiles["nodes"]["grid"],
                self.profiles["nodes"]["curve"][solver_name],
                label=solver_name,
            )
        for ax in axs.flatten():
            ax.grid(visible=True, which="major", axis="both")
            ax.grid(visible=True, which="minor", axis="both", alpha=0.2)
            ax.minorticks_on()
            ax.set_xscale("log")
        axs[0].set_xlabel("Time budget")
        axs[1].set_xlabel("Relax budget")
        axs[0].set_ylabel("Inst. solved")
        axs[0].legend()
        plt.show()

    def save_plot(self):
        print("Saving data...")
        save_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        for stat_name, stat_vars in self.profiles.items():
            table = pd.DataFrame({"grid": stat_vars["grid"]})
            for solver_name in self.config["solvers"]["solvers_name"]:
                table[solver_name] = stat_vars["curve"][solver_name]
            file_name = "{}_{}_{}".format(self.name, stat_name, save_uuid)
            save_path = self.saves_dir.joinpath("{}.csv".format(file_name))
            table.to_csv(save_path, index=False)
