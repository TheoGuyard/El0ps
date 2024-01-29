import argparse
import pathlib
import pickle
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from el0ps.solver import Status

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)


def graphic(config_path, save=False):
    config_path = pathlib.Path(config_path)
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    print("Recovering results...")
    results_dir = pathlib.Path(__file__).parent.absolute().joinpath("results")
    found, match, notcv = 0, 0, 0
    perf_times = {s: [] for s in config["solvers"]["solvers_name"]}
    perf_nodes = {s: [] for s in config["solvers"]["solvers_name"]}
    for result_path in results_dir.glob("*.pickle"):
        found += 1
        with open(result_path, "rb") as file:
            file_data = pickle.load(file)
            if file_data["config"] == config:
                match += 1
                for solver_name, result in file_data["results"].items():
                    if result is not None:
                        if result.status == Status.OPTIMAL:
                            perf_times[solver_name].append(result.solve_time)
                            perf_nodes[solver_name].append(result.iter_count)
                        else:
                            notcv += 1
    print("  {} files founds".format(found))
    print("  {} files matched".format(match))
    print("  {} not converged".format(notcv))
    print()

    if match == 0:
        return

    print("Computing statistics...")
    grid_perf_times = np.logspace(
        np.floor(np.log10(np.min([np.min(v) for v in perf_times.values()]))),
        np.ceil(np.log10(np.max([np.max(v) for v in perf_times.values()]))),
        100,
    )
    grid_perf_nodes = np.logspace(
        np.floor(np.log10(np.min([np.min(v) for v in perf_nodes.values()]))),
        np.ceil(np.log10(np.max([np.max(v) for v in perf_nodes.values()]))),
        100,
    )
    profile_perf_times = {
        solver_name: [np.sum(stats <= g) for g in grid_perf_times]
        for solver_name, stats in perf_times.items()
    }
    profile_perf_nodes = {
        solver_name: [np.sum(stats <= g) for g in grid_perf_nodes]
        for solver_name, stats in perf_nodes.items()
    }

    if save:
        print("Saving data...")
        save_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        stats = {
            "perfprofile_times": {
                "grid": grid_perf_times,
                "profile": profile_perf_times,
            },
            "perfprofile_nodes": {
                "grid": grid_perf_nodes,
                "profile": profile_perf_nodes,
            },
        }
        for stat_name, stat_vars in stats.items():
            table = pd.DataFrame({"grid": stat_vars["grid"]})
            for solver_name in config["solvers"]["solvers_name"]:
                table[solver_name] = stat_vars["profile"][solver_name]
            file_name = "{}_{}".format(stat_name, save_uuid)
            saves_dir = (
                pathlib.Path(__file__).parent.absolute().joinpath("saves")
            )
            save_path = saves_dir.joinpath("{}.csv".format(file_name))
            table.to_csv(save_path, index=False)
    else:
        print("Plotting figure...")
        _, axs = plt.subplots(1, 2)
        for solver_name in config["solvers"]["solvers_name"]:
            axs[0].plot(
                grid_perf_times,
                profile_perf_times[solver_name],
                label=solver_name,
            )
            axs[1].plot(
                grid_perf_nodes,
                profile_perf_nodes[solver_name],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("-s", "--save", action="store_true")
    args = parser.parse_args()
    graphic(args.config_path, save=args.save)
