import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pandas as pd
import pickle
import sys
import yaml
from el0ps.solver import Status

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graphics import get_solver_name_color  # noqa


def figure(config_path, save=False):
    print("Preprocessing...")
    base_dir = pathlib.Path(__file__).parent.absolute()
    result_dir = pathlib.Path(base_dir, "results")
    config_path = pathlib.Path(config_path)

    assert result_dir.is_dir()
    assert config_path.is_file()

    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    print("Recovering results...")
    lmbd_ratio_grid = np.logspace(
        np.log10(config["path_options"]["lmbd_ratio_max"]),
        np.log10(config["path_options"]["lmbd_ratio_min"]),
        config["path_options"]["lmbd_ratio_num"],
    )
    all_times = {
        solver_name: {i: [] for i in range(lmbd_ratio_grid.size)}
        for solver_name in config["solver_names"]
    }
    all_sparsities = {
        solver_name: {i: [] for i in range(lmbd_ratio_grid.size)}
        for solver_name in config["solver_names"]
    }

    found = 0
    matched = 0
    errored = 0
    notcved = 0
    for result_path in result_dir.glob("*.pickle"):
        found += 1
        try:
            with open(result_path, "rb") as file:
                data = pickle.load(file)
                for solver_name in config["solver_names"]:
                    for i in range(lmbd_ratio_grid.size):
                        result = data["results"][solver_name][i]
                        if result is not None:
                            if result["status"] == Status.OPTIMAL:
                                all_times[solver_name][i].append(
                                    result["solve_time"]
                                )
                                all_sparsities[solver_name][i].append(
                                    result["sparsity"]
                                )
                            else:
                                notcved += 1
                matched += 1
        except Exception as e:
            print(e)
            errored += 1

    print("{} files founds".format(found))
    print("{} files matched".format(matched))
    print("{} files errored".format(errored))
    print("{} solvers not converged".format(notcved))

    if matched == 0:
        return

    mean_times = {
        solver_name: [
            np.mean(times[i]) if len(times[i]) > 0 else np.nan
            for i in range(lmbd_ratio_grid.size)
        ]
        for solver_name, times in all_times.items()
    }
    mean_sparsities = {
        solver_name: [
            np.mean(sparsities[i]) if len(sparsities[i]) > 0 else np.nan
            for i in range(lmbd_ratio_grid.size)
        ]
        for solver_name, sparsities in all_sparsities.items()
    }

    if save:
        print("Saving...")
        name = "_".join([str(v) for v in config["dataset"].values()]) + ".csv"
        data = pd.DataFrame({"lmbd_ratio_grid": lmbd_ratio_grid})
        for k, v in mean_times.items():
            data[k] = v
        path = pathlib.Path(__file__).parent.joinpath("saves", name)
        data.to_csv(path, index=False)
    else:
        print("Plotting...")
        plt.rcParams["axes.prop_cycle"] = plt.cycler(
            "color", plt.cm.tab20c.colors
        )
        fig, axs = plt.subplots(1, 2)
        for solver_name in config["solver_names"]:
            axs[0].plot(
                lmbd_ratio_grid,
                mean_times[solver_name],
                label=solver_name,
                color=get_solver_name_color(solver_name),
            )
            axs[1].plot(
                lmbd_ratio_grid,
                mean_sparsities[solver_name],
                label=solver_name,
                color=get_solver_name_color(solver_name),
            )

        for i in [0, 1]:
            axs[i].set_xscale("log")
            axs[i].set_xscale("log")
            axs[i].grid(visible=True, which="major", axis="both")
            axs[i].grid(visible=True, which="minor", axis="both", alpha=0.2)
            axs[i].minorticks_on()
            axs[i].set_xlabel("lmbd/lmbd_max")

            axs[i].invert_xaxis()

        axs[0].set_ylabel("Solve time")
        axs[0].set_yscale("log")
        axs[0].legend()
        axs[1].set_ylabel("Sparsity")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("-s", "--save", action="store_true")
    args = parser.parse_args()
    figure(args.config_path, save=args.save)
