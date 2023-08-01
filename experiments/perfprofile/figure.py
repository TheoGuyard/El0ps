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
    results_dir = pathlib.Path(base_dir, "results")
    config_path = pathlib.Path(config_path)

    assert results_dir.is_dir()
    assert config_path.is_file()

    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    print("Recovering results...")
    all_times = {solver_name: [] for solver_name in config["solver_names"]}

    found = 0
    matched = 0
    errored = 0
    notcved = 0
    for result_path in results_dir.glob("*.pickle"):
        found += 1
        try:
            with open(result_path, "rb") as file:
                file_data = pickle.load(file)
                if file_data["config"] != config:
                    continue
                for solver_name, result in file_data["results"].items():
                    if result is not None:
                        if result.status == Status.OPTIMAL:
                            all_times[solver_name].append(result.solve_time)
                        else:
                            print("{} did not converged: {}".format(solver_name, result.status))
                            notcved += 1
                matched += 1
        except Exception:
            errored += 1

    print("{} files founds".format(found))
    print("{} files matched".format(matched))
    print("{} files errored".format(errored))
    print("{} not converged".format(notcved))

    if matched == 0:
        return

    min_times = np.min([np.min(v) if len(v) else np.inf for v in all_times.values()])
    max_times = np.max([np.max(v) if len(v) else -np.inf for v in all_times.values()])
    grid_times = np.logspace(
        np.floor(np.log10(min_times)), np.ceil(np.log10(max_times)), 100
    )
    perfprofiles = {
        k: [np.sum(v <= grid_time) / len(v) for grid_time in grid_times]
        for k, v in all_times.items()
    }

    if save:
        print("Saving...")
        name = (
            "_".join([str(v) for v in config["dataset"].values()]) + ".csv"
        )
        save_data = pd.DataFrame({"grid_times": grid_times})
        for k, v in perfprofiles.items():
            save_data[k] = v
        path = pathlib.Path(__file__).parent.joinpath("saves", name)
        save_data.to_csv(path, index=False)
    else:
        print("Plotting...")
        fig, axs = plt.subplots(1, 1)
        for solver_name, perfprofile in perfprofiles.items():
            axs.plot(
                grid_times,
                perfprofile,
                label=solver_name,
                color=get_solver_name_color(solver_name),
            )
        axs.set_xscale("log")
        axs.grid(visible=True, which="major", axis="both")
        axs.grid(visible=True, which="minor", axis="both", alpha=0.2)
        axs.minorticks_on()
        axs.set_xlabel("Time")
        axs.set_ylabel("Prop. solved")
        axs.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("setup_path")
    parser.add_argument("-s", "--save", action="store_true")
    args = parser.parse_args()

    figure(args.setup_path, save=args.save)
