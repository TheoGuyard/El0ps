import argparse
import pathlib
import pickle
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from el0ps.solver import Status


def get_solver_name_color(solver_name):
    if solver_name == "mosek":
        return "royalblue"
    elif solver_name == "cplex":
        return "skyblue"
    elif solver_name == "gurobi":
        return "forestgreen"
    elif solver_name == "l0bnb":
        return "orange"
    elif solver_name == "sbnb":
        return "orangered"
    elif solver_name.startswith("el0ps"):
        return "darkred"
    else:
        return None


def plot_perfprofile(config_path, save=False):
    print("Extracting config...")
    base_dir = pathlib.Path(__file__).parent.absolute()
    results_dir = pathlib.Path(base_dir, "results")
    config_path = pathlib.Path(config_path)
    assert results_dir.is_dir()
    assert config_path.is_file()
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    assert config["task"]["task_type"] == "solve"

    print("Recovering results...")
    all_solve_time = {
        solver_name: [] for solver_name in config["solvers"]["solvers_name"]
    }
    found = 0
    matched = 0
    errored = 0
    notcved = 0
    for result_path in results_dir.glob("*.pickle"):
        found += 1
        try:
            with open(result_path, "rb") as file:
                file_data = pickle.load(file)
                if (
                    file_data["config"]["expname"] != config["expname"]
                    or file_data["config"]["dataset"] != config["dataset"]
                    or file_data["config"]["solvers"]["solvers_opts"]
                    != config["solvers"]["solvers_opts"]
                    or file_data["config"]["task"] != config["task"]
                ):
                    continue
                for solver_name, result in file_data["results"].items():
                    if solver_name in all_solve_time.keys():
                        if result is not None:
                            if result.status == Status.OPTIMAL:
                                all_solve_time[solver_name].append(
                                    result.solve_time
                                )
                            else:
                                print(
                                    "  {} not converged: {}".format(
                                        solver_name, result.status
                                    )
                                )
                                notcved += 1
                matched += 1
        except Exception:
            errored += 1

    print("  {} files founds".format(found))
    print("  {} files matched".format(matched))
    for solver_name, values in all_solve_time.items():
        print(f"    {len(values)} datapoints for {solver_name}")
    print("  {} files errored".format(errored))
    print("  {} not converged".format(notcved))

    if matched == 0:
        return

    min_times = np.min(
        [np.min(v) if len(v) else np.inf for v in all_solve_time.values()]
    )
    max_times = config["solvers"]["solvers_opts"]["time_limit"]
    grid_times = np.logspace(
        np.floor(np.log10(min_times)),
        np.ceil(np.log10(max_times)),
        100,
    )
    perfprofiles = {
        k: [
            0.0 if len(v) == 0 else np.sum(v <= grid_time) / len(v)
            for grid_time in grid_times
        ]
        for k, v in all_solve_time.items()
    }

    if save:
        print("Saving data...")
        name = "_".join([str(v) for v in config["dataset"].values()]) + ".csv"
        save_data = pd.DataFrame({"grid_times": grid_times})
        for k, v in perfprofiles.items():
            save_data[k] = v
        path = pathlib.Path(__file__).parent.joinpath("saves", name)
        save_data.to_csv(path, index=False)
    else:
        print("Plotting figure...")
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


def plot_regpath(config_path, save=False):
    print("Preprocessing...")
    base_dir = pathlib.Path(__file__).parent.absolute()
    result_dir = pathlib.Path(base_dir, "results")
    config_path = pathlib.Path(config_path)
    assert result_dir.is_dir()
    assert config_path.is_file()
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    assert config["task"]["task_type"] == "fitpath"

    print("Recovering results...")
    lmbd_ratio_grid = np.logspace(
        np.log10(config["task"]["task_opts"]["lmbd_ratio_max"]),
        np.log10(config["task"]["task_opts"]["lmbd_ratio_min"]),
        config["task"]["task_opts"]["lmbd_ratio_num"],
    )
    all_stat = {
        "solve_time": {"log": True},
        "node_count": {"log": True},
        "objective_value": {"log": False},
        "datafit_value": {"log": False},
        "penalty_value": {"log": False},
        "n_nnz": {"log": False},
    }
    all_data = {
        stat_name: {
            solver_name: {i: [] for i in range(lmbd_ratio_grid.size)}
            for solver_name in config["solvers"]["solvers_name"]
        }
        for stat_name in all_stat.keys()
    }

    found = 0
    matched = 0
    errored = 0
    notcved = 0
    for result_path in result_dir.glob("*.pickle"):
        found += 1
        try:
            with open(result_path, "rb") as file:
                file_data = pickle.load(file)
                if (
                    file_data["config"]["expname"] != config["expname"]
                    or file_data["config"]["dataset"] != config["dataset"]
                    or file_data["config"]["solvers"]["solvers_opts"]
                    != config["solvers"]["solvers_opts"]
                    or file_data["config"]["task"] != config["task"]
                ):
                    continue
                for solver_name, result in file_data["results"].items():
                    if result is not None:
                        if solver_name in config["solvers"]["solvers_name"]:
                            for stat_name in all_stat.keys():
                                for i in range(len(result["lmbd_ratio"])):
                                    if result["status"][i] == Status.OPTIMAL:
                                        all_data[stat_name][solver_name][
                                            i
                                        ].append(result[stat_name][i])
                                    else:
                                        print(
                                            "  {} not converged: {}".format(
                                                solver_name,
                                                result["status"][i],
                                            )
                                        )
                                        notcved += 1
                matched += 1
        except Exception as e:
            print(e)
            errored += 1

    print("  {} files founds".format(found))
    print("  {} files matched".format(matched))
    print("  {} files errored".format(errored))
    print("  {} solvers not converged".format(notcved))

    if matched == 0:
        return

    mean_data = {
        name: {
            solver_name: [
                np.mean(values[i]) if len(values[i]) > 0 else np.nan
                for i in range(lmbd_ratio_grid.size)
            ]
            for solver_name, values in data.items()
        }
        for name, data in all_data.items()
    }

    if save:
        print("Saving...")
        save_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        save_file = "{}_{}.csv".format(config["expname"], save_uuid)
        info_file = "{}_{}.yaml".format(config["expname"], save_uuid)
        table = pd.DataFrame({"lmbd_ratio_grid": lmbd_ratio_grid})
        for stat_name, stat_data in mean_data.items():
            for solver_name, values in stat_data.items():
                table[solver_name + "_" + stat_name] = values
        save_path = pathlib.Path(__file__).parent.joinpath("saves", save_file)
        info_path = pathlib.Path(__file__).parent.joinpath("saves", info_file)
        table.to_csv(save_path, index=False)
        with open(info_path, "w") as file:
            yaml.dump(config, file)
    else:
        print("Plotting...")
        plt.rcParams["axes.prop_cycle"] = plt.cycler(
            "color", plt.cm.tab20c.colors
        )
        fig, axs = plt.subplots(1, len(mean_data.keys()))
        for i, (stat_name, stat_data) in enumerate(mean_data.items()):
            for solver_name, values in stat_data.items():
                axs[i].plot(
                    lmbd_ratio_grid,
                    values,
                    label=solver_name,
                    color=get_solver_name_color(solver_name),
                )
            axs[i].set_xscale("log")
            axs[i].grid(visible=True, which="major", axis="both")
            axs[i].grid(visible=True, which="minor", axis="both", alpha=0.2)
            axs[i].minorticks_on()
            axs[i].set_xlabel("lmbd/lmbd_max")
            axs[i].set_ylabel(stat_name)
            axs[i].invert_xaxis()
            if all_stat[stat_name]["log"]:
                axs[i].set_yscale("log")
        axs[0].legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["perfprofile", "regpath"])
    parser.add_argument("config_path")
    parser.add_argument("-s", "--save", action="store_true")
    args = parser.parse_args()
    if args.task == "perfprofile":
        plot_perfprofile(args.config_path, save=args.save)
    elif args.task == "regpath":
        plot_regpath(args.config_path, save=args.save)
    else:
        raise NotImplementedError
