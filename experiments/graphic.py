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


def graphic_perfprofile(config_path, save=False):
    config_path = pathlib.Path(config_path)
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    print("Recovering results...")
    results_dir = pathlib.Path(__file__).parent.absolute().joinpath("results")

    solve_times = {s: [] for s in config["solvers"]["solvers_name"]}
    solve_nodes = {s: [] for s in config["solvers"]["solvers_name"]}
    # supps_times = {s: [] for s in config["solvers"]["solvers_name"]}
    # supps_nodes = {s: [] for s in config["solvers"]["solvers_name"]}
    # relax_times = {s: [] for s in config["solvers"]["solvers_name"]}
    # depth_nodes = {s: [] for s in config["solvers"]["solvers_name"]}

    found = 0
    match = 0
    empty = 0
    notcv = 0
    for result_path in results_dir.glob("perfprofile_*.pickle"):
        found += 1
        with open(result_path, "rb") as file:
            file_data = pickle.load(file)
            if file_data["config"] == config:
                match += 1
                if not any(file_data["results"].values()):
                    empty += 1
                    continue
                for solver_name, result in file_data["results"].items():
                    if result is not None:
                        if result.status == Status.OPTIMAL:
                            solve_times[solver_name].append(result.solve_time)
                            solve_nodes[solver_name].append(result.iter_count)
                            # supps_times[solver_name] += list(
                            #     zip(
                            #         result.trace["solve_time"],
                            #         result.trace["supp_left"],
                            #     )
                            # )
                            # supps_nodes[solver_name] += list(
                            #     zip(
                            #         result.trace["iter_count"],
                            #         result.trace["supp_left"],
                            #     )
                            # )
                            # relax_times[solver_name] += result.trace[
                            #     "node_time_lower_bound"
                            # ]
                            # depth_nodes[solver_name] += result.trace[
                            #     "node_depth"
                            # ]
                        else:
                            notcv += 1

    print("  {} files founds".format(found))
    print("  {} files matched".format(match))
    print("  {} empty results".format(empty))
    print("  {} not converged".format(notcv))

    if (match == 0) or (match == empty):
        return

    print("Computing statistics...")
    grid_solve_times = np.logspace(
        np.floor(np.log10(np.min([np.min(v) for v in solve_times.values()]))),
        np.ceil(np.log10(np.max([np.max(v) for v in solve_times.values()]))),
        100,
    )
    profile_solve_times = {
        solver_name: [np.sum(stats <= g) for g in grid_solve_times]
        for solver_name, stats in solve_times.items()
    }

    # grid_supps_times = np.logspace(
    #     np.floor(
    #         np.log10(
    #             np.min(
    #                 [
    #                     np.min([st for (st, _) in v])
    #                     for v in supps_times.values()
    #                 ]
    #             )
    #         )
    #     ),
    #     np.ceil(
    #         np.log10(
    #             np.max(
    #                 [
    #                     np.max([st for (st, _) in v])
    #                     for v in supps_times.values()
    #                 ]
    #             )
    #         )
    #     ),
    #     100,
    # )
    # profile_supps_times = {}
    # for solver_name, stats in supps_times.items():
    #     profile_supps_times[solver_name] = []
    #     for g in grid_supps_times:
    #         stat = [sl for (st, sl) in stats if st <= g]
    #         if len(stat) == 0:
    #             profile_supps_times[solver_name].append(1.0)
    #         elif len(stat) == len(stats):
    #             profile_supps_times[solver_name].append(0.0)
    #         else:
    #             profile_supps_times[solver_name].append(np.mean(stat))

    grid_solve_nodes = np.logspace(
        np.floor(np.log10(np.min([np.min(v) for v in solve_nodes.values()]))),
        np.ceil(np.log10(np.max([np.max(v) for v in solve_nodes.values()]))),
        100,
    )
    profile_solve_nodes = {
        solver_name: [np.sum(stats <= g) for g in grid_solve_nodes]
        for solver_name, stats in solve_nodes.items()
    }

    # grid_supps_nodes = np.logspace(
    #     np.floor(
    #         np.log10(
    #             np.min(
    #                 [
    #                     np.min([nc for (nc, _) in v])
    #                     for v in supps_nodes.values()
    #                 ]
    #             )
    #         )
    #     ),
    #     np.ceil(
    #         np.log10(
    #             np.max(
    #                 [
    #                     np.max([nc for (nc, _) in v])
    #                     for v in supps_nodes.values()
    #                 ]
    #             )
    #         )
    #     ),
    #     100,
    # )
    # profile_supps_nodes = {}
    # for solver_name, stats in supps_nodes.items():
    #     profile_supps_nodes[solver_name] = []
    #     for g in grid_supps_nodes:
    #         stat = [sl for (nc, sl) in stats if nc <= g]
    #         if len(stat) == 0:
    #             profile_supps_nodes[solver_name].append(1.0)
    #         elif len(stat) == len(stats):
    #             profile_supps_nodes[solver_name].append(0.0)
    #         else:
    #             profile_supps_nodes[solver_name].append(np.mean(stat))

    # grid_relax_times = np.logspace(
    #     np.floor(np.log10(np.min([np.min(v) for v in relax_times.values()]))),  # noqa 501
    #     np.ceil(np.log10(np.max([np.max(v) for v in relax_times.values()]))),
    #     100,
    # )
    # profile_relax_times = {
    #     solver_name: [np.mean(stats <= g) for g in grid_relax_times]
    #     for solver_name, stats in relax_times.items()
    # }

    # grid_depth_nodes = np.array(
    #     range(
    #         np.min([np.min(v) for v in depth_nodes.values()]),
    #         np.max([np.max(v) for v in depth_nodes.values()]) + 1,
    #         1,
    #     )
    # )
    # profile_depth_nodes = {
    #     solver_name: [np.mean(stats <= g) for g in grid_depth_nodes]
    #     for solver_name, stats in depth_nodes.items()
    # }

    if save:
        print("Saving data...")
        save_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        stats = {
            "solve_times": {
                "grid": grid_solve_times,
                "profile": profile_solve_times,
            },
            "solve_nodes": {
                "grid": grid_solve_nodes,
                "profile": profile_solve_nodes,
            },
            # "supps_times": {
            #     "grid": grid_supps_times,
            #     "profile": profile_supps_times,
            # },
            # "supps_nodes": {
            #     "grid": grid_supps_nodes,
            #     "profile": profile_supps_nodes,
            # },
            # "relax_times": {
            #     "grid": grid_relax_times,
            #     "profile": profile_relax_times,
            # },
            # "depth_nodes": {
            #     "grid": grid_depth_nodes,
            #     "profile": profile_depth_nodes,
            # },
        }
        for stat_name, stat_vars in stats.items():
            table = pd.DataFrame({"grid": stat_vars["grid"]})
            for solver_name in config["solvers"]["solvers_name"]:
                table[solver_name] = stat_vars["profile"][solver_name]
            file_name = "{}_{}_{}".format(
                config["expname"], stat_name, save_uuid
            )
            save_path = pathlib.Path(__file__).parent.joinpath(
                "saves", "{}.csv".format(file_name)
            )
            info_path = pathlib.Path(__file__).parent.joinpath(
                "saves", "{}.yaml".format(file_name)
            )
            table.to_csv(save_path, index=False)
            with open(info_path, "w") as file:
                yaml.dump(config, file)
    else:
        print("Plotting figure...")
        _, axs = plt.subplots(1, 6, squeeze=False)
        for solver_name in config["solvers"]["solvers_name"]:
            axs[0, 0].plot(
                grid_solve_times,
                profile_solve_times[solver_name],
                label=solver_name,
            )
            # axs[0, 1].plot(
            #     grid_supps_times,
            #     profile_supps_times[solver_name],
            #     label=solver_name,
            # )
            axs[0, 2].plot(
                grid_solve_nodes,
                profile_solve_nodes[solver_name],
                label=solver_name,
            )
            # axs[0, 3].plot(
            #     grid_supps_nodes,
            #     profile_supps_nodes[solver_name],
            #     label=solver_name,
            # )
            # axs[0, 4].plot(
            #     grid_relax_times,
            #     profile_relax_times[solver_name],
            #     label=solver_name,
            # )
            # axs[0, 5].plot(
            #     grid_depth_nodes,
            #     profile_depth_nodes[solver_name],
            #     label=solver_name,
            # )
        for ax in axs.flatten():
            ax.grid(visible=True, which="major", axis="both")
            ax.grid(visible=True, which="minor", axis="both", alpha=0.2)
            ax.minorticks_on()
        for i in [0, 1, 4]:
            axs[0, i].set_xscale("log")
            axs[0, i].set_xlabel("Time")
        for i in [2, 3]:
            axs[0, i].set_xscale("log")
            axs[0, i].set_xlabel("Nodes")
        for i in [0, 2]:
            axs[0, 0].set_ylabel("Inst. solved")
        for i in [1, 3]:
            axs[0, 0].set_ylabel("Supp. pruned")
        axs[0, 4].set_ylabel("Prop. relax time")
        axs[0, 5].set_xscale("log")
        axs[0, 5].set_xlabel("Depth")
        axs[0, 5].set_ylabel("Prop. depth")
        axs[0, 0].legend()
        plt.show()


def graphic_regpath(config_path, save=False):
    print("Preprocessing...")
    base_dir = pathlib.Path(__file__).parent.absolute()
    result_dir = pathlib.Path(base_dir, "results")
    config_path = pathlib.Path(config_path)
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    print("Recovering results...")
    lmbd_ratio_grid = np.logspace(
        np.log10(config["path_opts"]["lmbd_ratio_max"]),
        np.log10(config["path_opts"]["lmbd_ratio_min"]),
        config["path_opts"]["lmbd_ratio_num"],
    )
    stats_specs = {
        "solve_time": {"log": True, "ratio": None},
        "objective_value": {"log": False, "ratio": None},
        "datafit_value": {"log": False, "ratio": None},
        "n_nnz": {"log": False, "ratio": None},
    }
    stats = {
        stat_key: {
            solver_name: {i: [] for i in range(lmbd_ratio_grid.size)}
            for solver_name in config["solvers"]["solvers_name"]
        }
        for stat_key in stats_specs.keys()
    }

    found = 0
    match = 0
    empty = 0
    notcv = 0
    for result_path in result_dir.glob("regpath_*.pickle"):
        found += 1
        with open(result_path, "rb") as file:
            file_data = pickle.load(file)
            if (
                file_data["config"]["expname"] == config["expname"]
                and file_data["config"]["dataset"] == config["dataset"]
                and file_data["config"]["solvers"]["solvers_opts"]
                == config["solvers"]["solvers_opts"]
                and file_data["config"]["path_opts"] == config["path_opts"]
            ):
                match += 1
                if not any(file_data["results"].values()):
                    empty += 1
                    continue
                for solver_name, result in file_data["results"].items():
                    if result is not None:
                        for i in range(lmbd_ratio_grid.size):
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

    print("  {} files founds".format(found))
    print("  {} files matched".format(match))
    print("  {} empty results".format(empty))
    print("  {} solvers not converged".format(notcv))
    for k, v in stats["solve_time"].items():
        print("{} : {}".format(k, len(v[0])))

    if (match == 0) or (match == empty):
        return

    mean_stats = {
        stat_key: {
            solver_key: [
                np.mean(solver_values[i]) if len(solver_values[i]) else np.nan
                for i in range(lmbd_ratio_grid.size)
            ]
            for solver_key, solver_values in stat_values.items()
        }
        for stat_key, stat_values in stats.items()
    }

    if save:
        print("Saving...")
        save_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        save_file = "{}_{}.csv".format(config["expname"], save_uuid)
        info_file = "{}_{}.yaml".format(config["expname"], save_uuid)
        table = pd.DataFrame({"lmbd_ratio_grid": lmbd_ratio_grid})
        for stat_name, stat_values in mean_stats.items():
            for solver_name, solver_values in stat_values.items():
                table[solver_name + "_" + stat_name] = solver_values
        save_path = pathlib.Path(__file__).parent.joinpath("saves", save_file)
        info_path = pathlib.Path(__file__).parent.joinpath("saves", info_file)
        table.to_csv(save_path, index=False)
        with open(info_path, "w") as file:
            yaml.dump(config, file)
    else:
        print("Plotting...")
        _, axs = plt.subplots(1, len(mean_stats), squeeze=False)
        for i, (stat_name, stat_values) in enumerate(mean_stats.items()):
            if stats_specs[stat_name]["ratio"] is not None:
                reference = stat_values[stats_specs[stat_name]["ratio"]]
            else:
                reference = 1.0
            for solver_name, solver_values in stat_values.items():
                axs[0, i].plot(
                    lmbd_ratio_grid,
                    np.divide(solver_values, reference),
                    label=solver_name,
                )
            axs[0, i].set_xscale("log")
            axs[0, i].grid(visible=True, which="major", axis="both")
            axs[0, i].grid(visible=True, which="minor", axis="both", alpha=0.2)
            axs[0, i].minorticks_on()
            axs[0, i].set_xlabel("lmbd/lmbd_max")
            axs[0, i].set_ylabel(stat_name)
            axs[0, i].invert_xaxis()
            if stats_specs[stat_name]["log"]:
                axs[0, i].set_yscale("log")
        axs[0, 0].legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task", choices=["perfprofile", "sensibility", "regpath"]
    )
    parser.add_argument("config_path")
    parser.add_argument("-s", "--save", action="store_true")
    args = parser.parse_args()
    if args.task == "perfprofile":
        graphic_perfprofile(args.config_path, save=args.save)
    elif args.task == "regpath":
        graphic_regpath(args.config_path, save=args.save)
    else:
        raise NotImplementedError
