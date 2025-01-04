import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from exprun import Experiment, Runner
from el0ps.compilation import CompilableClass, compiled_clone
from el0ps.path import Path

from experiments.solver import (
    get_solver,
    can_handle_instance,
    can_handle_compilation,
    precompile_solver,
)
from experiments.instance import calibrate_parameters


class Microscopy(Experiment):

    def setup(self) -> None:

        A = np.load(pathlib.Path(__file__).parent.joinpath("A.npy"))
        y = np.load(pathlib.Path(__file__).parent.joinpath("y.npy"))

        datafit, penalty, lmbd, x_l0learn = calibrate_parameters(
            "Leastsquares",
            self.config["penalty"],
            A,
            y,
        )

        self.x_l0learn = x_l0learn
        self.datafit = datafit
        self.penalty = penalty
        self.A = A
        self.lmbd = lmbd

        if isinstance(self.datafit, CompilableClass):
            self.datafit_compiled = compiled_clone(self.datafit)
        else:
            self.datafit_compiled = None
        if isinstance(self.penalty, CompilableClass):
            self.penalty_compiled = compiled_clone(self.penalty)
        else:
            self.penalty_compiled = None

    def run(self) -> dict:
        result = {}
        for solver_name, solver_keys in self.config["solvers"].items():
            if can_handle_instance(
                solver_keys["solver"],
                solver_keys["params"],
                str(self.datafit),
                str(self.penalty),
            ):
                print("Running {}...".format(solver_name))
                solver = get_solver(
                    solver_keys["solver"],
                    solver_keys["params"],
                )
                path = Path(**self.config["path_opts"])
                if can_handle_compilation(solver_keys["solver"]):
                    precompile_solver(
                        solver,
                        self.datafit_compiled,
                        self.penalty_compiled,
                        self.A,
                        self.lmbd,
                    )
                    result[solver_name] = path.fit(
                        solver,
                        self.datafit_compiled,
                        self.penalty_compiled,
                        self.A,
                    )
                else:
                    result[solver_name] = path.fit(
                        solver,
                        self.datafit,
                        self.penalty,
                        self.A,
                    )
                del result[solver_name]["x"]
            else:
                print("Skipping {}".format(solver_name))
                result[solver_name] = None

        return result

    def cleanup(self) -> None:
        pass

    def plot(self, results: list) -> None:

        lgrid = np.logspace(
            np.log10(self.config["path_opts"]["lmbd_max"]),
            np.log10(self.config["path_opts"]["lmbd_min"]),
            self.config["path_opts"]["lmbd_num"],
        )

        stats = {
            "solve_time": {},
            "objective_value": {},
            "datafit_value": {},
            "n_nnz": {},
        }
        plot_opts = {
            "solve_time": {"scale": "log", "normalize": False},
            "objective_value": {"scale": "linear", "normalize": True},
            "datafit_value": {"scale": "linear", "normalize": True},
            "n_nnz": {"scale": "linear", "normalize": False},
        }

        for result in results:
            for solver_name, solver_result in result.items():
                for stat_name, stat_value in stats.items():
                    if solver_name not in stat_value:
                        stat_value[solver_name] = [[] for _ in lgrid]
                    if solver_result is not None:
                        for i in range(len(lgrid)):
                            if len(solver_result[stat_name]) > i:
                                stat_value[solver_name][i].append(
                                    solver_result[stat_name][i]
                                )
                            else:
                                stat_value[solver_name][i].append(np.nan)

        curves = {}
        for stat_name, stat_values in stats.items():
            curves[stat_name] = {}
            for solver_name, solver_values in stat_values.items():
                curves[stat_name][solver_name] = []
                for i in range(len(lgrid)):
                    curves[stat_name][solver_name].append(
                        np.nanmean(solver_values[i])
                    )

        for stat_name, stat_values in curves.items():
            if plot_opts[stat_name]["normalize"]:
                for solver_name, solver_values in stat_values.items():
                    curves[stat_name][solver_name] = np.array(
                        curves[stat_name][solver_name]
                    ) / max(curves[stat_name][solver_name])

        _, axs = plt.subplots(1, len(curves), squeeze=False)
        for i, (stat_name, stat_curve) in enumerate(curves.items()):
            for solver_name, solver_curve in stat_curve.items():
                if plot_opts[stat_name]["normalize"]:
                    stat_label = solver_name + " (normalized)"
                else:
                    stat_label = solver_name
                axs[0, i].plot(lgrid, solver_curve, label=solver_name)
            axs[0, i].set_xscale("log")
            axs[0, i].set_yscale(plot_opts[stat_name]["scale"])
            axs[0, i].invert_xaxis()
            axs[0, i].set_xlabel("lmbd/lmbd_max")
            axs[0, i].set_ylabel(stat_label)
        axs[0, 0].legend()

        plt.show()

        table = {"lgrid": lgrid}
        for stat_name, stat_values in curves.items():
            for solver_name, solver_values in stat_values.items():
                table[solver_name + "_" + stat_name] = solver_values

        return table

    def save_plot(self, table, save_dir):
        name = "{}-{}={}-{}={:.2e}-{}={:.2e}-{}={:d}".format(
            "microscopy",
            "penalty",
            self.config["penalty"],
            "lmbd_max",
            self.config["path_opts"]["lmbd_max"],
            "lmbd_min",
            self.config["path_opts"]["lmbd_min"],
            "lmbd_num",
            self.config["path_opts"]["lmbd_num"],
        )
        df = pd.DataFrame(table)
        df.to_csv(
            pathlib.Path(save_dir).joinpath(name).with_suffix(".csv"),
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["run", "plot"])
    parser.add_argument("--config_path", "-c", type=str)
    parser.add_argument("--result_dir", "-r", type=str)
    parser.add_argument("--save_dir", "-s", type=str, default=None)
    parser.add_argument("--repeats", "-n", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    runner = Runner(verbose=args.verbose)

    if args.command == "run":
        runner.run(
            Microscopy,
            args.config_path,
            args.result_dir,
            args.repeats,
        )
    elif args.command == "plot":
        runner.plot(
            Microscopy,
            args.config_path,
            args.result_dir,
            args.save_dir,
        )
    else:
        raise ValueError(f"Unknown command {args.command}.")
