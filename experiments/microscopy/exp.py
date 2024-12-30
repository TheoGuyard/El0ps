import argparse
import numpy as np
import pathlib
from expflow import Experiment, Runner
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

        data = np.load(pathlib.Path(__file__).parent.joinpath("exp.npz"))
        A = data["X"]
        y = data["y"]

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

        return result

    def cleanup(self) -> None:
        pass

    def plot(self, results: list[dict]) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["run", "plot"])
    parser.add_argument("--config_path", "-c", type=str)
    parser.add_argument("--results_dir", "-r", type=str)
    parser.add_argument("--repeats", "-n", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    runner = Runner(verbose=args.verbose)

    if args.command == "run":
        runner.run(
            Microscopy, args.config_path, args.results_dir, args.repeats
        )
    elif args.command == "plot":
        runner.plot(Microscopy, args.config_path, args.results_dir)
    else:
        raise ValueError(f"Unknown command {args.command}.")
