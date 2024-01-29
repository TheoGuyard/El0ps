import argparse
import pathlib
import pickle
import os
import sys
import yaml
from copy import deepcopy
from datetime import datetime
from el0ps.problem import Problem


sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
from experiments.instances import get_data, calibrate_objective  # noqa
from experiments.solvers import get_solver, can_handle  # noqa


def onerun(config_path, nosave=False):
    print("Performance profile experiment")
    print()

    config_path = pathlib.Path(config_path)
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    print("Loading data...")
    A, y, x_true = get_data(config["dataset"])
    print("  A shape: {}".format(A.shape))
    print("  y shape: {}".format(y.shape))
    print("  x shape: {}".format(None if x_true is None else x_true.shape))
    print()

    print("Calibrating parameters...")
    datafit, penalty, lmbd, _ = calibrate_objective(
        config["dataset"]["datafit_name"],
        config["dataset"]["penalty_name"],
        A,
        y,
        x_true,
    )
    print()

    problem = Problem(datafit, penalty, A, lmbd)
    print(problem)

    print("Precompiling...")
    solver_opts = deepcopy(config["solvers"]["solvers_opts"])
    solver_opts["time_limit"] = 5.0
    solver = get_solver("el0ps", solver_opts)
    solver.solve(problem)
    print()

    results = {}
    for solver_name in config["solvers"]["solvers_name"]:
        solver = get_solver(solver_name, config["solvers"]["solvers_opts"])
        print("Running {}...".format(solver_name))
        result = solver.solve(problem)
        results[solver_name] = result
        print(result)
        print()

    if not nosave:
        print("Saving results...")
        base_dir = pathlib.Path(__file__).parent.absolute()
        result_dir = pathlib.Path(base_dir, "results")
        result_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        result_file = "{}.pickle".format(result_uuid)
        result_path = pathlib.Path(base_dir, result_dir, result_file)
        with open(result_path, "wb") as file:
            data = {"config": config, "results": results}
            pickle.dump(data, file)
        print("  File name : {}".format(result_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--nosave", action="store_true")
    args = parser.parse_args()
    onerun(args.config_path, nosave=args.nosave)
