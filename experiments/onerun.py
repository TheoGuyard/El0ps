import argparse
import pathlib
import pickle
import os
import sys
import yaml
import numpy as np
from copy import deepcopy
from datetime import datetime
from el0ps.problem import Problem, compute_lmbd_max
from el0ps.path import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.instances import get_data, calibrate_objective  # noqa
from experiments.solvers import get_solver, can_handle  # noqa


def onerun(config_path, nosave=False):
    config_path = pathlib.Path(config_path)
    assert config_path.is_file()
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    print("Loading data...")
    A, y, x_true = get_data(config["dataset"])
    print("  A shape: {}".format(A.shape))
    print("  y shape: {}".format(y.shape))
    print("  x shape: {}".format(None if x_true is None else x_true.shape))
    print()

    print("Calibrating parameters...")
    datafit, penalty, lmbd, x_cal = calibrate_objective(
        config["dataset"]["datafit_name"],
        config["dataset"]["penalty_name"],
        A,
        y,
        x_true,
    )
    print("  num nz: {}".format(np.count_nonzero(x_cal)))
    print("  lambda: {}".format(lmbd))
    print("  lratio: {}".format(lmbd / compute_lmbd_max(datafit, penalty, A)))
    for param_name, param_value in penalty.params_to_dict().items():
        print("  {}\t: {}".format(param_name, param_value))
    print()

    problem = Problem(datafit, penalty, A, lmbd)
    print(problem)

    print("Precompiling...")
    for solver_name in config["solvers"]["solvers_name"]:
        if can_handle(
            solver_name,
            config["dataset"]["datafit_name"],
            config["dataset"]["penalty_name"],
        ):
            try:
                print("  Precompiling {}".format(solver_name))
                solver_opts = deepcopy(config["solvers"]["solvers_opts"])
                solver_opts["time_limit"] = 5.0
                solver = get_solver(solver_name, solver_opts)
                solver.solve(problem)
            except Exception as e:
                print("  Error: {}".format(e))
        else:
            print("  Skipping {}".format(solver_name))
    print()

    print("Running experiment...")
    results = {}
    for solver_name in config["solvers"]["solvers_name"]:
        solver = get_solver(solver_name, config["solvers"]["solvers_opts"])
        if can_handle(
            solver_name,
            config["dataset"]["datafit_name"],
            config["dataset"]["penalty_name"],
        ):
            print("  Running {}".format(solver_name))
            try:
                if config["task"]["task_type"] == "solve":
                    result = solver.solve(problem)
                elif config["task"]["task_type"] == "fitpath":
                    path = Path(**config["task"]["task_opts"])
                    result = path.fit(solver, datafit, penalty, A)
                    del result["x"]
                else:
                    raise ValueError("Unknown task type.")
            except Exception as e:
                print("    Error: {}".format(e))
                result = None
        else:
            print("  Skipping {}".format(solver_name))
            result = None
        results[solver_name] = result

    if any(results.values()) and not nosave:
        print("Saving results...")
        base_dir = pathlib.Path(__file__).parent.absolute()
        result_dir = pathlib.Path(base_dir, "results")
        result_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        result_file = "{}_{}.pickle".format(config["expname"], result_uuid)
        result_path = pathlib.Path(base_dir, result_dir, result_file)
        assert result_dir.is_dir()
        assert not result_path.is_file()
        with open(result_path, "wb") as file:
            data = {"config": config, "results": results}
            pickle.dump(data, file)
        print("  File name: {}".format(result_file))
    else:
        print("No results to save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--nosave", action="store_true")
    args = parser.parse_args()
    onerun(args.config_path, nosave=args.nosave)
