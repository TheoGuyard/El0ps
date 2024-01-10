import argparse
import pathlib
import pickle
import pprint
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


def onerun_perfprofile(config_path, nosave=False):
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
    datafit, penalty, lmbd, x_cal = calibrate_objective(
        config["dataset"]["datafit_name"],
        config["dataset"]["penalty_name"],
        A,
        y,
        x_true,
    )
    print("  num nz: {}".format(np.count_nonzero(x_cal)))
    print("  lratio: {}".format(lmbd / compute_lmbd_max(datafit, penalty, A)))
    for param_name, param_value in penalty.params_to_dict().items():
        print("  {}\t: {}".format(param_name, param_value))
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
        if can_handle(
            solver_name,
            config["dataset"]["datafit_name"],
            config["dataset"]["penalty_name"],
        ):
            print("Running {}...".format(solver_name))
            result = solver.solve(problem)
            print("  status    : {}".format(result.status))
            print("  obj. value: {}".format(result.objective_value))
            print("  non-zeros : {}".format(result.n_nnz))
            print("  solve time: {}".format(result.solve_time))
            print("  iter count: {}".format(result.iter_count))
        else:
            print("Skipping {}".format(solver_name))
            result = None
        results[solver_name] = result

    if not nosave:
        print("Saving results...")
        base_dir = pathlib.Path(__file__).parent.absolute()
        result_dir = pathlib.Path(base_dir, "results")
        result_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        result_file = "{}_{}.pickle".format(config["expname"], result_uuid)
        result_path = pathlib.Path(base_dir, result_dir, result_file)
        with open(result_path, "wb") as file:
            data = {"config": config, "results": results}
            pickle.dump(data, file)
        print("  File name: {}".format(result_file))
    else:
        print("No results to save")

def onerun_sensibility(config_path, nosave=False):
    config_path = pathlib.Path(config_path)
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    print("Running experiment...")
    results = {
        param_name: {
            param_value: {
                solver_name: [] for solver_name in config["solvers"]["solvers_name"]
            } for param_value in param_values
        } for param_name, param_values in config["dataset"]["variations"].items()
    }
    for param_name, param_values in config["dataset"]["variations"].items():
        for param_value in param_values:
            print("  {} = {}".format(param_name, param_value))
            
            print("    Loading data...")
            config_dataset = deepcopy(config["dataset"]["base"])
            config_dataset["dataset_opts"][param_name] = param_value
            A, y, x_true = get_data(config_dataset)
            print("      datafit: {}".format(config_dataset["datafit_name"]))
            print("      penalty: {}".format(config_dataset["penalty_name"]))
            print("      A shape: {}".format(A.shape))
            print("      y shape: {}".format(y.shape))
            print("      x shape: {}".format(None if x_true is None else x_true.shape))
            
            print("    Calibrating parameters...")
            datafit, penalty, lmbd, _ = calibrate_objective(
                config_dataset["datafit_name"],
                config_dataset["penalty_name"],
                A,
                y,
                x_true,
            )
            print("      lratio: {}".format(lmbd / compute_lmbd_max(datafit, penalty, A)))
            for penalty_param_name, penalty_param_value in penalty.params_to_dict().items():
                print("      {}\t: {}".format(penalty_param_name, penalty_param_value))
            
            problem = Problem(datafit, penalty, A, lmbd)
            solvers = {
                solver_name: get_solver(solver_name, config["solvers"]["solvers_opts"])
                for solver_name in config["solvers"]["solvers_name"]
            }

            for solver_name, solver in solvers.items():
                if can_handle(
                    solver_name,
                    config_dataset["datafit_name"],
                    config_dataset["penalty_name"],
                ):
                    print("    Running {}".format(solver_name))
                    print("      Precompiling...")
                    solver.options.time_limit = 5.0
                    result = solver.solve(problem)
                    solver.options.time_limit = config["solvers"]["solvers_opts"]["time_limit"]
                    print("      Solving problem...")
                    result = solver.solve(problem)
                    print("        status    : {}".format(result.status))
                    print("        obj. value: {}".format(result.objective_value))
                    print("        non-zeros : {}".format(result.n_nnz))
                    print("        solve time: {}".format(result.solve_time))
                    print("        iter count: {}".format(result.iter_count))
                else:
                    print("    Skipping {}".format(solver_name))
                    result = None
                results[param_name][param_value][solver_name] = result
            print()

    if not nosave:
        print("Saving results...")
        base_dir = pathlib.Path(__file__).parent.absolute()
        result_dir = pathlib.Path(base_dir, "results")
        result_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        result_file = "{}_{}.pickle".format(config["expname"], result_uuid)
        result_path = pathlib.Path(base_dir, result_dir, result_file)
        with open(result_path, "wb") as file:
            data = {"config": config, "results": results}
            pickle.dump(data, file)
        print("  File name: {}".format(result_file))
    else:
        print("No results to save")

def onerun_regpath(config_path, nosave=False):
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
    datafit, penalty, lmbd, x_cal = calibrate_objective(
        config["dataset"]["datafit_name"],
        config["dataset"]["penalty_name"],
        A,
        y,
        x_true,
    )
    print("  num nz: {}".format(np.count_nonzero(x_cal)))
    print("  lratio: {}".format(lmbd / compute_lmbd_max(datafit, penalty, A)))
    for param_name, param_value in penalty.params_to_dict().items():
        print("  {}\t: {}".format(param_name, param_value))
    print()

    problem = Problem(datafit, penalty, A, lmbd)
    print(problem)

    print("Precompiling...")
    solver_opts = deepcopy(config["solvers"]["solvers_opts"])
    solver_opts["time_limit"] = 5.0
    solver = get_solver("el0ps", solver_opts)
    solver.solve(problem)
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
            path = Path(**config["path_opts"])
            result = path.fit(solver, datafit, penalty, A)
            del result["x"]
        else:
            print("  Skipping {}".format(solver_name))
            result = None
        results[solver_name] = result

    if not nosave:
        print("Saving results...")
        base_dir = pathlib.Path(__file__).parent.absolute()
        result_dir = pathlib.Path(base_dir, "results")
        result_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        result_file = "{}_{}.pickle".format(config["expname"], result_uuid)
        result_path = pathlib.Path(base_dir, result_dir, result_file)
        with open(result_path, "wb") as file:
            data = {"config": config, "results": results}
            pickle.dump(data, file)
        print("  File name: {}".format(result_file))
    else:
        print("No results to save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["perfprofile", "sensibility", "regpath"])
    parser.add_argument("config_path")
    parser.add_argument("--nosave", action="store_true")
    args = parser.parse_args()
    if args.experiment == "perfprofile":
        onerun_perfprofile(args.config_path, nosave=args.nosave)
    elif args.experiment == "sensibility":
        onerun_sensibility(args.config_path, nosave=args.nosave)
    elif args.experiment == "regpath":
        onerun_regpath(args.config_path, nosave=args.nosave)
    else:
        raise NotImplementedError
