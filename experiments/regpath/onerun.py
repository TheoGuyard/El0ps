import argparse
import os
import pathlib
import pickle
import sys
import yaml
import numpy as np
from datetime import datetime
from el0ps.problem import Problem, compute_lmbd_max

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.instances import get_data  # noqa
from utils.solvers import get_solver, precompile, can_handle  # noqa


def onerun(config_path):
    print("Preprocessing...")
    base_dir = pathlib.Path(__file__).parent.absolute()
    result_dir = pathlib.Path(base_dir, "results")
    result_file = "{}.pickle".format(
        datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
    )
    result_path = pathlib.Path(base_dir, result_dir, result_file)
    config_path = pathlib.Path(config_path)

    assert result_dir.is_dir()
    assert config_path.is_file()
    assert not result_path.is_file()

    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    print("Generating data...")
    datafit, penalty, A, lmbd, x_true = get_data(config["dataset"])

    print("Precompiling...")
    precomilation_problem = Problem(datafit, penalty, A, lmbd)
    for solver_name in config["solver_names"]:
        solver = get_solver(solver_name, config["solver_options"])
        if can_handle(
            solver_name,
            config["dataset"]["datafit_name"],
            config["dataset"]["penalty_name"],
        ):
            precompile(precomilation_problem, solver)

    print("Running...")
    lmbd_ratio_grid = np.logspace(
        np.log10(config["path_options"]["lmbd_ratio_max"]),
        np.log10(config["path_options"]["lmbd_ratio_min"]),
        config["path_options"]["lmbd_ratio_num"],
    )
    results = {
        solver_name: {i: None for i in range(lmbd_ratio_grid.size)}
        for solver_name in config["solver_names"]
    }
    for solver_name in config["solver_names"]:
        solver = get_solver(solver_name, config["solver_options"])
        if can_handle(
            solver_name,
            config["dataset"]["datafit_name"],
            config["dataset"]["penalty_name"],
        ):
            print("  Solver: {}".format(solver_name))
            x_init = np.zeros(A.shape[1])
            lmbd_max = compute_lmbd_max(datafit, penalty, A)
            for i, lmbd_ratio in enumerate(lmbd_ratio_grid):
                print(
                    "    Lambda ratio: {:.2e} ({}/{})...".format(
                        lmbd_ratio, i + 1, lmbd_ratio_grid.size
                    )
                )
                problem = Problem(datafit, penalty, A, lmbd_ratio * lmbd_max)
                result = solver.solve(problem, x_init=x_init)
                x_init = np.copy(result.x)
                print("      Status    : {}".format(result.status.value))
                print(
                    "      Solve time: {:.6f} seconds".format(
                        result.solve_time
                    )
                )
                print(
                    "      Objective : {:.6f}".format(result.objective_value)
                )
                print(
                    "      Non-zeros : {:d}".format(
                        int(np.round(np.sum(result.z)))
                    )
                )
                results[solver_name][i] = {
                    "status": result.status,
                    "solve_time": result.solve_time,
                    "objective_value": result.objective_value,
                }
        else:
            print("  Skipping {}...".format(solver_name))
            results[solver_name][i] = None

    print("Saving results...")
    with open(result_path, "wb") as file:
        data = {"config": config, "results": results}
        pickle.dump(data, file)

    print("  File name: {}".format(result_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    onerun(args.config_path)
