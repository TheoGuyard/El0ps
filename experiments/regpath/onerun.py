import argparse
import os
import pathlib
import pickle
import sys
import yaml
from datetime import datetime
from el0ps.problem import Problem, compute_lmbd_max
from el0ps.path import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.instances import get_data  # noqa
from utils.solvers import get_solver, precompile, can_handle  # noqa


def onerun(config_path):
    print("Preprocessing...")
    config_path = pathlib.Path(config_path)
    assert config_path.is_file()
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    print("Generating data...")
    datafit, penalty, A, lmbd, x_true = get_data(config["dataset"])

    print("Precompiling...")
    lmbd = 0.1 * compute_lmbd_max(datafit, penalty, A)
    problem = Problem(datafit, penalty, A, lmbd)
    for solver_name in config["solver_names"]:
        solver = get_solver(solver_name, config["solver_options"])
        if can_handle(
            solver_name,
            config["dataset"]["datafit_name"],
            config["dataset"]["penalty_name"],
        ):
            precompile(problem, solver)

    print("Running...")
    results = {solver_name: None for solver_name in config["solver_names"]}
    for solver_name in config["solver_names"]:
        solver = get_solver(solver_name, config["solver_options"])
        if can_handle(
            solver_name,
            config["dataset"]["datafit_name"],
            config["dataset"]["penalty_name"],
        ):
            print("  Solver: {}".format(solver_name))
            path = Path(
                lmbd_ratio_max=config["path_options"]["lmbd_ratio_max"],
                lmbd_ratio_min=config["path_options"]["lmbd_ratio_min"],
                lmbd_ratio_num=config["path_options"]["lmbd_ratio_num"],
                stop_if_not_optimal=config["path_options"][
                    "stop_if_not_optimal"
                ],
                verbose=True,
            )
            fit_data = path.fit(solver, datafit, penalty, A)
            del fit_data["x"]
            results[solver_name] = fit_data
        else:
            print("  Skipping {}...".format(solver_name))
            results[solver_name] = None

    print("Saving results...")
    base_dir = pathlib.Path(__file__).parent.absolute()
    result_dir = pathlib.Path(base_dir, "results")
    result_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
    result_file = "{}.pickle".format(result_uuid)
    result_path = pathlib.Path(base_dir, result_dir, result_file)
    assert result_dir.is_dir()
    assert not result_path.is_file()
    with open(result_path, "wb") as file:
        data = {"config": config, "results": results}
        pickle.dump(data, file)
    print("  File name: {}".format(result_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    onerun(args.config_path)
