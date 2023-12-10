import argparse
import pathlib
import pickle
import os
import sys
import yaml
from copy import deepcopy
from datetime import datetime
from el0ps.problem import Problem
from el0ps.path import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.instances import get_data  # noqa
from experiments.solvers import get_solver, can_handle  # noqa


def onerun(config_path, nosave=False):
    print("Loading configuration...")
    config_path = pathlib.Path(config_path)
    assert config_path.is_file()
    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    print("Generating data...")
    datafit, penalty, A, lmbd, x_true = get_data(config["dataset"])
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

    print("Running experiment...")
    results = {
        solver_name: None for solver_name in config["solvers"]["solvers_name"]
    }
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
                    print(result)
                elif config["task"]["task_type"] == "fitpath":
                    path = Path(
                        lmbd_ratio_max=config["task"]["task_opts"][
                            "lmbd_ratio_max"
                        ],
                        lmbd_ratio_min=config["task"]["task_opts"][
                            "lmbd_ratio_min"
                        ],
                        lmbd_ratio_num=config["task"]["task_opts"][
                            "lmbd_ratio_num"
                        ],
                        stop_if_not_optimal=config["task"]["task_opts"][
                            "stop_if_not_optimal"
                        ],
                    )
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

    if not nosave:
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
    parser.add_argument("--nosave", action="store_true")
    args = parser.parse_args()
    onerun(args.config_path, nosave=args.nosave)
