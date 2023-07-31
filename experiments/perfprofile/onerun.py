import argparse
import os
import pathlib
import pickle
import random
import string
import sys
import yaml
from el0ps.problem import Problem
from el0ps.solver import BnbSolver

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.instances import synthetic_data
from common.solvers import get_solver, precompile, can_handle


def exp(config_path):
    print("Preprocessing...")
    base_dir = pathlib.Path(__file__).parent.absolute()
    result_dir = pathlib.Path(base_dir, "results")
    result_file = "{}.pickle".format("".join(random.choice(string.ascii_lowercase) for _ in range(10)))
    result_path = pathlib.Path(base_dir, result_dir, result_file)
    config_path = pathlib.Path(config_path)

    assert result_dir.is_dir()
    assert config_path.is_file()
    assert not result_path.is_file()

    with open(config_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    print("Generating data...")
    datafit, penalty, A, lmbd, x_true = synthetic_data(
        config["dataset"]["datafit_name"],
        config["dataset"]["penalty_name"],
        config["dataset"]["k"],
        config["dataset"]["m"],
        config["dataset"]["n"],
        config["dataset"]["rho"],
        config["dataset"]["snr"],
        config["dataset"]["normalize"],
    )
    problem = Problem(datafit, penalty, A, lmbd)

    print("Running...")
    results = {}
    for solver_name in config["solver_names"]:
        solver = get_solver(solver_name, config["options"])
        if can_handle(
            solver_name, 
            config["dataset"]["datafit_name"], 
            config["dataset"]["penalty_name"],
        ):
            print("  Solver: {}".format(solver_name))
            print("    Precompiling...")
            precompile(problem, solver)
            print("    Solving...")
            result = solver.solve(problem)
            print(result)
        else:
            print("  Skipping {}...".format(solver_name))
            result = None
        results[solver_name] = result

    print("Saving results...")
    with open(result_path, "wb") as file:
        data = {"config": config, "results": results}
        pickle.dump(data, file)

    print("  File name: {}".format(result_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    exp(args.config_path)
