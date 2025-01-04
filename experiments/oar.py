import argparse
from copy import deepcopy
import pathlib
import random
import shutil
import string
import subprocess
import yaml

# Global variables
SRC_PATH = "~/Documents/Github/El0ps"
DST_PATH = "tguyard@access.grid5000.fr:rennes/gits"
HOME_DIR = "/home/tguyard"
LOGS_DIR = "/home/tguyard/logs"
BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
EXPS_DIR = BASE_DIR.joinpath("experiments")
TMPS_DIR = EXPS_DIR.joinpath("tmps")
RUN_PATH = EXPS_DIR.joinpath("run.sh")


# Experiments setups


def get_exp_mixtures():
    exp = {
        "name": "mixtures",
        "walltime": "11:00:00",
        "besteffort": True,
        "production": True,
        "setups": [],
        "repeats": 10,
    }

    time_limit = 600.0
    relative_gap = 1e-8
    verbose = False

    base_setup = {
        "expname": "mixtures",
        "dataset": {
            "k": 10,
            "m": 500,
            "n": 1000,
            "r": 0.9,
            "s": 10.0,
            "distrib_name": "gaussian",
            "distrib_opts": {"scale": 1.0},
        },
        "solvers": {
            "el0ps": {
                "solver": "el0ps",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "l0bnb": {
                "solver": "l0bnb",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "cplex": {
                "solver": "mip",
                "params": {
                    "optimizer_name": "cplex",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            # "gurobi": {
            #     "solver": "mip",
            #     "params": {
            #         "optimizer_name": "gurobi",
            #         "time_limit": time_limit,
            #         "relative_gap": relative_gap,
            #         "verbose": verbose,
            #     },
            # },
            "mosek": {
                "solver": "mip",
                "params": {
                    "optimizer_name": "mosek",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "mosek_oa": {
                "solver": "oa",
                "params": {
                    "optimizer_name": "mosek",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
        },
    }

    for distrib_name, distrib_opts in [
        ("gaussian", {"scale": 1.0}),
        ("laplace", {"scale": 1.0}),
        ("uniform", {"low": 0.0, "high": 1.0}),
        ("halfgaussian", {"scale": 1.0}),
        ("halflaplace", {"scale": 1.0}),
        ("gausslaplace", {"scale1": 1.0, "scale2": 1.0}),
    ]:
        setup = deepcopy(base_setup)
        setup["dataset"]["distrib_name"] = distrib_name
        setup["dataset"]["distrib_opts"] = distrib_opts
        exp["setups"].append(setup)
    return exp


def get_exp_microscopy():
    exp = {
        "name": "microscopy",
        "walltime": "11:00:00",
        "besteffort": True,
        "production": True,
        "setups": [],
        "repeats": 10,
    }

    time_limit = 60.0
    relative_gap = 1e-8
    verbose = False

    base_setup = {
        "expname": "microscopy",
        "penalty": "BigmL1norm",
        "solvers": {
            "el0ps": {
                "solver": "el0ps",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "l0bnb": {
                "solver": "l0bnb",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "cplex": {
                "solver": "mip",
                "params": {
                    "optimizer_name": "cplex",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            # "gurobi": {
            #     "solver": "mip",
            #     "params": {
            #         "optimizer_name": "gurobi",
            #         "time_limit": time_limit,
            #         "relative_gap": relative_gap,
            #         "verbose": verbose,
            #     },
            # },
            "mosek": {
                "solver": "mip",
                "params": {
                    "optimizer_name": "mosek",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "mosek_oa": {
                "solver": "oa",
                "params": {
                    "optimizer_name": "mosek",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
        },
        "path_opts": {
            "lmbd_max": 1e-0,
            "lmbd_min": 1e-3,
            "lmbd_num": 31,
            "lmbd_scaled": True,
            "stop_if_not_optimal": True,
            "verbose": True,
        },
    }

    for penalty in ["BigmL1norm", "BigmL2norm"]:
        setup = deepcopy(base_setup)
        setup["penalty"] = penalty
        exp["setups"].append(setup)

    return exp


def get_exp_regpath():
    exp = {
        "name": "regpath",
        "walltime": "11:00:00",
        "besteffort": False,
        "production": True,
        "setups": [],
        "repeats": 1,
    }

    time_limit = 600.0
    relative_gap = 1e-8
    verbose = False

    base_setup = {
        "expname": "regpath",
        "dataset": "riboflavin",
        "datafit": "Leastsquares",
        "penalty": "BigmL2norm",
        "solvers": {
            "el0ps": {
                "solver": "el0ps",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "l0bnb": {
                "solver": "l0bnb",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "cplex": {
                "solver": "mip",
                "params": {
                    "optimizer_name": "cplex",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            # "gurobi": {
            #     "solver": "mip",
            #     "params": {
            #         "optimizer_name": "gurobi",
            #         "time_limit": time_limit,
            #         "relative_gap": relative_gap,
            #         "verbose": verbose,
            #     },
            # },
            "mosek": {
                "solver": "mip",
                "params": {
                    "optimizer_name": "mosek",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "mosek_oa": {
                "solver": "oa",
                "params": {
                    "optimizer_name": "mosek",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
        },
        "path_opts": {
            "lmbd_max": 1e-0,
            "lmbd_min": 1e-3,
            "lmbd_num": 31,
            "lmbd_scaled": True,
            "stop_if_not_optimal": True,
            "verbose": True,
        },
    }

    for dataset, datafit in [
        ("riboflavin", "Leastsquares"),
        ("bctcga", "Leastsquares"),
        ("colon-cancer", "Logistic"),
        ("leukemia", "Logistic"),
        ("breast-cancer", "Squaredhinge"),
        ("arcene", "Squaredhinge"),
    ]:
        for penalty in ["BigmL1norm", "BigmL2norm"]:
            setup = deepcopy(base_setup)
            setup["dataset"] = dataset
            setup["datafit"] = datafit
            setup["penalty"] = penalty
            exp["setups"].append(setup)

    return exp


# List of experiments

experiments = [
    get_exp_mixtures(),
    get_exp_microscopy(),
    get_exp_regpath(),
]


# OAR functions


def oar_send():
    print("oar send")
    cmd_str = " ".join(
        [
            "rsync -amv",
            "--exclude '.git'",
            "--exclude '.github'",
            "--exclude '.venv'",
            "--exclude '.DS_Store'",
            "--exclude 'doc/'",
            "--exclude '**/results/*.pkl'",
            "--exclude '**/saves/*.csv'",
            "--exclude '**/saves/*.pkl'",
            "--exclude '**/__pycache__'",
            "--exclude '**/.pytest_cache'",
            "{} {}".format(SRC_PATH, DST_PATH),
        ]
    )
    subprocess.run(cmd_str, shell=True)


def oar_receive():
    print("oar receive")
    for experiment in experiments:
        print(f"  processing {experiment['name']}")
        results_src_path = pathlib.Path(
            DST_PATH, "El0ps", "experiments", experiment["name"], "results/*"
        )
        results_dst_path = pathlib.Path(
            EXPS_DIR, experiment["name"], "results"
        )
        cmd_str = "rsync -amv {} {}".format(results_src_path, results_dst_path)
        subprocess.run(cmd_str, shell=True)


def oar_install():
    print("oar install")
    cmd_strs = [
        "module load conda",
        "conda activate el0ps",
        "pip install -q -e .[exp]",
    ]
    for cmd_str in cmd_strs:
        subprocess.run(cmd_str, shell=True)


def oar_make():
    print("oar make run")

    # Run file stream
    run_stream = "\n".join(
        [
            "# !/bin/sh",
            "expname=$1",
            "repeats=$2",
            "for i in $(seq 1 $repeats);",
            "do",
            "   oarsub --project simsmart -S {}/$expname/oar.sh".format(
                TMPS_DIR
            ),
            "done",
        ]
    )

    # Write run file
    with open(RUN_PATH, "w") as file:
        file.write(run_stream)
    subprocess.run("chmod u+x {}".format(RUN_PATH), shell=True)

    # Create the scripts dir (remove old one)
    if TMPS_DIR.is_dir():
        shutil.rmtree(TMPS_DIR)
    else:
        TMPS_DIR.mkdir()

    for experiment in experiments:
        print("oar make {}".format(experiment["name"]))

        # Create the experiment dir
        experiment_dir = TMPS_DIR.joinpath(experiment["name"])
        experiment_dir.mkdir()

        # Create the args file (remove old ones)
        configs_path = experiment_dir.joinpath("configs.txt")
        if configs_path.is_file():
            configs_path.unlink()

        # Oar file stream
        stream = "\n".join(
            [
                "# !/bin/sh",
                "#OAR -n el0ps-{}".format(experiment["name"]),
                "#OAR -O {}/el0ps-{}.%jobid%.stdout".format(
                    LOGS_DIR, experiment["name"]
                ),
                "#OAR -E {}/el0ps-{}.%jobid%.stderr".format(
                    LOGS_DIR, experiment["name"]
                ),
                "#OAR -l walltime={}".format(experiment["walltime"]),
                "#OAR -t besteffort" if experiment["besteffort"] else "",
                "#OAR -q production" if experiment["production"] else "",
                "#OAR -p gpu_count=0",
                "#OAR --array-param-file {}".format(configs_path),
                "set -xv",
                "source {}/.profile".format(HOME_DIR),
                "module load conda gurobi cplex",
                "conda activate el0ps",
                "{} {}/{}/exp.py run -r {}/{}/results -c $* -n {} -v".format(
                    "python",
                    EXPS_DIR,
                    experiment["name"],
                    EXPS_DIR,
                    experiment["name"],
                    experiment["repeats"],
                ),
            ]
        )

        # Write oar file
        oar_path = experiment_dir.joinpath("oar.sh")
        with open(oar_path, "w") as file:
            file.write(stream)
        subprocess.run("chmod u+x {}".format(oar_path), shell=True)

        # Create setups dir
        setups_dir = experiment_dir.joinpath("setups")
        setups_dir.mkdir()

        # Generate setups and configs file
        for setup in experiment["setups"]:
            setup_name = "".join(
                random.choice(string.ascii_lowercase) for _ in range(10)
            )
            setup_file = "{}.yml".format(setup_name)
            setup_path = pathlib.Path(setups_dir, setup_file)

            with open(setup_path, "w") as file:
                yaml.dump(setup, file)

            with open(configs_path, "a") as file:
                file.write(str(setup_path))
                file.write("\n")


def oar_clean():
    print("oar clean")
    if RUN_PATH.is_file():
        RUN_PATH.unlink()
    if TMPS_DIR.is_dir():
        shutil.rmtree(TMPS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "oar_cmd", choices=["send", "install", "make", "clean", "receive"]
    )
    args = parser.parse_args()

    if args.oar_cmd == "send":
        oar_send()
    elif args.oar_cmd == "install":
        oar_install()
    elif args.oar_cmd == "make":
        oar_make()
    elif args.oar_cmd == "clean":
        oar_clean()
    elif args.oar_cmd == "receive":
        oar_receive()
    else:
        raise ValueError(f"Unknown command {args.oar_cmd}.")
