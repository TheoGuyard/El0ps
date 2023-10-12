import argparse
import pathlib
import random
import shutil
import string
import subprocess
import yaml

src_path = "~/Documents/Github/El0ps"
dst_path = "tguyard@access.grid5000.fr:nancy/gits"
logs_dir = "/home/tguyard/logs"

base_dir = pathlib.Path(__file__).parent.parent.parent.absolute()
experiments_dir = base_dir.joinpath("experiments")
script_dir = experiments_dir.joinpath("scripts")
run_file = "run.sh"
run_path = script_dir.joinpath(run_file)

experiments = [
    {
        "name": "perfprofile",
        "walltime": "05:00:00",
        "besteffort": False,
        "production": True,
        "setups": [
            {
                "expname": "perfprofile",
                "dataset": {
                    "datatype": "synthetic",
                    "datafit_name": "Leastsquares",
                    "penalty_name": penalty,
                    "k": k,
                    "m": 250,
                    "n": 500,
                    "rho": 0.1,
                    "snr": 10.0,
                    "normalize": True,
                },
                "solver_names": ["el0ps", "sbnb", "l0bnb", "mosek"],
                "solver_options": {
                    "time_limit": 3600.0,
                    "rel_tol": 1.0e-4,
                    "int_tol": 1.0e-8,
                    "verbose": False,
                },
            }
            for penalty in ["Bigm", "BigmL2norm"]
            for k in [5, 10]
        ],
    },
    {
        "name": "atp",
        "walltime": "04:30:00",
        "besteffort": False,
        "production": True,
        "setups": [
            {
                "expname": "atp",
                "dataset": {
                    "datatype": "atp",
                    "datafit_name": "Leastsquares",
                    "penalty_name": "NeglogTriangular",
                    "nnz_proba": 0.1,
                    "m": 10,
                    "n": 30,
                    "snr": 10.0,
                    "t": t,
                },
                "solver_names": ["el0ps", "gurobi", "mosek"],
                "solver_options": {
                    "time_limit": 600.0,
                    "rel_tol": 1.0e-4,
                    "int_tol": 1.0e-8,
                    "verbose": False,
                },
                "path_options": {
                    "lmbd_ratio_max": 1.0e-0,
                    "lmbd_ratio_min": 1.0e-3,
                    "lmbd_ratio_num": 10,
                    "warm_start": True,
                },
            }
            for t in [1, 2, 3]
        ],
    },
]


def oar_send():
    print("oar send")
    cmd_str = " ".join(
        [
            "rsync -am",
            "--exclude '.git'",
            "--exclude '.venv'",
            "--exclude '**/results/*.pickle'",
            "--exclude '**/saves/*.csv'",
            "{} {}".format(src_path, dst_path),
        ]
    )
    subprocess.run(cmd_str, shell=True)


def oar_receive():
    for experiment in experiments:
        print("oar receive {}".format(experiment["name"]))
        results_src_path = pathlib.Path(
            dst_path, "El0ps", "experiments", experiment["name"], "results/*"
        )
        results_dst_path = pathlib.Path(
            experiments_dir, experiment["name"], "results"
        )
        cmd_str = "rsync -am {} {}".format(results_src_path, results_dst_path)
        subprocess.run(cmd_str, shell=True)


def oar_install():
    print("oar install")
    cmd_strings = [
        "module load python",
        "python -m pip install -q -e .[exp]",
    ]
    for cmd_string in cmd_strings
        subprocess.run(cmd_string, shell=True)


def oar_make():

    print("oar make run")
    stream = "\n".join(
        [
            "#!/bin/sh",
            "expname=$1",
            "repeats=$2",
            "for i in $(seq 1 $repeats);",
            "do",
            "   oarsub -S {}/$expname/oar.sh".format(script_dir),
            "done",
        ]
    )
    with open(run_path, "w") as file:
        file.write(stream)
    subprocess.run("chmod u+x {}".format(run_path), shell=True)

    for experiment in experiments:
        print("oar make {}".format(experiment["name"]))
        
        # Create the experiment dir
        experiment_dir = script_dir.joinpath(experiment["name"])
        experiment_dir.mkdir()

        # Create the args file (remove old ones)
        args_path = experiment_dir.joinpath("args.txt")
        if args_path.is_file():
            args_path.unlink()

        # Create the oar file
        stream = "\n".join(
            [
                "#!/bin/sh",
                "#OAR -n el0ps-{}".format(experiment["name"]),
                "#OAR -O {}/el0ps-{}.%jobid%.stdout".format(
                    logs_dir, experiment["name"]
                ),
                "#OAR -E {}/el0ps-{}.%jobid%.stderr".format(
                    logs_dir, experiment["name"]
                ),
                "#OAR -l walltime={}".format(experiment["walltime"]),
                "#OAR -t besteffort" if experiment["besteffort"] else "",
                "#OAR -q production" if experiment["production"] else "",
                "#OAR --array-param-file {}".format(args_path),
                "set -xv",
                "source ~/.profile",
                "module load python julia gurobi cplex",
                "python {}/{}/onerun.py $*".format(
                    experiments_dir, experiment["name"]
                ),
            ]
        )
        oar_path = experiment_dir.joinpath("oar.sh")
        with open(oar_path, "w") as file:
            file.write(stream)
        subprocess.run("chmod u+x {}".format(oar_path), shell=True)

        # Create the setups dir
        setups_dir = experiment_dir.joinpath("setups")
        setups_dir.mkdir()

        # Generate the setups and the arg file
        for setup in experiment["setups"]:
            setup_name = "".join(
                random.choice(string.ascii_lowercase) for _ in range(10)
            )
            setup_file = "{}.yaml".format(setup_name)
            setup_path = pathlib.Path(setups_dir, setup_file)

            with open(setup_path, "w") as file:
                yaml.dump(setup, file)

            with open(args_path, "a") as file:
                file.write(str(setup_path))
                file.write("\n")


def oar_clean():
    print("oar clean")
    if run_path.is_file():
        run_path.unlink()
    for experiment in experiments:
        experiment_dir = script_dir.joinpath(experiment["name"])
        if experiment_dir.is_dir():
            shutil.rmtree(experiment_dir)


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
