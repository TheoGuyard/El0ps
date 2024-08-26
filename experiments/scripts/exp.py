import os
import sys
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solvers import can_handle_instance  # noqa


def get_exp_bigm():
    exp = {
        "name": "bigm",
        "command": "bigm",
        "walltime": "00:30:00",
        "besteffort": True,
        "production": True,
        "setups": [],
    }

    base_setup = {
        "expname": "bigm",
        "dataset": {
            "dataset_type": "synthetic",
            "dataset_opts": {
                "matrix": "correlated(0.9)",
                "model": "linear",
                "supp_pos": "equispaced",
                "supp_val": "normal(0.,1.)",
                "k": 5,
                "m": 500,
                "n": 1000,
                "s": 10.0,
                "normalize": True,
            },
            "process_opts": {"center": True, "normalize": True},
            "datafit_name": "Leastsquares",
            "penalty_name": "BoundsConstraint",
            "bigmfactor": 1.0,
        },
        "solvers": {
            "solvers_name": [
                "el0ps[simpruning=False]",
                "el0ps[simpruning=False,peeling=True]",
            ],
            "solvers_opts": {
                "time_limit": 600.0,
                "rel_tol": 1.0e-4,
                "int_tol": 1.0e-8,
                "verbose": False,
                "trace": True,
            },
        },
    }

    for bigmfactor in [1.0, 2.0, 3.0]:
        setup = deepcopy(base_setup)
        setup["dataset"]["bigmfactor"] = bigmfactor
        exp["setups"].append(setup)

    # exp["setups"].append(base_setup)

    return exp


def get_exp_perfprofile():
    exp = {
        "name": "perfprofile",
        "command": "perfprofile",
        "walltime": "02:10:00",
        "besteffort": False,
        "production": True,
        "setups": [],
    }

    base_setup = {
        "expname": "perfprofile",
        "dataset": {
            "dataset_type": "synthetic",
            "dataset_opts": {
                "matrix": "correlated(0.9)",
                "model": "linear",
                "supp_pos": "equispaced",
                "supp_val": "normal(0.,1.)",
                "k": 5,
                "m": 500,
                "n": 1000,
                "s": 10.0,
                "normalize": True,
            },
            "process_opts": {"center": True, "normalize": True},
            "datafit_name": "Leastsquares",
            "penalty_name": "Bigm",
        },
        "solvers": {
            "solvers_name": [
                "el0ps",
                "el0ps[simpruning=False]",
            ],
            "solvers_opts": {
                "time_limit": 3600.0,
                "rel_tol": 1.0e-4,
                "int_tol": 1.0e-8,
                "verbose": False,
            },
        },
    }

    import numpy as np

    for n in np.linspace(1000, 10000, 7):
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["n"] = int(n)
        exp["setups"].append(setup)
    for one_minus_rho in np.linspace(0.1, 0.01, 7):
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["matrix"] = "correlated({})".format(
            1.0 - one_minus_rho
        )
        exp["setups"].append(setup)
    for k in np.linspace(5, 10, 5):
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["k"] = int(k)
        exp["setups"].append(setup)
    for s in np.linspace(2, 10, 5):
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["s"] = float(s)
        exp["setups"].append(setup)

    # exp["setups"].append(base_setup)

    return exp


def get_exp_regpath():
    exp = {
        "name": "regpath",
        "command": "regpath",
        "walltime": "05:00:00",
        "besteffort": False,
        "production": True,
        "setups": [],
    }

    base_setup = {
        "expname": "regpath",
        "dataset": {
            "dataset_type": "hardcoded",
            "dataset_opts": {"dataset_name": "riboflavin"},
            "process_opts": {"center": True, "normalize": True},
            "datafit_name": "Leastsquares",
            "penalty_name": "BigmL2norm",
        },
        "solvers": {
            "solvers_name": ["el0ps"],
            "solvers_opts": {
                "time_limit": 3600.0,
                "rel_tol": 1.0e-4,
                "int_tol": 1.0e-8,
                "verbose": False,
            },
        },
        "path_opts": {
            "lmbd_max": 1.0e-0,
            "lmbd_min": 1.0e-5,
            "lmbd_num": 101,
            "lmbd_scaled": True,
            "stop_if_not_optimal": True,
        },
    }

    for dataset_name, datafit_name, penalty_name in [
        ("riboflavin", "Leastsquares", "BoundsConstraint"),
        ("bctcga", "Leastsquares", "BoundsConstraint"),
        ("colon-cancer", "Logistic", "BoundsConstraint"),
        ("leukemia", "Logistic", "BoundsConstraint"),
        ("arcene", "Squaredhinge", "BoundsConstraint"),
        ("breast-cancer", "Squaredhinge", "BoundsConstraint"),
    ]:
        for solver_name in [
            "el0ps[simpruning=False]",
            "el0ps[simpruning=False,peeling=True]",
            "mip[optimizer_name=cplex]",
            "mip[optimizer_name=gurobi]",
            "mip[optimizer_name=mosek]",
            "l0bnb",
        ]:
            if can_handle_instance(solver_name, datafit_name, penalty_name):
                setup = deepcopy(base_setup)
                setup["dataset"]["dataset_opts"]["dataset_name"] = dataset_name
                setup["dataset"]["datafit_name"] = datafit_name
                setup["dataset"]["penalty_name"] = penalty_name
                setup["solvers"]["solvers_name"] = [solver_name]
                exp["setups"].append(setup)

    return exp


def get_exp_statistics():
    exp = {
        "name": "statistics",
        "command": "statistics",
        "walltime": "01:00:00",
        "besteffort": True,
        "production": True,
        "setups": [],
    }

    base_setup = {
        "expname": "statistics",
        "dataset": {
            "dataset_type": "synthetic",
            "dataset_opts": {
                "matrix": "correlated(0.9)",
                "model": "linear",
                "supp_pos": "equispaced",
                "supp_val": "unit",
                "k": 10,
                "m": 150,
                "n": 200,
                "s": 10.0,
                "normalize": True,
            },
            "process_opts": {"center": True, "normalize": True},
            "datafit_name": "Leastsquares",
            "penalty_name": "BigmL2norm",
            "test_size": 0.3333,
        },
        "solvers": {
            "solvers_name": [
                "el0ps",
            ],
            "solvers_opts": {
                "time_limit": 600.0,
                "rel_tol": 1.0e-4,
                "int_tol": 1.0e-8,
                "verbose": False,
            },
        },
        "relaxed_solvers": ["Omp", "Lasso", "Enet", "L05", "Mcp", "Scad"],
        "path_opts": {
            "lmbd_max": 1.0e-0,
            "lmbd_min": 1.0e-5,
            "lmbd_num": 101,
            "lmbd_scaled": True,
            "stop_if_not_optimal": True,
            "max_nnz": 20,
            "verbose": False,
        },
    }

    exp["setups"].append(base_setup)

    return exp


def get_exp(exp_name):
    get_exp_func = "get_exp_" + exp_name
    return eval(get_exp_func)()
