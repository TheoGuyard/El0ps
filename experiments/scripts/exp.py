from copy import deepcopy


def get_exp_perfprofile():
    exp = {
        "name": "perfprofile",
        "walltime": "03:15:00",
        "besteffort": True,
        "production": True,
        "setups": [],
    }

    base_setup = {
        "expname": "perfprofile",
        "dataset": {
            "dataset_type": "synthetic",
            "dataset_opts": {
                "model": "linear",
                "k": 5,
                "m": 500,
                "n": 1_000,
                "rho": 0.9,
                "snr": 10.0,
                "normalize": True,
            },
            "process_opts": {"center": True, "normalize": True},
            "datafit_name": "Leastsquares",
            "penalty_name": "Bigm",
        },
        "solvers": {
            "solvers_name": [
                "el0ps",
                "el0ps[l0screening=False]",
                "el0ps[l0screening=False,dualpruning=False]",
            ],
            "solvers_opts": {
                "time_limit": 3600.0,
                "rel_tol": 1.0e-4,
                "int_tol": 1.0e-8,
                "verbose": False,
            },
        },
    }

    for k in [5, 7, 9]:
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["k"] = k
        exp["setups"].append(setup)

    for n in [1_000, 10_000, 100_000]:
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["n"] = n
        exp["setups"].append(setup)

    for rho in [0.0, 0.9, 0.99]:
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["rho"] = rho
        exp["setups"].append(setup)

    for snr in [10.0, 3.1623, 1.2589]:
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["snr"] = snr
        exp["setups"].append(setup)

    return exp


def get_exp_regpath():
    exp = {
        "name": "regpath",
        "walltime": "12:00:00",
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
            "lmbd_ratio_max": 1.0e-0,
            "lmbd_ratio_min": 1.0e-5,
            "lmbd_ratio_num": 101,
            "stop_if_not_optimal": True,
        },
    }

    for dataset_name, datafit_name, penalty_name in [
        ("riboflavin", "Leastsquares", "BigmL1norm"),
        ("riboflavin", "Leastsquares", "BigmL2norm"),
        ("bctcga", "Leastsquares", "BigmL1norm"),
        ("bctcga", "Leastsquares", "BigmL2norm"),
        ("colon-cancer", "Logistic", "BigmL1norm"),
        ("colon-cancer", "Logistic", "BigmL2norm"),
        ("leukemia", "Logistic", "BigmL1norm"),
        ("leukemia", "Logistic", "BigmL2norm"),
        ("arcene", "Squaredhinge", "BigmL1norm"),
        ("arcene", "Squaredhinge", "BigmL2norm"),
        ("breast-cancer", "Squaredhinge", "BigmL1norm"),
        ("breast-cancer", "Squaredhinge", "BigmL2norm"),
    ]:
        for solver_name in [
            "el0ps",
            "el0ps[l0screening=False]",
            "el0ps[l0screening=False,dualpruning=False]",
            "l0bnb",
            "cplex",
            "gurobi",
            "mosek",
        ]:
            setup = deepcopy(base_setup)
            setup["dataset"]["dataset_opts"]["dataset_name"] = dataset_name
            setup["dataset"]["datafit_name"] = datafit_name
            setup["dataset"]["penalty_name"] = penalty_name
            setup["solvers"]["solvers_name"] = [solver_name]
            exp["setups"].append(setup)

    return exp


def get_exp(exp_name):
    get_exp_func = "get_exp_" + exp_name
    return eval(get_exp_func)()
