from copy import deepcopy


def get_exp_perfprofile():
    exp = {
        "name": "perfprofile",
        "command": "perfprofile",
        "walltime": "07:15:00",
        "besteffort": True,
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
                "supp_val": "unit",
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
                "el0ps[simpruning=False,dualpruning=False]",
                "l0bnb",
                "cplex",
                "gurobi",
                "mosek",
            ],
            "solvers_opts": {
                "time_limit": 3600.0,
                "rel_tol": 1.0e-4,
                "int_tol": 1.0e-8,
                "verbose": False,
            },
        },
    }

    exp["setups"].append(base_setup)

    return exp


def get_exp_regpath():
    exp = {
        "name": "regpath",
        "command": "regpath",
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
            "el0ps[simpruning=False]",
            "el0ps[simpruning=False,dualpruning=False]",
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
                "l0bnb",
            ],
            "solvers_opts": {
                "time_limit": 600.0,
                "rel_tol": 1.0e-4,
                "int_tol": 1.0e-8,
                "verbose": False,
            },
        },
        "relaxed_solvers": ["Omp", "Lasso", "Enet"],
        "path_opts": {
            "lmbd_ratio_max": 1.0e-0,
            "lmbd_ratio_min": 1.0e-5,
            "lmbd_ratio_num": 101,
            "stop_if_not_optimal": True,
            "max_nnz": 20,
        },
    }

    exp["setups"].append(base_setup)

    return exp


def get_exp_mcpquality():
    exp = {
        "name": "mcpquality",
        "command": "mcpquality",
        "walltime": "00:05:00",
        "besteffort": True,
        "production": True,
        "setups": [],
    }

    base_setup = {
        "expname": "mcpquality",
        "dataset": {
            "dataset_type": "synthetic",
            "dataset_opts": {
                "matrix": "correlated(0.9)",
                "model": "linear",
                "supp_pos": "equispaced",
                "supp_val": "unit",
                "k": 5,
                "m": 100,
                "n": 50,
                "s": 10.0,
                "normalize": True,
            },
            "process_opts": {"center": True, "normalize": True},
            "datafit_name": "Leastsquares",
            "penalty_name": "L2norm",
        },
        "regfunc_types": ["convex", "concave_eig", "concave_etp"],
    }

    setup = deepcopy(base_setup)
    exp["setups"].append(setup)

    for k in [10]:
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["k"] = k
        exp["setups"].append(setup)
    for m in [150]:
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["m"] = m
        exp["setups"].append(setup)
    for n in [75]:
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["n"] = n
        exp["setups"].append(setup)
    for r in [0.1]:
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["matrix"] = f"correlated({r})"
        exp["setups"].append(setup)
    for s in [1.0]:
        setup = deepcopy(base_setup)
        setup["dataset"]["dataset_opts"]["s"] = s
        exp["setups"].append(setup)

    return exp


def get_exp(exp_name):
    get_exp_func = "get_exp_" + exp_name
    return eval(get_exp_func)()
