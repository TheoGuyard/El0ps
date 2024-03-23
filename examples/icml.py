import pathlib
import sys

sys.path.append(pathlib.Path(__file__).parent.parent.absolute())
from experiments.instances import get_data_synthetic, calibrate_parameters  # noqa
from experiments.solvers import get_solver  # noqa

# Generate sparse regression data
A, y, x_true = get_data_synthetic(
    matrix      = "correlated(0.9)",    # matrix type
    model       = "linear",             # model type
    k           = 5,                    # sparsity level
    m           = 500,                  # number of rows in A
    n           = 1000,                 # number of cols in A
    s           = 10.0,                 # snr level
    normalize   = True,                 # normalize the columns in A
)

# Calibrate hyperparameters of the L0-problem using L0learn. The datafit and
# penalty names must be strings corresponding to a subclass of
# el0ps.datafit.AbstractDatafit and el0ps.datafit.AbstractPenalty.
datafit, penalty, lmbd, _ = calibrate_parameters(
    "Leastsquares", "Bigm", A, y, x_true
)

# Precompile datafit and penalty with numba
solver = get_solver("el0ps", {"time_limit": 5.0})
result = solver.solve(datafit, penalty, A, lmbd)

# Solvers to evaluate to use cplex, gurobi or mosek, you need to install these
# solvers locally, see the `docplex`, `gurobipy` and `mosek` python packages
# for install instructions.
solvers_name = [
    "el0ps[simpruning=True]",   # solver with the simultaneous pruning
    "el0ps[simpruning=False]",  # solver without the simultaneous pruning
    "l0bnb",                    # l0bnb solver from Hazimeh et al.
    # "cplex",
    # "gurobi",
    # "mosek",
]
solvers_opts = {
    "time_limit": 3600.0,   # time limit in seconds
    "rel_tol": 1.0e-4,      # relative optimality tolerance targeted
    "int_tol": 1.0e-8,      # integer tolerance
    "verbose": False,       # whether to toogle verbosity
}

for solver_name in solvers_name:
    solver = get_solver(solver_name, solvers_opts)
    result = solver.solve(datafit, penalty, A, lmbd)
    print(solver_name)
    print(result)
    print()
