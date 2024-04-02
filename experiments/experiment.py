import pathlib
import pickle
import sys
import yaml
from abc import abstractmethod
from copy import deepcopy
from datetime import datetime
from el0ps.utils import compiled_clone, compute_lmbd_max

sys.path.append(pathlib.Path(__file__).parent.parent.absolute())
from experiments.instances import get_data, calibrate_parameters  # noqa
from experiments.solvers import get_solver  # noqa


class Experiment:
    name = "experiment"
    results_dir = pathlib.Path(__file__).parent.absolute().joinpath("results")
    saves_dir = pathlib.Path(__file__).parent.absolute().joinpath("saves")

    def __init__(self, config_path):
        self.config_path = config_path
        self.results = None

    def setup(self):
        print("Loading config...")
        with open(pathlib.Path(self.config_path), "r") as file:
            self.config = yaml.load(file, Loader=yaml.Loader)

    def load_problem(self):
        print("Loading data...")
        A, y, x_true = get_data(self.config["dataset"])
        print("  A shape: {}".format(A.shape))
        print("  y shape: {}".format(y.shape))
        print("  x shape: {}".format(None if x_true is None else x_true.shape))

        print("Calibrating parameters...")
        datafit, penalty, lmbd, x_cal = calibrate_parameters(
            self.config["dataset"]["datafit_name"],
            self.config["dataset"]["penalty_name"],
            A,
            y,
            x_true,
        )
        lmbd_max = compute_lmbd_max(datafit, penalty, A)
        print("  num nz: {}".format(sum(x_cal != 0.0)))
        print("  lratio: {}".format(lmbd / lmbd_max))
        for param_name, param_value in penalty.params_to_dict().items():
            print("  {}\t: {}".format(param_name, param_value))
        self.x_true = x_true
        self.datafit = datafit
        self.penalty = penalty
        self.A = A
        self.lmbd = lmbd

    def precompile(self):
        print("Precompiling...")
        self.compiled_datafit = compiled_clone(self.datafit)
        self.compiled_penalty = compiled_clone(self.penalty)
        solver_opts = deepcopy(self.config["solvers"]["solvers_opts"])
        solver_opts["time_limit"] = 5.0
        solver = get_solver("el0ps", solver_opts)
        solver.solve(
            self.compiled_datafit, self.compiled_penalty, self.A, self.lmbd
        )

    @abstractmethod
    def run(self):
        pass

    def save_results(self):
        print("Saving results...")
        result_uuid = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
        result_file = "{}_{}.pickle".format(self.name, result_uuid)
        result_path = pathlib.Path(self.results_dir, result_file)
        with open(result_path, "wb") as file:
            data = {"config": self.config, "results": self.results}
            pickle.dump(data, file)
        print("  File name: {}".format(result_file))

    @abstractmethod
    def load_results(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def save_plot(self):
        pass
