import argparse
import numpy as np
from exprun import Experiment, Runner
from el0ps.compilation import CompilableClass, compiled_clone

from experiments.solver import (
    get_solver,
    can_handle_instance,
    can_handle_compilation,
    precompile_solver,
)
from experiments.instance import calibrate_parameters, preprocess_data


class Synthetic(Experiment):

    def generate_data(
        self,
        t: float = 0.0,
        k: int = 10,
        m: int = 500,
        n: int = 1000,
        r: float = 0.5,
        s: float = 10.0,
        seed=None,
    ):
        r"""Generate synthetic sparse regression data

        .. math:: y = Ax + e

        where

        - :math:`x \in R^n` is the ground truth vector with :math:`k`
        non-zero entries randomly distributed over :math:`\{1,\dots,n\}` with
        i.i.d. amplitude :math:`x_i = u_i` where `u_i ~ N(0,t)`. When
        :math:`t = 0`, the amplitudes are drawn randomly from :math:`\{-1,1\}`.

        - :math:`A \in R^{m \times n}` is a design matrix with i.i.d. columns
        drawn as :math:`a_i \sim N(0,K)` where :math:`K_{ij} = r^{|i-j|}`.

        - :math:`e ~ N(0,\sigma I_{m \times m\})` is a noise vector with
        signal-to-noise ratio :math:`s` with respect to :math:`y`.

        Parameters
        ----------
        t : float
            Standard deviation of the ground truth non-zero entries amplitude.
        k : int
            Number of non-zero entries in the ground truth.
        m : int
            Number of rows in the design matrix.
        n : int
            Number of columns in the design matrix.
        r : float
            Correlation coefficient between columns of the design matrix.
        s : float
            Signal-to-noise ratio of the noise.
        """

        assert t >= 0.0
        assert n >= k > 0
        assert m > 0
        assert 0.0 <= r < 1.0
        assert s > 0.0

        if seed is not None:
            np.random.seed(seed)

        # Ground truth
        x = np.zeros(n)
        S = np.random.choice(n, size=k, replace=False)
        if t == 0.0:
            x[S] = np.sign(np.random.randn(k))
        else:
            x[S] = np.random.normal(0.0, t, k)

        # Design matrix
        M = np.zeros(n)
        N1 = np.repeat(np.arange(n).reshape(n, 1), n).reshape(n, n)
        N2 = np.repeat(np.arange(n).reshape(1, n), n).reshape(n, n).T
        K = np.power(r, np.abs(N1 - N2))
        A = np.random.multivariate_normal(M, K, size=m)
        A /= np.linalg.norm(A, axis=0, ord=2)

        # Observation vector
        y = A @ x
        e = np.random.randn(m)
        e *= np.sqrt((y @ y) / (s * (e @ e)))
        y += e

        return A, y, x

    def setup(self) -> None:

        A, y, x_true = self.generate_data(**self.config["dataset"])
        A, y, x_true = preprocess_data(A, y, x_true)
        datafit, penalty, lmbd, x_l0learn = calibrate_parameters(
            "Leastsquares",
            self.config["penalty"],
            A,
            y,
            x_true,
        )

        self.x_true = x_true
        self.x_l0learn = x_l0learn
        self.datafit = datafit
        self.penalty = penalty
        self.A = A
        self.lmbd = lmbd

        if isinstance(self.datafit, CompilableClass):
            self.datafit_compiled = compiled_clone(self.datafit)
        else:
            self.datafit_compiled = None
        if isinstance(self.penalty, CompilableClass):
            self.penalty_compiled = compiled_clone(self.penalty)
        else:
            self.penalty_compiled = None

    def run(self) -> dict:
        result = {}
        for solver_name, solver_keys in self.config["solvers"].items():
            if can_handle_instance(
                solver_keys["solver"],
                solver_keys["params"],
                str(self.datafit),
                str(self.penalty),
            ):
                print("Running {}...".format(solver_name))
                solver = get_solver(
                    solver_keys["solver"],
                    solver_keys["params"],
                )
                if can_handle_compilation(solver_keys["solver"]):
                    precompile_solver(
                        solver,
                        self.datafit_compiled,
                        self.penalty_compiled,
                        self.A,
                        self.lmbd,
                    )
                    result[solver_name] = solver.solve(
                        self.datafit_compiled,
                        self.penalty_compiled,
                        self.A,
                        self.lmbd,
                    )
                else:
                    result[solver_name] = solver.solve(
                        self.datafit, self.penalty, self.A, self.lmbd
                    )
            else:
                print("Skipping {}".format(solver_name))
                result[solver_name] = None

            if result[solver_name] is not None:
                print(result[solver_name])

        return result

    def cleanup(self) -> None:
        pass

    def plot(self, results: list) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["run", "plot"])
    parser.add_argument("--config_path", "-c", type=str)
    parser.add_argument("--results_dir", "-r", type=str)
    parser.add_argument("--repeats", "-n", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    runner = Runner(verbose=args.verbose)

    if args.command == "run":
        runner.run(Synthetic, args.config_path, args.results_dir, args.repeats)
    elif args.command == "plot":
        runner.plot(Synthetic, args.config_path, args.results_dir)
    else:
        raise ValueError(f"Unknown command {args.command}.")
