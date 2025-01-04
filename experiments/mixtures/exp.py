import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erfc
from exprun import Experiment, Runner
from el0ps.compilation import CompilableClass, compiled_clone
from el0ps.datafit import Leastsquares
from el0ps.penalty import (
    BigmL1norm,
    L2norm,
    L1L2norm,
    Bounds,
    BigmPositiveL1norm,
    PositiveL2norm,
)
from el0ps.solver import Status

from experiments.solver import (
    get_solver,
    can_handle_instance,
    can_handle_compilation,
    precompile_solver,
)


class Mixtures(Experiment):

    def inverse_transform_sampling(
        self,
        pdf,
        theta,
        left_domain_boundary,
        right_domain_boundary,
        size=1,
        n=1000,
    ):
        xs = np.linspace(left_domain_boundary, right_domain_boundary, n)
        cs = np.array(
            [
                quad(lambda t: pdf(t, theta), left_domain_boundary, x)[0]
                for x in xs
            ]
        )
        cs /= cs[-1]
        ic = interp1d(
            cs,
            xs,
            bounds_error=False,
            fill_value=(left_domain_boundary, right_domain_boundary),
        )
        v = np.random.uniform(0, 1, size)
        return ic(v)[0]

    def sample_distribution(self, distrib_name: str, distrib_opts: dict):
        """Sample a random variable from a distribution.

        Parameters
        ----------
        distrib_name : str
            Name of the mixture distribution to sample from. Available options
            are:
                - 'gaussian': Gaussian distribution. The option to
                specify in `distrib_opts` is 'scale'. See `np.random.normal`
                for more details.
                - 'laplace': Laplace distribution. The option to
                specify in `distrib_opts` is 'scale'. See `np.random.laplace`
                for more details.
                - 'uniform': Uniform distribution. Options to specify in
                `distrib_opts` are 'low' and 'high'. See `np.random.uniform`
                for more details.
                - 'halfgaussian': Half-Gaussian distribution. The option to
                specify in `distrib_opts` is 'scale'. See `np.random.normal`
                for more details.
                - 'halflaplace': Half-Laplace distribution. The option to
                specify in `distrib_opts` is 'scale'. See `np.random.laplace`
                for more details.
                - 'gauss-laplace': Gaussian-Laplace distribution. The options
                to specify in `distrib_opts` are 'alpha' and 'beta'. The
                probability density function of the distribution is defined as
                in Eq. 6 of "Chaari, L., Batatia, H., Dobigeon, N.,
                Tourneret, J. Y. (2014, May). A hierarchical
                sparsity-smoothness Bayesian model for L0+L1+L2 regularization.
                In 2014 IEEE International Conference on Acoustics, Speech and
                Signal Processing (ICASSP) (pp. 1901-1905). IEEE." with
                :math:`\alpha = 1 / \text{distrib_opts["scale1"]}` and
                :math:`\beta = 1 / \text{distrib_opts["scale2"]}^2`.
        distrib_opts : dict
            Options for the mixture distribution.
        """
        if distrib_name == "gaussian":
            u = np.random.normal(0.0, distrib_opts["scale"])
        elif distrib_name == "laplace":
            u = np.random.laplace(0.0, distrib_opts["scale"])
        elif distrib_name == "uniform":
            assert distrib_opts["low"] <= 0.0 <= distrib_opts["high"]
            u = np.random.uniform(distrib_opts["low"], distrib_opts["high"])
        elif distrib_name == "halfgaussian":
            u = np.abs(np.random.normal(0.0, distrib_opts["scale"]))
        elif distrib_name == "halflaplace":
            u = np.abs(np.random.laplace(0.0, distrib_opts["scale"]))
        elif distrib_name == "gausslaplace":
            theta = (distrib_opts["scale1"], distrib_opts["scale2"])

            def gauss_laplace_pdf(x, theta):
                scale1 = theta[0]
                scale2 = theta[1]
                c1 = np.sqrt(0.5 / (np.pi * scale2**2))
                c2 = erfc(1.0 / (scale1 * np.sqrt(2.0 / scale2**2)))
                c3 = (
                    (1.0 / scale1) * np.abs(x)
                    + 0.5 * (x / scale2) ** 2
                    + 0.5 * (scale2 / scale1) ** 2
                )
                return (c1 / c2) * np.exp(-c3)

            u = self.inverse_transform_sampling(
                gauss_laplace_pdf,
                theta,
                -2.326 / distrib_opts["scale2"] ** 2,  # 0.01 normal quantile
                2.326 / distrib_opts["scale2"] ** 2,  # 0.99 normal quantile
            )
        else:
            raise ValueError(f"Unknown distribution name {distrib_name}.")
        return u

    def calibrate_penalty(
        self,
        x_true: ArrayLike,
        sigma: float,
        n: int,
        distrib_name: str = "gaussian",
        distrib_opts: dict = {},
    ):

        if distrib_name == "gaussian":
            penalty = L2norm(0.5 * (sigma / distrib_opts["scale"]) ** 2)
        elif distrib_name == "laplace":
            penalty = BigmL1norm(
                M=np.max(np.abs(x_true)),
                alpha=sigma**2 / distrib_opts["scale"],
            )
        elif distrib_name == "uniform":
            penalty = Bounds(
                distrib_opts["low"] * np.ones(n),
                distrib_opts["high"] * np.ones(n),
            )
        elif distrib_name == "halfgaussian":
            penalty = PositiveL2norm(
                0.5 * (sigma / distrib_opts["scale"]) ** 2
            )
        elif distrib_name == "halflaplace":
            penalty = BigmPositiveL1norm(
                M=np.max(np.abs(x_true)),
                alpha=sigma**2 / distrib_opts["scale"],
            )
        elif distrib_name == "gausslaplace":
            penalty = L1L2norm(
                sigma**2 / distrib_opts["scale1"],
                0.5 * (sigma / distrib_opts["scale2"]) ** 2,
            )
        return penalty

    def calibrate_lambda(self, sigma: float, k: int, n: int):
        return sigma**2 * np.log((n - k) / k)

    def sample_data(
        self,
        k: int = 10,
        m: int = 500,
        n: int = 1000,
        r: float = 0.5,
        s: float = 10.0,
        distrib_name: str = "gaussian",
        distrib_opts: dict = {},
        seed=None,
    ):
        r"""Generate synthetic sparse regression data

        .. math:: y = Ax + e

        where

        - :math:`x \in R^n` is the ground truth vector with k non-zero entries
        evenly-spaced over the support and with amplitude drawn from the
        distribution specified in `distrib_name` and `distrib_opts`.

        - :math:`A \in R^{m \times n}` is a design matrix with i.i.d. columns
        drawn as :math:`a_i \sim N(0,K)` where :math:`K_{ij} = r^{|i-j|}`.

        - :math:`e` is a white Gaussian noise vector with signal-to-noise ratio
        :math:`s` with respect to :math:`y`.

        Parameters
        ----------
        k : int
            Number of non-zero entries targeted in the ground truth.
        m : int
            Number of rows in the design matrix.
        n : int
            Number of columns in the design matrix.
        r : float
            Correlation coefficient between columns of the design matrix.
        s : float
            Signal-to-noise ratio.
        distrib_name : str
            Name of the distribution to sample the amplitudes of the ground
            truth. See `sample_distribution` for more details.
        distrib_opts : dict
            Options of the distribution to sample the amplitudes of the ground
            truth. See `sample_distribution` for more details.
        """

        assert n >= k > 0
        assert m > 0
        assert 0.0 <= r < 1.0
        assert s > 0.0

        if seed is not None:
            np.random.seed(seed)

        # Ground truth
        x = np.zeros(n)
        for i in np.linspace(0, n - 1, num=k, dtype=int):
            x[i] = self.sample_distribution(distrib_name, distrib_opts)

        # Design matrix
        M = np.zeros(n)
        N1 = np.repeat(np.arange(n).reshape(n, 1), n).reshape(n, n)
        N2 = np.repeat(np.arange(n).reshape(1, n), n).reshape(n, n).T
        K = np.power(r, np.abs(N1 - N2))
        A = np.random.multivariate_normal(M, K, size=m)
        A /= np.linalg.norm(A, axis=0, ord=2)

        # Noise vector
        w = A @ x
        sigma = np.sqrt((w @ w) / (m * s))
        e = np.random.normal(0.0, sigma, m)

        # Observation vector
        y = w + e

        return A, y, x, np.sqrt(sigma)

    def setup(self) -> None:

        A, y, x_true, sigma = self.sample_data(**self.config["dataset"])

        datafit = Leastsquares(y)
        penalty = self.calibrate_penalty(
            x_true,
            sigma,
            self.config["dataset"]["n"],
            self.config["dataset"]["distrib_name"],
            self.config["dataset"]["distrib_opts"],
        )
        lmbd = self.calibrate_lambda(
            sigma,
            self.config["dataset"]["k"],
            self.config["dataset"]["n"],
        )

        self.A = A
        self.y = y
        self.x_true = x_true
        self.datafit = datafit
        self.penalty = penalty
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
                    solver_keys["solver"], solver_keys["params"]
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

        return result

    def cleanup(self) -> None:
        pass

    def plot(self, results: list) -> None:

        perfs = {}
        for result in results:
            for solver_name, solver_result in result.items():
                if solver_name not in perfs:
                    perfs[solver_name] = []
                if solver_result is not None:
                    if solver_result.status == Status.OPTIMAL:
                        perfs[solver_name].append(solver_result.solve_time)

        tgrid = np.logspace(
            np.floor(
                np.log10(
                    np.nanmin(
                        [
                            np.nanmin(times) if len(times) else np.nan
                            for times in perfs.values()
                        ]
                    )
                )
            ),
            np.ceil(
                np.log10(
                    np.nanmax(
                        [
                            np.nanmax(times) if len(times) else np.nan
                            for times in perfs.values()
                        ]
                    )
                )
            ),
            100,
        )
        curves = {solver: [] for solver in perfs}

        for solver_name, solver_times in perfs.items():
            for t in tgrid:
                count = sum(solver_time <= t for solver_time in solver_times)
                curves[solver_name].append(count)

        plt.figure(figsize=(10, 6))
        for solver_name, solver_curve in curves.items():
            plt.plot(tgrid, solver_curve, label=solver_name)

        plt.xlabel("time")
        plt.xscale("log")
        plt.ylabel("instances solved")
        plt.title(
            "{} {}".format(
                self.config["dataset"]["distrib_name"],
                self.config["dataset"]["distrib_opts"],
            )
        )
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        curves["tgrid"] = tgrid

        return curves

    def save_plot(self, curves, save_dir):

        name = "{}-k={:d}-m={:d}-n={:d}-r={:.2f}-s={:.2f}-{}={}-{}".format(
            "mixtures",
            self.config["dataset"]["k"],
            self.config["dataset"]["m"],
            self.config["dataset"]["n"],
            self.config["dataset"]["r"],
            self.config["dataset"]["s"],
            "distrib",
            self.config["dataset"]["distrib_name"],
            "-".join(
                [
                    f"{k}={v}"
                    for k, v in self.config["dataset"]["distrib_opts"].items()
                ]
            ),
        )
        df = pd.DataFrame(curves)
        df.to_csv(
            pathlib.Path(save_dir).joinpath(name).with_suffix(".csv"),
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["run", "plot"])
    parser.add_argument("--config_path", "-c", type=str)
    parser.add_argument("--result_dir", "-r", type=str)
    parser.add_argument("--save_dir", "-s", type=str, default=None)
    parser.add_argument("--repeats", "-n", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    runner = Runner(verbose=args.verbose)

    if args.command == "run":
        runner.run(
            Mixtures,
            args.config_path,
            args.result_dir,
            args.repeats,
        )
    elif args.command == "plot":
        runner.plot(
            Mixtures,
            args.config_path,
            args.result_dir,
            args.save_dir,
        )
    else:
        raise ValueError(f"Unknown command {args.command}.")
