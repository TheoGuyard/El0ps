import sys
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from el0ps.problem import Problem, compute_lmbd_max
from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty
from el0ps.solver import BaseSolver, Results, Status


@dataclass
class PathOptions:
    lmbd_ratio_max: float = 1e-0
    lmbd_ratio_min: float = 1e-2
    lmbd_ratio_num: int = 10
    max_nnz: int = sys.maxsize
    stop_if_not_optimal: bool = True
    verbose: bool = True

    def validate_types(self):
        for field_name, field_def in self.__dataclass_fields__.items():
            actual_type = type(getattr(self, field_name))
            if not issubclass(actual_type, field_def.type):
                raise ValueError(
                    "Expected '{}' for argument '{}', got '{}'.".format(
                        field_def.type, field_name, actual_type
                    )
                )

    def __post_init__(self):
        self.validate_types()
        if not 0.0 <= self.lmbd_ratio_min <= self.lmbd_ratio_max <= 1.0:
            raise ValueError(
                "Parameters must satisfy"
                "`0 <= lmbd_ratio_min <= lmbd_ratio_max <= 1`."
            )
        if not self.lmbd_ratio_num >= 0.0:
            raise ValueError("Parameters `lmbd_ratio_num` must be positive.")
        if not self.max_nnz >= 0.0:
            raise ValueError("Parameters max_nnz must be positive.")


class Path:
    path_hstr = "   ratio   status     time    nodes    value     nnz"
    path_fstr = "{:>7.2e}  {:>7}  {:>7.2f}  {:>7}  {:>7.2f} {:>7}"
    path_keys = [
        "lmbd_ratio",
        "status",
        "solve_time",
        "node_count",
        "rel_gap",
        "x",
        "objective_value",
        "n_nnz",
    ]

    def __init__(self, **kwargs) -> None:
        self.options = PathOptions(**kwargs)
        self.fit_data = {k: [] for k in self.path_keys}

    def reset_fit(self) -> None:
        self.fit_data = {k: [] for k in self.path_keys}

    def display_path_head(self) -> None:
        ruler = "-" * len(self.path_hstr)
        print(f"{ruler}\n{self.path_hstr}\n{ruler}")

    def display_path_info(self) -> None:
        path_istr = self.path_fstr.format(
            *[
                self.fit_data[k][-1]
                for k in [
                    "lmbd_ratio",
                    "status",
                    "solve_time",
                    "node_count",
                    "objective_value",
                    "n_nnz",
                ]
            ]
        )
        print(path_istr)

    def display_path_foot(self) -> None:
        print("-" * len(self.path_hstr))

    def can_continue(self) -> bool:
        if (
            self.options.stop_if_not_optimal
            and not self.fit_data["status"][-1] == Status.OPTIMAL
        ):
            return False
        if self.fit_data["n_nnz"][-1] > self.options.max_nnz:
            return False
        return True

    def fill_fit_data(self, lmbd_ratio: float, results: Results) -> None:
        for k in self.path_keys:
            if k == "lmbd_ratio":
                self.fit_data[k].append(lmbd_ratio)
            else:
                self.fit_data[k].append(getattr(results, k))

    def fit(
        self,
        solver: BaseSolver,
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: NDArray,
    ) -> dict:
        if self.options.verbose:
            self.display_path_head()

        lmbd_ratio_grid = np.logspace(
            np.log10(self.options.lmbd_ratio_max),
            np.log10(self.options.lmbd_ratio_min),
            self.options.lmbd_ratio_num,
        )
        lmbd_max = compute_lmbd_max(datafit, penalty, A)
        x_init = np.zeros(A.shape[1])

        for lmbd_ratio in lmbd_ratio_grid:
            problem = Problem(datafit, penalty, A, lmbd_ratio * lmbd_max)
            results = solver.solve(problem, x_init=x_init)
            x_init = np.copy(results.x)
            self.fill_fit_data(lmbd_ratio, results)
            if self.options.verbose:
                self.display_path_info()
            if not self.can_continue():
                break

        if self.options.verbose:
            self.display_path_foot()

        return self.fit_data
