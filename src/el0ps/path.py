import sys
import numpy as np
from dataclasses import dataclass
from typing import Union
from numba.experimental.jitclass.base import JitClassType
from numpy.typing import ArrayLike
from el0ps.utils import compiled_clone, compute_lmbd_max
from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty
from el0ps.solver import BaseSolver, Result, Status


@dataclass
class PathOptions:
    """``Path`` options.

    Parameters
    ----------
    lmbd_ratio_max: float = 1e-0
        Maximum value of ``lmbd/lmbd_max`` in the path. The value ``lmbd_max``
        is computed using the :func:`.compute_lmbd_max` function.
    lmbd_ratio_min: float = 1e-2
        Minimum value of ``lmbd/lmbd_max`` in the path. The value ``lmbd_max``
        is computed using the :func:`.compute_lmbd_max` function.
    lmbd_ratio_num: int = 10
        Number of different values of ``lmbd`` in the path.
    max_nnz: int = sys.maxsize
        Stop the path fitting when a solution with more than ``max_nnz``
        non-zero coefficients is found.
    stop_if_not_optimal: bool = True
        Stop the path fitting when the problem at a given value of ``lmbd`` is
        not solved to optimality.
    verbose: bool = True
        Toogle displays during path fitting.
    """

    lmbd_ratio_max: float = 1e-0
    lmbd_ratio_min: float = 1e-2
    lmbd_ratio_num: int = 10
    max_nnz: int = sys.maxsize
    stop_if_not_optimal: bool = True
    verbose: bool = True

    def _validate_types(self):
        for field_name, field_def in self.__dataclass_fields__.items():
            actual_type = type(getattr(self, field_name))
            if not issubclass(actual_type, field_def.type):
                raise ValueError(
                    "Expected '{}' for argument '{}', got '{}'.".format(
                        field_def.type, field_name, actual_type
                    )
                )

    def __post_init__(self):
        self._validate_types()
        if not 0.0 <= self.lmbd_ratio_min <= self.lmbd_ratio_max <= 1.0:
            raise ValueError(
                "Parameters must satisfy "
                "`0 <= lmbd_ratio_min <= lmbd_ratio_max <= 1`."
            )
        if not self.lmbd_ratio_num >= 0.0:
            raise ValueError("Parameters `lmbd_ratio_num` must be positive.")
        if not self.max_nnz >= 0.0:
            raise ValueError("Parameters max_nnz must be positive.")


class Path:
    """L0-regularization path.

    The L0-regularization path corresponds to different solutions of an
    L0-penalized problem when the regularization parameter ``lmbd`` is varied.

    Parameters
    ----------
    kwargs: dict
        Path options passed to :class:`path.PathOptions`.

    Attributes
    ----------
    options: PathOptions
        Path options.
    fit_data: dict
        Path fitting data, see `_path_keys` attributes for the keys considered.
    """

    _path_hstr = "   ratio   status     time    nodes  obj. val  los. val  pen. val  n-zero"  # noqa: E501
    _path_fstr = (
        "{:>7.2e}  {:>7}  {:>7.2f}  {:>7}  {:>7.2e}  {:>7.2e}  {:>7.2e} {:>7}"
    )
    _path_keys = [
        "lmbd_ratio",
        "status",
        "solve_time",
        "iter_count",
        "rel_gap",
        "x",
        "lmbd_value",
        "objective_value",
        "datafit_value",
        "penalty_value",
        "n_nnz",
    ]

    def __init__(self, **kwargs) -> None:
        self.options = PathOptions(**kwargs)
        self.fit_data = {k: [] for k in self._path_keys}

    def display_path_head(self) -> None:
        ruler = "-" * len(self._path_hstr)
        print(f"{ruler}\n{self._path_hstr}\n{ruler}")

    def display_path_info(self) -> None:
        path_istr = self._path_fstr.format(
            *[
                self.fit_data[k][-1]
                for k in [
                    "lmbd_ratio",
                    "status",
                    "solve_time",
                    "iter_count",
                    "objective_value",
                    "datafit_value",
                    "penalty_value",
                    "n_nnz",
                ]
            ]
        )
        print(path_istr)

    def display_path_foot(self) -> None:
        print("-" * len(self._path_hstr))

    def can_continue(self) -> bool:
        if (
            self.options.stop_if_not_optimal
            and not self.fit_data["status"][-1] == Status.OPTIMAL
        ):
            return False
        if self.fit_data["n_nnz"][-1] > self.options.max_nnz:
            return False
        return True

    def fill_fit_data(
        self,
        lmbd_ratio: float,
        datafit: JitClassType,
        penalty: JitClassType,
        A: ArrayLike,
        lmbd: float,
        results: Result,
    ) -> None:
        for k in self._path_keys:
            if k == "lmbd_ratio":
                self.fit_data[k].append(lmbd_ratio)
            elif k == "lmbd_value":
                self.fit_data[k].append(lmbd)
            elif k == "datafit_value":
                self.fit_data[k].append(datafit.value(A @ results.x))
            elif k == "penalty_value":
                self.fit_data[k].append(
                    np.sum([penalty.value(xi) for xi in results.x])
                )
            else:
                self.fit_data[k].append(getattr(results, k))

    def fit(
        self,
        solver: BaseSolver,
        datafit: Union[BaseDatafit, JitClassType],
        penalty: Union[BasePenalty, JitClassType],
        A: ArrayLike,
    ) -> dict:
        """Fit the regularization path.

        Parameters
        ----------
        datafit: BaseDatafit
            Datafit function.
        penalty: BasePenalty
            Penalty function.
        A: ArrayLike
            Linear operator.

        Returns
        -------
        fit_data: dict
            The path fitting data stored in ``self.fit_data``.
        """

        if not str(type(datafit)).startswith(
            "<class 'numba.experimental.jitclass"
        ):
            datafit = compiled_clone(datafit)
        if not str(type(penalty)).startswith(
            "<class 'numba.experimental.jitclass"
        ):
            penalty = compiled_clone(penalty)
        if not A.flags.f_contiguous:
            A = np.array(A, order="F")

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
            lmbd = lmbd_ratio * lmbd_max
            results = solver.solve(datafit, penalty, A, lmbd, x_init=x_init)
            x_init = np.copy(results.x)
            self.fill_fit_data(lmbd_ratio, datafit, penalty, A, lmbd, results)
            if self.options.verbose:
                self.display_path_info()
            if not self.can_continue():
                break

        if self.options.verbose:
            self.display_path_foot()

        return self.fit_data

    def reset_fit(self) -> None:
        """Reset the path fitting data."""
        self.fit_data = {k: [] for k in self._path_keys}
