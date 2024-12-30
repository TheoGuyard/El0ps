import sys
import numpy as np
from typing import Union
from numba.experimental.jitclass.base import JitClassType
from numpy.typing import ArrayLike

from el0ps.utils import compute_lmbd_max
from el0ps.compilation import compiled_clone
from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty
from el0ps.solver import BaseSolver, Result, Status, BnbSolver


class Path:
    """Path fitting for L0-regularized problems.

    The optimization problem considered is

    .. math:: \min f(Xw) + \lambda \|w\|_0 + h(w)

    where :math:`f` is a datafit term, :math:`h` is a penalty term and
    :math:`\lambda` is the L0-norm weight. It is solved for a range of values
    of the parameter :math:`\lambda`.

    Parameters
    ----------
    lmbd_max: float = 1e-0
        Maximum value of ``lmbd`` in the path.
    lmbd_min: float = 1e-2
        Minimum value of ``lmbd`` in the path.
    lmbd_num: int = 10
        Number of values of ``lmbd`` in the path.
    lmbd_scaled: bool = False
        If false, the values of :math:`\lambda` in the path are scaled by a
        factor :math:`\lambda_{\max}` so that when `\lambda=1`, the solution
        to the problem is the all-zero vector. The value of
        :math:`\lambda_{\max}` is computed using the
        :func:`.utils.compute_lmbd_max` function.
    max_nnz: int = sys.maxsize
        Stop the path fitting when a solution with more than ``max_nnz``
        non-zero coefficients is found.
    stop_if_not_optimal: bool = True
        Stop the path fitting when the problem at a given value of ``lmbd`` is
        not solved to optimality.
    verbose: bool = True
        Toogle displays during path fitting.
    """  # noqa: W605

    _path_hstr = "  lambda   status     time    nodes  obj. val  los. val  pen. val  n-zero"  # noqa: E501
    _path_fstr = (
        "{:>7.2e}  {:>7}  {:>7.2f}  {:>7}  {:>7.2e}  {:>7.2e}  {:>7.2e} {:>7}"
    )
    _path_keys = [
        "lmbd",
        "status",
        "solve_time",
        "iter_count",
        "x",
        "objective_value",
        "datafit_value",
        "penalty_value",
        "n_nnz",
    ]

    def __init__(
        self,
        lmbd_max: float = 1e-0,
        lmbd_min: float = 1e-2,
        lmbd_num: int = 10,
        lmbd_scaled: bool = True,
        max_nnz: int = sys.maxsize,
        stop_if_not_optimal: bool = True,
        verbose: bool = True,
    ) -> None:
        self.lmbd_max = lmbd_max
        self.lmbd_min = lmbd_min
        self.lmbd_num = lmbd_num
        self.lmbd_scaled = lmbd_scaled
        self.max_nnz = max_nnz
        self.stop_if_not_optimal = stop_if_not_optimal
        self.verbose = verbose
        self.fit_data = {k: [] for k in self._path_keys}

    def display_path_head(self) -> None:
        ruler = "-" * len(self._path_hstr)
        print(f"{ruler}\n{self._path_hstr}\n{ruler}")

    def display_path_info(self) -> None:
        path_istr = self._path_fstr.format(
            *[
                self.fit_data[k][-1]
                for k in [
                    "lmbd",
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
            self.stop_if_not_optimal
            and not self.fit_data["status"][-1] == Status.OPTIMAL
        ):
            return False
        if self.fit_data["n_nnz"][-1] > self.max_nnz:
            return False
        return True

    def fill_fit_data(
        self,
        datafit: JitClassType,
        penalty: JitClassType,
        A: ArrayLike,
        lmbd: float,
        results: Result,
    ) -> None:
        for k in self._path_keys:
            if k == "lmbd":
                self.fit_data[k].append(lmbd)
            elif k == "datafit_value":
                self.fit_data[k].append(datafit.value(A @ results.x))
            elif k == "penalty_value":
                self.fit_data[k].append(penalty.value(results.x))
            elif k == "n_nnz":
                self.fit_data[k].append(np.flatnonzero(results.x).size)
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

        if isinstance(solver, BnbSolver):
            if isinstance(datafit, BaseDatafit):
                datafit = compiled_clone(datafit)
            if isinstance(penalty, BasePenalty):
                penalty = compiled_clone(penalty)
        if not A.flags.f_contiguous:
            A = np.array(A, order="F")

        if self.verbose:
            self.display_path_head()

        lmbd_grid = np.logspace(
            np.log10(self.lmbd_max),
            np.log10(self.lmbd_min),
            self.lmbd_num,
        )
        if self.lmbd_scaled:
            lmbd_max = compute_lmbd_max(datafit, penalty, A)
            lmbd_grid *= lmbd_max
        x_init = np.zeros(A.shape[1])

        for lmbd in lmbd_grid:
            results = solver.solve(datafit, penalty, A, lmbd, x_init)
            x_init = np.copy(results.x)
            self.fill_fit_data(datafit, penalty, A, lmbd, results)
            if self.verbose:
                self.display_path_info()
            if not self.can_continue():
                break

        if self.verbose:
            self.display_path_foot()

        return self.fit_data

    def reset_fit(self) -> None:
        """Reset the path fitting data."""
        self.fit_data = {k: [] for k in self._path_keys}
