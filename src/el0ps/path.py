import sys
import numpy as np
from typing import Union
from numba.experimental.jitclass.base import JitClassType
from numpy.typing import NDArray

from el0ps.utils import compute_lmbd_max
from el0ps.compilation import compiled_clone
from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty
from el0ps.solver import BaseSolver, Result, Status


class Path:
    """Regularization path fitting for L0-regularized problems expressed as

        `min_{x in R^n} f(Ax) + lmbd * |x|_0 + h(x)`

    where `f` is a datafit function, `A` is a matrix, `h` is a penalty
    function, and `lmbd` is a positive scalar. The path fitting consists of
    solving the problem over a range of values of parameter `lmbd`.

    Parameters
    ----------
    lmbds : Union[None, list] = None
        Values of parameter `lmbd` to consider. If `None`, the values
        considered is computed from the other parameters `lmbd_max`,
        `lmbd_min`, `lmbd_num`, `lmbd_normalized`, and `lmbd_spacing`.
    lmbd_max : float = 1e-0
        Maximum value of `lmbd` to consider. If `lmbds` is not `None`, this
        value is ignored.
    lmbd_min : float = 1e-2
        Minimum value of `lmbd` to consider. If `lmbds` is not `None`, this
        value is ignored.
    lmbd_num : int = 10
        Number of values of `lmbd` to consider. If `lmbds` is not `None`, this
        value is ignored.
    lmbd_scale : str = "log"
        Scale of the values of `lmbd` to consider, either `lin` or `log`. If
        `lmbds` is not `None`, this value is ignored.
    lmbd_normalized : bool = True
        If `True`, the values of `lmbd` considered are scaled by the
        value outputted by the function `el0ps.utils.compute_lmbd_max` so that
        at `lmbd_max`, the solution to the problem is the all-zero vector. If
        `lmbds` is not `None`, this value is ignored.
    max_nnz : int = sys.maxsize
        Stop the path fitting when a solution with more than `max_nnz`
        non-zero coefficients is found for a given value of `lmbd`.
    stop_if_not_optimal : bool = True
        Stop the path fitting when the problem at a given value of `lmbd`
        is not solved to optimality.
    verbose : bool = True
        Toggle displays during path fitting.
    """  # noqa: W605

    def __init__(
        self,
        lmbds: Union[None, list] = None,
        lmbd_max: float = 1e-0,
        lmbd_min: float = 1e-2,
        lmbd_num: int = 10,
        lmbd_scale: str = "log",
        lmbd_normalized: bool = True,
        max_nnz: int = sys.maxsize,
        stop_if_not_optimal: bool = True,
        verbose: bool = True,
    ) -> None:
        self.lmbds = lmbds
        self.lmbd_max = lmbd_max
        self.lmbd_min = lmbd_min
        self.lmbd_num = lmbd_num
        self.lmbd_scale = lmbd_scale
        self.lmbd_normalized = lmbd_normalized
        self.max_nnz = max_nnz
        self.stop_if_not_optimal = stop_if_not_optimal
        self.verbose = verbose

    def get_lmbd_grid(
        self,
        datafit: Union[BaseDatafit, JitClassType],
        penalty: Union[BasePenalty, JitClassType],
        A: NDArray,
    ) -> NDArray:

        if self.lmbds is not None:
            lmbd_grid = np.array(self.lmbds)
        elif self.lmbd_scale == "log":
            lmbd_grid = np.logspace(
                np.log10(self.lmbd_max),
                np.log10(self.lmbd_min),
                self.lmbd_num
            )
        elif self.lmbd_scale == "lin":
            lmbd_grid = np.linspace(
                self.lmbd_max,
                self.lmbd_min,
                self.lmbd_num
            )
        else:
            raise ValueError(
                "Invalid value for parameter `lmbd_scale`. Must be either "
                "'log' or 'lin'."
            )

        if self.lmbd_normalized:
            lmbd_max_grid = compute_lmbd_max(datafit, penalty, A)
            lmbd_grid *= lmbd_max_grid / np.max(lmbd_grid)

        return lmbd_grid

    def display_path_head(self) -> None:
        s = "  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}".format(
            "lambda",
            "status",
            "time",
            "objective",
            "num nnz",
        )
        print("-" * len(s))
        print(s)
        print("-" * len(s))

    def display_path_info(self, lmbd: float, result: Result) -> None:
        print(
          "  {:>10.2f}  {:>10}  {:>10.4f}  {:>10.2f}  {:>10d}".format(
                lmbd,
                result.status,
                result.solve_time,
                result.objective_value,
                np.count_nonzero(result.x),
            )
        )

    def display_path_foot(self) -> None:
        print("-" * 60)

    def fit(
        self,
        solver: BaseSolver,
        datafit: Union[BaseDatafit, JitClassType],
        penalty: Union[BasePenalty, JitClassType],
        A: NDArray,
    ) -> dict:
        """Fit the regularization path.

        Parameters
        ----------
        datafit: BaseDatafit
            Datafit function.
        penalty: BasePenalty
            Penalty function.
        A: NDArray
            Linear operator.

        Returns
        -------
        results: dict
            The path fitting results indexed by the values of `lmbd`.
        """

        if solver.accept_jitclass:
            if isinstance(datafit, BaseDatafit):
                datafit = compiled_clone(datafit)
            if isinstance(penalty, BasePenalty):
                penalty = compiled_clone(penalty)
        if not A.flags.f_contiguous:
            A = np.array(A, order="F")

        lmbd_grid = self.get_lmbd_grid(datafit, penalty, A)

        if self.verbose:
            self.display_path_head()

        x_init = None
        results = {}

        for lmbd in lmbd_grid:

            # Solve the problem for the current value of lmbd
            result = solver.solve(datafit, penalty, A, lmbd, x_init)

            # Store results
            results[lmbd] = result
            x_init = np.copy(result.x)

            # Displays
            if self.verbose:
                self.display_path_info(lmbd, result)

            # Termination criteria
            if self.stop_if_not_optimal and result.status != Status.OPTIMAL:
                break
            if np.count_nonzero(result.x) > self.max_nnz:
                break

        if self.verbose:
            self.display_path_foot()

        return results
