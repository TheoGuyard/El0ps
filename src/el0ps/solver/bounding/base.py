import numpy as np
from abc import abstractmethod
from typing import Union
from el0ps.problem import Problem
from el0ps.solver import BnbNode
from numpy.typing import NDArray


class BnbBoundingSolver:
    """Base class for for Branch-and-bound bounding solvers."""

    @abstractmethod
    def setup(
        self,
        problem: Problem,
        x_init: Union[NDArray[np.float64], None] = None,
        S0_init: Union[NDArray[np.bool_], None] = None,
        S1_init: Union[NDArray[np.bool_], None] = None,
    ) -> None:
        """Initialize internal attributes of the bounding solver.

        Parameters
        ----------
        problem: Problem
            Problem to solve.
        x_init: Union[NDArray[np.float64], None]
            Warm-start value of x.
        S0_init: Union[NDArray[np.bool_], None]
            Set of indices of x forced to be zero.
        S1_init: Union[NDArray[np.bool_], None]
            Set of indices of x forced to be non-zero.
        """
        ...

    @abstractmethod
    def bound(
        self,
        problem: Problem,
        node: BnbNode,
        upper_bound: float,
        abs_tol: float,
        rel_tol: float,
        l1screening: bool,
        l0screening: bool,
        incumbent: bool = False,
    ):
        """Solve the bounding problem at a given node of the Branch-and-Bound.

        Parameters
        ----------

        problem: Problem
            Problem data.
        node: BnbNode
            Node to bound.
        upper_bound: float
            Best upper bound in the BnB algorithm.
        abs_tol: float
            Absolute MIP tolerance of the BnB algorithm.
        rel_tol: float
            Relative MIP tolerance of the BnB algorithm.
        l1screening: bool
            Whether to use screening acceleration.
        l0screening: bool
            Whether to use node-screening acceleration.
        incumbent: bool = False
            Whether to generate an incumbent solution instead of performing a
            lower bounding operation.
        """
        ...
