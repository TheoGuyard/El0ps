import numpy as np
from abc import abstractmethod
from typing import Union
from numba import njit
from numpy.typing import NDArray
from el0ps.datafit import BaseDatafit
from el0ps.problem import Problem
from el0ps.solver.node import BnbNode


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

    @staticmethod
    @njit
    def l0screening(
        datafit: BaseDatafit,
        A: NDArray[np.float64],
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        u: NDArray[np.float64],
        p: NDArray[np.float64],
        ub: float,
        dv: float,
        S0: NDArray[np.bool_],
        S1: NDArray[np.bool_],
        Sb: NDArray[np.bool_],
        Ws: NDArray[np.bool_],
        Sbi: NDArray[np.bool_],
        Sb1: NDArray[np.bool_],
    ) -> None:
        for i in np.flatnonzero(Sb & ~np.isnan(p)):
            if dv + np.maximum(-p[i], 0.0) > ub:
                Sb[i] = False
                S0[i] = True
                Ws[i] = False
                Sbi[i] = False
                Sb1[i] = False
                if x[i] != 0.0:
                    w -= x[i] * A[:, i]
                    u[:] = -datafit.gradient(w)
                    x[i] = 0.0
            elif dv + np.maximum(p[i], 0.0) > ub:
                Sb[i] = False
                S1[i] = True
                Ws[i] = True
                Sbi[i] = False
                Sb1[i] = False

    @staticmethod
    @njit
    def l1screening(
        datafit: BaseDatafit,
        A: NDArray[np.float64],
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        L: float,
        tau: float,
        pv: float,
        dv: float,
        Ws: NDArray[np.bool_],
        Sb0: NDArray[np.bool_],
        Sbi: NDArray[np.bool_],
        Sb1: NDArray[np.bool_],
    ) -> None:
        r = np.sqrt(2.0 * np.abs(pv - dv) * L)
        for i in np.flatnonzero(Sbi & ~np.isnan(v)):
            vi = v[i]
            if np.abs(vi) + r < tau:
                if x[i] != 0.0:
                    w -= x[i] * A[:, i]
                    u[:] = -datafit.gradient(w)
                    x[i] = 0.0
                Ws[i] = False
                Sbi[i] = False
                Sb0[i] = True
            elif np.abs(vi) - r > tau:
                Ws[i] = True
                Sbi[i] = False
                Sb1[i] = True
