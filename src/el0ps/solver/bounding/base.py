import numpy as np
from abc import abstractmethod
from typing import Union
from numba import njit
from numpy.typing import NDArray
from el0ps.datafit import BaseDatafit
from el0ps.penalty import BasePenalty
from el0ps.problem import Problem
from el0ps.solver.node import BnbNode


@njit
def softhresh(x: float, eta: float) -> float:
    return np.sign(x) * np.maximum(np.abs(x) - eta, 0.0)


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
    def compute_pv(
        datafit: BaseDatafit,
        penalty: BasePenalty,
        lmbd: float,
        tau: float,
        mu: float,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        S1: NDArray[np.bool_],
        Sb: NDArray[np.bool_],
    ) -> float:
        pv = datafit.value(w)
        for i in np.flatnonzero(S1 | (Sb & (x != 0.0))):
            if Sb[i] and np.abs(x[i]) <= mu:
                pv += tau * np.abs(x[i])
            else:
                pv += penalty.value(x[i]) + lmbd
        return pv

    @staticmethod
    @njit
    def compute_dv(
        datafit: BaseDatafit,
        penalty: BasePenalty,
        A: NDArray[np.float64],
        lmbd: float,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        p: NDArray[np.float64],
        S1: NDArray[np.bool_],
        Sb: NDArray[np.bool_],
    ) -> float:
        nz = np.flatnonzero(S1 | Sb)
        sf = np.empty(v.shape)
        for i in nz:
            v[i] = np.dot(A[:, i], u)
            p[i] = penalty.conjugate(v[i]) - lmbd
            sf[i] = penalty.conjugate_scaling_factor(v[i])
        g_sf = 1.0 if nz.size == 0 else np.min(sf[nz])
        u_sf = g_sf * u
        v_sf = g_sf * v
        dv = -datafit.conjugate(-u_sf)
        for i in nz:
            p_sf = penalty.conjugate(v_sf[i]) - lmbd
            dv -= np.maximum(p_sf, 0.0) if Sb[i] else p_sf
        return dv

    @staticmethod
    @njit
    def abs_gap(pv: float, dv: float) -> float:
        return np.abs(pv - dv)

    @staticmethod
    @njit
    def rel_gap(pv: float, dv: float) -> float:
        return np.abs(pv - dv) / (np.abs(pv) + 1e-16)

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
