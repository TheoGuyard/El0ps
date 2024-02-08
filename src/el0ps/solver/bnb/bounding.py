import numpy as np
import time
from typing import Union
from numba import njit
from numba.experimental.jitclass.base import JitClassType
from numpy.typing import NDArray
from el0ps.datafit import BaseDatafit, SmoothDatafit
from el0ps.penalty import BasePenalty
from el0ps.utils import compiled_clone
from .node import BnbNode


class BnbBoundingSolver:
    r"""Node bounding problem solver.

    The bounding problem has the form

    .. math:: \textstyle\min_{x} f(Ax) + \sum_{i=1}^{n}g_i(x_i)

    with

    .. math::

        \begin{cases}
            g_i(t) = 0 &\text{if} \ i \in S_0 \\
            g_i(t) = h(t) + \lambda &\text{if} \ i \in S_1 \\
            g_i(t) = \c1|t| &\text{if} \ i \in S_{\bullet} \ \text{and} \ |t| \leq \c2 \\
            g_i(t) = h(t) + \lambda &\text{if} \ i \in S_{\bullet} \ \text{and} \ |t| > \c2 \\
        \end{cases}

    where :math:`f(\cdot)`, :math:`\lambda` and :math:`h(\cdot)` are parts of
    the problem objective function and where :math:`S_0`, :math:`S_1` and
    :math:`S_{\bullet}` are sets of indices forced to be zero, non-zero and
    unfixed, respectively, at the current node of the Branch-and-Bound tree.
    The quantities :math:`\c1` and :math:`\c2` are the ones returned by the
    :func:`penalty.BasePenalty.param_pertslope` and
    :func:`penalty.BasePenalty.param_pertlimit` methods, respectively.
    """  # noqa: E501

    def __init__(
        self,
        maxiter_inner=1_000,
        maxiter_outer=100,
    ):
        self.maxiter_inner = maxiter_inner
        self.maxiter_outer = maxiter_outer

    def setup(
        self,
        datafit: Union[BaseDatafit, JitClassType],
        penalty: Union[BasePenalty, JitClassType],
        A: NDArray,
        lmbd: float,
    ) -> None:
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

        # Problem data
        self.m, self.n = A.shape
        self.datafit = datafit
        self.penalty = penalty
        self.A = A
        self.lmbd = lmbd

        # Precomputed constants
        self.c1 = penalty.param_slope(lmbd)
        self.c2 = penalty.param_limit(lmbd)
        self.c3 = np.linalg.norm(A, ord=2, axis=0) ** 2
        self.c4 = datafit.L * self.c3
        self.c5 = 1.0 / self.c4
        self.c6 = self.c1 * self.c5
        self.c7 = self.c6 + self.c2

    def bound(
        self,
        node: BnbNode,
        ub: float,
        rel_tol: float,
        workingsets: bool,
        dualpruning: bool,
        l1screening: bool,
        l0screening: bool,
        upper: bool = False,
    ):
        start_time = time.time()

        # ----- Initialization ----- #

        S0 = (node.S0 | node.Sb) if upper else node.S0
        S1 = node.S1
        Sb = np.zeros(self.n, dtype=np.bool_) if upper else node.Sb
        Sb0 = np.zeros(self.n, dtype=np.bool_)
        Sbi = np.copy(Sb)
        x = node.x_inc if upper else node.x
        w = self.A[:, S1] @ x[S1] if upper else node.w
        u = -self.datafit.gradient(w)
        v = np.empty(self.n)
        p = np.empty(self.n)
        pv = np.inf
        dv = np.nan
        Ws = S1 | (x != 0.0 if workingsets else Sb)

        # ----- Outer loop ----- #

        rel_tol_inner = 0.5 * rel_tol
        for _ in range(self.maxiter_outer):
            v = np.empty(self.n)
            p = np.empty(self.n)
            dv = np.nan

            # ----- Inner loop ----- #

            for _ in range(self.maxiter_inner):
                pv_old = pv
                self.cd_loop(
                    self.datafit,
                    self.penalty,
                    self.A,
                    self.c5,
                    self.c6,
                    self.c7,
                    x,
                    w,
                    u,
                    Ws,
                    Sb,
                )
                pv = self.compute_pv(
                    self.datafit,
                    self.penalty,
                    self.lmbd,
                    self.c1,
                    self.c2,
                    x,
                    w,
                    S1,
                    Sb,
                )

                # ----- Inner solver stopping criterion ----- #

                if upper:
                    dv = self.compute_dv(
                        self.datafit,
                        self.penalty,
                        self.A,
                        self.lmbd,
                        u,
                        v,
                        p,
                        S1,
                        Sb,
                    )
                    if self.rel_gap(pv, dv) <= rel_tol:
                        break
                elif self.rel_gap(pv, pv_old) <= rel_tol_inner:
                    break

            # ----- Working-set update ----- #

            flag = self.ws_update(
                self.penalty,
                self.A,
                self.lmbd,
                self.c1,
                u,
                v,
                p,
                Ws,
                Sbi,
            )

            # ----- Stopping criterion ----- #

            if upper:
                break
            if not flag:
                if np.isnan(dv):
                    dv = self.compute_dv(
                        self.datafit,
                        self.penalty,
                        self.A,
                        self.lmbd,
                        u,
                        v,
                        p,
                        S1,
                        Sb,
                    )
                if self.rel_gap(pv, dv) < rel_tol:
                    break
                if rel_tol_inner <= 1e-8:
                    break
                rel_tol_inner *= 1e-2

            # ----- Accelerations ----- #

            if dualpruning or l1screening or l0screening:
                if np.isnan(dv):
                    dv = self.compute_dv(
                        self.datafit,
                        self.penalty,
                        self.A,
                        self.lmbd,
                        u,
                        v,
                        p,
                        S1,
                        Sb,
                    )
                if dualpruning and dv > ub:
                    break
                if l1screening:
                    self.l1screening(
                        self.datafit,
                        self.A,
                        x,
                        w,
                        u,
                        v,
                        self.c1,
                        self.c4,
                        pv,
                        dv,
                        Ws,
                        Sb0,
                        Sbi,
                    )  # noqa
                if l0screening:
                    self.l0screening(
                        self.datafit,
                        self.A,
                        x,
                        w,
                        u,
                        p,
                        ub,
                        dv,
                        S0,
                        S1,
                        Sb,
                        Ws,
                        Sb0,
                        Sbi,
                    )  # noqa

        # ----- Post-processing ----- #

        if upper:
            node.upper_bound = pv
            node.time_upper_bound = time.time() - start_time
        else:
            if np.isnan(dv):
                dv = self.compute_dv(
                    self.datafit,
                    self.penalty,
                    self.A,
                    self.lmbd,
                    u,
                    v,
                    p,
                    S1,
                    Sb,
                )
            node.lower_bound = dv
            node.time_lower_bound = time.time() - start_time

    @staticmethod
    @njit
    def cd_loop(
        datafit: SmoothDatafit,
        penalty: BasePenalty,
        A: NDArray[np.float64],
        c5: NDArray[np.float64],
        c6: NDArray[np.float64],
        c7: NDArray[np.float64],
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        u: NDArray[np.float64],
        Ws: NDArray[np.bool_],
        Sb: NDArray[np.bool_],
    ) -> float:
        for i in np.flatnonzero(Ws):
            ai = A[:, i]
            xi = x[i]
            ci = xi + c5[i] * np.dot(ai, u)
            if Sb[i] and np.abs(ci) <= c7[i]:
                x[i] = np.sign(ci) * np.maximum(np.abs(ci) - c6[i], 0.0)
            else:
                x[i] = penalty.prox(ci, c5[i])
            if x[i] != xi:
                w += (x[i] - xi) * ai
                u[:] = -datafit.gradient(w)

    @staticmethod
    @njit
    def ws_update(
        penalty: BasePenalty,
        A: NDArray[np.float64],
        lmbd: float,
        c1: float,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        p: NDArray[np.float64],
        Ws: NDArray[np.bool_],
        Sbi: NDArray[np.bool_],
    ) -> bool:
        flag = False
        for i in np.flatnonzero(~Ws & Sbi):
            v[i] = np.dot(A[:, i], u)
            p[i] = penalty.conjugate(v[i]) - lmbd
            if np.abs(v[i]) > c1:
                flag = True
                Ws[i] = True
        return flag

    @staticmethod
    @njit
    def compute_pv(
        datafit: BaseDatafit,
        penalty: BasePenalty,
        lmbd: float,
        c1: float,
        c2: float,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        S1: NDArray[np.bool_],
        Sb: NDArray[np.bool_],
    ) -> float:
        """Compute the primal value of the bounding problem.

        Parameters
        ----------
        datafit: BaseDatafit
            Datafit function.
        penalty: BasePenalty
            Penalty function.
        lmbd: float
            Constant offset of the penalty.
        c1: float
            L1-norm weight.
        c2: float
            L1-norm threshold.
        x: NDArray[np.float64]
            Value at which the primal is evaluated.
        w: NDArray[np.float64]
            Value of ``A @ x``.
        S1: NDArray[np.bool_]
            Set of indices forced to be non-zero.
        Sb: NDArray[np.bool_]
            Set of unfixed indices.
        """
        pv = datafit.value(w)
        for i in np.flatnonzero(S1 | Sb):
            if Sb[i] and np.abs(x[i]) <= c2:
                pv += c1 * np.abs(x[i])
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
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        p: NDArray[np.float64],
        S1: NDArray[np.bool_],
        Sb: NDArray[np.bool_],
    ) -> float:
        """Compute the dual value of the bounding problem.

        Parameters
        ----------
        datafit: BaseDatafit
            Datafit function.
        penalty: BasePenalty
            Penalty function.
        A: NDArray[np.float64]
            Linear operator.
        lmbd: float
            Constant offset of the penalty.
        u: NDArray[np.float64]
            Value at which the dual is evaluated.
        w: NDArray[np.float64]
            Value of ``A.T @ u``.
        p: NDArray[np.float64]
            Empty vector to store the values of ``penalty.conjugate(v) - lmbd``
            over indices in ``S1`` or ``Sb``.
        S1: NDArray[np.bool_]
            Set of indices forced to be non-zero.
        Sb: NDArray[np.bool_]
            Set of unfixed indices.
        """
        dv = -datafit.conjugate(-u)
        for i in np.flatnonzero(S1 | Sb):
            v[i] = np.dot(A[:, i], u)
            p[i] = penalty.conjugate(v[i]) - lmbd
            dv -= np.maximum(p[i], 0.0) if Sb[i] else p[i]
        return dv

    @staticmethod
    @njit
    def rel_gap(pv: float, dv: float) -> float:
        """Relative duality gap between primal and dual values.

        Parameters
        ----------
        pv: float
            Primal value.
        dv: float
            Dual value.
        """
        return (pv - dv) / (np.abs(pv) + 1e-10)

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
        Sb0: NDArray[np.bool_],
        Sbi: NDArray[np.bool_],
    ) -> None:
        flag = False
        for i in np.flatnonzero(Sb):
            if dv + np.maximum(-p[i], 0.0) > ub:
                Sb[i] = False
                S0[i] = True
                Ws[i] = False
                Sb0[i] = False
                Sbi[i] = False
                if x[i] != 0.0:
                    w -= x[i] * A[:, i]
                    x[i] = 0.0
                    flag = True
            elif dv + np.maximum(p[i], 0.0) > ub:
                Sb[i] = False
                S1[i] = True
                Ws[i] = True
                Sb0[i] = False
                Sbi[i] = False
        if flag:
            u[:] = -datafit.gradient(w)

    @staticmethod
    @njit
    def l1screening(
        datafit: BaseDatafit,
        A: NDArray[np.float64],
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        c1: float,
        c4: float,
        pv: float,
        dv: float,
        Ws: NDArray[np.bool_],
        Sb0: NDArray[np.bool_],
        Sbi: NDArray[np.bool_],
    ) -> None:
        flag = False
        r = np.sqrt(2.0 * np.abs(pv - dv) * c4)
        for i in np.flatnonzero(Sbi):
            vi = v[i]
            if np.abs(vi) + r[i] < c1:
                if x[i] != 0.0:
                    w -= x[i] * A[:, i]
                    x[i] = 0.0
                    flag = True
                Ws[i] = False
                Sbi[i] = False
                Sb0[i] = True
        if flag:
            u[:] = -datafit.gradient(w)
