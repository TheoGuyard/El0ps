import numpy as np
from numba import njit
from numpy.typing import NDArray
from el0ps.datafit import BaseDatafit, SmoothDatafit
from el0ps.penalty import BasePenalty
from el0ps.problem import Problem
from el0ps.solver.node import BnbNode


class BoundingSolver:
    r"""Node bounding problem solver.

    The bounding problem has the form

    .. math:: \textstyle\min_{x} f(Ax) + \sum_{i=1}^{n}g_i(x_i)

    with

    .. math::

        \begin{cases}
            g_i(t) = 0 &\text{if} \ i \in S_0 \\
            g_i(t) = h(t) + \lambda &\text{if} \ i \in S_1 \\
            g_i(t) = \tau|t| &\text{if} \ i \in S_{\bullet} \ \text{and} \ |t| \leq \mu \\
            g_i(t) = h(t) + \lambda &\text{if} \ i \in S_{\bullet} \ \text{and} \ |t| > \mu \\
        \end{cases}

    where :math:`f(\cdot)`, :math:`\lambda` and :math:`h(\cdot)` are the
    elements of :class:`.Problem` and where :math:`S_0`, :math:`S_1` and
    :math:`S_{\bullet}` are sets of indices forced to be zero, non-zero and
    unfixed, respectively, at the current node of the Branch-and-Bound tree.
    The quantities :math:`\tau` and :math:`\mu` are the ones returned by the
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

    def setup(self, problem: Problem) -> None:
        self.A_colnorm = np.linalg.norm(problem.A, ord=2, axis=0) ** 2
        self.tau = problem.penalty.param_slope(problem.lmbd)
        self.mu = problem.penalty.param_limit(problem.lmbd)
        self.rho = 1.0 / (problem.datafit.L * self.A_colnorm)
        self.eta = self.tau * self.rho
        self.delta = self.eta + self.mu

    def bound(
        self,
        problem: Problem,
        node: BnbNode,
        ub: float,
        rel_tol: float,
        workingsets: bool,
        dualpruning: bool,
        l1screening: bool,
        l0screening: bool,
        upper: bool = False,
    ):
        # Handle the root case and case where the upper-bounding problem yields
        # the same solutiona s the parent node.
        if upper:
            if not np.any(node.S1):
                node.x_inc = np.zeros(problem.n)
                node.upper_bound = problem.datafit.value(np.zeros(problem.m))
                return
            elif node.category == 0:
                return

        # ----- Initialization ----- #

        S0 = (node.S0 | node.Sb) if upper else node.S0
        S1 = node.S1
        Sb = (node.Sb & False) if upper else node.Sb
        Sb0 = node.Sb & False
        Sbi = node.Sb | True
        Sb1 = node.Sb & False
        x = node.x_inc if upper else node.x
        w = problem.A[:, S1] @ x[S1] if upper else node.w
        u = -problem.datafit.gradient(w)
        v = np.empty(problem.n)
        p = np.empty(problem.n)
        pv = np.inf
        dv = np.nan
        Ws = S1 | (x != 0.0 if workingsets else Sb)

        # ----- Outer loop ----- #

        rel_tol_inner = 0.5 * rel_tol
        for _ in range(self.maxiter_outer):
            v = np.empty(problem.n)
            p = np.empty(problem.n)
            dv = np.nan

            # ----- Inner loop ----- #

            for _ in range(self.maxiter_inner):
                pv_old = pv
                self.cd_loop(
                    problem.datafit,
                    problem.penalty,
                    problem.A,
                    self.rho,
                    self.eta,
                    self.delta,
                    x,
                    w,
                    u,
                    Ws,
                    Sb,
                )
                pv = self.compute_pv(
                    problem.datafit,
                    problem.penalty,
                    problem.lmbd,
                    self.tau,
                    self.mu,
                    x,
                    w,
                    S1,
                    Sb,
                )

                # ----- Inner solver stopping criterion ----- #

                if upper:
                    dv = self.compute_dv(
                        problem.datafit,
                        problem.penalty,
                        problem.A,
                        problem.lmbd,
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
                problem.penalty,
                problem.A,
                problem.lmbd,
                self.tau,
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
                dv = self.compute_dv(
                    problem.datafit,
                    problem.penalty,
                    problem.A,
                    problem.lmbd,
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
                dv = self.compute_dv(
                    problem.datafit,
                    problem.penalty,
                    problem.A,
                    problem.lmbd,
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
                        problem.datafit,
                        problem.A,
                        x,
                        w,
                        u,
                        v,
                        self.tau,
                        pv,
                        dv,
                        Ws,
                        Sb0,
                        Sbi,
                        Sb1,
                    )  # noqa
                if l0screening:
                    self.l0screening(
                        problem.datafit,
                        problem.A,
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
                        Sbi,
                        Sb1,
                    )  # noqa

        # ----- Post-processing ----- #

        if upper:
            node.upper_bound = pv
        else:
            dv = self.compute_dv(
                problem.datafit,
                problem.penalty,
                problem.A,
                problem.lmbd,
                u,
                v,
                p,
                S1,
                Sb,
            )
            node.lower_bound = dv

    @staticmethod
    @njit
    def cd_loop(
        datafit: SmoothDatafit,
        penalty: BasePenalty,
        A: NDArray[np.float64],
        rho: NDArray[np.float64],
        eta: NDArray[np.float64],
        delta: NDArray[np.float64],
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        u: NDArray[np.float64],
        Ws: NDArray[np.bool_],
        Sb: NDArray[np.bool_],
    ) -> float:
        for i in np.flatnonzero(Ws):
            ai = A[:, i]
            xi = x[i]
            ci = xi + rho[i] * np.dot(ai, u)
            if Sb[i] and np.abs(ci) <= delta[i]:
                x[i] = np.sign(ci) * np.maximum(np.abs(ci) - eta[i], 0.0)
            else:
                x[i] = penalty.prox(ci, rho[i])
            if x[i] != xi:
                w += (x[i] - xi) * ai
                u[:] = -datafit.gradient(w)

    @staticmethod
    @njit
    def ws_update(
        penalty: BasePenalty,
        A: NDArray[np.float64],
        lmbd: float,
        tau: float,
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
            if np.abs(v[i]) > tau:
                flag = True
                Ws[i] = True
        return flag

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
        """Compute the primal value of the bounding problem.

        Parameters
        ----------
        datafit: BaseDatafit
            Datafit function.
        penalty: BasePenalty
            Penalty function.
        lmbd: float
            Constant offset of the penalty.
        tau: float
            L1-norm weight.
        mu: float
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
    def rel_gap(pv: float, dv: float) -> float:
        """Relative duality gap between primal and dual values.

        Parameters
        ----------
        pv: float
            Primal value.
        dv: float
            Dual value.
        """
        return (pv - dv) / (np.abs(pv) + 1e-16)

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
        tau: float,
        pv: float,
        dv: float,
        Ws: NDArray[np.bool_],
        Sb0: NDArray[np.bool_],
        Sbi: NDArray[np.bool_],
        Sb1: NDArray[np.bool_],
    ) -> None:
        r = np.sqrt(2.0 * np.abs(pv - dv) * datafit.L)
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
