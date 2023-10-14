import numpy as np
from typing import Union
from numpy.typing import NDArray
from numba import njit
from el0ps.datafit import SmoothDatafit
from el0ps.penalty import BasePenalty
from el0ps.problem import Problem
from el0ps.solver import BnbNode
from .base import BnbBoundingSolver, softhresh


class CdBoundingSolver(BnbBoundingSolver):
    """Coordinate descent bounding solver.

    Parameters
    ----------
    iter_limit_cd: int
        Maximum number of iterations of the inner coordinate descent solver.
    iter_limit_as: int
        Maximum number of iterations of the outer active-set loop.
    """

    def __init__(self, iter_limit_cd=1_000, iter_limit_as=100):
        self.iter_limit_cd = iter_limit_cd
        self.iter_limit_as = iter_limit_as

    def setup(
        self,
        problem: Problem,
        x_init: Union[NDArray[np.float64], None] = None,
        S0_init: Union[NDArray[np.bool_], None] = None,
        S1_init: Union[NDArray[np.bool_], None] = None,
    ) -> None:
        self.L = problem.datafit.L
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
        l1screening: bool,
        l0screening: bool,
        incumbent: bool = False,
    ):
        # Handle the root case and case where the upper-bounding problem yields
        # the same solutiona s the parent node.
        if incumbent:
            if not np.any(node.S1):
                node.x_inc = np.zeros(problem.n)
                node.upper_bound = problem.datafit.value(np.zeros(problem.m))
                return
            elif node.category == 0:
                return

        # Working values
        datafit = problem.datafit
        penalty = problem.penalty
        A = problem.A
        lmbd = problem.lmbd
        L = self.L
        tau = self.tau
        mu = self.mu
        rho = self.rho
        eta = self.eta
        delta = self.delta
        v = np.empty(problem.n)
        p = np.empty(problem.n)
        if incumbent:
            S0 = node.S0 | node.Sb
            S1 = node.S1
            Sb = np.zeros(node.Sb.shape, dtype=np.bool_)
            x = node.x_inc
            w = A[:, S1] @ x[S1]
            u = -datafit.gradient(w)
        else:
            S0 = node.S0
            S1 = node.S1
            Sb = node.Sb
            x = node.x
            w = node.w
            u = node.u

        # Working set configuration
        Sb0 = np.zeros(node.Sb.shape, dtype=np.bool_)
        Sbi = np.copy(Sb)
        Sb1 = np.zeros(node.Sb.shape, dtype=np.bool_)
        Ws = S1 | (x != 0.0)

        # Primal and dual objective values
        pv = np.inf
        dv = np.nan

        # ----- Outer active set loop ----- #

        rel_tol_inner = 0.5 * rel_tol
        it_as = 0
        while True:
            v[:] = np.nan
            p[:] = np.nan
            dv = np.nan

            # ----- Inner coordinate descent solver ----- #

            it_cd = 0
            while True:
                it_cd += 1
                pv_old = pv
                self._cd_loop(
                    datafit,
                    penalty,
                    A,
                    lmbd,
                    tau,
                    mu,
                    rho,
                    eta,
                    delta,
                    x,
                    w,
                    u,
                    Ws,
                    Sb,
                )
                pv = self.compute_pv(
                    datafit, penalty, lmbd, tau, mu, x, w, S1, Sb
                )

                # Inner solver stopping criterion
                #   - in lower bounding: no more progress in primal objective
                #   - in upper bounding: dual gap below the target mip gap
                #   - in both cases: maximum number of iterations reached
                if incumbent:
                    dv = self.compute_dv(
                        datafit, penalty, A, lmbd, u, v, p, S1, Sb
                    )
                    if self.rel_gap(pv, dv) <= rel_tol:
                        break
                elif self.rel_gap(pv, pv_old) <= rel_tol_inner:
                    break
                elif it_cd >= self.iter_limit_cd:
                    break

            # ----- Active-set update ----- #

            flag = self._ws_update(penalty, A, lmbd, tau, u, v, p, Ws, Sbi)

            # ----- Stopping criterion ----- #

            # Outer solver stopping criterion
            #   - in lower bounding: no optimality conditions are violated
            #     and that one the following condition is met:
            #       i) the relative tolearance is met
            #       ii) the dual value is above the best upper bound
            #       iii) the tolearance of the inner solver is almost zero
            #     If optimality conditions are not violated but none of the
            #     above conditions are met, this means that the inner solver
            #     tolearance must be decreased.
            #   - in upper bounding: stops since the active set is fixed
            #   - in both cases: maximum number of iterations reached
            if incumbent == "upper":
                break
            if not flag:
                dv = self.compute_dv(
                    datafit, penalty, A, lmbd, u, v, p, S1, Sb
                )
                if self.rel_gap(pv, dv) < rel_tol:
                    break
                if dv >= ub:
                    break
                if rel_tol_inner <= 1e-8:
                    break
                rel_tol_inner *= 1e-2
            if it_as >= self.iter_limit_as:
                break

            # ----- Accelerations ----- #

            if not incumbent and (l1screening or l0screening):
                if np.isnan(dv):
                    dv = self.compute_dv(
                        datafit, penalty, A, lmbd, u, v, p, S1, Sb
                    )
                if l1screening:
                    self.l1screening(
                        datafit,
                        A,
                        x,
                        w,
                        u,
                        v,
                        L,
                        tau,
                        pv,
                        dv,
                        Ws,
                        Sb0,
                        Sbi,
                        Sb1,
                    )  # noqa
                if l0screening:
                    self.l0screening(
                        datafit,
                        A,
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

        if incumbent:
            node.upper_bound = pv
        else:
            if np.isnan(dv):
                dv = self.compute_dv(
                    datafit, penalty, A, lmbd, u, v, p, S1, Sb
                )
            node.lower_bound = dv

    @staticmethod
    @njit
    def _cd_loop(
        datafit: SmoothDatafit,
        penalty: BasePenalty,
        A: NDArray[np.float64],
        lmbd: float,
        tau: float,
        mu: float,
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
                x[i] = softhresh(ci, eta[i])
            else:
                x[i] = penalty.prox(ci, rho[i])
            if x[i] != xi:
                w += (x[i] - xi) * ai
                u[:] = -datafit.gradient(w)

    @staticmethod
    @njit
    def _ws_update(
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
