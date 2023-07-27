import numpy as np
from typing import Union
from numpy.typing import NDArray
from numba import njit
from el0ps.problem import Problem
from .base import BnbBoundingSolver


@njit
def prox_l1_scalar(x: float, eta: float) -> float:
    return np.sign(x) * np.maximum(np.abs(x) - eta, 0.0)


@njit
def abs_gap(pv, dv):
    return np.abs(pv - dv)


@njit
def rel_gap(pv, dv):
    return np.abs(pv - dv) / (np.abs(pv) + 1e-16)


class CdBoundingSolver(BnbBoundingSolver):
    def __init__(
        self, rel_gap_tol=1e-4, iter_limit_cd=1_000, iter_limit_as=100
    ):
        self.rel_gap_tol = rel_gap_tol
        self.iter_limit_cd = iter_limit_cd
        self.iter_limit_as = iter_limit_as

    def setup(
        self,
        problem: Problem,
        x_init: Union[NDArray, None] = None,
        S0_init: Union[NDArray, None] = None,
        S1_init: Union[NDArray, None] = None,
    ) -> None:
        self.A_colnorm = np.linalg.norm(problem.A, ord=2, axis=0) ** 2

    def compute_pv(self, datafit, penalty, lmbd, x, w, tau, mu, S, Sb):
        fval = datafit.value(w)
        gval = 0.0
        for i in np.nonzero(S)[0]:
            xi = x[i]
            if np.logical_and(Sb[i], np.abs(xi) <= mu[i]):
                gval += tau[i] * np.abs(xi)
            else:
                gval += penalty.value_scalar(i, xi) + lmbd
        return fval + gval

    def compute_dv(self, datafit, penalty, A, lmbd, u, v, p, S, Sb):
        cfval = datafit.conjugate(-u)
        cgval = 0.0
        for i in np.nonzero(S)[0]:
            v[i] = np.dot(A[:, i], u)
            p[i] = penalty.conjugate_scalar(i, v[i]) - lmbd
            cgval += np.maximum(p[i], 0.0) if Sb[i] else p[i]
        return -cfval - cgval

    def cd_loop(self, datafit, penalty, A, x, w, u, rho, eta, delta, S, Sb):
        for i in np.nonzero(S)[0]:
            ai = A[:, i]
            xi = x[i]
            ci = xi + rho[i] * np.dot(ai, u)
            if Sb[i] and (np.abs(ci) <= delta[i]):
                x[i] = prox_l1_scalar(ci, eta[i])
            else:
                x[i] = penalty.prox_scalar(i, ci, rho[i])
            if x[i] != xi:
                w += (x[i] - xi) * ai
                u[:] = -datafit.gradient(w)

    def bound(self, problem, node, bnb, bounding_type):
        # Problem data
        datafit = problem.datafit
        penalty = problem.penalty
        A = problem.A
        lmbd = problem.lmbd

        # Node data
        if bounding_type == "lower":
            S1 = node.S1
            Sb = node.Sb
            x = node.x
            w = node.w
            u = node.u
        else:
            S1 = node.S1
            Sb = np.zeros(node.Sb.shape, dtype=bool)
            x = np.zeros(node.x.shape)
            x[S1] = node.x[S1]
            w = A[:, S1] @ x[S1]
            u = -datafit.gradient(w)

        # Support configuration
        Sbi = np.copy(Sb)
        S = np.logical_or(x != 0.0, S1)

        # Parameter values
        time_limit = bnb.options.time_limit
        abs_tol = bnb.options.abs_tol
        rel_tol = bnb.options.rel_tol
        rel_tol_cd = 0.5 * self.rel_gap_tol

        # Working values
        tau = np.array(
            [penalty.param_levlimit(i, lmbd) for i in range(x.size)]
        )
        mu = np.array([penalty.param_sublimit(i, lmbd) for i in range(x.size)])
        rho = 1.0 / (datafit.L * self.A_colnorm)
        eta = tau * rho
        delta = eta + mu
        v = np.empty(x.shape)
        p = np.empty(x.shape)

        # BnB upper bound and primal/dual objective values
        ub = bnb.upper_bound
        pv = +np.inf
        dv = np.nan

        # ----- Outer active set loop ----- #

        it_cd = 0
        it_as = 0
        while True:
            it_as += 1
            v[:] = np.nan
            p[:] = np.nan
            dv = np.nan

            # ----- Inner coordinate descent solver ----- #

            while True:
                it_cd += 1
                pv_old = pv

                self.cd_loop(
                    datafit, penalty, A, x, w, u, rho, eta, delta, S, Sb
                )
                pv = self.compute_pv(
                    datafit, penalty, lmbd, x, w, tau, mu, S, Sb
                )

                # Inner solver convergence criterion
                #   - lower bounding: no more progress in primal objective
                #   - upper bounding: dual gap agrees with target mip gap
                if bounding_type == "lower":
                    if rel_gap(pv, pv_old) <= rel_tol_cd:
                        break
                else:
                    dv = self.compute_dv(
                        datafit, penalty, A, lmbd, u, v, p, S, Sb
                    )
                    # print(f"dv : {dv}")
                    if (
                        abs_gap(pv, dv) <= abs_tol
                        and rel_gap(pv, dv) <= rel_tol
                    ):
                        break

                # Inner solver stopping criterion
                if it_cd >= self.iter_limit_cd:
                    break

            # ----- Active-set update ----- #

            flag = False
            for i in np.nonzero(np.logical_and(np.logical_not(S), Sbi))[0]:
                v[i] = np.dot(A[:, i], u)
                p[i] = penalty.conjugate_scalar(i, v[i]) - lmbd
                if np.abs(v[i]) > tau[i]:
                    flag = True
                    S[i] = True

            # ----- Stopping criterion ----- #

            if bounding_type == "upper":
                break
            if it_as >= self.iter_limit_as:
                break
            if bnb.solve_time >= time_limit:
                break
            if not flag:
                dv = self.compute_dv(datafit, penalty, A, lmbd, u, v, p, S, Sb)
                if rel_gap(pv, dv) < rel_tol:
                    break
                if dv >= ub:
                    break
                if rel_tol_cd <= 1e-8:
                    break
                rel_tol_cd *= 1e-2

        # ----- Post-processing ----- #

        if bounding_type == "lower":
            if np.isnan(dv):
                dv = self.compute_dv(datafit, penalty, A, lmbd, u, v, p, S, Sb)
            node.lower_bound = dv
        else:
            node.upper_bound = pv
            node.x_inc = np.copy(x)
