import cvxpy as cp
import numpy as np
import time
from abc import abstractmethod
from numba import njit, float64
from numba.experimental.jitclass.base import JitClassType
from numpy.typing import ArrayLike
from el0ps.utils import compiled_clone
from .node import BnbNode


def calibrate_mcptwo(
    datafit: JitClassType,
    penalty: JitClassType,
    A: ArrayLike,
    regfunc_type: str,
):
    n = A.shape[1]
    if regfunc_type == "convex":
        mcptwo = 2.0 * penalty.alpha * np.ones(n)
    elif regfunc_type == "concave_eig":
        G = datafit.strong_convexity_constant() * (A.T @ A)
        g = np.min(np.real(np.linalg.eigvals(G)))
        mcptwo = (g + 2.0 * penalty.alpha) * np.ones(n)
    elif regfunc_type == "concave_etp":
        G = datafit.strong_convexity_constant() * (A.T @ A)
        var = cp.Variable(n)
        obj = cp.Maximize(cp.sum(var))
        cst = [
            G + 2.0 * penalty.alpha * np.eye(n) - cp.diag(var) >> 0.0,
            var >= 0.0,
        ]
        problem = cp.Problem(obj, cst)
        problem.solve()
        mcptwo = np.array(var.value).flatten()
    else:
        raise ValueError(f"Invalid regfunc_type {regfunc_type}.")
    return mcptwo


class BaseRegfunc:
    """Base class for regularization functions that are lower-bounds on

    ..math: g(x) = h(x) + lmbd * |x|_0

    where `h` corresponds to a `BasePenalty`.
    """

    @abstractmethod
    def get_spec(self) -> tuple:
        """Specify the numba types of the class attributes.

        Returns
        -------
        spec: Tuple of (attr_name, dtype)
            Specs to be passed to Numba jitclass to compile the class.
        """
        ...

    @abstractmethod
    def params_to_dict(self) -> dict:
        """Get the parameters to initialize an instance of the class.

        Returns
        -------
        dict_of_params: dict
            The parameters to instantiate an object of the class.
        """
        ...

    @abstractmethod
    def value(self, i: int, lmbd: float, x: float) -> float:
        """Value of the i-th splitting term of the function at ``x``.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        lmbd: float
            L0-norm weight.
        x: float
            Value at which the function is evaluated.

        Returns
        -------
        value: float
            The value of the i-th splitting term the function at ``x``.
        """
        ...

    @abstractmethod
    def conjugate(self, i: int, lmbd: float, x: float) -> float:
        """Value of the i-th splitting term the conjugate of the function at
        ``x``.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        lmbd: float
            L0-norm weight.
        x: float
            Value at which the conjugate is evaluated.

        Returns
        -------
        value: float
            The value of the conjugate of the i-th splitting term the function
            at ``x``.
        """
        ...

    @abstractmethod
    def prox(self, i: int, lmbd: float, x: float, eta: float) -> float:
        """Proximity operator of ``eta`` times the i-th splitting term the
        function at ``x``.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        lmbd: float
            L0-norm weight.
        x: float
            Value at which the prox is evaluated.
        eta: float, positive
            Multiplicative factor of the function.

        Returns
        -------
        p: float
            The proximity operator of ``eta`` times the the i-th splitting term
            function at ``x``.
        """
        ...

    @abstractmethod
    def subdiff(self, i: int, lmbd: float, x: float) -> tuple:
        """Subdifferential operator of the i-th splitting term the function at
        ``x``.

        Parameters
        ----------
        i: int
            Index of the splitting term.
        lmbd: float
            L0-norm weight.
        x: float
            Value at which the prox is evaluated.

        Returns
        -------
        s: float
            The subdifferential operator of the i-th splitting term of the
            function at ``x``.
        """
        ...


class ConvexRegfunc(BaseRegfunc):
    """Regularization function corresponding to the convex enveloppe of

    ..math: g(x) = h(x) + lmbd * |x|_0

    where `h` corresponds to a `BasePenalty`.
    """

    def __init__(self, penalty: JitClassType) -> None:
        self.penalty = penalty

    def get_spec(self) -> tuple:
        spec = (
            ("penalty", self.penalty._numba_type_.class_type.instance_type),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(penalty=self.penalty)

    def value(self, i: int, lmbd: float, x: float) -> float:
        z = np.abs(x)
        if z <= self.penalty.param_limit(i, lmbd):
            return self.penalty.param_slope(i, lmbd) * z
        else:
            return lmbd + self.penalty.value(i, x)

    def conjugate(self, i: int, lmbd: float, x: float) -> float:
        return np.maximum(self.penalty.conjugate(i, x) - lmbd, 0.0)

    def prox(self, i: int, lmbd: float, x: float, eta: float) -> float:
        s = self.penalty.param_slope(i, lmbd)
        z = np.abs(x)
        if z <= eta * s:
            return 0.0
        elif z <= eta * s + self.penalty.param_limit(i, lmbd):
            return x - eta * s * np.sign(x)
        else:
            return self.penalty.prox(i, x, eta)

    def subdiff(self, i: int, lmbd: float, x: float) -> ArrayLike:
        z = np.abs(x)
        if z == 0.0:
            s = self.penalty.param_slope(i, lmbd)
            return [-s, s]
        elif z < self.penalty.param_limit(i, lmbd):
            s = self.penalty.param_slope(i, lmbd) * np.sign(x)
            return [s, s]
        else:
            return self.penalty.subdiff(i, x)


class ConcaveRegfunc(BaseRegfunc):

    def __init__(
        self,
        penalty: JitClassType,
        mcptwo: ArrayLike,
    ) -> None:

        if str(penalty) != "L2norm":
            raise Exception

        self.penalty = penalty
        self.mcptwo = mcptwo

    def get_spec(self) -> tuple:
        spec = (
            ("penalty", self.penalty._numba_type_.class_type.instance_type),
            ("mcptwo", float64[:]),
        )
        return spec

    def params_to_dict(self) -> dict:
        return dict(
            penalty=self.penalty,
            mcptwo=self.mcptwo,
        )

    def mcp_value(self, i: int, lmbd: float, x: float) -> float:
        c = np.sqrt(2.0 * lmbd * self.mcptwo[i])
        z = np.abs(x)
        if z <= c / self.mcptwo[i]:
            return c * z - 0.5 * self.mcptwo[i] * z**2
        else:
            return lmbd

    def mcp_prox(self, i: int, lmbd: float, x: float, eta: float) -> float:
        c = np.sqrt(2.0 * lmbd * self.mcptwo[i])
        z = np.abs(x)
        if z <= c * eta:
            return 0.0
        if z > c / self.mcptwo[i]:
            return x
        return np.sign(x) * (z - c * eta) / (1.0 - eta * self.mcptwo[i])

    def mcp_subdiff(self, i: int, lmbd: float, x: float) -> ArrayLike:
        c = np.sqrt(2.0 * lmbd * self.mcptwo[i])
        z = np.abs(x)
        if z == 0.0:
            return [-c, c]
        elif z <= c / self.mcptwo[i]:
            s = np.sign(x) * c - self.mcptwo[i] * x
            return [s, s]
        else:
            return [1.0, 1.0]

    def value(self, i: int, lmbd: float, x: float) -> float:
        return self.mcp_value(i, lmbd, x) + self.penalty.value(i, x)

    def conjugate(self, i: int, lmbd: float, x: float) -> float:
        c = 1.0 / (2.0 * self.penalty.alpha)
        z = c * x
        p = self.mcp_prox(i, lmbd, z, c)
        return (
            0.5 * c * x**2
            - self.mcp_value(i, lmbd, p)
            - self.penalty.value(i, p - z)
        )

    def prox(self, i: int, lmbd: float, x: float, eta: float) -> float:
        c = 1.0 / (eta * 2.0 * self.penalty.alpha + 1.0)
        return self.mcp_prox(i, lmbd, c * x, c * eta)

    def subdiff(self, i: int, lmbd: float, x: float) -> ArrayLike:
        s_mcp = self.mcp_subdiff(i, lmbd, x)
        s_pen = self.penalty.subdiff(i, x)
        return [s_mcp[0] + s_pen[0], s_mcp[-1] + s_pen[-1]]


class BnbBoundingSolver:
    r"""Node bounding problem solver.

    The bounding problem has the form

    .. math:: \textstyle\min_{x} f(Ax) + \sum_{i=1}^{n}g_i(x_i)
    """

    def __init__(
        self,
        regfunc_type: str = "convex",
        maxiter_inner: int = 1_000,
        maxiter_outer: int = 100,
    ):
        self.regfunc_type = regfunc_type
        self.maxiter_inner = maxiter_inner
        self.maxiter_outer = maxiter_outer

    def setup(
        self,
        datafit: JitClassType,
        penalty: JitClassType,
        A: ArrayLike,
    ):

        # Problem data
        self.m, self.n = A.shape
        self.datafit = datafit
        self.penalty = penalty
        self.A = A

        # Regularization function
        if self.regfunc_type == "convex":
            regfunc = ConvexRegfunc(penalty)
        elif self.regfunc_type.startswith("concave"):
            mcptwo = calibrate_mcptwo(datafit, penalty, A, self.regfunc_type)
            regfunc = ConcaveRegfunc(penalty, mcptwo)
        else:
            raise ValueError(f"Invalid regfunc_type {self.regfunc_type}.")
        self.regfunc = compiled_clone(regfunc)

        # Constants
        self.lipschitz = self.datafit.lipschitz_constant()
        self.A_colnorm = np.linalg.norm(A, ord=2, axis=0)
        self.stepsize = 1.0 / (self.lipschitz * self.A_colnorm**2)

    def bound(
        self,
        node: BnbNode,
        lmbd: float,
        ub: float,
        rel_tol: float,
        workingsets: bool,
        dualpruning: bool,
        l1screening: bool,
        simpruning: bool,
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
        pv = np.inf
        dv = np.nan
        Ws = S1 | (x != 0.0 if workingsets else Sb)
        threshold = np.array(
            [self.regfunc.subdiff(i, lmbd, 0.0)[1] for i in range(x.size)]
        )

        # ----- Outer loop ----- #

        rel_tol_inner = 0.1 * rel_tol
        for _ in range(self.maxiter_outer):
            v = np.empty(self.n)
            dv = np.nan

            # ----- Inner loop ----- #

            for _ in range(self.maxiter_inner):

                # Inner problem solver
                pv_old = pv
                self.inner_solve(
                    self.datafit,
                    self.penalty,
                    self.regfunc,
                    self.A,
                    lmbd,
                    x,
                    w,
                    u,
                    Ws,
                    Sb,
                    self.stepsize,
                )
                pv = self.compute_pv(
                    self.datafit,
                    self.penalty,
                    self.regfunc,
                    lmbd,
                    x,
                    w,
                    S1,
                    Sb,
                )

                # Inner stopping criterion
                if upper:
                    dv = self.compute_dv(
                        self.datafit,
                        self.penalty,
                        self.regfunc,
                        self.A,
                        lmbd,
                        u,
                        v,
                        S1,
                        Sb,
                    )
                    if self.rel_gap(pv, dv) <= rel_tol:
                        break
                elif self.rel_gap(pv, pv_old) <= rel_tol_inner:
                    break

            # Working-set update
            flag = self.ws_update(
                self.regfunc, self.A, lmbd, u, v, Ws, Sbi, threshold
            )

            # Outer stopping criterion
            if upper:
                break
            if not flag:
                if np.isnan(dv):
                    dv = self.compute_dv(
                        self.datafit,
                        self.penalty,
                        self.regfunc,
                        self.A,
                        lmbd,
                        u,
                        v,
                        S1,
                        Sb,
                    )
                if self.rel_gap(pv, dv) < rel_tol:
                    break
                if rel_tol_inner <= 1e-12:
                    break
                rel_tol_inner *= 1e-2

            # Accelerations
            if dualpruning or l1screening or simpruning:
                if np.isnan(dv):
                    dv = self.compute_dv(
                        self.datafit,
                        self.penalty,
                        self.regfunc,
                        self.A,
                        lmbd,
                        u,
                        v,
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
                        pv,
                        dv,
                        Ws,
                        Sb0,
                        Sbi,
                        self.lipschitz,
                        self.A_colnorm,
                        threshold,
                    )  # noqa
                if simpruning:
                    self.simpruning(
                        self.datafit,
                        self.penalty,
                        self.regfunc,
                        self.A,
                        lmbd,
                        x,
                        w,
                        u,
                        v,
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
                    self.regfunc,
                    self.A,
                    lmbd,
                    u,
                    v,
                    S1,
                    Sb,
                )
            node.lower_bound = dv
            node.time_lower_bound = time.time() - start_time

    @staticmethod
    @njit
    def inner_solve(
        datafit: JitClassType,
        penalty: JitClassType,
        regfunc: JitClassType,
        A: ArrayLike,
        lmbd: float,
        x: ArrayLike,
        w: ArrayLike,
        u: ArrayLike,
        Ws: ArrayLike,
        Sb: ArrayLike,
        stepsize: ArrayLike,
    ) -> float:
        for i in np.flatnonzero(Ws):
            ai = A[:, i]
            xi = x[i]
            ci = xi + stepsize[i] * np.dot(ai, u)
            if Sb[i]:
                x[i] = regfunc.prox(i, lmbd, ci, stepsize[i])
            else:
                x[i] = penalty.prox(i, ci, stepsize[i])
            if x[i] != xi:
                w += (x[i] - xi) * ai
                u[:] = -datafit.gradient(w)

    @staticmethod
    @njit
    def ws_update(
        regfunc: JitClassType,
        A: ArrayLike,
        lmbd: float,
        u: ArrayLike,
        v: ArrayLike,
        Ws: ArrayLike,
        Sbi: ArrayLike,
        threshold: ArrayLike,
    ) -> bool:
        flag = False
        for i in np.flatnonzero(~Ws & Sbi):
            v[i] = np.dot(A[:, i], u)
            if np.abs(v[i]) > threshold[i]:
                flag = True
                Ws[i] = True
        return flag

    @staticmethod
    @njit
    def compute_pv(
        datafit: JitClassType,
        penalty: JitClassType,
        regfunc: JitClassType,
        lmbd: float,
        x: ArrayLike,
        w: ArrayLike,
        S1: ArrayLike,
        Sb: ArrayLike,
    ) -> float:
        """Compute the primal value of the bounding problem.

        Parameters
        ----------
        datafit: BaseDatafit
            Datafit function.
        penalty: BasePenalty
            Penalty function.
        regfunc: BaseRegfunc
            Regularization function.
        lmbd: float
            Constant offset of the penalty.
        x: ArrayLike
            Value at which the primal is evaluated.
        w: ArrayLike
            Value of ``A @ x``.
        S1: ArrayLike
            Set of indices forced to be non-zero.
        Sb: ArrayLike
            Set of unfixed indices.
        """
        pv = datafit.value(w)
        for i in np.flatnonzero(S1):
            pv += penalty.value(i, x[i]) + lmbd
        for i in np.flatnonzero(Sb):
            pv += regfunc.value(i, lmbd, x[i])
        return pv

    @staticmethod
    @njit
    def compute_dv(
        datafit: JitClassType,
        penalty: JitClassType,
        regfunc: JitClassType,
        A: ArrayLike,
        lmbd: float,
        u: ArrayLike,
        v: ArrayLike,
        S1: ArrayLike,
        Sb: ArrayLike,
    ) -> float:
        """Compute the dual value of the bounding problem.

        Parameters
        ----------
        datafit: BaseDatafit
            Datafit function.
        penalty: BasePenalty
            Penalty function.
        regfunc: BaseRegfunc
            Regularization function.
        A: ArrayLike
            Linear operator.
        lmbd: float
            Constant offset of the penalty.
        u: ArrayLike
            Value at which the dual is evaluated.
        w: ArrayLike
            Value of ``A.T @ u``.
        S1: ArrayLike
            Set of indices forced to be non-zero.
        Sb: ArrayLike
            Set of unfixed indices.
        """
        dv = -datafit.conjugate(-u)
        for i in np.flatnonzero(S1):
            v[i] = np.dot(A[:, i], u)
            dv -= penalty.conjugate(i, v[i]) - lmbd
        for i in np.flatnonzero(Sb):
            v[i] = np.dot(A[:, i], u)
            dv -= regfunc.conjugate(i, lmbd, v[i])
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
    def simpruning(
        datafit: JitClassType,
        penalty: JitClassType,
        regfunc: JitClassType,
        A: ArrayLike,
        lmbd: float,
        x: ArrayLike,
        w: ArrayLike,
        u: ArrayLike,
        v: ArrayLike,
        ub: float,
        dv: float,
        S0: ArrayLike,
        S1: ArrayLike,
        Sb: ArrayLike,
        Ws: ArrayLike,
        Sb0: ArrayLike,
        Sbi: ArrayLike,
    ) -> None:
        flag = False
        for i in np.flatnonzero(Sb):
            g = regfunc.conjugate(i, lmbd, v[i])
            p = penalty.conjugate(i, v[i]) - lmbd
            if dv + g - p > ub:
                Sb[i] = False
                S0[i] = True
                Ws[i] = False
                Sb0[i] = False
                Sbi[i] = False
                if x[i] != 0.0:
                    w -= x[i] * A[:, i]
                    x[i] = 0.0
                    flag = True
            elif dv + g > ub:
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
        datafit: JitClassType,
        A: ArrayLike,
        x: ArrayLike,
        w: ArrayLike,
        u: ArrayLike,
        v: ArrayLike,
        pv: float,
        dv: float,
        Ws: ArrayLike,
        Sb0: ArrayLike,
        Sbi: ArrayLike,
        lipschitz: float,
        A_colnorm: ArrayLike,
        threshold: ArrayLike,
    ) -> None:
        flag = False
        r = np.sqrt(2.0 * np.abs(pv - dv) * lipschitz)
        for i in np.flatnonzero(Sbi):
            vi = v[i]
            if np.abs(vi) + r * A_colnorm[i] < threshold[i]:
                if x[i] != 0.0:
                    w -= x[i] * A[:, i]
                    x[i] = 0.0
                    flag = True
                Ws[i] = False
                Sbi[i] = False
                Sb0[i] = True
        if flag:
            u[:] = -datafit.gradient(w)
