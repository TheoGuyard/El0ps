import numpy as np
import time
from abc import abstractmethod
from numba import njit
from numba.experimental.jitclass.base import JitClassType
from numpy.typing import ArrayLike
from el0ps.utils import compiled_clone
from .bnb_node import BnbNode


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
    def value_scalar(self, i: int, lmbd: float, x: float) -> float:
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
    def conjugate_scalar(self, i: int, lmbd: float, x: float) -> float:
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
    def prox_scalar(self, i: int, lmbd: float, x: float, eta: float) -> float:
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
    def subdiff_scalar(self, i: int, lmbd: float, x: float) -> tuple:
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

    def value_scalar(self, i: int, lmbd: float, x: float) -> float:
        z = np.abs(x)
        if z <= self.penalty.param_limit_scalar(i, lmbd):
            return self.penalty.param_slope_scalar(i, lmbd) * z
        else:
            return lmbd + self.penalty.value_scalar(i, x)

    def conjugate_scalar(self, i: int, lmbd: float, x: float) -> float:
        return np.maximum(self.penalty.conjugate_scalar(i, x) - lmbd, 0.0)

    def prox_scalar(self, i: int, lmbd: float, x: float, eta: float) -> float:
        s = self.penalty.param_slope_scalar(i, lmbd)
        z = np.abs(x)
        if z <= eta * s:
            return 0.0
        elif z <= eta * s + self.penalty.param_limit_scalar(i, lmbd):
            return x - eta * s * np.sign(x)
        else:
            return self.penalty.prox_scalar(i, x, eta)

    def subdiff_scalar(self, i: int, lmbd: float, x: float) -> ArrayLike:
        z = np.abs(x)
        if z == 0.0:
            s = self.penalty.param_slope_scalar(i, lmbd)
            return [-s, s]
        elif z < self.penalty.param_limit_scalar(i, lmbd):
            s = self.penalty.param_slope_scalar(i, lmbd) * np.sign(x)
            return [s, s]
        else:
            return self.penalty.subdiff_scalar(i, x)


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
        assert self.regfunc_type == "convex"
        regfunc = ConvexRegfunc(self.penalty)
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
        th = np.array(
            [self.penalty.param_slope_scalar(i, lmbd) for i in range(x.size)]
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
            flag = self.ws_update(self.A, u, v, Ws, Sbi, th)

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
                        th,
                        self.lipschitz,
                        self.A_colnorm,
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
                x[i] = regfunc.prox_scalar(i, lmbd, ci, stepsize[i])
            else:
                x[i] = penalty.prox_scalar(i, ci, stepsize[i])
            if x[i] != xi:
                w += (x[i] - xi) * ai
                u[:] = -datafit.gradient(w)

    @staticmethod
    @njit
    def ws_update(
        A: ArrayLike,
        u: ArrayLike,
        v: ArrayLike,
        Ws: ArrayLike,
        Sbi: ArrayLike,
        th: ArrayLike,
    ) -> bool:
        flag = False
        for i in np.flatnonzero(~Ws & Sbi):
            v[i] = np.dot(A[:, i], u)
            if np.abs(v[i]) > th[i]:
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
            pv += penalty.value_scalar(i, x[i]) + lmbd
        for i in np.flatnonzero(Sb):
            pv += regfunc.value_scalar(i, lmbd, x[i])
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
        v: ArrayLike
            Value of ``A.T @ u``.
        S1: ArrayLike
            Set of indices forced to be non-zero.
        Sb: ArrayLike
            Set of unfixed indices.
        """
        dv = -datafit.conjugate(-u)
        for i in np.flatnonzero(S1):
            v[i] = np.dot(A[:, i], u)
            dv -= penalty.conjugate_scalar(i, v[i]) - lmbd
        for i in np.flatnonzero(Sb):
            v[i] = np.dot(A[:, i], u)
            dv -= regfunc.conjugate_scalar(i, lmbd, v[i])
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
            g = regfunc.conjugate_scalar(i, lmbd, v[i])
            p = penalty.conjugate_scalar(i, v[i]) - lmbd
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
        th: ArrayLike,
        lipschitz: float,
        A_colnorm: ArrayLike,
    ) -> None:
        flag = False
        r = np.sqrt(2.0 * np.abs(pv - dv) * lipschitz)
        for i in np.flatnonzero(Sbi):
            vi = v[i]
            if np.abs(vi) + r * A_colnorm[i] < th[i]:
                if x[i] != 0.0:
                    w -= x[i] * A[:, i]
                    x[i] = 0.0
                    flag = True
                Ws[i] = False
                Sbi[i] = False
                Sb0[i] = True
        if flag:
            u[:] = -datafit.gradient(w)
