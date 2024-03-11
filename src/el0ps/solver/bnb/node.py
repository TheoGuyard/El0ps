import numpy as np
from numpy.typing import ArrayLike


class BnbNode:
    """:class:`.solver.BnbSolver` tree node.

    Parameters
    ----------
    category: int
        Node category (root: -1, zero: 0, one: 1).
    S0: ArrayLike
        Set of indices forced to be zero.
    S1: ArrayLike
        Set of indices forced to be non-zero.
    Sb: ArrayLike
        Set of free indices.
    lower_bound: float
        BnbNode lower bound.
    upper_bound: float
        BnbNode upper bound.
    time_lower_bound: float
        Time to compute the lower bound.
    time_upper_bound: float
        Time to compute the upper bound.
    x: ArrayLike
        Relaxation solution.
    w: ArrayLike
        Value of `A @ self.x`.
    x_inc: ArrayLike
        Incumbent solution.
    """

    def __init__(
        self,
        category: int,
        S0: ArrayLike,
        S1: ArrayLike,
        Sb: ArrayLike,
        lower_bound: float,
        upper_bound: float,
        time_lower_bound: float,
        time_upper_bound: float,
        x: ArrayLike,
        w: ArrayLike,
        x_inc: ArrayLike,
    ) -> None:
        self.category = category
        self.S0 = S0
        self.S1 = S1
        self.Sb = Sb
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.time_lower_bound = time_lower_bound
        self.time_upper_bound = time_upper_bound
        self.x = x
        self.w = w
        self.x_inc = x_inc

    def __str__(self) -> str:
        s = ""
        s += "BnbNode\n"
        s += "  Category    : {}".format(self.category)
        s += "  S0/S1/Sb    : {}/{}/{}\n".format(
            np.sum(self.S0), np.sum(self.S1), np.sum(self.Sb)
        )
        s += "  Lower bound : {:.4f}\n".format(self.lower_bound)
        s += "  Upper bound : {:.4f}\n".format(self.upper_bound)
        return s

    def __copy__(self):
        return BnbNode(
            self.category,
            np.copy(self.S0),
            np.copy(self.S1),
            np.copy(self.Sb),
            self.lower_bound,
            self.upper_bound,
            self.time_lower_bound,
            self.time_upper_bound,
            np.copy(self.x),
            np.copy(self.w),
            np.copy(self.x_inc),
        )

    @property
    def rel_gap(self):
        """Relative gap between the lower and upper bounds."""
        return (self.upper_bound - self.lower_bound) / (
            np.abs(self.upper_bound) + 1e-10
        )

    @property
    def card_S0(self):
        return np.sum(self.S0)

    @property
    def card_S1(self):
        return np.sum(self.S1)

    @property
    def card_Sb(self):
        return np.sum(self.Sb)

    @property
    def depth(self):
        return self.card_S0 + self.card_S1

    def fix_to(self, idx: int, val: bool, A: ArrayLike):
        """Fix an extry of the node to zero or non-zero. Update the
        corresponding attributes of the node.

        Parameters
        ----------
        idx: int
            Index to fix.
        val: bool
            If ``False``, the entry is fixed to zero. If ``True``, the entry is
            fixed to non-zero.
        A: ArrayLike
            Matrix A.
        """
        self.Sb[idx] = False
        if val:
            self.category = 1
            self.S1[idx] = True
        else:
            self.category = 0
            self.S0[idx] = True
            if self.x[idx] != 0.0:
                self.w -= self.x[idx] * A[:, idx]
                self.x[idx] = 0.0

    def child(self, idx: int, val: bool, A: ArrayLike):
        child = BnbNode(
            int(val),
            np.copy(self.S0),
            np.copy(self.S1),
            np.copy(self.Sb),
            self.lower_bound,
            self.upper_bound,
            0.0,
            0.0,
            np.copy(self.x),
            np.copy(self.w),
            np.copy(self.x_inc),
        )
        child.fix_to(idx, val, A)
        return child
