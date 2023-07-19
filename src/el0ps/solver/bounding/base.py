from abc import ABCMeta, abstractmethod
from typing import Union
from el0ps.problem import Problem
from numpy.typing import NDArray


class BnbBoundingSolver(metaclass=ABCMeta):
    """Base class for for Branch-and-bound bounding solvers."""

    @abstractmethod
    def setup(
        self,
        problem: Problem,
        x_init: Union[NDArray, None] = None,
        S0_init: Union[NDArray, None] = None,
        S1_init: Union[NDArray, None] = None,
    ) -> None:
        """Initialize internal attributes of the bounding solver.

        Parameters
        ----------
        problem : Problem
            Problem to solve.
        x_init : Union[NDArray, None]
            Warm-start value of x.
        S0_init : Union[NDArray, None]
            Set of indices of x forced to be zero.
        S1_init : Union[NDArray, None]
            Set of indices of x forced to be non-zero.
        """
        ...

    @abstractmethod
    def bound(self, problem, node, bnb, bounding_type):
        """Solve the bounding problem at a given node of the Branch-and-Bound.

        Parameters
        ----------

        problem : Problem
            Problem data.
        bnb : BnbSolver
            Branch-and-Bound solver with its current state attributes.
        node : BnbNode
            Node to bound.
        bounding_type : BnbBoundingType
            The type of bounding the solver is responsible to perform.
        """
        ...
