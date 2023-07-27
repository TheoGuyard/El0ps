import numpy as np
from typing import Union
from numpy.typing import NDArray
from el0ps.problem import Problem
from el0ps.solver.gurobi import GurobiSolver
from .base import BnbBoundingSolver


class GurobiBoundingSolver(BnbBoundingSolver, GurobiSolver):
    _default_options = {"OutputFlag": 0.0}

    def __init__(self, options: dict = _default_options):
        self.options = options

    def setup(
        self,
        problem: Problem,
        x_init: Union[NDArray, None] = None,
        S0_init: Union[NDArray, None] = None,
        S1_init: Union[NDArray, None] = None,
    ) -> None:
        self.build_model(problem, relax=True)
        self.set_init(x_init, S0_init, S1_init)
        self.set_options()

    def bound(self, problem, node, _, bounding_type):
        # Add branching constraints
        cstr_S0 = self.model.addConstr(self.z_var[node.S0] == 0.0)
        cstr_S1 = self.model.addConstr(self.z_var[node.S1] == 1.0)
        if bounding_type == "upper":
            cstr_Sb = self.model.addConstr(self.z_var[node.Sb] == 0.0)
        self.model.update()

        # Solver the bounding problem
        self.model.optimize()

        if bounding_type == "lower":
            node.x = np.array(self.x_var.X)
            node.w = problem.A @ node.x
            node.u = -problem.datafit.gradient(node.w)
            node.lower_bound = self.model.ObjVal  # TODO: use robust dual bound
        elif bounding_type == "upper":
            node.x_inc = np.array(self.x_var.X)
            node.upper_bound = self.model.ObjVal
        else:
            raise ValueError("Unknown bounding `{}`".format(bounding_type))

        # Remove branching constraints
        self.model.remove(cstr_S0)
        self.model.remove(cstr_S1)
        if bounding_type == "upper":
            self.model.remove(cstr_Sb)
        self.model.update()
