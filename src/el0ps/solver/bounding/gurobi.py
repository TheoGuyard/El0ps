import numpy as np
from typing import Union
from numpy.typing import NDArray
from el0ps.problem import Problem
from el0ps.solver.gurobi import GurobiSolver
from .base import BnbBoundingSolver


class GurobiBoundingSolver(BnbBoundingSolver, GurobiSolver):
    _default_options = {
        "OutputFlag": 0.0,
        "MIPGap": 1e-4,
        "MIPGapAbs": 1e-8,
        "IntFeasTol": 1e-8,
    }

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

        if S0_init is not None:
            for i in S0_init:
                self.model.addConstr(self.x_var[i] == 0.0)
                self.model.addConstr(self.z_var[i] == 0.0)
        if S1_init is not None:
            for i in S1_init:
                self.model.addConstr(self.z_var[i] == 1.0)
        if x_init is not None:
            for i in range(problem.A.shape[1]):
                self.x_var[i].Start = x_init[i]

        for k, v in self.options.items():
            self.model.setParam(k, v)

    def bound(self, problem, node, bnb, bounding_type):
        cstr_S0 = self.model.addConstr(self.z_var[node.S0] == 0.0)
        cstr_S1 = self.model.addConstr(self.z_var[node.S1] == 1.0)
        if bounding_type == "upper":
            cstr_Sb = self.model.addConstr(self.z_var[node.Sb] == 0.0)
        self.model.update()

        self.model.optimize()

        if bounding_type == "lower":
            node.x = np.array(self.x_var.X)
            node.w = problem.A @ node.x
            node.u = -problem.datafit.gradient(node.w)
            node.lower_bound = float(self.model.ObjVal)  # TODO: use dual bound
        elif bounding_type == "upper":
            node.x_inc = np.array(self.x_var.X)
            node.upper_bound = self.model.ObjVal
        else:
            raise ValueError("Unknown bounding `{}`".format(bounding_type))

        self.model.remove(cstr_S0)
        self.model.remove(cstr_S1)
        if bounding_type == "upper":
            self.model.remove(cstr_Sb)
        self.model.update()
