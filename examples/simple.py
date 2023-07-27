import numpy as np
from el0ps import Problem, compute_lmbd_max
from el0ps.datafit import Quadratic
from el0ps.penalty import Bigm
from el0ps.solver import BnbSolver, GurobiSolver

# Syntehtic sparse regression data
k, m, n = 5, 50, 100
x = np.zeros(n)
s = np.array(np.floor(np.linspace(0, n - 1, num=k)), dtype=int)
x[s] = np.random.randn(k)
A = np.random.randn(n, m).T
y = A @ x
y += np.random.randn(m) * 0.1 * (np.linalg.norm(y)**2 / m)
M = 1.5 * np.max(np.abs(x))

datafit = Quadratic(y)
penalty = Bigm(M)
lmbd = 0.1 * compute_lmbd_max(datafit, penalty, A)
problem = Problem(datafit, penalty, A, lmbd)
print(problem)

solver = BnbSolver()
result = solver.solve(problem)
print(result)

solver = GurobiSolver()
result = solver.solve(problem)
print(result)
