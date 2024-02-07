import numpy as np
from el0ps.datafit import Leastsquares
from el0ps.penalty import Bigm
from el0ps.solver import BnbSolver
from el0ps.utils import compute_lmbd_max

k, m, n = 5, 50, 100
x = np.zeros(n)
s = np.array(np.floor(np.linspace(0, n - 1, num=k)), dtype=int)
x[s] = np.random.randn(k)
A = np.random.randn(m, n)
y = A @ x
y += np.random.randn(m) * 0.1 * (np.linalg.norm(y)**2 / m)
M = 1.5 * np.max(np.abs(x))

datafit = Leastsquares(y)
penalty = Bigm(M)
lmbd = 0.1 * compute_lmbd_max(datafit, penalty, A)

solver = BnbSolver()
result = solver.solve(datafit, penalty, A, lmbd)
print(result)
