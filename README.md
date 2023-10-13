El0ps
=====
*-An Exact L0-Problem Solver-*


[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/release/python-380/)
[![Documentation](https://img.shields.io/badge/documentation-latest-blue)](https://el0ps.github.io)
[![Test Status](https://github.com/TheoGuyard/el0ps/actions/workflows/test.yml/badge.svg)](https://github.com/TheoGuyard/el0ps/actions/workflows/test.yml)
[![Codecov](https://codecov.io/gh/TheoGuyard/El0ps/graph/badge.svg?token=H2IA4O67X6)](https://codecov.io/gh/TheoGuyard/El0ps)
[![PyPI version](https://badge.fury.io/py/el0ps.svg)](https://pypi.org/project/el0ps/)
[![License](https://img.shields.io/badge/License-AGPL--v3-red.svg)](https://github.com/benchopt/benchopt/blob/main/LICENSE)


`el0ps` is a Python package to solve L0-penalized optimization problems of the form

$$\textstyle\min_{\mathbf{x}} \ f(\mathbf{A}\mathbf{x}) + \lambda\|\|\mathbf{x}\|\|_0 + h(\mathbf{x})$$

where $f(\cdot)$ is a datafit function, $\mathbf{A}$ is a linear operator, $h(\cdot)$ is a penalty function and $\lambda>0$ is the L0-regularization weight.
`el0ps` is designed to be numerically efficient and supports a wide range of datafit and penalty functions.
You can pick from `el0ps`'s already-made datafits and penalties or customize your own by building on top of the available template classes.


## Installation

`el0ps` is available on [pipy](https://pypi.org). 
Get the latest version of the package by running the following command.

```shell
$ pip install el0ps
```

Please report any bug in the [issue page](https://github.com/TheoGuyard/El0ps/issues).
Feel free to contribute by opening a [pull request](https://github.com/TheoGuyard/El0ps/pulls).

## Getting started

Here is a simple example showing how to solve an instance of L0-penalized problem with a Least-squares datafit and a Big-M constraint penalty for a fixed $\lambda$.

```python
import numpy as np
from sklearn.datasets import make_regression
from el0ps import Problem
from el0ps.datafit import Leastsquares
from el0ps.penalty import Bigm
from el0ps.solver import BnbSolver

# Generate sparse regression data using scikit-learn
A, y, x = make_regression(n_samples=50, n_features=100, coef=True)
M = 10. * np.linalg.norm(x, np.inf)

# Instantiate the problem and solve it using el0ps's BnB solver
datafit, penalty = Leastsquares(y), Bigm(M)
problem = Problem(datafit, penalty, A, lmbd=10.)
solver = BnbSolver()
result = solver.solve(problem)
```

You can also fit a regularization path, i.e., solve the problem over a range of $\lambda$, as simply as follows.

```python
from el0ps import Path
path = Path()
data = path.fit(solver, datafit, penalty, A)
```

The documentation references `el0ps`'s already-made datafits and penalties other than the `Leastsquares` and `Bigm` ones.
It also explains how to create custom datafits and penalties by building on top of the available template classes.


## Cite

`el0ps` is distributed under
[AGPL v3 license](https://github.com/TheoGuyard/El0ps/blob/main/LICENSE).
Please cite the package as follows:

> Todo: Add citation
