El0ps
=====
*-An Exact L0-Problem Solver-*

[![Documentation](https://img.shields.io/badge/documentation-latest-blue)](https://theoguyard.github.io/El0ps/html/index.html)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/release/python-390/)
[![codecov](https://codecov.io/github/TheoGuyard/El0ps/graph/badge.svg?token=H2IA4O67X6)](https://codecov.io/github/TheoGuyard/El0ps)
[![Test Status](https://github.com/TheoGuyard/el0ps/actions/workflows/test.yml/badge.svg)](https://github.com/TheoGuyard/el0ps/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-AGPL--v3-red.svg)](https://github.com/TheoGuyard/El0ps/blob/main/LICENSE)
<!-- [![PyPI version](https://badge.fury.io/py/el0ps.svg)](https://pypi.org/project/el0ps/) -->

``el0ps`` is a Python package providing **generic** and **efficient** solvers and utilities to handle **L0-norm** problems.
It also implements [scikit-learn](https://scikit-learn.org>) compatible estimators involving these problems.
You can use some already-made problem instances or **customize your own** based on several templates and utilities.

Check out the [documentation](https://theoguyard.github.io/El0ps/html/index.html) for a starting tour of the package.

## Installation

`el0ps` will be available on [pypi](https://pypi.org>) soon. The latest version of the package can currently be installed as


```shell
pip install git+https://github.com/TheoGuyard/El0ps.git
```

## Quick start

``el0ps`` addresses optimization problems expressed as

$$\tag{$\mathcal{P}$}\textstyle\min_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{Ax}) + \lambda\|\|\mathbf{x}\|\|_0 + h(\mathbf{x})$$

where $f(\cdot)$ is a **datafit** function, $h(\cdot)$ is a **penalty** function, $\mathbf{A} \in \mathbb{R}^{m \times n}$ is a matrix and $\lambda>0$ is an hyperparameter.
The package provides efficient solvers for this family of problems, methods to fit regularization paths, bindings for [scikit-learn](https://scikit-learn.org>) estimators and other utilities.

### Create and solve problem instances

An instance of problem $(\mathcal{P})$ can be created and solved as simply as follows.

```python
import numpy as np
from el0ps.datafits import Leastsquares
from el0ps.penalties import L2norm
from el0ps.solvers import BnbSolver

# Generate sparse regression data
np.random.seed(0)
x = np.zeros(100)
s = np.random.randint(100, size=5)
x[s] = 1.
A = np.random.randn(50, 100)
A /= np.linalg.norm(A, ord=2)
y = A @ x
e = np.random.randn(50)
e *= np.sqrt((y @ y) / (10. * (e @ e)))
y += e

# Instantiate the function f(Ax) = (1/2) * ||y - Ax||_2^2
datafit = Leastsquares(y)

# Instantiate the function h(x) = beta * ||x||_2^2
penalty = L2norm(beta=0.1)

# Solve the problem with el0ps' Branch-and-Bound solver
solver = BnbSolver()
result = solver.solve(datafit, penalty, A, lmbd=0.01)
```

You can pass various options to the solver and once the problem is solved, you can recover different quantities such as the solver status, the solution or the optimal value of the problem from the ``result`` variable.


### Fitting regularization paths

You can also fit a regularization path where problem $(\mathcal{P})$ is solved over a grid of $\lambda$.
Fitting a path with `lmbd_num` different values of this parameter logarithmically spaced from some `lmbd_max` to some `lmbd_min` can be simply done as follows.


```python
from el0ps.path import Path

path = Path(lmbd_max=1e-0, lmbd_min=1e-2, lmbd_num=20)
data = path.fit(solver, datafit, penalty, A)
```

Once the path is fitted, you can recover different statistics `data` variable such as the number of non-zeros in the solution, the datafit value or the solution time.
Various other options can be passed to the path object.
An option of interest is the `lmbd_scaled` which is `False` by default.
When setting `lmbd_scaled=True`, the values of the parameter $\lambda$ are scaled so that the first solution constructed in the path when `lmbd=lmbd_max` correponds to the all-zero vector. 


### Scikit-Learn estimators

`el0ps` also provides [scikit-learn](https://scikit-learn.org>) compatible estimators based on problem $(\mathcal{P})$.
They can be used similarly to any other estimator in the package pipeline as follows.

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from el0ps.estimators import L0Regressor

# Generate sparse regression data
A, y = make_regression(n_informative=5, n_samples=100, n_features=200)

# Split training and testing sets
A_train, A_test, y_train, y_test = train_test_split(A, y)

# Initialize a regerssor with L0-norm regularization with Big-M constraint
estimator = L0Regressor(lmbd=0.1, M=1.)

# Fit and score the estimator manually ...
estimator.fit(A_train, y_train)
estimator.score(A_test, y_test)

# ... or in a pipeline
pipeline = Pipeline([('estimator', estimator)])
pipeline.fit(A_train, y_train)
pipeline.score(A_test, y_test)
```

Like datafit and penalty functions, you can build your own estimators.

## Contribute

`el0ps` is still under development.
As an open-source project, we kindly welcome any contributions.
Feel free to report any bug on the [issue](https://github.com/TheoGuyard/El0ps/issues) page or to open a [pull request](https://github.com/TheoGuyard/El0ps/pulls).

## Cite

`el0ps` is distributed under
[AGPL v3 license](https://github.com/TheoGuyard/El0ps/blob/main/LICENSE).
Please cite the package as follows:

```bibtex
@inproceedings{guyard2024el0ps,
    title        = {A New Branch-and-Bound Pruning Framework for L0-Regularized Problems},
    author       = {Guyard, Th{\'e}o and Herzet, C{\'e}dric and Elvira, Cl{\'e}ment and Ayse-Nur Arslan},
    booktitle    = {International Conference on Machine Learning (ICML)},
    year         = {2024},
    organization = {PMLR},
}
```
