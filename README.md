El0ps
=====
*-An Exact L0-Problem Solver-*

[![Documentation](https://img.shields.io/badge/documentation-latest-blue)](https://el0ps.github.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/release/python-380/)
[![Codecov](https://codecov.io/gh/TheoGuyard/El0ps/graph/badge.svg?token=H2IA4O67X6)](https://codecov.io/gh/TheoGuyard/El0ps)
[![License](https://img.shields.io/badge/License-AGPL--v3-red.svg)](https://github.com/benchopt/benchopt/blob/main/LICENSE)
<!-- [![PyPI version](https://badge.fury.io/py/el0ps.svg)](https://pypi.org/project/el0ps/) -->
<!-- [![Test Status](https://github.com/TheoGuyard/el0ps/actions/workflows/test.yml/badge.svg)](https://github.com/TheoGuyard/el0ps/actions/workflows/test.yml) -->

``el0ps`` a Python package providing **generic** and **efficient** solvers for **L0-norm** problems.
It also implements [scikit-learn](https://scikit-learn.org>) compatible estimators involving the L0-norm.
You can use some already-made problem instances or **customize your own** based on several templates and utilities.
Check out the [documentation](https://el0ps.github.io) for a starting tour of the package.

## Installation

`el0ps` will be available on [pypi](https://pypi.org>) soon. The latest version of the package can currently be installed as


```shell
$ pip install https://github.com/TheoGuyard/El0ps.git
```

Feel free to contribute by reporting any bug on the [issue](https://github.com/TheoGuyard/El0ps/issues) page or opening a [pull request](https://github.com/TheoGuyard/El0ps/pulls).

## Quick start

`el0ps` solves L0-norm optimization problems involving a datafit and a penalty function.

### Create and solve problem instances

An instance can be created and solved as simply as follows.

```python
from sklearn.datasets import make_regression
from el0ps.datafits import Leastsquares
from el0ps.penalties import L2norm
from el0ps.solvers import BnbSolver

# Target problem: min_x f(Ax) + lmbd * ||x||_0 + h(x)

# Define the problem data
A, y = make_regression()   # sparse regression data y ~ Ax
datafit = Leastsquares(y)  # function f(Ax) = (1/2) * ||y - Ax||_2^2
penalty = L2norm(alpha=1.) # function  h(x) = alpha * ||x||_2^2

# Solve the problem using a Branch-and-Bound solver
solver = BnbSolver()
result = solver.solver(datafit, penalty, A, lmbd=0.1)
```

You can use some already-made datafits and penalties or **customize your own** based on template classes.

### Fit regularization paths

`el0ps` provides utilities to solve a sequence of problems with varying values of $\lambda$ through regularization paths.

```python
from el0ps.path import Path

# Solve the problem over 100 values of lmbd from 1e2 to 1e-2
path = Path(lmbd_max=1e2, lmbd_min=1e-2, lmbd_num=100)
path.fit(solver, datafit, penalty, A)
```

### `scikit-learn` estimators

`el0ps` also provides [scikit-learn](https://scikit-learn.org>) compatible estimators to use in a pipeline.

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from el0ps.estimators import L0L2Regression

# Create training and testing data
A_train, A_test, y_train, y_test = train_test_split(A, y)

# Initialize a regerssion with L0L2-norm regularization (Leastsquares datafit and L2norm penalty)
estimator = L0L2Regression(lmbd=0.1, alpha=1.)

# Fit and score the estimator manually ...
estimator.fit(A_train, y_train)
estimator.score(A_test, y_test)

# ... or in a pipeline
pipeline = Pipeline([('estimator', estimator)])
pipeline.fit(A_train, y_train)
pipeline.score(A_test, y_test)
```

You can also build your own estimators upon used-defined datafits and penalties.


## Cite

`el0ps` is distributed under
[AGPL v3 license](https://github.com/TheoGuyard/El0ps/blob/main/LICENSE).
Please cite the package as follows:

```bibtex
@inproceedings{el0ps2024guyard,
    title        = {A New Branch-and-Bound Pruning Framework for L0-Regularized Problems},
    author       = {Guyard, Th{\'e}o and Herzet, C{\'e}dric and Elvira, Cl{\'e}ment and Ayse-Nur Arslan},
    booktitle    = {International Conference on Machine Learning (ICML)},
    year         = {2024},
    organization = {PMLR},
}
```
