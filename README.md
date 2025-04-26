El0ps: An Exact L0-Problem Solver
=====

[![Documentation](https://img.shields.io/badge/documentation-latest-blue)](https://theoguyard.github.io/El0ps/html/index.html)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/release/python-390/)
[![PyPI version](https://badge.fury.io/py/el0ps.svg)](https://pypi.org/project/el0ps/)
[![codecov](https://codecov.io/github/TheoGuyard/El0ps/graph/badge.svg?token=H2IA4O67X6)](https://codecov.io/github/TheoGuyard/El0ps)
[![Test Status](https://github.com/TheoGuyard/el0ps/actions/workflows/test.yml/badge.svg)](https://github.com/TheoGuyard/el0ps/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-AGPL--v3-red.svg)](https://github.com/TheoGuyard/El0ps/blob/main/LICENSE)

``el0ps`` is a Python package providing utilities to handle **L0-regularized** optimization problems expressed as

$$\textstyle\min_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{Ax}) + \lambda\|\|\mathbf{x}\|\|_0 + h(\mathbf{x})$$

appearing in several applications.
These problems aim at minimizing a trade off between a data-fidelity function $f$ composed with a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ and the L0-norm which counts the number of non-zeros in its argument to promote sparse solutions.
The additional penalty function $h$ can be used to enforce other desirable properties on the solutions and is involved in the construction of efficient solution methods.

The package includes
- A **flexible** framework with built-in problem instances and the possibility to define custom ones,
- A **state-of-the-art** solver based on a specialized Branch-and-Bound algorithm,
- A **[scikit-learn](https://scikit-learn.org>)** compatible interface providing linear model estimators based on L0-regularized optimization problems.

Check out the [documentation](https://theoguyard.github.io/El0ps/html/index.html) for a starting tour of the package.

## Installation

`el0ps` is available on [pypi](https://pypi.org/project/el0ps) and its latest version can be installed as follows:


```shell
pip install el0ps
```

## Quick start

``el0ps`` addresses L0-regularized optimization problems 


### Creating and solving problem instances

An instance of L0-regularized problem can be created and solved using few lines of code.
The following example illustrates how to use built-in utilities provided by `el0ps` to instantiate an solve a problem.

```python
from sklearn.datasets import make_regression
from el0ps.datafit import Leastsquares
from el0ps.penalty import L2norm
from el0ps.solver import BnbSolver

# Generate sparse regression data using sklearn
A, y = make_regression(n_samples=30, n_features=50, n_informative=5)

# Instantiate a least-squares loss f(w) = 0.5 * ||y - w||_2^2
datafit = Leastsquares(y)

# Instantiate an L2-norm penalty h(x) = beta * ||x||_2^2
penalty = L2norm(beta=0.1)

# Set the L0-regularization weight
lmbd = 10.0

# Solve the corresponding problem with el0ps' solver
solver = BnbSolver()
result = solver.solve(datafit, penalty, A, lmbd)

# Displays of result
>>> Result
>>>   Status     : optimal
>>>   Solve time : 0.045835 seconds
>>>   Iter count : 583
>>>   Objective  : 707.432177
>>>   Non-zeros  : 5
```

Various options can be passed to the `BnbSolver` class to tune its behavior. The problem solution can be recovered from the `result.x` attribute. Several other statistics on the solution process are also available in the `result` object.

### Fitting regularization paths

`el0ps` also provides a convenient pipeline to fit regularization paths, that is, solve an L0-regularized problem over a grid of parameter $\lambda$.

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
from el0ps.estimator import L0L2Regressor

# Generate sparse regression data
A, y = make_regression(n_informative=5, n_samples=100, n_features=200)

# Split training and testing sets
A_train, A_test, y_train, y_test = train_test_split(A, y)

# Initialize a regressor with L0L2-norm regularization
estimator = L0L2Regressor(lmbd=0.1, beta=1.)

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

`el0ps` is still in its early stages of development.
Feel free to contribute by report any bug on the [issue](https://github.com/TheoGuyard/El0ps/issues) page or by opening a [pull request](https://github.com/TheoGuyard/El0ps/pulls).
Any feedback or contribution is welcome.
Check out the [Contribution](https://theoguyard.github.io/El0ps/html/contribute.html) page for more information.

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
