Experiments
===========

This directory contains experiments published in papers linked to `el0ps`.

## Installation

To run experiments, you first need to have a working installation of [python](https://www.python.org) and [pip](https://pypi.org/project/pip/).
Then, you need to clone the `el0ps` repository and install the package with its extra dependencies as follows:

```shell
$ git clone https://github.com/TheoGuyard/El0ps.git
$ cd el0ps
$ pip install -e .
$ pip install -e '.[exp]'
$ pip install -e '.[mip]'
```

To check that everything is working properly, you can run the tests with the following commands:

```shell
$ pip install pytest
$ pytest -v
```

> **_NOTE:_** When running experiments requiring the mixed-integer programming solvers [CPLEX](https://www.ibm.com/analytics/cplex-optimizer), [Gurobi](https://www.gurobi.com) or [Mosek](https://www.mosek.com), you need to have installed it priorly and have a valid license.

## Running experiments

An experiment can be run from the root folder of `el0ps` using the command

```shell
$ python experiments/run.py <experiment_name> onerun <path_to_setup_file> --save
```

where `<experiment_name>` is the name of the experiment and ``<path_to_setup_file>`` is the path to the setup file.
The option `--save` allows saving the results of the experiment in the `results` folder.
Graphics can be generated from saved experiments matching a given configuration using the command

```shell
$ python experiments/run.py <experiment_name> graphic <path_to_setup_file>
```

and the option `--save` allows saving the results of the experiment in the `saves` folder.
Examples of configuration files can be found in the `experiments/icml` folder.

## Available experiments

Available experiments are:

* `perfprofile`: Solve a problem instance with different solvers and build performance profiles.
* `regpath`: Fit a regularization path with different solvers and compare the performance.
* `statistics`: Compute statistics on the solutions of a regularization path for L0-regularized problem against other sparse methods.
