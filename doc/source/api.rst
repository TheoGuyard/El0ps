.. _api_references:

==============
API references
==============

Estimators
==========


.. currentmodule:: el0ps.estimator

.. autosummary::
    :toctree: generated/

    L0Estimator
    L0Classifier
    L0L1Classifier
    L0L2Classifier
    L0L1L2Classifier
    L0Regressor
    L0L1Regressor
    L0L2Regressor
    L0L1L2Regressor
    L0SVC
    L0L1SVC
    L0L2SVC
    L0L1L2SVC


Datafits
========

.. currentmodule:: el0ps.datafit

.. autosummary::
    :toctree: generated/

    BaseDatafit
    MipDatafit
    KullbackLeibler
    Leastsquares
    Logcosh
    Logistic
    Squaredhinge


Penalties
=========

.. currentmodule:: el0ps.penalty

.. autosummary::
    :toctree: generated/

    BasePenalty
    SymmetricPenalty
    MipPenalty
    compute_param_slope_scalar
    compute_param_limit_scalar
    Bigm
    BigmL1norm
    BigmL2norm
    L1norm
    L2norm
    L1L2norm


Solvers
=======

.. currentmodule:: el0ps.solver

.. autosummary::
    :toctree: generated/

    Status
    Result
    BaseSolver
    BnbSolver
    MipSolver
    OaSolver

Path
====

.. currentmodule:: el0ps.path

.. autosummary::
    :toctree: generated/

    Path
    PathOptions


Utils
=====

.. currentmodule:: el0ps.utils

.. autosummary::
    :toctree: generated/

    compute_lmbd_max
