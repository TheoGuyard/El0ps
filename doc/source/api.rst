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
    Bigm
    BigmL1norm
    BigmL2norm
    BigmPositiveL1norm
    BigmPositiveL2norm
    Bounds
    L1norm
    L2norm
    L1L2norm
    PositiveL1norm
    PositiveL2norm
    compute_param_slope_pos
    compute_param_slope_neg


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

Compilation
===========

.. currentmodule:: el0ps.compilation

.. autosummary::
    :toctree: generated/

    CompilableClass
    compiled_clone

Path
====

.. currentmodule:: el0ps.path

.. autosummary::
    :toctree: generated/

    Path


Utils
=====

.. currentmodule:: el0ps.utils

.. autosummary::
    :toctree: generated/

    compute_lmbd_max
