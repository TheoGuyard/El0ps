.. _api_references:

==============
API references
==============

Estimators
==========


.. currentmodule:: el0ps.estimators

.. autosummary::
    :toctree: generated/

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

.. currentmodule:: el0ps.datafits

.. autosummary::
    :toctree: generated/

    BaseDatafit
    SmoothDatafit
    MipDatafit
    Kullbackleibler
    Leastsquares
    Logcosh
    Logistic
    Squaredhinge


Penalties
=========


.. currentmodule:: el0ps.penalties

.. autosummary::
    :toctree: generated/

    BasePenalty
    MipPenalty
    Bigm
    BigmL1norm
    BigmL2norm
    BigmL1L2norm
    L1norm
    L2norm
    L1L2norm


Solvers
=======

.. currentmodule:: el0ps.solvers

.. autosummary::
    :toctree: generated/

    Status
    Result
    BaseSolver
    BnbSolver
    BnbOptions
    MipSolver
    MipOptions

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
    compute_param_slope_scalar
    compute_param_limit_scalar
    compute_param_maxdom_scalar
    compute_param_maxval_scalar
