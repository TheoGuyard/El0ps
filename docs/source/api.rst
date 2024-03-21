.. _api_references:

==============
API references
==============

Datafits
========

.. currentmodule:: el0ps.datafit

.. autosummary::
    :toctree: generated/

    BaseDatafit
    SmoothDatafit
    Leastsquares
    Logistic
    Squaredhinge


Penalties
=========


.. currentmodule:: el0ps.penalty

.. autosummary::
    :toctree: generated/

    BasePenalty
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

    BaseSolver
    Result
    Status
    BnbNode
    BnbBranchingStrategy
    BnbExplorationStrategy
    BnbOptions
    BnbSolver


Bounding
========

.. currentmodule:: el0ps.solver.bounding

.. autosummary::
    :toctree: generated/

    BnbBoundingSolver


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
    compute_param_slope
    compute_param_limit