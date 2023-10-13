.. _api_references:

==============
API references
==============

.. currentmodule:: el0ps

Estimators
==========

.. currentmodule:: el0ps

.. autosummary::
    :toctree: generated/

    Problem
    compute_lmbd_max
    Path
    PathOptions


Datafits
========

.. currentmodule:: el0ps.datafit

.. autosummary::
    :toctree: generated/

    BaseDatafit
    ProximableDatafit
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
    ProximablePenalty
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
    Results
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
    CdBoundingSolver
