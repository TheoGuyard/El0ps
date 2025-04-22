.. _custom:

================
Custom instances
================

Besides built-in instances of datafit, penalty, and estimators provided by ``el0ps``, the package also provides template classes allowing users to define their own ingredients to better fit their application needs.
This process is articulated around the following template classes:

- :class:`el0ps.datafit.BaseDatafit` to define datafit functions,
- :class:`el0ps.penalty.BasePenalty` to define penalty functions,
- :class:`el0ps.estimator.L0Estimator` to define estimators based on L0-problem solutions.

When properly overloaded, derives classes are fully compatible with all the utilities provided by ``el0ps`` without further implementation requirements.
This section explains how to proceed.

.. note::

    Examples of custom datafit and penalty function instantiation are given in the :ref:`examples` section.

.. _custom-maths:

Mathematical background
-----------------------

Defining custom datafit and penalty functions requires some mathematical background on convex analysis theory.
In particular, given a function :math:`\omega: \mathbb{R}^d \rightarrow \mathbb{R} \cup \{+\infty\}`, the notions of properness, lower-semicontinuity, convexity, coercivity, gradient :math:`\nabla\omega`, Lipschitz-continuity, convex conjugate :math:`\omega^{\ast}`, proximal operator :math:`\mathrm{prox}_{\omega}`, and subdifferential :math:`\partial\omega` are used in the rest of this section.
We refer users to the following monograph for a comprehensive introduction to these notions.

    Beck, A. (2017). First-order methods in optimization. SIAM.


.. _custom-datafit:

Custom datafit functions
------------------------

.. currentmodule:: el0ps.datafit

In ``el0ps``, datafit objects represent mathematical functions defined as

.. math::
    
    f: \mathbb{R^m} \rightarrow \mathbb{R} \cup \{+\infty\}

which are required to be proper, lower-semicontinuous, and convex.
Currently, ``el0ps`` only support datafit functions that are differentiable with a Lipschitz-continuous gradient, but future versions may allow for non-differentiable datafit functions.
Any datafit function fulfilling these requirements can be implemented in the package to benefit from all its features.
This is done by deriving from the :class:`BaseDatafit` class, which requires implementing the following methods:

- ``value(self, w: NDArray) -> float`` : value of :math:`f(\mathbf{w})`,
- ``conjugate(self, w: NDArray) -> float`` : value of :math:`f^{\ast}(\mathbf{w})`,
- ``gradient(self, w: NDArray) -> NDArray`` : value of :math:`\nabla f(\mathbf{w})`,
- ``gradient_lipschitz_constant(self) -> float`` : Lipschitz constant of :math:`\nabla f`.

.. currentmodule:: el0ps.solver

.. note:: 

    Datafit functions defined as explained above can be handled by the :class:`BnbSolver` solver.
    To be compatible with the :class:`MipSolver` and :class:`OaSolver` solvers, they must implement an additional method, as explained in the :ref:`custom-mip` section.

.. _custom-penalty:

Custom penalty functions
------------------------

.. currentmodule:: el0ps.penalty

In ``el0ps``, penalty objects represent mathematical functions defined as

.. math::
    
    h: \mathbb{R^n} \rightarrow \mathbb{R} \cup \{+\infty\}

which are separable are :math:`h(\mathbf{x}) = \sum_{i=1}^n h_i(x_i)` for all :math:`\mathbf{x} \in \mathbb{R^n}`, where each splitting term :math:`h_i : \mathbb{R} \rightarrow \mathbb{R} \cup \{+\infty\}` is required to be proper, lower-semicontinuous, convex, coercive, non-negative, and minimized at :math:`x_i = 0`.
Any penalty function fulfilling these requirements can be implemented in the package to benefit from all its features.
This is done by deriving from the :class:`BasePenalty` class, which requires implementing the following methods:

- ``value(self, i: int, x: float) -> float`` : value of :math:`h_i(x)`,
- ``conjugate(self, i: int, x: float) -> float`` : value of :math:`h_i^{\ast}(x)`,
- ``prox(self, i: int, x: float, eta: float) -> float`` : value of :math:`\mathrm{prox}_{\eta h_i}(x)` with :math:`\eta > 0`,
- ``subdiff(self, i: int, x: float) -> NDArray`` : bounds of the interval :math:`\partial h_i(x)`,
- ``conjugate_subdiff(self, i: int, x: float) -> NDArray`` : bounds of the interval :math:`\partial h_i^{\ast}(x)`.

Some features of ``el0ps`` involve the the quantities

.. math::

    \begin{align*}
        \tau_i^+ &= \sup\{x \in \mathbb{R} : h_i^{\ast}(x) \leq \lambda\} \\
        \tau_i^- &= \inf\{x \in \mathbb{R} : h_i^{\ast}(x) \leq \lambda\} \\
    \end{align*}

defined for some :math:`\lambda > 0`.
By default, these values are automatically approximated in classes deriving from :class:`BasePenalty` using an iterative procedure in the :meth:`compute_param_slope_pos` and :meth:`compute_param_slope_neg` methods.
However, this can represent a substantial overhead in terms of computation time.
To improve performance, users can also provide a closed form expression of these values by overloading the following methods of the :class:`BasePenalty` class:

- ``param_slope_pos(self, i: int, lmbd: float) -> float`` : value of :math:`\tau_i^+`,
- ``param_slope_neg(self, i: int, lmbd: float) -> float`` : value of :math:`\tau_i^-`,

To ease this process for penalties that are even, classes deriving from :class:`BasePenalty` can also inherit from the :class:`SymmetricPenalty` class.
In this case, only the following methods need to be implemented:

- ``param_slope(self, i: int, lmbd: float) -> float`` : value of :math:`\tau_i^+`,

since the values :math:`\tau_i^-` can be deduced symmetrically.

.. currentmodule:: el0ps.solver

.. note:: 

    Penalty functions defined as explained above can be handled by the :class:`BnbSolver` solver.
    To be compatible with the :class:`MipSolver` and :class:`OaSolver` solvers, they must implement an additional method, as explained in the :ref:`custom-mip` section.

.. _custom-estimators:

Custom estimators
-----------------

.. todo:: TODO

.. _custom-compilation:

Compilable classes
------------------

.. currentmodule:: el0ps.compilation

Custom loss and penalty functions defined by the user can derive from the :class:`CompilableClass` class to take advantage of just-in-time compilation provided by `numba <https://numba.org>`_.
This requires that all operations performed by the class are compatible with `numba operations <https://numba.readthedocs.io/en/stable/user/5minguide.html#will-numba-work-for-my-code>`_.
Moreover, users need to instantiate two additional methods:

- ``get_spec(self) -> tuple`` : returns a tuple where each element corresponds to a tuple referring to one of the class attribute defined in the ``__init__`` function of the class and specifying its name as a string as well as its `numba type <https://numba.readthedocs.io/en/stable/reference/types.html>`_.

- ``params_to_dict(self) -> dict`` : returns a dictionary where each key-value association refers to one of the class attribute defined in the ``__init__`` function of the class and specifies its name as a string as well as its value.

This functionality inspired from a similar one implemented in `skglm <https://contrib.scikit-learn.org/skglm/stable/index.html>`_.

.. _custom-mip:

Mixed-integer programming solvers
---------------------------------

.. todo:: TODO
