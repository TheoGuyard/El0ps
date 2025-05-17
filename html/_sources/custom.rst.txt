.. _custom:

================
Custom instances
================

In addition to built-in instances of datafit, penalty, and estimators natively implemented in ``el0ps``, the package provides a flexible workflow allowing for **users-defined** instances of these objects to better fit application needs.
This process is articulated around the following template classes:

- :class:`el0ps.datafit.BaseDatafit` to define datafit functions,
- :class:`el0ps.penalty.BasePenalty` to define penalty functions,
- :class:`el0ps.estimator.L0Estimator` to define L0-problem-based estimators based.

When properly implemented, derived classes are fully compatible with all the utilities provided by ``el0ps`` without further requirements.
This section explains how to proceed.

.. note::

    Examples of user-defined datafit, penalty and estimators instantiation are given in the :ref:`examples` section.

.. _custom-maths:

Mathematical background
-----------------------

Defining custom datafit, penalty and estimators requires some mathematical background on convex analysis.
In particular, given a function :math:`\omega: \mathbb{R}^d \rightarrow \mathbb{R} \cup \{+\infty\}`, the notions of properness, lower-semicontinuity, convexity, coercivity, Lipschitz-continuity, gradient :math:`\nabla\omega`, subdifferential :math:`\partial\omega`, convex conjugate :math:`\omega^{\ast}`, and proximal operator :math:`\mathrm{prox}_{\omega}` will be used in the rest of this section.
We refer users to the following monograph for a comprehensive introduction and definition of these notions.

    Beck, A. (2017). First-order methods in optimization. SIAM.


.. _custom-datafit:

Custom datafit functions
------------------------

.. currentmodule:: el0ps.datafit

In ``el0ps``, datafit objects represent mathematical functions defined as

.. math::
    
    f: \mathbb{R}^m \rightarrow \mathbb{R} \cup \{+\infty\}

which are required to be proper, lower-semicontinuous, convex and differentiable with a Lipschitz-continuous gradient.
Any datafit function not natively provided by ``el0ps`` but fulfilling these specifications can be implemented to benefit from all the package features.
This is done by deriving from the :class:`BaseDatafit` class, which requires implementing the following methods:

- ``value(self, w: NDArray) -> float`` : value of :math:`f(\mathbf{w})`,
- ``conjugate(self, w: NDArray) -> float`` : value of :math:`f^{\ast}(\mathbf{w})`,
- ``gradient(self, w: NDArray) -> NDArray`` : value of :math:`\nabla f(\mathbf{w})`,
- ``gradient_lipschitz_constant(self) -> float`` : Lipschitz constant of :math:`\nabla f`.

Once done, users can fully enjoy all features provided by ``el0ps`` with their custom datafit function.

.. currentmodule:: el0ps.solver

.. note:: 

    Datafit functions defined as explained above can be handled by the :class:`BnbSolver` and :class:`OaSolver` solvers.
    To be compatible with the :class:`MipSolver` solver, they must implement one additional method, as explained in the :ref:`custom-mip` section.

.. _custom-penalty:

Custom penalty functions
------------------------

.. currentmodule:: el0ps.penalty

In ``el0ps``, penalty objects represent mathematical functions defined as

.. math::
    
    h: \mathbb{R}^n \rightarrow \mathbb{R} \cup \{+\infty\}

which are separable as :math:`h(\mathbf{x}) = \sum_{i=1}^n h_i(x_i)` for all :math:`\mathbf{x} \in \mathbb{R^n}`, where each splitting term :math:`h_i : \mathbb{R} \rightarrow \mathbb{R} \cup \{+\infty\}` is required to be proper, lower-semicontinuous, convex, coercive, and such that :math:`h_i(x_i) \geq h_i(0) = 0`.
Any penalty function not natively provided by ``el0ps`` but fulfilling these specifications can be implemented to benefit from all the package features.
This is done by deriving from the :class:`BasePenalty` class, which requires implementing the following methods:

- ``value(self, i: int, x: float) -> float`` : value of :math:`h_i(x)`,
- ``conjugate(self, i: int, x: float) -> float`` : value of :math:`h_i^{\ast}(x)`,
- ``prox(self, i: int, x: float, eta: float) -> float`` : value of :math:`\mathrm{prox}_{\eta h_i}(x)` with :math:`\eta > 0`,
- ``subdiff(self, i: int, x: float) -> NDArray`` : bounds of the interval :math:`\partial h_i(x)`,
- ``conjugate_subdiff(self, i: int, x: float) -> NDArray`` : bounds of the interval :math:`\partial h_i^{\ast}(x)`.

Once done, users can fully enjoy all features provided by ``el0ps`` with their custom penalty function.

**Improving numerical efficiency:** Some features of ``el0ps`` involve the quantities

.. math::

    \begin{align*}
        \tau_i^+(\lambda) &= \sup\{x \in \mathbb{R} : h_i^{\ast}(x) \leq \lambda\} \\
        \tau_i^-(\lambda) &= \inf\{x \in \mathbb{R} : h_i^{\ast}(x) \leq \lambda\} \\
    \end{align*}

associated with each splitting term :math:`h_i` and depending on the L0-norm weight parameter :math:`\lambda > 0`.
By default, these values are automatically approximated in classes deriving from :class:`BasePenalty` using the iterative procedure implemented in the :meth:`compute_param_slope_pos` and :meth:`compute_param_slope_neg` methods.
To improve performance, users can provide a closed form expression of these values by overloading the following methods of the :class:`BasePenalty` class:

- ``param_slope_pos(self, i: int, lmbd: float) -> float`` : value of :math:`\tau_i^+(\lambda)`,
- ``param_slope_neg(self, i: int, lmbd: float) -> float`` : value of :math:`\tau_i^-(\lambda)`.

Doing so avoid the use of the iterative procedure every time these values are required.
To ease this process in case of **even** penalty functions for which :math:`\tau_i^+(\lambda) = -\tau_i^-(\lambda)`, classes deriving from :class:`BasePenalty` can also inherit from the :class:`SymmetricPenalty` class.
In this case, only the following method needs to be implemented:

- ``param_slope(self, i: int, lmbd: float) -> float`` : value of :math:`\tau_i^+(\lambda) = -\tau_i^-(\lambda)`

and the functions ``param_slope_pos`` and ``param_slope_neg`` are automatically linked to its output.


.. currentmodule:: el0ps.solver

.. note:: 

    Penalty functions defined as explained above can be handled by the :class:`BnbSolver` and :class:`OaSolver` solvers.
    To be compatible with the :class:`MipSolver` solver, they must implement one additional method, as explained in the :ref:`custom-mip` section.

.. _custom-estimator:

Custom estimators
-----------------

.. currentmodule:: el0ps.estimator

In ``el0ps``, estimator objects represent solutions of L0-regularized problems expressed as

.. math::
    
    \textstyle\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{Ax}) + \lambda\|\mathbf{x}\|_0 + h(\mathbf{x})

where :math:`f` and :math:`h` are required to verify the assumptions of datafit and penalty functions as explained in the :ref:`custom-datafit` and :ref:`custom-penalty` sections.
Any estimator not natively provided by ``el0ps`` but fulfilling these specifications can be implemented in the package to benefit from all its features.
This is done by deriving from the :class:`L0Estimator` class, which only requires implementing a constructor  ``__init__(self, lmbd: float, *args, **kwargs) -> None`` for the class which specifies the value of :math:`\lambda` as well as potential parameters for the functions :math:`f` and :math:`h`.
This constructor **must** end with the call to the super constructor of :class:`L0Estimator` as

- ``super().__init__(datafit, penalty, lmbd)``

for some ``datafit`` object derived from :class:`el0ps.datafit.BaseDatafit`, some ``penalty`` object derived from :class:`el0ps.penalty.BasePenalty`, and some float ``lmbd``.
Once done, users can fully enjoy all features provided by ``el0ps`` with their custom estimator based on L0-norm problem.



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

MIP solver
----------

.. currentmodule:: el0ps.solver

Implementing custom datafit and penalty functions as explained in the previous sections allows benefiting from all the features provided by ``el0ps``, provided that the resulting L0-regularized problems are tackled via either the :class:`BnbSolver` or the :class:`OaSolver` solvers.
Users may also want to leverage the :class:`MipSolver` solver to address these problems.
This solver calls a generic mixed-integer programming optimizer on a model formulated as

.. math::
    
    \left\{
        \begin{array}{ll}
            \min & f_{\mathrm{val}} + \lambda \mathbf{1}^{\top}\mathbf{z} + h_{\mathrm{val}} \\
            \text{s.t.} & \mathbf{w} = \mathbf{Ax} \\
            & f_{\mathrm{val}} \geq f(\mathbf{w}) \\
            & h_{\mathrm{val}} \geq 
            \begin{cases}
                h(\mathbf{x}) & \text{if } z_i = 0 \implies x_i = 0 \quad \forall i \in \{1,\dots,n\} \\
                +\infty & \text{otherwise}
            \end{cases} \\
            & f_{\mathrm{val}} \in \mathbb{R} \cup \{+\infty\}, \ g_{\mathrm{val}} \in \mathbb{R} \cup \{+\infty\} \\
            & \mathbf{x} \in \mathbb{R}^n, \ \mathbf{w} \in \mathbb{R}^m, \ \mathbf{z} \in \{0,1\}^n \\
        \end{array}
    \right.

where the scalar variables :math:`f_{\mathrm{val}}` and :math:`h_{\mathrm{val}}` model the epigraph of the functions :math:`f` and :math:`h`, respectively, and where a binary variable :math:`\mathbf{z} \in \{0,1\}^n` encodes the nullity of the entries in :math:`\mathbf{x} \in \mathbb{R}^n` to linearize the L0-norm expression.

.. currentmodule:: el0ps.datafit

**Using custom datafit:** To use the mixed-integer programming solver with custom a datafit function, it must derive from the :class:`MipDatafit` class.
This requires implementing the function

- ``bind_model(self, model: pyomo.kernel.block) -> None``

which has the purpose of adding the constraint 

.. math::
    
    f_{\mathrm{val}} \geq f(\mathbf{w})

to the ``model`` object.
The latter is a `pyomo kernel block <https://pyomo.readthedocs.io/en/stable/api/pyomo.core.kernel.block.block.html>`_ instance already containing the attributes ``model.f``, ``model.w`` and ``model.M`` of the mixed-integer programming model that can be used to model this constraint.


.. currentmodule:: el0ps.penalty

**Using custom penalties:** To use the mixed-integer programming solver with custom a penalty function, it must derive from the :class:`MipPenalty` class.
This requires implementing the function

- ``bind_model(self, model: pyomo.kernel.block) -> None``

which has the purpose of adding the constraint 

.. math::
    
    h_{\mathrm{val}} \geq 
    \begin{cases}
        h(\mathbf{x}) & \text{if } z_i = 0 \implies x_i = 0 \quad \forall i \in \{1,\dots,n\} \\
        +\infty & \text{otherwise}
    \end{cases}

to the ``model`` object.
The latter is a `pyomo kernel block <https://pyomo.readthedocs.io/en/stable/api/pyomo.core.kernel.block.block.html>`_ instance already containing the attributes ``model.h``, ``model.x``, ``model.z`` and ``model.N`` of the mixed-integer programming model that can be used to model this constraint.

