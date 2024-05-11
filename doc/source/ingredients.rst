.. _ingredients:

===========
Ingredients
===========

``el0ps`` addresses optimization problems expressed as

.. math::

   \tag{$\mathcal{P}$}\textstyle\min_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{Ax}) + \lambda\|\mathbf{x}\|_0 + h(\mathbf{x})

where :math:`f(\cdot)` is a datafit function, :math:`h(\cdot)` is a penalty function, :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` is a matrix and :math:`\lambda>0` is an hyperparameter.
To construct and solve instances of this problem, ``el0ps`` is build on three main classes:

- :class:`.datafits.BaseDatafit` defining datafit functions
- :class:`.penalties.BasePenalty` defining penalty functions
- :class:`.solvers.BaseSolver` defining problem solvers

The matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}` and the hyperparameter :math:`\lambda > 0` do not have dedicated classes and can be any `numpy <https://numpy.org>`_-compatible arrays and float, respectively.



Datafit
-------

Datafit functions :math:`f: \mathbb{R}^{m} \mapsto \mathbb{R} \cup \{+\infty\}` considered by ``el0ps`` are proper, lower-semicontinuous and convex.
An instance of a datafit function inherits from :class:`.datafits.BaseDatafit` and implements the following methods:

- ``self.value(w)`` returning the value of the datafit at ``w``
- ``self.conjugate(w)`` returning the value of the datafit conjugate [1]_ at ``w``

A datafit function can also derive from :class:`.datafits.SmoothDatafit`, meaning that it is differentiable with a Lipschitz-continuous gradient.
If so, it also implements the following methods:

- ``self.gradient(w)`` returning the datadit gradient at ``w``
- ``self.lipschitz_constant()`` returning the Lipschitz constant of the datafit gradient

Finally, a datafit function can also derive from :class:`.datafits.MipDatafit`, meaning that it can be modeled into a `pyomo <https://pyomo.readthedocs.io>`_ mixed-integer program.
If so, it also implements the following method:

- ``self.bind_model(model)`` which adds the constraint ``model.f >= self.value(model.w)`` to the ``model``, which already contains the variables ``model.w`` and ``model.f``.

``el0ps`` provides already-made datafit functions.
They are listed in the :ref:`api_references` page.
You can also create your own datafit functions as explained in the :ref:`custom` page.

Penalty
-------

Penalty functions :math:`h: \mathbb{R}^{n} \mapsto \mathbb{R} \cup \{+\infty\}` considered by ``el0ps`` are proper, lower-semicontinuous, convex, even and verify :math:`h(\mathbf{x}) \geq h(\mathbf{0}) = 0`.
An instance of a penalty function inherits from :class:`.datafits.BasePenalty` and implements the following methods:

- ``self.value(x)`` returning the value of the penalty at ``x``
- ``self.conjugate(x)`` returning the value of the penalty conjugate [1]_ at ``x``
- ``self.prox(x)`` returning the proximal operator of the penalty [2]_ at ``x``
- ``self.subdiff(x)`` returning the subdifferential of the penalty [3]_ at ``x``
- ``self.conjugate_subdiff(x)`` returning the the subdifferential [3]_ of the penalty conjugate [1]_ at ``x``

The penalties are also separable, meaning that they can be written as

.. math:: \textstyle h(\mathbf{x}) = \sum_{i=1}^{n} h_i(x_i)

where :math:`h_i: \mathbb{R} \mapsto \mathbb{R} \cup \{+\infty\}` for all :math:`i \in \{1, \ldots, n\}`.
The functions listed above are also defined in a coordinate-wise manner, see :class:`.penalties.BasePenalty`, for more details.
Finally, a penalty function can also derive from :class:`.penalties.MipPenalty`, meaning that it can be modeled into a `pyomo <https://pyomo.readthedocs.io>`_ mixed-integer program.
If so, it also implements the following method:

- ``self.bind_model(model, lmbd)`` which adds the constraint ``model.g >= lmbd * sum(model.z) + self.value(model.x)`` to the ``model``, which already contains the variables ``model.x``, ``model.z`` and ``model.f``. The ``bind_model`` function also ensures that ``model.x[i] == 0`` whenever ``model.z[i] == 0``.

``el0ps`` provides already-made penalty functions.
They are listed in the :ref:`api_references` page.
You can also create your own penalty functions as explained in the :ref:`custom` page.

Solver
------

A solver takes a datafit function, a penalty function, a numpy array ``A``, a float ``lmbd`` and solve the corresponding instance of problem :math:`(\mathcal{P})`.
An instance of a solver inherits from :class:`.solvers.BaseSolver` and implements the following method:

- ``self.solve(datafit, penalty, A, lmbd, x_init=None)`` solving the problem and returning a :class:`.solvers.Result` object.

``el0ps`` provides already-made solvers.
They are listed in the :ref:`api_references` page.
You can also create your own solver as explained in the :ref:`custom` page.

References
----------

.. [1] Chapiter 4 in "Beck, A. (2017). First-order methods in optimization. Society for Industrial and Applied Mathematics."
.. [2] Chapiter 6 in "Beck, A. (2017). First-order methods in optimization. Society for Industrial and Applied Mathematics."
.. [3] Chapiter 3 in "Beck, A. (2017). First-order methods in optimization. Society for Industrial and Applied Mathematics."