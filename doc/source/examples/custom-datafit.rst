.. _examples-custom-datafit:

Custom datafit instantiation
----------------------------

The next example shows how to implement a custom datafit function in ``el0ps``, as explained in the :ref:`custom-datafit` section.

.. code-block:: python
    
    import numpy as np
    from np.typing import NDArray
    from el0ps.datafit import BaseDatafit

    class Huber(BaseDatafit):
        """
        Huber loss function defined as f(w) = sum_{j=1}^m fj(wj) where

            fj(wj) = 0.5 * |wj - yj|^2          if |wj - yj| <= d
            fj(wj) = d * (|wj - yj| - 0.5 * d)  otherwise
        
        for some y in R^m and d > 0.
        """

        def __init__(self, y: NDArray, d: float):
            self.y = y
            self.d = d

        def value(self, w: NDArray):
            z = np.abs(w - self.y)
            v = 0.
            for zj in z:
                if zj <= self.d:
                    v += 0.5 * zj ** 2
                else:
                    v += self.d * (zj - 0.5 * self.d)
            return v
        
        def conjugate(self, w: NDArray):
            if np.any(np.abs(w) > self.d):
                return np.inf
            else:
                return np.sum(w + self.y)

        def gradient(self, w: NDArray):
            z = w - self.y
            g = np.empty_like(w)
            for j, zj in enumerate(z):
                if np.abs(zj) <= self.d:
                    g[j] = zj
                else:
                    g[j] = self.d * np.sign(zj)
            return g
    
        def gradient_lipschitz_constant(self):
            return 2. * self.d

